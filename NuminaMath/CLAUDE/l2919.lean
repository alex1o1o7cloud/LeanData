import Mathlib

namespace NUMINAMATH_CALUDE_angle_measure_l2919_291924

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l2919_291924


namespace NUMINAMATH_CALUDE_emilys_friends_with_color_boxes_l2919_291963

def rainbow_colors : ℕ := 7
def total_pencils : ℕ := 56

theorem emilys_friends_with_color_boxes :
  ∀ (pencils_per_box : ℕ) (total_boxes : ℕ),
    pencils_per_box = rainbow_colors →
    total_pencils = pencils_per_box * total_boxes →
    total_boxes - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_emilys_friends_with_color_boxes_l2919_291963


namespace NUMINAMATH_CALUDE_triangle_count_proof_l2919_291972

/-- The number of triangles formed by 9 distinct lines in a plane -/
def num_triangles : ℕ := 23

/-- The total number of ways to choose 3 lines from 9 lines -/
def total_combinations : ℕ := Nat.choose 9 3

/-- The number of intersections where exactly three lines meet -/
def num_intersections : ℕ := 61

theorem triangle_count_proof :
  num_triangles = total_combinations - num_intersections :=
by sorry

end NUMINAMATH_CALUDE_triangle_count_proof_l2919_291972


namespace NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l2919_291933

def is_monic_cubic (q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, q x = x^3 + a*x^2 + b*x + c

theorem cubic_polynomial_uniqueness (q : ℝ → ℂ) :
  is_monic_cubic (λ x : ℝ ↦ (q x).re) →
  q (5 - 3*I) = 0 →
  q 0 = -80 →
  ∀ x, q x = x^3 - 10*x^2 + 40*x - 80 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l2919_291933


namespace NUMINAMATH_CALUDE_line_intersection_with_x_axis_l2919_291948

/-- A line parallel to y = -3x that passes through (0, -2) intersects the x-axis at (-2/3, 0) -/
theorem line_intersection_with_x_axis :
  ∀ (k b : ℝ),
  (∀ x y : ℝ, y = k * x + b ↔ y = -3 * x + b) →  -- Line is parallel to y = -3x
  -2 = k * 0 + b →                               -- Line passes through (0, -2)
  ∃ x : ℝ, x = -2/3 ∧ 0 = k * x + b :=           -- Intersection point with x-axis
by sorry

end NUMINAMATH_CALUDE_line_intersection_with_x_axis_l2919_291948


namespace NUMINAMATH_CALUDE_max_value_of_sin_cos_ratio_l2919_291998

theorem max_value_of_sin_cos_ratio (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_acute_γ : 0 < γ ∧ γ < π/2)
  (h_sum_sin_sq : Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1) :
  (Real.sin α + Real.sin β + Real.sin γ) / (Real.cos α + Real.cos β + Real.cos γ) ≤ Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_max_value_of_sin_cos_ratio_l2919_291998


namespace NUMINAMATH_CALUDE_smartphone_price_l2919_291952

/-- The original sticker price of the smartphone -/
def sticker_price : ℝ := 950

/-- The price at store X after discount and rebate -/
def price_x (p : ℝ) : ℝ := 0.8 * p - 120

/-- The price at store Y after discount -/
def price_y (p : ℝ) : ℝ := 0.7 * p

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem smartphone_price :
  price_x sticker_price + 25 = price_y sticker_price :=
by sorry

end NUMINAMATH_CALUDE_smartphone_price_l2919_291952


namespace NUMINAMATH_CALUDE_carpenters_completion_time_l2919_291954

/-- The time it takes for two carpenters to complete a job together -/
theorem carpenters_completion_time 
  (rate1 : ℚ) -- Work rate of the first carpenter
  (rate2 : ℚ) -- Work rate of the second carpenter
  (h1 : rate1 = 1 / 7) -- First carpenter's work rate
  (h2 : rate2 = 1 / (35/2)) -- Second carpenter's work rate
  : (1 : ℚ) / (rate1 + rate2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_carpenters_completion_time_l2919_291954


namespace NUMINAMATH_CALUDE_expected_value_specialized_coin_l2919_291990

/-- A specialized coin with given probabilities and payoffs -/
structure Coin where
  prob_heads : ℚ
  prob_tails : ℚ
  payoff_heads : ℚ
  payoff_tails : ℚ

/-- The expected value of a single flip of the coin -/
def expected_value (c : Coin) : ℚ :=
  c.prob_heads * c.payoff_heads + c.prob_tails * c.payoff_tails

/-- The expected value of two flips of the coin -/
def expected_value_two_flips (c : Coin) : ℚ :=
  2 * expected_value c

theorem expected_value_specialized_coin :
  let c : Coin := {
    prob_heads := 1/4,
    prob_tails := 3/4,
    payoff_heads := 4,
    payoff_tails := -3
  }
  expected_value_two_flips c = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_specialized_coin_l2919_291990


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l2919_291977

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  (∀ a : ℝ, a = 1 → |a| = 1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ |a| = 1) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l2919_291977


namespace NUMINAMATH_CALUDE_balls_in_box_perfect_square_l2919_291918

theorem balls_in_box_perfect_square (a v : ℕ) : 
  (2 * a * v : ℚ) / ((a + v) * (a + v - 1) / 2) = 1 / 2 → 
  ∃ n : ℕ, a + v = n^2 := by
sorry

end NUMINAMATH_CALUDE_balls_in_box_perfect_square_l2919_291918


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_34_l2919_291982

theorem smallest_four_digit_divisible_by_34 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 34 ∣ n → n ≥ 1020 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_34_l2919_291982


namespace NUMINAMATH_CALUDE_stating_probability_theorem_l2919_291950

/-- Represents the number of guests -/
def num_guests : ℕ := 3

/-- Represents the number of roll types -/
def num_roll_types : ℕ := 4

/-- Represents the total number of rolls -/
def total_rolls : ℕ := 12

/-- Represents the number of rolls per guest -/
def rolls_per_guest : ℕ := 4

/-- Represents the number of each type of roll -/
def rolls_per_type : ℕ := 3

/-- 
Calculates the probability that each guest receives one roll of each type 
when rolls are randomly distributed.
-/
def probability_all_different_rolls : ℚ := sorry

/-- 
Theorem stating that the probability of each guest receiving one roll of each type 
is equal to 2/165720
-/
theorem probability_theorem : 
  probability_all_different_rolls = 2 / 165720 := sorry

end NUMINAMATH_CALUDE_stating_probability_theorem_l2919_291950


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l2919_291987

theorem largest_divisor_of_difference_of_squares (m n : ℕ) : 
  Odd m → Odd n → n < m → m - n > 2 → 
  (∀ k : ℕ, k > 4 → ∃ x y : ℕ, Odd x ∧ Odd y ∧ y < x ∧ x - y > 2 ∧ ¬(k ∣ x^2 - y^2)) ∧ 
  (∀ x y : ℕ, Odd x → Odd y → y < x → x - y > 2 → (4 ∣ x^2 - y^2)) := by
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l2919_291987


namespace NUMINAMATH_CALUDE_grassy_plot_width_l2919_291959

/-- Given a rectangular grassy plot with a gravel path, calculate its width -/
theorem grassy_plot_width : ℝ :=
  let plot_length : ℝ := 110
  let path_width : ℝ := 2.5
  let gravelling_cost_per_sq_meter : ℝ := 0.80
  let total_gravelling_cost : ℝ := 680
  let plot_width : ℝ := 97.5
  
  have h1 : plot_length > 0 := by sorry
  have h2 : path_width > 0 := by sorry
  have h3 : gravelling_cost_per_sq_meter > 0 := by sorry
  have h4 : total_gravelling_cost > 0 := by sorry
  
  have path_area : ℝ := 
    (plot_length + 2 * path_width) * (plot_width + 2 * path_width) - 
    plot_length * plot_width
  
  have total_cost_equation : 
    gravelling_cost_per_sq_meter * path_area = total_gravelling_cost := by sorry
  
  plot_width

end NUMINAMATH_CALUDE_grassy_plot_width_l2919_291959


namespace NUMINAMATH_CALUDE_intersection_when_a_is_two_range_of_a_when_intersection_is_empty_l2919_291905

def U : Set ℝ := Set.univ

def A (a : ℝ) : Set ℝ := {x : ℝ | 2 * x + a > 0}

def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

theorem intersection_when_a_is_two :
  A 2 ∩ B = {x : ℝ | x > 3} := by sorry

theorem range_of_a_when_intersection_is_empty :
  ∀ a : ℝ, A a ∩ (U \ B) = ∅ ↔ a ≤ -6 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_two_range_of_a_when_intersection_is_empty_l2919_291905


namespace NUMINAMATH_CALUDE_problem_statement_l2919_291927

theorem problem_statement (x y z : ℝ) 
  (h1 : 1 / x = 2 / (y + z))
  (h2 : 1 / x = 3 / (z + x))
  (h3 : 1 / x = (x^2 - y - z) / (x + y + z))
  (h4 : x ≠ 0)
  (h5 : y + z ≠ 0)
  (h6 : z + x ≠ 0)
  (h7 : x + y + z ≠ 0) :
  (z - y) / x = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2919_291927


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2919_291902

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x - 2) * (x + 2) < 5} = {x : ℝ | -3 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2919_291902


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l2919_291907

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l2919_291907


namespace NUMINAMATH_CALUDE_tournament_games_count_l2919_291936

/-- A single-elimination tournament structure -/
structure Tournament :=
  (total_teams : ℕ)
  (bye_teams : ℕ)
  (h_bye : bye_teams ≤ total_teams)

/-- The number of games played in a single-elimination tournament -/
def games_played (t : Tournament) : ℕ :=
  t.total_teams - 1

theorem tournament_games_count (t : Tournament) 
  (h_total : t.total_teams = 32) 
  (h_bye : t.bye_teams = 8) : 
  games_played t = 32 := by
sorry

end NUMINAMATH_CALUDE_tournament_games_count_l2919_291936


namespace NUMINAMATH_CALUDE_train_length_l2919_291973

theorem train_length (platform1_length platform2_length : ℝ)
                     (time1 time2 : ℝ)
                     (h1 : platform1_length = 110)
                     (h2 : platform2_length = 250)
                     (h3 : time1 = 15)
                     (h4 : time2 = 20)
                     (h5 : time1 > 0)
                     (h6 : time2 > 0) :
  let train_length := (platform2_length * time1 - platform1_length * time2) / (time2 - time1)
  train_length = 310 := by
sorry

end NUMINAMATH_CALUDE_train_length_l2919_291973


namespace NUMINAMATH_CALUDE_sets_partition_l2919_291917

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define a property for primes greater than 2013
def IsPrimeGreaterThan2013 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n > 2013

-- Define the property for the special difference condition
def SpecialDifference (A B : Set ℕ) : Prop :=
  ∀ (x y : ℕ), x ∈ PositiveIntegers → y ∈ PositiveIntegers →
    IsPrimeGreaterThan2013 (x - y) →
    ((x ∈ A ∧ y ∈ B) ∨ (x ∈ B ∧ y ∈ A))

theorem sets_partition (A B : Set ℕ) :
  (A ∪ B = PositiveIntegers) →
  (A ∩ B = ∅) →
  SpecialDifference A B →
  ((∀ n : ℕ, n ∈ A ↔ n ∈ PositiveIntegers ∧ Even n) ∧
   (∀ n : ℕ, n ∈ B ↔ n ∈ PositiveIntegers ∧ Odd n)) :=
by sorry

end NUMINAMATH_CALUDE_sets_partition_l2919_291917


namespace NUMINAMATH_CALUDE_no_extrema_in_interval_l2919_291965

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the open interval (-1, 1)
def openInterval : Set ℝ := {x | -1 < x ∧ x < 1}

-- Theorem statement
theorem no_extrema_in_interval :
  ¬∃ (max_val min_val : ℝ), 
    (∀ x ∈ openInterval, f x ≤ max_val) ∧
    (∃ x_max ∈ openInterval, f x_max = max_val) ∧
    (∀ x ∈ openInterval, min_val ≤ f x) ∧
    (∃ x_min ∈ openInterval, f x_min = min_val) :=
sorry

end NUMINAMATH_CALUDE_no_extrema_in_interval_l2919_291965


namespace NUMINAMATH_CALUDE_sine_range_theorem_l2919_291986

theorem sine_range_theorem (x : ℝ) :
  x ∈ Set.Icc (0 : ℝ) (2 * Real.pi) →
  (Set.Icc (0 : ℝ) (2 * Real.pi) ∩ {x | Real.sin x ≥ Real.sqrt 3 / 2}) =
  Set.Icc (Real.pi / 3) ((2 * Real.pi) / 3) :=
by sorry

end NUMINAMATH_CALUDE_sine_range_theorem_l2919_291986


namespace NUMINAMATH_CALUDE_speaker_must_be_trulalya_l2919_291921

/-- Represents the two brothers --/
inductive Brother
| T1 -- Tralyalya
| T2 -- Trulalya

/-- Represents the two possible card suits --/
inductive Suit
| Orange
| Purple

/-- Represents the statement made by a brother --/
structure Statement where
  speaker : Brother
  claimed_suit : Suit

/-- Represents the actual state of the cards --/
structure CardState where
  T1_card : Suit
  T2_card : Suit

/-- Determines if a statement is truthful given the actual card state --/
def is_truthful (s : Statement) (cs : CardState) : Prop :=
  match s.speaker with
  | Brother.T1 => s.claimed_suit = cs.T1_card
  | Brother.T2 => s.claimed_suit = cs.T2_card

/-- The main theorem: Given the conditions, the speaker must be Trulalya (T2) --/
theorem speaker_must_be_trulalya :
  ∀ (s : Statement) (cs : CardState),
    s.claimed_suit = Suit.Purple →
    is_truthful s cs →
    s.speaker = Brother.T2 :=
by sorry

end NUMINAMATH_CALUDE_speaker_must_be_trulalya_l2919_291921


namespace NUMINAMATH_CALUDE_det_special_matrix_l2919_291940

theorem det_special_matrix (x y : ℝ) : 
  Matrix.det !![0, Real.cos x, Real.sin x; 
                -Real.cos x, 0, Real.cos y; 
                -Real.sin x, -Real.cos y, 0] = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l2919_291940


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2919_291956

/-- The side length of a rhombus given its diagonal lengths -/
theorem rhombus_side_length (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  ∃ (side : ℝ), side = 13 ∧ side^2 = (d1/2)^2 + (d2/2)^2 := by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2919_291956


namespace NUMINAMATH_CALUDE_problem_solution_l2919_291992

theorem problem_solution (x : ℝ) (h : 1 - 6/x + 9/x^2 = 0) : 2/x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2919_291992


namespace NUMINAMATH_CALUDE_binomial_150_150_l2919_291911

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_l2919_291911


namespace NUMINAMATH_CALUDE_sequence_properties_l2919_291945

def a (n : ℕ) : ℕ := 3 * (n^2 + n) + 7

theorem sequence_properties :
  (∀ k : ℕ, 
    5 ∣ a (5*k + 2) ∧ 
    ¬(5 ∣ a (5*k)) ∧ 
    ¬(5 ∣ a (5*k + 1)) ∧ 
    ¬(5 ∣ a (5*k + 3)) ∧ 
    ¬(5 ∣ a (5*k + 4))) ∧
  (∀ n t : ℕ, a n ≠ t^3) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2919_291945


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l2919_291910

theorem cubic_polynomials_common_roots :
  ∃! (a b : ℝ), 
    (∃ (r s : ℝ) (h : r ≠ s), 
      (∀ x : ℝ, x^3 + a*x^2 + 14*x + 7 = 0 ↔ x = r ∨ x = s ∨ x^3 + a*x^2 + 14*x + 7 = 0) ∧
      (∀ x : ℝ, x^3 + b*x^2 + 21*x + 15 = 0 ↔ x = r ∨ x = s ∨ x^3 + b*x^2 + 21*x + 15 = 0)) ∧
    a = 5 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l2919_291910


namespace NUMINAMATH_CALUDE_smallest_perfect_cube_divisor_l2919_291916

theorem smallest_perfect_cube_divisor (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) : 
  let n := p^2 * q^3 * r^5
  ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = k^3) → n ∣ m → p^6 * q^9 * r^15 ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_cube_divisor_l2919_291916


namespace NUMINAMATH_CALUDE_unique_c_value_l2919_291941

theorem unique_c_value : ∃! c : ℝ, ∀ x : ℝ, x * (3 * x + 1) - c > 0 ↔ x > -5/3 ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_value_l2919_291941


namespace NUMINAMATH_CALUDE_basketball_substitutions_l2919_291909

/-- The number of ways to make substitutions in a basketball game --/
def substitution_ways (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  let substitutes := total_players - starting_players
  let ways_0 := 1
  let ways_1 := starting_players * substitutes
  let ways_2 := ways_1 * (starting_players - 1) * (substitutes - 1)
  let ways_3 := ways_2 * (starting_players - 2) * (substitutes - 2)
  let ways_4 := ways_3 * (starting_players - 3) * (substitutes - 3)
  ways_0 + ways_1 + ways_2 + ways_3 + ways_4

/-- The main theorem about basketball substitutions --/
theorem basketball_substitutions :
  let total_ways := substitution_ways 15 5 4
  total_ways = 648851 ∧ total_ways % 100 = 51 := by
  sorry

#eval substitution_ways 15 5 4
#eval (substitution_ways 15 5 4) % 100

end NUMINAMATH_CALUDE_basketball_substitutions_l2919_291909


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l2919_291900

/-- The nth positive integer that is both odd and a multiple of 5 -/
def oddMultipleOf5 (n : ℕ) : ℕ := 2 * n * 5 - 5

theorem fifteenth_odd_multiple_of_5 : oddMultipleOf5 15 = 145 := by sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l2919_291900


namespace NUMINAMATH_CALUDE_boy_age_problem_l2919_291913

theorem boy_age_problem (total_boys : ℕ) (avg_all : ℕ) (avg_first : ℕ) (avg_last : ℕ) 
  (h_total : total_boys = 11)
  (h_avg_all : avg_all = 50)
  (h_avg_first : avg_first = 49)
  (h_avg_last : avg_last = 52) :
  (total_boys * avg_all : ℕ) = 
  (6 * avg_first : ℕ) + (6 * avg_last : ℕ) - 56 := by
  sorry

#check boy_age_problem

end NUMINAMATH_CALUDE_boy_age_problem_l2919_291913


namespace NUMINAMATH_CALUDE_parabola_closest_point_l2919_291961

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define the distance between two points
def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 - x2)^2 + (y1 - y2)^2

-- Theorem statement
theorem parabola_closest_point (a : ℝ) :
  (∀ x y : ℝ, parabola x y →
    ∃ xv yv : ℝ, parabola xv yv ∧
      ∀ x' y' : ℝ, parabola x' y' →
        distance_squared xv yv 0 a ≤ distance_squared x' y' 0 a) →
  a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_closest_point_l2919_291961


namespace NUMINAMATH_CALUDE_exists_20_digit_singular_l2919_291996

/-- A number is singular if it's a 2n-digit perfect square, and both its first n digits
    and last n digits are also perfect squares. --/
def is_singular (x : ℕ) : Prop :=
  ∃ (n : ℕ), 
    (x ≥ 10^(2*n - 1)) ∧ 
    (x < 10^(2*n)) ∧
    (∃ (y : ℕ), x = y^2) ∧
    (∃ (a b : ℕ), 
      x = a * 10^n + b ∧
      (∃ (c : ℕ), a = c^2) ∧
      (∃ (d : ℕ), b = d^2) ∧
      (a ≥ 10^(n-1)) ∧
      (b > 0))

/-- There exists a 20-digit singular number. --/
theorem exists_20_digit_singular : ∃ (x : ℕ), is_singular x ∧ (x ≥ 10^19) ∧ (x < 10^20) :=
sorry

end NUMINAMATH_CALUDE_exists_20_digit_singular_l2919_291996


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2919_291923

theorem trigonometric_identity (α β : Real) (h : α + β = Real.pi / 3) :
  Real.tan α + Real.tan β + Real.sqrt 3 * Real.tan α * Real.tan β = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2919_291923


namespace NUMINAMATH_CALUDE_average_difference_theorem_l2919_291920

/-- A school with students and teachers -/
structure School where
  num_students : ℕ
  num_teachers : ℕ
  class_sizes : List ℕ

/-- Calculate the average class size from a teacher's perspective -/
def teacher_average (school : School) : ℚ :=
  (school.class_sizes.sum : ℚ) / school.num_teachers

/-- Calculate the average class size from a student's perspective -/
def student_average (school : School) : ℚ :=
  (school.class_sizes.map (λ size => size * size)).sum / school.num_students

/-- The main theorem to prove -/
theorem average_difference_theorem (school : School) 
    (h1 : school.num_students = 120)
    (h2 : school.num_teachers = 5)
    (h3 : school.class_sizes = [60, 30, 20, 5, 5])
    (h4 : school.class_sizes.sum = school.num_students) : 
    teacher_average school - student_average school = -17.25 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_theorem_l2919_291920


namespace NUMINAMATH_CALUDE_intersection_dot_product_converse_not_always_true_l2919_291942

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line passing through (3,0)
def line_through_3_0 (l : ℝ → ℝ) : Prop := l 3 = 0

-- Define intersection points of a line and the parabola
def intersection_points (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ l A.1 = A.2 ∧ l B.1 = B.2

-- Define dot product of OA and OB
def dot_product (A B : ℝ × ℝ) : ℝ := A.1 * B.1 + A.2 * B.2

-- Theorem 1
theorem intersection_dot_product (l : ℝ → ℝ) (A B : ℝ × ℝ) :
  line_through_3_0 l → intersection_points l A B → dot_product A B = 3 :=
sorry

-- Theorem 2
theorem converse_not_always_true : 
  ∃ (A B : ℝ × ℝ) (l : ℝ → ℝ), parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  dot_product A B = 3 ∧ ¬(line_through_3_0 l) ∧ l A.1 = A.2 ∧ l B.1 = B.2 :=
sorry

end NUMINAMATH_CALUDE_intersection_dot_product_converse_not_always_true_l2919_291942


namespace NUMINAMATH_CALUDE_fraction_equality_l2919_291975

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / ((a : ℚ) + 37) = 925 / 1000 → a = 455 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2919_291975


namespace NUMINAMATH_CALUDE_hispanic_west_percentage_l2919_291935

def hispanic_ne : ℕ := 10
def hispanic_mw : ℕ := 8
def hispanic_south : ℕ := 22
def hispanic_west : ℕ := 15

def total_hispanic : ℕ := hispanic_ne + hispanic_mw + hispanic_south + hispanic_west

def percent_in_west : ℚ := hispanic_west / total_hispanic * 100

theorem hispanic_west_percentage :
  round percent_in_west = 27 :=
sorry

end NUMINAMATH_CALUDE_hispanic_west_percentage_l2919_291935


namespace NUMINAMATH_CALUDE_parabola_shift_l2919_291993

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := 2 * x^2

/-- The shifted parabola function -/
def shifted_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 - 5

/-- The horizontal shift amount -/
def h_shift : ℝ := 1

/-- The vertical shift amount -/
def v_shift : ℝ := -5

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - h_shift) + v_shift :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l2919_291993


namespace NUMINAMATH_CALUDE_find_t_value_l2919_291914

theorem find_t_value (t : ℝ) : 
  let A : Set ℝ := {-4, t^2}
  let B : Set ℝ := {t-5, 9, 1-t}
  9 ∈ A ∩ B → t = -3 :=
by sorry

end NUMINAMATH_CALUDE_find_t_value_l2919_291914


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2919_291994

theorem quadratic_roots_sum (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^3 + p^4*q^2 + p^2*q^4 + q^3 = 503 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2919_291994


namespace NUMINAMATH_CALUDE_f_sum_zero_l2919_291947

-- Define the function f(x) = ax^2 + bx
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem f_sum_zero (a b x₁ x₂ : ℝ) 
  (h₁ : a * b ≠ 0) 
  (h₂ : f a b x₁ = f a b x₂) 
  (h₃ : x₁ ≠ x₂) : 
  f a b (x₁ + x₂) = 0 := by
sorry

end NUMINAMATH_CALUDE_f_sum_zero_l2919_291947


namespace NUMINAMATH_CALUDE_rope_length_problem_l2919_291960

theorem rope_length_problem (L : ℝ) : 
  (L / 3 + 0.3 * (2 * L / 3)) - (L - (L / 3 + 0.3 * (2 * L / 3))) = 0.4 → L = 6 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_problem_l2919_291960


namespace NUMINAMATH_CALUDE_range_of_m_l2919_291915

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 4*x + 3
def g (m x : ℝ) : ℝ := m*x + 3 - 2*m

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, 
  (∀ x₁ ∈ Set.Icc 0 4, ∃ x₂ ∈ Set.Icc 0 4, f x₁ = g m x₂) ↔ 
  m ∈ Set.Iic (-2) ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2919_291915


namespace NUMINAMATH_CALUDE_volume_of_region_l2919_291929

def region (x y z : ℝ) : Prop :=
  abs (x + y + z) + abs (x + y - z) ≤ 12 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

theorem volume_of_region : 
  MeasureTheory.volume {p : ℝ × ℝ × ℝ | region p.1 p.2.1 p.2.2} = 108 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_region_l2919_291929


namespace NUMINAMATH_CALUDE_choir_members_proof_l2919_291925

theorem choir_members_proof :
  ∃! n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧ n % 7 = 3 ∧ n % 11 = 6 ∧ n = 220 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_proof_l2919_291925


namespace NUMINAMATH_CALUDE_same_gender_probability_l2919_291932

/-- The probability of selecting two students of the same gender from a group of 3 male and 2 female students -/
theorem same_gender_probability (male_students female_students : ℕ) 
  (h1 : male_students = 3)
  (h2 : female_students = 2) : 
  (Nat.choose male_students 2 + Nat.choose female_students 2) / Nat.choose (male_students + female_students) 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_same_gender_probability_l2919_291932


namespace NUMINAMATH_CALUDE_population_growth_rate_l2919_291966

theorem population_growth_rate (initial_population : ℝ) (population_after_two_years : ℝ) 
  (h1 : initial_population = 12000)
  (h2 : population_after_two_years = 18451.2) : 
  ∃ (r : ℝ), r = 24 ∧ population_after_two_years = initial_population * (1 + r / 100)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_l2919_291966


namespace NUMINAMATH_CALUDE_framing_for_enlarged_picture_l2919_291938

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (original_width original_height enlargement_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  ((perimeter_inches + 11) / 12 : ℕ)

/-- Theorem stating that for a 4x6 inch picture, quadrupled and with a 3-inch border, 9 feet of framing is needed. -/
theorem framing_for_enlarged_picture :
  min_framing_feet 4 6 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_framing_for_enlarged_picture_l2919_291938


namespace NUMINAMATH_CALUDE_exponential_function_point_l2919_291957

theorem exponential_function_point (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^x + 1
  f 0 = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_point_l2919_291957


namespace NUMINAMATH_CALUDE_range_of_f_leq_3_l2919_291934

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 8 then x^(1/3) else 2 * Real.exp (x - 8)

-- Theorem statement
theorem range_of_f_leq_3 :
  {x : ℝ | f x ≤ 3} = {x : ℝ | x ≤ 27} := by sorry

end NUMINAMATH_CALUDE_range_of_f_leq_3_l2919_291934


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2919_291985

theorem fractional_equation_solution (k : ℝ) : 
  (k / 2 + (2 - 3) / (2 - 1) = 1) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2919_291985


namespace NUMINAMATH_CALUDE_rectangle_area_is_220_l2919_291937

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of rectangle PQRS -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculate the area of a rectangle given three of its vertices -/
def rectangleArea (rect : Rectangle) : ℝ :=
  let width := abs (rect.Q.x - rect.P.x)
  let height := abs (rect.Q.y - rect.R.y)
  width * height

/-- Theorem: The area of rectangle PQRS with given vertices is 220 -/
theorem rectangle_area_is_220 : ∃ (S : Point),
  let rect : Rectangle := {
    P := { x := 15, y := 55 },
    Q := { x := 26, y := 55 },
    R := { x := 26, y := 35 },
    S := S
  }
  rectangleArea rect = 220 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_220_l2919_291937


namespace NUMINAMATH_CALUDE_sara_frosting_cans_l2919_291997

/-- Represents the data for each day's baking and frosting --/
structure DayData where
  baked : ℕ
  eaten : ℕ
  frostingPerCake : ℕ

/-- Calculates the total frosting cans needed for the remaining cakes --/
def totalFrostingCans (data : List DayData) : ℕ :=
  data.foldl (fun acc day => acc + (day.baked - day.eaten) * day.frostingPerCake) 0

/-- The main theorem stating the total number of frosting cans needed --/
theorem sara_frosting_cans : 
  let bakingData : List DayData := [
    ⟨7, 4, 2⟩,
    ⟨12, 6, 3⟩,
    ⟨8, 3, 4⟩,
    ⟨10, 2, 3⟩,
    ⟨15, 3, 2⟩
  ]
  totalFrostingCans bakingData = 92 := by
  sorry

#eval totalFrostingCans [
  ⟨7, 4, 2⟩,
  ⟨12, 6, 3⟩,
  ⟨8, 3, 4⟩,
  ⟨10, 2, 3⟩,
  ⟨15, 3, 2⟩
]

end NUMINAMATH_CALUDE_sara_frosting_cans_l2919_291997


namespace NUMINAMATH_CALUDE_three_toys_picked_l2919_291931

def toy_count : ℕ := 4

def probability_yo_yo_and_ball (n : ℕ) : ℚ :=
  if n < 2 then 0
  else (Nat.choose 2 (n - 2) : ℚ) / (Nat.choose toy_count n : ℚ)

theorem three_toys_picked :
  ∃ (n : ℕ), n ≤ toy_count ∧ probability_yo_yo_and_ball n = 1/2 ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_three_toys_picked_l2919_291931


namespace NUMINAMATH_CALUDE_quadratic_shift_l2919_291974

def f (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 - b * x + 3

def g (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 + b * x + 3

theorem quadratic_shift (b : ℝ) : 
  (∀ x, g b x = f b (x + 6)) → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_l2919_291974


namespace NUMINAMATH_CALUDE_jordan_rectangle_width_l2919_291953

theorem jordan_rectangle_width
  (carol_length : ℝ)
  (carol_width : ℝ)
  (jordan_length : ℝ)
  (jordan_width : ℝ)
  (h1 : carol_length = 5)
  (h2 : carol_width = 24)
  (h3 : jordan_length = 8)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 15 := by
sorry

end NUMINAMATH_CALUDE_jordan_rectangle_width_l2919_291953


namespace NUMINAMATH_CALUDE_both_make_shots_probability_l2919_291951

/-- The probability that both person A and person B make their shots -/
def prob_both_make_shots (prob_A prob_B : ℝ) : ℝ := prob_A * prob_B

theorem both_make_shots_probability :
  let prob_A : ℝ := 2/5
  let prob_B : ℝ := 1/2
  prob_both_make_shots prob_A prob_B = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_both_make_shots_probability_l2919_291951


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2919_291939

/-- Given a rectangle with side OA and diagonal OB, prove the value of k. -/
theorem rectangle_diagonal (OA OB : ℝ × ℝ) (k : ℝ) : 
  OA = (-3, 1) → 
  OB = (-2, k) → 
  (OA.1 * (OB.1 - OA.1) + OA.2 * (OB.2 - OA.2) = 0) → 
  k = 4 := by
  sorry

#check rectangle_diagonal

end NUMINAMATH_CALUDE_rectangle_diagonal_l2919_291939


namespace NUMINAMATH_CALUDE_fiveDigitNumbers_eq_ten_l2919_291969

/-- The number of five-digit natural numbers formed with digits 1 and 0, containing exactly three 1s -/
def fiveDigitNumbers : ℕ :=
  Nat.choose 5 2

/-- Theorem stating that the number of such five-digit numbers is 10 -/
theorem fiveDigitNumbers_eq_ten : fiveDigitNumbers = 10 := by
  sorry

end NUMINAMATH_CALUDE_fiveDigitNumbers_eq_ten_l2919_291969


namespace NUMINAMATH_CALUDE_caps_lost_per_year_l2919_291989

def caps_first_year : ℕ := 3 * 12
def caps_subsequent_years (years : ℕ) : ℕ := 5 * 12 * years
def christmas_caps (years : ℕ) : ℕ := 40 * years
def total_collection_years : ℕ := 5
def current_cap_count : ℕ := 401

theorem caps_lost_per_year :
  let total_caps := caps_first_year + 
                    caps_subsequent_years (total_collection_years - 1) + 
                    christmas_caps total_collection_years
  let total_lost := total_caps - current_cap_count
  (total_lost / total_collection_years : ℚ) = 15 := by sorry

end NUMINAMATH_CALUDE_caps_lost_per_year_l2919_291989


namespace NUMINAMATH_CALUDE_min_value_of_squares_l2919_291978

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_value_of_squares (p q r s t u v w : Int)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_set : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S)
  (h_sum : p + q + r + s ≥ 5) :
  (∀ a b c d e f g h : Int,
    a ∈ S → b ∈ S → c ∈ S → d ∈ S → e ∈ S → f ∈ S → g ∈ S → h ∈ S →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h →
    a + b + c + d ≥ 5 →
    (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 26) ∧
  (∃ a b c d e f g h : Int,
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    a + b + c + d ≥ 5 ∧
    (a + b + c + d)^2 + (e + f + g + h)^2 = 26) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l2919_291978


namespace NUMINAMATH_CALUDE_book_completion_time_l2919_291981

/-- Calculates the number of weeks needed to complete a book given the writing schedule and book length -/
theorem book_completion_time (writing_hours_per_day : ℕ) (pages_per_hour : ℕ) (total_pages : ℕ) :
  writing_hours_per_day = 3 →
  pages_per_hour = 5 →
  total_pages = 735 →
  (total_pages / (writing_hours_per_day * pages_per_hour) + 6) / 7 = 7 :=
by
  sorry

#check book_completion_time

end NUMINAMATH_CALUDE_book_completion_time_l2919_291981


namespace NUMINAMATH_CALUDE_sandy_potatoes_l2919_291999

theorem sandy_potatoes (nancy_potatoes : ℕ) (total_potatoes : ℕ) (sandy_potatoes : ℕ) : 
  nancy_potatoes = 6 → 
  total_potatoes = 13 → 
  total_potatoes = nancy_potatoes + sandy_potatoes → 
  sandy_potatoes = 7 := by
sorry

end NUMINAMATH_CALUDE_sandy_potatoes_l2919_291999


namespace NUMINAMATH_CALUDE_coin_difference_l2919_291912

def coin_values : List Nat := [5, 10, 25, 50]

def total_amount : Nat := 60

def min_coins (values : List Nat) (amount : Nat) : Nat :=
  sorry

def max_coins (values : List Nat) (amount : Nat) : Nat :=
  sorry

theorem coin_difference :
  max_coins coin_values total_amount - min_coins coin_values total_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_coin_difference_l2919_291912


namespace NUMINAMATH_CALUDE_school_function_participants_l2919_291943

theorem school_function_participants (boys girls : ℕ) 
  (h1 : 2 * (boys - girls) = 3 * 400)
  (h2 : 3 * girls = 4 * 150)
  (h3 : 2 * boys + 3 * girls = 3 * 550) :
  boys + girls = 800 := by
  sorry

end NUMINAMATH_CALUDE_school_function_participants_l2919_291943


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2919_291976

theorem fractional_equation_solution :
  ∃ x : ℚ, (1 / (2 * x) = 2 / (x - 3)) ∧ x = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2919_291976


namespace NUMINAMATH_CALUDE_deepak_present_age_l2919_291930

/-- Represents the ages of two people with a given ratio --/
structure AgeRatio where
  x : ℕ
  rahul_age : ℕ := 4 * x
  deepak_age : ℕ := 3 * x

/-- The theorem stating Deepak's present age given the conditions --/
theorem deepak_present_age (ar : AgeRatio) 
  (h1 : ar.rahul_age + 6 = 50) : 
  ar.deepak_age = 33 := by
  sorry

#check deepak_present_age

end NUMINAMATH_CALUDE_deepak_present_age_l2919_291930


namespace NUMINAMATH_CALUDE_angle_EFG_is_60_degrees_l2919_291901

-- Define the angles as real numbers
variable (x : ℝ)
variable (angle_CFG angle_CEB angle_BEA angle_EFG : ℝ)

-- Define the parallel lines property
variable (AD_parallel_FG : Prop)

-- State the theorem
theorem angle_EFG_is_60_degrees 
  (h1 : AD_parallel_FG)
  (h2 : angle_CFG = 1.5 * x)
  (h3 : angle_CEB = x)
  (h4 : angle_BEA = 2 * x)
  (h5 : angle_EFG = angle_CFG) :
  angle_EFG = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_EFG_is_60_degrees_l2919_291901


namespace NUMINAMATH_CALUDE_ellipse_incenter_ratio_theorem_l2919_291919

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Represents the foci of an ellipse -/
structure Foci (e : Ellipse) where
  left : Point
  right : Point

/-- Represents a triangle formed by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Represents the incenter of a triangle -/
def incenter (t : Triangle) : Point :=
  sorry -- Definition of incenter

/-- Checks if a point is on a line segment -/
def isOnSegment (p : Point) (a : Point) (b : Point) : Prop :=
  sorry -- Definition of being on a line segment

theorem ellipse_incenter_ratio_theorem
  (e : Ellipse) (m : Point) (f : Foci e) (p n : Point) :
  isOnEllipse e m →
  p = incenter (Triangle.mk m f.left f.right) →
  isOnSegment n f.left f.right →
  isOnSegment n m p →
  (m.x - n.x)^2 + (m.y - n.y)^2 > 0 →
  (n.x - p.x)^2 + (n.y - p.y)^2 > 0 →
  ∃ (r : ℝ), r > 0 ∧
    r = (m.x - n.x)^2 + (m.y - n.y)^2 / ((n.x - p.x)^2 + (n.y - p.y)^2) ∧
    r = (m.x - f.left.x)^2 + (m.y - f.left.y)^2 / ((f.left.x - p.x)^2 + (f.left.y - p.y)^2) ∧
    r = (m.x - f.right.x)^2 + (m.y - f.right.y)^2 / ((f.right.x - p.x)^2 + (f.right.y - p.y)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_incenter_ratio_theorem_l2919_291919


namespace NUMINAMATH_CALUDE_min_value_theorem_l2919_291903

theorem min_value_theorem (x y : ℝ) (h : x * y > 0) :
  ∃ (min_val : ℝ), min_val = 4 - 2 * Real.sqrt 2 ∧
  ∀ (z : ℝ), y / (x + y) + 2 * x / (2 * x + y) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2919_291903


namespace NUMINAMATH_CALUDE_total_pokemon_cards_l2919_291955

/-- The number of people with Pokemon cards -/
def num_people : ℕ := 4

/-- The number of Pokemon cards each person has -/
def cards_per_person : ℕ := 14

/-- The total number of Pokemon cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem total_pokemon_cards : total_cards = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_pokemon_cards_l2919_291955


namespace NUMINAMATH_CALUDE_student_path_probability_l2919_291926

/-- Represents a point on the city map -/
structure Point where
  east : Nat
  south : Nat

/-- Calculates the number of paths between two points -/
def num_paths (start finish : Point) : Nat :=
  Nat.choose (finish.east - start.east + finish.south - start.south) (finish.east - start.east)

/-- The probability of choosing a specific path -/
def path_probability (start finish : Point) : ℚ :=
  1 / 2 ^ (finish.east - start.east + finish.south - start.south)

theorem student_path_probability :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨4, 3⟩
  let C : Point := ⟨2, 1⟩
  let D : Point := ⟨3, 2⟩
  let total_paths := num_paths A B
  let paths_through_C_and_D := num_paths A C * num_paths C D * num_paths D B
  paths_through_C_and_D / total_paths = 12 / 35 := by
  sorry

#eval num_paths ⟨0, 0⟩ ⟨4, 3⟩  -- Should output 35
#eval num_paths ⟨0, 0⟩ ⟨2, 1⟩ * num_paths ⟨2, 1⟩ ⟨3, 2⟩ * num_paths ⟨3, 2⟩ ⟨4, 3⟩  -- Should output 12

end NUMINAMATH_CALUDE_student_path_probability_l2919_291926


namespace NUMINAMATH_CALUDE_moores_law_decade_l2919_291928

/-- Moore's Law transistor growth over a decade -/
theorem moores_law_decade (initial_transistors : ℕ) (years : ℕ) : 
  initial_transistors = 250000 →
  years = 10 →
  initial_transistors * (2 ^ (years / 2)) = 8000000 :=
by sorry

end NUMINAMATH_CALUDE_moores_law_decade_l2919_291928


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l2919_291949

/-- The area of a square with one side on y = 7 and endpoints on y = x^2 + 4x + 3 is 32 -/
theorem square_area_on_parabola : ∃ (s : ℝ),
  (∃ (x₁ x₂ : ℝ),
    x₁^2 + 4*x₁ + 3 = 7 ∧
    x₂^2 + 4*x₂ + 3 = 7 ∧
    s = |x₂ - x₁|) →
  s^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l2919_291949


namespace NUMINAMATH_CALUDE_integral_x_zero_to_one_l2919_291967

theorem integral_x_zero_to_one :
  ∫ x in (0 : ℝ)..1, x = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_integral_x_zero_to_one_l2919_291967


namespace NUMINAMATH_CALUDE_expression_simplification_l2919_291991

theorem expression_simplification (x : ℝ) :
  2*x*(4*x^2 - 3*x + 1) - 7*(2*x^2 - 3*x + 4) = 8*x^3 - 20*x^2 + 23*x - 28 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2919_291991


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2919_291988

theorem geometric_sequence_fifth_term 
  (t : ℕ → ℝ) 
  (h_positive : ∀ n, t n > 0) 
  (h_decreasing : t 1 > t 2) 
  (h_sum : t 1 + t 2 = 15/2) 
  (h_sum_squares : t 1^2 + t 2^2 = 153/4) 
  (h_geometric : ∃ r : ℝ, ∀ n, t (n+1) = t n * r) :
  t 5 = 3/128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2919_291988


namespace NUMINAMATH_CALUDE_josh_marbles_l2919_291984

theorem josh_marbles (initial_marbles final_marbles received_marbles : ℕ) :
  final_marbles = initial_marbles + received_marbles →
  final_marbles = 42 →
  received_marbles = 20 →
  initial_marbles = 22 := by sorry

end NUMINAMATH_CALUDE_josh_marbles_l2919_291984


namespace NUMINAMATH_CALUDE_salary_increase_l2919_291944

theorem salary_increase (num_employees : ℕ) (avg_salary : ℚ) (manager_salary : ℚ) :
  num_employees = 20 ∧ avg_salary = 1500 ∧ manager_salary = 14100 →
  (((num_employees : ℚ) * avg_salary + manager_salary) / (num_employees + 1 : ℚ)) - avg_salary = 600 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_l2919_291944


namespace NUMINAMATH_CALUDE_second_car_traveled_5km_l2919_291968

/-- Represents the distance traveled by the second car -/
def second_car_distance : ℝ := 5

/-- The initial distance between the two cars -/
def initial_distance : ℝ := 105

/-- The distance traveled by the first car before turning back -/
def first_car_distance : ℝ := 25 + 15 + 25

/-- The final distance between the two cars -/
def final_distance : ℝ := 20

/-- Theorem stating that the second car traveled 5 km -/
theorem second_car_traveled_5km :
  initial_distance - (first_car_distance + 15 + second_car_distance) = final_distance :=
by sorry

end NUMINAMATH_CALUDE_second_car_traveled_5km_l2919_291968


namespace NUMINAMATH_CALUDE_f_derivative_roots_l2919_291964

-- Define the function f
def f (x : ℝ) : ℝ := (1 - x) * (2 - x) * (3 - x) * (4 - x)

-- State the theorem
theorem f_derivative_roots :
  ∃ (r₁ r₂ r₃ : ℝ),
    (1 < r₁ ∧ r₁ < 2) ∧
    (2 < r₂ ∧ r₂ < 3) ∧
    (3 < r₃ ∧ r₃ < 4) ∧
    (∀ x : ℝ, deriv f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) :=
sorry

end NUMINAMATH_CALUDE_f_derivative_roots_l2919_291964


namespace NUMINAMATH_CALUDE_discount_doubles_with_time_ratio_two_l2919_291922

/-- Represents the true discount calculation for a bill -/
structure BillDiscount where
  face_value : ℝ
  initial_discount : ℝ
  time_ratio : ℝ

/-- Calculates the discount for a different time period based on the time ratio -/
def discount_for_different_time (bill : BillDiscount) : ℝ :=
  bill.initial_discount * bill.time_ratio

/-- Theorem stating that for a bill of 110 with initial discount of 10 and time ratio of 2,
    the discount for the different time period is 20 -/
theorem discount_doubles_with_time_ratio_two :
  let bill := BillDiscount.mk 110 10 2
  discount_for_different_time bill = 20 := by
  sorry

#eval discount_for_different_time (BillDiscount.mk 110 10 2)

end NUMINAMATH_CALUDE_discount_doubles_with_time_ratio_two_l2919_291922


namespace NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l2919_291970

theorem negation_of_cube_odd_is_odd :
  (¬ ∀ n : ℤ, Odd n → Odd (n^3)) ↔ (∃ n : ℤ, Odd n ∧ Even (n^3)) :=
sorry

end NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l2919_291970


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l2919_291995

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line_equation (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

/-- A point (x, y) is a reflection of (x₀, y₀) across the x-axis if x = x₀ and y = -y₀ -/
def is_reflection_x_axis (x y x₀ y₀ : ℝ) : Prop :=
  x = x₀ ∧ y = -y₀

theorem reflected_ray_equation :
  is_reflection_x_axis 2 (-1) 2 1 →
  line_equation 2 (-1) 4 5 x y ↔ 3 * x - y - 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l2919_291995


namespace NUMINAMATH_CALUDE_sequence_growth_l2919_291906

theorem sequence_growth (k : ℕ) (h : k ≥ 2) : 
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end NUMINAMATH_CALUDE_sequence_growth_l2919_291906


namespace NUMINAMATH_CALUDE_max_value_d_l2919_291979

theorem max_value_d (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (product_condition : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_d_l2919_291979


namespace NUMINAMATH_CALUDE_greek_yogurt_cost_per_pack_l2919_291904

theorem greek_yogurt_cost_per_pack 
  (total_packs : ℕ)
  (expired_percentage : ℚ)
  (total_refund : ℚ)
  (h1 : total_packs = 80)
  (h2 : expired_percentage = 40 / 100)
  (h3 : total_refund = 384) :
  total_refund / (expired_percentage * total_packs) = 12 := by
sorry

end NUMINAMATH_CALUDE_greek_yogurt_cost_per_pack_l2919_291904


namespace NUMINAMATH_CALUDE_value_of_x_l2919_291908

theorem value_of_x : (2009^2 - 2009) / 2009 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2919_291908


namespace NUMINAMATH_CALUDE_trash_can_problem_l2919_291946

/-- Represents the unit price of trash can A -/
def price_A : ℝ := 60

/-- Represents the unit price of trash can B -/
def price_B : ℝ := 100

/-- Represents the total number of trash cans needed -/
def total_cans : ℕ := 200

/-- Represents the maximum total cost allowed -/
def max_cost : ℝ := 15000

theorem trash_can_problem :
  (3 * price_A + 4 * price_B = 580) ∧
  (6 * price_A + 5 * price_B = 860) ∧
  (∀ a : ℕ, a ≥ 125 → 
    (price_A * a + price_B * (total_cans - a) ≤ max_cost)) ∧
  (∀ a : ℕ, a < 125 → 
    (price_A * a + price_B * (total_cans - a) > max_cost)) :=
by sorry

end NUMINAMATH_CALUDE_trash_can_problem_l2919_291946


namespace NUMINAMATH_CALUDE_widget_selling_price_l2919_291980

-- Define the problem parameters
def widget_cost : ℝ := 3
def monthly_rent : ℝ := 10000
def tax_rate : ℝ := 0.20
def worker_salary : ℝ := 2500
def num_workers : ℕ := 4
def widgets_sold : ℕ := 5000
def total_profit : ℝ := 4000

-- Define the theorem
theorem widget_selling_price :
  let worker_expenses : ℝ := worker_salary * num_workers
  let total_expenses : ℝ := monthly_rent + worker_expenses
  let widget_expenses : ℝ := widget_cost * widgets_sold
  let taxes : ℝ := tax_rate * total_profit
  let total_expenses_with_taxes : ℝ := total_expenses + widget_expenses + taxes
  let total_revenue : ℝ := total_expenses_with_taxes + total_profit
  let selling_price : ℝ := total_revenue / widgets_sold
  selling_price = 7.96 := by
  sorry

end NUMINAMATH_CALUDE_widget_selling_price_l2919_291980


namespace NUMINAMATH_CALUDE_m_divided_by_8_l2919_291958

theorem m_divided_by_8 (m : ℕ) (h : m = 16^1024) : m / 8 = 2^4093 := by
  sorry

end NUMINAMATH_CALUDE_m_divided_by_8_l2919_291958


namespace NUMINAMATH_CALUDE_four_line_segment_lengths_exists_distinct_positive_integer_lengths_l2919_291971

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a function to check if three lines are concurrent
def areConcurrent (l₁ l₂ l₃ : Line) : Prop := sorry

-- Define a function to check if two lines intersect
def intersect (l₁ l₂ : Line) : Prop := sorry

-- Define a function to get the length of a line segment
def segmentLength (p₁ p₂ : Point) : ℝ := sorry

-- Define the configuration of four lines
structure FourLineConfiguration :=
  (lines : Fin 4 → Line)
  (intersectionPoints : Fin 6 → Point)
  (segmentLengths : Fin 8 → ℝ)
  (twoLinesIntersect : ∀ i j, i ≠ j → intersect (lines i) (lines j))
  (noThreeConcurrent : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬areConcurrent (lines i) (lines j) (lines k))
  (eightSegments : ∀ i, segmentLengths i > 0)
  (distinctSegments : ∀ i j, i ≠ j → segmentLengths i ≠ segmentLengths j)

theorem four_line_segment_lengths 
  (config : FourLineConfiguration) : 
  (∀ i : Fin 8, config.segmentLengths i = i.val + 1) → False :=
sorry

theorem exists_distinct_positive_integer_lengths 
  (config : FourLineConfiguration) :
  ∃ (lengths : Fin 8 → ℕ), ∀ i : Fin 8, config.segmentLengths i = lengths i ∧ lengths i > 0 :=
sorry

end NUMINAMATH_CALUDE_four_line_segment_lengths_exists_distinct_positive_integer_lengths_l2919_291971


namespace NUMINAMATH_CALUDE_grey_perimeter_fraction_five_strips_l2919_291962

/-- A square divided into strips -/
structure StrippedSquare where
  num_strips : ℕ
  num_grey_strips : ℕ
  h_grey_strips : num_grey_strips ≤ num_strips

/-- The fraction of the perimeter that is grey -/
def grey_perimeter_fraction (s : StrippedSquare) : ℚ :=
  s.num_grey_strips / s.num_strips

/-- Theorem: In a square divided into 5 strips with 2 grey strips, 
    the fraction of the perimeter that is grey is 2/5 -/
theorem grey_perimeter_fraction_five_strips 
  (s : StrippedSquare) 
  (h_five_strips : s.num_strips = 5)
  (h_two_grey : s.num_grey_strips = 2) : 
  grey_perimeter_fraction s = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_grey_perimeter_fraction_five_strips_l2919_291962


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_35_l2919_291983

theorem inverse_of_3_mod_35 : ∃ x : ℕ, x < 35 ∧ (3 * x) % 35 = 1 :=
by
  use 12
  sorry

end NUMINAMATH_CALUDE_inverse_of_3_mod_35_l2919_291983
