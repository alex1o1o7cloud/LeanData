import Mathlib

namespace NUMINAMATH_CALUDE_binge_watching_duration_l2509_250995

/-- Proves that given a TV show with 90 episodes of 20 minutes each, and a viewing time of 2 hours per day, it will take 15 days to finish watching the entire show. -/
theorem binge_watching_duration (num_episodes : ℕ) (episode_length : ℕ) (daily_viewing_time : ℕ) : 
  num_episodes = 90 → 
  episode_length = 20 → 
  daily_viewing_time = 120 → 
  (num_episodes * episode_length) / daily_viewing_time = 15 := by
  sorry

#check binge_watching_duration

end NUMINAMATH_CALUDE_binge_watching_duration_l2509_250995


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_36_l2509_250942

/-- Represents the number of fire drill sites -/
def num_sites : ℕ := 3

/-- Represents the number of fire brigades -/
def num_brigades : ℕ := 4

/-- Represents the condition that each site must have at least one brigade -/
def min_brigade_per_site : ℕ := 1

/-- The number of ways to allocate fire brigades to sites -/
def allocation_schemes : ℕ := sorry

/-- Theorem stating that the number of allocation schemes is 36 -/
theorem allocation_schemes_eq_36 : allocation_schemes = 36 := by sorry

end NUMINAMATH_CALUDE_allocation_schemes_eq_36_l2509_250942


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_40_l2509_250939

/-- Represents a triangle divided into four triangles and one quadrilateral -/
structure DividedTriangle where
  total_area : ℝ
  triangle1_area : ℝ
  triangle2_area : ℝ
  triangle3_area : ℝ
  triangle4_area : ℝ
  quadrilateral_area : ℝ

/-- The sum of all areas equals the total area of the triangle -/
def area_sum (dt : DividedTriangle) : Prop :=
  dt.total_area = dt.triangle1_area + dt.triangle2_area + dt.triangle3_area + dt.triangle4_area + dt.quadrilateral_area

/-- The theorem stating that given the areas of the four triangles, the area of the quadrilateral is 40 -/
theorem quadrilateral_area_is_40 (dt : DividedTriangle) 
  (h1 : dt.triangle1_area = 5)
  (h2 : dt.triangle2_area = 10)
  (h3 : dt.triangle3_area = 10)
  (h4 : dt.triangle4_area = 15)
  (h_sum : area_sum dt) : 
  dt.quadrilateral_area = 40 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_40_l2509_250939


namespace NUMINAMATH_CALUDE_discount_percentage_l2509_250927

theorem discount_percentage (regular_price : ℝ) (num_shirts : ℕ) (total_paid : ℝ) : 
  regular_price = 50 ∧ num_shirts = 2 ∧ total_paid = 60 →
  (regular_price * num_shirts - total_paid) / (regular_price * num_shirts) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l2509_250927


namespace NUMINAMATH_CALUDE_average_and_relation_implies_values_l2509_250921

theorem average_and_relation_implies_values :
  ∀ x y : ℝ,
  (15 + 30 + x + y) / 4 = 25 →
  x = y + 10 →
  x = 32.5 ∧ y = 22.5 := by
sorry

end NUMINAMATH_CALUDE_average_and_relation_implies_values_l2509_250921


namespace NUMINAMATH_CALUDE_first_player_wins_l2509_250949

-- Define the game state
structure GameState where
  hour : Nat

-- Define player moves
def firstPlayerMove (state : GameState) : GameState :=
  { hour := (state.hour + 2) % 12 }

def secondPlayerMoveInitial (state : GameState) : GameState :=
  { hour := 5 }

def secondPlayerMoveSubsequent (state : GameState) (move : Nat) : GameState :=
  { hour := (state.hour + move) % 12 }

def firstPlayerMoveSubsequent (state : GameState) (move : Nat) : GameState :=
  { hour := (state.hour + move) % 12 }

-- Define the game sequence
def gameSequence (secondPlayerLastMove : Nat) : GameState :=
  let initial := { hour := 0 }  -- 12 o'clock
  let afterFirstMove := firstPlayerMove initial
  let afterSecondMove := secondPlayerMoveInitial afterFirstMove
  let afterThirdMove := firstPlayerMoveSubsequent afterSecondMove 3
  let afterFourthMove := secondPlayerMoveSubsequent afterThirdMove secondPlayerLastMove
  let finalState := 
    if secondPlayerLastMove = 2 then
      firstPlayerMoveSubsequent afterFourthMove 3
    else
      firstPlayerMoveSubsequent afterFourthMove 2

  finalState

-- Theorem statement
theorem first_player_wins (secondPlayerLastMove : Nat) 
  (h : secondPlayerLastMove = 2 ∨ secondPlayerLastMove = 3) : 
  (gameSequence secondPlayerLastMove).hour = 6 := by
  sorry

end NUMINAMATH_CALUDE_first_player_wins_l2509_250949


namespace NUMINAMATH_CALUDE_sum_of_powers_positive_l2509_250926

theorem sum_of_powers_positive 
  (a b c : ℝ) 
  (h1 : a * b * c > 0) 
  (h2 : a + b + c > 0) : 
  ∀ n : ℕ, a^n + b^n + c^n > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_positive_l2509_250926


namespace NUMINAMATH_CALUDE_inequality_solution_implies_k_value_l2509_250980

theorem inequality_solution_implies_k_value (k : ℝ) : 
  (∀ x : ℝ, |k * x - 4| ≤ 2 ↔ 1 ≤ x ∧ x ≤ 3) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_k_value_l2509_250980


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2509_250964

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 1) > 4/x + 19/10) ↔ (-2 < x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2509_250964


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2509_250972

theorem arithmetic_calculation : (5 * 4)^2 + (10 * 2) - 36 / 3 = 408 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2509_250972


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l2509_250962

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n → n ≤ 85 := by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l2509_250962


namespace NUMINAMATH_CALUDE_expansion_terms_abcd_efghi_l2509_250957

/-- The number of terms in the expansion of a product of two sums -/
def expansion_terms (n m : ℕ) : ℕ := n * m

/-- The first group (a+b+c+d) has 4 terms -/
def first_group_terms : ℕ := 4

/-- The second group (e+f+g+h+i) has 5 terms -/
def second_group_terms : ℕ := 5

/-- Theorem: The number of terms in the expansion of (a+b+c+d)(e+f+g+h+i) is 20 -/
theorem expansion_terms_abcd_efghi :
  expansion_terms first_group_terms second_group_terms = 20 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_abcd_efghi_l2509_250957


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l2509_250907

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

/-- The base 3 representation of the number -/
def base3Digits : List Nat := [2, 0, 2, 0, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Digits = 182 := by
  sorry

#eval base3ToBase10 base3Digits

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l2509_250907


namespace NUMINAMATH_CALUDE_three_positions_from_six_people_l2509_250946

def number_of_people : ℕ := 6
def number_of_positions : ℕ := 3

theorem three_positions_from_six_people :
  (number_of_people.factorial) / ((number_of_people - number_of_positions).factorial) = 120 :=
by sorry

end NUMINAMATH_CALUDE_three_positions_from_six_people_l2509_250946


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2509_250976

/-- Given two vectors a and b in ℝ², if they are parallel and a = (3, 4) and b = (x, 1/2), then x = 3/8 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) :
  a = (3, 4) →
  b = (x, 1/2) →
  ∃ (k : ℝ), a = k • b →
  x = 3/8 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2509_250976


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2509_250953

theorem complex_equation_solution (i : ℂ) (m : ℝ) 
  (h1 : i ^ 2 = -1)
  (h2 : (2 : ℂ) / (1 + i) = 1 + m * i) : 
  m = -1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2509_250953


namespace NUMINAMATH_CALUDE_other_solution_quadratic_l2509_250969

theorem other_solution_quadratic (h : 40 * (4/5)^2 - 69 * (4/5) + 24 = 0) :
  40 * (3/8)^2 - 69 * (3/8) + 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_l2509_250969


namespace NUMINAMATH_CALUDE_climbing_solution_l2509_250981

/-- Represents the climbing problem with given conditions -/
def ClimbingProblem (v : ℝ) : Prop :=
  let t₁ : ℝ := 14 / 2 + 1  -- Time on first day
  let t₂ : ℝ := 14 / 2 - 1  -- Time on second day
  let v₁ : ℝ := v - 0.5     -- Speed on first day
  let v₂ : ℝ := v           -- Speed on second day
  (v₁ * t₁ + v₂ * t₂ = 52) ∧ (t₁ + t₂ = 14)

/-- The theorem stating the solution to the climbing problem -/
theorem climbing_solution : ∃ v : ℝ, ClimbingProblem v ∧ v = 4 := by
  sorry

end NUMINAMATH_CALUDE_climbing_solution_l2509_250981


namespace NUMINAMATH_CALUDE_expand_expression_l2509_250997

theorem expand_expression (x y z : ℝ) : 
  (x + 10 + y) * (2 * z + 10) = 2 * x * z + 2 * y * z + 10 * x + 10 * y + 20 * z + 100 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2509_250997


namespace NUMINAMATH_CALUDE_snooker_ticket_difference_l2509_250902

theorem snooker_ticket_difference (vip_price gen_price : ℚ) 
  (total_tickets total_revenue : ℚ) (min_vip min_gen : ℕ) :
  vip_price = 40 →
  gen_price = 15 →
  total_tickets = 320 →
  total_revenue = 7500 →
  min_vip = 80 →
  min_gen = 100 →
  ∃ (vip_sold gen_sold : ℕ),
    vip_sold + gen_sold = total_tickets ∧
    vip_price * vip_sold + gen_price * gen_sold = total_revenue ∧
    vip_sold ≥ min_vip ∧
    gen_sold ≥ min_gen ∧
    gen_sold - vip_sold = 104 :=
by sorry

end NUMINAMATH_CALUDE_snooker_ticket_difference_l2509_250902


namespace NUMINAMATH_CALUDE_jessica_paper_count_l2509_250903

def paper_weight : ℚ := 1/5
def envelope_weight : ℚ := 2/5

def total_weight (num_papers : ℕ) : ℚ :=
  paper_weight * num_papers + envelope_weight

theorem jessica_paper_count :
  ∃ (num_papers : ℕ),
    (1 < total_weight num_papers) ∧
    (total_weight num_papers ≤ 2) ∧
    (num_papers = 8) := by
  sorry

end NUMINAMATH_CALUDE_jessica_paper_count_l2509_250903


namespace NUMINAMATH_CALUDE_shadow_arrangements_l2509_250955

def word_length : Nat := 6
def selection_size : Nat := 4
def remaining_letters : Nat := word_length - 1  -- excluding 'a'
def letters_to_choose : Nat := selection_size - 1  -- excluding 'a'

theorem shadow_arrangements : 
  (Nat.choose remaining_letters letters_to_choose) * 
  (Nat.factorial selection_size) = 240 := by
sorry

end NUMINAMATH_CALUDE_shadow_arrangements_l2509_250955


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2509_250988

theorem triangle_angle_proof (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α = 100 →          -- One angle is 100°
  β = 2 * γ →        -- One angle is twice the other
  γ = 26 :=          -- The smallest angle is 26°
by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2509_250988


namespace NUMINAMATH_CALUDE_corral_area_ratio_l2509_250985

/-- The side length of a small equilateral triangular corral -/
def small_side : ℝ := sorry

/-- The side length of the large equilateral triangular corral -/
def large_side : ℝ := 3 * small_side

/-- The area of a single small equilateral triangular corral -/
def small_area : ℝ := sorry

/-- The area of the large equilateral triangular corral -/
def large_area : ℝ := sorry

/-- The total area of all nine small equilateral triangular corrals -/
def total_small_area : ℝ := 9 * small_area

theorem corral_area_ratio : total_small_area = large_area := by
  sorry

end NUMINAMATH_CALUDE_corral_area_ratio_l2509_250985


namespace NUMINAMATH_CALUDE_blue_chip_fraction_l2509_250954

theorem blue_chip_fraction (total : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : total = 60)
  (h2 : red = 34)
  (h3 : green = 16) :
  (total - red - green : ℚ) / total = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_blue_chip_fraction_l2509_250954


namespace NUMINAMATH_CALUDE_other_sales_percentage_l2509_250941

theorem other_sales_percentage (pen_sales pencil_sales notebook_sales : ℝ) 
  (h_pen : pen_sales = 20)
  (h_pencil : pencil_sales = 15)
  (h_notebook : notebook_sales = 30)
  (h_total : pen_sales + pencil_sales + notebook_sales + 100 - (pen_sales + pencil_sales + notebook_sales) = 100) :
  100 - (pen_sales + pencil_sales + notebook_sales) = 35 := by
  sorry

end NUMINAMATH_CALUDE_other_sales_percentage_l2509_250941


namespace NUMINAMATH_CALUDE_count_increasing_digit_numbers_eq_502_l2509_250918

def is_increasing_digits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get i < (n.digits 10).get j

def count_increasing_digit_numbers : ℕ :=
  (Finset.range 8).sum (λ k => Nat.choose 9 (k + 2))

theorem count_increasing_digit_numbers_eq_502 :
  count_increasing_digit_numbers = 502 :=
sorry

end NUMINAMATH_CALUDE_count_increasing_digit_numbers_eq_502_l2509_250918


namespace NUMINAMATH_CALUDE_felicity_store_visits_l2509_250909

/-- The number of times Felicity's family goes to the store per week -/
def store_visits_per_week (total_sticks : ℕ) (completion_percentage : ℚ) (weeks : ℕ) : ℚ :=
  (total_sticks : ℚ) * completion_percentage / weeks

theorem felicity_store_visits :
  store_visits_per_week 400 (60 / 100) 80 = 3 := by
  sorry

end NUMINAMATH_CALUDE_felicity_store_visits_l2509_250909


namespace NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l2509_250924

theorem integer_solutions_of_inequalities :
  {x : ℤ | (2 * x - 1 < x + 1) ∧ (1 - 2 * (x - 1) ≤ 3)} = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l2509_250924


namespace NUMINAMATH_CALUDE_closest_whole_number_to_expression_l2509_250984

theorem closest_whole_number_to_expression : 
  ∃ (n : ℕ), n = 1000 ∧ 
  ∀ (m : ℕ), |((10^2010 + 5 * 10^2012) : ℚ) / ((2 * 10^2011 + 3 * 10^2011) : ℚ) - n| ≤ 
             |((10^2010 + 5 * 10^2012) : ℚ) / ((2 * 10^2011 + 3 * 10^2011) : ℚ) - m| :=
by sorry

end NUMINAMATH_CALUDE_closest_whole_number_to_expression_l2509_250984


namespace NUMINAMATH_CALUDE_marble_arrangement_theorem_l2509_250970

/-- The number of green marbles -/
def green_marbles : ℕ := 6

/-- The maximum number of red marbles that satisfies the arrangement condition -/
def max_red_marbles : ℕ := 18

/-- The total number of marbles in the arrangement -/
def total_marbles : ℕ := green_marbles + max_red_marbles

/-- The number of ways to arrange the marbles -/
def arrangement_count : ℕ := Nat.choose total_marbles green_marbles

theorem marble_arrangement_theorem :
  arrangement_count % 1000 = 564 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_theorem_l2509_250970


namespace NUMINAMATH_CALUDE_draw_red_black_red_prob_value_l2509_250935

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (hTotal : total_cards = 52)
  (hRed : red_cards = 26)
  (hBlack : black_cards = 26)
  (hSum : red_cards + black_cards = total_cards)

/-- The probability of drawing a red card first, then a black card, and then a red card -/
def draw_red_black_red_prob (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards *
  (d.black_cards : ℚ) / (d.total_cards - 1) *
  ((d.red_cards - 1) : ℚ) / (d.total_cards - 2)

/-- Theorem stating the probability is 13/102 -/
theorem draw_red_black_red_prob_value (d : Deck) :
  draw_red_black_red_prob d = 13 / 102 := by
  sorry

end NUMINAMATH_CALUDE_draw_red_black_red_prob_value_l2509_250935


namespace NUMINAMATH_CALUDE_geometric_sequence_min_sum_l2509_250977

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_min_sum
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_prod : a 3 * a 5 = 64) :
  ∃ (min : ℝ), min = 16 ∧ ∀ (a' : ℕ → ℝ),
    GeometricSequence a' → (∀ n, a' n > 0) → a' 3 * a' 5 = 64 →
    a' 1 + a' 7 ≥ min :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_sum_l2509_250977


namespace NUMINAMATH_CALUDE_sum_f_negative_l2509_250945

noncomputable def f (x : ℝ) : ℝ := x + x^3 + x^5

theorem sum_f_negative (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ < 0) 
  (h₂ : x₂ + x₃ < 0) 
  (h₃ : x₃ + x₁ < 0) : 
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l2509_250945


namespace NUMINAMATH_CALUDE_two_numbers_problem_l2509_250994

theorem two_numbers_problem (a b : ℝ) (h1 : a + b = 40) (h2 : a * b = 375) :
  |a - b| = 10 := by sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l2509_250994


namespace NUMINAMATH_CALUDE_complex_number_problem_l2509_250971

theorem complex_number_problem (z₁ z₂ : ℂ) 
  (h₁ : (z₁ - 2) * Complex.I = 1 + Complex.I)
  (h₂ : z₂.im = 2)
  (h₃ : (z₁ * z₂).im = 0) :
  z₂ = 6 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2509_250971


namespace NUMINAMATH_CALUDE_sqrt_x_plus_y_plus_five_halves_l2509_250929

theorem sqrt_x_plus_y_plus_five_halves (x y : ℝ) : 
  y = Real.sqrt (2 * x - 3) + Real.sqrt (3 - 2 * x) + 5 →
  Real.sqrt (x + y + 5 / 2) = 3 ∨ Real.sqrt (x + y + 5 / 2) = -3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_y_plus_five_halves_l2509_250929


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2509_250952

theorem quadratic_inequality_equivalence (a b c : ℝ) :
  (a > 0 ∧ b^2 - 4*a*c < 0) ↔ (∀ x : ℝ, a*x^2 + b*x + c > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2509_250952


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2509_250968

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + 2*x*y + 2*y^2 + 3*x - 5*y ≥ -17/2 ∧ 
  ∃ x y : ℝ, x^2 + 2*x*y + 2*y^2 + 3*x - 5*y = -17/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2509_250968


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2509_250940

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the increase in average between two BatsmanPerformance instances -/
def averageIncrease (before after : BatsmanPerformance) : ℚ :=
  after.average - before.average

theorem batsman_average_increase
  (performance16 : BatsmanPerformance)
  (performance17 : BatsmanPerformance)
  (h1 : performance17.innings = performance16.innings + 1)
  (h2 : performance17.innings = 17)
  (h3 : performance17.totalRuns = performance16.totalRuns + 82)
  (h4 : performance17.average = 34)
  : averageIncrease performance16 performance17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l2509_250940


namespace NUMINAMATH_CALUDE_sine_cosine_product_l2509_250932

theorem sine_cosine_product (α : Real) (h : (1 / Real.sin α) + (1 / Real.cos α) = Real.sqrt 3) :
  Real.sin α * Real.cos α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_product_l2509_250932


namespace NUMINAMATH_CALUDE_larger_number_l2509_250963

theorem larger_number (x y : ℕ) (h1 : y = x + 10) (h2 : x + y = 34) : y = 22 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l2509_250963


namespace NUMINAMATH_CALUDE_base6_calculation_l2509_250989

/-- Represents a number in base 6 --/
def Base6 : Type := ℕ

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : Base6 := sorry

/-- Adds two numbers in base 6 --/
def addBase6 (a b : Base6) : Base6 := sorry

/-- Subtracts two numbers in base 6 --/
def subBase6 (a b : Base6) : Base6 := sorry

/-- Theorem: 15₆ - 4₆ + 20₆ = 31₆ in base 6 --/
theorem base6_calculation : 
  let a := toBase6 15
  let b := toBase6 4
  let c := toBase6 20
  let d := toBase6 31
  addBase6 (subBase6 a b) c = d := by sorry

end NUMINAMATH_CALUDE_base6_calculation_l2509_250989


namespace NUMINAMATH_CALUDE_expression_simplification_l2509_250975

theorem expression_simplification (y : ℝ) : 
  4*y + 9*y^2 + 8 - (3 - 4*y - 9*y^2) = 18*y^2 + 8*y + 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2509_250975


namespace NUMINAMATH_CALUDE_screen_to_body_ratio_increases_l2509_250966

theorem screen_to_body_ratio_increases
  (b a m : ℝ)
  (h1 : 0 < b)
  (h2 : b < a)
  (h3 : 0 < m) :
  b / a < (b + m) / (a + m) :=
by sorry

end NUMINAMATH_CALUDE_screen_to_body_ratio_increases_l2509_250966


namespace NUMINAMATH_CALUDE_matrix_N_property_l2509_250974

variable (N : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_N_property (h1 : N.mulVec ![3, -2] = ![4, 1])
                          (h2 : N.mulVec ![-2, 4] = ![0, 2]) :
  N.mulVec ![7, 0] = ![14, 7] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_property_l2509_250974


namespace NUMINAMATH_CALUDE_a_value_satisfies_condition_l2509_250944

-- Define the property that needs to be satisfied
def satisfies_condition (a : ℕ) : Prop :=
  ∀ K : ℤ, K ≠ 27 → (27 - K) ∣ (a - K^1964)

-- State the theorem
theorem a_value_satisfies_condition :
  satisfies_condition (3^5892) :=
sorry

end NUMINAMATH_CALUDE_a_value_satisfies_condition_l2509_250944


namespace NUMINAMATH_CALUDE_betty_age_l2509_250967

theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 14) :
  betty = 7 := by
sorry

end NUMINAMATH_CALUDE_betty_age_l2509_250967


namespace NUMINAMATH_CALUDE_odd_as_difference_of_squares_l2509_250914

theorem odd_as_difference_of_squares :
  ∀ n : ℤ, Odd n → ∃ a b : ℤ, n = a^2 - b^2 :=
by sorry

end NUMINAMATH_CALUDE_odd_as_difference_of_squares_l2509_250914


namespace NUMINAMATH_CALUDE_max_discarded_grapes_proof_l2509_250951

/-- The number of children among whom the grapes are to be distributed. -/
def num_children : ℕ := 8

/-- The maximum number of grapes that could be discarded. -/
def max_discarded_grapes : ℕ := num_children - 1

/-- Theorem stating that the maximum number of discarded grapes is one less than the number of children. -/
theorem max_discarded_grapes_proof :
  max_discarded_grapes = num_children - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_discarded_grapes_proof_l2509_250951


namespace NUMINAMATH_CALUDE_max_pieces_from_cake_l2509_250906

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 21

/-- The size of the small cake pieces in inches -/
def small_piece_size : ℕ := 3

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := (large_cake_size * large_cake_size) / (small_piece_size * small_piece_size)

theorem max_pieces_from_cake : max_pieces = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_from_cake_l2509_250906


namespace NUMINAMATH_CALUDE_small_to_large_triangle_area_ratio_l2509_250900

/-- The ratio of the total area of three small equilateral triangles to the area of one large equilateral triangle with the same perimeter -/
theorem small_to_large_triangle_area_ratio :
  let small_side : ℝ := 2
  let small_perimeter : ℝ := 3 * small_side
  let total_perimeter : ℝ := 3 * small_perimeter
  let large_side : ℝ := total_perimeter / 3
  let small_area : ℝ := (Real.sqrt 3 / 4) * small_side ^ 2
  let large_area : ℝ := (Real.sqrt 3 / 4) * large_side ^ 2
  (3 * small_area) / large_area = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_small_to_large_triangle_area_ratio_l2509_250900


namespace NUMINAMATH_CALUDE_cookies_bought_l2509_250938

theorem cookies_bought (total_groceries cake_packs : ℕ) 
  (h1 : total_groceries = 14)
  (h2 : cake_packs = 12)
  (h3 : ∃ cookie_packs : ℕ, cookie_packs + cake_packs = total_groceries) :
  ∃ cookie_packs : ℕ, cookie_packs = 2 ∧ cookie_packs + cake_packs = total_groceries :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_bought_l2509_250938


namespace NUMINAMATH_CALUDE_weight_calculation_l2509_250937

theorem weight_calculation (a b c : ℝ) (h : (a + b + c + (a + b) + (b + c) + (c + a) + (a + b + c)) / 7 = 95.42857142857143) : 
  a + b + c = 222.66666666666666 := by
  sorry

end NUMINAMATH_CALUDE_weight_calculation_l2509_250937


namespace NUMINAMATH_CALUDE_g_expression_l2509_250959

-- Define polynomials f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom poly_sum : ∀ x, f x + g x = 2 * x^2 + 3 * x + 4
axiom f_def : ∀ x, f x = 2 * x^3 - x^2 - 4 * x + 5

-- State the theorem
theorem g_expression : ∀ x, g x = -2 * x^3 + 3 * x^2 + 7 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l2509_250959


namespace NUMINAMATH_CALUDE_smallest_k_with_remainders_l2509_250947

theorem smallest_k_with_remainders (k : ℕ) : 
  k > 1 ∧ 
  k % 13 = 1 ∧ 
  k % 8 = 1 ∧ 
  k % 4 = 1 ∧
  (∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 8 = 1 ∧ m % 4 = 1 → k ≤ m) →
  k = 105 := by
sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainders_l2509_250947


namespace NUMINAMATH_CALUDE_alloy_composition_theorem_l2509_250996

/-- Represents the composition of an alloy -/
structure AlloyComposition where
  copper : ℝ
  tin : ℝ
  zinc : ℝ
  sum_to_one : copper + tin + zinc = 1

/-- The conditions given in the problem -/
def satisfies_conditions (c : AlloyComposition) : Prop :=
  c.copper - c.tin = 1/10 ∧ c.tin - c.zinc = 3/10

/-- The theorem to be proved -/
theorem alloy_composition_theorem :
  ∃ (c : AlloyComposition),
    satisfies_conditions c ∧
    c.copper = 0.5 ∧ c.tin = 0.4 ∧ c.zinc = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_alloy_composition_theorem_l2509_250996


namespace NUMINAMATH_CALUDE_plane_perp_from_line_relations_l2509_250936

/-- Two lines are parallel -/
def parallel_lines (m n : Line) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (m : Line) (α : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def perp_planes (α β : Plane) : Prop := sorry

/-- Main theorem: If m is parallel to n, m is contained in α, and n is perpendicular to β, then α is perpendicular to β -/
theorem plane_perp_from_line_relations 
  (m n : Line) (α β : Plane) 
  (h1 : parallel_lines m n) 
  (h2 : line_in_plane m α) 
  (h3 : line_perp_plane n β) : 
  perp_planes α β := by sorry

end NUMINAMATH_CALUDE_plane_perp_from_line_relations_l2509_250936


namespace NUMINAMATH_CALUDE_marble_distribution_l2509_250912

/-- Given the distribution of marbles between two classes, prove the difference between
    the number of marbles each male in Class 2 receives and the total number of marbles
    taken by Class 1. -/
theorem marble_distribution (total_marbles : ℕ) (class1_marbles : ℕ) (class2_marbles : ℕ)
  (boys_marbles : ℕ) (girls_marbles : ℕ) (num_boys : ℕ) :
  total_marbles = 1000 →
  class1_marbles = class2_marbles + 50 →
  class1_marbles + class2_marbles = total_marbles →
  boys_marbles = girls_marbles + 35 →
  boys_marbles + girls_marbles = class2_marbles →
  num_boys = 17 →
  class1_marbles - (boys_marbles / num_boys) = 510 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l2509_250912


namespace NUMINAMATH_CALUDE_max_ratio_squared_max_ratio_squared_achieved_l2509_250956

theorem max_ratio_squared (a b x y : ℝ) : 
  0 < a → 0 < b → a ≥ b → 
  0 ≤ x → x < a → 0 ≤ y → y < b →
  a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b + y)^2 →
  (a / b)^2 ≤ 2 :=
by sorry

theorem max_ratio_squared_achieved (a b : ℝ) :
  ∃ x y : ℝ, 0 < a → 0 < b → a ≥ b → 
  0 ≤ x → x < a → 0 ≤ y → y < b →
  a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b + y)^2 →
  (a / b)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_squared_max_ratio_squared_achieved_l2509_250956


namespace NUMINAMATH_CALUDE_returning_players_l2509_250923

theorem returning_players (new_players : ℕ) (total_groups : ℕ) (players_per_group : ℕ) : 
  new_players = 48 → total_groups = 9 → players_per_group = 6 →
  total_groups * players_per_group - new_players = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_returning_players_l2509_250923


namespace NUMINAMATH_CALUDE_total_amount_calculation_l2509_250979

/-- Calculate the total amount paid for a suit, shoes, dress shirt, and tie, considering discounts and taxes. -/
theorem total_amount_calculation (suit_price suit_discount suit_tax_rate : ℚ)
                                 (shoes_price shoes_discount shoes_tax_rate : ℚ)
                                 (shirt_price shirt_tax_rate : ℚ)
                                 (tie_price tie_tax_rate : ℚ)
                                 (shirt_tie_discount_rate : ℚ) :
  suit_price = 430 →
  suit_discount = 100 →
  suit_tax_rate = 5/100 →
  shoes_price = 190 →
  shoes_discount = 30 →
  shoes_tax_rate = 7/100 →
  shirt_price = 80 →
  shirt_tax_rate = 6/100 →
  tie_price = 50 →
  tie_tax_rate = 4/100 →
  shirt_tie_discount_rate = 20/100 →
  ∃ total_amount : ℚ,
    total_amount = (suit_price - suit_discount) * (1 + suit_tax_rate) +
                   (shoes_price - shoes_discount) * (1 + shoes_tax_rate) +
                   ((shirt_price + tie_price) * (1 - shirt_tie_discount_rate)) * 
                   ((shirt_price / (shirt_price + tie_price)) * (1 + shirt_tax_rate) +
                    (tie_price / (shirt_price + tie_price)) * (1 + tie_tax_rate)) ∧
    total_amount = 627.14 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l2509_250979


namespace NUMINAMATH_CALUDE_min_rectangle_dimensions_l2509_250904

/-- A rectangle with length twice its width and area at least 500 square feet has minimum dimensions of width = 5√10 feet and length = 10√10 feet. -/
theorem min_rectangle_dimensions (w : ℝ) (h : w > 0) :
  (2 * w ^ 2 ≥ 500) → (∀ x > 0, 2 * x ^ 2 ≥ 500 → w ≤ x) → w = 5 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_min_rectangle_dimensions_l2509_250904


namespace NUMINAMATH_CALUDE_square_not_always_positive_l2509_250973

theorem square_not_always_positive : ∃ (a : ℝ), ¬(a^2 > 0) := by sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l2509_250973


namespace NUMINAMATH_CALUDE_total_crayons_l2509_250908

/-- The total number of crayons given the specified box counts and colors -/
theorem total_crayons : 
  let orange_boxes : ℕ := 6
  let orange_per_box : ℕ := 8
  let blue_boxes : ℕ := 7
  let blue_per_box : ℕ := 5
  let red_boxes : ℕ := 1
  let red_per_box : ℕ := 11
  orange_boxes * orange_per_box + blue_boxes * blue_per_box + red_boxes * red_per_box = 94 :=
by sorry

end NUMINAMATH_CALUDE_total_crayons_l2509_250908


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l2509_250934

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_k_value
  (a : ℕ → ℝ) (d : ℝ) (k : ℕ) 
  (h1 : d ≠ 0)
  (h2 : arithmetic_sequence a d)
  (h3 : a 1 = 4 * d)
  (h4 : a k ^ 2 = a 1 * a (2 * k)) :
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l2509_250934


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2509_250928

/-- Given a line L1 with equation x - y + 1 = 0 and a point P (2, -4),
    prove that the line L2 passing through P and parallel to L1
    has the equation x - y - 6 = 0 -/
theorem parallel_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => x - y + 1 = 0
  let P : ℝ × ℝ := (2, -4)
  let L2 : ℝ → ℝ → Prop := λ x y => x - y - 6 = 0
  (∀ x y, L2 x y ↔ (x - y = 6)) ∧
  L2 P.1 P.2 ∧
  (∀ x1 y1 x2 y2, L1 x1 y1 ∧ L2 x2 y2 → x1 - y1 = x2 - y2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2509_250928


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l2509_250911

/-- Represents the capital contribution and time invested by a partner --/
structure Investment where
  capital : ℕ
  months : ℕ

/-- Calculates the capital-months for an investment --/
def capitalMonths (inv : Investment) : ℕ := inv.capital * inv.months

theorem profit_sharing_ratio 
  (a_investment : Investment) 
  (b_investment : Investment) 
  (h1 : a_investment.capital = 3500) 
  (h2 : a_investment.months = 12) 
  (h3 : b_investment.capital = 10500) 
  (h4 : b_investment.months = 6) :
  (capitalMonths a_investment) / (capitalMonths a_investment).gcd (capitalMonths b_investment) = 2 ∧ 
  (capitalMonths b_investment) / (capitalMonths a_investment).gcd (capitalMonths b_investment) = 3 :=
sorry

end NUMINAMATH_CALUDE_profit_sharing_ratio_l2509_250911


namespace NUMINAMATH_CALUDE_work_distribution_l2509_250917

theorem work_distribution (total_work : ℝ) (h1 : total_work > 0) : 
  let top_20_percent_work := 0.8 * total_work
  let remaining_work := total_work - top_20_percent_work
  let next_20_percent_work := 0.25 * remaining_work
  ∃ (work_40_percent : ℝ), work_40_percent ≥ top_20_percent_work + next_20_percent_work ∧ 
                            work_40_percent / total_work ≥ 0.85 := by
  sorry

end NUMINAMATH_CALUDE_work_distribution_l2509_250917


namespace NUMINAMATH_CALUDE_triangle_side_length_l2509_250960

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 60 * π / 180 →  -- Convert 60° to radians
  c = Real.sqrt 3 →
  b = 2 * Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2509_250960


namespace NUMINAMATH_CALUDE_subtract_like_terms_l2509_250948

theorem subtract_like_terms (a : ℝ) : 2 * a - 3 * a = -a := by
  sorry

end NUMINAMATH_CALUDE_subtract_like_terms_l2509_250948


namespace NUMINAMATH_CALUDE_combined_swim_time_l2509_250933

def freestyle_time : ℕ := 48
def backstroke_time : ℕ := freestyle_time + 4
def butterfly_time : ℕ := backstroke_time + 3
def breaststroke_time : ℕ := butterfly_time + 2

theorem combined_swim_time : 
  freestyle_time + backstroke_time + butterfly_time + breaststroke_time = 212 := by
  sorry

end NUMINAMATH_CALUDE_combined_swim_time_l2509_250933


namespace NUMINAMATH_CALUDE_garden_comparison_l2509_250990

-- Define the dimensions of Karl's garden
def karl_length : ℕ := 30
def karl_width : ℕ := 40

-- Define the dimensions of Makenna's garden
def makenna_side : ℕ := 35

-- Theorem to prove the comparison of areas and perimeters
theorem garden_comparison :
  (makenna_side * makenna_side - karl_length * karl_width = 25) ∧
  (2 * (karl_length + karl_width) = 4 * makenna_side) :=
by sorry

end NUMINAMATH_CALUDE_garden_comparison_l2509_250990


namespace NUMINAMATH_CALUDE_four_heads_in_five_tosses_l2509_250999

/-- The probability of getting exactly k successes in n trials with probability p of success in each trial. -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem: The probability of getting exactly 4 heads in 5 tosses of a fair coin is 0.15625. -/
theorem four_heads_in_five_tosses :
  binomialProbability 5 4 (1/2) = 0.15625 := by
sorry

end NUMINAMATH_CALUDE_four_heads_in_five_tosses_l2509_250999


namespace NUMINAMATH_CALUDE_chord_line_equation_l2509_250910

/-- Given an ellipse and a chord midpoint, prove the equation of the line containing the chord -/
theorem chord_line_equation (x y : ℝ) :
  (x^2 / 16 + y^2 / 9 = 1) →  -- Ellipse equation
  (∃ (x1 y1 x2 y2 : ℝ),       -- Existence of chord endpoints
    x1^2 / 16 + y1^2 / 9 = 1 ∧
    x2^2 / 16 + y2^2 / 9 = 1 ∧
    (x1 + x2) / 2 = 2 ∧       -- Midpoint x-coordinate
    (y1 + y2) / 2 = 3/2) →    -- Midpoint y-coordinate
  (∃ (a b c : ℝ),             -- Existence of line equation
    a*x + b*y + c = 0 ∧       -- General form of line equation
    a = 3 ∧ b = 4 ∧ c = -12)  -- Specific coefficients
  := by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l2509_250910


namespace NUMINAMATH_CALUDE_lana_muffin_sales_l2509_250925

/-- Lana's muffin sales problem -/
theorem lana_muffin_sales (goal : ℕ) (morning_sales : ℕ) (afternoon_sales : ℕ)
  (h1 : goal = 20)
  (h2 : morning_sales = 12)
  (h3 : afternoon_sales = 4) :
  goal - morning_sales - afternoon_sales = 4 := by
  sorry

end NUMINAMATH_CALUDE_lana_muffin_sales_l2509_250925


namespace NUMINAMATH_CALUDE_xy_neq_6_sufficient_not_necessary_l2509_250930

theorem xy_neq_6_sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x * y ≠ 6 → (x ≠ 2 ∨ y ≠ 3)) ∧
  (∃ x y, (x ≠ 2 ∨ y ≠ 3) ∧ x * y = 6) :=
sorry

end NUMINAMATH_CALUDE_xy_neq_6_sufficient_not_necessary_l2509_250930


namespace NUMINAMATH_CALUDE_irrational_among_given_numbers_l2509_250991

theorem irrational_among_given_numbers : 
  (∃ q : ℚ, |3 / (-8)| = q) ∧ 
  (∃ q : ℚ, |22 / 7| = q) ∧ 
  (∃ q : ℚ, 3.14 = q) ∧ 
  (∀ q : ℚ, |Real.sqrt 3| ≠ q) := by
  sorry

end NUMINAMATH_CALUDE_irrational_among_given_numbers_l2509_250991


namespace NUMINAMATH_CALUDE_sum_four_digit_ending_zero_value_l2509_250992

/-- The sum of all four-digit positive integers ending in 0 -/
def sum_four_digit_ending_zero : ℕ :=
  let first_term := 1000
  let last_term := 9990
  let common_difference := 10
  let num_terms := (last_term - first_term) / common_difference + 1
  num_terms * (first_term + last_term) / 2

theorem sum_four_digit_ending_zero_value : 
  sum_four_digit_ending_zero = 4945500 := by
  sorry

end NUMINAMATH_CALUDE_sum_four_digit_ending_zero_value_l2509_250992


namespace NUMINAMATH_CALUDE_isosceles_triangles_not_necessarily_congruent_l2509_250958

/-- An isosceles triangle with acute angles -/
structure AcuteIsoscelesTriangle where
  /-- The length of the equal sides (legs) of the triangle -/
  legLength : ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The angle at the apex of the triangle (in radians) -/
  apexAngle : ℝ
  /-- The apex angle is acute -/
  acuteAngle : apexAngle < Real.pi / 4
  /-- The leg length is positive -/
  legPositive : legLength > 0
  /-- The inradius is positive -/
  inradiusPositive : inradius > 0

/-- The theorem stating that two isosceles triangles with the same leg length and inradius
    are not necessarily congruent -/
theorem isosceles_triangles_not_necessarily_congruent :
  ∃ (t1 t2 : AcuteIsoscelesTriangle),
    t1.legLength = t2.legLength ∧
    t1.inradius = t2.inradius ∧
    t1.apexAngle ≠ t2.apexAngle :=
  sorry

end NUMINAMATH_CALUDE_isosceles_triangles_not_necessarily_congruent_l2509_250958


namespace NUMINAMATH_CALUDE_xy_value_l2509_250950

theorem xy_value (x y : ℝ) (h : |3*x + y - 2| + (2*x + 3*y + 1)^2 = 0) : x * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2509_250950


namespace NUMINAMATH_CALUDE_average_first_16_even_numbers_l2509_250998

theorem average_first_16_even_numbers :
  let first_even : ℕ := 2
  let last_even : ℕ := 32
  let count : ℕ := 16
  (first_even + last_even) / 2 = 17 :=
by sorry

end NUMINAMATH_CALUDE_average_first_16_even_numbers_l2509_250998


namespace NUMINAMATH_CALUDE_equation_solution_l2509_250920

theorem equation_solution : ∃! x : ℝ, (3 : ℝ) / (x - 3) = (4 : ℝ) / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2509_250920


namespace NUMINAMATH_CALUDE_retirement_total_is_70_l2509_250922

/-- The required total of age and years of employment for retirement -/
def retirement_total : ℕ := 70

/-- The year the employee was hired -/
def hire_year : ℕ := 1988

/-- The employee's age when hired -/
def hire_age : ℕ := 32

/-- The year the employee becomes eligible for retirement -/
def retirement_year : ℕ := 2007

theorem retirement_total_is_70 :
  retirement_total = 
    (retirement_year - hire_year) + -- Years of employment
    (retirement_year - hire_year + hire_age) -- Age at retirement
  := by sorry

end NUMINAMATH_CALUDE_retirement_total_is_70_l2509_250922


namespace NUMINAMATH_CALUDE_min_common_edges_l2509_250916

/-- Represents a closed route on a grid -/
def ClosedRoute (n : ℕ) := List (Fin n × Fin n)

/-- The number of edges in an n×n grid -/
def gridEdges (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- The number of edges traversed by a closed route visiting all vertices once -/
def routeEdges (n : ℕ) : ℕ := n * n

theorem min_common_edges (route1 route2 : ClosedRoute 8) :
  (∀ v : Fin 8 × Fin 8, v ∈ route1.toFinset ∧ v ∈ route2.toFinset) →
  (∀ v : Fin 8 × Fin 8, (route1.count v = 1 ∧ route2.count v = 1)) →
  (∃ m : ℕ, m = 16 ∧ m = routeEdges 8 + routeEdges 8 - gridEdges 7) := by
  sorry

end NUMINAMATH_CALUDE_min_common_edges_l2509_250916


namespace NUMINAMATH_CALUDE_carpet_needed_proof_l2509_250915

/-- Given a room with length and width, and an amount of existing carpet,
    calculate the additional carpet needed to cover the whole floor. -/
def additional_carpet_needed (length width existing_carpet : ℝ) : ℝ :=
  length * width - existing_carpet

/-- Proof that for a room of 4 feet by 20 feet with 18 square feet of existing carpet,
    62 square feet of additional carpet is needed. -/
theorem carpet_needed_proof :
  additional_carpet_needed 4 20 18 = 62 := by
  sorry

#eval additional_carpet_needed 4 20 18

end NUMINAMATH_CALUDE_carpet_needed_proof_l2509_250915


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2509_250943

theorem hemisphere_surface_area (r : ℝ) (h : r = 5) :
  let sphere_area (r : ℝ) := 4 * π * r^2
  let hemisphere_curved_area (r : ℝ) := (sphere_area r) / 2
  let base_area (r : ℝ) := π * r^2
  hemisphere_curved_area r + base_area r = 75 * π := by
sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2509_250943


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l2509_250986

/-- A seven-digit number in the form 8n46325 where n is a single digit -/
def sevenDigitNumber (n : ℕ) : ℕ := 8000000 + 1000000*n + 46325

/-- Predicate to check if a natural number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := n < 10

theorem divisible_by_eleven (n : ℕ) : 
  isSingleDigit n → (sevenDigitNumber n) % 11 = 0 → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l2509_250986


namespace NUMINAMATH_CALUDE_binomial_10_2_l2509_250913

theorem binomial_10_2 : Nat.choose 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_2_l2509_250913


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2509_250993

-- Define the conditions
def p (x : ℝ) : Prop := x^2 - 4*x - 5 ≤ 0
def q (x a : ℝ) : Prop := |x - 3| < a ∧ a > 0

-- Define the solution set of p
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | 3 - a < x ∧ x < 3 + a}

-- Theorem statement
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x) → a > 4 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2509_250993


namespace NUMINAMATH_CALUDE_arc_length_for_given_radius_and_angle_l2509_250983

/-- Given an arc with radius 24 and central angle 60°, its length is 8π. -/
theorem arc_length_for_given_radius_and_angle :
  let r : ℝ := 24
  let θ : ℝ := 60
  let l : ℝ := (θ / 360) * (2 * π * r)
  l = 8 * π := by sorry

end NUMINAMATH_CALUDE_arc_length_for_given_radius_and_angle_l2509_250983


namespace NUMINAMATH_CALUDE_expression_increase_l2509_250987

theorem expression_increase (x y : ℝ) : 
  let original := 3 * x^2 * y
  let new_x := 1.2 * x
  let new_y := 2.4 * y
  let new_expression := 3 * new_x^2 * new_y
  new_expression = 3.456 * original := by
sorry

end NUMINAMATH_CALUDE_expression_increase_l2509_250987


namespace NUMINAMATH_CALUDE_car_speed_comparison_l2509_250961

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  (2 * u * v) / (u + v) ≤ (u + v) / 2 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l2509_250961


namespace NUMINAMATH_CALUDE_conference_room_arrangements_l2509_250965

/-- The number of distinct arrangements of seats in a conference room -/
theorem conference_room_arrangements (n m : ℕ) (hn : n = 12) (hm : m = 4) :
  (Nat.choose n m) = 495 := by sorry

end NUMINAMATH_CALUDE_conference_room_arrangements_l2509_250965


namespace NUMINAMATH_CALUDE_prob_different_cards_proof_l2509_250982

/-- The probability of drawing two cards with different numbers from a set of 10 cards (numbered 0 to 9) with replacement -/
def prob_different_cards : ℚ := 9 / 10

/-- The number of cards in the set -/
def num_cards : ℕ := 10

theorem prob_different_cards_proof :
  prob_different_cards = (num_cards - 1) / num_cards :=
by sorry

end NUMINAMATH_CALUDE_prob_different_cards_proof_l2509_250982


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2509_250919

theorem inequality_equivalence (x : ℝ) : 
  (6 * x - 2 < (x + 1)^2 ∧ (x + 1)^2 < 8 * x - 4) ↔ (3 < x ∧ x < 5) := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2509_250919


namespace NUMINAMATH_CALUDE_vector_b_magnitude_l2509_250901

def a : ℝ × ℝ := (1, -2)

theorem vector_b_magnitude (b : ℝ × ℝ) (h : 2 • a - b = (-1, 0)) : 
  ‖b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_b_magnitude_l2509_250901


namespace NUMINAMATH_CALUDE_unanswered_questions_count_l2509_250978

/-- Represents the test scenario with given conditions -/
structure TestScenario where
  total_questions : ℕ
  first_set_questions : ℕ
  second_set_questions : ℕ
  third_set_questions : ℕ
  first_set_time : ℕ  -- in minutes
  second_set_time : ℕ  -- in seconds
  third_set_time : ℕ  -- in minutes
  total_time : ℕ  -- in hours

/-- Calculates the number of unanswered questions in the given test scenario -/
def unanswered_questions (scenario : TestScenario) : ℕ :=
  scenario.total_questions - (scenario.first_set_questions + scenario.second_set_questions + scenario.third_set_questions)

/-- Theorem stating that for the given test scenario, the number of unanswered questions is 75 -/
theorem unanswered_questions_count (scenario : TestScenario) 
  (h1 : scenario.total_questions = 200)
  (h2 : scenario.first_set_questions = 50)
  (h3 : scenario.second_set_questions = 50)
  (h4 : scenario.third_set_questions = 25)
  (h5 : scenario.first_set_time = 1)
  (h6 : scenario.second_set_time = 90)
  (h7 : scenario.third_set_time = 2)
  (h8 : scenario.total_time = 4) :
  unanswered_questions scenario = 75 := by
  sorry

#eval unanswered_questions {
  total_questions := 200,
  first_set_questions := 50,
  second_set_questions := 50,
  third_set_questions := 25,
  first_set_time := 1,
  second_set_time := 90,
  third_set_time := 2,
  total_time := 4
}

end NUMINAMATH_CALUDE_unanswered_questions_count_l2509_250978


namespace NUMINAMATH_CALUDE_maryville_population_increase_l2509_250905

/-- The average annual population increase in Maryville between 2000 and 2005 -/
def average_annual_increase (pop_2000 pop_2005 : ℕ) : ℚ :=
  (pop_2005 - pop_2000 : ℚ) / 5

/-- Theorem stating that the average annual population increase in Maryville between 2000 and 2005 is 3400 -/
theorem maryville_population_increase :
  average_annual_increase 450000 467000 = 3400 := by
  sorry

end NUMINAMATH_CALUDE_maryville_population_increase_l2509_250905


namespace NUMINAMATH_CALUDE_complex_coordinates_of_i_times_one_minus_i_l2509_250931

theorem complex_coordinates_of_i_times_one_minus_i :
  let i : ℂ := Complex.I
  (i * (1 - i)).re = 1 ∧ (i * (1 - i)).im = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_coordinates_of_i_times_one_minus_i_l2509_250931
