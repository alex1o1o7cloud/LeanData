import Mathlib

namespace NUMINAMATH_CALUDE_camel_division_theorem_l2911_291106

/-- A representation of the "camel" figure --/
structure CamelFigure where
  area : ℕ
  has_spaced_cells : Bool

/-- Represents a division of the figure --/
inductive Division
  | GridLines
  | Arbitrary

/-- Represents the result of attempting to form a square --/
inductive SquareFormation
  | Possible
  | Impossible

/-- Function to determine if a square can be formed from the division --/
def can_form_square (figure : CamelFigure) (division : Division) : SquareFormation :=
  match division with
  | Division.GridLines => 
      if figure.has_spaced_cells then SquareFormation.Impossible else SquareFormation.Possible
  | Division.Arbitrary => 
      if figure.area == 25 then SquareFormation.Possible else SquareFormation.Impossible

/-- The main theorem about the camel figure --/
theorem camel_division_theorem (camel : CamelFigure) 
    (h1 : camel.area = 25) 
    (h2 : camel.has_spaced_cells = true) : 
    (can_form_square camel Division.GridLines = SquareFormation.Impossible) ∧ 
    (can_form_square camel Division.Arbitrary = SquareFormation.Possible) := by
  sorry

end NUMINAMATH_CALUDE_camel_division_theorem_l2911_291106


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l2911_291163

-- Define a quadratic polynomial with integer coefficients
def QuadraticPolynomial (a b c : ℤ) : ℝ → ℝ := fun x ↦ (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

theorem quadratic_polynomial_property (a b c : ℤ) :
  let f := QuadraticPolynomial a b c
  (f (Real.sqrt 3) - f (Real.sqrt 2) = 4) →
  (f (Real.sqrt 10) - f (Real.sqrt 7) = 12) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l2911_291163


namespace NUMINAMATH_CALUDE_percent_decrease_proof_l2911_291151

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 30) : 
  (original_price - sale_price) / original_price * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_proof_l2911_291151


namespace NUMINAMATH_CALUDE_specific_garden_area_l2911_291185

/-- Represents a circular garden with a path through it. -/
structure GardenWithPath where
  diameter : ℝ
  pathWidth : ℝ

/-- Calculates the remaining area of the garden not covered by the path. -/
def remainingArea (g : GardenWithPath) : ℝ :=
  sorry

/-- Theorem stating the remaining area for a specific garden configuration. -/
theorem specific_garden_area :
  let g : GardenWithPath := { diameter := 14, pathWidth := 4 }
  remainingArea g = 29 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_specific_garden_area_l2911_291185


namespace NUMINAMATH_CALUDE_max_teams_advancing_l2911_291176

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  num_teams : Nat
  min_points_to_advance : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- Calculate the total number of games in the tournament -/
def total_games (t : FootballTournament) : Nat :=
  t.num_teams * (t.num_teams - 1) / 2

/-- Calculate the maximum total points that can be distributed in the tournament -/
def max_total_points (t : FootballTournament) : Nat :=
  (total_games t) * t.points_for_win

/-- Theorem stating the maximum number of teams that can advance -/
theorem max_teams_advancing (t : FootballTournament) 
  (h1 : t.num_teams = 6)
  (h2 : t.min_points_to_advance = 12)
  (h3 : t.points_for_win = 3)
  (h4 : t.points_for_draw = 1)
  (h5 : t.points_for_loss = 0) :
  ∃ (n : Nat), n ≤ 3 ∧ 
    n * t.min_points_to_advance ≤ max_total_points t ∧
    ∀ (m : Nat), m * t.min_points_to_advance ≤ max_total_points t → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_max_teams_advancing_l2911_291176


namespace NUMINAMATH_CALUDE_max_two_digit_times_max_one_digit_is_three_digit_l2911_291165

theorem max_two_digit_times_max_one_digit_is_three_digit : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = 99 * 9 := by
  sorry

end NUMINAMATH_CALUDE_max_two_digit_times_max_one_digit_is_three_digit_l2911_291165


namespace NUMINAMATH_CALUDE_f_properties_l2911_291136

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + Real.pi/2) * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3/4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≤ 1/4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → f x ≥ -1/2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = 1/4) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) ∧ f x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2911_291136


namespace NUMINAMATH_CALUDE_system_equations_proof_l2911_291179

theorem system_equations_proof (a x y : ℝ) : 
  x + y = -7 - a → 
  x - y = 1 + 3*a → 
  x ≤ 0 → 
  y < 0 → 
  (-2 < a ∧ a ≤ 3) ∧ 
  (abs (a - 3) + abs (a + 2) = 5) ∧ 
  (∀ (a : ℤ), -2 < a ∧ a ≤ 3 → (∀ x, 2*a*x + x > 2*a + 1 ↔ x < 1) ↔ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_equations_proof_l2911_291179


namespace NUMINAMATH_CALUDE_museum_trip_total_l2911_291169

theorem museum_trip_total (first_bus second_bus third_bus fourth_bus : ℕ) : 
  first_bus = 12 →
  second_bus = 2 * first_bus →
  third_bus = second_bus - 6 →
  fourth_bus = first_bus + 9 →
  first_bus + second_bus + third_bus + fourth_bus = 75 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_total_l2911_291169


namespace NUMINAMATH_CALUDE_segment_ratio_l2911_291146

/-- Given five consecutive points on a line, prove that the ratio of two specific segments is 2:1 -/
theorem segment_ratio (a b c d e : ℝ) : 
  (b < c) ∧ (c < d) ∧  -- Consecutive points
  (d - e = 4) ∧        -- de = 4
  (a - b = 5) ∧        -- ab = 5
  (a - c = 11) ∧       -- ac = 11
  (a - e = 18) →       -- ae = 18
  (c - b) / (d - c) = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_segment_ratio_l2911_291146


namespace NUMINAMATH_CALUDE_sum_of_digits_999_base_7_l2911_291183

def base_7_representation (n : ℕ) : List ℕ :=
  sorry

def sum_of_digits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_999_base_7 :
  sum_of_digits (base_7_representation 999) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_999_base_7_l2911_291183


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2911_291100

theorem quadratic_root_implies_m_value (m : ℝ) :
  (3^2 - 2*3 + m = 0) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2911_291100


namespace NUMINAMATH_CALUDE_tan_theta_value_l2911_291117

theorem tan_theta_value (θ : Real) : 
  Real.tan (π / 4 + θ) = 1 / 2 → Real.tan θ = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l2911_291117


namespace NUMINAMATH_CALUDE_johanna_turtle_loss_l2911_291148

/-- The fraction of turtles Johanna loses -/
def johanna_loss_fraction (owen_initial : ℕ) (johanna_diff : ℕ) (owen_final : ℕ) : ℚ :=
  let owen_after_month := 2 * owen_initial
  let johanna_initial := owen_initial - johanna_diff
  1 - (owen_final - owen_after_month) / johanna_initial

theorem johanna_turtle_loss 
  (owen_initial : ℕ) 
  (johanna_diff : ℕ) 
  (owen_final : ℕ) 
  (h1 : owen_initial = 21)
  (h2 : johanna_diff = 5)
  (h3 : owen_final = 50) :
  johanna_loss_fraction owen_initial johanna_diff owen_final = 1/2 := by
  sorry

#eval johanna_loss_fraction 21 5 50

end NUMINAMATH_CALUDE_johanna_turtle_loss_l2911_291148


namespace NUMINAMATH_CALUDE_max_profit_zongzi_l2911_291156

/-- Represents the cost and selling prices of zongzi types A and B -/
structure ZongziPrices where
  cost_a : ℚ
  cost_b : ℚ
  sell_a : ℚ
  sell_b : ℚ

/-- Represents the purchase quantities of zongzi types A and B -/
structure ZongziQuantities where
  qty_a : ℕ
  qty_b : ℕ

/-- Calculates the profit given prices and quantities -/
def profit (p : ZongziPrices) (q : ZongziQuantities) : ℚ :=
  (p.sell_a - p.cost_a) * q.qty_a + (p.sell_b - p.cost_b) * q.qty_b

/-- Theorem stating the maximum profit achievable under given conditions -/
theorem max_profit_zongzi (p : ZongziPrices) (q : ZongziQuantities) :
  p.cost_b = p.cost_a + 2 →
  1000 / p.cost_a = 1200 / p.cost_b →
  p.sell_a = 12 →
  p.sell_b = 15 →
  q.qty_a + q.qty_b = 200 →
  q.qty_a ≥ 2 * q.qty_b →
  ∃ (max_q : ZongziQuantities),
    max_q.qty_a = 134 ∧
    max_q.qty_b = 66 ∧
    ∀ (other_q : ZongziQuantities),
      other_q.qty_a + other_q.qty_b = 200 →
      other_q.qty_a ≥ 2 * other_q.qty_b →
      profit p max_q ≥ profit p other_q :=
sorry

end NUMINAMATH_CALUDE_max_profit_zongzi_l2911_291156


namespace NUMINAMATH_CALUDE_tenth_row_white_squares_l2911_291180

/-- Represents the number of squares in the nth row of a stair-step figure -/
def totalSquares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of white squares in the nth row of a stair-step figure -/
def whiteSquares (n : ℕ) : ℕ := (totalSquares n) / 2

theorem tenth_row_white_squares :
  whiteSquares 10 = 9 := by sorry

end NUMINAMATH_CALUDE_tenth_row_white_squares_l2911_291180


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2911_291102

theorem train_speed_calculation (v : ℝ) : 
  v > 0 → -- The speed is positive
  (v + 42) * (5 / 18) * 9 = 280 → -- Equation derived from the problem
  v = 70 := by
sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2911_291102


namespace NUMINAMATH_CALUDE_vector_sum_norm_equality_implies_parallel_l2911_291177

/-- Given two non-zero vectors a and b, if |a + b| = |a| - |b|, then a and b are parallel -/
theorem vector_sum_norm_equality_implies_parallel
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ‖a + b‖ = ‖a‖ - ‖b‖) :
  ∃ (k : ℝ), a = k • b :=
sorry

end NUMINAMATH_CALUDE_vector_sum_norm_equality_implies_parallel_l2911_291177


namespace NUMINAMATH_CALUDE_percent_both_correct_l2911_291168

theorem percent_both_correct
  (percent_first : ℝ)
  (percent_second : ℝ)
  (percent_neither : ℝ)
  (h1 : percent_first = 75)
  (h2 : percent_second = 25)
  (h3 : percent_neither = 20)
  : ℝ :=
by
  -- Define the percentage of students who answered both questions correctly
  let percent_both : ℝ := percent_first + percent_second - (100 - percent_neither)
  
  -- Prove that percent_both equals 20
  have : percent_both = 20 := by sorry
  
  -- Return the result
  exact percent_both

end NUMINAMATH_CALUDE_percent_both_correct_l2911_291168


namespace NUMINAMATH_CALUDE_days_without_calls_l2911_291186

-- Define the number of days in the year
def total_days : ℕ := 365

-- Define the periods of the calls
def period1 : ℕ := 3
def period2 : ℕ := 4
def period3 : ℕ := 5

-- Function to calculate the number of days with at least one call
def days_with_calls : ℕ :=
  (total_days / period1) +
  (total_days / period2) +
  (total_days / period3) -
  (total_days / (period1 * period2)) -
  (total_days / (period2 * period3)) -
  (total_days / (period1 * period3)) +
  (total_days / (period1 * period2 * period3))

-- Theorem to prove
theorem days_without_calls :
  total_days - days_with_calls = 146 :=
by sorry

end NUMINAMATH_CALUDE_days_without_calls_l2911_291186


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_points_sum_reciprocal_bounds_l2911_291162

/-- The maximum and minimum values of the sum of reciprocals of distances from the center to two perpendicular points on an ellipse -/
theorem ellipse_perpendicular_points_sum_reciprocal_bounds
  (a b : ℝ) (ha : 0 < b) (hab : b < a)
  (P Q : ℝ × ℝ)
  (hP : (P.1 / a) ^ 2 + (P.2 / b) ^ 2 = 1)
  (hQ : (Q.1 / a) ^ 2 + (Q.2 / b) ^ 2 = 1)
  (hPOQ : (P.1 * Q.1 + P.2 * Q.2) / (Real.sqrt (P.1^2 + P.2^2) * Real.sqrt (Q.1^2 + Q.2^2)) = 0) :
  (a + b) / (a * b) ≤ 1 / Real.sqrt (P.1^2 + P.2^2) + 1 / Real.sqrt (Q.1^2 + Q.2^2) ∧
  1 / Real.sqrt (P.1^2 + P.2^2) + 1 / Real.sqrt (Q.1^2 + Q.2^2) ≤ Real.sqrt (2 * (a^2 + b^2)) / (a * b) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_points_sum_reciprocal_bounds_l2911_291162


namespace NUMINAMATH_CALUDE_greatest_base9_digit_sum_l2911_291101

/-- Represents a positive integer in base 9 --/
structure Base9Int where
  digits : List Nat
  positive : digits ≠ []
  valid : ∀ d ∈ digits, d < 9

/-- Converts a Base9Int to its decimal (base 10) representation --/
def toDecimal (n : Base9Int) : Nat :=
  n.digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Calculates the sum of digits of a Base9Int --/
def digitSum (n : Base9Int) : Nat :=
  n.digits.sum

/-- The main theorem to be proved --/
theorem greatest_base9_digit_sum :
  ∃ (max : Nat), 
    (∀ (n : Base9Int), toDecimal n < 3000 → digitSum n ≤ max) ∧ 
    (∃ (n : Base9Int), toDecimal n < 3000 ∧ digitSum n = max) ∧
    max = 24 := by
  sorry

end NUMINAMATH_CALUDE_greatest_base9_digit_sum_l2911_291101


namespace NUMINAMATH_CALUDE_animal_lifespan_l2911_291116

theorem animal_lifespan (bat_lifespan hamster_lifespan frog_lifespan : ℕ) : 
  bat_lifespan = 10 →
  hamster_lifespan = bat_lifespan - 6 →
  frog_lifespan = 4 * hamster_lifespan →
  bat_lifespan + hamster_lifespan + frog_lifespan = 30 := by
  sorry

end NUMINAMATH_CALUDE_animal_lifespan_l2911_291116


namespace NUMINAMATH_CALUDE_abs_x_minus_one_necessary_not_sufficient_l2911_291173

theorem abs_x_minus_one_necessary_not_sufficient :
  (∀ x : ℝ, x * (x + 1) < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x + 1) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_necessary_not_sufficient_l2911_291173


namespace NUMINAMATH_CALUDE_simplify_fraction_l2911_291137

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2911_291137


namespace NUMINAMATH_CALUDE_correlation_coefficients_relation_l2911_291103

def X : List ℝ := [16, 18, 20, 22]
def Y : List ℝ := [15.10, 12.81, 9.72, 3.21]
def U : List ℝ := [10, 20, 30]
def V : List ℝ := [7.5, 9.5, 16.6]

def r1 : ℝ := sorry
def r2 : ℝ := sorry

theorem correlation_coefficients_relation : r1 < 0 ∧ 0 < r2 := by sorry

end NUMINAMATH_CALUDE_correlation_coefficients_relation_l2911_291103


namespace NUMINAMATH_CALUDE_probability_two_diamonds_l2911_291191

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of suits in a standard deck
def num_suits : ℕ := 4

-- Define the number of ranks in a standard deck
def num_ranks : ℕ := 13

-- Define the number of cards of a single suit (Diamonds in this case)
def cards_per_suit : ℕ := total_cards / num_suits

-- Theorem statement
theorem probability_two_diamonds (total_cards num_suits num_ranks cards_per_suit : ℕ) 
  (h1 : total_cards = 52)
  (h2 : num_suits = 4)
  (h3 : num_ranks = 13)
  (h4 : cards_per_suit = total_cards / num_suits) :
  (cards_per_suit.choose 2 : ℚ) / (total_cards.choose 2) = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_diamonds_l2911_291191


namespace NUMINAMATH_CALUDE_customer_money_problem_l2911_291132

/-- Represents the initial amount of money a customer has --/
structure Money where
  dollars : ℕ
  cents : ℕ

/-- Represents the conditions of the problem --/
def satisfiesConditions (m : Money) : Prop :=
  let totalCents := 100 * m.dollars + m.cents
  let remainingCents := totalCents / 2
  let remainingDollars := remainingCents / 100
  let remainingCentsOnly := remainingCents % 100
  remainingCentsOnly = m.dollars ∧ remainingDollars = 2 * m.cents

/-- The theorem to be proved --/
theorem customer_money_problem :
  ∃ (m : Money), satisfiesConditions m ∧ m.dollars = 99 ∧ m.cents = 98 := by
  sorry

end NUMINAMATH_CALUDE_customer_money_problem_l2911_291132


namespace NUMINAMATH_CALUDE_range_of_a_l2911_291138

-- Define the sets A and B
def A : Set ℝ := {0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2911_291138


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l2911_291113

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_t_value :
  let m : ℝ × ℝ := (2, 8)
  let n : ℝ → ℝ × ℝ := fun t ↦ (-4, t)
  ∀ t : ℝ, parallel m (n t) → t = -16 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l2911_291113


namespace NUMINAMATH_CALUDE_tuesday_sales_total_l2911_291129

/-- Represents the types of flowers sold in the shop -/
inductive FlowerType
  | Rose
  | Lilac
  | Gardenia
  | Tulip
  | Orchid

/-- Represents the sales data for a given day -/
structure SalesData where
  roses : ℕ
  lilacs : ℕ
  gardenias : ℕ
  tulips : ℕ
  orchids : ℕ

/-- Calculate the total number of flowers sold -/
def totalFlowers (sales : SalesData) : ℕ :=
  sales.roses + sales.lilacs + sales.gardenias + sales.tulips + sales.orchids

/-- Apply Tuesday sales factors to Monday's sales -/
def applyTuesdayFactors (monday : SalesData) : SalesData :=
  { roses := monday.roses - monday.roses * 4 / 100,
    lilacs := monday.lilacs + monday.lilacs * 5 / 100,
    gardenias := monday.gardenias,
    tulips := monday.tulips - monday.tulips * 7 / 100,
    orchids := monday.orchids }

/-- Theorem: Given the conditions, the total number of flowers sold on Tuesday is 214 -/
theorem tuesday_sales_total (monday : SalesData)
  (h1 : monday.lilacs = 15)
  (h2 : monday.roses = 3 * monday.lilacs)
  (h3 : monday.gardenias = monday.lilacs / 2)
  (h4 : monday.tulips = 2 * (monday.roses + monday.gardenias))
  (h5 : monday.orchids = (monday.roses + monday.gardenias + monday.tulips) / 3)
  : totalFlowers (applyTuesdayFactors monday) = 214 := by
  sorry


end NUMINAMATH_CALUDE_tuesday_sales_total_l2911_291129


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_sufficient_but_not_necessary_l2911_291159

/-- Two lines are parallel if their slopes are equal -/
def parallel (m : ℝ) : Prop := 2 / m = (m - 1) / 1

/-- Sufficient condition: m = 2 implies the lines are parallel -/
theorem sufficient_condition : parallel 2 := by sorry

/-- Not necessary: there exists m ≠ 2 such that the lines are parallel -/
theorem not_necessary : ∃ m : ℝ, m ≠ 2 ∧ parallel m := by sorry

/-- m = 2 is a sufficient but not necessary condition for the lines to be parallel -/
theorem sufficient_but_not_necessary : 
  (parallel 2) ∧ (∃ m : ℝ, m ≠ 2 ∧ parallel m) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_sufficient_but_not_necessary_l2911_291159


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l2911_291199

theorem muffin_banana_cost_ratio :
  ∀ (muffin_cost banana_cost : ℝ),
  (6 * muffin_cost + 5 * banana_cost = (3 * muffin_cost + 20 * banana_cost) / 2) →
  (muffin_cost / banana_cost = 10 / 9) :=
by
  sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l2911_291199


namespace NUMINAMATH_CALUDE_greatest_x_value_l2911_291152

theorem greatest_x_value (x : ℝ) : 
  (4 * x^2 + 6 * x + 3 = 5) → x ≤ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2911_291152


namespace NUMINAMATH_CALUDE_greening_investment_equation_l2911_291135

theorem greening_investment_equation (initial_investment : ℝ) (final_investment : ℝ) (x : ℝ) :
  initial_investment = 20000 →
  final_investment = 25000 →
  (initial_investment / 1000) * (1 + x)^2 = (final_investment / 1000) :=
by
  sorry

end NUMINAMATH_CALUDE_greening_investment_equation_l2911_291135


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2911_291125

theorem fraction_equals_zero (x : ℝ) : 
  (x - 5) / (4 * x^2 - 1) = 0 ↔ x = 5 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2911_291125


namespace NUMINAMATH_CALUDE_negation_existence_divisibility_l2911_291124

theorem negation_existence_divisibility :
  (¬ ∃ n : ℕ+, 10 ∣ (n^2 + 3*n)) ↔ (∀ n : ℕ+, ¬(10 ∣ (n^2 + 3*n))) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_divisibility_l2911_291124


namespace NUMINAMATH_CALUDE_steve_pages_written_l2911_291104

/-- Calculates the total number of pages Steve writes in a month -/
def total_pages_written (days_in_month : ℕ) (letter_frequency : ℕ) (regular_letter_time : ℕ) 
  (time_per_page : ℕ) (long_letter_time : ℕ) : ℕ :=
  let regular_letters := days_in_month / letter_frequency
  let pages_per_regular_letter := regular_letter_time / time_per_page
  let regular_letter_pages := regular_letters * pages_per_regular_letter
  let long_letter_pages := long_letter_time / (2 * time_per_page)
  regular_letter_pages + long_letter_pages

theorem steve_pages_written :
  total_pages_written 30 3 20 10 80 = 24 := by
  sorry

end NUMINAMATH_CALUDE_steve_pages_written_l2911_291104


namespace NUMINAMATH_CALUDE_sum_of_digits_square_l2911_291149

/-- Sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ :=
  sorry

/-- Theorem: A positive integer equals the square of the sum of its digits if and only if it's 1 or 81 -/
theorem sum_of_digits_square (n : ℕ+) : n = (sum_of_digits n)^2 ↔ n = 1 ∨ n = 81 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_l2911_291149


namespace NUMINAMATH_CALUDE_problem_solution_l2911_291197

theorem problem_solution :
  (∀ x : ℝ, -3 * x * (2 * x^2 - x + 4) = -6 * x^3 + 3 * x^2 - 12 * x) ∧
  (∀ a b : ℝ, (2 * a - b) * (2 * a + b) = 4 * a^2 - b^2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2911_291197


namespace NUMINAMATH_CALUDE_simplify_expression_l2911_291172

theorem simplify_expression (x : ℝ) : (2*x)^5 - (5*x)*(x^4) = 27*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2911_291172


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2911_291114

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (lost_by : ℕ)
  (h_total : total_votes = 20000)
  (h_lost : lost_by = 16000) :
  (total_votes - lost_by) / total_votes * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l2911_291114


namespace NUMINAMATH_CALUDE_max_m_inequality_l2911_291130

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, (3 / a + 1 / b ≥ m / (a + 3 * b)) → m ≤ 12) ∧
  ∃ m : ℝ, m = 12 ∧ (3 / a + 1 / b ≥ m / (a + 3 * b)) :=
by sorry

end NUMINAMATH_CALUDE_max_m_inequality_l2911_291130


namespace NUMINAMATH_CALUDE_ten_pound_bag_cost_l2911_291134

/-- Represents the cost and weight of a bag of grass seed -/
structure Bag where
  weight : ℕ
  cost : ℚ

/-- Represents the purchase constraints and known information -/
structure PurchaseInfo where
  minWeight : ℕ
  maxWeight : ℕ
  leastCost : ℚ
  bag5lb : Bag
  bag25lb : Bag

/-- Calculates the cost of a 10-pound bag given the purchase information -/
def calculate10lbBagCost (info : PurchaseInfo) : ℚ :=
  info.leastCost - 3 * info.bag25lb.cost

/-- Theorem stating that the cost of the 10-pound bag is $1.98 -/
theorem ten_pound_bag_cost (info : PurchaseInfo) 
  (h1 : info.minWeight = 65)
  (h2 : info.maxWeight = 80)
  (h3 : info.leastCost = 98.73)
  (h4 : info.bag5lb = ⟨5, 13.80⟩)
  (h5 : info.bag25lb = ⟨25, 32.25⟩) :
  calculate10lbBagCost info = 1.98 := by sorry

end NUMINAMATH_CALUDE_ten_pound_bag_cost_l2911_291134


namespace NUMINAMATH_CALUDE_f_decreasing_interval_f_max_value_l2911_291112

-- Define the function f(x) = x^3 - 3x^2
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Theorem for the decreasing interval
theorem f_decreasing_interval :
  ∀ x ∈ (Set.Ioo 0 2), ∀ y ∈ (Set.Ioo 0 2), x < y → f x > f y :=
sorry

-- Theorem for the maximum value on [-4, 3]
theorem f_max_value :
  ∀ x ∈ (Set.Icc (-4) 3), f x ≤ 0 ∧ ∃ y ∈ (Set.Icc (-4) 3), f y = 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_f_max_value_l2911_291112


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_64_l2911_291108

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_64_l2911_291108


namespace NUMINAMATH_CALUDE_parabola_through_point_l2911_291143

/-- A parabola is defined by the equation y = ax^2 + bx + c where a ≠ 0 --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- A parabola opens upwards if a > 0 --/
def Parabola.opensUpwards (p : Parabola) : Prop := p.a > 0

/-- A point (x, y) lies on a parabola if it satisfies the parabola's equation --/
def Parabola.containsPoint (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- The theorem states that there exists a parabola that opens upwards and passes through (0, -2) --/
theorem parabola_through_point : ∃ p : Parabola, 
  p.opensUpwards ∧ p.containsPoint 0 (-2) ∧ p.a = 1 ∧ p.b = 0 ∧ p.c = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_l2911_291143


namespace NUMINAMATH_CALUDE_rock_paper_scissors_winning_probability_l2911_291178

/-- Represents the possible outcomes of a single round of Rock, Paper, Scissors -/
inductive RockPaperScissorsOutcome
  | Win
  | Lose
  | Draw

/-- Represents a two-player game of Rock, Paper, Scissors -/
structure RockPaperScissors where
  player1 : String
  player2 : String

/-- The probability of winning for each player in Rock, Paper, Scissors -/
def winningProbability (game : RockPaperScissors) : ℚ :=
  1 / 3

/-- Theorem: The probability of winning for each player in Rock, Paper, Scissors is 1/3 -/
theorem rock_paper_scissors_winning_probability (game : RockPaperScissors) :
  winningProbability game = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rock_paper_scissors_winning_probability_l2911_291178


namespace NUMINAMATH_CALUDE_total_volume_is_114_l2911_291171

/-- The volume of a cube with side length s -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- The number of Carl's cubes -/
def carl_cubes : ℕ := 4

/-- The side length of Carl's cubes -/
def carl_side_length : ℝ := 3

/-- The number of Kate's cubes -/
def kate_cubes : ℕ := 6

/-- The side length of Kate's cubes -/
def kate_side_length : ℝ := 1

/-- The total volume of all cubes -/
def total_volume : ℝ :=
  (carl_cubes : ℝ) * cube_volume carl_side_length +
  (kate_cubes : ℝ) * cube_volume kate_side_length

theorem total_volume_is_114 : total_volume = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_is_114_l2911_291171


namespace NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l2911_291142

/-- The parabola equation: x = -y^2 + 2y + 3 -/
def parabola (x y : ℝ) : Prop := x = -y^2 + 2*y + 3

/-- An x-intercept is a point where the parabola crosses the x-axis (y = 0) -/
def is_x_intercept (x : ℝ) : Prop := parabola x 0

/-- The parabola has exactly one x-intercept -/
theorem parabola_has_one_x_intercept : ∃! x : ℝ, is_x_intercept x := by sorry

end NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l2911_291142


namespace NUMINAMATH_CALUDE_shifted_parabola_passes_through_point_l2911_291128

/-- The original parabola function -/
def f (x : ℝ) : ℝ := -(x + 1)^2 + 4

/-- The shifted parabola function -/
def g (x : ℝ) : ℝ := -x^2 + 2

/-- Theorem stating that the shifted parabola passes through (-1, 1) -/
theorem shifted_parabola_passes_through_point :
  g (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_shifted_parabola_passes_through_point_l2911_291128


namespace NUMINAMATH_CALUDE_pink_crayons_count_l2911_291160

def total_crayons : ℕ := 24
def red_crayons : ℕ := 8
def blue_crayons : ℕ := 6
def green_crayons : ℕ := (2 * blue_crayons) / 3

theorem pink_crayons_count :
  total_crayons - red_crayons - blue_crayons - green_crayons = 6 := by
  sorry

end NUMINAMATH_CALUDE_pink_crayons_count_l2911_291160


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_l2911_291189

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1: Solution set when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_l2911_291189


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l2911_291118

theorem max_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h1 : x + y = 16) (h2 : x = 2 * y) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 16 → 1/a + 1/b ≤ 1/x + 1/y) ∧ 1/x + 1/y = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l2911_291118


namespace NUMINAMATH_CALUDE_problem_solution_l2911_291144

-- Define the linear function
def linear_function (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

-- Define the quadratic function
def quadratic_function (m n : ℝ) : ℝ → ℝ := λ x ↦ x^2 + m * x + n

theorem problem_solution 
  (k b m n : ℝ) 
  (h1 : k ≠ 0)
  (h2 : linear_function k b (-3) = 0)
  (h3 : linear_function k b 0 = -3)
  (h4 : quadratic_function m n (-3) = 0)
  (h5 : quadratic_function m n 0 = 3)
  (h6 : n > 0)
  (h7 : m ≤ 5) :
  (∃ t : ℝ, 
    (k = -1 ∧ b = -3) ∧ 
    (∃ x y : ℝ, x^2 + m*x + n = -x - 3 ∧ ∀ z : ℝ, z^2 + m*z + n ≥ x^2 + m*x + n) ∧
    (-9/4 < t ∧ t ≤ -1/4 ∧ ∀ z : ℝ, z^2 + m*z + n ≥ t)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2911_291144


namespace NUMINAMATH_CALUDE_square_root_equation_l2911_291150

theorem square_root_equation (y : ℝ) : 
  Real.sqrt (9 + Real.sqrt (4 * y - 5)) = Real.sqrt 10 → y = (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l2911_291150


namespace NUMINAMATH_CALUDE_fraction_inequality_l2911_291139

theorem fraction_inequality (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 3 →
  (5 * x + 3 > 9 - 3 * x) ↔ (x ∈ Set.Ioo (3/4 : ℝ) 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2911_291139


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l2911_291120

theorem cos_2alpha_value (α : ℝ) (a : ℝ × ℝ) :
  a = (Real.cos α, Real.sqrt 2 / 2) →
  ‖a‖ = Real.sqrt 3 / 2 →
  Real.cos (2 * α) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l2911_291120


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2911_291131

theorem simplify_sqrt_expression : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 75 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2911_291131


namespace NUMINAMATH_CALUDE_concert_tickets_l2911_291109

theorem concert_tickets (section_a_price section_b_price : ℝ)
  (total_tickets : ℕ) (total_revenue : ℝ) :
  section_a_price = 8 →
  section_b_price = 4.25 →
  total_tickets = 4500 →
  total_revenue = 30000 →
  ∃ (section_a_sold section_b_sold : ℕ),
    section_a_sold + section_b_sold = total_tickets ∧
    section_a_price * (section_a_sold : ℝ) + section_b_price * (section_b_sold : ℝ) = total_revenue ∧
    section_b_sold = 1600 :=
by sorry

end NUMINAMATH_CALUDE_concert_tickets_l2911_291109


namespace NUMINAMATH_CALUDE_minimum_value_problem_l2911_291181

theorem minimum_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  y / x + 4 / y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ y₀ / x₀ + 4 / y₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_problem_l2911_291181


namespace NUMINAMATH_CALUDE_f_difference_512_256_l2911_291184

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ := sorry

-- Define the f function as described in the problem
def f (n : ℕ) : ℚ := (sum_of_divisors n : ℚ) / n

-- Theorem statement
theorem f_difference_512_256 : f 512 - f 256 = 1 / 512 := by sorry

end NUMINAMATH_CALUDE_f_difference_512_256_l2911_291184


namespace NUMINAMATH_CALUDE_bacteria_count_scientific_notation_l2911_291192

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Checks if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (n : ℝ) : Prop :=
  n = sn.coefficient * (10 : ℝ) ^ sn.exponent

/-- The number of bacteria in a fly's stomach -/
def bacteria_count : ℕ := 28000000

/-- The scientific notation representation of the bacteria count -/
def bacteria_scientific : ScientificNotation where
  coefficient := 2.8
  exponent := 7
  h1 := by sorry

/-- Theorem stating that the scientific notation correctly represents the bacteria count -/
theorem bacteria_count_scientific_notation :
    represents bacteria_scientific (bacteria_count : ℝ) := by sorry

end NUMINAMATH_CALUDE_bacteria_count_scientific_notation_l2911_291192


namespace NUMINAMATH_CALUDE_player_in_first_and_last_game_l2911_291145

/-- Represents a chess tournament. -/
structure ChessTournament (n : ℕ) where
  /-- The number of players in the tournament. -/
  num_players : ℕ
  /-- The total number of games played in the tournament. -/
  num_games : ℕ
  /-- Condition that the number of players is 2n+3. -/
  player_count : num_players = 2*n + 3
  /-- Condition that the number of games is (2n+3)*(2n+2)/2. -/
  game_count : num_games = (num_players * (num_players - 1)) / 2
  /-- Function that returns true if a player played in a specific game. -/
  played_in_game : ℕ → ℕ → Prop
  /-- Condition that each player rests for at least n games after each match. -/
  rest_condition : ∀ p g₁ g₂, played_in_game p g₁ → played_in_game p g₂ → g₁ < g₂ → g₂ - g₁ > n

/-- Theorem stating that a player who played in the first game also played in the last game. -/
theorem player_in_first_and_last_game (n : ℕ) (tournament : ChessTournament n) :
  ∃ p, tournament.played_in_game p 1 ∧ tournament.played_in_game p tournament.num_games :=
sorry

end NUMINAMATH_CALUDE_player_in_first_and_last_game_l2911_291145


namespace NUMINAMATH_CALUDE_sine_double_angle_l2911_291193

theorem sine_double_angle (A : ℝ) (h : Real.cos (π/4 + A) = 5/13) : 
  Real.sin (2 * A) = 119/169 := by
  sorry

end NUMINAMATH_CALUDE_sine_double_angle_l2911_291193


namespace NUMINAMATH_CALUDE_max_garden_area_l2911_291170

/-- Represents a rectangular garden with given constraints -/
structure Garden where
  length : ℝ
  width : ℝ
  perimeter_eq : length * 2 + width * 2 = 400
  length_ge : length ≥ 100
  width_ge : width ≥ 50

/-- The area of a garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Theorem stating the maximum area of a garden with given constraints -/
theorem max_garden_area :
  ∀ g : Garden, g.area ≤ 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l2911_291170


namespace NUMINAMATH_CALUDE_remainder_17_pow_2090_mod_23_l2911_291147

theorem remainder_17_pow_2090_mod_23 : 17^2090 % 23 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_pow_2090_mod_23_l2911_291147


namespace NUMINAMATH_CALUDE_intersection_point_unique_l2911_291123

/-- The line equation -/
def line (x y z : ℝ) : Prop :=
  (x - 1) / 8 = (y - 8) / (-5) ∧ (x - 1) / 8 = (z + 5) / 12

/-- The plane equation -/
def plane (x y z : ℝ) : Prop :=
  x - 2*y - 3*z + 18 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (9, 3, 7)

theorem intersection_point_unique :
  ∃! p : ℝ × ℝ × ℝ, line p.1 p.2.1 p.2.2 ∧ plane p.1 p.2.1 p.2.2 ∧ p = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l2911_291123


namespace NUMINAMATH_CALUDE_prob_same_length_regular_hexagon_l2911_291161

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℕ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The probability of selecting two segments of the same length from T -/
def prob_same_length : ℚ := sorry

theorem prob_same_length_regular_hexagon :
  prob_same_length = 17 / 35 := by sorry

end NUMINAMATH_CALUDE_prob_same_length_regular_hexagon_l2911_291161


namespace NUMINAMATH_CALUDE_base_8_addition_problem_l2911_291140

/-- Converts a base 8 digit to base 10 -/
def to_base_10 (d : Nat) : Nat :=
  d

/-- Converts a base 10 number to base 8 -/
def to_base_8 (n : Nat) : Nat :=
  n

theorem base_8_addition_problem (X Y : Nat) 
  (h1 : X < 8 ∧ Y < 8)  -- X and Y are single digits in base 8
  (h2 : to_base_8 (4 * 8 * 8 + X * 8 + Y) + to_base_8 (5 * 8 + 3) = to_base_8 (6 * 8 * 8 + 1 * 8 + X)) :
  to_base_10 X + to_base_10 Y = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_8_addition_problem_l2911_291140


namespace NUMINAMATH_CALUDE_expression_value_l2911_291155

theorem expression_value (a b c d x : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1)
  (h3 : |x| = Real.sqrt 7) :
  x^2 + (a + b) * c * d * x + Real.sqrt (a + b) + (c * d)^(1/3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2911_291155


namespace NUMINAMATH_CALUDE_forty_percent_of_jacquelines_candy_bars_l2911_291153

def fred_candy_bars : ℕ := 12
def uncle_bob_candy_bars : ℕ := fred_candy_bars + 6
def total_fred_and_bob : ℕ := fred_candy_bars + uncle_bob_candy_bars
def jacqueline_candy_bars : ℕ := 10 * total_fred_and_bob

theorem forty_percent_of_jacquelines_candy_bars :
  (40 : ℚ) / 100 * jacqueline_candy_bars = 120 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_jacquelines_candy_bars_l2911_291153


namespace NUMINAMATH_CALUDE_three_and_negative_three_are_opposite_l2911_291126

-- Define opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Theorem statement
theorem three_and_negative_three_are_opposite :
  are_opposite 3 (-3) :=
sorry

end NUMINAMATH_CALUDE_three_and_negative_three_are_opposite_l2911_291126


namespace NUMINAMATH_CALUDE_log2_derivative_l2911_291174

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log2_derivative_l2911_291174


namespace NUMINAMATH_CALUDE_highest_backing_is_5000_l2911_291196

/-- Represents the financial backing levels for a crowdfunding campaign -/
structure FinancialBacking where
  lowest_level : ℕ
  second_level : ℕ
  highest_level : ℕ
  backers_lowest : ℕ
  backers_second : ℕ
  backers_highest : ℕ
  total_raised : ℕ

/-- The financial backing levels satisfy the given conditions -/
def ValidFinancialBacking (fb : FinancialBacking) : Prop :=
  fb.second_level = 10 * fb.lowest_level ∧
  fb.highest_level = 10 * fb.second_level ∧
  fb.backers_lowest = 10 ∧
  fb.backers_second = 3 ∧
  fb.backers_highest = 2 ∧
  fb.total_raised = 12000 ∧
  fb.total_raised = fb.backers_lowest * fb.lowest_level + 
                    fb.backers_second * fb.second_level + 
                    fb.backers_highest * fb.highest_level

/-- Theorem: The highest level of financial backing is $5000 -/
theorem highest_backing_is_5000 (fb : FinancialBacking) 
  (h : ValidFinancialBacking fb) : fb.highest_level = 5000 := by
  sorry

end NUMINAMATH_CALUDE_highest_backing_is_5000_l2911_291196


namespace NUMINAMATH_CALUDE_minimum_square_formation_l2911_291154

theorem minimum_square_formation :
  ∃ (n : ℕ), 
    (∃ (k : ℕ), n = k^2) ∧ 
    (∃ (m : ℕ), 11*n + 1 = m^2) ∧
    (∀ (x : ℕ), x < n → ¬(∃ (j : ℕ), x = j^2) ∨ ¬(∃ (l : ℕ), 11*x + 1 = l^2)) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_minimum_square_formation_l2911_291154


namespace NUMINAMATH_CALUDE_class_size_problem_l2911_291158

theorem class_size_problem (total : ℕ) (sum_fraction : ℕ) 
  (h1 : total = 85) 
  (h2 : sum_fraction = 42) : ∃ (a b : ℕ), 
  a + b = total ∧ 
  (3 * a) / 8 + (3 * b) / 5 = sum_fraction ∧ 
  a = 40 ∧ 
  b = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_size_problem_l2911_291158


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2911_291127

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 19 * n ≡ 2701 [ZMOD 9] ∧ ∀ (m : ℕ), m > 0 → 19 * m ≡ 2701 [ZMOD 9] → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2911_291127


namespace NUMINAMATH_CALUDE_joe_running_speed_l2911_291198

/-- Proves that Joe's running speed is 16 km/h given the problem conditions --/
theorem joe_running_speed : 
  ∀ (joe_speed pete_speed : ℝ),
  joe_speed = 2 * pete_speed →  -- Joe runs twice as fast as Pete
  (joe_speed + pete_speed) * (40 / 60) = 16 →  -- Total distance after 40 minutes
  joe_speed = 16 := by
  sorry

end NUMINAMATH_CALUDE_joe_running_speed_l2911_291198


namespace NUMINAMATH_CALUDE_tangent_line_of_cubic_with_even_derivative_l2911_291107

/-- The tangent line equation for a cubic function with specific properties -/
theorem tangent_line_of_cubic_with_even_derivative (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a - 3)*x
  let f' : ℝ → ℝ := λ x ↦ (3*x^2 + 2*a*x + (a - 3))
  (∀ x, f' x = f' (-x)) →
  (λ x ↦ -3*x) = (λ x ↦ f' 0 * x) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_of_cubic_with_even_derivative_l2911_291107


namespace NUMINAMATH_CALUDE_original_book_count_l2911_291119

/-- Represents a bookshelf with three layers of books -/
structure Bookshelf :=
  (layer1 : ℕ)
  (layer2 : ℕ)
  (layer3 : ℕ)

/-- The total number of books on the bookshelf -/
def total_books (b : Bookshelf) : ℕ := b.layer1 + b.layer2 + b.layer3

/-- The bookshelf after moving books between layers -/
def move_books (b : Bookshelf) : Bookshelf :=
  { layer1 := b.layer1 - 20,
    layer2 := b.layer2 + 20 + 17,
    layer3 := b.layer3 - 17 }

/-- Theorem stating the original number of books on each layer -/
theorem original_book_count :
  ∀ b : Bookshelf,
    total_books b = 270 →
    (let b' := move_books b
     b'.layer1 = b'.layer2 ∧ b'.layer2 = b'.layer3) →
    b.layer1 = 110 ∧ b.layer2 = 53 ∧ b.layer3 = 107 :=
by sorry


end NUMINAMATH_CALUDE_original_book_count_l2911_291119


namespace NUMINAMATH_CALUDE_min_value_implies_t_l2911_291141

/-- Given a real number t, f(x) is defined as the sum of the absolute values of (x-t) and (5-x) -/
def f (t : ℝ) (x : ℝ) : ℝ := |x - t| + |5 - x|

/-- The theorem states that if the minimum value of f(x) is 3, then t must be either 2 or 8 -/
theorem min_value_implies_t (t : ℝ) (h : ∀ x, f t x ≥ 3) (h_min : ∃ x, f t x = 3) : t = 2 ∨ t = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_t_l2911_291141


namespace NUMINAMATH_CALUDE_sibling_age_equation_l2911_291133

/-- Represents the ages of two siblings -/
structure SiblingAges where
  sister : ℕ
  brother : ℕ

/-- The condition of the ages this year -/
def this_year (ages : SiblingAges) : Prop :=
  ages.brother = 2 * ages.sister

/-- The condition of the ages four years ago -/
def four_years_ago (ages : SiblingAges) : Prop :=
  (ages.brother - 4) = 3 * (ages.sister - 4)

/-- The theorem representing the problem -/
theorem sibling_age_equation (x : ℕ) :
  ∃ (ages : SiblingAges),
    ages.sister = x ∧
    this_year ages ∧
    four_years_ago ages →
    2 * x - 4 = 3 * (x - 4) :=
by
  sorry

end NUMINAMATH_CALUDE_sibling_age_equation_l2911_291133


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2911_291115

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, Real.exp x > Real.log x)) ↔ (∃ x₀ : ℝ, Real.exp x₀ ≤ Real.log x₀) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2911_291115


namespace NUMINAMATH_CALUDE_cinnamon_swirl_sharing_l2911_291175

theorem cinnamon_swirl_sharing (total_pieces : ℕ) (jane_pieces : ℕ) (h1 : total_pieces = 12) (h2 : jane_pieces = 4) :
  total_pieces / jane_pieces = 3 :=
by
  sorry

#check cinnamon_swirl_sharing

end NUMINAMATH_CALUDE_cinnamon_swirl_sharing_l2911_291175


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2911_291111

def complex_number : ℂ := Complex.I * (1 - Complex.I)

theorem complex_number_in_first_quadrant : 
  complex_number.re > 0 ∧ complex_number.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2911_291111


namespace NUMINAMATH_CALUDE_sum_abc_l2911_291188

theorem sum_abc (a b c : ℤ) 
  (eq1 : 2 * a + 3 * b = 52)
  (eq2 : 3 * b + c = 41)
  (eq3 : b * c = 60) :
  a + b + c = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_l2911_291188


namespace NUMINAMATH_CALUDE_parabola_c_value_l2911_291110

theorem parabola_c_value (b c : ℝ) : 
  (2^2 + 2*b + c = 10) → 
  (4^2 + 4*b + c = 31) → 
  c = -3 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2911_291110


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l2911_291157

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (median : ℕ) : 
  n = 64 → sum = 4096 → sum = n * median → median = 64 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l2911_291157


namespace NUMINAMATH_CALUDE_parabola_vertex_l2911_291187

/-- The vertex of the parabola y = x^2 - 2 is at (0, -2) -/
theorem parabola_vertex :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2
  ∃! (h k : ℝ), (∀ x, f x = (x - h)^2 + k) ∧ h = 0 ∧ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2911_291187


namespace NUMINAMATH_CALUDE_abc_inequality_l2911_291122

theorem abc_inequality (a b c : ℝ) (ha : -1 ≤ a ∧ a ≤ 2) (hb : -1 ≤ b ∧ b ≤ 2) (hc : -1 ≤ c ∧ c ≤ 2) :
  a * b * c + 4 ≥ a * b + b * c + c * a := by
sorry

end NUMINAMATH_CALUDE_abc_inequality_l2911_291122


namespace NUMINAMATH_CALUDE_problem_solution_l2911_291105

theorem problem_solution (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 384) 
  (h2 : 3*m*n + 2*n^2 = 560) : 
  2*m^2 + 13*m*n + 6*n^2 - 444 = 2004 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2911_291105


namespace NUMINAMATH_CALUDE_students_in_different_clubs_l2911_291195

/-- The number of clubs in the school -/
def num_clubs : ℕ := 3

/-- The probability of a student joining any specific club -/
def prob_join_club : ℚ := 1 / num_clubs

/-- The probability of two students joining different clubs -/
def prob_different_clubs : ℚ := 2 / 3

theorem students_in_different_clubs :
  prob_different_clubs = 1 - (num_clubs : ℚ) * prob_join_club * prob_join_club := by
  sorry

end NUMINAMATH_CALUDE_students_in_different_clubs_l2911_291195


namespace NUMINAMATH_CALUDE_function_is_negation_l2911_291164

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (g x + y) = g x + g (g y + g (-x)) - x

/-- The main theorem stating that g(x) = -x for all x -/
theorem function_is_negation (g : ℝ → ℝ) (h : FunctionalEquation g) :
  ∀ x : ℝ, g x = -x :=
sorry

end NUMINAMATH_CALUDE_function_is_negation_l2911_291164


namespace NUMINAMATH_CALUDE_mirror_area_l2911_291121

theorem mirror_area (frame_length frame_width frame_side_width : ℝ) 
  (h1 : frame_length = 80)
  (h2 : frame_width = 60)
  (h3 : frame_side_width = 10) :
  (frame_length - 2 * frame_side_width) * (frame_width - 2 * frame_side_width) = 2400 :=
by sorry

end NUMINAMATH_CALUDE_mirror_area_l2911_291121


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2911_291182

/-- Given a fixed positive integer N, prove that any function f satisfying
    the given conditions is identically zero. -/
theorem functional_equation_solution (N : ℕ+) (f : ℤ → ℝ)
  (h1 : ∀ k : ℤ, f (2 * k) = 2 * f k)
  (h2 : ∀ k : ℤ, f (N - k) = f k) :
  ∀ a : ℤ, f a = 0 := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2911_291182


namespace NUMINAMATH_CALUDE_minimum_value_sum_reciprocals_l2911_291166

theorem minimum_value_sum_reciprocals (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_eq_three : a + b + c = 3) :
  (1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a)) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_sum_reciprocals_l2911_291166


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2911_291194

theorem polynomial_coefficient_sum (a k n : ℤ) : 
  (∀ x : ℝ, (3 * x^2 + 2) * (2 * x^3 - 7) = a * x^5 + k * x^2 + n) →
  a - n + k = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2911_291194


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2911_291190

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ  -- Half-length of the transverse axis
  b : ℝ  -- Half-length of the conjugate axis
  c : ℝ  -- Focal distance

/-- The standard equation of a hyperbola -/
def standardEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / h.a^2 - x^2 / h.b^2 = 1

/-- Theorem: Given a hyperbola with specific properties, its standard equation is y²/4 - x²/4 = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
  (vertex_condition : h.a = 2)
  (axis_sum_condition : 2 * h.a + 2 * h.b = Real.sqrt 2 * 2 * h.c) :
  standardEquation h x y ↔ y^2 / 4 - x^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2911_291190


namespace NUMINAMATH_CALUDE_caden_coin_ratio_l2911_291167

/-- Represents the number of coins of each type -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCounts) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes + 25 * coins.quarters

/-- Represents Caden's coin collection -/
def cadenCoins : CoinCounts where
  pennies := 120
  nickels := 40
  dimes := 8
  quarters := 16

theorem caden_coin_ratio :
  cadenCoins.pennies = 120 ∧
  cadenCoins.pennies = 3 * cadenCoins.nickels ∧
  cadenCoins.quarters = 2 * cadenCoins.dimes ∧
  totalValue cadenCoins = 800 →
  cadenCoins.nickels = 5 * cadenCoins.dimes :=
by sorry

end NUMINAMATH_CALUDE_caden_coin_ratio_l2911_291167
