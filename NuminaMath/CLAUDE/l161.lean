import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_set_l161_16162

def solution_set : Set Int := {0, -6, -12, -14, -18, -20, -21, -24, -26, -27, -28, -30, -32, -33, -34, -35, -36, -38, -39, -40, -41, -44, -45, -46, -47, -49, -50, -51, -52, -53, -55, -57, -58, -59, -61, -64, -65, -67, -71, -73, -79, -85}

theorem equation_solution_set :
  {x : Int | Int.floor (x / 2) + Int.floor (x / 3) + Int.floor (x / 7) = x} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_set_l161_16162


namespace NUMINAMATH_CALUDE_simplify_expression_l161_16133

theorem simplify_expression (a : ℝ) (h : a ≠ 2) :
  (a - 2) * ((a^2 - 4) / (a^2 - 4*a + 4)) = a + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l161_16133


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l161_16152

theorem large_rectangle_perimeter : 
  ∀ (square_perimeter small_rect_perimeter : ℝ),
  square_perimeter = 24 →
  small_rect_perimeter = 16 →
  let square_side := square_perimeter / 4
  let small_rect_length := square_side
  let small_rect_width := (small_rect_perimeter / 2) - small_rect_length
  let large_rect_height := square_side + 2 * small_rect_width
  let large_rect_width := 3 * square_side
  2 * (large_rect_height + large_rect_width) = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_large_rectangle_perimeter_l161_16152


namespace NUMINAMATH_CALUDE_natural_number_pairs_equality_l161_16168

theorem natural_number_pairs_equality (m n : ℕ) : 
  n * (n - 1) * (n - 2) * (n - 3) = m * (m - 1) ↔ 
  ((m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 1 ∧ n = 3)) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pairs_equality_l161_16168


namespace NUMINAMATH_CALUDE_wendy_makeup_time_l161_16173

/-- Calculates the time spent on make-up given the number of facial products,
    waiting time between products, and total time for the full face routine. -/
def makeupTime (numProducts : ℕ) (waitingTime : ℕ) (totalTime : ℕ) : ℕ :=
  totalTime - (numProducts - 1) * waitingTime

/-- Proves that given 5 facial products, 5 minutes waiting time between each product,
    and a total of 55 minutes for the "full face," the time spent on make-up is 35 minutes. -/
theorem wendy_makeup_time :
  makeupTime 5 5 55 = 35 := by
  sorry

#eval makeupTime 5 5 55

end NUMINAMATH_CALUDE_wendy_makeup_time_l161_16173


namespace NUMINAMATH_CALUDE_zoes_purchase_cost_l161_16155

/-- The total cost of soda and pizza for a group, given the cost per item and number of people -/
def totalCost (sodaCost pizzaCost : ℚ) (numPeople : ℕ) : ℚ :=
  numPeople * (sodaCost + pizzaCost)

/-- Theorem: The total cost for soda and pizza for 6 people is $9.00 -/
theorem zoes_purchase_cost :
  totalCost (1/2) 1 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_zoes_purchase_cost_l161_16155


namespace NUMINAMATH_CALUDE_log_inequality_l161_16116

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

-- State the theorem
theorem log_inequality (m : ℝ) 
  (h1 : ∀ x : ℝ, 1/m - 4 ≥ f m x)
  (h2 : m > 0) :
  Real.log (m + 2) / Real.log (m + 1) > Real.log (m + 3) / Real.log (m + 2) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_l161_16116


namespace NUMINAMATH_CALUDE_complex_rearrangements_with_vowels_first_l161_16167

def word : String := "COMPLEX"

def is_vowel (c : Char) : Bool :=
  c = 'O' || c = 'E'

def vowels : List Char :=
  word.data.filter is_vowel

def consonants : List Char :=
  word.data.filter (fun c => !is_vowel c)

theorem complex_rearrangements_with_vowels_first :
  (vowels.permutations.length) * (consonants.permutations.length) = 240 := by
  sorry

end NUMINAMATH_CALUDE_complex_rearrangements_with_vowels_first_l161_16167


namespace NUMINAMATH_CALUDE_catchup_distance_proof_l161_16192

/-- The speed of walker A in km/h -/
def speed_A : ℝ := 10

/-- The speed of cyclist B in km/h -/
def speed_B : ℝ := 20

/-- The time difference between A's start and B's start in hours -/
def time_difference : ℝ := 7

/-- The distance at which B catches up with A in km -/
def catchup_distance : ℝ := 140

theorem catchup_distance_proof :
  speed_A * time_difference +
  speed_A * (catchup_distance / speed_B) =
  catchup_distance :=
sorry

end NUMINAMATH_CALUDE_catchup_distance_proof_l161_16192


namespace NUMINAMATH_CALUDE_plane_train_speed_ratio_l161_16176

/-- The ratio of plane speed to train speed given specific travel conditions -/
theorem plane_train_speed_ratio :
  -- Train travel time
  ∀ (train_time : ℝ) (plane_time : ℝ) (wait_time : ℝ) (meet_time : ℝ),
  train_time = 20 →
  -- Plane travel time (including waiting)
  plane_time = 10 →
  -- Waiting time is more than 5 hours after train departure
  wait_time > 5 →
  -- Plane is above train 8/9 hours after departure
  meet_time = 8/9 →
  -- At that point, plane and train have traveled the same distance
  ∃ (train_speed : ℝ) (plane_speed : ℝ),
    train_speed * (wait_time + meet_time) = plane_speed * meet_time →
    train_speed * train_time = plane_speed * (plane_time - wait_time) →
    -- The ratio of plane speed to train speed is 5.75
    plane_speed / train_speed = 5.75 := by
  sorry

end NUMINAMATH_CALUDE_plane_train_speed_ratio_l161_16176


namespace NUMINAMATH_CALUDE_natashas_average_speed_l161_16149

/-- Natasha's hill climbing problem -/
theorem natashas_average_speed
  (climb_time : ℝ)
  (descent_time : ℝ)
  (climb_speed : ℝ)
  (h_climb_time : climb_time = 4)
  (h_descent_time : descent_time = 2)
  (h_climb_speed : climb_speed = 3)
  : (2 * climb_speed * climb_time) / (climb_time + descent_time) = 4 :=
by sorry

end NUMINAMATH_CALUDE_natashas_average_speed_l161_16149


namespace NUMINAMATH_CALUDE_pen_retailer_profit_percentage_retailer_profit_example_l161_16193

/-- Calculates the profit percentage for a retailer selling pens with a discount -/
theorem pen_retailer_profit_percentage 
  (num_pens : ℕ) 
  (price_in_pens : ℕ) 
  (discount_percent : ℚ) : ℚ :=
  let market_price := price_in_pens
  let cost_price := market_price
  let selling_price := num_pens * (1 - discount_percent / 100) * market_price / price_in_pens
  let profit := selling_price - cost_price
  let profit_percentage := profit / cost_price * 100
  profit_percentage

/-- The profit percentage for a retailer buying 140 pens at the price of 36 pens
    and selling with a 1% discount is approximately 285% -/
theorem retailer_profit_example :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |pen_retailer_profit_percentage 140 36 1 - 285| < ε :=
sorry

end NUMINAMATH_CALUDE_pen_retailer_profit_percentage_retailer_profit_example_l161_16193


namespace NUMINAMATH_CALUDE_vector_is_direction_vector_l161_16166

/-- A line in 2D space --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A vector in 2D space --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a vector is a direction vector of a line --/
def isDirectionVector (l : Line2D) (v : Vector2D) : Prop :=
  l.a * v.x + l.b * v.y = 0

/-- The given line x - 3y + 1 = 0 --/
def givenLine : Line2D :=
  { a := 1, b := -3, c := 1 }

/-- The vector (3,1) --/
def givenVector : Vector2D :=
  { x := 3, y := 1 }

/-- Theorem: (3,1) is a direction vector of the line x - 3y + 1 = 0 --/
theorem vector_is_direction_vector : isDirectionVector givenLine givenVector := by
  sorry

end NUMINAMATH_CALUDE_vector_is_direction_vector_l161_16166


namespace NUMINAMATH_CALUDE_log_xy_value_l161_16170

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^4) = 1) (h2 : Real.log (x^3 * y) = 1) :
  Real.log (x * y) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_value_l161_16170


namespace NUMINAMATH_CALUDE_same_solution_k_value_l161_16181

theorem same_solution_k_value (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 * k = 21 ↔ 5 * x + 3 = 0) → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_k_value_l161_16181


namespace NUMINAMATH_CALUDE_inequality_system_solution_l161_16128

theorem inequality_system_solution (x : ℝ) :
  (x - 2) / (x - 1) < 1 ∧ -x^2 + x + 2 < 0 → x > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l161_16128


namespace NUMINAMATH_CALUDE_win_sector_area_l161_16183

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/3) :
  p * π * r^2 = 48 * π := by sorry

end NUMINAMATH_CALUDE_win_sector_area_l161_16183


namespace NUMINAMATH_CALUDE_jogger_train_distance_l161_16143

/-- Proof that a jogger is 240 meters ahead of a train's engine given specific conditions -/
theorem jogger_train_distance (jogger_speed train_speed train_length pass_time : ℝ) 
  (h1 : jogger_speed = 9) -- jogger's speed in km/hr
  (h2 : train_speed = 45) -- train's speed in km/hr
  (h3 : train_length = 120) -- train's length in meters
  (h4 : pass_time = 36) -- time taken for train to pass jogger in seconds
  : (train_speed - jogger_speed) * (5/18) * pass_time - train_length = 240 :=
by
  sorry

#eval (45 - 9) * (5/18) * 36 - 120 -- Should evaluate to 240

end NUMINAMATH_CALUDE_jogger_train_distance_l161_16143


namespace NUMINAMATH_CALUDE_f_properties_l161_16169

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

theorem f_properties :
  (∀ x, x < -1/3 ∨ x > 1 → f' x > 0) ∧
  (∀ x, -1/3 < x ∧ x < 1 → f' x < 0) ∧
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ≤ 2) ∧
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ≥ -10) ∧
  (∃ x, x ∈ Set.Icc (-2 : ℝ) 2 ∧ f x = 2) ∧
  (∃ x, x ∈ Set.Icc (-2 : ℝ) 2 ∧ f x = -10) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l161_16169


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l161_16177

theorem chinese_remainder_theorem_example :
  ∃ x : ℤ, x % 3 = 2 ∧ x % 4 = 3 ∧ x % 5 = 1 ∧ x % 60 = 11 := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l161_16177


namespace NUMINAMATH_CALUDE_largest_domain_l161_16159

-- Define the property of the function f
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (f x + f (1/x) = x^2)

-- Define the domain of f
def domain (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | x ≠ 0 ∧ ∃ y : ℝ, f x = y}

-- Theorem statement
theorem largest_domain (f : ℝ → ℝ) (h : has_property f) :
  domain f = {x : ℝ | x ≠ 0} :=
sorry

end NUMINAMATH_CALUDE_largest_domain_l161_16159


namespace NUMINAMATH_CALUDE_at_least_one_square_is_one_l161_16189

theorem at_least_one_square_is_one (a b c : ℤ) 
  (h : |a + b + c| + 2 = |a| + |b| + |c|) : 
  a^2 = 1 ∨ b^2 = 1 ∨ c^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_square_is_one_l161_16189


namespace NUMINAMATH_CALUDE_existence_of_abc_l161_16102

theorem existence_of_abc (n k : ℕ) (hn : n > 20) (hk : k > 1) (hdiv : k^2 ∣ n) :
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ n = a * b + b * c + c * a :=
sorry

end NUMINAMATH_CALUDE_existence_of_abc_l161_16102


namespace NUMINAMATH_CALUDE_range_of_m_l161_16191

-- Define the polynomials P and Q
def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def Q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the negations of P and Q
def not_P (x : ℝ) : Prop := ¬(P x)
def not_Q (x m : ℝ) : Prop := ¬(Q x m)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, 
    (m > 0) →
    (∀ x : ℝ, not_P x → not_Q x m) →
    (∃ x : ℝ, not_Q x m ∧ ¬(not_P x)) →
    (0 < m ∧ m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l161_16191


namespace NUMINAMATH_CALUDE_expression_non_negative_l161_16184

theorem expression_non_negative (x y : ℝ) : 
  5 * x^2 + 5 * y^2 + 8 * x * y + 2 * y - 2 * x + 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_non_negative_l161_16184


namespace NUMINAMATH_CALUDE_janice_purchase_problem_l161_16117

theorem janice_purchase_problem :
  ∀ (a b c : ℕ),
    a + b + c = 60 →
    15 * a + 400 * b + 500 * c = 6000 →
    a = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_janice_purchase_problem_l161_16117


namespace NUMINAMATH_CALUDE_athlete_A_most_stable_l161_16156

/-- Represents an athlete with their performance variance -/
structure Athlete where
  name : String
  variance : Float

/-- Determines if an athlete has the most stable performance among a list of athletes -/
def hasMostStablePerformance (a : Athlete) (athletes : List Athlete) : Prop :=
  ∀ b ∈ athletes, a.variance ≤ b.variance

/-- The list of athletes with their variances -/
def athleteList : List Athlete :=
  [⟨"A", 0.019⟩, ⟨"B", 0.021⟩, ⟨"C", 0.020⟩, ⟨"D", 0.022⟩]

theorem athlete_A_most_stable :
  ∃ a ∈ athleteList, a.name = "A" ∧ hasMostStablePerformance a athleteList := by
  sorry


end NUMINAMATH_CALUDE_athlete_A_most_stable_l161_16156


namespace NUMINAMATH_CALUDE_sum_of_number_and_square_l161_16180

theorem sum_of_number_and_square : 11 + 11^2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_square_l161_16180


namespace NUMINAMATH_CALUDE_expression_evaluation_l161_16123

theorem expression_evaluation : -2^3 + |2 - 3| - 2 * (-1)^2023 = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l161_16123


namespace NUMINAMATH_CALUDE_product_of_cubic_fractions_l161_16144

theorem product_of_cubic_fractions :
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 57 / 84 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cubic_fractions_l161_16144


namespace NUMINAMATH_CALUDE_tangent_sum_arithmetic_sequence_l161_16115

theorem tangent_sum_arithmetic_sequence (x y z : ℝ) :
  (∃ k : ℝ, x = y - π/2 ∧ z = y + π/2) →
  Real.tan x * Real.tan y + Real.tan y * Real.tan z + Real.tan z * Real.tan x = -3 := by
sorry

end NUMINAMATH_CALUDE_tangent_sum_arithmetic_sequence_l161_16115


namespace NUMINAMATH_CALUDE_sixth_power_sum_l161_16161

/-- Given real numbers a, b, x, and y satisfying certain conditions, 
    prove that ax^6 + by^6 = 1531.25 -/
theorem sixth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 12)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 80) :
  a * x^6 + b * y^6 = 1531.25 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l161_16161


namespace NUMINAMATH_CALUDE_unique_satisfying_polynomial_l161_16163

/-- A polynomial satisfying the given conditions -/
def satisfying_polynomial (P : ℝ → ℝ) : Prop :=
  (P 2017 = 2016) ∧ 
  (∀ x : ℝ, (P x + 1)^2 = P (x^2 + 1))

/-- The theorem stating that the only polynomial satisfying the conditions is x - 1 -/
theorem unique_satisfying_polynomial : 
  ∀ P : ℝ → ℝ, satisfying_polynomial P → (∀ x : ℝ, P x = x - 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_satisfying_polynomial_l161_16163


namespace NUMINAMATH_CALUDE_division_problem_solution_l161_16165

theorem division_problem_solution :
  ∃! y : ℝ, y > 0 ∧ (2 * (62.5 + 5) / y) - 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_solution_l161_16165


namespace NUMINAMATH_CALUDE_diamond_3_2_l161_16135

/-- The diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 + 3*a^2*b + 3*a*b^2 + b^3

/-- Theorem: The diamond operation applied to 3 and 2 equals 125 -/
theorem diamond_3_2 : diamond 3 2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_diamond_3_2_l161_16135


namespace NUMINAMATH_CALUDE_prob_not_both_odd_is_five_sixths_l161_16100

/-- The set of numbers to choose from -/
def S : Finset ℕ := {1, 2, 3, 4}

/-- The set of odd numbers in S -/
def odd_numbers : Finset ℕ := {1, 3}

/-- The probability of selecting two numbers without replacement from S such that not both are odd -/
def prob_not_both_odd : ℚ :=
  1 - (Finset.card odd_numbers).choose 2 / (Finset.card S).choose 2

theorem prob_not_both_odd_is_five_sixths : 
  prob_not_both_odd = 5/6 := by sorry

end NUMINAMATH_CALUDE_prob_not_both_odd_is_five_sixths_l161_16100


namespace NUMINAMATH_CALUDE_power_exceeds_thresholds_l161_16171

theorem power_exceeds_thresholds : ∃ (n1 n2 n3 m1 m2 m3 : ℕ), 
  (1.01 : ℝ) ^ n1 > 1000000000000 ∧
  (1.001 : ℝ) ^ n2 > 1000000000000 ∧
  (1.000001 : ℝ) ^ n3 > 1000000000000 ∧
  (1.01 : ℝ) ^ m1 > 1000000000000000000 ∧
  (1.001 : ℝ) ^ m2 > 1000000000000000000 ∧
  (1.000001 : ℝ) ^ m3 > 1000000000000000000 :=
by sorry

end NUMINAMATH_CALUDE_power_exceeds_thresholds_l161_16171


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l161_16112

theorem simplify_algebraic_expression (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ 3) (h3 : x ≠ 1) :
  (1 - 4 / (x + 3)) / ((x^2 - 2*x + 1) / (x^2 - 9)) = (x - 3) / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l161_16112


namespace NUMINAMATH_CALUDE_action_figure_price_l161_16142

theorem action_figure_price (board_game_cost : ℝ) (num_figures : ℕ) (total_cost : ℝ) :
  board_game_cost = 2 →
  num_figures = 4 →
  total_cost = 30 →
  ∃ (figure_price : ℝ), figure_price = 7 ∧ total_cost = board_game_cost + num_figures * figure_price :=
by
  sorry

end NUMINAMATH_CALUDE_action_figure_price_l161_16142


namespace NUMINAMATH_CALUDE_triangle_side_not_unique_l161_16130

/-- Represents a triangle with sides a, b, and c, and area A -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ

/-- Theorem stating that the length of side 'a' in a triangle cannot be uniquely determined
    given only the lengths of two sides and the area -/
theorem triangle_side_not_unique (t : Triangle) (h1 : t.b = 19) (h2 : t.c = 5) (h3 : t.A = 47.5) :
  ¬ ∃! a : ℝ, t.a = a ∧ 0 < a ∧ a + t.b > t.c ∧ a + t.c > t.b ∧ t.b + t.c > a :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_not_unique_l161_16130


namespace NUMINAMATH_CALUDE_parallelogram_angles_l161_16107

/-- Given a parallelogram with perimeter to larger diagonal ratio k,
    where the larger diagonal divides one angle in the ratio 1:2,
    prove that its angles are 3 arccos((2+k)/(2k)) and π - 3 arccos((2+k)/(2k)). -/
theorem parallelogram_angles (k : ℝ) (k_pos : k > 0) :
  ∃ (angle₁ angle₂ : ℝ),
    angle₁ = 3 * Real.arccos ((2 + k) / (2 * k)) ∧
    angle₂ = Real.pi - 3 * Real.arccos ((2 + k) / (2 * k)) ∧
    angle₁ + angle₂ = Real.pi ∧
    angle₁ > 0 ∧ angle₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_angles_l161_16107


namespace NUMINAMATH_CALUDE_roots_are_irrational_l161_16137

theorem roots_are_irrational (k : ℝ) : 
  (∃ x y : ℝ, x * y = 10 ∧ x^2 - 3*k*x + 2*k^2 - 1 = 0 ∧ y^2 - 3*k*y + 2*k^2 - 1 = 0) →
  (∃ x y : ℝ, x * y = 10 ∧ x^2 - 3*k*x + 2*k^2 - 1 = 0 ∧ y^2 - 3*k*y + 2*k^2 - 1 = 0 ∧ 
   (¬∃ m n : ℤ, x = m / n ∨ y = m / n)) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_irrational_l161_16137


namespace NUMINAMATH_CALUDE_rational_inequalities_l161_16198

theorem rational_inequalities (a b : ℚ) : 
  ((a + b < a) → (b < 0)) ∧ ((a - b < a) → (b > 0)) := by sorry

end NUMINAMATH_CALUDE_rational_inequalities_l161_16198


namespace NUMINAMATH_CALUDE_circle_properties_l161_16187

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 8*x - 10*y = 10 - y^2 + 6*x

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 5)

-- Define the radius of the circle
def radius : ℝ := 6

-- Theorem to prove
theorem circle_properties :
  (∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
  center.1 + center.2 + radius = 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l161_16187


namespace NUMINAMATH_CALUDE_lottery_probabilities_l161_16127

/-- Represents the probability of drawing a red ball from Box A -/
def prob_red_A : ℚ := 4 / 10

/-- Represents the probability of drawing a red ball from Box B -/
def prob_red_B : ℚ := 1 / 2

/-- Represents the probability of winning the first prize in one draw -/
def prob_first_prize : ℚ := prob_red_A * prob_red_B

/-- Represents the probability of winning the second prize in one draw -/
def prob_second_prize : ℚ := prob_red_A * (1 - prob_red_B) + (1 - prob_red_A) * prob_red_B

/-- Represents the probability of winning a prize in one draw -/
def prob_win_prize : ℚ := prob_first_prize + prob_second_prize

/-- Represents the number of independent lottery draws -/
def num_draws : ℕ := 3

theorem lottery_probabilities :
  (prob_win_prize = 7 / 10) ∧
  (1 - prob_first_prize ^ num_draws = 124 / 125) := by
  sorry

end NUMINAMATH_CALUDE_lottery_probabilities_l161_16127


namespace NUMINAMATH_CALUDE_soft_drink_cost_l161_16138

/-- Proves that the cost of each soft drink is $4 given the conditions of Benny's purchase. -/
theorem soft_drink_cost (num_soft_drinks : ℕ) (num_candy_bars : ℕ) (total_spent : ℚ) (candy_bar_cost : ℚ) :
  num_soft_drinks = 2 →
  num_candy_bars = 5 →
  total_spent = 28 →
  candy_bar_cost = 4 →
  ∃ (soft_drink_cost : ℚ), 
    soft_drink_cost * num_soft_drinks + candy_bar_cost * num_candy_bars = total_spent ∧
    soft_drink_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_soft_drink_cost_l161_16138


namespace NUMINAMATH_CALUDE_total_cost_for_all_puppies_l161_16114

/-- Represents the cost calculation for a dog breed -/
structure BreedCost where
  mothers : Nat
  puppiesPerLitter : Nat
  shotCost : Nat
  shotsPerPuppy : Nat
  additionalCosts : Nat

/-- Calculates the total cost for a breed -/
def totalBreedCost (breed : BreedCost) : Nat :=
  breed.mothers * breed.puppiesPerLitter * 
  (breed.shotCost * breed.shotsPerPuppy + breed.additionalCosts)

/-- Theorem stating the total cost for all puppies -/
theorem total_cost_for_all_puppies :
  let goldenRetrievers : BreedCost := {
    mothers := 3,
    puppiesPerLitter := 4,
    shotCost := 5,
    shotsPerPuppy := 2,
    additionalCosts := 72  -- 6 months of vitamins at $12 per month
  }
  let germanShepherds : BreedCost := {
    mothers := 2,
    puppiesPerLitter := 5,
    shotCost := 8,
    shotsPerPuppy := 3,
    additionalCosts := 40  -- microchip ($25) + special toy ($15)
  }
  let bulldogs : BreedCost := {
    mothers := 4,
    puppiesPerLitter := 3,
    shotCost := 10,
    shotsPerPuppy := 4,
    additionalCosts := 38  -- customized collar ($20) + exclusive chew toy ($18)
  }
  totalBreedCost goldenRetrievers + totalBreedCost germanShepherds + totalBreedCost bulldogs = 2560 :=
by
  sorry


end NUMINAMATH_CALUDE_total_cost_for_all_puppies_l161_16114


namespace NUMINAMATH_CALUDE_M_equiv_NotFirstOrThirdQuadrant_l161_16124

/-- The set M of points (x,y) in ℝ² where xy ≤ 0 -/
def M : Set (ℝ × ℝ) := {p | p.1 * p.2 ≤ 0}

/-- The set of points not in the first or third quadrants of ℝ² -/
def NotFirstOrThirdQuadrant : Set (ℝ × ℝ) := 
  {p | p.1 * p.2 ≤ 0}

/-- Theorem stating that M is equivalent to the set of points not in the first or third quadrants -/
theorem M_equiv_NotFirstOrThirdQuadrant : M = NotFirstOrThirdQuadrant := by
  sorry


end NUMINAMATH_CALUDE_M_equiv_NotFirstOrThirdQuadrant_l161_16124


namespace NUMINAMATH_CALUDE_planes_perpendicular_l161_16164

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (a b c : Line) 
  (α β γ : Plane) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h3 : parallel_line_plane a α) 
  (h4 : contained_in b β) 
  (h5 : parallel_lines a b) : 
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l161_16164


namespace NUMINAMATH_CALUDE_berry_farm_kept_fraction_l161_16157

/-- Given a berry farm scenario, prove that half of the fresh berries need to be kept. -/
theorem berry_farm_kept_fraction (total_berries : ℕ) (rotten_fraction : ℚ) (berries_to_sell : ℕ) :
  total_berries = 60 →
  rotten_fraction = 1/3 →
  berries_to_sell = 20 →
  (total_berries - (rotten_fraction * total_berries).num - berries_to_sell) / 
  (total_berries - (rotten_fraction * total_berries).num) = 1/2 := by
  sorry

#check berry_farm_kept_fraction

end NUMINAMATH_CALUDE_berry_farm_kept_fraction_l161_16157


namespace NUMINAMATH_CALUDE_money_theorem_l161_16160

/-- The amount of money Fritz has -/
def fritz_money : ℕ := 40

/-- The amount of money Sean has -/
def sean_money : ℕ := fritz_money / 2 + 4

/-- The amount of money Rick has -/
def rick_money : ℕ := 3 * sean_money

/-- The amount of money Lindsey has -/
def lindsey_money : ℕ := 2 * (sean_money + rick_money)

/-- The total amount of money Lindsey, Rick, and Sean have combined -/
def total_money : ℕ := lindsey_money + rick_money + sean_money

theorem money_theorem : total_money = 288 := by
  sorry

end NUMINAMATH_CALUDE_money_theorem_l161_16160


namespace NUMINAMATH_CALUDE_rachel_theorem_l161_16140

def rachel_problem (initial_amount lunch_fraction dvd_fraction : ℚ) : Prop :=
  let lunch_expense := initial_amount * lunch_fraction
  let dvd_expense := initial_amount * dvd_fraction
  let remaining_amount := initial_amount - lunch_expense - dvd_expense
  remaining_amount = 50

theorem rachel_theorem :
  rachel_problem 200 (1/4) (1/2) :=
by sorry

end NUMINAMATH_CALUDE_rachel_theorem_l161_16140


namespace NUMINAMATH_CALUDE_shirt_price_reduction_l161_16158

theorem shirt_price_reduction (original_price : ℝ) (original_price_positive : original_price > 0) : 
  let first_sale_price := 0.9 * original_price
  let final_price := 0.9 * first_sale_price
  final_price / original_price = 0.81 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_reduction_l161_16158


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l161_16110

theorem min_sum_with_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1/x + 4/y = 1) : x + y ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l161_16110


namespace NUMINAMATH_CALUDE_total_calories_burned_l161_16103

/-- The number of times players run up and down the bleachers -/
def num_runs : ℕ := 60

/-- The number of stairs in the first half of the staircase -/
def stairs_first_half : ℕ := 20

/-- The number of stairs in the second half of the staircase -/
def stairs_second_half : ℕ := 25

/-- The number of calories burned per stair in the first half -/
def calories_per_stair_first_half : ℕ := 3

/-- The number of calories burned per stair in the second half -/
def calories_per_stair_second_half : ℕ := 4

/-- The total number of stairs in the staircase -/
def total_stairs : ℕ := stairs_first_half + stairs_second_half

/-- Theorem stating the total calories burned by each player -/
theorem total_calories_burned :
  num_runs * (stairs_first_half * calories_per_stair_first_half +
              stairs_second_half * calories_per_stair_second_half) = 9600 :=
by sorry

end NUMINAMATH_CALUDE_total_calories_burned_l161_16103


namespace NUMINAMATH_CALUDE_division_line_exists_l161_16175

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given 2000 distinct points in a plane, there exists a line that divides them into two equal sets -/
theorem division_line_exists (points : Finset Point) (h : points.card = 2000) :
  ∃ (a : ℝ), (points.filter (λ p => p.x < a)).card = 1000 ∧ (points.filter (λ p => p.x > a)).card = 1000 := by
  sorry

end NUMINAMATH_CALUDE_division_line_exists_l161_16175


namespace NUMINAMATH_CALUDE_allocation_theorem_l161_16182

/-- Represents the number of students -/
def num_students : ℕ := 5

/-- Represents the number of groups -/
def num_groups : ℕ := 3

/-- Function to calculate the number of allocation methods -/
def allocation_methods (n : ℕ) (k : ℕ) (excluded_pair : Bool) : ℕ :=
  sorry

/-- Theorem stating the number of allocation methods -/
theorem allocation_theorem :
  allocation_methods num_students num_groups true = 114 :=
sorry

end NUMINAMATH_CALUDE_allocation_theorem_l161_16182


namespace NUMINAMATH_CALUDE_quadratic_inequality_l161_16139

theorem quadratic_inequality (a b c : ℝ) 
  (h1 : 4 * a - 2 * b + c > 0) 
  (h2 : a + b + c < 0) : 
  b^2 > a * c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l161_16139


namespace NUMINAMATH_CALUDE_same_score_probability_l161_16106

theorem same_score_probability (p_A p_B : ℝ) 
  (h_A : p_A = 0.6) 
  (h_B : p_B = 0.8) 
  (h_independent : True) -- Representing independence of events
  (h_score : ℕ → ℝ) 
  (h_score_success : h_score 1 = 2) 
  (h_score_fail : h_score 0 = 0) : 
  p_A * p_B + (1 - p_A) * (1 - p_B) = 0.56 := by
sorry

end NUMINAMATH_CALUDE_same_score_probability_l161_16106


namespace NUMINAMATH_CALUDE_jennifer_pears_l161_16122

/-- Proves that Jennifer initially had 10 pears given the problem conditions -/
theorem jennifer_pears : ∃ P : ℕ, 
  (P + 20 + 2*P) - 6 = 44 ∧ P = 10 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_pears_l161_16122


namespace NUMINAMATH_CALUDE_inequality_range_l161_16136

theorem inequality_range (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∀ t : ℝ, (∀ x y : ℝ, a * x^2 + t * y^2 ≥ (a * x + t * y)^2) ↔ (0 ≤ t ∧ t ≤ 1 - a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l161_16136


namespace NUMINAMATH_CALUDE_raw_materials_cost_l161_16105

/-- The amount spent on raw materials given the total amount, machinery cost, and cash percentage. -/
theorem raw_materials_cost (total : ℝ) (machinery : ℝ) (cash_percent : ℝ) 
  (h1 : total = 1000)
  (h2 : machinery = 400)
  (h3 : cash_percent = 0.1)
  (h4 : ∃ (raw_materials : ℝ), raw_materials + machinery + cash_percent * total = total) :
  ∃ (raw_materials : ℝ), raw_materials = 500 := by
sorry

end NUMINAMATH_CALUDE_raw_materials_cost_l161_16105


namespace NUMINAMATH_CALUDE_one_ton_equals_2200_pounds_l161_16126

/-- Represents the weight of a packet in pounds -/
def packet_weight : ℚ := 16 + 4 / 16

/-- Represents the number of packets -/
def num_packets : ℕ := 1760

/-- Represents the capacity of the gunny bag in tons -/
def bag_capacity : ℕ := 13

/-- Theorem stating that one ton equals 2200 pounds -/
theorem one_ton_equals_2200_pounds :
  (num_packets : ℚ) * packet_weight / (bag_capacity : ℚ) = 2200 := by
  sorry

end NUMINAMATH_CALUDE_one_ton_equals_2200_pounds_l161_16126


namespace NUMINAMATH_CALUDE_network_sum_is_132_l161_16101

/-- Represents a network of interconnected circles with integers --/
structure Network where
  size : Nat
  sum_of_ends : Nat
  given_numbers : Fin 2 → Nat

/-- The total sum of all integers in a completed network --/
def total_sum (n : Network) : Nat :=
  n.size * (n.given_numbers 0 + n.given_numbers 1) / 2

/-- Theorem stating the total sum of the specific network described in the problem --/
theorem network_sum_is_132 (n : Network) 
  (h_size : n.size = 24)
  (h_given : n.given_numbers = ![4, 7]) :
  total_sum n = 132 := by
  sorry

#eval total_sum { size := 24, sum_of_ends := 11, given_numbers := ![4, 7] }

end NUMINAMATH_CALUDE_network_sum_is_132_l161_16101


namespace NUMINAMATH_CALUDE_temperature_at_noon_l161_16154

/-- Given the lowest temperature of a day and the fact that the temperature at noon
    is 10°C higher, this theorem proves the temperature at noon. -/
theorem temperature_at_noon (a : ℝ) : 
  let lowest_temp := a
  let temp_diff := 10
  lowest_temp + temp_diff = a + 10 := by sorry

end NUMINAMATH_CALUDE_temperature_at_noon_l161_16154


namespace NUMINAMATH_CALUDE_quadratic_factor_implies_n_l161_16196

theorem quadratic_factor_implies_n (n : ℤ) : 
  (∃ k : ℤ, ∀ x : ℤ, x^2 + 7*x + n = (x + 5) * (x + k)) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factor_implies_n_l161_16196


namespace NUMINAMATH_CALUDE_x_value_proof_l161_16131

theorem x_value_proof (x : Real) 
  (h1 : Real.sin (π / 2 - x) = -Real.sqrt 3 / 2)
  (h2 : π < x ∧ x < 2 * π) : 
  x = 7 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l161_16131


namespace NUMINAMATH_CALUDE_equal_inclination_l161_16134

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (hf : Continuous f ∧ ContinuousDeriv f)

-- Define points A, B, and P
variable (A B P : ℝ × ℝ)

-- Define that A and B are on the curve
variable (hA : A.2 = f A.1)
variable (hB : B.2 = f B.1)

-- Define that P is on the curve
variable (hP : P.2 = f P.1)

-- Define that P is between A and B
variable (hAP : A.1 ≤ P.1)
variable (hPB : P.1 ≤ B.1)

-- Define that the arc AB is concave to the chord AB
variable (hConcave : ∀ x ∈ Set.Icc A.1 B.1, f x ≤ (B.2 - A.2) / (B.1 - A.1) * (x - A.1) + A.2)

-- Define that AP + PB is maximal at P
variable (hMaximal : ∀ Q : ℝ × ℝ, Q.2 = f Q.1 → A.1 ≤ Q.1 → Q.1 ≤ B.1 → 
  Real.sqrt ((A.1 - Q.1)^2 + (A.2 - Q.2)^2) + Real.sqrt ((B.1 - Q.1)^2 + (B.2 - Q.2)^2) ≤
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) + Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2))

-- State the theorem
theorem equal_inclination :
  let tangent_slope := deriv f P.1
  let PA_slope := (A.2 - P.2) / (A.1 - P.1)
  let PB_slope := (B.2 - P.2) / (B.1 - P.1)
  abs ((tangent_slope - PA_slope) / (1 + tangent_slope * PA_slope)) =
  abs ((PB_slope - tangent_slope) / (1 + PB_slope * tangent_slope)) :=
by sorry

end NUMINAMATH_CALUDE_equal_inclination_l161_16134


namespace NUMINAMATH_CALUDE_system_solutions_l161_16188

/-- System of equations -/
def system (x y z p : ℝ) : Prop :=
  x^2 - 3*y + p = z ∧ y^2 - 3*z + p = x ∧ z^2 - 3*x + p = y

theorem system_solutions :
  (∀ p : ℝ, p = 4 → ∀ x y z : ℝ, system x y z p → x = 2 ∧ y = 2 ∧ z = 2) ∧
  (∀ p : ℝ, 1 < p ∧ p < 4 → ∀ x y z : ℝ, system x y z p → x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l161_16188


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l161_16151

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_nth_term
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 3)
  (h3 : ∃ n : ℕ, a n = 50) :
  ∃ n : ℕ, n = 17 ∧ a n = 50 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l161_16151


namespace NUMINAMATH_CALUDE_range_of_x_l161_16119

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 < 0
def q (x : ℝ) : Prop := 1/(x-2) < 0

-- State the theorem
theorem range_of_x (x : ℝ) : p x ∧ q x ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l161_16119


namespace NUMINAMATH_CALUDE_joeys_rope_length_l161_16153

/-- Given that the ratio of Joey's rope length to Chad's rope length is 8:3,
    and Chad's rope is 21 cm long, prove that Joey's rope is 56 cm long. -/
theorem joeys_rope_length (chad_rope_length : ℝ) (ratio : ℚ) :
  chad_rope_length = 21 →
  ratio = 8 / 3 →
  ∃ joey_rope_length : ℝ,
    joey_rope_length / chad_rope_length = ratio ∧
    joey_rope_length = 56 :=
by sorry

end NUMINAMATH_CALUDE_joeys_rope_length_l161_16153


namespace NUMINAMATH_CALUDE_perpendicular_lines_theorem_l161_16178

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v1 v2 : Vector2D) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Perpendicularity of two 2D vectors -/
def perpendicular (v1 v2 : Vector2D) : Prop := dot_product v1 v2 = 0

theorem perpendicular_lines_theorem (b c : ℝ) :
  let v1 : Vector2D := ⟨4, 1⟩
  let v2 : Vector2D := ⟨b, -8⟩
  let v3 : Vector2D := ⟨5, c⟩
  perpendicular v1 v3 ∧ perpendicular v2 v3 → b = 2 ∧ c = -20 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_theorem_l161_16178


namespace NUMINAMATH_CALUDE_tile_placement_theorem_l161_16109

/-- Represents a rectangular grid --/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangular tile --/
structure Tile :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the maximum number of tiles that can be placed in a grid --/
def max_tiles (g : Grid) (t : Tile) : ℕ :=
  sorry

/-- Calculates the number of cells left unpaved --/
def unpaved_cells (g : Grid) (t : Tile) : ℕ :=
  sorry

theorem tile_placement_theorem (g : Grid) (t : Tile) : 
  g.rows = 14 ∧ g.cols = 14 ∧ t.width = 1 ∧ t.height = 4 →
  max_tiles g t = 48 ∧ unpaved_cells g t = 4 :=
sorry

end NUMINAMATH_CALUDE_tile_placement_theorem_l161_16109


namespace NUMINAMATH_CALUDE_min_a_for_p_geq_half_l161_16195

def p (a : ℕ) : ℚ :=
  (Nat.choose (36 - a) 2 + Nat.choose (a - 1) 2) / Nat.choose 50 2

theorem min_a_for_p_geq_half :
  ∀ a : ℕ, 1 ≤ a ∧ a ≤ 37 → p a < 1/2 ∧ p 38 ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_min_a_for_p_geq_half_l161_16195


namespace NUMINAMATH_CALUDE_students_taking_no_subjects_l161_16186

theorem students_taking_no_subjects (total : ℕ) (math physics chemistry : ℕ) 
  (math_physics math_chemistry physics_chemistry : ℕ) (all_three : ℕ) 
  (h1 : total = 60)
  (h2 : math = 40)
  (h3 : physics = 30)
  (h4 : chemistry = 25)
  (h5 : math_physics = 18)
  (h6 : physics_chemistry = 10)
  (h7 : math_chemistry = 12)
  (h8 : all_three = 5) :
  total - ((math + physics + chemistry) - (math_physics + physics_chemistry + math_chemistry) + all_three) = 5 := by
sorry


end NUMINAMATH_CALUDE_students_taking_no_subjects_l161_16186


namespace NUMINAMATH_CALUDE_final_positions_l161_16132

structure Person where
  cash : Int
  has_car : Bool

def initial_c : Person := { cash := 15000, has_car := true }
def initial_d : Person := { cash := 17000, has_car := false }

def transaction1 (c d : Person) : Person × Person :=
  ({ cash := c.cash + 16000, has_car := false },
   { cash := d.cash - 16000, has_car := true })

def transaction2 (c d : Person) : Person × Person :=
  ({ cash := c.cash - 14000, has_car := true },
   { cash := d.cash + 14000, has_car := false })

def transaction3 (c d : Person) : Person × Person :=
  ({ cash := c.cash + 15500, has_car := false },
   { cash := d.cash - 15500, has_car := true })

theorem final_positions :
  let (c1, d1) := transaction1 initial_c initial_d
  let (c2, d2) := transaction2 c1 d1
  let (c3, d3) := transaction3 c2 d2
  c3.cash = 32500 ∧ ¬c3.has_car ∧ d3.cash = -500 ∧ d3.has_car := by
  sorry

end NUMINAMATH_CALUDE_final_positions_l161_16132


namespace NUMINAMATH_CALUDE_quadratic_root_l161_16145

theorem quadratic_root (a b c : ℝ) : 
  (∃ d : ℝ, d ≥ 0 ∧ b = a + d ∧ c = a + 2*d) →  -- arithmetic sequence
  a ≤ b → b ≤ c → c ≤ 10 → c > 0 →  -- given inequalities
  (∃! x : ℝ, a*x^2 + b*x + c = 0) →  -- exactly one root
  (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ x = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_l161_16145


namespace NUMINAMATH_CALUDE_total_weight_in_kg_l161_16146

-- Define the weight of one envelope in grams
def envelope_weight : ℝ := 8.5

-- Define the number of envelopes
def num_envelopes : ℕ := 850

-- Define the conversion factor from grams to kilograms
def grams_to_kg : ℝ := 1000

-- Theorem statement
theorem total_weight_in_kg :
  (envelope_weight * num_envelopes) / grams_to_kg = 7.225 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_in_kg_l161_16146


namespace NUMINAMATH_CALUDE_fifteen_people_in_house_l161_16148

/-- The number of people in a house --/
def num_people_in_house (initial_bedroom : ℕ) (entered_bedroom : ℕ) (living_room : ℕ) : ℕ :=
  initial_bedroom + entered_bedroom + living_room

/-- Theorem: Given the initial conditions, there are 15 people in the house --/
theorem fifteen_people_in_house :
  num_people_in_house 2 5 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_people_in_house_l161_16148


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l161_16113

/-- The time required for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) 
  (h1 : jogger_speed = 9 * 5 / 18) -- Convert 9 kmph to m/s
  (h2 : train_speed = 45 * 5 / 18) -- Convert 45 kmph to m/s
  (h3 : train_length = 120)
  (h4 : initial_distance = 270) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 39 := by
sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l161_16113


namespace NUMINAMATH_CALUDE_opposite_solutions_l161_16120

theorem opposite_solutions (k : ℝ) : k = 7 →
  ∃ (x y : ℝ), x = -y ∧ 
  (3 * (2 * x - 1) = 1 - 2 * x) ∧
  (8 - k = 2 * (y + 1)) := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_l161_16120


namespace NUMINAMATH_CALUDE_cars_on_remaining_days_l161_16129

/-- Represents the number of cars passing through a toll booth on different days of the week -/
structure TollBoothTraffic where
  total : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  remaining_days : ℕ

/-- Theorem stating the number of cars passing through the toll booth on each remaining day -/
theorem cars_on_remaining_days (t : TollBoothTraffic) : 
  t.total = 450 ∧ 
  t.monday = 50 ∧ 
  t.tuesday = 50 ∧ 
  t.wednesday = 2 * t.monday ∧ 
  t.thursday = 2 * t.monday ∧ 
  t.remaining_days * 3 = t.total - (t.monday + t.tuesday + t.wednesday + t.thursday) → 
  t.remaining_days = 50 := by
  sorry


end NUMINAMATH_CALUDE_cars_on_remaining_days_l161_16129


namespace NUMINAMATH_CALUDE_function_zeros_theorem_l161_16118

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp x - k * x + k

theorem function_zeros_theorem (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : f k x₁ = 0) 
  (h2 : f k x₂ = 0) 
  (h3 : x₁ ≠ x₂) : 
  k > Real.exp 2 ∧ x₁ + x₂ > 4 := by
  sorry

end NUMINAMATH_CALUDE_function_zeros_theorem_l161_16118


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l161_16197

theorem min_value_theorem (x : ℝ) (h : x > 3) : x + 4 / (x - 3) ≥ 7 :=
sorry

theorem equality_condition (x : ℝ) (h : x > 3) : 
  x + 4 / (x - 3) = 7 ↔ x = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l161_16197


namespace NUMINAMATH_CALUDE_wilfred_wednesday_carrots_l161_16174

/-- The number of carrots Wilfred ate on Tuesday -/
def tuesday_carrots : ℕ := 4

/-- The number of carrots Wilfred ate on Thursday -/
def thursday_carrots : ℕ := 5

/-- The total number of carrots Wilfred wants to eat from Tuesday to Thursday -/
def total_carrots : ℕ := 15

/-- The number of carrots Wilfred ate on Wednesday -/
def wednesday_carrots : ℕ := total_carrots - tuesday_carrots - thursday_carrots

theorem wilfred_wednesday_carrots : wednesday_carrots = 6 := by
  sorry

end NUMINAMATH_CALUDE_wilfred_wednesday_carrots_l161_16174


namespace NUMINAMATH_CALUDE_track_circumference_l161_16185

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (v1 v2 t : ℝ) (h1 : v1 = 4.5) (h2 : v2 = 3.75) (h3 : t = 4.8 / 60) :
  2 * (v1 * t + v2 * t) = 1.32 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_l161_16185


namespace NUMINAMATH_CALUDE_parabola_max_sum_l161_16172

/-- Given a parabola y = -x^2 - 3x + 3 and a point P(m, n) on this parabola,
    the maximum value of m + n is 4. -/
theorem parabola_max_sum (m n : ℝ) : 
  n = -m^2 - 3*m + 3 → (∀ x y : ℝ, y = -x^2 - 3*x + 3 → m + n ≥ x + y) → m + n = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_sum_l161_16172


namespace NUMINAMATH_CALUDE_sports_club_participation_l161_16190

theorem sports_club_participation (total students_swimming students_basketball students_both : ℕ) 
  (h1 : total = 75)
  (h2 : students_swimming = 46)
  (h3 : students_basketball = 34)
  (h4 : students_both = 22) :
  total - (students_swimming + students_basketball - students_both) = 17 := by
sorry

end NUMINAMATH_CALUDE_sports_club_participation_l161_16190


namespace NUMINAMATH_CALUDE_language_learning_hours_difference_l161_16108

def hours_english : ℝ := 2
def hours_chinese : ℝ := 5
def hours_spanish : ℝ := 4
def hours_french : ℝ := 3
def hours_german : ℝ := 1.5

theorem language_learning_hours_difference : 
  (hours_chinese + hours_french) - (hours_german + hours_spanish) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_language_learning_hours_difference_l161_16108


namespace NUMINAMATH_CALUDE_equation_solution_l161_16125

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  x > 0 ∧ Real.sqrt ((log10 x)^2 + log10 (x^2) + 1) + log10 x + 1 = 0

-- Theorem statement
theorem equation_solution :
  ∀ x : ℝ, equation x ↔ (0 < x ∧ x ≤ (1/10)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l161_16125


namespace NUMINAMATH_CALUDE_cost_price_is_fifty_l161_16194

/-- The cost price per meter of cloth given the total meters sold, total selling price, and loss per meter. -/
def cost_price_per_meter (total_meters : ℕ) (total_selling_price : ℕ) (loss_per_meter : ℕ) : ℚ :=
  (total_selling_price + total_meters * loss_per_meter : ℚ) / total_meters

/-- Theorem stating that the cost price per meter of cloth is $50 given the problem conditions. -/
theorem cost_price_is_fifty :
  cost_price_per_meter 400 18000 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_fifty_l161_16194


namespace NUMINAMATH_CALUDE_sin_405_degrees_l161_16111

theorem sin_405_degrees (h : 405 = 360 + 45) : Real.sin (405 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_405_degrees_l161_16111


namespace NUMINAMATH_CALUDE_binomial_even_iff_power_of_two_l161_16104

theorem binomial_even_iff_power_of_two (n : ℕ) : 
  (∃ m : ℕ, n = 2^m) ↔ 
  (∀ k : ℕ, 1 ≤ k ∧ k < n → Even (n.choose k)) := by
sorry

end NUMINAMATH_CALUDE_binomial_even_iff_power_of_two_l161_16104


namespace NUMINAMATH_CALUDE_abs_negative_two_thousand_l161_16150

theorem abs_negative_two_thousand : |(-2000 : ℤ)| = 2000 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_two_thousand_l161_16150


namespace NUMINAMATH_CALUDE_factors_of_x4_plus_81_l161_16147

theorem factors_of_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 3*x + 9) * (x^2 - 3*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factors_of_x4_plus_81_l161_16147


namespace NUMINAMATH_CALUDE_system_solution_l161_16179

theorem system_solution : 
  ∃ (x y : ℝ), (3 * x^2 + 2 * y^2 + 2 * x + 3 * y = 0 ∧
                4 * x^2 - 3 * y^2 - 3 * x + 4 * y = 0) ↔
               ((x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l161_16179


namespace NUMINAMATH_CALUDE_parabola_focus_l161_16121

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := x = 4 * y^2

-- Define the focus of a parabola
def focus (p : ℝ × ℝ) (parabola : (ℝ × ℝ → Prop)) : Prop :=
  ∃ (a : ℝ), parabola = λ (x, y) => y^2 = a * (x - p.1) ∧ p.2 = 0

-- Theorem statement
theorem parabola_focus :
  focus (1/16, 0) (λ (x, y) => parabola_equation x y) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l161_16121


namespace NUMINAMATH_CALUDE_f_even_and_decreasing_l161_16141

-- Define the function f(x) = -x² + 1
def f (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem f_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_decreasing_l161_16141


namespace NUMINAMATH_CALUDE_problem_solution_l161_16199

-- Define proposition p
def p : Prop := ∀ a : ℝ, ∃ x : ℝ, a^(x + 1) = 1 ∧ x = 0

-- Define proposition q
def q : Prop := ∀ f : ℝ → ℝ, (∀ x : ℝ, f x = f (-x)) → 
  (∀ x : ℝ, f (x + 1) = f (2 - x))

-- Theorem statement
theorem problem_solution : (¬p ∧ ¬q) → (p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l161_16199
