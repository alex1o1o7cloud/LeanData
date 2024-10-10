import Mathlib

namespace lcm_gcd_220_126_l1041_104129

theorem lcm_gcd_220_126 :
  (Nat.lcm 220 126 = 13860) ∧ (Nat.gcd 220 126 = 2) := by
  sorry

end lcm_gcd_220_126_l1041_104129


namespace min_cos_C_in_triangle_l1041_104109

theorem min_cos_C_in_triangle (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sin : Real.sin A + 2 * Real.sin B = 3 * Real.sin C) : 
  (2 * Real.sqrt 10 - 2) / 9 ≤ Real.cos C :=
sorry

end min_cos_C_in_triangle_l1041_104109


namespace scientific_notation_correct_l1041_104114

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 393000

/-- The scientific notation representation of the number -/
def scientific_form : ScientificNotation :=
  { coefficient := 3.93
    exponent := 5
    coefficient_range := by sorry }

/-- Theorem stating that the scientific notation is correct -/
theorem scientific_notation_correct :
  (scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent) = number := by sorry

end scientific_notation_correct_l1041_104114


namespace base_c_problem_l1041_104184

/-- Representation of a number in base c -/
def baseC (n : ℕ) (c : ℕ) : ℕ → ℕ
| 0 => n % c
| i + 1 => baseC (n / c) c i

/-- Given that in base c, 33_c squared equals 1201_c, prove that c = 10 -/
theorem base_c_problem (c : ℕ) (h : c > 1) :
  (baseC 33 c 1 * c + baseC 33 c 0)^2 = 
  baseC 1201 c 3 * c^3 + baseC 1201 c 2 * c^2 + baseC 1201 c 1 * c + baseC 1201 c 0 →
  c = 10 := by
sorry

end base_c_problem_l1041_104184


namespace geometric_sequence_increasing_iff_l1041_104195

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_increasing_iff (a : ℕ → ℝ) 
  (h : is_geometric_sequence a) :
  (a 1 < a 2 ∧ a 2 < a 3) ↔ is_increasing_sequence a :=
sorry

end geometric_sequence_increasing_iff_l1041_104195


namespace battleship_max_ships_l1041_104127

/-- Represents a game board --/
structure Board :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a ship --/
structure Ship :=
  (length : Nat)
  (width : Nat)

/-- Calculates the maximum number of ships that can be placed on a board --/
def maxShips (board : Board) (ship : Ship) : Nat :=
  (board.rows * board.cols) / (ship.length * ship.width)

theorem battleship_max_ships :
  let board : Board := ⟨10, 10⟩
  let ship : Ship := ⟨4, 1⟩
  maxShips board ship = 25 := by
  sorry

#eval maxShips ⟨10, 10⟩ ⟨4, 1⟩

end battleship_max_ships_l1041_104127


namespace class_size_quotient_l1041_104187

theorem class_size_quotient (N H J : ℝ) 
  (h1 : N / H = 1.2) 
  (h2 : H / J = 5/6) : 
  N / J = 1 := by
  sorry

end class_size_quotient_l1041_104187


namespace quadratic_inequality_range_l1041_104157

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) ≤ 0) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end quadratic_inequality_range_l1041_104157


namespace arithmetic_mean_theorem_l1041_104171

theorem arithmetic_mean_theorem (x a b : ℝ) (hx : x ≠ 0) (hb : b ≠ 0) :
  (1 / 2) * ((x * b + a) / x + (x * b - a) / x) = b := by
  sorry

end arithmetic_mean_theorem_l1041_104171


namespace perfect_squares_as_sum_of_powers_of_two_l1041_104136

theorem perfect_squares_as_sum_of_powers_of_two (n a b : ℕ) (h1 : a ≥ b) (h2 : n^2 = 2^a + 2^b) :
  ∃ k : ℕ, n^2 = 4^(k+1) ∨ n^2 = 9 * 4^k :=
sorry

end perfect_squares_as_sum_of_powers_of_two_l1041_104136


namespace base7_product_l1041_104158

/-- Converts a base 7 number represented as a list of digits to its decimal (base 10) equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 7 * acc) 0

/-- Converts a decimal (base 10) number to its base 7 representation as a list of digits -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- The problem statement -/
theorem base7_product : 
  toBase7 (toDecimal [1,3,2,4] * toDecimal [2,3]) = [3,1,4,1,5] := by
  sorry

end base7_product_l1041_104158


namespace simplify_expression_l1041_104118

theorem simplify_expression : (5^7 + 2^8) * (1^5 - (-1)^5)^10 = 80263680 := by
  sorry

end simplify_expression_l1041_104118


namespace hockey_league_games_l1041_104167

/-- The number of teams in the hockey league -/
def num_teams : ℕ := 15

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- Calculates the total number of games in the season -/
def total_games : ℕ := (num_teams * (num_teams - 1) / 2) * games_per_pair

/-- Theorem: The total number of games in the season is 1050 -/
theorem hockey_league_games : total_games = 1050 := by
  sorry

end hockey_league_games_l1041_104167


namespace floor_breadth_correct_l1041_104151

/-- The length of the rectangular floor in meters -/
def floor_length : ℝ := 16.25

/-- The number of square tiles required to cover the floor -/
def number_of_tiles : ℕ := 3315

/-- The breadth of the rectangular floor in meters -/
def floor_breadth : ℝ := 204

/-- Theorem stating that the given breadth is correct for the rectangular floor -/
theorem floor_breadth_correct : 
  floor_length * floor_breadth = (number_of_tiles : ℝ) := by sorry

end floor_breadth_correct_l1041_104151


namespace second_discount_percentage_l1041_104133

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 400 →
  first_discount = 12 →
  final_price = 334.4 →
  ∃ (second_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 5 := by
  sorry

end second_discount_percentage_l1041_104133


namespace prob_all_even_is_one_tenth_and_half_l1041_104123

/-- Represents a die with a given number of sides -/
structure Die :=
  (sides : ℕ)
  (sides_pos : sides > 0)

/-- The number of even outcomes on a die -/
def evenOutcomes (d : Die) : ℕ :=
  d.sides / 2

/-- The probability of rolling an even number on a die -/
def probEven (d : Die) : ℚ :=
  evenOutcomes d / d.sides

/-- The three dice in the problem -/
def die1 : Die := ⟨6, by norm_num⟩
def die2 : Die := ⟨7, by norm_num⟩
def die3 : Die := ⟨9, by norm_num⟩

/-- The theorem to be proved -/
theorem prob_all_even_is_one_tenth_and_half :
  probEven die1 * probEven die2 * probEven die3 = 1 / (10 : ℚ) + 1 / (20 : ℚ) :=
sorry

end prob_all_even_is_one_tenth_and_half_l1041_104123


namespace delta_value_l1041_104132

theorem delta_value (Δ : ℤ) : 4 * (-3) = Δ + 3 → Δ = -15 := by
  sorry

end delta_value_l1041_104132


namespace symmetric_function_property_l1041_104146

def symmetricAround (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y ↔ f y = x

def symmetricAfterShift (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + 1) = y ↔ f y = x

theorem symmetric_function_property (f : ℝ → ℝ)
  (h1 : symmetricAround f)
  (h2 : symmetricAfterShift f)
  (h3 : f 1 = 0) :
  f 2011 = -2010 := by
  sorry

end symmetric_function_property_l1041_104146


namespace inequality_proof_l1041_104141

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end inequality_proof_l1041_104141


namespace three_two_zero_zero_properties_l1041_104180

/-- Represents a number with its decimal representation -/
structure DecimalNumber where
  value : ℝ
  representation : String

/-- Counts the number of significant figures in a decimal representation -/
def countSignificantFigures (n : DecimalNumber) : ℕ :=
  sorry

/-- Determines the precision of a decimal representation -/
def getPrecision (n : DecimalNumber) : String :=
  sorry

/-- The main theorem about the number 3200 -/
theorem three_two_zero_zero_properties :
  let n : DecimalNumber := ⟨3200, "0.320"⟩
  countSignificantFigures n = 3 ∧ getPrecision n = "thousandth" := by
  sorry

end three_two_zero_zero_properties_l1041_104180


namespace combined_mean_l1041_104120

theorem combined_mean (set1_count : ℕ) (set1_mean : ℝ) (set2_count : ℕ) (set2_mean : ℝ) :
  set1_count = 8 →
  set2_count = 10 →
  set1_mean = 17 →
  set2_mean = 23 →
  let total_count := set1_count + set2_count
  let combined_mean := (set1_count * set1_mean + set2_count * set2_mean) / total_count
  combined_mean = (8 * 17 + 10 * 23) / 18 :=
by
  sorry

end combined_mean_l1041_104120


namespace a_positive_sufficient_not_necessary_for_abs_a_positive_l1041_104106

theorem a_positive_sufficient_not_necessary_for_abs_a_positive :
  (∃ a : ℝ, (a > 0 → abs a > 0) ∧ ¬(abs a > 0 → a > 0)) := by
  sorry

end a_positive_sufficient_not_necessary_for_abs_a_positive_l1041_104106


namespace negate_all_guitarists_proficient_l1041_104162

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Musician : U → Prop)
variable (Guitarist : U → Prop)
variable (ProficientViolinist : U → Prop)

-- Theorem statement
theorem negate_all_guitarists_proficient :
  (∃ x, Guitarist x ∧ ¬ProficientViolinist x) ↔ 
  ¬(∀ x, Guitarist x → ProficientViolinist x) :=
by sorry

end negate_all_guitarists_proficient_l1041_104162


namespace waiter_earnings_theorem_l1041_104147

/-- Calculates the total earnings for the first four nights of a five-day work week,
    given the target average per night and the required earnings for the last night. -/
def earnings_first_four_nights (days_per_week : ℕ) (target_average : ℚ) (last_night_earnings : ℚ) : ℚ :=
  days_per_week * target_average - last_night_earnings

theorem waiter_earnings_theorem :
  earnings_first_four_nights 5 50 115 = 135 := by
  sorry

end waiter_earnings_theorem_l1041_104147


namespace max_product_of_distances_l1041_104159

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_product_of_distances (P : ℝ × ℝ) (h : is_on_ellipse P.1 P.2) :
  ∃ (max : ℝ), max = 8 ∧ ∀ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 →
    distance Q F1 * distance Q F2 ≤ max :=
sorry

end max_product_of_distances_l1041_104159


namespace senegal_total_points_l1041_104130

-- Define the point values for victory and draw
def victory_points : ℕ := 3
def draw_points : ℕ := 1

-- Define Senegal's match results
def senegal_victories : ℕ := 1
def senegal_draws : ℕ := 2

-- Define the function to calculate total points
def calculate_points (victories draws : ℕ) : ℕ :=
  victories * victory_points + draws * draw_points

-- Theorem to prove
theorem senegal_total_points :
  calculate_points senegal_victories senegal_draws = 5 := by
  sorry

end senegal_total_points_l1041_104130


namespace quadratic_roots_theorem_l1041_104193

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given quadratic equation x^2 - 6x + 2m - 1 = 0 with two real roots -/
def givenEquation (m : ℝ) : QuadraticEquation :=
  { a := 1, b := -6, c := 2 * m - 1 }

theorem quadratic_roots_theorem (m : ℝ) :
  let eq := givenEquation m
  let x₁ : ℝ := 1
  let x₂ : ℝ := 5
  (∃ (x₁ x₂ : ℝ), x₁^2 - 6*x₁ + 2*m - 1 = 0 ∧ x₂^2 - 6*x₂ + 2*m - 1 = 0) →
  x₁ = 1 →
  (x₂ = 5 ∧ m = 3) ∧
  (∃ m' : ℝ, (x₁ - 1) * (x₂ - 1) = 6 / (m' - 5) ∧ m' = 2) :=
by
  sorry


end quadratic_roots_theorem_l1041_104193


namespace greatest_c_for_no_real_solutions_l1041_104185

theorem greatest_c_for_no_real_solutions : 
  (∃ c : ℤ, c = (Nat.floor (Real.sqrt 116)) ∧ 
   ∀ x : ℝ, x^2 + c*x + 29 ≠ 0 ∧
   ∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 29 = 0) ∧
  (Nat.floor (Real.sqrt 116) = 10) :=
by sorry

end greatest_c_for_no_real_solutions_l1041_104185


namespace sufficient_not_necessary_subset_condition_l1041_104128

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x | -1 - 2*a ≤ x ∧ x ≤ a - 2}

-- Statement for part (1)
theorem sufficient_not_necessary (a : ℝ) :
  (∀ x, x ∈ A → x ∈ B a) ∧ (∃ x, x ∈ B a ∧ x ∉ A) ↔ a ≥ 7 :=
sorry

-- Statement for part (2)
theorem subset_condition (a : ℝ) :
  B a ⊆ A ↔ a < 1/3 :=
sorry

end sufficient_not_necessary_subset_condition_l1041_104128


namespace lower_selling_price_l1041_104153

/-- Proves that the lower selling price is 340 given the conditions of the problem -/
theorem lower_selling_price (cost_price selling_price : ℕ) :
  cost_price = 250 →
  selling_price = 350 →
  (selling_price - cost_price : ℚ) / cost_price = 
    ((340 - cost_price : ℚ) / cost_price) + 4 / 100 →
  340 = (selling_price - cost_price) * 100 / 104 + cost_price :=
by sorry

end lower_selling_price_l1041_104153


namespace inequality_solution_set_l1041_104169

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (a * x) / (x - 1) < (a - 1) / (x - 1)) ↔
  (a > 0 ∧ (a - 1) / a < x ∧ x < 1) ∨
  (a = 0 ∧ x < 1) ∨
  (a < 0 ∧ (x > (a - 1) / a ∨ x < 1)) :=
by sorry

end inequality_solution_set_l1041_104169


namespace regression_correction_l1041_104190

/-- Represents a data point with x and y coordinates -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Represents a linear regression equation -/
structure RegressionEquation where
  slope : ℝ
  intercept : ℝ

/-- Represents the center of sample points -/
structure SampleCenter where
  x : ℝ
  y : ℝ

theorem regression_correction (data : List DataPoint) 
  (initial_eq : RegressionEquation) 
  (initial_center : SampleCenter)
  (incorrect_point1 incorrect_point2 correct_point1 correct_point2 : DataPoint)
  (corrected_slope : ℝ)
  (h1 : data.length = 8)
  (h2 : initial_eq.slope = 2 ∧ initial_eq.intercept = 5)
  (h3 : initial_center.x = 2)
  (h4 : incorrect_point1 = ⟨7, 3⟩ ∧ correct_point1 = ⟨3, 7⟩)
  (h5 : incorrect_point2 = ⟨4, -6⟩ ∧ correct_point2 = ⟨4, 6⟩)
  (h6 : corrected_slope = 13/3) :
  ∃ k : ℝ, k = 9/2 ∧ 
    ∀ x y : ℝ, y = corrected_slope * x + k → 
      ∃ center : SampleCenter, center.x = 3/2 ∧ center.y = 11 ∧
        y = corrected_slope * center.x + k := by
  sorry

end regression_correction_l1041_104190


namespace gcd_1987_2025_l1041_104175

theorem gcd_1987_2025 : Nat.gcd 1987 2025 = 1 := by
  sorry

end gcd_1987_2025_l1041_104175


namespace intersection_of_A_and_B_l1041_104131

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 0 ≤ x}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_of_A_and_B_l1041_104131


namespace total_subscription_is_50000_l1041_104110

/-- Represents the subscription amounts and profit distribution for a business venture -/
structure BusinessSubscription where
  /-- Subscription amount for C -/
  c : ℕ
  /-- Total profit -/
  totalProfit : ℕ
  /-- A's share of the profit -/
  aProfit : ℕ

/-- Calculates the total subscription amount based on the given conditions -/
def totalSubscription (bs : BusinessSubscription) : ℕ :=
  3 * bs.c + 14000

/-- Theorem stating that the total subscription amount is 50000 given the problem conditions -/
theorem total_subscription_is_50000 (bs : BusinessSubscription) 
  (h1 : bs.totalProfit = 70000)
  (h2 : bs.aProfit = 29400)
  (h3 : bs.aProfit * (3 * bs.c + 14000) = bs.totalProfit * (bs.c + 9000)) :
  totalSubscription bs = 50000 := by
  sorry

end total_subscription_is_50000_l1041_104110


namespace coin_division_problem_l1041_104117

theorem coin_division_problem : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 ∧ m % 8 = 6 ∧ m % 7 = 5 → n ≤ m) ∧ 
  n % 8 = 6 ∧ 
  n % 7 = 5 ∧ 
  n % 9 = 0 := by
sorry

end coin_division_problem_l1041_104117


namespace complex_square_in_second_quadrant_l1041_104116

theorem complex_square_in_second_quadrant :
  let z : ℂ := (1/2 : ℝ) + (Real.sqrt 3/2 : ℝ) * Complex.I
  let w : ℂ := z^2
  (w.re < 0) ∧ (w.im > 0) := by sorry

end complex_square_in_second_quadrant_l1041_104116


namespace pancakes_and_honey_cost_l1041_104119

theorem pancakes_and_honey_cost (x y : ℕ) : 25 * x + 340 * y ≤ 2000 :=
by sorry

end pancakes_and_honey_cost_l1041_104119


namespace sufficient_condition_range_l1041_104148

theorem sufficient_condition_range (m : ℝ) : 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 3) → x ≤ m) → m ≥ 3 := by
  sorry

end sufficient_condition_range_l1041_104148


namespace expression_value_at_three_l1041_104145

theorem expression_value_at_three :
  let x : ℝ := 3
  x^5 - (5*x)^2 = 18 := by sorry

end expression_value_at_three_l1041_104145


namespace quadratic_negative_root_l1041_104115

theorem quadratic_negative_root (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ x^2 - 6*a*x - 2 + 2*a + 9*a^2 = 0) ↔ 
  a < (-1 + Real.sqrt 19) / 9 :=
by sorry

end quadratic_negative_root_l1041_104115


namespace perpendicular_parallel_implies_plane_parallel_plane_parallel_line_in_plane_implies_line_parallel_plane_l1041_104142

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Axioms
axiom distinct_lines (m n : Line) : m ≠ n
axiom non_coincident_planes (α β : Plane) : α ≠ β

-- Theorem 1
theorem perpendicular_parallel_implies_plane_parallel 
  (m n : Line) (α β : Plane) :
  perpendicular m α → perpendicular n β → parallel m n → 
  plane_parallel α β :=
sorry

-- Theorem 2
theorem plane_parallel_line_in_plane_implies_line_parallel_plane 
  (m : Line) (α β : Plane) :
  plane_parallel α β → contains α m → line_parallel_plane m β :=
sorry

end perpendicular_parallel_implies_plane_parallel_plane_parallel_line_in_plane_implies_line_parallel_plane_l1041_104142


namespace fraction_of_girls_l1041_104161

theorem fraction_of_girls (total : ℕ) (boys : ℕ) (h1 : total = 45) (h2 : boys = 30) :
  (total - boys : ℚ) / total = 1 / 3 := by
  sorry

end fraction_of_girls_l1041_104161


namespace book_pages_count_l1041_104111

theorem book_pages_count : 
  let days : ℕ := 10
  let first_four_days_avg : ℕ := 20
  let first_four_days_count : ℕ := 4
  let break_day_count : ℕ := 1
  let next_four_days_avg : ℕ := 30
  let next_four_days_count : ℕ := 4
  let last_day_pages : ℕ := 15
  (first_four_days_avg * first_four_days_count) + 
  (next_four_days_avg * next_four_days_count) + 
  last_day_pages = 215 := by
sorry

end book_pages_count_l1041_104111


namespace charles_pictures_l1041_104189

theorem charles_pictures (total_papers : ℕ) (pictures_before_work : ℕ) (pictures_after_work : ℕ) (papers_left : ℕ) :
  let total_yesterday := pictures_before_work + pictures_after_work
  let used_papers := total_papers - papers_left
  used_papers - total_yesterday = total_papers - papers_left - (pictures_before_work + pictures_after_work) :=
by sorry

end charles_pictures_l1041_104189


namespace equation_solution_l1041_104186

theorem equation_solution : 
  ∃ (S : Set ℝ), S = {x : ℝ | (x + 2)^4 + x^4 = 82} ∧ S = {-3, 1} := by
  sorry

end equation_solution_l1041_104186


namespace brick_height_proof_l1041_104188

/-- Proves that given a wall of specific dimensions and bricks of specific dimensions,
    if a certain number of bricks are used, then the height of each brick is 6 cm. -/
theorem brick_height_proof (wall_length wall_height wall_thickness : ℝ)
                           (brick_length brick_width brick_height : ℝ)
                           (num_bricks : ℝ) :
  wall_length = 8 →
  wall_height = 6 →
  wall_thickness = 0.02 →
  brick_length = 0.05 →
  brick_width = 0.11 →
  brick_height = 0.06 →
  num_bricks = 2909.090909090909 →
  brick_height * 100 = 6 := by
  sorry

end brick_height_proof_l1041_104188


namespace max_digits_product_5_4_l1041_104163

theorem max_digits_product_5_4 : 
  ∀ a b : ℕ, 
  10000 ≤ a ∧ a < 100000 → 
  1000 ≤ b ∧ b < 10000 → 
  a * b < 1000000000 :=
by sorry

end max_digits_product_5_4_l1041_104163


namespace mowers_for_three_hours_l1041_104196

/-- The number of mowers required to drink a barrel of kvass in a given time -/
def mowers_required (initial_mowers : ℕ) (initial_hours : ℕ) (target_hours : ℕ) : ℕ :=
  (initial_mowers * initial_hours) / target_hours

/-- Theorem stating that 16 mowers are required to drink a barrel of kvass in 3 hours,
    given that 6 mowers can drink it in 8 hours -/
theorem mowers_for_three_hours :
  mowers_required 6 8 3 = 16 := by
  sorry

end mowers_for_three_hours_l1041_104196


namespace minimum_value_of_function_l1041_104137

theorem minimum_value_of_function (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a^4 / (a^3 + b^2 + c^2)) + (b^4 / (b^3 + a^2 + c^2)) + (c^4 / (c^3 + b^2 + a^2)) ≥ 1/7 := by
  sorry

end minimum_value_of_function_l1041_104137


namespace optimal_seating_l1041_104174

/-- Represents the conference hall seating problem --/
def ConferenceSeating (total_chairs : ℕ) (chairs_per_row : ℕ) (attendees : ℕ) : Prop :=
  ∃ (chairs_to_remove : ℕ),
    let remaining_chairs := total_chairs - chairs_to_remove
    remaining_chairs % chairs_per_row = 0 ∧
    remaining_chairs ≥ attendees ∧
    ∀ (n : ℕ), n < chairs_to_remove →
      (total_chairs - n) % chairs_per_row ≠ 0 ∨
      (total_chairs - n) < attendees ∨
      (total_chairs - n) - attendees > remaining_chairs - attendees

theorem optimal_seating :
  ConferenceSeating 156 13 100 ∧
  (∃ (chairs_to_remove : ℕ), chairs_to_remove = 52 ∧
    let remaining_chairs := 156 - chairs_to_remove
    remaining_chairs % 13 = 0 ∧
    remaining_chairs ≥ 100 ∧
    ∀ (n : ℕ), n < chairs_to_remove →
      (156 - n) % 13 ≠ 0 ∨
      (156 - n) < 100 ∨
      (156 - n) - 100 > remaining_chairs - 100) :=
by sorry

end optimal_seating_l1041_104174


namespace marble_ratio_l1041_104107

/-- Represents the number of marbles of each color in a box -/
structure MarbleBox where
  red : ℕ
  green : ℕ
  yellow : ℕ
  other : ℕ

/-- Conditions for the marble box problem -/
def MarbleBoxConditions (box : MarbleBox) : Prop :=
  box.red = 20 ∧
  box.yellow = box.green / 5 ∧
  box.red + box.green + box.yellow + box.other = 3 * box.green ∧
  box.other = 88

theorem marble_ratio (box : MarbleBox) 
  (h : MarbleBoxConditions box) : 
  box.green = 3 * box.red := by
  sorry

end marble_ratio_l1041_104107


namespace louisa_average_speed_l1041_104166

/-- Proves that given the conditions of Louisa's travel, her average speed was 60 miles per hour -/
theorem louisa_average_speed :
  ∀ (v : ℝ), -- v represents the average speed in miles per hour
  v > 0 → -- speed is positive
  ∃ (t : ℝ), -- t represents the time for the 240-mile trip
  t > 0 → -- time is positive
  240 = v * t ∧ -- equation for the first day's travel
  420 = v * (t + 3) → -- equation for the second day's travel
  v = 60 := by
sorry

end louisa_average_speed_l1041_104166


namespace baseball_game_opponent_score_l1041_104172

theorem baseball_game_opponent_score :
  let team_scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let total_games : ℕ := team_scores.length
  let lost_games : ℕ := 8
  let won_games : ℕ := total_games - lost_games
  let lost_score_diff : ℕ := 2
  let won_score_ratio : ℕ := 3
  ∃ (opponent_scores : List ℕ),
    opponent_scores.length = total_games ∧
    (∀ i ∈ Finset.range lost_games,
      opponent_scores[i]! = team_scores[i]! + lost_score_diff) ∧
    (∀ i ∈ Finset.range won_games,
      team_scores[lost_games + i]! = won_score_ratio * opponent_scores[lost_games + i]!) ∧
    opponent_scores.sum = 78 :=
by sorry

end baseball_game_opponent_score_l1041_104172


namespace find_n_l1041_104101

theorem find_n (n : ℕ) 
  (h1 : Nat.gcd n 180 = 12) 
  (h2 : Nat.lcm n 180 = 720) : 
  n = 48 := by
sorry

end find_n_l1041_104101


namespace principal_is_8000_l1041_104179

/-- The principal amount that satisfies the given compound interest conditions -/
def find_principal : ℝ := by sorry

/-- The annual interest rate -/
def interest_rate : ℝ := by sorry

theorem principal_is_8000 :
  find_principal = 8000 ∧
  find_principal * (1 + interest_rate)^2 = 8820 ∧
  find_principal * (1 + interest_rate)^3 = 9261 := by sorry

end principal_is_8000_l1041_104179


namespace predicted_value_theorem_l1041_104173

/-- A linear regression model with given slope and sample centroid -/
structure LinearRegressionModel where
  slope : ℝ
  centroid_x : ℝ
  centroid_y : ℝ

/-- Calculate the predicted value of the dependent variable -/
def predict (model : LinearRegressionModel) (x : ℝ) : ℝ :=
  let intercept := model.centroid_y - model.slope * model.centroid_x
  model.slope * x + intercept

theorem predicted_value_theorem (model : LinearRegressionModel) 
  (h1 : model.slope = 1.23)
  (h2 : model.centroid_x = 4)
  (h3 : model.centroid_y = 5)
  (x : ℝ)
  (h4 : x = 10) :
  predict model x = 12.38 := by
  sorry

end predicted_value_theorem_l1041_104173


namespace square_perimeter_ratio_l1041_104154

theorem square_perimeter_ratio (area1 area2 perimeter1 perimeter2 : ℝ) :
  area1 > 0 ∧ area2 > 0 →
  area1 / area2 = 49 / 64 →
  perimeter1 / perimeter2 = 7 / 8 :=
by sorry

end square_perimeter_ratio_l1041_104154


namespace light_off_after_odd_presses_l1041_104178

def LightSwitch : Type := Bool

def press (state : LightSwitch) : LightSwitch :=
  !state

def press_n_times (state : LightSwitch) (n : ℕ) : LightSwitch :=
  match n with
  | 0 => state
  | m + 1 => press (press_n_times state m)

theorem light_off_after_odd_presses (n : ℕ) (h : Odd n) :
  press_n_times true n = false :=
sorry

end light_off_after_odd_presses_l1041_104178


namespace largest_divisor_of_three_consecutive_even_integers_l1041_104112

theorem largest_divisor_of_three_consecutive_even_integers :
  ∃ (d : ℕ), d = 24 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (2*n) * (2*n + 2) * (2*n + 4)) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (2*m) * (2*m + 2) * (2*m + 4))) :=
by sorry

end largest_divisor_of_three_consecutive_even_integers_l1041_104112


namespace y_value_l1041_104191

theorem y_value (y : ℚ) (h : (1 : ℚ) / 3 - (1 : ℚ) / 4 = 4 / y) : y = (1 : ℚ) / 3 := by
  sorry

end y_value_l1041_104191


namespace complex_equality_l1041_104140

theorem complex_equality (z : ℂ) : z = -1 + I ↔ Complex.abs (z - 2) = Complex.abs (z + 4) ∧ Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end complex_equality_l1041_104140


namespace four_card_selection_with_face_l1041_104134

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Number of face cards per suit -/
def FaceCardsPerSuit : ℕ := 3

/-- Number of cards per suit -/
def CardsPerSuit : ℕ := 13

/-- Theorem: Number of ways to choose 4 cards from a standard deck
    such that all four cards are of different suits and at least one is a face card -/
theorem four_card_selection_with_face (deck : ℕ) (suits : ℕ) (face_per_suit : ℕ) (cards_per_suit : ℕ)
    (h1 : deck = StandardDeck)
    (h2 : suits = NumSuits)
    (h3 : face_per_suit = FaceCardsPerSuit)
    (h4 : cards_per_suit = CardsPerSuit) :
  suits * face_per_suit * (cards_per_suit ^ (suits - 1)) = 26364 :=
sorry

end four_card_selection_with_face_l1041_104134


namespace no_solution_to_inequality_l1041_104105

theorem no_solution_to_inequality (x : ℝ) :
  x ≥ -1/4 → ¬(-1 - 1/(3*x + 4) < 2) :=
by sorry

end no_solution_to_inequality_l1041_104105


namespace subset_implies_a_geq_two_disjoint_implies_a_leq_one_l1041_104103

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for case (1)
theorem subset_implies_a_geq_two (a : ℝ) : A ⊆ B a → a ≥ 2 := by
  sorry

-- Theorem for case (2)
theorem disjoint_implies_a_leq_one (a : ℝ) : A ∩ B a = ∅ → a ≤ 1 := by
  sorry

end subset_implies_a_geq_two_disjoint_implies_a_leq_one_l1041_104103


namespace count_eight_digit_integers_l1041_104181

/-- The number of different 8-digit positive integers where the first digit is not 0
    and the last digit is neither 0 nor 1 -/
def eight_digit_integers : ℕ :=
  9 * (10 ^ 6) * 8

theorem count_eight_digit_integers :
  eight_digit_integers = 72000000 := by
  sorry

end count_eight_digit_integers_l1041_104181


namespace fraction_product_equality_l1041_104126

theorem fraction_product_equality : (3 + 5 + 7) / (2 + 4 + 6) * (4 + 8 + 12) / (1 + 3 + 5) = 10 / 3 := by
  sorry

end fraction_product_equality_l1041_104126


namespace isosceles_triangle_perimeter_l1041_104139

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 2 → b = 4 → c = 4 →  -- Two sides are equal (isosceles) and one side is 2
  a + b + c = 10 :=         -- The perimeter is 10
by
  sorry

end isosceles_triangle_perimeter_l1041_104139


namespace solve_linear_equation_l1041_104192

theorem solve_linear_equation :
  ∀ x : ℚ, 3 * x + 8 = -4 * x - 16 → x = -24 / 7 := by
  sorry

end solve_linear_equation_l1041_104192


namespace basketball_wins_l1041_104177

theorem basketball_wins (x : ℚ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : x + (5/8)*x + (x + (5/8)*x) = 130) : x = 40 := by
  sorry

end basketball_wins_l1041_104177


namespace hash_seven_two_l1041_104182

-- Define the # operation
def hash (a b : ℤ) : ℤ := 4 * a - 2 * b

-- State the theorem
theorem hash_seven_two : hash 7 2 = 24 := by
  sorry

end hash_seven_two_l1041_104182


namespace correct_bouquet_flowers_l1041_104124

def flowers_for_bouquets (tulips roses extra : ℕ) : ℕ :=
  tulips + roses - extra

theorem correct_bouquet_flowers :
  flowers_for_bouquets 39 49 7 = 81 := by
  sorry

end correct_bouquet_flowers_l1041_104124


namespace donut_selection_problem_l1041_104199

theorem donut_selection_problem :
  let n : ℕ := 6  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 84 :=
by sorry

end donut_selection_problem_l1041_104199


namespace shifted_line_equation_l1041_104160

/-- Represents a line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line vertically by a given amount -/
def vertical_shift (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift }

/-- The theorem stating that shifting y = -2x up by 5 units results in y = -2x + 5 -/
theorem shifted_line_equation :
  let original_line : Line := { slope := -2, intercept := 0 }
  let shifted_line := vertical_shift original_line 5
  shifted_line = { slope := -2, intercept := 5 } := by sorry

end shifted_line_equation_l1041_104160


namespace expression_result_l1041_104197

theorem expression_result : (3.242 * 16) / 100 = 0.51872 := by
  sorry

end expression_result_l1041_104197


namespace sufficient_not_necessary_l1041_104121

/-- A right triangle with side lengths a, b, and c (a < b < c) -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  a_lt_b : a < b
  b_lt_c : b < c
  pythagoras : a^2 + b^2 = c^2

/-- The condition a:b:c = 3:4:5 -/
def is_345_ratio (t : RightTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k

/-- The condition that a, b, c form an arithmetic progression -/
def is_arithmetic_progression (t : RightTriangle) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ t.b - t.a = d ∧ t.c - t.b = d

theorem sufficient_not_necessary :
  (∀ t : RightTriangle, is_345_ratio t → is_arithmetic_progression t) ∧
  (∃ t : RightTriangle, is_arithmetic_progression t ∧ ¬is_345_ratio t) := by
  sorry

end sufficient_not_necessary_l1041_104121


namespace a_spends_95_percent_l1041_104143

/-- Represents the salaries and spending percentages of two individuals A and B -/
structure SalaryData where
  total_salary : ℝ
  a_salary : ℝ
  b_spend_percent : ℝ
  a_spend_percent : ℝ

/-- Calculates the savings of an individual given their salary and spending percentage -/
def savings (salary : ℝ) (spend_percent : ℝ) : ℝ :=
  salary * (1 - spend_percent)

/-- Theorem stating that under given conditions, A spends 95% of their salary -/
theorem a_spends_95_percent (data : SalaryData) 
  (h1 : data.total_salary = 3000)
  (h2 : data.a_salary = 2250)
  (h3 : data.b_spend_percent = 0.85)
  (h4 : savings data.a_salary data.a_spend_percent = 
        savings (data.total_salary - data.a_salary) data.b_spend_percent) :
  data.a_spend_percent = 0.95 := by
  sorry


end a_spends_95_percent_l1041_104143


namespace avg_days_before_trial_is_four_l1041_104164

/-- The average number of days spent in jail before trial -/
def avg_days_before_trial (num_cities num_days arrests_per_day total_weeks : ℕ) : ℚ :=
  let total_arrests := num_cities * num_days * arrests_per_day
  let total_jail_days := total_weeks * 7
  let days_after_trial := 7
  (total_jail_days / total_arrests : ℚ) - days_after_trial

theorem avg_days_before_trial_is_four :
  avg_days_before_trial 21 30 10 9900 = 4 := by
  sorry

end avg_days_before_trial_is_four_l1041_104164


namespace bowtie_equation_solution_l1041_104155

-- Define the bowties operation
noncomputable def bowtie (c d : ℝ) : ℝ := c - Real.sqrt (d - Real.sqrt (d - Real.sqrt d))

-- Theorem statement
theorem bowtie_equation_solution :
  ∃ x : ℝ, bowtie 7 x = 3 → x = 20 := by
  sorry

end bowtie_equation_solution_l1041_104155


namespace gcd_102_238_l1041_104198

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l1041_104198


namespace sum_of_prime_factors_1320_l1041_104104

def sum_of_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_of_prime_factors_1320 :
  sum_of_prime_factors 1320 = 21 := by
  sorry

end sum_of_prime_factors_1320_l1041_104104


namespace square_times_square_minus_one_div_12_l1041_104113

theorem square_times_square_minus_one_div_12 (k : ℤ) : 
  12 ∣ (k^2 * (k^2 - 1)) := by sorry

end square_times_square_minus_one_div_12_l1041_104113


namespace pool_filling_rate_prove_pool_filling_rate_l1041_104150

/-- Proves that the rate of filling the pool during the second and third hours is 10 gallons/hour -/
theorem pool_filling_rate : ℝ → Prop :=
  fun (R : ℝ) ↦
    (8 : ℝ) +         -- Water added in 1st hour
    (R * 2) +         -- Water added in 2nd and 3rd hours
    (14 : ℝ) -        -- Water added in 4th hour
    (8 : ℝ) =         -- Water lost in 5th hour
    (34 : ℝ) →        -- Total water after 5 hours
    R = (10 : ℝ)      -- Rate during 2nd and 3rd hours

/-- Proof of the theorem -/
theorem prove_pool_filling_rate : pool_filling_rate (10 : ℝ) := by
  sorry

end pool_filling_rate_prove_pool_filling_rate_l1041_104150


namespace y_derivative_l1041_104152

noncomputable def y (x : ℝ) : ℝ :=
  2 * x - Real.log (1 + Real.sqrt (1 - Real.exp (4 * x))) - Real.exp (-2 * x) * Real.arcsin (Real.exp (2 * x))

theorem y_derivative (x : ℝ) :
  deriv y x = 2 * Real.exp (-2 * x) * Real.arcsin (Real.exp (2 * x)) :=
by sorry

end y_derivative_l1041_104152


namespace equation_solution_exists_l1041_104102

theorem equation_solution_exists : ∃ x : ℝ, 
  (0.76 : ℝ)^3 - (0.1 : ℝ)^3 / (0.76 : ℝ)^2 + x + (0.1 : ℝ)^2 = 0.66 := by
  sorry

end equation_solution_exists_l1041_104102


namespace expression_evaluation_l1041_104183

theorem expression_evaluation :
  let a : ℝ := Real.sqrt 2 - 1
  (1 - 1 / (a + 1)) * ((a^2 + 2*a + 1) / a) = Real.sqrt 2 := by
  sorry

end expression_evaluation_l1041_104183


namespace multiplicative_inverse_of_110_mod_667_l1041_104108

-- Define the triangle
def leg1 : ℕ := 65
def leg2 : ℕ := 156
def hypotenuse : ℕ := 169

-- Define the relation C = A + B
def relation (A B C : ℕ) : Prop := C = A + B

-- Define the modulus
def modulus : ℕ := 667

-- Define the number we're finding the inverse for
def num : ℕ := 110

-- Theorem statement
theorem multiplicative_inverse_of_110_mod_667 :
  (∃ (A B : ℕ), relation A B hypotenuse ∧ leg1^2 + leg2^2 = hypotenuse^2) →
  ∃ (n : ℕ), n < modulus ∧ (num * n) % modulus = 1 ∧ n = 608 :=
by sorry

end multiplicative_inverse_of_110_mod_667_l1041_104108


namespace arithmetic_sequence_sum_l1041_104170

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  arithmetic_sequence a (-2) →
  a 1 + a 4 + a 7 = 50 →
  a 6 + a 9 + a 12 = 20 := by
  sorry

end arithmetic_sequence_sum_l1041_104170


namespace jason_nickels_l1041_104122

theorem jason_nickels : 
  ∀ (n q : ℕ), 
    n = q + 10 → 
    5 * n + 10 * q = 680 → 
    n = 52 := by
  sorry

end jason_nickels_l1041_104122


namespace total_cost_is_correct_l1041_104168

/-- Represents the quantity of an ingredient on a given day -/
structure IngredientQuantity where
  day1 : Float
  day7 : Float

/-- Represents the price of an ingredient -/
structure IngredientPrice where
  value : Float
  unit : String

/-- Represents an ingredient with its quantities and price -/
structure Ingredient where
  name : String
  quantity : IngredientQuantity
  price : IngredientPrice
  unit : String

def ingredients : List Ingredient := [
  { name := "Baking powder",
    quantity := { day1 := 12, day7 := 6 },
    price := { value := 3, unit := "per pound" },
    unit := "lbs" },
  { name := "Flour",
    quantity := { day1 := 6, day7 := 3.5 },
    price := { value := 1.5, unit := "per pound" },
    unit := "kg" },
  { name := "Sugar",
    quantity := { day1 := 20, day7 := 15 },
    price := { value := 0.5, unit := "per pound" },
    unit := "lbs" },
  { name := "Chocolate chips",
    quantity := { day1 := 5000, day7 := 1500 },
    price := { value := 0.015, unit := "per gram" },
    unit := "g" }
]

def kgToPounds : Float := 2.20462
def gToPounds : Float := 0.00220462

def calculateTotalCost (ingredients : List Ingredient) : Float :=
  sorry

theorem total_cost_is_correct :
  calculateTotalCost ingredients = 81.27 := by
  sorry

end total_cost_is_correct_l1041_104168


namespace madeline_work_hours_l1041_104156

/-- Calculates the minimum number of work hours needed to cover expenses and savings --/
def min_work_hours (rent : ℕ) (groceries : ℕ) (medical : ℕ) (utilities : ℕ) (savings : ℕ) (hourly_wage : ℕ) : ℕ :=
  let total_expenses := rent + groceries + medical + utilities + savings
  (total_expenses + hourly_wage - 1) / hourly_wage

theorem madeline_work_hours :
  min_work_hours 1200 400 200 60 200 15 = 138 := by
  sorry

end madeline_work_hours_l1041_104156


namespace total_chocolate_bars_l1041_104194

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 16

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 25

/-- The total number of chocolate bars in the large box -/
def total_bars : ℕ := num_small_boxes * bars_per_small_box

theorem total_chocolate_bars : total_bars = 400 := by
  sorry

end total_chocolate_bars_l1041_104194


namespace fractional_equation_positive_root_l1041_104100

theorem fractional_equation_positive_root (x m : ℝ) : 
  (∃ x > 0, (3 / (x - 4) = 1 - (x + m) / (4 - x))) → m = -1 := by
  sorry

end fractional_equation_positive_root_l1041_104100


namespace circumscribed_diagonals_center_implies_rhombus_l1041_104125

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Check if a quadrilateral is circumscribed around a circle -/
def isCircumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Check if the diagonals of a quadrilateral intersect at a given point -/
def diagonalsIntersectAt (q : Quadrilateral) (p : ℝ × ℝ) : Prop := sorry

/-- Check if a quadrilateral is a rhombus -/
def isRhombus (q : Quadrilateral) : Prop := sorry

/-- Main theorem -/
theorem circumscribed_diagonals_center_implies_rhombus (q : Quadrilateral) (c : Circle) :
  isCircumscribed q c → diagonalsIntersectAt q c.center → isRhombus q := by sorry

end circumscribed_diagonals_center_implies_rhombus_l1041_104125


namespace betty_order_total_cost_l1041_104135

/-- Calculate the total cost of Betty's order -/
theorem betty_order_total_cost :
  let slipper_quantity : ℕ := 6
  let slipper_price : ℚ := 5/2
  let lipstick_quantity : ℕ := 4
  let lipstick_price : ℚ := 5/4
  let hair_color_quantity : ℕ := 8
  let hair_color_price : ℚ := 3
  let total_items : ℕ := slipper_quantity + lipstick_quantity + hair_color_quantity
  let total_cost : ℚ := slipper_quantity * slipper_price + 
                        lipstick_quantity * lipstick_price + 
                        hair_color_quantity * hair_color_price
  total_items = 18 ∧ total_cost = 44 := by
  sorry

end betty_order_total_cost_l1041_104135


namespace hyperbola_eccentricity_l1041_104176

-- Define the hyperbola
def hyperbola (m : ℤ) (x y : ℝ) : Prop :=
  x^2 / m^2 + y^2 / (m^2 - 4) = 1

-- Define the eccentricity
def eccentricity (m : ℤ) : ℝ :=
  2

-- Theorem statement
theorem hyperbola_eccentricity (m : ℤ) :
  ∃ (e : ℝ), e = eccentricity m ∧ 
  ∀ (x y : ℝ), hyperbola m x y → e = 2 :=
sorry

end hyperbola_eccentricity_l1041_104176


namespace function_symmetry_l1041_104138

def symmetric_about (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (2 * p.1 - x) = 2 * p.2 - y

theorem function_symmetry (f : ℝ → ℝ) 
    (h1 : symmetric_about f (-1, 0))
    (h2 : ∀ x > 0, f x = 1 / x) :
    ∀ x < -2, f x = 1 / (x + 2) := by
  sorry

end function_symmetry_l1041_104138


namespace sector_area_l1041_104144

/-- Given a sector with central angle 2 radians and arc length 2, its area is 1. -/
theorem sector_area (θ : Real) (l : Real) (r : Real) : 
  θ = 2 → l = 2 → l = r * θ → (1/2) * r * θ = 1 := by
  sorry

end sector_area_l1041_104144


namespace cats_given_by_mr_sheridan_l1041_104149

/-- The number of cats Mrs. Sheridan initially had -/
def initial_cats : ℕ := 17

/-- The total number of cats Mrs. Sheridan has now -/
def total_cats : ℕ := 31

/-- The number of cats Mr. Sheridan gave to Mrs. Sheridan -/
def given_cats : ℕ := total_cats - initial_cats

theorem cats_given_by_mr_sheridan : given_cats = 14 := by sorry

end cats_given_by_mr_sheridan_l1041_104149


namespace f_properties_l1041_104165

def f (x : ℝ) : ℝ := x^2 + x - 6

theorem f_properties :
  (f 0 = -6) ∧ (∀ x : ℝ, f x = 0 → x = -3 ∨ x = 2) := by
  sorry

end f_properties_l1041_104165
