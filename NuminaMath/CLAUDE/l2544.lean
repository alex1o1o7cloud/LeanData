import Mathlib

namespace NUMINAMATH_CALUDE_solve_sqrt_equation_l2544_254431

theorem solve_sqrt_equation (x : ℝ) :
  Real.sqrt ((2 / x) + 3) = 4 / 3 → x = -18 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_sqrt_equation_l2544_254431


namespace NUMINAMATH_CALUDE_carrot_broccoli_ratio_is_two_to_one_l2544_254460

/-- Represents the sales data for a farmers' market --/
structure MarketSales where
  total : ℕ
  broccoli : ℕ
  cauliflower : ℕ
  spinach_offset : ℕ

/-- Calculates the ratio of carrot sales to broccoli sales --/
def carrot_broccoli_ratio (sales : MarketSales) : ℚ :=
  let carrot_sales := sales.total - sales.broccoli - sales.cauliflower - 
    (sales.spinach_offset + (sales.total - sales.broccoli - sales.cauliflower - sales.spinach_offset) / 2)
  carrot_sales / sales.broccoli

/-- Theorem stating that the ratio of carrot sales to broccoli sales is 2:1 --/
theorem carrot_broccoli_ratio_is_two_to_one (sales : MarketSales) 
  (h1 : sales.total = 380)
  (h2 : sales.broccoli = 57)
  (h3 : sales.cauliflower = 136)
  (h4 : sales.spinach_offset = 16) :
  carrot_broccoli_ratio sales = 2 := by
  sorry

end NUMINAMATH_CALUDE_carrot_broccoli_ratio_is_two_to_one_l2544_254460


namespace NUMINAMATH_CALUDE_problem_solution_l2544_254400

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem problem_solution (m : ℤ) (h_odd : m % 2 = 1) (h_eq : g (g (g m)) = 39) : m = 63 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2544_254400


namespace NUMINAMATH_CALUDE_unchanged_fraction_l2544_254452

theorem unchanged_fraction (x y : ℝ) : 
  (2 * x) / (3 * x - y) = (2 * (3 * x)) / (3 * (3 * x) - (3 * y)) :=
by sorry

end NUMINAMATH_CALUDE_unchanged_fraction_l2544_254452


namespace NUMINAMATH_CALUDE_fair_distribution_l2544_254418

/-- Represents the number of books each player brings to the game -/
def books_per_player : ℕ := 4

/-- Represents the total number of books in the game -/
def total_books : ℕ := 2 * books_per_player

/-- Represents the number of points needed to win the game -/
def points_to_win : ℕ := 3

/-- Represents player A's current points -/
def a_points : ℕ := 2

/-- Represents player B's current points -/
def b_points : ℕ := 1

/-- Represents the probability of player A winning the game -/
def prob_a_wins : ℚ := 3/4

/-- Represents the probability of player B winning the game -/
def prob_b_wins : ℚ := 1/4

/-- Theorem stating the fair distribution of books -/
theorem fair_distribution :
  let a_books := (total_books : ℚ) * prob_a_wins
  let b_books := (total_books : ℚ) * prob_b_wins
  a_books = 6 ∧ b_books = 2 := by
  sorry

end NUMINAMATH_CALUDE_fair_distribution_l2544_254418


namespace NUMINAMATH_CALUDE_cost_per_item_proof_l2544_254428

/-- The cost per item in the first batch of fruits -/
def cost_per_item_first_batch : ℝ := 120

/-- The total cost of the first batch of fruits -/
def total_cost_first_batch : ℝ := 600

/-- The total cost of the second batch of fruits -/
def total_cost_second_batch : ℝ := 1250

/-- The number of items in the second batch is twice the number in the first batch -/
axiom double_items : ∃ n : ℝ, n * cost_per_item_first_batch = total_cost_first_batch ∧
                               2 * n * (cost_per_item_first_batch + 5) = total_cost_second_batch

theorem cost_per_item_proof : 
  cost_per_item_first_batch = 120 :=
sorry

end NUMINAMATH_CALUDE_cost_per_item_proof_l2544_254428


namespace NUMINAMATH_CALUDE_line_through_points_l2544_254419

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in general form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem line_through_points :
  let p1 : Point2D := ⟨-2, 2⟩
  let p2 : Point2D := ⟨0, 6⟩
  let l : Line := ⟨2, -1, 6⟩
  pointOnLine p1 l ∧ pointOnLine p2 l := by sorry

end NUMINAMATH_CALUDE_line_through_points_l2544_254419


namespace NUMINAMATH_CALUDE_alcohol_percentage_original_mixture_l2544_254469

/-- Proves that the percentage of alcohol in the original mixture is 20% --/
theorem alcohol_percentage_original_mixture :
  let original_volume : ℝ := 15
  let added_water : ℝ := 5
  let new_alcohol_percentage : ℝ := 15
  let new_volume : ℝ := original_volume + added_water
  let original_alcohol_volume : ℝ := original_volume * (original_alcohol_percentage / 100)
  let new_alcohol_volume : ℝ := new_volume * (new_alcohol_percentage / 100)
  ∀ original_alcohol_percentage : ℝ,
    original_alcohol_volume = new_alcohol_volume →
    original_alcohol_percentage = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_alcohol_percentage_original_mixture_l2544_254469


namespace NUMINAMATH_CALUDE_reciprocal_problem_l2544_254496

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 4) : 200 * (1 / x) = 400 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l2544_254496


namespace NUMINAMATH_CALUDE_cheesecake_problem_l2544_254457

/-- A problem about cheesecakes in a bakery -/
theorem cheesecake_problem
  (initial_display : ℕ)
  (sold : ℕ)
  (total_left : ℕ)
  (h1 : initial_display = 10)
  (h2 : sold = 7)
  (h3 : total_left = 18) :
  initial_display - sold + (total_left - (initial_display - sold)) = 15 :=
by sorry

end NUMINAMATH_CALUDE_cheesecake_problem_l2544_254457


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l2544_254403

/-- The height of a ball thrown upwards is given by y = -20t^2 + 32t + 60,
    where y is the height in feet and t is the time in seconds.
    This theorem proves that the time when the ball hits the ground (y = 0)
    is (4 + √91) / 5 seconds. -/
theorem ball_hitting_ground_time :
  let y (t : ℝ) := -20 * t^2 + 32 * t + 60
  ∃ t : ℝ, y t = 0 ∧ t = (4 + Real.sqrt 91) / 5 :=
by sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l2544_254403


namespace NUMINAMATH_CALUDE_problem_statement_l2544_254474

-- Define the line x + y - 3 = 0
def line1 (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the vector (1, -1)
def vector1 : ℝ × ℝ := (1, -1)

-- Define the lines x + 2y - 4 = 0 and 2x + 4y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 2*y - 4 = 0
def line3 (x y : ℝ) : Prop := 2*x + 4*y + 1 = 0

-- Define the point (3, 4)
def point1 : ℝ × ℝ := (3, 4)

-- Define a function to check if a line has equal intercepts on both axes
def has_equal_intercepts (a b c : ℝ) : Prop :=
  ∃ (t : ℝ), a * t + b * t + c = 0 ∧ t ≠ 0

theorem problem_statement :
  -- 1. (1,-1) is a directional vector of the line x+y-3=0
  (∀ (t : ℝ), line1 (vector1.1 * t) (vector1.2 * t)) ∧
  -- 2. The distance between lines x+2y-4=0 and 2x+4y+1=0 is 9√5/10
  (let d := (9 * Real.sqrt 5) / 10;
   ∀ (x y : ℝ), line2 x y → ∀ (x' y' : ℝ), line3 x' y' →
   ((x - x')^2 + (y - y')^2).sqrt = d) ∧
  -- 3. There are exactly 2 lines passing through point (3,4) with equal intercepts on the two coordinate axes
  (∃! (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    a₁ * point1.1 + b₁ * point1.2 + c₁ = 0 ∧
    a₂ * point1.1 + b₂ * point1.2 + c₂ = 0 ∧
    has_equal_intercepts a₁ b₁ c₁ ∧
    has_equal_intercepts a₂ b₂ c₂ ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂)) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l2544_254474


namespace NUMINAMATH_CALUDE_invalid_prism_diagonals_l2544_254489

/-- Represents the lengths of the extended diagonals of a right regular prism -/
structure PrismDiagonals where
  d1 : ℝ
  d2 : ℝ
  d3 : ℝ

/-- Checks if the given lengths can be the extended diagonals of a right regular prism -/
def is_valid_prism_diagonals (d : PrismDiagonals) : Prop :=
  ∃ (a b c : ℝ),
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (d.d1^2 = a^2 + b^2 ∧ d.d2^2 = b^2 + c^2 ∧ d.d3^2 = a^2 + c^2)

/-- The main theorem stating that {3, 4, 6} cannot be the lengths of extended diagonals -/
theorem invalid_prism_diagonals :
  ¬ is_valid_prism_diagonals ⟨3, 4, 6⟩ :=
sorry

end NUMINAMATH_CALUDE_invalid_prism_diagonals_l2544_254489


namespace NUMINAMATH_CALUDE_growth_rate_ratio_l2544_254423

/-- Given a linear regression equation y = ax + b where a = 4.4,
    prove that the ratio of the growth rate between x and y is 5/22 -/
theorem growth_rate_ratio (a b : ℝ) (h : a = 4.4) :
  (1 / a : ℝ) = 5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_growth_rate_ratio_l2544_254423


namespace NUMINAMATH_CALUDE_only_negative_three_squared_positive_l2544_254405

theorem only_negative_three_squared_positive :
  let a := 0 * ((-2019) ^ 2018)
  let b := (-3) ^ 2
  let c := -2 / ((-3) ^ 4)
  let d := (-2) ^ 3
  (a ≤ 0 ∧ b > 0 ∧ c < 0 ∧ d < 0) := by sorry

end NUMINAMATH_CALUDE_only_negative_three_squared_positive_l2544_254405


namespace NUMINAMATH_CALUDE_decimal_100_to_base_4_has_four_digits_l2544_254422

def to_base_4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem decimal_100_to_base_4_has_four_digits :
  (to_base_4 100).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_decimal_100_to_base_4_has_four_digits_l2544_254422


namespace NUMINAMATH_CALUDE_sum_x_y_equals_three_l2544_254454

/-- Given a system of linear equations, prove that x + y = 3 -/
theorem sum_x_y_equals_three (x y : ℝ) 
  (eq1 : 2 * x + y = 5) 
  (eq2 : x + 2 * y = 4) : 
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_three_l2544_254454


namespace NUMINAMATH_CALUDE_short_trees_calculation_l2544_254472

/-- The number of short trees currently in the park -/
def current_short_trees : ℕ := 112

/-- The number of short trees to be planted -/
def trees_to_plant : ℕ := 105

/-- The total number of short trees after planting -/
def total_short_trees : ℕ := 217

/-- Theorem stating that the current number of short trees plus the number of trees to be planted equals the total number of short trees after planting -/
theorem short_trees_calculation :
  current_short_trees + trees_to_plant = total_short_trees := by sorry

end NUMINAMATH_CALUDE_short_trees_calculation_l2544_254472


namespace NUMINAMATH_CALUDE_field_length_proof_l2544_254442

/-- Proves that for a rectangular field with given conditions, the length is 24 meters -/
theorem field_length_proof (width : ℝ) (length : ℝ) : 
  width = 13.5 → length = 2 * width - 3 → length = 24 := by
  sorry

end NUMINAMATH_CALUDE_field_length_proof_l2544_254442


namespace NUMINAMATH_CALUDE_black_balls_count_l2544_254477

theorem black_balls_count (total : ℕ) (red : ℕ) (prob_white : ℚ) 
  (h_total : total = 100)
  (h_red : red = 30)
  (h_prob_white : prob_white = 47/100) :
  total - red - (total * prob_white).num = 23 := by
sorry

end NUMINAMATH_CALUDE_black_balls_count_l2544_254477


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l2544_254466

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 16 * x + k = 0) ↔ k = 8 := by
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l2544_254466


namespace NUMINAMATH_CALUDE_square_in_base_seven_l2544_254424

theorem square_in_base_seven :
  ∃ (b : ℕ) (h : b > 6), 
    (1 * b^4 + 6 * b^3 + 3 * b^2 + 2 * b + 4) = (1 * b^2 + 2 * b + 5)^2 ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_in_base_seven_l2544_254424


namespace NUMINAMATH_CALUDE_no_integer_solution_l2544_254416

theorem no_integer_solution : ¬ ∃ (x : ℤ), 7 - 3 * (x^2 - 2) > 19 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2544_254416


namespace NUMINAMATH_CALUDE_constant_sum_property_l2544_254450

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the ellipse -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 / 8 + p.y^2 / 4 = 1

/-- Definition of a line passing through a point -/
def Line (p : Point) (m : ℝ) :=
  {q : Point | q.y - p.y = m * (q.x - p.x)}

/-- The focus point of the ellipse -/
def F : Point := ⟨2, 0⟩

/-- Theorem stating the constant sum property -/
theorem constant_sum_property 
  (A B P : Point) 
  (hA : isOnEllipse A) 
  (hB : isOnEllipse B) 
  (hP : P.x = 0) 
  (hline : ∃ (m : ℝ), A ∈ Line F m ∧ B ∈ Line F m ∧ P ∈ Line F m)
  (m n : ℝ)
  (hm : (P.x - A.x, P.y - A.y) = m • (A.x - F.x, A.y - F.y))
  (hn : (P.x - B.x, P.y - B.y) = n • (B.x - F.x, B.y - F.y)) :
  m + n = -4 := by
    sorry


end NUMINAMATH_CALUDE_constant_sum_property_l2544_254450


namespace NUMINAMATH_CALUDE_bridge_length_l2544_254427

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 265 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2544_254427


namespace NUMINAMATH_CALUDE_equation_solution_l2544_254449

theorem equation_solution (x : ℝ) : 
  (x^2 - 36) / 3 = (x^2 + 3*x + 9) / 6 ↔ x = 9 ∨ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2544_254449


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2544_254486

theorem greatest_divisor_with_remainders : 
  let a := 1657 - 6
  let b := 2037 - 5
  Nat.gcd a b = 127 := by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2544_254486


namespace NUMINAMATH_CALUDE_edward_games_count_l2544_254430

theorem edward_games_count :
  let sold_games : ℕ := 19
  let boxes_used : ℕ := 2
  let games_per_box : ℕ := 8
  let packed_games : ℕ := boxes_used * games_per_box
  let total_games : ℕ := sold_games + packed_games
  total_games = 35 := by sorry

end NUMINAMATH_CALUDE_edward_games_count_l2544_254430


namespace NUMINAMATH_CALUDE_no_four_digit_numbers_divisible_by_11_sum_10_l2544_254493

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_nonzero : a ≠ 0
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10
  d_digit : d < 10

/-- Checks if a four-digit number is divisible by 11 -/
def isDivisibleBy11 (n : FourDigitNumber) : Prop :=
  (1000 * n.a + 100 * n.b + 10 * n.c + n.d) % 11 = 0

/-- Checks if the sum of digits of a four-digit number is 10 -/
def sumOfDigitsIs10 (n : FourDigitNumber) : Prop :=
  n.a + n.b + n.c + n.d = 10

/-- Theorem: There are no four-digit numbers divisible by 11 with digits summing to 10 -/
theorem no_four_digit_numbers_divisible_by_11_sum_10 :
  ¬ ∃ (n : FourDigitNumber), isDivisibleBy11 n ∧ sumOfDigitsIs10 n := by
  sorry

end NUMINAMATH_CALUDE_no_four_digit_numbers_divisible_by_11_sum_10_l2544_254493


namespace NUMINAMATH_CALUDE_max_product_value_l2544_254436

/-- Given two real-valued functions f and g with specified ranges and a condition on their maxima,
    this theorem states that the maximum value of their product is 35. -/
theorem max_product_value (f g : ℝ → ℝ) (hf : ∀ x, 1 ≤ f x ∧ f x ≤ 7) 
    (hg : ∀ x, -3 ≤ g x ∧ g x ≤ 5) 
    (hmax : ∃ x, f x = 7 ∧ g x = 5) : 
    (∃ b, ∀ x, f x * g x ≤ b) ∧ (∀ b, (∀ x, f x * g x ≤ b) → b ≥ 35) :=
sorry

end NUMINAMATH_CALUDE_max_product_value_l2544_254436


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l2544_254475

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem arithmetic_progression_x_value :
  ∀ x : ℝ, is_arithmetic_progression (2*x - 3) (3*x - 2) (5*x + 2) → x = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l2544_254475


namespace NUMINAMATH_CALUDE_rhombus_area_l2544_254437

/-- The area of a rhombus with side length 4 cm and an angle of 30° between adjacent sides is 8√3 cm². -/
theorem rhombus_area (side_length : ℝ) (angle : ℝ) :
  side_length = 4 →
  angle = 30 * π / 180 →
  let area := side_length * side_length * Real.sin angle
  area = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l2544_254437


namespace NUMINAMATH_CALUDE_crayons_difference_l2544_254484

/-- Given the initial number of crayons, the number of crayons given away, and the number of crayons lost,
    prove that the difference between crayons given away and crayons lost is 410. -/
theorem crayons_difference (initial : ℕ) (given_away : ℕ) (lost : ℕ)
  (h1 : initial = 589)
  (h2 : given_away = 571)
  (h3 : lost = 161) :
  given_away - lost = 410 := by
  sorry

end NUMINAMATH_CALUDE_crayons_difference_l2544_254484


namespace NUMINAMATH_CALUDE_algebraic_expression_properties_l2544_254467

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^5 + b * x^3 + 3 * x + c

theorem algebraic_expression_properties :
  (f 0 = -1) →
  (f 1 = -1) →
  (f 3 = -10) →
  (c = -1 ∧ a + b + c = -4 ∧ f (-3) = 8) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_properties_l2544_254467


namespace NUMINAMATH_CALUDE_binary_101_equals_5_l2544_254417

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.foldr (fun bit acc => 2 * acc + bit) 0

theorem binary_101_equals_5 : binary_to_decimal [1, 0, 1] = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_equals_5_l2544_254417


namespace NUMINAMATH_CALUDE_intersection_S_T_l2544_254481

def S : Set ℝ := {x | x + 1 ≥ 2}
def T : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_S_T : S ∩ T = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_S_T_l2544_254481


namespace NUMINAMATH_CALUDE_product_magnitude_l2544_254441

open Complex

theorem product_magnitude (z₁ z₂ : ℂ) (h1 : abs z₁ = 3) (h2 : z₂ = 2 + I) : 
  abs (z₁ * z₂) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_product_magnitude_l2544_254441


namespace NUMINAMATH_CALUDE_only_D_is_comprehensive_l2544_254426

/-- Represents the type of survey --/
inductive SurveyType
  | Comprehensive
  | Sampling

/-- Represents the different survey options --/
inductive SurveyOption
  | A  -- Understanding the service life of a certain light bulb
  | B  -- Understanding whether a batch of cold drinks meets quality standards
  | C  -- Understanding the vision status of eighth-grade students nationwide
  | D  -- Understanding which month has the most births in a certain class

/-- Determines the appropriate survey type for a given option --/
def determineSurveyType (option : SurveyOption) : SurveyType :=
  match option with
  | SurveyOption.A => SurveyType.Sampling
  | SurveyOption.B => SurveyType.Sampling
  | SurveyOption.C => SurveyType.Sampling
  | SurveyOption.D => SurveyType.Comprehensive

/-- Theorem stating that only Option D is suitable for a comprehensive survey --/
theorem only_D_is_comprehensive :
  ∀ (option : SurveyOption),
    determineSurveyType option = SurveyType.Comprehensive ↔ option = SurveyOption.D :=
by sorry

#check only_D_is_comprehensive

end NUMINAMATH_CALUDE_only_D_is_comprehensive_l2544_254426


namespace NUMINAMATH_CALUDE_sum_of_digits_multiple_of_990_l2544_254476

/-- Given a six-digit number 123abc that is a multiple of 990, 
    prove that the sum of its hundreds, tens, and units digits (a + b + c) is 12 -/
theorem sum_of_digits_multiple_of_990 (a b c : ℕ) : 
  (0 < a) → (a < 10) →
  (0 ≤ b) → (b < 10) →
  (0 ≤ c) → (c < 10) →
  (123000 + 100 * a + 10 * b + c) % 990 = 0 →
  a + b + c = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_multiple_of_990_l2544_254476


namespace NUMINAMATH_CALUDE_gcf_32_48_l2544_254413

theorem gcf_32_48 : Nat.gcd 32 48 = 16 := by
  sorry

end NUMINAMATH_CALUDE_gcf_32_48_l2544_254413


namespace NUMINAMATH_CALUDE_number_count_proof_l2544_254488

theorem number_count_proof (total_avg : ℝ) (group1_avg : ℝ) (group2_avg : ℝ) (group3_avg : ℝ) :
  total_avg = 2.5 →
  group1_avg = 1.1 →
  group2_avg = 1.4 →
  group3_avg = 5 →
  ∃ (n : ℕ), n = 6 ∧ 
    n * total_avg = 2 * group1_avg + 2 * group2_avg + 2 * group3_avg :=
by sorry

end NUMINAMATH_CALUDE_number_count_proof_l2544_254488


namespace NUMINAMATH_CALUDE_edward_rides_l2544_254471

theorem edward_rides (total_tickets : ℕ) (spent_tickets : ℕ) (cost_per_ride : ℕ) : 
  total_tickets = 79 → spent_tickets = 23 → cost_per_ride = 7 →
  (total_tickets - spent_tickets) / cost_per_ride = 8 := by
sorry

end NUMINAMATH_CALUDE_edward_rides_l2544_254471


namespace NUMINAMATH_CALUDE_exists_k_in_interval_l2544_254445

theorem exists_k_in_interval (x : ℝ) (hx_pos : 0 < x) (hx_le_one : x ≤ 1) :
  ∃ k : ℕ+, (4/3 : ℝ) < (k : ℝ) * x ∧ (k : ℝ) * x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_k_in_interval_l2544_254445


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2544_254443

theorem parallelogram_base_length 
  (area : ℝ) (height : ℝ) (base : ℝ) 
  (h1 : area = 448) 
  (h2 : height = 14) 
  (h3 : area = base * height) : 
  base = 32 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2544_254443


namespace NUMINAMATH_CALUDE_tennis_tournament_l2544_254478

theorem tennis_tournament (n : ℕ) : 
  (∃ (total_matches : ℕ) (women_wins men_wins : ℕ),
    -- Total number of players
    (n + (2*n + 1) = 3*n + 1) ∧
    -- Total matches calculation
    (total_matches = (3*n + 1) * (3*n) / 2 + 2*n) ∧
    -- Ratio of wins
    (3 * men_wins = 2 * women_wins) ∧
    -- Total wins equal total matches
    (women_wins + men_wins = total_matches) ∧
    -- n is a positive integer
    (n > 0)) →
  n = 2 :=
by sorry

end NUMINAMATH_CALUDE_tennis_tournament_l2544_254478


namespace NUMINAMATH_CALUDE_parallel_postulate_introduction_l2544_254468

-- Define the concept of a geometric theorem
def GeometricTheorem : Type := Unit

-- Define the concept of Euclid's parallel postulate
def EuclidParallelPostulate : Type := Unit

-- Define the property of a theorem being independent of the parallel postulate
def independent (t : GeometricTheorem) (p : EuclidParallelPostulate) : Prop := True

-- Define the concept of introducing a postulate in geometry
def introduced_later (p : EuclidParallelPostulate) : Prop := True

theorem parallel_postulate_introduction 
  (many_theorems : Set GeometricTheorem)
  (parallel_postulate : EuclidParallelPostulate)
  (h : ∀ t ∈ many_theorems, independent t parallel_postulate) :
  introduced_later parallel_postulate :=
by
  sorry

#check parallel_postulate_introduction

end NUMINAMATH_CALUDE_parallel_postulate_introduction_l2544_254468


namespace NUMINAMATH_CALUDE_a_minus_b_eq_neg_seven_l2544_254494

theorem a_minus_b_eq_neg_seven
  (h1 : Real.sqrt (a ^ 2) = 3)
  (h2 : Real.sqrt b = 2)
  (h3 : a * b < 0)
  : a - b = -7 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_eq_neg_seven_l2544_254494


namespace NUMINAMATH_CALUDE_b_contribution_is_9000_l2544_254464

/-- Represents the business partnership between A and B -/
structure Partnership where
  a_initial_investment : ℕ
  b_join_month : ℕ
  total_months : ℕ
  profit_ratio_a : ℕ
  profit_ratio_b : ℕ

/-- Calculates B's contribution to the capital given the partnership details -/
def calculate_b_contribution (p : Partnership) : ℕ :=
  sorry

/-- Theorem stating that B's contribution is 9000 rupees given the problem conditions -/
theorem b_contribution_is_9000 :
  let p : Partnership := {
    a_initial_investment := 3500,
    b_join_month := 5,
    total_months := 12,
    profit_ratio_a := 2,
    profit_ratio_b := 3
  }
  calculate_b_contribution p = 9000 := by
  sorry

end NUMINAMATH_CALUDE_b_contribution_is_9000_l2544_254464


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l2544_254421

theorem binomial_coefficient_sum (n : ℕ) : 4^n - 2^n = 992 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l2544_254421


namespace NUMINAMATH_CALUDE_common_integer_root_l2544_254483

theorem common_integer_root (a : ℤ) : 
  (∃ x : ℤ, a * x + a = 7 ∧ 3 * x - a = 17) ↔ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_common_integer_root_l2544_254483


namespace NUMINAMATH_CALUDE_ned_games_before_l2544_254410

/-- The number of games Ned had before giving away some -/
def games_before : ℕ := sorry

/-- The number of games Ned gave away -/
def games_given_away : ℕ := 13

/-- The number of games Ned has now -/
def games_now : ℕ := 6

/-- Theorem stating the number of games Ned had before -/
theorem ned_games_before :
  games_before = games_given_away + games_now := by sorry

end NUMINAMATH_CALUDE_ned_games_before_l2544_254410


namespace NUMINAMATH_CALUDE_lexie_family_age_ratio_l2544_254414

/-- Proves that given the age relationships in Lexie's family, the ratio of her sister's age to Lexie's age is 2. -/
theorem lexie_family_age_ratio :
  ∀ (lexie_age brother_age sister_age : ℕ),
    lexie_age = 8 →
    lexie_age = brother_age + 6 →
    sister_age - brother_age = 14 →
    ∃ (k : ℕ), sister_age = k * lexie_age →
    sister_age / lexie_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_lexie_family_age_ratio_l2544_254414


namespace NUMINAMATH_CALUDE_min_value_of_f_l2544_254473

/-- Given positive real numbers a, b, c, x, y, z satisfying the given conditions,
    the minimum value of the function f is 1/2 -/
theorem min_value_of_f (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (eq1 : c * y + b * z = a)
  (eq2 : a * z + c * x = b)
  (eq3 : b * x + a * y = c) :
  (∀ x' y' z' : ℝ, 0 < x' → 0 < y' → 0 < z' →
    c * y' + b * z' = a →
    a * z' + c * x' = b →
    b * x' + a * y' = c →
    x'^2 / (1 + x') + y'^2 / (1 + y') + z'^2 / (1 + z') ≥ 1/2) ∧
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2544_254473


namespace NUMINAMATH_CALUDE_diagonal_four_sides_squared_l2544_254497

/-- A regular nonagon -/
structure RegularNonagon where
  /-- The length of a side -/
  a : ℝ
  /-- The length of a diagonal that jumps over four sides -/
  d : ℝ
  /-- Ensure the side length is positive -/
  a_pos : a > 0

/-- In a regular nonagon, the square of the length of a diagonal that jumps over four sides
    is equal to five times the square of the side length -/
theorem diagonal_four_sides_squared (n : RegularNonagon) : n.d^2 = 5 * n.a^2 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_four_sides_squared_l2544_254497


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2544_254485

-- Define the cyclic sum function
def cyclicSum (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c + f b c a + f c a b

-- State the theorem
theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  cyclicSum (fun x y z => (y + z - x)^2 / (x^2 + (y + z)^2)) a b c ≥ 3/5 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2544_254485


namespace NUMINAMATH_CALUDE_special_ellipse_major_axis_length_l2544_254458

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : Bool
  /-- The x-coordinate of both foci -/
  foci_x : ℝ
  /-- The y-coordinate of the first focus -/
  focus1_y : ℝ
  /-- The y-coordinate of the second focus -/
  focus2_y : ℝ

/-- The length of the major axis of the special ellipse -/
def majorAxisLength (e : SpecialEllipse) : ℝ := 2

/-- Theorem stating that the length of the major axis is 2 for the given ellipse -/
theorem special_ellipse_major_axis_length (e : SpecialEllipse) 
  (h1 : e.tangent_to_axes = true)
  (h2 : e.foci_x = 4)
  (h3 : e.focus1_y = 1 + 2 * Real.sqrt 2)
  (h4 : e.focus2_y = 1 - 2 * Real.sqrt 2) :
  majorAxisLength e = 2 := by sorry

end NUMINAMATH_CALUDE_special_ellipse_major_axis_length_l2544_254458


namespace NUMINAMATH_CALUDE_quadrangular_pyramid_faces_l2544_254456

/-- A quadrangular pyramid is a geometric shape with triangular lateral faces and a quadrilateral base. -/
structure QuadrangularPyramid where
  lateral_faces : Nat
  base_face : Nat
  lateral_faces_are_triangles : lateral_faces = 4
  base_is_quadrilateral : base_face = 1

/-- The total number of faces in a quadrangular pyramid is 5. -/
theorem quadrangular_pyramid_faces (p : QuadrangularPyramid) : 
  p.lateral_faces + p.base_face = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadrangular_pyramid_faces_l2544_254456


namespace NUMINAMATH_CALUDE_decrease_by_percentage_eighty_decreased_by_eightyfive_percent_l2544_254444

theorem decrease_by_percentage (x : ℝ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 100) :
  x - (p / 100) * x = x * (1 - p / 100) :=
sorry

theorem eighty_decreased_by_eightyfive_percent :
  80 - (85 / 100) * 80 = 12 :=
sorry

end NUMINAMATH_CALUDE_decrease_by_percentage_eighty_decreased_by_eightyfive_percent_l2544_254444


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2544_254440

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 5}

theorem complement_of_M_in_U :
  (U \ M) = {3, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2544_254440


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2544_254465

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a = 6 ∧ b = 8 ∧ c^2 = a^2 + b^2) ∨ 
  (a = 6 ∧ c = 8 ∧ b^2 = c^2 - a^2) ∨
  (b = 6 ∧ c = 8 ∧ a^2 = c^2 - b^2) →
  c = 10 ∨ c = 2 * Real.sqrt 7 ∨ b = 10 ∨ b = 2 * Real.sqrt 7 ∨ a = 10 ∨ a = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2544_254465


namespace NUMINAMATH_CALUDE_units_digit_G_500_l2544_254409

/-- The Modified Fermat number for a given n -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_G_500 : units_digit (G 500) = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_G_500_l2544_254409


namespace NUMINAMATH_CALUDE_ones_more_frequent_than_fives_l2544_254446

-- Define the upper bound of the sequence
def upperBound : ℕ := 1000000000

-- Define a function that computes the digital root of a number
def digitalRoot (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

-- Define a function that counts occurrences of a digit in the final sequence
def countDigit (d : ℕ) : ℕ :=
  (upperBound / 9) + if d = 1 then 1 else 0

-- Theorem statement
theorem ones_more_frequent_than_fives :
  countDigit 1 > countDigit 5 := by
sorry

end NUMINAMATH_CALUDE_ones_more_frequent_than_fives_l2544_254446


namespace NUMINAMATH_CALUDE_at_least_one_multiple_of_11_l2544_254434

def base_n_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem at_least_one_multiple_of_11 :
  ∃ n : Nat, 2 ≤ n ∧ n ≤ 101 ∧ 
  (base_n_to_decimal [3, 4, 5, 7, 6, 2] n) % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_multiple_of_11_l2544_254434


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l2544_254459

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (correct_marks : ℕ) 
  (incorrect_marks : ℕ) 
  (h1 : total_sums = 30) 
  (h2 : total_marks = 60) 
  (h3 : correct_marks = 3) 
  (h4 : incorrect_marks = 2) : 
  ∃ (correct : ℕ), correct = 24 ∧ 
    correct + (total_sums - correct) = total_sums ∧ 
    (correct_marks : ℤ) * correct - incorrect_marks * (total_sums - correct) = total_marks :=
by sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l2544_254459


namespace NUMINAMATH_CALUDE_flour_amount_second_combination_l2544_254491

/-- The cost per pound of sugar and flour -/
def cost_per_pound : ℝ := 0.45

/-- The total cost of both combinations -/
def total_cost : ℝ := 26

/-- The amount of sugar in the first combination -/
def sugar_amount_1 : ℝ := 40

/-- The amount of flour in the first combination -/
def flour_amount_1 : ℝ := 16

/-- The amount of sugar in the second combination -/
def sugar_amount_2 : ℝ := 30

/-- The amount of flour in the second combination -/
def flour_amount_2 : ℝ := 28

theorem flour_amount_second_combination :
  sugar_amount_1 * cost_per_pound + flour_amount_1 * cost_per_pound = total_cost ∧
  sugar_amount_2 * cost_per_pound + flour_amount_2 * cost_per_pound = total_cost :=
by sorry

end NUMINAMATH_CALUDE_flour_amount_second_combination_l2544_254491


namespace NUMINAMATH_CALUDE_max_dimes_count_l2544_254490

/-- Represents the value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- Represents the total amount of money Liam has in dollars -/
def total_money : ℚ := 4.80

/-- 
Given that Liam has $4.80 in U.S. coins and an equal number of dimes and nickels,
this theorem states that the maximum number of dimes he could have is 32.
-/
theorem max_dimes_count : 
  ∃ (d : ℕ), d * (dime_value + nickel_value) = total_money ∧ 
             ∀ (x : ℕ), x * (dime_value + nickel_value) ≤ total_money → x ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_dimes_count_l2544_254490


namespace NUMINAMATH_CALUDE_janet_bird_count_l2544_254415

theorem janet_bird_count (crows hawks : ℕ) : 
  hawks = crows + (crows * 6 / 10) →
  crows + hawks = 78 →
  crows = 30 := by
sorry

end NUMINAMATH_CALUDE_janet_bird_count_l2544_254415


namespace NUMINAMATH_CALUDE_isosceles_if_root_is_one_roots_of_equilateral_l2544_254498

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

-- Define the quadratic equation
def quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 - 2 * t.b * x + (t.a - t.c)

theorem isosceles_if_root_is_one (t : Triangle) :
  quadratic t 1 = 0 → t.a = t.b :=
by sorry

theorem roots_of_equilateral (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (∀ x : ℝ, quadratic t x = 0 ↔ x = 0 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_if_root_is_one_roots_of_equilateral_l2544_254498


namespace NUMINAMATH_CALUDE_distance_between_cities_l2544_254451

/-- The distance between City A and City B in miles -/
def distance : ℝ := 427.5

/-- The initial travel time from City A to City B in hours -/
def initial_time_AB : ℝ := 6

/-- The initial travel time from City B to City A in hours -/
def initial_time_BA : ℝ := 4.5

/-- The time saved on each trip in hours -/
def time_saved : ℝ := 0.5

/-- The average speed for the round trip after saving time in miles per hour -/
def average_speed : ℝ := 90

theorem distance_between_cities :
  2 * distance / (initial_time_AB + initial_time_BA - 2 * time_saved) = average_speed :=
sorry

end NUMINAMATH_CALUDE_distance_between_cities_l2544_254451


namespace NUMINAMATH_CALUDE_beggars_and_mothers_attitude_l2544_254470

structure Neighborhood where
  has_nearby_railway : Bool
  has_frequent_beggars : Bool

structure Mother where
  treats_beggars_equally : Bool
  provides_newspapers : Bool
  father_helped_in_depression : Bool

def reason_for_beggars_visits (n : Neighborhood) : Bool :=
  n.has_nearby_railway

def mother_treatment_of_beggars (m : Mother) : Bool :=
  m.treats_beggars_equally

def purpose_of_newspapers (m : Mother) : Bool :=
  m.provides_newspapers

def explanation_for_mothers_attitude (m : Mother) : Bool :=
  m.father_helped_in_depression

theorem beggars_and_mothers_attitude 
  (n : Neighborhood) 
  (m : Mother) 
  (h1 : n.has_nearby_railway = true)
  (h2 : n.has_frequent_beggars = true)
  (h3 : m.treats_beggars_equally = true)
  (h4 : m.provides_newspapers = true)
  (h5 : m.father_helped_in_depression = true) :
  reason_for_beggars_visits n = true ∧
  mother_treatment_of_beggars m = true ∧
  purpose_of_newspapers m = true ∧
  explanation_for_mothers_attitude m = true := by
  sorry

end NUMINAMATH_CALUDE_beggars_and_mothers_attitude_l2544_254470


namespace NUMINAMATH_CALUDE_range_of_a_for_sqrt_function_l2544_254462

theorem range_of_a_for_sqrt_function (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = Real.sqrt (x^2 + 2*a*x + 1)) → 
  (∀ x, ∃ y, f x = y) →
  -1 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_sqrt_function_l2544_254462


namespace NUMINAMATH_CALUDE_baseball_league_games_l2544_254453

theorem baseball_league_games (P Q : ℕ) : 
  P > 2 * Q →
  Q > 6 →
  4 * P + 5 * Q = 82 →
  4 * P = 52 := by
sorry

end NUMINAMATH_CALUDE_baseball_league_games_l2544_254453


namespace NUMINAMATH_CALUDE_astros_win_in_seven_l2544_254487

/-- The probability of the Dodgers winning a single game -/
def p_dodgers : ℚ := 3/4

/-- The probability of the Astros winning a single game -/
def p_astros : ℚ := 1 - p_dodgers

/-- The number of games needed to win the World Series -/
def games_to_win : ℕ := 4

/-- The total number of games in a full World Series -/
def total_games : ℕ := 2 * games_to_win - 1

/-- The probability of the Astros winning the World Series in exactly 7 games -/
def p_astros_win_in_seven : ℚ := 135/4096

theorem astros_win_in_seven :
  p_astros_win_in_seven = (Nat.choose 6 3 : ℚ) * p_astros^3 * p_dodgers^3 * p_astros := by sorry

end NUMINAMATH_CALUDE_astros_win_in_seven_l2544_254487


namespace NUMINAMATH_CALUDE_final_pen_count_l2544_254408

def pen_collection (initial : ℕ) (mike_gave : ℕ) (sharon_took : ℕ) : ℕ :=
  ((initial + mike_gave) * 2) - sharon_took

theorem final_pen_count : pen_collection 25 22 19 = 75 := by
  sorry

end NUMINAMATH_CALUDE_final_pen_count_l2544_254408


namespace NUMINAMATH_CALUDE_discussions_probability_l2544_254482

def word := "DISCUSSIONS"

theorem discussions_probability : 
  let total_arrangements := Nat.factorial 11 / (Nat.factorial 4 * Nat.factorial 2)
  let favorable_arrangements := Nat.factorial 8 / Nat.factorial 2
  (favorable_arrangements : ℚ) / total_arrangements = 4 / 165 := by
  sorry

end NUMINAMATH_CALUDE_discussions_probability_l2544_254482


namespace NUMINAMATH_CALUDE_customized_bowling_ball_volume_l2544_254412

/-- The volume of a customized bowling ball -/
theorem customized_bowling_ball_volume :
  let ball_diameter : ℝ := 24
  let hole_depth : ℝ := 10
  let hole_diameters : List ℝ := [1.5, 1.5, 2, 2.5]
  let sphere_volume := (4 / 3) * π * (ball_diameter / 2) ^ 3
  let hole_volumes := hole_diameters.map (fun d => π * (d / 2) ^ 2 * hole_depth)
  sphere_volume - hole_volumes.sum = 2233.375 * π := by
  sorry

end NUMINAMATH_CALUDE_customized_bowling_ball_volume_l2544_254412


namespace NUMINAMATH_CALUDE_triangle_area_is_one_l2544_254492

/-- The area of a triangle bounded by the x-axis and two lines -/
def triangleArea (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop) : ℝ :=
  1 -- We define the area as 1 based on the problem statement

/-- The first line equation: y - 2x = 2 -/
def line1 (x y : ℝ) : Prop :=
  y - 2*x = 2

/-- The second line equation: 2y - x = 1 -/
def line2 (x y : ℝ) : Prop :=
  2*y - x = 1

/-- Theorem stating that the area of the triangle is 1 -/
theorem triangle_area_is_one :
  triangleArea line1 line2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_is_one_l2544_254492


namespace NUMINAMATH_CALUDE_parabola_shift_l2544_254407

/-- Given a parabola y = x^2, shifting it 3 units right and 4 units up results in y = (x-3)^2 + 4 -/
theorem parabola_shift (x y : ℝ) : 
  (y = x^2) → 
  (∃ (y' : ℝ → ℝ), 
    (∀ x, y' x = (x - 3)^2) ∧ 
    (∀ x, y' x + 4 = (x - 3)^2 + 4)) := by
  sorry


end NUMINAMATH_CALUDE_parabola_shift_l2544_254407


namespace NUMINAMATH_CALUDE_isosceles_minimizes_side_l2544_254404

/-- Represents a triangle with sides a, b, c and angle α opposite to side a -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  area : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ α > 0 ∧ area > 0
  h_angle : α < π
  h_area : area = (1/2) * b * c * Real.sin α

/-- Given a fixed angle α and area S, the triangle that minimizes side a is isosceles with b = c -/
theorem isosceles_minimizes_side (α S : ℝ) (h_α : 0 < α ∧ α < π) (h_S : S > 0) :
  ∃ (t : Triangle), t.α = α ∧ t.area = S ∧ t.b = t.c ∧
  ∀ (u : Triangle), u.α = α → u.area = S → t.a ≤ u.a :=
sorry

end NUMINAMATH_CALUDE_isosceles_minimizes_side_l2544_254404


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l2544_254499

/-- The number of questions asked to the Magic 8 Ball -/
def num_questions : ℕ := 7

/-- The number of possible responses from the Magic 8 Ball -/
def num_responses : ℕ := 3

/-- The probability of each type of response -/
def response_probability : ℚ := 1 / 3

/-- The number of desired positive responses -/
def desired_positive : ℕ := 3

/-- The number of desired neutral responses -/
def desired_neutral : ℕ := 2

/-- Theorem stating the probability of getting exactly 3 positive answers and 2 neutral answers
    when asking a Magic 8 Ball 7 questions, where each type of response has an equal probability of 1/3 -/
theorem magic_8_ball_probability :
  (Nat.choose num_questions desired_positive *
   Nat.choose (num_questions - desired_positive) desired_neutral *
   response_probability ^ num_questions) = 70 / 243 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l2544_254499


namespace NUMINAMATH_CALUDE_square_is_quadratic_l2544_254402

/-- A function f: ℝ → ℝ is quadratic if there exist constants a, b, c with a ≠ 0 such that
    f(x) = a * x^2 + b * x + c for all x -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x^2 is quadratic -/
theorem square_is_quadratic : IsQuadratic (fun x ↦ x^2) := by
  sorry

end NUMINAMATH_CALUDE_square_is_quadratic_l2544_254402


namespace NUMINAMATH_CALUDE_officer_combinations_count_l2544_254461

def totalMembers : ℕ := 25
def officerPositions : ℕ := 3

def chooseOfficers (n m : ℕ) : ℕ := n * (n - 1) * (n - 2)

def officersCombinations : ℕ :=
  let withoutPairs := chooseOfficers (totalMembers - 4) officerPositions
  let withAliceBob := 3 * 2 * (totalMembers - 4)
  let withCharlesDiana := 3 * 2 * (totalMembers - 4)
  withoutPairs + withAliceBob + withCharlesDiana

theorem officer_combinations_count :
  officersCombinations = 8232 :=
sorry

end NUMINAMATH_CALUDE_officer_combinations_count_l2544_254461


namespace NUMINAMATH_CALUDE_magic_mike_calculation_l2544_254447

/-- The problem statement --/
theorem magic_mike_calculation (p q r s t : ℝ) : 
  p = 3 ∧ q = 4 ∧ r = 5 ∧ s = 6 →
  (p - q + r * s - t = p - (q - (r * (s - t)))) →
  t = 0 := by
sorry

end NUMINAMATH_CALUDE_magic_mike_calculation_l2544_254447


namespace NUMINAMATH_CALUDE_simplify_expression_l2544_254480

theorem simplify_expression (x y : ℝ) : 2 - (3 - (2 + (5 - (3*y - x)))) = 6 - 3*y + x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2544_254480


namespace NUMINAMATH_CALUDE_sum_of_distances_is_ten_l2544_254432

/-- Given a circle tangent to the sides of an angle at points A and B, with a point C on the circle,
    this structure represents the distances and conditions of the problem. -/
structure CircleTangentProblem where
  -- Distance from C to line AB
  h : ℝ
  -- Distance from C to the side of the angle passing through A
  h_A : ℝ
  -- Distance from C to the side of the angle passing through B
  h_B : ℝ
  -- Condition: h = 4
  h_eq_four : h = 4
  -- Condition: One distance is four times the other
  one_distance_four_times_other : h_B = 4 * h_A

/-- The theorem stating that the sum of distances from C to the sides of the angle is 10. -/
theorem sum_of_distances_is_ten (p : CircleTangentProblem) : p.h_A + p.h_B = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distances_is_ten_l2544_254432


namespace NUMINAMATH_CALUDE_f_properties_l2544_254425

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x) / Real.log 2

theorem f_properties :
  (∀ x ≠ 0, f x = f (-x)) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2544_254425


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l2544_254495

theorem half_abs_diff_squares_20_15 : (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l2544_254495


namespace NUMINAMATH_CALUDE_community_A_sample_l2544_254455

/-- Represents the number of low-income households in a community -/
structure Community where
  households : ℕ

/-- Represents the total number of affordable housing units -/
def housing_units : ℕ := 90

/-- Calculates the number of households to be sampled from a community using stratified sampling -/
def stratified_sample (community : Community) (total_households : ℕ) : ℕ :=
  (community.households * housing_units) / total_households

/-- Theorem: The number of low-income households to be sampled from community A is 40 -/
theorem community_A_sample :
  let community_A : Community := ⟨360⟩
  let community_B : Community := ⟨270⟩
  let community_C : Community := ⟨180⟩
  let total_households := community_A.households + community_B.households + community_C.households
  stratified_sample community_A total_households = 40 := by
  sorry

end NUMINAMATH_CALUDE_community_A_sample_l2544_254455


namespace NUMINAMATH_CALUDE_triangle_area_l2544_254448

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = 5 →
  B = π / 3 →
  Real.cos A = 11 / 14 →
  let S := (1 / 2) * a * c * Real.sin B
  S = 10 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2544_254448


namespace NUMINAMATH_CALUDE_stratified_sampling_selection_l2544_254411

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of liberal arts students -/
def liberal_arts_students : ℕ := 5

/-- The number of science students -/
def science_students : ℕ := 10

/-- The number of liberal arts students to be selected -/
def selected_liberal_arts : ℕ := 2

/-- The number of science students to be selected -/
def selected_science : ℕ := 4

theorem stratified_sampling_selection :
  (binomial liberal_arts_students selected_liberal_arts) * 
  (binomial science_students selected_science) = 2100 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_selection_l2544_254411


namespace NUMINAMATH_CALUDE_distance_product_range_l2544_254429

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y^2 = 4*x
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 8

-- Define the point P on C₁
def P (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

-- Define the line l passing through P with slope 1
def l (t x : ℝ) : ℝ := x + 2*t - t^2

-- Define the product of distances |PQ||PR|
def distance_product (t : ℝ) : ℝ := (t^2 - 2)^2 + 4

-- Main theorem
theorem distance_product_range :
  ∀ t : ℝ, C₁ (P t).1 (P t).2 →
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
      C₂ x₁ (l t x₁) ∧ C₂ x₂ (l t x₂)) →
    distance_product t ∈ Set.Icc 4 8 ∪ Set.Ioo 8 36 :=
by sorry

end NUMINAMATH_CALUDE_distance_product_range_l2544_254429


namespace NUMINAMATH_CALUDE_greatest_common_divisor_480_90_under_60_l2544_254401

theorem greatest_common_divisor_480_90_under_60 : 
  ∃ n : ℕ, n > 0 ∧ 
    n ∣ 480 ∧ 
    n < 60 ∧ 
    n ∣ 90 ∧ 
    ∀ m : ℕ, m > 0 → m ∣ 480 → m < 60 → m ∣ 90 → m ≤ n :=
by
  use 30
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_480_90_under_60_l2544_254401


namespace NUMINAMATH_CALUDE_four_variable_inequality_l2544_254420

theorem four_variable_inequality (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_one : a + b + c + d = 1) :
  a * b * c * d + b * c * d * a + c * d * a * b + d * a * b * c ≤ 1 / 27 + 176 / 27 * a * b * c * d := by
  sorry

end NUMINAMATH_CALUDE_four_variable_inequality_l2544_254420


namespace NUMINAMATH_CALUDE_water_remaining_l2544_254435

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 11/8 → remaining = initial - used → remaining = 13/8 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l2544_254435


namespace NUMINAMATH_CALUDE_second_discount_percentage_l2544_254433

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) :
  original_price = 400 ∧
  first_discount = 30 ∧
  final_price = 224 →
  ∃ second_discount : ℝ,
    second_discount = 20 ∧
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l2544_254433


namespace NUMINAMATH_CALUDE_marbles_distribution_l2544_254463

theorem marbles_distribution (total_marbles : ℕ) (num_friends : ℕ) (marbles_per_friend : ℕ) :
  total_marbles = 5504 →
  num_friends = 64 →
  marbles_per_friend = total_marbles / num_friends →
  marbles_per_friend = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l2544_254463


namespace NUMINAMATH_CALUDE_squirrel_acorns_l2544_254439

/-- Represents the number of acorns each animal hides per hole -/
structure AcornsPerHole where
  chipmunk : ℕ
  squirrel : ℕ
  rabbit : ℕ

/-- Represents the number of holes each animal dug -/
structure Holes where
  chipmunk : ℕ
  squirrel : ℕ
  rabbit : ℕ

/-- The forest scenario with animals hiding acorns -/
def ForestScenario (a : AcornsPerHole) (h : Holes) : Prop :=
  -- Chipmunk and squirrel stash the same number of acorns
  a.chipmunk * h.chipmunk = a.squirrel * h.squirrel ∧
  -- Rabbit stashes the same number of acorns as the chipmunk
  a.rabbit * h.rabbit = a.chipmunk * h.chipmunk ∧
  -- Rabbit needs 3 more holes than the squirrel
  h.rabbit = h.squirrel + 3

/-- The theorem stating that the squirrel stashed 40 acorns -/
theorem squirrel_acorns (a : AcornsPerHole) (h : Holes)
  (ha : a.chipmunk = 4 ∧ a.squirrel = 5 ∧ a.rabbit = 3)
  (hf : ForestScenario a h) : 
  a.squirrel * h.squirrel = 40 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l2544_254439


namespace NUMINAMATH_CALUDE_divisible_by_50_l2544_254406

/-- A polygon drawn on a square grid -/
structure GridPolygon where
  area : ℕ
  divisible_by_2 : ∃ (half : ℕ), area = 2 * half
  divisible_by_25 : ∃ (part : ℕ), area = 25 * part

/-- The main theorem -/
theorem divisible_by_50 (p : GridPolygon) (h : p.area = 100) :
  ∃ (small : ℕ), p.area = 50 * small := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_50_l2544_254406


namespace NUMINAMATH_CALUDE_pumps_emptying_time_l2544_254479

/-- Represents the time (in hours) it takes for pumps A, B, and C to empty a pool when working together. -/
def combined_emptying_time (rate_A rate_B rate_C : ℚ) : ℚ :=
  1 / (rate_A + rate_B + rate_C)

/-- Theorem stating that pumps A, B, and C with given rates will empty the pool in 24/13 hours when working together. -/
theorem pumps_emptying_time :
  let rate_A : ℚ := 1/4
  let rate_B : ℚ := 1/6
  let rate_C : ℚ := 1/8
  combined_emptying_time rate_A rate_B rate_C = 24/13 := by
  sorry

#eval (24 : ℚ) / 13 * 60 -- Converts the result to minutes

end NUMINAMATH_CALUDE_pumps_emptying_time_l2544_254479


namespace NUMINAMATH_CALUDE_sum_and_count_integers_l2544_254438

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_integers (x y : ℕ) :
  x = sum_integers 40 60 ∧
  y = count_even_integers 40 60 ∧
  x + y = 1061 →
  x = 1050 := by sorry

end NUMINAMATH_CALUDE_sum_and_count_integers_l2544_254438
