import Mathlib

namespace ratio_problem_l835_83570

theorem ratio_problem (a b x m : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a / b = 4 / 5) 
  (h4 : x = a + 0.75 * a) 
  (h5 : m = b - 0.80 * b) : 
  m / x = 1 / 7 := by
sorry

end ratio_problem_l835_83570


namespace total_shelves_calculation_l835_83598

/-- Calculate the total number of shelves needed for coloring books and puzzle books --/
theorem total_shelves_calculation (initial_coloring : ℕ) (initial_puzzle : ℕ)
                                  (sold_coloring : ℕ) (sold_puzzle : ℕ)
                                  (coloring_per_shelf : ℕ) (puzzle_per_shelf : ℕ)
                                  (h1 : initial_coloring = 435)
                                  (h2 : initial_puzzle = 523)
                                  (h3 : sold_coloring = 218)
                                  (h4 : sold_puzzle = 304)
                                  (h5 : coloring_per_shelf = 17)
                                  (h6 : puzzle_per_shelf = 22) :
  (((initial_coloring - sold_coloring) + coloring_per_shelf - 1) / coloring_per_shelf +
   ((initial_puzzle - sold_puzzle) + puzzle_per_shelf - 1) / puzzle_per_shelf) = 23 := by
  sorry

#eval ((435 - 218) + 17 - 1) / 17 + ((523 - 304) + 22 - 1) / 22

end total_shelves_calculation_l835_83598


namespace min_moves_to_single_color_l835_83561

/-- Represents a move on the chessboard -/
structure Move where
  m : Nat
  n : Nat

/-- Represents the chessboard -/
def Chessboard := Fin 7 → Fin 7 → Bool

/-- Applies a move to the chessboard -/
def applyMove (board : Chessboard) (move : Move) : Chessboard :=
  sorry

/-- Checks if the board is of a single color -/
def isSingleColor (board : Chessboard) : Bool :=
  sorry

/-- Initial chessboard with alternating colors -/
def initialBoard : Chessboard :=
  sorry

/-- Theorem stating the minimum number of moves required -/
theorem min_moves_to_single_color :
  ∃ (moves : List Move),
    moves.length = 6 ∧
    isSingleColor (moves.foldl applyMove initialBoard) ∧
    ∀ (otherMoves : List Move),
      isSingleColor (otherMoves.foldl applyMove initialBoard) →
      otherMoves.length ≥ 6 :=
  sorry

end min_moves_to_single_color_l835_83561


namespace arithmetic_sequence_property_l835_83578

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. --/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 60) :
  2 * a 9 - a 10 = 12 :=
sorry

end arithmetic_sequence_property_l835_83578


namespace henry_trays_problem_l835_83577

theorem henry_trays_problem (trays_per_trip : ℕ) (trips : ℕ) (trays_second_table : ℕ) :
  trays_per_trip = 9 →
  trips = 9 →
  trays_second_table = 52 →
  trays_per_trip * trips - trays_second_table = 29 :=
by sorry

end henry_trays_problem_l835_83577


namespace xyz_value_l835_83501

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14) :
  x * y * z = 16 / 3 := by
sorry

end xyz_value_l835_83501


namespace expand_expression_l835_83547

theorem expand_expression (x y : ℝ) : (3*x + 5) * (4*y^2 + 15) = 12*x*y^2 + 45*x + 20*y^2 + 75 := by
  sorry

end expand_expression_l835_83547


namespace sqrt_x_minus_one_real_l835_83583

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_real_l835_83583


namespace at_least_three_solutions_nine_solutions_for_2019_l835_83590

/-- The number of solutions to the equation 1/x + 1/y = 1/a for positive integers x, y, and a > 1 -/
def num_solutions (a : ℕ) : ℕ := sorry

/-- The proposition that there are at least three distinct solutions for any a > 1 -/
theorem at_least_three_solutions (a : ℕ) (ha : a > 1) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℕ),
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    (1 : ℚ) / x₁ + (1 : ℚ) / y₁ = (1 : ℚ) / a ∧
    (1 : ℚ) / x₂ + (1 : ℚ) / y₂ = (1 : ℚ) / a ∧
    (1 : ℚ) / x₃ + (1 : ℚ) / y₃ = (1 : ℚ) / a :=
sorry

/-- The proposition that there are exactly 9 solutions when a = 2019 -/
theorem nine_solutions_for_2019 : num_solutions 2019 = 9 :=
sorry

end at_least_three_solutions_nine_solutions_for_2019_l835_83590


namespace insufficient_shots_l835_83596

-- Define the number of points on the circle
def n : ℕ := 29

-- Define the number of shots
def shots : ℕ := 134

-- Function to calculate binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

-- Total number of possible triangles
def total_triangles : ℕ := 
  binomial_coefficient n 3

-- Number of triangles that can be hit by one shot
def triangles_per_shot : ℕ := n - 2

-- Maximum number of triangles that can be hit by all shots
def max_hit_triangles : ℕ := 
  shots * triangles_per_shot

-- Theorem stating that 134 shots are insufficient
theorem insufficient_shots : max_hit_triangles < total_triangles := by
  sorry


end insufficient_shots_l835_83596


namespace loss_per_metre_calculation_l835_83555

/-- Calculates the loss per metre of cloth given the total metres sold, total selling price, and cost price per metre. -/
def loss_per_metre (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) : ℕ :=
  (total_metres * cost_price_per_metre - total_selling_price) / total_metres

/-- Theorem stating that given 200 metres of cloth sold for Rs. 18000 with a cost price of Rs. 95 per metre, the loss per metre is Rs. 5. -/
theorem loss_per_metre_calculation :
  loss_per_metre 200 18000 95 = 5 := by
  sorry

end loss_per_metre_calculation_l835_83555


namespace intersection_M_complement_N_l835_83522

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 > 4}

def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem intersection_M_complement_N :
  M ∩ (U \ N) = {x : ℝ | x > 3 ∨ x < -2} := by sorry

end intersection_M_complement_N_l835_83522


namespace original_average_age_l835_83580

/-- Proves that the original average age of a class is 40 years given the specified conditions. -/
theorem original_average_age (original_strength : ℕ) (new_students : ℕ) (new_average_age : ℕ) (average_decrease : ℕ) :
  original_strength = 17 →
  new_students = 17 →
  new_average_age = 32 →
  average_decrease = 4 →
  ∃ (original_average : ℕ), 
    original_average * original_strength + new_students * new_average_age = 
    (original_strength + new_students) * (original_average - average_decrease) ∧
    original_average = 40 :=
by sorry

end original_average_age_l835_83580


namespace ingredient_problem_l835_83507

/-- Represents the quantities and prices of ingredients A and B -/
structure Ingredients where
  total_quantity : ℕ
  price_a : ℕ
  price_b_base : ℕ
  price_b_decrease : ℚ
  quantity_b : ℕ

/-- The total cost function for the ingredients -/
def total_cost (i : Ingredients) : ℚ :=
  if i.quantity_b ≤ 300 then
    (i.total_quantity - i.quantity_b) * i.price_a + i.quantity_b * i.price_b_base
  else
    (i.total_quantity - i.quantity_b) * i.price_a + 
    i.quantity_b * (i.price_b_base - (i.quantity_b - 300) / 10 * i.price_b_decrease)

/-- The main theorem encompassing all parts of the problem -/
theorem ingredient_problem (i : Ingredients) 
  (h_total : i.total_quantity = 600)
  (h_price_a : i.price_a = 5)
  (h_price_b_base : i.price_b_base = 9)
  (h_price_b_decrease : i.price_b_decrease = 0.1)
  (h_quantity_b_multiple : i.quantity_b % 10 = 0) :
  (∃ (x : ℕ), x < 300 ∧ i.quantity_b = x ∧ total_cost i = 3800 → 
    i.total_quantity - x = 400 ∧ x = 200) ∧
  (∃ (x : ℕ), x > 300 ∧ i.quantity_b = x ∧ 2 * (i.total_quantity - x) ≥ x → 
    ∃ (min_cost : ℚ), min_cost = 4200 ∧ 
    ∀ (y : ℕ), y > 300 ∧ 2 * (i.total_quantity - y) ≥ y → 
      total_cost { i with quantity_b := y } ≥ min_cost) ∧
  (∃ (m : ℕ), m < 250 ∧ 
    (∀ (x : ℕ), x > 300 ∧ i.total_quantity - x ≤ m → 
      total_cost { i with quantity_b := x } ≤ 4000) ∧
    (∃ (x : ℕ), x > 300 ∧ i.total_quantity - x ≤ m ∧ 
      total_cost { i with quantity_b := x } = 4000) →
    m = 100) := by
  sorry

end ingredient_problem_l835_83507


namespace binomial_sum_l835_83528

theorem binomial_sum : Nat.choose 18 4 + Nat.choose 5 2 = 3070 := by
  sorry

end binomial_sum_l835_83528


namespace intersection_point_satisfies_equations_intersection_point_unique_l835_83518

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (75/23, -64/23)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 8 * x - 5 * y = 40

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 14

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations : 
  line1 intersection_point.1 intersection_point.2 ∧ 
  line2 intersection_point.1 intersection_point.2 := by
  sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem intersection_point_unique (x y : ℚ) : 
  line1 x y ∧ line2 x y → (x, y) = intersection_point := by
  sorry

end intersection_point_satisfies_equations_intersection_point_unique_l835_83518


namespace four_number_sequence_l835_83593

theorem four_number_sequence (x y z t : ℝ) : 
  (y - x = z - y) →  -- arithmetic sequence condition
  (z^2 = y * t) →    -- geometric sequence condition
  (x + t = 37) → 
  (y + z = 36) → 
  (x = 12 ∧ y = 16 ∧ z = 20 ∧ t = 25) := by
sorry

end four_number_sequence_l835_83593


namespace track_circumference_track_circumference_is_720_l835_83530

/-- The circumference of a circular track given specific meeting conditions of two joggers --/
theorem track_circumference : ℝ → ℝ → ℝ → Prop :=
  fun first_meet second_meet circumference =>
    let half_circumference := circumference / 2
    first_meet = 150 ∧
    second_meet = circumference - 90 ∧
    first_meet / (half_circumference - first_meet) = (half_circumference + 90) / (circumference - 90) →
    circumference = 720

/-- The main theorem stating that the track circumference is 720 yards --/
theorem track_circumference_is_720 : ∃ (first_meet second_meet : ℝ),
  track_circumference first_meet second_meet 720 := by
  sorry

end track_circumference_track_circumference_is_720_l835_83530


namespace simplify_polynomial_product_l835_83524

theorem simplify_polynomial_product (a : ℝ) : 
  (1 : ℝ) * (2 * a) * (3 * a^2) * (4 * a^3) * (5 * a^4) * (6 * a^5) = 720 * a^15 := by
  sorry

end simplify_polynomial_product_l835_83524


namespace statement_a_statement_d_l835_83513

-- Statement A
theorem statement_a (a b c : ℝ) (h1 : c ≠ 0) (h2 : a * c^2 > b * c^2) : a > b := by
  sorry

-- Statement D
theorem statement_d (a b : ℝ) (h : a > b ∧ b > 0) : a + 1/b > b + 1/a := by
  sorry

end statement_a_statement_d_l835_83513


namespace rent_comparison_l835_83505

theorem rent_comparison (last_year_earnings : ℝ) : 
  let last_year_rent := 0.25 * last_year_earnings
  let this_year_earnings := last_year_earnings * 1.35
  let this_year_rent := 0.40 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 216 := by
sorry

end rent_comparison_l835_83505


namespace geometric_series_sum_l835_83512

/-- The sum of a geometric series with first term 3, common ratio -2, and last term 768 is 513. -/
theorem geometric_series_sum : 
  ∀ (n : ℕ) (a : ℝ) (r : ℝ) (S : ℝ),
  a = 3 →
  r = -2 →
  a * r^(n-1) = 768 →
  S = (a * (1 - r^n)) / (1 - r) →
  S = 513 := by
sorry

end geometric_series_sum_l835_83512


namespace problem_triangle_integer_lengths_l835_83542

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths of line segments
    that can be drawn from a vertex to the opposite side in a right triangle -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle in the problem -/
def problemTriangle : RightTriangle :=
  { de := 15, ef := 36 }

/-- Theorem stating that the number of distinct integer lengths
    in the problem triangle is 24 -/
theorem problem_triangle_integer_lengths :
  countIntegerLengths problemTriangle = 24 := by
  sorry

end problem_triangle_integer_lengths_l835_83542


namespace shaded_area_of_square_with_circles_l835_83568

/-- Given a square with side length 24 inches and three circles, each tangent to two sides of the square
    and one adjacent circle, the shaded area (area not covered by the circles) is 576 - 108π square inches. -/
theorem shaded_area_of_square_with_circles (side : ℝ) (circles : ℕ) : 
  side = 24 → circles = 3 → (side^2 - circles * (side/4)^2 * Real.pi) = 576 - 108 * Real.pi := by
  sorry

end shaded_area_of_square_with_circles_l835_83568


namespace geometric_series_ratio_l835_83558

theorem geometric_series_ratio (a : ℝ) (r : ℝ) (h : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4 / (1 - r))) → r = 1/2 := by
  sorry

end geometric_series_ratio_l835_83558


namespace emily_orange_ratio_l835_83588

/-- Given the following conditions about oranges:
  * Emily has some times as many oranges as Sandra
  * Sandra has 3 times as many oranges as Betty
  * Betty has 12 oranges
  * Emily has 252 oranges
Prove that Emily has 7 times more oranges than Sandra. -/
theorem emily_orange_ratio (betty_oranges sandra_oranges emily_oranges : ℕ) 
  (h1 : sandra_oranges = 3 * betty_oranges)
  (h2 : betty_oranges = 12)
  (h3 : emily_oranges = 252) :
  emily_oranges / sandra_oranges = 7 := by
  sorry

end emily_orange_ratio_l835_83588


namespace nested_fraction_evaluation_l835_83563

theorem nested_fraction_evaluation :
  1 / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
sorry

end nested_fraction_evaluation_l835_83563


namespace snyder_cookies_l835_83527

/-- Given that Mrs. Snyder made a total of 86 cookies, with only red and pink colors,
    and 50 of them are pink, prove that she made 36 red cookies. -/
theorem snyder_cookies (total : ℕ) (pink : ℕ) (red : ℕ) : 
  total = 86 → pink = 50 → total = pink + red → red = 36 := by
  sorry

end snyder_cookies_l835_83527


namespace logarithm_equation_l835_83589

theorem logarithm_equation (x : ℝ) :
  1 - Real.log 5 = (1/3) * (Real.log (1/2) + Real.log x + (1/3) * Real.log 5) →
  x = 16 / Real.rpow 5 (1/3) :=
by sorry

end logarithm_equation_l835_83589


namespace davids_english_marks_l835_83551

/-- Given David's marks in various subjects and his average, prove his marks in English --/
theorem davids_english_marks :
  let math_marks : ℕ := 95
  let physics_marks : ℕ := 82
  let chemistry_marks : ℕ := 97
  let biology_marks : ℕ := 95
  let average_marks : ℕ := 93
  let total_subjects : ℕ := 5
  let total_marks : ℕ := average_marks * total_subjects
  let known_marks_sum : ℕ := math_marks + physics_marks + chemistry_marks + biology_marks
  let english_marks : ℕ := total_marks - known_marks_sum
  english_marks = 96 := by
sorry

end davids_english_marks_l835_83551


namespace smallest_whole_number_solution_l835_83545

theorem smallest_whole_number_solution : 
  (∀ n : ℕ, n < 6 → (2 : ℚ) / 5 + (n : ℚ) / 9 ≤ 1) ∧ 
  ((2 : ℚ) / 5 + (6 : ℚ) / 9 > 1) := by
  sorry

end smallest_whole_number_solution_l835_83545


namespace melanie_dimes_value_l835_83572

def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4
def dime_value : ℚ := 0.1

def total_dimes : ℕ := initial_dimes + dimes_from_dad + dimes_from_mom

theorem melanie_dimes_value :
  (total_dimes : ℚ) * dime_value = 1.9 := by sorry

end melanie_dimes_value_l835_83572


namespace final_position_theorem_supplement_angle_beta_theorem_l835_83552

-- Define the initial position of point A
def initial_position : Int := -5

-- Define the movement of point A
def move_right : Int := 4
def move_left : Int := 1

-- Define the angle α
def angle_alpha : Int := 40

-- Theorem for the final position of point A
theorem final_position_theorem :
  initial_position + move_right - move_left = -2 := by sorry

-- Theorem for the supplement of angle β
theorem supplement_angle_beta_theorem :
  180 - (90 - angle_alpha) = 130 := by sorry

end final_position_theorem_supplement_angle_beta_theorem_l835_83552


namespace dress_final_price_l835_83599

/-- The final price of a dress after discounts and taxes -/
def final_price (d : ℝ) : ℝ :=
  let discount_price := d * (1 - 0.45)
  let staff_discount_price := discount_price * (1 - 0.40)
  let employee_month_price := staff_discount_price * (1 - 0.10)
  let local_tax_price := employee_month_price * (1 + 0.08)
  local_tax_price * (1 + 0.05)

/-- Theorem stating the final price of the dress -/
theorem dress_final_price (d : ℝ) :
  final_price d = 0.3549 * d := by
  sorry

end dress_final_price_l835_83599


namespace consecutive_integers_sum_l835_83508

theorem consecutive_integers_sum (m : ℤ) :
  let sequence := [m, m+1, m+2, m+3, m+4, m+5, m+6]
  (sequence.sum - (sequence.take 3).sum) = 4*m + 18 := by
  sorry

end consecutive_integers_sum_l835_83508


namespace simple_interest_principal_calculation_l835_83516

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation 
  (rate : ℝ) (interest : ℝ) (time : ℝ) :
  rate = 4.783950617283951 →
  interest = 155 →
  time = 4 →
  ∃ (principal : ℝ), 
    (principal * rate * time) / 100 = interest ∧ 
    abs (principal - 810.13) < 0.01 :=
by sorry

end simple_interest_principal_calculation_l835_83516


namespace pond_capacity_l835_83546

theorem pond_capacity 
  (normal_rate : ℝ) 
  (drought_factor : ℝ) 
  (fill_time : ℝ) 
  (h1 : normal_rate = 6) 
  (h2 : drought_factor = 2/3) 
  (h3 : fill_time = 50) : 
  normal_rate * drought_factor * fill_time = 200 := by
  sorry

end pond_capacity_l835_83546


namespace edward_book_spending_l835_83514

/-- Given Edward's initial amount, amount spent on pens, and remaining amount,
    prove that the amount spent on books is $6. -/
theorem edward_book_spending (initial : ℕ) (spent_on_pens : ℕ) (remaining : ℕ) 
    (h1 : initial = 41)
    (h2 : spent_on_pens = 16)
    (h3 : remaining = 19) :
    initial - remaining - spent_on_pens = 6 := by
  sorry

end edward_book_spending_l835_83514


namespace line_equations_l835_83560

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := 2 * x + 4 * y - 1 = 0

-- Define a general line passing through a point
def line_through_point (a b c : ℝ) (x₀ y₀ : ℝ) : Prop :=
  a * x₀ + b * y₀ + c = 0

-- Define parallel lines
def parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

-- Define perpendicular lines
def perpendicular_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * a₂ + b₁ * b₂ = 0

theorem line_equations :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (∀ x y, l₁ x y ↔ a₁ * x + b₁ * y + c₁ = 0) ∧
    line_through_point a₂ b₂ c₂ 1 (-2) ∧
    ((parallel_lines a₁ b₁ c₁ a₂ b₂ c₂ →
      ∀ x y, a₂ * x + b₂ * y + c₂ = 0 ↔ x + 2 * y + 3 = 0) ∧
     (perpendicular_lines a₁ b₁ c₁ a₂ b₂ c₂ →
      ∀ x y, a₂ * x + b₂ * y + c₂ = 0 ↔ 2 * x - y - 4 = 0)) :=
by sorry

end line_equations_l835_83560


namespace total_fat_is_3600_l835_83521

/-- Represents the fat content of different fish types and the number of fish served -/
structure FishData where
  herring_fat : ℕ
  eel_fat : ℕ
  pike_fat_extra : ℕ
  fish_count : ℕ

/-- Calculates the total fat content from all fish served -/
def total_fat (data : FishData) : ℕ :=
  data.fish_count * data.herring_fat +
  data.fish_count * data.eel_fat +
  data.fish_count * (data.eel_fat + data.pike_fat_extra)

/-- Theorem stating that the total fat content is 3600 oz given the specific fish data -/
theorem total_fat_is_3600 (data : FishData)
  (h1 : data.herring_fat = 40)
  (h2 : data.eel_fat = 20)
  (h3 : data.pike_fat_extra = 10)
  (h4 : data.fish_count = 40) :
  total_fat data = 3600 := by
  sorry

end total_fat_is_3600_l835_83521


namespace brush_width_ratio_l835_83591

theorem brush_width_ratio (w l b : ℝ) (h1 : w = 4) (h2 : l = 9) : 
  b * Real.sqrt (w^2 + l^2) = (w * l) / 3 → l / b = 3 * Real.sqrt 97 / 4 := by
  sorry

end brush_width_ratio_l835_83591


namespace total_oranges_approx_45_l835_83517

/-- The number of bags of oranges -/
def num_bags : ℝ := 1.956521739

/-- The number of pounds of oranges per bag -/
def pounds_per_bag : ℝ := 23.0

/-- The total pounds of oranges -/
def total_pounds : ℝ := num_bags * pounds_per_bag

/-- Theorem stating that the total pounds of oranges is approximately 45.00 pounds -/
theorem total_oranges_approx_45 :
  ∃ ε > 0, |total_pounds - 45.00| < ε :=
sorry

end total_oranges_approx_45_l835_83517


namespace rectangle_diagonal_pi_irrational_l835_83536

theorem rectangle_diagonal_pi_irrational 
  (m n p q : ℤ) 
  (hn : n ≠ 0) 
  (hq : q ≠ 0) :
  let l : ℚ := m / n
  let w : ℚ := p / q
  let d : ℝ := Real.sqrt ((l * l + w * w : ℚ) : ℝ)
  Irrational (π * d) := by
sorry

end rectangle_diagonal_pi_irrational_l835_83536


namespace root_of_polynomial_l835_83550

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 16*x^2 + 4

-- State the theorem
theorem root_of_polynomial :
  -- The polynomial is monic
  (∀ x, p x = x^4 - 16*x^2 + 4) ∧
  -- The polynomial has degree 4
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- √5 + √3 is a root of the polynomial
  p (Real.sqrt 5 + Real.sqrt 3) = 0 :=
by sorry

end root_of_polynomial_l835_83550


namespace mica_sandwich_options_l835_83538

-- Define the types of sandwich components
def BreadTypes : ℕ := 6
def MeatTypes : ℕ := 7
def CheeseTypes : ℕ := 6

-- Define the restricted combinations
def TurkeySwissCombinations : ℕ := BreadTypes
def SourdoughChickenCombinations : ℕ := CheeseTypes
def SalamiRyeCombinations : ℕ := CheeseTypes

-- Define the total number of restricted combinations
def TotalRestrictedCombinations : ℕ :=
  TurkeySwissCombinations + SourdoughChickenCombinations + SalamiRyeCombinations

-- Define the total number of possible sandwich combinations
def TotalPossibleCombinations : ℕ := BreadTypes * MeatTypes * CheeseTypes

-- Define the number of sandwiches Mica could order
def MicaSandwichOptions : ℕ := TotalPossibleCombinations - TotalRestrictedCombinations

-- Theorem statement
theorem mica_sandwich_options :
  MicaSandwichOptions = 234 := by sorry

end mica_sandwich_options_l835_83538


namespace cyclingProblemSolution_l835_83573

/-- Natalia's cycling distances over four days --/
def cyclingProblem (tuesday : ℕ) : Prop :=
  let monday : ℕ := 40
  let wednesday : ℕ := tuesday / 2
  let thursday : ℕ := monday + wednesday
  monday + tuesday + wednesday + thursday = 180

/-- The solution to Natalia's cycling problem --/
theorem cyclingProblemSolution : ∃ (tuesday : ℕ), cyclingProblem tuesday ∧ tuesday = 33 := by
  sorry

#check cyclingProblemSolution

end cyclingProblemSolution_l835_83573


namespace cube_roll_probability_l835_83500

theorem cube_roll_probability (total_faces green_faces : ℕ) 
  (h1 : total_faces = 6)
  (h2 : green_faces = 3)
  (h3 : total_faces - green_faces = 3) : 
  (green_faces : ℚ) / total_faces = 1 / 2 := by
sorry

end cube_roll_probability_l835_83500


namespace commodity_cost_proof_l835_83594

def total_cost (price1 price2 : ℕ) : ℕ := price1 + price2

theorem commodity_cost_proof (price1 price2 : ℕ) 
  (h1 : price1 = 477)
  (h2 : price1 = price2 + 127) :
  total_cost price1 price2 = 827 := by
  sorry

end commodity_cost_proof_l835_83594


namespace max_sales_revenue_l835_83587

/-- Sales price as a function of time -/
def sales_price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

/-- Daily sales volume as a function of time -/
def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

/-- Daily sales revenue as a function of time -/
def sales_revenue (t : ℕ) : ℝ :=
  sales_price t * sales_volume t

/-- The maximum daily sales revenue and the day it occurs -/
theorem max_sales_revenue :
  (∃ (t : ℕ), 0 < t ∧ t ≤ 30 ∧ 
    sales_revenue t = 1125 ∧
    ∀ (s : ℕ), 0 < s ∧ s ≤ 30 → sales_revenue s ≤ sales_revenue t) ∧
  (∀ (s : ℕ), 0 < s ∧ s ≤ 30 ∧ sales_revenue s = 1125 → s = 25) :=
by sorry

end max_sales_revenue_l835_83587


namespace factorization_of_x4_plus_81_l835_83553

theorem factorization_of_x4_plus_81 (x : ℝ) : 
  x^4 + 81 = (x^2 + 3*x + 4.5) * (x^2 - 3*x + 4.5) := by sorry

end factorization_of_x4_plus_81_l835_83553


namespace equilateral_triangle_count_l835_83581

/-- Represents a line in the 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (x y : ℝ) → a * x + b * y + c = 0

/-- Generates the set of lines given by y = k, y = x + 3k, and y = -x + 3k for k from -6 to 6 --/
def generateLines : Set Line := sorry

/-- Checks if three lines form an equilateral triangle of side 1 --/
def formEquilateralTriangle (l1 l2 l3 : Line) : Prop := sorry

/-- Counts the number of equilateral triangles formed by the intersection of lines --/
def countEquilateralTriangles (lines : Set Line) : ℕ := sorry

/-- The main theorem stating that the number of equilateral triangles is 444 --/
theorem equilateral_triangle_count :
  ∃ (lines : Set Line), 
    lines = generateLines ∧ 
    countEquilateralTriangles lines = 444 := by sorry

end equilateral_triangle_count_l835_83581


namespace arrangement_count_l835_83575

def valid_arrangements (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun k => (n.choose k) ^ 3)

theorem arrangement_count :
  valid_arrangements 4 =
    (Finset.sum (Finset.range 5) (fun k =>
      (Nat.choose 4 k) ^ 3)) :=
by sorry

end arrangement_count_l835_83575


namespace rectangle_area_l835_83565

/-- Given a square with side length 15 and a rectangle with length 18 and diagonal 27,
    prove that the area of the rectangle is 216 when its perimeter equals the square's perimeter. -/
theorem rectangle_area (square_side : ℝ) (rect_length rect_diagonal : ℝ) :
  square_side = 15 →
  rect_length = 18 →
  rect_diagonal = 27 →
  4 * square_side = 2 * rect_length + 2 * (rect_diagonal ^ 2 - rect_length ^ 2).sqrt →
  rect_length * (rect_diagonal ^ 2 - rect_length ^ 2).sqrt = 216 := by
  sorry

end rectangle_area_l835_83565


namespace lucy_snack_bar_total_cost_l835_83526

/-- The cost of a single sandwich at Lucy's Snack Bar -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda at Lucy's Snack Bar -/
def soda_cost : ℕ := 3

/-- The number of sandwiches Lucy wants to buy -/
def num_sandwiches : ℕ := 7

/-- The number of sodas Lucy wants to buy -/
def num_sodas : ℕ := 8

/-- The theorem stating that the total cost of Lucy's purchase is $52 -/
theorem lucy_snack_bar_total_cost : 
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = 52 := by
  sorry

end lucy_snack_bar_total_cost_l835_83526


namespace pythagorean_triple_example_l835_83511

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_example :
  (isPythagoreanTriple 6 8 10) ∧
  ¬(isPythagoreanTriple 2 3 4) ∧
  ¬(isPythagoreanTriple 4 5 6) ∧
  ¬(isPythagoreanTriple 7 8 9) :=
by sorry

end pythagorean_triple_example_l835_83511


namespace simplify_expression_l835_83554

theorem simplify_expression (x y : ℝ) :
  5 * x - 3 * y + 9 * x^2 + 8 - (4 - 5 * x + 3 * y - 9 * x^2) =
  18 * x^2 + 10 * x - 6 * y + 4 := by
  sorry

end simplify_expression_l835_83554


namespace smallest_money_sum_l835_83531

/-- Represents a sum of money in pounds, shillings, pence, and farthings -/
structure Money where
  pounds : ℕ
  shillings : ℕ
  pence : ℕ
  farthings : ℕ
  shillings_valid : shillings < 20
  pence_valid : pence < 12
  farthings_valid : farthings < 4

/-- Checks if a list of digits contains each of 1 to 9 exactly once -/
def valid_digits (digits : List ℕ) : Prop :=
  digits.length = 9 ∧ (∀ d, d ∈ digits → d ≥ 1 ∧ d ≤ 9) ∧ digits.Nodup

/-- Converts a Money value to its total value in farthings -/
def to_farthings (m : Money) : ℕ :=
  m.pounds * 960 + m.shillings * 48 + m.pence * 4 + m.farthings

/-- The theorem to be proved -/
theorem smallest_money_sum :
  ∃ (m : Money) (digits : List ℕ),
    valid_digits digits ∧
    to_farthings m = to_farthings ⟨2567, 18, 9, 3, by sorry, by sorry, by sorry⟩ ∧
    (∀ (m' : Money) (digits' : List ℕ),
      valid_digits digits' →
      to_farthings m' ≥ to_farthings m) :=
by sorry

end smallest_money_sum_l835_83531


namespace sum_difference_theorem_l835_83515

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (n : ℕ) : ℕ :=
  let m := n % 5
  if m < 3 then n - m else n + (5 - m)

def emma_sum (n : ℕ) : ℕ :=
  List.range n |> List.map round_to_nearest_five |> List.sum

theorem sum_difference_theorem :
  sum_to_n 100 - emma_sum 100 = 4750 := by sorry

end sum_difference_theorem_l835_83515


namespace sine_cosine_roots_l835_83585

theorem sine_cosine_roots (α : Real) (m : Real) : 
  α ∈ Set.Ioo 0 (2 * Real.pi) →
  (∃ (x y : Real), x = Real.sin α ∧ y = Real.cos α ∧
    2 * x^2 - (Real.sqrt 3 + 1) * x + m / 3 = 0 ∧
    2 * y^2 - (Real.sqrt 3 + 1) * y + m / 3 = 0) →
  m = 3 * Real.sqrt 3 / 2 := by
sorry

end sine_cosine_roots_l835_83585


namespace banana_apple_equivalence_l835_83506

-- Define the worth of bananas in terms of apples
def banana_worth (b : ℚ) : ℚ := 
  (12 : ℚ) / ((3 / 4) * 16)

-- Theorem statement
theorem banana_apple_equivalence : 
  banana_worth ((1 / 3) * 9) = 3 := by
  sorry

end banana_apple_equivalence_l835_83506


namespace three_digit_numbers_count_l835_83557

def Digits : Finset Nat := {1, 2, 3, 4}

theorem three_digit_numbers_count : 
  Finset.card (Finset.product (Finset.product Digits Digits) Digits) = 64 := by
  sorry

end three_digit_numbers_count_l835_83557


namespace new_house_cost_l835_83523

def first_house_cost : ℝ := 100000

def value_increase_percentage : ℝ := 0.25

def new_house_down_payment_percentage : ℝ := 0.25

theorem new_house_cost (old_house_value : ℝ) (new_house_cost : ℝ) : 
  old_house_value = first_house_cost * (1 + value_increase_percentage) ∧
  old_house_value = new_house_cost * new_house_down_payment_percentage →
  new_house_cost = 500000 := by
  sorry

end new_house_cost_l835_83523


namespace diane_age_when_condition_met_l835_83548

/-- Represents the ages of Diane, Alex, and Allison at the time when the condition is met -/
structure Ages where
  diane : ℕ
  alex : ℕ
  allison : ℕ

/-- Checks if the given ages satisfy the condition -/
def satisfiesCondition (ages : Ages) : Prop :=
  ages.diane = ages.alex / 2 ∧ ages.diane = 2 * ages.allison

/-- Represents the current ages of Diane, Alex, and Allison -/
structure CurrentAges where
  diane : ℕ
  alexPlusAllison : ℕ

/-- Theorem stating that Diane will be 78 when the condition is met -/
theorem diane_age_when_condition_met (current : CurrentAges)
    (h1 : current.diane = 16)
    (h2 : current.alexPlusAllison = 47) :
    ∃ (ages : Ages), satisfiesCondition ages ∧ ages.diane = 78 :=
  sorry

end diane_age_when_condition_met_l835_83548


namespace three_digit_congruence_solutions_l835_83584

theorem three_digit_congruence_solutions : 
  (Finset.filter (fun y : ℕ => 100 ≤ y ∧ y ≤ 999 ∧ (1945 * y + 243) % 17 = 605 % 17) 
    (Finset.range 1000)).card = 53 := by
  sorry

end three_digit_congruence_solutions_l835_83584


namespace function_inequality_l835_83567

-- Define a function f that is differentiable on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the condition that 2f(x) - f'(x) > 0 for all x in ℝ
variable (h : ∀ x : ℝ, 2 * f x - deriv f x > 0)

-- State the theorem
theorem function_inequality : f 1 > f 2 / Real.exp 2 := by
  sorry

end function_inequality_l835_83567


namespace cube_volume_doubled_edges_l835_83597

theorem cube_volume_doubled_edges (a : ℝ) (h : a > 0) :
  (2 * a)^3 = 8 * a^3 := by sorry

end cube_volume_doubled_edges_l835_83597


namespace monochromatic_four_clique_exists_l835_83529

/-- A two-color edge coloring of a complete graph. -/
def TwoColorEdgeColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- The existence of a monochromatic 4-clique in a two-color edge coloring of K_18. -/
theorem monochromatic_four_clique_exists :
  ∀ (coloring : TwoColorEdgeColoring 18),
  ∃ (vertices : Fin 4 → Fin 18),
    (∀ (i j : Fin 4), i ≠ j →
      coloring (vertices i) (vertices j) = coloring (vertices 0) (vertices 1)) :=
by sorry

end monochromatic_four_clique_exists_l835_83529


namespace max_volume_cutout_length_l835_83541

/-- The side length of the original square sheet of iron in centimeters -/
def original_side_length : ℝ := 36

/-- The volume of the box as a function of the side length of the cut-out square -/
def volume (x : ℝ) : ℝ := x * (original_side_length - 2*x)^2

/-- The derivative of the volume function -/
def volume_derivative (x : ℝ) : ℝ := 12 * (18 - x) * (6 - x)

theorem max_volume_cutout_length :
  ∃ (x : ℝ), 0 < x ∧ x < original_side_length / 2 ∧
  volume_derivative x = 0 ∧
  (∀ y, 0 < y → y < original_side_length / 2 → volume y ≤ volume x) ∧
  x = 6 := by sorry

end max_volume_cutout_length_l835_83541


namespace optimal_discount_savings_l835_83556

def initial_order : ℝ := 15000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def apply_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

def discount_set1 : List ℝ := [0.25, 0.15, 0.10]
def discount_set2 : List ℝ := [0.30, 0.10, 0.05]

theorem optimal_discount_savings :
  apply_discounts initial_order discount_set2 - apply_discounts initial_order discount_set1 = 371.25 := by
  sorry

end optimal_discount_savings_l835_83556


namespace vacation_cost_division_l835_83502

theorem vacation_cost_division (total_cost : ℕ) (cost_difference : ℕ) : 
  total_cost = 720 →
  (total_cost / 4 + cost_difference) * 3 = total_cost →
  cost_difference = 60 →
  3 = total_cost / (total_cost / 4 + cost_difference) :=
by
  sorry

end vacation_cost_division_l835_83502


namespace smallest_four_digit_remainder_five_mod_six_l835_83564

theorem smallest_four_digit_remainder_five_mod_six : 
  ∃ (n : ℕ), 
    (1000 ≤ n ∧ n ≤ 9999) ∧ 
    (n % 6 = 5) ∧
    (∀ m, (1000 ≤ m ∧ m ≤ 9999) → (m % 6 = 5) → n ≤ m) ∧
    n = 1001 :=
by sorry

end smallest_four_digit_remainder_five_mod_six_l835_83564


namespace one_pair_probability_l835_83574

/-- The number of colors of socks -/
def num_colors : ℕ := 5

/-- The number of socks per color -/
def socks_per_color : ℕ := 2

/-- The total number of socks -/
def total_socks : ℕ := num_colors * socks_per_color

/-- The number of socks drawn -/
def drawn_socks : ℕ := 5

/-- The probability of drawing exactly one pair of the same color and three different colored socks -/
def probability_one_pair : ℚ := 20 / 21

theorem one_pair_probability :
  probability_one_pair = (num_colors.choose 4 * 4 * 8) / total_socks.choose drawn_socks :=
by sorry

end one_pair_probability_l835_83574


namespace expand_and_evaluate_l835_83549

theorem expand_and_evaluate : 
  ∀ x : ℝ, (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 ∧ 
  (let x : ℝ := 5; 4 * x^2 + 4 * x - 24) = 96 := by sorry

end expand_and_evaluate_l835_83549


namespace inequality_solution_set_not_equal_function_always_negative_implies_k_range_negation_of_inequality_solution_set_is_true_l835_83534

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) * Real.sqrt (x^2 - 3*x + 2) ≥ 0

-- Define the solution set
def solution_set : Set ℝ := {x | x ≥ 2}

-- Define the function y
def y (k x : ℝ) : ℝ := k * x^2 - k * x - 1

theorem inequality_solution_set_not_equal : 
  {x : ℝ | inequality x} ≠ solution_set := by sorry

theorem function_always_negative_implies_k_range (k : ℝ) :
  (∀ x, y k x < 0) → -4 < k ∧ k ≤ 0 := by sorry

theorem negation_of_inequality_solution_set_is_true :
  ¬({x : ℝ | inequality x} = solution_set) := by sorry

end inequality_solution_set_not_equal_function_always_negative_implies_k_range_negation_of_inequality_solution_set_is_true_l835_83534


namespace average_of_w_and_x_l835_83566

theorem average_of_w_and_x (w x y : ℝ) 
  (h1 : 7 / w + 7 / x = 7 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = y / 2 := by
  sorry

end average_of_w_and_x_l835_83566


namespace average_speed_bicycle_and_walk_l835_83504

/-- Proves that the average speed of a pedestrian who rode a bicycle for 40 minutes at 5 m/s
    and then walked for 2 hours at 5 km/h is 8.25 km/h. -/
theorem average_speed_bicycle_and_walk (
  bicycle_time : Real) (bicycle_speed : Real) (walk_time : Real) (walk_speed : Real)
  (h1 : bicycle_time = 40 / 60) -- 40 minutes in hours
  (h2 : bicycle_speed = 5 * 3.6) -- 5 m/s converted to km/h
  (h3 : walk_time = 2) -- 2 hours
  (h4 : walk_speed = 5) -- 5 km/h
  : (bicycle_time * bicycle_speed + walk_time * walk_speed) / (bicycle_time + walk_time) = 8.25 := by
  sorry

end average_speed_bicycle_and_walk_l835_83504


namespace doughnuts_left_l835_83540

theorem doughnuts_left (total_doughnuts : ℕ) (staff_count : ℕ) (doughnuts_per_staff : ℕ) :
  total_doughnuts = 50 →
  staff_count = 19 →
  doughnuts_per_staff = 2 →
  total_doughnuts - (staff_count * doughnuts_per_staff) = 12 := by
sorry

end doughnuts_left_l835_83540


namespace expression_equality_l835_83586

theorem expression_equality : 
  (2 / 3 * Real.sqrt 15 - Real.sqrt 20) / (1 / 3 * Real.sqrt 5) = 2 * Real.sqrt 3 - 6 := by
  sorry

end expression_equality_l835_83586


namespace one_lamp_position_l835_83579

/-- Represents a position on the 5x5 grid -/
structure Position where
  x : Fin 5
  y : Fin 5

/-- Represents the state of the 5x5 grid of lamps -/
def Grid := Fin 5 → Fin 5 → Bool

/-- The operation of toggling a lamp and its adjacent lamps -/
def toggle (grid : Grid) (pos : Position) : Grid := sorry

/-- Checks if only one lamp is on in the grid -/
def onlyOneLampOn (grid : Grid) : Bool := sorry

/-- Checks if a position is either the center or directly diagonal to the center -/
def isCenterOrDiagonal (pos : Position) : Bool := sorry

/-- The main theorem: If only one lamp is on after a sequence of toggle operations,
    it must be in the center or directly diagonal to the center -/
theorem one_lamp_position (grid : Grid) (pos : Position) :
  (∃ (ops : List Position), onlyOneLampOn (ops.foldl toggle grid)) →
  (onlyOneLampOn grid ∧ grid pos.x pos.y = true) →
  isCenterOrDiagonal pos := sorry

end one_lamp_position_l835_83579


namespace min_value_sum_l835_83509

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 1 / Real.rpow 2 (1/3) :=
sorry

end min_value_sum_l835_83509


namespace mean_equality_implies_z_value_l835_83543

theorem mean_equality_implies_z_value : 
  (∃ z : ℝ, (8 + 10 + 24) / 3 = (16 + z) / 2) → 
  (∃ z : ℝ, (8 + 10 + 24) / 3 = (16 + z) / 2 ∧ z = 12) :=
by sorry

end mean_equality_implies_z_value_l835_83543


namespace inequality_solution_set_l835_83537

theorem inequality_solution_set (k : ℝ) :
  let S := {x : ℝ | k * x^2 - (k + 2) * x + 2 < 0}
  (k = 0 → S = {x : ℝ | x < 1}) ∧
  (0 < k ∧ k < 2 → S = {x : ℝ | x < 1 ∨ x > 2/k}) ∧
  (k = 2 → S = {x : ℝ | x ≠ 1}) ∧
  (k > 2 → S = {x : ℝ | x < 2/k ∨ x > 1}) ∧
  (k < 0 → S = {x : ℝ | 2/k < x ∧ x < 1}) :=
by sorry

end inequality_solution_set_l835_83537


namespace sum_squares_quadratic_roots_l835_83595

theorem sum_squares_quadratic_roots : 
  let a := 1
  let b := -10
  let c := 9
  let s₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let s₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  s₁^2 + s₂^2 = 82 :=
by sorry

end sum_squares_quadratic_roots_l835_83595


namespace largest_smallest_valid_numbers_l835_83539

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000000 ∧ n < 10000000000) ∧  -- 10-digit number
  (n % 11 = 0) ∧                        -- divisible by 11
  (∀ i j, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10))  -- no repeated digits

theorem largest_smallest_valid_numbers :
  (∀ n : ℕ, is_valid_number n → n ≤ 9876524130) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ 1024375869) ∧
  is_valid_number 9876524130 ∧
  is_valid_number 1024375869 :=
sorry

end largest_smallest_valid_numbers_l835_83539


namespace M_union_N_equals_R_l835_83569

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set N
def N : Set ℝ := {x | |x| < Real.sqrt 5}

-- Theorem statement
theorem M_union_N_equals_R : M ∪ N = Set.univ := by sorry

end M_union_N_equals_R_l835_83569


namespace spinster_cat_ratio_l835_83520

theorem spinster_cat_ratio :
  ∀ (s c : ℕ),
    s = 12 →
    c = s + 42 →
    ∃ (n : ℕ), n * s = 2 * c ∧ 9 * s = n * c :=
by
  sorry

end spinster_cat_ratio_l835_83520


namespace simplify_expression_l835_83535

theorem simplify_expression (x y : ℝ) : 3 * x^2 - 2 * x * y - 3 * x^2 + 4 * x * y - 1 = 2 * x * y - 1 := by
  sorry

end simplify_expression_l835_83535


namespace money_conditions_l835_83592

theorem money_conditions (a b : ℝ) 
  (h1 : b - 4*a < 78)
  (h2 : 6*a - b = 36)
  : a < 57 ∧ b > -36 := by
  sorry

end money_conditions_l835_83592


namespace right_triangle_side_lengths_l835_83525

theorem right_triangle_side_lengths (x : ℝ) :
  (((2*x + 2)^2 = (x + 4)^2 + (x + 2)^2 ∨ (x + 4)^2 = (2*x + 2)^2 + (x + 2)^2) ∧ 
   x > 0 ∧ 2*x + 2 > 0 ∧ x + 4 > 0 ∧ x + 2 > 0) ↔ 
  (x = 4 ∨ x = 1) :=
sorry

end right_triangle_side_lengths_l835_83525


namespace cubic_term_simplification_l835_83562

theorem cubic_term_simplification (a : ℝ) : a^3 + 7*a^3 - 5*a^3 = 3*a^3 := by
  sorry

end cubic_term_simplification_l835_83562


namespace total_cost_calculation_l835_83544

def cabinet_price : ℝ := 1200
def cabinet_discount : ℝ := 0.15
def dining_table_price : ℝ := 1800
def dining_table_discount : ℝ := 0.20
def sofa_price : ℝ := 2500
def sofa_discount : ℝ := 0.10
def sales_tax : ℝ := 0.08

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_discounted_price : ℝ :=
  discounted_price cabinet_price cabinet_discount +
  discounted_price dining_table_price dining_table_discount +
  discounted_price sofa_price sofa_discount

def total_cost : ℝ :=
  total_discounted_price * (1 + sales_tax)

theorem total_cost_calculation :
  total_cost = 5086.80 := by
  sorry

end total_cost_calculation_l835_83544


namespace percentage_increase_l835_83533

theorem percentage_increase (x : ℝ) (h : x = 105.6) :
  (x - 88) / 88 * 100 = 20 := by
  sorry

end percentage_increase_l835_83533


namespace root_quadratic_equation_l835_83510

theorem root_quadratic_equation (a : ℝ) : 2 * a^2 = a + 4 → 4 * a^2 - 2 * a = 8 := by
  sorry

end root_quadratic_equation_l835_83510


namespace decimal_representation_of_three_fortieths_l835_83519

theorem decimal_representation_of_three_fortieths : (3 : ℚ) / 40 = 0.075 := by
  sorry

end decimal_representation_of_three_fortieths_l835_83519


namespace f_decreasing_range_l835_83571

/-- A piecewise function f(x) defined by a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

/-- The theorem stating the range of a for which f is strictly decreasing -/
theorem f_decreasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) ↔ 
  (0 < a ∧ a ≤ 1/4) :=
sorry

end f_decreasing_range_l835_83571


namespace cake_box_height_l835_83582

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of items that can fit along a dimension -/
def maxItemsAlongDimension (containerSize itemSize : ℕ) : ℕ :=
  containerSize / itemSize

/-- Represents the problem of determining the height of cake boxes in a carton -/
def cakeBoxProblem (cartonDims : Dimensions) (cakeBoxBase : Dimensions) (maxBoxes : ℕ) : Prop :=
  let boxesAlongLength := maxItemsAlongDimension cartonDims.length cakeBoxBase.length
  let boxesAlongWidth := maxItemsAlongDimension cartonDims.width cakeBoxBase.width
  let boxesPerLayer := boxesAlongLength * boxesAlongWidth
  let numLayers := maxBoxes / boxesPerLayer
  let cakeBoxHeight := cartonDims.height / numLayers
  cakeBoxHeight = 5

/-- The main theorem stating that the height of a cake box is 5 inches -/
theorem cake_box_height :
  cakeBoxProblem
    (Dimensions.mk 25 42 60)  -- Carton dimensions
    (Dimensions.mk 8 7 0)     -- Cake box base dimensions (height is unknown)
    210                       -- Maximum number of boxes
  := by sorry

end cake_box_height_l835_83582


namespace matrix_inverse_proof_l835_83576

def A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -4; -3, 2]

theorem matrix_inverse_proof :
  ∃ (B : Matrix (Fin 2) (Fin 2) ℝ),
    B = !![1, 2; 1.5, 3.5] ∧ A * B = 1 ∧ B * A = 1 := by
  sorry

end matrix_inverse_proof_l835_83576


namespace polynomial_factorization_l835_83503

theorem polynomial_factorization (x : ℝ) : 
  x^6 + 6*x^5 + 15*x^4 + 20*x^3 + 15*x^2 + 6*x + 1 = (x+1)^6 := by
  sorry

end polynomial_factorization_l835_83503


namespace expression_value_l835_83559

theorem expression_value (a b : ℝ) 
  (ha : a = 2 * Real.sin (45 * π / 180) + 1)
  (hb : b = 2 * Real.cos (45 * π / 180) - 1) :
  ((a^2 + b^2) / (2*a*b) - 1) / ((a^2 - b^2) / (a^2*b + a*b^2)) = 1 := by
  sorry

end expression_value_l835_83559


namespace two_digit_divisible_by_digit_product_l835_83532

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def ones_digit (n : ℕ) : ℕ := n % 10

def divisible_by_digit_product (n : ℕ) : Prop :=
  is_two_digit n ∧ n % (tens_digit n * ones_digit n) = 0

theorem two_digit_divisible_by_digit_product :
  {n : ℕ | divisible_by_digit_product n} = {11, 12, 24, 36, 15} :=
by sorry

end two_digit_divisible_by_digit_product_l835_83532
