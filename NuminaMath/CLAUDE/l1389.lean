import Mathlib

namespace joshua_fruit_profit_l1389_138995

/-- Calculates the total profit in cents for Joshua's fruit sales --/
def fruit_profit (orange_qty : ℕ) (apple_qty : ℕ) (banana_qty : ℕ)
  (orange_cost : ℚ) (apple_cost : ℚ) (banana_cost : ℚ)
  (orange_sell : ℚ) (apple_sell : ℚ) (banana_sell : ℚ)
  (discount_threshold : ℕ) (discount_rate : ℚ) : ℕ :=
  let orange_total_cost := orange_qty * orange_cost
  let apple_total_cost := if apple_qty ≥ discount_threshold
    then apple_qty * (apple_cost * (1 - discount_rate))
    else apple_qty * apple_cost
  let banana_total_cost := if banana_qty ≥ discount_threshold
    then banana_qty * (banana_cost * (1 - discount_rate))
    else banana_qty * banana_cost
  let total_cost := orange_total_cost + apple_total_cost + banana_total_cost
  let total_revenue := orange_qty * orange_sell + apple_qty * apple_sell + banana_qty * banana_sell
  let profit := total_revenue - total_cost
  (profit * 100).floor.toNat

/-- Theorem stating that Joshua's profit is 2035 cents --/
theorem joshua_fruit_profit :
  fruit_profit 25 40 50 0.5 0.65 0.25 0.6 0.75 0.45 30 0.1 = 2035 := by
  sorry

end joshua_fruit_profit_l1389_138995


namespace range_when_p_true_range_when_p_or_q_and_p_and_q_true_l1389_138913

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 15 > 0}
def B : Set ℝ := {x | x - 6 < 0}

-- Define propositions p and q
def p (m : ℝ) : Prop := m ∈ A
def q (m : ℝ) : Prop := m ∈ B

-- Theorem for the first part
theorem range_when_p_true :
  {m : ℝ | p m} = {x | x < -3 ∨ x > 5} :=
sorry

-- Theorem for the second part
theorem range_when_p_or_q_and_p_and_q_true :
  {m : ℝ | (p m ∨ q m) ∧ (p m ∧ q m)} = {x | x < -3} :=
sorry

end range_when_p_true_range_when_p_or_q_and_p_and_q_true_l1389_138913


namespace range_of_a_l1389_138932

theorem range_of_a (a b c : ℝ) 
  (h1 : b^2 + c^2 = -a^2 + 14*a + 5) 
  (h2 : b*c = a^2 - 2*a + 10) : 
  1 ≤ a ∧ a ≤ 5 :=
by sorry

end range_of_a_l1389_138932


namespace quadruple_base_exponent_l1389_138967

theorem quadruple_base_exponent (a b x y s : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) (hs : s > 0)
  (h1 : s = (4*a)^(4*b))
  (h2 : s = a^b * y^b)
  (h3 : y = 4*x) : 
  x = 64 * a^3 := by sorry

end quadruple_base_exponent_l1389_138967


namespace division_problem_l1389_138957

theorem division_problem :
  let dividend : ℕ := 16698
  let divisor : ℝ := 187.46067415730337
  let quotient : ℕ := 89
  let remainder : ℕ := 14
  (dividend : ℝ) = divisor * quotient + remainder :=
by sorry

end division_problem_l1389_138957


namespace bmw_sales_l1389_138985

def total_cars : ℕ := 300
def ford_percentage : ℚ := 18 / 100
def nissan_percentage : ℚ := 25 / 100
def chevrolet_percentage : ℚ := 20 / 100

theorem bmw_sales : 
  let other_brands_percentage := ford_percentage + nissan_percentage + chevrolet_percentage
  let bmw_percentage := 1 - other_brands_percentage
  ↑⌊bmw_percentage * total_cars⌋ = 111 := by sorry

end bmw_sales_l1389_138985


namespace log_product_equals_one_l1389_138959

theorem log_product_equals_one : Real.log 2 / Real.log 5 * (Real.log 25 / Real.log 4) = 1 := by
  sorry

end log_product_equals_one_l1389_138959


namespace simplify_expression_l1389_138925

theorem simplify_expression (x : ℝ) : 1 - (2 - (1 + (2 - (1 - x)))) = 1 - x := by
  sorry

end simplify_expression_l1389_138925


namespace midpoint_chain_l1389_138968

/-- Given a line segment XY, we define points G, H, I, and J as follows:
  G is the midpoint of XY
  H is the midpoint of XG
  I is the midpoint of XH
  J is the midpoint of XI
  If XJ = 4, then XY = 64 -/
theorem midpoint_chain (X Y G H I J : ℝ) : 
  (G = (X + Y) / 2) →  -- G is midpoint of XY
  (H = (X + G) / 2) →  -- H is midpoint of XG
  (I = (X + H) / 2) →  -- I is midpoint of XH
  (J = (X + I) / 2) →  -- J is midpoint of XI
  (J - X = 4) →        -- XJ = 4
  (Y - X = 64) :=      -- XY = 64
by sorry

end midpoint_chain_l1389_138968


namespace min_value_of_sum_l1389_138908

theorem min_value_of_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y + 2 * x + y = 4 → 
  x + y ≥ 2 * Real.sqrt 6 - 3 ∧ 
  ∃ x y, x > 0 ∧ y > 0 ∧ x * y + 2 * x + y = 4 ∧ x + y = 2 * Real.sqrt 6 - 3 :=
by sorry

end min_value_of_sum_l1389_138908


namespace alfonso_work_weeks_l1389_138911

def hourly_rate : ℝ := 6
def monday_hours : ℝ := 2
def tuesday_hours : ℝ := 3
def wednesday_hours : ℝ := 4
def thursday_hours : ℝ := 2
def friday_hours : ℝ := 3
def helmet_cost : ℝ := 340
def gloves_cost : ℝ := 45
def current_savings : ℝ := 40
def miscellaneous_expenses : ℝ := 20

def weekly_hours : ℝ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
def weekly_earnings : ℝ := weekly_hours * hourly_rate
def total_cost : ℝ := helmet_cost + gloves_cost + miscellaneous_expenses
def additional_earnings_needed : ℝ := total_cost - current_savings

theorem alfonso_work_weeks : 
  ∃ n : ℕ, n * weekly_earnings ≥ additional_earnings_needed ∧ 
           (n - 1) * weekly_earnings < additional_earnings_needed ∧
           n = 5 :=
sorry

end alfonso_work_weeks_l1389_138911


namespace pet_food_inventory_l1389_138903

theorem pet_food_inventory (dog_food : ℕ) (difference : ℕ) (cat_food : ℕ) : 
  dog_food = 600 → 
  dog_food = cat_food + difference → 
  difference = 273 →
  cat_food = 327 := by
  sorry

end pet_food_inventory_l1389_138903


namespace parallel_vectors_magnitude_l1389_138931

/-- Given two vectors a and b in ℝ², where a is parallel to b,
    prove that the magnitude of b is 2√5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.1 = -2 →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end parallel_vectors_magnitude_l1389_138931


namespace machine_depletion_rate_l1389_138982

theorem machine_depletion_rate 
  (initial_value : ℝ) 
  (final_value : ℝ) 
  (time : ℝ) 
  (h1 : initial_value = 400)
  (h2 : final_value = 225)
  (h3 : time = 2) :
  ∃ (rate : ℝ), 
    final_value = initial_value * (1 - rate) ^ time ∧ 
    rate = 0.25 := by
sorry

end machine_depletion_rate_l1389_138982


namespace evaluate_64_to_5_6_l1389_138918

theorem evaluate_64_to_5_6 : (64 : ℝ) ^ (5/6) = 32 := by sorry

end evaluate_64_to_5_6_l1389_138918


namespace problem_statement_l1389_138952

theorem problem_statement (x y z : ℝ) 
  (h1 : x ≤ y) (h2 : y ≤ z) 
  (h3 : x + y + z = 12) 
  (h4 : x^2 + y^2 + z^2 = 54) : 
  x ≤ 3 ∧ z ≥ 5 ∧ 
  9 ≤ x * y ∧ x * y ≤ 25 ∧ 
  9 ≤ y * z ∧ y * z ≤ 25 ∧ 
  9 ≤ z * x ∧ z * x ≤ 25 := by
  sorry

end problem_statement_l1389_138952


namespace households_without_car_or_bike_l1389_138944

theorem households_without_car_or_bike 
  (total : ℕ) 
  (both : ℕ) 
  (with_car : ℕ) 
  (bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : both = 20)
  (h3 : with_car = 44)
  (h4 : bike_only = 35) :
  total - (with_car + bike_only) = 11 :=
by sorry

end households_without_car_or_bike_l1389_138944


namespace polynomial_transformation_l1389_138991

theorem polynomial_transformation (g : ℝ → ℝ) :
  (∀ x, g (x^2 - 2) = x^4 - 6*x^2 + 8) →
  (∀ x, g (x^2 - 1) = x^4 - 4*x^2 + 7) :=
by
  sorry

end polynomial_transformation_l1389_138991


namespace product_negative_from_positive_sum_negative_quotient_l1389_138980

theorem product_negative_from_positive_sum_negative_quotient
  (a b : ℝ) (h_sum : a + b > 0) (h_quotient : a / b < 0) :
  a * b < 0 :=
by sorry

end product_negative_from_positive_sum_negative_quotient_l1389_138980


namespace smallest_x_value_l1389_138961

theorem smallest_x_value (x : ℝ) : 
  (((5*x - 20)/(4*x - 5))^3 + ((5*x - 20)/(4*x - 5))^2 - ((5*x - 20)/(4*x - 5)) - 15 = 0) → 
  (∀ y : ℝ, (((5*y - 20)/(4*y - 5))^3 + ((5*y - 20)/(4*y - 5))^2 - ((5*y - 20)/(4*y - 5)) - 15 = 0) → 
  x ≤ y) → 
  x = 10/3 :=
by sorry

end smallest_x_value_l1389_138961


namespace smallest_sum_c_d_l1389_138966

theorem smallest_sum_c_d (c d : ℝ) : 
  c > 0 → d > 0 → 
  (∃ x : ℝ, x^2 + c*x + 3*d = 0) → 
  (∃ x : ℝ, x^2 + 3*d*x + c = 0) → 
  c + d ≥ 16/3 ∧ 
  ∃ c₀ d₀ : ℝ, c₀ > 0 ∧ d₀ > 0 ∧ 
    (∃ x : ℝ, x^2 + c₀*x + 3*d₀ = 0) ∧ 
    (∃ x : ℝ, x^2 + 3*d₀*x + c₀ = 0) ∧ 
    c₀ + d₀ = 16/3 :=
by sorry

end smallest_sum_c_d_l1389_138966


namespace pattern_A_cannot_fold_into_cube_l1389_138973

/-- Represents a pattern of squares -/
inductive Pattern
  | A  -- Five squares in a cross shape
  | B  -- Four squares in a "T" shape
  | C  -- Six squares in a "T" shape with an additional square
  | D  -- Three squares in a straight line

/-- Number of squares in a pattern -/
def squareCount (p : Pattern) : Nat :=
  match p with
  | .A => 5
  | .B => 4
  | .C => 6
  | .D => 3

/-- Number of squares required to form a cube -/
def cubeSquareCount : Nat := 6

/-- Checks if a pattern can be folded into a cube -/
def canFoldIntoCube (p : Pattern) : Prop :=
  squareCount p = cubeSquareCount ∧ 
  (p ≠ Pattern.A) -- Pattern A cannot be closed even with 5 squares

/-- Theorem: Pattern A cannot be folded into a cube -/
theorem pattern_A_cannot_fold_into_cube : 
  ¬ (canFoldIntoCube Pattern.A) := by
  sorry


end pattern_A_cannot_fold_into_cube_l1389_138973


namespace twentyFifthInBase6_l1389_138958

/-- Converts a natural number to its representation in base 6 --/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Converts a list of digits in base 6 to a natural number --/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 6 + d) 0

theorem twentyFifthInBase6 : fromBase6 [4, 1] = 25 := by
  sorry

#eval toBase6 25  -- Should output [4, 1]
#eval fromBase6 [4, 1]  -- Should output 25

end twentyFifthInBase6_l1389_138958


namespace max_xy_constraint_l1389_138979

theorem max_xy_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_constraint : 4 * x + 9 * y = 6) :
  x * y ≤ 1 / 4 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 4 * x₀ + 9 * y₀ = 6 ∧ x₀ * y₀ = 1 / 4 :=
sorry

end max_xy_constraint_l1389_138979


namespace special_trapezoid_angle_l1389_138970

/-- A trapezoid with special properties -/
structure SpecialTrapezoid where
  /-- The diagonals intersect at a right angle -/
  diagonals_right_angle : Bool
  /-- One diagonal is equal to the midsegment -/
  diagonal_equals_midsegment : Bool

/-- The angle formed by the special diagonal and the bases of the trapezoid -/
def diagonal_base_angle (t : SpecialTrapezoid) : Real :=
  sorry

/-- Theorem: In a special trapezoid, the angle between the special diagonal and the bases is 60° -/
theorem special_trapezoid_angle (t : SpecialTrapezoid) 
  (h1 : t.diagonals_right_angle = true) 
  (h2 : t.diagonal_equals_midsegment = true) : 
  diagonal_base_angle t = 60 := by
  sorry

end special_trapezoid_angle_l1389_138970


namespace negation_of_existence_negation_of_quadratic_inequality_l1389_138997

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality : 
  (¬ ∃ x : ℝ, x^2 + 4*x + 6 < 0) ↔ (∀ x : ℝ, x^2 + 4*x + 6 ≥ 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l1389_138997


namespace line_points_k_value_l1389_138923

/-- Given a line with equation x = 2y + 5, if (m, n) and (m + 1, n + k) are two points on this line, then k = 1/2 -/
theorem line_points_k_value (m n k : ℝ) : 
  (m = 2 * n + 5) →  -- (m, n) is on the line
  (m + 1 = 2 * (n + k) + 5) →  -- (m + 1, n + k) is on the line
  k = 1 / 2 := by sorry

end line_points_k_value_l1389_138923


namespace number_calculation_l1389_138935

theorem number_calculation (x : ℝ) : ((x + 1.4) / 3 - 0.7) * 9 = 5.4 ↔ x = 2.5 := by
  sorry

end number_calculation_l1389_138935


namespace movie_theater_screens_l1389_138960

theorem movie_theater_screens (open_hours : ℕ) (movie_duration : ℕ) (total_movies : ℕ) : 
  open_hours = 8 → movie_duration = 2 → total_movies = 24 → 
  (total_movies * movie_duration) / open_hours = 6 :=
by
  sorry

#check movie_theater_screens

end movie_theater_screens_l1389_138960


namespace work_completion_l1389_138990

/-- Given that 36 men can complete a piece of work in 18 days, and a different number of men can
    complete the same work in 24 days, prove that the number of men in the second group is 27. -/
theorem work_completion (total_work : ℕ) (men_group1 men_group2 : ℕ) (days_group1 days_group2 : ℕ) :
  men_group1 = 36 →
  days_group1 = 18 →
  days_group2 = 24 →
  total_work = men_group1 * days_group1 →
  total_work = men_group2 * days_group2 →
  men_group2 = 27 := by
  sorry

end work_completion_l1389_138990


namespace equal_distribution_classroom_l1389_138926

/-- Proves that given 4 classrooms, 56 boys, and 44 girls, with an equal distribution of boys and girls across all classrooms, the total number of students in each classroom is 25. -/
theorem equal_distribution_classroom (num_classrooms : ℕ) (num_boys : ℕ) (num_girls : ℕ) 
  (h1 : num_classrooms = 4)
  (h2 : num_boys = 56)
  (h3 : num_girls = 44)
  (h4 : num_boys % num_classrooms = 0)
  (h5 : num_girls % num_classrooms = 0) :
  (num_boys / num_classrooms) + (num_girls / num_classrooms) = 25 :=
by sorry

end equal_distribution_classroom_l1389_138926


namespace arithmetic_geometric_ratio_l1389_138955

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The terms a_1 + 1, a_3 + 2, and a_5 + 3 form a geometric sequence with ratio q -/
def GeometricSubsequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 3 + 2) = (a 1 + 1) * q ∧ (a 5 + 3) = (a 3 + 2) * q

theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ArithmeticSequence a) (h2 : GeometricSubsequence a q) : q = 1 := by
  sorry

end arithmetic_geometric_ratio_l1389_138955


namespace breath_holding_increase_l1389_138907

theorem breath_holding_increase (initial_time : ℝ) (final_time : ℝ) : 
  initial_time = 10 →
  final_time = 60 →
  let first_week := initial_time * 2
  let second_week := first_week * 2
  (final_time - second_week) / second_week * 100 = 50 := by
sorry

end breath_holding_increase_l1389_138907


namespace oil_price_reduction_l1389_138929

theorem oil_price_reduction (original_price : ℝ) : 
  (original_price > 0) →
  (1100 = (1100 / original_price) * original_price) →
  (1100 = ((1100 / original_price) + 5) * (0.75 * original_price)) →
  (0.75 * original_price = 55) := by
sorry

end oil_price_reduction_l1389_138929


namespace one_and_two_red_mutually_exclusive_not_opposing_l1389_138938

/-- Represents the number of red balls drawn -/
inductive RedBallsDrawn
  | zero
  | one
  | two
  | three

/-- The probability of drawing exactly one red ball -/
def prob_one_red : ℝ := sorry

/-- The probability of drawing exactly two red balls -/
def prob_two_red : ℝ := sorry

/-- The total number of balls in the bag -/
def total_balls : ℕ := 8

/-- The number of red balls in the bag -/
def red_balls : ℕ := 5

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

theorem one_and_two_red_mutually_exclusive_not_opposing :
  (prob_one_red * prob_two_red = 0) ∧ (prob_one_red + prob_two_red < 1) := by
  sorry

end one_and_two_red_mutually_exclusive_not_opposing_l1389_138938


namespace BC_length_l1389_138902

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A, B, C, B', C', and D
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def B' : ℝ × ℝ := sorry
def C' : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry

-- Define the conditions
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω
axiom C_on_ω : C ∈ ω
axiom BC_is_diameter : sorry -- BC is a diameter of ω
axiom B'C'_parallel_BC : sorry -- B'C' is parallel to BC
axiom B'C'_tangent_ω : sorry -- B'C' is tangent to ω at D
axiom B'D_length : dist B' D = 4
axiom C'D_length : dist C' D = 6

-- Define the theorem
theorem BC_length : dist B C = 24/5 := by sorry

end BC_length_l1389_138902


namespace share_face_value_l1389_138927

theorem share_face_value (dividend_rate : ℝ) (desired_return : ℝ) (market_value : ℝ) :
  dividend_rate = 0.09 →
  desired_return = 0.12 →
  market_value = 36.00000000000001 →
  (desired_return / dividend_rate) * market_value = 48.00000000000001 :=
by
  sorry

end share_face_value_l1389_138927


namespace unique_line_count_for_p_2_l1389_138977

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a type for lines in a plane
def Line : Type := Point → Point → Prop

-- Define a function to count the number of intersection points
def count_intersections (lines : List Line) : ℕ := sorry

-- Define a function to check if lines intersect at exactly p points
def intersect_at_p_points (lines : List Line) (p : ℕ) : Prop :=
  count_intersections lines = p

-- Theorem: When p = 2, there is a unique number of lines (3) that intersect at exactly p points
theorem unique_line_count_for_p_2 :
  ∃! n : ℕ, ∃ lines : List Line, intersect_at_p_points lines 2 ∧ lines.length = n :=
sorry

end unique_line_count_for_p_2_l1389_138977


namespace problem_solution_l1389_138972

theorem problem_solution (x y z : ℝ) 
  (h1 : 2 * x + y + z = 14)
  (h2 : 2 * x + y = 7)
  (h3 : x + 2 * y + Real.sqrt z = 10) :
  (x + y - z) / 3 = (-4 - Real.sqrt 7) / 3 := by
  sorry

end problem_solution_l1389_138972


namespace fast_food_cost_l1389_138909

/-- Represents the cost of items at a fast food restaurant -/
structure FastFoodCost where
  hamburger : ℝ
  milkshake : ℝ
  fries : ℝ

/-- Given the costs of different combinations, prove the cost of 2 hamburgers, 2 milkshakes, and 2 fries -/
theorem fast_food_cost (c : FastFoodCost) 
  (eq1 : 3 * c.hamburger + 5 * c.milkshake + c.fries = 23.5)
  (eq2 : 5 * c.hamburger + 9 * c.milkshake + c.fries = 39.5) :
  2 * c.hamburger + 2 * c.milkshake + 2 * c.fries = 15 := by
  sorry

end fast_food_cost_l1389_138909


namespace martin_tv_purchase_l1389_138924

/-- The initial amount Martin decided to spend on a TV -/
def initial_amount : ℝ := 1000

/-- The discount amount applied before the percentage discount -/
def initial_discount : ℝ := 100

/-- The percentage discount applied after the initial discount -/
def percentage_discount : ℝ := 0.20

/-- The difference between the initial amount and the final price -/
def price_difference : ℝ := 280

theorem martin_tv_purchase :
  initial_amount = 1000 ∧
  initial_amount - (initial_amount - initial_discount - 
    percentage_discount * (initial_amount - initial_discount)) = price_difference := by
  sorry

end martin_tv_purchase_l1389_138924


namespace count_dominoes_l1389_138978

/-- The number of different (noncongruent) dominoes in an m × n array -/
def num_dominoes (m n : ℕ) : ℚ :=
  m * n - m^2 / 2 + m / 2 - 1

/-- Theorem: The number of different (noncongruent) dominoes in an m × n array -/
theorem count_dominoes (m n : ℕ) (h : 0 < m ∧ m ≤ n) :
  num_dominoes m n = m * n - m^2 / 2 + m / 2 - 1 := by
  sorry

end count_dominoes_l1389_138978


namespace expression_meaningful_iff_l1389_138919

def meaningful_expression (x : ℝ) : Prop :=
  x ≠ -5

theorem expression_meaningful_iff (x : ℝ) :
  meaningful_expression x ↔ x ≠ -5 :=
by
  sorry

end expression_meaningful_iff_l1389_138919


namespace line_circle_intersections_l1389_138900

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 4 * x + 9 * y = 7

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the number of intersection points
def num_intersections : ℕ := 2

-- Theorem statement
theorem line_circle_intersections :
  ∃ (p q : ℝ × ℝ), 
    p ≠ q ∧ 
    line_eq p.1 p.2 ∧ circle_eq p.1 p.2 ∧
    line_eq q.1 q.2 ∧ circle_eq q.1 q.2 ∧
    (∀ (r : ℝ × ℝ), line_eq r.1 r.2 ∧ circle_eq r.1 r.2 → r = p ∨ r = q) :=
by sorry

end line_circle_intersections_l1389_138900


namespace f_2012_is_zero_l1389_138921

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2012_is_zero 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_f2 : f 2 = 0) 
  (h_period : ∀ x, f (x + 4) = f x + f 4) : 
  f 2012 = 0 := by
  sorry

end f_2012_is_zero_l1389_138921


namespace fraction_meaningfulness_l1389_138969

theorem fraction_meaningfulness (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x + 1)) ↔ x ≠ -1 := by
  sorry

end fraction_meaningfulness_l1389_138969


namespace sum_remainder_of_arithmetic_sequence_l1389_138988

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : List ℕ :=
  let n := (aₙ - a₁) / d + 1
  List.range n |>.map (λ i => a₁ + i * d)

theorem sum_remainder_of_arithmetic_sequence : 
  (arithmetic_sequence 3 8 283).sum % 8 = 4 := by
  sorry

end sum_remainder_of_arithmetic_sequence_l1389_138988


namespace triangle_properties_l1389_138976

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a = 2 ∧
  cos B = 3/5 →
  (b = 4 → sin A = 2/5) ∧
  (1/2 * a * c * sin B = 4 → b = Real.sqrt 17 ∧ c = 5) := by
sorry

end triangle_properties_l1389_138976


namespace min_value_expression_l1389_138941

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 20 - 8 * Real.sqrt 5 := by
  sorry

end min_value_expression_l1389_138941


namespace endpoint_sum_is_twelve_l1389_138914

/-- Given a line segment with one endpoint (6, -2) and midpoint (3, 5),
    the sum of the coordinates of the other endpoint is 12. -/
theorem endpoint_sum_is_twelve (x y : ℝ) : 
  (6 + x) / 2 = 3 → (-2 + y) / 2 = 5 → x + y = 12 := by
  sorry

end endpoint_sum_is_twelve_l1389_138914


namespace undefined_values_l1389_138981

theorem undefined_values (x : ℝ) :
  (x^2 - 21*x + 110 = 0) ↔ (x = 10 ∨ x = 11) := by sorry

end undefined_values_l1389_138981


namespace sum_of_squares_2870_l1389_138983

theorem sum_of_squares_2870 :
  ∃! (n : ℕ), n > 0 ∧ n * (n + 1) * (2 * n + 1) / 6 = 2870 :=
by sorry

end sum_of_squares_2870_l1389_138983


namespace segments_form_triangle_l1389_138962

/-- Triangle inequality theorem: the sum of the lengths of any two sides 
    of a triangle must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Function to check if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that line segments of lengths 13, 12, and 20 can form a triangle -/
theorem segments_form_triangle : can_form_triangle 13 12 20 := by
  sorry


end segments_form_triangle_l1389_138962


namespace x_varies_as_z_two_thirds_l1389_138996

/-- Given that x varies directly as the square of y, and y varies directly as the cube root of z,
    prove that x varies as z^(2/3). -/
theorem x_varies_as_z_two_thirds
  (x y z : ℝ)
  (h1 : ∃ k : ℝ, ∀ y, x = k * y^2)
  (h2 : ∃ j : ℝ, ∀ z, y = j * z^(1/3))
  : ∃ m : ℝ, x = m * z^(2/3) :=
sorry

end x_varies_as_z_two_thirds_l1389_138996


namespace work_fraction_left_l1389_138912

theorem work_fraction_left (p q : ℕ) (h1 : p = 15) (h2 : q = 20) : 
  1 - 4 * (1 / p.cast + 1 / q.cast) = 8 / 15 := by
  sorry

end work_fraction_left_l1389_138912


namespace last_three_digits_of_3_to_1000_l1389_138901

theorem last_three_digits_of_3_to_1000 (h : 3^200 ≡ 1 [ZMOD 500]) :
  3^1000 ≡ 1 [ZMOD 1000] :=
sorry

end last_three_digits_of_3_to_1000_l1389_138901


namespace clarinet_players_count_l1389_138974

/-- Represents the number of people in an orchestra section -/
structure OrchestraSection where
  count : ℕ

/-- Represents the composition of an orchestra -/
structure Orchestra where
  total : ℕ
  percussion : OrchestraSection
  brass : OrchestraSection
  strings : OrchestraSection
  flutes : OrchestraSection
  maestro : OrchestraSection
  clarinets : OrchestraSection

/-- Given an orchestra with the specified composition, prove that the number of clarinet players is 3 -/
theorem clarinet_players_count (o : Orchestra) 
  (h1 : o.total = 21)
  (h2 : o.percussion.count = 1)
  (h3 : o.brass.count = 7)
  (h4 : o.strings.count = 5)
  (h5 : o.flutes.count = 4)
  (h6 : o.maestro.count = 1)
  (h7 : o.total = o.percussion.count + o.brass.count + o.strings.count + o.flutes.count + o.maestro.count + o.clarinets.count) :
  o.clarinets.count = 3 := by
  sorry

end clarinet_players_count_l1389_138974


namespace certain_number_proof_l1389_138916

theorem certain_number_proof (k : ℕ) (x : ℕ) 
  (h1 : 823435 % (15^k) = 0)
  (h2 : x^k - k^5 = 1) : x = 2 := by
  sorry

end certain_number_proof_l1389_138916


namespace real_equal_roots_l1389_138910

theorem real_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - k * x + x + 8 = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - k * y + y + 8 = 0 → y = x) ↔ 
  (k = 9 ∨ k = -7) :=
sorry

end real_equal_roots_l1389_138910


namespace square_area_ratio_l1389_138994

/-- The ratio of areas between a smaller square and a larger square, given specific conditions -/
theorem square_area_ratio : ∀ (r : ℝ) (y : ℝ),
  r > 0 →  -- radius of circumscribed circle is positive
  r = 4 * Real.sqrt 2 →  -- radius of circumscribed circle
  y > 0 →  -- half side length of smaller square is positive
  y * (3 * y - 8 * Real.sqrt 2) = 0 →  -- condition for diagonal touching circle
  (2 * y)^2 / 8^2 = 8 / 9 := by
  sorry

end square_area_ratio_l1389_138994


namespace imaginary_part_of_z_l1389_138940

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 4 + 2 * Complex.I) :
  z.im = -1 := by sorry

end imaginary_part_of_z_l1389_138940


namespace function_composition_property_l1389_138971

def iteratedFunction (f : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, n => n
| (i + 1), n => f (iteratedFunction f i n)

theorem function_composition_property (k : ℕ) :
  (k ≥ 2) ↔
  (∃ (f g : ℕ → ℕ),
    (∀ (S : Set ℕ), (∃ n, g n ∉ S) → Set.Infinite S) ∧
    (∀ n, iteratedFunction f (g n) n = f n + k)) :=
sorry

end function_composition_property_l1389_138971


namespace casting_theorem_l1389_138992

def men : ℕ := 7
def women : ℕ := 5
def male_roles : ℕ := 3
def either_gender_roles : ℕ := 2
def total_roles : ℕ := male_roles + either_gender_roles

def casting_combinations : ℕ := (men.choose male_roles) * ((men + women - male_roles).choose either_gender_roles)

theorem casting_theorem : casting_combinations = 15120 := by sorry

end casting_theorem_l1389_138992


namespace sum_of_three_numbers_l1389_138937

theorem sum_of_three_numbers (A B C : ℚ) : 
  B = 30 → 
  A / B = 2 / 3 → 
  B / C = 5 / 8 → 
  A + B + C = 98 := by
sorry

end sum_of_three_numbers_l1389_138937


namespace quiz_variance_is_64_l1389_138987

/-- Represents a multiple-choice quiz -/
structure Quiz where
  num_questions : ℕ
  options_per_question : ℕ
  points_per_correct : ℕ
  total_points : ℕ
  correct_probability : ℝ

/-- Calculates the variance of a student's score in the quiz -/
def quiz_score_variance (q : Quiz) : ℝ :=
  q.num_questions * q.correct_probability * (1 - q.correct_probability) * q.points_per_correct^2

/-- Theorem stating that the variance of the student's score in the given quiz is 64 -/
theorem quiz_variance_is_64 : 
  let q : Quiz := {
    num_questions := 25,
    options_per_question := 4,
    points_per_correct := 4,
    total_points := 100,
    correct_probability := 0.8
  }
  quiz_score_variance q = 64 := by
  sorry

end quiz_variance_is_64_l1389_138987


namespace cubic_equation_value_l1389_138933

theorem cubic_equation_value (x : ℝ) (h : 3 * x^2 - x = 1) :
  6 * x^3 + 7 * x^2 - 5 * x + 2008 = 2011 := by
  sorry

end cubic_equation_value_l1389_138933


namespace father_son_age_sum_father_son_age_proof_l1389_138915

theorem father_son_age_sum : ℕ → ℕ → ℕ
  | father_age, son_age =>
    father_age + 2 * son_age

theorem father_son_age_proof (father_age son_age : ℕ) 
  (h1 : father_age = 40) 
  (h2 : son_age = 15) : 
  father_son_age_sum father_age son_age = 70 := by
  sorry

end father_son_age_sum_father_son_age_proof_l1389_138915


namespace jacob_wage_is_6_l1389_138906

-- Define the given conditions
def jake_earnings_multiplier : ℚ := 3
def jake_total_earnings : ℚ := 720
def work_days : ℕ := 5
def hours_per_day : ℕ := 8

-- Define Jake's hourly wage
def jake_hourly_wage : ℚ := jake_total_earnings / (work_days * hours_per_day)

-- Define Jacob's hourly wage
def jacob_hourly_wage : ℚ := jake_hourly_wage / jake_earnings_multiplier

-- Theorem to prove
theorem jacob_wage_is_6 : jacob_hourly_wage = 6 := by
  sorry

end jacob_wage_is_6_l1389_138906


namespace quadratic_tangent_theorem_l1389_138950

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determines if a quadratic function is tangent to the x-axis -/
def isTangentToXAxis (f : QuadraticFunction) : Prop :=
  f.b^2 - 4*f.a*f.c = 0

/-- Determines if the vertex of a quadratic function is a minimum point -/
def hasMinimumVertex (f : QuadraticFunction) : Prop :=
  f.a > 0

/-- The main theorem to be proved -/
theorem quadratic_tangent_theorem :
  ∀ (d : ℝ),
  let f : QuadraticFunction := ⟨3, 12, d⟩
  isTangentToXAxis f →
  d = 12 ∧ hasMinimumVertex f := by
  sorry

end quadratic_tangent_theorem_l1389_138950


namespace increasing_f_implies_a_in_closed_interval_l1389_138951

/-- A function f : ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The cosine function -/
noncomputable def cos : ℝ → ℝ := Real.cos

/-- The function f(x) = x - a * cos(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * cos x

theorem increasing_f_implies_a_in_closed_interval :
  ∀ a : ℝ, IsIncreasing (f a) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end increasing_f_implies_a_in_closed_interval_l1389_138951


namespace probability_age_20_to_40_l1389_138922

theorem probability_age_20_to_40 (total : ℕ) (below_20 : ℕ) (between_20_30 : ℕ) (between_30_40 : ℕ) :
  total = 350 →
  below_20 = 120 →
  between_20_30 = 105 →
  between_30_40 = 85 →
  (between_20_30 + between_30_40 : ℚ) / total = 19 / 35 := by
  sorry

end probability_age_20_to_40_l1389_138922


namespace train_speed_l1389_138999

/-- Proves that a train with given length and time to cross a pole has a specific speed -/
theorem train_speed (length : Real) (time : Real) (speed : Real) : 
  length = 400.032 →
  time = 9 →
  speed = (length / 1000) / time * 3600 →
  speed = 160.0128 := by
  sorry

#check train_speed

end train_speed_l1389_138999


namespace power_product_equals_sum_of_exponents_l1389_138975

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end power_product_equals_sum_of_exponents_l1389_138975


namespace product_base_8_units_digit_l1389_138989

def base_10_product : ℕ := 123 * 57

def base_8_units_digit (n : ℕ) : ℕ := n % 8

theorem product_base_8_units_digit :
  base_8_units_digit base_10_product = 7 := by
  sorry

end product_base_8_units_digit_l1389_138989


namespace sqrt_equation_solution_l1389_138963

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt x + Real.sqrt 243) / Real.sqrt 75 = 2.4 → x = 27 := by
  sorry

end sqrt_equation_solution_l1389_138963


namespace square_side_length_l1389_138953

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 12) (h2 : area = side ^ 2) :
  side = 2 * Real.sqrt 3 := by
  sorry

end square_side_length_l1389_138953


namespace exists_monochromatic_isosceles_triangle_l1389_138934

-- Define a color type
inductive Color
  | Red
  | Green
  | Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define an isosceles triangle
def isIsoscelesTriangle (p q r : Point) : Prop := sorry

-- Theorem statement
theorem exists_monochromatic_isosceles_triangle :
  ∃ (p q r : Point), 
    isIsoscelesTriangle p q r ∧ 
    coloring p = coloring q ∧ 
    coloring q = coloring r := 
by sorry

end exists_monochromatic_isosceles_triangle_l1389_138934


namespace line_intersects_segment_midpoint_l1389_138964

theorem line_intersects_segment_midpoint (b : ℝ) : 
  let p1 : ℝ × ℝ := (3, 2)
  let p2 : ℝ × ℝ := (7, 6)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (b = 9) ↔ (midpoint.1 + midpoint.2 = b) :=
by sorry

end line_intersects_segment_midpoint_l1389_138964


namespace f_derivative_at_fixed_point_l1389_138965

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos x)))))))

theorem f_derivative_at_fixed_point (a : ℝ) (h : a = Real.cos a) :
  deriv f a = a^8 - 4*a^6 + 6*a^4 - 4*a^2 + 1 := by
  sorry

end f_derivative_at_fixed_point_l1389_138965


namespace larger_number_in_ratio_l1389_138949

theorem larger_number_in_ratio (a b : ℝ) : 
  a / b = 8 / 3 → a + b = 143 → max a b = 104 := by sorry

end larger_number_in_ratio_l1389_138949


namespace smallest_sum_sequence_l1389_138942

theorem smallest_sum_sequence (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (∃ d : ℤ, C - B = d ∧ B - A = d) →  -- A, B, C form an arithmetic sequence
  (∃ r : ℚ, C = r * B ∧ D = r * C) →  -- B, C, D form a geometric sequence
  C = (7 : ℚ) / 3 * B →  -- C/B = 7/3
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 →
    (∃ d : ℤ, C' - B' = d ∧ B' - A' = d) →
    (∃ r : ℚ, C' = r * B' ∧ D' = r * C') →
    C' = (7 : ℚ) / 3 * B' →
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 76 := by
sorry

end smallest_sum_sequence_l1389_138942


namespace evaluate_expression_l1389_138986

theorem evaluate_expression : (64 / 0.08) - 2.5 = 797.5 := by
  sorry

end evaluate_expression_l1389_138986


namespace four_digit_number_with_sum_14_divisible_by_14_l1389_138920

theorem four_digit_number_with_sum_14_divisible_by_14 :
  ∃ n : ℕ,
    1000 ≤ n ∧ n ≤ 9999 ∧
    (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 14) ∧
    n % 14 = 0 := by
  sorry

end four_digit_number_with_sum_14_divisible_by_14_l1389_138920


namespace shape_arrangement_possible_l1389_138905

-- Define a type for geometric shapes
structure Shape :=
  (area : ℕ)

-- Define a type for arrangements of shapes
structure Arrangement :=
  (shapes : List Shape)
  (width : ℕ)
  (height : ℕ)

-- Define the properties of the desired arrangements
def is_square_with_cutout (arr : Arrangement) : Prop :=
  arr.width = 9 ∧ arr.height = 9 ∧
  ∃ (center : Shape), center ∈ arr.shapes ∧ center.area = 9

def is_rectangle (arr : Arrangement) : Prop :=
  arr.width = 9 ∧ arr.height = 12

-- Define the given set of shapes
def given_shapes : List Shape := sorry

-- State the theorem
theorem shape_arrangement_possible :
  ∃ (arr1 arr2 : Arrangement),
    (∀ s ∈ arr1.shapes, s ∈ given_shapes) ∧
    (∀ s ∈ arr2.shapes, s ∈ given_shapes) ∧
    is_square_with_cutout arr1 ∧
    is_rectangle arr2 :=
  sorry

end shape_arrangement_possible_l1389_138905


namespace max_value_of_expression_l1389_138930

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x * y / (2 * y + 3 * x) = 1) : 
  x / 2 + y / 3 ≤ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    x₀ * y₀ / (2 * y₀ + 3 * x₀) = 1 ∧ x₀ / 2 + y₀ / 3 = 4 := by
  sorry

end max_value_of_expression_l1389_138930


namespace time_on_other_subjects_is_40_l1389_138984

/-- Represents the time spent on homework for each subject -/
structure HomeworkTime where
  total : ℝ
  math : ℝ
  science : ℝ
  history : ℝ
  english : ℝ

/-- Calculates the time spent on other subjects -/
def timeOnOtherSubjects (hw : HomeworkTime) : ℝ :=
  hw.total - (hw.math + hw.science + hw.history + hw.english)

/-- Theorem stating the time spent on other subjects is 40 minutes -/
theorem time_on_other_subjects_is_40 (hw : HomeworkTime) : 
  hw.total = 150 ∧
  hw.math = 0.20 * hw.total ∧
  hw.science = 0.25 * hw.total ∧
  hw.history = 0.10 * hw.total ∧
  hw.english = 0.15 * hw.total ∧
  hw.history ≥ 20 ∧
  hw.science ≥ 20 →
  timeOnOtherSubjects hw = 40 := by
  sorry

#check time_on_other_subjects_is_40

end time_on_other_subjects_is_40_l1389_138984


namespace bridge_length_bridge_length_problem_l1389_138917

/-- The length of a bridge given train parameters --/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

/-- Proof of the bridge length problem --/
theorem bridge_length_problem : 
  bridge_length 360 75 24 = 140 := by
  sorry

end bridge_length_bridge_length_problem_l1389_138917


namespace square_diagonal_ratio_l1389_138948

theorem square_diagonal_ratio (a b : ℝ) (h : b^2 / a^2 = 4) :
  (b * Real.sqrt 2) / (a * Real.sqrt 2) = 2 :=
by sorry

end square_diagonal_ratio_l1389_138948


namespace compare_fractions_l1389_138956

theorem compare_fractions : (-5/6 : ℚ) > -|(-8/9 : ℚ)| := by sorry

end compare_fractions_l1389_138956


namespace expression_equality_l1389_138939

theorem expression_equality : (8 : ℕ)^6 * 27^6 * 8^18 * 27^18 = 216^24 := by sorry

end expression_equality_l1389_138939


namespace inscribed_polygon_sides_l1389_138904

theorem inscribed_polygon_sides (r : ℝ) (n : ℕ) (a : ℝ) : 
  r = 1 → 
  a = 2 * Real.sin (π / n) → 
  1 < a → 
  a < Real.sqrt 2 → 
  n = 5 := by sorry

end inscribed_polygon_sides_l1389_138904


namespace angle_relation_l1389_138947

theorem angle_relation (α β : Real) (x y : Real) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos (α + β) = -4/5)
  (h4 : Real.sin β = x)
  (h5 : Real.cos α = y)
  (h6 : 4/5 < x ∧ x < 1) :
  y = -4/5 * Real.sqrt (1 - x^2) + 3/5 * x := by
  sorry

#check angle_relation

end angle_relation_l1389_138947


namespace vegetable_ghee_ratio_l1389_138936

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 950

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 850

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3640

/-- The volume of brand 'a' in the mixture -/
def volume_a : ℝ := 2.4

/-- The volume of brand 'b' in the mixture -/
def volume_b : ℝ := 1.6

/-- Theorem stating that the ratio of volumes of brand 'a' to brand 'b' is 1.5:1 -/
theorem vegetable_ghee_ratio :
  volume_a / volume_b = 1.5 ∧
  volume_a + volume_b = total_volume ∧
  weight_a * volume_a + weight_b * volume_b = total_weight :=
by sorry

end vegetable_ghee_ratio_l1389_138936


namespace system_solution_l1389_138943

theorem system_solution (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ 
  x^2 + y * Real.sqrt (x * y) = 105 ∧
  y^2 + x * Real.sqrt (y * x) = 70 →
  x = 9 ∧ y = 4 :=
by sorry

end system_solution_l1389_138943


namespace building_shadow_length_l1389_138946

/-- Given a flagstaff and a building under similar conditions, prove that the length of the shadow
    cast by the building is 28.75 m. -/
theorem building_shadow_length
  (flagstaff_height : ℝ)
  (flagstaff_shadow : ℝ)
  (building_height : ℝ)
  (h1 : flagstaff_height = 17.5)
  (h2 : flagstaff_shadow = 40.25)
  (h3 : building_height = 12.5)
  : (building_height * flagstaff_shadow) / flagstaff_height = 28.75 := by
  sorry

end building_shadow_length_l1389_138946


namespace cone_lateral_surface_area_l1389_138945

/-- The lateral surface area of a cone with base radius 5 and height 12 is 65π. -/
theorem cone_lateral_surface_area :
  let r : ℝ := 5
  let h : ℝ := 12
  let l : ℝ := Real.sqrt (r^2 + h^2)
  π * r * l = 65 * π :=
by sorry

end cone_lateral_surface_area_l1389_138945


namespace private_schools_in_B_l1389_138928

/-- Represents the three types of schools -/
inductive SchoolType
  | Public
  | Parochial
  | PrivateIndependent

/-- Represents the three districts -/
inductive District
  | A
  | B
  | C

/-- The total number of high schools -/
def total_schools : ℕ := 50

/-- The number of public schools -/
def public_schools : ℕ := 25

/-- The number of parochial schools -/
def parochial_schools : ℕ := 16

/-- The number of private independent schools -/
def private_schools : ℕ := 9

/-- The number of schools in District A -/
def schools_in_A : ℕ := 18

/-- The number of schools in District B -/
def schools_in_B : ℕ := 17

/-- Function to calculate the number of schools in District C -/
def schools_in_C : ℕ := total_schools - schools_in_A - schools_in_B

/-- Function to calculate the number of each type of school in District C -/
def schools_per_type_in_C : ℕ := schools_in_C / 3

theorem private_schools_in_B : 
  private_schools - schools_per_type_in_C = 4 := by sorry

end private_schools_in_B_l1389_138928


namespace sqrt_inequality_l1389_138998

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c ≥ d) (h4 : d > 0) : 
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end sqrt_inequality_l1389_138998


namespace matt_darius_difference_l1389_138993

/-- The scores of three friends in a table football game. -/
structure Scores where
  darius : ℕ
  matt : ℕ
  marius : ℕ

/-- The conditions of the table football game. -/
def game_conditions (s : Scores) : Prop :=
  s.darius = 10 ∧
  s.marius = s.darius + 3 ∧
  s.matt > s.darius ∧
  s.darius + s.matt + s.marius = 38

/-- The theorem stating the difference between Matt's and Darius's scores. -/
theorem matt_darius_difference (s : Scores) (h : game_conditions s) : 
  s.matt - s.darius = 5 := by
  sorry

end matt_darius_difference_l1389_138993


namespace smallest_n_square_fifth_power_l1389_138954

theorem smallest_n_square_fifth_power : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 2 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^5) ∧ 
  (∀ (x : ℕ), x > 0 → x < n → 
    (¬∃ (y : ℕ), 2 * x = y^2) ∨ 
    (¬∃ (z : ℕ), 5 * x = z^5)) ∧
  n = 5000 := by
sorry

end smallest_n_square_fifth_power_l1389_138954
