import Mathlib

namespace NUMINAMATH_CALUDE_value_of_T_l821_82109

theorem value_of_T : ∃ T : ℝ, (1/3 : ℝ) * (1/6 : ℝ) * T = (1/4 : ℝ) * (1/5 : ℝ) * 120 ∧ T = 108 := by
  sorry

end NUMINAMATH_CALUDE_value_of_T_l821_82109


namespace NUMINAMATH_CALUDE_divisor_sum_840_l821_82197

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem divisor_sum_840 (i j : ℕ) (h : i > 0 ∧ j > 0) :
  sum_of_divisors (2^i * 3^j) = 840 → i + j = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_840_l821_82197


namespace NUMINAMATH_CALUDE_cone_slant_height_l821_82143

/-- The slant height of a cone given its base circumference and lateral surface sector angle -/
theorem cone_slant_height (base_circumference : ℝ) (sector_angle : ℝ) : 
  base_circumference = 2 * Real.pi → sector_angle = 120 → 3 = 
    (base_circumference * 180) / (sector_angle * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l821_82143


namespace NUMINAMATH_CALUDE_typing_time_proof_l821_82120

def typing_time (original_speed : ℕ) (speed_reduction : ℕ) (document_length : ℕ) : ℕ :=
  document_length / (original_speed - speed_reduction)

theorem typing_time_proof (original_speed : ℕ) (speed_reduction : ℕ) (document_length : ℕ) 
  (h1 : original_speed = 65)
  (h2 : speed_reduction = 20)
  (h3 : document_length = 810)
  (h4 : original_speed > speed_reduction) :
  typing_time original_speed speed_reduction document_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_proof_l821_82120


namespace NUMINAMATH_CALUDE_insurance_company_expenses_percentage_l821_82156

/-- Proves that given the conditions from the problem, the expenses in 2006 were 55.2% of the revenue in 2006 -/
theorem insurance_company_expenses_percentage (revenue2005 expenses2005 : ℝ) 
  (h1 : revenue2005 > 0)
  (h2 : expenses2005 > 0)
  (h3 : revenue2005 > expenses2005)
  (h4 : (1.25 * revenue2005 - 1.15 * expenses2005) = 1.4 * (revenue2005 - expenses2005)) :
  (1.15 * expenses2005) / (1.25 * revenue2005) = 0.552 := by
sorry

end NUMINAMATH_CALUDE_insurance_company_expenses_percentage_l821_82156


namespace NUMINAMATH_CALUDE_square_sum_eq_25_l821_82116

theorem square_sum_eq_25 (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 7) : p^2 + q^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_eq_25_l821_82116


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l821_82141

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (m + 2) * 0 - (m + 1) * 1 + m + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l821_82141


namespace NUMINAMATH_CALUDE_snowboard_discount_price_l821_82170

theorem snowboard_discount_price (original_price : ℝ) (friday_discount : ℝ) (monday_discount : ℝ) :
  original_price = 100 ∧ 
  friday_discount = 0.5 ∧ 
  monday_discount = 0.3 →
  original_price * (1 - friday_discount) * (1 - monday_discount) = 35 := by
  sorry

end NUMINAMATH_CALUDE_snowboard_discount_price_l821_82170


namespace NUMINAMATH_CALUDE_circle_k_range_l821_82172

/-- The equation of a potential circle -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + 5*k = 0

/-- The condition for the equation to represent a circle -/
def is_circle (k : ℝ) : Prop :=
  ∃ (x₀ y₀ r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y k ↔ (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The theorem stating the range of k for which the equation represents a circle -/
theorem circle_k_range :
  ∀ k : ℝ, is_circle k ↔ k < 1 :=
sorry

end NUMINAMATH_CALUDE_circle_k_range_l821_82172


namespace NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l821_82161

theorem geometric_sum_first_six_terms :
  let a₀ : ℚ := 1/2
  let r : ℚ := 1/2
  let n : ℕ := 6
  let S := a₀ * (1 - r^n) / (1 - r)
  S = 63/64 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l821_82161


namespace NUMINAMATH_CALUDE_estate_value_l821_82167

def estate_problem (E : ℝ) : Prop :=
  let younger_son := E / 5
  let elder_son := 2 * younger_son
  let husband := 3 * younger_son
  let charity := 4000
  (younger_son + elder_son = 3 * E / 5) ∧
  (elder_son = 2 * younger_son) ∧
  (husband = 3 * younger_son) ∧
  (E = younger_son + elder_son + husband + charity)

theorem estate_value : ∃ E : ℝ, estate_problem E ∧ E = 20000 := by
  sorry

end NUMINAMATH_CALUDE_estate_value_l821_82167


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l821_82180

theorem quadratic_root_problem (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*k*x + k - 1 = 0 ∧ x = 0) → 
  (∃ y : ℝ, y^2 + 2*k*y + k - 1 = 0 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l821_82180


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_even_reverse_l821_82175

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ Even (reverse_digits n)

theorem smallest_two_digit_prime_with_even_reverse : 
  satisfies_condition 23 ∧ ∀ n : ℕ, satisfies_condition n → 23 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_even_reverse_l821_82175


namespace NUMINAMATH_CALUDE_prob_intersects_inner_is_one_third_l821_82150

/-- Two concentric circles with radii 1 and 2 -/
structure ConcentricCircles where
  inner_radius : ℝ := 1
  outer_radius : ℝ := 2

/-- A chord on the outer circle -/
structure Chord where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Function to determine if a chord intersects the inner circle -/
def intersects_inner_circle (c : ConcentricCircles) (ch : Chord) : Prop :=
  sorry

/-- Function to calculate the probability of a random chord intersecting the inner circle -/
noncomputable def probability_intersects_inner (c : ConcentricCircles) : ℝ :=
  sorry

/-- Theorem stating that the probability of a random chord intersecting the inner circle is 1/3 -/
theorem prob_intersects_inner_is_one_third (c : ConcentricCircles) :
  probability_intersects_inner c = 1/3 :=
sorry

end NUMINAMATH_CALUDE_prob_intersects_inner_is_one_third_l821_82150


namespace NUMINAMATH_CALUDE_simplify_expression_l821_82159

theorem simplify_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l821_82159


namespace NUMINAMATH_CALUDE_one_third_of_recipe_l821_82135

theorem one_third_of_recipe (original_amount : ℚ) (reduced_amount : ℚ) : 
  original_amount = 5 + 3/4 → reduced_amount = (1/3) * original_amount → 
  reduced_amount = 1 + 11/12 := by
sorry

end NUMINAMATH_CALUDE_one_third_of_recipe_l821_82135


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l821_82157

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*k*x + 6 = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*k*y + 6 = 0 → y = x) ↔ 
  k = Real.sqrt 6 ∨ k = -Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l821_82157


namespace NUMINAMATH_CALUDE_problem_solution_l821_82171

theorem problem_solution (x y z w : ℕ+) 
  (h1 : x^3 = y^2) 
  (h2 : z^5 = w^4) 
  (h3 : z - x = 31) : 
  (w : ℤ) - y = -2351 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l821_82171


namespace NUMINAMATH_CALUDE_class_size_l821_82194

theorem class_size (N : ℕ) (S D B : ℕ) : 
  S = (3 * N) / 5 →  -- 3/5 of the class swims
  D = (3 * N) / 5 →  -- 3/5 of the class dances
  B = 5 →            -- 5 pupils both swim and dance
  N = S + D - B →    -- Total is sum of swimmers and dancers minus overlap
  N = 25 := by sorry

end NUMINAMATH_CALUDE_class_size_l821_82194


namespace NUMINAMATH_CALUDE_field_length_calculation_l821_82130

theorem field_length_calculation (width : ℝ) (pond_area : ℝ) (pond_percentage : ℝ) : 
  pond_area = 150 →
  pond_percentage = 0.4 →
  let length := 3 * width
  let field_area := length * width
  pond_area = pond_percentage * field_area →
  length = 15 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_field_length_calculation_l821_82130


namespace NUMINAMATH_CALUDE_race_finishing_orders_l821_82142

/-- The number of possible finishing orders for a race with n participants and no ties -/
def racePermutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of racers -/
def numRacers : ℕ := 4

theorem race_finishing_orders :
  racePermutations numRacers = 24 :=
by sorry

end NUMINAMATH_CALUDE_race_finishing_orders_l821_82142


namespace NUMINAMATH_CALUDE_trajectory_of_vertex_C_trajectory_of_vertex_C_proof_l821_82107

/-- The trajectory of vertex C in triangle ABC, where A(0, 2) and B(0, -2), 
    and the perimeter is 10, forms an ellipse. -/
theorem trajectory_of_vertex_C (C : ℝ × ℝ) : Prop :=
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := (0, -2)
  let perimeter : ℝ := 10
  let dist (P Q : ℝ × ℝ) := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  dist A B + dist B C + dist C A = perimeter →
  C.1 ≠ 0 →
  C.1^2 / 5 + C.2^2 / 9 = 1

/-- The proof of the theorem. -/
theorem trajectory_of_vertex_C_proof : ∀ C, trajectory_of_vertex_C C := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_vertex_C_trajectory_of_vertex_C_proof_l821_82107


namespace NUMINAMATH_CALUDE_largest_n_multiple_of_three_n_99998_is_solution_n_99998_is_largest_l821_82158

theorem largest_n_multiple_of_three (n : ℕ) : 
  n < 100000 → 
  (∃ k : ℤ, (n - 3)^5 - n^2 + 10*n - 30 = 3*k) → 
  n ≤ 99998 :=
sorry

theorem n_99998_is_solution : 
  ∃ k : ℤ, (99998 - 3)^5 - 99998^2 + 10*99998 - 30 = 3*k :=
sorry

theorem n_99998_is_largest : 
  ¬∃ n : ℕ, n > 99998 ∧ n < 100000 ∧ 
  (∃ k : ℤ, (n - 3)^5 - n^2 + 10*n - 30 = 3*k) :=
sorry

end NUMINAMATH_CALUDE_largest_n_multiple_of_three_n_99998_is_solution_n_99998_is_largest_l821_82158


namespace NUMINAMATH_CALUDE_course_length_proof_l821_82123

/-- Proves that the length of a course is 45 miles given the conditions of two cyclists --/
theorem course_length_proof (speed1 speed2 time : ℝ) 
  (h1 : speed1 = 14)
  (h2 : speed2 = 16)
  (h3 : time = 1.5)
  : speed1 * time + speed2 * time = 45 := by
  sorry

#check course_length_proof

end NUMINAMATH_CALUDE_course_length_proof_l821_82123


namespace NUMINAMATH_CALUDE_sqrt_21_bounds_l821_82106

theorem sqrt_21_bounds : 4 < Real.sqrt 21 ∧ Real.sqrt 21 < 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_21_bounds_l821_82106


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l821_82128

/-- Given that 3/5 of 15 bananas are worth as much as 12 oranges,
    prove that 2/3 of 9 bananas are worth as much as 8 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (3 / 5 : ℚ) * 15 * banana_value = 12 * orange_value →
  (2 / 3 : ℚ) * 9 * banana_value = 8 * orange_value :=
by sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l821_82128


namespace NUMINAMATH_CALUDE_total_spent_on_pens_l821_82190

def brand_x_price : ℝ := 4.00
def brand_y_price : ℝ := 2.20
def total_pens : ℕ := 12
def brand_x_count : ℕ := 6

theorem total_spent_on_pens : 
  brand_x_count * brand_x_price + (total_pens - brand_x_count) * brand_y_price = 37.20 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_on_pens_l821_82190


namespace NUMINAMATH_CALUDE_swallow_weight_ratio_l821_82183

/-- The weight an American swallow can carry -/
def american_weight : ℝ := 5

/-- The total number of swallows in the flock -/
def total_swallows : ℕ := 90

/-- The ratio of American swallows to European swallows -/
def american_to_european_ratio : ℕ := 2

/-- The maximum combined weight the flock can carry -/
def total_weight : ℝ := 600

/-- The weight a European swallow can carry -/
def european_weight : ℝ := 10

theorem swallow_weight_ratio : 
  european_weight / american_weight = 2 := by sorry

end NUMINAMATH_CALUDE_swallow_weight_ratio_l821_82183


namespace NUMINAMATH_CALUDE_pyramid_base_side_length_l821_82149

/-- Given a right pyramid with a square base, prove that the side length of the base is 6 meters 
    when the area of one lateral face is 120 square meters and the slant height is 40 meters. -/
theorem pyramid_base_side_length (lateral_face_area slant_height : ℝ) 
  (h1 : lateral_face_area = 120)
  (h2 : slant_height = 40) : 
  let base_side := lateral_face_area / (0.5 * slant_height)
  base_side = 6 := by sorry

end NUMINAMATH_CALUDE_pyramid_base_side_length_l821_82149


namespace NUMINAMATH_CALUDE_pen_sales_profit_l821_82129

/-- Calculates the total profit and profit percent for a pen sales scenario --/
def calculate_profit_and_percent (total_pens : ℕ) (marked_price : ℚ) 
  (discount_tier1 : ℚ) (discount_tier2 : ℚ) (discount_tier3 : ℚ)
  (pens_tier1 : ℕ) (pens_tier2 : ℕ)
  (sell_discount1 : ℚ) (sell_discount2 : ℚ)
  (pens_sold1 : ℕ) (pens_sold2 : ℕ) : ℚ × ℚ :=
  sorry

theorem pen_sales_profit :
  let total_pens : ℕ := 150
  let marked_price : ℚ := 240 / 100
  let discount_tier1 : ℚ := 5 / 100
  let discount_tier2 : ℚ := 10 / 100
  let discount_tier3 : ℚ := 15 / 100
  let pens_tier1 : ℕ := 50
  let pens_tier2 : ℕ := 50
  let sell_discount1 : ℚ := 4 / 100
  let sell_discount2 : ℚ := 2 / 100
  let pens_sold1 : ℕ := 75
  let pens_sold2 : ℕ := 75
  let (profit, percent) := calculate_profit_and_percent total_pens marked_price 
    discount_tier1 discount_tier2 discount_tier3
    pens_tier1 pens_tier2
    sell_discount1 sell_discount2
    pens_sold1 pens_sold2
  profit = 2520 / 100 ∧ abs (percent - 778 / 10000) < 1 / 10000 :=
by sorry

end NUMINAMATH_CALUDE_pen_sales_profit_l821_82129


namespace NUMINAMATH_CALUDE_player_one_winning_strategy_l821_82195

-- Define the chessboard
def Chessboard : Type := Fin 8 × Fin 8

-- Define the distance between two points on the chessboard
def distance (p1 p2 : Chessboard) : ℝ := sorry

-- Define a valid move
def validMove (prev curr next : Chessboard) : Prop :=
  distance curr next > distance prev curr

-- Define the game state
structure GameState :=
  (position : Chessboard)
  (lastMove : Option Chessboard)
  (playerTurn : Bool)  -- true for Player One, false for Player Two

-- Define the winning condition for Player One
def playerOneWins (game : GameState) : Prop :=
  ∀ move : Chessboard, ¬validMove (Option.getD game.lastMove game.position) game.position move

-- Theorem: Player One has a winning strategy
theorem player_one_winning_strategy :
  ∃ (strategy : GameState → Chessboard),
    ∀ (game : GameState),
      game.playerTurn → 
      validMove (Option.getD game.lastMove game.position) game.position (strategy game) ∧
      playerOneWins {
        position := strategy game,
        lastMove := some game.position,
        playerTurn := false
      } := sorry

end NUMINAMATH_CALUDE_player_one_winning_strategy_l821_82195


namespace NUMINAMATH_CALUDE_sandy_initial_money_l821_82138

/-- Given that Sandy spent $6 on a pie and has $57 left, prove that she initially had $63. -/
theorem sandy_initial_money :
  ∀ (initial_money spent_on_pie money_left : ℕ),
    spent_on_pie = 6 →
    money_left = 57 →
    initial_money = spent_on_pie + money_left →
    initial_money = 63 := by
  sorry

end NUMINAMATH_CALUDE_sandy_initial_money_l821_82138


namespace NUMINAMATH_CALUDE_part_one_part_two_l821_82114

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y - 1
def B (x y : ℝ) : ℝ := x^2 - x * y

-- Part 1
theorem part_one : ∀ x y : ℝ, (x + 1)^2 + |y - 2| = 0 → A x y - 2 * B x y = -7 := by
  sorry

-- Part 2
theorem part_two : (∃ c : ℝ, ∀ x y : ℝ, A x y - 2 * B x y = c) → 
  ∃ x : ℝ, x^2 - 2*x - 1 = -1/25 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l821_82114


namespace NUMINAMATH_CALUDE_same_terminal_side_l821_82179

theorem same_terminal_side (π : ℝ) : ∃ (k : ℤ), (7 / 6 * π) - (-5 / 6 * π) = k * (2 * π) := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l821_82179


namespace NUMINAMATH_CALUDE_average_speed_tony_l821_82101

def rollercoaster_speeds : List ℝ := [50, 62, 73, 70, 40]

theorem average_speed_tony (speeds := rollercoaster_speeds) : 
  (speeds.sum / speeds.length : ℝ) = 59 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_tony_l821_82101


namespace NUMINAMATH_CALUDE_specific_pairing_probability_l821_82198

/-- Represents a classroom with male and female students -/
structure Classroom where
  female_count : ℕ
  male_count : ℕ

/-- Represents a pairing of students -/
structure Pairing where
  classroom : Classroom
  is_opposite_gender : Bool

/-- Calculates the probability of a specific pairing -/
def probability_of_specific_pairing (c : Classroom) (p : Pairing) : ℚ :=
  1 / c.male_count

/-- Theorem: The probability of a specific female-male pairing in a classroom
    with 20 female students and 18 male students is 1/18 -/
theorem specific_pairing_probability :
  let c : Classroom := { female_count := 20, male_count := 18 }
  let p : Pairing := { classroom := c, is_opposite_gender := true }
  probability_of_specific_pairing c p = 1 / 18 := by
    sorry

end NUMINAMATH_CALUDE_specific_pairing_probability_l821_82198


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_pi_6_l821_82165

theorem sin_2alpha_minus_pi_6 (α : Real) :
  (∃ P : Real × Real, P.1 = -3/5 ∧ P.2 = 4/5 ∧ P.1^2 + P.2^2 = 1 ∧
    P.1 = Real.cos (α + π/6) ∧ P.2 = Real.sin (α + π/6)) →
  Real.sin (2*α - π/6) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_pi_6_l821_82165


namespace NUMINAMATH_CALUDE_light_flashes_l821_82136

/-- A light flashes every 15 seconds. This theorem proves that it will flash 180 times in ¾ of an hour. -/
theorem light_flashes (flash_interval : ℕ) (hour_fraction : ℚ) (flashes : ℕ) : 
  flash_interval = 15 → hour_fraction = 3/4 → flashes = 180 → 
  (hour_fraction * 3600) / flash_interval = flashes := by
  sorry

end NUMINAMATH_CALUDE_light_flashes_l821_82136


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l821_82199

/-- The number of possible letters in the license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in the license plate. -/
def num_digits : ℕ := 10

/-- The length of the letter sequence in the license plate. -/
def letter_seq_length : ℕ := 4

/-- The length of the digit sequence in the license plate. -/
def digit_seq_length : ℕ := 4

/-- The probability of a license plate containing at least one palindrome. -/
def palindrome_probability : ℚ := 775 / 67600

theorem license_plate_palindrome_probability :
  let letter_palindrome_prob := 1 / (num_letters ^ 2 : ℚ)
  let digit_palindrome_prob := 1 / (num_digits ^ 2 : ℚ)
  let total_prob := letter_palindrome_prob + digit_palindrome_prob - 
                    (letter_palindrome_prob * digit_palindrome_prob)
  total_prob = palindrome_probability := by sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l821_82199


namespace NUMINAMATH_CALUDE_trig_product_equals_one_l821_82131

theorem trig_product_equals_one :
  let x : Real := 30 * π / 180  -- 30 degrees in radians
  let y : Real := 60 * π / 180  -- 60 degrees in radians
  (1 - 1 / Real.cos x) * (1 + 1 / Real.sin y) * (1 - 1 / Real.sin x) * (1 + 1 / Real.cos y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_l821_82131


namespace NUMINAMATH_CALUDE_food_percentage_is_twenty_percent_l821_82166

/-- Represents the percentage of total amount spent on each category and their respective tax rates -/
structure ShoppingExpenses where
  clothing_percent : Real
  other_percent : Real
  clothing_tax : Real
  other_tax : Real
  total_tax : Real

/-- Calculates the percentage spent on food given the shopping expenses -/
def food_percent (e : ShoppingExpenses) : Real :=
  1 - e.clothing_percent - e.other_percent

/-- Calculates the total tax rate based on the expenses and tax rates -/
def total_tax_rate (e : ShoppingExpenses) : Real :=
  e.clothing_percent * e.clothing_tax + e.other_percent * e.other_tax

/-- Theorem stating that given the shopping conditions, the percentage spent on food is 20% -/
theorem food_percentage_is_twenty_percent (e : ShoppingExpenses) 
  (h1 : e.clothing_percent = 0.5)
  (h2 : e.other_percent = 0.3)
  (h3 : e.clothing_tax = 0.04)
  (h4 : e.other_tax = 0.1)
  (h5 : e.total_tax = 0.05)
  (h6 : total_tax_rate e = e.total_tax) :
  food_percent e = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_food_percentage_is_twenty_percent_l821_82166


namespace NUMINAMATH_CALUDE_baker_remaining_pastries_l821_82177

-- Define the initial number of pastries and the number of pastries sold
def initial_pastries : ℕ := 56
def sold_pastries : ℕ := 29

-- Define the function to calculate remaining pastries
def remaining_pastries : ℕ := initial_pastries - sold_pastries

-- Theorem statement
theorem baker_remaining_pastries : remaining_pastries = 27 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_pastries_l821_82177


namespace NUMINAMATH_CALUDE_desired_depth_is_50_l821_82164

/-- Represents the digging scenario with initial and new conditions -/
structure DiggingScenario where
  initial_men : ℕ
  initial_hours : ℕ
  initial_depth : ℕ
  new_hours : ℕ
  extra_men : ℕ

/-- Calculates the desired depth given a digging scenario -/
def desired_depth (scenario : DiggingScenario) : ℕ :=
  let initial_work := scenario.initial_men * scenario.initial_hours
  let new_men := scenario.initial_men + scenario.extra_men
  let new_work := new_men * scenario.new_hours
  (new_work * scenario.initial_depth) / initial_work

/-- The main theorem stating that the desired depth is 50 meters -/
theorem desired_depth_is_50 (scenario : DiggingScenario)
  (h1 : scenario.initial_men = 18)
  (h2 : scenario.initial_hours = 8)
  (h3 : scenario.initial_depth = 30)
  (h4 : scenario.new_hours = 6)
  (h5 : scenario.extra_men = 22) :
  desired_depth scenario = 50 := by
  sorry

end NUMINAMATH_CALUDE_desired_depth_is_50_l821_82164


namespace NUMINAMATH_CALUDE_equation_solution_l821_82176

theorem equation_solution : 
  ∃ x : ℝ, |Real.sqrt (x^2 + 8*x + 20) + Real.sqrt (x^2 - 2*x + 2)| = Real.sqrt 26 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l821_82176


namespace NUMINAMATH_CALUDE_power_minus_one_rational_l821_82186

/-- A complex number with rational real and imaginary parts and unit modulus -/
structure UnitRationalComplex where
  re : ℚ
  im : ℚ
  unit_modulus : re^2 + im^2 = 1

/-- The result of z^(2n) - 1 is rational for any integer n -/
theorem power_minus_one_rational (z : UnitRationalComplex) (n : ℤ) :
  ∃ (q : ℚ), (z.re + z.im * Complex.I)^(2*n) - 1 = q := by
  sorry

end NUMINAMATH_CALUDE_power_minus_one_rational_l821_82186


namespace NUMINAMATH_CALUDE_work_completion_time_l821_82155

/-- The number of days it takes 'a' to complete the work -/
def days_a : ℕ := 27

/-- The number of days it takes 'b' to complete the work -/
def days_b : ℕ := 2 * days_a

theorem work_completion_time : days_a = 27 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l821_82155


namespace NUMINAMATH_CALUDE_zacks_countries_l821_82145

theorem zacks_countries (alex george joseph patrick zack : ℕ) : 
  alex = 24 →
  george = alex / 4 →
  joseph = george / 2 →
  patrick = joseph * 3 →
  zack = patrick * 2 →
  zack = 18 :=
by sorry

end NUMINAMATH_CALUDE_zacks_countries_l821_82145


namespace NUMINAMATH_CALUDE_pyramid_volume_l821_82154

theorem pyramid_volume (base_length : ℝ) (base_width : ℝ) (height : ℝ) :
  base_length = 1/3 →
  base_width = 1/4 →
  height = 1 →
  (1/3) * (base_length * base_width) * height = 1/36 := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_l821_82154


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l821_82103

theorem consecutive_integers_product_812_sum_57 :
  ∀ n : ℕ, n > 0 ∧ n * (n + 1) = 812 → n + (n + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l821_82103


namespace NUMINAMATH_CALUDE_prove_length_l821_82148

-- Define the points
variable (A O B A1 B1 : ℝ)

-- Define the conditions
axiom collinear : ∃ (t : ℝ), O = t • A + (1 - t) • B
axiom symmetric_A : A1 - O = O - A
axiom symmetric_B : B1 - O = O - B
axiom given_length : abs (A - B1) = 2

-- State the theorem
theorem prove_length : abs (A1 - B) = 2 := by sorry

end NUMINAMATH_CALUDE_prove_length_l821_82148


namespace NUMINAMATH_CALUDE_sector_area_special_case_l821_82117

/-- The area of a circular sector with central angle 2π/3 radians and radius 2 is 4π/3. -/
theorem sector_area_special_case :
  let central_angle : Real := (2 * Real.pi) / 3
  let radius : Real := 2
  let sector_area : Real := (1 / 2) * radius^2 * central_angle
  sector_area = (4 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_special_case_l821_82117


namespace NUMINAMATH_CALUDE_painting_price_increase_percentage_l821_82137

/-- Proves that the percentage increase in the cost of each painting is 20% --/
theorem painting_price_increase_percentage :
  let original_jewelry_price : ℚ := 30
  let original_painting_price : ℚ := 100
  let jewelry_price_increase : ℚ := 10
  let jewelry_quantity : ℕ := 2
  let painting_quantity : ℕ := 5
  let total_cost : ℚ := 680
  let new_jewelry_price : ℚ := original_jewelry_price + jewelry_price_increase
  let painting_price_increase_percentage : ℚ := 20

  (jewelry_quantity : ℚ) * new_jewelry_price + 
  (painting_quantity : ℚ) * original_painting_price * (1 + painting_price_increase_percentage / 100) = 
  total_cost :=
by sorry

end NUMINAMATH_CALUDE_painting_price_increase_percentage_l821_82137


namespace NUMINAMATH_CALUDE_expression_evaluation_l821_82173

theorem expression_evaluation (x z : ℝ) (h : x = Real.sqrt z) :
  (x - 1 / x) * (Real.sqrt z + 1 / Real.sqrt z) = z - 1 / z := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l821_82173


namespace NUMINAMATH_CALUDE_our_system_is_linear_l821_82104

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  toFun : ℝ → ℝ → Prop := λ x y => a * x + b * y = c

/-- A system of two equations -/
structure SystemOfTwoEquations where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The specific system we want to prove is linear -/
def ourSystem : SystemOfTwoEquations where
  eq1 := { a := 1, b := 1, c := 2 }
  eq2 := { a := 1, b := -1, c := 4 }

/-- A predicate to check if a system is linear -/
def isLinearSystem (s : SystemOfTwoEquations) : Prop :=
  s.eq1.a ≠ 0 ∨ s.eq1.b ≠ 0 ∧
  s.eq2.a ≠ 0 ∨ s.eq2.b ≠ 0

theorem our_system_is_linear : isLinearSystem ourSystem := by
  sorry

end NUMINAMATH_CALUDE_our_system_is_linear_l821_82104


namespace NUMINAMATH_CALUDE_expression_simplification_l821_82127

theorem expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let expr := ((a^(3/4) - b^(3/4)) * (a^(3/4) + b^(3/4)) / (a^(1/2) - b^(1/2)) - Real.sqrt (a * b)) *
               (2 * Real.sqrt 2.5 * (a + b)⁻¹) / (Real.sqrt 1000)^(1/3)
  expr = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l821_82127


namespace NUMINAMATH_CALUDE_inequality_proof_l821_82105

theorem inequality_proof (x : ℝ) (h : x ≥ 0) :
  1 + x^2006 ≥ (2*x)^2005 / (1+x)^2004 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l821_82105


namespace NUMINAMATH_CALUDE_largest_divisor_l821_82118

theorem largest_divisor (A B : ℕ) (h1 : 13 = 4 * A + B) (h2 : B < A) : A ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_l821_82118


namespace NUMINAMATH_CALUDE_tangent_slope_at_pi_over_4_l821_82184

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sin x + Real.cos x) - 1/2

theorem tangent_slope_at_pi_over_4 :
  let df := deriv f
  df (π/4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_pi_over_4_l821_82184


namespace NUMINAMATH_CALUDE_refusing_managers_pair_l821_82187

/-- The number of managers to choose from -/
def total_managers : ℕ := 8

/-- The number of managers needed for the meeting -/
def meeting_size : ℕ := 4

/-- The number of ways to select managers for the meeting -/
def selection_ways : ℕ := 55

/-- Calculates the number of combinations -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The theorem to prove -/
theorem refusing_managers_pair : 
  ∃! (refusing_pairs : ℕ), 
    combinations total_managers meeting_size - 
    refusing_pairs * combinations (total_managers - 2) (meeting_size - 2) = 
    selection_ways :=
sorry

end NUMINAMATH_CALUDE_refusing_managers_pair_l821_82187


namespace NUMINAMATH_CALUDE_complex_abs_value_l821_82160

theorem complex_abs_value (z : ℂ) : z = (1 - Complex.I)^2 / (1 + Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_value_l821_82160


namespace NUMINAMATH_CALUDE_original_number_of_people_l821_82140

theorem original_number_of_people (x : ℕ) : 
  (x / 2 : ℚ) - (x / 2 : ℚ) / 3 = 12 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_original_number_of_people_l821_82140


namespace NUMINAMATH_CALUDE_simplify_expression_l821_82119

theorem simplify_expression (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  ((x + a)^4) / ((a - b)*(a - c)) + ((x + b)^4) / ((b - a)*(b - c)) + ((x + c)^4) / ((c - a)*(c - b)) = a + b + c + 4*x :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l821_82119


namespace NUMINAMATH_CALUDE_max_value_sum_fractions_l821_82185

theorem max_value_sum_fractions (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_eq_one : a + b + c + d = 1) :
  (a * b) / (a + b) + (a * c) / (a + c) + (a * d) / (a + d) +
  (b * c) / (b + c) + (b * d) / (b + d) + (c * d) / (c + d) ≤ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_fractions_l821_82185


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l821_82163

theorem triangle_side_and_area (a b c : ℝ) (B : ℝ) (h_a : a = 8) (h_b : b = 7) (h_B : B = Real.pi / 3) :
  c^2 - 4*c - 25 = 0 ∧ 
  ∃ S : ℝ, S = (1/2) * a * c * Real.sin B :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_and_area_l821_82163


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l821_82115

-- Define the conditions p and q
def p (x : ℝ) : Prop := x^2 - 4*x + 3 > 0
def q (x : ℝ) : Prop := x^2 < 1

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l821_82115


namespace NUMINAMATH_CALUDE_binomial_20_10_l821_82122

theorem binomial_20_10 (h1 : Nat.choose 18 8 = 31824) 
                        (h2 : Nat.choose 18 9 = 48620) 
                        (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 172822 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_10_l821_82122


namespace NUMINAMATH_CALUDE_firefighter_remaining_money_is_2340_l821_82191

/-- Calculates the remaining money for a firefighter after monthly expenses --/
def firefighter_remaining_money (hourly_rate : ℚ) (weekly_hours : ℚ) (food_expense : ℚ) (tax_expense : ℚ) : ℚ :=
  let weekly_earnings := hourly_rate * weekly_hours
  let monthly_earnings := weekly_earnings * 4
  let rent_expense := monthly_earnings / 3
  let total_expenses := rent_expense + food_expense + tax_expense
  monthly_earnings - total_expenses

/-- Theorem stating that the firefighter's remaining money is $2340 --/
theorem firefighter_remaining_money_is_2340 :
  firefighter_remaining_money 30 48 500 1000 = 2340 := by
  sorry

#eval firefighter_remaining_money 30 48 500 1000

end NUMINAMATH_CALUDE_firefighter_remaining_money_is_2340_l821_82191


namespace NUMINAMATH_CALUDE_guanaco_numbers_l821_82102

def is_guanaco (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    a ≠ 0 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    n = a * 1000 + b * 100 + c * 10 + d ∧
    (a * 10 + b) * (c * 10 + d) ∣ n

theorem guanaco_numbers :
  ∀ n : ℕ, is_guanaco n ↔ (n = 1352 ∨ n = 1734) :=
by sorry

end NUMINAMATH_CALUDE_guanaco_numbers_l821_82102


namespace NUMINAMATH_CALUDE_eric_sibling_product_l821_82188

/-- Represents a family with a given number of sisters and brothers -/
structure Family where
  sisters : ℕ
  brothers : ℕ

/-- Calculates the number of sisters and brothers for a sibling in the family -/
def siblingCounts (f : Family) : ℕ × ℕ :=
  (f.sisters + 1, f.brothers)

theorem eric_sibling_product (emmas_family : Family)
    (h1 : emmas_family.sisters = 4)
    (h2 : emmas_family.brothers = 6) :
    let (S, B) := siblingCounts emmas_family
    S * B = 30 := by
  sorry

end NUMINAMATH_CALUDE_eric_sibling_product_l821_82188


namespace NUMINAMATH_CALUDE_inequality_proof_l821_82181

theorem inequality_proof (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l821_82181


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_31_mod_26_l821_82126

theorem largest_five_digit_congruent_to_31_mod_26 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n ≡ 31 [MOD 26] → n ≤ 99975 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_31_mod_26_l821_82126


namespace NUMINAMATH_CALUDE_simplify_expression_l821_82174

theorem simplify_expression (a b : ℝ) :
  3 * a * (3 * a^3 + 2 * a^2) - 2 * a^2 * (b^2 + 1) = 9 * a^4 + 6 * a^3 - 2 * a^2 * b^2 - 2 * a^2 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l821_82174


namespace NUMINAMATH_CALUDE_negation_of_existence_irrational_square_l821_82147

theorem negation_of_existence_irrational_square :
  (¬ ∃ x : ℝ, Irrational (x^2)) ↔ (∀ x : ℝ, ¬ Irrational (x^2)) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_irrational_square_l821_82147


namespace NUMINAMATH_CALUDE_bike_purchase_weeks_l821_82152

def bike_cost : ℕ := 600
def gift_money : ℕ := 150
def weekly_earnings : ℕ := 20

def weeks_needed : ℕ := 23

theorem bike_purchase_weeks : 
  ∀ (w : ℕ), w ≥ weeks_needed ↔ gift_money + w * weekly_earnings ≥ bike_cost :=
by sorry

end NUMINAMATH_CALUDE_bike_purchase_weeks_l821_82152


namespace NUMINAMATH_CALUDE_expression_simplification_l821_82133

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 3) * (x - 4) * (x - 2)) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l821_82133


namespace NUMINAMATH_CALUDE_isabella_currency_exchange_l821_82113

/-- Represents the exchange of US dollars to Canadian dollars and subsequent spending -/
def exchange_and_spend (d : ℕ) : Prop :=
  (8 * d) / 5 - 75 = d

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The main theorem representing the problem -/
theorem isabella_currency_exchange :
  ∃ d : ℕ, exchange_and_spend d ∧ d = 125 ∧ sum_of_digits d = 8 :=
sorry

end NUMINAMATH_CALUDE_isabella_currency_exchange_l821_82113


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l821_82192

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Part 1
theorem solution_set_part1 (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 - a*x + 1 > 0 ↔ x < 1/3 ∨ x > 1/2) :=
sorry

-- Part 2
theorem range_of_a_part2 (a : ℝ) :
  (∀ x, x ∈ Set.Ioc (-1) 0 → f a (3-a) x ≥ 0) →
  a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l821_82192


namespace NUMINAMATH_CALUDE_pastry_sets_problem_l821_82112

theorem pastry_sets_problem (N : ℕ) 
  (h1 : ∃ (x y : ℕ), x + y = N ∧ 3*x + 5*y = 25)
  (h2 : ∃ (a b : ℕ), a + b = N ∧ 3*a + 5*b = 35) : 
  N = 7 := by
sorry

end NUMINAMATH_CALUDE_pastry_sets_problem_l821_82112


namespace NUMINAMATH_CALUDE_motorcycles_in_anytown_l821_82125

/-- Represents the number of vehicles of each type in Anytown -/
structure VehicleCounts where
  trucks : ℕ
  sedans : ℕ
  motorcycles : ℕ

/-- The ratio of vehicles in Anytown -/
def vehicle_ratio : VehicleCounts := ⟨3, 7, 2⟩

/-- The actual number of sedans in Anytown -/
def actual_sedans : ℕ := 9100

/-- Theorem stating the number of motorcycles in Anytown -/
theorem motorcycles_in_anytown : 
  ∃ (vc : VehicleCounts), 
    vc.sedans = actual_sedans ∧ 
    vc.trucks * vehicle_ratio.sedans = vc.sedans * vehicle_ratio.trucks ∧
    vc.sedans * vehicle_ratio.motorcycles = vc.motorcycles * vehicle_ratio.sedans ∧
    vc.motorcycles = 2600 := by
  sorry

end NUMINAMATH_CALUDE_motorcycles_in_anytown_l821_82125


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_50_l821_82146

theorem factorization_of_2x_squared_minus_50 (x : ℝ) : 2 * x^2 - 50 = 2 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_50_l821_82146


namespace NUMINAMATH_CALUDE_solution_set_l821_82100

def f (x : ℝ) := abs x + x^2 + 2

theorem solution_set (x : ℝ) :
  f (2*x - 1) > f (3 - x) ↔ x < -2 ∨ x > 4/3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l821_82100


namespace NUMINAMATH_CALUDE_range_of_A_l821_82111

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_A : ∀ a : ℝ, a ∈ A ↔ a ∈ Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_range_of_A_l821_82111


namespace NUMINAMATH_CALUDE_courses_selection_theorem_l821_82168

/-- The number of available courses -/
def n : ℕ := 4

/-- The number of courses each person selects -/
def k : ℕ := 2

/-- The number of ways to select courses with at least one different course -/
def select_courses : ℕ := 30

/-- Theorem stating that the number of ways to select courses with at least one different course is 30 -/
theorem courses_selection_theorem : select_courses = 30 := by
  sorry

end NUMINAMATH_CALUDE_courses_selection_theorem_l821_82168


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l821_82182

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
structure CircleConfiguration where
  O₁ : Circle
  O₂ : Circle
  O : Circle
  h₁ : O₁.radius ≠ O₂.radius
  h₂ : O₁.center ≠ O₂.center
  h₃ : ∀ p : ℝ × ℝ, (dist p O₁.center ≠ O₁.radius) ∨ (dist p O₂.center ≠ O₂.radius)
  h₄ : dist O.center O₁.center = O.radius + O₁.radius ∨ dist O.center O₁.center = abs (O.radius - O₁.radius)
  h₅ : dist O.center O₂.center = O.radius + O₂.radius ∨ dist O.center O₂.center = abs (O.radius - O₂.radius)

-- Define the trajectory types
inductive TrajectoryType
  | Hyperbola
  | Ellipse

-- State the theorem
theorem trajectory_of_moving_circle (config : CircleConfiguration) :
  ∃ t : TrajectoryType, t = TrajectoryType.Hyperbola ∨ t = TrajectoryType.Ellipse :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l821_82182


namespace NUMINAMATH_CALUDE_number_of_divisors_l821_82132

theorem number_of_divisors : ∃ n : ℕ, 
  (∀ d : ℕ, d ∣ (2008^3 + (3 * 2008 * 2009) + 1)^2 ↔ d ∈ Finset.range (n + 1) ∧ d ≠ 0) ∧
  n = 91 :=
sorry

end NUMINAMATH_CALUDE_number_of_divisors_l821_82132


namespace NUMINAMATH_CALUDE_factor_expression_l821_82124

theorem factor_expression (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l821_82124


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_16_l821_82144

theorem arithmetic_sqrt_16 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_16_l821_82144


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l821_82169

theorem perpendicular_tangents_intersection (a b : ℝ) :
  (2*a) * (2*b) = -1 →
  let A : ℝ × ℝ := (a, a^2)
  let B : ℝ × ℝ := (b, b^2)
  let P : ℝ × ℝ := ((a + b)/2, a*b)
  (P.2 = -1/4) := by sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l821_82169


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l821_82108

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l821_82108


namespace NUMINAMATH_CALUDE_sock_order_ratio_l821_82134

theorem sock_order_ratio (red_socks green_socks : ℕ) (price_green : ℝ) :
  red_socks = 5 →
  (red_socks * (3 * price_green) + green_socks * price_green) * 1.8 =
    green_socks * (3 * price_green) + red_socks * price_green →
  green_socks = 18 :=
by sorry

end NUMINAMATH_CALUDE_sock_order_ratio_l821_82134


namespace NUMINAMATH_CALUDE_prob_different_colors_specific_l821_82189

/-- Represents a box containing balls of different colors -/
structure Box where
  red : ℕ
  black : ℕ
  white : ℕ
  yellow : ℕ

/-- Calculates the total number of balls in a box -/
def Box.total (b : Box) : ℕ := b.red + b.black + b.white + b.yellow

/-- The probability of drawing two balls of different colors from two boxes -/
def prob_different_colors (boxA boxB : Box) : ℚ :=
  1 - (boxA.black * boxB.black + boxA.white * boxB.white : ℚ) / 
      ((boxA.total * boxB.total) : ℚ)

/-- The main theorem stating the probability of drawing different colored balls -/
theorem prob_different_colors_specific : 
  let boxA : Box := { red := 3, black := 3, white := 3, yellow := 0 }
  let boxB : Box := { red := 0, black := 2, white := 2, yellow := 2 }
  prob_different_colors boxA boxB = 7/9 := by
  sorry


end NUMINAMATH_CALUDE_prob_different_colors_specific_l821_82189


namespace NUMINAMATH_CALUDE_special_polygon_is_heptagon_l821_82110

/-- A polygon where all diagonals passing through one vertex divide it into 5 triangles -/
structure SpecialPolygon where
  /-- The number of triangles formed by diagonals passing through one vertex -/
  num_triangles : ℕ
  /-- The number of triangles is exactly 5 -/
  h_triangles : num_triangles = 5

/-- The number of sides in a SpecialPolygon -/
def SpecialPolygon.num_sides (p : SpecialPolygon) : ℕ :=
  p.num_triangles + 2

theorem special_polygon_is_heptagon (p : SpecialPolygon) : p.num_sides = 7 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_is_heptagon_l821_82110


namespace NUMINAMATH_CALUDE_digit_sum_equation_l821_82196

theorem digit_sum_equation (a : ℕ) : a * 1000 + a * 998 + a * 999 = 22997 → a = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_equation_l821_82196


namespace NUMINAMATH_CALUDE_min_value_of_abs_sum_l821_82178

theorem min_value_of_abs_sum (x : ℝ) : 
  |x - 4| + |x - 6| ≥ 2 ∧ ∃ y : ℝ, |y - 4| + |y - 6| = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_abs_sum_l821_82178


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l821_82139

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, 10) and (-4, -10) is 2. -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := 10
  let x₂ : ℝ := -4
  let y₂ : ℝ := -10
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 2 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l821_82139


namespace NUMINAMATH_CALUDE_line_plane_relationships_l821_82193

-- Define the basic structures
variable (α : Plane) (l m : Line)

-- Define the relationships
def not_contained_in (l : Line) (α : Plane) : Prop := sorry
def contained_in (m : Line) (α : Plane) : Prop := sorry
def perpendicular (l : Line) (α : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (α : Plane) : Prop := sorry
def perpendicular_lines (l m : Line) : Prop := sorry
def parallel_lines (l m : Line) : Prop := sorry

-- State the theorem
theorem line_plane_relationships :
  not_contained_in l α →
  contained_in m α →
  ((perpendicular l α → perpendicular_lines l m) ∧
   (parallel_lines l m → parallel_line_plane l α)) :=
by sorry

end NUMINAMATH_CALUDE_line_plane_relationships_l821_82193


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l821_82121

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  size : ℕ
  captainAge : ℕ
  wicketKeeperAge : ℕ
  averageAge : ℚ
  remainingAverageAge : ℚ

/-- The age difference between the wicket keeper and the captain -/
def ageDifference (team : CricketTeam) : ℕ :=
  team.wicketKeeperAge - team.captainAge

/-- Theorem stating the properties of the cricket team and the age difference -/
theorem cricket_team_age_difference (team : CricketTeam) 
  (h1 : team.size = 11)
  (h2 : team.captainAge = 26)
  (h3 : team.wicketKeeperAge > team.captainAge)
  (h4 : team.averageAge = 24)
  (h5 : team.remainingAverageAge = team.averageAge - 1)
  : ageDifference team = 5 := by
  sorry


end NUMINAMATH_CALUDE_cricket_team_age_difference_l821_82121


namespace NUMINAMATH_CALUDE_college_students_count_l821_82162

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 210) :
  boys + girls = 546 := by
sorry

end NUMINAMATH_CALUDE_college_students_count_l821_82162


namespace NUMINAMATH_CALUDE_smallest_cookie_boxes_l821_82153

theorem smallest_cookie_boxes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (15 * m - 3) % 11 = 0 → n ≤ m) ∧ 
  (15 * n - 3) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_boxes_l821_82153


namespace NUMINAMATH_CALUDE_vector_dot_product_problem_l821_82151

-- Define the type for 2D vectors
def Vector2D := ℝ × ℝ

-- Define vector addition
def add (v w : Vector2D) : Vector2D :=
  (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def smul (r : ℝ) (v : Vector2D) : Vector2D :=
  (r * v.1, r * v.2)

-- Define dot product
def dot (v w : Vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_problem (a b : Vector2D) 
  (h1 : add (smul 2 a) b = (1, 6)) 
  (h2 : add a (smul 2 b) = (-4, 9)) : 
  dot a b = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_problem_l821_82151
