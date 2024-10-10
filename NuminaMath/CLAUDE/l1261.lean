import Mathlib

namespace right_angled_triangle_set_l1261_126107

theorem right_angled_triangle_set : ∃! (a b c : ℝ), 
  ((a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3) ∨
   (a = 2 ∧ b = 3 ∧ c = 4) ∨
   (a = 4 ∧ b = 6 ∧ c = 8) ∨
   (a = 5 ∧ b = 12 ∧ c = 15)) ∧
  a^2 + b^2 = c^2 := by
sorry

end right_angled_triangle_set_l1261_126107


namespace quadratic_always_nonnegative_l1261_126130

theorem quadratic_always_nonnegative (m : ℝ) : 
  (∀ x : ℝ, x^2 - (m - 1) * x + 1 ≥ 0) ↔ m ∈ Set.Icc (-1 : ℝ) 3 := by
  sorry

end quadratic_always_nonnegative_l1261_126130


namespace quadratic_factorization_sum_l1261_126188

theorem quadratic_factorization_sum (d e f : ℝ) : 
  (∀ x, (x + d) * (x + e) = x^2 + 11*x + 24) →
  (∀ x, (x + e) * (x - f) = x^2 + 9*x - 36) →
  d + e + f = 14 := by sorry

end quadratic_factorization_sum_l1261_126188


namespace total_sheets_prepared_l1261_126121

/-- Given the number of sheets used for a crane and the number of sheets left,
    prove that the total number of sheets prepared at the beginning
    is equal to the sum of sheets used and sheets left. -/
theorem total_sheets_prepared
  (sheets_used : ℕ) (sheets_left : ℕ)
  (h1 : sheets_used = 12)
  (h2 : sheets_left = 9) :
  sheets_used + sheets_left = 21 := by
sorry

end total_sheets_prepared_l1261_126121


namespace jason_pokemon_cards_l1261_126141

theorem jason_pokemon_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 676 → cards_bought = 224 → remaining_cards = initial_cards - cards_bought → 
  remaining_cards = 452 := by
  sorry

end jason_pokemon_cards_l1261_126141


namespace area_scientific_notation_l1261_126169

-- Define the area in square kilometers
def area : ℝ := 6.4e6

-- Theorem to prove the scientific notation representation
theorem area_scientific_notation : area = 6.4 * (10 : ℝ)^6 := by
  sorry

end area_scientific_notation_l1261_126169


namespace recurrence_sequence_a1_l1261_126133

/-- A sequence of positive real numbers satisfying the given recurrence relation. -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  a 0 = 1 ∧ (∀ n : ℕ, 0 < a n) ∧ (∀ n : ℕ, a n = a (n + 1) + a (n + 2))

/-- The theorem stating that a₁ equals (√5 - 1) / 2 for the given recurrence sequence. -/
theorem recurrence_sequence_a1 (a : ℕ → ℝ) (h : RecurrenceSequence a) :
    a 1 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end recurrence_sequence_a1_l1261_126133


namespace simplify_expression_l1261_126101

theorem simplify_expression (a b : ℝ) : a - 4*(2*a - b) - 2*(a + 2*b) = -9*a := by
  sorry

end simplify_expression_l1261_126101


namespace system_solution_l1261_126193

theorem system_solution (a b c x y z : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hsum : x + y + z ≠ 0) (hyz : y + z ≠ 0) (hzx : z + x ≠ 0) (hxy : x + y ≠ 0)
  (eq1 : 1/x + 1/(x+y) = 1/a)
  (eq2 : 1/y + 1/(z+x) = 1/b)
  (eq3 : 1/z + 1/(x+y) = 1/c) :
  x = (2*(a*b + a*c + b*c) - (a^2 + b^2 + c^2)) / (2*(-a + b + c)) ∧
  y = (2*(a*b + a*c + b*c) - (a^2 + b^2 + c^2)) / (2*(a - b + c)) ∧
  z = (2*(a*b + a*c + b*c) - (a^2 + b^2 + c^2)) / (2*(a + b - c)) :=
by sorry

end system_solution_l1261_126193


namespace total_money_l1261_126113

def money_problem (john peter quincy andrew : ℝ) : Prop :=
  peter = 2 * john ∧
  quincy = peter + 20 ∧
  andrew = 1.15 * quincy ∧
  john + peter + quincy + andrew = 1211

theorem total_money :
  ∃ john peter quincy andrew : ℝ,
    money_problem john peter quincy andrew ∧
    john + peter + quincy + andrew = 1072.01 := by sorry

end total_money_l1261_126113


namespace alex_age_theorem_l1261_126179

theorem alex_age_theorem :
  ∃! x : ℕ, x > 0 ∧ x ≤ 100 ∧ 
  ∃ y : ℕ, x - 2 = y^2 ∧
  ∃ z : ℕ, x + 2 = z^3 :=
by
  sorry

end alex_age_theorem_l1261_126179


namespace evaluate_expression_l1261_126120

theorem evaluate_expression : (7 - 3)^2 + (7^2 - 3^2) = 56 := by
  sorry

end evaluate_expression_l1261_126120


namespace regression_lines_common_point_l1261_126173

-- Define the type for a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the type for a line in 2D space
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a function to check if a point lies on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Define the theorem
theorem regression_lines_common_point
  (s t : ℝ)
  (l₁ l₂ : Line)
  (h₁ : pointOnLine ⟨s, t⟩ l₁)
  (h₂ : pointOnLine ⟨s, t⟩ l₂) :
  ∃ (p : Point), pointOnLine p l₁ ∧ pointOnLine p l₂ :=
by sorry

end regression_lines_common_point_l1261_126173


namespace max_parts_three_planes_is_eight_l1261_126189

/-- The maximum number of parts that three planes can divide space into -/
def max_parts_three_planes : ℕ := 8

/-- Theorem: The maximum number of parts that three planes can divide space into is 8 -/
theorem max_parts_three_planes_is_eight :
  max_parts_three_planes = 8 := by sorry

end max_parts_three_planes_is_eight_l1261_126189


namespace estimated_probability_is_two_fifths_l1261_126108

/-- Represents a set of three-digit numbers -/
def RandomSet : Type := List Nat

/-- Checks if a number represents a rainy day (1-6) -/
def isRainyDay (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 6

/-- Counts the number of rainy days in a three-digit number -/
def countRainyDays (n : Nat) : Nat :=
  (if isRainyDay (n / 100) then 1 else 0) +
  (if isRainyDay ((n / 10) % 10) then 1 else 0) +
  (if isRainyDay (n % 10) then 1 else 0)

/-- Checks if a number represents exactly two rainy days -/
def hasTwoRainyDays (n : Nat) : Bool :=
  countRainyDays n = 2

/-- The given set of random numbers -/
def givenSet : RandomSet :=
  [180, 792, 454, 417, 165, 809, 798, 386, 196, 206]

/-- Theorem: The estimated probability of exactly two rainy days is 2/5 -/
theorem estimated_probability_is_two_fifths :
  (givenSet.filter hasTwoRainyDays).length / givenSet.length = 2 / 5 := by
  sorry

end estimated_probability_is_two_fifths_l1261_126108


namespace least_integer_square_condition_l1261_126160

theorem least_integer_square_condition (x : ℤ) : x^2 = 3*(2*x) + 50 → x ≥ -4 :=
by sorry

end least_integer_square_condition_l1261_126160


namespace sum_reciprocal_f_equals_251_385_l1261_126119

/-- The function f(n) that returns the integer closest to the cube root of n -/
def f (n : ℕ) : ℕ := sorry

/-- The sum of 1/f(k) from k=1 to 2023 -/
def sum_reciprocal_f : ℚ :=
  (Finset.range 2023).sum (λ k => 1 / (f (k + 1) : ℚ))

/-- The theorem stating that the sum of 1/f(k) from k=1 to 2023 is equal to 251.385 -/
theorem sum_reciprocal_f_equals_251_385 : sum_reciprocal_f = 251385 / 1000 := by sorry

end sum_reciprocal_f_equals_251_385_l1261_126119


namespace quadratic_root_implies_u_l1261_126171

theorem quadratic_root_implies_u (u : ℝ) : 
  (4 * (((-15 - Real.sqrt 145) / 8) ^ 2) + 15 * ((-15 - Real.sqrt 145) / 8) + u = 0) → 
  u = 5 := by
sorry

end quadratic_root_implies_u_l1261_126171


namespace M_mod_500_l1261_126186

/-- A sequence of positive integers whose binary representations have exactly 6 ones -/
def T : ℕ → ℕ :=
  sorry

/-- The 500th term in the sequence T -/
def M : ℕ :=
  T 500

theorem M_mod_500 : M % 500 = 198 := by
  sorry

end M_mod_500_l1261_126186


namespace storks_vs_birds_l1261_126170

theorem storks_vs_birds (initial_birds : ℕ) (additional_storks : ℕ) (additional_birds : ℕ) :
  initial_birds = 3 →
  additional_storks = 6 →
  additional_birds = 2 →
  additional_storks - (initial_birds + additional_birds) = 1 :=
by
  sorry

end storks_vs_birds_l1261_126170


namespace exists_partition_without_infinite_progression_l1261_126139

/-- A partition of natural numbers. -/
def Partition := ℕ → Bool

/-- Checks if a set contains an infinite arithmetic progression. -/
def HasInfiniteArithmeticProgression (p : Partition) : Prop :=
  ∃ a d : ℕ, d > 0 ∧ ∀ k : ℕ, p (a + k * d) = p a

/-- There exists a partition of natural numbers into two sets
    such that neither set contains an infinite arithmetic progression. -/
theorem exists_partition_without_infinite_progression :
  ∃ p : Partition, ¬HasInfiniteArithmeticProgression p ∧
                   ¬HasInfiniteArithmeticProgression (fun n => ¬(p n)) := by
  sorry

end exists_partition_without_infinite_progression_l1261_126139


namespace zoo_visitors_l1261_126180

def num_cars : ℕ := 3
def people_per_car : ℕ := 21

theorem zoo_visitors : num_cars * people_per_car = 63 := by
  sorry

end zoo_visitors_l1261_126180


namespace large_monkey_doll_cost_l1261_126116

def total_spent : ℝ := 300

def small_doll_discount : ℝ := 2

theorem large_monkey_doll_cost (large_cost : ℝ) 
  (h1 : large_cost > 0)
  (h2 : total_spent / (large_cost - small_doll_discount) = total_spent / large_cost + 25) :
  large_cost = 6 := by
sorry

end large_monkey_doll_cost_l1261_126116


namespace impossible_to_turn_all_lamps_off_l1261_126187

/-- Represents the state of a lamp (on or off) -/
inductive LampState
| On
| Off

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the chessboard state -/
def ChessboardState := Position → LampState

/-- Represents the allowed operations on the chessboard -/
inductive Operation
| InvertRow (row : Fin 8)
| InvertColumn (col : Fin 8)
| InvertDiagonal (d : ℤ) -- d represents the diagonal offset

/-- The initial state of the chessboard -/
def initialState : ChessboardState :=
  fun pos => if pos.row = 0 && pos.col = 3 then LampState.Off else LampState.On

/-- Apply an operation to the chessboard state -/
def applyOperation (state : ChessboardState) (op : Operation) : ChessboardState :=
  sorry

/-- Check if all lamps are off -/
def allLampsOff (state : ChessboardState) : Prop :=
  ∀ pos, state pos = LampState.Off

/-- The main theorem to be proved -/
theorem impossible_to_turn_all_lamps_off :
  ¬∃ (ops : List Operation), allLampsOff (ops.foldl applyOperation initialState) :=
sorry

end impossible_to_turn_all_lamps_off_l1261_126187


namespace triangle_area_with_given_base_and_height_l1261_126164

theorem triangle_area_with_given_base_and_height :
  ∀ (base height : ℝ), 
    base = 12 →
    height = 15 →
    (1 / 2 : ℝ) * base * height = 90 :=
by
  sorry

end triangle_area_with_given_base_and_height_l1261_126164


namespace simultaneous_pipe_filling_time_l1261_126109

theorem simultaneous_pipe_filling_time 
  (fill_time_A : ℝ) 
  (fill_time_B : ℝ) 
  (h1 : fill_time_A = 50) 
  (h2 : fill_time_B = 75) : 
  (1 / (1 / fill_time_A + 1 / fill_time_B)) = 30 := by
  sorry

end simultaneous_pipe_filling_time_l1261_126109


namespace inverse_proportionality_l1261_126191

/-- Given that α is inversely proportional to β, prove that if α = 4 when β = 9, 
    then α = -1/2 when β = -72 -/
theorem inverse_proportionality (α β : ℝ) (h : ∃ k, ∀ x y, x * y = k → α = x ∧ β = y) :
  (α = 4 ∧ β = 9) → (β = -72 → α = -1/2) := by
  sorry

end inverse_proportionality_l1261_126191


namespace starburst_candies_l1261_126143

theorem starburst_candies (mm_ratio : ℕ) (starburst_ratio : ℕ) (total_mm : ℕ) : ℕ :=
  let starburst_count := (starburst_ratio * total_mm) / mm_ratio
  by
    sorry

#check starburst_candies 13 8 143 = 88

end starburst_candies_l1261_126143


namespace basketball_price_correct_l1261_126161

/-- The price of a basketball that satisfies the given conditions -/
def basketball_price : ℚ := 29

/-- The number of basketballs bought by Coach A -/
def basketballs_count : ℕ := 10

/-- The price of each baseball -/
def baseball_price : ℚ := 5/2

/-- The number of baseballs bought by Coach B -/
def baseballs_count : ℕ := 14

/-- The price of the baseball bat -/
def bat_price : ℚ := 18

/-- The difference in spending between Coach A and Coach B -/
def spending_difference : ℚ := 237

theorem basketball_price_correct : 
  basketballs_count * basketball_price = 
  (baseballs_count * baseball_price + bat_price + spending_difference) :=
by sorry

end basketball_price_correct_l1261_126161


namespace simplify_sqrt_sum_l1261_126117

theorem simplify_sqrt_sum (x : ℝ) :
  Real.sqrt (4 * x^2 - 8 * x + 4) + Real.sqrt (4 * x^2 + 8 * x + 4) = 2 * (|x - 1| + |x + 1|) := by
  sorry

end simplify_sqrt_sum_l1261_126117


namespace f_of_two_equals_negative_twenty_six_l1261_126100

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_of_two_equals_negative_twenty_six 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
  (h2 : f (-2) = 10) :
  f 2 = -26 := by
  sorry

end f_of_two_equals_negative_twenty_six_l1261_126100


namespace function_property_l1261_126135

/-- Given a function f(x) = ax^5 + bx^3 + cx + 1, where a, b, and c are non-zero real numbers,
    if f(3) = 11, then f(-3) = -9. -/
theorem function_property (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + c * x + 1
  f 3 = 11 → f (-3) = -9 := by
sorry

end function_property_l1261_126135


namespace sqrt_18_div_sqrt_8_l1261_126196

theorem sqrt_18_div_sqrt_8 : Real.sqrt 18 / Real.sqrt 8 = 3 / 2 := by
  sorry

end sqrt_18_div_sqrt_8_l1261_126196


namespace rhombus_properties_l1261_126140

/-- Properties of a rhombus -/
structure Rhombus where
  /-- The diagonals of a rhombus are perpendicular to each other -/
  diagonals_perpendicular : Prop
  /-- The diagonals of a rhombus bisect each other -/
  diagonals_bisect : Prop

theorem rhombus_properties (R : Rhombus) : 
  (R.diagonals_perpendicular ∨ R.diagonals_bisect) ∧ 
  (R.diagonals_perpendicular ∧ R.diagonals_bisect) ∧ 
  ¬(¬R.diagonals_perpendicular) := by
  sorry

#check rhombus_properties

end rhombus_properties_l1261_126140


namespace appropriate_word_count_l1261_126199

def speech_duration_min : ℝ := 40
def speech_duration_max : ℝ := 50
def speech_rate : ℝ := 160
def word_count : ℕ := 7600

theorem appropriate_word_count : 
  speech_duration_min * speech_rate ≤ word_count ∧ 
  word_count ≤ speech_duration_max * speech_rate := by
  sorry

end appropriate_word_count_l1261_126199


namespace point_in_fourth_quadrant_m_range_l1261_126118

-- Define the point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (m + 3, m - 1)

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant_m_range :
  ∀ m : ℝ, in_fourth_quadrant (P m) ↔ -3 < m ∧ m < 1 :=
by sorry

end point_in_fourth_quadrant_m_range_l1261_126118


namespace no_tangent_line_with_slope_three_halves_for_sine_l1261_126172

theorem no_tangent_line_with_slope_three_halves_for_sine :
  ¬∃ (x : ℝ), Real.cos x = (3 : ℝ) / 2 := by
  sorry

end no_tangent_line_with_slope_three_halves_for_sine_l1261_126172


namespace cauliflower_sales_value_l1261_126163

def farmers_market_sales (total_earnings broccoli_sales carrot_sales spinach_sales tomato_sales cauliflower_sales : ℝ) : Prop :=
  total_earnings = 500 ∧
  broccoli_sales = 57 ∧
  carrot_sales = 2 * broccoli_sales ∧
  spinach_sales = (carrot_sales / 2) + 16 ∧
  tomato_sales = broccoli_sales + spinach_sales ∧
  total_earnings = broccoli_sales + carrot_sales + spinach_sales + tomato_sales + cauliflower_sales

theorem cauliflower_sales_value :
  ∀ total_earnings broccoli_sales carrot_sales spinach_sales tomato_sales cauliflower_sales : ℝ,
  farmers_market_sales total_earnings broccoli_sales carrot_sales spinach_sales tomato_sales cauliflower_sales →
  cauliflower_sales = 126 := by
sorry

end cauliflower_sales_value_l1261_126163


namespace problem_solution_l1261_126123

theorem problem_solution (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0)
    (h4 : 3 * a + 2 * b + c = 5) (h5 : 2 * a + b - 3 * c = 1) :
    (3 / 7 ≤ c ∧ c ≤ 7 / 11) ∧
    (∀ x, 3 * a + b - 7 * c ≤ x → x ≤ -1 / 11) ∧
    (∀ y, -5 / 7 ≤ y → y ≤ 3 * a + b - 7 * c) :=
  sorry

end problem_solution_l1261_126123


namespace speed_in_still_water_l1261_126114

/-- 
Given a man's upstream and downstream speeds, calculate his speed in still water.
-/
theorem speed_in_still_water 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 25) 
  (h2 : downstream_speed = 55) : 
  (upstream_speed + downstream_speed) / 2 = 40 := by
  sorry

end speed_in_still_water_l1261_126114


namespace unique_triple_sum_l1261_126144

theorem unique_triple_sum (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y : ℚ) / z + (y * z : ℚ) / x + (z * x : ℚ) / y = 3 → x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end unique_triple_sum_l1261_126144


namespace soda_difference_is_21_l1261_126149

/-- The number of regular soda bottles -/
def regular_soda : ℕ := 81

/-- The number of diet soda bottles -/
def diet_soda : ℕ := 60

/-- The difference between regular and diet soda bottles -/
def soda_difference : ℕ := regular_soda - diet_soda

theorem soda_difference_is_21 : soda_difference = 21 := by
  sorry

end soda_difference_is_21_l1261_126149


namespace angle_with_complement_half_supplement_l1261_126129

theorem angle_with_complement_half_supplement (x : ℝ) :
  (90 - x) = (1/2) * (180 - x) → x = 90 := by
  sorry

end angle_with_complement_half_supplement_l1261_126129


namespace dvd_book_capacity_l1261_126158

theorem dvd_book_capacity (total_capacity : ℕ) (current_count : ℕ) (h1 : total_capacity = 126) (h2 : current_count = 81) :
  total_capacity - current_count = 45 := by
  sorry

end dvd_book_capacity_l1261_126158


namespace bag_composition_for_expected_value_l1261_126197

/-- Represents the contents of a bag of slips --/
structure BagOfSlips where
  threes : ℕ
  fives : ℕ
  eights : ℕ

/-- Calculates the expected value of a randomly drawn slip --/
def expectedValue (bag : BagOfSlips) : ℚ :=
  (3 * bag.threes + 5 * bag.fives + 8 * bag.eights) / 20

/-- Theorem statement --/
theorem bag_composition_for_expected_value :
  ∃ (bag : BagOfSlips),
    bag.threes + bag.fives + bag.eights = 20 ∧
    expectedValue bag = 57/10 ∧
    bag.threes = 4 ∧
    bag.fives = 10 ∧
    bag.eights = 6 := by
  sorry

end bag_composition_for_expected_value_l1261_126197


namespace rug_area_theorem_l1261_126145

/-- Given three overlapping rugs, prove their combined area is 212 square meters -/
theorem rug_area_theorem (total_covered_area single_layer_area double_layer_area triple_layer_area : ℝ) :
  total_covered_area = 140 →
  double_layer_area = 24 →
  triple_layer_area = 24 →
  single_layer_area = total_covered_area - double_layer_area - triple_layer_area →
  single_layer_area + 2 * double_layer_area + 3 * triple_layer_area = 212 :=
by sorry

end rug_area_theorem_l1261_126145


namespace children_count_l1261_126125

/-- The number of children required to assemble one small robot -/
def small_robot_children : ℕ := 2

/-- The number of children required to assemble one large robot -/
def large_robot_children : ℕ := 3

/-- The number of small robots assembled -/
def small_robots : ℕ := 18

/-- The number of large robots assembled -/
def large_robots : ℕ := 12

/-- The total number of children -/
def total_children : ℕ := small_robot_children * small_robots + large_robot_children * large_robots

theorem children_count : total_children = 72 := by sorry

end children_count_l1261_126125


namespace equation_solution_l1261_126166

theorem equation_solution (a b x : ℝ) : 
  (a * Real.sin x + b) / (b * Real.cos x + a) = (a * Real.cos x + b) / (b * Real.sin x + a) ↔ 
  (∃ k : ℤ, x = k * Real.pi + Real.pi / 4) ∨ 
  (b = Real.sqrt 2 * a ∧ ∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 4) ∨
  (b = -Real.sqrt 2 * a ∧ ∃ k : ℤ, x = (2 * k + 1) * Real.pi) := by
sorry


end equation_solution_l1261_126166


namespace function_inequality_and_minimum_value_l1261_126127

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 2|

-- Define the solution set M
def M : Set ℝ := {x | 2/3 ≤ x ∧ x ≤ 6}

-- Define the theorem
theorem function_inequality_and_minimum_value :
  (∀ x ∈ M, f x ≥ -1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 4*a + b + c = 6 →
    1/(2*a + b) + 1/(2*a + c) ≥ 2/3) ∧
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 4*a + b + c = 6 ∧
    1/(2*a + b) + 1/(2*a + c) = 2/3) :=
by sorry

end function_inequality_and_minimum_value_l1261_126127


namespace calculate_expression_l1261_126183

theorem calculate_expression : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + |(-3)| = 4 := by
  sorry

end calculate_expression_l1261_126183


namespace bottle_cap_configurations_l1261_126147

theorem bottle_cap_configurations : ∃ (n m : ℕ), n ≠ m ∧ n > 0 ∧ m > 0 ∧ 3 ∣ n ∧ 4 ∣ n ∧ 3 ∣ m ∧ 4 ∣ m :=
by sorry

end bottle_cap_configurations_l1261_126147


namespace exponent_division_l1261_126137

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^4 / a^2 = a^2 := by
  sorry

end exponent_division_l1261_126137


namespace birds_total_distance_l1261_126110

def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30
def flight_time : ℕ := 2

def total_distance : ℕ := eagle_speed * flight_time + falcon_speed * flight_time + 
                           pelican_speed * flight_time + hummingbird_speed * flight_time

theorem birds_total_distance : total_distance = 248 := by
  sorry

end birds_total_distance_l1261_126110


namespace opposite_sqrt5_minus_2_l1261_126134

theorem opposite_sqrt5_minus_2 :
  -(Real.sqrt 5 - 2) = 2 - Real.sqrt 5 := by sorry

end opposite_sqrt5_minus_2_l1261_126134


namespace string_average_length_l1261_126182

/-- Given 6 strings where 2 strings have an average length of 70 cm
    and the other 4 strings have an average length of 85 cm,
    prove that the average length of all 6 strings is 80 cm. -/
theorem string_average_length :
  let total_strings : ℕ := 6
  let group1_strings : ℕ := 2
  let group2_strings : ℕ := 4
  let group1_avg_length : ℝ := 70
  let group2_avg_length : ℝ := 85
  (total_strings = group1_strings + group2_strings) →
  (group1_strings * group1_avg_length + group2_strings * group2_avg_length) / total_strings = 80 :=
by sorry

end string_average_length_l1261_126182


namespace strawberries_picked_l1261_126154

/-- Given that Paul started with 28 strawberries and ended up with 63 strawberries,
    prove that he picked 35 strawberries. -/
theorem strawberries_picked (initial : ℕ) (final : ℕ) (h1 : initial = 28) (h2 : final = 63) :
  final - initial = 35 := by
  sorry

end strawberries_picked_l1261_126154


namespace inscribed_cube_volume_l1261_126138

/-- A pyramid with a square base and equilateral triangular lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (base_is_square : base_side > 0)
  (lateral_faces_equilateral : True)

/-- A cube inscribed in a pyramid -/
structure InscribedCube (p : Pyramid) :=
  (edge_length : ℝ)
  (touches_base_center : True)
  (touches_apex : True)

/-- The volume of the inscribed cube -/
def cube_volume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.edge_length ^ 3

/-- The main theorem: volume of the inscribed cube in the given pyramid -/
theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p)
  (h_base : p.base_side = 2) :
  cube_volume p c = 2 * Real.sqrt 6 / 9 := by
  sorry

#check inscribed_cube_volume

end inscribed_cube_volume_l1261_126138


namespace prove_postcard_selection_l1261_126181

def postcardSelection (typeA : ℕ) (typeB : ℕ) (teachers : ℕ) : Prop :=
  typeA = 2 ∧ typeB = 3 ∧ teachers = 4 →
  (Nat.choose teachers typeA + Nat.choose (teachers - 1) (typeA - 1)) = 10

theorem prove_postcard_selection :
  postcardSelection 2 3 4 :=
by
  sorry

end prove_postcard_selection_l1261_126181


namespace company_a_bottles_company_a_bottles_proof_l1261_126165

/-- Proves that Company A sold 300 bottles given the problem conditions -/
theorem company_a_bottles : ℕ :=
  let company_a_price : ℚ := 4
  let company_b_price : ℚ := 7/2
  let company_b_bottles : ℕ := 350
  let revenue_difference : ℚ := 25
  300

theorem company_a_bottles_proof (company_a_price : ℚ) (company_b_price : ℚ) 
  (company_b_bottles : ℕ) (revenue_difference : ℚ) :
  company_a_price = 4 →
  company_b_price = 7/2 →
  company_b_bottles = 350 →
  revenue_difference = 25 →
  company_a_price * company_a_bottles = 
    company_b_price * company_b_bottles + revenue_difference :=
by sorry

end company_a_bottles_company_a_bottles_proof_l1261_126165


namespace square_of_negative_product_l1261_126112

theorem square_of_negative_product (a b : ℝ) : (-a * b^3)^2 = a^2 * b^6 := by
  sorry

end square_of_negative_product_l1261_126112


namespace circle_area_ratio_l1261_126152

theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0)
  (h_arc : C * (60 / 360) = D * (40 / 360)) :
  (C^2 / D^2 : ℝ) = 4/9 := by
sorry

end circle_area_ratio_l1261_126152


namespace train_speed_calculation_l1261_126177

theorem train_speed_calculation (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 250 →
  bridge_length = 150 →
  time = 41.142857142857146 →
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / time
  let speed_kmh := speed_ms * 3.6
  ⌊speed_kmh⌋ = 35 := by sorry

end train_speed_calculation_l1261_126177


namespace expression_is_integer_l1261_126151

theorem expression_is_integer (a b c : ℝ) 
  (h1 : a^2 + b^2 = 2*c^2)
  (h2 : a ≠ b)
  (h3 : c ≠ -a)
  (h4 : c ≠ -b) :
  ∃ n : ℤ, ((a+b+2*c)*(2*a^2-b^2-c^2)) / ((a-b)*(a+c)*(b+c)) = n := by
  sorry

end expression_is_integer_l1261_126151


namespace total_weight_of_hay_bales_l1261_126174

/-- Calculates the total weight of hay bales in a barn after adding new bales -/
theorem total_weight_of_hay_bales
  (initial_bales : ℕ)
  (initial_weight : ℕ)
  (total_bales : ℕ)
  (new_weight : ℕ)
  (h1 : initial_bales = 73)
  (h2 : initial_weight = 45)
  (h3 : total_bales = 96)
  (h4 : new_weight = 50)
  (h5 : total_bales > initial_bales) :
  initial_bales * initial_weight + (total_bales - initial_bales) * new_weight = 4435 :=
by sorry

#check total_weight_of_hay_bales

end total_weight_of_hay_bales_l1261_126174


namespace min_value_sum_min_value_achievable_l1261_126103

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a)) ≥ 1 / Real.rpow 6 (1/3) :=
by sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / (3 * b) + b / (6 * c) + c / (9 * a)) = 1 / Real.rpow 6 (1/3) :=
by sorry

end min_value_sum_min_value_achievable_l1261_126103


namespace diameter_equation_l1261_126126

/-- Given a circle and a line, if a diameter of the circle intersects with the line at the midpoint
    of the chord cut by the circle, then the equation of the line on which this diameter lies
    is 2x + y - 3 = 0. -/
theorem diameter_equation (x y : ℝ) :
  (∃ (a b : ℝ), (x - 2)^2 + (y + 1)^2 = 16 ∧ x - 2*y + 3 = 0 ∧
   (a - 2)^2 + (b + 1)^2 = 16 ∧
   (x + a)/2 - 2 = 0 ∧ (y + b)/2 + 1 = 0) →
  (2*x + y - 3 = 0) :=
sorry

end diameter_equation_l1261_126126


namespace houses_with_pool_count_l1261_126106

/-- Represents the number of houses in a development with various features -/
structure Development where
  total : ℕ
  with_garage : ℕ
  with_both : ℕ
  with_neither : ℕ

/-- The number of houses with an in-the-ground swimming pool in the development -/
def houses_with_pool (d : Development) : ℕ :=
  d.total - d.with_garage + d.with_both - d.with_neither

/-- Theorem stating that in the given development, 40 houses have an in-the-ground swimming pool -/
theorem houses_with_pool_count (d : Development) 
  (h1 : d.total = 65)
  (h2 : d.with_garage = 50)
  (h3 : d.with_both = 35)
  (h4 : d.with_neither = 10) : 
  houses_with_pool d = 40 := by
  sorry

end houses_with_pool_count_l1261_126106


namespace arithmetic_seq_problem_l1261_126190

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given conditions for the arithmetic sequence -/
structure ArithSeqConditions (a : ℕ → ℚ) : Prop :=
  (is_arith : ArithmeticSequence a)
  (prod_eq : a 5 * a 7 = 6)
  (sum_eq : a 2 + a 10 = 5)

/-- Theorem statement -/
theorem arithmetic_seq_problem (a : ℕ → ℚ) (h : ArithSeqConditions a) :
  (a 10 - a 6 = 2) ∨ (a 10 - a 6 = -2) := by
  sorry

end arithmetic_seq_problem_l1261_126190


namespace coin_count_l1261_126175

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the total value of coins in cents -/
def total_value : ℕ := 440

theorem coin_count (n : ℕ) : 
  n * (quarter_value + dime_value + nickel_value) = total_value → 
  n = 11 := by sorry

end coin_count_l1261_126175


namespace arithmetic_sequence_sum_l1261_126162

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The statement to be proved -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3)^2 - 6*(a 3) + 8 = 0 →
  (a 15)^2 - 6*(a 15) + 8 = 0 →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by sorry

end arithmetic_sequence_sum_l1261_126162


namespace wall_length_with_mirrors_l1261_126195

/-- The length of a rectangular wall with specific mirror configurations -/
theorem wall_length_with_mirrors (square_side : ℝ) (circle_diameter : ℝ) (wall_width : ℝ)
  (h_square : square_side = 18)
  (h_circle : circle_diameter = 20)
  (h_width : wall_width = 32)
  (h_combined_area : square_side ^ 2 + π * (circle_diameter / 2) ^ 2 = wall_width * wall_length / 2) :
  wall_length = (324 + 100 * π) / 16 := by
  sorry

#check wall_length_with_mirrors

end wall_length_with_mirrors_l1261_126195


namespace total_digits_memorized_l1261_126124

/-- The number of digits of pi memorized by each person --/
structure PiDigits where
  carlos : ℕ
  sam : ℕ
  mina : ℕ
  nina : ℕ

/-- The conditions given in the problem --/
def satisfies_conditions (p : PiDigits) : Prop :=
  p.sam = p.carlos + 6 ∧
  p.mina = 6 * p.carlos ∧
  p.nina = 4 * p.carlos ∧
  p.mina = 24

/-- The theorem to be proved --/
theorem total_digits_memorized (p : PiDigits) 
  (h : satisfies_conditions p) : 
  p.sam + p.carlos + p.mina + p.nina = 54 := by
  sorry


end total_digits_memorized_l1261_126124


namespace walter_zoo_time_l1261_126159

def time_at_zoo (seal_time penguin_factor elephant_time : ℕ) : ℕ :=
  seal_time + (seal_time * penguin_factor) + elephant_time

theorem walter_zoo_time :
  time_at_zoo 13 8 13 = 130 :=
by sorry

end walter_zoo_time_l1261_126159


namespace power_of_three_l1261_126104

theorem power_of_three (m n : ℕ) (h1 : 3^m = 4) (h2 : 3^n = 5) : 3^(2*m + n) = 80 := by
  sorry

end power_of_three_l1261_126104


namespace abs_greater_than_y_if_x_greater_than_y_l1261_126192

theorem abs_greater_than_y_if_x_greater_than_y (x y : ℝ) (h : x > y) : |x| > y := by
  sorry

end abs_greater_than_y_if_x_greater_than_y_l1261_126192


namespace computer_literate_female_employees_l1261_126157

theorem computer_literate_female_employees 
  (total_employees : ℕ)
  (female_percentage : ℚ)
  (male_literate_percentage : ℚ)
  (total_literate_percentage : ℚ)
  (h_total : total_employees = 1300)
  (h_female : female_percentage = 60 / 100)
  (h_male_literate : male_literate_percentage = 50 / 100)
  (h_total_literate : total_literate_percentage = 62 / 100) :
  ↑(total_employees * female_percentage * total_literate_percentage - 
    total_employees * (1 - female_percentage) * male_literate_percentage : ℚ).num = 546 := by
  sorry

end computer_literate_female_employees_l1261_126157


namespace vector_problem_l1261_126111

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the points
variable (A B C D : V)

-- State the theorem
theorem vector_problem (h1 : e₁ ≠ 0) (h2 : e₂ ≠ 0) (h3 : ¬ ∃ (r : ℝ), e₁ = r • e₂) 
  (h4 : B - A = 3 • e₁ + k • e₂) 
  (h5 : C - B = 4 • e₁ + e₂) 
  (h6 : D - C = 8 • e₁ - 9 • e₂)
  (h7 : ∃ (t : ℝ), D - A = t • (B - A)) :
  k = -2 := by sorry

end vector_problem_l1261_126111


namespace eight_coin_flip_probability_l1261_126150

theorem eight_coin_flip_probability :
  let n : ℕ := 8
  let p : ℚ := 1 / 2  -- probability of heads for a fair coin
  let prob_seven_heads : ℚ := n.choose (n - 1) * p^(n - 1) * (1 - p)
  let prob_seven_tails : ℚ := n.choose (n - 1) * (1 - p)^(n - 1) * p
  prob_seven_heads + prob_seven_tails = 1 / 16 := by
  sorry

end eight_coin_flip_probability_l1261_126150


namespace sufficient_but_not_necessary_l1261_126105

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x = 1 → x^2 ≠ 1) ∧ 
  ¬(∀ x, x^2 ≠ 1 → x = 1) :=
by sorry

end sufficient_but_not_necessary_l1261_126105


namespace wednesday_to_tuesday_ratio_l1261_126155

/-- The number of dinners sold on each day of the week --/
structure DinnerSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- The conditions of the dinner sales problem --/
def dinner_problem (sales : DinnerSales) : Prop :=
  sales.monday = 40 ∧
  sales.tuesday = sales.monday + 40 ∧
  sales.thursday = sales.wednesday + 3 ∧
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 203

/-- The theorem stating the ratio of Wednesday's sales to Tuesday's sales --/
theorem wednesday_to_tuesday_ratio (sales : DinnerSales) 
  (h : dinner_problem sales) : 
  (sales.wednesday : ℚ) / sales.tuesday = 1 / 2 := by
  sorry


end wednesday_to_tuesday_ratio_l1261_126155


namespace no_fixed_extreme_points_l1261_126185

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 3

/-- The derivative of f -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

/-- Theorem: There do not exist real numbers a and b such that f has two distinct extreme points that are also fixed points -/
theorem no_fixed_extreme_points :
  ¬ ∃ (a b x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧
    f' a b x₁ = 0 ∧
    f' a b x₂ = 0 ∧
    f a b x₁ = x₁ ∧
    f a b x₂ = x₂ := by
  sorry


end no_fixed_extreme_points_l1261_126185


namespace complex_modulus_problem_l1261_126153

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_modulus_problem_l1261_126153


namespace sum_of_valid_m_l1261_126132

def inequality_system (x m : ℤ) : Prop :=
  (x - 2) / 4 < (x - 1) / 3 ∧ 3 * x - m ≤ 3 - x

def equation_system (x y m : ℤ) : Prop :=
  m * x + y = 4 ∧ 3 * x - y = 0

theorem sum_of_valid_m :
  (∃ (s : Finset ℤ), 
    (∀ m ∈ s, 
      (∃! (a b : ℤ), inequality_system a m) ∧
      (∃ (x y : ℤ), equation_system x y m)) ∧
    (s.sum id = -3)) :=
sorry

end sum_of_valid_m_l1261_126132


namespace exponent_sum_equality_l1261_126198

theorem exponent_sum_equality : (-3)^(4^2) + 2^(3^2) = 43047233 := by
  sorry

end exponent_sum_equality_l1261_126198


namespace bowtie_equation_solution_l1261_126115

/-- The operation ⊗ as defined in the problem -/
noncomputable def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem stating that if 5 ⊗ g = 11, then g = 30 -/
theorem bowtie_equation_solution :
  ∃ g : ℝ, bowtie 5 g = 11 ∧ g = 30 := by sorry

end bowtie_equation_solution_l1261_126115


namespace solve_sandwich_problem_l1261_126142

def sandwich_problem (sandwich_cost : ℕ) (paid_amount : ℕ) (change_received : ℕ) : Prop :=
  let spent_amount := paid_amount - change_received
  let num_sandwiches := spent_amount / sandwich_cost
  num_sandwiches = 3

theorem solve_sandwich_problem :
  sandwich_problem 5 20 5 := by
  sorry

end solve_sandwich_problem_l1261_126142


namespace mr_slinkums_shipment_count_l1261_126167

theorem mr_slinkums_shipment_count : ∀ (total : ℕ), 
  (75 : ℚ) / 100 * total = 150 → total = 200 := by
  sorry

end mr_slinkums_shipment_count_l1261_126167


namespace inequality_system_solution_range_l1261_126156

theorem inequality_system_solution_range (a : ℝ) : 
  (∀ x : ℝ, (2 * x > 4 ∧ 3 * x + a > 0) ↔ x > 2) → 
  a ≥ -6 :=
by sorry

end inequality_system_solution_range_l1261_126156


namespace evaluate_expression_l1261_126146

theorem evaluate_expression : 3000 * (3000^3001)^2 = 3000^6003 := by
  sorry

end evaluate_expression_l1261_126146


namespace inequality_empty_solution_set_l1261_126122

theorem inequality_empty_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 2 * m * x + 1 ≥ 0) → 0 ≤ m ∧ m ≤ 1 := by
  sorry

end inequality_empty_solution_set_l1261_126122


namespace inscribed_circle_radius_l1261_126194

/-- Given a triangle DEF with side lengths DE = 8, DF = 5, and EF = 9,
    the radius of its inscribed circle is 6√11/11. -/
theorem inscribed_circle_radius (DE DF EF : ℝ) (hDE : DE = 8) (hDF : DF = 5) (hEF : EF = 9) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  area / s = 6 * Real.sqrt 11 / 11 := by sorry

end inscribed_circle_radius_l1261_126194


namespace two_fifths_of_n_is_80_l1261_126136

theorem two_fifths_of_n_is_80 (n : ℚ) (h : n = 5 / 6 * 240) : 2 / 5 * n = 80 := by
  sorry

end two_fifths_of_n_is_80_l1261_126136


namespace alcohol_mixture_proof_l1261_126131

/-- Proves that mixing 250 mL of 10% alcohol solution with 750 mL of 30% alcohol solution results in a 25% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 250
  let y_volume : ℝ := 750
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.25
  let total_volume : ℝ := x_volume + y_volume
  let total_alcohol : ℝ := x_volume * x_concentration + y_volume * y_concentration
  total_alcohol / total_volume = target_concentration := by
  sorry

#check alcohol_mixture_proof

end alcohol_mixture_proof_l1261_126131


namespace inscribed_quadrilateral_fourth_side_l1261_126102

/-- A quadrilateral inscribed in a circle with given side lengths -/
structure InscribedQuadrilateral where
  -- The radius of the circumscribed circle
  radius : ℝ
  -- The lengths of the four sides of the quadrilateral
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Theorem stating that for a quadrilateral inscribed in a circle with radius 300√2
    and three sides of lengths 300, 300, and 150√2, the fourth side has length 300√2 -/
theorem inscribed_quadrilateral_fourth_side
  (q : InscribedQuadrilateral)
  (h_radius : q.radius = 300 * Real.sqrt 2)
  (h_side1 : q.side1 = 300)
  (h_side2 : q.side2 = 300)
  (h_side3 : q.side3 = 150 * Real.sqrt 2) :
  q.side4 = 300 * Real.sqrt 2 := by
  sorry

end inscribed_quadrilateral_fourth_side_l1261_126102


namespace platform_length_l1261_126148

/-- The length of a platform given train speed and passing times -/
theorem platform_length (train_speed : ℝ) (platform_time : ℝ) (man_time : ℝ) : 
  train_speed = 54 → platform_time = 16 → man_time = 10 → 
  (train_speed * 5 / 18) * (platform_time - man_time) = 90 := by
  sorry

#check platform_length

end platform_length_l1261_126148


namespace max_value_complex_expression_l1261_126128

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 3) :
  Complex.abs ((z - 1) * (z + 1)^2) ≤ 3 * Real.sqrt 3 ∧
  ∃ w : ℂ, Complex.abs w = Real.sqrt 3 ∧ Complex.abs ((w - 1) * (w + 1)^2) = 3 * Real.sqrt 3 :=
by sorry

end max_value_complex_expression_l1261_126128


namespace base3_102012_equals_302_l1261_126176

def base3_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base3_102012_equals_302 :
  base3_to_base10 [1, 0, 2, 0, 1, 2] = 302 := by
  sorry

end base3_102012_equals_302_l1261_126176


namespace units_digit_of_sum_l1261_126178

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_sum : unitsDigit (42^2 + 25^3) = 9 := by
  sorry

end units_digit_of_sum_l1261_126178


namespace acid_solution_replacement_l1261_126184

/-- Proves that the fraction of original 50% acid solution replaced with 20% acid solution to obtain a 35% acid solution is 0.5 -/
theorem acid_solution_replacement (V : ℝ) (h : V > 0) :
  let original_concentration : ℝ := 0.5
  let replacement_concentration : ℝ := 0.2
  let final_concentration : ℝ := 0.35
  let x : ℝ := (original_concentration - final_concentration) / (original_concentration - replacement_concentration)
  x = 0.5 := by sorry

end acid_solution_replacement_l1261_126184


namespace lindas_mean_score_l1261_126168

def scores : List ℕ := [80, 86, 90, 92, 95, 97]

def jakes_mean : ℕ := 89

theorem lindas_mean_score (h1 : scores.length = 6)
  (h2 : ∃ (jake_scores linda_scores : List ℕ),
    jake_scores.length = 3 ∧
    linda_scores.length = 3 ∧
    jake_scores ++ linda_scores = scores)
  (h3 : ∃ (jake_scores : List ℕ),
    jake_scores.length = 3 ∧
    jake_scores.sum / jake_scores.length = jakes_mean) :
  ∃ (linda_scores : List ℕ),
    linda_scores.length = 3 ∧
    linda_scores.sum / linda_scores.length = 91 :=
by sorry

end lindas_mean_score_l1261_126168
