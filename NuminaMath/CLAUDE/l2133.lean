import Mathlib

namespace discount_effect_l2133_213385

theorem discount_effect (P N : ℝ) (h_pos_P : P > 0) (h_pos_N : N > 0) :
  let D : ℝ := 10
  let new_price : ℝ := (1 - D / 100) * P
  let new_quantity : ℝ := 1.25 * N
  let old_income : ℝ := P * N
  let new_income : ℝ := new_price * new_quantity
  (new_quantity / N = 1.25) ∧ (new_income / old_income = 1.125) :=
sorry

end discount_effect_l2133_213385


namespace max_value_on_circle_l2133_213375

theorem max_value_on_circle : 
  ∃ (M : ℝ), M = 8 ∧ 
  (∀ (x y : ℝ), (x - 2)^2 + y^2 = 1 → |3*x + 4*y - 3| ≤ M) ∧
  (∃ (x y : ℝ), (x - 2)^2 + y^2 = 1 ∧ |3*x + 4*y - 3| = M) :=
by sorry

end max_value_on_circle_l2133_213375


namespace difference_not_one_l2133_213361

theorem difference_not_one (a b : ℝ) (h : a^2 - b^2 + 2*a - 4*b - 3 ≠ 0) : a - b ≠ 1 := by
  sorry

end difference_not_one_l2133_213361


namespace solve_books_problem_l2133_213397

def books_problem (initial_books : ℕ) (new_books : ℕ) : Prop :=
  let after_nephew := initial_books - (initial_books / 4)
  let after_library := after_nephew - (after_nephew / 5)
  let after_neighbor := after_library - (after_library / 6)
  let final_books := after_neighbor + new_books
  final_books = 68

theorem solve_books_problem :
  books_problem 120 8 := by
  sorry

end solve_books_problem_l2133_213397


namespace pentagon_area_sqrt_sum_m_n_l2133_213398

/-- A pentagon constructed from 11 line segments of length 2 --/
structure Pentagon where
  /-- The number of line segments --/
  num_segments : ℕ
  /-- The length of each segment --/
  segment_length : ℝ
  /-- Assertion that the pentagon is constructed from 11 segments of length 2 --/
  h_segments : num_segments = 11 ∧ segment_length = 2

/-- The area of the pentagon --/
noncomputable def area (p : Pentagon) : ℝ := sorry

/-- Theorem stating that the area of the pentagon can be expressed as √11 + √12 --/
theorem pentagon_area_sqrt (p : Pentagon) : 
  area p = Real.sqrt 11 + Real.sqrt 12 := by sorry

/-- Corollary showing that m + n = 23 --/
theorem sum_m_n (p : Pentagon) : 
  ∃ (m n : ℕ), (m > 0 ∧ n > 0) ∧ area p = Real.sqrt m + Real.sqrt n ∧ m + n = 23 := by sorry

end pentagon_area_sqrt_sum_m_n_l2133_213398


namespace power_equality_l2133_213389

theorem power_equality (m : ℕ) : 9^4 = 3^(2*m) → m = 4 := by
  sorry

end power_equality_l2133_213389


namespace bird_count_l2133_213316

/-- The number of birds in a crape myrtle tree --/
theorem bird_count (swallows bluebirds cardinals : ℕ) : 
  swallows = 2 →
  bluebirds = 2 * swallows →
  cardinals = 3 * bluebirds →
  swallows + bluebirds + cardinals = 18 := by
sorry

end bird_count_l2133_213316


namespace binomial_510_510_l2133_213314

theorem binomial_510_510 : Nat.choose 510 510 = 1 := by
  sorry

end binomial_510_510_l2133_213314


namespace circle_m_range_l2133_213331

/-- A circle in the xy-plane can be represented by the equation x² + y² + dx + ey + f = 0,
    where d, e, and f are real constants, and d² + e² - 4f > 0 -/
def is_circle (d e f : ℝ) : Prop := d^2 + e^2 - 4*f > 0

/-- The equation x² + y² - 2x - 4y + m = 0 represents a circle -/
def represents_circle (m : ℝ) : Prop := is_circle (-2) (-4) m

theorem circle_m_range :
  ∀ m : ℝ, represents_circle m → m < 5 := by sorry

end circle_m_range_l2133_213331


namespace polynomial_value_at_five_l2133_213327

theorem polynomial_value_at_five : 
  let x : ℤ := 5
  x^5 - 3*x^3 - 5*x = 2725 := by
  sorry

end polynomial_value_at_five_l2133_213327


namespace projectile_collision_time_l2133_213364

-- Define the parameters
def initial_distance : ℝ := 1386 -- km
def speed1 : ℝ := 445 -- km/h
def speed2 : ℝ := 545 -- km/h

-- Define the theorem
theorem projectile_collision_time :
  let relative_speed : ℝ := speed1 + speed2
  let time_hours : ℝ := initial_distance / relative_speed
  let time_minutes : ℝ := time_hours * 60
  ∃ ε > 0, |time_minutes - 84| < ε :=
sorry

end projectile_collision_time_l2133_213364


namespace rectangular_box_diagonals_l2133_213362

theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (surface_area : 2 * (a * b + b * c + a * c) = 118) 
  (edge_sum : 4 * (a + b + c) = 52) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 51 := by
  sorry

end rectangular_box_diagonals_l2133_213362


namespace min_airlines_needed_l2133_213352

/-- Represents the number of towns -/
def num_towns : ℕ := 21

/-- Represents the size of the group of towns served by each airline -/
def group_size : ℕ := 5

/-- Calculates the total number of pairs of towns -/
def total_pairs : ℕ := num_towns.choose 2

/-- Calculates the number of pairs served by each airline -/
def pairs_per_airline : ℕ := group_size.choose 2

/-- Theorem stating the minimum number of airlines needed -/
theorem min_airlines_needed : 
  ∃ (n : ℕ), n * pairs_per_airline ≥ total_pairs ∧ 
  ∀ (m : ℕ), m * pairs_per_airline ≥ total_pairs → n ≤ m :=
by sorry

end min_airlines_needed_l2133_213352


namespace min_queries_for_parity_l2133_213309

/-- Represents a query about the parity of balls in 15 bags -/
def Query := Fin 100 → Bool

/-- Represents the state of all bags -/
def BagState := Fin 100 → Bool

/-- The result of a query given a bag state -/
def queryResult (q : Query) (s : BagState) : Bool :=
  (List.filter (fun i => q i) (List.range 100)).foldl (fun acc i => acc ≠ s i) false

/-- A set of queries is sufficient if it can determine the parity of bag 1 -/
def isSufficient (qs : List Query) : Prop :=
  ∀ s1 s2 : BagState, (∀ q ∈ qs, queryResult q s1 = queryResult q s2) → s1 0 = s2 0

theorem min_queries_for_parity : 
  (∃ qs : List Query, qs.length = 3 ∧ isSufficient qs) ∧
  (∀ qs : List Query, qs.length < 3 → ¬isSufficient qs) := by
  sorry

end min_queries_for_parity_l2133_213309


namespace three_digit_sum_theorem_l2133_213341

/-- The sum of all three-digit numbers -/
def sum_three_digit : ℕ := 494550

/-- Predicate to check if a number is three-digit -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem three_digit_sum_theorem (x y : ℕ) :
  is_three_digit x ∧ is_three_digit y ∧ 
  sum_three_digit - x - y = 600 * x →
  x = 823 ∧ y = 527 := by
sorry

end three_digit_sum_theorem_l2133_213341


namespace sum_of_squared_coefficients_l2133_213373

def polynomial (x : ℝ) : ℝ := 6 * (x^5 + 2*x^3 + x + 3)

theorem sum_of_squared_coefficients : 
  (6^2 : ℝ) + (12^2 : ℝ) + (6^2 : ℝ) + (18^2 : ℝ) = 540 := by
  sorry

end sum_of_squared_coefficients_l2133_213373


namespace craig_and_mother_age_difference_l2133_213350

/-- Craig and his mother's ages problem -/
theorem craig_and_mother_age_difference :
  ∀ (craig_age mother_age : ℕ),
    craig_age + mother_age = 56 →
    craig_age = 16 →
    mother_age - craig_age = 24 := by
  sorry

end craig_and_mother_age_difference_l2133_213350


namespace relay_race_sequences_l2133_213315

/-- Represents the number of athletes in the relay race -/
def numAthletes : ℕ := 4

/-- Represents the set of all possible permutations of athletes -/
def allPermutations : ℕ := Nat.factorial numAthletes

/-- Represents the number of permutations where athlete A runs the first leg -/
def permutationsAFirst : ℕ := Nat.factorial (numAthletes - 1)

/-- Represents the number of permutations where athlete B runs the fourth leg -/
def permutationsBLast : ℕ := Nat.factorial (numAthletes - 1)

/-- Represents the number of permutations where A runs first and B runs last -/
def permutationsAFirstBLast : ℕ := Nat.factorial (numAthletes - 2)

/-- The theorem stating the number of valid sequences in the relay race -/
theorem relay_race_sequences :
  allPermutations - permutationsAFirst - permutationsBLast + permutationsAFirstBLast = 14 := by
  sorry


end relay_race_sequences_l2133_213315


namespace factorization_of_cubic_l2133_213336

theorem factorization_of_cubic (x : ℝ) : 2 * x^3 - 4 * x^2 - 6 * x = 2 * x * (x - 3) * (x + 1) := by
  sorry

end factorization_of_cubic_l2133_213336


namespace age_half_in_ten_years_l2133_213382

def mother_age : ℕ := 50

def person_age : ℕ := (2 * mother_age) / 5

def years_until_half (y : ℕ) : Prop :=
  2 * (person_age + y) = mother_age + y

theorem age_half_in_ten_years :
  ∃ y : ℕ, years_until_half y ∧ y = 10 := by sorry

end age_half_in_ten_years_l2133_213382


namespace exists_valid_coloring_l2133_213322

-- Define a coloring function type
def ColoringFunction := ℝ × ℝ → Bool

-- Define a property for a valid coloring
def ValidColoring (f : ColoringFunction) : Prop :=
  ∀ A B : ℝ × ℝ, A ≠ B →
    ∃ t : ℝ, 0 < t ∧ t < 1 ∧
      let C := (1 - t) • A + t • B
      f C ≠ f A ∨ f C ≠ f B

-- Theorem statement
theorem exists_valid_coloring : ∃ f : ColoringFunction, ValidColoring f := by
  sorry

end exists_valid_coloring_l2133_213322


namespace area_at_stage_8_l2133_213333

/-- The area of a rectangle formed by adding squares in an arithmetic sequence -/
def rectangleArea (squareSize : ℕ) (stages : ℕ) : ℕ :=
  stages * (squareSize * squareSize)

/-- Theorem: The area of a rectangle formed by adding 4" by 4" squares
    in an arithmetic sequence for 8 stages is 128 square inches -/
theorem area_at_stage_8 : rectangleArea 4 8 = 128 := by
  sorry

end area_at_stage_8_l2133_213333


namespace robot_tracing_time_l2133_213353

/-- Represents a rectangular grid with width and height -/
structure Grid where
  width : ℕ
  height : ℕ

/-- Calculates the total length of lines in a grid -/
def totalLength (g : Grid) : ℕ :=
  (g.width + 1) * g.height + (g.height + 1) * g.width

/-- Represents the robot's tracing speed in grid units per minute -/
def robotSpeed (g : Grid) (time : ℚ) : ℚ :=
  (totalLength g : ℚ) / time

theorem robot_tracing_time 
  (g1 g2 : Grid) 
  (t1 : ℚ) 
  (hg1 : g1 = ⟨3, 7⟩) 
  (hg2 : g2 = ⟨5, 5⟩) 
  (ht1 : t1 = 26) :
  robotSpeed g1 t1 * (totalLength g2 : ℚ) = 30 := by
  sorry

end robot_tracing_time_l2133_213353


namespace quadratic_unique_root_l2133_213302

/-- Given real numbers p, q, r forming an arithmetic sequence with p ≥ q ≥ r ≥ 0,
    if the quadratic px^2 + qx + r has exactly one root, then this root is equal to 1 - √6/2 -/
theorem quadratic_unique_root (p q r : ℝ) 
  (arith_seq : ∃ k, q = p - k ∧ r = p - 2*k)
  (order : p ≥ q ∧ q ≥ r ∧ r ≥ 0)
  (unique_root : ∃! x, p*x^2 + q*x + r = 0) :
  ∃ x, p*x^2 + q*x + r = 0 ∧ x = 1 - Real.sqrt 6 / 2 := by
  sorry

end quadratic_unique_root_l2133_213302


namespace johns_purchase_price_l2133_213329

/-- Calculate the final price after rebate and tax -/
def finalPrice (originalPrice rebatePercent taxPercent : ℚ) : ℚ :=
  let priceAfterRebate := originalPrice * (1 - rebatePercent / 100)
  let salesTax := priceAfterRebate * (taxPercent / 100)
  priceAfterRebate + salesTax

/-- Theorem stating the final price for John's purchase -/
theorem johns_purchase_price :
  finalPrice 6650 6 10 = 6876.1 :=
sorry

end johns_purchase_price_l2133_213329


namespace flute_cost_l2133_213347

/-- The cost of a flute given the total spent and costs of other items --/
theorem flute_cost (total_spent music_stand_cost song_book_cost : ℚ) :
  total_spent = 158.35 →
  music_stand_cost = 8.89 →
  song_book_cost = 7 →
  total_spent - (music_stand_cost + song_book_cost) = 142.46 := by
  sorry

end flute_cost_l2133_213347


namespace set_intersection_problem_l2133_213370

def M : Set ℝ := {x | 0 < x ∧ x < 8}
def N : Set ℝ := {x | ∃ n : ℕ, x = 2 * n + 1}

theorem set_intersection_problem : M ∩ N = {1, 3, 5, 7} := by sorry

end set_intersection_problem_l2133_213370


namespace card_distribution_correct_l2133_213399

/-- The total number of cards to be distributed -/
def total_cards : ℕ := 363

/-- The ratio of cards Xiaoming gets to Xiaohua's cards -/
def ratio_xiaoming_xiaohua : ℚ := 7 / 6

/-- The ratio of cards Xiaogang gets to Xiaoming's cards -/
def ratio_xiaogang_xiaoming : ℚ := 8 / 5

/-- The number of cards Xiaoming receives -/
def xiaoming_cards : ℕ := 105

/-- The number of cards Xiaohua receives -/
def xiaohua_cards : ℕ := 90

/-- The number of cards Xiaogang receives -/
def xiaogang_cards : ℕ := 168

theorem card_distribution_correct :
  (xiaoming_cards + xiaohua_cards + xiaogang_cards = total_cards) ∧
  (xiaoming_cards : ℚ) / xiaohua_cards = ratio_xiaoming_xiaohua ∧
  (xiaogang_cards : ℚ) / xiaoming_cards = ratio_xiaogang_xiaoming :=
by sorry

end card_distribution_correct_l2133_213399


namespace muffin_banana_cost_ratio_l2133_213359

/-- The cost ratio of muffins to bananas --/
def cost_ratio (muffin_cost banana_cost : ℚ) : ℚ := muffin_cost / banana_cost

/-- Susie's purchase --/
def susie_purchase (muffin_cost banana_cost : ℚ) : ℚ := 5 * muffin_cost + 4 * banana_cost

/-- Calvin's purchase --/
def calvin_purchase (muffin_cost banana_cost : ℚ) : ℚ := 3 * muffin_cost + 20 * banana_cost

theorem muffin_banana_cost_ratio :
  ∀ (muffin_cost banana_cost : ℚ),
    muffin_cost > 0 →
    banana_cost > 0 →
    calvin_purchase muffin_cost banana_cost = 3 * susie_purchase muffin_cost banana_cost →
    cost_ratio muffin_cost banana_cost = 2/3 := by
  sorry

end muffin_banana_cost_ratio_l2133_213359


namespace sum_of_squares_of_rates_l2133_213371

theorem sum_of_squares_of_rates : ∀ (c j s : ℕ),
  3 * c + 2 * j + 2 * s = 80 →
  3 * j + 2 * s + 4 * c = 104 →
  c^2 + j^2 + s^2 = 592 :=
by
  sorry

end sum_of_squares_of_rates_l2133_213371


namespace hyperbola_asymptote_angle_l2133_213340

/-- Proves that for a hyperbola with equation x²/a² - y²/b² = 1, where a > b, 
    if the angle between the asymptotes is 45°, then a/b = 1/(-1 + √2). -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.pi / 4 = Real.arctan ((2 * b / a) / (1 - (b / a)^2))) →
  a / b = 1 / (-1 + Real.sqrt 2) :=
by sorry

end hyperbola_asymptote_angle_l2133_213340


namespace children_who_got_off_bus_l2133_213342

/-- Proves that 10 children got off the bus given the initial, final, and additional children counts -/
theorem children_who_got_off_bus 
  (initial_children : ℕ) 
  (children_who_got_on : ℕ) 
  (final_children : ℕ) 
  (h1 : initial_children = 21)
  (h2 : children_who_got_on = 5)
  (h3 : final_children = 16) :
  initial_children - final_children + children_who_got_on = 10 :=
by sorry

end children_who_got_off_bus_l2133_213342


namespace trajectory_of_M_l2133_213376

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the slope sum condition
def slope_sum_condition (x y : ℝ) : Prop :=
  y / (x + 2) + y / (x - 2) = 2

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  x * y - x^2 + 4 = 0

-- Theorem statement
theorem trajectory_of_M (x y : ℝ) (h1 : y ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ -2) :
  slope_sum_condition x y → trajectory_equation x y :=
by
  sorry

end trajectory_of_M_l2133_213376


namespace at_most_two_distinct_values_l2133_213392

theorem at_most_two_distinct_values (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (sum_squares_eq : a^2 + b^2 = c^2 + d^2) : 
  ∃ (x y : ℝ), (a = x ∨ a = y) ∧ (b = x ∨ b = y) ∧ (c = x ∨ c = y) ∧ (d = x ∨ d = y) :=
by sorry

end at_most_two_distinct_values_l2133_213392


namespace sum_of_fractions_equals_one_l2133_213388

theorem sum_of_fractions_equals_one
  (a b c x y z : ℝ)
  (eq1 : 15 * x + b * y + c * z = 0)
  (eq2 : a * x + 25 * y + c * z = 0)
  (eq3 : a * x + b * y + 45 * z = 0)
  (ha : a ≠ 15)
  (hb : b ≠ 25)
  (hx : x ≠ 0) :
  a / (a - 15) + b / (b - 25) + c / (c - 45) = 1 :=
sorry

end sum_of_fractions_equals_one_l2133_213388


namespace nell_ace_cards_l2133_213380

/-- The number of baseball cards Nell has now -/
def baseball_cards : ℕ := 178

/-- The difference between baseball cards and Ace cards Nell has now -/
def difference : ℕ := 123

/-- Theorem: The number of Ace cards Nell has now is 55 -/
theorem nell_ace_cards : 
  ∃ (ace_cards : ℕ), ace_cards = baseball_cards - difference ∧ ace_cards = 55 := by
  sorry

end nell_ace_cards_l2133_213380


namespace polynomial_simplification_l2133_213307

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^5 - x^4 + 2 * x^3 + 5 * x^2 - 3 * x + 7) + 
  (-x^5 + 4 * x^4 + x^3 - 6 * x^2 + 5 * x - 4) - 
  (2 * x^5 - 3 * x^4 + 4 * x^3 - x^2 - x + 2) = 
  6 * x^4 - x^3 + 3 * x + 1 := by
sorry

end polynomial_simplification_l2133_213307


namespace min_max_values_l2133_213349

theorem min_max_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 1/x + 1/y ≤ 1/a + 1/b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → Real.sqrt x + Real.sqrt y ≥ Real.sqrt a + Real.sqrt b) :=
by sorry

end min_max_values_l2133_213349


namespace part_one_part_two_l2133_213313

-- Define the propositions p and q
def p (t a : ℝ) : Prop := t^2 - 5*a*t + 4*a^2 < 0

def q (t : ℝ) : Prop := ∃ (x y : ℝ), x^2/(t-2) + y^2/(t-6) = 1 ∧ (t-2)*(t-6) < 0

-- Part I
theorem part_one (t : ℝ) : p t 1 ∧ q t → 2 < t ∧ t < 4 := by sorry

-- Part II
theorem part_two (a : ℝ) : (∀ t : ℝ, q t → p t a) → 3/2 ≤ a ∧ a ≤ 2 := by sorry

end part_one_part_two_l2133_213313


namespace tangent_line_at_x_1_l2133_213304

/-- The function f(x) = x³ --/
def f (x : ℝ) : ℝ := x^3

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_line_at_x_1 :
  let m := f 1
  let slope := f' 1
  (fun x y => y - m = slope * (x - 1)) = (fun x y => y = 3 * x - 2) := by
    sorry

end tangent_line_at_x_1_l2133_213304


namespace parallel_vectors_t_value_l2133_213324

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_t_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, t)
  parallel a b → t = -4 := by sorry

end parallel_vectors_t_value_l2133_213324


namespace line_through_point_equation_line_with_slope_equation_l2133_213343

-- Define a line in 2D space
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to calculate the area of a triangle formed by a line and coordinate axes
def triangleArea (l : Line) : ℝ :=
  sorry

-- Function to check if a line passes through a point
def linePassesPoint (l : Line) (p : Point) : Prop :=
  sorry

-- Theorem for condition 1
theorem line_through_point_equation (l : Line) (A : Point) :
  triangleArea l = 3 ∧ linePassesPoint l A ∧ A.x = -3 ∧ A.y = 4 →
  (∃ a b c, a * l.slope + b = 0 ∧ a = 2 ∧ b = 3 ∧ c = -6) ∨
  (∃ a b c, a * l.slope + b = 0 ∧ a = 8 ∧ b = 3 ∧ c = 12) :=
sorry

-- Theorem for condition 2
theorem line_with_slope_equation (l : Line) :
  triangleArea l = 3 ∧ l.slope = 1/6 →
  (∃ b, l.intercept = b ∧ (b = 1 ∨ b = -1)) :=
sorry

end line_through_point_equation_line_with_slope_equation_l2133_213343


namespace exactly_one_prop_true_l2133_213344

-- Define a type for lines
structure Line where
  -- Add necessary fields for a line

-- Define what it means for two lines to form equal angles with a third line
def form_equal_angles (l1 l2 l3 : Line) : Prop := sorry

-- Define what it means for a line to be perpendicular to another line
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the three propositions
def prop1 : Prop := ∀ l1 l2 l3 : Line, form_equal_angles l1 l2 l3 → parallel l1 l2
def prop2 : Prop := ∀ l1 l2 l3 : Line, perpendicular l1 l3 ∧ perpendicular l2 l3 → parallel l1 l2
def prop3 : Prop := ∀ l1 l2 l3 : Line, parallel l1 l3 ∧ parallel l2 l3 → parallel l1 l2

-- Theorem stating that exactly one proposition is true
theorem exactly_one_prop_true : (prop1 ∧ ¬prop2 ∧ ¬prop3) ∨ (¬prop1 ∧ prop2 ∧ ¬prop3) ∨ (¬prop1 ∧ ¬prop2 ∧ prop3) :=
  sorry

end exactly_one_prop_true_l2133_213344


namespace abc_value_l2133_213300

theorem abc_value (a b c : ℂ) 
  (eq1 : 2 * a * b + 3 * b = -21)
  (eq2 : 2 * b * c + 3 * c = -21)
  (eq3 : 2 * c * a + 3 * a = -21) :
  a * b * c = 105.75 := by
sorry

end abc_value_l2133_213300


namespace every_nonzero_nat_is_product_of_primes_l2133_213356

theorem every_nonzero_nat_is_product_of_primes :
  ∀ n : ℕ, n > 0 → ∃ (primes : List ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ n = primes.prod := by
  sorry

end every_nonzero_nat_is_product_of_primes_l2133_213356


namespace factors_of_1320_l2133_213395

/-- The number of distinct, positive factors of 1320 -/
def num_factors_1320 : ℕ := sorry

/-- 1320 has exactly 24 distinct, positive factors -/
theorem factors_of_1320 : num_factors_1320 = 24 := by sorry

end factors_of_1320_l2133_213395


namespace sandy_fish_problem_l2133_213387

/-- The number of pet fish Sandy has after buying more -/
def sandys_final_fish_count (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem: Sandy's final fish count is 32 given the initial conditions -/
theorem sandy_fish_problem :
  sandys_final_fish_count 26 6 = 32 := by
  sorry

end sandy_fish_problem_l2133_213387


namespace alberts_cabbage_patch_l2133_213390

/-- Albert's cabbage patch problem -/
theorem alberts_cabbage_patch (rows : ℕ) (heads_per_row : ℕ) 
  (h1 : rows = 12) (h2 : heads_per_row = 15) : 
  rows * heads_per_row = 180 := by
  sorry

end alberts_cabbage_patch_l2133_213390


namespace function_properties_l2133_213363

def is_additive (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x + y) = f x + f y

theorem function_properties (f : ℝ → ℝ) 
  (h_additive : is_additive f)
  (h_neg : ∀ x > 0, f x < 0)
  (h_f1 : f 1 = -2) :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (f (-3) = 6 ∧ f 3 = -6) := by
  sorry

end function_properties_l2133_213363


namespace polygon_exterior_angles_l2133_213369

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) :
  (n > 2) →
  (exterior_angle > 0) →
  (exterior_angle < 180) →
  (n * exterior_angle = 360) →
  (exterior_angle = 60) →
  n = 6 := by
sorry

end polygon_exterior_angles_l2133_213369


namespace point_not_on_graph_l2133_213339

/-- A linear function passing through (1, 2) -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x

/-- The theorem stating that (1, -2) is not on the graph of the function -/
theorem point_not_on_graph (k : ℝ) (h1 : k ≠ 0) (h2 : f k 1 = 2) :
  f k 1 ≠ -2 := by
  sorry

end point_not_on_graph_l2133_213339


namespace solve_system_equations_solve_system_inequalities_l2133_213383

-- Part 1: System of Equations
theorem solve_system_equations :
  ∃! (x y : ℝ), 3 * x + 2 * y = 12 ∧ 2 * x - y = 1 ∧ x = 2 ∧ y = 3 := by sorry

-- Part 2: System of Inequalities
theorem solve_system_inequalities :
  ∀ x : ℝ, (x - 1 < 2 * x ∧ 2 * (x - 3) ≤ 3 - x) ↔ (-1 < x ∧ x ≤ 3) := by sorry

end solve_system_equations_solve_system_inequalities_l2133_213383


namespace no_zero_roots_l2133_213384

theorem no_zero_roots : 
  (∀ x : ℝ, 5 * x^2 - 3 = 47 → x ≠ 0) ∧ 
  (∀ x : ℝ, (3 * x + 2)^2 = (x + 2)^2 → x ≠ 0) ∧ 
  (∀ x : ℝ, (2 * x^2 - 6 : ℝ) = (2 * x - 2 : ℝ) → x ≠ 0) :=
by sorry

end no_zero_roots_l2133_213384


namespace triangle_side_length_l2133_213326

theorem triangle_side_length (a b c : ℝ) (A : Real) :
  A = π / 3 →  -- Angle A = 60°
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →  -- Area of triangle = √3
  b + c = 6 →  -- Given condition
  a = 2 * Real.sqrt 6 := by  -- Prove that a = 2√6
sorry

end triangle_side_length_l2133_213326


namespace angle_measure_l2133_213345

theorem angle_measure (C D : ℝ) : 
  C + D = 180 →  -- Angles C and D are supplementary
  C = 5 * D →    -- Measure of angle C is 5 times angle D
  C = 150 :=     -- Measure of angle C is 150 degrees
by sorry

end angle_measure_l2133_213345


namespace micah_envelope_count_l2133_213301

def envelope_count (total_stamps : ℕ) (light_envelopes : ℕ) (stamps_per_light : ℕ) (stamps_per_heavy : ℕ) : ℕ :=
  let heavy_stamps := total_stamps - light_envelopes * stamps_per_light
  let heavy_envelopes := heavy_stamps / stamps_per_heavy
  light_envelopes + heavy_envelopes

theorem micah_envelope_count :
  envelope_count 52 6 2 5 = 14 := by
  sorry

end micah_envelope_count_l2133_213301


namespace det_max_value_l2133_213381

open Real Matrix

theorem det_max_value (θ : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 1, 1; 1, 1 + sin θ, 1; 1, 1, 1 + cos θ]
  ∀ φ : ℝ, det A ≤ det (!![1, 1, 1; 1, 1 + sin φ, 1; 1, 1, 1 + cos φ]) → det A ≤ 1 :=
by sorry

end det_max_value_l2133_213381


namespace diag_diff_octagon_heptagon_l2133_213318

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Number of diagonals in a heptagon -/
def A : ℕ := num_diagonals 7

/-- Number of diagonals in an octagon -/
def B : ℕ := num_diagonals 8

/-- The difference between the number of diagonals in an octagon and a heptagon is 6 -/
theorem diag_diff_octagon_heptagon : B - A = 6 := by sorry

end diag_diff_octagon_heptagon_l2133_213318


namespace product_digit_sum_l2133_213378

/-- Represents a 101-digit number that repeats a 3-digit pattern -/
def RepeatingNumber (a b c : Nat) : Nat :=
  -- Implementation details omitted
  sorry

/-- Returns the units digit of a number -/
def unitsDigit (n : Nat) : Nat :=
  n % 10

/-- Returns the thousands digit of a number -/
def thousandsDigit (n : Nat) : Nat :=
  (n / 1000) % 10

/-- The main theorem -/
theorem product_digit_sum :
  let n1 := RepeatingNumber 6 0 6
  let n2 := RepeatingNumber 7 0 7
  let product := n1 * n2
  (thousandsDigit product) + (unitsDigit product) = 6 := by
  sorry

end product_digit_sum_l2133_213378


namespace binomial_square_constant_l2133_213354

theorem binomial_square_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) → a = 25 := by
  sorry

end binomial_square_constant_l2133_213354


namespace restaurant_expenditure_l2133_213367

theorem restaurant_expenditure (num_people : ℕ) (regular_cost : ℚ) (num_regular : ℕ) (extra_cost : ℚ) :
  num_people = 7 →
  regular_cost = 11 →
  num_regular = 6 →
  extra_cost = 6 →
  let total_regular := num_regular * regular_cost
  let average := (total_regular + (total_regular + extra_cost) / num_people) / num_people
  let total_cost := total_regular + (average + extra_cost)
  total_cost = 84 := by
  sorry

end restaurant_expenditure_l2133_213367


namespace coin_combination_difference_l2133_213368

/-- Represents the denominations of coins available -/
inductive Coin : Type
  | five : Coin
  | ten : Coin
  | twenty : Coin

/-- The value of a coin in cents -/
def coinValue : Coin → Nat
  | Coin.five => 5
  | Coin.ten => 10
  | Coin.twenty => 20

/-- A combination of coins -/
def CoinCombination := List Coin

/-- The total value of a coin combination in cents -/
def combinationValue (combo : CoinCombination) : Nat :=
  combo.map coinValue |>.sum

/-- Predicate for valid coin combinations that sum to 30 cents -/
def isValidCombination (combo : CoinCombination) : Prop :=
  combinationValue combo = 30

/-- The number of coins in a combination -/
def coinCount (combo : CoinCombination) : Nat :=
  combo.length

theorem coin_combination_difference :
  ∃ (minCombo maxCombo : CoinCombination),
    isValidCombination minCombo ∧
    isValidCombination maxCombo ∧
    (∀ c : CoinCombination, isValidCombination c → 
      coinCount c ≥ coinCount minCombo ∧
      coinCount c ≤ coinCount maxCombo) ∧
    coinCount maxCombo - coinCount minCombo = 4 := by
  sorry

end coin_combination_difference_l2133_213368


namespace jerry_average_increase_l2133_213357

theorem jerry_average_increase :
  let initial_average : ℝ := 85
  let fourth_test_score : ℝ := 97
  let total_tests : ℕ := 4
  let sum_first_three : ℝ := initial_average * 3
  let sum_all_four : ℝ := sum_first_three + fourth_test_score
  let new_average : ℝ := sum_all_four / total_tests
  new_average - initial_average = 3 := by sorry

end jerry_average_increase_l2133_213357


namespace c_profit_share_l2133_213310

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_profit_share (investment : ℕ) (total_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * total_profit) / total_investment

theorem c_profit_share :
  let a_investment := 12000
  let b_investment := 16000
  let c_investment := 20000
  let total_investment := a_investment + b_investment + c_investment
  let total_profit := 86400
  calculate_profit_share c_investment total_investment total_profit = 36000 := by
sorry

#eval calculate_profit_share 20000 (12000 + 16000 + 20000) 86400

end c_profit_share_l2133_213310


namespace exists_number_with_2001_trailing_zeros_l2133_213312

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of all divisors of a natural number -/
def productOfDivisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with 2001 trailing zeros in its product of divisors -/
theorem exists_number_with_2001_trailing_zeros : 
  ∃ n : ℕ, trailingZeros (productOfDivisors n) = 2001 := by sorry

end exists_number_with_2001_trailing_zeros_l2133_213312


namespace garden_perimeter_garden_perimeter_proof_l2133_213305

/-- The perimeter of a rectangular garden with length 100 m and breadth 200 m is 600 m. -/
theorem garden_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun length breadth perimeter =>
    length = 100 ∧ 
    breadth = 200 ∧ 
    perimeter = 2 * (length + breadth) →
    perimeter = 600

-- The proof is omitted
theorem garden_perimeter_proof : garden_perimeter 100 200 600 := by sorry

end garden_perimeter_garden_perimeter_proof_l2133_213305


namespace system_solution_l2133_213379

theorem system_solution (x y : ℝ) (eq1 : 2 * x + y = 7) (eq2 : x + 2 * y = 5) : x - y = 2 := by
  sorry

end system_solution_l2133_213379


namespace polynomial_remainder_l2133_213346

def g (a b x : ℚ) : ℚ := a * x^3 - 8 * x^2 + b * x - 7

theorem polynomial_remainder (a b : ℚ) :
  (g a b 2 = 1) ∧ (g a b (-3) = -89) → a = -10/3 ∧ b = 100/3 := by
  sorry

end polynomial_remainder_l2133_213346


namespace david_subtraction_l2133_213360

theorem david_subtraction (n : ℕ) (h : n = 40) : n^2 - 79 = (n - 1)^2 := by
  sorry

end david_subtraction_l2133_213360


namespace geometric_sequence_property_l2133_213351

/-- A sequence a : ℕ → ℝ is geometric if there exists a non-zero real number r 
    such that for all n, a(n+1) = r * a(n) -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : IsGeometric a) 
  (h2 : a 3 * a 5 = 64) : a 4 = 8 ∨ a 4 = -8 := by
  sorry

end geometric_sequence_property_l2133_213351


namespace sum_of_three_numbers_l2133_213338

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a*b + b*c + c*a = 72) : 
  a + b + c = 14 := by sorry

end sum_of_three_numbers_l2133_213338


namespace nickel_count_l2133_213321

/-- Proves that given $4 in quarters, dimes, and nickels, with 10 quarters and 12 dimes, the number of nickels is 6. -/
theorem nickel_count (total : ℚ) (quarters dimes : ℕ) : 
  total = 4 → 
  quarters = 10 → 
  dimes = 12 → 
  ∃ (nickels : ℕ), 
    total = (0.25 * quarters + 0.1 * dimes + 0.05 * nickels) ∧ 
    nickels = 6 := by sorry

end nickel_count_l2133_213321


namespace calculate_birth_rate_l2133_213358

/-- Given a death rate and population increase rate, calculate the birth rate. -/
theorem calculate_birth_rate (death_rate : ℝ) (population_increase_rate : ℝ) : 
  death_rate = 11 → population_increase_rate = 2.1 → 
  ∃ (birth_rate : ℝ), birth_rate = 32 ∧ birth_rate - death_rate = population_increase_rate / 100 * 1000 := by
  sorry

#check calculate_birth_rate

end calculate_birth_rate_l2133_213358


namespace bank_deposit_l2133_213328

theorem bank_deposit (P : ℝ) (interest_rate : ℝ) (years : ℕ) (final_amount : ℝ) : 
  interest_rate = 0.1 →
  years = 2 →
  final_amount = 121 →
  P * (1 + interest_rate) ^ years = final_amount →
  P = 100 := by
sorry

end bank_deposit_l2133_213328


namespace complex_power_result_l2133_213306

theorem complex_power_result : 
  (3 * (Complex.cos (Real.pi / 6) + Complex.I * Complex.sin (Real.pi / 6)))^8 = 
  Complex.mk (-3280.5) (-3280.5 * Real.sqrt 3) := by sorry

end complex_power_result_l2133_213306


namespace shortest_side_is_thirteen_l2133_213365

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The length of the first segment of the divided side -/
  segment1 : ℝ
  /-- The length of the second segment of the divided side -/
  segment2 : ℝ
  /-- The length of the shortest side of the triangle -/
  shortest_side : ℝ
  /-- Condition: radius is positive -/
  radius_pos : radius > 0
  /-- Condition: segments are positive -/
  segment1_pos : segment1 > 0
  segment2_pos : segment2 > 0
  /-- Condition: shortest side is positive -/
  shortest_side_pos : shortest_side > 0

/-- Theorem: The shortest side of the triangle is 13 units -/
theorem shortest_side_is_thirteen (t : TriangleWithInscribedCircle) 
    (h1 : t.radius = 4)
    (h2 : t.segment1 = 6)
    (h3 : t.segment2 = 8) :
    t.shortest_side = 13 :=
  sorry


end shortest_side_is_thirteen_l2133_213365


namespace star_operation_result_l2133_213391

def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem star_operation_result :
  let M : Set ℕ := {1, 2, 3, 4, 5}
  let P : Set ℕ := {2, 3, 6}
  star P M = {6} := by sorry

end star_operation_result_l2133_213391


namespace no_solution_sqrt_plus_one_l2133_213330

theorem no_solution_sqrt_plus_one :
  ∀ x : ℝ, ¬(Real.sqrt (x + 4) + 1 = 0) := by
  sorry

end no_solution_sqrt_plus_one_l2133_213330


namespace evaluate_expression_l2133_213317

theorem evaluate_expression (b c : ℕ) (hb : b = 2) (hc : c = 5) :
  b^3 * b^4 * c^2 = 3200 := by
  sorry

end evaluate_expression_l2133_213317


namespace common_divisor_sequence_l2133_213334

theorem common_divisor_sequence (n : ℕ) : n = 4190 →
  ∀ k ∈ Finset.range 21, ∃ d > 1, d ∣ (n + k) ∧ d ∣ 30030 := by
  sorry

#check common_divisor_sequence

end common_divisor_sequence_l2133_213334


namespace angle_decomposition_negative_495_decomposition_l2133_213348

theorem angle_decomposition (angle : ℤ) : ∃ (k : ℤ) (θ : ℤ), 
  angle = k * 360 + θ ∧ -180 < θ ∧ θ ≤ 180 :=
by sorry

theorem negative_495_decomposition : 
  ∃ (k : ℤ), -495 = k * 360 + (-135) ∧ -180 < -135 ∧ -135 ≤ 180 :=
by sorry

end angle_decomposition_negative_495_decomposition_l2133_213348


namespace cistern_length_l2133_213323

/-- Given a cistern with specified dimensions, calculate its length -/
theorem cistern_length (width : Real) (water_depth : Real) (wet_area : Real)
  (h1 : width = 4)
  (h2 : water_depth = 1.25)
  (h3 : wet_area = 49)
  : ∃ (length : Real), length = wet_area / (width + 2 * water_depth) :=
by
  sorry

#check cistern_length

end cistern_length_l2133_213323


namespace oranges_left_uneaten_l2133_213374

theorem oranges_left_uneaten (total : Nat) (ripe_fraction : Rat) (ripe_eaten_fraction : Rat) (unripe_eaten_fraction : Rat) :
  total = 96 →
  ripe_fraction = 1/2 →
  ripe_eaten_fraction = 1/4 →
  unripe_eaten_fraction = 1/8 →
  total - (ripe_fraction * total * ripe_eaten_fraction + (1 - ripe_fraction) * total * unripe_eaten_fraction) = 78 := by
  sorry

end oranges_left_uneaten_l2133_213374


namespace sum_of_squares_of_roots_sum_of_squares_of_specific_roots_l2133_213335

theorem sum_of_squares_of_roots (a b c : ℚ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x₁^2 + b * x₁ + c = 0 ∧ 
  a * x₂^2 + b * x₂ + c = 0 →
  x₁^2 + x₂^2 = (b^2 / a^2) + 2 * (c / a) :=
by sorry

theorem sum_of_squares_of_specific_roots :
  let a : ℚ := 5
  let b : ℚ := -3
  let c : ℚ := -11
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x₁^2 + b * x₁ + c = 0 ∧ 
  a * x₂^2 + b * x₂ + c = 0 →
  x₁^2 + x₂^2 = 119 / 25 :=
by sorry

end sum_of_squares_of_roots_sum_of_squares_of_specific_roots_l2133_213335


namespace function_transformation_l2133_213377

open Real

theorem function_transformation (x : ℝ) :
  let f (x : ℝ) := sin (2 * x + π / 3)
  let g (x : ℝ) := 2 * f (x - π / 6)
  g x = 2 * sin (2 * x) := by
  sorry

end function_transformation_l2133_213377


namespace quadratic_symmetry_implies_ordering_l2133_213337

-- Define the quadratic function f
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_symmetry_implies_ordering (b c : ℝ) :
  (∀ x : ℝ, f b c (1 + x) = f b c (1 - x)) →
  f b c 4 > f b c 2 ∧ f b c 2 > f b c 1 :=
by sorry

end quadratic_symmetry_implies_ordering_l2133_213337


namespace parabola_tangent_line_existence_l2133_213396

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the isosceles right triangle condition
def isosceles_right_triangle (F₁ F₂ F : ℝ × ℝ) : Prop :=
  (F₁.1 = -1 ∧ F₁.2 = 0) ∧ (F₂.1 = 1 ∧ F₂.2 = 0) ∧ (F.1 = 0 ∧ F.2 = 1)

-- Define the line passing through E(-2, 0)
def line_through_E (x y : ℝ) : Prop := y = (1/2) * (x + 2)

-- Define the perpendicular tangent lines condition
def perpendicular_tangents (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 = -4

-- Main theorem
theorem parabola_tangent_line_existence :
  ∃ (A B : ℝ × ℝ),
    parabola A.1 A.2 ∧
    parabola B.1 B.2 ∧
    line_through_E A.1 A.2 ∧
    line_through_E B.1 B.2 ∧
    perpendicular_tangents A B :=
  sorry

end parabola_tangent_line_existence_l2133_213396


namespace bullet_evaluation_l2133_213386

-- Define the bullet operation
def bullet (a b : ℤ) : ℤ := 10 * a - b

-- State the theorem
theorem bullet_evaluation :
  bullet (bullet (bullet 2 0) 1) 3 = 1987 := by
  sorry

end bullet_evaluation_l2133_213386


namespace total_frisbees_sold_l2133_213308

/-- Represents the number of frisbees sold at $3 -/
def x : ℕ := sorry

/-- Represents the number of frisbees sold at $4 -/
def y : ℕ := sorry

/-- The total receipts from frisbee sales is $200 -/
axiom total_sales : 3 * x + 4 * y = 200

/-- The fewest number of $4 frisbees sold is 8 -/
axiom min_four_dollar_frisbees : y ≥ 8

/-- The total number of frisbees sold -/
def total_frisbees : ℕ := x + y

theorem total_frisbees_sold : total_frisbees = 64 := by sorry

end total_frisbees_sold_l2133_213308


namespace dots_per_blouse_is_twenty_l2133_213372

/-- The number of dots on each blouse -/
def dots_per_blouse (total_dye : ℕ) (num_blouses : ℕ) (dye_per_dot : ℕ) : ℕ :=
  (total_dye / num_blouses) / dye_per_dot

/-- Theorem stating that the number of dots per blouse is 20 -/
theorem dots_per_blouse_is_twenty :
  dots_per_blouse (50 * 400) 100 10 = 20 := by
  sorry

end dots_per_blouse_is_twenty_l2133_213372


namespace hockey_league_games_l2133_213320

/-- Represents the number of games played between two groups of teams -/
def games_between (n m : ℕ) (games_per_pair : ℕ) : ℕ := n * m * games_per_pair

/-- Represents the number of games played within a group of teams -/
def games_within (n : ℕ) (games_per_pair : ℕ) : ℕ := n * (n - 1) * games_per_pair / 2

/-- The total number of games played in the hockey league season -/
def total_games : ℕ :=
  let top5 := 5
  let mid5 := 5
  let bottom5 := 5
  let top_vs_top := games_within top5 12
  let top_vs_rest := games_between top5 (mid5 + bottom5) 8
  let mid_vs_mid := games_within mid5 10
  let mid_vs_bottom := games_between mid5 bottom5 6
  let bottom_vs_bottom := games_within bottom5 8
  top_vs_top + top_vs_rest + mid_vs_mid + mid_vs_bottom + bottom_vs_bottom

theorem hockey_league_games :
  total_games = 850 := by sorry

end hockey_league_games_l2133_213320


namespace initial_money_calculation_l2133_213325

theorem initial_money_calculation (clothes_percent grocery_percent electronics_percent dining_percent : ℚ)
  (remaining_money : ℚ) :
  clothes_percent = 20 / 100 →
  grocery_percent = 15 / 100 →
  electronics_percent = 10 / 100 →
  dining_percent = 5 / 100 →
  remaining_money = 15700 →
  ∃ initial_money : ℚ, 
    initial_money * (1 - (clothes_percent + grocery_percent + electronics_percent + dining_percent)) = remaining_money ∧
    initial_money = 31400 :=
by
  sorry

end initial_money_calculation_l2133_213325


namespace three_numbers_sum_l2133_213355

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- a, b, c are in ascending order
  b = 10 ∧  -- median is 10
  (a + b + c) / 3 = a + 20 ∧  -- mean is 20 more than least
  (a + b + c) / 3 = c - 10  -- mean is 10 less than greatest
  → a + b + c = 0 := by
sorry

end three_numbers_sum_l2133_213355


namespace symmetry_probability_l2133_213366

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Represents a square with a grid of points -/
structure GridSquare where
  size : Nat
  points : List GridPoint

/-- Checks if a line through two points is a symmetry line for the square -/
def isSymmetryLine (square : GridSquare) (p q : GridPoint) : Bool :=
  sorry

/-- Counts the number of points that form symmetry lines with a given point -/
def countSymmetryPoints (square : GridSquare) (p : GridPoint) : Nat :=
  sorry

theorem symmetry_probability (square : GridSquare) (p : GridPoint) :
  square.size = 7 ∧
  square.points.length = 49 ∧
  p = ⟨3, 4⟩ →
  (countSymmetryPoints square p : Rat) / (square.points.length - 1 : Rat) = 1/4 := by
  sorry

end symmetry_probability_l2133_213366


namespace fixed_point_on_moving_line_intersecting_parabola_l2133_213303

/-- Theorem: Fixed point on a moving line intersecting a parabola -/
theorem fixed_point_on_moving_line_intersecting_parabola
  (p : ℝ) (k : ℝ) (b : ℝ)
  (hp : p > 0)
  (hk : k ≠ 0)
  (hb : b ≠ 0)
  (h_slope_product : ∀ x₁ y₁ x₂ y₂ : ℝ,
    y₁^2 = 2*p*x₁ → y₂^2 = 2*p*x₂ →
    y₁ = k*x₁ + b → y₂ = k*x₂ + b →
    (y₁ / x₁) * (y₂ / x₂) = Real.sqrt 3) :
  let fixed_point : ℝ × ℝ := (-2*p/Real.sqrt 3, 0)
  ∃ b' : ℝ, k * fixed_point.1 + b' = fixed_point.2 ∧ b' = 2*p*k/Real.sqrt 3 :=
by sorry


end fixed_point_on_moving_line_intersecting_parabola_l2133_213303


namespace largest_three_digit_divisible_by_6_l2133_213393

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def divisible_by (a b : ℕ) : Prop := b ∣ a

theorem largest_three_digit_divisible_by_6 :
  ∀ n : ℕ, is_three_digit n → divisible_by n 6 → n ≤ 996 :=
by sorry

end largest_three_digit_divisible_by_6_l2133_213393


namespace multiple_is_two_l2133_213311

/-- The multiple of Period 2 students compared to Period 1 students -/
def multiple_of_period2 (period1_students period2_students : ℕ) : ℚ :=
  (period1_students + 5) / period2_students

theorem multiple_is_two :
  let period1_students : ℕ := 11
  let period2_students : ℕ := 8
  multiple_of_period2 period1_students period2_students = 2 := by
  sorry

end multiple_is_two_l2133_213311


namespace max_weekly_profit_l2133_213394

/-- Represents the weekly sales profit as a function of the price increase -/
def weekly_profit (x : ℝ) : ℝ := -10 * x^2 + 100 * x + 6000

/-- Represents the number of items sold per week as a function of the price increase -/
def items_sold (x : ℝ) : ℝ := 300 - 10 * x

theorem max_weekly_profit :
  ∀ x : ℝ, x ≤ 20 → weekly_profit x ≤ 6250 ∧
  ∃ x₀ : ℝ, x₀ ≤ 20 ∧ weekly_profit x₀ = 6250 :=
sorry

end max_weekly_profit_l2133_213394


namespace sphere_volume_in_cube_l2133_213319

/-- The volume of a sphere inscribed in a cube with surface area 6 cm² is (1/6)π cm³ -/
theorem sphere_volume_in_cube (cube_surface_area : ℝ) (sphere_volume : ℝ) :
  cube_surface_area = 6 →
  sphere_volume = (1 / 6) * Real.pi :=
by
  sorry

end sphere_volume_in_cube_l2133_213319


namespace bus_capacity_is_198_l2133_213332

/-- Represents the capacity of a double-decker bus -/
def BusCapacity : ℕ :=
  let lower_left := 15 * 3
  let lower_right := (15 - 3) * 3
  let lower_back := 11
  let lower_standing := 12
  let upper_left := 20 * 2
  let upper_right_regular := (18 - 5) * 2
  let upper_right_reserved := 5 * 4
  let upper_standing := 8
  lower_left + lower_right + lower_back + lower_standing +
  upper_left + upper_right_regular + upper_right_reserved + upper_standing

/-- Theorem stating that the bus capacity is 198 people -/
theorem bus_capacity_is_198 : BusCapacity = 198 := by
  sorry

#eval BusCapacity

end bus_capacity_is_198_l2133_213332
