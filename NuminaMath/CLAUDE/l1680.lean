import Mathlib

namespace mike_weekly_pullups_l1680_168012

/-- The number of pull-ups Mike does in a week -/
def weekly_pullups (pullups_per_visit : ℕ) (visits_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  pullups_per_visit * visits_per_day * days_per_week

/-- Theorem stating that Mike does 70 pull-ups in a week -/
theorem mike_weekly_pullups :
  weekly_pullups 2 5 7 = 70 := by
  sorry

end mike_weekly_pullups_l1680_168012


namespace smaller_number_problem_l1680_168025

theorem smaller_number_problem (x y : ℤ) : 
  y = 2 * x - 3 →  -- One number is 3 less than twice another
  x + y = 39 →     -- The sum of the two numbers is 39
  x = 14           -- The smaller number is 14
  := by sorry

end smaller_number_problem_l1680_168025


namespace loop_structure_and_body_l1680_168071

/-- Represents an algorithmic structure -/
structure AlgorithmicStructure where
  repeatedExecution : Bool
  conditionalExecution : Bool

/-- Represents a processing step in an algorithm -/
structure ProcessingStep where
  isRepeated : Bool

/-- Definition of a loop structure -/
def isLoopStructure (s : AlgorithmicStructure) : Prop :=
  s.repeatedExecution ∧ s.conditionalExecution

/-- Definition of a loop body -/
def isLoopBody (p : ProcessingStep) : Prop :=
  p.isRepeated

/-- Theorem stating the relationship between loop structures and loop bodies -/
theorem loop_structure_and_body 
    (s : AlgorithmicStructure) 
    (p : ProcessingStep) 
    (h1 : s.repeatedExecution) 
    (h2 : s.conditionalExecution) 
    (h3 : p.isRepeated) : 
  isLoopStructure s ∧ isLoopBody p := by
  sorry


end loop_structure_and_body_l1680_168071


namespace sand_pouring_problem_l1680_168029

/-- Represents the fraction of sand remaining after n pourings -/
def remaining_sand (n : ℕ) : ℚ :=
  2 / (n + 2)

/-- The number of pourings required to reach exactly 1/5 of the original sand -/
def required_pourings : ℕ := 8

theorem sand_pouring_problem :
  remaining_sand required_pourings = 1/5 := by
  sorry

end sand_pouring_problem_l1680_168029


namespace savings_calculation_l1680_168013

def folder_price : ℝ := 2.50
def num_folders : ℕ := 5
def discount_rate : ℝ := 0.20

theorem savings_calculation :
  let original_total := folder_price * num_folders
  let discounted_total := original_total * (1 - discount_rate)
  original_total - discounted_total = 2.50 := by
sorry

end savings_calculation_l1680_168013


namespace sons_age_l1680_168087

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 33 := by
sorry

end sons_age_l1680_168087


namespace rectangle_decomposition_theorem_l1680_168041

/-- A function that checks if a rectangle can be decomposed into n and m+n congruent squares -/
def has_unique_decomposition (m : ℕ+) : Prop :=
  ∃! n : ℕ+, ∃ a b : ℕ+, a^2 - b^2 = n ∧ a^2 - b^2 = m + n

/-- A function that checks if a number is an odd prime -/
def is_odd_prime (p : ℕ+) : Prop :=
  Nat.Prime p.val ∧ p.val % 2 = 1

/-- The main theorem stating the equivalence of the two conditions -/
theorem rectangle_decomposition_theorem (m : ℕ+) :
  has_unique_decomposition m ↔ 
  (∃ p : ℕ+, is_odd_prime p ∧ (m = p ∨ m = 2 * p ∨ m = 4 * p)) :=
sorry

end rectangle_decomposition_theorem_l1680_168041


namespace quadratic_always_positive_l1680_168009

theorem quadratic_always_positive (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 > 0) ↔ -2 < k ∧ k < 2 := by sorry

end quadratic_always_positive_l1680_168009


namespace final_pen_count_l1680_168047

def pen_collection (initial : ℕ) (mike_gives : ℕ) (cindy_multiplier : ℕ) (sharon_takes : ℕ) : ℕ :=
  ((initial + mike_gives) * cindy_multiplier) - sharon_takes

theorem final_pen_count : pen_collection 5 20 2 10 = 40 := by
  sorry

end final_pen_count_l1680_168047


namespace max_fleas_on_board_l1680_168094

/-- Represents a 10x10 board --/
def Board := Fin 10 → Fin 10 → Bool

/-- Represents the four possible directions of flea movement --/
inductive Direction
| Up
| Down
| Left
| Right

/-- Represents a flea's position and direction --/
structure Flea where
  pos : Fin 10 × Fin 10
  dir : Direction

/-- Represents the state of the board and fleas at a given time --/
structure BoardState where
  board : Board
  fleas : List Flea

/-- Simulates the movement of fleas for one hour (60 minutes) --/
def simulateMovement (initialState : BoardState) : BoardState :=
  sorry

/-- Checks if the simulation results in a valid state (no overlapping fleas) --/
def isValidSimulation (finalState : BoardState) : Bool :=
  sorry

/-- Theorem stating the maximum number of fleas --/
theorem max_fleas_on_board :
  ∀ (initialState : BoardState),
    isValidSimulation (simulateMovement initialState) →
    initialState.fleas.length ≤ 40 :=
  sorry

end max_fleas_on_board_l1680_168094


namespace geometric_sequence_property_l1680_168054

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_condition : a 1 * a 13 + 2 * (a 7)^2 = 5 * Real.pi) : 
  Real.cos (a 2 * a 12) = 1/2 := by
  sorry

end geometric_sequence_property_l1680_168054


namespace arithmetic_geometric_mean_squares_l1680_168022

theorem arithmetic_geometric_mean_squares (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 20)
  (h_geometric : Real.sqrt (a * b) = 10) : 
  a^2 + b^2 = 1400 := by
sorry

end arithmetic_geometric_mean_squares_l1680_168022


namespace vincent_sticker_packs_l1680_168021

theorem vincent_sticker_packs (yesterday_packs today_extra_packs : ℕ) :
  yesterday_packs = 15 →
  today_extra_packs = 10 →
  yesterday_packs + (yesterday_packs + today_extra_packs) = 40 := by
  sorry

end vincent_sticker_packs_l1680_168021


namespace reflected_polygon_area_equal_l1680_168095

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a polygon with n vertices -/
structure Polygon (n : ℕ) where
  vertices : Fin n → Point

/-- Calculates the area of a polygon -/
def area (p : Polygon n) : ℝ := sorry

/-- Reflects a point across the midpoint of two other points -/
def reflect (p : Point) (a : Point) (b : Point) : Point := sorry

/-- Creates a new polygon by reflecting each vertex of the given polygon
    across the midpoint of the corresponding side of the regular 2009-gon -/
def reflectedPolygon (p : Polygon 2009) (regularPolygon : Polygon 2009) : Polygon 2009 := sorry

/-- Theorem stating that the area of the reflected polygon is equal to the area of the original polygon -/
theorem reflected_polygon_area_equal (p : Polygon 2009) (regularPolygon : Polygon 2009) :
  area (reflectedPolygon p regularPolygon) = area p := by sorry

end reflected_polygon_area_equal_l1680_168095


namespace green_peaches_count_l1680_168098

theorem green_peaches_count (red_peaches : ℕ) (green_peaches : ℕ) 
  (h1 : red_peaches = 17)
  (h2 : red_peaches = green_peaches + 1) : 
  green_peaches = 16 := by
  sorry

end green_peaches_count_l1680_168098


namespace football_throw_distance_l1680_168051

theorem football_throw_distance (parker_distance grant_distance kyle_distance : ℝ) :
  parker_distance = 16 ∧
  grant_distance = parker_distance * 1.25 ∧
  kyle_distance = grant_distance * 2 →
  kyle_distance - parker_distance = 24 := by
  sorry

end football_throw_distance_l1680_168051


namespace height_comparison_l1680_168038

theorem height_comparison (a b : ℝ) (h : a = b * (1 - 0.25)) :
  b = a * (1 + 1/3) :=
by sorry

end height_comparison_l1680_168038


namespace equation_solution_l1680_168092

theorem equation_solution (a b x : ℝ) (h : b ≠ 0) :
  a * (Real.cos (x / 2))^2 - (a + 2 * b) * (Real.sin (x / 2))^2 = a * Real.cos x - b * Real.sin x ↔
  (∃ n : ℤ, x = 2 * n * Real.pi) ∨ (∃ k : ℤ, x = Real.pi / 2 * (4 * k + 1)) :=
sorry

end equation_solution_l1680_168092


namespace field_dimension_solution_l1680_168067

/-- Represents the dimensions of a rectangular field -/
structure FieldDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular field -/
def fieldArea (d : FieldDimensions) : ℝ := d.length * d.width

/-- Theorem: For a rectangular field with dimensions (3m + 4) and (m - 3),
    if the area is 80 square units, then m = 19/3 -/
theorem field_dimension_solution (m : ℝ) :
  let d := FieldDimensions.mk (3 * m + 4) (m - 3)
  fieldArea d = 80 → m = 19/3 := by
  sorry


end field_dimension_solution_l1680_168067


namespace degree_of_product_l1680_168032

-- Define polynomials h and j
variable (h j : Polynomial ℝ)

-- Define the degrees of h and j
variable (deg_h : Polynomial.degree h = 3)
variable (deg_j : Polynomial.degree j = 5)

-- Theorem statement
theorem degree_of_product :
  Polynomial.degree (h.comp (Polynomial.X ^ 4) * j.comp (Polynomial.X ^ 3)) = 27 := by
  sorry

end degree_of_product_l1680_168032


namespace sum_18_probability_l1680_168093

/-- The number of ways to distribute 10 points among 8 dice -/
def ways_to_distribute : ℕ := 19448

/-- The total number of possible outcomes when throwing 8 dice -/
def total_outcomes : ℕ := 6^8

/-- The probability of obtaining a sum of 18 when throwing 8 fair 6-sided dice -/
def probability_sum_18 : ℚ := ways_to_distribute / total_outcomes

theorem sum_18_probability :
  probability_sum_18 = 19448 / 6^8 :=
sorry

end sum_18_probability_l1680_168093


namespace problem_statement_l1680_168066

theorem problem_statement (x y z : ℝ) 
  (sum_eq : x + y + z = 12) 
  (sum_sq_eq : x^2 + y^2 + z^2 = 54) : 
  (9 ≤ x*y ∧ x*y ≤ 25) ∧ 
  (9 ≤ y*z ∧ y*z ≤ 25) ∧ 
  (9 ≤ z*x ∧ z*x ≤ 25) ∧
  ((x ≤ 3 ∧ (y ≥ 5 ∨ z ≥ 5)) ∨ 
   (y ≤ 3 ∧ (x ≥ 5 ∨ z ≥ 5)) ∨ 
   (z ≤ 3 ∧ (x ≥ 5 ∨ y ≥ 5))) := by
sorry

end problem_statement_l1680_168066


namespace colbert_treehouse_ratio_l1680_168023

/-- Proves that the ratio of planks from Colbert's parents to the total number of planks is 1:2 -/
theorem colbert_treehouse_ratio :
  let total_planks : ℕ := 200
  let storage_planks : ℕ := total_planks / 4
  let friends_planks : ℕ := 20
  let store_planks : ℕ := 30
  let parents_planks : ℕ := total_planks - (storage_planks + friends_planks + store_planks)
  (parents_planks : ℚ) / total_planks = 1 / 2 := by
  sorry

end colbert_treehouse_ratio_l1680_168023


namespace fuel_usage_proof_l1680_168056

theorem fuel_usage_proof (x : ℝ) : 
  x > 0 ∧ x + 0.8 * x = 27 → x = 15 := by
  sorry

end fuel_usage_proof_l1680_168056


namespace marble_selection_ways_l1680_168062

theorem marble_selection_ways (total_marbles : ℕ) (special_marbles : ℕ) (selection_size : ℕ) 
  (h1 : total_marbles = 15)
  (h2 : special_marbles = 6)
  (h3 : selection_size = 5) :
  (special_marbles : ℕ) * Nat.choose (total_marbles - special_marbles) (selection_size - 1) = 756 := by
  sorry

end marble_selection_ways_l1680_168062


namespace leading_coefficient_of_p_l1680_168046

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := -2*(x^5 - x^4 + 2*x^3) + 6*(x^5 + x^2 - 1) - 5*(3*x^5 + x^3 + 4)

/-- The leading coefficient of a polynomial -/
def leadingCoefficient (p : ℝ → ℝ) : ℝ :=
  sorry  -- Definition of leading coefficient

theorem leading_coefficient_of_p :
  leadingCoefficient p = -11 := by
  sorry

end leading_coefficient_of_p_l1680_168046


namespace range_of_a_l1680_168019

def p (x : ℝ) := |4*x - 3| ≤ 1

def q (x a : ℝ) := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

theorem range_of_a :
  (∀ x, q x a → p x) ∧
  (∃ x, p x ∧ ¬q x a) →
  a ∈ Set.Icc (0 : ℝ) (1/2) :=
sorry

end range_of_a_l1680_168019


namespace six_people_round_table_one_reserved_l1680_168043

/-- The number of ways to arrange people around a round table --/
def roundTableArrangements (n : ℕ) (reserved : ℕ) : ℕ :=
  Nat.factorial (n - reserved)

/-- Theorem: 6 people around a round table with 1 reserved seat --/
theorem six_people_round_table_one_reserved :
  roundTableArrangements 6 1 = 120 := by
  sorry

end six_people_round_table_one_reserved_l1680_168043


namespace negation_existence_real_l1680_168070

theorem negation_existence_real : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end negation_existence_real_l1680_168070


namespace prob_ace_then_king_standard_deck_l1680_168069

/-- A standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ace_count : ℕ)
  (king_count : ℕ)

/-- The probability of drawing an Ace then a King from a standard deck -/
def prob_ace_then_king (d : Deck) : ℚ :=
  (d.ace_count : ℚ) / d.total_cards * (d.king_count : ℚ) / (d.total_cards - 1)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52
  , ace_count := 4
  , king_count := 4 }

/-- Theorem: The probability of drawing an Ace then a King from a standard 52-card deck is 4/663 -/
theorem prob_ace_then_king_standard_deck :
  prob_ace_then_king standard_deck = 4 / 663 := by
  sorry

end prob_ace_then_king_standard_deck_l1680_168069


namespace smallest_four_digit_divisible_by_six_l1680_168096

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

def last_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

theorem smallest_four_digit_divisible_by_six :
  ∀ n : ℕ, is_four_digit n →
    (is_divisible_by n 6 → n ≥ 1002) ∧
    (is_divisible_by 1002 6) ∧
    is_four_digit 1002 :=
sorry

end smallest_four_digit_divisible_by_six_l1680_168096


namespace sum_zero_iff_fractions_sum_neg_two_l1680_168003

theorem sum_zero_iff_fractions_sum_neg_two (x y : ℝ) (h : x * y ≠ 0) :
  x + y = 0 ↔ x / y + y / x = -2 := by
  sorry

end sum_zero_iff_fractions_sum_neg_two_l1680_168003


namespace min_colors_is_23_l1680_168004

/-- Represents a coloring arrangement for 8 boxes with 6 balls each -/
structure ColorArrangement where
  n : ℕ  -- Number of colors
  boxes : Fin 8 → Finset (Fin n)
  all_boxes_size_six : ∀ i, (boxes i).card = 6
  no_duplicate_colors : ∀ i j, i ≠ j → (boxes i ∩ boxes j).card ≤ 1

/-- The minimum number of colors needed for a valid ColorArrangement -/
def min_colors : ℕ := 23

/-- Theorem stating that 23 is the minimum number of colors needed -/
theorem min_colors_is_23 :
  (∃ arrangement : ColorArrangement, arrangement.n = min_colors) ∧
  (∀ arrangement : ColorArrangement, arrangement.n ≥ min_colors) :=
sorry

end min_colors_is_23_l1680_168004


namespace odd_function_value_and_range_and_inequality_l1680_168072

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - 4 / (2 * a^x + a)

theorem odd_function_value_and_range_and_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, f a x = -f a (-x)) ∧
  (∀ x, -1 < (2^x - 1) / (2^x + 1) ∧ (2^x - 1) / (2^x + 1) < 1) ∧
  (∀ x ∈ Set.Ioo 0 1, ∃ t ≥ 0, ∀ s ≥ t, s * ((2^x - 1) / (2^x + 1)) ≥ 2^x - 2) :=
by sorry

end odd_function_value_and_range_and_inequality_l1680_168072


namespace minimize_sum_of_distances_l1680_168042

/-- Given points A, B, and C in a 2D plane, where:
    A has coordinates (4, 6)
    B has coordinates (3, 0)
    C has coordinates (k, 0)
    This theorem states that the value of k that minimizes
    the sum of distances AC + BC is 3. -/
theorem minimize_sum_of_distances :
  let A : ℝ × ℝ := (4, 6)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ → ℝ × ℝ := λ k => (k, 0)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let total_distance (k : ℝ) : ℝ := distance A (C k) + distance B (C k)
  ∃ k₀ : ℝ, k₀ = 3 ∧ ∀ k : ℝ, total_distance k₀ ≤ total_distance k :=
by sorry

end minimize_sum_of_distances_l1680_168042


namespace wrapping_paper_area_l1680_168026

/-- The area of wrapping paper needed to cover a rectangular box with a small cube on top -/
theorem wrapping_paper_area (w h : ℝ) (w_pos : 0 < w) (h_pos : 0 < h) :
  let box_width := 2 * w
  let box_length := w
  let box_height := h
  let cube_side := w / 2
  let total_height := box_height + cube_side
  let paper_width := box_width + 2 * total_height
  let paper_length := box_length + 2 * total_height
  paper_width * paper_length = (3 * w + 2 * h) * (2 * w + 2 * h) :=
by sorry


end wrapping_paper_area_l1680_168026


namespace baking_powder_difference_l1680_168064

-- Define the constants
def yesterday_supply : Real := 1.5 -- in kg
def today_supply : Real := 1.2 -- in kg (converted from 1200 grams)
def box_size : Real := 5 -- kg per box

-- Define the theorem
theorem baking_powder_difference :
  yesterday_supply - today_supply = 0.3 := by
  sorry

end baking_powder_difference_l1680_168064


namespace poultry_farm_daily_loss_l1680_168033

/-- Calculates the daily loss of guinea fowls in a poultry farm scenario --/
theorem poultry_farm_daily_loss (initial_chickens initial_turkeys initial_guinea_fowls : ℕ)
  (daily_chicken_loss daily_turkey_loss : ℕ) (total_birds_after_week : ℕ) :
  initial_chickens = 300 →
  initial_turkeys = 200 →
  initial_guinea_fowls = 80 →
  daily_chicken_loss = 20 →
  daily_turkey_loss = 8 →
  total_birds_after_week = 349 →
  ∃ (daily_guinea_fowl_loss : ℕ),
    daily_guinea_fowl_loss = 5 ∧
    total_birds_after_week = 
      initial_chickens - 7 * daily_chicken_loss +
      initial_turkeys - 7 * daily_turkey_loss +
      initial_guinea_fowls - 7 * daily_guinea_fowl_loss :=
by
  sorry


end poultry_farm_daily_loss_l1680_168033


namespace smallest_multiplier_for_perfect_square_l1680_168005

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_multiplier_for_perfect_square : 
  (∀ k : ℕ, k > 0 ∧ k < 7 → ¬ is_perfect_square (1008 * k)) ∧ 
  is_perfect_square (1008 * 7) := by
  sorry

#check smallest_multiplier_for_perfect_square

end smallest_multiplier_for_perfect_square_l1680_168005


namespace prob_one_white_one_black_l1680_168049

/-- The probability of drawing one white ball and one black ball in two draws -/
theorem prob_one_white_one_black (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : total_balls > 0)
  (h3 : white_balls = 7)
  (h4 : black_balls = 3) :
  (white_balls : ℚ) / total_balls * (black_balls : ℚ) / total_balls + 
  (black_balls : ℚ) / total_balls * (white_balls : ℚ) / total_balls = 
  (7 : ℚ) / 10 * (3 : ℚ) / 10 + (3 : ℚ) / 10 * (7 : ℚ) / 10 :=
sorry

end prob_one_white_one_black_l1680_168049


namespace one_and_half_times_product_of_digits_l1680_168007

/-- Function to calculate the product of digits of a natural number -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 48 and 0 are the only natural numbers that are 1.5 times the product of their digits -/
theorem one_and_half_times_product_of_digits :
  ∀ (A : ℕ), A = (3 / 2 : ℚ) * (productOfDigits A) ↔ A = 48 ∨ A = 0 := by sorry

end one_and_half_times_product_of_digits_l1680_168007


namespace ellipse_perpendicular_bisector_x_intersection_l1680_168058

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def perpendicular_bisector_intersects_x_axis (A B P : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ), (P.2 = 0) ∧ 
  (P.1 - (A.1 + B.1)/2) = m * ((A.2 + B.2)/2) ∧
  (B.2 - A.2) * (P.1 - (A.1 + B.1)/2) = (A.1 - B.1) * ((A.2 + B.2)/2)

theorem ellipse_perpendicular_bisector_x_intersection
  (a b : ℝ) (h_ab : a > b ∧ b > 0) (A B P : ℝ × ℝ) :
  ellipse a b A.1 A.2 →
  ellipse a b B.1 B.2 →
  perpendicular_bisector_intersects_x_axis A B P →
  -((a^2 - b^2)/a) < P.1 ∧ P.1 < (a^2 - b^2)/a :=
by sorry

end ellipse_perpendicular_bisector_x_intersection_l1680_168058


namespace laser_reflection_theorem_l1680_168080

/-- Regular hexagon with side length 2 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- Point G on BC where the laser beam hits -/
def G (h : RegularHexagon) : ℝ × ℝ := sorry

/-- Midpoint of DE -/
def M (h : RegularHexagon) : ℝ × ℝ := sorry

/-- Length of BG -/
def BG_length (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating that BG length is 2/5 -/
theorem laser_reflection_theorem (h : RegularHexagon) :
  let g := G h
  let m := M h
  (∃ (t : ℝ), t • (g.1 - h.A.1, g.2 - h.A.2) = (m.1 - g.1, m.2 - g.2)) →
  BG_length h = 2/5 := by sorry

end laser_reflection_theorem_l1680_168080


namespace point_not_on_transformed_plane_l1680_168006

/-- The similarity transformation coefficient -/
def k : ℚ := 5/2

/-- The original plane equation: x + y - 2z + 2 = 0 -/
def plane_a (x y z : ℚ) : Prop := x + y - 2*z + 2 = 0

/-- The transformed plane equation: x + y - 2z + 5 = 0 -/
def plane_a_transformed (x y z : ℚ) : Prop := x + y - 2*z + 5 = 0

/-- Point A -/
def point_A : ℚ × ℚ × ℚ := (2, -3, 1)

/-- Theorem: Point A does not belong to the image of plane a after similarity transformation -/
theorem point_not_on_transformed_plane :
  ¬ plane_a_transformed point_A.1 point_A.2.1 point_A.2.2 :=
by sorry

end point_not_on_transformed_plane_l1680_168006


namespace earl_owes_fred_l1680_168057

/-- Represents the financial state of Earl, Fred, and Greg -/
structure FinancialState where
  earl : Int
  fred : Int
  greg : Int

/-- Calculates the final financial state after debts are paid -/
def finalState (initial : FinancialState) (earlOwes : Int) : FinancialState :=
  { earl := initial.earl - earlOwes + 40,
    fred := initial.fred + earlOwes - 32,
    greg := initial.greg + 32 - 40 }

/-- The theorem to be proved -/
theorem earl_owes_fred (initial : FinancialState) :
  initial.earl = 90 →
  initial.fred = 48 →
  initial.greg = 36 →
  (let final := finalState initial 28
   final.earl + final.greg = 130) :=
by sorry

end earl_owes_fred_l1680_168057


namespace function_symmetry_and_periodicity_l1680_168086

/-- A function f: ℝ → ℝ is symmetric about the line x = a if f(2a - x) = f(x) for all x ∈ ℝ -/
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

/-- A function f: ℝ → ℝ is periodic with period p if f(x + p) = f(x) for all x ∈ ℝ -/
def Periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

/-- A function f: ℝ → ℝ is symmetric about the point (a, b) if f(2a - x) = 2b - f(x) for all x ∈ ℝ -/
def SymmetricAboutPoint (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = 2 * b - f x

theorem function_symmetry_and_periodicity (f : ℝ → ℝ) :
  (∀ x, f (2 - x) = f x) → SymmetricAboutLine f 1 ∧
  (∀ x, f (x - 1) = f (x + 1)) → Periodic f 2 ∧
  (∀ x, f (2 - x) = -f x) → SymmetricAboutPoint f 1 0 := by
  sorry


end function_symmetry_and_periodicity_l1680_168086


namespace point_on_unit_circle_l1680_168001

theorem point_on_unit_circle (t : ℝ) :
  let x := (t^3 - 1) / (t^3 + 1)
  let y := (2*t^3) / (t^3 + 1)
  x^2 + y^2 = 1 := by
sorry

end point_on_unit_circle_l1680_168001


namespace ten_thousand_one_divides_repeat_digit_number_l1680_168060

/-- An 8-digit positive integer with the first four digits repeated -/
def RepeatDigitNumber (a b c d : Nat) : Nat :=
  a * 10000000 + b * 1000000 + c * 100000 + d * 10000 +
  a * 1000 + b * 100 + c * 10 + d

/-- Theorem: 10001 is a factor of any 8-digit number with repeated first four digits -/
theorem ten_thousand_one_divides_repeat_digit_number 
  (a b c d : Nat) (ha : a > 0) (hb : b < 10) (hc : c < 10) (hd : d < 10) :
  10001 ∣ RepeatDigitNumber a b c d := by
  sorry

end ten_thousand_one_divides_repeat_digit_number_l1680_168060


namespace tangent_line_length_l1680_168059

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 9 = 0

-- Define the point P
def P : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_line_length :
  ∃ (t : ℝ × ℝ), 
    circle_equation t.1 t.2 ∧ 
    (t.1 - P.1)^2 + (t.2 - P.2)^2 = 8 :=
by sorry

end tangent_line_length_l1680_168059


namespace car_push_distance_l1680_168075

/-- Proves that the total distance traveled is 10 miles given the conditions of the problem --/
theorem car_push_distance : 
  let segment1_distance : ℝ := 3
  let segment1_speed : ℝ := 6
  let segment2_distance : ℝ := 3
  let segment2_speed : ℝ := 3
  let segment3_distance : ℝ := 4
  let segment3_speed : ℝ := 8
  let total_time : ℝ := 2
  segment1_distance / segment1_speed + 
  segment2_distance / segment2_speed + 
  segment3_distance / segment3_speed = total_time →
  segment1_distance + segment2_distance + segment3_distance = 10 :=
by
  sorry


end car_push_distance_l1680_168075


namespace corn_growth_ratio_l1680_168010

theorem corn_growth_ratio :
  ∀ (growth_week1 growth_week2 growth_week3 total_height : ℝ),
    growth_week1 = 2 →
    growth_week2 = 2 * growth_week1 →
    total_height = 22 →
    total_height = growth_week1 + growth_week2 + growth_week3 →
    growth_week3 / growth_week2 = 4 := by
  sorry

end corn_growth_ratio_l1680_168010


namespace solution_set_of_equation_l1680_168084

theorem solution_set_of_equation (x : ℝ) : 
  (Real.sin (2 * x) - π * Real.sin x) * Real.sqrt (11 * x^2 - x^4 - 10) = 0 ↔ 
  x ∈ ({-Real.sqrt 10, -π, -1, 1, π, Real.sqrt 10} : Set ℝ) := by
sorry

end solution_set_of_equation_l1680_168084


namespace integer_in_3_rows_and_3_cols_l1680_168082

/-- Represents a 21x21 array of integers -/
def Array21x21 := Fin 21 → Fin 21 → Int

/-- Predicate to check if a row has at most 6 different integers -/
def row_at_most_6_different (arr : Array21x21) (row : Fin 21) : Prop :=
  (Finset.univ.image (fun col => arr row col)).card ≤ 6

/-- Predicate to check if a column has at most 6 different integers -/
def col_at_most_6_different (arr : Array21x21) (col : Fin 21) : Prop :=
  (Finset.univ.image (fun row => arr row col)).card ≤ 6

/-- Predicate to check if an integer appears in at least 3 rows -/
def in_at_least_3_rows (arr : Array21x21) (n : Int) : Prop :=
  (Finset.univ.filter (fun row => ∃ col, arr row col = n)).card ≥ 3

/-- Predicate to check if an integer appears in at least 3 columns -/
def in_at_least_3_cols (arr : Array21x21) (n : Int) : Prop :=
  (Finset.univ.filter (fun col => ∃ row, arr row col = n)).card ≥ 3

theorem integer_in_3_rows_and_3_cols (arr : Array21x21) 
  (h_rows : ∀ row, row_at_most_6_different arr row)
  (h_cols : ∀ col, col_at_most_6_different arr col) :
  ∃ n : Int, in_at_least_3_rows arr n ∧ in_at_least_3_cols arr n := by
  sorry

end integer_in_3_rows_and_3_cols_l1680_168082


namespace voldemort_lunch_calories_l1680_168083

/-- Calculates the calories consumed for lunch given the daily calorie limit,
    calories from dinner items, breakfast, and remaining calories. -/
def calories_for_lunch (daily_limit : ℕ) (cake : ℕ) (chips : ℕ) (coke : ℕ)
                       (breakfast : ℕ) (remaining : ℕ) : ℕ :=
  daily_limit - (cake + chips + coke + breakfast + remaining)

/-- Proves that Voldemort consumed 780 calories for lunch. -/
theorem voldemort_lunch_calories :
  calories_for_lunch 2500 110 310 215 560 525 = 780 := by
  sorry

end voldemort_lunch_calories_l1680_168083


namespace fourth_root_equation_solutions_l1680_168020

theorem fourth_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (x ^ (1/4 : ℝ)) - 15 / (8 - x ^ (1/4 : ℝ))
  {x : ℝ | f x = 0} = {625, 81} := by
sorry

end fourth_root_equation_solutions_l1680_168020


namespace intersection_is_empty_l1680_168078

-- Define set A
def A : Set ℝ := {x | x^2 + 4 ≤ 5*x}

-- Define set B
def B : Set (ℝ × ℝ) := {p | p.2 = 3^p.1 + 2}

-- Theorem statement
theorem intersection_is_empty : A ∩ (B.image Prod.fst) = ∅ := by
  sorry

end intersection_is_empty_l1680_168078


namespace furniture_shop_cost_price_l1680_168053

theorem furniture_shop_cost_price (markup_percentage : ℝ) (selling_price : ℝ) (cost_price : ℝ) : 
  markup_percentage = 20 →
  selling_price = 3600 →
  selling_price = cost_price * (1 + markup_percentage / 100) →
  cost_price = 3000 := by
  sorry

end furniture_shop_cost_price_l1680_168053


namespace missing_sale_is_7225_l1680_168091

/-- Calculates the missing month's sale given the sales of other months and the target average --/
def calculate_missing_sale (sale1 sale2 sale3 sale5 sale6 target_average : ℕ) : ℕ :=
  6 * target_average - (sale1 + sale2 + sale3 + sale5 + sale6)

/-- Proves that the missing month's sale is 7225 given the problem conditions --/
theorem missing_sale_is_7225 :
  let sale1 : ℕ := 6235
  let sale2 : ℕ := 6927
  let sale3 : ℕ := 6855
  let sale5 : ℕ := 6562
  let sale6 : ℕ := 5191
  let target_average : ℕ := 6500
  calculate_missing_sale sale1 sale2 sale3 sale5 sale6 target_average = 7225 :=
by
  sorry

end missing_sale_is_7225_l1680_168091


namespace system_solution_characterization_l1680_168039

/-- The system of equations has either a unique solution or infinitely many solutions when m ≠ -1 -/
theorem system_solution_characterization (m : ℝ) (hm : m ≠ -1) :
  (∃! x y : ℝ, m * x + y = m + 1 ∧ x + m * y = 2 * m) ∨
  (∃ f g : ℝ → ℝ, ∀ t : ℝ, m * (f t) + (g t) = m + 1 ∧ (f t) + m * (g t) = 2 * m) :=
by sorry

end system_solution_characterization_l1680_168039


namespace inequality_solution_set_l1680_168030

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo 2 3 : Set ℝ) = {x | (x - 2) * (x - 3) / (x^2 + 1) < 0} := by
  sorry

end inequality_solution_set_l1680_168030


namespace fraction_simplification_l1680_168074

theorem fraction_simplification (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / y + 1 / x) / (1 / x) = 9 / 5 := by
  sorry

end fraction_simplification_l1680_168074


namespace journey_time_comparison_l1680_168000

/-- Represents the speed of walking -/
def walking_speed : ℝ := 1

/-- Represents the speed of cycling -/
def cycling_speed : ℝ := 2 * walking_speed

/-- Represents the speed of the bus -/
def bus_speed : ℝ := 5 * cycling_speed

/-- Represents half the total journey distance -/
def half_journey : ℝ := 1

theorem journey_time_comparison : 
  (half_journey / bus_speed + half_journey / walking_speed) > (2 * half_journey) / cycling_speed :=
sorry

end journey_time_comparison_l1680_168000


namespace sum_of_largest_and_smallest_prime_factors_of_1320_l1680_168073

theorem sum_of_largest_and_smallest_prime_factors_of_1320 :
  ∃ (smallest largest : Nat),
    smallest.Prime ∧
    largest.Prime ∧
    smallest ∣ 1320 ∧
    largest ∣ 1320 ∧
    (∀ p : Nat, p.Prime → p ∣ 1320 → p ≥ smallest) ∧
    (∀ p : Nat, p.Prime → p ∣ 1320 → p ≤ largest) ∧
    smallest + largest = 13 := by
  sorry

end sum_of_largest_and_smallest_prime_factors_of_1320_l1680_168073


namespace max_abs_z_value_l1680_168015

theorem max_abs_z_value (a b c z : ℂ) 
  (h1 : Complex.abs a = Complex.abs b) 
  (h2 : Complex.abs a = 2 * Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : 2 * a * z^2 + b * z + c * z = 0) : 
  Complex.abs z ≤ 3/4 := by
  sorry

end max_abs_z_value_l1680_168015


namespace inequality_solution_range_l1680_168035

theorem inequality_solution_range (k : ℝ) : 
  (1 : ℝ)^2 * k^2 - 6 * k * (1 : ℝ) + 8 ≥ 0 → k ≤ 2 ∨ k ≥ 4 :=
by sorry

end inequality_solution_range_l1680_168035


namespace rectangle_with_hole_area_l1680_168052

theorem rectangle_with_hole_area (x : ℝ) : 
  let large_length : ℝ := 2*x + 8
  let large_width : ℝ := x + 6
  let hole_length : ℝ := 3*x - 4
  let hole_width : ℝ := x + 1
  large_length * large_width - hole_length * hole_width = -x^2 + 22*x + 52 :=
by sorry

end rectangle_with_hole_area_l1680_168052


namespace distance_to_other_focus_l1680_168008

/-- The distance from a point on an ellipse to the other focus -/
theorem distance_to_other_focus (x y : ℝ) :
  x^2 / 9 + y^2 / 4 = 1 →  -- P is on the ellipse
  ∃ (f₁ f₂ : ℝ × ℝ),  -- existence of two foci
    (∀ (p : ℝ × ℝ), x^2 / 9 + y^2 / 4 = 1 →
      Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
      Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 6) →  -- definition of ellipse
    Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) = 1 →  -- distance to one focus is 1
    Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2) = 5  -- distance to other focus is 5
    := by sorry

end distance_to_other_focus_l1680_168008


namespace loan_income_is_135_l1680_168099

/-- Calculates the yearly annual income from two parts of a loan at different interest rates -/
def yearly_income (total : ℚ) (part1 : ℚ) (rate1 : ℚ) (rate2 : ℚ) : ℚ :=
  let part2 := total - part1
  part1 * rate1 + part2 * rate2

/-- Theorem stating that the yearly income from the given loan parts is 135 -/
theorem loan_income_is_135 :
  yearly_income 2500 1500 (5/100) (6/100) = 135 := by
  sorry

end loan_income_is_135_l1680_168099


namespace function_symmetry_origin_l1680_168028

/-- The function f(x) = x^3 + x is symmetric with respect to the origin. -/
theorem function_symmetry_origin (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + x
  f (-x) = -f x := by
  sorry

end function_symmetry_origin_l1680_168028


namespace cubic_roots_sum_of_cubes_l1680_168045

theorem cubic_roots_sum_of_cubes (p q s r₁ r₂ : ℝ) : 
  (∀ x, x^3 - p*x^2 + q*x - s = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = 0) →
  r₁^3 + r₂^3 = p^3 - 3*q*p :=
sorry

end cubic_roots_sum_of_cubes_l1680_168045


namespace usual_walking_time_l1680_168016

theorem usual_walking_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 → usual_time > 0 →
  (4 / 5 * usual_speed) * (usual_time + 10) = usual_speed * usual_time →
  usual_time = 40 := by
sorry

end usual_walking_time_l1680_168016


namespace smallest_perimeter_600_smallest_perimeter_144_l1680_168017

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The perimeter of a triangle -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- The product of side lengths of a triangle -/
def product (t : IntTriangle) : ℕ := t.a * t.b * t.c

theorem smallest_perimeter_600 :
  ∀ t : IntTriangle, product t = 600 →
  perimeter t ≥ perimeter ⟨10, 10, 6, sorry⟩ := by sorry

theorem smallest_perimeter_144 :
  ∀ t : IntTriangle, product t = 144 →
  perimeter t ≥ perimeter ⟨4, 6, 6, sorry⟩ := by sorry

end smallest_perimeter_600_smallest_perimeter_144_l1680_168017


namespace basketball_team_callbacks_l1680_168050

theorem basketball_team_callbacks (girls_tryout : ℕ) (boys_tryout : ℕ) (didnt_make_cut : ℕ) :
  girls_tryout = 9 →
  boys_tryout = 14 →
  didnt_make_cut = 21 →
  girls_tryout + boys_tryout - didnt_make_cut = 2 :=
by
  sorry

end basketball_team_callbacks_l1680_168050


namespace cubic_equation_root_l1680_168079

theorem cubic_equation_root : ∃ x : ℝ, x^3 + 6*x^2 + 12*x + 35 = 0 :=
  by
    use -5
    -- Proof goes here
    sorry

end cubic_equation_root_l1680_168079


namespace line_through_two_points_l1680_168036

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The line equation derived from two points -/
def lineEquation (p₁ p₂ : Point) (p : Point) : Prop :=
  (p.x - p₁.x) * (p₂.y - p₁.y) = (p.y - p₁.y) * (p₂.x - p₁.x)

theorem line_through_two_points (p₁ p₂ : Point) (h : p₁ ≠ p₂) :
  ∃! l : Line, Point.onLine p₁ l ∧ Point.onLine p₂ l ∧
  ∀ p, Point.onLine p l ↔ lineEquation p₁ p₂ p :=
sorry

end line_through_two_points_l1680_168036


namespace cos_ninety_degrees_l1680_168011

theorem cos_ninety_degrees : Real.cos (π / 2) = 0 := by
  sorry

end cos_ninety_degrees_l1680_168011


namespace y1_greater_than_y2_l1680_168048

/-- Given a linear function y = 8x - 1 and two points P₁(3, y₁) and P₂(2, y₂) on its graph,
    prove that y₁ > y₂. -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : y₁ > y₂ :=
  by
  -- Define the linear function
  have h1 : ∀ x y, y = 8 * x - 1 → (x, y) ∈ {(x, y) | y = 8 * x - 1} := by sorry
  
  -- P₁(3, y₁) lies on the graph
  have h2 : (3, y₁) ∈ {(x, y) | y = 8 * x - 1} := by sorry
  
  -- P₂(2, y₂) lies on the graph
  have h3 : (2, y₂) ∈ {(x, y) | y = 8 * x - 1} := by sorry
  
  sorry -- Proof goes here

end y1_greater_than_y2_l1680_168048


namespace daves_hourly_wage_l1680_168014

/-- Dave's hourly wage calculation --/
theorem daves_hourly_wage (monday_hours tuesday_hours total_amount : ℕ) 
  (h1 : monday_hours = 6)
  (h2 : tuesday_hours = 2)
  (h3 : total_amount = 48) :
  total_amount / (monday_hours + tuesday_hours) = 6 := by
  sorry

end daves_hourly_wage_l1680_168014


namespace total_fault_movement_total_movement_is_17_25_l1680_168031

/-- Represents the movement of a fault line over two years -/
structure FaultMovement where
  pastYear : Float
  yearBefore : Float

/-- Calculates the total movement of a fault line over two years -/
def totalMovement (fault : FaultMovement) : Float :=
  fault.pastYear + fault.yearBefore

/-- Theorem: The total movement of all fault lines is the sum of their individual movements -/
theorem total_fault_movement (faultA faultB faultC : FaultMovement) :
  totalMovement faultA + totalMovement faultB + totalMovement faultC =
  faultA.pastYear + faultA.yearBefore +
  faultB.pastYear + faultB.yearBefore +
  faultC.pastYear + faultC.yearBefore := by
  sorry

/-- Given fault movements -/
def faultA : FaultMovement := { pastYear := 1.25, yearBefore := 5.25 }
def faultB : FaultMovement := { pastYear := 2.5, yearBefore := 3.0 }
def faultC : FaultMovement := { pastYear := 0.75, yearBefore := 4.5 }

/-- Theorem: The total movement of the given fault lines is 17.25 inches -/
theorem total_movement_is_17_25 :
  totalMovement faultA + totalMovement faultB + totalMovement faultC = 17.25 := by
  sorry

end total_fault_movement_total_movement_is_17_25_l1680_168031


namespace word_game_possible_l1680_168085

structure WordDistribution where
  anya_only : ℕ
  borya_only : ℕ
  vasya_only : ℕ
  anya_borya : ℕ
  anya_vasya : ℕ
  borya_vasya : ℕ

def total_words (d : WordDistribution) : ℕ :=
  d.anya_only + d.borya_only + d.vasya_only + d.anya_borya + d.anya_vasya + d.borya_vasya

def anya_words (d : WordDistribution) : ℕ :=
  d.anya_only + d.anya_borya + d.anya_vasya

def borya_words (d : WordDistribution) : ℕ :=
  d.borya_only + d.anya_borya + d.borya_vasya

def vasya_words (d : WordDistribution) : ℕ :=
  d.vasya_only + d.anya_vasya + d.borya_vasya

def anya_score (d : WordDistribution) : ℕ :=
  2 * d.anya_only + d.anya_borya + d.anya_vasya

def borya_score (d : WordDistribution) : ℕ :=
  2 * d.borya_only + d.anya_borya + d.borya_vasya

def vasya_score (d : WordDistribution) : ℕ :=
  2 * d.vasya_only + d.anya_vasya + d.borya_vasya

theorem word_game_possible : ∃ d : WordDistribution,
  anya_words d > borya_words d ∧
  borya_words d > vasya_words d ∧
  vasya_score d > borya_score d ∧
  borya_score d > anya_score d :=
sorry

end word_game_possible_l1680_168085


namespace equation_solution_l1680_168076

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 3
def g (x y : ℝ) : ℝ := 3 * x + y

-- State the theorem
theorem equation_solution (x y : ℝ) :
  2 * (f x) - 11 + g x y = f (x - 2) ↔ y = -5 * x + 10 := by
  sorry

end equation_solution_l1680_168076


namespace right_triangle_perimeter_l1680_168077

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 30 →
  a^2 + b^2 = c^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 := by
sorry

end right_triangle_perimeter_l1680_168077


namespace absolute_value_inequality_l1680_168002

theorem absolute_value_inequality (x : ℝ) : 
  |x^2 - 5*x + 6| < x^2 - 4 ↔ x > 2 := by sorry

end absolute_value_inequality_l1680_168002


namespace smallest_prime_factor_of_1739_l1680_168088

theorem smallest_prime_factor_of_1739 : Nat.Prime 1739 := by
  sorry

end smallest_prime_factor_of_1739_l1680_168088


namespace log_equation_solution_l1680_168090

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  (Real.log x^3 / Real.log 3) + (Real.log x / Real.log (1/3)) = 8 → x = 81 := by
  sorry

end log_equation_solution_l1680_168090


namespace angle_CDE_value_l1680_168018

-- Define the points
variable (A B C D E : Point)

-- Define the angles
variable (angleA angleB angleC angleAEB angleBED angleAED angleADE angleCDE : Real)

-- State the given conditions
axiom right_angles : angleA = 90 ∧ angleB = 90 ∧ angleC = 90
axiom angle_AEB : angleAEB = 50
axiom angle_BED : angleBED = 45
axiom isosceles_ADE : angleAED = angleADE

-- State the theorem to be proved
theorem angle_CDE_value : angleCDE = 112.5 := by
  sorry

end angle_CDE_value_l1680_168018


namespace data_transformation_theorem_l1680_168044

/-- Represents a set of numerical data -/
structure DataSet where
  values : List ℝ

/-- Calculates the average of a DataSet -/
def average (d : DataSet) : ℝ := sorry

/-- Calculates the variance of a DataSet -/
def variance (d : DataSet) : ℝ := sorry

/-- Transforms a DataSet by subtracting a constant from each value -/
def transform (d : DataSet) (c : ℝ) : DataSet := sorry

theorem data_transformation_theorem (original : DataSet) :
  let transformed := transform original 80
  average transformed = 1.2 →
  variance transformed = 4.4 →
  average original = 81.2 ∧ variance original = 4.4 := by sorry

end data_transformation_theorem_l1680_168044


namespace units_digit_of_4_pow_10_l1680_168040

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The statement that the units digit of 4^10 is 6 -/
theorem units_digit_of_4_pow_10 : unitsDigit (4^10) = 6 := by sorry

end units_digit_of_4_pow_10_l1680_168040


namespace lloyds_hourly_rate_l1680_168027

def regular_hours : ℝ := 7.5
def overtime_rate : ℝ := 1.5
def total_hours : ℝ := 10.5
def total_earnings : ℝ := 66

def hourly_rate : ℝ := 5.5

theorem lloyds_hourly_rate : 
  regular_hours * hourly_rate + 
  (total_hours - regular_hours) * overtime_rate * hourly_rate = 
  total_earnings := by sorry

end lloyds_hourly_rate_l1680_168027


namespace divisibility_equivalence_l1680_168063

theorem divisibility_equivalence (n : ℕ) : 
  7 ∣ (3^n + n^3) ↔ 7 ∣ (3^n * n^3 + 1) := by
  sorry

end divisibility_equivalence_l1680_168063


namespace series_sum_l1680_168068

theorem series_sum : 
  let a : ℕ → ℚ := fun n => (4*n + 3) / ((4*n + 1)^2 * (4*n + 5)^2)
  ∑' n, a n = 1/200 := by
  sorry

end series_sum_l1680_168068


namespace five_topping_pizzas_l1680_168037

theorem five_topping_pizzas (n : Nat) (k : Nat) (h1 : n = 8) (h2 : k = 5) :
  Nat.choose n k = 56 := by
  sorry

end five_topping_pizzas_l1680_168037


namespace min_value_w_l1680_168024

theorem min_value_w (x y : ℝ) : 
  3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 30 ≥ 20.25 ∧ 
  ∃ (a b : ℝ), 3 * a^2 + 3 * b^2 + 9 * a - 6 * b + 30 = 20.25 :=
by sorry

end min_value_w_l1680_168024


namespace alien_year_conversion_l1680_168061

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The problem statement --/
theorem alien_year_conversion :
  base8ToBase10 [2, 6, 3] = 242 := by
  sorry

end alien_year_conversion_l1680_168061


namespace regular_hexagon_vector_relation_l1680_168089

-- Define a regular hexagon
structure RegularHexagon (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D E F : V)
  (is_regular : sorry)  -- This would typically include conditions that define a regular hexagon

-- Theorem statement
theorem regular_hexagon_vector_relation 
  {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (hex : RegularHexagon V) 
  (a b : V) 
  (h1 : hex.B - hex.A = a) 
  (h2 : hex.E - hex.A = b) : 
  hex.C - hex.B = (1/2 : ℝ) • a + (1/2 : ℝ) • b := by
  sorry

end regular_hexagon_vector_relation_l1680_168089


namespace perpendicular_lines_a_value_perpendicular_lines_a_value_proof_l1680_168097

/-- Given two lines that are perpendicular, find the value of 'a' -/
theorem perpendicular_lines_a_value : ℝ → Prop :=
  fun a => 
    let line1 := fun x y : ℝ => 3 * y - x + 4 = 0
    let line2 := fun x y : ℝ => 4 * y + a * x + 5 = 0
    let slope1 := (1 : ℝ) / 3
    let slope2 := -a / 4
    (∀ x y : ℝ, line1 x y ∧ line2 x y → slope1 * slope2 = -1) →
    a = 12

/-- Proof of the theorem -/
theorem perpendicular_lines_a_value_proof : perpendicular_lines_a_value 12 := by
  sorry

end perpendicular_lines_a_value_perpendicular_lines_a_value_proof_l1680_168097


namespace area_of_inscribed_rectangle_l1680_168055

/-- Rectangle ABCD inscribed in triangle EFG with the following properties:
    - Side AD of the rectangle is on side EG of the triangle
    - Triangle's altitude from F to side EG is 7 inches
    - EG = 10 inches
    - Length of segment AB is equal to half the length of segment AD -/
structure InscribedRectangle where
  EG : ℝ
  altitude : ℝ
  AB : ℝ
  AD : ℝ
  h_EG : EG = 10
  h_altitude : altitude = 7
  h_AB_AD : AB = AD / 2

/-- The area of the inscribed rectangle ABCD is 1225/72 square inches -/
theorem area_of_inscribed_rectangle (rect : InscribedRectangle) :
  rect.AB * rect.AD = 1225 / 72 := by
  sorry

end area_of_inscribed_rectangle_l1680_168055


namespace barbed_wire_rate_l1680_168065

/-- Given a square field with area 3136 sq m and a total cost of 932.40 Rs for drawing barbed wire
    around it, leaving two 1 m wide gates, the rate of drawing barbed wire per meter is 4.2 Rs/m. -/
theorem barbed_wire_rate (area : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  area = 3136 →
  total_cost = 932.40 →
  gate_width = 1 →
  num_gates = 2 →
  (total_cost / (4 * Real.sqrt area - num_gates * gate_width) : ℝ) = 4.2 := by
  sorry

end barbed_wire_rate_l1680_168065


namespace line_slope_intercept_product_l1680_168081

/-- Given a line with slope m and y-intercept b, prove that their product mb equals -6 -/
theorem line_slope_intercept_product :
  ∀ (m b : ℝ), m = 2 → b = -3 → m * b = -6 := by
  sorry

end line_slope_intercept_product_l1680_168081


namespace factorial_fraction_l1680_168034

theorem factorial_fraction (N : ℕ) (h : N > 2) :
  (Nat.factorial (N - 2) * (N - 1)) / Nat.factorial N = 1 / N := by
  sorry

end factorial_fraction_l1680_168034
