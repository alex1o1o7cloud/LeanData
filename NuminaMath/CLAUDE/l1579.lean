import Mathlib

namespace sufficient_but_not_necessary_condition_l1579_157968

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (|x| < 1 → x < a) ∧ ¬(x < a → |x| < 1)) →
  a ≥ 1 :=
by sorry

end sufficient_but_not_necessary_condition_l1579_157968


namespace f_intersects_negative_axes_l1579_157956

def f (x : ℝ) : ℝ := -x - 1

theorem f_intersects_negative_axes :
  (∃ x, x < 0 ∧ f x = 0) ∧ (∃ y, y < 0 ∧ f 0 = y) := by
  sorry

end f_intersects_negative_axes_l1579_157956


namespace solve_fish_problem_l1579_157959

def fish_problem (current_fish : ℕ) (added_fish : ℕ) (caught_fish : ℕ) : Prop :=
  let original_fish := current_fish - added_fish
  (caught_fish < original_fish) ∧ (original_fish - caught_fish = 4)

theorem solve_fish_problem :
  ∃ (caught_fish : ℕ), fish_problem 20 8 caught_fish :=
by sorry

end solve_fish_problem_l1579_157959


namespace concert_tickets_cost_l1579_157935

def total_cost (adult_tickets child_tickets adult_price child_price adult_discount child_discount total_discount : ℚ) : ℚ :=
  let adult_cost := adult_tickets * adult_price * (1 - adult_discount)
  let child_cost := child_tickets * child_price * (1 - child_discount)
  let subtotal := adult_cost + child_cost
  subtotal * (1 - total_discount)

theorem concert_tickets_cost :
  total_cost 12 12 10 5 0.4 0.3 0.1 = 102.6 := by
  sorry

end concert_tickets_cost_l1579_157935


namespace salad_total_calories_l1579_157957

/-- Represents the total calories in a salad. -/
def saladCalories (lettuce_cal : ℕ) (cucumber_cal : ℕ) (crouton_count : ℕ) (crouton_cal : ℕ) : ℕ :=
  lettuce_cal + cucumber_cal + crouton_count * crouton_cal

/-- Proves that the total calories in the salad is 350. -/
theorem salad_total_calories :
  saladCalories 30 80 12 20 = 350 := by
  sorry

end salad_total_calories_l1579_157957


namespace fraction_equality_implies_division_l1579_157954

theorem fraction_equality_implies_division (A B C : ℕ) : 
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C →
  1 - 1 / (6 + 1 / (6 + 1 / 6)) = 1 / (A + 1 / (B + 1 / C)) →
  (A + B) / C = 1 := by
sorry

end fraction_equality_implies_division_l1579_157954


namespace arithmetic_sequence_sum_l1579_157941

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 2 + a 3 = 32) →
  (a 11 + a 12 + a 13 = 118) →
  a 4 + a 10 = 50 := by
  sorry

end arithmetic_sequence_sum_l1579_157941


namespace bill_red_mushrooms_l1579_157917

/-- Proves that Bill gathered 12 red mushrooms based on the given conditions --/
theorem bill_red_mushrooms :
  ∀ (red_mushrooms : ℕ) 
    (brown_mushrooms : ℕ)
    (blue_mushrooms : ℕ)
    (green_mushrooms : ℕ)
    (white_spotted_mushrooms : ℕ),
  brown_mushrooms = 6 →
  blue_mushrooms = 6 →
  green_mushrooms = 14 →
  white_spotted_mushrooms = 17 →
  (blue_mushrooms / 2 : ℚ) + brown_mushrooms + (2 * red_mushrooms / 3 : ℚ) = white_spotted_mushrooms →
  red_mushrooms = 12 := by
sorry

end bill_red_mushrooms_l1579_157917


namespace distance_walked_l1579_157998

/-- Given a walking speed and a total walking time, calculate the distance walked. -/
theorem distance_walked (speed : ℝ) (time : ℝ) (h1 : speed = 1 / 15) (h2 : time = 45) :
  speed * time = 3 := by
  sorry

end distance_walked_l1579_157998


namespace find_a_l1579_157910

theorem find_a : ∀ a : ℚ, 
  (∀ y : ℚ, (y + a) / 2 = (2 * y - a) / 3 → y = 5 * a) →
  (∀ x : ℚ, 3 * a - x = x / 2 + 3 → x = 2 * a - 2) →
  (5 * a = (2 * a - 2) - 3) →
  a = -5 / 3 :=
by sorry

end find_a_l1579_157910


namespace opposite_of_negative_fraction_l1579_157989

theorem opposite_of_negative_fraction :
  let x : ℚ := -4/5
  let opposite (y : ℚ) : ℚ := -y
  opposite x = 4/5 := by
  sorry

end opposite_of_negative_fraction_l1579_157989


namespace rational_solutions_quadratic_l1579_157915

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + k = 0) ↔ k = 12 := by
  sorry

end rational_solutions_quadratic_l1579_157915


namespace skipping_rope_price_solution_l1579_157948

def skipping_rope_prices (price_A price_B : ℚ) : Prop :=
  (price_B = price_A + 10) ∧
  (3150 / price_A = 3900 / price_B) ∧
  (price_A = 42) ∧
  (price_B = 52)

theorem skipping_rope_price_solution :
  ∃ (price_A price_B : ℚ), skipping_rope_prices price_A price_B :=
sorry

end skipping_rope_price_solution_l1579_157948


namespace number_wall_theorem_l1579_157902

/-- Represents a number wall with 5 numbers in the bottom row -/
structure NumberWall :=
  (bottom : Fin 5 → ℕ)
  (second_row : Fin 4 → ℕ)
  (third_row : Fin 3 → ℕ)
  (fourth_row : Fin 2 → ℕ)
  (top : ℕ)

/-- The rule for constructing a number wall -/
def valid_wall (w : NumberWall) : Prop :=
  (∀ i : Fin 4, w.second_row i = w.bottom i + w.bottom (i + 1)) ∧
  (∀ i : Fin 3, w.third_row i = w.second_row i + w.second_row (i + 1)) ∧
  (∀ i : Fin 2, w.fourth_row i = w.third_row i + w.third_row (i + 1)) ∧
  (w.top = w.fourth_row 0 + w.fourth_row 1)

theorem number_wall_theorem (w : NumberWall) (h : valid_wall w) :
  w.bottom 1 = 5 ∧ w.bottom 2 = 9 ∧ w.bottom 3 = 7 ∧ w.bottom 4 = 12 ∧
  w.top = 54 ∧ w.third_row 1 = 34 →
  w.bottom 0 = 1 := by
  sorry

end number_wall_theorem_l1579_157902


namespace family_spent_38_dollars_l1579_157901

def regular_ticket_price : ℝ := 5
def popcorn_price : ℝ := 0.8 * regular_ticket_price
def ticket_discount_rate : ℝ := 0.1
def soda_discount_rate : ℝ := 0.5
def num_tickets : ℕ := 4
def num_popcorn : ℕ := 2
def num_sodas : ℕ := 4

def discounted_ticket_price : ℝ := regular_ticket_price * (1 - ticket_discount_rate)
def soda_price : ℝ := popcorn_price  -- Assuming soda price is the same as popcorn price
def discounted_soda_price : ℝ := soda_price * (1 - soda_discount_rate)

theorem family_spent_38_dollars :
  let total_ticket_cost := num_tickets * discounted_ticket_price
  let total_popcorn_cost := num_popcorn * popcorn_price
  let total_soda_cost := num_popcorn * discounted_soda_price + (num_sodas - num_popcorn) * soda_price
  total_ticket_cost + total_popcorn_cost + total_soda_cost = 38 := by
  sorry

end family_spent_38_dollars_l1579_157901


namespace complex_equation_sum_l1579_157993

theorem complex_equation_sum (x y : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : x - 3 * i = (8 * x - y) * i) : x + y = 3 := by
  sorry

end complex_equation_sum_l1579_157993


namespace carpenter_tables_total_l1579_157953

/-- The number of tables made this month -/
def tables_this_month : ℕ := 10

/-- The difference in tables made between this month and last month -/
def difference : ℕ := 3

/-- The number of tables made last month -/
def tables_last_month : ℕ := tables_this_month - difference

/-- The total number of tables made over two months -/
def total_tables : ℕ := tables_this_month + tables_last_month

theorem carpenter_tables_total :
  total_tables = 17 := by sorry

end carpenter_tables_total_l1579_157953


namespace subsets_of_B_l1579_157922

def B : Set ℕ := {0, 1, 2}

theorem subsets_of_B :
  {A : Set ℕ | A ⊆ B} =
  {∅, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, B} :=
by sorry

end subsets_of_B_l1579_157922


namespace imaginary_part_of_complex_product_l1579_157947

theorem imaginary_part_of_complex_product : Complex.im ((3 * Complex.I - 1) * Complex.I) = -1 := by
  sorry

end imaginary_part_of_complex_product_l1579_157947


namespace value_range_of_f_l1579_157924

-- Define the function f(x) = x^2 - 2x + 2
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the interval (0, 4]
def interval : Set ℝ := {x | 0 < x ∧ x ≤ 4}

-- Theorem statement
theorem value_range_of_f : 
  (∀ x ∈ interval, 1 ≤ f x) ∧ 
  (∀ x ∈ interval, f x ≤ 10) ∧ 
  (∃ x ∈ interval, f x = 1) ∧ 
  (∃ x ∈ interval, f x = 10) :=
sorry

end value_range_of_f_l1579_157924


namespace quadratic_equation_solution_l1579_157981

theorem quadratic_equation_solution :
  ∃ (a b : ℕ+), 
    (∀ x : ℝ, x > 0 → x^2 + 10*x = 34 ↔ x = Real.sqrt a - b) ∧
    a + b = 64 := by
  sorry

end quadratic_equation_solution_l1579_157981


namespace school_journey_problem_l1579_157985

/-- Represents the time taken for John's journey to and from school -/
structure SchoolJourney where
  road_one_way : ℕ        -- Time taken to walk one way by road
  shortcut_one_way : ℕ    -- Time taken to walk one way by shortcut

/-- The theorem representing John's school journey problem -/
theorem school_journey_problem (j : SchoolJourney) 
  (h1 : j.road_one_way + j.shortcut_one_way = 50)  -- Road + Shortcut = 50 minutes
  (h2 : 2 * j.shortcut_one_way = 30)               -- Shortcut both ways = 30 minutes
  : 2 * j.road_one_way = 70 := by                  -- Road both ways = 70 minutes
  sorry

#check school_journey_problem

end school_journey_problem_l1579_157985


namespace plant_arrangement_count_l1579_157933

theorem plant_arrangement_count : ℕ := by
  -- Define the number of each type of plant
  let basil_count : ℕ := 4
  let tomato_count : ℕ := 4
  let pepper_count : ℕ := 2

  -- Define the total number of groups (basil plants + tomato group + pepper group)
  let total_groups : ℕ := basil_count + 2

  -- Calculate the number of ways to arrange the groups
  let group_arrangements : ℕ := Nat.factorial total_groups

  -- Calculate the number of ways to arrange plants within their groups
  let tomato_arrangements : ℕ := Nat.factorial tomato_count
  let pepper_arrangements : ℕ := Nat.factorial pepper_count

  -- Calculate the total number of arrangements
  let total_arrangements : ℕ := group_arrangements * tomato_arrangements * pepper_arrangements

  -- Prove that the total number of arrangements is 34560
  have h : total_arrangements = 34560 := by sorry

  exact 34560

end plant_arrangement_count_l1579_157933


namespace niles_win_probability_l1579_157979

/-- Represents a die with six faces. -/
structure Die :=
  (faces : Fin 6 → ℕ)

/-- Billie's die -/
def billie_die : Die :=
  { faces := λ i => i.val + 1 }

/-- Niles' die -/
def niles_die : Die :=
  { faces := λ i => if i.val < 3 then 4 else 5 }

/-- The probability that Niles wins when rolling against Billie -/
def niles_win_prob : ℚ :=
  7 / 12

theorem niles_win_probability :
  let p := niles_win_prob.num
  let q := niles_win_prob.den
  7 * p + 11 * q = 181 := by sorry

end niles_win_probability_l1579_157979


namespace petya_win_probability_is_1_256_l1579_157960

/-- The "Heap of Stones" game -/
structure HeapOfStones where
  initial_stones : Nat
  min_take : Nat
  max_take : Nat

/-- Player types -/
inductive Player
  | Petya
  | Computer

/-- Game state -/
structure GameState where
  stones_left : Nat
  current_player : Player

/-- Optimal play function for the computer -/
def optimal_play (game : HeapOfStones) (state : GameState) : Nat :=
  sorry

/-- Random play function for Petya -/
def random_play (game : HeapOfStones) (state : GameState) : Nat :=
  sorry

/-- The probability of Petya winning the game -/
def petya_win_probability (game : HeapOfStones) : Real :=
  sorry

/-- Theorem stating the probability of Petya winning -/
theorem petya_win_probability_is_1_256 (game : HeapOfStones) :
  game.initial_stones = 16 ∧ 
  game.min_take = 1 ∧ 
  game.max_take = 4 →
  petya_win_probability game = 1 / 256 :=
by sorry

end petya_win_probability_is_1_256_l1579_157960


namespace reciprocal_inequality_l1579_157951

theorem reciprocal_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end reciprocal_inequality_l1579_157951


namespace problem_statement_l1579_157903

theorem problem_statement (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 5) :
  x^2 * y + x * y^2 = 15 := by
sorry

end problem_statement_l1579_157903


namespace square_sum_product_l1579_157911

theorem square_sum_product (x y : ℝ) 
  (h1 : (x + y)^2 = 49) 
  (h2 : x * y = 8) : 
  x^2 + y^2 + 3 * x * y = 57 := by
sorry

end square_sum_product_l1579_157911


namespace root_sum_reciprocal_products_l1579_157966

def polynomial (x : ℝ) : ℝ := x^4 + 6*x^3 + 11*x^2 + 7*x + 5

theorem root_sum_reciprocal_products (p q r s : ℝ) :
  polynomial p = 0 → polynomial q = 0 → polynomial r = 0 → polynomial s = 0 →
  (1 / (p * q)) + (1 / (p * r)) + (1 / (p * s)) + (1 / (q * r)) + (1 / (q * s)) + (1 / (r * s)) = 11 / 5 := by
  sorry

end root_sum_reciprocal_products_l1579_157966


namespace composite_sum_of_powers_l1579_157949

theorem composite_sum_of_powers (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ (a^1984 + b^1984 + c^1984 + d^1984 = x * y) := by
  sorry

end composite_sum_of_powers_l1579_157949


namespace simplify_and_multiply_l1579_157972

theorem simplify_and_multiply :
  (3 / 504 - 17 / 72) * (5 / 7) = -145 / 882 := by
  sorry

end simplify_and_multiply_l1579_157972


namespace saree_final_price_l1579_157938

def original_price : ℝ := 4000

def discount1 : ℝ := 0.15
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.08
def flat_discount : ℝ := 300

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price : ℝ :=
  apply_discount (apply_discount (apply_discount original_price discount1) discount2) discount3 - flat_discount

theorem saree_final_price :
  final_price = 2515.20 :=
by sorry

end saree_final_price_l1579_157938


namespace wire_bending_l1579_157906

theorem wire_bending (r : ℝ) (h : r = 56) : 
  let circle_circumference := 2 * Real.pi * r
  let square_side := circle_circumference / 4
  let square_area := square_side * square_side
  square_area = 784 * Real.pi^2 := by
sorry

end wire_bending_l1579_157906


namespace ceiling_minus_x_l1579_157980

theorem ceiling_minus_x (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) : ⌈x⌉ - x = 1 - (x - ⌊x⌋) := by
  sorry

end ceiling_minus_x_l1579_157980


namespace equation_solutions_l1579_157962

theorem equation_solutions : 
  let f (x : ℂ) := (x - 2)^4 + (x - 6)^4 + 16
  ∀ x : ℂ, f x = 0 ↔ 
    x = 4 + 2*I*Real.sqrt 3 ∨ 
    x = 4 - 2*I*Real.sqrt 3 ∨ 
    x = 4 + I*Real.sqrt 2 ∨ 
    x = 4 - I*Real.sqrt 2 :=
by sorry

end equation_solutions_l1579_157962


namespace floor_expression_equals_eight_l1579_157904

theorem floor_expression_equals_eight (n : ℕ) (h : n = 2009) :
  ⌊((n + 1)^3 / ((n - 1) * n : ℝ)) - ((n - 1)^3 / (n * (n + 1) : ℝ))⌋ = 8 := by
  sorry

end floor_expression_equals_eight_l1579_157904


namespace age_difference_proof_l1579_157976

theorem age_difference_proof (a b : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ b)
  (h4 : 10 * a + b + 10 = 3 * (10 * b + a + 10)) :
  (10 * a + b) - (10 * b + a) = 27 := by
  sorry

end age_difference_proof_l1579_157976


namespace goat_feed_theorem_l1579_157900

/-- Represents the number of days feed lasts for a given number of goats -/
def feed_duration (num_goats : ℕ) (days : ℕ) : Prop := True

theorem goat_feed_theorem (D : ℕ) :
  feed_duration 20 D →
  feed_duration 30 (D - 3) →
  feed_duration 15 (D + D) :=
by
  sorry

#check goat_feed_theorem

end goat_feed_theorem_l1579_157900


namespace vertex_on_x_axis_rising_right_side_simplest_form_f_satisfies_conditions_l1579_157973

/-- A quadratic function that satisfies the given conditions -/
def f (x : ℝ) : ℝ := x^2

/-- The vertex of f is on the x-axis -/
theorem vertex_on_x_axis : ∃ h : ℝ, f h = 0 ∧ ∀ x : ℝ, f x ≥ f h :=
sorry

/-- f is rising on the right side of the y-axis -/
theorem rising_right_side : ∀ x > 0, ∀ y > x, f y > f x :=
sorry

/-- f is in its simplest form -/
theorem simplest_form : ∀ a b c : ℝ, (∀ x : ℝ, a * x^2 + b * x + c = f x) → a = 1 ∧ b = 0 ∧ c = 0 :=
sorry

/-- f satisfies all the required conditions -/
theorem f_satisfies_conditions : 
  (∃ h : ℝ, f h = 0 ∧ ∀ x : ℝ, f x ≥ f h) ∧ 
  (∀ x > 0, ∀ y > x, f y > f x) ∧
  (∀ a b c : ℝ, (∀ x : ℝ, a * x^2 + b * x + c = f x) → a = 1 ∧ b = 0 ∧ c = 0) :=
sorry

end vertex_on_x_axis_rising_right_side_simplest_form_f_satisfies_conditions_l1579_157973


namespace stacy_growth_difference_l1579_157940

/-- Calculates the difference in growth between Stacy and her brother -/
def growth_difference (stacy_initial_height stacy_final_height brother_growth : ℕ) : ℕ :=
  (stacy_final_height - stacy_initial_height) - brother_growth

/-- Proves that the difference in growth between Stacy and her brother is 6 inches -/
theorem stacy_growth_difference :
  growth_difference 50 57 1 = 6 := by
  sorry

end stacy_growth_difference_l1579_157940


namespace sum_of_reciprocals_l1579_157929

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end sum_of_reciprocals_l1579_157929


namespace combined_shoe_size_l1579_157920

/-- Given the shoe sizes of Jasmine, Alexa, and Clara, prove their combined shoe size. -/
theorem combined_shoe_size 
  (jasmine_size : ℕ) 
  (alexa_size : ℕ) 
  (clara_size : ℕ) 
  (h1 : jasmine_size = 7)
  (h2 : alexa_size = 2 * jasmine_size)
  (h3 : clara_size = 3 * jasmine_size) : 
  jasmine_size + alexa_size + clara_size = 42 := by
  sorry

end combined_shoe_size_l1579_157920


namespace max_d_is_25_l1579_157927

/-- Sequence term definition -/
def a (n : ℕ) : ℕ := 100 + n^2 + 2*n

/-- Greatest common divisor of consecutive terms -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- The maximum value of d_n is 25 -/
theorem max_d_is_25 : ∃ k : ℕ, d k = 25 ∧ ∀ n : ℕ, d n ≤ 25 :=
  sorry

end max_d_is_25_l1579_157927


namespace hyperbola_asymptote_l1579_157912

/-- The hyperbola and parabola share a common focus -/
structure SharedFocus (a b : ℝ) :=
  (a_pos : a > 0)
  (b_pos : b > 0)
  (hyperbola : ℝ → ℝ → Prop)
  (parabola : ℝ → ℝ → Prop)
  (focus : ℝ × ℝ)
  (hyperbola_eq : ∀ x y, hyperbola x y ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (parabola_eq : ∀ x y, parabola x y ↔ y^2 = 8*x)
  (shared_focus : ∃ (x y : ℝ), hyperbola x y ∧ parabola x y)

/-- The intersection point P and its distance from the focus -/
structure IntersectionPoint (a b : ℝ) extends SharedFocus a b :=
  (P : ℝ × ℝ)
  (on_hyperbola : hyperbola P.1 P.2)
  (on_parabola : parabola P.1 P.2)
  (distance_PF : Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 5)

/-- The theorem statement -/
theorem hyperbola_asymptote 
  {a b : ℝ} (h : IntersectionPoint a b) :
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ 
  (∀ x y, h.hyperbola x y → (x = k*y ∨ x = -k*y)) :=
sorry

end hyperbola_asymptote_l1579_157912


namespace scientific_notation_63000_l1579_157925

theorem scientific_notation_63000 : 63000 = 6.3 * (10 ^ 4) := by sorry

end scientific_notation_63000_l1579_157925


namespace ratio_equality_l1579_157999

theorem ratio_equality : (2^2001 * 3^2003) / 6^2002 = 3/2 := by
  sorry

end ratio_equality_l1579_157999


namespace total_distance_is_75_miles_l1579_157975

/-- Calculates the total distance traveled given initial speed and time, where the second part of the journey is twice as long at twice the speed. -/
def totalDistance (initialSpeed : ℝ) (initialTime : ℝ) : ℝ :=
  let distance1 := initialSpeed * initialTime
  let distance2 := (2 * initialSpeed) * (2 * initialTime)
  distance1 + distance2

/-- Proves that given an initial speed of 30 mph and an initial time of 0.5 hours, the total distance traveled is 75 miles. -/
theorem total_distance_is_75_miles :
  totalDistance 30 0.5 = 75 := by
  sorry

end total_distance_is_75_miles_l1579_157975


namespace additional_cost_for_new_requirements_l1579_157928

/-- The additional cost for Farmer Brown to meet his new requirements -/
theorem additional_cost_for_new_requirements
  (initial_bales : ℕ)
  (original_cost_per_bale : ℕ)
  (better_quality_cost_per_bale : ℕ)
  (h1 : initial_bales = 10)
  (h2 : original_cost_per_bale = 15)
  (h3 : better_quality_cost_per_bale = 18) :
  (2 * initial_bales * better_quality_cost_per_bale) - (initial_bales * original_cost_per_bale) = 210 :=
by
  sorry

end additional_cost_for_new_requirements_l1579_157928


namespace necessary_but_not_sufficient_l1579_157930

theorem necessary_but_not_sufficient (a : ℝ) : 
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ y : ℝ, y > 1 ∧ y ≤ 2) := by
  sorry

end necessary_but_not_sufficient_l1579_157930


namespace parabola_unique_coefficients_l1579_157963

/-- A parabola is defined by the equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ := p.a * x^2 + p.b * x + p.c

/-- The slope of the tangent line to the parabola at a given x-coordinate -/
def Parabola.slope (p : Parabola) (x : ℝ) : ℝ := 2 * p.a * x + p.b

theorem parabola_unique_coefficients : 
  ∀ p : Parabola, 
    p.y 1 = 1 → 
    p.y 2 = -1 → 
    p.slope 2 = 1 → 
    p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry

end parabola_unique_coefficients_l1579_157963


namespace quadratic_factoring_l1579_157992

/-- A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The result of factoring a quadratic equation -/
inductive FactoredForm
  | Product : FactoredForm

/-- Factoring a quadratic equation results in a product form -/
theorem quadratic_factoring (eq : QuadraticEquation) : ∃ (f : FactoredForm), f = FactoredForm.Product := by
  sorry

end quadratic_factoring_l1579_157992


namespace negation_of_universal_proposition_l1579_157995

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 2 * x^4 - x^2 + 1 < 0) ↔ (∃ x : ℝ, 2 * x^4 - x^2 + 1 ≥ 0) :=
by sorry

end negation_of_universal_proposition_l1579_157995


namespace interest_rate_calculation_l1579_157965

theorem interest_rate_calculation (P r : ℝ) 
  (h1 : P * (1 + 4 * r) = 400)
  (h2 : P * (1 + 6 * r) = 500) :
  r = 0.25 := by
  sorry

end interest_rate_calculation_l1579_157965


namespace largest_prime_factor_l1579_157986

theorem largest_prime_factor :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (17^4 + 2*17^3 + 17^2 - 16^4) ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ (17^4 + 2*17^3 + 17^2 - 16^4) → q ≤ p :=
by
  use 17
  sorry

#check largest_prime_factor

end largest_prime_factor_l1579_157986


namespace ellipse_and_circle_properties_l1579_157990

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the circle D
def circle_D (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1/4

-- Theorem statement
theorem ellipse_and_circle_properties :
  -- The eccentricity of ellipse C is √3/2
  (∃ e : ℝ, e = Real.sqrt 3 / 2 ∧
    ∀ x y : ℝ, ellipse_C x y → 
      e = Real.sqrt (1 - (Real.sqrt (1 - x^2/4))^2) / 2) ∧
  -- Circle D lies entirely inside ellipse C
  (∀ x y : ℝ, circle_D x y → ellipse_C x y) :=
by sorry

end ellipse_and_circle_properties_l1579_157990


namespace michelles_necklace_l1579_157945

/-- Prove that the number of silver beads is 10 given the conditions of Michelle's necklace. -/
theorem michelles_necklace (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ) :
  total_beads = 40 →
  blue_beads = 5 →
  red_beads = 2 * blue_beads →
  white_beads = blue_beads + red_beads →
  silver_beads = total_beads - (blue_beads + red_beads + white_beads) →
  silver_beads = 10 := by
  sorry

end michelles_necklace_l1579_157945


namespace remaining_insects_l1579_157991

def playground_insects (spiders ants initial_ladybugs departed_ladybugs : ℕ) : ℕ :=
  spiders + ants + initial_ladybugs - departed_ladybugs

theorem remaining_insects : 
  playground_insects 3 12 8 2 = 21 := by sorry

end remaining_insects_l1579_157991


namespace tripled_base_and_exponent_l1579_157950

theorem tripled_base_and_exponent (a b : ℝ) (x : ℝ) (hx : x > 0) :
  (3*a)^(3*b) = a^b * x^b → x = 27 * a^2 := by
  sorry

end tripled_base_and_exponent_l1579_157950


namespace sheilas_weekly_earnings_l1579_157958

/-- Sheila's weekly earnings calculation -/
theorem sheilas_weekly_earnings :
  let hourly_rate : ℕ := 12
  let hours_mon_wed_fri : ℕ := 8
  let hours_tue_thu : ℕ := 6
  let days_8_hours : ℕ := 3
  let days_6_hours : ℕ := 2
  let earnings_8_hour_days : ℕ := hourly_rate * hours_mon_wed_fri * days_8_hours
  let earnings_6_hour_days : ℕ := hourly_rate * hours_tue_thu * days_6_hours
  let total_earnings : ℕ := earnings_8_hour_days + earnings_6_hour_days
  total_earnings = 432 :=
by sorry

end sheilas_weekly_earnings_l1579_157958


namespace tree_leaves_theorem_l1579_157942

/-- Calculates the number of leaves remaining on a tree after 5 weeks of shedding --/
def leaves_remaining (initial_leaves : ℕ) : ℕ :=
  let week1_remaining := initial_leaves - initial_leaves / 5
  let week2_shed := (week1_remaining * 30) / 100
  let week2_remaining := week1_remaining - week2_shed
  let week3_shed := (week2_shed * 60) / 100
  let week3_remaining := week2_remaining - week3_shed
  let week4_shed := week3_remaining / 2
  let week4_remaining := week3_remaining - week4_shed
  let week5_shed := (week3_shed * 2) / 3
  week4_remaining - week5_shed

/-- Theorem stating that a tree with 5000 initial leaves will have 560 leaves remaining after 5 weeks of shedding --/
theorem tree_leaves_theorem :
  leaves_remaining 5000 = 560 := by
  sorry

end tree_leaves_theorem_l1579_157942


namespace quadratic_expression_value_l1579_157955

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 4 * x + y = 20) 
  (h2 : x + 4 * y = 16) : 
  17 * x^2 + 20 * x * y + 17 * y^2 = 656 := by
sorry

end quadratic_expression_value_l1579_157955


namespace polynomial_properties_l1579_157926

/-- Definition of our polynomial -/
def p (x y : ℝ) : ℝ := -x^3 - 2*x^2*y^2 + 3*y^2

/-- The number of terms in our polynomial -/
def num_terms : ℕ := 3

/-- The degree of our polynomial -/
def poly_degree : ℕ := 4

/-- Theorem stating the properties of our polynomial -/
theorem polynomial_properties :
  (num_terms = 3) ∧ (poly_degree = 4) := by sorry

end polynomial_properties_l1579_157926


namespace ratio_to_percentage_difference_l1579_157984

theorem ratio_to_percentage_difference (A B : ℝ) (hA : A > 0) (hB : B > 0) (h_ratio : A / B = 1/6 / (1/5)) :
  (B - A) / A * 100 = 20 := by
  sorry

end ratio_to_percentage_difference_l1579_157984


namespace negation_of_universal_proposition_l1579_157994

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^4 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^4 < 0) :=
by sorry

end negation_of_universal_proposition_l1579_157994


namespace order_of_abc_l1579_157921

theorem order_of_abc (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : a^2 + b^2 < a^2 + c^2 ∧ a^2 + c^2 < b^2 + c^2) : 
  a < b ∧ b < c := by
  sorry

end order_of_abc_l1579_157921


namespace airsickness_gender_related_l1579_157916

/-- Represents the contingency table data for airsickness and gender --/
structure AirsicknessData :=
  (male_sick : ℕ)
  (male_not_sick : ℕ)
  (female_sick : ℕ)
  (female_not_sick : ℕ)

/-- Calculates the K² value for the given airsickness data --/
def calculate_k_squared (data : AirsicknessData) : ℚ :=
  let n := data.male_sick + data.male_not_sick + data.female_sick + data.female_not_sick
  let ad := data.male_sick * data.female_not_sick
  let bc := data.male_not_sick * data.female_sick
  let numerator := n * (ad - bc) * (ad - bc)
  let denominator := (data.male_sick + data.male_not_sick) * 
                     (data.female_sick + data.female_not_sick) * 
                     (data.male_sick + data.female_sick) * 
                     (data.male_not_sick + data.female_not_sick)
  numerator / denominator

/-- Theorem stating that the K² value for the given data indicates a relationship between airsickness and gender --/
theorem airsickness_gender_related (data : AirsicknessData) 
  (h1 : data.male_sick = 28)
  (h2 : data.male_not_sick = 28)
  (h3 : data.female_sick = 28)
  (h4 : data.female_not_sick = 56) :
  calculate_k_squared data > 3841 / 1000 :=
by sorry

end airsickness_gender_related_l1579_157916


namespace clock_hand_position_l1579_157964

/-- Represents the angle of the hour hand in degrees -/
def hour_angle (hours : ℕ) : ℝ := (hours * 30 : ℝ)

/-- Theorem: For a clock with radius 15 units, when the hour hand points to 7 hours,
    the cosine of the angle is -√3/2 and the horizontal displacement of the tip
    of the hour hand from the center is -15√3/2 units. -/
theorem clock_hand_position (radius : ℝ) (hours : ℕ) 
    (h1 : radius = 15)
    (h2 : hours = 7) :
    let angle := hour_angle hours
    (Real.cos angle = -Real.sqrt 3 / 2) ∧ 
    (radius * Real.cos angle = -15 * Real.sqrt 3 / 2) := by
  sorry

end clock_hand_position_l1579_157964


namespace city_partition_l1579_157944

/-- A graph representing cities and flight routes -/
structure CityGraph where
  V : Type* -- Set of vertices (cities)
  E : V → V → Prop -- Edge relation (flight routes)

/-- A partition of edges into k sets representing k airlines -/
def AirlinePartition (G : CityGraph) (k : ℕ) :=
  ∃ (P : Fin k → (G.V → G.V → Prop)), 
    (∀ u v, G.E u v ↔ ∃ i, P i u v) ∧
    (∀ i, ∀ {u v w x}, P i u v → P i w x → (u = w ∨ u = x ∨ v = w ∨ v = x))

/-- A partition of vertices into k+2 sets -/
def VertexPartition (G : CityGraph) (k : ℕ) :=
  ∃ (f : G.V → Fin (k + 2)), ∀ u v, G.E u v → f u ≠ f v

theorem city_partition (G : CityGraph) (k : ℕ) :
  AirlinePartition G k → VertexPartition G k := by sorry

end city_partition_l1579_157944


namespace this_is_2345_l1579_157977

def letter_to_digit : Char → Nat
| 'M' => 0
| 'A' => 1
| 'T' => 2
| 'H' => 3
| 'I' => 4
| 'S' => 5
| 'F' => 6
| 'U' => 7
| 'N' => 8
| _ => 9  -- Default case for completeness

def code_to_number (code : List Char) : Nat :=
  code.foldl (fun acc d => acc * 10 + letter_to_digit d) 0

theorem this_is_2345 :
  code_to_number ['T', 'H', 'I', 'S'] = 2345 := by
  sorry

end this_is_2345_l1579_157977


namespace system_solution_condition_l1579_157974

theorem system_solution_condition (a : ℝ) :
  (∃ (x y b : ℝ), y = x^2 + a ∧ x^2 + y^2 + 2*b^2 = 2*b*(x - y) + 1) →
  a ≤ Real.sqrt 2 + 1/4 := by
sorry

end system_solution_condition_l1579_157974


namespace real_y_condition_l1579_157919

theorem real_y_condition (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 2 * x * y + x + 5 = 0) ↔ x ≤ -3 ∨ x ≥ 5 := by
  sorry

end real_y_condition_l1579_157919


namespace unique_root_quadratic_theorem_l1579_157961

/-- A quadratic polynomial with exactly one root -/
def UniqueRootQuadratic (g : ℝ → ℝ) : Prop :=
  (∃ x₀, g x₀ = 0) ∧ (∀ x y, g x = 0 → g y = 0 → x = y)

theorem unique_root_quadratic_theorem
  (g : ℝ → ℝ)
  (h_unique : UniqueRootQuadratic g)
  (a b c d : ℝ)
  (h_ac : a ≠ c)
  (h_composed : UniqueRootQuadratic (fun x ↦ g (a * x + b) + g (c * x + d))) :
  ∃ x₀, g x₀ = 0 ∧ x₀ = (a * d - b * c) / (a - c) := by
  sorry

end unique_root_quadratic_theorem_l1579_157961


namespace friday_pushups_l1579_157909

/-- Calculates the number of push-ups Miriam does on Friday given her workout schedule --/
theorem friday_pushups (monday : ℕ) : 
  let tuesday := (monday : ℚ) * (14 : ℚ) / 10
  let wednesday := (monday : ℕ) * 2
  let thursday := ((monday : ℚ) + tuesday + (wednesday : ℚ)) / 2
  let friday := (monday : ℚ) + tuesday + (wednesday : ℚ) + thursday
  monday = 5 → friday = 33 := by
sorry


end friday_pushups_l1579_157909


namespace injective_function_equality_l1579_157988

def injective (f : ℕ → ℝ) : Prop := ∀ n m : ℕ, f n = f m → n = m

theorem injective_function_equality (f : ℕ → ℝ) (n m : ℕ) 
  (h_inj : injective f) 
  (h_eq : 1 / f n + 1 / f m = 4 / (f n + f m)) : 
  n = m := by
  sorry

end injective_function_equality_l1579_157988


namespace no_solution_implies_a_leq_two_l1579_157952

theorem no_solution_implies_a_leq_two (a : ℝ) :
  (∀ x : ℝ, ¬(x ≥ a + 2 ∧ x < 3*a - 2)) → a ≤ 2 := by
  sorry

end no_solution_implies_a_leq_two_l1579_157952


namespace three_card_sequence_l1579_157946

-- Define the ranks and suits
inductive Rank
| Ace | King

inductive Suit
| Heart | Diamond

-- Define a card as a pair of rank and suit
structure Card :=
  (rank : Rank)
  (suit : Suit)

def is_king (c : Card) : Prop := c.rank = Rank.King
def is_ace (c : Card) : Prop := c.rank = Rank.Ace
def is_heart (c : Card) : Prop := c.suit = Suit.Heart
def is_diamond (c : Card) : Prop := c.suit = Suit.Diamond

-- Define the theorem
theorem three_card_sequence (c1 c2 c3 : Card) : 
  -- Condition 1
  (is_king c2 ∨ is_king c3) ∧ is_ace c1 →
  -- Condition 2
  (is_king c1 ∨ is_king c2) ∧ is_king c3 →
  -- Condition 3
  (is_heart c1 ∨ is_heart c2) ∧ is_diamond c3 →
  -- Condition 4
  is_heart c1 ∧ (is_heart c2 ∨ is_heart c3) →
  -- Conclusion
  is_heart c1 ∧ is_ace c1 ∧ 
  is_heart c2 ∧ is_king c2 ∧
  is_diamond c3 ∧ is_king c3 := by
  sorry


end three_card_sequence_l1579_157946


namespace combined_mean_of_two_sets_l1579_157969

theorem combined_mean_of_two_sets (set1_mean set2_mean : ℚ) :
  set1_mean = 18 →
  set2_mean = 16 →
  (7 * set1_mean + 8 * set2_mean) / 15 = 254 / 15 := by
  sorry

end combined_mean_of_two_sets_l1579_157969


namespace smallest_n_congruence_l1579_157923

theorem smallest_n_congruence (n : ℕ+) : 
  (19 * n.val ≡ 1589 [MOD 9]) ↔ n = 5 := by sorry

end smallest_n_congruence_l1579_157923


namespace circle_common_chord_l1579_157970

theorem circle_common_chord (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = a^2) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 + a*y - 6 = 0) ∧ 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 = a^2 ∧ 
    x₂^2 + y₂^2 + a*y₂ - 6 = 0 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) → 
  a = 2 ∨ a = -2 := by
sorry

end circle_common_chord_l1579_157970


namespace prob_not_greater_than_four_is_two_thirds_l1579_157937

/-- A die is represented as a finite type with 6 elements -/
inductive Die : Type
  | one | two | three | four | five | six

/-- The probability of rolling a number not greater than 4 on a six-sided die -/
def prob_not_greater_than_four : ℚ :=
  (Finset.filter (fun x => x ≤ 4) (Finset.range 6)).card /
  (Finset.range 6).card

/-- Theorem stating that the probability of rolling a number not greater than 4 
    on a six-sided die is 2/3 -/
theorem prob_not_greater_than_four_is_two_thirds :
  prob_not_greater_than_four = 2 / 3 := by
  sorry


end prob_not_greater_than_four_is_two_thirds_l1579_157937


namespace triangle_angle_relationships_l1579_157914

/-- Given two triangles ABC and UVW with the specified side relationships,
    prove that ABC is acute-angled and express angles of UVW in terms of ABC. -/
theorem triangle_angle_relationships
  (a b c u v w : ℝ)
  (ha : a^2 = u * (v + w - u))
  (hb : b^2 = v * (w + u - v))
  (hc : c^2 = w * (u + v - w))
  : (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) ∧
    ∃ (A B C U V W : ℝ),
    (0 < A ∧ A < π / 2) ∧
    (0 < B ∧ B < π / 2) ∧
    (0 < C ∧ C < π / 2) ∧
    (A + B + C = π) ∧
    (U = π - 2 * A) ∧
    (V = π - 2 * B) ∧
    (W = π - 2 * C) := by
  sorry

end triangle_angle_relationships_l1579_157914


namespace student_number_problem_l1579_157934

theorem student_number_problem (x : ℝ) : 2 * x - 152 = 102 → x = 127 := by
  sorry

end student_number_problem_l1579_157934


namespace shaded_fraction_of_specific_quilt_l1579_157943

/-- Represents a square quilt made of unit squares -/
structure Quilt where
  size : Nat
  divided_squares : Finset (Nat × Nat)
  shaded_squares : Finset (Nat × Nat)

/-- The fraction of the quilt that is shaded -/
def shaded_fraction (q : Quilt) : Rat :=
  sorry

/-- Theorem stating the shaded fraction of the specific quilt configuration -/
theorem shaded_fraction_of_specific_quilt :
  ∃ (q : Quilt),
    q.size = 4 ∧
    q.shaded_squares.card = 6 ∧
    shaded_fraction q = 3 / 16 := by
  sorry

end shaded_fraction_of_specific_quilt_l1579_157943


namespace add_negative_three_l1579_157932

theorem add_negative_three : 2 + (-3) = -1 := by sorry

end add_negative_three_l1579_157932


namespace day_365_is_tuesday_l1579_157908

/-- Days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Function to determine the day of the week for a given day number -/
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem day_365_is_tuesday (h : dayOfWeek 15 = DayOfWeek.Tuesday) :
  dayOfWeek 365 = DayOfWeek.Tuesday := by
  sorry

end day_365_is_tuesday_l1579_157908


namespace min_value_theorem_l1579_157931

theorem min_value_theorem (a : ℝ) (h : 8 * a^2 + 6 * a + 2 = 4) :
  ∃ (m : ℝ), (3 * a + 1 ≥ m) ∧ (∀ x, 8 * x^2 + 6 * x + 2 = 4 → 3 * x + 1 ≥ m) ∧ (m = -2) :=
sorry

end min_value_theorem_l1579_157931


namespace anatoliy_handshakes_l1579_157971

theorem anatoliy_handshakes (n : ℕ) (total_handshakes : ℕ) : 
  total_handshakes = 197 →
  (n * (n - 1)) / 2 + 7 = total_handshakes →
  ∃ (k : ℕ), k = 7 ∧ k ≤ n ∧ (n * (n - 1)) / 2 + k = total_handshakes :=
by sorry

end anatoliy_handshakes_l1579_157971


namespace angle_x_measure_l1579_157936

-- Define the triangle ABD
structure Triangle :=
  (A B D : Point)

-- Define the angles in the triangle
def angle_ABC (t : Triangle) : ℝ := 108
def angle_ABD (t : Triangle) : ℝ := 180 - angle_ABC t
def angle_BAD (t : Triangle) : ℝ := 26

-- Theorem statement
theorem angle_x_measure (t : Triangle) :
  180 - angle_ABD t - angle_BAD t = 82 :=
sorry

end angle_x_measure_l1579_157936


namespace witnesses_same_type_l1579_157939

-- Define the types of people
inductive PersonType
| Knight
| Liar

-- Define the statements as functions
def statement_A (X Y : Prop) : Prop := X → Y
def statement_B (X Y : Prop) : Prop := ¬X ∨ Y

-- Main theorem
theorem witnesses_same_type (X Y : Prop) (A B : PersonType) :
  (A = PersonType.Knight ↔ statement_A X Y) →
  (B = PersonType.Knight ↔ statement_B X Y) →
  A = B :=
sorry

end witnesses_same_type_l1579_157939


namespace village_plots_count_l1579_157905

theorem village_plots_count (street_length : ℝ) (narrow_width wide_width : ℝ)
  (narrow_plot_diff : ℕ) (plot_area_diff : ℝ) :
  street_length = 1200 →
  narrow_width = 50 →
  wide_width = 60 →
  narrow_plot_diff = 5 →
  plot_area_diff = 1200 →
  ∃ (wide_plots narrow_plots : ℕ),
    narrow_plots = wide_plots + narrow_plot_diff ∧
    (narrow_plots : ℝ) * (street_length * narrow_width / narrow_plots) =
      (wide_plots : ℝ) * (street_length * wide_width / wide_plots - plot_area_diff) ∧
    wide_plots + narrow_plots = 45 :=
by sorry

end village_plots_count_l1579_157905


namespace sandwiches_al_can_order_correct_l1579_157967

/-- Represents the types of ingredients available at the deli -/
structure DeliIngredients where
  breads : Nat
  meats : Nat
  cheeses : Nat

/-- Represents the specific ingredients mentioned in the problem -/
structure SpecificIngredients where
  turkey : Bool
  salami : Bool
  swissCheese : Bool
  multiGrainBread : Bool

/-- Calculates the number of sandwiches Al can order -/
def sandwichesAlCanOrder (d : DeliIngredients) (s : SpecificIngredients) : Nat :=
  d.breads * d.meats * d.cheeses - d.breads - d.cheeses

/-- The theorem stating the number of sandwiches Al can order -/
theorem sandwiches_al_can_order_correct (d : DeliIngredients) (s : SpecificIngredients) :
  d.breads = 5 → d.meats = 7 → d.cheeses = 6 →
  s.turkey = true → s.salami = true → s.swissCheese = true → s.multiGrainBread = true →
  sandwichesAlCanOrder d s = 199 := by
  sorry

#check sandwiches_al_can_order_correct

end sandwiches_al_can_order_correct_l1579_157967


namespace rakesh_cash_calculation_l1579_157996

/-- Calculates the cash in hand after fixed deposit and grocery expenses --/
def cash_in_hand (salary : ℚ) (fixed_deposit_rate : ℚ) (grocery_rate : ℚ) : ℚ :=
  let fixed_deposit := salary * fixed_deposit_rate
  let remaining := salary - fixed_deposit
  let groceries := remaining * grocery_rate
  remaining - groceries

/-- Proves that given the conditions, the cash in hand is 2380 --/
theorem rakesh_cash_calculation :
  cash_in_hand 4000 (15/100) (30/100) = 2380 := by
  sorry

end rakesh_cash_calculation_l1579_157996


namespace unique_solution_lcm_gcd_l1579_157918

theorem unique_solution_lcm_gcd : 
  ∃! n : ℕ+, Nat.lcm n 180 = Nat.gcd n 180 + 630 ∧ n = 360 := by sorry

end unique_solution_lcm_gcd_l1579_157918


namespace exponent_equality_and_inequalities_l1579_157987

theorem exponent_equality_and_inequalities : 
  ((-2 : ℤ)^3 = -2^3) ∧ 
  ((-2 : ℤ)^2 ≠ -2^2) ∧ 
  (|(-2 : ℤ)|^2 ≠ -2^2) ∧ 
  (|(-2 : ℤ)|^3 ≠ -2^3) :=
by sorry

end exponent_equality_and_inequalities_l1579_157987


namespace one_third_of_six_to_thirty_l1579_157907

theorem one_third_of_six_to_thirty (x : ℚ) :
  x = (1 / 3) * (6 ^ 30) → x = 2 * (6 ^ 29) := by
  sorry

end one_third_of_six_to_thirty_l1579_157907


namespace expected_games_specific_l1579_157978

/-- Represents a game with given win probabilities -/
structure Game where
  p_frank : ℝ  -- Probability of Frank winning a game
  p_joe : ℝ    -- Probability of Joe winning a game
  games_to_win : ℕ  -- Number of games needed to win the match

/-- Expected number of games in a match -/
def expected_games (g : Game) : ℝ := sorry

/-- Theorem stating the expected number of games in the specific scenario -/
theorem expected_games_specific :
  let g : Game := {
    p_frank := 0.3,
    p_joe := 0.7,
    games_to_win := 21
  }
  expected_games g = 30 := by sorry

end expected_games_specific_l1579_157978


namespace max_movies_watched_l1579_157983

def movie_duration : ℕ := 90
def tuesday_watch_time : ℕ := 270
def wednesday_movie_multiplier : ℕ := 2

theorem max_movies_watched (movie_duration : ℕ) (tuesday_watch_time : ℕ) (wednesday_movie_multiplier : ℕ) :
  movie_duration = 90 →
  tuesday_watch_time = 270 →
  wednesday_movie_multiplier = 2 →
  (tuesday_watch_time / movie_duration + wednesday_movie_multiplier * (tuesday_watch_time / movie_duration)) = 9 :=
by sorry

end max_movies_watched_l1579_157983


namespace purely_imaginary_complex_number_l1579_157913

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ).re = 0 ∧ (a ^ 2 - 3 * a + 2 : ℂ).re = 0 → a = 2 := by
  sorry

end purely_imaginary_complex_number_l1579_157913


namespace distribute_six_balls_three_boxes_l1579_157997

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 28 ways to distribute 6 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 28 := by
  sorry

end distribute_six_balls_three_boxes_l1579_157997


namespace absolute_value_equation_product_l1579_157982

theorem absolute_value_equation_product (x : ℝ) : 
  (|15 / x + 4| = 3) → (∃ y : ℝ, (|15 / y + 4| = 3) ∧ (x * y = 225 / 7)) :=
by sorry

end absolute_value_equation_product_l1579_157982
