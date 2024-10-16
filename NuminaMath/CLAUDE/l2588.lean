import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2588_258865

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 4 → b = 4 → c = 2 →
  (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  a + b > c ∧ b + c > a ∧ a + c > b →  -- triangle inequality
  a + b + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2588_258865


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_l2588_258828

theorem consecutive_odd_integers (x : ℤ) : 
  (x % 2 = 1) →                           -- x is odd
  ((x + 2) % 2 = 1) →                     -- x + 2 is odd
  ((x + 4) % 2 = 1) →                     -- x + 4 is odd
  ((x + 2) + (x + 4) = x + 17) →          -- sum of last two equals first plus 17
  (x + 4 = 15) :=                         -- third integer is 15
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_l2588_258828


namespace NUMINAMATH_CALUDE_total_money_l2588_258844

theorem total_money (john emma lucas : ℚ) 
  (h1 : john = 4 / 5)
  (h2 : emma = 2 / 5)
  (h3 : lucas = 1 / 2) :
  john + emma + lucas = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2588_258844


namespace NUMINAMATH_CALUDE_fault_line_movement_l2588_258820

/-- The total movement of a fault line over two years, given its movement in each year. -/
theorem fault_line_movement (movement_past_year : ℝ) (movement_year_before : ℝ) 
  (h1 : movement_past_year = 1.25)
  (h2 : movement_year_before = 5.25) : 
  movement_past_year + movement_year_before = 6.50 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_movement_l2588_258820


namespace NUMINAMATH_CALUDE_red_marbles_in_bag_l2588_258893

theorem red_marbles_in_bag (total_marbles : ℕ) 
  (prob_two_non_red : ℚ) (red_marbles : ℕ) : 
  total_marbles = 48 →
  prob_two_non_red = 9/16 →
  (((total_marbles - red_marbles : ℚ) / total_marbles) ^ 2 = prob_two_non_red) →
  red_marbles = 12 := by
  sorry

end NUMINAMATH_CALUDE_red_marbles_in_bag_l2588_258893


namespace NUMINAMATH_CALUDE_pet_store_birds_l2588_258817

theorem pet_store_birds (total_birds talking_birds : ℕ) 
  (h1 : total_birds = 77)
  (h2 : talking_birds = 64) :
  total_birds - talking_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l2588_258817


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l2588_258824

theorem max_value_of_trigonometric_expression :
  let f : ℝ → ℝ := λ x => Real.sin (x + π/4) - Real.cos (x + π/3) + Real.sin (x + π/6)
  let domain : Set ℝ := {x | -π/4 ≤ x ∧ x ≤ 0}
  ∃ x ∈ domain, f x = 1 ∧ ∀ y ∈ domain, f y ≤ f x := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l2588_258824


namespace NUMINAMATH_CALUDE_oranges_in_box_l2588_258863

/-- The number of oranges Jonathan takes from the box -/
def oranges_taken : ℕ := 45

/-- The number of oranges left in the box after Jonathan takes some -/
def oranges_left : ℕ := 51

/-- The initial number of oranges in the box -/
def initial_oranges : ℕ := oranges_taken + oranges_left

theorem oranges_in_box : initial_oranges = 96 := by
  sorry

end NUMINAMATH_CALUDE_oranges_in_box_l2588_258863


namespace NUMINAMATH_CALUDE_smallest_congruent_n_l2588_258818

theorem smallest_congruent_n (a b : ℤ) (h1 : a ≡ 23 [ZMOD 60]) (h2 : b ≡ 95 [ZMOD 60]) :
  ∃ n : ℤ, 150 ≤ n ∧ n ≤ 191 ∧ a - b ≡ n [ZMOD 60] ∧
  ∀ m : ℤ, 150 ≤ m ∧ m < n → ¬(a - b ≡ m [ZMOD 60]) ∧ n = 168 := by
  sorry

end NUMINAMATH_CALUDE_smallest_congruent_n_l2588_258818


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l2588_258846

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 9*x^2 + 8*x + 2

-- Define the roots
variable (p q r : ℝ)

-- State that p, q, and r are roots of f
axiom root_p : f p = 0
axiom root_q : f q = 0
axiom root_r : f r = 0

-- State that p, q, and r are distinct
axiom distinct_roots : p ≠ q ∧ q ≠ r ∧ p ≠ r

-- Theorem to prove
theorem sum_of_reciprocal_squares : 1/p^2 + 1/q^2 + 1/r^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l2588_258846


namespace NUMINAMATH_CALUDE_billy_age_l2588_258873

theorem billy_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 60) : 
  billy = 45 := by
sorry

end NUMINAMATH_CALUDE_billy_age_l2588_258873


namespace NUMINAMATH_CALUDE_card_arrangements_sum_14_l2588_258886

-- Define the card suits
inductive Suit
| Hearts
| Clubs

-- Define the card values
def CardValue := Fin 4

-- Define a card as a pair of suit and value
def Card := Suit × CardValue

-- Define the deck of 8 cards
def deck : Finset Card := sorry

-- Function to calculate the sum of card values
def sumCardValues (hand : Finset Card) : Nat := sorry

-- Function to count different arrangements
def countArrangements (hand : Finset Card) : Nat := sorry

theorem card_arrangements_sum_14 :
  (Finset.filter (fun hand => hand.card = 4 ∧ sumCardValues hand = 14)
    (Finset.powerset deck)).sum countArrangements = 396 := by
  sorry

end NUMINAMATH_CALUDE_card_arrangements_sum_14_l2588_258886


namespace NUMINAMATH_CALUDE_same_color_probability_l2588_258885

theorem same_color_probability (total_balls green_balls red_balls : ℕ) 
  (h_total : total_balls = green_balls + red_balls)
  (h_green : green_balls = 6)
  (h_red : red_balls = 4) : 
  (green_balls / total_balls) ^ 2 + (red_balls / total_balls) ^ 2 = 13 / 25 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2588_258885


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_9_l2588_258854

theorem smallest_five_digit_mod_9 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≡ 4 [MOD 9] → 10003 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_9_l2588_258854


namespace NUMINAMATH_CALUDE_sum_of_xy_is_one_l2588_258861

theorem sum_of_xy_is_one (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^5 + 5*x^3*y + 5*x^2*y^2 + 5*x*y^3 + y^5 = 1) : 
  x + y = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_xy_is_one_l2588_258861


namespace NUMINAMATH_CALUDE_four_digit_integers_with_4_or_5_l2588_258851

/-- The number of four-digit positive integers -/
def four_digit_count : ℕ := 9000

/-- The number of options for the first digit when excluding 4 and 5 -/
def first_digit_options : ℕ := 7

/-- The number of options for each of the other three digits when excluding 4 and 5 -/
def other_digit_options : ℕ := 8

/-- The count of four-digit numbers without a 4 or 5 -/
def numbers_without_4_or_5 : ℕ := first_digit_options * other_digit_options * other_digit_options * other_digit_options

/-- The count of four-digit positive integers with at least one digit that is a 4 or a 5 -/
def numbers_with_4_or_5 : ℕ := four_digit_count - numbers_without_4_or_5

theorem four_digit_integers_with_4_or_5 : numbers_with_4_or_5 = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_integers_with_4_or_5_l2588_258851


namespace NUMINAMATH_CALUDE_min_value_at_three_l2588_258836

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 28 + Real.sqrt (9 - x^2)

theorem min_value_at_three :
  ∀ x : ℝ, 9 - x^2 ≥ 0 → f 3 ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_min_value_at_three_l2588_258836


namespace NUMINAMATH_CALUDE_line_circle_distance_theorem_l2588_258878

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Length of tangent from a point to a circle -/
def tangentLength (p : ℝ × ℝ) (c : Circle) : ℝ := sorry

/-- Whether a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop := sorry

/-- Whether a point is on a line -/
def onLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

theorem line_circle_distance_theorem (c : Circle) (l : Line) :
  (¬ intersects l c) →
  (∀ (A B : ℝ × ℝ), onLine A l → onLine B l →
    (distance A B > |tangentLength A c - tangentLength B c| ∧
     distance A B < tangentLength A c + tangentLength B c)) ∧
  (∃ (A B : ℝ × ℝ), onLine A l → onLine B l →
    (distance A B ≤ |tangentLength A c - tangentLength B c| ∨
     distance A B ≥ tangentLength A c + tangentLength B c) →
    intersects l c) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_distance_theorem_l2588_258878


namespace NUMINAMATH_CALUDE_optimal_method_is_random_then_stratified_l2588_258850

/-- Represents a sampling method -/
inductive SamplingMethod
  | Random
  | Stratified
  | RandomThenStratified
  | StratifiedThenRandom

/-- Represents a school with first-year classes -/
structure School where
  num_classes : Nat
  male_female_ratio : Real

/-- Represents the sampling scenario -/
structure SamplingScenario where
  school : School
  num_classes_to_sample : Nat

/-- Determines the optimal sampling method for a given scenario -/
def optimal_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- Theorem stating that the optimal sampling method for the given scenario
    is to use random sampling first, then stratified sampling -/
theorem optimal_method_is_random_then_stratified 
  (scenario : SamplingScenario) 
  (h1 : scenario.school.num_classes = 16) 
  (h2 : scenario.num_classes_to_sample = 2) :
  optimal_sampling_method scenario = SamplingMethod.RandomThenStratified :=
sorry

end NUMINAMATH_CALUDE_optimal_method_is_random_then_stratified_l2588_258850


namespace NUMINAMATH_CALUDE_target_perm_unreachable_cannot_reach_reverse_order_l2588_258864

/-- Represents the three colors of balls -/
inductive Color
  | Red
  | Blue
  | White

/-- Represents a permutation of the three balls -/
def Permutation := (Color × Color × Color)

/-- The initial permutation of the balls -/
def initial_perm : Permutation := (Color.Red, Color.Blue, Color.White)

/-- Checks if a permutation is valid (no ball in its original position) -/
def is_valid_perm (p : Permutation) : Prop :=
  p.1 ≠ Color.Red ∧ p.2.1 ≠ Color.Blue ∧ p.2.2 ≠ Color.White

/-- The set of all valid permutations -/
def valid_perms : Set Permutation :=
  {p | is_valid_perm p}

/-- The target permutation (reverse of initial) -/
def target_perm : Permutation := (Color.White, Color.Blue, Color.Red)

/-- Theorem stating that the target permutation is unreachable -/
theorem target_perm_unreachable : target_perm ∉ valid_perms := by
  sorry

/-- Main theorem: It's impossible to reach the target permutation after any number of valid rearrangements -/
theorem cannot_reach_reverse_order :
  ∀ n : ℕ, ∀ f : ℕ → Permutation,
    (f 0 = initial_perm) →
    (∀ i, i < n → is_valid_perm (f (i + 1))) →
    (f n ≠ target_perm) := by
  sorry

end NUMINAMATH_CALUDE_target_perm_unreachable_cannot_reach_reverse_order_l2588_258864


namespace NUMINAMATH_CALUDE_line_angle_theorem_l2588_258805

/-- Given a line with equation (√6 sin θ)x + √3y - 2 = 0 and oblique angle θ ≠ 0, prove θ = 3π/4 -/
theorem line_angle_theorem (θ : Real) (h1 : θ ≠ 0) :
  (∃ (x y : Real), (Real.sqrt 6 * Real.sin θ) * x + Real.sqrt 3 * y - 2 = 0) →
  (∀ (x y : Real), (Real.sqrt 6 * Real.sin θ) * x + Real.sqrt 3 * y - 2 = 0 →
    Real.tan θ = -(Real.sqrt 6 / Real.sqrt 3) * Real.sin θ) →
  θ = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_angle_theorem_l2588_258805


namespace NUMINAMATH_CALUDE_emma_milk_containers_l2588_258896

/-- The number of weeks Emma buys milk -/
def weeks : ℕ := 3

/-- The number of school days in a week -/
def school_days_per_week : ℕ := 5

/-- The total number of milk containers Emma buys in 3 weeks -/
def total_containers : ℕ := 30

/-- The number of containers Emma buys each school day -/
def containers_per_day : ℚ := total_containers / (weeks * school_days_per_week)

theorem emma_milk_containers : containers_per_day = 2 := by
  sorry

end NUMINAMATH_CALUDE_emma_milk_containers_l2588_258896


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l2588_258845

def Alex : ℚ := 1/6
def Beth : ℚ := 2/5
def Cyril : ℚ := 1/3
def Dan : ℚ := 3/10
def Ella : ℚ := 1 - (Alex + Beth + Cyril + Dan)

def siblings : List ℚ := [Beth, Cyril, Dan, Alex, Ella]

theorem pizza_consumption_order : 
  List.Sorted (fun a b => a ≥ b) siblings ∧ 
  siblings = [Beth, Cyril, Dan, Alex, Ella] :=
sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l2588_258845


namespace NUMINAMATH_CALUDE_garden_area_l2588_258866

/-- Represents a rectangular garden with given properties. -/
structure Garden where
  length : ℝ
  width : ℝ
  length_walk : ℝ
  perimeter_walk : ℝ
  length_condition : length * 30 = length_walk
  perimeter_condition : (2 * length + 2 * width) * 12 = perimeter_walk
  walk_equality : length_walk = perimeter_walk
  length_walk_value : length_walk = 1500

/-- The area of the garden with the given conditions is 625 square meters. -/
theorem garden_area (g : Garden) : g.length * g.width = 625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l2588_258866


namespace NUMINAMATH_CALUDE_fraction_simplification_l2588_258879

theorem fraction_simplification (x : ℝ) (h : x = 10) :
  (x^6 - 100*x^3 + 2500) / (x^3 - 50) = 950 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2588_258879


namespace NUMINAMATH_CALUDE_square_area_15m_l2588_258807

theorem square_area_15m (side_length : ℝ) (h : side_length = 15) : 
  side_length * side_length = 225 := by
sorry

end NUMINAMATH_CALUDE_square_area_15m_l2588_258807


namespace NUMINAMATH_CALUDE_largest_non_expressible_l2588_258811

def is_non_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def is_expressible (n : ℕ) : Prop :=
  ∃ (k m : ℕ), k > 0 ∧ is_non_prime m ∧ n = 47 * k + m

theorem largest_non_expressible : 
  (∀ n > 90, is_expressible n) ∧ 
  ¬is_expressible 90 ∧
  (∀ n < 90, ¬is_expressible n → n < 90) :=
sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l2588_258811


namespace NUMINAMATH_CALUDE_rahul_savings_l2588_258819

/-- Rahul's savings problem -/
theorem rahul_savings (NSC PPF : ℕ) (h1 : 3 * (NSC / 3) = 2 * (PPF / 2)) (h2 : PPF = 72000) :
  NSC + PPF = 180000 := by
  sorry

end NUMINAMATH_CALUDE_rahul_savings_l2588_258819


namespace NUMINAMATH_CALUDE_batsman_average_after_20th_innings_l2588_258894

/-- Represents a batsman's innings record -/
structure BatsmanRecord where
  innings : ℕ
  totalScore : ℕ
  avgIncrease : ℚ
  lastScore : ℕ

/-- Calculates the average score of a batsman -/
def calculateAverage (record : BatsmanRecord) : ℚ :=
  record.totalScore / record.innings

theorem batsman_average_after_20th_innings 
  (record : BatsmanRecord)
  (h1 : record.innings = 20)
  (h2 : record.lastScore = 90)
  (h3 : record.avgIncrease = 2)
  : calculateAverage record = 52 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_20th_innings_l2588_258894


namespace NUMINAMATH_CALUDE_aarons_playground_area_l2588_258897

/-- Represents a rectangular playground with fence posts. -/
structure Playground where
  total_posts : ℕ
  post_spacing : ℕ
  long_side_posts : ℕ
  short_side_posts : ℕ

/-- Calculates the area of the playground given its specifications. -/
def playground_area (p : Playground) : ℕ :=
  (p.short_side_posts - 1) * p.post_spacing * (p.long_side_posts - 1) * p.post_spacing

/-- Theorem stating the area of Aaron's playground is 400 square yards. -/
theorem aarons_playground_area :
  ∃ (p : Playground),
    p.total_posts = 24 ∧
    p.post_spacing = 5 ∧
    p.long_side_posts = 3 * p.short_side_posts - 2 ∧
    playground_area p = 400 := by
  sorry


end NUMINAMATH_CALUDE_aarons_playground_area_l2588_258897


namespace NUMINAMATH_CALUDE_fair_coin_prob_TTHH_l2588_258826

/-- The probability of getting tails on a single flip of a fair coin -/
def prob_tails : ℚ := 1 / 2

/-- The number of times the coin is flipped -/
def num_flips : ℕ := 4

/-- The probability of getting tails on the first two flips and heads on the last two flips -/
def prob_TTHH : ℚ := prob_tails * prob_tails * (1 - prob_tails) * (1 - prob_tails)

theorem fair_coin_prob_TTHH :
  prob_TTHH = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_fair_coin_prob_TTHH_l2588_258826


namespace NUMINAMATH_CALUDE_find_b_l2588_258858

-- Define the relationship between a and b
def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a^2 * Real.sqrt b = k

-- Define the theorem
theorem find_b (a b : ℝ) (h1 : inverse_relation a b) (h2 : a = 2 ∧ b = 81) (h3 : a * b = 48) :
  b = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l2588_258858


namespace NUMINAMATH_CALUDE_inverse_trig_sum_zero_l2588_258834

theorem inverse_trig_sum_zero : 
  Real.arctan (Real.sqrt 3 / 3) + Real.arcsin (-1/2) + Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_trig_sum_zero_l2588_258834


namespace NUMINAMATH_CALUDE_power_of_two_condition_l2588_258809

theorem power_of_two_condition (n : ℕ+) : 
  (∃ k : ℕ, n.val^3 + n.val - 2 = 2^k) ↔ (n.val = 2 ∨ n.val = 5) := by
sorry

end NUMINAMATH_CALUDE_power_of_two_condition_l2588_258809


namespace NUMINAMATH_CALUDE_library_book_difference_prove_book_difference_l2588_258852

theorem library_book_difference (initial_books : ℕ) (bought_two_years_ago : ℕ) 
  (donated_this_year : ℕ) (current_total : ℕ) : ℕ :=
  let books_before_last_year := initial_books + bought_two_years_ago
  let books_bought_last_year := current_total - books_before_last_year + donated_this_year
  books_bought_last_year - bought_two_years_ago

theorem prove_book_difference :
  library_book_difference 500 300 200 1000 = 100 := by
  sorry

end NUMINAMATH_CALUDE_library_book_difference_prove_book_difference_l2588_258852


namespace NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l2588_258891

theorem complex_magnitude_fourth_power : Complex.abs ((2 + 3*Complex.I)^4) = 169 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l2588_258891


namespace NUMINAMATH_CALUDE_dvd_price_calculation_l2588_258869

theorem dvd_price_calculation (num_dvd : ℕ) (num_bluray : ℕ) (bluray_price : ℚ) (avg_price : ℚ) :
  num_dvd = 8 →
  num_bluray = 4 →
  bluray_price = 18 →
  avg_price = 14 →
  ∃ (dvd_price : ℚ),
    dvd_price * num_dvd + bluray_price * num_bluray = avg_price * (num_dvd + num_bluray) ∧
    dvd_price = 12 :=
by sorry

end NUMINAMATH_CALUDE_dvd_price_calculation_l2588_258869


namespace NUMINAMATH_CALUDE_egg_count_l2588_258856

theorem egg_count (initial_eggs : ℕ) (used_eggs : ℕ) (num_chickens : ℕ) (eggs_per_chicken : ℕ) : 
  initial_eggs = 10 → 
  used_eggs = 5 → 
  num_chickens = 2 → 
  eggs_per_chicken = 3 → 
  initial_eggs - used_eggs + num_chickens * eggs_per_chicken = 11 := by
  sorry

#check egg_count

end NUMINAMATH_CALUDE_egg_count_l2588_258856


namespace NUMINAMATH_CALUDE_roxy_plants_l2588_258884

def plants_problem (initial_flowering : ℕ) (initial_fruiting_ratio : ℕ) 
  (bought_fruiting : ℕ) (given_away_flowering : ℕ) (given_away_fruiting : ℕ) 
  (final_total : ℕ) : Prop :=
  ∃ (bought_flowering : ℕ),
    let initial_fruiting := initial_flowering * initial_fruiting_ratio
    let initial_total := initial_flowering + initial_fruiting
    let after_buying := initial_total + bought_flowering + bought_fruiting
    let after_giving := after_buying - given_away_flowering - given_away_fruiting
    after_giving = final_total ∧ bought_flowering = 3

theorem roxy_plants : plants_problem 7 2 2 1 4 21 := by
  sorry

end NUMINAMATH_CALUDE_roxy_plants_l2588_258884


namespace NUMINAMATH_CALUDE_cubic_tangent_line_l2588_258825

/-- Given a cubic function f(x) = ax³ + bx + 1, if the tangent line
    at the point (1, f(1)) has the equation 4x - y - 1 = 0,
    then a + b = 2. -/
theorem cubic_tangent_line (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x + 1
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 + b
  (f' 1 = 4 ∧ f 1 = 3) → a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_tangent_line_l2588_258825


namespace NUMINAMATH_CALUDE_temperature_conversion_l2588_258843

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 20 → k = 68 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2588_258843


namespace NUMINAMATH_CALUDE_alpha_range_l2588_258857

theorem alpha_range (α : Real) :
  (Complex.exp (Complex.I * α) + 2 * Complex.I * Complex.cos α = 2 * Complex.I) ↔
  ∃ k : ℤ, α = 2 * k * Real.pi := by
sorry

end NUMINAMATH_CALUDE_alpha_range_l2588_258857


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l2588_258853

/-- The distance from a point P(-2, 3) to the y-axis is 2. -/
theorem distance_to_y_axis :
  let P : ℝ × ℝ := (-2, 3)
  abs P.1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l2588_258853


namespace NUMINAMATH_CALUDE_apple_pyramid_sum_l2588_258871

/-- Calculates the number of apples in a single layer of the pyramid --/
def layer_apples (base_width : ℕ) (base_length : ℕ) (layer : ℕ) : ℕ :=
  (base_width - layer + 1) * (base_length - layer + 1)

/-- Calculates the total number of apples in the pyramid --/
def total_apples (base_width : ℕ) (base_length : ℕ) : ℕ :=
  (List.range (min base_width base_length)).foldl (λ sum layer => sum + layer_apples base_width base_length layer) 0

theorem apple_pyramid_sum :
  total_apples 6 9 = 154 :=
by sorry

end NUMINAMATH_CALUDE_apple_pyramid_sum_l2588_258871


namespace NUMINAMATH_CALUDE_linda_coin_fraction_l2588_258868

/-- The fraction of Linda's coins representing states that joined the union during 1790-1799 -/
def fraction_of_coins (total_coins : ℕ) (states_joined : ℕ) : ℚ :=
  states_joined / total_coins

/-- Proof that the fraction of Linda's coins representing states from 1790-1799 is 4/15 -/
theorem linda_coin_fraction :
  fraction_of_coins 30 8 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_linda_coin_fraction_l2588_258868


namespace NUMINAMATH_CALUDE_trajectory_and_max_distance_l2588_258800

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the line l: x = 2
def l (x : ℝ) : Prop := x = 2

-- Define the distance ratio condition
def distance_ratio (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  (((x - F.1)^2 + (y - F.2)^2).sqrt / |2 - x|) = Real.sqrt 2 / 2

-- Define the ellipse equation
def ellipse (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  x^2 / 2 + y^2 = 1

-- Define the line for maximum distance calculation
def max_distance_line (x y : ℝ) : Prop :=
  x / Real.sqrt 2 + y = 1

-- Theorem statement
theorem trajectory_and_max_distance :
  -- Part 1: Trajectory is an ellipse
  (∀ M : ℝ × ℝ, distance_ratio M ↔ ellipse M) ∧
  -- Part 2: Maximum distance exists
  (∃ d : ℝ, ∀ M : ℝ × ℝ, ellipse M →
    let (x, y) := M
    abs (x / Real.sqrt 2 + y - 1) / Real.sqrt ((1 / 2) + 1) ≤ d) ∧
  -- Part 3: Maximum distance value
  (let max_d := (2 * Real.sqrt 3 + Real.sqrt 6) / 3
   ∃ M : ℝ × ℝ, ellipse M ∧
     let (x, y) := M
     abs (x / Real.sqrt 2 + y - 1) / Real.sqrt ((1 / 2) + 1) = max_d) :=
by sorry


end NUMINAMATH_CALUDE_trajectory_and_max_distance_l2588_258800


namespace NUMINAMATH_CALUDE_expression_evaluation_l2588_258821

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℤ := -4
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = -11 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2588_258821


namespace NUMINAMATH_CALUDE_total_lockers_l2588_258895

/-- Represents the layout of lockers in a school -/
structure LockerLayout where
  left : ℕ  -- Number of lockers to the left of Yunjeong's locker
  right : ℕ  -- Number of lockers to the right of Yunjeong's locker
  front : ℕ  -- Number of lockers in front of Yunjeong's locker
  back : ℕ  -- Number of lockers behind Yunjeong's locker

/-- Theorem stating the total number of lockers given Yunjeong's locker position -/
theorem total_lockers (layout : LockerLayout) : 
  layout.left = 6 → 
  layout.right = 12 → 
  layout.front = 7 → 
  layout.back = 13 → 
  (layout.left + 1 + layout.right) * (layout.front + 1 + layout.back) = 399 := by
  sorry

#check total_lockers

end NUMINAMATH_CALUDE_total_lockers_l2588_258895


namespace NUMINAMATH_CALUDE_relative_relationship_value_example_max_sum_given_relative_relationship_value_max_sum_achievable_l2588_258815

-- Define the relative relationship value
def relative_relationship_value (a b n : ℚ) : ℚ :=
  |a - n| + |b - n|

-- Theorem 1
theorem relative_relationship_value_example : 
  relative_relationship_value 2 (-5) 2 = 7 := by sorry

-- Theorem 2
theorem max_sum_given_relative_relationship_value :
  ∀ m n : ℚ, relative_relationship_value m n 2 = 2 → 
  m + n ≤ 6 := by sorry

-- Theorem to show that 6 is indeed achievable
theorem max_sum_achievable :
  ∃ m n : ℚ, relative_relationship_value m n 2 = 2 ∧ m + n = 6 := by sorry

end NUMINAMATH_CALUDE_relative_relationship_value_example_max_sum_given_relative_relationship_value_max_sum_achievable_l2588_258815


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2588_258881

universe u

theorem fixed_point_theorem (S : Type u) [Nonempty S] (f : Set S → Set S) 
  (h : ∀ (X Y : Set S), X ⊆ Y → f X ⊆ f Y) :
  ∃ (A : Set S), f A = A := by
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2588_258881


namespace NUMINAMATH_CALUDE_correct_mark_calculation_l2588_258816

theorem correct_mark_calculation (n : ℕ) (initial_avg correct_avg wrong_mark : ℝ) :
  n = 25 →
  initial_avg = 100 →
  wrong_mark = 60 →
  correct_avg = 98 →
  (n : ℝ) * initial_avg - wrong_mark + (n : ℝ) * correct_avg - (n : ℝ) * initial_avg = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_mark_calculation_l2588_258816


namespace NUMINAMATH_CALUDE_apple_basket_theorem_l2588_258877

/-- Represents the capacity of an apple basket -/
structure Basket where
  capacity : ℕ

/-- Represents the current state of Jack's basket -/
structure JackBasket extends Basket where
  current : ℕ
  space_left : ℕ

/-- Theorem about apple baskets -/
theorem apple_basket_theorem (jack : JackBasket) (jill : Basket) : 
  jack.capacity = 12 →
  jack.space_left = 4 →
  jill.capacity = 2 * jack.capacity →
  (jill.capacity / (jack.capacity - jack.space_left) : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_basket_theorem_l2588_258877


namespace NUMINAMATH_CALUDE_min_value_theorem_l2588_258880

/-- Given that f(x) = a^x - b and g(x) = x + 1, where a > 0, a ≠ 1, and b ∈ ℝ,
    if f(x) * g(x) ≤ 0 for all real x, then the minimum value of 1/a + 4/b is 4 -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  (∀ x : ℝ, (a^x - b) * (x + 1) ≤ 0) →
  (∃ m : ℝ, m = 4 ∧ ∀ a b : ℝ, a > 0 → a ≠ 1 → (∀ x : ℝ, (a^x - b) * (x + 1) ≤ 0) → 1/a + 4/b ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2588_258880


namespace NUMINAMATH_CALUDE_stream_current_speed_l2588_258804

/-- Represents the scenario of a rower traveling upstream and downstream -/
structure RowerScenario where
  distance : ℝ
  rower_speed : ℝ
  current_speed : ℝ
  time_diff : ℝ

/-- Represents the scenario when the rower increases their speed -/
structure IncreasedSpeedScenario extends RowerScenario where
  speed_increase : ℝ
  new_time_diff : ℝ

/-- The theorem stating the speed of the stream's current given the conditions -/
theorem stream_current_speed 
  (scenario : RowerScenario)
  (increased : IncreasedSpeedScenario)
  (h1 : scenario.distance = 18)
  (h2 : scenario.time_diff = 4)
  (h3 : increased.speed_increase = 0.5)
  (h4 : increased.new_time_diff = 2)
  (h5 : scenario.distance / (scenario.rower_speed + scenario.current_speed) + scenario.time_diff = 
        scenario.distance / (scenario.rower_speed - scenario.current_speed))
  (h6 : scenario.distance / ((1 + increased.speed_increase) * scenario.rower_speed + scenario.current_speed) + 
        increased.new_time_diff = 
        scenario.distance / ((1 + increased.speed_increase) * scenario.rower_speed - scenario.current_speed))
  : scenario.current_speed = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_current_speed_l2588_258804


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2588_258810

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point through which the line passes
def P : ℝ × ℝ := (2, 0)

-- Define the possible tangent lines
def line1 (x y : ℝ) : Prop := y = 0
def line2 (x y : ℝ) : Prop := 27*x - y - 54 = 0

theorem tangent_line_equation :
  ∃ (m : ℝ), 
    (∀ x y : ℝ, y = m*(x - P.1) + P.2 → 
      (∃ t : ℝ, x = t ∧ y = f t ∧ 
        (∀ s : ℝ, s ≠ t → y - f t < m*(s - t)))) ↔ 
    (line1 x y ∨ line2 x y) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2588_258810


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2588_258822

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7 * I
  let z₂ : ℂ := 4 - 7 * I
  (z₁ / z₂) + (z₂ / z₁) = -66 / 65 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2588_258822


namespace NUMINAMATH_CALUDE_marys_average_speed_l2588_258823

/-- Mary's round trip walk problem -/
theorem marys_average_speed (uphill_distance : ℝ) (downhill_distance : ℝ)
  (uphill_time : ℝ) (downhill_time : ℝ)
  (h1 : uphill_distance = 1.5)
  (h2 : downhill_distance = 1.5)
  (h3 : uphill_time = 45 / 60)  -- Convert 45 minutes to hours
  (h4 : downhill_time = 15 / 60)  -- Convert 15 minutes to hours
  : (uphill_distance + downhill_distance) / (uphill_time + downhill_time) = 3 := by
  sorry

#check marys_average_speed

end NUMINAMATH_CALUDE_marys_average_speed_l2588_258823


namespace NUMINAMATH_CALUDE_faster_walking_speed_l2588_258837

theorem faster_walking_speed 
  (actual_speed : ℝ) 
  (actual_distance : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_speed = 8) 
  (h2 : actual_distance = 40) 
  (h3 : additional_distance = 20) : 
  ∃ (faster_speed : ℝ), 
    faster_speed = (actual_distance + additional_distance) / (actual_distance / actual_speed) ∧ 
    faster_speed = 12 := by
  sorry


end NUMINAMATH_CALUDE_faster_walking_speed_l2588_258837


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2588_258849

/-- The eccentricity of an ellipse with equation x^2 + ky^2 = 3k (k > 0) 
    that shares a focus with the parabola y^2 = 12x is √3/2 -/
theorem ellipse_eccentricity (k : ℝ) (hk : k > 0) : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + k*y^2 = 3*k}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 12*x}
  let ellipse_focus : ℝ × ℝ := (3, 0)  -- Focus of the parabola
  ellipse_focus ∈ ellipse → -- Assuming the focus is on the ellipse
  let a := Real.sqrt (3*k)  -- Semi-major axis
  let c := 3  -- Distance from center to focus
  let e := c / a  -- Eccentricity
  e = Real.sqrt 3 / 2 := by
sorry


end NUMINAMATH_CALUDE_ellipse_eccentricity_l2588_258849


namespace NUMINAMATH_CALUDE_cube_root_8000_simplification_l2588_258812

theorem cube_root_8000_simplification :
  ∃ (a b : ℕ+), (a : ℝ) * (b : ℝ)^(1/3) = 8000^(1/3) ∧ 
  (∀ (c d : ℕ+), (c : ℝ) * (d : ℝ)^(1/3) = 8000^(1/3) → b ≤ d) ∧
  a = 20 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_8000_simplification_l2588_258812


namespace NUMINAMATH_CALUDE_lily_paint_cans_l2588_258898

/-- Given the initial paint coverage, lost cans, and remaining coverage, 
    calculate the number of cans used for the remaining rooms --/
def paint_cans_used (initial_coverage : ℕ) (lost_cans : ℕ) (remaining_coverage : ℕ) : ℕ :=
  (remaining_coverage * lost_cans) / (initial_coverage - remaining_coverage)

/-- Theorem stating that under the given conditions, 16 cans were used for 32 rooms --/
theorem lily_paint_cans : paint_cans_used 40 4 32 = 16 := by
  sorry

end NUMINAMATH_CALUDE_lily_paint_cans_l2588_258898


namespace NUMINAMATH_CALUDE_yellow_second_draw_probability_l2588_258842

/-- Represents the total number of ping-pong balls -/
def total_balls : ℕ := 10

/-- Represents the number of yellow balls -/
def yellow_balls : ℕ := 6

/-- Represents the number of white balls -/
def white_balls : ℕ := 4

/-- Represents the number of draws -/
def num_draws : ℕ := 2

/-- Calculates the probability of drawing a yellow ball on the second draw -/
def prob_yellow_second_draw : ℚ :=
  (white_balls : ℚ) / total_balls * yellow_balls / (total_balls - 1)

theorem yellow_second_draw_probability :
  prob_yellow_second_draw = 4 / 15 := by sorry

end NUMINAMATH_CALUDE_yellow_second_draw_probability_l2588_258842


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l2588_258838

theorem algebraic_expression_equality (x : ℝ) (h : x * (x + 2) = 2023) :
  2 * (x + 3) * (x - 1) - 2018 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l2588_258838


namespace NUMINAMATH_CALUDE_average_of_fifths_and_tenths_l2588_258875

/-- The average of two rational numbers -/
def average (a b : ℚ) : ℚ := (a + b) / 2

/-- Theorem: If the average of 1/5 and 1/10 is 1/x, then x = 20/3 -/
theorem average_of_fifths_and_tenths (x : ℚ) :
  average (1/5) (1/10) = 1/x → x = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_fifths_and_tenths_l2588_258875


namespace NUMINAMATH_CALUDE_knight_2008_winner_condition_l2588_258813

/-- Represents the game where n knights sit at a round table, count 1, 2, 3 clockwise,
    and those who say 2 or 3 are eliminated. -/
def KnightGame (n : ℕ) := True

/-- Predicate indicating whether a knight wins the game -/
def IsWinner (game : KnightGame n) (knight : ℕ) : Prop := True

theorem knight_2008_winner_condition (n : ℕ) :
  (∃ (game : KnightGame n), IsWinner game 2008) ↔
  (∃ (k : ℕ), k ≥ 6 ∧ (n = 1338 + 3^k ∨ n = 1338 + 2 * 3^k)) :=
sorry

end NUMINAMATH_CALUDE_knight_2008_winner_condition_l2588_258813


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2588_258876

/-- Represents a repeating decimal with a single digit repeating -/
def SingleDigitRepeatingDecimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  whole + repeating / 9

/-- Represents a repeating decimal with two digits repeating -/
def TwoDigitRepeatingDecimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  whole + repeating / 99

theorem sum_of_repeating_decimals :
  SingleDigitRepeatingDecimal 0 1 + TwoDigitRepeatingDecimal 0 1 = 4 / 33 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2588_258876


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2588_258874

theorem complex_sum_problem (p q r s t u : ℝ) : 
  q = 5 → 
  t = -p - r → 
  (p + q * I) + (r + s * I) + (t + u * I) = 4 * I → 
  s + u = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2588_258874


namespace NUMINAMATH_CALUDE_first_run_rate_l2588_258870

theorem first_run_rate (first_run_distance : ℝ) (second_run_distance : ℝ) 
  (second_run_rate : ℝ) (total_time : ℝ) :
  first_run_distance = 5 →
  second_run_distance = 4 →
  second_run_rate = 9.5 →
  total_time = 88 →
  first_run_distance * (total_time - second_run_distance * second_run_rate) / first_run_distance = 10 :=
by sorry

end NUMINAMATH_CALUDE_first_run_rate_l2588_258870


namespace NUMINAMATH_CALUDE_least_number_of_marbles_l2588_258829

def is_divisible_by_all (n : ℕ) : Prop :=
  ∀ i ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ), n % i = 0

theorem least_number_of_marbles :
  ∃ (n : ℕ), n > 0 ∧ is_divisible_by_all n ∧ ∀ m, 0 < m ∧ m < n → ¬is_divisible_by_all m :=
by
  use 840
  sorry

end NUMINAMATH_CALUDE_least_number_of_marbles_l2588_258829


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l2588_258806

theorem bowling_team_average_weight 
  (original_players : ℕ) 
  (original_average : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) :
  original_players = 7 →
  original_average = 103 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  let total_weight := original_players * original_average + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  let new_average := total_weight / new_total_players
  new_average = 99 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l2588_258806


namespace NUMINAMATH_CALUDE_probability_of_exact_score_l2588_258802

def num_questions : ℕ := 20
def num_choices : ℕ := 4
def correct_answers : ℕ := 10

def probability_correct : ℚ := 1 / num_choices
def probability_incorrect : ℚ := 1 - probability_correct

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_of_exact_score :
  (binomial_coefficient num_questions correct_answers : ℚ) *
  (probability_correct ^ correct_answers) *
  (probability_incorrect ^ (num_questions - correct_answers)) =
  93350805 / 1073741824 := by sorry

end NUMINAMATH_CALUDE_probability_of_exact_score_l2588_258802


namespace NUMINAMATH_CALUDE_square_value_l2588_258833

theorem square_value : ∃ (square : ℝ), (6400000 : ℝ) / 400 = 1.6 * square ∧ square = 10000 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l2588_258833


namespace NUMINAMATH_CALUDE_problem_solution_l2588_258827

theorem problem_solution : (120 / (6 / 3)) * 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2588_258827


namespace NUMINAMATH_CALUDE_garden_fence_length_l2588_258841

/-- The length of a fence surrounding a square garden -/
def fence_length (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The length of the fence surrounding a square garden with side length 28 meters is 112 meters -/
theorem garden_fence_length :
  fence_length 28 = 112 := by
  sorry

end NUMINAMATH_CALUDE_garden_fence_length_l2588_258841


namespace NUMINAMATH_CALUDE_other_solution_quadratic_equation_l2588_258890

theorem other_solution_quadratic_equation :
  let f (x : ℚ) := 42 * x^2 + 2 * x + 31 - (73 * x + 4)
  (f (3/7) = 0) → (f (3/2) = 0) := by sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_equation_l2588_258890


namespace NUMINAMATH_CALUDE_billys_dime_piles_l2588_258803

/-- Given Billy's coin arrangement, prove the number of dime piles -/
theorem billys_dime_piles 
  (quarter_piles : ℕ) 
  (coins_per_pile : ℕ)
  (total_coins : ℕ)
  (h1 : quarter_piles = 2)
  (h2 : coins_per_pile = 4)
  (h3 : total_coins = 20) :
  (total_coins - quarter_piles * coins_per_pile) / coins_per_pile = 3 :=
by sorry

end NUMINAMATH_CALUDE_billys_dime_piles_l2588_258803


namespace NUMINAMATH_CALUDE_triangle_properties_l2588_258892

open Real

theorem triangle_properties (a b c A B C : ℝ) (h1 : 0 < A) (h2 : A < π) 
  (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π) 
  (h7 : b * cos A = Real.sqrt 3 * a * sin B) (h8 : a = 1) :
  A = π / 6 ∧ 
  (∃ (S : ℝ), S = (2 + Real.sqrt 3) / 4 ∧ 
    ∀ (S' : ℝ), S' = 1 / 2 * b * c * sin A → S' ≤ S) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2588_258892


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2588_258808

theorem arithmetic_mean_of_fractions :
  let a := 3 / 5
  let b := 6 / 7
  let c := 4 / 5
  let arithmetic_mean := (a + b) / 2
  (arithmetic_mean ≠ c) ∧ (arithmetic_mean = 51 / 70) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2588_258808


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l2588_258889

theorem product_of_roots_quadratic (x : ℝ) : 
  (8 = -2 * x^2 - 6 * x) → (∃ α β : ℝ, (α * β = 4 ∧ 8 = -2 * α^2 - 6 * α ∧ 8 = -2 * β^2 - 6 * β)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_l2588_258889


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2588_258862

theorem quadratic_roots_condition (a : ℝ) : 
  (-1 < a ∧ a < 1) → 
  (∃ x₁ x₂ : ℝ, x₁ * x₂ = a - 2 ∧ x₁ + x₂ = -(a + 1) ∧ x₁ > 0 ∧ x₂ < 0) ∧
  ¬(∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ * x₂ = a - 2 ∧ x₁ + x₂ = -(a + 1) ∧ x₁ > 0 ∧ x₂ < 0) → (-1 < a ∧ a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2588_258862


namespace NUMINAMATH_CALUDE_circumcenter_rational_coords_l2588_258832

/-- Given a triangle with rational coordinates, its circumcenter has rational coordinates. -/
theorem circumcenter_rational_coords 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℚ) :
  ∃ (x y : ℚ), 
    (x - a₁)^2 + (y - b₁)^2 = (x - a₂)^2 + (y - b₂)^2 ∧
    (x - a₁)^2 + (y - b₁)^2 = (x - a₃)^2 + (y - b₃)^2 :=
by sorry

end NUMINAMATH_CALUDE_circumcenter_rational_coords_l2588_258832


namespace NUMINAMATH_CALUDE_regression_estimate_l2588_258872

/-- Represents a linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Represents a data point -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Parameters of the regression problem -/
structure RegressionProblem where
  original_regression : LinearRegression
  original_mean_x : ℝ
  removed_points : List DataPoint
  new_slope : ℝ

theorem regression_estimate (problem : RegressionProblem) :
  let new_intercept := problem.original_regression.intercept +
    problem.original_regression.slope * problem.original_mean_x -
    problem.new_slope * problem.original_mean_x
  let new_regression := LinearRegression.mk problem.new_slope new_intercept
  let estimate_at_6 := new_regression.slope * 6 + new_regression.intercept
  problem.original_regression = LinearRegression.mk 1.5 1 →
  problem.original_mean_x = 2 →
  problem.removed_points = [DataPoint.mk 2.6 2.8, DataPoint.mk 1.4 5.2] →
  problem.new_slope = 1.4 →
  estimate_at_6 = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_regression_estimate_l2588_258872


namespace NUMINAMATH_CALUDE_bounded_sequence_l2588_258840

/-- A sequence defined recursively with a parameter c -/
def x (c : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => (x c n)^2 + c

/-- The theorem stating the condition for boundedness of the sequence -/
theorem bounded_sequence (c : ℝ) (h : c > 0) :
  (∀ n, |x c n| < 2016) ↔ c ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_bounded_sequence_l2588_258840


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l2588_258899

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.05 * L) (h2 : B' * L' = 1.2075 * (B * L)) : B' = 1.15 * B := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l2588_258899


namespace NUMINAMATH_CALUDE_number_puzzle_l2588_258882

theorem number_puzzle (x : ℤ) : x - 13 = 31 → x + 11 = 55 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2588_258882


namespace NUMINAMATH_CALUDE_fourth_game_shots_correct_l2588_258867

-- Define the initial conditions
def initial_shots : ℕ := 45
def initial_made : ℕ := 18
def initial_average : ℚ := 40 / 100
def fourth_game_shots : ℕ := 15
def new_average : ℚ := 55 / 100

-- Define the function to calculate the number of shots made in the fourth game
def fourth_game_made : ℕ := 15

-- Theorem statement
theorem fourth_game_shots_correct :
  (initial_made + fourth_game_made : ℚ) / (initial_shots + fourth_game_shots) = new_average :=
by sorry

end NUMINAMATH_CALUDE_fourth_game_shots_correct_l2588_258867


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2588_258835

/-- A rhombus with area K and one diagonal three times the length of the other has side length √(5K/3) -/
theorem rhombus_side_length (K : ℝ) (K_pos : K > 0) : 
  ∃ (d₁ d₂ s : ℝ), d₁ > 0 ∧ d₂ > 0 ∧ s > 0 ∧ 
  d₂ = 3 * d₁ ∧ 
  K = (1/2) * d₁ * d₂ ∧
  s^2 = (d₁/2)^2 + (d₂/2)^2 ∧
  s = Real.sqrt ((5 * K) / 3) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2588_258835


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2588_258848

-- System (1)
theorem system_one_solution (x y : ℝ) : 
  x + y = 1 ∧ 3 * x + y = 5 → x = 2 ∧ y = -1 := by sorry

-- System (2)
theorem system_two_solution (x y : ℝ) : 
  3 * (x - 1) + 4 * y = 1 ∧ 2 * x + 3 * (y + 1) = 2 → x = 16 ∧ y = -11 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2588_258848


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2588_258801

theorem quadratic_factorization (m n : ℤ) : 
  (∀ x, x^2 - 7*x + n = (x - 3) * (x + m)) → m - n = -16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2588_258801


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_angle_in_range_same_terminal_side_750_l2588_258883

theorem same_terminal_side_angle : ℤ → ℝ → ℝ
  | k, α => k * 360 + α

theorem angle_in_range (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 360

theorem same_terminal_side_750 :
  ∃ (θ : ℝ), angle_in_range θ ∧ ∃ (k : ℤ), same_terminal_side_angle k θ = 750 ∧ θ = 30 :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_angle_in_range_same_terminal_side_750_l2588_258883


namespace NUMINAMATH_CALUDE_hibiscus_flower_ratio_l2588_258855

/-- Given Mario's hibiscus plants, prove the ratio of flowers on the third to second plant -/
theorem hibiscus_flower_ratio :
  let first_plant_flowers : ℕ := 2
  let second_plant_flowers : ℕ := 2 * first_plant_flowers
  let total_flowers : ℕ := 22
  let third_plant_flowers : ℕ := total_flowers - first_plant_flowers - second_plant_flowers
  third_plant_flowers / second_plant_flowers = 4 := by
sorry

end NUMINAMATH_CALUDE_hibiscus_flower_ratio_l2588_258855


namespace NUMINAMATH_CALUDE_smallest_positive_e_value_l2588_258888

theorem smallest_positive_e_value (a b c d e : ℤ) : 
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    (x = -3 ∨ x = 7 ∨ x = 8 ∨ x = -1/4)) →
  (∀ e' : ℤ, e' > 0 → e' ≥ 168) →
  e = 168 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_e_value_l2588_258888


namespace NUMINAMATH_CALUDE_number_multiplication_l2588_258830

theorem number_multiplication (x : ℝ) : x / 0.4 = 8 → x * 0.4 = 1.28 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_l2588_258830


namespace NUMINAMATH_CALUDE_ones_digit_of_9_to_53_l2588_258859

theorem ones_digit_of_9_to_53 : Nat.mod (9^53) 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_9_to_53_l2588_258859


namespace NUMINAMATH_CALUDE_sum_equals_product_implies_two_greater_than_one_l2588_258831

theorem sum_equals_product_implies_two_greater_than_one 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum_prod : a + b + c = a * b * c) : 
  (a > 1 ∧ b > 1) ∨ (a > 1 ∧ c > 1) ∨ (b > 1 ∧ c > 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_product_implies_two_greater_than_one_l2588_258831


namespace NUMINAMATH_CALUDE_problem_statements_l2588_258847

noncomputable section

variable (k : ℝ)

def f (x : ℝ) : ℝ := Real.log x

def g (x : ℝ) : ℝ := x^2 + k*x

def a (x₁ x₂ : ℝ) : ℝ := (f x₁ - f x₂) / (x₁ - x₂)

def b (x₁ x₂ : ℝ) : ℝ := (g k x₁ - g k x₂) / (x₁ - x₂)

theorem problem_statements :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → a x₁ x₂ > 0) ∧
  (∃ k : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ b k x₁ x₂ ≤ 0) ∧
  (∀ k : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ b k x₁ x₂ / a x₁ x₂ = 2) ∧
  (∀ k : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ b k x₁ x₂ / a x₁ x₂ = -2) → k < -4) :=
by sorry

end

end NUMINAMATH_CALUDE_problem_statements_l2588_258847


namespace NUMINAMATH_CALUDE_simplify_expression_l2588_258860

theorem simplify_expression (a : ℝ) :
  (((a ^ 16) ^ (1 / 8)) ^ (1 / 4)) ^ 3 * (((a ^ 16) ^ (1 / 4)) ^ (1 / 8)) ^ 3 = a ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2588_258860


namespace NUMINAMATH_CALUDE_birthday_1200th_day_l2588_258839

/-- Given a person born on a Monday, their 1200th day of life will fall on a Thursday. -/
theorem birthday_1200th_day : 
  ∀ (birth_day : Nat), 
  birth_day % 7 = 1 →  -- Monday is represented as 1 (1-based indexing for days of week)
  (birth_day + 1200) % 7 = 5  -- Thursday is represented as 5
  := by sorry

end NUMINAMATH_CALUDE_birthday_1200th_day_l2588_258839


namespace NUMINAMATH_CALUDE_sum_equals_result_l2588_258887

-- Define the sum
def sum : ℚ := 10/9 + 9/10

-- Define the result as a rational number (2 + 1/10)
def result : ℚ := 2 + 1/10

-- Theorem stating that the sum equals the result
theorem sum_equals_result : sum = result := by sorry

end NUMINAMATH_CALUDE_sum_equals_result_l2588_258887


namespace NUMINAMATH_CALUDE_handshake_count_l2588_258814

theorem handshake_count (n : ℕ) (couples : ℕ) (extra_exemptions : ℕ) : 
  n = 2 * couples → 
  n ≥ 2 →
  extra_exemptions ≤ n - 2 →
  (n * (n - 2) - extra_exemptions) / 2 = 57 :=
by
  sorry

#check handshake_count 12 6 2

end NUMINAMATH_CALUDE_handshake_count_l2588_258814
