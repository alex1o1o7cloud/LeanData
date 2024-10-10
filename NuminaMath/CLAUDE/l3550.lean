import Mathlib

namespace simplify_sqrt_fraction_simplify_sqrt_difference_squares_simplify_sqrt_fraction_product_simplify_sqrt_decimal_l3550_355061

-- Part 1
theorem simplify_sqrt_fraction : (1/2) * Real.sqrt (4/7) = Real.sqrt 7 / 7 := by sorry

-- Part 2
theorem simplify_sqrt_difference_squares : Real.sqrt (20^2 - 15^2) = 5 * Real.sqrt 7 := by sorry

-- Part 3
theorem simplify_sqrt_fraction_product : Real.sqrt ((32 * 9) / 25) = (12 * Real.sqrt 2) / 5 := by sorry

-- Part 4
theorem simplify_sqrt_decimal : Real.sqrt 22.5 = (3 * Real.sqrt 10) / 2 := by sorry

end simplify_sqrt_fraction_simplify_sqrt_difference_squares_simplify_sqrt_fraction_product_simplify_sqrt_decimal_l3550_355061


namespace election_votes_total_l3550_355082

theorem election_votes_total (winner_percentage : ℚ) (vote_majority : ℕ) : 
  winner_percentage = 7/10 →
  vote_majority = 280 →
  ∃ (total_votes : ℕ), 
    (winner_percentage * total_votes : ℚ) - ((1 - winner_percentage) * total_votes : ℚ) = vote_majority ∧
    total_votes = 700 := by
  sorry

end election_votes_total_l3550_355082


namespace quadratic_roots_and_inequality_l3550_355083

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - 2*a*x + a^2 + a - 2

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, quadratic a x = 0

-- Define the proposition q
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 3

theorem quadratic_roots_and_inequality (a m : ℝ) :
  (¬ p a → a > 2) ∧
  ((∀ m, p a → q m a) ∧ (∃ m, q m a ∧ ¬ p a) → m ≤ -1) :=
sorry

end quadratic_roots_and_inequality_l3550_355083


namespace value_of_a_l3550_355036

theorem value_of_a (a : ℝ) (h : a + a/4 = 5/2) : a = 2 := by
  sorry

end value_of_a_l3550_355036


namespace existence_of_x0_l3550_355053

theorem existence_of_x0 (a b : ℝ) : ∃ x0 : ℝ, x0 ∈ Set.Icc 1 9 ∧ |a * x0 + b + 9 / x0| ≥ 2 := by
  sorry

end existence_of_x0_l3550_355053


namespace shirt_ratio_l3550_355095

theorem shirt_ratio : 
  ∀ (steven andrew brian : ℕ),
  steven = 4 * andrew →
  brian = 3 →
  steven = 72 →
  andrew / brian = 6 :=
by
  sorry

end shirt_ratio_l3550_355095


namespace cheese_block_volume_l3550_355034

/-- Given a normal block of cheese with volume 3 cubic feet, 
    a large block with twice the width, twice the depth, and three times the length 
    of the normal block will have a volume of 36 cubic feet. -/
theorem cheese_block_volume : 
  ∀ (w d l : ℝ), 
    w * d * l = 3 → 
    (2 * w) * (2 * d) * (3 * l) = 36 := by
  sorry

end cheese_block_volume_l3550_355034


namespace perimeter_diagonal_ratio_bounds_l3550_355099

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Add appropriate convexity condition

/-- The perimeter of a convex quadrilateral -/
def perimeter (q : ConvexQuadrilateral) : ℝ := sorry

/-- The sum of diagonal lengths of a convex quadrilateral -/
def diagonalSum (q : ConvexQuadrilateral) : ℝ := sorry

/-- Theorem: The ratio of perimeter to diagonal sum is strictly between 1 and 2 -/
theorem perimeter_diagonal_ratio_bounds (q : ConvexQuadrilateral) :
  1 < perimeter q / diagonalSum q ∧ perimeter q / diagonalSum q < 2 := by
  sorry

end perimeter_diagonal_ratio_bounds_l3550_355099


namespace q_investment_correct_l3550_355028

/-- Represents the investment of two people in a business -/
structure Business where
  p_investment : ℕ
  q_investment : ℕ
  profit_ratio : Rat

/-- The business scenario with given conditions -/
def given_business : Business where
  p_investment := 40000
  q_investment := 60000
  profit_ratio := 2 / 3

/-- Theorem stating that q's investment is correct given the conditions -/
theorem q_investment_correct (b : Business) : 
  b.p_investment = 40000 ∧ 
  b.profit_ratio = 2 / 3 → 
  b.q_investment = 60000 := by
  sorry

#check q_investment_correct given_business

end q_investment_correct_l3550_355028


namespace second_player_winning_strategy_l3550_355046

/-- A game on a circle where two players mark points -/
structure CircleGame where
  /-- The number of points each player marks -/
  p : ℕ
  /-- Condition that p is greater than 1 -/
  p_gt_one : p > 1

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins
  | Draw

/-- A strategy for playing the game -/
def Strategy := CircleGame → GameResult

/-- The theorem stating that the second player has a winning strategy -/
theorem second_player_winning_strategy (game : CircleGame) : 
  ∃ (s : Strategy), ∀ (opponent_strategy : Strategy), 
    s game = GameResult.SecondPlayerWins :=
sorry

end second_player_winning_strategy_l3550_355046


namespace complex_arithmetic_evaluation_l3550_355031

theorem complex_arithmetic_evaluation : 2 - 2 * (2 - 2 * (2 - 2 * (4 - 2))) = -10 := by
  sorry

end complex_arithmetic_evaluation_l3550_355031


namespace emily_egg_collection_l3550_355029

/-- The number of baskets Emily used -/
def num_baskets : ℕ := 303

/-- The number of eggs in each basket -/
def eggs_per_basket : ℕ := 28

/-- The total number of eggs Emily collected -/
def total_eggs : ℕ := num_baskets * eggs_per_basket

theorem emily_egg_collection : total_eggs = 8484 := by
  sorry

end emily_egg_collection_l3550_355029


namespace unique_assignment_l3550_355020

/- Define the girls and colors as enums -/
inductive Girl : Type
  | Katya | Olya | Liza | Rita

inductive Color : Type
  | Pink | Green | Yellow | Blue

/- Define the assignment of colors to girls -/
def assignment : Girl → Color
  | Girl.Katya => Color.Green
  | Girl.Olya => Color.Blue
  | Girl.Liza => Color.Pink
  | Girl.Rita => Color.Yellow

/- Define the circular arrangement of girls -/
def nextGirl : Girl → Girl
  | Girl.Katya => Girl.Olya
  | Girl.Olya => Girl.Liza
  | Girl.Liza => Girl.Rita
  | Girl.Rita => Girl.Katya

/- Define the conditions -/
def conditions (a : Girl → Color) : Prop :=
  (a Girl.Katya ≠ Color.Pink ∧ a Girl.Katya ≠ Color.Blue) ∧
  (∃ g : Girl, a g = Color.Green ∧ 
    ((nextGirl g = Girl.Liza ∧ a (nextGirl (nextGirl g)) = Color.Yellow) ∨
     (nextGirl (nextGirl g) = Girl.Liza ∧ a (nextGirl g) = Color.Yellow))) ∧
  (a Girl.Rita ≠ Color.Green ∧ a Girl.Rita ≠ Color.Blue) ∧
  (∃ g : Girl, nextGirl g = Girl.Olya ∧ nextGirl (nextGirl g) = Girl.Rita ∧ 
    (a g = Color.Pink ∨ a (nextGirl (nextGirl (nextGirl g))) = Color.Pink))

/- Theorem statement -/
theorem unique_assignment : 
  ∀ a : Girl → Color, conditions a → a = assignment :=
sorry

end unique_assignment_l3550_355020


namespace living_room_area_l3550_355035

/-- Given a rectangular carpet covering 60% of a room's floor area,
    if the carpet measures 4 feet by 9 feet,
    then the total floor area of the room is 60 square feet. -/
theorem living_room_area
  (carpet_length : ℝ)
  (carpet_width : ℝ)
  (carpet_coverage : ℝ)
  (h1 : carpet_length = 4)
  (h2 : carpet_width = 9)
  (h3 : carpet_coverage = 0.6)
  : carpet_length * carpet_width / carpet_coverage = 60 := by
  sorry

end living_room_area_l3550_355035


namespace f_triple_composition_equals_self_l3550_355059

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 3 else n / 2

theorem f_triple_composition_equals_self (k : ℤ) :
  k % 2 = 1 → (f (f (f k)) = k ↔ k = 1) := by
  sorry

end f_triple_composition_equals_self_l3550_355059


namespace last_two_nonzero_digits_of_80_factorial_l3550_355043

theorem last_two_nonzero_digits_of_80_factorial (n : ℕ) : n = 80 → 
  ∃ k : ℕ, n.factorial = 100 * k + 12 ∧ k % 10 ≠ 0 :=
by sorry

end last_two_nonzero_digits_of_80_factorial_l3550_355043


namespace circumcircle_diameter_l3550_355051

theorem circumcircle_diameter (a b c : ℝ) (θ : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : Real.cos θ = 1/3) :
  let d := max a (max b c)
  2 * d / Real.sin θ = 9 * Real.sqrt 2 / 4 :=
by sorry

end circumcircle_diameter_l3550_355051


namespace quadratic_always_positive_l3550_355019

theorem quadratic_always_positive (b c : ℤ) 
  (h : ∀ x : ℤ, (x^2 : ℤ) + b*x + c > 0) : 
  b^2 - 4*c ≤ 0 := by
sorry

end quadratic_always_positive_l3550_355019


namespace course_total_hours_l3550_355010

/-- Calculates the total hours spent on a course given the course duration and weekly schedule. -/
def total_course_hours (weeks : ℕ) (class_hours_1 class_hours_2 class_hours_3 homework_hours : ℕ) : ℕ :=
  weeks * (class_hours_1 + class_hours_2 + class_hours_3 + homework_hours)

/-- Proves that a 24-week course with the given weekly schedule results in 336 total hours. -/
theorem course_total_hours :
  total_course_hours 24 3 3 4 4 = 336 := by
  sorry

end course_total_hours_l3550_355010


namespace inequality_holds_iff_m_in_range_l3550_355081

theorem inequality_holds_iff_m_in_range (m : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3*m) ↔ 
  (m ≤ -1 ∨ m ≥ 4) := by
sorry

end inequality_holds_iff_m_in_range_l3550_355081


namespace complete_square_k_value_l3550_355007

/-- A quadratic expression can be factored using the complete square formula if and only if
    it can be written in the form (x + a)^2 for some real number a. --/
def is_complete_square (k : ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x^2 + k*x + 9 = (x + a)^2

/-- If x^2 + kx + 9 can be factored using the complete square formula,
    then k = 6 or k = -6. --/
theorem complete_square_k_value (k : ℝ) :
  is_complete_square k → k = 6 ∨ k = -6 := by
  sorry

end complete_square_k_value_l3550_355007


namespace even_function_symmetry_l3550_355084

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def has_min_value_on (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

theorem even_function_symmetry (f : ℝ → ℝ) :
  is_even_function f →
  is_increasing_on f 3 7 →
  has_min_value_on f 3 7 2 →
  is_decreasing_on f (-7) (-3) ∧ has_min_value_on f (-7) (-3) 2 :=
sorry

end even_function_symmetry_l3550_355084


namespace constant_sum_of_roots_l3550_355037

theorem constant_sum_of_roots (b x : ℝ) (h : (6 / b) < x ∧ x < (10 / b)) :
  Real.sqrt (x^2 - 2*x + 1) + Real.sqrt (x^2 - 6*x + 9) = 2 := by
  sorry

end constant_sum_of_roots_l3550_355037


namespace candy_problem_l3550_355021

/-- The number of candy pieces remaining in a bowl after some are taken. -/
def remaining_candy (initial : ℕ) (taken_by_talitha : ℕ) (taken_by_solomon : ℕ) : ℕ :=
  initial - (taken_by_talitha + taken_by_solomon)

/-- Theorem stating that given the initial amount and amounts taken by Talitha and Solomon,
    the remaining candy pieces are 88. -/
theorem candy_problem :
  remaining_candy 349 108 153 = 88 := by
  sorry

end candy_problem_l3550_355021


namespace complex_equation_solution_l3550_355071

/-- Given that i is the imaginary unit and z is a complex number defined as
    z = ((1+i)^2 + 3(1-i)) / (2+i), prove that if z^2 + az + b = 1 + i
    where a and b are real numbers, then a = -3 and b = 4. -/
theorem complex_equation_solution (i : ℂ) (a b : ℝ) :
  i^2 = -1 →
  let z : ℂ := ((1 + i)^2 + 3*(1 - i)) / (2 + i)
  z^2 + a*z + b = 1 + i →
  a = -3 ∧ b = 4 := by
  sorry

end complex_equation_solution_l3550_355071


namespace train_length_l3550_355067

/-- The length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 55 * (1000 / 3600) →
  platform_length = 520 →
  crossing_time = 64.79481641468682 →
  (train_speed * crossing_time) - platform_length = 470 := by
  sorry

end train_length_l3550_355067


namespace coin_flip_expected_value_l3550_355054

/-- Calculates the expected value of a coin flip experiment -/
def expected_value (coin_values : List ℚ) (probability : ℚ) : ℚ :=
  (coin_values.sum * probability)

/-- The main theorem: expected value of the coin flip experiment -/
theorem coin_flip_expected_value :
  let coin_values : List ℚ := [1, 5, 10, 50, 100]
  let probability : ℚ := 1/2
  expected_value coin_values probability = 83 := by
  sorry

end coin_flip_expected_value_l3550_355054


namespace grocery_problem_l3550_355048

theorem grocery_problem (total_packs cookie_packs : ℕ) 
  (h1 : total_packs = 27)
  (h2 : cookie_packs = 23)
  (h3 : total_packs = cookie_packs + cake_packs) :
  cake_packs = 4 := by
  sorry

end grocery_problem_l3550_355048


namespace sqrt_sum_equals_seven_sqrt_three_over_three_l3550_355000

theorem sqrt_sum_equals_seven_sqrt_three_over_three :
  Real.sqrt 12 + Real.sqrt (1/3) = 7 * Real.sqrt 3 / 3 := by
  sorry

end sqrt_sum_equals_seven_sqrt_three_over_three_l3550_355000


namespace pears_picked_total_l3550_355032

/-- The number of pears Keith picked -/
def keith_pears : ℕ := 3

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 2

/-- The total number of pears picked -/
def total_pears : ℕ := keith_pears + jason_pears

theorem pears_picked_total :
  total_pears = 5 := by sorry

end pears_picked_total_l3550_355032


namespace vector_difference_magnitude_l3550_355093

/-- Given two 2D vectors a and b, prove that the magnitude of their difference is 5. -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by sorry

end vector_difference_magnitude_l3550_355093


namespace y_coordinate_of_P_l3550_355026

/-- A line through the origin equidistant from two points -/
structure EquidistantLine where
  slope : ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  origin_line : slope * P.1 = P.2 ∧ slope * Q.1 = Q.2
  equidistant : (P.1 - 0)^2 + (P.2 - 0)^2 = (Q.1 - 0)^2 + (Q.2 - 0)^2

/-- Theorem: Given the conditions, the y-coordinate of P is 3.2 -/
theorem y_coordinate_of_P (L : EquidistantLine)
  (h_slope : L.slope = 0.8)
  (h_x_coord : L.P.1 = 4) :
  L.P.2 = 3.2 := by
  sorry

end y_coordinate_of_P_l3550_355026


namespace daniels_improvement_l3550_355003

/-- Represents the jogging data for Daniel -/
structure JoggingData where
  initial_laps : ℕ
  initial_time : ℕ  -- in minutes
  final_laps : ℕ
  final_time : ℕ    -- in minutes

/-- Calculates the improvement in lap time (in seconds) given jogging data -/
def lapTimeImprovement (data : JoggingData) : ℕ :=
  let initial_lap_time := (data.initial_time * 60) / data.initial_laps
  let final_lap_time := (data.final_time * 60) / data.final_laps
  initial_lap_time - final_lap_time

/-- Theorem stating that Daniel's lap time improvement is 20 seconds -/
theorem daniels_improvement (data : JoggingData) 
  (h1 : data.initial_laps = 15) 
  (h2 : data.initial_time = 40)
  (h3 : data.final_laps = 18)
  (h4 : data.final_time = 42) : 
  lapTimeImprovement data = 20 := by
  sorry

end daniels_improvement_l3550_355003


namespace percentage_against_proposal_l3550_355064

def total_votes : ℕ := 290
def vote_difference : ℕ := 58

theorem percentage_against_proposal :
  let votes_against := (total_votes - vote_difference) / 2
  (votes_against : ℚ) / total_votes * 100 = 40 := by
  sorry

end percentage_against_proposal_l3550_355064


namespace total_seashells_l3550_355074

theorem total_seashells (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) 
  (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
  sorry

end total_seashells_l3550_355074


namespace farm_harvest_after_26_days_l3550_355069

/-- Represents the daily harvest rates for a fruit farm -/
structure HarvestRates where
  ripeOrangesOdd : ℕ
  unripeOrangesOdd : ℕ
  ripeOrangesEven : ℕ
  unripeOrangesEven : ℕ
  ripeApples : ℕ
  unripeApples : ℕ

/-- Calculates the total harvest for a given number of days -/
def totalHarvest (rates : HarvestRates) (days : ℕ) :
  ℕ × ℕ × ℕ × ℕ :=
  let oddDays := (days + 1) / 2
  let evenDays := days / 2
  ( oddDays * rates.ripeOrangesOdd + evenDays * rates.ripeOrangesEven
  , oddDays * rates.unripeOrangesOdd + evenDays * rates.unripeOrangesEven
  , days * rates.ripeApples
  , days * rates.unripeApples
  )

/-- The main theorem stating the total harvest after 26 days -/
theorem farm_harvest_after_26_days (rates : HarvestRates)
  (h1 : rates.ripeOrangesOdd = 32)
  (h2 : rates.unripeOrangesOdd = 46)
  (h3 : rates.ripeOrangesEven = 28)
  (h4 : rates.unripeOrangesEven = 52)
  (h5 : rates.ripeApples = 50)
  (h6 : rates.unripeApples = 30) :
  totalHarvest rates 26 = (780, 1274, 1300, 780) := by
  sorry

end farm_harvest_after_26_days_l3550_355069


namespace smallest_valid_number_l3550_355097

-- Define the property of being a nonprime integer greater than 1 with no prime factor less than 15
def is_valid_number (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n ∧ ∀ p : ℕ, Nat.Prime p → p < 15 → ¬ p ∣ n

-- State the theorem
theorem smallest_valid_number :
  ∃ n : ℕ, is_valid_number n ∧ ∀ m : ℕ, is_valid_number m → n ≤ m :=
sorry

end smallest_valid_number_l3550_355097


namespace arithmetic_sequence_property_l3550_355022

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (m n p : ℕ) 
  (h_arith : ArithmeticSequence a)
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_positive : 0 < m ∧ 0 < n ∧ 0 < p) :
  m * (a p - a n) + n * (a m - a p) + p * (a n - a m) = 0 :=
sorry

end arithmetic_sequence_property_l3550_355022


namespace amy_picture_files_l3550_355060

theorem amy_picture_files (music_files : ℝ) (video_files : ℝ) (total_files : ℕ) : 
  music_files = 4.0 →
  video_files = 21.0 →
  total_files = 48 →
  (total_files : ℝ) - (music_files + video_files) = 23 := by
sorry

end amy_picture_files_l3550_355060


namespace isosceles_triangle_base_angles_l3550_355091

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b) ∨ (b = c) ∨ (a = c)
  sumOfAngles : a + b + c = 180

-- Define the condition of angle ratio
def hasAngleRatio (t : IsoscelesTriangle) : Prop :=
  (t.a = 2 * t.b) ∨ (t.b = 2 * t.c) ∨ (t.a = 2 * t.c)

-- Theorem statement
theorem isosceles_triangle_base_angles 
  (t : IsoscelesTriangle) 
  (h : hasAngleRatio t) : 
  (t.a = 45 ∧ t.b = 45) ∨ (t.b = 72 ∧ t.c = 72) ∨ (t.a = 72 ∧ t.c = 72) :=
sorry

end isosceles_triangle_base_angles_l3550_355091


namespace gratuity_calculation_l3550_355002

def dish_price_1 : ℝ := 10
def dish_price_2 : ℝ := 13
def dish_price_3 : ℝ := 17
def tip_percentage : ℝ := 0.1

theorem gratuity_calculation : 
  (dish_price_1 + dish_price_2 + dish_price_3) * tip_percentage = 4 := by
sorry

end gratuity_calculation_l3550_355002


namespace rectangular_prism_surface_area_l3550_355080

theorem rectangular_prism_surface_area
  (r : ℝ) (l w h : ℝ) 
  (h_r : r = 3 * (36 / Real.pi))
  (h_l : l = 6)
  (h_w : w = 4)
  (h_vol_eq : (4 / 3) * Real.pi * r^3 = l * w * h) :
  2 * (l * w + l * h + w * h) = 88 := by
sorry

end rectangular_prism_surface_area_l3550_355080


namespace product_mod_five_l3550_355058

theorem product_mod_five : 2011 * 2012 * 2013 * 2014 % 5 = 4 := by
  sorry

end product_mod_five_l3550_355058


namespace sum_of_coefficients_l3550_355012

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^3 + 3*x^2 + 2*x > 0}
def B (a b : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem sum_of_coefficients (a b : ℝ) :
  (A ∩ B a b = {x : ℝ | 0 < x ∧ x ≤ 2}) ∧
  (A ∪ B a b = {x : ℝ | x > -2}) →
  a + b = -3 := by
sorry

end sum_of_coefficients_l3550_355012


namespace fraction_simplification_l3550_355088

theorem fraction_simplification : 3 / (2 - 3/4) = 12/5 := by
  sorry

end fraction_simplification_l3550_355088


namespace swimmer_speed_in_still_water_l3550_355044

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water (km/h)
  stream : ℝ   -- Speed of the stream (km/h)

/-- Calculates the effective speed when swimming downstream. -/
def downstreamSpeed (s : SwimmerSpeed) : ℝ := s.swimmer + s.stream

/-- Calculates the effective speed when swimming upstream. -/
def upstreamSpeed (s : SwimmerSpeed) : ℝ := s.swimmer - s.stream

/-- Theorem stating that given the conditions of the swimming problem, 
    the swimmer's speed in still water is 7 km/h. -/
theorem swimmer_speed_in_still_water : 
  ∀ s : SwimmerSpeed, 
    downstreamSpeed s = 36 / 6 → 
    upstreamSpeed s = 48 / 6 → 
    s.swimmer = 7 := by
  sorry

#check swimmer_speed_in_still_water

end swimmer_speed_in_still_water_l3550_355044


namespace spider_journey_l3550_355063

theorem spider_journey (r : ℝ) (final_leg : ℝ) : r = 75 ∧ final_leg = 90 →
  2 * r + r + final_leg = 315 := by
  sorry

end spider_journey_l3550_355063


namespace odd_function_extension_l3550_355062

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the properties of the function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_extension {f : ℝ → ℝ} 
  (h_odd : is_odd_function f)
  (h_pos : ∀ x > 0, f x = lg (x + 1)) :
  ∀ x < 0, f x = -lg (-x + 1) := by
  sorry

end odd_function_extension_l3550_355062


namespace three_integer_solutions_l3550_355045

theorem three_integer_solutions (n : ℕ) (x₁ y₁ : ℤ) 
  (h : x₁^3 - 3*x₁*y₁^2 + y₁^3 = n) : 
  ∃ (x₂ y₂ x₃ y₃ : ℤ), 
    (x₂^3 - 3*x₂*y₂^2 + y₂^3 = n) ∧ 
    (x₃^3 - 3*x₃*y₃^2 + y₃^3 = n) ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ 
    (x₂ ≠ x₃ ∨ y₂ ≠ y₃) := by
  sorry

end three_integer_solutions_l3550_355045


namespace platform_length_l3550_355039

/-- Given a train of length 300 meters that takes 42 seconds to cross a platform
    and 18 seconds to cross a signal pole, prove that the length of the platform
    is approximately 400.14 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 42)
  (h3 : pole_crossing_time = 18) :
  ∃ (platform_length : ℝ), abs (platform_length - 400.14) < 0.01 := by
  sorry

end platform_length_l3550_355039


namespace square_hexagon_area_l3550_355087

theorem square_hexagon_area (s : ℝ) (square_area : ℝ) (hexagon_area : ℝ) :
  square_area = Real.sqrt 3 →
  square_area = s ^ 2 →
  hexagon_area = 3 * Real.sqrt 3 * s ^ 2 / 2 →
  hexagon_area = 9 / 2 :=
by
  sorry

end square_hexagon_area_l3550_355087


namespace mountain_height_theorem_l3550_355015

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the problem setup -/
structure MountainProblem where
  A : Point3D
  C : Point3D
  P : Point3D
  F : Point3D
  AC_distance : ℝ
  AP_distance : ℝ
  C_elevation : ℝ
  AC_angle : ℝ
  AP_angle : ℝ
  magnetic_declination : ℝ
  latitude : ℝ

/-- The main theorem to prove -/
theorem mountain_height_theorem (problem : MountainProblem) :
  problem.AC_distance = 2200 →
  problem.AP_distance = 400 →
  problem.C_elevation = 550 →
  problem.AC_angle = 71 →
  problem.AP_angle = 64 →
  problem.magnetic_declination = 2 →
  problem.latitude = 49 →
  ∃ (height : ℝ), abs (height - 420) < 1 ∧ height = problem.A.z :=
sorry


end mountain_height_theorem_l3550_355015


namespace smallest_square_cover_l3550_355065

/-- The width of the rectangle -/
def rectangle_width : ℕ := 3

/-- The height of the rectangle -/
def rectangle_height : ℕ := 4

/-- The area of a single rectangle -/
def rectangle_area : ℕ := rectangle_width * rectangle_height

/-- The side length of the smallest square that can be covered exactly by the rectangles -/
def square_side : ℕ := lcm rectangle_width rectangle_height

/-- The area of the square -/
def square_area : ℕ := square_side * square_side

/-- The number of rectangles needed to cover the square -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem smallest_square_cover :
  num_rectangles = 12 ∧
  ∀ n : ℕ, n < square_side → ¬(n * n % rectangle_area = 0) :=
sorry

end smallest_square_cover_l3550_355065


namespace milk_carton_volume_l3550_355090

/-- The volume of a rectangular prism with given dimensions -/
def rectangular_prism_volume (width length height : ℝ) : ℝ :=
  width * length * height

/-- Theorem: The volume of a milk carton with given dimensions is 252 cubic centimeters -/
theorem milk_carton_volume :
  rectangular_prism_volume 9 4 7 = 252 := by
  sorry

end milk_carton_volume_l3550_355090


namespace points_per_treasure_l3550_355092

theorem points_per_treasure (treasures_level1 treasures_level2 total_score : ℕ) 
  (h1 : treasures_level1 = 6)
  (h2 : treasures_level2 = 2)
  (h3 : total_score = 32) :
  total_score / (treasures_level1 + treasures_level2) = 4 := by
  sorry

end points_per_treasure_l3550_355092


namespace root_equation_problem_l3550_355073

theorem root_equation_problem (c d n r s : ℝ) : 
  (c^2 - n*c + 3 = 0) → 
  (d^2 - n*d + 3 = 0) → 
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) → 
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) → 
  s = 16/3 := by
sorry

end root_equation_problem_l3550_355073


namespace honey_nights_l3550_355024

/-- Represents the number of servings of honey per cup of tea -/
def servings_per_cup : ℕ := 1

/-- Represents the number of cups of tea Tabitha drinks before bed each night -/
def cups_per_night : ℕ := 2

/-- Represents the size of the honey container in ounces -/
def container_size : ℕ := 16

/-- Represents the number of servings of honey per ounce -/
def servings_per_ounce : ℕ := 6

/-- Theorem stating how many nights Tabitha can enjoy honey in her tea -/
theorem honey_nights : 
  (container_size * servings_per_ounce) / (servings_per_cup * cups_per_night) = 48 := by
  sorry

end honey_nights_l3550_355024


namespace candy_problem_l3550_355018

theorem candy_problem :
  ∀ (N : ℕ) (S : ℕ),
    N > 0 →
    (∀ i : Fin N, ∃ (a : ℕ), a > 1 ∧ a = S - (N - 1) * a - 7) →
    S = 21 := by
  sorry

end candy_problem_l3550_355018


namespace cone_hemisphere_intersection_volume_l3550_355079

/-- The volume of the common part of a right circular cone and an inscribed hemisphere -/
theorem cone_hemisphere_intersection_volume 
  (m r : ℝ) 
  (h : m > r) 
  (h_pos : r > 0) : 
  ∃ V : ℝ, V = (2 * Real.pi * r^3 / 3) * (1 - (2 * m * r^3) / (m^2 + r^2)^2) := by
  sorry

end cone_hemisphere_intersection_volume_l3550_355079


namespace average_height_combined_l3550_355066

theorem average_height_combined (group1_count group2_count : ℕ) 
  (group1_avg group2_avg : ℝ) (total_count : ℕ) :
  group1_count = 20 →
  group2_count = 11 →
  group1_avg = 20 →
  group2_avg = 20 →
  total_count = group1_count + group2_count →
  (group1_count * group1_avg + group2_count * group2_avg) / total_count = 20 := by
  sorry

end average_height_combined_l3550_355066


namespace probability_after_removal_l3550_355070

/-- Represents a deck of cards -/
structure Deck :=
  (cards : Finset ℕ)
  (card_counts : ℕ → ℕ)
  (total_cards : ℕ)

/-- Initial deck configuration -/
def initial_deck : Deck :=
  { cards := Finset.range 13,
    card_counts := λ _ => 4,
    total_cards := 52 }

/-- Deck after removing two pairs -/
def deck_after_removal (d : Deck) : Deck :=
  { cards := d.cards,
    card_counts := λ n => if d.card_counts n ≥ 2 then d.card_counts n - 2 else d.card_counts n,
    total_cards := d.total_cards - 4 }

/-- Number of ways to choose 2 cards from n cards -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Probability of selecting a pair from the remaining deck -/
def pair_probability (d : Deck) : ℚ :=
  let total_choices := choose_two d.total_cards
  let pair_choices := d.cards.sum (λ n => choose_two (d.card_counts n))
  pair_choices / total_choices

/-- Main theorem -/
theorem probability_after_removal :
  pair_probability (deck_after_removal initial_deck) = 17 / 282 := by
  sorry

end probability_after_removal_l3550_355070


namespace max_sum_of_squares_l3550_355023

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 91 →
  a * d + b * c = 187 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 107 :=
by sorry

end max_sum_of_squares_l3550_355023


namespace det_of_matrix_l3550_355013

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![5, -4; 2, 3]

theorem det_of_matrix : Matrix.det matrix = 23 := by sorry

end det_of_matrix_l3550_355013


namespace christinas_age_problem_l3550_355049

theorem christinas_age_problem (C : ℝ) (Y : ℝ) :
  (C + 5 = Y / 2) →
  (21 = (3 / 5) * C) →
  Y = 80 := by
sorry

end christinas_age_problem_l3550_355049


namespace mater_cost_percentage_l3550_355006

theorem mater_cost_percentage (lightning_cost sally_cost mater_cost : ℝ) :
  lightning_cost = 140000 →
  sally_cost = 3 * mater_cost →
  sally_cost = 42000 →
  mater_cost / lightning_cost = 0.1 := by
sorry

end mater_cost_percentage_l3550_355006


namespace geometric_sequence_ninth_term_l3550_355077

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n > 0 → a (n + 1) / a n = r

theorem geometric_sequence_ninth_term (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, n > 0 → a (n + 1) / a n = 2^n) →
  a 1 = 1 →
  a 9 = 2^36 := by
  sorry

end geometric_sequence_ninth_term_l3550_355077


namespace quadratic_always_positive_l3550_355005

theorem quadratic_always_positive (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 2 > 0) → a > 9/8 := by
  sorry

end quadratic_always_positive_l3550_355005


namespace students_between_50_and_90_count_l3550_355016

/-- Represents the distribution of student scores -/
structure ScoreDistribution where
  total_students : ℕ
  mean : ℝ
  std_dev : ℝ
  above_90 : ℕ

/-- Calculates the number of students scoring between 50 and 90 -/
def students_between_50_and_90 (d : ScoreDistribution) : ℕ :=
  d.total_students - 2 * d.above_90

/-- Theorem stating the number of students scoring between 50 and 90 -/
theorem students_between_50_and_90_count
  (d : ScoreDistribution)
  (h1 : d.total_students = 10000)
  (h2 : d.mean = 70)
  (h3 : d.std_dev = 10)
  (h4 : d.above_90 = 230) :
  students_between_50_and_90 d = 9540 := by
  sorry

#check students_between_50_and_90_count

end students_between_50_and_90_count_l3550_355016


namespace tan_period_l3550_355027

theorem tan_period (x : ℝ) : 
  let f : ℝ → ℝ := fun x => Real.tan (3 * x / 4)
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ p = 4 * Real.pi / 3 := by
  sorry

end tan_period_l3550_355027


namespace trigonometric_identities_l3550_355030

variable (θ : Real)
variable (α : Real)

/-- Given tan θ = 2, prove the following statements -/
theorem trigonometric_identities (h : Real.tan θ = 2) :
  ((Real.sin α + Real.sqrt 2 * Real.cos α) / (Real.sin α - Real.sqrt 2 * Real.cos α) = 3 + 2 * Real.sqrt 2) ∧
  (Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5) := by
  sorry

end trigonometric_identities_l3550_355030


namespace income_problem_l3550_355057

theorem income_problem (m n o : ℕ) : 
  (m + n) / 2 = 5050 →
  (n + o) / 2 = 6250 →
  (m + o) / 2 = 5200 →
  m = 4000 := by
sorry

end income_problem_l3550_355057


namespace total_candles_in_small_boxes_l3550_355052

theorem total_candles_in_small_boxes 
  (small_boxes_per_big_box : ℕ) 
  (num_big_boxes : ℕ) 
  (candles_per_small_box : ℕ) : 
  small_boxes_per_big_box = 4 → 
  num_big_boxes = 50 → 
  candles_per_small_box = 40 → 
  small_boxes_per_big_box * num_big_boxes * candles_per_small_box = 8000 :=
by sorry

end total_candles_in_small_boxes_l3550_355052


namespace multiplicative_inverse_of_PQ_l3550_355089

theorem multiplicative_inverse_of_PQ (P Q : ℕ) (M : ℕ) : 
  P = 123321 → 
  Q = 246642 → 
  M = 69788 → 
  (P * Q * M) % 1000003 = 1 := by
sorry

end multiplicative_inverse_of_PQ_l3550_355089


namespace rectangle_perimeter_l3550_355050

theorem rectangle_perimeter (square_side : ℝ) (rect_length rect_breadth : ℝ) :
  square_side = 8 →
  rect_length = 8 →
  rect_breadth = 4 →
  let new_length := square_side + rect_length
  let new_breadth := square_side
  2 * (new_length + new_breadth) = 48 := by
sorry

end rectangle_perimeter_l3550_355050


namespace greatest_integer_less_than_M_over_100_l3550_355033

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def M : ℚ := 2 * 19 * factorial 19 * (
  1 / (factorial 3 * factorial 18) +
  1 / (factorial 4 * factorial 17) +
  1 / (factorial 5 * factorial 16) +
  1 / (factorial 6 * factorial 15) +
  1 / (factorial 7 * factorial 14) +
  1 / (factorial 8 * factorial 13) +
  1 / (factorial 9 * factorial 12) +
  1 / (factorial 10 * factorial 11)
)

theorem greatest_integer_less_than_M_over_100 :
  ⌊M / 100⌋ = 499 := by sorry

end greatest_integer_less_than_M_over_100_l3550_355033


namespace no_negative_roots_l3550_355014

theorem no_negative_roots :
  ∀ x : ℝ, x < 0 → x^4 - 5*x^3 - 4*x^2 - 7*x + 4 ≠ 0 := by
sorry

end no_negative_roots_l3550_355014


namespace unique_pair_l3550_355038

-- Define the properties of N and X
def is_valid_pair (N : ℕ) (X : ℚ) : Prop :=
  -- N is a two-digit natural number
  10 ≤ N ∧ N < 100 ∧
  -- X is a two-digit decimal number
  3 ≤ X ∧ X < 10 ∧
  -- N becomes 56.7 smaller when a decimal point is inserted between its digits
  (N : ℚ) = (N / 10 : ℚ) + 56.7 ∧
  -- X becomes twice as close to N after this change
  (N : ℚ) - X = 2 * ((N : ℚ) - (N / 10 : ℚ))

-- Theorem statement
theorem unique_pair : ∃! (N : ℕ) (X : ℚ), is_valid_pair N X ∧ N = 63 ∧ X = 3.6 := by
  sorry

end unique_pair_l3550_355038


namespace apple_production_solution_l3550_355076

/-- Represents the apple production of a tree over three years -/
structure AppleProduction where
  first_year : ℕ
  second_year : ℕ := 2 * first_year + 8
  third_year : ℕ := (3 * second_year) / 4

/-- Theorem stating the solution to the apple production problem -/
theorem apple_production_solution :
  ∃ (prod : AppleProduction),
    prod.first_year + prod.second_year + prod.third_year = 194 ∧
    prod.first_year = 40 := by
  sorry

end apple_production_solution_l3550_355076


namespace books_remaining_l3550_355055

theorem books_remaining (initial_books yard_sale_books day1_books day2_books day3_books : ℕ) :
  initial_books = 75 →
  yard_sale_books = 33 →
  day1_books = 15 →
  day2_books = 8 →
  day3_books = 12 →
  initial_books - (yard_sale_books + day1_books + day2_books + day3_books) = 7 :=
by sorry

end books_remaining_l3550_355055


namespace solution_to_equation_l3550_355008

theorem solution_to_equation :
  ∃ x y : ℝ, 3 * x^2 - 12 * y^2 + 6 * x = 0 ∧ x = 0 ∧ y = 0 := by
  sorry

end solution_to_equation_l3550_355008


namespace roots_irrational_l3550_355047

theorem roots_irrational (p q : ℤ) (hp : Odd p) (hq : Odd q) 
  (h_real_roots : ∃ x y : ℝ, x ≠ y ∧ x^2 + 2*p*x + 2*q = 0 ∧ y^2 + 2*p*y + 2*q = 0) :
  ∀ z : ℝ, z^2 + 2*p*z + 2*q = 0 → Irrational z :=
sorry

end roots_irrational_l3550_355047


namespace q_components_l3550_355040

/-- The rank of a rational number -/
def rank (q : ℚ) : ℕ :=
  sorry

/-- The largest rational number less than 1/4 with rank 3 -/
def q : ℚ :=
  sorry

/-- The components of q when expressed as a sum of three unit fractions -/
def a₁ : ℕ := sorry
def a₂ : ℕ := sorry
def a₃ : ℕ := sorry

/-- q is less than 1/4 -/
axiom q_lt_quarter : q < 1/4

/-- q has rank 3 -/
axiom q_rank : rank q = 3

/-- q is the largest such number -/
axiom q_largest (r : ℚ) : r < 1/4 → rank r = 3 → r ≤ q

/-- q is expressed as the sum of three unit fractions -/
axiom q_sum : q = 1/a₁ + 1/a₂ + 1/a₃

/-- Each aᵢ is the smallest positive integer satisfying the condition -/
axiom a₁_minimal : ∀ n : ℕ, n > 0 → q ≥ 1/n → n ≥ a₁
axiom a₂_minimal : ∀ n : ℕ, n > 0 → q ≥ 1/a₁ + 1/n → n ≥ a₂
axiom a₃_minimal : ∀ n : ℕ, n > 0 → q ≥ 1/a₁ + 1/a₂ + 1/n → n ≥ a₃

theorem q_components : a₁ = 5 ∧ a₂ = 21 ∧ a₃ = 421 :=
  sorry

end q_components_l3550_355040


namespace exists_face_sum_gt_25_l3550_355094

/-- Represents a cube with labeled edges -/
structure LabeledCube where
  edges : Fin 12 → ℕ
  edge_sum : ∀ i : Fin 12, edges i ∈ Finset.range 13 \ {0}

/-- Represents a face of the cube -/
def Face := Fin 4 → Fin 12

/-- The sum of the numbers on the edges of a face -/
def face_sum (c : LabeledCube) (f : Face) : ℕ :=
  (Finset.range 4).sum (λ i => c.edges (f i))

/-- Theorem: There exists a face with sum greater than 25 -/
theorem exists_face_sum_gt_25 (c : LabeledCube) : 
  ∃ f : Face, face_sum c f > 25 := by
  sorry


end exists_face_sum_gt_25_l3550_355094


namespace r_value_when_m_is_3_l3550_355042

theorem r_value_when_m_is_3 :
  let m : ℕ := 3
  let t : ℕ := 3^m + 2
  let r : ℕ := 5^t - 2*t
  r = 5^29 - 58 := by
  sorry

end r_value_when_m_is_3_l3550_355042


namespace john_uber_profit_l3550_355011

/-- Calculates the profit from driving Uber given the income, initial car cost, and trade-in value. -/
def uberProfit (income : ℕ) (carCost : ℕ) (tradeInValue : ℕ) : ℕ :=
  income - (carCost - tradeInValue)

/-- Proves that John's profit from driving Uber is $18,000 given the specified conditions. -/
theorem john_uber_profit :
  let income : ℕ := 30000
  let carCost : ℕ := 18000
  let tradeInValue : ℕ := 6000
  uberProfit income carCost tradeInValue = 18000 := by
  sorry

#eval uberProfit 30000 18000 6000

end john_uber_profit_l3550_355011


namespace slower_train_speed_l3550_355025

/-- Calculates the speed of the slower train given the conditions of two trains moving in the same direction. -/
theorem slower_train_speed
  (faster_train_speed : ℝ)
  (faster_train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : faster_train_speed = 72)
  (h2 : faster_train_length = 70)
  (h3 : crossing_time = 7)
  : ∃ (slower_train_speed : ℝ), slower_train_speed = 36 :=
by
  sorry

#check slower_train_speed

end slower_train_speed_l3550_355025


namespace pythagorean_linear_function_l3550_355004

theorem pythagorean_linear_function (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem
  ((-a/c + b/c)^2 = 1/3) →  -- Point (-1, √3/3) lies on y = (a/c)x + (b/c)
  (a * b / 2 = 4) →  -- Area of triangle is 4
  c = 2 * Real.sqrt 6 := by
sorry

end pythagorean_linear_function_l3550_355004


namespace three_people_seven_steps_l3550_355085

def staircase_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose k 3 * (Nat.choose n 3 + Nat.choose 3 1 * Nat.choose n 2)

theorem three_people_seven_steps :
  staircase_arrangements 7 7 = 336 := by
  sorry

end three_people_seven_steps_l3550_355085


namespace gray_area_calculation_l3550_355009

theorem gray_area_calculation (black_area : ℝ) (width1 height1 width2 height2 : ℝ) :
  black_area = 37 ∧ 
  width1 = 8 ∧ height1 = 10 ∧ 
  width2 = 12 ∧ height2 = 9 →
  width2 * height2 - (width1 * height1 - black_area) = 65 :=
by
  sorry

end gray_area_calculation_l3550_355009


namespace qinJiushao_v3_value_l3550_355056

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 2*x^5 + 5*x^4 + 8*x^3 + 7*x^2 - 6*x + 11

-- Define Qin Jiushao's algorithm for this specific polynomial
def qinJiushao (x : ℝ) : ℝ × ℝ × ℝ × ℝ := 
  let v1 := 2*x + 5
  let v2 := v1*x + 8
  let v3 := v2*x + 7
  let v4 := v3*x - 6
  (v1, v2, v3, v4)

-- Theorem statement
theorem qinJiushao_v3_value : 
  (qinJiushao 3).2.2 = 130 :=
by sorry

end qinJiushao_v3_value_l3550_355056


namespace base_eight_addition_l3550_355001

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Adds two numbers in base b -/
def addInBase (n1 n2 : List Nat) (b : Nat) : List Nat :=
  sorry

theorem base_eight_addition : ∃ b : Nat, 
  b > 1 ∧ 
  addInBase [4, 5, 2] [3, 1, 6] b = [7, 7, 0] ∧
  b = 8 := by
  sorry

end base_eight_addition_l3550_355001


namespace dual_colored_cubes_count_l3550_355086

/-- Represents a cube with colored faces -/
structure ColoredCube where
  size : ℕ
  blue_faces : Fin 3
  red_faces : Fin 3

/-- Counts the number of smaller cubes with both colors when a colored cube is sliced -/
def count_dual_colored_cubes (cube : ColoredCube) : ℕ :=
  sorry

/-- The main theorem stating that a 4x4x4 cube with two opposite blue faces and four red faces
    will have exactly 24 smaller cubes with both colors when sliced into 1x1x1 cubes -/
theorem dual_colored_cubes_count :
  let cube : ColoredCube := ⟨4, 2, 4⟩
  count_dual_colored_cubes cube = 24 := by sorry

end dual_colored_cubes_count_l3550_355086


namespace vector_relationships_l3550_355098

/-- Given vector a and unit vector b, prove their parallel and perpendicular relationships -/
theorem vector_relationships (a b : ℝ × ℝ) : 
  a = (3, 4) → 
  (b.1^2 + b.2^2 = 1) → 
  ((∃ k : ℝ, b = k • a) → (b = (3/5, 4/5) ∨ b = (-3/5, -4/5))) ∧ 
  ((a.1 * b.1 + a.2 * b.2 = 0) → (b = (-4/5, 3/5) ∨ b = (4/5, -3/5))) := by
  sorry

end vector_relationships_l3550_355098


namespace marks_per_correct_answer_l3550_355078

/-- Proves that the number of marks scored for each correct answer is 4 -/
theorem marks_per_correct_answer 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (total_marks : ℤ) 
  (wrong_answer_penalty : ℤ) 
  (h1 : total_questions = 50)
  (h2 : correct_answers = 36)
  (h3 : total_marks = 130)
  (h4 : wrong_answer_penalty = -1)
  (h5 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (marks_per_correct : ℤ), 
    marks_per_correct * correct_answers + 
    wrong_answer_penalty * (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 :=
by sorry

end marks_per_correct_answer_l3550_355078


namespace geometric_sequence_product_l3550_355072

/-- A geometric sequence with first term 3 and the sum of 1st, 3rd, and 5th terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 
    a 1 = 3 ∧ 
    (∀ n : ℕ, a (n + 1) = a n * q) ∧
    a 1 + a 3 + a 5 = 21

/-- The product of the 2nd and 6th terms of the geometric sequence is 72 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 2 * a 6 = 72 := by
  sorry

end geometric_sequence_product_l3550_355072


namespace theater_seat_count_l3550_355017

/-- The number of seats in a theater -/
def theater_seats (people_watching : ℕ) (empty_seats : ℕ) : ℕ :=
  people_watching + empty_seats

/-- Theorem: The theater has 750 seats -/
theorem theater_seat_count : theater_seats 532 218 = 750 := by
  sorry

end theater_seat_count_l3550_355017


namespace angle_sum_around_point_l3550_355041

theorem angle_sum_around_point (y : ℝ) : 
  3 * y + 6 * y + 2 * y + 4 * y + y = 360 → y = 22.5 := by
  sorry

end angle_sum_around_point_l3550_355041


namespace marco_marie_age_difference_l3550_355068

theorem marco_marie_age_difference (marie_age : ℕ) (total_age : ℕ) : 
  marie_age = 12 → 
  total_age = 37 → 
  ∃ (marco_age : ℕ), marco_age + marie_age = total_age ∧ marco_age = 2 * marie_age + 1 :=
by
  sorry

end marco_marie_age_difference_l3550_355068


namespace unique_A_value_l3550_355075

/-- A function that checks if a number is a four-digit number -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- Theorem stating that 2 is the only single-digit value for A that satisfies the equation -/
theorem unique_A_value : 
  ∃! (A : ℕ), isSingleDigit A ∧ 
    (∃ (B C D : ℕ), isSingleDigit B ∧ isSingleDigit C ∧ isSingleDigit D ∧
      isFourDigit (A * 1000 + 2 * 100 + B * 10 + 2) ∧
      isFourDigit (1000 + C * 100 + 10 + D) ∧
      (A * 1000 + 2 * 100 + B * 10 + 2) + (1000 + C * 100 + 10 + D) = 3333) :=
by
  sorry

end unique_A_value_l3550_355075


namespace smallest_number_of_eggs_proof_l3550_355096

/-- The smallest number of eggs given the conditions -/
def smallest_number_of_eggs : ℕ := 137

/-- The number of containers with 9 eggs -/
def containers_with_nine : ℕ := 3

/-- The capacity of a full container -/
def container_capacity : ℕ := 10

theorem smallest_number_of_eggs_proof :
  ∀ n : ℕ,
  n > 130 ∧
  n = container_capacity * (n / container_capacity) - containers_with_nine →
  n ≥ smallest_number_of_eggs :=
by
  sorry

#check smallest_number_of_eggs_proof

end smallest_number_of_eggs_proof_l3550_355096
