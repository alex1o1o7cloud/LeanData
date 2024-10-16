import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3870_387018

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, x > 1 → x > a) ∧ (∃ x, x > a ∧ x ≤ 1) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3870_387018


namespace NUMINAMATH_CALUDE_line_slope_l3870_387028

/-- Given a line with equation y - 3 = 4(x + 1), its slope is 4 -/
theorem line_slope (x y : ℝ) : y - 3 = 4 * (x + 1) → (y - 3) / (x - (-1)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l3870_387028


namespace NUMINAMATH_CALUDE_car_trade_profit_percentage_l3870_387084

/-- Calculates the profit percentage on the original price when a trader buys a car at a discount and sells it at an increase. -/
theorem car_trade_profit_percentage 
  (original_price : ℝ) 
  (discount_rate : ℝ) 
  (increase_rate : ℝ) 
  (h1 : original_price > 0)
  (h2 : discount_rate = 0.20)
  (h3 : increase_rate = 0.50) : 
  (((1 - discount_rate) * (1 + increase_rate) - 1) * 100 = 20) := by
sorry

end NUMINAMATH_CALUDE_car_trade_profit_percentage_l3870_387084


namespace NUMINAMATH_CALUDE_shortest_path_across_river_l3870_387020

/-- Given two points A and B on opposite sides of a straight line (river),
    with A being 5 km north and 1 km west of B,
    prove that the shortest path from A to B crossing the line perpendicularly
    is 6 km long. -/
theorem shortest_path_across_river (A B : ℝ × ℝ) : 
  A.1 = B.1 - 1 →  -- A is 1 km west of B
  A.2 = B.2 + 5 →  -- A is 5 km north of B
  ∃ (C : ℝ × ℝ), 
    (C.1 - A.1) * (B.2 - A.2) = (C.2 - A.2) * (B.1 - A.1) ∧  -- C is on the line AB
    (C.2 = A.2 ∨ C.2 = B.2) ∧  -- C is on the same level as A or B (representing the river)
    Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) + 
    Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_shortest_path_across_river_l3870_387020


namespace NUMINAMATH_CALUDE_largest_integer_square_four_digits_base7_l3870_387002

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := 66

/-- Convert a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Count the number of digits in a number's base 7 representation -/
def digitCountBase7 (n : ℕ) : ℕ :=
  (toBase7 n).length

theorem largest_integer_square_four_digits_base7 :
  (M * M ≥ 7^3) ∧
  (M * M < 7^4) ∧
  (digitCountBase7 (M * M) = 4) ∧
  (∀ n : ℕ, n > M → digitCountBase7 (n * n) ≠ 4) := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_square_four_digits_base7_l3870_387002


namespace NUMINAMATH_CALUDE_smallest_k_is_three_l3870_387051

/-- A coloring of positive integers with k colors -/
def Coloring (k : ℕ) := ℕ+ → Fin k

/-- Property (i): For all positive integers m, n of the same color, f(m+n) = f(m) + f(n) -/
def PropertyOne (f : ℕ+ → ℕ+) (c : Coloring k) :=
  ∀ m n : ℕ+, c m = c n → f (m + n) = f m + f n

/-- Property (ii): There exist positive integers m, n such that f(m+n) ≠ f(m) + f(n) -/
def PropertyTwo (f : ℕ+ → ℕ+) :=
  ∃ m n : ℕ+, f (m + n) ≠ f m + f n

/-- The main theorem statement -/
theorem smallest_k_is_three :
  (∃ k : ℕ+, ∃ c : Coloring k, ∃ f : ℕ+ → ℕ+, PropertyOne f c ∧ PropertyTwo f) ∧
  (∀ k : ℕ+, k < 3 → ¬∃ c : Coloring k, ∃ f : ℕ+ → ℕ+, PropertyOne f c ∧ PropertyTwo f) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_is_three_l3870_387051


namespace NUMINAMATH_CALUDE_max_ratio_on_circle_l3870_387026

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- Definition of the circle x^2 + y^2 = 25 -/
def on_circle (p : IntPoint) : Prop :=
  p.x^2 + p.y^2 = 25

/-- Definition of irrational distance between two points -/
def irrational_distance (p q : IntPoint) : Prop :=
  ∃ (d : ℝ), d^2 = (p.x - q.x)^2 + (p.y - q.y)^2 ∧ Irrational d

/-- Theorem statement -/
theorem max_ratio_on_circle (P Q R S : IntPoint) :
  on_circle P → on_circle Q → on_circle R → on_circle S →
  irrational_distance P Q → irrational_distance R S →
  ∃ (ratio : ℝ), (∀ (d_PQ d_RS : ℝ),
    d_PQ^2 = (P.x - Q.x)^2 + (P.y - Q.y)^2 →
    d_RS^2 = (R.x - S.x)^2 + (R.y - S.y)^2 →
    d_PQ / d_RS ≤ ratio) ∧
  ratio = 5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_ratio_on_circle_l3870_387026


namespace NUMINAMATH_CALUDE_translated_parabola_vertex_l3870_387099

/-- The vertex of a translated parabola -/
theorem translated_parabola_vertex :
  let f (x : ℝ) := -(x - 3)^2 - 2
  ∃! (h k : ℝ), (∀ x, f x = -(x - h)^2 + k) ∧ h = 3 ∧ k = -2 :=
sorry

end NUMINAMATH_CALUDE_translated_parabola_vertex_l3870_387099


namespace NUMINAMATH_CALUDE_distance_after_walking_l3870_387090

/-- The distance between two people walking in opposite directions for 1.5 hours -/
theorem distance_after_walking (jay_speed : ℝ) (paul_speed : ℝ) (time : ℝ) : 
  jay_speed = 0.75 * (60 / 15) →  -- Jay's speed in miles per hour
  paul_speed = 2.5 * (60 / 30) →  -- Paul's speed in miles per hour
  time = 1.5 →                    -- Time in hours
  jay_speed * time + paul_speed * time = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_walking_l3870_387090


namespace NUMINAMATH_CALUDE_even_increasing_inequality_l3870_387016

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

theorem even_increasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) (h_incr : increasing_on_pos f) : 
  f 1 < f (-2) ∧ f (-2) < f 3 := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_inequality_l3870_387016


namespace NUMINAMATH_CALUDE_total_water_volume_l3870_387088

theorem total_water_volume (num_boxes : ℕ) (bottles_per_box : ℕ) (bottle_capacity : ℝ) (fill_ratio : ℝ) : 
  num_boxes = 10 →
  bottles_per_box = 50 →
  bottle_capacity = 12 →
  fill_ratio = 3/4 →
  (num_boxes * bottles_per_box * bottle_capacity * fill_ratio : ℝ) = 4500 := by
  sorry

end NUMINAMATH_CALUDE_total_water_volume_l3870_387088


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3870_387072

/-- The radius of the inscribed circle in a triangle with side lengths 8, 10, and 14 is √6. -/
theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 8) (h2 : DF = 10) (h3 : EF = 14) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3870_387072


namespace NUMINAMATH_CALUDE_ball_drawing_properties_l3870_387017

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : Nat
  white : Nat
  white_gt_3 : white > 3

/-- Represents the possible outcomes when drawing two balls -/
inductive DrawOutcome
  | SameColor
  | DifferentColors

/-- Calculates the probability of an outcome given the bag contents -/
def probability (b : BagContents) (o : DrawOutcome) : Rat :=
  sorry

/-- Calculates the probability of drawing at least one red ball -/
def probabilityAtLeastOneRed (b : BagContents) : Rat :=
  sorry

theorem ball_drawing_properties (n : Nat) (h : n > 3) :
  let b : BagContents := ⟨3, n, h⟩
  -- Events "same color" and "different colors" are mutually exclusive
  (probability b DrawOutcome.SameColor + probability b DrawOutcome.DifferentColors = 1) ∧
  -- When P(SameColor) = P(DifferentColors), P(AtLeastOneRed) = 7/12
  (probability b DrawOutcome.SameColor = probability b DrawOutcome.DifferentColors →
   probabilityAtLeastOneRed b = 7/12) :=
by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_properties_l3870_387017


namespace NUMINAMATH_CALUDE_pet_store_cages_l3870_387023

theorem pet_store_cages (total_puppies : ℕ) (puppies_per_cage : ℕ) (last_cage_puppies : ℕ) :
  total_puppies = 38 →
  puppies_per_cage = 6 →
  last_cage_puppies = 4 →
  (total_puppies / puppies_per_cage + 1 : ℕ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3870_387023


namespace NUMINAMATH_CALUDE_employed_females_percentage_l3870_387082

theorem employed_females_percentage (total_population : ℝ) 
  (h1 : total_population > 0) 
  (employed_percentage : ℝ) 
  (h2 : employed_percentage = 60) 
  (employed_males_percentage : ℝ) 
  (h3 : employed_males_percentage = 45) : 
  (employed_percentage - employed_males_percentage) / employed_percentage * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l3870_387082


namespace NUMINAMATH_CALUDE_right_triangle_sin_c_l3870_387071

theorem right_triangle_sin_c (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : B = Real.pi / 2) (h3 : Real.tan A = 3 / 4) : Real.sin C = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_c_l3870_387071


namespace NUMINAMATH_CALUDE_rotate_right_triangle_surface_area_l3870_387096

/-- The surface area of a solid formed by rotating a right triangle with sides 3, 4, and 5 around its shortest side -/
theorem rotate_right_triangle_surface_area :
  let triangle : Fin 3 → ℝ := ![3, 4, 5]
  let shortest_side := triangle 0
  let hypotenuse := triangle 2
  let height := triangle 1
  let base_area := π * height ^ 2
  let lateral_area := π * height * hypotenuse
  base_area + lateral_area = 36 * π :=
by sorry

end NUMINAMATH_CALUDE_rotate_right_triangle_surface_area_l3870_387096


namespace NUMINAMATH_CALUDE_zachary_crunches_count_l3870_387097

/-- The number of push-ups and crunches done by David and Zachary -/
def gym_class (david_pushups david_crunches zachary_pushups zachary_crunches : ℕ) : Prop :=
  (david_pushups = zachary_pushups + 40) ∧ 
  (zachary_crunches = david_crunches + 17) ∧
  (david_crunches = 45) ∧
  (zachary_pushups = 34)

theorem zachary_crunches_count :
  ∀ (david_pushups david_crunches zachary_pushups zachary_crunches : ℕ),
  gym_class david_pushups david_crunches zachary_pushups zachary_crunches →
  zachary_crunches = 62 :=
by
  sorry

end NUMINAMATH_CALUDE_zachary_crunches_count_l3870_387097


namespace NUMINAMATH_CALUDE_real_complex_condition_l3870_387055

theorem real_complex_condition (a : ℝ) : (∃ (r : ℝ), a - 1 + (a - 2) * Complex.I = r) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_complex_condition_l3870_387055


namespace NUMINAMATH_CALUDE_sum_equals_difference_l3870_387070

theorem sum_equals_difference (N : ℤ) : 
  995 + 997 + 999 + 1001 + 1003 = 5100 - N → N = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_difference_l3870_387070


namespace NUMINAMATH_CALUDE_power_function_through_point_is_sqrt_l3870_387012

/-- A power function that passes through the point (4, 2) is equal to the square root function. -/
theorem power_function_through_point_is_sqrt (f : ℝ → ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, f x = x ^ a) →  -- f is a power function
  f 4 = 2 →                         -- f passes through (4, 2)
  ∀ x : ℝ, f x = Real.sqrt x :=     -- f is the square root function
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_is_sqrt_l3870_387012


namespace NUMINAMATH_CALUDE_train_crossing_time_l3870_387001

/-- Proves that a train with given length and speed takes the calculated time to cross a post -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 150 →
  train_speed_kmh = 27 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3870_387001


namespace NUMINAMATH_CALUDE_deluxe_premium_time_fraction_l3870_387081

/-- Represents the production details of stereos by Company S -/
structure StereoProduction where
  basicFraction : ℚ
  deluxeFraction : ℚ
  premiumFraction : ℚ
  deluxeTimeFactor : ℚ
  premiumTimeFactor : ℚ

/-- Calculates the fraction of total production time spent on deluxe and premium stereos -/
def deluxePremiumTimeFraction (prod : StereoProduction) : ℚ :=
  let totalTime := prod.basicFraction + prod.deluxeFraction * prod.deluxeTimeFactor + 
                   prod.premiumFraction * prod.premiumTimeFactor
  let deluxePremiumTime := prod.deluxeFraction * prod.deluxeTimeFactor + 
                           prod.premiumFraction * prod.premiumTimeFactor
  deluxePremiumTime / totalTime

/-- Theorem stating that the fraction of time spent on deluxe and premium stereos is 123/163 -/
theorem deluxe_premium_time_fraction :
  let prod : StereoProduction := {
    basicFraction := 2/5,
    deluxeFraction := 3/10,
    premiumFraction := 1 - 2/5 - 3/10,
    deluxeTimeFactor := 8/5,
    premiumTimeFactor := 5/2
  }
  deluxePremiumTimeFraction prod = 123/163 := by sorry

end NUMINAMATH_CALUDE_deluxe_premium_time_fraction_l3870_387081


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_2009_l3870_387043

/-- Define the sequence a_n -/
def a_seq (a : ℕ+) : ℕ → ℕ
  | 0 => a
  | n + 1 => a_seq a n + 40^(Nat.factorial (n + 1))

/-- Theorem: The sequence a_n has infinitely many numbers divisible by 2009 -/
theorem infinitely_many_divisible_by_2009 (a : ℕ+) :
  ∀ k : ℕ, ∃ n > k, 2009 ∣ a_seq a n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_2009_l3870_387043


namespace NUMINAMATH_CALUDE_second_player_wins_l3870_387079

/-- Represents the state of the game -/
structure GameState :=
  (boxes : Fin 11 → ℕ)

/-- Represents a move in the game -/
structure Move :=
  (skipped : Fin 11)

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  { boxes := λ i => if i = move.skipped then state.boxes i else state.boxes i + 1 }

/-- Checks if the game is won -/
def is_won (state : GameState) : Prop :=
  ∃ i, state.boxes i = 21

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Represents the game play -/
def play (initial_state : GameState) (strategy1 strategy2 : Strategy) : Prop :=
  ∃ (n : ℕ) (states : ℕ → GameState),
    states 0 = initial_state ∧
    (∀ k, states (k+1) = 
      if k % 2 = 0
      then apply_move (states k) (strategy1 (states k))
      else apply_move (states k) (strategy2 (states k))) ∧
    is_won (states (2*n + 1)) ∧ ¬is_won (states (2*n))

/-- The theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy2 : Strategy), ∀ (strategy1 : Strategy),
    play { boxes := λ _ => 0 } strategy1 strategy2 :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l3870_387079


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_equilateral_l3870_387045

/-- An ellipse with a vertex and foci forming an equilateral triangle has eccentricity 1/2 -/
theorem ellipse_eccentricity_equilateral (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive semi-axes and focal distance
  a^2 = b^2 + c^2 →        -- Relationship between semi-axes and focal distance
  b = Real.sqrt 3 * c →    -- Condition for equilateral triangle
  c / a = 1 / 2 :=         -- Eccentricity definition and target value
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_equilateral_l3870_387045


namespace NUMINAMATH_CALUDE_recipe_total_cups_l3870_387061

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℚ
  flour : ℚ
  sugar : ℚ

/-- Calculates the total cups of ingredients given a recipe ratio and cups of sugar used -/
def totalCups (ratio : RecipeRatio) (sugarCups : ℚ) : ℚ :=
  let partValue := sugarCups / ratio.sugar
  ratio.butter * partValue + ratio.flour * partValue + sugarCups

/-- Theorem: Given the specified recipe ratio and sugar amount, the total cups is 27.5 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := { butter := 1, flour := 6, sugar := 4 }
  totalCups ratio 10 = 27.5 := by
  sorry

#eval totalCups { butter := 1, flour := 6, sugar := 4 } 10

end NUMINAMATH_CALUDE_recipe_total_cups_l3870_387061


namespace NUMINAMATH_CALUDE_problem_statement_l3870_387073

theorem problem_statement (m n : ℤ) : 
  (∃ k : ℤ, 56786730 * k = m * n * (m^60 - n^60)) ∧ 
  (m^5 + 3*m^4*n - 5*m^3*n^2 - 15*m^2*n^3 + 4*m*n^4 + 12*n^5 ≠ 33) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3870_387073


namespace NUMINAMATH_CALUDE_lune_area_l3870_387093

/-- The area of a lune formed by two overlapping semicircles -/
theorem lune_area (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 4) : 
  (π * r₂^2 / 2) - (π * r₁^2 / 2) = 3.5 * π := by sorry

end NUMINAMATH_CALUDE_lune_area_l3870_387093


namespace NUMINAMATH_CALUDE_min_complex_sum_value_l3870_387050

theorem min_complex_sum_value (p q r : ℕ+) (ζ : ℂ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (h_ζ_fourth : ζ^4 = 1)
  (h_ζ_neq_one : ζ ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt 7 ∧ 
    ∀ (p' q' r' : ℕ+) (h_distinct' : p' ≠ q' ∧ q' ≠ r' ∧ p' ≠ r'),
      Complex.abs (p' + q' * ζ + r' * ζ^3) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_complex_sum_value_l3870_387050


namespace NUMINAMATH_CALUDE_sequence_a_correct_l3870_387054

def sequence_a (n : ℕ) : ℚ :=
  (3 * 4^n + 2 * (-1)^n) / (4^n - (-1)^n)

theorem sequence_a_correct (n : ℕ) : 
  sequence_a 1 = 2 ∧ 
  (∀ k ≥ 1, sequence_a (k + 1) = (2 * sequence_a k + 6) / (sequence_a k + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_correct_l3870_387054


namespace NUMINAMATH_CALUDE_omega_range_l3870_387033

theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∃ a b : ℝ, π ≤ a ∧ a < b ∧ b ≤ 2*π ∧ Real.sin (ω*a) + Real.sin (ω*b) = 2) →
  (9/4 ≤ ω ∧ ω < 5/2) ∨ (13/4 ≤ ω) :=
by sorry

end NUMINAMATH_CALUDE_omega_range_l3870_387033


namespace NUMINAMATH_CALUDE_right_isosceles_triangle_area_l3870_387040

/-- The area of a right isosceles triangle with perimeter 3p -/
theorem right_isosceles_triangle_area (p : ℝ) :
  let a : ℝ := p * (3 / (2 + Real.sqrt 2))
  let b : ℝ := a
  let c : ℝ := Real.sqrt 2 * a
  let perimeter : ℝ := a + b + c
  let area : ℝ := (1 / 2) * a * b
  (perimeter = 3 * p) → (area = (9 * p^2 * (3 - 2 * Real.sqrt 2)) / 4) :=
by sorry

end NUMINAMATH_CALUDE_right_isosceles_triangle_area_l3870_387040


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l3870_387060

theorem incorrect_observation_value 
  (n : ℕ) 
  (initial_mean correct_value new_mean : ℝ) 
  (h_n : n = 50)
  (h_initial : initial_mean = 36)
  (h_correct : correct_value = 46)
  (h_new : new_mean = 36.5) :
  ∃ (incorrect_value : ℝ),
    n * new_mean = (n - 1) * initial_mean + correct_value ∧
    incorrect_value = initial_mean * n - (n - 1) * initial_mean - correct_value ∧
    incorrect_value = 21 := by sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l3870_387060


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3870_387019

theorem rectangle_area_increase : 
  let original_length : ℕ := 13
  let original_width : ℕ := 10
  let increase : ℕ := 2
  let original_area := original_length * original_width
  let new_length := original_length + increase
  let new_width := original_width + increase
  let new_area := new_length * new_width
  new_area - original_area = 50 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l3870_387019


namespace NUMINAMATH_CALUDE_inverse_function_solution_l3870_387080

/-- Given a function g(x) = 1 / (cx + d) where c and d are nonzero constants,
    prove that the solution to g^(-1)(x) = 2 is x = (1 - 2d) / (2c) -/
theorem inverse_function_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  let g : ℝ → ℝ := λ x => 1 / (c * x + d)
  ∃! x, g x = 2⁻¹ ∧ x = (1 - 2 * d) / (2 * c) :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_solution_l3870_387080


namespace NUMINAMATH_CALUDE_total_baseball_cards_l3870_387067

/-- The number of people who have baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person has -/
def cards_per_person : ℕ := 3

/-- The total number of baseball cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem total_baseball_cards : total_cards = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_baseball_cards_l3870_387067


namespace NUMINAMATH_CALUDE_number_of_ways_to_buy_three_items_l3870_387048

/-- The number of headphones available -/
def num_headphones : ℕ := 9

/-- The number of computer mice available -/
def num_mice : ℕ := 13

/-- The number of keyboards available -/
def num_keyboards : ℕ := 5

/-- The number of "keyboard and mouse" sets available -/
def num_keyboard_mouse_sets : ℕ := 4

/-- The number of "headphones and mouse" sets available -/
def num_headphones_mouse_sets : ℕ := 5

/-- The theorem stating the number of ways to buy three items -/
theorem number_of_ways_to_buy_three_items : 
  num_keyboard_mouse_sets * num_headphones + 
  num_headphones_mouse_sets * num_keyboards + 
  num_headphones * num_mice * num_keyboards = 646 := by
  sorry


end NUMINAMATH_CALUDE_number_of_ways_to_buy_three_items_l3870_387048


namespace NUMINAMATH_CALUDE_sequence_growth_l3870_387042

theorem sequence_growth (a : ℕ → ℤ) (h1 : a 1 > a 0) (h2 : a 1 > 0)
  (h3 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) :
  a 100 > 299 := by
  sorry

end NUMINAMATH_CALUDE_sequence_growth_l3870_387042


namespace NUMINAMATH_CALUDE_max_attached_squares_l3870_387068

/-- Represents a square in 2D space -/
structure Square :=
  (side_length : ℝ)
  (center : ℝ × ℝ)

/-- Checks if two squares are touching but not overlapping -/
def are_touching (s1 s2 : Square) : Prop :=
  sorry

/-- Checks if a square is touching the perimeter of another square -/
def is_touching_perimeter (s1 s2 : Square) : Prop :=
  sorry

/-- The configuration of squares attached to a given square -/
structure SquareConfiguration :=
  (given_square : Square)
  (attached_squares : List Square)

/-- Checks if a configuration is valid according to the problem conditions -/
def is_valid_configuration (config : SquareConfiguration) : Prop :=
  ∀ s ∈ config.attached_squares,
    is_touching_perimeter s config.given_square ∧
    ∀ t ∈ config.attached_squares, s ≠ t → ¬(are_touching s t)

/-- The main theorem: maximum number of attached squares is 8 -/
theorem max_attached_squares (config : SquareConfiguration) :
  is_valid_configuration config →
  config.attached_squares.length ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_attached_squares_l3870_387068


namespace NUMINAMATH_CALUDE_exists_cubic_positive_l3870_387010

theorem exists_cubic_positive : ∃ x : ℝ, x^3 - x^2 + 1 > 0 := by sorry

end NUMINAMATH_CALUDE_exists_cubic_positive_l3870_387010


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3870_387085

theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (circle_eq : ∀ x y, x^2 + y^2 = c^2)
  (asymptote_eq : ∀ x, b / a * x = x)
  (point_M : ∃ x y, x^2 + y^2 = c^2 ∧ y = b / a * x ∧ x = a ∧ y = b)
  (distance_condition : ∀ x y, x^2 + y^2 = c^2 ∧ y = b / a * x → 
    Real.sqrt ((x + c)^2 + y^2) - Real.sqrt ((x - c)^2 + y^2) = 2 * b)
  (relation_abc : b^2 = a^2 - c^2)
  (eccentricity_def : c / a = e) :
  e^2 = (Real.sqrt 5 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3870_387085


namespace NUMINAMATH_CALUDE_square_39_equals_square_40_minus_79_l3870_387035

theorem square_39_equals_square_40_minus_79 : (39 : ℤ)^2 = (40 : ℤ)^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_39_equals_square_40_minus_79_l3870_387035


namespace NUMINAMATH_CALUDE_train_speed_through_tunnel_l3870_387049

/-- Calculates the speed of a train passing through a tunnel -/
theorem train_speed_through_tunnel
  (train_length : ℝ)
  (tunnel_length : ℝ)
  (time_to_pass : ℝ)
  (h1 : train_length = 300)
  (h2 : tunnel_length = 1200)
  (h3 : time_to_pass = 100)
  : (train_length + tunnel_length) / time_to_pass * 3.6 = 54 := by
  sorry

#check train_speed_through_tunnel

end NUMINAMATH_CALUDE_train_speed_through_tunnel_l3870_387049


namespace NUMINAMATH_CALUDE_number_ordering_l3870_387075

theorem number_ordering : (2 : ℝ)^30 < 10^10 ∧ 10^10 < 5^15 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l3870_387075


namespace NUMINAMATH_CALUDE_regular_price_is_18_l3870_387074

/-- The regular price of a medium pizza at Joe's pizzeria -/
def regular_price : ℝ := 18

/-- The cost of 3 medium pizzas with the promotion -/
def promotion_cost : ℝ := 15

/-- The total savings when taking full advantage of the promotion -/
def total_savings : ℝ := 39

/-- Theorem stating that the regular price of a medium pizza is $18 -/
theorem regular_price_is_18 :
  regular_price = (promotion_cost + total_savings) / 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_price_is_18_l3870_387074


namespace NUMINAMATH_CALUDE_inequality_solution_l3870_387086

theorem inequality_solution (x : ℝ) : 
  1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) > 1 / 4 ↔ 
  x < -2 ∨ (0 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3870_387086


namespace NUMINAMATH_CALUDE_system_solution_l3870_387022

theorem system_solution :
  ∃ (x₁ x₂ x₃ : ℝ),
    (3 * x₁ - 2 * x₂ + x₃ = -10) ∧
    (2 * x₁ + 3 * x₂ - 4 * x₃ = 16) ∧
    (x₁ - 4 * x₂ + 3 * x₃ = -18) ∧
    (x₁ = -1) ∧ (x₂ = 2) ∧ (x₃ = -3) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3870_387022


namespace NUMINAMATH_CALUDE_intersection_product_sum_l3870_387064

/-- Given a line and a circle in R², prove that the sum of the products of the x-coordinate of one
    intersection point and the y-coordinate of the other equals 16. -/
theorem intersection_product_sum (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁ + y₁ = 5) →
  (x₂ + y₂ = 5) →
  (x₁^2 + y₁^2 = 16) →
  (x₂^2 + y₂^2 = 16) →
  x₁ * y₂ + x₂ * y₁ = 16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_product_sum_l3870_387064


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l3870_387052

def number_of_knights : ℕ := 30
def chosen_knights : ℕ := 4

def probability_adjacent_knights : ℚ :=
  1 - (Nat.choose (number_of_knights - chosen_knights) chosen_knights : ℚ) /
      (Nat.choose number_of_knights chosen_knights : ℚ)

theorem adjacent_knights_probability :
  probability_adjacent_knights = 250 / 549 := by sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l3870_387052


namespace NUMINAMATH_CALUDE_min_distance_to_one_l3870_387091

open Complex

/-- Given a complex number z satisfying the equation, the minimum value of |z - 1| is √2 -/
theorem min_distance_to_one (z : ℂ) 
  (h : Complex.abs ((z^2 + 1) / (z + I)) + Complex.abs ((z^2 + 4*I - 3) / (z - I + 2)) = 4) :
  ∃ (min_dist : ℝ), (∀ (w : ℂ), Complex.abs ((w^2 + 1) / (w + I)) + Complex.abs ((w^2 + 4*I - 3) / (w - I + 2)) = 4 → 
    Complex.abs (w - 1) ≥ min_dist) ∧ min_dist = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_one_l3870_387091


namespace NUMINAMATH_CALUDE_percentage_problem_l3870_387027

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.1 * 500 - 5 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3870_387027


namespace NUMINAMATH_CALUDE_nail_trimming_sounds_l3870_387041

/-- The number of nails per customer -/
def nails_per_customer : ℕ := 20

/-- The number of customers -/
def num_customers : ℕ := 6

/-- The total number of nail trimming sounds -/
def total_sounds : ℕ := nails_per_customer * num_customers

theorem nail_trimming_sounds : total_sounds = 120 := by
  sorry

end NUMINAMATH_CALUDE_nail_trimming_sounds_l3870_387041


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l3870_387024

/-- Represents the categories of teachers -/
inductive TeacherCategory
  | Senior
  | Intermediate
  | Junior

/-- Represents the school's teacher population -/
structure SchoolPopulation where
  total : Nat
  senior : Nat
  intermediate : Nat
  junior : Nat

/-- Represents the selected sample of teachers -/
structure SelectedSample where
  total : Nat
  senior : Nat
  intermediate : Nat
  junior : Nat

/-- Checks if the sample maintains the same proportion as the population -/
def isProportionalSample (pop : SchoolPopulation) (sample : SelectedSample) : Prop :=
  pop.senior * sample.total = sample.senior * pop.total ∧
  pop.intermediate * sample.total = sample.intermediate * pop.total ∧
  pop.junior * sample.total = sample.junior * pop.total

/-- The main theorem stating that the given sample is proportional -/
theorem stratified_sampling_correct 
  (pop : SchoolPopulation)
  (sample : SelectedSample)
  (h1 : pop.total = 150)
  (h2 : pop.senior = 15)
  (h3 : pop.intermediate = 45)
  (h4 : pop.junior = 90)
  (h5 : sample.total = 30)
  (h6 : sample.senior = 3)
  (h7 : sample.intermediate = 9)
  (h8 : sample.junior = 18) :
  isProportionalSample pop sample :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l3870_387024


namespace NUMINAMATH_CALUDE_red_marbles_taken_away_l3870_387095

/-- Proves that the number of red marbles taken away is 3 --/
theorem red_marbles_taken_away :
  let initial_red : ℕ := 20
  let initial_blue : ℕ := 30
  let total_left : ℕ := 35
  ∃ (red_taken : ℕ),
    (initial_red - red_taken) + (initial_blue - 4 * red_taken) = total_left ∧
    red_taken = 3 :=
by sorry

end NUMINAMATH_CALUDE_red_marbles_taken_away_l3870_387095


namespace NUMINAMATH_CALUDE_complex_quotient_pure_imaginary_l3870_387039

theorem complex_quotient_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*Complex.I
  let z₂ : ℂ := 3 - 4*Complex.I
  (∃ b : ℝ, z₁ / z₂ = b*Complex.I ∧ b ≠ 0) → a = 8/3 := by
sorry

end NUMINAMATH_CALUDE_complex_quotient_pure_imaginary_l3870_387039


namespace NUMINAMATH_CALUDE_corn_profit_problem_l3870_387065

theorem corn_profit_problem (seeds_per_ear : ℕ) (ear_price : ℚ) (bag_price : ℚ) (seeds_per_bag : ℕ) (total_profit : ℚ) :
  seeds_per_ear = 4 →
  ear_price = 1/10 →
  bag_price = 1/2 →
  seeds_per_bag = 100 →
  total_profit = 40 →
  (total_profit / (ear_price - (bag_price / seeds_per_bag) * seeds_per_ear) : ℚ) = 500 := by
  sorry

end NUMINAMATH_CALUDE_corn_profit_problem_l3870_387065


namespace NUMINAMATH_CALUDE_circles_tangent_m_value_l3870_387003

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y m : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + m = 0

-- Define the tangency condition
def are_tangent (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), C1 x y ∧ C2 x y ∧
  ∀ (x' y' : ℝ), (C1 x' y' ∧ C2 x' y') → (x' = x ∧ y' = y)

-- Theorem statement
theorem circles_tangent_m_value :
  are_tangent circle_C1 (circle_C2 · · 9) :=
sorry

end NUMINAMATH_CALUDE_circles_tangent_m_value_l3870_387003


namespace NUMINAMATH_CALUDE_select_books_count_l3870_387004

/-- The number of ways to select 5 books from 8 books, where 3 of the books form a trilogy that must be selected together. -/
def select_books : ℕ := 
  Nat.choose 5 2 + Nat.choose 5 5

/-- Theorem stating that the number of ways to select the books is 11. -/
theorem select_books_count : select_books = 11 := by
  sorry

end NUMINAMATH_CALUDE_select_books_count_l3870_387004


namespace NUMINAMATH_CALUDE_calculation_proof_system_of_equations_proof_l3870_387029

-- Problem 1
theorem calculation_proof : (-1)^2023 + Real.sqrt 9 - Real.pi^0 + Real.sqrt (1/8) * Real.sqrt 32 = 3 := by
  sorry

-- Problem 2
theorem system_of_equations_proof :
  ∃ (x y : ℝ), 2*x - y = 5 ∧ 3*x + 2*y = 11 ∧ x = 3 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_system_of_equations_proof_l3870_387029


namespace NUMINAMATH_CALUDE_sequence_existence_l3870_387037

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → f n < f m

theorem sequence_existence
  (f : ℕ → ℕ) (h_inc : StrictlyIncreasing f) :
  (∃ y : ℕ → ℝ, (∀ n, y n > 0) ∧
    (∀ n m, n < m → y m < y n) ∧
    (∀ ε > 0, ∃ N, ∀ n ≥ N, y n < ε) ∧
    (∀ n, y n ≤ 2 * y (f n))) ∧
  (∀ x : ℕ → ℝ,
    (∀ n m, n < m → x m < x n) →
    (∀ ε > 0, ∃ N, ∀ n ≥ N, x n < ε) →
    ∃ y : ℕ → ℝ,
      (∀ n m, n < m → y m < y n) ∧
      (∀ ε > 0, ∃ N, ∀ n ≥ N, y n < ε) ∧
      (∀ n, x n ≤ y n ∧ y n ≤ 2 * y (f n))) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_existence_l3870_387037


namespace NUMINAMATH_CALUDE_floor_ceiling_calculation_l3870_387094

theorem floor_ceiling_calculation : 
  ⌊(15 : ℝ) / 8 * (-34 : ℝ) / 4⌋ - ⌈(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌉ = 0 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_calculation_l3870_387094


namespace NUMINAMATH_CALUDE_nested_expression_value_l3870_387098

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 1) + 2) + 3) + 4) + 5) + 6) = 1272 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l3870_387098


namespace NUMINAMATH_CALUDE_discount_percentage_l3870_387092

theorem discount_percentage (M : ℝ) (C : ℝ) (S : ℝ) 
  (h1 : C = 0.64 * M) 
  (h2 : S = C * 1.28125) : 
  (M - S) / M * 100 = 18.08 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l3870_387092


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l3870_387005

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that it is an isosceles right triangle -/
theorem isosceles_right_triangle 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : c = a * Real.cos B) 
  (h2 : b = a * Real.sin C) : 
  A = π / 2 ∧ B = C := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_l3870_387005


namespace NUMINAMATH_CALUDE_first_digit_after_500_erasure_l3870_387009

/-- Calculates the total number of digits when writing numbers from 1 to n in sequence -/
def totalDigits (n : ℕ) : ℕ := sorry

/-- Finds the first digit after erasing a certain number of digits from the sequence -/
def firstDigitAfterErasure (totalNumbers : ℕ) (erasedDigits : ℕ) : ℕ := sorry

theorem first_digit_after_500_erasure :
  firstDigitAfterErasure 500 500 = 3 := by sorry

end NUMINAMATH_CALUDE_first_digit_after_500_erasure_l3870_387009


namespace NUMINAMATH_CALUDE_intersection_point_k_l3870_387000

-- Define the three lines
def line1 (x y : ℚ) : Prop := y = 4 * x - 1
def line2 (x y : ℚ) : Prop := y = -1/3 * x + 11
def line3 (x y k : ℚ) : Prop := y = 2 * x + k

-- Define the condition that all three lines intersect at the same point
def lines_intersect (k : ℚ) : Prop :=
  ∃ x y : ℚ, line1 x y ∧ line2 x y ∧ line3 x y k

-- Theorem statement
theorem intersection_point_k :
  ∃! k : ℚ, lines_intersect k ∧ k = 59/13 := by sorry

end NUMINAMATH_CALUDE_intersection_point_k_l3870_387000


namespace NUMINAMATH_CALUDE_first_number_value_l3870_387021

theorem first_number_value : ∃ x y : ℤ, 
  (x + 2 * y = 124) ∧ (y = 43) → x = 38 := by
  sorry

end NUMINAMATH_CALUDE_first_number_value_l3870_387021


namespace NUMINAMATH_CALUDE_intersection_slope_range_l3870_387044

/-- Given two points P and Q in the Cartesian plane, and a linear function y = kx - 1
    that intersects the extension of line segment PQ (excluding Q),
    prove that the range of k is between 1/3 and 3/2 (exclusive). -/
theorem intersection_slope_range (P Q : ℝ × ℝ) (k : ℝ) : 
  P = (-1, 1) →
  Q = (2, 2) →
  (∃ x y : ℝ, y = k * x - 1 ∧ 
              (y - 1) / (x + 1) = (2 - 1) / (2 + 1) ∧
              (x, y) ≠ Q) →
  1/3 < k ∧ k < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_slope_range_l3870_387044


namespace NUMINAMATH_CALUDE_log_inequality_l3870_387089

theorem log_inequality : 
  let x := Real.log 2 / Real.log 5
  let y := Real.log 2
  let z := Real.sqrt 2
  x < y ∧ y < z := by sorry

end NUMINAMATH_CALUDE_log_inequality_l3870_387089


namespace NUMINAMATH_CALUDE_complex_multiplication_l3870_387031

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3870_387031


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l3870_387046

theorem triangle_angle_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b + c = 180) (h5 : a = 37) (h6 : b = 53) : c = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l3870_387046


namespace NUMINAMATH_CALUDE_johns_payment_is_1500_l3870_387069

/-- Calculates the personal payment for hearing aids given insurance details --/
def calculate_personal_payment (cost_per_aid : ℕ) (num_aids : ℕ) (deductible : ℕ) 
  (coverage_percent : ℚ) (coverage_limit : ℕ) : ℕ :=
  let total_cost := cost_per_aid * num_aids
  let after_deductible := total_cost - deductible
  let insurance_payment := min (coverage_limit) (↑(Nat.floor (coverage_percent * ↑after_deductible)))
  total_cost - insurance_payment

/-- Theorem stating that John's personal payment for hearing aids is $1500 --/
theorem johns_payment_is_1500 : 
  calculate_personal_payment 2500 2 500 (4/5) 3500 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_johns_payment_is_1500_l3870_387069


namespace NUMINAMATH_CALUDE_rubber_bands_distribution_l3870_387008

/-- The number of rubber bands Aira had -/
def aira_bands : ℕ := sorry

/-- The number of rubber bands Samantha had -/
def samantha_bands : ℕ := sorry

/-- The number of rubber bands Joe had -/
def joe_bands : ℕ := sorry

/-- The total number of rubber bands -/
def total_bands : ℕ := sorry

theorem rubber_bands_distribution :
  -- Condition 1 and 2: Equal division resulting in 6 bands each
  total_bands = 3 * 6 ∧
  -- Condition 3: Samantha had 5 more bands than Aira
  samantha_bands = aira_bands + 5 ∧
  -- Condition 4: Aira had 1 fewer band than Joe
  aira_bands + 1 = joe_bands ∧
  -- Total bands is the sum of all individual bands
  total_bands = aira_bands + samantha_bands + joe_bands →
  -- Conclusion: Aira had 4 rubber bands
  aira_bands = 4 := by
sorry

end NUMINAMATH_CALUDE_rubber_bands_distribution_l3870_387008


namespace NUMINAMATH_CALUDE_minimum_transportation_cost_l3870_387056

/-- Represents the capacity of a truck -/
structure TruckCapacity where
  tents : ℕ
  food : ℕ

/-- Represents a truck arrangement -/
structure TruckArrangement where
  typeA : ℕ
  typeB : ℕ

/-- Calculate the total items an arrangement can carry -/
def totalCapacity (c : TruckCapacity × TruckCapacity) (a : TruckArrangement) : ℕ × ℕ :=
  (a.typeA * c.1.tents + a.typeB * c.2.tents, a.typeA * c.1.food + a.typeB * c.2.food)

/-- Calculate the cost of an arrangement -/
def arrangementCost (costs : ℕ × ℕ) (a : TruckArrangement) : ℕ :=
  a.typeA * costs.1 + a.typeB * costs.2

theorem minimum_transportation_cost :
  let totalItems : ℕ := 320
  let tentsDiff : ℕ := 80
  let totalTrucks : ℕ := 8
  let typeACapacity : TruckCapacity := ⟨40, 10⟩
  let typeBCapacity : TruckCapacity := ⟨20, 20⟩
  let costs : ℕ × ℕ := (4000, 3600)
  let tents : ℕ := (totalItems + tentsDiff) / 2
  let food : ℕ := (totalItems - tentsDiff) / 2
  ∃ (a : TruckArrangement),
    a.typeA + a.typeB = totalTrucks ∧
    totalCapacity (typeACapacity, typeBCapacity) a = (tents, food) ∧
    ∀ (b : TruckArrangement),
      b.typeA + b.typeB = totalTrucks →
      totalCapacity (typeACapacity, typeBCapacity) b = (tents, food) →
      arrangementCost costs a ≤ arrangementCost costs b ∧
      arrangementCost costs a = 29600 :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_transportation_cost_l3870_387056


namespace NUMINAMATH_CALUDE_preimage_of_4_3_l3870_387013

/-- The mapping f from ℝ² to ℝ² defined by f(x, y) = (x + 2y, 2x - y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, 2 * p.1 - p.2)

/-- Theorem stating that (2, 1) is the pre-image of (4, 3) under the mapping f -/
theorem preimage_of_4_3 :
  f (2, 1) = (4, 3) ∧ ∀ p : ℝ × ℝ, f p = (4, 3) → p = (2, 1) :=
by sorry

end NUMINAMATH_CALUDE_preimage_of_4_3_l3870_387013


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l3870_387014

/-- The focus of the parabola (y-1)^2 = 4(x-1) has coordinates (0, 1) -/
theorem parabola_focus_coordinates (x y : ℝ) : 
  ((y - 1)^2 = 4*(x - 1)) → (x = 0 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l3870_387014


namespace NUMINAMATH_CALUDE_profit_ratio_of_partners_l3870_387036

theorem profit_ratio_of_partners (p q : ℕ) (investment_ratio : Rat) (time_p time_q : ℕ) 
  (h1 : investment_ratio = 7 / 5)
  (h2 : time_p = 7)
  (h3 : time_q = 14) :
  (p : Rat) / q = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_profit_ratio_of_partners_l3870_387036


namespace NUMINAMATH_CALUDE_bounded_difference_exists_l3870_387078

/-- A monotonous function satisfying the given inequality condition. -/
structure MonotonousFunction (f : ℝ → ℝ) (c₁ c₂ : ℝ) : Prop :=
  (mono : Monotone f)
  (pos_const : c₁ > 0 ∧ c₂ > 0)
  (ineq : ∀ x y : ℝ, f x + f y - c₁ ≤ f (x + y) ∧ f (x + y) ≤ f x + f y + c₂)

/-- The main theorem stating the existence of k such that f(x) - kx is bounded. -/
theorem bounded_difference_exists (f : ℝ → ℝ) (c₁ c₂ : ℝ) 
  (hf : MonotonousFunction f c₁ c₂) : 
  ∃ k : ℝ, ∃ M : ℝ, ∀ x : ℝ, |f x - k * x| ≤ M :=
sorry

end NUMINAMATH_CALUDE_bounded_difference_exists_l3870_387078


namespace NUMINAMATH_CALUDE_park_paving_problem_l3870_387006

/-- Represents a worker paving paths in a park -/
structure Worker where
  speed : ℝ
  path : List Char

/-- Represents the park paving scenario -/
structure ParkPaving where
  worker1 : Worker
  worker2 : Worker
  totalTime : ℝ

/-- Calculates the time spent on a specific segment -/
def timeOnSegment (w : Worker) (segment : String) (p : ParkPaving) : ℝ :=
  sorry

theorem park_paving_problem (p : ParkPaving) :
  p.worker1.path = ['A', 'B', 'C'] ∧
  p.worker2.path = ['A', 'D', 'E', 'F', 'C'] ∧
  p.totalTime = 9 ∧
  p.worker2.speed = 1.2 * p.worker1.speed →
  timeOnSegment p.worker2 "DE" p = 45 / 60 := by
  sorry

end NUMINAMATH_CALUDE_park_paving_problem_l3870_387006


namespace NUMINAMATH_CALUDE_division_simplification_l3870_387057

theorem division_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  -6 * a^3 * b / (3 * a * b) = -2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l3870_387057


namespace NUMINAMATH_CALUDE_expression_value_l3870_387025

theorem expression_value (a b c : ℤ) (ha : a = 10) (hb : b = 15) (hc : c = 3) :
  (a - (b - c)) - ((a - b) + c) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3870_387025


namespace NUMINAMATH_CALUDE_carrots_removed_count_l3870_387058

-- Define the given constants
def total_carrots : ℕ := 30
def remaining_carrots : ℕ := 27
def total_weight : ℚ := 5.94
def avg_weight_remaining : ℚ := 0.2
def avg_weight_removed : ℚ := 0.18

-- Define the number of removed carrots
def removed_carrots : ℕ := total_carrots - remaining_carrots

-- Theorem statement
theorem carrots_removed_count :
  removed_carrots = 3 := by sorry

end NUMINAMATH_CALUDE_carrots_removed_count_l3870_387058


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3870_387063

theorem sqrt_inequality (x : ℝ) : 0 < x → (Real.sqrt (x + 1) < 3 * x - 2 ↔ x > 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3870_387063


namespace NUMINAMATH_CALUDE_sum_floor_equals_126_l3870_387066

theorem sum_floor_equals_126 
  (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_squares : a^2 + b^2 = 2008 ∧ c^2 + d^2 = 2008)
  (products : a*c = 1000 ∧ b*d = 1000) : 
  ⌊a + b + c + d⌋ = 126 := by
  sorry

end NUMINAMATH_CALUDE_sum_floor_equals_126_l3870_387066


namespace NUMINAMATH_CALUDE_complex_power_sum_l3870_387032

theorem complex_power_sum : 
  let i : ℂ := Complex.I
  ((1 + i) / 2) ^ 8 + ((1 - i) / 2) ^ 8 = (1 : ℂ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3870_387032


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3870_387015

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_property : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2

/-- Theorem: If 2S3 - 3S2 = 15 for an arithmetic sequence, then its common difference is 5 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 - 3 * seq.S 2 = 15) : 
  seq.d = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3870_387015


namespace NUMINAMATH_CALUDE_donation_ratio_l3870_387007

def charity_raffle_problem (total_prize donation hotdog_cost leftover : ℕ) : Prop :=
  total_prize = donation + hotdog_cost + leftover ∧
  total_prize = 114 ∧
  hotdog_cost = 2 ∧
  leftover = 55

theorem donation_ratio (total_prize donation hotdog_cost leftover : ℕ) :
  charity_raffle_problem total_prize donation hotdog_cost leftover →
  (donation : ℚ) / total_prize = 55 / 114 := by
sorry

end NUMINAMATH_CALUDE_donation_ratio_l3870_387007


namespace NUMINAMATH_CALUDE_pizza_recipe_water_amount_l3870_387011

theorem pizza_recipe_water_amount :
  ∀ (water flour salt : ℚ),
    flour = 16 →
    salt = (1/2) * flour →
    water + flour + salt = 34 →
    water = 10 :=
by sorry

end NUMINAMATH_CALUDE_pizza_recipe_water_amount_l3870_387011


namespace NUMINAMATH_CALUDE_time_taken_BC_l3870_387076

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 40
def work_rate_B : ℚ := 1 / 60
def work_rate_C : ℚ := 1 / 80

-- Define the work done by A and B
def work_done_A : ℚ := 10 * work_rate_A
def work_done_B : ℚ := 5 * work_rate_B

-- Define the remaining work
def remaining_work : ℚ := 1 - (work_done_A + work_done_B)

-- Define the combined work rate of B and C
def combined_rate_BC : ℚ := work_rate_B + work_rate_C

-- Theorem stating the time taken by B and C to finish the remaining work
theorem time_taken_BC : (remaining_work / combined_rate_BC) = 160 / 7 := by
  sorry

end NUMINAMATH_CALUDE_time_taken_BC_l3870_387076


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3870_387077

theorem complex_modulus_problem (z : ℂ) (h : (1 - I) * z = 1 + I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3870_387077


namespace NUMINAMATH_CALUDE_trig_inequality_l3870_387062

theorem trig_inequality (x y z : ℝ) (h1 : 0 < z) (h2 : z < y) (h3 : y < x) (h4 : x < π/2) :
  π/2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l3870_387062


namespace NUMINAMATH_CALUDE_max_airline_services_l3870_387087

theorem max_airline_services (internet_percentage : ℝ) (snack_percentage : ℝ) 
  (h1 : internet_percentage = 35) 
  (h2 : snack_percentage = 70) : 
  ∃ (max_both_percentage : ℝ), max_both_percentage ≤ 35 ∧ 
  ∀ (both_percentage : ℝ), 
    (both_percentage ≤ internet_percentage ∧ 
     both_percentage ≤ snack_percentage) → 
    both_percentage ≤ max_both_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_airline_services_l3870_387087


namespace NUMINAMATH_CALUDE_opposite_of_abs_one_over_2023_l3870_387053

theorem opposite_of_abs_one_over_2023 :
  -(|1 / 2023|) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_abs_one_over_2023_l3870_387053


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3870_387083

/-- Proves that given the conditions of simple and compound interest, the interest rate is 18.50% -/
theorem interest_rate_calculation (P : ℝ) (R : ℝ) : 
  P * R * 2 / 100 = 55 →
  P * ((1 + R / 100)^2 - 1) = 56.375 →
  R = 18.50 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3870_387083


namespace NUMINAMATH_CALUDE_expression_evaluation_l3870_387030

theorem expression_evaluation (a b : ℤ) (ha : a = -4) (hb : b = 3) :
  -2 * a - b^3 + 2 * a * b + b^2 = -34 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3870_387030


namespace NUMINAMATH_CALUDE_fourth_root_ten_million_l3870_387059

theorem fourth_root_ten_million (x : ℝ) : x = 10 * (10 ^ (1/4)) → x^4 = 10000000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_ten_million_l3870_387059


namespace NUMINAMATH_CALUDE_ellipse_parabola_equations_l3870_387038

/-- Given an ellipse and a parabola with specific properties, 
    prove their equations. -/
theorem ellipse_parabola_equations 
  (a b c p : ℝ) 
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : p > 0)
  (h4 : c / a = 1 / 2)  -- eccentricity
  (h5 : a - c = 1 / 2)  -- distance from left focus to directrix
  (h6 : a = p / 2)      -- right vertex is focus of parabola
  : (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 + 4 * y^2 / 3 = 1) ∧
    (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_parabola_equations_l3870_387038


namespace NUMINAMATH_CALUDE_expression_simplification_l3870_387047

theorem expression_simplification (x y z : ℝ) 
  (h1 : x > y) (h2 : y > 1) (h3 : z > 0) : 
  (x^y * y^(x+z)) / (y^(y+z) * x^x) = (x/y)^(y-x) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3870_387047


namespace NUMINAMATH_CALUDE_circles_in_rectangle_l3870_387034

theorem circles_in_rectangle (targetSum : ℝ) (h : targetSum = 1962) :
  ∃ (α : ℝ), 0 < α ∧ α < 1 / 3925 ∧
  ∀ (rectangle : Set (ℝ × ℝ)),
    (∃ a b, rectangle = {(x, y) | 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b} ∧ a * b = 1) →
    ∃ (n m : ℕ),
      (n : ℝ) * (m : ℝ) * (α / 2) > targetSum :=
by sorry

end NUMINAMATH_CALUDE_circles_in_rectangle_l3870_387034
