import Mathlib

namespace NUMINAMATH_CALUDE_total_chips_in_bag_l3481_348143

/-- Represents the number of chips Marnie eats on the first day -/
def first_day_chips : ℕ := 10

/-- Represents the number of chips Marnie eats per day after the first day -/
def daily_chips : ℕ := 10

/-- Represents the total number of days it takes Marnie to finish the bag -/
def total_days : ℕ := 10

/-- Theorem stating that the total number of chips in the bag is 100 -/
theorem total_chips_in_bag : 
  first_day_chips + (total_days - 1) * daily_chips = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_chips_in_bag_l3481_348143


namespace NUMINAMATH_CALUDE_league_games_l3481_348111

theorem league_games (n : ℕ) (h : n = 10) : (n.choose 2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l3481_348111


namespace NUMINAMATH_CALUDE_quiz_score_problem_l3481_348183

theorem quiz_score_problem (initial_students : ℕ) (dropped_students : ℕ) 
  (initial_average : ℚ) (new_average : ℚ) : 
  initial_students = 16 ∧ 
  dropped_students = 3 ∧ 
  initial_average = 62.5 ∧ 
  new_average = 62 →
  (initial_students * initial_average - 
   (initial_students - dropped_students) * new_average : ℚ) = 194 := by
  sorry

end NUMINAMATH_CALUDE_quiz_score_problem_l3481_348183


namespace NUMINAMATH_CALUDE_probability_specific_order_correct_l3481_348197

/-- Represents a standard deck of 52 cards -/
def standardDeck : ℕ := 52

/-- Represents the number of cards in each suit -/
def cardsPerSuit : ℕ := 13

/-- Represents the number of cards to be drawn -/
def cardsDrawn : ℕ := 4

/-- Calculates the probability of drawing one card from each suit in a specific order -/
def probabilitySpecificOrder : ℚ :=
  (cardsPerSuit : ℚ) / standardDeck *
  (cardsPerSuit : ℚ) / (standardDeck - 1) *
  (cardsPerSuit : ℚ) / (standardDeck - 2) *
  (cardsPerSuit : ℚ) / (standardDeck - 3)

/-- Theorem: The probability of drawing one card from each suit in a specific order is 2197/499800 -/
theorem probability_specific_order_correct :
  probabilitySpecificOrder = 2197 / 499800 := by
  sorry

end NUMINAMATH_CALUDE_probability_specific_order_correct_l3481_348197


namespace NUMINAMATH_CALUDE_adrianna_gum_purchase_l3481_348137

/-- Calculates the number of gum pieces bought at the store -/
def gum_bought_at_store (initial_gum : ℕ) (friends_given : ℕ) (gum_left : ℕ) : ℕ :=
  friends_given + gum_left - initial_gum

/-- Theorem: Given the initial conditions, prove that Adrianna bought 3 pieces of gum at the store -/
theorem adrianna_gum_purchase :
  gum_bought_at_store 10 11 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_adrianna_gum_purchase_l3481_348137


namespace NUMINAMATH_CALUDE_inequality_proof_l3481_348123

theorem inequality_proof (x y z : ℝ) 
  (hx : 2 < x ∧ x < 4) 
  (hy : 2 < y ∧ y < 4) 
  (hz : 2 < z ∧ z < 4) : 
  x / (y^2 - z) + y / (z^2 - x) + z / (x^2 - y) > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3481_348123


namespace NUMINAMATH_CALUDE_leadership_configurations_count_l3481_348151

-- Define the number of members in the society
def society_size : ℕ := 12

-- Define the number of positions to be filled
def chief_count : ℕ := 1
def supporting_chief_count : ℕ := 2
def inferior_officers_A_count : ℕ := 3
def inferior_officers_B_count : ℕ := 2

-- Define the function to calculate the number of ways to establish the leadership configuration
def leadership_configurations : ℕ := 
  society_size * (society_size - 1) * (society_size - 2) * 
  (Nat.choose (society_size - 3) inferior_officers_A_count) * 
  (Nat.choose (society_size - 3 - inferior_officers_A_count) inferior_officers_B_count)

-- Theorem statement
theorem leadership_configurations_count : leadership_configurations = 1663200 := by
  sorry

end NUMINAMATH_CALUDE_leadership_configurations_count_l3481_348151


namespace NUMINAMATH_CALUDE_smallest_expression_l3481_348193

theorem smallest_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 * a * b) / (a + b) ≤ min ((a + b) / 2) (min (Real.sqrt (a * b)) (Real.sqrt ((a^2 + b^2) / 2))) :=
sorry

end NUMINAMATH_CALUDE_smallest_expression_l3481_348193


namespace NUMINAMATH_CALUDE_forty_seventh_digit_of_1_17_l3481_348112

/-- The decimal representation of 1/17 -/
def decimal_rep_1_17 : ℚ := 1 / 17

/-- The function that returns the nth digit after the decimal point in a rational number's decimal representation -/
noncomputable def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem: The 47th digit after the decimal point in the decimal representation of 1/17 is 6 -/
theorem forty_seventh_digit_of_1_17 : nth_digit_after_decimal decimal_rep_1_17 47 = 6 := by sorry

end NUMINAMATH_CALUDE_forty_seventh_digit_of_1_17_l3481_348112


namespace NUMINAMATH_CALUDE_expression_equals_19_96_l3481_348177

theorem expression_equals_19_96 : 
  (7 * (19 / 2015) * (6 * (19 / 2016)) - 13 * (1996 / 2015) * (2 * (1997 / 2016)) - 9 * (19 / 2015)) = 19 / 96 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_19_96_l3481_348177


namespace NUMINAMATH_CALUDE_picture_processing_time_l3481_348146

/-- Given 960 pictures and a processing time of 2 minutes per picture, 
    the total processing time in hours is equal to 32. -/
theorem picture_processing_time : 
  let num_pictures : ℕ := 960
  let processing_time_per_picture : ℕ := 2
  let minutes_per_hour : ℕ := 60
  (num_pictures * processing_time_per_picture) / minutes_per_hour = 32 := by
sorry

end NUMINAMATH_CALUDE_picture_processing_time_l3481_348146


namespace NUMINAMATH_CALUDE_total_travel_time_is_45_hours_l3481_348113

/-- Represents a city with its time zone offset from New Orleans --/
structure City where
  name : String
  offset : Int

/-- Represents a flight segment with departure and arrival cities, and duration --/
structure FlightSegment where
  departure : City
  arrival : City
  duration : Nat

/-- Represents a layover with city and duration --/
structure Layover where
  city : City
  duration : Nat

/-- Calculates the total travel time considering time zone changes --/
def totalTravelTime (segments : List FlightSegment) (layovers : List Layover) : Nat :=
  sorry

/-- The cities involved in Sue's journey --/
def newOrleans : City := { name := "New Orleans", offset := 0 }
def atlanta : City := { name := "Atlanta", offset := 0 }
def chicago : City := { name := "Chicago", offset := -1 }
def newYork : City := { name := "New York", offset := 0 }
def denver : City := { name := "Denver", offset := -2 }
def sanFrancisco : City := { name := "San Francisco", offset := -3 }

/-- Sue's flight segments --/
def flightSegments : List FlightSegment := [
  { departure := newOrleans, arrival := atlanta, duration := 2 },
  { departure := atlanta, arrival := chicago, duration := 5 },
  { departure := chicago, arrival := newYork, duration := 3 },
  { departure := newYork, arrival := denver, duration := 6 },
  { departure := denver, arrival := sanFrancisco, duration := 4 }
]

/-- Sue's layovers --/
def layovers : List Layover := [
  { city := atlanta, duration := 4 },
  { city := chicago, duration := 3 },
  { city := newYork, duration := 16 },
  { city := denver, duration := 5 }
]

/-- Theorem: The total travel time from New Orleans to San Francisco is 45 hours --/
theorem total_travel_time_is_45_hours :
  totalTravelTime flightSegments layovers = 45 := by sorry

end NUMINAMATH_CALUDE_total_travel_time_is_45_hours_l3481_348113


namespace NUMINAMATH_CALUDE_speed_ratio_of_travelers_l3481_348107

/-- Given two travelers A and B covering the same distance, where A takes 2 hours
    to reach the destination and B takes 30 minutes less than A, prove that the
    ratio of their speeds (vA/vB) is 3/4. -/
theorem speed_ratio_of_travelers (d : ℝ) (vA vB : ℝ) : 
  d > 0 ∧ vA > 0 ∧ vB > 0 ∧ d / vA = 120 ∧ d / vB = 90 → vA / vB = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_speed_ratio_of_travelers_l3481_348107


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3481_348196

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2*x + 8| = 4 - 3*x :=
by
  -- The unique solution is x = -4/5
  use -4/5
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3481_348196


namespace NUMINAMATH_CALUDE_pyramid_levels_theorem_l3481_348159

/-- Represents a pyramid of blocks -/
structure BlockPyramid where
  firstRowBlocks : ℕ
  decreaseRate : ℕ
  totalBlocks : ℕ

/-- Calculate the number of levels in a BlockPyramid -/
def pyramidLevels (p : BlockPyramid) : ℕ :=
  sorry

/-- Theorem: A pyramid with 25 total blocks, 9 blocks in the first row,
    and decreasing by 2 blocks in each row has 5 levels -/
theorem pyramid_levels_theorem (p : BlockPyramid) 
  (h1 : p.firstRowBlocks = 9)
  (h2 : p.decreaseRate = 2)
  (h3 : p.totalBlocks = 25) :
  pyramidLevels p = 5 :=
  sorry

end NUMINAMATH_CALUDE_pyramid_levels_theorem_l3481_348159


namespace NUMINAMATH_CALUDE_f_not_monotonic_l3481_348182

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + 3

-- State that f is an even function
axiom f_even (m : ℝ) : ∀ x, f m x = f m (-x)

-- Define the derivative of f
def f_deriv (m : ℝ) (x : ℝ) : ℝ := 2 * (m - 1) * x - 2 * m

-- Theorem: f is not monotonic on (-∞, 3)
theorem f_not_monotonic (m : ℝ) : 
  ¬(∀ x y, x < y → x < 3 → y < 3 → f m x < f m y) ∧ 
  ¬(∀ x y, x < y → x < 3 → y < 3 → f m x > f m y) :=
sorry

end NUMINAMATH_CALUDE_f_not_monotonic_l3481_348182


namespace NUMINAMATH_CALUDE_square_sum_equals_sixteen_l3481_348165

theorem square_sum_equals_sixteen (a b : ℝ) (h : a + b = 4) : a^2 + 2*a*b + b^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_sixteen_l3481_348165


namespace NUMINAMATH_CALUDE_binomial_coefficient_1300_2_l3481_348140

theorem binomial_coefficient_1300_2 : 
  Nat.choose 1300 2 = 844350 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1300_2_l3481_348140


namespace NUMINAMATH_CALUDE_jamal_book_cart_l3481_348148

theorem jamal_book_cart (history_books fiction_books childrens_books wrong_place_books remaining_books : ℕ) :
  history_books = 12 →
  fiction_books = 19 →
  childrens_books = 8 →
  wrong_place_books = 4 →
  remaining_books = 16 →
  history_books + fiction_books + childrens_books + wrong_place_books + remaining_books = 59 := by
  sorry

end NUMINAMATH_CALUDE_jamal_book_cart_l3481_348148


namespace NUMINAMATH_CALUDE_forty_percent_value_l3481_348144

theorem forty_percent_value (x : ℝ) (h : 0.1 * x = 40) : 0.4 * x = 160 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_value_l3481_348144


namespace NUMINAMATH_CALUDE_cricketer_average_score_l3481_348176

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (three_match_avg : ℝ) 
  (total_avg : ℝ) 
  (h1 : total_matches = 5)
  (h2 : three_match_avg = 40)
  (h3 : total_avg = 36) :
  (5 * total_avg - 3 * three_match_avg) / 2 = 30 :=
by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l3481_348176


namespace NUMINAMATH_CALUDE_largest_811_double_l3481_348167

/-- Converts a number from base 8 to base 10 --/
def base8To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 8 --/
def base10To8 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 11 --/
def base10To11 (n : ℕ) : ℕ := sorry

/-- Checks if a number is an 8-11 double --/
def is811Double (n : ℕ) : Prop :=
  base10To11 (base8To10 (base10To8 n)) = 2 * n

/-- The largest 8-11 double is 504 --/
theorem largest_811_double :
  (∀ m : ℕ, m > 504 → ¬ is811Double m) ∧ is811Double 504 := by sorry

end NUMINAMATH_CALUDE_largest_811_double_l3481_348167


namespace NUMINAMATH_CALUDE_min_red_chips_l3481_348104

theorem min_red_chips (w b r : ℕ) : 
  b ≥ w / 3 →
  b ≤ r / 4 →
  w + b ≥ 75 →
  ∀ r' : ℕ, (∃ w' b' : ℕ, b' ≥ w' / 3 ∧ b' ≤ r' / 4 ∧ w' + b' ≥ 75) → r' ≥ 76 :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_l3481_348104


namespace NUMINAMATH_CALUDE_inequality_proof_l3481_348166

theorem inequality_proof (x y z t : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : t ≥ 0) 
  (h5 : x + y + z + t = 5) : 
  Real.sqrt (x^2 + y^2) + Real.sqrt (x^2 + 1) + Real.sqrt (z^2 + y^2) + 
  Real.sqrt (z^2 + t^2) + Real.sqrt (t^2 + 9) ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3481_348166


namespace NUMINAMATH_CALUDE_geometric_sequence_and_sum_l3481_348147

/-- Represents the sum of the first n terms in a geometric sequence -/
def S (n : ℕ) : ℚ := sorry

/-- Represents the nth term of the geometric sequence -/
def a (n : ℕ) : ℚ := sorry

/-- Represents the nth term of the sequence b_n -/
def b (n : ℕ) : ℚ := 1 / (a n) + n

/-- Represents the sum of the first n terms of the sequence b_n -/
def T (n : ℕ) : ℚ := sorry

theorem geometric_sequence_and_sum :
  (S 3 = 7/2) → (S 6 = 63/16) →
  (∀ n, a n = (1/2)^(n-2)) ∧
  (∀ n, T n = (2^n + n^2 + n - 1) / 2) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_and_sum_l3481_348147


namespace NUMINAMATH_CALUDE_min_framing_for_specific_picture_l3481_348105

/-- Calculates the minimum number of linear feet of framing required for a picture with given dimensions and border width. -/
def min_framing_feet (original_width original_height border_width : ℕ) : ℕ :=
  let enlarged_width := 2 * original_width
  let enlarged_height := 2 * original_height
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (total_width + total_height)
  (perimeter + 11) / 12

/-- The minimum number of linear feet of framing required for a 5-inch by 8-inch picture,
    doubled in size and surrounded by a 4-inch border, is 7 feet. -/
theorem min_framing_for_specific_picture :
  min_framing_feet 5 8 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_framing_for_specific_picture_l3481_348105


namespace NUMINAMATH_CALUDE_not_always_equal_l3481_348163

theorem not_always_equal (a b c : ℝ) (h1 : a = b - c) :
  (a - b/2)^2 = (c - b/2)^2 → a = c ∨ a + c = b := by sorry

end NUMINAMATH_CALUDE_not_always_equal_l3481_348163


namespace NUMINAMATH_CALUDE_quadruple_sum_product_two_l3481_348184

theorem quadruple_sum_product_two (x₁ x₂ x₃ x₄ : ℝ) : 
  (x₁ + x₂ * x₃ * x₄ = 2 ∧
   x₂ + x₁ * x₃ * x₄ = 2 ∧
   x₃ + x₁ * x₂ * x₄ = 2 ∧
   x₄ + x₁ * x₂ * x₃ = 2) →
  ((x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = 3) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = 3 ∧ x₄ = -1) ∨
   (x₁ = -1 ∧ x₂ = 3 ∧ x₃ = -1 ∧ x₄ = -1) ∨
   (x₁ = 3 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_sum_product_two_l3481_348184


namespace NUMINAMATH_CALUDE_total_sandwiches_count_l3481_348179

/-- The number of people going to the zoo -/
def people : ℝ := 219.0

/-- The number of sandwiches per person -/
def sandwiches_per_person : ℝ := 3.0

/-- The total number of sandwiches prepared -/
def total_sandwiches : ℝ := people * sandwiches_per_person

/-- Theorem stating that the total number of sandwiches is 657.0 -/
theorem total_sandwiches_count : total_sandwiches = 657.0 := by
  sorry

end NUMINAMATH_CALUDE_total_sandwiches_count_l3481_348179


namespace NUMINAMATH_CALUDE_min_value_of_function_l3481_348199

theorem min_value_of_function (θ a b : ℝ) (n : ℕ) 
  (h1 : 0 < θ) (h2 : θ < π/2) (h3 : a > 0) (h4 : b > 0) (h5 : n > 0) :
  let f := fun θ => a / (Real.sin θ)^n + b / (Real.cos θ)^n
  ∃ (θ_min : ℝ), ∀ θ', 0 < θ' ∧ θ' < π/2 → 
    f θ' ≥ f θ_min ∧ f θ_min = (a^(2/(n+2:ℝ)) + b^(2/(n+2:ℝ)))^((n+2)/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3481_348199


namespace NUMINAMATH_CALUDE_cities_under_50k_l3481_348157

/-- City population distribution -/
structure CityDistribution where
  small : ℝ  -- Percentage of cities with fewer than 5,000 residents
  medium : ℝ  -- Percentage of cities with 5,000 to 49,999 residents
  large : ℝ  -- Percentage of cities with 50,000 or more residents

/-- The given city distribution -/
def givenDistribution : CityDistribution where
  small := 20
  medium := 35
  large := 45

/-- Theorem: The percentage of cities with fewer than 50,000 residents is 55% -/
theorem cities_under_50k (d : CityDistribution) 
  (h1 : d.small = 20) 
  (h2 : d.medium = 35) 
  (h3 : d.large = 45) : 
  d.small + d.medium = 55 := by
  sorry

#check cities_under_50k

end NUMINAMATH_CALUDE_cities_under_50k_l3481_348157


namespace NUMINAMATH_CALUDE_count_valid_sequences_l3481_348139

/-- Represents a sequence of non-negative integers -/
def Sequence := ℕ → ℕ

/-- Checks if a sequence satisfies the given conditions -/
def ValidSequence (a : Sequence) : Prop :=
  a 0 = 2016 ∧
  (∀ n, a (n + 1) ≤ Real.sqrt (a n)) ∧
  (∀ m n, m ≠ n → a m ≠ a n)

/-- Counts the number of valid sequences -/
def CountValidSequences : ℕ := sorry

/-- The main theorem stating that the count of valid sequences is 948 -/
theorem count_valid_sequences :
  CountValidSequences = 948 := by sorry

end NUMINAMATH_CALUDE_count_valid_sequences_l3481_348139


namespace NUMINAMATH_CALUDE_coordinate_sum_of_h_l3481_348189

theorem coordinate_sum_of_h (g : ℝ → ℝ) (h : ℝ → ℝ) : 
  g 4 = 8 → 
  (∀ x, h x = (g x)^2) → 
  4 + h 4 = 68 := by
sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_h_l3481_348189


namespace NUMINAMATH_CALUDE_combinatorics_identities_l3481_348110

theorem combinatorics_identities :
  (∀ n k : ℕ, Nat.choose n k = Nat.choose n (n - k)) ∧
  (Nat.choose 5 3 = Nat.choose 4 2 + Nat.choose 4 3) ∧
  (5 * Nat.factorial 5 = Nat.factorial 6 - Nat.factorial 5) :=
by sorry

end NUMINAMATH_CALUDE_combinatorics_identities_l3481_348110


namespace NUMINAMATH_CALUDE_jinas_mascots_l3481_348121

/-- The number of mascots Jina has -/
def total_mascots (teddies bunnies koalas additional_teddies : ℕ) : ℕ :=
  teddies + bunnies + koalas + additional_teddies

/-- Theorem stating the total number of Jina's mascots -/
theorem jinas_mascots :
  let teddies : ℕ := 5
  let bunnies : ℕ := 3 * teddies
  let koalas : ℕ := 1
  let additional_teddies : ℕ := 2 * bunnies
  total_mascots teddies bunnies koalas additional_teddies = 51 := by
  sorry

end NUMINAMATH_CALUDE_jinas_mascots_l3481_348121


namespace NUMINAMATH_CALUDE_will_initial_money_l3481_348164

-- Define the initial amount of money Will had
def initial_money : ℕ := sorry

-- Define the cost of the game
def game_cost : ℕ := 47

-- Define the number of toys bought
def num_toys : ℕ := 9

-- Define the cost of each toy
def toy_cost : ℕ := 4

-- Theorem to prove
theorem will_initial_money :
  initial_money = game_cost + num_toys * toy_cost :=
by sorry

end NUMINAMATH_CALUDE_will_initial_money_l3481_348164


namespace NUMINAMATH_CALUDE_prob_odd_second_roll_l3481_348142

/-- A fair die with six faces -/
structure Die :=
  (faces : Finset Nat)
  (fair : faces = {1, 2, 3, 4, 5, 6})

/-- The set of odd numbers on a die -/
def oddFaces (d : Die) : Finset Nat :=
  d.faces.filter (λ n => n % 2 = 1)

/-- Probability of an event in a finite sample space -/
def probability (event : Finset α) (sampleSpace : Finset α) : ℚ :=
  (event.card : ℚ) / (sampleSpace.card : ℚ)

theorem prob_odd_second_roll (d : Die) :
  probability (oddFaces d) d.faces = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_second_roll_l3481_348142


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_l3481_348168

theorem min_sum_with_constraint (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y)
  (h : a / x + b / y = 2) :
  x + y ≥ (a + b) / 2 + Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_l3481_348168


namespace NUMINAMATH_CALUDE_smallest_prime_8_less_odd_square_l3481_348190

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if a number is an odd perfect square -/
def isOddPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 ∧ k % 2 = 1

theorem smallest_prime_8_less_odd_square : 
  (∀ n : ℕ, n < 17 → ¬(isPrime n ∧ 
    ∃ m : ℕ, isOddPerfectSquare m ∧ 
    m ≥ 16^2 ∧ 
    n = m - 8)) ∧ 
  (isPrime 17 ∧ 
    ∃ m : ℕ, isOddPerfectSquare m ∧ 
    m ≥ 16^2 ∧ 
    17 = m - 8) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_8_less_odd_square_l3481_348190


namespace NUMINAMATH_CALUDE_equipment_maintenance_cost_calculation_l3481_348145

def equipment_maintenance_cost (initial_balance cheque_payment received_payment final_balance : ℕ) : ℕ :=
  initial_balance - cheque_payment + received_payment - final_balance

theorem equipment_maintenance_cost_calculation :
  equipment_maintenance_cost 2000 600 800 1000 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_equipment_maintenance_cost_calculation_l3481_348145


namespace NUMINAMATH_CALUDE_find_x_l3481_348185

theorem find_x : ∃ x : ℝ, x = 120 ∧ 5.76 = 0.12 * (0.40 * x) := by sorry

end NUMINAMATH_CALUDE_find_x_l3481_348185


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l3481_348120

theorem no_solution_absolute_value_equation :
  ¬ ∃ x : ℝ, |(-2 * x)| + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l3481_348120


namespace NUMINAMATH_CALUDE_least_N_for_P_condition_l3481_348103

def P (N k : ℕ) : ℚ :=
  (N + 1 - 2 * ⌈(2/5 : ℚ) * N⌉) / (N + 1 : ℚ)

theorem least_N_for_P_condition :
  ∀ N : ℕ, N > 0 → N % 10 = 0 →
    (P N 2 < 8/10 ↔ N ≥ 10) :=
by sorry

end NUMINAMATH_CALUDE_least_N_for_P_condition_l3481_348103


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l3481_348114

def is_inverse_undefined (a n : ℕ) : Prop :=
  ¬ (∃ b : ℕ, a * b ≡ 1 [MOD n])

theorem smallest_undefined_inverse : 
  (∀ a : ℕ, 0 < a → a < 10 → 
    ¬(is_inverse_undefined a 55 ∧ is_inverse_undefined a 66)) ∧ 
  (is_inverse_undefined 10 55 ∧ is_inverse_undefined 10 66) := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l3481_348114


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l3481_348106

theorem stratified_sampling_problem (total_students : ℕ) 
  (group1_students : ℕ) (selected_from_group1 : ℕ) (n : ℕ) : 
  total_students = 1230 → 
  group1_students = 480 → 
  selected_from_group1 = 16 → 
  (n : ℚ) / total_students = selected_from_group1 / group1_students → 
  n = 41 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l3481_348106


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l3481_348130

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its binary representation as a list of bits -/
def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec to_bits (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_bits (m / 2)
    to_bits n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, true, false, false, true, false, true]  -- 1010011₂
  binary_to_decimal a * binary_to_decimal b = binary_to_decimal c := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l3481_348130


namespace NUMINAMATH_CALUDE_amanda_notebooks_l3481_348131

/-- Calculate the final number of notebooks Amanda has -/
def final_notebooks (initial ordered lost : ℕ) : ℕ :=
  initial + ordered - lost

/-- Theorem stating that Amanda's final number of notebooks is 74 -/
theorem amanda_notebooks : final_notebooks 65 23 14 = 74 := by
  sorry

end NUMINAMATH_CALUDE_amanda_notebooks_l3481_348131


namespace NUMINAMATH_CALUDE_min_value_fraction_l3481_348136

theorem min_value_fraction (x : ℝ) (h : x > -2) :
  (x^2 + 6*x + 9) / (x + 2) ≥ 4 ∧ ∃ y > -2, (y^2 + 6*y + 9) / (y + 2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3481_348136


namespace NUMINAMATH_CALUDE_tan_45_degrees_l3481_348134

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l3481_348134


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3481_348109

theorem nested_fraction_evaluation :
  1 + (1 / (1 + (1 / (1 + (1 / 2))))) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3481_348109


namespace NUMINAMATH_CALUDE_haley_candy_eaten_l3481_348160

/-- Given Haley's initial candy count, the amount her sister gave her, and her final candy count,
    calculate how many pieces of candy Haley ate on the first night. -/
theorem haley_candy_eaten (initial : ℕ) (sister_gave : ℕ) (final : ℕ) : 
  initial = 33 → sister_gave = 19 → final = 35 → initial - (final - sister_gave) = 17 := by
  sorry

end NUMINAMATH_CALUDE_haley_candy_eaten_l3481_348160


namespace NUMINAMATH_CALUDE_square_field_perimeter_l3481_348172

theorem square_field_perimeter (a : ℝ) :
  (∃ s : ℝ, a = s^2) →  -- area is a square number
  (∃ P : ℝ, P = 36) →  -- perimeter is 36 feet
  (6 * a = 6 * (2 * 36 + 9)) →  -- given equation
  (2 * 36 = 72) :=  -- twice the perimeter is 72 feet
by
  sorry

end NUMINAMATH_CALUDE_square_field_perimeter_l3481_348172


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l3481_348129

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 + 0.000007 = 234567 / 1000000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l3481_348129


namespace NUMINAMATH_CALUDE_function_passes_through_point_one_one_l3481_348162

/-- The function f(x) = a^(x-1) always passes through the point (1, 1) for any a > 0 and a ≠ 1 -/
theorem function_passes_through_point_one_one (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1)
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_one_one_l3481_348162


namespace NUMINAMATH_CALUDE_angelina_walking_speed_l3481_348180

/-- Angelina's walking problem -/
theorem angelina_walking_speed
  (distance_home_grocery : ℝ)
  (distance_grocery_gym : ℝ)
  (time_difference : ℝ)
  (h1 : distance_home_grocery = 960)
  (h2 : distance_grocery_gym = 480)
  (h3 : time_difference = 40)
  (h4 : distance_grocery_gym / (distance_home_grocery / speed_home_grocery) 
      = distance_grocery_gym / ((distance_home_grocery / speed_home_grocery) - time_difference))
  (h5 : speed_grocery_gym = 2 * speed_home_grocery) :
  speed_grocery_gym = 36 :=
by sorry

#check angelina_walking_speed

end NUMINAMATH_CALUDE_angelina_walking_speed_l3481_348180


namespace NUMINAMATH_CALUDE_cherries_used_for_pie_l3481_348133

theorem cherries_used_for_pie (initial_cherries remaining_cherries : ℕ) 
  (h1 : initial_cherries = 77)
  (h2 : remaining_cherries = 17) :
  initial_cherries - remaining_cherries = 60 := by
sorry

end NUMINAMATH_CALUDE_cherries_used_for_pie_l3481_348133


namespace NUMINAMATH_CALUDE_power_equation_solutions_l3481_348192

theorem power_equation_solutions : 
  {(a, b, c) : ℕ × ℕ × ℕ | 2^a * 3^b = 7^c - 1} = {(1, 1, 1), (4, 1, 2)} := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solutions_l3481_348192


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l3481_348155

theorem certain_fraction_proof : 
  ∃ (x y : ℚ), (x / y) / (6 / 7) = (7 / 15) / (2 / 3) ∧ x / y = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l3481_348155


namespace NUMINAMATH_CALUDE_mistake_correction_l3481_348171

theorem mistake_correction (x : ℝ) : x + 10 = 21 → 10 * x = 110 := by
  sorry

end NUMINAMATH_CALUDE_mistake_correction_l3481_348171


namespace NUMINAMATH_CALUDE_union_M_N_l3481_348194

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N in U
def complement_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := U \ complement_N

-- Theorem statement
theorem union_M_N : M ∪ N = {x : ℝ | -3 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_union_M_N_l3481_348194


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l3481_348127

/-- If 4 dozen apples cost $31.20, then 5 dozen apples at the same rate will cost $39.00 -/
theorem apple_cost_calculation (cost_four_dozen : ℝ) (h : cost_four_dozen = 31.20) :
  let cost_per_dozen : ℝ := cost_four_dozen / 4
  let cost_five_dozen : ℝ := 5 * cost_per_dozen
  cost_five_dozen = 39.00 := by sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l3481_348127


namespace NUMINAMATH_CALUDE_car_trading_profit_l3481_348150

theorem car_trading_profit (P : ℝ) (h : P > 0) : 
  let discount_rate := 0.20
  let increase_rate := 0.55
  let buying_price := P * (1 - discount_rate)
  let selling_price := buying_price * (1 + increase_rate)
  let profit := selling_price - P
  let profit_percentage := (profit / P) * 100
  profit_percentage = 24 := by sorry

end NUMINAMATH_CALUDE_car_trading_profit_l3481_348150


namespace NUMINAMATH_CALUDE_simplify_expression_l3481_348135

theorem simplify_expression (a b : ℝ) : 
  (1 : ℝ) * (2 * a) * (3 * b) * (4 * a^2 * b) * (5 * a^3 * b^2) = 120 * a^6 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3481_348135


namespace NUMINAMATH_CALUDE_prop_two_prop_three_prop_one_false_prop_four_false_l3481_348128

-- Proposition ②
theorem prop_two (a b : ℝ) : a > |b| → a^2 > b^2 := by sorry

-- Proposition ③
theorem prop_three (a b : ℝ) : a > b → a^3 > b^3 := by sorry

-- Proposition ① is false
theorem prop_one_false : ∃ a b c : ℝ, a > b ∧ ¬(a*c^2 > b*c^2) := by sorry

-- Proposition ④ is false
theorem prop_four_false : ∃ a b : ℝ, |a| > b ∧ ¬(a^2 > b^2) := by sorry

end NUMINAMATH_CALUDE_prop_two_prop_three_prop_one_false_prop_four_false_l3481_348128


namespace NUMINAMATH_CALUDE_sin_theta_value_l3481_348186

theorem sin_theta_value (θ : Real) (h : Real.cos (π / 4 - θ / 2) = 2 / 3) : 
  Real.sin θ = -1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l3481_348186


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3481_348118

-- Define the triangle DEF
structure Triangle (D E F : ℝ × ℝ) : Prop where
  right_angle : (E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2) = 0
  de_length : (E.1 - D.1)^2 + (E.2 - D.2)^2 = 15^2

-- Define the squares DEFG and EFHI
structure OuterSquares (D E F G H I : ℝ × ℝ) : Prop where
  square_defg : (G.1 - D.1) = (E.1 - D.1) ∧ (G.2 - D.2) = (E.2 - D.2)
  square_efhi : (I.1 - E.1) = (F.1 - E.1) ∧ (I.2 - E.2) = (F.2 - E.2)

-- Define the circle passing through G, H, I, F
structure CircleGHIF (G H I F : ℝ × ℝ) : Prop where
  on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (G.1 - center.1)^2 + (G.2 - center.2)^2 = radius^2 ∧
    (H.1 - center.1)^2 + (H.2 - center.2)^2 = radius^2 ∧
    (I.1 - center.1)^2 + (I.2 - center.2)^2 = radius^2 ∧
    (F.1 - center.1)^2 + (F.2 - center.2)^2 = radius^2

-- Define the point J on DF
def PointJ (D F J : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, J = (D.1 + t * (F.1 - D.1), D.2 + t * (F.2 - D.2)) ∧ t ≠ 1

theorem triangle_perimeter 
  (D E F G H I J : ℝ × ℝ)
  (triangle : Triangle D E F)
  (squares : OuterSquares D E F G H I)
  (circle : CircleGHIF G H I F)
  (j_on_df : PointJ D F J)
  (jf_length : (J.1 - F.1)^2 + (J.2 - F.2)^2 = 3^2) :
  let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let fd := Real.sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2)
  de + ef + fd = 15 + 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3481_348118


namespace NUMINAMATH_CALUDE_decimal_multiplication_meaning_l3481_348173

theorem decimal_multiplication_meaning (a b : ℝ) : 
  ¬ (∀ (a b : ℝ), ∃ (n : ℕ), a * b = n * (min a b)) :=
sorry

end NUMINAMATH_CALUDE_decimal_multiplication_meaning_l3481_348173


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3481_348122

theorem inequality_and_equality_condition (x : ℝ) (hx : x ≥ 0) :
  1 + x^2 + x^6 + x^8 ≥ 4 * x^4 ∧
  (1 + x^2 + x^6 + x^8 = 4 * x^4 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3481_348122


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3481_348158

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N : M ∩ N = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3481_348158


namespace NUMINAMATH_CALUDE_range_of_decreasing_function_l3481_348178

/-- A decreasing function on the real line. -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The range of a function. -/
def Range (f : ℝ → ℝ) : Set ℝ :=
  {y | ∃ x, f x = y}

/-- Theorem: For a decreasing function on the real line, 
    the range of values for a is (0,2]. -/
theorem range_of_decreasing_function (f : ℝ → ℝ) 
  (h : DecreasingFunction f) : 
  Range f = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_decreasing_function_l3481_348178


namespace NUMINAMATH_CALUDE_right_triangle_cone_properties_l3481_348117

/-- Given a right triangle with legs a and b, if rotating about leg a produces a cone
    with volume 500π cm³ and rotating about leg b produces a cone with volume 1800π cm³,
    then the hypotenuse length is √(a² + b²) and the surface area of the smaller cone
    is πb√(a² + b²). -/
theorem right_triangle_cone_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1/3 * π * b^2 * a = 500 * π) →
  (1/3 * π * a^2 * b = 1800 * π) →
  ∃ (hypotenuse surface_area : ℝ),
    hypotenuse = Real.sqrt (a^2 + b^2) ∧
    surface_area = π * min a b * Real.sqrt (a^2 + b^2) := by
  sorry

#check right_triangle_cone_properties

end NUMINAMATH_CALUDE_right_triangle_cone_properties_l3481_348117


namespace NUMINAMATH_CALUDE_function_property_l3481_348126

theorem function_property (f : ℤ → ℤ) :
  (∀ a b c : ℤ, a + b + c = 0 → f a + f b + f c = a^2 + b^2 + c^2) →
  ∃ c : ℤ, ∀ x : ℤ, f x = x^2 + c * x :=
by sorry

end NUMINAMATH_CALUDE_function_property_l3481_348126


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3481_348100

-- Define the set S with exactly two subsets
def S (a b : ℝ) := {x : ℝ | x^2 + a*x + b = 0}

-- Theorem statement
theorem quadratic_equation_properties
  (a b : ℝ)
  (h_a_pos : a > 0)
  (h_two_subsets : ∃ (x y : ℝ), x ≠ y ∧ S a b = {x, y}) :
  (a^2 - b^2 ≤ 4) ∧
  (a^2 + 1/b ≥ 4) ∧
  (∀ c x₁ x₂ : ℝ, (∀ x : ℝ, x^2 + a*x + b < c ↔ x₁ < x ∧ x < x₂) →
    |x₁ - x₂| = 4 → c = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3481_348100


namespace NUMINAMATH_CALUDE_container_volume_transformation_l3481_348187

/-- A cuboid container with volume measured in gallons -/
structure Container where
  height : ℝ
  length : ℝ
  width : ℝ
  volume : ℝ
  volume_eq : volume = height * length * width

/-- Theorem stating that if a container with 3 gallon volume has its height doubled and length tripled, its new volume will be 18 gallons -/
theorem container_volume_transformation (c : Container) 
  (h_volume : c.volume = 3) :
  let new_container := Container.mk 
    (2 * c.height) 
    (3 * c.length) 
    c.width 
    ((2 * c.height) * (3 * c.length) * c.width)
    (by simp)
  new_container.volume = 18 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_transformation_l3481_348187


namespace NUMINAMATH_CALUDE_quadratic_solutions_inequality_solution_set_l3481_348169

-- Part 1: Quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

theorem quadratic_solutions : 
  ∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = -1 ∧ 
  ∀ x : ℝ, quadratic_equation x ↔ (x = x1 ∨ x = x2) := by sorry

-- Part 2: Inequality system
def inequality_system (x : ℝ) : Prop := 3*x - 1 ≥ 5 ∧ (1 + 2*x) / 3 > x - 1

theorem inequality_solution_set :
  ∀ x : ℝ, inequality_system x ↔ 2 ≤ x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_solutions_inequality_solution_set_l3481_348169


namespace NUMINAMATH_CALUDE_min_cars_correct_l3481_348188

/-- Represents the minimum number of cars needed for a given number of adults -/
def min_cars (adults : ℕ) : ℕ :=
  if adults ≤ 5 then 6 else 10

/-- Each car must rest one day a week -/
axiom car_rest_day : ∀ (c : ℕ), c > 0 → ∃ (d : ℕ), d ≤ 7 ∧ c % 7 = d

/-- All adults wish to drive daily -/
axiom adults_drive_daily : ∀ (a : ℕ), a > 0 → ∀ (d : ℕ), d ≤ 7 → ∃ (c : ℕ), c > 0

theorem min_cars_correct (adults : ℕ) (h : adults > 0) :
  ∀ (cars : ℕ), cars < min_cars adults →
    ∃ (d : ℕ), d ≤ 7 ∧ cars - (cars / 7) < adults :=
by sorry

#check min_cars_correct

end NUMINAMATH_CALUDE_min_cars_correct_l3481_348188


namespace NUMINAMATH_CALUDE_right_triangle_legs_sum_l3481_348152

theorem right_triangle_legs_sum : ∀ a b : ℕ,
  (a + 1 = b) →                 -- legs are consecutive whole numbers
  (a ^ 2 + b ^ 2 = 41 ^ 2) →    -- Pythagorean theorem with hypotenuse 41
  (a + b = 57) :=               -- sum of legs is 57
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_sum_l3481_348152


namespace NUMINAMATH_CALUDE_perfect_cube_pair_l3481_348161

theorem perfect_cube_pair (a b : ℕ+) :
  (∃ (m n : ℕ+), a^3 + 6*a*b + 1 = m^3 ∧ b^3 + 6*a*b + 1 = n^3) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_perfect_cube_pair_l3481_348161


namespace NUMINAMATH_CALUDE_free_throw_contest_ratio_l3481_348195

theorem free_throw_contest_ratio (alex sandra hector : ℕ) : 
  alex = 8 →
  hector = 2 * sandra →
  alex + sandra + hector = 80 →
  sandra = 3 * alex :=
by
  sorry

end NUMINAMATH_CALUDE_free_throw_contest_ratio_l3481_348195


namespace NUMINAMATH_CALUDE_tadd_number_count_l3481_348124

theorem tadd_number_count : 
  let n : ℕ := 20  -- number of rounds
  let a : ℕ := 1   -- first term of the sequence
  let d : ℕ := 2   -- common difference
  let l : ℕ := a + d * (n - 1)  -- last term
  (n : ℚ) / 2 * (a + l) = 400 := by
  sorry

end NUMINAMATH_CALUDE_tadd_number_count_l3481_348124


namespace NUMINAMATH_CALUDE_milk_water_ratio_problem_l3481_348119

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- The problem statement -/
theorem milk_water_ratio_problem 
  (initial : CanContents)
  (h_capacity : initial.milk + initial.water + 20 = 60)
  (h_ratio_after : (initial.milk + 20) / initial.water = 3) :
  initial.milk / initial.water = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_ratio_problem_l3481_348119


namespace NUMINAMATH_CALUDE_incorrect_average_l3481_348102

def numbers : List ℝ := [1200, 1300, 1400, 1520, 1530, 1200]
def given_average : ℝ := 1380

theorem incorrect_average : 
  (numbers.sum / numbers.length) ≠ given_average :=
sorry

end NUMINAMATH_CALUDE_incorrect_average_l3481_348102


namespace NUMINAMATH_CALUDE_plate_acceleration_l3481_348125

noncomputable def α : Real := Real.arccos 0.82
noncomputable def g : Real := 10

theorem plate_acceleration (R r m : Real) (h_R : R = 1) (h_r : r = 0.5) (h_m : m = 75) :
  let a := g * Real.sqrt ((1 - Real.cos α) / 2)
  let direction := α / 2
  a = 3 ∧ direction = Real.arcsin 0.2 := by sorry

end NUMINAMATH_CALUDE_plate_acceleration_l3481_348125


namespace NUMINAMATH_CALUDE_august_tips_multiple_l3481_348149

theorem august_tips_multiple (total_months : Nat) (august_ratio : Real) 
  (h1 : total_months = 7)
  (h2 : august_ratio = 0.4) :
  let other_months := total_months - 1
  let august_tips := august_ratio * total_months
  august_tips / other_months = 2.8 := by
  sorry

end NUMINAMATH_CALUDE_august_tips_multiple_l3481_348149


namespace NUMINAMATH_CALUDE_train_length_l3481_348138

/-- Given a train that crosses a tree in 100 seconds and takes 150 seconds to pass a platform 700 m long, prove that the length of the train is 1400 meters. -/
theorem train_length (tree_crossing_time platform_crossing_time platform_length : ℝ) 
  (h1 : tree_crossing_time = 100)
  (h2 : platform_crossing_time = 150)
  (h3 : platform_length = 700) : 
  ∃ train_length : ℝ, train_length = 1400 ∧ 
    train_length / tree_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by
  sorry


end NUMINAMATH_CALUDE_train_length_l3481_348138


namespace NUMINAMATH_CALUDE_correct_front_view_l3481_348181

def StackColumn := List Nat

def frontView (stacks : List StackColumn) : List Nat :=
  stacks.map (List.foldl max 0)

theorem correct_front_view (stacks : List StackColumn) :
  stacks = [[3, 5], [2, 6, 4], [1, 1, 3, 8], [5, 2]] →
  frontView stacks = [5, 6, 8, 5] := by
  sorry

end NUMINAMATH_CALUDE_correct_front_view_l3481_348181


namespace NUMINAMATH_CALUDE_race_catch_up_time_l3481_348175

/-- Proves that Nicky runs for 30 seconds before Cristina catches up in a 300-meter race --/
theorem race_catch_up_time 
  (race_distance : ℝ) 
  (head_start : ℝ) 
  (cristina_speed : ℝ) 
  (nicky_speed : ℝ) 
  (h1 : race_distance = 300)
  (h2 : head_start = 12)
  (h3 : cristina_speed = 5)
  (h4 : nicky_speed = 3) : 
  ∃ (t : ℝ), t = 30 ∧ 
  cristina_speed * (t - head_start) = nicky_speed * t := by
  sorry


end NUMINAMATH_CALUDE_race_catch_up_time_l3481_348175


namespace NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l3481_348132

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_prime_divisor_factorial_sum :
  ∃ (p : ℕ), isPrime p ∧ p ∣ (factorial 13 + factorial 14) ∧
  ∀ (q : ℕ), isPrime q → q ∣ (factorial 13 + factorial 14) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l3481_348132


namespace NUMINAMATH_CALUDE_logarithm_properties_l3481_348115

theorem logarithm_properties :
  (Real.log 2 / Real.log 10 + Real.log 5 / Real.log 10 = 1) ∧
  (Real.log 2 / Real.log 4 + 2^(Real.log 3 / Real.log 2 - 1) = 2) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_properties_l3481_348115


namespace NUMINAMATH_CALUDE_parallel_line_slope_l3481_348174

/-- Given a line with equation 3x + 6y = -24, prove that the slope of any parallel line is -1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x + 6 * y = -24) → (slope_of_parallel_line : ℝ) = -1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_line_slope_l3481_348174


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_intersection_chord_length_l3481_348116

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 2

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := y = x

-- Define the intersecting line
def intersecting_line (x y : ℝ) (a : ℝ) : Prop := x - y + a = 0

-- Statement 1: Circle C is tangent to y = x
theorem circle_tangent_to_line : ∃ (x y : ℝ), circle_C x y ∧ tangent_line x y :=
sorry

-- Statement 2: Finding the value of a
theorem intersection_chord_length (a : ℝ) :
  (a ≠ 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    intersecting_line x₁ y₁ a ∧ intersecting_line x₂ y₂ a ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) →
  (a = Real.sqrt 2 - 2 ∨ a = -Real.sqrt 2 - 2) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_intersection_chord_length_l3481_348116


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l3481_348191

/-- Two lines with the same non-zero y-intercept -/
structure TwoLines where
  b : ℝ
  u : ℝ
  v : ℝ
  b_nonzero : b ≠ 0
  line1_slope : 12 * u + b = 0
  line2_slope : 8 * v + b = 0

/-- The ratio of x-intercepts is 2/3 -/
theorem x_intercept_ratio (l : TwoLines) : u / v = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l3481_348191


namespace NUMINAMATH_CALUDE_completing_square_transformation_l3481_348198

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l3481_348198


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l3481_348156

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively, 
    where a/c = b/d = 3/5, the ratio of the area of Rectangle A to the area 
    of Rectangle B is 9:25. -/
theorem rectangle_area_ratio 
  (a b c d : ℝ) 
  (h1 : a / c = 3 / 5) 
  (h2 : b / d = 3 / 5) :
  (a * b) / (c * d) = 9 / 25 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_ratio_l3481_348156


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3481_348101

theorem algebraic_expression_equality (a b : ℝ) (h : a - 3 * b = -3) : 
  5 - a + 3 * b = 8 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3481_348101


namespace NUMINAMATH_CALUDE_airplane_bag_weight_l3481_348141

/-- Represents the maximum weight allowed for each bag on an airplane. -/
def max_bag_weight (people : ℕ) (bags_per_person : ℕ) (total_weight_capacity : ℕ) 
  (additional_bags : ℕ) : ℚ :=
  let total_bags := people * bags_per_person + additional_bags
  total_weight_capacity / total_bags

/-- Theorem stating that under the given conditions, the maximum weight allowed for each bag is 50 pounds. -/
theorem airplane_bag_weight :
  max_bag_weight 6 5 6000 90 = 50 := by
  sorry

end NUMINAMATH_CALUDE_airplane_bag_weight_l3481_348141


namespace NUMINAMATH_CALUDE_passing_percentage_l3481_348154

/-- Given a total of 500 marks, a student who got 150 marks and failed by 50 marks,
    prove that the percentage needed to pass is 40%. -/
theorem passing_percentage (total_marks : ℕ) (obtained_marks : ℕ) (failing_margin : ℕ) :
  total_marks = 500 →
  obtained_marks = 150 →
  failing_margin = 50 →
  (obtained_marks + failing_margin) / total_marks * 100 = 40 := by
  sorry


end NUMINAMATH_CALUDE_passing_percentage_l3481_348154


namespace NUMINAMATH_CALUDE_positive_number_and_square_sum_l3481_348153

theorem positive_number_and_square_sum : ∃ (n : ℝ), n > 0 ∧ n^2 + n = 210 ∧ n = 14 ∧ n^3 = 2744 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_and_square_sum_l3481_348153


namespace NUMINAMATH_CALUDE_worker_travel_time_l3481_348170

/-- Proves that the usual time taken by a worker to reach her office is 24 minutes,
    given that when she walks at 3/4 of her normal speed, she arrives 8 minutes later than usual. -/
theorem worker_travel_time :
  ∀ (T : ℝ) (S : ℝ),
    S > 0 →  -- Normal speed is positive
    T > 0 →  -- Normal time is positive
    S * T = (3/4 * S) * (T + 8) →  -- Distance equation
    T = 24 := by
  sorry

end NUMINAMATH_CALUDE_worker_travel_time_l3481_348170


namespace NUMINAMATH_CALUDE_percentage_problem_l3481_348108

theorem percentage_problem (x : ℝ) : (0.3 / 100) * x = 0.15 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3481_348108
