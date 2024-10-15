import Mathlib

namespace NUMINAMATH_CALUDE_complement_intersection_eq_singleton_l4051_405103

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_eq_singleton :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_eq_singleton_l4051_405103


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l4051_405153

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- The line y = x - 1 -/
def line_equation (p : Point) : Prop :=
  p.y = p.x - 1

/-- Theorem: Given the conditions, the hyperbola has the equation x²/2 - y²/5 = 1 -/
theorem hyperbola_theorem (h : Hyperbola) (f m n : Point) :
  -- Center at origin
  hyperbola_equation h ⟨0, 0⟩ →
  -- Focus at (√7, 0)
  f = ⟨Real.sqrt 7, 0⟩ →
  -- M and N are on the hyperbola and the line
  hyperbola_equation h m ∧ line_equation m →
  hyperbola_equation h n ∧ line_equation n →
  -- Midpoint x-coordinate is -2/3
  (m.x + n.x) / 2 = -2/3 →
  -- The hyperbola equation is x²/2 - y²/5 = 1
  h.a^2 = 2 ∧ h.b^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l4051_405153


namespace NUMINAMATH_CALUDE_final_ratio_is_11_to_14_l4051_405135

/-- Represents the number of students in a school --/
structure School where
  boys : ℕ
  girls : ℕ

def initial_school : School :=
  { boys := 120,
    girls := 160 }

def students_left : School :=
  { boys := 10,
    girls := 20 }

def final_school : School :=
  { boys := initial_school.boys - students_left.boys,
    girls := initial_school.girls - students_left.girls }

theorem final_ratio_is_11_to_14 :
  ∃ (k : ℕ), k > 0 ∧ final_school.boys = 11 * k ∧ final_school.girls = 14 * k :=
sorry

end NUMINAMATH_CALUDE_final_ratio_is_11_to_14_l4051_405135


namespace NUMINAMATH_CALUDE_difference_is_ten_l4051_405159

/-- Properties of a rectangular plot -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area_eq : area = 20 * breadth
  breadth_value : breadth = 10

/-- The area of a rectangle -/
def area (plot : RectangularPlot) : ℝ := plot.length * plot.breadth

/-- The difference between length and breadth -/
def length_breadth_difference (plot : RectangularPlot) : ℝ :=
  plot.length - plot.breadth

/-- Theorem: The difference between length and breadth is 10 meters -/
theorem difference_is_ten (plot : RectangularPlot) :
  length_breadth_difference plot = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_is_ten_l4051_405159


namespace NUMINAMATH_CALUDE_ranch_cows_count_l4051_405144

/-- Represents the number of cows and horses a rancher has -/
structure RanchAnimals where
  horses : ℕ
  cows : ℕ

/-- Represents the conditions of the ranch -/
def ranchConditions (animals : RanchAnimals) : Prop :=
  animals.cows = 5 * animals.horses ∧ animals.cows + animals.horses = 168

theorem ranch_cows_count :
  ∃ (animals : RanchAnimals), ranchConditions animals ∧ animals.cows = 140 := by
  sorry

end NUMINAMATH_CALUDE_ranch_cows_count_l4051_405144


namespace NUMINAMATH_CALUDE_half_animals_are_goats_l4051_405125

/-- The number of cows the farmer has initially -/
def initial_cows : ℕ := 7

/-- The number of sheep the farmer has initially -/
def initial_sheep : ℕ := 8

/-- The number of goats the farmer has initially -/
def initial_goats : ℕ := 6

/-- The total number of animals initially -/
def initial_total : ℕ := initial_cows + initial_sheep + initial_goats

/-- The number of goats to be bought -/
def goats_to_buy : ℕ := 9

/-- Theorem stating that buying 9 goats will make half of the animals goats -/
theorem half_animals_are_goats : 
  2 * (initial_goats + goats_to_buy) = initial_total + goats_to_buy := by
  sorry

#check half_animals_are_goats

end NUMINAMATH_CALUDE_half_animals_are_goats_l4051_405125


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l4051_405109

-- Define the sets P and Q
def P : Set ℝ := {x | x ≤ 0 ∨ x > 3}
def Q : Set ℝ := {0, 1, 2, 3}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl P) ∩ Q = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l4051_405109


namespace NUMINAMATH_CALUDE_ratio_equality_l4051_405183

theorem ratio_equality (a b : ℝ) (h1 : 3 * a = 4 * b) (h2 : a * b ≠ 0) :
  a / b = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_ratio_equality_l4051_405183


namespace NUMINAMATH_CALUDE_unique_common_one_position_l4051_405122

/-- A binary sequence of length n -/
def BinarySequence (n : ℕ) := Fin n → Bool

/-- The property that for any three sequences, there exists a position where all three have a 1 -/
def ThreeSequenceProperty (n : ℕ) (sequences : Finset (BinarySequence n)) : Prop :=
  ∀ s1 s2 s3 : BinarySequence n, s1 ∈ sequences → s2 ∈ sequences → s3 ∈ sequences →
    ∃ p : Fin n, s1 p = true ∧ s2 p = true ∧ s3 p = true

/-- The main theorem to be proved -/
theorem unique_common_one_position
  (n : ℕ) (sequences : Finset (BinarySequence n))
  (h_count : sequences.card = 2^(n-1))
  (h_three : ThreeSequenceProperty n sequences) :
  ∃! p : Fin n, ∀ s ∈ sequences, s p = true :=
sorry

end NUMINAMATH_CALUDE_unique_common_one_position_l4051_405122


namespace NUMINAMATH_CALUDE_no_intersection_points_l4051_405136

theorem no_intersection_points (x y : ℝ) : 
  ¬∃ x y, (y = 3 * x^2 - 4 * x + 5) ∧ (y = -x^2 + 6 * x - 8) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_points_l4051_405136


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l4051_405124

theorem max_imaginary_part_of_roots (z : ℂ) : 
  z^6 - z^4 + z^2 - z + 1 = 0 →
  ∃ (θ : ℝ), -π/2 ≤ θ ∧ θ ≤ π/2 ∧
  (∀ (w : ℂ), w^6 - w^4 + w^2 - w + 1 = 0 → Complex.im w ≤ Real.sin θ) ∧
  θ = 900 * π / (7 * 180) :=
by sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l4051_405124


namespace NUMINAMATH_CALUDE_result_has_five_digits_l4051_405191

-- Define a nonzero digit type
def NonzeroDigit := { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

-- Define the operation
def operation (A B C : NonzeroDigit) : ℕ :=
  9876 + A.val * 100 + 54 + B.val * 10 + 2 - C.val

-- Theorem statement
theorem result_has_five_digits (A B C : NonzeroDigit) :
  10000 ≤ operation A B C ∧ operation A B C < 100000 :=
sorry

end NUMINAMATH_CALUDE_result_has_five_digits_l4051_405191


namespace NUMINAMATH_CALUDE_certain_number_problem_l4051_405158

theorem certain_number_problem : ∃! x : ℝ, ((x - 50) / 4) * 3 + 28 = 73 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l4051_405158


namespace NUMINAMATH_CALUDE_sum_of_squares_l4051_405197

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 72) : x^2 + y^2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l4051_405197


namespace NUMINAMATH_CALUDE_walk_distance_l4051_405134

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the final position after walking in four segments -/
def finalPosition (d : ℝ) : Point :=
  { x := d + d,  -- East distance: second segment + fourth segment
    y := -d + d + d }  -- South, then North, then North again

/-- Theorem stating that if the final position is 40 meters north of the start,
    then the distance walked in each segment must be 40 meters -/
theorem walk_distance (d : ℝ) :
  (finalPosition d).y = 40 → d = 40 := by sorry

end NUMINAMATH_CALUDE_walk_distance_l4051_405134


namespace NUMINAMATH_CALUDE_number_equality_l4051_405198

theorem number_equality (x : ℚ) : 
  (35 / 100) * x = (30 / 100) * 50 → x = 300 / 7 := by
sorry

end NUMINAMATH_CALUDE_number_equality_l4051_405198


namespace NUMINAMATH_CALUDE_remainder_7n_mod_5_l4051_405117

theorem remainder_7n_mod_5 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_5_l4051_405117


namespace NUMINAMATH_CALUDE_spade_then_ace_probability_l4051_405119

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Theorem: The probability of drawing a spade first and an Ace second from a standard 52-card deck is 1/52 -/
theorem spade_then_ace_probability :
  (NumSpades / StandardDeck) * (NumAces / (StandardDeck - 1)) = 1 / StandardDeck :=
sorry

end NUMINAMATH_CALUDE_spade_then_ace_probability_l4051_405119


namespace NUMINAMATH_CALUDE_quiz_show_win_probability_l4051_405137

def num_questions : ℕ := 4
def num_options : ℕ := 3
def min_correct : ℕ := 3

def probability_correct_guess : ℚ := 1 / num_options

/-- The probability of winning the quiz show by answering at least 3 out of 4 questions correctly,
    where each question has 3 options and guesses are random. -/
theorem quiz_show_win_probability :
  (Finset.sum (Finset.range (num_questions - min_correct + 1))
    (fun k => (Nat.choose num_questions (num_questions - k)) *
              (probability_correct_guess ^ (num_questions - k)) *
              ((1 - probability_correct_guess) ^ k))) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_quiz_show_win_probability_l4051_405137


namespace NUMINAMATH_CALUDE_circle_area_diameter_4_l4051_405155

/-- The area of a circle with diameter 4 meters is 4π square meters. -/
theorem circle_area_diameter_4 :
  let diameter : ℝ := 4
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_diameter_4_l4051_405155


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l4051_405156

/-- Represents the distance traveled by a boat in one hour -/
structure BoatTravel where
  speedStillWater : ℝ
  distanceAgainstStream : ℝ
  timeTravel : ℝ

/-- Calculates the distance traveled along the stream -/
def distanceAlongStream (bt : BoatTravel) : ℝ :=
  let streamSpeed := bt.speedStillWater - bt.distanceAgainstStream
  (bt.speedStillWater + streamSpeed) * bt.timeTravel

/-- Theorem: Given the conditions, the boat travels 13 km along the stream -/
theorem boat_distance_along_stream :
  ∀ (bt : BoatTravel),
    bt.speedStillWater = 11 ∧
    bt.distanceAgainstStream = 9 ∧
    bt.timeTravel = 1 →
    distanceAlongStream bt = 13 := by
  sorry


end NUMINAMATH_CALUDE_boat_distance_along_stream_l4051_405156


namespace NUMINAMATH_CALUDE_circle_center_correct_l4051_405113

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -3)

/-- Theorem: The center of the circle defined by circle_equation is circle_center -/
theorem circle_center_correct :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 13 :=
sorry

end NUMINAMATH_CALUDE_circle_center_correct_l4051_405113


namespace NUMINAMATH_CALUDE_square_roots_problem_l4051_405147

theorem square_roots_problem (m : ℝ) (h : ∃ (x : ℝ), x > 0 ∧ (m - 3)^2 = x ∧ (m - 7)^2 = x) :
  (∃ (x : ℝ), x > 0 ∧ (m - 3)^2 = x ∧ (m - 7)^2 = x) →
  (m - 3)^2 = 4 ∧ (m^2 + 2)^(1/3) = 3 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l4051_405147


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l4051_405174

theorem trig_expression_equals_one (d : ℝ) (h : d = 2 * Real.pi / 13) :
  (Real.sin (4 * d) * Real.sin (7 * d) * Real.sin (11 * d) * Real.sin (14 * d) * Real.sin (17 * d)) /
  (Real.sin d * Real.sin (2 * d) * Real.sin (4 * d) * Real.sin (5 * d) * Real.sin (6 * d)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l4051_405174


namespace NUMINAMATH_CALUDE_paul_picked_72_cans_l4051_405128

/-- The number of cans Paul picked up on Saturday and Sunday --/
def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

/-- Theorem stating that Paul picked up 72 cans in total --/
theorem paul_picked_72_cans :
  total_cans 6 3 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_paul_picked_72_cans_l4051_405128


namespace NUMINAMATH_CALUDE_rectangle_max_area_l4051_405133

/-- A rectangle with perimeter 40 meters has a maximum area of 100 square meters. -/
theorem rectangle_max_area :
  ∃ (w h : ℝ), w > 0 ∧ h > 0 ∧ 2 * (w + h) = 40 ∧
  (∀ (w' h' : ℝ), w' > 0 → h' > 0 → 2 * (w' + h') = 40 → w' * h' ≤ w * h) ∧
  w * h = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l4051_405133


namespace NUMINAMATH_CALUDE_trebled_result_of_doubled_plus_nine_l4051_405105

theorem trebled_result_of_doubled_plus_nine (x : ℕ) : x = 4 → 3 * (2 * x + 9) = 51 := by
  sorry

end NUMINAMATH_CALUDE_trebled_result_of_doubled_plus_nine_l4051_405105


namespace NUMINAMATH_CALUDE_simplify_expression_l4051_405111

theorem simplify_expression (x : ℝ) (hx : x > 0) :
  2 / (3 * x) * Real.sqrt (9 * x^3) + 6 * Real.sqrt (x / 4) - 2 * x * Real.sqrt (1 / x) = 3 * Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l4051_405111


namespace NUMINAMATH_CALUDE_balloon_height_proof_l4051_405100

/-- Calculates the maximum height a helium balloon can fly given budget and costs. -/
def balloon_max_height (total_budget : ℚ) (sheet_cost rope_cost propane_cost : ℚ) 
  (helium_cost_per_oz : ℚ) (height_per_oz : ℚ) : ℚ :=
  let remaining_budget := total_budget - (sheet_cost + rope_cost + propane_cost)
  let helium_oz := remaining_budget / helium_cost_per_oz
  helium_oz * height_per_oz

/-- The maximum height of the balloon is 9,492 feet given the specified conditions. -/
theorem balloon_height_proof : 
  balloon_max_height 200 42 18 14 (3/2) 113 = 9492 := by
  sorry

end NUMINAMATH_CALUDE_balloon_height_proof_l4051_405100


namespace NUMINAMATH_CALUDE_divisor_sum_implies_exponent_sum_l4051_405177

def sum_of_geometric_series (a r : ℕ) (n : ℕ) : ℕ :=
  (a * (r^(n+1) - 1)) / (r - 1)

def sum_of_divisors (i j : ℕ) : ℕ :=
  (sum_of_geometric_series 1 2 i) * (sum_of_geometric_series 1 5 j)

theorem divisor_sum_implies_exponent_sum (i j : ℕ) :
  sum_of_divisors i j = 930 → i + j = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_implies_exponent_sum_l4051_405177


namespace NUMINAMATH_CALUDE_expand_expression_l4051_405172

theorem expand_expression (x y z : ℝ) :
  (2 * x + 15) * (3 * y + 20 * z + 25) = 6 * x * y + 40 * x * z + 50 * x + 45 * y + 300 * z + 375 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l4051_405172


namespace NUMINAMATH_CALUDE_santa_candy_distribution_l4051_405193

theorem santa_candy_distribution (n : ℕ) (total_candies left_candies : ℕ) :
  3 < n ∧ n < 15 →
  total_candies = 195 →
  left_candies = 8 →
  ∃ k : ℕ, k * n = total_candies - left_candies ∧ k = 17 := by
  sorry

end NUMINAMATH_CALUDE_santa_candy_distribution_l4051_405193


namespace NUMINAMATH_CALUDE_pictures_hanging_l4051_405169

theorem pictures_hanging (total : ℕ) (vertical : ℕ) (horizontal : ℕ) (haphazard : ℕ) : 
  total = 30 →
  vertical = 10 →
  horizontal = total / 2 →
  haphazard = total - vertical - horizontal →
  haphazard = 5 := by
sorry

end NUMINAMATH_CALUDE_pictures_hanging_l4051_405169


namespace NUMINAMATH_CALUDE_min_value_inequality_l4051_405170

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a + 1 / b + 2 * Real.sqrt (a * b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l4051_405170


namespace NUMINAMATH_CALUDE_siblings_age_ratio_l4051_405163

theorem siblings_age_ratio : 
  ∀ (henry_age sister_age : ℕ),
  henry_age = 4 * sister_age →
  henry_age + sister_age + 15 = 240 →
  sister_age / 15 = 3 := by
sorry

end NUMINAMATH_CALUDE_siblings_age_ratio_l4051_405163


namespace NUMINAMATH_CALUDE_mixed_lubricant_price_l4051_405199

/-- Represents an oil type with its volume, price, and discount or tax -/
structure OilType where
  volume : ℝ
  price : ℝ
  discount_or_tax : ℝ
  is_discount : Bool

/-- Calculates the total cost of an oil type after applying discount or tax -/
def calculate_cost (oil : OilType) : ℝ :=
  let base_cost := oil.volume * oil.price
  if oil.is_discount then
    base_cost * (1 - oil.discount_or_tax)
  else
    base_cost * (1 + oil.discount_or_tax)

/-- Theorem stating that the final price per litre of the mixed lubricant oil is approximately 52.80 -/
theorem mixed_lubricant_price (oils : List OilType) 
  (h1 : oils.length = 6)
  (h2 : oils[0] = OilType.mk 70 43 0.15 true)
  (h3 : oils[1] = OilType.mk 50 51 0.10 false)
  (h4 : oils[2] = OilType.mk 15 60 0.08 true)
  (h5 : oils[3] = OilType.mk 25 62 0.12 false)
  (h6 : oils[4] = OilType.mk 40 67 0.05 true)
  (h7 : oils[5] = OilType.mk 10 75 0.18 true) :
  let total_cost := oils.map calculate_cost |>.sum
  let total_volume := oils.map (·.volume) |>.sum
  abs (total_cost / total_volume - 52.80) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_mixed_lubricant_price_l4051_405199


namespace NUMINAMATH_CALUDE_sum_local_values_2345_l4051_405176

/-- The local value of a digit in a number based on its position -/
def local_value (digit : ℕ) (position : ℕ) : ℕ := digit * (10 ^ position)

/-- The sum of local values of digits in a four-digit number -/
def sum_local_values (d₁ d₂ d₃ d₄ : ℕ) : ℕ :=
  local_value d₁ 3 + local_value d₂ 2 + local_value d₃ 1 + local_value d₄ 0

/-- Theorem: The sum of local values of digits in 2345 is 2345 -/
theorem sum_local_values_2345 : sum_local_values 2 3 4 5 = 2345 := by
  sorry

#eval sum_local_values 2 3 4 5

end NUMINAMATH_CALUDE_sum_local_values_2345_l4051_405176


namespace NUMINAMATH_CALUDE_candy_cookies_per_tray_l4051_405123

/-- Represents the cookie distribution problem --/
structure CookieDistribution where
  num_trays : ℕ
  num_packs : ℕ
  cookies_per_pack : ℕ
  has_equal_trays : Bool

/-- The number of cookies in each tray given the distribution --/
def cookies_per_tray (d : CookieDistribution) : ℕ :=
  (d.num_packs * d.cookies_per_pack) / d.num_trays

/-- Theorem stating the number of cookies per tray in Candy's distribution --/
theorem candy_cookies_per_tray :
  let d : CookieDistribution := {
    num_trays := 4,
    num_packs := 8,
    cookies_per_pack := 12,
    has_equal_trays := true
  }
  cookies_per_tray d = 24 := by
  sorry


end NUMINAMATH_CALUDE_candy_cookies_per_tray_l4051_405123


namespace NUMINAMATH_CALUDE_all_blue_figures_are_small_l4051_405157

-- Define the universe of shapes
inductive Shape
| Square
| Triangle

-- Define colors
inductive Color
| Blue
| Red

-- Define sizes
inductive Size
| Large
| Small

-- Define a figure as a combination of shape, color, and size
structure Figure where
  shape : Shape
  color : Color
  size : Size

-- State the conditions
axiom large_is_square : 
  ∀ (f : Figure), f.size = Size.Large → f.shape = Shape.Square

axiom blue_is_triangle : 
  ∀ (f : Figure), f.color = Color.Blue → f.shape = Shape.Triangle

-- Theorem to prove
theorem all_blue_figures_are_small : 
  ∀ (f : Figure), f.color = Color.Blue → f.size = Size.Small :=
sorry

end NUMINAMATH_CALUDE_all_blue_figures_are_small_l4051_405157


namespace NUMINAMATH_CALUDE_convention_handshakes_l4051_405140

/-- The number of handshakes in a convention --/
def number_of_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating the number of handshakes for the given convention --/
theorem convention_handshakes :
  number_of_handshakes 5 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_l4051_405140


namespace NUMINAMATH_CALUDE_quadratic_value_at_5_l4051_405104

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_value_at_5 
  (a b c : ℝ) 
  (max_at_2 : ∀ x, quadratic a b c x ≤ quadratic a b c 2)
  (max_value : quadratic a b c 2 = 6)
  (passes_origin : quadratic a b c 0 = -10) :
  quadratic a b c 5 = -30 := by
sorry

end NUMINAMATH_CALUDE_quadratic_value_at_5_l4051_405104


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4051_405180

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4051_405180


namespace NUMINAMATH_CALUDE_knights_seating_probability_formula_l4051_405145

/-- The probability of three knights being seated at a round table with n chairs
    such that there is an empty chair on either side of each knight. -/
def knights_seating_probability (n : ℕ) : ℚ :=
  if n ≥ 6 then
    (n - 4) * (n - 5) / ((n - 1) * (n - 2))
  else
    0

/-- Theorem stating that the probability of three knights being seated at a round table
    with n chairs (where n ≥ 6) such that there is an empty chair on either side of
    each knight is equal to (n-4)(n-5) / ((n-1)(n-2)). -/
theorem knights_seating_probability_formula (n : ℕ) (h : n ≥ 6) :
  knights_seating_probability n = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_knights_seating_probability_formula_l4051_405145


namespace NUMINAMATH_CALUDE_ethel_mental_math_l4051_405188

theorem ethel_mental_math (square_50 : 50^2 = 2500) :
  49^2 = 2500 - 99 := by
  sorry

end NUMINAMATH_CALUDE_ethel_mental_math_l4051_405188


namespace NUMINAMATH_CALUDE_chocolate_syrup_usage_l4051_405142

/-- The number of ounces of chocolate syrup used in each shake -/
def syrup_per_shake : ℝ := 4

/-- The number of ounces of chocolate syrup used on each cone -/
def syrup_per_cone : ℝ := 6

/-- The number of shakes sold -/
def num_shakes : ℕ := 2

/-- The number of cones sold -/
def num_cones : ℕ := 1

/-- The total number of ounces of chocolate syrup used -/
def total_syrup : ℝ := 14

theorem chocolate_syrup_usage :
  syrup_per_shake * num_shakes + syrup_per_cone * num_cones = total_syrup :=
by sorry

end NUMINAMATH_CALUDE_chocolate_syrup_usage_l4051_405142


namespace NUMINAMATH_CALUDE_boys_percentage_l4051_405120

/-- Given a class with a 2:3 ratio of boys to girls and 30 total students,
    prove that 40% of the students are boys. -/
theorem boys_percentage (total_students : ℕ) (boy_girl_ratio : ℚ) : 
  total_students = 30 →
  boy_girl_ratio = 2 / 3 →
  (boy_girl_ratio / (1 + boy_girl_ratio)) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_boys_percentage_l4051_405120


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l4051_405130

theorem polynomial_divisibility (a b c α β γ p : ℤ) (hp : Prime p)
  (h_div_α : p ∣ (a * α^2 + b * α + c))
  (h_div_β : p ∣ (a * β^2 + b * β + c))
  (h_div_γ : p ∣ (a * γ^2 + b * γ + c))
  (h_diff_αβ : ¬(p ∣ (α - β)))
  (h_diff_βγ : ¬(p ∣ (β - γ)))
  (h_diff_γα : ¬(p ∣ (γ - α))) :
  (p ∣ a) ∧ (p ∣ b) ∧ (p ∣ c) ∧ (∀ x : ℤ, p ∣ (a * x^2 + b * x + c)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l4051_405130


namespace NUMINAMATH_CALUDE_shanna_garden_harvest_l4051_405161

/-- Represents Shanna's garden --/
structure Garden where
  tomato : ℕ
  eggplant : ℕ
  pepper : ℕ

/-- Calculates the number of vegetables harvested from Shanna's garden --/
def harvest_vegetables (g : Garden) : ℕ :=
  let remaining_tomato := g.tomato / 2
  let remaining_pepper := g.pepper - 1
  let remaining_eggplant := g.eggplant
  let total_remaining := remaining_tomato + remaining_pepper + remaining_eggplant
  total_remaining * 7

/-- Theorem stating the total number of vegetables harvested from Shanna's garden --/
theorem shanna_garden_harvest :
  let initial_garden : Garden := ⟨6, 2, 4⟩
  harvest_vegetables initial_garden = 56 := by
  sorry

end NUMINAMATH_CALUDE_shanna_garden_harvest_l4051_405161


namespace NUMINAMATH_CALUDE_integer_fraction_sum_l4051_405108

theorem integer_fraction_sum (n : ℤ) (h : n ≥ 8) :
  (∃ k : ℤ, n + 1 / (n - 7) = k) ↔ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_sum_l4051_405108


namespace NUMINAMATH_CALUDE_work_completion_time_l4051_405132

theorem work_completion_time (days_B : ℝ) (combined_work : ℝ) (combined_days : ℝ) (days_A : ℝ) : 
  days_B = 45 →
  combined_work = 7 / 18 →
  combined_days = 7 →
  (1 / days_A + 1 / days_B) * combined_days = combined_work →
  days_A = 90 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l4051_405132


namespace NUMINAMATH_CALUDE_exactly_ten_naas_l4051_405187

-- Define the set S
variable (S : Type)

-- Define gib and naa as elements of S
variable (gib naa : S)

-- Define the collection relation
variable (is_collection_of : S → S → Prop)

-- Define the belonging relation
variable (belongs_to : S → S → Prop)

-- P1: Every gib is a collection of naas
axiom P1 : ∀ g : S, (g = gib) → ∃ n : S, (n = naa) ∧ is_collection_of g n

-- P2: Any two distinct gibs have two and only two naas in common
axiom P2 : ∀ g1 g2 : S, (g1 = gib) ∧ (g2 = gib) ∧ (g1 ≠ g2) →
  ∃! n1 n2 : S, (n1 = naa) ∧ (n2 = naa) ∧ (n1 ≠ n2) ∧
  is_collection_of g1 n1 ∧ is_collection_of g1 n2 ∧
  is_collection_of g2 n1 ∧ is_collection_of g2 n2

-- P3: Every naa belongs to three and only three gibs
axiom P3 : ∀ n : S, (n = naa) →
  ∃! g1 g2 g3 : S, (g1 = gib) ∧ (g2 = gib) ∧ (g3 = gib) ∧
  (g1 ≠ g2) ∧ (g2 ≠ g3) ∧ (g1 ≠ g3) ∧
  belongs_to n g1 ∧ belongs_to n g2 ∧ belongs_to n g3

-- P4: There are exactly five gibs
axiom P4 : ∃! g1 g2 g3 g4 g5 : S,
  (g1 = gib) ∧ (g2 = gib) ∧ (g3 = gib) ∧ (g4 = gib) ∧ (g5 = gib) ∧
  (g1 ≠ g2) ∧ (g1 ≠ g3) ∧ (g1 ≠ g4) ∧ (g1 ≠ g5) ∧
  (g2 ≠ g3) ∧ (g2 ≠ g4) ∧ (g2 ≠ g5) ∧
  (g3 ≠ g4) ∧ (g3 ≠ g5) ∧
  (g4 ≠ g5)

-- Theorem: There are exactly ten naas
theorem exactly_ten_naas : ∃! n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 : S,
  (n1 = naa) ∧ (n2 = naa) ∧ (n3 = naa) ∧ (n4 = naa) ∧ (n5 = naa) ∧
  (n6 = naa) ∧ (n7 = naa) ∧ (n8 = naa) ∧ (n9 = naa) ∧ (n10 = naa) ∧
  (n1 ≠ n2) ∧ (n1 ≠ n3) ∧ (n1 ≠ n4) ∧ (n1 ≠ n5) ∧ (n1 ≠ n6) ∧ (n1 ≠ n7) ∧ (n1 ≠ n8) ∧ (n1 ≠ n9) ∧ (n1 ≠ n10) ∧
  (n2 ≠ n3) ∧ (n2 ≠ n4) ∧ (n2 ≠ n5) ∧ (n2 ≠ n6) ∧ (n2 ≠ n7) ∧ (n2 ≠ n8) ∧ (n2 ≠ n9) ∧ (n2 ≠ n10) ∧
  (n3 ≠ n4) ∧ (n3 ≠ n5) ∧ (n3 ≠ n6) ∧ (n3 ≠ n7) ∧ (n3 ≠ n8) ∧ (n3 ≠ n9) ∧ (n3 ≠ n10) ∧
  (n4 ≠ n5) ∧ (n4 ≠ n6) ∧ (n4 ≠ n7) ∧ (n4 ≠ n8) ∧ (n4 ≠ n9) ∧ (n4 ≠ n10) ∧
  (n5 ≠ n6) ∧ (n5 ≠ n7) ∧ (n5 ≠ n8) ∧ (n5 ≠ n9) ∧ (n5 ≠ n10) ∧
  (n6 ≠ n7) ∧ (n6 ≠ n8) ∧ (n6 ≠ n9) ∧ (n6 ≠ n10) ∧
  (n7 ≠ n8) ∧ (n7 ≠ n9) ∧ (n7 ≠ n10) ∧
  (n8 ≠ n9) ∧ (n8 ≠ n10) ∧
  (n9 ≠ n10) :=
sorry

end NUMINAMATH_CALUDE_exactly_ten_naas_l4051_405187


namespace NUMINAMATH_CALUDE_ball_drawing_problem_l4051_405154

theorem ball_drawing_problem (n : ℕ+) : 
  (3 * n) / ((n + 3) * (n + 2) : ℝ) = 7 / 30 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_problem_l4051_405154


namespace NUMINAMATH_CALUDE_specific_tournament_balls_used_l4051_405150

/-- A tennis tournament with specific rules for ball usage -/
structure TennisTournament where
  rounds : Nat
  games_per_round : List Nat
  cans_per_game : Nat
  balls_per_can : Nat

/-- Calculate the total number of tennis balls used in a tournament -/
def total_balls_used (t : TennisTournament) : Nat :=
  (t.games_per_round.sum * t.cans_per_game * t.balls_per_can)

/-- Theorem: The total number of tennis balls used in the specific tournament is 225 -/
theorem specific_tournament_balls_used :
  let t : TennisTournament := {
    rounds := 4,
    games_per_round := [8, 4, 2, 1],
    cans_per_game := 5,
    balls_per_can := 3
  }
  total_balls_used t = 225 := by
  sorry


end NUMINAMATH_CALUDE_specific_tournament_balls_used_l4051_405150


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l4051_405102

-- Define the complex number z
def z : ℂ := (2 - Complex.I) * (1 - Complex.I)

-- Theorem statement
theorem z_in_fourth_quadrant :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l4051_405102


namespace NUMINAMATH_CALUDE_halves_to_one_and_half_l4051_405152

theorem halves_to_one_and_half :
  (3 : ℚ) / 2 / ((1 : ℚ) / 2) = 3 :=
sorry

end NUMINAMATH_CALUDE_halves_to_one_and_half_l4051_405152


namespace NUMINAMATH_CALUDE_wifes_ring_to_first_ring_ratio_l4051_405168

/-- The cost of Jim's first ring in dollars -/
def first_ring_cost : ℝ := 10000

/-- The cost of Jim's wife's ring in dollars -/
def wifes_ring_cost : ℝ := 20000

/-- The amount Jim is out of pocket in dollars -/
def out_of_pocket : ℝ := 25000

/-- Theorem stating the ratio of the cost of Jim's wife's ring to the cost of the first ring -/
theorem wifes_ring_to_first_ring_ratio :
  wifes_ring_cost / first_ring_cost = 2 :=
by
  have h1 : first_ring_cost + wifes_ring_cost - first_ring_cost / 2 = out_of_pocket := by sorry
  sorry

#check wifes_ring_to_first_ring_ratio

end NUMINAMATH_CALUDE_wifes_ring_to_first_ring_ratio_l4051_405168


namespace NUMINAMATH_CALUDE_vector_at_t_zero_l4051_405196

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  point : ℝ → ℝ × ℝ × ℝ

/-- Given conditions for the parameterized line -/
def line_conditions (L : ParameterizedLine) : Prop :=
  L.point 1 = (2, 5, 7) ∧ L.point 4 = (8, -7, 1)

theorem vector_at_t_zero 
  (L : ParameterizedLine) 
  (h : line_conditions L) : 
  L.point 0 = (0, 9, 9) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_t_zero_l4051_405196


namespace NUMINAMATH_CALUDE_gcd_1908_4187_l4051_405127

theorem gcd_1908_4187 : Nat.gcd 1908 4187 = 53 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1908_4187_l4051_405127


namespace NUMINAMATH_CALUDE_unit_digit_of_expression_l4051_405160

theorem unit_digit_of_expression : ∃ n : ℕ, n % 10 = 4 ∧ 
  n = (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_expression_l4051_405160


namespace NUMINAMATH_CALUDE_no_function_satisfies_property_l4051_405175

-- Define the property that we want to disprove
def HasProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f x) = x^2 - 2

-- Theorem stating that no such function exists
theorem no_function_satisfies_property :
  ¬ ∃ f : ℝ → ℝ, HasProperty f :=
sorry

end NUMINAMATH_CALUDE_no_function_satisfies_property_l4051_405175


namespace NUMINAMATH_CALUDE_blocks_and_colors_l4051_405182

theorem blocks_and_colors (total_blocks : ℕ) (blocks_per_color : ℕ) (colors_used : ℕ) : 
  total_blocks = 49 → 
  blocks_per_color = 7 → 
  total_blocks = blocks_per_color * colors_used → 
  colors_used = 7 := by
sorry

end NUMINAMATH_CALUDE_blocks_and_colors_l4051_405182


namespace NUMINAMATH_CALUDE_remaining_area_approx_l4051_405192

/-- Represents a circular grass plot with a straight path cutting through it. -/
structure GrassPlot where
  diameter : ℝ
  pathWidth : ℝ
  pathEdgeDistance : ℝ

/-- Calculates the remaining grass area of the plot after the path is cut through. -/
def remainingGrassArea (plot : GrassPlot) : ℝ :=
  sorry

/-- Theorem stating that for the given dimensions, the remaining grass area is approximately 56π + 17 square feet. -/
theorem remaining_area_approx (plot : GrassPlot) 
  (h1 : plot.diameter = 16)
  (h2 : plot.pathWidth = 4)
  (h3 : plot.pathEdgeDistance = 2) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |remainingGrassArea plot - (56 * Real.pi + 17)| < ε :=
sorry

end NUMINAMATH_CALUDE_remaining_area_approx_l4051_405192


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l4051_405115

/-- Calculates the total surface area of a cube with holes --/
def cube_surface_area_with_holes (cube_edge_length : ℝ) (hole_side_length : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge_length^2
  let hole_area := 6 * hole_side_length^2
  let exposed_internal_area := 6 * 4 * hole_side_length^2
  original_surface_area - hole_area + exposed_internal_area

/-- Theorem stating the total surface area of the cube with holes --/
theorem cube_with_holes_surface_area :
  cube_surface_area_with_holes 5 2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l4051_405115


namespace NUMINAMATH_CALUDE_factorial_ratio_l4051_405114

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 47 = 117600 := by sorry

end NUMINAMATH_CALUDE_factorial_ratio_l4051_405114


namespace NUMINAMATH_CALUDE_ten_person_meeting_exchanges_l4051_405165

/-- The number of business card exchanges in a group meeting -/
def business_card_exchanges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 10 people, where each person exchanges business cards
    with every other person exactly once, the total number of exchanges is 45. -/
theorem ten_person_meeting_exchanges :
  business_card_exchanges 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_person_meeting_exchanges_l4051_405165


namespace NUMINAMATH_CALUDE_number_difference_l4051_405121

theorem number_difference (a b c : ℝ) : 
  a = 2 * b ∧ 
  a = 3 * c ∧ 
  (a + b + c) / 3 = 88 → 
  a - c = 96 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l4051_405121


namespace NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l4051_405143

/-- Given a quadratic equation equivalent to 5x - 2 = 3x^2, 
    prove that the coefficient of the linear term is -5 -/
theorem linear_coefficient_of_quadratic (a b c : ℝ) : 
  (5 : ℝ) * x - 2 = 3 * x^2 → 
  a * x^2 + b * x + c = 0 → 
  c = 2 →
  b = -5 := by
  sorry

#check linear_coefficient_of_quadratic

end NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l4051_405143


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l4051_405181

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (pos_p : p > 0) (pos_q : q > 0) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (p_neq_q : p ≠ q)
  (geom_seq : a^2 = p * q)
  (arith_seq : ∃ d : ℝ, b = p + d ∧ c = p + 2*d ∧ q = p + 3*d)
  : ∀ x : ℝ, b * x^2 - 2*a * x + c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l4051_405181


namespace NUMINAMATH_CALUDE_nova_annual_donation_l4051_405129

/-- Nova's monthly donation in dollars -/
def monthly_donation : ℕ := 1707

/-- Number of months in a year -/
def months_in_year : ℕ := 12

/-- Nova's total annual donation in dollars -/
def annual_donation : ℕ := monthly_donation * months_in_year

theorem nova_annual_donation :
  annual_donation = 20484 := by
  sorry

end NUMINAMATH_CALUDE_nova_annual_donation_l4051_405129


namespace NUMINAMATH_CALUDE_divisibility_of_S_l4051_405141

-- Define the conditions
def is_valid_prime_pair (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p > 3 ∧ q > 3 ∧ ∃ n : ℕ, q - p = 2^n ∨ p - q = 2^n

-- Define the function S
def S (p q m : ℕ) : ℕ := p^(2*m+1) + q^(2*m+1)

-- State the theorem
theorem divisibility_of_S (p q : ℕ) (h : is_valid_prime_pair p q) :
  ∀ m : ℕ, (3 : ℕ) ∣ S p q m :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_S_l4051_405141


namespace NUMINAMATH_CALUDE_reflected_light_ray_equation_l4051_405173

/-- Given an incident light ray following the line y = 2x + 1 and reflecting on the line y = x,
    the equation of the reflected light ray is x - 2y - 1 = 0 -/
theorem reflected_light_ray_equation (x y : ℝ) :
  (y = 2*x + 1) →  -- Incident light ray equation
  (y = x) →        -- Reflection line equation
  (x - 2*y - 1 = 0) -- Reflected light ray equation
  := by sorry

end NUMINAMATH_CALUDE_reflected_light_ray_equation_l4051_405173


namespace NUMINAMATH_CALUDE_division_problem_l4051_405139

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
    (h1 : dividend = 3086)
    (h2 : divisor = 85)
    (h3 : remainder = 26)
    (h4 : dividend = divisor * quotient + remainder) :
  quotient = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4051_405139


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l4051_405184

theorem sum_of_roots_cubic : ∃ (A B C : ℝ),
  (3 * A^3 - 9 * A^2 + 6 * A - 4 = 0) ∧
  (3 * B^3 - 9 * B^2 + 6 * B - 4 = 0) ∧
  (3 * C^3 - 9 * C^2 + 6 * C - 4 = 0) ∧
  (A + B + C = 3) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l4051_405184


namespace NUMINAMATH_CALUDE_nonagon_diagonals_count_l4051_405151

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals_nonagon : ℕ := 27

/-- A convex nonagon has 9 sides -/
def nonagon_sides : ℕ := 9

/-- The number of vertices each vertex can connect to (excluding itself and adjacent vertices) -/
def connections_per_vertex : ℕ := nonagon_sides - 3

theorem nonagon_diagonals_count :
  num_diagonals_nonagon = (nonagon_sides * connections_per_vertex) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_count_l4051_405151


namespace NUMINAMATH_CALUDE_min_ab_value_l4051_405149

theorem min_ab_value (a b : ℕ+) 
  (h : (fun x y : ℝ => x^2 + y^2 - 2*a*x + a^2*(1-b)) = 0 ↔ 
       (fun x y : ℝ => x^2 + y^2 - 2*y + 1 - a^2*b) = 0) : 
  (a : ℝ) * (b : ℝ) ≥ (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_min_ab_value_l4051_405149


namespace NUMINAMATH_CALUDE_expression_value_l4051_405195

theorem expression_value (x y : ℤ) (hx : x = -2) (hy : y = -4) :
  5 * (x - y)^2 - x * y = 12 := by sorry

end NUMINAMATH_CALUDE_expression_value_l4051_405195


namespace NUMINAMATH_CALUDE_expansion_simplification_l4051_405166

theorem expansion_simplification (x y : ℝ) :
  (2*x - y) * (2*x + 3*y) - (x + y) * (x - y) = 3*x^2 + 4*x*y - 2*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_simplification_l4051_405166


namespace NUMINAMATH_CALUDE_opposite_of_neg_six_l4051_405131

/-- The opposite of a real number -/
def opposite (a : ℝ) : ℝ := -a

/-- Theorem: The opposite of -6 is 6 -/
theorem opposite_of_neg_six : opposite (-6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_six_l4051_405131


namespace NUMINAMATH_CALUDE_square_sum_from_means_l4051_405164

theorem square_sum_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 20)
  (h_geometric : Real.sqrt (a * b) = Real.sqrt 96) :
  a^2 + b^2 = 1408 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l4051_405164


namespace NUMINAMATH_CALUDE_sqrt_difference_between_l4051_405146

theorem sqrt_difference_between (a b : ℝ) (h : a < b) : 
  ∃ (n k : ℕ), a < Real.sqrt n - Real.sqrt k ∧ Real.sqrt n - Real.sqrt k < b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_between_l4051_405146


namespace NUMINAMATH_CALUDE_original_number_proof_l4051_405148

theorem original_number_proof (x : ℝ) : 
  (x + 0.375 * x) - (x - 0.425 * x) = 85 → x = 106.25 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l4051_405148


namespace NUMINAMATH_CALUDE_annie_spending_l4051_405118

def television_count : ℕ := 5
def television_price : ℕ := 50
def figurine_count : ℕ := 10
def figurine_price : ℕ := 1

def total_spending : ℕ := television_count * television_price + figurine_count * figurine_price

theorem annie_spending :
  total_spending = 260 := by sorry

end NUMINAMATH_CALUDE_annie_spending_l4051_405118


namespace NUMINAMATH_CALUDE_axb_equals_bxa_l4051_405110

open Matrix

variable {n : ℕ}
variable (A B X : Matrix (Fin n) (Fin n) ℝ)

theorem axb_equals_bxa (h : A * X * B + A + B = 0) : A * X * B = B * X * A := by
  sorry

end NUMINAMATH_CALUDE_axb_equals_bxa_l4051_405110


namespace NUMINAMATH_CALUDE_system_solution_l4051_405190

theorem system_solution (x y : ℚ) : 
  (10 / (2 * x + 3 * y - 29) + 9 / (7 * x - 8 * y + 24) = 8) ∧ 
  ((2 * x + 3 * y - 29) / 2 = (7 * x - 8 * y) / 3 + 8) → 
  x = 5 ∧ y = 7 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l4051_405190


namespace NUMINAMATH_CALUDE_ollie_caught_five_fish_l4051_405186

/-- The number of fish caught by Ollie given the fishing results of Patrick and Angus -/
def ollies_fish (patrick_fish : ℕ) (angus_more_than_patrick : ℕ) (ollie_fewer_than_angus : ℕ) : ℕ :=
  patrick_fish + angus_more_than_patrick - ollie_fewer_than_angus

/-- Theorem stating that Ollie caught 5 fish given the problem conditions -/
theorem ollie_caught_five_fish :
  ollies_fish 8 4 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ollie_caught_five_fish_l4051_405186


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4051_405162

def A : Set Int := {-2, -1, 0, 1, 2}

def B : Set Int := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_of_A_and_B : A ∩ B = {-2, 0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4051_405162


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_Q_l4051_405185

theorem P_sufficient_not_necessary_Q :
  (∀ x y : ℝ, x + y ≠ 5 → (x ≠ 2 ∨ y ≠ 3)) ∧
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 3) ∧ x + y = 5) :=
by sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_Q_l4051_405185


namespace NUMINAMATH_CALUDE_circle_equation_k_range_l4051_405126

theorem circle_equation_k_range (x y k : ℝ) :
  (∃ r : ℝ, r > 0 ∧ ∀ x y, x^2 + y^2 - 2*x + y + k = 0 ↔ (x - 1)^2 + (y + 1/2)^2 = r^2) →
  k < 5/4 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_k_range_l4051_405126


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4051_405179

theorem polynomial_simplification (x : ℝ) :
  (5 * x^10 + 8 * x^8 + 3 * x^6) + (2 * x^12 + 3 * x^10 + x^8 + 4 * x^6 + 2 * x^2 + 7) =
  2 * x^12 + 8 * x^10 + 9 * x^8 + 7 * x^6 + 2 * x^2 + 7 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4051_405179


namespace NUMINAMATH_CALUDE_equation_solutions_l4051_405101

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 - 16 = 0 ↔ x = 3 ∨ x = -5) ∧
  (∀ x : ℝ, -2 * (x - 1)^3 = 16 ↔ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4051_405101


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l4051_405178

/-- Given a quadratic function f(x) = ax^2 + bx + c with vertex (5, -3) and
    one x-intercept at (1, 0), the x-coordinate of the other x-intercept is 9. -/
theorem other_x_intercept_of_quadratic 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 5 = -3 ∧ ∀ x, f x ≤ f 5) -- Vertex condition
  (h3 : f 1 = 0) -- Given x-intercept
  : ∃ x, x ≠ 1 ∧ f x = 0 ∧ x = 9 :=
by sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l4051_405178


namespace NUMINAMATH_CALUDE_number_division_problem_l4051_405171

theorem number_division_problem (x : ℚ) : x / 5 = 70 + x / 6 → x = 2100 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l4051_405171


namespace NUMINAMATH_CALUDE_johns_watermelon_weight_l4051_405112

theorem johns_watermelon_weight (michael_weight : ℕ) (clay_factor : ℕ) (john_factor : ℚ) :
  michael_weight = 8 →
  clay_factor = 3 →
  john_factor = 1/2 →
  (↑michael_weight * ↑clay_factor * john_factor : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_johns_watermelon_weight_l4051_405112


namespace NUMINAMATH_CALUDE_complex_distance_problem_l4051_405107

theorem complex_distance_problem (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) :
  Complex.abs (z - 1) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_problem_l4051_405107


namespace NUMINAMATH_CALUDE_current_for_given_resistance_l4051_405106

/-- Represents the relationship between voltage (V), current (I), and resistance (R) -/
def ohms_law (V I R : ℝ) : Prop := V = I * R

theorem current_for_given_resistance (V I R : ℝ) (h1 : V = 48) (h2 : R = 12) (h3 : ohms_law V I R) :
  I = 4 := by
  sorry

end NUMINAMATH_CALUDE_current_for_given_resistance_l4051_405106


namespace NUMINAMATH_CALUDE_circle_radius_l4051_405167

theorem circle_radius (x y : ℝ) : 
  (x^2 - 8*x + y^2 + 6*y + 1 = 0) → 
  ∃ (h k r : ℝ), r = 2 * Real.sqrt 6 ∧ 
    ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 - 8*x + y^2 + 6*y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l4051_405167


namespace NUMINAMATH_CALUDE_expression_value_l4051_405189

theorem expression_value (p q r s : ℝ) 
  (h1 : p^2 / q^3 = 4 / 5)
  (h2 : r^3 / s^2 = 7 / 9) :
  11 / (7 - r^3 / s^2) + (2 * q^3 - p^2) / (2 * q^3 + p^2) = 123 / 56 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4051_405189


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_value_l4051_405116

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

theorem tangent_slope_implies_a_value :
  ∀ a b : ℝ, f_derivative a 1 = -1 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_value_l4051_405116


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l4051_405194

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 0, -4; 3, -1, 5; 1, 2, 3]
  Matrix.det A = -54 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l4051_405194


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l4051_405138

/-- Given one-digit numbers A and B satisfying 8AA4 - BBB = BBBB, prove their sum is 12 -/
theorem sum_of_special_numbers (A B : ℕ) : 
  A < 10 → B < 10 → 
  8000 + 100 * A + 10 * A + 4 - (100 * B + 10 * B + B) = 1000 * B + 100 * B + 10 * B + B →
  A + B = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l4051_405138
