import Mathlib

namespace hardcover_book_weight_l671_67117

/-- Proves that the weight of each hardcover book is 1/2 pound given the problem conditions -/
theorem hardcover_book_weight :
  let bookcase_limit : ℚ := 80
  let hardcover_count : ℕ := 70
  let textbook_count : ℕ := 30
  let textbook_weight : ℚ := 2
  let knickknack_count : ℕ := 3
  let knickknack_weight : ℚ := 6
  let overweight : ℚ := 33
  let hardcover_weight : ℚ := 1/2

  hardcover_count * hardcover_weight + 
  textbook_count * textbook_weight + 
  knickknack_count * knickknack_weight = 
  bookcase_limit + overweight :=
by
  sorry

#check hardcover_book_weight

end hardcover_book_weight_l671_67117


namespace smallest_of_three_consecutive_odds_l671_67168

theorem smallest_of_three_consecutive_odds (x y z : ℤ) : 
  (∃ k : ℤ, x = 2*k + 1) →  -- x is odd
  y = x + 2 →               -- y is the next consecutive odd number
  z = y + 2 →               -- z is the next consecutive odd number after y
  x + y + z = 69 →          -- their sum is 69
  x = 21                    -- x (the smallest) is 21
:= by sorry

end smallest_of_three_consecutive_odds_l671_67168


namespace pentagon_angle_sum_l671_67147

/-- Given a pentagon with two additional interior lines forming angles as described,
    prove that the sum of two specific angles is 138°. -/
theorem pentagon_angle_sum (P Q R x z : ℝ) : 
  P = 34 → Q = 76 → R = 28 → 
  (360 - x) + Q + P + 90 + (118 - z) = 540 →
  x + z = 138 := by
  sorry

end pentagon_angle_sum_l671_67147


namespace ryan_english_study_hours_l671_67110

/-- Ryan's daily study schedule -/
structure StudySchedule where
  chinese_hours : ℕ
  english_hours : ℕ
  hours_difference : ℕ

/-- Theorem: Ryan's English study hours -/
theorem ryan_english_study_hours (schedule : StudySchedule)
  (h1 : schedule.chinese_hours = 5)
  (h2 : schedule.hours_difference = 2)
  (h3 : schedule.english_hours = schedule.chinese_hours + schedule.hours_difference) :
  schedule.english_hours = 7 := by
  sorry

end ryan_english_study_hours_l671_67110


namespace equation_solution_l671_67126

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end equation_solution_l671_67126


namespace park_area_l671_67128

/-- The area of a rectangular park given its length-to-breadth ratio and cycling time around its perimeter -/
theorem park_area (L B : ℝ) (h1 : L / B = 1 / 3) 
  (h2 : 2 * (L + B) = (12 * 1000 / 60) * 4) : L * B = 30000 := by
  sorry

end park_area_l671_67128


namespace intersection_condition_l671_67165

-- Define the sets A and B
def A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + a * p.1 - p.2 + 2 = 0}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + 1 = 0 ∧ p.1 > 0}

-- State the theorem
theorem intersection_condition (a : ℝ) :
  (A a ∩ B).Nonempty ↔ a ≤ 0 := by sorry

end intersection_condition_l671_67165


namespace expression_calculation_l671_67199

theorem expression_calculation : 1453 - 250 * 2 + 130 / 5 = 979 := by
  sorry

end expression_calculation_l671_67199


namespace boat_breadth_l671_67149

theorem boat_breadth (length : Real) (sink_depth : Real) (man_mass : Real) 
  (g : Real) (water_density : Real) : Real :=
by
  -- Define the given constants
  have h1 : length = 8 := by sorry
  have h2 : sink_depth = 0.01 := by sorry
  have h3 : man_mass = 160 := by sorry
  have h4 : g = 9.81 := by sorry
  have h5 : water_density = 1000 := by sorry

  -- Calculate the breadth
  let weight := man_mass * g
  let volume := weight / (water_density * g)
  let breadth := volume / (length * sink_depth)

  -- Prove that the breadth is equal to 2
  have h6 : breadth = 2 := by sorry

  exact breadth

/- Theorem statement: The breadth of a boat with length 8 m that sinks by 1 cm 
   when a 160 kg man gets on it is 2 m, given that the acceleration due to 
   gravity is 9.81 m/s² and the density of water is 1000 kg/m³. -/

end boat_breadth_l671_67149


namespace whatsis_equals_so_equals_four_l671_67138

/-- Given positive real numbers, prove that whatsis equals so and both equal 4 -/
theorem whatsis_equals_so_equals_four
  (whosis whatsis is so : ℝ)
  (h1 : whosis > 0)
  (h2 : whatsis > 0)
  (h3 : is > 0)
  (h4 : so > 0)
  (h5 : whosis = is)
  (h6 : so = so)
  (h7 : whosis = so)
  (h8 : so - is = 2)
  : whatsis = so ∧ so = 4 := by
  sorry

end whatsis_equals_so_equals_four_l671_67138


namespace f_increasing_and_range_l671_67187

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem f_increasing_and_range :
  (∀ a : ℝ, Monotone (f a)) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x = -(f a (-x))) →
    Set.range (f a) = Set.Ioo (-1/2 : ℝ) (1/2 : ℝ)) :=
by sorry

end f_increasing_and_range_l671_67187


namespace hyperbola_asymptotes_l671_67184

/-- Given a hyperbola E with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and eccentricity √7/2,
    prove that its asymptotes have the equation y = ±(√3/2)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt 7 / 2
  let c := e * a
  (c^2 / a^2 = 1 + (b/a)^2) →
  (b/a = Real.sqrt 3 / 2) :=
by sorry

end hyperbola_asymptotes_l671_67184


namespace divisibility_criterion_l671_67188

theorem divisibility_criterion (x : ℤ) : 
  (∃ k : ℤ, 3 * x + 7 = 14 * k) ↔ (∃ t : ℤ, x = 14 * t + 7) := by sorry

end divisibility_criterion_l671_67188


namespace square_area_sum_l671_67153

theorem square_area_sum (a b : ℕ) (h : a^2 + b^2 = 400) : a + b = 28 := by
  sorry

end square_area_sum_l671_67153


namespace exponent_multiplication_l671_67134

theorem exponent_multiplication (x : ℝ) : x^2 * x^4 = x^6 := by
  sorry

end exponent_multiplication_l671_67134


namespace sum_of_distances_is_36_root_3_l671_67148

/-- A regular hexagon with side length 12 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 12)

/-- A point on one side of the hexagon -/
structure PointOnSide (h : RegularHexagon) :=
  (point : ℝ × ℝ)
  (on_side : point.1 ≥ 0 ∧ point.1 ≤ h.side_length)

/-- The sum of distances from a point on one side to lines containing other sides -/
def sum_of_distances (h : RegularHexagon) (p : PointOnSide h) : ℝ :=
  sorry

/-- Theorem stating the sum of distances is 36√3 -/
theorem sum_of_distances_is_36_root_3 (h : RegularHexagon) (p : PointOnSide h) :
  sum_of_distances h p = 36 * Real.sqrt 3 :=
sorry

end sum_of_distances_is_36_root_3_l671_67148


namespace tan_eleven_pi_fourths_l671_67119

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by
  sorry

end tan_eleven_pi_fourths_l671_67119


namespace bd_length_l671_67167

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define our specific quadrilateral
def quadABCD : Quadrilateral :=
  { A := sorry,
    B := sorry,
    C := sorry,
    D := sorry }

-- State the theorem
theorem bd_length :
  let ABCD := quadABCD
  (length ABCD.A ABCD.B = 5) →
  (length ABCD.B ABCD.C = 17) →
  (length ABCD.C ABCD.D = 5) →
  (length ABCD.D ABCD.A = 9) →
  ∃ n : ℕ, (length ABCD.B ABCD.D = n) ∧ (n = 13) :=
sorry

end bd_length_l671_67167


namespace consecutive_integers_median_l671_67137

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (median : ℕ) : 
  n = 36 →
  sum = 3125 →
  (∃ (start : ℤ), sum = (start + start + n - 1) * n / 2) →
  median = 89 :=
by
  sorry

end consecutive_integers_median_l671_67137


namespace at_least_one_positive_discriminant_l671_67189

theorem at_least_one_positive_discriminant (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (4 * b^2 - 4 * a * c > 0) ∨ 
  (4 * c^2 - 4 * a * b > 0) ∨ 
  (4 * a^2 - 4 * b * c > 0) := by
  sorry

end at_least_one_positive_discriminant_l671_67189


namespace line_moved_down_by_two_l671_67166

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Moves a line vertically by a given amount -/
def moveVertically (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - amount }

theorem line_moved_down_by_two :
  let original := Line.mk 3 0
  let moved := moveVertically original 2
  moved = Line.mk 3 (-2) := by sorry

end line_moved_down_by_two_l671_67166


namespace equal_area_division_l671_67143

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The area of the right part of a triangle divided by a vertical line -/
def rightAreaDivided (t : Triangle) (a : ℝ) : ℝ := sorry

/-- Theorem: For a triangle ABC with vertices A = (0,2), B = (0,0), and C = (6,0),
    where line AC is horizontal, the vertical line x = 3 divides the triangle
    into two regions of equal area -/
theorem equal_area_division (t : Triangle) 
    (h1 : t.A = (0, 2)) 
    (h2 : t.B = (0, 0)) 
    (h3 : t.C = (6, 0)) 
    (h4 : t.A.2 = t.C.2) : -- Line AC is horizontal
  2 * rightAreaDivided t 3 = triangleArea t := by sorry

end equal_area_division_l671_67143


namespace min_value_expression_l671_67135

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 18 := by
  sorry

end min_value_expression_l671_67135


namespace b_age_is_ten_l671_67133

/-- Given three people a, b, and c, with the following conditions:
  1. a is two years older than b
  2. b is twice as old as c
  3. The sum of their ages is 27
  Prove that b is 10 years old. -/
theorem b_age_is_ten (a b c : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : a + b + c = 27) :
  b = 10 := by
  sorry

end b_age_is_ten_l671_67133


namespace arithmetic_sequence_sum_times_three_l671_67123

theorem arithmetic_sequence_sum_times_three : 
  let a := 75  -- first term
  let d := 2   -- common difference
  let n := 5   -- number of terms
  3 * (a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d)) = 1185 := by
sorry

end arithmetic_sequence_sum_times_three_l671_67123


namespace contractor_wage_l671_67156

def contractor_problem (total_days : ℕ) (absent_days : ℕ) (fine_per_day : ℚ) (total_amount : ℚ) : Prop :=
  ∃ (daily_wage : ℚ),
    (total_days - absent_days) * daily_wage - absent_days * fine_per_day = total_amount ∧
    daily_wage = 25

theorem contractor_wage :
  contractor_problem 30 8 (25/2) 490 :=
sorry

end contractor_wage_l671_67156


namespace collinear_points_l671_67125

theorem collinear_points (k : ℝ) : 
  let PA : ℝ × ℝ := (k, 12)
  let PB : ℝ × ℝ := (4, 5)
  let PC : ℝ × ℝ := (10, k)
  (k = -2 ∨ k = 11) ↔ 
    ∃ (t : ℝ), (PC.1 - PA.1, PC.2 - PA.2) = t • (PB.1 - PA.1, PB.2 - PA.2) :=
by sorry

end collinear_points_l671_67125


namespace prob_sum_25_l671_67192

/-- Represents a 20-faced die with specific numbering --/
structure Die :=
  (faces : Finset ℕ)
  (blank : Bool)
  (proper : faces.card + (if blank then 1 else 0) = 20)

/-- The first die with faces 1-19 and one blank --/
def die1 : Die :=
  { faces := Finset.range 20 \ {0},
    blank := true,
    proper := by sorry }

/-- The second die with faces 1-7, 9-19 and one blank --/
def die2 : Die :=
  { faces := (Finset.range 20 \ {0, 8}),
    blank := true,
    proper := by sorry }

/-- The probability of an event given the sample space --/
def probability (event : Finset (ℕ × ℕ)) (sample_space : Finset (ℕ × ℕ)) : ℚ :=
  event.card / sample_space.card

/-- The set of all possible outcomes when rolling both dice --/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product die1.faces die2.faces

/-- The set of outcomes where the sum is 25 --/
def sum_25_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun p => p.1 + p.2 = 25)

/-- The main theorem stating the probability of rolling a sum of 25 --/
theorem prob_sum_25 :
  probability sum_25_outcomes all_outcomes = 13 / 400 := by sorry

end prob_sum_25_l671_67192


namespace sqrt_product_equality_l671_67182

theorem sqrt_product_equality : Real.sqrt (4 / 75) * Real.sqrt 3 = 2 / 5 := by
  sorry

end sqrt_product_equality_l671_67182


namespace isosceles_triangle_perimeter_l671_67180

-- Define an isosceles triangle with side lengths 5, 5, and 2
def isoscelesTriangle (a b c : ℝ) : Prop :=
  a = 5 ∧ b = 5 ∧ c = 2

-- Define the perimeter of a triangle
def trianglePerimeter (a b c : ℝ) : ℝ :=
  a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, isoscelesTriangle a b c → trianglePerimeter a b c = 12 := by
  sorry

end isosceles_triangle_perimeter_l671_67180


namespace tinas_tangerines_l671_67146

/-- Represents the contents of Tina's bag -/
structure BagContents where
  apples : Nat
  oranges : Nat
  tangerines : Nat

/-- The condition after removing some fruits -/
def condition (b : BagContents) : Prop :=
  b.tangerines - 10 = (b.oranges - 2) + 4

/-- Theorem stating the number of tangerines in Tina's bag -/
theorem tinas_tangerines :
  ∃ (b : BagContents), b.apples = 9 ∧ b.oranges = 5 ∧ condition b ∧ b.tangerines = 17 :=
by sorry

end tinas_tangerines_l671_67146


namespace replaced_person_age_l671_67124

/-- Given a group of 10 people, if replacing one person with a 14-year-old
    decreases the average age by 3 years, then the replaced person was 44 years old. -/
theorem replaced_person_age (group_size : ℕ) (new_person_age : ℕ) (avg_decrease : ℕ) :
  group_size = 10 →
  new_person_age = 14 →
  avg_decrease = 3 →
  ∃ (replaced_age : ℕ),
    (group_size * (replaced_age / group_size) - 
     (group_size * ((replaced_age / group_size) - avg_decrease))) =
    (replaced_age - new_person_age) ∧
    replaced_age = 44 := by
  sorry

end replaced_person_age_l671_67124


namespace solve_equation_l671_67170

theorem solve_equation (x : ℝ) (h : 5 * x - 3 = 15 * x + 21) : 3 * (x + 10) = 22.8 := by
  sorry

end solve_equation_l671_67170


namespace sequence_and_max_sum_l671_67112

def f (x : ℝ) := -x^2 + 7*x

def S (n : ℕ) := f n

def a (n : ℕ+) := S n - S (n-1)

theorem sequence_and_max_sum :
  (∀ n : ℕ+, a n = -2*(n:ℝ) + 8) ∧
  (∃ n : ℕ+, S n = 12 ∧ ∀ m : ℕ+, S m ≤ 12) :=
sorry

end sequence_and_max_sum_l671_67112


namespace octagon_diagonals_l671_67121

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l671_67121


namespace inequality_chain_l671_67100

theorem inequality_chain (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > a*x ∧ a*x > a^2 := by
  sorry

end inequality_chain_l671_67100


namespace edgar_cookies_count_l671_67108

/-- The number of cookies a paper bag can hold -/
def cookies_per_bag : ℕ := 16

/-- The number of paper bags Edgar needs -/
def bags_needed : ℕ := 19

/-- The total number of cookies Edgar bought -/
def total_cookies : ℕ := cookies_per_bag * bags_needed

theorem edgar_cookies_count : total_cookies = 304 := by
  sorry

end edgar_cookies_count_l671_67108


namespace consecutive_cubes_divisibility_l671_67140

theorem consecutive_cubes_divisibility (n : ℕ) (h : ¬ 3 ∣ n) :
  9 * n ∣ ((n - 1)^3 + n^3 + (n + 1)^3) :=
by sorry

end consecutive_cubes_divisibility_l671_67140


namespace train_passing_platform_l671_67193

/-- Given a train of length 240 meters passing a pole in 24 seconds,
    this theorem proves that it takes 89 seconds to pass a platform of length 650 meters. -/
theorem train_passing_platform
  (train_length : ℝ)
  (pole_passing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 240)
  (h2 : pole_passing_time = 24)
  (h3 : platform_length = 650)
  : (train_length + platform_length) / (train_length / pole_passing_time) = 89 := by
  sorry

end train_passing_platform_l671_67193


namespace min_value_expression_lower_bound_attainable_l671_67142

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  ((2*a + b)^2 + (b - 2*c)^2 + (c - a)^2) / b^2 ≥ 4/3 :=
by sorry

theorem lower_bound_attainable (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  ∃ (a' b' c' : ℝ), b' > c' ∧ c' > a' ∧ b' ≠ 0 ∧
    ((2*a' + b')^2 + (b' - 2*c')^2 + (c' - a')^2) / b'^2 = 4/3 :=
by sorry

end min_value_expression_lower_bound_attainable_l671_67142


namespace cube_root_of_product_l671_67115

theorem cube_root_of_product (n : ℕ) : (2^9 * 5^3 * 7^6 : ℝ)^(1/3) = 1960 := by
  sorry

end cube_root_of_product_l671_67115


namespace walkway_area_is_296_l671_67144

/-- Represents the garden layout --/
structure Garden where
  rows : Nat
  columns : Nat
  bed_width : Nat
  bed_height : Nat
  walkway_width : Nat

/-- Calculates the total area of walkways in the garden --/
def walkway_area (g : Garden) : Nat :=
  let total_width := g.columns * g.bed_width + (g.columns + 1) * g.walkway_width
  let total_height := g.rows * g.bed_height + (g.rows + 1) * g.walkway_width
  let total_area := total_width * total_height
  let bed_area := g.rows * g.columns * g.bed_width * g.bed_height
  total_area - bed_area

/-- The theorem to be proved --/
theorem walkway_area_is_296 (g : Garden) :
  g.rows = 4 ∧ g.columns = 3 ∧ g.bed_width = 4 ∧ g.bed_height = 3 ∧ g.walkway_width = 2 →
  walkway_area g = 296 := by
  sorry

end walkway_area_is_296_l671_67144


namespace malcolm_white_lights_l671_67150

/-- Represents the brightness levels of lights --/
inductive Brightness
  | Low
  | Medium
  | High

/-- Represents different types of lights --/
inductive LightType
  | White
  | Red
  | Yellow
  | Blue
  | Green
  | Purple

/-- Returns the brightness value of a given brightness level --/
def brightnessValue (b : Brightness) : Rat :=
  match b with
  | Brightness.Low => 1/2
  | Brightness.Medium => 1
  | Brightness.High => 3/2

/-- Calculates the total brightness of a given number of lights with a specific brightness --/
def totalBrightness (count : Nat) (b : Brightness) : Rat :=
  count * brightnessValue b

/-- Represents Malcolm's initial light purchase --/
structure InitialPurchase where
  redCount : Nat
  yellowCount : Nat
  blueCount : Nat
  greenCount : Nat
  purpleCount : Nat
  redBrightness : Brightness
  yellowBrightness : Brightness
  blueBrightness : Brightness
  greenBrightness : Brightness
  purpleBrightness : Brightness

/-- Represents the additional lights Malcolm needs to buy --/
structure AdditionalPurchase where
  additionalBluePercentage : Rat
  additionalRedCount : Nat

/-- Theorem: Given Malcolm's initial and additional light purchases, prove that he had 38 white lights initially --/
theorem malcolm_white_lights (initial : InitialPurchase) (additional : AdditionalPurchase) :
  initial.redCount = 16 ∧
  initial.yellowCount = 4 ∧
  initial.blueCount = 2 * initial.yellowCount ∧
  initial.greenCount = 8 ∧
  initial.purpleCount = 3 ∧
  initial.redBrightness = Brightness.Low ∧
  initial.yellowBrightness = Brightness.High ∧
  initial.blueBrightness = Brightness.Medium ∧
  initial.greenBrightness = Brightness.Low ∧
  initial.purpleBrightness = Brightness.High ∧
  additional.additionalBluePercentage = 1/4 ∧
  additional.additionalRedCount = 10 →
  ∃ (whiteCount : Nat), whiteCount = 38 ∧
    totalBrightness whiteCount Brightness.Medium =
      totalBrightness initial.redCount initial.redBrightness +
      totalBrightness initial.yellowCount initial.yellowBrightness +
      totalBrightness initial.blueCount initial.blueBrightness +
      totalBrightness initial.greenCount initial.greenBrightness +
      totalBrightness initial.purpleCount initial.purpleBrightness +
      totalBrightness (Nat.ceil (additional.additionalBluePercentage * initial.blueCount)) Brightness.Medium +
      totalBrightness additional.additionalRedCount Brightness.Low :=
by
  sorry

end malcolm_white_lights_l671_67150


namespace absolute_value_inequality_solution_set_l671_67191

theorem absolute_value_inequality_solution_set :
  ∀ x : ℝ, |x^2 - 4| ≤ x + 2 ↔ (1 ≤ x ∧ x ≤ 3) ∨ x = -2 := by sorry

end absolute_value_inequality_solution_set_l671_67191


namespace jaco_gift_budget_l671_67169

/-- Calculates the budget for each parent's gift given a total budget, number of friends, 
    cost per friend's gift, and number of parents. -/
def parent_gift_budget (total_budget : ℚ) (num_friends : ℕ) (friend_gift_cost : ℚ) (num_parents : ℕ) : ℚ :=
  (total_budget - (num_friends : ℚ) * friend_gift_cost) / (num_parents : ℚ)

/-- Proves that given a total budget of $100, 8 friends' gifts costing $9 each, 
    and equal-cost gifts for two parents, the budget for each parent's gift is $14. -/
theorem jaco_gift_budget : parent_gift_budget 100 8 9 2 = 14 := by
  sorry

end jaco_gift_budget_l671_67169


namespace chairs_left_is_54_l671_67105

/-- The number of chairs left in Rodrigo's classroom after borrowing -/
def chairs_left : ℕ :=
  let red_chairs : ℕ := 4
  let yellow_chairs : ℕ := 2 * red_chairs
  let blue_chairs : ℕ := 3 * yellow_chairs
  let green_chairs : ℕ := blue_chairs / 2
  let orange_chairs : ℕ := green_chairs + 2
  let total_chairs : ℕ := red_chairs + yellow_chairs + blue_chairs + green_chairs + orange_chairs
  let borrowed_chairs : ℕ := 5 + 3
  total_chairs - borrowed_chairs

theorem chairs_left_is_54 : chairs_left = 54 := by
  sorry

end chairs_left_is_54_l671_67105


namespace absolute_value_equation_solution_l671_67106

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| = x^2 - 5*x + 6 ∧ x = 2 - Real.sqrt 2 := by
sorry

end absolute_value_equation_solution_l671_67106


namespace oxygen_mass_percentage_l671_67183

/-- Given a compound with a mass percentage of oxygen, prove that the ratio of oxygen mass to total mass equals the mass percentage expressed as a decimal. -/
theorem oxygen_mass_percentage (compound_mass total_mass oxygen_mass : ℝ) 
  (h1 : compound_mass > 0)
  (h2 : total_mass = compound_mass)
  (h3 : oxygen_mass > 0)
  (h4 : oxygen_mass ≤ total_mass)
  (h5 : (oxygen_mass / total_mass) * 100 = 58.33) :
  oxygen_mass / total_mass = 0.5833 := by
sorry

end oxygen_mass_percentage_l671_67183


namespace kingdom_guards_bound_l671_67198

/-- Represents a road between two castles with a number of guards -/
structure Road where
  castle1 : Nat
  castle2 : Nat
  guards : Nat

/-- Kingdom Wierdo with its castles and roads -/
structure Kingdom where
  N : Nat
  roads : List Road

/-- Check if the kingdom satisfies the guard policy -/
def satisfiesPolicy (k : Kingdom) : Prop :=
  (∀ r ∈ k.roads, r.guards ≤ 4) ∧
  (∀ a b c : Nat, a < k.N → b < k.N → c < k.N →
    (∃ r ∈ k.roads, (r.castle1 = a ∧ r.castle2 = b) ∨ (r.castle1 = b ∧ r.castle2 = a)) →
    (∃ r ∈ k.roads, (r.castle1 = b ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = b)) →
    (∃ r ∈ k.roads, (r.castle1 = a ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = a)) →
    ∀ r ∈ k.roads, ((r.castle1 = a ∧ r.castle2 = b) ∨ (r.castle1 = b ∧ r.castle2 = a) ∨
                    (r.castle1 = b ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = b) ∨
                    (r.castle1 = a ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = a)) →
      r.guards ≤ 3) ∧
  (∀ a b c d : Nat, a < k.N → b < k.N → c < k.N → d < k.N →
    (∃ r ∈ k.roads, (r.castle1 = a ∧ r.castle2 = b) ∨ (r.castle1 = b ∧ r.castle2 = a)) →
    (∃ r ∈ k.roads, (r.castle1 = a ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = a)) →
    (∃ r ∈ k.roads, (r.castle1 = a ∧ r.castle2 = d) ∨ (r.castle1 = d ∧ r.castle2 = a)) →
    (∃ r ∈ k.roads, (r.castle1 = b ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = b)) →
    (∃ r ∈ k.roads, (r.castle1 = b ∧ r.castle2 = d) ∨ (r.castle1 = d ∧ r.castle2 = b)) →
    (∃ r ∈ k.roads, (r.castle1 = c ∧ r.castle2 = d) ∨ (r.castle1 = d ∧ r.castle2 = c)) →
    ¬(∀ r ∈ k.roads, ((r.castle1 = a ∧ r.castle2 = b) ∨ (r.castle1 = b ∧ r.castle2 = a) ∨
                      (r.castle1 = a ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = a) ∨
                      (r.castle1 = a ∧ r.castle2 = d) ∨ (r.castle1 = d ∧ r.castle2 = a)) →
      r.guards = 3))

theorem kingdom_guards_bound (k : Kingdom) (h : satisfiesPolicy k) :
  (k.roads.map (·.guards)).sum ≤ k.N ^ 2 :=
sorry

end kingdom_guards_bound_l671_67198


namespace min_value_fraction_sum_l671_67101

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  1 / (a + 1) + 1 / b ≥ 1 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 3 ∧ 1 / (a₀ + 1) + 1 / b₀ = 1 := by
  sorry

end min_value_fraction_sum_l671_67101


namespace v_formation_sum_l671_67130

def isValidDigit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 9

def isDistinct (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

theorem v_formation_sum (a b c d e : ℕ)
  (h_valid : isValidDigit a ∧ isValidDigit b ∧ isValidDigit c ∧ isValidDigit d ∧ isValidDigit e)
  (h_distinct : isDistinct a b c d e)
  (h_left_sum : a + b + c = 16)
  (h_right_sum : c + d = 11) :
  a + b + c + d + e = 18 :=
sorry

end v_formation_sum_l671_67130


namespace unique_modular_solution_l671_67132

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ -250 ≡ n [ZMOD 31] := by sorry

end unique_modular_solution_l671_67132


namespace eighteen_percent_of_500_is_90_l671_67194

theorem eighteen_percent_of_500_is_90 (x : ℝ) : 
  (18 / 100) * x = 90 → x = 500 := by sorry

end eighteen_percent_of_500_is_90_l671_67194


namespace pairwise_coprime_fraction_squares_l671_67162

theorem pairwise_coprime_fraction_squares (x y z : ℕ+) 
  (h_coprime_xy : Nat.Coprime x.val y.val)
  (h_coprime_yz : Nat.Coprime y.val z.val)
  (h_coprime_xz : Nat.Coprime x.val z.val)
  (h_eq : (1 : ℚ) / x.val + (1 : ℚ) / y.val = (1 : ℚ) / z.val) :
  ∃ (a b c : ℕ), 
    (x.val + y.val = a ^ 2) ∧ 
    (x.val - z.val = b ^ 2) ∧ 
    (y.val - z.val = c ^ 2) := by
  sorry

end pairwise_coprime_fraction_squares_l671_67162


namespace a_explicit_form_l671_67163

def a : ℕ → ℤ
  | 0 => -1
  | 1 => 1
  | n + 2 => 2 * a (n + 1) + 3 * a n + 3^(n + 2)

theorem a_explicit_form (n : ℕ) :
  a n = (1 / 16) * ((4 * n - 3) * 3^(n + 1) - 7 * (-1)^n) :=
by sorry

end a_explicit_form_l671_67163


namespace round_trip_car_time_is_eight_l671_67155

/-- Represents the time in minutes for various trip configurations -/
structure TripTime where
  carAndWalk : ℕ  -- Time for car there and walk back
  walkBoth : ℕ    -- Time for walking both ways

/-- Calculates the time for a round trip by car given the TripTime -/
def roundTripCarTime (t : TripTime) : ℕ :=
  2 * (t.carAndWalk - t.walkBoth / 2)

/-- Theorem: Given the specific trip times, the round trip car time is 8 minutes -/
theorem round_trip_car_time_is_eight (t : TripTime) 
  (h1 : t.carAndWalk = 20) 
  (h2 : t.walkBoth = 32) : 
  roundTripCarTime t = 8 := by
  sorry

#eval roundTripCarTime { carAndWalk := 20, walkBoth := 32 }

end round_trip_car_time_is_eight_l671_67155


namespace cube_holes_surface_area_222_l671_67196

/-- Calculates the surface area of a cube with square holes cut through each face. -/
def cube_with_holes_surface_area (cube_edge : ℝ) (hole_edge : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge^2
  let hole_area := 6 * hole_edge^2
  let new_exposed_area := 6 * 4 * hole_edge^2
  original_surface_area - hole_area + new_exposed_area

/-- Theorem stating that a cube with edge length 5 and square holes of side length 2
    has a total surface area of 222 square meters. -/
theorem cube_holes_surface_area_222 :
  cube_with_holes_surface_area 5 2 = 222 := by
  sorry

end cube_holes_surface_area_222_l671_67196


namespace erased_number_proof_l671_67186

/-- The number of integers in the original sequence -/
def n : ℕ := 71

/-- The average of the remaining numbers after one is erased -/
def average : ℚ := 37 + 11/19

/-- The erased number -/
def x : ℕ := 2704

theorem erased_number_proof :
  (n * (n + 1) / 2 - x) / (n - 1) = average → x = 2704 := by
  sorry

end erased_number_proof_l671_67186


namespace f_is_integer_l671_67109

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def f (m n : ℕ) : ℚ :=
  (factorial (2 * m) * factorial (2 * n)) / (factorial m * factorial n * factorial (m + n))

theorem f_is_integer (m n : ℕ) : ∃ k : ℤ, f m n = k :=
sorry

end f_is_integer_l671_67109


namespace percent_of_decimal_zero_point_zero_one_is_ten_percent_of_zero_point_one_l671_67195

theorem percent_of_decimal (x y : ℝ) (h : y ≠ 0) :
  x / y * 100 = (x / y * 100 : ℝ) :=
by sorry

theorem zero_point_zero_one_is_ten_percent_of_zero_point_one :
  (0.01 : ℝ) / 0.1 * 100 = 10 :=
by sorry

end percent_of_decimal_zero_point_zero_one_is_ten_percent_of_zero_point_one_l671_67195


namespace quadratic_one_solution_l671_67122

theorem quadratic_one_solution (m : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + m = 0) → m = 49 / 12 := by
  sorry

end quadratic_one_solution_l671_67122


namespace equation_solution_l671_67120

theorem equation_solution (x : ℝ) : 
  (x^3 + 2*x + 1 > 0) → 
  ((16 * 5^(2*x - 1) - 2 * 5^(x - 1) - 0.048) * Real.log (x^3 + 2*x + 1) = 0) ↔ 
  (x = 0) := by
sorry

end equation_solution_l671_67120


namespace candy_bar_cost_l671_67141

/-- The cost of each candy bar given Benny's purchase -/
theorem candy_bar_cost (soft_drink_cost : ℝ) (num_candy_bars : ℕ) (total_spent : ℝ)
  (h1 : soft_drink_cost = 2)
  (h2 : num_candy_bars = 5)
  (h3 : total_spent = 27)
  : (total_spent - soft_drink_cost) / num_candy_bars = 5 := by
  sorry

end candy_bar_cost_l671_67141


namespace range_of_m_l671_67139

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, x - 2*y + 3 ≠ 0 ∨ y^2 ≠ m*x

def q (m : ℝ) : Prop := ∀ x y : ℝ, (x^2)/(5-2*m) + (y^2)/m = 1 → 
  (5-2*m < 0 ∧ m > 0) ∨ (5-2*m > 0 ∧ m < 0)

-- Define the theorem
theorem range_of_m : 
  ∀ m : ℝ, m ≠ 0 → (p m ∨ q m) → ¬(p m ∧ q m) → 
    m ≥ 3 ∨ m < 0 ∨ (0 < m ∧ m ≤ 5/2) := by sorry

end range_of_m_l671_67139


namespace rectangular_solid_length_l671_67129

/-- The surface area of a rectangular solid given its length, width, and depth. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: For a rectangular solid with width 8 meters, depth 5 meters, and 
    total surface area 314 square meters, the length is 9 meters. -/
theorem rectangular_solid_length :
  ∃ l : ℝ, surface_area l 8 5 = 314 ∧ l = 9 :=
by sorry

end rectangular_solid_length_l671_67129


namespace function_transformation_l671_67174

/-- Given a function f(x) = 3sin(2x + φ) where φ ∈ (0, π/2), if the graph of f(x) is translated
    left by π/6 units and is symmetric about the y-axis, then f(x) = 3sin(2x + π/6). -/
theorem function_transformation (φ : Real) (h1 : φ > 0) (h2 : φ < π/2) :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (2 * x + φ)
  let g : ℝ → ℝ := λ x ↦ f (x + π/6)
  (∀ x, g x = g (-x)) →  -- Symmetry about y-axis
  (∀ x, f x = 3 * Real.sin (2 * x + π/6)) := by
  sorry

end function_transformation_l671_67174


namespace salary_problem_l671_67175

theorem salary_problem (A B : ℝ) 
  (h1 : A + B = 4000)
  (h2 : A * 0.05 = B * 0.15) :
  A = 3000 := by
  sorry

end salary_problem_l671_67175


namespace sum_of_powers_equals_reciprocal_l671_67158

theorem sum_of_powers_equals_reciprocal (m : ℕ) (h_m_odd : Odd m) (h_m_gt_1 : m > 1) :
  let n := 2 * m
  let θ := Complex.exp (2 * Real.pi * Complex.I / n)
  (Finset.sum (Finset.range ((m - 1) / 2)) (fun i => θ^(2 * i + 1))) = 1 / (1 - θ) := by
sorry

end sum_of_powers_equals_reciprocal_l671_67158


namespace simplify_product_of_square_roots_l671_67116

theorem simplify_product_of_square_roots (x : ℝ) (hx : x > 0) :
  Real.sqrt (56 * x^3) * Real.sqrt (10 * x^2) * Real.sqrt (63 * x^4) = 84 * x^4 * Real.sqrt (5 * x) :=
by sorry

end simplify_product_of_square_roots_l671_67116


namespace box_dimensions_theorem_l671_67172

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given dimensions satisfy the consecutive integer condition -/
def isConsecutive (d : BoxDimensions) : Prop :=
  d.b = d.a + 1 ∧ d.c = d.a + 2

/-- Calculates the volume of the box -/
def volume (d : BoxDimensions) : ℕ :=
  d.a * d.b * d.c

/-- Calculates the surface area of the box -/
def surfaceArea (d : BoxDimensions) : ℕ :=
  2 * (d.a * d.b + d.b * d.c + d.c * d.a)

/-- The main theorem stating the conditions and the result to be proved -/
theorem box_dimensions_theorem (d : BoxDimensions) :
    d.a < d.b ∧ d.b < d.c ∧
    isConsecutive d ∧
    2 * surfaceArea d = volume d →
    d.a = 8 := by
  sorry

end box_dimensions_theorem_l671_67172


namespace negation_of_universal_proposition_l671_67177

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^2 - x < 0) ↔ 
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - x ≥ 0) := by
sorry

end negation_of_universal_proposition_l671_67177


namespace negation_of_forall_greater_than_one_negation_of_proposition_l671_67151

theorem negation_of_forall_greater_than_one (P : ℝ → Prop) :
  (¬ ∀ x > 1, P x) ↔ (∃ x > 1, ¬ P x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∀ x > 1, x^2 - x > 0) ↔ (∃ x > 1, x^2 - x ≤ 0) :=
by sorry

end negation_of_forall_greater_than_one_negation_of_proposition_l671_67151


namespace base_six_addition_problem_l671_67145

/-- Given a base-6 addition problem 5CD₆ + 52₆ = 64C₆, prove that C + D = 8 in base 10 -/
theorem base_six_addition_problem (C D : ℕ) : 
  (5 * 6^2 + C * 6 + D) + (5 * 6 + 2) = 6 * 6^2 + 4 * 6 + C →
  C + D = 8 := by
  sorry

end base_six_addition_problem_l671_67145


namespace dot_product_PA_PB_is_negative_one_l671_67107

/-- The dot product of vectors PA and PB is always -1, where P is any point on the curve y = x + 2/x (x > 0),
    A is the foot of the perpendicular from P to y = x, and B is the foot of the perpendicular from P to the y-axis. -/
theorem dot_product_PA_PB_is_negative_one :
  ∀ (x₀ : ℝ), x₀ > 0 →
  let P : ℝ × ℝ := (x₀, x₀ + 2 / x₀)
  let A : ℝ × ℝ := (x₀ + 1 / x₀, x₀ + 1 / x₀)
  let B : ℝ × ℝ := (0, x₀ + 2 / x₀)
  let PA : ℝ × ℝ := (A.1 - P.1, A.2 - P.2)
  let PB : ℝ × ℝ := (B.1 - P.1, B.2 - P.2)
  (PA.1 * PB.1 + PA.2 * PB.2 : ℝ) = -1 :=
by sorry

end dot_product_PA_PB_is_negative_one_l671_67107


namespace probability_green_or_blue_l671_67136

/-- The probability of drawing a green or blue marble from a bag -/
theorem probability_green_or_blue (green blue yellow : ℕ) 
  (hg : green = 4) (hb : blue = 3) (hy : yellow = 8) : 
  (green + blue : ℚ) / (green + blue + yellow) = 7 / 15 := by
  sorry

end probability_green_or_blue_l671_67136


namespace total_books_on_shelves_l671_67154

theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℕ) 
  (h1 : num_shelves = 150) (h2 : books_per_shelf = 15) : 
  num_shelves * books_per_shelf = 2250 := by
  sorry

end total_books_on_shelves_l671_67154


namespace squareable_numbers_l671_67164

-- Define what it means for a number to be squareable
def is_squareable (n : ℕ) : Prop :=
  ∃ (perm : Fin n → Fin n), Function.Bijective perm ∧
    ∀ i : Fin n, ∃ k : ℕ, (perm i).val + i.val + 1 = k^2

-- Theorem statement
theorem squareable_numbers :
  is_squareable 9 ∧ is_squareable 15 ∧ ¬is_squareable 7 ∧ ¬is_squareable 11 :=
sorry

end squareable_numbers_l671_67164


namespace perimeter_3x3_grid_l671_67114

/-- The perimeter of a square grid of unit squares -/
def grid_perimeter (rows columns : ℕ) : ℕ :=
  2 * (rows + columns)

/-- Theorem: The perimeter of a 3x3 grid of unit squares is 18 -/
theorem perimeter_3x3_grid : grid_perimeter 3 3 = 18 := by
  sorry

end perimeter_3x3_grid_l671_67114


namespace other_number_when_five_l671_67171

/-- Represents the invariant relation between Peter's numbers -/
def peterInvariant (a b : ℚ) : Prop :=
  2 * a * b - 5 * a - 5 * b = -11

/-- Peter's initial numbers satisfy the invariant -/
axiom initial_invariant : peterInvariant 1 2

/-- The invariant is preserved after each update -/
axiom invariant_preserved (a b c d : ℚ) :
  peterInvariant a b → peterInvariant c d → 
  ∀ p q : ℚ, (∃ m : ℚ, m * (p - a) * (p - b) = (p - c) * (p - d)) →
  peterInvariant p q

/-- When one of Peter's numbers is 5, the other satisfies the invariant -/
theorem other_number_when_five :
  ∃ b : ℚ, peterInvariant 5 b ∧ b = 14/5 := by sorry

end other_number_when_five_l671_67171


namespace circle_tangent_to_line_l671_67197

/-- A circle with center (0, 0) and radius r > 0 is tangent to the line 3x - 4y + 20 = 0 if and only if r = 4 -/
theorem circle_tangent_to_line (r : ℝ) (hr : r > 0) :
  (∀ x y : ℝ, x^2 + y^2 = r^2 ↔ (3*x - 4*y + 20 = 0 → x^2 + y^2 ≥ r^2) ∧ 
  (∃ x y : ℝ, 3*x - 4*y + 20 = 0 ∧ x^2 + y^2 = r^2)) ↔ r = 4 := by
  sorry

end circle_tangent_to_line_l671_67197


namespace problem_solution_l671_67102

theorem problem_solution (a b c d e : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |e| = 4) : 
  e^2 - (a + b)^2022 + (-c * d)^2021 = 15 := by sorry

end problem_solution_l671_67102


namespace junk_mail_calculation_l671_67176

theorem junk_mail_calculation (blocks : ℕ) (houses_per_block : ℕ) (mail_per_house : ℕ)
  (h1 : blocks = 16)
  (h2 : houses_per_block = 17)
  (h3 : mail_per_house = 4) :
  blocks * houses_per_block * mail_per_house = 1088 := by
  sorry

end junk_mail_calculation_l671_67176


namespace exam_marks_calculation_l671_67161

def math_passing_percentage : ℝ := 0.30
def science_passing_percentage : ℝ := 0.50
def english_passing_percentage : ℝ := 0.40

def math_marks_obtained : ℕ := 80
def math_marks_short : ℕ := 100

def science_marks_obtained : ℕ := 120
def science_marks_short : ℕ := 80

def english_marks_obtained : ℕ := 60
def english_marks_short : ℕ := 60

def math_max_marks : ℕ := 600
def science_max_marks : ℕ := 400
def english_max_marks : ℕ := 300

theorem exam_marks_calculation :
  (math_passing_percentage * math_max_marks : ℝ) = (math_marks_obtained + math_marks_short : ℝ) ∧
  (science_passing_percentage * science_max_marks : ℝ) = (science_marks_obtained + science_marks_short : ℝ) ∧
  (english_passing_percentage * english_max_marks : ℝ) = (english_marks_obtained + english_marks_short : ℝ) ∧
  math_max_marks + science_max_marks + english_max_marks = 1300 :=
by sorry

end exam_marks_calculation_l671_67161


namespace polynomial_simplification_l671_67160

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^2 + 15) - (x^6 + 4 * x^5 - 2 * x^3 + 20) = x^6 - x^5 + 2 * x^3 - 5 := by
  sorry

end polynomial_simplification_l671_67160


namespace geometric_progression_sum_ratio_l671_67127

theorem geometric_progression_sum_ratio (m : ℕ) : 
  let r : ℝ := 3
  let S (n : ℕ) := (1 - r^n) / (1 - r)
  (S 6) / (S m) = 28 → m = 3 := by
  sorry

end geometric_progression_sum_ratio_l671_67127


namespace snack_eaters_final_count_l671_67131

/-- Calculates the final number of snack eaters after a series of events -/
def final_snack_eaters (initial_gathering : ℕ) (initial_snackers : ℕ) 
  (first_outsiders : ℕ) (second_outsiders : ℕ) (third_leavers : ℕ) : ℕ :=
  let total_after_first_join := initial_snackers + first_outsiders
  let after_half_left := total_after_first_join / 2
  let after_second_join := after_half_left + second_outsiders
  let after_more_left := after_second_join - third_leavers
  after_more_left / 2

/-- Theorem stating that given the initial conditions, the final number of snack eaters is 20 -/
theorem snack_eaters_final_count :
  final_snack_eaters 200 100 20 10 30 = 20 := by
  sorry

end snack_eaters_final_count_l671_67131


namespace mary_books_checked_out_l671_67159

/-- Calculates the number of books Mary has checked out after a series of transactions --/
def books_checked_out (initial : ℕ) 
  (return1 checkout1 : ℕ) 
  (return2 checkout2 : ℕ) 
  (return3 checkout3 : ℕ) 
  (return4 checkout4 : ℕ) : ℕ :=
  initial - return1 + checkout1 - return2 + checkout2 - return3 + checkout3 - return4 + checkout4

/-- Proves that Mary has 22 books checked out given the problem conditions --/
theorem mary_books_checked_out : 
  books_checked_out 10 5 6 3 4 2 9 5 8 = 22 := by
  sorry

end mary_books_checked_out_l671_67159


namespace set_operations_l671_67185

def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_operations :
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((A ∩ B) ∩ C = ∅) ∧
  ((U \ A) ∩ (U \ B) = {0, 3}) := by
  sorry

end set_operations_l671_67185


namespace freight_yard_washing_machines_l671_67157

/-- Proves that the total number of washing machines removed is 30,000 given the conditions of the freight yard problem. -/
theorem freight_yard_washing_machines 
  (num_containers : ℕ) 
  (crates_per_container : ℕ) 
  (boxes_per_crate : ℕ) 
  (machines_per_box : ℕ) 
  (machines_removed_per_box : ℕ) 
  (h1 : num_containers = 50)
  (h2 : crates_per_container = 20)
  (h3 : boxes_per_crate = 10)
  (h4 : machines_per_box = 8)
  (h5 : machines_removed_per_box = 3) : 
  num_containers * crates_per_container * boxes_per_crate * machines_removed_per_box = 30000 := by
  sorry

#check freight_yard_washing_machines

end freight_yard_washing_machines_l671_67157


namespace unique_solution_l671_67118

/-- Represents a 3x3 grid with some fixed numbers and variables A, B, C, D --/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if two cells are adjacent in the grid --/
def adjacent (i j k l : Fin 3) : Prop :=
  (i = k ∧ (j.val + 1 = l.val ∨ l.val + 1 = j.val)) ∨
  (j = l ∧ (i.val + 1 = k.val ∨ k.val + 1 = i.val))

/-- The sum of any two adjacent numbers is less than 12 --/
def valid_sum (g : Grid) : Prop :=
  ∀ i j k l, adjacent i j k l → g i j + g k l < 12

/-- The grid contains all numbers from 1 to 9 --/
def contains_all_numbers (g : Grid) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → ∃ i j, g i j = n

/-- The given arrangement of known numbers in the grid --/
def given_arrangement (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 0 2 = 9 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7

/-- The theorem stating the unique solution for A, B, C, D --/
theorem unique_solution (g : Grid) 
  (h1 : valid_sum g) 
  (h2 : contains_all_numbers g) 
  (h3 : given_arrangement g) :
  g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by sorry

end unique_solution_l671_67118


namespace hundred_to_fifty_zeros_l671_67111

theorem hundred_to_fifty_zeros (n : ℕ) : 100^50 = 10^100 := by
  sorry

end hundred_to_fifty_zeros_l671_67111


namespace isosceles_right_triangle_angle_l671_67179

/-- An isosceles right triangle has two equal angles and one right angle (90°) -/
structure IsoscelesRightTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  is_isosceles : angle1 = angle2
  is_right : angle3 = 90
  sum_of_angles : angle1 + angle2 + angle3 = 180

/-- In an isosceles right triangle, if one of the angles is x°, then x = 45° -/
theorem isosceles_right_triangle_angle (t : IsoscelesRightTriangle) (x : ℝ) 
  (h : x = t.angle1 ∨ x = t.angle2 ∨ x = t.angle3) : x = 45 := by
  sorry

end isosceles_right_triangle_angle_l671_67179


namespace T_divisibility_l671_67103

def T : Set ℤ := {t | ∃ n : ℤ, t = n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2}

theorem T_divisibility :
  (∀ t ∈ T, ¬(9 ∣ t)) ∧ (∃ t ∈ T, 5 ∣ t) := by
  sorry

end T_divisibility_l671_67103


namespace pear_sales_ratio_l671_67104

/-- Given the total amount of pears sold in a day and the amount sold in the afternoon,
    prove that the ratio of pears sold in the afternoon to pears sold in the morning is 2:1. -/
theorem pear_sales_ratio (total : ℕ) (afternoon : ℕ) 
    (h1 : total = 510)
    (h2 : afternoon = 340) : 
  (afternoon : ℚ) / ((total - afternoon) : ℚ) = 2 := by
  sorry

end pear_sales_ratio_l671_67104


namespace arithmetic_progression_max_first_term_l671_67113

theorem arithmetic_progression_max_first_term 
  (b₁ : ℚ) 
  (d : ℚ) 
  (S₄ S₉ : ℕ) :
  (4 * b₁ + 6 * d = S₄) →
  (9 * b₁ + 36 * d = S₉) →
  (b₁ ≤ 3/4) →
  (∀ b₁' d' S₄' S₉' : ℚ, 
    (4 * b₁' + 6 * d' = S₄') →
    (9 * b₁' + 36 * d' = S₉') →
    (b₁' ≤ 3/4) →
    (b₁' ≤ b₁)) →
  b₁ = 11/15 :=
sorry

end arithmetic_progression_max_first_term_l671_67113


namespace circumscribed_circle_radius_l671_67173

/-- The radius of the circumscribed circle of a triangle with side lengths 3, 5, and 7 is 7√3/3 -/
theorem circumscribed_circle_radius (a b c : ℝ) (h_a : a = 3) (h_b : b = 5) (h_c : c = 7) :
  let R := c / (2 * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2 * a * b))^2))
  R = 7 * Real.sqrt 3 / 3 := by
  sorry

end circumscribed_circle_radius_l671_67173


namespace number_multiples_l671_67152

def is_valid_number (n : ℕ) : Prop :=
  ∃ (A B C D E F : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    n = A * 100000 + B * 10000 + C * 1000 + D * 100 + E * 10 + F

def satisfies_conditions (n : ℕ) : Prop :=
  is_valid_number n ∧
  ∃ (A B C D E F : ℕ),
    4 * n = A * 100000 + B * 10000 + C * 1000 + D * 100 + E * 10 + F ∧
    13 * n = F * 100000 + A * 10000 + B * 1000 + C * 100 + D * 10 + E ∧
    22 * n = C * 100000 + D * 10000 + E * 1000 + F * 100 + A * 10 + B

theorem number_multiples (n : ℕ) (h : satisfies_conditions n) :
  (∃ k : ℕ, n * k = 984126) ∧
  (∀ k : ℕ, n * k ≠ 269841) ∧
  (∀ k : ℕ, n * k ≠ 841269) :=
sorry

end number_multiples_l671_67152


namespace equal_tuesdays_fridays_count_l671_67178

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- The number of days in the month -/
def monthLength : Nat := 30

/-- Counts the number of occurrences of a specific weekday in a month -/
def countWeekday (startDay : Weekday) (targetDay : Weekday) : Nat :=
  sorry

/-- Checks if the number of Tuesdays equals the number of Fridays for a given start day -/
def hasSameTuesdaysAndFridays (startDay : Weekday) : Bool :=
  countWeekday startDay Weekday.Tuesday = countWeekday startDay Weekday.Friday

/-- The set of all possible start days that result in equal Tuesdays and Fridays -/
def validStartDays : Finset Weekday :=
  sorry

theorem equal_tuesdays_fridays_count :
  Finset.card validStartDays = 4 := by sorry

end equal_tuesdays_fridays_count_l671_67178


namespace max_areas_formula_l671_67181

/-- Represents a circular disk with n equally spaced radii and one off-center chord -/
structure DividedDisk where
  n : ℕ  -- number of equally spaced radii
  has_off_center_chord : Bool

/-- Calculates the maximum number of non-overlapping areas in a divided disk -/
def max_areas (d : DividedDisk) : ℕ :=
  2 * d.n + 2

/-- Theorem: The maximum number of non-overlapping areas in a divided disk is 2n + 2 -/
theorem max_areas_formula (d : DividedDisk) (h : d.has_off_center_chord = true) :
  max_areas d = 2 * d.n + 2 := by
  sorry

#check max_areas_formula

end max_areas_formula_l671_67181


namespace f_neg_two_eq_zero_l671_67190

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2007 + b * x + 1

theorem f_neg_two_eq_zero (a b : ℝ) : f a b 2 = 2 → f a b (-2) = 0 := by
  sorry

end f_neg_two_eq_zero_l671_67190
