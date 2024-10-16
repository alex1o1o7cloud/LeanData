import Mathlib

namespace NUMINAMATH_CALUDE_blocks_per_box_l1211_121179

def total_blocks : ℕ := 12
def num_boxes : ℕ := 2

theorem blocks_per_box : total_blocks / num_boxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_blocks_per_box_l1211_121179


namespace NUMINAMATH_CALUDE_unique_preimage_of_triple_l1211_121174

-- Define v₂ function
def v₂ (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n.log 2)

-- Define the properties of function f
def has_properties (f : ℕ → ℕ) : Prop :=
  (∀ x : ℕ, f x ≤ 3 * x) ∧ 
  (∀ x y : ℕ, v₂ (f x + f y) = v₂ (x + y))

-- State the theorem
theorem unique_preimage_of_triple (f : ℕ → ℕ) (h : has_properties f) :
  ∀ a : ℕ, ∃! x : ℕ, f x = 3 * a :=
sorry

end NUMINAMATH_CALUDE_unique_preimage_of_triple_l1211_121174


namespace NUMINAMATH_CALUDE_ferry_tourists_l1211_121177

/-- Calculates the total number of tourists transported by a ferry --/
def totalTourists (trips : ℕ) (initialTourists : ℕ) (decrease : ℕ) : ℕ :=
  trips * (2 * initialTourists - (trips - 1) * decrease) / 2

/-- Theorem: The ferry transports 904 tourists in total --/
theorem ferry_tourists : totalTourists 8 120 2 = 904 := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourists_l1211_121177


namespace NUMINAMATH_CALUDE_total_rainfall_2004_l1211_121123

/-- The average monthly rainfall in Mathborough in 2003 -/
def mathborough_2003 : ℝ := 41.5

/-- The increase in average monthly rainfall in Mathborough from 2003 to 2004 -/
def mathborough_increase : ℝ := 5

/-- The average monthly rainfall in Hightown in 2003 -/
def hightown_2003 : ℝ := 38

/-- The increase in average monthly rainfall in Hightown from 2003 to 2004 -/
def hightown_increase : ℝ := 3

/-- The number of months in a year -/
def months_in_year : ℕ := 12

theorem total_rainfall_2004 : 
  (mathborough_2003 + mathborough_increase) * months_in_year = 558 ∧
  (hightown_2003 + hightown_increase) * months_in_year = 492 := by
sorry

end NUMINAMATH_CALUDE_total_rainfall_2004_l1211_121123


namespace NUMINAMATH_CALUDE_two_dogs_food_consumption_l1211_121103

/-- The amount of dog food consumed by two dogs in a day -/
def total_dog_food_consumption (dog1_consumption dog2_consumption : Real) : Real :=
  dog1_consumption + dog2_consumption

/-- Theorem: Two dogs each consuming 0.125 scoop of dog food per day eat 0.25 scoop in total -/
theorem two_dogs_food_consumption :
  total_dog_food_consumption 0.125 0.125 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_two_dogs_food_consumption_l1211_121103


namespace NUMINAMATH_CALUDE_average_of_special_squares_l1211_121191

/-- Represents a 4x4 grid filled with numbers 1, 3, 5, and 7 -/
def Grid := Fin 4 → Fin 4 → Fin 4

/-- Checks if a row contains different numbers -/
def row_valid (g : Grid) (i : Fin 4) : Prop :=
  ∀ j k : Fin 4, j ≠ k → g i j ≠ g i k

/-- Checks if a column contains different numbers -/
def col_valid (g : Grid) (j : Fin 4) : Prop :=
  ∀ i k : Fin 4, i ≠ k → g i j ≠ g k j

/-- Checks if a 2x2 board contains different numbers -/
def board_valid (g : Grid) (i j : Fin 2) : Prop :=
  ∀ x y z w : Fin 2, (x, y) ≠ (z, w) → g (i + x) (j + y) ≠ g (i + z) (j + w)

/-- Checks if the entire grid is valid -/
def grid_valid (g : Grid) : Prop :=
  (∀ i : Fin 4, row_valid g i) ∧
  (∀ j : Fin 4, col_valid g j) ∧
  (∀ i j : Fin 2, board_valid g i j)

/-- The set of valid numbers in the grid -/
def valid_numbers : Finset (Fin 4) :=
  {0, 1, 2, 3}

/-- Maps Fin 4 to the actual numbers used in the grid -/
def to_actual_number (n : Fin 4) : ℕ :=
  2 * n + 1

/-- Theorem: The average of numbers in squares A, B, C, D is 4 -/
theorem average_of_special_squares (g : Grid) (hg : grid_valid g) :
  (to_actual_number (g 0 0) + to_actual_number (g 0 3) +
   to_actual_number (g 3 0) + to_actual_number (g 3 3)) / 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_special_squares_l1211_121191


namespace NUMINAMATH_CALUDE_largest_cube_surface_area_l1211_121124

/-- Given a cuboid with dimensions 12 cm, 16 cm, and 14 cm, 
    the surface area of the largest cube that can be cut from it is 864 cm^2 -/
theorem largest_cube_surface_area 
  (width : ℝ) (length : ℝ) (height : ℝ)
  (h_width : width = 12)
  (h_length : length = 16)
  (h_height : height = 14) :
  6 * (min width (min length height))^2 = 864 := by
  sorry

end NUMINAMATH_CALUDE_largest_cube_surface_area_l1211_121124


namespace NUMINAMATH_CALUDE_solution_pairs_l1211_121140

theorem solution_pairs : 
  ∀ (x y : ℕ), 2^(2*x+1) + 2^x + 1 = y^2 ↔ (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 23) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l1211_121140


namespace NUMINAMATH_CALUDE_cassidy_poster_count_l1211_121199

/-- Represents Cassidy's poster collection over time -/
structure PosterCollection where
  initial : Nat  -- Initial number of posters 3 years ago
  lost : Nat     -- Number of posters lost
  sold : Nat     -- Number of posters sold
  future : Nat   -- Number of posters to be added this summer

/-- Calculates the current number of posters in Cassidy's collection -/
def currentPosters (c : PosterCollection) : Nat :=
  2 * c.initial - 6

theorem cassidy_poster_count (c : PosterCollection) 
  (h1 : c.initial = 18)
  (h2 : c.lost = 2)
  (h3 : c.sold = 5)
  (h4 : c.future = 6) :
  currentPosters c = 30 := by
  sorry

#eval currentPosters { initial := 18, lost := 2, sold := 5, future := 6 }

end NUMINAMATH_CALUDE_cassidy_poster_count_l1211_121199


namespace NUMINAMATH_CALUDE_energy_usage_is_96_watts_l1211_121100

/-- Calculate total energy usage for three lights over a given time period -/
def totalEnergyUsage (baseWatts : ℕ) (hours : ℕ) : ℕ :=
  let lightA := baseWatts * hours
  let lightB := 3 * lightA
  let lightC := 4 * lightA
  lightA + lightB + lightC

/-- Theorem: The total energy usage for the given scenario is 96 watts -/
theorem energy_usage_is_96_watts :
  totalEnergyUsage 6 2 = 96 := by sorry

end NUMINAMATH_CALUDE_energy_usage_is_96_watts_l1211_121100


namespace NUMINAMATH_CALUDE_difference_10_6_l1211_121149

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  prop1 : a 5 * a 7 = 6
  prop2 : a 2 + a 10 = 5

/-- The difference between the 10th and 6th terms is either 2 or -2 -/
theorem difference_10_6 (seq : ArithmeticSequence) : 
  seq.a 10 - seq.a 6 = 2 ∨ seq.a 10 - seq.a 6 = -2 := by
  sorry

end NUMINAMATH_CALUDE_difference_10_6_l1211_121149


namespace NUMINAMATH_CALUDE_conic_is_circle_l1211_121170

-- Define the equation
def conic_equation (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 49

-- Theorem stating that the equation represents a circle
theorem conic_is_circle :
  ∃ (h k r : ℝ), r > 0 ∧ 
  (∀ (x y : ℝ), conic_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2) :=
sorry

end NUMINAMATH_CALUDE_conic_is_circle_l1211_121170


namespace NUMINAMATH_CALUDE_min_equation_solution_l1211_121157

theorem min_equation_solution (x : ℝ) : min (1/2 + x) (x^2) = 1 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_equation_solution_l1211_121157


namespace NUMINAMATH_CALUDE_equal_probability_implies_g_equals_5_l1211_121184

-- Define the number of marbles in each bag
def redMarbles1 : ℕ := 2
def blueMarbles1 : ℕ := 2
def redMarbles2 : ℕ := 2
def blueMarbles2 : ℕ := 2

-- Define the probability function for bag 1
def prob1 : ℚ := (redMarbles1 * (redMarbles1 - 1) + blueMarbles1 * (blueMarbles1 - 1)) / 
              ((redMarbles1 + blueMarbles1) * (redMarbles1 + blueMarbles1 - 1))

-- Define the probability function for bag 2
def prob2 (g : ℕ) : ℚ := (redMarbles2 * (redMarbles2 - 1) + blueMarbles2 * (blueMarbles2 - 1) + g * (g - 1)) / 
                       ((redMarbles2 + blueMarbles2 + g) * (redMarbles2 + blueMarbles2 + g - 1))

-- Theorem statement
theorem equal_probability_implies_g_equals_5 :
  ∃ (g : ℕ), g > 0 ∧ prob1 = prob2 g → g = 5 :=
sorry

end NUMINAMATH_CALUDE_equal_probability_implies_g_equals_5_l1211_121184


namespace NUMINAMATH_CALUDE_cone_volume_l1211_121132

/-- Given a cone with lateral surface area to base area ratio of 5:3 and height 4, 
    its volume is 12π. -/
theorem cone_volume (r : ℝ) (h : ℝ) (l : ℝ) : 
  h = 4 → l / r = 5 / 3 → l^2 = h^2 + r^2 → (1 / 3) * π * r^2 * h = 12 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l1211_121132


namespace NUMINAMATH_CALUDE_quadratic_solution_set_l1211_121142

/-- Given a quadratic function f(x) = x^2 + bx + c, 
    this theorem states that if the solution set of f(x) < 0 
    is the open interval (1, 3), then b + c = -1. -/
theorem quadratic_solution_set (b c : ℝ) : 
  ({x : ℝ | x^2 + b*x + c < 0} = {x : ℝ | 1 < x ∧ x < 3}) → 
  b + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_l1211_121142


namespace NUMINAMATH_CALUDE_value_of_a_l1211_121150

theorem value_of_a (a : ℝ) : (0.005 * a = 0.80) → (a = 160) := by sorry

end NUMINAMATH_CALUDE_value_of_a_l1211_121150


namespace NUMINAMATH_CALUDE_valid_palindrome_count_l1211_121156

def valid_digits : Finset Nat := {0, 7, 8, 9}

def is_palindrome (n : Nat) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def count_valid_palindromes : Nat :=
  (valid_digits.filter (· ≠ 0)).card *
  valid_digits.card ^ 2 *
  valid_digits.card ^ 2 *
  valid_digits.card

theorem valid_palindrome_count :
  count_valid_palindromes = 3072 := by sorry

end NUMINAMATH_CALUDE_valid_palindrome_count_l1211_121156


namespace NUMINAMATH_CALUDE_rotational_cipher_key_l1211_121139

/-- Represents the encoding function for a rotational cipher --/
def encode (key : ℕ) (letter : ℕ) : ℕ :=
  ((letter + key - 1) % 26) + 1

/-- Theorem: If the sum of encoded values for A, B, and C is 52, the key is 25 --/
theorem rotational_cipher_key (key : ℕ) 
  (h1 : 1 ≤ key ∧ key ≤ 26) 
  (h2 : encode key 1 + encode key 2 + encode key 3 = 52) : 
  key = 25 := by
  sorry

#check rotational_cipher_key

end NUMINAMATH_CALUDE_rotational_cipher_key_l1211_121139


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l1211_121168

theorem sum_of_three_squares (s t : ℝ) : 
  (3 * s + 2 * t = 27) → 
  (2 * s + 3 * t = 23) → 
  (s + 2 * t = 13) → 
  (3 * s = 21) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l1211_121168


namespace NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_is_11920_l1211_121112

/-- The sum of all integers between 100 and 500 which end in 3 -/
def sum_of_integers_ending_in_3 : ℕ :=
  let first_term := 103
  let last_term := 493
  let num_terms := (last_term - first_term) / 10 + 1
  num_terms * (first_term + last_term) / 2

theorem sum_of_integers_ending_in_3_is_11920 :
  sum_of_integers_ending_in_3 = 11920 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_is_11920_l1211_121112


namespace NUMINAMATH_CALUDE_fraction_comparisons_and_absolute_value_l1211_121111

theorem fraction_comparisons_and_absolute_value :
  (-3 : ℚ) / 7 < (-8 : ℚ) / 21 ∧
  (-5 : ℚ) / 6 > (-6 : ℚ) / 7 ∧
  |3.1 - Real.pi| = Real.pi - 3.1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparisons_and_absolute_value_l1211_121111


namespace NUMINAMATH_CALUDE_smallest_sum_of_four_consecutive_primes_divisible_by_four_l1211_121175

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_primes (p₁ p₂ p₃ p₄ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧
  ∀ q : ℕ, (is_prime q ∧ p₁ < q ∧ q < p₄) → (q = p₂ ∨ q = p₃)

theorem smallest_sum_of_four_consecutive_primes_divisible_by_four :
  ∃ p₁ p₂ p₃ p₄ : ℕ,
    consecutive_primes p₁ p₂ p₃ p₄ ∧
    (p₁ + p₂ + p₃ + p₄) % 4 = 0 ∧
    p₁ + p₂ + p₃ + p₄ = 36 ∧
    ∀ q₁ q₂ q₃ q₄ : ℕ,
      consecutive_primes q₁ q₂ q₃ q₄ →
      (q₁ + q₂ + q₃ + q₄) % 4 = 0 →
      q₁ + q₂ + q₃ + q₄ ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_four_consecutive_primes_divisible_by_four_l1211_121175


namespace NUMINAMATH_CALUDE_sum_of_integers_l1211_121198

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 128) : x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1211_121198


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l1211_121148

-- Define the displacement function
def s (t : ℝ) : ℝ := 4 - 2*t + t^2

-- State the theorem
theorem instantaneous_velocity_at_3_seconds :
  (deriv s) 3 = 4 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l1211_121148


namespace NUMINAMATH_CALUDE_bridge_length_l1211_121197

/-- The length of a bridge given train parameters -/
theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 120) 
  (h2 : train_speed_kmh = 45) 
  (h3 : crossing_time = 30) : 
  ∃ (bridge_length : ℝ), bridge_length = 255 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_l1211_121197


namespace NUMINAMATH_CALUDE_area_of_triangle_AEB_l1211_121131

-- Define the rectangle ABCD
structure Rectangle (A B C D : ℝ × ℝ) : Prop where
  is_rectangle : true -- We assume ABCD is a rectangle

-- Define points F and G on CD
def F : ℝ × ℝ := sorry
def G : ℝ × ℝ := sorry

-- Define point E as the intersection of AF and BG
def E : ℝ × ℝ := sorry

-- Define the lengths
def AB : ℝ := 7
def BC : ℝ := 4
def DF : ℝ := 2
def GC : ℝ := 3

-- Define the area of a triangle
def triangle_area (a b c : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_AEB 
  (A B C D : ℝ × ℝ) 
  (rect : Rectangle A B C D) : 
  triangle_area A E B = 28/5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_AEB_l1211_121131


namespace NUMINAMATH_CALUDE_interest_rate_proof_l1211_121196

/-- Compound interest calculation -/
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * ((1 + r) ^ t - 1)

/-- Approximate equality for real numbers -/
def approx_equal (x y : ℝ) (ε : ℝ) : Prop :=
  |x - y| < ε

theorem interest_rate_proof (P : ℝ) (CI : ℝ) (t : ℕ) (r : ℝ) 
  (h1 : P = 400)
  (h2 : CI = 100)
  (h3 : t = 2)
  (h4 : compound_interest P r t = CI) :
  approx_equal r 0.11803398875 0.00000001 :=
sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l1211_121196


namespace NUMINAMATH_CALUDE_carla_drink_problem_l1211_121135

/-- The amount of water Carla drank in ounces -/
def water_amount : ℝ := 15

/-- The total amount of liquid Carla drank in ounces -/
def total_amount : ℝ := 54

/-- The amount of soda Carla drank in ounces -/
def soda_amount (x : ℝ) : ℝ := 3 * water_amount - x

theorem carla_drink_problem :
  ∃ x : ℝ, x = 6 ∧ water_amount + soda_amount x = total_amount := by sorry

end NUMINAMATH_CALUDE_carla_drink_problem_l1211_121135


namespace NUMINAMATH_CALUDE_rohan_salary_l1211_121117

def monthly_salary (food_percent : ℚ) (rent_percent : ℚ) (entertainment_percent : ℚ) (conveyance_percent : ℚ) (savings : ℕ) : ℕ :=
  sorry

theorem rohan_salary :
  let food_percent : ℚ := 40 / 100
  let rent_percent : ℚ := 20 / 100
  let entertainment_percent : ℚ := 10 / 100
  let conveyance_percent : ℚ := 10 / 100
  let savings : ℕ := 1000
  monthly_salary food_percent rent_percent entertainment_percent conveyance_percent savings = 5000 := by
  sorry

end NUMINAMATH_CALUDE_rohan_salary_l1211_121117


namespace NUMINAMATH_CALUDE_pairing_count_l1211_121136

/-- The number of bowls -/
def num_bowls : ℕ := 4

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The total number of possible pairings -/
def total_pairings : ℕ := num_bowls * num_glasses

theorem pairing_count : total_pairings = 20 := by sorry

end NUMINAMATH_CALUDE_pairing_count_l1211_121136


namespace NUMINAMATH_CALUDE_arun_weight_upper_limit_l1211_121159

/-- The upper limit of Arun's weight according to his own opinion -/
def U : ℝ := sorry

/-- Arun's actual weight -/
def arun_weight : ℝ := sorry

/-- Arun's opinion: his weight is greater than 62 kg but less than U -/
axiom arun_opinion : 62 < arun_weight ∧ arun_weight < U

/-- Arun's brother's opinion: Arun's weight is greater than 60 kg but less than 70 kg -/
axiom brother_opinion : 60 < arun_weight ∧ arun_weight < 70

/-- Arun's mother's opinion: Arun's weight cannot be greater than 65 kg -/
axiom mother_opinion : arun_weight ≤ 65

/-- The average of different probable weights of Arun is 64 kg -/
axiom average_weight : (62 + U) / 2 = 64

theorem arun_weight_upper_limit : U = 65 := by sorry

end NUMINAMATH_CALUDE_arun_weight_upper_limit_l1211_121159


namespace NUMINAMATH_CALUDE_arccos_range_for_sin_l1211_121173

theorem arccos_range_for_sin (a : ℝ) (x : ℝ) (h1 : x = Real.sin a) (h2 : a ∈ Set.Icc (-π/4) (3*π/4)) :
  ∃ y ∈ Set.Icc 0 (3*π/4), y = Real.arccos x :=
sorry

end NUMINAMATH_CALUDE_arccos_range_for_sin_l1211_121173


namespace NUMINAMATH_CALUDE_parabola_equation_l1211_121162

/-- A parabola with focus on the x-axis passing through the point (1, 2) -/
structure Parabola where
  /-- The equation of the parabola in the form y^2 = 2px -/
  equation : ℝ → ℝ → Prop
  /-- The parabola passes through the point (1, 2) -/
  passes_through_point : equation 1 2
  /-- The focus of the parabola is on the x-axis -/
  focus_on_x_axis : ∃ p : ℝ, ∀ x y : ℝ, equation x y ↔ y^2 = 2*p*x

/-- The standard equation of the parabola is y^2 = 4x -/
theorem parabola_equation (p : Parabola) : 
  ∃ (f : ℝ → ℝ → Prop), (∀ x y : ℝ, f x y ↔ y^2 = 4*x) ∧ p.equation = f := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1211_121162


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1211_121176

-- Define the type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the type for lines in 2D space
structure Line2D where
  f : ℝ → ℝ → ℝ

-- Define the property of a point being on a line
def PointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.f p.x p.y = 0

-- Define the property of two lines being parallel
def ParallelLines (l1 l2 : Line2D) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), l1.f x y = k * l2.f x y

theorem parallel_line_through_point
  (l : Line2D) (p1 p2 : Point2D)
  (h1 : PointOnLine p1 l)
  (h2 : ¬PointOnLine p2 l) :
  let l2 : Line2D := { f := λ x y => l.f x y - l.f p2.x p2.y }
  ParallelLines l l2 ∧ PointOnLine p2 l2 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_line_through_point_l1211_121176


namespace NUMINAMATH_CALUDE_smallest_integer_gcd_18_is_6_l1211_121166

theorem smallest_integer_gcd_18_is_6 : 
  ∃ n : ℕ, n > 100 ∧ n.gcd 18 = 6 ∧ ∀ m : ℕ, m > 100 ∧ m.gcd 18 = 6 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_gcd_18_is_6_l1211_121166


namespace NUMINAMATH_CALUDE_sqrt_simplification_exists_l1211_121141

theorem sqrt_simplification_exists :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * Real.sqrt (b / a) = Real.sqrt (a * b) :=
sorry

end NUMINAMATH_CALUDE_sqrt_simplification_exists_l1211_121141


namespace NUMINAMATH_CALUDE_min_area_triangle_containing_unit_square_l1211_121190

/-- A triangle that contains a unit square. -/
structure TriangleContainingUnitSquare where
  /-- The area of the triangle. -/
  area : ℝ
  /-- The triangle contains a unit square. -/
  contains_unit_square : True

/-- The minimum area of a triangle containing a unit square is 2. -/
theorem min_area_triangle_containing_unit_square :
  ∀ t : TriangleContainingUnitSquare, t.area ≥ 2 ∧ ∃ t' : TriangleContainingUnitSquare, t'.area = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_area_triangle_containing_unit_square_l1211_121190


namespace NUMINAMATH_CALUDE_cosine_amplitude_l1211_121115

theorem cosine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.cos (b * x) ≤ 3) ∧
  (∃ x, a * Real.cos (b * x) = 3) ∧
  (∀ x, a * Real.cos (b * x) = a * Real.cos (b * (x + 2 * Real.pi))) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l1211_121115


namespace NUMINAMATH_CALUDE_movie_revenue_growth_equation_l1211_121127

theorem movie_revenue_growth_equation 
  (initial_revenue : ℝ) 
  (revenue_after_three_weeks : ℝ) 
  (x : ℝ) 
  (h1 : initial_revenue = 2.5)
  (h2 : revenue_after_three_weeks = 3.6)
  (h3 : ∀ t : ℕ, t < 3 → 
    initial_revenue * (1 + x)^t < initial_revenue * (1 + x)^(t+1)) :
  initial_revenue * (1 + x)^2 = revenue_after_three_weeks :=
sorry

end NUMINAMATH_CALUDE_movie_revenue_growth_equation_l1211_121127


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l1211_121119

-- Define the original number
def original_number : ℕ := 262883000000

-- Define the scientific notation components
def significand : ℚ := 2.62883
def exponent : ℕ := 11

-- Theorem statement
theorem scientific_notation_correct : 
  (significand * (10 : ℚ) ^ exponent) = original_number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l1211_121119


namespace NUMINAMATH_CALUDE_average_expenditure_feb_to_jul_l1211_121137

def average_expenditure_jan_to_jun : ℚ := 4200
def january_expenditure : ℚ := 1200
def july_expenditure : ℚ := 1500

theorem average_expenditure_feb_to_jul :
  let total_jan_to_jun := 6 * average_expenditure_jan_to_jun
  let total_feb_to_jun := total_jan_to_jun - january_expenditure
  let total_feb_to_jul := total_feb_to_jun + july_expenditure
  total_feb_to_jul / 6 = 4250 := by sorry

end NUMINAMATH_CALUDE_average_expenditure_feb_to_jul_l1211_121137


namespace NUMINAMATH_CALUDE_tan_product_equals_neg_one_fifth_l1211_121164

theorem tan_product_equals_neg_one_fifth 
  (α β : ℝ) (h : 2 * Real.cos (2 * α + β) - 3 * Real.cos β = 0) :
  Real.tan α * Real.tan (α + β) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_equals_neg_one_fifth_l1211_121164


namespace NUMINAMATH_CALUDE_square_fraction_count_l1211_121180

theorem square_fraction_count : 
  ∃! (s : Finset Int), 
    (∀ n ∈ s, ∃ k : Int, (n : ℚ) / (20 - n) = k^2) ∧ 
    (∀ n : Int, n ∉ s → ¬∃ k : Int, (n : ℚ) / (20 - n) = k^2) ∧
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_count_l1211_121180


namespace NUMINAMATH_CALUDE_odd_coefficients_in_binomial_expansion_l1211_121121

theorem odd_coefficients_in_binomial_expansion :
  let coefficients : List ℕ := List.map (fun k => Nat.choose 8 k) (List.range 9)
  ∃! n : ℕ, n = (coefficients.filter (fun a => a % 2 = 1)).length ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_coefficients_in_binomial_expansion_l1211_121121


namespace NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_minus_two_i_l1211_121154

theorem real_part_of_i_squared_times_one_minus_two_i : 
  Complex.re (Complex.I^2 * (1 - 2*Complex.I)) = -1 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_minus_two_i_l1211_121154


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1211_121122

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1211_121122


namespace NUMINAMATH_CALUDE_factor_between_l1211_121116

theorem factor_between (n a b : ℕ) (hn : n > 10) (ha : a > 0) (hb : b > 0) 
  (hab : a ≠ b) (hdiv_a : a ∣ n) (hdiv_b : b ∣ n) (heq : n = a^2 + b) : 
  ∃ k : ℕ, k ∣ n ∧ a < k ∧ k < b := by
  sorry

end NUMINAMATH_CALUDE_factor_between_l1211_121116


namespace NUMINAMATH_CALUDE_sum_of_squares_l1211_121147

/-- A structure representing a set of four-digit numbers formed from four distinct digits. -/
structure FourDigitSet where
  digits : Finset Nat
  first_number : Nat
  second_last_number : Nat
  (digit_count : digits.card = 4)
  (distinct_digits : ∀ d ∈ digits, d < 10)
  (number_count : (digits.powerset.filter (λ s : Finset Nat => s.card = 4)).card = 18)
  (ascending_order : first_number < second_last_number)
  (first_is_square : ∃ n : Nat, first_number = n ^ 2)
  (second_last_is_square : ∃ n : Nat, second_last_number = n ^ 2)

/-- The theorem stating that the sum of the first and second-last numbers is 10890. -/
theorem sum_of_squares (s : FourDigitSet) : s.first_number + s.second_last_number = 10890 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1211_121147


namespace NUMINAMATH_CALUDE_infinitely_many_n_for_f_congruence_l1211_121129

/-- The function f(p, n) represents the largest integer k such that p^k divides n! -/
def f (p n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem infinitely_many_n_for_f_congruence 
  (p : ℕ) 
  (m c : ℕ+) 
  (h_prime : Nat.Prime p) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, f p n ≡ c.val [MOD m.val] := by sorry

end NUMINAMATH_CALUDE_infinitely_many_n_for_f_congruence_l1211_121129


namespace NUMINAMATH_CALUDE_parabola_directrix_l1211_121171

/-- The directrix of the parabola y = 3x^2 + 6x + 5 is y = 23/12 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y = 3 * x^2 + 6 * x + 5 → 
  ∃ (k : ℝ), k = 23/12 ∧ (∀ (x₀ : ℝ), (x - x₀)^2 = 4 * (1/12) * (y - k)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1211_121171


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1211_121113

-- Define a monotonically increasing function on [0, +∞)
def monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Define the set of x that satisfies the inequality
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (2 * x - 1) < f (1 / 3)}

-- Theorem statement
theorem inequality_solution_set 
  (f : ℝ → ℝ) 
  (h_monotone : monotone_increasing_on_nonneg f) :
  solution_set f = Set.Ici (1 / 2) ∩ Set.Iio (2 / 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1211_121113


namespace NUMINAMATH_CALUDE_sector_radius_l1211_121104

/-- Given a sector with a central angle of 90° and an arc length of 3π, its radius is 6. -/
theorem sector_radius (θ : Real) (l : Real) (r : Real) : 
  θ = 90 → l = 3 * Real.pi → l = (θ * Real.pi * r) / 180 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l1211_121104


namespace NUMINAMATH_CALUDE_no_primes_in_factorial_range_l1211_121194

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 1) :
  ∀ k ∈ Set.Ioo (n! + 1) (n! + n), ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_factorial_range_l1211_121194


namespace NUMINAMATH_CALUDE_refrigerator_savings_l1211_121134

def cash_price : ℕ := 8000
def deposit : ℕ := 3000
def num_installments : ℕ := 30
def installment_amount : ℕ := 300

theorem refrigerator_savings : 
  deposit + num_installments * installment_amount - cash_price = 4000 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_savings_l1211_121134


namespace NUMINAMATH_CALUDE_three_person_subcommittees_l1211_121130

theorem three_person_subcommittees (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_l1211_121130


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l1211_121101

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (∀ y : ℝ, y > 0 → 13 = y^2 + 1/y^2 → x + 1/x ≥ y + 1/y) → x + 1/x = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l1211_121101


namespace NUMINAMATH_CALUDE_largest_quantity_l1211_121126

theorem largest_quantity (a b c d e : ℝ) 
  (eq1 : a = b + 3)
  (eq2 : b = c - 4)
  (eq3 : c = d + 5)
  (eq4 : d = e - 6) :
  e ≥ a ∧ e ≥ b ∧ e ≥ c ∧ e ≥ d := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l1211_121126


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l1211_121153

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of cubes of different sizes needed to fill a box -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  sorry

/-- The theorem stating the smallest number of cubes needed for the given box dimensions -/
theorem smallest_number_of_cubes_for_given_box :
  let box : BoxDimensions := { length := 98, width := 77, depth := 35 }
  smallestNumberOfCubes box = 770 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l1211_121153


namespace NUMINAMATH_CALUDE_committee_selections_theorem_l1211_121145

/-- The number of ways to select a committee with at least one former member -/
def committee_selections_with_former (total_candidates : ℕ) (former_members : ℕ) (committee_size : ℕ) : ℕ :=
  Nat.choose total_candidates committee_size - Nat.choose (total_candidates - former_members) committee_size

/-- Theorem stating the number of committee selections with at least one former member -/
theorem committee_selections_theorem :
  committee_selections_with_former 15 6 4 = 1239 := by
  sorry

end NUMINAMATH_CALUDE_committee_selections_theorem_l1211_121145


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l1211_121138

/-- A geometric sequence with real terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: In a geometric sequence with a₁ = 1 and a₃ = 2, a₅ = 4 -/
theorem geometric_sequence_a5 (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 1) 
  (h_a3 : a 3 = 2) : 
  a 5 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l1211_121138


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l1211_121125

/-- Represents a rectangular garden -/
structure RectangularGarden where
  width : ℝ
  length : ℝ

/-- The area of a rectangular garden -/
def area (g : RectangularGarden) : ℝ := g.width * g.length

/-- Theorem: The area of a rectangular garden with width 16 meters and length three times its width is 768 square meters -/
theorem rectangular_garden_area : 
  ∀ (g : RectangularGarden), 
  g.width = 16 → 
  g.length = 3 * g.width → 
  area g = 768 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l1211_121125


namespace NUMINAMATH_CALUDE_age_difference_l1211_121158

-- Define variables for ages
variable (a b c : ℕ)

-- Define the condition from the problem
def age_condition (a b c : ℕ) : Prop := a + b = b + c + 12

-- Theorem to prove
theorem age_difference (h : age_condition a b c) : a = c + 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1211_121158


namespace NUMINAMATH_CALUDE_gold_alloy_composition_l1211_121187

theorem gold_alloy_composition
  (initial_weight : ℝ)
  (initial_gold_percentage : ℝ)
  (target_gold_percentage : ℝ)
  (added_gold : ℝ)
  (h1 : initial_weight = 48)
  (h2 : initial_gold_percentage = 0.25)
  (h3 : target_gold_percentage = 0.40)
  (h4 : added_gold = 12) :
  let initial_gold := initial_weight * initial_gold_percentage
  let final_weight := initial_weight + added_gold
  let final_gold := initial_gold + added_gold
  (final_gold / final_weight) = target_gold_percentage :=
by
  sorry

#check gold_alloy_composition

end NUMINAMATH_CALUDE_gold_alloy_composition_l1211_121187


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l1211_121192

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (10 * bowling_ball_weight = 5 * canoe_weight) →
    (3 * canoe_weight = 120) →
    bowling_ball_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l1211_121192


namespace NUMINAMATH_CALUDE_x_minus_y_values_l1211_121160

theorem x_minus_y_values (x y : ℝ) (h1 : |x| = 4) (h2 : |y| = 5) (h3 : x > y) :
  x - y = 9 ∨ x - y = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l1211_121160


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l1211_121165

theorem greatest_integer_radius (r : ℕ) (A : ℝ) : 
  A < 75 * Real.pi → A = Real.pi * (r : ℝ)^2 → r ≤ 8 ∧ ∃ (s : ℕ), s = 8 ∧ Real.pi * (s : ℝ)^2 < 75 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l1211_121165


namespace NUMINAMATH_CALUDE_last_three_average_l1211_121118

theorem last_three_average (list : List ℝ) : 
  list.length = 6 →
  list.sum / list.length = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 3 = 65 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l1211_121118


namespace NUMINAMATH_CALUDE_centroid_quadrilateral_area_l1211_121102

/-- Given a square ABCD with side length 40 and a point Q inside the square
    such that AQ = 16 and BQ = 34, the area of the quadrilateral formed by
    the centroids of △ABQ, △BCQ, △CDQ, and △DAQ is 6400/9. -/
theorem centroid_quadrilateral_area (A B C D Q : ℝ × ℝ) : 
  let square_side : ℝ := 40
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  -- Square ABCD conditions
  (dist A B = square_side) ∧ 
  (dist B C = square_side) ∧ 
  (dist C D = square_side) ∧ 
  (dist D A = square_side) ∧ 
  -- Right angles
  ((B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) ∧
  -- Q inside square
  (0 < Q.1) ∧ (Q.1 < square_side) ∧ (0 < Q.2) ∧ (Q.2 < square_side) ∧
  -- AQ and BQ distances
  (dist A Q = 16) ∧ 
  (dist B Q = 34) →
  -- Area of quadrilateral formed by centroids
  let centroid (P1 P2 P3 : ℝ × ℝ) := 
    ((P1.1 + P2.1 + P3.1) / 3, (P1.2 + P2.2 + P3.2) / 3)
  let G1 := centroid A B Q
  let G2 := centroid B C Q
  let G3 := centroid C D Q
  let G4 := centroid D A Q
  let area := (dist G1 G3 * dist G2 G4) / 2
  area = 6400 / 9 := by
sorry

end NUMINAMATH_CALUDE_centroid_quadrilateral_area_l1211_121102


namespace NUMINAMATH_CALUDE_father_current_age_l1211_121188

/-- The father's age at the son's birth equals the son's current age -/
def father_age_at_son_birth (father_age_now son_age_now : ℕ) : Prop :=
  father_age_now - son_age_now = son_age_now

/-- The son's age 5 years ago was 26 -/
def son_age_five_years_ago (son_age_now : ℕ) : Prop :=
  son_age_now - 5 = 26

/-- Theorem stating that the father's current age is 62 years -/
theorem father_current_age :
  ∀ (father_age_now son_age_now : ℕ),
    father_age_at_son_birth father_age_now son_age_now →
    son_age_five_years_ago son_age_now →
    father_age_now = 62 :=
by
  sorry

end NUMINAMATH_CALUDE_father_current_age_l1211_121188


namespace NUMINAMATH_CALUDE_paper_pieces_l1211_121186

/-- The number of pieces of paper after n tears -/
def num_pieces (n : ℕ) : ℕ := 3 * n + 1

/-- Theorem stating the number of pieces after n tears -/
theorem paper_pieces (n : ℕ) : 
  (∀ k : ℕ, k ≤ n → num_pieces k = num_pieces (k - 1) + 3) → 
  num_pieces n = 3 * n + 1 :=
by sorry

end NUMINAMATH_CALUDE_paper_pieces_l1211_121186


namespace NUMINAMATH_CALUDE_sin_45_degrees_l1211_121120

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l1211_121120


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l1211_121143

def is_single_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

theorem cousins_ages_sum :
  ∀ (a b c d e : ℕ),
    is_single_digit a ∧ is_single_digit b ∧ is_single_digit c ∧ is_single_digit d ∧ is_single_digit e →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e →
    (a * b = 36 ∨ a * c = 36 ∨ a * d = 36 ∨ a * e = 36 ∨ b * c = 36 ∨ b * d = 36 ∨ b * e = 36 ∨ c * d = 36 ∨ c * e = 36 ∨ d * e = 36) →
    (a * b = 40 ∨ a * c = 40 ∨ a * d = 40 ∨ a * e = 40 ∨ b * c = 40 ∨ b * d = 40 ∨ b * e = 40 ∨ c * d = 40 ∨ c * e = 40 ∨ d * e = 40) →
    a + b + c + d + e = 33 :=
by
  sorry

#check cousins_ages_sum

end NUMINAMATH_CALUDE_cousins_ages_sum_l1211_121143


namespace NUMINAMATH_CALUDE_inequality_range_theorem_l1211_121107

/-- The range of values for the real number a that satisfies the inequality
    2*ln(x) ≥ -x^2 + ax - 3 for all x ∈ (0, +∞) is (-∞, 4]. -/
theorem inequality_range_theorem (a : ℝ) : 
  (∀ x > 0, 2 * Real.log x ≥ -x^2 + a*x - 3) ↔ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_theorem_l1211_121107


namespace NUMINAMATH_CALUDE_normal_price_of_pin_is_20_l1211_121181

/-- Calculates the normal price of a pin given the number of pins, discount rate, and total spent -/
def normalPriceOfPin (numPins : ℕ) (discountRate : ℚ) (totalSpent : ℚ) : ℚ :=
  totalSpent / (numPins * (1 - discountRate))

theorem normal_price_of_pin_is_20 :
  normalPriceOfPin 10 (15/100) 170 = 20 := by
  sorry

#eval normalPriceOfPin 10 (15/100) 170

end NUMINAMATH_CALUDE_normal_price_of_pin_is_20_l1211_121181


namespace NUMINAMATH_CALUDE_parallel_line_triangle_l1211_121161

theorem parallel_line_triangle (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y : ℝ), 
  let s := (a + b + c) / 2
  let perimeter_AXY := x + y + (a * (x + y)) / (b + c)
  let perimeter_XBCY := a + b + c - (x + y)
  (0 < x ∧ x < c) ∧ (0 < y ∧ y < b) ∧ 
  perimeter_AXY = perimeter_XBCY →
  (a * (x + y)) / (b + c) = s * (a / (b + c)) := by
sorry

end NUMINAMATH_CALUDE_parallel_line_triangle_l1211_121161


namespace NUMINAMATH_CALUDE_lincoln_county_houses_l1211_121182

theorem lincoln_county_houses : 
  let original_houses : ℕ := 128936
  let new_houses : ℕ := 359482
  original_houses + new_houses = 488418 :=
by sorry

end NUMINAMATH_CALUDE_lincoln_county_houses_l1211_121182


namespace NUMINAMATH_CALUDE_saplings_distribution_l1211_121169

theorem saplings_distribution (total : ℕ) (a b c d : ℕ) : 
  total = 2126 →
  a = 2 * b + 20 →
  a = 3 * c + 24 →
  a = 5 * d - 45 →
  a + b + c + d = total →
  a = 1050 := by
  sorry

end NUMINAMATH_CALUDE_saplings_distribution_l1211_121169


namespace NUMINAMATH_CALUDE_octal_addition_1275_164_l1211_121195

/-- Converts an octal (base 8) number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Represents an octal number -/
structure OctalNumber where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 8

/-- Addition of two octal numbers -/
def octal_add (a b : OctalNumber) : OctalNumber :=
  ⟨ -- implementation details omitted
    sorry,
    sorry ⟩

theorem octal_addition_1275_164 :
  let a : OctalNumber := ⟨[1, 2, 7, 5], sorry⟩
  let b : OctalNumber := ⟨[1, 6, 4], sorry⟩
  let result : OctalNumber := octal_add a b
  result.digits = [1, 5, 0, 3] := by
  sorry

end NUMINAMATH_CALUDE_octal_addition_1275_164_l1211_121195


namespace NUMINAMATH_CALUDE_northton_capsule_depth_l1211_121193

theorem northton_capsule_depth (southton_depth : ℝ) (northton_offset : ℝ) : 
  southton_depth = 15 →
  northton_offset = 12 →
  (4 * southton_depth + northton_offset) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_northton_capsule_depth_l1211_121193


namespace NUMINAMATH_CALUDE_edward_final_lives_l1211_121172

/-- Calculates the final number of lives Edward has after completing three stages of a game. -/
def final_lives (initial_lives : ℕ) 
                (stage1_loss stage1_gain : ℕ) 
                (stage2_loss stage2_gain : ℕ) 
                (stage3_loss stage3_gain : ℕ) : ℕ :=
  initial_lives - stage1_loss + stage1_gain - stage2_loss + stage2_gain - stage3_loss + stage3_gain

/-- Theorem stating that Edward's final number of lives is 23 given the specified conditions. -/
theorem edward_final_lives : 
  final_lives 50 18 7 10 5 13 2 = 23 := by
  sorry


end NUMINAMATH_CALUDE_edward_final_lives_l1211_121172


namespace NUMINAMATH_CALUDE_fred_paper_count_l1211_121185

theorem fred_paper_count (initial_sheets received_sheets given_sheets : ℕ) :
  initial_sheets = 212 →
  received_sheets = 307 →
  given_sheets = 156 →
  initial_sheets + received_sheets - given_sheets = 363 := by
  sorry

end NUMINAMATH_CALUDE_fred_paper_count_l1211_121185


namespace NUMINAMATH_CALUDE_new_students_admitted_l1211_121146

theorem new_students_admitted (initial_students_per_section : ℕ) 
  (new_sections : ℕ) (final_total_sections : ℕ) (final_students_per_section : ℕ) :
  initial_students_per_section = 24 →
  new_sections = 3 →
  final_total_sections = 16 →
  final_students_per_section = 21 →
  (final_total_sections * final_students_per_section) - 
  ((final_total_sections - new_sections) * initial_students_per_section) = 24 := by
  sorry

end NUMINAMATH_CALUDE_new_students_admitted_l1211_121146


namespace NUMINAMATH_CALUDE_log_abs_properties_l1211_121110

-- Define the function f(x) = log|x|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

-- State the theorem
theorem log_abs_properties :
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧  -- f is even
  (∀ x y : ℝ, x < y ∧ y < 0 → f y < f x)  -- f is monotonically decreasing on (-∞, 0)
  := by sorry

end NUMINAMATH_CALUDE_log_abs_properties_l1211_121110


namespace NUMINAMATH_CALUDE_exists_x_y_for_3k_l1211_121108

theorem exists_x_y_for_3k (k : ℕ+) : 
  ∃ (x y : ℤ), (¬ 3 ∣ x) ∧ (¬ 3 ∣ y) ∧ (x^2 + 2*y^2 = 3^(k.val)) := by
  sorry

end NUMINAMATH_CALUDE_exists_x_y_for_3k_l1211_121108


namespace NUMINAMATH_CALUDE_margaret_egg_collection_l1211_121189

/-- The number of groups Margaret's eggs can be organized into -/
def num_groups : ℕ := 5

/-- The number of eggs in each group -/
def eggs_per_group : ℕ := 7

/-- The total number of eggs in Margaret's collection -/
def total_eggs : ℕ := num_groups * eggs_per_group

theorem margaret_egg_collection : total_eggs = 35 := by
  sorry

end NUMINAMATH_CALUDE_margaret_egg_collection_l1211_121189


namespace NUMINAMATH_CALUDE_hat_saves_greater_percentage_l1211_121163

-- Define the given values
def shoes_spent : ℚ := 42.25
def shoes_saved : ℚ := 3.75
def hat_sale_price : ℚ := 18.20
def hat_discount : ℚ := 1.80

-- Define the calculated values
def shoes_original : ℚ := shoes_spent + shoes_saved
def hat_original : ℚ := hat_sale_price + hat_discount

-- Define the percentage saved function
def percentage_saved (saved amount : ℚ) : ℚ := (saved / amount) * 100

-- Theorem statement
theorem hat_saves_greater_percentage :
  percentage_saved hat_discount hat_original > percentage_saved shoes_saved shoes_original :=
sorry

end NUMINAMATH_CALUDE_hat_saves_greater_percentage_l1211_121163


namespace NUMINAMATH_CALUDE_train_length_calculation_l1211_121106

theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) : 
  train_speed = 90 ∧ bridge_length = 140 ∧ passing_time = 20 → 
  (train_speed * 1000 / 3600) * passing_time - bridge_length = 360 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l1211_121106


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1211_121144

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  1 / a + 3 / b ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1211_121144


namespace NUMINAMATH_CALUDE_cube_in_pyramid_l1211_121133

/-- The edge length of a cube inscribed in a regular quadrilateral pyramid -/
theorem cube_in_pyramid (a h : ℝ) (ha : a > 0) (hh : h > 0) :
  ∃ x : ℝ, x > 0 ∧ 
    (a * h) / (a + h * Real.sqrt 2) ≤ x ∧
    x ≤ (a * h) / (a + h) :=
by sorry

end NUMINAMATH_CALUDE_cube_in_pyramid_l1211_121133


namespace NUMINAMATH_CALUDE_multiple_of_x_l1211_121109

theorem multiple_of_x (x y k : ℤ) 
  (eq1 : k * x + y = 34)
  (eq2 : 2 * x - y = 20)
  (y_sq : y^2 = 4) :
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_x_l1211_121109


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1211_121167

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 > 0) → -2 < k ∧ k < 3 ∧ 
  ∃ k₀ : ℝ, -2 < k₀ ∧ k₀ < 3 ∧ ∃ x : ℝ, x^2 + k₀*x + 1 ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1211_121167


namespace NUMINAMATH_CALUDE_evaluate_expression_l1211_121152

theorem evaluate_expression (x : ℝ) (h : x = 2) : (3 * x^2 - 8 * x + 5) * (4 * x - 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1211_121152


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l1211_121178

theorem smallest_solution_quartic_equation :
  let f : ℝ → ℝ := λ x => x^4 - 40*x^2 + 144
  ∃ (x : ℝ), f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = -6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l1211_121178


namespace NUMINAMATH_CALUDE_tripod_new_height_l1211_121155

/-- Represents a tripod with given properties -/
structure Tripod where
  leg_length : ℝ
  initial_height : ℝ
  sink_depth : ℝ

/-- Calculates the new height of a tripod after one leg sinks -/
def new_height (t : Tripod) : ℝ :=
  sorry

/-- The specific tripod in the problem -/
def problem_tripod : Tripod :=
  { leg_length := 8,
    initial_height := 6,
    sink_depth := 2 }

/-- Theorem stating the new height of the tripod after one leg sinks -/
theorem tripod_new_height :
  new_height problem_tripod = 144 / Real.sqrt 262.2 := by
  sorry

end NUMINAMATH_CALUDE_tripod_new_height_l1211_121155


namespace NUMINAMATH_CALUDE_negation_of_implication_l1211_121151

theorem negation_of_implication (x : ℝ) :
  ¬(x < 0 → x < 1) ↔ (x ≥ 0 → x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1211_121151


namespace NUMINAMATH_CALUDE_unique_solution_characterization_l1211_121183

-- Define the function representing the equation
def f (a : ℝ) (x : ℝ) : Prop :=
  2 * Real.log (x + 3) = Real.log (a * x)

-- Define the set of a values for which the equation has a unique solution
def uniqueSolutionSet : Set ℝ :=
  {a : ℝ | a < 0 ∨ a = 12}

-- Theorem statement
theorem unique_solution_characterization (a : ℝ) :
  (∃! x : ℝ, f a x) ↔ a ∈ uniqueSolutionSet :=
sorry

end NUMINAMATH_CALUDE_unique_solution_characterization_l1211_121183


namespace NUMINAMATH_CALUDE_integer_less_than_sqrt5_l1211_121114

theorem integer_less_than_sqrt5 : ∃ z : ℤ, |z| < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_integer_less_than_sqrt5_l1211_121114


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1211_121105

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1211_121105


namespace NUMINAMATH_CALUDE_no_solution_to_inequalities_l1211_121128

theorem no_solution_to_inequalities : 
  ¬∃ (x y : ℝ), (4*x^2 + 4*x*y + 19*y^2 ≤ 2) ∧ (x - y ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequalities_l1211_121128
