import Mathlib

namespace NUMINAMATH_CALUDE_cheese_calories_theorem_l3322_332213

/-- Calculates the remaining calories in a block of cheese -/
def remaining_calories (total_servings : ℕ) (calories_per_serving : ℕ) (eaten_servings : ℕ) : ℕ :=
  (total_servings - eaten_servings) * calories_per_serving

/-- Theorem: The remaining calories in a block of cheese with 16 servings, 
    where each serving contains 110 calories, and 5 servings have been eaten, 
    is equal to 1210 calories. -/
theorem cheese_calories_theorem : 
  remaining_calories 16 110 5 = 1210 := by
  sorry

end NUMINAMATH_CALUDE_cheese_calories_theorem_l3322_332213


namespace NUMINAMATH_CALUDE_yuri_roll_less_than_yuko_l3322_332285

/-- Represents a player's dice roll in the board game -/
structure DiceRoll :=
  (d1 d2 d3 : Nat)

/-- The game state after both players have rolled -/
structure GameState :=
  (yuri_roll : DiceRoll)
  (yuko_roll : DiceRoll)
  (yuko_ahead : Bool)

/-- Calculate the sum of a dice roll -/
def roll_sum (roll : DiceRoll) : Nat :=
  roll.d1 + roll.d2 + roll.d3

/-- Theorem stating that if Yuko is ahead, Yuri's roll sum must be less than Yuko's -/
theorem yuri_roll_less_than_yuko (state : GameState) 
  (h1 : state.yuko_roll = DiceRoll.mk 1 5 6)
  (h2 : state.yuko_ahead = true) : 
  roll_sum state.yuri_roll < roll_sum state.yuko_roll :=
by
  sorry

end NUMINAMATH_CALUDE_yuri_roll_less_than_yuko_l3322_332285


namespace NUMINAMATH_CALUDE_spending_difference_l3322_332207

/-- Represents the price of masks and the quantities purchased by Jiajia and Qiqi. -/
structure MaskPurchase where
  a : ℝ  -- Price of N95 mask in yuan
  b : ℝ  -- Price of regular medical mask in yuan
  jiajia_n95 : ℕ := 5  -- Number of N95 masks Jiajia bought
  jiajia_regular : ℕ := 2  -- Number of regular masks Jiajia bought
  qiqi_n95 : ℕ := 2  -- Number of N95 masks Qiqi bought
  qiqi_regular : ℕ := 5  -- Number of regular masks Qiqi bought

/-- The price difference between N95 and regular masks is 3 yuan. -/
def price_difference (m : MaskPurchase) : Prop :=
  m.a = m.b + 3

/-- The difference in spending between Jiajia and Qiqi is 9 yuan. -/
theorem spending_difference (m : MaskPurchase) 
  (h : price_difference m) : 
  (m.jiajia_n95 : ℝ) * m.a + (m.jiajia_regular : ℝ) * m.b - 
  ((m.qiqi_n95 : ℝ) * m.a + (m.qiqi_regular : ℝ) * m.b) = 9 := by
  sorry

end NUMINAMATH_CALUDE_spending_difference_l3322_332207


namespace NUMINAMATH_CALUDE_extra_grass_seed_coverage_l3322_332294

/-- Calculates the extra coverage of grass seed after reseeding a lawn -/
theorem extra_grass_seed_coverage 
  (lawn_length : ℕ) 
  (lawn_width : ℕ) 
  (seed_bags : ℕ) 
  (coverage_per_bag : ℕ) : 
  lawn_length = 35 → 
  lawn_width = 48 → 
  seed_bags = 6 → 
  coverage_per_bag = 500 → 
  seed_bags * coverage_per_bag - lawn_length * lawn_width = 1320 :=
by
  sorry

#check extra_grass_seed_coverage

end NUMINAMATH_CALUDE_extra_grass_seed_coverage_l3322_332294


namespace NUMINAMATH_CALUDE_math_books_count_l3322_332236

theorem math_books_count (total_books : ℕ) (math_price history_price total_price : ℕ) :
  total_books = 80 →
  math_price = 4 →
  history_price = 5 →
  total_price = 373 →
  ∃ (math_books : ℕ), 
    math_books * math_price + (total_books - math_books) * history_price = total_price ∧
    math_books = 27 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l3322_332236


namespace NUMINAMATH_CALUDE_square_roots_and_cube_root_problem_l3322_332269

theorem square_roots_and_cube_root_problem (a b : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (3 * a - 14)^2 = k ∧ (a - 2)^2 = k) → 
  ((b - 15)^(1/3) = -3) → 
  (a = 4 ∧ b = -12 ∧ (∀ x : ℝ, x^2 = 4*a + b ↔ x = 2 ∨ x = -2)) :=
by sorry

end NUMINAMATH_CALUDE_square_roots_and_cube_root_problem_l3322_332269


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3322_332279

theorem complex_equation_solution (z : ℂ) : 2 * z * Complex.I = 1 + 3 * Complex.I → z = (3 / 2 : ℂ) - (1 / 2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3322_332279


namespace NUMINAMATH_CALUDE_expression_evaluation_l3322_332217

theorem expression_evaluation : 
  let f (x : ℚ) := (2 * x + 1) / (2 * x - 1)
  f 2 = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3322_332217


namespace NUMINAMATH_CALUDE_stephanie_oranges_l3322_332249

/-- The number of times Stephanie went to the store last month -/
def store_visits : ℕ := 8

/-- The total number of oranges Stephanie bought last month -/
def total_oranges : ℕ := 16

/-- The number of oranges Stephanie buys each time she goes to the store -/
def oranges_per_visit : ℕ := total_oranges / store_visits

theorem stephanie_oranges : oranges_per_visit = 2 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_oranges_l3322_332249


namespace NUMINAMATH_CALUDE_scientific_notation_of_505000_l3322_332250

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The number to be represented in scientific notation -/
def number : ℝ := 505000

/-- The expected scientific notation representation -/
def expected : ScientificNotation :=
  { coefficient := 5.05
    exponent := 5
    is_valid := by sorry }

/-- Theorem stating that the scientific notation of 505,000 is 5.05 × 10^5 -/
theorem scientific_notation_of_505000 :
  toScientificNotation number = expected := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_505000_l3322_332250


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3322_332228

theorem solve_linear_equation (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x + 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3322_332228


namespace NUMINAMATH_CALUDE_max_product_constrained_sum_l3322_332206

theorem max_product_constrained_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  x * y ≤ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 4 ∧ a * b = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_sum_l3322_332206


namespace NUMINAMATH_CALUDE_choose_two_from_eleven_l3322_332243

theorem choose_two_from_eleven (n : ℕ) (k : ℕ) : n = 11 → k = 2 → Nat.choose n k = 55 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_eleven_l3322_332243


namespace NUMINAMATH_CALUDE_sheila_attends_probability_l3322_332239

-- Define the probabilities
def prob_rain : ℝ := 0.5
def prob_sunny : ℝ := 1 - prob_rain
def prob_sheila_goes_rain : ℝ := 0.3
def prob_sheila_goes_sunny : ℝ := 0.7
def prob_friend_drives : ℝ := 0.5

-- Define the probability of Sheila attending the picnic
def prob_sheila_attends : ℝ :=
  (prob_rain * prob_sheila_goes_rain + prob_sunny * prob_sheila_goes_sunny) * prob_friend_drives

-- Theorem statement
theorem sheila_attends_probability :
  prob_sheila_attends = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_sheila_attends_probability_l3322_332239


namespace NUMINAMATH_CALUDE_de_length_l3322_332272

/-- Triangle ABC with sides AB = 24, AC = 26, and BC = 22 -/
structure Triangle :=
  (AB : ℝ) (AC : ℝ) (BC : ℝ)

/-- Points D and E on sides AB and AC respectively -/
structure PointsDE (T : Triangle) :=
  (D : ℝ) (E : ℝ)
  (hD : D ≥ 0 ∧ D ≤ T.AB)
  (hE : E ≥ 0 ∧ E ≤ T.AC)

/-- DE is parallel to BC and contains the center of the inscribed circle -/
def contains_incenter (T : Triangle) (P : PointsDE T) : Prop :=
  ∃ k : ℝ, P.D / T.AB = P.E / T.AC ∧ k > 0 ∧ k < 1 ∧
    P.D = k * T.AB ∧ P.E = k * T.AC

/-- The main theorem -/
theorem de_length (T : Triangle) (P : PointsDE T) 
    (h1 : T.AB = 24) (h2 : T.AC = 26) (h3 : T.BC = 22)
    (h4 : contains_incenter T P) : 
  P.E - P.D = 275 / 18 := by sorry

end NUMINAMATH_CALUDE_de_length_l3322_332272


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3322_332264

theorem quadratic_factorization (a b c d : ℤ) : 
  (∀ x : ℝ, 6 * x^2 + x - 12 = (a * x + b) * (c * x + d)) →
  |a| + |b| + |c| + |d| = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3322_332264


namespace NUMINAMATH_CALUDE_xyz_value_l3322_332242

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 16 * Real.rpow 4 (1/3))
  (h2 : x * z = 28 * Real.rpow 4 (1/3))
  (h3 : y * z = 112 / Real.rpow 4 (1/3)) :
  x * y * z = 112 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3322_332242


namespace NUMINAMATH_CALUDE_x_cubed_coef_sum_l3322_332271

def binomial_coef (n k : ℕ) : ℤ := (-1)^k * (n.choose k)

def expansion_coef (n : ℕ) : ℤ := binomial_coef n 3

theorem x_cubed_coef_sum :
  expansion_coef 5 + expansion_coef 6 + expansion_coef 7 + expansion_coef 8 = -121 :=
by sorry

end NUMINAMATH_CALUDE_x_cubed_coef_sum_l3322_332271


namespace NUMINAMATH_CALUDE_smallest_number_with_all_factors_l3322_332277

def alice_number : ℕ := 36

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → (p ∣ n → p ∣ m)

theorem smallest_number_with_all_factors :
  ∃ k : ℕ, k > 0 ∧ has_all_prime_factors alice_number k ∧
  ∀ m : ℕ, m > 0 → has_all_prime_factors alice_number m → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_all_factors_l3322_332277


namespace NUMINAMATH_CALUDE_even_sum_probability_l3322_332281

/-- Probability of obtaining an even sum when spinning two wheels -/
theorem even_sum_probability (wheel1_total : ℕ) (wheel1_even : ℕ) (wheel2_total : ℕ) (wheel2_even : ℕ)
  (h1 : wheel1_total = 6)
  (h2 : wheel1_even = 2)
  (h3 : wheel2_total = 5)
  (h4 : wheel2_even = 3) :
  (wheel1_even : ℚ) / wheel1_total * (wheel2_even : ℚ) / wheel2_total +
  ((wheel1_total - wheel1_even) : ℚ) / wheel1_total * ((wheel2_total - wheel2_even) : ℚ) / wheel2_total =
  7 / 15 :=
by sorry

end NUMINAMATH_CALUDE_even_sum_probability_l3322_332281


namespace NUMINAMATH_CALUDE_min_segments_11x11_grid_l3322_332290

/-- Represents a grid of lines -/
structure Grid :=
  (horizontal_lines : ℕ)
  (vertical_lines : ℕ)

/-- Calculates the number of internal nodes in a grid -/
def internal_nodes (g : Grid) : ℕ :=
  (g.horizontal_lines - 2) * (g.vertical_lines - 2)

/-- Calculates the minimum number of segments to erase -/
def min_segments_to_erase (g : Grid) : ℕ :=
  (internal_nodes g + 1) / 2

/-- The theorem stating the minimum number of segments to erase in an 11x11 grid -/
theorem min_segments_11x11_grid :
  ∃ (g : Grid), g.horizontal_lines = 11 ∧ g.vertical_lines = 11 ∧
  min_segments_to_erase g = 41 :=
sorry

end NUMINAMATH_CALUDE_min_segments_11x11_grid_l3322_332290


namespace NUMINAMATH_CALUDE_cd_length_sum_l3322_332233

theorem cd_length_sum : 
  let num_cds : ℕ := 3
  let regular_cd_length : ℝ := 1.5
  let long_cd_length : ℝ := 2 * regular_cd_length
  let total_length : ℝ := 2 * regular_cd_length + long_cd_length
  total_length = 6 := by sorry

end NUMINAMATH_CALUDE_cd_length_sum_l3322_332233


namespace NUMINAMATH_CALUDE_two_designs_are_three_fifths_l3322_332200

/-- Represents a design with a shaded region --/
structure Design where
  shaded_fraction : Rat

/-- Checks if a given fraction is equal to 3/5 --/
def is_three_fifths (f : Rat) : Bool :=
  f = 3 / 5

/-- Counts the number of designs with shaded region equal to 3/5 --/
def count_three_fifths (designs : List Design) : Nat :=
  designs.filter (fun d => is_three_fifths d.shaded_fraction) |>.length

/-- The main theorem stating that exactly 2 out of 5 given designs have 3/5 shaded area --/
theorem two_designs_are_three_fifths :
  let designs : List Design := [
    ⟨3 / 8⟩,
    ⟨12 / 20⟩,
    ⟨2 / 3⟩,
    ⟨15 / 25⟩,
    ⟨4 / 8⟩
  ]
  count_three_fifths designs = 2 := by
  sorry


end NUMINAMATH_CALUDE_two_designs_are_three_fifths_l3322_332200


namespace NUMINAMATH_CALUDE_sports_package_channels_l3322_332245

/-- The number of channels in Larry's cable package at different stages --/
structure CablePackage where
  initial : Nat
  after_replacement : Nat
  after_reduction : Nat
  after_sports : Nat
  after_supreme : Nat
  final : Nat

/-- The number of channels in the sports package --/
def sports_package (cp : CablePackage) : Nat :=
  cp.final - cp.after_supreme

theorem sports_package_channels : ∀ cp : CablePackage,
  cp.initial = 150 →
  cp.after_replacement = cp.initial - 20 + 12 →
  cp.after_reduction = cp.after_replacement - 10 →
  cp.after_supreme = cp.after_sports + 7 →
  cp.final = 147 →
  sports_package cp = 8 := by
  sorry

#eval sports_package { 
  initial := 150,
  after_replacement := 142,
  after_reduction := 132,
  after_sports := 140,
  after_supreme := 147,
  final := 147
}

end NUMINAMATH_CALUDE_sports_package_channels_l3322_332245


namespace NUMINAMATH_CALUDE_margo_walk_distance_l3322_332223

/-- Calculates the total distance walked given the time and speed for each direction -/
def totalDistanceWalked (timeToFriend timeFromFriend : ℚ) (speedToFriend speedFromFriend : ℚ) : ℚ :=
  timeToFriend * speedToFriend + timeFromFriend * speedFromFriend

theorem margo_walk_distance :
  let timeToFriend : ℚ := 15 / 60
  let timeFromFriend : ℚ := 25 / 60
  let speedToFriend : ℚ := 5
  let speedFromFriend : ℚ := 3
  totalDistanceWalked timeToFriend timeFromFriend speedToFriend speedFromFriend = 5 / 2 := by
  sorry

#eval totalDistanceWalked (15/60) (25/60) 5 3

end NUMINAMATH_CALUDE_margo_walk_distance_l3322_332223


namespace NUMINAMATH_CALUDE_dog_hare_speed_ratio_challenging_terrain_l3322_332219

/-- Represents the ratio of dog leaps to hare leaps -/
def dogHareLeapRatio : ℚ := 10 / 2

/-- Represents the ratio of dog leap distance to hare leap distance -/
def dogHareDistanceRatio : ℚ := 2 / 1

/-- Represents the reduction factor of dog's leap distance on challenging terrain -/
def dogReductionFactor : ℚ := 3 / 4

/-- Represents the reduction factor of hare's leap distance on challenging terrain -/
def hareReductionFactor : ℚ := 1 / 2

/-- Theorem stating the speed ratio of dog to hare on challenging terrain -/
theorem dog_hare_speed_ratio_challenging_terrain :
  (dogHareDistanceRatio * dogReductionFactor) / hareReductionFactor = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_dog_hare_speed_ratio_challenging_terrain_l3322_332219


namespace NUMINAMATH_CALUDE_sin_pi_sixth_minus_2alpha_l3322_332268

theorem sin_pi_sixth_minus_2alpha (α : ℝ) 
  (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.sin (π / 6 - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_sixth_minus_2alpha_l3322_332268


namespace NUMINAMATH_CALUDE_complex_equation_imaginary_part_l3322_332227

theorem complex_equation_imaginary_part :
  ∀ z : ℂ, (1 + Complex.I) / (3 * Complex.I + z) = Complex.I →
  z.im = -4 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_imaginary_part_l3322_332227


namespace NUMINAMATH_CALUDE_ellipse_equation_form_l3322_332253

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  center_x : ℝ
  center_y : ℝ
  foci_on_axes : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

-- Define the conditions
def satisfies_conditions (e : Ellipse) : Prop :=
  e.center_x = 0 ∧
  e.center_y = 0 ∧
  e.foci_on_axes ∧
  e.eccentricity = Real.sqrt 3 / 2 ∧
  e.passes_through = (2, 0)

-- Define the equation of the ellipse
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.center_x)^2 / e.a^2 + (y - e.center_y)^2 / e.b^2 = 1

-- Theorem statement
theorem ellipse_equation_form (e : Ellipse) :
  satisfies_conditions e →
  (∀ x y, ellipse_equation e x y ↔ (x^2 / 4 + y^2 = 1 ∨ x^2 / 4 + y^2 / 16 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_form_l3322_332253


namespace NUMINAMATH_CALUDE_egg_problem_solution_l3322_332248

/-- Calculates the difference between perfect and cracked eggs given the initial conditions --/
def egg_difference (total_dozens : ℕ) (broken : ℕ) : ℕ :=
  let total := total_dozens * 12
  let cracked := 2 * broken
  let perfect := total - broken - cracked
  perfect - cracked

theorem egg_problem_solution :
  egg_difference 2 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_egg_problem_solution_l3322_332248


namespace NUMINAMATH_CALUDE_prime_sum_equation_l3322_332201

theorem prime_sum_equation (a b n : ℕ) : 
  a < b ∧ 
  Nat.Prime a ∧ 
  Nat.Prime b ∧ 
  Odd n ∧ 
  a + b * n = 487 → 
  ((a = 2 ∧ b = 5 ∧ n = 97) ∨ (a = 2 ∧ b = 97 ∧ n = 5)) :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_equation_l3322_332201


namespace NUMINAMATH_CALUDE_boys_at_least_35_percent_l3322_332230

/-- Represents a child camp with 3-rooms and 4-rooms -/
structure ChildCamp where
  girls_3room : ℕ
  girls_4room : ℕ
  boys_3room : ℕ
  boys_4room : ℕ

/-- The proportion of boys in the camp -/
def boy_proportion (camp : ChildCamp) : ℚ :=
  (3 * camp.boys_3room + 4 * camp.boys_4room) / 
  (3 * camp.girls_3room + 4 * camp.girls_4room + 3 * camp.boys_3room + 4 * camp.boys_4room)

/-- Theorem stating that the proportion of boys is at least 35% -/
theorem boys_at_least_35_percent (camp : ChildCamp) 
  (h1 : 2 * (camp.girls_4room + camp.boys_4room) ≥ 
        camp.girls_3room + camp.girls_4room + camp.boys_3room + camp.boys_4room)
  (h2 : 3 * camp.girls_3room ≥ 8 * camp.girls_4room) :
  boy_proportion camp ≥ 7/20 := by
  sorry

end NUMINAMATH_CALUDE_boys_at_least_35_percent_l3322_332230


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3322_332265

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 4) :
  (∀ x : ℝ, g x = x + 5) ∨ (∀ x : ℝ, g x = -x - 3) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3322_332265


namespace NUMINAMATH_CALUDE_simplify_expression_l3322_332295

theorem simplify_expression : 18 * (8 / 15) * (1 / 12)^2 = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3322_332295


namespace NUMINAMATH_CALUDE_four_numbers_between_l3322_332211

theorem four_numbers_between :
  ∃ (a b c d : ℝ), 5.45 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < 5.47 := by
  sorry

end NUMINAMATH_CALUDE_four_numbers_between_l3322_332211


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_parallel_planes_l3322_332205

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- State the theorem
theorem lines_perpendicular_to_parallel_planes 
  (m n : Line) (α β : Plane) :
  m ≠ n →
  α ≠ β →
  parallel m n →
  perpendicular m α →
  perpendicular n β →
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_parallel_planes_l3322_332205


namespace NUMINAMATH_CALUDE_system_solution_l3322_332224

theorem system_solution (x y : ℝ) : 
  (3 * x^2 + 9 * x + 3 * y + 2 = 0 ∧ 3 * x + y + 4 = 0) ↔ 
  (y = -4 + Real.sqrt 30 ∨ y = -4 - Real.sqrt 30) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3322_332224


namespace NUMINAMATH_CALUDE_ramesh_refrigerator_price_l3322_332287

/-- Represents the price Ramesh paid for a refrigerator given certain conditions --/
def ramesh_paid_price (P : ℝ) : Prop :=
  let discount_rate : ℝ := 0.20
  let transport_cost : ℝ := 125
  let installation_cost : ℝ := 250
  let profit_rate : ℝ := 0.10
  let selling_price : ℝ := 20350
  (1 + profit_rate) * P = selling_price ∧
  (1 - discount_rate) * P + transport_cost + installation_cost = 15175

theorem ramesh_refrigerator_price :
  ∃ P : ℝ, ramesh_paid_price P :=
sorry

end NUMINAMATH_CALUDE_ramesh_refrigerator_price_l3322_332287


namespace NUMINAMATH_CALUDE_sqrt_two_subset_P_l3322_332263

def P : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem sqrt_two_subset_P : {Real.sqrt 2} ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_subset_P_l3322_332263


namespace NUMINAMATH_CALUDE_circle_chord_triangles_l3322_332266

-- Define the number of points on the circle
def n : ℕ := 9

-- Define the number of chords
def num_chords : ℕ := n.choose 2

-- Define the number of intersections inside the circle
def num_intersections : ℕ := n.choose 4

-- Define the number of triangles formed by intersections
def num_triangles : ℕ := num_intersections.choose 3

-- Theorem statement
theorem circle_chord_triangles :
  num_triangles = 315750 :=
sorry

end NUMINAMATH_CALUDE_circle_chord_triangles_l3322_332266


namespace NUMINAMATH_CALUDE_base6_45_equals_29_l3322_332222

/-- Converts a base-6 number to decimal --/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The decimal representation of 45 in base 6 --/
def base6_45 : Nat := base6ToDecimal [5, 4]

theorem base6_45_equals_29 : base6_45 = 29 := by sorry

end NUMINAMATH_CALUDE_base6_45_equals_29_l3322_332222


namespace NUMINAMATH_CALUDE_smallest_B_for_divisibility_by_three_l3322_332283

def seven_digit_number (B : Nat) : Nat :=
  4000000 + B * 100000 + 803942

theorem smallest_B_for_divisibility_by_three :
  ∃ (B : Nat), B < 10 ∧ 
    seven_digit_number B % 3 = 0 ∧
    ∀ (C : Nat), C < B → seven_digit_number C % 3 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_B_for_divisibility_by_three_l3322_332283


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3322_332235

theorem min_value_sum_squares (a b c : ℝ) (h : a + 2*b + 3*c = 6) :
  ∃ (min : ℝ), min = 12 ∧ a^2 + 4*b^2 + 9*c^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3322_332235


namespace NUMINAMATH_CALUDE_stock_purchase_probabilities_l3322_332237

/-- The number of stocks available for purchase -/
def num_stocks : ℕ := 6

/-- The number of individuals making purchases -/
def num_individuals : ℕ := 4

/-- The probability that all individuals purchase the same stock -/
def prob_all_same : ℚ := 1 / 216

/-- The probability that at most two individuals purchase the same stock -/
def prob_at_most_two_same : ℚ := 65 / 72

/-- Given 6 stocks and 4 individuals randomly selecting one stock each,
    prove the probabilities of certain outcomes -/
theorem stock_purchase_probabilities :
  (prob_all_same = 1 / num_stocks ^ (num_individuals - 1)) ∧
  (prob_at_most_two_same = 
    (num_stocks * (num_stocks - 1) * Nat.choose num_individuals 2 + 
     num_stocks * Nat.factorial num_individuals) / 
    (num_stocks ^ num_individuals)) := by
  sorry

end NUMINAMATH_CALUDE_stock_purchase_probabilities_l3322_332237


namespace NUMINAMATH_CALUDE_solve_equation_l3322_332261

theorem solve_equation : 45 / (7 - 3/4) = 36/5 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3322_332261


namespace NUMINAMATH_CALUDE_parabola_b_value_l3322_332241

/-- Given a parabola y = ax^2 + bx + c with vertex (p, p) and y-intercept (0, -2p), where p ≠ 0, 
    the value of b is 6/p. -/
theorem parabola_b_value (a b c p : ℝ) (h_p : p ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 + p) →
  (a * 0^2 + b * 0 + c = -2 * p) →
  b = 6 / p := by
  sorry

end NUMINAMATH_CALUDE_parabola_b_value_l3322_332241


namespace NUMINAMATH_CALUDE_complex_division_equality_l3322_332229

/-- Given that i is the imaginary unit, prove that (2 + 4i) / (1 + i) = 3 + i -/
theorem complex_division_equality : (2 + 4 * I) / (1 + I) = 3 + I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l3322_332229


namespace NUMINAMATH_CALUDE_bottles_drunk_per_day_l3322_332244

theorem bottles_drunk_per_day (initial_bottles : ℕ) (remaining_bottles : ℕ) (days : ℕ) : 
  initial_bottles = 301 → remaining_bottles = 157 → days = 1 →
  initial_bottles - remaining_bottles = 144 := by
sorry

end NUMINAMATH_CALUDE_bottles_drunk_per_day_l3322_332244


namespace NUMINAMATH_CALUDE_orange_stack_problem_l3322_332262

/-- Calculates the number of oranges in a pyramid-like stack --/
def orangeStackSum (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let layers := min base_width base_length
  let layerSum (n : ℕ) : ℕ := (base_width - n + 1) * (base_length - n + 1)
  (List.range layers).map layerSum |>.sum

/-- The pyramid-like stack of oranges problem --/
theorem orange_stack_problem :
  orangeStackSum 5 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_orange_stack_problem_l3322_332262


namespace NUMINAMATH_CALUDE_sqrt_53_plus_20_sqrt_7_representation_l3322_332208

theorem sqrt_53_plus_20_sqrt_7_representation : 
  ∃ (a b c : ℤ), 
    (∀ (n : ℕ), n > 1 → ¬ (∃ (k : ℕ), c = n^2 * k)) → 
    Real.sqrt (53 + 20 * Real.sqrt 7) = a + b * Real.sqrt c ∧ 
    a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_53_plus_20_sqrt_7_representation_l3322_332208


namespace NUMINAMATH_CALUDE_license_plate_count_l3322_332275

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The number of letters in a license plate --/
def letters_count : ℕ := 4

/-- The number of digits in a license plate --/
def digits_count : ℕ := 3

/-- The number of available digits (0-9) --/
def available_digits : ℕ := 10

/-- Calculates the number of license plate combinations --/
def license_plate_combinations : ℕ :=
  alphabet_size *
  (Nat.choose (alphabet_size - 1) 2) *
  (Nat.choose letters_count 2) *
  2 *
  available_digits *
  (available_digits - 1) *
  (available_digits - 2)

theorem license_plate_count :
  license_plate_combinations = 67392000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3322_332275


namespace NUMINAMATH_CALUDE_expression_simplification_l3322_332252

/-- Given real numbers x and y, prove that the expression
    ((x² + y²)(x² - y²)) / ((x² + y²) + (x² - y²)) + ((x² + y²) + (x² - y²)) / ((x² + y²)(x² - y²))
    simplifies to (x⁴ + y⁴)² / (2x²(x⁴ - y⁴)) -/
theorem expression_simplification (x y : ℝ) (h : x ≠ 0) :
  let P := x^2 + y^2
  let Q := x^2 - y^2
  (P * Q) / (P + Q) + (P + Q) / (P * Q) = (x^4 + y^4)^2 / (2 * x^2 * (x^4 - y^4)) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3322_332252


namespace NUMINAMATH_CALUDE_identity_proof_l3322_332274

theorem identity_proof (a b x y θ φ : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (h1 : (a - b) * Real.sin (θ / 2) * Real.cos (φ / 2) + 
        (a + b) * Real.cos (θ / 2) * Real.sin (φ / 2) = 0)
  (h2 : x / a * Real.cos θ + y / b * Real.sin θ = 1)
  (h3 : x / a * Real.cos φ + y / b * Real.sin φ = 1) :
  x^2 / a^2 + (b^2 - a^2) / b^4 * y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_identity_proof_l3322_332274


namespace NUMINAMATH_CALUDE_ava_distance_covered_l3322_332291

/-- Represents the race scenario where Aubrey and Ava are running --/
structure RaceScenario where
  race_length : ℝ  -- Length of the race in kilometers
  ava_remaining : ℝ  -- Distance Ava has left to finish in meters

/-- Calculates the distance Ava has covered in meters --/
def distance_covered (scenario : RaceScenario) : ℝ :=
  scenario.race_length * 1000 - scenario.ava_remaining

/-- Theorem stating that Ava covered 833 meters in the given scenario --/
theorem ava_distance_covered (scenario : RaceScenario)
  (h1 : scenario.race_length = 1)
  (h2 : scenario.ava_remaining = 167) :
  distance_covered scenario = 833 := by
  sorry

end NUMINAMATH_CALUDE_ava_distance_covered_l3322_332291


namespace NUMINAMATH_CALUDE_ball_purchase_solution_l3322_332234

/-- Represents the cost and quantity of soccer balls and basketballs -/
structure BallPurchase where
  soccer_cost : ℝ
  basketball_cost : ℝ
  soccer_quantity : ℕ
  basketball_quantity : ℕ

/-- Conditions for the ball purchase problem -/
def BallPurchaseConditions (bp : BallPurchase) : Prop :=
  7 * bp.soccer_cost = 5 * bp.basketball_cost ∧
  40 * bp.soccer_cost + 20 * bp.basketball_cost = 3400 ∧
  bp.soccer_quantity + bp.basketball_quantity = 100 ∧
  bp.soccer_cost * bp.soccer_quantity + bp.basketball_cost * bp.basketball_quantity ≤ 6300

/-- Theorem stating the solution to the ball purchase problem -/
theorem ball_purchase_solution (bp : BallPurchase) 
  (h : BallPurchaseConditions bp) : 
  bp.soccer_cost = 50 ∧ 
  bp.basketball_cost = 70 ∧ 
  bp.basketball_quantity ≤ 65 :=
sorry

end NUMINAMATH_CALUDE_ball_purchase_solution_l3322_332234


namespace NUMINAMATH_CALUDE_theater_rows_l3322_332204

/-- Represents the number of rows in the theater. -/
def num_rows : ℕ := sorry

/-- Represents the number of students in the first condition. -/
def students_first_condition : ℕ := 30

/-- Represents the number of students in the second condition. -/
def students_second_condition : ℕ := 26

/-- Represents the minimum number of empty rows in the second condition. -/
def min_empty_rows : ℕ := 3

theorem theater_rows :
  (∀ (seating : Fin students_first_condition → Fin num_rows),
    ∃ (row : Fin num_rows) (s1 s2 : Fin students_first_condition),
      s1 ≠ s2 ∧ seating s1 = seating s2) ∧
  (∀ (seating : Fin students_second_condition → Fin num_rows),
    ∃ (empty_rows : Finset (Fin num_rows)),
      empty_rows.card ≥ min_empty_rows ∧
      ∀ (row : Fin num_rows),
        row ∈ empty_rows ↔ ∀ (s : Fin students_second_condition), seating s ≠ row) →
  num_rows = 29 :=
sorry

end NUMINAMATH_CALUDE_theater_rows_l3322_332204


namespace NUMINAMATH_CALUDE_probability_at_least_two_correct_l3322_332296

-- Define the number of questions and choices
def total_questions : ℕ := 30
def choices_per_question : ℕ := 6
def guessed_questions : ℕ := 5

-- Define the probability of a correct answer
def p_correct : ℚ := 1 / choices_per_question

-- Define the binomial probability function
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

-- State the theorem
theorem probability_at_least_two_correct :
  1 - binomial_prob guessed_questions 0 p_correct
    - binomial_prob guessed_questions 1 p_correct = 763 / 3888 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_correct_l3322_332296


namespace NUMINAMATH_CALUDE_two_cos_forty_five_equals_sqrt_two_l3322_332270

theorem two_cos_forty_five_equals_sqrt_two : 2 * Real.cos (π / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_two_cos_forty_five_equals_sqrt_two_l3322_332270


namespace NUMINAMATH_CALUDE_evaluate_expression_l3322_332210

theorem evaluate_expression : 
  (3^1005 + 4^1006)^2 - (3^1005 - 4^1006)^2 = 16 * 12^1005 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3322_332210


namespace NUMINAMATH_CALUDE_rain_probability_three_days_l3322_332256

theorem rain_probability_three_days 
  (prob_friday : ℝ) 
  (prob_saturday : ℝ) 
  (prob_sunday : ℝ) 
  (h1 : prob_friday = 0.40) 
  (h2 : prob_saturday = 0.60) 
  (h3 : prob_sunday = 0.35) 
  (h4 : 0 ≤ prob_friday ∧ prob_friday ≤ 1) 
  (h5 : 0 ≤ prob_saturday ∧ prob_saturday ≤ 1) 
  (h6 : 0 ≤ prob_sunday ∧ prob_sunday ≤ 1) :
  prob_friday * prob_saturday * prob_sunday = 0.084 := by
sorry

end NUMINAMATH_CALUDE_rain_probability_three_days_l3322_332256


namespace NUMINAMATH_CALUDE_flower_bed_fraction_is_correct_l3322_332240

/-- Represents the dimensions and areas of a yard with a pool and flower beds. -/
structure YardLayout where
  yard_length : ℝ
  yard_width : ℝ
  pool_length : ℝ
  pool_width : ℝ
  trapezoid_side1 : ℝ
  trapezoid_side2 : ℝ

/-- Calculates the fraction of usable yard area occupied by flower beds. -/
def flower_bed_fraction (layout : YardLayout) : ℚ :=
  sorry

/-- Theorem stating that the fraction of usable yard occupied by flower beds is 9/260. -/
theorem flower_bed_fraction_is_correct (layout : YardLayout) : 
  layout.yard_length = 30 ∧ 
  layout.yard_width = 10 ∧ 
  layout.pool_length = 10 ∧ 
  layout.pool_width = 4 ∧
  layout.trapezoid_side1 = 16 ∧ 
  layout.trapezoid_side2 = 22 →
  flower_bed_fraction layout = 9 / 260 :=
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_is_correct_l3322_332240


namespace NUMINAMATH_CALUDE_camel_cannot_move_to_adjacent_l3322_332280

def Board := Fin 10 × Fin 10

def adjacent (a b : Board) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 = b.2 - 1)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 = b.1 - 1))

def camel_move (a b : Board) : Prop :=
  (a.1 = b.1 + 1 ∧ (a.2 = b.2 + 3 ∨ a.2 = b.2 - 3)) ∨
  (a.1 = b.1 - 1 ∧ (a.2 = b.2 + 3 ∨ a.2 = b.2 - 3)) ∨
  (a.2 = b.2 + 1 ∧ (a.1 = b.1 + 3 ∨ a.1 = b.1 - 3)) ∨
  (a.2 = b.2 - 1 ∧ (a.1 = b.1 + 3 ∨ a.1 = b.1 - 3))

theorem camel_cannot_move_to_adjacent :
  ∀ (start finish : Board), adjacent start finish → ¬ camel_move start finish :=
by sorry

end NUMINAMATH_CALUDE_camel_cannot_move_to_adjacent_l3322_332280


namespace NUMINAMATH_CALUDE_choir_arrangement_l3322_332212

theorem choir_arrangement (total_members : ℕ) (num_rows : ℕ) (h1 : total_members = 51) (h2 : num_rows = 4) :
  ∃ (row : ℕ), row ≤ num_rows ∧ 13 ≤ (total_members / num_rows + (if row ≤ total_members % num_rows then 1 else 0)) :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_l3322_332212


namespace NUMINAMATH_CALUDE_fourth_group_number_l3322_332298

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (start : ℕ) (interval : ℕ) (group : ℕ) : ℕ :=
  start + (group - 1) * interval

/-- Theorem: In a systematic sampling of 90 students, with adjacent group numbers 14 and 23,
    the student number from the fourth group is 32. -/
theorem fourth_group_number :
  let total := 90
  let start := 14
  let interval := 23 - 14
  let group := 4
  systematic_sample total start interval group = 32 := by
  sorry

end NUMINAMATH_CALUDE_fourth_group_number_l3322_332298


namespace NUMINAMATH_CALUDE_shortest_distance_point_to_parabola_l3322_332299

/-- The shortest distance between a point and a parabola -/
theorem shortest_distance_point_to_parabola :
  let point := (7, 15)
  let parabola := λ x : ℝ => (x, x^2)
  ∃ d : ℝ, d = 2 * Real.sqrt 13 ∧
    ∀ x : ℝ, d ≤ Real.sqrt ((7 - x)^2 + (15 - x^2)^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_point_to_parabola_l3322_332299


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l3322_332216

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l3322_332216


namespace NUMINAMATH_CALUDE_solve_equation_l3322_332289

theorem solve_equation (r : ℚ) : (r - 45) / 2 = (3 - 2 * r) / 5 → r = 77 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3322_332289


namespace NUMINAMATH_CALUDE_jerry_candy_count_jerry_candy_count_proof_l3322_332202

theorem jerry_candy_count : ℕ → Prop :=
  fun total_candy : ℕ =>
    ∃ (candy_per_bag : ℕ),
      -- Total number of bags
      (9 : ℕ) * candy_per_bag = total_candy ∧
      -- Number of non-chocolate bags
      (9 - 2 - 3 : ℕ) * candy_per_bag = 28 ∧
      -- The result we want to prove
      total_candy = 63

-- The proof of the theorem
theorem jerry_candy_count_proof : jerry_candy_count 63 := by
  sorry

end NUMINAMATH_CALUDE_jerry_candy_count_jerry_candy_count_proof_l3322_332202


namespace NUMINAMATH_CALUDE_cos_four_minus_sin_four_equals_cos_double_l3322_332218

theorem cos_four_minus_sin_four_equals_cos_double (θ : ℝ) :
  Real.cos θ ^ 4 - Real.sin θ ^ 4 = Real.cos (2 * θ) := by sorry

end NUMINAMATH_CALUDE_cos_four_minus_sin_four_equals_cos_double_l3322_332218


namespace NUMINAMATH_CALUDE_max_value_of_a_l3322_332231

theorem max_value_of_a : 
  (∃ (a : ℝ), ∀ (x : ℝ), x < a → x^2 - 2*x - 3 > 0) ∧ 
  (∀ (a : ℝ), ∃ (x : ℝ), x^2 - 2*x - 3 > 0 ∧ x ≥ a) →
  (∀ (b : ℝ), (∃ (a : ℝ), ∀ (x : ℝ), x < a → x^2 - 2*x - 3 > 0) ∧ 
              (∀ (a : ℝ), ∃ (x : ℝ), x^2 - 2*x - 3 > 0 ∧ x ≥ a) → 
              b ≤ -1) ∧
  (∃ (a : ℝ), a = -1 ∧ 
              (∀ (x : ℝ), x < a → x^2 - 2*x - 3 > 0) ∧ 
              (∃ (x : ℝ), x^2 - 2*x - 3 > 0 ∧ x ≥ a)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3322_332231


namespace NUMINAMATH_CALUDE_cone_slant_height_l3322_332255

/-- Given a cone with base radius 5 cm and unfolded side area 60π cm², 
    prove that its slant height is 12 cm -/
theorem cone_slant_height (r : ℝ) (A : ℝ) (l : ℝ) : 
  r = 5 → A = 60 * Real.pi → A = (Real.pi * r * l) → l = 12 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l3322_332255


namespace NUMINAMATH_CALUDE_harry_fish_count_l3322_332278

/-- The number of fish Sam has -/
def sam_fish : ℕ := 7

/-- The number of fish Joe has relative to Sam -/
def joe_multiplier : ℕ := 8

/-- The number of fish Harry has relative to Joe -/
def harry_multiplier : ℕ := 4

/-- The number of fish Joe has -/
def joe_fish : ℕ := joe_multiplier * sam_fish

/-- The number of fish Harry has -/
def harry_fish : ℕ := harry_multiplier * joe_fish

theorem harry_fish_count : harry_fish = 224 := by
  sorry

end NUMINAMATH_CALUDE_harry_fish_count_l3322_332278


namespace NUMINAMATH_CALUDE_power_mod_eleven_l3322_332286

theorem power_mod_eleven : 5^2023 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l3322_332286


namespace NUMINAMATH_CALUDE_double_negation_l3322_332215

theorem double_negation (x : ℝ) : -(-x) = x := by
  sorry

end NUMINAMATH_CALUDE_double_negation_l3322_332215


namespace NUMINAMATH_CALUDE_bowling_team_size_l3322_332258

/-- The number of players in a bowling team -/
def num_players : ℕ := sorry

/-- The league record average score per player per round -/
def league_record : ℕ := 287

/-- The number of rounds in a season -/
def num_rounds : ℕ := 10

/-- The team's current total score after 9 rounds -/
def current_score : ℕ := 10440

/-- The difference between the league record and the minimum average needed in the final round -/
def final_round_diff : ℕ := 27

theorem bowling_team_size :
  (num_players * league_record * num_rounds - current_score) / num_players = 
  league_record - final_round_diff ∧
  num_players = 4 := by sorry

end NUMINAMATH_CALUDE_bowling_team_size_l3322_332258


namespace NUMINAMATH_CALUDE_judy_caught_one_fish_l3322_332284

/-- Represents the number of fish caught by each family member and other fishing details -/
structure FishingTrip where
  ben_fish : ℕ
  billy_fish : ℕ
  jim_fish : ℕ
  susie_fish : ℕ
  thrown_back : ℕ
  total_filets : ℕ
  filets_per_fish : ℕ

/-- Calculates the number of fish Judy caught based on the fishing trip details -/
def judy_fish (trip : FishingTrip) : ℕ :=
  (trip.total_filets / trip.filets_per_fish) -
  (trip.ben_fish + trip.billy_fish + trip.jim_fish + trip.susie_fish - trip.thrown_back)

/-- Theorem stating that Judy caught 1 fish given the specific conditions of the fishing trip -/
theorem judy_caught_one_fish :
  let trip : FishingTrip := {
    ben_fish := 4,
    billy_fish := 3,
    jim_fish := 2,
    susie_fish := 5,
    thrown_back := 3,
    total_filets := 24,
    filets_per_fish := 2
  }
  judy_fish trip = 1 := by sorry

end NUMINAMATH_CALUDE_judy_caught_one_fish_l3322_332284


namespace NUMINAMATH_CALUDE_inverse_variation_l3322_332247

/-- Given quantities a and b that vary inversely, if b = 0.5 when a = 800, 
    then b = 0.25 when a = 1600 -/
theorem inverse_variation (a b : ℝ) (k : ℝ) (h1 : a * b = k) 
  (h2 : 800 * 0.5 = k) (h3 : a = 1600) : b = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_l3322_332247


namespace NUMINAMATH_CALUDE_pool_length_calculation_l3322_332282

/-- Calculates the length of a rectangular pool given its draining rate, width, depth, initial capacity, and time to drain. -/
theorem pool_length_calculation (drain_rate : ℝ) (width depth : ℝ) (initial_capacity : ℝ) (drain_time : ℝ) :
  drain_rate = 60 →
  width = 40 →
  depth = 10 →
  initial_capacity = 0.8 →
  drain_time = 800 →
  (drain_rate * drain_time) / initial_capacity / (width * depth) = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_pool_length_calculation_l3322_332282


namespace NUMINAMATH_CALUDE_labeling_existence_condition_l3322_332254

/-- A labeling function for lattice points -/
def LabelingFunction := ℤ × ℤ → ℕ+

/-- The property that a labeling satisfies the distance condition for a given c -/
def SatisfiesDistanceCondition (f : LabelingFunction) (c : ℝ) : Prop :=
  ∀ i : ℕ+, ∀ p q : ℤ × ℤ, f p = i ∧ f q = i → dist p q ≥ c ^ (i : ℝ)

/-- The property that a labeling uses only finitely many labels -/
def UsesFiniteLabels (f : LabelingFunction) : Prop :=
  ∃ n : ℕ, ∀ p : ℤ × ℤ, (f p : ℕ) ≤ n

/-- The main theorem -/
theorem labeling_existence_condition (c : ℝ) :
  (c > 0 ∧ c < Real.sqrt 2) ↔
  (∃ f : LabelingFunction, SatisfiesDistanceCondition f c ∧ UsesFiniteLabels f) :=
sorry

end NUMINAMATH_CALUDE_labeling_existence_condition_l3322_332254


namespace NUMINAMATH_CALUDE_thabo_books_l3322_332259

/-- The number of books Thabo owns -/
def total_books : ℕ := 220

/-- The number of hardcover nonfiction books Thabo owns -/
def hardcover_nonfiction : ℕ := sorry

/-- The number of paperback nonfiction books Thabo owns -/
def paperback_nonfiction : ℕ := sorry

/-- The number of paperback fiction books Thabo owns -/
def paperback_fiction : ℕ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem thabo_books :
  (paperback_nonfiction = hardcover_nonfiction + 20) ∧
  (paperback_fiction = 2 * paperback_nonfiction) ∧
  (hardcover_nonfiction + paperback_nonfiction + paperback_fiction = total_books) →
  hardcover_nonfiction = 40 := by
  sorry

end NUMINAMATH_CALUDE_thabo_books_l3322_332259


namespace NUMINAMATH_CALUDE_min_value_expression_l3322_332203

theorem min_value_expression (x : ℝ) (h : x > 10) : 
  x^2 / (x - 10) ≥ 40 ∧ ∃ x₀ > 10, x₀^2 / (x₀ - 10) = 40 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3322_332203


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3322_332232

/-- Given a triangle with inradius 2.5 cm and area 50 cm², its perimeter is 40 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 50 → A = r * (p / 2) → p = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3322_332232


namespace NUMINAMATH_CALUDE_cookie_jar_spending_ratio_l3322_332293

/-- Proves that the ratio of Martha's spending to Doris' spending is 1:2 --/
theorem cookie_jar_spending_ratio 
  (initial_amount : ℕ) 
  (doris_spent : ℕ) 
  (final_amount : ℕ) 
  (h1 : initial_amount = 24)
  (h2 : doris_spent = 6)
  (h3 : final_amount = 15) :
  ∃ (martha_spent : ℕ), 
    martha_spent = initial_amount - doris_spent - final_amount ∧
    martha_spent * 2 = doris_spent := by
  sorry

#check cookie_jar_spending_ratio

end NUMINAMATH_CALUDE_cookie_jar_spending_ratio_l3322_332293


namespace NUMINAMATH_CALUDE_two_fifths_percent_of_450_l3322_332297

theorem two_fifths_percent_of_450 : (2 / 5) / 100 * 450 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_percent_of_450_l3322_332297


namespace NUMINAMATH_CALUDE_least_sum_pqr_l3322_332292

theorem least_sum_pqr (p q r : ℕ) : 
  p > 1 → q > 1 → r > 1 → 
  17 * (p + 1) = 28 * (q + 1) ∧ 28 * (q + 1) = 35 * (r + 1) →
  ∀ p' q' r' : ℕ, 
    p' > 1 → q' > 1 → r' > 1 → 
    17 * (p' + 1) = 28 * (q' + 1) ∧ 28 * (q' + 1) = 35 * (r' + 1) →
    p + q + r ≤ p' + q' + r' ∧ p + q + r = 290 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_pqr_l3322_332292


namespace NUMINAMATH_CALUDE_binomial_variance_4_half_l3322_332225

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ :=
  ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: The variance of a binomial distribution B(4, 1/2) is 1 -/
theorem binomial_variance_4_half :
  ∀ ξ : BinomialDistribution, ξ.n = 4 ∧ ξ.p = 1/2 → variance ξ = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_4_half_l3322_332225


namespace NUMINAMATH_CALUDE_cheese_problem_l3322_332214

theorem cheese_problem (total_rats : ℕ) (cheese_first_night : ℕ) (rats_second_night : ℕ) :
  total_rats > rats_second_night →
  cheese_first_night = 10 →
  rats_second_night = 7 →
  (cheese_first_night : ℚ) / total_rats = 2 * ((1 : ℚ) / total_rats) →
  (∃ (original_cheese : ℕ), original_cheese = cheese_first_night + 1) :=
by
  sorry

#check cheese_problem

end NUMINAMATH_CALUDE_cheese_problem_l3322_332214


namespace NUMINAMATH_CALUDE_circle_condition_l3322_332273

/-- The equation of a circle in terms of parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0

/-- Theorem stating the condition for m to represent a circle -/
theorem circle_condition (m : ℝ) : 
  (∃ x y : ℝ, circle_equation x y m) ↔ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l3322_332273


namespace NUMINAMATH_CALUDE_sample_size_accuracy_l3322_332246

theorem sample_size_accuracy (population : Type) (sample : Set population) (estimate : Set population → ℝ) (accuracy : Set population → ℝ) :
  ∀ s₁ s₂ : Set population, s₁ ⊆ s₂ → accuracy s₁ ≤ accuracy s₂ := by
  sorry

end NUMINAMATH_CALUDE_sample_size_accuracy_l3322_332246


namespace NUMINAMATH_CALUDE_smallest_three_digit_perfect_square_append_l3322_332267

theorem smallest_three_digit_perfect_square_append : 
  ∃ (a : ℕ), 
    (100 ≤ a ∧ a ≤ 999) ∧ 
    (∃ (n : ℕ), 1001 * a + 1 = n^2) ∧
    (∀ (b : ℕ), 100 ≤ b ∧ b < a → ¬∃ (m : ℕ), 1001 * b + 1 = m^2) ∧
    a = 183 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_perfect_square_append_l3322_332267


namespace NUMINAMATH_CALUDE_second_book_has_32_pictures_l3322_332288

/-- The number of pictures in the second coloring book -/
def second_book_pictures (first_book_pictures colored_pictures remaining_pictures : ℕ) : ℕ :=
  (colored_pictures + remaining_pictures) - first_book_pictures

/-- Theorem stating that the second coloring book has 32 pictures -/
theorem second_book_has_32_pictures :
  second_book_pictures 23 44 11 = 32 := by
  sorry

end NUMINAMATH_CALUDE_second_book_has_32_pictures_l3322_332288


namespace NUMINAMATH_CALUDE_stream_speed_equation_l3322_332260

/-- The speed of the stream for a boat trip -/
theorem stream_speed_equation (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) 
  (h1 : boat_speed = 9)
  (h2 : distance = 210)
  (h3 : total_time = 84) :
  ∃ x : ℝ, x^2 = 39 ∧ 
    (distance / (boat_speed + x) + distance / (boat_speed - x) = total_time) := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_equation_l3322_332260


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3322_332220

def A : Set ℤ := {-1, 2, 3, 5}
def B : Set ℤ := {2, 4, 5}

theorem union_of_A_and_B : A ∪ B = {-1, 2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3322_332220


namespace NUMINAMATH_CALUDE_shape_area_is_94_l3322_332226

/-- A shape composed of three rectangles with given dimensions -/
structure Shape where
  rect1_width : ℕ
  rect1_height : ℕ
  rect2_width : ℕ
  rect2_height : ℕ
  rect3_width : ℕ
  rect3_height : ℕ

/-- Calculate the area of a rectangle -/
def rectangle_area (width height : ℕ) : ℕ := width * height

/-- Calculate the total area of the shape -/
def total_area (s : Shape) : ℕ :=
  rectangle_area s.rect1_width s.rect1_height +
  rectangle_area s.rect2_width s.rect2_height +
  rectangle_area s.rect3_width s.rect3_height

/-- The shape described in the problem -/
def problem_shape : Shape :=
  { rect1_width := 7
  , rect1_height := 7
  , rect2_width := 3
  , rect2_height := 5
  , rect3_width := 5
  , rect3_height := 6 }

theorem shape_area_is_94 : total_area problem_shape = 94 := by
  sorry


end NUMINAMATH_CALUDE_shape_area_is_94_l3322_332226


namespace NUMINAMATH_CALUDE_sum_of_integers_l3322_332257

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 180) : x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3322_332257


namespace NUMINAMATH_CALUDE_sum_equality_existence_l3322_332209

theorem sum_equality_existence (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h_n : n > 3)
  (h_pos : ∀ i, a i > 0)
  (h_strict : ∀ i j, i < j → a i < a j)
  (h_upper : a (Fin.last n) ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin n),
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧
    j ≠ k ∧ j ≠ l ∧ j ≠ m ∧
    k ≠ l ∧ k ≠ m ∧
    l ≠ m ∧
    a i.succ + a j.succ = a k.succ + a l.succ ∧
    a i.succ + a j.succ = a m.succ :=
by sorry

end NUMINAMATH_CALUDE_sum_equality_existence_l3322_332209


namespace NUMINAMATH_CALUDE_team_combinations_eq_18018_l3322_332276

/-- The number of ways to choose k elements from n elements --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players in the basketball team --/
def total_players : ℕ := 18

/-- The number of quadruplets in the team --/
def num_quadruplets : ℕ := 4

/-- The size of the team to be formed --/
def team_size : ℕ := 8

/-- The number of quadruplets that must be in the team --/
def required_quadruplets : ℕ := 2

/-- The number of ways to choose 8 players from a team of 18 players, 
    including exactly 2 out of 4 quadruplets --/
def team_combinations : ℕ :=
  choose num_quadruplets required_quadruplets * 
  choose (total_players - num_quadruplets) (team_size - required_quadruplets)

theorem team_combinations_eq_18018 : team_combinations = 18018 := by
  sorry

end NUMINAMATH_CALUDE_team_combinations_eq_18018_l3322_332276


namespace NUMINAMATH_CALUDE_fraction_division_l3322_332251

theorem fraction_division : (4 : ℚ) / 5 / ((8 : ℚ) / 15) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l3322_332251


namespace NUMINAMATH_CALUDE_sales_solution_l3322_332221

def sales_problem (sales1 sales2 sales3 sales4 desired_average : ℕ) : Prop :=
  let total_months : ℕ := 5
  let known_sales_sum : ℕ := sales1 + sales2 + sales3 + sales4
  let total_required : ℕ := desired_average * total_months
  let fifth_month_sales : ℕ := total_required - known_sales_sum
  fifth_month_sales = 7870

theorem sales_solution :
  sales_problem 5420 5660 6200 6350 6300 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_solution_l3322_332221


namespace NUMINAMATH_CALUDE_monotonic_increasing_range_a_l3322_332238

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 12*x - 1

-- State the theorem
theorem monotonic_increasing_range_a :
  (∀ x y : ℝ, x < y → f a x < f a y) → a ∈ Set.Icc (-6) 6 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_range_a_l3322_332238
