import Mathlib

namespace NUMINAMATH_CALUDE_zero_in_interval_l372_37260

noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.log x / Real.log 2

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 4, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l372_37260


namespace NUMINAMATH_CALUDE_complement_union_M_N_l372_37298

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2}
def N : Set Nat := {3, 4}

theorem complement_union_M_N : (U \ (M ∪ N)) = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l372_37298


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l372_37286

/-- Represents a bag of marbles with two colors -/
structure Bag where
  color1 : ℕ
  color2 : ℕ

/-- Calculate the probability of drawing a specific color from a bag -/
def probColor (bag : Bag) (color : ℕ) : ℚ :=
  color / (bag.color1 + bag.color2)

/-- The probability of drawing a yellow marble as the second marble -/
def probYellowSecond (bagX bagY bagZ : Bag) : ℚ :=
  probColor bagX bagX.color1 * probColor bagY bagY.color1 +
  probColor bagX bagX.color2 * probColor bagZ bagZ.color1

theorem yellow_marble_probability :
  let bagX : Bag := ⟨4, 5⟩  -- 4 white, 5 black
  let bagY : Bag := ⟨7, 3⟩  -- 7 yellow, 3 blue
  let bagZ : Bag := ⟨3, 6⟩  -- 3 yellow, 6 blue
  probYellowSecond bagX bagY bagZ = 67 / 135 := by
  sorry


end NUMINAMATH_CALUDE_yellow_marble_probability_l372_37286


namespace NUMINAMATH_CALUDE_new_person_weight_bus_weight_problem_l372_37297

theorem new_person_weight (initial_count : ℕ) (initial_average : ℝ) (weight_decrease : ℝ) : ℝ :=
  let total_weight := initial_count * initial_average
  let new_count := initial_count + 1
  let new_average := initial_average - weight_decrease
  let new_total_weight := new_count * new_average
  new_total_weight - total_weight

theorem bus_weight_problem :
  new_person_weight 30 102 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_bus_weight_problem_l372_37297


namespace NUMINAMATH_CALUDE_sum_of_three_different_digits_is_18_l372_37211

/-- Represents a non-zero digit (1-9) -/
def NonZeroDigit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The sum of three different non-zero digits is 18 -/
theorem sum_of_three_different_digits_is_18 :
  ∃ (a b c : NonZeroDigit), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a.val + b.val + c.val = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_different_digits_is_18_l372_37211


namespace NUMINAMATH_CALUDE_total_vegetables_l372_37231

def garden_vegetables (potatoes cucumbers peppers : ℕ) : Prop :=
  (cucumbers = potatoes - 60) ∧
  (peppers = 2 * cucumbers) ∧
  (potatoes + cucumbers + peppers = 768)

theorem total_vegetables : ∃ (cucumbers peppers : ℕ), 
  garden_vegetables 237 cucumbers peppers := by
  sorry

end NUMINAMATH_CALUDE_total_vegetables_l372_37231


namespace NUMINAMATH_CALUDE_misread_weight_calculation_l372_37240

theorem misread_weight_calculation (n : ℕ) (initial_avg correct_avg correct_weight : ℝ) :
  n = 20 ∧ 
  initial_avg = 58.4 ∧ 
  correct_avg = 58.6 ∧ 
  correct_weight = 60 →
  ∃ misread_weight : ℝ, 
    misread_weight = 56 ∧
    n * correct_avg - n * initial_avg = correct_weight - misread_weight :=
by sorry

end NUMINAMATH_CALUDE_misread_weight_calculation_l372_37240


namespace NUMINAMATH_CALUDE_fraction_equality_l372_37261

theorem fraction_equality (x y : ℝ) (h : x / y = 2) : (x - y) / x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l372_37261


namespace NUMINAMATH_CALUDE_base5_123_to_base10_l372_37253

/-- Converts a base-5 number to base-10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 5 * acc + d) 0

/-- The base-5 representation of 123 --/
def base5_123 : List Nat := [1, 2, 3]

theorem base5_123_to_base10 :
  base5ToBase10 base5_123 = 38 := by
  sorry

end NUMINAMATH_CALUDE_base5_123_to_base10_l372_37253


namespace NUMINAMATH_CALUDE_jessica_purchases_total_cost_l372_37285

/-- The total cost of Jessica's purchases is $41.44 -/
theorem jessica_purchases_total_cost :
  let cat_toy := 10.22
  let cage := 11.73
  let cat_food := 8.15
  let collar := 4.35
  let litter_box := 6.99
  cat_toy + cage + cat_food + collar + litter_box = 41.44 := by
  sorry

end NUMINAMATH_CALUDE_jessica_purchases_total_cost_l372_37285


namespace NUMINAMATH_CALUDE_non_defective_products_percentage_l372_37270

/-- Represents a machine in the factory -/
structure Machine where
  production_percentage : ℝ
  defective_percentage : ℝ

/-- The factory setup -/
def factory : List Machine := [
  ⟨0.25, 0.02⟩,  -- m1
  ⟨0.35, 0.04⟩,  -- m2
  ⟨0.40, 0.05⟩   -- m3
]

/-- Calculate the percentage of non-defective products -/
def non_defective_percentage (machines : List Machine) : ℝ :=
  1 - (machines.map (λ m => m.production_percentage * m.defective_percentage)).sum

/-- Theorem stating the percentage of non-defective products -/
theorem non_defective_products_percentage :
  non_defective_percentage factory = 0.961 := by
  sorry

#eval non_defective_percentage factory

end NUMINAMATH_CALUDE_non_defective_products_percentage_l372_37270


namespace NUMINAMATH_CALUDE_first_year_interest_l372_37216

theorem first_year_interest (initial_deposit : ℝ) (first_year_balance : ℝ) 
  (second_year_increase : ℝ) (total_increase : ℝ) :
  initial_deposit = 500 →
  first_year_balance = 600 →
  second_year_increase = 0.1 →
  total_increase = 0.32 →
  first_year_balance - initial_deposit = 100 := by
  sorry

end NUMINAMATH_CALUDE_first_year_interest_l372_37216


namespace NUMINAMATH_CALUDE_train_passenger_count_l372_37223

/-- Calculates the total number of passengers transported by a train -/
theorem train_passenger_count (one_way : ℕ) (return_way : ℕ) (additional_trips : ℕ) : 
  one_way = 100 → return_way = 60 → additional_trips = 3 → 
  (one_way + return_way) * (additional_trips + 1) = 640 := by
  sorry

end NUMINAMATH_CALUDE_train_passenger_count_l372_37223


namespace NUMINAMATH_CALUDE_arc_RS_range_l372_37291

/-- An isosceles triangle with a rolling circle -/
structure RollingCircleTriangle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The altitude of the isosceles triangle -/
  altitude : ℝ
  /-- The radius of the rolling circle -/
  radius : ℝ
  /-- The position of the tangent point P along the base (0 ≤ p ≤ base) -/
  p : ℝ
  /-- The triangle is isosceles -/
  isosceles : altitude = base / 2
  /-- The altitude is twice the radius -/
  altitude_radius : altitude = 2 * radius
  /-- The tangent point is on the base -/
  p_on_base : 0 ≤ p ∧ p ≤ base

/-- The arc RS of the rolling circle -/
def arc_RS (t : RollingCircleTriangle) : ℝ := sorry

/-- Theorem: The arc RS varies from 90° to 180° -/
theorem arc_RS_range (t : RollingCircleTriangle) : 
  90 ≤ arc_RS t ∧ arc_RS t ≤ 180 := by sorry

end NUMINAMATH_CALUDE_arc_RS_range_l372_37291


namespace NUMINAMATH_CALUDE_trigonometric_identity_l372_37263

theorem trigonometric_identity : 
  (Real.sqrt 3) / (Real.cos (10 * π / 180)) - 1 / (Real.sin (170 * π / 180)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l372_37263


namespace NUMINAMATH_CALUDE_line_intersection_with_circle_l372_37209

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define a line with slope 1
def line_with_slope_1 (x y m : ℝ) : Prop := y = x + m

-- Define a point on the circle
def point_on_circle (x y : ℝ) : Prop := circle_C x y

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define a chord of the circle
def chord (A B : ℝ × ℝ) : Prop :=
  point_on_circle A.1 A.2 ∧ point_on_circle B.1 B.2

-- Define a circle passing through three points
def circle_through_points (A B O : ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2 ∧
    (O.1 - center.1)^2 + (O.2 - center.2)^2 = radius^2

-- Theorem statement
theorem line_intersection_with_circle :
  ∃ (m : ℝ), m = 1 ∨ m = -4 ∧
  ∀ (x y : ℝ),
    line_with_slope_1 x y m →
    (∃ (A B : ℝ × ℝ),
      chord A B ∧
      line_with_slope_1 A.1 A.2 m ∧
      line_with_slope_1 B.1 B.2 m ∧
      circle_through_points A B origin) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_with_circle_l372_37209


namespace NUMINAMATH_CALUDE_sin_integral_minus_two_to_two_l372_37206

theorem sin_integral_minus_two_to_two : ∫ x in (-2)..2, Real.sin x = 0 := by sorry

end NUMINAMATH_CALUDE_sin_integral_minus_two_to_two_l372_37206


namespace NUMINAMATH_CALUDE_gift_card_value_l372_37237

/-- Represents the gift card problem -/
theorem gift_card_value (cost_per_pound : ℝ) (pounds_bought : ℝ) (remaining_balance : ℝ) :
  cost_per_pound = 8.58 →
  pounds_bought = 4.0 →
  remaining_balance = 35.68 →
  cost_per_pound * pounds_bought + remaining_balance = 70.00 := by
  sorry


end NUMINAMATH_CALUDE_gift_card_value_l372_37237


namespace NUMINAMATH_CALUDE_march_production_3000_l372_37239

/-- Represents the number of months since March -/
def months_since_march : Nat → Nat
  | 0 => 0  -- March
  | 1 => 1  -- April
  | 2 => 2  -- May
  | 3 => 3  -- June
  | 4 => 4  -- July
  | n + 5 => n + 5

/-- Calculates the mask production for a given month based on the initial production in March -/
def mask_production (initial_production : Nat) (month : Nat) : Nat :=
  initial_production * (2 ^ (months_since_march month))

theorem march_production_3000 :
  ∃ (initial_production : Nat),
    mask_production initial_production 4 = 48000 ∧
    initial_production = 3000 := by
  sorry

end NUMINAMATH_CALUDE_march_production_3000_l372_37239


namespace NUMINAMATH_CALUDE_expression_factorization_l372_37282

theorem expression_factorization (x : ℝ) : 
  (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10) = 5 * x * (5 * x^2 + 19) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l372_37282


namespace NUMINAMATH_CALUDE_coin_collection_value_l372_37267

theorem coin_collection_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℚ) :
  total_coins = 15 →
  sample_coins = 5 →
  sample_value = 12 →
  (total_coins : ℚ) * (sample_value / sample_coins) = 36 :=
by sorry

end NUMINAMATH_CALUDE_coin_collection_value_l372_37267


namespace NUMINAMATH_CALUDE_greatest_multiple_of_five_cubed_less_than_3375_l372_37220

theorem greatest_multiple_of_five_cubed_less_than_3375 :
  ∃ (x : ℕ), x > 0 ∧ 5 ∣ x ∧ x^3 < 3375 ∧ ∀ (y : ℕ), y > 0 → 5 ∣ y → y^3 < 3375 → y ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_five_cubed_less_than_3375_l372_37220


namespace NUMINAMATH_CALUDE_balloons_in_park_l372_37236

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 4

/-- The total number of balloons Allan and Jake have in the park -/
def total_balloons : ℕ := allan_balloons + jake_balloons

theorem balloons_in_park : total_balloons = 6 := by
  sorry

end NUMINAMATH_CALUDE_balloons_in_park_l372_37236


namespace NUMINAMATH_CALUDE_expression_evaluation_l372_37272

theorem expression_evaluation :
  let x : ℝ := 0
  let expr := (1 + 1 / (x - 2)) / ((x^2 - 2*x + 1) / (x - 2))
  expr = -1 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l372_37272


namespace NUMINAMATH_CALUDE_function_properties_monotonicity_condition_l372_37228

/-- The function f(x) = ax³ + bx² -/
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem function_properties (a b : ℝ) :
  (f a b 1 = 4) ∧ 
  (f_derivative a b 1 * 1 = -9) →
  (a = 1 ∧ b = 3) :=
sorry

theorem monotonicity_condition (m : ℝ) :
  (∀ x ∈ Set.Icc m (m + 1), f_derivative 1 3 x > 0) →
  (m ≥ 0 ∨ m ≤ -3) :=
sorry

end NUMINAMATH_CALUDE_function_properties_monotonicity_condition_l372_37228


namespace NUMINAMATH_CALUDE_no_valid_base_l372_37201

theorem no_valid_base : ¬ ∃ (base : ℝ), (1/5)^35 * (1/4)^18 = 1/(2*(base^35)) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_base_l372_37201


namespace NUMINAMATH_CALUDE_binary_11011011_equals_base4_3123_l372_37275

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 4) :: aux (m / 4)
    aux n |>.reverse

theorem binary_11011011_equals_base4_3123 :
  decimal_to_base4 (binary_to_decimal [true, true, false, true, true, false, true, true]) = [3, 1, 2, 3] := by
  sorry

#eval binary_to_decimal [true, true, false, true, true, false, true, true]
#eval decimal_to_base4 (binary_to_decimal [true, true, false, true, true, false, true, true])

end NUMINAMATH_CALUDE_binary_11011011_equals_base4_3123_l372_37275


namespace NUMINAMATH_CALUDE_robin_oatmeal_cookies_l372_37288

/-- Calculates the number of oatmeal cookies Robin had -/
def oatmeal_cookies (cookies_per_bag : ℕ) (chocolate_chip_cookies : ℕ) (baggies : ℕ) : ℕ :=
  cookies_per_bag * baggies - chocolate_chip_cookies

/-- Proves that Robin had 25 oatmeal cookies -/
theorem robin_oatmeal_cookies :
  oatmeal_cookies 6 23 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_robin_oatmeal_cookies_l372_37288


namespace NUMINAMATH_CALUDE_odd_function_domain_symmetry_l372_37204

/-- A function f is odd if its domain is symmetric about the origin -/
def is_odd_function (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x ∈ domain, -x ∈ domain

/-- The domain of the function -/
def function_domain (t : ℝ) : Set ℝ := Set.Ioo t (2*t + 3)

/-- Theorem: If f is an odd function with domain (t, 2t+3), then t = -1 -/
theorem odd_function_domain_symmetry (f : ℝ → ℝ) (t : ℝ) 
  (h : is_odd_function f (function_domain t)) : 
  t = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_domain_symmetry_l372_37204


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l372_37203

theorem real_roots_of_polynomial (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l372_37203


namespace NUMINAMATH_CALUDE_gcd_and_sum_of_1729_and_867_l372_37215

theorem gcd_and_sum_of_1729_and_867 :
  let a : ℕ := 1729
  let b : ℕ := 867
  (Nat.gcd a b = 1) ∧ (a + b = 2596) := by
  sorry

end NUMINAMATH_CALUDE_gcd_and_sum_of_1729_and_867_l372_37215


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l372_37295

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 / 3) = 9 := by sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l372_37295


namespace NUMINAMATH_CALUDE_problem_solution_l372_37218

-- Define the set B
def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1 : ℝ) 2, x^2 - 2*x - m ≤ 0}

-- Define the set A(a)
def A (a : ℝ) : Set ℝ := {x | (x - 2*a) * (x - (a + 1)) ≤ 0}

theorem problem_solution :
  (B = Set.Ici 3) ∧
  ({a : ℝ | A a ⊆ B ∧ A a ≠ B} = Set.Ici 2) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l372_37218


namespace NUMINAMATH_CALUDE_equilateral_triangles_count_l372_37265

/-- The number of points evenly spaced on a circle -/
def n : ℕ := 900

/-- The number of points needed to form an equilateral triangle on the circle -/
def equilateral_spacing : ℕ := n / 3

/-- The number of equilateral triangles with all vertices on the circle -/
def all_vertices_on_circle : ℕ := equilateral_spacing

/-- The number of ways to choose 2 points from n points -/
def choose_two : ℕ := n * (n - 1) / 2

/-- The total number of equilateral triangles with at least two vertices from the n points -/
def total_triangles : ℕ := 2 * choose_two - all_vertices_on_circle

theorem equilateral_triangles_count : total_triangles = 808800 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangles_count_l372_37265


namespace NUMINAMATH_CALUDE_permutations_theorem_l372_37205

-- Define the number of books
def n : ℕ := 30

-- Define the function to calculate the number of permutations where two specific objects are not adjacent
def permutations_not_adjacent (n : ℕ) : ℕ := 28 * Nat.factorial (n - 1)

-- Theorem statement
theorem permutations_theorem :
  permutations_not_adjacent n = (n - 2) * Nat.factorial (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_permutations_theorem_l372_37205


namespace NUMINAMATH_CALUDE_balloon_arrangements_l372_37254

def word_length : ℕ := 7
def repeating_letters : ℕ := 2
def repetitions_per_letter : ℕ := 2

theorem balloon_arrangements :
  (word_length.factorial) / ((repetitions_per_letter.factorial) ^ repeating_letters) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l372_37254


namespace NUMINAMATH_CALUDE_balloon_difference_l372_37283

theorem balloon_difference (allan_balloons jake_balloons : ℕ) 
  (h1 : allan_balloons = 5) 
  (h2 : jake_balloons = 3) : 
  allan_balloons - jake_balloons = 2 := by
sorry

end NUMINAMATH_CALUDE_balloon_difference_l372_37283


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l372_37245

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sequence_is_arithmetic (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a (n + 1) - a n = 3) : 
  is_arithmetic_sequence a :=
sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l372_37245


namespace NUMINAMATH_CALUDE_sqrt_identity_in_range_l372_37292

theorem sqrt_identity_in_range (θ : Real) (h : θ ∈ Set.Ioo (7 * Real.pi / 4) (2 * Real.pi)) :
  Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ) = Real.cos θ - Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sqrt_identity_in_range_l372_37292


namespace NUMINAMATH_CALUDE_quadratic_vertex_x_coordinate_l372_37248

/-- Given a quadratic function f(x) = ax^2 + bx + c that passes through
    the points (2, 5), (8, 5), and (9, 11), prove that the x-coordinate
    of its vertex is 5. -/
theorem quadratic_vertex_x_coordinate
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^2 + b * x + c)
  (h_point1 : f 2 = 5)
  (h_point2 : f 8 = 5)
  (h_point3 : f 9 = 11) :
  ∃ (vertex_x : ℝ), vertex_x = 5 ∧ ∀ x, f x ≤ f vertex_x :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_x_coordinate_l372_37248


namespace NUMINAMATH_CALUDE_batsman_new_average_is_38_l372_37243

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  totalRuns : ℕ
  innings : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (stats : BatsmanStats) (newInningScore : ℕ) : ℚ :=
  (stats.totalRuns + newInningScore) / (stats.innings + 1)

/-- Theorem: Given the conditions, the batsman's new average is 38 -/
theorem batsman_new_average_is_38 
  (stats : BatsmanStats)
  (h1 : stats.innings = 16)
  (h2 : newAverage stats 86 = stats.average + 3) :
  newAverage stats 86 = 38 := by
sorry

end NUMINAMATH_CALUDE_batsman_new_average_is_38_l372_37243


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l372_37266

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (m+2)*x + m
  -- The equation always has two distinct real roots
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  -- When the sum condition is satisfied, m = 3
  (x₁ + x₂ + 2*x₁*x₂ = 1 → m = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l372_37266


namespace NUMINAMATH_CALUDE_intersection_angle_l372_37278

/-- A regular hexagonal pyramid with lateral faces at 45° to the base -/
structure RegularHexagonalPyramid :=
  (base : Set (ℝ × ℝ))
  (apex : ℝ × ℝ × ℝ)
  (lateral_angle : Real)
  (is_regular : Bool)
  (lateral_angle_eq : lateral_angle = Real.pi / 4)

/-- A plane intersecting the pyramid -/
structure IntersectingPlane :=
  (base_edge : Set (ℝ × ℝ))
  (intersections : Set (ℝ × ℝ × ℝ))
  (is_parallel : Bool)

/-- The theorem to be proved -/
theorem intersection_angle (p : RegularHexagonalPyramid) (s : IntersectingPlane) :
  p.is_regular ∧ s.is_parallel →
  ∃ α : Real, α = Real.arctan (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_angle_l372_37278


namespace NUMINAMATH_CALUDE_network_connections_l372_37258

/-- The number of unique connections in a network of switches -/
def num_connections (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a network of 30 switches, where each switch is directly connected 
    to exactly 4 other switches, the total number of unique connections is 60 -/
theorem network_connections : num_connections 30 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l372_37258


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l372_37271

theorem ceiling_floor_difference : ⌈(15 / 8) * (-34 / 4)⌉ - ⌊(15 / 8) * ⌊-34 / 4⌋⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l372_37271


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l372_37213

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Nitrogen atoms in the compound -/
def num_N : ℕ := 2

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := num_N * atomic_weight_N + num_O * atomic_weight_O

theorem compound_molecular_weight : molecular_weight = 76.02 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l372_37213


namespace NUMINAMATH_CALUDE_track_team_size_l372_37276

/-- The length of the relay race in meters -/
def relay_length : ℕ := 150

/-- The distance each team member runs in meters -/
def individual_distance : ℕ := 30

/-- The number of people on the track team -/
def team_size : ℕ := relay_length / individual_distance

theorem track_team_size : team_size = 5 := by
  sorry

end NUMINAMATH_CALUDE_track_team_size_l372_37276


namespace NUMINAMATH_CALUDE_fixed_point_and_bisecting_line_l372_37290

/-- The line equation ax - y + 2 + a = 0 -/
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 + a = 0

/-- The line equation 4x + y + 3 = 0 -/
def line_l1 (x y : ℝ) : Prop := 4 * x + y + 3 = 0

/-- The line equation 3x - 5y - 5 = 0 -/
def line_l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 5 = 0

/-- The fixed point P -/
def point_P : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + 1 = 0 -/
def line_m (x y : ℝ) : Prop := 3 * x + y + 1 = 0

theorem fixed_point_and_bisecting_line :
  (∀ a : ℝ, line_l a (point_P.1) (point_P.2)) ∧
  (∀ x y : ℝ, line_m x y ↔ 
    (∃ t : ℝ, line_l1 (point_P.1 - t) (point_P.2 - t) ∧
              line_l2 (point_P.1 + t) (point_P.2 + t))) :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_and_bisecting_line_l372_37290


namespace NUMINAMATH_CALUDE_fraction_sum_l372_37212

theorem fraction_sum : (1 : ℚ) / 6 + (2 : ℚ) / 9 + (1 : ℚ) / 3 = (13 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l372_37212


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l372_37229

theorem geometric_arithmetic_sequence (x y z : ℝ) 
  (h1 : (4 * y)^2 = (3 * x) * (5 * z))  -- Geometric sequence condition
  (h2 : 2 / y = 1 / x + 1 / z)          -- Arithmetic sequence condition
  : x / z + z / x = 34 / 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l372_37229


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l372_37246

theorem absolute_value_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : abs a > abs b := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l372_37246


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l372_37289

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to its decimal representation -/
def TwoDigitNumber.toNat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

/-- The sum of digits of a TwoDigitNumber -/
def TwoDigitNumber.digitSum (n : TwoDigitNumber) : Nat :=
  n.tens + n.ones

/-- The product of digits of a TwoDigitNumber -/
def TwoDigitNumber.digitProduct (n : TwoDigitNumber) : Nat :=
  n.tens * n.ones

theorem unique_two_digit_number :
  ∃! (n : TwoDigitNumber),
    (n.toNat / n.digitSum = 4 ∧ n.toNat % n.digitSum = 3) ∧
    (n.toNat / n.digitProduct = 3 ∧ n.toNat % n.digitProduct = 5) ∧
    n.toNat = 23 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l372_37289


namespace NUMINAMATH_CALUDE_square_root_equality_l372_37224

theorem square_root_equality (k : ℕ) (h : k > 0) :
  (∀ (i : ℕ), i > 0 → Real.sqrt (i + i / (i^2 - 1)) = i * Real.sqrt (i / (i^2 - 1))) →
  (let n : ℝ := 6
   let m : ℝ := 35
   Real.sqrt (6 + n / m) = 6 * Real.sqrt (n / m) ∧ m + n = 41) := by
sorry

end NUMINAMATH_CALUDE_square_root_equality_l372_37224


namespace NUMINAMATH_CALUDE_estate_division_l372_37284

theorem estate_division (total_estate : ℚ) : 
  total_estate > 0 → 
  ∃ (son_share mother_share daughter_share : ℚ),
    son_share = (4 : ℚ) / 7 * total_estate ∧
    mother_share = (2 : ℚ) / 7 * total_estate ∧
    daughter_share = (1 : ℚ) / 7 * total_estate ∧
    son_share + mother_share + daughter_share = total_estate ∧
    son_share = 2 * mother_share ∧
    mother_share = 2 * daughter_share :=
by sorry

end NUMINAMATH_CALUDE_estate_division_l372_37284


namespace NUMINAMATH_CALUDE_at_least_one_real_root_l372_37234

theorem at_least_one_real_root (m : ℝ) : 
  ∃ x : ℝ, (x^2 - 5*x + m = 0) ∨ (2*x^2 + x + 6 - m = 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_real_root_l372_37234


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_sums_l372_37287

-- Define the polynomial
def p (x : ℝ) : ℝ := 10 * x^3 + 101 * x + 210

-- Define the roots
def roots_of_p (a b c : ℝ) : Prop := p a = 0 ∧ p b = 0 ∧ p c = 0

-- Theorem statement
theorem sum_of_cubes_of_sums (a b c : ℝ) :
  roots_of_p a b c → (a + b)^3 + (b + c)^3 + (c + a)^3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_sums_l372_37287


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l372_37251

/-- Given two quadratic functions f and g, if f has two distinct real roots,
    then g must have at least one real root. -/
theorem quadratic_roots_relation (a b c : ℝ) (h : a * c ≠ 0) :
  let f := fun x : ℝ ↦ a * x^2 + b * x + c
  let g := fun x : ℝ ↦ c * x^2 + b * x + a
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ z : ℝ, g z = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l372_37251


namespace NUMINAMATH_CALUDE_sams_cans_sams_final_can_count_l372_37252

/-- Sam's can collection problem -/
theorem sams_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) 
  (bags_given_away : ℕ) (large_cans_found : ℕ) : ℕ :=
  let total_bags := saturday_bags + sunday_bags
  let total_cans := total_bags * cans_per_bag
  let cans_given_away := bags_given_away * cans_per_bag
  let remaining_cans := total_cans - cans_given_away
  let large_cans_equivalent := large_cans_found * 2
  remaining_cans + large_cans_equivalent

/-- Proof of Sam's final can count -/
theorem sams_final_can_count : sams_cans 3 4 9 2 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_sams_cans_sams_final_can_count_l372_37252


namespace NUMINAMATH_CALUDE_complex_equation_solution_l372_37247

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := (1 + i) * z = 2 * i

-- Theorem statement
theorem complex_equation_solution :
  ∃ (z : ℂ), equation z ∧ z = 1 + i :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l372_37247


namespace NUMINAMATH_CALUDE_probability_at_least_one_three_l372_37210

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of at least one die showing a 3 when two fair dice are rolled -/
def prob_at_least_one_three : ℚ := 15 / 64

/-- Theorem stating that the probability of at least one die showing a 3
    when two fair 8-sided dice are rolled is 15/64 -/
theorem probability_at_least_one_three :
  prob_at_least_one_three = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_three_l372_37210


namespace NUMINAMATH_CALUDE_train_length_l372_37273

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 12 → ∃ length : ℝ, 
  (abs (length - 200.04) < 0.01) ∧ (length = speed * (1000 / 3600) * time) := by
  sorry

end NUMINAMATH_CALUDE_train_length_l372_37273


namespace NUMINAMATH_CALUDE_quadratic_roots_opposite_l372_37262

theorem quadratic_roots_opposite (k : ℝ) : 
  (∃ x y : ℝ, x^2 + (k-2)*x - 1 = 0 ∧ y^2 + (k-2)*y - 1 = 0 ∧ x = -y) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_opposite_l372_37262


namespace NUMINAMATH_CALUDE_train_speed_l372_37255

/-- Calculate the speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length time : ℝ) (length_positive : length > 0) (time_positive : time > 0) :
  length = 100 ∧ time = 5 → length / time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l372_37255


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l372_37244

theorem polygon_interior_angles (n : ℕ) (extra_angle : ℝ) : 
  (n ≥ 3) →
  (180 * (n - 2) + extra_angle = 1800) →
  (n = 11 ∧ extra_angle = 180) :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l372_37244


namespace NUMINAMATH_CALUDE_log_equation_solution_l372_37238

theorem log_equation_solution (x : ℝ) :
  (Real.log x / Real.log 2) + (Real.log x / Real.log 8) = 5 →
  x = 2^(15/4) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l372_37238


namespace NUMINAMATH_CALUDE_beong_gun_number_l372_37226

theorem beong_gun_number : ∃ x : ℚ, (x / 11 + 156 = 178) ∧ (x = 242) := by
  sorry

end NUMINAMATH_CALUDE_beong_gun_number_l372_37226


namespace NUMINAMATH_CALUDE_absolute_value_fraction_less_than_one_l372_37294

theorem absolute_value_fraction_less_than_one (a b : ℝ) 
  (ha : |a| < 1) (hb : |b| < 1) : 
  |((a + b) / (1 + a * b))| < 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_less_than_one_l372_37294


namespace NUMINAMATH_CALUDE_star_calculation_l372_37221

/-- The ⋆ operation for real numbers -/
def star (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Theorem stating that 2 ⋆ (3 ⋆ (4 ⋆ 6)) = -152877 -/
theorem star_calculation : star 2 (star 3 (star 4 6)) = -152877 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l372_37221


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l372_37222

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 15, 360, and 125 -/
def product : ℕ := 15 * 360 * 125

theorem product_trailing_zeros : trailingZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l372_37222


namespace NUMINAMATH_CALUDE_quarterly_interest_rate_proof_l372_37241

/-- Proves that the given annual interest payment for a loan with quarterly compounding
    is consistent with the calculated quarterly interest rate. -/
theorem quarterly_interest_rate_proof
  (principal : ℝ)
  (annual_interest : ℝ)
  (quarterly_rate : ℝ)
  (h_principal : principal = 10000)
  (h_annual_interest : annual_interest = 2155.06)
  (h_quarterly_rate : quarterly_rate = 0.05) :
  annual_interest = principal * ((1 + quarterly_rate) ^ 4 - 1) :=
by sorry

end NUMINAMATH_CALUDE_quarterly_interest_rate_proof_l372_37241


namespace NUMINAMATH_CALUDE_size_relationship_l372_37293

theorem size_relationship (a b c : ℝ) 
  (ha : a = 4^(1/2 : ℝ)) 
  (hb : b = 2^(1/3 : ℝ)) 
  (hc : c = 5^(1/2 : ℝ)) : 
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_size_relationship_l372_37293


namespace NUMINAMATH_CALUDE_triangle_inequality_last_three_terms_l372_37296

/-- An arithmetic sequence with positive terms and positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = a n + d

/-- Triangle inequality for the last three terms of a four-term arithmetic sequence -/
theorem triangle_inequality_last_three_terms
  (a : ℕ → ℝ) (d : ℝ) (h : ArithmeticSequence a d) :
  a 2 + a 3 > a 4 ∧ a 2 + a 4 > a 3 ∧ a 3 + a 4 > a 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_last_three_terms_l372_37296


namespace NUMINAMATH_CALUDE_min_distance_squared_l372_37235

/-- Given real numbers a, b, c, and d satisfying certain conditions,
    the minimum value of (a-c)² + (b-d)² is 1. -/
theorem min_distance_squared (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) : 
  ∃ (min : ℝ), min = 1 ∧ ∀ (a' b' c' d' : ℝ), 
    Real.log (b' + 1) + a' - 3 * b' = 0 → 
    2 * d' - c' + Real.sqrt 5 = 0 → 
    (a' - c')^2 + (b' - d')^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l372_37235


namespace NUMINAMATH_CALUDE_corner_with_same_color_l372_37280

/-- Definition of a "corner" figure -/
def Corner (square : Fin 2017 → Fin 2017 → Fin 120) : Prop :=
  ∃ (i j : Fin 2017) (dir : Bool),
    let horizontal := if dir then (fun k => square i (j + k)) else (fun k => square (i + k) j)
    let vertical := if dir then (fun k => square (i + k) j) else (fun k => square i (j + k))
    (∀ k : Fin 10, horizontal k ∈ Set.range horizontal) ∧
    (∀ k : Fin 10, vertical k ∈ Set.range vertical) ∧
    (square i j ∈ Set.range horizontal ∪ Set.range vertical)

/-- The main theorem -/
theorem corner_with_same_color (square : Fin 2017 → Fin 2017 → Fin 120) :
  ∃ (corner : Corner square), 
    ∃ (c1 c2 : Fin 2017 × Fin 2017), c1 ≠ c2 ∧ 
      square c1.1 c1.2 = square c2.1 c2.2 :=
sorry

end NUMINAMATH_CALUDE_corner_with_same_color_l372_37280


namespace NUMINAMATH_CALUDE_greatest_b_value_l372_37242

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 14 ≥ 0 → x ≤ 7) ∧ 
  (-7^2 + 9*7 - 14 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_greatest_b_value_l372_37242


namespace NUMINAMATH_CALUDE_green_disks_count_l372_37256

theorem green_disks_count (total : ℕ) (red green blue : ℕ) : 
  total = 14 →
  red = 2 * green →
  blue = green / 2 →
  total = red + green + blue →
  green = 4 := by
  sorry

end NUMINAMATH_CALUDE_green_disks_count_l372_37256


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l372_37259

/-- A rectangular prism with distinctly different dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_ne_width : length ≠ width
  length_ne_height : length ≠ height
  width_ne_height : width ≠ height

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 12

/-- Theorem: A rectangular prism with distinctly different dimensions has 12 pairs of parallel edges -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  parallel_edge_pairs prism = 12 := by sorry

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l372_37259


namespace NUMINAMATH_CALUDE_least_common_denominator_l372_37279

theorem least_common_denominator : 
  let denominators := [3, 4, 5, 6, 8, 9, 10]
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 3 4) 5) 6) 8) 9) 10 = 360 :=
by sorry

end NUMINAMATH_CALUDE_least_common_denominator_l372_37279


namespace NUMINAMATH_CALUDE_roots_on_circle_l372_37200

theorem roots_on_circle (z : ℂ) : 
  (z + 1)^4 = 16 * z^4 → Complex.abs (z - Complex.ofReal (1/3)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_roots_on_circle_l372_37200


namespace NUMINAMATH_CALUDE_mabel_marbles_l372_37299

/-- Given information about marbles of Amanda, Katrina, and Mabel -/
def marble_problem (amanda katrina mabel : ℕ) : Prop :=
  (amanda + 12 = 2 * katrina) ∧
  (mabel = 5 * katrina) ∧
  (mabel = amanda + 63)

/-- Theorem stating that under the given conditions, Mabel has 85 marbles -/
theorem mabel_marbles :
  ∀ amanda katrina mabel : ℕ,
  marble_problem amanda katrina mabel →
  mabel = 85 := by
  sorry

end NUMINAMATH_CALUDE_mabel_marbles_l372_37299


namespace NUMINAMATH_CALUDE_three_quantities_change_l372_37214

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

-- Define the line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle type
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the problem setup
def setup (A B P Q : Point) (l1 l2 lPQ : Line) : Prop :=
  (P.x = (A.x + B.x) / 2) ∧ 
  (P.y = (A.y + B.y) / 2) ∧
  (l1.a = lPQ.a ∧ l1.b = lPQ.b) ∧
  (l2.a = lPQ.a ∧ l2.b = lPQ.b)

-- Define the four quantities
def lengthAB (A B : Point) : ℝ := sorry
def perimeterAPB (A B P : Point) : ℝ := sorry
def areaAPB (A B P : Point) : ℝ := sorry
def distancePtoAB (A B P : Point) : ℝ := sorry

-- Define a function that counts how many quantities change
def countChangingQuantities (A B P Q : Point) (l1 l2 lPQ : Line) : ℕ := sorry

-- The main theorem
theorem three_quantities_change 
  (A B P Q : Point) (l1 l2 lPQ : Line) 
  (h : setup A B P Q l1 l2 lPQ) : 
  countChangingQuantities A B P Q l1 l2 lPQ = 3 := sorry

end NUMINAMATH_CALUDE_three_quantities_change_l372_37214


namespace NUMINAMATH_CALUDE_cubic_root_sum_l372_37264

theorem cubic_root_sum (k₁ k₂ : ℝ) (h : k₁ + k₂ ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ x^3 - k₁*x - k₂
  let roots := { x : ℝ | f x = 0 }
  ∃ (a b c : ℝ), a ∈ roots ∧ b ∈ roots ∧ c ∈ roots ∧
    (1+a)/(1-a) + (1+b)/(1-b) + (1+c)/(1-c) = (3 + k₁ + 3*k₂) / (1 - k₁ - k₂) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l372_37264


namespace NUMINAMATH_CALUDE_min_races_correct_l372_37202

/-- Represents a race strategy to find the top 3 fastest horses -/
structure RaceStrategy where
  numRaces : ℕ
  ensuresTop3 : Bool

/-- The minimum number of races needed to find the top 3 fastest horses -/
def minRaces : ℕ := 7

/-- The total number of horses -/
def totalHorses : ℕ := 25

/-- The maximum number of horses that can race together -/
def maxHorsesPerRace : ℕ := 5

/-- Predicate to check if a race strategy is valid -/
def isValidStrategy (s : RaceStrategy) : Prop :=
  s.numRaces ≥ minRaces ∧ s.ensuresTop3

/-- Theorem stating that the minimum number of races is correct -/
theorem min_races_correct :
  ∀ s : RaceStrategy,
    isValidStrategy s →
    s.numRaces ≥ minRaces :=
sorry

end NUMINAMATH_CALUDE_min_races_correct_l372_37202


namespace NUMINAMATH_CALUDE_problem_statement_l372_37217

theorem problem_statement (a b c : ℝ) (h : (2:ℝ)^a = (3:ℝ)^b ∧ (3:ℝ)^b = (18:ℝ)^c ∧ (18:ℝ)^c < 1) :
  b < 2*c ∧ (a + b)/c > 3 + 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l372_37217


namespace NUMINAMATH_CALUDE_double_reflection_of_D_l372_37249

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 1)  -- Translate down by 1
  let reflected := (-p'.2, -p'.1)  -- Reflect across y = -x
  (reflected.1, reflected.2 + 1)  -- Translate up by 1

def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_line (reflect_x p)

theorem double_reflection_of_D :
  double_reflection (4, 1) = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_of_D_l372_37249


namespace NUMINAMATH_CALUDE_hannahs_speed_l372_37274

/-- Proves that Hannah's speed is 15 km/h given the problem conditions --/
theorem hannahs_speed (glen_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h_glen_speed : glen_speed = 37)
  (h_distance : distance = 130)
  (h_time : time = 5) :
  ∃ hannah_speed : ℝ, hannah_speed = 15 ∧ 
  2 * distance = (glen_speed + hannah_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_hannahs_speed_l372_37274


namespace NUMINAMATH_CALUDE_money_distribution_l372_37269

theorem money_distribution (total : ℕ) (p q r : ℕ) : 
  total = 9000 →
  p + q + r = total →
  r = 2 * (p + q) / 3 →
  r = 3600 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l372_37269


namespace NUMINAMATH_CALUDE_emily_score_calculation_l372_37257

/-- Emily's trivia game score calculation -/
theorem emily_score_calculation 
  (first_round : ℕ) 
  (second_round : ℕ) 
  (final_score : ℕ) 
  (h1 : first_round = 16) 
  (h2 : second_round = 33) 
  (h3 : final_score = 1) : 
  (first_round + second_round) - final_score = 48 := by
  sorry

end NUMINAMATH_CALUDE_emily_score_calculation_l372_37257


namespace NUMINAMATH_CALUDE_guard_distance_proof_l372_37225

/-- Calculates the total distance walked by a guard around a rectangular warehouse -/
def total_distance_walked (length width : ℕ) (total_circles skipped_circles : ℕ) : ℕ :=
  2 * (length + width) * (total_circles - skipped_circles)

/-- Proves that the guard walks 16000 feet given the specific conditions -/
theorem guard_distance_proof :
  total_distance_walked 600 400 10 2 = 16000 := by
  sorry

end NUMINAMATH_CALUDE_guard_distance_proof_l372_37225


namespace NUMINAMATH_CALUDE_slices_per_pizza_large_pizza_has_12_slices_l372_37230

/-- Calculates the number of slices in a large pizza based on soccer team statistics -/
theorem slices_per_pizza (num_pizzas : ℕ) (num_games : ℕ) (avg_goals_per_game : ℕ) : ℕ :=
  let total_goals := num_games * avg_goals_per_game
  let total_slices := total_goals
  total_slices / num_pizzas

/-- Proves that a large pizza has 12 slices given the problem conditions -/
theorem large_pizza_has_12_slices :
  slices_per_pizza 6 8 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_slices_per_pizza_large_pizza_has_12_slices_l372_37230


namespace NUMINAMATH_CALUDE_win_sector_area_l372_37219

/-- Given a circular spinner with radius 8 cm and a probability of winning of 1/4,
    the area of the WIN sector is 16π square centimeters. -/
theorem win_sector_area (radius : ℝ) (win_probability : ℝ) (win_sector_area : ℝ) : 
  radius = 8 →
  win_probability = 1 / 4 →
  win_sector_area = win_probability * π * radius^2 →
  win_sector_area = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l372_37219


namespace NUMINAMATH_CALUDE_geometric_series_sum_times_four_fifths_l372_37250

theorem geometric_series_sum_times_four_fifths :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 5
  let S := (a * (1 - r^n)) / (1 - r)
  (S * 4/5 : ℚ) = 21/80 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_times_four_fifths_l372_37250


namespace NUMINAMATH_CALUDE_ladder_length_proof_l372_37233

/-- The length of a ladder leaning against a wall. -/
def ladder_length : ℝ := 18.027756377319946

/-- The initial distance of the ladder's bottom from the wall. -/
def initial_bottom_distance : ℝ := 6

/-- The distance the ladder's bottom moves when the top slips. -/
def bottom_slip_distance : ℝ := 12.480564970698127

/-- The distance the ladder's top slips down the wall. -/
def top_slip_distance : ℝ := 4

/-- Theorem stating the length of the ladder given the conditions. -/
theorem ladder_length_proof : 
  ∃ (initial_height : ℝ),
    ladder_length ^ 2 = initial_height ^ 2 + initial_bottom_distance ^ 2 ∧
    ladder_length ^ 2 = (initial_height - top_slip_distance) ^ 2 + bottom_slip_distance ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_ladder_length_proof_l372_37233


namespace NUMINAMATH_CALUDE_set_operations_l372_37268

def U : Set ℤ := {1, 2, 3, 4, 5, 6, 7, 8}

def A : Set ℤ := {x | x^2 - 3*x + 2 = 0}

def B : Set ℤ := {x | 1 ≤ x ∧ x ≤ 5}

def C : Set ℤ := {x | 2 < x ∧ x < 9}

theorem set_operations :
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5}) ∧
  ((U.compl ∩ B) ∪ (U.compl ∩ C) = {1, 2, 6, 7, 8}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l372_37268


namespace NUMINAMATH_CALUDE_range_of_m_l372_37208

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x = x^2 - 4*x - 6) →
  (Set.range f = Set.Icc (-10) (-6)) →
  m ∈ Set.Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l372_37208


namespace NUMINAMATH_CALUDE_system_solution_l372_37281

theorem system_solution : 
  let x : ℚ := -29/9
  let y : ℚ := -113/27
  (7 * x = -10 - 3 * y) ∧ (4 * x = 6 * y - 38) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l372_37281


namespace NUMINAMATH_CALUDE_fertilizer_prices_l372_37227

/-- Represents the price per ton of fertilizer A -/
def price_A : ℝ := sorry

/-- Represents the price per ton of fertilizer B -/
def price_B : ℝ := sorry

/-- The price difference between fertilizer A and B is $100 -/
axiom price_difference : price_A = price_B + 100

/-- The total cost of 2 tons of fertilizer A and 1 ton of fertilizer B is $1700 -/
axiom total_cost : 2 * price_A + price_B = 1700

theorem fertilizer_prices :
  price_A = 600 ∧ price_B = 500 := by sorry

end NUMINAMATH_CALUDE_fertilizer_prices_l372_37227


namespace NUMINAMATH_CALUDE_negation_equivalence_l372_37232

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Icc (0 : ℝ) 1, x^3 + x^2 > 1) ↔
  (∀ x ∈ Set.Icc (0 : ℝ) 1, x^3 + x^2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l372_37232


namespace NUMINAMATH_CALUDE_cistern_filling_fraction_l372_37207

/-- Given a pipe that can fill a cistern in 55 minutes, 
    this theorem proves that the fraction of the cistern 
    filled in 5 minutes is 1/11. -/
theorem cistern_filling_fraction 
  (total_time : ℕ) 
  (filling_time : ℕ) 
  (h1 : total_time = 55) 
  (h2 : filling_time = 5) : 
  (filling_time : ℚ) / total_time = 1 / 11 :=
by sorry

end NUMINAMATH_CALUDE_cistern_filling_fraction_l372_37207


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l372_37277

theorem quadratic_root_sum (m n : ℝ) : 
  (∀ x, m * x^2 - n * x - 2023 = 0 → x = -1 ∨ x ≠ -1) →
  (m * (-1)^2 - n * (-1) - 2023 = 0) →
  m + n = 2023 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l372_37277
