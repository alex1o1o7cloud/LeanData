import Mathlib

namespace NUMINAMATH_CALUDE_length_of_AB_is_seven_l3534_353458

-- Define the triangle ABC
structure TriangleABC where
  A : Point
  B : Point
  C : Point

-- Define the triangle CBD
structure TriangleCBD where
  C : Point
  B : Point
  D : Point

-- Define the properties of the triangles
def isIsosceles (t : TriangleABC) : Prop := sorry
def isEquilateral (t : TriangleABC) : Prop := sorry
def isIsoscelesCBD (t : TriangleCBD) : Prop := sorry
def perimeterCBD (t : TriangleCBD) : ℝ := sorry
def perimeterABC (t : TriangleABC) : ℝ := sorry
def lengthBD (t : TriangleCBD) : ℝ := sorry
def lengthAB (t : TriangleABC) : ℝ := sorry

theorem length_of_AB_is_seven 
  (abc : TriangleABC) 
  (cbd : TriangleCBD) 
  (h1 : isIsosceles abc)
  (h2 : isEquilateral abc)
  (h3 : isIsoscelesCBD cbd)
  (h4 : perimeterCBD cbd = 24)
  (h5 : perimeterABC abc = 21)
  (h6 : lengthBD cbd = 10) :
  lengthAB abc = 7 := by sorry

end NUMINAMATH_CALUDE_length_of_AB_is_seven_l3534_353458


namespace NUMINAMATH_CALUDE_divisor_problem_l3534_353492

theorem divisor_problem (original : Nat) (subtracted : Nat) (remaining : Nat) :
  original = 165826 →
  subtracted = 2 →
  remaining = original - subtracted →
  (∃ (d : Nat), d > 1 ∧ remaining % d = 0 ∧ ∀ (k : Nat), k > d → remaining % k ≠ 0) →
  (∃ (d : Nat), d = 2 ∧ remaining % d = 0 ∧ ∀ (k : Nat), k > d → remaining % k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l3534_353492


namespace NUMINAMATH_CALUDE_complex_operations_l3534_353496

theorem complex_operations (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 2 - 3 * Complex.I) 
  (h₂ : z₂ = (15 - 5 * Complex.I) / (2 + Complex.I)^2) : 
  z₁ * z₂ = -7 - 9 * Complex.I ∧ 
  z₁ / z₂ = 11/10 + 3/10 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_operations_l3534_353496


namespace NUMINAMATH_CALUDE_square_plus_product_equals_twelve_plus_two_sqrt_six_l3534_353436

theorem square_plus_product_equals_twelve_plus_two_sqrt_six :
  ∀ a b : ℝ,
  a = Real.sqrt 6 + 1 →
  b = Real.sqrt 6 - 1 →
  a^2 + a*b = 12 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_product_equals_twelve_plus_two_sqrt_six_l3534_353436


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3534_353419

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3534_353419


namespace NUMINAMATH_CALUDE_f_at_three_l3534_353485

/-- The polynomial function f(x) = 9x^4 + 7x^3 - 5x^2 + 3x - 6 -/
def f (x : ℝ) : ℝ := 9 * x^4 + 7 * x^3 - 5 * x^2 + 3 * x - 6

/-- Theorem stating that f(3) = 876 -/
theorem f_at_three : f 3 = 876 := by
  sorry

end NUMINAMATH_CALUDE_f_at_three_l3534_353485


namespace NUMINAMATH_CALUDE_function_root_implies_a_range_l3534_353497

theorem function_root_implies_a_range (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x - 1 = 0) → a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_function_root_implies_a_range_l3534_353497


namespace NUMINAMATH_CALUDE_arthur_muffins_l3534_353403

theorem arthur_muffins (initial_muffins : ℕ) : 
  initial_muffins + 48 = 83 → initial_muffins = 35 := by
  sorry

end NUMINAMATH_CALUDE_arthur_muffins_l3534_353403


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3534_353452

/-- A geometric sequence with specified terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_3 : a 3 = -4) 
  (h_7 : a 7 = -16) : 
  a 5 = -8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3534_353452


namespace NUMINAMATH_CALUDE_reflection_result_l3534_353411

/-- Reflects a point over the y-axis -/
def reflectOverYAxis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflectOverXAxis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The final position of point F after two reflections -/
def finalPosition (F : ℝ × ℝ) : ℝ × ℝ :=
  reflectOverXAxis (reflectOverYAxis F)

theorem reflection_result :
  finalPosition (-1, -1) = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_reflection_result_l3534_353411


namespace NUMINAMATH_CALUDE_new_ratio_after_addition_l3534_353407

theorem new_ratio_after_addition (a b c : ℤ) : 
  (3 * a = b) → 
  (b = 15) → 
  (c = a + 10) → 
  (c = b) := by sorry

end NUMINAMATH_CALUDE_new_ratio_after_addition_l3534_353407


namespace NUMINAMATH_CALUDE_gcd_102_238_l3534_353467

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l3534_353467


namespace NUMINAMATH_CALUDE_circle_line_distance_l3534_353425

theorem circle_line_distance (a : ℝ) : 
  let circle : ℝ × ℝ → Prop := λ p => (p.1^2 + p.2^2 - 2*p.1 - 4*p.2 = 0)
  let center : ℝ × ℝ := (1, 2)
  let line : ℝ × ℝ → Prop := λ p => (p.1 - p.2 + a = 0)
  let distance := |1 - 2 + a| / Real.sqrt 2
  (∀ p, circle p ↔ (p.1 - 1)^2 + (p.2 - 2)^2 = 5) →
  (distance = Real.sqrt 2) →
  (a = 3 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_l3534_353425


namespace NUMINAMATH_CALUDE_friend_walking_speed_difference_l3534_353417

theorem friend_walking_speed_difference 
  (total_distance : ℝ) 
  (p_distance : ℝ) 
  (hp : total_distance = 22) 
  (hpd : p_distance = 12) : 
  let q_distance := total_distance - p_distance
  let rate_ratio := p_distance / q_distance
  (rate_ratio - 1) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_friend_walking_speed_difference_l3534_353417


namespace NUMINAMATH_CALUDE_ratio_of_amounts_l3534_353449

theorem ratio_of_amounts (total_amount : ℕ) (r_amount : ℕ) 
  (h1 : total_amount = 8000)
  (h2 : r_amount = 3200) :
  r_amount / (total_amount - r_amount) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_amounts_l3534_353449


namespace NUMINAMATH_CALUDE_draw_with_replacement_l3534_353414

-- Define the number of balls in the bin
def num_balls : ℕ := 15

-- Define the number of draws
def num_draws : ℕ := 4

-- Define the function to calculate the number of ways to draw balls
def ways_to_draw (n : ℕ) (k : ℕ) : ℕ := n ^ k

-- Theorem statement
theorem draw_with_replacement :
  ways_to_draw num_balls num_draws = 50625 := by
  sorry

end NUMINAMATH_CALUDE_draw_with_replacement_l3534_353414


namespace NUMINAMATH_CALUDE_largest_number_l3534_353494

theorem largest_number : ∀ (a b c d : ℝ), 
  a = -3 → b = 0 → c = Real.sqrt 5 → d = 2 → 
  c ≥ a ∧ c ≥ b ∧ c ≥ d := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3534_353494


namespace NUMINAMATH_CALUDE_expand_product_l3534_353400

theorem expand_product (x : ℝ) : -3 * (2 * x + 4) * (x - 7) = -6 * x^2 + 30 * x + 84 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3534_353400


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3534_353493

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 48) →
  (a 2 + a 5 + a 8 = 40) →
  (a 3 + a 6 + a 9 = 32) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3534_353493


namespace NUMINAMATH_CALUDE_caravan_keepers_caravan_keepers_proof_l3534_353477

theorem caravan_keepers : ℕ → Prop :=
  fun k =>
    let hens : ℕ := 50
    let goats : ℕ := 45
    let camels : ℕ := 8
    let total_heads : ℕ := hens + goats + camels + k
    let total_feet : ℕ := hens * 2 + goats * 4 + camels * 4 + k * 2
    total_feet = total_heads + 224 → k = 15

-- The proof goes here
theorem caravan_keepers_proof : ∃ k : ℕ, caravan_keepers k :=
  sorry

end NUMINAMATH_CALUDE_caravan_keepers_caravan_keepers_proof_l3534_353477


namespace NUMINAMATH_CALUDE_amy_balloon_count_l3534_353471

/-- The number of balloons James has -/
def james_balloons : ℕ := 1222

/-- The difference between James' and Amy's balloon counts -/
def difference : ℕ := 208

/-- Amy's balloon count -/
def amy_balloons : ℕ := james_balloons - difference

theorem amy_balloon_count : amy_balloons = 1014 := by
  sorry

end NUMINAMATH_CALUDE_amy_balloon_count_l3534_353471


namespace NUMINAMATH_CALUDE_total_soccer_games_l3534_353438

def soccer_games_this_year : ℕ := 11
def soccer_games_last_year : ℕ := 13
def soccer_games_next_year : ℕ := 15

theorem total_soccer_games :
  soccer_games_this_year + soccer_games_last_year + soccer_games_next_year = 39 := by
  sorry

end NUMINAMATH_CALUDE_total_soccer_games_l3534_353438


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l3534_353435

theorem square_sum_given_sum_and_product (a b : ℝ) : a + b = 8 → a * b = -2 → a^2 + b^2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l3534_353435


namespace NUMINAMATH_CALUDE_cosine_product_sqrt_l3534_353437

theorem cosine_product_sqrt (π : Real) : 
  Real.sqrt ((3 - Real.cos (π / 9)^2) * (3 - Real.cos (2 * π / 9)^2) * (3 - Real.cos (4 * π / 9)^2)) = 39 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_sqrt_l3534_353437


namespace NUMINAMATH_CALUDE_count_integers_eq_25_l3534_353406

/-- The number of integers between 100 and 200 (exclusive) that have the same remainder when divided by 6 and 8 -/
def count_integers : ℕ :=
  (Finset.filter (λ n : ℕ => 
    100 < n ∧ n < 200 ∧ n % 6 = n % 8
  ) (Finset.range 200)).card

/-- Theorem stating that there are exactly 25 such integers -/
theorem count_integers_eq_25 : count_integers = 25 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_eq_25_l3534_353406


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eight_sqrt_two_l3534_353415

theorem sqrt_sum_equals_eight_sqrt_two : 
  Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eight_sqrt_two_l3534_353415


namespace NUMINAMATH_CALUDE_millet_cost_is_60_cents_l3534_353491

/-- Represents the cost of millet seed per pound -/
def millet_cost : ℝ := sorry

/-- The total weight of millet seed in pounds -/
def millet_weight : ℝ := 100

/-- The cost of sunflower seeds per pound -/
def sunflower_cost : ℝ := 1.10

/-- The total weight of sunflower seeds in pounds -/
def sunflower_weight : ℝ := 25

/-- The desired cost per pound of the mixture -/
def mixture_cost_per_pound : ℝ := 0.70

/-- The total weight of the mixture -/
def total_weight : ℝ := millet_weight + sunflower_weight

/-- Theorem stating that the cost of millet seed per pound is $0.60 -/
theorem millet_cost_is_60_cents :
  millet_cost = 0.60 :=
by
  sorry

end NUMINAMATH_CALUDE_millet_cost_is_60_cents_l3534_353491


namespace NUMINAMATH_CALUDE_polygon_with_150_degree_interior_angles_has_12_sides_l3534_353499

theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    interior_angle = 150 →
    (n : ℝ) * (180 - interior_angle) = 360 →
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_150_degree_interior_angles_has_12_sides_l3534_353499


namespace NUMINAMATH_CALUDE_area_bisector_l3534_353429

/-- A polygon in the xy-plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- The polygon described in the problem -/
def problemPolygon : Polygon :=
  { vertices := [(0, 0), (0, 4), (4, 4), (4, 2), (6, 2), (6, 0)] }

/-- Calculate the area of a polygon -/
def area (p : Polygon) : ℝ := sorry

/-- Calculate the area of a polygon on one side of a line y = mx passing through the origin -/
def areaOneSide (p : Polygon) (m : ℝ) : ℝ := sorry

/-- The main theorem -/
theorem area_bisector (p : Polygon) :
  p = problemPolygon →
  areaOneSide p (5/3) = (area p) / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_bisector_l3534_353429


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l3534_353487

theorem tan_double_angle_special_case (θ : Real) :
  (∃ (x y : Real), y = (1/2) * x ∧ x ≥ 0 ∧ y ≥ 0 ∧ Real.tan θ = y / x) →
  Real.tan (2 * θ) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l3534_353487


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3534_353448

/-- A right triangle with perimeter 40, area 30, and one angle of 45 degrees has a hypotenuse of length 2√30 -/
theorem right_triangle_hypotenuse (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  a + b + c = 40 →   -- Perimeter is 40
  a * b / 2 = 30 →   -- Area is 30
  a = b →            -- One angle is 45 degrees, so adjacent sides are equal
  c = 2 * Real.sqrt 30 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3534_353448


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3534_353445

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem necessary_but_not_sufficient (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b : E, a - 2 • b = 0 → ‖a - b‖ = ‖b‖) ∧
  (∃ a b : E, ‖a - b‖ = ‖b‖ ∧ a - 2 • b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3534_353445


namespace NUMINAMATH_CALUDE_janice_earnings_l3534_353495

/-- Calculates the total earnings for a week given specific working conditions --/
def calculate_earnings (weekday_hours : ℕ) (weekend_hours : ℕ) (holiday_hours : ℕ) : ℕ :=
  let weekday_rate := 10
  let weekend_rate := 12
  let holiday_rate := 2 * weekend_rate
  let weekday_earnings := weekday_hours * weekday_rate
  let weekend_earnings := weekend_hours * weekend_rate
  let holiday_earnings := holiday_hours * holiday_rate
  weekday_earnings + weekend_earnings + holiday_earnings

/-- Theorem stating that Janice's earnings for the given week are $720 --/
theorem janice_earnings : calculate_earnings 30 25 5 = 720 := by
  sorry

end NUMINAMATH_CALUDE_janice_earnings_l3534_353495


namespace NUMINAMATH_CALUDE_pencil_count_l3534_353426

/-- The total number of pencils after adding more to an initial amount -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: Given 41 initial pencils and 30 added pencils, the total is 71 -/
theorem pencil_count : total_pencils 41 30 = 71 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3534_353426


namespace NUMINAMATH_CALUDE_system_solution_no_solution_l3534_353421

-- Problem 1
theorem system_solution (x y : ℝ) :
  x - y = 8 ∧ 3*x + y = 12 → x = 5 ∧ y = -3 := by sorry

-- Problem 2
theorem no_solution (x : ℝ) :
  x ≠ 1 → 3 / (x - 1) - (x + 2) / (x * (x - 1)) ≠ 0 := by sorry

end NUMINAMATH_CALUDE_system_solution_no_solution_l3534_353421


namespace NUMINAMATH_CALUDE_distance_not_equal_addition_l3534_353420

theorem distance_not_equal_addition : ∀ (a b : ℤ), 
  a = -3 → b = 10 → (abs (b - a) ≠ -3 + 10) :=
by
  sorry

end NUMINAMATH_CALUDE_distance_not_equal_addition_l3534_353420


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3534_353402

theorem fraction_decomposition (A B : ℚ) : 
  (∀ x : ℚ, x ≠ -1 ∧ x ≠ 2/3 → (7*x - 18) / (3*x^2 - 5*x - 2) = A / (x + 1) + B / (3*x - 2)) → 
  A = -4/7 ∧ B = 61/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3534_353402


namespace NUMINAMATH_CALUDE_janice_age_l3534_353481

theorem janice_age (current_year : ℕ) (mark_birth_year : ℕ) (graham_age_difference : ℕ) :
  current_year = 2021 →
  mark_birth_year = 1976 →
  graham_age_difference = 3 →
  (current_year - mark_birth_year - graham_age_difference) / 2 = 21 :=
by sorry

end NUMINAMATH_CALUDE_janice_age_l3534_353481


namespace NUMINAMATH_CALUDE_sqrt_two_minus_one_power_l3534_353443

theorem sqrt_two_minus_one_power (n : ℕ+) :
  ∃ m : ℕ+, (Real.sqrt 2 - 1) ^ n.val = Real.sqrt m.val - Real.sqrt (m.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_one_power_l3534_353443


namespace NUMINAMATH_CALUDE_amelia_win_probability_l3534_353447

/-- Represents a player in the coin-tossing game -/
inductive Player
  | Amelia
  | Blaine
  | Calvin

/-- The probability of getting heads for each player -/
def headsProbability (p : Player) : ℚ :=
  match p with
  | Player.Amelia => 1/4
  | Player.Blaine => 1/3
  | Player.Calvin => 1/2

/-- The order of players in the game -/
def playerOrder : List Player := [Player.Amelia, Player.Blaine, Player.Calvin]

/-- The probability of Amelia winning the game -/
def ameliaWinProbability : ℚ := 1/3

/-- Theorem stating that Amelia's probability of winning is 1/3 -/
theorem amelia_win_probability : ameliaWinProbability = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_amelia_win_probability_l3534_353447


namespace NUMINAMATH_CALUDE_recipe_cups_needed_l3534_353432

theorem recipe_cups_needed (servings : ℝ) (cups_per_serving : ℝ) 
  (h1 : servings = 18.0) 
  (h2 : cups_per_serving = 2.0) : 
  servings * cups_per_serving = 36.0 := by
  sorry

end NUMINAMATH_CALUDE_recipe_cups_needed_l3534_353432


namespace NUMINAMATH_CALUDE_increasing_function_parameter_range_l3534_353489

/-- Given a function f(x) = -1/3 * x^3 + 1/2 * x^2 + 2ax, 
    if f(x) is increasing on the interval (2/3, +∞), 
    then a ∈ (-1/9, +∞) -/
theorem increasing_function_parameter_range (a : ℝ) : 
  (∀ x > 2/3, (deriv (fun x => -1/3 * x^3 + 1/2 * x^2 + 2*a*x) x) > 0) →
  a > -1/9 :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_parameter_range_l3534_353489


namespace NUMINAMATH_CALUDE_probability_heart_then_diamond_l3534_353410

/-- The probability of drawing a heart first and a diamond second from a standard deck of cards -/
theorem probability_heart_then_diamond (total_cards : ℕ) (suits : ℕ) (cards_per_suit : ℕ) 
  (h_total : total_cards = 52)
  (h_suits : suits = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_deck : total_cards = suits * cards_per_suit) :
  (cards_per_suit : ℚ) / total_cards * cards_per_suit / (total_cards - 1) = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_probability_heart_then_diamond_l3534_353410


namespace NUMINAMATH_CALUDE_gigi_mushrooms_l3534_353462

/-- The number of pieces each mushroom is cut into -/
def pieces_per_mushroom : ℕ := 4

/-- The number of mushroom pieces Kenny used -/
def kenny_pieces : ℕ := 38

/-- The number of mushroom pieces Karla used -/
def karla_pieces : ℕ := 42

/-- The number of mushroom pieces left on the cutting board -/
def remaining_pieces : ℕ := 8

/-- Theorem stating that the total number of whole mushrooms GiGi cut up is 22 -/
theorem gigi_mushrooms :
  (kenny_pieces + karla_pieces + remaining_pieces) / pieces_per_mushroom = 22 := by
  sorry

end NUMINAMATH_CALUDE_gigi_mushrooms_l3534_353462


namespace NUMINAMATH_CALUDE_only_36_satisfies_conditions_l3534_353480

/-- A two-digit integer is represented by 10a + b, where a and b are single digits -/
def TwoDigitInteger (a b : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

/-- The sum of digits of a two-digit integer -/
def SumOfDigits (a b : ℕ) : ℕ := a + b

/-- Twice the product of digits of a two-digit integer -/
def TwiceProductOfDigits (a b : ℕ) : ℕ := 2 * a * b

/-- The value of a two-digit integer -/
def IntegerValue (a b : ℕ) : ℕ := 10 * a + b

theorem only_36_satisfies_conditions :
  ∀ a b : ℕ,
    TwoDigitInteger a b →
    (IntegerValue a b % SumOfDigits a b = 0 ∧
     IntegerValue a b % TwiceProductOfDigits a b = 0) →
    IntegerValue a b = 36 :=
by sorry

end NUMINAMATH_CALUDE_only_36_satisfies_conditions_l3534_353480


namespace NUMINAMATH_CALUDE_miley_purchase_cost_l3534_353431

/-- Calculates the total cost of Miley's purchase including discounts and sales tax -/
def total_cost (cellphone_price earbuds_price case_price : ℝ)
               (cellphone_discount earbuds_discount case_discount sales_tax : ℝ) : ℝ :=
  let cellphone_total := 2 * cellphone_price * (1 - cellphone_discount)
  let earbuds_total := 2 * earbuds_price * (1 - earbuds_discount)
  let case_total := 2 * case_price * (1 - case_discount)
  let subtotal := cellphone_total + earbuds_total + case_total
  subtotal * (1 + sales_tax)

/-- Theorem stating that the total cost of Miley's purchase is $2006.64 -/
theorem miley_purchase_cost :
  total_cost 800 150 40 0.05 0.10 0.15 0.08 = 2006.64 := by
  sorry

end NUMINAMATH_CALUDE_miley_purchase_cost_l3534_353431


namespace NUMINAMATH_CALUDE_ticket_distribution_proof_l3534_353405

theorem ticket_distribution_proof (total_tickets : ℕ) (total_amount : ℚ) 
  (price_15 price_10 price_5_5 : ℚ) :
  total_tickets = 22 →
  total_amount = 229 →
  price_15 = 15 →
  price_10 = 10 →
  price_5_5 = (11 : ℚ) / 2 →
  ∃! (x y z : ℕ), 
    x + y + z = total_tickets ∧ 
    price_15 * x + price_10 * y + price_5_5 * z = total_amount ∧
    x = 9 ∧ y = 5 ∧ z = 8 := by
  sorry

end NUMINAMATH_CALUDE_ticket_distribution_proof_l3534_353405


namespace NUMINAMATH_CALUDE_computer_price_calculation_l3534_353490

theorem computer_price_calculation (P : ℝ) : 
  (P * 1.2 * 0.9 * 1.3 = 351) → P = 250 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_calculation_l3534_353490


namespace NUMINAMATH_CALUDE_oil_consumption_ranking_l3534_353424

/-- Oil consumption per person for each region -/
structure OilConsumption where
  west : ℝ
  nonWest : ℝ
  russia : ℝ

/-- The ranking of oil consumption is correct if Russia > Non-West > West -/
def correctRanking (consumption : OilConsumption) : Prop :=
  consumption.russia > consumption.nonWest ∧ consumption.nonWest > consumption.west

/-- Theorem stating that the given oil consumption data results in the correct ranking -/
theorem oil_consumption_ranking (consumption : OilConsumption) 
  (h_west : consumption.west = 55.084)
  (h_nonWest : consumption.nonWest = 214.59)
  (h_russia : consumption.russia = 1038.33) :
  correctRanking consumption := by
  sorry

#check oil_consumption_ranking

end NUMINAMATH_CALUDE_oil_consumption_ranking_l3534_353424


namespace NUMINAMATH_CALUDE_cracker_difference_l3534_353470

theorem cracker_difference (marcus_crackers : ℕ) (nicholas_crackers : ℕ) : 
  marcus_crackers = 27 →
  nicholas_crackers = 15 →
  ∃ (mona_crackers : ℕ), 
    marcus_crackers = 3 * mona_crackers ∧
    nicholas_crackers = mona_crackers + 6 :=
by
  sorry

end NUMINAMATH_CALUDE_cracker_difference_l3534_353470


namespace NUMINAMATH_CALUDE_solution_set_proof_l3534_353427

theorem solution_set_proof (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h0 : f 0 = 2) (h1 : ∀ x : ℝ, f x + (deriv f) x > 1) :
  {x : ℝ | Real.exp x * f x > Real.exp x + 1} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_proof_l3534_353427


namespace NUMINAMATH_CALUDE_first_row_chairs_l3534_353453

/-- Given a sequence of chair counts in rows, prove that the first row has 14 chairs. -/
theorem first_row_chairs (chairs : ℕ → ℕ) : 
  chairs 2 = 23 →                    -- Second row has 23 chairs
  (∀ n ≥ 2, chairs (n + 1) = chairs n + 9) →  -- Each subsequent row increases by 9
  chairs 6 = 59 →                    -- Sixth row has 59 chairs
  chairs 1 = 14 :=                   -- First row has 14 chairs
by sorry

end NUMINAMATH_CALUDE_first_row_chairs_l3534_353453


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3534_353454

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 8 → 
  (a + b + c) / 3 = a + 12 →
  (a + b + c) / 3 = c - 20 →
  a + b + c = 48 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3534_353454


namespace NUMINAMATH_CALUDE_correct_num_technicians_l3534_353434

/-- The number of technicians in a workshop. -/
def num_technicians : ℕ := 7

/-- The total number of workers in the workshop. -/
def total_workers : ℕ := 42

/-- The average salary of all workers in the workshop. -/
def avg_salary_all : ℕ := 8000

/-- The average salary of technicians in the workshop. -/
def avg_salary_technicians : ℕ := 18000

/-- The average salary of non-technician workers in the workshop. -/
def avg_salary_others : ℕ := 6000

/-- Theorem stating that the number of technicians is correct given the workshop conditions. -/
theorem correct_num_technicians :
  num_technicians = 7 ∧
  num_technicians + (total_workers - num_technicians) = total_workers ∧
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_others) / total_workers = avg_salary_all :=
by sorry

end NUMINAMATH_CALUDE_correct_num_technicians_l3534_353434


namespace NUMINAMATH_CALUDE_building_height_average_l3534_353428

def measurements : List ℝ := [79.4, 80.6, 80.8, 79.1, 80, 79.6, 80.5]

theorem building_height_average : 
  (measurements.sum / measurements.length : ℝ) = 80 := by sorry

end NUMINAMATH_CALUDE_building_height_average_l3534_353428


namespace NUMINAMATH_CALUDE_kids_wearing_shoes_l3534_353466

theorem kids_wearing_shoes (total : ℕ) (socks : ℕ) (both : ℕ) (barefoot : ℕ) 
  (h_total : total = 22)
  (h_socks : socks = 12)
  (h_both : both = 6)
  (h_barefoot : barefoot = 8) :
  total - barefoot - (socks - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_kids_wearing_shoes_l3534_353466


namespace NUMINAMATH_CALUDE_unique_total_prices_l3534_353479

def gift_prices : Finset ℕ := {2, 5, 8, 11, 14}
def box_prices : Finset ℕ := {3, 6, 9, 12, 15}

def total_prices : Finset ℕ := 
  Finset.image (λ (p : ℕ × ℕ) => p.1 + p.2) (gift_prices.product box_prices)

theorem unique_total_prices : Finset.card total_prices = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_total_prices_l3534_353479


namespace NUMINAMATH_CALUDE_find_P_l3534_353473

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set M
def M (P : ℝ) : Set ℕ := {x ∈ U | x^2 - 5*x + P = 0}

-- Define the complement of M in U
def complement_M (P : ℝ) : Set ℕ := U \ M P

-- Theorem statement
theorem find_P : ∃ P : ℝ, complement_M P = {2, 3} ∧ P = 4 := by sorry

end NUMINAMATH_CALUDE_find_P_l3534_353473


namespace NUMINAMATH_CALUDE_f_fixed_points_l3534_353484

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem f_fixed_points : {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_f_fixed_points_l3534_353484


namespace NUMINAMATH_CALUDE_probability_square_divisor_15_factorial_l3534_353476

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of positive integer divisors of n that are perfect squares -/
def num_square_divisors (n : ℕ) : ℕ := sorry

/-- The total number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Two natural numbers are coprime -/
def coprime (a b : ℕ) : Prop := sorry

theorem probability_square_divisor_15_factorial :
  ∃ m n : ℕ, 
    coprime m n ∧ 
    (m : ℚ) / n = (num_square_divisors (factorial 15) : ℚ) / (num_divisors (factorial 15)) ∧
    m = 1 ∧ n = 84 := by
  sorry

end NUMINAMATH_CALUDE_probability_square_divisor_15_factorial_l3534_353476


namespace NUMINAMATH_CALUDE_hyperbola_focus_l3534_353488

/-- The hyperbola equation -2x^2 + 3y^2 + 8x - 18y - 8 = 0 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  -2 * x^2 + 3 * y^2 + 8 * x - 18 * y - 8 = 0

/-- A point (x, y) is a focus of the hyperbola if it satisfies the focus condition -/
def is_focus (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (p q : ℝ), hyperbola_equation p q →
  (p - x)^2 + (q - y)^2 = ((p - 2)^2 / (2 * b^2) - (q - 3)^2 / (2 * a^2) + 1)^2 * (a^2 + b^2)

theorem hyperbola_focus :
  is_focus 2 7.5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l3534_353488


namespace NUMINAMATH_CALUDE_complex_simplification_l3534_353455

theorem complex_simplification :
  (4 - 3*Complex.I) - (7 + 5*Complex.I) + 2*(1 - 2*Complex.I) = -1 - 12*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l3534_353455


namespace NUMINAMATH_CALUDE_five_sixths_of_twelve_fifths_l3534_353472

theorem five_sixths_of_twelve_fifths : (5 / 6 : ℚ) * (12 / 5 : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_five_sixths_of_twelve_fifths_l3534_353472


namespace NUMINAMATH_CALUDE_work_completion_time_l3534_353430

theorem work_completion_time (man_time son_time : ℝ) (h1 : man_time = 6) (h2 : son_time = 6) :
  1 / (1 / man_time + 1 / son_time) = 3 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3534_353430


namespace NUMINAMATH_CALUDE_locus_of_circle_centers_l3534_353486

/-- Given a point O and a radius R, the locus of centers C of circles with radius R
    passing through O is a circle with center O and radius R. -/
theorem locus_of_circle_centers (O : ℝ × ℝ) (R : ℝ) :
  {C : ℝ × ℝ | ∃ P, dist P C = R ∧ P = O} = {C : ℝ × ℝ | dist C O = R} := by sorry

end NUMINAMATH_CALUDE_locus_of_circle_centers_l3534_353486


namespace NUMINAMATH_CALUDE_mean_of_four_numbers_l3534_353440

theorem mean_of_four_numbers (a b c d : ℚ) (h : a + b + c + d = 3/4) : 
  (a + b + c + d) / 4 = 3/16 := by
sorry

end NUMINAMATH_CALUDE_mean_of_four_numbers_l3534_353440


namespace NUMINAMATH_CALUDE_difference_of_squares_fifty_thirty_l3534_353475

theorem difference_of_squares_fifty_thirty : 50^2 - 30^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_fifty_thirty_l3534_353475


namespace NUMINAMATH_CALUDE_population_after_10_years_l3534_353444

/-- The population growth over a period of years -/
def population_growth (M : ℝ) (p : ℝ) (years : ℕ) : ℝ :=
  M * (1 + p) ^ years

/-- Theorem: The population after 10 years given initial population M and growth rate p -/
theorem population_after_10_years (M : ℝ) (p : ℝ) :
  population_growth M p 10 = M * (1 + p)^10 := by
  sorry

end NUMINAMATH_CALUDE_population_after_10_years_l3534_353444


namespace NUMINAMATH_CALUDE_fencing_cost_calculation_l3534_353460

/-- Represents a rectangular plot with given dimensions and fencing cost -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ

/-- Calculates the total cost of fencing for a rectangular plot -/
def total_fencing_cost (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth) * plot.fencing_cost_per_meter

/-- Theorem stating the total fencing cost for a specific rectangular plot -/
theorem fencing_cost_calculation :
  let plot : RectangularPlot :=
    { length := 58
    , breadth := 58 - 16
    , fencing_cost_per_meter := 26.5 }
  total_fencing_cost plot = 5300 := by
  sorry


end NUMINAMATH_CALUDE_fencing_cost_calculation_l3534_353460


namespace NUMINAMATH_CALUDE_scientific_notation_of_1050000_l3534_353478

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  norm : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_1050000 :
  toScientificNotation 1050000 = ScientificNotation.mk 1.05 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1050000_l3534_353478


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3534_353482

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - x*y = 0) :
  x + 2*y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ - x₀*y₀ = 0 ∧ x₀ + 2*y₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3534_353482


namespace NUMINAMATH_CALUDE_chord_equation_l3534_353498

structure Curve where
  equation : ℝ → ℝ × ℝ

structure Line where
  equation : ℝ × ℝ → Prop

def parabola : Curve :=
  { equation := λ t => (4 * t^2, 4 * t) }

def point_on_curve (c : Curve) (p : ℝ × ℝ) : Prop :=
  ∃ t, c.equation t = p

def perpendicular (l1 l2 : Line) : Prop :=
  sorry

def chord_length_product (c : Curve) (l : Line) (p : ℝ × ℝ) : ℝ :=
  sorry

theorem chord_equation (c : Curve) (ab cd : Line) (p : ℝ × ℝ) :
  c = parabola →
  point_on_curve c p →
  p = (2, 2) →
  perpendicular ab cd →
  chord_length_product c ab p = chord_length_product c cd p →
  (ab.equation = λ (x, y) => y = x) ∨ 
  (ab.equation = λ (x, y) => x + y = 4) :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l3534_353498


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3534_353464

/-- The length of the major axis of an ellipse formed by intersecting a right circular cylinder --/
def majorAxisLength (cylinderRadius : ℝ) (majorAxisLongerRatio : ℝ) : ℝ :=
  2 * cylinderRadius * (1 + majorAxisLongerRatio)

/-- Theorem stating the length of the major axis of the ellipse --/
theorem ellipse_major_axis_length :
  majorAxisLength 2 0.3 = 5.2 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3534_353464


namespace NUMINAMATH_CALUDE_weight_loss_program_result_l3534_353408

/-- Calculates the final weight after a weight loss program -/
def finalWeight (initialWeight : ℕ) (weeklyLoss1 weeklyLoss2 : ℕ) (weeks1 weeks2 : ℕ) : ℕ :=
  initialWeight - (weeklyLoss1 * weeks1 + weeklyLoss2 * weeks2)

/-- Proves that the weight loss program results in the correct final weight -/
theorem weight_loss_program_result :
  finalWeight 250 3 2 4 8 = 222 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_program_result_l3534_353408


namespace NUMINAMATH_CALUDE_trig_equation_solution_l3534_353416

noncomputable def solve_trig_equation (x : ℝ) : Prop :=
  (1 - Real.sin (2 * x) ≠ 0) ∧ 
  (1 - Real.tan x ≠ 0) ∧ 
  (Real.cos x ≠ 0) ∧
  ((1 + Real.sin (2 * x)) / (1 - Real.sin (2 * x)) + 
   2 * ((1 + Real.tan x) / (1 - Real.tan x)) - 3 = 0)

theorem trig_equation_solution :
  ∀ x : ℝ, solve_trig_equation x ↔ 
    (∃ k : ℤ, x = k * Real.pi) ∨
    (∃ n : ℤ, x = Real.arctan 2 + n * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l3534_353416


namespace NUMINAMATH_CALUDE_inverse_variation_result_l3534_353469

/-- Given that c² varies inversely with d⁴, this function represents their relationship -/
def inverse_relation (k : ℝ) (c d : ℝ) : Prop :=
  c^2 * d^4 = k

theorem inverse_variation_result (k : ℝ) :
  inverse_relation k 8 2 →
  inverse_relation k c 4 →
  c^2 = 4 := by
  sorry

#check inverse_variation_result

end NUMINAMATH_CALUDE_inverse_variation_result_l3534_353469


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3534_353456

theorem quadratic_inequality_condition (a : ℝ) (h : 0 ≤ a ∧ a < 4) :
  ∀ x : ℝ, a * x^2 - a * x + 1 > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3534_353456


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l3534_353474

theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) (s : ℝ) : 
  b₁ = 3 → 
  b₂ = b₁ * s → 
  b₃ = b₂ * s → 
  ∀ x : ℝ, 3 * b₂ + 7 * b₃ ≥ -18/7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l3534_353474


namespace NUMINAMATH_CALUDE_point_inside_circle_l3534_353457

theorem point_inside_circle (a : ℝ) : 
  (∃ (x y : ℝ), x = 2*a ∧ y = a - 1 ∧ x^2 + y^2 - 2*y - 4 < 0) ↔ 
  (-1/5 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_point_inside_circle_l3534_353457


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3534_353461

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + y = 12) 
  (eq2 : x + 3 * y = 16) : 
  10 * x^2 + 14 * x * y + 10 * y^2 = 422.5 := by sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3534_353461


namespace NUMINAMATH_CALUDE_stock_sale_cash_realization_l3534_353401

/-- The cash realized on selling a stock, given the brokerage rate and total amount including brokerage -/
theorem stock_sale_cash_realization (brokerage_rate : ℚ) (total_with_brokerage : ℚ) :
  brokerage_rate = 1 / 400 →
  total_with_brokerage = 106 →
  ∃ cash_realized : ℚ, cash_realized + cash_realized * brokerage_rate = total_with_brokerage ∧
                    cash_realized = 42400 / 401 := by
  sorry

end NUMINAMATH_CALUDE_stock_sale_cash_realization_l3534_353401


namespace NUMINAMATH_CALUDE_rectangle_area_l3534_353450

theorem rectangle_area (perimeter : ℝ) (length width : ℝ) (h1 : perimeter = 280) 
  (h2 : length / width = 5 / 2) (h3 : perimeter = 2 * (length + width)) 
  (h4 : width * Real.sqrt 2 = length / 2) : length * width = 4000 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3534_353450


namespace NUMINAMATH_CALUDE_fraction_addition_l3534_353418

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3534_353418


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l3534_353439

/-- Given a geometric sequence {a_n} where a₁ = x, a₂ = x-1, and a₃ = 2x-2,
    prove that the general term is a_n = -2^(n-1) -/
theorem geometric_sequence_general_term (x : ℝ) (a : ℕ → ℝ) (h1 : a 1 = x) (h2 : a 2 = x - 1) (h3 : a 3 = 2*x - 2) :
  ∀ n : ℕ, a n = -2^(n-1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l3534_353439


namespace NUMINAMATH_CALUDE_det_scale_l3534_353459

theorem det_scale (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 10 →
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = 90 := by
sorry

end NUMINAMATH_CALUDE_det_scale_l3534_353459


namespace NUMINAMATH_CALUDE_batteries_in_controllers_l3534_353446

def batteries_problem (total flashlights toys controllers : ℕ) : Prop :=
  total = 19 ∧ flashlights = 2 ∧ toys = 15 ∧ total = flashlights + toys + controllers

theorem batteries_in_controllers :
  ∀ total flashlights toys controllers : ℕ,
    batteries_problem total flashlights toys controllers →
    controllers = 2 :=
by sorry

end NUMINAMATH_CALUDE_batteries_in_controllers_l3534_353446


namespace NUMINAMATH_CALUDE_mike_oil_changes_l3534_353404

/-- Represents the time in minutes for various car maintenance tasks and Mike's work schedule --/
structure CarMaintenance where
  wash_time : Nat  -- Time to wash a car in minutes
  oil_change_time : Nat  -- Time to change oil in minutes
  tire_change_time : Nat  -- Time to change a set of tires in minutes
  total_work_time : Nat  -- Total work time in minutes
  cars_washed : Nat  -- Number of cars washed
  tire_sets_changed : Nat  -- Number of tire sets changed

/-- Calculates the number of cars Mike changed oil on given the car maintenance data --/
def calculate_oil_changes (data : CarMaintenance) : Nat :=
  let total_wash_time := data.wash_time * data.cars_washed
  let total_tire_change_time := data.tire_change_time * data.tire_sets_changed
  let remaining_time := data.total_work_time - total_wash_time - total_tire_change_time
  remaining_time / data.oil_change_time

/-- Theorem stating that given the problem conditions, Mike changed oil on 6 cars --/
theorem mike_oil_changes (data : CarMaintenance) 
  (h1 : data.wash_time = 10)
  (h2 : data.oil_change_time = 15)
  (h3 : data.tire_change_time = 30)
  (h4 : data.total_work_time = 4 * 60)
  (h5 : data.cars_washed = 9)
  (h6 : data.tire_sets_changed = 2) :
  calculate_oil_changes data = 6 := by
  sorry

end NUMINAMATH_CALUDE_mike_oil_changes_l3534_353404


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3534_353423

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x, (d * x + e)^2 + f = 4 * x^2 - 28 * x + 49) →
  d * e = -14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3534_353423


namespace NUMINAMATH_CALUDE_board_intersection_area_l3534_353451

/-- The area of intersection of two rectangular boards crossing at a 45-degree angle -/
theorem board_intersection_area (width1 width2 : ℝ) (angle : ℝ) :
  width1 = 5 →
  width2 = 7 →
  angle = 45 →
  (width1 * width2 : ℝ) = 35 :=
by sorry

end NUMINAMATH_CALUDE_board_intersection_area_l3534_353451


namespace NUMINAMATH_CALUDE_hill_climbing_speed_l3534_353413

/-- Proves that the average speed while climbing is 2.625 km/h given the conditions of the journey -/
theorem hill_climbing_speed 
  (uphill_time : ℝ) 
  (downhill_time : ℝ) 
  (total_average_speed : ℝ) 
  (h1 : uphill_time = 4)
  (h2 : downhill_time = 2)
  (h3 : total_average_speed = 3.5) : 
  (total_average_speed * (uphill_time + downhill_time)) / (2 * uphill_time) = 2.625 := by
  sorry

#eval (3.5 * (4 + 2)) / (2 * 4)  -- This should evaluate to 2.625

end NUMINAMATH_CALUDE_hill_climbing_speed_l3534_353413


namespace NUMINAMATH_CALUDE_alyssa_games_last_year_l3534_353468

/-- The number of soccer games Alyssa attended last year -/
def games_last_year : ℕ := sorry

/-- The number of soccer games Alyssa attended this year -/
def games_this_year : ℕ := 11

/-- The number of soccer games Alyssa plans to attend next year -/
def games_next_year : ℕ := 15

/-- The total number of soccer games Alyssa will have attended -/
def total_games : ℕ := 39

/-- Theorem stating that Alyssa attended 13 soccer games last year -/
theorem alyssa_games_last_year : 
  games_last_year + games_this_year + games_next_year = total_games ∧ 
  games_last_year = 13 := by sorry

end NUMINAMATH_CALUDE_alyssa_games_last_year_l3534_353468


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3534_353433

/-- The function f(x) = 2a^(x+1) - 3 has a fixed point at (-1, -1) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x + 1) - 3
  f (-1) = -1 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3534_353433


namespace NUMINAMATH_CALUDE_biker_bob_route_l3534_353412

theorem biker_bob_route (distance_AB : Real) (net_west : Real) (initial_north : Real) :
  distance_AB = 20.615528128088304 →
  net_west = 5 →
  initial_north = 15 →
  ∃ x : Real, 
    x ≥ 0 ∧ 
    (x + initial_north)^2 + net_west^2 = distance_AB^2 ∧ 
    abs (x - 5.021531) < 0.000001 := by
  sorry

#check biker_bob_route

end NUMINAMATH_CALUDE_biker_bob_route_l3534_353412


namespace NUMINAMATH_CALUDE_pizza_solution_l3534_353422

/-- Represents the number of slices in a pizza --/
structure PizzaSlices where
  small : ℕ
  large : ℕ

/-- Represents the number of pizzas purchased --/
structure PizzasPurchased where
  small : ℕ
  large : ℕ

/-- Represents the number of slices eaten by each person --/
structure SlicesEaten where
  george : ℕ
  bob : ℕ
  susie : ℕ
  others : ℕ

def pizza_theorem (slices : PizzaSlices) (purchased : PizzasPurchased) (eaten : SlicesEaten) : Prop :=
  slices.small = 4 ∧
  slices.large = 8 ∧
  purchased.small = 3 ∧
  purchased.large = 2 ∧
  eaten.bob = eaten.george + 1 ∧
  eaten.susie = (eaten.bob + 1) / 2 ∧
  eaten.others = 9 ∧
  (slices.small * purchased.small + slices.large * purchased.large) - 
    (eaten.george + eaten.bob + eaten.susie + eaten.others) = 10 →
  eaten.george = 6

theorem pizza_solution : 
  ∃ (slices : PizzaSlices) (purchased : PizzasPurchased) (eaten : SlicesEaten),
    pizza_theorem slices purchased eaten := by
  sorry

end NUMINAMATH_CALUDE_pizza_solution_l3534_353422


namespace NUMINAMATH_CALUDE_line_through_coefficients_l3534_353483

/-- Given two lines that intersect at (2,3), prove that the line passing through their coefficients has a specific equation -/
theorem line_through_coefficients 
  (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : 2*a₁ + 3*b₁ + 1 = 0) 
  (h₂ : 2*a₂ + 3*b₂ + 1 = 0) :
  ∀ (x y : ℝ), (x = a₁ ∧ y = b₁) ∨ (x = a₂ ∧ y = b₂) → 2*x + 3*y + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_coefficients_l3534_353483


namespace NUMINAMATH_CALUDE_cosine_product_theorem_l3534_353465

theorem cosine_product_theorem :
  (1 + Real.cos (π / 10)) * (1 + Real.cos (3 * π / 10)) *
  (1 + Real.cos (7 * π / 10)) * (1 + Real.cos (9 * π / 10)) =
  (3 - Real.sqrt 5) / 32 := by
sorry

end NUMINAMATH_CALUDE_cosine_product_theorem_l3534_353465


namespace NUMINAMATH_CALUDE_restaurant_bill_average_cost_l3534_353409

theorem restaurant_bill_average_cost
  (total_bill : ℝ)
  (gratuity_rate : ℝ)
  (num_people : ℕ)
  (h1 : total_bill = 720)
  (h2 : gratuity_rate = 0.2)
  (h3 : num_people = 6) :
  (total_bill / (1 + gratuity_rate)) / num_people = 100 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_average_cost_l3534_353409


namespace NUMINAMATH_CALUDE_max_area_region_T_l3534_353442

/-- A configuration of four circles tangent to a line -/
structure CircleConfiguration where
  radii : Fin 4 → ℝ
  tangent_point : ℝ × ℝ
  line : Set (ℝ × ℝ)

/-- The region T formed by points inside exactly one circle -/
def region_T (config : CircleConfiguration) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem stating the maximum area of region T -/
theorem max_area_region_T :
  ∃ (config : CircleConfiguration),
    (config.radii 0 = 2) ∧
    (config.radii 1 = 4) ∧
    (config.radii 2 = 6) ∧
    (config.radii 3 = 8) ∧
    (∀ (other_config : CircleConfiguration),
      (other_config.radii 0 = 2) →
      (other_config.radii 1 = 4) →
      (other_config.radii 2 = 6) →
      (other_config.radii 3 = 8) →
      area (region_T config) ≥ area (region_T other_config)) ∧
    area (region_T config) = 84 * Real.pi :=
  sorry

end NUMINAMATH_CALUDE_max_area_region_T_l3534_353442


namespace NUMINAMATH_CALUDE_point_on_same_side_l3534_353441

/-- A point (x, y) is on the same side of the line 2x - y + 1 = 0 as (1, 2) if both points satisfy 2x - y + 1 > 0 -/
def same_side (x y : ℝ) : Prop :=
  2*x - y + 1 > 0 ∧ 2*1 - 2 + 1 > 0

/-- The point (1, 0) is on the same side of the line 2x - y + 1 = 0 as the point (1, 2) -/
theorem point_on_same_side : same_side 1 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_same_side_l3534_353441


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l3534_353463

theorem quadratic_equation_real_roots (a b : ℝ) : 
  (∃ x : ℝ, x^2 + 2*(1+a)*x + (3*a^2 + 4*a*b + 4*b^2 + 2) = 0) → 
  (a = 1 ∧ b = -1/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l3534_353463
