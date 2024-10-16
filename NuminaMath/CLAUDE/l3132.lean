import Mathlib

namespace NUMINAMATH_CALUDE_integral_f_equals_pi_over_2_plus_4_over_3_l3132_313263

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 1 then Real.sqrt (1 - x^2)
  else if 1 ≤ x ∧ x ≤ 2 then x^2 - 1
  else 0

theorem integral_f_equals_pi_over_2_plus_4_over_3 :
  ∫ x in (-1)..(2), f x = π / 2 + 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_f_equals_pi_over_2_plus_4_over_3_l3132_313263


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l3132_313297

theorem cookie_boxes_problem (n : ℕ) : 
  (∃ (m a : ℕ), 
    m = n - 9 ∧ 
    a = n - 2 ∧ 
    m ≥ 1 ∧ 
    a ≥ 1 ∧ 
    m + a < n) → 
  n ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_cookie_boxes_problem_l3132_313297


namespace NUMINAMATH_CALUDE_min_white_fraction_is_one_eighth_l3132_313227

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cubes : ℕ
  red_cubes : ℕ
  white_cubes : ℕ

/-- Calculates the surface area of a cube -/
def surface_area (c : LargeCube) : ℕ := 6 * c.edge_length * c.edge_length

/-- Calculates the minimum number of white cubes needed to have at least one on each face -/
def min_white_cubes_on_surface : ℕ := 4

/-- Calculates the white surface area when white cubes are placed optimally -/
def white_surface_area : ℕ := min_white_cubes_on_surface * 3

/-- The theorem to be proved -/
theorem min_white_fraction_is_one_eighth (c : LargeCube) 
    (h1 : c.edge_length = 4)
    (h2 : c.small_cubes = 64)
    (h3 : c.red_cubes = 56)
    (h4 : c.white_cubes = 8) :
  (white_surface_area : ℚ) / (surface_area c : ℚ) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_min_white_fraction_is_one_eighth_l3132_313227


namespace NUMINAMATH_CALUDE_dorothy_profit_l3132_313270

/-- Dorothy's doughnut business profit calculation -/
theorem dorothy_profit (ingredients_cost : ℕ) (num_doughnuts : ℕ) (price_per_doughnut : ℕ) :
  ingredients_cost = 53 →
  num_doughnuts = 25 →
  price_per_doughnut = 3 →
  num_doughnuts * price_per_doughnut - ingredients_cost = 22 :=
by
  sorry

#check dorothy_profit

end NUMINAMATH_CALUDE_dorothy_profit_l3132_313270


namespace NUMINAMATH_CALUDE_grandma_backpacks_l3132_313271

def backpack_problem (original_price : ℝ) (discount_rate : ℝ) (monogram_cost : ℝ) (total_cost : ℝ) : Prop :=
  let discounted_price := original_price * (1 - discount_rate)
  let final_price := discounted_price + monogram_cost
  let num_grandchildren := total_cost / final_price
  num_grandchildren = 5

theorem grandma_backpacks :
  backpack_problem 20 0.2 12 140 := by
  sorry

end NUMINAMATH_CALUDE_grandma_backpacks_l3132_313271


namespace NUMINAMATH_CALUDE_circle_placement_possible_l3132_313282

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Theorem: In a 20x25 rectangle with 120 unit squares, there exists a point
    that is at least 0.5 units away from any edge and at least √2/2 units
    away from the center of any unit square -/
theorem circle_placement_possible (rect : Rectangle) 
    (squares : Finset Point) : 
    rect.width = 20 ∧ rect.height = 25 ∧ squares.card = 120 →
    ∃ p : Point, 
      0.5 ≤ p.x ∧ p.x ≤ rect.width - 0.5 ∧
      0.5 ≤ p.y ∧ p.y ≤ rect.height - 0.5 ∧
      ∀ s ∈ squares, (p.x - s.x)^2 + (p.y - s.y)^2 ≥ 0.5 := by
  sorry

end NUMINAMATH_CALUDE_circle_placement_possible_l3132_313282


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_odds_l3132_313276

def is_sum_of_five_consecutive_odds (n : ℤ) : Prop :=
  ∃ k : ℤ, n = (2*k-3) + (2*k-1) + (2*k+1) + (2*k+3) + (2*k+5)

theorem sum_of_five_consecutive_odds :
  is_sum_of_five_consecutive_odds 25 ∧
  is_sum_of_five_consecutive_odds 55 ∧
  is_sum_of_five_consecutive_odds 85 ∧
  is_sum_of_five_consecutive_odds 105 ∧
  ¬ is_sum_of_five_consecutive_odds 150 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_odds_l3132_313276


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3132_313251

def y := 2^(3^5 * 4^4 * 5^7 * 6^5 * 7^3 * 8^6 * 9^10)

theorem smallest_multiplier_for_perfect_square (k : ℕ) : 
  k > 0 ∧ (∃ m : ℕ, k * y = m^2) ∧ (∀ l < k, l > 0 → ¬∃ m : ℕ, l * y = m^2) ↔ k = 70 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3132_313251


namespace NUMINAMATH_CALUDE_sequence_terms_l3132_313203

def S (n : ℕ) : ℕ := n^2 + 1

def a (n : ℕ) : ℕ := S n - S (n-1)

theorem sequence_terms : a 3 = 5 ∧ a 5 = 9 := by sorry

end NUMINAMATH_CALUDE_sequence_terms_l3132_313203


namespace NUMINAMATH_CALUDE_suzy_twice_mary_age_l3132_313240

/-- The number of years in the future when Suzy will be twice Mary's age -/
def future_years : ℕ := 4

/-- Suzy's current age -/
def suzy_age : ℕ := 20

/-- Mary's current age -/
def mary_age : ℕ := 8

/-- Theorem stating that in 'future_years', Suzy will be twice Mary's age -/
theorem suzy_twice_mary_age : 
  suzy_age + future_years = 2 * (mary_age + future_years) := by sorry

end NUMINAMATH_CALUDE_suzy_twice_mary_age_l3132_313240


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3132_313299

theorem fraction_evaluation : (3 : ℚ) / (2 - 4 / (-5)) = 15 / 14 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3132_313299


namespace NUMINAMATH_CALUDE_work_completion_time_l3132_313221

/-- Given two workers x and y who can complete a work in 10 and 15 days respectively,
    prove that they can complete the work together in 6 days. -/
theorem work_completion_time (x y : ℝ) (hx : x = 1 / 10) (hy : y = 1 / 15) :
  1 / (x + y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3132_313221


namespace NUMINAMATH_CALUDE_exam_candidates_girls_l3132_313284

theorem exam_candidates_girls (total : ℕ) (boys_pass_rate girls_pass_rate fail_rate : ℚ) :
  total = 2000 ∧
  boys_pass_rate = 28/100 ∧
  girls_pass_rate = 32/100 ∧
  fail_rate = 702/1000 →
  ∃ (girls : ℕ), 
    girls + (total - girls) = total ∧
    (girls_pass_rate * girls + boys_pass_rate * (total - girls)) / total = 1 - fail_rate ∧
    girls = 900 := by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_girls_l3132_313284


namespace NUMINAMATH_CALUDE_vector_addition_rule_l3132_313219

variable {V : Type*} [AddCommGroup V]

theorem vector_addition_rule (A B C : V) : 
  (C - A) + (B - C) = B - A :=
sorry

end NUMINAMATH_CALUDE_vector_addition_rule_l3132_313219


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3132_313294

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The original number we want to express in scientific notation -/
def original_number : ℝ := 0.00000164

/-- The scientific notation representation we want to prove is correct -/
def scientific_rep : ScientificNotation := {
  coefficient := 1.64
  exponent := -6
  coefficient_range := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : original_number = scientific_rep.coefficient * (10 : ℝ) ^ scientific_rep.exponent := by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3132_313294


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l3132_313201

theorem quadratic_root_m_value (m : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x - m = 0 ∧ x = 1) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l3132_313201


namespace NUMINAMATH_CALUDE_number_of_divisors_of_2002_l3132_313204

theorem number_of_divisors_of_2002 : ∃ (d : ℕ → ℕ), d 2002 = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_2002_l3132_313204


namespace NUMINAMATH_CALUDE_frog_probability_l3132_313230

-- Define the number of lily pads
def num_pads : ℕ := 9

-- Define the set of predator positions
def predator_positions : Set ℕ := {2, 5, 6}

-- Define the target position
def target_position : ℕ := 7

-- Define the probability of moving 1 or 2 positions
def move_probability : ℚ := 1/2

-- Define the function to calculate the probability of reaching the target
def reach_probability (start : ℕ) (target : ℕ) (predators : Set ℕ) (p : ℚ) : ℚ :=
  sorry

-- Theorem statement
theorem frog_probability :
  reach_probability 0 target_position predator_positions move_probability = 1/16 :=
sorry

end NUMINAMATH_CALUDE_frog_probability_l3132_313230


namespace NUMINAMATH_CALUDE_sector_area_l3132_313218

/-- The area of a circular sector with central angle 60° and radius 10 cm is 50π/3 cm² -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 60) (h2 : r = 10) :
  (θ / 360) * π * r^2 = 50 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3132_313218


namespace NUMINAMATH_CALUDE_product_of_successive_numbers_l3132_313285

theorem product_of_successive_numbers : 
  let n : Real := 88.49858755935034
  let product := n * (n + 1)
  ∃ ε > 0, |product - 7913| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_successive_numbers_l3132_313285


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3132_313252

theorem problem_1 (x y : ℝ) (h1 : x - y = 3) (h2 : x * y = 2) :
  x^2 + y^2 = 13 := by sorry

theorem problem_2 (a : ℝ) (h : (4 - a)^2 + (a + 3)^2 = 7) :
  (4 - a) * (a + 3) = 21 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3132_313252


namespace NUMINAMATH_CALUDE_quadratic_fit_energy_production_l3132_313223

/-- Represents a quadratic function of the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the quadratic function at a given x -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Theorem: There exists a quadratic function that fits the given data points
    and predicts the correct value for 2007 -/
theorem quadratic_fit_energy_production : ∃ f : QuadraticFunction,
  f.evaluate 0 = 8.6 ∧
  f.evaluate 5 = 10.4 ∧
  f.evaluate 10 = 12.9 ∧
  f.evaluate 15 = 16.1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_fit_energy_production_l3132_313223


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l3132_313205

theorem square_rectangle_area_relation (x : ℝ) :
  let square_side := x - 4
  let rect_length := x - 5
  let rect_width := x + 6
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  rect_area = 3 * square_area →
  ∃ x₁ x₂ : ℝ, (x = x₁ ∨ x = x₂) ∧ x₁ + x₂ = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l3132_313205


namespace NUMINAMATH_CALUDE_range_when_p_true_range_when_p_false_and_q_true_l3132_313278

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - a*x + a > 0

def q (a : ℝ) : Prop := ∃ x y : ℝ, x^2 / (a^2 + 12) - y^2 / (4 - a^2) = 1

-- Theorem for the first part
theorem range_when_p_true :
  {a : ℝ | p a} = {a : ℝ | 0 < a ∧ a < 4} :=
sorry

-- Theorem for the second part
theorem range_when_p_false_and_q_true :
  {a : ℝ | ¬(p a) ∧ q a} = {a : ℝ | -2 < a ∧ a ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_range_when_p_true_range_when_p_false_and_q_true_l3132_313278


namespace NUMINAMATH_CALUDE_nina_weekend_sales_l3132_313237

/-- Calculates the total money Nina made from jewelry sales over the weekend -/
def weekend_sales (necklace_price bracelet_price earring_price ensemble_price : ℚ)
                  (necklaces_sold bracelets_sold earrings_sold ensembles_sold : ℕ) : ℚ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_price * earrings_sold +
  ensemble_price * ensembles_sold

/-- Proves that Nina's weekend sales totaled $565.00 -/
theorem nina_weekend_sales :
  weekend_sales 25 15 10 45 5 10 20 2 = 565 := by
  sorry

end NUMINAMATH_CALUDE_nina_weekend_sales_l3132_313237


namespace NUMINAMATH_CALUDE_vector_ratio_implies_k_l3132_313261

/-- Given vectors a and b in ℝ², if (a + 2b) / (3a - b) exists, then k = -6 -/
theorem vector_ratio_implies_k (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 3))
  (h2 : b = (-2, k))
  (h3 : ∃ (r : ℝ), r • (3 • a - b) = a + 2 • b) :
  k = -6 := by
  sorry

end NUMINAMATH_CALUDE_vector_ratio_implies_k_l3132_313261


namespace NUMINAMATH_CALUDE_cone_lateral_surface_l3132_313224

theorem cone_lateral_surface (l r : ℝ) (h : l > 0) (k : r > 0) : 
  (2 * π * r) / l = 4 * π / 3 → r / l = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_l3132_313224


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l3132_313211

theorem discounted_price_calculation (list_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  list_price = 70 ∧ 
  discount1 = 0.1 ∧ 
  discount2 = 0.04999999999999997 →
  list_price * (1 - discount1) * (1 - discount2) = 59.85 :=
by
  sorry

#eval (70 : ℝ) * (1 - 0.1) * (1 - 0.04999999999999997)

end NUMINAMATH_CALUDE_discounted_price_calculation_l3132_313211


namespace NUMINAMATH_CALUDE_prob_all_white_value_l3132_313286

/-- The number of small cubes forming the larger cube -/
def num_cubes : ℕ := 8

/-- The probability of a single small cube showing a white face after flipping -/
def prob_white_face : ℚ := 5/6

/-- The probability of all surfaces of the larger cube becoming white after flipping -/
def prob_all_white : ℚ := (prob_white_face ^ num_cubes).num / (prob_white_face ^ num_cubes).den

theorem prob_all_white_value : prob_all_white = 390625/1679616 := by sorry

end NUMINAMATH_CALUDE_prob_all_white_value_l3132_313286


namespace NUMINAMATH_CALUDE_number_of_broadcasting_methods_l3132_313289

/-- Represents the number of commercial advertisements -/
def num_commercial_ads : ℕ := 4

/-- Represents the number of public service advertisements -/
def num_public_service_ads : ℕ := 2

/-- Represents the total number of advertisements -/
def total_ads : ℕ := num_commercial_ads + num_public_service_ads

/-- Represents the fact that public service ads must be at the beginning and end -/
def public_service_ads_fixed : Prop := True

theorem number_of_broadcasting_methods : 
  (num_commercial_ads = 4 ∧ 
   num_public_service_ads = 2 ∧ 
   total_ads = 6 ∧ 
   public_service_ads_fixed) → 
  (Nat.factorial num_commercial_ads = 24) := by
  sorry

end NUMINAMATH_CALUDE_number_of_broadcasting_methods_l3132_313289


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3132_313241

def total_toppings : ℕ := 9
def toppings_to_select : ℕ := 4
def required_toppings : ℕ := 2

theorem pizza_toppings_combinations :
  (Nat.choose total_toppings toppings_to_select) -
  (Nat.choose (total_toppings - required_toppings) toppings_to_select) = 91 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3132_313241


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3132_313267

theorem inequality_equivalence (x : ℝ) : x + 1 < (4 + 3 * x) / 2 ↔ x > -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3132_313267


namespace NUMINAMATH_CALUDE_max_area_rectangle_l3132_313209

/-- The perimeter of the rectangle formed by matches --/
def perimeter : ℕ := 22

/-- Function to calculate the area of a rectangle given its length and width --/
def area (length width : ℕ) : ℕ := length * width

/-- Theorem stating that the rectangle with dimensions 6 × 5 has the maximum area
    among all rectangles with a perimeter of 22 units --/
theorem max_area_rectangle :
  ∀ l w : ℕ, 
    2 * (l + w) = perimeter → 
    area l w ≤ area 6 5 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l3132_313209


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l3132_313287

theorem sum_of_solutions_squared_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 7)^2 = 36 ∧ (x₂ - 7)^2 = 36 ∧ x₁ + x₂ = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l3132_313287


namespace NUMINAMATH_CALUDE_alice_wins_coin_game_l3132_313246

def coin_game (initial_coins : ℕ) : Prop :=
  ∃ (k : ℕ),
    k^2 ≤ initial_coins ∧
    initial_coins < k * (k + 1) - 1

theorem alice_wins_coin_game :
  coin_game 1331 :=
sorry

end NUMINAMATH_CALUDE_alice_wins_coin_game_l3132_313246


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_l3132_313255

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x * Real.sin x

theorem derivative_f_at_pi :
  deriv f π = -Real.sqrt π := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_l3132_313255


namespace NUMINAMATH_CALUDE_product_of_roots_l3132_313258

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 17 → ∃ y : ℝ, (x + 3) * (x - 4) = 17 ∧ (x * y = -29) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3132_313258


namespace NUMINAMATH_CALUDE_average_rst_l3132_313293

theorem average_rst (r s t : ℝ) (h : (5 / 4) * (r + s + t) = 20) :
  (r + s + t) / 3 = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_rst_l3132_313293


namespace NUMINAMATH_CALUDE_safe_opening_l3132_313233

theorem safe_opening (a b c n m k : ℕ) :
  ∃ (x y z : ℕ), ∃ (w : ℕ),
    (a^n * b^m * c^k = w^3) ∨
    (a^n * b^y * c^z = w^3) ∨
    (a^x * b^m * c^z = w^3) ∨
    (a^x * b^y * c^k = w^3) ∨
    (x^n * b^m * c^z = w^3) ∨
    (x^n * b^y * c^k = w^3) ∨
    (x^y * b^m * c^k = w^3) ∨
    (a^x * y^m * c^z = w^3) ∨
    (a^x * y^z * c^k = w^3) ∨
    (x^n * y^m * c^z = w^3) ∨
    (x^n * y^z * c^k = w^3) ∨
    (x^y * z^m * c^k = w^3) ∨
    (a^x * y^m * z^k = w^3) ∨
    (x^n * y^m * z^k = w^3) ∨
    (x^y * b^m * z^k = w^3) ∨
    (x^y * z^m * c^k = w^3) :=
by sorry


end NUMINAMATH_CALUDE_safe_opening_l3132_313233


namespace NUMINAMATH_CALUDE_music_club_members_not_playing_l3132_313212

theorem music_club_members_not_playing (total_members guitar_players piano_players both_players : ℕ) 
  (h1 : total_members = 80)
  (h2 : guitar_players = 45)
  (h3 : piano_players = 30)
  (h4 : both_players = 18) :
  total_members - (guitar_players + piano_players - both_players) = 23 := by
  sorry

end NUMINAMATH_CALUDE_music_club_members_not_playing_l3132_313212


namespace NUMINAMATH_CALUDE_b_is_positive_l3132_313206

theorem b_is_positive (x a : ℤ) (h1 : x < a) (h2 : a < 0) (b : ℤ) (h3 : b = x^2 - a^2) : b > 0 := by
  sorry

end NUMINAMATH_CALUDE_b_is_positive_l3132_313206


namespace NUMINAMATH_CALUDE_min_cuts_correct_l3132_313216

/-- The minimum number of cuts required to transform a square into 100 20-gons -/
def min_cuts : ℕ := 1699

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a 20-gon -/
def twenty_gon_sides : ℕ := 20

/-- The number of 20-gons we want to obtain -/
def target_polygons : ℕ := 100

/-- The maximum increase in the number of sides per cut -/
def max_side_increase : ℕ := 4

/-- The total number of sides in the final configuration -/
def total_final_sides : ℕ := target_polygons * twenty_gon_sides

theorem min_cuts_correct :
  min_cuts = (total_final_sides - square_sides) / max_side_increase + 
             (target_polygons - 1) :=
by sorry

end NUMINAMATH_CALUDE_min_cuts_correct_l3132_313216


namespace NUMINAMATH_CALUDE_female_fraction_is_25_69_l3132_313254

/-- Represents the basketball club membership data --/
structure ClubData where
  maleLastYear : ℕ
  totalIncrease : ℚ
  maleIncrease : ℚ
  femaleIncrease : ℚ

/-- Calculates the fraction of female members this year --/
def femaleFraction (data : ClubData) : ℚ :=
  let maleThisYear := data.maleLastYear * (1 + data.maleIncrease)
  let femaleLastYear := (data.maleLastYear : ℚ) * (1 + data.totalIncrease - 1) / (data.femaleIncrease - 1)
  let femaleThisYear := femaleLastYear * (1 + data.femaleIncrease)
  let totalThisYear := maleThisYear + femaleThisYear
  femaleThisYear / totalThisYear

/-- Theorem stating that given the conditions, the fraction of female members this year is 25/69 --/
theorem female_fraction_is_25_69 (data : ClubData) 
  (h1 : data.maleLastYear = 30)
  (h2 : data.totalIncrease = 0.15)
  (h3 : data.maleIncrease = 0.10)
  (h4 : data.femaleIncrease = 0.25) :
  femaleFraction data = 25 / 69 := by
  sorry


end NUMINAMATH_CALUDE_female_fraction_is_25_69_l3132_313254


namespace NUMINAMATH_CALUDE_equal_perimeter_triangles_l3132_313273

theorem equal_perimeter_triangles (a b c x y : ℝ) : 
  a = 7 → b = 12 → c = 9 → x = 2 → y = 7 → x + y = c →
  (a + x + (b - a)) = (b + y + (b - a)) := by sorry

end NUMINAMATH_CALUDE_equal_perimeter_triangles_l3132_313273


namespace NUMINAMATH_CALUDE_vector_equality_iff_collinear_l3132_313292

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- The theorem states that for arbitrary points O, A, B, C in a vector space and a scalar k,
    the equality OC = k*OA + (1-k)*OB is equivalent to A, B, and C being collinear. -/
theorem vector_equality_iff_collinear 
  (O A B C : V) (k : ℝ) : 
  (C - O = k • (A - O) + (1 - k) • (B - O)) ↔ 
  ∃ t : ℝ, C - B = t • (A - B) :=
sorry

end NUMINAMATH_CALUDE_vector_equality_iff_collinear_l3132_313292


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l3132_313214

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l3132_313214


namespace NUMINAMATH_CALUDE_number_of_plates_l3132_313249

def actual_weight : ℕ := 470
def perceived_weight : ℕ := 600
def bar_weight : ℕ := 45
def weight_mistake_per_plate : ℕ := 10

theorem number_of_plates : 
  (perceived_weight - actual_weight) / weight_mistake_per_plate = 13 := by
  sorry

end NUMINAMATH_CALUDE_number_of_plates_l3132_313249


namespace NUMINAMATH_CALUDE_min_even_integers_l3132_313296

theorem min_even_integers (a b c d e f g : ℤ) : 
  a + b + c = 40 →
  a + b + c + d + e + f = 70 →
  a + b + c + d + e + f + g = 92 →
  (∃ (evens : Finset ℤ), evens ⊆ {a, b, c, d, e, f, g} ∧ 
    (∀ x ∈ evens, Even x) ∧ 
    evens.card = 4 ∧
    (∀ (other_evens : Finset ℤ), other_evens ⊆ {a, b, c, d, e, f, g} ∧ 
      (∀ x ∈ other_evens, Even x) →
      other_evens.card ≥ 4)) :=
sorry

end NUMINAMATH_CALUDE_min_even_integers_l3132_313296


namespace NUMINAMATH_CALUDE_coefficient_of_x14_is_neg_one_l3132_313295

-- Define the dividend and divisor polynomials
def dividend (x : ℝ) : ℝ := x^1951 - 1
def divisor (x : ℝ) : ℝ := x^4 + x^3 + 2*x^2 + x + 1

-- Define the quotient function
noncomputable def quotient (x : ℝ) : ℝ := dividend x / divisor x

-- Theorem statement
theorem coefficient_of_x14_is_neg_one :
  ∃ (q : ℝ → ℝ), (∀ x, quotient x = q x) ∧ 
  (∃ (a : ℝ → ℝ) (b : ℝ → ℝ), 
    (∀ x, q x = a x + x^14 * (-1) + x^15 * b x)) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x14_is_neg_one_l3132_313295


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_pi_6_l3132_313231

theorem sin_2alpha_plus_pi_6 (α : Real) (h : Real.cos (α - π / 6) = 1 / 3) :
  Real.sin (2 * α + π / 6) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_pi_6_l3132_313231


namespace NUMINAMATH_CALUDE_expression_evaluation_l3132_313238

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := -1/3
  b^2 - a^2 + 2*(a^2 + a*b) - (a^2 + b^2) = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3132_313238


namespace NUMINAMATH_CALUDE_exists_number_with_digit_sum_property_l3132_313217

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of digits of (n + 18) 
    is equal to the sum of digits of n minus 18 -/
theorem exists_number_with_digit_sum_property : 
  ∃ n : ℕ, sumOfDigits (n + 18) = sumOfDigits n - 18 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_digit_sum_property_l3132_313217


namespace NUMINAMATH_CALUDE_movie_ticket_distribution_l3132_313265

theorem movie_ticket_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (n.descFactorial k) = 720 :=
sorry

end NUMINAMATH_CALUDE_movie_ticket_distribution_l3132_313265


namespace NUMINAMATH_CALUDE_product_of_numbers_l3132_313290

theorem product_of_numbers (x y : ℝ) 
  (sum_of_squares : x^2 + y^2 = 289) 
  (sum_of_numbers : x + y = 23) : 
  x * y = 120 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3132_313290


namespace NUMINAMATH_CALUDE_asymptotic_lines_of_hyperbola_l3132_313274

/-- The asymptotic lines of the hyperbola x²/9 - y² = 1 are y = ±x/3 -/
theorem asymptotic_lines_of_hyperbola :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2/9 - y^2 = 1}
  let asymptotic_lines := {(x, y) : ℝ × ℝ | y = x/3 ∨ y = -x/3}
  (∀ (p : ℝ × ℝ), p ∈ asymptotic_lines ↔ 
    (∃ (ε : ℝ → ℝ), (∀ t, t ≠ 0 → |ε t| < |t|) ∧
      ∀ t : ℝ, t ≠ 0 → (t*p.1 + ε t, t*p.2 + ε t) ∈ hyperbola)) :=
by sorry

end NUMINAMATH_CALUDE_asymptotic_lines_of_hyperbola_l3132_313274


namespace NUMINAMATH_CALUDE_s_range_for_composites_l3132_313245

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def s (n : ℕ) : ℕ := sorry

theorem s_range_for_composites :
  (∀ n : ℕ, is_composite n → s n ≥ 12) ∧
  (∀ m : ℕ, m ≥ 12 → ∃ n : ℕ, is_composite n ∧ s n = m) :=
sorry

end NUMINAMATH_CALUDE_s_range_for_composites_l3132_313245


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l3132_313202

theorem smallest_multiple_of_6_and_15 : 
  ∃ b : ℕ+, (∀ n : ℕ+, 6 ∣ n ∧ 15 ∣ n → b ≤ n) ∧ 6 ∣ b ∧ 15 ∣ b ∧ b = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l3132_313202


namespace NUMINAMATH_CALUDE_container_capacity_proof_l3132_313275

theorem container_capacity_proof :
  ∀ (C : ℝ),
    (C > 0) →
    (0.3 * C + 27 = 0.75 * C) →
    C = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_container_capacity_proof_l3132_313275


namespace NUMINAMATH_CALUDE_fourth_selected_is_48_l3132_313279

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  total : ℕ
  sample_size : ℕ
  first_three : Fin 3 → ℕ

/-- Calculates the interval for systematic sampling -/
def sampling_interval (s : SystematicSampling) : ℕ :=
  s.total / s.sample_size

/-- Theorem: In the given systematic sampling scenario, the fourth selected number is 48 -/
theorem fourth_selected_is_48 (s : SystematicSampling) 
  (h_total : s.total = 60)
  (h_sample_size : s.sample_size = 4)
  (h_first_three : s.first_three = ![3, 18, 33]) :
  s.first_three 2 + sampling_interval s = 48 := by
  sorry

end NUMINAMATH_CALUDE_fourth_selected_is_48_l3132_313279


namespace NUMINAMATH_CALUDE_right_triangle_identification_l3132_313232

theorem right_triangle_identification (a b c : ℝ) : 
  (a = 3 ∧ b = 4 ∧ c = 5) → 
  (a^2 + b^2 = c^2) ∧ 
  ¬(2^2 + 4^2 = 5^2) ∧ 
  ¬((Real.sqrt 3)^2 + (Real.sqrt 4)^2 = (Real.sqrt 5)^2) ∧ 
  ¬(5^2 + 13^2 = 14^2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l3132_313232


namespace NUMINAMATH_CALUDE_selection_schemes_count_l3132_313281

/-- The number of people in the group -/
def totalPeople : ℕ := 5

/-- The number of cities to be visited -/
def totalCities : ℕ := 4

/-- The number of people who can visit Paris (excluding A) -/
def parisVisitors : ℕ := totalPeople - 1

/-- Calculate the number of selection schemes -/
def selectionSchemes : ℕ :=
  parisVisitors * (totalPeople - 1) * (totalPeople - 2) * (totalPeople - 3)

/-- Theorem stating the number of selection schemes is 96 -/
theorem selection_schemes_count :
  selectionSchemes = 96 := by sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l3132_313281


namespace NUMINAMATH_CALUDE_min_value_fraction_l3132_313244

theorem min_value_fraction (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_eq_one : x + y + z + w = 1) :
  ∀ a b c d : ℝ, 0 < a → 0 < b → 0 < c → 0 < d → a + b + c + d = 1 →
  (x + y + z) / (x * y * z * w) ≤ (a + b + c) / (a * b * c * d) ∧
  (x + y + z) / (x * y * z * w) = 144 := by
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3132_313244


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l3132_313253

theorem arctan_equation_solution :
  ∃ y : ℝ, y > 0 ∧ Real.arctan (2 / y) + Real.arctan (1 / y^2) = π / 4 :=
by
  -- The proof would go here
  sorry

#check arctan_equation_solution

end NUMINAMATH_CALUDE_arctan_equation_solution_l3132_313253


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3132_313242

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3132_313242


namespace NUMINAMATH_CALUDE_fair_haired_employees_percentage_l3132_313208

theorem fair_haired_employees_percentage :
  -- Define the total number of employees
  ∀ (total_employees : ℕ),
  total_employees > 0 →
  -- Define the number of women with fair hair
  ∀ (women_fair_hair : ℕ),
  women_fair_hair = (28 * total_employees) / 100 →
  -- Define the number of fair-haired employees
  ∀ (fair_haired_employees : ℕ),
  women_fair_hair = (40 * fair_haired_employees) / 100 →
  -- The percentage of employees with fair hair is 70%
  (fair_haired_employees : ℚ) / total_employees = 70 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_fair_haired_employees_percentage_l3132_313208


namespace NUMINAMATH_CALUDE_ice_cream_permutations_l3132_313243

/-- The number of distinct permutations of n items, where some items may be identical -/
def distinctPermutations (n : ℕ) (itemCounts : List ℕ) : ℕ :=
  Nat.factorial n / (itemCounts.map Nat.factorial).prod

theorem ice_cream_permutations :
  distinctPermutations 4 [2, 1, 1] = 12 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_permutations_l3132_313243


namespace NUMINAMATH_CALUDE_inequality_proof_l3132_313225

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  1 / (a - b) < 1 / a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3132_313225


namespace NUMINAMATH_CALUDE_x_intercept_of_perpendicular_line_l3132_313298

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℚ := -l.y_intercept / l.slope

/-- Two lines are perpendicular if their slopes are negative reciprocals -/
def perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

theorem x_intercept_of_perpendicular_line (given_line perp_line : Line) :
  given_line.slope = -5/3 →
  perpendicular given_line perp_line →
  perp_line.y_intercept = -4 →
  x_intercept perp_line = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_perpendicular_line_l3132_313298


namespace NUMINAMATH_CALUDE_hcf_of_numbers_l3132_313272

theorem hcf_of_numbers (x y : ℕ+) 
  (sum_eq : x + y = 45)
  (lcm_eq : Nat.lcm x y = 120)
  (sum_recip_eq : (1 : ℚ) / x + (1 : ℚ) / y = 11 / 120) :
  Nat.gcd x y = 1 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_numbers_l3132_313272


namespace NUMINAMATH_CALUDE_one_third_point_coordinates_l3132_313268

/-- 
Given two points (x₁, y₁) and (x₂, y₂) in a 2D plane, and a rational number t between 0 and 1,
this function returns the coordinates of a point that is t of the way from (x₁, y₁) to (x₂, y₂).
-/
def pointOnLine (x₁ y₁ x₂ y₂ t : ℚ) : ℚ × ℚ :=
  ((1 - t) * x₁ + t * x₂, (1 - t) * y₁ + t * y₂)

theorem one_third_point_coordinates :
  let p := pointOnLine 2 6 8 (-2) (1/3)
  p.1 = 4 ∧ p.2 = 10/3 := by sorry

end NUMINAMATH_CALUDE_one_third_point_coordinates_l3132_313268


namespace NUMINAMATH_CALUDE_ellipse_dot_product_constant_l3132_313260

/-- The ellipse with semi-major axis 2 and semi-minor axis √2 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + (p.2^2 / 2) = 1}

/-- The line x = 2 -/
def Line : Set (ℝ × ℝ) :=
  {p | p.1 = 2}

/-- Left vertex of the ellipse -/
def A₁ : ℝ × ℝ := (-2, 0)

/-- Theorem: For any point C on the ellipse and D on the line x = 2,
    if A₁C = 2CD, then OC · OD = 4 -/
theorem ellipse_dot_product_constant
    (C : ℝ × ℝ) (hC : C ∈ Ellipse)
    (D : ℝ × ℝ) (hD : D ∈ Line)
    (h : dist A₁ C = 2 * dist C D) :
  C.1 * D.1 + C.2 * D.2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_dot_product_constant_l3132_313260


namespace NUMINAMATH_CALUDE_rental_cost_equality_l3132_313213

/-- The daily rate for Sunshine Car Rentals in dollars -/
def sunshine_daily_rate : ℝ := 17.99

/-- The per-mile rate for Sunshine Car Rentals in dollars -/
def sunshine_mile_rate : ℝ := 0.18

/-- The daily rate for City Rentals in dollars -/
def city_daily_rate : ℝ := 18.95

/-- The per-mile rate for City Rentals in dollars -/
def city_mile_rate : ℝ := 0.16

/-- The mileage at which the cost is the same for both rental companies -/
def equal_cost_mileage : ℝ := 48

theorem rental_cost_equality :
  sunshine_daily_rate + sunshine_mile_rate * equal_cost_mileage =
  city_daily_rate + city_mile_rate * equal_cost_mileage :=
by sorry

end NUMINAMATH_CALUDE_rental_cost_equality_l3132_313213


namespace NUMINAMATH_CALUDE_baking_ingredient_calculation_l3132_313222

/-- Represents the ingredients needed for baking --/
structure BakingIngredients where
  flour_cake : ℝ
  flour_cookies : ℝ
  sugar_cake : ℝ
  sugar_cookies : ℝ

/-- Represents the available ingredients --/
structure AvailableIngredients where
  flour : ℝ
  sugar : ℝ

/-- Calculates the difference between available and needed ingredients --/
def ingredientDifference (needed : BakingIngredients) (available : AvailableIngredients) : 
  ℝ × ℝ :=
  let total_flour_needed := needed.flour_cake + needed.flour_cookies
  let total_sugar_needed := needed.sugar_cake + needed.sugar_cookies
  (available.flour - total_flour_needed, available.sugar - total_sugar_needed)

theorem baking_ingredient_calculation 
  (needed : BakingIngredients) 
  (available : AvailableIngredients) : 
  needed.flour_cake = 6 ∧ 
  needed.flour_cookies = 2 ∧ 
  needed.sugar_cake = 3.5 ∧ 
  needed.sugar_cookies = 1.5 ∧
  available.flour = 8 ∧ 
  available.sugar = 4 → 
  ingredientDifference needed available = (0, -1) := by
  sorry

end NUMINAMATH_CALUDE_baking_ingredient_calculation_l3132_313222


namespace NUMINAMATH_CALUDE_water_needed_for_lemonade_l3132_313257

-- Define the ratio of water to lemon juice
def water_ratio : ℚ := 4
def lemon_juice_ratio : ℚ := 1

-- Define the total volume in gallons
def total_volume : ℚ := 3

-- Define the conversion factor from gallons to quarts
def quarts_per_gallon : ℚ := 4

-- Theorem statement
theorem water_needed_for_lemonade :
  let total_ratio : ℚ := water_ratio + lemon_juice_ratio
  let total_quarts : ℚ := total_volume * quarts_per_gallon
  let quarts_per_part : ℚ := total_quarts / total_ratio
  let water_quarts : ℚ := water_ratio * quarts_per_part
  water_quarts = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_for_lemonade_l3132_313257


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3132_313207

/-- Given a hyperbola with equation mx^2 + y^2 = 1, if one of its asymptotes has a slope of 2, then m = -4 -/
theorem hyperbola_asymptote_slope (m : ℝ) : 
  (∃ (x y : ℝ), m * x^2 + y^2 = 1) →  -- Hyperbola equation exists
  (∃ (k : ℝ), k = 2 ∧ k^2 = -m) →    -- One asymptote has slope 2
  m = -4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3132_313207


namespace NUMINAMATH_CALUDE_sequence_product_l3132_313256

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) / b n = b (n + 2) / b (n + 1)

/-- The main theorem -/
theorem sequence_product (a : ℕ → ℝ) (b : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 11 = 8 →
  geometric_sequence b →
  b 7 = a 7 →
  b 6 * b 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_sequence_product_l3132_313256


namespace NUMINAMATH_CALUDE_ball_purchase_theorem_l3132_313250

theorem ball_purchase_theorem (M : ℕ) (V B : ℕ) : 
  (M / 15 = 15) →  -- Can buy 15 volleyballs
  (M / 12 = 12) →  -- Can buy 12 basketballs
  (V + B = 14) →   -- Total of 14 balls bought
  ((M / 15) * V + (M / 12) * B = M) →  -- Total cost equals available money
  (V - B = 6) :=   -- Difference between volleyballs and basketballs
by sorry

end NUMINAMATH_CALUDE_ball_purchase_theorem_l3132_313250


namespace NUMINAMATH_CALUDE_work_completion_time_l3132_313288

/-- The time taken for two workers to complete three times a piece of work -/
def time_to_complete_work (aarti_rate : ℚ) (bina_rate : ℚ) : ℚ :=
  3 / (aarti_rate + bina_rate)

/-- Theorem stating that Aarti and Bina will take approximately 9.23 days to complete three times the work -/
theorem work_completion_time :
  let aarti_rate : ℚ := 1 / 5
  let bina_rate : ℚ := 1 / 8
  abs (time_to_complete_work aarti_rate bina_rate - 9.23) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3132_313288


namespace NUMINAMATH_CALUDE_delivery_cost_fraction_l3132_313229

/-- Proves that the fraction of the remaining amount spent on delivery costs is 1/4 -/
theorem delivery_cost_fraction (total_cost : ℝ) (salary_fraction : ℝ) (order_cost : ℝ)
  (h1 : total_cost = 4000)
  (h2 : salary_fraction = 2/5)
  (h3 : order_cost = 1800) :
  let salary_cost := salary_fraction * total_cost
  let remaining_after_salary := total_cost - salary_cost
  let delivery_cost := remaining_after_salary - order_cost
  delivery_cost / remaining_after_salary = 1/4 := by
sorry

end NUMINAMATH_CALUDE_delivery_cost_fraction_l3132_313229


namespace NUMINAMATH_CALUDE_fathers_age_l3132_313226

theorem fathers_age (S F : ℕ) 
  (h1 : 2 * S + F = 70) 
  (h2 : S + 2 * F = 95) : 
  F = 40 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l3132_313226


namespace NUMINAMATH_CALUDE_phone_profit_optimization_l3132_313239

/-- Represents the profit calculation and optimization problem for two types of phones. -/
theorem phone_profit_optimization
  (profit_A_B : ℕ → ℕ → ℝ)
  (total_phones : ℕ)
  (h1 : profit_A_B 1 1 = 600)
  (h2 : profit_A_B 3 2 = 1400)
  (h3 : total_phones = 20)
  (h4 : ∀ x y, x + y = total_phones → y ≤ 2 / 3 * x) :
  ∃ (x y : ℕ),
    x + y = total_phones ∧
    y ≤ 2 / 3 * x ∧
    ∀ (a b : ℕ), a + b = total_phones → a ≥ 0 → b ≥ 0 →
      profit_A_B x y ≥ profit_A_B a b ∧
      profit_A_B x y = 5600 :=
by sorry

end NUMINAMATH_CALUDE_phone_profit_optimization_l3132_313239


namespace NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l3132_313283

/-- Triangle with positive integer side lengths --/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Isosceles triangle where two sides are equal --/
def IsoscelesTriangle (t : IntegerTriangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Perimeter of a triangle --/
def Perimeter (t : IntegerTriangle) : ℕ :=
  t.a.val + t.b.val + t.c.val

/-- Angle bisector theorem relation --/
def AngleBisectorRelation (t : IntegerTriangle) (bisectorLength : ℕ+) : Prop :=
  ∃ (x y : ℕ+), x + y = t.c ∧ bisectorLength * t.c = t.a * y

/-- Main theorem --/
theorem smallest_perimeter_isosceles_triangle :
  ∀ (t : IntegerTriangle),
    IsoscelesTriangle t →
    AngleBisectorRelation t 8 →
    (∀ (t' : IntegerTriangle),
      IsoscelesTriangle t' →
      AngleBisectorRelation t' 8 →
      Perimeter t ≤ Perimeter t') →
    Perimeter t = 108 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l3132_313283


namespace NUMINAMATH_CALUDE_f_equals_g_l3132_313264

-- Define the functions f and g
def f (x : ℝ) : ℝ := x - 1
def g (t : ℝ) : ℝ := t - 1

-- Theorem stating that f and g represent the same function
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  sorry

end NUMINAMATH_CALUDE_f_equals_g_l3132_313264


namespace NUMINAMATH_CALUDE_min_value_expression_l3132_313269

theorem min_value_expression (a b c : ℝ) (h1 : c > b) (h2 : b > a) (h3 : c ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (c - b)^2) / c^2 ≥ 0 ∧
  ∃ a b, ((a + b)^2 + (b - c)^2 + (c - b)^2) / c^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3132_313269


namespace NUMINAMATH_CALUDE_nell_final_baseball_cards_l3132_313248

/-- Represents the number of cards Nell has --/
structure Cards where
  initial_baseball : ℕ
  initial_ace : ℕ
  final_ace : ℕ
  difference : ℕ

/-- Calculates the final number of baseball cards Nell has --/
def final_baseball_cards (c : Cards) : ℕ :=
  c.final_ace - c.difference

/-- Theorem stating that Nell's final baseball card count is 111 --/
theorem nell_final_baseball_cards :
  let c : Cards := {
    initial_baseball := 239,
    initial_ace := 38,
    final_ace := 376,
    difference := 265
  }
  final_baseball_cards c = 111 := by
  sorry

end NUMINAMATH_CALUDE_nell_final_baseball_cards_l3132_313248


namespace NUMINAMATH_CALUDE_cricket_team_left_handed_fraction_l3132_313234

/-- Proves that the fraction of left-handed non-throwers is 1/3 given the conditions of the cricket team -/
theorem cricket_team_left_handed_fraction 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (right_handed : ℕ) 
  (h1 : total_players = 61) 
  (h2 : throwers = 37) 
  (h3 : right_handed = 53) 
  (h4 : throwers ≤ right_handed) : 
  (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_left_handed_fraction_l3132_313234


namespace NUMINAMATH_CALUDE_quadratic_complex_root_l3132_313215

/-- Given a quadratic equation x^2 + px + q = 0 with real coefficients,
    if 1 + i is a root, then q = 2. -/
theorem quadratic_complex_root (p q : ℝ) : 
  (∀ x : ℂ, x^2 + p * x + q = 0 ↔ x = (1 + I) ∨ x = (1 - I)) → q = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_complex_root_l3132_313215


namespace NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l3132_313228

-- Define the number of coin tosses
def num_tosses : ℕ := 8

-- Define the number of heads we're looking for
def target_heads : ℕ := 3

-- Define a function to calculate the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := sorry

-- Define a function to calculate the probability of getting exactly k heads in n tosses
def probability_exactly_k_heads (n k : ℕ) : ℚ :=
  (binomial_coefficient n k : ℚ) / (2 ^ n : ℚ)

-- Theorem statement
theorem probability_three_heads_in_eight_tosses :
  probability_exactly_k_heads num_tosses target_heads = 7 / 32 := by sorry

end NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l3132_313228


namespace NUMINAMATH_CALUDE_inequality_proof_l3132_313259

theorem inequality_proof (x : ℝ) (n : ℕ) (a : ℝ) 
  (h1 : x > 0) (h2 : n > 0) (h3 : x + a / x^n ≥ n + 1) : 
  a = n^n := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3132_313259


namespace NUMINAMATH_CALUDE_distance_to_concert_l3132_313291

/-- The distance to a concert given the distance driven before stopping for gas and the remaining distance after getting gas -/
theorem distance_to_concert 
  (distance_before_gas : ℕ) 
  (distance_after_gas : ℕ) 
  (h1 : distance_before_gas = 32)
  (h2 : distance_after_gas = 46) :
  distance_before_gas + distance_after_gas = 78 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_concert_l3132_313291


namespace NUMINAMATH_CALUDE_chocolates_distribution_l3132_313280

theorem chocolates_distribution (total_chocolates : ℕ) (total_children : ℕ) (boys : ℕ) (girls : ℕ) 
  (chocolates_per_girl : ℕ) (h1 : total_chocolates = 3000) (h2 : total_children = 120) 
  (h3 : boys = 60) (h4 : girls = 60) (h5 : chocolates_per_girl = 3) 
  (h6 : total_children = boys + girls) : 
  (total_chocolates - girls * chocolates_per_girl) / boys = 47 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_distribution_l3132_313280


namespace NUMINAMATH_CALUDE_question_mark_value_l3132_313200

theorem question_mark_value : ∃ (x : ℕ), x * 240 = 347 * 480 ∧ x = 694 := by
  sorry

end NUMINAMATH_CALUDE_question_mark_value_l3132_313200


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3132_313247

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem line_perpendicular_to_plane 
  (l m : Line) (α : Plane) : 
  perpendicular l α → parallel l m → perpendicular m α := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3132_313247


namespace NUMINAMATH_CALUDE_total_oranges_picked_l3132_313236

theorem total_oranges_picked (mary_oranges jason_oranges amanda_oranges : ℕ)
  (h1 : mary_oranges = 14)
  (h2 : jason_oranges = 41)
  (h3 : amanda_oranges = 56) :
  mary_oranges + jason_oranges + amanda_oranges = 111 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_picked_l3132_313236


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3132_313220

theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) 
  (h1 : principal = 900) 
  (h2 : interest_paid = 729) : ∃ (rate : ℝ), 
  interest_paid = principal * rate * rate / 100 ∧ rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3132_313220


namespace NUMINAMATH_CALUDE_find_a_empty_solution_set_l3132_313235

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem for part (1)
theorem find_a : 
  ∀ a : ℝ, (∀ x : ℝ, f a (2*x) ≤ 4 ↔ 0 ≤ x ∧ x ≤ 4) → a = 4 := by sorry

-- Theorem for part (2)
theorem empty_solution_set (m : ℝ) : 
  (∀ x : ℝ, ¬(f 4 x + f 4 (x + m) < 2)) ↔ (m ≥ 2 ∨ m ≤ -2) := by sorry

end NUMINAMATH_CALUDE_find_a_empty_solution_set_l3132_313235


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3132_313210

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3132_313210


namespace NUMINAMATH_CALUDE_square_side_length_l3132_313277

theorem square_side_length (rectangle_width rectangle_length : ℝ) 
  (h1 : rectangle_width = 6)
  (h2 : rectangle_length = 24)
  (h3 : rectangle_width > 0)
  (h4 : rectangle_length > 0) :
  ∃ (square_side : ℝ), 
    square_side ^ 2 = rectangle_width * rectangle_length ∧ 
    square_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3132_313277


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3132_313266

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  -- Length of edge AB
  ab_length : ℝ
  -- Length of edge CD
  cd_length : ℝ
  -- Distance between lines AB and CD
  line_distance : ℝ
  -- Angle between lines AB and CD
  line_angle : ℝ

/-- Calculates the volume of the tetrahedron -/
def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 1/2 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    ab_length := 1,
    cd_length := Real.sqrt 3,
    line_distance := 2,
    line_angle := π / 3
  }
  tetrahedron_volume t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3132_313266


namespace NUMINAMATH_CALUDE_car_replacement_cost_l3132_313262

/-- Given an old car worth $20,000 sold at 80% of its value and a new car with
    a sticker price of $30,000 bought at 90% of its value, prove that the
    difference in cost (out of pocket) is $11,000. -/
theorem car_replacement_cost (old_car_value : ℝ) (new_car_price : ℝ)
    (old_car_sale_percentage : ℝ) (new_car_buy_percentage : ℝ)
    (h1 : old_car_value = 20000)
    (h2 : new_car_price = 30000)
    (h3 : old_car_sale_percentage = 0.8)
    (h4 : new_car_buy_percentage = 0.9) :
    new_car_buy_percentage * new_car_price - old_car_sale_percentage * old_car_value = 11000 :=
by sorry

end NUMINAMATH_CALUDE_car_replacement_cost_l3132_313262
