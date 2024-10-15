import Mathlib

namespace NUMINAMATH_CALUDE_stamp_sale_difference_l1352_135261

def red_stamps : ℕ := 30
def white_stamps : ℕ := 80
def red_stamp_price : ℚ := 50 / 100
def white_stamp_price : ℚ := 20 / 100

theorem stamp_sale_difference :
  white_stamps * white_stamp_price - red_stamps * red_stamp_price = 1 := by sorry

end NUMINAMATH_CALUDE_stamp_sale_difference_l1352_135261


namespace NUMINAMATH_CALUDE_M_characterization_l1352_135237

def M (m : ℝ) : Set ℝ := {x | x^2 - m*x + 6 = 0}

def valid_set (S : Set ℝ) : Prop :=
  S = {2, 3} ∨ S = {1, 6} ∨ S = ∅

def valid_m (m : ℝ) : Prop :=
  m = 7 ∨ m = 5 ∨ (m > -2*Real.sqrt 6 ∧ m < 2*Real.sqrt 6)

theorem M_characterization (m : ℝ) :
  (M m ∩ {1, 2, 3, 6} = M m) →
  (valid_set (M m) ∧ valid_m m) :=
sorry

end NUMINAMATH_CALUDE_M_characterization_l1352_135237


namespace NUMINAMATH_CALUDE_device_usage_probability_l1352_135225

theorem device_usage_probability (pA pB pC : ℝ) 
  (hA : pA = 0.4) 
  (hB : pB = 0.5) 
  (hC : pC = 0.7) 
  (hpA : 0 ≤ pA ∧ pA ≤ 1) 
  (hpB : 0 ≤ pB ∧ pB ≤ 1) 
  (hpC : 0 ≤ pC ∧ pC ≤ 1) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.91 := by
  sorry

end NUMINAMATH_CALUDE_device_usage_probability_l1352_135225


namespace NUMINAMATH_CALUDE_fraction_of_loss_example_l1352_135283

/-- Calculates the fraction of loss given the cost price and selling price -/
def fractionOfLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice

/-- Theorem: The fraction of loss for an item with cost price 18 and selling price 17 is 1/18 -/
theorem fraction_of_loss_example : fractionOfLoss 18 17 = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_loss_example_l1352_135283


namespace NUMINAMATH_CALUDE_exactly_one_prob_l1352_135291

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.4

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.5

/-- The events A and B are independent -/
axiom independent : True

/-- The probability that exactly one of A or B occurs -/
def prob_exactly_one : ℝ := (1 - prob_A) * prob_B + prob_A * (1 - prob_B)

/-- Theorem: The probability that exactly one of A or B occurs is 0.5 -/
theorem exactly_one_prob : prob_exactly_one = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_prob_l1352_135291


namespace NUMINAMATH_CALUDE_train_combined_speed_l1352_135200

/-- The combined speed of two trains moving in opposite directions -/
theorem train_combined_speed 
  (train1_length : ℝ) 
  (train1_time : ℝ) 
  (train2_speed : ℝ) 
  (h1 : train1_length = 180) 
  (h2 : train1_time = 12) 
  (h3 : train2_speed = 30) : 
  train1_length / train1_time + train2_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_combined_speed_l1352_135200


namespace NUMINAMATH_CALUDE_option_B_more_cost_effective_l1352_135266

/-- The cost function for Option A -/
def cost_A (x : ℝ) : ℝ := 60 + 18 * x

/-- The cost function for Option B -/
def cost_B (x : ℝ) : ℝ := 150 + 15 * x

/-- Theorem: Option B is more cost-effective for 40 kilograms of blueberries -/
theorem option_B_more_cost_effective :
  cost_B 40 < cost_A 40 := by
  sorry

end NUMINAMATH_CALUDE_option_B_more_cost_effective_l1352_135266


namespace NUMINAMATH_CALUDE_benny_cards_l1352_135215

theorem benny_cards (added_cards : ℕ) (remaining_cards : ℕ) : 
  added_cards = 4 →
  remaining_cards = 34 →
  ∃ (initial_cards : ℕ),
    initial_cards + added_cards = 2 * remaining_cards ∧
    initial_cards = 64 := by
  sorry

end NUMINAMATH_CALUDE_benny_cards_l1352_135215


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l1352_135251

theorem no_real_solution_for_log_equation :
  ∀ x : ℝ, ¬(Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 8*x + 15)) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l1352_135251


namespace NUMINAMATH_CALUDE_distance_from_apex_l1352_135280

/-- A right octagonal pyramid with two parallel cross sections -/
structure OctagonalPyramid where
  /-- Area of the smaller cross section in square feet -/
  area_small : ℝ
  /-- Area of the larger cross section in square feet -/
  area_large : ℝ
  /-- Distance between the two cross sections in feet -/
  distance_between : ℝ

/-- Theorem about the distance of the larger cross section from the apex -/
theorem distance_from_apex (pyramid : OctagonalPyramid)
  (h_area_small : pyramid.area_small = 256 * Real.sqrt 2)
  (h_area_large : pyramid.area_large = 576 * Real.sqrt 2)
  (h_distance : pyramid.distance_between = 10) :
  ∃ (d : ℝ), d = 30 ∧ d > 0 ∧ 
  d * d * pyramid.area_small = (d - pyramid.distance_between) * (d - pyramid.distance_between) * pyramid.area_large :=
sorry

end NUMINAMATH_CALUDE_distance_from_apex_l1352_135280


namespace NUMINAMATH_CALUDE_conference_arrangements_l1352_135260

/-- The number of lecturers at the conference -/
def total_lecturers : ℕ := 8

/-- The number of lecturers with specific ordering requirements -/
def ordered_lecturers : ℕ := 3

/-- Calculate the number of permutations for the remaining lecturers -/
def remaining_permutations : ℕ := (total_lecturers - ordered_lecturers).factorial

/-- Calculate the number of ways to arrange the ordered lecturers -/
def ordered_arrangements : ℕ := (total_lecturers - 2) * (total_lecturers - 1) * total_lecturers

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := ordered_arrangements * remaining_permutations

theorem conference_arrangements :
  total_arrangements = 40320 := by sorry

end NUMINAMATH_CALUDE_conference_arrangements_l1352_135260


namespace NUMINAMATH_CALUDE_intersection_M_N_l1352_135271

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1352_135271


namespace NUMINAMATH_CALUDE_family_suitcases_l1352_135281

theorem family_suitcases (num_siblings : ℕ) (suitcases_per_sibling : ℕ) (parent_suitcases : ℕ) : 
  num_siblings = 4 →
  suitcases_per_sibling = 2 →
  parent_suitcases = 3 →
  num_siblings * suitcases_per_sibling + parent_suitcases * 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_family_suitcases_l1352_135281


namespace NUMINAMATH_CALUDE_expression_evaluation_l1352_135202

theorem expression_evaluation : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1352_135202


namespace NUMINAMATH_CALUDE_arith_geom_seq_iff_not_squarefree_l1352_135262

/-- A sequence in ℤ/mℤ is both arithmetic and geometric progression -/
def is_arith_geom_seq (m : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∃ (a d r : ℕ), ∀ n : ℕ,
    (seq n) % m = (a + n * d) % m ∧
    (seq n) % m = (a * r^n) % m

/-- A sequence is nonconstant -/
def is_nonconstant (m : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∃ i j : ℕ, (seq i) % m ≠ (seq j) % m

/-- m is not squarefree -/
def not_squarefree (m : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ p^2 ∣ m

/-- Main theorem -/
theorem arith_geom_seq_iff_not_squarefree (m : ℕ) :
  (∃ seq : ℕ → ℕ, is_arith_geom_seq m seq ∧ is_nonconstant m seq) ↔ not_squarefree m :=
sorry

end NUMINAMATH_CALUDE_arith_geom_seq_iff_not_squarefree_l1352_135262


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1352_135227

theorem triangle_abc_properties (a b c A B C : ℝ) (h1 : c * Real.cos A = 5) 
  (h2 : a * Real.sin C = 4) (h3 : (1/2) * a * b * Real.sin C = 16) : 
  c = Real.sqrt 41 ∧ a + b + c = 13 + Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1352_135227


namespace NUMINAMATH_CALUDE_no_cube_sum_three_consecutive_squares_l1352_135290

theorem no_cube_sum_three_consecutive_squares :
  ¬ ∃ (x y : ℤ), x^3 = (y-1)^2 + y^2 + (y+1)^2 :=
by sorry

end NUMINAMATH_CALUDE_no_cube_sum_three_consecutive_squares_l1352_135290


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1352_135272

/-- The probability of rolling an even number on a fair 6-sided die -/
def prob_even : ℚ := 1/2

/-- The probability of rolling an odd number less than 3 on a fair 6-sided die -/
def prob_odd_lt_3 : ℚ := 1/6

/-- The number of ways to arrange two even numbers and one odd number -/
def num_arrangements : ℕ := 3

theorem dice_roll_probability :
  num_arrangements * (prob_even^2 * prob_odd_lt_3) = 1/8 := by
sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1352_135272


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_main_theorem_l1352_135245

/-- Represents a tetrahedron A-BCD with specific properties -/
structure Tetrahedron where
  /-- Base BCD is an equilateral triangle with side length 2 -/
  base_side_length : ℝ
  base_is_equilateral : base_side_length = 2
  /-- Projection of vertex A onto base BCD is the center of triangle BCD -/
  vertex_projection_is_center : Bool
  /-- E is the midpoint of side BC -/
  e_is_midpoint : Bool
  /-- Sine of angle formed by line AE with base BCD is 2√2 -/
  sine_angle_ae_base : ℝ
  sine_angle_ae_base_value : sine_angle_ae_base = 2 * Real.sqrt 2

/-- The surface area of the circumscribed sphere of the tetrahedron is 6π -/
theorem circumscribed_sphere_surface_area (t : Tetrahedron) : ℝ := by
  sorry

/-- Main theorem: The surface area of the circumscribed sphere is 6π -/
theorem main_theorem (t : Tetrahedron) : circumscribed_sphere_surface_area t = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_main_theorem_l1352_135245


namespace NUMINAMATH_CALUDE_arithmetic_seq_2016_l1352_135213

/-- An arithmetic sequence with common difference 2 and a_2007 = 2007 -/
def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  (∀ n m : ℕ, a (n + 1) - a n = 2) ∧ 
  (a 2007 = 2007)

theorem arithmetic_seq_2016 (a : ℕ → ℕ) (h : arithmetic_seq a) : 
  a 2016 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_2016_l1352_135213


namespace NUMINAMATH_CALUDE_chess_game_players_l1352_135270

def number_of_players : ℕ := 15
def total_games : ℕ := 105

theorem chess_game_players :
  ∃ k : ℕ,
    k > 0 ∧
    k < number_of_players ∧
    (number_of_players.choose k) = total_games ∧
    k = 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_players_l1352_135270


namespace NUMINAMATH_CALUDE_book_selection_theorem_l1352_135220

theorem book_selection_theorem :
  let mystery_count : ℕ := 4
  let fantasy_count : ℕ := 3
  let biography_count : ℕ := 3
  let different_genre_pairs : ℕ := 
    mystery_count * fantasy_count + 
    mystery_count * biography_count + 
    fantasy_count * biography_count
  different_genre_pairs = 33 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l1352_135220


namespace NUMINAMATH_CALUDE_probability_of_stopping_is_43_103_l1352_135233

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightCycle where
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the probability of stopping at a traffic light -/
def probabilityOfStopping (cycle : TrafficLightCycle) : ℚ :=
  let totalCycleTime := cycle.red + cycle.green + cycle.yellow
  let stoppingTime := cycle.red + cycle.yellow
  stoppingTime / totalCycleTime

/-- The specific traffic light cycle in the problem -/
def problemCycle : TrafficLightCycle :=
  { red := 40, green := 60, yellow := 3 }

theorem probability_of_stopping_is_43_103 :
  probabilityOfStopping problemCycle = 43 / 103 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_stopping_is_43_103_l1352_135233


namespace NUMINAMATH_CALUDE_simplify_fraction_l1352_135218

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2 + 1) = 16250 / 601 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1352_135218


namespace NUMINAMATH_CALUDE_triangle_isosceles_l1352_135203

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_angle : 0 < A ∧ A < π

-- Define the theorem
theorem triangle_isosceles (t : Triangle) 
  (h : Real.log (t.a^2) = Real.log (t.b^2) + Real.log (t.c^2) - Real.log (2 * t.b * t.c * Real.cos t.A)) :
  t.a = t.b ∨ t.a = t.c := by
  sorry


end NUMINAMATH_CALUDE_triangle_isosceles_l1352_135203


namespace NUMINAMATH_CALUDE_system_solution_l1352_135249

theorem system_solution :
  ∀ (x y : ℝ),
    (x * y^2 = 15 * x^2 + 17 * x * y + 15 * y^2) ∧
    (x^2 * y = 20 * x^2 + 3 * y^2) →
    ((x = 0 ∧ y = 0) ∨ (x = -19 ∧ y = -2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1352_135249


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1352_135239

/-- Given a geometric sequence with first term a₁ and common ratio q,
    if the sum of the first two terms S₂ = 3a₁, then q = 2 -/
theorem geometric_sequence_common_ratio (a₁ : ℝ) (q : ℝ) :
  a₁ ≠ 0 → a₁ + a₁ * q = 3 * a₁ → q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1352_135239


namespace NUMINAMATH_CALUDE_prob_four_green_marbles_l1352_135248

def total_marbles : ℕ := 15
def green_marbles : ℕ := 10
def purple_marbles : ℕ := 5
def total_draws : ℕ := 8
def green_draws : ℕ := 4

theorem prob_four_green_marbles :
  (Nat.choose total_draws green_draws : ℚ) *
  (green_marbles / total_marbles : ℚ) ^ green_draws *
  (purple_marbles / total_marbles : ℚ) ^ (total_draws - green_draws) =
  1120 / 6561 := by sorry

end NUMINAMATH_CALUDE_prob_four_green_marbles_l1352_135248


namespace NUMINAMATH_CALUDE_car_speed_problem_l1352_135250

theorem car_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 450 ∧ original_time = 6 ∧ new_time_factor = 3/2 →
  (distance / (original_time * new_time_factor)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1352_135250


namespace NUMINAMATH_CALUDE_angle_ABH_measure_l1352_135299

/-- A regular octagon is a polygon with 8 sides of equal length and 8 equal angles. -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- The measure of an angle in a regular octagon in degrees. -/
def regular_octagon_angle : ℝ := 135

/-- The measure of angle ABH in a regular octagon ABCDEFGH in degrees. -/
def angle_ABH (octagon : RegularOctagon) : ℝ := 22.5

/-- Theorem: In a regular octagon ABCDEFGH, the measure of angle ABH is 22.5°. -/
theorem angle_ABH_measure (octagon : RegularOctagon) :
  angle_ABH octagon = 22.5 := by sorry

end NUMINAMATH_CALUDE_angle_ABH_measure_l1352_135299


namespace NUMINAMATH_CALUDE_geometric_sequence_and_sum_l1352_135265

-- Define the geometric sequence a_n
def a (n : ℕ) : ℝ := 2 * 3^(n - 1)

-- Define the arithmetic sequence c_n
def c (n : ℕ) : ℝ := 2 * n + 2

-- Define the sequence b_n
def b (n : ℕ) : ℝ := c n - a n

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℝ := n^2 + 3*n - 3^n + 1

theorem geometric_sequence_and_sum :
  (∀ n, a (n + 1) / a n > 1) ∧  -- Common ratio > 1
  a 2 = 6 ∧
  a 1 + a 2 + a 3 = 26 ∧
  (∀ n, c n = a n + b n) ∧
  (∀ n, c (n + 1) - c n = c 2 - c 1) ∧  -- c_n is arithmetic
  b 1 = a 1 ∧
  b 3 = -10 →
  (∀ n, a n = 2 * 3^(n - 1)) ∧
  (∀ n, S n = n^2 + 3*n - 3^n + 1) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_and_sum_l1352_135265


namespace NUMINAMATH_CALUDE_room_length_calculation_l1352_135292

theorem room_length_calculation (area : ℝ) (width : ℝ) (length : ℝ) :
  area = 10 ∧ width = 2 ∧ area = length * width → length = 5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l1352_135292


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l1352_135289

theorem gcd_from_lcm_and_ratio (A B : ℕ+) : 
  Nat.lcm A B = 200 → A * 5 = B * 2 → Nat.gcd A B = 20 := by
  sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l1352_135289


namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_fourth_l1352_135278

theorem sin_alpha_plus_pi_fourth (α : Real) 
  (h1 : α > 0) (h2 : α < Real.pi / 2) (h3 : Real.sin (2 * α) = 1 / 2) : 
  Real.sin (α + Real.pi / 4) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_fourth_l1352_135278


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l1352_135224

/-- Given a car traveling for two hours with a speed of 98 km/h in the first hour
    and an average speed of 84 km/h over the two hours,
    prove that the speed of the car in the second hour is 70 km/h. -/
theorem car_speed_second_hour :
  let speed_first_hour : ℝ := 98
  let average_speed : ℝ := 84
  let total_time : ℝ := 2
  let speed_second_hour : ℝ := (average_speed * total_time) - speed_first_hour
  speed_second_hour = 70 := by
sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l1352_135224


namespace NUMINAMATH_CALUDE_magnet_to_stuffed_animals_ratio_l1352_135231

-- Define the cost of the magnet
def magnet_cost : ℚ := 3

-- Define the cost of a single stuffed animal
def stuffed_animal_cost : ℚ := 6

-- Define the combined cost of two stuffed animals
def two_stuffed_animals_cost : ℚ := 2 * stuffed_animal_cost

-- Theorem stating the ratio of magnet cost to combined stuffed animals cost
theorem magnet_to_stuffed_animals_ratio :
  magnet_cost / two_stuffed_animals_cost = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_magnet_to_stuffed_animals_ratio_l1352_135231


namespace NUMINAMATH_CALUDE_largest_s_value_l1352_135228

theorem largest_s_value : ∃ (s : ℝ), 
  (∀ (t : ℝ), (15 * t^2 - 40 * t + 18) / (4 * t - 3) + 6 * t = 7 * t - 1 → t ≤ s) ∧
  (15 * s^2 - 40 * s + 18) / (4 * s - 3) + 6 * s = 7 * s - 1 ∧
  s = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_s_value_l1352_135228


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1352_135238

theorem polynomial_factorization (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3) * (8*x^2 + x - 3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1352_135238


namespace NUMINAMATH_CALUDE_profit_percentage_example_l1352_135244

/-- Calculate the profit percentage given selling price and cost price -/
def profit_percentage (selling_price : ℚ) (cost_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The profit percentage is 25% when the selling price is 400 and the cost price is 320 -/
theorem profit_percentage_example : profit_percentage 400 320 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_example_l1352_135244


namespace NUMINAMATH_CALUDE_point_not_in_third_quadrant_l1352_135247

/-- A point P(x, y) on the line y = -x + 1 cannot be in the third quadrant -/
theorem point_not_in_third_quadrant (x y : ℝ) (h : y = -x + 1) :
  ¬(x < 0 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_third_quadrant_l1352_135247


namespace NUMINAMATH_CALUDE_equation_solution_range_l1352_135243

theorem equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 1 ∧ (2 * x + m) / (x - 1) = 1) → 
  (m ≤ -1 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l1352_135243


namespace NUMINAMATH_CALUDE_integer_solutions_l1352_135230

def satisfies_inequalities (x : ℤ) : Prop :=
  (x + 8 : ℚ) / (x + 2 : ℚ) > 2 ∧ Real.log (x - 1 : ℝ) < 1

theorem integer_solutions :
  {x : ℤ | satisfies_inequalities x} = {2, 3} := by sorry

end NUMINAMATH_CALUDE_integer_solutions_l1352_135230


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l1352_135210

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l1352_135210


namespace NUMINAMATH_CALUDE_fraction_inequality_l1352_135208

theorem fraction_inequality (a b : ℝ) : ¬(∀ a b, a / b = (a + 1) / (b + 1)) :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1352_135208


namespace NUMINAMATH_CALUDE_largest_expression_l1352_135232

theorem largest_expression (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  (a + b) ≥ max (2 * Real.sqrt (a * b)) (max (a^2 + b^2) (2 * a * b)) := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l1352_135232


namespace NUMINAMATH_CALUDE_rational_roots_of_quadratic_l1352_135277

theorem rational_roots_of_quadratic 
  (p q n : ℚ) : 
  ∃ (x : ℚ), (p + q + n) * x^2 - 2*(p + q) * x + (p + q - n) = 0 :=
by sorry

end NUMINAMATH_CALUDE_rational_roots_of_quadratic_l1352_135277


namespace NUMINAMATH_CALUDE_five_distinct_dice_probability_l1352_135255

def standard_dice_sides : ℕ := 6

def distinct_rolls (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | k + 1 => (standard_dice_sides - k) * distinct_rolls k

theorem five_distinct_dice_probability : 
  (distinct_rolls 5 : ℚ) / (standard_dice_sides ^ 5) = 5 / 54 := by
  sorry

end NUMINAMATH_CALUDE_five_distinct_dice_probability_l1352_135255


namespace NUMINAMATH_CALUDE_tangent_slope_three_points_l1352_135275

theorem tangent_slope_three_points (x : ℝ) :
  (3 * x^2 = 3) → (x = 1 ∨ x = -1) := by sorry

#check tangent_slope_three_points

end NUMINAMATH_CALUDE_tangent_slope_three_points_l1352_135275


namespace NUMINAMATH_CALUDE_number_categorization_l1352_135205

def given_numbers : List ℚ := [-10, 2/3, 0, -0.6, 4, -4 - 2/7]

def positive_numbers (numbers : List ℚ) : Set ℚ :=
  {x | x ∈ numbers ∧ x > 0}

def negative_numbers (numbers : List ℚ) : Set ℚ :=
  {x | x ∈ numbers ∧ x < 0}

def integer_numbers (numbers : List ℚ) : Set ℚ :=
  {x | x ∈ numbers ∧ ∃ n : ℤ, x = n}

def negative_fractions (numbers : List ℚ) : Set ℚ :=
  {x | x ∈ numbers ∧ x < 0 ∧ ¬∃ n : ℤ, x = n}

theorem number_categorization :
  positive_numbers given_numbers = {2/3, 4} ∧
  negative_numbers given_numbers = {-10, -0.6, -4 - 2/7} ∧
  integer_numbers given_numbers = {-10, 0, 4} ∧
  negative_fractions given_numbers = {-0.6, -4 - 2/7} := by
  sorry

end NUMINAMATH_CALUDE_number_categorization_l1352_135205


namespace NUMINAMATH_CALUDE_first_half_chop_count_l1352_135282

/-- The number of trees that need to be planted for each tree chopped down -/
def replantRatio : ℕ := 3

/-- The number of trees chopped down in the second half of the year -/
def secondHalfChop : ℕ := 300

/-- The total number of trees that need to be planted -/
def totalPlant : ℕ := 1500

/-- The number of trees chopped down in the first half of the year -/
def firstHalfChop : ℕ := (totalPlant - replantRatio * secondHalfChop) / replantRatio

theorem first_half_chop_count : firstHalfChop = 200 := by
  sorry

end NUMINAMATH_CALUDE_first_half_chop_count_l1352_135282


namespace NUMINAMATH_CALUDE_yellow_beans_percentage_approx_32_percent_l1352_135296

def bag1_total : ℕ := 24
def bag2_total : ℕ := 32
def bag3_total : ℕ := 34

def bag1_yellow_percent : ℚ := 40 / 100
def bag2_yellow_percent : ℚ := 30 / 100
def bag3_yellow_percent : ℚ := 25 / 100

def total_beans : ℕ := bag1_total + bag2_total + bag3_total

def yellow_beans : ℚ := 
  bag1_total * bag1_yellow_percent + 
  bag2_total * bag2_yellow_percent + 
  bag3_total * bag3_yellow_percent

def mixed_yellow_percent : ℚ := yellow_beans / total_beans

theorem yellow_beans_percentage_approx_32_percent :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |mixed_yellow_percent - 32/100| < ε :=
sorry

end NUMINAMATH_CALUDE_yellow_beans_percentage_approx_32_percent_l1352_135296


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l1352_135253

theorem bobby_candy_problem (total_candy : ℕ) (chocolate_eaten : ℕ) (gummy_eaten : ℕ)
  (h1 : total_candy = 36)
  (h2 : chocolate_eaten = 12)
  (h3 : gummy_eaten = 9)
  (h4 : chocolate_eaten = 2 * (chocolate_eaten + (total_candy - chocolate_eaten - gummy_eaten)) / 3)
  (h5 : gummy_eaten = 3 * (gummy_eaten + (total_candy - chocolate_eaten - gummy_eaten)) / 4) :
  total_candy - chocolate_eaten - gummy_eaten = 9 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l1352_135253


namespace NUMINAMATH_CALUDE_younger_son_future_age_l1352_135207

def age_difference : ℕ := 10
def elder_son_current_age : ℕ := 40
def years_in_future : ℕ := 30

theorem younger_son_future_age :
  let younger_son_current_age := elder_son_current_age - age_difference
  younger_son_current_age + years_in_future = 60 := by sorry

end NUMINAMATH_CALUDE_younger_son_future_age_l1352_135207


namespace NUMINAMATH_CALUDE_jellybean_count_is_84_l1352_135269

/-- Calculates the final number of jellybeans in a jar after a series of actions. -/
def final_jellybean_count (initial : ℕ) (samantha_took : ℕ) (shelby_ate : ℕ) : ℕ :=
  let remaining_after_samantha := initial - samantha_took
  let remaining_after_shelby := remaining_after_samantha - shelby_ate
  let scarlett_returned := shelby_ate
  let shannon_added := (samantha_took + shelby_ate) / 2
  remaining_after_shelby + scarlett_returned + shannon_added

/-- Theorem stating that given the initial conditions, the final number of jellybeans is 84. -/
theorem jellybean_count_is_84 :
  final_jellybean_count 90 24 12 = 84 := by
  sorry

#eval final_jellybean_count 90 24 12

end NUMINAMATH_CALUDE_jellybean_count_is_84_l1352_135269


namespace NUMINAMATH_CALUDE_cosine_value_l1352_135284

theorem cosine_value (α : Real) 
  (h : Real.sin (π / 6 - α) = 5 / 13) : 
  Real.cos (π / 3 + α) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_l1352_135284


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_conditions_l1352_135217

theorem three_digit_numbers_with_conditions :
  ∃ (a b : ℕ),
    100 ≤ a ∧ a < b ∧ b < 1000 ∧
    ∃ (k : ℕ), a + b = 498 * k ∧
    ∃ (m : ℕ), b = 5 * m * a ∧
    a = 166 ∧ b = 830 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_conditions_l1352_135217


namespace NUMINAMATH_CALUDE_some_number_equation_l1352_135264

theorem some_number_equation (x : ℤ) : |x - 8*(3 - 12)| - |5 - 11| = 70 ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_some_number_equation_l1352_135264


namespace NUMINAMATH_CALUDE_third_restaurant_meals_l1352_135256

/-- The number of meals served by Gordon's third restaurant per day -/
def third_restaurant_meals_per_day (
  total_restaurants : ℕ)
  (first_restaurant_meals_per_day : ℕ)
  (second_restaurant_meals_per_day : ℕ)
  (total_meals_per_week : ℕ) : ℕ :=
  (total_meals_per_week - 7 * (first_restaurant_meals_per_day + second_restaurant_meals_per_day)) / 7

theorem third_restaurant_meals (
  total_restaurants : ℕ)
  (first_restaurant_meals_per_day : ℕ)
  (second_restaurant_meals_per_day : ℕ)
  (total_meals_per_week : ℕ)
  (h1 : total_restaurants = 3)
  (h2 : first_restaurant_meals_per_day = 20)
  (h3 : second_restaurant_meals_per_day = 40)
  (h4 : total_meals_per_week = 770) :
  third_restaurant_meals_per_day total_restaurants first_restaurant_meals_per_day second_restaurant_meals_per_day total_meals_per_week = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_third_restaurant_meals_l1352_135256


namespace NUMINAMATH_CALUDE_expression_value_l1352_135226

theorem expression_value : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1352_135226


namespace NUMINAMATH_CALUDE_inequality_proof_l1352_135236

theorem inequality_proof (a b c d k : ℝ) 
  (h1 : |k| < 2) 
  (h2 : a^2 + b^2 - k*a*b = 1) 
  (h3 : c^2 + d^2 - k*c*d = 1) : 
  |a*c - b*d| ≤ 2 / Real.sqrt (4 - k^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1352_135236


namespace NUMINAMATH_CALUDE_blue_marbles_count_l1352_135252

/-- The number of blue marbles Jason has -/
def jason_blue_marbles : ℕ := 44

/-- The number of blue marbles Tom has -/
def tom_blue_marbles : ℕ := 24

/-- The total number of blue marbles Jason and Tom have together -/
def total_blue_marbles : ℕ := jason_blue_marbles + tom_blue_marbles

theorem blue_marbles_count : total_blue_marbles = 68 := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l1352_135252


namespace NUMINAMATH_CALUDE_monotone_function_k_range_l1352_135267

/-- Given a function f(x) = e^x + kx - ln x that is monotonically increasing on (1, +∞),
    prove that k ∈ [1-e, +∞) -/
theorem monotone_function_k_range (k : ℝ) :
  (∀ x > 1, Monotone (fun x => Real.exp x + k * x - Real.log x)) →
  k ∈ Set.Ici (1 - Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_monotone_function_k_range_l1352_135267


namespace NUMINAMATH_CALUDE_hostel_provisions_theorem_l1352_135288

/-- The number of days provisions last for the initial group -/
def initial_days : ℕ := 50

/-- The number of days provisions last with 20 fewer people -/
def extended_days : ℕ := 250

/-- The number of fewer people in the extended scenario -/
def fewer_people : ℕ := 20

/-- The function to calculate the daily consumption rate given the number of people and days -/
def daily_consumption_rate (people : ℕ) (days : ℕ) : ℚ :=
  1 / (people.cast * days.cast)

theorem hostel_provisions_theorem (initial_girls : ℕ) :
  (daily_consumption_rate initial_girls initial_days =
   daily_consumption_rate (initial_girls + fewer_people) extended_days) →
  initial_girls = 25 := by
  sorry

end NUMINAMATH_CALUDE_hostel_provisions_theorem_l1352_135288


namespace NUMINAMATH_CALUDE_dans_age_l1352_135212

theorem dans_age (dans_present_age : ℕ) : dans_present_age = 6 :=
  by
  have h : dans_present_age + 18 = 8 * (dans_present_age - 3) :=
    by sorry
  
  sorry

end NUMINAMATH_CALUDE_dans_age_l1352_135212


namespace NUMINAMATH_CALUDE_remainder_of_large_sum_l1352_135211

theorem remainder_of_large_sum (n : ℕ) : (7 * 10^20 + 2^20) % 11 = 9 :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_large_sum_l1352_135211


namespace NUMINAMATH_CALUDE_none_of_statements_true_l1352_135258

theorem none_of_statements_true (s x y : ℝ) 
  (h_s : s > 1) 
  (h_xy : x^2 * y ≠ 0) 
  (h_ineq : x * s^2 > y * s^2) : 
  ¬(-x > -y) ∧ ¬(-x > y) ∧ ¬(1 > -y/x) ∧ ¬(1 < y/x) := by
sorry

end NUMINAMATH_CALUDE_none_of_statements_true_l1352_135258


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_a_eq_neg_two_l1352_135273

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)

/-- Theorem: If z(a) is purely imaginary, then a = -2 -/
theorem purely_imaginary_implies_a_eq_neg_two :
  ∀ a : ℝ, isPurelyImaginary (z a) → a = -2 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_a_eq_neg_two_l1352_135273


namespace NUMINAMATH_CALUDE_power_of_ten_multiplication_l1352_135259

theorem power_of_ten_multiplication (a b : ℕ) : (10 : ℝ) ^ a * (10 : ℝ) ^ b = (10 : ℝ) ^ (a + b) := by
  sorry

end NUMINAMATH_CALUDE_power_of_ten_multiplication_l1352_135259


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1352_135234

theorem greatest_divisor_with_remainders :
  ∃ (d : ℕ), d > 0 ∧
    (150 % d = 50) ∧
    (230 % d = 5) ∧
    (175 % d = 25) ∧
    (∀ (k : ℕ), k > 0 →
      (150 % k = 50) →
      (230 % k = 5) →
      (175 % k = 25) →
      k ≤ d) ∧
    d = 25 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1352_135234


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l1352_135257

/-- Given two cylinders with the following properties:
  * S₁ and S₂ are their base areas
  * υ₁ and υ₂ are their volumes
  * They have equal lateral areas
  * S₁/S₂ = 16/9
Then υ₁/υ₂ = 4/3 -/
theorem cylinder_volume_ratio (S₁ S₂ υ₁ υ₂ : ℝ) (h_positive : S₁ > 0 ∧ S₂ > 0 ∧ υ₁ > 0 ∧ υ₂ > 0)
    (h_base_ratio : S₁ / S₂ = 16 / 9) (h_equal_lateral : ∃ (r₁ r₂ h₁ h₂ : ℝ), 
    S₁ = π * r₁^2 ∧ S₂ = π * r₂^2 ∧ υ₁ = S₁ * h₁ ∧ υ₂ = S₂ * h₂ ∧ 2 * π * r₁ * h₁ = 2 * π * r₂ * h₂) :
  υ₁ / υ₂ = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l1352_135257


namespace NUMINAMATH_CALUDE_fourth_power_nested_roots_l1352_135235

theorem fourth_power_nested_roots : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 2)))^4 = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_nested_roots_l1352_135235


namespace NUMINAMATH_CALUDE_sum_of_products_l1352_135279

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 250)
  (h2 : a + b + c = 16) :
  a*b + b*c + c*a = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_products_l1352_135279


namespace NUMINAMATH_CALUDE_point_symmetric_range_l1352_135294

/-- 
Given a point P(a+1, 2a-3) that is symmetric about the x-axis and lies in the first quadrant,
prove that the range of a is (-1, 3/2).
-/
theorem point_symmetric_range (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = a + 1 ∧ P.2 = 2*a - 3 ∧ P.1 > 0 ∧ P.2 > 0) ↔ 
  -1 < a ∧ a < 3/2 := by sorry

end NUMINAMATH_CALUDE_point_symmetric_range_l1352_135294


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l1352_135274

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l1352_135274


namespace NUMINAMATH_CALUDE_contrapositive_proof_l1352_135222

theorem contrapositive_proof : 
  (∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ 
  (∀ a b : ℝ, a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_proof_l1352_135222


namespace NUMINAMATH_CALUDE_giraffe_ratio_l1352_135263

theorem giraffe_ratio (total_giraffes : ℕ) (difference : ℕ) : 
  total_giraffes = 300 →
  difference = 290 →
  total_giraffes = (total_giraffes - difference) + difference →
  (total_giraffes : ℚ) / (total_giraffes - difference) = 30 := by
  sorry

end NUMINAMATH_CALUDE_giraffe_ratio_l1352_135263


namespace NUMINAMATH_CALUDE_number_equation_solution_l1352_135201

theorem number_equation_solution : 
  ∃ x : ℝ, (45 + 3 * x = 72) ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1352_135201


namespace NUMINAMATH_CALUDE_complex_quadrant_l1352_135209

theorem complex_quadrant (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l1352_135209


namespace NUMINAMATH_CALUDE_distance_between_cars_l1352_135268

theorem distance_between_cars (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) :
  initial_distance = 105 →
  car1_distance = 50 →
  car2_distance = 35 →
  initial_distance - (car1_distance + car2_distance) = 20 := by
sorry

end NUMINAMATH_CALUDE_distance_between_cars_l1352_135268


namespace NUMINAMATH_CALUDE_volume_problem_l1352_135276

/-- Given a volume that is the product of three numbers, where two of the numbers are 18 and 6,
    and 48 cubes of edge 3 can be inserted into this volume, prove that the first number in the product is 12. -/
theorem volume_problem (volume : ℝ) (first_number : ℝ) : 
  volume = first_number * 18 * 6 →
  volume = 48 * (3 : ℝ)^3 →
  first_number = 12 := by
  sorry

end NUMINAMATH_CALUDE_volume_problem_l1352_135276


namespace NUMINAMATH_CALUDE_vector_problem_l1352_135295

theorem vector_problem (a b : ℝ × ℝ) :
  a + b = (2, 3) → a - b = (-2, 1) → a - 2 * b = (-4, 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l1352_135295


namespace NUMINAMATH_CALUDE_lines_perpendicular_l1352_135216

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the two lines
def line1 (t : Triangle) (x y : Real) : Prop :=
  x * Real.sin t.A + t.a * y + t.c = 0

def line2 (t : Triangle) (x y : Real) : Prop :=
  t.b * x - y * Real.sin t.B + Real.sin t.C = 0

-- Theorem statement
theorem lines_perpendicular (t : Triangle) : 
  (∀ x y, line1 t x y → line2 t x y → False) ∨ 
  (∃ x y, line1 t x y ∧ line2 t x y) :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_l1352_135216


namespace NUMINAMATH_CALUDE_tangent_line_is_correct_l1352_135246

/-- The equation of a circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The point on the circle -/
def point_on_circle : ℝ × ℝ := (4, 1)

/-- The proposed tangent line equation -/
def tangent_line_equation (x y : ℝ) : Prop := 3*x + 4*y - 16 = 0 ∨ x = 4

/-- Theorem stating that the proposed equation represents the tangent line -/
theorem tangent_line_is_correct :
  tangent_line_equation (point_on_circle.1) (point_on_circle.2) ∧
  ∀ (x y : ℝ), circle_equation x y →
    tangent_line_equation x y →
    (x, y) = point_on_circle ∨
    ∃ (t : ℝ), (x, y) = (point_on_circle.1 + t, point_on_circle.2 + t) ∧ t ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_is_correct_l1352_135246


namespace NUMINAMATH_CALUDE_taco_truck_beef_per_taco_l1352_135240

theorem taco_truck_beef_per_taco 
  (total_beef : ℝ) 
  (selling_price : ℝ) 
  (cost_per_taco : ℝ) 
  (total_profit : ℝ) 
  (h1 : total_beef = 100)
  (h2 : selling_price = 2)
  (h3 : cost_per_taco = 1.5)
  (h4 : total_profit = 200) :
  ∃ (beef_per_taco : ℝ), 
    beef_per_taco = 1/4 ∧ 
    (total_beef / beef_per_taco) * (selling_price - cost_per_taco) = total_profit := by
  sorry

end NUMINAMATH_CALUDE_taco_truck_beef_per_taco_l1352_135240


namespace NUMINAMATH_CALUDE_subset_relation_l1352_135242

theorem subset_relation (M N : Set ℕ) : 
  M = {1, 2, 3, 4} → N = {2, 3, 4} → N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_subset_relation_l1352_135242


namespace NUMINAMATH_CALUDE_interlaced_roots_l1352_135241

/-- A quadratic function -/
def QuadraticFunction := ℝ → ℝ

/-- Predicate to check if two real numbers are distinct roots of a quadratic function -/
def are_distinct_roots (f : QuadraticFunction) (x y : ℝ) : Prop :=
  x ≠ y ∧ f x = 0 ∧ f y = 0

/-- Predicate to check if four real numbers are interlaced -/
def are_interlaced (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (x₁ < x₃ ∧ x₃ < x₂ ∧ x₂ < x₄) ∨ (x₃ < x₁ ∧ x₁ < x₄ ∧ x₄ < x₂)

theorem interlaced_roots 
  (f g : QuadraticFunction) (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : are_distinct_roots f x₁ x₂)
  (h₂ : are_distinct_roots g x₃ x₄)
  (h₃ : g x₁ * g x₂ < 0) :
  are_interlaced x₁ x₂ x₃ x₄ :=
sorry

end NUMINAMATH_CALUDE_interlaced_roots_l1352_135241


namespace NUMINAMATH_CALUDE_joey_age_when_beth_was_joeys_current_age_l1352_135229

/-- Represents a person's age at different points in time -/
structure Person where
  current_age : ℕ
  future_age : ℕ
  past_age : ℕ

/-- Given the conditions of the problem, prove that Joey was 4 years old when Beth was Joey's current age -/
theorem joey_age_when_beth_was_joeys_current_age 
  (joey : Person) 
  (beth : Person)
  (h1 : joey.current_age = 9)
  (h2 : joey.future_age = beth.current_age)
  (h3 : joey.future_age = joey.current_age + 5) :
  joey.past_age = 4 := by
sorry

end NUMINAMATH_CALUDE_joey_age_when_beth_was_joeys_current_age_l1352_135229


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l1352_135214

theorem complex_power_magnitude : 
  Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l1352_135214


namespace NUMINAMATH_CALUDE_power_comparison_l1352_135223

theorem power_comparison : 3^17 < 8^9 ∧ 8^9 < 4^15 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l1352_135223


namespace NUMINAMATH_CALUDE_sqrt_abs_equation_solution_l1352_135219

theorem sqrt_abs_equation_solution :
  ∀ x y : ℝ, Real.sqrt (2 * x + 3 * y) + |x + 3| = 0 → x = -3 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_equation_solution_l1352_135219


namespace NUMINAMATH_CALUDE_expand_expression_l1352_135297

theorem expand_expression (x : ℝ) : (7*x^2 + 5*x + 8) * 3*x = 21*x^3 + 15*x^2 + 24*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1352_135297


namespace NUMINAMATH_CALUDE_parabola_directrix_l1352_135293

/-- Given a parabola with equation y² = 6x, its directrix equation is x = -3/2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y^2 = 6*x) → (∃ (k : ℝ), k = -3/2 ∧ x = k) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1352_135293


namespace NUMINAMATH_CALUDE_local_minimum_of_f_l1352_135206

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- State the theorem
theorem local_minimum_of_f :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x ≥ f x₀) ∧ x₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_of_f_l1352_135206


namespace NUMINAMATH_CALUDE_hank_reads_seven_days_a_week_l1352_135221

/-- Represents Hank's reading habits and total reading time in a week -/
structure ReadingHabits where
  weekdayReadingTime : ℕ  -- Daily reading time on weekdays in minutes
  weekendReadingTime : ℕ  -- Daily reading time on weekends in minutes
  totalWeeklyTime : ℕ     -- Total reading time in a week in minutes

/-- Calculates the number of days Hank reads in a week based on his reading habits -/
def daysReadingPerWeek (habits : ReadingHabits) : ℕ :=
  if (5 * habits.weekdayReadingTime + 2 * habits.weekendReadingTime) = habits.totalWeeklyTime
  then 7
  else 0

/-- Theorem stating that Hank reads 7 days a week given his reading habits -/
theorem hank_reads_seven_days_a_week :
  let habits : ReadingHabits := {
    weekdayReadingTime := 90,
    weekendReadingTime := 180,
    totalWeeklyTime := 810
  }
  daysReadingPerWeek habits = 7 := by sorry

end NUMINAMATH_CALUDE_hank_reads_seven_days_a_week_l1352_135221


namespace NUMINAMATH_CALUDE_expression_simplification_l1352_135287

theorem expression_simplification :
  0.7264 * 0.4329 + 0.1235 * 0.3412 + 0.1289 * 0.5634 - 0.3785 * 0.4979 = 0.2407 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1352_135287


namespace NUMINAMATH_CALUDE_umbrella_count_l1352_135298

theorem umbrella_count (y b r : ℕ) 
  (h1 : b = (y + r) / 2)
  (h2 : r = (y + b) / 3)
  (h3 : y = 45) :
  b = 36 ∧ r = 27 := by
  sorry

end NUMINAMATH_CALUDE_umbrella_count_l1352_135298


namespace NUMINAMATH_CALUDE_shrub_height_after_two_years_l1352_135204

def shrub_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

theorem shrub_height_after_two_years 
  (h : shrub_height (shrub_height 9 2) 3 = 243) : 
  shrub_height 9 2 = 9 :=
by
  sorry

#check shrub_height_after_two_years

end NUMINAMATH_CALUDE_shrub_height_after_two_years_l1352_135204


namespace NUMINAMATH_CALUDE_tetrahedral_die_expected_steps_l1352_135286

def expected_steps (n : Nat) : ℚ :=
  match n with
  | 1 => 1
  | 2 => 5/4
  | 3 => 25/16
  | 4 => 125/64
  | _ => 0

theorem tetrahedral_die_expected_steps :
  let total_expectation := 1 + (expected_steps 1 + expected_steps 2 + expected_steps 3 + expected_steps 4) / 4
  total_expectation = 625/256 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedral_die_expected_steps_l1352_135286


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1352_135254

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a > 2 → a ∈ Set.Ici 2) ∧ (∃ x, x ∈ Set.Ici 2 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1352_135254


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1352_135285

/-- Given an arrangement of four congruent rectangles around an inner square,
    prove that the ratio of the longer side to the shorter side of each rectangle is 2 -/
theorem rectangle_ratio (s : ℝ) (x y : ℝ) : 
  s > 0 →  -- inner square side length is positive
  x > 0 ∧ y > 0 →  -- rectangle sides are positive
  s + 2*y = 3*s →  -- outer square side length
  x + s = 3*s →  -- outer square side length (alternate direction)
  (3*s)^2 = 9*s^2 →  -- area relation
  x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1352_135285
