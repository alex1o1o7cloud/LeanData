import Mathlib

namespace NUMINAMATH_CALUDE_ariel_fish_count_l903_90322

theorem ariel_fish_count (total : ℕ) (male_fraction : ℚ) (female_count : ℕ) : 
  total = 45 → 
  male_fraction = 2/3 → 
  female_count = total - (total * male_fraction).num → 
  female_count = 15 :=
by sorry

end NUMINAMATH_CALUDE_ariel_fish_count_l903_90322


namespace NUMINAMATH_CALUDE_base_r_is_seven_l903_90351

/-- Represents a number in base r --/
def BaseR (n : ℕ) (r : ℕ) : ℕ → ℕ
| 0 => 0
| (k+1) => (n % r) * r^k + BaseR (n / r) r k

/-- The equation representing the transaction in base r --/
def TransactionEquation (r : ℕ) : Prop :=
  BaseR 210 r 2 + BaseR 260 r 2 = BaseR 500 r 2

theorem base_r_is_seven :
  ∃ r : ℕ, r > 1 ∧ TransactionEquation r ∧ r = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_r_is_seven_l903_90351


namespace NUMINAMATH_CALUDE_train_journey_time_l903_90361

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (4 / 5 * usual_speed) * (usual_time + 3 / 4) = usual_speed * usual_time → 
  usual_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l903_90361


namespace NUMINAMATH_CALUDE_log_base_value_l903_90330

theorem log_base_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = Real.log x / Real.log a) →  -- Definition of f as logarithm base a
  a > 0 →                                     -- Condition: a > 0
  a ≠ 1 →                                     -- Condition: a ≠ 1
  f 9 = 2 →                                   -- Condition: f(9) = 2
  a = 3 :=                                    -- Conclusion: a = 3
by sorry

end NUMINAMATH_CALUDE_log_base_value_l903_90330


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l903_90344

theorem power_tower_mod_500 : 5^(5^(5^5)) ≡ 125 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l903_90344


namespace NUMINAMATH_CALUDE_prob_at_least_one_consonant_l903_90386

def word : String := "barkhint"

def is_consonant (c : Char) : Bool :=
  c ∈ ['b', 'r', 'k', 'h', 'n', 't']

def num_letters : Nat := word.length

def num_vowels : Nat := word.toList.filter (fun c => !is_consonant c) |>.length

def num_ways_to_select_two : Nat := num_letters * (num_letters - 1) / 2

def num_ways_to_select_two_vowels : Nat := num_vowels * (num_vowels - 1) / 2

theorem prob_at_least_one_consonant :
  (1 : ℚ) - (num_ways_to_select_two_vowels : ℚ) / num_ways_to_select_two = 27 / 28 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_consonant_l903_90386


namespace NUMINAMATH_CALUDE_fish_upstream_speed_l903_90349

/-- The upstream speed of a fish given its downstream speed and speed in still water -/
theorem fish_upstream_speed (downstream_speed still_water_speed : ℝ) :
  downstream_speed = 55 →
  still_water_speed = 45 →
  still_water_speed - (downstream_speed - still_water_speed) = 35 := by
  sorry

#check fish_upstream_speed

end NUMINAMATH_CALUDE_fish_upstream_speed_l903_90349


namespace NUMINAMATH_CALUDE_volleyball_tournament_games_l903_90337

/-- The number of games played in a volleyball tournament -/
def tournament_games (n : ℕ) (g : ℕ) : ℕ :=
  (n * (n - 1) * g) / 2

/-- Theorem: A volleyball tournament with 10 teams, where each team plays 4 games
    with every other team, has a total of 180 games. -/
theorem volleyball_tournament_games :
  tournament_games 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_games_l903_90337


namespace NUMINAMATH_CALUDE_cube_side_area_l903_90334

/-- Given a cube with a total surface area of 54.3 square centimeters,
    the area of one side is 9.05 square centimeters. -/
theorem cube_side_area (total_area : ℝ) (h1 : total_area = 54.3) : ∃ (side_area : ℝ), 
  side_area = 9.05 ∧ 6 * side_area = total_area := by
  sorry

end NUMINAMATH_CALUDE_cube_side_area_l903_90334


namespace NUMINAMATH_CALUDE_sum_of_square_roots_geq_one_l903_90305

theorem sum_of_square_roots_geq_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a / (a + 3 * b)) + Real.sqrt (b / (b + 3 * a)) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_geq_one_l903_90305


namespace NUMINAMATH_CALUDE_order_of_trig_values_l903_90353

theorem order_of_trig_values :
  let a := Real.tan (70 * π / 180)
  let b := Real.sin (25 * π / 180)
  let c := Real.cos (25 * π / 180)
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_order_of_trig_values_l903_90353


namespace NUMINAMATH_CALUDE_ral_to_suri_age_ratio_l903_90307

def suri_future_age : ℕ := 16
def years_to_future : ℕ := 3
def ral_current_age : ℕ := 26

def suri_current_age : ℕ := suri_future_age - years_to_future

theorem ral_to_suri_age_ratio :
  ral_current_age / suri_current_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_ral_to_suri_age_ratio_l903_90307


namespace NUMINAMATH_CALUDE_always_odd_l903_90367

theorem always_odd (a b c : ℕ+) (ha : a.val % 2 = 1) (hb : b.val % 2 = 1) :
  (3^a.val + (b.val - 1)^2 * c.val) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l903_90367


namespace NUMINAMATH_CALUDE_farmer_rabbit_problem_l903_90313

theorem farmer_rabbit_problem :
  ∀ (initial_rabbits : ℕ),
    (∃ (rabbits_per_cage : ℕ),
      initial_rabbits + 6 = 17 * rabbits_per_cage) →
    initial_rabbits = 28 := by
  sorry

end NUMINAMATH_CALUDE_farmer_rabbit_problem_l903_90313


namespace NUMINAMATH_CALUDE_complex_sum_equals_two_l903_90318

theorem complex_sum_equals_two (z : ℂ) (h : z = Complex.exp (2 * Real.pi * I / 5)) :
  z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_two_l903_90318


namespace NUMINAMATH_CALUDE_talitha_took_108_pieces_l903_90392

/-- Given an initial candy count, the number of pieces Solomon took, and the final candy count,
    calculate the number of pieces Talitha took. -/
def talitha_candy_count (initial : ℕ) (solomon_took : ℕ) (final : ℕ) : ℕ :=
  initial - solomon_took - final

/-- Theorem stating that Talitha took 108 pieces of candy. -/
theorem talitha_took_108_pieces :
  talitha_candy_count 349 153 88 = 108 := by
  sorry

end NUMINAMATH_CALUDE_talitha_took_108_pieces_l903_90392


namespace NUMINAMATH_CALUDE_grid_routes_3x2_l903_90377

theorem grid_routes_3x2 :
  let total_moves : ℕ := 3 + 2
  let right_moves : ℕ := 3
  let down_moves : ℕ := 2
  let num_routes : ℕ := Nat.choose total_moves down_moves
  num_routes = 10 := by sorry

end NUMINAMATH_CALUDE_grid_routes_3x2_l903_90377


namespace NUMINAMATH_CALUDE_smallest_number_l903_90319

def base_6_to_decimal (x : ℕ) : ℕ := x

def base_4_to_decimal (x : ℕ) : ℕ := x

def base_2_to_decimal (x : ℕ) : ℕ := x

theorem smallest_number 
  (h1 : base_6_to_decimal 210 = 78)
  (h2 : base_4_to_decimal 100 = 16)
  (h3 : base_2_to_decimal 111111 = 63) :
  base_4_to_decimal 100 < base_6_to_decimal 210 ∧ 
  base_4_to_decimal 100 < base_2_to_decimal 111111 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l903_90319


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l903_90339

/-- A convex nonagon is a 9-sided polygon -/
def ConvexNonagon : Type := Unit

/-- The number of sides in a convex nonagon -/
def num_sides (n : ConvexNonagon) : ℕ := 9

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals (n : ConvexNonagon) : ℕ := 27

theorem nonagon_diagonals (n : ConvexNonagon) : 
  num_diagonals n = (num_sides n * (num_sides n - 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l903_90339


namespace NUMINAMATH_CALUDE_park_warden_citations_l903_90368

theorem park_warden_citations :
  ∀ (littering off_leash parking : ℕ),
    littering = off_leash →
    parking = 2 * (littering + off_leash) →
    littering + off_leash + parking = 24 →
    littering = 4 := by
  sorry

end NUMINAMATH_CALUDE_park_warden_citations_l903_90368


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l903_90306

theorem circle_line_intersection_range (k : ℝ) : 
  (∃ (a b : ℝ), (b = k * a - 2) ∧ 
    (∃ (x y : ℝ), (x^2 + y^2 + 8*x + 15 = 0) ∧ 
      ((x - a)^2 + (y - b)^2 = 1))) →
  -4/3 ≤ k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l903_90306


namespace NUMINAMATH_CALUDE_farm_bird_difference_l903_90315

/-- Given a farm with chickens, ducks, and geese, calculate the difference between
    the combined number of chickens and geese and the number of ducks. -/
theorem farm_bird_difference (chickens ducks geese : ℕ) : 
  chickens = 42 →
  ducks = 48 →
  geese = chickens →
  chickens + geese - ducks = 36 := by
sorry

end NUMINAMATH_CALUDE_farm_bird_difference_l903_90315


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l903_90302

theorem simplify_trig_expression :
  (Real.sin (10 * π / 180) + Real.sin (20 * π / 180)) /
  (Real.cos (10 * π / 180) + Real.cos (20 * π / 180)) =
  Real.tan (15 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l903_90302


namespace NUMINAMATH_CALUDE_Q_3_volume_l903_90335

/-- Recursive definition of the volume of Qᵢ -/
def Q_volume : ℕ → ℚ
  | 0 => 1
  | n + 1 => Q_volume n + 4 * 4^n * (1 / 27)^(n + 1)

/-- The volume of Q₃ is 73/81 -/
theorem Q_3_volume : Q_volume 3 = 73 / 81 := by
  sorry

end NUMINAMATH_CALUDE_Q_3_volume_l903_90335


namespace NUMINAMATH_CALUDE_min_abs_ab_for_perpendicular_lines_l903_90310

theorem min_abs_ab_for_perpendicular_lines (a b : ℝ) : 
  (∀ x y : ℝ, x + a^2 * y + 1 = 0 ∧ (a^2 + 1) * x - b * y + 3 = 0 → 
    (1 : ℝ) + a^2 * (-b) = 0) → 
  ∃ m : ℝ, m = 2 ∧ ∀ k : ℝ, k = |a * b| → k ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_abs_ab_for_perpendicular_lines_l903_90310


namespace NUMINAMATH_CALUDE_one_carton_per_case_l903_90398

/-- Given a case containing cartons, each carton containing b boxes,
    each box containing 400 paper clips, and 800 paper clips in 2 cases,
    prove that there is 1 carton in a case. -/
theorem one_carton_per_case (b : ℕ) (h1 : b ≥ 1) :
  ∃ (c : ℕ), c = 1 ∧ 2 * c * b * 400 = 800 := by
  sorry

#check one_carton_per_case

end NUMINAMATH_CALUDE_one_carton_per_case_l903_90398


namespace NUMINAMATH_CALUDE_fruits_left_l903_90379

def fruits_problem (plums guavas apples given_away : ℕ) : ℕ :=
  (plums + guavas + apples) - given_away

theorem fruits_left (plums guavas apples given_away : ℕ) 
  (h : given_away ≤ plums + guavas + apples) : 
  fruits_problem plums guavas apples given_away = 
  (plums + guavas + apples) - given_away :=
by
  sorry

end NUMINAMATH_CALUDE_fruits_left_l903_90379


namespace NUMINAMATH_CALUDE_spider_sock_shoe_arrangements_l903_90347

/-- The number of legs the spider has -/
def num_legs : ℕ := 10

/-- The total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- The number of valid arrangements for putting on socks and shoes -/
def valid_arrangements : ℕ := (Nat.factorial total_items) / (2^num_legs)

/-- Theorem stating the number of valid arrangements for the spider to put on socks and shoes -/
theorem spider_sock_shoe_arrangements :
  valid_arrangements = (Nat.factorial total_items) / (2^num_legs) :=
sorry

end NUMINAMATH_CALUDE_spider_sock_shoe_arrangements_l903_90347


namespace NUMINAMATH_CALUDE_star_calculation_l903_90393

def star (x y : ℤ) : ℤ := x * y - 1

theorem star_calculation : (star (star 2 3) 4) = 19 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l903_90393


namespace NUMINAMATH_CALUDE_trees_planted_specific_plot_l903_90311

/-- Calculates the number of trees planted around a rectangular plot -/
def trees_planted (length width spacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let total_intervals := perimeter / spacing
  total_intervals - 4

/-- Theorem stating the number of trees planted around the specific rectangular plot -/
theorem trees_planted_specific_plot :
  trees_planted 60 30 6 = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_trees_planted_specific_plot_l903_90311


namespace NUMINAMATH_CALUDE_tobys_sharing_l903_90321

theorem tobys_sharing (initial_amount : ℚ) (remaining_amount : ℚ) (num_brothers : ℕ) :
  initial_amount = 343 →
  remaining_amount = 245 →
  num_brothers = 2 →
  (initial_amount - remaining_amount) / (num_brothers * initial_amount) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tobys_sharing_l903_90321


namespace NUMINAMATH_CALUDE_stream_speed_l903_90396

theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 14 →
  distance = 4864 →
  total_time = 700 →
  (distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed) = total_time) →
  stream_speed = 1.2 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l903_90396


namespace NUMINAMATH_CALUDE_motorcycle_vs_car_profit_difference_l903_90300

/-- Represents the production and sales data for a vehicle type -/
structure VehicleData where
  materialCost : ℕ
  quantity : ℕ
  price : ℕ

/-- Calculates the profit for a given vehicle type -/
def profit (data : VehicleData) : ℤ :=
  (data.quantity * data.price) - data.materialCost

/-- Theorem: The difference in profit between motorcycle and car production is $50 -/
theorem motorcycle_vs_car_profit_difference :
  let carData : VehicleData := ⟨100, 4, 50⟩
  let motorcycleData : VehicleData := ⟨250, 8, 50⟩
  profit motorcycleData - profit carData = 50 := by
  sorry

#eval profit ⟨250, 8, 50⟩ - profit ⟨100, 4, 50⟩

end NUMINAMATH_CALUDE_motorcycle_vs_car_profit_difference_l903_90300


namespace NUMINAMATH_CALUDE_smallest_initial_value_l903_90323

theorem smallest_initial_value : 
  ∃ (x : ℕ), x + 42 = 456 ∧ ∀ (y : ℕ), y + 42 = 456 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_initial_value_l903_90323


namespace NUMINAMATH_CALUDE_lava_lamp_probability_is_one_seventh_l903_90375

/-- The probability of a specific arrangement of lava lamps -/
def lava_lamp_probability : ℚ :=
  let total_lamps : ℕ := 4 + 4  -- 4 red + 4 blue lamps
  let lamps_on : ℕ := 4  -- 4 lamps are turned on
  let remaining_lamps : ℕ := total_lamps - 2  -- excluding leftmost and rightmost
  let remaining_on : ℕ := lamps_on - 1  -- excluding the rightmost lamp which is on
  let favorable_arrangements : ℕ := Nat.choose remaining_lamps (total_lamps / 2 - 1)  -- arranging remaining red lamps
  let favorable_on_choices : ℕ := Nat.choose (total_lamps - 1) (lamps_on - 1)  -- choosing remaining on lamps
  let total_arrangements : ℕ := Nat.choose total_lamps (total_lamps / 2)  -- total ways to arrange red and blue lamps
  let total_on_choices : ℕ := Nat.choose total_lamps lamps_on  -- total ways to choose on lamps
  (favorable_arrangements * favorable_on_choices : ℚ) / (total_arrangements * total_on_choices)

/-- The probability of the specific lava lamp arrangement is 1/7 -/
theorem lava_lamp_probability_is_one_seventh : lava_lamp_probability = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_is_one_seventh_l903_90375


namespace NUMINAMATH_CALUDE_square_area_l903_90324

/-- A square with a circle tangent to three sides and passing through the diagonal midpoint -/
structure SquareWithCircle where
  s : ℝ  -- side length of the square
  r : ℝ  -- radius of the circle
  s_pos : 0 < s  -- side length is positive
  r_pos : 0 < r  -- radius is positive
  tangent_condition : s = 4 * r  -- derived from the tangent and midpoint conditions

/-- The area of the square is 16r^2 -/
theorem square_area (config : SquareWithCircle) : config.s^2 = 16 * config.r^2 := by
  sorry

#check square_area

end NUMINAMATH_CALUDE_square_area_l903_90324


namespace NUMINAMATH_CALUDE_expected_red_balls_l903_90382

/-- The number of red balls in the bag -/
def red_balls : ℕ := 4

/-- The number of white balls in the bag -/
def white_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := red_balls + white_balls

/-- The probability of drawing a red ball in a single draw -/
def p_red : ℚ := red_balls / total_balls

/-- The number of draws -/
def num_draws : ℕ := 6

/-- The random variable representing the number of red balls drawn -/
def ξ : ℕ → ℚ := sorry

/-- The expected value of ξ -/
def E_ξ : ℚ := num_draws * p_red

theorem expected_red_balls : E_ξ = 4 := by sorry

end NUMINAMATH_CALUDE_expected_red_balls_l903_90382


namespace NUMINAMATH_CALUDE_least_k_factorial_divisible_by_315_l903_90360

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

theorem least_k_factorial_divisible_by_315 :
  ∀ k : ℕ, k > 1 → (factorial k) % 315 = 0 → k ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_least_k_factorial_divisible_by_315_l903_90360


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l903_90384

/-- A geometric sequence with sum of first n terms S_n -/
structure GeometricSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given conditions for the geometric sequence -/
def given_sequence : GeometricSequence where
  S := fun n => 
    if n = 2 then 6
    else if n = 4 then 18
    else 0  -- We only know S_2 and S_4, other values are placeholders

theorem geometric_sequence_sum (seq : GeometricSequence) :
  seq.S 2 = 6 → seq.S 4 = 18 → seq.S 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l903_90384


namespace NUMINAMATH_CALUDE_min_students_all_correct_l903_90303

theorem min_students_all_correct (total_students : ℕ) 
  (q1_correct q2_correct q3_correct q4_correct : ℕ) 
  (h1 : total_students = 45)
  (h2 : q1_correct = 35)
  (h3 : q2_correct = 27)
  (h4 : q3_correct = 41)
  (h5 : q4_correct = 38) :
  total_students - (total_students - q1_correct) - 
  (total_students - q2_correct) - (total_students - q3_correct) - 
  (total_students - q4_correct) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_students_all_correct_l903_90303


namespace NUMINAMATH_CALUDE_cheerful_not_green_l903_90387

-- Define the universe of snakes
variable (Snake : Type)

-- Define properties of snakes
variable (green : Snake → Prop)
variable (cheerful : Snake → Prop)
variable (can_sing : Snake → Prop)
variable (can_multiply : Snake → Prop)

-- Define the conditions
axiom all_cheerful_can_sing : ∀ s : Snake, cheerful s → can_sing s
axiom no_green_can_multiply : ∀ s : Snake, green s → ¬can_multiply s
axiom cannot_multiply_cannot_sing : ∀ s : Snake, ¬can_multiply s → ¬can_sing s

-- Theorem to prove
theorem cheerful_not_green : ∀ s : Snake, cheerful s → ¬green s := by
  sorry

end NUMINAMATH_CALUDE_cheerful_not_green_l903_90387


namespace NUMINAMATH_CALUDE_opposite_face_of_B_is_H_l903_90308

-- Define a cube type
structure Cube where
  faces : Fin 6 → Char

-- Define the set of valid labels
def ValidLabels : Set Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Define a property that all faces have valid labels
def has_valid_labels (c : Cube) : Prop :=
  ∀ i : Fin 6, c.faces i ∈ ValidLabels

-- Define a property that all faces are unique
def has_unique_faces (c : Cube) : Prop :=
  ∀ i j : Fin 6, i ≠ j → c.faces i ≠ c.faces j

-- Define the theorem
theorem opposite_face_of_B_is_H (c : Cube) 
  (h1 : has_valid_labels c) 
  (h2 : has_unique_faces c) 
  (h3 : ∃ i : Fin 6, c.faces i = 'B') : 
  ∃ j : Fin 6, c.faces j = 'H' ∧ 
  (∀ k : Fin 6, c.faces k = 'B' → k.val + j.val = 5) :=
sorry

end NUMINAMATH_CALUDE_opposite_face_of_B_is_H_l903_90308


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l903_90343

/-- The number of flavors in the ice cream shop -/
def F : ℕ := sorry

/-- The number of flavors Gretchen tried two years ago -/
def tried_two_years_ago : ℚ := F / 4

/-- The number of flavors Gretchen tried last year -/
def tried_last_year : ℚ := 2 * tried_two_years_ago

/-- The number of flavors Gretchen still needs to try this year -/
def flavors_left : ℕ := 25

theorem ice_cream_flavors :
  F = 100 ∧
  tried_two_years_ago = F / 4 ∧
  tried_last_year = 2 * tried_two_years_ago ∧
  flavors_left = 25 ∧
  F = (tried_two_years_ago + tried_last_year + flavors_left) :=
sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l903_90343


namespace NUMINAMATH_CALUDE_polynomial_expansion_l903_90327

theorem polynomial_expansion (x : ℝ) : 
  (x^2 - 3*x + 3) * (x^2 + 3*x + 1) = x^4 - 5*x^2 + 6*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l903_90327


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l903_90309

def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ+, a (n + 1) = q * a n

def satisfies_condition (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n * a (n + 3) = a (n + 1) * a (n + 2)

theorem geometric_sequence_property :
  (∀ a : ℕ+ → ℝ, is_geometric_sequence a → satisfies_condition a) ∧
  (∃ a : ℕ+ → ℝ, satisfies_condition a ∧ ¬is_geometric_sequence a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l903_90309


namespace NUMINAMATH_CALUDE_vitamin_pack_size_l903_90394

theorem vitamin_pack_size (vitamin_a_pack_size : ℕ) 
  (vitamin_a_packs : ℕ) (vitamin_d_packs : ℕ) : 
  (vitamin_a_pack_size * vitamin_a_packs = 17 * vitamin_d_packs) →  -- Equal quantities condition
  (vitamin_a_pack_size * vitamin_a_packs = 119) →                   -- Smallest number condition
  (∀ x y : ℕ, x * y = 119 → x ≤ vitamin_a_pack_size ∨ y ≤ vitamin_a_packs) →  -- Smallest positive integer values
  vitamin_a_pack_size = 7 :=
by sorry

end NUMINAMATH_CALUDE_vitamin_pack_size_l903_90394


namespace NUMINAMATH_CALUDE_wrapping_paper_area_formula_l903_90380

/-- Represents a rectangular box with length, width, and height. -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- Calculates the area of wrapping paper needed to wrap a box. -/
def wrappingPaperArea (box : Box) : ℝ :=
  6 * box.length * box.height + 2 * box.width * box.height

/-- Theorem stating that the wrapping paper area for a box is 6lh + 2wh. -/
theorem wrapping_paper_area_formula (box : Box) :
  wrappingPaperArea box = 6 * box.length * box.height + 2 * box.width * box.height :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_formula_l903_90380


namespace NUMINAMATH_CALUDE_suki_coffee_bags_suki_coffee_bags_proof_l903_90301

theorem suki_coffee_bags (suki_bag_weight jimmy_bag_weight container_weight : ℕ)
                         (jimmy_bags : ℚ)
                         (num_containers : ℕ)
                         (suki_bags : ℕ) : Prop :=
  suki_bag_weight = 22 →
  jimmy_bag_weight = 18 →
  jimmy_bags = 4.5 →
  container_weight = 8 →
  num_containers = 28 →
  (↑suki_bags * suki_bag_weight + jimmy_bags * jimmy_bag_weight : ℚ) = ↑(num_containers * container_weight) →
  suki_bags = 6

theorem suki_coffee_bags_proof : suki_coffee_bags 22 18 8 (4.5 : ℚ) 28 6 := by
  sorry

end NUMINAMATH_CALUDE_suki_coffee_bags_suki_coffee_bags_proof_l903_90301


namespace NUMINAMATH_CALUDE_polynomial_factorization_l903_90356

theorem polynomial_factorization :
  (∀ x : ℝ, x^2 + 14*x + 49 = (x + 7)^2) ∧
  (∀ m n : ℝ, (m - 1) + n^2*(1 - m) = (m - 1)*(1 - n)*(1 + n)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l903_90356


namespace NUMINAMATH_CALUDE_consecutive_square_roots_l903_90383

theorem consecutive_square_roots (x : ℝ) (n : ℕ) :
  (∃ m : ℕ, n = m ∧ x^2 = m) →
  Real.sqrt ((n + 1 : ℝ)) = Real.sqrt (x^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_consecutive_square_roots_l903_90383


namespace NUMINAMATH_CALUDE_min_distance_intersection_l903_90320

/-- The minimum distance between intersection points --/
theorem min_distance_intersection (m : ℝ) : 
  let f (x : ℝ) := |x - (x + Real.exp x + 3) / 2|
  ∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_intersection_l903_90320


namespace NUMINAMATH_CALUDE_circle_area_increase_l903_90333

theorem circle_area_increase (r : ℝ) : 
  let initial_area := π * r^2
  let final_area := π * (r + 3)^2
  final_area - initial_area = 6 * π * r + 9 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_increase_l903_90333


namespace NUMINAMATH_CALUDE_emerald_count_l903_90314

/-- Represents a box of gemstones -/
structure GemBox where
  count : ℕ

/-- Represents the collection of gem boxes -/
structure GemCollection where
  diamonds : Array GemBox
  rubies : Array GemBox
  emeralds : Array GemBox

/-- The theorem to be proved -/
theorem emerald_count (collection : GemCollection) : 
  collection.diamonds.size = 2 ∧ 
  collection.rubies.size = 2 ∧ 
  collection.emeralds.size = 2 ∧ 
  (collection.rubies.foldl (λ acc box => acc + box.count) 0 = 
   collection.diamonds.foldl (λ acc box => acc + box.count) 0 + 15) →
  collection.emeralds.foldl (λ acc box => acc + box.count) 0 = 12 := by
  sorry

end NUMINAMATH_CALUDE_emerald_count_l903_90314


namespace NUMINAMATH_CALUDE_not_necessarily_square_lt_of_lt_l903_90391

theorem not_necessarily_square_lt_of_lt {a b : ℝ} (h : a < b) : 
  ¬(∀ a b : ℝ, a < b → a^2 < b^2) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_square_lt_of_lt_l903_90391


namespace NUMINAMATH_CALUDE_intersection_sum_problem_l903_90325

theorem intersection_sum_problem (digits : Finset ℕ) 
  (h_digits : digits.card = 6 ∧ digits ⊆ Finset.range 10 ∧ 1 ∈ digits)
  (vertical : Finset ℕ) (horizontal : Finset ℕ)
  (h_vert : vertical.card = 3 ∧ vertical ⊆ digits)
  (h_horiz : horizontal.card = 4 ∧ horizontal ⊆ digits)
  (h_intersect : (vertical ∩ horizontal).card = 1)
  (h_vert_sum : vertical.sum id = 25)
  (h_horiz_sum : horizontal.sum id = 14) :
  digits.sum id = 31 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_problem_l903_90325


namespace NUMINAMATH_CALUDE_planes_parallel_to_line_not_necessarily_parallel_l903_90328

-- Define a 3D space
variable (V : Type) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Fact (finrank ℝ V = 3)]

-- Define planes and lines in the space
variable (Plane : Type) (Line : Type)

-- Define a relation for a plane being parallel to a line
variable (plane_parallel_to_line : Plane → Line → Prop)

-- Define a relation for two planes being parallel
variable (planes_parallel : Plane → Plane → Prop)

-- Theorem: Two planes parallel to the same line are not necessarily parallel
theorem planes_parallel_to_line_not_necessarily_parallel 
  (P1 P2 : Plane) (L : Line) 
  (h1 : plane_parallel_to_line P1 L) 
  (h2 : plane_parallel_to_line P2 L) :
  ¬ (∀ P1 P2 : Plane, ∀ L : Line, 
    plane_parallel_to_line P1 L → plane_parallel_to_line P2 L → planes_parallel P1 P2) :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_line_not_necessarily_parallel_l903_90328


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_theorem_l903_90390

-- Define the ellipse T
def ellipse_T (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the hyperbola S
def hyperbola_S (x y m n : ℝ) : Prop :=
  x^2 / m^2 - y^2 / n^2 = 1 ∧ m > 0 ∧ n > 0

-- Define the common focus
def common_focus (a b m n : ℝ) : Prop :=
  a^2 - b^2 = m^2 + n^2 ∧ a^2 - b^2 = 4

-- Define the asymptotic line l
def asymptotic_line (x y m n : ℝ) : Prop :=
  y = (n / m) * x

-- Define the symmetry condition
def symmetry_condition (a b m n : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola_S x y m n ∧
  ((x = m^2 - 2 ∧ y = m * n) ∨ (x = 4*b/5 ∧ y = 3*b/5))

-- Main theorem
theorem ellipse_hyperbola_theorem (a b m n : ℝ) :
  ellipse_T 0 b a b ∧
  hyperbola_S 2 0 m n ∧
  common_focus a b m n ∧
  (∃ (x y : ℝ), hyperbola_S x y m n ∧ asymptotic_line x y m n) ∧
  symmetry_condition a b m n →
  a^2 = 5 ∧ b^2 = 4 ∧ m^2 = 4/5 ∧ n^2 = 16/5 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_theorem_l903_90390


namespace NUMINAMATH_CALUDE_sophie_donuts_l903_90381

/-- The number of donuts left for Sophie after giving some away -/
def donuts_left (total_boxes : ℕ) (donuts_per_box : ℕ) (boxes_given : ℕ) (donuts_given : ℕ) : ℕ :=
  (total_boxes - boxes_given) * donuts_per_box - donuts_given

/-- Theorem stating that Sophie is left with 30 donuts -/
theorem sophie_donuts :
  donuts_left 4 12 1 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sophie_donuts_l903_90381


namespace NUMINAMATH_CALUDE_max_generatable_number_l903_90395

def powers_of_three : List ℕ := [1, 3, 9, 27, 81, 243, 729]

def can_generate (n : ℤ) : Prop :=
  ∃ (coeffs : List ℤ), 
    coeffs.length = powers_of_three.length ∧ 
    (∀ c ∈ coeffs, c = 1 ∨ c = 0 ∨ c = -1) ∧
    n = List.sum (List.zipWith (· * ·) coeffs (powers_of_three.map Int.ofNat))

theorem max_generatable_number :
  (∀ n : ℕ, n ≤ 1093 → can_generate n) ∧
  ¬(can_generate 1094) :=
sorry

end NUMINAMATH_CALUDE_max_generatable_number_l903_90395


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l903_90317

theorem sum_of_squares_problem (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (eq1 : a^2 + a*b + b^2 = 1)
  (eq2 : b^2 + b*c + c^2 = 3)
  (eq3 : c^2 + c*a + a^2 = 4) :
  a + b + c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l903_90317


namespace NUMINAMATH_CALUDE_negative_four_cubed_inequality_l903_90341

theorem negative_four_cubed_inequality : (-4)^3 ≠ -4^3 := by
  -- Define the left-hand side
  have h1 : (-4)^3 = (-4) * (-4) * (-4) := by sorry
  -- Define the right-hand side
  have h2 : -4^3 = -(4 * 4 * 4) := by sorry
  -- Prove the inequality
  sorry

end NUMINAMATH_CALUDE_negative_four_cubed_inequality_l903_90341


namespace NUMINAMATH_CALUDE_total_cats_l903_90388

/-- The number of cats owned by Mr. Thompson -/
def thompson_cats : ℕ := 15

/-- The number of cats owned by Mrs. Sheridan -/
def sheridan_cats : ℕ := 11

/-- The number of cats owned by Mrs. Garrett -/
def garrett_cats : ℕ := 24

/-- The number of cats owned by Mr. Ravi -/
def ravi_cats : ℕ := 18

/-- The theorem stating that the total number of cats is 68 -/
theorem total_cats : thompson_cats + sheridan_cats + garrett_cats + ravi_cats = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_l903_90388


namespace NUMINAMATH_CALUDE_triangle_with_ap_angles_and_altitudes_is_equilateral_l903_90364

/-- A triangle with angles and altitudes in arithmetic progression is equilateral -/
theorem triangle_with_ap_angles_and_altitudes_is_equilateral 
  (A B C : ℝ) (a b c : ℝ) (ha hb hc : ℝ) : 
  (∃ (d : ℝ), A = B - d ∧ C = B + d) →  -- Angles in arithmetic progression
  (A + B + C = 180) →                   -- Sum of angles in a triangle
  (ha + hc = 2 * hb) →                  -- Altitudes in arithmetic progression
  (ha = 2 * area / a) →                 -- Relation between altitude and side
  (hb = 2 * area / b) → 
  (hc = 2 * area / c) → 
  (b^2 = a^2 + c^2 - a*c) →             -- Law of cosines for 60° angle
  (a = b ∧ b = c) :=                    -- Triangle is equilateral
by sorry

end NUMINAMATH_CALUDE_triangle_with_ap_angles_and_altitudes_is_equilateral_l903_90364


namespace NUMINAMATH_CALUDE_fraction_inequality_l903_90366

theorem fraction_inequality (x : ℝ) : (x - 1) / (x + 2) ≥ 0 ↔ x < -2 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l903_90366


namespace NUMINAMATH_CALUDE_positive_xy_l903_90348

theorem positive_xy (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ -2) : x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_xy_l903_90348


namespace NUMINAMATH_CALUDE_largest_solution_quadratic_l903_90378

theorem largest_solution_quadratic (x : ℝ) : 
  (3 * (8 * x^2 + 10 * x + 8) = x * (8 * x - 34)) →
  x ≤ (-4 + Real.sqrt 10) / 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_quadratic_l903_90378


namespace NUMINAMATH_CALUDE_person_savings_l903_90358

theorem person_savings (income expenditure savings : ℕ) : 
  (income : ℚ) / expenditure = 5 / 4 →
  income = 15000 →
  savings = income - expenditure →
  savings = 3000 := by
sorry

end NUMINAMATH_CALUDE_person_savings_l903_90358


namespace NUMINAMATH_CALUDE_sophie_shopping_budget_l903_90312

def initial_budget : ℚ := 260
def shirt_cost : ℚ := 18.5
def num_shirts : ℕ := 2
def trouser_cost : ℚ := 63
def num_additional_items : ℕ := 4

theorem sophie_shopping_budget :
  let total_spent := shirt_cost * num_shirts + trouser_cost
  let remaining_budget := initial_budget - total_spent
  remaining_budget / num_additional_items = 40 := by
  sorry

end NUMINAMATH_CALUDE_sophie_shopping_budget_l903_90312


namespace NUMINAMATH_CALUDE_expand_expression_l903_90369

theorem expand_expression (x : ℝ) : (x - 3) * (4 * x + 8) = 4 * x^2 - 4 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l903_90369


namespace NUMINAMATH_CALUDE_work_completion_time_l903_90316

/-- If a group of people can complete a work in 8 days, then twice the number of people can complete half the work in 2 days. -/
theorem work_completion_time 
  (P : ℕ) -- Number of people
  (W : ℝ) -- Amount of work
  (h : P > 0) -- Assumption that there is at least one person
  (completion_time : ℝ) -- Time to complete the work
  (h_completion : completion_time = 8) -- Given that the work is completed in 8 days
  : (2 * P) * (W / 2) / W * completion_time = 2 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l903_90316


namespace NUMINAMATH_CALUDE_expression_simplification_l903_90336

theorem expression_simplification (x y : ℝ) :
  (2 * x - (3 * y - (2 * x + 1))) - ((3 * y - (2 * x + 1)) - 2 * x) = 8 * x - 6 * y + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l903_90336


namespace NUMINAMATH_CALUDE_irrationality_of_pi_l903_90329

-- Define rational numbers
def isRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrational numbers as the complement of rational numbers
def isIrrational (x : ℝ) : Prop := ¬(isRational x)

-- Theorem statement
theorem irrationality_of_pi :
  isIrrational π ∧ isRational 0 ∧ isRational (22/7) ∧ isRational (Real.rpow 8 (1/3)) := by
  sorry


end NUMINAMATH_CALUDE_irrationality_of_pi_l903_90329


namespace NUMINAMATH_CALUDE_gift_cost_theorem_l903_90352

def polo_price : ℚ := 26
def necklace_price : ℚ := 83
def game_price : ℚ := 90
def sock_price : ℚ := 7
def book_price : ℚ := 15
def scarf_price : ℚ := 22

def polo_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def game_quantity : ℕ := 1
def sock_quantity : ℕ := 4
def book_quantity : ℕ := 3
def scarf_quantity : ℕ := 2

def sales_tax_rate : ℚ := 13 / 200  -- 6.5%
def book_discount_rate : ℚ := 1 / 10  -- 10%
def rebate : ℚ := 12

def total_cost : ℚ :=
  polo_price * polo_quantity +
  necklace_price * necklace_quantity +
  game_price * game_quantity +
  sock_price * sock_quantity +
  book_price * book_quantity +
  scarf_price * scarf_quantity

def discounted_book_cost : ℚ := book_price * book_quantity * (1 - book_discount_rate)

def total_cost_after_book_discount : ℚ :=
  total_cost - (book_price * book_quantity) + discounted_book_cost

def total_cost_with_tax : ℚ :=
  total_cost_after_book_discount * (1 + sales_tax_rate)

def final_cost : ℚ := total_cost_with_tax - rebate

theorem gift_cost_theorem :
  final_cost = 46352 / 100 := by sorry

end NUMINAMATH_CALUDE_gift_cost_theorem_l903_90352


namespace NUMINAMATH_CALUDE_playground_width_l903_90372

/-- The number of playgrounds -/
def num_playgrounds : ℕ := 8

/-- The length of each playground in meters -/
def playground_length : ℝ := 300

/-- The total area of all playgrounds in square kilometers -/
def total_area_km2 : ℝ := 0.6

/-- Conversion factor from square kilometers to square meters -/
def km2_to_m2 : ℝ := 1000000

theorem playground_width :
  ∀ (width : ℝ),
  (width * playground_length * num_playgrounds = total_area_km2 * km2_to_m2) →
  width = 250 := by
sorry

end NUMINAMATH_CALUDE_playground_width_l903_90372


namespace NUMINAMATH_CALUDE_double_frosted_cubes_count_l903_90397

/-- Represents a cube with dimensions n × n × n -/
structure Cube (n : ℕ) where
  size : ℕ := n

/-- Represents a cake with frosting on top and sides, but not on bottom -/
structure FrostedCake (n : ℕ) extends Cube n where
  frosted_top : Bool := true
  frosted_sides : Bool := true
  frosted_bottom : Bool := false

/-- Counts the number of 1×1×1 cubes with exactly two frosted faces in a FrostedCake -/
def count_double_frosted_cubes (cake : FrostedCake 4) : ℕ :=
  sorry

theorem double_frosted_cubes_count :
  ∀ (cake : FrostedCake 4), count_double_frosted_cubes cake = 20 :=
by sorry

end NUMINAMATH_CALUDE_double_frosted_cubes_count_l903_90397


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l903_90342

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Define set B
def B : Set ℝ := { y | ∃ x ∈ A, y = x + 1 }

-- Statement to prove
theorem intersection_A_complement_B : A ∩ (U \ B) = Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l903_90342


namespace NUMINAMATH_CALUDE_tanner_video_game_cost_l903_90350

/-- The cost of Tanner's video game purchase -/
def video_game_cost (september_savings october_savings november_savings remaining_amount : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - remaining_amount

/-- Theorem stating the cost of Tanner's video game -/
theorem tanner_video_game_cost :
  video_game_cost 17 48 25 41 = 49 := by
  sorry

end NUMINAMATH_CALUDE_tanner_video_game_cost_l903_90350


namespace NUMINAMATH_CALUDE_fourth_group_frequency_count_l903_90373

theorem fourth_group_frequency_count 
  (f₁ f₂ f₃ : ℝ) 
  (n₁ : ℕ) 
  (h₁ : f₁ = 0.1) 
  (h₂ : f₂ = 0.3) 
  (h₃ : f₃ = 0.4) 
  (h₄ : n₁ = 5) 
  (h₅ : f₁ + f₂ + f₃ < 1) : 
  ∃ (N : ℕ) (n₄ : ℕ), 
    N > 0 ∧ 
    f₁ = n₁ / N ∧ 
    n₄ = N * (1 - (f₁ + f₂ + f₃)) ∧ 
    n₄ = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_group_frequency_count_l903_90373


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l903_90345

/-- An ellipse with foci F₁ and F₂, and constant sum of distances from any point to the foci. -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  sum_distances : ℝ

/-- The center, semi-major axis, and semi-minor axis of an ellipse. -/
structure EllipseParameters where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Theorem: For the given ellipse, h + k + a + b = 9 + √7 -/
theorem ellipse_parameter_sum (E : Ellipse) (P : EllipseParameters) :
  E.F₁ = (0, 2) →
  E.F₂ = (6, 2) →
  E.sum_distances = 8 →
  P.h + P.k + P.a + P.b = 9 + Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l903_90345


namespace NUMINAMATH_CALUDE_scientific_notation_570_million_l903_90346

theorem scientific_notation_570_million :
  (570000000 : ℝ) = 5.7 * (10 : ℝ) ^ 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_570_million_l903_90346


namespace NUMINAMATH_CALUDE_ratio_in_interval_l903_90338

theorem ratio_in_interval (a : Fin 10 → ℕ) (h : ∀ i, a i ≤ 91) :
  ∃ i j, i ≠ j ∧ 2/3 ≤ (a i : ℚ) / (a j : ℚ) ∧ (a i : ℚ) / (a j : ℚ) ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_ratio_in_interval_l903_90338


namespace NUMINAMATH_CALUDE_ap_sum_possible_n_values_l903_90399

theorem ap_sum_possible_n_values :
  let S (n : ℕ) (a : ℤ) := (n : ℤ) * (2 * a + (n - 1) * 3) / 2
  (∃! k : ℕ, k > 1 ∧ (∃ a : ℤ, S k a = 180) ∧
    ∀ m : ℕ, m > 1 → (∃ b : ℤ, S m b = 180) → m ∈ Finset.range k) :=
by sorry

end NUMINAMATH_CALUDE_ap_sum_possible_n_values_l903_90399


namespace NUMINAMATH_CALUDE_quadratic_always_positive_iff_a_in_range_l903_90332

theorem quadratic_always_positive_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + 1 > 0) ↔ -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_iff_a_in_range_l903_90332


namespace NUMINAMATH_CALUDE_obtain_x_squared_and_xy_l903_90359

/-- Given positive real numbers x and y, prove that x^2 and xy can be obtained
    using operations of addition, subtraction, multiplication, division, and reciprocal. -/
theorem obtain_x_squared_and_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ (f g : ℝ → ℝ → ℝ), f x y = x^2 ∧ g x y = x*y :=
by sorry

end NUMINAMATH_CALUDE_obtain_x_squared_and_xy_l903_90359


namespace NUMINAMATH_CALUDE_containers_used_l903_90304

def initial_balls : ℕ := 100
def balls_per_container : ℕ := 10

theorem containers_used :
  let remaining_balls := initial_balls / 2
  remaining_balls / balls_per_container = 5 := by
  sorry

end NUMINAMATH_CALUDE_containers_used_l903_90304


namespace NUMINAMATH_CALUDE_harry_age_l903_90362

/-- Given the ages of Kiarra, Bea, Job, Figaro, and Harry, prove that Harry is 26 years old. -/
theorem harry_age (kiarra bea job figaro harry : ℕ) 
  (h1 : kiarra = 2 * bea)
  (h2 : job = 3 * bea)
  (h3 : figaro = job + 7)
  (h4 : harry * 2 = figaro)
  (h5 : kiarra = 30) : 
  harry = 26 := by
  sorry

end NUMINAMATH_CALUDE_harry_age_l903_90362


namespace NUMINAMATH_CALUDE_negation_of_forall_exp_geq_x_plus_one_l903_90365

theorem negation_of_forall_exp_geq_x_plus_one :
  (¬ ∀ x : ℝ, Real.exp x ≥ x + 1) ↔ (∃ x₀ : ℝ, Real.exp x₀ < x₀ + 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_exp_geq_x_plus_one_l903_90365


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_nine_l903_90376

theorem greatest_integer_with_gcf_nine : ∃ n : ℕ, n < 200 ∧ 
  Nat.gcd n 45 = 9 ∧ 
  ∀ m : ℕ, m < 200 → Nat.gcd m 45 = 9 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_nine_l903_90376


namespace NUMINAMATH_CALUDE_fathers_age_is_32_l903_90370

/-- The present age of the father -/
def father_age : ℕ := 32

/-- The present age of the older son -/
def older_son_age : ℕ := 22

/-- The present age of the younger son -/
def younger_son_age : ℕ := 18

/-- The average age of the father and his two sons is 24 years -/
axiom average_age : (father_age + older_son_age + younger_son_age) / 3 = 24

/-- 5 years ago, the average age of the two sons was 15 years -/
axiom sons_average_age_5_years_ago : (older_son_age - 5 + younger_son_age - 5) / 2 = 15

/-- The difference between the ages of the two sons is 4 years -/
axiom sons_age_difference : older_son_age - younger_son_age = 4

/-- Theorem: Given the conditions, the father's present age is 32 years -/
theorem fathers_age_is_32 : father_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_is_32_l903_90370


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l903_90357

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 3*x - 4 < 0} = Set.Ioo (-4 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l903_90357


namespace NUMINAMATH_CALUDE_max_rabbit_population_l903_90355

/-- Represents the properties of a rabbit population --/
structure RabbitPopulation where
  total : ℕ
  longEars : ℕ
  jumpFar : ℕ
  bothTraits : ℕ

/-- Checks if a rabbit population satisfies the given conditions --/
def isValidPopulation (pop : RabbitPopulation) : Prop :=
  pop.longEars = 13 ∧
  pop.jumpFar = 17 ∧
  pop.bothTraits ≥ 3 ∧
  pop.longEars + pop.jumpFar - pop.bothTraits ≤ pop.total

/-- Theorem stating that 27 is the maximum number of rabbits satisfying the conditions --/
theorem max_rabbit_population :
  ∀ (pop : RabbitPopulation), isValidPopulation pop → pop.total ≤ 27 :=
sorry

end NUMINAMATH_CALUDE_max_rabbit_population_l903_90355


namespace NUMINAMATH_CALUDE_min_value_function_l903_90326

theorem min_value_function (a b : ℝ) (h : a + b = 1) :
  ∃ (min : ℝ), min = 5 * Real.sqrt 11 ∧
  ∀ (x y : ℝ), x + y = 1 →
  3 * Real.sqrt (1 + 2 * x^2) + 2 * Real.sqrt (40 + 9 * y^2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_function_l903_90326


namespace NUMINAMATH_CALUDE_frogs_eaten_by_fish_l903_90331

/-- The number of flies eaten by each frog per day -/
def flies_per_frog : ℕ := 30

/-- The number of fish eaten by each gharial per day -/
def fish_per_gharial : ℕ := 15

/-- The number of gharials in the swamp -/
def num_gharials : ℕ := 9

/-- The total number of flies eaten per day -/
def total_flies_eaten : ℕ := 32400

/-- The number of frogs each fish needs to eat per day -/
def frogs_per_fish : ℕ := 8

theorem frogs_eaten_by_fish :
  frogs_per_fish = 
    total_flies_eaten / (flies_per_frog * (num_gharials * fish_per_gharial)) :=
by sorry

end NUMINAMATH_CALUDE_frogs_eaten_by_fish_l903_90331


namespace NUMINAMATH_CALUDE_sin_product_equality_l903_90371

theorem sin_product_equality : (1 - Real.sin (π / 6)) * (1 - Real.sin (5 * π / 6)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equality_l903_90371


namespace NUMINAMATH_CALUDE_inequality_proof_l903_90385

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*a*c)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l903_90385


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l903_90354

theorem arccos_equation_solution :
  ∃ x : ℝ, x = Real.sqrt (1 / (64 - 36 * Real.sqrt 3)) ∧ 
    Real.arccos (3 * x) - Real.arccos x = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l903_90354


namespace NUMINAMATH_CALUDE_profit_percentage_is_twenty_l903_90363

/-- Calculates the percentage profit on wholesale price given wholesale price, retail price, and discount percentage. -/
def percentage_profit (wholesale_price retail_price discount_percent : ℚ) : ℚ :=
  let discount := discount_percent * retail_price / 100
  let selling_price := retail_price - discount
  let profit := selling_price - wholesale_price
  (profit / wholesale_price) * 100

/-- Theorem stating that given the specific values in the problem, the percentage profit is 20%. -/
theorem profit_percentage_is_twenty :
  percentage_profit 108 144 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_twenty_l903_90363


namespace NUMINAMATH_CALUDE_jellyfish_cost_l903_90340

theorem jellyfish_cost (jellyfish_cost eel_cost : ℝ) : 
  eel_cost = 9 * jellyfish_cost →
  jellyfish_cost + eel_cost = 200 →
  jellyfish_cost = 20 := by
sorry

end NUMINAMATH_CALUDE_jellyfish_cost_l903_90340


namespace NUMINAMATH_CALUDE_triangle_sides_perfect_square_l903_90389

theorem triangle_sides_perfect_square
  (a b c : ℤ)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (gcd_condition : Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs = 1)
  (int_quotient_1 : ∃ k : ℤ, k * (a + b - c) = a^2 + b^2 - c^2)
  (int_quotient_2 : ∃ k : ℤ, k * (b + c - a) = b^2 + c^2 - a^2)
  (int_quotient_3 : ∃ k : ℤ, k * (c + a - b) = c^2 + a^2 - b^2) :
  ∃ n : ℤ, (a + b - c) * (b + c - a) * (c + a - b) = n^2 ∨
           2 * (a + b - c) * (b + c - a) * (c + a - b) = n^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sides_perfect_square_l903_90389


namespace NUMINAMATH_CALUDE_student_age_problem_l903_90374

theorem student_age_problem (n : ℕ) : 
  n < 10 →
  (8 : ℝ) * n = (10 : ℝ) * (n + 1) - 28 →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_student_age_problem_l903_90374
