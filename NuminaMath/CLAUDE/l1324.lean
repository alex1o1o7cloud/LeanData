import Mathlib

namespace NUMINAMATH_CALUDE_sin_4phi_value_l1324_132442

theorem sin_4phi_value (φ : ℝ) : 
  Complex.exp (Complex.I * φ) = (3 + Complex.I * Real.sqrt 8) / 5 →
  Real.sin (4 * φ) = 12 * Real.sqrt 8 / 625 := by
sorry

end NUMINAMATH_CALUDE_sin_4phi_value_l1324_132442


namespace NUMINAMATH_CALUDE_intersecting_plane_theorem_l1324_132408

/-- Represents a 3D cube composed of unit cubes -/
structure Cube where
  side_length : ℕ
  total_units : ℕ

/-- Represents a plane intersecting the cube -/
structure IntersectingPlane where
  perpendicular_to_diagonal : Bool
  distance_ratio : ℚ

/-- Calculates the number of unit cubes intersected by the plane -/
def intersected_cubes (c : Cube) (p : IntersectingPlane) : ℕ :=
  sorry

/-- Theorem stating that a plane intersecting a 4x4x4 cube at 1/4 of its diagonal intersects 36 unit cubes -/
theorem intersecting_plane_theorem (c : Cube) (p : IntersectingPlane) :
  c.side_length = 4 ∧ c.total_units = 64 ∧ p.perpendicular_to_diagonal = true ∧ p.distance_ratio = 1/4 →
  intersected_cubes c p = 36 :=
sorry

end NUMINAMATH_CALUDE_intersecting_plane_theorem_l1324_132408


namespace NUMINAMATH_CALUDE_james_profit_20_weeks_l1324_132480

/-- Calculates the profit from James' media empire over a given number of weeks. -/
def calculate_profit (movie_cost : ℕ) (dvd_cost : ℕ) (price_multiplier : ℚ) 
                     (daily_sales : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  let selling_price := dvd_cost * price_multiplier
  let profit_per_dvd := selling_price - dvd_cost
  let daily_profit := profit_per_dvd * daily_sales
  let weekly_profit := daily_profit * days_per_week
  let total_profit := weekly_profit * num_weeks
  (total_profit - movie_cost).floor.toNat

/-- Theorem stating that James' profit over 20 weeks is $448,000. -/
theorem james_profit_20_weeks : 
  calculate_profit 2000 6 (5/2) 500 5 20 = 448000 := by
  sorry

end NUMINAMATH_CALUDE_james_profit_20_weeks_l1324_132480


namespace NUMINAMATH_CALUDE_garden_ratio_l1324_132436

/-- Represents a rectangular garden with given perimeter and length -/
structure RectangularGarden where
  perimeter : ℝ
  length : ℝ
  width : ℝ
  perimeter_eq : perimeter = 2 * (length + width)
  length_positive : length > 0
  width_positive : width > 0

/-- The ratio of length to width for a rectangular garden with perimeter 150 and length 50 is 2:1 -/
theorem garden_ratio (garden : RectangularGarden) 
  (h_perimeter : garden.perimeter = 150) 
  (h_length : garden.length = 50) : 
  garden.length / garden.width = 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l1324_132436


namespace NUMINAMATH_CALUDE_ken_share_l1324_132438

def total_amount : ℕ := 5250

theorem ken_share (ken : ℕ) (tony : ℕ) 
  (h1 : ken + tony = total_amount) 
  (h2 : tony = 2 * ken) : 
  ken = 1750 := by
  sorry

end NUMINAMATH_CALUDE_ken_share_l1324_132438


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l1324_132449

theorem complex_sum_magnitude (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 1)
  (h₂ : Complex.abs z₂ = 1)
  (h₃ : Complex.abs (z₁ - z₂) = 1) :
  Complex.abs (z₁ + z₂) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l1324_132449


namespace NUMINAMATH_CALUDE_factor_and_divisor_statements_l1324_132439

-- Define what it means for a number to be a factor of another
def is_factor (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

-- Define what it means for a number to be a divisor of another
def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

theorem factor_and_divisor_statements :
  (is_factor 4 100) ∧
  (is_divisor 19 133 ∧ ¬ is_divisor 19 51) ∧
  (is_divisor 30 90 ∨ is_divisor 30 53) ∧
  (is_divisor 7 21 ∧ is_divisor 7 49) ∧
  (is_factor 10 200) := by sorry

end NUMINAMATH_CALUDE_factor_and_divisor_statements_l1324_132439


namespace NUMINAMATH_CALUDE_equipment_production_l1324_132496

theorem equipment_production (total : ℕ) (sample_size : ℕ) (sample_A : ℕ) (products_B : ℕ) : 
  total = 4800 → 
  sample_size = 80 → 
  sample_A = 50 → 
  products_B = total - (total * sample_A / sample_size) →
  products_B = 1800 := by
sorry

end NUMINAMATH_CALUDE_equipment_production_l1324_132496


namespace NUMINAMATH_CALUDE_comparison_of_scientific_notation_l1324_132404

theorem comparison_of_scientific_notation :
  (1.9 : ℝ) * (10 : ℝ) ^ 5 > (9.1 : ℝ) * (10 : ℝ) ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_scientific_notation_l1324_132404


namespace NUMINAMATH_CALUDE_room_height_is_12_l1324_132444

def room_length : ℝ := 25
def room_width : ℝ := 15
def door_area : ℝ := 6 * 3
def window_area : ℝ := 4 * 3
def num_windows : ℕ := 3
def whitewash_cost_per_sqft : ℝ := 2
def total_cost : ℝ := 1812

theorem room_height_is_12 (h : ℝ) :
  (2 * (room_length + room_width) * h - (door_area + num_windows * window_area)) * whitewash_cost_per_sqft = total_cost →
  h = 12 := by
  sorry

end NUMINAMATH_CALUDE_room_height_is_12_l1324_132444


namespace NUMINAMATH_CALUDE_distance_satisfies_conditions_l1324_132411

/-- The distance traveled by both the train and the ship -/
def distance : ℝ := 480

/-- The speed of the train in km/h -/
def train_speed : ℝ := 48

/-- The speed of the ship in km/h -/
def ship_speed : ℝ := 60

/-- The time difference between the train and ship journeys in hours -/
def time_difference : ℝ := 2

/-- Theorem stating that the given distance satisfies the problem conditions -/
theorem distance_satisfies_conditions :
  distance / train_speed = distance / ship_speed + time_difference :=
by sorry

end NUMINAMATH_CALUDE_distance_satisfies_conditions_l1324_132411


namespace NUMINAMATH_CALUDE_age_problem_l1324_132472

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 32 → 
  b = 12 := by sorry

end NUMINAMATH_CALUDE_age_problem_l1324_132472


namespace NUMINAMATH_CALUDE_solve_system_and_calculate_l1324_132469

theorem solve_system_and_calculate (x y : ℚ) 
  (eq1 : 2 * x + y = 26) 
  (eq2 : x + 2 * y = 10) : 
  (x + y) / 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_system_and_calculate_l1324_132469


namespace NUMINAMATH_CALUDE_product_simplification_l1324_132415

theorem product_simplification : 
  (1 + 2 / 1) * (1 + 2 / 2) * (1 + 2 / 3) * (1 + 2 / 4) * (1 + 2 / 5) * (1 + 2 / 6) - 1 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l1324_132415


namespace NUMINAMATH_CALUDE_football_club_balance_l1324_132435

/-- Calculates the final balance of a football club after player transactions -/
def final_balance (initial_balance : ℝ) (players_sold : ℕ) (selling_price : ℝ) 
  (players_bought : ℕ) (buying_price : ℝ) : ℝ :=
  initial_balance + players_sold * selling_price - players_bought * buying_price

/-- Theorem: The final balance of the football club is $60 million -/
theorem football_club_balance : 
  final_balance 100 2 10 4 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_football_club_balance_l1324_132435


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1324_132400

theorem inequality_solution_set (a : ℝ) (h : a > 1) :
  {x : ℝ | (x - a) * (x - 1/a) > 0} = {x : ℝ | x < 1/a ∨ x > a} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1324_132400


namespace NUMINAMATH_CALUDE_subject_choice_theorem_l1324_132455

/-- The number of subjects available --/
def num_subjects : ℕ := 7

/-- The number of subjects each student must choose --/
def subjects_to_choose : ℕ := 3

/-- The number of ways Student A can choose subjects --/
def ways_for_A : ℕ := Nat.choose (num_subjects - 1) (subjects_to_choose - 1)

/-- The probability that both Students B and C choose physics --/
def prob_B_and_C_physics : ℚ := 
  (Nat.choose (num_subjects - 1) (subjects_to_choose - 1) ^ 2 : ℚ) / 
  (Nat.choose num_subjects subjects_to_choose ^ 2 : ℚ)

theorem subject_choice_theorem : 
  ways_for_A = 15 ∧ prob_B_and_C_physics = 9 / 49 := by sorry

end NUMINAMATH_CALUDE_subject_choice_theorem_l1324_132455


namespace NUMINAMATH_CALUDE_work_completion_time_l1324_132402

theorem work_completion_time 
  (total_work : ℝ) 
  (a b c : ℝ) 
  (h1 : a + b + c = total_work / 4)  -- a, b, and c together finish in 4 days
  (h2 : b = total_work / 9)          -- b alone finishes in 9 days
  (h3 : c = total_work / 18)         -- c alone finishes in 18 days
  : a = total_work / 12 :=           -- a alone finishes in 12 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1324_132402


namespace NUMINAMATH_CALUDE_perpendicular_lines_condition_l1324_132421

/-- Two lines y = m₁x + b₁ and y = m₂x + b₂ are perpendicular if and only if m₁ * m₂ = -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The statement "a = 2 is a sufficient but not necessary condition for the lines
    y = -ax + 2 and y = (a/4)x - 1 to be perpendicular" -/
theorem perpendicular_lines_condition (a : ℝ) :
  (a = 2 → perpendicular (-a) (a/4)) ∧ 
  ¬(perpendicular (-a) (a/4) → a = 2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_condition_l1324_132421


namespace NUMINAMATH_CALUDE_blocks_used_for_tower_l1324_132495

/-- Given Randy's block usage, prove the number of blocks used for the tower -/
theorem blocks_used_for_tower 
  (total_blocks : ℕ) 
  (blocks_for_house : ℕ) 
  (blocks_for_tower : ℕ) 
  (h1 : total_blocks = 95) 
  (h2 : blocks_for_house = 20) 
  (h3 : blocks_for_tower = blocks_for_house + 30) : 
  blocks_for_tower = 50 := by
  sorry

end NUMINAMATH_CALUDE_blocks_used_for_tower_l1324_132495


namespace NUMINAMATH_CALUDE_stating_escalator_step_count_l1324_132413

/-- Represents the number of steps counted on an escalator under different conditions -/
structure EscalatorSteps where
  down : ℕ  -- steps counted running down
  up : ℕ    -- steps counted running up
  stationary : ℕ  -- steps counted on a stationary escalator

/-- 
Given the number of steps counted running down and up a moving escalator,
calculates the number of steps on a stationary escalator
-/
def calculateStationarySteps (e : EscalatorSteps) : Prop :=
  e.down = 30 ∧ e.up = 150 → e.stationary = 50

/-- 
Theorem stating that if a person counts 30 steps running down a moving escalator
and 150 steps running up the same escalator at the same speed relative to the escalator,
then they would count 50 steps on a stationary escalator
-/
theorem escalator_step_count : ∃ e : EscalatorSteps, calculateStationarySteps e :=
  sorry

end NUMINAMATH_CALUDE_stating_escalator_step_count_l1324_132413


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1324_132441

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3*b = 1) :
  1/a + 3/b ≥ 16 :=
sorry

theorem min_value_achievable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3*b = 1 ∧ 1/a + 3/b = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l1324_132441


namespace NUMINAMATH_CALUDE_find_m_l1324_132489

def U : Set ℕ := {1, 2, 3, 4}

def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5*x + m = 0}

theorem find_m : ∃ m : ℕ, (U \ A m) = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1324_132489


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1324_132440

theorem simplify_and_evaluate (x : ℝ) (h : x = 2) : (x + 3) * (x - 2) + x * (4 - x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1324_132440


namespace NUMINAMATH_CALUDE_custom_mult_chain_l1324_132498

/-- Custom multiplication operation -/
def star_mult (a b : ℚ) : ℚ := (a - b) / (1 - a * b)

/-- Main theorem -/
theorem custom_mult_chain : star_mult 5 (star_mult 6 (star_mult 7 (star_mult 8 9))) = 3588 / 587 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_chain_l1324_132498


namespace NUMINAMATH_CALUDE_number_of_happy_arrangements_l1324_132428

/-- Represents the types of chains --/
inductive Chain : Type
| Silver : Chain
| Gold : Chain
| Iron : Chain

/-- Represents the types of stones --/
inductive Stone : Type
| CubicZirconia : Stone
| Emerald : Stone
| Quartz : Stone

/-- Represents the types of pendants --/
inductive Pendant : Type
| Star : Pendant
| Sun : Pendant
| Moon : Pendant

/-- Represents a piece of jewelry --/
structure Jewelry :=
  (chain : Chain)
  (stone : Stone)
  (pendant : Pendant)

/-- Represents an arrangement of three pieces of jewelry --/
structure Arrangement :=
  (left : Jewelry)
  (middle : Jewelry)
  (right : Jewelry)

/-- Predicate to check if an arrangement satisfies Polina's conditions --/
def satisfiesConditions (arr : Arrangement) : Prop :=
  (arr.middle.chain = Chain.Iron ∧ arr.middle.pendant = Pendant.Sun) ∧
  ((arr.left.chain = Chain.Gold ∧ arr.right.chain = Chain.Silver) ∨
   (arr.left.chain = Chain.Silver ∧ arr.right.chain = Chain.Gold)) ∧
  (arr.left.stone ≠ arr.middle.stone ∧ arr.left.stone ≠ arr.right.stone ∧ arr.middle.stone ≠ arr.right.stone) ∧
  (arr.left.pendant ≠ arr.middle.pendant ∧ arr.left.pendant ≠ arr.right.pendant ∧ arr.middle.pendant ≠ arr.right.pendant) ∧
  (arr.left.chain ≠ arr.middle.chain ∧ arr.left.chain ≠ arr.right.chain ∧ arr.middle.chain ≠ arr.right.chain)

/-- The theorem to be proved --/
theorem number_of_happy_arrangements :
  ∃! (n : ℕ), ∃ (arrangements : Finset Arrangement),
    arrangements.card = n ∧
    (∀ arr ∈ arrangements, satisfiesConditions arr) ∧
    (∀ arr : Arrangement, satisfiesConditions arr → arr ∈ arrangements) :=
sorry

end NUMINAMATH_CALUDE_number_of_happy_arrangements_l1324_132428


namespace NUMINAMATH_CALUDE_quadratic_sum_of_b_and_c_l1324_132466

/-- Given a quadratic expression x^2 - 24x + 50, when written in the form (x+b)^2 + c,
    the sum of b and c is equal to -106. -/
theorem quadratic_sum_of_b_and_c : ∃ (b c : ℝ), 
  (∀ x, x^2 - 24*x + 50 = (x + b)^2 + c) ∧ (b + c = -106) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_b_and_c_l1324_132466


namespace NUMINAMATH_CALUDE_reflection_coordinates_sum_l1324_132453

/-- Given a point C with coordinates (3, -2) and its reflection D over the y-axis,
    the sum of their four coordinate values is -4. -/
theorem reflection_coordinates_sum :
  let C : ℝ × ℝ := (3, -2)
  let D : ℝ × ℝ := (-C.1, C.2)  -- Reflection over y-axis
  C.1 + C.2 + D.1 + D.2 = -4 :=
by sorry

end NUMINAMATH_CALUDE_reflection_coordinates_sum_l1324_132453


namespace NUMINAMATH_CALUDE_max_lateral_area_inscribed_cylinder_l1324_132470

/-- The maximum lateral surface area of a cylinder inscribed in a sphere -/
theorem max_lateral_area_inscribed_cylinder (r : ℝ) (h : r > 0) :
  ∃ (cylinder_area : ℝ),
    (∀ (inscribed_cylinder_area : ℝ), inscribed_cylinder_area ≤ cylinder_area) ∧
    cylinder_area = 2 * Real.pi * r^2 :=
sorry

end NUMINAMATH_CALUDE_max_lateral_area_inscribed_cylinder_l1324_132470


namespace NUMINAMATH_CALUDE_alternative_interest_rate_l1324_132457

theorem alternative_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (chosen_rate : ℝ) 
  (interest_difference : ℝ) : ℝ :=
  let alternative_rate := 
    (principal * chosen_rate * time - interest_difference) / (principal * time)
  
  -- Assumptions
  have h1 : principal = 7000 := by sorry
  have h2 : time = 2 := by sorry
  have h3 : chosen_rate = 0.15 := by sorry
  have h4 : interest_difference = 420 := by sorry

  -- Theorem statement
  alternative_rate * 100

/- Proof
  sorry
-/

end NUMINAMATH_CALUDE_alternative_interest_rate_l1324_132457


namespace NUMINAMATH_CALUDE_basketball_teams_count_l1324_132419

/-- The number of combinations of n items taken k at a time -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of people available for the basketball game -/
def total_people : ℕ := 7

/-- The number of players needed for each team -/
def team_size : ℕ := 4

/-- Theorem: The number of different teams of 4 that can be formed from 7 people is 35 -/
theorem basketball_teams_count : binomial total_people team_size = 35 := by sorry

end NUMINAMATH_CALUDE_basketball_teams_count_l1324_132419


namespace NUMINAMATH_CALUDE_transformed_function_theorem_l1324_132473

def original_function (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

def rotate_180_degrees (f : ℝ → ℝ) : ℝ → ℝ := λ x => -f x

def translate_upwards (f : ℝ → ℝ) (units : ℝ) : ℝ → ℝ := λ x => f x + units

theorem transformed_function_theorem :
  (translate_upwards (rotate_180_degrees original_function) 3) = λ x => -2 * x^2 - 4 * x :=
by sorry

end NUMINAMATH_CALUDE_transformed_function_theorem_l1324_132473


namespace NUMINAMATH_CALUDE_joe_chocolate_spending_l1324_132479

theorem joe_chocolate_spending (total : ℚ) (fruit_fraction : ℚ) (left : ℚ) 
  (h1 : total = 450)
  (h2 : fruit_fraction = 2/5)
  (h3 : left = 220) :
  (total - left - fruit_fraction * total) / total = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_joe_chocolate_spending_l1324_132479


namespace NUMINAMATH_CALUDE_chord_length_for_60_degree_line_and_circle_l1324_132446

/-- The length of the chord formed by the intersection of a line passing through the origin
    with a slope angle of 60° and the circle x² + y² - 4y = 0 is equal to 2√3. -/
theorem chord_length_for_60_degree_line_and_circle : 
  let line := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * x}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 4*y = 0}
  let chord := line ∩ circle
  ∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_for_60_degree_line_and_circle_l1324_132446


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l1324_132499

/-- Represents the repeating decimal 7.036036036... -/
def repeating_decimal : ℚ := 7 + 36 / 999

/-- The repeating decimal 7.036036036... is equal to the fraction 781/111 -/
theorem repeating_decimal_as_fraction : repeating_decimal = 781 / 111 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l1324_132499


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l1324_132478

theorem max_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ x * y = 1000000 ∧ 
  ∀ (a b : ℤ), a + b = 2000 → a * b ≤ 1000000 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l1324_132478


namespace NUMINAMATH_CALUDE_octagon_area_l1324_132414

/-- The area of an octagon formed by the intersection of two concentric squares -/
theorem octagon_area (side_large : ℝ) (side_small : ℝ) (octagon_side : ℝ) : 
  side_large = 2 →
  side_small = 1 →
  octagon_side = 17/36 →
  let octagon_area := 8 * (1/2 * octagon_side * side_small)
  octagon_area = 17/9 := by
sorry

end NUMINAMATH_CALUDE_octagon_area_l1324_132414


namespace NUMINAMATH_CALUDE_pascal_triangle_sum_l1324_132443

/-- The number of elements in a row of Pascal's Triangle -/
def elementsInRow (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def sumOfElements (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

theorem pascal_triangle_sum :
  sumOfElements 29 = 465 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_sum_l1324_132443


namespace NUMINAMATH_CALUDE_sqrt_21000_l1324_132427

theorem sqrt_21000 (h : Real.sqrt 2.1 = 1.449) : Real.sqrt 21000 = 144.9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_21000_l1324_132427


namespace NUMINAMATH_CALUDE_canoe_downstream_speed_l1324_132485

/-- Given a canoe that rows upstream at 6 km/hr and a stream with a speed of 2 km/hr,
    this theorem proves that the speed of the canoe when rowing downstream is 10 km/hr. -/
theorem canoe_downstream_speed
  (upstream_speed : ℝ)
  (stream_speed : ℝ)
  (h1 : upstream_speed = 6)
  (h2 : stream_speed = 2) :
  upstream_speed + 2 * stream_speed = 10 :=
by sorry

end NUMINAMATH_CALUDE_canoe_downstream_speed_l1324_132485


namespace NUMINAMATH_CALUDE_floor_sqrt_26_squared_l1324_132417

theorem floor_sqrt_26_squared : ⌊Real.sqrt 26⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_26_squared_l1324_132417


namespace NUMINAMATH_CALUDE_next_roll_for_average_three_l1324_132407

def rolls : List Nat := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

theorem next_roll_for_average_three (rolls : List Nat) : 
  rolls.length = 10 → 
  rolls.sum = 31 → 
  ∃ (next_roll : Nat), 
    (rolls.sum + next_roll) / (rolls.length + 1 : Nat) = 3 ∧ 
    next_roll = 2 := by
  sorry

#check next_roll_for_average_three rolls

end NUMINAMATH_CALUDE_next_roll_for_average_three_l1324_132407


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l1324_132433

theorem journey_speed_calculation (D : ℝ) (v : ℝ) (h1 : D > 0) (h2 : v > 0) : 
  (D / ((0.8 * D / 80) + (0.2 * D / v)) = 50) → v = 20 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l1324_132433


namespace NUMINAMATH_CALUDE_students_playing_all_sports_l1324_132410

/-- The number of students playing all three sports in a school with given sport participation data -/
theorem students_playing_all_sports (total : ℕ) (football cricket basketball : ℕ) 
  (neither : ℕ) (football_cricket football_basketball cricket_basketball : ℕ) :
  total = 580 →
  football = 300 →
  cricket = 250 →
  basketball = 180 →
  neither = 60 →
  football_cricket = 120 →
  football_basketball = 80 →
  cricket_basketball = 70 →
  ∃ (all_sports : ℕ), 
    all_sports = 140 ∧
    total = football + cricket + basketball - football_cricket - football_basketball - cricket_basketball + all_sports + neither :=
by sorry

end NUMINAMATH_CALUDE_students_playing_all_sports_l1324_132410


namespace NUMINAMATH_CALUDE_cans_recycled_l1324_132477

/-- Proves the number of cans recycled given the bottle and can deposits, number of bottles, and total money earned -/
theorem cans_recycled 
  (bottle_deposit : ℚ) 
  (can_deposit : ℚ) 
  (bottles_recycled : ℕ) 
  (total_money : ℚ) 
  (h1 : bottle_deposit = 10 / 100)
  (h2 : can_deposit = 5 / 100)
  (h3 : bottles_recycled = 80)
  (h4 : total_money = 15) :
  (total_money - (bottle_deposit * bottles_recycled)) / can_deposit = 140 := by
sorry

end NUMINAMATH_CALUDE_cans_recycled_l1324_132477


namespace NUMINAMATH_CALUDE_sam_total_dimes_l1324_132487

def initial_dimes : ℕ := 9
def received_dimes : ℕ := 7

theorem sam_total_dimes : initial_dimes + received_dimes = 16 := by
  sorry

end NUMINAMATH_CALUDE_sam_total_dimes_l1324_132487


namespace NUMINAMATH_CALUDE_intersection_points_coincide_l1324_132494

/-- Two circles in a plane -/
structure TwoCircles where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Radius of the first circle -/
  radius1 : ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of the second circle -/
  radius2 : ℝ

/-- The square of the distance between intersection points of two circles -/
def intersectionPointsDistanceSquared (circles : TwoCircles) : ℝ := sorry

/-- Theorem: The square of the distance between intersection points is zero for the given circles -/
theorem intersection_points_coincide (circles : TwoCircles) 
  (h1 : circles.center1 = (3, -2))
  (h2 : circles.radius1 = 5)
  (h3 : circles.center2 = (3, 6))
  (h4 : circles.radius2 = 3) :
  intersectionPointsDistanceSquared circles = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_points_coincide_l1324_132494


namespace NUMINAMATH_CALUDE_find_N_l1324_132448

theorem find_N : ∃ N : ℚ, (5 + 6 + 7 + 8) / 4 = (2014 + 2015 + 2016 + 2017) / N ∧ N = 1240 := by
  sorry

end NUMINAMATH_CALUDE_find_N_l1324_132448


namespace NUMINAMATH_CALUDE_median_of_special_list_l1324_132422

def list_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem median_of_special_list : 
  let total_elements : ℕ := list_sum 100
  let median_position : ℕ := total_elements / 2
  let cumulative_count (k : ℕ) : ℕ := list_sum k
  ∃ n : ℕ, 
    cumulative_count n ≥ median_position ∧ 
    cumulative_count (n-1) < median_position ∧
    n = 71 := by
  sorry

#check median_of_special_list

end NUMINAMATH_CALUDE_median_of_special_list_l1324_132422


namespace NUMINAMATH_CALUDE_different_color_probability_l1324_132483

def blue_chips : ℕ := 8
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4
def green_chips : ℕ := 3

def total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

def prob_different_colors : ℚ :=
  (blue_chips * (red_chips + yellow_chips + green_chips) +
   red_chips * (blue_chips + yellow_chips + green_chips) +
   yellow_chips * (blue_chips + red_chips + green_chips) +
   green_chips * (blue_chips + red_chips + yellow_chips)) /
  (total_chips * total_chips)

theorem different_color_probability :
  prob_different_colors = 143 / 200 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l1324_132483


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1324_132447

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 5 * x + b

-- Define the solution set type
def SolutionSet := Set ℝ

-- State the theorem
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h1 : a = -5) 
  (h2 : b = 30) 
  (h3 : {x : ℝ | f a b x > 0} = {x : ℝ | -3 < x ∧ x < 2}) :
  {x : ℝ | f b (-a) x > 0} = {x : ℝ | x < -1/3 ∨ x > 1/2} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1324_132447


namespace NUMINAMATH_CALUDE_cookie_problem_l1324_132476

theorem cookie_problem (initial_cookies : ℕ) : 
  (initial_cookies : ℚ) * (1/4) * (1/2) = 8 → initial_cookies = 64 := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l1324_132476


namespace NUMINAMATH_CALUDE_nested_square_root_18_l1324_132434

theorem nested_square_root_18 :
  ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (18 + x) → x = 6 := by sorry

end NUMINAMATH_CALUDE_nested_square_root_18_l1324_132434


namespace NUMINAMATH_CALUDE_min_value_x2_plus_y2_l1324_132445

theorem min_value_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 * y^2 + y^4 = 1) :
  x^2 + y^2 ≥ 4/5 ∧ ∃ x y : ℝ, 5 * x^2 * y^2 + y^4 = 1 ∧ x^2 + y^2 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x2_plus_y2_l1324_132445


namespace NUMINAMATH_CALUDE_both_first_prize_probability_X_distribution_l1324_132491

structure StudentPopulation where
  total : ℕ
  male : ℕ
  female : ℕ
  male_first_prize : ℕ
  male_second_prize : ℕ
  male_third_prize : ℕ
  female_first_prize : ℕ
  female_second_prize : ℕ
  female_third_prize : ℕ

def sample : StudentPopulation := {
  total := 500,
  male := 200,
  female := 300,
  male_first_prize := 10,
  male_second_prize := 15,
  male_third_prize := 15,
  female_first_prize := 25,
  female_second_prize := 25,
  female_third_prize := 40
}

def prob_both_first_prize (s : StudentPopulation) : ℚ :=
  (s.male_first_prize : ℚ) / s.male * (s.female_first_prize : ℚ) / s.female

def prob_male_award (s : StudentPopulation) : ℚ :=
  (s.male_first_prize + s.male_second_prize + s.male_third_prize : ℚ) / s.male

def prob_female_award (s : StudentPopulation) : ℚ :=
  (s.female_first_prize + s.female_second_prize + s.female_third_prize : ℚ) / s.female

def prob_X (s : StudentPopulation) : Fin 3 → ℚ
| 0 => (1 - prob_male_award s) * (1 - prob_female_award s)
| 1 => 1 - (1 - prob_male_award s) * (1 - prob_female_award s) - prob_male_award s * prob_female_award s
| 2 => prob_male_award s * prob_female_award s

theorem both_first_prize_probability :
  prob_both_first_prize sample = 1 / 240 := by sorry

theorem X_distribution :
  prob_X sample 0 = 28 / 50 ∧
  prob_X sample 1 = 19 / 50 ∧
  prob_X sample 2 = 3 / 50 := by sorry

end NUMINAMATH_CALUDE_both_first_prize_probability_X_distribution_l1324_132491


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1324_132454

/-- Given a hyperbola C with equation x²/m - y² = 1 where m > 0,
    and its asymptote √3x + my = 0, the focal length of C is 4. -/
theorem hyperbola_focal_length (m : ℝ) (hm : m > 0) : 
  (∃ (C : Set (ℝ × ℝ)), 
    C = {(x, y) | x^2 / m - y^2 = 1} ∧
    (∃ (asymptote : Set (ℝ × ℝ)), 
      asymptote = {(x, y) | Real.sqrt 3 * x + m * y = 0})) →
  (∃ (focal_length : ℝ), focal_length = 4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1324_132454


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1324_132493

theorem partial_fraction_decomposition :
  ∃! (A B : ℝ), ∀ (x : ℝ), x ≠ 5 → x ≠ 6 →
    (5 * x - 8) / (x^2 - 11 * x + 30) = A / (x - 5) + B / (x - 6) ∧ A = -17 ∧ B = 22 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1324_132493


namespace NUMINAMATH_CALUDE_polynomial_product_theorem_l1324_132467

theorem polynomial_product_theorem (p q : ℚ) : 
  (∀ x, (x^2 + p*x - 1/3) * (x^2 - 3*x + q) = x^4 + (q - 3*p - 1/3)*x^2 - q/3) → 
  (p = 3 ∧ q = -1/3 ∧ (-2*p^2*q)^2 + 3*p*q = 33) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_theorem_l1324_132467


namespace NUMINAMATH_CALUDE_janice_starting_sentences_janice_starting_sentences_proof_l1324_132437

/-- Proves the number of sentences Janice started with today -/
theorem janice_starting_sentences : ℕ :=
  let typing_speed : ℕ := 6  -- sentences per minute
  let first_session : ℕ := 20  -- minutes
  let second_session : ℕ := 15  -- minutes
  let third_session : ℕ := 18  -- minutes
  let erased_sentences : ℕ := 40
  let total_sentences : ℕ := 536

  let total_typed : ℕ := typing_speed * (first_session + second_session + third_session)
  let net_added : ℕ := total_typed - erased_sentences
  
  total_sentences - net_added

/-- The theorem statement -/
theorem janice_starting_sentences_proof : janice_starting_sentences = 258 := by
  sorry

end NUMINAMATH_CALUDE_janice_starting_sentences_janice_starting_sentences_proof_l1324_132437


namespace NUMINAMATH_CALUDE_mixed_fraction_calculation_l1324_132432

theorem mixed_fraction_calculation : 
  (-4 - 2/3) - (1 + 5/6) - (-18 - 1/2) + (-13 - 3/4) = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_mixed_fraction_calculation_l1324_132432


namespace NUMINAMATH_CALUDE_line_through_point_l1324_132412

/-- Given a line equation ax + (a+4)y = a + 5 passing through (5, -10), prove a = -7.5 -/
theorem line_through_point (a : ℝ) : 
  (∀ x y : ℝ, a * x + (a + 4) * y = a + 5 → x = 5 ∧ y = -10) → 
  a = -7.5 := by sorry

end NUMINAMATH_CALUDE_line_through_point_l1324_132412


namespace NUMINAMATH_CALUDE_expression_evaluation_l1324_132431

theorem expression_evaluation : 7500 + (1250 / 50) = 7525 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1324_132431


namespace NUMINAMATH_CALUDE_pencil_sharpening_theorem_l1324_132452

/-- Calculates the final length of a pencil after sharpening on two consecutive days. -/
def pencil_length (initial_length : ℕ) (day1_sharpening : ℕ) (day2_sharpening : ℕ) : ℕ :=
  initial_length - day1_sharpening - day2_sharpening

/-- Proves that a 22-inch pencil sharpened by 2 inches on two consecutive days will be 18 inches long. -/
theorem pencil_sharpening_theorem :
  pencil_length 22 2 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_theorem_l1324_132452


namespace NUMINAMATH_CALUDE_lord_moneybag_puzzle_l1324_132490

/-- Lord Moneybag's Christmas money puzzle -/
theorem lord_moneybag_puzzle :
  ∃! n : ℕ, 300 ≤ n ∧ n ≤ 500 ∧ 
  6 ∣ n ∧
  5 ∣ (n - 1) ∧
  4 ∣ (n - 2) ∧
  3 ∣ (n - 3) ∧
  2 ∣ (n - 4) ∧
  Nat.Prime (n - 5) ∧
  n = 426 := by
sorry

end NUMINAMATH_CALUDE_lord_moneybag_puzzle_l1324_132490


namespace NUMINAMATH_CALUDE_larger_number_value_l1324_132464

theorem larger_number_value (x y : ℝ) 
  (h1 : 4 * y = 5 * x) 
  (h2 : y - x = 10) : 
  y = 50 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_value_l1324_132464


namespace NUMINAMATH_CALUDE_fake_coin_determinable_l1324_132418

/-- Represents the result of a weighing on a two-pan balance scale -/
inductive WeighingResult
  | Left : WeighingResult  -- Left pan is heavier
  | Right : WeighingResult -- Right pan is heavier
  | Equal : WeighingResult -- Pans are balanced

/-- Represents the state of a coin -/
inductive CoinState
  | Normal : CoinState
  | Heavier : CoinState
  | Lighter : CoinState

/-- Represents a weighing on a two-pan balance scale -/
def Weighing := (Fin 25 → Bool) → WeighingResult

/-- Represents the strategy for determining the state of the fake coin -/
def Strategy := Weighing → Weighing → CoinState

/-- Theorem stating that it's possible to determine whether the fake coin
    is lighter or heavier using only two weighings -/
theorem fake_coin_determinable :
  ∃ (s : Strategy),
    ∀ (fake : Fin 25) (state : CoinState),
      state ≠ CoinState.Normal →
        ∀ (w₁ w₂ : Weighing),
          (∀ (f : Fin 25 → Bool),
            w₁ f = WeighingResult.Left ↔ (state = CoinState.Heavier ∧ f fake) ∨
                                         (state = CoinState.Lighter ∧ ¬f fake)) →
          (∀ (f : Fin 25 → Bool),
            w₂ f = WeighingResult.Left ↔ (state = CoinState.Heavier ∧ f fake) ∨
                                         (state = CoinState.Lighter ∧ ¬f fake)) →
          s w₁ w₂ = state :=
sorry

end NUMINAMATH_CALUDE_fake_coin_determinable_l1324_132418


namespace NUMINAMATH_CALUDE_logarithm_inequality_l1324_132460

theorem logarithm_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
  Real.log a^2 / Real.log (b + c) + Real.log b^2 / Real.log (a + c) + Real.log c^2 / Real.log (a + b) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l1324_132460


namespace NUMINAMATH_CALUDE_tangent_slope_condition_l1324_132405

/-- The curve function -/
def f (a : ℝ) (x : ℝ) : ℝ := x^5 - a*(x + 1)

/-- The derivative of the curve function -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 5*x^4 - a

theorem tangent_slope_condition (a : ℝ) :
  (f_derivative a 1 > 1) ↔ (a < 4) := by sorry

end NUMINAMATH_CALUDE_tangent_slope_condition_l1324_132405


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1324_132456

theorem root_sum_theorem : ∃ (a b : ℝ), 
  (∃ (x y : ℝ), x ≠ y ∧ 
    ((a * x^2 - 24 * x + b) / (x^2 - 1) = x) ∧ 
    ((a * y^2 - 24 * y + b) / (y^2 - 1) = y) ∧
    x + y = 12) ∧
  ((a = 11 ∧ b = -35) ∨ (a = 35 ∧ b = -5819)) := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l1324_132456


namespace NUMINAMATH_CALUDE_prob_allison_between_brian_and_noah_l1324_132462

/-- Represents a 6-sided cube with specific face values -/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- Allison's cube with all faces showing 6 -/
def allison_cube : Cube :=
  { faces := λ _ => 6 }

/-- Brian's cube with faces numbered 1 to 6 -/
def brian_cube : Cube :=
  { faces := λ i => i.val + 1 }

/-- Noah's cube with three faces showing 4 and three faces showing 7 -/
def noah_cube : Cube :=
  { faces := λ i => if i.val < 3 then 4 else 7 }

/-- The probability of rolling a specific value or higher on a given cube -/
def prob_roll_ge (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i ≥ n) (Finset.univ : Finset (Fin 6))).card / 6

/-- The probability of rolling a specific value or lower on a given cube -/
def prob_roll_le (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i ≤ n) (Finset.univ : Finset (Fin 6))).card / 6

/-- The main theorem stating the probability of Allison rolling higher than Brian but lower than Noah -/
theorem prob_allison_between_brian_and_noah :
  prob_roll_ge brian_cube 6 * prob_roll_ge noah_cube 7 = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_allison_between_brian_and_noah_l1324_132462


namespace NUMINAMATH_CALUDE_nurse_lice_check_l1324_132458

/-- The number of Kindergarteners the nurse needs to check -/
def kindergarteners_to_check : ℕ := by sorry

theorem nurse_lice_check :
  let first_graders : ℕ := 19
  let second_graders : ℕ := 20
  let third_graders : ℕ := 25
  let minutes_per_check : ℕ := 2
  let total_hours : ℕ := 3
  let total_minutes : ℕ := total_hours * 60
  
  kindergarteners_to_check = 
    (total_minutes - 
      (first_graders + second_graders + third_graders) * minutes_per_check) / 
    minutes_per_check :=
by sorry

end NUMINAMATH_CALUDE_nurse_lice_check_l1324_132458


namespace NUMINAMATH_CALUDE_min_sum_of_positive_integers_l1324_132401

theorem min_sum_of_positive_integers (a b : ℕ+) (h : a.val * b.val - 7 * a.val - 11 * b.val + 13 = 0) :
  ∃ (a₀ b₀ : ℕ+), a₀.val * b₀.val - 7 * a₀.val - 11 * b₀.val + 13 = 0 ∧
    ∀ (x y : ℕ+), x.val * y.val - 7 * x.val - 11 * y.val + 13 = 0 → a₀.val + b₀.val ≤ x.val + y.val ∧
    a₀.val + b₀.val = 34 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_positive_integers_l1324_132401


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_area_for_radius_6_l1324_132429

/-- The area of the largest possible triangle inscribed in a circle,
    where one side of the triangle is a diameter of the circle. -/
def largest_inscribed_triangle_area (r : ℝ) : ℝ :=
  2 * r * r

theorem largest_inscribed_triangle_area_for_radius_6 :
  largest_inscribed_triangle_area 6 = 36 := by
  sorry

#eval largest_inscribed_triangle_area 6

end NUMINAMATH_CALUDE_largest_inscribed_triangle_area_for_radius_6_l1324_132429


namespace NUMINAMATH_CALUDE_substitution_sequences_remainder_l1324_132468

/-- Represents the number of possible substitution sequences in a basketball game -/
def substitutionSequences (totalPlayers startingPlayers maxSubstitutions : ℕ) : ℕ :=
  let substitutes := totalPlayers - startingPlayers
  let a0 := 1  -- No substitutions
  let a1 := startingPlayers * substitutes  -- One substitution
  let a2 := a1 * (startingPlayers - 1) * (substitutes - 1)  -- Two substitutions
  let a3 := a2 * (startingPlayers - 2) * (substitutes - 2)  -- Three substitutions
  let a4 := a3 * (startingPlayers - 3) * (substitutes - 3)  -- Four substitutions
  a0 + a1 + a2 + a3 + a4

/-- The main theorem stating the remainder of substitution sequences divided by 100 -/
theorem substitution_sequences_remainder :
  substitutionSequences 15 5 4 % 100 = 51 := by
  sorry


end NUMINAMATH_CALUDE_substitution_sequences_remainder_l1324_132468


namespace NUMINAMATH_CALUDE_ella_gives_one_sixth_l1324_132463

-- Define the initial cookie distribution
def initial_distribution (luke_cookies : ℚ) : ℚ × ℚ × ℚ :=
  (2 * luke_cookies, 4 * luke_cookies, luke_cookies)

-- Define the function to calculate the fraction Ella gives to Luke
def fraction_ella_gives (luke_cookies : ℚ) : ℚ :=
  let (ella_cookies, connor_cookies, luke_cookies) := initial_distribution luke_cookies
  let total_cookies := ella_cookies + connor_cookies + luke_cookies
  let equal_share := total_cookies / 3
  (equal_share - luke_cookies) / ella_cookies

-- Theorem statement
theorem ella_gives_one_sixth :
  ∀ (luke_cookies : ℚ), luke_cookies > 0 → fraction_ella_gives luke_cookies = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ella_gives_one_sixth_l1324_132463


namespace NUMINAMATH_CALUDE_train_length_l1324_132403

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 265 →
  train_speed * crossing_time - bridge_length = 110 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1324_132403


namespace NUMINAMATH_CALUDE_inequality_of_logarithms_l1324_132465

theorem inequality_of_logarithms (a b c : ℝ) 
  (ha : a = Real.log 2) 
  (hb : b = Real.log 3) 
  (hc : c = Real.log 5) : 
  c / 5 < a / 2 ∧ a / 2 < b / 3 := by sorry

end NUMINAMATH_CALUDE_inequality_of_logarithms_l1324_132465


namespace NUMINAMATH_CALUDE_quadratic_polynomial_discriminant_l1324_132423

theorem quadratic_polynomial_discriminant 
  (a b c : ℝ) (ha : a ≠ 0) 
  (h1 : ∃! x, a * x^2 + b * x + c = x - 2) 
  (h2 : ∃! x, a * x^2 + b * x + c = 1 - x / 2) : 
  b^2 - 4*a*c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_discriminant_l1324_132423


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_8_l1324_132461

def binary_number : ℕ := 0b1110101101101

theorem remainder_of_binary_div_8 :
  binary_number % 8 = 5 := by sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_8_l1324_132461


namespace NUMINAMATH_CALUDE_bank_interest_rate_is_five_percent_l1324_132450

/-- Proves that the bank interest rate is 5% given the investment conditions -/
theorem bank_interest_rate_is_five_percent 
  (total_investment : ℝ)
  (bank_investment : ℝ)
  (bond_investment : ℝ)
  (total_annual_income : ℝ)
  (bond_return_rate : ℝ)
  (h1 : total_investment = 10000)
  (h2 : bank_investment = 6000)
  (h3 : bond_investment = 4000)
  (h4 : total_annual_income = 660)
  (h5 : bond_return_rate = 0.09)
  (h6 : total_investment = bank_investment + bond_investment)
  (h7 : total_annual_income = bank_investment * bank_interest_rate + bond_investment * bond_return_rate) :
  bank_interest_rate = 0.05 := by
  sorry

#check bank_interest_rate_is_five_percent

end NUMINAMATH_CALUDE_bank_interest_rate_is_five_percent_l1324_132450


namespace NUMINAMATH_CALUDE_square_greater_than_self_when_less_than_negative_one_l1324_132420

theorem square_greater_than_self_when_less_than_negative_one (x : ℝ) : 
  x < -1 → x^2 > x := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_self_when_less_than_negative_one_l1324_132420


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1324_132488

/-- Given that when x = 1, the value of (1/2)ax³ - 3bx + 4 is 9,
    prove that when x = -1, the value of the expression is -1 -/
theorem algebraic_expression_value (a b : ℝ) :
  (1/2 * a * 1^3 - 3 * b * 1 + 4 = 9) →
  (1/2 * a * (-1)^3 - 3 * b * (-1) + 4 = -1) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1324_132488


namespace NUMINAMATH_CALUDE_correct_sacks_per_day_l1324_132424

/-- The number of sacks harvested per day -/
def sacks_per_day : ℕ := 38

/-- The number of days of harvest -/
def days_of_harvest : ℕ := 49

/-- The total number of sacks after the harvest period -/
def total_sacks : ℕ := 1862

/-- The number of oranges in each sack -/
def oranges_per_sack : ℕ := 42

/-- Theorem stating that the number of sacks harvested per day is correct -/
theorem correct_sacks_per_day : 
  sacks_per_day * days_of_harvest = total_sacks :=
sorry

end NUMINAMATH_CALUDE_correct_sacks_per_day_l1324_132424


namespace NUMINAMATH_CALUDE_nice_sequence_divisibility_exists_nice_sequence_not_divisible_l1324_132486

/-- Definition of a nice sequence -/
def NiceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ (∀ n, a (2 * n) = 2 * a n)

theorem nice_sequence_divisibility (a : ℕ → ℕ) (p : ℕ) (hp : Prime p) (h_nice : NiceSequence a) (h_p_gt_a1 : p > a 1) :
  ∃ k, p ∣ a k := by
  sorry

theorem exists_nice_sequence_not_divisible (p : ℕ) (hp : Prime p) (h_p_gt_2 : p > 2) :
  ∃ a : ℕ → ℕ, NiceSequence a ∧ ∀ n, ¬(p ∣ a n) := by
  sorry

end NUMINAMATH_CALUDE_nice_sequence_divisibility_exists_nice_sequence_not_divisible_l1324_132486


namespace NUMINAMATH_CALUDE_f_2005_equals_2_l1324_132482

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_2005_equals_2 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : ∀ x, f (x + 6) = f x + f 3)
  (h_f_1 : f 1 = 2) :
  f 2005 = 2 := by
sorry

end NUMINAMATH_CALUDE_f_2005_equals_2_l1324_132482


namespace NUMINAMATH_CALUDE_sin_translation_equivalence_l1324_132484

theorem sin_translation_equivalence :
  ∀ x : ℝ, 2 * Real.sin (3 * x + π / 6) = 2 * Real.sin (3 * (x + π / 18)) :=
by sorry

end NUMINAMATH_CALUDE_sin_translation_equivalence_l1324_132484


namespace NUMINAMATH_CALUDE_regular_ngon_construction_l1324_132471

/-- Theorem about the construction of points on the extensions of a regular n-gon's sides -/
theorem regular_ngon_construction (n : ℕ) (a : ℝ) (h_n : n ≥ 5) :
  let α : ℝ := π - (2 * π) / n
  ∀ (x : ℕ → ℝ), 
    (∀ k, x k = (a + x ((k + 1) % n)) * Real.cos α) →
    ∀ k, x k = (a * Real.cos α) / (1 - Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_regular_ngon_construction_l1324_132471


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l1324_132475

theorem unique_triplet_solution (a b p : ℕ+) (h_prime : Nat.Prime p) :
  (a + b : ℕ+) ^ (p : ℕ) = p ^ (a : ℕ) + p ^ (b : ℕ) ↔ a = 1 ∧ b = 1 ∧ p = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l1324_132475


namespace NUMINAMATH_CALUDE_angle_measure_from_area_ratio_l1324_132481

/-- Given three concentric circles and two lines passing through their center,
    prove that the acute angle between the lines is 12π/77 radians when the
    shaded area is 3/4 of the unshaded area. -/
theorem angle_measure_from_area_ratio :
  ∀ (r₁ r₂ r₃ : ℝ) (shaded_area unshaded_area : ℝ) (θ : ℝ),
  r₁ = 4 →
  r₂ = 3 →
  r₃ = 2 →
  shaded_area = (3/4) * unshaded_area →
  shaded_area + unshaded_area = π * (r₁^2 + r₂^2 + r₃^2) →
  shaded_area = θ * (r₁^2 + r₃^2) + (π - θ) * r₂^2 →
  θ = 12 * π / 77 :=
by sorry

end NUMINAMATH_CALUDE_angle_measure_from_area_ratio_l1324_132481


namespace NUMINAMATH_CALUDE_farm_animal_count_l1324_132416

theorem farm_animal_count :
  ∀ (cows chickens ducks : ℕ),
    (4 * cows + 2 * chickens + 2 * ducks = 20 + 2 * (cows + chickens + ducks)) →
    (chickens + ducks = 2 * cows) →
    cows = 10 := by
  sorry

end NUMINAMATH_CALUDE_farm_animal_count_l1324_132416


namespace NUMINAMATH_CALUDE_trajectory_equation_l1324_132426

theorem trajectory_equation (x y : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (y / (x + 1)) * (y / (x - 1)) = -2 → x^2 + y^2 / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1324_132426


namespace NUMINAMATH_CALUDE_inequality_representation_l1324_132425

theorem inequality_representation (x : ℝ) : 
  (x + 4 < 10) ↔ (∃ y, y = x + 4 ∧ y < 10) :=
sorry

end NUMINAMATH_CALUDE_inequality_representation_l1324_132425


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_f_l1324_132492

noncomputable def f (x : ℝ) : ℤ :=
  if x > 0 then Int.ceil (1 / (x + 1))
  else if x < 0 then Int.ceil (1 / (x - 1))
  else 0  -- This value doesn't matter as we exclude x = 0

theorem zero_not_in_range_of_f :
  ∀ x : ℝ, x ≠ 0 → f x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_f_l1324_132492


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l1324_132497

theorem min_value_quadratic_expression :
  ∃ (min_val : ℝ), min_val = -7208 ∧
  ∀ (x y : ℝ), 2*x^2 + 3*x*y + 4*y^2 - 8*x - 10*y ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l1324_132497


namespace NUMINAMATH_CALUDE_simplify_expression_l1324_132451

theorem simplify_expression (x y : ℝ) :
  4 * x + 8 * x^2 + y^3 + 6 - (3 - 4 * x - 8 * x^2 - y^3) = 16 * x^2 + 8 * x + 2 * y^3 + 3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1324_132451


namespace NUMINAMATH_CALUDE_expression_lower_bound_l1324_132459

theorem expression_lower_bound (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : b ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (a - c)^2) / b^2 ≥ 2.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_lower_bound_l1324_132459


namespace NUMINAMATH_CALUDE_inequalities_for_negative_a_l1324_132409

theorem inequalities_for_negative_a (a b : ℝ) (ha : a < 0) :
  (a < b) ∧ (a^2 + b^2 > 2) ∧ 
  (∃ b, ¬(a + b < a*b)) ∧ (∃ b, ¬(|a| > |b|)) :=
sorry

end NUMINAMATH_CALUDE_inequalities_for_negative_a_l1324_132409


namespace NUMINAMATH_CALUDE_rectangle_width_l1324_132406

theorem rectangle_width (length width : ℝ) : 
  length / width = 6 / 5 → length = 24 → width = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l1324_132406


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1324_132430

theorem quadratic_solution_difference_squared :
  ∀ f g : ℝ,
  (2 * f^2 + 8 * f - 42 = 0) →
  (2 * g^2 + 8 * g - 42 = 0) →
  (f ≠ g) →
  (f - g)^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1324_132430


namespace NUMINAMATH_CALUDE_probability_at_least_one_heart_or_king_l1324_132474

def standard_deck : ℕ := 52
def hearts_and_kings : ℕ := 16

theorem probability_at_least_one_heart_or_king :
  let p : ℚ := 1 - (1 - hearts_and_kings / standard_deck) ^ 2
  p = 88 / 169 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_heart_or_king_l1324_132474
