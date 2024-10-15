import Mathlib

namespace NUMINAMATH_CALUDE_marbles_redistribution_l2383_238373

/-- The number of marbles Tyrone gives to Eric -/
def marblesGiven : ℕ := 19

/-- The initial number of marbles Tyrone had -/
def tyronesInitial : ℕ := 120

/-- The initial number of marbles Eric had -/
def ericsInitial : ℕ := 15

/-- Proposition: The number of marbles Tyrone gave to Eric satisfies the conditions -/
theorem marbles_redistribution :
  (tyronesInitial - marblesGiven) = 3 * (ericsInitial + marblesGiven) ∧
  marblesGiven > 0 ∧
  marblesGiven < tyronesInitial :=
by
  sorry

#check marbles_redistribution

end NUMINAMATH_CALUDE_marbles_redistribution_l2383_238373


namespace NUMINAMATH_CALUDE_blake_change_l2383_238330

-- Define the quantities and prices
def num_lollipops : ℕ := 4
def num_chocolate_packs : ℕ := 6
def lollipop_price : ℕ := 2
def num_bills : ℕ := 6
def bill_value : ℕ := 10

-- Define the relationship between chocolate and lollipop prices
def chocolate_pack_price : ℕ := 4 * lollipop_price

-- Calculate the total cost
def total_cost : ℕ := num_lollipops * lollipop_price + num_chocolate_packs * chocolate_pack_price

-- Calculate the amount given
def amount_given : ℕ := num_bills * bill_value

-- Theorem to prove
theorem blake_change : amount_given - total_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_blake_change_l2383_238330


namespace NUMINAMATH_CALUDE_proposition_1_proposition_4_l2383_238385

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Axioms for the relations
axiom perpendicular_def (l : Line) (p : Plane) :
  perpendicular l p → ∀ (l' : Line), parallel_line_plane l' p → perpendicular_lines l l'

axiom parallel_plane_trans (p1 p2 p3 : Plane) :
  parallel_plane p1 p2 → parallel_plane p2 p3 → parallel_plane p1 p3

axiom perpendicular_parallel (l : Line) (p1 p2 : Plane) :
  perpendicular l p1 → parallel_plane p1 p2 → perpendicular l p2

-- Theorem 1
theorem proposition_1 (m n : Line) (α : Plane) :
  perpendicular m α → parallel_line_plane n α → perpendicular_lines m n := by sorry

-- Theorem 2
theorem proposition_4 (m : Line) (α β γ : Plane) :
  parallel_plane α β → parallel_plane β γ → perpendicular m α → perpendicular m γ := by sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_4_l2383_238385


namespace NUMINAMATH_CALUDE_rectangles_in_3x2_grid_l2383_238362

/-- The number of rectangles in a grid -/
def count_rectangles (m n : ℕ) : ℕ :=
  let one_by_one := m * n
  let one_by_two := m * (n - 1)
  let two_by_one := (m - 1) * n
  let two_by_two := (m - 1) * (n - 1)
  one_by_one + one_by_two + two_by_one + two_by_two

/-- Theorem: The number of rectangles in a 3x2 grid is 14 -/
theorem rectangles_in_3x2_grid :
  count_rectangles 3 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_3x2_grid_l2383_238362


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2383_238389

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and vectors m = (√3, -1) and n = (cos A, sin A), prove that if m ⊥ n and
    a * cos B + b * cos A = c * sin C, then B = π/6. -/
theorem triangle_angle_proof (a b c A B C : ℝ) (m n : ℝ × ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  m = (Real.sqrt 3, -1) →
  n = (Real.cos A, Real.sin A) →
  m.1 * n.1 + m.2 * n.2 = 0 →
  a * Real.cos B + b * Real.cos A = c * Real.sin C →
  B = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2383_238389


namespace NUMINAMATH_CALUDE_set_operations_l2383_238340

-- Define the sets A and B
def A : Set ℝ := {x | x < 1 ∨ x > 2}
def B : Set ℝ := {x | x < -3 ∨ x ≥ 1}

-- State the theorem
theorem set_operations :
  (Set.univ \ A = {x | 1 ≤ x ∧ x ≤ 2}) ∧
  (Set.univ \ B = {x | -3 ≤ x ∧ x < 1}) ∧
  (A ∩ B = {x | x < -3 ∨ x > 2}) ∧
  (A ∪ B = Set.univ) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2383_238340


namespace NUMINAMATH_CALUDE_divisibility_properties_l2383_238306

theorem divisibility_properties (a : ℤ) : 
  (∃ k : ℤ, a^5 - a = 30 * k) ∧
  (∃ l : ℤ, a^17 - a = 510 * l) ∧
  (∃ m : ℤ, a^11 - a = 66 * m) ∧
  (∃ n : ℤ, a^73 - a = (2 * 3 * 5 * 7 * 13 * 19 * 37 * 73) * n) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_properties_l2383_238306


namespace NUMINAMATH_CALUDE_negation_equivalence_l2383_238352

theorem negation_equivalence : 
  (¬∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2383_238352


namespace NUMINAMATH_CALUDE_hen_count_is_28_l2383_238350

/-- Represents the count of animals on a farm -/
structure FarmCount where
  hens : ℕ
  cows : ℕ

/-- Checks if the farm count satisfies the given conditions -/
def isValidCount (farm : FarmCount) : Prop :=
  farm.hens + farm.cows = 48 ∧
  2 * farm.hens + 4 * farm.cows = 136

theorem hen_count_is_28 :
  ∃ (farm : FarmCount), isValidCount farm ∧ farm.hens = 28 :=
by
  sorry

#check hen_count_is_28

end NUMINAMATH_CALUDE_hen_count_is_28_l2383_238350


namespace NUMINAMATH_CALUDE_gilled_mushroom_count_l2383_238342

/-- Represents the number of mushrooms on a log -/
structure MushroomCount where
  total : ℕ
  gilled : ℕ
  spotted : ℕ

/-- Conditions for the mushroom problem -/
def mushroom_conditions (m : MushroomCount) : Prop :=
  m.total = 30 ∧
  m.gilled + m.spotted = m.total ∧
  m.spotted = 9 * m.gilled

/-- Theorem stating the number of gilled mushrooms -/
theorem gilled_mushroom_count (m : MushroomCount) :
  mushroom_conditions m → m.gilled = 3 := by
  sorry

end NUMINAMATH_CALUDE_gilled_mushroom_count_l2383_238342


namespace NUMINAMATH_CALUDE_min_value_sum_of_roots_l2383_238359

theorem min_value_sum_of_roots (x : ℝ) :
  ∃ (y : ℝ), (∀ (x : ℝ), Real.sqrt (x^2 + (1 + x)^2) + Real.sqrt ((1 + x)^2 + (1 - x)^2) ≥ Real.sqrt 5) ∧
  (Real.sqrt (y^2 + (1 + y)^2) + Real.sqrt ((1 + y)^2 + (1 - y)^2) = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_roots_l2383_238359


namespace NUMINAMATH_CALUDE_parabola_property_l2383_238320

/-- Given a parabola y = ax² + bx + c with the following points:
    (-2, 0), (-1, 4), (0, 6), (1, 6)
    Prove that (a - b + c)(4a + 2b + c) > 0 -/
theorem parabola_property (a b c : ℝ) : 
  (4 * a - 2 * b + c = 0) →
  (a - b + c = 4) →
  (c = 6) →
  (a * 1^2 + b * 1 + c = 6) →
  (a - b + c) * (4 * a + 2 * b + c) > 0 := by
sorry

end NUMINAMATH_CALUDE_parabola_property_l2383_238320


namespace NUMINAMATH_CALUDE_line_intercepts_l2383_238347

/-- The equation of the line -/
def line_equation (x y : ℚ) : Prop := 4 * x + 7 * y = 28

/-- Definition of x-intercept -/
def is_x_intercept (x : ℚ) : Prop := line_equation x 0

/-- Definition of y-intercept -/
def is_y_intercept (y : ℚ) : Prop := line_equation 0 y

/-- Theorem: The x-intercept of the line 4x + 7y = 28 is (7, 0), and the y-intercept is (0, 4) -/
theorem line_intercepts : is_x_intercept 7 ∧ is_y_intercept 4 := by sorry

end NUMINAMATH_CALUDE_line_intercepts_l2383_238347


namespace NUMINAMATH_CALUDE_zoo_animals_count_l2383_238370

theorem zoo_animals_count (zebras camels monkeys giraffes : ℕ) : 
  zebras = 12 →
  camels = zebras / 2 →
  monkeys = 4 * camels →
  monkeys = giraffes + 22 →
  giraffes = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_count_l2383_238370


namespace NUMINAMATH_CALUDE_sin_alpha_minus_pi_sixth_l2383_238353

theorem sin_alpha_minus_pi_sixth (α : Real) 
  (h : Real.sin (α + π/6) + 2 * Real.sin (α/2)^2 = 1 - Real.sqrt 2 / 2) : 
  Real.sin (α - π/6) = - Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_pi_sixth_l2383_238353


namespace NUMINAMATH_CALUDE_square_of_sum_l2383_238335

theorem square_of_sum (x y : ℝ) 
  (h1 : 3 * x * (2 * x + y) = 14) 
  (h2 : y * (2 * x + y) = 35) : 
  (2 * x + y)^2 = 49 := by sorry

end NUMINAMATH_CALUDE_square_of_sum_l2383_238335


namespace NUMINAMATH_CALUDE_custom_op_result_l2383_238374

def custom_op (a b : ℤ) : ℤ := b^2 - a*b

theorem custom_op_result : custom_op (custom_op (-1) 2) 3 = -9 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l2383_238374


namespace NUMINAMATH_CALUDE_jared_popcorn_theorem_l2383_238322

/-- The number of pieces of popcorn in a serving -/
def popcorn_per_serving : ℕ := 30

/-- The number of pieces of popcorn each of Jared's friends can eat -/
def friend_popcorn_consumption : ℕ := 60

/-- The number of Jared's friends -/
def number_of_friends : ℕ := 3

/-- The number of servings Jared should order -/
def servings_ordered : ℕ := 9

/-- The number of pieces of popcorn Jared can eat -/
def jared_popcorn_consumption : ℕ := 
  servings_ordered * popcorn_per_serving - number_of_friends * friend_popcorn_consumption

theorem jared_popcorn_theorem : jared_popcorn_consumption = 90 := by
  sorry

end NUMINAMATH_CALUDE_jared_popcorn_theorem_l2383_238322


namespace NUMINAMATH_CALUDE_beach_attendance_l2383_238349

theorem beach_attendance (initial_group : ℕ) (joined : ℕ) (left : ℕ) : 
  initial_group = 3 → joined = 100 → left = 40 → 
  initial_group + joined - left = 63 := by
  sorry

end NUMINAMATH_CALUDE_beach_attendance_l2383_238349


namespace NUMINAMATH_CALUDE_average_of_five_numbers_l2383_238361

theorem average_of_five_numbers (x : ℝ) : 
  (3 + 5 + 6 + 8 + x) / 5 = 7 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_numbers_l2383_238361


namespace NUMINAMATH_CALUDE_rotation_transform_triangles_l2383_238399

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Clockwise rotation around a point -/
def rotateClockwise (center : Point) (angle : ℝ) (p : Point) : Point :=
  sorry

/-- Check if two triangles are congruent -/
def areCongruent (t1 t2 : Triangle) : Prop :=
  sorry

theorem rotation_transform_triangles (m x y : ℝ) : 
  let ABC := Triangle.mk (Point.mk 0 0) (Point.mk 0 12) (Point.mk 16 0)
  let A'B'C' := Triangle.mk (Point.mk 24 18) (Point.mk 36 18) (Point.mk 24 2)
  let center := Point.mk x y
  0 < m → m < 180 →
  (areCongruent (Triangle.mk 
    (rotateClockwise center m ABC.A)
    (rotateClockwise center m ABC.B)
    (rotateClockwise center m ABC.C)) A'B'C') →
  m + x + y = 108 :=
sorry

end NUMINAMATH_CALUDE_rotation_transform_triangles_l2383_238399


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2383_238393

theorem quadratic_equation_solution :
  let a : ℝ := 2
  let b : ℝ := -5
  let c : ℝ := 3
  let x₁ : ℝ := 3/2
  let x₂ : ℝ := 1
  (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equation_solution_l2383_238393


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l2383_238371

/-- Represents the number of balls and boxes -/
def n : ℕ := 5

/-- Calculates the number of ways to place n balls into n boxes with one empty box -/
def ways_one_empty (n : ℕ) : ℕ := sorry

/-- Calculates the number of ways to place n balls into n boxes with no empty box and not all numbers matching -/
def ways_no_empty_not_all_match (n : ℕ) : ℕ := sorry

/-- Calculates the number of ways to place n balls into n boxes with one ball in each box and at least two balls matching their box numbers -/
def ways_at_least_two_match (n : ℕ) : ℕ := sorry

/-- Theorem stating the correct number of ways for each scenario with 5 balls and 5 boxes -/
theorem ball_placement_theorem :
  ways_one_empty n = 1200 ∧
  ways_no_empty_not_all_match n = 119 ∧
  ways_at_least_two_match n = 31 := by sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l2383_238371


namespace NUMINAMATH_CALUDE_friday_ice_cream_amount_l2383_238375

/-- The amount of ice cream eaten on Friday night -/
def friday_ice_cream : ℝ := 3.5 - 0.25

/-- The total amount of ice cream eaten over two nights -/
def total_ice_cream : ℝ := 3.5

/-- The amount of ice cream eaten on Saturday night -/
def saturday_ice_cream : ℝ := 0.25

/-- Proof that the amount of ice cream eaten on Friday night is 3.25 pints -/
theorem friday_ice_cream_amount : friday_ice_cream = 3.25 := by
  sorry

end NUMINAMATH_CALUDE_friday_ice_cream_amount_l2383_238375


namespace NUMINAMATH_CALUDE_power_division_rule_l2383_238311

theorem power_division_rule (a : ℝ) : a^5 / a^3 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l2383_238311


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2383_238395

/-- Given a circle with two inscribed squares:
    - The first square is inscribed in the circle
    - The second square is inscribed in the segment of the circle cut off by one side of the first square
    This theorem states that the ratio of the side lengths of these squares is 5:1 -/
theorem inscribed_squares_ratio (r : ℝ) (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (2 * a) ^ 2 + (2 * a) ^ 2 = (2 * r) ^ 2 →  -- First square inscribed in circle
  (a + 2 * b) ^ 2 + b ^ 2 = r ^ 2 →          -- Second square inscribed in segment
  a / b = 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2383_238395


namespace NUMINAMATH_CALUDE_fourth_month_sale_proof_l2383_238323

/-- Calculates the sale in the fourth month given sales for other months and the average --/
def fourthMonthSale (sale1 sale2 sale3 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale3 + sale5 + sale6)

theorem fourth_month_sale_proof (sale1 sale2 sale3 sale5 sale6 average : ℕ) 
  (h1 : sale1 = 3435)
  (h2 : sale2 = 3927)
  (h3 : sale3 = 3855)
  (h5 : sale5 = 3562)
  (h6 : sale6 = 1991)
  (h_avg : average = 3500) :
  fourthMonthSale sale1 sale2 sale3 sale5 sale6 average = 4230 := by
  sorry

#eval fourthMonthSale 3435 3927 3855 3562 1991 3500

end NUMINAMATH_CALUDE_fourth_month_sale_proof_l2383_238323


namespace NUMINAMATH_CALUDE_september_solution_l2383_238313

/-- A function that maps a month number to its corresponding solution in the equations -/
def month_solution : ℕ → ℝ
| 2 => 2  -- February
| 4 => 4  -- April
| 9 => 9  -- September
| _ => 0  -- Other months (not relevant for this problem)

/-- The theorem stating that the solution of 48 = 5x + 3 corresponds to the 9th month -/
theorem september_solution :
  (month_solution 2 - 1 = 1) ∧
  (18 - 2 * month_solution 4 = 10) ∧
  (48 = 5 * month_solution 9 + 3) := by
  sorry

#check september_solution

end NUMINAMATH_CALUDE_september_solution_l2383_238313


namespace NUMINAMATH_CALUDE_tony_fish_count_l2383_238345

def fish_count (initial : ℕ) (years : ℕ) (yearly_addition : ℕ) (yearly_loss : ℕ) : ℕ :=
  initial + years * (yearly_addition - yearly_loss)

theorem tony_fish_count :
  fish_count 2 5 2 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tony_fish_count_l2383_238345


namespace NUMINAMATH_CALUDE_election_votes_theorem_l2383_238336

theorem election_votes_theorem :
  ∀ (total_votes : ℕ) (valid_votes : ℕ) (invalid_votes : ℕ),
    invalid_votes = 100 →
    valid_votes = total_votes - invalid_votes →
    ∃ (loser_votes winner_votes : ℕ),
      loser_votes = (30 * valid_votes) / 100 ∧
      winner_votes = valid_votes - loser_votes ∧
      winner_votes = loser_votes + 5000 →
      total_votes = 12600 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l2383_238336


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_range_l2383_238382

/-- The function f(x) defined in terms of parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (a / (a^2 - 2)) * (a^x - a^(-x))

/-- Theorem stating the conditions for f to be an increasing function -/
theorem f_increasing_iff_a_range (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (a > Real.sqrt 2 ∨ 0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_range_l2383_238382


namespace NUMINAMATH_CALUDE_train_platform_problem_l2383_238301

/-- Represents the speed of the train in km/hr -/
def train_speed : ℝ := 72

/-- Represents the time taken to cross the platform in minutes -/
def crossing_time : ℝ := 1

/-- Calculates the length of the train in meters -/
def train_length : ℝ := 600

theorem train_platform_problem :
  ∀ (train_length platform_length : ℝ),
  train_length = platform_length →
  train_length = train_speed * (1000 / 3600) * (crossing_time * 60) / 2 →
  train_length = 600 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_problem_l2383_238301


namespace NUMINAMATH_CALUDE_inequality_proof_l2383_238360

theorem inequality_proof (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2383_238360


namespace NUMINAMATH_CALUDE_sector_to_cone_base_radius_l2383_238379

/-- Given a sector with central angle 120° and radius 9 cm, when formed into a cone,
    the radius of the base circle is 3 cm. -/
theorem sector_to_cone_base_radius (θ : ℝ) (R : ℝ) (r : ℝ) : 
  θ = 120 → R = 9 → r = (θ / 360) * R → r = 3 :=
by sorry

end NUMINAMATH_CALUDE_sector_to_cone_base_radius_l2383_238379


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l2383_238367

/-- Given a line with equation 3x - 4y + 5 = 0, its symmetric line with respect to the y-axis
    has the equation 3x + 4y - 5 = 0. -/
theorem symmetric_line_wrt_y_axis :
  ∀ (x y : ℝ), (3 * (-x) - 4 * y + 5 = 0) ↔ (3 * x + 4 * y - 5 = 0) := by
sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l2383_238367


namespace NUMINAMATH_CALUDE_initial_population_size_l2383_238310

/-- Given a population with specific birth and death rates, prove the initial population size. -/
theorem initial_population_size 
  (P : ℕ) 
  (birth_rate : ℕ) 
  (death_rate : ℕ) 
  (net_growth_rate : ℚ) 
  (h1 : birth_rate = 52) 
  (h2 : death_rate = 16) 
  (h3 : net_growth_rate = 12/1000) 
  (h4 : (birth_rate - death_rate : ℚ) / P = net_growth_rate) : 
  P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_initial_population_size_l2383_238310


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2383_238396

theorem pure_imaginary_complex_number (x : ℝ) : 
  (((x^2 - 4) : ℂ) + (x^2 + 3*x + 2)*I = (0 : ℂ) + y*I ∧ y ≠ 0) → x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2383_238396


namespace NUMINAMATH_CALUDE_parkway_soccer_players_l2383_238343

theorem parkway_soccer_players (total_students : ℕ) (boys : ℕ) (girls_not_playing : ℕ) 
  (h1 : total_students = 470)
  (h2 : boys = 300)
  (h3 : girls_not_playing = 135)
  (h4 : (86 : ℚ) / 100 * (total_students - (total_students - boys - girls_not_playing)) = boys - (total_students - boys - girls_not_playing)) :
  total_students - (total_students - boys - girls_not_playing) = 250 := by
  sorry

end NUMINAMATH_CALUDE_parkway_soccer_players_l2383_238343


namespace NUMINAMATH_CALUDE_northern_shoe_capital_relocation_l2383_238326

structure XionganNewArea where
  green_ecological : Bool
  innovation_driven : Bool
  coordinated_development : Bool
  open_development : Bool

structure AnxinCounty where
  santai_town : Bool
  traditional_shoemaking : Bool
  northern_shoe_capital : Bool
  nationwide_market : Bool
  adequate_transportation : Bool

def industrial_structure_adjustment (county : AnxinCounty) (new_area : XionganNewArea) : Bool :=
  county.traditional_shoemaking ∧ 
  (new_area.green_ecological ∧ new_area.innovation_driven ∧ 
   new_area.coordinated_development ∧ new_area.open_development)

def relocation_reason (county : AnxinCounty) (new_area : XionganNewArea) : String :=
  if industrial_structure_adjustment county new_area then
    "Industrial structure adjustment"
  else
    "Other reasons"

theorem northern_shoe_capital_relocation 
  (anxin : AnxinCounty) 
  (xiong_an : XionganNewArea) 
  (h1 : anxin.santai_town = true)
  (h2 : anxin.traditional_shoemaking = true)
  (h3 : anxin.northern_shoe_capital = true)
  (h4 : anxin.nationwide_market = true)
  (h5 : anxin.adequate_transportation = true)
  (h6 : xiong_an.green_ecological = true)
  (h7 : xiong_an.innovation_driven = true)
  (h8 : xiong_an.coordinated_development = true)
  (h9 : xiong_an.open_development = true) :
  relocation_reason anxin xiong_an = "Industrial structure adjustment" := by
  sorry

#check northern_shoe_capital_relocation

end NUMINAMATH_CALUDE_northern_shoe_capital_relocation_l2383_238326


namespace NUMINAMATH_CALUDE_inequality_proof_l2383_238334

theorem inequality_proof (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2383_238334


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2383_238355

def A : Set Nat := {1, 3, 5, 7}
def B : Set Nat := {4, 5, 6, 7}

theorem intersection_of_A_and_B : A ∩ B = {5, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2383_238355


namespace NUMINAMATH_CALUDE_tourist_tax_theorem_l2383_238331

/-- Calculates the tax paid given the total value of goods -/
def tax_paid (total_value : ℝ) : ℝ :=
  0.08 * (total_value - 600)

/-- Theorem stating that if $89.6 tax is paid, the total value of goods is $1720 -/
theorem tourist_tax_theorem (total_value : ℝ) :
  tax_paid total_value = 89.6 → total_value = 1720 := by
  sorry

end NUMINAMATH_CALUDE_tourist_tax_theorem_l2383_238331


namespace NUMINAMATH_CALUDE_pirate_total_distance_l2383_238339

def island1_distances : List ℝ := [10, 15, 20]
def island1_increase : ℝ := 1.1
def island2_distance : ℝ := 40
def island2_increase : ℝ := 1.15
def island3_morning : ℝ := 25
def island3_afternoon : ℝ := 20
def island3_days : ℕ := 2
def island3_increase : ℝ := 1.2
def island4_distance : ℝ := 35
def island4_increase : ℝ := 1.25

theorem pirate_total_distance :
  let island1_total := (island1_distances.map (· * island1_increase)).sum
  let island2_total := island2_distance * island2_increase
  let island3_total := (island3_morning + island3_afternoon) * island3_increase * island3_days
  let island4_total := island4_distance * island4_increase
  island1_total + island2_total + island3_total + island4_total = 247.25 := by
  sorry

end NUMINAMATH_CALUDE_pirate_total_distance_l2383_238339


namespace NUMINAMATH_CALUDE_max_value_theorem_l2383_238312

theorem max_value_theorem (x y : ℝ) (h : x^2 + y^2 = 18*x + 8*y + 10) :
  ∀ z : ℝ, 4*x + 3*y ≤ z → z ≤ 63 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2383_238312


namespace NUMINAMATH_CALUDE_sad_children_count_l2383_238397

theorem sad_children_count (total_children : ℕ) (happy_children : ℕ) (neither_happy_nor_sad : ℕ)
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neither_happy_nor_sad_boys : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : neither_happy_nor_sad = 20)
  (h4 : boys = 17)
  (h5 : girls = 43)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neither_happy_nor_sad_boys = 5)
  (h9 : total_children = happy_children + neither_happy_nor_sad + (total_children - happy_children - neither_happy_nor_sad))
  (h10 : boys + girls = total_children) :
  total_children - happy_children - neither_happy_nor_sad = 10 := by
  sorry

end NUMINAMATH_CALUDE_sad_children_count_l2383_238397


namespace NUMINAMATH_CALUDE_train_speed_l2383_238308

theorem train_speed (t_pole : ℝ) (t_cross : ℝ) (l_stationary : ℝ) :
  t_pole = 8 →
  t_cross = 18 →
  l_stationary = 400 →
  ∃ v l, v = l / t_pole ∧ v = (l + l_stationary) / t_cross ∧ v = 40 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l2383_238308


namespace NUMINAMATH_CALUDE_complex_equation_implies_real_equation_l2383_238341

theorem complex_equation_implies_real_equation (a b : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (3 + 4 * Complex.I) * (a + b * Complex.I) = 10 * Complex.I →
  3 * a - 4 * b = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_implies_real_equation_l2383_238341


namespace NUMINAMATH_CALUDE_square_form_existence_l2383_238332

theorem square_form_existence (a b : ℕ+) (h : a.val^3 + 4 * a.val = b.val^2) :
  ∃ t : ℕ+, a.val = 2 * t.val^2 := by
sorry

end NUMINAMATH_CALUDE_square_form_existence_l2383_238332


namespace NUMINAMATH_CALUDE_incorrect_expression_l2383_238381

theorem incorrect_expression (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬(b / a > 1) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l2383_238381


namespace NUMINAMATH_CALUDE_stone_growth_prevention_l2383_238300

/-- The amount of stone consumed by one warrior per day -/
def warrior_consumption : ℝ := 1

/-- The number of days it takes for the stone to pierce the sky with 14 warriors -/
def days_with_14 : ℕ := 16

/-- The number of days it takes for the stone to pierce the sky with 15 warriors -/
def days_with_15 : ℕ := 24

/-- The daily growth rate of the stone -/
def stone_growth_rate : ℝ := 17 * warrior_consumption

/-- The minimum number of warriors needed to prevent the stone from piercing the sky -/
def min_warriors : ℕ := 17

theorem stone_growth_prevention :
  (↑min_warriors * warrior_consumption = stone_growth_rate) ∧
  (∀ n : ℕ, n < min_warriors → ↑n * warrior_consumption < stone_growth_rate) := by
  sorry

#check stone_growth_prevention

end NUMINAMATH_CALUDE_stone_growth_prevention_l2383_238300


namespace NUMINAMATH_CALUDE_probability_ones_not_adjacent_l2383_238364

def total_arrangements : ℕ := 10

def favorable_arrangements : ℕ := 6

theorem probability_ones_not_adjacent :
  (favorable_arrangements : ℚ) / total_arrangements = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_ones_not_adjacent_l2383_238364


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l2383_238348

/-- The quadratic equation x^2 - 6kx + 9k has exactly one real root if and only if k = 1, where k is positive. -/
theorem unique_root_quadratic (k : ℝ) (h : k > 0) :
  (∃! x : ℝ, x^2 - 6*k*x + 9*k = 0) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l2383_238348


namespace NUMINAMATH_CALUDE_rectangle_to_tetrahedron_sphere_area_l2383_238354

/-- A rectangle ABCD with sides AB and BC -/
structure Rectangle where
  AB : ℝ
  BC : ℝ

/-- A tetrahedron formed by folding a rectangle along its diagonal -/
structure Tetrahedron where
  base : Rectangle

/-- The surface area of the circumscribed sphere of a tetrahedron -/
def circumscribed_sphere_area (t : Tetrahedron) : ℝ := sorry

theorem rectangle_to_tetrahedron_sphere_area 
  (r : Rectangle) 
  (h1 : r.AB = 8) 
  (h2 : r.BC = 6) : 
  circumscribed_sphere_area (Tetrahedron.mk r) = 100 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_tetrahedron_sphere_area_l2383_238354


namespace NUMINAMATH_CALUDE_count_convex_polygons_l2383_238309

/-- A point in the 2D plane with integer coordinates -/
structure Point :=
  (x : ℕ)
  (y : ℕ)

/-- A convex polygon with vertices as a list of points -/
structure ConvexPolygon :=
  (vertices : List Point)
  (is_convex : Bool)

/-- Function to check if a polygon contains the required three consecutive vertices -/
def has_required_vertices (p : ConvexPolygon) : Bool :=
  sorry

/-- Function to count the number of valid convex polygons -/
def count_valid_polygons : ℕ :=
  sorry

/-- The main theorem stating that the count of valid convex polygons is 77 -/
theorem count_convex_polygons :
  count_valid_polygons = 77 :=
sorry

end NUMINAMATH_CALUDE_count_convex_polygons_l2383_238309


namespace NUMINAMATH_CALUDE_two_dice_same_side_probability_l2383_238302

/-- Represents a 10-sided die with specific side distributions -/
structure TenSidedDie :=
  (gold : Nat)
  (silver : Nat)
  (diamond : Nat)
  (rainbow : Nat)
  (total : Nat)
  (sides_sum : gold + silver + diamond + rainbow = total)

/-- The probability of rolling two dice and getting the same color or pattern -/
def sameSideProbability (die : TenSidedDie) : ℚ :=
  (die.gold ^ 2 + die.silver ^ 2 + die.diamond ^ 2 + die.rainbow ^ 2) / die.total ^ 2

/-- Theorem: The probability of rolling two 10-sided dice with the given distribution
    and getting the same color or pattern is 3/10 -/
theorem two_dice_same_side_probability :
  ∃ (die : TenSidedDie),
    die.gold = 3 ∧
    die.silver = 4 ∧
    die.diamond = 2 ∧
    die.rainbow = 1 ∧
    die.total = 10 ∧
    sameSideProbability die = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_two_dice_same_side_probability_l2383_238302


namespace NUMINAMATH_CALUDE_smallest_square_cover_l2383_238388

/-- The smallest square that can be covered by 3x4 rectangles -/
def smallest_square_side : ℕ := 12

/-- The area of a 3x4 rectangle -/
def rectangle_area : ℕ := 3 * 4

/-- The number of 3x4 rectangles needed to cover the smallest square -/
def num_rectangles : ℕ := smallest_square_side^2 / rectangle_area

theorem smallest_square_cover :
  ∀ (side : ℕ), 
  side % smallest_square_side = 0 →
  side^2 % rectangle_area = 0 →
  (side^2 / rectangle_area ≥ num_rectangles) ∧
  (num_rectangles * rectangle_area = smallest_square_side^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_cover_l2383_238388


namespace NUMINAMATH_CALUDE_pencil_count_l2383_238303

/-- The number of pencils Reeta has -/
def reeta_pencils : ℕ := 30

/-- The number of pencils Anika has -/
def anika_pencils : ℕ := 2 * reeta_pencils + 4

/-- The number of pencils Kamal has -/
def kamal_pencils : ℕ := 3 * reeta_pencils - 2

/-- The total number of pencils all three have together -/
def total_pencils : ℕ := reeta_pencils + anika_pencils + kamal_pencils

theorem pencil_count : total_pencils = 182 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l2383_238303


namespace NUMINAMATH_CALUDE_lucas_100th_mod8_l2383_238307

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | (n + 2) => lucas n + lucas (n + 1)

def lucas_mod8 (n : ℕ) : ℕ := lucas n % 8

theorem lucas_100th_mod8 : lucas_mod8 99 = 7 := by sorry

end NUMINAMATH_CALUDE_lucas_100th_mod8_l2383_238307


namespace NUMINAMATH_CALUDE_batsman_highest_score_l2383_238337

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46)
  (h1 : average = 60)
  (h2 : score_difference = 140)
  (h3 : average_excluding_extremes = 58) : 
  ∃ (highest_score lowest_score : ℕ), 
    highest_score - lowest_score = score_difference ∧ 
    (total_innings : ℚ) * average = 
      ((total_innings - 2 : ℚ) * average_excluding_extremes + highest_score + lowest_score) ∧
    highest_score = 174 := by
  sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l2383_238337


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2383_238315

/-- Given a repeating decimal 3.565656..., prove it equals 353/99 -/
theorem repeating_decimal_to_fraction : 
  ∀ (x : ℚ), (∃ (n : ℕ), x = 3 + (56 : ℚ) / (10^2 - 1) / 10^n) → x = 353 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2383_238315


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2383_238368

/-- A geometric sequence with sum of first n terms S_n = 4^n + a has a = -1 -/
theorem geometric_sequence_sum (S : ℕ → ℝ) (a : ℝ) :
  (∀ n : ℕ, S n = 4^n + a) →
  (∃ r : ℝ, ∀ n : ℕ, S (n + 1) - S n = r * (S n - S (n - 1))) →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2383_238368


namespace NUMINAMATH_CALUDE_fifth_term_value_l2383_238372

def a (n : ℕ+) : ℚ := n / (n^2 + 25)

theorem fifth_term_value : a 5 = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l2383_238372


namespace NUMINAMATH_CALUDE_coins_missing_l2383_238351

theorem coins_missing (initial : ℚ) : 
  initial > 0 → 
  let lost := (1 : ℚ) / 3 * initial
  let found := (2 : ℚ) / 3 * lost
  let remaining := initial - lost + found
  (initial - remaining) / initial = (1 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_coins_missing_l2383_238351


namespace NUMINAMATH_CALUDE_river_current_speed_l2383_238383

/-- Represents the speed of a motorboat in various conditions -/
structure MotorboatSpeed where
  still : ℝ  -- Speed in still water
  current : ℝ  -- River current speed
  wind : ℝ  -- Wind speed (positive for tailwind, negative for headwind)

/-- Calculates the effective speed of the motorboat -/
def effectiveSpeed (s : MotorboatSpeed) : ℝ := s.still + s.current + s.wind

/-- Theorem: River current speed is 1 mile per hour -/
theorem river_current_speed 
  (distance : ℝ) 
  (downstream_time upstream_time : ℝ) 
  (h : distance = 24 ∧ downstream_time = 4 ∧ upstream_time = 6) 
  (s : MotorboatSpeed) 
  (h_downstream : effectiveSpeed { still := s.still, current := s.current, wind := -s.wind } * downstream_time = distance) 
  (h_upstream : effectiveSpeed { still := s.still, current := -s.current, wind := s.wind } * upstream_time = distance) :
  s.current = 1 := by
  sorry

end NUMINAMATH_CALUDE_river_current_speed_l2383_238383


namespace NUMINAMATH_CALUDE_lady_bird_biscuits_l2383_238398

/-- The number of biscuits Lady Bird can make with a given amount of flour -/
def biscuits_from_flour (flour : ℚ) : ℚ :=
  (flour * 9) / (5/4)

/-- The number of biscuits per guest Lady Bird can allow -/
def biscuits_per_guest (total_biscuits : ℚ) (guests : ℕ) : ℚ :=
  total_biscuits / guests

theorem lady_bird_biscuits :
  let flour_used : ℚ := 5
  let num_guests : ℕ := 18
  let total_biscuits := biscuits_from_flour flour_used
  biscuits_per_guest total_biscuits num_guests = 2 := by
sorry

end NUMINAMATH_CALUDE_lady_bird_biscuits_l2383_238398


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l2383_238338

theorem least_positive_linear_combination : 
  ∃ (n : ℕ), n > 0 ∧ (∀ (x y : ℤ), 24 * x + 16 * y = n ∨ 24 * x + 16 * y < 0 ∨ 24 * x + 16 * y > n) ∧ 
  (∃ (a b : ℤ), 24 * a + 16 * b = n) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l2383_238338


namespace NUMINAMATH_CALUDE_peg_arrangement_count_l2383_238327

/-- The number of ways to arrange colored pegs on a triangular board. -/
def arrangeColoredPegs (yellow red green blue orange : Nat) : Nat :=
  Nat.factorial yellow * Nat.factorial red * Nat.factorial green * Nat.factorial blue * Nat.factorial orange

/-- Theorem stating the number of arrangements for the given peg counts. -/
theorem peg_arrangement_count :
  arrangeColoredPegs 6 5 4 3 2 = 12441600 := by
  sorry

end NUMINAMATH_CALUDE_peg_arrangement_count_l2383_238327


namespace NUMINAMATH_CALUDE_log_simplification_l2383_238344

theorem log_simplification : (2 * (Real.log 3 / Real.log 4) + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l2383_238344


namespace NUMINAMATH_CALUDE_expand_product_l2383_238329

theorem expand_product (x : ℝ) : (x + 3) * (x - 8) = x^2 - 5*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2383_238329


namespace NUMINAMATH_CALUDE_count_valid_assignments_l2383_238356

/-- Represents a student --/
inductive Student : Type
| jia : Student
| other : Fin 4 → Student

/-- Represents a dormitory --/
inductive Dormitory : Type
| A : Dormitory
| B : Dormitory
| C : Dormitory

/-- An assignment of students to dormitories --/
def Assignment := Student → Dormitory

/-- Checks if an assignment is valid --/
def isValidAssignment (a : Assignment) : Prop :=
  (∃ s, a s = Dormitory.A) ∧
  (∃ s, a s = Dormitory.B) ∧
  (∃ s, a s = Dormitory.C) ∧
  (a Student.jia ≠ Dormitory.A)

/-- The number of valid assignments --/
def numValidAssignments : ℕ := sorry

theorem count_valid_assignments :
  numValidAssignments = 40 := by sorry

end NUMINAMATH_CALUDE_count_valid_assignments_l2383_238356


namespace NUMINAMATH_CALUDE_local_minimum_condition_l2383_238363

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

-- Define the derivative of f(x)
def f_prime (b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*b

-- Theorem statement
theorem local_minimum_condition (b : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ IsLocalMin (f b) x) →
  (f_prime b 0 < 0 ∧ f_prime b 1 > 0) :=
sorry

end NUMINAMATH_CALUDE_local_minimum_condition_l2383_238363


namespace NUMINAMATH_CALUDE_janabel_widget_sales_l2383_238314

theorem janabel_widget_sales (n : ℕ) (h : n = 15) : 
  let a₁ := 2
  let d := 3
  let aₙ := a₁ + (n - 1) * d
  n / 2 * (a₁ + aₙ) = 345 := by
  sorry

end NUMINAMATH_CALUDE_janabel_widget_sales_l2383_238314


namespace NUMINAMATH_CALUDE_unique_solution_trig_system_l2383_238324

theorem unique_solution_trig_system (x : ℝ) :
  (Real.arccos (3 * x) - Real.arcsin x = π / 6 ∧
   Real.arccos (3 * x) + Real.arcsin x = 5 * π / 6) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trig_system_l2383_238324


namespace NUMINAMATH_CALUDE_shortest_chord_line_l2383_238333

/-- The circle C in the 2D plane -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 2)^2 = 5}

/-- The line l passing through (1,1) -/
def l : Set (ℝ × ℝ) := {p | p.1 - p.2 = 0}

/-- The point (1,1) -/
def A : ℝ × ℝ := (1, 1)

/-- Theorem: The line l intersects the circle C with the shortest chord length -/
theorem shortest_chord_line :
  A ∈ l ∧
  (∃ p q : ℝ × ℝ, p ∈ l ∧ q ∈ l ∧ p ∈ C ∧ q ∈ C ∧ p ≠ q) ∧
  (∀ m : Set (ℝ × ℝ), A ∈ m →
    (∃ p q : ℝ × ℝ, p ∈ m ∧ q ∈ m ∧ p ∈ C ∧ q ∈ C ∧ p ≠ q) →
    ∃ p q : ℝ × ℝ, p ∈ l ∧ q ∈ l ∧ p ∈ C ∧ q ∈ C ∧
    ∀ r s : ℝ × ℝ, r ∈ m ∧ s ∈ m ∧ r ∈ C ∧ s ∈ C →
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((r.1 - s.1)^2 + (r.2 - s.2)^2)) :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_line_l2383_238333


namespace NUMINAMATH_CALUDE_percent_problem_l2383_238391

theorem percent_problem (x : ℝ) : 0.01 = (10 / 100) * x → x = 0.1 := by sorry

end NUMINAMATH_CALUDE_percent_problem_l2383_238391


namespace NUMINAMATH_CALUDE_optimal_container_l2383_238394

-- Define the container parameters
def volume : ℝ := 8
def length : ℝ := 2
def min_height : ℝ := 3
def bottom_cost : ℝ := 40
def lateral_cost : ℝ := 20

-- Define the cost function
def cost (width height : ℝ) : ℝ :=
  bottom_cost * length * width + lateral_cost * (2 * (length + width) * height)

-- State the theorem
theorem optimal_container :
  ∃ (width height : ℝ),
    width > 0 ∧
    height ≥ min_height ∧
    length * width * height = volume ∧
    cost width height = 1520 / 3 ∧
    width = 4 / 3 ∧
    ∀ (w h : ℝ), w > 0 → h ≥ min_height → length * w * h = volume → cost w h ≥ 1520 / 3 := by
  sorry

end NUMINAMATH_CALUDE_optimal_container_l2383_238394


namespace NUMINAMATH_CALUDE_unique_valid_number_l2383_238346

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (n / 10 + n % 10 = 9) ∧
  (10 * (n % 10) + (n / 10) = n + 9)

theorem unique_valid_number : ∃! n : ℕ, is_valid_number n :=
  sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2383_238346


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l2383_238357

/-- A function from non-negative reals to non-negative reals. -/
def NonNegativeRealFunction := {f : ℝ → ℝ // ∀ x, 0 ≤ x → 0 ≤ f x}

/-- The functional equation f(f(x)) + f(x) = 6x for all x ≥ 0. -/
def FunctionalEquation (f : NonNegativeRealFunction) : Prop :=
  ∀ x : ℝ, 0 ≤ x → f.val (f.val x) + f.val x = 6 * x

theorem unique_function_satisfying_equation :
  ∀ f : NonNegativeRealFunction, FunctionalEquation f → 
    ∀ x : ℝ, 0 ≤ x → f.val x = 2 * x :=
sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l2383_238357


namespace NUMINAMATH_CALUDE_submarine_invention_uses_analogy_l2383_238328

/-- Represents the type of reasoning used in an invention process. -/
inductive ReasoningType
  | Analogy
  | Deduction
  | Induction

/-- Represents an invention process. -/
structure Invention where
  name : String
  inspiration : String
  reasoning : ReasoningType

/-- The submarine invention process. -/
def submarineInvention : Invention :=
  { name := "submarine",
    inspiration := "fish shape",
    reasoning := ReasoningType.Analogy }

/-- Theorem stating that the reasoning used in inventing submarines by imitating
    the shape of fish is analogy. -/
theorem submarine_invention_uses_analogy :
  submarineInvention.reasoning = ReasoningType.Analogy := by
  sorry

end NUMINAMATH_CALUDE_submarine_invention_uses_analogy_l2383_238328


namespace NUMINAMATH_CALUDE_new_people_in_country_l2383_238387

theorem new_people_in_country (born : ℕ) (immigrated : ℕ) : 
  born = 90171 → immigrated = 16320 → born + immigrated = 106491 :=
by
  sorry

end NUMINAMATH_CALUDE_new_people_in_country_l2383_238387


namespace NUMINAMATH_CALUDE_pentagon_reconstruction_l2383_238378

-- Define the pentagon and extended points
variable (A B C D E A' A'' B' C' D' E' : ℝ × ℝ)

-- Define the conditions of the construction
axiom midpoint_AB' : A' = 2 * B - A
axiom midpoint_A'A'' : A'' = 2 * B' - A'
axiom midpoint_BC' : C' = 2 * C - B
axiom midpoint_CD' : D' = 2 * D - C
axiom midpoint_DE' : E' = 2 * E - D
axiom midpoint_EA' : A' = 2 * A - E

-- State the theorem
theorem pentagon_reconstruction :
  A = (1/31 : ℝ) • A' + (2/31 : ℝ) • A'' + (4/31 : ℝ) • B' + 
      (8/31 : ℝ) • C' + (16/31 : ℝ) • D' + (0 : ℝ) • E' :=
sorry

end NUMINAMATH_CALUDE_pentagon_reconstruction_l2383_238378


namespace NUMINAMATH_CALUDE_circle_equation_correct_l2383_238325

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  center : PolarPoint
  radius : ℝ

/-- The equation of a circle in polar coordinates -/
def circleEquation (c : PolarCircle) (p : PolarPoint) : Prop :=
  p.ρ = 2 * Real.cos (p.θ - c.center.θ)

theorem circle_equation_correct (c : PolarCircle) :
  c.center.ρ = 1 ∧ c.center.θ = 1 ∧ c.radius = 1 →
  ∀ p : PolarPoint, circleEquation c p ↔
    (p.ρ * Real.cos p.θ - c.center.ρ * Real.cos c.center.θ)^2 +
    (p.ρ * Real.sin p.θ - c.center.ρ * Real.sin c.center.θ)^2 = c.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l2383_238325


namespace NUMINAMATH_CALUDE_tylers_age_l2383_238384

theorem tylers_age (tyler clay : ℕ) 
  (h1 : tyler = 3 * clay + 1) 
  (h2 : tyler + clay = 21) : 
  tyler = 16 := by
sorry

end NUMINAMATH_CALUDE_tylers_age_l2383_238384


namespace NUMINAMATH_CALUDE_skew_lines_properties_l2383_238319

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (inPlane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)
variable (planePlaneIntersection : Plane → Plane → Line)

-- Define the theorem
theorem skew_lines_properties
  (α β : Plane)
  (a b c : Line)
  (h1 : inPlane a α)
  (h2 : inPlane b β)
  (h3 : c = planePlaneIntersection α β)
  (h4 : skew a b) :
  (∃ (config : Prop), intersect c a ∧ intersect c b) ∧
  (∃ (lines : ℕ → Line), ∀ (i j : ℕ), i ≠ j → skew (lines i) (lines j)) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_properties_l2383_238319


namespace NUMINAMATH_CALUDE_triangle_problem_l2383_238305

theorem triangle_problem (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a = Real.sqrt 7 →
  b = 3 →
  Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3 →
  a / Real.sin A = b / Real.sin B →
  c / Real.sin C = a / Real.sin A →
  A + B + C = π →
  (A = π/3 ∧ Real.sin (2*B + π/6) = -1/7) := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l2383_238305


namespace NUMINAMATH_CALUDE_abc_inequality_l2383_238377

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c = 1/a + 1/b + 1/c) : a + b + c ≥ 3 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2383_238377


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l2383_238366

theorem hexagon_angle_measure (A B C D E F : ℝ) : 
  -- ABCDEF is a convex hexagon (sum of angles is 720°)
  A + B + C + D + E + F = 720 →
  -- Angles A, B, and C are congruent
  A = B ∧ B = C →
  -- Angles D, E, and F are congruent
  D = E ∧ E = F →
  -- Measure of angle A is 20 degrees less than measure of angle D
  A + 20 = D →
  -- Prove that the measure of angle D is 130 degrees
  D = 130 := by
sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l2383_238366


namespace NUMINAMATH_CALUDE_intersection_point_x_coord_l2383_238321

/-- Hyperbola C with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  eccentricity : ℝ
  left_vertex : ℝ × ℝ
  right_vertex : ℝ × ℝ

/-- Line intersecting the left branch of the hyperbola -/
structure IntersectingLine where
  point : ℝ × ℝ
  intersection1 : ℝ × ℝ
  intersection2 : ℝ × ℝ

/-- Point P is the intersection of lines MA₁ and NA₂ -/
def intersection_point (h : Hyperbola) (l : IntersectingLine) : ℝ × ℝ := sorry

/-- Main theorem: The x-coordinate of point P is always -1 -/
theorem intersection_point_x_coord (h : Hyperbola) (l : IntersectingLine) :
  h.center = (0, 0) →
  h.left_focus = (-2 * Real.sqrt 5, 0) →
  h.eccentricity = Real.sqrt 5 →
  h.left_vertex = (-2, 0) →
  h.right_vertex = (2, 0) →
  l.point = (-4, 0) →
  l.intersection1.1 < 0 ∧ l.intersection1.2 > 0 →  -- M is in the second quadrant
  (intersection_point h l).1 = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_point_x_coord_l2383_238321


namespace NUMINAMATH_CALUDE_james_carrot_sticks_l2383_238317

/-- Given that James originally had 50 carrot sticks, ate 22 before dinner,
    ate 15 after dinner, and gave away 8 during dinner, prove that he has 5 left. -/
theorem james_carrot_sticks (original : ℕ) (eaten_before : ℕ) (eaten_after : ℕ) (given_away : ℕ)
    (h1 : original = 50)
    (h2 : eaten_before = 22)
    (h3 : eaten_after = 15)
    (h4 : given_away = 8) :
    original - eaten_before - eaten_after - given_away = 5 := by
  sorry

end NUMINAMATH_CALUDE_james_carrot_sticks_l2383_238317


namespace NUMINAMATH_CALUDE_no_810_triple_l2383_238380

/-- Converts a list of digits in base 8 to a natural number -/
def fromBase8 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Converts a list of digits in base 10 to a natural number -/
def fromBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 10 * acc + d) 0

/-- Checks if a number is an 8-10 triple -/
def is810Triple (n : Nat) : Prop :=
  n > 0 ∧ ∃ digits : List Nat, 
    (∀ d ∈ digits, d < 8) ∧
    fromBase8 digits = n ∧
    fromBase10 digits = 3 * n

theorem no_810_triple : ¬∃ n : Nat, is810Triple n := by
  sorry

end NUMINAMATH_CALUDE_no_810_triple_l2383_238380


namespace NUMINAMATH_CALUDE_tangent_line_problem_l2383_238318

theorem tangent_line_problem (x y a : ℝ) :
  (∃ m : ℝ, y = 3 * x - 2 ∧ y = x^3 - 2 * a ∧ 3 * x^2 = 3) →
  (a = 0 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l2383_238318


namespace NUMINAMATH_CALUDE_problem_solution_l2383_238392

def f (x : ℝ) := |x + 1| + |x - 3|

theorem problem_solution :
  (∀ x : ℝ, f x < 6 ↔ -2 < x ∧ x < 4) ∧
  (∀ a : ℝ, (∃ x : ℝ, f x = |a - 2|) → (a ≥ 6 ∨ a ≤ -2)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2383_238392


namespace NUMINAMATH_CALUDE_needle_intersection_probability_l2383_238369

/-- Represents the experimental data for needle throwing --/
structure NeedleExperiment where
  trials : ℕ
  intersections : ℕ
  frequency : ℚ

/-- The set of experimental data --/
def experimentalData : List NeedleExperiment := [
  ⟨50, 23, 23/50⟩,
  ⟨100, 48, 12/25⟩,
  ⟨200, 83, 83/200⟩,
  ⟨500, 207, 207/500⟩,
  ⟨1000, 404, 101/250⟩,
  ⟨2000, 802, 401/1000⟩
]

/-- The distance between adjacent lines in cm --/
def lineDistance : ℚ := 5

/-- The length of the needle in cm --/
def needleLength : ℚ := 3

/-- The estimated probability of intersection --/
def estimatedProbability : ℚ := 2/5

/-- Theorem stating that the estimated probability approaches 0.4 as trials increase --/
theorem needle_intersection_probability :
  ∀ ε > 0, ∃ N : ℕ, ∀ e ∈ experimentalData,
    e.trials ≥ N → |e.frequency - estimatedProbability| < ε :=
sorry

end NUMINAMATH_CALUDE_needle_intersection_probability_l2383_238369


namespace NUMINAMATH_CALUDE_f_sum_eq_two_l2383_238386

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem f_sum_eq_two :
  let f' := deriv f
  f 2016 + f' 2016 + f (-2016) - f' (-2016) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_eq_two_l2383_238386


namespace NUMINAMATH_CALUDE_alchemists_less_than_half_l2383_238365

theorem alchemists_less_than_half (k : ℕ) (c a : ℕ) : 
  k > 0 → 
  k = c + a → 
  c > a → 
  a < k / 2 := by
sorry

end NUMINAMATH_CALUDE_alchemists_less_than_half_l2383_238365


namespace NUMINAMATH_CALUDE_equation_equivalence_implies_mnp_30_l2383_238316

theorem equation_equivalence_implies_mnp_30 
  (b x z c : ℝ) (m n p : ℤ) 
  (h : ∀ x z c, b^8*x*z - b^7*z - b^6*x = b^5*(c^5 - 1) ↔ (b^m*x-b^n)*(b^p*z-b^3)=b^5*c^5) : 
  m * n * p = 30 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_implies_mnp_30_l2383_238316


namespace NUMINAMATH_CALUDE_price_difference_l2383_238358

def original_price : ℝ := 1200

def price_after_increase (p : ℝ) : ℝ := p * 1.1

def price_after_decrease (p : ℝ) : ℝ := p * 0.85

def final_price : ℝ := price_after_decrease (price_after_increase original_price)

theorem price_difference : original_price - final_price = 78 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_l2383_238358


namespace NUMINAMATH_CALUDE_garden_tree_distance_l2383_238376

/-- Calculates the distance between consecutive trees in a garden. -/
def distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : ℚ :=
  if num_trees > 1 then
    (yard_length : ℚ) / ((num_trees - 1) : ℚ)
  else
    0

/-- Proves that the distance between consecutive trees is 28 meters. -/
theorem garden_tree_distance :
  distance_between_trees 700 26 = 28 := by
  sorry

end NUMINAMATH_CALUDE_garden_tree_distance_l2383_238376


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_ratio_l2383_238390

theorem right_triangle_acute_angle_ratio (α β : ℝ) : 
  α > 0 ∧ β > 0 ∧  -- Angles are positive
  α + β = 90 ∧     -- Sum of acute angles in a right triangle is 90°
  β = 5 * α →      -- One angle is 5 times the other
  β = 75 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_ratio_l2383_238390


namespace NUMINAMATH_CALUDE_triangle_properties_max_area_l2383_238304

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition √2 sin A = √3 cos A -/
def condition (t : Triangle) : Prop :=
  Real.sqrt 2 * Real.sin t.A = Real.sqrt 3 * Real.cos t.A

/-- The equation a² - c² = b² - mbc -/
def equation (t : Triangle) (m : ℝ) : Prop :=
  t.a^2 - t.c^2 = t.b^2 - m * t.b * t.c

theorem triangle_properties (t : Triangle) (m : ℝ) 
    (h1 : condition t) 
    (h2 : equation t m) : 
    m = 1 := by sorry

theorem max_area (t : Triangle) 
    (h1 : condition t) 
    (h2 : t.a = 2) : 
    (t.b * t.c * Real.sin t.A / 2) ≤ Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_max_area_l2383_238304
