import Mathlib

namespace NUMINAMATH_CALUDE_third_player_games_l735_73573

/-- Represents a table tennis game with three players -/
structure TableTennisGame where
  total_games : ℕ
  player1_games : ℕ
  player2_games : ℕ
  player3_games : ℕ

/-- The rules and conditions of the game -/
def valid_game (g : TableTennisGame) : Prop :=
  g.total_games = g.player1_games ∧
  g.total_games = g.player2_games + g.player3_games ∧
  g.player1_games = 21 ∧
  g.player2_games = 10

/-- Theorem stating that under the given conditions, the third player must have played 11 games -/
theorem third_player_games (g : TableTennisGame) (h : valid_game g) : 
  g.player3_games = 11 := by
  sorry

end NUMINAMATH_CALUDE_third_player_games_l735_73573


namespace NUMINAMATH_CALUDE_jellybean_probability_l735_73544

/-- The total number of jellybeans in the jar -/
def total_jellybeans : ℕ := 15

/-- The number of red jellybeans in the jar -/
def red_jellybeans : ℕ := 6

/-- The number of blue jellybeans in the jar -/
def blue_jellybeans : ℕ := 3

/-- The number of white jellybeans in the jar -/
def white_jellybeans : ℕ := 6

/-- The number of jellybeans picked -/
def picked_jellybeans : ℕ := 4

/-- The probability of picking at least 3 red jellybeans out of 4 -/
def prob_at_least_three_red : ℚ := 13 / 91

theorem jellybean_probability :
  let total_outcomes := Nat.choose total_jellybeans picked_jellybeans
  let favorable_outcomes := Nat.choose red_jellybeans 3 * Nat.choose (total_jellybeans - red_jellybeans) 1 +
                            Nat.choose red_jellybeans 4
  (favorable_outcomes : ℚ) / total_outcomes = prob_at_least_three_red :=
by sorry

end NUMINAMATH_CALUDE_jellybean_probability_l735_73544


namespace NUMINAMATH_CALUDE_total_points_is_265_l735_73574

/-- Given information about Paul's point assignment in the first quarter -/
structure PointAssignment where
  homework_points : ℕ
  quiz_points : ℕ
  test_points : ℕ
  hw_quiz_relation : quiz_points = homework_points + 5
  quiz_test_relation : test_points = 4 * quiz_points
  hw_given : homework_points = 40

/-- The total points assigned by Paul in the first quarter -/
def total_points (pa : PointAssignment) : ℕ :=
  pa.homework_points + pa.quiz_points + pa.test_points

/-- Theorem stating that the total points assigned is 265 -/
theorem total_points_is_265 (pa : PointAssignment) : total_points pa = 265 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_265_l735_73574


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l735_73586

/-- Given two vectors a and b in ℝ², prove that if a = (3,1) and b = (x,-1) are parallel, then x = -3 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![3, 1]
  let b : Fin 2 → ℝ := ![x, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  x = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l735_73586


namespace NUMINAMATH_CALUDE_percentage_girls_like_basketball_l735_73513

/-- Given a class with the following properties:
  * There are 25 students in total
  * 60% of students are girls
  * 40% of boys like playing basketball
  * The number of girls who like basketball is double the number of boys who don't like it
  Prove that 80% of girls like playing basketball -/
theorem percentage_girls_like_basketball :
  ∀ (total_students : ℕ) 
    (girls boys boys_like_basketball boys_dont_like_basketball girls_like_basketball : ℕ),
  total_students = 25 →
  girls = (60 : ℕ) * total_students / 100 →
  boys = total_students - girls →
  boys_like_basketball = (40 : ℕ) * boys / 100 →
  boys_dont_like_basketball = boys - boys_like_basketball →
  girls_like_basketball = 2 * boys_dont_like_basketball →
  (girls_like_basketball : ℚ) / girls * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_percentage_girls_like_basketball_l735_73513


namespace NUMINAMATH_CALUDE_parabola_focus_on_x_eq_one_l735_73512

/-- A parabola is a conic section with a focus and directrix. -/
structure Parabola where
  /-- The focus of the parabola -/
  focus : ℝ × ℝ

/-- The standard form of a parabola equation -/
def standard_form (p : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y^2 = 4 * (x - p.focus.1)

/-- Theorem: For a parabola with its focus on the line x = 1, its standard equation is y^2 = 4x -/
theorem parabola_focus_on_x_eq_one (p : Parabola) 
    (h : p.focus.1 = 1) : 
    ∀ x y, standard_form p x y ↔ y^2 = 4*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_on_x_eq_one_l735_73512


namespace NUMINAMATH_CALUDE_day_care_toddlers_l735_73587

/-- Given the initial ratio of toddlers to infants and the ratio after more infants join,
    prove the number of toddlers -/
theorem day_care_toddlers (t i : ℕ) (h1 : t * 3 = i * 7) (h2 : t * 5 = (i + 12) * 7) : t = 42 := by
  sorry

end NUMINAMATH_CALUDE_day_care_toddlers_l735_73587


namespace NUMINAMATH_CALUDE_jan_cindy_age_difference_l735_73582

def age_difference (cindy_age jan_age marcia_age greg_age : ℕ) : Prop :=
  (cindy_age = 5) ∧
  (jan_age > cindy_age) ∧
  (marcia_age = 2 * jan_age) ∧
  (greg_age = marcia_age + 2) ∧
  (greg_age = 16) ∧
  (jan_age - cindy_age = 2)

theorem jan_cindy_age_difference :
  ∃ (cindy_age jan_age marcia_age greg_age : ℕ),
    age_difference cindy_age jan_age marcia_age greg_age := by
  sorry

end NUMINAMATH_CALUDE_jan_cindy_age_difference_l735_73582


namespace NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_of_16777_l735_73569

def n : ℕ := 16777

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (m : ℕ) : ℕ := sorry

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_digits_greatest_prime_divisor_of_16777 :
  sum_of_digits (greatest_prime_divisor n) = 2 := by sorry

end NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_of_16777_l735_73569


namespace NUMINAMATH_CALUDE_weight_loss_days_l735_73575

/-- The number of days it takes to lose a given amount of weight, given daily calorie intake, burn rate, and calories needed to lose one pound. -/
def days_to_lose_weight (calories_eaten : ℕ) (calories_burned : ℕ) (calories_per_pound : ℕ) (pounds_to_lose : ℕ) : ℕ :=
  let daily_deficit := calories_burned - calories_eaten
  let days_per_pound := calories_per_pound / daily_deficit
  days_per_pound * pounds_to_lose

/-- Theorem stating that it takes 80 days to lose 10 pounds under given conditions -/
theorem weight_loss_days : days_to_lose_weight 1800 2300 4000 10 = 80 := by
  sorry

#eval days_to_lose_weight 1800 2300 4000 10

end NUMINAMATH_CALUDE_weight_loss_days_l735_73575


namespace NUMINAMATH_CALUDE_linear_and_quadratic_sequences_properties_l735_73515

def is_second_order_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, (a (n + 2) - a (n + 1)) - (a (n + 1) - a n) = d

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

def is_local_geometric (a : ℕ → ℝ) : Prop :=
  ¬is_geometric a ∧ ∃ i j k : ℕ, i < j ∧ j < k ∧ (a j)^2 = a i * a k

theorem linear_and_quadratic_sequences_properties :
  (is_second_order_arithmetic (fun n => n : ℕ → ℝ) ∧
   is_local_geometric (fun n => n : ℕ → ℝ)) ∧
  (is_second_order_arithmetic (fun n => n^2 : ℕ → ℝ) ∧
   is_local_geometric (fun n => n^2 : ℕ → ℝ)) := by sorry

end NUMINAMATH_CALUDE_linear_and_quadratic_sequences_properties_l735_73515


namespace NUMINAMATH_CALUDE_water_tank_capacity_l735_73548

/-- The capacity of a water tank given its filling rate and time to reach 3/4 capacity -/
theorem water_tank_capacity (fill_rate : ℝ) (time_to_three_quarters : ℝ) 
  (h1 : fill_rate = 10) 
  (h2 : time_to_three_quarters = 300) : 
  fill_rate * time_to_three_quarters / (3/4) = 4000 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l735_73548


namespace NUMINAMATH_CALUDE_banana_arrangements_l735_73526

/-- The number of distinct arrangements of letters in a word with repeated letters -/
def distinctArrangements (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The word "banana" has 6 letters total -/
def bananaLength : ℕ := 6

/-- The repeated letters in "banana" are [3, 2, 1] (for 'a', 'n', 'b' respectively) -/
def bananaRepeatedLetters : List ℕ := [3, 2, 1]

theorem banana_arrangements :
  distinctArrangements bananaLength bananaRepeatedLetters = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l735_73526


namespace NUMINAMATH_CALUDE_probability_theorem_l735_73543

/-- Represents the contents of the magician's box -/
structure Box :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)

/-- Calculates the probability of drawing all red chips before blue and green chips -/
def probability_all_red_first (b : Box) : ℚ :=
  sorry

/-- The magician's box -/
def magicians_box : Box :=
  { red := 4, green := 3, blue := 1 }

/-- Theorem stating the probability of drawing all red chips first -/
theorem probability_theorem :
  probability_all_red_first magicians_box = 5 / 6720 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l735_73543


namespace NUMINAMATH_CALUDE_selling_price_calculation_l735_73547

def cost_price : ℝ := 225
def profit_percentage : ℝ := 20

theorem selling_price_calculation :
  let profit := (profit_percentage / 100) * cost_price
  let selling_price := cost_price + profit
  selling_price = 270 := by
sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l735_73547


namespace NUMINAMATH_CALUDE_total_food_items_donated_l735_73591

/-- The total number of food items donated by five companies given specific donation rules -/
theorem total_food_items_donated (foster_chickens : ℕ) : foster_chickens = 45 →
  ∃ (american_water hormel_chickens boudin_chickens delmonte_water : ℕ),
    american_water = 2 * foster_chickens ∧
    hormel_chickens = 3 * foster_chickens ∧
    boudin_chickens = hormel_chickens / 3 ∧
    delmonte_water = american_water - 30 ∧
    (boudin_chickens + delmonte_water) % 7 = 0 ∧
    foster_chickens = (hormel_chickens + boudin_chickens) / 2 ∧
    foster_chickens + american_water + hormel_chickens + boudin_chickens + delmonte_water = 375 :=
by sorry

end NUMINAMATH_CALUDE_total_food_items_donated_l735_73591


namespace NUMINAMATH_CALUDE_potato_ratio_l735_73524

theorem potato_ratio (total : ℕ) (wedges : ℕ) (chip_wedge_diff : ℕ) :
  total = 67 →
  wedges = 13 →
  chip_wedge_diff = 436 →
  let remaining := total - wedges
  let fries := remaining / 2
  let chips := remaining / 2
  fries = chips := by sorry

end NUMINAMATH_CALUDE_potato_ratio_l735_73524


namespace NUMINAMATH_CALUDE_intersection_A_B_l735_73555

def U : Set Int := {-1, 3, 5, 7, 9}
def complement_A : Set Int := {-1, 9}
def B : Set Int := {3, 7, 9}

def A : Set Int := U \ complement_A

theorem intersection_A_B :
  A ∩ B = {3, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l735_73555


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l735_73557

theorem purely_imaginary_complex_number (a : ℝ) : 
  (2 : ℂ) + Complex.I * ((1 : ℂ) - a + a * Complex.I) = Complex.I * (Complex.I.im * ((1 : ℂ) - a + a * Complex.I)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l735_73557


namespace NUMINAMATH_CALUDE_pure_imaginary_modulus_l735_73553

theorem pure_imaginary_modulus (m : ℝ) : 
  let z : ℂ := Complex.mk (m^2 - 9) (m^2 + 2*m - 3)
  (Complex.re z = 0 ∧ Complex.im z ≠ 0) → Complex.abs z = 12 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_modulus_l735_73553


namespace NUMINAMATH_CALUDE_florist_roses_l735_73539

theorem florist_roses (initial : ℕ) (sold : ℕ) (picked : ℕ) : 
  initial = 37 → sold = 16 → picked = 19 → initial - sold + picked = 40 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l735_73539


namespace NUMINAMATH_CALUDE_tangent_line_equation_l735_73509

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the point P
def point_P : ℝ × ℝ := (3, 1)

-- Define a general circle
def general_circle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the tangent property
def is_tangent (circle1 circle2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y ∧
  ∀ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) → ¬(circle1 x' y' ∧ circle2 x' y')

-- State the theorem
theorem tangent_line_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    general_circle center radius point_P.1 point_P.2 ∧
    is_tangent (general_circle center radius) circle_C →
    ∃ (A B : ℝ × ℝ),
      circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
      ∀ (x y : ℝ), 2*x + y - 3 = 0 ↔ (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l735_73509


namespace NUMINAMATH_CALUDE_only_cylinder_produces_quadrilateral_section_l735_73585

-- Define the types of geometric solids
inductive GeometricSolid
  | Cone
  | Sphere
  | Cylinder

-- Define a function that checks if a geometric solid can produce a quadrilateral section
def can_produce_quadrilateral_section (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => True
  | _ => False

-- Theorem statement
theorem only_cylinder_produces_quadrilateral_section :
  ∀ (solid : GeometricSolid),
    can_produce_quadrilateral_section solid ↔ solid = GeometricSolid.Cylinder :=
by
  sorry


end NUMINAMATH_CALUDE_only_cylinder_produces_quadrilateral_section_l735_73585


namespace NUMINAMATH_CALUDE_smallest_number_l735_73552

def numbers : List ℤ := [0, -2, 1, 5]

theorem smallest_number (n : ℤ) (hn : n ∈ numbers) : -2 ≤ n := by
  sorry

#check smallest_number

end NUMINAMATH_CALUDE_smallest_number_l735_73552


namespace NUMINAMATH_CALUDE_existence_of_p_and_q_l735_73596

theorem existence_of_p_and_q : ∃ (p q : ℝ), 
  ((p - 1)^2 - 4*q > 0) ∧ 
  ((p + 1)^2 - 4*q > 0) ∧ 
  (p^2 - 4*q ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_existence_of_p_and_q_l735_73596


namespace NUMINAMATH_CALUDE_circle_properties_l735_73564

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 0)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem circle_properties :
  ∀ (x y : ℝ), circle_equation x y ↔ 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l735_73564


namespace NUMINAMATH_CALUDE_equation_solutions_l735_73551

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4) ∧
  (∀ x : ℝ, -2 * (x^3 - 1) = 18 ↔ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l735_73551


namespace NUMINAMATH_CALUDE_histogram_total_area_is_one_l735_73523

/-- A histogram representing a data distribution -/
structure Histogram where
  -- We don't need to define the internal structure of the histogram
  -- as we're only concerned with its total area property

/-- The total area of a histogram -/
def total_area (h : Histogram) : ℝ := sorry

/-- Theorem: The total area of a histogram representing a data distribution is equal to 1 -/
theorem histogram_total_area_is_one (h : Histogram) : total_area h = 1 := by
  sorry

end NUMINAMATH_CALUDE_histogram_total_area_is_one_l735_73523


namespace NUMINAMATH_CALUDE_probability_three_green_apples_l735_73570

theorem probability_three_green_apples (total_apples green_apples selected_apples : ℕ) :
  total_apples = 10 →
  green_apples = 4 →
  selected_apples = 3 →
  (Nat.choose green_apples selected_apples : ℚ) / (Nat.choose total_apples selected_apples) = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_green_apples_l735_73570


namespace NUMINAMATH_CALUDE_ruby_starting_lineup_combinations_l735_73561

def total_players : ℕ := 15
def all_stars : ℕ := 5
def starting_lineup : ℕ := 7

theorem ruby_starting_lineup_combinations :
  Nat.choose (total_players - all_stars) (starting_lineup - all_stars) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ruby_starting_lineup_combinations_l735_73561


namespace NUMINAMATH_CALUDE_total_days_1996_to_2000_l735_73532

/-- The number of days in a regular year -/
def regularYearDays : ℕ := 365

/-- The number of additional days in a leap year -/
def leapYearExtraDays : ℕ := 1

/-- The start year of our range -/
def startYear : ℕ := 1996

/-- The end year of our range -/
def endYear : ℕ := 2000

/-- The number of leap years in our range -/
def leapYearsCount : ℕ := 2

/-- Theorem: The total number of days from 1996 to 2000 (inclusive) is 1827 -/
theorem total_days_1996_to_2000 : 
  (endYear - startYear + 1) * regularYearDays + leapYearsCount * leapYearExtraDays = 1827 := by
  sorry

end NUMINAMATH_CALUDE_total_days_1996_to_2000_l735_73532


namespace NUMINAMATH_CALUDE_leon_payment_l735_73581

/-- The total amount Leon paid for toy organizers, gaming chairs, and delivery fee. -/
def total_paid (toy_organizer_sets : ℕ) (toy_organizer_price : ℚ) 
                (gaming_chairs : ℕ) (gaming_chair_price : ℚ) 
                (delivery_fee_percentage : ℚ) : ℚ :=
  let total_sales := toy_organizer_sets * toy_organizer_price + gaming_chairs * gaming_chair_price
  let delivery_fee := delivery_fee_percentage * total_sales
  total_sales + delivery_fee

/-- Theorem stating that Leon paid $420 in total -/
theorem leon_payment : 
  total_paid 3 78 2 83 (5/100) = 420 := by
  sorry

end NUMINAMATH_CALUDE_leon_payment_l735_73581


namespace NUMINAMATH_CALUDE_prob_black_then_red_standard_deck_l735_73540

/-- A deck of cards with black cards, red cards, and jokers. -/
structure Deck :=
  (total : ℕ)
  (black : ℕ)
  (red : ℕ)
  (jokers : ℕ)

/-- The probability of drawing a black card first and a red card second from a given deck. -/
def prob_black_then_red (d : Deck) : ℚ :=
  (d.black : ℚ) / d.total * (d.red : ℚ) / (d.total - 1)

/-- The standard deck with 54 cards including jokers. -/
def standard_deck : Deck :=
  { total := 54
  , black := 26
  , red := 26
  , jokers := 2 }

theorem prob_black_then_red_standard_deck :
  prob_black_then_red standard_deck = 338 / 1431 := by
  sorry

end NUMINAMATH_CALUDE_prob_black_then_red_standard_deck_l735_73540


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l735_73580

def old_cost : ℝ := 150
def new_cost : ℝ := 195

theorem percentage_increase_proof :
  (new_cost - old_cost) / old_cost * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l735_73580


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l735_73500

theorem shoe_price_calculation (num_shoes : ℕ) (num_shirts : ℕ) (shirt_price : ℚ) (total_earnings_per_person : ℚ) :
  num_shoes = 6 →
  num_shirts = 18 →
  shirt_price = 2 →
  total_earnings_per_person = 27 →
  ∃ (shoe_price : ℚ), 
    (num_shoes * shoe_price + num_shirts * shirt_price) / 2 = total_earnings_per_person ∧
    shoe_price = 3 :=
by sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l735_73500


namespace NUMINAMATH_CALUDE_odd_function_sum_property_l735_73503

def is_odd_function (v : ℝ → ℝ) : Prop := ∀ x, v (-x) = -v x

theorem odd_function_sum_property (v : ℝ → ℝ) (a b : ℝ) 
  (h : is_odd_function v) : 
  v (-a) + v (-b) + v b + v a = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_property_l735_73503


namespace NUMINAMATH_CALUDE_product_factors_count_l735_73546

/-- A natural number with exactly three factors is a perfect square of a prime. -/
def has_three_factors (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p^2

/-- The main theorem statement -/
theorem product_factors_count
  (a b c d : ℕ)
  (ha : has_three_factors a)
  (hb : has_three_factors b)
  (hc : has_three_factors c)
  (hd : has_three_factors d)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d)
  (hbc : b ≠ c) (hbd : b ≠ d)
  (hcd : c ≠ d) :
  (Nat.factors (a^3 * b^2 * c^4 * d^5)).length = 3465 :=
sorry

end NUMINAMATH_CALUDE_product_factors_count_l735_73546


namespace NUMINAMATH_CALUDE_anna_apples_total_l735_73565

def apples_eaten (tuesday wednesday thursday : ℕ) : ℕ :=
  tuesday + wednesday + thursday

theorem anna_apples_total :
  ∀ (tuesday : ℕ),
    tuesday = 4 →
    ∀ (wednesday thursday : ℕ),
      wednesday = 2 * tuesday →
      thursday = tuesday / 2 →
      apples_eaten tuesday wednesday thursday = 14 := by
sorry

end NUMINAMATH_CALUDE_anna_apples_total_l735_73565


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_l735_73560

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x * Real.sin x

theorem derivative_f_at_pi :
  deriv f π = -Real.sqrt π := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_l735_73560


namespace NUMINAMATH_CALUDE_movie_length_ratio_l735_73562

/-- The lengths of favorite movies for Joyce, Michael, Nikki, and Ryn. -/
structure MovieLengths where
  michael : ℝ
  joyce : ℝ
  nikki : ℝ
  ryn : ℝ

/-- The conditions of the movie length problem. -/
def movieConditions (m : MovieLengths) : Prop :=
  m.joyce = m.michael + 2 ∧
  m.ryn = (4/5) * m.nikki ∧
  m.nikki = 30 ∧
  m.michael + m.joyce + m.nikki + m.ryn = 76

/-- The theorem stating that under the given conditions, 
    the ratio of Nikki's movie length to Michael's is 3:1. -/
theorem movie_length_ratio (m : MovieLengths) 
  (h : movieConditions m) : m.nikki / m.michael = 3 := by
  sorry

end NUMINAMATH_CALUDE_movie_length_ratio_l735_73562


namespace NUMINAMATH_CALUDE_not_q_is_false_l735_73521

theorem not_q_is_false (p q : Prop) (hp : ¬p) (hq : q) : ¬(¬q) :=
by sorry

end NUMINAMATH_CALUDE_not_q_is_false_l735_73521


namespace NUMINAMATH_CALUDE_smallest_natural_numbers_satisfying_equation_l735_73520

theorem smallest_natural_numbers_satisfying_equation :
  ∃ (A B : ℕ+),
    (360 : ℝ) / ((A : ℝ) * (A : ℝ) * (A : ℝ) / (B : ℝ)) = 5 ∧
    ∀ (A' B' : ℕ+),
      (360 : ℝ) / ((A' : ℝ) * (A' : ℝ) * (A' : ℝ) / (B' : ℝ)) = 5 →
      (A ≤ A' ∧ B ≤ B') ∧
    A = 6 ∧
    B = 3 ∧
    A + B = 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_natural_numbers_satisfying_equation_l735_73520


namespace NUMINAMATH_CALUDE_water_left_proof_l735_73534

def water_problem (initial_amount mother_drink father_extra sister_drink : ℝ) : Prop :=
  let father_drink := mother_drink + father_extra
  let total_consumed := mother_drink + father_drink + sister_drink
  let water_left := initial_amount - total_consumed
  water_left = 0.3

theorem water_left_proof :
  water_problem 1 0.1 0.2 0.3 := by
  sorry

end NUMINAMATH_CALUDE_water_left_proof_l735_73534


namespace NUMINAMATH_CALUDE_salon_extra_cans_l735_73514

/-- Represents the daily operations of a hair salon --/
structure Salon where
  customers : ℕ
  cans_bought : ℕ
  cans_per_customer : ℕ

/-- Calculates the number of extra cans of hairspray bought by the salon each day --/
def extra_cans (s : Salon) : ℕ :=
  s.cans_bought - (s.customers * s.cans_per_customer)

/-- Theorem stating that the salon buys 5 extra cans of hairspray each day --/
theorem salon_extra_cans :
  ∀ (s : Salon), s.customers = 14 ∧ s.cans_bought = 33 ∧ s.cans_per_customer = 2 →
  extra_cans s = 5 := by
  sorry

end NUMINAMATH_CALUDE_salon_extra_cans_l735_73514


namespace NUMINAMATH_CALUDE_no_seven_edge_polyhedron_exists_polyhedron_with_n_edges_l735_73528

/-- A convex polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure ConvexPolyhedron where
  -- We don't need to specify the internal structure for this problem
  mk :: -- Constructor

/-- The number of edges in a convex polyhedron -/
def num_edges (p : ConvexPolyhedron) : ℕ := sorry

/-- Theorem stating that no convex polyhedron has exactly 7 edges -/
theorem no_seven_edge_polyhedron : ¬∃ (p : ConvexPolyhedron), num_edges p = 7 := by sorry

/-- Theorem stating that for all natural numbers n ≥ 6 and n ≠ 7, there exists a convex polyhedron with n edges -/
theorem exists_polyhedron_with_n_edges (n : ℕ) (h1 : n ≥ 6) (h2 : n ≠ 7) : 
  ∃ (p : ConvexPolyhedron), num_edges p = n := by sorry

end NUMINAMATH_CALUDE_no_seven_edge_polyhedron_exists_polyhedron_with_n_edges_l735_73528


namespace NUMINAMATH_CALUDE_unique_prime_203B21_l735_73506

/-- A function that generates a six-digit number of the form 203B21 given a single digit B -/
def generate_number (B : Nat) : Nat :=
  203000 + B * 100 + 21

/-- Predicate to check if a number is of the form 203B21 where B is a single digit -/
def is_valid_form (n : Nat) : Prop :=
  ∃ B : Nat, B < 10 ∧ n = generate_number B

theorem unique_prime_203B21 :
  ∃! n : Nat, is_valid_form n ∧ Nat.Prime n ∧ n = 203521 := by sorry

end NUMINAMATH_CALUDE_unique_prime_203B21_l735_73506


namespace NUMINAMATH_CALUDE_unique_arithmetic_progression_l735_73518

theorem unique_arithmetic_progression : ∃! (a b : ℝ),
  (a - 15 = b - a) ∧ (ab - b = b - a) ∧ (a - b = 5) ∧ (a = 10) ∧ (b = 5) := by
  sorry

end NUMINAMATH_CALUDE_unique_arithmetic_progression_l735_73518


namespace NUMINAMATH_CALUDE_quadratic_equation_transform_sum_l735_73508

theorem quadratic_equation_transform_sum (x r s : ℝ) : 
  (16 * x^2 - 64 * x - 144 = 0) →
  ((x + r)^2 = s) →
  (r + s = -7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_transform_sum_l735_73508


namespace NUMINAMATH_CALUDE_defective_components_probability_l735_73525

-- Define the probability function
def probability (p q r : ℕ) : ℚ :=
  let total_components := p + q
  let numerator := q * (Nat.descFactorial (r-1) (q-1)) * (Nat.descFactorial p (r-q)) +
                   p * (Nat.descFactorial (r-1) (p-1)) * (Nat.descFactorial q (r-p))
  let denominator := Nat.descFactorial total_components r
  ↑numerator / ↑denominator

-- State the theorem
theorem defective_components_probability (p q r : ℕ) 
  (h1 : q < p) (h2 : p < r) (h3 : r < p + q) :
  probability p q r = (↑q * Nat.descFactorial (r-1) (q-1) * Nat.descFactorial p (r-q) + 
                       ↑p * Nat.descFactorial (r-1) (p-1) * Nat.descFactorial q (r-p)) / 
                      Nat.descFactorial (p+q) r :=
by
  sorry


end NUMINAMATH_CALUDE_defective_components_probability_l735_73525


namespace NUMINAMATH_CALUDE_smallest_perimeter_of_rectangle_l735_73531

theorem smallest_perimeter_of_rectangle (a b : ℕ) : 
  a * b = 1000 → 
  2 * (a + b) ≥ 130 ∧ 
  ∃ (x y : ℕ), x * y = 1000 ∧ 2 * (x + y) = 130 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_of_rectangle_l735_73531


namespace NUMINAMATH_CALUDE_unique_number_with_properties_l735_73522

def has_two_prime_factors (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = p^a * q^b

def count_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem unique_number_with_properties :
  ∃! n : ℕ, n > 0 ∧ has_two_prime_factors n ∧ count_divisors n = 6 ∧ sum_of_divisors n = 28 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_properties_l735_73522


namespace NUMINAMATH_CALUDE_no_equal_divisors_for_squares_l735_73550

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def divisors_3k_plus_1 (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => d > 0 ∧ n % d = 0 ∧ d % 3 = 1)

def divisors_3k_plus_2 (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => d > 0 ∧ n % d = 0 ∧ d % 3 = 2)

theorem no_equal_divisors_for_squares :
  ∀ n : ℕ, is_square n → (divisors_3k_plus_1 n).card ≠ (divisors_3k_plus_2 n).card :=
by sorry

end NUMINAMATH_CALUDE_no_equal_divisors_for_squares_l735_73550


namespace NUMINAMATH_CALUDE_sector_area_l735_73530

/-- Given a circular sector with perimeter 6 and central angle 1 radian, its area is 2. -/
theorem sector_area (r : ℝ) (h1 : r + 2 * r = 6) (h2 : 1 = 1) : r * r / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l735_73530


namespace NUMINAMATH_CALUDE_blood_cells_in_first_sample_l735_73507

theorem blood_cells_in_first_sample
  (total_cells : ℕ)
  (second_sample_cells : ℕ)
  (h1 : total_cells = 7341)
  (h2 : second_sample_cells = 3120) :
  total_cells - second_sample_cells = 4221 := by
  sorry

end NUMINAMATH_CALUDE_blood_cells_in_first_sample_l735_73507


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l735_73536

theorem count_integers_satisfying_inequality : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, Real.sqrt (2 * n) ≤ Real.sqrt (5 * n - 8) ∧ 
               Real.sqrt (5 * n - 8) < Real.sqrt (3 * n + 7)) ∧
    S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l735_73536


namespace NUMINAMATH_CALUDE_max_area_rectangle_l735_73597

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

end NUMINAMATH_CALUDE_max_area_rectangle_l735_73597


namespace NUMINAMATH_CALUDE_equation_solution_set_l735_73595

theorem equation_solution_set : 
  {(x, y) : ℕ × ℕ | 3 * x^2 + 2 * 9^y = x * (4^(y + 1) - 1)} = 
  {(2, 1), (3, 1), (3, 2), (18, 2)} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_set_l735_73595


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l735_73502

theorem solution_set_implies_sum (a b : ℝ) : 
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l735_73502


namespace NUMINAMATH_CALUDE_complex_number_proof_l735_73517

theorem complex_number_proof (z : ℂ) :
  (z.re = Complex.im (-Real.sqrt 2 + 7 * Complex.I)) ∧
  (z.im = Complex.re (Real.sqrt 7 * Complex.I + 5 * Complex.I^2)) →
  z = 7 - 5 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_number_proof_l735_73517


namespace NUMINAMATH_CALUDE_neg_sufficient_but_not_necessary_l735_73556

-- Define the propositions
variable (p q : Prop)

-- Define the concept of sufficient but not necessary condition
def SufficientButNotNecessary (p q : Prop) : Prop :=
  (p → q) ∧ ¬(q → p)

-- State the theorem
theorem neg_sufficient_but_not_necessary (h : SufficientButNotNecessary p q) :
  SufficientButNotNecessary (¬q) (¬p) :=
sorry

end NUMINAMATH_CALUDE_neg_sufficient_but_not_necessary_l735_73556


namespace NUMINAMATH_CALUDE_gcd_divisibility_l735_73572

theorem gcd_divisibility (p q r s : ℕ+) 
  (h1 : Nat.gcd p.val q.val = 40)
  (h2 : Nat.gcd q.val r.val = 50)
  (h3 : Nat.gcd r.val s.val = 75)
  (h4 : 80 < Nat.gcd s.val p.val)
  (h5 : Nat.gcd s.val p.val < 120) :
  17 ∣ p.val :=
by sorry

end NUMINAMATH_CALUDE_gcd_divisibility_l735_73572


namespace NUMINAMATH_CALUDE_integer_solutions_of_quadratic_equation_l735_73593

theorem integer_solutions_of_quadratic_equation :
  ∀ x y : ℤ, x^2 = y^2 * (x + y^4 + 2*y^2) →
  (x = 0 ∧ y = 0) ∨ (x = 12 ∧ y = 2) ∨ (x = -8 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_quadratic_equation_l735_73593


namespace NUMINAMATH_CALUDE_unique_intersection_point_l735_73563

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point satisfies all four linear equations -/
def satisfiesAllEquations (p : Point2D) : Prop :=
  3 * p.x - 2 * p.y = 12 ∧
  2 * p.x + 5 * p.y = -1 ∧
  p.x + 4 * p.y = 8 ∧
  5 * p.x - 3 * p.y = 15

/-- Theorem stating that there exists exactly one point satisfying all equations -/
theorem unique_intersection_point :
  ∃! p : Point2D, satisfiesAllEquations p :=
sorry


end NUMINAMATH_CALUDE_unique_intersection_point_l735_73563


namespace NUMINAMATH_CALUDE_simplify_expression_l735_73571

theorem simplify_expression (x : ℝ) : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l735_73571


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l735_73510

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

-- State the theorem
theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 0, f a x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 0, f a x = 1) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l735_73510


namespace NUMINAMATH_CALUDE_tuesday_monday_ratio_l735_73566

/-- Represents the number of visitors to a library on different days of the week -/
structure LibraryVisitors where
  monday : ℕ
  tuesday : ℕ
  remainingDaysAverage : ℕ
  totalWeek : ℕ

/-- The ratio of Tuesday visitors to Monday visitors is 2:1 -/
theorem tuesday_monday_ratio (v : LibraryVisitors) 
  (h1 : v.monday = 50)
  (h2 : v.remainingDaysAverage = 20)
  (h3 : v.totalWeek = 250)
  (h4 : v.totalWeek = v.monday + v.tuesday + 5 * v.remainingDaysAverage) :
  v.tuesday / v.monday = 2 := by
  sorry

#check tuesday_monday_ratio

end NUMINAMATH_CALUDE_tuesday_monday_ratio_l735_73566


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l735_73583

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l735_73583


namespace NUMINAMATH_CALUDE_record_deal_profit_difference_l735_73505

/-- Calculates the difference in profit between two deals for selling records -/
theorem record_deal_profit_difference 
  (total_records : ℕ) 
  (sammy_price : ℚ) 
  (bryan_price_interested : ℚ) 
  (bryan_price_not_interested : ℚ) : 
  total_records = 200 →
  sammy_price = 4 →
  bryan_price_interested = 6 →
  bryan_price_not_interested = 1 →
  (total_records : ℚ) * sammy_price - 
  ((total_records / 2 : ℚ) * bryan_price_interested + 
   (total_records / 2 : ℚ) * bryan_price_not_interested) = 100 := by
  sorry

#check record_deal_profit_difference

end NUMINAMATH_CALUDE_record_deal_profit_difference_l735_73505


namespace NUMINAMATH_CALUDE_power_of_three_difference_l735_73579

theorem power_of_three_difference : 3^(1+2+3) - (3^1 + 3^2 + 3^3) = 690 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_difference_l735_73579


namespace NUMINAMATH_CALUDE_sam_and_joan_books_l735_73529

/-- Given that Sam has 110 books and Joan has 102 books, prove that they have 212 books together. -/
theorem sam_and_joan_books : 
  let sam_books : ℕ := 110
  let joan_books : ℕ := 102
  sam_books + joan_books = 212 :=
by sorry

end NUMINAMATH_CALUDE_sam_and_joan_books_l735_73529


namespace NUMINAMATH_CALUDE_eighth_group_frequency_l735_73599

theorem eighth_group_frequency 
  (total_sample : ℕ) 
  (num_groups : ℕ) 
  (freq_1 freq_2 freq_3 freq_4 : ℕ) 
  (sum_freq_5_to_7 : ℚ) :
  total_sample = 100 →
  num_groups = 8 →
  freq_1 = 15 →
  freq_2 = 17 →
  freq_3 = 11 →
  freq_4 = 13 →
  sum_freq_5_to_7 = 32 / 100 →
  (freq_1 + freq_2 + freq_3 + freq_4 + (sum_freq_5_to_7 * total_sample).num + 
    (total_sample - freq_1 - freq_2 - freq_3 - freq_4 - (sum_freq_5_to_7 * total_sample).num)) / total_sample = 1 →
  (total_sample - freq_1 - freq_2 - freq_3 - freq_4 - (sum_freq_5_to_7 * total_sample).num) / total_sample = 12 / 100 :=
by sorry

end NUMINAMATH_CALUDE_eighth_group_frequency_l735_73599


namespace NUMINAMATH_CALUDE_test_scores_theorem_l735_73584

/-- Represents the test scores for three students -/
structure TestScores where
  alisson : ℕ
  jose : ℕ
  meghan : ℕ

/-- Calculates the total score for the three students -/
def totalScore (scores : TestScores) : ℕ :=
  scores.alisson + scores.jose + scores.meghan

/-- Theorem stating the total score for the three students -/
theorem test_scores_theorem (scores : TestScores) : totalScore scores = 210 :=
  by
  have h1 : scores.jose = scores.alisson + 40 := sorry
  have h2 : scores.meghan = scores.jose - 20 := sorry
  have h3 : scores.jose = 100 - 10 := sorry
  sorry

#check test_scores_theorem

end NUMINAMATH_CALUDE_test_scores_theorem_l735_73584


namespace NUMINAMATH_CALUDE_prime_cube_minus_one_divisibility_l735_73519

theorem prime_cube_minus_one_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_ge_3 : p ≥ 3) :
  30 ∣ (p^3 - 1) ↔ p ≡ 1 [MOD 15] := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_minus_one_divisibility_l735_73519


namespace NUMINAMATH_CALUDE_monotonic_difference_increasing_decreasing_l735_73598

-- Define monotonic functions on ℝ
def Monotonic (f : ℝ → ℝ) : Prop := 
  ∀ x y, x ≤ y → f x ≤ f y ∨ f x ≥ f y

-- Define increasing function
def Increasing (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f x < f y

-- Define decreasing function
def Decreasing (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f x > f y

-- Theorem statement
theorem monotonic_difference_increasing_decreasing 
  (f g : ℝ → ℝ) 
  (hf : Monotonic f) (hg : Monotonic g) :
  (Increasing f ∧ Decreasing g → Increasing (fun x ↦ f x - g x)) ∧
  (Decreasing f ∧ Increasing g → Decreasing (fun x ↦ f x - g x)) := by
  sorry


end NUMINAMATH_CALUDE_monotonic_difference_increasing_decreasing_l735_73598


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l735_73537

theorem sum_of_three_numbers : 2.12 + 0.004 + 0.345 = 2.469 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l735_73537


namespace NUMINAMATH_CALUDE_car_tank_capacity_l735_73567

def distance_to_home : ℝ := 220
def fuel_efficiency : ℝ := 20
def additional_distance : ℝ := 100

theorem car_tank_capacity :
  let total_distance := distance_to_home + additional_distance
  let tank_capacity := total_distance / fuel_efficiency
  tank_capacity = 16 := by sorry

end NUMINAMATH_CALUDE_car_tank_capacity_l735_73567


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l735_73588

theorem complex_number_quadrant (i : ℂ) (h : i * i = -1) :
  let z : ℂ := 2 * i / (1 + i)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l735_73588


namespace NUMINAMATH_CALUDE_exists_valid_marking_configuration_l735_73541

/-- A type representing a cell in the grid -/
structure Cell :=
  (row : Fin 19)
  (col : Fin 19)

/-- A type representing a marking configuration of the grid -/
def MarkingConfiguration := Cell → Bool

/-- A function to count marked cells in a 10x10 square -/
def countMarkedCells (config : MarkingConfiguration) (topLeft : Cell) : Nat :=
  sorry

/-- A predicate to check if all 10x10 squares have different counts -/
def allSquaresDifferent (config : MarkingConfiguration) : Prop :=
  ∀ s1 s2 : Cell, s1 ≠ s2 → 
    countMarkedCells config s1 ≠ countMarkedCells config s2

/-- The main theorem stating the existence of a valid marking configuration -/
theorem exists_valid_marking_configuration : 
  ∃ (config : MarkingConfiguration), allSquaresDifferent config :=
sorry

end NUMINAMATH_CALUDE_exists_valid_marking_configuration_l735_73541


namespace NUMINAMATH_CALUDE_mans_swimming_speed_l735_73533

/-- The speed of the stream in km/h -/
def stream_speed : ℝ := 1.6666666666666667

/-- Proves that the man's swimming speed in still water is 5 km/h -/
theorem mans_swimming_speed (t : ℝ) (h : t > 0) : 
  let downstream_time := t
  let upstream_time := 2 * t
  let mans_speed : ℝ := stream_speed * 3
  upstream_time * (mans_speed - stream_speed) = downstream_time * (mans_speed + stream_speed) →
  mans_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_mans_swimming_speed_l735_73533


namespace NUMINAMATH_CALUDE_correct_snow_globes_count_l735_73516

/-- The number of snow globes in each box of Christmas decorations -/
def snow_globes_per_box : ℕ := 5

/-- The number of pieces of tinsel in each box -/
def tinsel_per_box : ℕ := 4

/-- The number of Christmas trees in each box -/
def trees_per_box : ℕ := 1

/-- The total number of boxes distributed -/
def total_boxes : ℕ := 12

/-- The total number of decorations handed out -/
def total_decorations : ℕ := 120

/-- Theorem stating that the number of snow globes per box is correct -/
theorem correct_snow_globes_count :
  snow_globes_per_box = (total_decorations - total_boxes * (tinsel_per_box + trees_per_box)) / total_boxes :=
by sorry

end NUMINAMATH_CALUDE_correct_snow_globes_count_l735_73516


namespace NUMINAMATH_CALUDE_fifth_to_third_grade_ratio_l735_73594

/-- Proves that the ratio of fifth-graders to third-graders is 1:2 given the conditions -/
theorem fifth_to_third_grade_ratio : 
  ∀ (third_graders fourth_graders fifth_graders : ℕ),
  third_graders = 20 →
  fourth_graders = 2 * third_graders →
  third_graders + fourth_graders + fifth_graders = 70 →
  fifth_graders.gcd third_graders * 2 = fifth_graders ∧ 
  fifth_graders.gcd third_graders * 1 = fifth_graders.gcd third_graders :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_to_third_grade_ratio_l735_73594


namespace NUMINAMATH_CALUDE_triangle_semiperimeter_from_side_and_excircle_radii_l735_73549

/-- Given a side 'a' of a triangle and the radii of the excircles opposite 
    the other two sides 'ρ_b' and 'ρ_c', the semiperimeter 's' of the 
    triangle is equal to a/2 + √((a/2)² + ρ_b * ρ_c). -/
theorem triangle_semiperimeter_from_side_and_excircle_radii 
  (a ρ_b ρ_c : ℝ) (ha : a > 0) (hb : ρ_b > 0) (hc : ρ_c > 0) :
  ∃ s : ℝ, s > 0 ∧ s = a / 2 + Real.sqrt ((a / 2)^2 + ρ_b * ρ_c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_semiperimeter_from_side_and_excircle_radii_l735_73549


namespace NUMINAMATH_CALUDE_total_work_experience_approx_l735_73558

def daysPerYear : ℝ := 365
def daysPerMonth : ℝ := 30.44
def daysPerWeek : ℝ := 7

def bartenderYears : ℝ := 9
def bartenderMonths : ℝ := 8

def managerYears : ℝ := 3
def managerMonths : ℝ := 6

def salesMonths : ℝ := 11

def coordinatorYears : ℝ := 2
def coordinatorMonths : ℝ := 5
def coordinatorWeeks : ℝ := 3

def totalWorkExperience : ℝ :=
  (bartenderYears * daysPerYear + bartenderMonths * daysPerMonth) +
  (managerYears * daysPerYear + managerMonths * daysPerMonth) +
  (salesMonths * daysPerMonth) +
  (coordinatorYears * daysPerYear + coordinatorMonths * daysPerMonth + coordinatorWeeks * daysPerWeek)

theorem total_work_experience_approx :
  ⌊totalWorkExperience⌋ = 6044 := by sorry

end NUMINAMATH_CALUDE_total_work_experience_approx_l735_73558


namespace NUMINAMATH_CALUDE_inequality_holds_infinitely_often_l735_73504

theorem inequality_holds_infinitely_often (a : ℕ → ℝ) 
  (h : ∀ n, a n > 0) : 
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ 1 + a n > a (n - 1) * (2 ^ (1 / n : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_infinitely_often_l735_73504


namespace NUMINAMATH_CALUDE_julie_school_year_earnings_l735_73592

/-- Julie's summer work details and school year work conditions -/
structure WorkDetails where
  summer_weeks : ℕ
  summer_hours_per_week : ℕ
  summer_earnings : ℕ
  school_weeks : ℕ
  school_hours_per_week : ℕ
  rate_increase : ℚ

/-- Calculate Julie's school year earnings based on her work details -/
def calculate_school_year_earnings (w : WorkDetails) : ℚ :=
  let summer_hourly_rate := w.summer_earnings / (w.summer_weeks * w.summer_hours_per_week)
  let school_hourly_rate := summer_hourly_rate * (1 + w.rate_increase)
  school_hourly_rate * w.school_weeks * w.school_hours_per_week

/-- Theorem stating that Julie's school year earnings are $3750 -/
theorem julie_school_year_earnings :
  let w : WorkDetails := {
    summer_weeks := 10,
    summer_hours_per_week := 40,
    summer_earnings := 4000,
    school_weeks := 30,
    school_hours_per_week := 10,
    rate_increase := 1/4
  }
  calculate_school_year_earnings w = 3750 := by sorry

end NUMINAMATH_CALUDE_julie_school_year_earnings_l735_73592


namespace NUMINAMATH_CALUDE_max_roses_theorem_l735_73589

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : Nat  -- Price in cents for an individual rose
  dozen : Nat       -- Price in cents for a dozen roses
  two_dozen : Nat   -- Price in cents for two dozen roses

/-- Calculates the maximum number of roses that can be purchased with a given budget -/
def max_roses_purchasable (pricing : RosePricing) (budget : Nat) : Nat :=
  sorry

/-- The theorem stating the maximum number of roses purchasable with the given pricing and budget -/
theorem max_roses_theorem (pricing : RosePricing) (budget : Nat) :
  pricing.individual = 630 →
  pricing.dozen = 3600 →
  pricing.two_dozen = 5000 →
  budget = 68000 →
  max_roses_purchasable pricing budget = 316 :=
sorry

end NUMINAMATH_CALUDE_max_roses_theorem_l735_73589


namespace NUMINAMATH_CALUDE_theft_loss_calculation_l735_73535

/-- Represents the percentage of profit taken by the shopkeeper -/
def profit_percentage : ℝ := 10

/-- Represents the overall loss percentage -/
def overall_loss_percentage : ℝ := 23

/-- Represents the percentage of goods lost during theft -/
def theft_loss_percentage : ℝ := 30

/-- Theorem stating the relationship between profit, overall loss, and theft loss -/
theorem theft_loss_calculation (cost : ℝ) (cost_positive : cost > 0) :
  let selling_price := cost * (1 + profit_percentage / 100)
  let actual_revenue := cost * (1 - overall_loss_percentage / 100)
  selling_price * (1 - theft_loss_percentage / 100) = actual_revenue :=
sorry

end NUMINAMATH_CALUDE_theft_loss_calculation_l735_73535


namespace NUMINAMATH_CALUDE_increase_by_percentage_l735_73527

/-- Theorem: Increasing 550 by 35% results in 742.5 -/
theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 550 → percentage = 35 → result = initial * (1 + percentage / 100) → result = 742.5 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l735_73527


namespace NUMINAMATH_CALUDE_tree_spacing_l735_73501

/-- Proves that the distance between consecutive trees is 18 meters
    given a yard of 414 meters with 24 equally spaced trees. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 414)
  (h2 : num_trees = 24)
  (h3 : num_trees ≥ 2) :
  yard_length / (num_trees - 1) = 18 := by
sorry

end NUMINAMATH_CALUDE_tree_spacing_l735_73501


namespace NUMINAMATH_CALUDE_songs_deleted_l735_73554

theorem songs_deleted (pictures : ℕ) (text_files : ℕ) (total_files : ℕ) (songs : ℕ) : 
  pictures = 2 → text_files = 7 → total_files = 17 → pictures + songs + text_files = total_files → songs = 8 := by
  sorry

end NUMINAMATH_CALUDE_songs_deleted_l735_73554


namespace NUMINAMATH_CALUDE_range_of_m_l735_73576

theorem range_of_m (x y z : ℝ) (h1 : 6 * x = 3 * y + 12) (h2 : 6 * x = 2 * z) 
  (h3 : y ≥ 0) (h4 : z ≤ 9) : 
  let m := 2 * x + y - 3 * z
  ∀ m', m = m' → -19 ≤ m' ∧ m' ≤ -14 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l735_73576


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_2019_l735_73578

theorem smallest_n_divisible_by_2019 : ∃ (n : ℕ), n = 2000 ∧ 
  (∀ (m : ℕ), m < n → ¬(2019 ∣ (m^2 + 20*m + 19))) ∧ 
  (2019 ∣ (n^2 + 20*n + 19)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_2019_l735_73578


namespace NUMINAMATH_CALUDE_parallel_vectors_x_equals_9_l735_73538

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, prove that if they are parallel, then x = 9 -/
theorem parallel_vectors_x_equals_9 (x : ℝ) :
  let a : ℝ × ℝ := (x, 3)
  let b : ℝ × ℝ := (3, 1)
  parallel a b → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_equals_9_l735_73538


namespace NUMINAMATH_CALUDE_math_problem_time_calculation_l735_73590

theorem math_problem_time_calculation 
  (num_problems : ℕ) 
  (time_per_problem : ℕ) 
  (checking_time : ℕ) : 
  num_problems = 7 → 
  time_per_problem = 4 → 
  checking_time = 3 → 
  num_problems * time_per_problem + checking_time = 31 :=
by sorry

end NUMINAMATH_CALUDE_math_problem_time_calculation_l735_73590


namespace NUMINAMATH_CALUDE_house_painting_theorem_l735_73542

/-- Represents the number of worker-hours required to paint a house -/
def totalWorkerHours : ℕ := 32

/-- Represents the number of people who started painting -/
def initialWorkers : ℕ := 6

/-- Represents the number of hours the initial workers painted -/
def initialHours : ℕ := 2

/-- Represents the total time available to paint the house -/
def totalTime : ℕ := 4

/-- Calculates the number of additional workers needed to complete the painting -/
def additionalWorkersNeeded : ℕ :=
  (totalWorkerHours - initialWorkers * initialHours) / (totalTime - initialHours) - initialWorkers

theorem house_painting_theorem :
  additionalWorkersNeeded = 4 := by
  sorry

#eval additionalWorkersNeeded

end NUMINAMATH_CALUDE_house_painting_theorem_l735_73542


namespace NUMINAMATH_CALUDE_question_mark_value_l735_73577

theorem question_mark_value : ∃ (x : ℕ), x * 240 = 347 * 480 ∧ x = 694 := by
  sorry

end NUMINAMATH_CALUDE_question_mark_value_l735_73577


namespace NUMINAMATH_CALUDE_quadruple_inequality_l735_73545

theorem quadruple_inequality (a p q r : ℕ) 
  (ha : a > 1) (hp : p > 1) (hq : q > 1) (hr : r > 1)
  (hdiv_p : p ∣ a * q * r + 1)
  (hdiv_q : q ∣ a * p * r + 1)
  (hdiv_r : r ∣ a * p * q + 1) :
  a ≥ (p * q * r - 1) / (p * q + q * r + r * p) :=
sorry

end NUMINAMATH_CALUDE_quadruple_inequality_l735_73545


namespace NUMINAMATH_CALUDE_archer_arrow_recovery_percentage_l735_73559

-- Define the given constants
def shots_per_day : ℕ := 200
def days_per_week : ℕ := 4
def arrow_cost : ℚ := 5.5
def team_payment_percentage : ℚ := 0.7
def archer_weekly_spend : ℚ := 1056

-- Define the theorem
theorem archer_arrow_recovery_percentage :
  let total_shots := shots_per_day * days_per_week
  let total_cost := archer_weekly_spend / (1 - team_payment_percentage)
  let arrows_bought := total_cost / arrow_cost
  let arrows_recovered := total_shots - arrows_bought
  arrows_recovered / total_shots = 1/5 := by
sorry

end NUMINAMATH_CALUDE_archer_arrow_recovery_percentage_l735_73559


namespace NUMINAMATH_CALUDE_teachers_daughter_age_l735_73511

theorem teachers_daughter_age 
  (P : ℤ → ℤ)  -- P is a function from integers to integers
  (a : ℕ+)     -- a is a positive natural number
  (p : ℕ)      -- p is a natural number
  (h_poly : ∀ x y : ℤ, (x - y) ∣ (P x - P y))  -- P is a polynomial with integer coefficients
  (h_pa : P a = a)    -- P(a) = a
  (h_p0 : P 0 = p)    -- P(0) = p
  (h_prime : Nat.Prime p)  -- p is prime
  (h_p_gt_a : p > a)  -- p > a
  : a = 1 :=
by sorry

end NUMINAMATH_CALUDE_teachers_daughter_age_l735_73511


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l735_73568

-- Define a geometric sequence
def is_geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

-- Theorem statement
theorem geometric_sequence_condition (a b c d : ℝ) :
  (is_geometric_sequence a b c d → a * d = b * c) ∧
  ∃ a b c d : ℝ, a * d = b * c ∧ ¬(is_geometric_sequence a b c d) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l735_73568
