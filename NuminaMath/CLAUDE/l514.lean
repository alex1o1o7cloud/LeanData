import Mathlib

namespace NUMINAMATH_CALUDE_minimum_occupied_seats_theorem_l514_51466

/-- Represents a row of seats -/
structure SeatRow where
  total_seats : ℕ
  occupied_seats : ℕ

/-- Checks if the next person must sit next to someone already seated -/
def next_person_sits_next (row : SeatRow) : Prop :=
  row.occupied_seats * 2 ≥ row.total_seats

/-- The theorem to be proved -/
theorem minimum_occupied_seats_theorem (row : SeatRow) 
  (h1 : row.total_seats = 180) 
  (h2 : row.occupied_seats = 90) :
  (∀ n : ℕ, n < 90 → ¬(next_person_sits_next ⟨180, n⟩)) ∧ 
  next_person_sits_next row :=
sorry

end NUMINAMATH_CALUDE_minimum_occupied_seats_theorem_l514_51466


namespace NUMINAMATH_CALUDE_square_overlap_areas_l514_51440

/-- Given a square with side length 3 cm cut along its diagonal, prove the areas of overlap in specific arrangements --/
theorem square_overlap_areas :
  let square_side : ℝ := 3
  let small_square_side : ℝ := 1
  let triangle_area : ℝ := square_side ^ 2 / 2
  let small_square_area : ℝ := small_square_side ^ 2
  
  -- Area of overlap when a 1 cm × 1 cm square is placed inside one of the resulting triangles
  let overlap_area_b : ℝ := small_square_area / 4
  
  -- Area of overlap when the two triangles are arranged to form a rectangle of 1 cm × 3 cm with an additional overlap
  let overlap_area_c : ℝ := triangle_area / 2
  
  (overlap_area_b = 0.25 ∧ overlap_area_c = 2.25) := by
  sorry


end NUMINAMATH_CALUDE_square_overlap_areas_l514_51440


namespace NUMINAMATH_CALUDE_min_queries_30_cards_min_queries_31_cards_min_queries_32_cards_min_queries_50_cards_circular_l514_51410

/-- Represents a card with either 1 or -1 written on it -/
inductive Card
| one : Card
| negOne : Card

/-- Represents a query that returns the product of any 3 cards -/
def Query := Card → Card → Card → Int

/-- Represents a set of cards -/
def CardSet := List Card

/-- Function to determine if a set of queries is sufficient to determine the product of all cards -/
def isSufficientQueries (cards : CardSet) (queries : List Query) : Prop :=
  sorry

/-- Theorem for 30 cards -/
theorem min_queries_30_cards (cards : CardSet) (h : cards.length = 30) :
  ∃ (queries : List Query), isSufficientQueries cards queries ∧ queries.length = 10 :=
sorry

/-- Theorem for 31 cards -/
theorem min_queries_31_cards (cards : CardSet) (h : cards.length = 31) :
  ∃ (queries : List Query), isSufficientQueries cards queries ∧ queries.length = 11 :=
sorry

/-- Theorem for 32 cards -/
theorem min_queries_32_cards (cards : CardSet) (h : cards.length = 32) :
  ∃ (queries : List Query), isSufficientQueries cards queries ∧ queries.length = 12 :=
sorry

/-- Represents a circular arrangement of cards -/
def CircularCardSet := List Card

/-- Function to determine if a set of queries is sufficient to determine the product of all cards in a circular arrangement -/
def isSufficientQueriesCircular (cards : CircularCardSet) (queries : List Query) : Prop :=
  sorry

/-- Theorem for 50 cards in a circle -/
theorem min_queries_50_cards_circular (cards : CircularCardSet) (h : cards.length = 50) :
  ∃ (queries : List Query), isSufficientQueriesCircular cards queries ∧ queries.length = 50 :=
sorry

end NUMINAMATH_CALUDE_min_queries_30_cards_min_queries_31_cards_min_queries_32_cards_min_queries_50_cards_circular_l514_51410


namespace NUMINAMATH_CALUDE_inequality_theorem_l514_51461

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hk₁ : x₁ * y₁ - z₁^2 > 0) (hk₂ : x₂ * y₂ - z₂^2 > 0) : 
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ∧
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔ 
    x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂ ∧ x₁ * y₁ - z₁^2 = x₂ * y₂ - z₂^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l514_51461


namespace NUMINAMATH_CALUDE_expression_evaluation_l514_51444

theorem expression_evaluation : 12 - 5 * 3^2 + 8 / 2 - 7 + 4^2 = -20 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l514_51444


namespace NUMINAMATH_CALUDE_exchange_probability_l514_51409

/-- Represents the colors of balls -/
inductive Color
  | Red | Green | Yellow | Violet | Black | Orange

/-- Represents a bag of balls -/
def Bag := List Color

/-- Initial configuration of Arjun's bag -/
def arjunInitialBag : Bag :=
  [Color.Red, Color.Red, Color.Green, Color.Yellow, Color.Violet]

/-- Initial configuration of Becca's bag -/
def beccaInitialBag : Bag :=
  [Color.Black, Color.Black, Color.Orange]

/-- Represents the exchange process -/
def exchange (bag1 bag2 : Bag) : Bag × Bag :=
  sorry

/-- Checks if a bag has exactly 3 different colors -/
def hasThreeColors (bag : Bag) : Bool :=
  sorry

/-- Calculates the probability of the final configuration -/
def finalProbability (arjunBag beccaBag : Bag) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem exchange_probability :
  finalProbability arjunInitialBag beccaInitialBag = 3/10 :=
sorry

end NUMINAMATH_CALUDE_exchange_probability_l514_51409


namespace NUMINAMATH_CALUDE_ellipse_point_properties_l514_51417

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the point P
structure Point (x₀ y₀ : ℝ) where
  inside_ellipse : 0 < x₀^2 / 2 + y₀^2
  inside_ellipse' : x₀^2 / 2 + y₀^2 < 1

-- Define the line passing through P
def line (x₀ y₀ x y : ℝ) : Prop := x₀ * x / 2 + y₀ * y = 1

-- Theorem statement
theorem ellipse_point_properties {x₀ y₀ : ℝ} (P : Point x₀ y₀) :
  -- 1. Range of |PF₁| + |PF₂|
  ∃ (PF₁ PF₂ : ℝ), 2 ≤ PF₁ + PF₂ ∧ PF₁ + PF₂ < 2 * Real.sqrt 2 ∧
  -- 2. No common points between the line and ellipse
  ∀ (x y : ℝ), line x₀ y₀ x y → ¬ ellipse x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_point_properties_l514_51417


namespace NUMINAMATH_CALUDE_intersection_line_hyperbola_l514_51420

theorem intersection_line_hyperbola (a : ℝ) :
  (∃ A B : ℝ × ℝ, 
    (A.2 = a * A.1 + 1 ∧ 3 * A.1^2 - A.2^2 = 1) ∧
    (B.2 = a * B.1 + 1 ∧ 3 * B.1^2 - B.2^2 = 1) ∧
    A ≠ B) →
  (∃ A B : ℝ × ℝ, 
    (A.2 = a * A.1 + 1 ∧ 3 * A.1^2 - A.2^2 = 1) ∧
    (B.2 = a * B.1 + 1 ∧ 3 * B.1^2 - B.2^2 = 1) ∧
    A ≠ B ∧
    A.1 * B.1 + A.2 * B.2 = 0) →
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_line_hyperbola_l514_51420


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l514_51453

theorem jason_pokemon_cards (initial_cards given_away_cards : ℕ) :
  initial_cards = 9 →
  given_away_cards = 4 →
  initial_cards - given_away_cards = 5 :=
by sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l514_51453


namespace NUMINAMATH_CALUDE_prob_hit_third_shot_prob_hit_at_least_once_l514_51419

-- Define the probability of hitting the target in one shot
def hit_probability : ℝ := 0.9

-- Define the number of shots
def num_shots : ℕ := 4

-- Theorem for the probability of hitting the target on the 3rd shot
theorem prob_hit_third_shot : 
  hit_probability = 0.9 := by sorry

-- Theorem for the probability of hitting the target at least once
theorem prob_hit_at_least_once : 
  1 - (1 - hit_probability) ^ num_shots = 1 - 0.1 ^ 4 := by sorry

end NUMINAMATH_CALUDE_prob_hit_third_shot_prob_hit_at_least_once_l514_51419


namespace NUMINAMATH_CALUDE_triangle_angle_property_l514_51467

theorem triangle_angle_property (α : Real) :
  (0 < α) ∧ (α < π) →  -- α is an interior angle of a triangle
  (1 / Real.sin α + 1 / Real.cos α = 2) →
  α = π + (1 / 2) * Real.arcsin ((1 - Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_property_l514_51467


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l514_51405

/-- Given a hyperbola with equation x²/144 - y²/81 = 1, prove that the slope of its asymptotes is 3/4 -/
theorem hyperbola_asymptote_slope :
  ∃ (m : ℚ), (∀ (x y : ℚ), x^2 / 144 - y^2 / 81 = 1 →
    (y = m * x ∨ y = -m * x) → m = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l514_51405


namespace NUMINAMATH_CALUDE_apple_running_rate_l514_51415

/-- Given Mac's and Apple's running rates, prove that Apple's rate is 3 miles per hour -/
theorem apple_running_rate (mac_rate apple_rate : ℝ) : 
  mac_rate = 4 →  -- Mac's running rate is 4 miles per hour
  (24 / mac_rate) * 60 + 120 = (24 / apple_rate) * 60 →  -- Mac runs 24 miles 120 minutes faster than Apple
  apple_rate = 3 :=  -- Apple's running rate is 3 miles per hour
by
  sorry


end NUMINAMATH_CALUDE_apple_running_rate_l514_51415


namespace NUMINAMATH_CALUDE_function_identity_l514_51458

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), f (f m ^ 2 + 2 * f n ^ 2) = m ^ 2 + 2 * n ^ 2) : 
  ∀ (n : ℕ+), f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l514_51458


namespace NUMINAMATH_CALUDE_minimum_bottles_l514_51446

def bottle_capacity : ℕ := 15
def minimum_volume : ℕ := 150

theorem minimum_bottles : 
  ∀ n : ℕ, (n * bottle_capacity ≥ minimum_volume ∧ 
  ∀ m : ℕ, m < n → m * bottle_capacity < minimum_volume) → n = 10 :=
by sorry

end NUMINAMATH_CALUDE_minimum_bottles_l514_51446


namespace NUMINAMATH_CALUDE_unique_number_l514_51422

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem unique_number : ∃! n : ℕ,
  is_two_digit n ∧
  Odd n ∧
  n % 9 = 0 ∧
  is_perfect_square (digit_product n) ∧
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_unique_number_l514_51422


namespace NUMINAMATH_CALUDE_num_al_sandwiches_l514_51484

/-- Represents the number of different types of bread available at the deli. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available at the deli. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available at the deli. -/
def num_cheeses : ℕ := 5

/-- Represents whether ham is available at the deli. -/
def ham_available : Prop := True

/-- Represents whether turkey is available at the deli. -/
def turkey_available : Prop := True

/-- Represents whether cheddar cheese is available at the deli. -/
def cheddar_available : Prop := True

/-- Represents whether rye bread is available at the deli. -/
def rye_available : Prop := True

/-- Represents the number of sandwiches with ham and cheddar cheese combination. -/
def ham_cheddar_combos : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread and turkey combination. -/
def rye_turkey_combos : ℕ := num_cheeses

/-- Theorem stating the number of different sandwiches Al could order. -/
theorem num_al_sandwiches : 
  num_breads * num_meats * num_cheeses - ham_cheddar_combos - rye_turkey_combos = 165 := by
  sorry

end NUMINAMATH_CALUDE_num_al_sandwiches_l514_51484


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_nonnegative_l514_51469

/-- A function that represents f(x) = x^2 + |x - a| + b -/
def f (a b x : ℝ) : ℝ := x^2 + |x - a| + b

/-- Theorem: If f(x) is decreasing on (-∞, 0], then a ≥ 0 -/
theorem f_decreasing_implies_a_nonnegative (a b : ℝ) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f a b x ≥ f a b y) → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_nonnegative_l514_51469


namespace NUMINAMATH_CALUDE_barn_painted_area_l514_51427

/-- Calculates the total area to be painted for a rectangular barn -/
def total_painted_area (width length height : ℝ) : ℝ :=
  2 * (width * height + length * height) + width * length

/-- Theorem stating the total area to be painted for the given barn dimensions -/
theorem barn_painted_area :
  total_painted_area 12 15 6 = 828 := by
  sorry

end NUMINAMATH_CALUDE_barn_painted_area_l514_51427


namespace NUMINAMATH_CALUDE_superhero_speed_in_miles_per_hour_l514_51402

-- Define the superhero's speed in kilometers per minute
def superhero_speed_km_per_min : ℝ := 1000

-- Define the conversion factor from kilometers to miles
def km_to_miles : ℝ := 0.6

-- Define the number of minutes in an hour
def minutes_per_hour : ℝ := 60

-- Theorem statement
theorem superhero_speed_in_miles_per_hour :
  superhero_speed_km_per_min * minutes_per_hour * km_to_miles = 36000 := by
  sorry

end NUMINAMATH_CALUDE_superhero_speed_in_miles_per_hour_l514_51402


namespace NUMINAMATH_CALUDE_twenty_seven_power_divided_by_nine_l514_51498

theorem twenty_seven_power_divided_by_nine (m : ℕ) :
  m = 27^1001 → m / 9 = 3^3001 := by
  sorry

end NUMINAMATH_CALUDE_twenty_seven_power_divided_by_nine_l514_51498


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l514_51460

/-- An isosceles triangle with side lengths 3 and 6 has a perimeter of 15 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 6 → b = 6 → c = 3 →
  (a = b ∨ a = c ∨ b = c) →  -- Isosceles condition
  a + b + c = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l514_51460


namespace NUMINAMATH_CALUDE_bookcase_weight_theorem_l514_51482

/-- Represents the weight of the bookcase and items -/
def BookcaseWeightProblem : Prop :=
  let bookcaseLimit : ℕ := 80
  let hardcoverCount : ℕ := 70
  let hardcoverWeight : ℚ := 1/2
  let textbookCount : ℕ := 30
  let textbookWeight : ℕ := 2
  let knickknackCount : ℕ := 3
  let knickknackWeight : ℕ := 6
  let totalWeight : ℚ := 
    hardcoverCount * hardcoverWeight + 
    textbookCount * textbookWeight + 
    knickknackCount * knickknackWeight
  totalWeight - bookcaseLimit = 33

theorem bookcase_weight_theorem : BookcaseWeightProblem := by
  sorry

end NUMINAMATH_CALUDE_bookcase_weight_theorem_l514_51482


namespace NUMINAMATH_CALUDE_reciprocal_problem_l514_51493

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 5) : 50 * (1 / x) = 80 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l514_51493


namespace NUMINAMATH_CALUDE_quadratic_function_value_l514_51488

/-- Given a quadratic function y = ax^2 + bx + 5 (a ≠ 0) with two points
    (x₁, 2002) and (x₂, 2002) on its graph, the value of the function
    at x = x₁ + x₂ is equal to 5. -/
theorem quadratic_function_value (a b x₁ x₂ : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + 5
  f x₁ = 2002 ∧ f x₂ = 2002 → f (x₁ + x₂) = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l514_51488


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l514_51470

theorem polynomial_division_remainder (p q : ℝ) : 
  (∀ x, (x^3 - 3*x^2 + 9*x - 7) = (x - p) * (ax^2 + bx + c) + (2*x + q) → p = 1 ∧ q = -2) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l514_51470


namespace NUMINAMATH_CALUDE_quadrilateral_sum_l514_51464

/-- A quadrilateral ABCD with specific side lengths and angles -/
structure Quadrilateral :=
  (BC : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (angleA : ℝ)
  (angleB : ℝ)
  (p : ℕ)
  (q : ℕ)
  (h_BC : BC = 10)
  (h_CD : CD = 15)
  (h_AD : AD = 12)
  (h_angleA : angleA = 60)
  (h_angleB : angleB = 120)
  (h_AB : p + Real.sqrt q = AD + BC)

/-- The sum of p and q in the quadrilateral ABCD is 17 -/
theorem quadrilateral_sum (ABCD : Quadrilateral) : ABCD.p + ABCD.q = 17 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_sum_l514_51464


namespace NUMINAMATH_CALUDE_min_value_S_l514_51479

/-- The minimum value of (x-a)^2 + (ln x - a)^2 is 1/2, where x > 0 and a is real. -/
theorem min_value_S (x a : ℝ) (hx : x > 0) : 
  ∃ (min : ℝ), min = (1/2 : ℝ) ∧ ∀ y > 0, (y - a)^2 + (Real.log y - a)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_S_l514_51479


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l514_51437

/-- Parabola defined by x = 4t^2 and y = 4t -/
structure Parabola where
  t : ℝ
  x : ℝ := 4 * t^2
  y : ℝ := 4 * t

/-- The focus of the parabola -/
def focus : ℝ × ℝ := sorry

/-- Point P on the parabola -/
def P (m : ℝ) : ℝ × ℝ := (3, m)

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_focus_distance (m : ℝ) :
  ∃ (para : Parabola), P m = (para.x, para.y) → distance (P m) focus = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l514_51437


namespace NUMINAMATH_CALUDE_roots_star_zero_l514_51477

-- Define the new operation ※
def star (a b : ℝ) : ℝ := a * b - a - b

-- Define the theorem
theorem roots_star_zero {x₁ x₂ : ℝ} (h : x₁^2 + x₁ - 1 = 0 ∧ x₂^2 + x₂ - 1 = 0) : 
  star x₁ x₂ = 0 := by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_roots_star_zero_l514_51477


namespace NUMINAMATH_CALUDE_tims_weekend_ride_distance_l514_51406

/-- Tim's weekly biking schedule and distance calculation -/
theorem tims_weekend_ride_distance 
  (work_distance : ℝ) 
  (work_days : ℕ) 
  (speed : ℝ) 
  (total_biking_hours : ℝ) 
  (h1 : work_distance = 20)
  (h2 : work_days = 5)
  (h3 : speed = 25)
  (h4 : total_biking_hours = 16) :
  let workday_distance := 2 * work_distance * work_days
  let workday_hours := workday_distance / speed
  let weekend_hours := total_biking_hours - workday_hours
  weekend_hours * speed = 200 := by
sorry


end NUMINAMATH_CALUDE_tims_weekend_ride_distance_l514_51406


namespace NUMINAMATH_CALUDE_journey_distance_l514_51430

/-- The total distance of a journey is the sum of miles driven and miles remaining. -/
theorem journey_distance (miles_driven miles_remaining : ℕ) 
  (h1 : miles_driven = 923)
  (h2 : miles_remaining = 277) :
  miles_driven + miles_remaining = 1200 := by
sorry

end NUMINAMATH_CALUDE_journey_distance_l514_51430


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l514_51463

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 4)
  (h_a4 : a 4 = 2) :
  a 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l514_51463


namespace NUMINAMATH_CALUDE_complex_product_theorem_l514_51490

theorem complex_product_theorem (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 3)
  (h3 : 3 * z₁ - 2 * z₂ = 2 - Complex.I) :
  z₁ * z₂ = -18/5 + 24/5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l514_51490


namespace NUMINAMATH_CALUDE_projection_of_sum_onto_a_l514_51431

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-2, 1)

theorem projection_of_sum_onto_a :
  let sum := (a.1 + b.1, a.2 + b.2)
  let dot_product := sum.1 * a.1 + sum.2 * a.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  dot_product / magnitude_a = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_projection_of_sum_onto_a_l514_51431


namespace NUMINAMATH_CALUDE_shopping_time_calculation_l514_51448

-- Define the total shopping trip time in minutes
def total_shopping_time : ℕ := 90

-- Define the waiting times
def wait_for_cart : ℕ := 3
def wait_for_employee : ℕ := 13
def wait_for_restock : ℕ := 14
def wait_in_line : ℕ := 18

-- Define the theorem
theorem shopping_time_calculation :
  total_shopping_time - (wait_for_cart + wait_for_employee + wait_for_restock + wait_in_line) = 42 := by
  sorry

end NUMINAMATH_CALUDE_shopping_time_calculation_l514_51448


namespace NUMINAMATH_CALUDE_magazine_revenue_calculation_l514_51435

/-- Calculates the revenue from magazine sales given the total sales, newspaper sales, prices, and total revenue -/
theorem magazine_revenue_calculation 
  (total_items : ℕ) 
  (newspaper_count : ℕ) 
  (newspaper_price : ℚ) 
  (magazine_price : ℚ) 
  (total_revenue : ℚ) 
  (h1 : total_items = 425)
  (h2 : newspaper_count = 275)
  (h3 : newspaper_price = 5/2)
  (h4 : magazine_price = 19/4)
  (h5 : total_revenue = 123025/100)
  (h6 : newspaper_count ≤ total_items) :
  (total_items - newspaper_count) * magazine_price = 54275/100 := by
  sorry

end NUMINAMATH_CALUDE_magazine_revenue_calculation_l514_51435


namespace NUMINAMATH_CALUDE_four_special_numbers_l514_51487

theorem four_special_numbers : ∃ (a b c d : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧  -- distinct
  (100 ≤ a ∧ a < 1000) ∧ (100 ≤ b ∧ b < 1000) ∧ 
  (100 ≤ c ∧ c < 1000) ∧ (100 ≤ d ∧ d < 1000) ∧  -- three-digit
  (a / 100 = b / 100 ∧ a / 100 = c / 100 ∧ a / 100 = d / 100) ∧  -- same first digit
  ((a + b + c + d) % a = 0) ∧ 
  ((a + b + c + d) % b = 0) ∧ 
  ((a + b + c + d) % c = 0) ∧ 
  ((a + b + c + d) % d = 0) :=  -- sum divisible by each
by
  -- The proof would go here
  sorry

#eval (108 + 135 + 180 + 117) % 108  -- Should evaluate to 0
#eval (108 + 135 + 180 + 117) % 135  -- Should evaluate to 0
#eval (108 + 135 + 180 + 117) % 180  -- Should evaluate to 0
#eval (108 + 135 + 180 + 117) % 117  -- Should evaluate to 0

end NUMINAMATH_CALUDE_four_special_numbers_l514_51487


namespace NUMINAMATH_CALUDE_f_properties_l514_51496

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - x) * Real.exp x - 1

theorem f_properties :
  ∃ (a : ℝ),
    (∀ x ≠ 0, f a x / x < 1) ∧
    (∀ x : ℝ, f 1 x ≤ 0) ∧
    (f 1 0 = 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l514_51496


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l514_51447

theorem triangle_angle_inequality (a : ℝ) : 
  (∃ (α β : ℝ), 0 < α ∧ 0 < β ∧ α + β < π ∧ 
    Real.cos (Real.sqrt α) + Real.cos (Real.sqrt β) > a + Real.cos (Real.sqrt (α * β))) 
  → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l514_51447


namespace NUMINAMATH_CALUDE_complex_imaginary_solution_l514_51471

theorem complex_imaginary_solution (z : ℂ) : 
  (∃ b : ℝ, z = b * I) → 
  (∃ c : ℝ, (z - 3)^2 + 12 * I = c * I) → 
  (z = 3 * I ∨ z = -3 * I) := by
sorry

end NUMINAMATH_CALUDE_complex_imaginary_solution_l514_51471


namespace NUMINAMATH_CALUDE_balanced_scale_l514_51454

/-- The weight of a children's book in kilograms. -/
def book_weight : ℝ := 1.1

/-- The weight of a doll in kilograms. -/
def doll_weight : ℝ := 0.3

/-- The weight of a toy car in kilograms. -/
def toy_car_weight : ℝ := 0.5

/-- The number of dolls on the scale. -/
def num_dolls : ℕ := 2

/-- The number of toy cars on the scale. -/
def num_toy_cars : ℕ := 1

theorem balanced_scale : 
  book_weight = num_dolls * doll_weight + num_toy_cars * toy_car_weight :=
by sorry

end NUMINAMATH_CALUDE_balanced_scale_l514_51454


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l514_51423

theorem absolute_value_inequality (y : ℝ) : 
  |((8 - 2*y) / 4)| < 3 ↔ -2 < y ∧ y < 10 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l514_51423


namespace NUMINAMATH_CALUDE_stamp_exhibition_problem_l514_51428

theorem stamp_exhibition_problem (x : ℕ) : 
  (∃ (s : ℕ), s = 3 * (s / x) + 24 ∧ s = 4 * (s / x) - 26) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_stamp_exhibition_problem_l514_51428


namespace NUMINAMATH_CALUDE_painted_cube_one_third_blue_iff_three_l514_51411

/-- Represents a cube with side length n, painted blue on all faces and cut into n^3 unit cubes -/
structure PaintedCube where
  n : ℕ

/-- The total number of faces of all unit cubes -/
def PaintedCube.totalFaces (c : PaintedCube) : ℕ := 6 * c.n^3

/-- The number of blue faces among all unit cubes -/
def PaintedCube.blueFaces (c : PaintedCube) : ℕ := 6 * c.n^2

/-- The condition that exactly one-third of the total faces are blue -/
def PaintedCube.oneThirdBlue (c : PaintedCube) : Prop :=
  3 * c.blueFaces = c.totalFaces

theorem painted_cube_one_third_blue_iff_three (c : PaintedCube) :
  c.oneThirdBlue ↔ c.n = 3 := by sorry

end NUMINAMATH_CALUDE_painted_cube_one_third_blue_iff_three_l514_51411


namespace NUMINAMATH_CALUDE_fraction_calculation_l514_51494

theorem fraction_calculation (N : ℝ) (h : 0.4 * N = 240) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l514_51494


namespace NUMINAMATH_CALUDE_soccer_team_goalies_l514_51445

theorem soccer_team_goalies :
  ∀ (goalies defenders midfielders strikers : ℕ),
    defenders = 10 →
    midfielders = 2 * defenders →
    strikers = 7 →
    goalies + defenders + midfielders + strikers = 40 →
    goalies = 3 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_goalies_l514_51445


namespace NUMINAMATH_CALUDE_bingo_prize_distribution_l514_51468

theorem bingo_prize_distribution (total_prize : ℝ) (first_winner_share : ℝ) (remaining_winners : ℕ) : 
  total_prize = 2400 →
  first_winner_share = total_prize / 3 →
  remaining_winners = 10 →
  (total_prize - first_winner_share) / remaining_winners = 160 := by
  sorry

end NUMINAMATH_CALUDE_bingo_prize_distribution_l514_51468


namespace NUMINAMATH_CALUDE_quadrilateral_property_l514_51441

-- Define the quadrilateral ABCD
variable (A B C D : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- Define the distance between two points
def distance (P Q : Point) : ℝ := sorry

-- State the theorem
theorem quadrilateral_property (h1 : angle A D C = 135)
  (h2 : angle A D B - angle A B D = 2 * angle D A B)
  (h3 : angle A D B - angle A B D = 4 * angle C B D)
  (h4 : distance B C = Real.sqrt 2 * distance C D) :
  distance A B = distance B C + distance A D := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_property_l514_51441


namespace NUMINAMATH_CALUDE_person_A_age_l514_51462

theorem person_A_age (current_age_A current_age_B past_age_A past_age_B years_ago : ℕ) : 
  current_age_A + current_age_B = 70 →
  current_age_A - years_ago = current_age_B →
  past_age_B = past_age_A / 2 →
  past_age_A = current_age_A →
  past_age_B = current_age_B - years_ago →
  current_age_A = 42 := by
  sorry

end NUMINAMATH_CALUDE_person_A_age_l514_51462


namespace NUMINAMATH_CALUDE_permutations_of_three_eq_six_l514_51472

/-- The number of permutations of 3 distinct elements -/
def permutations_of_three : ℕ := 3 * 2 * 1

/-- Theorem stating that the number of permutations of 3 distinct elements is 6 -/
theorem permutations_of_three_eq_six : permutations_of_three = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_three_eq_six_l514_51472


namespace NUMINAMATH_CALUDE_binomial_product_l514_51475

theorem binomial_product (x : ℝ) : (4 * x + 3) * (x - 6) = 4 * x^2 - 21 * x - 18 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l514_51475


namespace NUMINAMATH_CALUDE_factorization_of_3x_squared_minus_12_l514_51413

theorem factorization_of_3x_squared_minus_12 (x : ℝ) :
  3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_3x_squared_minus_12_l514_51413


namespace NUMINAMATH_CALUDE_symmetric_line_l514_51421

/-- Given a line l with equation x - y + 1 = 0, prove that its symmetric line l' 
    with respect to x = 2 has the equation x + y - 5 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (x - y + 1 = 0) → 
  (∃ x' y', x' + y' - 5 = 0 ∧ x' = 4 - x ∧ y' = y) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_l514_51421


namespace NUMINAMATH_CALUDE_prime_square_product_theorem_l514_51486

theorem prime_square_product_theorem :
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℕ),
    Prime x₁ ∧ Prime x₂ ∧ Prime x₃ ∧ Prime x₄ ∧
    Prime x₅ ∧ Prime x₆ ∧ Prime x₇ ∧ Prime x₈ →
    4 * (x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈) -
    (x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 + x₆^2 + x₇^2 + x₈^2) = 992 →
    x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧
    x₅ = 2 ∧ x₆ = 2 ∧ x₇ = 2 ∧ x₈ = 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_square_product_theorem_l514_51486


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l514_51416

theorem algebraic_expression_value (a b : ℝ) : 
  (a * 1^3 + b * 1 + 1 = 5) → (a * (-1)^3 + b * (-1) + 1 = -3) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l514_51416


namespace NUMINAMATH_CALUDE_sqrt_28_div_sqrt_7_l514_51442

theorem sqrt_28_div_sqrt_7 : Real.sqrt 28 / Real.sqrt 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_28_div_sqrt_7_l514_51442


namespace NUMINAMATH_CALUDE_jimmy_passing_points_l514_51414

/-- The minimum number of points required to pass to the next class -/
def min_points_to_pass : ℕ := 50

/-- The number of points earned per exam -/
def points_per_exam : ℕ := 20

/-- The number of exams taken -/
def num_exams : ℕ := 3

/-- The number of points lost for bad behavior -/
def points_lost_behavior : ℕ := 5

/-- The maximum number of additional points Jimmy can lose and still pass -/
def max_additional_points_to_lose : ℕ := 5

theorem jimmy_passing_points :
  max_additional_points_to_lose = 
    points_per_exam * num_exams - points_lost_behavior - min_points_to_pass := by
  sorry

end NUMINAMATH_CALUDE_jimmy_passing_points_l514_51414


namespace NUMINAMATH_CALUDE_basketball_teams_l514_51497

theorem basketball_teams (total : ℕ) (bad : ℕ) (rich : ℕ) (both : ℕ) : 
  total = 60 → 
  bad = (3 * total) / 5 →
  rich = (2 * total) / 3 →
  both ≤ bad :=
by sorry

end NUMINAMATH_CALUDE_basketball_teams_l514_51497


namespace NUMINAMATH_CALUDE_cylinder_surface_area_doubling_l514_51412

theorem cylinder_surface_area_doubling (r h : ℝ) : 
  r > 0 → h > 0 →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 300 →
  8 * Real.pi * r^2 + 4 * Real.pi * r * h = 900 →
  h = r →
  2 * Real.pi * r^2 + 2 * Real.pi * r * (2 * h) = 450 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_doubling_l514_51412


namespace NUMINAMATH_CALUDE_orange_count_after_changes_l514_51491

/-- The number of oranges in a bin after removing some and adding new ones. -/
def oranges_in_bin (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Theorem stating that starting with 50 oranges, removing 40, and adding 24 results in 34 oranges. -/
theorem orange_count_after_changes : oranges_in_bin 50 40 24 = 34 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_after_changes_l514_51491


namespace NUMINAMATH_CALUDE_coloring_exists_l514_51433

/-- A coloring of numbers from 1 to 2n -/
def Coloring (n : ℕ) := Fin (2*n) → Fin n

/-- Predicate to check if a coloring is valid -/
def ValidColoring (n : ℕ) (c : Coloring n) : Prop :=
  (∀ color : Fin n, ∃! (a b : Fin (2*n)), c a = color ∧ c b = color ∧ a ≠ b) ∧
  (∀ diff : Fin n, ∃! (a b : Fin (2*n)), c a = c b ∧ a ≠ b ∧ a.val - b.val = diff.val + 1)

/-- The sequence of n for which the coloring is possible -/
def ColoringSequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 3 * ColoringSequence n + 1

theorem coloring_exists (m : ℕ) : ∃ c : Coloring (ColoringSequence m), ValidColoring (ColoringSequence m) c := by
  sorry

end NUMINAMATH_CALUDE_coloring_exists_l514_51433


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l514_51459

def P : Set ℝ := {1, 2}
def Q : Set ℝ := {x | |x| < 2}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l514_51459


namespace NUMINAMATH_CALUDE_expression_zero_at_two_l514_51480

theorem expression_zero_at_two (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  x = 2 → (1 / (x - 1) + 3 / (1 - x^2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_zero_at_two_l514_51480


namespace NUMINAMATH_CALUDE_cos_105_degrees_l514_51400

theorem cos_105_degrees : 
  Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l514_51400


namespace NUMINAMATH_CALUDE_compare_x_y_z_l514_51425

open Real

theorem compare_x_y_z (x y z : ℝ) (hx : x = log π) (hy : y = log 2 / log 5) (hz : z = exp (-1/2)) :
  y < z ∧ z < x := by sorry

end NUMINAMATH_CALUDE_compare_x_y_z_l514_51425


namespace NUMINAMATH_CALUDE_max_value_2sin_l514_51478

theorem max_value_2sin (x : ℝ) : ∃ (M : ℝ), M = 2 ∧ ∀ y : ℝ, 2 * Real.sin y ≤ M := by
  sorry

end NUMINAMATH_CALUDE_max_value_2sin_l514_51478


namespace NUMINAMATH_CALUDE_project_completion_time_l514_51408

/-- The number of days person A takes to complete the project alone -/
def days_A : ℝ := 45

/-- The number of days person B takes to complete the project alone -/
def days_B : ℝ := 30

/-- The number of days person B works alone initially -/
def initial_days_B : ℝ := 22

/-- The total number of days to complete the project -/
def total_days : ℝ := 34

theorem project_completion_time :
  (total_days - initial_days_B) / days_A + initial_days_B / days_B = 1 := by sorry

end NUMINAMATH_CALUDE_project_completion_time_l514_51408


namespace NUMINAMATH_CALUDE_min_value_theorem_l514_51451

theorem min_value_theorem (a b k m n : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (∀ x, a^(x-1) + 1 = b → x = k) → 
  m > 0 → 
  n > 0 → 
  m + n = b - k → 
  ∀ m' n', m' > 0 → n' > 0 → m' + n' = b - k → 
    9/m + 1/n ≤ 9/m' + 1/n' :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l514_51451


namespace NUMINAMATH_CALUDE_unbounded_expression_l514_51499

theorem unbounded_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ∀ M : ℝ, ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ (x*y + 1)^2 + (x - y)^2 > M :=
sorry

end NUMINAMATH_CALUDE_unbounded_expression_l514_51499


namespace NUMINAMATH_CALUDE_poncelet_theorem_l514_51403

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a triangle type
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define an incircle
def incircle (t : Triangle) : Circle := sorry

-- Function to check if a point lies on a circle
def lies_on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Theorem statement
theorem poncelet_theorem 
  (ABC DEF : Triangle) 
  (common_incircle : incircle ABC = incircle DEF)
  (c : Circle)
  (A_on_c : lies_on_circle ABC.A c)
  (B_on_c : lies_on_circle ABC.B c)
  (C_on_c : lies_on_circle ABC.C c)
  (D_on_c : lies_on_circle DEF.A c)
  (E_on_c : lies_on_circle DEF.B c) :
  lies_on_circle DEF.C c := by
  sorry


end NUMINAMATH_CALUDE_poncelet_theorem_l514_51403


namespace NUMINAMATH_CALUDE_different_color_pairs_count_l514_51443

/- Given a drawer with distinguishable socks: -/
def white_socks : ℕ := 6
def brown_socks : ℕ := 5
def blue_socks : ℕ := 4

/- Define the function to calculate the number of ways to choose two socks of different colors -/
def different_color_pairs : ℕ :=
  white_socks * brown_socks +
  brown_socks * blue_socks +
  white_socks * blue_socks

/- The theorem to prove -/
theorem different_color_pairs_count : different_color_pairs = 74 := by
  sorry

end NUMINAMATH_CALUDE_different_color_pairs_count_l514_51443


namespace NUMINAMATH_CALUDE_complex_number_equality_l514_51432

theorem complex_number_equality (z : ℂ) : (1 - Complex.I) * z = Complex.abs (1 + Complex.I * Real.sqrt 3) → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l514_51432


namespace NUMINAMATH_CALUDE_arc_length_unit_circle_30_degrees_l514_51424

theorem arc_length_unit_circle_30_degrees :
  let r : ℝ := 1  -- radius of unit circle
  let θ : ℝ := 30 -- central angle in degrees
  let l : ℝ := θ * π * r / 180 -- arc length formula
  l = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_unit_circle_30_degrees_l514_51424


namespace NUMINAMATH_CALUDE_gas_price_and_travel_distance_l514_51436

/-- Represents the problem of calculating gas prices and travel distance --/
theorem gas_price_and_travel_distance 
  (initial_gallons : ℝ) 
  (actual_gallons : ℝ) 
  (price_increase : ℝ) 
  (fuel_efficiency : ℝ) 
  (h1 : initial_gallons = 12)
  (h2 : actual_gallons = 10)
  (h3 : price_increase = 0.3)
  (h4 : fuel_efficiency = 25) :
  ∃ (original_price : ℝ) (new_distance : ℝ),
    initial_gallons * original_price = actual_gallons * (original_price + price_increase) ∧
    original_price = 1.5 ∧
    new_distance = actual_gallons * fuel_efficiency ∧
    new_distance = 250 := by
  sorry


end NUMINAMATH_CALUDE_gas_price_and_travel_distance_l514_51436


namespace NUMINAMATH_CALUDE_sqrt_six_diamond_sqrt_six_l514_51481

-- Define the operation ¤
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- Theorem statement
theorem sqrt_six_diamond_sqrt_six : diamond (Real.sqrt 6) (Real.sqrt 6) = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_diamond_sqrt_six_l514_51481


namespace NUMINAMATH_CALUDE_train_speed_l514_51439

/-- The speed of a train given its length and time to cross a point -/
theorem train_speed (length time : ℝ) (h1 : length = 400) (h2 : time = 20) :
  length / time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l514_51439


namespace NUMINAMATH_CALUDE_stu_has_four_books_l514_51404

/-- Given the number of books for Elmo, Laura, and Stu, we define their relationships --/
def book_relation (elmo laura stu : ℕ) : Prop :=
  elmo = 3 * laura ∧ laura = 2 * stu ∧ elmo = 24

/-- Theorem stating that if the book relation holds, then Stu has 4 books --/
theorem stu_has_four_books (elmo laura stu : ℕ) :
  book_relation elmo laura stu → stu = 4 := by
  sorry

end NUMINAMATH_CALUDE_stu_has_four_books_l514_51404


namespace NUMINAMATH_CALUDE_deduction_from_second_number_l514_51418

theorem deduction_from_second_number 
  (n : ℕ) 
  (avg_initial : ℚ)
  (avg_final : ℚ)
  (deduct_first : ℚ)
  (deduct_third : ℚ)
  (deduct_fourth_to_ninth : List ℚ)
  (h1 : n = 10)
  (h2 : avg_initial = 16)
  (h3 : avg_final = 11.5)
  (h4 : deduct_first = 9)
  (h5 : deduct_third = 7)
  (h6 : deduct_fourth_to_ninth = [6, 5, 4, 3, 2, 1]) :
  ∃ (deduct_second : ℚ), deduct_second = 8 ∧
    (n * avg_final = n * avg_initial - 
      (deduct_first + deduct_second + deduct_third + 
       deduct_fourth_to_ninth.sum)) :=
by sorry

end NUMINAMATH_CALUDE_deduction_from_second_number_l514_51418


namespace NUMINAMATH_CALUDE_dogs_in_park_l514_51483

/-- The number of dogs in the park -/
def D : ℕ := 88

/-- The number of dogs running -/
def running : ℕ := 12

/-- The number of dogs doing nothing -/
def doing_nothing : ℕ := 10

theorem dogs_in_park :
  D = running + D / 2 + D / 4 + doing_nothing :=
sorry


end NUMINAMATH_CALUDE_dogs_in_park_l514_51483


namespace NUMINAMATH_CALUDE_sixth_power_of_sqrt_two_plus_sqrt_two_l514_51407

theorem sixth_power_of_sqrt_two_plus_sqrt_two :
  (Real.sqrt (2 + Real.sqrt 2)) ^ 6 = 16 + 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_of_sqrt_two_plus_sqrt_two_l514_51407


namespace NUMINAMATH_CALUDE_expected_rolls_in_year_l514_51452

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieOutcome
  | Composite
  | Prime
  | RollAgain

/-- The probability distribution of the die outcomes -/
def dieProb : DieOutcome → ℚ
  | DieOutcome.Composite => 3/8
  | DieOutcome.Prime => 1/2
  | DieOutcome.RollAgain => 1/8

/-- The expected number of rolls on a single day -/
def expectedRollsPerDay : ℚ := 1

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- The expected number of rolls in a non-leap year -/
def expectedRollsInYear : ℚ := expectedRollsPerDay * daysInYear

theorem expected_rolls_in_year :
  expectedRollsInYear = 365 := by sorry

end NUMINAMATH_CALUDE_expected_rolls_in_year_l514_51452


namespace NUMINAMATH_CALUDE_amusement_park_elementary_students_l514_51474

theorem amusement_park_elementary_students 
  (total_women : ℕ) 
  (women_elementary : ℕ) 
  (more_men : ℕ) 
  (men_not_elementary : ℕ) 
  (h1 : total_women = 1518)
  (h2 : women_elementary = 536)
  (h3 : more_men = 525)
  (h4 : men_not_elementary = 1257) :
  women_elementary + (total_women + more_men - men_not_elementary) = 1322 :=
by
  sorry

end NUMINAMATH_CALUDE_amusement_park_elementary_students_l514_51474


namespace NUMINAMATH_CALUDE_intersection_chord_length_l514_51438

-- Define the polar equations
def line_polar (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - 2 * Real.pi / 3) = -Real.sqrt 3

def circle_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ + 2 * Real.sin θ

-- Define the Cartesian equations
def line_cartesian (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 2 * Real.sqrt 3

def circle_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Theorem statement
theorem intersection_chord_length :
  ∀ A B : ℝ × ℝ,
  (∃ θ_A ρ_A, line_polar ρ_A θ_A ∧ circle_polar ρ_A θ_A ∧ A = (ρ_A * Real.cos θ_A, ρ_A * Real.sin θ_A)) →
  (∃ θ_B ρ_B, line_polar ρ_B θ_B ∧ circle_polar ρ_B θ_B ∧ B = (ρ_B * Real.cos θ_B, ρ_B * Real.sin θ_B)) →
  A ≠ B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l514_51438


namespace NUMINAMATH_CALUDE_boat_purchase_l514_51455

theorem boat_purchase (a b c d : ℝ) 
  (h1 : a + b + c + d = 60)
  (h2 : a = (1/2) * (b + c + d))
  (h3 : b = (1/3) * (a + c + d))
  (h4 : c = (1/4) * (a + b + d))
  (h5 : a ≥ 0) (h6 : b ≥ 0) (h7 : c ≥ 0) (h8 : d ≥ 0) : d = 13 := by
  sorry

end NUMINAMATH_CALUDE_boat_purchase_l514_51455


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l514_51450

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (last_sampled : ℕ) : List ℕ :=
  sorry

/-- Theorem for systematic sampling results -/
theorem systematic_sampling_result 
  (total : ℕ) 
  (sample_size : ℕ) 
  (last_sampled : ℕ) 
  (h1 : total = 8000) 
  (h2 : sample_size = 50) 
  (h3 : last_sampled = 7894) :
  let segment_size := total / sample_size
  let last_segment_start := total - segment_size
  let samples := systematic_sample total sample_size last_sampled
  (last_segment_start = 7840 ∧ 
   samples.take 5 = [54, 214, 374, 534, 694]) :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l514_51450


namespace NUMINAMATH_CALUDE_triangle_abc_cosine_sine_l514_51495

theorem triangle_abc_cosine_sine (A B C : ℝ) (cosC_half : ℝ) (BC AC : ℝ) :
  cosC_half = Real.sqrt 5 / 5 →
  BC = 1 →
  AC = 5 →
  (Real.cos C = -3/5 ∧ Real.sin A = Real.sqrt 2 / 10) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_cosine_sine_l514_51495


namespace NUMINAMATH_CALUDE_polynomial_factorization_l514_51492

theorem polynomial_factorization (x : ℝ) :
  x^4 - 5*x^2 + 4 = (x + 1)*(x - 1)*(x + 2)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l514_51492


namespace NUMINAMATH_CALUDE_cafeteria_students_l514_51476

theorem cafeteria_students (total : ℕ) (no_lunch : ℕ) (cafeteria : ℕ) : 
  total = 60 → 
  no_lunch = 20 → 
  total = cafeteria + 3 * cafeteria + no_lunch → 
  cafeteria = 10 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_students_l514_51476


namespace NUMINAMATH_CALUDE_courtyard_width_is_14_l514_51485

/-- Represents the dimensions of a paving stone -/
structure PavingStone where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular shape -/
def area (length width : ℝ) : ℝ := length * width

/-- Theorem: The width of the courtyard is 14 meters -/
theorem courtyard_width_is_14 (stone : PavingStone) (yard : Courtyard) 
    (h1 : stone.length = 3)
    (h2 : stone.width = 2)
    (h3 : yard.length = 60)
    (h4 : area yard.length yard.width = 140 * area stone.length stone.width) :
  yard.width = 14 := by
  sorry

#check courtyard_width_is_14

end NUMINAMATH_CALUDE_courtyard_width_is_14_l514_51485


namespace NUMINAMATH_CALUDE_constant_width_interior_angle_ge_120_l514_51401

/-- A curve of constant width. -/
class ConstantWidthCurve (α : Type*) [MetricSpace α] where
  width : ℝ
  is_constant_width : ∀ (x y : α), dist x y ≤ width

/-- The interior angle at a point on a curve. -/
def interior_angle {α : Type*} [MetricSpace α] (c : ConstantWidthCurve α) (p : α) : ℝ := sorry

/-- Theorem: The interior angle at any corner point of a curve of constant width is at least 120 degrees. -/
theorem constant_width_interior_angle_ge_120 
  {α : Type*} [MetricSpace α] (c : ConstantWidthCurve α) (p : α) :
  interior_angle c p ≥ 120 := by sorry

end NUMINAMATH_CALUDE_constant_width_interior_angle_ge_120_l514_51401


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l514_51489

theorem divisibility_of_expression (q : ℕ) (h1 : q.Prime) (h2 : q % 2 = 1) :
  ∃ k : ℤ, (q + 1 : ℤ)^(q - 1) - 1 = k * q :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l514_51489


namespace NUMINAMATH_CALUDE_fraction_equality_l514_51429

theorem fraction_equality (x : ℚ) : (5 + x) / (8 + x) = (2 + x) / (3 + x) ↔ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l514_51429


namespace NUMINAMATH_CALUDE_dot_product_equation_is_line_l514_51434

/-- Represents a 2D vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot (v w : Vec2D) : ℝ := v.x * w.x + v.y * w.y

/-- Theorem stating that the equation r ⋅ a = m represents a line -/
theorem dot_product_equation_is_line (a : Vec2D) (m : ℝ) :
  ∃ (A B C : ℝ), ∀ (r : Vec2D), dot r a = m ↔ A * r.x + B * r.y + C = 0 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_equation_is_line_l514_51434


namespace NUMINAMATH_CALUDE_f_inequality_l514_51456

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem f_inequality (x : ℝ) : f x > f (2*x - 1) ↔ 1/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l514_51456


namespace NUMINAMATH_CALUDE_range_of_sum_and_abs_l514_51473

theorem range_of_sum_and_abs (a b : ℝ) 
  (ha : -1 ≤ a ∧ a ≤ 3) 
  (hb : -5 < b ∧ b < 3) : 
  ∀ x, x ∈ Set.Icc (-1 : ℝ) 8 ↔ ∃ (a' b' : ℝ), 
    -1 ≤ a' ∧ a' ≤ 3 ∧ 
    -5 < b' ∧ b' < 3 ∧ 
    x = a' + |b'| :=
by sorry

end NUMINAMATH_CALUDE_range_of_sum_and_abs_l514_51473


namespace NUMINAMATH_CALUDE_combination_problem_l514_51457

theorem combination_problem (n : ℕ) 
  (h : Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8) : n = 14 := by
  sorry

end NUMINAMATH_CALUDE_combination_problem_l514_51457


namespace NUMINAMATH_CALUDE_train_speed_l514_51465

/-- Given a train of length 160 meters that crosses a stationary point in 18 seconds, 
    its speed is 32 km/h. -/
theorem train_speed (length : Real) (time : Real) (speed : Real) : 
  length = 160 ∧ time = 18 → speed = (length / time) * 3.6 → speed = 32 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l514_51465


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l514_51426

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^3 + 3*X^2 = (X^2 + 4*X + 2) * q + (-X^2 - 2*X) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l514_51426


namespace NUMINAMATH_CALUDE_joel_puzzles_l514_51449

/-- The number of puzzles Joel collected -/
def puzzles : ℕ := sorry

/-- The number of toys Joel's sister donated -/
def sister_toys : ℕ := sorry

/-- The total number of toys Joel donated -/
def total_toys : ℕ := 108

/-- The number of stuffed animals Joel collected -/
def stuffed_animals : ℕ := 18

/-- The number of action figures Joel collected -/
def action_figures : ℕ := 42

/-- The number of board games Joel collected -/
def board_games : ℕ := 2

/-- The number of toys Joel added from his own closet -/
def joel_toys : ℕ := 22

theorem joel_puzzles :
  puzzles = 13 ∧
  sister_toys * 2 = joel_toys ∧
  stuffed_animals + action_figures + board_games + puzzles + sister_toys + joel_toys = total_toys :=
sorry

end NUMINAMATH_CALUDE_joel_puzzles_l514_51449
