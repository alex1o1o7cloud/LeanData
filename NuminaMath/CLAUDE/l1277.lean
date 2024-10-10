import Mathlib

namespace bert_crossword_theorem_l1277_127758

/-- Represents a crossword puzzle --/
structure Crossword where
  size : Nat × Nat
  words : Nat

/-- Represents Bert's crossword solving habits --/
structure CrosswordHabit where
  puzzlesPerDay : Nat
  daysToUsePencil : Nat
  wordsPerPencil : Nat

/-- Calculate the average words per puzzle --/
def avgWordsPerPuzzle (habit : CrosswordHabit) : Nat :=
  habit.wordsPerPencil / (habit.puzzlesPerDay * habit.daysToUsePencil)

/-- Calculate the estimated words for a given puzzle size --/
def estimatedWords (baseSize : Nat × Nat) (baseWords : Nat) (newSize : Nat × Nat) : Nat :=
  let baseArea := baseSize.1 * baseSize.2
  let newArea := newSize.1 * newSize.2
  (baseWords * newArea) / baseArea

/-- Main theorem about Bert's crossword habits --/
theorem bert_crossword_theorem (habit : CrosswordHabit)
  (h1 : habit.puzzlesPerDay = 1)
  (h2 : habit.daysToUsePencil = 14)
  (h3 : habit.wordsPerPencil = 1050) :
  avgWordsPerPuzzle habit = 75 ∧
  estimatedWords (15, 15) 75 (21, 21) - 75 = 72 := by
  sorry

end bert_crossword_theorem_l1277_127758


namespace fiscal_revenue_scientific_notation_l1277_127779

/-- Converts a number in billions to scientific notation -/
def to_scientific_notation (x : ℝ) : ℝ × ℤ :=
  (1.14, 9)

/-- The fiscal revenue in billions -/
def fiscal_revenue : ℝ := 1.14

theorem fiscal_revenue_scientific_notation :
  to_scientific_notation fiscal_revenue = (1.14, 9) := by
  sorry

end fiscal_revenue_scientific_notation_l1277_127779


namespace azalea_profit_l1277_127708

/-- Calculates the shearer's payment based on the amount of wool produced -/
def shearer_payment (wool_amount : ℕ) : ℕ :=
  1000 + 
  (if wool_amount > 1000 then 1500 else 0) + 
  (if wool_amount > 2000 then (wool_amount - 2000) / 2 else 0)

/-- Calculates the revenue from wool sales based on quality distribution -/
def wool_revenue (total_wool : ℕ) : ℕ :=
  (total_wool / 2) * 30 +  -- High-quality
  (total_wool * 3 / 10) * 20 +  -- Medium-quality
  (total_wool / 5) * 10  -- Low-quality

theorem azalea_profit :
  let total_wool := 2400
  let revenue := wool_revenue total_wool
  let payment := shearer_payment total_wool
  revenue - payment = 52500 := by sorry

end azalea_profit_l1277_127708


namespace parallel_line_equation_line_K_equation_l1277_127733

/-- Given a line with equation y = mx + b, this function returns the y-intercept of a parallel line
    that is d units away from the original line. -/
def parallelLineYIntercept (m : ℝ) (b : ℝ) (d : ℝ) : Set ℝ :=
  {y | ∃ (sign : ℝ), sign = 1 ∨ sign = -1 ∧ y = b + sign * d * Real.sqrt (m^2 + 1)}

theorem parallel_line_equation (m b d : ℝ) :
  parallelLineYIntercept m b d = {b + d * Real.sqrt (m^2 + 1), b - d * Real.sqrt (m^2 + 1)} := by
  sorry

/-- The equation of line K, which is parallel to y = 1/2x + 3 and 5 units away from it. -/
theorem line_K_equation :
  parallelLineYIntercept (1/2) 3 5 = {3 + 5 * Real.sqrt 5 / 2, 3 - 5 * Real.sqrt 5 / 2} := by
  sorry

end parallel_line_equation_line_K_equation_l1277_127733


namespace watch_payment_in_dimes_l1277_127745

/-- The number of dimes in one dollar -/
def dimes_per_dollar : ℕ := 10

/-- The cost of the watch in dollars -/
def watch_cost : ℕ := 5

/-- Theorem: If a watch costs 5 dollars and is paid for entirely in dimes, 
    the number of dimes used is 50. -/
theorem watch_payment_in_dimes : 
  watch_cost * dimes_per_dollar = 50 := by sorry

end watch_payment_in_dimes_l1277_127745


namespace quadratic_equation_condition_l1277_127766

theorem quadratic_equation_condition (m : ℝ) : 
  (m^2 - 2 = 2 ∧ m + 2 ≠ 0) ↔ m = 2 :=
by sorry

end quadratic_equation_condition_l1277_127766


namespace dawn_hourly_income_l1277_127717

/-- Calculates the hourly income for Dawn's painting project -/
theorem dawn_hourly_income (num_paintings : ℕ) 
                            (sketch_time painting_time finish_time : ℝ) 
                            (watercolor_payment sketch_payment finish_payment : ℝ) : 
  num_paintings = 12 ∧ 
  sketch_time = 1.5 ∧ 
  painting_time = 2 ∧ 
  finish_time = 0.5 ∧
  watercolor_payment = 3600 ∧ 
  sketch_payment = 1200 ∧ 
  finish_payment = 300 → 
  (watercolor_payment + sketch_payment + finish_payment) / 
  (num_paintings * (sketch_time + painting_time + finish_time)) = 106.25 := by
sorry

end dawn_hourly_income_l1277_127717


namespace tribe_leadership_choices_l1277_127704

/-- The number of ways to choose leadership in a tribe --/
def choose_leadership (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (Nat.choose (n - 3) 2) * (Nat.choose (n - 5) 2)

/-- Theorem: For a tribe of 10 members with the given leadership structure, 
    there are 151200 ways to choose the leadership --/
theorem tribe_leadership_choices :
  choose_leadership 10 = 151200 := by
  sorry

end tribe_leadership_choices_l1277_127704


namespace square_sum_quadruple_l1277_127730

theorem square_sum_quadruple (n : ℕ) (h : n ≥ 8) :
  ∃ (a b c d : ℕ),
    a = 3*n^2 - 18*n - 39 ∧
    b = 3*n^2 + 6 ∧
    c = 3*n^2 + 18*n + 33 ∧
    d = 3*n^2 + 36*n + 42 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (w x y z : ℕ),
      a + b + c = w^2 ∧
      a + b + d = x^2 ∧
      a + c + d = y^2 ∧
      b + c + d = z^2 :=
by sorry

end square_sum_quadruple_l1277_127730


namespace polygon_side_length_l1277_127736

theorem polygon_side_length (n : ℕ) (h : n > 0) :
  ∃ (side_length : ℝ),
    side_length ≥ Real.sqrt (1/2 * (1 - Real.cos (π / n))) ∧
    (∃ (vertices : Fin (2*n) → ℝ × ℝ),
      (∀ i : Fin (2*n), ∃ j : Fin (2*n), i ≠ j ∧
        ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 1) ∧
      (∃ i j : Fin (2*n), i ≠ j ∧
        ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 ≥ side_length^2)) :=
by sorry

end polygon_side_length_l1277_127736


namespace circle_transformation_l1277_127701

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point to the right by a given distance -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

/-- The main theorem -/
theorem circle_transformation (C : ℝ × ℝ) (h : C = (3, -4)) :
  (translate_right (reflect_x C) 5) = (8, 4) := by sorry

end circle_transformation_l1277_127701


namespace f_behavior_at_infinity_l1277_127720

def f (x : ℝ) := -3 * x^3 + 4 * x^2 + 1

theorem f_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → f x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → f x > M) :=
by sorry

end f_behavior_at_infinity_l1277_127720


namespace homework_selection_is_systematic_l1277_127770

/-- Represents a sampling method --/
inductive SamplingMethod
  | Stratified
  | Lottery
  | Random
  | Systematic

/-- Represents a school's homework selection process --/
structure HomeworkSelection where
  selectFromEachClass : Bool
  selectionCriteria : Nat → Bool
  studentsArranged : Bool
  largeStudentPopulation : Bool

/-- Determines the sampling method based on the selection process --/
def determineSamplingMethod (selection : HomeworkSelection) : SamplingMethod :=
  sorry

/-- Theorem stating that the given selection process is Systematic Sampling --/
theorem homework_selection_is_systematic 
  (selection : HomeworkSelection)
  (h1 : selection.selectFromEachClass = true)
  (h2 : selection.selectionCriteria = λ id => id % 10 = 5)
  (h3 : selection.studentsArranged = true)
  (h4 : selection.largeStudentPopulation = true) :
  determineSamplingMethod selection = SamplingMethod.Systematic :=
sorry

end homework_selection_is_systematic_l1277_127770


namespace complex_sum_equals_negative_two_l1277_127787

theorem complex_sum_equals_negative_two (z : ℂ) 
  (h1 : z = Complex.exp (6 * Real.pi * Complex.I / 11))
  (h2 : z^11 = 1) : 
  (z / (1 + z^3)) + (z^2 / (1 + z^6)) + (z^4 / (1 + z^9)) = -2 := by
  sorry

end complex_sum_equals_negative_two_l1277_127787


namespace strawberry_basket_count_l1277_127702

theorem strawberry_basket_count (baskets : ℕ) (friends : ℕ) (total : ℕ) :
  baskets = 6 →
  friends = 3 →
  total = 1200 →
  ∃ (strawberries_per_basket : ℕ),
    strawberries_per_basket * baskets * (friends + 1) = total ∧
    strawberries_per_basket = 50 := by
  sorry

end strawberry_basket_count_l1277_127702


namespace no_solution_for_fermat_like_equation_l1277_127734

theorem no_solution_for_fermat_like_equation :
  ∀ (x y z k : ℕ), x < k → y < k → x^k + y^k ≠ z^k := by
  sorry

end no_solution_for_fermat_like_equation_l1277_127734


namespace line_inclination_angle_l1277_127747

-- Define the parametric equations
def x (t : ℝ) : ℝ := 1 + t
def y (t : ℝ) : ℝ := 1 - t

-- Define the line using the parametric equations
def line : Set (ℝ × ℝ) := {(x t, y t) | t : ℝ}

-- State the theorem
theorem line_inclination_angle :
  let slope := (y 1 - y 0) / (x 1 - x 0)
  let inclination_angle := Real.arctan slope
  inclination_angle = 3 * π / 4 := by
  sorry

end line_inclination_angle_l1277_127747


namespace bookstore_change_percentage_l1277_127724

def book_prices : List ℝ := [10, 8, 6, 4, 3, 5]
def discount_rate : ℝ := 0.1
def payment_amount : ℝ := 50

theorem bookstore_change_percentage :
  let total_price := book_prices.sum
  let discounted_price := total_price * (1 - discount_rate)
  let change := payment_amount - discounted_price
  let change_percentage := (change / payment_amount) * 100
  change_percentage = 35.2 := by sorry

end bookstore_change_percentage_l1277_127724


namespace quadratic_factorization_l1277_127740

theorem quadratic_factorization (x : ℝ) : 
  (x^2 - 6*x - 11 = 0) ↔ ((x - 3)^2 = 20) := by
sorry

end quadratic_factorization_l1277_127740


namespace mary_james_seating_probability_l1277_127778

/-- The number of chairs in the row -/
def total_chairs : ℕ := 10

/-- The number of chairs Mary can choose from -/
def mary_choices : ℕ := 9

/-- The number of chairs James can choose from -/
def james_choices : ℕ := 10

/-- The probability that Mary and James do not sit next to each other -/
def prob_not_adjacent : ℚ := 8/9

theorem mary_james_seating_probability :
  prob_not_adjacent = 1 - (mary_choices.pred / (mary_choices * james_choices)) :=
by sorry

end mary_james_seating_probability_l1277_127778


namespace fermat_sum_of_two_squares_l1277_127769

theorem fermat_sum_of_two_squares (p : ℕ) (h_prime : Nat.Prime p) (h_mod : p % 4 = 1) :
  ∃ a b : ℤ, p = a^2 + b^2 := by sorry

end fermat_sum_of_two_squares_l1277_127769


namespace complement_A_union_B_l1277_127776

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B : Set ℝ := {y | ∃ x, y = |x|}

-- State the theorem
theorem complement_A_union_B :
  (Aᶜ ∪ B) = {x : ℝ | x > -1} := by sorry

end complement_A_union_B_l1277_127776


namespace line_slope_intercept_product_l1277_127705

theorem line_slope_intercept_product (m b : ℚ) : 
  m = 3/4 → b = 2 → m * b > 1 := by sorry

end line_slope_intercept_product_l1277_127705


namespace min_value_theorem_l1277_127703

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line equation -/
def line_eq (a b x y : ℝ) : Prop := 2*a*x - b*y + 2 = 0

/-- The center of the circle satisfies the circle equation -/
def center_satisfies_circle (x₀ y₀ : ℝ) : Prop := circle_eq x₀ y₀

/-- The line passes through the center of the circle -/
def line_passes_through_center (a b x₀ y₀ : ℝ) : Prop := line_eq a b x₀ y₀

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (x₀ y₀ : ℝ) (h_center : center_satisfies_circle x₀ y₀) 
  (h_line : line_passes_through_center a b x₀ y₀) : 
  (1 / a + 1 / b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), 1 / a₀ + 1 / b₀ = 4 := by
  sorry

end min_value_theorem_l1277_127703


namespace cube_surface_area_l1277_127783

theorem cube_surface_area (a : ℝ) (h : a > 0) : 
  ∃ (s : ℝ), s > 0 ∧ s * Real.sqrt 3 = a ∧ 6 * s^2 = 2 * a^2 :=
sorry

end cube_surface_area_l1277_127783


namespace total_turtles_received_l1277_127788

theorem total_turtles_received (martha_turtles : ℕ) (marion_extra_turtles : ℕ) : 
  martha_turtles = 40 → 
  marion_extra_turtles = 20 → 
  martha_turtles + (martha_turtles + marion_extra_turtles) = 100 := by
sorry

end total_turtles_received_l1277_127788


namespace max_square_plots_l1277_127764

/-- Represents the dimensions of the field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available fencing -/
def availableFence : ℕ := 2250

/-- Calculates the number of square plots given the side length -/
def numPlots (dimensions : FieldDimensions) (sideLength : ℕ) : ℕ :=
  (dimensions.width / sideLength) * (dimensions.length / sideLength)

/-- Calculates the required fencing for a given configuration -/
def requiredFencing (dimensions : FieldDimensions) (sideLength : ℕ) : ℕ :=
  (dimensions.width / sideLength - 1) * dimensions.length +
  (dimensions.length / sideLength - 1) * dimensions.width

/-- Checks if a given side length is valid for the field dimensions -/
def isValidSideLength (dimensions : FieldDimensions) (sideLength : ℕ) : Prop :=
  sideLength > 0 ∧
  dimensions.width % sideLength = 0 ∧
  dimensions.length % sideLength = 0

theorem max_square_plots (dimensions : FieldDimensions)
    (h1 : dimensions.width = 30)
    (h2 : dimensions.length = 45) :
    (∃ (sideLength : ℕ),
      isValidSideLength dimensions sideLength ∧
      requiredFencing dimensions sideLength ≤ availableFence ∧
      numPlots dimensions sideLength = 150 ∧
      (∀ (s : ℕ), isValidSideLength dimensions s →
        requiredFencing dimensions s ≤ availableFence →
        numPlots dimensions s ≤ 150)) :=
  sorry

end max_square_plots_l1277_127764


namespace distance_to_place_distance_calculation_l1277_127725

/-- Calculates the distance to a place given rowing speeds and time -/
theorem distance_to_place (still_water_speed : ℝ) (current_velocity : ℝ) (total_time : ℝ) : ℝ :=
  let downstream_speed := still_water_speed + current_velocity
  let upstream_speed := still_water_speed - current_velocity
  let downstream_time := (total_time * upstream_speed) / (downstream_speed + upstream_speed)
  let distance := downstream_time * downstream_speed
  distance

/-- The distance to the place is approximately 10.83 km -/
theorem distance_calculation : 
  abs (distance_to_place 8 2.5 3 - 10.83) < 0.01 := by
  sorry

end distance_to_place_distance_calculation_l1277_127725


namespace fraction_simplification_l1277_127712

theorem fraction_simplification (x : ℝ) (h : x = 7) : 
  (x^6 - 36*x^3 + 324) / (x^3 - 18) = 325 := by
  sorry

end fraction_simplification_l1277_127712


namespace classmate_heights_most_suitable_l1277_127719

-- Define the survey options
inductive SurveyOption
  | LightBulbLifespan
  | WaterQualityGanRiver
  | TVProgramViewership
  | ClassmateHeights

-- Define the characteristic of being suitable for a comprehensive survey
def SuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.ClassmateHeights => True
  | _ => False

-- Theorem statement
theorem classmate_heights_most_suitable :
  SuitableForComprehensiveSurvey SurveyOption.ClassmateHeights ∧
  (∀ option : SurveyOption, option ≠ SurveyOption.ClassmateHeights →
    ¬SuitableForComprehensiveSurvey option) :=
by sorry

end classmate_heights_most_suitable_l1277_127719


namespace intersection_circles_power_l1277_127759

/-- Given two circles centered on the x-axis that intersect at points M(3a-b, 5) and N(9, 2a+3b), prove that a^b = 1/8 -/
theorem intersection_circles_power (a b : ℝ) : 
  (3 * a - b = 9) → (2 * a + 3 * b = -5) → a^b = 1/8 := by
  sorry

end intersection_circles_power_l1277_127759


namespace parabola_circle_tangent_line_l1277_127785

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition that a circle's center is on the parabola
def circle_center_on_parabola (c : Circle) : Prop :=
  parabola c.center.1 c.center.2

-- Define the condition that a circle passes through a point
def circle_passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 = c.radius^2

-- Define a line by its equation y = mx + b
structure Line where
  m : ℝ
  b : ℝ

-- Define the condition that a circle is tangent to a line
def circle_tangent_to_line (c : Circle) (l : Line) : Prop :=
  ∃ (x y : ℝ), y = l.m * x + l.b ∧
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ∧
  (c.center.1 - x)^2 + (c.center.2 - y)^2 = c.radius^2

theorem parabola_circle_tangent_line :
  ∀ (c : Circle) (l : Line),
  circle_center_on_parabola c →
  circle_passes_through c (0, 1) →
  circle_tangent_to_line c l →
  l.m = 0 ∧ l.b = -1 :=
sorry

end parabola_circle_tangent_line_l1277_127785


namespace factorization_equality_l1277_127754

theorem factorization_equality (x y : ℝ) : 25*x - x*y^2 = x*(5+y)*(5-y) := by
  sorry

end factorization_equality_l1277_127754


namespace april_sales_calculation_l1277_127772

def january_sales : ℕ := 90
def february_sales : ℕ := 50
def march_sales : ℕ := 70
def average_sales : ℕ := 72
def total_months : ℕ := 5

theorem april_sales_calculation :
  ∃ (april_sales may_sales : ℕ),
    (january_sales + february_sales + march_sales + april_sales + may_sales) / total_months = average_sales ∧
    april_sales = 75 := by
  sorry

end april_sales_calculation_l1277_127772


namespace arcade_change_machine_l1277_127791

theorem arcade_change_machine (total_bills : ℕ) (one_dollar_bills : ℕ) : 
  total_bills = 200 → one_dollar_bills = 175 → 
  (total_bills - one_dollar_bills) * 5 + one_dollar_bills = 300 := by
  sorry

end arcade_change_machine_l1277_127791


namespace quadratic_function_property_l1277_127737

/-- Given a quadratic function y = ax^2 + bx - 1 where a ≠ 0 and 
    the graph passes through the point (1, 1), prove that 1 - a - b = -1 -/
theorem quadratic_function_property (a b : ℝ) (h1 : a ≠ 0) 
    (h2 : a * 1^2 + b * 1 - 1 = 1) : 1 - a - b = -1 := by
  sorry

end quadratic_function_property_l1277_127737


namespace early_arrival_time_l1277_127711

/-- Proves that a boy walking at 5/4 of his usual rate arrives 4 minutes early when his usual time is 20 minutes. -/
theorem early_arrival_time (usual_time : ℝ) (usual_rate : ℝ) (faster_rate : ℝ) :
  usual_time = 20 →
  faster_rate = (5 / 4) * usual_rate →
  usual_time - (usual_time * usual_rate / faster_rate) = 4 := by
sorry

end early_arrival_time_l1277_127711


namespace movie_savings_theorem_l1277_127723

/-- Represents the savings calculation for a movie outing --/
def movie_savings (regular_price : ℚ) (student_discount : ℚ) (senior_discount : ℚ) 
  (early_discount : ℚ) (popcorn_price : ℚ) (popcorn_discount : ℚ) 
  (nachos_price : ℚ) (nachos_discount : ℚ) (hotdog_price : ℚ) 
  (hotdog_discount : ℚ) (combo_discount : ℚ) : ℚ :=
  let regular_tickets := 2 * regular_price
  let student_ticket := regular_price - student_discount
  let senior_ticket := regular_price - senior_discount
  let early_factor := 1 - early_discount
  let early_tickets := (regular_tickets + student_ticket + senior_ticket) * early_factor
  let ticket_savings := regular_tickets + student_ticket + senior_ticket - early_tickets
  let food_regular := popcorn_price + nachos_price + hotdog_price
  let food_discounted := popcorn_price * (1 - popcorn_discount) + 
                         nachos_price * (1 - nachos_discount) + 
                         hotdog_price * (1 - hotdog_discount)
  let food_combo := popcorn_price * (1 - popcorn_discount) + 
                    nachos_price * (1 - nachos_discount) + 
                    hotdog_price * (1 - hotdog_discount) * (1 - combo_discount)
  let food_savings := food_regular - food_combo
  ticket_savings + food_savings

/-- The total savings for the movie outing is $16.80 --/
theorem movie_savings_theorem : 
  movie_savings 10 2 3 (1/5) 10 (1/2) 8 (3/10) 6 (1/5) (1/4) = 84/5 := by
  sorry

end movie_savings_theorem_l1277_127723


namespace intersection_implies_m_and_n_l1277_127771

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x + 2| < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - m) * (x - 2) < 0}

-- State the theorem
theorem intersection_implies_m_and_n (m n : ℝ) :
  A ∩ B m = Set.Ioo (-1) n → m = -1 ∧ n = 1 := by
  sorry

end intersection_implies_m_and_n_l1277_127771


namespace union_of_given_sets_l1277_127757

theorem union_of_given_sets :
  let A : Set ℕ := {0, 1}
  let B : Set ℕ := {1, 2}
  A ∪ B = {0, 1, 2} := by sorry

end union_of_given_sets_l1277_127757


namespace solution_value_l1277_127797

theorem solution_value (x y m : ℝ) : x - 2*y = m → x = 2 → y = 1 → m = 0 := by
  sorry

end solution_value_l1277_127797


namespace half_plus_five_equals_thirteen_l1277_127732

theorem half_plus_five_equals_thirteen (n : ℕ) (h : n = 16) : n / 2 + 5 = 13 := by
  sorry

end half_plus_five_equals_thirteen_l1277_127732


namespace rhombus_area_l1277_127748

/-- The area of a rhombus with side length 13 units and one interior angle of 60 degrees is (169√3)/2 square units. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 13) (h2 : θ = π / 3) :
  s^2 * Real.sin θ = (169 * Real.sqrt 3) / 2 := by
  sorry

end rhombus_area_l1277_127748


namespace find_a_l1277_127780

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2^x else a + 2*x

-- State the theorem
theorem find_a : ∃ a : ℝ, (f a (f a (-1)) = 2) ∧ (a = 1) := by
  sorry

end find_a_l1277_127780


namespace simple_interest_rate_equivalence_l1277_127728

theorem simple_interest_rate_equivalence (P : ℝ) (P_pos : P > 0) :
  let initial_rate : ℝ := 5 / 100
  let initial_time : ℝ := 8
  let new_time : ℝ := 5
  let new_rate : ℝ := 8 / 100
  (P * initial_rate * initial_time) = (P * new_rate * new_time) := by
sorry

end simple_interest_rate_equivalence_l1277_127728


namespace max_value_abc_l1277_127777

theorem max_value_abc (a b c : ℝ) (h : a + 3 * b + c = 6) :
  (∀ x y z : ℝ, x + 3 * y + z = 6 → a * b + a * c + b * c ≥ x * y + x * z + y * z) →
  a * b + a * c + b * c = 12 :=
sorry

end max_value_abc_l1277_127777


namespace phillips_remaining_money_l1277_127744

/-- Calculates the remaining money after Phillip's shopping trip --/
def remaining_money (initial_amount : ℚ) 
  (orange_price : ℚ) (orange_quantity : ℚ)
  (apple_price : ℚ) (apple_quantity : ℚ)
  (candy_price : ℚ)
  (egg_price : ℚ) (egg_quantity : ℚ)
  (milk_price : ℚ)
  (sales_tax_rate : ℚ)
  (apple_discount_rate : ℚ) : ℚ :=
  sorry

/-- Theorem stating that Phillip's remaining money is $51.91 --/
theorem phillips_remaining_money :
  remaining_money 95 3 2 3.5 4 6 6 2 4 0.08 0.15 = 51.91 :=
  sorry

end phillips_remaining_money_l1277_127744


namespace quadratic_roots_imply_c_l1277_127721

theorem quadratic_roots_imply_c (c : ℚ) : 
  (∀ x : ℚ, 2 * x^2 + 14 * x + c = 0 ↔ x = (-14 + Real.sqrt 10) / 4 ∨ x = (-14 - Real.sqrt 10) / 4) →
  c = 93 / 4 := by
sorry

end quadratic_roots_imply_c_l1277_127721


namespace range_of_x_l1277_127760

theorem range_of_x (x : ℝ) : 
  (Real.sqrt ((1 - 2*x)^2) = 2*x - 1) → x ≥ (1/2 : ℝ) :=
by sorry

end range_of_x_l1277_127760


namespace building_height_l1277_127793

/-- Given a flagpole and a building casting shadows under similar conditions,
    this theorem proves that the height of the building can be determined
    using the principle of similar triangles. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (flagpole_height_pos : 0 < flagpole_height)
  (flagpole_shadow_pos : 0 < flagpole_shadow)
  (building_shadow_pos : 0 < building_shadow)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 70) :
  (flagpole_height / flagpole_shadow) * building_shadow = 28 :=
by sorry

end building_height_l1277_127793


namespace pinecone_count_l1277_127752

theorem pinecone_count (initial : ℕ) : 
  (initial : ℝ) * 0.2 = initial * 0.2 ∧  -- 20% eaten by reindeer
  (initial : ℝ) * 0.4 = 2 * (initial * 0.2) ∧  -- Twice as many eaten by squirrels
  (initial : ℝ) * 0.25 * 0.4 = initial * 0.1 ∧  -- 25% of remainder collected for fires
  (initial : ℝ) * 0.3 = 600 →  -- 600 pinecones left
  initial = 2000 := by
sorry

end pinecone_count_l1277_127752


namespace two_distinct_roots_condition_l1277_127709

theorem two_distinct_roots_condition (b : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ |2^x - 1| = b ∧ |2^y - 1| = b) ↔ 0 < b ∧ b < 1 := by
  sorry

end two_distinct_roots_condition_l1277_127709


namespace smallest_prime_2018_factorial_l1277_127765

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem smallest_prime_2018_factorial :
  ∃ p : ℕ, 
    Prime p ∧ 
    p = 509 ∧ 
    is_divisible (factorial 2018) (p^3) ∧ 
    ¬is_divisible (factorial 2018) (p^4) ∧
    ∀ q : ℕ, Prime q → q < p → 
      ¬(is_divisible (factorial 2018) (q^3) ∧ ¬is_divisible (factorial 2018) (q^4)) :=
by sorry

end smallest_prime_2018_factorial_l1277_127765


namespace convex_polyhedron_has_32_faces_l1277_127756

/-- A convex polyhedron with pentagonal and hexagonal faces -/
structure ConvexPolyhedron where
  /-- Number of pentagonal faces -/
  pentagonFaces : ℕ
  /-- Number of hexagonal faces -/
  hexagonFaces : ℕ
  /-- Three faces meet at each vertex -/
  threeAtVertex : True
  /-- Each pentagon shares edges with 5 hexagons -/
  pentagonSharing : pentagonFaces * 5 = hexagonFaces * 3
  /-- Each hexagon shares edges with 3 pentagons -/
  hexagonSharing : hexagonFaces * 3 = pentagonFaces * 5

/-- The total number of faces in the polyhedron -/
def ConvexPolyhedron.totalFaces (p : ConvexPolyhedron) : ℕ :=
  p.pentagonFaces + p.hexagonFaces

/-- Theorem: The convex polyhedron has exactly 32 faces -/
theorem convex_polyhedron_has_32_faces (p : ConvexPolyhedron) :
  p.totalFaces = 32 := by
  sorry

#eval ConvexPolyhedron.totalFaces ⟨12, 20, trivial, rfl, rfl⟩

end convex_polyhedron_has_32_faces_l1277_127756


namespace sin_x_minus_pi_third_l1277_127755

theorem sin_x_minus_pi_third (x : ℝ) (h : Real.cos (x + π / 6) = 1 / 3) :
  Real.sin (x - π / 3) = -1 / 3 := by
  sorry

end sin_x_minus_pi_third_l1277_127755


namespace right_triangle_existence_l1277_127796

noncomputable def f (x : ℝ) : ℝ :=
  if x < Real.exp 1 then -x^3 + x^2 else Real.log x

theorem right_triangle_existence (a : ℝ) :
  (∃ t : ℝ, t ≥ Real.exp 1 ∧
    ((-t^2 + f t * (-t^3 + t^2) = 0) ∧
     (∃ P Q : ℝ × ℝ, P = (t, f t) ∧ Q = (-t, f (-t)) ∧
       (P.1 * Q.1 + P.2 * Q.2 = 0) ∧
       ((P.1 + Q.1) / 2 = 0))))
  ↔ (0 < a ∧ a ≤ 1 / (Real.exp 1 * Real.log (Real.exp 1) + 1)) :=
by sorry

end right_triangle_existence_l1277_127796


namespace power_mod_seven_l1277_127718

theorem power_mod_seven : 3^2023 % 7 = 3 := by sorry

end power_mod_seven_l1277_127718


namespace sally_nickels_l1277_127716

-- Define the initial state and gifts
def initial_nickels : ℕ := 7
def dad_gift : ℕ := 9
def mom_gift : ℕ := 2

-- Theorem to prove
theorem sally_nickels : initial_nickels + dad_gift + mom_gift = 18 := by
  sorry

end sally_nickels_l1277_127716


namespace cake_recipe_flour_flour_in_recipe_l1277_127794

theorem cake_recipe_flour (salt_cups : ℕ) (flour_added : ℕ) (flour_salt_diff : ℕ) : ℕ :=
  let total_flour := salt_cups + flour_salt_diff
  total_flour

theorem flour_in_recipe :
  cake_recipe_flour 7 2 3 = 10 := by
  sorry

end cake_recipe_flour_flour_in_recipe_l1277_127794


namespace arithmetic_sequence_difference_l1277_127713

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Theorem statement
theorem arithmetic_sequence_difference
  (a : ℕ → ℝ) (d : ℝ) (h : isArithmeticSequence a d) (h_d : d = 2) :
  a 5 - a 2 = 6 :=
by
  sorry

end arithmetic_sequence_difference_l1277_127713


namespace mary_number_is_14_l1277_127751

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def switch_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem mary_number_is_14 :
  ∃! x : ℕ, is_two_digit x ∧
    91 ≤ switch_digits (4 * x - 7) ∧
    switch_digits (4 * x - 7) ≤ 95 :=
by
  sorry

end mary_number_is_14_l1277_127751


namespace expression_simplification_l1277_127735

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (1 / (x - 2) + 1) / ((x^2 - 2*x + 1) / (x - 2)) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l1277_127735


namespace truck_load_after_deliveries_l1277_127739

theorem truck_load_after_deliveries :
  let initial_load : ℝ := 50000
  let first_unload_percentage : ℝ := 0.1
  let second_unload_percentage : ℝ := 0.2
  let after_first_delivery := initial_load * (1 - first_unload_percentage)
  let final_load := after_first_delivery * (1 - second_unload_percentage)
  final_load = 36000 := by sorry

end truck_load_after_deliveries_l1277_127739


namespace arithmetic_mean_of_18_27_45_l1277_127738

theorem arithmetic_mean_of_18_27_45 (S : Finset ℕ) :
  S = {18, 27, 45} →
  (S.sum id) / S.card = 30 :=
by
  sorry

end arithmetic_mean_of_18_27_45_l1277_127738


namespace basic_computer_price_is_correct_l1277_127714

/-- The price of the basic computer -/
def basic_computer_price : ℝ := 1040

/-- The price of the printer -/
def printer_price : ℝ := 2500 - basic_computer_price

/-- The total price of the basic computer and printer -/
def total_price : ℝ := 2500

/-- The price of the first enhanced computer -/
def enhanced_computer1_price : ℝ := basic_computer_price + 800

/-- The price of the second enhanced computer -/
def enhanced_computer2_price : ℝ := basic_computer_price + 1100

/-- The price of the third enhanced computer -/
def enhanced_computer3_price : ℝ := basic_computer_price + 1500

theorem basic_computer_price_is_correct :
  basic_computer_price + printer_price = total_price ∧
  enhanced_computer1_price + (1/5) * (enhanced_computer1_price + printer_price) = total_price ∧
  enhanced_computer2_price + (1/8) * (enhanced_computer2_price + printer_price) = total_price ∧
  enhanced_computer3_price + (1/10) * (enhanced_computer3_price + printer_price) = total_price :=
by sorry

end basic_computer_price_is_correct_l1277_127714


namespace zoo_animals_l1277_127749

theorem zoo_animals (M B L : ℕ) : 
  (26 ≤ M + B + L) ∧ (M + B + L ≤ 32) ∧
  (M + L > B) ∧
  (B + L = 2 * M) ∧
  (M + B = 3 * L + 3) ∧
  (2 * B = L) →
  B = 13 := by
sorry

end zoo_animals_l1277_127749


namespace expression_evaluation_l1277_127799

theorem expression_evaluation : (2^3 - 2^2) - (3^3 - 3^2) + (4^3 - 4^2) - (5^3 - 5^2) = -66 := by
  sorry

end expression_evaluation_l1277_127799


namespace table_tennis_lineups_l1277_127707

/-- Represents a team in the table tennis competition -/
structure Team :=
  (members : Finset Nat)
  (size : members.card = 5)

/-- Represents a lineup for the competition -/
structure Lineup :=
  (singles1 : Nat)
  (singles2 : Nat)
  (doubles1 : Nat)
  (doubles2 : Nat)
  (all_different : singles1 ≠ singles2 ∧ singles1 ≠ doubles1 ∧ singles1 ≠ doubles2 ∧ 
                   singles2 ≠ doubles1 ∧ singles2 ≠ doubles2 ∧ doubles1 ≠ doubles2)

/-- The theorem to be proved -/
theorem table_tennis_lineups (t : Team) (a : Nat) (h : a ∈ t.members) : 
  (∃ l : Finset Lineup, l.card = 60 ∧ ∀ lineup ∈ l, lineup.singles1 ∈ t.members ∧ 
                                                    lineup.singles2 ∈ t.members ∧ 
                                                    lineup.doubles1 ∈ t.members ∧ 
                                                    lineup.doubles2 ∈ t.members) ∧
  (∃ l : Finset Lineup, l.card = 36 ∧ ∀ lineup ∈ l, lineup.singles1 ∈ t.members ∧ 
                                                    lineup.singles2 ∈ t.members ∧ 
                                                    lineup.doubles1 ∈ t.members ∧ 
                                                    lineup.doubles2 ∈ t.members ∧ 
                                                    (lineup.doubles1 ≠ a ∧ lineup.doubles2 ≠ a)) :=
by sorry

end table_tennis_lineups_l1277_127707


namespace express_x_in_terms_of_y_l1277_127729

theorem express_x_in_terms_of_y (x y : ℝ) (h : 3 * x - 4 * y = 8) :
  x = (4 * y + 8) / 3 := by
  sorry

end express_x_in_terms_of_y_l1277_127729


namespace remainder_17_pow_2047_mod_23_l1277_127761

theorem remainder_17_pow_2047_mod_23 :
  (17 : ℤ) ^ 2047 % 23 = 11 := by sorry

end remainder_17_pow_2047_mod_23_l1277_127761


namespace money_division_l1277_127774

theorem money_division (p q r : ℕ) (total : ℕ) :
  p + q + r = total →
  p = 3 * (total / 22) →
  q = 7 * (total / 22) →
  r = 12 * (total / 22) →
  r - q = 3000 →
  q - p = 2400 :=
by sorry

end money_division_l1277_127774


namespace smallest_factorizable_b_is_correct_l1277_127782

/-- The smallest positive integer b for which x^2 + bx + 1760 factors into (x + p)(x + q) with integer p and q -/
def smallest_factorizable_b : ℕ := 84

/-- A polynomial of the form x^2 + bx + 1760 -/
def polynomial (b : ℕ) (x : ℤ) : ℤ := x^2 + b * x + 1760

/-- Checks if a polynomial can be factored into (x + p)(x + q) with integer p and q -/
def is_factorizable (b : ℕ) : Prop :=
  ∃ (p q : ℤ), ∀ x, polynomial b x = (x + p) * (x + q)

theorem smallest_factorizable_b_is_correct :
  (is_factorizable smallest_factorizable_b) ∧
  (∀ b : ℕ, b < smallest_factorizable_b → ¬(is_factorizable b)) :=
sorry

end smallest_factorizable_b_is_correct_l1277_127782


namespace rectangle_circumference_sum_l1277_127781

/-- Calculates the sum of coins around the circumference of a rectangle formed by coins -/
def circumference_sum (horizontal : Nat) (vertical : Nat) (coin_value : Nat) : Nat :=
  let horizontal_edge := 2 * (horizontal - 2)
  let vertical_edge := 2 * (vertical - 2)
  let corners := 4
  (horizontal_edge + vertical_edge + corners) * coin_value

/-- Theorem stating that the sum of coins around the circumference of a 6x4 rectangle of 100-won coins is 1600 won -/
theorem rectangle_circumference_sum :
  circumference_sum 6 4 100 = 1600 := by
  sorry

end rectangle_circumference_sum_l1277_127781


namespace french_students_count_l1277_127784

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 87)
  (h_german : german = 22)
  (h_both : both = 9)
  (h_neither : neither = 33) :
  ∃ french : ℕ, french = total - german + both - neither :=
by
  sorry

end french_students_count_l1277_127784


namespace graduating_class_size_l1277_127795

theorem graduating_class_size :
  let num_boys : ℕ := 138
  let girls_more_than_boys : ℕ := 69
  let num_girls : ℕ := num_boys + girls_more_than_boys
  let total_students : ℕ := num_boys + num_girls
  total_students = 345 := by sorry

end graduating_class_size_l1277_127795


namespace price_increase_percentage_l1277_127767

/-- Proves that the percentage increase in prices is 15% given the problem conditions -/
theorem price_increase_percentage (orange_price : ℝ) (mango_price : ℝ) (new_total_cost : ℝ) :
  orange_price = 40 →
  mango_price = 50 →
  new_total_cost = 1035 →
  10 * (orange_price * (1 + 15 / 100)) + 10 * (mango_price * (1 + 15 / 100)) = new_total_cost :=
by sorry

end price_increase_percentage_l1277_127767


namespace system_solution_set_system_solutions_l1277_127750

def system_has_solution (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x - 4 = a * (y^3 - 2)) ∧ (2 * x / (|y^3| + y^3) = Real.sqrt x)

theorem system_solution_set :
  {a : ℝ | system_has_solution a} = Set.Ioi 2 ∪ Set.Iic 0 :=
sorry

theorem system_solutions (a : ℝ) (h : system_has_solution a) :
  (∃ x y : ℝ, x = 4 ∧ y^3 = 2) ∨
  (∃ x y : ℝ, x = 0 ∧ y^3 = 2*a - 4) :=
sorry

end system_solution_set_system_solutions_l1277_127750


namespace rain_probabilities_l1277_127790

theorem rain_probabilities (p_monday p_tuesday : ℝ) 
  (h_monday : p_monday = 0.4)
  (h_tuesday : p_tuesday = 0.3)
  (h_independent : True)  -- This represents the independence assumption
  : (p_monday * p_tuesday = 0.12) ∧ 
    ((1 - p_monday) * (1 - p_tuesday) = 0.42) := by
  sorry

end rain_probabilities_l1277_127790


namespace calculate_expression_l1277_127731

theorem calculate_expression : 20.17 * 69 + 201.7 * 1.3 - 8.2 * 1.7 = 1640 := by
  sorry

end calculate_expression_l1277_127731


namespace sum_of_max_and_min_f_l1277_127700

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem sum_of_max_and_min_f : 
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
               (∀ x, m ≤ f x) ∧ (∃ x, f x = m) ∧
               (M + m = 2) :=
sorry

end sum_of_max_and_min_f_l1277_127700


namespace fast_reader_time_l1277_127753

/-- Given two people, where one reads 4 times faster than the other, 
    prove that if the slower reader takes 90 minutes to read a book, 
    the faster reader will take 22.5 minutes to read the same book. -/
theorem fast_reader_time (slow_reader_time : ℝ) (speed_ratio : ℝ) 
    (h1 : slow_reader_time = 90) 
    (h2 : speed_ratio = 4) : 
  slow_reader_time / speed_ratio = 22.5 := by
  sorry

#check fast_reader_time

end fast_reader_time_l1277_127753


namespace marvelous_divisible_by_five_infinitely_many_marvelous_numbers_l1277_127775

def is_marvelous (n : ℕ+) : Prop :=
  ∃ (a b c d e : ℕ+),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    (n : ℕ) % a = 0 ∧ (n : ℕ) % b = 0 ∧ (n : ℕ) % c = 0 ∧ (n : ℕ) % d = 0 ∧ (n : ℕ) % e = 0 ∧
    n = a^4 + b^4 + c^4 + d^4 + e^4

theorem marvelous_divisible_by_five (n : ℕ+) (h : is_marvelous n) :
  (n : ℕ) % 5 = 0 :=
sorry

theorem infinitely_many_marvelous_numbers :
  ∀ k : ℕ, ∃ n : ℕ+, n > k ∧ is_marvelous n :=
sorry

end marvelous_divisible_by_five_infinitely_many_marvelous_numbers_l1277_127775


namespace smallest_x_value_l1277_127762

theorem smallest_x_value (x y : ℕ+) (h : (0.8 : ℚ) = y / (186 + x)) : 
  x ≥ 4 ∧ ∃ (y' : ℕ+), (0.8 : ℚ) = y' / (186 + 4) := by
  sorry

end smallest_x_value_l1277_127762


namespace multiplication_value_proof_l1277_127727

theorem multiplication_value_proof (x : ℚ) : (3 / 4) * x = 9 → x = 12 := by
  sorry

end multiplication_value_proof_l1277_127727


namespace geometric_sequence_a6_l1277_127792

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  geometric_sequence a → a 3 = -3 → a 4 = 6 → a 6 = 24 := by
  sorry

end geometric_sequence_a6_l1277_127792


namespace largest_n_for_unique_k_l1277_127741

theorem largest_n_for_unique_k : ∃ (n : ℕ), n > 0 ∧ 
  (∃! (k : ℤ), (9 : ℚ)/17 < n/(n + k) ∧ n/(n + k) < 8/15) ∧
  (∀ (m : ℕ), m > n → ¬(∃! (k : ℤ), (9 : ℚ)/17 < m/(m + k) ∧ m/(m + k) < 8/15)) :=
by sorry

end largest_n_for_unique_k_l1277_127741


namespace hemisphere_surface_area_l1277_127786

theorem hemisphere_surface_area (base_area : ℝ) (h : base_area = 225 * Real.pi) :
  let radius := Real.sqrt (base_area / Real.pi)
  let curved_area := 2 * Real.pi * radius^2
  let total_area := curved_area + base_area
  total_area = 675 * Real.pi := by
sorry

end hemisphere_surface_area_l1277_127786


namespace faster_train_speed_l1277_127746

theorem faster_train_speed
  (train_length : ℝ)
  (slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 100)  -- Length of each train in meters
  (h2 : slower_speed = 36)   -- Speed of slower train in km/hr
  (h3 : passing_time = 72)   -- Time taken to pass in seconds
  : ∃ (faster_speed : ℝ), faster_speed = 46 :=
by
  sorry

end faster_train_speed_l1277_127746


namespace total_watch_time_l1277_127789

/-- Calculate the total watch time for John's videos in a week -/
theorem total_watch_time
  (short_video_length : ℕ)
  (long_video_multiplier : ℕ)
  (short_videos_per_day : ℕ)
  (long_videos_per_day : ℕ)
  (days_per_week : ℕ)
  (retention_rate : ℝ)
  (h1 : short_video_length = 2)
  (h2 : long_video_multiplier = 6)
  (h3 : short_videos_per_day = 2)
  (h4 : long_videos_per_day = 1)
  (h5 : days_per_week = 7)
  (h6 : 0 < retention_rate)
  (h7 : retention_rate ≤ 100)
  : ℝ :=
by
  sorry

#check total_watch_time

end total_watch_time_l1277_127789


namespace sum_of_digits_twice_square_222222222_l1277_127763

def n : ℕ := 9

def Y : ℕ := 2 * (10^n - 1) / 9

def Y_squared : ℕ := Y * Y

def doubled_Y_squared : ℕ := 2 * Y_squared

def sum_of_digits (x : ℕ) : ℕ :=
  if x < 10 then x else x % 10 + sum_of_digits (x / 10)

theorem sum_of_digits_twice_square_222222222 :
  sum_of_digits doubled_Y_squared = 126 := by sorry

end sum_of_digits_twice_square_222222222_l1277_127763


namespace cracked_to_broken_ratio_l1277_127798

/-- Represents the number of eggs in each category --/
structure EggCounts where
  total : ℕ
  broken : ℕ
  perfect : ℕ
  cracked : ℕ

/-- Theorem stating the ratio of cracked to broken eggs --/
theorem cracked_to_broken_ratio (e : EggCounts) : 
  e.total = 24 →
  e.broken = 3 →
  e.perfect - e.cracked = 9 →
  e.total = e.perfect + e.cracked + e.broken →
  (e.cracked : ℚ) / e.broken = 2 := by
  sorry

#check cracked_to_broken_ratio

end cracked_to_broken_ratio_l1277_127798


namespace square_equals_cube_of_digit_sum_l1277_127706

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem square_equals_cube_of_digit_sum (n : ℕ) :
  n ∈ Finset.range 1000 →
  (n^2 = (sum_of_digits n)^3) ↔ (n = 1 ∨ n = 27) := by sorry

end square_equals_cube_of_digit_sum_l1277_127706


namespace reflection_properties_l1277_127710

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a figure as a set of points
def Figure := Set Point2D

-- Define the reflection of a point about a line
def reflect (p : Point2D) (l : Line2D) : Point2D :=
  sorry

-- Define the reflection of a figure about a line
def reflectFigure (f : Figure) (l : Line2D) : Figure :=
  sorry

-- Define a predicate to check if a point is on a specific side of a line
def onSide (p : Point2D) (l : Line2D) (side : Bool) : Prop :=
  sorry

-- Define a predicate to check if a figure is on a specific side of a line
def figureOnSide (f : Figure) (l : Line2D) (side : Bool) : Prop :=
  sorry

-- Define a predicate to check if two figures have the same shape
def sameShape (f1 f2 : Figure) : Prop :=
  sorry

-- Define a predicate to check if a figure touches a line at given points
def touchesAt (f : Figure) (l : Line2D) (p q : Point2D) : Prop :=
  sorry

theorem reflection_properties 
  (f : Figure) (l : Line2D) (p q : Point2D) (side : Bool) :
  figureOnSide f l side →
  touchesAt f l p q →
  let f' := reflectFigure f l
  figureOnSide f' l (!side) ∧
  sameShape f f' ∧
  touchesAt f' l p q :=
by
  sorry

end reflection_properties_l1277_127710


namespace cosine_angle_equality_l1277_127726

theorem cosine_angle_equality (n : ℤ) : 
  (0 ≤ n ∧ n ≤ 180) ∧ (Real.cos (n * π / 180) = Real.cos (1124 * π / 180)) ↔ n = 44 := by
  sorry

end cosine_angle_equality_l1277_127726


namespace union_complement_problem_l1277_127742

def U : Set ℤ := {x : ℤ | |x| < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_complement_problem :
  A ∪ (U \ B) = {0, 1, 2} := by sorry

end union_complement_problem_l1277_127742


namespace pentagon_area_theorem_l1277_127743

/-- A pentagon is a polygon with 5 sides -/
structure Pentagon where
  sides : Fin 5 → ℝ

/-- The area of a pentagon -/
noncomputable def Pentagon.area (p : Pentagon) : ℝ := sorry

/-- Theorem: There exists a pentagon with sides 18, 25, 30, 28, and 25 units, and its area is 950 square units -/
theorem pentagon_area_theorem : 
  ∃ (p : Pentagon), 
    p.sides 0 = 18 ∧ 
    p.sides 1 = 25 ∧ 
    p.sides 2 = 30 ∧ 
    p.sides 3 = 28 ∧ 
    p.sides 4 = 25 ∧ 
    p.area = 950 := by sorry

end pentagon_area_theorem_l1277_127743


namespace at_least_one_black_certain_l1277_127773

/-- Represents the color of a ball -/
inductive BallColor
  | Black
  | White

/-- Represents the composition of balls in the bag -/
structure BagComposition where
  blackBalls : Nat
  whiteBalls : Nat

/-- Represents the result of drawing two balls -/
structure DrawResult where
  firstBall : BallColor
  secondBall : BallColor

/-- Defines the event of drawing at least one black ball -/
def AtLeastOneBlack (result : DrawResult) : Prop :=
  result.firstBall = BallColor.Black ∨ result.secondBall = BallColor.Black

/-- The theorem to be proved -/
theorem at_least_one_black_certain (bag : BagComposition) 
    (h1 : bag.blackBalls = 2) 
    (h2 : bag.whiteBalls = 1) : 
    ∀ (result : DrawResult), AtLeastOneBlack result :=
  sorry

end at_least_one_black_certain_l1277_127773


namespace petes_flag_problem_l1277_127768

/-- The number of stars on the US flag -/
def us_stars : ℕ := 50

/-- The number of stripes on the US flag -/
def us_stripes : ℕ := 13

/-- The number of circles on Pete's flag -/
def petes_circles : ℕ := us_stars / 2 - 3

/-- The number of squares on Pete's flag -/
def petes_squares (x : ℕ) : ℕ := 2 * us_stripes + x

/-- The total number of shapes on Pete's flag -/
def total_shapes : ℕ := 54

theorem petes_flag_problem :
  ∃ x : ℕ, petes_circles + petes_squares x = total_shapes ∧ x = 6 := by
  sorry

end petes_flag_problem_l1277_127768


namespace only_setC_in_proportion_l1277_127715

-- Define a structure for a set of four line segments
structure FourSegments where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the property of being in proportion
def isInProportion (segments : FourSegments) : Prop :=
  segments.a * segments.d = segments.b * segments.c

-- Define the four sets of line segments
def setA : FourSegments := ⟨3, 5, 6, 9⟩
def setB : FourSegments := ⟨3, 5, 8, 9⟩
def setC : FourSegments := ⟨3, 9, 10, 30⟩
def setD : FourSegments := ⟨3, 6, 7, 9⟩

-- State the theorem
theorem only_setC_in_proportion :
  isInProportion setC ∧
  ¬isInProportion setA ∧
  ¬isInProportion setB ∧
  ¬isInProportion setD :=
sorry

end only_setC_in_proportion_l1277_127715


namespace science_tech_group_size_l1277_127722

theorem science_tech_group_size :
  ∀ (girls boys : ℕ),
  girls = 18 →
  girls = 2 * boys - 2 →
  girls + boys = 28 :=
by
  sorry

end science_tech_group_size_l1277_127722
