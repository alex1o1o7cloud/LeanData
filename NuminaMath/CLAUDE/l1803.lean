import Mathlib

namespace NUMINAMATH_CALUDE_x_intercept_distance_l1803_180365

/-- Two lines with slopes 4 and 6 intersecting at (8,12) have x-intercepts with distance 1 -/
theorem x_intercept_distance (line1 line2 : ℝ → ℝ) : 
  (∀ x, line1 x = 4*x - 20) →  -- Equation of line with slope 4
  (∀ x, line2 x = 6*x - 36) →  -- Equation of line with slope 6
  line1 8 = 12 →              -- Lines intersect at (8,12)
  line2 8 = 12 →              -- Lines intersect at (8,12)
  ∃ x1 x2, line1 x1 = 0 ∧ line2 x2 = 0 ∧ |x2 - x1| = 1 := by
sorry

end NUMINAMATH_CALUDE_x_intercept_distance_l1803_180365


namespace NUMINAMATH_CALUDE_least_positive_integer_with_congruences_l1803_180392

theorem least_positive_integer_with_congruences : ∃ b : ℕ+, 
  (b : ℤ) ≡ 2 [ZMOD 3] ∧ 
  (b : ℤ) ≡ 3 [ZMOD 4] ∧ 
  (b : ℤ) ≡ 4 [ZMOD 5] ∧ 
  (b : ℤ) ≡ 6 [ZMOD 7] ∧ 
  ∀ c : ℕ+, 
    ((c : ℤ) ≡ 2 [ZMOD 3] ∧ 
     (c : ℤ) ≡ 3 [ZMOD 4] ∧ 
     (c : ℤ) ≡ 4 [ZMOD 5] ∧ 
     (c : ℤ) ≡ 6 [ZMOD 7]) → 
    b ≤ c :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_congruences_l1803_180392


namespace NUMINAMATH_CALUDE_lent_sum_theorem_l1803_180357

/-- Represents the sum of money lent in two parts -/
structure LentSum where
  first_part : ℕ
  second_part : ℕ
  total : ℕ

/-- Calculates the interest on a principal amount for a given rate and time -/
def calculate_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time

theorem lent_sum_theorem (s : LentSum) :
  s.second_part = 1672 →
  calculate_interest s.first_part 3 8 = calculate_interest s.second_part 5 3 →
  s.total = s.first_part + s.second_part →
  s.total = 2717 := by
    sorry

end NUMINAMATH_CALUDE_lent_sum_theorem_l1803_180357


namespace NUMINAMATH_CALUDE_waiter_new_customers_l1803_180314

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (customers_left : ℕ) 
  (final_customers : ℕ) 
  (h1 : initial_customers = 47) 
  (h2 : customers_left = 41) 
  (h3 : final_customers = 26) : 
  final_customers - (initial_customers - customers_left) = 20 :=
by sorry

end NUMINAMATH_CALUDE_waiter_new_customers_l1803_180314


namespace NUMINAMATH_CALUDE_monkeys_eating_bananas_l1803_180344

/-- Given the rate at which monkeys eat bananas, prove that 6 monkeys are needed to eat 18 bananas in 18 minutes -/
theorem monkeys_eating_bananas 
  (initial_monkeys : ℕ) 
  (initial_time : ℕ) 
  (initial_bananas : ℕ) 
  (target_time : ℕ) 
  (target_bananas : ℕ) 
  (h1 : initial_monkeys = 6) 
  (h2 : initial_time = 6) 
  (h3 : initial_bananas = 6) 
  (h4 : target_time = 18) 
  (h5 : target_bananas = 18) : 
  (target_bananas * initial_time * initial_monkeys) / (initial_bananas * target_time) = 6 := by
  sorry

end NUMINAMATH_CALUDE_monkeys_eating_bananas_l1803_180344


namespace NUMINAMATH_CALUDE_wheel_speed_proof_l1803_180305

/-- Proves that the original speed of a wheel is 7.5 mph given specific conditions -/
theorem wheel_speed_proof (wheel_circumference : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  wheel_circumference = 15 →  -- circumference in feet
  speed_increase = 8 →        -- speed increase in mph
  time_decrease = 1/3 →       -- time decrease in seconds
  ∃ (original_speed : ℝ),
    original_speed = 7.5 ∧    -- original speed in mph
    (original_speed + speed_increase) * (3600 * (15 / (5280 * original_speed)) - time_decrease / 3600) =
    15 / 5280 * 3600 :=
by
  sorry


end NUMINAMATH_CALUDE_wheel_speed_proof_l1803_180305


namespace NUMINAMATH_CALUDE_maria_trip_fraction_l1803_180361

theorem maria_trip_fraction (total_distance : ℝ) (first_stop_fraction : ℝ) (final_leg : ℝ) :
  total_distance = 480 →
  first_stop_fraction = 1/2 →
  final_leg = 180 →
  (total_distance - first_stop_fraction * total_distance - final_leg) / (total_distance - first_stop_fraction * total_distance) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_maria_trip_fraction_l1803_180361


namespace NUMINAMATH_CALUDE_rectangle_area_l1803_180334

/-- Given a rectangle with length thrice its breadth and diagonal 26 meters,
    prove that its area is 202.8 square meters. -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let d := 26
  d^2 = l^2 + b^2 → b * l = 202.8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1803_180334


namespace NUMINAMATH_CALUDE_pizza_delivery_theorem_l1803_180325

/-- Represents a pizza delivery scenario -/
structure PizzaDelivery where
  total_pizzas : ℕ
  double_pizza_stops : ℕ
  single_pizza_stops : ℕ
  total_time : ℕ

/-- Calculates the average time per stop for a pizza delivery -/
def average_time_per_stop (pd : PizzaDelivery) : ℚ :=
  pd.total_time / (pd.double_pizza_stops + pd.single_pizza_stops)

/-- Theorem: Given the conditions, the average time per stop is 4 minutes -/
theorem pizza_delivery_theorem (pd : PizzaDelivery) 
  (h1 : pd.total_pizzas = 12)
  (h2 : pd.double_pizza_stops = 2)
  (h3 : pd.single_pizza_stops = pd.total_pizzas - 2 * pd.double_pizza_stops)
  (h4 : pd.total_time = 40) :
  average_time_per_stop pd = 4 := by
  sorry


end NUMINAMATH_CALUDE_pizza_delivery_theorem_l1803_180325


namespace NUMINAMATH_CALUDE_dans_pokemon_cards_l1803_180337

/-- The number of Pokemon cards Dan has -/
def dans_cards : ℕ := 41

/-- Sally's initial number of Pokemon cards -/
def sallys_initial_cards : ℕ := 27

/-- The number of Pokemon cards Sally bought -/
def cards_sally_bought : ℕ := 20

/-- The difference between Sally's and Dan's cards -/
def card_difference : ℕ := 6

theorem dans_pokemon_cards :
  sallys_initial_cards + cards_sally_bought = dans_cards + card_difference :=
sorry

end NUMINAMATH_CALUDE_dans_pokemon_cards_l1803_180337


namespace NUMINAMATH_CALUDE_arrasta_um_min_moves_l1803_180376

/-- Represents the Arrasta Um game board -/
structure ArrastaUmBoard (n : ℕ) where
  size : n ≥ 2

/-- Represents a move in the Arrasta Um game -/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Calculates the minimum number of moves required to complete the game -/
def minMoves (board : ArrastaUmBoard n) : ℕ :=
  6 * n - 8

/-- Theorem stating that the minimum number of moves to complete Arrasta Um on an n × n board is 6n - 8 -/
theorem arrasta_um_min_moves (n : ℕ) (board : ArrastaUmBoard n) :
  minMoves board = 6 * n - 8 :=
by sorry

end NUMINAMATH_CALUDE_arrasta_um_min_moves_l1803_180376


namespace NUMINAMATH_CALUDE_quadratic_polynomial_negative_root_l1803_180369

/-- A quadratic polynomial with two distinct real roots -/
structure QuadraticPolynomial where
  P : ℝ → ℝ
  is_quadratic : ∃ (a b c : ℝ), ∀ x, P x = a * x^2 + b * x + c
  has_distinct_roots : ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ P r₁ = 0 ∧ P r₂ = 0

/-- The inequality condition for the polynomial -/
def SatisfiesInequality (P : ℝ → ℝ) : Prop :=
  ∀ (a b : ℝ), (abs a ≥ 2017 ∧ abs b ≥ 2017) → P (a^2 + b^2) ≥ P (2*a*b)

/-- The main theorem -/
theorem quadratic_polynomial_negative_root (p : QuadraticPolynomial) 
    (h : SatisfiesInequality p.P) : 
    ∃ (x : ℝ), x < 0 ∧ p.P x = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_negative_root_l1803_180369


namespace NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l1803_180347

/-- Given a line l with equation 2x - y - 4 = 0, prove that the line with equation
    x + 2y - 2 = 0 is perpendicular to l and passes through the point where l
    intersects the x-axis. -/
theorem perpendicular_line_through_intersection (x y : ℝ) : 
  let l : ℝ → ℝ → Prop := λ x y ↦ 2 * x - y - 4 = 0
  let m : ℝ × ℝ := (2, 0)  -- Intersection point of l with x-axis
  let perp : ℝ → ℝ → Prop := λ x y ↦ x + 2 * y - 2 = 0
  (∀ x y, l x y → (x - m.1) * (x - m.1) + (y - m.2) * (y - m.2) ≠ 0 →
    (perp x y ↔ (x - m.1) * (2) + (y - m.2) * (-1) = 0)) ∧
  perp m.1 m.2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l1803_180347


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l1803_180300

theorem smaller_solution_of_quadratic (x : ℝ) :
  x^2 - 13*x + 36 = 0 →
  (∃ y : ℝ, y ≠ x ∧ y^2 - 13*y + 36 = 0) →
  (∀ z : ℝ, z^2 - 13*z + 36 = 0 → z ≥ 4) ∧
  (∃ w : ℝ, w^2 - 13*w + 36 = 0 ∧ w = 4) :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l1803_180300


namespace NUMINAMATH_CALUDE_decreasing_interval_l1803_180343

def f (x : ℝ) := x^2 - 6*x + 8

theorem decreasing_interval (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, ∀ y ∈ Set.Icc 1 a, x < y → f x > f y) ↔ 1 < a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_l1803_180343


namespace NUMINAMATH_CALUDE_polygon_sides_l1803_180317

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1803_180317


namespace NUMINAMATH_CALUDE_student_sister_weight_l1803_180370

theorem student_sister_weight (student_weight : ℝ) (weight_loss : ℝ) :
  student_weight = 90 ∧
  (student_weight - weight_loss) = 2 * ((student_weight - weight_loss) / 2) ∧
  weight_loss = 6 →
  student_weight + ((student_weight - weight_loss) / 2) = 132 :=
by sorry

end NUMINAMATH_CALUDE_student_sister_weight_l1803_180370


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l1803_180390

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry

-- Theorem statement
theorem line_perp_parallel_planes 
  (α β : Plane) (l : Line) 
  (h1 : α ≠ β) 
  (h2 : perpendicular l β) 
  (h3 : parallel α β) : 
  perpendicular l α :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l1803_180390


namespace NUMINAMATH_CALUDE_exam_score_per_correct_answer_l1803_180391

/-- Proves the number of marks scored for each correct answer in an exam -/
theorem exam_score_per_correct_answer 
  (total_questions : ℕ) 
  (total_marks : ℤ) 
  (correct_answers : ℕ) 
  (wrong_penalty : ℤ) 
  (h1 : total_questions = 60)
  (h2 : total_marks = 110)
  (h3 : correct_answers = 34)
  (h4 : wrong_penalty = -1)
  (h5 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (score_per_correct : ℤ), 
    score_per_correct * correct_answers + wrong_penalty * (total_questions - correct_answers) = total_marks ∧ 
    score_per_correct = 4 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_per_correct_answer_l1803_180391


namespace NUMINAMATH_CALUDE_purchase_price_problem_l1803_180307

/-- A linear function relating purchase quantity (y) to unit price (x) -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem purchase_price_problem (k b : ℝ) 
  (h1 : 1000 = linear_function k b 800)
  (h2 : 2000 = linear_function k b 700) :
  linear_function k b 5000 = 400 := by sorry

end NUMINAMATH_CALUDE_purchase_price_problem_l1803_180307


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l1803_180363

/-- Given that f(x) = x³(a⋅2^x - 2^(-x)) is an even function, prove that a = 1 --/
theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l1803_180363


namespace NUMINAMATH_CALUDE_train_length_calculation_l1803_180374

/-- Proves that given a train and platform of equal length, if the train crosses the platform
    in one minute at a speed of 216 km/hr, then the length of the train is 1800 meters. -/
theorem train_length_calculation (train_length platform_length : ℝ) 
    (speed : ℝ) (time : ℝ) :
  train_length = platform_length →
  speed = 216 →
  time = 1 / 60 →
  train_length = 1800 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1803_180374


namespace NUMINAMATH_CALUDE_radio_operator_distribution_probability_l1803_180329

theorem radio_operator_distribution_probability :
  let total_soldiers : ℕ := 12
  let radio_operators : ℕ := 3
  let group_sizes : List ℕ := [3, 4, 5]
  
  let total_distributions : ℕ := (total_soldiers.choose group_sizes[0]!) * ((total_soldiers - group_sizes[0]!).choose group_sizes[1]!) * 1
  
  let favorable_distributions : ℕ := ((total_soldiers - radio_operators).choose (group_sizes[0]! - 1)) *
    ((total_soldiers - radio_operators - (group_sizes[0]! - 1)).choose (group_sizes[1]! - 1)) * 
    ((radio_operators).factorial)
  
  (favorable_distributions : ℚ) / total_distributions = 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_radio_operator_distribution_probability_l1803_180329


namespace NUMINAMATH_CALUDE_crypto_puzzle_solution_l1803_180346

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem crypto_puzzle_solution :
  ∀ (A B C : ℕ),
    is_digit A →
    is_digit B →
    is_digit C →
    A + B + 1 = C + 10 →
    B = A + 2 →
    A ≠ B ∧ A ≠ C ∧ B ≠ C →
    C = 1 :=
by sorry

end NUMINAMATH_CALUDE_crypto_puzzle_solution_l1803_180346


namespace NUMINAMATH_CALUDE_cleaning_earnings_proof_l1803_180318

/-- Calculates the total earnings for cleaning all rooms in a building -/
def total_earnings (floors : ℕ) (rooms_per_floor : ℕ) (hours_per_room : ℕ) (dollars_per_hour : ℕ) : ℕ :=
  floors * rooms_per_floor * hours_per_room * dollars_per_hour

/-- Proves that the total earnings for cleaning the given building is $32,000 -/
theorem cleaning_earnings_proof :
  total_earnings 10 20 8 20 = 32000 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_earnings_proof_l1803_180318


namespace NUMINAMATH_CALUDE_power_difference_l1803_180367

theorem power_difference (a m n : ℝ) (hm : a^m = 12) (hn : a^n = 3) : a^(m-n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l1803_180367


namespace NUMINAMATH_CALUDE_marathon_volunteer_assignment_l1803_180396

def number_of_students : ℕ := 5
def number_of_tasks : ℕ := 4
def number_of_students_who_can_drive : ℕ := 3

theorem marathon_volunteer_assignment :
  let total_arrangements := 
    (Nat.choose number_of_students_who_can_drive 1 * 
     Nat.choose (number_of_students - 1) 2 * 
     Nat.factorial 3) +
    (Nat.choose number_of_students_who_can_drive 2 * 
     Nat.factorial 3)
  total_arrangements = 
    Nat.choose number_of_students_who_can_drive 1 * 
    Nat.choose number_of_students 2 * 
    Nat.factorial 3 +
    Nat.choose number_of_students_who_can_drive 2 * 
    Nat.factorial 3 := by
  sorry

end NUMINAMATH_CALUDE_marathon_volunteer_assignment_l1803_180396


namespace NUMINAMATH_CALUDE_rhombus_side_length_l1803_180356

/-- A rhombus with area K and one diagonal three times the length of the other has side length √(5K/3) -/
theorem rhombus_side_length (K : ℝ) (d₁ d₂ s : ℝ) (h₁ : K > 0) (h₂ : d₁ > 0) (h₃ : d₂ > 0) (h₄ : s > 0) :
  d₂ = 3 * d₁ →
  K = (1/2) * d₁ * d₂ →
  s^2 = (d₁/2)^2 + (d₂/2)^2 →
  s = Real.sqrt ((5 * K) / 3) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l1803_180356


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1803_180393

/-- Given a parabola and a hyperbola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (x y : ℝ) :
  -- Parabola equation
  (∃ (x₀ y₀ : ℝ), y₀^2 = 8*x₀) →
  -- Hyperbola general form
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2/a^2 - y^2/b^2 = 1) →
  -- Directrix of parabola (x = -2) passes through a focus of the hyperbola
  (∃ (x₁ y₁ : ℝ), x₁ = -2 ∧ x₁^2/a^2 - y₁^2/b^2 = 1) →
  -- Eccentricity of the hyperbola is 2
  (∃ (c : ℝ), c/a = 2 ∧ c^2 = a^2 + b^2) →
  -- Prove the equation of the hyperbola
  x^2 - y^2/3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1803_180393


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1803_180388

theorem x_minus_y_value (x y : ℝ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 36) : x - y = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1803_180388


namespace NUMINAMATH_CALUDE_chest_value_is_35000_l1803_180366

/-- Represents the pirate treasure distribution problem -/
structure PirateTreasure where
  total_pirates : ℕ
  total_chests : ℕ
  pirates_with_chests : ℕ
  contribution_per_chest : ℕ

/-- The specific instance of the pirate treasure problem -/
def pirate_problem : PirateTreasure := {
  total_pirates := 7
  total_chests := 5
  pirates_with_chests := 5
  contribution_per_chest := 10000
}

/-- Calculates the value of one chest based on the given problem parameters -/
def chest_value (p : PirateTreasure) : ℕ :=
  let total_contribution := p.pirates_with_chests * p.contribution_per_chest
  let pirates_without_chests := p.total_pirates - p.pirates_with_chests
  let compensation_per_pirate := total_contribution / pirates_without_chests
  p.total_pirates * compensation_per_pirate / p.total_chests

/-- Theorem stating that the chest value for the given problem is 35000 -/
theorem chest_value_is_35000 : chest_value pirate_problem = 35000 := by
  sorry

end NUMINAMATH_CALUDE_chest_value_is_35000_l1803_180366


namespace NUMINAMATH_CALUDE_second_platform_length_l1803_180397

/-- Calculates the length of a second platform given train and first platform details -/
theorem second_platform_length
  (train_length : ℝ)
  (first_platform_length : ℝ)
  (first_crossing_time : ℝ)
  (second_crossing_time : ℝ)
  (h1 : train_length = 310)
  (h2 : first_platform_length = 110)
  (h3 : first_crossing_time = 15)
  (h4 : second_crossing_time = 20) :
  (second_crossing_time * (train_length + first_platform_length) / first_crossing_time) - train_length = 250 :=
by sorry

end NUMINAMATH_CALUDE_second_platform_length_l1803_180397


namespace NUMINAMATH_CALUDE_two_self_inverse_matrices_l1803_180354

def is_self_inverse (a d : ℝ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, 3; -8, d]
  M * M = 1

theorem two_self_inverse_matrices :
  ∃! (n : ℕ), ∃ (S : Finset (ℝ × ℝ)),
    S.card = n ∧
    (∀ (p : ℝ × ℝ), p ∈ S ↔ is_self_inverse p.1 p.2) :=
  sorry

end NUMINAMATH_CALUDE_two_self_inverse_matrices_l1803_180354


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1803_180389

theorem cube_root_equation_solution :
  ∀ x : ℝ, (10 - 6 * x)^(1/3 : ℝ) = -2 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1803_180389


namespace NUMINAMATH_CALUDE_prism_pyramid_sum_l1803_180379

/-- Represents a shape formed by adding a pyramid to one rectangular face of a right rectangular prism -/
structure PrismPyramid where
  prism_faces : Nat
  prism_edges : Nat
  prism_vertices : Nat
  pyramid_new_faces : Nat
  pyramid_new_edges : Nat
  pyramid_new_vertex : Nat

/-- The total number of exterior faces, edges, and vertices of the combined shape -/
def total_elements (shape : PrismPyramid) : Nat :=
  (shape.prism_faces + shape.pyramid_new_faces - 1) +
  (shape.prism_edges + shape.pyramid_new_edges) +
  (shape.prism_vertices + shape.pyramid_new_vertex)

/-- Theorem stating that the sum of exterior faces, edges, and vertices is 34 -/
theorem prism_pyramid_sum :
  ∀ (shape : PrismPyramid),
    shape.prism_faces = 6 ∧
    shape.prism_edges = 12 ∧
    shape.prism_vertices = 8 ∧
    shape.pyramid_new_faces = 4 ∧
    shape.pyramid_new_edges = 4 ∧
    shape.pyramid_new_vertex = 1 →
    total_elements shape = 34 := by
  sorry

end NUMINAMATH_CALUDE_prism_pyramid_sum_l1803_180379


namespace NUMINAMATH_CALUDE_octagon_area_l1803_180319

/-- The area of an octagon formed by removing equilateral triangles from the corners of a square -/
theorem octagon_area (square_side : ℝ) (triangle_side : ℝ) : 
  square_side = 1 + Real.sqrt 3 →
  triangle_side = 1 →
  let square_area := square_side ^ 2
  let triangle_area := (Real.sqrt 3 / 4) * triangle_side ^ 2
  let octagon_area := square_area - 4 * triangle_area
  octagon_area = 4 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_octagon_area_l1803_180319


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1803_180328

-- Define the quadratic function f
def f : ℝ → ℝ := fun x ↦ 2 * x^2 - 4 * x + 3

-- State the theorem
theorem quadratic_function_properties :
  (∀ x : ℝ, f x ≥ 1) ∧  -- minimum value is 1
  (f 0 = 3) ∧ (f 2 = 3) ∧  -- f(0) = f(2) = 3
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-3) (-1) → f x > 2 * x + 2 * m + 1) → m < 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1803_180328


namespace NUMINAMATH_CALUDE_first_day_is_tuesday_l1803_180385

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Counts the occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: In a 31-day month with exactly 4 Fridays and 4 Mondays, the first day is Tuesday -/
theorem first_day_is_tuesday (m : Month) 
  (h1 : m.days = 31)
  (h2 : countDayOccurrences m DayOfWeek.Friday = 4)
  (h3 : countDayOccurrences m DayOfWeek.Monday = 4) :
  m.firstDay = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_first_day_is_tuesday_l1803_180385


namespace NUMINAMATH_CALUDE_arkos_population_2070_l1803_180315

def population_growth (initial_population : ℕ) (start_year end_year doubling_period : ℕ) : ℕ :=
  initial_population * (2 ^ ((end_year - start_year) / doubling_period))

theorem arkos_population_2070 :
  population_growth 150 1960 2070 20 = 4800 :=
by
  sorry

end NUMINAMATH_CALUDE_arkos_population_2070_l1803_180315


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l1803_180330

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 / 2 ∧ 
   r^2 + 10*r + k = 0 ∧ s^2 + 10*s + k = 0) → k = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l1803_180330


namespace NUMINAMATH_CALUDE_some_mythical_creatures_are_winged_animals_l1803_180358

-- Define the sets
variable (D : Type) -- Dragons
variable (M : Type) -- Mythical creatures
variable (W : Type) -- Winged animals

-- Define the relations
variable (isDragon : D → Prop)
variable (isMythical : M → Prop)
variable (isWinged : W → Prop)

-- Define the conditions
variable (h1 : ∀ d : D, ∃ m : M, isMythical m)
variable (h2 : ∃ w : W, ∃ d : D, isDragon d ∧ isWinged w)

-- Theorem to prove
theorem some_mythical_creatures_are_winged_animals :
  ∃ m : M, ∃ w : W, isMythical m ∧ isWinged w :=
sorry

end NUMINAMATH_CALUDE_some_mythical_creatures_are_winged_animals_l1803_180358


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l1803_180352

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 - 4*y^2 - 10*x + 20*y + 25 = 0

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
    ∀ x y, f x y ↔ ((x - h) / a)^2 - ((y - k) / b)^2 = 1

/-- Theorem: The given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l1803_180352


namespace NUMINAMATH_CALUDE_height_at_3_l1803_180350

/-- Represents the height of a tree over time -/
def TreeHeight : ℕ → ℝ
  | 0 => 1  -- Initial height (arbitrary, not given in the problem)
  | n + 1 => 
    if n < 4 then 3 * TreeHeight n  -- Triple height for first 4 years
    else 2 * TreeHeight n           -- Double height for next 3 years

/-- The height of the tree after 7 years is 648 feet -/
axiom height_at_7 : TreeHeight 7 = 648

/-- The height of the tree after 3 years is 27 feet -/
theorem height_at_3 : TreeHeight 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_height_at_3_l1803_180350


namespace NUMINAMATH_CALUDE_candy_mixture_proof_l1803_180394

/-- Proves that mixing 1 pound of candy A with 4 pounds of candy B
    results in 5 pounds of mixture costing $2 per pound -/
theorem candy_mixture_proof :
  let candy_a_price : ℝ := 3.20
  let candy_b_price : ℝ := 1.70
  let candy_a_amount : ℝ := 1
  let candy_b_amount : ℝ := 4
  let total_amount : ℝ := candy_a_amount + candy_b_amount
  let total_cost : ℝ := candy_a_price * candy_a_amount + candy_b_price * candy_b_amount
  let mixture_price_per_pound : ℝ := total_cost / total_amount
  total_amount = 5 ∧ mixture_price_per_pound = 2 := by
sorry

end NUMINAMATH_CALUDE_candy_mixture_proof_l1803_180394


namespace NUMINAMATH_CALUDE_complex_addition_l1803_180322

theorem complex_addition : ∃ z : ℂ, (5 - 3*I + z = -2 + 9*I) ∧ (z = -7 + 12*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_addition_l1803_180322


namespace NUMINAMATH_CALUDE_max_value_expression_l1803_180323

theorem max_value_expression (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  (∀ x y z w, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 0 ≤ z ∧ z ≤ 1 → 0 ≤ w ∧ w ≤ 1 → 
    x + y + z + w - x*y - y*z - z*w - w*x ≤ a + b + c + d - a*b - b*c - c*d - d*a) → 
  a + b + c + d - a*b - b*c - c*d - d*a = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1803_180323


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_l1803_180310

def initial_price : ℝ := 200
def first_year_increase : ℝ := 0.50
def second_year_decrease : ℝ := 0.30

theorem stock_price_after_two_years :
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price = 210 := by sorry

end NUMINAMATH_CALUDE_stock_price_after_two_years_l1803_180310


namespace NUMINAMATH_CALUDE_indistinguishable_balls_in_boxes_l1803_180372

/-- The number of partitions of n indistinguishable objects into k or fewer non-empty parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- The balls are indistinguishable -/
def balls : ℕ := 4

/-- The boxes are indistinguishable -/
def boxes : ℕ := 4

theorem indistinguishable_balls_in_boxes : partition_count balls boxes = 5 := by sorry

end NUMINAMATH_CALUDE_indistinguishable_balls_in_boxes_l1803_180372


namespace NUMINAMATH_CALUDE_hanks_reading_time_l1803_180339

/-- Represents Hank's weekly reading habits -/
structure ReadingHabits where
  weekdayMorningMinutes : ℕ
  weekdayEveningMinutes : ℕ
  weekdayDays : ℕ
  weekendMultiplier : ℕ

/-- Calculates the total reading time in minutes for a week -/
def totalReadingTime (habits : ReadingHabits) : ℕ :=
  let weekdayTotal := habits.weekdayDays * (habits.weekdayMorningMinutes + habits.weekdayEveningMinutes)
  let weekendDays := 7 - habits.weekdayDays
  let weekendTotal := weekendDays * habits.weekendMultiplier * (habits.weekdayMorningMinutes + habits.weekdayEveningMinutes)
  weekdayTotal + weekendTotal

/-- Theorem stating that Hank's total reading time in a week is 810 minutes -/
theorem hanks_reading_time :
  let hanksHabits : ReadingHabits := {
    weekdayMorningMinutes := 30,
    weekdayEveningMinutes := 60,
    weekdayDays := 5,
    weekendMultiplier := 2
  }
  totalReadingTime hanksHabits = 810 := by
  sorry


end NUMINAMATH_CALUDE_hanks_reading_time_l1803_180339


namespace NUMINAMATH_CALUDE_unique_triple_gcd_sum_square_l1803_180353

theorem unique_triple_gcd_sum_square : 
  ∃! (m n l : ℕ), 
    m + n = (Nat.gcd m n)^2 ∧
    m + l = (Nat.gcd m l)^2 ∧
    n + l = (Nat.gcd n l)^2 ∧
    m = 2 ∧ n = 2 ∧ l = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_gcd_sum_square_l1803_180353


namespace NUMINAMATH_CALUDE_flock_max_weight_l1803_180309

/-- Represents the types of swallows --/
inductive SwallowType
| American
| European

/-- Calculates the maximum weight a swallow can carry based on its type --/
def maxWeightCarried (s : SwallowType) : ℕ :=
  match s with
  | SwallowType.American => 5
  | SwallowType.European => 10

/-- The total number of swallows in the flock --/
def totalSwallows : ℕ := 90

/-- The ratio of American to European swallows --/
def americanToEuropeanRatio : ℕ := 2

/-- Theorem stating the maximum combined weight the flock can carry --/
theorem flock_max_weight :
  let europeanCount := totalSwallows / (americanToEuropeanRatio + 1)
  let americanCount := totalSwallows - europeanCount
  europeanCount * maxWeightCarried SwallowType.European +
  americanCount * maxWeightCarried SwallowType.American = 600 := by
  sorry


end NUMINAMATH_CALUDE_flock_max_weight_l1803_180309


namespace NUMINAMATH_CALUDE_permutation_inequality_solution_l1803_180311

def A (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

theorem permutation_inequality_solution :
  ∃! x : ℕ+, A 8 x < 6 * A 8 (x - 2) ∧ x = 8 := by sorry

end NUMINAMATH_CALUDE_permutation_inequality_solution_l1803_180311


namespace NUMINAMATH_CALUDE_square_circumcenter_segment_length_l1803_180395

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square with side length 1 -/
structure UnitSquare where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The circumcenter of a triangle -/
def circumcenter (A B C : Point) : Point :=
  sorry

/-- The length of a segment between two points -/
def segmentLength (P Q : Point) : ℝ :=
  sorry

/-- The main theorem -/
theorem square_circumcenter_segment_length 
  (ABCD : UnitSquare) 
  (P Q : Point) 
  (h1 : Q = circumcenter B P C) 
  (h2 : D = circumcenter P Q ABCD.A) : 
  segmentLength P Q = Real.sqrt (2 - Real.sqrt 3) ∨ 
  segmentLength P Q = Real.sqrt (2 + Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_square_circumcenter_segment_length_l1803_180395


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l1803_180340

/-- Given that Virginia, Adrienne, and Dennis have taught history for a combined total of 93 years,
    Virginia has taught for 9 more years than Adrienne, and Virginia has taught for 9 fewer years than Dennis,
    prove that Dennis has taught for 40 years. -/
theorem dennis_teaching_years (v a d : ℕ) : 
  v + a + d = 93 →
  v = a + 9 →
  d = v + 9 →
  d = 40 := by
sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l1803_180340


namespace NUMINAMATH_CALUDE_second_field_rows_l1803_180304

/-- Represents a corn field with a certain number of full rows -/
structure CornField where
  rows : ℕ

/-- Represents a farm with two corn fields -/
structure Farm where
  field1 : CornField
  field2 : CornField

def cobsPerRow : ℕ := 4

theorem second_field_rows (farm : Farm) 
  (h1 : farm.field1.rows = 13) 
  (h2 : farm.field1.rows * cobsPerRow + farm.field2.rows * cobsPerRow = 116) : 
  farm.field2.rows = 16 := by
  sorry

end NUMINAMATH_CALUDE_second_field_rows_l1803_180304


namespace NUMINAMATH_CALUDE_parabola_properties_l1803_180326

def is_valid_parabola (a b c : ℝ) : Prop :=
  a ≠ 0 ∧
  a * (-1)^2 + b * (-1) + c = -1 ∧
  c = 1 ∧
  a * (-2)^2 + b * (-2) + c > 1

theorem parabola_properties (a b c : ℝ) 
  (h : is_valid_parabola a b c) : 
  a * b * c > 0 ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c - 3 = 0 ∧ a * x₂^2 + b * x₂ + c - 3 = 0) ∧
  a + b + c > 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1803_180326


namespace NUMINAMATH_CALUDE_festival_average_surfers_l1803_180338

/-- The average number of surfers at the Rip Curl Myrtle Beach Surf Festival -/
def average_surfers (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℚ :=
  (day1 + day2 + day3) / 3

/-- Theorem: The average number of surfers at the Festival for three days is 1400 -/
theorem festival_average_surfers :
  let day1 : ℕ := 1500
  let day2 : ℕ := day1 + 600
  let day3 : ℕ := day1 * 2 / 5
  average_surfers day1 day2 day3 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_festival_average_surfers_l1803_180338


namespace NUMINAMATH_CALUDE_correct_classification_l1803_180378

-- Define the set of statement numbers
def StatementNumbers : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define the function that classifies numbers as precise or approximate
def classify : Nat → Bool
| 1 => true  -- Xiao Ming's books (precise)
| 2 => true  -- War cost (precise)
| 3 => true  -- DVD sales (precise)
| 4 => false -- Brain cells (approximate)
| 5 => true  -- Xiao Hong's score (precise)
| 6 => false -- Coal reserves (approximate)
| _ => false -- For completeness

-- Theorem statement
theorem correct_classification :
  {n ∈ StatementNumbers | classify n = true} = {1, 2, 3, 5} ∧
  {n ∈ StatementNumbers | classify n = false} = {4, 6} := by
  sorry


end NUMINAMATH_CALUDE_correct_classification_l1803_180378


namespace NUMINAMATH_CALUDE_solve_for_B_l1803_180336

theorem solve_for_B : ∀ (A B : ℕ), 
  (A ≥ 1 ∧ A ≤ 9) →  -- Ensure A is a single digit
  (B ≥ 0 ∧ B ≤ 9) →  -- Ensure B is a single digit
  632 - (100 * A + 10 * B + 1) = 41 → 
  B = 9 := by
sorry

end NUMINAMATH_CALUDE_solve_for_B_l1803_180336


namespace NUMINAMATH_CALUDE_smallest_b_is_correct_l1803_180332

/-- A function that checks if a number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = n

/-- The smallest integer b > 5 for which 43_b is a perfect cube -/
def smallest_b : ℕ := 6

theorem smallest_b_is_correct :
  (smallest_b > 5) ∧ 
  (is_perfect_cube (4 * smallest_b + 3)) ∧ 
  (∀ b : ℕ, b > 5 ∧ b < smallest_b → ¬(is_perfect_cube (4 * b + 3))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_is_correct_l1803_180332


namespace NUMINAMATH_CALUDE_guards_per_team_is_five_l1803_180351

/-- The number of forwards in the league -/
def num_forwards : ℕ := 32

/-- The number of guards in the league -/
def num_guards : ℕ := 80

/-- The number of guards per team when creating the maximum number of teams -/
def guards_per_team : ℕ := num_guards / Nat.gcd num_forwards num_guards

theorem guards_per_team_is_five : guards_per_team = 5 := by
  sorry

end NUMINAMATH_CALUDE_guards_per_team_is_five_l1803_180351


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1803_180333

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1803_180333


namespace NUMINAMATH_CALUDE_lorelai_jellybeans_l1803_180348

/-- The number of jellybeans each person has -/
structure JellyBeans where
  gigi : ℕ
  rory : ℕ
  luke : ℕ
  lane : ℕ
  lorelai : ℕ

/-- The conditions of the jellybean distribution -/
def jellybean_conditions (jb : JellyBeans) : Prop :=
  jb.gigi = 15 ∧
  jb.rory = jb.gigi + 30 ∧
  jb.luke = 2 * jb.rory ∧
  jb.lane = jb.gigi + 10 ∧
  jb.lorelai = 3 * (jb.gigi + jb.luke + jb.lane)

/-- Theorem stating that Lorelai has eaten 390 jellybeans -/
theorem lorelai_jellybeans (jb : JellyBeans) 
  (h : jellybean_conditions jb) : jb.lorelai = 390 := by
  sorry

end NUMINAMATH_CALUDE_lorelai_jellybeans_l1803_180348


namespace NUMINAMATH_CALUDE_total_birds_after_breeding_l1803_180381

/-- Represents the types of birds on the farm -/
inductive BirdType
  | Hen
  | Duck
  | Goose
  | Pigeon

/-- Represents the count and breeding information for each bird type -/
structure BirdInfo where
  count : ℕ
  maleRatio : ℚ
  femaleRatio : ℚ
  offspringPerFemale : ℕ
  breedingSuccessRate : ℚ

/-- Calculates the total number of birds after the breeding season -/
def totalBirdsAfterBreeding (birdCounts : BirdType → BirdInfo) (pigeonHatchRate : ℚ) : ℕ :=
  sorry

/-- The main theorem stating the total number of birds after breeding -/
theorem total_birds_after_breeding :
  let birdCounts : BirdType → BirdInfo
    | BirdType.Hen => ⟨40, 2/9, 7/9, 7, 85/100⟩
    | BirdType.Duck => ⟨20, 1/4, 3/4, 9, 75/100⟩
    | BirdType.Goose => ⟨10, 3/11, 8/11, 5, 90/100⟩
    | BirdType.Pigeon => ⟨30, 1/2, 1/2, 2, 80/100⟩
  totalBirdsAfterBreeding birdCounts (80/100) = 442 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_after_breeding_l1803_180381


namespace NUMINAMATH_CALUDE_contractor_payment_proof_l1803_180382

/-- Calculates the total amount received by a contractor given the contract terms and absence information. -/
def contractor_payment (total_days : ℕ) (payment_per_day : ℚ) (fine_per_day : ℚ) (absent_days : ℕ) : ℚ :=
  (total_days - absent_days : ℕ) * payment_per_day - absent_days * fine_per_day

/-- Proves that the contractor receives Rs. 555 given the specified conditions. -/
theorem contractor_payment_proof :
  contractor_payment 30 25 (15/2) 6 = 555 := by
  sorry

end NUMINAMATH_CALUDE_contractor_payment_proof_l1803_180382


namespace NUMINAMATH_CALUDE_max_pairs_sum_l1803_180384

theorem max_pairs_sum (n : ℕ) (h : n = 2023) :
  ∃ (k : ℕ) (pairs : List (ℕ × ℕ)),
    k = 813 ∧
    pairs.length = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 2033) ∧
    (∀ (m : ℕ) (pairs' : List (ℕ × ℕ)),
      m > k →
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) →
      (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
      (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 + p.2 ≤ 2033) →
      False) :=
by
  sorry

end NUMINAMATH_CALUDE_max_pairs_sum_l1803_180384


namespace NUMINAMATH_CALUDE_divisible_by_three_after_rotation_l1803_180355

theorem divisible_by_three_after_rotation (n : ℕ) : 
  n = 857142 → 
  (n % 3 = 0) ∧ 
  ((285714 : ℕ) % 3 = 0) := by
sorry

end NUMINAMATH_CALUDE_divisible_by_three_after_rotation_l1803_180355


namespace NUMINAMATH_CALUDE_square_minus_product_equals_one_l1803_180368

theorem square_minus_product_equals_one : 2002^2 - 2001 * 2003 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_equals_one_l1803_180368


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1803_180359

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := Real.sqrt 3 * x = 2 * y ∨ Real.sqrt 3 * x = -2 * y

-- Theorem statement
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola x y → asymptotes x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1803_180359


namespace NUMINAMATH_CALUDE_initial_children_on_bus_proof_initial_children_l1803_180321

theorem initial_children_on_bus : ℕ → Prop :=
  fun initial_children =>
    initial_children + 7 = 25

theorem proof_initial_children : 
  ∃ initial_children : ℕ, initial_children_on_bus initial_children ∧ initial_children = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_children_on_bus_proof_initial_children_l1803_180321


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1803_180371

theorem complex_equation_solution :
  ∀ z : ℂ, z * (1 - Complex.I) = (1 + Complex.I)^3 → z = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1803_180371


namespace NUMINAMATH_CALUDE_S_properties_l1803_180364

def S : Set ℤ := {x | ∃ n : ℤ, x = (n-1)^2 + n^2 + (n+1)^2}

theorem S_properties : 
  (∀ x ∈ S, ¬(3 ∣ x)) ∧ (∃ x ∈ S, 11 ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_S_properties_l1803_180364


namespace NUMINAMATH_CALUDE_no_matrix_sine_exists_l1803_180335

open Matrix

/-- Definition of matrix sine function -/
noncomputable def matrixSine (A : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ∑' n, ((-1)^n / (2*n+1).factorial : ℝ) • (A^(2*n+1))

/-- The statement to be proved -/
theorem no_matrix_sine_exists : 
  ¬ ∃ A : Matrix (Fin 2) (Fin 2) ℝ, matrixSine A = ![![1, 1996], ![0, 1]] :=
sorry

end NUMINAMATH_CALUDE_no_matrix_sine_exists_l1803_180335


namespace NUMINAMATH_CALUDE_f_divisible_by_13_l1803_180386

def f : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | (n + 2) => (4 * (n + 2) * f (n + 1) - 16 * (n + 1) * f n + n^2 * n^2) / n

theorem f_divisible_by_13 :
  13 ∣ f 1989 ∧ 13 ∣ f 1990 ∧ 13 ∣ f 1991 := by sorry

end NUMINAMATH_CALUDE_f_divisible_by_13_l1803_180386


namespace NUMINAMATH_CALUDE_inverse_sum_theorem_l1803_180324

-- Define the function g(x) = x³
def g (x : ℝ) : ℝ := x^3

-- State the theorem
theorem inverse_sum_theorem : g⁻¹ 8 + g⁻¹ (-64) = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_theorem_l1803_180324


namespace NUMINAMATH_CALUDE_sharon_coffee_cost_l1803_180302

/-- Calculates the total cost of coffee pods for Sharon's vacation -/
def coffee_cost (vacation_days : ℕ) (light_daily : ℕ) (medium_daily : ℕ) (decaf_daily : ℕ)
  (light_box_qty : ℕ) (medium_box_qty : ℕ) (decaf_box_qty : ℕ)
  (light_box_price : ℕ) (medium_box_price : ℕ) (decaf_box_price : ℕ) : ℕ :=
  let light_pods := vacation_days * light_daily
  let medium_pods := vacation_days * medium_daily
  let decaf_pods := vacation_days * decaf_daily
  let light_boxes := (light_pods + light_box_qty - 1) / light_box_qty
  let medium_boxes := (medium_pods + medium_box_qty - 1) / medium_box_qty
  let decaf_boxes := (decaf_pods + decaf_box_qty - 1) / decaf_box_qty
  light_boxes * light_box_price + medium_boxes * medium_box_price + decaf_boxes * decaf_box_price

/-- Theorem stating that the total cost for Sharon's vacation coffee is $80 -/
theorem sharon_coffee_cost :
  coffee_cost 40 2 1 1 20 25 30 10 12 8 = 80 :=
by sorry


end NUMINAMATH_CALUDE_sharon_coffee_cost_l1803_180302


namespace NUMINAMATH_CALUDE_cos_180_degrees_l1803_180399

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l1803_180399


namespace NUMINAMATH_CALUDE_unique_a_value_l1803_180313

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem unique_a_value (a : ℝ) (h : 1 ∈ A a) : a = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1803_180313


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l1803_180331

/-- The number of possible starting lineups for a basketball team --/
theorem basketball_lineup_combinations (total_players : ℕ) 
  (guaranteed_players : ℕ) (excluded_players : ℕ) (lineup_size : ℕ) : 
  total_players = 15 → 
  guaranteed_players = 2 → 
  excluded_players = 1 → 
  lineup_size = 6 → 
  Nat.choose (total_players - guaranteed_players - excluded_players) 
             (lineup_size - guaranteed_players) = 495 := by
  sorry

#check basketball_lineup_combinations

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l1803_180331


namespace NUMINAMATH_CALUDE_beth_wins_743_l1803_180306

/-- Represents a configuration of brick walls -/
def Configuration := List Nat

/-- Calculates the nim-value of a single wall -/
noncomputable def nimValue (n : Nat) : Nat :=
  sorry

/-- Calculates the nim-sum of a list of nim-values -/
def nimSum (values : List Nat) : Nat :=
  sorry

/-- Determines if a configuration is a winning position for the current player -/
def isWinningPosition (config : Configuration) : Prop :=
  nimSum (config.map nimValue) ≠ 0

/-- The game of brick removal -/
theorem beth_wins_743 (config : Configuration) :
  config = [7, 4, 4] → ¬isWinningPosition config :=
  sorry

end NUMINAMATH_CALUDE_beth_wins_743_l1803_180306


namespace NUMINAMATH_CALUDE_alex_score_l1803_180345

-- Define the total number of shots
def total_shots : ℕ := 40

-- Define the success rates
def three_point_success_rate : ℚ := 1/4
def two_point_success_rate : ℚ := 1/5

-- Define the point values
def three_point_value : ℕ := 3
def two_point_value : ℕ := 2

-- Theorem statement
theorem alex_score :
  ∀ x y : ℕ,
  x + y = total_shots →
  (x : ℚ) * three_point_success_rate * three_point_value +
  (y : ℚ) * two_point_success_rate * two_point_value = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_alex_score_l1803_180345


namespace NUMINAMATH_CALUDE_balance_weights_theorem_l1803_180308

/-- The double factorial of an odd number -/
def oddDoubleFactorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => (k + 1) * oddDoubleFactorial k

/-- The number of ways to place weights on a balance -/
def balanceWeights (n : ℕ) : ℕ :=
  oddDoubleFactorial (2 * n - 1)

/-- Theorem: The number of ways to place n weights (2^0, 2^1, ..., 2^(n-1)) on a balance,
    such that the right pan is never heavier than the left pan, is equal to (2n-1)!! -/
theorem balance_weights_theorem (n : ℕ) (h : n > 0) :
  balanceWeights n = oddDoubleFactorial (2 * n - 1) :=
by
  sorry

#eval balanceWeights 3  -- Expected output: 15
#eval balanceWeights 4  -- Expected output: 105

end NUMINAMATH_CALUDE_balance_weights_theorem_l1803_180308


namespace NUMINAMATH_CALUDE_parallel_lines_count_parallel_lines_problem_l1803_180320

/-- Given two sets of intersecting parallel lines, the number of parallelograms formed is the product of the spaces between the lines in each set. -/
def parallelogram_count (lines_set1 lines_set2 : ℕ) : ℕ := (lines_set1 - 1) * (lines_set2 - 1)

/-- The problem statement -/
theorem parallel_lines_count (lines_set1 : ℕ) (parallelograms : ℕ) : ℕ :=
  let lines_set2 := (parallelograms / (lines_set1 - 1)) + 1
  lines_set2

/-- The main theorem to prove -/
theorem parallel_lines_problem :
  parallel_lines_count 6 420 = 85 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_count_parallel_lines_problem_l1803_180320


namespace NUMINAMATH_CALUDE_loan_amount_to_B_l1803_180303

/-- Calculates the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem loan_amount_to_B (amountToC : ℚ) (timeB timeC : ℚ) (rate : ℚ) (totalInterest : ℚ) :
  amountToC = 3000 →
  timeB = 2 →
  timeC = 4 →
  rate = 8 →
  totalInterest = 1760 →
  ∃ amountToB : ℚ, 
    simpleInterest amountToB rate timeB + simpleInterest amountToC rate timeC = totalInterest ∧
    amountToB = 5000 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_to_B_l1803_180303


namespace NUMINAMATH_CALUDE_parallelogram_area_l1803_180383

/-- The area of a parallelogram with base 48 cm and height 36 cm is 1728 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 48 → 
  height = 36 → 
  area = base * height → 
  area = 1728 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1803_180383


namespace NUMINAMATH_CALUDE_book_costs_proof_l1803_180373

theorem book_costs_proof (total_cost : ℝ) (book1 book2 book3 book4 book5 : ℝ) :
  total_cost = 24 ∧
  book1 = book2 + 2 ∧
  book3 = book1 + 4 ∧
  book4 = book3 - 3 ∧
  book5 = book2 ∧
  book1 ≠ book2 ∧ book1 ≠ book3 ∧ book1 ≠ book4 ∧ book1 ≠ book5 ∧
  book2 ≠ book3 ∧ book2 ≠ book4 ∧
  book3 ≠ book4 ∧ book3 ≠ book5 ∧
  book4 ≠ book5 →
  book1 = 4.6 ∧ book2 = 2.6 ∧ book3 = 8.6 ∧ book4 = 5.6 ∧ book5 = 2.6 ∧
  total_cost = book1 + book2 + book3 + book4 + book5 := by
sorry

end NUMINAMATH_CALUDE_book_costs_proof_l1803_180373


namespace NUMINAMATH_CALUDE_scientific_notation_657000_l1803_180380

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_657000 :
  toScientificNotation 657000 = ScientificNotation.mk 6.57 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_657000_l1803_180380


namespace NUMINAMATH_CALUDE_body_lotion_cost_is_60_l1803_180398

/-- Represents the cost of items and total spent at Target --/
structure TargetPurchase where
  tanya_face_moisturizer_cost : ℕ
  tanya_face_moisturizer_count : ℕ
  tanya_body_lotion_count : ℕ
  total_spent : ℕ

/-- Calculates the cost of each body lotion based on the given conditions --/
def body_lotion_cost (p : TargetPurchase) : ℕ :=
  let tanya_total := p.tanya_face_moisturizer_cost * p.tanya_face_moisturizer_count + 
                     p.tanya_body_lotion_count * (p.total_spent / 3)
  (p.total_spent / 3 - p.tanya_face_moisturizer_cost * p.tanya_face_moisturizer_count) / p.tanya_body_lotion_count

/-- Theorem stating that the cost of each body lotion is $60 --/
theorem body_lotion_cost_is_60 (p : TargetPurchase) 
  (h1 : p.tanya_face_moisturizer_cost = 50)
  (h2 : p.tanya_face_moisturizer_count = 2)
  (h3 : p.tanya_body_lotion_count = 4)
  (h4 : p.total_spent = 1020) :
  body_lotion_cost p = 60 := by
  sorry


end NUMINAMATH_CALUDE_body_lotion_cost_is_60_l1803_180398


namespace NUMINAMATH_CALUDE_complex_number_properties_l1803_180327

theorem complex_number_properties (z : ℂ) (h : (z - 2*I) / z = 2 + I) : 
  z.im = -1 ∧ Complex.abs z = Real.sqrt 2 ∧ z^6 = -8*I := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l1803_180327


namespace NUMINAMATH_CALUDE_prob_different_suits_is_78_103_l1803_180387

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)

/-- Represents two mixed standard 52-card decks -/
def MixedDecks := Deck × Deck

/-- The probability of picking two different cards of different suits from mixed decks -/
def prob_different_suits (decks : MixedDecks) : ℚ :=
  78 / 103

/-- Theorem stating the probability of picking two different cards of different suits -/
theorem prob_different_suits_is_78_103 (decks : MixedDecks) :
  prob_different_suits decks = 78 / 103 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_suits_is_78_103_l1803_180387


namespace NUMINAMATH_CALUDE_power_function_comparison_l1803_180377

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- State the theorem
theorem power_function_comparison
  (f : ℝ → ℝ)
  (h_power : isPowerFunction f)
  (h_condition : f 8 = 4) :
  f (Real.sqrt 2 / 2) > f (-Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_power_function_comparison_l1803_180377


namespace NUMINAMATH_CALUDE_player_b_wins_in_five_l1803_180360

/-- The probability that Player B wins a best-of-five series in exactly 5 matches,
    given that Player A wins each match with probability 3/4 -/
theorem player_b_wins_in_five (p : ℚ) (h : p = 3/4) :
  let q := 1 - p
  let prob_tied_after_four := 6 * q^2 * p^2
  let prob_b_wins_fifth := q
  prob_tied_after_four * prob_b_wins_fifth = 27/512 :=
by sorry

end NUMINAMATH_CALUDE_player_b_wins_in_five_l1803_180360


namespace NUMINAMATH_CALUDE_hyperbola_condition_l1803_180342

-- Define the equation
def equation (x y k : ℝ) : Prop :=
  x^2 / (k + 1) + y^2 / (k - 5) = 1

-- Define what it means for the equation to represent a hyperbola
def represents_hyperbola (k : ℝ) : Prop :=
  (k + 1 > 0 ∧ k - 5 < 0) ∨ (k + 1 < 0 ∧ k - 5 > 0)

-- State the theorem
theorem hyperbola_condition (k : ℝ) :
  (∀ x y, equation x y k ↔ represents_hyperbola k) ↔ k ∈ Set.Ioo (-1 : ℝ) 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l1803_180342


namespace NUMINAMATH_CALUDE_bisector_line_l1803_180301

/-- Given two lines l₁ and l₂, and a point P, this theorem states that
    the line passing through P and (4, 0) bisects the line segment formed by
    its intersections with l₁ and l₂. -/
theorem bisector_line (P : ℝ × ℝ) (l₁ l₂ : Set (ℝ × ℝ)) :
  P = (0, 1) →
  l₁ = {(x, y) | 2*x + y - 8 = 0} →
  l₂ = {(x, y) | x - 3*y + 10 = 0} →
  ∃ (A B : ℝ × ℝ),
    A ∈ l₁ ∧
    B ∈ l₂ ∧
    (∃ (t : ℝ), A = (1-t) • P + t • (4, 0) ∧ B = (1-t) • (4, 0) + t • P) :=
by sorry

end NUMINAMATH_CALUDE_bisector_line_l1803_180301


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l1803_180316

theorem unique_prime_with_prime_sums : ∃! p : ℕ, 
  Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l1803_180316


namespace NUMINAMATH_CALUDE_circular_table_dice_probability_l1803_180375

def num_people : ℕ := 5
def die_sides : ℕ := 6

def probability_no_adjacent_same : ℚ :=
  375 / 2592

theorem circular_table_dice_probability :
  let total_outcomes := die_sides ^ num_people
  let favorable_outcomes := 
    (die_sides * (die_sides - 1)^(num_people - 1) * (die_sides - 2)) +
    (die_sides * (die_sides - 1)^(num_people - 1) * (die_sides - 1) / die_sides)
  favorable_outcomes / total_outcomes = probability_no_adjacent_same := by
  sorry

end NUMINAMATH_CALUDE_circular_table_dice_probability_l1803_180375


namespace NUMINAMATH_CALUDE_pants_price_is_6_l1803_180362

-- Define variables for pants and shirt prices
variable (pants_price : ℝ)
variable (shirt_price : ℝ)

-- Define Peter's purchase
def peter_total : ℝ := 2 * pants_price + 5 * shirt_price

-- Define Jessica's purchase
def jessica_total : ℝ := 2 * shirt_price

-- Theorem stating the price of one pair of pants
theorem pants_price_is_6 
  (h1 : peter_total = 62)
  (h2 : jessica_total = 20) :
  pants_price = 6 := by
sorry

end NUMINAMATH_CALUDE_pants_price_is_6_l1803_180362


namespace NUMINAMATH_CALUDE_derivative_of_f_composite_l1803_180349

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem derivative_of_f_composite (a b : ℝ) :
  deriv (fun x => f (a - b*x)) = fun x => -3*b*(a - b*x)^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_composite_l1803_180349


namespace NUMINAMATH_CALUDE_sqrt_three_squared_four_to_fourth_l1803_180312

theorem sqrt_three_squared_four_to_fourth : Real.sqrt (3^2 * 4^4) = 48 := by sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_four_to_fourth_l1803_180312


namespace NUMINAMATH_CALUDE_minimum_framing_feet_l1803_180341

-- Define the original picture dimensions
def original_width : ℕ := 5
def original_height : ℕ := 7

-- Define the enlargement factor
def enlargement_factor : ℕ := 2

-- Define the border width
def border_width : ℕ := 3

-- Define the number of inches in a foot
def inches_per_foot : ℕ := 12

-- Theorem statement
theorem minimum_framing_feet :
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (total_width + total_height)
  (perimeter + inches_per_foot - 1) / inches_per_foot = 6 := by
  sorry

end NUMINAMATH_CALUDE_minimum_framing_feet_l1803_180341
