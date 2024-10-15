import Mathlib

namespace NUMINAMATH_CALUDE_evaluate_expression_l438_43848

theorem evaluate_expression (a b c : ℚ) (ha : a = 1/2) (hb : b = 3/4) (hc : c = 8) :
  a^3 * b^2 * c = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l438_43848


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l438_43899

theorem quadratic_roots_expression (m n : ℝ) : 
  m ^ 2 + 2015 * m - 1 = 0 ∧ n ^ 2 + 2015 * n - 1 = 0 → m ^ 2 * n + m * n ^ 2 - m * n = 2016 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l438_43899


namespace NUMINAMATH_CALUDE_platinum_to_gold_ratio_is_two_to_one_l438_43874

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards -/
structure SallysCards where
  gold : CreditCard
  platinum : CreditCard

theorem platinum_to_gold_ratio_is_two_to_one 
  (cards : SallysCards)
  (h1 : cards.gold.balance = cards.gold.limit / 3)
  (h2 : cards.platinum.balance = cards.platinum.limit / 6)
  (h3 : cards.platinum.balance + cards.gold.balance = cards.platinum.limit / 3) :
  cards.platinum.limit / cards.gold.limit = 2 := by
  sorry

end NUMINAMATH_CALUDE_platinum_to_gold_ratio_is_two_to_one_l438_43874


namespace NUMINAMATH_CALUDE_large_cube_surface_area_l438_43867

theorem large_cube_surface_area (small_cube_volume : ℝ) (num_small_cubes : ℕ) :
  small_cube_volume = 512 →
  num_small_cubes = 8 →
  let small_cube_side := small_cube_volume ^ (1/3)
  let large_cube_side := 2 * small_cube_side
  let large_cube_surface_area := 6 * large_cube_side^2
  large_cube_surface_area = 1536 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_surface_area_l438_43867


namespace NUMINAMATH_CALUDE_count_integer_solutions_l438_43895

theorem count_integer_solutions : ∃! A : ℕ, 
  A = (Finset.filter (fun p : ℕ × ℕ => 
    p.1 + p.2 ≥ A ∧ 
    p.1 ≤ 6 ∧ 
    p.2 ≤ 7
  ) (Finset.product (Finset.range 7) (Finset.range 8))).card ∧
  A = 10 := by
  sorry

end NUMINAMATH_CALUDE_count_integer_solutions_l438_43895


namespace NUMINAMATH_CALUDE_problem_proof_l438_43829

theorem problem_proof : 2^0 - |(-3)| + (-1/2) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l438_43829


namespace NUMINAMATH_CALUDE_circle_through_points_l438_43889

-- Define the points
def O : ℝ × ℝ := (0, 0)
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (4, 2)

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Theorem statement
theorem circle_through_points : 
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (4, -3) ∧ 
    radius = 5 ∧
    O ∈ Circle center radius ∧
    M1 ∈ Circle center radius ∧
    M2 ∈ Circle center radius ∧
    Circle center radius = {p : ℝ × ℝ | (p.1 - 4)^2 + (p.2 + 3)^2 = 25} := by
  sorry

end NUMINAMATH_CALUDE_circle_through_points_l438_43889


namespace NUMINAMATH_CALUDE_complement_of_intersection_l438_43842

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {1,3,5}
def B : Set Nat := {1,2}

theorem complement_of_intersection (U A B : Set Nat) :
  U = {1,2,3,4,5,6} →
  A = {1,3,5} →
  B = {1,2} →
  (U \ (A ∩ B)) = {2,3,4,5,6} :=
by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l438_43842


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l438_43875

/-- The minimum squared distance between a point on y = x^2 + 3ln(x) and a point on y = x + 2 -/
theorem min_distance_between_curves : ∀ (a b c d : ℝ),
  b = a^2 + 3 * Real.log a →  -- P(a,b) is on y = x^2 + 3ln(x)
  d = c + 2 →                 -- Q(c,d) is on y = x + 2
  (∀ x y z w : ℝ, 
    y = x^2 + 3 * Real.log x → 
    w = z + 2 → 
    (a - c)^2 + (b - d)^2 ≤ (x - z)^2 + (y - w)^2) →
  (a - c)^2 + (b - d)^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l438_43875


namespace NUMINAMATH_CALUDE_three_numbers_sum_l438_43800

theorem three_numbers_sum (A B C : ℤ) : 
  A + B + C = 180 ∧ B = 3*C - 2 ∧ A = 2*C + 8 → A = 66 ∧ B = 85 ∧ C = 29 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l438_43800


namespace NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l438_43815

theorem roots_of_cubic_polynomial :
  let f : ℝ → ℝ := λ x => x^3 - 2*x^2 - 5*x + 6
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3) := by sorry

end NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l438_43815


namespace NUMINAMATH_CALUDE_lewis_items_found_l438_43859

theorem lewis_items_found (tanya_items samantha_items lewis_items : ℕ) : 
  tanya_items = 4 →
  samantha_items = 4 * tanya_items →
  lewis_items = samantha_items + 4 →
  lewis_items = 20 := by
  sorry

end NUMINAMATH_CALUDE_lewis_items_found_l438_43859


namespace NUMINAMATH_CALUDE_gcd_459_357_l438_43879

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l438_43879


namespace NUMINAMATH_CALUDE_winning_strategy_iff_not_div_four_l438_43830

/-- A game where two players take turns removing stones from a pile. -/
structure StoneGame where
  n : ℕ  -- Initial number of stones

/-- Represents a valid move in the game -/
inductive ValidMove : ℕ → ℕ → Prop where
  | prime_divisor {n m : ℕ} (h : m.Prime) (d : m ∣ n) : ValidMove n m
  | one {n : ℕ} : ValidMove n 1

/-- Defines a winning strategy for the first player -/
def has_winning_strategy (game : StoneGame) : Prop :=
  ∃ (strategy : ℕ → ℕ),
    ∀ (opponent_move : ℕ → ℕ),
      ValidMove game.n (strategy game.n) ∧
      (∀ k, k < game.n →
        ValidMove k (opponent_move k) →
          ValidMove (k - opponent_move k) (strategy (k - opponent_move k)))

/-- The main theorem: The first player has a winning strategy iff n is not divisible by 4 -/
theorem winning_strategy_iff_not_div_four (game : StoneGame) :
  has_winning_strategy game ↔ ¬(4 ∣ game.n) :=
sorry

end NUMINAMATH_CALUDE_winning_strategy_iff_not_div_four_l438_43830


namespace NUMINAMATH_CALUDE_first_discount_percentage_l438_43861

theorem first_discount_percentage (original_price : ℝ) (second_discount : ℝ) (final_price : ℝ) :
  original_price = 175 →
  second_discount = 5 →
  final_price = 133 →
  ∃ (first_discount : ℝ),
    first_discount = 20 ∧
    final_price = original_price * (100 - first_discount) / 100 * (100 - second_discount) / 100 :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l438_43861


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l438_43828

theorem equal_roots_quadratic (m : ℝ) :
  (∃ x : ℝ, 4 * x^2 - 6 * x + m = 0 ∧
   ∀ y : ℝ, 4 * y^2 - 6 * y + m = 0 → y = x) →
  m = 9/4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l438_43828


namespace NUMINAMATH_CALUDE_find_C_value_l438_43881

-- Define the structure of the 8-digit numbers
def FirstNumber (A B : ℕ) : ℕ := 85000000 + A * 100000 + 73000 + B * 100 + 20
def SecondNumber (A B C : ℕ) : ℕ := 41000000 + 700000 + A * 10000 + B * 1000 + 500 + C * 10 + 9

-- Define the condition for being a multiple of 5
def IsMultipleOf5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

-- State the theorem
theorem find_C_value (A B : ℕ) (h1 : IsMultipleOf5 (FirstNumber A B)) 
  (h2 : ∃ C : ℕ, IsMultipleOf5 (SecondNumber A B C)) : 
  ∃ C : ℕ, C = 1 ∧ IsMultipleOf5 (SecondNumber A B C) :=
sorry

end NUMINAMATH_CALUDE_find_C_value_l438_43881


namespace NUMINAMATH_CALUDE_coin_stack_count_l438_43898

/-- Thickness of a 2p coin in millimeters -/
def thickness_2p : ℚ := 205/100

/-- Thickness of a 10p coin in millimeters -/
def thickness_10p : ℚ := 195/100

/-- Total height of the stack in millimeters -/
def stack_height : ℚ := 19

/-- The number of coins in the stack -/
def total_coins : ℕ := 10

/-- Theorem stating that the total number of coins in a stack of 19 mm height,
    consisting only of 2p and 10p coins, is 10 -/
theorem coin_stack_count :
  ∃ (x y : ℕ), x + y = total_coins ∧
  x * thickness_2p + y * thickness_10p = stack_height :=
sorry

end NUMINAMATH_CALUDE_coin_stack_count_l438_43898


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l438_43811

/-- Given a rectangle with one side of length 18 and the sum of its area and perimeter being 2016,
    the perimeter of the rectangle is 234. -/
theorem rectangle_perimeter : ∀ w : ℝ,
  let l : ℝ := 18
  let area : ℝ := l * w
  let perimeter : ℝ := 2 * (l + w)
  area + perimeter = 2016 →
  perimeter = 234 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l438_43811


namespace NUMINAMATH_CALUDE_symmetric_lines_symmetric_line_equation_l438_43813

/-- Given two lines in the 2D plane and a point, this theorem states that these lines are symmetric with respect to the given point. -/
theorem symmetric_lines (x y : ℝ) : 
  (2 * x + 3 * y - 6 = 0) ↔ (2 * (2 - x) + 3 * (-2 - y) - 6 = 0) := by
  sorry

/-- The equation of the line symmetric to 2x + 3y - 6 = 0 with respect to the point (1, -1) is 2x + 3y + 8 = 0. -/
theorem symmetric_line_equation : 
  ∀ x y : ℝ, (2 * x + 3 * y - 6 = 0) ↔ (2 * ((2 - x) - 1) + 3 * ((-2 - y) - (-1)) + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_lines_symmetric_line_equation_l438_43813


namespace NUMINAMATH_CALUDE_inverse_true_implies_negation_true_l438_43894

theorem inverse_true_implies_negation_true (P : Prop) : 
  (¬P → ¬(¬P)) → (¬P) := by
  sorry

end NUMINAMATH_CALUDE_inverse_true_implies_negation_true_l438_43894


namespace NUMINAMATH_CALUDE_car_speed_time_relationship_car_q_graph_representation_l438_43821

/-- Represents a car's travel characteristics -/
structure CarTravel where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The theorem stating the relationship between Car P and Car Q's travel characteristics -/
theorem car_speed_time_relationship 
  (p q : CarTravel) 
  (h1 : p.distance = q.distance) 
  (h2 : q.speed = 3 * p.speed) : 
  q.time = p.time / 3 := by
sorry

/-- The theorem proving the graphical representation of Car Q's travel -/
theorem car_q_graph_representation 
  (p q : CarTravel) 
  (h1 : p.distance = q.distance) 
  (h2 : q.speed = 3 * p.speed) : 
  q.speed = 3 * p.speed ∧ q.time = p.time / 3 := by
sorry

end NUMINAMATH_CALUDE_car_speed_time_relationship_car_q_graph_representation_l438_43821


namespace NUMINAMATH_CALUDE_inequality_solution_l438_43866

theorem inequality_solution (x : ℝ) : 
  (x^2 + 1)/(x-2) ≥ 3/(x+2) + 2/3 ↔ x ∈ Set.Ioo (-2 : ℝ) (5/3 : ℝ) ∪ Set.Ioi (2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l438_43866


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l438_43834

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (x - 1) / (x + 2) ≤ 0 → -2 ≤ x ∧ x ≤ 1) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 1 ∧ (x - 1) / (x + 2) > 0) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l438_43834


namespace NUMINAMATH_CALUDE_Q_neither_sufficient_nor_necessary_for_P_l438_43808

/-- Proposition P: The solution sets of two quadratic inequalities are the same -/
def P (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  ∀ x, (a₁ * x^2 + b₁ * x + c₁ > 0) ↔ (a₂ * x^2 + b₂ * x + c₂ > 0)

/-- Proposition Q: The ratios of corresponding coefficients are equal -/
def Q (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂

theorem Q_neither_sufficient_nor_necessary_for_P :
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, Q a₁ b₁ c₁ a₂ b₂ c₂ ∧ ¬P a₁ b₁ c₁ a₂ b₂ c₂) ∧
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, P a₁ b₁ c₁ a₂ b₂ c₂ ∧ ¬Q a₁ b₁ c₁ a₂ b₂ c₂) :=
sorry

end NUMINAMATH_CALUDE_Q_neither_sufficient_nor_necessary_for_P_l438_43808


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l438_43841

def complex_number : ℂ := 2 - Complex.I

theorem complex_number_in_fourth_quadrant :
  Real.sign (complex_number.re) = 1 ∧ Real.sign (complex_number.im) = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l438_43841


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_hyperbola_standard_equation_l438_43843

-- Ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

theorem ellipse_standard_equation
  (major_axis : ℝ)
  (focal_distance : ℝ)
  (h_major_axis : major_axis = 4)
  (h_focal_distance : focal_distance = 2)
  (h_foci_on_x_axis : True) :
  ∀ x y : ℝ, ellipse_equation x y ↔ 
    x^2 / (major_axis^2 / 4) + y^2 / ((major_axis^2 / 4) - focal_distance^2) = 1 :=
sorry

-- Hyperbola
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

theorem hyperbola_standard_equation
  (k : ℝ)
  (d : ℝ)
  (h_asymptote : k = 3/4)
  (h_directrix : d = 16/5) :
  ∀ x y : ℝ, hyperbola_equation x y ↔ 
    x^2 / (d^2 / (1 + k^2)) - y^2 / ((d^2 * k^2) / (1 + k^2)) = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_hyperbola_standard_equation_l438_43843


namespace NUMINAMATH_CALUDE_rabbit_count_l438_43804

/-- The number of ducks in Eunji's house -/
def num_ducks : ℕ := 52

/-- The number of chickens in Eunji's house -/
def num_chickens : ℕ := 78

/-- The number of rabbits in Eunji's house -/
def num_rabbits : ℕ := 38

/-- Theorem stating the relationship between the number of animals and proving the number of rabbits -/
theorem rabbit_count : num_chickens = num_ducks + num_rabbits - 12 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_count_l438_43804


namespace NUMINAMATH_CALUDE_new_york_temperature_l438_43851

/-- Given temperatures in three cities with specific relationships, prove the temperature in New York --/
theorem new_york_temperature (t_ny : ℝ) : 
  let t_miami := t_ny + 10
  let t_sandiego := t_miami + 25
  (t_ny + t_miami + t_sandiego) / 3 = 95 →
  t_ny = 80 := by
sorry

end NUMINAMATH_CALUDE_new_york_temperature_l438_43851


namespace NUMINAMATH_CALUDE_fourth_circle_radius_l438_43847

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (O₁ O₂ O₃ O₄ : Circle) : Prop :=
  -- O₁ and O₂ are externally tangent
  let (x₁, y₁) := O₁.center
  let (x₂, y₂) := O₂.center
  ((x₂ - x₁)^2 + (y₂ - y₁)^2 = (O₁.radius + O₂.radius)^2) ∧
  -- Radii of O₁ and O₂
  (O₁.radius = 7) ∧
  (O₂.radius = 14) ∧
  -- O₃ is tangent to both O₁ and O₂
  let (x₃, y₃) := O₃.center
  ((x₃ - x₁)^2 + (y₃ - y₁)^2 = (O₁.radius + O₃.radius)^2) ∧
  ((x₃ - x₂)^2 + (y₃ - y₂)^2 = (O₂.radius + O₃.radius)^2) ∧
  -- Center of O₃ is on the line connecting centers of O₁ and O₂
  ((y₃ - y₁) * (x₂ - x₁) = (x₃ - x₁) * (y₂ - y₁)) ∧
  -- O₄ is tangent to O₁, O₂, and O₃
  let (x₄, y₄) := O₄.center
  ((x₄ - x₁)^2 + (y₄ - y₁)^2 = (O₁.radius + O₄.radius)^2) ∧
  ((x₄ - x₂)^2 + (y₄ - y₂)^2 = (O₂.radius + O₄.radius)^2) ∧
  ((x₄ - x₃)^2 + (y₄ - y₃)^2 = (O₃.radius - O₄.radius)^2)

theorem fourth_circle_radius (O₁ O₂ O₃ O₄ : Circle) :
  problem_setup O₁ O₂ O₃ O₄ → O₄.radius = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_l438_43847


namespace NUMINAMATH_CALUDE_platform_length_l438_43892

/-- The length of a platform given train specifications and crossing time -/
theorem platform_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 55 →
  crossing_time = 50.395968322534195 →
  ∃ platform_length : ℝ, abs (platform_length - 520) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l438_43892


namespace NUMINAMATH_CALUDE_cake_brownie_calorie_difference_l438_43838

def cake_slices : ℕ := 8
def calories_per_cake_slice : ℕ := 347
def brownies : ℕ := 6
def calories_per_brownie : ℕ := 375

theorem cake_brownie_calorie_difference :
  cake_slices * calories_per_cake_slice - brownies * calories_per_brownie = 526 := by
sorry

end NUMINAMATH_CALUDE_cake_brownie_calorie_difference_l438_43838


namespace NUMINAMATH_CALUDE_four_digit_numbers_count_l438_43833

/-- Represents the set of cards with their numbers -/
def cards : Multiset ℕ := {1, 1, 1, 2, 2, 3, 4}

/-- The number of cards drawn -/
def draw_count : ℕ := 4

/-- Calculates the number of different four-digit numbers that can be formed -/
noncomputable def four_digit_numbers (c : Multiset ℕ) (d : ℕ) : ℕ := sorry

/-- Theorem stating that the number of different four-digit numbers is 114 -/
theorem four_digit_numbers_count : four_digit_numbers cards draw_count = 114 := by sorry

end NUMINAMATH_CALUDE_four_digit_numbers_count_l438_43833


namespace NUMINAMATH_CALUDE_no_numbers_satisfying_conditions_l438_43891

theorem no_numbers_satisfying_conditions : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 300 →
    (6 ∣ n ∧ 8 ∣ n) → (4 ∣ n ∨ 11 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_no_numbers_satisfying_conditions_l438_43891


namespace NUMINAMATH_CALUDE_count_D3_le_200_eq_9_l438_43832

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- Count of positive integers n ≤ 200 for which D(n) = 3 -/
def count_D3_le_200 : ℕ := sorry

theorem count_D3_le_200_eq_9 : count_D3_le_200 = 9 := by sorry

end NUMINAMATH_CALUDE_count_D3_le_200_eq_9_l438_43832


namespace NUMINAMATH_CALUDE_chord_minimum_value_l438_43837

theorem chord_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let line := {p : ℝ × ℝ | a * p.1 - b * p.2 + 2 = 0}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - 4*p.2 + 1 = 0}
  let chord_length := 4
  (∃ (p q : ℝ × ℝ), p ∈ line ∧ q ∈ line ∧ p ∈ circle ∧ q ∈ circle ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) →
  (2/a + 3/b ≥ 4 + 2 * Real.sqrt 3) ∧ 
  (∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ 2/a' + 3/b' = 4 + 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_chord_minimum_value_l438_43837


namespace NUMINAMATH_CALUDE_two_digit_cube_sum_square_l438_43862

theorem two_digit_cube_sum_square : 
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ (((n / 10) + (n % 10))^3 = n^2) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_cube_sum_square_l438_43862


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l438_43826

theorem polynomial_division_remainder : ∃ (r : ℚ),
  ∀ (z : ℚ), 4 * z^3 - 5 * z^2 - 18 * z + 4 = (4 * z + 6) * (z^2 - 4 * z + 2/3) + r :=
by
  use 10/3
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l438_43826


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l438_43850

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The general term of the sequence. -/
def a (n : ℕ) : ℝ := 2 * n + 5

/-- The theorem stating that the given sequence is arithmetic with first term 7 and common difference 2. -/
theorem sequence_is_arithmetic :
  IsArithmeticSequence a ∧ a 1 = 7 ∧ ∀ n : ℕ, a (n + 1) - a n = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l438_43850


namespace NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l438_43835

theorem margin_in_terms_of_selling_price (n : ℕ) (C S M : ℝ) 
  (h_n : n > 0)
  (h_margin : M = (1/2) * (S - (1/n) * C))
  (h_cost : C = S - M) :
  M = ((n - 1) / (2 * n - 1)) * S := by
sorry

end NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l438_43835


namespace NUMINAMATH_CALUDE_largest_lcm_with_15_l438_43869

theorem largest_lcm_with_15 :
  (List.maximum [Nat.lcm 15 2, Nat.lcm 15 3, Nat.lcm 15 5, Nat.lcm 15 6, Nat.lcm 15 9, Nat.lcm 15 10]).get! = 45 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_15_l438_43869


namespace NUMINAMATH_CALUDE_teacher_books_l438_43885

theorem teacher_books (num_children : ℕ) (books_per_child : ℕ) (total_books : ℕ) : 
  num_children = 10 → books_per_child = 7 → total_books = 78 →
  total_books - (num_children * books_per_child) = 8 := by
sorry

end NUMINAMATH_CALUDE_teacher_books_l438_43885


namespace NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l438_43880

/-- Given three positive real numbers forming an arithmetic sequence,
    the sum of their ratio and its reciprocal is at least 5/2 -/
theorem min_value_arithmetic_sequence (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_arith : b - a = c - b) : (a + c) / b + b / (a + c) ≥ 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_arithmetic_sequence_l438_43880


namespace NUMINAMATH_CALUDE_condition_relationship_l438_43807

theorem condition_relationship (a b : ℝ) : 
  (∀ a b, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧ 
  (∃ a b, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l438_43807


namespace NUMINAMATH_CALUDE_election_win_percentage_l438_43824

/-- The minimum percentage of votes needed to win an election --/
def min_win_percentage (total_votes : ℕ) (geoff_percentage : ℚ) (additional_votes_needed : ℕ) : ℚ :=
  ((geoff_percentage * total_votes + additional_votes_needed) / total_votes) * 100

/-- Theorem stating the minimum percentage of votes needed to win the election --/
theorem election_win_percentage :
  let total_votes : ℕ := 6000
  let geoff_percentage : ℚ := 1/200
  let additional_votes_needed : ℕ := 3000
  min_win_percentage total_votes geoff_percentage additional_votes_needed = 101/2 := by
  sorry

end NUMINAMATH_CALUDE_election_win_percentage_l438_43824


namespace NUMINAMATH_CALUDE_multiple_of_nine_implies_multiple_of_three_l438_43805

theorem multiple_of_nine_implies_multiple_of_three 
  (h1 : ∀ n : ℕ, 9 ∣ n → 3 ∣ n) 
  (k : ℕ) 
  (h2 : Odd k) 
  (h3 : 9 ∣ k) : 
  3 ∣ k := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_nine_implies_multiple_of_three_l438_43805


namespace NUMINAMATH_CALUDE_carrot_weight_theorem_l438_43822

/-- Given 30 carrots, where 27 of them have an average weight of 200 grams
    and 3 of them have an average weight of 180 grams,
    the total weight of all 30 carrots is 5.94 kg. -/
theorem carrot_weight_theorem :
  let total_carrots : ℕ := 30
  let remaining_carrots : ℕ := 27
  let removed_carrots : ℕ := 3
  let avg_weight_remaining : ℝ := 200 -- in grams
  let avg_weight_removed : ℝ := 180 -- in grams
  let total_weight_grams : ℝ := remaining_carrots * avg_weight_remaining + removed_carrots * avg_weight_removed
  let total_weight_kg : ℝ := total_weight_grams / 1000
  total_weight_kg = 5.94 := by
  sorry

end NUMINAMATH_CALUDE_carrot_weight_theorem_l438_43822


namespace NUMINAMATH_CALUDE_novel_reading_distribution_l438_43890

/-- Represents the reading assignment for three friends -/
structure ReadingAssignment where
  total_pages : ℕ
  alice_speed : ℕ
  bob_speed : ℕ
  chandra_speed : ℕ
  alice_pages : ℕ
  bob_pages : ℕ
  chandra_pages : ℕ

/-- Theorem stating the correct distribution of pages for the given conditions -/
theorem novel_reading_distribution (assignment : ReadingAssignment) :
  assignment.total_pages = 912 ∧
  assignment.alice_speed = 40 ∧
  assignment.bob_speed = 60 ∧
  assignment.chandra_speed = 48 ∧
  assignment.chandra_pages = 420 →
  assignment.alice_pages = 295 ∧
  assignment.bob_pages = 197 ∧
  assignment.alice_pages + assignment.bob_pages + assignment.chandra_pages = assignment.total_pages :=
by sorry

end NUMINAMATH_CALUDE_novel_reading_distribution_l438_43890


namespace NUMINAMATH_CALUDE_isaac_number_problem_l438_43812

theorem isaac_number_problem (a b : ℤ) : 
  (2 * a + 3 * b = 100) → 
  ((a = 28 ∨ b = 28) → (a = 8 ∨ b = 8)) :=
by sorry

end NUMINAMATH_CALUDE_isaac_number_problem_l438_43812


namespace NUMINAMATH_CALUDE_cos_540_degrees_l438_43823

theorem cos_540_degrees : Real.cos (540 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_cos_540_degrees_l438_43823


namespace NUMINAMATH_CALUDE_hotel_operations_cost_l438_43878

/-- Proves that the total cost of operations is $100 given the specified conditions --/
theorem hotel_operations_cost (cost : ℝ) (payments : ℝ) (loss : ℝ) : 
  payments = (3/4) * cost → 
  loss = 25 → 
  payments + loss = cost → 
  cost = 100 := by
  sorry

end NUMINAMATH_CALUDE_hotel_operations_cost_l438_43878


namespace NUMINAMATH_CALUDE_max_value_a_l438_43857

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 2 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 150) : 
  a ≤ 4460 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 4460 ∧ 
    b' = 1487 ∧ 
    c' = 744 ∧ 
    d' = 149 ∧ 
    a' < 3 * b' ∧ 
    b' < 2 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 150 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l438_43857


namespace NUMINAMATH_CALUDE_girls_in_class_l438_43852

theorem girls_in_class (total : ℕ) (g b t : ℕ) : 
  total = 60 →
  g + b + t = total →
  3 * t = g →
  2 * t = b →
  g = 30 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l438_43852


namespace NUMINAMATH_CALUDE_collinear_probability_is_7_6325_l438_43855

/-- A 5x5 grid of dots -/
structure Grid :=
  (size : Nat)
  (h_size : size = 5)

/-- The number of collinear sets of four dots in a 5x5 grid -/
def collinear_sets (g : Grid) : Nat := 14

/-- The total number of ways to choose 4 dots from 25 -/
def total_sets (g : Grid) : Nat := 12650

/-- The probability of selecting four collinear dots from a 5x5 grid -/
def collinear_probability (g : Grid) : ℚ :=
  (collinear_sets g : ℚ) / (total_sets g : ℚ)

/-- Theorem stating that the probability of selecting four collinear dots from a 5x5 grid is 7/6325 -/
theorem collinear_probability_is_7_6325 (g : Grid) :
  collinear_probability g = 7 / 6325 := by
  sorry

end NUMINAMATH_CALUDE_collinear_probability_is_7_6325_l438_43855


namespace NUMINAMATH_CALUDE_lowest_sale_price_percentage_l438_43853

theorem lowest_sale_price_percentage (list_price : ℝ) (max_regular_discount : ℝ) (summer_discount : ℝ) :
  list_price = 80 →
  max_regular_discount = 0.5 →
  summer_discount = 0.2 →
  let regular_discounted_price := list_price * (1 - max_regular_discount)
  let summer_discount_amount := list_price * summer_discount
  let final_sale_price := regular_discounted_price - summer_discount_amount
  (final_sale_price / list_price) * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_lowest_sale_price_percentage_l438_43853


namespace NUMINAMATH_CALUDE_power_of_one_fourth_l438_43868

theorem power_of_one_fourth (a b : ℕ) : 
  (2^a : ℕ) = (180 / (180 / 2^a : ℕ) : ℕ) →
  (3^b : ℕ) = (180 / (180 / 3^b : ℕ) : ℕ) →
  (1/4 : ℚ)^(b - a) = 1 := by
sorry

end NUMINAMATH_CALUDE_power_of_one_fourth_l438_43868


namespace NUMINAMATH_CALUDE_ratio_problem_l438_43882

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 4) (h2 : c/b = 2) :
  (a + b + c) / (a + b) = 13/5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l438_43882


namespace NUMINAMATH_CALUDE_continued_fraction_sum_l438_43876

theorem continued_fraction_sum (x y z : ℕ+) : 
  (151 : ℚ) / 44 = 3 + 1 / (x.val + 1 / (y.val + 1 / z.val)) → 
  x.val + y.val + z.val = 11 := by
sorry

end NUMINAMATH_CALUDE_continued_fraction_sum_l438_43876


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l438_43877

theorem quadratic_roots_sum_of_squares (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 8 = 0) → 
  (x₂^2 + 2*x₂ - 8 = 0) → 
  (x₁^2 + x₂^2 = 20) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l438_43877


namespace NUMINAMATH_CALUDE_mrs_hilt_apple_pies_l438_43806

/-- The number of apple pies Mrs. Hilt baked -/
def apple_pies : ℕ := 150 - 16

/-- Theorem: Mrs. Hilt baked 134 apple pies -/
theorem mrs_hilt_apple_pies : apple_pies = 134 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_apple_pies_l438_43806


namespace NUMINAMATH_CALUDE_train_length_l438_43860

/-- The length of a train given its crossing times over a post and a platform -/
theorem train_length (post_time : ℝ) (platform_length platform_time : ℝ) : 
  post_time = 10 →
  platform_length = 150 →
  platform_time = 20 →
  ∃ (train_length : ℝ), 
    train_length / post_time = (train_length + platform_length) / platform_time ∧
    train_length = 150 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l438_43860


namespace NUMINAMATH_CALUDE_nested_expression_evaluation_l438_43872

theorem nested_expression_evaluation : (5*(5*(5*(5+1)+1)+1)+1) = 781 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_evaluation_l438_43872


namespace NUMINAMATH_CALUDE_ben_win_probability_l438_43836

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 3/7) 
  (h2 : ∀ (tie_prob : ℚ), tie_prob = 0) : 
  1 - lose_prob = 4/7 := by
sorry

end NUMINAMATH_CALUDE_ben_win_probability_l438_43836


namespace NUMINAMATH_CALUDE_michelle_sandwiches_l438_43825

theorem michelle_sandwiches (total : ℕ) (given : ℕ) (kept : ℕ) (remaining : ℕ) : 
  total = 20 → 
  given = 4 → 
  kept = 2 * given → 
  remaining = total - given - kept → 
  remaining = 8 := by
sorry

end NUMINAMATH_CALUDE_michelle_sandwiches_l438_43825


namespace NUMINAMATH_CALUDE_number_division_remainder_l438_43865

theorem number_division_remainder (n : ℕ) : 
  (n / 8 = 8 ∧ n % 8 = 0) → n % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_division_remainder_l438_43865


namespace NUMINAMATH_CALUDE_vector_equality_iff_collinear_l438_43873

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- The theorem states that for arbitrary points O, A, B, C in a vector space and a scalar k,
    the equality OC = k*OA + (1-k)*OB is equivalent to A, B, and C being collinear. -/
theorem vector_equality_iff_collinear 
  (O A B C : V) (k : ℝ) : 
  (C - O = k • (A - O) + (1 - k) • (B - O)) ↔ 
  ∃ t : ℝ, C - B = t • (A - B) :=
sorry

end NUMINAMATH_CALUDE_vector_equality_iff_collinear_l438_43873


namespace NUMINAMATH_CALUDE_least_value_theorem_l438_43846

theorem least_value_theorem (p q : ℕ) (x : ℚ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : 17 * (p + 1) = x * (q + 1))
  (h4 : ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = x * (q' + 1) → p' + q' ≥ 40)
  (h5 : p + q = 40) : 
  x = 14 := by
sorry

end NUMINAMATH_CALUDE_least_value_theorem_l438_43846


namespace NUMINAMATH_CALUDE_inequality_proof_l438_43809

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) / (x * y * z) ^ (1/3) ≤ x/y + y/z + z/x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l438_43809


namespace NUMINAMATH_CALUDE_eldoria_population_2070_l438_43845

/-- The population growth function for Eldoria -/
def eldoria_population (initial_population : ℕ) (years_since_2000 : ℕ) : ℕ :=
  initial_population * (2 ^ (years_since_2000 / 15))

/-- Theorem: The population of Eldoria in 2070 is 8000 -/
theorem eldoria_population_2070 : 
  eldoria_population 500 70 = 8000 := by
  sorry

#eval eldoria_population 500 70

end NUMINAMATH_CALUDE_eldoria_population_2070_l438_43845


namespace NUMINAMATH_CALUDE_two_part_trip_average_speed_l438_43863

/-- Calculates the average speed of a two-part trip -/
theorem two_part_trip_average_speed
  (total_distance : ℝ)
  (first_part_distance : ℝ)
  (first_part_speed : ℝ)
  (second_part_speed : ℝ)
  (h1 : total_distance = 450)
  (h2 : first_part_distance = 300)
  (h3 : first_part_speed = 20)
  (h4 : second_part_speed = 15)
  (h5 : first_part_distance < total_distance) :
  (total_distance) / ((first_part_distance / first_part_speed) + ((total_distance - first_part_distance) / second_part_speed)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_two_part_trip_average_speed_l438_43863


namespace NUMINAMATH_CALUDE_consecutive_substring_perfect_square_l438_43883

/-- A type representing a 16-digit positive integer -/
def SixteenDigitInteger := { n : ℕ // 10^15 ≤ n ∧ n < 10^16 }

/-- A function that checks if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- A function that returns the product of digits in a substring of a number -/
def substring_product (n : ℕ) (start finish : ℕ) : ℕ := sorry

/-- The main theorem: For any 16-digit positive integer, there exists a consecutive
    substring of digits whose product is a perfect square -/
theorem consecutive_substring_perfect_square (A : SixteenDigitInteger) :
  ∃ start finish : ℕ, start ≤ finish ∧ finish ≤ 16 ∧
    is_perfect_square (substring_product A.val start finish) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_substring_perfect_square_l438_43883


namespace NUMINAMATH_CALUDE_count_valid_numbers_l438_43893

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ 
  (n % 1000 = n / 6)

theorem count_valid_numbers : 
  ∃ (s : Finset ℕ), (∀ n ∈ s, is_valid_number n) ∧ s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l438_43893


namespace NUMINAMATH_CALUDE_bridgets_score_l438_43801

theorem bridgets_score (total_students : ℕ) (students_before : ℕ) (avg_before : ℚ) (avg_after : ℚ) 
  (h1 : total_students = 18)
  (h2 : students_before = 17)
  (h3 : avg_before = 76)
  (h4 : avg_after = 78) :
  (total_students : ℚ) * avg_after - (students_before : ℚ) * avg_before = 112 := by
  sorry

end NUMINAMATH_CALUDE_bridgets_score_l438_43801


namespace NUMINAMATH_CALUDE_sin_cos_roots_quadratic_l438_43814

theorem sin_cos_roots_quadratic (θ : Real) (m : Real) : 
  (4 * (Real.sin θ)^2 + 2 * m * (Real.sin θ) + m = 0) ∧ 
  (4 * (Real.cos θ)^2 + 2 * m * (Real.cos θ) + m = 0) →
  m = 1 - Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_roots_quadratic_l438_43814


namespace NUMINAMATH_CALUDE_heels_cost_calculation_solve_shopping_problem_l438_43817

def shopping_problem (initial_amount jumper_cost tshirt_cost remaining_amount : ℕ) : Prop :=
  ∃ (heels_cost : ℕ),
    initial_amount = jumper_cost + tshirt_cost + heels_cost + remaining_amount

theorem heels_cost_calculation (initial_amount jumper_cost tshirt_cost remaining_amount : ℕ) 
  (h : shopping_problem initial_amount jumper_cost tshirt_cost remaining_amount) :
  ∃ (heels_cost : ℕ), heels_cost = initial_amount - jumper_cost - tshirt_cost - remaining_amount :=
by
  sorry

#check @heels_cost_calculation

theorem solve_shopping_problem :
  shopping_problem 26 9 4 8 ∧ 
  (∃ (heels_cost : ℕ), heels_cost = 26 - 9 - 4 - 8 ∧ heels_cost = 5) :=
by
  sorry

#check @solve_shopping_problem

end NUMINAMATH_CALUDE_heels_cost_calculation_solve_shopping_problem_l438_43817


namespace NUMINAMATH_CALUDE_calzone_ratio_l438_43887

def calzone_problem (onion_time garlic_time knead_time rest_time assemble_time total_time : ℕ) : Prop :=
  let pepper_time := garlic_time
  onion_time = 20 ∧
  garlic_time = onion_time / 4 ∧
  knead_time = 30 ∧
  rest_time = 2 * knead_time ∧
  total_time = 124 ∧
  total_time = onion_time + garlic_time + pepper_time + knead_time + rest_time + assemble_time

theorem calzone_ratio (onion_time garlic_time knead_time rest_time assemble_time total_time : ℕ) :
  calzone_problem onion_time garlic_time knead_time rest_time assemble_time total_time →
  (assemble_time : ℚ) / (knead_time + rest_time : ℚ) = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_calzone_ratio_l438_43887


namespace NUMINAMATH_CALUDE_sum_coordinates_of_X_l438_43831

/-- Given three points X, Y, and Z in the plane satisfying certain conditions,
    prove that the sum of the coordinates of X is -28. -/
theorem sum_coordinates_of_X (X Y Z : ℝ × ℝ) : 
  (∃ (k : ℝ), k = 1/2 ∧ Z - X = k • (Y - X) ∧ Y - Z = k • (Y - X)) → 
  Y = (3, 9) →
  Z = (1, -9) →
  X.1 + X.2 = -28 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_X_l438_43831


namespace NUMINAMATH_CALUDE_factors_of_12_and_18_l438_43840

def factors (n : ℕ) : Set ℕ := {x | x ∣ n}

theorem factors_of_12_and_18 : 
  factors 12 = {1, 2, 3, 4, 6, 12} ∧ factors 18 = {1, 2, 3, 6, 9, 18} := by
  sorry

end NUMINAMATH_CALUDE_factors_of_12_and_18_l438_43840


namespace NUMINAMATH_CALUDE_purple_or_orange_probability_l438_43819

/-- A die with colored faces -/
structure ColoredDie where
  sides : ℕ
  green : ℕ
  purple : ℕ
  orange : ℕ
  sum_faces : green + purple + orange = sides

/-- The probability of an event -/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

/-- The main theorem -/
theorem purple_or_orange_probability (d : ColoredDie)
    (h : d.sides = 10 ∧ d.green = 5 ∧ d.purple = 3 ∧ d.orange = 2) :
    probability (d.purple + d.orange) d.sides = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_purple_or_orange_probability_l438_43819


namespace NUMINAMATH_CALUDE_sum_of_digits_product_of_nines_l438_43884

/-- 
Given a natural number n, define a function that calculates the product:
9 × 99 × 9999 × ⋯ × (99...99) where the number of nines doubles in each factor
and the last factor has 2^n nines.
-/
def productOfNines (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * (10^(2^i) - 1)) 9

/-- 
Sum of digits function
-/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- 
Theorem: The sum of the digits of the product of nines is equal to 9 * 2^n
-/
theorem sum_of_digits_product_of_nines (n : ℕ) :
  sumOfDigits (productOfNines n) = 9 * 2^n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_product_of_nines_l438_43884


namespace NUMINAMATH_CALUDE_commission_is_25_l438_43858

/-- Represents the sales data for a salesman selling security systems --/
structure SalesData where
  second_street_sales : Nat
  fourth_street_sales : Nat
  total_commission : Nat

/-- Calculates the total number of security systems sold --/
def total_sales (data : SalesData) : Nat :=
  data.second_street_sales + (data.second_street_sales / 2) + data.fourth_street_sales

/-- Calculates the commission per security system --/
def commission_per_system (data : SalesData) : Nat :=
  data.total_commission / (total_sales data)

/-- Theorem stating that given the sales conditions, the commission per system is $25 --/
theorem commission_is_25 (data : SalesData) 
  (h1 : data.second_street_sales = 4)
  (h2 : data.fourth_street_sales = 1)
  (h3 : data.total_commission = 175) :
  commission_per_system data = 25 := by
  sorry

#eval commission_per_system { second_street_sales := 4, fourth_street_sales := 1, total_commission := 175 }

end NUMINAMATH_CALUDE_commission_is_25_l438_43858


namespace NUMINAMATH_CALUDE_parallel_quadrilateral_coordinates_l438_43897

/-- A quadrilateral with parallel sides and non-intersecting diagonals -/
structure ParallelQuadrilateral (a b c d : ℝ) :=
  (xC : ℝ)
  (xD : ℝ)
  (yC : ℝ)
  (side_AB : ℝ := a)
  (side_BC : ℝ := b)
  (side_CD : ℝ := c)
  (side_DA : ℝ := d)
  (parallel : yC = yC)  -- AB parallel to CD
  (non_intersecting : c = xC - xD)  -- BC and DA do not intersect
  (length_BC : b^2 = xC^2 + yC^2)
  (length_AD : d^2 = (xD + a)^2 + yC^2)

/-- The x-coordinates of points C and D in a parallel quadrilateral -/
theorem parallel_quadrilateral_coordinates
  (a b c d : ℝ) (quad : ParallelQuadrilateral a b c d)
  (h_a : a ≠ c) :
  quad.xD = (d^2 - b^2 - a^2 + c^2) / (2*(a - c)) ∧
  quad.xC = (d^2 - b^2 - a^2 + c^2) / (2*(a - c)) + c :=
sorry

end NUMINAMATH_CALUDE_parallel_quadrilateral_coordinates_l438_43897


namespace NUMINAMATH_CALUDE_sum_of_composite_function_l438_43870

def p (x : ℝ) : ℝ := 2 * abs x - 1

def q (x : ℝ) : ℝ := -abs x - 1

def xValues : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_composite_function : 
  (xValues.map (fun x => q (p x))).sum = -42 := by sorry

end NUMINAMATH_CALUDE_sum_of_composite_function_l438_43870


namespace NUMINAMATH_CALUDE_swimmer_time_proof_l438_43810

/-- Proves that a swimmer takes 3 hours for both downstream and upstream swims given specific conditions -/
theorem swimmer_time_proof (downstream_distance upstream_distance still_water_speed : ℝ) 
  (h1 : downstream_distance = 18)
  (h2 : upstream_distance = 12)
  (h3 : still_water_speed = 5)
  (h4 : downstream_distance / (still_water_speed + (downstream_distance - upstream_distance) / 6) = 
        upstream_distance / (still_water_speed - (downstream_distance - upstream_distance) / 6)) :
  downstream_distance / (still_water_speed + (downstream_distance - upstream_distance) / 6) = 3 ∧
  upstream_distance / (still_water_speed - (downstream_distance - upstream_distance) / 6) = 3 := by
  sorry

#check swimmer_time_proof

end NUMINAMATH_CALUDE_swimmer_time_proof_l438_43810


namespace NUMINAMATH_CALUDE_fraction_equality_l438_43803

theorem fraction_equality (w x y : ℝ) (hw : w / x = 1 / 3) (hxy : (x + y) / y = 3) :
  w / y = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l438_43803


namespace NUMINAMATH_CALUDE_complex_equation_solution_l438_43896

theorem complex_equation_solution (z : ℂ) : (z - 2) * (1 + Complex.I) = 1 - Complex.I → z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l438_43896


namespace NUMINAMATH_CALUDE_first_player_wins_l438_43818

/-- Represents the game state -/
structure GameState where
  m : Nat
  n : Nat

/-- Represents a move in the game -/
structure Move where
  row : Nat
  col : Nat

/-- Determines if a move is valid for a given game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  1 ≤ move.row ∧ move.row ≤ state.m ∧ 1 ≤ move.col ∧ move.col ≤ state.n

/-- Applies a move to a game state, returning the new state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  { m := move.row - 1, n := move.col - 1 }

/-- Determines if a game state is terminal (i.e., only the losing square remains) -/
def isTerminal (state : GameState) : Prop :=
  state.m = 1 ∧ state.n = 1

/-- Theorem: The first player has a winning strategy in the chocolate bar game -/
theorem first_player_wins (initialState : GameState) : 
  initialState.m ≥ 1 ∧ initialState.n ≥ 1 → 
  ∃ (strategy : GameState → Move), 
    (∀ (state : GameState), isValidMove state (strategy state)) ∧ 
    (∀ (state : GameState), ¬isTerminal state → 
      ¬∃ (counterStrategy : GameState → Move), 
        (∀ (s : GameState), isValidMove s (counterStrategy s)) ∧
        isTerminal (applyMove (applyMove state (strategy state)) (counterStrategy (applyMove state (strategy state))))) :=
by
  sorry

end NUMINAMATH_CALUDE_first_player_wins_l438_43818


namespace NUMINAMATH_CALUDE_triangle_inequality_third_side_range_l438_43844

theorem triangle_inequality (a b c : ℝ) : 
  (0 < a ∧ 0 < b ∧ 0 < c) → (a < b + c ∧ b < a + c ∧ c < a + b) := by sorry

theorem third_side_range : 
  ∀ a : ℝ, (∃ (s1 s2 : ℝ), s1 = 3 ∧ s2 = 5 ∧ 0 < a ∧ 
    (a < s1 + s2 ∧ s1 < a + s2 ∧ s2 < a + s1)) → 
  (2 < a ∧ a < 8) := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_third_side_range_l438_43844


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_15_l438_43816

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  (List.range n).map factorial |> List.sum

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_sum_factorials_15 :
  units_digit (sum_factorials 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_15_l438_43816


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l438_43820

theorem simplify_trig_expression : 
  (Real.sqrt 3 / Real.cos (10 * π / 180)) - (1 / Real.sin (170 * π / 180)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l438_43820


namespace NUMINAMATH_CALUDE_point_on_curve_iff_satisfies_equation_l438_43827

-- Define a curve C in 2D space
def Curve (F : ℝ → ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | F p.1 p.2 = 0}

-- Define a point P
def Point (a b : ℝ) : ℝ × ℝ := (a, b)

-- Theorem statement
theorem point_on_curve_iff_satisfies_equation (F : ℝ → ℝ → ℝ) (a b : ℝ) :
  Point a b ∈ Curve F ↔ F a b = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_iff_satisfies_equation_l438_43827


namespace NUMINAMATH_CALUDE_speed_ratio_is_two_l438_43886

/-- Represents the driving scenario for Daniel's commute --/
structure DrivingScenario where
  x : ℝ  -- Speed on Sunday in miles per hour
  y : ℝ  -- Speed for first 32 miles on Monday in miles per hour
  total_distance : ℝ  -- Total distance in miles
  first_part_distance : ℝ  -- Distance of first part on Monday in miles

/-- The theorem stating the ratio of speeds --/
theorem speed_ratio_is_two (scenario : DrivingScenario) : 
  scenario.x > 0 → 
  scenario.y > 0 → 
  scenario.total_distance = 60 → 
  scenario.first_part_distance = 32 → 
  (scenario.first_part_distance / scenario.y + (scenario.total_distance - scenario.first_part_distance) / (scenario.x / 2)) = 
    1.2 * (scenario.total_distance / scenario.x) → 
  scenario.y / scenario.x = 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_is_two_l438_43886


namespace NUMINAMATH_CALUDE_inequality_solution_set_l438_43871

theorem inequality_solution_set (x : ℝ) :
  (8 * x^3 - 6 * x^2 + 5 * x - 1 < 4) ↔ (x < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l438_43871


namespace NUMINAMATH_CALUDE_julio_fishing_result_l438_43802

/-- Calculates the number of fish Julio has after fishing for a given duration and losing some fish. -/
def fish_remaining (rate : ℕ) (duration : ℕ) (loss : ℕ) : ℕ :=
  rate * duration - loss

/-- Proves that Julio has 48 fish after fishing for 9 hours at a rate of 7 fish per hour and losing 15 fish. -/
theorem julio_fishing_result :
  fish_remaining 7 9 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_julio_fishing_result_l438_43802


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l438_43854

theorem polynomial_division_remainder : 
  ∃ (q : Polynomial ℝ), 
    x^4 + 2*x^3 - 3*x^2 + 4*x - 5 = (x^2 - 3*x + 2) * q + (24*x - 25) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l438_43854


namespace NUMINAMATH_CALUDE_intersection_line_equation_l438_43856

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

-- Define the line
def line (x y : ℝ) : Prop := 2*x + y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (A B : ℝ × ℝ),
  (circle1 A.1 A.2 ∧ circle1 B.1 B.2) ∧
  (circle2 A.1 A.2 ∧ circle2 B.1 B.2) ∧
  A ≠ B →
  line A.1 A.2 ∧ line B.1 B.2 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l438_43856


namespace NUMINAMATH_CALUDE_sin_negative_1290_degrees_l438_43888

theorem sin_negative_1290_degrees (θ : ℝ) :
  (∀ k : ℤ, Real.sin (θ + k * (2 * π)) = Real.sin θ) →
  (∀ θ : ℝ, Real.sin (π - θ) = Real.sin θ) →
  Real.sin (π / 6) = 1 / 2 →
  Real.sin (-1290 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_1290_degrees_l438_43888


namespace NUMINAMATH_CALUDE_min_difference_of_h_l438_43849

noncomputable section

variable (a : ℝ) (x₁ x₂ : ℝ)

def h (x : ℝ) : ℝ := x - 1/x + a * Real.log x

theorem min_difference_of_h (ha : a > 0) (hx₁ : 0 < x₁ ∧ x₁ ≤ 1/Real.exp 1)
  (hroots : x₁^2 + a*x₁ + 1 = 0 ∧ x₂^2 + a*x₂ + 1 = 0) :
  ∃ (m : ℝ), m = 4/Real.exp 1 ∧ ∀ y₁ y₂, 
    (0 < y₁ ∧ y₁ ≤ 1/Real.exp 1) → 
    (y₁^2 + a*y₁ + 1 = 0 ∧ y₂^2 + a*y₂ + 1 = 0) → 
    h a y₁ - h a y₂ ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_difference_of_h_l438_43849


namespace NUMINAMATH_CALUDE_cubic_factorization_l438_43839

theorem cubic_factorization (m : ℝ) : m^3 - 4*m = m*(m+2)*(m-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l438_43839


namespace NUMINAMATH_CALUDE_odd_function_interval_l438_43864

/-- A function f is odd on an interval [a, b] if and only if
    the interval is symmetric around the origin -/
def is_odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, f (-x) = -f x ∧ -x ∈ Set.Icc a b

theorem odd_function_interval (f : ℝ → ℝ) (b : ℝ) :
  is_odd_on_interval f (b - 1) 2 → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_interval_l438_43864
