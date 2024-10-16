import Mathlib

namespace NUMINAMATH_CALUDE_very_spicy_peppers_l1880_188056

/-- The number of peppers needed for very spicy curries -/
def V : ℕ := sorry

/-- The number of peppers needed for spicy curries -/
def spicy_peppers : ℕ := 2

/-- The number of peppers needed for mild curries -/
def mild_peppers : ℕ := 1

/-- The number of spicy curries after adjustment -/
def spicy_curries : ℕ := 15

/-- The number of mild curries after adjustment -/
def mild_curries : ℕ := 90

/-- The reduction in the number of peppers bought after adjustment -/
def pepper_reduction : ℕ := 40

theorem very_spicy_peppers : 
  V = pepper_reduction := by sorry

end NUMINAMATH_CALUDE_very_spicy_peppers_l1880_188056


namespace NUMINAMATH_CALUDE_shooting_game_equations_l1880_188087

/-- Represents the shooting game scenario -/
structure ShootingGame where
  x : ℕ  -- number of baskets Xiao Ming made
  y : ℕ  -- number of baskets his father made

/-- The conditions of the shooting game -/
def valid_game (g : ShootingGame) : Prop :=
  g.x + g.y = 20 ∧ 3 * g.x = g.y

theorem shooting_game_equations (g : ShootingGame) :
  valid_game g ↔ g.x + g.y = 20 ∧ 3 * g.x = g.y :=
sorry

end NUMINAMATH_CALUDE_shooting_game_equations_l1880_188087


namespace NUMINAMATH_CALUDE_number_difference_l1880_188009

theorem number_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) (h3 : y > x) :
  y - x = 8.58 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1880_188009


namespace NUMINAMATH_CALUDE_line_equation_l1880_188083

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

/-- Point M -/
def M : ℝ × ℝ := (4, 1)

/-- Line passing through two points -/
def line_through (p₁ p₂ : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p₁.2) * (p₂.1 - p₁.1) = (x - p₁.1) * (p₂.2 - p₁.2)

/-- Midpoint of two points -/
def is_midpoint (m p₁ p₂ : ℝ × ℝ) : Prop :=
  m.1 = (p₁.1 + p₂.1) / 2 ∧ m.2 = (p₁.2 + p₂.2) / 2

theorem line_equation :
  ∃ (A B : ℝ × ℝ),
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    is_midpoint M A B ∧
    (∀ x y, line_through M (x, y) x y ↔ y = 8*x - 31) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l1880_188083


namespace NUMINAMATH_CALUDE_positive_roots_range_l1880_188094

theorem positive_roots_range (m : ℝ) :
  (∀ x : ℝ, x^2 + (m+2)*x + m+5 = 0 → x > 0) ↔ -5 < m ∧ m ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_positive_roots_range_l1880_188094


namespace NUMINAMATH_CALUDE_coin_flip_game_properties_l1880_188016

/-- Represents the coin-flipping game where a player wins if heads come up on an even-numbered throw
    or loses if tails come up on an odd-numbered throw. -/
def CoinFlipGame :=
  { win_prob : ℝ // win_prob = 1/3 } × { expected_flips : ℝ // expected_flips = 2 }

/-- The probability of winning the coin-flipping game is 1/3, and the expected number of flips is 2. -/
theorem coin_flip_game_properties : ∃ (game : CoinFlipGame), True :=
sorry

end NUMINAMATH_CALUDE_coin_flip_game_properties_l1880_188016


namespace NUMINAMATH_CALUDE_problem_statement_l1880_188090

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2016 + b^2016 = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1880_188090


namespace NUMINAMATH_CALUDE_tan_value_from_trig_equation_l1880_188075

theorem tan_value_from_trig_equation (x : Real) 
  (h1 : 0 < x) (h2 : x < π/2) 
  (h3 : (Real.sin x)^4 / 9 + (Real.cos x)^4 / 4 = 1/13) : 
  Real.tan x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_trig_equation_l1880_188075


namespace NUMINAMATH_CALUDE_w_squared_value_l1880_188008

theorem w_squared_value (w : ℝ) (h : (2*w + 19)^2 = (4*w + 9)*(3*w + 13)) :
  w^2 = ((6 + Real.sqrt 524) / 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l1880_188008


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_sum_of_30th_and_29th_triangular_numbers_l1880_188047

-- Define the triangular number function
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem for the 30th triangular number
theorem thirtieth_triangular_number : triangularNumber 30 = 465 := by
  sorry

-- Theorem for the sum of 30th and 29th triangular numbers
theorem sum_of_30th_and_29th_triangular_numbers :
  triangularNumber 30 + triangularNumber 29 = 900 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_sum_of_30th_and_29th_triangular_numbers_l1880_188047


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1880_188073

theorem rectangle_perimeter (L B : ℝ) (h1 : L - B = 23) (h2 : L * B = 2030) :
  2 * (L + B) = 186 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1880_188073


namespace NUMINAMATH_CALUDE_weight_change_problem_l1880_188011

/-- Represents the scenario of replacing a man in a group and the resulting weight change -/
structure WeightChangeScenario where
  initial_count : ℕ
  initial_average : ℝ
  replaced_weight : ℝ
  new_weight : ℝ
  average_increase : ℝ

/-- The theorem representing the weight change problem -/
theorem weight_change_problem (scenario : WeightChangeScenario) 
  (h1 : scenario.initial_count = 10)
  (h2 : scenario.replaced_weight = 58)
  (h3 : scenario.average_increase = 2.5) :
  scenario.new_weight = 83 ∧ 
  ∀ (x : ℝ), ∃ (scenario' : WeightChangeScenario), 
    scenario'.initial_average = x ∧
    scenario'.initial_count = scenario.initial_count ∧
    scenario'.replaced_weight = scenario.replaced_weight ∧
    scenario'.new_weight = scenario.new_weight ∧
    scenario'.average_increase = scenario.average_increase :=
by sorry

end NUMINAMATH_CALUDE_weight_change_problem_l1880_188011


namespace NUMINAMATH_CALUDE_extraneous_root_value_l1880_188084

theorem extraneous_root_value (x m : ℝ) : 
  ((x + 7) / (x - 1) + 2 = (m + 5) / (x - 1)) ∧ 
  (x = 1) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_extraneous_root_value_l1880_188084


namespace NUMINAMATH_CALUDE_log_problem_l1880_188096

theorem log_problem (x y : ℝ) (h1 : Real.log (x * y^4) = 1) (h2 : Real.log (x^3 * y) = 1) :
  Real.log (x^3 * y^2) = 13/11 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l1880_188096


namespace NUMINAMATH_CALUDE_third_side_length_l1880_188099

/-- A scalene triangle with integer side lengths satisfying certain conditions -/
structure ScaleneTriangle where
  a : ℤ
  b : ℤ
  c : ℤ
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  condition : (a - 3)^2 + (b - 2)^2 = 0

/-- The third side of the triangle is either 2, 3, or 4 -/
theorem third_side_length (t : ScaleneTriangle) : t.c = 2 ∨ t.c = 3 ∨ t.c = 4 :=
  sorry

end NUMINAMATH_CALUDE_third_side_length_l1880_188099


namespace NUMINAMATH_CALUDE_choose_two_from_three_l1880_188082

theorem choose_two_from_three (n : ℕ) (k : ℕ) : n = 3 ∧ k = 2 → Nat.choose n k = 3 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_three_l1880_188082


namespace NUMINAMATH_CALUDE_probability_three_same_color_l1880_188057

def total_marbles : ℕ := 23
def red_marbles : ℕ := 6
def white_marbles : ℕ := 8
def blue_marbles : ℕ := 9

def probability_same_color : ℚ := 160 / 1771

theorem probability_three_same_color :
  probability_same_color = (Nat.choose red_marbles 3 + Nat.choose white_marbles 3 + Nat.choose blue_marbles 3) / Nat.choose total_marbles 3 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_same_color_l1880_188057


namespace NUMINAMATH_CALUDE_system_two_solutions_l1880_188078

/-- The system of equations has exactly two solutions when a is in the specified interval -/
theorem system_two_solutions (a b : ℝ) : 
  (∃ x y : ℝ, 
    Real.arcsin ((a - y) / 3) = Real.arcsin ((4 - x) / 4) ∧
    x^2 + y^2 - 8*x - 8*y = b) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    Real.arcsin ((a - y₁) / 3) = Real.arcsin ((4 - x₁) / 4) ∧
    x₁^2 + y₁^2 - 8*x₁ - 8*y₁ = b ∧
    Real.arcsin ((a - y₂) / 3) = Real.arcsin ((4 - x₂) / 4) ∧
    x₂^2 + y₂^2 - 8*x₂ - 8*y₂ = b ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
    ∀ x₃ y₃ : ℝ, 
      Real.arcsin ((a - y₃) / 3) = Real.arcsin ((4 - x₃) / 4) ∧
      x₃^2 + y₃^2 - 8*x₃ - 8*y₃ = b →
      (x₃ = x₁ ∧ y₃ = y₁) ∨ (x₃ = x₂ ∧ y₃ = y₂)) ↔
  -13/3 < a ∧ a < 37/3 :=
by sorry

end NUMINAMATH_CALUDE_system_two_solutions_l1880_188078


namespace NUMINAMATH_CALUDE_product_of_solutions_eq_neg_nine_l1880_188062

theorem product_of_solutions_eq_neg_nine :
  ∃ (z₁ z₂ : ℂ), z₁ ≠ z₂ ∧ 
  (Complex.abs z₁ = 3 * (Complex.abs z₁ - 2)) ∧
  (Complex.abs z₂ = 3 * (Complex.abs z₂ - 2)) ∧
  (z₁ * z₂ = -9) := by
sorry

end NUMINAMATH_CALUDE_product_of_solutions_eq_neg_nine_l1880_188062


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruence_solution_l1880_188034

theorem smallest_four_digit_congruence_solution :
  let x : ℕ := 1001
  (∀ y : ℕ, 1000 ≤ y ∧ y < x →
    ¬(11 * y ≡ 33 [ZMOD 22] ∧
      3 * y + 10 ≡ 19 [ZMOD 12] ∧
      5 * y - 3 ≡ 2 * y [ZMOD 36] ∧
      y ≡ 3 [ZMOD 4])) ∧
  (11 * x ≡ 33 [ZMOD 22] ∧
   3 * x + 10 ≡ 19 [ZMOD 12] ∧
   5 * x - 3 ≡ 2 * x [ZMOD 36] ∧
   x ≡ 3 [ZMOD 4]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruence_solution_l1880_188034


namespace NUMINAMATH_CALUDE_bottles_per_crate_l1880_188021

theorem bottles_per_crate 
  (total_bottles : ℕ) 
  (num_crates : ℕ) 
  (unpacked_bottles : ℕ) 
  (h1 : total_bottles = 130) 
  (h2 : num_crates = 10) 
  (h3 : unpacked_bottles = 10) 
  : (total_bottles - unpacked_bottles) / num_crates = 12 := by
  sorry

end NUMINAMATH_CALUDE_bottles_per_crate_l1880_188021


namespace NUMINAMATH_CALUDE_coffee_calculation_l1880_188019

/-- Calculates the total tablespoons of coffee needed for guests with different preferences -/
def total_coffee_tablespoons (total_guests : ℕ) : ℕ :=
  let weak_guests := total_guests / 3
  let medium_guests := total_guests / 3
  let strong_guests := total_guests - (weak_guests + medium_guests)
  let weak_cups := weak_guests * 2
  let medium_cups := medium_guests * 3
  let strong_cups := strong_guests * 1
  let weak_tablespoons := weak_cups * 1
  let medium_tablespoons := (medium_cups * 3) / 2
  let strong_tablespoons := strong_cups * 2
  weak_tablespoons + medium_tablespoons + strong_tablespoons

theorem coffee_calculation :
  total_coffee_tablespoons 18 = 51 := by
  sorry

end NUMINAMATH_CALUDE_coffee_calculation_l1880_188019


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1880_188035

theorem fraction_subtraction_simplification : 
  (9 : ℚ) / 19 - 5 / 57 - 2 / 38 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1880_188035


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_is_180_l1880_188043

/-- Triangle DEF with given side lengths -/
structure Triangle :=
  (DE : ℝ)
  (EF : ℝ)
  (FD : ℝ)

/-- Parallel lines intersecting the triangle -/
structure ParallelLines :=
  (m_D : ℝ)
  (m_E : ℝ)
  (m_F : ℝ)

/-- The perimeter of the inner triangle formed by parallel lines -/
def inner_triangle_perimeter (t : Triangle) (p : ParallelLines) : ℝ :=
  p.m_D + p.m_E + p.m_F

/-- Theorem stating the perimeter of the inner triangle -/
theorem inner_triangle_perimeter_is_180 
  (t : Triangle) 
  (p : ParallelLines) 
  (h1 : t.DE = 140) 
  (h2 : t.EF = 260) 
  (h3 : t.FD = 200) 
  (h4 : p.m_D = 65) 
  (h5 : p.m_E = 85) 
  (h6 : p.m_F = 30) : 
  inner_triangle_perimeter t p = 180 := by
  sorry

#check inner_triangle_perimeter_is_180

end NUMINAMATH_CALUDE_inner_triangle_perimeter_is_180_l1880_188043


namespace NUMINAMATH_CALUDE_evaluate_expression_l1880_188007

theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = 2) : 
  z * (z - 4 * x) = -28 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1880_188007


namespace NUMINAMATH_CALUDE_jessica_seashells_l1880_188095

/-- The number of seashells Jessica gave to Joan -/
def seashells_given : ℕ := 6

/-- The number of seashells Jessica kept -/
def seashells_kept : ℕ := 2

/-- The initial number of seashells Jessica found -/
def initial_seashells : ℕ := seashells_given + seashells_kept

theorem jessica_seashells : initial_seashells = 8 := by
  sorry

end NUMINAMATH_CALUDE_jessica_seashells_l1880_188095


namespace NUMINAMATH_CALUDE_star_product_six_equals_twentyfour_l1880_188030

/-- Custom operation definition -/
def star (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

/-- Theorem stating that if a * b = 6, then a ¤ b = 24 -/
theorem star_product_six_equals_twentyfour (a b : ℝ) (h : a * b = 6) : star a b = 24 := by
  sorry

end NUMINAMATH_CALUDE_star_product_six_equals_twentyfour_l1880_188030


namespace NUMINAMATH_CALUDE_line_always_intersects_circle_min_chord_length_shortest_chord_line_equation_l1880_188022

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the line l
def line_l (k x y : ℝ) : Prop := k*x - y - 4*k + 3 = 0

-- Theorem 1: Line l always intersects circle C
theorem line_always_intersects_circle :
  ∀ k : ℝ, ∃ x y : ℝ, circle_C x y ∧ line_l k x y :=
sorry

-- Theorem 2: Minimum chord length is 2√2
theorem min_chord_length :
  ∃ k : ℝ, ∀ x y : ℝ, 
    circle_C x y ∧ line_l k x y →
    ∃ x' y' : ℝ, circle_C x' y' ∧ line_l k x' y' ∧
    ((x - x')^2 + (y - y')^2)^(1/2) ≥ 2 * (2^(1/2)) :=
sorry

-- Theorem 3: Equation of the line with shortest chord
theorem shortest_chord_line_equation :
  ∃ k : ℝ, k = 1 ∧
  (∀ x y : ℝ, line_l k x y ↔ x - y - 1 = 0) ∧
  (∀ k' : ℝ, k' ≠ k →
    ∃ x y x' y' : ℝ,
      circle_C x y ∧ circle_C x' y' ∧
      line_l k x y ∧ line_l k x' y' ∧
      line_l k' x y ∧ line_l k' x' y' ∧
      (x - x')^2 + (y - y')^2 < (x - x')^2 + (y - y')^2) :=
sorry

end NUMINAMATH_CALUDE_line_always_intersects_circle_min_chord_length_shortest_chord_line_equation_l1880_188022


namespace NUMINAMATH_CALUDE_biff_hourly_rate_l1880_188051

/-- Biff's bus trip expenses and earnings -/
def biff_trip (hourly_rate : ℚ) : Prop :=
  let ticket : ℚ := 11
  let snacks : ℚ := 3
  let headphones : ℚ := 16
  let wifi_rate : ℚ := 2
  let trip_duration : ℚ := 3
  let total_expenses : ℚ := ticket + snacks + headphones + wifi_rate * trip_duration
  hourly_rate * trip_duration = total_expenses

/-- Theorem stating Biff's hourly rate for online work -/
theorem biff_hourly_rate : 
  ∃ (rate : ℚ), biff_trip rate ∧ rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_biff_hourly_rate_l1880_188051


namespace NUMINAMATH_CALUDE_parts_probability_theorem_l1880_188058

/-- Represents the outcome of drawing a part -/
inductive DrawOutcome
| Standard
| NonStandard

/-- Represents the type of part that was lost -/
inductive LostPart
| Standard
| NonStandard

/-- The probability model for the parts problem -/
structure PartsModel where
  initialStandard : ℕ
  initialNonStandard : ℕ
  lostPart : LostPart
  drawnPart : DrawOutcome

def PartsModel.totalInitial (m : PartsModel) : ℕ :=
  m.initialStandard + m.initialNonStandard

def PartsModel.remainingTotal (m : PartsModel) : ℕ :=
  m.totalInitial - 1

def PartsModel.remainingStandard (m : PartsModel) : ℕ :=
  match m.lostPart with
  | LostPart.Standard => m.initialStandard - 1
  | LostPart.NonStandard => m.initialStandard

def PartsModel.probability (m : PartsModel) (event : PartsModel → Prop) : ℚ :=
  sorry

theorem parts_probability_theorem (m : PartsModel) 
  (h1 : m.initialStandard = 21)
  (h2 : m.initialNonStandard = 10)
  (h3 : m.drawnPart = DrawOutcome.Standard) :
  (m.probability (fun model => model.lostPart = LostPart.Standard) = 2/3) ∧
  (m.probability (fun model => model.lostPart = LostPart.NonStandard) = 1/3) :=
sorry

end NUMINAMATH_CALUDE_parts_probability_theorem_l1880_188058


namespace NUMINAMATH_CALUDE_sequence_convergence_l1880_188059

def is_smallest_prime_divisor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ p ∣ n ∧ ∀ q, Nat.Prime q → q ∣ n → p ≤ q

def sequence_condition (a p : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0 ∧ p n > 0) ∧
  a 1 ≥ 2 ∧
  (∀ n, is_smallest_prime_divisor (p n) (a n)) ∧
  (∀ n, a (n + 1) = a n + a n / p n)

theorem sequence_convergence (a p : ℕ → ℕ) (h : sequence_condition a p) :
  ∃ N, ∀ n > N, a (n + 3) = 3 * a n := by
  sorry

end NUMINAMATH_CALUDE_sequence_convergence_l1880_188059


namespace NUMINAMATH_CALUDE_max_female_students_with_four_teachers_min_group_size_exists_min_group_l1880_188014

/-- Represents the composition of a study group --/
structure StudyGroup where
  male_students : ℕ
  female_students : ℕ
  teachers : ℕ

/-- Checks if a study group satisfies the given conditions --/
def is_valid_group (g : StudyGroup) : Prop :=
  g.male_students > g.female_students ∧
  g.female_students > g.teachers ∧
  2 * g.teachers > g.male_students

theorem max_female_students_with_four_teachers :
  ∀ g : StudyGroup,
  is_valid_group g → g.teachers = 4 →
  g.female_students ≤ 6 :=
sorry

theorem min_group_size :
  ∀ g : StudyGroup,
  is_valid_group g →
  g.male_students + g.female_students + g.teachers ≥ 12 :=
sorry

theorem exists_min_group :
  ∃ g : StudyGroup,
  is_valid_group g ∧
  g.male_students + g.female_students + g.teachers = 12 :=
sorry

end NUMINAMATH_CALUDE_max_female_students_with_four_teachers_min_group_size_exists_min_group_l1880_188014


namespace NUMINAMATH_CALUDE_bobbit_worm_aquarium_l1880_188044

def fish_count (initial_fish : ℕ) (daily_consumption : ℕ) (added_fish : ℕ) (days_before_adding : ℕ) (total_days : ℕ) : ℕ :=
  initial_fish - (daily_consumption * total_days) + added_fish

theorem bobbit_worm_aquarium (initial_fish : ℕ) (daily_consumption : ℕ) (added_fish : ℕ) (days_before_adding : ℕ) (total_days : ℕ)
  (h1 : initial_fish = 60)
  (h2 : daily_consumption = 2)
  (h3 : added_fish = 8)
  (h4 : days_before_adding = 14)
  (h5 : total_days = 21) :
  fish_count initial_fish daily_consumption added_fish days_before_adding total_days = 26 := by
  sorry

end NUMINAMATH_CALUDE_bobbit_worm_aquarium_l1880_188044


namespace NUMINAMATH_CALUDE_amusement_park_theorem_l1880_188006

/-- Represents the amusement park scenario with two roller coasters and a group of friends. -/
structure AmusementPark where
  friends : ℕ
  first_coaster_cost : ℕ
  second_coaster_cost : ℕ
  first_coaster_rides : ℕ
  second_coaster_rides : ℕ
  discount_rate : ℚ
  discount_threshold : ℕ

/-- Calculates the total number of tickets needed for the group. -/
def total_tickets (park : AmusementPark) : ℕ :=
  park.friends * (park.first_coaster_cost * park.first_coaster_rides + 
                  park.second_coaster_cost * park.second_coaster_rides)

/-- Calculates the cost difference between non-discounted and discounted tickets. -/
def cost_difference (park : AmusementPark) : ℚ :=
  let total_cost := total_tickets park
  if total_tickets park ≥ park.discount_threshold then
    (total_cost : ℚ) * park.discount_rate
  else
    0

/-- Theorem stating the correct number of tickets and cost difference for the given scenario. -/
theorem amusement_park_theorem (park : AmusementPark) 
  (h1 : park.friends = 8)
  (h2 : park.first_coaster_cost = 6)
  (h3 : park.second_coaster_cost = 8)
  (h4 : park.first_coaster_rides = 2)
  (h5 : park.second_coaster_rides = 1)
  (h6 : park.discount_rate = 15 / 100)
  (h7 : park.discount_threshold = 10) :
  total_tickets park = 160 ∧ cost_difference park = 24 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_theorem_l1880_188006


namespace NUMINAMATH_CALUDE_wall_length_proof_l1880_188037

/-- Proves that the length of a wall is 900 cm given specific brick and wall dimensions --/
theorem wall_length_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
                          (wall_height : ℝ) (wall_width : ℝ) (num_bricks : ℕ) :
  brick_length = 25 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_height = 600 →
  wall_width = 22.5 →
  num_bricks = 7200 →
  (brick_length * brick_width * brick_height * num_bricks) / (wall_height * wall_width) = 900 :=
by
  sorry

#check wall_length_proof

end NUMINAMATH_CALUDE_wall_length_proof_l1880_188037


namespace NUMINAMATH_CALUDE_mice_ratio_l1880_188026

theorem mice_ratio (white_mice brown_mice : ℕ) 
  (hw : white_mice = 14) 
  (hb : brown_mice = 7) : 
  (white_mice : ℚ) / (white_mice + brown_mice) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mice_ratio_l1880_188026


namespace NUMINAMATH_CALUDE_basketball_free_throws_l1880_188042

theorem basketball_free_throws :
  ∀ (a b x : ℚ),
  3 * b = 2 * a →
  x = 2 * a - 2 →
  2 * a + 3 * b + x = 78 →
  x = 74 / 3 := by
sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l1880_188042


namespace NUMINAMATH_CALUDE_a_sequence_square_values_l1880_188036

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | (n + 3) => a (n + 2) + a (n + 1) + a n

theorem a_sequence_square_values (n : ℕ) : 
  (n > 0 ∧ a (n - 1) = n^2) ↔ (n = 1 ∨ n = 9) := by
  sorry

#check a_sequence_square_values

end NUMINAMATH_CALUDE_a_sequence_square_values_l1880_188036


namespace NUMINAMATH_CALUDE_cost_price_percentage_l1880_188001

theorem cost_price_percentage (C S : ℝ) (h : C > 0) (h' : S > 0) :
  (S - C) / C = 3 → C / S = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l1880_188001


namespace NUMINAMATH_CALUDE_no_natural_number_power_of_two_l1880_188017

theorem no_natural_number_power_of_two : 
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), n^2012 - 1 = 2^k := by
  sorry

end NUMINAMATH_CALUDE_no_natural_number_power_of_two_l1880_188017


namespace NUMINAMATH_CALUDE_xiao_jun_age_problem_l1880_188091

/-- Represents the current age of Xiao Jun -/
def xiao_jun_age : ℕ := 6

/-- Represents the current age ratio between Xiao Jun's mother and Xiao Jun -/
def current_age_ratio : ℕ := 5

/-- Represents the future age ratio between Xiao Jun's mother and Xiao Jun -/
def future_age_ratio : ℕ := 3

/-- Calculates the number of years that need to pass for Xiao Jun's mother's age 
    to be 3 times Xiao Jun's age -/
def years_passed : ℕ := 6

theorem xiao_jun_age_problem : 
  xiao_jun_age * current_age_ratio + years_passed = 
  (xiao_jun_age + years_passed) * future_age_ratio :=
sorry

end NUMINAMATH_CALUDE_xiao_jun_age_problem_l1880_188091


namespace NUMINAMATH_CALUDE_centroid_coordinates_specific_triangle_centroid_l1880_188055

/-- The centroid of a triangle is located at the arithmetic mean of its vertices. -/
theorem centroid_coordinates (A B C : ℝ × ℝ) :
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  A = (-1, 3) → B = (1, 2) → C = (2, -5) → G = (2/3, 0) := by
  sorry

/-- The centroid of the specific triangle ABC is at (2/3, 0). -/
theorem specific_triangle_centroid :
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (1, 2)
  let C : ℝ × ℝ := (2, -5)
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  G = (2/3, 0) := by
  sorry

end NUMINAMATH_CALUDE_centroid_coordinates_specific_triangle_centroid_l1880_188055


namespace NUMINAMATH_CALUDE_triangles_in_50th_ring_l1880_188065

/-- The number of unit triangles in the nth ring of a triangular array -/
def triangles_in_ring (n : ℕ) : ℕ := 9 + 6 * (n - 1)

/-- The number of unit triangles in the 50th ring is 303 -/
theorem triangles_in_50th_ring : triangles_in_ring 50 = 303 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_50th_ring_l1880_188065


namespace NUMINAMATH_CALUDE_exists_valid_numbering_scheme_l1880_188077

/-- Represents a numbering scheme for 7 pins and 7 holes -/
def NumberingScheme := Fin 7 → Fin 7

/-- Checks if a numbering scheme satisfies the condition for a given rotation -/
def isValidForRotation (scheme : NumberingScheme) (rotation : Fin 7) : Prop :=
  ∃ k : Fin 7, scheme k = (k + rotation : Fin 7)

/-- The main theorem stating that there exists a valid numbering scheme -/
theorem exists_valid_numbering_scheme :
  ∃ scheme : NumberingScheme, ∀ rotation : Fin 7, isValidForRotation scheme rotation := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_numbering_scheme_l1880_188077


namespace NUMINAMATH_CALUDE_seventh_term_value_l1880_188074

def sequence_with_sum_rule (a : ℕ → ℕ) : Prop :=
  a 1 = 5 ∧ a 4 = 13 ∧ a 6 = 40 ∧
  ∀ n ≥ 4, a n = a (n-3) + a (n-2) + a (n-1)

theorem seventh_term_value (a : ℕ → ℕ) (h : sequence_with_sum_rule a) : a 7 = 74 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_value_l1880_188074


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l1880_188028

-- Define the two lines
def line1 (x : ℝ) : ℝ := 3 * x + 4
def line2 (x y : ℝ) : Prop := 3 * x + y = 25

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line2 x y ∧ y = line1 x ∧ x = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l1880_188028


namespace NUMINAMATH_CALUDE_eight_p_plus_one_composite_l1880_188064

theorem eight_p_plus_one_composite (p : ℕ) (h1 : Nat.Prime p) (h2 : Nat.Prime (8 * p - 1)) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = 8 * p + 1 :=
by sorry

end NUMINAMATH_CALUDE_eight_p_plus_one_composite_l1880_188064


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_of_cubes_l1880_188020

theorem consecutive_even_integers_sum_of_cubes (x y z : ℤ) : 
  (∃ n : ℤ, x = 2*n ∧ y = 2*n + 2 ∧ z = 2*n + 4) →  -- consecutive even integers
  x^2 + y^2 + z^2 = 2960 →                         -- sum of squares is 2960
  x^3 + y^3 + z^3 = 90117 :=                       -- sum of cubes is 90117
by sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_of_cubes_l1880_188020


namespace NUMINAMATH_CALUDE_exist_six_consecutive_naturals_lcm_property_l1880_188049

theorem exist_six_consecutive_naturals_lcm_property :
  ∃ n : ℕ, lcm (lcm n (n + 1)) (n + 2) > lcm (lcm (n + 3) (n + 4)) (n + 5) := by
  sorry

end NUMINAMATH_CALUDE_exist_six_consecutive_naturals_lcm_property_l1880_188049


namespace NUMINAMATH_CALUDE_new_person_weight_is_77_l1880_188054

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (total_persons : ℕ) (average_weight_increase : ℝ) (replaced_person_weight : ℝ) : ℝ :=
  replaced_person_weight + total_persons * average_weight_increase

/-- Theorem stating that the weight of the new person is 77 kg given the problem conditions -/
theorem new_person_weight_is_77 :
  weight_of_new_person 8 1.5 65 = 77 := by
  sorry

#eval weight_of_new_person 8 1.5 65

end NUMINAMATH_CALUDE_new_person_weight_is_77_l1880_188054


namespace NUMINAMATH_CALUDE_good_carrots_count_l1880_188098

/-- Given that Carol picked 29 carrots, her mother picked 16 carrots, and they had 7 bad carrots,
    prove that the number of good carrots is 38. -/
theorem good_carrots_count (carol_carrots : ℕ) (mom_carrots : ℕ) (bad_carrots : ℕ)
    (h1 : carol_carrots = 29)
    (h2 : mom_carrots = 16)
    (h3 : bad_carrots = 7) :
    carol_carrots + mom_carrots - bad_carrots = 38 := by
  sorry

end NUMINAMATH_CALUDE_good_carrots_count_l1880_188098


namespace NUMINAMATH_CALUDE_problem_statement_l1880_188066

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x - Real.sqrt 3 / 2

theorem problem_statement :
  (∃ (M : ℝ), M = 1 ∧ ∀ x ∈ Set.Ioo 0 (Real.pi / 2), f x ≤ M) ∧
  (∀ A B C : ℝ, 0 < A ∧ A < B ∧ B < C ∧ A + B + C = Real.pi →
    f A = 1/2 → f B = 1/2 → Real.sin C / Real.sin A = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1880_188066


namespace NUMINAMATH_CALUDE_pencil_distribution_ways_l1880_188068

/-- The number of ways to distribute n identical objects into k distinct groups --/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute pencils among friends --/
def distributePencils (totalPencils friendCount minPencils : ℕ) : ℕ :=
  let remainingPencils := totalPencils - friendCount * minPencils
  starsAndBars remainingPencils friendCount

theorem pencil_distribution_ways :
  distributePencils 12 4 2 = 35 := by
  sorry

#eval distributePencils 12 4 2

end NUMINAMATH_CALUDE_pencil_distribution_ways_l1880_188068


namespace NUMINAMATH_CALUDE_polynomial_equality_l1880_188072

theorem polynomial_equality (x t s : ℝ) : 
  (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + s) = 
  15 * x^4 - 22 * x^3 + (41 + s) * x^2 - 34 * x + 9 * s ↔ 
  t = -2 ∧ s = s := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1880_188072


namespace NUMINAMATH_CALUDE_leftover_tarts_l1880_188003

/-- The number of leftover tarts in a restaurant, given the fractions of different flavored tarts. -/
theorem leftover_tarts (cherry : ℝ) (blueberry : ℝ) (peach : ℝ) 
  (h_cherry : cherry = 0.08)
  (h_blueberry : blueberry = 0.75)
  (h_peach : peach = 0.08) :
  cherry + blueberry + peach = 0.91 := by
  sorry

end NUMINAMATH_CALUDE_leftover_tarts_l1880_188003


namespace NUMINAMATH_CALUDE_jeans_price_ratio_l1880_188000

theorem jeans_price_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let sale_price := marked_price / 2
  let cost := (5 / 8) * sale_price
  cost / marked_price = 5 / 16 := by
sorry

end NUMINAMATH_CALUDE_jeans_price_ratio_l1880_188000


namespace NUMINAMATH_CALUDE_product_of_positive_real_solutions_l1880_188086

theorem product_of_positive_real_solutions (x : ℂ) : 
  (x^6 = -729) → 
  (∃ (S : Finset ℂ), 
    (∀ z ∈ S, z^6 = -729 ∧ z.re > 0) ∧ 
    (∀ z, z^6 = -729 ∧ z.re > 0 → z ∈ S) ∧
    (S.prod id = 9)) := by
sorry

end NUMINAMATH_CALUDE_product_of_positive_real_solutions_l1880_188086


namespace NUMINAMATH_CALUDE_parallelogram_base_l1880_188040

theorem parallelogram_base (height area : ℝ) (h1 : height = 32) (h2 : area = 896) : 
  area / height = 28 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l1880_188040


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l1880_188052

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 5 - |x + a| - |x - 2|

theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

theorem range_of_a_when_f_leq_one :
  {a : ℝ | ∀ x, f a x ≤ 1} = {a : ℝ | a ≤ -6 ∨ a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_f_leq_one_l1880_188052


namespace NUMINAMATH_CALUDE_highest_seat_number_is_44_l1880_188085

/-- Calculates the highest seat number in a systematic sample -/
def highest_seat_number (total_students : ℕ) (sample_size : ℕ) (first_student : ℕ) : ℕ :=
  let interval := total_students / sample_size
  first_student + (sample_size - 1) * interval

/-- Theorem: The highest seat number in the sample is 44 -/
theorem highest_seat_number_is_44 :
  highest_seat_number 56 4 2 = 44 := by
  sorry

#eval highest_seat_number 56 4 2

end NUMINAMATH_CALUDE_highest_seat_number_is_44_l1880_188085


namespace NUMINAMATH_CALUDE_min_abs_sum_with_constraints_l1880_188079

theorem min_abs_sum_with_constraints (α β γ : ℝ) 
  (sum_constraint : α + β + γ = 2)
  (product_constraint : α * β * γ = 4) :
  ∃ v : ℝ, v = 6 ∧ ∀ α' β' γ' : ℝ, 
    (α' + β' + γ' = 2) → (α' * β' * γ' = 4) → 
    v ≤ |α'| + |β'| + |γ'| :=
by sorry

end NUMINAMATH_CALUDE_min_abs_sum_with_constraints_l1880_188079


namespace NUMINAMATH_CALUDE_smallest_k_inequality_l1880_188067

theorem smallest_k_inequality (k : ℝ) : k = 1 ↔ 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + k * (x - y)^2 ≥ Real.sqrt (x^2 + y^2)) ∧ 
  (∀ k' : ℝ, k' > 0 → k' < k → ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ Real.sqrt (x * y) + k' * (x - y)^2 < Real.sqrt (x^2 + y^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_inequality_l1880_188067


namespace NUMINAMATH_CALUDE_power_sum_theorem_l1880_188061

/-- Given four real numbers a, b, c, d satisfying certain conditions,
    prove statements about their powers. -/
theorem power_sum_theorem (a b c d : ℝ) 
  (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  (∃ a b c d : ℝ, (a + b = c + d ∧ a^3 + b^3 = c^3 + d^3 ∧ a^4 + b^4 ≠ c^4 + d^4)) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_theorem_l1880_188061


namespace NUMINAMATH_CALUDE_initial_money_calculation_l1880_188032

theorem initial_money_calculation (clothes_percent grocery_percent electronics_percent dining_percent : ℚ)
  (remaining_money : ℚ) :
  clothes_percent = 20 / 100 →
  grocery_percent = 15 / 100 →
  electronics_percent = 10 / 100 →
  dining_percent = 5 / 100 →
  remaining_money = 15700 →
  ∃ initial_money : ℚ, 
    initial_money * (1 - (clothes_percent + grocery_percent + electronics_percent + dining_percent)) = remaining_money ∧
    initial_money = 31400 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l1880_188032


namespace NUMINAMATH_CALUDE_z_minus_two_purely_imaginary_l1880_188060

def z : ℂ := Complex.mk 2 (-1)

theorem z_minus_two_purely_imaginary :
  Complex.im (z - 2) = Complex.im z ∧ Complex.re (z - 2) = 0 :=
sorry

end NUMINAMATH_CALUDE_z_minus_two_purely_imaginary_l1880_188060


namespace NUMINAMATH_CALUDE_square_remainder_sum_quotient_l1880_188031

theorem square_remainder_sum_quotient : 
  let squares := List.map (fun n => n^2) (List.range 6)
  let remainders := List.map (fun x => x % 13) squares
  let distinct_remainders := List.eraseDups remainders
  let m := distinct_remainders.sum
  m / 13 = 3 := by
sorry

end NUMINAMATH_CALUDE_square_remainder_sum_quotient_l1880_188031


namespace NUMINAMATH_CALUDE_right_triangle_area_l1880_188029

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 10 →
  angle = 30 * π / 180 →
  let shorter_leg := hypotenuse / 2
  let longer_leg := shorter_leg * Real.sqrt 3
  let area := (shorter_leg * longer_leg) / 2
  area = (25 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1880_188029


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1880_188050

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 - 2*i) / (1 + 2*i) = -3/5 - 4/5*i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1880_188050


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1880_188046

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 3*I
  let z₂ : ℂ := -1 + 3*I
  z₁ / z₂ = (-1.2 : ℝ) - 1.2*I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1880_188046


namespace NUMINAMATH_CALUDE_fraction_equality_l1880_188038

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 2) 
  (h2 : s / u = 7 / 11) : 
  (5 * p * s - 3 * q * u) / (7 * q * u - 4 * p * s) = 109 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1880_188038


namespace NUMINAMATH_CALUDE_marbles_given_to_joan_l1880_188081

theorem marbles_given_to_joan (initial_marbles : ℝ) (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 9.0) 
  (h2 : remaining_marbles = 6) :
  initial_marbles - remaining_marbles = 3 := by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_joan_l1880_188081


namespace NUMINAMATH_CALUDE_max_profit_is_850_l1880_188092

def fruit_problem (m : ℝ) : Prop :=
  let total_weight : ℝ := 200
  let profit_A : ℝ := 20 - 16
  let profit_B : ℝ := 25 - 20
  let total_profit : ℝ := m * profit_A + (total_weight - m) * profit_B
  0 ≤ m ∧ m ≤ total_weight ∧ m ≥ 3 * (total_weight - m) →
  total_profit ≤ 850

theorem max_profit_is_850 :
  ∃ m : ℝ, fruit_problem m ∧
  (∀ n : ℝ, fruit_problem n → 
    m * (20 - 16) + (200 - m) * (25 - 20) ≥ n * (20 - 16) + (200 - n) * (25 - 20)) :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_850_l1880_188092


namespace NUMINAMATH_CALUDE_museum_admission_difference_l1880_188004

theorem museum_admission_difference (men women free_admission : ℕ) 
  (h1 : men = 194)
  (h2 : women = 235)
  (h3 : free_admission = 68) :
  (men + women) - free_admission - free_admission = 293 := by
  sorry

end NUMINAMATH_CALUDE_museum_admission_difference_l1880_188004


namespace NUMINAMATH_CALUDE_circular_garden_radius_l1880_188013

/-- 
Theorem: For a circular garden with radius r, if the length of the fence (circumference) 
is 1/4 of the area of the garden, then r = 8.
-/
theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1/4) * π * r^2 → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l1880_188013


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1880_188080

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x + 1) / (x - 1)) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1880_188080


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1880_188015

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 3}

-- Theorem statement
theorem intersection_A_complement_B : 
  A ∩ (U \ B) = {x : ℝ | -2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1880_188015


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l1880_188010

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_t_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, t)
  parallel a b → t = -4 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l1880_188010


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l1880_188053

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_condition : a + b + c = 13) 
  (product_sum_condition : a * b + a * c + b * c = 40) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 637 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l1880_188053


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_228_l1880_188018

/-- A trapezoid with given side lengths -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  EG : ℝ
  FH : ℝ
  h_EF_longer : EF > GH

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.EF + t.GH + t.EG + t.FH

/-- Theorem stating that the perimeter of the given trapezoid is 228 -/
theorem trapezoid_perimeter_is_228 : 
  ∀ (t : Trapezoid), t.EF = 90 ∧ t.GH = 40 ∧ t.EG = 53 ∧ t.FH = 45 → perimeter t = 228 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_228_l1880_188018


namespace NUMINAMATH_CALUDE_matrix_is_square_iff_a_eq_zero_l1880_188027

def A (a : ℚ) : Matrix (Fin 4) (Fin 4) ℚ :=
  !![a,   -a,  -1,   0;
     a,   -a,   0,  -1;
     1,    0,   a,  -a;
     0,    1,   a,  -a]

theorem matrix_is_square_iff_a_eq_zero (a : ℚ) :
  (∃ C : Matrix (Fin 4) (Fin 4) ℚ, A a = C ^ 2) ↔ a = 0 := by sorry

end NUMINAMATH_CALUDE_matrix_is_square_iff_a_eq_zero_l1880_188027


namespace NUMINAMATH_CALUDE_raft_sticks_total_l1880_188093

def simon_sticks : ℕ := 36

def gerry_sticks : ℕ := (2 * simon_sticks) / 3

def micky_sticks : ℕ := simon_sticks + gerry_sticks + 9

def darryl_sticks : ℕ := simon_sticks + gerry_sticks + micky_sticks + 1

def total_sticks : ℕ := simon_sticks + gerry_sticks + micky_sticks + darryl_sticks

theorem raft_sticks_total : total_sticks = 259 := by
  sorry

end NUMINAMATH_CALUDE_raft_sticks_total_l1880_188093


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_expression_l1880_188069

/-- Given two vectors a and b in R², prove that if they are parallel,
    then a specific trigonometric expression involving their components equals 1/3. -/
theorem parallel_vectors_trig_expression (α : ℝ) :
  let a : ℝ × ℝ := (1, Real.sin α)
  let b : ℝ × ℝ := (2, Real.cos α)
  (∃ (k : ℝ), a = k • b) →
  (Real.cos α - Real.sin α) / (2 * Real.cos (-α) - Real.sin α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_expression_l1880_188069


namespace NUMINAMATH_CALUDE_predictor_variable_is_fertilizer_l1880_188039

/-- Represents a variable in the study -/
inductive StudyVariable
  | YieldOfCrops
  | AmountOfFertilizer
  | Experimenter
  | OtherVariables

/-- Defines the characteristics of the study -/
structure CropStudy where
  predictedVariable : StudyVariable
  predictorVariable : StudyVariable
  aim : String

/-- Theorem stating that the predictor variable in the crop yield study is the amount of fertilizer -/
theorem predictor_variable_is_fertilizer (study : CropStudy) :
  study.aim = "determine whether the yield of crops can be predicted based on the amount of fertilizer applied" →
  study.predictedVariable = StudyVariable.YieldOfCrops →
  study.predictorVariable = StudyVariable.AmountOfFertilizer :=
by sorry

end NUMINAMATH_CALUDE_predictor_variable_is_fertilizer_l1880_188039


namespace NUMINAMATH_CALUDE_initial_number_proof_l1880_188002

theorem initial_number_proof (N : ℕ) : 
  (∀ k < 5, ¬ (23 ∣ (N + k))) → 
  (23 ∣ (N + 5)) → 
  N = 18 :=
by sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1880_188002


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l1880_188070

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 1) (hb : b > 2) (hab : a * b = 2 * a + b) :
  (∀ x y : ℝ, x > 1 ∧ y > 2 ∧ x * y = 2 * x + y → a + b ≤ x + y) ∧ a + b = 2 * Real.sqrt 2 + 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l1880_188070


namespace NUMINAMATH_CALUDE_range_of_a_l1880_188023

def prop_A (a : ℝ) : Prop :=
  ∀ x, x^2 + (a - 1) * x + a^2 > 0

def prop_B (a : ℝ) : Prop :=
  ∀ x y, x < y → (2 * a^2 - a)^x < (2 * a^2 - a)^y

def exclusive_or (P Q : Prop) : Prop :=
  (P ∧ ¬Q) ∨ (¬P ∧ Q)

theorem range_of_a : 
  {a : ℝ | exclusive_or (prop_A a) (prop_B a)} = 
  {a : ℝ | -1 ≤ a ∧ a < -1/2 ∨ 1/3 < a ∧ a ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1880_188023


namespace NUMINAMATH_CALUDE_binomial_square_constant_l1880_188025

theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 60*x + c = (x + a)^2) → c = 900 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l1880_188025


namespace NUMINAMATH_CALUDE_simplify_power_expression_l1880_188076

theorem simplify_power_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l1880_188076


namespace NUMINAMATH_CALUDE_female_workers_count_l1880_188071

/-- Represents the number of workers of each type and their wages --/
structure WorkforceData where
  male_workers : ℕ
  child_workers : ℕ
  male_wage : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the number of female workers based on the given workforce data --/
def calculate_female_workers (data : WorkforceData) : ℕ :=
  sorry

/-- Theorem stating that the number of female workers is 15 --/
theorem female_workers_count (data : WorkforceData) 
  (h1 : data.male_workers = 20)
  (h2 : data.child_workers = 5)
  (h3 : data.male_wage = 35)
  (h4 : data.female_wage = 20)
  (h5 : data.child_wage = 8)
  (h6 : data.average_wage = 26) :
  calculate_female_workers data = 15 := by
  sorry

end NUMINAMATH_CALUDE_female_workers_count_l1880_188071


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1880_188045

/-- The coefficient of x^2 in the expansion of (2x+1)^5 is 40 -/
theorem coefficient_x_squared_in_expansion : 
  (Finset.range 6).sum (fun k => 
    Nat.choose 5 k * (2^(5-k)) * (1^k) * 
    if 5 - k = 2 then 1 else 0) = 40 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1880_188045


namespace NUMINAMATH_CALUDE_dye_making_water_amount_l1880_188012

/-- Given a dye-making process where:
  * The total mixture is 27 liters
  * 5/6 of 18 liters of vinegar is used
  * The water used is 3/5 of the total water available
  Prove that the amount of water used is 12 liters -/
theorem dye_making_water_amount (total_mixture : ℝ) (vinegar_amount : ℝ) (water_fraction : ℝ) :
  total_mixture = 27 →
  vinegar_amount = 5 / 6 * 18 →
  water_fraction = 3 / 5 →
  total_mixture - vinegar_amount = 12 :=
by sorry

end NUMINAMATH_CALUDE_dye_making_water_amount_l1880_188012


namespace NUMINAMATH_CALUDE_a_closed_form_l1880_188033

def a : ℕ → ℤ
  | 0 => -1
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + 3 * a n + 3^(n + 2)

theorem a_closed_form (n : ℕ) :
  a n = (1 / 16) * ((4 * n - 3) * 3^(n + 1) - 7 * (-1)^n) :=
by sorry

end NUMINAMATH_CALUDE_a_closed_form_l1880_188033


namespace NUMINAMATH_CALUDE_extreme_value_implies_params_l1880_188005

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := a * x^3 + b * x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem extreme_value_implies_params (a b : ℝ) :
  (f a b 1 = -2) ∧ (f' a b 1 = 0) → a = 1 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_params_l1880_188005


namespace NUMINAMATH_CALUDE_height_difference_l1880_188024

/-- Given heights of Jana, Jess, and Kelly, prove the height difference between Jess and Kelly. -/
theorem height_difference (jana_height jess_height : ℕ) : 
  jana_height = 74 →
  jess_height = 72 →
  ∃ kelly_height : ℕ, 
    jana_height = kelly_height + 5 ∧ 
    jess_height - kelly_height = 3 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l1880_188024


namespace NUMINAMATH_CALUDE_equation_real_roots_l1880_188063

theorem equation_real_roots (a : ℝ) : 
  (∃ x : ℝ, 9^(-|x - 2|) - 4 * 3^(-|x - 2|) - a = 0) ↔ -3 ≤ a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_real_roots_l1880_188063


namespace NUMINAMATH_CALUDE_valid_selections_count_l1880_188048

/-- The number of ways to select 4 out of 6 people to visit 4 distinct places -/
def total_selections : ℕ := 360

/-- The number of ways where person A visits the restricted place -/
def a_restricted : ℕ := 60

/-- The number of ways where person B visits the restricted place -/
def b_restricted : ℕ := 60

/-- The number of valid selection schemes -/
def valid_selections : ℕ := total_selections - a_restricted - b_restricted

theorem valid_selections_count : valid_selections = 240 := by sorry

end NUMINAMATH_CALUDE_valid_selections_count_l1880_188048


namespace NUMINAMATH_CALUDE_zoo_population_increase_l1880_188088

theorem zoo_population_increase (c p : ℕ) (h1 : c * 3 = p) (h2 : (c + 2) * 3 = p + 6) : True :=
by sorry

end NUMINAMATH_CALUDE_zoo_population_increase_l1880_188088


namespace NUMINAMATH_CALUDE_inequality_proof_l1880_188041

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / c + c / b ≥ 4 * a / (a + b) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1880_188041


namespace NUMINAMATH_CALUDE_smallest_non_even_units_digit_l1880_188097

def EvenUnitsDigits : Set Nat := {0, 2, 4, 6, 8}

theorem smallest_non_even_units_digit : 
  (∀ d : Nat, d < 10 → d ∉ EvenUnitsDigits → 1 ≤ d) ∧ 1 ∉ EvenUnitsDigits := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_even_units_digit_l1880_188097


namespace NUMINAMATH_CALUDE_parallel_lines_probability_l1880_188089

/-- The number of points (centers of cube faces) -/
def num_points : ℕ := 6

/-- The number of ways to select 2 points from num_points -/
def num_lines : ℕ := num_points.choose 2

/-- The total number of ways for two people to each select a line -/
def total_selections : ℕ := num_lines * num_lines

/-- The number of pairs of lines that are parallel but not coincident -/
def parallel_pairs : ℕ := 12

/-- The probability of selecting two parallel but not coincident lines -/
def probability : ℚ := parallel_pairs / total_selections

theorem parallel_lines_probability :
  probability = 4 / 75 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_probability_l1880_188089
