import Mathlib

namespace sunnydale_farm_arrangement_l3416_341695

/-- The number of ways to arrange animals in a row -/
def arrange_animals (chickens dogs cats rabbits : ℕ) : ℕ :=
  Nat.factorial 4 * Nat.factorial chickens * Nat.factorial dogs * Nat.factorial cats * Nat.factorial rabbits

/-- Theorem stating the number of arrangements for the given animal counts -/
theorem sunnydale_farm_arrangement :
  arrange_animals 5 3 4 3 = 2488320 :=
by sorry

end sunnydale_farm_arrangement_l3416_341695


namespace correct_sample_sizes_l3416_341697

def model1_production : ℕ := 1600
def model2_production : ℕ := 6000
def model3_production : ℕ := 2000
def total_sample_size : ℕ := 48

theorem correct_sample_sizes :
  let total_production := model1_production + model2_production + model3_production
  let sample1 := (model1_production * total_sample_size) / total_production
  let sample2 := (model2_production * total_sample_size) / total_production
  let sample3 := (model3_production * total_sample_size) / total_production
  sample1 = 8 ∧ sample2 = 30 ∧ sample3 = 10 :=
by sorry

end correct_sample_sizes_l3416_341697


namespace sufficient_condition_for_f_less_than_one_l3416_341601

theorem sufficient_condition_for_f_less_than_one
  (a : ℝ) (h_a : a > 1) :
  ∃ (x : ℝ), -1 < x ∧ x < 0 ∧ a * x + 2 * x < 1 :=
sorry

end sufficient_condition_for_f_less_than_one_l3416_341601


namespace chanhee_walking_distance_l3416_341609

/-- Calculates the total distance walked given step length, duration, and pace. -/
def total_distance (step_length : Real) (duration : Real) (pace : Real) : Real :=
  step_length * duration * pace

/-- Proves that Chanhee walked 526.5 meters given the specified conditions. -/
theorem chanhee_walking_distance :
  let step_length : Real := 0.45
  let duration : Real := 13
  let pace : Real := 90
  total_distance step_length duration pace = 526.5 := by
  sorry

end chanhee_walking_distance_l3416_341609


namespace cone_rotation_ratio_l3416_341635

theorem cone_rotation_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  2 * π * Real.sqrt (r^2 + h^2) = 20 * π * r →
  h / r = Real.sqrt 399 := by
sorry

end cone_rotation_ratio_l3416_341635


namespace binomial_17_5_l3416_341689

theorem binomial_17_5 (h1 : Nat.choose 15 3 = 455)
                      (h2 : Nat.choose 15 4 = 1365)
                      (h3 : Nat.choose 15 5 = 3003) :
  Nat.choose 17 5 = 6188 := by
  sorry

end binomial_17_5_l3416_341689


namespace probability_all_green_apples_l3416_341638

/-- The probability of selecting all green apples when choosing 3 out of 10 apples, 
    given that there are 4 green apples. -/
theorem probability_all_green_apples (total : Nat) (green : Nat) (choose : Nat) : 
  total = 10 → green = 4 → choose = 3 → 
  (Nat.choose green choose : Rat) / (Nat.choose total choose) = 1 / 30 := by
  sorry

end probability_all_green_apples_l3416_341638


namespace gear_rotation_problem_l3416_341611

/-- 
Given two gears p and q rotating at constant speeds:
- q makes 40 revolutions per minute
- After 4 seconds, q has made exactly 2 more revolutions than p
Prove that p makes 10 revolutions per minute
-/
theorem gear_rotation_problem (p q : ℝ) 
  (hq : q = 40) -- q makes 40 revolutions per minute
  (h_diff : q * 4 / 60 = p * 4 / 60 + 2) -- After 4 seconds, q has made 2 more revolutions than p
  : p = 10 := by sorry

end gear_rotation_problem_l3416_341611


namespace school_experiment_l3416_341671

theorem school_experiment (boys girls : ℕ) (h1 : boys = 100) (h2 : girls = 125) : 
  (girls - boys) / girls * 100 = 20 ∧ (girls - boys) / boys * 100 = 25 := by
  sorry

end school_experiment_l3416_341671


namespace sum_through_base3_l3416_341612

/-- Converts a natural number from base 10 to base 3 --/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a number from base 3 (represented as a list of digits) to base 10 --/
def fromBase3 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base 3 (represented as lists of digits) --/
def addBase3 (a b : List ℕ) : List ℕ :=
  sorry

/-- Theorem stating that the sum of 10 and 23 in base 10 is equal to 33
    when performed through base 3 conversion and addition --/
theorem sum_through_base3 :
  fromBase3 (addBase3 (toBase3 10) (toBase3 23)) = 33 :=
by
  sorry

end sum_through_base3_l3416_341612


namespace b_nonempty_implies_a_geq_two_thirds_a_intersect_b_eq_b_implies_a_geq_two_l3416_341655

-- Define sets A and B
def A : Set ℝ := {x | x ≥ 1}
def B (a : ℝ) : Set ℝ := {x | (1/2) * a ≤ x ∧ x ≤ 2*a - 1}

-- Theorem 1: If B is non-empty, then a ≥ 2/3
theorem b_nonempty_implies_a_geq_two_thirds (a : ℝ) :
  (B a).Nonempty → a ≥ 2/3 := by sorry

-- Theorem 2: If A ∩ B = B, then a ≥ 2
theorem a_intersect_b_eq_b_implies_a_geq_two (a : ℝ) :
  A ∩ (B a) = B a → a ≥ 2 := by sorry

end b_nonempty_implies_a_geq_two_thirds_a_intersect_b_eq_b_implies_a_geq_two_l3416_341655


namespace money_distribution_problem_l3416_341629

/-- Represents the shares of P, Q, and R in the money distribution problem. -/
structure Shares where
  p : ℕ
  q : ℕ
  r : ℕ

/-- Represents the problem constraints and solution. -/
theorem money_distribution_problem (s : Shares) : 
  -- The ratio condition
  s.p + s.q + s.r > 0 ∧ 
  3 * s.q = 7 * s.p ∧ 
  3 * s.r = 4 * s.q ∧ 
  -- The difference between P and Q's shares
  s.q - s.p = 2800 ∧ 
  -- Total amount condition
  50000 ≤ s.p + s.q + s.r ∧ 
  s.p + s.q + s.r ≤ 75000 ∧ 
  -- Minimum and maximum share conditions
  s.p ≥ 5000 ∧ 
  s.r ≤ 45000 
  -- The difference between Q and R's shares
  → s.r - s.q = 14000 := by sorry

end money_distribution_problem_l3416_341629


namespace swimmer_speed_in_still_water_l3416_341617

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmerSpeed : ℝ
  streamSpeed : ℝ

/-- Calculates the effective speed when swimming downstream. -/
def downstreamSpeed (s : SwimmerSpeed) : ℝ :=
  s.swimmerSpeed + s.streamSpeed

/-- Calculates the effective speed when swimming upstream. -/
def upstreamSpeed (s : SwimmerSpeed) : ℝ :=
  s.swimmerSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 7 km/h. -/
theorem swimmer_speed_in_still_water :
  ∀ (s : SwimmerSpeed),
    (downstreamSpeed s * 4 = 32) →
    (upstreamSpeed s * 4 = 24) →
    s.swimmerSpeed = 7 :=
by sorry

end swimmer_speed_in_still_water_l3416_341617


namespace train_crossing_time_l3416_341606

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 240 → 
  train_speed_kmh = 144 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 6 := by
  sorry

end train_crossing_time_l3416_341606


namespace milk_processing_time_l3416_341670

/-- Given two milk processing plants with the following conditions:
    - They process equal amounts of milk
    - The second plant starts 'a' days later than the first
    - The second plant processes 'm' liters more per day than the first
    - After 5a/9 days of joint work, 1/3 of the task remains
    - The work finishes simultaneously
    - Each plant processes half of the total volume

    Prove that the total number of days required to complete the task is 2a
-/
theorem milk_processing_time (a m : ℝ) (a_pos : 0 < a) (m_pos : 0 < m) : 
  ∃ (n : ℝ), n > 0 ∧ 
  (∃ (x : ℝ), x > 0 ∧ 
    (n * x = (n - a) * (x + m)) ∧ 
    (a * x + (5 * a / 9) * (2 * x + m) = 2 / 3) ∧
    (n * x = 1 / 2)) ∧
  n = 2 * a := by
  sorry

end milk_processing_time_l3416_341670


namespace parabola_constant_term_l3416_341649

theorem parabola_constant_term (b c : ℝ) : 
  (2 = 2*(1^2) + b*1 + c) ∧ (2 = 2*(3^2) + b*3 + c) → c = 8 := by
sorry

end parabola_constant_term_l3416_341649


namespace arithmetic_progression_sum_15_l3416_341687

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression (α : Type*) [Add α] [SMul ℕ α] where
  a₁ : α
  d : α

variable {α : Type*} [LinearOrderedField α]

def term (ap : ArithmeticProgression α) (n : ℕ) : α :=
  ap.a₁ + (n - 1) • ap.d

def sum_n_terms (ap : ArithmeticProgression α) (n : ℕ) : α :=
  (n : α) * (ap.a₁ + term ap n) / 2

theorem arithmetic_progression_sum_15
  (ap : ArithmeticProgression α)
  (h_sum : term ap 3 + term ap 9 = 6)
  (h_prod : term ap 3 * term ap 9 = 135 / 16) :
  sum_n_terms ap 15 = 37.5 ∨ sum_n_terms ap 15 = 52.5 := by
  sorry

end arithmetic_progression_sum_15_l3416_341687


namespace sunday_reading_time_is_46_l3416_341675

def book_a_assignment : ℕ := 60
def book_b_assignment : ℕ := 45
def friday_book_a : ℕ := 16
def saturday_book_a : ℕ := 28
def saturday_book_b : ℕ := 15

def sunday_reading_time : ℕ := 
  (book_a_assignment - (friday_book_a + saturday_book_a)) + 
  (book_b_assignment - saturday_book_b)

theorem sunday_reading_time_is_46 : sunday_reading_time = 46 := by
  sorry

end sunday_reading_time_is_46_l3416_341675


namespace parkway_soccer_players_l3416_341693

theorem parkway_soccer_players (total_students : ℕ) (boys : ℕ) (girls_not_playing : ℕ) :
  total_students = 420 →
  boys = 312 →
  girls_not_playing = 53 →
  ∃ (soccer_players : ℕ),
    soccer_players = 250 ∧
    (soccer_players : ℚ) * (78 / 100) = boys - (total_students - boys - girls_not_playing) :=
by sorry

end parkway_soccer_players_l3416_341693


namespace solution_set_of_inequality_l3416_341661

theorem solution_set_of_inequality (x : ℝ) :
  (((1 : ℝ) / Real.pi) ^ (-x + 1) > ((1 : ℝ) / Real.pi) ^ (x^2 - x)) ↔ (x > 1 ∨ x < -1) :=
by sorry

end solution_set_of_inequality_l3416_341661


namespace acute_angle_range_l3416_341699

theorem acute_angle_range (α : Real) (h_acute : 0 < α ∧ α < Real.pi / 2) :
  (∃ x : Real, 3 * x^2 * Real.sin α - 4 * x * Real.cos α + 2 = 0) →
  0 < α ∧ α ≤ Real.pi / 6 := by
sorry

end acute_angle_range_l3416_341699


namespace rational_function_value_l3416_341650

-- Define the property for the rational function f
def satisfies_equation (f : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 3 * f (1 / x) + 2 * f x / x = x^2

-- State the theorem
theorem rational_function_value :
  ∀ f : ℚ → ℚ, satisfies_equation f → f (-2) = 67 / 20 :=
by
  sorry

end rational_function_value_l3416_341650


namespace translation_theorem_l3416_341677

/-- Represents a point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in the 2D Cartesian coordinate system -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def applyTranslation (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

/-- Theorem: Given points A, B, and C, where AB is translated to CD,
    prove that D has the correct coordinates -/
theorem translation_theorem (A B C : Point)
    (h1 : A = { x := -1, y := 0 })
    (h2 : B = { x := 1, y := 2 })
    (h3 : C = { x := 1, y := -2 }) :
  let t : Translation := { dx := C.x - A.x, dy := C.y - A.y }
  let D : Point := applyTranslation B t
  D = { x := 3, y := 0 } := by
  sorry


end translation_theorem_l3416_341677


namespace area_enclosed_by_function_and_line_l3416_341604

theorem area_enclosed_by_function_and_line (c : ℝ) : 
  30 = (1/2) * (c + 2) * (c - 2) → c = 8 := by
sorry

end area_enclosed_by_function_and_line_l3416_341604


namespace max_factors_power_function_l3416_341615

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- b^n where b and n are positive integers less than or equal to 10 -/
def power_function (b n : ℕ) : ℕ := 
  if b ≤ 10 ∧ n ≤ 10 ∧ b > 0 ∧ n > 0 then b^n else 0

theorem max_factors_power_function :
  ∃ b n : ℕ, b ≤ 10 ∧ n ≤ 10 ∧ b > 0 ∧ n > 0 ∧
    num_factors (power_function b n) = 31 ∧
    ∀ b' n' : ℕ, b' ≤ 10 → n' ≤ 10 → b' > 0 → n' > 0 →
      num_factors (power_function b' n') ≤ 31 :=
sorry

end max_factors_power_function_l3416_341615


namespace monomial_sum_implies_mn_twelve_l3416_341654

/-- If the sum of 2x³yⁿ and -½xᵐy⁴ is a monomial, then mn = 12 -/
theorem monomial_sum_implies_mn_twelve (x y : ℝ) (m n : ℕ) :
  (∃ (c : ℝ), ∀ x y, 2 * x^3 * y^n - 1/2 * x^m * y^4 = c * x^3 * y^4) →
  m * n = 12 := by
sorry

end monomial_sum_implies_mn_twelve_l3416_341654


namespace fraction_comparison_l3416_341643

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a / b < c / d) 
  (h2 : b > d) 
  (h3 : d > 0) : 
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) := by
  sorry

end fraction_comparison_l3416_341643


namespace product_sum_inequality_l3416_341664

theorem product_sum_inequality (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : a₁ < a₂) (h2 : b₁ < b₂) : 
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end product_sum_inequality_l3416_341664


namespace water_temperature_difference_l3416_341614

theorem water_temperature_difference (n : ℕ) : 
  let T_h := (T_c : ℝ) + 64/3
  let T_n := T_h - (1/4)^n * (T_h - T_c)
  (T_h - T_n ≠ 1/2) ∧ (T_h - T_n ≠ 3) :=
by sorry

end water_temperature_difference_l3416_341614


namespace emily_small_gardens_l3416_341627

/-- Calculates the number of small gardens Emily has based on her seed distribution --/
def number_of_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_type : ℕ) (vegetable_types : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / (seeds_per_type * vegetable_types)

/-- Theorem stating that Emily has 4 small gardens --/
theorem emily_small_gardens :
  number_of_small_gardens 125 45 4 5 = 4 := by
  sorry

end emily_small_gardens_l3416_341627


namespace rectangle_area_l3416_341639

/-- Given a square, a circle, and two rectangles in a plane with the following properties:
    - The length of rectangle1 is two-fifths of the circle's radius
    - The circle's radius equals the square's side
    - The square's area is 900 sq. units
    - The width of rectangle1 is 10 units
    - The width of the square is thrice the width of rectangle2
    - The length of rectangle2 when tripled and added to the length of rectangle1 equals the length of rectangle1
    - The area of rectangle2 is half of the square's area
    Prove that the area of rectangle1 is 120 sq. units -/
theorem rectangle_area (square : Real) (circle : Real) (rectangle1 : Real × Real) (rectangle2 : Real × Real)
  (h1 : rectangle1.1 = (2/5) * circle)
  (h2 : circle = square)
  (h3 : square ^ 2 = 900)
  (h4 : rectangle1.2 = 10)
  (h5 : square = 3 * rectangle2.2)
  (h6 : 3 * rectangle2.1 + rectangle1.1 = rectangle1.1)
  (h7 : rectangle2.1 * rectangle2.2 = (1/2) * square ^ 2) :
  rectangle1.1 * rectangle1.2 = 120 := by sorry

end rectangle_area_l3416_341639


namespace common_chord_length_l3416_341647

theorem common_chord_length (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, x^2 + y^2 = 4 ∧ x^2 + y^2 + 2*a*y - 6 = 0) ∧
  (∀ x y : ℝ, x^2 + y^2 = 4 ∧ x^2 + y^2 + 2*a*y - 6 = 0 → y = 1/a) →
  a = 1 :=
sorry


end common_chord_length_l3416_341647


namespace solve_system_l3416_341688

theorem solve_system (x y : ℚ) (eq1 : 3 * x - 2 * y = 8) (eq2 : 2 * x + 3 * y = 11) :
  x = 46 / 13 := by
  sorry

end solve_system_l3416_341688


namespace modular_equation_solution_l3416_341618

theorem modular_equation_solution : ∃ (n : ℤ), 0 ≤ n ∧ n < 144 ∧ (143 * n) % 144 = 105 % 144 ∧ n = 39 := by
  sorry

end modular_equation_solution_l3416_341618


namespace dice_divisible_by_seven_l3416_341676

/-- A die is represented as a function from face index to digit -/
def Die := Fin 6 → Fin 6

/-- Property that opposite faces of a die sum to 7 -/
def OppositeFacesSum7 (d : Die) : Prop :=
  ∀ i : Fin 3, d i + d (i + 3) = 7

/-- A set of six dice -/
def DiceSet := Fin 6 → Die

/-- Property that all dice in a set have opposite faces summing to 7 -/
def AllDiceOppositeFacesSum7 (ds : DiceSet) : Prop :=
  ∀ i : Fin 6, OppositeFacesSum7 (ds i)

/-- A configuration of dice is a function from die position to face index -/
def DiceConfiguration := Fin 6 → Fin 6

/-- The number formed by a dice configuration -/
def NumberFormed (ds : DiceSet) (dc : DiceConfiguration) : ℕ :=
  (ds 0 (dc 0)) * 100000 + (ds 1 (dc 1)) * 10000 + (ds 2 (dc 2)) * 1000 +
  (ds 3 (dc 3)) * 100 + (ds 4 (dc 4)) * 10 + (ds 5 (dc 5))

theorem dice_divisible_by_seven (ds : DiceSet) (h : AllDiceOppositeFacesSum7 ds) :
  ∃ dc : DiceConfiguration, NumberFormed ds dc % 7 = 0 := by
  sorry

end dice_divisible_by_seven_l3416_341676


namespace cube_volume_scaling_l3416_341694

theorem cube_volume_scaling (V : ℝ) (V_pos : V > 0) :
  let original_side := V ^ (1/3)
  let new_side := 2 * original_side
  let new_volume := new_side ^ 3
  new_volume = 8 * V := by sorry

end cube_volume_scaling_l3416_341694


namespace andrew_sandwiches_l3416_341690

/-- The number of friends coming over to Andrew's game night. -/
def num_friends : ℕ := 4

/-- The number of sandwiches Andrew made for each friend. -/
def sandwiches_per_friend : ℕ := 3

/-- The total number of sandwiches Andrew made. -/
def total_sandwiches : ℕ := num_friends * sandwiches_per_friend

/-- Theorem stating that the total number of sandwiches Andrew made is 12. -/
theorem andrew_sandwiches : total_sandwiches = 12 := by
  sorry

end andrew_sandwiches_l3416_341690


namespace product_sum_equality_l3416_341659

theorem product_sum_equality : ∃ (p q r s : ℝ),
  (∀ x, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = p * x^3 + q * x^2 + r * x + s) →
  8 * p + 4 * q + 2 * r + s = 18 := by
  sorry

end product_sum_equality_l3416_341659


namespace opposite_number_l3416_341637

theorem opposite_number (a : ℝ) : 
  -(3 * a - 2) = -3 * a + 2 := by sorry

end opposite_number_l3416_341637


namespace unit_vector_magnitude_is_one_l3416_341686

variable {V : Type*} [NormedAddCommGroup V]

/-- The magnitude of a unit vector is equal to 1. -/
theorem unit_vector_magnitude_is_one (v : V) (h : ‖v‖ = 1) : ‖v‖ = 1 := by
  sorry

end unit_vector_magnitude_is_one_l3416_341686


namespace obtuse_triangle_side_ratio_l3416_341672

/-- An obtuse triangle with sides a, b, and c, where a is the longest side -/
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  a_longest : a ≥ b ∧ a ≥ c
  obtuse : a^2 > b^2 + c^2

/-- The ratio of the sum of squares of two shorter sides to the square of the longest side
    in an obtuse triangle is always greater than or equal to 1/2 -/
theorem obtuse_triangle_side_ratio (t : ObtuseTriangle) :
  (t.b^2 + t.c^2) / t.a^2 ≥ (1/2 : ℝ) := by
  sorry

end obtuse_triangle_side_ratio_l3416_341672


namespace john_payment_is_8000_l3416_341620

/-- Calculates John's payment for lawyer fees --/
def johnPayment (upfrontPayment : ℕ) (hourlyRate : ℕ) (courtTime : ℕ) : ℕ :=
  let totalTime := courtTime + 2 * courtTime
  let totalFee := upfrontPayment + hourlyRate * totalTime
  totalFee / 2

/-- Theorem: John's payment for lawyer fees is $8,000 --/
theorem john_payment_is_8000 :
  johnPayment 1000 100 50 = 8000 := by
  sorry

end john_payment_is_8000_l3416_341620


namespace x_1997_value_l3416_341669

def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => x n + (x n / (n + 1)) + 2

theorem x_1997_value : x 1996 = 23913 := by
  sorry

end x_1997_value_l3416_341669


namespace trigonometric_simplification_l3416_341666

theorem trigonometric_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (10 * π / 180) = 2 * (3 * Real.sqrt 3 + 4) / 9 := by
  sorry

end trigonometric_simplification_l3416_341666


namespace symmetric_points_sum_l3416_341619

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite. -/
def symmetric_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- Given points P(x,-3) and Q(4,y) that are symmetric with respect to the x-axis,
    prove that x + y = 7 -/
theorem symmetric_points_sum (x y : ℝ) :
  symmetric_x_axis (x, -3) (4, y) → x + y = 7 := by
  sorry

end symmetric_points_sum_l3416_341619


namespace production_days_calculation_l3416_341656

theorem production_days_calculation (n : ℕ) : 
  (70 * n + 90 = 75 * (n + 1)) → n = 3 :=
by
  sorry

end production_days_calculation_l3416_341656


namespace number_of_children_l3416_341613

def total_cupcakes : ℕ := 96
def cupcakes_per_child : ℕ := 12

theorem number_of_children : 
  total_cupcakes / cupcakes_per_child = 8 := by sorry

end number_of_children_l3416_341613


namespace max_plus_shapes_in_square_l3416_341648

theorem max_plus_shapes_in_square (side_length : ℕ) (l_shape_area : ℕ) (plus_shape_area : ℕ) 
  (h_side : side_length = 7)
  (h_l : l_shape_area = 3)
  (h_plus : plus_shape_area = 5) :
  ∃ (num_l num_plus : ℕ),
    num_l * l_shape_area + num_plus * plus_shape_area = side_length ^ 2 ∧
    num_l ≥ 4 ∧
    ∀ (other_num_l other_num_plus : ℕ),
      other_num_l * l_shape_area + other_num_plus * plus_shape_area = side_length ^ 2 →
      other_num_l ≥ 4 →
      other_num_plus ≤ num_plus :=
by sorry

end max_plus_shapes_in_square_l3416_341648


namespace sams_trip_length_l3416_341678

theorem sams_trip_length (total : ℚ) 
  (h1 : total / 4 + 24 + total / 6 = total) : total = 288 / 7 := by
  sorry

end sams_trip_length_l3416_341678


namespace root_in_interval_l3416_341632

/-- The function f(x) = ln x + 3x - 7 has a root in the interval (2, 3) -/
theorem root_in_interval : ∃ x : ℝ, 2 < x ∧ x < 3 ∧ Real.log x + 3 * x - 7 = 0 := by
  sorry

end root_in_interval_l3416_341632


namespace perpendicular_projection_vector_l3416_341662

/-- Two-dimensional vector -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- Line represented by a point and a direction vector -/
structure Line where
  point : Vec2
  dir : Vec2

def l : Line :=
  { point := { x := 2, y := 5 }
    dir := { x := 3, y := 2 } }

def m : Line :=
  { point := { x := 1, y := 3 }
    dir := { x := 2, y := 2 } }

def v : Vec2 :=
  { x := 1, y := -1 }

theorem perpendicular_projection_vector :
  (v.x * m.dir.x + v.y * m.dir.y = 0) ∧
  (2 * v.x - v.y = 3) := by sorry

end perpendicular_projection_vector_l3416_341662


namespace ball_max_height_l3416_341685

/-- The height function of the ball's trajectory -/
def f (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the ball -/
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), f s ≤ f t ∧ f t = 161 := by
  sorry

end ball_max_height_l3416_341685


namespace expression_equality_l3416_341644

theorem expression_equality (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z - z/x ≠ 0) :
  (x^2 - 1/y^2) / (z - z/x) = x/z :=
by sorry

end expression_equality_l3416_341644


namespace parallel_line_slope_l3416_341663

/-- Given a line with equation 5x - 3y = 21, prove that the slope of any parallel line is 5/3 -/
theorem parallel_line_slope (x y : ℝ) :
  (5 * x - 3 * y = 21) → 
  (∃ (m : ℝ), m = 5 / 3 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ → 
    (5 * x₁ - 3 * y₁ = 21 ∧ 5 * x₂ - 3 * y₂ = 21) → 
    (y₂ - y₁) / (x₂ - x₁) = m) :=
by sorry

end parallel_line_slope_l3416_341663


namespace cos_2x_eq_cos_2y_l3416_341646

theorem cos_2x_eq_cos_2y (x y : ℝ) 
  (h1 : Real.sin x + Real.cos y = 1) 
  (h2 : Real.cos x + Real.sin y = -1) : 
  Real.cos (2 * x) = Real.cos (2 * y) := by
  sorry

end cos_2x_eq_cos_2y_l3416_341646


namespace inequality_problem_l3416_341652

theorem inequality_problem (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / (a - 1) ≥ 1 / b) ∧
  (1 / b < 1 / a) ∧
  (|a| > -b) ∧
  (Real.sqrt (-a) > Real.sqrt (-b)) := by
  sorry

end inequality_problem_l3416_341652


namespace quadratic_equation_solution_l3416_341603

open Real

theorem quadratic_equation_solution (A : ℝ) (h1 : 0 < A) (h2 : A < π) :
  (∃ x y : ℝ, x^2 * cos A - 2*x + cos A = 0 ∧
              y^2 * cos A - 2*y + cos A = 0 ∧
              x^2 - y^2 = 3/8) →
  sin A = (sqrt 265 - 16) / 3 :=
sorry

end quadratic_equation_solution_l3416_341603


namespace quadratic_curve_focal_distance_l3416_341626

theorem quadratic_curve_focal_distance (a : ℝ) (h1 : a ≠ 0) :
  (∃ (x y : ℝ), x^2 + a*y^2 + a^2 = 0) ∧
  (∃ (c : ℝ), c = 2 ∧ c^2 = a^2 + (-a)) →
  a = (1 - Real.sqrt 17) / 2 :=
by sorry

end quadratic_curve_focal_distance_l3416_341626


namespace sin_difference_bound_l3416_341665

theorem sin_difference_bound (N : ℕ) :
  ∃ (n k : ℕ), n ≠ k ∧ n ≤ N + 1 ∧ k ≤ N + 1 ∧ |Real.sin n - Real.sin k| < 2 / N :=
sorry

end sin_difference_bound_l3416_341665


namespace set_intersection_example_l3416_341680

theorem set_intersection_example :
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {2, 3, 4}
  M ∩ N = {2, 3} := by
sorry

end set_intersection_example_l3416_341680


namespace min_group_size_with_94_percent_boys_l3416_341696

theorem min_group_size_with_94_percent_boys (boys girls : ℕ) :
  boys > 0 →
  girls > 0 →
  (boys : ℚ) / (boys + girls : ℚ) > 94 / 100 →
  boys + girls ≥ 17 :=
sorry

end min_group_size_with_94_percent_boys_l3416_341696


namespace communication_arrangement_l3416_341607

def letter_arrangement (n : ℕ) (triple : ℕ) (double : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial 3 * (Nat.factorial 2)^double * Nat.factorial (n - triple - 2*double))

theorem communication_arrangement :
  letter_arrangement 14 1 2 = 908107825 := by
  sorry

end communication_arrangement_l3416_341607


namespace car_speed_second_hour_l3416_341660

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 10)
  (h2 : average_speed = 35) :
  let speed_second_hour := 2 * average_speed - speed_first_hour
  speed_second_hour = 60 := by
  sorry

#check car_speed_second_hour

end car_speed_second_hour_l3416_341660


namespace smallest_bench_configuration_l3416_341674

theorem smallest_bench_configuration (adults_per_bench children_per_bench : ℕ) 
  (adults_per_bench_pos : adults_per_bench > 0)
  (children_per_bench_pos : children_per_bench > 0)
  (adults_per_bench_def : adults_per_bench = 9)
  (children_per_bench_def : children_per_bench = 15) :
  ∃ (M : ℕ), M > 0 ∧ M * adults_per_bench = M * children_per_bench ∧
  ∀ (N : ℕ), N > 0 → N * adults_per_bench = N * children_per_bench → M ≤ N :=
by sorry

end smallest_bench_configuration_l3416_341674


namespace jaymee_is_22_l3416_341625

/-- The age of Shara -/
def shara_age : ℕ := 10

/-- The age of Jaymee -/
def jaymee_age : ℕ := 2 * shara_age + 2

/-- Theorem stating Jaymee's age is 22 -/
theorem jaymee_is_22 : jaymee_age = 22 := by
  sorry

end jaymee_is_22_l3416_341625


namespace sum_remainder_three_l3416_341698

theorem sum_remainder_three (n : ℤ) : (5 - n + (n + 4)) % 6 = 3 := by
  sorry

end sum_remainder_three_l3416_341698


namespace unique_students_in_musical_groups_l3416_341623

/-- The number of unique students in four musical groups -/
theorem unique_students_in_musical_groups 
  (orchestra : Nat) (band : Nat) (choir : Nat) (jazz : Nat)
  (orchestra_band : Nat) (orchestra_choir : Nat) (band_choir : Nat)
  (band_jazz : Nat) (orchestra_jazz : Nat) (choir_jazz : Nat)
  (orchestra_band_choir : Nat) (all_four : Nat)
  (h1 : orchestra = 25)
  (h2 : band = 40)
  (h3 : choir = 30)
  (h4 : jazz = 15)
  (h5 : orchestra_band = 5)
  (h6 : orchestra_choir = 6)
  (h7 : band_choir = 4)
  (h8 : band_jazz = 3)
  (h9 : orchestra_jazz = 2)
  (h10 : choir_jazz = 4)
  (h11 : orchestra_band_choir = 3)
  (h12 : all_four = 1) :
  orchestra + band + choir + jazz
  - orchestra_band - orchestra_choir - band_choir
  - band_jazz - orchestra_jazz - choir_jazz
  + orchestra_band_choir + all_four = 90 :=
by sorry

end unique_students_in_musical_groups_l3416_341623


namespace current_velocity_l3416_341657

theorem current_velocity (rowing_speed : ℝ) (distance : ℝ) (total_time : ℝ) :
  rowing_speed = 10 ∧ distance = 48 ∧ total_time = 10 →
  ∃ v : ℝ, v = 2 ∧ 
    distance / (rowing_speed - v) + distance / (rowing_speed + v) = total_time :=
by sorry

end current_velocity_l3416_341657


namespace equation_equivalence_l3416_341641

theorem equation_equivalence (x : ℝ) : x^2 - 4*x - 4 = 0 ↔ (x - 2)^2 = 8 := by sorry

end equation_equivalence_l3416_341641


namespace quadratic_equation_solution_l3416_341628

theorem quadratic_equation_solution : 
  ∃ (a b : ℝ), 
    (a^2 - 6*a + 9 = 15) ∧ 
    (b^2 - 6*b + 9 = 15) ∧ 
    (a ≥ b) ∧ 
    (3*a - b = 6 + 4*Real.sqrt 15) := by
  sorry

end quadratic_equation_solution_l3416_341628


namespace sunflower_count_l3416_341622

theorem sunflower_count (total_flowers : ℕ) (other_flowers : ℕ) 
  (h1 : total_flowers = 160) 
  (h2 : other_flowers = 40) : 
  total_flowers - other_flowers = 120 := by
  sorry

end sunflower_count_l3416_341622


namespace parabola_height_comparison_l3416_341640

theorem parabola_height_comparison (x₁ x₂ : ℝ) (h1 : -4 < x₁ ∧ x₁ < -2) (h2 : 0 < x₂ ∧ x₂ < 2) :
  (x₁ ^ 2 : ℝ) > x₂ ^ 2 := by sorry

end parabola_height_comparison_l3416_341640


namespace quadratic_equation_roots_l3416_341600

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - 19 * x + k = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 - 19 * y + k = 0 ∧ y = 16/3) :=
by
  sorry

end quadratic_equation_roots_l3416_341600


namespace perimeter_ratio_after_folding_and_cutting_l3416_341692

theorem perimeter_ratio_after_folding_and_cutting (s : ℝ) (h : s > 0) :
  let original_square_perimeter := 4 * s
  let small_rectangle_perimeter := 2 * (s / 2 + s / 4)
  small_rectangle_perimeter / original_square_perimeter = 3 / 8 := by
sorry

end perimeter_ratio_after_folding_and_cutting_l3416_341692


namespace apples_eaten_by_keith_l3416_341653

theorem apples_eaten_by_keith (mike_apples nancy_apples apples_left : ℝ) 
  (h1 : mike_apples = 7.0)
  (h2 : nancy_apples = 3.0)
  (h3 : apples_left = 4.0) :
  mike_apples + nancy_apples - apples_left = 6.0 := by
  sorry

end apples_eaten_by_keith_l3416_341653


namespace bus_speed_problem_l3416_341616

/-- Proves that given the conditions of the bus problem, the average speed for the 220 km distance is 40 kmph -/
theorem bus_speed_problem (total_distance : ℝ) (total_time : ℝ) (distance_at_x : ℝ) (speed_known : ℝ) :
  total_distance = 250 →
  total_time = 6 →
  distance_at_x = 220 →
  speed_known = 60 →
  ∃ x : ℝ,
    x > 0 ∧
    (distance_at_x / x) + ((total_distance - distance_at_x) / speed_known) = total_time ∧
    x = 40 :=
by sorry

end bus_speed_problem_l3416_341616


namespace sequence_inequality_l3416_341673

/-- A sequence of positive real numbers satisfying the given inequality -/
def PositiveSequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, i > 0 → a i > 0 ∧ i * (a i)^2 ≥ (i + 1) * (a (i - 1)) * (a (i + 1))

/-- Definition of the sequence b in terms of a -/
def b (a : ℕ → ℝ) (x y : ℝ) : ℕ → ℝ :=
  λ i => x * (a i) + y * (a (i - 1))

theorem sequence_inequality (a : ℕ → ℝ) (x y : ℝ) 
    (h_pos : PositiveSequence a) (h_x : x > 0) (h_y : y > 0) :
    ∀ i : ℕ, i ≥ 2 → i * (b a x y i)^2 > (i + 1) * (b a x y (i - 1)) * (b a x y (i + 1)) := by
  sorry

end sequence_inequality_l3416_341673


namespace power_of_two_consecutive_zeros_l3416_341691

/-- For any positive integer k, there exists a positive integer n such that
    the decimal representation of 2^n contains exactly k consecutive zeros. -/
theorem power_of_two_consecutive_zeros (k : ℕ) (hk : k ≥ 1) :
  ∃ n : ℕ, ∃ a b : ℕ, ∃ m : ℕ,
    (a ≠ 0) ∧ (b ≠ 0) ∧ (m > k) ∧
    (2^n : ℕ) = a * 10^m + b * 10^(m-k) :=
sorry

end power_of_two_consecutive_zeros_l3416_341691


namespace solution_not_zero_l3416_341610

theorem solution_not_zero (a : ℝ) : ∀ x : ℝ, x = a * x + 1 → x ≠ 0 := by
  sorry

end solution_not_zero_l3416_341610


namespace prop_1_prop_4_l3416_341683

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

variable (m n : Line)
variable (α β : Plane)

-- Proposition 1
theorem prop_1 (h1 : parallel_planes α β) (h2 : subset m α) :
  parallel_line_plane m β := by sorry

-- Proposition 4
theorem prop_4 (h1 : parallel_line_plane m β) (h2 : subset m α) (h3 : intersect α β n) :
  parallel_lines m n := by sorry

end prop_1_prop_4_l3416_341683


namespace binary_sum_equals_decimal_l3416_341681

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_sum_equals_decimal : 
  let binary1 := [true, false, true, false, true, false, true]  -- 1010101₂
  let binary2 := [false, false, false, true, true, true]        -- 111000₂
  binaryToDecimal binary1 + binaryToDecimal binary2 = 141 := by
sorry

end binary_sum_equals_decimal_l3416_341681


namespace overestimation_proof_l3416_341608

theorem overestimation_proof (p q k d : ℤ) 
  (p_round q_round k_round d_round : ℚ)
  (hp : p = 150) (hq : q = 50) (hk : k = 2) (hd : d = 3)
  (hp_round : p_round = 160) (hq_round : q_round = 45) 
  (hk_round : k_round = 1) (hd_round : d_round = 4) :
  (p_round / q_round - k_round + d_round) > (p / q - k + d) := by
  sorry

#check overestimation_proof

end overestimation_proof_l3416_341608


namespace arithmetic_sequence_common_difference_l3416_341630

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h1 : a 1 + a 5 = 10)
  (h2 : a 4 = 7)
  (h_arith : arithmetic_sequence a) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l3416_341630


namespace greatest_prime_factor_of_expression_l3416_341667

theorem greatest_prime_factor_of_expression : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (5^8 + 10^7) ∧ ∀ (q : ℕ), q.Prime → q ∣ (5^8 + 10^7) → q ≤ p :=
by
  -- Proof goes here
  sorry

end greatest_prime_factor_of_expression_l3416_341667


namespace village_population_l3416_341605

theorem village_population (initial_population : ℕ) : 
  (initial_population : ℝ) * (1 - 0.08) * (1 - 0.15) = 3553 → 
  initial_population = 4547 := by
  sorry

end village_population_l3416_341605


namespace min_value_of_expression_lower_bound_achievable_l3416_341651

theorem min_value_of_expression (x : ℝ) : (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 := by
  sorry

theorem lower_bound_achievable : ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = 2023 := by
  sorry

end min_value_of_expression_lower_bound_achievable_l3416_341651


namespace f_has_zero_in_interval_l3416_341682

def f (x : ℝ) := 3 * x - x^2

theorem f_has_zero_in_interval :
  ∃ x : ℝ, x ∈ Set.Icc (-1) 0 ∧ f x = 0 := by
  sorry

end f_has_zero_in_interval_l3416_341682


namespace average_height_problem_l3416_341624

theorem average_height_problem (parker daisy reese : ℕ) : 
  parker + 4 = daisy →
  daisy = reese + 8 →
  reese = 60 →
  (parker + daisy + reese) / 3 = 64 := by
sorry

end average_height_problem_l3416_341624


namespace loop_structure_requirement_l3416_341602

/-- Represents a computational task that may or may not require a loop structure. -/
inductive ComputationalTask
  | SolveLinearSystem
  | CalculatePiecewiseFunction
  | CalculateFixedSum
  | FindSmallestNaturalNumber

/-- Determines if a given computational task requires a loop structure. -/
def requiresLoopStructure (task : ComputationalTask) : Prop :=
  match task with
  | ComputationalTask.FindSmallestNaturalNumber => true
  | _ => false

/-- The sum of natural numbers from 1 to n. -/
def sumUpTo (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

/-- Theorem stating that finding the smallest natural number n such that 1+2+3+...+n > 100
    requires a loop structure, while other given tasks do not. -/
theorem loop_structure_requirement :
  (∀ n : ℕ, sumUpTo n ≤ 100 → sumUpTo (n + 1) > 100) →
  (requiresLoopStructure ComputationalTask.FindSmallestNaturalNumber ∧
   ¬requiresLoopStructure ComputationalTask.SolveLinearSystem ∧
   ¬requiresLoopStructure ComputationalTask.CalculatePiecewiseFunction ∧
   ¬requiresLoopStructure ComputationalTask.CalculateFixedSum) :=
by sorry


end loop_structure_requirement_l3416_341602


namespace max_additional_plates_l3416_341679

/-- Represents a set of letters for license plates --/
def LetterSet := List Char

/-- Calculate the number of license plates given three sets of letters --/
def calculatePlates (set1 set2 set3 : LetterSet) : Nat :=
  set1.length * set2.length * set3.length

/-- The initial sets of letters --/
def initialSet1 : LetterSet := ['C', 'H', 'L', 'P', 'R', 'S']
def initialSet2 : LetterSet := ['A', 'I', 'O', 'U']
def initialSet3 : LetterSet := ['D', 'M', 'N', 'T', 'V']

/-- The number of new letters to be added --/
def newLettersCount : Nat := 3

/-- Theorem: The maximum number of additional license plates is 96 --/
theorem max_additional_plates :
  (∀ newSet1 newSet2 newSet3 : LetterSet,
    newSet1.length + newSet2.length + newSet3.length = initialSet1.length + initialSet2.length + initialSet3.length + newLettersCount →
    calculatePlates newSet1 newSet2 newSet3 - calculatePlates initialSet1 initialSet2 initialSet3 ≤ 96) ∧
  (∃ newSet1 newSet2 newSet3 : LetterSet,
    newSet1.length + newSet2.length + newSet3.length = initialSet1.length + initialSet2.length + initialSet3.length + newLettersCount ∧
    calculatePlates newSet1 newSet2 newSet3 - calculatePlates initialSet1 initialSet2 initialSet3 = 96) := by
  sorry


end max_additional_plates_l3416_341679


namespace linear_equation_solution_l3416_341634

theorem linear_equation_solution (a : ℝ) : 
  (a * 1 + (-2) = 3) → a = 5 := by
  sorry

end linear_equation_solution_l3416_341634


namespace sum_difference_is_50_l3416_341636

def sam_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_20 (x : ℕ) : ℕ :=
  20 * ((x + 10) / 20)

def alex_sum (n : ℕ) : ℕ :=
  List.sum (List.map round_to_nearest_20 (List.range n))

theorem sum_difference_is_50 :
  sam_sum 100 - alex_sum 100 = 50 := by
  sorry

#eval sam_sum 100 - alex_sum 100

end sum_difference_is_50_l3416_341636


namespace correct_observation_value_l3416_341631

theorem correct_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (wrong_value : ℝ) 
  (corrected_mean : ℝ) 
  (h1 : n = 40) 
  (h2 : initial_mean = 100) 
  (h3 : wrong_value = 75) 
  (h4 : corrected_mean = 99.075) : 
  (n : ℝ) * corrected_mean - ((n : ℝ) * initial_mean - wrong_value) = 38 := by
  sorry

end correct_observation_value_l3416_341631


namespace trapezoid_with_perpendicular_bisector_quadrilateral_is_isosceles_l3416_341658

-- Define the trapezoid and quadrilateral
variable (A B C D K L M N : Point)

-- Define the trapezoid ABCD
def is_trapezoid (A B C D : Point) : Prop := sorry

-- Define the angle bisectors of the trapezoid
def angle_bisectors_intersect (A B C D K L M N : Point) : Prop := sorry

-- Define the quadrilateral KLMN formed by the intersection of angle bisectors
def quadrilateral_from_bisectors (A B C D K L M N : Point) : Prop := sorry

-- Define perpendicular diagonals of KLMN
def perpendicular_diagonals (K L M N : Point) : Prop := sorry

-- Define an isosceles trapezoid
def is_isosceles_trapezoid (A B C D : Point) : Prop := sorry

-- Theorem statement
theorem trapezoid_with_perpendicular_bisector_quadrilateral_is_isosceles 
  (h1 : is_trapezoid A B C D)
  (h2 : angle_bisectors_intersect A B C D K L M N)
  (h3 : quadrilateral_from_bisectors A B C D K L M N)
  (h4 : perpendicular_diagonals K L M N) :
  is_isosceles_trapezoid A B C D := by sorry

end trapezoid_with_perpendicular_bisector_quadrilateral_is_isosceles_l3416_341658


namespace system_solution_l3416_341642

theorem system_solution (a b c d e : ℝ) : 
  (3 * a = (b + c + d)^3 ∧
   3 * b = (c + d + e)^3 ∧
   3 * c = (d + e + a)^3 ∧
   3 * d = (e + a + b)^3 ∧
   3 * e = (a + b + c)^3) →
  ((a = 1/3 ∧ b = 1/3 ∧ c = 1/3 ∧ d = 1/3 ∧ e = 1/3) ∨
   (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 ∧ e = 0) ∨
   (a = -1/3 ∧ b = -1/3 ∧ c = -1/3 ∧ d = -1/3 ∧ e = -1/3)) :=
by sorry

end system_solution_l3416_341642


namespace jeff_cabinets_l3416_341633

/-- Calculates the total number of cabinets Jeff has after installations -/
def total_cabinets (initial : ℕ) (counters : ℕ) (additional : ℕ) : ℕ :=
  initial + counters * (2 * initial) + additional

/-- Proves that Jeff has 26 cabinets in total -/
theorem jeff_cabinets : total_cabinets 3 3 5 = 26 := by
  sorry

end jeff_cabinets_l3416_341633


namespace quadratic_one_solution_sum_l3416_341668

theorem quadratic_one_solution_sum (a : ℝ) : 
  let f := fun x : ℝ => 3 * x^2 + a * x + 6 * x + 7
  (∃! x, f x = 0) → 
  ∃ a₁ a₂ : ℝ, a = a₁ ∨ a = a₂ ∧ a₁ + a₂ = -12 :=
by sorry

end quadratic_one_solution_sum_l3416_341668


namespace greatest_x_quadratic_inequality_l3416_341621

theorem greatest_x_quadratic_inequality :
  ∀ x : ℝ, -x^2 + 11*x - 28 ≥ 0 → x ≤ 7 :=
by sorry

end greatest_x_quadratic_inequality_l3416_341621


namespace triangle_similarity_l3416_341645

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Triangle : Type :=
  (A B C : Point)

-- Define the properties
def isAcute (t : Triangle) : Prop := sorry

def incircleTouchPoints (t : Triangle) (D E F : Point) : Prop := sorry

def isCircumcenter (P : Point) (t : Triangle) : Prop := sorry

-- Main theorem
theorem triangle_similarity (A B C D E F P Q R : Point) :
  let ABC := Triangle.mk A B C
  let AEF := Triangle.mk A E F
  let BDF := Triangle.mk B D F
  let CDE := Triangle.mk C D E
  let PQR := Triangle.mk P Q R
  isAcute ABC →
  incircleTouchPoints ABC D E F →
  isCircumcenter P AEF →
  isCircumcenter Q BDF →
  isCircumcenter R CDE →
  -- Conclusion: ABC and PQR are similar
  ∃ (k : ℝ), k > 0 ∧
    (P.x - Q.x)^2 + (P.y - Q.y)^2 = k * ((A.x - B.x)^2 + (A.y - B.y)^2) ∧
    (Q.x - R.x)^2 + (Q.y - R.y)^2 = k * ((B.x - C.x)^2 + (B.y - C.y)^2) ∧
    (R.x - P.x)^2 + (R.y - P.y)^2 = k * ((C.x - A.x)^2 + (C.y - A.y)^2) :=
by
  sorry


end triangle_similarity_l3416_341645


namespace circle_line_intersection_l3416_341684

/-- Given a circle and a line with a specific chord length, prove the possible values of 'a' -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + (y + a)^2 = 4 ∧ x - y - 2 = 0) →  -- Circle and line equations
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + (y₁ + a)^2 = 4 ∧ 
    x₁ - y₁ - 2 = 0 ∧
    x₂^2 + (y₂ + a)^2 = 4 ∧ 
    x₂ - y₂ - 2 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →  -- Chord length condition
  a = 0 ∨ a = 4 := by
sorry

end circle_line_intersection_l3416_341684
