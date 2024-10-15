import Mathlib

namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l2644_264464

theorem profit_percentage_previous_year
  (revenue_prev : ℝ)
  (profit_prev : ℝ)
  (revenue_decrease : ℝ)
  (profit_percentage_2009 : ℝ)
  (profit_increase : ℝ)
  (h1 : revenue_decrease = 0.2)
  (h2 : profit_percentage_2009 = 0.15)
  (h3 : profit_increase = 1.5)
  (h4 : profit_prev > 0)
  (h5 : revenue_prev > 0) :
  profit_prev / revenue_prev = 0.08 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l2644_264464


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2644_264417

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 4370000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { significand := 4.37
    exponent := 6
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.significand * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2644_264417


namespace NUMINAMATH_CALUDE_boxes_with_neither_l2644_264438

-- Define the set universe
def U : Set Nat := {n | n ≤ 15}

-- Define the set of boxes with crayons
def C : Set Nat := {n ∈ U | n ≤ 9}

-- Define the set of boxes with markers
def M : Set Nat := {n ∈ U | n ≤ 5}

-- Define the set of boxes with both crayons and markers
def B : Set Nat := {n ∈ U | n ≤ 4}

theorem boxes_with_neither (hU : Fintype U) (hC : Fintype C) (hM : Fintype M) (hB : Fintype B) :
  Fintype.card U - (Fintype.card C + Fintype.card M - Fintype.card B) = 5 := by
  sorry


end NUMINAMATH_CALUDE_boxes_with_neither_l2644_264438


namespace NUMINAMATH_CALUDE_truck_capacity_problem_l2644_264421

/-- The capacity of a large truck in tons -/
def large_truck_capacity : ℝ := sorry

/-- The capacity of a small truck in tons -/
def small_truck_capacity : ℝ := sorry

/-- The total capacity of a given number of large and small trucks -/
def total_capacity (large_trucks small_trucks : ℕ) : ℝ :=
  (large_trucks : ℝ) * large_truck_capacity + (small_trucks : ℝ) * small_truck_capacity

theorem truck_capacity_problem :
  total_capacity 3 4 = 22 ∧ total_capacity 5 2 = 25 →
  total_capacity 4 3 = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_truck_capacity_problem_l2644_264421


namespace NUMINAMATH_CALUDE_hyperbola_and_line_properties_l2644_264491

/-- Hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_focus : 2 = Real.sqrt (a^2 + b^2)
  h_eccentricity : 2 = Real.sqrt (a^2 + b^2) / a

/-- Line intersecting the hyperbola -/
structure IntersectingLine where
  k : ℝ
  m : ℝ
  h_slope : k = 1
  h_distinct : ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + m ∧ y₂ = k * x₂ + m ∧
    x₁^2 - y₁^2/3 = 1 ∧ x₂^2 - y₂^2/3 = 1
  h_area : ∃ (x₀ y₀ : ℝ), 
    x₀ = (k * m) / (3 - k^2) ∧
    y₀ = (3 * m) / (3 - k^2) ∧
    1/2 * |4 * k * m / (3 - k^2)| * |4 * m / (3 - k^2)| = 4

/-- Main theorem -/
theorem hyperbola_and_line_properties (C : Hyperbola) (l : IntersectingLine) :
  (C.a = 1 ∧ C.b = Real.sqrt 3) ∧ 
  (l.m = Real.sqrt 2 ∨ l.m = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_and_line_properties_l2644_264491


namespace NUMINAMATH_CALUDE_tg_ctg_roots_relation_l2644_264492

-- Define the tangent and cotangent functions
noncomputable def tg (α : Real) : Real := Real.tan α
noncomputable def ctg (α : Real) : Real := 1 / Real.tan α

-- State the theorem
theorem tg_ctg_roots_relation (p q r s α β : Real) :
  (tg α)^2 - p * (tg α) + q = 0 ∧
  (tg β)^2 - p * (tg β) + q = 0 ∧
  (ctg α)^2 - r * (ctg α) + s = 0 ∧
  (ctg β)^2 - r * (ctg β) + s = 0 →
  r * s = p / q^2 := by
sorry

end NUMINAMATH_CALUDE_tg_ctg_roots_relation_l2644_264492


namespace NUMINAMATH_CALUDE_new_students_l2644_264443

theorem new_students (initial : ℕ) (left : ℕ) (final : ℕ) : 
  initial = 33 → left = 18 → final = 29 → final - (initial - left) = 14 := by
  sorry

end NUMINAMATH_CALUDE_new_students_l2644_264443


namespace NUMINAMATH_CALUDE_square_greater_than_negative_double_l2644_264499

theorem square_greater_than_negative_double {a : ℝ} (h : a < -2) : a^2 > -2*a := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_negative_double_l2644_264499


namespace NUMINAMATH_CALUDE_garden_snake_length_l2644_264416

-- Define the lengths of the snakes
def boa_length : Float := 1.428571429
def garden_snake_ratio : Float := 7.0

-- Theorem statement
theorem garden_snake_length : 
  boa_length * garden_snake_ratio = 10.0 := by sorry

end NUMINAMATH_CALUDE_garden_snake_length_l2644_264416


namespace NUMINAMATH_CALUDE_polar_bear_fish_consumption_l2644_264482

/-- Calculates the total number of fish buckets required for three polar bears for a week -/
theorem polar_bear_fish_consumption 
  (bear1_trout bear1_salmon : ℝ)
  (bear2_trout bear2_salmon : ℝ)
  (bear3_trout bear3_salmon : ℝ)
  (h1 : bear1_trout = 0.2)
  (h2 : bear1_salmon = 0.4)
  (h3 : bear2_trout = 0.3)
  (h4 : bear2_salmon = 0.5)
  (h5 : bear3_trout = 0.25)
  (h6 : bear3_salmon = 0.45)
  : (bear1_trout + bear1_salmon + bear2_trout + bear2_salmon + bear3_trout + bear3_salmon) * 7 = 14.7 := by
  sorry

#check polar_bear_fish_consumption

end NUMINAMATH_CALUDE_polar_bear_fish_consumption_l2644_264482


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l2644_264429

theorem hexagon_angle_measure :
  ∀ (a b c d e f : ℝ),
    a = 135 ∧ b = 120 ∧ c = 105 ∧ d = 150 ∧ e = 110 →
    (a + b + c + d + e + f = 720) →
    f = 100 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l2644_264429


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2644_264475

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 1 + a 3 = 2 →
  a 3 + a 5 = 4 →
  a 7 + a 9 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2644_264475


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2644_264473

theorem complex_number_in_fourth_quadrant (z : ℂ) (h : (1 + Complex.I) * z = 2 - Complex.I) :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2644_264473


namespace NUMINAMATH_CALUDE_total_battery_after_exam_l2644_264489

def calculator_remaining_battery (full_capacity : ℝ) (used_proportion : ℝ) (exam_duration : ℝ) : ℝ :=
  full_capacity * (1 - used_proportion) - exam_duration

def total_remaining_battery (calc1_capacity : ℝ) (calc1_used : ℝ) 
                            (calc2_capacity : ℝ) (calc2_used : ℝ)
                            (calc3_capacity : ℝ) (calc3_used : ℝ)
                            (exam_duration : ℝ) : ℝ :=
  calculator_remaining_battery calc1_capacity calc1_used exam_duration +
  calculator_remaining_battery calc2_capacity calc2_used exam_duration +
  calculator_remaining_battery calc3_capacity calc3_used exam_duration

theorem total_battery_after_exam :
  total_remaining_battery 60 (3/4) 80 (1/2) 120 (2/3) 2 = 89 := by
  sorry

end NUMINAMATH_CALUDE_total_battery_after_exam_l2644_264489


namespace NUMINAMATH_CALUDE_license_plate_count_l2644_264420

/-- The number of consonants in the alphabet, including Y -/
def num_consonants : ℕ := 21

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_consonants^2 * num_vowels^2 * num_digits^2

theorem license_plate_count : total_license_plates = 1102500 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2644_264420


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2644_264408

/-- A line mx-y+2=0 is tangent to the circle x^2+y^2=1 if and only if m = ± √3 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∃ (x y : ℝ), m*x - y + 2 = 0 ∧ x^2 + y^2 = 1 ∧ 
   ∀ (x' y' : ℝ), m*x' - y' + 2 = 0 → x'^2 + y'^2 ≥ 1) ↔ 
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2644_264408


namespace NUMINAMATH_CALUDE_train_length_calculation_l2644_264427

/-- Calculates the length of a train given its speed and time to cross a pole. -/
theorem train_length_calculation (speed_km_hr : ℝ) (time_seconds : ℝ) : 
  speed_km_hr = 30 → time_seconds = 12 → 
  ∃ (length_meters : ℝ), 
    (abs (length_meters - 100) < 1) ∧ 
    (length_meters = speed_km_hr * (1000 / 3600) * time_seconds) := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2644_264427


namespace NUMINAMATH_CALUDE_loan_duration_is_seven_years_l2644_264497

/-- Calculates the duration of a loan given the principal, interest rate, and interest paid. -/
def loanDuration (principal interestPaid interestRate : ℚ) : ℚ :=
  (interestPaid * 100) / (principal * interestRate)

/-- Theorem stating that for the given loan conditions, the duration is 7 years. -/
theorem loan_duration_is_seven_years 
  (principal : ℚ) 
  (interestPaid : ℚ) 
  (interestRate : ℚ) 
  (h1 : principal = 1500)
  (h2 : interestPaid = 735)
  (h3 : interestRate = 7) :
  loanDuration principal interestPaid interestRate = 7 := by
  sorry

#eval loanDuration 1500 735 7

end NUMINAMATH_CALUDE_loan_duration_is_seven_years_l2644_264497


namespace NUMINAMATH_CALUDE_expand_expression_l2644_264495

theorem expand_expression (x : ℝ) : 24 * (3 * x + 4 - 2) = 72 * x + 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2644_264495


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l2644_264456

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  (x - 1) / (x + 1) = 0 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l2644_264456


namespace NUMINAMATH_CALUDE_pavan_travel_distance_l2644_264463

theorem pavan_travel_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h_total_time : total_time = 11)
  (h_speed1 : speed1 = 30)
  (h_speed2 : speed2 = 25)
  (h_half_distance : ∀ d : ℝ, d / 2 / speed1 + d / 2 / speed2 = total_time) :
  ∃ d : ℝ, d = 150 ∧ d / 2 / speed1 + d / 2 / speed2 = total_time :=
by sorry

end NUMINAMATH_CALUDE_pavan_travel_distance_l2644_264463


namespace NUMINAMATH_CALUDE_inequality1_sufficient_not_necessary_l2644_264401

-- Define the two inequalities
def inequality1 (x : ℝ) : Prop := 1 + 3 / (x - 1) ≥ 0
def inequality2 (x : ℝ) : Prop := (x + 2) * (x - 1) ≥ 0

-- Theorem statement
theorem inequality1_sufficient_not_necessary :
  (∀ x, inequality1 x → inequality2 x) ∧
  (∃ x, inequality2 x ∧ ¬inequality1 x) := by sorry

end NUMINAMATH_CALUDE_inequality1_sufficient_not_necessary_l2644_264401


namespace NUMINAMATH_CALUDE_skylar_donation_l2644_264469

/-- Calculates the total donation amount given starting age, current age, and annual donation. -/
def total_donation (start_age : ℕ) (current_age : ℕ) (annual_donation : ℕ) : ℕ :=
  (current_age - start_age) * annual_donation

/-- Proves that Skylar's total donation is $432,000 given the specified conditions. -/
theorem skylar_donation :
  let start_age : ℕ := 17
  let current_age : ℕ := 71
  let annual_donation : ℕ := 8000
  total_donation start_age current_age annual_donation = 432000 := by
  sorry

end NUMINAMATH_CALUDE_skylar_donation_l2644_264469


namespace NUMINAMATH_CALUDE_james_bills_denomination_l2644_264405

/-- Proves that the denomination of each bill James found is $20 -/
theorem james_bills_denomination (initial_amount : ℕ) (final_amount : ℕ) (num_bills : ℕ) :
  initial_amount = 75 →
  final_amount = 135 →
  num_bills = 3 →
  (final_amount - initial_amount) / num_bills = 20 :=
by sorry

end NUMINAMATH_CALUDE_james_bills_denomination_l2644_264405


namespace NUMINAMATH_CALUDE_translation_problem_l2644_264466

def translation (z w : ℂ) : ℂ := z + w

theorem translation_problem (t : ℂ → ℂ) :
  (∃ w : ℂ, ∀ z, t z = translation z w) →
  t (1 + 3*I) = 5 + 7*I →
  t (2 - 2*I) = 6 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_translation_problem_l2644_264466


namespace NUMINAMATH_CALUDE_data_center_connections_l2644_264498

theorem data_center_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_data_center_connections_l2644_264498


namespace NUMINAMATH_CALUDE_distance_between_points_l2644_264442

/-- The distance between two points given round trip time and speed -/
theorem distance_between_points (speed : ℝ) (time : ℝ) (h1 : speed > 0) (h2 : time > 0) :
  let total_distance := speed * time
  let distance_between := total_distance / 2
  distance_between = 120 :=
by
  sorry

#check distance_between_points 60 4

end NUMINAMATH_CALUDE_distance_between_points_l2644_264442


namespace NUMINAMATH_CALUDE_candy_bar_division_l2644_264467

theorem candy_bar_division (total_candy : ℝ) (num_bags : ℕ) 
  (h1 : total_candy = 15.5) 
  (h2 : num_bags = 5) : 
  total_candy / (num_bags : ℝ) = 3.1 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_division_l2644_264467


namespace NUMINAMATH_CALUDE_quadrilateral_division_theorem_l2644_264446

/-- Represents a convex quadrilateral with areas of its four parts --/
structure ConvexQuadrilateral :=
  (area1 : ℝ)
  (area2 : ℝ)
  (area3 : ℝ)
  (area4 : ℝ)

/-- The theorem stating the relationship between the areas of the four parts --/
theorem quadrilateral_division_theorem (q : ConvexQuadrilateral) 
  (h1 : q.area1 = 360)
  (h2 : q.area2 = 720)
  (h3 : q.area3 = 900) :
  q.area4 = 540 := by
  sorry

#check quadrilateral_division_theorem

end NUMINAMATH_CALUDE_quadrilateral_division_theorem_l2644_264446


namespace NUMINAMATH_CALUDE_division_equality_not_always_true_l2644_264425

theorem division_equality_not_always_true (x y m : ℝ) :
  ¬(∀ x y m : ℝ, x = y → x / m = y / m) :=
sorry

end NUMINAMATH_CALUDE_division_equality_not_always_true_l2644_264425


namespace NUMINAMATH_CALUDE_equation_solution_l2644_264474

theorem equation_solution : ∃ x : ℝ, (Real.sqrt (x + 42) + Real.sqrt (x + 10) = 16) ∧ (x = 39) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2644_264474


namespace NUMINAMATH_CALUDE_olive_oil_price_increase_l2644_264412

def highest_price : ℝ := 24
def lowest_price : ℝ := 16

theorem olive_oil_price_increase :
  (highest_price - lowest_price) / lowest_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_olive_oil_price_increase_l2644_264412


namespace NUMINAMATH_CALUDE_point_c_coordinates_l2644_264435

/-- Given two points A and B in ℝ², if vector BC is half of vector BA, 
    then the coordinates of point C are (0, 3/2) -/
theorem point_c_coordinates 
  (A B : ℝ × ℝ)
  (h_A : A = (1, 1))
  (h_B : B = (-1, 2))
  (h_BC : ∃ (C : ℝ × ℝ), C - B = (1/2) • (A - B)) :
  ∃ (C : ℝ × ℝ), C = (0, 3/2) := by
sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l2644_264435


namespace NUMINAMATH_CALUDE_positive_difference_is_zero_l2644_264484

/-- The quadratic equation from the problem -/
def quadratic_equation (x : ℂ) : Prop :=
  x^2 + 5*x + 20 = 2*x + 16

/-- The solutions of the quadratic equation -/
def solutions : Set ℂ :=
  {x : ℂ | quadratic_equation x}

/-- The positive difference between the solutions is 0 -/
theorem positive_difference_is_zero :
  ∃ (x y : ℂ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x.re - y.re| = 0 :=
sorry

end NUMINAMATH_CALUDE_positive_difference_is_zero_l2644_264484


namespace NUMINAMATH_CALUDE_product_eleven_cubed_sum_l2644_264461

theorem product_eleven_cubed_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 11^3 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 133 := by
sorry

end NUMINAMATH_CALUDE_product_eleven_cubed_sum_l2644_264461


namespace NUMINAMATH_CALUDE_miranda_stuffs_six_pillows_l2644_264458

/-- The number of pillows Miranda can stuff given the conditions -/
def miranda_pillows : ℕ :=
  let feathers_per_pound : ℕ := 300
  let goose_feathers : ℕ := 3600
  let pounds_per_pillow : ℕ := 2
  let total_pounds : ℕ := goose_feathers / feathers_per_pound
  total_pounds / pounds_per_pillow

/-- Proof that Miranda can stuff 6 pillows -/
theorem miranda_stuffs_six_pillows : miranda_pillows = 6 := by
  sorry

end NUMINAMATH_CALUDE_miranda_stuffs_six_pillows_l2644_264458


namespace NUMINAMATH_CALUDE_first_player_wins_l2644_264437

/-- Represents a position on the rectangular table -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the game state -/
structure GameState :=
  (table : Set Position)
  (occupied : Set Position)
  (currentPlayer : Nat)

/-- Defines a valid move in the game -/
def validMove (state : GameState) (pos : Position) : Prop :=
  pos ∈ state.table ∧ pos ∉ state.occupied

/-- Defines the winning condition for a player -/
def winningStrategy (player : Nat) : Prop :=
  ∀ (state : GameState), 
    state.currentPlayer = player → 
    ∃ (move : Position), validMove state move

/-- The main theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Position), 
    winningStrategy 1 ∧ 
    (∀ (state : GameState), 
      state.currentPlayer = 1 → 
      validMove state (strategy state)) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l2644_264437


namespace NUMINAMATH_CALUDE_complex_number_problem_l2644_264433

theorem complex_number_problem (z : ℂ) 
  (h1 : ∃ (r : ℝ), z + 2*I = r)
  (h2 : ∃ (s : ℝ), z / (2 - I) = s) :
  z = 4 - 2*I ∧ 
  ∀ (a : ℝ), (z - a*I)^2 ∈ {w : ℂ | 0 < w.re ∧ 0 < w.im} ↔ -6 < a ∧ a < -2 :=
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2644_264433


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2644_264471

theorem sum_of_decimals : (5.47 + 4.96 : ℝ) = 10.43 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2644_264471


namespace NUMINAMATH_CALUDE_system_solution_expression_simplification_l2644_264413

-- Problem 1
theorem system_solution (s t : ℚ) : 
  2 * s + 3 * t = 2 ∧ 2 * s - 6 * t = -1 → s = 1/2 ∧ t = 1/3 := by sorry

-- Problem 2
theorem expression_simplification (x y : ℚ) (h : x ≠ 0) : 
  ((x - y)^2 + (x + y) * (x - y)) / (2 * x) = x - y := by sorry

end NUMINAMATH_CALUDE_system_solution_expression_simplification_l2644_264413


namespace NUMINAMATH_CALUDE_perpendicular_tangents_condition_l2644_264414

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + x^2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.sqrt 2 * a * Real.sin (x/2) * Real.cos (x/2) - x

theorem perpendicular_tangents_condition (a : ℝ) : 
  (∀ x₁ : ℝ, x₁ > -1 → ∃ x₂ : ℝ, 
    (1 / (x₁ + 1) + 2 * x₁) * (Real.sqrt 2 / 2 * Real.cos x₂ - 1) = -1) → 
  |a| ≥ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_condition_l2644_264414


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l2644_264457

theorem sum_of_solutions_squared_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁ + 6)^2 = 49 ∧ (x₂ + 6)^2 = 49 ∧ x₁ + x₂ = -12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l2644_264457


namespace NUMINAMATH_CALUDE_alternating_sum_of_coefficients_l2644_264462

theorem alternating_sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀*x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅) →
  |a₀| - |a₁| + |a₂| - |a₃| + |a₄| - |a₅| = 1 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sum_of_coefficients_l2644_264462


namespace NUMINAMATH_CALUDE_complex_cube_roots_sum_of_powers_l2644_264404

theorem complex_cube_roots_sum_of_powers (ω ω' : ℂ) :
  ω^3 = 1 → ω'^3 = 1 → ω = (-1 + Complex.I * Real.sqrt 3) / 2 → ω' = (-1 - Complex.I * Real.sqrt 3) / 2 →
  ω^12 + ω'^12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_roots_sum_of_powers_l2644_264404


namespace NUMINAMATH_CALUDE_line_through_midpoint_parallel_to_PR_l2644_264486

/-- Given points P, Q, R in a 2D plane, prove that if a line y = mx + b is parallel to PR
    and passes through the midpoint of QR, then b = -4. -/
theorem line_through_midpoint_parallel_to_PR (P Q R : ℝ × ℝ) (m b : ℝ) : 
  P = (0, 0) →
  Q = (4, 0) →
  R = (1, 2) →
  (∀ x y : ℝ, y = m * x + b ↔ (∃ t : ℝ, (x, y) = ((1 - t) * P.1 + t * R.1, (1 - t) * P.2 + t * R.2))) →
  (m * ((Q.1 + R.1) / 2) + b = (Q.2 + R.2) / 2) →
  b = -4 := by
  sorry


end NUMINAMATH_CALUDE_line_through_midpoint_parallel_to_PR_l2644_264486


namespace NUMINAMATH_CALUDE_average_geometric_sequence_l2644_264483

theorem average_geometric_sequence (z : ℝ) : 
  (z + 3*z + 9*z + 27*z + 81*z) / 5 = 24.2 * z := by
  sorry

end NUMINAMATH_CALUDE_average_geometric_sequence_l2644_264483


namespace NUMINAMATH_CALUDE_weighted_distances_sum_l2644_264439

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  V : ℝ  -- Volume
  S : Fin 4 → ℝ  -- Face areas
  d : Fin 4 → ℝ  -- Distances from a point to each face
  k : ℝ  -- Constant ratio
  h_positive : V > 0
  S_positive : ∀ i, S i > 0
  d_positive : ∀ i, d i > 0
  k_positive : k > 0
  h_ratio : ∀ i : Fin 4, S i / (i.val + 1 : ℝ) = k

/-- The sum of weighted distances equals three times the volume divided by k -/
theorem weighted_distances_sum (p : TriangularPyramid) :
  (p.d 0) + 2 * (p.d 1) + 3 * (p.d 2) + 4 * (p.d 3) = 3 * p.V / p.k := by
  sorry

end NUMINAMATH_CALUDE_weighted_distances_sum_l2644_264439


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_four_l2644_264444

theorem least_positive_integer_to_multiple_of_four (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬((530 + m) % 4 = 0)) ∧ ((530 + n) % 4 = 0) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_four_l2644_264444


namespace NUMINAMATH_CALUDE_prob_connected_formula_l2644_264406

/-- The number of vertices in the graph -/
def n : ℕ := 20

/-- The number of edges removed -/
def k : ℕ := 35

/-- The total number of edges in a complete graph with n vertices -/
def total_edges : ℕ := n * (n - 1) / 2

/-- The probability that the graph remains connected after removing k edges -/
def prob_connected : ℚ :=
  1 - (n : ℚ) * (Nat.choose (total_edges - n + 1) (k - n + 1) : ℚ) / (Nat.choose total_edges k : ℚ)

theorem prob_connected_formula :
  prob_connected = 1 - (20 : ℚ) * (Nat.choose 171 16 : ℚ) / (Nat.choose 190 35 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_prob_connected_formula_l2644_264406


namespace NUMINAMATH_CALUDE_slices_per_banana_l2644_264448

/-- Given information about yogurt preparation and banana usage, 
    calculate the number of slices per banana. -/
theorem slices_per_banana 
  (slices_per_yogurt : ℕ) 
  (yogurts_to_make : ℕ) 
  (bananas_needed : ℕ) 
  (h1 : slices_per_yogurt = 8) 
  (h2 : yogurts_to_make = 5) 
  (h3 : bananas_needed = 4) : 
  (slices_per_yogurt * yogurts_to_make) / bananas_needed = 10 := by
sorry

end NUMINAMATH_CALUDE_slices_per_banana_l2644_264448


namespace NUMINAMATH_CALUDE_p_plus_q_equals_twenty_l2644_264403

theorem p_plus_q_equals_twenty (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) →
  P + Q = 20 := by
sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_twenty_l2644_264403


namespace NUMINAMATH_CALUDE_water_cooler_problem_l2644_264430

/-- Represents the problem of calculating remaining water in coolers after filling cups for a meeting --/
theorem water_cooler_problem (gallons_per_ounce : ℚ) 
  (first_cooler_gallons second_cooler_gallons : ℚ)
  (small_cup_ounces large_cup_ounces : ℚ)
  (rows chairs_per_row : ℕ) :
  first_cooler_gallons = 4.5 →
  second_cooler_gallons = 3.25 →
  small_cup_ounces = 4 →
  large_cup_ounces = 8 →
  rows = 7 →
  chairs_per_row = 12 →
  gallons_per_ounce = 1 / 128 →
  (first_cooler_gallons / gallons_per_ounce) - 
    (↑(rows * chairs_per_row) * small_cup_ounces) = 240 :=
by sorry

end NUMINAMATH_CALUDE_water_cooler_problem_l2644_264430


namespace NUMINAMATH_CALUDE_fraction_problem_l2644_264424

theorem fraction_problem (x y : ℚ) : 
  y / (x - 1) = 1 / 3 → (y + 4) / x = 1 / 2 → y / x = 7 / 22 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2644_264424


namespace NUMINAMATH_CALUDE_shannon_bracelets_l2644_264428

/-- Given Shannon has 48 heart-shaped stones and each bracelet requires 8 stones,
    prove that she can make 6 bracelets. -/
theorem shannon_bracelets :
  let total_stones : ℕ := 48
  let stones_per_bracelet : ℕ := 8
  let num_bracelets : ℕ := total_stones / stones_per_bracelet
  num_bracelets = 6 := by
sorry

end NUMINAMATH_CALUDE_shannon_bracelets_l2644_264428


namespace NUMINAMATH_CALUDE_negative_two_times_inequality_l2644_264465

theorem negative_two_times_inequality {a b : ℝ} (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_two_times_inequality_l2644_264465


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l2644_264485

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 8 ways to distribute 7 indistinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l2644_264485


namespace NUMINAMATH_CALUDE_no_prime_sum_10003_l2644_264402

/-- A function that returns the number of ways to write n as the sum of two primes -/
def count_prime_sum_ways (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range n)).card

/-- Theorem stating that 10003 cannot be written as the sum of two primes -/
theorem no_prime_sum_10003 : count_prime_sum_ways 10003 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_10003_l2644_264402


namespace NUMINAMATH_CALUDE_population_change_l2644_264477

theorem population_change (P : ℝ) : 
  P > 0 →
  (P * 1.25 * 0.75 = 18750) →
  P = 20000 := by
sorry

end NUMINAMATH_CALUDE_population_change_l2644_264477


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l2644_264409

theorem polynomial_division_quotient : 
  ∀ (x : ℝ), (10 * x^3 + 20 * x^2 - 9 * x + 6) = (2 * x + 6) * (5 * x^2 - 5 * x + 3) + (-57) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l2644_264409


namespace NUMINAMATH_CALUDE_trivia_team_score_l2644_264447

/-- Represents a trivia team member's performance -/
structure MemberPerformance where
  two_point_questions : ℕ
  four_point_questions : ℕ
  six_point_questions : ℕ

/-- Calculates the total points for a member's performance -/
def calculate_member_points (performance : MemberPerformance) : ℕ :=
  2 * performance.two_point_questions +
  4 * performance.four_point_questions +
  6 * performance.six_point_questions

/-- The trivia team's performance -/
def team_performance : List MemberPerformance := [
  ⟨3, 0, 0⟩, -- Member A
  ⟨0, 5, 1⟩, -- Member B
  ⟨0, 0, 2⟩, -- Member C
  ⟨4, 2, 0⟩, -- Member D
  ⟨1, 3, 0⟩, -- Member E
  ⟨0, 0, 5⟩, -- Member F
  ⟨1, 2, 0⟩, -- Member G
  ⟨2, 0, 3⟩, -- Member H
  ⟨0, 1, 4⟩, -- Member I
  ⟨7, 1, 0⟩  -- Member J
]

theorem trivia_team_score :
  (team_performance.map calculate_member_points).sum = 182 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_score_l2644_264447


namespace NUMINAMATH_CALUDE_existence_of_equal_point_l2644_264472

theorem existence_of_equal_point
  (f g : ℝ → ℝ)
  (hf : Continuous f)
  (hg : Continuous g)
  (hg_diff : Differentiable ℝ g)
  (h_condition : (f 0 - deriv g 0) * (deriv g 1 - f 1) > 0) :
  ∃ c ∈ (Set.Ioo 0 1), f c = deriv g c :=
sorry

end NUMINAMATH_CALUDE_existence_of_equal_point_l2644_264472


namespace NUMINAMATH_CALUDE_sin_sum_zero_l2644_264453

theorem sin_sum_zero : 
  Real.sin (-1071 * π / 180) * Real.sin (99 * π / 180) + 
  Real.sin (-171 * π / 180) * Real.sin (-261 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_zero_l2644_264453


namespace NUMINAMATH_CALUDE_line_plane_relationships_l2644_264407

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the relationships between planes
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the relationship between lines
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)

-- Define the given conditions
variable (l m : Line) (α β : Plane)
variable (h1 : perpendicular l α)
variable (h2 : parallel m β)

-- State the theorem
theorem line_plane_relationships :
  (plane_parallel α β → line_perpendicular l m) ∧
  (line_parallel l m → plane_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationships_l2644_264407


namespace NUMINAMATH_CALUDE_fraction_simplification_l2644_264454

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2644_264454


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l2644_264436

theorem system_of_inequalities_solution (x : ℝ) :
  (2 * x + 1 > x ∧ x < -3 * x + 8) ↔ (-1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l2644_264436


namespace NUMINAMATH_CALUDE_paint_usage_fraction_l2644_264431

theorem paint_usage_fraction (initial_paint : ℚ) (first_week_fraction : ℚ) (total_used : ℚ) :
  initial_paint = 360 →
  first_week_fraction = 1/4 →
  total_used = 180 →
  let remaining_after_first_week := initial_paint - first_week_fraction * initial_paint
  let used_second_week := total_used - first_week_fraction * initial_paint
  used_second_week / remaining_after_first_week = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_fraction_l2644_264431


namespace NUMINAMATH_CALUDE_line_through_ellipse_midpoint_l2644_264476

/-- Given an ellipse and a line passing through its midpoint, prove the line's equation -/
theorem line_through_ellipse_midpoint (A B : ℝ × ℝ) :
  let M : ℝ × ℝ := (1, 1)
  let ellipse (p : ℝ × ℝ) := p.1^2 / 4 + p.2^2 / 3 = 1
  ellipse A ∧ ellipse B ∧  -- A and B are on the ellipse
  (∃ (k m : ℝ), ∀ (x y : ℝ), y = k * x + m ↔ ((x, y) = A ∨ (x, y) = B ∨ (x, y) = M)) ∧  -- A, B, and M are collinear
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  ∃ (k m : ℝ), k = 3 ∧ m = -7 ∧ ∀ (x y : ℝ), y = k * x + m ↔ ((x, y) = A ∨ (x, y) = B ∨ (x, y) = M) :=
by sorry


end NUMINAMATH_CALUDE_line_through_ellipse_midpoint_l2644_264476


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_find_circle_parameter_l2644_264419

-- Part 1
theorem circle_equation_from_diameter (P₁ P₂ : ℝ × ℝ) (h : P₁ = (4, 9) ∧ P₂ = (6, 3)) :
  ∃ C : ℝ × ℝ, ∃ r : ℝ, ∀ x y : ℝ,
    (x - C.1)^2 + (y - C.2)^2 = r^2 ↔ (x - 5)^2 + (y - 6)^2 = 10 :=
sorry

-- Part 2
theorem find_circle_parameter (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, x - y + 3 = 0 ∧ (x - a)^2 + (y - 2)^2 = 4) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ - y₁ + 3 = 0 ∧ x₂ - y₂ + 3 = 0 ∧
    (x₁ - a)^2 + (y₁ - 2)^2 = 4 ∧ (x₂ - a)^2 + (y₂ - 2)^2 = 4 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_find_circle_parameter_l2644_264419


namespace NUMINAMATH_CALUDE_min_value_implies_a_l2644_264470

/-- Given a function f(x) = 4x + a/x where x > 0 and a > 0, 
    if f attains its minimum value at x = 2, then a = 16 -/
theorem min_value_implies_a (a : ℝ) (h_a : a > 0) :
  (∀ x > 0, 4 * x + a / x ≥ 4 * 2 + a / 2) →
  (∃ x > 0, 4 * x + a / x = 4 * 2 + a / 2) →
  a = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l2644_264470


namespace NUMINAMATH_CALUDE_charlie_feathers_l2644_264480

theorem charlie_feathers (total_needed : ℕ) (still_needed : ℕ) 
  (h1 : total_needed = 900)
  (h2 : still_needed = 513) :
  total_needed - still_needed = 387 := by
  sorry

end NUMINAMATH_CALUDE_charlie_feathers_l2644_264480


namespace NUMINAMATH_CALUDE_certain_amount_proof_l2644_264490

theorem certain_amount_proof (A : ℝ) : 
  (0.20 * 1050 = 0.15 * 1500 - A) → A = 15 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l2644_264490


namespace NUMINAMATH_CALUDE_sin_inequality_solution_set_l2644_264400

theorem sin_inequality_solution_set (a : ℝ) (θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (h3 : θ = Real.arcsin a) :
  {x : ℝ | ∃ n : ℤ, (2*n - 1)*π - θ < x ∧ x < 2*n*π + θ} = {x : ℝ | Real.sin x < a} :=
by sorry

end NUMINAMATH_CALUDE_sin_inequality_solution_set_l2644_264400


namespace NUMINAMATH_CALUDE_common_tangents_count_l2644_264494

/-- Circle C₁ with equation x² + y² + 2x + 8y + 16 = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 8*p.2 + 16 = 0}

/-- Circle C₂ with equation x² + y² - 4x - 4y - 1 = 0 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 - 1 = 0}

/-- The number of common tangents to circles C₁ and C₂ -/
def numCommonTangents : ℕ := 4

/-- Theorem stating that the number of common tangents to C₁ and C₂ is 4 -/
theorem common_tangents_count :
  numCommonTangents = 4 :=
sorry

end NUMINAMATH_CALUDE_common_tangents_count_l2644_264494


namespace NUMINAMATH_CALUDE_union_covers_reals_implies_a_leq_neg_one_l2644_264440

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < 4}

-- State the theorem
theorem union_covers_reals_implies_a_leq_neg_one (a : ℝ) :
  A ∪ B a = Set.univ → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_implies_a_leq_neg_one_l2644_264440


namespace NUMINAMATH_CALUDE_pulley_system_force_l2644_264452

/-- The force required to move a load using a pulley system -/
def required_force (m : ℝ) (g : ℝ) : ℝ := 2 * m * g

/-- Theorem: The required force to move a 2 kg load with a pulley system is 20 N -/
theorem pulley_system_force :
  let m : ℝ := 2 -- mass of the load in kg
  let g : ℝ := 10 -- acceleration due to gravity in m/s²
  required_force m g = 20 := by
  sorry

#check pulley_system_force

end NUMINAMATH_CALUDE_pulley_system_force_l2644_264452


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2644_264450

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- The binary representation of 1101101₂ -/
def binary_num : List Bool := [true, true, false, true, true, false, true]

/-- The expected quaternary representation -/
def expected_quaternary : List Nat := [3, 1, 2, 1]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_num) = expected_quaternary :=
by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2644_264450


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l2644_264415

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n.val 2 = n.val * (n.val - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l2644_264415


namespace NUMINAMATH_CALUDE_solution_pairs_count_l2644_264481

theorem solution_pairs_count : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    4 * p.1 + 7 * p.2 = 548 ∧ p.1 > 0 ∧ p.2 > 0) 
    (Finset.product (Finset.range 548) (Finset.range 548))).card ∧ n = 19 := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_count_l2644_264481


namespace NUMINAMATH_CALUDE_trigonometric_calculation_and_algebraic_simplification_l2644_264423

theorem trigonometric_calculation_and_algebraic_simplification :
  (2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (30 * π / 180) + |(-1/2)| = 1) ∧
  (let a := 2 * Real.sin (60 * π / 180) - 3 * Real.tan (45 * π / 180)
   let b := 3
   1 - (a - b) / (a + 2*b) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) = -Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_calculation_and_algebraic_simplification_l2644_264423


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2644_264496

/-- Given a rhombus with area 80 and one diagonal of length 16, 
    prove that the other diagonal has length 10. -/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_area : area = 80) 
  (h_d1 : d1 = 16) 
  (h_rhombus : area = d1 * d2 / 2) : 
  d2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l2644_264496


namespace NUMINAMATH_CALUDE_greatest_common_divisor_under_60_l2644_264449

theorem greatest_common_divisor_under_60 : ∃ (d : ℕ), d = 36 ∧ 
  d ∣ 468 ∧ d ∣ 108 ∧ d < 60 ∧ 
  ∀ (x : ℕ), x ∣ 468 ∧ x ∣ 108 ∧ x < 60 → x ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_under_60_l2644_264449


namespace NUMINAMATH_CALUDE_ryan_tokens_l2644_264434

def arcade_tokens (initial_tokens : ℕ) : ℕ :=
  let pacman_tokens := (2 * initial_tokens) / 3
  let remaining_after_pacman := initial_tokens - pacman_tokens
  let candy_crush_tokens := remaining_after_pacman / 2
  let remaining_after_candy_crush := remaining_after_pacman - candy_crush_tokens
  let skeball_tokens := min remaining_after_candy_crush 7
  let parents_bought := 10 * skeball_tokens
  remaining_after_candy_crush - skeball_tokens + parents_bought

theorem ryan_tokens : arcade_tokens 36 = 66 := by
  sorry

end NUMINAMATH_CALUDE_ryan_tokens_l2644_264434


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2644_264478

/-- Given a square with diagonal length 40, prove its area is 800 -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 40) : d^2 / 2 = 800 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2644_264478


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l2644_264488

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (80 * π / 180)) / Real.cos (10 * π / 180) =
  2 / (Real.sqrt 3 * Real.sin (70 * π / 180) * Real.sin (10 * π / 180) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l2644_264488


namespace NUMINAMATH_CALUDE_derivative_h_at_one_l2644_264411

-- Define a function f
variable (f : ℝ → ℝ)

-- Define g(x) = f(x) - f(2x)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - f (2 * x)

-- Define h(x) = f(x) - f(4x)
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - f (4 * x)

-- State the theorem
theorem derivative_h_at_one (f : ℝ → ℝ) 
  (hg1 : deriv (g f) 1 = 5)
  (hg2 : deriv (g f) 2 = 7) :
  deriv (h f) 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_derivative_h_at_one_l2644_264411


namespace NUMINAMATH_CALUDE_apple_distribution_l2644_264468

theorem apple_distribution (total_apples : ℕ) (alice_min : ℕ) (becky_min : ℕ) (chris_min : ℕ)
  (h1 : total_apples = 30)
  (h2 : alice_min = 3)
  (h3 : becky_min = 2)
  (h4 : chris_min = 2) :
  (Nat.choose (total_apples - alice_min - becky_min - chris_min + 2) 2) = 300 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l2644_264468


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l2644_264451

theorem quadratic_factorization_sum (a b c : ℤ) :
  (∀ x, x^2 + 14*x + 45 = (x + a)*(x + b)) →
  (∀ x, x^2 - 19*x + 90 = (x - b)*(x - c)) →
  a + b + c = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l2644_264451


namespace NUMINAMATH_CALUDE_fraction_simplification_l2644_264422

theorem fraction_simplification (x : ℝ) : (3*x - 2)/4 + (5 - 2*x)/3 = (x + 14)/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2644_264422


namespace NUMINAMATH_CALUDE_point_four_units_from_one_l2644_264445

theorem point_four_units_from_one (x : ℝ) : 
  (x = 1 + 4 ∨ x = 1 - 4) ↔ (x = 5 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_point_four_units_from_one_l2644_264445


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2644_264460

theorem quadratic_roots_property (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ x₁ - x₂ = 5 ∧ x₁^3 - x₂^3 = 35) →
  ((p = 1 ∧ q = -6) ∨ (p = -1 ∧ q = -6)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2644_264460


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2644_264432

/-- The equation of a line perpendicular to another line and passing through a given point. -/
theorem perpendicular_line_equation (m : ℚ) (b : ℚ) (x₀ : ℚ) (y₀ : ℚ) :
  let l₁ : ℚ → ℚ := λ x => m * x + b
  let m₂ : ℚ := -1 / m
  let l₂ : ℚ → ℚ := λ x => m₂ * (x - x₀) + y₀
  (∀ x, x - 2 * l₁ x + 3 = 0) →
  (∀ x, 2 * x + l₂ x - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2644_264432


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l2644_264479

theorem probability_at_least_one_correct (total_questions : Nat) (options_per_question : Nat) (guessed_questions : Nat) : 
  total_questions = 30 → 
  options_per_question = 6 → 
  guessed_questions = 5 → 
  (1 - (options_per_question - 1 : ℚ) / options_per_question ^ guessed_questions) = 4651 / 7776 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l2644_264479


namespace NUMINAMATH_CALUDE_dessert_division_l2644_264487

/-- Represents the number of dessert items -/
structure DessertItems where
  cinnamon_swirls : ℕ
  brownie_bites : ℕ
  fruit_tartlets : ℕ

/-- Represents the number of people sharing the desserts -/
def num_people : ℕ := 8

/-- The actual dessert items from the problem -/
def desserts : DessertItems := {
  cinnamon_swirls := 15,
  brownie_bites := 24,
  fruit_tartlets := 18
}

/-- Theorem stating that brownie bites can be equally divided, while others cannot -/
theorem dessert_division (d : DessertItems) (p : ℕ) (h_p : p = num_people) :
  d.brownie_bites / p = 3 ∧
  ¬(∃ (n : ℕ), n * p = d.cinnamon_swirls) ∧
  ¬(∃ (m : ℕ), m * p = d.fruit_tartlets) :=
sorry

end NUMINAMATH_CALUDE_dessert_division_l2644_264487


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l2644_264418

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 7}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 6}

-- State the theorem
theorem set_operations_and_subset :
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x : ℝ | 2 < x ∧ x < 10}) ∧
  (∀ a : ℝ, A ⊆ C a ↔ 2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l2644_264418


namespace NUMINAMATH_CALUDE_counterexample_exists_l2644_264459

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a * b > 0 ∧ 1/a ≥ 1/b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2644_264459


namespace NUMINAMATH_CALUDE_vegetable_difference_is_30_l2644_264426

/-- Calculates the difference between initial and remaining vegetables after exchanges --/
def vegetable_difference (
  initial_tomatoes : ℕ)
  (initial_carrots : ℕ)
  (initial_cucumbers : ℕ)
  (initial_bell_peppers : ℕ)
  (picked_tomatoes : ℕ)
  (picked_carrots : ℕ)
  (picked_cucumbers : ℕ)
  (picked_bell_peppers : ℕ)
  (neighbor1_tomatoes : ℕ)
  (neighbor1_carrots : ℕ)
  (neighbor2_tomatoes : ℕ)
  (neighbor2_cucumbers : ℕ)
  (neighbor2_radishes : ℕ)
  (neighbor3_bell_peppers : ℕ) : ℕ :=
  let initial_total := initial_tomatoes + initial_carrots + initial_cucumbers + initial_bell_peppers
  let remaining_tomatoes := initial_tomatoes - picked_tomatoes - neighbor1_tomatoes - neighbor2_tomatoes
  let remaining_carrots := initial_carrots - picked_carrots - neighbor1_carrots
  let remaining_cucumbers := initial_cucumbers - picked_cucumbers - neighbor2_cucumbers
  let remaining_bell_peppers := initial_bell_peppers - picked_bell_peppers - neighbor3_bell_peppers
  let remaining_total := remaining_tomatoes + remaining_carrots + remaining_cucumbers + remaining_bell_peppers + neighbor2_radishes
  initial_total - remaining_total

/-- The difference between initial and remaining vegetables is 30 --/
theorem vegetable_difference_is_30 : 
  vegetable_difference 17 13 8 15 5 6 3 8 3 2 2 3 5 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_difference_is_30_l2644_264426


namespace NUMINAMATH_CALUDE_third_month_sale_l2644_264455

def average_sale : ℕ := 5500
def number_of_months : ℕ := 6
def sales : List ℕ := [5435, 5927, 6230, 5562, 3991]

theorem third_month_sale :
  (average_sale * number_of_months - sales.sum) = 5855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l2644_264455


namespace NUMINAMATH_CALUDE_average_payment_is_460_l2644_264441

/-- The total number of installments -/
def total_installments : ℕ := 52

/-- The number of initial payments -/
def initial_payments : ℕ := 12

/-- The amount of each initial payment -/
def initial_payment_amount : ℚ := 410

/-- The additional amount for each remaining payment -/
def additional_amount : ℚ := 65

/-- The amount of each remaining payment -/
def remaining_payment_amount : ℚ := initial_payment_amount + additional_amount

/-- The number of remaining payments -/
def remaining_payments : ℕ := total_installments - initial_payments

theorem average_payment_is_460 :
  (initial_payments * initial_payment_amount + remaining_payments * remaining_payment_amount) / total_installments = 460 := by
  sorry

end NUMINAMATH_CALUDE_average_payment_is_460_l2644_264441


namespace NUMINAMATH_CALUDE_sin_75_165_minus_sin_15_105_eq_zero_l2644_264410

theorem sin_75_165_minus_sin_15_105_eq_zero :
  Real.sin (75 * π / 180) * Real.sin (165 * π / 180) -
  Real.sin (15 * π / 180) * Real.sin (105 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_165_minus_sin_15_105_eq_zero_l2644_264410


namespace NUMINAMATH_CALUDE_jasmine_solution_problem_l2644_264493

theorem jasmine_solution_problem (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_jasmine : ℝ) (final_concentration : ℝ) (x : ℝ) : 
  initial_volume = 90 →
  initial_concentration = 0.05 →
  added_jasmine = 8 →
  final_concentration = 0.125 →
  initial_volume * initial_concentration + added_jasmine = 
    (initial_volume + added_jasmine + x) * final_concentration →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_jasmine_solution_problem_l2644_264493
