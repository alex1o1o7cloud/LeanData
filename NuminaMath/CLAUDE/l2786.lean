import Mathlib

namespace NUMINAMATH_CALUDE_triangle_trigonometric_identities_l2786_278676

theorem triangle_trigonometric_identities
  (α β γ : Real) (p r R : Real)
  (h_triangle : α + β + γ = Real.pi)
  (h_positive : 0 < p ∧ 0 < r ∧ 0 < R)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_inradius : r = area / p)
  (h_circumradius : R = (a * b * c) / (4 * area)) :
  (Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2 = (p^2 - r^2 - 4*r*R) / (2*R^2) ∧
  4 * R^2 * Real.cos α * Real.cos β * Real.cos γ = p^2 - (2*R + r)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identities_l2786_278676


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2786_278697

theorem fraction_to_decimal : (125 : ℚ) / 144 = 0.78125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2786_278697


namespace NUMINAMATH_CALUDE_game_result_l2786_278662

def score (n : ℕ) : ℕ :=
  if n % 3 = 0 then 5
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [3, 5, 6, 2, 4]
def betty_rolls : List ℕ := [3, 2, 1, 6, 4]

def total_score (rolls : List ℕ) : ℕ :=
  rolls.map score |>.sum

theorem game_result : total_score allie_rolls * total_score betty_rolls = 256 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l2786_278662


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_no_solution_product_equation_l2786_278660

/-- Given x and y are positive real numbers satisfying x^2 + y^2 = x + y -/
def satisfies_equation (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x^2 + y^2 = x + y

/-- The minimum value of 1/x + 1/y is 2 -/
theorem min_value_sum_reciprocals {x y : ℝ} (h : satisfies_equation x y) :
  1/x + 1/y ≥ 2 := by
  sorry

/-- There do not exist x and y satisfying (x+1)(y+1) = 5 -/
theorem no_solution_product_equation {x y : ℝ} (h : satisfies_equation x y) :
  (x + 1) * (y + 1) ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_no_solution_product_equation_l2786_278660


namespace NUMINAMATH_CALUDE_max_angle_APB_l2786_278683

/-- An ellipse with focus F and directrix l -/
structure Ellipse where
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- The focus of the ellipse -/
  F : ℝ × ℝ
  /-- The point where the directrix intersects the axis of symmetry -/
  P : ℝ × ℝ

/-- A chord of the ellipse passing through the focus -/
structure Chord (E : Ellipse) where
  /-- One endpoint of the chord -/
  A : ℝ × ℝ
  /-- The other endpoint of the chord -/
  B : ℝ × ℝ
  /-- The chord passes through the focus -/
  passes_through_focus : A.1 < E.F.1 ∧ E.F.1 < B.1

/-- The angle APB formed by a chord AB and the point P -/
def angle_APB (E : Ellipse) (C : Chord E) : ℝ :=
  sorry

/-- The theorem stating that the maximum value of angle APB is 2 arctan e -/
theorem max_angle_APB (E : Ellipse) :
  ∀ C : Chord E, angle_APB E C ≤ 2 * Real.arctan E.e ∧
  ∃ C : Chord E, angle_APB E C = 2 * Real.arctan E.e :=
sorry

end NUMINAMATH_CALUDE_max_angle_APB_l2786_278683


namespace NUMINAMATH_CALUDE_sheridan_fish_count_l2786_278652

/-- The number of fish Mrs. Sheridan initially had -/
def initial_fish : ℕ := 22

/-- The number of fish Mrs. Sheridan received from her sister -/
def received_fish : ℕ := 47

/-- The total number of fish Mrs. Sheridan has now -/
def total_fish : ℕ := initial_fish + received_fish

theorem sheridan_fish_count : total_fish = 69 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_fish_count_l2786_278652


namespace NUMINAMATH_CALUDE_impossible_sum_and_reciprocal_sum_l2786_278658

theorem impossible_sum_and_reciprocal_sum (x y z : ℝ) :
  x + y + z = 0 ∧ 1/x + 1/y + 1/z = 0 →
  x^1988 + y^1988 + z^1988 = 1/x^1988 + 1/y^1988 + 1/z^1988 :=
by sorry

end NUMINAMATH_CALUDE_impossible_sum_and_reciprocal_sum_l2786_278658


namespace NUMINAMATH_CALUDE_closest_estimate_l2786_278654

-- Define the constants from the problem
def cars_observed : ℕ := 8
def observation_time : ℕ := 20
def delay_time : ℕ := 15
def total_time : ℕ := 210  -- 3 minutes and 30 seconds in seconds

-- Define the function to calculate the estimated number of cars
def estimate_cars : ℚ :=
  let rate : ℚ := cars_observed / observation_time
  let missed_cars : ℚ := rate * delay_time
  let observed_cars : ℚ := rate * (total_time - delay_time)
  missed_cars + observed_cars

-- Define the given options
def options : List ℕ := [120, 150, 210, 240, 280]

-- Theorem statement
theorem closest_estimate :
  ∃ (n : ℕ), n ∈ options ∧ 
  ∀ (m : ℕ), m ∈ options → |n - estimate_cars| ≤ |m - estimate_cars| ∧
  n = 120 := by
  sorry

end NUMINAMATH_CALUDE_closest_estimate_l2786_278654


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l2786_278650

-- Define the number of red and black balls
def num_red_balls : ℕ := 7
def num_black_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_black_balls

-- Define the probability of drawing a red ball
def prob_red_ball : ℚ := num_red_balls / total_balls

-- Theorem statement
theorem probability_of_red_ball :
  prob_red_ball = 7 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l2786_278650


namespace NUMINAMATH_CALUDE_max_value_of_sum_l2786_278699

theorem max_value_of_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 5) :
  ∃ (M : ℝ), M = Real.sqrt 70 ∧ x + 2*y + 3*z ≤ M ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ + 2*y₀ + 3*z₀ = M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l2786_278699


namespace NUMINAMATH_CALUDE_red_blood_cell_diameter_scientific_notation_l2786_278679

/-- Expresses a given decimal number in scientific notation -/
def scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem red_blood_cell_diameter_scientific_notation :
  scientific_notation 0.00077 = (7.7, -4) :=
sorry

end NUMINAMATH_CALUDE_red_blood_cell_diameter_scientific_notation_l2786_278679


namespace NUMINAMATH_CALUDE_remaining_etching_price_l2786_278605

def total_etchings : ℕ := 16
def total_revenue : ℕ := 630
def etchings_at_35 : ℕ := 9
def price_at_35 : ℕ := 35

theorem remaining_etching_price :
  let remaining_etchings := total_etchings - etchings_at_35
  let revenue_from_35 := etchings_at_35 * price_at_35
  let remaining_revenue := total_revenue - revenue_from_35
  remaining_revenue / remaining_etchings = 45 := by
sorry

end NUMINAMATH_CALUDE_remaining_etching_price_l2786_278605


namespace NUMINAMATH_CALUDE_solution_value_l2786_278637

theorem solution_value (a b t : ℝ) : 
  a^2 + 4*b = t^2 →
  a^2 - b^2 = 4 →
  b > 0 →
  b = t - 2 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l2786_278637


namespace NUMINAMATH_CALUDE_train_length_l2786_278687

/-- Given a train that crosses a platform in 39 seconds and a signal pole in 18 seconds,
    where the platform is 350 meters long, prove that the length of the train is 300 meters. -/
theorem train_length (platform_crossing_time : ℕ) (pole_crossing_time : ℕ) (platform_length : ℕ)
    (h1 : platform_crossing_time = 39)
    (h2 : pole_crossing_time = 18)
    (h3 : platform_length = 350) :
    let train_length := (platform_crossing_time * platform_length) / (platform_crossing_time - pole_crossing_time)
    train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2786_278687


namespace NUMINAMATH_CALUDE_calculations_proof_l2786_278636

-- Define the calculations
def calc1 : ℝ := 70.8 - 1.25 - 1.75
def calc2 : ℝ := (8 + 0.8) * 1.25
def calc3 : ℝ := 125 * 0.48
def calc4 : ℝ := 6.7 * (9.3 * (6.2 + 1.7))

-- Theorem to prove the calculations
theorem calculations_proof :
  calc1 = 67.8 ∧
  calc2 = 11 ∧
  calc3 = 600 ∧
  calc4 = 554.559 := by
  sorry

#eval calc1
#eval calc2
#eval calc3
#eval calc4

end NUMINAMATH_CALUDE_calculations_proof_l2786_278636


namespace NUMINAMATH_CALUDE_inequality_solution_l2786_278617

theorem inequality_solution (x : ℝ) : 
  (x^2 + 2*x^3 - 3*x^4) / (2*x + 3*x^2 - 4*x^3) ≥ -1 ↔ 
  (x ∈ Set.Icc (-1) ((-3 - Real.sqrt 41) / -8) ∪ 
   Set.Ioo ((-3 - Real.sqrt 41) / -8) ((-3 + Real.sqrt 41) / -8) ∪
   Set.Ioo ((-3 + Real.sqrt 41) / -8) 0 ∪
   Set.Ioi 0) ∧
  (x ≠ 0) ∧ (x ≠ ((-3 - Real.sqrt 41) / -8)) ∧ (x ≠ ((-3 + Real.sqrt 41) / -8)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2786_278617


namespace NUMINAMATH_CALUDE_min_value_a_l2786_278690

theorem min_value_a (a : ℝ) : 
  (∀ x > a, 2 * x + 2 / (x - a) ≥ 7) ↔ a ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l2786_278690


namespace NUMINAMATH_CALUDE_paint_cost_exceeds_budget_l2786_278646

/-- Represents the paint requirements for a mansion --/
structure MansionPaint where
  bedroom_count : Nat
  bathroom_count : Nat
  kitchen_count : Nat
  living_room_count : Nat
  dining_room_count : Nat
  study_room_count : Nat
  bedroom_paint : Nat
  bathroom_paint : Nat
  kitchen_paint : Nat
  living_room_paint : Nat
  dining_room_paint : Nat
  study_room_paint : Nat
  colored_paint_price : Nat
  white_paint_can_size : Nat
  white_paint_can_price : Nat
  budget : Nat

/-- Calculates the total cost of paint for the mansion --/
def total_paint_cost (m : MansionPaint) : Nat :=
  let colored_paint_gallons := 
    m.bedroom_count * m.bedroom_paint +
    m.kitchen_count * m.kitchen_paint +
    m.living_room_count * m.living_room_paint +
    m.dining_room_count * m.dining_room_paint +
    m.study_room_count * m.study_room_paint
  let white_paint_gallons := m.bathroom_count * m.bathroom_paint
  let white_paint_cans := (white_paint_gallons + m.white_paint_can_size - 1) / m.white_paint_can_size
  colored_paint_gallons * m.colored_paint_price + white_paint_cans * m.white_paint_can_price

/-- Theorem stating that the total paint cost exceeds the budget --/
theorem paint_cost_exceeds_budget (m : MansionPaint) 
  (h : m = { bedroom_count := 5, bathroom_count := 10, kitchen_count := 1, 
             living_room_count := 2, dining_room_count := 1, study_room_count := 1,
             bedroom_paint := 3, bathroom_paint := 2, kitchen_paint := 4,
             living_room_paint := 6, dining_room_paint := 4, study_room_paint := 3,
             colored_paint_price := 18, white_paint_can_size := 3, 
             white_paint_can_price := 40, budget := 500 }) : 
  total_paint_cost m > m.budget := by
  sorry


end NUMINAMATH_CALUDE_paint_cost_exceeds_budget_l2786_278646


namespace NUMINAMATH_CALUDE_inverse_g_87_l2786_278656

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 6

-- State the theorem
theorem inverse_g_87 : g⁻¹ 87 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_87_l2786_278656


namespace NUMINAMATH_CALUDE_expression_simplification_l2786_278618

theorem expression_simplification :
  4 * Real.sqrt 2 * Real.sqrt 3 - Real.sqrt 12 / Real.sqrt 2 + Real.sqrt 24 = 5 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2786_278618


namespace NUMINAMATH_CALUDE_minimum_freight_charges_l2786_278641

theorem minimum_freight_charges 
  (total_trucks : ℕ) 
  (large_capacity small_capacity : ℕ) 
  (total_sugar : ℕ) 
  (large_freight_A small_freight_A : ℕ) 
  (large_freight_B small_freight_B : ℕ) 
  (trucks_to_A : ℕ) 
  (min_sugar_A : ℕ) :
  total_trucks = 20 →
  large_capacity = 15 →
  small_capacity = 10 →
  total_sugar = 240 →
  large_freight_A = 630 →
  small_freight_A = 420 →
  large_freight_B = 750 →
  small_freight_B = 550 →
  trucks_to_A = 10 →
  min_sugar_A = 115 →
  ∃ (large_A small_A large_B small_B : ℕ),
    large_A + small_A = trucks_to_A ∧
    large_B + small_B = total_trucks - trucks_to_A ∧
    large_A + large_B = 8 ∧
    small_A + small_B = 12 ∧
    large_capacity * large_A + small_capacity * small_A ≥ min_sugar_A ∧
    large_capacity * (large_A + large_B) + small_capacity * (small_A + small_B) = total_sugar ∧
    large_freight_A * large_A + small_freight_A * small_A + 
    large_freight_B * large_B + small_freight_B * small_B = 11330 ∧
    ∀ (x y z w : ℕ),
      x + y = trucks_to_A →
      z + w = total_trucks - trucks_to_A →
      x + z = 8 →
      y + w = 12 →
      large_capacity * x + small_capacity * y ≥ min_sugar_A →
      large_capacity * (x + z) + small_capacity * (y + w) = total_sugar →
      large_freight_A * x + small_freight_A * y + 
      large_freight_B * z + small_freight_B * w ≥ 11330 :=
by
  sorry


end NUMINAMATH_CALUDE_minimum_freight_charges_l2786_278641


namespace NUMINAMATH_CALUDE_circle_intersection_range_l2786_278670

theorem circle_intersection_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x - 2*Real.sqrt 3*y - m = 0 ∧ x^2 + y^2 = 1) ↔ 
  -3 ≤ m ∧ m ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l2786_278670


namespace NUMINAMATH_CALUDE_fraction_pattern_sum_of_fractions_sum_of_irrational_fractions_l2786_278691

-- Part 1: Pattern for positive integers
theorem fraction_pattern (n : ℕ+) : 
  1 / n.val * (1 / (n.val + 1)) = 1 / n.val - 1 / (n.val + 1) := by sorry

-- Part 2: Sum of fractions
theorem sum_of_fractions (x : ℝ) : 
  1 / (x * (x + 1)) + 1 / ((x + 1) * (x + 2)) + 1 / ((x + 2) * (x + 3)) + 1 / ((x + 3) * (x + 4)) = 
  4 / (x^2 + 4*x) := by sorry

-- Part 3: Sum of irrational fractions
theorem sum_of_irrational_fractions : 
  1 / (1 + Real.sqrt 2) + 1 / (Real.sqrt 2 + Real.sqrt 3) + 1 / (Real.sqrt 3 + 2) + 1 / (2 + Real.sqrt 5) +
  1 / (Real.sqrt 5 + Real.sqrt 6) + 1 / (Real.sqrt 6 + 3) + 1 / (3 + Real.sqrt 10) = 
  -1 + Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_fraction_pattern_sum_of_fractions_sum_of_irrational_fractions_l2786_278691


namespace NUMINAMATH_CALUDE_flower_bed_area_and_perimeter_l2786_278653

/-- Represents a rectangular flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Calculate the area of a rectangular flower bed -/
def area (fb : FlowerBed) : ℝ :=
  fb.length * fb.width

/-- Calculate the perimeter of a rectangular flower bed -/
def perimeter (fb : FlowerBed) : ℝ :=
  2 * (fb.length + fb.width)

theorem flower_bed_area_and_perimeter :
  let fb : FlowerBed := { length := 60, width := 45 }
  area fb = 2700 ∧ perimeter fb = 210 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_area_and_perimeter_l2786_278653


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l2786_278608

/-- A polynomial of the form x^3 - 2ax^2 + bx - 2a -/
def cubic_polynomial (a b : ℝ) (x : ℝ) : ℝ := x^3 - 2*a*x^2 + b*x - 2*a

/-- The condition that a polynomial has all real roots -/
def has_all_real_roots (p : ℝ → ℝ) : Prop :=
  ∃ r s t : ℝ, ∀ x : ℝ, p x = (x - r) * (x - s) * (x - t)

/-- The theorem stating the relationship between a and b for the given polynomial -/
theorem cubic_polynomial_roots (a b : ℝ) :
  (a > 0 ∧ a = 3 * Real.sqrt 3 / 2 ∧ b = 81 / 4) ↔
  (has_all_real_roots (cubic_polynomial a b) ∧
   ∀ a' > 0, has_all_real_roots (cubic_polynomial a' b) → a ≤ a') :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l2786_278608


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l2786_278631

-- Define what it means for three real numbers to form a geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r

-- Theorem statement
theorem geometric_sequence_condition (a b c : ℝ) :
  (is_geometric_sequence a b c → a * c = b^2) ∧
  ∃ a b c : ℝ, a * c = b^2 ∧ ¬is_geometric_sequence a b c :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l2786_278631


namespace NUMINAMATH_CALUDE_portfolio_growth_l2786_278685

/-- Calculates the final portfolio value after two years of investment -/
theorem portfolio_growth (initial_investment : ℝ) (growth_rate_1 : ℝ) (additional_investment : ℝ) (growth_rate_2 : ℝ) 
  (h1 : initial_investment = 80)
  (h2 : growth_rate_1 = 0.15)
  (h3 : additional_investment = 28)
  (h4 : growth_rate_2 = 0.10) :
  let value_after_year_1 := initial_investment * (1 + growth_rate_1)
  let value_before_year_2 := value_after_year_1 + additional_investment
  let final_value := value_before_year_2 * (1 + growth_rate_2)
  final_value = 132 := by
  sorry

end NUMINAMATH_CALUDE_portfolio_growth_l2786_278685


namespace NUMINAMATH_CALUDE_quadratic_sequence_inconsistency_l2786_278612

def isQuadraticSequence (seq : List ℤ) : Prop :=
  let firstDiffs := List.zipWith (·-·) seq.tail seq
  let secondDiffs := List.zipWith (·-·) firstDiffs.tail firstDiffs
  secondDiffs.all (· = secondDiffs.head!)

def findInconsistentTerm (seq : List ℤ) : Option ℤ :=
  let firstDiffs := List.zipWith (·-·) seq.tail seq
  let secondDiffs := List.zipWith (·-·) firstDiffs.tail firstDiffs
  if h : secondDiffs.all (· = secondDiffs.head!) then
    none
  else
    some (seq[secondDiffs.findIndex (· ≠ secondDiffs.head!) + 1]!)

theorem quadratic_sequence_inconsistency 
  (seq : List ℤ) 
  (hseq : seq = [2107, 2250, 2402, 2574, 2738, 2920, 3094, 3286]) : 
  ¬isQuadraticSequence seq ∧ findInconsistentTerm seq = some 2574 :=
sorry

end NUMINAMATH_CALUDE_quadratic_sequence_inconsistency_l2786_278612


namespace NUMINAMATH_CALUDE_gift_packaging_combinations_l2786_278674

/-- The number of varieties of packaging paper -/
def paper_varieties : ℕ := 10

/-- The number of colors of ribbon -/
def ribbon_colors : ℕ := 4

/-- The number of types of decorative stickers -/
def sticker_types : ℕ := 5

/-- The total number of gift packaging combinations -/
def total_combinations : ℕ := paper_varieties * ribbon_colors * sticker_types

theorem gift_packaging_combinations :
  total_combinations = 200 := by sorry

end NUMINAMATH_CALUDE_gift_packaging_combinations_l2786_278674


namespace NUMINAMATH_CALUDE_specific_pyramid_has_180_balls_l2786_278640

/-- Represents a pyramid display of balls -/
structure PyramidDisplay where
  bottomLayer : ℕ
  topLayer : ℕ
  difference : ℤ

/-- Calculates the total number of balls in a pyramid display -/
def totalBalls (p : PyramidDisplay) : ℕ :=
  sorry

/-- Theorem stating that the specific pyramid display has 180 balls -/
theorem specific_pyramid_has_180_balls :
  let p : PyramidDisplay := {
    bottomLayer := 35,
    topLayer := 1,
    difference := -4
  }
  totalBalls p = 180 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_has_180_balls_l2786_278640


namespace NUMINAMATH_CALUDE_correct_average_equals_initial_l2786_278613

theorem correct_average_equals_initial (n : ℕ) (initial_avg : ℚ) 
  (correct1 incorrect1 correct2 incorrect2 : ℚ) : 
  n = 15 → 
  initial_avg = 37 → 
  correct1 = 64 → 
  incorrect1 = 52 → 
  correct2 = 27 → 
  incorrect2 = 39 → 
  (n * initial_avg - incorrect1 - incorrect2 + correct1 + correct2) / n = initial_avg := by
  sorry

end NUMINAMATH_CALUDE_correct_average_equals_initial_l2786_278613


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l2786_278682

theorem reciprocal_of_negative_fraction (n : ℤ) (n_nonzero : n ≠ 0) :
  ((-1 : ℚ) / n)⁻¹ = -n := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l2786_278682


namespace NUMINAMATH_CALUDE_positive_number_square_root_l2786_278693

theorem positive_number_square_root (x : ℝ) : 
  x > 0 → (Real.sqrt ((4 * x) / 3) = x) → x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_square_root_l2786_278693


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2786_278689

def age_ratio (a_current : ℕ) (b_current : ℕ) : ℚ :=
  (a_current + 10 : ℚ) / (b_current - 10)

theorem age_ratio_problem (b_current : ℕ) (h1 : b_current = 39) (h2 : b_current + 9 = a_current) :
  age_ratio a_current b_current = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2786_278689


namespace NUMINAMATH_CALUDE_cos_24_cos_36_minus_sin_24_sin_36_l2786_278673

theorem cos_24_cos_36_minus_sin_24_sin_36 :
  Real.cos (24 * π / 180) * Real.cos (36 * π / 180) - 
  Real.sin (24 * π / 180) * Real.sin (36 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_24_cos_36_minus_sin_24_sin_36_l2786_278673


namespace NUMINAMATH_CALUDE_sticker_distribution_l2786_278644

/-- The number of ways to distribute n identical objects among k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 9 identical stickers among 3 distinct sheets of paper -/
theorem sticker_distribution : distribute 9 3 = 55 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2786_278644


namespace NUMINAMATH_CALUDE_least_four_prime_divisible_correct_l2786_278663

/-- The least positive integer divisible by four distinct primes -/
def least_four_prime_divisible : ℕ := 210

/-- A function that checks if a number is divisible by four distinct primes -/
def has_four_distinct_prime_divisors (n : ℕ) : Prop :=
  ∃ p q r s : ℕ, 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    n % p = 0 ∧ n % q = 0 ∧ n % r = 0 ∧ n % s = 0

theorem least_four_prime_divisible_correct :
  has_four_distinct_prime_divisors least_four_prime_divisible ∧
  ∀ m : ℕ, m < least_four_prime_divisible → ¬(has_four_distinct_prime_divisors m) :=
by sorry

end NUMINAMATH_CALUDE_least_four_prime_divisible_correct_l2786_278663


namespace NUMINAMATH_CALUDE_negation_of_neither_odd_l2786_278601

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem negation_of_neither_odd (a b : ℤ) : 
  ¬(¬(is_odd a) ∧ ¬(is_odd b)) ↔ (is_odd a ∨ is_odd b) :=
sorry

end NUMINAMATH_CALUDE_negation_of_neither_odd_l2786_278601


namespace NUMINAMATH_CALUDE_sequence_term_expression_l2786_278667

def S (n : ℕ) : ℤ := 2 * n^2 - n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then 2 else 4 * n - 3

theorem sequence_term_expression (n : ℕ) :
  n ≥ 1 → a n = S n - S (n - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_term_expression_l2786_278667


namespace NUMINAMATH_CALUDE_m_greater_than_n_l2786_278635

theorem m_greater_than_n (a : ℝ) : 5 * a^2 - a + 1 > 4 * a^2 + a - 1 := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l2786_278635


namespace NUMINAMATH_CALUDE_inequality_statements_l2786_278616

theorem inequality_statements :
  (∃ (a b : ℝ) (c : ℝ), c < 0 ∧ a < b ∧ c * a > c * b) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b → (2 * a * b) / (a + b) < Real.sqrt (a * b)) ∧
  (∀ (k : ℝ), k > 0 → ∀ (a b : ℝ), a > 0 → b > 0 → a * b = k → 
    (∀ (x y : ℝ), x > 0 → y > 0 → x * y = k → a + b ≤ x + y)) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → (a^2 + b^2) / 2 < (a + b)^2) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → (a + b)^2 ≥ a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_statements_l2786_278616


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2786_278659

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 2) = 8 → x = 66 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2786_278659


namespace NUMINAMATH_CALUDE_problem_solution_l2786_278695

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = 0) : 
  y = 21/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2786_278695


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2786_278642

/-- Set A defined by the quadratic inequality -/
def A (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

/-- Set B defined by the given inequality -/
def B : Set ℝ := {x | (x-3)*(2-x) ≥ 0}

/-- Theorem stating the condition for A to be a necessary but not sufficient condition for B -/
theorem necessary_not_sufficient_condition (a : ℝ) :
  (a > 0) → (∀ x, x ∈ B → x ∈ A a) ∧ (∃ x, x ∈ A a ∧ x ∉ B) ↔ a > 1 ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2786_278642


namespace NUMINAMATH_CALUDE_prime_square_remainders_mod_180_l2786_278688

theorem prime_square_remainders_mod_180 :
  ∃ (S : Finset Nat), 
    (∀ p : Nat, Prime p → p > 5 → ∃ r ∈ S, p^2 % 180 = r) ∧ 
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_remainders_mod_180_l2786_278688


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l2786_278625

theorem binomial_expansion_problem (x y : ℝ) (n : ℕ) 
  (h1 : n * x^(n-1) * y = 240)
  (h2 : n * (n-1) / 2 * x^(n-2) * y^2 = 720)
  (h3 : n * (n-1) * (n-2) / 6 * x^(n-3) * y^3 = 1080) :
  x = 2 ∧ y = 3 ∧ n = 5 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l2786_278625


namespace NUMINAMATH_CALUDE_family_boys_count_l2786_278680

/-- A family where one child has 3 brothers and 6 sisters, and another child has 4 brothers and 5 sisters -/
structure Family where
  total_children : ℕ
  child1_brothers : ℕ
  child1_sisters : ℕ
  child2_brothers : ℕ
  child2_sisters : ℕ
  h1 : child1_brothers = 3
  h2 : child1_sisters = 6
  h3 : child2_brothers = 4
  h4 : child2_sisters = 5

/-- The number of boys in the family -/
def num_boys (f : Family) : ℕ := f.child1_brothers + 1

theorem family_boys_count (f : Family) : num_boys f = 4 := by
  sorry

end NUMINAMATH_CALUDE_family_boys_count_l2786_278680


namespace NUMINAMATH_CALUDE_grapes_count_l2786_278638

/-- The number of grapes in Rob's bowl -/
def rob_grapes : ℕ := 25

/-- The number of grapes in Allie's bowl -/
def allie_grapes : ℕ := rob_grapes + 2

/-- The number of grapes in Allyn's bowl -/
def allyn_grapes : ℕ := allie_grapes + 4

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := rob_grapes + allie_grapes + allyn_grapes

theorem grapes_count : total_grapes = 83 := by
  sorry

end NUMINAMATH_CALUDE_grapes_count_l2786_278638


namespace NUMINAMATH_CALUDE_dan_bought_five_notebooks_l2786_278661

/-- Represents the purchase of school supplies -/
structure SchoolSupplies where
  totalSpent : ℕ
  backpackCost : ℕ
  penCost : ℕ
  pencilCost : ℕ
  notebookCost : ℕ

/-- Calculates the number of notebooks bought -/
def notebooksBought (supplies : SchoolSupplies) : ℕ :=
  (supplies.totalSpent - (supplies.backpackCost + supplies.penCost + supplies.pencilCost)) / supplies.notebookCost

/-- Theorem stating that Dan bought 5 notebooks -/
theorem dan_bought_five_notebooks (supplies : SchoolSupplies)
  (h1 : supplies.totalSpent = 32)
  (h2 : supplies.backpackCost = 15)
  (h3 : supplies.penCost = 1)
  (h4 : supplies.pencilCost = 1)
  (h5 : supplies.notebookCost = 3) :
  notebooksBought supplies = 5 := by
  sorry

end NUMINAMATH_CALUDE_dan_bought_five_notebooks_l2786_278661


namespace NUMINAMATH_CALUDE_internet_bill_is_100_l2786_278694

/-- Represents the financial transactions and balances in Liza's checking account --/
structure AccountState where
  initialBalance : ℕ
  rentPayment : ℕ
  paycheckDeposit : ℕ
  electricityBill : ℕ
  phoneBill : ℕ
  finalBalance : ℕ

/-- Calculates the internet bill given the account state --/
def calculateInternetBill (state : AccountState) : ℕ :=
  state.initialBalance + state.paycheckDeposit - state.rentPayment - state.electricityBill - state.phoneBill - state.finalBalance

/-- Theorem stating that the internet bill is $100 given the specified account state --/
theorem internet_bill_is_100 (state : AccountState) 
  (h1 : state.initialBalance = 800)
  (h2 : state.rentPayment = 450)
  (h3 : state.paycheckDeposit = 1500)
  (h4 : state.electricityBill = 117)
  (h5 : state.phoneBill = 70)
  (h6 : state.finalBalance = 1563) :
  calculateInternetBill state = 100 := by
  sorry

end NUMINAMATH_CALUDE_internet_bill_is_100_l2786_278694


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l2786_278655

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ballsInBoxes (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 82 ways to put 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : ballsInBoxes 6 3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l2786_278655


namespace NUMINAMATH_CALUDE_handshake_theorem_l2786_278607

theorem handshake_theorem (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧ 
  (∃ (f : Fin n → ℕ), (∀ x, f x ≤ n - 1) ∧ f i = k ∧ f j = k) :=
sorry

end NUMINAMATH_CALUDE_handshake_theorem_l2786_278607


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2786_278684

theorem average_marks_chemistry_mathematics 
  (P C M : ℕ) 
  (h : P + C + M = P + 110) : 
  (C + M) / 2 = 55 := by
sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2786_278684


namespace NUMINAMATH_CALUDE_tangent_line_constant_l2786_278600

theorem tangent_line_constant (a k b : ℝ) : 
  (∀ x, a * x^2 + 2 + Real.log x = k * x + b) →  -- The line is tangent to the curve
  (1 : ℝ)^2 * a + 2 + Real.log 1 = k * 1 + b →   -- The point (1, 4) lies on both the line and curve
  (4 : ℝ) = k * 1 + b →                          -- The y-coordinate of P is 4
  (∀ x, 2 * a * x + 1 / x = k) →                 -- The derivatives are equal at x = 1
  b = -1 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_constant_l2786_278600


namespace NUMINAMATH_CALUDE_cuboid_colored_cubes_theorem_l2786_278632

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of cubes colored on only one side when a cuboid is cut into unit cubes -/
def cubesColoredOnOneSide (c : Cuboid) : ℕ :=
  2 * ((c.width - 2) * (c.length - 2) + (c.width - 2) * (c.height - 2) + (c.length - 2) * (c.height - 2))

theorem cuboid_colored_cubes_theorem (c : Cuboid) 
    (h_width : c.width = 5)
    (h_length : c.length = 4)
    (h_height : c.height = 3) :
  cubesColoredOnOneSide c = 22 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_colored_cubes_theorem_l2786_278632


namespace NUMINAMATH_CALUDE_solution_exists_unique_solution_l2786_278665

theorem solution_exists : ∃ x : ℚ, 60 + x * 12 / (180 / 3) = 61 :=
by
  use 5
  sorry

theorem unique_solution (x : ℚ) : 60 + x * 12 / (180 / 3) = 61 ↔ x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_exists_unique_solution_l2786_278665


namespace NUMINAMATH_CALUDE_sum_of_solutions_equals_sqrt_five_l2786_278630

theorem sum_of_solutions_equals_sqrt_five (x₀ y₀ : ℝ) 
  (h1 : y₀ = 1 / x₀) 
  (h2 : y₀ = |x₀| + 1) : 
  x₀ + y₀ = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_equals_sqrt_five_l2786_278630


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2786_278615

theorem gcd_of_three_numbers : Nat.gcd 1734 (Nat.gcd 816 1343) = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2786_278615


namespace NUMINAMATH_CALUDE_power_division_sum_product_difference_l2786_278664

theorem power_division_sum_product_difference (a b c d e f g : ℤ) :
  a = -4 ∧ b = 4 ∧ c = 2 ∧ d = 3 ∧ e = 7 →
  a^6 / b^4 + c^5 * d - e^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_power_division_sum_product_difference_l2786_278664


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l2786_278671

theorem initial_number_of_persons (n : ℕ) 
  (avg_weight_increase : ℝ) 
  (weight_difference : ℝ) : 
  avg_weight_increase = 2.5 →
  weight_difference = 20 →
  weight_difference = n * avg_weight_increase →
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l2786_278671


namespace NUMINAMATH_CALUDE_geometry_theorem_l2786_278666

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem geometry_theorem 
  (m n : Line) (α β : Plane)
  (h_distinct_lines : m ≠ n)
  (h_distinct_planes : α ≠ β) :
  (∀ (m : Line) (β α : Plane), 
    subset m β → plane_parallel α β → parallel m α) ∧
  (∀ (m n : Line) (α β : Plane),
    perpendicular m α → perpendicular n β → plane_parallel α β → line_parallel m n) :=
by sorry

end NUMINAMATH_CALUDE_geometry_theorem_l2786_278666


namespace NUMINAMATH_CALUDE_store_change_calculation_l2786_278611

def payment : ℕ := 20
def num_items : ℕ := 3
def item_cost : ℕ := 2

theorem store_change_calculation :
  payment - (num_items * item_cost) = 14 := by
  sorry

end NUMINAMATH_CALUDE_store_change_calculation_l2786_278611


namespace NUMINAMATH_CALUDE_opposite_of_2021_l2786_278643

theorem opposite_of_2021 : ∃ x : ℝ, x + 2021 = 0 ∧ x = -2021 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2021_l2786_278643


namespace NUMINAMATH_CALUDE_outbound_speed_calculation_l2786_278639

-- Define the problem parameters
def distance : ℝ := 19.999999999999996
def return_speed : ℝ := 4
def total_time : ℝ := 5.8

-- Define the theorem
theorem outbound_speed_calculation :
  ∃ (v : ℝ), v > 0 ∧ (distance / v + distance / return_speed = total_time) → v = 25 := by
  sorry

end NUMINAMATH_CALUDE_outbound_speed_calculation_l2786_278639


namespace NUMINAMATH_CALUDE_vanya_finished_first_l2786_278629

/-- Represents a participant in the competition -/
structure Participant where
  name : String
  predicted_position : Nat
  actual_position : Nat

/-- The competition setup and results -/
structure Competition where
  participants : List Participant
  vanya : Participant

/-- Axioms for the competition -/
axiom all_positions_different (c : Competition) :
  ∀ p1 p2 : Participant, p1 ∈ c.participants → p2 ∈ c.participants → p1 ≠ p2 →
    p1.actual_position ≠ p2.actual_position

axiom vanya_predicted_last (c : Competition) :
  c.vanya.predicted_position = c.participants.length

axiom others_worse_than_predicted (c : Competition) :
  ∀ p : Participant, p ∈ c.participants → p ≠ c.vanya →
    p.actual_position > p.predicted_position

/-- Theorem: Vanya must have finished first -/
theorem vanya_finished_first (c : Competition) :
  c.vanya.actual_position = 1 :=
sorry

end NUMINAMATH_CALUDE_vanya_finished_first_l2786_278629


namespace NUMINAMATH_CALUDE_solve_for_e_l2786_278677

-- Define the functions p and q
def p (x : ℝ) : ℝ := 5 * x - 17
def q (e : ℝ) (x : ℝ) : ℝ := 4 * x - e

-- State the theorem
theorem solve_for_e : 
  ∀ e : ℝ, p (q e 3) = 23 → e = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_e_l2786_278677


namespace NUMINAMATH_CALUDE_compare_sqrt_expressions_l2786_278626

theorem compare_sqrt_expressions : 7 * Real.sqrt 2 < 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_compare_sqrt_expressions_l2786_278626


namespace NUMINAMATH_CALUDE_vector_coordinates_l2786_278623

/-- Given a vector a with magnitude √5 that is parallel to vector b=(1,2),
    prove that the coordinates of a are either (1,2) or (-1,-2) -/
theorem vector_coordinates (a b : ℝ × ℝ) : 
  (‖a‖ = Real.sqrt 5) → 
  (b = (1, 2)) → 
  (∃ (k : ℝ), a = k • b) → 
  (a = (1, 2) ∨ a = (-1, -2)) := by
  sorry

#check vector_coordinates

end NUMINAMATH_CALUDE_vector_coordinates_l2786_278623


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2786_278604

theorem inequality_equivalence (x : ℝ) : 2 - x > 3 + x ↔ x < -1/2 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2786_278604


namespace NUMINAMATH_CALUDE_total_marbles_l2786_278603

/-- Represents the number of marbles each boy has -/
structure MarbleDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The ratio of marbles between the three boys -/
def marbleRatio : MarbleDistribution := ⟨5, 2, 3⟩

/-- The number of additional marbles the first boy has -/
def additionalMarbles : ℕ := 3

/-- The number of marbles the middle (second) boy has -/
def middleBoyMarbles : ℕ := 12

/-- The theorem stating the total number of marbles -/
theorem total_marbles :
  ∃ (x : ℕ),
    x * marbleRatio.second = middleBoyMarbles ∧
    (x * marbleRatio.first + additionalMarbles) +
    (x * marbleRatio.second) +
    (x * marbleRatio.third) = 63 := by
  sorry


end NUMINAMATH_CALUDE_total_marbles_l2786_278603


namespace NUMINAMATH_CALUDE_journey_speed_l2786_278651

theorem journey_speed (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) :
  total_distance = 240 ∧ 
  total_time = 20 ∧ 
  first_half_speed = 10 →
  (total_distance / 2) / ((total_time - (total_distance / 2) / first_half_speed)) = 15 :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_l2786_278651


namespace NUMINAMATH_CALUDE_odd_prime_sum_iff_floor_sum_odd_l2786_278645

theorem odd_prime_sum_iff_floor_sum_odd (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1)
  (a b : ℕ) (ha : 0 < a ∧ a < p) (hb : 0 < b ∧ b < p) :
  a + b = p ↔
  ∀ n : ℕ, 0 < n → n < p →
    ∃ k : ℕ, Int.floor ((2 * a * n : ℚ) / p) + Int.floor ((2 * b * n : ℚ) / p) = 2 * k + 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_prime_sum_iff_floor_sum_odd_l2786_278645


namespace NUMINAMATH_CALUDE_bag_balls_count_l2786_278668

theorem bag_balls_count (blue_balls : ℕ) (prob_blue : ℚ) : 
  blue_balls = 8 →
  prob_blue = 1/3 →
  ∃ (green_balls : ℕ),
    (blue_balls : ℚ) / ((blue_balls : ℚ) + (green_balls : ℚ)) = prob_blue ∧
    blue_balls + green_balls = 24 :=
by sorry

end NUMINAMATH_CALUDE_bag_balls_count_l2786_278668


namespace NUMINAMATH_CALUDE_essay_time_calculation_l2786_278686

/-- The time Rachel spent on her essay -/
def essay_time (
  page_writing_time : ℕ)  -- Time to write one page in seconds
  (research_time : ℕ)     -- Time spent researching in seconds
  (outline_time : ℕ)      -- Time spent on outline in minutes
  (brainstorm_time : ℕ)   -- Time spent brainstorming in seconds
  (total_pages : ℕ)       -- Total number of pages written
  (break_time : ℕ)        -- Break time after each page in seconds
  (editing_time : ℕ)      -- Time spent editing in seconds
  (proofreading_time : ℕ) -- Time spent proofreading in seconds
  : ℚ :=
  let total_seconds : ℕ := 
    research_time + 
    (outline_time * 60) + 
    brainstorm_time + 
    (total_pages * page_writing_time) + 
    (total_pages * break_time) + 
    editing_time + 
    proofreading_time
  (total_seconds : ℚ) / 3600

theorem essay_time_calculation : 
  essay_time 1800 2700 15 1200 6 600 4500 1800 = 25500 / 3600 := by
  sorry

#eval essay_time 1800 2700 15 1200 6 600 4500 1800

end NUMINAMATH_CALUDE_essay_time_calculation_l2786_278686


namespace NUMINAMATH_CALUDE_largest_integer_solution_l2786_278620

theorem largest_integer_solution (x : ℤ) : (∀ y : ℤ, -y + 3 > 1 → y ≤ x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l2786_278620


namespace NUMINAMATH_CALUDE_frieda_corner_probability_l2786_278621

/-- Represents the different types of squares on the 4x4 grid -/
inductive GridSquare
| Corner
| Edge
| Center

/-- Represents the possible directions of movement -/
inductive Direction
| Up
| Down
| Left
| Right

/-- Represents the state of Frieda on the grid -/
structure FriedaState :=
(position : GridSquare)
(hops : Nat)

/-- The probability of reaching a corner square within n hops -/
def probability_reach_corner (n : Nat) (start : GridSquare) : Rat :=
sorry

/-- The main theorem stating the probability of reaching a corner within 5 hops -/
theorem frieda_corner_probability :
  probability_reach_corner 5 GridSquare.Edge = 299 / 1024 :=
sorry

end NUMINAMATH_CALUDE_frieda_corner_probability_l2786_278621


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2786_278649

theorem largest_prime_factor_of_expression : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (20^4 + 15^3 - 10^5) ∧ 
  ∀ (q : ℕ), q.Prime → q ∣ (20^4 + 15^3 - 10^5) → q ≤ p ∧ p = 1787 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2786_278649


namespace NUMINAMATH_CALUDE_two_fifths_of_number_l2786_278678

theorem two_fifths_of_number (n : ℝ) : (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 16 → (2/5 : ℝ) * n = 192 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_of_number_l2786_278678


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l2786_278669

/-- Calculates the total charge for a taxi trip -/
def total_charge (initial_fee : ℚ) (charge_per_increment : ℚ) (increment_distance : ℚ) (trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / increment_distance).floor * charge_per_increment

theorem taxi_charge_calculation :
  let initial_fee : ℚ := 9/4  -- $2.25
  let charge_per_increment : ℚ := 7/20  -- $0.35
  let increment_distance : ℚ := 2/5  -- 2/5 mile
  let trip_distance : ℚ := 18/5  -- 3.6 miles
  total_charge initial_fee charge_per_increment increment_distance trip_distance = 27/5  -- $5.40
:= by sorry

end NUMINAMATH_CALUDE_taxi_charge_calculation_l2786_278669


namespace NUMINAMATH_CALUDE_representative_distribution_l2786_278698

/-- The number of ways to distribute n items into k groups with at least one item in each group -/
def distribute_with_minimum (n k : ℕ) : ℕ := sorry

/-- The number of classes from which representatives are selected -/
def num_classes : ℕ := 4

/-- The total number of student representatives to be selected -/
def total_representatives : ℕ := 6

/-- Theorem stating that the number of ways to distribute 6 representatives among 4 classes,
    with at least one representative in each class, is equal to 10 -/
theorem representative_distribution :
  distribute_with_minimum total_representatives num_classes = 10 := by sorry

end NUMINAMATH_CALUDE_representative_distribution_l2786_278698


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2786_278633

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2786_278633


namespace NUMINAMATH_CALUDE_triangle_perimeter_sum_limit_l2786_278627

/-- Given an initial equilateral triangle with side length b, and a sequence of triangles
    where each subsequent triangle has sides 1/3 the length of the previous one,
    the limit of the sum of the perimeters of all triangles is 9b/2. -/
theorem triangle_perimeter_sum_limit (b : ℝ) (h : b > 0) :
  let perimeter_seq := fun n : ℕ => b * (3 : ℝ) * (1 / 3) ^ n
  ∑' n, perimeter_seq n = (9 * b) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_sum_limit_l2786_278627


namespace NUMINAMATH_CALUDE_inequality_comparison_l2786_278692

theorem inequality_comparison (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) (hc : c ≠ 0) :
  (∀ c, a + c > b + c) ∧
  (∀ c, a - 3*c > b - 3*c) ∧
  (¬∀ c, a*c > b*c) ∧
  (∀ c, a/c^2 > b/c^2) ∧
  (∀ c, a*c^3 > b*c^3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_comparison_l2786_278692


namespace NUMINAMATH_CALUDE_new_lights_total_wattage_l2786_278606

def increase_wattage (w : ℝ) : ℝ := w * 1.25

def original_wattages : List ℝ := [60, 80, 100, 120]

theorem new_lights_total_wattage :
  (original_wattages.map increase_wattage).sum = 450 := by
  sorry

end NUMINAMATH_CALUDE_new_lights_total_wattage_l2786_278606


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2786_278675

/-- Given a hyperbola with center (2, 0), one focus at (2, 8), and one vertex at (2, 5),
    prove that h + k + a + b = 7 + √39, where (h, k) is the center, a is the distance
    from the center to a vertex, and b is derived from b^2 = c^2 - a^2, with c being
    the distance from the center to a focus. -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 2 ∧ k = 0 ∧ a = 5 ∧ c = 8 ∧ b^2 = c^2 - a^2 →
  h + k + a + b = 7 + Real.sqrt 39 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2786_278675


namespace NUMINAMATH_CALUDE_four_solutions_l2786_278614

/-- The number of positive integer solutions to the equation 3x + y = 15 -/
def solution_count : Nat :=
  (Finset.filter (fun p : Nat × Nat => 3 * p.1 + p.2 = 15 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 15) (Finset.range 15))).card

/-- Theorem stating that there are exactly 4 pairs of positive integers (x, y) satisfying 3x + y = 15 -/
theorem four_solutions : solution_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_solutions_l2786_278614


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2786_278602

/-- Given a line segment from (2, 5) to (x, y) with length 10, prove that (x, y) = (8, 13) -/
theorem line_segment_endpoint (x y : ℝ) (h1 : x > 2) (h2 : y > 5) 
  (h3 : Real.sqrt ((x - 2)^2 + (y - 5)^2) = 10) : x = 8 ∧ y = 13 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2786_278602


namespace NUMINAMATH_CALUDE_range_of_a_l2786_278657

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else Real.log x

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → -1 ≤ a ∧ a < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2786_278657


namespace NUMINAMATH_CALUDE_proposition_relationship_l2786_278624

theorem proposition_relationship :
  ∀ (p q : Prop),
  (p → q) →                        -- Proposition A: p is sufficient for q
  (p ↔ q) →                        -- Proposition B: p is necessary and sufficient for q
  ((p ↔ q) → (p → q)) ∧            -- Proposition A is necessary for Proposition B
  ¬((p → q) → (p ↔ q)) :=          -- Proposition A is not sufficient for Proposition B
by
  sorry

end NUMINAMATH_CALUDE_proposition_relationship_l2786_278624


namespace NUMINAMATH_CALUDE_hallway_tiling_l2786_278681

theorem hallway_tiling (hallway_length hallway_width : ℕ) 
  (border_tile_size interior_tile_size : ℕ) : 
  hallway_length = 20 → 
  hallway_width = 14 → 
  border_tile_size = 2 → 
  interior_tile_size = 3 → 
  (2 * (hallway_length - 2 * border_tile_size) / border_tile_size + 
   2 * (hallway_width - 2 * border_tile_size) / border_tile_size + 4) + 
  ((hallway_length - 2 * border_tile_size) * 
   (hallway_width - 2 * border_tile_size)) / (interior_tile_size^2) = 48 := by
  sorry

end NUMINAMATH_CALUDE_hallway_tiling_l2786_278681


namespace NUMINAMATH_CALUDE_jason_initial_cards_l2786_278647

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 4

/-- The number of Pokemon cards Jason has now -/
def cards_remaining : ℕ := 5

/-- The initial number of Pokemon cards Jason had -/
def initial_cards : ℕ := cards_given_away + cards_remaining

theorem jason_initial_cards : initial_cards = 9 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l2786_278647


namespace NUMINAMATH_CALUDE_outside_classroom_trash_l2786_278648

def total_trash : ℕ := 1576
def classroom_trash : ℕ := 344

theorem outside_classroom_trash :
  total_trash - classroom_trash = 1232 := by
  sorry

end NUMINAMATH_CALUDE_outside_classroom_trash_l2786_278648


namespace NUMINAMATH_CALUDE_hyperbola_equation_hyperbola_final_equation_l2786_278609

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, prove that if its eccentricity is 2
    and its asymptotes are tangent to the circle (x-a)² + y² = 3/4, then a = 1 and b = √3. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (2 : ℝ) = (Real.sqrt (a^2 + b^2)) / a  -- eccentricity is 2
  → (∃ (x y : ℝ), (y = (b/a) * x ∨ y = -(b/a) * x) ∧ (x - a)^2 + y^2 = 3/4)  -- asymptotes tangent to circle
  → a = 1 ∧ b = Real.sqrt 3 :=
by sorry

/-- The equation of the hyperbola is x² - y²/3 = 1. -/
theorem hyperbola_final_equation (x y : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (2 : ℝ) = (Real.sqrt (a^2 + b^2)) / a ∧
    (∃ (x' y' : ℝ), (y' = (b/a) * x' ∨ y' = -(b/a) * x') ∧ (x' - a)^2 + y'^2 = 3/4) ∧
    x^2 / a^2 - y^2 / b^2 = 1)
  → x^2 - y^2 / 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_hyperbola_final_equation_l2786_278609


namespace NUMINAMATH_CALUDE_chicken_price_per_pound_l2786_278619

-- Define the given values
def num_steaks : ℕ := 4
def steak_weight : ℚ := 1/2
def steak_price_per_pound : ℚ := 15
def chicken_weight : ℚ := 3/2
def total_spent : ℚ := 42

-- Define the theorem
theorem chicken_price_per_pound :
  (total_spent - (num_steaks * steak_weight * steak_price_per_pound)) / chicken_weight = 8 := by
  sorry

end NUMINAMATH_CALUDE_chicken_price_per_pound_l2786_278619


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2786_278628

theorem consecutive_integers_sum (n : ℕ) (h : n > 0) :
  (n * (n + 1) * (n + 2) = 336) → (n + (n + 1) + (n + 2) = 21) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2786_278628


namespace NUMINAMATH_CALUDE_inequality_proof_l2786_278622

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  a / (b + c^2) + b / (c + a^2) + c / (a + b^2) ≥ 9/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2786_278622


namespace NUMINAMATH_CALUDE_craig_total_apples_l2786_278696

def craig_initial_apples : ℝ := 20.0
def eugene_apples : ℝ := 7.0

theorem craig_total_apples : 
  craig_initial_apples + eugene_apples = 27.0 := by sorry

end NUMINAMATH_CALUDE_craig_total_apples_l2786_278696


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2786_278672

theorem coin_flip_probability (p : ℝ) (h : p = 1 / 2) :
  p * p * (1 - p) * (1 - p) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2786_278672


namespace NUMINAMATH_CALUDE_weight_four_moles_CaBr2_l2786_278634

/-- The atomic weight of calcium in g/mol -/
def calcium_weight : ℝ := 40.08

/-- The atomic weight of bromine in g/mol -/
def bromine_weight : ℝ := 79.904

/-- The number of calcium atoms in a molecule of CaBr2 -/
def calcium_atoms : ℕ := 1

/-- The number of bromine atoms in a molecule of CaBr2 -/
def bromine_atoms : ℕ := 2

/-- The number of moles of CaBr2 -/
def moles_CaBr2 : ℝ := 4

/-- The weight of a given number of moles of CaBr2 -/
def weight_CaBr2 (moles : ℝ) : ℝ :=
  moles * (calcium_atoms * calcium_weight + bromine_atoms * bromine_weight)

/-- Theorem stating that the weight of 4 moles of CaBr2 is 799.552 grams -/
theorem weight_four_moles_CaBr2 :
  weight_CaBr2 moles_CaBr2 = 799.552 := by sorry

end NUMINAMATH_CALUDE_weight_four_moles_CaBr2_l2786_278634


namespace NUMINAMATH_CALUDE_print_shop_cost_difference_l2786_278610

def print_shop_x_price : ℝ := 1.25
def print_shop_y_price : ℝ := 2.75
def print_shop_x_discount : ℝ := 0.10
def print_shop_y_discount : ℝ := 0.05
def print_shop_x_tax : ℝ := 0.07
def print_shop_y_tax : ℝ := 0.09
def num_copies : ℕ := 40

def calculate_total_cost (base_price discount tax : ℝ) (copies : ℕ) : ℝ :=
  let pre_discount := base_price * copies
  let discounted := pre_discount * (1 - discount)
  discounted * (1 + tax)

theorem print_shop_cost_difference :
  calculate_total_cost print_shop_y_price print_shop_y_discount print_shop_y_tax num_copies -
  calculate_total_cost print_shop_x_price print_shop_x_discount print_shop_x_tax num_copies =
  65.755 := by sorry

end NUMINAMATH_CALUDE_print_shop_cost_difference_l2786_278610
