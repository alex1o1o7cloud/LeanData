import Mathlib

namespace NUMINAMATH_CALUDE_power_of_power_l1997_199705

theorem power_of_power : (3^4)^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1997_199705


namespace NUMINAMATH_CALUDE_cafe_tables_l1997_199786

def base5_to_decimal (n : Nat) : Nat :=
  3 * 5^2 + 1 * 5^1 + 0 * 5^0

theorem cafe_tables :
  let total_chairs := base5_to_decimal 310
  let people_per_table := 3
  (total_chairs / people_per_table : Nat) = 26 := by
  sorry

end NUMINAMATH_CALUDE_cafe_tables_l1997_199786


namespace NUMINAMATH_CALUDE_interest_rate_difference_l1997_199740

/-- Given a principal amount, time period, and difference in interest earned,
    calculate the difference between two simple interest rates. -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_diff : ℝ)
  (h_principal : principal = 2300)
  (h_time : time = 3)
  (h_interest_diff : interest_diff = 69) :
  let rate_diff := interest_diff / (principal * time / 100)
  rate_diff = 1 := by sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l1997_199740


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1997_199723

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x + y = x * y) :
  x + y ≥ 13 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1997_199723


namespace NUMINAMATH_CALUDE_card_distribution_l1997_199703

theorem card_distribution (total_cards : ℕ) (num_people : ℕ) 
  (h1 : total_cards = 60) (h2 : num_people = 9) : 
  ∃ (people_with_fewer : ℕ), people_with_fewer = 3 ∧ 
  people_with_fewer = num_people - (total_cards % num_people) :=
by
  sorry

end NUMINAMATH_CALUDE_card_distribution_l1997_199703


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_product_l1997_199712

theorem seventh_root_of_unity_product (z : ℂ) (h1 : z^7 = 1) (h2 : z ≠ 1) :
  (z - 1) * (z^2 - 1) * (z^3 - 1) * (z^4 - 1) * (z^5 - 1) * (z^6 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_product_l1997_199712


namespace NUMINAMATH_CALUDE_boys_camp_total_l1997_199736

theorem boys_camp_total (total : ℕ) : 
  (total : ℝ) * 0.2 * 0.7 = 42 → total = 300 := by
  sorry

end NUMINAMATH_CALUDE_boys_camp_total_l1997_199736


namespace NUMINAMATH_CALUDE_no_formula_matches_l1997_199731

/-- Represents the given formulas --/
inductive Formula
  | A
  | B
  | C
  | D

/-- Evaluates a formula for a given x --/
def evaluate (f : Formula) (x : ℝ) : ℝ :=
  match f with
  | .A => x^3 + 3*x + 3
  | .B => x^2 + 4*x + 3
  | .C => x^3 + x^2 + 2*x + 1
  | .D => 2*x^3 - x + 5

/-- The set of given (x, y) pairs --/
def pairs : List (ℝ × ℝ) := [(1, 7), (2, 17), (3, 31), (4, 49), (5, 71)]

/-- Checks if a formula matches all given pairs --/
def matchesAll (f : Formula) : Prop :=
  ∀ (p : ℝ × ℝ), p ∈ pairs → evaluate f p.1 = p.2

theorem no_formula_matches : ∀ (f : Formula), ¬(matchesAll f) := by
  sorry

end NUMINAMATH_CALUDE_no_formula_matches_l1997_199731


namespace NUMINAMATH_CALUDE_problem_solution_l1997_199706

theorem problem_solution : (2010^2 - 2010) / 2010^2 = 2009 / 2010 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1997_199706


namespace NUMINAMATH_CALUDE_train_speed_theorem_l1997_199760

-- Define the length of the train in meters
def train_length : ℝ := 300

-- Define the time taken to cross the platform in seconds
def crossing_time : ℝ := 60

-- Define the total distance covered (train length + platform length)
def total_distance : ℝ := 2 * train_length

-- Define the speed conversion factor from m/s to km/h
def speed_conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_theorem :
  let speed_ms := total_distance / crossing_time
  let speed_kmh := speed_ms * speed_conversion_factor
  speed_kmh = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_theorem_l1997_199760


namespace NUMINAMATH_CALUDE_emmanuel_december_charges_l1997_199728

/-- Emmanuel's total charges for December -/
def total_charges (regular_plan_cost : ℝ) (days_in_guam : ℕ) (international_data_cost : ℝ) : ℝ :=
  regular_plan_cost + (days_in_guam : ℝ) * international_data_cost

/-- Theorem: Emmanuel's total charges for December are $210 -/
theorem emmanuel_december_charges :
  total_charges 175 10 3.5 = 210 := by
  sorry

end NUMINAMATH_CALUDE_emmanuel_december_charges_l1997_199728


namespace NUMINAMATH_CALUDE_cos_four_thirds_pi_plus_alpha_l1997_199792

theorem cos_four_thirds_pi_plus_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos ((4 / 3) * π + α) = -(1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_cos_four_thirds_pi_plus_alpha_l1997_199792


namespace NUMINAMATH_CALUDE_fraction_equality_l1997_199701

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x - 2 * y) / (2 * x + 6 * y) = 3) : 
  (2 * x - 6 * y) / (4 * x + y) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1997_199701


namespace NUMINAMATH_CALUDE_select_two_from_seven_l1997_199753

theorem select_two_from_seven : Nat.choose 7 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_select_two_from_seven_l1997_199753


namespace NUMINAMATH_CALUDE_apples_remaining_l1997_199770

def initial_apples : ℕ := 128
def sale_percentage : ℚ := 25 / 100

theorem apples_remaining (initial : ℕ) (sale_percent : ℚ) : 
  initial - ⌊initial * sale_percent⌋ - ⌊(initial - ⌊initial * sale_percent⌋) * sale_percent⌋ - 1 = 71 :=
by sorry

end NUMINAMATH_CALUDE_apples_remaining_l1997_199770


namespace NUMINAMATH_CALUDE_number_relations_with_180_l1997_199752

theorem number_relations_with_180 :
  (∃ n : ℤ, n = 180 + 15 ∧ n = 195) ∧
  (∃ m : ℤ, m = 180 - 15 ∧ m = 165) := by
  sorry

end NUMINAMATH_CALUDE_number_relations_with_180_l1997_199752


namespace NUMINAMATH_CALUDE_fourth_number_value_l1997_199766

theorem fourth_number_value (numbers : List ℝ) 
  (h1 : numbers.length = 6)
  (h2 : numbers.sum / numbers.length = 30)
  (h3 : (numbers.take 4).sum / 4 = 25)
  (h4 : (numbers.drop 3).sum / 3 = 35) :
  numbers[3] = 25 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_value_l1997_199766


namespace NUMINAMATH_CALUDE_final_balance_percentage_l1997_199749

def starting_balance : ℝ := 125
def initial_increase : ℝ := 0.25
def first_usd_to_eur : ℝ := 0.85
def decrease_in_eur : ℝ := 0.20
def eur_to_usd : ℝ := 1.15
def increase_in_usd : ℝ := 0.15
def decrease_in_usd : ℝ := 0.10
def final_usd_to_eur : ℝ := 0.88

theorem final_balance_percentage (starting_balance initial_increase first_usd_to_eur
  decrease_in_eur eur_to_usd increase_in_usd decrease_in_usd final_usd_to_eur : ℝ) :
  let initial_eur := starting_balance * (1 + initial_increase) * first_usd_to_eur
  let after_decrease_eur := initial_eur * (1 - decrease_in_eur)
  let back_to_usd := after_decrease_eur * eur_to_usd
  let after_increase_usd := back_to_usd * (1 + increase_in_usd)
  let after_decrease_usd := after_increase_usd * (1 - decrease_in_usd)
  let final_eur := after_decrease_usd * final_usd_to_eur
  let starting_eur := starting_balance * first_usd_to_eur
  (final_eur / starting_eur) * 100 = 104.75 :=
by sorry

end NUMINAMATH_CALUDE_final_balance_percentage_l1997_199749


namespace NUMINAMATH_CALUDE_fourth_animal_is_sheep_l1997_199722

/-- Represents the different types of animals -/
inductive Animal
  | Horse
  | Cow
  | Pig
  | Sheep
  | Rabbit
  | Squirrel

/-- The sequence of animals entering the fence -/
def animalSequence : List Animal :=
  [Animal.Horse, Animal.Cow, Animal.Pig, Animal.Sheep, Animal.Rabbit, Animal.Squirrel]

/-- Theorem stating that the 4th animal in the sequence is a sheep -/
theorem fourth_animal_is_sheep :
  animalSequence[3] = Animal.Sheep := by sorry

end NUMINAMATH_CALUDE_fourth_animal_is_sheep_l1997_199722


namespace NUMINAMATH_CALUDE_chocolate_chip_cookie_coverage_l1997_199707

theorem chocolate_chip_cookie_coverage : 
  let cookie_radius : ℝ := 3
  let chip_radius : ℝ := 0.3
  let cookie_area : ℝ := π * cookie_radius^2
  let chip_area : ℝ := π * chip_radius^2
  let coverage_ratio : ℝ := 1/4
  let num_chips : ℕ := 25
  (↑num_chips * chip_area = coverage_ratio * cookie_area) ∧ 
  (∀ k : ℕ, k ≠ num_chips → ↑k * chip_area ≠ coverage_ratio * cookie_area) :=
by sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookie_coverage_l1997_199707


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1997_199768

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation : 
  let l1 : Line := { a := 4, b := -5, c := 9 }
  let p : Point := { x := 4, y := 1 }
  let l2 : Line := { a := 5, b := 4, c := -24 }
  (l2.contains p ∧ Line.perpendicular l1 l2) → 
  ∀ (x y : ℝ), 5 * x + 4 * y - 24 = 0 ↔ l2.contains { x := x, y := y } :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1997_199768


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1997_199739

theorem complex_fraction_simplification :
  ((3 + 2*Complex.I) / (2 - 3*Complex.I)) - ((3 - 2*Complex.I) / (2 + 3*Complex.I)) = 2*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1997_199739


namespace NUMINAMATH_CALUDE_factorial_ratio_l1997_199711

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_ratio : factorial 4 / factorial (4 - 3) = 24 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1997_199711


namespace NUMINAMATH_CALUDE_transform_second_to_third_l1997_199776

/-- A point in the 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the second quadrant. -/
def isInSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Transforms a point according to the given rule. -/
def transformPoint (p : Point2D) : Point2D :=
  ⟨3 * p.x - 2, -p.y⟩

/-- Determines if a point is in the third quadrant. -/
def isInThirdQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- 
Theorem: If a point is in the second quadrant, 
then its transformed point is in the third quadrant.
-/
theorem transform_second_to_third (p : Point2D) :
  isInSecondQuadrant p → isInThirdQuadrant (transformPoint p) := by
  sorry

end NUMINAMATH_CALUDE_transform_second_to_third_l1997_199776


namespace NUMINAMATH_CALUDE_solve_for_r_l1997_199779

theorem solve_for_r : 
  let r := Real.sqrt (8^2 + 15^2) / Real.sqrt 25
  r = 17 / 5 := by sorry

end NUMINAMATH_CALUDE_solve_for_r_l1997_199779


namespace NUMINAMATH_CALUDE_lara_swimming_theorem_l1997_199777

/-- The number of minutes Lara must swim on the ninth day to average 100 minutes per day over 9 days -/
def minutes_to_swim_on_ninth_day (
  days_at_80_min : ℕ)  -- Number of days Lara swam 80 minutes
  (days_at_105_min : ℕ) -- Number of days Lara swam 105 minutes
  (target_average : ℕ)  -- Target average minutes per day
  (total_days : ℕ)      -- Total number of days
  : ℕ :=
  target_average * total_days - (days_at_80_min * 80 + days_at_105_min * 105)

/-- Theorem stating the correct number of minutes Lara must swim on the ninth day -/
theorem lara_swimming_theorem :
  minutes_to_swim_on_ninth_day 6 2 100 9 = 210 := by
  sorry

end NUMINAMATH_CALUDE_lara_swimming_theorem_l1997_199777


namespace NUMINAMATH_CALUDE_unique_solution_3644_l1997_199758

def repeating_decimal_ab (a b : ℕ) : ℚ := (10 * a + b : ℚ) / 99

def repeating_decimal_abcd (a b c d : ℕ) : ℚ := (1000 * a + 100 * b + 10 * c + d : ℚ) / 9999

theorem unique_solution_3644 (a b c d : ℕ) :
  a ∈ Finset.range 10 →
  b ∈ Finset.range 10 →
  c ∈ Finset.range 10 →
  d ∈ Finset.range 10 →
  repeating_decimal_ab a b + repeating_decimal_abcd a b c d = 27 / 37 →
  a = 3 ∧ b = 6 ∧ c = 4 ∧ d = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_3644_l1997_199758


namespace NUMINAMATH_CALUDE_f_properties_l1997_199746

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x - 1) / (a^x + 1)

theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1997_199746


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l1997_199716

/-- The y-coordinate of the point on the y-axis equidistant from (1, 0) and (4, 3) -/
theorem equidistant_point_y_coordinate : ∃ y : ℝ, 
  (Real.sqrt ((1 - 0)^2 + (0 - y)^2) = Real.sqrt ((4 - 0)^2 + (3 - y)^2)) ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l1997_199716


namespace NUMINAMATH_CALUDE_chord_cosine_l1997_199785

theorem chord_cosine (r : ℝ) (γ δ : ℝ) : 
  0 < r →
  0 < γ →
  0 < δ →
  γ + δ < π →
  5^2 = 2 * r^2 * (1 - Real.cos γ) →
  12^2 = 2 * r^2 * (1 - Real.cos δ) →
  13^2 = 2 * r^2 * (1 - Real.cos (γ + δ)) →
  Real.cos γ = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_chord_cosine_l1997_199785


namespace NUMINAMATH_CALUDE_farmer_apples_l1997_199733

/-- The number of apples given to the neighbor -/
def apples_given (initial current : ℕ) : ℕ := initial - current

/-- Theorem: The number of apples given to the neighbor is the difference between
    the initial number of apples and the current number of apples -/
theorem farmer_apples (initial current : ℕ) (h : initial ≥ current) :
  apples_given initial current = initial - current :=
by sorry

end NUMINAMATH_CALUDE_farmer_apples_l1997_199733


namespace NUMINAMATH_CALUDE_non_redundant_password_count_l1997_199767

/-- A password is a string of characters. -/
def Password := String

/-- The set of available characters for passwords. -/
def AvailableChars : Finset Char := sorry

/-- A password is redundant if it contains a block of consecutive characters
    that can be colored red and blue such that the red and blue substrings are identical. -/
def IsRedundant (p : Password) : Prop := sorry

/-- The number of non-redundant passwords of length n. -/
def NonRedundantCount (n : ℕ) : ℕ := sorry

/-- There are at least 18^n non-redundant passwords of length n for any n ≥ 1. -/
theorem non_redundant_password_count (n : ℕ) (h : n ≥ 1) :
  NonRedundantCount n ≥ 18^n := by sorry

end NUMINAMATH_CALUDE_non_redundant_password_count_l1997_199767


namespace NUMINAMATH_CALUDE_complex_sum_roots_of_unity_l1997_199751

theorem complex_sum_roots_of_unity (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  (Finset.range 16).sum (λ k => ω^(20 + 4*k)) = -ω^2 := by sorry

end NUMINAMATH_CALUDE_complex_sum_roots_of_unity_l1997_199751


namespace NUMINAMATH_CALUDE_gumball_distribution_l1997_199778

/-- Theorem: Gumball Distribution
Given:
- Joanna initially had 40 gumballs
- Jacques initially had 60 gumballs
- They each purchased 4 times their initial amount
- They put all gumballs together and shared equally

Prove: Each person gets 250 gumballs -/
theorem gumball_distribution (joanna_initial : Nat) (jacques_initial : Nat)
  (h1 : joanna_initial = 40)
  (h2 : jacques_initial = 60)
  (purchase_multiplier : Nat)
  (h3 : purchase_multiplier = 4) :
  let joanna_total := joanna_initial + joanna_initial * purchase_multiplier
  let jacques_total := jacques_initial + jacques_initial * purchase_multiplier
  let total_gumballs := joanna_total + jacques_total
  (total_gumballs / 2 : Nat) = 250 := by
  sorry

end NUMINAMATH_CALUDE_gumball_distribution_l1997_199778


namespace NUMINAMATH_CALUDE_quadrilateral_sum_of_squares_l1997_199756

/-- A quadrilateral with sides a, b, c, d, diagonals m, n, and distance t between midpoints of diagonals -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  m : ℝ
  n : ℝ
  t : ℝ

/-- The sum of squares of sides equals the sum of squares of diagonals plus four times the square of the distance between midpoints of diagonals -/
theorem quadrilateral_sum_of_squares (q : Quadrilateral) :
  q.a^2 + q.b^2 + q.c^2 + q.d^2 = q.m^2 + q.n^2 + 4 * q.t^2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_sum_of_squares_l1997_199756


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l1997_199715

theorem cubic_sum_theorem (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l1997_199715


namespace NUMINAMATH_CALUDE_triangle_area_l1997_199745

/-- Given a triangle ABC with sides AC = 8 and BC = 10, and the condition that 32 cos(A - B) = 31,
    prove that the area of the triangle is 15√7. -/
theorem triangle_area (A B C : ℝ) (AC BC : ℝ) (h1 : AC = 8) (h2 : BC = 10) 
    (h3 : 32 * Real.cos (A - B) = 31) : 
    (1/2 : ℝ) * AC * BC * Real.sin (A + B - π) = 15 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1997_199745


namespace NUMINAMATH_CALUDE_popsicle_melting_speed_l1997_199782

theorem popsicle_melting_speed (n : ℕ) (a : ℕ → ℝ) :
  n = 6 →
  (∀ i, 1 ≤ i → i < n → a (i + 1) = 2 * a i) →
  a n = 32 * a 1 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_melting_speed_l1997_199782


namespace NUMINAMATH_CALUDE_correct_calculation_l1997_199725

theorem correct_calculation : 
  (-2 + 3 = 1) ∧ 
  (-2 - 3 ≠ 1) ∧ 
  (-2 / (-1/2) ≠ 1) ∧ 
  ((-2)^3 ≠ -6) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1997_199725


namespace NUMINAMATH_CALUDE_blueberries_count_l1997_199730

/-- Represents the number of blueberries in each blue box -/
def blueberries : ℕ := sorry

/-- Represents the number of strawberries in each red box -/
def strawberries : ℕ := sorry

/-- The increase in total berries when replacing a blue box with a red box -/
def total_increase : ℕ := 20

/-- The increase in difference between strawberries and blueberries after replacement -/
def difference_increase : ℕ := 80

theorem blueberries_count : blueberries = 60 :=
  by sorry

end NUMINAMATH_CALUDE_blueberries_count_l1997_199730


namespace NUMINAMATH_CALUDE_inverse_function_inequality_l1997_199755

/-- A function satisfying f(x₁x₂) = f(x₁) + f(x₂) for positive x₁ and x₂ -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂

/-- Theorem statement for the given problem -/
theorem inverse_function_inequality (f : ℝ → ℝ) (hf : FunctionalEquation f)
    (hfinv : ∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ Function.RightInverse g f) :
    ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 →
      f⁻¹ x₁ + f⁻¹ x₂ ≥ 2 * (f⁻¹ (x₁ / 2) * f⁻¹ (x₂ / 2)) :=
  sorry

end NUMINAMATH_CALUDE_inverse_function_inequality_l1997_199755


namespace NUMINAMATH_CALUDE_sad_children_count_l1997_199747

theorem sad_children_count (total : ℕ) (happy : ℕ) (neither : ℕ) (boys : ℕ) (girls : ℕ) 
  (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ) : ℕ :=
  by
  -- Assume the given conditions
  have h1 : total = 60 := by sorry
  have h2 : happy = 30 := by sorry
  have h3 : neither = 20 := by sorry
  have h4 : boys = 16 := by sorry
  have h5 : girls = 44 := by sorry
  have h6 : happy_boys = 6 := by sorry
  have h7 : sad_girls = 4 := by sorry
  have h8 : neither_boys = 4 := by sorry

  -- Prove that the number of sad children is 10
  exact total - happy - neither

end NUMINAMATH_CALUDE_sad_children_count_l1997_199747


namespace NUMINAMATH_CALUDE_haley_cider_production_l1997_199791

/-- Represents the number of pints of cider Haley can make -/
def cider_pints (golden_per_pint pink_per_pint farmhands apples_per_hour work_hours golden_ratio pink_ratio : ℕ) : ℕ :=
  let total_apples := farmhands * apples_per_hour * work_hours
  let apples_per_pint := golden_per_pint + pink_per_pint
  total_apples / apples_per_pint

/-- Theorem stating that Haley can make 120 pints of cider given the conditions -/
theorem haley_cider_production :
  cider_pints 20 40 6 240 5 1 2 = 120 := by
  sorry

#eval cider_pints 20 40 6 240 5 1 2

end NUMINAMATH_CALUDE_haley_cider_production_l1997_199791


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1997_199783

/-- Given a hyperbola with one focus at (2√5, 0) and asymptotes y = ±(1/2)x, 
    its standard equation is x²/16 - y²/4 = 1 -/
theorem hyperbola_equation (f : ℝ × ℝ) (m : ℝ) :
  f = (2 * Real.sqrt 5, 0) →
  m = 1/2 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 ↔
      (y = m*x ∨ y = -m*x) ∧ 
      (x - f.1)^2 / a^2 - (y - f.2)^2 / b^2 = 1) ∧
    a^2 = 16 ∧ b^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1997_199783


namespace NUMINAMATH_CALUDE_travel_time_to_madison_l1997_199784

/-- Represents the travel time problem from Gardensquare to Madison -/
theorem travel_time_to_madison 
  (map_distance : ℝ) 
  (map_scale : ℝ) 
  (average_speed : ℝ) 
  (h1 : map_distance = 5) 
  (h2 : map_scale = 0.016666666666666666) 
  (h3 : average_speed = 60) : 
  map_distance / (map_scale * average_speed) = 5 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_to_madison_l1997_199784


namespace NUMINAMATH_CALUDE_courier_cost_formula_l1997_199789

/-- The cost function for a courier service --/
def courier_cost (P : ℕ) : ℕ :=
  5 + 12 + 5 * (P - 1)

/-- Theorem: The courier cost is equal to 5P + 12 --/
theorem courier_cost_formula (P : ℕ) (h : P ≥ 1) : courier_cost P = 5 * P + 12 := by
  sorry

#check courier_cost_formula

end NUMINAMATH_CALUDE_courier_cost_formula_l1997_199789


namespace NUMINAMATH_CALUDE_inequality_solution_l1997_199741

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the inequality function
def f (x m : ℝ) : Prop := lg x ^ 2 - (2 + m) * lg x + m - 1 > 0

-- State the theorem
theorem inequality_solution :
  ∀ m : ℝ, |m| ≤ 1 →
    {x : ℝ | f x m} = {x : ℝ | 0 < x ∧ x < (1/10) ∨ x > 1000} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1997_199741


namespace NUMINAMATH_CALUDE_sequences_theorem_l1997_199714

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n + 1

-- Define the sum S_n of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (n + 2)

-- Define the geometric sequence b_n
def b (n : ℕ) : ℚ := 3^n

-- Define T_n as the sum of the first n terms of 1/S_n
def T (n : ℕ) : ℚ := 3/4 - (2*n + 3) / (2 * (n+1) * (n+2))

-- State the theorem
theorem sequences_theorem (n : ℕ) : 
  (a n = 2 * n + 1) ∧ 
  (b n = 3^n) ∧ 
  (T n = 3/4 - (2*n + 3) / (2 * (n+1) * (n+2))) ∧
  (a 1 = b 1) ∧ 
  (a 4 = b 2) ∧ 
  (a 13 = b 3) :=
by sorry

end NUMINAMATH_CALUDE_sequences_theorem_l1997_199714


namespace NUMINAMATH_CALUDE_age_difference_l1997_199718

theorem age_difference (masc_age sam_age : ℕ) : 
  masc_age > sam_age →
  masc_age + sam_age = 27 →
  masc_age = 17 →
  sam_age = 10 →
  masc_age - sam_age = 7 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1997_199718


namespace NUMINAMATH_CALUDE_positive_integer_pairs_satisfying_equation_l1997_199796

theorem positive_integer_pairs_satisfying_equation :
  ∀ x y : ℕ+, 
    (x : ℤ)^2 + (y : ℤ)^2 - 5*(x : ℤ)*(y : ℤ) + 5 = 0 ↔ 
    ((x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) ∨ (x = 9 ∧ y = 2) ∨ (x = 1 ∧ y = 2)) :=
by sorry

#check positive_integer_pairs_satisfying_equation

end NUMINAMATH_CALUDE_positive_integer_pairs_satisfying_equation_l1997_199796


namespace NUMINAMATH_CALUDE_sqrt_256_equals_2_to_n_l1997_199737

theorem sqrt_256_equals_2_to_n (n : ℕ) : (256 : ℝ)^(1/2) = 2^n → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_256_equals_2_to_n_l1997_199737


namespace NUMINAMATH_CALUDE_equation_solution_l1997_199724

theorem equation_solution : ∃ x : ℝ, x ≠ -2 ∧ (4*x^2 - 3*x + 2) / (x + 2) = 4*x - 3 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1997_199724


namespace NUMINAMATH_CALUDE_selling_price_l1997_199769

/-- Represents the labelled price of a refrigerator -/
def R : ℝ := sorry

/-- Represents the labelled price of a washing machine -/
def W : ℝ := sorry

/-- The condition that the total discounted price is 35000 -/
axiom purchase_price : 0.80 * R + 0.85 * W = 35000

/-- The theorem stating the selling price formula -/
theorem selling_price : 
  0.80 * R + 0.85 * W = 35000 → 
  (1.10 * R + 1.12 * W) = (1.10 * R + 1.12 * W) :=
by sorry

end NUMINAMATH_CALUDE_selling_price_l1997_199769


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l1997_199732

theorem arctan_equation_solution (x : ℝ) : 
  x = Real.sqrt ((1 - 3 * Real.sqrt 3) / 13) ∨ 
  x = -Real.sqrt ((1 - 3 * Real.sqrt 3) / 13) → 
  Real.arctan (2 / x) + Real.arctan (1 / (2 * x^2)) = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l1997_199732


namespace NUMINAMATH_CALUDE_baseball_price_proof_l1997_199795

/-- The price of a basketball in dollars -/
def basketball_price : ℝ := 29

/-- The number of basketballs bought by Coach A -/
def num_basketballs : ℕ := 10

/-- The number of baseballs bought by Coach B -/
def num_baseballs : ℕ := 14

/-- The price of the baseball bat in dollars -/
def bat_price : ℝ := 18

/-- The difference in spending between Coach A and Coach B in dollars -/
def spending_difference : ℝ := 237

/-- The price of a baseball in dollars -/
def baseball_price : ℝ := 2.5

theorem baseball_price_proof :
  num_basketballs * basketball_price = 
  num_baseballs * baseball_price + bat_price + spending_difference :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_price_proof_l1997_199795


namespace NUMINAMATH_CALUDE_log_27_3_l1997_199708

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  -- Define 27 as 3³
  have h : 27 = 3^3 := by norm_num
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_log_27_3_l1997_199708


namespace NUMINAMATH_CALUDE_sally_pokemon_cards_l1997_199771

theorem sally_pokemon_cards (initial cards_from_dan cards_bought cards_traded cards_lost : ℕ) 
  (h1 : initial = 27)
  (h2 : cards_from_dan = 41)
  (h3 : cards_bought = 20)
  (h4 : cards_traded = 15)
  (h5 : cards_lost = 7) :
  initial + cards_from_dan + cards_bought - cards_traded - cards_lost = 66 := by
  sorry

end NUMINAMATH_CALUDE_sally_pokemon_cards_l1997_199771


namespace NUMINAMATH_CALUDE_geometric_series_relation_l1997_199799

/-- Given real numbers c and d satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 5/7 -/
theorem geometric_series_relation (c d : ℝ) 
    (h : (∑' n, c / d^n) = 5) :
    (∑' n, c / (c + 2*d)^n) = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l1997_199799


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1997_199781

theorem solution_satisfies_system :
  let x₁ : ℚ := 1
  let x₂ : ℚ := -1
  let x₃ : ℚ := 1
  let x₄ : ℚ := -1
  let x₅ : ℚ := 1
  (x₁ + 2*x₂ + 2*x₃ + 2*x₄ + 2*x₅ = 1) ∧
  (x₁ + 3*x₂ + 4*x₃ + 4*x₄ + 4*x₅ = 2) ∧
  (x₁ + 3*x₂ + 5*x₃ + 6*x₄ + 6*x₅ = 3) ∧
  (x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 8*x₅ = 4) ∧
  (x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ = 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l1997_199781


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l1997_199773

theorem complex_magnitude_equality (t : ℝ) (h : t > 0) :
  Complex.abs (-4 + 2 * t * Complex.I) = 3 * Real.sqrt 5 ↔ t = Real.sqrt 29 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l1997_199773


namespace NUMINAMATH_CALUDE_black_white_area_ratio_l1997_199709

/-- The ratio of black to white area in concentric circles -/
theorem black_white_area_ratio :
  let radii : Fin 5 → ℝ := ![2, 4, 6, 8, 10]
  let circle_area (r : ℝ) := π * r^2
  let ring_area (i : Fin 4) := circle_area (radii (i + 1)) - circle_area (radii i)
  let black_area := circle_area (radii 0) + ring_area 1 + ring_area 3
  let white_area := ring_area 0 + ring_area 2
  black_area / white_area = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_black_white_area_ratio_l1997_199709


namespace NUMINAMATH_CALUDE_min_value_ab_l1997_199727

theorem min_value_ab (b a : ℝ) (h1 : b > 0)
  (h2 : (b^2 + 1) * (-1 / a) = 1 / b^2) : 
  ∀ x : ℝ, a * b ≥ 2 ∧ (a * b = 2 ↔ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l1997_199727


namespace NUMINAMATH_CALUDE_parallel_lines_d_value_l1997_199720

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

/-- The slope of the first line -/
def slope₁ : ℝ := -3

/-- The slope of the second line -/
def slope₂ (d : ℝ) : ℝ := -6 * d

theorem parallel_lines_d_value :
  ∀ d : ℝ, parallel slope₁ (slope₂ d) → d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_d_value_l1997_199720


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1997_199750

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x ∈ Set.Ici (-2) → x + 3 ≥ 1)) ↔ 
  (∃ x : ℝ, x ∈ Set.Ici (-2) ∧ x + 3 < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1997_199750


namespace NUMINAMATH_CALUDE_water_in_sport_is_105_l1997_199763

/-- Represents the ratios of ingredients in a flavored drink formulation -/
structure DrinkFormulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the flavored drink -/
def standard : DrinkFormulation :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

/-- The sport formulation of the flavored drink -/
def sport : DrinkFormulation :=
  { flavoring := standard.flavoring,
    corn_syrup := standard.corn_syrup / 3,
    water := standard.water * 2 }

/-- The amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 7

/-- Calculates the amount of water in the sport formulation -/
def water_in_sport : ℚ :=
  (sport_corn_syrup * sport.water) / sport.corn_syrup

/-- Theorem stating that the amount of water in the sport formulation is 105 ounces -/
theorem water_in_sport_is_105 : water_in_sport = 105 := by
  sorry


end NUMINAMATH_CALUDE_water_in_sport_is_105_l1997_199763


namespace NUMINAMATH_CALUDE_rhombus_dot_product_l1997_199793

/-- A rhombus OABC in a Cartesian coordinate system -/
structure Rhombus where
  O : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Vector representation -/
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The main theorem -/
theorem rhombus_dot_product (r : Rhombus) : 
  r.O = (0, 0) → 
  r.A = (1, 1) → 
  dot_product (vec r.O r.A) (vec r.O r.C) = 1 → 
  dot_product (vec r.A r.B) (vec r.A r.C) = 1 := by
  sorry

#check rhombus_dot_product

end NUMINAMATH_CALUDE_rhombus_dot_product_l1997_199793


namespace NUMINAMATH_CALUDE_function_properties_l1997_199729

noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := (x^2 + 1) / (b * x + c)

theorem function_properties (b c : ℝ) :
  (∀ x : ℝ, x ≠ 0 → b * x + c ≠ 0) →
  f b c 1 = 2 →
  (∃ g : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f b c x = g x) ∧ 
                (∀ x : ℝ, x ≠ 0 → g x = x + 1/x) ∧
                (∀ x y : ℝ, 1 ≤ x ∧ x < y → g x < g y) ∧
                (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → g x ≤ 5/2) ∧
                (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → 2 ≤ g x) ∧
                g 2 = 5/2 ∧
                g 1 = 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1997_199729


namespace NUMINAMATH_CALUDE_solve_system_l1997_199748

theorem solve_system (B C : ℝ) (eq1 : 5 * B - 3 = 32) (eq2 : 2 * B + 2 * C = 18) :
  B = 7 ∧ C = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1997_199748


namespace NUMINAMATH_CALUDE_investment_options_count_l1997_199719

/-- The number of ways to distribute 3 distinct projects among 5 cities, 
    with no more than 2 projects per city. -/
def investmentOptions : ℕ := 120

/-- The number of cities available for investment. -/
def numCities : ℕ := 5

/-- The number of projects to be distributed. -/
def numProjects : ℕ := 3

/-- The maximum number of projects allowed in a single city. -/
def maxProjectsPerCity : ℕ := 2

theorem investment_options_count :
  investmentOptions = 
    (numCities.factorial / (numCities - numProjects).factorial) +
    (numCities.choose 1) * (numProjects.choose 2) * ((numCities - 1).choose 1) :=
by sorry

end NUMINAMATH_CALUDE_investment_options_count_l1997_199719


namespace NUMINAMATH_CALUDE_teal_survey_l1997_199798

theorem teal_survey (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 150)
  (h2 : more_green = 90)
  (h3 : both = 40)
  (h4 : neither = 20) :
  ∃ more_blue : ℕ, more_blue = 80 ∧ 
    total = more_green + more_blue - both + neither :=
by sorry

end NUMINAMATH_CALUDE_teal_survey_l1997_199798


namespace NUMINAMATH_CALUDE_feed_animals_theorem_l1997_199742

/-- The number of ways to feed animals in a conservatory -/
def feed_animals (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * n * feed_animals (n - 1)

/-- Theorem: Given 5 pairs of different animals, alternating between male and female,
    and starting with a female hippopotamus, there are 2880 ways to complete feeding all animals -/
theorem feed_animals_theorem : feed_animals 5 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_feed_animals_theorem_l1997_199742


namespace NUMINAMATH_CALUDE_single_burger_cost_is_one_l1997_199772

/-- Calculates the cost of a single burger given the total spent, total number of hamburgers,
    number of double burgers, and cost of a double burger. -/
def single_burger_cost (total_spent : ℚ) (total_burgers : ℕ) (double_burgers : ℕ) (double_cost : ℚ) : ℚ :=
  let single_burgers := total_burgers - double_burgers
  let double_total := double_burgers * double_cost
  let single_total := total_spent - double_total
  single_total / single_burgers

/-- Proves that the cost of a single burger is $1.00 given the specified conditions. -/
theorem single_burger_cost_is_one :
  single_burger_cost 70.50 50 41 1.50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_single_burger_cost_is_one_l1997_199772


namespace NUMINAMATH_CALUDE_inequality_proof_l1997_199794

theorem inequality_proof (a b c : ℝ) (M N P : ℝ)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < 1)
  (hM : M = 2^a) (hN : N = 5^(-b)) (hP : P = Real.log c) :
  P < N ∧ N < M := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1997_199794


namespace NUMINAMATH_CALUDE_grasshopper_jumps_l1997_199775

/-- Represents the state of three objects in a line -/
inductive Position
| ABC
| ACB
| BAC
| BCA
| CAB
| CBA

/-- Represents a single jump of one object over another -/
def jump (p : Position) : Position :=
  match p with
  | Position.ABC => Position.BAC
  | Position.ACB => Position.CAB
  | Position.BAC => Position.BCA
  | Position.BCA => Position.CBA
  | Position.CAB => Position.ACB
  | Position.CBA => Position.ABC

/-- Applies n jumps to a given position -/
def jumpN (p : Position) (n : Nat) : Position :=
  match n with
  | 0 => p
  | n + 1 => jump (jumpN p n)

theorem grasshopper_jumps (n : Nat) (h : Odd n) :
  ∀ p : Position, jumpN p n ≠ p :=
sorry

end NUMINAMATH_CALUDE_grasshopper_jumps_l1997_199775


namespace NUMINAMATH_CALUDE_specific_triangle_l1997_199790

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

/-- The main theorem about the specific acute triangle -/
theorem specific_triangle (t : AcuteTriangle) 
  (h1 : t.a = 2 * t.b * Real.sin t.A)
  (h2 : t.a = 3 * Real.sqrt 3)
  (h3 : t.c = 5) :
  t.B = π/6 ∧ t.b = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_specific_triangle_l1997_199790


namespace NUMINAMATH_CALUDE_sum_of_exponents_l1997_199702

theorem sum_of_exponents (x y z : ℕ) 
  (h : 800670 = 8 * 10^x + 6 * 10^y + 7 * 10^z) : 
  x + y + z = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_exponents_l1997_199702


namespace NUMINAMATH_CALUDE_first_chapter_pages_l1997_199788

theorem first_chapter_pages (total_chapters : Nat) (second_chapter_pages : Nat) (third_chapter_pages : Nat) (total_pages : Nat)
  (h1 : total_chapters = 3)
  (h2 : second_chapter_pages = 35)
  (h3 : third_chapter_pages = 24)
  (h4 : total_pages = 125) :
  total_pages - (second_chapter_pages + third_chapter_pages) = 66 := by
  sorry

end NUMINAMATH_CALUDE_first_chapter_pages_l1997_199788


namespace NUMINAMATH_CALUDE_parabola_a_range_l1997_199774

/-- Parabola defined by y = ax^2 - 2a^2x + c -/
structure Parabola where
  a : ℝ
  c : ℝ
  h_a : a ≠ 0
  h_c : c > 0

/-- Point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y = p.a * x^2 - 2 * p.a^2 * x + p.c

theorem parabola_a_range (p : Parabola) 
  (point1 : PointOnParabola p) 
  (point2 : PointOnParabola p)
  (h_x1 : point1.x = 2 * p.a + 1)
  (h_x2 : 2 ≤ point2.x ∧ point2.x ≤ 4)
  (h_y : point1.y > p.c ∧ p.c > point2.y) :
  p.a > 2 ∨ p.a < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_a_range_l1997_199774


namespace NUMINAMATH_CALUDE_prime_power_plus_one_prime_l1997_199717

theorem prime_power_plus_one_prime (x y z : ℕ) : 
  Prime x ∧ Prime y ∧ Prime z ∧ x^y + 1 = z → (x = 2 ∧ y = 2 ∧ z = 5) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_plus_one_prime_l1997_199717


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l1997_199700

def U : Set Int := {-2, -1, 1, 3, 5}
def A : Set Int := {-1, 3}

theorem complement_of_A_wrt_U :
  U \ A = {-2, 1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l1997_199700


namespace NUMINAMATH_CALUDE_range_of_a_equiv_l1997_199726

/-- Proposition p: The equation x² + 2ax + 1 = 0 has two real roots greater than -1 -/
def prop_p (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > -1 ∧ y > -1 ∧ x ≠ y ∧ x^2 + 2*a*x + 1 = 0 ∧ y^2 + 2*a*y + 1 = 0

/-- Proposition q: The solution set of the inequality ax² - ax + 1 > 0 with respect to x is ℝ -/
def prop_q (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - a*x + 1 > 0

/-- The main theorem stating the equivalence of the conditions and the range of a -/
theorem range_of_a_equiv (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬prop_q a ↔ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_equiv_l1997_199726


namespace NUMINAMATH_CALUDE_matrix_not_invertible_sum_l1997_199765

/-- Given a 3x3 matrix with real entries x, y, z in the form
    [[x, y, z], [y, z, x], [z, x, y]],
    if the matrix is not invertible, then the sum
    x/(y+z) + y/(z+x) + z/(x+y) is equal to either -3 or 3/2 -/
theorem matrix_not_invertible_sum (x y z : ℝ) :
  let M := ![![x, y, z], ![y, z, x], ![z, x, y]]
  ¬ IsUnit (Matrix.det M) →
  (x / (y + z) + y / (z + x) + z / (x + y) = -3) ∨
  (x / (y + z) + y / (z + x) + z / (x + y) = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_sum_l1997_199765


namespace NUMINAMATH_CALUDE_inequality_characterization_l1997_199735

theorem inequality_characterization (x y : ℝ) :
  2 * |x + y| ≤ |x| + |y| ↔
  (x ≥ 0 ∧ -3 * x ≤ y ∧ y ≤ -(1/3) * x) ∨
  (x < 0 ∧ -(1/3) * x ≤ y ∧ y ≤ -3 * x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_characterization_l1997_199735


namespace NUMINAMATH_CALUDE_total_yellow_balloons_l1997_199734

/-- The total number of yellow balloons given the number of balloons each person has -/
def total_balloons (fred_balloons sam_balloons mary_balloons : ℕ) : ℕ :=
  fred_balloons + sam_balloons + mary_balloons

/-- Theorem stating that the total number of yellow balloons is 18 -/
theorem total_yellow_balloons :
  total_balloons 5 6 7 = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_yellow_balloons_l1997_199734


namespace NUMINAMATH_CALUDE_resort_tips_fraction_l1997_199797

theorem resort_tips_fraction (total_months : ℕ) (special_month_factor : ℕ) 
  (h1 : total_months = 7) 
  (h2 : special_month_factor = 4) : 
  (special_month_factor : ℚ) / ((total_months - 1 : ℕ) + special_month_factor : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_resort_tips_fraction_l1997_199797


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l1997_199761

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The circle E passing through (0,0), (1,1), and (2,0) -/
def circle_E : Circle :=
  { center := (1, 0), radius := 1 }

/-- Point P -/
def point_P : ℝ × ℝ := (2, 3)

/-- Theorem stating the properties of circle E and line l -/
theorem circle_and_tangent_line :
  (∀ (x y : ℝ), (x - circle_E.center.1)^2 + (y - circle_E.center.2)^2 = circle_E.radius^2 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 0)) ∧
  (∃ (l : Line), 
    (l.a * point_P.1 + l.b * point_P.2 + l.c = 0) ∧
    (∀ (x y : ℝ), (x - circle_E.center.1)^2 + (y - circle_E.center.2)^2 = circle_E.radius^2 →
      (l.a * x + l.b * y + l.c)^2 ≥ (l.a^2 + l.b^2) * circle_E.radius^2) ∧
    ((l.a = 1 ∧ l.b = 0 ∧ l.c = -2) ∨ (l.a = 4 ∧ l.b = -3 ∧ l.c = 1))) :=
sorry


end NUMINAMATH_CALUDE_circle_and_tangent_line_l1997_199761


namespace NUMINAMATH_CALUDE_surface_area_of_drilled_cube_l1997_199743

-- Define the cube
def cube_side_length : ℝ := 10

-- Define points on the cube
def point_A : ℝ × ℝ × ℝ := (0, 0, 0)
def point_G : ℝ × ℝ × ℝ := (cube_side_length, cube_side_length, cube_side_length)

-- Define the distance of H, I, J from A
def distance_from_A : ℝ := 3

-- Define the solid T
def solid_T (cube_side_length : ℝ) (distance_from_A : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

-- Calculate the surface area of the solid T
def surface_area_T (t : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem surface_area_of_drilled_cube :
  surface_area_T (solid_T cube_side_length distance_from_A) = 526.5 := by sorry

end NUMINAMATH_CALUDE_surface_area_of_drilled_cube_l1997_199743


namespace NUMINAMATH_CALUDE_cindy_jump_rope_time_l1997_199780

/-- Cindy's jump rope time in minutes -/
def cindy_time : ℕ := 12

/-- Betsy's jump rope time in minutes -/
def betsy_time : ℕ := cindy_time / 2

/-- Tina's jump rope time in minutes -/
def tina_time : ℕ := 3 * betsy_time

theorem cindy_jump_rope_time :
  cindy_time = 12 ∧
  betsy_time = cindy_time / 2 ∧
  tina_time = 3 * betsy_time ∧
  tina_time = cindy_time + 6 :=
by sorry

end NUMINAMATH_CALUDE_cindy_jump_rope_time_l1997_199780


namespace NUMINAMATH_CALUDE_second_outlet_pipe_rate_l1997_199754

-- Define the volume of the tank in cubic inches
def tank_volume : ℝ := 30 * 1728

-- Define the inlet pipe rate in cubic inches per minute
def inlet_rate : ℝ := 3

-- Define the first outlet pipe rate in cubic inches per minute
def outlet_rate_1 : ℝ := 6

-- Define the time to empty the tank in minutes
def emptying_time : ℝ := 3456

-- Define the unknown rate of the second outlet pipe
def outlet_rate_2 : ℝ := 12

-- Theorem statement
theorem second_outlet_pipe_rate : 
  tank_volume / (outlet_rate_1 + outlet_rate_2 - inlet_rate) = emptying_time :=
sorry

end NUMINAMATH_CALUDE_second_outlet_pipe_rate_l1997_199754


namespace NUMINAMATH_CALUDE_only_ball_draw_is_classical_l1997_199710

/-- Represents a probability experiment -/
inductive Experiment
| ballDraw
| busWait
| coinToss
| waterTest

/-- Checks if an experiment has a finite number of outcomes -/
def isFinite (e : Experiment) : Prop :=
  match e with
  | Experiment.ballDraw => true
  | Experiment.busWait => false
  | Experiment.coinToss => true
  | Experiment.waterTest => false

/-- Checks if an experiment has equally likely outcomes -/
def isEquallyLikely (e : Experiment) : Prop :=
  match e with
  | Experiment.ballDraw => true
  | Experiment.busWait => false
  | Experiment.coinToss => false
  | Experiment.waterTest => false

/-- Defines a classical probability model -/
def isClassicalProbabilityModel (e : Experiment) : Prop :=
  isFinite e ∧ isEquallyLikely e

/-- Theorem stating that only the ball draw experiment is a classical probability model -/
theorem only_ball_draw_is_classical : 
  ∀ e : Experiment, isClassicalProbabilityModel e ↔ e = Experiment.ballDraw :=
by sorry

end NUMINAMATH_CALUDE_only_ball_draw_is_classical_l1997_199710


namespace NUMINAMATH_CALUDE_words_per_page_l1997_199704

theorem words_per_page (total_pages : Nat) (total_words_mod : Nat) (modulus : Nat) 
  (h1 : total_pages = 154)
  (h2 : total_words_mod = 145)
  (h3 : modulus = 221)
  (h4 : ∃ (words_per_page : Nat), words_per_page ≤ 120 ∧ 
        (total_pages * words_per_page) % modulus = total_words_mod) :
  ∃ (words_per_page : Nat), words_per_page = 96 ∧
    (total_pages * words_per_page) % modulus = total_words_mod := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_l1997_199704


namespace NUMINAMATH_CALUDE_subcommittees_with_coach_count_l1997_199721

def total_members : ℕ := 12
def coach_members : ℕ := 5
def subcommittee_size : ℕ := 5

def subcommittees_with_coach : ℕ := Nat.choose total_members subcommittee_size - Nat.choose (total_members - coach_members) subcommittee_size

theorem subcommittees_with_coach_count : subcommittees_with_coach = 771 := by
  sorry

end NUMINAMATH_CALUDE_subcommittees_with_coach_count_l1997_199721


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_equation_l1997_199759

theorem equal_roots_quadratic_equation :
  ∃! r : ℝ, ∀ x : ℝ, x^2 - r*x - r^2 = 0 → (∃! y : ℝ, y^2 - r*y - r^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_equation_l1997_199759


namespace NUMINAMATH_CALUDE_als_original_portion_l1997_199762

theorem als_original_portion
  (total_initial : ℝ)
  (total_final : ℝ)
  (h_total_initial : total_initial = 1200)
  (h_total_final : total_final = 1800)
  (a b c : ℝ)
  (h_initial_sum : a + b + c = total_initial)
  (h_final_sum : (a - 150) + (2 * b) + (3 * c) = total_final) :
  a = 550 := by
sorry

end NUMINAMATH_CALUDE_als_original_portion_l1997_199762


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l1997_199764

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 40 → area = (perimeter / 4)^2 → area = 100 := by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l1997_199764


namespace NUMINAMATH_CALUDE_R3_sequence_arithmetic_l1997_199744

def is_R3_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 3 → a (n - 3) + a (n + 3) = 2 * a n) ∧
  (∀ n : ℕ, a (n + 1) ≥ a n)

def is_arithmetic_subsequence (b : ℕ → ℝ) (start : ℕ) (step : ℕ) (count : ℕ) : Prop :=
  ∃ d : ℝ, ∀ i : ℕ, i < count → b (start + i * step) - b (start + (i + 1) * step) = d

theorem R3_sequence_arithmetic (a : ℕ → ℝ) (h1 : is_R3_sequence a) 
  (h2 : ∃ p : ℕ, p > 1 ∧ is_arithmetic_subsequence a (3 * p - 3) 2 4) :
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d := by
  sorry

end NUMINAMATH_CALUDE_R3_sequence_arithmetic_l1997_199744


namespace NUMINAMATH_CALUDE_cherry_strawberry_cost_ratio_l1997_199787

/-- The cost of a pound of strawberries in dollars -/
def strawberry_cost : ℚ := 2.20

/-- The cost of 5 pounds of strawberries and 5 pounds of cherries in dollars -/
def total_cost : ℚ := 77

/-- The ratio of the cost of cherries to strawberries -/
def cherry_strawberry_ratio : ℚ := 6

theorem cherry_strawberry_cost_ratio :
  ∃ (cherry_cost : ℚ),
    cherry_cost > 0 ∧
    5 * strawberry_cost + 5 * cherry_cost = total_cost ∧
    cherry_cost / strawberry_cost = cherry_strawberry_ratio :=
by sorry

end NUMINAMATH_CALUDE_cherry_strawberry_cost_ratio_l1997_199787


namespace NUMINAMATH_CALUDE_monkeys_for_48_bananas_l1997_199738

/-- Given that 8 monkeys can eat 8 bananas in some time, 
    this function calculates the number of monkeys needed to eat 48 bananas in 48 minutes -/
def monkeys_needed (initial_monkeys : ℕ) (initial_bananas : ℕ) (target_bananas : ℕ) : ℕ :=
  initial_monkeys * (target_bananas / initial_bananas)

/-- Theorem stating that 48 monkeys are needed to eat 48 bananas in 48 minutes -/
theorem monkeys_for_48_bananas : monkeys_needed 8 8 48 = 48 := by
  sorry

end NUMINAMATH_CALUDE_monkeys_for_48_bananas_l1997_199738


namespace NUMINAMATH_CALUDE_calculate_divisor_l1997_199757

/-- Given a dividend, quotient, and remainder, calculate the divisor -/
theorem calculate_divisor (dividend : ℝ) (quotient : ℝ) (remainder : ℝ) :
  dividend = 63584 ∧ quotient = 127.8 ∧ remainder = 45.5 →
  ∃ divisor : ℝ, divisor = 497.1 ∧ dividend = divisor * quotient + remainder :=
by sorry

end NUMINAMATH_CALUDE_calculate_divisor_l1997_199757


namespace NUMINAMATH_CALUDE_total_eggs_calculation_l1997_199713

theorem total_eggs_calculation (eggs_per_omelet : ℕ) (num_people : ℕ) (omelets_per_person : ℕ)
  (h1 : eggs_per_omelet = 4)
  (h2 : num_people = 3)
  (h3 : omelets_per_person = 3) :
  eggs_per_omelet * num_people * omelets_per_person = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_calculation_l1997_199713
