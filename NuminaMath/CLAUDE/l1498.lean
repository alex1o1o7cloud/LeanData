import Mathlib

namespace NUMINAMATH_CALUDE_runner_picture_probability_l1498_149887

/-- Rachel's lap time in seconds -/
def rachel_lap_time : ℕ := 100

/-- Robert's lap time in seconds -/
def robert_lap_time : ℕ := 70

/-- Duration of the observation period in seconds -/
def observation_period : ℕ := 60

/-- Fraction of the track captured in the picture -/
def picture_fraction : ℚ := 1/5

/-- Time when the picture is taken (in seconds after start) -/
def picture_time : ℕ := 720  -- 12 minutes

theorem runner_picture_probability :
  let rachel_position := picture_time % rachel_lap_time
  let robert_position := robert_lap_time - (picture_time % robert_lap_time)
  let rachel_in_picture := rachel_position ≤ (rachel_lap_time * picture_fraction / 2) ∨
                           rachel_position ≥ rachel_lap_time - (rachel_lap_time * picture_fraction / 2)
  let robert_in_picture := robert_position ≤ (robert_lap_time * picture_fraction / 2) ∨
                           robert_position ≥ robert_lap_time - (robert_lap_time * picture_fraction / 2)
  (∃ t : ℕ, t ≥ picture_time ∧ t < picture_time + observation_period ∧
            rachel_in_picture ∧ robert_in_picture) →
  (1 : ℚ) / 16 = ↑(Nat.card {t : ℕ | t ≥ picture_time ∧ t < picture_time + observation_period ∧
                              rachel_in_picture ∧ robert_in_picture}) / observation_period :=
by sorry

end NUMINAMATH_CALUDE_runner_picture_probability_l1498_149887


namespace NUMINAMATH_CALUDE_no_definitive_conclusion_l1498_149863

-- Define the sets
variable (Beta Zeta Yota : Set α)

-- Define the hypotheses
variable (h1 : ∃ x, x ∈ Beta ∧ x ∉ Zeta)
variable (h2 : Zeta ⊆ Yota)

-- Define the statements that cannot be conclusively proven
def statement_A := ∃ x, x ∈ Beta ∧ x ∉ Yota
def statement_B := Beta ⊆ Yota
def statement_C := Beta ∩ Yota = ∅
def statement_D := ∃ x, x ∈ Beta ∧ x ∈ Yota

-- Theorem stating that none of the statements can be definitively concluded
theorem no_definitive_conclusion :
  ¬(statement_A Beta Yota ∨ statement_B Beta Yota ∨ statement_C Beta Yota ∨ statement_D Beta Yota) :=
sorry

end NUMINAMATH_CALUDE_no_definitive_conclusion_l1498_149863


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1498_149832

def num_coins : ℕ := 6

def all_outcomes : ℕ := 2^num_coins

def favorable_outcomes : ℕ := 2 + 2 * (num_coins.choose 1)

theorem coin_flip_probability : 
  (favorable_outcomes : ℚ) / all_outcomes = 7 / 32 :=
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1498_149832


namespace NUMINAMATH_CALUDE_condo_units_count_l1498_149816

/-- Represents a condo development with regular and penthouse floors. -/
structure Condo where
  total_floors : Nat
  penthouse_floors : Nat
  regular_units : Nat
  penthouse_units : Nat

/-- Calculates the total number of units in a condo. -/
def total_units (c : Condo) : Nat :=
  (c.total_floors - c.penthouse_floors) * c.regular_units + c.penthouse_floors * c.penthouse_units

/-- Theorem stating that a condo with the given specifications has 256 units. -/
theorem condo_units_count : 
  let c : Condo := {
    total_floors := 23,
    penthouse_floors := 2,
    regular_units := 12,
    penthouse_units := 2
  }
  total_units c = 256 := by
  sorry

#check condo_units_count

end NUMINAMATH_CALUDE_condo_units_count_l1498_149816


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_l1498_149897

theorem triangle_isosceles_or_right 
  (a b c : ℝ) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_lengths_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h : a^2 * c^2 - b^2 * c^2 = a^4 - b^4) : 
  (a = b) ∨ (a^2 + b^2 = c^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_or_right_l1498_149897


namespace NUMINAMATH_CALUDE_prob_same_color_24_sided_die_l1498_149831

/-- Represents a 24-sided die with colored sides -/
structure ColoredDie :=
  (purple : Nat)
  (blue : Nat)
  (red : Nat)
  (gold : Nat)
  (total : Nat)
  (h_total : purple + blue + red + gold = total)

/-- The probability of rolling a specific color on a single die -/
def prob_color (d : ColoredDie) (color : Nat) : Rat :=
  color / d.total

/-- The probability of rolling the same color on two identical dice -/
def prob_same_color (d : ColoredDie) : Rat :=
  (prob_color d d.purple)^2 + (prob_color d d.blue)^2 +
  (prob_color d d.red)^2 + (prob_color d d.gold)^2

/-- The specific 24-sided die configuration from the problem -/
def problem_die : ColoredDie :=
  { purple := 5
    blue := 8
    red := 10
    gold := 1
    total := 24
    h_total := by simp }

theorem prob_same_color_24_sided_die :
  prob_same_color problem_die = 95 / 288 := by
  sorry


end NUMINAMATH_CALUDE_prob_same_color_24_sided_die_l1498_149831


namespace NUMINAMATH_CALUDE_length_of_AB_l1498_149873

/-- Given a line segment AB with points P and Q, prove that AB has length 70 -/
theorem length_of_AB (A B P Q : ℝ) : 
  (0 < A ∧ A < P ∧ P < Q ∧ Q < B) →  -- P and Q are in AB and on the same side of midpoint
  (P - A) / (B - P) = 2 / 3 →        -- P divides AB in ratio 2:3
  (Q - A) / (B - Q) = 3 / 4 →        -- Q divides AB in ratio 3:4
  Q - P = 2 →                        -- PQ = 2
  B - A = 70 := by                   -- AB has length 70
sorry


end NUMINAMATH_CALUDE_length_of_AB_l1498_149873


namespace NUMINAMATH_CALUDE_complementary_angles_adjustment_l1498_149820

-- Define the ratio of the two complementary angles
def angle_ratio : ℚ := 3 / 7

-- Define the increase percentage for the smaller angle
def small_angle_increase : ℚ := 1 / 5

-- Function to calculate the decrease percentage for the larger angle
def large_angle_decrease (ratio : ℚ) (increase : ℚ) : ℚ :=
  1 - (90 - 90 * ratio / (1 + ratio) * (1 + increase)) / (90 * ratio / (1 + ratio))

-- Theorem statement
theorem complementary_angles_adjustment :
  large_angle_decrease angle_ratio small_angle_increase = 43 / 500 := by
  sorry

#eval large_angle_decrease angle_ratio small_angle_increase

end NUMINAMATH_CALUDE_complementary_angles_adjustment_l1498_149820


namespace NUMINAMATH_CALUDE_bart_notepad_spending_l1498_149808

/-- The amount of money Bart spent on notepads -/
def money_spent (cost_per_notepad : ℚ) (pages_per_notepad : ℕ) (total_pages : ℕ) : ℚ :=
  (total_pages / pages_per_notepad) * cost_per_notepad

/-- Theorem: Given the conditions, Bart spent $10 on notepads -/
theorem bart_notepad_spending :
  let cost_per_notepad : ℚ := 5/4  -- $1.25 represented as a rational number
  let pages_per_notepad : ℕ := 60
  let total_pages : ℕ := 480
  money_spent cost_per_notepad pages_per_notepad total_pages = 10 := by
  sorry


end NUMINAMATH_CALUDE_bart_notepad_spending_l1498_149808


namespace NUMINAMATH_CALUDE_team_average_goals_is_seven_l1498_149890

/-- The average number of goals scored by a soccer team per game -/
def team_average_goals (carter_goals shelby_goals judah_goals : ℝ) : ℝ :=
  carter_goals + shelby_goals + judah_goals

/-- Theorem: Given the conditions, the team's average goals per game is 7 -/
theorem team_average_goals_is_seven :
  ∀ (carter_goals shelby_goals judah_goals : ℝ),
    carter_goals = 4 →
    shelby_goals = carter_goals / 2 →
    judah_goals = 2 * shelby_goals - 3 →
    team_average_goals carter_goals shelby_goals judah_goals = 7 := by
  sorry

end NUMINAMATH_CALUDE_team_average_goals_is_seven_l1498_149890


namespace NUMINAMATH_CALUDE_soda_cost_calculation_l1498_149814

def restaurant_bill (num_adults num_children : ℕ) (adult_meal_cost child_meal_cost soda_cost : ℚ) : Prop :=
  let total_people := num_adults + num_children
  let meal_cost := (num_adults * adult_meal_cost) + (num_children * child_meal_cost)
  let total_bill := meal_cost + (total_people * soda_cost)
  (num_adults = 6) ∧ 
  (num_children = 2) ∧ 
  (adult_meal_cost = 6) ∧ 
  (child_meal_cost = 4) ∧ 
  (total_bill = 60)

theorem soda_cost_calculation :
  ∃ (soda_cost : ℚ), restaurant_bill 6 2 6 4 soda_cost ∧ soda_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_calculation_l1498_149814


namespace NUMINAMATH_CALUDE_square_of_polynomial_l1498_149821

theorem square_of_polynomial (x a : ℝ) : 
  (x+a)*(x+2*a)*(x+3*a)*(x+4*a) + a^4 = (x^2 + 5*a*x + 5*a^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_polynomial_l1498_149821


namespace NUMINAMATH_CALUDE_janet_earnings_l1498_149869

/-- Calculates the total earnings of Janet based on her exterminator work and sculpture sales. -/
theorem janet_earnings (
  hourly_rate : ℝ)
  (sculpture_price_per_pound : ℝ)
  (hours_worked : ℝ)
  (sculpture1_weight : ℝ)
  (sculpture2_weight : ℝ)
  (h1 : hourly_rate = 70)
  (h2 : sculpture_price_per_pound = 20)
  (h3 : hours_worked = 20)
  (h4 : sculpture1_weight = 5)
  (h5 : sculpture2_weight = 7) :
  hourly_rate * hours_worked + sculpture_price_per_pound * (sculpture1_weight + sculpture2_weight) = 1640 :=
by
  sorry


end NUMINAMATH_CALUDE_janet_earnings_l1498_149869


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l1498_149801

/-- A geometric sequence with positive common ratio -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  a 4 * a 8 = 2 * (a 5)^2 →
  a 3 = 1 →
  a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l1498_149801


namespace NUMINAMATH_CALUDE_square_area_ratio_l1498_149898

/-- If the perimeter of one square is 4 times the perimeter of another square,
    then the area of the larger square is 16 times the area of the smaller square. -/
theorem square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_perimeter : 4 * a = 4 * (4 * b)) : a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1498_149898


namespace NUMINAMATH_CALUDE_smallest_dividend_l1498_149838

theorem smallest_dividend (q r : ℕ) (h1 : q = 12) (h2 : r = 3) :
  ∃ (a b : ℕ), a = b * q + r ∧ b > r ∧ ∀ (a' b' : ℕ), (a' = b' * q + r ∧ b' > r) → a ≤ a' :=
by sorry

end NUMINAMATH_CALUDE_smallest_dividend_l1498_149838


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1498_149842

theorem fraction_subtraction_simplification :
  (7 : ℚ) / 17 - (4 : ℚ) / 51 = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1498_149842


namespace NUMINAMATH_CALUDE_local_max_implies_neg_local_min_l1498_149886

open Function Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define x₀ as a non-zero real number
variable (x₀ : ℝ)
variable (hx₀ : x₀ ≠ 0)

-- Define that x₀ is a local maximum point of f
def IsLocalMaxAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x ≤ f x₀

-- Define local minimum
def IsLocalMinAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x₀ ≤ f x

-- State the theorem
theorem local_max_implies_neg_local_min
  (h : IsLocalMaxAt f x₀) :
  IsLocalMinAt (fun x ↦ -f (-x)) (-x₀) :=
sorry

end NUMINAMATH_CALUDE_local_max_implies_neg_local_min_l1498_149886


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l1498_149881

theorem sphere_volume_ratio (R : ℝ) (h : R > 0) :
  (4 / 3 * Real.pi * (2 * R)^3) / (4 / 3 * Real.pi * R^3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l1498_149881


namespace NUMINAMATH_CALUDE_remainder_102_104_plus_6_div_9_l1498_149811

theorem remainder_102_104_plus_6_div_9 : (102 * 104 + 6) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_102_104_plus_6_div_9_l1498_149811


namespace NUMINAMATH_CALUDE_a_in_A_l1498_149878

def A : Set ℝ := {x | x ≥ 2 * Real.sqrt 2}

theorem a_in_A : 3 ∈ A := by
  sorry

end NUMINAMATH_CALUDE_a_in_A_l1498_149878


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l1498_149896

theorem imaginary_power_sum : ∃ i : ℂ, i^2 = -1 ∧ i^50 + i^250 = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l1498_149896


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1498_149819

theorem greatest_divisor_four_consecutive_integers (n : ℕ+) :
  ∃ (k : ℕ), k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
  ∀ (m : ℕ), m ∣ (n * (n + 1) * (n + 2) * (n + 3)) → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1498_149819


namespace NUMINAMATH_CALUDE_team_a_games_l1498_149879

theorem team_a_games (a : ℕ) : 
  (2 : ℚ) / 3 * a = (5 : ℚ) / 8 * (a + 14) - 7 → a = 42 := by
  sorry

end NUMINAMATH_CALUDE_team_a_games_l1498_149879


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1498_149847

theorem fraction_to_decimal : (59 : ℚ) / 160 = (36875 : ℚ) / 100000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1498_149847


namespace NUMINAMATH_CALUDE_prob_a_b_not_same_class_l1498_149888

/-- The number of students to be distributed -/
def num_students : ℕ := 4

/-- The number of classes -/
def num_classes : ℕ := 3

/-- The probability that students A and B are not in the same class -/
def prob_not_same_class : ℚ := 5/6

/-- The total number of ways to distribute students into classes -/
def total_distributions : ℕ := num_students.choose 2 * num_classes.factorial

/-- The number of distributions where A and B are in different classes -/
def favorable_distributions : ℕ := total_distributions - num_classes.factorial

theorem prob_a_b_not_same_class :
  (favorable_distributions : ℚ) / total_distributions = prob_not_same_class :=
sorry

end NUMINAMATH_CALUDE_prob_a_b_not_same_class_l1498_149888


namespace NUMINAMATH_CALUDE_race_finish_positions_l1498_149860

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  position : ℝ

/-- Represents the state of the race -/
structure RaceState where
  a : Runner
  b : Runner
  c : Runner

/-- The race is 100 meters long -/
def race_length : ℝ := 100

theorem race_finish_positions (initial : RaceState) 
  (h1 : initial.a.position = race_length) 
  (h2 : initial.b.position = race_length - 5)
  (h3 : initial.c.position = race_length - 10)
  (h4 : ∀ r : Runner, r.speed > 0) :
  ∃ (final : RaceState), 
    final.b.position = race_length ∧ 
    final.c.position = race_length - (5 * 5 / 19) := by
  sorry

end NUMINAMATH_CALUDE_race_finish_positions_l1498_149860


namespace NUMINAMATH_CALUDE_parabola_properties_l1498_149827

def parabola (x : ℝ) : ℝ := x^2 - 4*x - 4

theorem parabola_properties :
  (∀ x : ℝ, parabola x ≥ parabola 2) ∧
  (∀ x : ℝ, parabola x = parabola (4 - x)) ∧
  (parabola 2 = -8) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1498_149827


namespace NUMINAMATH_CALUDE_smallest_x_value_l1498_149882

theorem smallest_x_value : 
  ∃ (x : ℝ), x > 1 ∧ 
  ((5*x - 20) / (4*x - 5))^2 + (5*x - 20) / (4*x - 5) = 20 ∧
  (∀ (y : ℝ), y > 1 ∧ 
   ((5*y - 20) / (4*y - 5))^2 + (5*y - 20) / (4*y - 5) = 20 → 
   x ≤ y) ∧
  x = 9/5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1498_149882


namespace NUMINAMATH_CALUDE_min_value_problem_l1498_149805

theorem min_value_problem (x y : ℝ) :
  (abs y ≤ 1) →
  (2 * x + y = 1) →
  (∀ x' y' : ℝ, abs y' ≤ 1 → 2 * x' + y' = 1 → 2 * x'^2 + 16 * x' + 3 * y'^2 ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1498_149805


namespace NUMINAMATH_CALUDE_one_half_of_one_third_of_one_sixth_of_180_l1498_149802

theorem one_half_of_one_third_of_one_sixth_of_180 : 
  (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 180 = 5 := by
  sorry

end NUMINAMATH_CALUDE_one_half_of_one_third_of_one_sixth_of_180_l1498_149802


namespace NUMINAMATH_CALUDE_a_perp_b_l1498_149825

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def are_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The given vectors -/
def a : ℝ × ℝ := (-5, 6)
def b : ℝ × ℝ := (6, 5)

/-- Theorem: Vectors a and b are perpendicular -/
theorem a_perp_b : are_perpendicular a b := by sorry

end NUMINAMATH_CALUDE_a_perp_b_l1498_149825


namespace NUMINAMATH_CALUDE_greatest_4digit_base7_divisible_by_7_l1498_149834

/-- Converts a base 7 number to decimal --/
def toDecimal (n : List Nat) : Nat :=
  n.reverse.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Checks if a number is divisible by 7 --/
def isDivisibleBy7 (n : Nat) : Bool :=
  n % 7 = 0

/-- Converts a decimal number to base 7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Checks if a list represents a 4-digit base 7 number --/
def is4DigitBase7 (n : List Nat) : Bool :=
  n.length = 4 && n.all (· < 7) && n.head! ≠ 0

theorem greatest_4digit_base7_divisible_by_7 :
  let n := [6, 6, 6, 0]
  is4DigitBase7 n ∧
  isDivisibleBy7 (toDecimal n) ∧
  ∀ m, is4DigitBase7 m → isDivisibleBy7 (toDecimal m) → toDecimal m ≤ toDecimal n :=
by sorry

end NUMINAMATH_CALUDE_greatest_4digit_base7_divisible_by_7_l1498_149834


namespace NUMINAMATH_CALUDE_cost_per_book_l1498_149853

theorem cost_per_book (total_books : ℕ) (total_spent : ℕ) (h1 : total_books = 14) (h2 : total_spent = 224) :
  total_spent / total_books = 16 := by
sorry

end NUMINAMATH_CALUDE_cost_per_book_l1498_149853


namespace NUMINAMATH_CALUDE_major_premise_identification_l1498_149807

theorem major_premise_identification (α : ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, |Real.sin x| ≤ 1)
  (h2 : m = Real.sin α)
  (h3 : |m| ≤ 1) :
  (∀ x : ℝ, |Real.sin x| ≤ 1) = (|Real.sin x| ≤ 1) := by
sorry

end NUMINAMATH_CALUDE_major_premise_identification_l1498_149807


namespace NUMINAMATH_CALUDE_sum_of_possible_x_values_l1498_149855

/-- An isosceles triangle with two angles of 50° and x° --/
structure IsoscelesTriangle where
  x : ℝ
  is_isosceles : Bool
  has_50_degree_angle : Bool
  has_x_degree_angle : Bool

/-- The sum of angles in a triangle is 180° --/
axiom angle_sum (t : IsoscelesTriangle) : t.x + 50 + (180 - t.x - 50) = 180

/-- In an isosceles triangle, at least two angles are equal --/
axiom isosceles_equal_angles (t : IsoscelesTriangle) : t.is_isosceles → 
  (t.x = 50 ∨ t.x = (180 - 50) / 2 ∨ t.x = 180 - 2 * 50)

/-- The theorem to be proved --/
theorem sum_of_possible_x_values : 
  ∀ t : IsoscelesTriangle, t.is_isosceles ∧ t.has_50_degree_angle ∧ t.has_x_degree_angle → 
    50 + (180 - 50) / 2 + (180 - 2 * 50) = 195 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_values_l1498_149855


namespace NUMINAMATH_CALUDE_correct_delivery_probability_l1498_149854

/-- The number of houses and packages -/
def n : ℕ := 5

/-- The number of correctly delivered packages -/
def k : ℕ := 3

/-- Probability of exactly k out of n packages being delivered to the correct houses -/
def prob_correct_delivery (n k : ℕ) : ℚ :=
  (Nat.choose n k * (Nat.factorial k) * (Nat.factorial (n - k) / Nat.factorial n)) / Nat.factorial n

theorem correct_delivery_probability :
  prob_correct_delivery n k = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_correct_delivery_probability_l1498_149854


namespace NUMINAMATH_CALUDE_circumcircle_diameter_perpendicular_to_DK_l1498_149876

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0

-- Define the altitude CD
def altitude (A B C D : ℝ × ℝ) : Prop :=
  (D.1 - C.1) * (B.1 - A.1) + (D.2 - C.2) * (B.2 - A.2) = 0

-- Define point K such that |AK| = |AC|
def point_K (A C K : ℝ × ℝ) : Prop :=
  (K.1 - A.1)^2 + (K.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

-- Define the circumcircle of triangle ABK
def circumcircle (A B K O : ℝ × ℝ) : Prop :=
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (B.1 - O.1)^2 + (B.2 - O.2)^2 ∧
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (K.1 - O.1)^2 + (K.2 - O.2)^2

-- Define perpendicularity
def perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (S.1 - R.1) + (Q.2 - P.2) * (S.2 - R.2) = 0

-- Theorem statement
theorem circumcircle_diameter_perpendicular_to_DK 
  (A B C D K O : ℝ × ℝ) 
  (h1 : Triangle A B C)
  (h2 : altitude A B C D)
  (h3 : point_K A C K)
  (h4 : circumcircle A B K O) :
  perpendicular A O D K :=
sorry

end NUMINAMATH_CALUDE_circumcircle_diameter_perpendicular_to_DK_l1498_149876


namespace NUMINAMATH_CALUDE_factorial_difference_l1498_149846

theorem factorial_difference : Nat.factorial 9 - Nat.factorial 8 = 322560 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l1498_149846


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1498_149889

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 25) →
  (a 2 + a 8 = 10) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1498_149889


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1498_149806

/-- 
Given a line y = mx + b, if the reflection of point (-4, 0) across this line 
is (2, 6), then m + b = 1.
-/
theorem reflection_line_sum (m b : ℝ) : 
  (∀ (x y : ℝ), y = m * x + b → 
    (x = -1 ∧ y = 3) ↔ 
    (x = ((-4 + 2) / 2) ∧ y = ((0 + 6) / 2))) →
  (m = -1) →
  (m + b = 1) := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l1498_149806


namespace NUMINAMATH_CALUDE_rectangle_diagonal_intersection_l1498_149810

/-- Given a rectangle with opposite vertices at (2, -3) and (14, 9),
    prove that its diagonals intersect at the point (8, 3). -/
theorem rectangle_diagonal_intersection :
  let a : ℝ × ℝ := (2, -3)
  let c : ℝ × ℝ := (14, 9)
  let midpoint : ℝ × ℝ := ((a.1 + c.1) / 2, (a.2 + c.2) / 2)
  midpoint = (8, 3) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_intersection_l1498_149810


namespace NUMINAMATH_CALUDE_women_who_left_l1498_149815

theorem women_who_left (initial_men : ℕ) (initial_women : ℕ) (final_men : ℕ) (final_women : ℕ) :
  initial_men * 5 = initial_women * 4 →
  final_men = initial_men + 2 →
  final_men = 14 →
  final_women = 24 →
  final_women = 2 * (initial_women - (initial_women - final_women / 2)) →
  initial_women - final_women / 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_women_who_left_l1498_149815


namespace NUMINAMATH_CALUDE_intersection_union_eq_l1498_149872

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {1, 3, 4}
def C : Set ℝ := {x : ℝ | x^2 - 3*x + 2 > 0}

theorem intersection_union_eq : (A ∪ B) ∩ C = {0, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_union_eq_l1498_149872


namespace NUMINAMATH_CALUDE_percent_of_percent_l1498_149862

theorem percent_of_percent (x : ℝ) (h : x ≠ 0) : (0.3 * 0.7 * x) / x = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l1498_149862


namespace NUMINAMATH_CALUDE_bookstore_inventory_calculation_l1498_149804

/-- Represents the inventory of a bookstore -/
structure BookInventory where
  fiction : ℕ
  nonFiction : ℕ
  children : ℕ

/-- Represents the sales figures for a day -/
structure DailySales where
  inStoreFiction : ℕ
  inStoreNonFiction : ℕ
  inStoreChildren : ℕ
  online : ℕ

/-- Calculate the total number of books in the inventory -/
def totalBooks (inventory : BookInventory) : ℕ :=
  inventory.fiction + inventory.nonFiction + inventory.children

/-- Calculate the total in-store sales -/
def totalInStoreSales (sales : DailySales) : ℕ :=
  sales.inStoreFiction + sales.inStoreNonFiction + sales.inStoreChildren

theorem bookstore_inventory_calculation 
  (initialInventory : BookInventory)
  (saturdaySales : DailySales)
  (sundayInStoreSalesMultiplier : ℕ)
  (sundayOnlineSalesIncrease : ℕ)
  (newShipment : ℕ)
  (h1 : totalBooks initialInventory = 743)
  (h2 : initialInventory.fiction = 520)
  (h3 : initialInventory.nonFiction = 123)
  (h4 : initialInventory.children = 100)
  (h5 : totalInStoreSales saturdaySales = 37)
  (h6 : saturdaySales.inStoreFiction = 15)
  (h7 : saturdaySales.inStoreNonFiction = 12)
  (h8 : saturdaySales.inStoreChildren = 10)
  (h9 : saturdaySales.online = 128)
  (h10 : sundayInStoreSalesMultiplier = 2)
  (h11 : sundayOnlineSalesIncrease = 34)
  (h12 : newShipment = 160)
  : totalBooks initialInventory - 
    (totalInStoreSales saturdaySales + saturdaySales.online) - 
    (sundayInStoreSalesMultiplier * totalInStoreSales saturdaySales + 
     saturdaySales.online + sundayOnlineSalesIncrease) + 
    newShipment = 502 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_inventory_calculation_l1498_149804


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_cube_positive_l1498_149868

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) :=
by sorry

theorem negation_of_cube_positive :
  (¬ ∀ x > 0, x^3 > 0) ↔ (∃ x > 0, x^3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_cube_positive_l1498_149868


namespace NUMINAMATH_CALUDE_tourist_contact_probability_l1498_149800

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
def contact_probability (p : ℝ) : ℝ := 1 - (1 - p)^42

/-- Theorem stating the probability of contact between two groups of tourists -/
theorem tourist_contact_probability 
  (p : ℝ) 
  (h1 : 0 ≤ p) 
  (h2 : p ≤ 1) 
  (group1 : Fin 6 → Type) 
  (group2 : Fin 7 → Type) :
  contact_probability p = 1 - (1 - p)^42 := by
sorry

end NUMINAMATH_CALUDE_tourist_contact_probability_l1498_149800


namespace NUMINAMATH_CALUDE_division_remainder_l1498_149877

theorem division_remainder : ∃ q : ℕ, 1234567 = 321 * q + 264 ∧ 264 < 321 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1498_149877


namespace NUMINAMATH_CALUDE_trig_inequality_l1498_149828

theorem trig_inequality (α β : Real) (h_α : 0 < α ∧ α < π / 2) (h_β : 0 < β ∧ β < π / 2) :
  Real.sin α ^ 3 * Real.cos β ^ 3 + Real.sin α ^ 3 * Real.sin β ^ 3 + Real.cos α ^ 3 ≥ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l1498_149828


namespace NUMINAMATH_CALUDE_sugar_measurement_l1498_149866

theorem sugar_measurement (sugar_needed : ℚ) (cup_capacity : ℚ) : 
  sugar_needed = 5/2 ∧ cup_capacity = 1/4 → sugar_needed / cup_capacity = 10 := by
  sorry

end NUMINAMATH_CALUDE_sugar_measurement_l1498_149866


namespace NUMINAMATH_CALUDE_triangle_angles_l1498_149861

-- Define a triangle XYZ
structure Triangle :=
  (X Y Z : Point)

-- Define the angles in the triangle
def angle_YXZ (t : Triangle) : ℝ := sorry
def angle_XYZ (t : Triangle) : ℝ := sorry
def angle_XZY (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_angles (t : Triangle) :
  angle_YXZ t = 40 ∧ angle_XYZ t = 80 → angle_XZY t = 60 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_l1498_149861


namespace NUMINAMATH_CALUDE_linear_equation_exponent_l1498_149858

/-- If x^(m+1) - 2 = 1 is a linear equation with respect to x, then m = 0 -/
theorem linear_equation_exponent (m : ℕ) : 
  (∀ x, ∃ a b : ℝ, x^(m+1) - 2 = a*x + b) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_l1498_149858


namespace NUMINAMATH_CALUDE_tiling_uniqueness_l1498_149830

/-- A rectangular grid --/
structure RectangularGrid where
  rows : ℕ
  cols : ℕ

/-- A cell in the grid --/
structure Cell where
  row : ℕ
  col : ℕ

/-- A tiling of the grid --/
def Tiling (grid : RectangularGrid) := Set (Set Cell)

/-- The set of central cells for a given tiling --/
def CentralCells (grid : RectangularGrid) (tiling : Tiling grid) : Set Cell :=
  sorry

/-- Theorem: The set of central cells uniquely determines the tiling --/
theorem tiling_uniqueness (grid : RectangularGrid) 
  (tiling1 tiling2 : Tiling grid) :
  CentralCells grid tiling1 = CentralCells grid tiling2 → tiling1 = tiling2 :=
sorry

end NUMINAMATH_CALUDE_tiling_uniqueness_l1498_149830


namespace NUMINAMATH_CALUDE_area_ratio_of_rectangles_l1498_149874

/-- Given two rectangles A and B with specified dimensions, prove that the ratio of their areas is 12/21 -/
theorem area_ratio_of_rectangles (length_A width_A length_B width_B : ℕ) 
  (h1 : length_A = 36) (h2 : width_A = 20) (h3 : length_B = 42) (h4 : width_B = 30) :
  (length_A * width_A : ℚ) / (length_B * width_B) = 12 / 21 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_rectangles_l1498_149874


namespace NUMINAMATH_CALUDE_chinese_team_gold_medal_probability_l1498_149870

theorem chinese_team_gold_medal_probability 
  (prob_A prob_B : ℚ)
  (h1 : prob_A = 3 / 7)
  (h2 : prob_B = 1 / 4)
  (h3 : ∀ x y : ℚ, x + y = prob_A + prob_B → x ≤ prob_A ∧ y ≤ prob_B) :
  prob_A + prob_B = 19 / 28 := by
sorry

end NUMINAMATH_CALUDE_chinese_team_gold_medal_probability_l1498_149870


namespace NUMINAMATH_CALUDE_sum_f_equals_1326_l1498_149865

/-- A lattice point is a point with integer coordinates -/
def is_lattice_point (p : ℤ × ℤ) : Prop := True

/-- f(n) is the number of lattice points on the segment from (0,0) to (n, n+3), excluding endpoints -/
def f (n : ℕ) : ℕ := 
  if n % 3 = 0 then 2 else 0

/-- The sum of f(n) from 1 to 1990 -/
def sum_f : ℕ := (Finset.range 1990).sum f

theorem sum_f_equals_1326 : sum_f = 1326 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_equals_1326_l1498_149865


namespace NUMINAMATH_CALUDE_difference_of_squares_l1498_149892

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1498_149892


namespace NUMINAMATH_CALUDE_bicycle_exchange_point_exists_l1498_149894

/-- Represents the problem of finding the optimal bicycle exchange point --/
theorem bicycle_exchange_point_exists :
  ∃ x : ℝ, 0 < x ∧ x < 20 ∧
  (x / 10 + (20 - x) / 4 = (20 - x) / 8 + x / 5) := by
  sorry

#check bicycle_exchange_point_exists

end NUMINAMATH_CALUDE_bicycle_exchange_point_exists_l1498_149894


namespace NUMINAMATH_CALUDE_factorization_equality_l1498_149864

theorem factorization_equality (x y : ℝ) : 
  (x + y)^2 + 4*(x - y)^2 - 4*(x^2 - y^2) = (x - 3*y)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l1498_149864


namespace NUMINAMATH_CALUDE_largest_multiple_under_500_l1498_149835

theorem largest_multiple_under_500 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_under_500_l1498_149835


namespace NUMINAMATH_CALUDE_complementary_events_l1498_149845

-- Define the sample space for throwing 3 coins
def CoinOutcome := Fin 2 × Fin 2 × Fin 2

-- Define the event "No more than one head"
def NoMoreThanOneHead (outcome : CoinOutcome) : Prop :=
  (outcome.1 + outcome.2.1 + outcome.2.2 : ℕ) ≤ 1

-- Define the event "At least two heads"
def AtLeastTwoHeads (outcome : CoinOutcome) : Prop :=
  (outcome.1 + outcome.2.1 + outcome.2.2 : ℕ) ≥ 2

-- Theorem stating that the two events are complementary
theorem complementary_events :
  ∀ (outcome : CoinOutcome), NoMoreThanOneHead outcome ↔ ¬(AtLeastTwoHeads outcome) :=
by
  sorry


end NUMINAMATH_CALUDE_complementary_events_l1498_149845


namespace NUMINAMATH_CALUDE_limit_of_sequence_l1498_149871

def a (n : ℕ) : ℚ := (4 * n - 3) / (2 * n + 1)

theorem limit_of_sequence : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l1498_149871


namespace NUMINAMATH_CALUDE_find_x_value_l1498_149852

theorem find_x_value (x : ℝ) (hx : x ≠ 0) 
  (h : x = (1/x) * (-x) + 3) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l1498_149852


namespace NUMINAMATH_CALUDE_shift_down_three_units_l1498_149856

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := 2 * x

theorem shift_down_three_units (x : ℝ) : f x - 3 = g x := by
  sorry

end NUMINAMATH_CALUDE_shift_down_three_units_l1498_149856


namespace NUMINAMATH_CALUDE_direction_field_properties_l1498_149812

open Real

-- Define the differential equation
def y' (x y : ℝ) : ℝ := x^2 + y^2

-- Theorem statement
theorem direction_field_properties :
  -- 1. Slope at origin is 0
  y' 0 0 = 0 ∧
  -- 2. Slope at (1, 0) is 1
  y' 1 0 = 1 ∧
  -- 3. Slope is 1 for any point on the unit circle
  (∀ x y : ℝ, x^2 + y^2 = 1 → y' x y = 1) ∧
  -- 4. Slope increases as distance from origin increases
  (∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 < x2^2 + y2^2 → y' x1 y1 < y' x2 y2) :=
by sorry

end NUMINAMATH_CALUDE_direction_field_properties_l1498_149812


namespace NUMINAMATH_CALUDE_hat_number_problem_l1498_149893

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem hat_number_problem (alice_number bob_number : ℕ) : 
  alice_number ∈ Finset.range 50 →
  bob_number ∈ Finset.range 50 →
  alice_number ≠ bob_number →
  alice_number ≠ 1 →
  bob_number < alice_number →
  is_prime bob_number →
  bob_number < 10 →
  is_perfect_square (100 * bob_number + alice_number) →
  alice_number = 24 ∧ bob_number = 3 :=
by sorry

end NUMINAMATH_CALUDE_hat_number_problem_l1498_149893


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l1498_149826

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (2 / (x + 3))
  else if x < -3 then Int.floor (2 / (x + 3))
  else 0  -- This value doesn't matter as g is undefined at x = -3

theorem zero_not_in_range_of_g :
  ¬ ∃ (x : ℝ), g x = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l1498_149826


namespace NUMINAMATH_CALUDE_photocopy_cost_calculation_l1498_149848

/-- The cost of a single photocopy -/
def photocopy_cost : ℝ := 0.02

/-- The discount rate for large orders -/
def discount_rate : ℝ := 0.25

/-- The number of copies in a large order -/
def large_order_threshold : ℕ := 100

/-- The number of copies each person orders -/
def copies_per_person : ℕ := 80

/-- The savings per person when combining orders -/
def savings_per_person : ℝ := 0.40

theorem photocopy_cost_calculation :
  let total_copies := 2 * copies_per_person
  let undiscounted_total := total_copies * photocopy_cost
  let discounted_total := undiscounted_total * (1 - discount_rate)
  discounted_total = undiscounted_total - 2 * savings_per_person :=
by sorry

end NUMINAMATH_CALUDE_photocopy_cost_calculation_l1498_149848


namespace NUMINAMATH_CALUDE_solution_characterization_l1498_149884

/-- The set of solutions to the equation (n+1)^k = n! + 1 for natural numbers n and k -/
def SolutionSet : Set (ℕ × ℕ) :=
  {(1, 1), (2, 1), (4, 2)}

/-- The equation (n+1)^k = n! + 1 -/
def EquationHolds (n k : ℕ) : Prop :=
  (n + 1) ^ k = Nat.factorial n + 1

theorem solution_characterization :
  ∀ (n k : ℕ), EquationHolds n k ↔ (n, k) ∈ SolutionSet := by
  sorry

#check solution_characterization

end NUMINAMATH_CALUDE_solution_characterization_l1498_149884


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1498_149880

/-- Given two vectors a and b in ℝ³, where a is (1,1,2) and b is (2,x,y),
    and a is parallel to b, prove that the magnitude of b is 2√6. -/
theorem parallel_vectors_magnitude (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (1, 1, 2)
  let b : ℝ × ℝ × ℝ := (2, x, y)
  (∃ (k : ℝ), b.1 = k * a.1 ∧ b.2.1 = k * a.2.1 ∧ b.2.2 = k * a.2.2) →
  ‖(b.1, b.2.1, b.2.2)‖ = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1498_149880


namespace NUMINAMATH_CALUDE_negative_fraction_range_l1498_149859

theorem negative_fraction_range (x : ℝ) : (-5 : ℝ) / (2 - x) < 0 → x < 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_range_l1498_149859


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1498_149843

/-- A circle M with center (a, 2) and radius 2 -/
def circle_M (a x y : ℝ) : Prop := (x - a)^2 + (y - 2)^2 = 4

/-- A line l with equation x - y + 3 = 0 -/
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

/-- The chord intercepted by line l on circle M has a length of 4 -/
def chord_length_4 (a : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ,
  circle_M a x₁ y₁ ∧ circle_M a x₂ y₂ ∧
  line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16

theorem circle_line_intersection (a : ℝ) :
  circle_M a a 2 ∧ line_l a 2 ∧ chord_length_4 a → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1498_149843


namespace NUMINAMATH_CALUDE_root_set_equivalence_l1498_149875

/-- The function f(x) = x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

/-- The set of real roots of f(x) = 0 -/
def rootSet (a : ℝ) : Set ℝ := {x : ℝ | f a x = 0}

/-- The set of real roots of f(f(x)) = 0 -/
def composedRootSet (a : ℝ) : Set ℝ := {x : ℝ | f a (f a x) = 0}

/-- The theorem stating the equivalence between the condition and the range of a -/
theorem root_set_equivalence :
  ∀ a : ℝ, (rootSet a = composedRootSet a ∧ rootSet a ≠ ∅) ↔ 0 ≤ a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_root_set_equivalence_l1498_149875


namespace NUMINAMATH_CALUDE_solution_set_f_positive_solution_set_f_leq_g_l1498_149857

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + (2 - m) * x - m

/-- The function g(x) defined in the problem -/
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - x + 2 * m

/-- Theorem for part (1) of the problem -/
theorem solution_set_f_positive :
  {x : ℝ | f 1 x > 0} = {x : ℝ | x > 1/2 ∨ x < -1} := by sorry

/-- Theorem for part (2) of the problem -/
theorem solution_set_f_leq_g (m : ℝ) (h : m > 0) :
  {x : ℝ | f m x ≤ g m x} = {x : ℝ | -3 ≤ x ∧ x ≤ m} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_solution_set_f_leq_g_l1498_149857


namespace NUMINAMATH_CALUDE_bounded_fraction_exists_l1498_149837

theorem bounded_fraction_exists (C : ℝ) : ∃ C, ∀ k : ℤ, 
  |((k^8 - 2*k + 1) / (k^4 - 3))| < C :=
sorry

end NUMINAMATH_CALUDE_bounded_fraction_exists_l1498_149837


namespace NUMINAMATH_CALUDE_prove_age_difference_l1498_149883

def age_difference (freyja_age eli_age sarah_age kaylin_age : ℕ) : Prop :=
  freyja_age = 10 ∧
  eli_age = freyja_age + 9 ∧
  sarah_age = 2 * eli_age ∧
  kaylin_age = 33 ∧
  sarah_age - kaylin_age = 5

theorem prove_age_difference :
  ∃ (freyja_age eli_age sarah_age kaylin_age : ℕ),
    age_difference freyja_age eli_age sarah_age kaylin_age :=
by
  sorry

end NUMINAMATH_CALUDE_prove_age_difference_l1498_149883


namespace NUMINAMATH_CALUDE_point_plane_line_sphere_ratio_l1498_149841

/-- Given a point (a,b,c) on a plane and a line through the origin, 
    and (p,q,r) as the center of a sphere passing through specific points,
    prove that (a+b+c)/(p+q+r) = 1 -/
theorem point_plane_line_sphere_ratio 
  (a b c d e f p q r : ℝ) 
  (h1 : ∃ (t : ℝ), a = t * d ∧ b = t * e ∧ c = t * f)  -- (a,b,c) on line with direction (d,e,f)
  (h2 : ∃ (α β γ : ℝ), α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0 ∧        -- A, B, C distinct from O
        a/α + b/β + c/γ = 1)                          -- (a,b,c) on plane through A, B, C
  (h3 : p^2 + q^2 + r^2 = (p - α)^2 + q^2 + r^2)       -- O and A equidistant from (p,q,r)
  (h4 : p^2 + q^2 + r^2 = p^2 + (q - β)^2 + r^2)       -- O and B equidistant from (p,q,r)
  (h5 : p^2 + q^2 + r^2 = p^2 + q^2 + (r - γ)^2)       -- O and C equidistant from (p,q,r)
  (h6 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)                        -- Avoid division by zero
  : (a + b + c) / (p + q + r) = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_plane_line_sphere_ratio_l1498_149841


namespace NUMINAMATH_CALUDE_sweets_spending_proof_l1498_149803

/-- Calculates the amount spent on sweets given a weekly allowance, junk food spending ratio, and savings amount. -/
def amount_spent_on_sweets (allowance : ℚ) (junk_food_ratio : ℚ) (savings : ℚ) : ℚ :=
  allowance - allowance * junk_food_ratio - savings

/-- Proves that given a weekly allowance of $30, spending 1/3 on junk food, and saving $12, the amount spent on sweets is $8. -/
theorem sweets_spending_proof :
  amount_spent_on_sweets 30 (1/3) 12 = 8 := by
  sorry

#eval amount_spent_on_sweets 30 (1/3) 12

end NUMINAMATH_CALUDE_sweets_spending_proof_l1498_149803


namespace NUMINAMATH_CALUDE_problem_1_l1498_149829

theorem problem_1 : (Real.sqrt 5 + Real.sqrt 3) * (Real.sqrt 5 - Real.sqrt 3) + (Real.sqrt 3 - 2)^2 = 9 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1498_149829


namespace NUMINAMATH_CALUDE_no_four_consecutive_integers_product_perfect_square_l1498_149839

theorem no_four_consecutive_integers_product_perfect_square :
  ∀ x : ℕ+, ∃ y : ℕ+, x * (x + 1) * (x + 2) * (x + 3) = y^2 → False :=
by sorry

end NUMINAMATH_CALUDE_no_four_consecutive_integers_product_perfect_square_l1498_149839


namespace NUMINAMATH_CALUDE_linear_equations_solution_l1498_149822

theorem linear_equations_solution :
  (∃ x : ℚ, 2 * x - (x + 10) = 5 * x + 2 * (x - 1) ∧ x = -4/3) ∧
  (∃ y : ℚ, (3 * y + 2) / 2 - 1 = (2 * y - 1) / 4 - (2 * y + 1) / 5 ∧ y = -9/28) :=
by sorry

end NUMINAMATH_CALUDE_linear_equations_solution_l1498_149822


namespace NUMINAMATH_CALUDE_problem_solution_l1498_149840

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 4)
  (h2 : x * y = 64) :
  (x + y) / 2 = 13 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1498_149840


namespace NUMINAMATH_CALUDE_f_value_at_3_l1498_149899

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 3

-- State the theorem
theorem f_value_at_3 (a b c : ℝ) :
  f a b c (-3) = 7 → f a b c 3 = -13 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l1498_149899


namespace NUMINAMATH_CALUDE_f_pi_third_is_nonnegative_reals_l1498_149895

-- Define the set f(x)
def f (φ : Real) : Set Real :=
  {x : Real | x ≥ 0}

-- Theorem statement
theorem f_pi_third_is_nonnegative_reals :
  f (π / 3) = {x : Real | x ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_f_pi_third_is_nonnegative_reals_l1498_149895


namespace NUMINAMATH_CALUDE_f_range_on_interval_l1498_149891

/-- The function f(x) = 1 - 4x - 2x^2 -/
def f (x : ℝ) : ℝ := 1 - 4*x - 2*x^2

/-- The range of f(x) on the interval (1, +∞) is (-∞, -5) -/
theorem f_range_on_interval :
  Set.range (fun x => f x) ∩ Set.Ioi 1 = Set.Iio (-5) := by sorry

end NUMINAMATH_CALUDE_f_range_on_interval_l1498_149891


namespace NUMINAMATH_CALUDE_pineapples_sold_l1498_149813

theorem pineapples_sold (initial : ℕ) (rotten : ℕ) (fresh : ℕ) : 
  initial = 86 → rotten = 9 → fresh = 29 → initial - (fresh + rotten) = 48 := by
sorry

end NUMINAMATH_CALUDE_pineapples_sold_l1498_149813


namespace NUMINAMATH_CALUDE_equal_sides_implies_rhombus_rhombus_equal_diagonals_implies_square_equal_angles_implies_rectangle_l1498_149809

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define properties of quadrilaterals
def Quadrilateral.is_rhombus (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.is_square (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.is_rectangle (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.has_equal_sides (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.has_equal_diagonals (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.has_equal_angles (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statements
theorem equal_sides_implies_rhombus (q : Quadrilateral) :
  q.has_equal_sides → q.is_rhombus :=
sorry

theorem rhombus_equal_diagonals_implies_square (q : Quadrilateral) :
  q.is_rhombus → q.has_equal_diagonals → q.is_square :=
sorry

theorem equal_angles_implies_rectangle (q : Quadrilateral) :
  q.has_equal_angles → q.is_rectangle :=
sorry

end NUMINAMATH_CALUDE_equal_sides_implies_rhombus_rhombus_equal_diagonals_implies_square_equal_angles_implies_rectangle_l1498_149809


namespace NUMINAMATH_CALUDE_salaria_tree_count_l1498_149851

/-- Represents the types of orange trees --/
inductive TreeType
| A
| B

/-- Calculates the number of good oranges per tree per month --/
def goodOrangesPerTree (t : TreeType) : ℚ :=
  match t with
  | TreeType.A => 10 * (60 / 100)
  | TreeType.B => 15 * (1 / 3)

/-- Calculates the average number of good oranges per tree per month --/
def avgGoodOrangesPerTree : ℚ :=
  (goodOrangesPerTree TreeType.A + goodOrangesPerTree TreeType.B) / 2

/-- The total number of good oranges Salaria gets per month --/
def totalGoodOranges : ℚ := 55

/-- Theorem stating that the total number of trees Salaria has is 10 --/
theorem salaria_tree_count :
  totalGoodOranges / avgGoodOrangesPerTree = 10 := by
  sorry


end NUMINAMATH_CALUDE_salaria_tree_count_l1498_149851


namespace NUMINAMATH_CALUDE_characterize_bijection_condition_l1498_149849

/-- Given an even positive integer m, characterize all positive integers n for which
    there exists a bijection f from [1,n] to [1,n] satisfying the condition that
    for all x and y in [1,n] where n divides mx - y, n+1 divides f(x)^m - f(y). -/
theorem characterize_bijection_condition (m : ℕ) (h_m : Even m) (h_m_pos : 0 < m) :
  ∀ n : ℕ, 0 < n →
    (∃ f : Fin n → Fin n, Function.Bijective f ∧
      ∀ x y : Fin n, n ∣ m * x - y →
        (n + 1) ∣ (f x)^m - (f y)) ↔
    Nat.Prime (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_characterize_bijection_condition_l1498_149849


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1498_149817

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 8 * x - 18 = 0 ↔ x = 3 ∨ x = -4/3) → k = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1498_149817


namespace NUMINAMATH_CALUDE_greatest_integer_gcd_30_is_10_l1498_149867

theorem greatest_integer_gcd_30_is_10 : 
  ∃ n : ℕ, n < 100 ∧ Nat.gcd n 30 = 10 ∧ ∀ m : ℕ, m < 100 → Nat.gcd m 30 = 10 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_gcd_30_is_10_l1498_149867


namespace NUMINAMATH_CALUDE_not_p_equiv_p_and_q_equiv_l1498_149818

-- Define propositions p and q
def p (x : ℝ) := x * (x - 2) ≥ 0
def q (x : ℝ) := |x - 2| < 1

-- Theorem 1: Negation of p is equivalent to 0 < x < 2
theorem not_p_equiv (x : ℝ) : ¬(p x) ↔ 0 < x ∧ x < 2 := by sorry

-- Theorem 2: p and q together are equivalent to 2 ≤ x < 3
theorem p_and_q_equiv (x : ℝ) : p x ∧ q x ↔ 2 ≤ x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_not_p_equiv_p_and_q_equiv_l1498_149818


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_97_l1498_149836

theorem gcd_of_powers_of_97 : 
  Nat.Prime 97 → Nat.gcd (97^7 + 1) (97^7 + 97^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_97_l1498_149836


namespace NUMINAMATH_CALUDE_solve_equation_l1498_149844

-- Define the custom operation *
def star (a b : ℚ) : ℚ := 4 * a - 2 * b

-- State the theorem
theorem solve_equation : ∃ x : ℚ, star 3 (star 6 x) = 2 ∧ x = 19/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1498_149844


namespace NUMINAMATH_CALUDE_rhett_tax_percentage_l1498_149823

/-- Proof of Rhett's tax percentage --/
theorem rhett_tax_percentage :
  ∀ (salary : ℝ) (rent : ℝ) (tax_rate : ℝ),
    salary = 5000 →
    rent = 1350 →
    3 / 5 * (salary - tax_rate / 100 * salary) = 2 * rent →
    tax_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_rhett_tax_percentage_l1498_149823


namespace NUMINAMATH_CALUDE_chord_length_l1498_149850

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l1498_149850


namespace NUMINAMATH_CALUDE_distribute_5_3_l1498_149885

/-- The number of ways to distribute n indistinguishable objects into k distinct containers,
    with each container containing at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

theorem distribute_5_3 : distribute 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l1498_149885


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1498_149833

/-- Two vectors are parallel if their coordinates are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2*x, -3)
  are_parallel a b → x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1498_149833


namespace NUMINAMATH_CALUDE_exists_sum_of_five_squares_l1498_149824

theorem exists_sum_of_five_squares : 
  ∃ (n : ℕ) (a b c d e : ℤ), 
    (n : ℤ)^2 = a^2 + b^2 + c^2 + d^2 + e^2 ∧ 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
     c ≠ d ∧ c ≠ e ∧ 
     d ≠ e) ∧
    (a = 7 ∨ b = 7 ∨ c = 7 ∨ d = 7 ∨ e = 7) :=
by sorry

end NUMINAMATH_CALUDE_exists_sum_of_five_squares_l1498_149824
