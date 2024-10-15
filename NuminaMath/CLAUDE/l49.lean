import Mathlib

namespace NUMINAMATH_CALUDE_z_minus_two_purely_imaginary_l49_4998

def z : ℂ := Complex.mk 2 (-1)

theorem z_minus_two_purely_imaginary :
  Complex.im (z - 2) = Complex.im z ∧ Complex.re (z - 2) = 0 :=
sorry

end NUMINAMATH_CALUDE_z_minus_two_purely_imaginary_l49_4998


namespace NUMINAMATH_CALUDE_wedding_bouquets_l49_4906

/-- Represents the number of flowers of each type --/
structure FlowerCount where
  roses : ℕ
  lilies : ℕ
  tulips : ℕ
  sunflowers : ℕ

/-- Represents the requirements for a single bouquet --/
def BouquetRequirement : FlowerCount :=
  { roses := 2, lilies := 1, tulips := 3, sunflowers := 1 }

/-- Calculates the number of complete bouquets that can be made --/
def completeBouquets (available : FlowerCount) : ℕ :=
  min (available.roses / BouquetRequirement.roses)
    (min (available.lilies / BouquetRequirement.lilies)
      (min (available.tulips / BouquetRequirement.tulips)
        (available.sunflowers / BouquetRequirement.sunflowers)))

theorem wedding_bouquets :
  let initial : FlowerCount := { roses := 48, lilies := 40, tulips := 76, sunflowers := 34 }
  let wilted : FlowerCount := { roses := 24, lilies := 10, tulips := 14, sunflowers := 7 }
  let remaining : FlowerCount := {
    roses := initial.roses - wilted.roses,
    lilies := initial.lilies - wilted.lilies,
    tulips := initial.tulips - wilted.tulips,
    sunflowers := initial.sunflowers - wilted.sunflowers
  }
  completeBouquets remaining = 12 := by sorry

end NUMINAMATH_CALUDE_wedding_bouquets_l49_4906


namespace NUMINAMATH_CALUDE_complex_equation_solution_l49_4965

theorem complex_equation_solution :
  ∃ z : ℂ, (3 : ℂ) - 2 * Complex.I * z = -2 + 3 * Complex.I * z ∧ z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l49_4965


namespace NUMINAMATH_CALUDE_sphere_diameter_triple_volume_l49_4980

theorem sphere_diameter_triple_volume (π : ℝ) (h_π : π > 0) : 
  let r₁ : ℝ := 6
  let v₁ : ℝ := (4 / 3) * π * r₁^3
  let v₂ : ℝ := 3 * v₁
  let r₂ : ℝ := (v₂ * 3 / (4 * π))^(1/3)
  let d₂ : ℝ := 2 * r₂
  d₂ = 18 * (12 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_diameter_triple_volume_l49_4980


namespace NUMINAMATH_CALUDE_prob_different_suits_expanded_deck_l49_4996

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- Calculates the probability of drawing two cards of different suits -/
def prob_different_suits (d : Deck) : ℚ :=
  let remaining_cards := d.total_cards - 1
  let different_suit_cards := d.total_cards - d.cards_per_suit
  different_suit_cards / remaining_cards

/-- Theorem: The probability of drawing two cards of different suits
    from a 78-card deck with 6 suits of 13 cards each is 65/77 -/
theorem prob_different_suits_expanded_deck :
  let d : Deck := ⟨78, 6, 13, rfl⟩
  prob_different_suits d = 65 / 77 := by sorry

end NUMINAMATH_CALUDE_prob_different_suits_expanded_deck_l49_4996


namespace NUMINAMATH_CALUDE_altitude_inradius_sum_implies_equilateral_l49_4939

/-- A triangle with side lengths a, b, c, altitudes h₁, h₂, h₃, and inradius r. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  r : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_h₁ : 0 < h₁
  pos_h₂ : 0 < h₂
  pos_h₃ : 0 < h₃
  pos_r : 0 < r
  altitude_sum : h₁ + h₂ + h₃ = 9 * r

/-- A triangle is equilateral if all its sides are equal. -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- If the altitudes and the radius of the inscribed circle of a triangle satisfy
    h₁ + h₂ + h₃ = 9r, then the triangle is equilateral. -/
theorem altitude_inradius_sum_implies_equilateral (t : Triangle) :
  t.isEquilateral :=
sorry

end NUMINAMATH_CALUDE_altitude_inradius_sum_implies_equilateral_l49_4939


namespace NUMINAMATH_CALUDE_negative_roots_quadratic_l49_4966

/-- For a quadratic polynomial x^2 + 2(p+1)x + 9p - 5, both roots are negative if and only if 5/9 < p ≤ 1 or p ≥ 6 -/
theorem negative_roots_quadratic (p : ℝ) : 
  (∀ x : ℝ, x^2 + 2*(p+1)*x + 9*p - 5 = 0 → x < 0) ↔ 
  (5/9 < p ∧ p ≤ 1) ∨ p ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_negative_roots_quadratic_l49_4966


namespace NUMINAMATH_CALUDE_all_statements_valid_l49_4976

/-- Represents a simple programming language statement --/
inductive Statement
  | Assignment (var : String) (value : Int)
  | MultiAssignment (vars : List String) (values : List Int)
  | Input (prompt : Option String) (var : String)
  | Print (prompt : Option String) (expr : Option String)

/-- Checks if a statement is valid according to our rules --/
def isValid : Statement → Bool
  | Statement.Assignment _ _ => true
  | Statement.MultiAssignment vars values => vars.length == values.length
  | Statement.Input _ _ => true
  | Statement.Print _ _ => true

/-- The set of corrected statements --/
def correctedStatements : List Statement := [
  Statement.MultiAssignment ["A", "B"] [50, 50],
  Statement.MultiAssignment ["x", "y", "z"] [1, 2, 3],
  Statement.Input (some "How old are you?") "x",
  Statement.Input none "x",
  Statement.Print (some "A+B=") (some "C"),
  Statement.Print (some "Good-bye!") none
]

theorem all_statements_valid : ∀ s ∈ correctedStatements, isValid s := by sorry

end NUMINAMATH_CALUDE_all_statements_valid_l49_4976


namespace NUMINAMATH_CALUDE_fourth_power_equals_sixteenth_l49_4989

theorem fourth_power_equals_sixteenth (n : ℝ) : (1/4 : ℝ)^n = 0.0625 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_equals_sixteenth_l49_4989


namespace NUMINAMATH_CALUDE_hamburgers_left_over_l49_4961

def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

theorem hamburgers_left_over : hamburgers_made - hamburgers_served = 6 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_left_over_l49_4961


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l49_4900

theorem arithmetic_mean_problem (a b c d : ℝ) :
  (a + b + c + d + 130) / 5 = 90 →
  (a + b + c + d) / 4 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l49_4900


namespace NUMINAMATH_CALUDE_dime_difference_l49_4946

/-- Represents the types of coins in the piggy bank -/
inductive Coin
  | Nickel
  | Dime
  | Quarter

/-- Represents the piggy bank with its coin composition -/
structure PiggyBank where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  total_coins : nickels + dimes + quarters = 100
  total_value : 5 * nickels + 10 * dimes + 25 * quarters = 1005

/-- The value of a given coin in cents -/
def coinValue : Coin → ℕ
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Theorem stating the difference between max and min number of dimes -/
theorem dime_difference (pb : PiggyBank) : 
  ∃ (min_dimes max_dimes : ℕ), 
    (∀ pb' : PiggyBank, pb'.dimes ≥ min_dimes) ∧ 
    (∃ pb' : PiggyBank, pb'.dimes = min_dimes) ∧
    (∀ pb' : PiggyBank, pb'.dimes ≤ max_dimes) ∧ 
    (∃ pb' : PiggyBank, pb'.dimes = max_dimes) ∧
    max_dimes - min_dimes = 100 :=
  sorry


end NUMINAMATH_CALUDE_dime_difference_l49_4946


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l49_4931

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- The perimeter of the trapezoid -/
  perimeter : ℝ
  /-- The diagonal bisects the obtuse angle -/
  diagonal_bisects_obtuse_angle : Bool

/-- Calculates the area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific isosceles trapezoid is 96 -/
theorem isosceles_trapezoid_area :
  ∀ t : IsoscelesTrapezoid,
    t.shorter_base = 3 ∧
    t.perimeter = 42 ∧
    t.diagonal_bisects_obtuse_angle = true →
    area t = 96 :=
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l49_4931


namespace NUMINAMATH_CALUDE_fraction_sum_l49_4907

theorem fraction_sum : (3 : ℚ) / 5 + 2 / 15 = 11 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l49_4907


namespace NUMINAMATH_CALUDE_bus_speed_problem_l49_4999

theorem bus_speed_problem (bus_length : ℝ) (fast_bus_speed : ℝ) (passing_time : ℝ) :
  bus_length = 3125 →
  fast_bus_speed = 40 →
  passing_time = 50/3600 →
  ∃ (slow_bus_speed : ℝ),
    slow_bus_speed = (2 * bus_length / 1000) / passing_time - fast_bus_speed ∧
    slow_bus_speed = 410 :=
by sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l49_4999


namespace NUMINAMATH_CALUDE_max_sum_perfect_square_fraction_l49_4911

def is_perfect_square (n : ℚ) : Prop := ∃ m : ℕ, n = (m : ℚ) ^ 2

def is_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

theorem max_sum_perfect_square_fraction :
  ∀ A B C D : ℕ,
    is_digit A → is_digit B → is_digit C → is_digit D →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    is_perfect_square ((A + B : ℚ) / (C + D)) →
    ∀ A' B' C' D' : ℕ,
      is_digit A' → is_digit B' → is_digit C' → is_digit D' →
      A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
      is_perfect_square ((A' + B' : ℚ) / (C' + D')) →
      (A + B : ℚ) / (C + D) ≥ (A' + B' : ℚ) / (C' + D') →
      A + B = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_perfect_square_fraction_l49_4911


namespace NUMINAMATH_CALUDE_cubic_root_product_l49_4914

theorem cubic_root_product (a b c : ℂ) : 
  (3 * a^3 - 9 * a^2 + a - 7 = 0) ∧ 
  (3 * b^3 - 9 * b^2 + b - 7 = 0) ∧ 
  (3 * c^3 - 9 * c^2 + c - 7 = 0) →
  a * b * c = 7/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l49_4914


namespace NUMINAMATH_CALUDE_jeff_journey_distance_l49_4952

-- Define the journey segments
def segment1_speed : ℝ := 80
def segment1_time : ℝ := 6
def segment2_speed : ℝ := 60
def segment2_time : ℝ := 4
def segment3_speed : ℝ := 40
def segment3_time : ℝ := 2

-- Define the total distance function
def total_distance : ℝ := 
  segment1_speed * segment1_time + 
  segment2_speed * segment2_time + 
  segment3_speed * segment3_time

-- Theorem statement
theorem jeff_journey_distance : total_distance = 800 := by
  sorry

end NUMINAMATH_CALUDE_jeff_journey_distance_l49_4952


namespace NUMINAMATH_CALUDE_max_dot_product_on_ellipse_l49_4974

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

/-- Definition of the center O -/
def O : ℝ × ℝ := (0, 0)

/-- Definition of the left focus F -/
def F : ℝ × ℝ := (-1, 0)

/-- Definition of the dot product of OP and FP -/
def dot_product (x y : ℝ) : ℝ := x^2 + x + y^2

theorem max_dot_product_on_ellipse :
  ∃ (max : ℝ), max = 6 ∧
  ∀ (x y : ℝ), is_on_ellipse x y →
  dot_product x y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_on_ellipse_l49_4974


namespace NUMINAMATH_CALUDE_inequality_solution_l49_4959

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l49_4959


namespace NUMINAMATH_CALUDE_max_cards_in_original_position_l49_4944

/-- Represents a two-digit number card -/
structure Card :=
  (tens : Nat)
  (ones : Nat)
  (h1 : tens < 10)
  (h2 : ones < 10)

/-- The list of all cards from 00 to 99 in ascending order -/
def initial_arrangement : List Card := sorry

/-- Checks if two cards are adjacent according to the rearrangement rule -/
def are_adjacent (c1 c2 : Card) : Prop := sorry

/-- A valid rearrangement of cards -/
def valid_rearrangement (arrangement : List Card) : Prop :=
  arrangement.length = 100 ∧
  ∀ i, i < 99 → are_adjacent (arrangement.get ⟨i, sorry⟩) (arrangement.get ⟨i+1, sorry⟩)

/-- The number of cards in their original positions after rearrangement -/
def cards_in_original_position (arrangement : List Card) : Nat := sorry

/-- Theorem stating the maximum number of cards that can remain in their original positions -/
theorem max_cards_in_original_position :
  ∀ arrangement : List Card,
    valid_rearrangement arrangement →
    cards_in_original_position arrangement ≤ 50 :=
sorry

end NUMINAMATH_CALUDE_max_cards_in_original_position_l49_4944


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l49_4910

/-- A fourth-degree polynomial with real coefficients -/
def fourth_degree_poly (g : ℝ → ℝ) : Prop :=
  ∃ a b c d e : ℝ, ∀ x, g x = a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem polynomial_value_theorem (g : ℝ → ℝ) 
  (h_poly : fourth_degree_poly g)
  (h_m1 : |g (-1)| = 15)
  (h_0 : |g 0| = 15)
  (h_2 : |g 2| = 15)
  (h_3 : |g 3| = 15)
  (h_4 : |g 4| = 15) :
  |g 1| = 11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l49_4910


namespace NUMINAMATH_CALUDE_cylinder_radius_theorem_l49_4970

/-- The original radius of a cylinder with the given properties -/
def original_radius : ℝ := 8

/-- The original height of the cylinder -/
def original_height : ℝ := 3

/-- The increase in either radius or height -/
def increase : ℝ := 4

theorem cylinder_radius_theorem :
  (π * (original_radius + increase)^2 * original_height = 
   π * original_radius^2 * (original_height + increase)) ∧
  original_radius > 0 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_theorem_l49_4970


namespace NUMINAMATH_CALUDE_quadratic_inequality_l49_4985

theorem quadratic_inequality (x : ℝ) : x^2 + 4*x - 21 > 0 ↔ x < -7 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l49_4985


namespace NUMINAMATH_CALUDE_triangle_circumcircle_distance_sum_bounds_l49_4912

theorem triangle_circumcircle_distance_sum_bounds :
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (2 * Real.sqrt 3, 0)
  let B : ℝ × ℝ := (0, 2)
  let C : ℝ → ℝ × ℝ := fun θ ↦ (Real.sqrt 3 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)
  ∀ θ : ℝ,
    let P := C θ
    let dist_squared (X Y : ℝ × ℝ) := (X.1 - Y.1)^2 + (X.2 - Y.2)^2
    let sum := dist_squared P O + dist_squared P A + dist_squared P B
    sum ≤ 32 ∧ sum ≥ 16 ∧ (∃ θ₁ θ₂, C θ₁ = 32 ∧ C θ₂ = 16) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_circumcircle_distance_sum_bounds_l49_4912


namespace NUMINAMATH_CALUDE_right_triangle_ratio_minimum_l49_4984

theorem right_triangle_ratio_minimum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  c / (a + b) ≥ Real.sqrt 2 / 2 ∧ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 = z^2 ∧ z / (x + y) = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_minimum_l49_4984


namespace NUMINAMATH_CALUDE_replaced_person_weight_l49_4962

theorem replaced_person_weight
  (num_persons : ℕ)
  (avg_weight_increase : ℝ)
  (new_person_weight : ℝ) :
  num_persons = 5 →
  avg_weight_increase = 1.5 →
  new_person_weight = 72.5 →
  new_person_weight - (num_persons : ℝ) * avg_weight_increase = 65 :=
by sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l49_4962


namespace NUMINAMATH_CALUDE_ewan_sequence_contains_113_l49_4993

def ewanSequence (n : ℕ) : ℤ := 3 + 11 * (n - 1)

theorem ewan_sequence_contains_113 :
  ∃ n : ℕ, ewanSequence n = 113 ∧
  (∀ m : ℕ, ewanSequence m ≠ 111) ∧
  (∀ m : ℕ, ewanSequence m ≠ 112) ∧
  (∀ m : ℕ, ewanSequence m ≠ 110) ∧
  (∀ m : ℕ, ewanSequence m ≠ 114) :=
by sorry


end NUMINAMATH_CALUDE_ewan_sequence_contains_113_l49_4993


namespace NUMINAMATH_CALUDE_sinusoidal_function_properties_l49_4909

/-- Given a sinusoidal function with specific properties, prove its expression and range -/
theorem sinusoidal_function_properties (f : ℝ → ℝ) (A ω φ : ℝ)
  (h_def : ∀ x, f x = A * Real.sin (ω * x + φ))
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π)
  (h_symmetry : (π / 2) = π / ω)  -- Distance between adjacent axes of symmetry
  (h_lowest : f (2 * π / 3) = -1/2)  -- One of the lowest points
  : (∀ x, f x = (1/2) * Real.sin (2 * x + π/6)) ∧
    (∀ x, x ∈ Set.Icc (-π/6) (π/3) → f x ∈ Set.Icc (-1/4) (1/2)) := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_function_properties_l49_4909


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l49_4958

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Theorem: Given S_3 = 2 and S_6 = 8, then S_9 = 18 -/
theorem arithmetic_sequence_sum 
  (seq : ArithmeticSequence) 
  (h1 : seq.S 3 = 2) 
  (h2 : seq.S 6 = 8) : 
  seq.S 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l49_4958


namespace NUMINAMATH_CALUDE_point_outside_circle_l49_4968

def imaginary_unit : ℂ := Complex.I

theorem point_outside_circle (a b : ℝ) (h : a + b * imaginary_unit = (2 + imaginary_unit) / (1 - imaginary_unit)) :
  a^2 + b^2 > 2 := by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l49_4968


namespace NUMINAMATH_CALUDE_alex_last_five_shots_l49_4972

/-- Represents the number of shots made by Alex -/
structure ShotsMade where
  initial : ℕ
  after_60 : ℕ
  final : ℕ

/-- Represents the shooting percentages at different stages -/
structure ShootingPercentages where
  initial : ℚ
  after_60 : ℚ
  final : ℚ

/-- Theorem stating the number of shots Alex made in the last 5 attempts -/
theorem alex_last_five_shots 
  (shots : ShotsMade)
  (percentages : ShootingPercentages)
  (h1 : shots.initial = 30)
  (h2 : shots.after_60 = 37)
  (h3 : shots.final = 39)
  (h4 : percentages.initial = 3/5)
  (h5 : percentages.after_60 = 31/50)
  (h6 : percentages.final = 3/5) :
  shots.final - shots.after_60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_alex_last_five_shots_l49_4972


namespace NUMINAMATH_CALUDE_museum_survey_visitors_l49_4977

/-- Represents the survey results of visitors to a modern art museum --/
structure MuseumSurvey where
  total : ℕ
  enjoyed_and_understood : ℕ
  neither_enjoyed_nor_understood : ℕ

/-- The conditions of the survey --/
def survey_conditions (s : MuseumSurvey) : Prop :=
  s.neither_enjoyed_nor_understood = 110 ∧
  s.enjoyed_and_understood = (3 : ℚ) / 4 * s.total

/-- The theorem to be proved --/
theorem museum_survey_visitors (s : MuseumSurvey) :
  survey_conditions s → s.total = 440 := by
  sorry


end NUMINAMATH_CALUDE_museum_survey_visitors_l49_4977


namespace NUMINAMATH_CALUDE_max_volume_sphere_in_cube_l49_4908

theorem max_volume_sphere_in_cube (edge_length : Real) (h : edge_length = 1) :
  let sphere_volume := (4 / 3) * Real.pi * (edge_length / 2)^3
  sphere_volume = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_max_volume_sphere_in_cube_l49_4908


namespace NUMINAMATH_CALUDE_laura_rental_cost_l49_4925

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, number of days, and miles driven. -/
def rentalCost (dailyRate : ℝ) (mileageRate : ℝ) (days : ℕ) (miles : ℝ) : ℝ :=
  dailyRate * (days : ℝ) + mileageRate * miles

/-- Theorem stating that the total cost of Laura's car rental is $165. -/
theorem laura_rental_cost :
  let dailyRate : ℝ := 30
  let mileageRate : ℝ := 0.25
  let days : ℕ := 3
  let miles : ℝ := 300
  rentalCost dailyRate mileageRate days miles = 165 := by
  sorry

end NUMINAMATH_CALUDE_laura_rental_cost_l49_4925


namespace NUMINAMATH_CALUDE_square_plus_one_geq_double_for_all_reals_l49_4904

theorem square_plus_one_geq_double_for_all_reals :
  ∀ a : ℝ, a^2 + 1 ≥ 2*a := by sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_double_for_all_reals_l49_4904


namespace NUMINAMATH_CALUDE_problem_solution_l49_4935

theorem problem_solution (x y : ℚ) 
  (eq1 : 5 * x - 3 * y = 8) 
  (eq2 : 2 * x + 7 * y = 1) : 
  3 * (x + y + 4) = 12 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l49_4935


namespace NUMINAMATH_CALUDE_dense_S_l49_4975

-- Define the set S
def S : Set ℝ := {x : ℝ | ∃ (m n : ℕ+), x = Real.sqrt m - Real.sqrt n}

-- State the theorem
theorem dense_S : ∀ (a b : ℝ), a < b → Set.Infinite (S ∩ Set.Ioo a b) := by sorry

end NUMINAMATH_CALUDE_dense_S_l49_4975


namespace NUMINAMATH_CALUDE_mike_car_parts_cost_l49_4913

-- Define the cost of speakers
def speaker_cost : ℚ := 118.54

-- Define the cost of tires
def tire_cost : ℚ := 106.33

-- Define the total cost
def total_cost : ℚ := speaker_cost + tire_cost

-- Theorem to prove
theorem mike_car_parts_cost : total_cost = 224.87 := by
  sorry

end NUMINAMATH_CALUDE_mike_car_parts_cost_l49_4913


namespace NUMINAMATH_CALUDE_percentage_increase_johns_raise_l49_4918

theorem percentage_increase (original new : ℝ) (h1 : original > 0) (h2 : new > original) :
  (new - original) / original * 100 = 100 ↔ new = 2 * original :=
by sorry

theorem johns_raise :
  let original : ℝ := 40
  let new : ℝ := 80
  (new - original) / original * 100 = 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_johns_raise_l49_4918


namespace NUMINAMATH_CALUDE_polynomial_expansion_l49_4947

theorem polynomial_expansion (x : ℝ) :
  (5 * x - 3) * (2 * x^2 + 4 * x + 1) = 10 * x^3 + 14 * x^2 - 7 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l49_4947


namespace NUMINAMATH_CALUDE_integer_set_average_l49_4936

theorem integer_set_average : ∀ (a b c d : ℤ),
  a < b ∧ b < c ∧ c < d →  -- Ensure the integers are different and ordered
  d = 90 →                 -- The largest integer is 90
  a ≥ 5 →                  -- The smallest integer is at least 5
  (a + b + c + d) / 4 = 27 -- The average is 27
  := by sorry

end NUMINAMATH_CALUDE_integer_set_average_l49_4936


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_l49_4903

def numbers : List Nat := [9, 15, 27]

/-- The greatest common factor of 9, 15, and 27 -/
def A : Nat := numbers.foldl Nat.gcd 0

/-- The least common multiple of 9, 15, and 27 -/
def B : Nat := numbers.foldl Nat.lcm 1

/-- Theorem stating that the sum of the greatest common factor and 
    the least common multiple of 9, 15, and 27 is equal to 138 -/
theorem gcf_lcm_sum : A + B = 138 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_l49_4903


namespace NUMINAMATH_CALUDE_cattle_market_problem_l49_4951

/-- The number of animals each person brought to the market satisfies the given conditions --/
theorem cattle_market_problem (j h d : ℕ) : 
  (j + 5 = 2 * (h - 5)) →  -- Condition 1
  (h + 13 = 3 * (d - 13)) →  -- Condition 2
  (d + 3 = 6 * (j - 3)) →  -- Condition 3
  j = 7 ∧ h = 11 ∧ d = 21 := by
sorry

end NUMINAMATH_CALUDE_cattle_market_problem_l49_4951


namespace NUMINAMATH_CALUDE_A_equals_B_l49_4948

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

theorem A_equals_B : A = B := by
  sorry

end NUMINAMATH_CALUDE_A_equals_B_l49_4948


namespace NUMINAMATH_CALUDE_cubic_sum_reciprocal_l49_4997

theorem cubic_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^3 + 1/a^3 = 2 * Real.sqrt 5 ∨ a^3 + 1/a^3 = -2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_cubic_sum_reciprocal_l49_4997


namespace NUMINAMATH_CALUDE_juniors_in_program_l49_4953

theorem juniors_in_program (total_students : ℕ) (junior_club_percent : ℚ) (senior_club_percent : ℚ)
  (junior_senior_ratio : ℚ) (h_total : total_students = 40)
  (h_junior_percent : junior_club_percent = 3/10)
  (h_senior_percent : senior_club_percent = 1/5)
  (h_ratio : junior_senior_ratio = 3/2) :
  ∃ (juniors seniors : ℕ),
    juniors + seniors = total_students ∧
    (junior_club_percent * juniors : ℚ) / (senior_club_percent * seniors) = junior_senior_ratio ∧
    juniors = 20 := by
  sorry

end NUMINAMATH_CALUDE_juniors_in_program_l49_4953


namespace NUMINAMATH_CALUDE_min_cube_sum_l49_4922

theorem min_cube_sum (w z : ℂ) (h1 : Complex.abs (w + z) = 2) (h2 : Complex.abs (w^2 + z^2) = 16) :
  Complex.abs (w^3 + z^3) ≥ 22 := by
  sorry

end NUMINAMATH_CALUDE_min_cube_sum_l49_4922


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l49_4915

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sqrt (x + 1)}
def B : Set ℝ := {x | 1 / (x + 1) < 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | x > 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l49_4915


namespace NUMINAMATH_CALUDE_number_above_200_is_91_l49_4963

/-- Represents the array where the k-th row contains the first 2k natural numbers -/
def array_sum (k : ℕ) : ℕ := k * (2 * k + 1) / 2

/-- The row number in which 200 is located -/
def row_of_200 : ℕ := 14

/-- The starting number of the row containing 200 -/
def start_of_row_200 : ℕ := array_sum (row_of_200 - 1) + 1

/-- The position of 200 in its row -/
def position_of_200 : ℕ := 200 - start_of_row_200 + 1

/-- The number directly above 200 -/
def number_above_200 : ℕ := array_sum (row_of_200 - 1)

theorem number_above_200_is_91 : number_above_200 = 91 := by
  sorry

end NUMINAMATH_CALUDE_number_above_200_is_91_l49_4963


namespace NUMINAMATH_CALUDE_functional_equation_unique_solution_l49_4969

open Set

theorem functional_equation_unique_solution
  (f : ℝ → ℝ) (a b : ℝ) :
  (0 < a) →
  (0 < b) →
  (∀ x, 0 ≤ x → 0 ≤ f x) →
  (∀ x, 0 ≤ x → f (f x) + a * f x = b * (a + b) * x) →
  (∀ x, 0 ≤ x → f x = b * x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_unique_solution_l49_4969


namespace NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_21_l49_4937

theorem constant_term_expansion (x : ℝ) : 
  (x^3 + x^2 + 3) * (2*x^4 + x^2 + 7) = 2*x^7 + x^5 + 7*x^3 + 2*x^6 + x^4 + 7*x^2 + 6*x^4 + 3*x^2 + 21 :=
by sorry

theorem constant_term_is_21 : 
  ∃ p : Polynomial ℝ, (Polynomial.eval 0 p = 21 ∧ 
    ∀ x : ℝ, p.eval x = (x^3 + x^2 + 3) * (2*x^4 + x^2 + 7)) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_21_l49_4937


namespace NUMINAMATH_CALUDE_constant_value_l49_4992

theorem constant_value : ∀ (x : ℝ) (c : ℝ),
  (5 * x + c = 10 * x - 22) →
  (x = 5) →
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l49_4992


namespace NUMINAMATH_CALUDE_valid_n_values_l49_4928

def is_valid_n (f : ℤ → ℤ) (n : ℤ) : Prop :=
  f 1 = -1 ∧ f 4 = 2 ∧ f 8 = 34 ∧ f n = n^2 - 4*n - 18

theorem valid_n_values (f : ℤ → ℤ) (n : ℤ) 
  (h : is_valid_n f n) : n = 3 ∨ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_valid_n_values_l49_4928


namespace NUMINAMATH_CALUDE_rod_cutting_l49_4960

theorem rod_cutting (rod_length piece_length : ℝ) (h1 : rod_length = 42.5) (h2 : piece_length = 0.85) :
  ⌊rod_length / piece_length⌋ = 50 := by
sorry

end NUMINAMATH_CALUDE_rod_cutting_l49_4960


namespace NUMINAMATH_CALUDE_no_natural_solutions_l49_4933

theorem no_natural_solutions : ¬∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l49_4933


namespace NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l49_4979

/-- The problem of Jack, Christina, and Lindy meeting --/
theorem jack_christina_lindy_meeting
  (initial_distance : ℝ)
  (jack_speed : ℝ)
  (christina_speed : ℝ)
  (lindy_speed : ℝ)
  (h1 : initial_distance = 360)
  (h2 : jack_speed = 5)
  (h3 : christina_speed = 7)
  (h4 : lindy_speed = 12)
  (h5 : jack_speed > 0)
  (h6 : christina_speed > 0)
  (h7 : lindy_speed > jack_speed + christina_speed) :
  let meeting_time := initial_distance / (jack_speed + christina_speed)
  lindy_speed * meeting_time = initial_distance :=
sorry

end NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l49_4979


namespace NUMINAMATH_CALUDE_sum_of_radii_is_14_l49_4955

-- Define the circle with center C
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the condition of being tangent to positive x- and y-axes
def tangentToAxes (c : Circle) : Prop :=
  c.center.1 = c.radius ∧ c.center.2 = c.radius

-- Define the condition of being externally tangent to another circle
def externallyTangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

-- Theorem statement
theorem sum_of_radii_is_14 :
  ∃ (c1 c2 : Circle),
    tangentToAxes c1 ∧
    tangentToAxes c2 ∧
    c1.center ≠ c2.center ∧
    externallyTangent c1 { center := (5, 0), radius := 2 } ∧
    externallyTangent c2 { center := (5, 0), radius := 2 } ∧
    c1.radius + c2.radius = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_radii_is_14_l49_4955


namespace NUMINAMATH_CALUDE_min_distance_curve_line_l49_4901

theorem min_distance_curve_line (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) :
  ∃ (x y : ℝ), (a - c)^2 + (b - d)^2 ≥ (x - y)^2 ∧ (x - y)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_min_distance_curve_line_l49_4901


namespace NUMINAMATH_CALUDE_rect_to_polar_8_8_l49_4924

/-- Conversion from rectangular to polar coordinates -/
theorem rect_to_polar_8_8 :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 8 * Real.sqrt 2 ∧ θ = π / 4 ∧
  8 = r * Real.cos θ ∧ 8 = r * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_rect_to_polar_8_8_l49_4924


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l49_4943

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : Nat
  blue : Nat
  white : Nat

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : Nat
  blue_blue : Nat
  red_white : Nat
  white_white : Nat

/-- The main theorem to prove -/
theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.red = 4 ∧ 
  counts.blue = 7 ∧ 
  counts.white = 10 ∧
  pairs.red_red = 3 ∧
  pairs.blue_blue = 4 ∧
  pairs.red_white = 3 →
  pairs.white_white = 4 := by
  sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l49_4943


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l49_4930

theorem decimal_to_fraction : 
  ∀ (n : ℕ), (3 : ℚ) / 10 + (24 : ℚ) / (99 * 10^n) = 19 / 33 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l49_4930


namespace NUMINAMATH_CALUDE_sixThousandthTerm_l49_4957

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a₁ : ℝ
  -- Common difference
  d : ℝ
  -- Parameters p and r
  p : ℝ
  r : ℝ
  -- Conditions on the first four terms
  h₁ : a₁ = 2 * p
  h₂ : a₁ + d = 14
  h₃ : a₁ + 2 * d = 4 * p - r
  h₄ : a₁ + 3 * d = 4 * p + r

/-- The nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1) * seq.d

/-- Theorem stating that the 6000th term is 24006 -/
theorem sixThousandthTerm (seq : ArithmeticSequence) : nthTerm seq 6000 = 24006 := by
  sorry

end NUMINAMATH_CALUDE_sixThousandthTerm_l49_4957


namespace NUMINAMATH_CALUDE_ab_length_is_eleven_l49_4954

-- Define the triangle structures
structure Triangle :=
  (a b c : ℝ)

-- Define isosceles property
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Define perimeter
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

theorem ab_length_is_eleven 
  (ABC CBD : Triangle)
  (ABC_isosceles : isIsosceles ABC)
  (CBD_isosceles : isIsosceles CBD)
  (CBD_perimeter : perimeter CBD = 24)
  (ABC_perimeter : perimeter ABC = 25)
  (BD_length : CBD.c = 10) :
  ABC.c = 11 := by
  sorry

end NUMINAMATH_CALUDE_ab_length_is_eleven_l49_4954


namespace NUMINAMATH_CALUDE_total_money_earned_l49_4934

/-- The price per kg of fish in dollars -/
def price_per_kg : ℝ := 20

/-- The amount of fish in kg caught in the past four months -/
def past_four_months_catch : ℝ := 80

/-- The amount of fish in kg caught today -/
def today_catch : ℝ := 2 * past_four_months_catch

/-- The total amount of fish in kg caught in the past four months including today -/
def total_catch : ℝ := past_four_months_catch + today_catch

/-- Theorem: The total money earned by Erica in the past four months including today is $4800 -/
theorem total_money_earned : price_per_kg * total_catch = 4800 := by
  sorry

end NUMINAMATH_CALUDE_total_money_earned_l49_4934


namespace NUMINAMATH_CALUDE_laura_five_dollar_bills_l49_4995

/-- Represents the number of bills of each denomination in Laura's piggy bank -/
structure PiggyBank where
  ones : ℕ
  twos : ℕ
  fives : ℕ

/-- The conditions of Laura's piggy bank -/
def laura_piggy_bank (pb : PiggyBank) : Prop :=
  pb.ones + pb.twos + pb.fives = 40 ∧
  pb.ones + 2 * pb.twos + 5 * pb.fives = 120 ∧
  pb.twos = 2 * pb.ones

theorem laura_five_dollar_bills :
  ∃ (pb : PiggyBank), laura_piggy_bank pb ∧ pb.fives = 16 :=
sorry

end NUMINAMATH_CALUDE_laura_five_dollar_bills_l49_4995


namespace NUMINAMATH_CALUDE_distance_to_outside_point_gt_three_l49_4991

/-- A circle with center O and radius 3 -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 3)

/-- A point outside the circle -/
structure OutsidePoint (c : Circle) :=
  (point : ℝ × ℝ)
  (h_outside : dist point c.center > c.radius)

/-- The theorem stating that the distance from the center to an outside point is greater than 3 -/
theorem distance_to_outside_point_gt_three (c : Circle) (p : OutsidePoint c) :
  dist p.point c.center > 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_outside_point_gt_three_l49_4991


namespace NUMINAMATH_CALUDE_right_angle_triangle_identification_l49_4988

/-- Checks if three lengths can form a right-angled triangle -/
def isRightAngleTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

theorem right_angle_triangle_identification :
  (¬ isRightAngleTriangle 2 3 4) ∧
  (¬ isRightAngleTriangle 3 3 4) ∧
  (isRightAngleTriangle 9 12 15) ∧
  (¬ isRightAngleTriangle 4 5 6) := by
  sorry

#check right_angle_triangle_identification

end NUMINAMATH_CALUDE_right_angle_triangle_identification_l49_4988


namespace NUMINAMATH_CALUDE_seal_releases_three_songs_per_month_l49_4923

/-- Represents the earnings per song in dollars -/
def earnings_per_song : ℕ := 2000

/-- Represents the total earnings in the first 3 years in dollars -/
def total_earnings : ℕ := 216000

/-- Represents the number of months in 3 years -/
def months_in_three_years : ℕ := 3 * 12

/-- Represents the number of songs released per month -/
def songs_per_month : ℕ := total_earnings / earnings_per_song / months_in_three_years

theorem seal_releases_three_songs_per_month :
  songs_per_month = 3 :=
by sorry

end NUMINAMATH_CALUDE_seal_releases_three_songs_per_month_l49_4923


namespace NUMINAMATH_CALUDE_employee_salary_problem_l49_4982

theorem employee_salary_problem (num_employees : ℕ) (salary_increase : ℕ) (manager_salary : ℕ) :
  num_employees = 24 →
  salary_increase = 400 →
  manager_salary = 11500 →
  ∃ (avg_salary : ℕ),
    avg_salary * num_employees + manager_salary = (avg_salary + salary_increase) * (num_employees + 1) ∧
    avg_salary = 1500 := by
  sorry

end NUMINAMATH_CALUDE_employee_salary_problem_l49_4982


namespace NUMINAMATH_CALUDE_junior_score_l49_4967

theorem junior_score (n : ℝ) (junior_score : ℝ) : 
  n > 0 →
  (0.3 * n * junior_score + 0.7 * n * 75) / n = 78 →
  junior_score = 85 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l49_4967


namespace NUMINAMATH_CALUDE_perpendicular_slope_l49_4938

theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 6 * y = 12) →
  (∃ m : ℝ, m = -3/2 ∧ m * (2/3) = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l49_4938


namespace NUMINAMATH_CALUDE_x_minus_y_equals_six_l49_4971

theorem x_minus_y_equals_six (x y : ℚ) 
  (eq1 : 3 * x + 2 * y = 16) 
  (eq2 : x + 3 * y = 26/5) : 
  x - y = 6 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_six_l49_4971


namespace NUMINAMATH_CALUDE_expansion_terms_count_expansion_terms_count_is_ten_l49_4949

theorem expansion_terms_count : ℕ :=
  let expression := (fun (x y : ℝ) => ((x + 5*y)^3 * (x - 5*y)^3)^3)
  let simplified := (fun (x y : ℝ) => (x^2 - 25*y^2)^9)
  let distinct_terms_count := 10
  distinct_terms_count

#check expansion_terms_count

theorem expansion_terms_count_is_ten : expansion_terms_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_expansion_terms_count_is_ten_l49_4949


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l49_4902

theorem isosceles_right_triangle (a b c h_a h_b h_c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  h_a > 0 → h_b > 0 → h_c > 0 →
  a * h_a = 2 * area → b * h_b = 2 * area → c * h_c = 2 * area →
  a ≤ h_a → b ≤ h_b →
  a = b ∧ c = a * Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_isosceles_right_triangle_l49_4902


namespace NUMINAMATH_CALUDE_factor_implies_coefficient_value_l49_4926

/-- Given a polynomial Q(x) = x^4 + 3x^3 + ax^2 + 17x + 27, 
    if (x-3) is a factor of Q(x), then a = -80/3 -/
theorem factor_implies_coefficient_value (a : ℚ) : 
  let Q := fun (x : ℚ) => x^4 + 3*x^3 + a*x^2 + 17*x + 27
  (∃ (P : ℚ → ℚ), Q = fun x => P x * (x - 3)) → a = -80/3 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_coefficient_value_l49_4926


namespace NUMINAMATH_CALUDE_savings_theorem_l49_4964

/-- Represents the prices of food items and meals --/
structure FoodPrices where
  burger : ℝ
  fries : ℝ
  drink : ℝ
  burgerMeal : ℝ
  kidsBurger : ℝ
  kidsFries : ℝ
  kidsJuice : ℝ
  kidsMeal : ℝ

/-- Calculates the savings when buying meals instead of individual items --/
def calculateSavings (prices : FoodPrices) : ℝ :=
  let individualCost := 
    2 * (prices.burger + prices.fries + prices.drink) +
    2 * (prices.kidsBurger + prices.kidsFries + prices.kidsJuice)
  let mealCost := 2 * prices.burgerMeal + 2 * prices.kidsMeal
  individualCost - mealCost

/-- The savings theorem --/
theorem savings_theorem (prices : FoodPrices) 
  (h1 : prices.burger = 5)
  (h2 : prices.fries = 3)
  (h3 : prices.drink = 3)
  (h4 : prices.burgerMeal = 9.5)
  (h5 : prices.kidsBurger = 3)
  (h6 : prices.kidsFries = 2)
  (h7 : prices.kidsJuice = 2)
  (h8 : prices.kidsMeal = 5) :
  calculateSavings prices = 7 := by
  sorry

end NUMINAMATH_CALUDE_savings_theorem_l49_4964


namespace NUMINAMATH_CALUDE_money_transfer_proof_l49_4941

/-- The amount of money (in won) the older brother gave to the younger brother -/
def money_transferred : ℕ := sorry

/-- The initial amount of money (in won) the older brother had -/
def older_brother_initial : ℕ := 2800

/-- The initial amount of money (in won) the younger brother had -/
def younger_brother_initial : ℕ := 1500

/-- The difference in money (in won) between the brothers after the transfer -/
def final_difference : ℕ := 360

theorem money_transfer_proof :
  (older_brother_initial - money_transferred) - (younger_brother_initial + money_transferred) = final_difference ∧
  money_transferred = 470 := by sorry

end NUMINAMATH_CALUDE_money_transfer_proof_l49_4941


namespace NUMINAMATH_CALUDE_value_set_of_m_l49_4973

def A : Set ℝ := {x : ℝ | x^2 + 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m*x + 1 = 0}

theorem value_set_of_m : 
  ∀ m : ℝ, (A ∪ B m = A) ↔ m ∈ ({0, 1/2, 1/3} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_value_set_of_m_l49_4973


namespace NUMINAMATH_CALUDE_domain_transformation_l49_4978

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_plus_one : Set ℝ := Set.Icc (-1) 0

-- Define the domain of f(2x)
def domain_f_double : Set ℝ := Set.Ico 0 (1/2)

-- Theorem statement
theorem domain_transformation (h : ∀ x ∈ domain_f_plus_one, f (x + 1) = f (x + 1)) :
  ∀ x ∈ domain_f_double, f (2 * x) = f (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_domain_transformation_l49_4978


namespace NUMINAMATH_CALUDE_cubic_diophantine_equation_l49_4916

theorem cubic_diophantine_equation :
  ∀ x y z : ℤ, x^3 + 2*y^3 = 4*z^3 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_diophantine_equation_l49_4916


namespace NUMINAMATH_CALUDE_no_cross_sum_2018_l49_4981

theorem no_cross_sum_2018 (n : ℕ) (h : n ∈ Finset.range 4901) : 5 * n ≠ 2018 := by
  sorry

end NUMINAMATH_CALUDE_no_cross_sum_2018_l49_4981


namespace NUMINAMATH_CALUDE_polynomial_division_condition_l49_4990

theorem polynomial_division_condition (a b : ℝ) : 
  (∀ x : ℝ, (x - 1)^2 ∣ (a * x^4 + b * x^3 + 1)) ↔ (a = 3 ∧ b = -4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_condition_l49_4990


namespace NUMINAMATH_CALUDE_smallest_y_for_inequality_l49_4920

theorem smallest_y_for_inequality : ∃ y : ℕ, (∀ z : ℕ, 27^z > 3^24 → y ≤ z) ∧ 27^y > 3^24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_for_inequality_l49_4920


namespace NUMINAMATH_CALUDE_two_real_roots_implies_m_geq_one_l49_4956

theorem two_real_roots_implies_m_geq_one (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x + 5)^2 = m - 1 ∧ (y + 5)^2 = m - 1) →
  m ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_two_real_roots_implies_m_geq_one_l49_4956


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l49_4940

/-- Given an ellipse with eccentricity e and focal length 2c, 
    prove that its standard equation is of the form x²/a² + y²/b² = 1 
    where a and b are the semi-major and semi-minor axes respectively. -/
theorem ellipse_standard_equation 
  (e : ℝ) 
  (c : ℝ) 
  (h_e : e = 2/3) 
  (h_c : 2*c = 16) : 
  ∃ (a b : ℝ), 
    (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ↔ 
      (x^2/144 + y^2/80 = 1 ∨ x^2/80 + y^2/144 = 1)) := by
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l49_4940


namespace NUMINAMATH_CALUDE_range_of_a_l49_4983

theorem range_of_a : 
  (∀ x, 0 < x ∧ x < 1 → ∀ a, (x - a) * (x - (a + 2)) ≤ 0) ∧ 
  (∃ x a, ¬(0 < x ∧ x < 1) ∧ (x - a) * (x - (a + 2)) ≤ 0) →
  ∀ a, a ∈ Set.Icc (-1 : ℝ) 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l49_4983


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l49_4919

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (n_count : ℕ) (h_count : ℕ) (br_count : ℕ) 
  (n_weight : ℝ) (h_weight : ℝ) (br_weight : ℝ) : ℝ :=
  n_count * n_weight + h_count * h_weight + br_count * br_weight

/-- The molecular weight of a compound with 1 N, 4 H, and 1 Br atom is 97.95 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight 1 4 1 14.01 1.01 79.90 = 97.95 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l49_4919


namespace NUMINAMATH_CALUDE_white_balls_fewest_l49_4950

/-- Represents the number of balls of each color -/
structure BallCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- The conditions of the ball counting problem -/
def ballProblemConditions (counts : BallCounts) : Prop :=
  counts.red + counts.blue + counts.white = 108 ∧
  counts.blue = counts.red / 3 ∧
  counts.white = counts.blue / 2

theorem white_balls_fewest (counts : BallCounts) 
  (h : ballProblemConditions counts) : 
  counts.white = 12 ∧ 
  counts.white < counts.blue ∧ 
  counts.white < counts.red :=
sorry

end NUMINAMATH_CALUDE_white_balls_fewest_l49_4950


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l49_4986

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 = 7 → a 5 = 13 → a 7 = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l49_4986


namespace NUMINAMATH_CALUDE_sqrt_5_minus_1_over_2_gt_half_l49_4987

theorem sqrt_5_minus_1_over_2_gt_half : 
  (4 < 5) → (5 < 9) → (Real.sqrt 5 - 1) / 2 > 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_minus_1_over_2_gt_half_l49_4987


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l49_4945

-- Define the rhombus
structure Rhombus where
  d1 : ℝ  -- First diagonal
  d2 : ℝ  -- Second diagonal
  area : ℝ  -- Area of the rhombus

-- Define the theorem
theorem inscribed_circle_radius (r : Rhombus) (h1 : r.d1 = 8) (h2 : r.d2 = 30) (h3 : r.area = 120) :
  let side := Real.sqrt ((r.d1/2)^2 + (r.d2/2)^2)
  let radius := r.area / (2 * side)
  radius = 60 / Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l49_4945


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l49_4927

theorem subtraction_of_decimals : 5.75 - 1.46 = 4.29 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l49_4927


namespace NUMINAMATH_CALUDE_distinguishable_triangles_l49_4905

/-- Represents the number of available colors for small triangles -/
def num_colors : ℕ := 8

/-- Represents the number of small triangles in the inner part of the large triangle -/
def inner_triangles : ℕ := 3

/-- Represents the number of small triangles in the outer part of the large triangle -/
def outer_triangles : ℕ := 3

/-- Represents the total number of small triangles in the large triangle -/
def total_triangles : ℕ := inner_triangles + outer_triangles

/-- Calculates the number of ways to color the inner triangle -/
def inner_colorings : ℕ := 
  num_colors + (num_colors * (num_colors - 1)) + (num_colors.choose inner_triangles * inner_triangles.factorial)

/-- Calculates the number of ways to color the outer triangle -/
def outer_colorings : ℕ := 
  num_colors + (num_colors * (num_colors - 1)) + (num_colors.choose outer_triangles * outer_triangles.factorial)

/-- Theorem stating the total number of distinguishable large triangles -/
theorem distinguishable_triangles : 
  inner_colorings * outer_colorings = 116096 := by sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_l49_4905


namespace NUMINAMATH_CALUDE_mary_juan_income_ratio_l49_4917

/-- Given that Mary's income is 60% more than Tim's income, and Tim's income is 60% less than Juan's income,
    prove that Mary's income is 64% of Juan's income. -/
theorem mary_juan_income_ratio (juan tim mary : ℝ) 
  (h1 : tim = 0.4 * juan)
  (h2 : mary = 1.6 * tim) : 
  mary = 0.64 * juan := by
  sorry

end NUMINAMATH_CALUDE_mary_juan_income_ratio_l49_4917


namespace NUMINAMATH_CALUDE_largest_number_is_541_l49_4994

def digits : List Nat := [1, 4, 5]

def is_valid_number (n : Nat) : Prop :=
  let digits_of_n := n.digits 10
  digits_of_n.length = 3 ∧ digits_of_n.toFinset = digits.toFinset

theorem largest_number_is_541 :
  ∀ n : Nat, is_valid_number n → n ≤ 541 :=
sorry

end NUMINAMATH_CALUDE_largest_number_is_541_l49_4994


namespace NUMINAMATH_CALUDE_totalChargeDifference_l49_4929

/-- Represents the pricing structure for an air conditioner company -/
structure ACCompany where
  price : ℝ
  surchargeRate : ℝ
  installationCharge : ℝ
  warrantyFee : ℝ
  maintenanceFee : ℝ

/-- Calculates the total charge for a company -/
def totalCharge (c : ACCompany) : ℝ :=
  c.price + (c.surchargeRate * c.price) + c.installationCharge + c.warrantyFee + c.maintenanceFee

/-- Company X's pricing information -/
def companyX : ACCompany :=
  { price := 575
  , surchargeRate := 0.04
  , installationCharge := 82.50
  , warrantyFee := 125
  , maintenanceFee := 50 }

/-- Company Y's pricing information -/
def companyY : ACCompany :=
  { price := 530
  , surchargeRate := 0.03
  , installationCharge := 93.00
  , warrantyFee := 150
  , maintenanceFee := 40 }

/-- Theorem stating the difference in total charges between Company X and Company Y -/
theorem totalChargeDifference :
  totalCharge companyX - totalCharge companyY = 26.60 := by
  sorry

end NUMINAMATH_CALUDE_totalChargeDifference_l49_4929


namespace NUMINAMATH_CALUDE_mrs_hilt_marbles_l49_4942

/-- The number of marbles Mrs. Hilt lost -/
def marbles_lost : ℕ := 15

/-- The number of marbles Mrs. Hilt has left -/
def marbles_left : ℕ := 23

/-- The initial number of marbles Mrs. Hilt had -/
def initial_marbles : ℕ := marbles_lost + marbles_left

theorem mrs_hilt_marbles : initial_marbles = 38 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_marbles_l49_4942


namespace NUMINAMATH_CALUDE_amount_after_two_years_l49_4932

/-- Calculates the amount after n years given an initial amount and annual increase rate -/
def amount_after_years (initial_amount : ℝ) (increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + increase_rate) ^ years

/-- Theorem stating that an initial amount of 1600 increasing by 1/8 annually becomes 2025 after 2 years -/
theorem amount_after_two_years :
  let initial_amount : ℝ := 1600
  let increase_rate : ℝ := 1/8
  let years : ℕ := 2
  amount_after_years initial_amount increase_rate years = 2025 := by
  sorry


end NUMINAMATH_CALUDE_amount_after_two_years_l49_4932


namespace NUMINAMATH_CALUDE_prime_pair_divisibility_l49_4921

theorem prime_pair_divisibility (n p : ℕ+) : 
  Nat.Prime p.val ∧ 
  n.val ≤ 2 * p.val ∧ 
  (n.val^(p.val - 1) ∣ (p.val - 1)^n.val + 1) → 
  ((n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
sorry

end NUMINAMATH_CALUDE_prime_pair_divisibility_l49_4921
