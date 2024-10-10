import Mathlib

namespace regular_polygon_perimeter_l3596_359651

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 2 ∧
  side_length = 7 ∧
  exterior_angle = 90 ∧
  (360 : ℝ) / n = exterior_angle →
  n * side_length = 28 := by
  sorry

end regular_polygon_perimeter_l3596_359651


namespace equation_solution_exists_l3596_359656

theorem equation_solution_exists : ∃ (d e f g : ℕ+), 
  (3 : ℝ) * ((7 : ℝ)^(1/3) + (6 : ℝ)^(1/3))^(1/2) = d^(1/3) - e^(1/3) + f^(1/3) + g^(1/3) ∧
  d + e + f + g = 96 := by
  sorry

end equation_solution_exists_l3596_359656


namespace base5_calculation_l3596_359601

/-- Converts a number from base 5 to base 10 --/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Theorem stating that the result of (2434₅ + 132₅) ÷ 21₅ in base 5 is 122₅ --/
theorem base5_calculation : 
  base10ToBase5 ((base5ToBase10 [4, 3, 4, 2] + base5ToBase10 [2, 3, 1]) / base5ToBase10 [1, 2]) = [2, 2, 1] := by
  sorry


end base5_calculation_l3596_359601


namespace complex_fraction_equality_l3596_359673

theorem complex_fraction_equality : (1 - Complex.I) / (1 + Complex.I) = -Complex.I := by
  sorry

end complex_fraction_equality_l3596_359673


namespace line_slope_intercept_sum_l3596_359618

/-- Given a line passing through the points (1,2) and (4,20), 
    prove that the sum of its slope and y-intercept is 8. -/
theorem line_slope_intercept_sum : ∀ (m b : ℝ),
  (∀ x y : ℝ, y = m * x + b ↔ (x = 1 ∧ y = 2) ∨ (x = 4 ∧ y = 20)) →
  m + b = 8 := by
  sorry

end line_slope_intercept_sum_l3596_359618


namespace stock_market_investment_l3596_359670

theorem stock_market_investment
  (initial_investment : ℝ)
  (first_year_increase : ℝ)
  (net_increase : ℝ)
  (h1 : first_year_increase = 0.8)
  (h2 : net_increase = 0.26)
  : ∃ (second_year_decrease : ℝ),
    second_year_decrease = 0.3 ∧
    (1 + first_year_increase) * (1 - second_year_decrease) = 1 + net_increase :=
by sorry

end stock_market_investment_l3596_359670


namespace single_elimination_tournament_matches_l3596_359678

/-- Calculates the number of matches in a single-elimination tournament. -/
def matches_played (num_players : ℕ) : ℕ :=
  num_players - 1

/-- Theorem: In a single-elimination tournament with 512 players,
    511 matches are played to declare the winner. -/
theorem single_elimination_tournament_matches :
  matches_played 512 = 511 := by
  sorry

end single_elimination_tournament_matches_l3596_359678


namespace ellipse_minor_axis_length_l3596_359652

/-- The length of the minor axis of the ellipse 9x^2 + y^2 = 36 is 4 -/
theorem ellipse_minor_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 36}
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (∀ x y : ℝ, (x, y) ∈ ellipse ↔ (x^2 / a^2) + (y^2 / b^2) = 1) ∧
    2 * b = 4 :=
by sorry

end ellipse_minor_axis_length_l3596_359652


namespace velvet_fabric_cost_l3596_359687

/-- Proves that the cost of velvet fabric per yard is $24 -/
theorem velvet_fabric_cost
  (total_spent : ℝ)
  (pattern_cost : ℝ)
  (thread_cost_per_spool : ℝ)
  (num_thread_spools : ℕ)
  (fabric_yards : ℝ)
  (h1 : total_spent = 141)
  (h2 : pattern_cost = 15)
  (h3 : thread_cost_per_spool = 3)
  (h4 : num_thread_spools = 2)
  (h5 : fabric_yards = 5)
  : (total_spent - pattern_cost - thread_cost_per_spool * num_thread_spools) / fabric_yards = 24 := by
  sorry

end velvet_fabric_cost_l3596_359687


namespace girls_left_class_l3596_359605

/-- Represents the number of girls who left the class -/
def girls_who_left (initial_girls : ℕ) (final_girls : ℕ) : ℕ :=
  initial_girls - final_girls

/-- Theorem stating the number of girls who left the class -/
theorem girls_left_class :
  ∀ (initial_girls initial_boys final_girls : ℕ),
    -- Initial ratio of girls to boys is 5:6
    initial_girls * 6 = initial_boys * 5 →
    -- Final ratio of girls to boys is 2:3
    final_girls * 3 = initial_boys * 2 →
    -- There are 120 boys in the class
    initial_boys = 120 →
    -- The number of girls who left is 20
    girls_who_left initial_girls final_girls = 20 := by
  sorry

end girls_left_class_l3596_359605


namespace investment_sum_l3596_359609

/-- 
Given a sum P invested at two different simple interest rates for two years,
prove that P = 12000 if the difference in interest is 720.
-/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 720) → P = 12000 := by
  sorry

end investment_sum_l3596_359609


namespace original_proposition_contrapositive_converse_false_inverse_false_negation_false_l3596_359614

-- Original proposition
theorem original_proposition : ∀ x : ℝ, x = 1 → x^2 = 1 := by sorry

-- Contrapositive
theorem contrapositive : ∀ x : ℝ, x^2 ≠ 1 → x ≠ 1 := by sorry

-- Converse (as a counterexample)
theorem converse_false : ∃ x : ℝ, x^2 = 1 ∧ x ≠ 1 := by sorry

-- Inverse (as a counterexample)
theorem inverse_false : ∃ x : ℝ, x ≠ 1 ∧ x^2 = 1 := by sorry

-- Negation (as false)
theorem negation_false : ¬(∀ x : ℝ, x = 1 → x^2 ≠ 1) := by sorry

end original_proposition_contrapositive_converse_false_inverse_false_negation_false_l3596_359614


namespace blue_marbles_percentage_l3596_359677

/-- Represents the number of marbles Pete has initially -/
def initial_marbles : ℕ := 10

/-- Represents the number of marbles Pete has after trading -/
def final_marbles : ℕ := 15

/-- Represents the number of red marbles Pete keeps after trading -/
def kept_red_marbles : ℕ := 1

/-- Calculates the percentage of blue marbles initially -/
def blue_percentage (blue : ℕ) : ℚ :=
  (blue : ℚ) / (initial_marbles : ℚ) * 100

/-- The main theorem stating the percentage of blue marbles -/
theorem blue_marbles_percentage :
  ∃ (blue red : ℕ),
    blue + red = initial_marbles ∧
    blue + 2 * (red - kept_red_marbles) + kept_red_marbles = final_marbles ∧
    blue_percentage blue = 40 := by
  sorry

end blue_marbles_percentage_l3596_359677


namespace expand_expression_l3596_359649

theorem expand_expression (x : ℝ) : (13 * x + 15) * (2 * x) = 26 * x^2 + 30 * x := by
  sorry

end expand_expression_l3596_359649


namespace sandwich_jam_cost_l3596_359633

/-- Represents the cost of jam used in sandwiches given specific conditions -/
theorem sandwich_jam_cost :
  ∀ (N B J : ℕ),
  N > 1 →
  B > 0 →
  J > 0 →
  N * (4 * B + 5 * J) = 253 →
  (N * J * 5 : ℚ) / 100 = 1.65 :=
by sorry

end sandwich_jam_cost_l3596_359633


namespace four_even_numbers_sum_100_l3596_359668

theorem four_even_numbers_sum_100 :
  ∃ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∃ (k₁ k₂ k₃ k₄ : ℕ), a = 2 * k₁ ∧ b = 2 * k₂ ∧ c = 2 * k₃ ∧ d = 2 * k₄) ∧
    a + b + c + d = 50 ∧
    c < 10 ∧ d < 10 ∧
    6 * a + 15 * b + 26 * c = 500 ∧
    ((a = 31 ∧ b = 14 ∧ c = 4 ∧ d = 1) ∨ (a = 24 ∧ b = 22 ∧ c = 1 ∧ d = 3)) :=
by
  sorry

end four_even_numbers_sum_100_l3596_359668


namespace second_jumper_height_l3596_359628

/-- The height of Ravi's jump in inches -/
def ravi_jump : ℝ := 39

/-- The height of the first next highest jumper in inches -/
def first_jumper : ℝ := 23

/-- The height of the third next highest jumper in inches -/
def third_jumper : ℝ := 28

/-- The ratio of Ravi's jump height to the average of the three next highest jumpers -/
def ravi_ratio : ℝ := 1.5

/-- The height of the second next highest jumper in inches -/
def second_jumper : ℝ := 27

theorem second_jumper_height :
  ravi_jump = ravi_ratio * (first_jumper + second_jumper + third_jumper) / 3 →
  second_jumper = 27 := by
  sorry

end second_jumper_height_l3596_359628


namespace max_men_with_all_attributes_l3596_359659

/-- Represents the population of men in the city with various attributes -/
structure CityPopulation where
  total : ℕ
  married : ℕ
  withTV : ℕ
  withRadio : ℕ
  withAC : ℕ
  withCar : ℕ
  withSmartphone : ℕ

/-- The given population data for the city -/
def cityData : CityPopulation := {
  total := 3000,
  married := 2300,
  withTV := 2100,
  withRadio := 2600,
  withAC := 1800,
  withCar := 2500,
  withSmartphone := 2200
}

/-- Theorem stating that the maximum number of men with all attributes is at most 1800 -/
theorem max_men_with_all_attributes (p : CityPopulation) (h : p = cityData) :
  ∃ n : ℕ, n ≤ 1800 ∧ n ≤ p.married ∧ n ≤ p.withTV ∧ n ≤ p.withRadio ∧
           n ≤ p.withAC ∧ n ≤ p.withCar ∧ n ≤ p.withSmartphone :=
by sorry

end max_men_with_all_attributes_l3596_359659


namespace x_plus_y_values_l3596_359648

theorem x_plus_y_values (x y : ℝ) (hx : -x = 3) (hy : |y| = 5) : 
  x + y = -8 ∨ x + y = 2 := by
sorry

end x_plus_y_values_l3596_359648


namespace inequality_condition_l3596_359603

theorem inequality_condition (x y m : ℝ) : 
  x > 0 → 
  y > 0 → 
  2/x + 1/y = 1 → 
  (∀ x y, x > 0 ∧ y > 0 ∧ 2/x + 1/y = 1 → 2*x + y > m^2 + 8*m) ↔ 
  -9 < m ∧ m < 1 := by
sorry

end inequality_condition_l3596_359603


namespace spectators_with_type_A_l3596_359634

/-- Represents the number of spectators who received type A wristbands for both hands -/
def x : ℕ := sorry

/-- Represents the number of spectators who received type B wristbands for both hands -/
def y : ℕ := sorry

/-- The ratio of spectators with type A to type B wristbands is 3:2 -/
axiom ratio_constraint : 2 * x = 3 * y

/-- The total number of wristbands distributed is 460 -/
axiom total_wristbands : 2 * x + 2 * y = 460

theorem spectators_with_type_A : x = 138 := by sorry

end spectators_with_type_A_l3596_359634


namespace exists_number_with_digit_sum_divisible_by_13_l3596_359639

-- Define a function to calculate the sum of digits of a natural number
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_number_with_digit_sum_divisible_by_13 (n : ℕ) :
  ∃ k : ℕ, k ∈ Finset.range 79 ∧ (sum_of_digits (n + k)) % 13 = 0 := by sorry

end exists_number_with_digit_sum_divisible_by_13_l3596_359639


namespace min_value_sum_reciprocals_l3596_359684

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum : x + y + z = 2) :
  1/x + 1/y + 1/z ≥ 9/2 := by
sorry

end min_value_sum_reciprocals_l3596_359684


namespace combination_sum_identity_l3596_359693

theorem combination_sum_identity (k m n : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ m) (h3 : m ≤ n) :
  (Finset.range (k + 1)).sum (fun i => Nat.choose n i * Nat.choose n (m - i)) = Nat.choose (n + k) m := by
  sorry

end combination_sum_identity_l3596_359693


namespace max_value_on_circle_l3596_359672

/-- Given a complex number z = x + yi where x and y are real numbers,
    and |z - 3| = 1, the maximum value of x^2 + y^2 + 4x + 1 is 33. -/
theorem max_value_on_circle (x y : ℝ) (z : ℂ) (h : z = x + y * Complex.I) 
    (h_circle : Complex.abs (z - 3) = 1) : 
    (∀ a b : ℝ, Complex.abs ((a + b * Complex.I) - 3) = 1 → 
      x^2 + y^2 + 4*x + 1 ≥ a^2 + b^2 + 4*a + 1) ∧
    (∃ u v : ℝ, Complex.abs ((u + v * Complex.I) - 3) = 1 ∧ 
      u^2 + v^2 + 4*u + 1 = 33) :=
by sorry

end max_value_on_circle_l3596_359672


namespace line_circle_intersection_l3596_359615

/-- The line y = kx + 1 intersects the circle (x-2)² + (y-1)² = 4 at points P and Q.
    If the distance between P and Q is greater than or equal to 2√2,
    then k is in the interval [-1, 1]. -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ P Q : ℝ × ℝ, 
    (P.2 = k * P.1 + 1) ∧ 
    (Q.2 = k * Q.1 + 1) ∧
    ((P.1 - 2)^2 + (P.2 - 1)^2 = 4) ∧
    ((Q.1 - 2)^2 + (Q.2 - 1)^2 = 4) ∧
    ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 ≥ 8)) →
  -1 ≤ k ∧ k ≤ 1 :=
sorry

end line_circle_intersection_l3596_359615


namespace extreme_values_of_f_l3596_359604

-- Define the function f(x) = x³ - 3x² - 9x
def f (x : ℝ) := x^3 - 3*x^2 - 9*x

-- Define the open interval (-2, 2)
def I := Set.Ioo (-2 : ℝ) 2

-- State the theorem
theorem extreme_values_of_f :
  (∃ (x : ℝ), x ∈ I ∧ f x = 5) ∧
  (∀ (x : ℝ), x ∈ I → f x ≤ 5) ∧
  (∃ (x : ℝ), x ∈ I ∧ f x = -2) ∧
  (∀ (x : ℝ), x ∈ I → f x ≥ -2) := by
  sorry

end extreme_values_of_f_l3596_359604


namespace cube_points_form_octahedron_l3596_359643

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A cube with edge length a -/
structure Cube (a : ℝ) where
  vertices : Fin 8 → Point3D

/-- An octahedron -/
structure Octahedron where
  vertices : Fin 6 → Point3D

/-- Function to select points on the edges of a cube -/
def selectPointsOnCube (c : Cube a) : Fin 6 → Point3D :=
  sorry

/-- Function to check if points form an octahedron -/
def isOctahedron (points : Fin 6 → Point3D) : Prop :=
  sorry

/-- Theorem stating that it's possible to select 6 points on the edges of a cube
    such that they form the vertices of an octahedron -/
theorem cube_points_form_octahedron (a : ℝ) (h : a > 0) :
  ∃ (c : Cube a), isOctahedron (selectPointsOnCube c) :=
by
  sorry

end cube_points_form_octahedron_l3596_359643


namespace min_value_in_intersection_l3596_359658

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | let (x, y) := p; (y - x) * (y - 18 / (25 * x)) ≥ 0}
def B : Set (ℝ × ℝ) := {p | let (x, y) := p; (x - 1)^2 + (y - 1)^2 ≤ 1}

-- Define the objective function
def f (p : ℝ × ℝ) : ℝ := let (x, y) := p; 2 * x - y

-- Theorem statement
theorem min_value_in_intersection :
  (∀ p ∈ A ∩ B, f p ≥ -1) ∧ (∃ p ∈ A ∩ B, f p = -1) :=
sorry

end min_value_in_intersection_l3596_359658


namespace rect_to_spherical_conversion_l3596_359607

/-- Conversion from rectangular to spherical coordinates -/
theorem rect_to_spherical_conversion :
  let x : ℝ := 2 * Real.sqrt 3
  let y : ℝ := 6
  let z : ℝ := -4
  let ρ : ℝ := 8
  let θ : ℝ := π / 3
  let φ : ℝ := 2 * π / 3
  (ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ 0 ≤ φ ∧ φ ≤ π) →
  (x = ρ * Real.sin φ * Real.cos θ ∧
   y = ρ * Real.sin φ * Real.sin θ ∧
   z = ρ * Real.cos φ) :=
by sorry

end rect_to_spherical_conversion_l3596_359607


namespace tan_45_degrees_l3596_359689

theorem tan_45_degrees : Real.tan (45 * π / 180) = 1 := by
  sorry

end tan_45_degrees_l3596_359689


namespace honey_bee_count_l3596_359662

/-- The number of honey bees that produce a given amount of honey in a fixed time period. -/
def num_honey_bees (total_honey : ℕ) (honey_per_bee : ℕ) : ℕ :=
  total_honey / honey_per_bee

/-- Theorem stating that the number of honey bees is 30 given the problem conditions. -/
theorem honey_bee_count : num_honey_bees 30 1 = 30 := by
  sorry

end honey_bee_count_l3596_359662


namespace odd_function_properties_l3596_359617

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y
def has_min_value_on (f : ℝ → ℝ) (v : ℝ) (a b : ℝ) : Prop := ∀ x, a ≤ x → x ≤ b → v ≤ f x
def has_max_value_on (f : ℝ → ℝ) (v : ℝ) (a b : ℝ) : Prop := ∀ x, a ≤ x → x ≤ b → f x ≤ v

-- State the theorem
theorem odd_function_properties (f : ℝ → ℝ) :
  is_odd f →
  is_increasing_on f 1 3 →
  has_min_value_on f 7 1 3 →
  is_increasing_on f (-3) (-1) ∧ has_max_value_on f (-7) (-3) (-1) :=
by sorry

end odd_function_properties_l3596_359617


namespace compare_quadratic_expressions_compare_fractions_l3596_359638

-- Problem 1
theorem compare_quadratic_expressions (x : ℝ) : 2 * x^2 - x > x^2 + x - 2 := by
  sorry

-- Problem 2
theorem compare_fractions (c a b : ℝ) (hc : c > a) (ha : a > b) (hb : b > 0) :
  a / (c - a) > b / (c - b) := by
  sorry

end compare_quadratic_expressions_compare_fractions_l3596_359638


namespace A_maximized_at_19_l3596_359644

def factorial (n : ℕ) : ℕ := Nat.factorial n

def A (n : ℕ+) : ℚ := (20^n.val + 11^n.val) / factorial n.val

theorem A_maximized_at_19 : ∀ n : ℕ+, A n ≤ A 19 := by sorry

end A_maximized_at_19_l3596_359644


namespace conic_is_ellipse_l3596_359624

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

-- Define the foci
def focus1 : ℝ × ℝ := (0, 2)
def focus2 : ℝ × ℝ := (6, -4)

-- Theorem stating that the equation describes an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧
  ∀ (x y : ℝ), conic_equation x y ↔
    (x - (focus1.1 + focus2.1) / 2)^2 / a^2 +
    (y - (focus1.2 + focus2.2) / 2)^2 / b^2 = 1 :=
sorry

end conic_is_ellipse_l3596_359624


namespace gcf_of_75_and_100_l3596_359681

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end gcf_of_75_and_100_l3596_359681


namespace no_prime_arithmetic_progression_l3596_359680

def arithmeticProgression (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem no_prime_arithmetic_progression :
  ∀ (a₁ d : ℕ), Prime a₁ → d ≠ 0 →
  ∃ (n : ℕ), ¬ Prime (arithmeticProgression a₁ d n) :=
by sorry

end no_prime_arithmetic_progression_l3596_359680


namespace complex_subtraction_imaginary_part_l3596_359642

theorem complex_subtraction_imaginary_part : 
  (Complex.im ((2 + Complex.I) / (1 - Complex.I) - (2 - Complex.I) / (1 + Complex.I)) = 3) := by
  sorry

end complex_subtraction_imaginary_part_l3596_359642


namespace integral_exp_abs_l3596_359694

theorem integral_exp_abs : ∫ x in (-2)..4, Real.exp (abs x) = Real.exp 4 + Real.exp (-2) - 2 := by sorry

end integral_exp_abs_l3596_359694


namespace pizza_dough_milk_calculation_l3596_359696

/-- Calculates the amount of milk needed for pizza dough given the flour quantity and milk-to-flour ratio -/
def milk_needed (flour_quantity : ℕ) (milk_per_flour_unit : ℚ) : ℚ :=
  (flour_quantity : ℚ) * milk_per_flour_unit

/-- Proves the correct amount of milk for one and two batches of pizza dough -/
theorem pizza_dough_milk_calculation :
  let flour_quantity : ℕ := 1200
  let milk_per_flour_unit : ℚ := 60 / 300
  let milk_for_one_batch : ℚ := milk_needed flour_quantity milk_per_flour_unit
  let milk_for_two_batches : ℚ := 2 * milk_for_one_batch
  milk_for_one_batch = 240 ∧ milk_for_two_batches = 480 := by
  sorry

#check pizza_dough_milk_calculation

end pizza_dough_milk_calculation_l3596_359696


namespace cubs_cardinals_home_run_difference_l3596_359606

/-- Represents a baseball team -/
inductive Team
| Cubs
| Cardinals

/-- Represents an inning in a baseball game -/
inductive Inning
| Second
| Third
| Fifth
| Eighth

/-- Records the number of home runs scored by a team in an inning -/
def home_runs_in_inning (team : Team) (inning : Inning) : ℕ :=
  match team, inning with
  | Team.Cubs, Inning.Third => 2
  | Team.Cubs, Inning.Fifth => 1
  | Team.Cubs, Inning.Eighth => 2
  | Team.Cardinals, Inning.Second => 1
  | Team.Cardinals, Inning.Fifth => 1
  | _, _ => 0

/-- Calculates the total number of home runs for a team -/
def total_home_runs (team : Team) : ℕ :=
  (home_runs_in_inning team Inning.Second) +
  (home_runs_in_inning team Inning.Third) +
  (home_runs_in_inning team Inning.Fifth) +
  (home_runs_in_inning team Inning.Eighth)

/-- The main theorem stating the difference in home runs between the Cubs and Cardinals -/
theorem cubs_cardinals_home_run_difference :
  total_home_runs Team.Cubs - total_home_runs Team.Cardinals = 3 := by
  sorry


end cubs_cardinals_home_run_difference_l3596_359606


namespace hostel_cost_calculation_hostel_cost_23_days_l3596_359632

/-- Cost calculation for student youth hostel stay --/
theorem hostel_cost_calculation (first_week_rate : ℝ) (additional_week_rate : ℝ) (total_days : ℕ) : 
  first_week_rate = 18 →
  additional_week_rate = 11 →
  total_days = 23 →
  (7 * first_week_rate + (total_days - 7) * additional_week_rate : ℝ) = 302 := by
  sorry

/-- Main theorem: Cost of 23-day stay is $302.00 --/
theorem hostel_cost_23_days : 
  ∃ (first_week_rate additional_week_rate : ℝ),
    first_week_rate = 18 ∧
    additional_week_rate = 11 ∧
    (7 * first_week_rate + 16 * additional_week_rate : ℝ) = 302 := by
  sorry

end hostel_cost_calculation_hostel_cost_23_days_l3596_359632


namespace fraction_to_decimal_l3596_359661

theorem fraction_to_decimal : (45 : ℚ) / (2^2 * 5^3) = (9 : ℚ) / 100 := by
  sorry

end fraction_to_decimal_l3596_359661


namespace f_sum_equals_four_l3596_359635

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_sum_equals_four (f : ℝ → ℝ) 
  (h_even : isEven f)
  (h_period : ∀ x, f (x + 4) = f x + f 2)
  (h_f_one : f 1 = 4) :
  f 3 + f 10 = 4 := by sorry

end f_sum_equals_four_l3596_359635


namespace factor_x12_minus_729_l3596_359631

theorem factor_x12_minus_729 (x : ℝ) :
  x^12 - 729 = (x^3 + 3) * (x - Real.rpow 3 (1/3)) * 
               (x^2 + x * Real.rpow 3 (1/3) + Real.rpow 3 (2/3)) * 
               (x^12 + 9*x^6 + 81) :=
by sorry

end factor_x12_minus_729_l3596_359631


namespace series_sum_equals_half_l3596_359621

/-- The series defined by the nth term: 2^n / (3^(2^n) + 1) -/
def series (n : ℕ) : ℚ := (2^n : ℚ) / ((3^(2^n) : ℚ) + 1)

/-- The sum of the series from 0 to infinity -/
noncomputable def seriesSum : ℚ := ∑' n, series n

/-- Theorem stating that the sum of the series equals 1/2 -/
theorem series_sum_equals_half : seriesSum = 1/2 := by sorry

end series_sum_equals_half_l3596_359621


namespace arctans_and_arcsins_sum_l3596_359645

theorem arctans_and_arcsins_sum : 
  Real.arctan (1/3) + Real.arctan (1/5) + Real.arcsin (1/Real.sqrt 50) + Real.arcsin (1/Real.sqrt 65) = π/4 := by
  sorry

end arctans_and_arcsins_sum_l3596_359645


namespace total_stars_l3596_359682

theorem total_stars (num_students : ℕ) (stars_per_student : ℕ) : 
  num_students = 124 → stars_per_student = 3 → num_students * stars_per_student = 372 := by
  sorry

end total_stars_l3596_359682


namespace counterexample_exists_l3596_359646

theorem counterexample_exists : ∃ n : ℕ, 
  n > 1 ∧ ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 2)) ∧ ¬(Nat.Prime (n - 3)) :=
by sorry

end counterexample_exists_l3596_359646


namespace problem_statement_l3596_359625

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x/y + y/x = 8) :
  (x + y)/(x - y) = Real.sqrt (5/3) := by
  sorry

end problem_statement_l3596_359625


namespace computer_operations_l3596_359654

/-- Calculates the total number of operations a computer can perform given its operation rate and runtime. -/
theorem computer_operations (rate : ℝ) (time : ℝ) (h1 : rate = 4 * 10^8) (h2 : time = 6 * 10^5) :
  rate * time = 2.4 * 10^14 := by
  sorry

#check computer_operations

end computer_operations_l3596_359654


namespace mollys_age_l3596_359699

/-- Given that the ratio of Sandy's age to Molly's age is 4:3, and Sandy will be 66 years old in 6 years, prove that Molly's current age is 45 years. -/
theorem mollys_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / molly_age = 4 / 3 →
  sandy_age + 6 = 66 →
  molly_age = 45 := by
sorry


end mollys_age_l3596_359699


namespace imaginary_part_of_z_l3596_359622

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2) : 
  z.im = -1 := by sorry

end imaginary_part_of_z_l3596_359622


namespace sequence_properties_l3596_359637

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2 * a n + 1

def c (n : ℕ) : ℚ := 1 / ((2 * n + 1) * (2 * n + 3))

def T (n : ℕ) : ℚ := n / (6 * n + 9)

theorem sequence_properties :
  (∀ n : ℕ, a n + 1 = 2^(n + 1)) ∧
  (∀ n : ℕ, a n = 2^(n + 1) - 1) ∧
  (∀ n : ℕ, T n = n / (6 * n + 9)) ∧
  (∀ n : ℕ+, T n > 1 / a 5) ∧
  (∀ m : ℕ+, m < 5 → ∃ n : ℕ+, T n ≤ 1 / a m) :=
by sorry

end sequence_properties_l3596_359637


namespace perpendicular_lines_l3596_359655

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines (a : ℝ) :
  perpendicular a (1/a) → a = 0 := by
  sorry

end perpendicular_lines_l3596_359655


namespace cookies_difference_l3596_359620

/-- Given Paco's cookie situation, prove the difference between bought and eaten cookies --/
theorem cookies_difference (initial : ℕ) (eaten : ℕ) (bought : ℕ)
  (h1 : initial = 13)
  (h2 : eaten = 2)
  (h3 : bought = 36) :
  bought - eaten = 34 := by
  sorry

end cookies_difference_l3596_359620


namespace fraction_equality_l3596_359647

theorem fraction_equality : (18 : ℚ) / (9 * 47 * 5) = 2 / 235 := by
  sorry

end fraction_equality_l3596_359647


namespace shepherds_pie_pieces_l3596_359640

theorem shepherds_pie_pieces (customers_shepherds : ℕ) (customers_chicken : ℕ) (chicken_pieces : ℕ) (total_pies : ℕ) :
  customers_shepherds = 52 →
  customers_chicken = 80 →
  chicken_pieces = 5 →
  total_pies = 29 →
  ∃ (shepherds_pieces : ℕ), 
    shepherds_pieces = 4 ∧
    (customers_shepherds / shepherds_pieces : ℚ) + (customers_chicken / chicken_pieces : ℚ) = total_pies :=
by sorry

end shepherds_pie_pieces_l3596_359640


namespace circle_M_properties_l3596_359602

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x + 3*y - 2 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 0

-- Theorem statement
theorem circle_M_properties :
  (∃ (c_x c_y r : ℝ), ∀ (x y : ℝ), circle_M x y ↔ (x - c_x)^2 + (y - c_y)^2 = r^2) ∧
  (∀ (x y : ℝ), circle_M x y → circle_M (2*(2-x) - x) (2*(2-y) - y)) ∧
  (∃ (t_x t_y : ℝ), circle_M t_x t_y ∧ tangent_line t_x t_y ∧
    ∀ (x y : ℝ), circle_M x y → (x - t_x)^2 + (y - t_y)^2 ≥ 0) :=
by sorry

end circle_M_properties_l3596_359602


namespace eventually_periodic_sequence_l3596_359660

def RecursiveSequence (p : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 4 → p n * (p (n-1) * p (n-2) + p (n-3) + p (n-4)) = 
    p (n-1) + p (n-2) + p (n-3) * p (n-4)

theorem eventually_periodic_sequence 
  (p : ℕ → ℤ) 
  (h_bounded : ∃ M : ℤ, ∀ n : ℕ, |p n| ≤ M) 
  (h_recursive : RecursiveSequence p) : 
  ∃ (T : ℕ) (N : ℕ), T > 0 ∧ ∀ n : ℕ, n ≥ N → p n = p (n + T) := by
sorry

end eventually_periodic_sequence_l3596_359660


namespace selection_probability_l3596_359613

/-- Represents the probability of a student being chosen in the selection process -/
def probability_of_selection (total_students : ℕ) (eliminated : ℕ) (selected : ℕ) : ℚ :=
  (total_students - eliminated : ℚ) / total_students * selected / (total_students - eliminated)

/-- Theorem stating that the probability of each student being chosen is 4/43 -/
theorem selection_probability :
  let total_students : ℕ := 86
  let eliminated : ℕ := 6
  let selected : ℕ := 8
  probability_of_selection total_students eliminated selected = 4 / 43 := by
sorry

end selection_probability_l3596_359613


namespace twelve_non_congruent_triangles_l3596_359626

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℤ)
  (y : ℤ)

/-- The set of points on the grid -/
def grid_points : List Point := [
  ⟨0, 0⟩, ⟨1, 0⟩, ⟨2, 0⟩, ⟨3, 0⟩,
  ⟨0, 1⟩, ⟨1, 1⟩, ⟨2, 1⟩, ⟨3, 1⟩
]

/-- Determines if two triangles are congruent -/
def are_congruent (t1 t2 : Point × Point × Point) : Prop := sorry

/-- Counts the number of non-congruent triangles -/
def count_non_congruent_triangles (points : List Point) : ℕ := sorry

/-- The main theorem stating that there are 12 non-congruent triangles -/
theorem twelve_non_congruent_triangles :
  count_non_congruent_triangles grid_points = 12 := by sorry

end twelve_non_congruent_triangles_l3596_359626


namespace sufficient_but_not_necessary_l3596_359675

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x ≥ -2) ∧
  (∃ x : ℝ, x ≥ -2 ∧ ¬(-1 ≤ x ∧ x ≤ 1)) := by
  sorry

end sufficient_but_not_necessary_l3596_359675


namespace steak_eaten_l3596_359674

theorem steak_eaten (original_weight : ℝ) (burn_ratio : ℝ) (eat_ratio : ℝ) : 
  original_weight = 30 ∧ 
  burn_ratio = 0.5 ∧ 
  eat_ratio = 0.8 → 
  original_weight * (1 - burn_ratio) * eat_ratio = 12 := by
sorry

end steak_eaten_l3596_359674


namespace quarter_circle_perimeter_l3596_359611

/-- The perimeter of a region defined by quarter circles on the corners of a rectangle --/
theorem quarter_circle_perimeter (π : ℝ) (h : π > 0) : 
  let shorter_side : ℝ := 2 / π
  let longer_side : ℝ := 4 / π
  let quarter_circle_perimeter : ℝ := π * shorter_side / 2
  4 * quarter_circle_perimeter = 4 := by
  sorry

end quarter_circle_perimeter_l3596_359611


namespace cyclic_sum_inequality_l3596_359691

theorem cyclic_sum_inequality (a b c : ℝ) :
  |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| ≤ 
  (9 * Real.sqrt 2 / 32) * (a^2 + b^2 + c^2)^2 := by
sorry

end cyclic_sum_inequality_l3596_359691


namespace shawn_pebbles_count_l3596_359666

/-- The number of pebbles Shawn collected initially -/
def total_pebbles : ℕ := 40

/-- The number of red pebbles -/
def red_pebbles : ℕ := 9

/-- The number of blue pebbles -/
def blue_pebbles : ℕ := 13

/-- The difference between blue and yellow pebbles -/
def blue_yellow_diff : ℕ := 7

/-- The number of groups the remaining pebbles are divided into -/
def remaining_groups : ℕ := 3

theorem shawn_pebbles_count :
  total_pebbles = red_pebbles + blue_pebbles + remaining_groups * ((blue_pebbles - blue_yellow_diff)) :=
by sorry

end shawn_pebbles_count_l3596_359666


namespace company_age_distribution_l3596_359676

def Department : Type := List Nat

def mode (dept : Department) : Nat :=
  sorry

def median (dept : Department) : Nat :=
  sorry

def average (dept : Department) : Real :=
  sorry

theorem company_age_distribution 
  (dept_A dept_B : Department) 
  (h_A : dept_A = [17, 22, 23, 24, 24, 25, 26, 32, 32, 32])
  (h_B : dept_B = [18, 20, 21, 24, 24, 28, 28, 30, 32, 50]) :
  (mode dept_A = 32) ∧ 
  (median dept_B = 26) ∧ 
  (average dept_A < average dept_B) :=
sorry

end company_age_distribution_l3596_359676


namespace total_crayons_l3596_359665

/-- Given that each child has 5 crayons and there are 10 children, 
    prove that the total number of crayons is 50. -/
theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
  (h1 : crayons_per_child = 5) 
  (h2 : num_children = 10) : 
  crayons_per_child * num_children = 50 := by
  sorry

end total_crayons_l3596_359665


namespace check_amount_problem_l3596_359619

theorem check_amount_problem :
  ∃ (x y : ℕ),
    10 ≤ x ∧ x ≤ 99 ∧
    10 ≤ y ∧ y ≤ 99 ∧
    100 * y + x - (100 * x + y) = 1782 ∧
    y = 2 * x :=
by sorry

end check_amount_problem_l3596_359619


namespace fraction_to_decimal_l3596_359692

theorem fraction_to_decimal : 
  ∃ (n : ℕ), (58 : ℚ) / 160 = (3625 : ℚ) / (10^n) := by
  sorry

end fraction_to_decimal_l3596_359692


namespace value_of_E_l3596_359688

/-- Given random integer values for letters of the alphabet, prove that E = 25 -/
theorem value_of_E (Z Q U I T E : ℤ) : Z = 15 → Q + U + I + Z = 60 → Q + U + I + E + T = 75 → Q + U + I + T = 50 → E = 25 := by
  sorry

end value_of_E_l3596_359688


namespace equal_intercepts_condition_l3596_359685

/-- A line with equation ax + y - 2 + a = 0 has equal intercepts on both coordinate axes if and only if a = 2 or a = 1 -/
theorem equal_intercepts_condition (a : ℝ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ 
    (∀ (x y : ℝ), a * x + y - 2 + a = 0 ↔ (x = k ∧ y = 0) ∨ (x = 0 ∧ y = k))) ↔ 
  (a = 2 ∨ a = 1) :=
sorry

end equal_intercepts_condition_l3596_359685


namespace cuboid_diagonal_l3596_359686

/-- 
Given a cuboid with edge lengths a, b, and c, where:
- ab = √2
- bc = √3
- ca = √6
The length of the diagonal is √6.
-/
theorem cuboid_diagonal (a b c : ℝ) 
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : c * a = Real.sqrt 6) : 
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 6 := by
  sorry

end cuboid_diagonal_l3596_359686


namespace imaginary_part_of_complex_product_l3596_359636

theorem imaginary_part_of_complex_product : Complex.im ((3 - 2 * Complex.I) * (1 + Complex.I)) = 1 := by
  sorry

end imaginary_part_of_complex_product_l3596_359636


namespace gcf_of_2836_and_8965_l3596_359641

theorem gcf_of_2836_and_8965 : Nat.gcd 2836 8965 = 1 := by
  sorry

end gcf_of_2836_and_8965_l3596_359641


namespace bills_equal_at_122_minutes_l3596_359679

/-- United Telephone pricing structure -/
def united_base_rate : ℝ := 8.00
def united_per_minute : ℝ := 0.25
def united_tax_rate : ℝ := 0.10
def united_regulatory_fee : ℝ := 1.00

/-- Atlantic Call pricing structure -/
def atlantic_base_rate : ℝ := 12.00
def atlantic_per_minute : ℝ := 0.20
def atlantic_tax_rate : ℝ := 0.15
def atlantic_compatibility_fee : ℝ := 1.50

/-- Calculate the bill for United Telephone -/
def united_bill (minutes : ℝ) : ℝ :=
  let subtotal := united_base_rate + united_per_minute * minutes
  subtotal + united_tax_rate * subtotal + united_regulatory_fee

/-- Calculate the bill for Atlantic Call -/
def atlantic_bill (minutes : ℝ) : ℝ :=
  let subtotal := atlantic_base_rate + atlantic_per_minute * minutes
  subtotal + atlantic_tax_rate * subtotal + atlantic_compatibility_fee

/-- Theorem stating that the bills are equal at 122 minutes -/
theorem bills_equal_at_122_minutes :
  united_bill 122 = atlantic_bill 122 := by sorry

end bills_equal_at_122_minutes_l3596_359679


namespace normal_block_volume_l3596_359600

-- Define the volume of a normal block
def normal_volume (w d l : ℝ) : ℝ := w * d * l

-- Define the volume of a large block
def large_volume (w d l : ℝ) : ℝ := (2*w) * (2*d) * (2*l)

-- Theorem statement
theorem normal_block_volume :
  ∀ w d l : ℝ, w > 0 → d > 0 → l > 0 →
  large_volume w d l = 32 →
  normal_volume w d l = 4 := by
  sorry

end normal_block_volume_l3596_359600


namespace triangle_problem_l3596_359627

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.b * Real.cos t.A + t.a * Real.cos t.B = -2 * t.c * Real.cos t.C)
  (h2 : t.b = 2 * t.a)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3) :
  t.C = 2 * Real.pi / 3 ∧ t.c = 2 * Real.sqrt 7 := by
  sorry


end triangle_problem_l3596_359627


namespace algebraic_expression_value_l3596_359671

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 3) : 1 + a - b = 4 := by
  sorry

end algebraic_expression_value_l3596_359671


namespace quadratic_expression_values_l3596_359695

theorem quadratic_expression_values (m n : ℤ) 
  (h1 : |m| = 3) 
  (h2 : |n| = 2) 
  (h3 : m < n) : 
  m^2 + m*n + n^2 = 7 ∨ m^2 + m*n + n^2 = 19 := by
sorry

end quadratic_expression_values_l3596_359695


namespace nh4oh_remaining_is_zero_l3596_359608

-- Define the molecules and their initial quantities
def NH4Cl : ℕ := 2
def NaOH : ℕ := 2
def H2SO4 : ℕ := 3
def KOH : ℕ := 4

-- Define the reactions
def reaction1 (nh4cl naoh : ℕ) : ℕ := min nh4cl naoh
def reaction2 (nh4cl h2so4 : ℕ) : ℕ := min (nh4cl / 2) h2so4 * 2
def reaction3 (naoh h2so4 : ℕ) : ℕ := min (naoh / 2) h2so4 * 2
def reaction4 (koh h2so4 : ℕ) : ℕ := min koh h2so4
def reaction5 (nh4oh koh : ℕ) : ℕ := min nh4oh koh

-- Theorem statement
theorem nh4oh_remaining_is_zero :
  let nh4oh_formed := reaction1 NH4Cl NaOH
  let h2so4_remaining := H2SO4 - reaction2 NH4Cl H2SO4
  let koh_remaining := KOH - reaction4 KOH h2so4_remaining
  nh4oh_formed - reaction5 nh4oh_formed koh_remaining = 0 := by sorry

end nh4oh_remaining_is_zero_l3596_359608


namespace placemat_length_l3596_359616

theorem placemat_length (R : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) : 
  R = 5 → n = 8 → w = 1 → 
  y = (R^2 - (w/2)^2).sqrt - R * (2 - (2:ℝ).sqrt).sqrt / 2 := by
  sorry

end placemat_length_l3596_359616


namespace sara_bought_fifteen_cards_l3596_359650

/-- Calculates the number of baseball cards Sara bought from Sally -/
def cards_bought_by_sara (initial_cards torn_cards remaining_cards : ℕ) : ℕ :=
  initial_cards - torn_cards - remaining_cards

/-- Theorem stating that Sara bought 15 cards from Sally -/
theorem sara_bought_fifteen_cards :
  let initial_cards := 39
  let torn_cards := 9
  let remaining_cards := 15
  cards_bought_by_sara initial_cards torn_cards remaining_cards = 15 := by
  sorry

end sara_bought_fifteen_cards_l3596_359650


namespace area_between_parallel_chords_l3596_359690

/-- The area between two parallel chords in a circle -/
theorem area_between_parallel_chords (R : ℝ) (h : R > 0) :
  let circle_area := π * R^2
  let segment_60 := circle_area / 6 - R^2 * Real.sqrt 3 / 4
  let segment_120 := circle_area / 3 - R^2 * Real.sqrt 3 / 4
  circle_area - segment_60 - segment_120 = R^2 * (π + Real.sqrt 3) / 2 := by
sorry

end area_between_parallel_chords_l3596_359690


namespace convex_quadrilateral_probability_l3596_359663

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The probability of 4 randomly selected chords from 8 points on a circle forming a convex quadrilateral -/
theorem convex_quadrilateral_probability :
  (n.choose k : ℚ) / (total_chords.choose k : ℚ) = 2 / 585 :=
sorry

end convex_quadrilateral_probability_l3596_359663


namespace alissa_presents_count_l3596_359669

/-- The number of presents Ethan has -/
def ethan_presents : ℝ := 31.0

/-- The difference between Ethan's and Alissa's presents -/
def difference : ℝ := 22.0

/-- The number of presents Alissa has -/
def alissa_presents : ℝ := ethan_presents - difference

theorem alissa_presents_count : alissa_presents = 9.0 := by sorry

end alissa_presents_count_l3596_359669


namespace tree_planting_event_l3596_359657

theorem tree_planting_event (boys : ℕ) (girls : ℕ) 
  (h1 : boys = 600)
  (h2 : girls > boys)
  (h3 : (boys + girls) * 60 / 100 = 960) :
  girls - boys = 400 := by
sorry

end tree_planting_event_l3596_359657


namespace message_clearing_time_l3596_359683

/-- The number of days needed to clear all unread messages -/
def days_to_clear_messages (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) : ℕ :=
  (initial_messages + new_per_day - 1) / (read_per_day - new_per_day)

/-- Theorem stating that it takes 88 days to clear messages under given conditions -/
theorem message_clearing_time : days_to_clear_messages 350 22 18 = 88 := by
  sorry

end message_clearing_time_l3596_359683


namespace unique_solution_system_l3596_359653

theorem unique_solution_system (x y : ℝ) : 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - 2*y = (x - 2) + (2*y - 2)) → 
  x = 6 ∧ y = 1 := by
sorry

end unique_solution_system_l3596_359653


namespace particle_position_after_2020_minutes_l3596_359629

/-- Represents the position of a particle -/
structure Position :=
  (x : ℤ)
  (y : ℤ)

/-- Calculates the position of the particle after a given number of minutes -/
def particle_position (minutes : ℕ) : Position :=
  sorry

/-- The movement pattern of the particle as described in the problem -/
axiom movement_pattern : 
  (∀ m : ℕ, m > 1 → 
    ∃ n : ℕ, 
      (particle_position (m - 1)).x < (particle_position m).x ∧ 
      (particle_position (m - 1)).y < (particle_position m).y) ∧
  (∀ m : ℕ, m > 2 → 
    ∃ n : ℕ, 
      (particle_position (m - 1)).x > (particle_position m).x ∧ 
      (particle_position (m - 1)).y > (particle_position m).y)

/-- The particle starts at the origin -/
axiom start_at_origin : particle_position 0 = ⟨0, 0⟩

/-- After one minute, the particle is at (1,1) -/
axiom first_minute : particle_position 1 = ⟨1, 1⟩

/-- The theorem to be proved -/
theorem particle_position_after_2020_minutes : 
  particle_position 2020 = ⟨30, 40⟩ := by
  sorry

end particle_position_after_2020_minutes_l3596_359629


namespace fifth_roots_sum_l3596_359623

theorem fifth_roots_sum (x y : ℂ) : 
  x = Complex.exp (2 * Real.pi * Complex.I / 5) →
  y = Complex.exp (-2 * Real.pi * Complex.I / 5) →
  x^5 = 1 →
  y^5 = 1 →
  x^5 + y^5 = 2 :=
by sorry

end fifth_roots_sum_l3596_359623


namespace right_triangle_area_perimeter_l3596_359697

theorem right_triangle_area_perimeter : 
  ∀ (a b c : ℝ),
  a = 5 →
  c = 13 →
  a^2 + b^2 = c^2 →
  (1/2 * a * b = 30 ∧ a + b + c = 30) :=
λ a b c h1 h2 h3 =>
  sorry

#check right_triangle_area_perimeter

end right_triangle_area_perimeter_l3596_359697


namespace cone_cross_section_area_l3596_359664

/-- Given a cone with surface area 36π and sector central angle 2π/3 when unfolded,
    the area of the cross-section along its axis is 18√2. -/
theorem cone_cross_section_area (R a h : ℝ) : 
  R > 0 ∧ a > 0 ∧ h > 0 →
  π * R^2 + (2/3) * π * a^2 = 36 * π →
  (2/3) * 2 * π * R = 2 * π * a →
  a^2 = h^2 + R^2 →
  R * h = 18 * Real.sqrt 2 := by
sorry

end cone_cross_section_area_l3596_359664


namespace sufficient_not_necessary_l3596_359612

theorem sufficient_not_necessary (a b : ℝ) :
  (a > b ∧ b > 0) → (1 / a < 1 / b) ∧
  ¬ ((1 / a < 1 / b) → (a > b ∧ b > 0)) :=
by sorry

end sufficient_not_necessary_l3596_359612


namespace repeat_perfect_square_exists_l3596_359630

theorem repeat_perfect_square_exists : ∃ (n : ℕ+) (k : ℕ), 
  (n : ℤ) * (10^k + 1) = (m : ℤ)^2 ∧ 
  10^k ≤ n ∧ n < 10^(k+1) :=
sorry

end repeat_perfect_square_exists_l3596_359630


namespace unique_volume_constraint_l3596_359610

def box_volume (x : ℕ) : ℕ := (x + 5) * (x - 5) * (x^2 + 5*x)

theorem unique_volume_constraint : ∃! x : ℕ, x > 5 ∧ box_volume x < 1000 := by
  sorry

end unique_volume_constraint_l3596_359610


namespace complex_pure_imaginary_l3596_359667

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Given that 2 - a/(2-i) is a pure imaginary number, prove that a = 5 -/
theorem complex_pure_imaginary (a : ℝ) : 
  (∃ b : ℝ, 2 - a / (2 - i) = b * i) → a = 5 := by
  sorry

end complex_pure_imaginary_l3596_359667


namespace parabola_equation_l3596_359698

/-- A parabola with directrix x = 1/2 has the standard equation y^2 = -2x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ x = 1/2) →  -- directrix is x = 1/2
  (∃! f : ℝ × ℝ → Prop, ∀ x y, f (x, y) ↔ y^2 = -2*x) := by sorry

end parabola_equation_l3596_359698
