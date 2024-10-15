import Mathlib

namespace NUMINAMATH_CALUDE_tournament_rankings_l1649_164909

/-- Represents a team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- Represents a match between two teams -/
structure Match :=
(team1 : Team)
(team2 : Team)

/-- Represents the tournament structure -/
structure Tournament :=
(saturday_matches : List Match)
(sunday_winners_round_robin : List Team)
(sunday_losers_round_robin : List Team)

/-- Represents a ranking of teams -/
def Ranking := List Team

/-- Function to calculate the number of possible rankings -/
def number_of_rankings (t : Tournament) : Nat :=
  6 * 6

/-- The main theorem stating the number of possible ranking sequences -/
theorem tournament_rankings (t : Tournament) :
  number_of_rankings t = 36 :=
sorry

end NUMINAMATH_CALUDE_tournament_rankings_l1649_164909


namespace NUMINAMATH_CALUDE_simplify_fraction_l1649_164972

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1649_164972


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l1649_164914

theorem weight_loss_challenge (initial_weight : ℝ) (clothes_weight_percentage : ℝ) 
  (h1 : clothes_weight_percentage > 0)
  (h2 : initial_weight > 0) : 
  (0.90 * initial_weight + clothes_weight_percentage * 0.90 * initial_weight) / initial_weight = 0.918 → 
  clothes_weight_percentage = 0.02 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l1649_164914


namespace NUMINAMATH_CALUDE_smallest_base_for_101_l1649_164969

/-- A number n can be expressed in base b using only two digits if b ≤ n < b^2 -/
def expressibleInTwoDigits (n : ℕ) (b : ℕ) : Prop :=
  b ≤ n ∧ n < b^2

/-- The smallest whole number b such that 101 can be expressed in base b using only two digits -/
def smallestBase : ℕ := 10

theorem smallest_base_for_101 :
  (∀ b : ℕ, b < smallestBase → ¬expressibleInTwoDigits 101 b) ∧
  expressibleInTwoDigits 101 smallestBase := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_101_l1649_164969


namespace NUMINAMATH_CALUDE_fixed_point_on_circle_l1649_164990

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the directrix
def directrix : ℝ := -1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the circle
def circle_tangent_to_directrix (m : PointOnParabola) (p : ℝ × ℝ) : Prop :=
  let r := m.x - directrix
  (p.1 - m.x)^2 + (p.2 - m.y)^2 = r^2

-- Theorem statement
theorem fixed_point_on_circle (m : PointOnParabola) :
  circle_tangent_to_directrix m focus := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_circle_l1649_164990


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1649_164959

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) =
  (x^2 + 6*x + 1) * (x^2 + 6*x + 37) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1649_164959


namespace NUMINAMATH_CALUDE_z_pure_imaginary_iff_z_in_fourth_quadrant_iff_l1649_164955

def z (m : ℝ) : ℂ := (1 + Complex.I) * m^2 - 3 * Complex.I * m + 2 * Complex.I - 1

theorem z_pure_imaginary_iff (m : ℝ) : 
  z m = Complex.I * Complex.im (z m) ↔ m = -1 :=
sorry

theorem z_in_fourth_quadrant_iff (m : ℝ) :
  Complex.re (z m) > 0 ∧ Complex.im (z m) < 0 ↔ 1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_z_pure_imaginary_iff_z_in_fourth_quadrant_iff_l1649_164955


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1649_164903

theorem inequality_system_solution : 
  {x : ℝ | x + 1 > 0 ∧ -2 * x ≤ 6} = {x : ℝ | x > -1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1649_164903


namespace NUMINAMATH_CALUDE_one_student_owns_all_pets_l1649_164902

/-- Represents the pet ownership distribution in Sara's class -/
structure PetOwnership where
  total : ℕ
  dog_owners : ℕ
  cat_owners : ℕ
  bird_owners : ℕ
  no_pets : ℕ
  just_dogs : ℕ
  just_cats : ℕ
  just_birds : ℕ
  dogs_and_cats : ℕ
  dogs_and_birds : ℕ
  cats_and_birds : ℕ
  all_three : ℕ

/-- The theorem stating that exactly one student owns all three types of pets -/
theorem one_student_owns_all_pets (p : PetOwnership) : 
  p.total = 48 ∧ 
  p.dog_owners = p.total / 2 ∧ 
  p.cat_owners = p.total * 5 / 16 ∧ 
  p.bird_owners = 8 ∧ 
  p.no_pets = 7 ∧
  p.just_dogs = 12 ∧
  p.just_cats = 2 ∧
  p.just_birds = 4 ∧
  p.dog_owners = p.just_dogs + p.dogs_and_cats + p.dogs_and_birds + p.all_three ∧
  p.cat_owners = p.just_cats + p.dogs_and_cats + p.cats_and_birds + p.all_three ∧
  p.bird_owners = p.just_birds + p.dogs_and_birds + p.cats_and_birds + p.all_three ∧
  p.total = p.just_dogs + p.just_cats + p.just_birds + p.dogs_and_cats + p.dogs_and_birds + p.cats_and_birds + p.all_three + p.no_pets
  →
  p.all_three = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_student_owns_all_pets_l1649_164902


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1649_164952

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |x - 5| = 3 * x + 1 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1649_164952


namespace NUMINAMATH_CALUDE_triangle_balls_proof_l1649_164930

/-- The number of balls in an equilateral triangle arrangement -/
def triangle_balls : ℕ := 820

/-- The number of balls added to form a square -/
def added_balls : ℕ := 424

/-- The difference in side length between the triangle and the square -/
def side_difference : ℕ := 8

/-- Formula for the sum of the first n natural numbers -/
def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The side length of the triangle -/
def triangle_side : ℕ := 40

/-- The side length of the square -/
def square_side : ℕ := triangle_side - side_difference

theorem triangle_balls_proof :
  triangle_balls = triangle_sum triangle_side ∧
  triangle_balls + added_balls = square_side^2 ∧
  triangle_side = square_side + side_difference :=
sorry

end NUMINAMATH_CALUDE_triangle_balls_proof_l1649_164930


namespace NUMINAMATH_CALUDE_find_divisor_l1649_164995

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 3 + remainder) :
  3 = dividend / quotient :=
by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1649_164995


namespace NUMINAMATH_CALUDE_range_of_a_l1649_164999

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a^2 * x - 2 ≤ 0) → -2 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1649_164999


namespace NUMINAMATH_CALUDE_lcm_12_21_30_l1649_164976

theorem lcm_12_21_30 : Nat.lcm (Nat.lcm 12 21) 30 = 420 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_21_30_l1649_164976


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1649_164994

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x - 1) - 5 / (x - 2) + 5 / (x - 3) - 2 / (x - 4) < 1 / 20) ↔ 
  (x < -1 ∨ (2 < x ∧ x < 3) ∨ (5 < x ∧ x < 6)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1649_164994


namespace NUMINAMATH_CALUDE_additional_profit_special_house_l1649_164965

/-- The selling price of standard houses in the area -/
def standard_house_price : ℝ := 320000

/-- The additional cost to build the special house -/
def additional_build_cost : ℝ := 100000

/-- The factor by which the special house sells compared to standard houses -/
def special_house_price_factor : ℝ := 1.5

/-- Theorem stating the additional profit made by building the special house -/
theorem additional_profit_special_house : 
  (special_house_price_factor * standard_house_price - standard_house_price) - additional_build_cost = 60000 := by
  sorry

end NUMINAMATH_CALUDE_additional_profit_special_house_l1649_164965


namespace NUMINAMATH_CALUDE_astronomy_collections_l1649_164989

/-- Represents the distinct letters in "ASTRONOMY" --/
inductive AstronomyLetter
| A
| O
| S
| T
| R
| N
| M
| Y

/-- The number of each letter in "ASTRONOMY" --/
def letter_count : AstronomyLetter → Nat
| AstronomyLetter.A => 1
| AstronomyLetter.O => 2
| AstronomyLetter.S => 1
| AstronomyLetter.T => 1
| AstronomyLetter.R => 1
| AstronomyLetter.N => 2
| AstronomyLetter.M => 1
| AstronomyLetter.Y => 1

/-- The set of vowels in "ASTRONOMY" --/
def vowels : Set AstronomyLetter := {AstronomyLetter.A, AstronomyLetter.O}

/-- The set of consonants in "ASTRONOMY" --/
def consonants : Set AstronomyLetter := {AstronomyLetter.S, AstronomyLetter.T, AstronomyLetter.R, AstronomyLetter.N, AstronomyLetter.M, AstronomyLetter.Y}

/-- The number of distinct ways to choose 3 vowels and 3 consonants from "ASTRONOMY" --/
def distinct_collections : Nat := 100

theorem astronomy_collections :
  distinct_collections = 100 := by sorry


end NUMINAMATH_CALUDE_astronomy_collections_l1649_164989


namespace NUMINAMATH_CALUDE_assignments_for_thirty_points_l1649_164957

/-- Calculates the number of assignments needed for a given number of points -/
def assignments_needed (points : ℕ) : ℕ :=
  if points ≤ 10 then points
  else if points ≤ 20 then 10 + 2 * (points - 10)
  else 30 + 3 * (points - 20)

/-- Theorem stating that 60 assignments are needed for 30 points -/
theorem assignments_for_thirty_points :
  assignments_needed 30 = 60 := by sorry

end NUMINAMATH_CALUDE_assignments_for_thirty_points_l1649_164957


namespace NUMINAMATH_CALUDE_shekar_science_marks_l1649_164978

/-- Calculates the marks in science given other subject marks and the average -/
def calculate_science_marks (math social english biology average : ℕ) : ℕ :=
  5 * average - (math + social + english + biology)

/-- Proves that Shekar's science marks are 65 given his other marks and average -/
theorem shekar_science_marks :
  let math := 76
  let social := 82
  let english := 62
  let biology := 85
  let average := 74
  calculate_science_marks math social english biology average = 65 := by
  sorry

#eval calculate_science_marks 76 82 62 85 74

end NUMINAMATH_CALUDE_shekar_science_marks_l1649_164978


namespace NUMINAMATH_CALUDE_boys_ages_l1649_164946

theorem boys_ages (age1 age2 age3 : ℕ) : 
  age1 + age2 + age3 = 29 →
  age1 = age2 →
  age3 = 11 →
  age1 = 9 := by
sorry

end NUMINAMATH_CALUDE_boys_ages_l1649_164946


namespace NUMINAMATH_CALUDE_total_tax_deduction_in_cents_l1649_164923

-- Define the hourly wage in dollars
def hourly_wage : ℝ := 25

-- Define the local tax rate as a percentage
def local_tax_rate : ℝ := 2

-- Define the state tax rate as a percentage
def state_tax_rate : ℝ := 0.5

-- Define the conversion rate from dollars to cents
def dollars_to_cents : ℝ := 100

-- Theorem statement
theorem total_tax_deduction_in_cents :
  (hourly_wage * dollars_to_cents) * (local_tax_rate / 100 + state_tax_rate / 100) = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_total_tax_deduction_in_cents_l1649_164923


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1649_164910

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (2 * x + 1)^3 = a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀) →
  a₀ + a₁ + a₂ + a₃ = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1649_164910


namespace NUMINAMATH_CALUDE_sandrine_dishes_washed_l1649_164951

def number_of_pears_picked : ℕ := 50

def number_of_bananas_cooked (pears : ℕ) : ℕ := 3 * pears

def number_of_dishes_washed (bananas : ℕ) : ℕ := bananas + 10

theorem sandrine_dishes_washed :
  number_of_dishes_washed (number_of_bananas_cooked number_of_pears_picked) = 160 := by
  sorry

end NUMINAMATH_CALUDE_sandrine_dishes_washed_l1649_164951


namespace NUMINAMATH_CALUDE_quadratic_polynomial_condition_l1649_164973

/-- 
Given a polynomial p(x) = 2a x^4 + 5a x^3 - 13 x^2 - x^4 + 2021 + 2x + bx^3 - bx^4 - 13x^3,
if p(x) is a quadratic polynomial, then a^2 + b^2 = 13.
-/
theorem quadratic_polynomial_condition (a b : ℝ) : 
  (∀ x : ℝ, (2*a - b - 1) * x^4 + (5*a + b - 13) * x^3 - 13 * x^2 + 2 * x + 2021 = 
             -13 * x^2 + 2 * x + 2021) → 
  a^2 + b^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_condition_l1649_164973


namespace NUMINAMATH_CALUDE_combined_cost_apples_strawberries_l1649_164966

def total_cost : ℕ := 82
def banana_cost : ℕ := 12
def bread_cost : ℕ := 9
def milk_cost : ℕ := 7
def apple_cost : ℕ := 15
def orange_cost : ℕ := 13
def strawberry_cost : ℕ := 26

theorem combined_cost_apples_strawberries :
  apple_cost + strawberry_cost = 41 :=
by sorry

end NUMINAMATH_CALUDE_combined_cost_apples_strawberries_l1649_164966


namespace NUMINAMATH_CALUDE_imaginary_unit_power_2013_l1649_164922

theorem imaginary_unit_power_2013 (i : ℂ) (h : i^2 = -1) : i^2013 = i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_2013_l1649_164922


namespace NUMINAMATH_CALUDE_bus_capacity_proof_l1649_164970

theorem bus_capacity_proof (C : ℕ) : 
  (3 : ℚ) / 5 * C + 32 = C → C = 80 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_proof_l1649_164970


namespace NUMINAMATH_CALUDE_square_a_times_a_plus_four_l1649_164929

theorem square_a_times_a_plus_four (a : ℝ) (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_a_times_a_plus_four_l1649_164929


namespace NUMINAMATH_CALUDE_difference_of_squares_23_15_l1649_164919

theorem difference_of_squares_23_15 : (23 + 15)^2 - (23 - 15)^2 = 304 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_23_15_l1649_164919


namespace NUMINAMATH_CALUDE_principal_calculation_l1649_164900

/-- Given an interest rate, time period, and a relationship between
    the principal and interest, prove that the principal is 9200. -/
theorem principal_calculation (r t : ℝ) (P : ℝ) :
  r = 0.12 →
  t = 3 →
  P * r * t = P - 5888 →
  P = 9200 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l1649_164900


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1649_164906

theorem consecutive_integers_sum (x y z : ℤ) : 
  (x = y + 1) → 
  (y = z + 1) → 
  (x > y) → 
  (y > z) → 
  (2 * x + 3 * y + 3 * z = 5 * y + 11) → 
  z = 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1649_164906


namespace NUMINAMATH_CALUDE_stock_price_uniqueness_l1649_164908

theorem stock_price_uniqueness : ¬∃ (k m : ℕ), (117/100)^k * (83/100)^m = 1 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_uniqueness_l1649_164908


namespace NUMINAMATH_CALUDE_shot_put_surface_area_l1649_164907

/-- The surface area of a sphere with diameter 5 inches is 25π square inches. -/
theorem shot_put_surface_area :
  let diameter : ℝ := 5
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 25 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shot_put_surface_area_l1649_164907


namespace NUMINAMATH_CALUDE_S_6_value_l1649_164971

theorem S_6_value (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12077 := by
  sorry

end NUMINAMATH_CALUDE_S_6_value_l1649_164971


namespace NUMINAMATH_CALUDE_rearrangements_count_l1649_164936

def word : String := "Alejandro"
def subwords : List String := ["ned", "den"]

theorem rearrangements_count : 
  (List.length word.data - 2) * (Nat.factorial (List.length word.data - 2) / 2) * (List.length subwords) = 40320 := by
  sorry

end NUMINAMATH_CALUDE_rearrangements_count_l1649_164936


namespace NUMINAMATH_CALUDE_hockey_tournament_points_l1649_164964

/-- The number of teams in the tournament -/
def num_teams : ℕ := 2016

/-- The number of points awarded for a win -/
def points_per_win : ℕ := 3

/-- The total number of games played in the tournament -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- The total number of points awarded in the tournament -/
def total_points : ℕ := total_games * points_per_win

theorem hockey_tournament_points :
  total_points = 6093360 := by sorry

end NUMINAMATH_CALUDE_hockey_tournament_points_l1649_164964


namespace NUMINAMATH_CALUDE_sin_neg_pi_third_l1649_164933

theorem sin_neg_pi_third : Real.sin (-π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_pi_third_l1649_164933


namespace NUMINAMATH_CALUDE_notebook_cost_l1649_164945

theorem notebook_cost (x y : ℚ) 
  (eq1 : 5 * x + 4 * y = 380)
  (eq2 : 3 * x + 6 * y = 354) :
  x = 48 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l1649_164945


namespace NUMINAMATH_CALUDE_walking_scenario_l1649_164980

def distance_between_people (initial_distance : ℝ) (person1_movement : ℝ) (person2_movement : ℝ) : ℝ :=
  initial_distance + person1_movement - person2_movement

theorem walking_scenario (initial_distance : ℝ) (person1_movement : ℝ) (person2_movement : ℝ) :
  initial_distance = 400 ∧ person1_movement = 200 ∧ person2_movement = -200 →
  distance_between_people initial_distance person1_movement person2_movement = 400 :=
by
  sorry

#check walking_scenario

end NUMINAMATH_CALUDE_walking_scenario_l1649_164980


namespace NUMINAMATH_CALUDE_siblings_ratio_l1649_164928

/-- Given the number of siblings for Masud, Janet, and Carlos, prove the ratio of Carlos's to Masud's siblings -/
theorem siblings_ratio (masud_siblings : ℕ) (janet_siblings : ℕ) (carlos_siblings : ℕ) : 
  masud_siblings = 60 →
  janet_siblings = 4 * masud_siblings - 60 →
  janet_siblings = carlos_siblings + 135 →
  carlos_siblings * 4 = masud_siblings * 3 := by
  sorry

#check siblings_ratio

end NUMINAMATH_CALUDE_siblings_ratio_l1649_164928


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1649_164927

theorem complex_magnitude_problem (a b c : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : Complex.abs (a + b + c) = 1)
  (h5 : Complex.abs (a - b) = Complex.abs (a - c))
  (h6 : b ≠ c) :
  Complex.abs (a + b) * Complex.abs (a + c) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1649_164927


namespace NUMINAMATH_CALUDE_derived_point_relation_find_original_point_translated_derived_point_l1649_164968

/-- Definition of an a-th order derived point -/
def derived_point (a : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (a * P.1 + P.2, P.1 + a * P.2)

/-- Theorem stating the relationship between a point and its a-th order derived point -/
theorem derived_point_relation (a : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) :
  a ≠ 0 → Q = derived_point a P ↔ 
    Q.1 = a * P.1 + P.2 ∧ Q.2 = P.1 + a * P.2 := by
  sorry

/-- Theorem for finding the original point given its a-th order derived point -/
theorem find_original_point (a : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) :
  a ≠ 0 ∧ a ≠ 1 → Q = derived_point a P →
    P = ((a * Q.2 - Q.1) / (a * a - 1), (a * Q.1 - Q.2) / (a * a - 1)) := by
  sorry

/-- Theorem for the composition of translation and derived point transformation -/
theorem translated_derived_point (a c : ℝ) :
  let P : ℝ × ℝ := (c + 1, 2 * c - 1)
  let P₁ : ℝ × ℝ := (c - 1, 2 * c)
  let P₂ : ℝ × ℝ := derived_point (-3) P₁
  (P₂.1 = 0 ∨ P₂.2 = 0) →
    (P₂ = (0, -16) ∨ P₂ = (16/5, 0)) := by
  sorry

end NUMINAMATH_CALUDE_derived_point_relation_find_original_point_translated_derived_point_l1649_164968


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1649_164901

theorem cubic_equation_roots (k m : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ x : ℝ, x^3 - 9*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  k + m = 50 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1649_164901


namespace NUMINAMATH_CALUDE_weight_of_ten_moles_example_l1649_164921

/-- Calculates the weight of a given number of moles of a compound with a known molecular weight. -/
def weight_of_compound (moles : ℝ) (molecular_weight : ℝ) : ℝ :=
  moles * molecular_weight

/-- Proves that the weight of 10 moles of a compound with a molecular weight of 1080 grams/mole is 10800 grams. -/
theorem weight_of_ten_moles_example : weight_of_compound 10 1080 = 10800 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_ten_moles_example_l1649_164921


namespace NUMINAMATH_CALUDE_bisection_method_next_interval_l1649_164944

def f (x : ℝ) := x^3 - 2*x - 5

theorem bisection_method_next_interval :
  let a := 2
  let b := 3
  let x₀ := (a + b) / 2
  f a < 0 ∧ f b > 0 ∧ f x₀ > 0 →
  ∃ x ∈ Set.Ioo a x₀, f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_bisection_method_next_interval_l1649_164944


namespace NUMINAMATH_CALUDE_second_player_loses_l1649_164949

/-- Represents the game state -/
structure GameState :=
  (diamonds : ℕ)

/-- Represents a move in the game -/
def is_valid_move (s : GameState) (s' : GameState) : Prop :=
  s'.diamonds = s.diamonds + 1 ∧ s'.diamonds ≤ 2017

/-- The game ends when there are 2017 piles (diamonds) -/
def game_over (s : GameState) : Prop :=
  s.diamonds = 2017

/-- The number of moves required to finish the game -/
def moves_to_end (s : GameState) : ℕ :=
  2017 - s.diamonds

/-- Theorem: The second player loses in a game starting with 2017 diamonds -/
theorem second_player_loses :
  ∀ (s : GameState),
    s.diamonds = 1 →
    ∃ (strategy : GameState → GameState),
      (∀ (s' : GameState), is_valid_move s s' → is_valid_move s' (strategy s')) →
      (moves_to_end s) % 2 = 0 →
      game_over (strategy s) :=
sorry

end NUMINAMATH_CALUDE_second_player_loses_l1649_164949


namespace NUMINAMATH_CALUDE_mary_candy_ratio_l1649_164913

/-- The number of times Mary initially has more candy than Megan -/
def candy_ratio (megan_candy : ℕ) (mary_final_candy : ℕ) (mary_added_candy : ℕ) : ℚ :=
  (mary_final_candy - mary_added_candy : ℚ) / megan_candy

theorem mary_candy_ratio :
  candy_ratio 5 25 10 = 3 := by sorry

end NUMINAMATH_CALUDE_mary_candy_ratio_l1649_164913


namespace NUMINAMATH_CALUDE_long_knight_min_moves_l1649_164983

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents a move of the long knight -/
inductive LongKnightMove
  | horizontal : Bool → LongKnightMove  -- True for right, False for left
  | vertical : Bool → LongKnightMove    -- True for up, False for down

/-- The size of the chessboard -/
def boardSize : Nat := 8

/-- Applies a long knight move to a position -/
def applyMove (pos : Position) (move : LongKnightMove) : Position :=
  match move with
  | LongKnightMove.horizontal right =>
      let newX := if right then min (pos.x + 3) (boardSize - 1) else max (pos.x - 3) 0
      let newY := if right then min (pos.y + 1) (boardSize - 1) else max (pos.y - 1) 0
      ⟨newX, newY⟩
  | LongKnightMove.vertical up =>
      let newX := if up then min (pos.x + 1) (boardSize - 1) else max (pos.x - 1) 0
      let newY := if up then min (pos.y + 3) (boardSize - 1) else max (pos.y - 3) 0
      ⟨newX, newY⟩

/-- Checks if a position is at the opposite corner -/
def isOppositeCorner (pos : Position) : Prop :=
  pos.x = boardSize - 1 ∧ pos.y = boardSize - 1

/-- Theorem: The minimum number of moves for a long knight to reach the opposite corner is 5 -/
theorem long_knight_min_moves :
  ∀ (moves : List LongKnightMove),
    let finalPos := moves.foldl applyMove ⟨0, 0⟩
    isOppositeCorner finalPos → moves.length ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_long_knight_min_moves_l1649_164983


namespace NUMINAMATH_CALUDE_not_perfect_square_exists_l1649_164918

theorem not_perfect_square_exists (a b : ℕ) (ha : a > 1) (hb : b > 1) (hab : a ≠ b) :
  ∃ n : ℕ+, ¬ ∃ k : ℕ, (a^n.val - 1) * (b^n.val - 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_exists_l1649_164918


namespace NUMINAMATH_CALUDE_larger_number_proof_l1649_164905

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365) 
  (h2 : L = 4 * S + 15) : 
  L = 1815 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1649_164905


namespace NUMINAMATH_CALUDE_picture_area_is_6600_l1649_164998

/-- Calculates the area of a picture within a rectangular frame. -/
def picture_area (outer_height outer_width short_frame_width long_frame_width : ℕ) : ℕ :=
  (outer_height - 2 * short_frame_width) * (outer_width - 2 * long_frame_width)

/-- Theorem stating that for a frame with given dimensions, the enclosed picture has an area of 6600 cm². -/
theorem picture_area_is_6600 :
  picture_area 100 140 20 15 = 6600 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_is_6600_l1649_164998


namespace NUMINAMATH_CALUDE_sum_of_squares_perfect_square_l1649_164993

theorem sum_of_squares_perfect_square (n p k : ℤ) :
  ∃ m : ℤ, n^2 + p^2 + k^2 = m^2 ↔ n * k = (p / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_perfect_square_l1649_164993


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1649_164938

theorem power_fraction_simplification :
  (16 ^ 10 * 8 ^ 6) / (4 ^ 22) = 2 ^ 14 := by sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1649_164938


namespace NUMINAMATH_CALUDE_aqua_park_earnings_l1649_164904

def admission_cost : ℚ := 12
def tour_cost : ℚ := 6
def meal_cost : ℚ := 10
def souvenir_cost : ℚ := 8

def group1_size : ℕ := 10
def group2_size : ℕ := 15
def group3_size : ℕ := 8

def group1_discount_rate : ℚ := 0.10
def group2_meal_discount_rate : ℚ := 0.05

def group1_total (admission_cost tour_cost meal_cost souvenir_cost : ℚ) (group_size : ℕ) (discount_rate : ℚ) : ℚ :=
  (1 - discount_rate) * (admission_cost + tour_cost + meal_cost + souvenir_cost) * group_size

def group2_total (admission_cost meal_cost : ℚ) (group_size : ℕ) (meal_discount_rate : ℚ) : ℚ :=
  admission_cost * group_size + (1 - meal_discount_rate) * meal_cost * group_size

def group3_total (admission_cost tour_cost souvenir_cost : ℚ) (group_size : ℕ) : ℚ :=
  (admission_cost + tour_cost + souvenir_cost) * group_size

theorem aqua_park_earnings : 
  group1_total admission_cost tour_cost meal_cost souvenir_cost group1_size group1_discount_rate +
  group2_total admission_cost meal_cost group2_size group2_meal_discount_rate +
  group3_total admission_cost tour_cost souvenir_cost group3_size = 854.5 := by
  sorry

end NUMINAMATH_CALUDE_aqua_park_earnings_l1649_164904


namespace NUMINAMATH_CALUDE_counterpositive_equivalence_l1649_164985

theorem counterpositive_equivalence (a b c : ℝ) :
  (a^2 + b^2 + c^2 < 3 → a + b + c ≠ 3) ↔
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_counterpositive_equivalence_l1649_164985


namespace NUMINAMATH_CALUDE_star_theorem_l1649_164977

/-- The star operation for real numbers -/
def star (a b : ℝ) : ℝ := (a - b) ^ 3

/-- Theorem: For real numbers x and y, (x-y)^3 ⋆ (y-x)^3 = 8(x-y)^9 -/
theorem star_theorem (x y : ℝ) : star ((x - y) ^ 3) ((y - x) ^ 3) = 8 * (x - y) ^ 9 := by
  sorry

end NUMINAMATH_CALUDE_star_theorem_l1649_164977


namespace NUMINAMATH_CALUDE_intersection_A_B_empty_complement_A_union_B_a_geq_one_l1649_164987

-- Define the sets A, B, U, and C
def A : Set ℝ := {x | x^2 + 6*x + 5 < 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 1}
def U : Set ℝ := {x | |x| < 5}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem 1: A ∩ B = ∅
theorem intersection_A_B_empty : A ∩ B = ∅ := by sorry

-- Theorem 2: ∁_U(A ∪ B) = {x | 1 ≤ x < 5}
theorem complement_A_union_B : 
  (A ∪ B)ᶜ ∩ U = {x : ℝ | 1 ≤ x ∧ x < 5} := by sorry

-- Theorem 3: B ∩ C = B implies a ≥ 1
theorem a_geq_one (a : ℝ) (h : B ∩ C a = B) : a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_empty_complement_A_union_B_a_geq_one_l1649_164987


namespace NUMINAMATH_CALUDE_function_always_negative_iff_m_in_range_l1649_164992

/-- The function f(x) = mx^2 - mx - 1 is negative for all real x
    if and only if m is in the interval (-4, 0]. -/
theorem function_always_negative_iff_m_in_range (m : ℝ) :
  (∀ x, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_function_always_negative_iff_m_in_range_l1649_164992


namespace NUMINAMATH_CALUDE_parallel_transitivity_l1649_164941

-- Define a type for lines in a plane
def Line : Type := ℝ × ℝ → Prop

-- Define parallel relationship between lines
def parallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem parallel_transitivity (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l1649_164941


namespace NUMINAMATH_CALUDE_min_value_xy_l1649_164991

theorem min_value_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h_geom : ∃ r : ℝ, (Real.log x) * r = (1/2) ∧ (1/2) * r = Real.log y) : 
  (∀ a b : ℝ, a > 1 → b > 1 → 
    (∃ r : ℝ, (Real.log a) * r = (1/2) ∧ (1/2) * r = Real.log b) → 
    x * y ≤ a * b) → 
  x * y = Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_l1649_164991


namespace NUMINAMATH_CALUDE_log_equation_solution_l1649_164925

theorem log_equation_solution :
  ∃! x : ℝ, x > 0 ∧ Real.log x - Real.log 6 = 2 :=
by
  use 3/2
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1649_164925


namespace NUMINAMATH_CALUDE_fraction_simplification_l1649_164924

theorem fraction_simplification :
  (1^2 + 1) * (2^2 + 1) * (3^2 + 1) / ((2^2 - 1) * (3^2 - 1) * (4^2 - 1)) = 5 / 18 :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1649_164924


namespace NUMINAMATH_CALUDE_petrol_price_increase_l1649_164960

theorem petrol_price_increase (original_price : ℝ) (original_consumption : ℝ) : 
  let consumption_reduction : ℝ := 0.2857142857142857
  let new_consumption : ℝ := original_consumption * (1 - consumption_reduction)
  let new_price : ℝ := original_price * original_consumption / new_consumption
  new_price / original_price - 1 = 0.4 := by sorry

end NUMINAMATH_CALUDE_petrol_price_increase_l1649_164960


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1649_164939

/-- Represents the number of students in each grade level -/
structure GradePopulation where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ

/-- Represents the sample size for each grade level -/
structure GradeSample where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ

/-- The total population of the school -/
def totalPopulation (gp : GradePopulation) : ℕ :=
  gp.freshmen + gp.sophomores + gp.juniors

/-- The total sample size -/
def totalSample (gs : GradeSample) : ℕ :=
  gs.freshmen + gs.sophomores + gs.juniors

/-- Checks if the sample is proportional to the population for each grade -/
def isProportionalSample (gp : GradePopulation) (gs : GradeSample) : Prop :=
  gs.freshmen * totalPopulation gp = gp.freshmen * totalSample gs ∧
  gs.sophomores * totalPopulation gp = gp.sophomores * totalSample gs ∧
  gs.juniors * totalPopulation gp = gp.juniors * totalSample gs

theorem stratified_sampling_theorem (gp : GradePopulation) (gs : GradeSample) :
  gp.freshmen = 300 →
  gp.sophomores = 200 →
  gp.juniors = 400 →
  totalPopulation gp = 900 →
  totalSample gs = 45 →
  isProportionalSample gp gs →
  gs.freshmen = 15 ∧ gs.sophomores = 10 ∧ gs.juniors = 20 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1649_164939


namespace NUMINAMATH_CALUDE_papaya_height_after_five_years_l1649_164920

/-- The height of a papaya tree after n years -/
def papayaHeight (n : ℕ) : ℝ :=
  match n with
  | 0 => 0
  | 1 => 2
  | 2 => papayaHeight 1 + 1.5 * papayaHeight 1
  | 3 => papayaHeight 2 + 1.5 * papayaHeight 2
  | 4 => papayaHeight 3 + 2 * papayaHeight 3
  | 5 => papayaHeight 4 + 0.5 * papayaHeight 4
  | _ => 0  -- undefined for years beyond 5

theorem papaya_height_after_five_years :
  papayaHeight 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_papaya_height_after_five_years_l1649_164920


namespace NUMINAMATH_CALUDE_birthday_75_days_later_l1649_164932

theorem birthday_75_days_later (birthday : ℕ) : 
  (birthday % 7 = 0) → ((birthday + 75) % 7 = 5) := by
  sorry

#check birthday_75_days_later

end NUMINAMATH_CALUDE_birthday_75_days_later_l1649_164932


namespace NUMINAMATH_CALUDE_p_greater_than_q_greater_than_r_l1649_164915

def P : ℚ := -1 / (201603 * 201604)
def Q : ℚ := -1 / (201602 * 201604)
def R : ℚ := -1 / (201602 * 201603)

theorem p_greater_than_q_greater_than_r : P > Q ∧ Q > R := by sorry

end NUMINAMATH_CALUDE_p_greater_than_q_greater_than_r_l1649_164915


namespace NUMINAMATH_CALUDE_centroid_count_l1649_164916

/-- Represents a point on the perimeter of the square -/
structure PerimeterPoint where
  x : ℚ
  y : ℚ
  on_perimeter : (x = 0 ∨ x = 15 ∨ y = 0 ∨ y = 15) ∧ 
                 (0 ≤ x ∧ x ≤ 15) ∧ 
                 (0 ≤ y ∧ y ≤ 15)

/-- The set of 64 equally spaced points on the square's perimeter -/
def perimeter_points : Finset PerimeterPoint := sorry

/-- Checks if three points are collinear -/
def collinear (p q r : PerimeterPoint) : Prop := sorry

/-- Represents the centroid of a triangle -/
structure Centroid where
  x : ℚ
  y : ℚ
  inside_square : (0 < x ∧ x < 15) ∧ (0 < y ∧ y < 15)

/-- Calculates the centroid of a triangle given three points -/
def triangle_centroid (p q r : PerimeterPoint) : Centroid := sorry

/-- The set of all possible centroids -/
def all_centroids : Finset Centroid := sorry

/-- The main theorem to prove -/
theorem centroid_count : 
  (Finset.card perimeter_points = 64) →
  (∀ p ∈ perimeter_points, ∃ m n : ℕ, p.x = m / 15 ∧ p.y = n / 15) →
  (Finset.card all_centroids = 1849) := sorry

end NUMINAMATH_CALUDE_centroid_count_l1649_164916


namespace NUMINAMATH_CALUDE_hcf_of_ratio_and_lcm_l1649_164979

/-- 
Given three positive integers a, b, and c that are in the ratio 3:4:5 and 
have a least common multiple of 2400, prove that their highest common factor is 20.
-/
theorem hcf_of_ratio_and_lcm (a b c : ℕ+) 
  (h_ratio : ∃ (k : ℕ+), a = 3 * k ∧ b = 4 * k ∧ c = 5 * k)
  (h_lcm : Nat.lcm a (Nat.lcm b c) = 2400) :
  Nat.gcd a (Nat.gcd b c) = 20 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_ratio_and_lcm_l1649_164979


namespace NUMINAMATH_CALUDE_triangle_altitude_segment_l1649_164974

/-- Given a triangle with sides 35, 85, and 90 units, prove that when an altitude is dropped on the side of length 90, the length of the larger segment cut off by the altitude is 78.33 units. -/
theorem triangle_altitude_segment (a b c : ℝ) (h1 : a = 35) (h2 : b = 85) (h3 : c = 90) :
  let x := (c^2 + a^2 - b^2) / (2 * c)
  c - x = 78.33 := by sorry

end NUMINAMATH_CALUDE_triangle_altitude_segment_l1649_164974


namespace NUMINAMATH_CALUDE_max_abs_z_value_l1649_164956

theorem max_abs_z_value (a b c z : ℂ) (d : ℝ) 
  (h1 : Complex.abs a = Complex.abs b)
  (h2 : Complex.abs a = (1 / 2) * Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : a * z^2 + b * z + d * c = 0)
  (h5 : d = 1) :
  ∃ (M : ℝ), M = 2 ∧ ∀ z', a * z'^2 + b * z' + d * c = 0 → Complex.abs z' ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_value_l1649_164956


namespace NUMINAMATH_CALUDE_charity_ticket_revenue_l1649_164950

theorem charity_ticket_revenue :
  ∀ (full_price_tickets half_price_tickets : ℕ) (full_price : ℕ),
    full_price_tickets + half_price_tickets = 180 →
    full_price_tickets * full_price + half_price_tickets * (full_price / 2) = 2750 →
    full_price_tickets * full_price = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_charity_ticket_revenue_l1649_164950


namespace NUMINAMATH_CALUDE_condition_equivalence_l1649_164942

theorem condition_equivalence (p q : Prop) :
  (¬(p ∧ q) ∧ (p ∨ q)) ↔ (p ≠ q) :=
sorry

end NUMINAMATH_CALUDE_condition_equivalence_l1649_164942


namespace NUMINAMATH_CALUDE_marks_lost_is_one_l1649_164962

/-- Represents an examination with given parameters -/
structure Examination where
  total_questions : ℕ
  correct_score : ℕ
  total_score : ℤ
  correct_answers : ℕ

/-- Calculates the marks lost for each wrong answer in the examination -/
def marks_lost_per_wrong_answer (exam : Examination) : ℚ :=
  let wrong_answers := exam.total_questions - exam.correct_answers
  let total_correct_score := exam.correct_score * exam.correct_answers
  (total_correct_score - exam.total_score) / wrong_answers

/-- Theorem stating that the marks lost for each wrong answer is 1 -/
theorem marks_lost_is_one (exam : Examination) 
    (h1 : exam.total_questions = 60)
    (h2 : exam.correct_score = 4)
    (h3 : exam.total_score = 110)
    (h4 : exam.correct_answers = 34) :
  marks_lost_per_wrong_answer exam = 1 := by
  sorry

#eval marks_lost_per_wrong_answer { 
  total_questions := 60, 
  correct_score := 4, 
  total_score := 110, 
  correct_answers := 34 
}

end NUMINAMATH_CALUDE_marks_lost_is_one_l1649_164962


namespace NUMINAMATH_CALUDE_current_calculation_l1649_164934

theorem current_calculation (Q R t I : ℝ) 
  (heat_eq : Q = I^2 * R * t)
  (resistance : R = 8)
  (heat_generated : Q = 72)
  (time : t = 2) :
  I = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_current_calculation_l1649_164934


namespace NUMINAMATH_CALUDE_divisible_by_24_l1649_164911

theorem divisible_by_24 (a : ℤ) : ∃ k : ℤ, (a^2 + 3*a + 1)^2 - 1 = 24*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l1649_164911


namespace NUMINAMATH_CALUDE_sqrt_50_simplified_l1649_164996

theorem sqrt_50_simplified : Real.sqrt 50 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_simplified_l1649_164996


namespace NUMINAMATH_CALUDE_folded_square_area_l1649_164967

/-- The area of a shape formed by folding a square along its diagonal -/
theorem folded_square_area (side_length : ℝ) (h : side_length = 2) : 
  (side_length ^ 2) / 2 = 2 := by sorry

end NUMINAMATH_CALUDE_folded_square_area_l1649_164967


namespace NUMINAMATH_CALUDE_camp_food_ratio_l1649_164975

/-- The ratio of food eaten by a dog to a puppy -/
def food_ratio (num_puppies num_dogs : ℕ) 
               (puppy_meal_frequency dog_meal_frequency : ℕ) 
               (dog_food_per_meal : ℝ) 
               (total_food_per_day : ℝ) : ℚ := by
  -- Define the ratio of food eaten by a dog to a puppy
  sorry

/-- Theorem stating the food ratio given the problem conditions -/
theorem camp_food_ratio : 
  food_ratio 4 3 9 3 4 108 = 2 := by
  sorry

end NUMINAMATH_CALUDE_camp_food_ratio_l1649_164975


namespace NUMINAMATH_CALUDE_velvet_area_for_box_l1649_164947

/-- The total area of velvet needed to line the inside of a box with given dimensions -/
theorem velvet_area_for_box (long_side_length long_side_width short_side_length short_side_width top_bottom_area : ℕ) 
  (h1 : long_side_length = 8)
  (h2 : long_side_width = 6)
  (h3 : short_side_length = 5)
  (h4 : short_side_width = 6)
  (h5 : top_bottom_area = 40) :
  2 * (long_side_length * long_side_width) + 
  2 * (short_side_length * short_side_width) + 
  2 * top_bottom_area = 236 := by
  sorry

#eval 2 * (8 * 6) + 2 * (5 * 6) + 2 * 40

end NUMINAMATH_CALUDE_velvet_area_for_box_l1649_164947


namespace NUMINAMATH_CALUDE_water_weight_l1649_164912

/-- Proves that a gallon of water weighs 8 pounds given the conditions of the water tank problem -/
theorem water_weight (tank_capacity : ℝ) (empty_tank_weight : ℝ) (fill_percentage : ℝ) (current_weight : ℝ)
  (h1 : tank_capacity = 200)
  (h2 : empty_tank_weight = 80)
  (h3 : fill_percentage = 0.8)
  (h4 : current_weight = 1360) :
  (current_weight - empty_tank_weight) / (fill_percentage * tank_capacity) = 8 := by
  sorry

end NUMINAMATH_CALUDE_water_weight_l1649_164912


namespace NUMINAMATH_CALUDE_eight_to_power_divided_by_four_l1649_164926

theorem eight_to_power_divided_by_four (n : ℕ) : 
  n = 8^2022 → n / 4 = 4^3032 := by sorry

end NUMINAMATH_CALUDE_eight_to_power_divided_by_four_l1649_164926


namespace NUMINAMATH_CALUDE_first_group_size_is_eight_l1649_164935

/-- The number of men in the first group -/
def first_group_size : ℕ := 8

/-- The number of hours worked per day -/
def hours_per_day : ℕ := 8

/-- The number of days the first group works -/
def days_first_group : ℕ := 24

/-- The number of men in the second group -/
def second_group_size : ℕ := 12

/-- The number of days the second group works -/
def days_second_group : ℕ := 16

theorem first_group_size_is_eight :
  first_group_size * hours_per_day * days_first_group =
  second_group_size * hours_per_day * days_second_group :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_is_eight_l1649_164935


namespace NUMINAMATH_CALUDE_purchase_price_calculation_l1649_164961

/-- Given a markup of $50, which includes 25% of cost for overhead and $12 of net profit,
    the purchase price of the article is $152. -/
theorem purchase_price_calculation (markup overhead_percentage net_profit : ℚ) 
    (h1 : markup = 50)
    (h2 : overhead_percentage = 25 / 100)
    (h3 : net_profit = 12)
    (h4 : markup = overhead_percentage * purchase_price + net_profit) :
  purchase_price = 152 :=
by sorry


end NUMINAMATH_CALUDE_purchase_price_calculation_l1649_164961


namespace NUMINAMATH_CALUDE_area_cyclic_quadrilateral_l1649_164986

/-- Given a quadrilateral ABCD inscribed in a circle with radius R,
    where φ is the angle between its diagonals,
    the area S of the quadrilateral is equal to 2R^2 * sin(A) * sin(B) * sin(φ). -/
theorem area_cyclic_quadrilateral (R : ℝ) (A B φ : ℝ) (S : ℝ) 
    (hR : R > 0) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hφ : 0 < φ ∧ φ < π) :
  S = 2 * R^2 * Real.sin A * Real.sin B * Real.sin φ := by
  sorry

end NUMINAMATH_CALUDE_area_cyclic_quadrilateral_l1649_164986


namespace NUMINAMATH_CALUDE_triangle_inequality_l1649_164931

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  let p := (a + b + c) / 2
  a * Real.sqrt ((p - b) * (p - c) / (b * c)) +
  b * Real.sqrt ((p - c) * (p - a) / (a * c)) +
  c * Real.sqrt ((p - a) * (p - b) / (a * b)) ≥ p := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1649_164931


namespace NUMINAMATH_CALUDE_sum_bound_l1649_164953

theorem sum_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 4/b = 1) :
  a + b > 9 ∧ ∀ ε > 0, ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 1/a' + 4/b' = 1 ∧ a' + b' < 9 + ε :=
sorry

end NUMINAMATH_CALUDE_sum_bound_l1649_164953


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1649_164917

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1649_164917


namespace NUMINAMATH_CALUDE_inspector_group_b_count_l1649_164948

/-- Represents the problem of determining the number of inspectors in Group B -/
theorem inspector_group_b_count : 
  ∀ (a b : ℕ) (group_b_count : ℕ),
  a > 0 → b > 0 →
  (2 * (a + 2 * b)) / 2 = (2 * (a + 5 * b)) / 3 →  -- Equation from Group A's work
  (5 * (a + 5 * b)) / (group_b_count * 5) = (2 * (a + 2 * b)) / (8 * 2) →  -- Equation comparing Group A and B's work
  group_b_count = 12 := by
    sorry


end NUMINAMATH_CALUDE_inspector_group_b_count_l1649_164948


namespace NUMINAMATH_CALUDE_m_range_l1649_164997

theorem m_range (m : ℝ) : 
  let M := Set.Iic m
  let P := {x : ℝ | x ≥ -1}
  M ∩ P = ∅ → m < -1 :=
by
  sorry

end NUMINAMATH_CALUDE_m_range_l1649_164997


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1649_164943

/-- Given a right triangle ABC where:
    - The altitude from C to AB is 12 km
    - The sum of all sides (AB + BC + AC) is 60 km
    Prove that the length of AB is 22.5 km -/
theorem right_triangle_side_length 
  (A B C : ℝ × ℝ) 
  (is_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (altitude : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2 - 
    (((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2))^2 / 
    ((B.1 - A.1)^2 + (B.2 - A.2)^2))) = 12)
  (perimeter : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) + 
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) + 
    Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 60) :
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_side_length_l1649_164943


namespace NUMINAMATH_CALUDE_at_least_five_primes_in_cubic_l1649_164958

theorem at_least_five_primes_in_cubic (f : ℕ → ℕ) : 
  (∀ n : ℕ, f n = n^3 - 10*n^2 + 31*n - 17) →
  ∃ (a b c d e : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    Nat.Prime (f a) ∧ Nat.Prime (f b) ∧ Nat.Prime (f c) ∧ Nat.Prime (f d) ∧ Nat.Prime (f e) :=
by sorry

end NUMINAMATH_CALUDE_at_least_five_primes_in_cubic_l1649_164958


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1649_164937

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1649_164937


namespace NUMINAMATH_CALUDE_polynomial_inequality_l1649_164984

theorem polynomial_inequality (x : ℝ) : 1 + x^2 + x^6 + x^8 ≥ 4 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l1649_164984


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1649_164981

/-- The total cost of sandwiches and sodas -/
def total_cost (sandwich_quantity : ℕ) (sandwich_price : ℚ) (soda_quantity : ℕ) (soda_price : ℚ) : ℚ :=
  sandwich_quantity * sandwich_price + soda_quantity * soda_price

/-- Proof that the total cost of 2 sandwiches at $1.49 each and 4 sodas at $0.87 each is $6.46 -/
theorem total_cost_calculation : total_cost 2 (149/100) 4 (87/100) = 646/100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1649_164981


namespace NUMINAMATH_CALUDE_units_digit_of_six_to_sixth_l1649_164940

theorem units_digit_of_six_to_sixth (n : ℕ) : n = 6^6 → n % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_six_to_sixth_l1649_164940


namespace NUMINAMATH_CALUDE_negation_of_implication_l1649_164988

theorem negation_of_implication (A B : Set α) :
  ¬(∀ a, a ∈ A → b ∈ B) ↔ ∃ a, a ∈ A ∧ b ∉ B :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1649_164988


namespace NUMINAMATH_CALUDE_meet_twice_l1649_164954

/-- Represents the meeting scenario between Michael and the garbage truck -/
structure MeetingScenario where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of times Michael and the truck meet -/
def number_of_meetings (scenario : MeetingScenario) : ℕ :=
  sorry

/-- The theorem stating that Michael and the truck meet exactly twice -/
theorem meet_twice (scenario : MeetingScenario) 
  (h1 : scenario.michael_speed = 6)
  (h2 : scenario.truck_speed = 12)
  (h3 : scenario.pail_distance = 240)
  (h4 : scenario.truck_stop_time = 40)
  (h5 : scenario.initial_distance = 240) :
  number_of_meetings scenario = 2 :=
sorry

end NUMINAMATH_CALUDE_meet_twice_l1649_164954


namespace NUMINAMATH_CALUDE_largest_valid_number_l1649_164963

def is_valid_number (n : ℕ) : Prop :=
  ∃ (r : ℕ) (i : ℕ), 
    i > 0 ∧ 
    i < (Nat.digits 10 n).length ∧ 
    n % 10 ≠ 0 ∧
    r > 1 ∧
    r * (n / 10^(i + 1) * 10^i + n % 10^i) = n

theorem largest_valid_number : 
  is_valid_number 180625 ∧ 
  ∀ m : ℕ, m > 180625 → ¬(is_valid_number m) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l1649_164963


namespace NUMINAMATH_CALUDE_green_balls_removal_l1649_164982

theorem green_balls_removal (total : ℕ) (red_percent : ℚ) (green_removed : ℕ) :
  total = 150 →
  red_percent = 2/5 →
  green_removed = 75 →
  (red_percent * ↑total : ℚ) / (↑total - ↑green_removed : ℚ) = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_green_balls_removal_l1649_164982
