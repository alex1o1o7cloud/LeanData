import Mathlib

namespace NUMINAMATH_CALUDE_square_root_of_25_l2950_295082

theorem square_root_of_25 : ∃ (a b : ℝ), a ≠ b ∧ a^2 = 25 ∧ b^2 = 25 := by
  sorry

#check square_root_of_25

end NUMINAMATH_CALUDE_square_root_of_25_l2950_295082


namespace NUMINAMATH_CALUDE_absolute_value_problem_l2950_295046

theorem absolute_value_problem : |-5| + (3 - Real.sqrt 2) ^ 0 - 2 * Real.tan (π / 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l2950_295046


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_1023_l2950_295069

theorem units_digit_of_7_to_1023 : ∃ n : ℕ, 7^1023 ≡ 3 [ZMOD 10] :=
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_1023_l2950_295069


namespace NUMINAMATH_CALUDE_triangle_angle_sum_and_side_inequality_l2950_295050

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the properties of the triangle
def has_sine_cosine_property (t : Triangle) : Prop :=
  Real.sin t.A + Real.cos t.B = Real.sqrt 2 ∧
  Real.cos t.A + Real.sin t.B = Real.sqrt 2

-- Define the angle bisector property
def has_angle_bisector (t : Triangle) (D : ℝ) : Prop :=
  -- We don't define the specifics of the bisector, just that it exists
  true

-- State the theorem
theorem triangle_angle_sum_and_side_inequality
  (t : Triangle) (D : ℝ) 
  (h1 : has_sine_cosine_property t)
  (h2 : has_angle_bisector t D) :
  t.A + t.B = 90 ∧ t.A > D :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_and_side_inequality_l2950_295050


namespace NUMINAMATH_CALUDE_valentine_spending_percentage_l2950_295080

def total_students : ℕ := 30
def valentine_percentage : ℚ := 60 / 100
def valentine_cost : ℚ := 2
def total_money : ℚ := 40

theorem valentine_spending_percentage :
  (↑total_students * valentine_percentage * valentine_cost) / total_money * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_valentine_spending_percentage_l2950_295080


namespace NUMINAMATH_CALUDE_orthocenter_property_l2950_295079

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the orthocenter of a triangle
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the cosine of the sum of two angles
def cos_sum_angles (α β : ℝ) : ℝ := sorry

-- Define the measure of an angle
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem orthocenter_property (t : Triangle) :
  let O := orthocenter t
  (angle_measure t.A t.B t.C > π / 2) →  -- Angle A is obtuse
  (dist O t.A = dist t.B t.C) →  -- AO = BC
  (cos_sum_angles (angle_measure O t.B t.C) (angle_measure O t.C t.B) = -Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_orthocenter_property_l2950_295079


namespace NUMINAMATH_CALUDE_cube_sum_greater_than_mixed_terms_l2950_295086

theorem cube_sum_greater_than_mixed_terms (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_greater_than_mixed_terms_l2950_295086


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_absolute_values_l2950_295043

theorem min_value_of_sum_of_absolute_values :
  ∃ (m : ℝ), (∀ x : ℝ, m ≤ |x + 2| + |x - 2| + |x - 1|) ∧ (∃ y : ℝ, m = |y + 2| + |y - 2| + |y - 1|) ∧ m = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_absolute_values_l2950_295043


namespace NUMINAMATH_CALUDE_fraction_of_girls_l2950_295011

theorem fraction_of_girls (total_students : ℕ) (boys : ℕ) (h1 : total_students = 160) (h2 : boys = 60) :
  (total_students - boys : ℚ) / total_students = 5 / 8 := by
  sorry

#check fraction_of_girls

end NUMINAMATH_CALUDE_fraction_of_girls_l2950_295011


namespace NUMINAMATH_CALUDE_tennis_percentage_is_31_percent_l2950_295059

/-- The percentage of students who prefer tennis in both schools combined -/
def combined_tennis_percentage (north_total : ℕ) (south_total : ℕ) 
  (north_tennis_percent : ℚ) (south_tennis_percent : ℚ) : ℚ :=
  let north_tennis := (north_total : ℚ) * north_tennis_percent
  let south_tennis := (south_total : ℚ) * south_tennis_percent
  let total_tennis := north_tennis + south_tennis
  let total_students := (north_total + south_total : ℚ)
  total_tennis / total_students

/-- Theorem stating that the percentage of students who prefer tennis in both schools combined is 31% -/
theorem tennis_percentage_is_31_percent :
  combined_tennis_percentage 1800 2700 (25/100) (35/100) = 31/100 := by
  sorry

end NUMINAMATH_CALUDE_tennis_percentage_is_31_percent_l2950_295059


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l2950_295037

theorem unique_solution_power_equation :
  ∃! (a b c d : ℕ), 7^a = 4^b + 5^c + 6^d :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l2950_295037


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2950_295006

theorem square_area_from_diagonal (a b : ℝ) :
  let diagonal := Real.sqrt (a^2 + 4 * b^2)
  (diagonal^2 / 2) = (a^2 + 4 * b^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2950_295006


namespace NUMINAMATH_CALUDE_money_sum_l2950_295005

theorem money_sum (a b : ℝ) (h1 : (3/10) * a = (1/5) * b) (h2 : b = 60) : a + b = 100 := by
  sorry

end NUMINAMATH_CALUDE_money_sum_l2950_295005


namespace NUMINAMATH_CALUDE_fraction_problem_l2950_295094

theorem fraction_problem (x : ℚ) : 
  x / (4 * x - 9) = 3 / 4 → x = 27 / 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l2950_295094


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2950_295093

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 1 → z = (1 / 2 : ℂ) + Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2950_295093


namespace NUMINAMATH_CALUDE_kevins_siblings_l2950_295071

-- Define the traits
inductive EyeColor
| Green
| Grey

inductive HairColor
| Red
| Brown

-- Define a child with their traits
structure Child where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor

-- Define the function to check if two children share a trait
def shareTrait (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor

-- Define the children
def Oliver : Child := ⟨"Oliver", EyeColor.Green, HairColor.Red⟩
def Kevin : Child := ⟨"Kevin", EyeColor.Grey, HairColor.Brown⟩
def Lily : Child := ⟨"Lily", EyeColor.Grey, HairColor.Red⟩
def Emma : Child := ⟨"Emma", EyeColor.Green, HairColor.Brown⟩
def Noah : Child := ⟨"Noah", EyeColor.Green, HairColor.Red⟩
def Mia : Child := ⟨"Mia", EyeColor.Green, HairColor.Brown⟩

-- Define the theorem
theorem kevins_siblings :
  (shareTrait Kevin Emma ∧ shareTrait Kevin Mia ∧ shareTrait Emma Mia) ∧
  (¬ (shareTrait Kevin Oliver ∧ shareTrait Kevin Noah ∧ shareTrait Oliver Noah)) ∧
  (¬ (shareTrait Kevin Lily ∧ shareTrait Kevin Noah ∧ shareTrait Lily Noah)) ∧
  (¬ (shareTrait Kevin Oliver ∧ shareTrait Kevin Lily ∧ shareTrait Oliver Lily)) :=
sorry

end NUMINAMATH_CALUDE_kevins_siblings_l2950_295071


namespace NUMINAMATH_CALUDE_triangle_constant_sum_squares_l2950_295036

/-- Given a triangle XYZ where YZ = 10 and the length of median XM is 7,
    the value of XZ^2 + XY^2 is constant. -/
theorem triangle_constant_sum_squares (X Y Z M : ℝ × ℝ) :
  let d (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  (d Y Z = 10) →
  (M = ((Y.1 + Z.1) / 2, (Y.2 + Z.2) / 2)) →
  (d X M = 7) →
  ∃ (c : ℝ), ∀ (X' : ℝ × ℝ), d X' M = 7 → (d X' Y)^2 + (d X' Z)^2 = c :=
by sorry

end NUMINAMATH_CALUDE_triangle_constant_sum_squares_l2950_295036


namespace NUMINAMATH_CALUDE_mortdecai_mall_delivery_l2950_295096

/-- Represents the egg collection and distribution for Mortdecai in a week -/
structure EggDistribution where
  collected_per_day : ℕ  -- dozens of eggs collected on Tuesday and Thursday
  market_delivery : ℕ    -- dozens of eggs delivered to the market
  pie_usage : ℕ          -- dozens of eggs used for pie
  charity_donation : ℕ   -- dozens of eggs donated to charity

/-- Calculates the number of dozens of eggs delivered to the mall -/
def mall_delivery (ed : EggDistribution) : ℕ :=
  2 * ed.collected_per_day - (ed.market_delivery + ed.pie_usage + ed.charity_donation)

/-- Theorem stating that Mortdecai delivers 5 dozen eggs to the mall -/
theorem mortdecai_mall_delivery :
  let ed : EggDistribution := {
    collected_per_day := 8,
    market_delivery := 3,
    pie_usage := 4,
    charity_donation := 4
  }
  mall_delivery ed = 5 := by sorry

end NUMINAMATH_CALUDE_mortdecai_mall_delivery_l2950_295096


namespace NUMINAMATH_CALUDE_inequality_proof_l2950_295003

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 - b*c) / (2*a^2 + b*c) + (b^2 - c*a) / (2*b^2 + c*a) + (c^2 - a*b) / (2*c^2 + a*b) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2950_295003


namespace NUMINAMATH_CALUDE_min_sum_of_primes_l2950_295047

/-- Given distinct positive integers a and b, where 20a + 17b and 17a + 20b
    are both prime numbers, the minimum sum of these prime numbers is 296. -/
theorem min_sum_of_primes (a b : ℕ+) (h_distinct : a ≠ b)
  (h_prime1 : Nat.Prime (20 * a + 17 * b))
  (h_prime2 : Nat.Prime (17 * a + 20 * b)) :
  (20 * a + 17 * b) + (17 * a + 20 * b) ≥ 296 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_primes_l2950_295047


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l2950_295085

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (|x| - 1) / (x - 1) = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l2950_295085


namespace NUMINAMATH_CALUDE_exercise_book_count_l2950_295032

/-- Given a shop with pencils, pens, and exercise books in a specific ratio,
    calculate the number of exercise books based on the number of pencils. -/
theorem exercise_book_count (pencil_ratio : ℕ) (pen_ratio : ℕ) (book_ratio : ℕ) 
    (pencil_count : ℕ) (h1 : pencil_ratio = 10) (h2 : pen_ratio = 2) 
    (h3 : book_ratio = 3) (h4 : pencil_count = 120) : 
    (pencil_count / pencil_ratio) * book_ratio = 36 := by
  sorry

end NUMINAMATH_CALUDE_exercise_book_count_l2950_295032


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_intersection_A_B_empty_intersection_A_B_equals_A_l2950_295052

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

-- Theorem 1
theorem intersection_A_complement_B (m : ℝ) : 
  m = 3 → A ∩ (Set.univ \ B m) = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2
theorem intersection_A_B_empty (m : ℝ) : 
  A ∩ B m = ∅ ↔ m ≤ -2 := by sorry

-- Theorem 3
theorem intersection_A_B_equals_A (m : ℝ) : 
  A ∩ B m = A ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_intersection_A_B_empty_intersection_A_B_equals_A_l2950_295052


namespace NUMINAMATH_CALUDE_round_robin_cyclic_triples_l2950_295040

/-- Represents a round-robin tournament. -/
structure Tournament where
  teams : ℕ
  games_won : ℕ
  games_lost : ℕ

/-- Represents a cyclic triple in the tournament. -/
structure CyclicTriple where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The number of cyclic triples in the tournament. -/
def count_cyclic_triples (t : Tournament) : ℕ :=
  sorry

theorem round_robin_cyclic_triples :
  ∀ t : Tournament,
    t.teams = t.games_won + t.games_lost + 1 →
    t.games_won = 12 →
    t.games_lost = 8 →
    count_cyclic_triples t = 144 :=
  sorry

end NUMINAMATH_CALUDE_round_robin_cyclic_triples_l2950_295040


namespace NUMINAMATH_CALUDE_dirichlet_approximation_l2950_295075

theorem dirichlet_approximation (N : ℕ) (hN : N > 0) :
  ∃ (a b : ℕ), 1 ≤ b ∧ b ≤ N ∧ |a - b * Real.sqrt 2| ≤ 1 / N :=
by sorry

end NUMINAMATH_CALUDE_dirichlet_approximation_l2950_295075


namespace NUMINAMATH_CALUDE_range_of_x_minus_2y_l2950_295021

theorem range_of_x_minus_2y (x y : ℝ) (hx : -1 ≤ x ∧ x < 2) (hy : 0 < y ∧ y ≤ 1) :
  ∃ (z : ℝ), -3 ≤ z ∧ z < 2 ∧ ∃ (x' y' : ℝ), -1 ≤ x' ∧ x' < 2 ∧ 0 < y' ∧ y' ≤ 1 ∧ z = x' - 2*y' :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_x_minus_2y_l2950_295021


namespace NUMINAMATH_CALUDE_answer_key_combinations_l2950_295018

/-- Represents the number of answer choices for a multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- Represents the number of true-false questions -/
def true_false_questions : ℕ := 3

/-- Represents the number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 3

/-- Calculates the number of valid true-false combinations -/
def valid_true_false_combinations : ℕ := 2^true_false_questions - 2

/-- Calculates the number of multiple-choice combinations -/
def multiple_choice_combinations : ℕ := multiple_choice_options^multiple_choice_questions

/-- Theorem: The number of ways to create an answer key for the quiz is 384 -/
theorem answer_key_combinations : 
  valid_true_false_combinations * multiple_choice_combinations = 384 := by
  sorry


end NUMINAMATH_CALUDE_answer_key_combinations_l2950_295018


namespace NUMINAMATH_CALUDE_reconstruct_axes_and_unit_l2950_295092

-- Define the parabola
def parabola : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

-- Define the concept of constructible points
def constructible (p : ℝ × ℝ) : Prop := sorry

-- Define the concept of constructible lines
def constructibleLine (l : Set (ℝ × ℝ)) : Prop := sorry

-- Define the x-axis
def xAxis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define the y-axis
def yAxis : Set (ℝ × ℝ) := {p | p.1 = 0}

-- Define the unit point (1, 1)
def unitPoint : ℝ × ℝ := (1, 1)

-- Theorem stating that the coordinate axes and unit length can be reconstructed
theorem reconstruct_axes_and_unit : 
  ∃ (x y : Set (ℝ × ℝ)) (u : ℝ × ℝ),
    constructibleLine x ∧ 
    constructibleLine y ∧ 
    constructible u ∧
    x = xAxis ∧ 
    y = yAxis ∧ 
    u = unitPoint :=
  sorry

end NUMINAMATH_CALUDE_reconstruct_axes_and_unit_l2950_295092


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l2950_295023

theorem largest_angle_in_triangle (x : ℝ) : 
  x + 40 + 60 = 180 → 
  max x (max 40 60) = 80 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l2950_295023


namespace NUMINAMATH_CALUDE_max_product_constrained_l2950_295062

theorem max_product_constrained (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 8 * b = 48) :
  a * b ≤ 24 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 8 * b₀ = 48 ∧ a₀ * b₀ = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_l2950_295062


namespace NUMINAMATH_CALUDE_fraction_simplification_l2950_295001

theorem fraction_simplification (y : ℝ) (h : y = 5) : 
  (y^4 - 8*y^2 + 16) / (y^2 - 4) = 21 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2950_295001


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2950_295095

theorem arithmetic_calculation : 3^2 * 4 + 5 * (6 + 3) - 15 / 3 = 76 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2950_295095


namespace NUMINAMATH_CALUDE_wheel_probability_l2950_295002

theorem wheel_probability :
  let total_ratio : ℕ := 6 + 2 + 1 + 4
  let red_ratio : ℕ := 6
  let blue_ratio : ℕ := 1
  let target_ratio : ℕ := red_ratio + blue_ratio
  (target_ratio : ℚ) / total_ratio = 7 / 13 :=
by sorry

end NUMINAMATH_CALUDE_wheel_probability_l2950_295002


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_calculation_l2950_295033

/-- The cost of the Ferris wheel ride -/
def ferris_wheel_cost : ℝ := 2.0

/-- The cost of the roller coaster ride -/
def roller_coaster_cost : ℝ := 7.0

/-- The discount for multiple rides -/
def multiple_ride_discount : ℝ := 1.0

/-- The value of the newspaper coupon -/
def coupon_value : ℝ := 1.0

/-- The total number of tickets needed for both rides -/
def total_tickets_needed : ℝ := 7.0

theorem ferris_wheel_cost_calculation :
  ferris_wheel_cost + roller_coaster_cost - multiple_ride_discount - coupon_value = total_tickets_needed :=
sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_calculation_l2950_295033


namespace NUMINAMATH_CALUDE_teacher_selection_and_assignment_l2950_295056

-- Define the number of male and female teachers
def num_male_teachers : ℕ := 5
def num_female_teachers : ℕ := 4

-- Define the number of male and female teachers to be selected
def selected_male_teachers : ℕ := 3
def selected_female_teachers : ℕ := 2

-- Define the total number of teachers to be selected
def total_selected_teachers : ℕ := selected_male_teachers + selected_female_teachers

-- Define the number of villages
def num_villages : ℕ := 5

-- Theorem statement
theorem teacher_selection_and_assignment :
  (Nat.choose num_male_teachers selected_male_teachers) *
  (Nat.choose num_female_teachers selected_female_teachers) *
  (Nat.factorial total_selected_teachers) = 7200 :=
sorry

end NUMINAMATH_CALUDE_teacher_selection_and_assignment_l2950_295056


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l2950_295048

def euler_family_ages : List ℕ := [8, 8, 12, 10, 10, 16]

theorem euler_family_mean_age :
  (euler_family_ages.sum : ℚ) / euler_family_ages.length = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l2950_295048


namespace NUMINAMATH_CALUDE_investment_growth_l2950_295053

/-- The initial investment amount -/
def P : ℝ := 248.52

/-- The interest rate as a decimal -/
def r : ℝ := 0.12

/-- The number of years -/
def n : ℕ := 6

/-- The final amount -/
def A : ℝ := 500

/-- Theorem stating that the initial investment P, when compounded annually
    at rate r for n years, results in approximately the final amount A -/
theorem investment_growth (ε : ℝ) (h_ε : ε > 0) : 
  |P * (1 + r)^n - A| < ε := by
  sorry


end NUMINAMATH_CALUDE_investment_growth_l2950_295053


namespace NUMINAMATH_CALUDE_min_cost_trees_l2950_295016

/-- The cost function for purchasing trees -/
def cost_function (x : ℕ) : ℕ := 20 * x + 12000

/-- The constraint on the number of cypress trees -/
def cypress_constraint (x : ℕ) : Prop := x ≥ 3 * (150 - x)

/-- The total number of trees to be purchased -/
def total_trees : ℕ := 150

/-- The theorem stating the minimum cost and optimal purchase -/
theorem min_cost_trees :
  ∃ (x : ℕ), 
    x ≤ total_trees ∧
    cypress_constraint x ∧
    (∀ (y : ℕ), y ≤ total_trees → cypress_constraint y → cost_function x ≤ cost_function y) ∧
    x = 113 ∧
    cost_function x = 14260 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_trees_l2950_295016


namespace NUMINAMATH_CALUDE_remaining_potatoes_l2950_295051

/-- Given an initial number of potatoes and a number of eaten potatoes,
    prove that the remaining number of potatoes is equal to their difference. -/
theorem remaining_potatoes (initial : ℕ) (eaten : ℕ) :
  initial ≥ eaten → initial - eaten = initial - eaten :=
by sorry

end NUMINAMATH_CALUDE_remaining_potatoes_l2950_295051


namespace NUMINAMATH_CALUDE_nine_workers_needed_workers_to_build_nine_cars_l2950_295078

/-- The number of workers needed to build a given number of cars in 9 days -/
def workers_needed (cars : ℕ) : ℕ :=
  cars

theorem nine_workers_needed : workers_needed 9 = 9 :=
by
  -- Proof goes here
  sorry

/-- Given condition: 7 workers can build 7 cars in 9 days -/
axiom seven_workers_seven_cars : workers_needed 7 = 7

-- The main theorem
theorem workers_to_build_nine_cars : ∃ w : ℕ, workers_needed 9 = w ∧ w = 9 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_nine_workers_needed_workers_to_build_nine_cars_l2950_295078


namespace NUMINAMATH_CALUDE_adjacent_probability_in_row_of_five_l2950_295024

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- The probability of two specific people sitting adjacent in a row of 5 people -/
theorem adjacent_probability_in_row_of_five :
  let total_arrangements := factorial 5
  let adjacent_arrangements := 2 * factorial 4
  (adjacent_arrangements : ℚ) / total_arrangements = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_adjacent_probability_in_row_of_five_l2950_295024


namespace NUMINAMATH_CALUDE_mom_tshirt_count_l2950_295077

/-- The number of t-shirts in a package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom buys -/
def packages_bought : ℕ := 71

/-- Theorem: Mom will have 426 white t-shirts -/
theorem mom_tshirt_count : shirts_per_package * packages_bought = 426 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_count_l2950_295077


namespace NUMINAMATH_CALUDE_fraction_square_product_l2950_295055

theorem fraction_square_product : (8 / 9 : ℚ)^2 * (1 / 3 : ℚ)^2 = 64 / 729 := by sorry

end NUMINAMATH_CALUDE_fraction_square_product_l2950_295055


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2950_295015

/-- Given that k, -1, and b form an arithmetic sequence,
    prove that the line y = kx + b passes through (1, -2) for all k. -/
theorem line_passes_through_fixed_point (k b : ℝ) :
  ((-1) = (k + b) / 2) →
  ∀ (x y : ℝ), y = k * x + b → (x = 1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2950_295015


namespace NUMINAMATH_CALUDE_sqrt_comparison_l2950_295063

theorem sqrt_comparison : Real.sqrt 7 - Real.sqrt 3 < Real.sqrt 6 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_comparison_l2950_295063


namespace NUMINAMATH_CALUDE_stock_price_after_three_years_l2950_295064

theorem stock_price_after_three_years (initial_price : ℝ) :
  initial_price = 120 →
  let price_after_year1 := initial_price * 1.5
  let price_after_year2 := price_after_year1 * 0.7
  let price_after_year3 := price_after_year2 * 1.2
  price_after_year3 = 151.2 := by
sorry

end NUMINAMATH_CALUDE_stock_price_after_three_years_l2950_295064


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l2950_295017

/-- The intersection points of the parabolas y = (x + 2)^2 and x + 2 = (y - 1)^2 lie on a circle with r^2 = 2 -/
theorem intersection_points_on_circle : ∃ (c : ℝ × ℝ) (r : ℝ),
  (∀ x y : ℝ, y = (x + 2)^2 ∧ x + 2 = (y - 1)^2 →
    (x - c.1)^2 + (y - c.2)^2 = r^2) ∧
  r^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l2950_295017


namespace NUMINAMATH_CALUDE_replaced_person_weight_l2950_295097

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating the weight of the replaced person under the given conditions -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (5/2) 85 = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l2950_295097


namespace NUMINAMATH_CALUDE_shopkeeper_milk_ounces_l2950_295091

/-- Calculates the total amount of milk in fluid ounces bought by a shopkeeper -/
theorem shopkeeper_milk_ounces (packets : ℕ) (ml_per_packet : ℕ) (ml_per_fl_oz : ℕ) 
    (h1 : packets = 150)
    (h2 : ml_per_packet = 250)
    (h3 : ml_per_fl_oz = 30) : 
  (packets * ml_per_packet) / ml_per_fl_oz = 1250 := by
  sorry


end NUMINAMATH_CALUDE_shopkeeper_milk_ounces_l2950_295091


namespace NUMINAMATH_CALUDE_emilys_spending_l2950_295054

theorem emilys_spending (X : ℝ) 
  (friday : X ≥ 0)
  (saturday : 2 * X ≥ 0)
  (sunday : 3 * X ≥ 0)
  (total : X + 2 * X + 3 * X = 120) : X = 20 := by
  sorry

end NUMINAMATH_CALUDE_emilys_spending_l2950_295054


namespace NUMINAMATH_CALUDE_morse_high_school_seniors_l2950_295084

/-- The number of seniors at Morse High School -/
def num_seniors : ℕ := 300

/-- The number of students in the lower grades (freshmen, sophomores, and juniors) -/
def num_lower_grades : ℕ := 900

/-- The percentage of seniors who have cars -/
def senior_car_percentage : ℚ := 1/2

/-- The percentage of lower grade students who have cars -/
def lower_grade_car_percentage : ℚ := 1/10

/-- The percentage of all students who have cars -/
def total_car_percentage : ℚ := 1/5

theorem morse_high_school_seniors :
  (num_seniors * senior_car_percentage + num_lower_grades * lower_grade_car_percentage : ℚ) = 
  ((num_seniors + num_lower_grades) * total_car_percentage : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_morse_high_school_seniors_l2950_295084


namespace NUMINAMATH_CALUDE_fraction_equality_expression_equality_l2950_295007

-- Problem 1
theorem fraction_equality : (2021 * 2023) / (2022^2 - 1) = 1 := by sorry

-- Problem 2
theorem expression_equality : 2 * 101^2 + 2 * 101 * 98 + 2 * 49^2 = 45000 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_expression_equality_l2950_295007


namespace NUMINAMATH_CALUDE_kenneth_fabric_price_l2950_295034

/-- The price Kenneth paid for an oz of fabric -/
def kenneth_price : ℝ := 40

/-- The amount of fabric Kenneth bought in oz -/
def kenneth_amount : ℝ := 700

/-- The ratio of fabric Nicholas bought compared to Kenneth -/
def nicholas_ratio : ℝ := 6

/-- The additional amount Nicholas paid compared to Kenneth -/
def price_difference : ℝ := 140000

theorem kenneth_fabric_price :
  kenneth_price * kenneth_amount * nicholas_ratio =
  kenneth_price * kenneth_amount + price_difference :=
by sorry

end NUMINAMATH_CALUDE_kenneth_fabric_price_l2950_295034


namespace NUMINAMATH_CALUDE_smallest_valid_number_l2950_295041

def is_valid (n : ℕ) : Prop :=
  ∀ k : ℕ, 2 ≤ k → k ≤ 12 → n % k = k - 1

theorem smallest_valid_number : 
  (is_valid 27719) ∧ (∀ m : ℕ, m < 27719 → ¬(is_valid m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l2950_295041


namespace NUMINAMATH_CALUDE_stamps_per_page_l2950_295057

theorem stamps_per_page (book1 book2 book3 : ℕ) 
  (h1 : book1 = 945) 
  (h2 : book2 = 1260) 
  (h3 : book3 = 1575) : 
  Nat.gcd book1 (Nat.gcd book2 book3) = 315 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_page_l2950_295057


namespace NUMINAMATH_CALUDE_difference_is_10q_minus_10_l2950_295009

/-- The difference in dimes between two people's money, given their quarter amounts -/
def difference_in_dimes (charles_quarters richard_quarters : ℤ) : ℚ :=
  2.5 * (charles_quarters - richard_quarters)

/-- Proof that the difference in dimes between Charles and Richard's money is 10(q - 1) -/
theorem difference_is_10q_minus_10 (q : ℤ) :
  difference_in_dimes (5 * q + 1) (q + 5) = 10 * (q - 1) := by
  sorry

#check difference_is_10q_minus_10

end NUMINAMATH_CALUDE_difference_is_10q_minus_10_l2950_295009


namespace NUMINAMATH_CALUDE_area_enclosed_by_curves_l2950_295010

-- Define the curves
def curve1 (x y : ℝ) : Prop := y^2 = x
def curve2 (x y : ℝ) : Prop := y = x^2

-- Define the enclosed area
noncomputable def enclosed_area : ℝ := sorry

-- Theorem statement
theorem area_enclosed_by_curves : enclosed_area = 1/3 := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_curves_l2950_295010


namespace NUMINAMATH_CALUDE_washing_machine_payment_l2950_295067

theorem washing_machine_payment (remaining_payment : ℝ) (remaining_percentage : ℝ) 
  (part_payment_percentage : ℝ) (h1 : remaining_payment = 3683.33) 
  (h2 : remaining_percentage = 85) (h3 : part_payment_percentage = 15) : 
  (part_payment_percentage / 100) * (remaining_payment / (remaining_percentage / 100)) = 649.95 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_payment_l2950_295067


namespace NUMINAMATH_CALUDE_expression_value_l2950_295072

theorem expression_value (a b : ℝ) 
  (h1 : 10 * a^2 - 3 * b^2 + 5 * a * b = 0) 
  (h2 : 9 * a^2 - b^2 ≠ 0) : 
  (2 * a - b) / (3 * a - b) + (5 * b - a) / (3 * a + b) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2950_295072


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2950_295020

theorem nested_fraction_equality : 
  2 - (1 / (2 + (1 / (2 - (1 / 2))))) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2950_295020


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2950_295076

theorem min_value_sum_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^x * 3^(3*y))) :
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 1/(3*b) ≥ 1/x + 1/(3*y)) → 1/x + 1/(3*y) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2950_295076


namespace NUMINAMATH_CALUDE_subtracted_number_l2950_295026

theorem subtracted_number (x y : ℤ) (h1 : x = 30) (h2 : 8 * x - y = 102) : y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l2950_295026


namespace NUMINAMATH_CALUDE_gumball_difference_l2950_295090

/-- The number of gumballs Hector purchased -/
def total_gumballs : ℕ := 45

/-- The number of gumballs Hector gave to Todd -/
def todd_gumballs : ℕ := 4

/-- The number of gumballs Hector gave to Alisha -/
def alisha_gumballs : ℕ := 2 * todd_gumballs

/-- The number of gumballs Hector had remaining -/
def remaining_gumballs : ℕ := 6

/-- The number of gumballs Hector gave to Bobby -/
def bobby_gumballs : ℕ := total_gumballs - todd_gumballs - alisha_gumballs - remaining_gumballs

theorem gumball_difference : 
  4 * alisha_gumballs - bobby_gumballs = 5 := by sorry

end NUMINAMATH_CALUDE_gumball_difference_l2950_295090


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2950_295068

theorem fraction_evaluation : (5 * 6 - 3 * 4) / (6 + 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2950_295068


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2950_295058

theorem solution_set_equivalence (x y : ℝ) :
  (x^2 + 3*x*y + 2*y^2) * (x^2*y^2 - 1) = 0 ↔
  y = -x/2 ∨ y = -x ∨ y = -1/x ∨ y = 1/x :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2950_295058


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l2950_295025

/-- Calculates the sampling interval for systematic sampling. -/
def samplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The sampling interval for a population of 1500 and sample size of 30 is 50. -/
theorem systematic_sampling_interval :
  samplingInterval 1500 30 = 50 := by
  sorry

#eval samplingInterval 1500 30

end NUMINAMATH_CALUDE_systematic_sampling_interval_l2950_295025


namespace NUMINAMATH_CALUDE_trivia_team_size_l2950_295035

theorem trivia_team_size :
  let members_absent : ℝ := 2
  let total_score : ℝ := 6
  let score_per_member : ℝ := 2
  let members_present : ℝ := total_score / score_per_member
  let total_members : ℝ := members_present + members_absent
  total_members = 5 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_size_l2950_295035


namespace NUMINAMATH_CALUDE_payment_function_correct_l2950_295070

/-- Represents the payment function for book purchases with a discount. -/
def payment_function (x : ℝ) : ℝ :=
  20 * x + 100

/-- Theorem stating the correctness of the payment function. -/
theorem payment_function_correct (x : ℝ) (h : x > 20) :
  payment_function x = (x - 20) * (25 * 0.8) + 20 * 25 := by
  sorry

#check payment_function_correct

end NUMINAMATH_CALUDE_payment_function_correct_l2950_295070


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2950_295019

theorem z_in_first_quadrant (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2950_295019


namespace NUMINAMATH_CALUDE_jaylen_cucumbers_count_l2950_295098

/-- The number of cucumbers Jaylen has -/
def jaylen_cucumbers (jaylen_carrots jaylen_bell_peppers jaylen_green_beans kristin_bell_peppers kristin_green_beans jaylen_total : ℕ) : ℕ :=
  jaylen_total - (jaylen_carrots + jaylen_bell_peppers + jaylen_green_beans)

theorem jaylen_cucumbers_count :
  ∀ (jaylen_carrots jaylen_bell_peppers jaylen_green_beans kristin_bell_peppers kristin_green_beans jaylen_total : ℕ),
  jaylen_carrots = 5 →
  jaylen_bell_peppers = 2 * kristin_bell_peppers →
  jaylen_green_beans = kristin_green_beans / 2 - 3 →
  kristin_bell_peppers = 2 →
  kristin_green_beans = 20 →
  jaylen_total = 18 →
  jaylen_cucumbers jaylen_carrots jaylen_bell_peppers jaylen_green_beans kristin_bell_peppers kristin_green_beans jaylen_total = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_jaylen_cucumbers_count_l2950_295098


namespace NUMINAMATH_CALUDE_regression_analysis_appropriate_for_height_weight_l2950_295028

/-- Represents a statistical analysis method -/
inductive AnalysisMethod
  | ResidualAnalysis
  | RegressionAnalysis
  | IsoplethBarChart
  | IndependenceTest

/-- Represents a variable in the context of statistical analysis -/
structure Variable where
  name : String

/-- Represents a relationship between two variables -/
structure Relationship where
  var1 : Variable
  var2 : Variable
  correlated : Bool

/-- Determines if a given analysis method is appropriate for analyzing a relationship between two variables -/
def is_appropriate_method (method : AnalysisMethod) (rel : Relationship) : Prop :=
  method = AnalysisMethod.RegressionAnalysis ∧ rel.correlated = true

/-- Main theorem: Regression analysis is the appropriate method for analyzing the relationship between height and weight -/
theorem regression_analysis_appropriate_for_height_weight :
  let height : Variable := ⟨"height"⟩
  let weight : Variable := ⟨"weight"⟩
  let height_weight_rel : Relationship := ⟨height, weight, true⟩
  is_appropriate_method AnalysisMethod.RegressionAnalysis height_weight_rel :=
by
  sorry


end NUMINAMATH_CALUDE_regression_analysis_appropriate_for_height_weight_l2950_295028


namespace NUMINAMATH_CALUDE_garden_comparison_l2950_295088

-- Define the dimensions of Karl's garden
def karl_length : ℕ := 30
def karl_width : ℕ := 40

-- Define the dimensions of Makenna's garden
def makenna_side : ℕ := 35

-- Theorem to prove the comparison of areas and perimeters
theorem garden_comparison :
  (makenna_side * makenna_side - karl_length * karl_width = 25) ∧
  (2 * (karl_length + karl_width) = 4 * makenna_side) :=
by sorry

end NUMINAMATH_CALUDE_garden_comparison_l2950_295088


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l2950_295074

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1)

theorem fifth_term_of_arithmetic_sequence
  (a d : ℝ)
  (h1 : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 = 10)
  (h2 : arithmetic_sequence a d 1 + arithmetic_sequence a d 3 = 8) :
  arithmetic_sequence a d 5 = 7 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l2950_295074


namespace NUMINAMATH_CALUDE_tangent_line_and_positivity_l2950_295066

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x + 1) - a * Real.log x + a

theorem tangent_line_and_positivity (a : ℝ) (h : a > 0) :
  (∃ m b : ℝ, ∀ x y : ℝ, y = f 1 x → m * x - y + b = 0) ∧
  (∀ x : ℝ, x > 0 → f a x > 0) ↔ 0 < a ∧ a < Real.exp 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_positivity_l2950_295066


namespace NUMINAMATH_CALUDE_new_video_card_cost_l2950_295065

theorem new_video_card_cost (initial_cost : ℕ) (old_card_sale : ℕ) (total_spent : ℕ) : 
  initial_cost = 1200 →
  old_card_sale = 300 →
  total_spent = 1400 →
  total_spent - (initial_cost - old_card_sale) = 500 := by
sorry

end NUMINAMATH_CALUDE_new_video_card_cost_l2950_295065


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l2950_295042

theorem quadratic_form_ratio (j : ℝ) (c p q : ℝ) : 
  8 * j^2 - 6 * j + 16 = c * (j + p)^2 + q → q / p = -119 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l2950_295042


namespace NUMINAMATH_CALUDE_sock_order_ratio_l2950_295008

/-- Represents the number of pairs of socks --/
structure SockOrder where
  green : ℕ
  red : ℕ

/-- Represents the price of socks --/
structure SockPrice where
  red : ℝ
  green : ℝ

/-- Calculates the total cost of a sock order --/
def totalCost (order : SockOrder) (price : SockPrice) : ℝ :=
  order.green * price.green + order.red * price.red

theorem sock_order_ratio (original : SockOrder) (price : SockPrice) :
  original.green = 6 →
  price.green = 3 * price.red →
  let interchanged : SockOrder := ⟨original.red, original.green⟩
  totalCost interchanged price = 1.2 * totalCost original price →
  2 * original.red = 3 * original.green := by
  sorry

end NUMINAMATH_CALUDE_sock_order_ratio_l2950_295008


namespace NUMINAMATH_CALUDE_bret_nap_time_l2950_295083

/-- Calculates the remaining time for napping during a train ride -/
def remaining_nap_time (total_duration reading_time eating_time movie_time : ℕ) : ℕ :=
  total_duration - (reading_time + eating_time + movie_time)

/-- Theorem: Given Bret's 9-hour train ride and his activities, he has 3 hours left for napping -/
theorem bret_nap_time :
  remaining_nap_time 9 2 1 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bret_nap_time_l2950_295083


namespace NUMINAMATH_CALUDE_two_digit_numbers_product_sum_l2950_295038

theorem two_digit_numbers_product_sum (x y : ℕ) : 
  (10 ≤ x ∧ x < 100) ∧ 
  (10 ≤ y ∧ y < 100) ∧ 
  (2000 ≤ x * y ∧ x * y < 3000) ∧ 
  (100 ≤ x + y ∧ x + y < 1000) ∧ 
  (x * y = 2000 + (x + y)) →
  ((x = 24 ∧ y = 88) ∨ (x = 88 ∧ y = 24) ∨ (x = 30 ∧ y = 70) ∨ (x = 70 ∧ y = 30)) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_product_sum_l2950_295038


namespace NUMINAMATH_CALUDE_solution_set1_correct_solution_set2_correct_l2950_295073

open Set

-- Define the solution sets
def solution_set1 : Set ℝ := Iic (-3) ∪ Ici 1
def solution_set2 : Set ℝ := Ico (-3) 1 ∪ Ioc 3 7

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := (4 - x) / (x^2 + x + 1) ≤ 1
def inequality2 (x : ℝ) : Prop := 1 < |x - 2| ∧ |x - 2| ≤ 5

-- Theorem statements
theorem solution_set1_correct :
  ∀ x : ℝ, x ∈ solution_set1 ↔ inequality1 x :=
sorry

theorem solution_set2_correct :
  ∀ x : ℝ, x ∈ solution_set2 ↔ inequality2 x :=
sorry

end NUMINAMATH_CALUDE_solution_set1_correct_solution_set2_correct_l2950_295073


namespace NUMINAMATH_CALUDE_no_prime_of_form_3811_11_l2950_295013

def a (n : ℕ) : ℕ := 3 * 10^(n+1) + 8 * 10^n + (10^n - 1) / 9

theorem no_prime_of_form_3811_11 (n : ℕ) (h : n ≥ 1) : ¬ Nat.Prime (a n) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_of_form_3811_11_l2950_295013


namespace NUMINAMATH_CALUDE_complement_angle_l2950_295004

theorem complement_angle (A : ℝ) (h : A = 25) : 90 - A = 65 := by
  sorry

end NUMINAMATH_CALUDE_complement_angle_l2950_295004


namespace NUMINAMATH_CALUDE_solve_equation_l2950_295030

theorem solve_equation (q r x : ℚ) : 
  (5 / 6 : ℚ) = q / 90 ∧ 
  (5 / 6 : ℚ) = (q + r) / 102 ∧ 
  (5 / 6 : ℚ) = (x - r) / 150 → 
  x = 135 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l2950_295030


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2950_295014

/-- A quadratic function passing through two points with constrained x values -/
def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

theorem quadratic_function_properties :
  ∃ (a b : ℝ),
    (quadratic_function a b 0 = 6) ∧
    (quadratic_function a b 1 = 5) ∧
    (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 →
      (quadratic_function a b x = x^2 - 2*x + 6) ∧
      (quadratic_function a b x ≥ 5) ∧
      (quadratic_function a b x ≤ 14) ∧
      (quadratic_function a b 1 = 5) ∧
      (quadratic_function a b (-2) = 14)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2950_295014


namespace NUMINAMATH_CALUDE_min_sum_m_n_l2950_295081

theorem min_sum_m_n (m n : ℕ+) (h : 108 * m = n ^ 3) : 
  ∀ (k l : ℕ+), 108 * k = l ^ 3 → m + n ≤ k + l :=
sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l2950_295081


namespace NUMINAMATH_CALUDE_expression_simplification_l2950_295045

/-- Given that |2+y|+(x-1)^2=0, prove that 5x^2*y-[3x*y^2-2(3x*y^2-7/2*x^2*y)] = 16 -/
theorem expression_simplification (x y : ℝ) 
  (h : |2 + y| + (x - 1)^2 = 0) : 
  5*x^2*y - (3*x*y^2 - 2*(3*x*y^2 - 7/2*x^2*y)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2950_295045


namespace NUMINAMATH_CALUDE_periodic_sum_implies_constant_l2950_295044

/-- A function is periodic with period a if f(x + a) = f(x) for all x --/
def IsPeriodic (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a ≠ 0 ∧ ∀ x, f (x + a) = f x

theorem periodic_sum_implies_constant
  (f g : ℝ → ℝ) (a b : ℝ)
  (hfa : IsPeriodic f a)
  (hgb : IsPeriodic g b)
  (ha_rat : ℚ)
  (hb_irrat : Irrational b)
  (h_sum_periodic : ∃ c, IsPeriodic (f + g) c) :
  (∃ k, ∀ x, f x = k) ∨ (∃ k, ∀ x, g x = k) := by
  sorry

end NUMINAMATH_CALUDE_periodic_sum_implies_constant_l2950_295044


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2950_295029

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² + 5x = 0 -/
def f (x : ℝ) : ℝ := x^2 + 5*x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l2950_295029


namespace NUMINAMATH_CALUDE_triangle_point_distance_l2950_295049

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point on a line segment
def PointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define the angle between two vectors
def Angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem triangle_point_distance (ABC : Triangle) (D E : ℝ × ℝ) :
  -- Given conditions
  (ABC.A.1 - ABC.B.1)^2 + (ABC.A.2 - ABC.B.2)^2 = 17^2 →
  (ABC.B.1 - ABC.C.1)^2 + (ABC.B.2 - ABC.C.2)^2 = 19^2 →
  (ABC.C.1 - ABC.A.1)^2 + (ABC.C.2 - ABC.A.2)^2 = 16^2 →
  PointOnSegment D ABC.B ABC.C →
  PointOnSegment E ABC.B ABC.C →
  (D.1 - ABC.B.1)^2 + (D.2 - ABC.B.2)^2 = 7^2 →
  Angle ABC.B ABC.A E = Angle ABC.C ABC.A D →
  -- Conclusion
  (E.1 - D.1)^2 + (E.2 - D.2)^2 = (-251/41)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_point_distance_l2950_295049


namespace NUMINAMATH_CALUDE_smallest_b_for_inequality_l2950_295039

theorem smallest_b_for_inequality : ∃ b : ℕ, (∀ k : ℕ, 27^k > 3^24 → k ≥ b) ∧ 27^b > 3^24 :=
  sorry

end NUMINAMATH_CALUDE_smallest_b_for_inequality_l2950_295039


namespace NUMINAMATH_CALUDE_negative_expression_l2950_295000

theorem negative_expression : 
  (-(-3) > 0) ∧ (-3^2 < 0) ∧ ((-3)^2 > 0) ∧ (|(-3)| > 0) :=
by sorry


end NUMINAMATH_CALUDE_negative_expression_l2950_295000


namespace NUMINAMATH_CALUDE_area_between_squares_l2950_295061

/-- The area of the region between two squares, where a smaller square is entirely contained within a larger square -/
theorem area_between_squares (larger_side smaller_side : ℝ) 
  (h1 : larger_side = 8) 
  (h2 : smaller_side = 4) 
  (h3 : smaller_side ≤ larger_side) : 
  larger_side ^ 2 - smaller_side ^ 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_area_between_squares_l2950_295061


namespace NUMINAMATH_CALUDE_irrational_among_given_numbers_l2950_295089

theorem irrational_among_given_numbers : 
  (∃ q : ℚ, |3 / (-8)| = q) ∧ 
  (∃ q : ℚ, |22 / 7| = q) ∧ 
  (∃ q : ℚ, 3.14 = q) ∧ 
  (∀ q : ℚ, |Real.sqrt 3| ≠ q) := by
  sorry

end NUMINAMATH_CALUDE_irrational_among_given_numbers_l2950_295089


namespace NUMINAMATH_CALUDE_function_range_condition_l2950_295060

/-- Given functions f and g, prove that m ≥ 3/2 under specified conditions -/
theorem function_range_condition (m : ℝ) (h_m : m > 0) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₂ ∈ Set.Icc (-1 : ℝ) 2, 
    ((1/2 : ℝ) ^ x₁) = m * x₂ - 1) → 
  m ≥ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_function_range_condition_l2950_295060


namespace NUMINAMATH_CALUDE_tan_2016_in_terms_of_sin_36_l2950_295012

theorem tan_2016_in_terms_of_sin_36 (a : ℝ) (h : Real.sin (36 * π / 180) = a) :
  Real.tan (2016 * π / 180) = a / Real.sqrt (1 - a^2) := by
  sorry

end NUMINAMATH_CALUDE_tan_2016_in_terms_of_sin_36_l2950_295012


namespace NUMINAMATH_CALUDE_abs_neg_two_plus_two_l2950_295022

theorem abs_neg_two_plus_two : |(-2 : ℤ)| + 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_plus_two_l2950_295022


namespace NUMINAMATH_CALUDE_remainder_product_l2950_295027

theorem remainder_product (n : ℤ) : n % 24 = 19 → (n % 3) * (n % 8) = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_product_l2950_295027


namespace NUMINAMATH_CALUDE_find_c_l2950_295031

def f (x : ℝ) : ℝ := x - 2

def F (x y : ℝ) : ℝ := y^2 + x

theorem find_c (b : ℝ) : ∃ c : ℝ, c = F 3 (f b) ∧ c = 199 := by
  sorry

end NUMINAMATH_CALUDE_find_c_l2950_295031


namespace NUMINAMATH_CALUDE_third_square_side_length_l2950_295099

/-- Given three squares with perimeters 60 cm, 48 cm, and 36 cm respectively,
    if the area of the third square is equal to the difference of the areas of the first two squares,
    then the side length of the third square is 9 cm. -/
theorem third_square_side_length 
  (s1 s2 s3 : ℝ) 
  (h1 : 4 * s1 = 60) 
  (h2 : 4 * s2 = 48) 
  (h3 : 4 * s3 = 36) 
  (h4 : s3^2 = s1^2 - s2^2) : 
  s3 = 9 := by
sorry

end NUMINAMATH_CALUDE_third_square_side_length_l2950_295099


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l2950_295087

/-- The ratio of a cone's height to its radius when its volume is one-third of a sphere with the same radius -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3) * ((4 / 3) * Real.pi * r^3) = (1 / 3) * Real.pi * r^2 * h → h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l2950_295087
