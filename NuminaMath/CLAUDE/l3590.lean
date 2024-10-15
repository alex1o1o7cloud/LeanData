import Mathlib

namespace NUMINAMATH_CALUDE_total_cinnamon_swirls_l3590_359030

/-- The number of people eating cinnamon swirls -/
def num_people : ℕ := 3

/-- The number of pieces Jane ate -/
def janes_pieces : ℕ := 4

/-- Theorem: The total number of cinnamon swirl pieces prepared is 12 -/
theorem total_cinnamon_swirls :
  ∀ (pieces_per_person : ℕ),
  (pieces_per_person = janes_pieces) →
  (num_people * pieces_per_person = 12) :=
by sorry

end NUMINAMATH_CALUDE_total_cinnamon_swirls_l3590_359030


namespace NUMINAMATH_CALUDE_subscription_ratio_l3590_359024

/-- Represents the number of magazine subscriptions sold to different people --/
structure Subscriptions where
  parents : ℕ
  grandfather : ℕ
  nextDoorNeighbor : ℕ
  otherNeighbor : ℕ

/-- Calculates the total earnings from selling subscriptions --/
def totalEarnings (s : Subscriptions) (pricePerSubscription : ℕ) : ℕ :=
  (s.parents + s.grandfather + s.nextDoorNeighbor + s.otherNeighbor) * pricePerSubscription

/-- Theorem stating the ratio of subscriptions sold to other neighbor vs next-door neighbor --/
theorem subscription_ratio (s : Subscriptions) (pricePerSubscription totalEarned : ℕ) :
  s.parents = 4 →
  s.grandfather = 1 →
  s.nextDoorNeighbor = 2 →
  pricePerSubscription = 5 →
  totalEarnings s pricePerSubscription = totalEarned →
  totalEarned = 55 →
  s.otherNeighbor = 2 * s.nextDoorNeighbor :=
by sorry


end NUMINAMATH_CALUDE_subscription_ratio_l3590_359024


namespace NUMINAMATH_CALUDE_solve_equations_l3590_359026

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x - 4 = 2 * x + 5
def equation2 (x : ℝ) : Prop := (x - 3) / 4 - (2 * x + 1) / 2 = 1

-- State the theorem
theorem solve_equations :
  (∃! x : ℝ, equation1 x) ∧ (∃! x : ℝ, equation2 x) ∧
  (∀ x : ℝ, equation1 x → x = 9) ∧
  (∀ x : ℝ, equation2 x → x = -6) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l3590_359026


namespace NUMINAMATH_CALUDE_enemies_left_undefeated_l3590_359050

theorem enemies_left_undefeated 
  (total_enemies : ℕ) 
  (points_per_enemy : ℕ) 
  (points_earned : ℕ) : 
  total_enemies = 6 → 
  points_per_enemy = 3 → 
  points_earned = 12 → 
  total_enemies - (points_earned / points_per_enemy) = 2 := by
sorry

end NUMINAMATH_CALUDE_enemies_left_undefeated_l3590_359050


namespace NUMINAMATH_CALUDE_binomial_coefficient_30_3_l3590_359011

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_30_3_l3590_359011


namespace NUMINAMATH_CALUDE_expenditure_difference_l3590_359093

theorem expenditure_difference 
  (original_price : ℝ) 
  (required_amount : ℝ) 
  (price_increase_percentage : ℝ) 
  (purchased_amount_percentage : ℝ) :
  price_increase_percentage = 40 →
  purchased_amount_percentage = 62 →
  let new_price := original_price * (1 + price_increase_percentage / 100)
  let new_amount := required_amount * (purchased_amount_percentage / 100)
  let original_expenditure := original_price * required_amount
  let new_expenditure := new_price * new_amount
  let difference := new_expenditure - original_expenditure
  difference / original_expenditure = -0.132 :=
by sorry

end NUMINAMATH_CALUDE_expenditure_difference_l3590_359093


namespace NUMINAMATH_CALUDE_power_multiplication_l3590_359082

theorem power_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3590_359082


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3590_359023

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3590_359023


namespace NUMINAMATH_CALUDE_smallest_k_cube_sum_multiple_360_k_38_cube_sum_multiple_360_smallest_k_is_38_l3590_359031

theorem smallest_k_cube_sum_multiple_360 : 
  ∀ k : ℕ, k > 0 → (k * (k + 1) / 2)^2 % 360 = 0 → k ≥ 38 :=
by sorry

theorem k_38_cube_sum_multiple_360 : 
  (38 * (38 + 1) / 2)^2 % 360 = 0 :=
by sorry

theorem smallest_k_is_38 :
  ∀ k : ℕ, k > 0 → (k * (k + 1) / 2)^2 % 360 = 0 → k ≥ 38 ∧ 
  (38 * (38 + 1) / 2)^2 % 360 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_cube_sum_multiple_360_k_38_cube_sum_multiple_360_smallest_k_is_38_l3590_359031


namespace NUMINAMATH_CALUDE_triangle_area_l3590_359071

def a : Fin 2 → ℝ := ![5, 1]
def b : Fin 2 → ℝ := ![2, 4]

theorem triangle_area : 
  (1/2 : ℝ) * |Matrix.det ![a, b]| = 9 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3590_359071


namespace NUMINAMATH_CALUDE_binomial_variance_l3590_359072

/-- A binomial distribution with parameter p -/
structure BinomialDistribution (p : ℝ) where
  (h1 : 0 < p)
  (h2 : p < 1)

/-- The variance of a binomial distribution -/
def variance (p : ℝ) (X : BinomialDistribution p) : ℝ := sorry

theorem binomial_variance (p : ℝ) (X : BinomialDistribution p) :
  variance p X = p * (1 - p) := by sorry

end NUMINAMATH_CALUDE_binomial_variance_l3590_359072


namespace NUMINAMATH_CALUDE_min_value_of_2a_plus_1_l3590_359095

theorem min_value_of_2a_plus_1 (a : ℝ) (h : 9*a^2 + 7*a + 5 = 2) : 
  ∃ (min : ℝ), min = -1 ∧ ∀ (x : ℝ), 9*x^2 + 7*x + 5 = 2 → 2*x + 1 ≥ min := by
sorry

end NUMINAMATH_CALUDE_min_value_of_2a_plus_1_l3590_359095


namespace NUMINAMATH_CALUDE_right_triangle_area_l3590_359070

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3590_359070


namespace NUMINAMATH_CALUDE_power_of_two_sum_l3590_359048

theorem power_of_two_sum (y : ℕ) : 8^3 + 8^3 + 8^3 + 8^3 = 2^y → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_sum_l3590_359048


namespace NUMINAMATH_CALUDE_ellipse_focus_directrix_distance_l3590_359068

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 64 + y^2 / 28 = 1

/-- Distance from P to the left focus -/
def distance_to_left_focus (P : ℝ × ℝ) : ℝ := 4

/-- Distance from P to the right directrix -/
def distance_to_right_directrix (P : ℝ × ℝ) : ℝ := 16

/-- Theorem: If P is on the ellipse and 4 units from the left focus,
    then it is 16 units from the right directrix -/
theorem ellipse_focus_directrix_distance (P : ℝ × ℝ) :
  is_on_ellipse P.1 P.2 →
  distance_to_left_focus P = 4 →
  distance_to_right_directrix P = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_directrix_distance_l3590_359068


namespace NUMINAMATH_CALUDE_integral_proof_l3590_359027

theorem integral_proof (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) : 
  (deriv (fun x => Real.log (abs (x - 2)) - 3 / (2 * (x + 2)^2))) x = 
  (x^3 + 6*x^2 + 15*x + 2) / ((x - 2) * (x + 2)^3) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l3590_359027


namespace NUMINAMATH_CALUDE_remaining_bottles_l3590_359077

/-- Calculates the number of remaining bottles of juice after some are broken -/
theorem remaining_bottles (total_crates : ℕ) (bottles_per_crate : ℕ) (broken_crates : ℕ) :
  total_crates = 7 →
  bottles_per_crate = 6 →
  broken_crates = 3 →
  total_crates * bottles_per_crate - broken_crates * bottles_per_crate = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_bottles_l3590_359077


namespace NUMINAMATH_CALUDE_triangle_circle_area_difference_l3590_359084

/-- The difference between the area of an equilateral triangle with side length 6
    and the area of an inscribed circle with radius 3 -/
theorem triangle_circle_area_difference : ∃ (circle_area triangle_area : ℝ),
  circle_area = 9 * Real.pi ∧
  triangle_area = 9 * Real.sqrt 3 ∧
  triangle_area - circle_area = 9 * Real.sqrt 3 - 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_area_difference_l3590_359084


namespace NUMINAMATH_CALUDE_smartphone_cost_decrease_l3590_359088

def original_cost : ℝ := 600
def new_cost : ℝ := 450

theorem smartphone_cost_decrease :
  (original_cost - new_cost) / original_cost * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_cost_decrease_l3590_359088


namespace NUMINAMATH_CALUDE_A_satisfies_conditions_l3590_359096

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 0}

-- Define set A
def A : Set ℝ := {1, 2}

-- Theorem statement
theorem A_satisfies_conditions : (A ∩ B = A) := by sorry

end NUMINAMATH_CALUDE_A_satisfies_conditions_l3590_359096


namespace NUMINAMATH_CALUDE_boys_at_reunion_l3590_359060

/-- The number of handshakes when each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: There are 12 boys at the reunion given the conditions. -/
theorem boys_at_reunion : ∃ (n : ℕ), n > 0 ∧ handshakes n = 66 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_boys_at_reunion_l3590_359060


namespace NUMINAMATH_CALUDE_exam_type_a_time_l3590_359046

/-- Represents the examination setup -/
structure Exam where
  totalTime : ℕ  -- Total time in minutes
  totalQuestions : ℕ
  typeAQuestions : ℕ
  typeAMultiplier : ℕ  -- How many times longer type A questions take compared to type B

/-- Calculates the time spent on type A problems -/
def timeOnTypeA (e : Exam) : ℚ :=
  let totalTypeB := e.totalQuestions - e.typeAQuestions
  let x := e.totalTime / (e.typeAQuestions * e.typeAMultiplier + totalTypeB)
  e.typeAQuestions * e.typeAMultiplier * x

/-- Theorem stating that for the given exam setup, 40 minutes should be spent on type A problems -/
theorem exam_type_a_time :
  let e : Exam := {
    totalTime := 180,  -- 3 hours * 60 minutes
    totalQuestions := 200,
    typeAQuestions := 25,
    typeAMultiplier := 2
  }
  timeOnTypeA e = 40 := by sorry


end NUMINAMATH_CALUDE_exam_type_a_time_l3590_359046


namespace NUMINAMATH_CALUDE_plastic_for_rulers_l3590_359074

theorem plastic_for_rulers (plastic_per_ruler : ℕ) (rulers_made : ℕ) : 
  plastic_per_ruler = 8 → rulers_made = 103 → plastic_per_ruler * rulers_made = 824 := by
  sorry

end NUMINAMATH_CALUDE_plastic_for_rulers_l3590_359074


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3590_359092

/-- Given a geometric sequence {aₙ} with a₁ = 1/16 and a₃a₇ = 2a₅ - 1, prove that a₃ = 1/4. -/
theorem geometric_sequence_problem (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 / 16 →
  a 3 * a 7 = 2 * a 5 - 1 →
  a 3 = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3590_359092


namespace NUMINAMATH_CALUDE_car_distance_l3590_359014

theorem car_distance (efficiency : ℝ) (gas : ℝ) (distance : ℝ) :
  efficiency = 20 →
  gas = 5 →
  distance = efficiency * gas →
  distance = 100 := by sorry

end NUMINAMATH_CALUDE_car_distance_l3590_359014


namespace NUMINAMATH_CALUDE_right_triangle_sin_a_l3590_359049

theorem right_triangle_sin_a (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : C = Real.pi / 2) (h3 : Real.cos B = 1 / 2) : Real.sin A = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_a_l3590_359049


namespace NUMINAMATH_CALUDE_point_on_line_p_value_l3590_359042

/-- Given that (m, n) and (m + p, n + 15) both lie on the line x = (y / 5) - (2 / 5),
    prove that p = 3. -/
theorem point_on_line_p_value (m n p : ℝ) : 
  (m = n / 5 - 2 / 5) → 
  (m + p = (n + 15) / 5 - 2 / 5) → 
  p = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_p_value_l3590_359042


namespace NUMINAMATH_CALUDE_second_car_speed_l3590_359008

/-- Theorem: Given two cars starting from opposite ends of a 500-mile highway
    at the same time, with one car traveling at 40 mph and both cars meeting
    after 5 hours, the speed of the second car is 60 mph. -/
theorem second_car_speed
  (highway_length : ℝ)
  (first_car_speed : ℝ)
  (meeting_time : ℝ)
  (second_car_speed : ℝ) :
  highway_length = 500 →
  first_car_speed = 40 →
  meeting_time = 5 →
  highway_length = first_car_speed * meeting_time + second_car_speed * meeting_time →
  second_car_speed = 60 :=
by
  sorry

#check second_car_speed

end NUMINAMATH_CALUDE_second_car_speed_l3590_359008


namespace NUMINAMATH_CALUDE_new_people_count_l3590_359001

/-- The number of people born in the country last year -/
def born : ℕ := 90171

/-- The number of people who immigrated to the country last year -/
def immigrated : ℕ := 16320

/-- The total number of new people who began living in the country last year -/
def total_new_people : ℕ := born + immigrated

/-- Theorem stating that the total number of new people is 106,491 -/
theorem new_people_count : total_new_people = 106491 := by
  sorry

end NUMINAMATH_CALUDE_new_people_count_l3590_359001


namespace NUMINAMATH_CALUDE_salary_calculation_l3590_359099

theorem salary_calculation (food_fraction : Rat) (rent_fraction : Rat) (clothes_fraction : Rat) 
  (savings_fraction : Rat) (tax_fraction : Rat) (remaining_amount : ℝ) :
  food_fraction = 1/5 →
  rent_fraction = 1/10 →
  clothes_fraction = 3/5 →
  savings_fraction = 1/20 →
  tax_fraction = 1/8 →
  remaining_amount = 18000 →
  ∃ S : ℝ, (7/160 : ℝ) * S = remaining_amount :=
by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l3590_359099


namespace NUMINAMATH_CALUDE_extreme_value_and_minimum_a_l3590_359040

noncomputable def f (a : ℤ) (x : ℝ) : ℝ := (1/2) * a * x^2 - Real.log x

theorem extreme_value_and_minimum_a :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 1 x ≤ f 1 y) ∧
  f 1 1 = (1/2) ∧
  (∀ (a : ℤ), (∀ (x : ℝ), x > 0 → f a x ≥ (1 - a) * x + 1) → a ≥ 2) ∧
  (∀ (x : ℝ), x > 0 → f 2 x ≥ (1 - 2) * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_and_minimum_a_l3590_359040


namespace NUMINAMATH_CALUDE_no_negative_exponents_l3590_359057

theorem no_negative_exponents (a b c d : ℤ) (h : (4:ℝ)^a + (4:ℝ)^b = (8:ℝ)^c + (27:ℝ)^d) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_no_negative_exponents_l3590_359057


namespace NUMINAMATH_CALUDE_power_seven_mod_twelve_l3590_359029

theorem power_seven_mod_twelve : 7^253 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_twelve_l3590_359029


namespace NUMINAMATH_CALUDE_inversion_similarity_l3590_359006

/-- Inversion of a point with respect to a circle -/
def inversion (O : ℝ × ℝ) (R : ℝ) (P : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Similarity of triangles -/
def triangles_similar (A B C D E F : ℝ × ℝ) : Prop := sorry

theorem inversion_similarity 
  (O A B : ℝ × ℝ) 
  (R : ℝ) 
  (A' B' : ℝ × ℝ) 
  (h1 : A' = inversion O R A) 
  (h2 : B' = inversion O R B) : 
  triangles_similar O A B B' O A' := 
sorry

end NUMINAMATH_CALUDE_inversion_similarity_l3590_359006


namespace NUMINAMATH_CALUDE_power_of_product_l3590_359047

theorem power_of_product (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3590_359047


namespace NUMINAMATH_CALUDE_solution_set_equals_target_set_l3590_359079

/-- The set of solutions for the system of equations with parameter a -/
def SolutionSet : Set (ℝ × ℝ) :=
  {(x, y) | ∃ a : ℝ, a * x + y = 2 * a + 3 ∧ x - a * y = a + 4}

/-- The circle with center (3, 1) and radius √5, excluding (2, -1) -/
def TargetSet : Set (ℝ × ℝ) :=
  {(x, y) | (x - 3)^2 + (y - 1)^2 = 5 ∧ (x, y) ≠ (2, -1)}

theorem solution_set_equals_target_set : SolutionSet = TargetSet := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_target_set_l3590_359079


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_l3590_359078

theorem fraction_inequality_solution (x : ℝ) : 
  x ≠ 3 → (x * (x + 1) / (x - 3)^2 ≥ 9 ↔ 
    (2.13696 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4.73804)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_l3590_359078


namespace NUMINAMATH_CALUDE_smallest_percentage_both_l3590_359012

/-- The smallest possible percentage of people eating both ice cream and chocolate in a town -/
theorem smallest_percentage_both (ice_cream_eaters chocolate_eaters : ℝ) 
  (h_ice_cream : ice_cream_eaters = 0.9)
  (h_chocolate : chocolate_eaters = 0.8) :
  ∃ (both : ℝ), both ≥ 0.7 ∧ 
    ∀ (x : ℝ), x ≥ 0 ∧ x ≤ 1 ∧ ice_cream_eaters + chocolate_eaters - x ≤ 1 → x ≥ both := by
  sorry


end NUMINAMATH_CALUDE_smallest_percentage_both_l3590_359012


namespace NUMINAMATH_CALUDE_semicircle_perimeter_semicircle_area_l3590_359043

-- Define constants
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 8
def π : ℝ := 3.14

-- Define the semicircle
def semicircle_diameter : ℝ := rectangle_length

-- Theorem for the perimeter of the semicircle
theorem semicircle_perimeter :
  π * semicircle_diameter / 2 + semicircle_diameter = 25.7 :=
sorry

-- Theorem for the area of the semicircle
theorem semicircle_area :
  π * (semicircle_diameter / 2)^2 / 2 = 39.25 :=
sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_semicircle_area_l3590_359043


namespace NUMINAMATH_CALUDE_nine_by_nine_corner_sum_l3590_359053

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- The value at a given position in the grid -/
def Grid.value (g : Grid) (row col : ℕ) : ℕ :=
  (row - 1) * g.size + col

/-- The sum of the corner values in the grid -/
def Grid.cornerSum (g : Grid) : ℕ :=
  g.value 1 1 + g.value 1 g.size + g.value g.size 1 + g.value g.size g.size

/-- Theorem: The sum of corner values in a 9x9 grid is 164 -/
theorem nine_by_nine_corner_sum :
  ∀ g : Grid, g.size = 9 → g.cornerSum = 164 :=
by
  sorry

end NUMINAMATH_CALUDE_nine_by_nine_corner_sum_l3590_359053


namespace NUMINAMATH_CALUDE_little_john_money_distribution_l3590_359081

def problem (initial_amount : ℚ) (sweets_cost : ℚ) (num_friends : ℕ) (remaining_amount : ℚ) : Prop :=
  let total_spent : ℚ := initial_amount - remaining_amount
  let amount_given_away : ℚ := total_spent - sweets_cost
  let amount_per_friend : ℚ := amount_given_away / num_friends
  amount_per_friend = 1

theorem little_john_money_distribution :
  problem 7.1 1.05 2 4.05 := by
  sorry

end NUMINAMATH_CALUDE_little_john_money_distribution_l3590_359081


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3590_359086

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  cos C + (cos A - Real.sqrt 3 * sin A) * cos B = 0 ∧
  b = Real.sqrt 3 ∧
  c = 1 →
  B = π / 3 ∧
  (1 / 2) * a * c * sin B = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3590_359086


namespace NUMINAMATH_CALUDE_c_leq_one_sufficient_not_necessary_l3590_359056

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

def sequence_a (c : ℝ) (n : ℕ) : ℝ :=
  |n - c|

theorem c_leq_one_sufficient_not_necessary (c : ℝ) :
  (c ≤ 1 → is_increasing (sequence_a c)) ∧
  ¬(is_increasing (sequence_a c) → c ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_c_leq_one_sufficient_not_necessary_l3590_359056


namespace NUMINAMATH_CALUDE_remainder_theorem_l3590_359097

theorem remainder_theorem (n : ℤ) : n % 9 = 5 → (4 * n - 6) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3590_359097


namespace NUMINAMATH_CALUDE_a_less_than_sqrt_a_iff_l3590_359005

theorem a_less_than_sqrt_a_iff (a : ℝ) : 0 < a ∧ a < 1 ↔ a < Real.sqrt a := by sorry

end NUMINAMATH_CALUDE_a_less_than_sqrt_a_iff_l3590_359005


namespace NUMINAMATH_CALUDE_dessert_probability_l3590_359020

theorem dessert_probability (p_dessert : ℝ) (p_dessert_no_coffee : ℝ) :
  p_dessert = 0.6 →
  p_dessert_no_coffee = 0.2 * p_dessert →
  1 - p_dessert = 0.4 := by
sorry

end NUMINAMATH_CALUDE_dessert_probability_l3590_359020


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3590_359015

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3590_359015


namespace NUMINAMATH_CALUDE_negation_of_even_multiple_of_two_l3590_359025

theorem negation_of_even_multiple_of_two :
  ¬(∀ n : ℕ, Even n → (∃ k : ℕ, n = 2 * k)) ↔ 
  (∃ n : ℕ, Even n ∧ ¬(∃ k : ℕ, n = 2 * k)) :=
sorry

end NUMINAMATH_CALUDE_negation_of_even_multiple_of_two_l3590_359025


namespace NUMINAMATH_CALUDE_race_probability_l3590_359076

theorem race_probability (total_cars : ℕ) (prob_Y prob_Z prob_XYZ : ℝ) : 
  total_cars = 15 →
  prob_Y = 1/8 →
  prob_Z = 1/12 →
  prob_XYZ = 0.4583333333333333 →
  ∃ (prob_X : ℝ), prob_X + prob_Y + prob_Z = prob_XYZ ∧ prob_X = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_race_probability_l3590_359076


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3590_359036

theorem right_triangle_sides (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 30 → 
  x^2 + y^2 + z^2 = 338 → 
  x^2 + y^2 = z^2 →
  ((x = 5 ∧ y = 12 ∧ z = 13) ∨ (x = 12 ∧ y = 5 ∧ z = 13)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3590_359036


namespace NUMINAMATH_CALUDE_trigonometric_properties_l3590_359062

theorem trigonometric_properties :
  (∀ x : ℝ, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1) ∧
  ¬(∃ x : ℝ, Real.sin x ^ 2 + Real.cos x ^ 2 > 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_properties_l3590_359062


namespace NUMINAMATH_CALUDE_hyperbola_sum_l3590_359041

/-- Proves that for a hyperbola with given properties, h + k + a + b = 11 -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 3 ∧ 
  k = -1 ∧ 
  c = Real.sqrt 41 ∧ 
  a = 4 ∧ 
  c^2 = a^2 + b^2 →
  h + k + a + b = 11 := by sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l3590_359041


namespace NUMINAMATH_CALUDE_y_value_l3590_359045

theorem y_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3590_359045


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l3590_359000

theorem arithmetic_sequence_count (start end_ diff : ℕ) (h1 : start = 24) (h2 : end_ = 162) (h3 : diff = 6) :
  (end_ - start) / diff + 1 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l3590_359000


namespace NUMINAMATH_CALUDE_max_player_salary_l3590_359051

theorem max_player_salary (num_players : ℕ) (min_salary : ℕ) (total_salary_cap : ℕ) :
  num_players = 18 →
  min_salary = 20000 →
  total_salary_cap = 900000 →
  ∃ (max_salary : ℕ),
    max_salary = 560000 ∧
    max_salary + (num_players - 1) * min_salary ≤ total_salary_cap ∧
    ∀ (s : ℕ), s > max_salary →
      s + (num_players - 1) * min_salary > total_salary_cap :=
by sorry


end NUMINAMATH_CALUDE_max_player_salary_l3590_359051


namespace NUMINAMATH_CALUDE_max_integers_greater_than_20_l3590_359064

theorem max_integers_greater_than_20 (integers : List ℤ) : 
  integers.length = 8 → 
  integers.sum = -20 → 
  (integers.filter (λ x => x > 20)).length ≤ 7 ∧ 
  ∃ (valid_list : List ℤ), 
    valid_list.length = 8 ∧ 
    valid_list.sum = -20 ∧ 
    (valid_list.filter (λ x => x > 20)).length = 7 :=
by sorry

end NUMINAMATH_CALUDE_max_integers_greater_than_20_l3590_359064


namespace NUMINAMATH_CALUDE_circle_equation_l3590_359094

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Check if a circle is tangent to a line -/
def Circle.tangentTo (c : Circle) (l : Line) : Prop :=
  let (cx, cy) := c.center
  |l.a * cx + l.b * cy + l.c| = c.radius * Real.sqrt (l.a^2 + l.b^2)

/-- The main theorem -/
theorem circle_equation (C : Circle) (l : Line) :
  C.contains (0, 0) →
  C.radius^2 * Real.pi = 2 * Real.pi →
  l.a = 1 ∧ l.b = -1 ∧ l.c = 2 →
  C.tangentTo l →
  (C.center = (1, 1) ∧ C.radius^2 = 2) ∨ (C.center = (-1, -1) ∧ C.radius^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3590_359094


namespace NUMINAMATH_CALUDE_bill_face_value_l3590_359037

/-- Calculates the face value of a bill given the true discount, time period, and annual interest rate -/
def calculate_face_value (true_discount : ℚ) (time_months : ℚ) (annual_rate : ℚ) : ℚ :=
  let time_years := time_months / 12
  let rate_decimal := annual_rate / 100
  (true_discount * (100 + (rate_decimal * time_years * 100))) / (rate_decimal * time_years * 100)

/-- Theorem stating that given the specific conditions, the face value of the bill is 1764 -/
theorem bill_face_value :
  let true_discount : ℚ := 189
  let time_months : ℚ := 9
  let annual_rate : ℚ := 16
  calculate_face_value true_discount time_months annual_rate = 1764 := by
  sorry

end NUMINAMATH_CALUDE_bill_face_value_l3590_359037


namespace NUMINAMATH_CALUDE_least_number_with_remainder_four_l3590_359009

theorem least_number_with_remainder_four (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → 
    (m % 6 ≠ 4 ∨ m % 9 ≠ 4 ∨ m % 12 ≠ 4 ∨ m % 18 ≠ 4)) ∧
  n % 6 = 4 ∧ n % 9 = 4 ∧ n % 12 = 4 ∧ n % 18 = 4 → 
  n = 40 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_four_l3590_359009


namespace NUMINAMATH_CALUDE_system_solution_l3590_359034

theorem system_solution (x y z : ℝ) : 
  (x^3 + y^3 = 3*y + 3*z + 4 ∧
   y^3 + z^3 = 3*z + 3*x + 4 ∧
   x^3 + z^3 = 3*x + 3*y + 4) ↔ 
  ((x = 2 ∧ y = 2 ∧ z = 2) ∨ (x = -1 ∧ y = -1 ∧ z = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3590_359034


namespace NUMINAMATH_CALUDE_smallest_positive_b_l3590_359085

/-- Definition of circle w1 -/
def w1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y - 23 = 0

/-- Definition of circle w2 -/
def w2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 8*y + 9 = 0

/-- Definition of a circle externally tangent to w2 and internally tangent to w1 -/
def tangent_circle (h k r : ℝ) : Prop :=
  (r + 2)^2 = (h + 3)^2 + (k + 4)^2 ∧ (6 - r)^2 = (h - 3)^2 + (k + 4)^2

/-- The line y = bx contains the center of the tangent circle -/
def center_on_line (h k b : ℝ) : Prop := k = b * h

/-- The main theorem -/
theorem smallest_positive_b :
  ∃ (b : ℝ), b > 0 ∧
  (∀ (h k r : ℝ), tangent_circle h k r → center_on_line h k b) ∧
  (∀ (b' : ℝ), 0 < b' ∧ b' < b →
    ¬(∀ (h k r : ℝ), tangent_circle h k r → center_on_line h k b')) ∧
  b^2 = 64/25 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_b_l3590_359085


namespace NUMINAMATH_CALUDE_equal_roots_condition_l3590_359083

/-- 
If the quadratic equation 2x^2 - ax + 2 = 0 has two equal real roots, 
then a = 4 or a = -4
-/
theorem equal_roots_condition (a : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - a * x + 2 = 0 ∧ 
   (∀ y : ℝ, 2 * y^2 - a * y + 2 = 0 → y = x)) → 
  (a = 4 ∨ a = -4) := by
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l3590_359083


namespace NUMINAMATH_CALUDE_three_nap_simultaneously_l3590_359059

-- Define the type for mathematicians
def Mathematician := Fin 5

-- Define the type for nap times
variable {T : Type*}

-- Define the nap function that assigns two nap times to each mathematician
variable (nap : Mathematician → Fin 2 → T)

-- Define the property that any two mathematicians share a nap time
variable (share_nap : ∀ m1 m2 : Mathematician, m1 ≠ m2 → ∃ t : T, (∃ i : Fin 2, nap m1 i = t) ∧ (∃ j : Fin 2, nap m2 j = t))

-- Theorem statement
theorem three_nap_simultaneously :
  ∃ t : T, ∃ m1 m2 m3 : Mathematician, m1 ≠ m2 ∧ m2 ≠ m3 ∧ m1 ≠ m3 ∧
  (∃ i j k : Fin 2, nap m1 i = t ∧ nap m2 j = t ∧ nap m3 k = t) :=
sorry

end NUMINAMATH_CALUDE_three_nap_simultaneously_l3590_359059


namespace NUMINAMATH_CALUDE_f_negative_nine_l3590_359044

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_negative_nine (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_period : has_period f 4) 
  (h_f_one : f 1 = 1) : 
  f (-9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_nine_l3590_359044


namespace NUMINAMATH_CALUDE_parallel_line_family_l3590_359069

/-- The line equation as a function of x, y, and a -/
def line_equation (x y a : ℝ) : ℝ := (a - 1) * x - y + 2 * a + 1

/-- Theorem stating that the lines form a parallel family -/
theorem parallel_line_family :
  ∀ a₁ a₂ : ℝ, ∃ k : ℝ, ∀ x y : ℝ,
    line_equation x y a₁ = 0 ↔ line_equation x y a₂ = k := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_family_l3590_359069


namespace NUMINAMATH_CALUDE_highest_divisible_digit_l3590_359002

theorem highest_divisible_digit : ∃ (a : ℕ), a ≤ 9 ∧ 
  (365 * 10 + a) * 100 + 16 % 8 = 0 ∧ 
  ∀ (b : ℕ), b ≤ 9 → b > a → (365 * 10 + b) * 100 + 16 % 8 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_highest_divisible_digit_l3590_359002


namespace NUMINAMATH_CALUDE_expression_simplification_l3590_359016

def x : ℚ := -2
def y : ℚ := 1/2

theorem expression_simplification :
  (x + 4*y) * (x - 4*y) + (x - 4*y)^2 - (4*x^2 - x*y) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3590_359016


namespace NUMINAMATH_CALUDE_net_population_increase_l3590_359018

/-- Calculates the net population increase given birth, immigration, emigration, death rate, and initial population. -/
theorem net_population_increase
  (births : ℕ)
  (immigrants : ℕ)
  (emigrants : ℕ)
  (death_rate : ℚ)
  (initial_population : ℕ)
  (h_births : births = 90171)
  (h_immigrants : immigrants = 16320)
  (h_emigrants : emigrants = 8212)
  (h_death_rate : death_rate = 8 / 10000)
  (h_initial_population : initial_population = 2876543) :
  (births + immigrants) - (emigrants + Int.floor (death_rate * initial_population)) = 96078 :=
by sorry

end NUMINAMATH_CALUDE_net_population_increase_l3590_359018


namespace NUMINAMATH_CALUDE_boxes_shipped_this_week_l3590_359028

/-- Represents the number of pomelos in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of boxes shipped last week -/
def last_week_boxes : ℕ := 10

/-- Represents the total number of pomelos shipped last week -/
def last_week_pomelos : ℕ := 240

/-- Represents the number of dozens of pomelos shipped this week -/
def this_week_dozens : ℕ := 60

/-- Calculates the number of boxes shipped this week -/
def boxes_this_week : ℕ :=
  (this_week_dozens * dozen) / (last_week_pomelos / last_week_boxes)

theorem boxes_shipped_this_week :
  boxes_this_week = 30 := by sorry

end NUMINAMATH_CALUDE_boxes_shipped_this_week_l3590_359028


namespace NUMINAMATH_CALUDE_coin_distribution_theorem_l3590_359033

/-- Represents the number of rounds in the coin distribution -/
def x : ℕ := sorry

/-- Pete's coins after distribution -/
def pete_coins (x : ℕ) : ℕ := x * (x + 1) / 2

/-- Paul's coins after distribution -/
def paul_coins (x : ℕ) : ℕ := x

/-- The condition that Pete has three times as many coins as Paul -/
axiom pete_triple_paul : pete_coins x = 3 * paul_coins x

/-- The total number of coins -/
def total_coins (x : ℕ) : ℕ := pete_coins x + paul_coins x

theorem coin_distribution_theorem : total_coins x = 20 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_theorem_l3590_359033


namespace NUMINAMATH_CALUDE_triangle_properties_l3590_359065

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  tan C = (sin A + sin B) / (cos A + cos B) →
  C = π / 3 ∧
  (∀ r : ℝ, r > 0 → 2 * r = 1 →
    3/4 < a^2 + b^2 ∧ a^2 + b^2 ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3590_359065


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3590_359038

def R : Set ℝ := Set.univ

def A : Set ℝ := {1, 2, 3, 4, 5}

def B : Set ℝ := {x : ℝ | x * (4 - x) < 0}

theorem intersection_complement_equality :
  A ∩ (Set.compl B) = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3590_359038


namespace NUMINAMATH_CALUDE_max_value_of_z_l3590_359061

-- Define the objective function
def z (x y : ℝ) : ℝ := 3 * x + 2 * y

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4

-- Theorem statement
theorem max_value_of_z :
  ∃ (x y : ℝ), feasible_region x y ∧
  ∀ (x' y' : ℝ), feasible_region x' y' → z x y ≥ z x' y' ∧
  z x y = 12 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l3590_359061


namespace NUMINAMATH_CALUDE_max_placement_1002nd_round_max_placement_1001st_round_l3590_359091

/-- Represents the state of an election round -/
structure ElectionRound where
  candidateCount : Nat
  votes : List Nat

/-- Defines the election process -/
def runElection (initialRound : ElectionRound) : Nat → Option Nat :=
  sorry

/-- Theorem for the maximum initial placement allowing victory in the 1002nd round -/
theorem max_placement_1002nd_round 
  (initialCandidates : Nat) 
  (ostapInitialPlacement : Nat) :
  initialCandidates = 2002 →
  (∃ (initialVotes : List Nat), 
    initialVotes.length = initialCandidates ∧
    ostapInitialPlacement = 2001 ∧
    runElection ⟨initialCandidates, initialVotes⟩ 1002 = some ostapInitialPlacement) ∧
  (∀ k > 2001, ∀ (initialVotes : List Nat),
    initialVotes.length = initialCandidates →
    runElection ⟨initialCandidates, initialVotes⟩ 1002 ≠ some k) :=
  sorry

/-- Theorem for the maximum initial placement allowing victory in the 1001st round -/
theorem max_placement_1001st_round 
  (initialCandidates : Nat) 
  (ostapInitialPlacement : Nat) :
  initialCandidates = 2002 →
  (∃ (initialVotes : List Nat), 
    initialVotes.length = initialCandidates ∧
    ostapInitialPlacement = 1001 ∧
    runElection ⟨initialCandidates, initialVotes⟩ 1001 = some ostapInitialPlacement) ∧
  (∀ k > 1001, ∀ (initialVotes : List Nat),
    initialVotes.length = initialCandidates →
    runElection ⟨initialCandidates, initialVotes⟩ 1001 ≠ some k) :=
  sorry

end NUMINAMATH_CALUDE_max_placement_1002nd_round_max_placement_1001st_round_l3590_359091


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3590_359032

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (1 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 68 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3590_359032


namespace NUMINAMATH_CALUDE_polygon_with_900_degree_sum_is_heptagon_l3590_359075

theorem polygon_with_900_degree_sum_is_heptagon :
  ∀ n : ℕ, 
    n ≥ 3 →
    (n - 2) * 180 = 900 →
    n = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_900_degree_sum_is_heptagon_l3590_359075


namespace NUMINAMATH_CALUDE_third_roll_probability_l3590_359019

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 3
def biased_die_prob : ℚ := 2 / 3

-- Define the probability of rolling sixes or fives twice for each die
def fair_die_two_rolls : ℚ := fair_die_prob ^ 2
def biased_die_two_rolls : ℚ := biased_die_prob ^ 2

-- Define the normalized probabilities of using each die given the first two rolls
def prob_fair_die : ℚ := fair_die_two_rolls / (fair_die_two_rolls + biased_die_two_rolls)
def prob_biased_die : ℚ := biased_die_two_rolls / (fair_die_two_rolls + biased_die_two_rolls)

-- Theorem: The probability of rolling a six or five on the third roll is 3/5
theorem third_roll_probability : 
  prob_fair_die * fair_die_prob + prob_biased_die * biased_die_prob = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_third_roll_probability_l3590_359019


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l3590_359055

theorem number_exceeding_percentage (x : ℝ) : x = 60 ↔ x = 0.12 * x + 52.8 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l3590_359055


namespace NUMINAMATH_CALUDE_fraction_value_l3590_359010

theorem fraction_value (a b c d : ℚ) 
  (h1 : a = 3 * b) 
  (h2 : b = 2 * c) 
  (h3 : c = 4 * d) 
  (h4 : d ≠ 0) : a * c / (b * d) = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3590_359010


namespace NUMINAMATH_CALUDE_number_of_history_books_l3590_359052

theorem number_of_history_books (total_books geography_books math_books : ℕ) 
  (h1 : total_books = 100)
  (h2 : geography_books = 25)
  (h3 : math_books = 43) :
  total_books - geography_books - math_books = 32 :=
by sorry

end NUMINAMATH_CALUDE_number_of_history_books_l3590_359052


namespace NUMINAMATH_CALUDE_unique_root_implies_k_range_l3590_359003

/-- A function f(x) with parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + (1-k)*x - k

/-- Theorem: If f(x) has exactly one root in (2,3), then k is in (2,3) -/
theorem unique_root_implies_k_range (k : ℝ) :
  (∃! x, x ∈ (Set.Ioo 2 3) ∧ f k x = 0) → k ∈ Set.Ioo 2 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_implies_k_range_l3590_359003


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l3590_359087

-- Define the set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the set B
def B : Set ℝ := {x | x * (x - 3) < 0}

-- Define the result set
def result : Set ℝ := {x | x ≤ 2 ∨ x ≥ 3}

-- Theorem statement
theorem union_of_A_and_complement_of_B : A ∪ (Set.univ \ B) = result := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l3590_359087


namespace NUMINAMATH_CALUDE_sunday_bicycles_bought_l3590_359021

/-- Represents the number of bicycles in Hank's store. -/
def BicycleCount := ℤ

/-- Represents the change in bicycle count for a day. -/
structure DailyChange where
  sold : ℕ
  bought : ℕ

/-- Calculates the net change in bicycle count for a day. -/
def netChange (dc : DailyChange) : ℤ :=
  dc.bought - dc.sold

/-- Represents the changes in bicycle count over three days. -/
structure ThreeDayChanges where
  friday : DailyChange
  saturday : DailyChange
  sunday_sold : ℕ

theorem sunday_bicycles_bought 
  (changes : ThreeDayChanges)
  (h_friday : changes.friday = ⟨10, 15⟩)
  (h_saturday : changes.saturday = ⟨12, 8⟩)
  (h_sunday_sold : changes.sunday_sold = 9)
  (h_net_increase : netChange changes.friday + netChange changes.saturday + 
    (sunday_bought - changes.sunday_sold) = 3)
  : ∃ (sunday_bought : ℕ), sunday_bought = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_sunday_bicycles_bought_l3590_359021


namespace NUMINAMATH_CALUDE_water_added_to_mixture_water_added_is_ten_l3590_359063

/-- Given a mixture of alcohol and water, prove the amount of water added to change the ratio. -/
theorem water_added_to_mixture (initial_ratio : ℚ) (final_ratio : ℚ) (alcohol_quantity : ℚ) : ℚ :=
  let initial_water := (alcohol_quantity * 5) / 2
  let water_added := (7 * alcohol_quantity) / 2 - initial_water
  by
    -- Assumptions
    have h1 : initial_ratio = 2 / 5 := by sorry
    have h2 : final_ratio = 2 / 7 := by sorry
    have h3 : alcohol_quantity = 10 := by sorry

    -- Proof
    sorry

/-- The amount of water added to the mixture is 10 liters. -/
theorem water_added_is_ten : water_added_to_mixture (2/5) (2/7) 10 = 10 := by sorry

end NUMINAMATH_CALUDE_water_added_to_mixture_water_added_is_ten_l3590_359063


namespace NUMINAMATH_CALUDE_directrix_of_specific_parabola_l3590_359090

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Defines the directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop := sorry

/-- The specific parabola with equation x^2 = 8y -/
def specific_parabola : Parabola :=
  { equation := fun x y => x^2 = 8*y }

/-- Theorem stating that the directrix of the specific parabola is y = -2 -/
theorem directrix_of_specific_parabola :
  directrix specific_parabola = fun y => y = -2 := by sorry

end NUMINAMATH_CALUDE_directrix_of_specific_parabola_l3590_359090


namespace NUMINAMATH_CALUDE_one_fourth_divided_by_two_l3590_359035

theorem one_fourth_divided_by_two : (1 / 4 : ℚ) / 2 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_divided_by_two_l3590_359035


namespace NUMINAMATH_CALUDE_unit_conversion_l3590_359067

/-- Conversion rates --/
def hectare_to_square_meter : ℝ := 10000
def meter_to_centimeter : ℝ := 100
def square_kilometer_to_hectare : ℝ := 100
def hour_to_minute : ℝ := 60
def kilogram_to_gram : ℝ := 1000

/-- Unit conversion theorem --/
theorem unit_conversion :
  (360 / hectare_to_square_meter = 0.036) ∧
  (504 / meter_to_centimeter = 5.04) ∧
  (0.06 * square_kilometer_to_hectare = 6) ∧
  (15 / hour_to_minute = 0.25) ∧
  (5.45 = 5 + 450 / kilogram_to_gram) :=
by sorry

end NUMINAMATH_CALUDE_unit_conversion_l3590_359067


namespace NUMINAMATH_CALUDE_football_team_linemen_l3590_359054

-- Define the constants from the problem
def cooler_capacity : ℕ := 126
def skill_players : ℕ := 10
def lineman_consumption : ℕ := 8
def skill_player_consumption : ℕ := 6
def skill_players_drinking : ℕ := 5

-- Define the number of linemen as a variable
def num_linemen : ℕ := sorry

-- Theorem statement
theorem football_team_linemen :
  num_linemen * lineman_consumption +
  skill_players_drinking * skill_player_consumption = cooler_capacity :=
by sorry

end NUMINAMATH_CALUDE_football_team_linemen_l3590_359054


namespace NUMINAMATH_CALUDE_ian_money_left_l3590_359004

/-- Calculates the amount of money left after spending half of earnings from surveys -/
def money_left (hours_worked : ℕ) (hourly_rate : ℚ) : ℚ :=
  (hours_worked : ℚ) * hourly_rate / 2

/-- Theorem: Given the conditions, prove that Ian has $72 left -/
theorem ian_money_left : money_left 8 18 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ian_money_left_l3590_359004


namespace NUMINAMATH_CALUDE_beetle_speed_l3590_359039

/-- Calculates the speed of a beetle given the ant's distance and the beetle's relative speed -/
theorem beetle_speed (ant_distance : Real) (time_minutes : Real) (beetle_relative_speed : Real) :
  let beetle_distance := ant_distance * (1 - beetle_relative_speed)
  let time_hours := time_minutes / 60
  let speed_km_h := (beetle_distance / 1000) / time_hours
  speed_km_h = 2.55 :=
by
  sorry

#check beetle_speed 600 12 0.15

end NUMINAMATH_CALUDE_beetle_speed_l3590_359039


namespace NUMINAMATH_CALUDE_tan_2alpha_l3590_359013

theorem tan_2alpha (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α - β) = 5) :
  Real.tan (2 * α) = -4/7 := by sorry

end NUMINAMATH_CALUDE_tan_2alpha_l3590_359013


namespace NUMINAMATH_CALUDE_fraction_calculation_l3590_359017

theorem fraction_calculation (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  ((x + 1) / (y - 1)) / ((y + 2) / (x - 2)) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3590_359017


namespace NUMINAMATH_CALUDE_octal_year_to_decimal_l3590_359066

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The octal representation of the year -/
def octal_year : List Nat := [7, 4, 2]

/-- Theorem stating that the octal year 742 is equal to 482 in decimal -/
theorem octal_year_to_decimal :
  octal_to_decimal octal_year = 482 := by sorry

end NUMINAMATH_CALUDE_octal_year_to_decimal_l3590_359066


namespace NUMINAMATH_CALUDE_f_derivative_l3590_359089

noncomputable def f (x : ℝ) : ℝ := x^3 * (Real.cos x)^5 * (Real.arctan x)^7 * (Real.log x)^4 * (Real.arcsin x)^10

theorem f_derivative (x : ℝ) (hx : x ≠ 0 ∧ x^2 < 1) : 
  deriv f x = f x * (3/x - 5*Real.tan x + 7/(Real.arctan x * (1 + x^2)) + 4/(x * Real.log x) + 10/(Real.arcsin x * Real.sqrt (1 - x^2))) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_l3590_359089


namespace NUMINAMATH_CALUDE_book_fraction_is_half_l3590_359058

-- Define the total amount Jennifer had
def total_money : ℚ := 120

-- Define the fraction spent on sandwich
def sandwich_fraction : ℚ := 1 / 5

-- Define the fraction spent on museum ticket
def museum_fraction : ℚ := 1 / 6

-- Define the amount left over
def left_over : ℚ := 16

-- Theorem to prove
theorem book_fraction_is_half :
  let sandwich_cost := total_money * sandwich_fraction
  let museum_cost := total_money * museum_fraction
  let total_spent := total_money - left_over
  let book_cost := total_spent - sandwich_cost - museum_cost
  book_cost / total_money = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_book_fraction_is_half_l3590_359058


namespace NUMINAMATH_CALUDE_toms_seashells_l3590_359073

theorem toms_seashells (sally_shells : ℕ) (jessica_shells : ℕ) (total_shells : ℕ) 
  (h1 : sally_shells = 9)
  (h2 : jessica_shells = 5)
  (h3 : total_shells = 21) :
  total_shells - (sally_shells + jessica_shells) = 7 := by
  sorry

end NUMINAMATH_CALUDE_toms_seashells_l3590_359073


namespace NUMINAMATH_CALUDE_max_groups_is_100_l3590_359022

/-- Represents the number of cards for each value -/
def CardCount : ℕ := 200

/-- Represents the target sum for each group -/
def TargetSum : ℕ := 9

/-- Represents the maximum number of groups that can be formed -/
def MaxGroups : ℕ := 100

/-- Proves that the maximum number of groups that can be formed is 100 -/
theorem max_groups_is_100 :
  ∀ (groups : ℕ) (cards_5 cards_2 cards_1 : ℕ),
    cards_5 = CardCount →
    cards_2 = CardCount →
    cards_1 = CardCount →
    (∀ g : ℕ, g ≤ groups → ∃ (a b c : ℕ),
      a + b + c = TargetSum ∧
      a * 5 + b * 2 + c * 1 = TargetSum ∧
      a ≤ cards_5 ∧ b ≤ cards_2 ∧ c ≤ cards_1) →
    groups ≤ MaxGroups :=
  sorry

end NUMINAMATH_CALUDE_max_groups_is_100_l3590_359022


namespace NUMINAMATH_CALUDE_smallest_n_with_properties_l3590_359007

def has_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = a * 10^c + d₁ * 10^b + d₂

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a * 5^b

theorem smallest_n_with_properties : 
  ∀ n : ℕ, 
    n > 0 ∧ 
    is_terminating_decimal n ∧ 
    has_digits n 9 5 → 
    n ≥ 9000000 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_properties_l3590_359007


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3590_359098

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3590_359098


namespace NUMINAMATH_CALUDE_max_cube_sum_four_squares_l3590_359080

theorem max_cube_sum_four_squares {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = 4) :
  a^3 + b^3 + c^3 + d^3 ≤ 8 ∧ ∃ (a₀ b₀ c₀ d₀ : ℝ), a₀^2 + b₀^2 + c₀^2 + d₀^2 = 4 ∧ a₀^3 + b₀^3 + c₀^3 + d₀^3 = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_cube_sum_four_squares_l3590_359080
