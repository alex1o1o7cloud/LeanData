import Mathlib

namespace NUMINAMATH_CALUDE_shopping_ratio_l1719_171994

theorem shopping_ratio : 
  let emma_spent : ℕ := 58
  let elsa_spent : ℕ := 2 * emma_spent
  let total_spent : ℕ := 638
  let elizabeth_spent : ℕ := total_spent - (emma_spent + elsa_spent)
  (elizabeth_spent : ℚ) / (elsa_spent : ℚ) = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_shopping_ratio_l1719_171994


namespace NUMINAMATH_CALUDE_kennel_problem_l1719_171928

theorem kennel_problem (total : ℕ) (long_fur : ℕ) (brown : ℕ) (neither : ℕ) 
  (h_total : total = 45)
  (h_long_fur : long_fur = 26)
  (h_brown : brown = 22)
  (h_neither : neither = 8) :
  long_fur + brown - (total - neither) = 11 :=
by sorry

end NUMINAMATH_CALUDE_kennel_problem_l1719_171928


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1719_171914

-- Define the line Ax + By + C = 0
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (l : Line) : Prop :=
  l.A * l.C < 0 ∧ l.B * l.C > 0

-- Define the second quadrant
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- Theorem statement
theorem line_not_in_second_quadrant (l : Line) 
  (h : satisfies_conditions l) :
  ¬∃ (x y : ℝ), l.A * x + l.B * y + l.C = 0 ∧ in_second_quadrant x y :=
by sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1719_171914


namespace NUMINAMATH_CALUDE_power_function_through_point_l1719_171945

theorem power_function_through_point (a : ℝ) :
  (2 : ℝ) ^ a = Real.sqrt 2 → a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1719_171945


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1719_171938

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, x^2 - 6*x + 11 = 23 ↔ x = a ∨ x = b) →
  a ≥ b →
  3*a + 2*b = 15 + Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1719_171938


namespace NUMINAMATH_CALUDE_interest_rate_problem_l1719_171936

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem interest_rate_problem (principal interest time : ℚ) 
  (h1 : principal = 4000)
  (h2 : interest = 640)
  (h3 : time = 2)
  (h4 : simple_interest principal (8 : ℚ) time = interest) :
  8 = (interest * 100) / (principal * time) :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l1719_171936


namespace NUMINAMATH_CALUDE_station_A_relay_ways_l1719_171929

/-- Represents a communication station -/
inductive Station : Type
| A | B | C | D

/-- The number of stations excluding A -/
def num_other_stations : Nat := 3

/-- The total number of ways station A can relay the message -/
def total_relay_ways : Nat := 16

/-- Theorem stating the number of ways station A can relay the message -/
theorem station_A_relay_ways :
  (∀ s₁ s₂ : Station, s₁ ≠ s₂ → (∃ t : Nat, t > 0)) →  -- Stations can communicate pairwise
  (∀ s : Station, ∃ t : Nat, t > 0) →  -- Space station can send to any station
  (∀ s : Station, ∀ n : Nat, n > 1 → ¬∃ t : Nat, t > 0) →  -- No simultaneous transmissions
  (∃ n : Nat, n = 3) →  -- Three transmissions occurred
  (∀ s : Station, ∃ m : Nat, m > 0) →  -- All stations received the message
  total_relay_ways = (2^num_other_stations - 1) + num_other_stations * 2^(num_other_stations - 1) :=
by sorry

end NUMINAMATH_CALUDE_station_A_relay_ways_l1719_171929


namespace NUMINAMATH_CALUDE_limit_fraction_is_two_l1719_171968

theorem limit_fraction_is_two : ∀ ε > 0, ∃ N : ℕ, ∀ n > N,
  |((2 * n - 3 : ℝ) / (n + 2 : ℝ)) - 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_fraction_is_two_l1719_171968


namespace NUMINAMATH_CALUDE_shirts_cost_after_discount_l1719_171972

def first_shirt_cost : ℝ := 15
def price_difference : ℝ := 6
def discount_rate : ℝ := 0.1

def second_shirt_cost : ℝ := first_shirt_cost - price_difference
def total_cost : ℝ := first_shirt_cost + second_shirt_cost
def discounted_cost : ℝ := total_cost * (1 - discount_rate)

theorem shirts_cost_after_discount :
  discounted_cost = 21.60 := by sorry

end NUMINAMATH_CALUDE_shirts_cost_after_discount_l1719_171972


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_rectangular_solid_l1719_171991

/-- The surface area of a sphere containing a rectangular solid -/
theorem sphere_surface_area_with_rectangular_solid :
  ∀ (a b c : ℝ) (S : ℝ),
    a = 3 →
    b = 4 →
    c = 5 →
    S = 4 * Real.pi * ((a^2 + b^2 + c^2) / 4) →
    S = 50 * Real.pi :=
by
  sorry

#check sphere_surface_area_with_rectangular_solid

end NUMINAMATH_CALUDE_sphere_surface_area_with_rectangular_solid_l1719_171991


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l1719_171921

theorem chicken_rabbit_problem :
  ∀ (chickens rabbits : ℕ),
    chickens + rabbits = 100 →
    2 * chickens = 4 * rabbits + 26 →
    chickens = 71 :=
by
  sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l1719_171921


namespace NUMINAMATH_CALUDE_min_value_expression_l1719_171974

theorem min_value_expression (m n : ℝ) (h : m > 2 * n) :
  ∃ (min : ℝ), min = 6 ∧ 
  (∀ x, x = m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) → x ≥ min) ∧
  (∃ y, y = m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) ∧ y = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1719_171974


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1719_171932

/-- The maximum marks for an exam where:
  1. The passing mark is 33% of the maximum marks
  2. A student who got 175 marks failed by 56 marks
-/
theorem exam_maximum_marks : ∃ (M : ℕ), 
  (M * 33 / 100 = 175 + 56) ∧ 
  M = 700 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1719_171932


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1719_171978

theorem sphere_surface_area (r h : ℝ) (h1 : r = 1) (h2 : h = Real.sqrt 3) : 
  let R := (2 * Real.sqrt 3) / 3
  4 * π * R^2 = (16 * π) / 3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1719_171978


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1719_171955

/-- Given a point P with coordinates (2, -3), prove that its coordinates with respect to the origin are (2, -3) -/
theorem point_coordinates_wrt_origin : 
  let P : ℝ × ℝ := (2, -3)
  P = (2, -3) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1719_171955


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l1719_171953

/-- Given a line expressed as a dot product of vectors, prove its slope-intercept form --/
theorem line_equation_equivalence (x y : ℝ) : 
  (3 : ℝ) * (x - 2) + (-4 : ℝ) * (y - 8) = 0 ↔ y = (3/4 : ℝ) * x + (13/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l1719_171953


namespace NUMINAMATH_CALUDE_range_of_m_l1719_171927

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*m*x + 4 = 0
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*(m-2)*x - 3*m + 10 = 0

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∧ ¬(q m)) → (2 ≤ m ∧ m < 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1719_171927


namespace NUMINAMATH_CALUDE_expression_evaluation_l1719_171982

theorem expression_evaluation (a b : ℤ) (h1 : a = 2) (h2 : b = -2) :
  5 * a * (b - 2) + 2 * a * (2 - b) = -24 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1719_171982


namespace NUMINAMATH_CALUDE_min_c_value_l1719_171904

/-- Given five consecutive positive integers a, b, c, d, e,
    if b + c + d is a perfect square and a + b + c + d + e is a perfect cube,
    then the minimum value of c is 675. -/
theorem min_c_value (a b c d e : ℕ) : 
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e → 
  ∃ m : ℕ, b + c + d = m^2 →
  ∃ n : ℕ, a + b + c + d + e = n^3 →
  ∀ c' : ℕ, (∃ a' b' d' e' : ℕ, 
    a' + 1 = b' ∧ b' + 1 = c' ∧ c' + 1 = d' ∧ d' + 1 = e' ∧
    ∃ m' : ℕ, b' + c' + d' = m'^2 ∧
    ∃ n' : ℕ, a' + b' + c' + d' + e' = n'^3) →
  c' ≥ 675 :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l1719_171904


namespace NUMINAMATH_CALUDE_missing_sale_is_1000_l1719_171965

/-- Calculates the missing sale amount given the sales for 5 months and the average sale for 6 months -/
def calculate_missing_sale (sale1 sale2 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale4 + sale5 + sale6)

/-- Theorem stating that given the specific sales and average, the missing sale must be 1000 -/
theorem missing_sale_is_1000 :
  calculate_missing_sale 800 900 700 800 900 850 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_missing_sale_is_1000_l1719_171965


namespace NUMINAMATH_CALUDE_sin_2theta_plus_pi_third_l1719_171960

theorem sin_2theta_plus_pi_third (θ : Real) 
  (h1 : θ > π / 2) (h2 : θ < π) 
  (h3 : 1 / Real.sin θ + 1 / Real.cos θ = 2 * Real.sqrt 2) : 
  Real.sin (2 * θ + π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_plus_pi_third_l1719_171960


namespace NUMINAMATH_CALUDE_inequality_proof_l1719_171967

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 5/2) (hy : y ≥ 5/2) (hz : z ≥ 5/2) :
  (1 + 1/(2+x)) * (1 + 1/(2+y)) * (1 + 1/(2+z)) ≥ (1 + 1/(2 + (x*y*z)^(1/3)))^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1719_171967


namespace NUMINAMATH_CALUDE_set_operations_l1719_171996

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem set_operations :
  (A ∩ B = {x | 3 ≤ x ∧ x < 7}) ∧
  (Aᶜ = {x | x < 3 ∨ 7 ≤ x}) ∧
  ((A ∪ B)ᶜ = {x | x ≤ 2 ∨ 10 ≤ x}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1719_171996


namespace NUMINAMATH_CALUDE_motel_rent_theorem_l1719_171916

/-- Represents the total rent charged by a motel on a Saturday night. -/
def TotalRent : ℕ → ℕ → ℕ 
  | r50, r60 => 50 * r50 + 60 * r60

/-- Represents the condition that changing 10 rooms from $60 to $50 reduces the rent by 25%. -/
def RentReductionCondition (r50 r60 : ℕ) : Prop :=
  4 * (TotalRent (r50 + 10) (r60 - 10)) = 3 * (TotalRent r50 r60)

theorem motel_rent_theorem :
  ∃ (r50 r60 : ℕ), RentReductionCondition r50 r60 ∧ TotalRent r50 r60 = 400 :=
sorry

end NUMINAMATH_CALUDE_motel_rent_theorem_l1719_171916


namespace NUMINAMATH_CALUDE_tv_cost_l1719_171910

theorem tv_cost (total_budget : ℕ) (computer_cost : ℕ) (fridge_extra_cost : ℕ) :
  total_budget = 1600 →
  computer_cost = 250 →
  fridge_extra_cost = 500 →
  ∃ tv_cost : ℕ, tv_cost = 600 ∧ 
    tv_cost + (computer_cost + fridge_extra_cost) + computer_cost = total_budget :=
by sorry

end NUMINAMATH_CALUDE_tv_cost_l1719_171910


namespace NUMINAMATH_CALUDE_paint_cans_for_house_l1719_171909

/-- Represents the number of paint cans needed for a house painting job -/
def paint_cans_needed (num_bedrooms : ℕ) (paint_per_room : ℕ) 
  (color_can_size : ℕ) (white_can_size : ℕ) : ℕ :=
  let num_other_rooms := 2 * num_bedrooms
  let total_rooms := num_bedrooms + num_other_rooms
  let total_paint_needed := total_rooms * paint_per_room
  let color_cans := num_bedrooms * paint_per_room
  let white_paint_needed := num_other_rooms * paint_per_room
  let white_cans := (white_paint_needed + white_can_size - 1) / white_can_size
  color_cans + white_cans

/-- Theorem stating that the number of paint cans needed for the given conditions is 10 -/
theorem paint_cans_for_house : paint_cans_needed 3 2 1 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_for_house_l1719_171909


namespace NUMINAMATH_CALUDE_average_of_x_and_y_is_16_l1719_171990

theorem average_of_x_and_y_is_16 (x y : ℝ) : 
  3 = 0.15 * x → 3 = 0.25 * y → (x + y) / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_of_x_and_y_is_16_l1719_171990


namespace NUMINAMATH_CALUDE_no_integer_solution_2016_equation_l1719_171913

theorem no_integer_solution_2016_equation :
  ¬∃ (x y z : ℤ), (2016 : ℚ) = (x^2 + y^2 + z^2 : ℚ) / (x*y + y*z + z*x : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_2016_equation_l1719_171913


namespace NUMINAMATH_CALUDE_amy_yard_area_l1719_171940

theorem amy_yard_area :
  ∀ (short_posts long_posts : ℕ) 
    (post_distance : ℝ) 
    (total_posts : ℕ),
  short_posts > 1 →
  long_posts > 1 →
  post_distance > 0 →
  total_posts = 24 →
  long_posts = (3 * short_posts) / 2 →
  total_posts = 2 * short_posts + 2 * long_posts - 4 →
  post_distance = 3 →
  (short_posts - 1 : ℝ) * post_distance * ((long_posts - 1 : ℝ) * post_distance) = 189 :=
by sorry

end NUMINAMATH_CALUDE_amy_yard_area_l1719_171940


namespace NUMINAMATH_CALUDE_quadratic_through_origin_l1719_171959

theorem quadratic_through_origin (a : ℝ) : 
  (∀ x y : ℝ, y = (a - 1) * x^2 - x + a^2 - 1 → (x = 0 → y = 0)) → 
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_through_origin_l1719_171959


namespace NUMINAMATH_CALUDE_no_prime_sum_72_l1719_171905

theorem no_prime_sum_72 : ¬∃ (p q k : ℕ), Prime p ∧ Prime q ∧ p + q = 72 ∧ p * q = k := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_72_l1719_171905


namespace NUMINAMATH_CALUDE_expression_evaluation_l1719_171995

theorem expression_evaluation (b c : ℕ) (h1 : b = 2) (h2 : c = 3) : 
  (b^3 * b^4) + c^2 = 137 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1719_171995


namespace NUMINAMATH_CALUDE_nine_possible_values_for_D_l1719_171998

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define the addition equation
def addition_equation (A B C D : Digit) : Prop :=
  10000 * A.val + 1000 * B.val + 100 * A.val + 10 * C.val + A.val +
  10000 * C.val + 1000 * A.val + 100 * D.val + 10 * A.val + B.val =
  10000 * D.val + 1000 * C.val + 100 * D.val + 10 * D.val + D.val

-- Define the distinct digits condition
def distinct_digits (A B C D : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Theorem statement
theorem nine_possible_values_for_D :
  ∃ (s : Finset Digit),
    s.card = 9 ∧
    (∀ D ∈ s, ∃ A B C, distinct_digits A B C D ∧ addition_equation A B C D) ∧
    (∀ D, D ∉ s → ¬∃ A B C, distinct_digits A B C D ∧ addition_equation A B C D) :=
sorry

end NUMINAMATH_CALUDE_nine_possible_values_for_D_l1719_171998


namespace NUMINAMATH_CALUDE_chemical_mixture_composition_l1719_171986

/-- Given the composition of two chemical solutions and their mixture, 
    prove the percentage of chemical b in solution y -/
theorem chemical_mixture_composition 
  (x_a : Real) (x_b : Real) (y_a : Real) (y_b : Real) 
  (mix_a : Real) (mix_x : Real) : 
  x_a = 0.1 → 
  x_b = 0.9 → 
  y_a = 0.2 → 
  mix_a = 0.12 → 
  mix_x = 0.8 → 
  y_b = 0.8 := by
  sorry

#check chemical_mixture_composition

end NUMINAMATH_CALUDE_chemical_mixture_composition_l1719_171986


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1719_171941

theorem simplify_trig_expression :
  let tan_45 : ℝ := 1
  let cot_45 : ℝ := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1719_171941


namespace NUMINAMATH_CALUDE_solution_when_m_is_one_solution_for_general_m_l1719_171987

-- Define the inequality
def inequality (m x : ℝ) : Prop := (2*m - m*x)/2 > x/2 - 1

-- Theorem for part 1
theorem solution_when_m_is_one :
  ∀ x : ℝ, inequality 1 x ↔ x < 2 := by sorry

-- Theorem for part 2
theorem solution_for_general_m :
  ∀ m x : ℝ, m ≠ -1 →
    (inequality m x ↔ (m > -1 ∧ x < 2) ∨ (m < -1 ∧ x > 2)) := by sorry

end NUMINAMATH_CALUDE_solution_when_m_is_one_solution_for_general_m_l1719_171987


namespace NUMINAMATH_CALUDE_bridge_length_l1719_171999

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 235 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1719_171999


namespace NUMINAMATH_CALUDE_min_vertical_distance_l1719_171908

/-- The absolute value function -/
def f (x : ℝ) : ℝ := |x|

/-- The quadratic function -/
def g (x : ℝ) : ℝ := -x^2 - 4*x - 3

/-- The theorem stating the minimum vertical distance between f and g -/
theorem min_vertical_distance :
  ∃ (min : ℝ), min = 3/4 ∧ ∀ (x : ℝ), |f x - g x| ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l1719_171908


namespace NUMINAMATH_CALUDE_score_change_effect_l1719_171924

/-- Proves that changing one student's score from 86 to 74 in a group of 8 students
    with an initial average of 82.5 decreases the average by 1.5 points -/
theorem score_change_effect (n : ℕ) (initial_avg : ℚ) (old_score new_score : ℚ) :
  n = 8 →
  initial_avg = 82.5 →
  old_score = 86 →
  new_score = 74 →
  initial_avg - (n * initial_avg - old_score + new_score) / n = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_score_change_effect_l1719_171924


namespace NUMINAMATH_CALUDE_circle_symmetry_l1719_171942

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

-- Define circle C1
def circle_C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define circle C2
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

-- Define symmetry relation between two points with respect to the line
def symmetric_points (x1 y1 x2 y2 : ℝ) : Prop :=
  line_of_symmetry ((x1 + x2) / 2) ((y1 + y2) / 2)

-- Theorem statement
theorem circle_symmetry :
  ∀ x y : ℝ,
  (∃ x1 y1 : ℝ, circle_C1 x1 y1 ∧ symmetric_points x1 y1 x y) →
  circle_C2 x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1719_171942


namespace NUMINAMATH_CALUDE_mary_bought_48_cards_l1719_171907

/-- Calculates the number of baseball cards Mary bought -/
def cards_mary_bought (initial_cards : ℕ) (torn_cards : ℕ) (cards_from_fred : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial_cards - torn_cards + cards_from_fred)

/-- Proves that Mary bought 48 baseball cards -/
theorem mary_bought_48_cards : cards_mary_bought 18 8 26 84 = 48 := by
  sorry

#eval cards_mary_bought 18 8 26 84

end NUMINAMATH_CALUDE_mary_bought_48_cards_l1719_171907


namespace NUMINAMATH_CALUDE_triangle_condition_l1719_171926

def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + 4 + k^2

theorem triangle_condition (k : ℝ) : 
  (∀ a b c : ℝ, 0 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 3 → 
    f k a + f k b > f k c ∧ 
    f k b + f k c > f k a ∧ 
    f k c + f k a > f k b) ↔ 
  k > 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_condition_l1719_171926


namespace NUMINAMATH_CALUDE_sandy_jacket_return_l1719_171988

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℝ := 13.99

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℝ := 12.14

/-- The net amount Sandy spent on clothes -/
def net_spent : ℝ := 18.70

/-- The amount Sandy received for returning the jacket -/
def jacket_return : ℝ := shorts_cost + shirt_cost - net_spent

theorem sandy_jacket_return : jacket_return = 7.43 := by
  sorry

end NUMINAMATH_CALUDE_sandy_jacket_return_l1719_171988


namespace NUMINAMATH_CALUDE_statement_B_statement_D_l1719_171963

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (contained : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Statement B
theorem statement_B :
  perpendicular m n →
  perpendicular_plane m α →
  perpendicular_plane n β →
  perpendicular_planes α β :=
sorry

-- Statement D
theorem statement_D :
  parallel_planes α β →
  perpendicular_plane m α →
  parallel n β →
  perpendicular m n :=
sorry

end NUMINAMATH_CALUDE_statement_B_statement_D_l1719_171963


namespace NUMINAMATH_CALUDE_other_denomination_is_70_l1719_171900

/-- Proves that the other denomination of travelers checks is $70 --/
theorem other_denomination_is_70 
  (total_checks : ℕ)
  (total_worth : ℕ)
  (known_denomination : ℕ)
  (known_count : ℕ)
  (remaining_average : ℕ)
  (h1 : total_checks = 30)
  (h2 : total_worth = 1800)
  (h3 : known_denomination = 50)
  (h4 : known_count = 15)
  (h5 : remaining_average = 70)
  (h6 : known_count * known_denomination + (total_checks - known_count) * remaining_average = total_worth) :
  ∃ (other_denomination : ℕ), other_denomination = 70 ∧ 
    known_count * known_denomination + (total_checks - known_count) * other_denomination = total_worth :=
by sorry

end NUMINAMATH_CALUDE_other_denomination_is_70_l1719_171900


namespace NUMINAMATH_CALUDE_property_P_implies_m_range_l1719_171948

open Real

/-- Property P(a) for a function f -/
def has_property_P (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ h : ℝ → ℝ, (∀ x > 1, h x > 0) ∧
    (∀ x > 1, deriv f x = h x * (x^2 - a*x + 1))

theorem property_P_implies_m_range
  (g : ℝ → ℝ) (hg : has_property_P g 2)
  (x₁ x₂ : ℝ) (hx : 1 < x₁ ∧ x₁ < x₂)
  (m : ℝ) (α β : ℝ)
  (hα : α = m*x₁ + (1-m)*x₂)
  (hβ : β = (1-m)*x₁ + m*x₂)
  (hαβ : α > 1 ∧ β > 1)
  (hineq : |g α - g β| < |g x₁ - g x₂|) :
  0 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_property_P_implies_m_range_l1719_171948


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1719_171944

theorem partial_fraction_decomposition (x : ℝ) (A B C : ℝ) :
  (1 : ℝ) / (x^3 - 7*x^2 + 10*x + 24) = A / (x - 2) + B / (x - 6) + C / (x - 6)^2 →
  x^3 - 7*x^2 + 10*x + 24 = (x - 2) * (x - 6)^2 →
  A = 1/16 := by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1719_171944


namespace NUMINAMATH_CALUDE_total_ladybugs_l1719_171937

theorem total_ladybugs (num_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
  (h1 : num_leaves = 84) 
  (h2 : ladybugs_per_leaf = 139) : 
  num_leaves * ladybugs_per_leaf = 11676 := by
  sorry

end NUMINAMATH_CALUDE_total_ladybugs_l1719_171937


namespace NUMINAMATH_CALUDE_infinite_inequality_occurrences_l1719_171979

theorem infinite_inequality_occurrences (a : ℕ → ℕ+) : 
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
    ∀ n ∈ S, (1 : ℝ) + a n > (a (n-1) : ℝ) * (2 : ℝ) ^ (1 / n) :=
sorry

end NUMINAMATH_CALUDE_infinite_inequality_occurrences_l1719_171979


namespace NUMINAMATH_CALUDE_sin_cos_roots_quadratic_l1719_171902

theorem sin_cos_roots_quadratic (θ : Real) (a : Real) : 
  (4 * Real.sin θ ^ 2 + 2 * a * Real.sin θ + a = 0) ∧ 
  (4 * Real.cos θ ^ 2 + 2 * a * Real.cos θ + a = 0) →
  a = 1 - Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_roots_quadratic_l1719_171902


namespace NUMINAMATH_CALUDE_point_distance_from_x_axis_l1719_171992

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (p : Point) : ℝ := |p.y|

theorem point_distance_from_x_axis (a : ℝ) :
  let p : Point := ⟨2, a⟩
  distanceFromXAxis p = 3 → a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_from_x_axis_l1719_171992


namespace NUMINAMATH_CALUDE_james_two_point_shots_l1719_171947

/-- Represents the number of 2-point shots scored by James -/
def two_point_shots : ℕ := sorry

/-- Represents the number of 3-point shots scored by James -/
def three_point_shots : ℕ := 13

/-- Represents the total points scored by James -/
def total_points : ℕ := 79

/-- Theorem stating that James scored 20 two-point shots -/
theorem james_two_point_shots : 
  two_point_shots = 20 ∧ 
  2 * two_point_shots + 3 * three_point_shots = total_points := by
  sorry

end NUMINAMATH_CALUDE_james_two_point_shots_l1719_171947


namespace NUMINAMATH_CALUDE_exists_hole_free_square_meter_l1719_171956

/-- Represents a point on the carpet -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the carpet with its dimensions and holes -/
structure Carpet where
  side_length : ℝ
  holes : Finset Point

/-- Represents a square piece that could be cut from the carpet -/
structure SquarePiece where
  bottom_left : Point
  side_length : ℝ

/-- Checks if a point is inside a square piece -/
def point_in_square (p : Point) (s : SquarePiece) : Prop :=
  s.bottom_left.x ≤ p.x ∧ p.x < s.bottom_left.x + s.side_length ∧
  s.bottom_left.y ≤ p.y ∧ p.y < s.bottom_left.y + s.side_length

/-- The main theorem to be proved -/
theorem exists_hole_free_square_meter (c : Carpet) 
    (h_side : c.side_length = 275)
    (h_holes : c.holes.card = 4) :
    ∃ (s : SquarePiece), s.side_length = 100 ∧ 
    s.bottom_left.x + s.side_length ≤ c.side_length ∧
    s.bottom_left.y + s.side_length ≤ c.side_length ∧
    ∀ (p : Point), p ∈ c.holes → ¬point_in_square p s :=
  sorry

end NUMINAMATH_CALUDE_exists_hole_free_square_meter_l1719_171956


namespace NUMINAMATH_CALUDE_pen_collection_problem_l1719_171934

/-- Represents the pen collection problem --/
theorem pen_collection_problem (initial_pens : ℕ) (final_pens : ℕ) (sharon_pens : ℕ) 
  (h1 : initial_pens = 25)
  (h2 : final_pens = 75)
  (h3 : sharon_pens = 19) :
  ∃ (mike_pens : ℕ), 2 * (initial_pens + mike_pens) - sharon_pens = final_pens ∧ mike_pens = 22 := by
  sorry

end NUMINAMATH_CALUDE_pen_collection_problem_l1719_171934


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1719_171933

/-- Given two circles r and s, if the diameter of r is 50% of the diameter of s,
    then the area of r is 25% of the area of s. -/
theorem circle_area_ratio (r s : Real) (hr : r > 0) (hs : s > 0) 
  (h_diameter : 2 * r = 0.5 * (2 * s)) : 
  π * r^2 = 0.25 * (π * s^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1719_171933


namespace NUMINAMATH_CALUDE_warrens_event_capacity_l1719_171946

theorem warrens_event_capacity :
  let total_tables : ℕ := 252
  let large_tables : ℕ := 93
  let medium_tables : ℕ := 97
  let small_tables : ℕ := total_tables - large_tables - medium_tables
  let unusable_small_tables : ℕ := 20
  let usable_small_tables : ℕ := small_tables - unusable_small_tables
  let large_table_capacity : ℕ := 6
  let medium_table_capacity : ℕ := 5
  let small_table_capacity : ℕ := 4
  
  large_tables * large_table_capacity +
  medium_tables * medium_table_capacity +
  usable_small_tables * small_table_capacity = 1211 :=
by
  sorry

#eval
  let total_tables : ℕ := 252
  let large_tables : ℕ := 93
  let medium_tables : ℕ := 97
  let small_tables : ℕ := total_tables - large_tables - medium_tables
  let unusable_small_tables : ℕ := 20
  let usable_small_tables : ℕ := small_tables - unusable_small_tables
  let large_table_capacity : ℕ := 6
  let medium_table_capacity : ℕ := 5
  let small_table_capacity : ℕ := 4
  
  large_tables * large_table_capacity +
  medium_tables * medium_table_capacity +
  usable_small_tables * small_table_capacity

end NUMINAMATH_CALUDE_warrens_event_capacity_l1719_171946


namespace NUMINAMATH_CALUDE_car_wash_remaining_amount_l1719_171976

def car_wash_fundraiser (goal : ℕ) (families_10 : ℕ) (amount_10 : ℕ) (families_5 : ℕ) (amount_5 : ℕ) : ℕ :=
  goal - (families_10 * amount_10 + families_5 * amount_5)

theorem car_wash_remaining_amount :
  car_wash_fundraiser 150 3 10 15 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_car_wash_remaining_amount_l1719_171976


namespace NUMINAMATH_CALUDE_smallest_valid_n_l1719_171920

def is_valid (n : ℕ) : Prop :=
  Real.sqrt (n : ℝ) - Real.sqrt ((n - 1) : ℝ) < 0.01

theorem smallest_valid_n :
  (∀ m : ℕ, m < 2501 → ¬(is_valid m)) ∧ is_valid 2501 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l1719_171920


namespace NUMINAMATH_CALUDE_total_pies_eq_750_l1719_171925

/-- The number of mini meat pies made by the first team -/
def team1_pies : ℕ := 235

/-- The number of mini meat pies made by the second team -/
def team2_pies : ℕ := 275

/-- The number of mini meat pies made by the third team -/
def team3_pies : ℕ := 240

/-- The total number of teams -/
def num_teams : ℕ := 3

/-- The total number of mini meat pies made by all teams -/
def total_pies : ℕ := team1_pies + team2_pies + team3_pies

theorem total_pies_eq_750 : total_pies = 750 := by
  sorry

end NUMINAMATH_CALUDE_total_pies_eq_750_l1719_171925


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_2010_l1719_171973

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_2010 : 
  units_digit (factorial_sum 2010) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_2010_l1719_171973


namespace NUMINAMATH_CALUDE_exactly_six_numbers_l1719_171977

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧
  ∃ k : ℕ, n - reverse_digits n = k^3 ∧ k > 0

theorem exactly_six_numbers :
  ∃! (s : Finset ℕ), s.card = 6 ∧ ∀ n, n ∈ s ↔ satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_exactly_six_numbers_l1719_171977


namespace NUMINAMATH_CALUDE_clock_sale_second_price_l1719_171930

/-- Represents the sale and resale of a clock in a shop. -/
def ClockSale (original_cost : ℝ) : Prop :=
  let first_sale_price := 1.2 * original_cost
  let buy_back_price := 0.6 * original_cost
  let second_sale_price := 1.08 * original_cost
  (original_cost - buy_back_price = 100) ∧
  (second_sale_price = 270)

/-- Proves that the shop's second selling price of the clock is $270 given the conditions. -/
theorem clock_sale_second_price :
  ∃ (original_cost : ℝ), ClockSale original_cost :=
sorry

end NUMINAMATH_CALUDE_clock_sale_second_price_l1719_171930


namespace NUMINAMATH_CALUDE_last_score_entered_last_score_is_95_l1719_171984

def scores : List ℕ := [75, 81, 85, 87, 95]

def is_integer_average (subset : List ℕ) : Prop :=
  ∃ n : ℕ, n * subset.length = subset.sum

theorem last_score_entered (last : ℕ) : Prop :=
  last ∈ scores ∧
  ∀ subset : List ℕ, subset ⊆ scores → last ∈ subset →
    is_integer_average subset

theorem last_score_is_95 : 
  ∃ last : ℕ, last_score_entered last ∧ last = 95 := by
  sorry

end NUMINAMATH_CALUDE_last_score_entered_last_score_is_95_l1719_171984


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1719_171952

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 8 * x + 15 = 0 ↔ (x = 3 ∨ x = 5)) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1719_171952


namespace NUMINAMATH_CALUDE_august_mail_total_l1719_171954

/-- Calculates the total number of pieces of mail sent in August --/
def august_mail (july_mail : ℕ) (business_days : ℕ) (weekend_days : ℕ) : ℕ :=
  (2 * july_mail * business_days) + (july_mail / 2 * weekend_days)

/-- Theorem stating the total number of pieces of mail sent in August --/
theorem august_mail_total : august_mail 40 22 9 = 1940 := by
  sorry

end NUMINAMATH_CALUDE_august_mail_total_l1719_171954


namespace NUMINAMATH_CALUDE_prime_sum_product_l1719_171981

theorem prime_sum_product (p₁ p₂ p₃ p₄ : ℕ) 
  (h_prime₁ : Nat.Prime p₁) (h_prime₂ : Nat.Prime p₂) 
  (h_prime₃ : Nat.Prime p₃) (h_prime₄ : Nat.Prime p₄)
  (h_order : p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄)
  (h_sum : p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882) :
  p₁ = 7 ∧ p₂ = 11 ∧ p₃ = 13 ∧ p₄ = 17 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l1719_171981


namespace NUMINAMATH_CALUDE_sum_of_divisors_330_l1719_171957

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_330 : sum_of_divisors 330 = 864 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_330_l1719_171957


namespace NUMINAMATH_CALUDE_equation_solutions_l1719_171993

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4) ∧
  (∀ x : ℝ, -2 * (x^3 - 1) = 18 ↔ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1719_171993


namespace NUMINAMATH_CALUDE_world_grain_ratio_2010_l1719_171997

theorem world_grain_ratio_2010 : 
  let supply : ℕ := 1800000
  let demand : ℕ := 2400000
  (supply : ℚ) / demand = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_world_grain_ratio_2010_l1719_171997


namespace NUMINAMATH_CALUDE_modulus_of_imaginary_unit_l1719_171989

theorem modulus_of_imaginary_unit (z : ℂ) (h : z^2 + 1 = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_imaginary_unit_l1719_171989


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1719_171903

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : 
  k = 2008^2 + 2^2008 → (k^2 + 2^k) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1719_171903


namespace NUMINAMATH_CALUDE_flowers_after_one_month_l1719_171931

/-- Represents the number of flowers in Mark's garden -/
structure GardenFlowers where
  yellow : ℕ
  purple : ℕ
  green : ℕ
  red : ℕ

/-- Calculates the number of flowers after one month -/
def flowersAfterOneMonth (initial : GardenFlowers) : ℕ :=
  let yellowAfter := initial.yellow + (initial.yellow / 2)
  let purpleAfter := initial.purple * 2
  let greenAfter := initial.green - (initial.green / 5)
  let redAfter := initial.red + (initial.red * 4 / 5)
  yellowAfter + purpleAfter + greenAfter + redAfter

/-- Theorem stating the number of flowers after one month -/
theorem flowers_after_one_month :
  ∃ (initial : GardenFlowers),
    initial.yellow = 10 ∧
    initial.purple = initial.yellow + (initial.yellow * 4 / 5) ∧
    initial.green = (initial.yellow + initial.purple) / 4 ∧
    initial.red = ((initial.yellow + initial.purple + initial.green) * 35) / 100 ∧
    flowersAfterOneMonth initial = 77 :=
  sorry

end NUMINAMATH_CALUDE_flowers_after_one_month_l1719_171931


namespace NUMINAMATH_CALUDE_next_integer_divisibility_l1719_171971

theorem next_integer_divisibility (n : ℕ) :
  ∃ k : ℤ, (k : ℝ) = ⌊(Real.sqrt 3 + 1)^(2*n)⌋ + 1 ∧ (2^(n+1) : ℤ) ∣ k :=
sorry

end NUMINAMATH_CALUDE_next_integer_divisibility_l1719_171971


namespace NUMINAMATH_CALUDE_sues_necklace_beads_l1719_171951

/-- The number of beads in Sue's necklace -/
def total_beads (purple : ℕ) (blue : ℕ) (green : ℕ) : ℕ :=
  purple + blue + green

/-- Theorem stating the total number of beads in Sue's necklace -/
theorem sues_necklace_beads : 
  ∀ (purple blue green : ℕ),
    purple = 7 →
    blue = 2 * purple →
    green = blue + 11 →
    total_beads purple blue green = 46 := by
  sorry

end NUMINAMATH_CALUDE_sues_necklace_beads_l1719_171951


namespace NUMINAMATH_CALUDE_binomial_factorial_l1719_171939

theorem binomial_factorial : Nat.factorial (Nat.choose 8 5) = Nat.factorial 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_factorial_l1719_171939


namespace NUMINAMATH_CALUDE_local_max_at_two_l1719_171917

def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

theorem local_max_at_two (c : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), f c x ≤ f c 2) →
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_local_max_at_two_l1719_171917


namespace NUMINAMATH_CALUDE_max_difference_reverse_digits_l1719_171980

/-- Two-digit positive integer -/
def TwoDigitInt (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Main theorem -/
theorem max_difference_reverse_digits (q r : ℕ) :
  TwoDigitInt q ∧ TwoDigitInt r ∧
  r = reverseDigits q ∧
  q > r ∧
  q - r < 20 →
  q - r ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_max_difference_reverse_digits_l1719_171980


namespace NUMINAMATH_CALUDE_total_pens_l1719_171912

theorem total_pens (black_pens blue_pens : ℕ) : 
  black_pens = 4 → blue_pens = 4 → black_pens + blue_pens = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_pens_l1719_171912


namespace NUMINAMATH_CALUDE_product_of_roots_equals_one_l1719_171966

theorem product_of_roots_equals_one :
  let A := Real.sqrt 2019 + Real.sqrt 2020 + 1
  let B := -Real.sqrt 2019 - Real.sqrt 2020 - 1
  let C := Real.sqrt 2019 - Real.sqrt 2020 + 1
  let D := Real.sqrt 2020 - Real.sqrt 2019 - 1
  A * B * C * D = 1 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_equals_one_l1719_171966


namespace NUMINAMATH_CALUDE_product_digits_sum_base7_l1719_171961

/-- Converts a base-7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Sums the digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

theorem product_digits_sum_base7 :
  let a := 35
  let b := 42
  let product := toBase7 (toDecimal a * toDecimal b)
  sumDigitsBase7 product = 15
  := by sorry

end NUMINAMATH_CALUDE_product_digits_sum_base7_l1719_171961


namespace NUMINAMATH_CALUDE_nested_square_root_simplification_l1719_171949

theorem nested_square_root_simplification :
  Real.sqrt (25 * Real.sqrt (25 * Real.sqrt 25)) = 5 * (5 ^ (3/4)) := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_simplification_l1719_171949


namespace NUMINAMATH_CALUDE_set_of_values_for_a_l1719_171985

theorem set_of_values_for_a (a : ℝ) : 
  (2 ∉ {x : ℝ | x - a < 0}) ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_set_of_values_for_a_l1719_171985


namespace NUMINAMATH_CALUDE_second_month_sale_l1719_171943

def sales_data : List ℕ := [800, 1000, 700, 800, 900]
def num_months : ℕ := 6
def average_sale : ℕ := 850

theorem second_month_sale :
  ∃ (second_month : ℕ),
    (List.sum sales_data + second_month) / num_months = average_sale ∧
    second_month = 900 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l1719_171943


namespace NUMINAMATH_CALUDE_one_root_condition_l1719_171958

theorem one_root_condition (k : ℝ) : 
  (∃! x : ℝ, Real.log (k * x) = 2 * Real.log (x + 1)) → (k < 0 ∨ k = 4) :=
sorry

end NUMINAMATH_CALUDE_one_root_condition_l1719_171958


namespace NUMINAMATH_CALUDE_equation_solution_l1719_171923

theorem equation_solution : ∃ x : ℝ, (x - 3) ^ 4 = (1 / 16)⁻¹ ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1719_171923


namespace NUMINAMATH_CALUDE_sum_of_tens_digits_l1719_171901

/-- Given single-digit numbers A, B, C, D such that A + B + C + D = 22,
    the sum of the tens digits of (A + B) and (C + D) equals 4. -/
theorem sum_of_tens_digits (A B C D : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10)
    (h5 : A + B + C + D = 22) : (A + B) / 10 + (C + D) / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tens_digits_l1719_171901


namespace NUMINAMATH_CALUDE_village_walk_speeds_l1719_171935

/-- Proves that given the conditions of the problem, the speeds of the two people are 2 km/h and 5 km/h respectively. -/
theorem village_walk_speeds (distance : ℝ) (speed_diff : ℝ) (time_diff : ℝ)
  (h1 : distance = 10)
  (h2 : speed_diff = 3)
  (h3 : time_diff = 3)
  (h4 : ∀ x : ℝ, distance / x = distance / (x + speed_diff) + time_diff → x = 2) :
  ∃ (speed1 speed2 : ℝ), speed1 = 2 ∧ speed2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_village_walk_speeds_l1719_171935


namespace NUMINAMATH_CALUDE_linear_function_problem_l1719_171964

-- Define a linear function f
def f (x : ℝ) : ℝ := sorry

-- Define the inverse function of f
def f_inv (x : ℝ) : ℝ := sorry

-- State the theorem
theorem linear_function_problem :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) →  -- f is linear
  (∀ x : ℝ, f x = 3 * f_inv x + 9) →         -- f(x) = 3f^(-1)(x) + 9
  f 3 = 6 →                                  -- f(3) = 6
  f 6 = 10.5 * Real.sqrt 3 - 4.5 :=          -- f(6) = 10.5√3 - 4.5
by sorry

end NUMINAMATH_CALUDE_linear_function_problem_l1719_171964


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l1719_171970

-- Define the function f
def f (x : ℝ) : ℝ := x^2 * (2 - x)

-- Theorem statement
theorem f_strictly_increasing : 
  ∀ x y, 0 < x ∧ x < y ∧ y < 4/3 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l1719_171970


namespace NUMINAMATH_CALUDE_tournament_games_count_l1719_171983

/-- Represents a basketball tournament with a preliminary round and main tournament. -/
structure BasketballTournament where
  preliminaryTeams : Nat
  preliminarySpots : Nat
  mainTournamentTeams : Nat

/-- Calculates the number of games in the preliminary round. -/
def preliminaryGames (t : BasketballTournament) : Nat :=
  t.preliminarySpots

/-- Calculates the number of games in the main tournament. -/
def mainTournamentGames (t : BasketballTournament) : Nat :=
  t.mainTournamentTeams - 1

/-- Calculates the total number of games in the entire tournament. -/
def totalGames (t : BasketballTournament) : Nat :=
  preliminaryGames t + mainTournamentGames t

/-- Theorem stating that the total number of games in the specific tournament setup is 17. -/
theorem tournament_games_count :
  ∃ (t : BasketballTournament),
    t.preliminaryTeams = 4 ∧
    t.preliminarySpots = 2 ∧
    t.mainTournamentTeams = 16 ∧
    totalGames t = 17 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_count_l1719_171983


namespace NUMINAMATH_CALUDE_cleaning_frequency_in_year_l1719_171962

/-- The number of times a person cleans themselves in 52 weeks, given they take
    a bath twice a week and a shower once a week. -/
def cleaningFrequency (bathsPerWeek showerPerWeek weeksInYear : ℕ) : ℕ :=
  (bathsPerWeek + showerPerWeek) * weeksInYear

/-- Theorem stating that a person who takes a bath twice a week and a shower once a week
    cleans themselves 156 times in 52 weeks. -/
theorem cleaning_frequency_in_year :
  cleaningFrequency 2 1 52 = 156 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_frequency_in_year_l1719_171962


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1719_171922

/-- Given a journey of 234 miles that takes 27/4 hours, prove that the average speed is 936/27 miles per hour. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 234) (h2 : time = 27/4) :
  distance / time = 936 / 27 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1719_171922


namespace NUMINAMATH_CALUDE_f_min_value_inequality_property_l1719_171950

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Theorem for the minimum value of f
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 4) ∧ (∃ x : ℝ, f x = 4) := by sorry

-- Theorem for the inequality
theorem inequality_property (a b x : ℝ) (ha : |a| < 2) (hb : |b| < 2) :
  |a + b| + |a - b| < f x := by sorry

end NUMINAMATH_CALUDE_f_min_value_inequality_property_l1719_171950


namespace NUMINAMATH_CALUDE_g_properties_imply_g_50_l1719_171906

noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem g_properties_imply_g_50 (p q r s : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  g p q r s 23 = 23 →
  g p q r s 101 = 101 →
  (∀ x : ℝ, x ≠ -s/r → g p q r s (g p q r s x) = x) →
  g p q r s 50 = -61 := by sorry

end NUMINAMATH_CALUDE_g_properties_imply_g_50_l1719_171906


namespace NUMINAMATH_CALUDE_square_binomial_equality_l1719_171919

theorem square_binomial_equality (a b : ℝ) : 
  (-1 + a * b^2)^2 = 1 - 2 * a * b^2 + a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_equality_l1719_171919


namespace NUMINAMATH_CALUDE_larger_number_is_ten_l1719_171915

theorem larger_number_is_ten (x y : ℕ) (h1 : x * y = 30) (h2 : x + y = 13) : 
  max x y = 10 := by
sorry

end NUMINAMATH_CALUDE_larger_number_is_ten_l1719_171915


namespace NUMINAMATH_CALUDE_prism_with_hole_volume_formula_l1719_171918

/-- The volume of a rectangular prism with a hole running through it -/
def prism_with_hole_volume (x : ℝ) : ℝ :=
  let large_prism_volume := (x + 8) * (x + 6) * 4
  let hole_volume := (2*x - 4) * (x - 3) * 4
  large_prism_volume - hole_volume

/-- Theorem stating the volume of the prism with a hole -/
theorem prism_with_hole_volume_formula (x : ℝ) :
  prism_with_hole_volume x = -4*x^2 + 96*x + 144 :=
by sorry

end NUMINAMATH_CALUDE_prism_with_hole_volume_formula_l1719_171918


namespace NUMINAMATH_CALUDE_ferry_river_crossing_l1719_171975

/-- The width of a river crossed by two ferries --/
def river_width : ℝ := 1280

/-- The distance from the nearest shore where the ferries first meet --/
def first_meeting_distance : ℝ := 720

/-- The distance from the other shore where the ferries meet on the return trip --/
def second_meeting_distance : ℝ := 400

/-- Theorem stating that the width of the river is 1280 meters given the conditions --/
theorem ferry_river_crossing :
  let w := river_width
  let d1 := first_meeting_distance
  let d2 := second_meeting_distance
  (d1 + (w - d1) = w) ∧
  (3 * w = 2 * w + 2 * d1) ∧
  (3 * d1 = 2 * w - d2) →
  w = 1280 := by sorry


end NUMINAMATH_CALUDE_ferry_river_crossing_l1719_171975


namespace NUMINAMATH_CALUDE_august_matches_l1719_171911

/-- Calculates the number of matches played in August given the initial and final winning percentages and the number of additional matches won. -/
def matches_in_august (initial_percentage : ℚ) (final_percentage : ℚ) (additional_wins : ℕ) : ℕ :=
  sorry

theorem august_matches :
  matches_in_august (22 / 100) (52 / 100) 75 = 120 :=
sorry

end NUMINAMATH_CALUDE_august_matches_l1719_171911


namespace NUMINAMATH_CALUDE_power_division_equality_l1719_171969

theorem power_division_equality : (10^8 : ℝ) / (2 * 10^6) = 50 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l1719_171969
