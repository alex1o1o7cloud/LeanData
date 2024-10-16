import Mathlib

namespace NUMINAMATH_CALUDE_megan_books_count_l3170_317074

theorem megan_books_count :
  ∀ (m k g : ℕ),
  k = m / 4 →
  g = 2 * k + 9 →
  m + k + g = 65 →
  m = 32 :=
by sorry

end NUMINAMATH_CALUDE_megan_books_count_l3170_317074


namespace NUMINAMATH_CALUDE_solve_for_a_find_b_find_c_find_d_l3170_317044

-- Part 1
def simultaneous_equations (a u : ℝ) : Prop :=
  3/a + 1/u = 7/2 ∧ 2/a - 3/u = 6

theorem solve_for_a : ∃ a u : ℝ, simultaneous_equations a u ∧ a = 3/2 :=
sorry

-- Part 2
def equation_with_solutions (p q b : ℝ) (a : ℝ) : Prop :=
  p * 0 + q * (3*a) + b * 1 = 1 ∧
  p * (9*a) + q * (-1) + b * 2 = 1 ∧
  p * 0 + q * (3*a) + b * 0 = 1

theorem find_b : ∃ p q b a : ℝ, equation_with_solutions p q b a ∧ b = 0 :=
sorry

-- Part 3
def line_through_points (m c b : ℝ) : Prop :=
  5 = m * (b + 4) + c ∧
  2 = m * (-2) + c

theorem find_c : ∃ m c b : ℝ, line_through_points m c b ∧ c = 3 :=
sorry

-- Part 4
def inequality_solution (c d : ℝ) : Prop :=
  ∀ x : ℝ, d ≤ x ∧ x ≤ 1 ↔ x^2 + 5*x - 2*c ≤ 0

theorem find_d : ∃ c d : ℝ, inequality_solution c d ∧ d = -6 :=
sorry

end NUMINAMATH_CALUDE_solve_for_a_find_b_find_c_find_d_l3170_317044


namespace NUMINAMATH_CALUDE_square_cut_perimeter_sum_l3170_317027

theorem square_cut_perimeter_sum (s : Real) 
  (h1 : s > 0) 
  (h2 : ∃ (a b c d : Real), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
        a + b = 1 ∧ c + d = 1 ∧
        s = 2*(a+b) + 2*(c+d) + 2*(a+c) + 2*(b+d)) :
  s = 8 ∨ s = 10 := by
sorry

end NUMINAMATH_CALUDE_square_cut_perimeter_sum_l3170_317027


namespace NUMINAMATH_CALUDE_computer_purchase_cost_effectiveness_l3170_317088

def store_A_cost (x : ℕ) : ℝ := 4500 * x + 1500
def store_B_cost (x : ℕ) : ℝ := 4800 * x

theorem computer_purchase_cost_effectiveness (x : ℕ) :
  (x < 5 → store_B_cost x < store_A_cost x) ∧
  (x > 5 → store_A_cost x < store_B_cost x) ∧
  (x = 5 → store_A_cost x = store_B_cost x) := by
  sorry

end NUMINAMATH_CALUDE_computer_purchase_cost_effectiveness_l3170_317088


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l3170_317065

theorem base_10_to_base_7 : 
  (2 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0 : ℕ) = 789 := by
sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l3170_317065


namespace NUMINAMATH_CALUDE_prob_divisible_by_3_and_8_prob_divisible_by_3_and_8_value_l3170_317023

/-- The probability of a randomly selected three-digit number being divisible by both 3 and 8 -/
theorem prob_divisible_by_3_and_8 : ℚ :=
  let three_digit_numbers := Finset.Icc 100 999
  let divisible_by_24 := three_digit_numbers.filter (λ n => n % 24 = 0)
  (divisible_by_24.card : ℚ) / three_digit_numbers.card

/-- The probability of a randomly selected three-digit number being divisible by both 3 and 8 is 37/900 -/
theorem prob_divisible_by_3_and_8_value :
  prob_divisible_by_3_and_8 = 37 / 900 := by
  sorry


end NUMINAMATH_CALUDE_prob_divisible_by_3_and_8_prob_divisible_by_3_and_8_value_l3170_317023


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3170_317029

/-- Given two vectors a and b in ℝ², and a real number k, 
    we define vector c as a sum of a and k * b. 
    If b is perpendicular to c, then k equals -3. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) :
  a = (10, 20) →
  b = (5, 5) →
  let c := (a.1 + k * b.1, a.2 + k * b.2)
  (b.1 * c.1 + b.2 * c.2 = 0) →
  k = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3170_317029


namespace NUMINAMATH_CALUDE_sum_faces_edges_vertices_num_diagonals_square_base_l3170_317040

/-- A square pyramid is a polyhedron with a square base and triangular sides. -/
structure SquarePyramid where
  /-- The number of faces in a square pyramid -/
  faces : Nat
  /-- The number of edges in a square pyramid -/
  edges : Nat
  /-- The number of vertices in a square pyramid -/
  vertices : Nat
  /-- The number of sides in the base of a square pyramid -/
  base_sides : Nat

/-- Properties of a square pyramid -/
def square_pyramid : SquarePyramid :=
  { faces := 5
  , edges := 8
  , vertices := 5
  , base_sides := 4 }

/-- The sum of faces, edges, and vertices in a square pyramid is 18 -/
theorem sum_faces_edges_vertices (sp : SquarePyramid) : 
  sp.faces + sp.edges + sp.vertices = 18 := by
  sorry

/-- The number of diagonals in a polygon -/
def num_diagonals (n : Nat) : Nat :=
  n * (n - 3) / 2

/-- The number of diagonals in the square base of a square pyramid is 2 -/
theorem num_diagonals_square_base (sp : SquarePyramid) : 
  num_diagonals sp.base_sides = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_faces_edges_vertices_num_diagonals_square_base_l3170_317040


namespace NUMINAMATH_CALUDE_triangle_angle_sum_bound_l3170_317048

theorem triangle_angle_sum_bound (A B C : Real) (h_triangle : A + B + C = Real.pi)
  (h_sin_sum : Real.sin A + Real.sin B + Real.sin C ≤ 1) :
  min (A + B) (min (B + C) (C + A)) < Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_bound_l3170_317048


namespace NUMINAMATH_CALUDE_log_function_range_l3170_317046

/-- The function f(x) = lg(ax^2 - 2x + 2) has a range of ℝ if and only if a ∈ (0, 1/2] -/
theorem log_function_range (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, y = Real.log (a * x^2 - 2 * x + 2)) ↔ (0 < a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_log_function_range_l3170_317046


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_geq_2_l3170_317041

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 3

-- Define the property of f being decreasing on (-∞, 2)
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → x < 2 → y < 2 → f a x > f a y

-- Theorem statement
theorem f_decreasing_implies_a_geq_2 :
  ∀ a : ℝ, is_decreasing_on_interval a → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_geq_2_l3170_317041


namespace NUMINAMATH_CALUDE_quantities_total_l3170_317064

theorem quantities_total (total_avg : ℝ) (subset1_avg : ℝ) (subset2_avg : ℝ) 
  (h1 : total_avg = 8)
  (h2 : subset1_avg = 4)
  (h3 : subset2_avg = 14)
  (h4 : 3 * subset1_avg + 2 * subset2_avg = 5 * total_avg) : 
  5 = (3 * subset1_avg + 2 * subset2_avg) / total_avg :=
by sorry

end NUMINAMATH_CALUDE_quantities_total_l3170_317064


namespace NUMINAMATH_CALUDE_starting_team_combinations_l3170_317059

/-- The number of members in the water polo team -/
def team_size : ℕ := 18

/-- The number of players in the starting team -/
def starting_team_size : ℕ := 7

/-- The number of interchangeable positions -/
def interchangeable_positions : ℕ := 5

/-- The number of ways to choose the starting team -/
def choose_starting_team : ℕ := team_size * (team_size - 1) * (Nat.choose (team_size - 2) interchangeable_positions)

theorem starting_team_combinations :
  choose_starting_team = 1338176 := by
  sorry

end NUMINAMATH_CALUDE_starting_team_combinations_l3170_317059


namespace NUMINAMATH_CALUDE_electricity_relationship_l3170_317047

/-- Represents the relationship between electricity consumption and fee -/
structure ElectricityRelation where
  consumption : ℝ  -- Electricity consumption in kWh
  fee : ℝ          -- Electricity fee in yuan
  linear : fee = 0.55 * consumption  -- Linear relationship

/-- Proves the functional relationship and calculates consumption for a given fee -/
theorem electricity_relationship (r : ElectricityRelation) :
  r.fee = 0.55 * r.consumption ∧ 
  (r.fee = 40.7 → r.consumption = 74) := by
  sorry

#check electricity_relationship

end NUMINAMATH_CALUDE_electricity_relationship_l3170_317047


namespace NUMINAMATH_CALUDE_triangle_property_l3170_317099

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.c / 2 = t.b - t.a * Real.cos t.C) : 
  (Real.cos t.A = 1 / 2) ∧ 
  (t.a = Real.sqrt 15 → t.b = 4 → t.c^2 - 4*t.c + 1 = 0) := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_triangle_property_l3170_317099


namespace NUMINAMATH_CALUDE_balloon_distribution_l3170_317038

theorem balloon_distribution (red white green chartreuse : ℕ) 
  (h1 : red = 24)
  (h2 : white = 38)
  (h3 : green = 68)
  (h4 : chartreuse = 75)
  (friends : ℕ)
  (h5 : friends = 10) :
  (red + white + green + chartreuse) % friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_balloon_distribution_l3170_317038


namespace NUMINAMATH_CALUDE_cricketer_average_increase_l3170_317002

/-- Represents the increase in average score for a cricketer -/
def average_score_increase (total_innings : ℕ) (runs_last_inning : ℕ) (new_average : ℚ) : ℚ :=
  let old_total := (total_innings - 1) * (new_average * total_innings - runs_last_inning) / (total_innings - 1)
  let old_average := old_total / (total_innings - 1)
  new_average - old_average

/-- Theorem stating the increase in average score for the given cricketer -/
theorem cricketer_average_increase :
  average_score_increase 19 95 23 = 4 := by sorry

end NUMINAMATH_CALUDE_cricketer_average_increase_l3170_317002


namespace NUMINAMATH_CALUDE_identity_function_satisfies_equation_l3170_317067

theorem identity_function_satisfies_equation (f : ℕ → ℕ) :
  (∀ x y : ℕ, f (x + f y) = f x + y) → (∀ x : ℕ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_equation_l3170_317067


namespace NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_of_16777_l3170_317009

def n : ℕ := 16777

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (m : ℕ) : ℕ := sorry

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_digits_greatest_prime_divisor_of_16777 :
  sum_of_digits (greatest_prime_divisor n) = 2 := by sorry

end NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_of_16777_l3170_317009


namespace NUMINAMATH_CALUDE_truck_departure_time_l3170_317037

/-- Proves that given a car traveling at 55 mph and a truck traveling at 65 mph
    on the same road in the same direction, if it takes 6.5 hours for the truck
    to pass the car, then the truck left the station 1 hour after the car. -/
theorem truck_departure_time (car_speed truck_speed : ℝ) (passing_time : ℝ) :
  car_speed = 55 →
  truck_speed = 65 →
  passing_time = 6.5 →
  (truck_speed - car_speed) * passing_time / truck_speed = 1 :=
by sorry

end NUMINAMATH_CALUDE_truck_departure_time_l3170_317037


namespace NUMINAMATH_CALUDE_arithmetic_sequence_l3170_317019

def a (n : ℕ) : ℤ := 3 * n + 1

theorem arithmetic_sequence : ∀ n : ℕ, a (n + 1) - a n = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_l3170_317019


namespace NUMINAMATH_CALUDE_smaller_number_proof_l3170_317036

theorem smaller_number_proof (x y : ℝ) (sum_eq : x + y = 79) (diff_eq : x - y = 15) :
  y = 32 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l3170_317036


namespace NUMINAMATH_CALUDE_square_sum_product_l3170_317072

theorem square_sum_product (a b : ℝ) (ha : a = Real.sqrt 2 + 1) (hb : b = Real.sqrt 2 - 1) :
  a^2 + a*b + b^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l3170_317072


namespace NUMINAMATH_CALUDE_f_extrema_and_roots_l3170_317012

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - (x + 1)^2

theorem f_extrema_and_roots :
  (∃ x₀ ∈ Set.Icc (-1 : ℝ) 2, ∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≥ f x₀ ∧ f x₀ = -(Real.log 2)^2 - 1) ∧
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≤ f x₁ ∧ f x₁ = 2 * Real.exp 2 - 9) ∧
  (∀ a < -1, (∃! x, f x = a * x - 1)) ∧
  (∀ a > -1, (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = a * x₁ - 1 ∧ f x₂ = a * x₂ - 1 ∧ f x₃ = a * x₃ - 1)) :=
by sorry


end NUMINAMATH_CALUDE_f_extrema_and_roots_l3170_317012


namespace NUMINAMATH_CALUDE_sodium_chloride_formation_l3170_317043

-- Define the chemical reaction
structure Reaction where
  hcl : ℕ  -- moles of Hydrochloric acid
  nahco3 : ℕ  -- moles of Sodium bicarbonate
  nacl : ℕ  -- moles of Sodium chloride produced

-- Define the stoichiometric relationship
def stoichiometric_relationship (r : Reaction) : Prop :=
  r.nacl = min r.hcl r.nahco3

-- Theorem statement
theorem sodium_chloride_formation (r : Reaction) 
  (h1 : r.hcl = 2)  -- 2 moles of Hydrochloric acid
  (h2 : r.nahco3 = 2)  -- 2 moles of Sodium bicarbonate
  (h3 : stoichiometric_relationship r)  -- The reaction follows the stoichiometric relationship
  : r.nacl = 2 :=
by sorry

end NUMINAMATH_CALUDE_sodium_chloride_formation_l3170_317043


namespace NUMINAMATH_CALUDE_course_selection_proof_l3170_317017

theorem course_selection_proof :
  let total_courses : ℕ := 6
  let courses_per_student : ℕ := 3
  let common_courses : ℕ := 1
  
  (Nat.choose total_courses common_courses) *
  (Nat.choose (total_courses - common_courses) (courses_per_student - common_courses)) *
  (Nat.choose (total_courses - courses_per_student) (courses_per_student - common_courses)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_proof_l3170_317017


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3170_317031

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3170_317031


namespace NUMINAMATH_CALUDE_cantor_bernstein_decomposition_l3170_317003

universe u

theorem cantor_bernstein_decomposition 
  {E F : Type u} (f : E → F) (g : F → E) 
  (hf : Function.Injective f) (hg : Function.Injective g) :
  ∃ φ : E ≃ F, True :=
sorry

end NUMINAMATH_CALUDE_cantor_bernstein_decomposition_l3170_317003


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3170_317083

/-- The distance between the foci of a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1 is 2√(a^2 + b^2) -/
theorem hyperbola_foci_distance (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let distance := 2 * Real.sqrt (a^2 + b^2)
  distance = 2 * Real.sqrt 34 ↔ a^2 = 25 ∧ b^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3170_317083


namespace NUMINAMATH_CALUDE_area_of_specific_trapezoid_l3170_317071

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- The length of each lateral side -/
  lateral_side : ℝ
  /-- The radius of the inscribed circle -/
  inscribed_radius : ℝ

/-- The area of an isosceles trapezoid with an inscribed circle -/
def area (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  2 * t.lateral_side * t.inscribed_radius

/-- Theorem: The area of an isosceles trapezoid with lateral side length 9 and an inscribed circle of radius 4 is 72 -/
theorem area_of_specific_trapezoid :
  let t : IsoscelesTrapezoidWithInscribedCircle := ⟨9, 4⟩
  area t = 72 := by sorry

end NUMINAMATH_CALUDE_area_of_specific_trapezoid_l3170_317071


namespace NUMINAMATH_CALUDE_billys_candy_count_l3170_317011

/-- The total number of candy pieces given the number of boxes and pieces per box -/
def total_candy (boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  boxes * pieces_per_box

/-- Theorem: Billy's total candy pieces -/
theorem billys_candy_count :
  total_candy 7 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_billys_candy_count_l3170_317011


namespace NUMINAMATH_CALUDE_spatial_relationship_l3170_317068

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (para_planes : Plane → Plane → Prop)
variable (perp_lines : Line → Line → Prop)

-- State the theorem
theorem spatial_relationship 
  (m l : Line) 
  (α β : Plane) 
  (h1 : m ≠ l) 
  (h2 : α ≠ β) 
  (h3 : perp m α) 
  (h4 : para l β) 
  (h5 : para_planes α β) : 
  perp_lines m l :=
sorry

end NUMINAMATH_CALUDE_spatial_relationship_l3170_317068


namespace NUMINAMATH_CALUDE_meal_cost_l3170_317075

theorem meal_cost (adults children : ℕ) (total_bill : ℚ) :
  adults = 2 →
  children = 5 →
  total_bill = 21 →
  (total_bill / (adults + children : ℚ)) = 3 := by
sorry

end NUMINAMATH_CALUDE_meal_cost_l3170_317075


namespace NUMINAMATH_CALUDE_tom_painted_fraction_l3170_317055

def tom_paint_rate : ℚ := 1 / 60
def jerry_dig_time : ℚ := 40 / 60

theorem tom_painted_fraction :
  tom_paint_rate * jerry_dig_time = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_tom_painted_fraction_l3170_317055


namespace NUMINAMATH_CALUDE_cubic_inequality_l3170_317078

theorem cubic_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3170_317078


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3170_317039

theorem perfect_square_condition (n : ℕ+) : 
  (∃ m : ℕ, 4^27 + 4^1000 + 4^(n:ℕ) = m^2) → n ≤ 1972 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3170_317039


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3170_317035

/-- The quadratic function f(x) = x^2 + 14x + 24 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 24

theorem quadratic_minimum :
  (∃ (x : ℝ), f x = -25) ∧ (∀ (y : ℝ), f y ≥ -25) ∧ (f (-7) = -25) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3170_317035


namespace NUMINAMATH_CALUDE_mrs_hilt_pencils_l3170_317058

/-- The number of pencils Mrs. Hilt can buy -/
def pencils_bought (total_money : ℕ) (cost_per_pencil : ℕ) : ℕ :=
  total_money / cost_per_pencil

/-- Proof that Mrs. Hilt can buy 10 pencils -/
theorem mrs_hilt_pencils :
  pencils_bought 50 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pencils_l3170_317058


namespace NUMINAMATH_CALUDE_dorothy_income_l3170_317054

theorem dorothy_income (annual_income : ℝ) : 
  annual_income * (1 - 0.18) = 49200 → annual_income = 60000 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_income_l3170_317054


namespace NUMINAMATH_CALUDE_cheese_cost_is_seven_l3170_317087

/-- The cost of a pound of cheese, given Tony's initial amount, beef cost, and remaining amount after purchase. -/
def cheese_cost (initial_amount beef_cost remaining_amount : ℚ) : ℚ :=
  (initial_amount - beef_cost - remaining_amount) / 3

/-- Theorem stating that the cost of a pound of cheese is $7 under the given conditions. -/
theorem cheese_cost_is_seven :
  let initial_amount : ℚ := 87
  let beef_cost : ℚ := 5
  let remaining_amount : ℚ := 61
  cheese_cost initial_amount beef_cost remaining_amount = 7 := by
  sorry

end NUMINAMATH_CALUDE_cheese_cost_is_seven_l3170_317087


namespace NUMINAMATH_CALUDE_equality_of_cyclic_system_l3170_317020

theorem equality_of_cyclic_system (x y z : ℝ) 
  (eq1 : x^3 = 2*y - 1)
  (eq2 : y^3 = 2*z - 1)
  (eq3 : z^3 = 2*x - 1) :
  x = y ∧ y = z := by
  sorry

end NUMINAMATH_CALUDE_equality_of_cyclic_system_l3170_317020


namespace NUMINAMATH_CALUDE_kaleb_initial_books_l3170_317013

/-- Represents the number of books Kaleb had initially. -/
def initial_books : ℕ := 34

/-- Represents the number of books Kaleb sold. -/
def books_sold : ℕ := 17

/-- Represents the number of new books Kaleb bought. -/
def new_books : ℕ := 7

/-- Represents the number of books Kaleb has now. -/
def current_books : ℕ := 24

/-- Proves that given the conditions, Kaleb must have had 34 books initially. -/
theorem kaleb_initial_books :
  initial_books - books_sold + new_books = current_books :=
by sorry

end NUMINAMATH_CALUDE_kaleb_initial_books_l3170_317013


namespace NUMINAMATH_CALUDE_same_and_different_signs_l3170_317025

theorem same_and_different_signs (a b : ℝ) : 
  (a * b > 0 ↔ (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0)) ∧
  (a * b < 0 ↔ (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_same_and_different_signs_l3170_317025


namespace NUMINAMATH_CALUDE_inscribed_circle_distance_l3170_317095

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  true  -- We don't need to define the specific coordinates for this problem

-- Define the inscribed circle with center O in triangle ABC
def InscribedCircle (O : ℝ × ℝ) (ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  true  -- We don't need to define the specific properties of the inscribed circle

-- Define points M and N where the circle touches sides AB and AC
def TouchPoints (M N : ℝ × ℝ) (ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  true  -- We don't need to define the specific properties of these points

-- Define the inscribed circle with center Q in triangle AMN
def InscribedCircleAMN (Q : ℝ × ℝ) (A M N : ℝ × ℝ) : Prop :=
  true  -- We don't need to define the specific properties of this inscribed circle

-- Define the distances between points
def Distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry  -- We don't need to implement this function for the statement

theorem inscribed_circle_distance
  (A B C O Q M N : ℝ × ℝ)
  (h1 : Triangle A B C)
  (h2 : InscribedCircle O (A, B, C))
  (h3 : TouchPoints M N (A, B, C))
  (h4 : InscribedCircleAMN Q A M N)
  (h5 : Distance A B = 13)
  (h6 : Distance B C = 15)
  (h7 : Distance A C = 14) :
  Distance O Q = 4 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_distance_l3170_317095


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3170_317094

theorem min_sum_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 2*a + 3*b + 4*c = 120) :
  a^2 + b^2 + c^2 ≥ 14400/29 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    2*a₀ + 3*b₀ + 4*c₀ = 120 ∧ a₀^2 + b₀^2 + c₀^2 = 14400/29 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3170_317094


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3170_317001

theorem simplify_and_evaluate (x : ℝ) (h : x ≠ 1) : 
  (2 / (x - 1) + 1) / ((x + 1) / (x^2 - 2*x + 1)) = x - 1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3170_317001


namespace NUMINAMATH_CALUDE_total_distance_theorem_l3170_317069

/-- Calculates the total distance covered by two cyclists in a week -/
def total_distance_in_week (
  onur_speed : ℝ
  ) (hanil_speed : ℝ
  ) (onur_hours : ℝ
  ) (onur_rest_day : ℕ
  ) (hanil_rest_day : ℕ
  ) (hanil_extra_distance : ℝ
  ) (days_in_week : ℕ
  ) : ℝ :=
  let onur_daily_distance := onur_speed * onur_hours
  let hanil_daily_distance := onur_daily_distance + hanil_extra_distance
  let onur_biking_days := days_in_week - (days_in_week / onur_rest_day)
  let hanil_biking_days := days_in_week - (days_in_week / hanil_rest_day)
  let onur_total_distance := onur_daily_distance * onur_biking_days
  let hanil_total_distance := hanil_daily_distance * hanil_biking_days
  onur_total_distance + hanil_total_distance

/-- Theorem stating the total distance covered by Onur and Hanil in a week -/
theorem total_distance_theorem :
  total_distance_in_week 35 45 7 3 4 40 7 = 2935 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_theorem_l3170_317069


namespace NUMINAMATH_CALUDE_hockey_team_ties_l3170_317063

theorem hockey_team_ties (wins ties : ℕ) : 
  wins = ties + 12 →
  2 * wins + ties = 60 →
  ties = 12 := by
sorry

end NUMINAMATH_CALUDE_hockey_team_ties_l3170_317063


namespace NUMINAMATH_CALUDE_cube_root_equation_l3170_317061

theorem cube_root_equation (x : ℝ) : 
  x = 2 / (2 - Real.rpow 3 (1/3)) → 
  x = (2 * (2 + Real.rpow 3 (1/3))) / (4 - Real.rpow 9 (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_l3170_317061


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l3170_317084

theorem least_number_with_remainders : ∃ n : ℕ, 
  n > 0 ∧
  n % 5 = 3 ∧
  n % 6 = 3 ∧
  n % 7 = 3 ∧
  n % 8 = 3 ∧
  n % 9 = 0 ∧
  (∀ m : ℕ, m > 0 ∧ m % 5 = 3 ∧ m % 6 = 3 ∧ m % 7 = 3 ∧ m % 8 = 3 ∧ m % 9 = 0 → n ≤ m) ∧
  n = 1683 :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l3170_317084


namespace NUMINAMATH_CALUDE_min_correct_answers_for_score_l3170_317050

/-- Represents the scoring system and conditions of the math competition --/
structure MathCompetition where
  total_questions : Nat
  attempted_questions : Nat
  correct_points : Nat
  incorrect_deduction : Nat
  unanswered_points : Nat
  min_required_score : Nat

/-- Calculates the score based on the number of correct answers --/
def calculate_score (comp : MathCompetition) (correct_answers : Nat) : Int :=
  let incorrect_answers := comp.attempted_questions - correct_answers
  let unanswered := comp.total_questions - comp.attempted_questions
  (correct_answers * comp.correct_points : Int) -
  (incorrect_answers * comp.incorrect_deduction) +
  (unanswered * comp.unanswered_points)

/-- Theorem stating the minimum number of correct answers needed to achieve the required score --/
theorem min_correct_answers_for_score (comp : MathCompetition)
  (h1 : comp.total_questions = 25)
  (h2 : comp.attempted_questions = 20)
  (h3 : comp.correct_points = 8)
  (h4 : comp.incorrect_deduction = 2)
  (h5 : comp.unanswered_points = 2)
  (h6 : comp.min_required_score = 150) :
  ∃ n : Nat, (∀ m : Nat, calculate_score comp m ≥ comp.min_required_score → m ≥ n) ∧
             calculate_score comp n ≥ comp.min_required_score ∧
             n = 18 := by
  sorry


end NUMINAMATH_CALUDE_min_correct_answers_for_score_l3170_317050


namespace NUMINAMATH_CALUDE_solve_for_a_l3170_317085

theorem solve_for_a : ∃ a : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = -3 → a * x - y = 1) ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3170_317085


namespace NUMINAMATH_CALUDE_specific_pyramid_volume_l3170_317090

/-- Represents a triangular pyramid with vertex P and base ABC -/
structure TriangularPyramid where
  BC : ℝ
  CA : ℝ
  AB : ℝ
  dihedral_angle : ℝ

/-- The volume of a triangular pyramid -/
def volume (p : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem: The volume of the specific triangular pyramid is 2 -/
theorem specific_pyramid_volume :
  let p : TriangularPyramid := {
    BC := 3,
    CA := 4,
    AB := 5,
    dihedral_angle := π / 4  -- 45° in radians
  }
  volume p = 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_volume_l3170_317090


namespace NUMINAMATH_CALUDE_fractional_equation_to_polynomial_l3170_317066

theorem fractional_equation_to_polynomial (x y : ℝ) (h1 : (2*x - 1)/x^2 + x^2/(2*x - 1) = 5) (h2 : (2*x - 1)/x^2 = y) : y^2 - 5*y + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_to_polynomial_l3170_317066


namespace NUMINAMATH_CALUDE_investment_comparison_l3170_317051

/-- Represents the value of an investment over time -/
structure Investment where
  initial : ℝ
  year1_change : ℝ
  year2_change : ℝ

/-- Calculates the final value of an investment after two years -/
def final_value (inv : Investment) : ℝ :=
  inv.initial * (1 + inv.year1_change) * (1 + inv.year2_change)

/-- The problem setup -/
def problem_setup : (Investment × Investment × Investment) :=
  ({ initial := 150, year1_change := 0.1, year2_change := 0.15 },
   { initial := 150, year1_change := -0.3, year2_change := 0.5 },
   { initial := 150, year1_change := 0, year2_change := -0.1 })

theorem investment_comparison :
  let (a, b, c) := problem_setup
  final_value a > final_value b ∧ final_value b > final_value c :=
by sorry

end NUMINAMATH_CALUDE_investment_comparison_l3170_317051


namespace NUMINAMATH_CALUDE_square_root_of_nine_l3170_317018

theorem square_root_of_nine : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l3170_317018


namespace NUMINAMATH_CALUDE_cuboid_reduction_impossibility_l3170_317032

theorem cuboid_reduction_impossibility (a b c a' b' c' : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0)
  (haa' : a ≥ a') (hbb' : b ≥ b') (hcc' : c ≥ c') :
  ¬(a' * b' * c' = (1/2) * a * b * c ∧ 
    2 * (a' * b' + b' * c' + c' * a') = 2 * (a * b + b * c + c * a)) := by
  sorry

end NUMINAMATH_CALUDE_cuboid_reduction_impossibility_l3170_317032


namespace NUMINAMATH_CALUDE_inheritance_calculation_l3170_317008

theorem inheritance_calculation (x : ℝ) : 
  0.25 * x + 0.15 * (0.75 * x - 5000) + 5000 = 16500 → x = 33794 :=
by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l3170_317008


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3170_317091

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 4)
  (h2 : x⁻¹ - y⁻¹ = -5) : 
  x + y = -16/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3170_317091


namespace NUMINAMATH_CALUDE_hulk_jumps_theorem_l3170_317053

def jump_sequence (n : ℕ) : ℝ := 4 * (3 : ℝ) ^ (n - 1)

def total_distance (n : ℕ) : ℝ := 2 * ((3 : ℝ) ^ n - 1)

theorem hulk_jumps_theorem :
  (∀ k < 8, total_distance k ≤ 5000) ∧ total_distance 8 > 5000 := by
  sorry

end NUMINAMATH_CALUDE_hulk_jumps_theorem_l3170_317053


namespace NUMINAMATH_CALUDE_andrew_flooring_theorem_l3170_317081

def andrew_flooring_problem (bedroom living_room kitchen guest_bedroom hallway leftover : ℕ) : Prop :=
  let total_used := bedroom + living_room + kitchen + guest_bedroom + 2 * hallway
  let total_original := total_used + leftover
  let ruined_per_bedroom := total_original - total_used
  (bedroom = 8) ∧
  (living_room = 20) ∧
  (kitchen = 11) ∧
  (guest_bedroom = bedroom - 2) ∧
  (hallway = 4) ∧
  (leftover = 6) ∧
  (ruined_per_bedroom = 6)

theorem andrew_flooring_theorem :
  ∀ bedroom living_room kitchen guest_bedroom hallway leftover,
  andrew_flooring_problem bedroom living_room kitchen guest_bedroom hallway leftover :=
by
  sorry

end NUMINAMATH_CALUDE_andrew_flooring_theorem_l3170_317081


namespace NUMINAMATH_CALUDE_shirt_cost_calculation_l3170_317080

/-- The amount Mary spent on clothing -/
def total_spent : ℝ := 25.31

/-- The amount Mary spent on the jacket -/
def jacket_cost : ℝ := 12.27

/-- The number of shops Mary visited -/
def shops_visited : ℕ := 2

/-- The amount Mary spent on the shirt -/
def shirt_cost : ℝ := total_spent - jacket_cost

theorem shirt_cost_calculation : 
  shirt_cost = total_spent - jacket_cost :=
by sorry

end NUMINAMATH_CALUDE_shirt_cost_calculation_l3170_317080


namespace NUMINAMATH_CALUDE_both_languages_difference_l3170_317079

/-- The total number of students in the school -/
def total_students : ℕ := 2500

/-- The minimum percentage of students studying Italian -/
def min_italian_percent : ℚ := 70 / 100

/-- The maximum percentage of students studying Italian -/
def max_italian_percent : ℚ := 75 / 100

/-- The minimum percentage of students studying German -/
def min_german_percent : ℚ := 35 / 100

/-- The maximum percentage of students studying German -/
def max_german_percent : ℚ := 45 / 100

/-- The number of students studying Italian -/
def italian_students (n : ℕ) : Prop :=
  ⌈(min_italian_percent * total_students : ℚ)⌉ ≤ n ∧ n ≤ ⌊(max_italian_percent * total_students : ℚ)⌋

/-- The number of students studying German -/
def german_students (n : ℕ) : Prop :=
  ⌈(min_german_percent * total_students : ℚ)⌉ ≤ n ∧ n ≤ ⌊(max_german_percent * total_students : ℚ)⌋

/-- The theorem stating the difference between max and min number of students studying both languages -/
theorem both_languages_difference :
  ∃ (max min : ℕ),
    (∀ i g b, italian_students i → german_students g → i + g - b = total_students → b ≤ max) ∧
    (∀ i g b, italian_students i → german_students g → i + g - b = total_students → min ≤ b) ∧
    max - min = 375 := by
  sorry

end NUMINAMATH_CALUDE_both_languages_difference_l3170_317079


namespace NUMINAMATH_CALUDE_geometric_sequence_alternating_l3170_317030

def is_alternating_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) < 0

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_alternating
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_sum1 : a 1 + a 2 = -3/2)
  (h_sum2 : a 4 + a 5 = 12) :
  is_alternating_sequence a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_alternating_l3170_317030


namespace NUMINAMATH_CALUDE_soccer_games_played_l3170_317024

theorem soccer_games_played (wins losses ties total : ℕ) : 
  wins + losses + ties = total →
  4 * ties = wins →
  3 * ties = losses →
  losses = 9 →
  total = 24 := by
sorry

end NUMINAMATH_CALUDE_soccer_games_played_l3170_317024


namespace NUMINAMATH_CALUDE_fold_symmetry_l3170_317034

-- Define the fold line
def fold_line (x y : ℝ) : Prop := y = 2 * x

-- Define the symmetric point
def symmetric_point (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  fold_line ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) ∧
  (x₂ - x₁) = (y₂ - y₁) / 2

-- Define the perpendicular bisector property
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  fold_line ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

-- Theorem statement
theorem fold_symmetry :
  perpendicular_bisector 10 0 (-6) 8 →
  symmetric_point (-4) 2 4 (-2) :=
sorry

end NUMINAMATH_CALUDE_fold_symmetry_l3170_317034


namespace NUMINAMATH_CALUDE_cake_shop_problem_l3170_317098

theorem cake_shop_problem :
  ∃ (N n K : ℕ+), 
    (N - n * K = 6) ∧ 
    (N = (n - 1) * 8 + 1) ∧ 
    (N = 97) := by
  sorry

end NUMINAMATH_CALUDE_cake_shop_problem_l3170_317098


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_n_between_3_and_5_l3170_317004

/-- A simple polygon in 2D space -/
structure SimplePolygon where
  vertices : List (ℝ × ℝ)
  is_simple : Bool  -- Assume this is true for a simple polygon
  is_counterclockwise : Bool -- Assume this is true for counterclockwise orientation

/-- Represents a half-plane in 2D space -/
structure HalfPlane where
  normal : ℝ × ℝ
  offset : ℝ

/-- Function to get the positive half-planes of a simple polygon -/
def getPositiveHalfPlanes (p : SimplePolygon) : List HalfPlane :=
  sorry  -- Implementation details omitted

/-- Function to check if the intersection of half-planes is non-empty -/
def isIntersectionNonEmpty (planes : List HalfPlane) : Bool :=
  sorry  -- Implementation details omitted

/-- The main theorem -/
theorem intersection_nonempty_iff_n_between_3_and_5 (n : ℕ) :
  (∀ p : SimplePolygon, p.vertices.length = n →
    isIntersectionNonEmpty (getPositiveHalfPlanes p)) ↔ (3 ≤ n ∧ n ≤ 5) :=
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_n_between_3_and_5_l3170_317004


namespace NUMINAMATH_CALUDE_triangle_properties_l3170_317060

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a - 2 * t.c * Real.cos t.B = t.c) 
  (h2 : Real.cos t.B = 1/3) 
  (h3 : t.c = 3) 
  (h4 : 0 < t.A ∧ t.A < Real.pi/2) 
  (h5 : 0 < t.B ∧ t.B < Real.pi/2) 
  (h6 : 0 < t.C ∧ t.C < Real.pi/2) :
  t.b = 2 * Real.sqrt 6 ∧ 
  1/2 < Real.sin t.C ∧ Real.sin t.C < Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3170_317060


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3170_317000

theorem inequality_equivalence (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (2 + y) / x < (4 - x) / y ↔
  ((x * y > 0 → (x - 2)^2 + (y + 1)^2 < 5) ∧
   (x * y < 0 → (x - 2)^2 + (y + 1)^2 > 5)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3170_317000


namespace NUMINAMATH_CALUDE_safari_creatures_l3170_317045

/-- Proves that given 150 creatures with 624 legs total, where some are two-legged ostriches
    and others are six-legged aliens, the number of ostriches is 69. -/
theorem safari_creatures (total_creatures : ℕ) (total_legs : ℕ) 
    (h1 : total_creatures = 150)
    (h2 : total_legs = 624) : 
  ∃ (ostriches aliens : ℕ),
    ostriches + aliens = total_creatures ∧
    2 * ostriches + 6 * aliens = total_legs ∧
    ostriches = 69 := by
  sorry

end NUMINAMATH_CALUDE_safari_creatures_l3170_317045


namespace NUMINAMATH_CALUDE_six_tricycles_l3170_317016

/-- Represents the number of children riding each type of vehicle -/
structure VehicleCounts where
  bicycles : ℕ
  tricycles : ℕ
  unicycles : ℕ

/-- The total number of children -/
def total_children : ℕ := 10

/-- The total number of wheels -/
def total_wheels : ℕ := 26

/-- Calculates the total number of children based on vehicle counts -/
def count_children (v : VehicleCounts) : ℕ :=
  v.bicycles + v.tricycles + v.unicycles

/-- Calculates the total number of wheels based on vehicle counts -/
def count_wheels (v : VehicleCounts) : ℕ :=
  2 * v.bicycles + 3 * v.tricycles + v.unicycles

/-- Theorem stating that there are 6 tricycles -/
theorem six_tricycles : 
  ∃ v : VehicleCounts, 
    count_children v = total_children ∧ 
    count_wheels v = total_wheels ∧ 
    v.tricycles = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_tricycles_l3170_317016


namespace NUMINAMATH_CALUDE_max_angles_less_than_108_is_4_l3170_317086

/-- The maximum number of angles less than 108° in a convex polygon -/
def max_angles_less_than_108 (n : ℕ) : ℕ := 4

/-- Theorem stating that the maximum number of angles less than 108° in a convex n-gon is 4 -/
theorem max_angles_less_than_108_is_4 (n : ℕ) (h : n ≥ 3) :
  max_angles_less_than_108 n = 4 := by sorry

end NUMINAMATH_CALUDE_max_angles_less_than_108_is_4_l3170_317086


namespace NUMINAMATH_CALUDE_jacket_cost_l3170_317033

/-- Represents the cost of clothing items and shipments -/
structure ClothingCost where
  sweater : ℝ
  jacket : ℝ

/-- Represents a shipment of clothing items -/
structure Shipment where
  sweaters : ℕ
  jackets : ℕ
  totalCost : ℝ

/-- The problem statement -/
theorem jacket_cost (cost : ClothingCost) (shipment1 shipment2 : Shipment) :
  shipment1.sweaters = 10 →
  shipment1.jackets = 20 →
  shipment1.totalCost = 800 →
  shipment2.sweaters = 5 →
  shipment2.jackets = 15 →
  shipment2.totalCost = 550 →
  shipment1.sweaters * cost.sweater + shipment1.jackets * cost.jacket = shipment1.totalCost →
  shipment2.sweaters * cost.sweater + shipment2.jackets * cost.jacket = shipment2.totalCost →
  cost.jacket = 30 := by
  sorry


end NUMINAMATH_CALUDE_jacket_cost_l3170_317033


namespace NUMINAMATH_CALUDE_percentage_relationship_l3170_317015

theorem percentage_relationship (x y z : ℝ) (h1 : y = 0.7 * z) (h2 : x = 0.84 * z) :
  x = y * 1.2 :=
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l3170_317015


namespace NUMINAMATH_CALUDE_probability_two_slate_rocks_l3170_317076

/-- The probability of selecting two slate rocks from a field with 12 slate rocks, 
    17 pumice rocks, and 8 granite rocks, when choosing 2 rocks at random without replacement. -/
theorem probability_two_slate_rocks (slate : ℕ) (pumice : ℕ) (granite : ℕ) 
  (h_slate : slate = 12) (h_pumice : pumice = 17) (h_granite : granite = 8) :
  let total := slate + pumice + granite
  (slate / total) * ((slate - 1) / (total - 1)) = 132 / 1332 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_slate_rocks_l3170_317076


namespace NUMINAMATH_CALUDE_dean_has_30_insects_l3170_317022

-- Define the number of insects for each person
def angela_insects : ℕ := 75
def jacob_insects : ℕ := 2 * angela_insects
def dean_insects : ℕ := jacob_insects / 5

-- Theorem to prove
theorem dean_has_30_insects : dean_insects = 30 := by
  sorry

end NUMINAMATH_CALUDE_dean_has_30_insects_l3170_317022


namespace NUMINAMATH_CALUDE_solution_range_l3170_317010

theorem solution_range (x : ℝ) : 
  x ≥ 1 → 
  Real.sqrt (x + 2 - 5 * Real.sqrt (x - 1)) + Real.sqrt (x + 5 - 7 * Real.sqrt (x - 1)) = 2 → 
  5 ≤ x ∧ x ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l3170_317010


namespace NUMINAMATH_CALUDE_num_purchasing_plans_eq_600_l3170_317077

/-- The number of different purchasing plans for souvenirs -/
def num_purchasing_plans : ℕ :=
  (Finset.filter (fun (x, y, z) => x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1)
    (Finset.filter (fun (x, y, z) => x + 2*y + 4*z = 101)
      (Finset.product (Finset.range 102)
        (Finset.product (Finset.range 51) (Finset.range 26))))).card

/-- Theorem stating that the number of purchasing plans is 600 -/
theorem num_purchasing_plans_eq_600 : num_purchasing_plans = 600 := by
  sorry

end NUMINAMATH_CALUDE_num_purchasing_plans_eq_600_l3170_317077


namespace NUMINAMATH_CALUDE_lady_eagles_score_l3170_317073

theorem lady_eagles_score (total_points : ℕ) (games : ℕ) (jessie_points : ℕ)
  (h1 : total_points = 311)
  (h2 : games = 5)
  (h3 : jessie_points = 41) :
  total_points - 3 * jessie_points = 188 := by
  sorry

end NUMINAMATH_CALUDE_lady_eagles_score_l3170_317073


namespace NUMINAMATH_CALUDE_frog_prob_theorem_l3170_317062

/-- A triangular pond with 9 regions -/
structure TriangularPond :=
  (regions : Fin 9)

/-- The frog's position in the pond -/
inductive Position
  | A
  | Adjacent

/-- The probability of the frog being in a specific position after k jumps -/
def probability (k : ℕ) (pos : Position) : ℝ :=
  sorry

/-- The probability of the frog being in region A after 2022 jumps -/
def prob_in_A_after_2022 : ℝ :=
  probability 2022 Position.A

/-- The theorem stating the probability of the frog being in region A after 2022 jumps -/
theorem frog_prob_theorem :
  prob_in_A_after_2022 = 2/9 * (1/2)^1010 + 1/9 :=
sorry

end NUMINAMATH_CALUDE_frog_prob_theorem_l3170_317062


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l3170_317093

def f (x : ℝ) := x^2 - x - 1

theorem f_has_two_zeros : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l3170_317093


namespace NUMINAMATH_CALUDE_compound_composition_l3170_317097

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of aluminum atoms in the compound -/
def num_Al : ℕ := 1

/-- The number of fluorine atoms in the compound -/
def num_F : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 84

theorem compound_composition :
  (num_Al : ℝ) * atomic_weight_Al + (num_F : ℝ) * atomic_weight_F = molecular_weight := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l3170_317097


namespace NUMINAMATH_CALUDE_balloons_rearrangements_l3170_317057

def word : String := "BALLOONS"

def is_vowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

def vowels : List Char :=
  word.toList.filter is_vowel

def consonants : List Char :=
  word.toList.filter (fun c => ¬(is_vowel c))

theorem balloons_rearrangements :
  (vowels.length.factorial / (vowels.countP (· = 'O')).factorial) *
  (consonants.length.factorial / (consonants.countP (· = 'L')).factorial) = 180 := by
  sorry

end NUMINAMATH_CALUDE_balloons_rearrangements_l3170_317057


namespace NUMINAMATH_CALUDE_chest_contents_l3170_317052

-- Define the types of coins
inductive CoinType
| Gold
| Silver
| Copper

-- Define the chests
structure Chest where
  inscription : CoinType → Prop
  content : CoinType

-- Define the problem setup
def chestProblem (c1 c2 c3 : Chest) : Prop :=
  -- All inscriptions are incorrect
  (¬c1.inscription c1.content) ∧
  (¬c2.inscription c2.content) ∧
  (¬c3.inscription c3.content) ∧
  -- Each chest contains a different type of coin
  (c1.content ≠ c2.content) ∧
  (c2.content ≠ c3.content) ∧
  (c3.content ≠ c1.content) ∧
  -- Inscriptions on the chests
  (c1.inscription = fun c => c = CoinType.Gold) ∧
  (c2.inscription = fun c => c = CoinType.Silver) ∧
  (c3.inscription = fun c => c = CoinType.Gold ∨ c = CoinType.Silver)

-- The theorem to prove
theorem chest_contents (c1 c2 c3 : Chest) 
  (h : chestProblem c1 c2 c3) : 
  c1.content = CoinType.Silver ∧ 
  c2.content = CoinType.Gold ∧ 
  c3.content = CoinType.Copper := by
  sorry

end NUMINAMATH_CALUDE_chest_contents_l3170_317052


namespace NUMINAMATH_CALUDE_jellybean_box_scaling_l3170_317007

theorem jellybean_box_scaling (bert_jellybeans : ℕ) (scale_factor : ℕ) : 
  bert_jellybeans = 150 → scale_factor = 3 →
  (scale_factor ^ 3 : ℕ) * bert_jellybeans = 4050 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_box_scaling_l3170_317007


namespace NUMINAMATH_CALUDE_negation_of_exists_negation_of_proposition_l3170_317042

theorem negation_of_exists (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ ∀ n, ¬ p n :=
by sorry

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_negation_of_proposition_l3170_317042


namespace NUMINAMATH_CALUDE_count_cows_l3170_317028

def group_of_animals (ducks cows : ℕ) : Prop :=
  2 * ducks + 4 * cows = 22 + 2 * (ducks + cows)

theorem count_cows : ∃ ducks : ℕ, group_of_animals ducks 11 :=
sorry

end NUMINAMATH_CALUDE_count_cows_l3170_317028


namespace NUMINAMATH_CALUDE_smallest_N_proof_l3170_317021

/-- The smallest number of pies per batch that satisfies the conditions --/
def smallest_N : ℕ := 80

/-- The number of batches of pies --/
def num_batches : ℕ := 21

/-- The number of pies per tray --/
def pies_per_tray : ℕ := 70

theorem smallest_N_proof :
  (∀ N : ℕ, N > 70 → (num_batches * N) % pies_per_tray = 0 → N ≥ smallest_N) ∧
  smallest_N > 70 ∧
  (num_batches * smallest_N) % pies_per_tray = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_N_proof_l3170_317021


namespace NUMINAMATH_CALUDE_triangle_radii_inequality_l3170_317089

theorem triangle_radii_inequality (a b c r ra rb rc : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pos_r : r > 0) (h_pos_ra : ra > 0) (h_pos_rb : rb > 0) (h_pos_rc : rc > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_inradius : r = (a * b * c) / (4 * ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)).sqrt))
  (h_exradius_a : ra = ((a + b + c) * (b + c - a)) / (4 * ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)).sqrt))
  (h_exradius_b : rb = ((a + b + c) * (c + a - b)) / (4 * ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)).sqrt))
  (h_exradius_c : rc = ((a + b + c) * (a + b - c)) / (4 * ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)).sqrt)) :
  (a + b + c) / (a^2 + b^2 + c^2).sqrt ≤ 2 * (ra^2 + rb^2 + rc^2).sqrt / (ra + rb + rc - 3 * r) :=
by sorry

end NUMINAMATH_CALUDE_triangle_radii_inequality_l3170_317089


namespace NUMINAMATH_CALUDE_ice_cream_volume_l3170_317005

/-- The volume of ice cream in a cone with hemisphere and cylindrical layer -/
theorem ice_cream_volume (h_cone : ℝ) (r : ℝ) (h_cylinder : ℝ) : 
  h_cone = 10 ∧ r = 3 ∧ h_cylinder = 2 →
  (1/3 * π * r^2 * h_cone) + (2/3 * π * r^3) + (π * r^2 * h_cylinder) = 66 * π := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l3170_317005


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_one_l3170_317070

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_negative_one :
  log10 (5 / 2) + 2 * log10 2 - (1 / 2)⁻¹ = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_one_l3170_317070


namespace NUMINAMATH_CALUDE_cab_driver_income_l3170_317014

/-- Proves that given the incomes for days 1, 3, 4, and 5, and the average income for all 5 days, the income for day 2 must be $50. -/
theorem cab_driver_income
  (income_day1 : ℕ)
  (income_day3 : ℕ)
  (income_day4 : ℕ)
  (income_day5 : ℕ)
  (average_income : ℕ)
  (h1 : income_day1 = 45)
  (h3 : income_day3 = 60)
  (h4 : income_day4 = 65)
  (h5 : income_day5 = 70)
  (h_avg : average_income = 58)
  : ∃ (income_day2 : ℕ), income_day2 = 50 ∧ 
    (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / 5 = average_income :=
by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l3170_317014


namespace NUMINAMATH_CALUDE_solve_equation_l3170_317092

theorem solve_equation : ∃ y : ℚ, 2*y + 3*y = 500 - (4*y + 6*y) → y = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3170_317092


namespace NUMINAMATH_CALUDE_min_p_plus_q_l3170_317096

def is_repeating_decimal (p q : ℕ+) : Prop :=
  (p : ℚ) / q = 0.198

theorem min_p_plus_q (p q : ℕ+) (h : is_repeating_decimal p q) 
  (h_min : ∀ (p' q' : ℕ+), is_repeating_decimal p' q' → q ≤ q') : 
  p + q = 121 := by
  sorry

end NUMINAMATH_CALUDE_min_p_plus_q_l3170_317096


namespace NUMINAMATH_CALUDE_problem_statement_l3170_317049

theorem problem_statement : 3^(1 + Real.log 2 / Real.log 3) + Real.log 5 + (Real.log 2 / Real.log 3) * (Real.log 3 / Real.log 2) * Real.log 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3170_317049


namespace NUMINAMATH_CALUDE_sequence_problem_l3170_317082

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define a geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom : is_geometric_sequence b)
  (h_eq : 2 * a 5 - (a 8)^2 + 2 * a 11 = 0)
  (h_b8 : b 8 = a 8) :
  b 7 * b 9 = 4 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l3170_317082


namespace NUMINAMATH_CALUDE_karen_homework_paragraphs_l3170_317026

/-- Represents the homework assignment structure -/
structure HomeworkAssignment where
  shortAnswerTime : ℕ
  paragraphTime : ℕ
  essayTime : ℕ
  essayCount : ℕ
  shortAnswerCount : ℕ
  totalTime : ℕ

/-- Calculates the number of paragraphs in the homework assignment -/
def calculateParagraphs (hw : HomeworkAssignment) : ℕ :=
  (hw.totalTime - hw.essayCount * hw.essayTime - hw.shortAnswerCount * hw.shortAnswerTime) / hw.paragraphTime

/-- Theorem stating that Karen's homework assignment results in 5 paragraphs -/
theorem karen_homework_paragraphs :
  let hw : HomeworkAssignment := {
    shortAnswerTime := 3,
    paragraphTime := 15,
    essayTime := 60,
    essayCount := 2,
    shortAnswerCount := 15,
    totalTime := 240
  }
  calculateParagraphs hw = 5 := by sorry

end NUMINAMATH_CALUDE_karen_homework_paragraphs_l3170_317026


namespace NUMINAMATH_CALUDE_sin_theta_value_l3170_317006

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 8 * (Real.cos (x / 2))^2

theorem sin_theta_value (θ : ℝ) :
  (∀ x, f x ≤ f θ) → Real.sin θ = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_sin_theta_value_l3170_317006


namespace NUMINAMATH_CALUDE_smallest_cut_prevents_triangle_smallest_cut_is_minimal_l3170_317056

/-- The smallest positive integer that, when subtracted from the original lengths,
    prevents the formation of a triangle. -/
def smallest_cut : ℕ := 2

/-- Original lengths of the sticks -/
def original_lengths : List ℕ := [9, 12, 20]

/-- Check if three lengths can form a triangle -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The remaining lengths after cutting -/
def remaining_lengths (x : ℕ) : List ℕ :=
  original_lengths.map (λ l => l - x)

theorem smallest_cut_prevents_triangle :
  ∀ x : ℕ, x < smallest_cut →
    ∃ a b c, a::b::c::[] = remaining_lengths x ∧ can_form_triangle a b c :=
by sorry

theorem smallest_cut_is_minimal :
  ¬∃ a b c, a::b::c::[] = remaining_lengths smallest_cut ∧ can_form_triangle a b c :=
by sorry

end NUMINAMATH_CALUDE_smallest_cut_prevents_triangle_smallest_cut_is_minimal_l3170_317056
