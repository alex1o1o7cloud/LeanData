import Mathlib

namespace isosceles_triangle_condition_l3610_361031

/-- 
If in a triangle ABC, where a, b, c are the lengths of sides opposite to angles A, B, C respectively, 
and a * cos(B) = b * cos(A), then the triangle ABC is isosceles.
-/
theorem isosceles_triangle_condition (A B C a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : a * Real.cos B = b * Real.cos A) :
  a = b ∨ b = c ∨ a = c := by
  sorry

end isosceles_triangle_condition_l3610_361031


namespace same_hair_count_l3610_361026

theorem same_hair_count (population : ℕ) (hair_count : Fin population → ℕ) 
  (h1 : population > 500001) 
  (h2 : ∀ p, hair_count p ≤ 500000) : 
  ∃ p1 p2, p1 ≠ p2 ∧ hair_count p1 = hair_count p2 := by
  sorry

end same_hair_count_l3610_361026


namespace billy_is_48_l3610_361025

-- Define Billy's age and Joe's age
def billy_age : ℕ := sorry
def joe_age : ℕ := sorry

-- State the conditions
axiom age_relation : billy_age = 3 * joe_age
axiom age_sum : billy_age + joe_age = 64

-- Theorem to prove
theorem billy_is_48 : billy_age = 48 := by
  sorry

end billy_is_48_l3610_361025


namespace min_chips_to_capture_all_l3610_361080

/-- Represents a rhombus-shaped game board -/
structure GameBoard :=
  (angle : ℝ)
  (side_divisions : ℕ)

/-- Represents a chip on the game board -/
structure Chip :=
  (position : ℕ × ℕ)

/-- Calculates the number of cells captured by a single chip -/
def cells_captured_by_chip (board : GameBoard) (chip : Chip) : ℕ :=
  sorry

/-- Calculates the total number of cells on the game board -/
def total_cells (board : GameBoard) : ℕ :=
  sorry

/-- Checks if a set of chips captures all cells on the board -/
def captures_all_cells (board : GameBoard) (chips : Finset Chip) : Prop :=
  sorry

/-- The main theorem stating the minimum number of chips required -/
theorem min_chips_to_capture_all (board : GameBoard) :
  board.angle = 60 ∧ board.side_divisions = 9 →
  ∃ (chips : Finset Chip), chips.card = 6 ∧ captures_all_cells board chips ∧
  ∀ (other_chips : Finset Chip), captures_all_cells board other_chips → other_chips.card ≥ 6 :=
sorry

end min_chips_to_capture_all_l3610_361080


namespace hyperbola_condition_l3610_361013

/-- Represents the equation of a conic section --/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 + m) + y^2 / (m + 1) = 1 ∧ 
  ((2 + m > 0 ∧ m + 1 < 0) ∨ (2 + m < 0 ∧ m + 1 > 0))

/-- The main theorem stating the condition for the equation to represent a hyperbola --/
theorem hyperbola_condition (m : ℝ) : 
  is_hyperbola m ↔ -2 < m ∧ m < -1 :=
sorry

end hyperbola_condition_l3610_361013


namespace muslim_boys_percentage_l3610_361066

/-- The percentage of Muslim boys in a school -/
def percentage_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) : ℚ :=
  let non_muslim_boys := (hindu_percentage + sikh_percentage) * total_boys + other_boys
  let muslim_boys := total_boys - non_muslim_boys
  (muslim_boys / total_boys) * 100

/-- Theorem stating that the percentage of Muslim boys is approximately 44% -/
theorem muslim_boys_percentage :
  let total_boys : ℕ := 850
  let hindu_percentage : ℚ := 28 / 100
  let sikh_percentage : ℚ := 10 / 100
  let other_boys : ℕ := 153
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
    |percentage_muslim_boys total_boys hindu_percentage sikh_percentage other_boys - 44| < ε :=
sorry

end muslim_boys_percentage_l3610_361066


namespace intersection_M_N_l3610_361091

def M : Set ℝ := { x | -1 ≤ x ∧ x < 3 }

def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} := by
  sorry

end intersection_M_N_l3610_361091


namespace inequality_proof_l3610_361010

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let K := a^4*(b^2*c + b*c^2) + a^3*(b^3*c + b*c^3) + a^2*(b^3*c^2 + b^2*c^3 + b^2*c + b*c^2) + a*(b^3*c + b*c^3) + (b^3*c^2 + b^2*c^3)
  K ≥ 12*a^2*b^2*c^2 ∧ (K = 12*a^2*b^2*c^2 ↔ a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

end inequality_proof_l3610_361010


namespace sin_two_alpha_value_l3610_361012

theorem sin_two_alpha_value (α : ℝ) (h : Real.sin α - Real.cos α = 4/3) : 
  Real.sin (2 * α) = -7/9 := by
  sorry

end sin_two_alpha_value_l3610_361012


namespace part_one_part_two_part_three_l3610_361019

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  x + y = 2 * a - b - 4 ∧ x - y = b - 4

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

-- Part 1
theorem part_one (a b : ℝ) (P : Point) :
  a = 1 → b = 2 → system a b P.x P.y → P.x = -3 ∧ P.y = -1 := by sorry

-- Part 2
theorem part_two (a b : ℝ) (P : Point) :
  system a b P.x P.y →
  P.x < 0 ∧ P.y > 0 →
  (∃ (n : ℕ), n = 4 ∧ ∀ (m : ℤ), (∃ (a' : ℝ), a' = a ∧ system a' b P.x P.y) → m ≤ n) →
  -1 ≤ b ∧ b < 0 := by sorry

-- Part 3
theorem part_three (a b t : ℝ) (P : Point) :
  system a b P.x P.y →
  (∃! (z : ℝ), z = 2 ∧ P.y * z + P.x + 4 = 0) →
  (a * t > b ↔ t > 3/2 ∨ t < 3/2) := by sorry

end part_one_part_two_part_three_l3610_361019


namespace reciprocal_inequality_l3610_361047

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end reciprocal_inequality_l3610_361047


namespace min_value_x_l3610_361038

theorem min_value_x (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log x / Real.log 3 ≥ Real.log 5 / Real.log 3 + (1/3) * (Real.log y / Real.log 3)) :
  x ≥ 5 * Real.sqrt 5 := by
  sorry

end min_value_x_l3610_361038


namespace intersection_when_m_is_one_union_equal_B_iff_m_in_range_l3610_361068

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

-- Theorem 1
theorem intersection_when_m_is_one :
  A 1 ∩ B = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2
theorem union_equal_B_iff_m_in_range (m : ℝ) :
  A m ∪ B = B ↔ m ≥ 3 ∨ m ≤ -3 := by sorry

end intersection_when_m_is_one_union_equal_B_iff_m_in_range_l3610_361068


namespace clusters_per_box_l3610_361075

/-- Given the following conditions:
    1. There are 4 clusters of oats in each spoonful.
    2. There are 25 spoonfuls of cereal in each bowl.
    3. There are 5 bowlfuls of cereal in each box.
    Prove that the number of clusters of oats in each box is equal to 500. -/
theorem clusters_per_box (clusters_per_spoon : ℕ) (spoons_per_bowl : ℕ) (bowls_per_box : ℕ)
  (h1 : clusters_per_spoon = 4)
  (h2 : spoons_per_bowl = 25)
  (h3 : bowls_per_box = 5) :
  clusters_per_spoon * spoons_per_bowl * bowls_per_box = 500 := by
  sorry

end clusters_per_box_l3610_361075


namespace carnation_percentage_l3610_361058

/-- Represents a flower bouquet with various types of flowers -/
structure Bouquet where
  total : ℕ
  pink_roses : ℕ
  red_roses : ℕ
  pink_carnations : ℕ
  red_carnations : ℕ
  yellow_tulips : ℕ

/-- Conditions for the flower bouquet -/
def validBouquet (b : Bouquet) : Prop :=
  b.pink_roses + b.red_roses + b.pink_carnations + b.red_carnations + b.yellow_tulips = b.total ∧
  b.pink_roses + b.pink_carnations = b.total / 2 ∧
  b.pink_roses = (b.pink_roses + b.pink_carnations) * 2 / 5 ∧
  b.red_carnations = (b.red_roses + b.red_carnations) * 6 / 7 ∧
  b.yellow_tulips = b.total / 5

/-- Theorem stating that for a valid bouquet, 55% of the flowers are carnations -/
theorem carnation_percentage (b : Bouquet) (h : validBouquet b) :
  (b.pink_carnations + b.red_carnations) * 100 / b.total = 55 := by
  sorry

end carnation_percentage_l3610_361058


namespace cost_of_graveling_roads_l3610_361055

/-- The cost of graveling two intersecting roads on a rectangular lawn. -/
theorem cost_of_graveling_roads
  (lawn_length lawn_width road_width : ℕ)
  (cost_per_sq_m : ℚ)
  (h1 : lawn_length = 80)
  (h2 : lawn_width = 60)
  (h3 : road_width = 10)
  (h4 : cost_per_sq_m = 2) :
  (lawn_length * road_width + lawn_width * road_width - road_width * road_width) * cost_per_sq_m = 2600 :=
by sorry

end cost_of_graveling_roads_l3610_361055


namespace seating_theorem_l3610_361090

/-- The number of different seating arrangements for n people in m seats,
    where exactly two empty seats are adjacent -/
def seating_arrangements (m n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that for 6 seats and 3 people,
    the number of seating arrangements with exactly two adjacent empty seats is 72 -/
theorem seating_theorem : seating_arrangements 6 3 = 72 :=
  sorry

end seating_theorem_l3610_361090


namespace range_of_a_l3610_361096

-- Define the propositions p and q
def p (x : ℝ) : Prop := 2 * x^2 - 5 * x + 3 < 0
def q (x a : ℝ) : Prop := (x - (2 * a + 1)) * (x - 2 * a) ≤ 0

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, q x a → p x) ∧ (∃ x, p x ∧ ¬q x a)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, necessary_not_sufficient a ↔ (1/4 ≤ a ∧ a ≤ 1/2) :=
sorry

end range_of_a_l3610_361096


namespace correlation_function_is_even_l3610_361082

/-- Represents a stationary random process -/
class StationaryRandomProcess (X : ℝ → ℝ) : Prop where
  is_stationary : ∀ t₁ t₂ τ : ℝ, X (t₁ + τ) = X (t₂ + τ)

/-- Correlation function for a stationary random process -/
def correlationFunction (X : ℝ → ℝ) [StationaryRandomProcess X] (τ : ℝ) : ℝ :=
  sorry -- Definition of correlation function

/-- Theorem: The correlation function of a stationary random process is an even function -/
theorem correlation_function_is_even
  (X : ℝ → ℝ) [StationaryRandomProcess X] :
  ∀ τ : ℝ, correlationFunction X τ = correlationFunction X (-τ) := by
  sorry


end correlation_function_is_even_l3610_361082


namespace product_mod_25_l3610_361086

theorem product_mod_25 : ∃ n : ℕ, 0 ≤ n ∧ n < 25 ∧ (123 * 456 * 789) % 25 = n ∧ n = 2 := by
  sorry

end product_mod_25_l3610_361086


namespace circle_area_outside_square_is_zero_l3610_361044

/-- A square with an inscribed circle -/
structure SquareWithCircle where
  /-- Side length of the square -/
  side_length : ℝ
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- The circle is inscribed in the square -/
  circle_inscribed : radius = side_length / 2

/-- The area of the portion of the circle outside the square is zero -/
theorem circle_area_outside_square_is_zero (s : SquareWithCircle) (h : s.side_length = 10) :
  Real.pi * s.radius ^ 2 - s.side_length ^ 2 = 0 := by
  sorry


end circle_area_outside_square_is_zero_l3610_361044


namespace intersection_equals_open_interval_l3610_361023

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B : Set ℝ := {x | 3 - x > 0}

-- Define the open interval (1, 2)
def open_interval : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = open_interval := by
  sorry

end intersection_equals_open_interval_l3610_361023


namespace x_squared_minus_y_squared_l3610_361062

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 7/12) 
  (h2 : x - y = 1/12) : 
  x^2 - y^2 = 7/144 := by
sorry

end x_squared_minus_y_squared_l3610_361062


namespace intersection_point_is_solution_l3610_361003

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (45/23, -64/23)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 8 * x - 3 * y = 24

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 10 * x + 2 * y = 14

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y :=
by sorry

end intersection_point_is_solution_l3610_361003


namespace carrots_grown_total_l3610_361070

/-- The number of carrots grown by Sally -/
def sally_carrots : ℕ := 6

/-- The number of carrots grown by Fred -/
def fred_carrots : ℕ := 4

/-- The total number of carrots grown by Sally and Fred -/
def total_carrots : ℕ := sally_carrots + fred_carrots

theorem carrots_grown_total :
  total_carrots = 10 := by sorry

end carrots_grown_total_l3610_361070


namespace sin_product_10_30_50_70_l3610_361097

theorem sin_product_10_30_50_70 : 
  Real.sin (10 * π / 180) * Real.sin (30 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) = 1 / 16 :=
by sorry

end sin_product_10_30_50_70_l3610_361097


namespace nuts_in_boxes_l3610_361014

theorem nuts_in_boxes (x y z : ℕ) 
  (h1 : x + 6 = y + z) 
  (h2 : y + 10 = x + z) : 
  z = 8 := by
sorry

end nuts_in_boxes_l3610_361014


namespace trip_duration_is_eight_hours_l3610_361063

/-- Represents a car trip with varying speeds -/
structure CarTrip where
  initial_hours : ℝ
  initial_speed : ℝ
  additional_speed : ℝ
  average_speed : ℝ

/-- Calculates the total duration of the car trip -/
def trip_duration (trip : CarTrip) : ℝ :=
  sorry

/-- Theorem stating that the trip duration is 8 hours given the specific conditions -/
theorem trip_duration_is_eight_hours (trip : CarTrip) 
  (h1 : trip.initial_hours = 4)
  (h2 : trip.initial_speed = 50)
  (h3 : trip.additional_speed = 80)
  (h4 : trip.average_speed = 65) :
  trip_duration trip = 8 := by
  sorry

end trip_duration_is_eight_hours_l3610_361063


namespace smallest_number_l3610_361069

def yoongi_number : ℕ := 4
def jungkook_number : ℕ := 6 + 3
def yuna_number : ℕ := 5

theorem smallest_number : 
  yoongi_number ≤ jungkook_number ∧ yoongi_number ≤ yuna_number :=
by sorry

end smallest_number_l3610_361069


namespace quadratic_root_existence_l3610_361089

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_root_existence 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hf1 : f a b c 1.4 = -0.24) 
  (hf2 : f a b c 1.5 = 0.25) :
  ∃ x₁ : ℝ, f a b c x₁ = 0 ∧ 1.4 < x₁ ∧ x₁ < 1.5 :=
sorry

end quadratic_root_existence_l3610_361089


namespace square_triangle_equal_area_l3610_361049

/-- Given a square with perimeter 80 and a right triangle with height 72,
    if their areas are equal, then the base of the triangle is 100/9 -/
theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) 
  (triangle_base : ℝ) : 
  square_perimeter = 80 →
  triangle_height = 72 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * triangle_base →
  triangle_base = 100 / 9 := by
  sorry

end square_triangle_equal_area_l3610_361049


namespace arthur_walk_distance_l3610_361094

theorem arthur_walk_distance :
  let blocks_east : ℕ := 8
  let blocks_north : ℕ := 15
  let miles_per_block : ℝ := 0.25
  let miles_east : ℝ := blocks_east * miles_per_block
  let miles_north : ℝ := blocks_north * miles_per_block
  let diagonal_miles : ℝ := Real.sqrt (miles_east^2 + miles_north^2)
  let total_miles : ℝ := miles_east + miles_north + diagonal_miles
  total_miles = 10 := by
sorry

end arthur_walk_distance_l3610_361094


namespace max_xy_value_l3610_361015

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : 
  x * y ≤ 168 :=
sorry

end max_xy_value_l3610_361015


namespace division_remainder_3500_74_l3610_361039

theorem division_remainder_3500_74 : ∃ q : ℕ, 3500 = 74 * q + 22 ∧ 22 < 74 := by
  sorry

end division_remainder_3500_74_l3610_361039


namespace investment_problem_l3610_361018

theorem investment_problem (total : ℝ) (rate_greater rate_smaller : ℝ) (income_diff : ℝ) :
  total = 10000 ∧ 
  rate_greater = 0.06 ∧ 
  rate_smaller = 0.05 ∧ 
  income_diff = 160 →
  ∃ (greater smaller : ℝ),
    greater + smaller = total ∧
    rate_greater * greater = rate_smaller * smaller + income_diff ∧
    smaller = 4000 :=
by sorry

end investment_problem_l3610_361018


namespace square_root_problem_l3610_361008

theorem square_root_problem (m : ℝ) (h1 : m > 0) (h2 : ∃ a : ℝ, (3 - a)^2 = m ∧ (2*a + 1)^2 = m) : m = 49 := by
  sorry

end square_root_problem_l3610_361008


namespace snow_probability_l3610_361048

theorem snow_probability (p : ℚ) (n : ℕ) (hp : p = 3/4) (hn : n = 5) :
  1 - (1 - p)^n = 1023/1024 := by
  sorry

end snow_probability_l3610_361048


namespace committee_selection_l3610_361017

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 5) :
  Nat.choose n k = 792 := by
  sorry

end committee_selection_l3610_361017


namespace polygon_sides_l3610_361002

/-- A convex polygon with n sides where the sum of all angles except one is 2970 degrees -/
structure ConvexPolygon where
  n : ℕ
  sum_except_one : ℝ
  convex : sum_except_one = 2970

theorem polygon_sides (p : ConvexPolygon) : p.n = 19 := by
  sorry

end polygon_sides_l3610_361002


namespace divisibility_by_fifteen_l3610_361060

theorem divisibility_by_fifteen (n : ℕ) : n < 10 →
  (∃ k : ℕ, 80000 + 10000 * n + 945 = 15 * k) ↔ n % 3 = 1 := by
  sorry

end divisibility_by_fifteen_l3610_361060


namespace min_sum_p_q_l3610_361059

theorem min_sum_p_q (p q : ℕ) (hp : p > 1) (hq : q > 1) (h_eq : 17 * (p + 1) = 20 * (q + 1)) :
  ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = 20 * (q' + 1) → p + q ≤ p' + q' ∧ p + q = 37 := by
sorry

end min_sum_p_q_l3610_361059


namespace no_tip_customers_l3610_361071

theorem no_tip_customers (total_customers : ℕ) (tip_amount : ℕ) (total_tips : ℕ) : 
  total_customers = 9 →
  tip_amount = 8 →
  total_tips = 32 →
  total_customers - (total_tips / tip_amount) = 5 :=
by sorry

end no_tip_customers_l3610_361071


namespace smallest_with_15_divisors_l3610_361067

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a given natural number has exactly 15 positive divisors -/
def has_15_divisors (n : ℕ+) : Prop := num_divisors n = 15

theorem smallest_with_15_divisors :
  (∀ m : ℕ+, m < 24 → ¬(has_15_divisors m)) ∧ has_15_divisors 24 := by sorry

end smallest_with_15_divisors_l3610_361067


namespace problem_statement_l3610_361009

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.exp x = 0.1

def l₁ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ x - a * y = 0

def l₂ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ 2 * x + a * y - 1 = 0

def q : Prop := ∀ a : ℝ, (∀ x y : ℝ, l₁ a x y ∧ l₂ a x y → a = Real.sqrt 2)

theorem problem_statement : p ∧ ¬q := by
  sorry

end problem_statement_l3610_361009


namespace sphere_volume_from_surface_area_l3610_361077

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ),
  (4 * π * r^2 : ℝ) = 256 * π →
  (4 / 3 * π * r^3 : ℝ) = 2048 / 3 * π := by
  sorry

end sphere_volume_from_surface_area_l3610_361077


namespace total_cost_calculation_total_cost_is_832_l3610_361051

/-- Calculate the total cost of sandwiches and sodas with discount and tax -/
theorem total_cost_calculation (sandwich_price soda_price : ℚ) 
  (sandwich_quantity soda_quantity : ℕ) 
  (sandwich_discount tax_rate : ℚ) : ℚ :=
  let sandwich_cost := sandwich_price * sandwich_quantity
  let soda_cost := soda_price * soda_quantity
  let discounted_sandwich_cost := sandwich_cost * (1 - sandwich_discount)
  let subtotal := discounted_sandwich_cost + soda_cost
  let total_with_tax := subtotal * (1 + tax_rate)
  total_with_tax

/-- Prove that the total cost is $8.32 given the specific conditions -/
theorem total_cost_is_832 :
  total_cost_calculation 2.44 0.87 2 4 0.15 0.09 = 8.32 := by
  sorry

end total_cost_calculation_total_cost_is_832_l3610_361051


namespace smallest_alpha_beta_inequality_optimal_alpha_beta_optimal_beta_value_l3610_361001

theorem smallest_alpha_beta_inequality (α : ℝ) (β : ℝ) :
  (α > 0 ∧ β > 0 ∧
   ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^α / β) →
  α ≥ 2 :=
by sorry

theorem optimal_alpha_beta :
  ∃ β : ℝ, β > 0 ∧
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 →
    Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β :=
by sorry

theorem optimal_beta_value (β : ℝ) :
  (β > 0 ∧
   ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 →
     Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β) →
  β ≥ 4 :=
by sorry

end smallest_alpha_beta_inequality_optimal_alpha_beta_optimal_beta_value_l3610_361001


namespace quadratic_function_negative_on_interval_l3610_361095

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_negative_on_interval
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : a + b + c = 0) :
  ∀ x ∈ Set.Ioo 0 1, f a b c x < 0 :=
by sorry

end quadratic_function_negative_on_interval_l3610_361095


namespace parabola_vertex_l3610_361033

/-- The parabola equation -/
def parabola_equation (x : ℝ) : ℝ := -(x - 5)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (5, 3)

/-- Theorem: The vertex of the parabola y = -(x-5)^2 + 3 is (5, 3) -/
theorem parabola_vertex :
  ∀ (x : ℝ), parabola_equation x ≤ parabola_equation (vertex.1) ∧
  parabola_equation (vertex.1) = vertex.2 :=
sorry

end parabola_vertex_l3610_361033


namespace horner_method_v4_l3610_361073

def f (x : ℝ) : ℝ := 3*x^5 + 5*x^4 + 6*x^3 - 8*x^2 + 35*x + 12

def horner_v4 (a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₅ * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀

theorem horner_method_v4 :
  horner_v4 3 5 6 (-8) 35 12 (-2) = 83 :=
sorry

end horner_method_v4_l3610_361073


namespace inequality_solution_l3610_361024

theorem inequality_solution (x : ℝ) : 
  (10 * x^3 + 20 * x^2 - 75 * x - 105) / ((3 * x - 4) * (x + 5)) < 5 ↔ 
  (x > -5 ∧ x < -1) ∨ (x > 4/3) :=
by sorry

end inequality_solution_l3610_361024


namespace tangent_line_slope_l3610_361092

/-- Given a curve y = x^3 and its tangent line y = kx + 2, prove that k = 3 -/
theorem tangent_line_slope (x : ℝ) :
  let f : ℝ → ℝ := fun x => x^3
  let f' : ℝ → ℝ := fun x => 3 * x^2
  let tangent_line (k : ℝ) (x : ℝ) := k * x + 2
  ∃ m : ℝ, f m = tangent_line k m ∧ f' m = k → k = 3 := by
  sorry

end tangent_line_slope_l3610_361092


namespace yoga_studio_average_weight_l3610_361074

theorem yoga_studio_average_weight
  (num_men : ℕ)
  (num_women : ℕ)
  (avg_weight_men : ℝ)
  (avg_weight_women : ℝ)
  (h1 : num_men = 8)
  (h2 : num_women = 6)
  (h3 : avg_weight_men = 190)
  (h4 : avg_weight_women = 120)
  : (num_men * avg_weight_men + num_women * avg_weight_women) / (num_men + num_women) = 160 :=
by
  sorry

end yoga_studio_average_weight_l3610_361074


namespace square_field_with_pond_area_l3610_361084

/-- The area of a square field with a circular pond in its center -/
theorem square_field_with_pond_area (side : Real) (radius : Real) 
  (h1 : side = 14) (h2 : radius = 3) : 
  side^2 - π * radius^2 = 196 - 9 * π := by
  sorry

end square_field_with_pond_area_l3610_361084


namespace arrangement_count_is_240_l3610_361028

/-- The number of ways to arrange 8 distinct objects in a row,
    where the two smallest objects must be at the ends and
    the largest object must be in the middle. -/
def arrangement_count : ℕ := 240

/-- Theorem stating that the number of arrangements is 240 -/
theorem arrangement_count_is_240 : arrangement_count = 240 := by
  sorry

end arrangement_count_is_240_l3610_361028


namespace insurance_payment_calculation_l3610_361056

/-- The amount of a quarterly insurance payment in dollars. -/
def quarterly_payment : ℕ := 378

/-- The number of quarters in a year. -/
def quarters_per_year : ℕ := 4

/-- The annual insurance payment in dollars. -/
def annual_payment : ℕ := quarterly_payment * quarters_per_year

theorem insurance_payment_calculation :
  annual_payment = 1512 :=
by sorry

end insurance_payment_calculation_l3610_361056


namespace equation_solutions_l3610_361027

theorem equation_solutions :
  let f (x : ℝ) := (8*x^2 - 20*x + 3)/(2*x - 1) + 7*x
  ∀ x : ℝ, f x = 9*x - 3 ↔ x = 1/2 ∨ x = 3 := by sorry

end equation_solutions_l3610_361027


namespace lcm_24_36_45_l3610_361000

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end lcm_24_36_45_l3610_361000


namespace initial_distance_calculation_l3610_361079

/-- Calculates the initial distance between a criminal and a policeman given their speeds and the distance after a certain time. -/
theorem initial_distance_calculation 
  (criminal_speed : ℝ) 
  (policeman_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : criminal_speed = 8) 
  (h2 : policeman_speed = 9) 
  (h3 : time = 3 / 60) 
  (h4 : final_distance = 190) : 
  ∃ (initial_distance : ℝ), 
    initial_distance = final_distance + (policeman_speed - criminal_speed) * time ∧ 
    initial_distance = 190.05 := by
  sorry

end initial_distance_calculation_l3610_361079


namespace triangle_side_b_l3610_361046

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_side_b (t : Triangle) : 
  t.C = 4 * t.A →  -- ∠C = 4∠A
  t.a = 15 →       -- side a = 15
  t.c = 60 →       -- side c = 60
  t.b = 15 * Real.sqrt (2 + Real.sqrt 2) := by
    sorry


end triangle_side_b_l3610_361046


namespace book_cost_proof_l3610_361006

-- Define the cost of one book
def p : ℝ := 1.76

-- State the theorem
theorem book_cost_proof :
  14 * p < 25 ∧ 16 * p > 28 := by
  sorry

end book_cost_proof_l3610_361006


namespace water_average_l3610_361088

def water_problem (day1 day2 day3 : ℕ) : Prop :=
  day1 = 215 ∧
  day2 = day1 + 76 ∧
  day3 = day2 - 53 ∧
  (day1 + day2 + day3) / 3 = 248

theorem water_average : ∃ day1 day2 day3 : ℕ, water_problem day1 day2 day3 := by
  sorry

end water_average_l3610_361088


namespace quadratic_function_property_l3610_361093

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_property (a b c : ℝ) :
  QuadraticFunction a b c 0 = -1 →
  QuadraticFunction a b c 4 = QuadraticFunction a b c 5 →
  ∃ (n : ℤ), QuadraticFunction a b c 11 = n →
  QuadraticFunction a b c 11 = -1 := by
  sorry

end quadratic_function_property_l3610_361093


namespace triangle_theorem_l3610_361041

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Represents a 2D vector --/
structure Vector2D where
  x : ℝ
  y : ℝ

theorem triangle_theorem (t : Triangle) 
  (h_acute : 0 < t.A ∧ t.A < π/2 ∧ 0 < t.B ∧ t.B < π/2 ∧ 0 < t.C ∧ t.C < π/2)
  (h_sum : t.A + t.B + t.C = π)
  (μ : Vector2D)
  (v : Vector2D)
  (h_μ : μ = ⟨t.a^2 + t.c^2 - t.b^2, Real.sqrt 3 * t.a * t.c⟩)
  (h_v : v = ⟨Real.cos t.B, Real.sin t.B⟩)
  (h_parallel : ∃ (k : ℝ), μ = Vector2D.mk (k * v.x) (k * v.y)) :
  t.B = π/3 ∧ 3 * Real.sqrt 3 / 2 < Real.sin t.A + Real.sin t.C ∧ 
  Real.sin t.A + Real.sin t.C ≤ Real.sqrt 3 := by
  sorry

end triangle_theorem_l3610_361041


namespace grade_assignment_count_l3610_361052

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of possible grades -/
def num_grades : ℕ := 4

/-- The number of ways to assign grades to all students -/
def ways_to_assign_grades : ℕ := num_grades ^ num_students

theorem grade_assignment_count :
  ways_to_assign_grades = 16777216 :=
sorry

end grade_assignment_count_l3610_361052


namespace high_school_ten_games_l3610_361098

/-- The number of teams in the conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team -/
def games_per_pair : ℕ := 2

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season -/
def total_games : ℕ := (num_teams.choose 2) * games_per_pair + num_teams * non_conference_games

theorem high_school_ten_games : total_games = 150 := by
  sorry

end high_school_ten_games_l3610_361098


namespace min_distance_ellipse_to_line_l3610_361011

/-- The minimum distance from any point on the ellipse x² + y²/3 = 1 to the line x + y = 4 is √2. -/
theorem min_distance_ellipse_to_line :
  ∀ (x y : ℝ), x^2 + y^2/3 = 1 →
  (∃ (x' y' : ℝ), x' + y' = 4 ∧ (x - x')^2 + (y - y')^2 ≥ 2) ∧
  (∃ (x₀ y₀ : ℝ), x₀ + y₀ = 4 ∧ (x - x₀)^2 + (y - y₀)^2 = 2) :=
by sorry


end min_distance_ellipse_to_line_l3610_361011


namespace salary_change_percentage_l3610_361016

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let increased_salary := initial_salary * 1.5
  let final_salary := increased_salary * 0.9
  (final_salary - initial_salary) / initial_salary * 100 = 35 := by
  sorry

end salary_change_percentage_l3610_361016


namespace elsa_marbles_proof_l3610_361050

/-- The number of marbles in Elsa's new bag -/
def new_bag_marbles : ℕ := by sorry

theorem elsa_marbles_proof :
  let initial_marbles : ℕ := 40
  let lost_at_breakfast : ℕ := 3
  let given_to_susie : ℕ := 5
  let final_marbles : ℕ := 54
  
  new_bag_marbles = 
    final_marbles - 
    (initial_marbles - lost_at_breakfast - given_to_susie + 2 * given_to_susie) := by sorry

end elsa_marbles_proof_l3610_361050


namespace mayoral_election_votes_l3610_361078

theorem mayoral_election_votes (candidate_A_percentage : Real)
                               (candidate_B_percentage : Real)
                               (candidate_C_percentage : Real)
                               (candidate_D_percentage : Real)
                               (vote_difference : ℕ) :
  candidate_A_percentage = 0.35 →
  candidate_B_percentage = 0.40 →
  candidate_C_percentage = 0.15 →
  candidate_D_percentage = 0.10 →
  candidate_A_percentage + candidate_B_percentage + candidate_C_percentage + candidate_D_percentage = 1 →
  vote_difference = 2340 →
  ∃ total_votes : ℕ, 
    (candidate_B_percentage - candidate_A_percentage) * total_votes = vote_difference ∧
    total_votes = 46800 :=
by sorry

end mayoral_election_votes_l3610_361078


namespace milk_selection_l3610_361036

theorem milk_selection (total : ℕ) (soda_count : ℕ) (milk_percent : ℚ) (soda_percent : ℚ) :
  soda_percent = 60 / 100 →
  milk_percent = 20 / 100 →
  soda_count = 72 →
  (milk_percent / soda_percent) * soda_count = 24 := by
  sorry

end milk_selection_l3610_361036


namespace job_candidate_probability_l3610_361072

theorem job_candidate_probability (excel_probability : Real) (day_shift_probability : Real) :
  excel_probability = 0.2 →
  day_shift_probability = 0.7 →
  (1 - day_shift_probability) * excel_probability = 0.06 := by
  sorry

end job_candidate_probability_l3610_361072


namespace unique_positive_root_in_interval_l3610_361053

-- Define the function f(x) = x^2 - x - 1
def f (x : ℝ) : ℝ := x^2 - x - 1

-- State the theorem
theorem unique_positive_root_in_interval :
  (∃! r : ℝ, r > 0 ∧ f r = 0) →  -- There exists a unique positive root
  ∃ r : ℝ, r ∈ Set.Ioo 1 2 ∧ f r = 0 :=  -- The root is in the open interval (1, 2)
by
  sorry

end unique_positive_root_in_interval_l3610_361053


namespace largest_angle_right_isosceles_triangle_l3610_361035

theorem largest_angle_right_isosceles_triangle (D E F : Real) :
  -- Triangle DEF is a right isosceles triangle
  D + E + F = 180 →
  D = E →
  (D = 90 ∨ E = 90 ∨ F = 90) →
  -- Angle D measures 45°
  D = 45 →
  -- The largest interior angle measures 90°
  max D (max E F) = 90 := by
sorry

end largest_angle_right_isosceles_triangle_l3610_361035


namespace counterexample_exists_l3610_361043

theorem counterexample_exists : ∃ (a b c : ℝ), 
  (a^2 + b^2) / (b^2 + c^2) = a/c ∧ a/b ≠ b/c := by
  sorry

end counterexample_exists_l3610_361043


namespace public_area_diameter_l3610_361076

/-- The diameter of the outer boundary of a circular public area -/
def outer_boundary_diameter (play_area_diameter : ℝ) (garden_width : ℝ) (track_width : ℝ) : ℝ :=
  play_area_diameter + 2 * (garden_width + track_width)

/-- Theorem: The diameter of the outer boundary of the running track is 34 feet -/
theorem public_area_diameter : outer_boundary_diameter 14 6 4 = 34 := by
  sorry

end public_area_diameter_l3610_361076


namespace inequality_solution_set_a_range_l3610_361061

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2|

-- Part 1
theorem inequality_solution_set (x : ℝ) :
  (2 * f x < 4 - |x - 1|) ↔ (-7/3 < x ∧ x < -1) :=
sorry

-- Part 2
theorem a_range (a m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) ↔ (-6 ≤ a ∧ a ≤ 2) :=
sorry

end inequality_solution_set_a_range_l3610_361061


namespace initial_discount_percentage_l3610_361020

-- Define the original price of the dress
variable (d : ℝ)
-- Define the initial discount percentage
variable (x : ℝ)

-- Theorem statement
theorem initial_discount_percentage
  (h1 : d > 0)  -- Assuming the original price is positive
  (h2 : 0 ≤ x ∧ x ≤ 100)  -- The discount percentage is between 0 and 100
  (h3 : d * (1 - x / 100) * (1 - 40 / 100) = d * 0.33)  -- The equation representing the final price
  : x = 45 := by
  sorry

end initial_discount_percentage_l3610_361020


namespace subset_condition_intersection_empty_condition_l3610_361042

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
def B : Set ℝ := {x | x < -2 ∨ x > 5}

-- Theorem for part (1)
theorem subset_condition (m : ℝ) : A m ⊆ B ↔ m < 2 ∨ m > 4 := by sorry

-- Theorem for part (2)
theorem intersection_empty_condition (m : ℝ) : A m ∩ B = ∅ ↔ m ≤ 3 := by sorry

end subset_condition_intersection_empty_condition_l3610_361042


namespace new_savings_amount_l3610_361081

def monthly_salary : ℝ := 5750
def initial_savings_rate : ℝ := 0.20
def expense_increase_rate : ℝ := 0.20

theorem new_savings_amount :
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let new_expenses := initial_expenses * (1 + expense_increase_rate)
  let new_savings := monthly_salary - new_expenses
  new_savings = 230 := by sorry

end new_savings_amount_l3610_361081


namespace biography_increase_l3610_361045

theorem biography_increase (B : ℝ) (b n : ℝ) 
  (h1 : b = 0.20 * B)  -- Initial biographies are 20% of total
  (h2 : b + n = 0.32 * (B + n))  -- After purchase, biographies are 32% of new total
  : (n / b) * 100 = 1500 / 17 := by
  sorry

end biography_increase_l3610_361045


namespace alpha_sum_sixth_power_l3610_361064

theorem alpha_sum_sixth_power (α₁ α₂ α₃ : ℂ) 
  (sum_zero : α₁ + α₂ + α₃ = 0)
  (sum_squares : α₁^2 + α₂^2 + α₃^2 = 2)
  (sum_cubes : α₁^3 + α₂^3 + α₃^3 = 4) :
  α₁^6 + α₂^6 + α₃^6 = 7 := by
  sorry

end alpha_sum_sixth_power_l3610_361064


namespace product_of_ratios_l3610_361057

theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2030) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2029)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2030) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2029)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2030) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2029)
  (h₇ : y₁ ≠ 0) (h₈ : y₂ ≠ 0) (h₉ : y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1/1015 := by
sorry

end product_of_ratios_l3610_361057


namespace main_result_l3610_361034

/-- A function satisfying the given property for all real numbers -/
def satisfies_property (g : ℝ → ℝ) : Prop :=
  ∀ a c : ℝ, c^3 * g a = a^3 * g c

/-- The main theorem -/
theorem main_result (g : ℝ → ℝ) (h1 : satisfies_property g) (h2 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 := by
  sorry

end main_result_l3610_361034


namespace consecutive_numbers_product_l3610_361099

theorem consecutive_numbers_product (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 48) : 
  n * (n + 2) = 255 := by
  sorry

end consecutive_numbers_product_l3610_361099


namespace eight_fourth_equals_sixteen_n_l3610_361083

theorem eight_fourth_equals_sixteen_n (n : ℕ) : 8^4 = 16^n → n = 3 := by
  sorry

end eight_fourth_equals_sixteen_n_l3610_361083


namespace missing_files_l3610_361005

theorem missing_files (total : ℕ) (organized_afternoon : ℕ) : total = 60 → organized_afternoon = 15 → total - (total / 2 + organized_afternoon) = 15 := by
  sorry

end missing_files_l3610_361005


namespace complement_of_M_in_U_l3610_361040

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}

theorem complement_of_M_in_U :
  (U \ M) = {-3, -4} := by sorry

end complement_of_M_in_U_l3610_361040


namespace gumball_draw_theorem_l3610_361087

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)

/-- Represents the minimum number of gumballs to draw to guarantee 4 of the same color -/
def minDrawToGuaranteeFour (machine : GumballMachine) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem gumball_draw_theorem (machine : GumballMachine) 
  (h1 : machine.red = 9)
  (h2 : machine.white = 7)
  (h3 : machine.blue = 12) :
  minDrawToGuaranteeFour machine = 12 :=
sorry

end gumball_draw_theorem_l3610_361087


namespace equation_solution_l3610_361085

theorem equation_solution :
  ∃ y : ℚ, (3 / y - 3 / y * y / 5 = 1.2) ∧ (y = 5 / 3) :=
by sorry

end equation_solution_l3610_361085


namespace arithmetic_sequence_geometric_mean_l3610_361029

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_geometric_mean 
  (d : ℝ) (k : ℕ) 
  (h_d : d ≠ 0) 
  (h_k : k > 0) :
  let a := arithmetic_sequence (9 * d) d
  (a k) ^ 2 = a 1 * a (2 * k) → k = 4 := by
sorry

end arithmetic_sequence_geometric_mean_l3610_361029


namespace quadratic_inequality_solution_l3610_361065

def quadratic_inequality (x : ℝ) : Prop := 3 * x^2 + 9 * x + 6 ≤ 0

theorem quadratic_inequality_solution :
  {x : ℝ | quadratic_inequality x} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} := by sorry

end quadratic_inequality_solution_l3610_361065


namespace courtyard_width_l3610_361021

/-- Proves that the width of a courtyard is 25 feet given specific conditions --/
theorem courtyard_width : ∀ (width : ℝ),
  (width > 0) →  -- Ensure width is positive
  (4 * 10 * width * (0.4 * 3 + 0.6 * 1.5) = 2100) →
  width = 25 := by
  sorry

end courtyard_width_l3610_361021


namespace twenty_eight_billion_scientific_notation_l3610_361037

/-- Represents 28 billion -/
def twenty_eight_billion : ℕ := 28000000000

/-- The scientific notation representation of 28 billion -/
def scientific_notation : ℝ := 2.8 * (10 ^ 9)

theorem twenty_eight_billion_scientific_notation : 
  (twenty_eight_billion : ℝ) = scientific_notation := by
  sorry

end twenty_eight_billion_scientific_notation_l3610_361037


namespace number_value_l3610_361004

theorem number_value (number y : ℝ) 
  (h1 : (number + 5) * (y - 5) = 0)
  (h2 : ∀ a b : ℝ, (a + 5) * (b - 5) = 0 → a^2 + b^2 ≥ number^2 + y^2)
  (h3 : number^2 + y^2 = 25) : 
  number = -5 := by
sorry

end number_value_l3610_361004


namespace train_length_train_length_proof_l3610_361032

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * crossing_time

/-- Proof that a train with speed 180 km/h crossing a pole in 1.9998400127989762 seconds is approximately 99.992 meters long -/
theorem train_length_proof : 
  ∃ (ε : ℝ), ε > 0 ∧ |train_length 180 1.9998400127989762 - 99.992| < ε :=
sorry

end train_length_train_length_proof_l3610_361032


namespace water_depth_when_upright_l3610_361007

/-- Represents a right cylindrical water tank -/
structure WaterTank where
  height : ℝ
  baseDiameter : ℝ

/-- Calculates the volume of water in the tank when horizontal -/
def horizontalWaterVolume (tank : WaterTank) (depth : ℝ) : ℝ :=
  sorry

/-- Calculates the depth of water when the tank is upright -/
def uprightWaterDepth (tank : WaterTank) (horizontalDepth : ℝ) : ℝ :=
  sorry

theorem water_depth_when_upright 
  (tank : WaterTank) 
  (h1 : tank.height = 20)
  (h2 : tank.baseDiameter = 6)
  (h3 : horizontalWaterVolume tank 4 = π * (tank.baseDiameter / 2)^2 * tank.height) :
  uprightWaterDepth tank 4 = 20 := by
  sorry

end water_depth_when_upright_l3610_361007


namespace coat_price_reduction_percentage_l3610_361022

/-- The percentage reduction when a coat's price is reduced from $500 to $150 is 70% -/
theorem coat_price_reduction_percentage : 
  let original_price : ℚ := 500
  let reduced_price : ℚ := 150
  let reduction : ℚ := original_price - reduced_price
  let percentage_reduction : ℚ := (reduction / original_price) * 100
  percentage_reduction = 70 := by sorry

end coat_price_reduction_percentage_l3610_361022


namespace same_last_four_digits_l3610_361030

theorem same_last_four_digits (N : ℕ) : 
  N > 0 ∧ 
  ∃ (a b c d : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
  N % 10000 = a * 1000 + b * 100 + c * 10 + d ∧
  (N * N) % 10000 = a * 1000 + b * 100 + c * 10 + d →
  N / 1000 = 937 :=
by sorry

end same_last_four_digits_l3610_361030


namespace tutors_next_meeting_l3610_361054

theorem tutors_next_meeting (elise_schedule fiona_schedule george_schedule harry_schedule : ℕ) 
  (h_elise : elise_schedule = 5)
  (h_fiona : fiona_schedule = 6)
  (h_george : george_schedule = 8)
  (h_harry : harry_schedule = 9) :
  Nat.lcm elise_schedule (Nat.lcm fiona_schedule (Nat.lcm george_schedule harry_schedule)) = 360 := by
  sorry

end tutors_next_meeting_l3610_361054
