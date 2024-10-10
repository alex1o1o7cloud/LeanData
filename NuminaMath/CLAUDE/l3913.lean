import Mathlib

namespace sqrt_pattern_l3913_391393

theorem sqrt_pattern (n : ℕ+) : 
  Real.sqrt (n + 1 / (n + 2)) = ((n + 1) * Real.sqrt (n + 2)) / (n + 2) := by
  sorry

end sqrt_pattern_l3913_391393


namespace m_geq_n_l3913_391377

theorem m_geq_n (a x : ℝ) (h : a > 2) : a + 1 / (a - 2) ≥ 4 - x^2 := by
  sorry

end m_geq_n_l3913_391377


namespace distance_to_plane_value_l3913_391333

-- Define the sphere and points
def Sphere : Type := ℝ × ℝ × ℝ
def Point : Type := ℝ × ℝ × ℝ

-- Define the center and radius of the sphere
def S : Sphere := sorry
def radius : ℝ := 25

-- Define the points on the sphere
def P : Point := sorry
def Q : Point := sorry
def R : Point := sorry

-- Define the distances between points
def PQ : ℝ := 20
def QR : ℝ := 21
def RP : ℝ := 29

-- Define the distance from S to the plane of triangle PQR
def distance_to_plane : ℝ := sorry

-- State the theorem
theorem distance_to_plane_value : distance_to_plane = (266 : ℝ) * Real.sqrt 154 / 14 := by sorry

end distance_to_plane_value_l3913_391333


namespace tank_filling_time_l3913_391342

/-- The time (in hours) it takes to fill the tank without a leak -/
def fill_time : ℝ := 5

/-- The time (in hours) it takes for the leak to empty a full tank -/
def empty_time : ℝ := 30

/-- The extra time (in hours) it takes to fill the tank due to the leak -/
def extra_time : ℝ := 1

theorem tank_filling_time :
  extra_time = (1 / ((1 / fill_time) - (1 / empty_time))) - fill_time :=
by sorry

end tank_filling_time_l3913_391342


namespace multiplicative_inverse_problem_l3913_391311

theorem multiplicative_inverse_problem :
  let A : ℕ := 111112
  let B : ℕ := 142858
  let M : ℕ := 1000003
  let N : ℕ := 513487
  (A * B * N) % M = 1 := by sorry

end multiplicative_inverse_problem_l3913_391311


namespace floor_plus_self_eq_seventeen_fourths_l3913_391341

theorem floor_plus_self_eq_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17/4 ∧ x = 9/4 := by sorry

end floor_plus_self_eq_seventeen_fourths_l3913_391341


namespace bc_length_l3913_391309

-- Define the triangle
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 0) -- Exact coordinates don't matter for this proof
def C : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (0, 0)

-- Define the given lengths
def AD : ℝ := 47
def CD : ℝ := 25
def AC : ℝ := 24

-- Define the theorem
theorem bc_length :
  Triangle A B C →
  Triangle A B D →
  D.1 < C.1 →
  D.2 = B.2 →
  C.2 = B.2 →
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = AD^2 →
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = CD^2 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = AC^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 20.16^2 := by
  sorry

end bc_length_l3913_391309


namespace quadratic_roots_prime_sum_of_digits_l3913_391308

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The main theorem -/
theorem quadratic_roots_prime_sum_of_digits (c : ℕ) :
  (∃ p q : ℕ, 
    Prime p ∧ Prime q ∧ 
    p ≠ q ∧
    p * q = c ∧
    p + q = 85 ∧
    ∀ x : ℝ, x^2 - 85*x + c = 0 ↔ x = p ∨ x = q) →
  sum_of_digits c = 13 :=
sorry

end quadratic_roots_prime_sum_of_digits_l3913_391308


namespace total_birds_l3913_391357

def geese : ℕ := 58
def ducks : ℕ := 37

theorem total_birds : geese + ducks = 95 := by
  sorry

end total_birds_l3913_391357


namespace divisor_problem_l3913_391389

theorem divisor_problem (x k m y : ℤ) 
  (h1 : x = 62 * k + 7)
  (h2 : x + 11 = y * m + 18)
  (h3 : y > 18)
  (h4 : 62 % y = 0) :
  y = 31 := by
  sorry

end divisor_problem_l3913_391389


namespace geometric_sequence_sum_l3913_391359

/-- Given a geometric sequence, returns the sum of the first n terms -/
noncomputable def geometricSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- Proves that for a geometric sequence with specific properties, 
    the sum of the first 9000 terms is 1355 -/
theorem geometric_sequence_sum 
  (a r : ℝ) 
  (h1 : geometricSum a r 3000 = 500)
  (h2 : geometricSum a r 6000 = 950) :
  geometricSum a r 9000 = 1355 := by
  sorry

end geometric_sequence_sum_l3913_391359


namespace find_number_to_multiply_l3913_391371

theorem find_number_to_multiply : ∃ x : ℕ, 
  (43 * x) - (34 * x) = 1224 ∧ x = 136 := by
  sorry

end find_number_to_multiply_l3913_391371


namespace inverse_proportion_problem_l3913_391384

-- Define the inverse proportionality constant
def k : ℝ → ℝ → ℝ := λ x y => x * y

-- Define the conditions
def conditions (x y : ℝ) : Prop :=
  ∃ (c : ℝ), k x y = c ∧ x + y = 30 ∧ x - y = 10

-- Theorem statement
theorem inverse_proportion_problem :
  ∀ x y : ℝ, conditions x y → (x = 4 → y = 50) :=
by sorry

end inverse_proportion_problem_l3913_391384


namespace perimeter_difference_l3913_391319

/-- The perimeter of a rectangle --/
def rectanglePerimeter (length width : ℕ) : ℕ := 2 * (length + width)

/-- The perimeter of a stack of rectangles --/
def stackedRectanglesPerimeter (length width count : ℕ) : ℕ :=
  2 * length + 2 * (width * count)

/-- The difference in perimeters between a 6x1 rectangle and three 3x1 rectangles stacked vertically --/
theorem perimeter_difference :
  rectanglePerimeter 6 1 - stackedRectanglesPerimeter 3 1 3 = 2 := by
  sorry


end perimeter_difference_l3913_391319


namespace arrangements_of_six_acts_l3913_391321

/-- The number of ways to insert two distinguishable items into a sequence of n fixed items -/
def insert_two_items (n : ℕ) : ℕ :=
  (n + 1) * (n + 2)

/-- Theorem stating that inserting 2 items into a sequence of 4 fixed items results in 30 arrangements -/
theorem arrangements_of_six_acts : insert_two_items 4 = 30 := by
  sorry

end arrangements_of_six_acts_l3913_391321


namespace min_handshakes_35_people_l3913_391337

/-- Represents a gathering of people and their handshakes -/
structure Gathering where
  people : ℕ
  handshakes_per_person : ℕ

/-- Calculates the total number of handshakes in a gathering -/
def total_handshakes (g : Gathering) : ℕ :=
  g.people * g.handshakes_per_person / 2

/-- Theorem: In a gathering of 35 people where each person shakes hands with 
    exactly 3 others, the minimum possible number of handshakes is 105 -/
theorem min_handshakes_35_people : 
  ∃ (g : Gathering), g.people = 35 ∧ g.handshakes_per_person = 6 ∧ total_handshakes g = 105 := by
  sorry

#check min_handshakes_35_people

end min_handshakes_35_people_l3913_391337


namespace circle_center_and_radius_l3913_391313

theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 11 = 0 →
  ∃ (h k r : ℝ), h = -1 ∧ k = 2 ∧ r = 2 ∧
  (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end circle_center_and_radius_l3913_391313


namespace quadratic_root_in_arithmetic_sequence_l3913_391334

/-- Given real numbers x, y, z forming an arithmetic sequence with x ≥ y ≥ z ≥ 0,
    if the quadratic zx^2 + yx + x has exactly one root, then this root is -3/4. -/
theorem quadratic_root_in_arithmetic_sequence (x y z : ℝ) :
  (∃ d : ℝ, y = x - d ∧ z = x - 2*d) →  -- arithmetic sequence condition
  x ≥ y →
  y ≥ z →
  z ≥ 0 →
  (∃! r : ℝ, z*r^2 + y*r + x = 0) →  -- exactly one root condition
  (∃ r : ℝ, z*r^2 + y*r + x = 0 ∧ r = -3/4) :=
by sorry

end quadratic_root_in_arithmetic_sequence_l3913_391334


namespace three_petal_percentage_is_75_l3913_391303

/-- The percentage of clovers with three petals -/
def three_petal_percentage : ℝ := 100 - 24 - 1

/-- The total percentage of clovers with two, three, and four petals -/
def total_percentage : ℝ := 100

/-- The percentage of clovers with two petals -/
def two_petal_percentage : ℝ := 24

/-- The percentage of clovers with four petals -/
def four_petal_percentage : ℝ := 1

theorem three_petal_percentage_is_75 :
  three_petal_percentage = 75 :=
by sorry

end three_petal_percentage_is_75_l3913_391303


namespace problem_solution_l3913_391325

theorem problem_solution (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_abc : a * b * c = 1)
  (h_a_c : a + 1 / c = 8)
  (h_b_a : b + 1 / a = 20) : 
  c + 1 / b = 10 / 53 := by
  sorry

end problem_solution_l3913_391325


namespace simplify_sqrt_sum_l3913_391328

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l3913_391328


namespace min_n_for_inequality_l3913_391302

theorem min_n_for_inequality : 
  ∃ (n : ℕ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 ≤ n * (x^4 + y^4 + z^4)) ∧ 
  (∀ (m : ℕ), m < n → ∃ (x y z : ℝ), x^2 + y^2 + z^2 > m * (x^4 + y^4 + z^4)) :=
by
  use 3
  sorry

end min_n_for_inequality_l3913_391302


namespace rectangular_triangle_condition_l3913_391386

theorem rectangular_triangle_condition (A B C : Real) 
  (h : (Real.sin A)^2 + (Real.sin B)^2 + (Real.sin C)^2 = 
       2 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2)) 
  (triangle_angles : A + B + C = Real.pi) :
  A = Real.pi/2 ∨ B = Real.pi/2 ∨ C = Real.pi/2 := by
sorry

end rectangular_triangle_condition_l3913_391386


namespace sector_angle_l3913_391373

/-- Given a sector with radius 1 and area 3π/8, its central angle is 3π/4 -/
theorem sector_angle (r : ℝ) (A : ℝ) (α : ℝ) : 
  r = 1 → A = (3 * π) / 8 → A = (1 / 2) * α * r^2 → α = (3 * π) / 4 := by
  sorry

end sector_angle_l3913_391373


namespace correct_donations_l3913_391322

/-- Represents the donations to five orphanages -/
structure OrphanageDonations where
  total : ℝ
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ
  fifth : ℝ

/-- Checks if the donations satisfy the given conditions -/
def validDonations (d : OrphanageDonations) : Prop :=
  d.total = 1300 ∧
  d.first = 0.2 * d.total ∧
  d.second = d.first / 2 ∧
  d.third = 2 * d.second ∧
  d.fourth = d.fifth ∧
  d.fourth + d.fifth = d.third

/-- Theorem stating that the given donations satisfy all conditions -/
theorem correct_donations :
  ∃ d : OrphanageDonations,
    validDonations d ∧
    d.first = 260 ∧
    d.second = 130 ∧
    d.third = 260 ∧
    d.fourth = 130 ∧
    d.fifth = 130 :=
sorry

end correct_donations_l3913_391322


namespace min_socks_for_fifteen_pairs_l3913_391395

/-- The minimum number of socks needed to ensure at least n pairs of the same color
    when randomly picking from a set of socks with m different colors. -/
def min_socks (n : ℕ) (m : ℕ) : ℕ :=
  m + 1 + 2 * (n - 1)

/-- Theorem: Given a set of socks with 4 different colors, 
    the minimum number of socks that must be randomly picked 
    to ensure at least 15 pairs of the same color is 33. -/
theorem min_socks_for_fifteen_pairs : min_socks 15 4 = 33 := by
  sorry

end min_socks_for_fifteen_pairs_l3913_391395


namespace burger_cost_is_13_l3913_391391

/-- The cost of a single burger given the conditions of Alice's burger purchases in June. -/
def burger_cost (burgers_per_day : ℕ) (days_in_june : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (burgers_per_day * days_in_june)

/-- Theorem stating that the cost of each burger is 13 dollars under the given conditions. -/
theorem burger_cost_is_13 :
  burger_cost 4 30 1560 = 13 := by
  sorry

end burger_cost_is_13_l3913_391391


namespace min_m_plus_n_for_1978_power_divisibility_l3913_391314

theorem min_m_plus_n_for_1978_power_divisibility (m n : ℕ) : 
  m > n → n ≥ 1 → (1000 ∣ 1978^m - 1978^n) → m + n ≥ 106 ∧ ∃ (m₀ n₀ : ℕ), m₀ > n₀ ∧ n₀ ≥ 1 ∧ (1000 ∣ 1978^m₀ - 1978^n₀) ∧ m₀ + n₀ = 106 :=
by sorry

end min_m_plus_n_for_1978_power_divisibility_l3913_391314


namespace museum_trip_total_l3913_391339

/-- The number of people on the first bus -/
def first_bus : ℕ := 12

/-- The number of people on the second bus -/
def second_bus : ℕ := 2 * first_bus

/-- The number of people on the third bus -/
def third_bus : ℕ := second_bus - 6

/-- The number of people on the fourth bus -/
def fourth_bus : ℕ := first_bus + 9

/-- The total number of people going to the museum -/
def total_people : ℕ := first_bus + second_bus + third_bus + fourth_bus

theorem museum_trip_total : total_people = 75 := by
  sorry

end museum_trip_total_l3913_391339


namespace simplify_expression_l3913_391306

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - x^3 - 2) / (3 * x^3))^2) =
  (Real.sqrt (x^12 - 2*x^9 + 6*x^6 - 2*x^3 + 4)) / (3 * x^3) := by
  sorry

end simplify_expression_l3913_391306


namespace hare_run_distance_l3913_391363

/-- The distance between trees in meters -/
def tree_distance : ℕ := 5

/-- The number of the first tree -/
def first_tree : ℕ := 1

/-- The number of the last tree -/
def last_tree : ℕ := 10

/-- The total distance between the first and last tree -/
def total_distance : ℕ := tree_distance * (last_tree - first_tree)

theorem hare_run_distance :
  total_distance = 45 := by sorry

end hare_run_distance_l3913_391363


namespace mass_percentage_N_is_9_66_l3913_391399

/-- The mass percentage of N in a certain compound -/
def mass_percentage_N : ℝ := 9.66

/-- Theorem stating that the mass percentage of N in the compound is 9.66% -/
theorem mass_percentage_N_is_9_66 : mass_percentage_N = 9.66 := by
  sorry

end mass_percentage_N_is_9_66_l3913_391399


namespace cube_volume_from_surface_area_l3913_391330

/-- Given a cube with surface area 6x^2, where x is the length of one side,
    prove that the volume of the cube is x^3. -/
theorem cube_volume_from_surface_area (x : ℝ) (h : x > 0) :
  let surface_area := 6 * x^2
  let side_length := x
  let volume := side_length^3
  surface_area = 6 * side_length^2 → volume = x^3 := by
sorry

end cube_volume_from_surface_area_l3913_391330


namespace ellipse_line_intersection_l3913_391304

/-- The ellipse E -/
def E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The line l -/
def l (k m x y : ℝ) : Prop := y = k*x + m

/-- Predicate to check if a point is on the ellipse E -/
def on_ellipse (x y : ℝ) : Prop := E x y

/-- Predicate to check if a point is on the line l -/
def on_line (k m x y : ℝ) : Prop := l k m x y

/-- The right vertex of the ellipse -/
def right_vertex : ℝ × ℝ := (2, 0)

/-- Predicate to check if two points are different -/
def different (p1 p2 : ℝ × ℝ) : Prop := p1 ≠ p2

theorem ellipse_line_intersection (k m : ℝ) :
  ∃ (M N : ℝ × ℝ),
    on_ellipse M.1 M.2 ∧
    on_ellipse N.1 N.2 ∧
    on_line k m M.1 M.2 ∧
    on_line k m N.1 N.2 ∧
    different M right_vertex ∧
    different N right_vertex ∧
    different M N →
    on_line k m (2/7) 0 :=
sorry

end ellipse_line_intersection_l3913_391304


namespace smallest_divisor_of_427395_l3913_391345

theorem smallest_divisor_of_427395 : 
  ∀ d : ℕ, d > 0 ∧ d < 5 → ¬(427395 % d = 0) ∧ 427395 % 5 = 0 := by sorry

end smallest_divisor_of_427395_l3913_391345


namespace multiple_of_x_l3913_391301

theorem multiple_of_x (x y m : ℤ) : 
  (4 * x + y = 34) →
  (m * x - y = 20) →
  (y^2 = 4) →
  m = 2 := by sorry

end multiple_of_x_l3913_391301


namespace max_servings_is_16_l3913_391390

/-- Represents the number of servings that can be made from a given ingredient --/
def servings_from_ingredient (available : ℕ) (required : ℕ) : ℕ :=
  (available * 4) / required

/-- Represents the recipe requirements for 4 servings --/
structure Recipe :=
  (bananas : ℕ)
  (yogurt : ℕ)
  (honey : ℕ)
  (strawberries : ℕ)

/-- Represents the available ingredients --/
structure Available :=
  (bananas : ℕ)
  (yogurt : ℕ)
  (honey : ℕ)
  (strawberries : ℕ)

/-- Calculates the maximum number of servings that can be made --/
def max_servings (recipe : Recipe) (available : Available) : ℕ :=
  min
    (min (servings_from_ingredient available.bananas recipe.bananas)
         (servings_from_ingredient available.yogurt recipe.yogurt))
    (min (servings_from_ingredient available.honey recipe.honey)
         (servings_from_ingredient available.strawberries recipe.strawberries))

theorem max_servings_is_16 (recipe : Recipe) (available : Available) :
  recipe.bananas = 3 ∧ recipe.yogurt = 1 ∧ recipe.honey = 2 ∧ recipe.strawberries = 2 ∧
  available.bananas = 12 ∧ available.yogurt = 6 ∧ available.honey = 16 ∧ available.strawberries = 8 →
  max_servings recipe available = 16 := by
  sorry

end max_servings_is_16_l3913_391390


namespace largest_integer_solution_largest_integer_value_negative_four_satisfies_largest_integer_is_negative_four_l3913_391324

theorem largest_integer_solution (x : ℤ) : (5 - 4*x > 17) ↔ (x < -3) :=
  sorry

theorem largest_integer_value : ∀ x : ℤ, (5 - 4*x > 17) → (x ≤ -4) :=
  sorry

theorem negative_four_satisfies : (5 - 4*(-4) > 17) :=
  sorry

theorem largest_integer_is_negative_four : 
  ∀ x : ℤ, (5 - 4*x > 17) → (x ≤ -4) ∧ (-4 ≤ x) → x = -4 :=
  sorry

end largest_integer_solution_largest_integer_value_negative_four_satisfies_largest_integer_is_negative_four_l3913_391324


namespace lychees_remaining_l3913_391398

theorem lychees_remaining (initial : ℕ) (sold_fraction : ℚ) (eaten_fraction : ℚ) : 
  initial = 500 → 
  sold_fraction = 1/2 → 
  eaten_fraction = 3/5 → 
  (initial - initial * sold_fraction - (initial - initial * sold_fraction) * eaten_fraction : ℚ) = 100 := by
  sorry

end lychees_remaining_l3913_391398


namespace min_value_function_compare_squares_min_value_M_l3913_391364

-- Part 1
theorem min_value_function (x : ℝ) (h : x > -1) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 2 + 3 ∧
  ∀ (y : ℝ), y = ((x + 2) * (x + 3)) / (x + 1) → y ≥ min_val :=
sorry

-- Part 2
theorem compare_squares (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1)
  (h : x^2 / a^2 - y^2 / b^2 = 1) :
  a^2 - b^2 ≤ (x - y)^2 :=
sorry

-- Part 3
theorem min_value_M (m : ℝ) (hm : m ≥ 1) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 3 / 2 ∧
  ∀ (M : ℝ), M = Real.sqrt (4 * m - 3) - Real.sqrt (m - 1) → M ≥ min_val :=
sorry

end min_value_function_compare_squares_min_value_M_l3913_391364


namespace sedans_sold_prediction_l3913_391374

/-- The ratio of sports cars to sedans -/
def car_ratio : ℚ := 3 / 5

/-- The number of sports cars predicted to be sold -/
def sports_cars_sold : ℕ := 36

/-- The number of sedans expected to be sold -/
def sedans_sold : ℕ := 60

/-- Theorem stating the relationship between sports cars and sedans sold -/
theorem sedans_sold_prediction :
  (car_ratio * sports_cars_sold : ℚ) = sedans_sold := by sorry

end sedans_sold_prediction_l3913_391374


namespace scaled_variance_l3913_391348

def variance (data : List ℝ) : ℝ := sorry

theorem scaled_variance (data : List ℝ) (h : variance data = 3) :
  variance (List.map (· * 2) data) = 12 := by sorry

end scaled_variance_l3913_391348


namespace banana_bread_theorem_l3913_391378

def bananas_per_loaf : ℕ := 4
def monday_loaves : ℕ := 3
def tuesday_loaves : ℕ := 2 * monday_loaves

def total_loaves : ℕ := monday_loaves + tuesday_loaves
def total_bananas : ℕ := total_loaves * bananas_per_loaf

theorem banana_bread_theorem : total_bananas = 36 := by
  sorry

end banana_bread_theorem_l3913_391378


namespace select_with_defective_test_methods_l3913_391312

-- Define the total number of products and the number of defective products
def total_products : ℕ := 10
def defective_products : ℕ := 3

-- Define the function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem for the first question
theorem select_with_defective :
  binomial total_products 3 - binomial (total_products - defective_products) 3 = 85 :=
sorry

-- Theorem for the second question
theorem test_methods :
  binomial defective_products 1 * (binomial (total_products - defective_products) 2 * binomial 2 2) * binomial 4 4 = 1512 :=
sorry

end select_with_defective_test_methods_l3913_391312


namespace recurring_decimal_fraction_l3913_391320

theorem recurring_decimal_fraction :
  (5 : ℚ) / 33 / ((2401 : ℚ) / 999) = 4995 / 79233 := by sorry

end recurring_decimal_fraction_l3913_391320


namespace a_subset_M_l3913_391332

noncomputable def a : ℝ := Real.sqrt 3

def M : Set ℝ := {x | x ≤ 3}

theorem a_subset_M : {a} ⊆ M := by sorry

end a_subset_M_l3913_391332


namespace equal_water_after_operations_l3913_391396

theorem equal_water_after_operations (x : ℝ) (h : x > 0) :
  let barrel1 := x * 0.9 * 1.1
  let barrel2 := x * 1.1 * 0.9
  barrel1 = barrel2 := by sorry

end equal_water_after_operations_l3913_391396


namespace sine_cosine_sum_l3913_391382

theorem sine_cosine_sum (α : Real) (h : Real.sin (α - π/6) = 1/3) :
  Real.sin (2*α - π/6) + Real.cos (2*α) = 7/9 := by
  sorry

end sine_cosine_sum_l3913_391382


namespace index_card_area_l3913_391327

theorem index_card_area (length width : ℝ) (h1 : length = 5) (h2 : width = 7) : 
  (∃ side, (side - 2) * width = 21 ∨ length * (side - 2) = 21) →
  (length * (width - 2) = 25 ∨ (length - 2) * width = 25) := by
sorry

end index_card_area_l3913_391327


namespace min_value_a_l3913_391354

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (a - 1) * x^2 - 2 * Real.sqrt 2 * x * y + a * y^2 ≥ 0) →
  a ≥ 2 ∧ ∀ b : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → (b - 1) * x^2 - 2 * Real.sqrt 2 * x * y + b * y^2 ≥ 0) → b ≥ a :=
by sorry

end min_value_a_l3913_391354


namespace piggy_bank_savings_l3913_391380

theorem piggy_bank_savings (first_year : ℝ) : 
  first_year + 2 * first_year + 4 * first_year + 8 * first_year = 450 →
  first_year = 30 := by
sorry

end piggy_bank_savings_l3913_391380


namespace marble_ratio_proof_l3913_391361

def marble_problem (initial_marbles : ℕ) (lost_through_hole : ℕ) (final_marbles : ℕ) : Prop :=
  let dog_eaten : ℕ := lost_through_hole / 2
  let before_giving_away : ℕ := initial_marbles - lost_through_hole - dog_eaten
  let given_away : ℕ := before_giving_away - final_marbles
  (given_away : ℚ) / lost_through_hole = 2

theorem marble_ratio_proof :
  marble_problem 24 4 10 := by
  sorry

end marble_ratio_proof_l3913_391361


namespace horse_rider_ratio_l3913_391318

theorem horse_rider_ratio :
  ∀ (total_horses : ℕ) (total_legs_walking : ℕ),
    total_horses = 12 →
    total_legs_walking = 60 →
    ∃ (riding_owners walking_owners : ℕ),
      riding_owners + walking_owners = total_horses ∧
      walking_owners * 6 = total_legs_walking ∧
      riding_owners * 6 = total_horses := by
  sorry

end horse_rider_ratio_l3913_391318


namespace unique_A_for_multiple_of_9_l3913_391344

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def sum_of_digits (A : ℕ) : ℕ := 2 + A + 3 + A

def four_digit_number (A : ℕ) : ℕ := 2000 + 100 * A + 30 + A

theorem unique_A_for_multiple_of_9 :
  ∃! A : ℕ, A < 10 ∧ is_multiple_of_9 (four_digit_number A) ∧ A = 2 :=
sorry

end unique_A_for_multiple_of_9_l3913_391344


namespace student_rank_theorem_l3913_391392

/-- Calculates the rank from the last given the total number of students and rank from the top -/
def rankFromLast (totalStudents : ℕ) (rankFromTop : ℕ) : ℕ :=
  totalStudents - rankFromTop + 1

/-- Theorem stating that in a class of 35 students, if a student ranks 14th from the top, their rank from the last is 22nd -/
theorem student_rank_theorem (totalStudents : ℕ) (rankFromTop : ℕ) 
  (h1 : totalStudents = 35) (h2 : rankFromTop = 14) : 
  rankFromLast totalStudents rankFromTop = 22 := by
  sorry

end student_rank_theorem_l3913_391392


namespace line_passes_through_point_l3913_391347

/-- The line equation passes through the point (2,2) for all values of k -/
theorem line_passes_through_point :
  ∀ (k : ℝ), (1 + 4*k) * 2 - (2 - 3*k) * 2 + 2 - 14*k = 0 := by
  sorry

end line_passes_through_point_l3913_391347


namespace number_comparison_l3913_391385

theorem number_comparison : 22^44 > 33^33 ∧ 33^33 > 44^22 := by sorry

end number_comparison_l3913_391385


namespace five_digit_multiple_of_9_l3913_391336

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

theorem five_digit_multiple_of_9 (d : ℕ) (h1 : d < 10) :
  is_multiple_of_9 (63470 + d) ↔ d = 7 := by sorry

end five_digit_multiple_of_9_l3913_391336


namespace ball_radius_from_hole_l3913_391317

theorem ball_radius_from_hole (hole_diameter : ℝ) (hole_depth : ℝ) (ball_radius : ℝ) : 
  hole_diameter = 24 →
  hole_depth = 8 →
  (hole_diameter / 2) ^ 2 + (ball_radius - hole_depth) ^ 2 = ball_radius ^ 2 →
  ball_radius = 13 := by
sorry

end ball_radius_from_hole_l3913_391317


namespace inequality_solution_set_l3913_391335

theorem inequality_solution_set :
  {x : ℝ | |x - 1| + |2*x + 5| < 8} = Set.Ioo (-4 : ℝ) (4/3) := by sorry

end inequality_solution_set_l3913_391335


namespace fashion_show_runway_time_l3913_391383

/-- The fashion show runway problem -/
theorem fashion_show_runway_time :
  let num_models : ℕ := 6
  let bathing_suits_per_model : ℕ := 2
  let evening_wear_per_model : ℕ := 3
  let time_per_trip : ℕ := 2

  let total_trips_per_model : ℕ := bathing_suits_per_model + evening_wear_per_model
  let total_trips : ℕ := num_models * total_trips_per_model
  let total_time : ℕ := total_trips * time_per_trip

  total_time = 60
  := by sorry

end fashion_show_runway_time_l3913_391383


namespace unique_projection_l3913_391381

def vector_projection (a b s p : ℝ × ℝ) : Prop :=
  let line_dir := (b.1 - a.1, b.2 - a.2)
  let shifted_line (t : ℝ) := (a.1 + s.1 + t * line_dir.1, a.2 + s.2 + t * line_dir.2)
  ∃ t : ℝ, 
    p = shifted_line t ∧ 
    line_dir.1 * (p.1 - (a.1 + s.1)) + line_dir.2 * (p.2 - (a.2 + s.2)) = 0

theorem unique_projection :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (-1, 4)
  let s : ℝ × ℝ := (1, 1)
  let p : ℝ × ℝ := (16/13, 41/26)
  vector_projection a b s p ∧ 
  ∀ q : ℝ × ℝ, vector_projection a b s q → q = p := by sorry

end unique_projection_l3913_391381


namespace polygon_sides_range_l3913_391368

/-- Represents the count of vertices with different internal angles -/
structure VertexCounts where
  a : ℕ  -- Count of 60° angles
  b : ℕ  -- Count of 90° angles
  c : ℕ  -- Count of 120° angles
  d : ℕ  -- Count of 150° angles

/-- Theorem stating the possible values of n for a convex n-sided polygon 
    formed by combining equilateral triangles and squares -/
theorem polygon_sides_range (n : ℕ) : 
  (∃ v : VertexCounts, 
    v.a + v.b + v.c + v.d = n ∧ 
    4 * v.a + 3 * v.b + 2 * v.c + v.d = 12 ∧
    v.a + v.b > 0 ∧ v.c + v.d > 0) ↔ 
  5 ≤ n ∧ n ≤ 12 :=
sorry

end polygon_sides_range_l3913_391368


namespace largest_quantity_l3913_391307

def A : ℚ := 3003 / 3002 + 3003 / 3004
def B : ℚ := 3003 / 3004 + 3005 / 3004
def C : ℚ := 3004 / 3003 + 3004 / 3005

theorem largest_quantity : A > B ∧ A > C := by sorry

end largest_quantity_l3913_391307


namespace square_of_cube_of_fourth_smallest_prime_l3913_391315

def fourth_smallest_prime : ℕ := 7

theorem square_of_cube_of_fourth_smallest_prime :
  (fourth_smallest_prime ^ 3) ^ 2 = 117649 := by
  sorry

end square_of_cube_of_fourth_smallest_prime_l3913_391315


namespace total_turnips_count_l3913_391387

/-- The number of turnips grown by Sally -/
def sally_turnips : ℕ := 113

/-- The number of turnips grown by Mary -/
def mary_turnips : ℕ := 129

/-- The total number of turnips grown by Sally and Mary -/
def total_turnips : ℕ := sally_turnips + mary_turnips

theorem total_turnips_count : total_turnips = 242 := by
  sorry

end total_turnips_count_l3913_391387


namespace smallest_c_for_inequality_l3913_391329

theorem smallest_c_for_inequality (m n : ℕ) : 
  (∀ c : ℕ, (27 ^ c) * (2 ^ (24 - n)) > (3 ^ (24 + m)) * (5 ^ n) → c ≥ 9) ∧ 
  ((27 ^ 9) * (2 ^ (24 - n)) > (3 ^ (24 + m)) * (5 ^ n)) := by
  sorry

end smallest_c_for_inequality_l3913_391329


namespace zero_exponent_rule_l3913_391316

theorem zero_exponent_rule (x : ℚ) (h : x ≠ 0) : x ^ (0 : ℕ) = 1 := by
  sorry

end zero_exponent_rule_l3913_391316


namespace advantage_is_most_appropriate_l3913_391358

/-- Represents the beneficial aspect of language skills in a job context -/
def BeneficialAspect : Type := String

/-- The set of possible words to fill in the blank -/
def WordChoices : Set String := {"chance", "ability", "possibility", "advantage"}

/-- Predicate to check if a word appropriately describes the beneficial aspect of language skills -/
def IsAppropriateWord (word : String) : Prop :=
  word ∈ WordChoices ∧ 
  word = "advantage"

/-- Theorem stating that "advantage" is the most appropriate word -/
theorem advantage_is_most_appropriate : 
  ∃ (word : String), IsAppropriateWord word ∧ 
  ∀ (other : String), IsAppropriateWord other → other = word :=
sorry

end advantage_is_most_appropriate_l3913_391358


namespace rectangular_field_area_l3913_391355

theorem rectangular_field_area (L W : ℝ) : 
  L = 10 →                   -- One side is 10 feet
  2 * W + L = 146 →          -- Total fencing is 146 feet
  L * W = 680 :=             -- Area of the field is 680 square feet
by
  sorry

end rectangular_field_area_l3913_391355


namespace whale_prediction_correct_l3913_391351

/-- The number of whales predicted for next year -/
def whales_next_year : ℕ := 8800

/-- The number of whales last year -/
def whales_last_year : ℕ := 4000

/-- The number of whales this year -/
def whales_this_year : ℕ := 2 * whales_last_year

/-- The predicted increase in the number of whales for next year -/
def predicted_increase : ℕ := whales_next_year - whales_this_year

theorem whale_prediction_correct : predicted_increase = 800 := by
  sorry

end whale_prediction_correct_l3913_391351


namespace integral_sqrt_plus_x_l3913_391356

theorem integral_sqrt_plus_x :
  ∫ (x : ℝ) in (0)..(1), (Real.sqrt (1 - x^2) + x) = π / 4 + 1 / 2 := by
  sorry

end integral_sqrt_plus_x_l3913_391356


namespace max_cookies_andy_l3913_391362

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_cookies_andy (total : ℕ) (x : ℕ) (p : ℕ) 
  (h_total : total = 30)
  (h_prime : is_prime p)
  (h_all_eaten : x + p * x = total) :
  x ≤ 10 ∧ ∃ (x₀ : ℕ) (p₀ : ℕ), x₀ = 10 ∧ is_prime p₀ ∧ x₀ + p₀ * x₀ = total :=
sorry

end max_cookies_andy_l3913_391362


namespace solution_set_equivalence_l3913_391352

/-- The solution set of the system of equations {x - 2y = 1, x^3 - 6xy - 8y^3 = 1} 
    is equivalent to the line y = (x-1)/2 -/
theorem solution_set_equivalence (x y : ℝ) : 
  (x - 2*y = 1 ∧ x^3 - 6*x*y - 8*y^3 = 1) ↔ y = (x - 1) / 2 :=
sorry

end solution_set_equivalence_l3913_391352


namespace area_of_efgh_is_72_l3913_391331

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with two opposite corners -/
structure Rectangle where
  corner1 : ℝ × ℝ
  corner2 : ℝ × ℝ

/-- The configuration of circles and rectangle in the problem -/
structure CircleConfiguration where
  efgh : Rectangle
  circleA : Circle
  circleB : Circle
  circleC : Circle
  circleD : Circle

/-- Checks if two circles are congruent -/
def areCongruentCircles (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

/-- Checks if a circle is tangent to two adjacent sides of a rectangle -/
def isTangentToAdjacentSides (c : Circle) (r : Rectangle) : Prop :=
  sorry -- Definition omitted for brevity

/-- Checks if the centers of four circles form a rectangle -/
def centersFormRectangle (c1 c2 c3 c4 : Circle) : Prop :=
  sorry -- Definition omitted for brevity

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  sorry -- Definition omitted for brevity

theorem area_of_efgh_is_72 (config : CircleConfiguration) :
  (areCongruentCircles config.circleA config.circleB) →
  (areCongruentCircles config.circleA config.circleC) →
  (areCongruentCircles config.circleA config.circleD) →
  (config.circleB.radius = 3) →
  (isTangentToAdjacentSides config.circleB config.efgh) →
  (centersFormRectangle config.circleA config.circleB config.circleC config.circleD) →
  rectangleArea config.efgh = 72 :=
by
  sorry

end area_of_efgh_is_72_l3913_391331


namespace quadratic_coefficient_l3913_391379

/-- A quadratic function with vertex form (x + h)^2 passing through a specific point -/
def QuadraticFunction (a : ℝ) (h : ℝ) (x₀ : ℝ) (y₀ : ℝ) : Prop :=
  y₀ = a * (x₀ + h)^2

theorem quadratic_coefficient (a : ℝ) :
  QuadraticFunction a 3 2 (-50) → a = -2 := by
  sorry

end quadratic_coefficient_l3913_391379


namespace eleventh_inning_score_l3913_391343

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalScore : ℕ

/-- Calculates the average score of a batsman -/
def average (stats : BatsmanStats) : ℚ :=
  stats.totalScore / stats.innings

/-- Theorem: Given the conditions, the score in the 11th inning is 110 runs -/
theorem eleventh_inning_score
  (stats10 : BatsmanStats)
  (stats11 : BatsmanStats)
  (h1 : stats10.innings = 10)
  (h2 : stats11.innings = 11)
  (h3 : average stats11 = 60)
  (h4 : average stats11 - average stats10 = 5) :
  stats11.totalScore - stats10.totalScore = 110 := by
  sorry

end eleventh_inning_score_l3913_391343


namespace fraction_equality_l3913_391360

theorem fraction_equality (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (1 / (y + 1)) / (1 / (x + 2)) = 1 := by
  sorry

end fraction_equality_l3913_391360


namespace y_value_l3913_391338

theorem y_value (y : ℝ) (h : (9 : ℝ) / y^2 = y / 81) : y = 9 := by
  sorry

end y_value_l3913_391338


namespace largest_possible_median_l3913_391394

def number_set (x : ℤ) : Finset ℤ := {x, 2*x, 3, 2, 5}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (λ y => y ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ y => y ≥ m)).card ≥ s.card

theorem largest_possible_median (x : ℤ) :
  ∃ m : ℤ, is_median m (number_set x) ∧ ∀ n : ℤ, is_median n (number_set x) → n ≤ m :=
by
  sorry

end largest_possible_median_l3913_391394


namespace truncated_pyramid_overlap_l3913_391372

/-- Regular triangular pyramid with planar angle α at the vertex -/
structure RegularTriangularPyramid where
  α : ℝ  -- Planar angle at the vertex

/-- Regular truncated pyramid cut from a regular triangular pyramid -/
structure RegularTruncatedPyramid (p : RegularTriangularPyramid) where

/-- Unfolded development of a regular truncated pyramid -/
def UnfoldedDevelopment (t : RegularTruncatedPyramid p) : Type := sorry

/-- Predicate to check if an unfolded development overlaps itself -/
def is_self_overlapping (d : UnfoldedDevelopment t) : Prop := sorry

theorem truncated_pyramid_overlap (p : RegularTriangularPyramid) 
  (t : RegularTruncatedPyramid p) (d : UnfoldedDevelopment t) :
  is_self_overlapping d ↔ 100 * π / 180 < p.α ∧ p.α < 120 * π / 180 := by
  sorry

end truncated_pyramid_overlap_l3913_391372


namespace exponential_inequality_l3913_391388

theorem exponential_inequality (a b m : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : a ≠ 1) 
  (h4 : b ≠ 1) 
  (h5 : 0 < m) 
  (h6 : m < 1) : 
  m^a < m^b := by
  sorry

end exponential_inequality_l3913_391388


namespace point_transformation_l3913_391323

/-- Rotation of 90 degrees around the z-axis -/
def rotateZ90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

/-- Reflection through the xy-plane -/
def reflectXY (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

/-- Reflection through the yz-plane -/
def reflectYZ (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

/-- The sequence of transformations applied to the point -/
def transformPoint (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotateZ90 |> reflectXY |> reflectYZ |> rotateZ90 |> reflectYZ

theorem point_transformation :
  transformPoint (2, 3, 4) = (2, 3, -4) := by
  sorry

end point_transformation_l3913_391323


namespace area_between_concentric_circles_l3913_391367

/-- Given three concentric circles with radii r, s, and t, where r > s > t,
    and p, q as defined in the problem, prove that the area between the
    largest and smallest circles is π(p² + q²). -/
theorem area_between_concentric_circles
  (r s t p q : ℝ)
  (h_order : r > s ∧ s > t)
  (h_p : p = (r^2 - s^2).sqrt)
  (h_q : q = (s^2 - t^2).sqrt) :
  π * (r^2 - t^2) = π * (p^2 + q^2) := by
  sorry

end area_between_concentric_circles_l3913_391367


namespace plane_perpendicular_deduction_l3913_391365

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_deduction 
  (m n : Line) (α β : Plane) 
  (h1 : parallel m n) 
  (h2 : subset m α) 
  (h3 : perp_line_plane n β) : 
  perp_plane α β :=
sorry

end plane_perpendicular_deduction_l3913_391365


namespace ostap_chess_scenario_exists_l3913_391300

theorem ostap_chess_scenario_exists : ∃ (N : ℕ), N + 5 * N + 10 * N = 64 := by
  sorry

end ostap_chess_scenario_exists_l3913_391300


namespace absolute_value_inequality_l3913_391310

theorem absolute_value_inequality (x : ℝ) :
  (abs (x + 2) + abs (x - 1) ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) := by sorry

end absolute_value_inequality_l3913_391310


namespace sprint_competition_races_l3913_391366

/-- The number of races needed to determine a champion in a sprint competition --/
def racesNeeded (totalSprinters : ℕ) (lanesPerRace : ℕ) (eliminatedPerRace : ℕ) : ℕ :=
  Nat.ceil ((totalSprinters - 1) / eliminatedPerRace)

/-- Theorem stating that 46 races are needed for the given conditions --/
theorem sprint_competition_races : 
  racesNeeded 320 8 7 = 46 := by
  sorry

end sprint_competition_races_l3913_391366


namespace number_thought_of_l3913_391326

theorem number_thought_of (x : ℝ) : x / 5 + 23 = 42 → x = 95 := by
  sorry

end number_thought_of_l3913_391326


namespace square_minus_three_product_plus_square_l3913_391353

theorem square_minus_three_product_plus_square (a b : ℝ) 
  (sum_eq : a + b = 8) 
  (product_eq : a * b = 9) : 
  a^2 - 3*a*b + b^2 = 19 := by
sorry

end square_minus_three_product_plus_square_l3913_391353


namespace boat_speed_in_still_water_l3913_391370

/-- The speed of a boat in still water, given that:
    1. It takes 90 minutes less to travel 36 miles downstream than upstream.
    2. The speed of the stream is 2 mph. -/
theorem boat_speed_in_still_water : ∃ (b : ℝ),
  (36 / (b - 2) - 36 / (b + 2) = 1.5) ∧ b = 10 := by sorry

end boat_speed_in_still_water_l3913_391370


namespace solution_satisfies_system_solution_is_unique_l3913_391305

/-- The solution to the system of equations 4x - 6y = -3 and 9x + 3y = 6.3 -/
def solution_pair : ℝ × ℝ := (0.436, 0.792)

/-- The first equation of the system -/
def equation1 (x y : ℝ) : Prop := 4 * x - 6 * y = -3

/-- The second equation of the system -/
def equation2 (x y : ℝ) : Prop := 9 * x + 3 * y = 6.3

/-- Theorem stating that the solution_pair satisfies both equations -/
theorem solution_satisfies_system : 
  let (x, y) := solution_pair
  equation1 x y ∧ equation2 x y :=
by sorry

/-- Theorem stating that the solution is unique -/
theorem solution_is_unique :
  ∀ (x y : ℝ), equation1 x y ∧ equation2 x y → (x, y) = solution_pair :=
by sorry

end solution_satisfies_system_solution_is_unique_l3913_391305


namespace village_population_l3913_391350

theorem village_population (P : ℕ) : 
  (((((P * 95 / 100) * 85 / 100) * 93 / 100) * 80 / 100) * 90 / 100) * 75 / 100 = 3553 →
  P = 9262 := by
  sorry

end village_population_l3913_391350


namespace abc_inequality_l3913_391340

theorem abc_inequality (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
sorry

end abc_inequality_l3913_391340


namespace max_value_of_function_l3913_391346

theorem max_value_of_function (x : ℝ) (h : x < 5/4) :
  (4 * x - 2 + 1 / (4 * x - 5)) ≤ 1 ∧ 
  ∃ y : ℝ, y < 5/4 ∧ 4 * y - 2 + 1 / (4 * y - 5) = 1 :=
by sorry

end max_value_of_function_l3913_391346


namespace nancy_problem_rate_l3913_391375

/-- Given Nancy's homework details, prove she can finish 8 problems per hour -/
theorem nancy_problem_rate :
  let math_problems : ℝ := 17.0
  let spelling_problems : ℝ := 15.0
  let total_hours : ℝ := 4.0
  let total_problems := math_problems + spelling_problems
  let problems_per_hour := total_problems / total_hours
  problems_per_hour = 8 := by sorry

end nancy_problem_rate_l3913_391375


namespace remainder_problem_l3913_391376

theorem remainder_problem (k : ℕ+) (h : 80 % (k^2 : ℕ) = 8) : 150 % (k : ℕ) = 6 := by
  sorry

end remainder_problem_l3913_391376


namespace orthogonal_trajectories_and_intersection_angle_l3913_391349

-- Define the family of conics
def conic (a : ℝ) (x y : ℝ) : Prop :=
  (x + 2*y)^2 = a*(x + y)

-- Define the orthogonal trajectory
def orthogonal_trajectory (c : ℝ) (x y : ℝ) : Prop :=
  y = c*x^2 - 3*x

-- Theorem statement
theorem orthogonal_trajectories_and_intersection_angle :
  ∀ (a c : ℝ),
  (∃ (x y : ℝ), conic a x y ∧ orthogonal_trajectory c x y) ∧
  (∃ (x y : ℝ), conic a x y ∧ x = 0 ∧ y = 0 ∧
    ∃ (x' y' : ℝ), orthogonal_trajectory c x' y' ∧ x' = 0 ∧ y' = 0 ∧
    Real.arctan ((y' - y) / (x' - x)) = π / 4) :=
by sorry


end orthogonal_trajectories_and_intersection_angle_l3913_391349


namespace smallest_common_difference_l3913_391397

/-- Represents a quadratic equation ax^2 + bx + c = 0 --/
structure QuadraticEquation where
  a : Int
  b : Int
  c : Int

/-- Checks if a quadratic equation has two distinct roots --/
def hasTwoDistinctRoots (eq : QuadraticEquation) : Prop :=
  eq.b * eq.b - 4 * eq.a * eq.c > 0

/-- Generates all possible quadratic equations with coefficients a, b, 2c --/
def generateQuadraticEquations (a b c : Int) : List QuadraticEquation :=
  [
    ⟨a, b, 2*c⟩, ⟨a, 2*c, b⟩, ⟨b, a, 2*c⟩,
    ⟨b, 2*c, a⟩, ⟨2*c, a, b⟩, ⟨2*c, b, a⟩
  ]

theorem smallest_common_difference
  (a b c : Int)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_arithmetic : ∃ d : Int, b = a + d ∧ c = a + 2*d)
  (h_increasing : a < b ∧ b < c)
  (h_distinct_roots : ∀ eq ∈ generateQuadraticEquations a b c, hasTwoDistinctRoots eq) :
  ∃ d : Int, d = 4 ∧ a = -5 ∧ b = -1 ∧ c = 3 :=
sorry

end smallest_common_difference_l3913_391397


namespace bella_friends_count_l3913_391369

/-- The number of beads needed per bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of beads Bella currently has -/
def current_beads : ℕ := 36

/-- The number of additional beads Bella needs -/
def additional_beads : ℕ := 12

/-- The number of friends Bella is making bracelets for -/
def num_friends : ℕ := (current_beads + additional_beads) / beads_per_bracelet

theorem bella_friends_count : num_friends = 6 := by
  sorry

end bella_friends_count_l3913_391369
