import Mathlib

namespace greatest_integer_for_all_real_domain_l3778_377851

theorem greatest_integer_for_all_real_domain : 
  ∃ (b : ℤ), (∀ (x : ℝ), x^2 + b * x + 12 ≠ 0) ∧ 
  (∀ (c : ℤ), c > b → ∃ (x : ℝ), x^2 + c * x + 12 = 0) :=
by sorry

end greatest_integer_for_all_real_domain_l3778_377851


namespace tens_digit_of_8_power_1701_l3778_377832

theorem tens_digit_of_8_power_1701 : ∃ n : ℕ, 8^1701 ≡ n [ZMOD 100] ∧ n < 100 ∧ (n / 10 : ℕ) = 0 :=
sorry

end tens_digit_of_8_power_1701_l3778_377832


namespace power_of_i_sum_l3778_377818

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_sum : i^123 - i^321 + i^432 = -2*i + 1 := by
  sorry

end power_of_i_sum_l3778_377818


namespace cafeteria_tables_l3778_377849

/-- The number of tables in a cafeteria --/
def num_tables : ℕ := 15

/-- The number of seats per table --/
def seats_per_table : ℕ := 10

/-- The fraction of seats usually left unseated --/
def unseated_fraction : ℚ := 1 / 10

/-- The number of seats usually taken --/
def seats_taken : ℕ := 135

/-- Theorem stating the number of tables in the cafeteria --/
theorem cafeteria_tables :
  num_tables = seats_taken / (seats_per_table * (1 - unseated_fraction)) := by
  sorry

end cafeteria_tables_l3778_377849


namespace cubic_fraction_factorization_l3778_377817

theorem cubic_fraction_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end cubic_fraction_factorization_l3778_377817


namespace system_equations_and_inequality_l3778_377800

theorem system_equations_and_inequality (a x y : ℝ) : 
  x - y = 1 + 3 * a →
  x + y = -7 - a →
  x ≤ 0 →
  y < 0 →
  (-2 < a ∧ a ≤ 3) →
  (∀ x, 2 * a * x + x > 2 * a + 1 ↔ x < 1) →
  a = -1 := by sorry

end system_equations_and_inequality_l3778_377800


namespace log_sin_in_terms_of_m_n_l3778_377876

open Real

theorem log_sin_in_terms_of_m_n (α m n : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2)
  (h3 : log (1 + cos α) = m)
  (h4 : log (1 / (1 - cos α)) = n) :
  log (sin α) = (1 / 2) * (m - 1 / n) := by
  sorry

end log_sin_in_terms_of_m_n_l3778_377876


namespace arithmetic_geometric_mean_digit_swap_l3778_377808

theorem arithmetic_geometric_mean_digit_swap : ∃ (x₁ x₂ : ℕ), 
  x₁ ≠ x₂ ∧
  (let A := (x₁ + x₂) / 2
   let G := Int.sqrt (x₁ * x₂)
   10 ≤ A ∧ A < 100 ∧
   10 ≤ G ∧ G < 100 ∧
   ((A / 10 = G % 10 ∧ A % 10 = G / 10) ∨
    (A % 10 = G / 10 ∧ A / 10 = G % 10)) ∧
   x₁ = 98 ∧
   x₂ = 32) :=
by
  sorry

#eval (98 + 32) / 2  -- Expected output: 65
#eval Int.sqrt (98 * 32)  -- Expected output: 56

end arithmetic_geometric_mean_digit_swap_l3778_377808


namespace pens_given_to_sharon_l3778_377806

/-- The number of pens given to Sharon in a pen collection scenario --/
theorem pens_given_to_sharon (initial_pens : ℕ) (mike_pens : ℕ) (final_pens : ℕ) : 
  initial_pens = 25 →
  mike_pens = 22 →
  final_pens = 75 →
  (initial_pens + mike_pens) * 2 - final_pens = 19 :=
by
  sorry

end pens_given_to_sharon_l3778_377806


namespace prime_divides_product_l3778_377801

theorem prime_divides_product (p a b : ℕ) : 
  Prime p → (p ∣ (a * b)) → (p ∣ a) ∨ (p ∣ b) := by
  sorry

end prime_divides_product_l3778_377801


namespace walk_distance_l3778_377870

/-- The total distance walked by Erin and Susan -/
def total_distance (susan_distance erin_distance : ℕ) : ℕ :=
  susan_distance + erin_distance

/-- Theorem stating the total distance walked by Erin and Susan -/
theorem walk_distance :
  ∀ (susan_distance erin_distance : ℕ),
    susan_distance = 9 →
    erin_distance = susan_distance - 3 →
    total_distance susan_distance erin_distance = 15 := by
  sorry

end walk_distance_l3778_377870


namespace gasoline_reduction_l3778_377883

theorem gasoline_reduction (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 1.2 * P
  let new_total_cost := 1.14 * (P * Q)
  let new_quantity := new_total_cost / new_price
  (Q - new_quantity) / Q = 0.05 := by
sorry

end gasoline_reduction_l3778_377883


namespace hyperbola_sum_l3778_377834

/-- Given a hyperbola with center (1, 0), one focus at (1 + √41, 0), and one vertex at (-2, 0),
    prove that h + k + a + b = 1 + 0 + 3 + 4√2, where (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1
    is the equation of the hyperbola. -/
theorem hyperbola_sum (h k a b : ℝ) : 
  (1 : ℝ) = h ∧ (0 : ℝ) = k ∧  -- center at (1, 0)
  (1 + Real.sqrt 41 : ℝ) = h + Real.sqrt (c^2) ∧ -- focus at (1 + √41, 0)
  (-2 : ℝ) = h - a ∧ -- vertex at (-2, 0)
  (∀ x y : ℝ, (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1) → -- equation of hyperbola
  h + k + a + b = 1 + 0 + 3 + 4 * Real.sqrt 2 :=
by sorry

end hyperbola_sum_l3778_377834


namespace derivative_at_zero_l3778_377863

/-- Given a function f(x) = e^x + sin x - cos x, prove that its derivative at x = 0 is 2 -/
theorem derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp x + Real.sin x - Real.cos x) :
  deriv f 0 = 2 := by
  sorry

end derivative_at_zero_l3778_377863


namespace population_increase_l3778_377816

theorem population_increase (x : ℝ) : 
  (3 + 3 * x / 100 = 12) → x = 300 := by
  sorry

end population_increase_l3778_377816


namespace christmas_book_sales_l3778_377821

/-- Given a ratio of books to bookmarks and the number of bookmarks sold,
    calculate the number of books sold. -/
def books_sold (book_ratio : ℕ) (bookmark_ratio : ℕ) (bookmarks_sold : ℕ) : ℕ :=
  (book_ratio * bookmarks_sold) / bookmark_ratio

/-- Theorem stating that given the specific ratio and number of bookmarks sold,
    the number of books sold is 72. -/
theorem christmas_book_sales : books_sold 9 2 16 = 72 := by
  sorry

end christmas_book_sales_l3778_377821


namespace revenue_decrease_l3778_377823

/-- Proves that a 43.529411764705884% decrease to $48.0 billion results in an original revenue of $85.0 billion -/
theorem revenue_decrease (current_revenue : ℝ) (decrease_percentage : ℝ) (original_revenue : ℝ) :
  current_revenue = 48.0 ∧
  decrease_percentage = 43.529411764705884 ∧
  current_revenue = original_revenue * (1 - decrease_percentage / 100) →
  original_revenue = 85.0 := by
sorry

end revenue_decrease_l3778_377823


namespace bryan_pushups_l3778_377803

theorem bryan_pushups (planned_sets : ℕ) (pushups_per_set : ℕ) (actual_total : ℕ)
  (h1 : planned_sets = 3)
  (h2 : pushups_per_set = 15)
  (h3 : actual_total = 40) :
  planned_sets * pushups_per_set - actual_total = 5 := by
  sorry

end bryan_pushups_l3778_377803


namespace solve_equation_l3778_377853

-- Define the @ operation
def at_op (a b : ℝ) : ℝ := (a + 5) * b

-- State the theorem
theorem solve_equation (x : ℝ) (h : at_op x 1.3 = 11.05) : x = 3.5 := by
  sorry

end solve_equation_l3778_377853


namespace units_digit_sum_factorials_l3778_377812

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials :
  unitsDigit (sumFactorials 100) = unitsDigit (sumFactorials 4) := by
  sorry

end units_digit_sum_factorials_l3778_377812


namespace scaling_transform_line_l3778_377805

/-- Scaling transformation that maps (x, y) to (x', y') -/
def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (3 * x, 2 * y)

theorem scaling_transform_line : 
  ∀ (x y : ℝ), x + y = 1 → 
  let (x', y') := scaling_transform x y
  2 * x' + 3 * y' = 6 := by
sorry

end scaling_transform_line_l3778_377805


namespace suitcase_electronics_weight_l3778_377862

/-- Proves that the weight of electronics is 12 pounds given the conditions of the suitcase problem -/
theorem suitcase_electronics_weight 
  (B C E : ℝ) -- Weights of books, clothes, and electronics
  (h1 : B / C = 7 / 4) -- Initial ratio of books to clothes
  (h2 : C / E = 4 / 3) -- Initial ratio of clothes to electronics
  (h3 : B / (C - 8) = 2 * (B / C)) -- Ratio doubles after removing 8 pounds of clothes
  : E = 12 := by
  sorry

end suitcase_electronics_weight_l3778_377862


namespace point_b_coordinates_l3778_377841

/-- Given point A and vector a, if vector AB = 2a, then point B has specific coordinates -/
theorem point_b_coordinates (A B : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (1, -3) →
  a = (3, 4) →
  B - A = 2 • a →
  B = (7, 5) := by
  sorry

end point_b_coordinates_l3778_377841


namespace boat_speed_solution_l3778_377895

def boat_problem (downstream_time upstream_time stream_speed : ℝ) : Prop :=
  downstream_time > 0 ∧ 
  upstream_time > 0 ∧ 
  stream_speed > 0 ∧
  ∃ (distance boat_speed : ℝ),
    distance > 0 ∧
    boat_speed > stream_speed ∧
    distance = (boat_speed + stream_speed) * downstream_time ∧
    distance = (boat_speed - stream_speed) * upstream_time

theorem boat_speed_solution :
  boat_problem 1 1.5 3 →
  ∃ (distance boat_speed : ℝ),
    boat_speed = 15 ∧
    distance > 0 ∧
    boat_speed > 3 ∧
    distance = (boat_speed + 3) * 1 ∧
    distance = (boat_speed - 3) * 1.5 :=
by
  sorry

#check boat_speed_solution

end boat_speed_solution_l3778_377895


namespace gold_coin_value_l3778_377889

theorem gold_coin_value :
  let silver_coin_value : ℕ := 25
  let gold_coins : ℕ := 3
  let silver_coins : ℕ := 5
  let cash : ℕ := 30
  let total_value : ℕ := 305
  ∃ (gold_coin_value : ℕ),
    gold_coin_value * gold_coins + silver_coin_value * silver_coins + cash = total_value ∧
    gold_coin_value = 50 :=
by sorry

end gold_coin_value_l3778_377889


namespace remaining_customers_l3778_377894

/-- Given an initial number of customers and a number of customers who left,
    prove that the remaining number of customers is equal to the
    initial number minus the number who left. -/
theorem remaining_customers
  (initial : ℕ) (left : ℕ) (h : left ≤ initial) :
  initial - left = initial - left :=
by sorry

end remaining_customers_l3778_377894


namespace volume_common_part_equal_cones_l3778_377865

/-- Given two equal cones with common height and parallel bases, 
    the volume of their common part is 1/4 of the volume of each cone. -/
theorem volume_common_part_equal_cones (R h : ℝ) (hR : R > 0) (hh : h > 0) : 
  let V_cone := (1/3) * π * R^2 * h
  let V_common := (1/12) * π * R^2 * h
  V_common = (1/4) * V_cone := by
  sorry

end volume_common_part_equal_cones_l3778_377865


namespace light_flash_time_l3778_377882

/-- The time taken for a light to flash 600 times, given that it flashes every 6 seconds, is equal to 1 hour -/
theorem light_flash_time (flash_interval : ℕ) (total_flashes : ℕ) (seconds_per_hour : ℕ) :
  flash_interval = 6 →
  total_flashes = 600 →
  seconds_per_hour = 3600 →
  (flash_interval * total_flashes) / seconds_per_hour = 1 :=
by sorry

end light_flash_time_l3778_377882


namespace park_area_is_20000_l3778_377819

/-- Represents a rectangular park -/
structure RectangularPark where
  length : ℝ
  breadth : ℝ
  cyclingSpeed : ℝ
  cyclingTime : ℝ

/-- Calculates the area of a rectangular park -/
def parkArea (park : RectangularPark) : ℝ :=
  park.length * park.breadth

/-- Calculates the perimeter of a rectangular park -/
def parkPerimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.breadth)

/-- Theorem: Given the conditions, the area of the park is 20,000 square meters -/
theorem park_area_is_20000 (park : RectangularPark) 
    (h1 : park.length = park.breadth / 2)
    (h2 : park.cyclingSpeed = 6)  -- in km/hr
    (h3 : park.cyclingTime = 1/10)  -- 6 minutes in hours
    (h4 : parkPerimeter park = park.cyclingSpeed * park.cyclingTime * 1000) : 
    parkArea park = 20000 := by
  sorry


end park_area_is_20000_l3778_377819


namespace complement_of_M_in_U_l3778_377809

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x ∈ U | x^2 - 4*x + 3 = 0}

theorem complement_of_M_in_U : (U \ M) = {2, 4} := by sorry

end complement_of_M_in_U_l3778_377809


namespace fraction_subtraction_l3778_377861

theorem fraction_subtraction : (4 + 6 + 8) / (3 + 5 + 7) - (3 + 5 + 7) / (4 + 6 + 8) = 11 / 30 := by
  sorry

end fraction_subtraction_l3778_377861


namespace bridge_length_calculation_l3778_377897

/-- Calculates the length of a bridge given the train's length, speed, and time to cross. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 255 :=
by
  sorry

end bridge_length_calculation_l3778_377897


namespace rectangle_width_decrease_l3778_377827

/-- Theorem: Rectangle Width Decrease
Given a rectangle where:
- The length increases by 20%
- The area increases by 4%
Then the width must decrease by 40/3% (approximately 13.33%) -/
theorem rectangle_width_decrease (L W : ℝ) (L' W' : ℝ) (h1 : L' = 1.2 * L) (h2 : L' * W' = 1.04 * L * W) :
  W' = (1 - 40 / 300) * W :=
sorry

end rectangle_width_decrease_l3778_377827


namespace male_employees_count_l3778_377835

/-- Proves the number of male employees in a company given certain conditions --/
theorem male_employees_count :
  ∀ (m f : ℕ),
  (m : ℚ) / f = 7 / 8 →
  ((m + 3 : ℚ) / f = 8 / 9) →
  m = 189 := by
sorry

end male_employees_count_l3778_377835


namespace quadratic_equation_roots_l3778_377824

theorem quadratic_equation_roots (a b c : ℝ) (h1 : a = 1) (h2 : b = -4) (h3 : c = -5) :
  ∃ (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end quadratic_equation_roots_l3778_377824


namespace f_monotonic_decreasing_iff_a_in_range_l3778_377868

-- Define the function f(x) = ax|x-a|
def f (a : ℝ) (x : ℝ) : ℝ := a * x * abs (x - a)

-- Define the property of being monotonically decreasing on an interval
def monotonically_decreasing (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → g y < g x

-- State the theorem
theorem f_monotonic_decreasing_iff_a_in_range :
  ∀ a : ℝ, (monotonically_decreasing (f a) 1 (3/2)) ↔ 
    (a < 0 ∨ (3/2 ≤ a ∧ a ≤ 2)) :=
sorry

end f_monotonic_decreasing_iff_a_in_range_l3778_377868


namespace arithmetic_mean_geq_geometric_mean_l3778_377899

theorem arithmetic_mean_geq_geometric_mean {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end arithmetic_mean_geq_geometric_mean_l3778_377899


namespace town_population_growth_l3778_377820

/-- Represents the population of a town over time -/
structure TownPopulation where
  pop1991 : Nat
  pop2006 : Nat
  pop2016 : Nat

/-- Conditions for the town population -/
def ValidTownPopulation (t : TownPopulation) : Prop :=
  ∃ (n m k : Nat),
    t.pop1991 = n * n ∧
    t.pop2006 = t.pop1991 + 120 ∧
    t.pop2006 = m * m - 1 ∧
    t.pop2016 = t.pop2006 + 180 ∧
    t.pop2016 = k * k

/-- Calculate percent growth -/
def PercentGrowth (initial : Nat) (final : Nat) : ℚ :=
  (final - initial : ℚ) / initial * 100

/-- Main theorem stating the percent growth is 5% -/
theorem town_population_growth (t : TownPopulation) 
  (h : ValidTownPopulation t) : 
  PercentGrowth t.pop1991 t.pop2016 = 5 := by
  sorry

end town_population_growth_l3778_377820


namespace cat_max_distance_l3778_377869

/-- The maximum distance a cat can be from the origin, given it's tied to a post -/
theorem cat_max_distance (post_x post_y rope_length : ℝ) : 
  post_x = 6 → post_y = 8 → rope_length = 15 → 
  ∃ (max_distance : ℝ), max_distance = 25 ∧ 
  ∀ (cat_x cat_y : ℝ), 
    (cat_x - post_x)^2 + (cat_y - post_y)^2 ≤ rope_length^2 → 
    cat_x^2 + cat_y^2 ≤ max_distance^2 :=
by sorry

end cat_max_distance_l3778_377869


namespace external_equilaterals_centers_theorem_l3778_377826

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (base : Point)
  (apex : Point)

/-- Returns the center of an equilateral triangle -/
def centerOfEquilateral (t : EquilateralTriangle) : Point := sorry

/-- Returns the centroid of a triangle -/
def centroid (t : Triangle) : Point := sorry

/-- Constructs equilateral triangles on the sides of a given triangle -/
def constructExternalEquilaterals (t : Triangle) : 
  (EquilateralTriangle × EquilateralTriangle × EquilateralTriangle) := sorry

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (A B C : Point) : Prop := sorry

theorem external_equilaterals_centers_theorem (t : Triangle) :
  let (eqAB, eqBC, eqCA) := constructExternalEquilaterals t
  let centerAB := centerOfEquilateral eqAB
  let centerBC := centerOfEquilateral eqBC
  let centerCA := centerOfEquilateral eqCA
  isEquilateral centerAB centerBC centerCA ∧
  centroid (Triangle.mk centerAB centerBC centerCA) = centroid t := by sorry

end external_equilaterals_centers_theorem_l3778_377826


namespace two_faces_same_edge_count_l3778_377846

/-- A polyhedron with n faces, where each face has between 3 and n-1 edges. -/
structure Polyhedron (n : ℕ) where
  faces : Fin n → ℕ
  face_edge_count_lower_bound : ∀ i, faces i ≥ 3
  face_edge_count_upper_bound : ∀ i, faces i ≤ n - 1

/-- There exist at least two faces with the same number of edges in any polyhedron. -/
theorem two_faces_same_edge_count {n : ℕ} (h : n > 2) (P : Polyhedron n) :
  ∃ i j, i ≠ j ∧ P.faces i = P.faces j := by
  sorry

end two_faces_same_edge_count_l3778_377846


namespace max_salary_320000_l3778_377892

/-- Represents a baseball team with salary constraints -/
structure BaseballTeam where
  num_players : ℕ
  min_salary : ℕ
  total_salary_cap : ℕ

/-- Calculates the maximum possible salary for a single player in a baseball team -/
def max_single_player_salary (team : BaseballTeam) : ℕ :=
  team.total_salary_cap - (team.num_players - 1) * team.min_salary

/-- Theorem stating the maximum possible salary for a single player in a specific baseball team -/
theorem max_salary_320000 :
  let team : BaseballTeam := ⟨25, 20000, 800000⟩
  max_single_player_salary team = 320000 := by
  sorry

#eval max_single_player_salary ⟨25, 20000, 800000⟩

end max_salary_320000_l3778_377892


namespace condition_sufficient_not_necessary_l3778_377855

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define what it means for a sequence to be increasing
def IsIncreasing (a : Sequence) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

-- Define the condition a_{n+1} > |a_n|
def StrictlyGreaterThanAbs (a : Sequence) : Prop :=
  ∀ n : ℕ, a (n + 1) > |a n|

-- Theorem statement
theorem condition_sufficient_not_necessary :
  (∀ a : Sequence, StrictlyGreaterThanAbs a → IsIncreasing a) ∧
  (∃ a : Sequence, IsIncreasing a ∧ ¬StrictlyGreaterThanAbs a) :=
by sorry

end condition_sufficient_not_necessary_l3778_377855


namespace prob_ace_king_correct_l3778_377804

/-- A standard deck of cards. -/
structure Deck :=
  (cards : Fin 52)

/-- The probability of drawing an Ace first and a King second from a standard deck. -/
def prob_ace_then_king (d : Deck) : ℚ :=
  4 / 663

/-- Theorem: The probability of drawing an Ace first and a King second from a standard 52-card deck is 4/663. -/
theorem prob_ace_king_correct (d : Deck) : prob_ace_then_king d = 4 / 663 := by
  sorry

end prob_ace_king_correct_l3778_377804


namespace a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3778_377884

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  (∀ a : ℝ, a = 1 → |a| = 1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ |a| = 1) := by
  sorry

end a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3778_377884


namespace modulus_of_specific_complex_number_l3778_377881

theorem modulus_of_specific_complex_number :
  let i : ℂ := Complex.I
  let z : ℂ := 2 * i + 2 / (1 + i)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_specific_complex_number_l3778_377881


namespace total_cost_after_discounts_and_cashback_l3778_377879

/-- The total cost of an iPhone 12 and an iWatch after discounts and cashback -/
theorem total_cost_after_discounts_and_cashback :
  let iphone_price : ℚ := 800
  let iwatch_price : ℚ := 300
  let iphone_discount : ℚ := 15 / 100
  let iwatch_discount : ℚ := 10 / 100
  let cashback_rate : ℚ := 2 / 100
  let iphone_discounted := iphone_price * (1 - iphone_discount)
  let iwatch_discounted := iwatch_price * (1 - iwatch_discount)
  let total_before_cashback := iphone_discounted + iwatch_discounted
  let cashback_amount := total_before_cashback * cashback_rate
  let final_cost := total_before_cashback - cashback_amount
  final_cost = 931 :=
by sorry

end total_cost_after_discounts_and_cashback_l3778_377879


namespace commercial_break_duration_l3778_377872

theorem commercial_break_duration :
  let five_minute_commercials : ℕ := 3
  let two_minute_commercials : ℕ := 11
  let five_minute_duration : ℕ := 5
  let two_minute_duration : ℕ := 2
  (five_minute_commercials * five_minute_duration + two_minute_commercials * two_minute_duration : ℕ) = 37 :=
by sorry

end commercial_break_duration_l3778_377872


namespace puppies_feeding_theorem_l3778_377830

/-- Given the number of formula portions, puppies, and days, calculate the number of feedings per day. -/
def feedings_per_day (portions : ℕ) (puppies : ℕ) (days : ℕ) : ℚ :=
  (portions : ℚ) / (puppies * days)

/-- Theorem stating that given 105 portions of formula for 7 puppies over 5 days, the number of feedings per day is equal to 3. -/
theorem puppies_feeding_theorem :
  feedings_per_day 105 7 5 = 3 := by
  sorry

end puppies_feeding_theorem_l3778_377830


namespace new_player_weight_l3778_377844

/-- Represents a basketball team --/
structure BasketballTeam where
  players : ℕ
  averageWeight : ℝ
  totalWeight : ℝ

/-- Calculates the total weight of a team --/
def totalWeight (team : BasketballTeam) : ℝ :=
  team.players * team.averageWeight

/-- Represents the change in team composition --/
structure TeamChange where
  oldTeam : BasketballTeam
  newTeam : BasketballTeam
  replacedWeight1 : ℝ
  replacedWeight2 : ℝ
  newPlayerWeight : ℝ

/-- Theorem stating the weight of the new player --/
theorem new_player_weight (change : TeamChange) 
  (h1 : change.oldTeam.players = 12)
  (h2 : change.oldTeam.averageWeight = 80)
  (h3 : change.newTeam.players = change.oldTeam.players)
  (h4 : change.newTeam.averageWeight = change.oldTeam.averageWeight + 2.5)
  (h5 : change.replacedWeight1 = 65)
  (h6 : change.replacedWeight2 = 75) :
  change.newPlayerWeight = 170 := by
  sorry

end new_player_weight_l3778_377844


namespace max_area_right_triangle_l3778_377867

/-- The maximum area of a right-angled triangle with perimeter √2 + 1 is 1/4 -/
theorem max_area_right_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a^2 + b^2 = c^2 → 
  a + b + c = Real.sqrt 2 + 1 → 
  (1/2 * a * b) ≤ 1/4 := by
  sorry

end max_area_right_triangle_l3778_377867


namespace quadratic_equations_solutions_l3778_377837

theorem quadratic_equations_solutions :
  (∀ x : ℝ, 2 * x^2 + 4 * x + 1 = 0 ↔ x = -1 + Real.sqrt 2 / 2 ∨ x = -1 - Real.sqrt 2 / 2) ∧
  (∀ x : ℝ, x^2 + 6 * x = 5 ↔ x = -3 + Real.sqrt 14 ∨ x = -3 - Real.sqrt 14) :=
by sorry

end quadratic_equations_solutions_l3778_377837


namespace fraction_evaluation_l3778_377877

theorem fraction_evaluation : (1/4 - 1/6) / (1/3 - 1/4) = 1 := by
  sorry

end fraction_evaluation_l3778_377877


namespace total_riding_time_two_weeks_l3778_377850

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the riding time in minutes for a given day -/
def ridingTime (d : Day) : ℕ :=
  match d with
  | Day.Monday    => 60
  | Day.Tuesday   => 30
  | Day.Wednesday => 60
  | Day.Thursday  => 30
  | Day.Friday    => 60
  | Day.Saturday  => 120
  | Day.Sunday    => 0

/-- Calculates the total riding time for one week in minutes -/
def weeklyRidingTime : ℕ :=
  (ridingTime Day.Monday) + (ridingTime Day.Tuesday) + (ridingTime Day.Wednesday) +
  (ridingTime Day.Thursday) + (ridingTime Day.Friday) + (ridingTime Day.Saturday) +
  (ridingTime Day.Sunday)

/-- Theorem: Bethany rides for 12 hours in total over a 2-week period -/
theorem total_riding_time_two_weeks :
  (2 * weeklyRidingTime) / 60 = 12 := by sorry

end total_riding_time_two_weeks_l3778_377850


namespace inequality_condition_l3778_377848

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x + a / x ≥ 2) ↔ a ≥ 1 := by sorry

end inequality_condition_l3778_377848


namespace triangle_existence_l3778_377847

theorem triangle_existence (x : ℕ) : 
  (∃ (a b c : ℝ), a = 8 ∧ b = 12 ∧ c = x^3 + 1 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ (x = 2 ∨ x = 3) :=
sorry

end triangle_existence_l3778_377847


namespace sum_of_squares_l3778_377840

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 35)
  (h2 : x^2 * y + x * y^2 = 210) : 
  x^2 + y^2 = 154 := by
  sorry

end sum_of_squares_l3778_377840


namespace cube_divisibility_l3778_377880

theorem cube_divisibility (k : ℕ) (n : ℕ) : 
  (k ≥ 30) → 
  (∀ m : ℕ, m ≥ 30 → m < k → ¬(∃ p : ℕ, m^3 = p * n)) →
  (∃ q : ℕ, k^3 = q * n) →
  n = 27000 := by
sorry

end cube_divisibility_l3778_377880


namespace m_equals_three_l3778_377886

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

/-- Definition of the complex number z in terms of m -/
def z (m : ℝ) : ℂ := m^2 * (1 + Complex.I) - m * (3 + 6 * Complex.I)

/-- Theorem: If z(m) is pure imaginary, then m = 3 -/
theorem m_equals_three (h : isPureImaginary (z m)) : m = 3 := by
  sorry

end m_equals_three_l3778_377886


namespace pure_imaginary_complex_number_l3778_377825

theorem pure_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = 1 := by
  sorry

end pure_imaginary_complex_number_l3778_377825


namespace harry_worked_35_hours_l3778_377828

/-- Represents the pay structure and hours worked for Harry and James -/
structure PayStructure where
  x : ℝ  -- Base hourly rate
  james_overtime_rate : ℝ  -- James' overtime rate as a multiple of x
  harry_hours : ℕ  -- Total hours Harry worked
  harry_overtime : ℕ  -- Hours Harry worked beyond 21
  james_hours : ℕ  -- Total hours James worked
  james_overtime : ℕ  -- Hours James worked beyond 40

/-- Calculates Harry's total pay -/
def harry_pay (p : PayStructure) : ℝ :=
  21 * p.x + p.harry_overtime * (1.5 * p.x)

/-- Calculates James' total pay -/
def james_pay (p : PayStructure) : ℝ :=
  40 * p.x + p.james_overtime * (p.james_overtime_rate * p.x)

/-- Theorem stating that Harry worked 35 hours given the problem conditions -/
theorem harry_worked_35_hours :
  ∀ (p : PayStructure),
    p.james_hours = 41 →
    p.james_overtime = 1 →
    p.harry_hours = p.harry_overtime + 21 →
    harry_pay p = james_pay p →
    p.harry_hours = 35 := by
  sorry


end harry_worked_35_hours_l3778_377828


namespace soda_price_calculation_l3778_377864

/-- The cost of a burger in cents -/
def burger_cost : ℕ := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℕ := sorry

/-- The cost of a side dish in cents -/
def side_dish_cost : ℕ := 30

theorem soda_price_calculation :
  (3 * burger_cost + 2 * soda_cost + side_dish_cost = 510) →
  (2 * burger_cost + 3 * soda_cost = 540) →
  soda_cost = 132 := by sorry

end soda_price_calculation_l3778_377864


namespace probability_one_of_each_l3778_377838

def forks : ℕ := 8
def spoons : ℕ := 9
def knives : ℕ := 10
def teaspoons : ℕ := 7

def total_silverware : ℕ := forks + spoons + knives + teaspoons

theorem probability_one_of_each (forks spoons knives teaspoons : ℕ) 
  (h1 : forks = 8) (h2 : spoons = 9) (h3 : knives = 10) (h4 : teaspoons = 7) :
  (forks * spoons * knives * teaspoons : ℚ) / (Nat.choose total_silverware 4) = 40 / 367 := by
  sorry

end probability_one_of_each_l3778_377838


namespace sports_club_membership_l3778_377811

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ)
  (h1 : total = 30)
  (h2 : badminton = 16)
  (h3 : tennis = 19)
  (h4 : both = 7) :
  total - (badminton + tennis - both) = 2 := by
  sorry

end sports_club_membership_l3778_377811


namespace line_point_k_value_l3778_377845

/-- A line contains the points (8,10), (0,k), and (-8,3). This theorem proves that k = 13/2. -/
theorem line_point_k_value : 
  ∀ (k : ℚ), 
  (∃ (line : Set (ℚ × ℚ)), 
    (8, 10) ∈ line ∧ 
    (0, k) ∈ line ∧ 
    (-8, 3) ∈ line ∧ 
    (∀ (x y z : ℚ × ℚ), x ∈ line → y ∈ line → z ∈ line → 
      (x.2 - y.2) * (y.1 - z.1) = (y.2 - z.2) * (x.1 - y.1))) → 
  k = 13 / 2 := by
sorry

end line_point_k_value_l3778_377845


namespace age_ratio_problem_l3778_377854

theorem age_ratio_problem (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 24) :
  (x : ℚ) / y = 7 / 3 := by
  sorry

end age_ratio_problem_l3778_377854


namespace arithmetic_mean_change_l3778_377891

theorem arithmetic_mean_change (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 10 →
  (b + c + d) / 3 = 11 →
  (a + c + d) / 3 = 12 →
  (a + b + d) / 3 = 13 →
  (a + b + c) / 3 = 4 :=
by sorry

end arithmetic_mean_change_l3778_377891


namespace symmetry_condition_implies_symmetric_about_one_l3778_377878

/-- A function f: ℝ → ℝ is symmetric about x = 1 if f(1 + x) = f(1 - x) for all x ∈ ℝ -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + x) = f (1 - x)

/-- Main theorem: If f(x) - f(2 - x) = 0 for all x, then f is symmetric about x = 1 -/
theorem symmetry_condition_implies_symmetric_about_one (f : ℝ → ℝ) 
    (h : ∀ x, f x - f (2 - x) = 0) : SymmetricAboutOne f := by
  sorry

#check symmetry_condition_implies_symmetric_about_one

end symmetry_condition_implies_symmetric_about_one_l3778_377878


namespace ellipse_parabola_tangent_lines_l3778_377874

/-- Given an ellipse and a parabola with specific properties, prove the equation of the parabola and its tangent lines. -/
theorem ellipse_parabola_tangent_lines :
  ∀ (b : ℝ) (p : ℝ),
  0 < b → b < 2 → p > 0 →
  (∀ (x y : ℝ), x^2 / 4 + y^2 / b^2 = 1 → (x^2 + y^2) / 4 = 3 / 4) →
  (∀ (x y : ℝ), x^2 = 2 * p * y) →
  (∃ (x₀ y₀ : ℝ), x₀^2 / 4 + y₀^2 / b^2 = 1 ∧ x₀^2 = 2 * p * y₀ ∧ (x₀ = 0 ∨ y₀ = 1 ∨ y₀ = -1)) →
  (∀ (x y : ℝ), x^2 = 4 * y) ∧
  (∀ (x y : ℝ), (y = 0 ∨ x + y + 1 = 0) → 
    (x + 1)^2 = 4 * y ∧ (x + 1 = -1 → y = 0)) :=
by sorry

end ellipse_parabola_tangent_lines_l3778_377874


namespace bakers_pastry_problem_l3778_377871

/-- Baker's pastry problem -/
theorem bakers_pastry_problem 
  (total_cakes : ℕ) 
  (total_pastries : ℕ) 
  (sold_pastries : ℕ) 
  (remaining_pastries : ℕ) 
  (h1 : total_cakes = 7)
  (h2 : total_pastries = 148)
  (h3 : sold_pastries = 103)
  (h4 : remaining_pastries = 45)
  (h5 : total_pastries = sold_pastries + remaining_pastries) :
  ¬∃! sold_cakes : ℕ, sold_cakes ≤ total_cakes :=
sorry

end bakers_pastry_problem_l3778_377871


namespace sum_of_integers_l3778_377890

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_integers_l3778_377890


namespace amanda_works_ten_hours_l3778_377839

/-- Amanda's work scenario -/
def amanda_scenario (hourly_rate : ℝ) (withheld_pay : ℝ) (hours_worked : ℝ) : Prop :=
  hourly_rate = 50 ∧
  withheld_pay = 400 ∧
  withheld_pay = 0.8 * (hourly_rate * hours_worked)

/-- Theorem: Amanda works 10 hours per day -/
theorem amanda_works_ten_hours :
  ∃ (hourly_rate withheld_pay hours_worked : ℝ),
    amanda_scenario hourly_rate withheld_pay hours_worked ∧
    hours_worked = 10 :=
sorry

end amanda_works_ten_hours_l3778_377839


namespace solve_annas_candy_problem_l3778_377885

def annas_candy_problem (initial_money : ℚ) 
                         (gum_price : ℚ) 
                         (gum_quantity : ℕ) 
                         (chocolate_price : ℚ) 
                         (chocolate_quantity : ℕ) 
                         (candy_cane_price : ℚ) 
                         (money_left : ℚ) : Prop :=
  let total_spent := gum_price * gum_quantity + chocolate_price * chocolate_quantity
  let money_for_candy_canes := initial_money - total_spent - money_left
  let candy_canes_bought := money_for_candy_canes / candy_cane_price
  candy_canes_bought = 2

theorem solve_annas_candy_problem : 
  annas_candy_problem 10 1 3 1 5 (1/2) 1 := by
  sorry

end solve_annas_candy_problem_l3778_377885


namespace value_of_y_l3778_377859

theorem value_of_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end value_of_y_l3778_377859


namespace employee_income_change_l3778_377829

theorem employee_income_change 
  (payment_increase : Real) 
  (time_decrease : Real) : 
  payment_increase = 0.3333 → 
  time_decrease = 0.3333 → 
  let new_payment := 1 + payment_increase
  let new_time := 1 - time_decrease
  let income_change := new_payment * new_time - 1
  income_change = -0.1111 := by sorry

end employee_income_change_l3778_377829


namespace juice_vitamin_c_content_l3778_377815

/-- Vitamin C content in milligrams for different juice combinations -/
theorem juice_vitamin_c_content 
  (apple orange grapefruit : ℝ) 
  (h1 : apple + orange + grapefruit = 275) 
  (h2 : 2 * apple + 3 * orange + 4 * grapefruit = 683) : 
  orange + 2 * grapefruit = 133 := by
sorry

end juice_vitamin_c_content_l3778_377815


namespace points_per_round_l3778_377810

/-- Given a card game where:
  * Jane ends up with 60 points
  * She lost 20 points
  * She played 8 rounds
  Prove that the number of points awarded for winning one round is 10. -/
theorem points_per_round (final_points : ℕ) (lost_points : ℕ) (rounds : ℕ) :
  final_points = 60 →
  lost_points = 20 →
  rounds = 8 →
  (final_points + lost_points) / rounds = 10 :=
by sorry

end points_per_round_l3778_377810


namespace quadratic_equation_solutions_linear_equation_solutions_l3778_377856

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ 2*x^2 - 6*x - 5
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 19) / 2 ∧ 
              x₂ = (3 - Real.sqrt 19) / 2 ∧ 
              f x₁ = 0 ∧ f x₂ = 0 :=
sorry

theorem linear_equation_solutions :
  let g : ℝ → ℝ := λ x ↦ 3*x*(4-x) - 2*(x-4)
  ∃ x₁ x₂ : ℝ, x₁ = 4 ∧ 
              x₂ = -2/3 ∧ 
              g x₁ = 0 ∧ g x₂ = 0 :=
sorry

end quadratic_equation_solutions_linear_equation_solutions_l3778_377856


namespace certain_number_problem_l3778_377852

theorem certain_number_problem (x : ℝ) : 
  ((x + 20) * 2) / 2 - 2 = 88 / 2 → x = 26 := by
  sorry

end certain_number_problem_l3778_377852


namespace baker_cakes_l3778_377888

/-- The number of cakes Baker made initially -/
def initial_cakes : ℕ := sorry

/-- The number of cakes Baker's friend bought -/
def friend_bought : ℕ := 140

/-- The number of cakes Baker still has -/
def remaining_cakes : ℕ := 15

/-- Theorem stating that the initial number of cakes is 155 -/
theorem baker_cakes : initial_cakes = friend_bought + remaining_cakes := by sorry

end baker_cakes_l3778_377888


namespace gala_trees_count_l3778_377836

theorem gala_trees_count (total : ℕ) (fuji gala honeycrisp : ℕ) : 
  total = fuji + gala + honeycrisp →
  fuji = (2 * total) / 3 →
  honeycrisp = total / 6 →
  fuji + (125 * fuji) / 1000 + (75 * fuji) / 1000 = 315 →
  gala = 66 := by
  sorry

end gala_trees_count_l3778_377836


namespace unique_solution_absolute_value_system_l3778_377802

theorem unique_solution_absolute_value_system :
  ∃! (x y : ℝ), 
    (abs (x + y) + abs (1 - x) = 6) ∧
    (abs (x + y + 1) + abs (1 - y) = 4) :=
by
  -- The proof goes here
  sorry

end unique_solution_absolute_value_system_l3778_377802


namespace cubic_equation_solution_l3778_377833

theorem cubic_equation_solution : 
  ∃ (a : ℝ), (a^3 - 4*a^2 + 7*a - 28 = 0) ∧ 
  (∀ x : ℝ, x^3 - 4*x^2 + 7*x - 28 = 0 → x ≤ a) →
  2*a + 0 = 8 := by
  sorry

end cubic_equation_solution_l3778_377833


namespace billy_ate_twenty_apples_l3778_377893

/-- The number of apples Billy ate on each day of the week --/
structure BillyApples where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Billy's apple consumption --/
def billyConditions (b : BillyApples) : Prop :=
  b.monday = 2 ∧
  b.tuesday = 2 * b.monday ∧
  b.wednesday = 9 ∧
  b.thursday = 4 * b.friday ∧
  b.friday = b.monday / 2

/-- The total number of apples Billy ate in the week --/
def totalApples (b : BillyApples) : ℕ :=
  b.monday + b.tuesday + b.wednesday + b.thursday + b.friday

/-- Theorem stating that Billy ate 20 apples in total --/
theorem billy_ate_twenty_apples :
  ∃ b : BillyApples, billyConditions b ∧ totalApples b = 20 := by
  sorry


end billy_ate_twenty_apples_l3778_377893


namespace sqrt_288_simplification_l3778_377866

theorem sqrt_288_simplification : Real.sqrt 288 = 12 * Real.sqrt 2 := by
  sorry

end sqrt_288_simplification_l3778_377866


namespace area_between_quartic_and_line_l3778_377875

/-- The area between a quartic function and a line that touch at two points -/
theorem area_between_quartic_and_line 
  (a b c d e p q α β : ℝ) 
  (ha : a ≠ 0) 
  (hαβ : α < β) : 
  let f := fun (x : ℝ) ↦ a * x^4 + b * x^3 + c * x^2 + d * x + e
  let g := fun (x : ℝ) ↦ p * x + q
  (∃ (x : ℝ), x = α ∨ x = β → f x = g x ∧ (deriv f) x = (deriv g) x) →
  ∫ x in α..β, |f x - g x| = a * (β - α)^5 / 30 := by
sorry

end area_between_quartic_and_line_l3778_377875


namespace magnitude_of_complex_power_l3778_377860

theorem magnitude_of_complex_power (z : ℂ) :
  z = (4:ℝ)/7 + (3:ℝ)/7 * Complex.I →
  Complex.abs (z^8) = 390625/5764801 := by
sorry

end magnitude_of_complex_power_l3778_377860


namespace inverse_sum_product_l3778_377887

theorem inverse_sum_product (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : 3*a + b/3 ≠ 0) :
  (3*a + b/3)⁻¹ * ((3*a)⁻¹ + (b/3)⁻¹) = (a*b)⁻¹ := by
  sorry

end inverse_sum_product_l3778_377887


namespace card_distribution_implies_square_l3778_377896

theorem card_distribution_implies_square (n : ℕ) (m : ℕ) (h_n : n ≥ 3) 
  (h_m : m = n * (n - 1) / 2) (h_m_even : Even m) 
  (a : Fin n → ℕ) (h_a_range : ∀ i, 1 ≤ a i ∧ a i ≤ m) 
  (h_a_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_sums_distinct : ∀ i j k l, (i ≠ j ∧ k ≠ l) → (i, j) ≠ (k, l) → 
    (a i + a j) % m ≠ (a k + a l) % m) :
  ∃ k : ℕ, n = k^2 := by
  sorry

end card_distribution_implies_square_l3778_377896


namespace geometric_sequence_common_ratio_l3778_377814

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_prod : a 1 * a 5 = 4)
  (h_a4 : a 4 = 1) :
  ∃ q : ℝ, q = 1/2 ∧ ∀ n : ℕ, a (n + 1) = a n * q := by
  sorry

end geometric_sequence_common_ratio_l3778_377814


namespace lily_score_l3778_377807

/-- Represents the score for hitting a specific ring -/
structure RingScore where
  inner : ℕ
  middle : ℕ
  outer : ℕ

/-- Represents the number of hits for each ring -/
structure Hits where
  inner : ℕ
  middle : ℕ
  outer : ℕ

/-- Calculates the total score given ring scores and hits -/
def totalScore (rs : RingScore) (h : Hits) : ℕ :=
  rs.inner * h.inner + rs.middle * h.middle + rs.outer * h.outer

theorem lily_score 
  (rs : RingScore) 
  (tom_hits john_hits : Hits) 
  (h1 : tom_hits.inner + tom_hits.middle + tom_hits.outer = 6)
  (h2 : john_hits.inner + john_hits.middle + john_hits.outer = 6)
  (h3 : totalScore rs tom_hits = 46)
  (h4 : totalScore rs john_hits = 34)
  (h5 : totalScore rs { inner := 4, middle := 4, outer := 4 } = 80) :
  totalScore rs { inner := 2, middle := 2, outer := 2 } = 40 := by
  sorry

#check lily_score

end lily_score_l3778_377807


namespace consecutive_integers_around_sqrt28_l3778_377842

theorem consecutive_integers_around_sqrt28 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 28) → (Real.sqrt 28 < b) → (a + b = 11) := by
  sorry

end consecutive_integers_around_sqrt28_l3778_377842


namespace triangles_not_always_congruent_l3778_377843

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ

-- Define the condition for the theorem
def satisfies_condition (t1 t2 : Triangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ 
   ((t1.a < t1.b ∧ t1.angle_A = t2.angle_A) ∨ 
    (t1.b < t1.a ∧ t1.angle_B = t2.angle_B)))

-- Define triangle congruence
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c ∧
  t1.angle_A = t2.angle_A ∧ t1.angle_B = t2.angle_B ∧ t1.angle_C = t2.angle_C

-- Theorem statement
theorem triangles_not_always_congruent :
  ∃ (t1 t2 : Triangle), satisfies_condition t1 t2 ∧ ¬(congruent t1 t2) :=
sorry

end triangles_not_always_congruent_l3778_377843


namespace correct_num_double_burgers_l3778_377858

/-- Represents the number of double burgers Caleb bought. -/
def num_double_burgers : ℕ := 37

/-- Represents the number of single burgers Caleb bought. -/
def num_single_burgers : ℕ := 50 - num_double_burgers

/-- The total cost of all burgers in cents. -/
def total_cost : ℕ := 6850

/-- The cost of a single burger in cents. -/
def single_burger_cost : ℕ := 100

/-- The cost of a double burger in cents. -/
def double_burger_cost : ℕ := 150

/-- The total number of burgers. -/
def total_burgers : ℕ := 50

theorem correct_num_double_burgers :
  num_single_burgers * single_burger_cost + num_double_burgers * double_burger_cost = total_cost ∧
  num_single_burgers + num_double_burgers = total_burgers :=
by sorry

end correct_num_double_burgers_l3778_377858


namespace angle_in_specific_pyramid_l3778_377813

/-- A triangular pyramid with specific properties -/
structure TriangularPyramid where
  AB : ℝ
  CD : ℝ
  distance : ℝ
  volume : ℝ

/-- The angle between two lines in a triangular pyramid -/
def angle_between_lines (p : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating the angle between AB and CD in the specific triangular pyramid -/
theorem angle_in_specific_pyramid :
  let p : TriangularPyramid := {
    AB := 8,
    CD := 12,
    distance := 6,
    volume := 48
  }
  angle_between_lines p = 30 * π / 180 :=
sorry

end angle_in_specific_pyramid_l3778_377813


namespace fourth_root_sum_of_fourth_powers_l3778_377898

theorem fourth_root_sum_of_fourth_powers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ c : ℝ, c = (a^4 + b^4)^(1/4) :=
by sorry

end fourth_root_sum_of_fourth_powers_l3778_377898


namespace negative_difference_equals_reversed_difference_l3778_377831

theorem negative_difference_equals_reversed_difference (a b : ℝ) : 
  -(a - b) = b - a := by sorry

end negative_difference_equals_reversed_difference_l3778_377831


namespace intersection_of_A_and_B_l3778_377873

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | (x - 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l3778_377873


namespace promotional_activity_choices_l3778_377822

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose volunteers for the promotional activity -/
def chooseVolunteers (totalVolunteers boyCount girlCount chosenCount : ℕ) : ℕ :=
  choose boyCount 3 * choose girlCount 1 + choose boyCount 2 * choose girlCount 2

theorem promotional_activity_choices :
  chooseVolunteers 6 4 2 4 = 14 := by sorry

end promotional_activity_choices_l3778_377822


namespace unique_quaternary_polynomial_l3778_377857

/-- A polynomial with coefficients in {0, 1, 2, 3} -/
def QuaternaryPolynomial := List (Fin 4)

/-- Evaluate a quaternary polynomial at x = 2 -/
def evalAt2 (p : QuaternaryPolynomial) : ℕ :=
  p.enum.foldl (fun acc (i, coef) => acc + coef.val * 2^i) 0

theorem unique_quaternary_polynomial (n : ℕ) (hn : n > 0) :
  ∃! p : QuaternaryPolynomial, evalAt2 p = n := by sorry

end unique_quaternary_polynomial_l3778_377857
