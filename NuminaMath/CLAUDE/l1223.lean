import Mathlib

namespace NUMINAMATH_CALUDE_corner_cut_pentagon_area_l1223_122366

/-- A pentagon formed by cutting a triangular corner from a rectangular sheet. -/
structure CornerCutPentagon where
  sides : Finset ℝ
  is_valid : sides = {14, 21, 22, 28, 35}

/-- The area of a CornerCutPentagon is 759.5 -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : ∃ (area : ℝ), area = 759.5 := by
  sorry

#check corner_cut_pentagon_area

end NUMINAMATH_CALUDE_corner_cut_pentagon_area_l1223_122366


namespace NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l1223_122318

/-- The price of a single window -/
def window_price : ℕ := 100

/-- The number of windows needed to get one free -/
def windows_for_free : ℕ := 3

/-- The number of windows Dave needs -/
def dave_windows : ℕ := 11

/-- The number of windows Doug needs -/
def doug_windows : ℕ := 9

/-- Calculate the cost of windows given the number needed -/
def calculate_cost (windows_needed : ℕ) : ℕ :=
  let free_windows := windows_needed / windows_for_free
  let paid_windows := windows_needed - free_windows
  paid_windows * window_price

/-- The theorem stating that there's no savings when purchasing together -/
theorem no_savings_on_joint_purchase :
  calculate_cost dave_windows + calculate_cost doug_windows =
  calculate_cost (dave_windows + doug_windows) :=
sorry

end NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l1223_122318


namespace NUMINAMATH_CALUDE_even_sin_function_phi_l1223_122314

theorem even_sin_function_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin ((x + φ) / 3)) →
  (0 ≤ φ ∧ φ ≤ 2 * Real.pi) →
  (∀ x, f x = f (-x)) →
  φ = 3 * Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_even_sin_function_phi_l1223_122314


namespace NUMINAMATH_CALUDE_T_value_for_K_9_l1223_122392

-- Define the equation T = 4hK + 2
def T (h K : ℝ) : ℝ := 4 * h * K + 2

-- State the theorem
theorem T_value_for_K_9 (h : ℝ) :
  (T h 7 = 58) → (T h 9 = 74) := by
  sorry

end NUMINAMATH_CALUDE_T_value_for_K_9_l1223_122392


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1223_122342

def M : Set ℤ := {-2, 1, 2}
def N : Set ℤ := {1, 2, 4}

theorem intersection_of_M_and_N :
  M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1223_122342


namespace NUMINAMATH_CALUDE_antons_number_l1223_122302

def matches_one_digit (a b : ℕ) : Prop :=
  (a / 100 = b / 100 ∧ a % 100 ≠ b % 100) ∨
  (a % 100 / 10 = b % 100 / 10 ∧ a / 100 ≠ b / 100 ∧ a % 10 ≠ b % 10) ∨
  (a % 10 = b % 10 ∧ a / 10 ≠ b / 10)

theorem antons_number (x : ℕ) :
  100 ≤ x ∧ x < 1000 ∧
  matches_one_digit x 109 ∧
  matches_one_digit x 704 ∧
  matches_one_digit x 124 →
  x = 729 := by
  sorry

end NUMINAMATH_CALUDE_antons_number_l1223_122302


namespace NUMINAMATH_CALUDE_unit_fraction_decomposition_l1223_122333

theorem unit_fraction_decomposition (n : ℕ+) : 
  (1 : ℚ) / n = 1 / (2 * n) + 1 / (3 * n) + 1 / (6 * n) := by
  sorry

end NUMINAMATH_CALUDE_unit_fraction_decomposition_l1223_122333


namespace NUMINAMATH_CALUDE_lcm_ratio_sum_l1223_122337

theorem lcm_ratio_sum (a b : ℕ+) : 
  Nat.lcm a b = 42 → 
  a * 3 = b * 2 → 
  a + b = 70 := by
sorry

end NUMINAMATH_CALUDE_lcm_ratio_sum_l1223_122337


namespace NUMINAMATH_CALUDE_problem_solution_l1223_122360

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 31) :
  z + 1 / y = 5 / 27 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1223_122360


namespace NUMINAMATH_CALUDE_square_root_of_64_l1223_122310

theorem square_root_of_64 : {x : ℝ | x^2 = 64} = {-8, 8} := by sorry

end NUMINAMATH_CALUDE_square_root_of_64_l1223_122310


namespace NUMINAMATH_CALUDE_fruit_cost_theorem_l1223_122380

def calculate_fruit_cost (quantity : ℝ) (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let cost_before_discount := quantity * price
  let discounted_cost := cost_before_discount * (1 - discount)
  let tax_amount := discounted_cost * tax
  discounted_cost + tax_amount

def grapes_cost := calculate_fruit_cost 8 70 0.1 0.05
def mangoes_cost := calculate_fruit_cost 9 65 0.05 0.06
def oranges_cost := calculate_fruit_cost 6 60 0 0.03
def apples_cost := calculate_fruit_cost 4 80 0.12 0.07

def total_cost := grapes_cost + mangoes_cost + oranges_cost + apples_cost

theorem fruit_cost_theorem : total_cost = 1790.407 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_theorem_l1223_122380


namespace NUMINAMATH_CALUDE_product_of_distinct_prime_factors_of_B_l1223_122358

def divisors_of_60 : List ℕ := [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]

def B : ℕ := (List.prod divisors_of_60)

theorem product_of_distinct_prime_factors_of_B :
  (Finset.prod (Finset.filter Nat.Prime (Finset.range (B + 1))) id) = 30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_prime_factors_of_B_l1223_122358


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1223_122323

/-- Given a quadratic function f(x) = ax^2 + bx, prove that if f(-1) is between -1 and 2,
    and f(1) is between 2 and 4, then f(-2) is between -1 and 10. -/
theorem quadratic_function_range (a b : ℝ) :
  let f := fun x : ℝ => a * x^2 + b * x
  ((-1 : ℝ) ≤ f (-1) ∧ f (-1) ≤ 2) →
  (2 ≤ f 1 ∧ f 1 ≤ 4) →
  ((-1 : ℝ) ≤ f (-2) ∧ f (-2) ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1223_122323


namespace NUMINAMATH_CALUDE_shortest_player_height_l1223_122363

theorem shortest_player_height (tallest_height shortest_height height_difference : ℝ) :
  tallest_height = 77.75 →
  height_difference = 9.5 →
  tallest_height = shortest_height + height_difference →
  shortest_height = 68.25 := by
sorry

end NUMINAMATH_CALUDE_shortest_player_height_l1223_122363


namespace NUMINAMATH_CALUDE_horse_food_per_day_l1223_122349

/-- Given the ratio of sheep to horses, number of sheep, and total horse food,
    prove the amount of food each horse gets per day. -/
theorem horse_food_per_day
  (sheep_horse_ratio : ℚ) -- Ratio of sheep to horses
  (num_sheep : ℕ) -- Number of sheep
  (total_horse_food : ℕ) -- Total amount of horse food in ounces
  (h1 : sheep_horse_ratio = 2 / 7) -- The ratio of sheep to horses is 2:7
  (h2 : num_sheep = 16) -- There are 16 sheep on the farm
  (h3 : total_horse_food = 12880) -- The farm needs 12,880 ounces of horse food per day
  : ℕ :=
by
  sorry

#check horse_food_per_day

end NUMINAMATH_CALUDE_horse_food_per_day_l1223_122349


namespace NUMINAMATH_CALUDE_quarters_spent_at_arcade_l1223_122382

theorem quarters_spent_at_arcade (initial_quarters : ℕ) (remaining_quarters : ℕ) 
  (h1 : initial_quarters = 88) 
  (h2 : remaining_quarters = 79) : 
  initial_quarters - remaining_quarters = 9 := by
  sorry

end NUMINAMATH_CALUDE_quarters_spent_at_arcade_l1223_122382


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1223_122340

/-- Given a boat that travels 11 km along a stream and 5 km against the stream in one hour,
    prove that its speed in still water is 8 km/hr. -/
theorem boat_speed_in_still_water : 
  ∀ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = 11 →
    boat_speed - stream_speed = 5 →
    boat_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1223_122340


namespace NUMINAMATH_CALUDE_decreasing_cubic_function_l1223_122350

-- Define the function f(x) = ax³ - 2x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x

-- State the theorem
theorem decreasing_cubic_function (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_cubic_function_l1223_122350


namespace NUMINAMATH_CALUDE_number_equality_l1223_122304

theorem number_equality : ∃ x : ℝ, (0.4 * x = 0.3 * 50) ∧ (x = 37.5) := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l1223_122304


namespace NUMINAMATH_CALUDE_value_of_expression_l1223_122330

theorem value_of_expression (m n : ℤ) (h : m - n = -2) : 2 - 5*m + 5*n = 12 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1223_122330


namespace NUMINAMATH_CALUDE_interest_rate_multiple_l1223_122345

theorem interest_rate_multiple (P r m : ℝ) 
  (h1 : P * r^2 = 40)
  (h2 : P * (m * r)^2 = 360)
  : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_multiple_l1223_122345


namespace NUMINAMATH_CALUDE_y_plus_2z_positive_l1223_122300

theorem y_plus_2z_positive (x y z : ℝ) 
  (hx : 0 < x ∧ x < 2) 
  (hy : -2 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 3) : 
  y + 2*z > 0 := by
  sorry

end NUMINAMATH_CALUDE_y_plus_2z_positive_l1223_122300


namespace NUMINAMATH_CALUDE_cube_of_integer_l1223_122324

theorem cube_of_integer (n p : ℕ+) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3)
  (h_div1 : n ∣ (p - 3)) (h_div2 : p ∣ ((n + 1)^3 - 1)) :
  p * n + 1 = (n + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_integer_l1223_122324


namespace NUMINAMATH_CALUDE_diana_bottle_caps_l1223_122303

/-- The number of bottle caps Diana starts with -/
def initial_caps : ℕ := 65

/-- The number of bottle caps eaten by the hippopotamus -/
def eaten_caps : ℕ := 4

/-- The number of bottle caps Diana ends with -/
def final_caps : ℕ := initial_caps - eaten_caps

theorem diana_bottle_caps : final_caps = 61 := by
  sorry

end NUMINAMATH_CALUDE_diana_bottle_caps_l1223_122303


namespace NUMINAMATH_CALUDE_fruits_problem_solution_l1223_122391

def fruits_problem (x : ℕ) : Prop :=
  let last_night_apples : ℕ := 3
  let last_night_bananas : ℕ := 1
  let last_night_oranges : ℕ := 4
  let today_apples : ℕ := last_night_apples + 4
  let today_bananas : ℕ := x * last_night_bananas
  let today_oranges : ℕ := 2 * today_apples
  let total_fruits : ℕ := 39
  (last_night_apples + last_night_bananas + last_night_oranges + 
   today_apples + today_bananas + today_oranges) = total_fruits

theorem fruits_problem_solution : fruits_problem 10 := by
  sorry

end NUMINAMATH_CALUDE_fruits_problem_solution_l1223_122391


namespace NUMINAMATH_CALUDE_value_of_M_l1223_122368

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.35 * 1200) ∧ (M = 1680) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l1223_122368


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l1223_122388

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the exponentiation operation for natural numbers
def pow (base : ℕ) (exp : ℕ) : ℕ := base ^ exp

-- Theorem statement
theorem units_digit_of_sum (a b c d : ℕ) :
  unitsDigit (pow a b + pow c d) = 9 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l1223_122388


namespace NUMINAMATH_CALUDE_amount_distribution_l1223_122332

theorem amount_distribution (A : ℝ) : 
  (A / 14 = A / 18 + 80) → A = 5040 := by
  sorry

end NUMINAMATH_CALUDE_amount_distribution_l1223_122332


namespace NUMINAMATH_CALUDE_trapezoid_crop_distribution_l1223_122386

theorem trapezoid_crop_distribution (a b h : ℝ) (angle : ℝ) :
  a > 0 → b > 0 → h > 0 →
  angle > 0 → angle < π / 2 →
  a = 100 → b = 200 → h = 50 * Real.sqrt 3 → angle = π / 3 →
  let total_area := (a + b) * h / 2
  let closest_to_longest_side := (b + (b - a) / 4) * h / 2
  closest_to_longest_side / total_area = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_crop_distribution_l1223_122386


namespace NUMINAMATH_CALUDE_investment_value_change_l1223_122399

theorem investment_value_change (k m : ℝ) : 
  let increase_factor := 1 + k / 100
  let decrease_factor := 1 - m / 100
  let overall_factor := increase_factor * decrease_factor
  overall_factor = 1 + (k - m - k * m / 100) / 100 :=
by sorry

end NUMINAMATH_CALUDE_investment_value_change_l1223_122399


namespace NUMINAMATH_CALUDE_simplify_expression_l1223_122317

theorem simplify_expression (x : ℝ) : 4 * (x^2 - 5*x) - 5 * (2*x^2 + 3*x) = -6*x^2 - 35*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1223_122317


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1223_122311

/-- The area of an equilateral triangle with altitude √8 is 32√3/3 square units. -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = Real.sqrt 8) :
  let side := (4 * Real.sqrt 6) / 3
  let area := (Real.sqrt 3 / 4) * side^2
  area = 32 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_area_l1223_122311


namespace NUMINAMATH_CALUDE_tire_circumference_l1223_122374

-- Define the given conditions
def car_speed : Real := 168 -- km/h
def tire_revolutions : Real := 400 -- revolutions per minute

-- Define the conversion factors
def km_to_m : Real := 1000 -- 1 km = 1000 m
def hour_to_minute : Real := 60 -- 1 hour = 60 minutes

-- Theorem statement
theorem tire_circumference :
  let speed_m_per_minute : Real := car_speed * km_to_m / hour_to_minute
  let circumference : Real := speed_m_per_minute / tire_revolutions
  circumference = 7 := by sorry

end NUMINAMATH_CALUDE_tire_circumference_l1223_122374


namespace NUMINAMATH_CALUDE_intersection_and_parallel_line_equation_l1223_122372

/-- Given two lines in the plane and their intersection point, prove that a third line
    passing through the intersection point and parallel to a fourth line has a specific equation. -/
theorem intersection_and_parallel_line_equation :
  -- Define the first line: 2x - 3y - 3 = 0
  let l₁ : Set (ℝ × ℝ) := {p | 2 * p.1 - 3 * p.2 - 3 = 0}
  -- Define the second line: x + y + 2 = 0
  let l₂ : Set (ℝ × ℝ) := {p | p.1 + p.2 + 2 = 0}
  -- Define the parallel line: 3x + y - 1 = 0
  let l_parallel : Set (ℝ × ℝ) := {p | 3 * p.1 + p.2 - 1 = 0}
  -- Define the intersection point of l₁ and l₂
  let intersection : ℝ × ℝ := (-3/5, -7/5)
  -- Assume the intersection point lies on both l₁ and l₂
  (intersection ∈ l₁) ∧ (intersection ∈ l₂) →
  -- Define the line we want to prove
  let l : Set (ℝ × ℝ) := {p | 15 * p.1 + 5 * p.2 + 16 = 0}
  -- The line l passes through the intersection point
  (intersection ∈ l) ∧
  -- The line l is parallel to l_parallel
  (∀ (p q : ℝ × ℝ), p ∈ l → q ∈ l → p ≠ q →
    ∃ (r s : ℝ × ℝ), r ∈ l_parallel ∧ s ∈ l_parallel ∧ r ≠ s ∧
      (s.2 - r.2) / (s.1 - r.1) = (q.2 - p.2) / (q.1 - p.1)) :=
by
  sorry


end NUMINAMATH_CALUDE_intersection_and_parallel_line_equation_l1223_122372


namespace NUMINAMATH_CALUDE_aunt_uncle_gift_amount_l1223_122354

/-- The amount of money Chris had before his birthday -/
def initial_amount : ℕ := 159

/-- The amount Chris received from his grandmother -/
def grandmother_gift : ℕ := 25

/-- The amount Chris received from his parents -/
def parents_gift : ℕ := 75

/-- The total amount Chris has after his birthday -/
def final_amount : ℕ := 279

/-- The amount Chris received from his aunt and uncle -/
def aunt_uncle_gift : ℕ := final_amount - initial_amount - grandmother_gift - parents_gift

theorem aunt_uncle_gift_amount : aunt_uncle_gift = 20 := by
  sorry

end NUMINAMATH_CALUDE_aunt_uncle_gift_amount_l1223_122354


namespace NUMINAMATH_CALUDE_nth_root_inequality_l1223_122395

theorem nth_root_inequality (n : ℕ) (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / y) ^ (1 / (n + 1 : ℝ)) ≤ (x + n * y) / ((n + 1) * y) := by
  sorry

end NUMINAMATH_CALUDE_nth_root_inequality_l1223_122395


namespace NUMINAMATH_CALUDE_barbara_initial_candies_l1223_122328

/-- The number of candies Barbara used -/
def candies_used : ℝ := 9.0

/-- The number of candies Barbara has left -/
def candies_left : ℕ := 9

/-- The initial number of candies Barbara had -/
def initial_candies : ℝ := candies_used + candies_left

/-- Theorem stating that Barbara initially had 18 candies -/
theorem barbara_initial_candies : initial_candies = 18 := by
  sorry

end NUMINAMATH_CALUDE_barbara_initial_candies_l1223_122328


namespace NUMINAMATH_CALUDE_complex_modulus_l1223_122376

theorem complex_modulus (i : ℂ) (h : i * i = -1) : 
  Complex.abs (5 * i / (2 - i)) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_l1223_122376


namespace NUMINAMATH_CALUDE_pig_price_l1223_122375

/-- Given 5 pigs and 15 hens with a total cost of 2100 currency units,
    and an average price of 30 currency units per hen,
    prove that the average price of a pig is 330 currency units. -/
theorem pig_price (num_pigs : ℕ) (num_hens : ℕ) (total_cost : ℕ) (hen_price : ℕ) :
  num_pigs = 5 →
  num_hens = 15 →
  total_cost = 2100 →
  hen_price = 30 →
  (total_cost - num_hens * hen_price) / num_pigs = 330 := by
  sorry

end NUMINAMATH_CALUDE_pig_price_l1223_122375


namespace NUMINAMATH_CALUDE_abs_neg_five_l1223_122384

theorem abs_neg_five : abs (-5 : ℤ) = 5 := by sorry

end NUMINAMATH_CALUDE_abs_neg_five_l1223_122384


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l1223_122316

theorem fraction_sum_inequality (b x y z : ℝ) 
  (hb : b > 0) 
  (hx : 0 < x ∧ x < b) 
  (hy : 0 < y ∧ y < b) 
  (hz : 0 < z ∧ z < b) : 
  (x / (b^2 + b*y + z*x)) + (y / (b^2 + b*z + x*y)) + (z / (b^2 + b*x + y*z)) < 1/b := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l1223_122316


namespace NUMINAMATH_CALUDE_triangle_abc_right_angled_l1223_122326

theorem triangle_abc_right_angled (A B C : ℝ) 
  (h1 : A = (1/2) * B) 
  (h2 : A = (1/3) * C) 
  (h3 : A + B + C = 180) : 
  C = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_right_angled_l1223_122326


namespace NUMINAMATH_CALUDE_power_equation_l1223_122341

theorem power_equation (y : ℝ) : (12 : ℝ)^2 * 6^y / 432 = 72 → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l1223_122341


namespace NUMINAMATH_CALUDE_ellipse_and_segment_length_l1223_122394

noncomputable section

-- Define the circles and ellipse
def F₁ (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 9
def F₂ (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 1
def C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the centers of the circles
def center_F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
def center_F₂ : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the line x = 2√3
def line (y : ℝ) : ℝ × ℝ := (2 * Real.sqrt 3, y)

-- Define the theorem
theorem ellipse_and_segment_length 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_foci : C (center_F₁.1) (center_F₁.2) a b ∧ C (center_F₂.1) (center_F₂.2) a b)
  (h_intersection : ∀ x y, F₁ x y ∧ F₂ x y → C x y a b)
  (M N : ℝ × ℝ) 
  (h_M : M.1 = 2 * Real.sqrt 3 ∧ M.2 > 0)
  (h_N : N.1 = 2 * Real.sqrt 3)
  (h_orthogonal : (M.1 - center_F₁.1) * (N.1 - center_F₂.1) + 
                  (M.2 - center_F₁.2) * (N.2 - center_F₂.2) = 0)
  (Q : ℝ × ℝ)
  (h_Q : ∃ t₁ t₂ : ℝ, 
    Q.1 = center_F₁.1 + t₁ * (M.1 - center_F₁.1) ∧
    Q.2 = center_F₁.2 + t₁ * (M.2 - center_F₁.2) ∧
    Q.1 = center_F₂.1 + t₂ * (N.1 - center_F₂.1) ∧
    Q.2 = center_F₂.2 + t₂ * (N.2 - center_F₂.2))
  (h_min : ∀ M' N' : ℝ × ℝ, M'.1 = 2 * Real.sqrt 3 ∧ N'.1 = 2 * Real.sqrt 3 → 
    (M'.1 - center_F₁.1) * (N'.1 - center_F₂.1) + 
    (M'.2 - center_F₁.2) * (N'.2 - center_F₂.2) = 0 → 
    (M.2 - N.2)^2 ≤ (M'.2 - N'.2)^2) :
  (∀ x y, C x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  ((M.1 - Q.1)^2 + (M.2 - Q.2)^2 = 9) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_segment_length_l1223_122394


namespace NUMINAMATH_CALUDE_arithmetic_progression_formula_l1223_122378

/-- An arithmetic progression with specific conditions -/
def ArithmeticProgression (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 3 + a 11 = 24 ∧
  a 4 = 3

/-- The general term formula for the arithmetic progression -/
def GeneralTermFormula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = 3 * n - 9

/-- Theorem stating that the given arithmetic progression has the specified general term formula -/
theorem arithmetic_progression_formula (a : ℕ → ℝ) :
  ArithmeticProgression a → GeneralTermFormula a := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_formula_l1223_122378


namespace NUMINAMATH_CALUDE_banana_permutations_eq_60_l1223_122348

/-- The number of distinct permutations of the word BANANA -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_eq_60_l1223_122348


namespace NUMINAMATH_CALUDE_class_average_weight_l1223_122393

/-- Given two sections A and B in a class, prove that the average weight of the whole class is 38 kg -/
theorem class_average_weight (students_A : ℕ) (students_B : ℕ) (avg_weight_A : ℝ) (avg_weight_B : ℝ)
  (h1 : students_A = 30)
  (h2 : students_B = 20)
  (h3 : avg_weight_A = 40)
  (h4 : avg_weight_B = 35) :
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = 38 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l1223_122393


namespace NUMINAMATH_CALUDE_ham_and_cake_probability_l1223_122346

/-- The probability of packing a ham sandwich and cake on the same day -/
def prob_ham_and_cake (total_days : ℕ) (ham_days : ℕ) (cake_days : ℕ) : ℚ :=
  (ham_days : ℚ) / total_days * (cake_days : ℚ) / total_days

theorem ham_and_cake_probability :
  let total_days : ℕ := 5
  let ham_days : ℕ := 3
  let cake_days : ℕ := 1
  prob_ham_and_cake total_days ham_days cake_days = 12 / 100 := by
sorry

end NUMINAMATH_CALUDE_ham_and_cake_probability_l1223_122346


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1223_122396

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : a 2 = 1/2) 
  (h2 : a 5 = 4) 
  (h_geom : ∀ n : ℕ, n ≥ 1 → ∃ q : ℝ, a (n + 1) = a n * q) :
  ∃ q : ℝ, (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * q) ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1223_122396


namespace NUMINAMATH_CALUDE_rectangle_y_value_l1223_122347

/-- A rectangle with vertices at (0, 0), (0, 5), (y, 5), and (y, 0) has an area of 35 square units. -/
def rectangle_area (y : ℝ) : Prop :=
  y > 0 ∧ y * 5 = 35

/-- The value of y for which the rectangle has an area of 35 square units is 7. -/
theorem rectangle_y_value : ∃ y : ℝ, rectangle_area y ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l1223_122347


namespace NUMINAMATH_CALUDE_three_weighings_sufficient_and_necessary_l1223_122319

/-- Represents the outcome of a weighing on a balance scale -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- A type representing a weighing strategy -/
def WeighStrategy := List (List Nat × List Nat)

/-- Represents the state of knowledge about which coin might be fake -/
structure FakeCoinInfo where
  possibleFakes : List Nat
  isHeavy : Option Bool

/-- The total number of coins -/
def totalCoins : Nat := 13

/-- A theorem stating that 3 weighings are sufficient and necessary to identify the fake coin -/
theorem three_weighings_sufficient_and_necessary :
  ∃ (strategy : WeighStrategy),
    (strategy.length ≤ 3) ∧
    (∀ (fakeCoin : Nat) (isHeavy : Bool),
      fakeCoin < totalCoins →
      ∃ (finalInfo : FakeCoinInfo),
        finalInfo.possibleFakes = [fakeCoin] ∧
        finalInfo.isHeavy = some isHeavy) ∧
    (∀ (strategy' : WeighStrategy),
      strategy'.length < 3 →
      ∃ (fakeCoin1 fakeCoin2 : Nat) (isHeavy1 isHeavy2 : Bool),
        fakeCoin1 ≠ fakeCoin2 ∧
        fakeCoin1 < totalCoins ∧
        fakeCoin2 < totalCoins ∧
        ¬∃ (finalInfo : FakeCoinInfo),
          (finalInfo.possibleFakes = [fakeCoin1] ∧ finalInfo.isHeavy = some isHeavy1) ∨
          (finalInfo.possibleFakes = [fakeCoin2] ∧ finalInfo.isHeavy = some isHeavy2)) :=
by sorry

end NUMINAMATH_CALUDE_three_weighings_sufficient_and_necessary_l1223_122319


namespace NUMINAMATH_CALUDE_solution_characterization_l1223_122313

def valid_solution (a b c x y z : ℕ) : Prop :=
  a + b + c = x * y * z ∧
  x + y + z = a * b * c ∧
  a ≥ b ∧ b ≥ c ∧ c ≥ 1 ∧
  x ≥ y ∧ y ≥ z ∧ z ≥ 1

def solution_set : Set (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) :=
  {(2, 2, 2, 6, 1, 1), (5, 2, 1, 8, 1, 1), (3, 3, 1, 7, 1, 1), (3, 2, 1, 6, 2, 1)}

theorem solution_characterization :
  ∀ a b c x y z : ℕ, valid_solution a b c x y z ↔ (a, b, c, x, y, z) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_solution_characterization_l1223_122313


namespace NUMINAMATH_CALUDE_fraction_addition_simplest_form_l1223_122379

theorem fraction_addition : (13 : ℚ) / 15 + (7 : ℚ) / 9 = (74 : ℚ) / 45 := by
  sorry

theorem simplest_form : Int.gcd 74 45 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_simplest_form_l1223_122379


namespace NUMINAMATH_CALUDE_verandah_flooring_rate_l1223_122362

def hall_length : ℝ := 20
def hall_width : ℝ := 15
def verandah_width : ℝ := 2.5
def total_cost : ℝ := 700

def total_length : ℝ := hall_length + 2 * verandah_width
def total_width : ℝ := hall_width + 2 * verandah_width

def hall_area : ℝ := hall_length * hall_width
def total_area : ℝ := total_length * total_width
def verandah_area : ℝ := total_area - hall_area

theorem verandah_flooring_rate :
  total_cost / verandah_area = 3.5 := by sorry

end NUMINAMATH_CALUDE_verandah_flooring_rate_l1223_122362


namespace NUMINAMATH_CALUDE_marbles_fraction_taken_l1223_122370

theorem marbles_fraction_taken (chris_marbles ryan_marbles remaining_marbles : ℕ) 
  (h1 : chris_marbles = 12)
  (h2 : ryan_marbles = 28)
  (h3 : remaining_marbles = 20) :
  (chris_marbles + ryan_marbles - remaining_marbles) / (chris_marbles + ryan_marbles) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_fraction_taken_l1223_122370


namespace NUMINAMATH_CALUDE_driving_time_ratio_l1223_122367

theorem driving_time_ratio : 
  ∀ (t_28 t_60 : ℝ),
  t_28 + t_60 = 30 →
  t_28 * 28 + t_60 * 60 = 11 * 120 →
  t_28 = 15 ∧ t_28 / (t_28 + t_60) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_driving_time_ratio_l1223_122367


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1223_122315

theorem trigonometric_equation_solution (x : ℝ) : 
  (2 * Real.sin x - Real.sin (2 * x)) / (2 * Real.sin x + Real.sin (2 * x)) + 
  (Real.cos (x / 2) / Real.sin (x / 2))^2 = 10 / 3 →
  ∃ k : ℤ, x = π / 3 * (3 * ↑k + 1) ∨ x = π / 3 * (3 * ↑k - 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1223_122315


namespace NUMINAMATH_CALUDE_megan_fourth_game_score_l1223_122381

/-- Represents Megan's basketball scores --/
structure MeganScores where
  threeGameAverage : ℝ
  fourGameAverage : ℝ

/-- Calculates Megan's score in the fourth game --/
def fourthGameScore (scores : MeganScores) : ℝ :=
  4 * scores.fourGameAverage - 3 * scores.threeGameAverage

/-- Theorem stating Megan's score in the fourth game --/
theorem megan_fourth_game_score :
  ∀ (scores : MeganScores),
    scores.threeGameAverage = 18 →
    scores.fourGameAverage = 17 →
    fourthGameScore scores = 14 := by
  sorry

#eval fourthGameScore { threeGameAverage := 18, fourGameAverage := 17 }

end NUMINAMATH_CALUDE_megan_fourth_game_score_l1223_122381


namespace NUMINAMATH_CALUDE_distinct_arrangements_count_l1223_122339

/-- A regular six-pointed star -/
structure SixPointedStar :=
  (points : Fin 12)

/-- The symmetry group of a regular six-pointed star -/
def star_symmetry_group_order : ℕ := 12

/-- The number of distinct arrangements of 12 unique objects on a regular six-pointed star,
    considering reflections and rotations as equivalent -/
def distinct_arrangements (star : SixPointedStar) : ℕ :=
  Nat.factorial 12 / star_symmetry_group_order

theorem distinct_arrangements_count :
  ∀ (star : SixPointedStar), distinct_arrangements star = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_count_l1223_122339


namespace NUMINAMATH_CALUDE_daves_shirts_l1223_122373

theorem daves_shirts (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) (not_washed : ℕ) : 
  long_sleeve = 27 →
  washed = 20 →
  not_washed = 16 →
  short_sleeve + long_sleeve = washed + not_washed →
  short_sleeve = 9 := by
sorry

end NUMINAMATH_CALUDE_daves_shirts_l1223_122373


namespace NUMINAMATH_CALUDE_gloria_leftover_money_l1223_122361

/-- Calculates the amount of money Gloria has left after selling her trees and buying a cabin -/
def gloria_money_left (initial_cash : ℕ) (cypress_count : ℕ) (pine_count : ℕ) (maple_count : ℕ)
  (cypress_price : ℕ) (pine_price : ℕ) (maple_price : ℕ) (cabin_price : ℕ) : ℕ :=
  let total_earned := initial_cash + cypress_count * cypress_price + pine_count * pine_price + maple_count * maple_price
  total_earned - cabin_price

/-- Theorem stating that Gloria will have $350 left after buying the cabin -/
theorem gloria_leftover_money :
  gloria_money_left 150 20 600 24 100 200 300 129000 = 350 := by
  sorry

end NUMINAMATH_CALUDE_gloria_leftover_money_l1223_122361


namespace NUMINAMATH_CALUDE_sequence_inequality_l1223_122309

theorem sequence_inequality (A B : ℝ) (a : ℕ → ℝ) 
  (hA : A > 1) (hB : B > 1) (ha : ∀ n, 1 ≤ a n ∧ a n ≤ A * B) :
  ∃ b : ℕ → ℝ, (∀ n, 1 ≤ b n ∧ b n ≤ A) ∧
    (∀ m n : ℕ, a m / a n ≤ B * (b m / b n)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1223_122309


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1223_122343

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * (((t.leg : ℝ) ^ 2 - ((t.base : ℝ) / 2) ^ 2).sqrt) / 2

theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t1.base = 4 * t2.base ∧
    perimeter t1 = 740 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s1.base = 4 * s2.base →
      perimeter s1 ≥ 740) :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1223_122343


namespace NUMINAMATH_CALUDE_shirt_cost_theorem_l1223_122385

theorem shirt_cost_theorem (cost_first : ℕ) (cost_difference : ℕ) : 
  cost_first = 15 → cost_difference = 6 → cost_first + (cost_first - cost_difference) = 24 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_theorem_l1223_122385


namespace NUMINAMATH_CALUDE_tamara_kim_height_ratio_l1223_122336

/-- Given Tamara's height and the combined height of Tamara and Kim, 
    prove that Tamara is 17/6 times taller than Kim. -/
theorem tamara_kim_height_ratio :
  ∀ (tamara_height kim_height : ℝ),
    tamara_height = 68 →
    tamara_height + kim_height = 92 →
    tamara_height / kim_height = 17 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_tamara_kim_height_ratio_l1223_122336


namespace NUMINAMATH_CALUDE_red_cars_count_l1223_122306

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 70 → ratio_red = 3 → ratio_black = 8 → 
  (ratio_red : ℚ) / (ratio_black : ℚ) * black_cars = 26 := by
  sorry

end NUMINAMATH_CALUDE_red_cars_count_l1223_122306


namespace NUMINAMATH_CALUDE_dave_initial_money_l1223_122301

-- Define the given amounts
def derek_initial : ℕ := 40
def derek_spend1 : ℕ := 14
def derek_spend2 : ℕ := 11
def derek_spend3 : ℕ := 5
def dave_spend : ℕ := 7
def dave_extra : ℕ := 33

-- Define Derek's total spending
def derek_total_spend : ℕ := derek_spend1 + derek_spend2 + derek_spend3

-- Define Derek's remaining money
def derek_remaining : ℕ := derek_initial - derek_total_spend

-- Define Dave's remaining money
def dave_remaining : ℕ := derek_remaining + dave_extra

-- Theorem to prove
theorem dave_initial_money : dave_remaining + dave_spend = 50 := by
  sorry

end NUMINAMATH_CALUDE_dave_initial_money_l1223_122301


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_sixth_l1223_122344

theorem cos_alpha_minus_pi_sixth (α : ℝ) 
  (h : Real.sin (α + π / 6) + Real.cos α = 4 * Real.sqrt 3 / 5) : 
  Real.cos (α - π / 6) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_sixth_l1223_122344


namespace NUMINAMATH_CALUDE_remaining_honey_l1223_122369

/-- Theorem: Remaining honey after bear consumption --/
theorem remaining_honey (total_honey : ℝ) (eaten_honey : ℝ) 
  (h1 : total_honey = 0.36)
  (h2 : eaten_honey = 0.05) : 
  total_honey - eaten_honey = 0.31 := by
sorry

end NUMINAMATH_CALUDE_remaining_honey_l1223_122369


namespace NUMINAMATH_CALUDE_car_average_speed_l1223_122351

/-- Given a car that travels 65 km in the first hour and 45 km in the second hour,
    prove that its average speed is 55 km/h. -/
theorem car_average_speed (distance1 : ℝ) (distance2 : ℝ) (time : ℝ) 
  (h1 : distance1 = 65)
  (h2 : distance2 = 45)
  (h3 : time = 2) :
  (distance1 + distance2) / time = 55 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l1223_122351


namespace NUMINAMATH_CALUDE_cookie_radius_l1223_122387

theorem cookie_radius (x y : ℝ) :
  (x^2 + y^2 + 36 = 6*x + 9*y) →
  ∃ (center_x center_y : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 = (3*Real.sqrt 5 / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_cookie_radius_l1223_122387


namespace NUMINAMATH_CALUDE_complex_absolute_value_l1223_122398

theorem complex_absolute_value (t : ℝ) (h : t > 0) :
  Complex.abs (-5 + t * Complex.I) = 3 * Real.sqrt 13 → t = 2 * Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l1223_122398


namespace NUMINAMATH_CALUDE_sum_due_proof_l1223_122312

/-- Represents the relationship between banker's discount, true discount, and face value. -/
def bankers_discount_relation (bd td fv : ℚ) : Prop :=
  bd = td + (td * bd / fv)

/-- Proves that given a banker's discount of 80 and a true discount of 70,
    the face value (sum due) is 560. -/
theorem sum_due_proof :
  ∃ (fv : ℚ), bankers_discount_relation 80 70 fv ∧ fv = 560 :=
by sorry

end NUMINAMATH_CALUDE_sum_due_proof_l1223_122312


namespace NUMINAMATH_CALUDE_two_wizards_theorem_l1223_122307

/-- Represents a student in the wizardry school -/
structure Student where
  id : Fin 13
  hasDiploma : Bool

/-- The configuration of students around the table -/
def StudentConfiguration := Fin 13 → Student

/-- Check if a student's prediction is correct -/
def isPredictionCorrect (config : StudentConfiguration) (s : Student) : Bool :=
  let otherStudents := (List.range 13).filter (fun i => 
    i ≠ s.id.val ∧ 
    i ≠ (s.id.val + 1) % 13 ∧ 
    i ≠ (s.id.val + 12) % 13)
  otherStudents.all (fun i => ¬(config i).hasDiploma)

/-- The main theorem to prove -/
theorem two_wizards_theorem :
  ∃ (config : StudentConfiguration),
    (∀ s, (config s.id = s)) ∧
    (∃! (s1 s2 : Student), s1.hasDiploma ∧ s2.hasDiploma ∧ s1 ≠ s2) ∧
    (∀ s, s.hasDiploma ↔ isPredictionCorrect config s) := by
  sorry


end NUMINAMATH_CALUDE_two_wizards_theorem_l1223_122307


namespace NUMINAMATH_CALUDE_correct_observation_value_l1223_122329

theorem correct_observation_value (n : ℕ) (initial_mean corrected_mean wrong_value : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : corrected_mean = 36.02)
  (h4 : wrong_value = 47) :
  let total_sum := n * initial_mean
  let remaining_sum := total_sum - wrong_value
  let corrected_total := n * corrected_mean
  corrected_total - remaining_sum = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_observation_value_l1223_122329


namespace NUMINAMATH_CALUDE_orange_bin_problem_l1223_122325

theorem orange_bin_problem (initial : ℕ) (thrown_away : ℕ) (final : ℕ) 
  (h1 : initial = 40)
  (h2 : thrown_away = 25)
  (h3 : final = 36) :
  final - (initial - thrown_away) = 21 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_problem_l1223_122325


namespace NUMINAMATH_CALUDE_construction_worker_wage_l1223_122322

/-- Represents the daily wage structure for a construction project -/
structure WageStructure where
  worker_wage : ℝ
  electrician_wage : ℝ
  plumber_wage : ℝ
  total_cost : ℝ

/-- Defines the wage structure based on the given conditions -/
def project_wage_structure (w : ℝ) : WageStructure :=
  { worker_wage := w
  , electrician_wage := 2 * w
  , plumber_wage := 2.5 * w
  , total_cost := 2 * w + 2 * w + 2.5 * w }

/-- Theorem stating that the daily wage of a construction worker is $100 -/
theorem construction_worker_wage :
  ∃ w : ℝ, (project_wage_structure w).total_cost = 650 ∧ w = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_construction_worker_wage_l1223_122322


namespace NUMINAMATH_CALUDE_subtraction_with_division_l1223_122352

theorem subtraction_with_division : 6000 - (105 / 21.0) = 5995 := by sorry

end NUMINAMATH_CALUDE_subtraction_with_division_l1223_122352


namespace NUMINAMATH_CALUDE_library_avg_megabytes_per_hour_l1223_122390

/-- Calculates the average megabytes per hour of music in a digital library, rounded to the nearest whole number -/
def avgMegabytesPerHour (days : ℕ) (totalMB : ℕ) : ℕ :=
  let hoursPerDay : ℕ := 24
  let totalHours : ℕ := days * hoursPerDay
  let exactAvg : ℚ := totalMB / totalHours
  (exactAvg + 1/2).floor.toNat

/-- Theorem stating that for a 15-day library with 20,000 MB, the average is 56 MB/hour -/
theorem library_avg_megabytes_per_hour :
  avgMegabytesPerHour 15 20000 = 56 := by
  sorry

end NUMINAMATH_CALUDE_library_avg_megabytes_per_hour_l1223_122390


namespace NUMINAMATH_CALUDE_triangle_area_l1223_122335

-- Define the lines
def line1 (x : ℝ) : ℝ := x
def line2 (x : ℝ) : ℝ := -x
def line3 : ℝ := 8

-- Define the theorem
theorem triangle_area : 
  let A : ℝ × ℝ := (8, 8)
  let B : ℝ × ℝ := (-8, 8)
  let O : ℝ × ℝ := (0, 0)
  let base := |A.1 - B.1|
  let height := |O.2 - line3|
  (1 / 2 : ℝ) * base * height = 64 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1223_122335


namespace NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l1223_122338

theorem cubic_factorization_sum_of_squares (a b c d e f : ℤ) :
  (∀ x : ℝ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l1223_122338


namespace NUMINAMATH_CALUDE_f_always_negative_l1223_122353

/-- The function f(x) = ax^2 + ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 1

/-- Theorem stating that f(x) < 0 for all x ∈ ℝ if and only if -4 < a ≤ 0 -/
theorem f_always_negative (a : ℝ) : 
  (∀ x : ℝ, f a x < 0) ↔ (-4 < a ∧ a ≤ 0) := by sorry

end NUMINAMATH_CALUDE_f_always_negative_l1223_122353


namespace NUMINAMATH_CALUDE_string_cutting_problem_l1223_122383

theorem string_cutting_problem (s l : ℝ) (h1 : s > 0) (h2 : l > 0) 
  (h3 : l - s = 48) (h4 : l + s = 64) : l / s = 7 := by
  sorry

end NUMINAMATH_CALUDE_string_cutting_problem_l1223_122383


namespace NUMINAMATH_CALUDE_cube_net_opposite_face_l1223_122389

-- Define the faces of the cube
inductive Face : Type
  | W | X | Y | Z | V | z

-- Define the concept of opposite faces
def opposite (f1 f2 : Face) : Prop := sorry

-- Define the concept of adjacent faces in the net
def adjacent_in_net (f1 f2 : Face) : Prop := sorry

-- Define the concept of a valid cube net
def valid_cube_net (net : List Face) : Prop := sorry

-- Theorem statement
theorem cube_net_opposite_face (net : List Face) 
  (h_valid : valid_cube_net net)
  (h_z_central : adjacent_in_net Face.z Face.W ∧ 
                 adjacent_in_net Face.z Face.X ∧ 
                 adjacent_in_net Face.z Face.Y)
  (h_v_not_adjacent : ¬adjacent_in_net Face.z Face.V) :
  opposite Face.z Face.V := by sorry

end NUMINAMATH_CALUDE_cube_net_opposite_face_l1223_122389


namespace NUMINAMATH_CALUDE_star_example_l1223_122371

-- Define the star operation
def star (m n p q : ℚ) : ℚ := m * p * (n / q)

-- Theorem statement
theorem star_example : star (5/9) (10/6) = 75 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l1223_122371


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1223_122397

/-- Given a geometric sequence of 9 terms where the first term is 4 and the last term is 2097152,
    prove that the 7th term is 1048576 -/
theorem seventh_term_of_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, 1 ≤ n → n < 9 → a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 4 →                                            -- first term
  a 9 = 2097152 →                                      -- last term
  a 7 = 1048576 :=                                     -- seventh term to prove
by sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1223_122397


namespace NUMINAMATH_CALUDE_derivative_x_squared_cos_x_l1223_122356

theorem derivative_x_squared_cos_x (x : ℝ) :
  deriv (fun x => x^2 * Real.cos x) x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_derivative_x_squared_cos_x_l1223_122356


namespace NUMINAMATH_CALUDE_distance_from_circle_center_to_point_l1223_122327

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4*x + 6*y + 3

-- Define the center of the circle
def circle_center : ℝ × ℝ :=
  let center_x := 2
  let center_y := 3
  (center_x, center_y)

-- Define the given point
def given_point : ℝ × ℝ := (10, 5)

-- State the theorem
theorem distance_from_circle_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  Real.sqrt ((px - cx)^2 + (py - cy)^2) = 2 * Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_distance_from_circle_center_to_point_l1223_122327


namespace NUMINAMATH_CALUDE_homework_problem_count_l1223_122359

theorem homework_problem_count (math_pages reading_pages problems_per_page : ℕ) : 
  math_pages = 4 → reading_pages = 6 → problems_per_page = 4 →
  (math_pages + reading_pages) * problems_per_page = 40 := by
sorry

end NUMINAMATH_CALUDE_homework_problem_count_l1223_122359


namespace NUMINAMATH_CALUDE_yangbajing_largest_in_1975_l1223_122365

/-- Represents a geothermal power station -/
structure GeothermalStation where
  name : String
  capacity : ℕ  -- capacity in kilowatts
  country : String
  year_established : ℕ

/-- The set of all geothermal power stations in China in 1975 -/
def china_geothermal_stations_1975 : Set GeothermalStation :=
  sorry

/-- The Yangbajing Geothermal Power Station -/
def yangbajing : GeothermalStation :=
  { name := "Yangbajing Geothermal Power Station"
  , capacity := 50  -- 50 kilowatts in 1975
  , country := "China"
  , year_established := 1975 }

/-- Theorem: Yangbajing was the largest geothermal power station in China in 1975 -/
theorem yangbajing_largest_in_1975 :
  yangbajing ∈ china_geothermal_stations_1975 ∧
  ∀ s ∈ china_geothermal_stations_1975, s.capacity ≤ yangbajing.capacity :=
by
  sorry

end NUMINAMATH_CALUDE_yangbajing_largest_in_1975_l1223_122365


namespace NUMINAMATH_CALUDE_bounded_function_periodic_l1223_122357

/-- A bounded real function satisfying a specific functional equation is periodic with period 1. -/
theorem bounded_function_periodic (f : ℝ → ℝ) 
  (hbounded : ∃ M, ∀ x, |f x| ≤ M) 
  (hcond : ∀ x, f (x + 1/3) + f (x + 1/2) = f x + f (x + 5/6)) : 
  ∀ x, f (x + 1) = f x := by
  sorry

end NUMINAMATH_CALUDE_bounded_function_periodic_l1223_122357


namespace NUMINAMATH_CALUDE_triangle_area_l1223_122355

/-- The area of a triangle with base 8 and height 4 is 16 -/
theorem triangle_area : 
  ∀ (base height area : ℝ), 
  base = 8 → 
  height = 4 → 
  area = (base * height) / 2 → 
  area = 16 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1223_122355


namespace NUMINAMATH_CALUDE_journey_equations_l1223_122320

/-- Represents a journey between two points with an uphill and a flat section -/
structure Journey where
  uphill_length : ℝ  -- Length of uphill section in km
  flat_length : ℝ    -- Length of flat section in km
  uphill_speed : ℝ   -- Speed on uphill section in km/h
  flat_speed : ℝ     -- Speed on flat section in km/h
  downhill_speed : ℝ -- Speed on downhill section in km/h
  time_ab : ℝ        -- Time from A to B in minutes
  time_ba : ℝ        -- Time from B to A in minutes

/-- The correct system of equations for the journey -/
def correct_equations (j : Journey) : Prop :=
  (j.uphill_length / j.uphill_speed + j.flat_length / j.flat_speed = j.time_ab / 60) ∧
  (j.uphill_length / j.downhill_speed + j.flat_length / j.flat_speed = j.time_ba / 60)

/-- Theorem stating that the given journey satisfies the correct system of equations -/
theorem journey_equations (j : Journey) 
  (h1 : j.uphill_speed = 3)
  (h2 : j.flat_speed = 4)
  (h3 : j.downhill_speed = 5)
  (h4 : j.time_ab = 54)
  (h5 : j.time_ba = 42) :
  correct_equations j := by
  sorry

end NUMINAMATH_CALUDE_journey_equations_l1223_122320


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1223_122308

theorem simplify_sqrt_expression :
  2 * Real.sqrt 5 - 3 * Real.sqrt 25 + 4 * Real.sqrt 80 = 18 * Real.sqrt 5 - 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1223_122308


namespace NUMINAMATH_CALUDE_unique_age_group_split_l1223_122331

theorem unique_age_group_split (total_students : ℕ) 
  (under_10_fraction : ℚ) (between_10_12_fraction : ℚ) (between_12_14_fraction : ℚ) :
  total_students = 60 →
  under_10_fraction = 1/4 →
  between_10_12_fraction = 1/2 →
  between_12_14_fraction = 1/6 →
  ∃! (under_10 between_10_12 between_12_14 above_14 : ℕ),
    under_10 + between_10_12 + between_12_14 + above_14 = total_students ∧
    under_10 = (under_10_fraction * total_students).num ∧
    between_10_12 = (between_10_12_fraction * total_students).num ∧
    between_12_14 = (between_12_14_fraction * total_students).num ∧
    above_14 = total_students - (under_10 + between_10_12 + between_12_14) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_age_group_split_l1223_122331


namespace NUMINAMATH_CALUDE_t_shape_perimeter_specific_l1223_122305

/-- The perimeter of a T-shape formed by two rectangles --/
def t_shape_perimeter (length width overlap : ℝ) : ℝ :=
  2 * (2 * (length + width)) - 2 * overlap

/-- Theorem: The perimeter of a T-shape formed by two 3x5 inch rectangles with a 1.5 inch overlap is 29 inches --/
theorem t_shape_perimeter_specific : t_shape_perimeter 5 3 1.5 = 29 := by
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_specific_l1223_122305


namespace NUMINAMATH_CALUDE_ratio_of_negatives_l1223_122364

theorem ratio_of_negatives (x y : ℝ) (hx : x < 0) (hy : y < 0) (h : 3 * x - 2 * y = Real.sqrt (x * y)) : 
  y / x = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_negatives_l1223_122364


namespace NUMINAMATH_CALUDE_multiply_polynomials_l1223_122321

theorem multiply_polynomials (x : ℝ) : (x^4 + 8*x^2 + 64) * (x^2 - 8) = x^4 + 16*x^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomials_l1223_122321


namespace NUMINAMATH_CALUDE_station_distance_l1223_122377

theorem station_distance (d : ℝ) : 
  (d > 0) → 
  (∃ (x_speed y_speed : ℝ), x_speed > 0 ∧ y_speed > 0 ∧ 
    (d + 100) / x_speed = (d - 100) / y_speed ∧
    (2 * d + 300) / x_speed = (d + 400) / y_speed) →
  (2 * d = 600) := by sorry

end NUMINAMATH_CALUDE_station_distance_l1223_122377


namespace NUMINAMATH_CALUDE_correct_calculation_l1223_122334

theorem correct_calculation (x : ℝ) : x + 10 = 21 → x * 10 = 110 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1223_122334
