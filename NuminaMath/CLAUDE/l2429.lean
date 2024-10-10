import Mathlib

namespace sum_of_five_consecutive_numbers_mod_9_l2429_242976

theorem sum_of_five_consecutive_numbers_mod_9 (n : ℕ) (h : n = 9154) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) % 9 = 1 := by
  sorry

end sum_of_five_consecutive_numbers_mod_9_l2429_242976


namespace factorization_equality_l2429_242932

theorem factorization_equality (x : ℝ) : x^3 - 2*x^2 + x = x*(x-1)^2 := by
  sorry

end factorization_equality_l2429_242932


namespace max_integer_for_inequality_l2429_242951

theorem max_integer_for_inequality : 
  (∀ a : ℕ+, a ≤ 12 → Real.sqrt 3 + Real.sqrt 8 > 1 + Real.sqrt (a : ℝ)) ∧
  (Real.sqrt 3 + Real.sqrt 8 ≤ 1 + Real.sqrt 13) :=
by sorry

end max_integer_for_inequality_l2429_242951


namespace roots_of_unity_real_roots_l2429_242900

theorem roots_of_unity_real_roots (n : ℕ) : 
  ¬ (∀ z : ℂ, z^n = 1 → (z.im = 0 → z = 1)) :=
by sorry

end roots_of_unity_real_roots_l2429_242900


namespace log_product_one_l2429_242923

theorem log_product_one : Real.log 5 / Real.log 2 * Real.log 2 / Real.log 3 * Real.log 3 / Real.log 5 = 1 := by
  sorry

end log_product_one_l2429_242923


namespace cos_300_degrees_l2429_242912

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by sorry

end cos_300_degrees_l2429_242912


namespace train_length_calculation_l2429_242991

/-- Calculates the length of a train given its speed, the speed of a person walking in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 67 →
  person_speed = 5 →
  passing_time = 6 →
  (train_speed + person_speed) * passing_time * (1000 / 3600) = 120 := by
  sorry


end train_length_calculation_l2429_242991


namespace net_profit_calculation_l2429_242972

/-- Calculates the net profit percentage given a markup percentage and discount percentage -/
def netProfitPercentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let markedPrice := 1 + markup
  let sellingPrice := markedPrice * (1 - discount)
  sellingPrice - 1

/-- Theorem stating that a 20% markup followed by a 15% discount results in a 2% net profit -/
theorem net_profit_calculation :
  netProfitPercentage 0.2 0.15 = 0.02 := by
  sorry

#eval netProfitPercentage 0.2 0.15

end net_profit_calculation_l2429_242972


namespace intersection_of_A_and_B_l2429_242970

def A : Set ℝ := {-1, 0, 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by sorry

end intersection_of_A_and_B_l2429_242970


namespace complex_fraction_simplification_l2429_242927

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - i) / (1 + i) = (1 : ℂ) / 2 - (3 : ℂ) / 2 * i :=
by
  sorry

end complex_fraction_simplification_l2429_242927


namespace growth_rate_correct_optimal_price_correct_l2429_242936

-- Define the visitor numbers
def visitors_2022 : ℕ := 200000
def visitors_2024 : ℕ := 288000

-- Define the milk tea shop parameters
def cost_price : ℕ := 6
def base_price : ℕ := 25
def base_sales : ℕ := 300
def price_elasticity : ℕ := 30
def target_profit : ℕ := 6300

-- Part 1: Growth rate
def average_growth_rate : ℚ := 1/5

theorem growth_rate_correct :
  (visitors_2022 : ℚ) * (1 + average_growth_rate)^2 = visitors_2024 := by sorry

-- Part 2: Optimal price
def optimal_price : ℕ := 20

theorem optimal_price_correct :
  (optimal_price - cost_price) * (base_sales + price_elasticity * (base_price - optimal_price)) = target_profit ∧
  ∀ p : ℕ, p < base_price → p > optimal_price →
    (p - cost_price) * (base_sales + price_elasticity * (base_price - p)) < target_profit := by sorry

end growth_rate_correct_optimal_price_correct_l2429_242936


namespace handshake_count_l2429_242958

theorem handshake_count (n : ℕ) (h : n = 7) : (n * (n - 1)) / 2 = 21 := by
  sorry

end handshake_count_l2429_242958


namespace sum_and_simplification_l2429_242946

theorem sum_and_simplification : 
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (7 : ℚ) / 8 + (11 : ℚ) / 12 = (n : ℚ) / d ∧ 
  (∀ (k : ℕ), k > 1 → ¬(k ∣ n ∧ k ∣ d)) :=
by sorry

end sum_and_simplification_l2429_242946


namespace solve_diamond_equation_l2429_242904

-- Define the binary operation ◇
noncomputable def diamond (a b : ℝ) : ℝ :=
  a / b

-- Axioms for the diamond operation
axiom diamond_assoc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond a (diamond b c) = (diamond a b) * c

axiom diamond_self (a : ℝ) (ha : a ≠ 0) :
  diamond a a = 1

-- Theorem to prove
theorem solve_diamond_equation :
  ∃ x : ℝ, x ≠ 0 ∧ diamond 504 (diamond 12 x) = 50 → x = 25 / 21 := by
  sorry

end solve_diamond_equation_l2429_242904


namespace sin_315_degrees_l2429_242977

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_315_degrees_l2429_242977


namespace equation_relation_l2429_242944

theorem equation_relation (x y : ℝ) (h : 2 * x - y = 4) : 6 * x - 3 * y = 12 := by
  sorry

end equation_relation_l2429_242944


namespace triangle_perimeter_l2429_242934

-- Define the triangle's side lengths
def side1 : ℝ := 7
def side2 : ℝ := 10
def side3 : ℝ := 15

-- Define the perimeter of the triangle
def perimeter : ℝ := side1 + side2 + side3

-- Theorem: The perimeter of the triangle is 32
theorem triangle_perimeter : perimeter = 32 := by
  sorry

end triangle_perimeter_l2429_242934


namespace points_per_round_l2429_242996

theorem points_per_round (total_rounds : ℕ) (total_points : ℕ) 
  (h1 : total_rounds = 177)
  (h2 : total_points = 8142) : 
  total_points / total_rounds = 46 := by
  sorry

end points_per_round_l2429_242996


namespace coffee_cost_per_pound_l2429_242987

/-- Calculates the cost per pound of coffee given the initial gift card amount,
    the amount left after purchase, and the number of pounds bought. -/
def cost_per_pound (initial_amount : ℚ) (amount_left : ℚ) (pounds_bought : ℚ) : ℚ :=
  (initial_amount - amount_left) / pounds_bought

/-- Proves that the cost per pound of coffee is $8.58 given the problem conditions. -/
theorem coffee_cost_per_pound :
  cost_per_pound 70 35.68 4 = 8.58 := by
  sorry

end coffee_cost_per_pound_l2429_242987


namespace empty_solution_set_range_subset_solution_set_range_l2429_242910

/-- The solution set of the quadratic inequality mx² - (m+1)x + (m+1) ≥ 0 -/
def solution_set (m : ℝ) : Set ℝ :=
  {x : ℝ | m * x^2 - (m + 1) * x + (m + 1) ≥ 0}

/-- The range of m for which the solution set is empty -/
theorem empty_solution_set_range : 
  ∀ m : ℝ, solution_set m = ∅ ↔ m < -1 :=
sorry

/-- The range of m for which (1,+∞) is a subset of the solution set -/
theorem subset_solution_set_range : 
  ∀ m : ℝ, Set.Ioi 1 ⊆ solution_set m ↔ m ≥ 1/3 :=
sorry

end empty_solution_set_range_subset_solution_set_range_l2429_242910


namespace simplify_fraction_l2429_242980

theorem simplify_fraction : (5^4 + 5^2) / (5^3 - 5) = 65 / 12 := by sorry

end simplify_fraction_l2429_242980


namespace quadratic_factorization_l2429_242955

theorem quadratic_factorization (x : ℝ) : x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by
  sorry

end quadratic_factorization_l2429_242955


namespace grandmother_dolls_l2429_242937

/-- The number of dolls Peggy's grandmother gave her -/
def G : ℕ := 30

/-- Peggy's initial number of dolls -/
def initial_dolls : ℕ := 6

/-- Peggy's final number of dolls -/
def final_dolls : ℕ := 51

theorem grandmother_dolls :
  initial_dolls + G + G / 2 = final_dolls :=
sorry

end grandmother_dolls_l2429_242937


namespace not_perfect_square_l2429_242949

theorem not_perfect_square (n d : ℕ+) (h : d ∣ (2 * n^2)) : ¬ ∃ m : ℕ, (n : ℤ)^2 + d = m^2 := by
  sorry

end not_perfect_square_l2429_242949


namespace equal_area_rectangles_intersection_l2429_242919

/-- A rectangle represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents a horizontal line segment -/
structure HorizontalSegment where
  y : ℝ
  x1 : ℝ
  x2 : ℝ

theorem equal_area_rectangles_intersection 
  (r1 r2 : Rectangle) 
  (h_equal_area : r1.area = r2.area) :
  ∃ (f : ℝ × ℝ → ℝ × ℝ) (g : ℝ × ℝ → ℝ × ℝ),
    (∀ x y, f (x, y) = g (x, y) → 
      ∃ (s1 s2 : HorizontalSegment), 
        s1.y = s2.y ∧ 
        s1.x2 - s1.x1 = s2.x2 - s2.x1 ∧
        (s1.x1, s1.y) ∈ Set.range f ∧
        (s1.x2, s1.y) ∈ Set.range f ∧
        (s2.x1, s2.y) ∈ Set.range g ∧
        (s2.x2, s2.y) ∈ Set.range g) := by
  sorry

end equal_area_rectangles_intersection_l2429_242919


namespace min_value_reciprocal_sum_min_value_attained_l2429_242978

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 := by sorry

theorem min_value_attained (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ (1 / a₀ + 3 / b₀) = 16 := by sorry

end min_value_reciprocal_sum_min_value_attained_l2429_242978


namespace geometric_progression_solution_l2429_242993

/-- A geometric progression is defined by its first term and common ratio -/
structure GeometricProgression where
  first_term : ℝ
  common_ratio : ℝ

/-- Get the nth term of a geometric progression -/
def GeometricProgression.nth_term (gp : GeometricProgression) (n : ℕ) : ℝ :=
  gp.first_term * gp.common_ratio ^ (n - 1)

theorem geometric_progression_solution :
  ∀ (gp : GeometricProgression),
    (gp.nth_term 1 * gp.nth_term 2 * gp.nth_term 3 = 1728) →
    (gp.nth_term 1 + gp.nth_term 2 + gp.nth_term 3 = 63) →
    ((gp.first_term = 3 ∧ gp.common_ratio = 4) ∨ 
     (gp.first_term = 48 ∧ gp.common_ratio = 1/4)) :=
by sorry

end geometric_progression_solution_l2429_242993


namespace lynne_spent_75_l2429_242942

/-- The total amount Lynne spent on books and magazines -/
def total_spent (cat_books solar_books magazines book_price magazine_price : ℕ) : ℕ :=
  (cat_books + solar_books) * book_price + magazines * magazine_price

/-- Theorem stating that Lynne spent $75 in total -/
theorem lynne_spent_75 : 
  total_spent 7 2 3 7 4 = 75 := by
  sorry

end lynne_spent_75_l2429_242942


namespace y₁_less_than_y₂_l2429_242983

/-- A linear function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

/-- Point P₁ on the graph of f -/
def P₁ : ℝ × ℝ := (-3, f (-3))

/-- Point P₂ on the graph of f -/
def P₂ : ℝ × ℝ := (2, f 2)

/-- y₁ is the y-coordinate of P₁ -/
def y₁ : ℝ := (P₁.2)

/-- y₂ is the y-coordinate of P₂ -/
def y₂ : ℝ := (P₂.2)

theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end y₁_less_than_y₂_l2429_242983


namespace wallpaper_coverage_l2429_242924

/-- Given information about wallpaper coverage, proves the area covered by three layers. -/
theorem wallpaper_coverage (total_wallpaper : ℝ) (total_wall : ℝ) (two_layer : ℝ)
  (h1 : total_wallpaper = 300)
  (h2 : total_wall = 180)
  (h3 : two_layer = 30) :
  ∃ (one_layer three_layer : ℝ),
    one_layer + 2 * two_layer + 3 * three_layer = total_wallpaper ∧
    one_layer + two_layer + three_layer = total_wall ∧
    three_layer = 90 :=
by sorry

end wallpaper_coverage_l2429_242924


namespace shirt_cost_without_discount_main_theorem_l2429_242992

theorem shirt_cost_without_discount (team_size : ℕ) 
  (discounted_shirt_cost discounted_pants_cost discounted_socks_cost : ℚ)
  (total_savings : ℚ) : ℚ :=
  let total_discounted_cost := team_size * (discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost)
  let total_undiscounted_cost := total_discounted_cost + total_savings
  let undiscounted_pants_and_socks_cost := team_size * (discounted_pants_cost + discounted_socks_cost)
  let total_undiscounted_shirts_cost := total_undiscounted_cost - undiscounted_pants_and_socks_cost
  total_undiscounted_shirts_cost / team_size

theorem main_theorem : 
  shirt_cost_without_discount 12 (6.75 : ℚ) (13.50 : ℚ) (3.75 : ℚ) (36 : ℚ) = (9.75 : ℚ) := by
  sorry

end shirt_cost_without_discount_main_theorem_l2429_242992


namespace balloon_tanks_l2429_242956

theorem balloon_tanks (num_balloons : ℕ) (air_per_balloon : ℕ) (tank_capacity : ℕ) :
  num_balloons = 1000 →
  air_per_balloon = 10 →
  tank_capacity = 500 →
  (num_balloons * air_per_balloon + tank_capacity - 1) / tank_capacity = 20 :=
by
  sorry

end balloon_tanks_l2429_242956


namespace quadratic_function_minimum_l2429_242959

theorem quadratic_function_minimum (a b c : ℝ) (ha : a > 0) :
  let f := λ x : ℝ => a * x^2 + b * x + c
  let x₀ := -b / (2 * a)
  ¬ (∀ x : ℝ, f x ≤ f x₀) := by
sorry

end quadratic_function_minimum_l2429_242959


namespace green_balls_count_l2429_242914

theorem green_balls_count (red : ℕ) (blue : ℕ) (prob : ℚ) (green : ℕ) : 
  red = 3 → 
  blue = 2 → 
  prob = 1/12 → 
  (red : ℚ)/(red + blue + green : ℚ) * ((red - 1 : ℚ)/(red + blue + green - 1 : ℚ)) = prob → 
  green = 4 :=
by sorry

end green_balls_count_l2429_242914


namespace probability_one_shirt_two_pants_one_sock_l2429_242999

def num_shirts : ℕ := 3
def num_pants : ℕ := 6
def num_socks : ℕ := 9
def total_items : ℕ := num_shirts + num_pants + num_socks
def num_items_to_remove : ℕ := 4

def probability_specific_combination : ℚ :=
  (Nat.choose num_shirts 1 * Nat.choose num_pants 2 * Nat.choose num_socks 1) /
  Nat.choose total_items num_items_to_remove

theorem probability_one_shirt_two_pants_one_sock :
  probability_specific_combination = 15 / 114 := by
  sorry

end probability_one_shirt_two_pants_one_sock_l2429_242999


namespace fraction_decomposition_l2429_242997

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 / (x^2 - 1) = 1 / (x - 1) - 1 / (x + 1)) ∧
  (2*x / (x^2 - 1) = 1 / (x - 1) + 1 / (x + 1)) := by
  sorry

end fraction_decomposition_l2429_242997


namespace line_through_circle_center_parallel_to_given_line_l2429_242948

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 3 = 0

-- Define the given parallel line
def parallel_line (x y : ℝ) : Prop := x + 2*y + 11 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, -2)

-- Define the equation of the line we want to prove
def target_line (x y : ℝ) : Prop := x + 2*y + 1 = 0

-- Theorem statement
theorem line_through_circle_center_parallel_to_given_line :
  ∀ (x y : ℝ),
    (target_line x y ↔ 
      (∃ (t : ℝ), x = circle_center.1 + t ∧ y = circle_center.2 - t/2) ∧
      (∀ (x₁ y₁ x₂ y₂ : ℝ), target_line x₁ y₁ ∧ target_line x₂ y₂ → 
        y₂ - y₁ = -(1/2) * (x₂ - x₁))) :=
sorry

end line_through_circle_center_parallel_to_given_line_l2429_242948


namespace womens_doubles_handshakes_l2429_242994

/-- Calculate the number of handshakes in a tournament -/
def tournament_handshakes (n : ℕ) : ℕ :=
  (n * (n - 2)) / 2

/-- Theorem: In a women's doubles tennis tournament with 4 teams (8 players),
    the total number of handshakes is 24 -/
theorem womens_doubles_handshakes :
  tournament_handshakes 8 = 24 := by
  sorry

end womens_doubles_handshakes_l2429_242994


namespace smallest_four_digit_multiple_of_13_l2429_242990

theorem smallest_four_digit_multiple_of_13 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 13 ∣ n → 1001 ≤ n :=
by sorry

end smallest_four_digit_multiple_of_13_l2429_242990


namespace ellipse_eccentricity_range_l2429_242989

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    where a line passing through the left focus perpendicular to the x-axis 
    intersects the ellipse at points A and B, and the triangle ABF_2 
    (where F_2 is the right focus) is acute, 
    then the eccentricity e of the ellipse is between sqrt(2)-1 and 1. -/
theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Real.sqrt (1 - b^2 / a^2)
  let c := a * e
  let x_A := -c
  let y_A := b^2 / a
  let x_B := -c
  let y_B := -b^2 / a
  let x_F2 := c
  let y_F2 := 0
  (∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → 
    ((x = x_A ∧ y = y_A) ∨ (x = x_B ∧ y = y_B))) →
  (x_A - x_F2)^2 + (y_A - y_F2)^2 < (x_A - x_B)^2 + (y_A - y_B)^2 →
  Real.sqrt 2 - 1 < e ∧ e < 1 :=
sorry

end ellipse_eccentricity_range_l2429_242989


namespace colorcrafter_secret_codes_l2429_242945

/-- The number of available colors in the ColorCrafter game -/
def num_colors : ℕ := 8

/-- The number of slots to fill in the ColorCrafter game -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in the ColorCrafter game -/
def num_secret_codes : ℕ := num_colors ^ num_slots

theorem colorcrafter_secret_codes :
  num_secret_codes = 32768 :=
by sorry

end colorcrafter_secret_codes_l2429_242945


namespace f_has_unique_root_in_interval_l2429_242998

/-- The polynomial function we're analyzing -/
def f (x : ℝ) : ℝ := x^11 + 9*x^10 + 20*x^9 + 2000*x^8 - 1500*x^7

/-- Theorem stating that f has exactly one root in (0,2) -/
theorem f_has_unique_root_in_interval :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end f_has_unique_root_in_interval_l2429_242998


namespace rectangle_area_change_l2429_242979

theorem rectangle_area_change (L W x : ℝ) (h_positive : L > 0 ∧ W > 0) : 
  L * (1 + x / 100) * W * (1 - x / 100) = 1.01 * L * W → x = 10 := by
sorry

end rectangle_area_change_l2429_242979


namespace sufficient_but_not_necessary_l2429_242986

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, (x < -1 → (x < -1 ∨ x > 1)) ∧ ¬((x < -1 ∨ x > 1) → x < -1)) := by
  sorry

end sufficient_but_not_necessary_l2429_242986


namespace exp_T_equals_eleven_fourths_l2429_242975

/-- The integral T is defined as the definite integral of (2e^(3x) + e^(2x) - 1) / (e^(3x) + e^(2x) - e^x + 1) from 0 to ln(2) -/
noncomputable def T : ℝ := ∫ x in (0)..(Real.log 2), (2 * Real.exp (3 * x) + Real.exp (2 * x) - 1) / (Real.exp (3 * x) + Real.exp (2 * x) - Real.exp x + 1)

/-- The theorem states that e^T equals 11/4 -/
theorem exp_T_equals_eleven_fourths : Real.exp T = 11 / 4 := by sorry

end exp_T_equals_eleven_fourths_l2429_242975


namespace equation_system_result_l2429_242954

theorem equation_system_result : ∃ (x y z : ℝ), 
  z ≠ 0 ∧ 
  3*x - 4*y - 2*z = 0 ∧ 
  x + 4*y - 20*z = 0 ∧ 
  (x^2 + 4*x*y) / (y^2 + z^2) = 106496/36324 := by
  sorry

end equation_system_result_l2429_242954


namespace sum_of_ages_is_50_l2429_242921

/-- The sum of ages of 5 children born 2 years apart, where the eldest child is 14 years old -/
def sum_of_ages : ℕ → Prop
| n => ∃ (a b c d e : ℕ),
    a = 14 ∧
    b = a - 2 ∧
    c = b - 2 ∧
    d = c - 2 ∧
    e = d - 2 ∧
    n = a + b + c + d + e

/-- Theorem stating that the sum of ages is 50 years -/
theorem sum_of_ages_is_50 : sum_of_ages 50 := by
  sorry

end sum_of_ages_is_50_l2429_242921


namespace demolition_time_with_injury_l2429_242935

/-- The time it takes to demolish a building given the work rates of different combinations of workers and an injury to one worker. -/
theorem demolition_time_with_injury 
  (carl_bob_rate : ℚ)
  (anne_bob_rate : ℚ)
  (anne_carl_rate : ℚ)
  (h_carl_bob : carl_bob_rate = 1 / 6)
  (h_anne_bob : anne_bob_rate = 1 / 3)
  (h_anne_carl : anne_carl_rate = 1 / 5) :
  let all_rate := carl_bob_rate + anne_bob_rate + anne_carl_rate - 1 / 2
  let work_done_day_one := all_rate
  let remaining_work := 1 - work_done_day_one
  let time_for_remainder := remaining_work / anne_bob_rate
  1 + time_for_remainder = 59 / 20 := by
sorry

end demolition_time_with_injury_l2429_242935


namespace fair_hair_percentage_l2429_242985

theorem fair_hair_percentage 
  (total_employees : ℝ) 
  (women_fair_hair_percentage : ℝ) 
  (women_percentage_of_fair_hair : ℝ) 
  (h1 : women_fair_hair_percentage = 30) 
  (h2 : women_percentage_of_fair_hair = 40) 
  (h3 : total_employees > 0) :
  (women_fair_hair_percentage * total_employees / 100) / 
  (women_percentage_of_fair_hair / 100) / 
  total_employees * 100 = 75 := by
sorry

end fair_hair_percentage_l2429_242985


namespace garbs_are_morks_and_plogs_l2429_242902

-- Define the sets
variable (U : Type) -- Universe set
variable (Mork Plog Snark Garb : Set U)

-- Define the given conditions
variable (h1 : Mork ⊆ Plog)    -- All Morks are Plogs
variable (h2 : Snark ⊆ Mork)   -- All Snarks are Morks
variable (h3 : Garb ⊆ Plog)    -- All Garbs are Plogs
variable (h4 : Garb ⊆ Snark)   -- All Garbs are Snarks

-- Theorem to prove
theorem garbs_are_morks_and_plogs : Garb ⊆ Mork ∩ Plog :=
sorry

end garbs_are_morks_and_plogs_l2429_242902


namespace binomial_expansion_coefficient_l2429_242957

theorem binomial_expansion_coefficient (m : ℝ) : 
  m > 0 → 
  (Nat.choose 5 2 * m^2 = Nat.choose 5 1 * m + 30) → 
  m = 2 := by
sorry

end binomial_expansion_coefficient_l2429_242957


namespace files_per_folder_l2429_242928

theorem files_per_folder (initial_files : ℕ) (deleted_files : ℕ) (num_folders : ℕ) :
  initial_files = 93 →
  deleted_files = 21 →
  num_folders = 9 →
  (initial_files - deleted_files) / num_folders = 8 :=
by
  sorry

end files_per_folder_l2429_242928


namespace marla_bags_per_trip_l2429_242911

/-- The number of ounces in a pound -/
def ounces_per_pound : ℕ := 16

/-- The number of shopping trips -/
def num_trips : ℕ := 300

/-- The amount of CO2 released by the canvas bag (in pounds) -/
def canvas_bag_co2 : ℕ := 600

/-- The amount of CO2 released by each plastic bag (in ounces) -/
def plastic_bag_co2 : ℕ := 4

/-- The number of plastic bags Marla uses per shopping trip -/
def bags_per_trip : ℕ := 8

theorem marla_bags_per_trip :
  (bags_per_trip * plastic_bag_co2 * num_trips) / ounces_per_pound = canvas_bag_co2 :=
sorry

end marla_bags_per_trip_l2429_242911


namespace right_triangle_among_sets_l2429_242973

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_among_sets : 
  (¬ is_right_triangle 2 3 4) ∧
  (is_right_triangle 3 4 5) ∧
  (¬ is_right_triangle 4 5 6) ∧
  (¬ is_right_triangle 6 8 9) :=
sorry

end right_triangle_among_sets_l2429_242973


namespace coin_toss_frequency_l2429_242933

/-- Given a coin tossed 10 times with 6 heads, prove that the frequency of heads is 3/5 -/
theorem coin_toss_frequency :
  ∀ (total_tosses : ℕ) (heads : ℕ),
  total_tosses = 10 →
  heads = 6 →
  (heads : ℚ) / total_tosses = 3 / 5 := by
sorry

end coin_toss_frequency_l2429_242933


namespace sum_of_reciprocal_equations_l2429_242962

theorem sum_of_reciprocal_equations (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : 1/x - 1/y = 1) : 
  x + y = 5/6 := by
  sorry

end sum_of_reciprocal_equations_l2429_242962


namespace number_satisfying_equation_l2429_242915

theorem number_satisfying_equation : ∃! x : ℚ, x + 72 = 2 * x / (2/3) := by
  sorry

end number_satisfying_equation_l2429_242915


namespace marker_selection_ways_l2429_242907

theorem marker_selection_ways :
  let n : ℕ := 15  -- Total number of markers
  let k : ℕ := 5   -- Number of markers to be selected
  Nat.choose n k = 3003 :=
by sorry

end marker_selection_ways_l2429_242907


namespace money_division_l2429_242981

theorem money_division (total : ℕ) (p q r : ℕ) : 
  p + q + r = total ∧ 
  3 * p = 7 * q ∧ 
  7 * q = 12 * r ∧ 
  r - q = 3500 → 
  q - p = 2800 := by sorry

end money_division_l2429_242981


namespace dana_friday_hours_l2429_242967

/-- Dana's hourly rate in dollars -/
def hourly_rate : ℕ := 13

/-- Hours worked on Saturday -/
def saturday_hours : ℕ := 10

/-- Hours worked on Sunday -/
def sunday_hours : ℕ := 3

/-- Total earnings for all three days in dollars -/
def total_earnings : ℕ := 286

/-- Calculates the number of hours worked on Friday -/
def friday_hours : ℕ :=
  (total_earnings - (hourly_rate * (saturday_hours + sunday_hours))) / hourly_rate

theorem dana_friday_hours :
  friday_hours = 9 := by
  sorry

end dana_friday_hours_l2429_242967


namespace technicians_sample_size_l2429_242968

/-- Represents the number of technicians to be included in a stratified sample -/
def technicians_in_sample (total_engineers : ℕ) (total_technicians : ℕ) (total_workers : ℕ) (sample_size : ℕ) : ℕ :=
  (total_technicians * sample_size) / (total_engineers + total_technicians + total_workers)

/-- Theorem stating that the number of technicians in the sample is 5 -/
theorem technicians_sample_size :
  technicians_in_sample 20 100 280 20 = 5 := by
  sorry

#eval technicians_in_sample 20 100 280 20

end technicians_sample_size_l2429_242968


namespace dodecahedral_die_expected_value_l2429_242920

/-- A fair dodecahedral die with faces numbered 1 to 12 -/
def DodecahedralDie : Finset ℕ := Finset.range 12

/-- The probability of each outcome for a fair die -/
def prob (n : ℕ) : ℚ := 1 / 12

/-- The expected value of rolling the dodecahedral die -/
def expected_value : ℚ := (DodecahedralDie.sum (fun i => prob i * (i + 1)))

/-- Theorem: The expected value of rolling a fair dodecahedral die is 6.5 -/
theorem dodecahedral_die_expected_value : expected_value = 13 / 2 := by
  sorry

end dodecahedral_die_expected_value_l2429_242920


namespace samuel_remaining_money_l2429_242938

theorem samuel_remaining_money (total : ℝ) (samuel_fraction : ℝ) (spent_fraction : ℝ) : 
  total = 240 →
  samuel_fraction = 3/8 →
  spent_fraction = 1/5 →
  samuel_fraction * total - spent_fraction * total = 42 := by
sorry

end samuel_remaining_money_l2429_242938


namespace function_properties_l2429_242939

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h_odd : is_odd (fun x => f (2*x + 1)))
  (h_period : has_period (fun x => f (2*x + 1)) 2) :
  (∀ x, f (x + 1) + f (-x + 1) = 0) ∧
  (∀ x, f x = f (x + 4)) := by
sorry

end function_properties_l2429_242939


namespace sector_area_l2429_242943

/-- Given a circular sector with central angle π/3 and chord length 3 cm,
    the area of the sector is 3π/2 cm². -/
theorem sector_area (θ : Real) (chord_length : Real) (area : Real) :
  θ = π / 3 →
  chord_length = 3 →
  area = 3 * π / 2 :=
by sorry

end sector_area_l2429_242943


namespace polynomial_roots_l2429_242950

theorem polynomial_roots : 
  let p : ℝ → ℝ := fun x ↦ x^3 - 4*x^2 - 7*x + 10
  ∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 5 ∨ x = -2 :=
by
  sorry

end polynomial_roots_l2429_242950


namespace jackie_free_time_l2429_242960

theorem jackie_free_time (total_hours work_hours exercise_hours sleep_hours : ℕ)
  (h1 : total_hours = 24)
  (h2 : work_hours = 8)
  (h3 : exercise_hours = 3)
  (h4 : sleep_hours = 8) :
  total_hours - (work_hours + exercise_hours + sleep_hours) = 5 := by
  sorry

end jackie_free_time_l2429_242960


namespace max_value_of_fraction_l2429_242940

theorem max_value_of_fraction (x : ℝ) (hx : x < 0) :
  (1 + x^2) / x ≤ -2 ∧ ((1 + x^2) / x = -2 ↔ x = -1) :=
by sorry

end max_value_of_fraction_l2429_242940


namespace exists_valid_grid_l2429_242984

/-- Represents a 6x6 grid of natural numbers -/
def Grid := Fin 6 → Fin 6 → Nat

/-- Checks if a given row contains numbers 1 to 6 without repetition -/
def valid_row (g : Grid) (row : Fin 6) : Prop :=
  ∀ n : Fin 6, ∃! col : Fin 6, g row col = n.val.succ

/-- Checks if a given column contains numbers 1 to 6 without repetition -/
def valid_column (g : Grid) (col : Fin 6) : Prop :=
  ∀ n : Fin 6, ∃! row : Fin 6, g row col = n.val.succ

/-- Checks if a given 2x3 block contains numbers 1 to 6 without repetition -/
def valid_block (g : Grid) (start_row start_col : Fin 3) : Prop :=
  ∀ n : Fin 6, ∃! (row : Fin 2) (col : Fin 3), 
    g (start_row * 2 + row) (start_col * 3 + col) = n.val.succ

/-- Checks if the number between two adjacent cells is their sum or product -/
def valid_between (g : Grid) : Prop :=
  ∀ (row col : Fin 6) (n : Nat),
    (row.val < 5 → n = g row col + g (row.succ) col ∨ n = g row col * g (row.succ) col) ∧
    (col.val < 5 → n = g row col + g row (col.succ) ∨ n = g row col * g row (col.succ))

/-- Main theorem: There exists a valid grid satisfying all conditions -/
theorem exists_valid_grid : ∃ (g : Grid),
  (∀ row : Fin 6, valid_row g row) ∧
  (∀ col : Fin 6, valid_column g col) ∧
  (∀ start_row start_col : Fin 3, valid_block g start_row start_col) ∧
  valid_between g :=
sorry

end exists_valid_grid_l2429_242984


namespace not_both_perfect_squares_l2429_242966

theorem not_both_perfect_squares (x y z t : ℕ+) 
  (h1 : x.val * y.val - z.val * t.val = x.val + y.val)
  (h2 : x.val + y.val = z.val + t.val) : 
  ¬(∃ (a c : ℕ+), (x.val * y.val = a.val ^ 2) ∧ (z.val * t.val = c.val ^ 2)) :=
sorry

end not_both_perfect_squares_l2429_242966


namespace function_sum_property_l2429_242922

/-- Given a function f(x) = ax^7 - bx^5 + cx^3 + 2, where f(-5) = 3, prove that f(5) + f(-5) = 4 -/
theorem function_sum_property (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^7 - b * x^5 + c * x^3 + 2
  (f (-5) = 3) → (f 5 + f (-5) = 4) := by
  sorry

end function_sum_property_l2429_242922


namespace ice_bag_cost_calculation_l2429_242953

def small_bag_cost : ℚ := 80 / 100
def large_bag_cost : ℚ := 146 / 100
def total_bags : ℕ := 30
def small_bags : ℕ := 18
def discount_rate : ℚ := 12 / 100

theorem ice_bag_cost_calculation :
  let large_bags : ℕ := total_bags - small_bags
  let total_cost_before_discount : ℚ := small_bag_cost * small_bags + large_bag_cost * large_bags
  let discount_amount : ℚ := total_cost_before_discount * discount_rate
  let total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount
  ∃ (rounded_cost : ℚ), (rounded_cost * 100).floor = 2809 ∧ 
    |rounded_cost - total_cost_after_discount| ≤ 1 / 200 :=
by sorry

end ice_bag_cost_calculation_l2429_242953


namespace probability_at_least_four_mismatched_l2429_242995

/-- The number of derangements for n elements -/
def derangement (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | n + 2 => (n + 1) * (derangement (n + 1) + derangement n)

/-- The number of students and subjects -/
def num_students : ℕ := 5

/-- The probability of at least 4 out of 5 students receiving a mismatched test paper -/
def probability_mismatched_tests : ℚ :=
  (num_students * derangement (num_students - 1) + derangement num_students) / (num_students.factorial)

theorem probability_at_least_four_mismatched :
  probability_mismatched_tests = 89 / 120 := by
  sorry


end probability_at_least_four_mismatched_l2429_242995


namespace time_marking_of_7_45_l2429_242913

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : minutes < 60

/-- Converts a Time to minutes since midnight -/
def timeToMinutes (t : Time) : ℕ := t.hours * 60 + t.minutes

/-- The base time (10:00 AM) -/
def baseTime : Time := ⟨10, 0, by norm_num⟩

/-- The time unit in minutes -/
def timeUnit : ℕ := 45

/-- Calculates the time marking for a given time -/
def timeMarking (t : Time) : ℤ :=
  (timeToMinutes t - timeToMinutes baseTime : ℤ) / timeUnit

/-- The time to be marked (7:45 AM) -/
def givenTime : Time := ⟨7, 45, by norm_num⟩

theorem time_marking_of_7_45 : timeMarking givenTime = -3 := by sorry

end time_marking_of_7_45_l2429_242913


namespace minimize_y_l2429_242906

/-- The function to be minimized -/
def y (x a b : ℝ) : ℝ := 3 * (x - a)^2 + (x - b)^2

/-- The derivative of y with respect to x -/
def y_derivative (x a b : ℝ) : ℝ := 8 * x - 6 * a - 2 * b

/-- The second derivative of y with respect to x -/
def y_second_derivative : ℝ := 8

theorem minimize_y (a b : ℝ) :
  ∃ (x : ℝ), (∀ (z : ℝ), y z a b ≥ y x a b) ∧ x = (3 * a + b) / 4 := by
  sorry

#check minimize_y

end minimize_y_l2429_242906


namespace transport_probabilities_l2429_242905

/-- Transportation method probabilities -/
structure TransportProb where
  train : ℝ
  ship : ℝ
  car : ℝ
  airplane : ℝ
  sum_to_one : train + ship + car + airplane = 1
  all_nonneg : train ≥ 0 ∧ ship ≥ 0 ∧ car ≥ 0 ∧ airplane ≥ 0

/-- Given probabilities for each transportation method -/
def given_probs : TransportProb where
  train := 0.3
  ship := 0.1
  car := 0.2
  airplane := 0.4
  sum_to_one := by norm_num
  all_nonneg := by norm_num

/-- Theorem stating the probabilities of combined events -/
theorem transport_probabilities (p : TransportProb) :
  p.train + p.airplane = 0.7 ∧ 1 - p.ship = 0.9 := by sorry

end transport_probabilities_l2429_242905


namespace committee_formation_count_l2429_242974

theorem committee_formation_count (n m k : ℕ) (hn : n = 10) (hm : m = 5) (hk : k = 1) :
  (Nat.choose (n - k) (m - k)) = 126 := by
  sorry

end committee_formation_count_l2429_242974


namespace largest_non_representable_l2429_242909

def is_representable (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), n = 15 * x + 18 * y + 20 * z

theorem largest_non_representable : 
  (∀ m > 97, is_representable m) ∧ ¬(is_representable 97) :=
sorry

end largest_non_representable_l2429_242909


namespace find_a_lower_bound_m_l2429_242925

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1: Prove that a = 2
theorem find_a : 
  (∀ x : ℝ, f 2 x ≤ 3 ↔ x ∈ Set.Icc (-1) 5) → 
  (∃! a : ℝ, ∀ x : ℝ, f a x ≤ 3 ↔ x ∈ Set.Icc (-1) 5) :=
sorry

-- Theorem 2: Prove that f(x) + f(x + 5) ≥ 5 for all real x
theorem lower_bound_m (x : ℝ) : f 2 x + f 2 (x + 5) ≥ 5 :=
sorry

end find_a_lower_bound_m_l2429_242925


namespace log_minus_x_decreasing_l2429_242971

theorem log_minus_x_decreasing (a b c : ℝ) (h : 1 < a ∧ a < b ∧ b < c) :
  Real.log a - a > Real.log b - b ∧ Real.log b - b > Real.log c - c := by
  sorry

end log_minus_x_decreasing_l2429_242971


namespace smallest_perimeter_rectangle_l2429_242931

/-- A polygon made of unit squares -/
structure UnitSquarePolygon where
  area : ℕ

/-- The problem setup -/
def problem_setup (p1 p2 : UnitSquarePolygon) : Prop :=
  p1.area + p2.area = 16

/-- The perimeter of a rectangle -/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- A rectangle can contain the polygons if its area is at least the sum of the polygons' areas -/
def can_contain (length width : ℕ) (p1 p2 : UnitSquarePolygon) : Prop :=
  length * width ≥ p1.area + p2.area

/-- The theorem statement -/
theorem smallest_perimeter_rectangle (p1 p2 : UnitSquarePolygon) 
  (h : problem_setup p1 p2) :
  ∃ (length width : ℕ), 
    can_contain length width p1 p2 ∧ 
    (∀ (l w : ℕ), can_contain l w p1 p2 → rectangle_perimeter length width ≤ rectangle_perimeter l w) ∧
    rectangle_perimeter length width = 18 := by
  sorry

end smallest_perimeter_rectangle_l2429_242931


namespace find_constant_k_l2429_242917

theorem find_constant_k : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4) → k = -16 := by
  sorry

end find_constant_k_l2429_242917


namespace symmetry_of_shifted_even_function_l2429_242929

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The axis of symmetry for a function f is a vertical line x = a such that
    f(a + x) = f(a - x) for all x in the domain of f -/
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) :
  IsEven (fun x ↦ f (x + 1)) → AxisOfSymmetry f 1 := by sorry

end symmetry_of_shifted_even_function_l2429_242929


namespace unknown_dimension_is_15_l2429_242963

/-- Represents the dimensions and features of a room -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ
  door_area : ℝ
  window_area : ℝ
  window_count : ℕ
  whitewash_cost_per_sqft : ℝ
  total_cost : ℝ

/-- Calculates the area to be whitewashed in the room -/
def area_to_whitewash (r : Room) : ℝ :=
  2 * (r.length * r.height + r.width * r.height) - (r.door_area + r.window_count * r.window_area)

/-- Theorem stating the conditions and the result to be proved -/
theorem unknown_dimension_is_15 (r : Room) 
    (h1 : r.length = 25)
    (h2 : r.height = 12)
    (h3 : r.door_area = 6 * 3)
    (h4 : r.window_area = 4 * 3)
    (h5 : r.window_count = 3)
    (h6 : r.whitewash_cost_per_sqft = 2)
    (h7 : r.total_cost = 1812)
    (h8 : r.total_cost = area_to_whitewash r * r.whitewash_cost_per_sqft) :
  r.width = 15 := by
  sorry

end unknown_dimension_is_15_l2429_242963


namespace inradius_of_specific_triangle_l2429_242908

/-- Represents a triangle with side lengths a, b, c and incenter distance to one vertex d -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the inradius of a triangle -/
def inradius (t : Triangle) : ℝ := sorry

/-- Theorem stating that for the given triangle, the inradius is 0.8 * √14 -/
theorem inradius_of_specific_triangle :
  let t : Triangle := { a := 30, b := 36, c := 34, d := 18 }
  inradius t = 0.8 * Real.sqrt 14 := by sorry

end inradius_of_specific_triangle_l2429_242908


namespace intersection_of_A_and_B_l2429_242941

def A : Set ℝ := {x | x^2 - x + 1 ≥ 0}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

end intersection_of_A_and_B_l2429_242941


namespace five_digit_number_formation_l2429_242926

theorem five_digit_number_formation (m n : ℕ) : 
  (100 ≤ m) ∧ (m < 1000) ∧ (10 ≤ n) ∧ (n < 100) → 
  (m * 100 + n = 100 * m + n) := by
  sorry

end five_digit_number_formation_l2429_242926


namespace inverse_of_AB_l2429_242930

def A : Matrix (Fin 2) (Fin 2) ℚ := !![1, 0; 0, 2]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![1, 1/2; 0, 1]

theorem inverse_of_AB :
  (A * B)⁻¹ = !![1, -1; 0, 1/2] := by sorry

end inverse_of_AB_l2429_242930


namespace xy_value_l2429_242961

theorem xy_value (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) : x * y = 1 := by
  sorry

end xy_value_l2429_242961


namespace composition_result_l2429_242965

noncomputable def P (x : ℝ) : ℝ := 3 * Real.sqrt x

def Q (x : ℝ) : ℝ := x^3

theorem composition_result :
  P (Q (P (Q (P (Q 2))))) = 846 * Real.sqrt 2 := by
  sorry

end composition_result_l2429_242965


namespace endpoint_coordinate_sum_l2429_242982

/-- Given a segment with midpoint (10, -5) and one endpoint (15, 10),
    the sum of the coordinates of the other endpoint is -15. -/
theorem endpoint_coordinate_sum :
  let midpoint : ℝ × ℝ := (10, -5)
  let endpoint1 : ℝ × ℝ := (15, 10)
  let endpoint2 : ℝ × ℝ := (2 * midpoint.1 - endpoint1.1, 2 * midpoint.2 - endpoint1.2)
  endpoint2.1 + endpoint2.2 = -15 := by
sorry

end endpoint_coordinate_sum_l2429_242982


namespace solution_set_min_sum_reciprocals_equality_condition_l2429_242901

-- Define the inequality
def inequality (x : ℝ) : Prop := |x + 1| + |2*x - 1| ≤ 3

-- Theorem 1: Solution set of the inequality
theorem solution_set : 
  {x : ℝ | inequality x} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: Minimum value of the sum of reciprocals
theorem min_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_constraint : a + b + c = 2) :
  1/a + 1/b + 1/c ≥ 9/2 := by sorry

-- Theorem 3: Condition for equality
theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_constraint : a + b + c = 2) :
  1/a + 1/b + 1/c = 9/2 ↔ a = 2/3 ∧ b = 2/3 ∧ c = 2/3 := by sorry

end solution_set_min_sum_reciprocals_equality_condition_l2429_242901


namespace sin_945_degrees_l2429_242988

theorem sin_945_degrees : Real.sin (945 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_945_degrees_l2429_242988


namespace extra_apples_l2429_242916

theorem extra_apples (red_apples green_apples students : ℕ) 
  (h1 : red_apples = 33)
  (h2 : green_apples = 23)
  (h3 : students = 21)
  (h4 : ∀ s, s ≤ students → s = 1) : 
  red_apples + green_apples - students = 35 := by
  sorry

end extra_apples_l2429_242916


namespace cube_volume_from_surface_area_l2429_242903

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l2429_242903


namespace cos_tan_values_l2429_242964

-- Define the angle θ and the parameter m
variable (θ m : ℝ)

-- Define the conditions
def terminal_side_condition : Prop := ∃ (r : ℝ), r * (-Real.sqrt 3) = r * Real.cos θ ∧ r * m = r * Real.sin θ
def sin_condition : Prop := Real.sin θ = (Real.sqrt 2 / 4) * m

-- Define the theorem
theorem cos_tan_values (h1 : terminal_side_condition θ m) (h2 : sin_condition θ m) :
  (m = 0 ∧ Real.cos θ = -1 ∧ Real.tan θ = 0) ∨
  (m = Real.sqrt 5 ∧ Real.cos θ = -Real.sqrt 6 / 4 ∧ Real.tan θ = -Real.sqrt 15 / 3) ∨
  (m = -Real.sqrt 5 ∧ Real.cos θ = -Real.sqrt 6 / 4 ∧ Real.tan θ = Real.sqrt 15 / 3) :=
by sorry

end cos_tan_values_l2429_242964


namespace inequality_proof_l2429_242947

theorem inequality_proof (a : ℝ) (h : a > 0) : 
  Real.sqrt (a + 1/a) - Real.sqrt 2 ≥ Real.sqrt a + 1/(Real.sqrt a) - 2 := by
  sorry

end inequality_proof_l2429_242947


namespace thomas_monthly_earnings_l2429_242952

/-- Thomas's weekly earnings in the factory -/
def weekly_earnings : ℕ := 4550

/-- Number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- Calculates monthly earnings based on weekly earnings and number of weeks in a month -/
def monthly_earnings (w : ℕ) (n : ℕ) : ℕ := w * n

/-- Theorem stating that Thomas's monthly earnings are 18200 -/
theorem thomas_monthly_earnings :
  monthly_earnings weekly_earnings weeks_in_month = 18200 := by
  sorry

end thomas_monthly_earnings_l2429_242952


namespace negative_f_reflection_l2429_242969

-- Define a function f
variable (f : ℝ → ℝ)

-- Define reflection across x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Theorem: The graph of y = -f(x) is the reflection of y = f(x) across the x-axis
theorem negative_f_reflection (x : ℝ) : 
  reflect_x (x, f x) = (x, -f x) := by sorry

end negative_f_reflection_l2429_242969


namespace total_tickets_sold_l2429_242918

/-- Represents the number of tickets sold at different prices and times -/
structure TicketSales where
  reducedFirstWeek : ℕ
  reducedRemainingWeeks : ℕ
  fullPrice : ℕ

/-- Calculates the total number of tickets sold -/
def totalTicketsSold (sales : TicketSales) : ℕ :=
  sales.reducedFirstWeek + sales.reducedRemainingWeeks + sales.fullPrice

/-- Theorem stating the total number of tickets sold given the conditions -/
theorem total_tickets_sold :
  ∀ (sales : TicketSales),
    sales.reducedFirstWeek = 5400 →
    sales.fullPrice = 16500 →
    sales.fullPrice = 5 * sales.reducedRemainingWeeks →
    totalTicketsSold sales = 25200 := by
  sorry

end total_tickets_sold_l2429_242918
