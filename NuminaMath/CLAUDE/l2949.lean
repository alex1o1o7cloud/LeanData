import Mathlib

namespace NUMINAMATH_CALUDE_min_value_f_and_sum_squares_l2949_294971

def f (x : ℝ) : ℝ := |x - 4| + |x - 3|

theorem min_value_f_and_sum_squares :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ (∃ (y : ℝ), f y = m) ∧ m = 1) ∧
  (∀ (a b c : ℝ), a + 2*b + 3*c = 1 → a^2 + b^2 + c^2 ≥ 1/14) ∧
  (∃ (a b c : ℝ), a + 2*b + 3*c = 1 ∧ a^2 + b^2 + c^2 = 1/14) := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_and_sum_squares_l2949_294971


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2949_294970

theorem geometric_sequence_fourth_term (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence condition
  q = 2 →                       -- Common ratio is 2
  a 1 * a 3 = 6 * a 2 →         -- Given condition
  a 4 = 24 :=                   -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2949_294970


namespace NUMINAMATH_CALUDE_middle_number_of_three_consecutive_squares_l2949_294962

theorem middle_number_of_three_consecutive_squares (n : ℕ) : 
  n^2 + (n+1)^2 + (n+2)^2 = 2030 → n + 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_of_three_consecutive_squares_l2949_294962


namespace NUMINAMATH_CALUDE_staircase_pencils_staircase_steps_l2949_294968

/-- Represents a staircase with n steps, where each step k has three segments of k pencils each. -/
def Staircase (n : ℕ) : ℕ := 3 * (n * (n + 1) / 2)

/-- Theorem stating that a staircase with 15 steps uses exactly 360 pencils. -/
theorem staircase_pencils : Staircase 15 = 360 := by
  sorry

/-- Theorem stating that if a staircase uses 360 pencils, it must have 15 steps. -/
theorem staircase_steps (n : ℕ) (h : Staircase n = 360) : n = 15 := by
  sorry

end NUMINAMATH_CALUDE_staircase_pencils_staircase_steps_l2949_294968


namespace NUMINAMATH_CALUDE_harry_fish_count_l2949_294992

/-- Given three friends with fish, prove Harry's fish count -/
theorem harry_fish_count (sam joe harry : ℕ) : 
  sam = 7 →
  joe = 8 * sam →
  harry = 4 * joe →
  harry = 224 := by
  sorry

end NUMINAMATH_CALUDE_harry_fish_count_l2949_294992


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l2949_294917

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (max_val : ℝ), max_val = 24 * Real.sqrt 3 ∧
  ∀ (w : ℂ), Complex.abs w = 2 →
    Complex.abs ((w - 2)^3 * (w + 2)) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l2949_294917


namespace NUMINAMATH_CALUDE_probability_same_color_is_15_364_l2949_294955

def total_marbles : ℕ := 14
def red_marbles : ℕ := 3
def white_marbles : ℕ := 4
def blue_marbles : ℕ := 5
def green_marbles : ℕ := 2

def probability_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) +
   green_marbles * (green_marbles - 1) * (green_marbles - 2)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2))

theorem probability_same_color_is_15_364 :
  probability_same_color = 15 / 364 := by sorry

end NUMINAMATH_CALUDE_probability_same_color_is_15_364_l2949_294955


namespace NUMINAMATH_CALUDE_number_divided_by_6_multiplied_by_12_l2949_294918

theorem number_divided_by_6_multiplied_by_12 :
  ∃ x : ℝ, (x / 6) * 12 = 15 ∧ x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_6_multiplied_by_12_l2949_294918


namespace NUMINAMATH_CALUDE_workshop_workers_l2949_294998

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 21

/-- Represents the number of technicians in the workshop -/
def num_technicians : ℕ := 7

/-- Represents the average salary of all workers in the workshop -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℕ := 12000

/-- Represents the average salary of non-technicians -/
def avg_salary_others : ℕ := 6000

/-- Theorem stating that the total number of workers is 21 -/
theorem workshop_workers :
  (total_workers * avg_salary_all = 
   num_technicians * avg_salary_technicians + 
   (total_workers - num_technicians) * avg_salary_others) →
  total_workers = 21 :=
by
  sorry


end NUMINAMATH_CALUDE_workshop_workers_l2949_294998


namespace NUMINAMATH_CALUDE_ferry_passengers_with_hats_l2949_294959

theorem ferry_passengers_with_hats (total_passengers : ℕ) 
  (percent_men : ℚ) (percent_women_with_hats : ℚ) (percent_men_with_hats : ℚ) :
  total_passengers = 1500 →
  percent_men = 2/5 →
  percent_women_with_hats = 3/20 →
  percent_men_with_hats = 3/25 →
  ∃ (total_with_hats : ℕ), total_with_hats = 207 :=
by
  sorry

end NUMINAMATH_CALUDE_ferry_passengers_with_hats_l2949_294959


namespace NUMINAMATH_CALUDE_graduating_class_size_l2949_294915

/-- The number of boys in the graduating class -/
def num_boys : ℕ := 208

/-- The difference between the number of girls and boys -/
def girl_boy_difference : ℕ := 69

/-- The total number of students in the graduating class -/
def total_students : ℕ := num_boys + (num_boys + girl_boy_difference)

theorem graduating_class_size :
  total_students = 485 :=
by sorry

end NUMINAMATH_CALUDE_graduating_class_size_l2949_294915


namespace NUMINAMATH_CALUDE_range_of_t_l2949_294914

/-- Given a set A containing 1 and a real number t, 
    the range of t is all real numbers except 1 -/
theorem range_of_t (t : ℝ) (A : Set ℝ) (h : A = {1, t}) : 
  {x : ℝ | x ≠ 1} = {x : ℝ | ∃ y ∈ A, y = x ∧ y ≠ 1} := by
sorry

end NUMINAMATH_CALUDE_range_of_t_l2949_294914


namespace NUMINAMATH_CALUDE_imaginary_unit_power_sum_l2949_294976

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_unit_power_sum : i^2 + i^4 = 0 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_sum_l2949_294976


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l2949_294979

theorem cubic_equation_real_root (k : ℝ) : ∃ x : ℝ, x^3 + 3*k*x^2 + 3*k^2*x + k^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_real_root_l2949_294979


namespace NUMINAMATH_CALUDE_acute_triangle_theorem_l2949_294991

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : a^2 = b^2 + c^2 - 2*b*c*Real.cos A

/-- Main theorem about the acute triangle -/
theorem acute_triangle_theorem (t : AcuteTriangle) :
  (Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A) →
  (t.c = Real.sqrt 7 ∧ t.a * t.b = 6) →
  (t.C = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7) :=
by sorry


end NUMINAMATH_CALUDE_acute_triangle_theorem_l2949_294991


namespace NUMINAMATH_CALUDE_cake_chord_length_squared_l2949_294949

theorem cake_chord_length_squared (d : ℝ) (n : ℕ) (l : ℝ) : 
  d = 18 → n = 4 → l = (d / 2) * Real.sqrt 2 → l^2 = 162 := by
  sorry

end NUMINAMATH_CALUDE_cake_chord_length_squared_l2949_294949


namespace NUMINAMATH_CALUDE_parrots_per_cage_l2949_294939

/-- Given a pet store with bird cages, prove the number of parrots in each cage. -/
theorem parrots_per_cage 
  (num_cages : ℕ) 
  (parakeets_per_cage : ℕ) 
  (total_birds : ℕ) 
  (h1 : num_cages = 9)
  (h2 : parakeets_per_cage = 6)
  (h3 : total_birds = 72) :
  (total_birds - num_cages * parakeets_per_cage) / num_cages = 2 := by
sorry

end NUMINAMATH_CALUDE_parrots_per_cage_l2949_294939


namespace NUMINAMATH_CALUDE_one_eighth_divided_by_one_fourth_l2949_294931

theorem one_eighth_divided_by_one_fourth (a b c : ℚ) : 
  a = 1/8 → b = 1/4 → c = a / b → c = 1/2 := by sorry

end NUMINAMATH_CALUDE_one_eighth_divided_by_one_fourth_l2949_294931


namespace NUMINAMATH_CALUDE_john_concert_probability_l2949_294902

theorem john_concert_probability
  (p_rain : ℝ)
  (p_john_if_rain : ℝ)
  (p_john_if_sunny : ℝ)
  (h_rain : p_rain = 0.50)
  (h_john_rain : p_john_if_rain = 0.30)
  (h_john_sunny : p_john_if_sunny = 0.90) :
  p_rain * p_john_if_rain + (1 - p_rain) * p_john_if_sunny = 0.60 :=
by sorry

end NUMINAMATH_CALUDE_john_concert_probability_l2949_294902


namespace NUMINAMATH_CALUDE_xyz_inequality_l2949_294921

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  Real.sqrt (1 + 8 * x) + Real.sqrt (1 + 8 * y) + Real.sqrt (1 + 8 * z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2949_294921


namespace NUMINAMATH_CALUDE_overall_percentage_correct_l2949_294937

theorem overall_percentage_correct (score1 score2 score3 : ℚ)
  (problems1 problems2 problems3 : ℕ) : 
  score1 = 75 / 100 ∧ 
  score2 = 85 / 100 ∧ 
  score3 = 60 / 100 ∧
  problems1 = 20 ∧
  problems2 = 50 ∧
  problems3 = 15 →
  (score1 * problems1 + score2 * problems2 + score3 * problems3) / 
  (problems1 + problems2 + problems3) = 79 / 100 := by
sorry

#eval (15 + 43 + 9) / (20 + 50 + 15)  -- Should evaluate to approximately 0.7882

end NUMINAMATH_CALUDE_overall_percentage_correct_l2949_294937


namespace NUMINAMATH_CALUDE_area_of_closed_figure_l2949_294903

noncomputable def f (x : ℝ) : ℝ := (4*x + 2) / ((x + 1) * (3*x + 1))

theorem area_of_closed_figure :
  ∫ x in (0)..(1), f x = (5/3) * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_closed_figure_l2949_294903


namespace NUMINAMATH_CALUDE_parallel_not_sufficient_nor_necessary_l2949_294919

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def parallel (m : Line) (α : Plane) : Prop := sorry
def perpendicular (m : Line) (β : Plane) : Prop := sorry
def planes_perpendicular (α β : Plane) : Prop := sorry

-- Theorem statement
theorem parallel_not_sufficient_nor_necessary 
  (m : Line) (α β : Plane) 
  (h_perp : planes_perpendicular α β) :
  ¬(∀ m α β, parallel m α → perpendicular m β) ∧ 
  ¬(∀ m α β, perpendicular m β → parallel m α) := by
  sorry

end NUMINAMATH_CALUDE_parallel_not_sufficient_nor_necessary_l2949_294919


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l2949_294928

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 2 * x - 1

-- Define the theorem
theorem y1_greater_than_y2 (y1 y2 : ℝ) 
  (h1 : line_equation (-3) y1) 
  (h2 : line_equation (-5) y2) : 
  y1 > y2 := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l2949_294928


namespace NUMINAMATH_CALUDE_no_valid_acute_triangle_l2949_294958

def is_valid_angle (α : ℕ) : Prop :=
  α % 10 = 0 ∧ α ≠ 30 ∧ α ≠ 60 ∧ α > 0 ∧ α < 90

def is_acute_triangle (α β γ : ℕ) : Prop :=
  α + β + γ = 180 ∧ α < 90 ∧ β < 90 ∧ γ < 90

theorem no_valid_acute_triangle :
  ¬ ∃ (α β γ : ℕ), is_valid_angle α ∧ is_valid_angle β ∧ is_valid_angle γ ∧
  is_acute_triangle α β γ ∧ α ≠ β ∧ β ≠ γ ∧ α ≠ γ :=
sorry

end NUMINAMATH_CALUDE_no_valid_acute_triangle_l2949_294958


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2949_294920

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0) : 
  (s + 2*y = 3*s) → (x + s = 3*s) → (x/y = 2) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2949_294920


namespace NUMINAMATH_CALUDE_glass_pane_impact_l2949_294938

/-- Represents a point inside a rectangle --/
structure ImpactPoint (width height : ℝ) where
  x : ℝ
  y : ℝ
  x_bound : 0 < x ∧ x < width
  y_bound : 0 < y ∧ y < height

/-- The glass pane problem --/
theorem glass_pane_impact
  (width : ℝ)
  (height : ℝ)
  (p : ImpactPoint width height)
  (h_width : width = 8)
  (h_height : height = 6)
  (h_right_area : p.x * height = 3 * (width - p.x) * height)
  (h_bottom_area : p.y * width = 2 * (height - p.y) * p.x) :
  p.x = 2 ∧ (width - p.x) = 6 ∧ p.y = 3 ∧ (height - p.y) = 3 := by
  sorry

end NUMINAMATH_CALUDE_glass_pane_impact_l2949_294938


namespace NUMINAMATH_CALUDE_truck_travel_distance_l2949_294951

/-- Given a truck that travels 150 miles on 5 gallons of diesel,
    prove that it can travel 210 miles on 7 gallons of diesel,
    assuming a constant rate of travel. -/
theorem truck_travel_distance 
  (initial_distance : ℝ) 
  (initial_fuel : ℝ) 
  (new_fuel : ℝ) 
  (h1 : initial_distance = 150) 
  (h2 : initial_fuel = 5) 
  (h3 : new_fuel = 7) :
  (initial_distance / initial_fuel) * new_fuel = 210 :=
by sorry

end NUMINAMATH_CALUDE_truck_travel_distance_l2949_294951


namespace NUMINAMATH_CALUDE_mike_car_payment_l2949_294906

def car_price : ℝ := 35000
def loan_amount : ℝ := 20000
def interest_rate : ℝ := 0.15
def loan_period : ℝ := 1

def total_amount_to_pay : ℝ := car_price + loan_amount * interest_rate * loan_period

theorem mike_car_payment : total_amount_to_pay = 38000 :=
by sorry

end NUMINAMATH_CALUDE_mike_car_payment_l2949_294906


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2949_294989

/-- The maximum value of y = -x^2 + 4x + 3, where x is a real number, is 7. -/
theorem max_value_quadratic :
  ∃ (y_max : ℝ), y_max = 7 ∧ ∀ (x : ℝ), -x^2 + 4*x + 3 ≤ y_max :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2949_294989


namespace NUMINAMATH_CALUDE_power_equation_solution_l2949_294982

theorem power_equation_solution (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^26 → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2949_294982


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l2949_294973

theorem binomial_coefficient_divisible_by_prime (p k : ℕ) :
  Nat.Prime p → 1 ≤ k → k < p →
  ∃ m : ℕ, Nat.choose p k = m * p := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisible_by_prime_l2949_294973


namespace NUMINAMATH_CALUDE_share_calculation_l2949_294950

theorem share_calculation (total : ℝ) (a b c : ℝ) 
  (h_total : total = 700)
  (h_a_b : a = (1/2) * b)
  (h_b_c : b = (1/2) * c)
  (h_sum : a + b + c = total) :
  c = 400 := by
sorry

end NUMINAMATH_CALUDE_share_calculation_l2949_294950


namespace NUMINAMATH_CALUDE_cosine_even_and_decreasing_l2949_294999

-- Define the properties of evenness and decreasing for a function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def IsDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

-- State the theorem
theorem cosine_even_and_decreasing :
  IsEven Real.cos ∧ IsDecreasingOn Real.cos 0 3 := by sorry

end NUMINAMATH_CALUDE_cosine_even_and_decreasing_l2949_294999


namespace NUMINAMATH_CALUDE_binomial_inequality_l2949_294961

theorem binomial_inequality (n k : ℕ) (h1 : n > k) (h2 : k > 0) : 
  (1 : ℝ) / (n + 1 : ℝ) * (n^n : ℝ) / ((k^k : ℝ) * ((n-k)^(n-k) : ℝ)) < 
  (n.factorial : ℝ) / ((k.factorial : ℝ) * ((n-k).factorial : ℝ)) ∧
  (n.factorial : ℝ) / ((k.factorial : ℝ) * ((n-k).factorial : ℝ)) < 
  (n^n : ℝ) / ((k^k : ℝ) * ((n-k)^(n-k) : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_inequality_l2949_294961


namespace NUMINAMATH_CALUDE_dan_found_two_dimes_l2949_294956

/-- The number of dimes Dan found -/
def dimes_found (barry_dimes dan_initial_dimes dan_final_dimes : ℕ) : ℕ :=
  dan_final_dimes - dan_initial_dimes

theorem dan_found_two_dimes :
  ∀ (barry_dimes dan_initial_dimes dan_final_dimes : ℕ),
    barry_dimes = 100 →
    dan_initial_dimes = barry_dimes / 2 →
    dan_final_dimes = 52 →
    dimes_found barry_dimes dan_initial_dimes dan_final_dimes = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_dan_found_two_dimes_l2949_294956


namespace NUMINAMATH_CALUDE_chicken_egg_production_roberto_chicken_problem_l2949_294929

/-- Represents the problem of determining the number of eggs each chicken needs to produce per week -/
theorem chicken_egg_production (num_chickens : ℕ) (chicken_cost : ℚ) (weekly_feed_cost : ℚ) 
  (eggs_per_dozen : ℕ) (dozen_cost : ℚ) (weeks : ℕ) : ℚ :=
  let total_chicken_cost := num_chickens * chicken_cost
  let total_feed_cost := weeks * weekly_feed_cost
  let total_chicken_expenses := total_chicken_cost + total_feed_cost
  let total_egg_expenses := weeks * dozen_cost
  let eggs_per_week := eggs_per_dozen
  (eggs_per_week / num_chickens : ℚ)

/-- Proves that each chicken needs to produce 3 eggs per week to be cheaper than buying eggs after 81 weeks -/
theorem roberto_chicken_problem : 
  chicken_egg_production 4 20 1 12 2 81 = 3 := by
  sorry

end NUMINAMATH_CALUDE_chicken_egg_production_roberto_chicken_problem_l2949_294929


namespace NUMINAMATH_CALUDE_translation_preserves_vector_translation_problem_l2949_294974

/-- A translation in 2D space -/
structure Translation (α : Type*) [Add α] :=
  (dx dy : α)

/-- Apply a translation to a point -/
def apply_translation {α : Type*} [Add α] (t : Translation α) (p : α × α) : α × α :=
  (p.1 + t.dx, p.2 + t.dy)

theorem translation_preserves_vector {α : Type*} [AddCommGroup α] 
  (t : Translation α) (a b c d : α × α) :
  apply_translation t a = c →
  apply_translation t b = d →
  c.1 - a.1 = d.1 - b.1 ∧ c.2 - a.2 = d.2 - b.2 :=
sorry

/-- The main theorem to prove -/
theorem translation_problem :
  ∃ (t : Translation ℤ),
    apply_translation t (-1, 4) = (3, 6) ∧
    apply_translation t (-3, 2) = (1, 4) :=
sorry

end NUMINAMATH_CALUDE_translation_preserves_vector_translation_problem_l2949_294974


namespace NUMINAMATH_CALUDE_salary_decrease_increase_l2949_294952

theorem salary_decrease_increase (original_salary : ℝ) (h : original_salary > 0) :
  let decreased_salary := original_salary * 0.5
  let final_salary := decreased_salary * 1.5
  final_salary = original_salary * 0.75 ∧ 
  (original_salary - final_salary) / original_salary = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_salary_decrease_increase_l2949_294952


namespace NUMINAMATH_CALUDE_equation_solution_l2949_294935

def f (x : ℝ) : ℝ := 2 * x - 3

theorem equation_solution :
  let d : ℝ := 4
  ∃ x : ℝ, 2 * (f x) - 21 = f (x - d) ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2949_294935


namespace NUMINAMATH_CALUDE_sixty_arrangements_l2949_294904

/-- Represents a hotel with a specified number of triple and double rooms. -/
structure Hotel where
  tripleRooms : ℕ
  doubleRooms : ℕ

/-- Represents a group of guests with a specified number of adults and children. -/
structure GuestGroup where
  adults : ℕ
  children : ℕ

/-- Calculates the number of ways to arrange accommodation for a group of guests in a hotel. -/
def accommodationArrangements (hotel : Hotel) (guests : GuestGroup) : ℕ :=
  sorry

/-- Theorem stating that for the given hotel and guest configuration, there are 60 different accommodation arrangements. -/
theorem sixty_arrangements :
  let hotel := { tripleRooms := 1, doubleRooms := 2 : Hotel }
  let guests := { adults := 3, children := 2 : GuestGroup }
  accommodationArrangements hotel guests = 60 :=
sorry

end NUMINAMATH_CALUDE_sixty_arrangements_l2949_294904


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l2949_294993

/-- Given a parabola and a hyperbola satisfying certain conditions, 
    prove their equations. -/
theorem parabola_hyperbola_equations :
  ∀ (a b : ℝ) (parabola hyperbola : ℝ → ℝ → Prop),
    a > 0 → b > 0 →
    (∀ x y, hyperbola x y ↔ x^2 / a^2 - y^2 / b^2 = 1) →
    (∃ f, f > 0 ∧ ∀ x y, parabola x y ↔ y^2 = 4 * f * x) →
    (∃ xf yf, hyperbola xf yf ∧ ∀ x y, parabola x y → (x - xf)^2 + y^2 = f^2) →
    parabola (3/2) (Real.sqrt 6) →
    hyperbola (3/2) (Real.sqrt 6) →
    (∀ x y, parabola x y ↔ y^2 = 4 * x) ∧
    (∀ x y, hyperbola x y ↔ 4 * x^2 - 4 * y^2 / 3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l2949_294993


namespace NUMINAMATH_CALUDE_doris_hourly_rate_l2949_294953

/-- Doris's hourly rate for babysitting -/
def hourly_rate : ℝ := 20

/-- Minimum amount Doris needs to earn in 3 weeks -/
def minimum_earnings : ℝ := 1200

/-- Number of hours Doris babysits on weekdays -/
def weekday_hours : ℝ := 3

/-- Number of hours Doris babysits on Saturdays -/
def saturday_hours : ℝ := 5

/-- Number of weekdays in a week -/
def weekdays_per_week : ℝ := 5

/-- Number of Saturdays in a week -/
def saturdays_per_week : ℝ := 1

/-- Number of weeks Doris needs to work to earn minimum_earnings -/
def weeks_to_earn : ℝ := 3

theorem doris_hourly_rate :
  hourly_rate = minimum_earnings / (weeks_to_earn * (weekdays_per_week * weekday_hours + saturdays_per_week * saturday_hours)) := by
  sorry

end NUMINAMATH_CALUDE_doris_hourly_rate_l2949_294953


namespace NUMINAMATH_CALUDE_fraction_equality_l2949_294984

theorem fraction_equality (P Q : ℤ) :
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ 4 →
    (P / (x + 3) + Q / (x^2 - 10*x + 16) = (x^2 - 6*x + 18) / (x^3 - 7*x^2 + 14*x - 48))) →
  (Q : ℚ) / P = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2949_294984


namespace NUMINAMATH_CALUDE_power_function_through_point_l2949_294941

/-- A power function that passes through (2, 2√2) and evaluates to 27 at x = 9 -/
theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x ^ a) →  -- f is a power function
  f 2 = 2 * Real.sqrt 2 →  -- f passes through (2, 2√2)
  f 9 = 27 :=  -- prove that f(9) = 27
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2949_294941


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_b_l2949_294925

theorem gcd_polynomial_and_b (b : ℤ) (h : ∃ k : ℤ, b = 350 * k) :
  Int.gcd (2 * b^3 + 3 * b^2 + 5 * b + 70) b = 70 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_b_l2949_294925


namespace NUMINAMATH_CALUDE_student_grades_l2949_294912

theorem student_grades (grade1 grade2 grade3 : ℚ) : 
  grade1 = 60 → 
  grade3 = 85 → 
  (grade1 + grade2 + grade3) / 3 = 75 → 
  grade2 = 80 := by
sorry

end NUMINAMATH_CALUDE_student_grades_l2949_294912


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2949_294940

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x - 1) / (x + 2) ≤ 3 ↔ x ∈ Set.Iic (-7) ∪ Set.Ioi (-2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2949_294940


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l2949_294965

theorem min_sum_of_squares (x y : ℝ) (h : 4 * x^2 + 5 * x * y + 4 * y^2 = 5) :
  ∃ (S_min : ℝ), S_min = 10/13 ∧ x^2 + y^2 ≥ S_min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l2949_294965


namespace NUMINAMATH_CALUDE_point_on_parabola_l2949_294922

theorem point_on_parabola (a : ℝ) : (a, -9) ∈ {(x, y) | y = -x^2} → (a = 3 ∨ a = -3) := by
  sorry

end NUMINAMATH_CALUDE_point_on_parabola_l2949_294922


namespace NUMINAMATH_CALUDE_total_pupils_count_l2949_294901

/-- The number of girls in the school -/
def num_girls : ℕ := 232

/-- The number of boys in the school -/
def num_boys : ℕ := 253

/-- The total number of pupils in the school -/
def total_pupils : ℕ := num_girls + num_boys

/-- Theorem: The total number of pupils in the school is 485 -/
theorem total_pupils_count : total_pupils = 485 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_count_l2949_294901


namespace NUMINAMATH_CALUDE_second_agency_per_mile_charge_l2949_294997

/-- The problem of determining the second agency's per-mile charge -/
theorem second_agency_per_mile_charge 
  (first_agency_daily : ℝ) 
  (first_agency_per_mile : ℝ) 
  (second_agency_daily : ℝ) 
  (crossover_miles : ℝ) :
  first_agency_daily = 20.25 →
  first_agency_per_mile = 0.14 →
  second_agency_daily = 18.25 →
  crossover_miles = 25.0 →
  ∃ (second_agency_per_mile : ℝ),
    first_agency_daily + first_agency_per_mile * crossover_miles = 
    second_agency_daily + second_agency_per_mile * crossover_miles ∧
    second_agency_per_mile = 0.22 := by
sorry

end NUMINAMATH_CALUDE_second_agency_per_mile_charge_l2949_294997


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l2949_294908

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x ∈ Set.Ioo 0 2, P x) ↔ (∀ x ∈ Set.Ioo 0 2, ¬P x) := by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x ∈ Set.Ioo 0 2, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x ∈ Set.Ioo 0 2, x^2 + 2*x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l2949_294908


namespace NUMINAMATH_CALUDE_sum_greater_than_two_l2949_294934

theorem sum_greater_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^7 > y^6) (h2 : y^7 > x^6) : x + y > 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_two_l2949_294934


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_range_l2949_294978

def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem cubic_polynomial_sum_range (a b c d : ℝ) (h_a : a ≠ 0) :
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 2 * f a b c d 2 = t ∧ 3 * f a b c d 3 = t ∧ 4 * f a b c d 4 = t) →
  (∃ y : ℝ, 0 < y ∧ y < 1 ∧ f a b c d 1 + f a b c d 5 = y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_range_l2949_294978


namespace NUMINAMATH_CALUDE_contact_lenses_sales_l2949_294985

/-- Proves that the total number of pairs of contact lenses sold is 11 given the problem conditions --/
theorem contact_lenses_sales (soft_price hard_price : ℕ) (soft_hard_diff total_sales : ℕ) :
  soft_price = 150 →
  hard_price = 85 →
  soft_hard_diff = 5 →
  total_sales = 1455 →
  ∃ (soft hard : ℕ),
    soft = hard + soft_hard_diff ∧
    soft_price * soft + hard_price * hard = total_sales ∧
    soft + hard = 11 := by
  sorry

end NUMINAMATH_CALUDE_contact_lenses_sales_l2949_294985


namespace NUMINAMATH_CALUDE_sin_210_deg_l2949_294977

/-- The sine of 210 degrees is equal to -1/2 --/
theorem sin_210_deg : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_deg_l2949_294977


namespace NUMINAMATH_CALUDE_total_area_of_fields_l2949_294905

/-- The total area of three fields with given dimensions -/
theorem total_area_of_fields (d₁ : ℝ) (l₂ w₂ : ℝ) (b₃ h₃ : ℝ) : 
  d₁ = 12 → l₂ = 15 → w₂ = 8 → b₃ = 18 → h₃ = 10 → 
  (d₁^2 / 2) + (l₂ * w₂) + (b₃ * h₃) = 372 := by
  sorry

end NUMINAMATH_CALUDE_total_area_of_fields_l2949_294905


namespace NUMINAMATH_CALUDE_problem_solution_l2949_294911

/-- Calculates the total number of new cans that can be made from a given number of cans,
    considering that newly made cans can also be recycled. -/
def totalNewCans (initialCans : ℕ) (damagedCans : ℕ) (requiredForNewCan : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions of the problem,
    the total number of new cans that can be made is 95. -/
theorem problem_solution :
  totalNewCans 500 20 6 = 95 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2949_294911


namespace NUMINAMATH_CALUDE_grasshopper_impossibility_l2949_294944

/-- A point in the 2D plane with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Check if a move from p1 to p2 is parallel to the line segment from p3 to p4 -/
def parallel_move (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x) * (p4.y - p3.y) = (p2.y - p1.y) * (p4.x - p3.x)

/-- A valid move in the grasshopper game -/
inductive ValidMove : List Point → List Point → Prop where
  | move (p1 p2 p3 p1' : Point) (rest : List Point) :
      parallel_move p1 p1' p2 p3 →
      ValidMove [p1, p2, p3] (p1' :: rest)

/-- A sequence of valid moves -/
def ValidMoveSequence : List Point → List Point → Prop :=
  Relation.ReflTransGen ValidMove

/-- The main theorem: impossibility of reaching the final configuration -/
theorem grasshopper_impossibility :
  ¬∃ (final : List Point),
    ValidMoveSequence [Point.mk 1 0, Point.mk 0 0, Point.mk 0 1] final ∧
    final = [Point.mk 0 0, Point.mk (-1) (-1), Point.mk 1 1] :=
sorry


end NUMINAMATH_CALUDE_grasshopper_impossibility_l2949_294944


namespace NUMINAMATH_CALUDE_investment_rate_calculation_l2949_294966

theorem investment_rate_calculation (total_investment : ℝ) (invested_at_18_percent : ℝ) (total_interest : ℝ) :
  total_investment = 22000 →
  invested_at_18_percent = 7000 →
  total_interest = 3360 →
  let remaining_investment := total_investment - invested_at_18_percent
  let interest_from_18_percent := invested_at_18_percent * 0.18
  let remaining_interest := total_interest - interest_from_18_percent
  let unknown_rate := remaining_interest / remaining_investment
  unknown_rate = 0.14 := by sorry

end NUMINAMATH_CALUDE_investment_rate_calculation_l2949_294966


namespace NUMINAMATH_CALUDE_greatest_valid_integer_l2949_294942

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 30 = 5

theorem greatest_valid_integer : 
  (∀ m : ℕ, is_valid m → m ≤ 145) ∧ is_valid 145 :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_integer_l2949_294942


namespace NUMINAMATH_CALUDE_min_swaps_to_reverse_l2949_294957

/-- Represents a strip of cells containing tokens -/
def Strip := Fin 100 → ℕ

/-- Reverses the order of tokens in the strip -/
def reverse (s : Strip) : Strip :=
  fun i => s (99 - i)

/-- Represents a swap operation -/
inductive Swap
  | adjacent : Fin 100 → Swap
  | free : Fin 96 → Swap

/-- Applies a swap operation to a strip -/
def applySwap (s : Strip) (swap : Swap) : Strip :=
  match swap with
  | Swap.adjacent i => 
      if i < 99 then
        fun j => if j = i then s (i+1) 
                 else if j = i+1 then s i
                 else s j
      else s
  | Swap.free i => 
      fun j => if j = i then s (i+4)
               else if j = i+4 then s i
               else s j

/-- A sequence of swap operations -/
def SwapSequence := List Swap

/-- Applies a sequence of swaps to a strip -/
def applySwaps (s : Strip) : SwapSequence → Strip
  | [] => s
  | (swap :: rest) => applySwaps (applySwap s swap) rest

/-- Counts the number of adjacent swaps in a sequence -/
def countAdjacentSwaps : SwapSequence → ℕ
  | [] => 0
  | (Swap.adjacent _ :: rest) => 1 + countAdjacentSwaps rest
  | (_ :: rest) => countAdjacentSwaps rest

/-- The main theorem: proving that 50 adjacent swaps are required to reverse the strip -/
theorem min_swaps_to_reverse (s : Strip) : 
  (∃ swaps : SwapSequence, applySwaps s swaps = reverse s) → 
  (∃ minSwaps : SwapSequence, 
    applySwaps s minSwaps = reverse s ∧ 
    countAdjacentSwaps minSwaps = 50 ∧
    ∀ swaps : SwapSequence, applySwaps s swaps = reverse s → 
      countAdjacentSwaps minSwaps ≤ countAdjacentSwaps swaps) :=
by sorry

end NUMINAMATH_CALUDE_min_swaps_to_reverse_l2949_294957


namespace NUMINAMATH_CALUDE_semicircle_circumference_from_rectangle_l2949_294960

/-- The circumference of a semicircle given rectangle dimensions --/
theorem semicircle_circumference_from_rectangle (l b : ℝ) (h1 : l = 24) (h2 : b = 16) :
  let rectangle_perimeter := 2 * (l + b)
  let square_side := rectangle_perimeter / 4
  let semicircle_circumference := π * square_side / 2 + square_side
  ‖semicircle_circumference - 51.40‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_circumference_from_rectangle_l2949_294960


namespace NUMINAMATH_CALUDE_opposite_boys_implies_total_l2949_294927

/-- Represents a circular arrangement of boys -/
structure CircularArrangement where
  num_boys : ℕ
  is_opposite : (a b : ℕ) → Prop

/-- The property that the 5th boy is opposite to the 20th boy -/
def fifth_opposite_twentieth (c : CircularArrangement) : Prop :=
  c.is_opposite 5 20

/-- Theorem stating that if the 5th boy is opposite to the 20th boy,
    then the total number of boys is 33 -/
theorem opposite_boys_implies_total (c : CircularArrangement) :
  fifth_opposite_twentieth c → c.num_boys = 33 := by
  sorry

end NUMINAMATH_CALUDE_opposite_boys_implies_total_l2949_294927


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l2949_294945

/-- Given a cylinder with volume 72π cm³, prove that a cone with the same radius
    and half the height has a volume of 12π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  π * r^2 * h = 72 * π →
  (1/3) * π * r^2 * (h/2) = 12 * π := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l2949_294945


namespace NUMINAMATH_CALUDE_twenty_paise_coins_count_l2949_294943

/-- Given a total of 344 coins consisting of 20 paise and 25 paise coins,
    with a total value of Rs. 71, prove that the number of 20 paise coins is 300. -/
theorem twenty_paise_coins_count :
  ∀ (x y : ℕ),
  x + y = 344 →
  20 * x + 25 * y = 7100 →
  x = 300 :=
by sorry

end NUMINAMATH_CALUDE_twenty_paise_coins_count_l2949_294943


namespace NUMINAMATH_CALUDE_arcsin_neg_half_equals_neg_pi_sixth_l2949_294990

theorem arcsin_neg_half_equals_neg_pi_sixth : 
  Real.arcsin (-1/2) = -π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_neg_half_equals_neg_pi_sixth_l2949_294990


namespace NUMINAMATH_CALUDE_binomial_sum_simplification_l2949_294969

theorem binomial_sum_simplification (n : ℕ) (p : ℝ) :
  (Finset.range (n + 1)).sum (λ k => k * (n.choose k) * p^k * (1 - p)^(n - k)) = n * p := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_simplification_l2949_294969


namespace NUMINAMATH_CALUDE_garrison_size_l2949_294932

/-- Given a garrison with provisions and reinforcements, calculate the initial number of men. -/
theorem garrison_size (initial_days : ℕ) (reinforcement_arrival : ℕ) (remaining_days : ℕ) (reinforcement_size : ℕ) : 
  initial_days = 54 →
  reinforcement_arrival = 15 →
  remaining_days = 20 →
  reinforcement_size = 1900 →
  ∃ (initial_men : ℕ), 
    initial_men * (initial_days - reinforcement_arrival) = 
    (initial_men + reinforcement_size) * remaining_days ∧
    initial_men = 2000 :=
by sorry

end NUMINAMATH_CALUDE_garrison_size_l2949_294932


namespace NUMINAMATH_CALUDE_max_abc_value_l2949_294907

theorem max_abc_value (a b c : ℝ) (sum_eq : a + b + c = 5) (prod_sum_eq : a * b + b * c + c * a = 7) :
  ∀ x y z : ℝ, x + y + z = 5 → x * y + y * z + z * x = 7 → a * b * c ≥ x * y * z ∧ ∃ p q r : ℝ, p + q + r = 5 ∧ p * q + q * r + r * p = 7 ∧ p * q * r = 3 :=
sorry

end NUMINAMATH_CALUDE_max_abc_value_l2949_294907


namespace NUMINAMATH_CALUDE_daughters_age_is_twelve_l2949_294964

/-- Proves that the daughter's age is 12 given the conditions about the father and daughter's ages -/
theorem daughters_age_is_twelve (D : ℕ) (F : ℕ) : 
  F = 3 * D →  -- Father's age is three times daughter's age this year
  F + 12 = 2 * (D + 12) →  -- After 12 years, father's age will be twice daughter's age
  D = 12 :=  -- Daughter's current age is 12
by
  sorry


end NUMINAMATH_CALUDE_daughters_age_is_twelve_l2949_294964


namespace NUMINAMATH_CALUDE_increasing_interval_of_sine_l2949_294946

theorem increasing_interval_of_sine (f : ℝ → ℝ) (h : f = λ x => Real.sin (2 * x + π / 6)) :
  ∀ x ∈ Set.Icc (-π / 3) (π / 6), ∀ y ∈ Set.Icc (-π / 3) (π / 6),
    x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_increasing_interval_of_sine_l2949_294946


namespace NUMINAMATH_CALUDE_area_greater_than_four_thirds_e_cubed_greater_than_twenty_l2949_294986

-- Define the area function S(t)
noncomputable def S (t : ℝ) : ℝ :=
  ∫ x in (0)..(1/t), (Real.exp (t^2 * x))

-- State the theorem
theorem area_greater_than_four_thirds :
  ∀ t > 0, S t > 4/3 :=
by
  sorry

-- Additional fact that can be used in the proof
theorem e_cubed_greater_than_twenty : Real.exp 3 > 20 :=
by
  sorry

end NUMINAMATH_CALUDE_area_greater_than_four_thirds_e_cubed_greater_than_twenty_l2949_294986


namespace NUMINAMATH_CALUDE_smallest_number_l2949_294930

theorem smallest_number : 
  -5 < -Real.pi ∧ -5 < -Real.sqrt 3 ∧ -5 < 0 := by sorry

end NUMINAMATH_CALUDE_smallest_number_l2949_294930


namespace NUMINAMATH_CALUDE_divisibility_implication_l2949_294909

theorem divisibility_implication (x y : ℤ) : 
  (∃ k : ℤ, 4*x - y = 3*k) → (∃ m : ℤ, 4*x^2 + 7*x*y - 2*y^2 = 9*m) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implication_l2949_294909


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_6_pow_5_l2949_294981

theorem units_digit_of_7_pow_6_pow_5 : 7^(6^5) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_6_pow_5_l2949_294981


namespace NUMINAMATH_CALUDE_some_number_value_l2949_294924

theorem some_number_value (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = some_number * 35 * 45 * 35) : 
  some_number = 1 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2949_294924


namespace NUMINAMATH_CALUDE_parallel_line_theorem_perpendicular_lines_theorem_l2949_294947

-- Define the line l1
def l1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the parallel line l2
def l2_parallel (x y : ℝ) : Prop := 3 * x + 4 * y - 9 = 0

-- Define the perpendicular lines l2
def l2_perp_pos (x y : ℝ) : Prop := 4 * x - 3 * y + 4 * Real.sqrt 6 = 0
def l2_perp_neg (x y : ℝ) : Prop := 4 * x - 3 * y - 4 * Real.sqrt 6 = 0

-- Theorem for parallel line
theorem parallel_line_theorem :
  (∀ x y, l2_parallel x y ↔ ∃ k, 3 * x + 4 * y = k) ∧
  l2_parallel (-1) 3 := by sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines_theorem :
  (∀ x y, (l2_perp_pos x y ∨ l2_perp_neg x y) → 
    (3 * 4 + 4 * (-3) = 0)) ∧
  (∀ x y, l2_perp_pos x y → 
    (1/2 * |x| * |y| = 4 ∧ 4 * x = 0 → y = Real.sqrt 6 ∧ 3 * y = 0 → x = 4/3 * Real.sqrt 6)) ∧
  (∀ x y, l2_perp_neg x y → 
    (1/2 * |x| * |y| = 4 ∧ 4 * x = 0 → y = Real.sqrt 6 ∧ 3 * y = 0 → x = 4/3 * Real.sqrt 6)) := by sorry

end NUMINAMATH_CALUDE_parallel_line_theorem_perpendicular_lines_theorem_l2949_294947


namespace NUMINAMATH_CALUDE_remaining_three_digit_numbers_l2949_294963

/-- The number of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The number of three-digit numbers where the first and last digits are the same
    but the middle digit is different -/
def excluded_numbers : ℕ := 81

/-- The number of valid three-digit numbers after exclusion -/
def valid_numbers : ℕ := total_three_digit_numbers - excluded_numbers

theorem remaining_three_digit_numbers : valid_numbers = 819 := by
  sorry

end NUMINAMATH_CALUDE_remaining_three_digit_numbers_l2949_294963


namespace NUMINAMATH_CALUDE_unique_prime_perfect_square_l2949_294954

theorem unique_prime_perfect_square : 
  ∃! p : ℕ, Prime p ∧ ∃ n : ℕ, 5^p + 4*p^4 = n^2 ∧ p = 31 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_perfect_square_l2949_294954


namespace NUMINAMATH_CALUDE_negation_equivalence_l2949_294910

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^3 > 0) ↔ (∀ x : ℝ, x^3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2949_294910


namespace NUMINAMATH_CALUDE_ellipse_major_minor_distance_l2949_294975

/-- An ellipse with equation 4(x+2)^2 + 16y^2 = 64 -/
structure Ellipse where
  eq : ∀ x y : ℝ, 4 * (x + 2)^2 + 16 * y^2 = 64

/-- Point C is an endpoint of the major axis -/
def C (e : Ellipse) : ℝ × ℝ := sorry

/-- Point D is an endpoint of the minor axis -/
def D (e : Ellipse) : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_major_minor_distance (e : Ellipse) : 
  distance (C e) (D e) = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_minor_distance_l2949_294975


namespace NUMINAMATH_CALUDE_zoe_pool_cleaning_earnings_l2949_294936

/-- Represents Zoe's earnings and babysitting information --/
structure ZoeEarnings where
  total : ℕ
  zacharyRate : ℕ
  julieRate : ℕ
  chloeRate : ℕ
  zacharyEarnings : ℕ

/-- Calculates Zoe's earnings from pool cleaning --/
def poolCleaningEarnings (z : ZoeEarnings) : ℕ :=
  let zacharyHours := z.zacharyEarnings / z.zacharyRate
  let chloeHours := zacharyHours * 5
  let julieHours := zacharyHours * 3
  let babysittingEarnings := 
    zacharyHours * z.zacharyRate + 
    chloeHours * z.chloeRate + 
    julieHours * z.julieRate
  z.total - babysittingEarnings

/-- Theorem stating that Zoe's pool cleaning earnings are $5,200 --/
theorem zoe_pool_cleaning_earnings :
  poolCleaningEarnings {
    total := 8000,
    zacharyRate := 15,
    julieRate := 10,
    chloeRate := 5,
    zacharyEarnings := 600
  } = 5200 := by
  sorry

end NUMINAMATH_CALUDE_zoe_pool_cleaning_earnings_l2949_294936


namespace NUMINAMATH_CALUDE_sum_of_gcd_values_l2949_294923

theorem sum_of_gcd_values (n : ℕ+) : 
  (Finset.sum (Finset.range 4) (λ i => (Nat.gcd (5 * n + 6) n).succ)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_gcd_values_l2949_294923


namespace NUMINAMATH_CALUDE_smallest_possible_median_l2949_294995

def number_set (x : ℤ) : Finset ℤ := {x, 3*x, 4, 3, 7}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧
  2 * (s.filter (· ≥ m)).card ≥ s.card

theorem smallest_possible_median :
  ∀ x : ℤ, ∃ m : ℤ, is_median m (number_set x) ∧ m = 3 ∧ 
  ∀ m' : ℤ, is_median m' (number_set x) → m ≤ m' :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_median_l2949_294995


namespace NUMINAMATH_CALUDE_common_tangents_count_l2949_294980

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 10*y + 13 = 0

-- Define a function to count common tangents
def count_common_tangents (C1 C2 : ℝ → ℝ → Prop) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents C1 C2 = 1 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l2949_294980


namespace NUMINAMATH_CALUDE_polynomial_sum_equals_256_l2949_294933

theorem polynomial_sum_equals_256 
  (a a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, (3 - x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) :
  a - a₁ + a₂ - a₃ + a₄ = 256 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_equals_256_l2949_294933


namespace NUMINAMATH_CALUDE_reasonable_prize_distribution_l2949_294994

/-- The most reasonable prize distribution for a math competition problem --/
theorem reasonable_prize_distribution
  (total_prize : ℝ)
  (prob_A : ℝ)
  (prob_B : ℝ)
  (h_total : total_prize = 190)
  (h_prob_A : prob_A = 3/4)
  (h_prob_B : prob_B = 4/5)
  (h_prob_valid : 0 ≤ prob_A ∧ prob_A ≤ 1 ∧ 0 ≤ prob_B ∧ prob_B ≤ 1) :
  let expected_A := (prob_A * (1 - prob_B) * total_prize + prob_A * prob_B * (total_prize / 2))
  let expected_B := (prob_B * (1 - prob_A) * total_prize + prob_A * prob_B * (total_prize / 2))
  expected_A = 90 ∧ expected_B = 100 :=
by sorry


end NUMINAMATH_CALUDE_reasonable_prize_distribution_l2949_294994


namespace NUMINAMATH_CALUDE_existence_implies_range_l2949_294988

theorem existence_implies_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ Real.exp x * (x - a) < 1) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_existence_implies_range_l2949_294988


namespace NUMINAMATH_CALUDE_second_smallest_radius_l2949_294987

/-- A configuration of four circles tangent to each other and two parallel lines -/
structure CircleConfiguration where
  radii : Fin 4 → ℝ
  tangent_to_lines : Bool
  tangent_to_each_other : Bool

/-- The radii form a geometric sequence -/
def is_geometric_sequence (c : CircleConfiguration) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 3, c.radii i.succ = c.radii i * r

theorem second_smallest_radius 
  (c : CircleConfiguration)
  (h1 : c.tangent_to_lines)
  (h2 : c.tangent_to_each_other)
  (h3 : is_geometric_sequence c)
  (h4 : c.radii 0 = 5)
  (h5 : c.radii 3 = 20) :
  c.radii 1 = 10 := by
sorry

end NUMINAMATH_CALUDE_second_smallest_radius_l2949_294987


namespace NUMINAMATH_CALUDE_equation_identity_l2949_294967

theorem equation_identity (a b c x : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  c * ((x - a) * (x - b)) / ((c - a) * (c - b)) +
  b * ((x - a) * (x - c)) / ((b - a) * (b - c)) +
  a * ((x - b) * (x - c)) / ((a - b) * (a - c)) = x := by
  sorry

end NUMINAMATH_CALUDE_equation_identity_l2949_294967


namespace NUMINAMATH_CALUDE_roots_equation_m_value_l2949_294972

theorem roots_equation_m_value (α : ℝ) (m : ℝ) : 
  (∀ x, x^2 + 3*x + m = 0 ↔ x = 1/Real.cos α ∨ x = Real.tan α) →
  m = 20/9 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_m_value_l2949_294972


namespace NUMINAMATH_CALUDE_franks_final_score_l2949_294948

/-- Calculates the final score in a trivia competition given the number of correct and incorrect answers in each half. -/
def final_score (first_half_correct first_half_incorrect second_half_correct second_half_incorrect : ℕ) : ℤ :=
  let points_per_correct : ℤ := 3
  let points_per_incorrect : ℤ := -1
  (first_half_correct * points_per_correct + first_half_incorrect * points_per_incorrect) +
  (second_half_correct * points_per_correct + second_half_incorrect * points_per_incorrect)

/-- Theorem stating that Frank's final score in the trivia competition is 39 points. -/
theorem franks_final_score :
  final_score 6 4 10 5 = 39 := by
  sorry

end NUMINAMATH_CALUDE_franks_final_score_l2949_294948


namespace NUMINAMATH_CALUDE_apple_pile_count_l2949_294900

-- Define the initial number of apples
def initial_apples : ℕ := 8

-- Define the number of apples added
def added_apples : ℕ := 5

-- Theorem to prove
theorem apple_pile_count : initial_apples + added_apples = 13 := by
  sorry

end NUMINAMATH_CALUDE_apple_pile_count_l2949_294900


namespace NUMINAMATH_CALUDE_price_changes_l2949_294983

/-- The original price of an item that, after a 5% decrease and a 40% increase,
    results in a price $1352.06 less than twice the original price. -/
def original_price : ℝ := 2018

theorem price_changes (x : ℝ) (hx : x = original_price) :
  let price_after_decrease := 0.95 * x
  let price_after_increase := price_after_decrease * 1.4
  price_after_increase = 2 * x - 1352.06 := by sorry

end NUMINAMATH_CALUDE_price_changes_l2949_294983


namespace NUMINAMATH_CALUDE_largest_x_satisfying_equation_l2949_294926

theorem largest_x_satisfying_equation : 
  ∃ x : ℚ, x = 3/25 ∧ 
  (∀ y : ℚ, y ≥ 0 → Real.sqrt (3 * y) = 5 * y → y ≤ x) ∧
  Real.sqrt (3 * x) = 5 * x :=
by sorry

end NUMINAMATH_CALUDE_largest_x_satisfying_equation_l2949_294926


namespace NUMINAMATH_CALUDE_smallest_repetition_for_divisibility_l2949_294913

/-- The sum of digits of 2013 -/
def sum_digits_2013 : ℕ := 2 + 0 + 1 + 3

/-- Function to check if a number is divisible by 9 -/
def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

/-- The concatenated number when 2013 is repeated n times -/
def repeated_2013 (n : ℕ) : ℕ := 
  let digits := sum_digits_2013 * n
  digits

/-- The smallest positive integer n such that the number formed by 
    concatenating 2013 n times is divisible by 9 is equal to 3 -/
theorem smallest_repetition_for_divisibility : 
  (∃ n : ℕ, n > 0 ∧ is_divisible_by_9 (repeated_2013 n)) ∧ 
  (∀ k : ℕ, k > 0 → is_divisible_by_9 (repeated_2013 k) → k ≥ 3) ∧
  is_divisible_by_9 (repeated_2013 3) :=
sorry

end NUMINAMATH_CALUDE_smallest_repetition_for_divisibility_l2949_294913


namespace NUMINAMATH_CALUDE_circle_area_with_special_condition_l2949_294996

theorem circle_area_with_special_condition (r : ℝ) (h : r > 0) :
  (5 : ℝ) * (1 / (2 * Real.pi * r)) = r / 2 → π * r^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_special_condition_l2949_294996


namespace NUMINAMATH_CALUDE_unused_ribbon_theorem_l2949_294916

/-- Represents the pattern of ribbon pieces -/
inductive RibbonPiece
  | two
  | four
  | six
  | eight
  | ten

/-- Returns the length of a ribbon piece in meters -/
def piece_length (p : RibbonPiece) : ℕ :=
  match p with
  | .two => 2
  | .four => 4
  | .six => 6
  | .eight => 8
  | .ten => 10

/-- Represents the pattern of ribbon usage -/
def ribbon_pattern : List RibbonPiece :=
  [.two, .two, .two, .four, .four, .six, .six, .six, .six, .eight, .ten, .ten]

/-- Calculates the unused ribbon length after following the pattern once -/
def unused_ribbon (total_length : ℕ) (pattern : List RibbonPiece) : ℕ :=
  let used := pattern.foldl (fun acc p => acc + piece_length p) 0
  total_length - (used % total_length)

theorem unused_ribbon_theorem :
  unused_ribbon 30 ribbon_pattern = 4 := by sorry

#eval unused_ribbon 30 ribbon_pattern

end NUMINAMATH_CALUDE_unused_ribbon_theorem_l2949_294916
