import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l170_17089

theorem expression_evaluation : (28 + 48 / 69) * 69 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l170_17089


namespace NUMINAMATH_CALUDE_total_cost_is_44_l170_17046

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda in dollars -/
def soda_cost : ℕ := 3

/-- The cost of a single cookie in dollars -/
def cookie_cost : ℕ := 1

/-- The number of sandwiches to purchase -/
def num_sandwiches : ℕ := 4

/-- The number of sodas to purchase -/
def num_sodas : ℕ := 6

/-- The number of cookies to purchase -/
def num_cookies : ℕ := 10

/-- Theorem stating that the total cost of the purchase is $44 -/
theorem total_cost_is_44 :
  num_sandwiches * sandwich_cost + num_sodas * soda_cost + num_cookies * cookie_cost = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_44_l170_17046


namespace NUMINAMATH_CALUDE_unit_square_tiling_l170_17086

/-- A rectangle is considered "good" if it can be tiled by rectangles similar to 1 × (3 + ∛3) -/
def is_good (a b : ℝ) : Prop := sorry

/-- The scaling property of good rectangles -/
axiom good_scale (a b c : ℝ) (h : c > 0) :
  is_good a b → is_good (a * c) (b * c)

/-- The integer multiple property of good rectangles -/
axiom good_int_multiple (m n : ℝ) (j : ℕ) (h : j > 0) :
  is_good m n → is_good m (n * j)

/-- The main theorem: the unit square can be tiled with rectangles similar to 1 × (3 + ∛3) -/
theorem unit_square_tiling :
  ∃ (tiling : Set (ℝ × ℝ)), 
    (∀ (rect : ℝ × ℝ), rect ∈ tiling → is_good rect.1 rect.2) ∧
    (∃ (f : ℝ × ℝ → ℝ × ℝ), 
      (∀ x y, f (x, y) = (x, y)) ∧
      (∀ (rect : ℝ × ℝ), rect ∈ tiling → 
        ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ f (rect.1, rect.2) = (a, b) ∧ 
        b / a = 3 + Real.rpow 3 (1/3 : ℝ))) :=
sorry

end NUMINAMATH_CALUDE_unit_square_tiling_l170_17086


namespace NUMINAMATH_CALUDE_detergent_per_pound_l170_17081

/-- Given that Mrs. Hilt used 18 ounces of detergent to wash 9 pounds of clothes,
    prove that she uses 2 ounces of detergent per pound of clothes. -/
theorem detergent_per_pound (total_detergent : ℝ) (total_clothes : ℝ) 
  (h1 : total_detergent = 18) 
  (h2 : total_clothes = 9) : 
  total_detergent / total_clothes = 2 := by
sorry

end NUMINAMATH_CALUDE_detergent_per_pound_l170_17081


namespace NUMINAMATH_CALUDE_parallel_line_plane_l170_17047

-- Define the types for lines and planes
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- Theorem statement
theorem parallel_line_plane 
  (α β : Plane) (n : Line) 
  (h1 : parallel α β) 
  (h2 : subset n α) : 
  line_parallel n β :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_l170_17047


namespace NUMINAMATH_CALUDE_cyclist_speed_ratio_l170_17032

theorem cyclist_speed_ratio :
  ∀ (T₁ T₂ o₁ o₂ : ℝ),
  T₁ > 0 ∧ T₂ > 0 ∧ o₁ > 0 ∧ o₂ > 0 →
  o₁ + T₁ = o₂ + T₂ →
  T₁ = 2 * o₂ →
  T₂ = 4 * o₁ →
  T₁ / T₂ = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_ratio_l170_17032


namespace NUMINAMATH_CALUDE_log_problem_l170_17062

theorem log_problem (y : ℝ) (k : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 125 / Real.log 2 = k * y) → 
  k = 9 := by
sorry

end NUMINAMATH_CALUDE_log_problem_l170_17062


namespace NUMINAMATH_CALUDE_class_size_problem_l170_17022

theorem class_size_problem (n : ℕ) 
  (h1 : 20 ≤ n ∧ n ≤ 30) 
  (h2 : ∃ x y : ℕ, x < n ∧ y < n ∧ x ≠ y ∧ 2 * x + 1 = n - x ∧ 3 * y + 1 = n - y) :
  n = 25 := by
sorry

end NUMINAMATH_CALUDE_class_size_problem_l170_17022


namespace NUMINAMATH_CALUDE_keychain_cost_decrease_l170_17068

theorem keychain_cost_decrease (P : ℝ) : 
  P > 0 →                           -- Selling price is positive
  P - 50 = 0.5 * P →                -- New profit is 50% of selling price
  P - 0.75 * P = 0.25 * P →         -- Initial profit was 25% of selling price
  0.75 * P = 75 :=                  -- Initial cost was $75
by
  sorry

end NUMINAMATH_CALUDE_keychain_cost_decrease_l170_17068


namespace NUMINAMATH_CALUDE_combined_mpg_l170_17085

/-- Combined rate of miles per gallon for two cars -/
theorem combined_mpg (ray_mpg tom_mpg ray_miles tom_miles : ℚ) :
  ray_mpg = 50 →
  tom_mpg = 25 →
  ray_miles = 100 →
  tom_miles = 200 →
  (ray_miles + tom_miles) / (ray_miles / ray_mpg + tom_miles / tom_mpg) = 30 := by
  sorry


end NUMINAMATH_CALUDE_combined_mpg_l170_17085


namespace NUMINAMATH_CALUDE_right_angled_triangle_l170_17090

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem statement
theorem right_angled_triangle (abc : Triangle) 
  (h : Real.sin abc.A = Real.sin abc.C * Real.cos abc.B) : 
  abc.C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l170_17090


namespace NUMINAMATH_CALUDE_negation_of_existence_is_universal_nonequality_l170_17020

theorem negation_of_existence_is_universal_nonequality :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔ (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_universal_nonequality_l170_17020


namespace NUMINAMATH_CALUDE_right_triangle_altitude_l170_17095

theorem right_triangle_altitude (a b c m : ℝ) (h_positive : a > 0) : 
  b^2 + c^2 = a^2 →   -- Pythagorean theorem
  m^2 = (b - c)^2 →   -- Difference of legs equals altitude
  b * c = a * m →     -- Area relation
  m = (a * (Real.sqrt 5 - 1)) / 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_l170_17095


namespace NUMINAMATH_CALUDE_periodic_even_function_theorem_l170_17065

def periodic_even_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 2) = f x) ∧ 
  (∀ x, f (-x) = f x)

theorem periodic_even_function_theorem (f : ℝ → ℝ) 
  (h_periodic_even : periodic_even_function f)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
  sorry

end NUMINAMATH_CALUDE_periodic_even_function_theorem_l170_17065


namespace NUMINAMATH_CALUDE_theater_occupancy_l170_17073

theorem theater_occupancy (total_seats empty_seats : ℕ) 
  (h1 : total_seats = 750) 
  (h2 : empty_seats = 218) : 
  total_seats - empty_seats = 532 := by
  sorry

end NUMINAMATH_CALUDE_theater_occupancy_l170_17073


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l170_17052

theorem complex_power_magnitude : 
  Complex.abs ((2 : ℂ) + (2 * Complex.I * Real.sqrt 3))^4 = 256 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l170_17052


namespace NUMINAMATH_CALUDE_first_podium_height_calculation_l170_17037

/-- The height of the second prize podium in centimeters -/
def second_podium_height : ℚ := 53 + 7 / 10

/-- Hyeonjoo's measured height on the second prize podium in centimeters -/
def height_on_second_podium : ℚ := 190

/-- Hyeonjoo's measured height on the first prize podium in centimeters -/
def height_on_first_podium : ℚ := 232 + 5 / 10

/-- The height of the first prize podium in centimeters -/
def first_podium_height : ℚ := height_on_first_podium - (height_on_second_podium - second_podium_height)

theorem first_podium_height_calculation :
  first_podium_height = 96.2 := by sorry

end NUMINAMATH_CALUDE_first_podium_height_calculation_l170_17037


namespace NUMINAMATH_CALUDE_correct_addition_result_l170_17057

theorem correct_addition_result 
  (correct_addend : ℕ)
  (mistaken_addend : ℕ)
  (other_addend : ℕ)
  (mistaken_result : ℕ)
  (h1 : correct_addend = 420)
  (h2 : mistaken_addend = 240)
  (h3 : mistaken_result = mistaken_addend + other_addend)
  : correct_addend + other_addend = 570 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_addition_result_l170_17057


namespace NUMINAMATH_CALUDE_parabola_focus_l170_17040

/-- The focus of a parabola y = ax^2 (a ≠ 0) is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  let parabola := {(x, y) : ℝ × ℝ | y = a * x^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (0, 1 / (4 * a)) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l170_17040


namespace NUMINAMATH_CALUDE_max_gcd_sum_780_l170_17012

theorem max_gcd_sum_780 :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = 780 ∧
  ∀ (a b : ℕ), a > 0 → b > 0 → a + b = 780 → Nat.gcd x y ≥ Nat.gcd a b :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_sum_780_l170_17012


namespace NUMINAMATH_CALUDE_fraction_simplification_l170_17013

theorem fraction_simplification :
  (2 - 4 + 8 - 16 + 32 - 64 + 128 - 256) / (4 - 8 + 16 - 32 + 64 - 128 + 256 - 512) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l170_17013


namespace NUMINAMATH_CALUDE_negation_of_implication_l170_17007

theorem negation_of_implication (a : ℝ) :
  ¬(a > 1 → a^2 > 1) ↔ (a ≤ 1 → a^2 ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l170_17007


namespace NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l170_17005

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l170_17005


namespace NUMINAMATH_CALUDE_product_value_l170_17074

theorem product_value : 
  (1 / 4 : ℚ) * 8 * (1 / 16 : ℚ) * 32 * (1 / 64 : ℚ) * 128 * (1 / 256 : ℚ) * 512 * (1 / 1024 : ℚ) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_value_l170_17074


namespace NUMINAMATH_CALUDE_pen_cost_is_30_l170_17008

-- Define the daily expenditures
def daily_expenditures : List ℝ := [450, 600, 400, 500, 550, 300]

-- Define the mean expenditure
def mean_expenditure : ℝ := 500

-- Define the number of days
def num_days : ℕ := 7

-- Define the cost of the notebook
def notebook_cost : ℝ := 50

-- Define the cost of the earphone
def earphone_cost : ℝ := 620

-- Theorem to prove
theorem pen_cost_is_30 :
  let total_week_expenditure := mean_expenditure * num_days
  let total_other_days := daily_expenditures.sum
  let friday_expenditure := total_week_expenditure - total_other_days
  friday_expenditure - (notebook_cost + earphone_cost) = 30 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_is_30_l170_17008


namespace NUMINAMATH_CALUDE_feet_to_inches_conversion_l170_17016

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

/-- Initial height of the tree in feet -/
def initial_height : ℕ := 52

/-- Annual growth of the tree in feet -/
def annual_growth : ℕ := 5

/-- Time period in years -/
def time_period : ℕ := 8

/-- Theorem stating that the conversion factor from feet to inches is 12 -/
theorem feet_to_inches_conversion :
  feet_to_inches = 12 :=
sorry

end NUMINAMATH_CALUDE_feet_to_inches_conversion_l170_17016


namespace NUMINAMATH_CALUDE_calculation_proof_l170_17092

theorem calculation_proof : 
  (3 + 1 / 117) * (4 + 1 / 119) - (1 + 116 / 117) * (5 + 118 / 119) - 5 / 119 = 10 / 117 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l170_17092


namespace NUMINAMATH_CALUDE_decimal_524_to_octal_l170_17080

-- Define a function to convert decimal to octal
def decimalToOctal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec helper (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else helper (m / 8) ((m % 8) :: acc)
    helper n []

-- Theorem statement
theorem decimal_524_to_octal :
  decimalToOctal 524 = [1, 0, 1, 4] := by sorry

end NUMINAMATH_CALUDE_decimal_524_to_octal_l170_17080


namespace NUMINAMATH_CALUDE_point_above_line_l170_17041

/-- A point (x, y) is above a line ax + by + c = 0 if ax + by + c < 0 -/
def IsAboveLine (x y a b c : ℝ) : Prop := a * x + b * y + c < 0

theorem point_above_line (t : ℝ) :
  IsAboveLine (-2) t 1 (-2) 4 → t > 1 := by
  sorry

end NUMINAMATH_CALUDE_point_above_line_l170_17041


namespace NUMINAMATH_CALUDE_smallest_lcm_for_four_digit_gcd_five_l170_17076

theorem smallest_lcm_for_four_digit_gcd_five (m n : ℕ) : 
  m ≥ 1000 ∧ m ≤ 9999 ∧ n ≥ 1000 ∧ n ≤ 9999 ∧ Nat.gcd m n = 5 →
  Nat.lcm m n ≥ 203010 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_for_four_digit_gcd_five_l170_17076


namespace NUMINAMATH_CALUDE_vacation_cost_l170_17091

/-- The total cost of a vacation satisfying specific conditions -/
theorem vacation_cost : ∃ (C P : ℝ), 
  C = 5 * P ∧ 
  C = 7 * (P - 40) ∧ 
  C = 8 * (P - 60) ∧ 
  C = 700 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l170_17091


namespace NUMINAMATH_CALUDE_kickball_players_l170_17067

theorem kickball_players (wednesday : ℕ) (thursday : ℕ) (difference : ℕ) : 
  wednesday = 37 →
  difference = 9 →
  thursday = wednesday - difference →
  wednesday + thursday = 65 := by
sorry

end NUMINAMATH_CALUDE_kickball_players_l170_17067


namespace NUMINAMATH_CALUDE_gcd_problem_l170_17072

theorem gcd_problem (x : ℤ) (h : ∃ k : ℤ, x = 17248 * k) :
  Int.gcd ((5*x+4)*(8*x+1)*(11*x+6)*(3*x+9)) x = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l170_17072


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l170_17088

theorem expression_simplification_and_evaluation :
  let x : ℝ := 3 * Real.cos (60 * π / 180)
  let original_expression := (2 * x) / (x + 1) - (2 * x - 4) / (x^2 - 1) / ((x - 2) / (x^2 - 2 * x + 1))
  let simplified_expression := 4 / (x + 1)
  original_expression = simplified_expression ∧ simplified_expression = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l170_17088


namespace NUMINAMATH_CALUDE_red_blood_cell_surface_area_calculation_l170_17098

/-- The sum of the surface areas of all red blood cells in a normal adult body. -/
def red_blood_cell_surface_area (body_surface_area : ℝ) : ℝ :=
  2000 * body_surface_area

/-- Theorem: The sum of the surface areas of all red blood cells in an adult body
    with a body surface area of 1800 cm² is 3.6 × 10⁶ cm². -/
theorem red_blood_cell_surface_area_calculation :
  red_blood_cell_surface_area 1800 = 3.6 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_red_blood_cell_surface_area_calculation_l170_17098


namespace NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_nth_term_is_298_implies_n_is_100_l170_17009

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  1 + 3 * (n - 1)

/-- Theorem stating that the 100th term of the sequence is 298 -/
theorem arithmetic_sequence_100th_term :
  arithmetic_sequence 100 = 298 :=
sorry

/-- Theorem proving that when the nth term is 298, n must be 100 -/
theorem nth_term_is_298_implies_n_is_100 (n : ℕ) :
  arithmetic_sequence n = 298 → n = 100 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_nth_term_is_298_implies_n_is_100_l170_17009


namespace NUMINAMATH_CALUDE_area_of_inscribing_square_l170_17036

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 4*y = 12

/-- The circle is inscribed in a square with sides parallel to y-axis -/
axiom inscribed_in_square : ∃ (side : ℝ), ∀ (x y : ℝ), 
  circle_equation x y → (0 ≤ x ∧ x ≤ side) ∧ (0 ≤ y ∧ y ≤ side)

/-- The area of the square inscribing the circle -/
def square_area : ℝ := 68

/-- Theorem: The area of the square inscribing the circle is 68 square units -/
theorem area_of_inscribing_square : 
  ∃ (side : ℝ), (∀ (x y : ℝ), circle_equation x y → 
    (0 ≤ x ∧ x ≤ side) ∧ (0 ≤ y ∧ y ≤ side)) ∧ side^2 = square_area := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribing_square_l170_17036


namespace NUMINAMATH_CALUDE_ice_cream_yogurt_cost_difference_l170_17039

def ice_cream_cartons : ℕ := 20
def yogurt_cartons : ℕ := 2
def ice_cream_price : ℕ := 6
def yogurt_price : ℕ := 1

theorem ice_cream_yogurt_cost_difference :
  ice_cream_cartons * ice_cream_price - yogurt_cartons * yogurt_price = 118 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_yogurt_cost_difference_l170_17039


namespace NUMINAMATH_CALUDE_school_event_ticket_revenue_l170_17063

theorem school_event_ticket_revenue :
  ∀ (f h : ℕ) (p : ℚ),
    f + h = 160 →
    f * p + h * (p / 2) = 2400 →
    f * p = 800 :=
by sorry

end NUMINAMATH_CALUDE_school_event_ticket_revenue_l170_17063


namespace NUMINAMATH_CALUDE_salt_to_flour_ratio_l170_17061

/-- Represents the ingredients for making pizza --/
structure PizzaIngredients where
  water : ℕ
  flour : ℕ
  salt : ℕ

/-- Theorem stating the ratio of salt to flour in the pizza recipe --/
theorem salt_to_flour_ratio (ingredients : PizzaIngredients) : 
  ingredients.water = 10 →
  ingredients.flour = 16 →
  ingredients.water + ingredients.flour + ingredients.salt = 34 →
  ingredients.salt * 2 = ingredients.flour := by
  sorry

end NUMINAMATH_CALUDE_salt_to_flour_ratio_l170_17061


namespace NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l170_17083

theorem newspaper_conference_max_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) (x : ℕ) :
  total = 100 →
  writers = 45 →
  editors > 36 →
  writers + editors - x + 2 * x = total →
  x ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l170_17083


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l170_17082

theorem right_triangle_ratio (x d : ℝ) (h1 : x > d) (h2 : d > 0) : 
  (x^2)^2 + (x^2 - d)^2 = (x^2 + d)^2 → x / d = 8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l170_17082


namespace NUMINAMATH_CALUDE_cost_of_bread_and_drinks_l170_17053

/-- The cost of buying bread and drinks -/
theorem cost_of_bread_and_drinks 
  (a b : ℝ) 
  (h1 : a ≥ 0) 
  (h2 : b ≥ 0) : 
  a + 2 * b = (1 : ℝ) * a + (2 : ℝ) * b := by sorry

end NUMINAMATH_CALUDE_cost_of_bread_and_drinks_l170_17053


namespace NUMINAMATH_CALUDE_k_values_l170_17045

def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | (1 - p.2) / (1 + p.1) = 3}

def B (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + 3}

theorem k_values (k : ℝ) : A ∩ B k = ∅ → k = 2 ∨ k = -3 := by
  sorry

end NUMINAMATH_CALUDE_k_values_l170_17045


namespace NUMINAMATH_CALUDE_tonyas_age_l170_17060

/-- Proves Tonya's age given the conditions of the problem -/
theorem tonyas_age (john mary tonya : ℕ) 
  (h1 : john = 2 * mary)
  (h2 : john = tonya / 2)
  (h3 : (john + mary + tonya) / 3 = 35) :
  tonya = 60 := by sorry

end NUMINAMATH_CALUDE_tonyas_age_l170_17060


namespace NUMINAMATH_CALUDE_projectile_max_height_l170_17070

def f (t : ℝ) : ℝ := -8 * t^2 + 64 * t + 36

theorem projectile_max_height :
  ∃ (max : ℝ), max = 164 ∧ ∀ (t : ℝ), f t ≤ max :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l170_17070


namespace NUMINAMATH_CALUDE_square_root_equation_l170_17066

theorem square_root_equation : Real.sqrt 1936 / 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l170_17066


namespace NUMINAMATH_CALUDE_oldest_sibling_age_l170_17003

theorem oldest_sibling_age (average_age : ℝ) (age1 age2 age3 : ℕ) :
  average_age = 9 ∧ age1 = 5 ∧ age2 = 8 ∧ age3 = 7 →
  ∃ (oldest_age : ℕ), (age1 + age2 + age3 + oldest_age) / 4 = average_age ∧ oldest_age = 16 :=
by sorry

end NUMINAMATH_CALUDE_oldest_sibling_age_l170_17003


namespace NUMINAMATH_CALUDE_books_rebecca_received_l170_17001

theorem books_rebecca_received (books_initial : ℕ) (books_remaining : ℕ) 
  (h1 : books_initial = 220)
  (h2 : books_remaining = 60) : 
  ∃ (rebecca_books : ℕ), 
    rebecca_books = (books_initial - books_remaining) / 4 ∧ 
    rebecca_books = 40 := by
  sorry

end NUMINAMATH_CALUDE_books_rebecca_received_l170_17001


namespace NUMINAMATH_CALUDE_similar_triangles_sequence_l170_17054

/-- Given a sequence of six similar right triangles with vertex A, where AB = 24 and AC = 54,
    prove that the length of AD (hypotenuse of the third triangle) is 36. -/
theorem similar_triangles_sequence (a b x c d : ℝ) : 
  (24 : ℝ) / a = a / b ∧ 
  a / b = b / x ∧ 
  b / x = x / c ∧ 
  x / c = c / d ∧ 
  c / d = d / 54 → 
  x = 36 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_sequence_l170_17054


namespace NUMINAMATH_CALUDE_dany_farm_cows_l170_17015

/-- Represents the number of cows on Dany's farm -/
def num_cows : ℕ := 4

/-- Represents the number of sheep on Dany's farm -/
def num_sheep : ℕ := 3

/-- Represents the number of chickens on Dany's farm -/
def num_chickens : ℕ := 7

/-- Represents the number of bushels a sheep eats per day -/
def sheep_bushels : ℕ := 2

/-- Represents the number of bushels a chicken eats per day -/
def chicken_bushels : ℕ := 3

/-- Represents the number of bushels a cow eats per day -/
def cow_bushels : ℕ := 2

/-- Represents the total number of bushels needed for all animals per day -/
def total_bushels : ℕ := 35

theorem dany_farm_cows :
  num_cows * cow_bushels + num_sheep * sheep_bushels + num_chickens * chicken_bushels = total_bushels :=
by sorry

end NUMINAMATH_CALUDE_dany_farm_cows_l170_17015


namespace NUMINAMATH_CALUDE_triangle_side_values_l170_17011

theorem triangle_side_values (x : ℕ+) :
  (9 + 12 > x^2 ∧ x^2 + 9 > 12 ∧ x^2 + 12 > 9) ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l170_17011


namespace NUMINAMATH_CALUDE_area_of_triangle_DEF_l170_17014

-- Define the square PQRS
def square_PQRS : Real := 36

-- Define the side length of the smaller squares
def small_square_side : Real := 2

-- Define the triangle DEF
structure Triangle_DEF where
  DE : Real
  EF : Real
  isIsosceles : DE = DF

-- Define the folding property
def folding_property (t : Triangle_DEF) : Prop :=
  ∃ (center : Real), 
    center = (square_PQRS.sqrt / 2) ∧
    t.DE = center + 2 * small_square_side

-- Theorem statement
theorem area_of_triangle_DEF : 
  ∀ (t : Triangle_DEF), 
    folding_property t → 
    (1/2 : Real) * t.EF * t.DE = 7 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_DEF_l170_17014


namespace NUMINAMATH_CALUDE_runner_b_speed_l170_17049

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ  -- Speed in m/s

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ  -- Race distance in meters
  runnerA : Runner
  runnerB : Runner
  distanceDiff : ℝ  -- Distance difference at finish in meters
  timeDiff : ℝ  -- Time difference at finish in seconds

/-- Theorem stating that in a specific race scenario, runner B's speed is 8 m/s -/
theorem runner_b_speed (race : Race) : 
  race.distance = 1000 ∧ 
  race.distanceDiff = 200 ∧ 
  race.timeDiff = 25 → 
  race.runnerB.speed = 8 := by
  sorry


end NUMINAMATH_CALUDE_runner_b_speed_l170_17049


namespace NUMINAMATH_CALUDE_product_19_reciprocal_squares_sum_l170_17084

theorem product_19_reciprocal_squares_sum :
  ∀ a b : ℕ+, a * b = 19 → (1 : ℚ) / a^2 + (1 : ℚ) / b^2 = 362 / 361 := by
  sorry

end NUMINAMATH_CALUDE_product_19_reciprocal_squares_sum_l170_17084


namespace NUMINAMATH_CALUDE_function_inequality_l170_17030

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, x * (deriv f x) + f x > 0) (a b : ℝ) (hab : a > b) : 
  a * f a > b * f b := by sorry

end NUMINAMATH_CALUDE_function_inequality_l170_17030


namespace NUMINAMATH_CALUDE_curve_C_cartesian_to_polar_l170_17056

/-- The curve C in the Cartesian plane -/
def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- The polar equation of curve C -/
def polar_C (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The relationship between Cartesian and polar coordinates -/
def polar_to_cartesian (ρ θ x y : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

theorem curve_C_cartesian_to_polar :
  ∀ x y ρ θ : ℝ, 
    polar_to_cartesian ρ θ x y →
    (C x y ↔ polar_C ρ θ) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_cartesian_to_polar_l170_17056


namespace NUMINAMATH_CALUDE_inequality_proof_l170_17059

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : a * b + b * c + c * a = 1 / 3) : 
  1 / (a^2 - b*c + 1) + 1 / (b^2 - c*a + 1) + 1 / (c^2 - a*b + 1) ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l170_17059


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l170_17028

theorem solution_set_equivalence (x : ℝ) :
  (Real.log (|x - π/3|) / Real.log (1/2) ≥ Real.log (π/2) / Real.log (1/2)) ↔
  (-π/6 ≤ x ∧ x ≤ 5*π/6 ∧ x ≠ π/3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l170_17028


namespace NUMINAMATH_CALUDE_smallest_integer_l170_17097

theorem smallest_integer (a b : ℕ+) (h1 : a = 60) 
  (h2 : Nat.lcm a b / Nat.gcd a b = 75) : 
  ∀ c : ℕ+, (c < b → ¬(Nat.lcm a c / Nat.gcd a c = 75)) → b = 500 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l170_17097


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_cube_l170_17021

/-- Given a cube with surface area 6a^2 and all its vertices on a sphere,
    prove that the surface area of the sphere is 3πa^2 -/
theorem sphere_surface_area_from_cube (a : ℝ) (h : a > 0) :
  let cube_surface_area := 6 * a^2
  let cube_diagonal := a * Real.sqrt 3
  let sphere_radius := cube_diagonal / 2
  let sphere_surface_area := 4 * Real.pi * sphere_radius^2
  sphere_surface_area = 3 * Real.pi * a^2 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_cube_l170_17021


namespace NUMINAMATH_CALUDE_percentage_excess_l170_17051

/-- Given two positive real numbers A and B with a specific ratio and sum condition,
    this theorem proves the formula for the percentage by which B exceeds A. -/
theorem percentage_excess (x y A B : ℝ) : 
  x > 0 → y > 0 → A > 0 → B > 0 →
  A / B = (5 * y^2) / (6 * x) →
  2 * x + 3 * y = 42 →
  ((B - A) / A) * 100 = ((126 - 9*y - 5*y^2) / (5*y^2)) * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_excess_l170_17051


namespace NUMINAMATH_CALUDE_cos_750_degrees_l170_17043

theorem cos_750_degrees : Real.cos (750 * π / 180) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_750_degrees_l170_17043


namespace NUMINAMATH_CALUDE_equation_solution_l170_17000

theorem equation_solution :
  let x : ℝ := 12 / 0.17
  0.05 * x + 0.12 * (30 + x) = 15.6 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l170_17000


namespace NUMINAMATH_CALUDE_cats_adopted_count_l170_17006

/-- The cost to get a cat ready for adoption -/
def cat_cost : ℕ := 50

/-- The cost to get an adult dog ready for adoption -/
def adult_dog_cost : ℕ := 100

/-- The cost to get a puppy ready for adoption -/
def puppy_cost : ℕ := 150

/-- The number of adult dogs adopted -/
def adult_dogs_adopted : ℕ := 3

/-- The number of puppies adopted -/
def puppies_adopted : ℕ := 2

/-- The total cost for all adopted animals -/
def total_cost : ℕ := 700

/-- Theorem stating that the number of cats adopted is 2 -/
theorem cats_adopted_count : 
  ∃ (c : ℕ), c * cat_cost + adult_dogs_adopted * adult_dog_cost + puppies_adopted * puppy_cost = total_cost ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_cats_adopted_count_l170_17006


namespace NUMINAMATH_CALUDE_initial_quarters_l170_17077

def quarters_after_events (initial : ℕ) : ℕ :=
  let after_doubling := initial * 2
  let after_second_year := after_doubling + 3 * 12
  let after_third_year := after_second_year + 4
  let before_loss := after_third_year
  (before_loss * 3) / 4

theorem initial_quarters (initial : ℕ) : 
  quarters_after_events initial = 105 ↔ initial = 50 := by
  sorry

#eval quarters_after_events 50  -- Should output 105

end NUMINAMATH_CALUDE_initial_quarters_l170_17077


namespace NUMINAMATH_CALUDE_used_car_selections_l170_17069

theorem used_car_selections (num_cars : ℕ) (num_clients : ℕ) (selections_per_client : ℕ)
  (h1 : num_cars = 12)
  (h2 : num_clients = 9)
  (h3 : selections_per_client = 4) :
  (num_clients * selections_per_client) / num_cars = 3 := by
  sorry

end NUMINAMATH_CALUDE_used_car_selections_l170_17069


namespace NUMINAMATH_CALUDE_correct_count_of_students_using_both_colors_l170_17075

/-- The number of students using both green and red colors in a painting activity. -/
def students_using_both_colors (total_students green_users red_users : ℕ) : ℕ :=
  green_users + red_users - total_students

/-- Theorem stating that the number of students using both colors is correct. -/
theorem correct_count_of_students_using_both_colors
  (total_students green_users red_users : ℕ)
  (h1 : total_students = 70)
  (h2 : green_users = 52)
  (h3 : red_users = 56) :
  students_using_both_colors total_students green_users red_users = 38 := by
  sorry

end NUMINAMATH_CALUDE_correct_count_of_students_using_both_colors_l170_17075


namespace NUMINAMATH_CALUDE_abs_a_minus_b_range_l170_17042

theorem abs_a_minus_b_range (a b : ℝ) :
  let A := {x : ℝ | |x - a| < 1}
  let B := {x : ℝ | |x - b| > 3}
  A ⊆ B → |a - b| ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_a_minus_b_range_l170_17042


namespace NUMINAMATH_CALUDE_polynomial_simplification_l170_17017

/-- Proves the equality of two polynomial expressions -/
theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) - (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5) =
  x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l170_17017


namespace NUMINAMATH_CALUDE_parabola_tangent_theorem_l170_17025

-- Define the parabola and points
def Parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

structure Point where
  x : ℝ
  y : ℝ

-- Define the given conditions
def IsValidConfiguration (p : ℝ) (P A B C D Q : Point) : Prop :=
  p > 0 ∧
  Parabola p A.x A.y ∧
  Parabola p B.x B.y ∧
  C.y = 0 ∧
  D.y = 0 ∧
  Q.x = 0 ∧
  -- PA and PB are tangent to G at A and B (implied)
  -- P is outside the parabola (implied)
  -- C and D are on x-axis (y = 0)
  -- Q is on y-axis (x = 0)
  true -- Additional conditions could be added here if needed

-- Define what it means for PCQD to be a parallelogram
def IsParallelogram (P C Q D : Point) : Prop :=
  (P.x - C.x = Q.x - D.x) ∧ (P.y - C.y = Q.y - D.y)

-- Define the main theorem
theorem parabola_tangent_theorem (p : ℝ) (P A B C D Q : Point) :
  IsValidConfiguration p P A B C D Q →
  (IsParallelogram P C Q D ∧
   (IsParallelogram P C Q D ∧ (P.x - C.x)^2 + (P.y - C.y)^2 = (Q.x - D.x)^2 + (Q.y - D.y)^2 ↔ Q.y = p/2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_theorem_l170_17025


namespace NUMINAMATH_CALUDE_rectangle_packing_l170_17035

/-- Represents the maximum number of non-overlapping 2-by-3 rectangles
    that can be placed in an m-by-n rectangle -/
def max_rectangles (m n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of 2-by-3 rectangles
    that can be placed in an m-by-n rectangle is at least ⌊mn/6⌋ -/
theorem rectangle_packing (m n : ℕ) (hm : m > 1) (hn : n > 1) :
  max_rectangles m n ≥ (m * n) / 6 :=
sorry

end NUMINAMATH_CALUDE_rectangle_packing_l170_17035


namespace NUMINAMATH_CALUDE_seventh_group_selection_l170_17055

/-- Represents a systematic sampling method for a class of students. -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  group_size : ℕ
  third_group_selection : ℕ

/-- Calculates the number drawn from a specific group in a systematic sampling method. -/
def number_drawn (s : SystematicSampling) (group : ℕ) : ℕ :=
  (group - 1) * s.group_size + (s.third_group_selection - ((3 - 1) * s.group_size))

/-- Theorem stating that if the number drawn from the third group is 13,
    then the number drawn from the seventh group is 33. -/
theorem seventh_group_selection
  (s : SystematicSampling)
  (h1 : s.total_students = 50)
  (h2 : s.num_groups = 10)
  (h3 : s.group_size = s.total_students / s.num_groups)
  (h4 : s.third_group_selection = 13) :
  number_drawn s 7 = 33 := by
  sorry

end NUMINAMATH_CALUDE_seventh_group_selection_l170_17055


namespace NUMINAMATH_CALUDE_bryans_book_collection_l170_17079

theorem bryans_book_collection (books_per_continent : ℕ) (total_books : ℕ) 
  (h1 : books_per_continent = 122) 
  (h2 : total_books = 488) : 
  total_books / books_per_continent = 4 := by
sorry

end NUMINAMATH_CALUDE_bryans_book_collection_l170_17079


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l170_17058

/-- Represents a stratified sample from a population -/
structure StratifiedSample where
  total_population : ℕ
  boys_population : ℕ
  girls_population : ℕ
  sample_size : ℕ

/-- Calculates the number of boys in the sample -/
def boys_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.boys_population) / s.total_population

/-- Calculates the number of girls in the sample -/
def girls_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.girls_population) / s.total_population

/-- Theorem stating the correct number of boys and girls in the sample -/
theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.total_population = 700)
  (h2 : s.boys_population = 385)
  (h3 : s.girls_population = 315)
  (h4 : s.sample_size = 60) :
  boys_in_sample s = 33 ∧ girls_in_sample s = 27 := by
  sorry

#eval boys_in_sample { total_population := 700, boys_population := 385, girls_population := 315, sample_size := 60 }
#eval girls_in_sample { total_population := 700, boys_population := 385, girls_population := 315, sample_size := 60 }

end NUMINAMATH_CALUDE_stratified_sample_theorem_l170_17058


namespace NUMINAMATH_CALUDE_car_distance_problem_l170_17010

/-- Proves that Car X travels 105 miles from when Car Y starts until both cars stop -/
theorem car_distance_problem (speed_x speed_y : ℝ) (head_start : ℝ) (distance : ℝ) : 
  speed_x = 35 →
  speed_y = 49 →
  head_start = 1.2 →
  distance = speed_x * (head_start + (distance - speed_x * head_start) / (speed_y - speed_x)) →
  distance - speed_x * head_start = 105 := by
  sorry

#check car_distance_problem

end NUMINAMATH_CALUDE_car_distance_problem_l170_17010


namespace NUMINAMATH_CALUDE_sum_of_angles_complex_roots_l170_17026

theorem sum_of_angles_complex_roots (z₁ z₂ z₃ z₄ : ℂ) (r₁ r₂ r₃ r₄ : ℝ) (θ₁ θ₂ θ₃ θ₄ : ℝ) :
  z₁^4 = -16*I ∧ z₂^4 = -16*I ∧ z₃^4 = -16*I ∧ z₄^4 = -16*I ∧
  z₁ = r₁ * (Complex.cos θ₁ + Complex.I * Complex.sin θ₁) ∧
  z₂ = r₂ * (Complex.cos θ₂ + Complex.I * Complex.sin θ₂) ∧
  z₃ = r₃ * (Complex.cos θ₃ + Complex.I * Complex.sin θ₃) ∧
  z₄ = r₄ * (Complex.cos θ₄ + Complex.I * Complex.sin θ₄) ∧
  r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ r₄ > 0 ∧
  0 ≤ θ₁ ∧ θ₁ < 2*π ∧
  0 ≤ θ₂ ∧ θ₂ < 2*π ∧
  0 ≤ θ₃ ∧ θ₃ < 2*π ∧
  0 ≤ θ₄ ∧ θ₄ < 2*π →
  θ₁ + θ₂ + θ₃ + θ₄ = (810 * π) / 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_angles_complex_roots_l170_17026


namespace NUMINAMATH_CALUDE_benjamins_dinner_cost_l170_17027

-- Define the prices of items
def burger_price : ℕ := 5
def fries_price : ℕ := 2
def salad_price : ℕ := 3 * fries_price

-- Define the quantities of items
def burger_quantity : ℕ := 1
def fries_quantity : ℕ := 2
def salad_quantity : ℕ := 1

-- Define the total cost function
def total_cost : ℕ := 
  burger_price * burger_quantity + 
  fries_price * fries_quantity + 
  salad_price * salad_quantity

-- Theorem statement
theorem benjamins_dinner_cost : total_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_benjamins_dinner_cost_l170_17027


namespace NUMINAMATH_CALUDE_inequality_equivalence_l170_17038

theorem inequality_equivalence (x : ℝ) (h : x > 0) :
  x^(0.5 * Real.log x / Real.log 0.5 - 3) ≥ 0.5^(3 - 2.5 * Real.log x / Real.log 0.5) ↔ 
  0.125 ≤ x ∧ x ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l170_17038


namespace NUMINAMATH_CALUDE_equation_solutions_l170_17029

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => (x + 3)^2 - 4*(x - 1)^2
  (f (-1/3) = 0) ∧ (f 5 = 0) ∧ 
  (∀ x : ℝ, f x = 0 → (x = -1/3 ∨ x = 5)) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l170_17029


namespace NUMINAMATH_CALUDE_spelling_bee_contestants_l170_17071

theorem spelling_bee_contestants (initial_students : ℕ) : 
  (initial_students : ℚ) * (1 - 0.6) * (1 / 2) * (1 / 4) = 15 → 
  initial_students = 300 := by
sorry

end NUMINAMATH_CALUDE_spelling_bee_contestants_l170_17071


namespace NUMINAMATH_CALUDE_certain_number_value_l170_17093

theorem certain_number_value : ∃ x : ℝ, 
  (x + 40 + 60) / 3 = (10 + 70 + 13) / 3 + 9 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l170_17093


namespace NUMINAMATH_CALUDE_infinite_solutions_equation_l170_17094

theorem infinite_solutions_equation (A B C : ℚ) : 
  (∀ x : ℚ, (x + B) * (A * x + 40) = 3 * (x + C) * (x + 10)) →
  (A = 3 ∧ B = 10/9 ∧ C = 40/9 ∧ 
   (- 40/9) + (-10) = -130/9) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_equation_l170_17094


namespace NUMINAMATH_CALUDE_mark_friends_percentage_l170_17044

/-- Calculates the initial percentage of friends kept by Mark -/
def initialFriendsPercentage (initialFriends : ℕ) (finalFriends : ℕ) (responseRate : ℚ) : ℚ :=
  let contactedFriends := initialFriends - (finalFriends - (initialFriends * responseRate * (1 - responseRate)))
  (initialFriends - contactedFriends) / initialFriends

theorem mark_friends_percentage :
  initialFriendsPercentage 100 70 (1/2) = 2/5 := by
  sorry

#eval initialFriendsPercentage 100 70 (1/2)

end NUMINAMATH_CALUDE_mark_friends_percentage_l170_17044


namespace NUMINAMATH_CALUDE_adam_final_amount_l170_17033

def initial_amount : ℝ := 1025.25
def console_percentage : ℝ := 0.45
def euro_found : ℝ := 50
def exchange_rate : ℝ := 1.18
def allowance_percentage : ℝ := 0.10

theorem adam_final_amount :
  let amount_spent := initial_amount * console_percentage
  let money_left := initial_amount - amount_spent
  let euro_exchanged := euro_found * exchange_rate
  let money_after_exchange := money_left + euro_exchanged
  let allowance := initial_amount * allowance_percentage
  let final_amount := money_after_exchange + allowance
  final_amount = 725.4125 := by
  sorry

end NUMINAMATH_CALUDE_adam_final_amount_l170_17033


namespace NUMINAMATH_CALUDE_find_coefficient_a_l170_17002

theorem find_coefficient_a (f' : ℝ → ℝ) (a : ℝ) :
  (∀ x, f' x = 2 * x^3 + a * x^2 + x) →
  f' 1 = 9 →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_find_coefficient_a_l170_17002


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l170_17024

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  midpoint_line : ℝ
  shorter_base : ℝ

/-- The trapezoid satisfies the given conditions -/
def satisfies_conditions (t : Trapezoid) : Prop :=
  t.longer_base = 105 ∧ t.midpoint_line = 7

/-- The theorem to be proved -/
theorem trapezoid_shorter_base (t : Trapezoid) 
  (h : satisfies_conditions t) : t.shorter_base = 91 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l170_17024


namespace NUMINAMATH_CALUDE_fifth_month_sale_l170_17078

theorem fifth_month_sale
  (sales : Fin 6 → ℕ)
  (h1 : sales 0 = 7435)
  (h2 : sales 1 = 7920)
  (h3 : sales 2 = 7855)
  (h4 : sales 3 = 8230)
  (h5 : sales 5 = 6000)
  (h_avg : (sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = 7500) :
  sales 4 = 7560 :=
by sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l170_17078


namespace NUMINAMATH_CALUDE_password_digit_l170_17034

theorem password_digit (n : ℕ) : 
  n = 5678 * 6789 → 
  ∃ (a b c d e f g h i : ℕ),
    n = a * 10^8 + b * 10^7 + c * 10^6 + d * 10^5 + e * 10^4 + f * 10^3 + g * 10^2 + h * 10 + i ∧
    a = 3 ∧ b = 8 ∧ c = 5 ∧ d = 4 ∧ f = 9 ∧ g = 4 ∧ h = 2 ∧
    e = 7 :=
sorry

end NUMINAMATH_CALUDE_password_digit_l170_17034


namespace NUMINAMATH_CALUDE_C_power_100_l170_17023

def C : Matrix (Fin 2) (Fin 2) ℝ := !![5, -1; 12, 3]

theorem C_power_100 : 
  C^100 = (3^99 : ℝ) • !![1, 100; 6000, -200] := by sorry

end NUMINAMATH_CALUDE_C_power_100_l170_17023


namespace NUMINAMATH_CALUDE_unique_pair_satisfying_inequality_l170_17087

theorem unique_pair_satisfying_inequality :
  ∃! (a b : ℝ), ∀ x : ℝ, x ∈ Set.Icc 0 1 →
    |Real.sqrt (1 - x^2) - a*x - b| ≤ (Real.sqrt 2 - 1) / 2 ∧ a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_satisfying_inequality_l170_17087


namespace NUMINAMATH_CALUDE_hash_solution_l170_17019

/-- Definition of the # operation -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem stating that 2 is the number that satisfies x # 3 = 12 -/
theorem hash_solution : ∃ x : ℝ, hash x 3 = 12 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_hash_solution_l170_17019


namespace NUMINAMATH_CALUDE_cross_product_example_l170_17018

/-- The cross product of two 3D vectors -/
def cross_product (v w : Fin 3 → ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => v 1 * w 2 - v 2 * w 1
  | 1 => v 2 * w 0 - v 0 * w 2
  | 2 => v 0 * w 1 - v 1 * w 0

/-- The first vector -/
def v : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => -3
  | 1 => 4
  | 2 => 5

/-- The second vector -/
def w : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 2
  | 1 => -1
  | 2 => 4

theorem cross_product_example : cross_product v w = fun i =>
  match i with
  | 0 => 21
  | 1 => 22
  | 2 => -5 := by sorry

end NUMINAMATH_CALUDE_cross_product_example_l170_17018


namespace NUMINAMATH_CALUDE_min_value_of_expression_l170_17096

/-- Given vectors OA, OB, OC, where O is the origin, prove that the minimum value of 1/a + 2/b is 8 -/
theorem min_value_of_expression (a b : ℝ) (OA OB OC : ℝ × ℝ) : 
  a > 0 → b > 0 → 
  OA = (1, -2) → OB = (a, -1) → OC = (-b, 0) →
  (∃ (t : ℝ), (OB.1 - OA.1, OB.2 - OA.2) = t • (OC.1 - OA.1, OC.2 - OA.2)) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1 / a' + 2 / b' ≥ 8) ∧ 
  (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 1 / a' + 2 / b' = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l170_17096


namespace NUMINAMATH_CALUDE_base_eight_digits_of_512_l170_17050

theorem base_eight_digits_of_512 : ∃ n : ℕ, n > 0 ∧ 8^(n-1) ≤ 512 ∧ 512 < 8^n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_digits_of_512_l170_17050


namespace NUMINAMATH_CALUDE_expression_simplification_l170_17048

theorem expression_simplification (x : ℝ) (hx : x^2 - 2*x = 0) (hx_nonzero : x ≠ 0) :
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l170_17048


namespace NUMINAMATH_CALUDE_three_zeroes_implies_a_range_l170_17031

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then (x - 1) / Real.exp x else -x - 1

-- Define g as a function of f and b
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a x - b

-- State the theorem
theorem three_zeroes_implies_a_range :
  ∀ a : ℝ, (∃ b : ℝ, (∃! z1 z2 z3 : ℝ, z1 ≠ z2 ∧ z1 ≠ z3 ∧ z2 ≠ z3 ∧
    g a b z1 = 0 ∧ g a b z2 = 0 ∧ g a b z3 = 0 ∧
    (∀ z : ℝ, g a b z = 0 → z = z1 ∨ z = z2 ∨ z = z3))) →
  a > -1 / Real.exp 2 - 1 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_zeroes_implies_a_range_l170_17031


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l170_17099

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^22 + i^222 = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l170_17099


namespace NUMINAMATH_CALUDE_camping_cost_equalization_l170_17004

theorem camping_cost_equalization 
  (X Y Z : ℝ) 
  (h_order : X < Y ∧ Y < Z) :
  let total_cost := X + Y + Z
  let equal_share := total_cost / 3
  (equal_share - X) = (Y + Z - 2 * X) / 3 := by
sorry

end NUMINAMATH_CALUDE_camping_cost_equalization_l170_17004


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l170_17064

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary (sum to 90°)
  a = 4 * b →   -- The ratio of the angles is 4:1
  b = 18 :=     -- The smaller angle is 18°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l170_17064
