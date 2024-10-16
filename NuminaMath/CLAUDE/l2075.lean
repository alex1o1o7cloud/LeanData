import Mathlib

namespace NUMINAMATH_CALUDE_function_properties_l2075_207520

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

theorem function_properties 
  (ω φ : ℝ) 
  (hω : ω > 0) 
  (hφ : 0 < φ ∧ φ < Real.pi / 2) 
  (hperiod : ∀ x, f ω φ (x + Real.pi) = f ω φ x)
  (hsymmetry : ∀ x, f ω φ (-Real.pi/24 + x) = f ω φ (-Real.pi/24 - x))
  (A B C : ℝ)
  (ha : ∀ a b c : ℝ, a = 3 → b + c = 6 → a^2 = b^2 + c^2 - 2*b*c*Real.cos A)
  (hf : f ω φ (-A/2) = Real.sqrt 2) :
  ω = 2 ∧ φ = Real.pi/12 ∧ ∃ (b c : ℝ), b = 3 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2075_207520


namespace NUMINAMATH_CALUDE_modulus_of_one_plus_i_l2075_207548

/-- The modulus of the complex number z = 1 + i is √2 -/
theorem modulus_of_one_plus_i : Complex.abs (1 + Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_one_plus_i_l2075_207548


namespace NUMINAMATH_CALUDE_brad_daily_reading_l2075_207599

/-- Brad's daily reading in pages -/
def brad_pages : ℕ := 26

/-- Greg's daily reading in pages -/
def greg_pages : ℕ := 18

/-- The difference in pages read between Brad and Greg -/
def page_difference : ℕ := 8

theorem brad_daily_reading :
  brad_pages = greg_pages + page_difference :=
by sorry

end NUMINAMATH_CALUDE_brad_daily_reading_l2075_207599


namespace NUMINAMATH_CALUDE_pencils_per_box_correct_l2075_207577

/-- Represents the number of pencils in each box -/
def pencils_per_box : ℕ := 80

/-- Represents the number of boxes of pencils ordered -/
def boxes : ℕ := 15

/-- Represents the cost of a single pencil in dollars -/
def pencil_cost : ℕ := 4

/-- Represents the cost of a single pen in dollars -/
def pen_cost : ℕ := 5

/-- Represents the total cost of all stationery in dollars -/
def total_cost : ℕ := 18300

/-- Theorem stating that the number of pencils per box satisfies the given conditions -/
theorem pencils_per_box_correct : 
  let total_pencils := pencils_per_box * boxes
  let total_pens := 2 * total_pencils + 300
  total_pencils * pencil_cost + total_pens * pen_cost = total_cost := by
  sorry


end NUMINAMATH_CALUDE_pencils_per_box_correct_l2075_207577


namespace NUMINAMATH_CALUDE_movie_cost_ratio_l2075_207595

/-- Proves that the ratio of the cost per minute of the new movie to the previous movie is 1/5 -/
theorem movie_cost_ratio :
  let previous_length : ℝ := 2 * 60  -- in minutes
  let new_length : ℝ := previous_length * 1.6
  let previous_cost_per_minute : ℝ := 50
  let total_new_cost : ℝ := 1920
  let new_cost_per_minute : ℝ := total_new_cost / new_length
  new_cost_per_minute / previous_cost_per_minute = 1 / 5 := by
sorry


end NUMINAMATH_CALUDE_movie_cost_ratio_l2075_207595


namespace NUMINAMATH_CALUDE_soccer_game_theorem_l2075_207501

def soccer_game (total_players : ℕ) (starting_players : ℕ) (first_half_subs : ℕ) : Prop :=
  let second_half_subs := 2 * first_half_subs
  let players_who_played := starting_players + first_half_subs + second_half_subs
  let players_who_didnt_play := total_players - players_who_played
  players_who_didnt_play = 7

theorem soccer_game_theorem :
  soccer_game 24 11 2 := by
  sorry

end NUMINAMATH_CALUDE_soccer_game_theorem_l2075_207501


namespace NUMINAMATH_CALUDE_compute_expression_l2075_207503

theorem compute_expression : 3 * 3^4 + 9^60 / 9^59 - 27^3 = -19431 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2075_207503


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l2075_207571

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l2075_207571


namespace NUMINAMATH_CALUDE_difference_of_squares_form_l2075_207596

theorem difference_of_squares_form (x y : ℝ) :
  ∃ (a b : ℝ), (2*x + y) * (y - 2*x) = -(a^2 - b^2) :=
sorry

end NUMINAMATH_CALUDE_difference_of_squares_form_l2075_207596


namespace NUMINAMATH_CALUDE_function_equation_implies_odd_l2075_207535

/-- A non-zero function satisfying the given functional equation is odd -/
theorem function_equation_implies_odd (f : ℝ → ℝ) 
  (h_nonzero : ∃ x, f x ≠ 0)
  (h_eq : ∀ a b : ℝ, f (a * b) = a * f b + b * f a) :
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_implies_odd_l2075_207535


namespace NUMINAMATH_CALUDE_unique_special_number_l2075_207567

/-- A three-digit number is represented by its digits a, b, and c -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a > 0
  h2 : a ≤ 9
  h3 : b ≤ 9
  h4 : c ≤ 9

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of digits of a three-digit number -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.a + n.b + n.c

/-- A three-digit number is special if it equals 11 times the sum of its digits -/
def isSpecial (n : ThreeDigitNumber) : Prop :=
  value n = 11 * digitSum n

theorem unique_special_number :
  ∃! n : ThreeDigitNumber, isSpecial n ∧ value n = 198 :=
sorry

end NUMINAMATH_CALUDE_unique_special_number_l2075_207567


namespace NUMINAMATH_CALUDE_solution_set_f_geq_0_max_value_f_range_of_m_l2075_207540

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 20| - |16 - x|

-- Theorem for the solution set of f(x) ≥ 0
theorem solution_set_f_geq_0 : 
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≥ -2} := by sorry

-- Theorem for the maximum value of f(x)
theorem max_value_f : 
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 36 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), f x ≥ m) ↔ m ≤ 36 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_0_max_value_f_range_of_m_l2075_207540


namespace NUMINAMATH_CALUDE_triangle_value_l2075_207518

theorem triangle_value (triangle q r : ℚ) 
  (eq1 : triangle + q = 75)
  (eq2 : (triangle + q) + r = 138)
  (eq3 : r = q / 3) :
  triangle = -114 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l2075_207518


namespace NUMINAMATH_CALUDE_dog_food_calculation_l2075_207537

/-- Calculates the total amount of dog food needed per day for a given list of dog weights -/
def totalDogFood (weights : List ℕ) : ℕ :=
  (weights.map (· / 10)).sum

/-- Theorem: Given five dogs with specific weights, the total dog food needed is 15 pounds -/
theorem dog_food_calculation :
  totalDogFood [20, 40, 10, 30, 50] = 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_calculation_l2075_207537


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2075_207557

theorem perfect_square_trinomial (a k : ℝ) : 
  (∃ b : ℝ, a^2 - k*a + 25 = (a - b)^2) → k = 10 ∨ k = -10 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2075_207557


namespace NUMINAMATH_CALUDE_positive_roots_of_x_power_x_l2075_207564

theorem positive_roots_of_x_power_x (x : ℝ) : 
  x > 0 → (x^x = 1 / Real.sqrt 2 ↔ x = 1/2 ∨ x = 1/4) := by
  sorry

end NUMINAMATH_CALUDE_positive_roots_of_x_power_x_l2075_207564


namespace NUMINAMATH_CALUDE_empty_seats_count_l2075_207588

structure Section where
  capacity : Nat
  attendance : Nat

def theater : List Section := [
  { capacity := 250, attendance := 195 },
  { capacity := 180, attendance := 143 },
  { capacity := 150, attendance := 110 },
  { capacity := 300, attendance := 261 },
  { capacity := 230, attendance := 157 },
  { capacity := 90, attendance := 66 }
]

def totalCapacity : Nat := List.foldl (fun acc s => acc + s.capacity) 0 theater
def totalAttendance : Nat := List.foldl (fun acc s => acc + s.attendance) 0 theater

theorem empty_seats_count :
  totalCapacity - totalAttendance = 268 :=
by sorry

end NUMINAMATH_CALUDE_empty_seats_count_l2075_207588


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l2075_207514

theorem simultaneous_equations_solution :
  ∀ x y : ℝ, 
    (2 * x - 3 * y = 0.4 * (x + y)) →
    (5 * y = 1.2 * x) →
    (x = 0 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l2075_207514


namespace NUMINAMATH_CALUDE_find_number_l2075_207555

theorem find_number : ∃ x : ℝ, (5 * x) / (180 / 3) + 70 = 71 ∧ x = 12 := by sorry

end NUMINAMATH_CALUDE_find_number_l2075_207555


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2075_207579

noncomputable def hypotenuse_length (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem right_triangle_hypotenuse (a b : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (vol1 : (1/3) * Real.pi * b^2 * a = 1250 * Real.pi)
  (vol2 : (1/3) * Real.pi * a^2 * b = 2700 * Real.pi) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (hypotenuse_length a b - 21.33) < ε :=
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2075_207579


namespace NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coord_l2075_207516

theorem degenerate_ellipse_max_y_coord :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ (x^2 / 36) + ((y + 5)^2 / 16)
  ∀ (x y : ℝ), f (x, y) = 0 → y ≤ -5 :=
by sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coord_l2075_207516


namespace NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l2075_207584

/-- The area of the shaded region in a square with side length 20 cm and four quarter circles
    with radius 10 cm drawn at the corners is 400 - 100π cm². -/
theorem shaded_area_square_with_quarter_circles :
  let square_side : ℝ := 20
  let circle_radius : ℝ := 10
  let square_area : ℝ := square_side ^ 2
  let quarter_circle_area : ℝ := π * circle_radius ^ 2 / 4
  let total_quarter_circles_area : ℝ := 4 * quarter_circle_area
  let shaded_area : ℝ := square_area - total_quarter_circles_area
  shaded_area = 400 - 100 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l2075_207584


namespace NUMINAMATH_CALUDE_cafeteria_problem_l2075_207536

/-- The cafeteria problem -/
theorem cafeteria_problem 
  (initial_apples : ℕ)
  (apple_cost orange_cost : ℚ)
  (total_earnings : ℚ)
  (apples_left oranges_left : ℕ)
  (h1 : initial_apples = 50)
  (h2 : apple_cost = 8/10)
  (h3 : orange_cost = 1/2)
  (h4 : total_earnings = 49)
  (h5 : apples_left = 10)
  (h6 : oranges_left = 6) :
  ∃ initial_oranges : ℕ, 
    initial_oranges = 40 ∧
    (initial_apples - apples_left) * apple_cost + 
    (initial_oranges - oranges_left) * orange_cost = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_problem_l2075_207536


namespace NUMINAMATH_CALUDE_sum_of_digits_equals_16_l2075_207515

/-- The sum of the digits of (10^38) - 85 when written as a base 10 integer -/
def sumOfDigits : ℕ :=
  -- Define the sum of digits here
  sorry

/-- Theorem stating that the sum of the digits of (10^38) - 85 is 16 -/
theorem sum_of_digits_equals_16 : sumOfDigits = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_equals_16_l2075_207515


namespace NUMINAMATH_CALUDE_susie_q_investment_l2075_207597

def pretty_penny_rate : ℝ := 0.03
def five_and_dime_rate : ℝ := 0.05
def total_investment : ℝ := 1000
def total_after_two_years : ℝ := 1090.02
def years : ℕ := 2

theorem susie_q_investment (x : ℝ) :
  x * (1 + pretty_penny_rate) ^ years + (total_investment - x) * (1 + five_and_dime_rate) ^ years = total_after_two_years →
  x = 300 := by
sorry

end NUMINAMATH_CALUDE_susie_q_investment_l2075_207597


namespace NUMINAMATH_CALUDE_fraction_addition_l2075_207531

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2075_207531


namespace NUMINAMATH_CALUDE_females_over30_prefer_l2075_207549

/-- Represents the survey data from WebStream --/
structure WebStreamSurvey where
  total_surveyed : ℕ
  total_prefer : ℕ
  males_prefer : ℕ
  females_under30_not_prefer : ℕ
  females_over30_not_prefer : ℕ

/-- Theorem stating the number of females over 30 who prefer WebStream --/
theorem females_over30_prefer (survey : WebStreamSurvey)
  (h1 : survey.total_surveyed = 420)
  (h2 : survey.total_prefer = 200)
  (h3 : survey.males_prefer = 80)
  (h4 : survey.females_under30_not_prefer = 90)
  (h5 : survey.females_over30_not_prefer = 70) :
  ∃ (females_over30_prefer : ℕ), females_over30_prefer = 110 := by
  sorry


end NUMINAMATH_CALUDE_females_over30_prefer_l2075_207549


namespace NUMINAMATH_CALUDE_number_problem_l2075_207521

theorem number_problem : ∃ n : ℝ, n - (1002 / 20.04) = 2984 ∧ n = 3034 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2075_207521


namespace NUMINAMATH_CALUDE_parallelogram_reflection_theorem_l2075_207582

-- Define the type for 2D points
def Point := ℝ × ℝ

-- Define the reflection across x-axis
def reflectX (p : Point) : Point :=
  (p.1, -p.2)

-- Define the reflection across y = x - 2
def reflectYXMinus2 (p : Point) : Point :=
  let p' := (p.1, p.2 + 2)  -- Translate up by 2
  let p'' := (p'.2, p'.1)   -- Reflect across y = x
  (p''.1, p''.2 - 2)        -- Translate back down by 2

-- Define the theorem
theorem parallelogram_reflection_theorem (A B C D : Point)
  (hA : A = (3, 7))
  (hB : B = (5, 11))
  (hC : C = (7, 7))
  (hD : D = (5, 3))
  : reflectYXMinus2 (reflectX D) = (-1, 3) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_reflection_theorem_l2075_207582


namespace NUMINAMATH_CALUDE_part_one_part_two_l2075_207539

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| - 3

-- Part I
theorem part_one (m : ℝ) :
  (∀ x, f m x ≥ 0 ↔ x ≤ -2 ∨ x ≥ 4) → m = 1 := by sorry

-- Part II
theorem part_two :
  ∀ t, (∃ x, f 1 x ≥ t + |2 - x|) → t ≤ -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2075_207539


namespace NUMINAMATH_CALUDE_rectangle_ratio_theorem_l2075_207575

theorem rectangle_ratio_theorem (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_a_le_b : a ≤ b) :
  (a / b = (a + b) / Real.sqrt (a^2 + b^2)) →
  (a / b)^2 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_theorem_l2075_207575


namespace NUMINAMATH_CALUDE_cubic_decreasing_l2075_207561

-- Define the function f(x) = mx³ - x
def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 - x

-- State the theorem
theorem cubic_decreasing (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x ≥ f m y) ↔ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_decreasing_l2075_207561


namespace NUMINAMATH_CALUDE_fruit_drink_volume_l2075_207598

/-- Represents the composition of a fruit drink -/
structure FruitDrink where
  orange : ℝ
  watermelon : ℝ
  grape : ℝ
  apple : ℝ
  pineapple : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_volume (drink : FruitDrink)
  (h1 : drink.orange = 0.1)
  (h2 : drink.watermelon = 0.4)
  (h3 : drink.grape = 0.2)
  (h4 : drink.apple = 0.15)
  (h5 : drink.pineapple = 0.15)
  (h6 : drink.orange + drink.watermelon + drink.grape + drink.apple + drink.pineapple = 1)
  (h7 : 24 / drink.grape = 36 / drink.apple) :
  24 / drink.grape = 240 := by
  sorry

end NUMINAMATH_CALUDE_fruit_drink_volume_l2075_207598


namespace NUMINAMATH_CALUDE_rectangle_area_l2075_207511

/-- A rectangle ABCD is divided into four identical squares and has a perimeter of 160 cm. -/
structure Rectangle :=
  (side : ℝ)
  (perimeter_eq : 10 * side = 160)

/-- The area of the rectangle ABCD is 1024 square centimeters. -/
theorem rectangle_area (rect : Rectangle) : 4 * rect.side^2 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2075_207511


namespace NUMINAMATH_CALUDE_circle_equation_l2075_207532

theorem circle_equation (r : ℝ) (h1 : r = 6) :
  ∃ (a b : ℝ),
    (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2) ∧
    (b = r) ∧
    (∃ (x y : ℝ), x^2 + y^2 - 6*y + 8 = 0 ∧ (x - a)^2 + (y - b)^2 = (r - 1)^2) →
    ((a = 4 ∨ a = -4) ∧ b = 6) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2075_207532


namespace NUMINAMATH_CALUDE_two_solutions_exist_l2075_207543

def A (x : ℝ) : Set ℝ := {0, 1, 2, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

theorem two_solutions_exist :
  ∃! (s : Set ℝ), (∃ (x₁ x₂ : ℝ), s = {x₁, x₂} ∧ 
    ∀ (x : ℝ), (A x ∪ B x = A x) ↔ (x ∈ s)) ∧ 
    (∀ (x : ℝ), x ∈ s → x^2 = 2) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_exist_l2075_207543


namespace NUMINAMATH_CALUDE_right_triangle_area_l2075_207594

/-- The area of a right triangle with hypotenuse 15 and one angle 45° --/
theorem right_triangle_area (h : ℝ) (α : ℝ) (area : ℝ) 
  (hyp : h = 15)
  (angle : α = 45 * Real.pi / 180)
  (right_angle : α + α + Real.pi / 2 = Real.pi) : 
  area = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2075_207594


namespace NUMINAMATH_CALUDE_heidi_painting_fraction_l2075_207552

/-- If a person can paint a wall in a given time, this function calculates
    the fraction of the wall they can paint in a shorter time. -/
def fractionPainted (totalTime minutes : ℕ) : ℚ :=
  minutes / totalTime

/-- Theorem stating that if Heidi can paint a wall in 60 minutes,
    she can paint 1/5 of the wall in 12 minutes. -/
theorem heidi_painting_fraction :
  fractionPainted 60 12 = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_heidi_painting_fraction_l2075_207552


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2075_207586

theorem trigonometric_identity : 
  Real.sin (40 * π / 180) * Real.sin (10 * π / 180) + 
  Real.cos (40 * π / 180) * Real.sin (80 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2075_207586


namespace NUMINAMATH_CALUDE_additional_hamburgers_l2075_207593

theorem additional_hamburgers (initial : ℕ) (total : ℕ) (h1 : initial = 9) (h2 : total = 12) :
  total - initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_additional_hamburgers_l2075_207593


namespace NUMINAMATH_CALUDE_mass_of_man_is_72_l2075_207580

/-- The density of water in kg/m³ -/
def water_density : ℝ := 1000

/-- Calculates the mass of a man based on boat dimensions and sinking depth -/
def mass_of_man (boat_length boat_breadth sinking_depth : ℝ) : ℝ :=
  water_density * boat_length * boat_breadth * sinking_depth

/-- Theorem stating that the mass of the man is 72 kg given the boat's dimensions and sinking depth -/
theorem mass_of_man_is_72 :
  mass_of_man 3 2 0.012 = 72 := by sorry

end NUMINAMATH_CALUDE_mass_of_man_is_72_l2075_207580


namespace NUMINAMATH_CALUDE_correct_equation_l2075_207581

theorem correct_equation (x : ℝ) : 
  (550 + x) + (460 + x) + (359 + x) + (340 + x) = 2012 + x ↔ x = 75.75 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l2075_207581


namespace NUMINAMATH_CALUDE_inequality_solution_l2075_207533

theorem inequality_solution (x : ℝ) (h : x ≠ 5) :
  (x * (x^2 + x + 1)) / ((x - 5)^2) ≥ 15 ↔ x ∈ Set.Iio 5 ∪ Set.Ioi 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2075_207533


namespace NUMINAMATH_CALUDE_number_composition_l2075_207500

def number_from_parts (ten_thousands : ℕ) (ones : ℕ) : ℕ :=
  ten_thousands * 10000 + ones

theorem number_composition :
  number_from_parts 45 64 = 450064 := by
  sorry

end NUMINAMATH_CALUDE_number_composition_l2075_207500


namespace NUMINAMATH_CALUDE_train_delay_l2075_207541

/-- Proves that a train moving at 4/5 of its usual speed will be 30 minutes late on a journey that usually takes 2 hours -/
theorem train_delay (usual_speed : ℝ) (usual_time : ℝ) (h1 : usual_time = 2) :
  let reduced_speed := (4/5 : ℝ) * usual_speed
  let reduced_time := usual_time * (5/4 : ℝ)
  reduced_time - usual_time = 1/2 := by sorry

#check train_delay

end NUMINAMATH_CALUDE_train_delay_l2075_207541


namespace NUMINAMATH_CALUDE_square_roots_problem_l2075_207517

theorem square_roots_problem (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2*a + 1)^2 = x ∧ (a + 5)^2 = x) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2075_207517


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l2075_207523

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 28 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l2075_207523


namespace NUMINAMATH_CALUDE_total_balls_in_box_l2075_207522

theorem total_balls_in_box (yellow_balls : ℕ) (prob_yellow : ℚ) (total_balls : ℕ) : 
  yellow_balls = 6 → 
  prob_yellow = 1 / 9 → 
  prob_yellow = yellow_balls / total_balls → 
  total_balls = 54 := by sorry

end NUMINAMATH_CALUDE_total_balls_in_box_l2075_207522


namespace NUMINAMATH_CALUDE_simplify_fraction_division_l2075_207507

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 3) * (x - 4) * (x - 2)) := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_division_l2075_207507


namespace NUMINAMATH_CALUDE_base4_to_decimal_example_l2075_207573

/-- Converts a base-4 number to decimal --/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The base-4 representation of the number --/
def base4Number : List Nat := [2, 1, 0, 0, 3]

/-- Theorem: The base-4 number 30012₍₄₎ is equal to 774 in decimal notation --/
theorem base4_to_decimal_example : base4ToDecimal base4Number = 774 := by
  sorry

end NUMINAMATH_CALUDE_base4_to_decimal_example_l2075_207573


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_A_complement_BC_l2075_207508

/-- The universal set U is ℝ -/
def U : Set ℝ := Set.univ

/-- Set A: { x | y = ln(x² - 9) } -/
def A : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 9)}

/-- Set B: { x | (x - 7)/(x + 1) > 0 } -/
def B : Set ℝ := {x | (x - 7) / (x + 1) > 0}

/-- Set C: { x | |x - 2| < 4 } -/
def C : Set ℝ := {x | |x - 2| < 4}

/-- Theorem 1: A ∩ B = { x | x < -3 or x > 7 } -/
theorem intersection_A_B : A ∩ B = {x | x < -3 ∨ x > 7} := by sorry

/-- Theorem 2: A ∩ (U \ (B ∩ C)) = { x | x < -3 or x > 3 } -/
theorem intersection_A_complement_BC : A ∩ (U \ (B ∩ C)) = {x | x < -3 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_A_complement_BC_l2075_207508


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l2075_207554

def ben_lap_time : ℕ := 5
def clara_lap_time : ℕ := 9
def david_lap_time : ℕ := 8

theorem earliest_meeting_time :
  let meeting_time := Nat.lcm (Nat.lcm ben_lap_time clara_lap_time) david_lap_time
  meeting_time = 360 := by sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l2075_207554


namespace NUMINAMATH_CALUDE_arcade_spending_fraction_l2075_207572

theorem arcade_spending_fraction (allowance : ℚ) (arcade_fraction : ℚ) : 
  allowance = 3/2 →
  (2/3 * (1 - arcade_fraction) * allowance = 2/5) →
  arcade_fraction = 3/5 := by
sorry

end NUMINAMATH_CALUDE_arcade_spending_fraction_l2075_207572


namespace NUMINAMATH_CALUDE_point_on_extension_line_l2075_207538

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the points
variable (O A B P : V)

-- Define the conditions
variable (h_not_collinear : ¬Collinear ℝ {O, A, B})
variable (h_vector_equation : (2 : ℝ) • (P - O) = (2 : ℝ) • (A - O) + (2 : ℝ) • (B - O))

-- Theorem statement
theorem point_on_extension_line :
  ∃ (t : ℝ), t < 0 ∧ P = A + t • (B - A) :=
sorry

end NUMINAMATH_CALUDE_point_on_extension_line_l2075_207538


namespace NUMINAMATH_CALUDE_amount_with_r_l2075_207545

/-- Given a total amount shared among three parties where one party has
    two-thirds of the combined amount of the other two, this function
    calculates the amount held by the third party. -/
def calculate_third_party_amount (total : ℚ) : ℚ :=
  (2 / 3) * (3 / 5) * total

/-- Theorem stating that given the problem conditions, 
    the amount held by r is 3200. -/
theorem amount_with_r (total : ℚ) (h_total : total = 8000) :
  calculate_third_party_amount total = 3200 := by
  sorry

#eval calculate_third_party_amount 8000

end NUMINAMATH_CALUDE_amount_with_r_l2075_207545


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2075_207529

/-- Represents the relationship between y, x, and z -/
def relation (k : ℝ) (x y z : ℝ) : Prop :=
  7 * y = (k * z) / (2 * x)^2

theorem inverse_variation_problem (k : ℝ) :
  relation k 1 20 5 →
  relation k 8 0.625 10 :=
by
  sorry


end NUMINAMATH_CALUDE_inverse_variation_problem_l2075_207529


namespace NUMINAMATH_CALUDE_binomial_2023_2_l2075_207558

theorem binomial_2023_2 : Nat.choose 2023 2 = 2045323 := by
  sorry

end NUMINAMATH_CALUDE_binomial_2023_2_l2075_207558


namespace NUMINAMATH_CALUDE_square_vector_properties_l2075_207512

/-- Given a square ABCD with side length 2 and vectors a and b satisfying the given conditions,
    prove that a · b = 2 and (b - 4a) ⊥ b -/
theorem square_vector_properties (a b : ℝ × ℝ) :
  let A := (0, 0)
  let B := (2, 0)
  let C := (2, 2)
  let D := (0, 2)
  let AB := B - A
  let BC := C - B
  AB = 2 • a →
  BC = b - 2 • a →
  a • b = 2 ∧ (b - 4 • a) • b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_vector_properties_l2075_207512


namespace NUMINAMATH_CALUDE_two_women_probability_l2075_207587

/-- The number of young men in the group -/
def num_men : ℕ := 5

/-- The number of young women in the group -/
def num_women : ℕ := 5

/-- The total number of people in the group -/
def total_people : ℕ := num_men + num_women

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The total number of ways to form pairs -/
def total_ways : ℕ := (total_people.factorial) / ((2^num_pairs) * num_pairs.factorial)

/-- The number of ways to form pairs with no two women together -/
def ways_no_two_women : ℕ := num_pairs.factorial

/-- The probability of at least one pair consisting of two young women -/
def prob_two_women : ℚ := (total_ways - ways_no_two_women : ℚ) / total_ways

theorem two_women_probability :
  prob_two_women = 825 / 945 :=
sorry

end NUMINAMATH_CALUDE_two_women_probability_l2075_207587


namespace NUMINAMATH_CALUDE_circle_radius_is_one_l2075_207585

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the circle in rectangular coordinates
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Theorem statement
theorem circle_radius_is_one :
  ∀ ρ θ x y : ℝ,
  polar_equation ρ θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  circle_equation x y →
  1 = 1 := by sorry

end NUMINAMATH_CALUDE_circle_radius_is_one_l2075_207585


namespace NUMINAMATH_CALUDE_sophia_book_reading_l2075_207526

theorem sophia_book_reading (total_pages : ℕ) (pages_read : ℕ) :
  total_pages = 90 →
  pages_read = (total_pages - pages_read) + 30 →
  pages_read = (2 : ℚ) / 3 * total_pages :=
by
  sorry

end NUMINAMATH_CALUDE_sophia_book_reading_l2075_207526


namespace NUMINAMATH_CALUDE_second_to_first_layer_ratio_l2075_207566

/-- Given a three-layer cake recipe, this theorem proves the ratio of the second layer to the first layer. -/
theorem second_to_first_layer_ratio 
  (sugar_first_layer : ℝ) 
  (sugar_third_layer : ℝ) 
  (third_to_second_ratio : ℝ) 
  (h1 : sugar_first_layer = 2)
  (h2 : sugar_third_layer = 12)
  (h3 : third_to_second_ratio = 3) :
  (sugar_third_layer / sugar_first_layer) / third_to_second_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_to_first_layer_ratio_l2075_207566


namespace NUMINAMATH_CALUDE_tan_theta_eq_seven_l2075_207565

theorem tan_theta_eq_seven (θ : Real) 
  (h1 : θ > π/4 ∧ θ < π/2) 
  (h2 : Real.cos (θ - π/4) = 4/5) : 
  Real.tan θ = 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_eq_seven_l2075_207565


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1717_l2075_207530

theorem largest_prime_factor_of_1717 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 1717 ∧ ∀ (q : ℕ), q.Prime → q ∣ 1717 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1717_l2075_207530


namespace NUMINAMATH_CALUDE_square_weight_calculation_l2075_207589

theorem square_weight_calculation (density : ℝ) (thickness : ℝ) 
  (side_length1 : ℝ) (weight1 : ℝ) (side_length2 : ℝ) 
  (h1 : density > 0) (h2 : thickness > 0) 
  (h3 : side_length1 = 4) (h4 : weight1 = 16) (h5 : side_length2 = 6) :
  let weight2 := density * thickness * side_length2^2
  weight2 = 36 := by
  sorry

#check square_weight_calculation

end NUMINAMATH_CALUDE_square_weight_calculation_l2075_207589


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l2075_207583

theorem square_area_error_percentage (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let area_error_percentage := (area_error / actual_area) * 100
  area_error_percentage = 4.04 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_percentage_l2075_207583


namespace NUMINAMATH_CALUDE_index_card_area_l2075_207504

theorem index_card_area (length width : ℝ) : 
  length = 8 ∧ width = 3 →
  (∃ new_length, new_length = length - 2 ∧ new_length * width = 18) →
  (width - 2) * length = 8 :=
by sorry

end NUMINAMATH_CALUDE_index_card_area_l2075_207504


namespace NUMINAMATH_CALUDE_lucy_grocery_shopping_l2075_207560

theorem lucy_grocery_shopping (total_packs noodle_packs : ℕ) 
  (h1 : total_packs = 28)
  (h2 : noodle_packs = 16)
  (h3 : ∃ cookie_packs : ℕ, total_packs = cookie_packs + noodle_packs) :
  ∃ cookie_packs : ℕ, cookie_packs = 12 ∧ total_packs = cookie_packs + noodle_packs :=
by
  sorry

end NUMINAMATH_CALUDE_lucy_grocery_shopping_l2075_207560


namespace NUMINAMATH_CALUDE_product_over_sum_equals_756_l2075_207568

theorem product_over_sum_equals_756 : (6 * 7 * 8 * 9 * 10) / (6 + 7 + 8 + 9 + 10) = 756 := by
  sorry

end NUMINAMATH_CALUDE_product_over_sum_equals_756_l2075_207568


namespace NUMINAMATH_CALUDE_power_division_equality_l2075_207509

theorem power_division_equality : (3^2)^4 / 3^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l2075_207509


namespace NUMINAMATH_CALUDE_rational_power_difference_integer_implies_integer_l2075_207544

theorem rational_power_difference_integer_implies_integer 
  (a b : ℚ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_distinct : a ≠ b) 
  (h_inf_int : ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → ∃ (k : ℤ), k > 0 ∧ a^n - b^n = k) :
  ∃ (m n : ℕ), (m : ℚ) = a ∧ (n : ℚ) = b :=
sorry

end NUMINAMATH_CALUDE_rational_power_difference_integer_implies_integer_l2075_207544


namespace NUMINAMATH_CALUDE_solution_set_f_gt_2_range_m_common_points_l2075_207506

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2*|x + 1|

-- Define the quadratic function g
def g (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Theorem 1: Solution set of f(x) > 2 when m = 5
theorem solution_set_f_gt_2 :
  {x : ℝ | f 5 x > 2} = Set.Ioo (-4/3 : ℝ) 0 := by sorry

-- Theorem 2: Range of m for which f and g always have common points
theorem range_m_common_points :
  {m : ℝ | ∀ y, ∃ x, f m x = y ∧ g x = y} = Set.Ici 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_2_range_m_common_points_l2075_207506


namespace NUMINAMATH_CALUDE_rhombus_triangle_area_l2075_207502

/-- Given a rhombus with diagonals of length 15 and 20, prove that the area of each constituent triangle is 75 -/
theorem rhombus_triangle_area (A : Real) (d₁ d₂ : Real) (h₁ : d₁ = 15) (h₂ : d₂ = 20) 
  (h₃ : A = (d₁ * d₂) / 2) : A / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_triangle_area_l2075_207502


namespace NUMINAMATH_CALUDE_trash_can_count_l2075_207505

theorem trash_can_count (x : ℕ) 
  (h1 : (x / 2 + 8) / 2 + x = 34) : x = 24 := by
  sorry

end NUMINAMATH_CALUDE_trash_can_count_l2075_207505


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2075_207525

/-- The eccentricity of a hyperbola with equation x²/4 - y²/12 = 1 is 2 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = 2 ∧ 
  ∀ x y : ℝ, x^2/4 - y^2/12 = 1 → 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 = 4 ∧ b^2 = 12 ∧ c^2 = a^2 + b^2 ∧ e = c/a :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2075_207525


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2075_207590

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2075_207590


namespace NUMINAMATH_CALUDE_trench_digging_time_l2075_207510

theorem trench_digging_time (a b c d : ℝ) : 
  (a + b + c + d = 1/6) →
  (2*a + (1/2)*b + c + d = 1/6) →
  ((1/2)*a + 2*b + c + d = 1/4) →
  (a + b + c = 1/6) :=
by sorry

end NUMINAMATH_CALUDE_trench_digging_time_l2075_207510


namespace NUMINAMATH_CALUDE_odd_square_octal_property_l2075_207553

theorem odd_square_octal_property (n : ℤ) : 
  ∃ (m : ℤ), (2*n + 1)^2 % 8 = 1 ∧ ((2*n + 1)^2 - 1) / 8 = m * (m + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_odd_square_octal_property_l2075_207553


namespace NUMINAMATH_CALUDE_keiths_purchases_total_cost_l2075_207550

theorem keiths_purchases_total_cost : 
  let rabbit_toy_cost : ℚ := 651/100
  let pet_food_cost : ℚ := 579/100
  let cage_cost : ℚ := 1251/100
  rabbit_toy_cost + pet_food_cost + cage_cost = 2481/100 := by
sorry

end NUMINAMATH_CALUDE_keiths_purchases_total_cost_l2075_207550


namespace NUMINAMATH_CALUDE_triangle_midpoint_x_coordinate_sum_l2075_207551

theorem triangle_midpoint_x_coordinate_sum (a b c : ℝ) :
  let vertex_sum := a + b + c
  let midpoint_sum := (a + b) / 2 + (b + c) / 2 + (c + a) / 2
  midpoint_sum = vertex_sum := by
sorry

end NUMINAMATH_CALUDE_triangle_midpoint_x_coordinate_sum_l2075_207551


namespace NUMINAMATH_CALUDE_supermarket_spending_l2075_207591

theorem supermarket_spending (total : ℝ) : 
  (1/2 : ℝ) * total + (1/3 : ℝ) * total + (1/10 : ℝ) * total + 10 = total → 
  total = 150 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l2075_207591


namespace NUMINAMATH_CALUDE_log_product_equals_one_l2075_207546

theorem log_product_equals_one : Real.log 2 / Real.log 5 * (Real.log 25 / Real.log 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_one_l2075_207546


namespace NUMINAMATH_CALUDE_shirt_cost_l2075_207578

/-- Proves that the cost of each shirt is $50 given the sales and commission information --/
theorem shirt_cost (commission_rate : ℝ) (suit_price : ℝ) (suit_count : ℕ)
  (shirt_count : ℕ) (loafer_price : ℝ) (loafer_count : ℕ) (total_commission : ℝ) :
  commission_rate = 0.15 →
  suit_price = 700 →
  suit_count = 2 →
  shirt_count = 6 →
  loafer_price = 150 →
  loafer_count = 2 →
  total_commission = 300 →
  ∃ (shirt_price : ℝ), 
    total_commission = commission_rate * (suit_price * suit_count + shirt_price * shirt_count + loafer_price * loafer_count) ∧
    shirt_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l2075_207578


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achieved_l2075_207592

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z) ≥ 36 := by
sorry

theorem min_value_achieved (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 1) : 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
  x₀ + y₀ + z₀ = 1 ∧ 
  (1 / x₀ + 4 / y₀ + 9 / z₀) = 36 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achieved_l2075_207592


namespace NUMINAMATH_CALUDE_dartboard_sector_angle_l2075_207563

theorem dartboard_sector_angle (probability : ℝ) (angle : ℝ) : 
  probability = 1/4 → angle = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_dartboard_sector_angle_l2075_207563


namespace NUMINAMATH_CALUDE_inequality_proof_l2075_207569

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c ≥ a * b * c) : 
  (2/a + 3/b + 6/c ≥ 6 ∧ 2/b + 3/c + 6/a ≥ 6) ∨
  (2/b + 3/c + 6/a ≥ 6 ∧ 2/c + 3/a + 6/b ≥ 6) ∨
  (2/c + 3/a + 6/b ≥ 6 ∧ 2/a + 3/b + 6/c ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2075_207569


namespace NUMINAMATH_CALUDE_intersection_equals_open_closed_interval_l2075_207562

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem intersection_equals_open_closed_interval : M ∩ N = Set.Ioc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_closed_interval_l2075_207562


namespace NUMINAMATH_CALUDE_bill_amount_calculation_l2075_207559

/-- Calculates the face value of a bill given the true discount, interest rate, and time to maturity. -/
def faceBill (trueDiscount : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  trueDiscount * (1 + rate * time)

/-- Theorem: Given a true discount of 150 on a bill due in 9 months at 16% per annum, the amount of the bill is 168. -/
theorem bill_amount_calculation : 
  let trueDiscount : ℝ := 150
  let rate : ℝ := 0.16  -- 16% per annum
  let time : ℝ := 0.75  -- 9 months = 9/12 years = 0.75 years
  faceBill trueDiscount rate time = 168 := by
  sorry


end NUMINAMATH_CALUDE_bill_amount_calculation_l2075_207559


namespace NUMINAMATH_CALUDE_profit_percentage_before_decrease_l2075_207528

/-- Proves that the profit percentage before the decrease in manufacturing cost was 20% --/
theorem profit_percentage_before_decrease
  (selling_price : ℝ)
  (manufacturing_cost_before : ℝ)
  (manufacturing_cost_after : ℝ)
  (h1 : manufacturing_cost_before = 80)
  (h2 : manufacturing_cost_after = 50)
  (h3 : selling_price - manufacturing_cost_after = 0.5 * selling_price) :
  (selling_price - manufacturing_cost_before) / selling_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_before_decrease_l2075_207528


namespace NUMINAMATH_CALUDE_search_rescue_selection_methods_l2075_207513

def chinese_ships : ℕ := 4
def chinese_planes : ℕ := 3
def foreign_ships : ℕ := 5
def foreign_planes : ℕ := 2

def units_per_side : ℕ := 2
def total_units : ℕ := 4
def required_planes : ℕ := 1

theorem search_rescue_selection_methods :
  (chinese_ships.choose units_per_side * chinese_planes.choose required_planes * foreign_ships.choose units_per_side) +
  (chinese_ships.choose units_per_side * foreign_ships.choose (units_per_side - 1) * foreign_planes.choose required_planes) = 180 := by
  sorry

end NUMINAMATH_CALUDE_search_rescue_selection_methods_l2075_207513


namespace NUMINAMATH_CALUDE_fraction_simplification_l2075_207519

theorem fraction_simplification :
  (1^2 + 1) * (2^2 + 1) * (3^2 + 1) / ((2^2 - 1) * (3^2 - 1) * (4^2 - 1)) = 5 / 18 :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2075_207519


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2075_207524

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 4 ∧ x₁^2 - 6*x₁ + 8 = 0 ∧ x₂^2 - 6*x₂ + 8 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 4 + Real.sqrt 15 ∧ x₂ = 4 - Real.sqrt 15 ∧ x₁^2 - 8*x₁ + 1 = 0 ∧ x₂^2 - 8*x₂ + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2075_207524


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l2075_207570

theorem smallest_next_divisor_after_221 (n : ℕ) (h1 : 1000 ≤ n ∧ n ≤ 9999) 
  (h2 : Even n) (h3 : 221 ∣ n) : 
  ∃ (d : ℕ), d ∣ n ∧ d > 221 ∧ d ≥ 238 ∧ ∀ (x : ℕ), x ∣ n → x > 221 → x ≥ d :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l2075_207570


namespace NUMINAMATH_CALUDE_projection_of_a_onto_b_l2075_207574

def a : Fin 2 → ℝ := ![3, 4]
def b : Fin 2 → ℝ := ![1, 2]

theorem projection_of_a_onto_b :
  let proj := (((a 0) * (b 0) + (a 1) * (b 1)) / ((b 0)^2 + (b 1)^2)) • b
  proj 0 = 11/5 ∧ proj 1 = 22/5 := by
  sorry

end NUMINAMATH_CALUDE_projection_of_a_onto_b_l2075_207574


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2075_207576

theorem repeating_decimal_sum (a b : ℕ) : 
  (5 : ℚ) / 13 = (a * 10 + b : ℚ) / 99 → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2075_207576


namespace NUMINAMATH_CALUDE_students_remaining_after_four_stops_l2075_207556

theorem students_remaining_after_four_stops :
  let initial_students : ℕ := 60
  let stops : ℕ := 4
  let fraction_remaining : ℚ := 2 / 3
  let final_students := initial_students * fraction_remaining ^ stops
  final_students = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_remaining_after_four_stops_l2075_207556


namespace NUMINAMATH_CALUDE_school_girls_count_l2075_207534

theorem school_girls_count (total_students : ℕ) (boys_girls_difference : ℕ) 
  (h1 : total_students = 1250)
  (h2 : boys_girls_difference = 124) : 
  ∃ (girls : ℕ), girls = 563 ∧ 
  girls + (girls + boys_girls_difference) = total_students :=
sorry

end NUMINAMATH_CALUDE_school_girls_count_l2075_207534


namespace NUMINAMATH_CALUDE_pq_ratio_implies_pg_ps_ratio_l2075_207527

/-- Triangle PQR with angle bisector PS intersecting MN at G -/
structure Triangle (P Q R S M N G : ℝ × ℝ) :=
  (M_on_PQ : ∃ t : ℝ, M = (1 - t) • P + t • Q ∧ 0 ≤ t ∧ t ≤ 1)
  (N_on_PR : ∃ t : ℝ, N = (1 - t) • P + t • R ∧ 0 ≤ t ∧ t ≤ 1)
  (S_angle_bisector : ∃ t : ℝ, S = (1 - t) • P + t • ((Q + R) / 2) ∧ 0 < t)
  (G_on_MN : ∃ t : ℝ, G = (1 - t) • M + t • N ∧ 0 ≤ t ∧ t ≤ 1)
  (G_on_PS : ∃ t : ℝ, G = (1 - t) • P + t • S ∧ 0 ≤ t ∧ t ≤ 1)

/-- The main theorem -/
theorem pq_ratio_implies_pg_ps_ratio 
  (P Q R S M N G : ℝ × ℝ) 
  (h : Triangle P Q R S M N G) 
  (hPM_MQ : ∃ (t : ℝ), M = (1 - t) • P + t • Q ∧ t = 1/4) 
  (hPN_NR : ∃ (t : ℝ), N = (1 - t) • P + t • R ∧ t = 1/4) :
  ∃ (t : ℝ), G = (1 - t) • P + t • S ∧ t = 5/18 :=
sorry

end NUMINAMATH_CALUDE_pq_ratio_implies_pg_ps_ratio_l2075_207527


namespace NUMINAMATH_CALUDE_unique_common_difference_l2075_207542

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℝ  -- first term
  d : ℝ  -- common difference
  n : ℕ  -- number of terms
  third_term_is_7 : a + 2 * d = 7
  last_term_is_37 : a + (n - 1) * d = 37
  sum_is_198 : n * (2 * a + (n - 1) * d) / 2 = 198

/-- Theorem stating the existence and uniqueness of the common difference -/
theorem unique_common_difference (seq : ArithmeticSequence) : 
  ∃! d : ℝ, seq.d = d := by sorry

end NUMINAMATH_CALUDE_unique_common_difference_l2075_207542


namespace NUMINAMATH_CALUDE_movie_theater_screens_l2075_207547

theorem movie_theater_screens (open_hours : ℕ) (movie_duration : ℕ) (total_movies : ℕ) : 
  open_hours = 8 → movie_duration = 2 → total_movies = 24 → 
  (total_movies * movie_duration) / open_hours = 6 :=
by
  sorry

#check movie_theater_screens

end NUMINAMATH_CALUDE_movie_theater_screens_l2075_207547
