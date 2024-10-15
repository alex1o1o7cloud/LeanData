import Mathlib

namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l3870_387066

theorem factorization_difference_of_squares (y : ℝ) : 1 - 4 * y^2 = (1 - 2*y) * (1 + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l3870_387066


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3870_387069

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ+,
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 2 + (1 : ℚ) / z →
  (x = 1 ∧ y = 2 ∧ z = 1) ∨
  (x = 2 ∧ ((y = 1 ∧ z = 1) ∨ (y = z ∧ y ≥ 2))) ∨
  (x = 3 ∧ ((y = 3 ∧ z = 6) ∨ (y = 4 ∧ z = 12) ∨ (y = 5 ∧ z = 30) ∨ (y = 2 ∧ z = 3))) ∨
  (x ≥ 4 ∧ y ≥ 4 → False) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3870_387069


namespace NUMINAMATH_CALUDE_mod_sum_powers_l3870_387086

theorem mod_sum_powers (n : ℕ) : (36^1724 + 18^1724) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_sum_powers_l3870_387086


namespace NUMINAMATH_CALUDE_fraction_value_l3870_387004

theorem fraction_value : (1 * 2 * 3 * 4) / (1 + 2 + 3 + 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3870_387004


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3870_387061

theorem fraction_decomposition (x y A B : ℝ) (h : x * y ≠ 0) (h' : x + y ≠ 5) :
  (7 * x - 20 * y + 3) / (3 * x^2 * y + 2 * x * y^2 - 15 * x * y) =
  A / (x * y + 5) + B / (3 * x * y - 3) →
  A = -6.333 ∧ B = -3 := by
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3870_387061


namespace NUMINAMATH_CALUDE_leading_digit_logarithm_l3870_387070

-- Define a function to get the leading digit of a real number
noncomputable def leadingDigit (x : ℝ) : ℕ := sorry

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) : ℝ := sorry

-- State the theorem
theorem leading_digit_logarithm (M : ℝ) (a : ℕ) :
  (leadingDigit (6 * 47 * log10 M) = a) →
  ((leadingDigit (log10 (1000 / M)) = 3 - a) ∨
   (leadingDigit (log10 (1000 / M)) = 2 - a)) :=
by sorry

end NUMINAMATH_CALUDE_leading_digit_logarithm_l3870_387070


namespace NUMINAMATH_CALUDE_bakery_problem_solution_l3870_387051

/-- Represents the problem of buying sandwiches and cakes --/
def BakeryProblem (total_money sandwich_cost cake_cost max_items : ℚ) : Prop :=
  ∃ (sandwiches cakes : ℕ),
    sandwiches * sandwich_cost + cakes * cake_cost ≤ total_money ∧
    sandwiches + cakes ≤ max_items ∧
    ∀ (s c : ℕ),
      s * sandwich_cost + c * cake_cost ≤ total_money →
      s + c ≤ max_items →
      s + c ≤ sandwiches + cakes

/-- The maximum number of items that can be bought is 12 --/
theorem bakery_problem_solution :
  BakeryProblem 50 5 (5/2) 12 →
  ∃ (sandwiches cakes : ℕ), sandwiches + cakes = 12 :=
by sorry

end NUMINAMATH_CALUDE_bakery_problem_solution_l3870_387051


namespace NUMINAMATH_CALUDE_unique_integer_sum_l3870_387027

theorem unique_integer_sum : ∃! (b₃ b₄ b₅ b₆ b₇ b₈ : ℕ),
  (11 : ℚ) / 9 = b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 + b₈ / 40320 ∧
  b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧ b₈ < 8 ∧
  b₃ + b₄ + b₅ + b₆ + b₇ + b₈ = 25 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_sum_l3870_387027


namespace NUMINAMATH_CALUDE_english_chinese_difference_l3870_387040

/-- Represents the number of hours Ryan spends studying each subject on weekdays and weekends --/
structure StudyHours where
  english_weekday : ℕ
  chinese_weekday : ℕ
  english_weekend : ℕ
  chinese_weekend : ℕ

/-- Calculates the total hours spent on a subject in a week --/
def total_hours (hours : StudyHours) (weekday : ℕ) (weekend : ℕ) : ℕ :=
  hours.english_weekday * weekday + hours.chinese_weekday * weekday +
  hours.english_weekend * weekend + hours.chinese_weekend * weekend

/-- Theorem stating the difference in hours spent on English vs Chinese --/
theorem english_chinese_difference (hours : StudyHours) 
  (h1 : hours.english_weekday = 6)
  (h2 : hours.chinese_weekday = 3)
  (h3 : hours.english_weekend = 2)
  (h4 : hours.chinese_weekend = 1)
  : total_hours hours 5 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_english_chinese_difference_l3870_387040


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_problem_l3870_387093

theorem consecutive_odd_integers_problem (x : ℤ) (k : ℕ) : 
  (x % 2 = 1) →  -- x is odd
  ((x + 2) % 2 = 1) →  -- x + 2 is odd
  ((x + 4) % 2 = 1) →  -- x + 4 is odd
  (x + (x + 4) = k * (x + 2) - 131) →  -- sum of 1st and 3rd is 131 less than k times 2nd
  (x + (x + 2) + (x + 4) = 133) →  -- sum of all three is 133
  (k = 2) := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_problem_l3870_387093


namespace NUMINAMATH_CALUDE_unique_solution_rational_equation_l3870_387087

theorem unique_solution_rational_equation :
  ∃! x : ℝ, x ≠ -2 ∧ (x^2 + 2*x - 8)/(x + 2) = 3*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_rational_equation_l3870_387087


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3870_387039

open Set

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (9 - x^2)}
def B : Set ℝ := {x | ∃ y, y = Real.sqrt (4*x - x^2)}

-- State the theorem
theorem intersection_complement_equality :
  A ∩ (Bᶜ) = Ioo (-3) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3870_387039


namespace NUMINAMATH_CALUDE_compare_fractions_l3870_387025

theorem compare_fractions : -3/2 > -(5/3) := by sorry

end NUMINAMATH_CALUDE_compare_fractions_l3870_387025


namespace NUMINAMATH_CALUDE_tom_charges_twelve_l3870_387003

/-- Represents Tom's lawn mowing business --/
structure LawnBusiness where
  gas_cost : ℕ
  lawns_mowed : ℕ
  extra_income : ℕ
  total_profit : ℕ

/-- Calculates the price per lawn given Tom's business details --/
def price_per_lawn (b : LawnBusiness) : ℚ :=
  (b.total_profit + b.gas_cost - b.extra_income) / b.lawns_mowed

/-- Theorem stating that Tom charges $12 per lawn --/
theorem tom_charges_twelve (tom : LawnBusiness) 
  (h1 : tom.gas_cost = 17)
  (h2 : tom.lawns_mowed = 3)
  (h3 : tom.extra_income = 10)
  (h4 : tom.total_profit = 29) : 
  price_per_lawn tom = 12 := by
  sorry


end NUMINAMATH_CALUDE_tom_charges_twelve_l3870_387003


namespace NUMINAMATH_CALUDE_probability_ratio_l3870_387082

def total_slips : ℕ := 50
def numbers_per_slip : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 4

def probability_same_number (total : ℕ) (per_number : ℕ) (numbers : ℕ) (drawn : ℕ) : ℚ :=
  (numbers * Nat.choose per_number drawn) / Nat.choose total drawn

def probability_three_same_one_different (total : ℕ) (per_number : ℕ) (numbers : ℕ) (drawn : ℕ) : ℚ :=
  (numbers * Nat.choose per_number (drawn - 1) * (numbers - 1) * per_number) / Nat.choose total drawn

theorem probability_ratio :
  probability_three_same_one_different total_slips slips_per_number numbers_per_slip drawn_slips /
  probability_same_number total_slips slips_per_number numbers_per_slip drawn_slips = 90 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l3870_387082


namespace NUMINAMATH_CALUDE_vector_operation_l3870_387009

def a : Fin 2 → ℝ := ![1, 1]
def b : Fin 2 → ℝ := ![1, -1]

theorem vector_operation :
  (3 • a - 2 • b) = ![1, 5] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3870_387009


namespace NUMINAMATH_CALUDE_melissa_games_played_l3870_387072

def total_points : ℕ := 81
def points_per_game : ℕ := 27

theorem melissa_games_played :
  total_points / points_per_game = 3 := by sorry

end NUMINAMATH_CALUDE_melissa_games_played_l3870_387072


namespace NUMINAMATH_CALUDE_roots_on_circle_l3870_387047

theorem roots_on_circle (z : ℂ) : 
  (z + 1)^4 = 16 * z^4 → Complex.abs (z - Complex.ofReal (1/3)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_roots_on_circle_l3870_387047


namespace NUMINAMATH_CALUDE_probability_two_ones_l3870_387080

def num_dice : ℕ := 15
def num_sides : ℕ := 6
def target_num : ℕ := 1
def target_count : ℕ := 2

theorem probability_two_ones (n : ℕ) (k : ℕ) (p : ℚ) :
  n = num_dice →
  k = target_count →
  p = 1 / num_sides →
  (n.choose k * p^k * (1 - p)^(n - k) : ℚ) = (105 * 5^13 : ℚ) / 6^14 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_ones_l3870_387080


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3870_387029

theorem divisibility_equivalence (a b : ℤ) :
  (13 ∣ (2 * a + 3 * b)) ↔ (13 ∣ (2 * b - 3 * a)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3870_387029


namespace NUMINAMATH_CALUDE_total_money_is_36000_l3870_387063

/-- The number of phones Vivienne has -/
def vivienne_phones : ℕ := 40

/-- The difference in number of phones between Aliyah and Vivienne -/
def phone_difference : ℕ := 10

/-- The price of each phone -/
def phone_price : ℕ := 400

/-- The total amount of money Aliyah and Vivienne have together after selling their phones -/
def total_money : ℕ := (vivienne_phones + (vivienne_phones + phone_difference)) * phone_price

theorem total_money_is_36000 : total_money = 36000 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_36000_l3870_387063


namespace NUMINAMATH_CALUDE_cabbage_production_l3870_387016

theorem cabbage_production (last_year_side : ℕ) (this_year_side : ℕ) : 
  (this_year_side : ℤ)^2 - (last_year_side : ℤ)^2 = 127 →
  this_year_side^2 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_cabbage_production_l3870_387016


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3870_387055

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3870_387055


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_2018_l3870_387077

def a (n : ℕ) : ℕ := 2 * 10^(n+2) + 18

theorem infinitely_many_divisible_by_2018 :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, 2018 ∣ a n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_2018_l3870_387077


namespace NUMINAMATH_CALUDE_slices_per_pizza_large_pizza_has_12_slices_l3870_387015

/-- Calculates the number of slices in a large pizza based on soccer team statistics -/
theorem slices_per_pizza (num_pizzas : ℕ) (num_games : ℕ) (avg_goals_per_game : ℕ) : ℕ :=
  let total_goals := num_games * avg_goals_per_game
  let total_slices := total_goals
  total_slices / num_pizzas

/-- Proves that a large pizza has 12 slices given the problem conditions -/
theorem large_pizza_has_12_slices :
  slices_per_pizza 6 8 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_slices_per_pizza_large_pizza_has_12_slices_l3870_387015


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l3870_387008

/-- A complex cube root of unity -/
noncomputable def ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

/-- The theorem statement -/
theorem complex_ratio_theorem (a b c : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a / b = b / c) (hbc : b / c = c / a) :
  (a + b - c) / (a - b + c) = 1 ∨
  (a + b - c) / (a - b + c) = ω ∨
  (a + b - c) / (a - b + c) = ω^2 :=
sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l3870_387008


namespace NUMINAMATH_CALUDE_cone_base_radius_l3870_387057

/-- Given a cone with slant height 5 cm and lateral area 15π cm², 
    the radius of its base circle is 3 cm. -/
theorem cone_base_radius (s : ℝ) (A : ℝ) (r : ℝ) : 
  s = 5 →  -- slant height is 5 cm
  A = 15 * Real.pi →  -- lateral area is 15π cm²
  A = Real.pi * r * s →  -- formula for lateral area of a cone
  r = 3 :=  -- radius of base circle is 3 cm
by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3870_387057


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l3870_387014

theorem geometric_arithmetic_sequence (x y z : ℝ) 
  (h1 : (4 * y)^2 = (3 * x) * (5 * z))  -- Geometric sequence condition
  (h2 : 2 / y = 1 / x + 1 / z)          -- Arithmetic sequence condition
  : x / z + z / x = 34 / 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l3870_387014


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l3870_387019

theorem polygon_interior_angles (n : ℕ) (extra_angle : ℝ) : 
  (n ≥ 3) →
  (180 * (n - 2) + extra_angle = 1800) →
  (n = 11 ∧ extra_angle = 180) :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l3870_387019


namespace NUMINAMATH_CALUDE_line_plane_relationship_l3870_387099

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (l : Line) (α : Plane) (m : Line)
  (h : parallel l m ∧ contained_in m α) :
  contained_in l α ∨ parallel_plane l α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l3870_387099


namespace NUMINAMATH_CALUDE_three_digit_number_appended_to_1220_l3870_387002

theorem three_digit_number_appended_to_1220 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (1220000 + n) % 2014 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_three_digit_number_appended_to_1220_l3870_387002


namespace NUMINAMATH_CALUDE_sam_final_penny_count_l3870_387091

def initial_pennies : ℕ := 980
def found_pennies : ℕ := 930
def exchanged_pennies : ℕ := 725
def gift_pennies : ℕ := 250

theorem sam_final_penny_count :
  initial_pennies + found_pennies - exchanged_pennies + gift_pennies = 1435 :=
by sorry

end NUMINAMATH_CALUDE_sam_final_penny_count_l3870_387091


namespace NUMINAMATH_CALUDE_bus_capacity_equality_l3870_387078

theorem bus_capacity_equality (x : ℕ) : 50 * x + 10 = 52 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_equality_l3870_387078


namespace NUMINAMATH_CALUDE_solve_brownies_problem_l3870_387031

def brownies_problem (initial_brownies : ℕ) (remaining_brownies : ℕ) : Prop :=
  let admin_brownies := initial_brownies / 2
  let after_admin := initial_brownies - admin_brownies
  let carl_brownies := after_admin / 2
  let after_carl := after_admin - carl_brownies
  let final_brownies := 3
  ∃ (simon_brownies : ℕ), 
    simon_brownies = after_carl - final_brownies ∧
    simon_brownies = 2

theorem solve_brownies_problem :
  brownies_problem 20 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_brownies_problem_l3870_387031


namespace NUMINAMATH_CALUDE_misread_weight_calculation_l3870_387011

theorem misread_weight_calculation (n : ℕ) (initial_avg correct_avg correct_weight : ℝ) :
  n = 20 ∧ 
  initial_avg = 58.4 ∧ 
  correct_avg = 58.6 ∧ 
  correct_weight = 60 →
  ∃ misread_weight : ℝ, 
    misread_weight = 56 ∧
    n * correct_avg - n * initial_avg = correct_weight - misread_weight :=
by sorry

end NUMINAMATH_CALUDE_misread_weight_calculation_l3870_387011


namespace NUMINAMATH_CALUDE_uncovered_area_three_circles_l3870_387032

theorem uncovered_area_three_circles (R : ℝ) (h : R = 10) :
  let r := R / 2
  let larger_circle_area := π * R^2
  let smaller_circle_area := π * r^2
  let total_smaller_circles_area := 3 * smaller_circle_area
  larger_circle_area - total_smaller_circles_area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_uncovered_area_three_circles_l3870_387032


namespace NUMINAMATH_CALUDE_base_conversion_equality_l3870_387054

/-- Given that the base 6 number 62₆ is equal to the base b number 124ᵦ,
    prove that the unique positive integer solution for b is 4. -/
theorem base_conversion_equality : ∃! (b : ℕ), b > 0 ∧ (6 * 6 + 2) = (1 * b^2 + 2 * b + 4) := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l3870_387054


namespace NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l3870_387046

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 12 = 43200 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l3870_387046


namespace NUMINAMATH_CALUDE_meal_combinations_eq_100_l3870_387098

/-- The number of items on the menu -/
def menu_items : ℕ := 10

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- The number of different combinations of meals that can be ordered -/
def meal_combinations : ℕ := menu_items ^ num_people

/-- Theorem stating that the number of meal combinations is 100 -/
theorem meal_combinations_eq_100 : meal_combinations = 100 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_eq_100_l3870_387098


namespace NUMINAMATH_CALUDE_train_length_l3870_387017

/-- Given a train traveling at 45 km/hr that crosses a 220.03-meter bridge in 30 seconds,
    the length of the train is 154.97 meters. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 →
  bridge_length = 220.03 →
  crossing_time = 30 →
  (train_speed * 1000 / 3600) * crossing_time - bridge_length = 154.97 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3870_387017


namespace NUMINAMATH_CALUDE_certain_event_l3870_387094

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents a bag of balls -/
def Bag := List Color

/-- Represents the result of drawing two balls -/
def Draw := (Color × Color)

/-- The bag containing 2 red balls and 1 white ball -/
def initialBag : Bag := [Color.Red, Color.Red, Color.White]

/-- Function to check if a draw contains at least one red ball -/
def hasRed (draw : Draw) : Prop :=
  draw.1 = Color.Red ∨ draw.2 = Color.Red

/-- All possible draws from the bag -/
def allDraws : List Draw := [
  (Color.Red, Color.Red),
  (Color.Red, Color.White),
  (Color.White, Color.Red)
]

/-- Theorem stating that any draw from the bag must contain at least one red ball -/
theorem certain_event : ∀ (draw : Draw), draw ∈ allDraws → hasRed draw := by sorry

end NUMINAMATH_CALUDE_certain_event_l3870_387094


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3870_387097

theorem cubic_root_sum (k₁ k₂ : ℝ) (h : k₁ + k₂ ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ x^3 - k₁*x - k₂
  let roots := { x : ℝ | f x = 0 }
  ∃ (a b c : ℝ), a ∈ roots ∧ b ∈ roots ∧ c ∈ roots ∧
    (1+a)/(1-a) + (1+b)/(1-b) + (1+c)/(1-c) = (3 + k₁ + 3*k₂) / (1 - k₁ - k₂) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3870_387097


namespace NUMINAMATH_CALUDE_derivative_of_ln_2_minus_3x_l3870_387033

open Real

theorem derivative_of_ln_2_minus_3x (x : ℝ) : 
  deriv (λ x => Real.log (2 - 3*x)) x = 3 / (3*x - 2) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_ln_2_minus_3x_l3870_387033


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3870_387030

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < 1}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3870_387030


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3870_387089

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧  -- four-digit number
  (let a := n / 1000
   let b := (n / 100) % 10
   let c := (n / 10) % 10
   let d := n % 10
   1000 * c + 100 * d + 10 * a + b - n = 5940) ∧  -- swapping condition
  n % 9 = 8 ∧  -- divisibility condition
  n % 2 = 1  -- odd number

theorem smallest_valid_number :
  is_valid_number 1979 ∧ ∀ m : ℕ, is_valid_number m → m ≥ 1979 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3870_387089


namespace NUMINAMATH_CALUDE_two_books_from_different_genres_l3870_387092

theorem two_books_from_different_genres :
  let num_genres : ℕ := 3
  let books_per_genre : ℕ := 4
  let choose_genres : ℕ := 2
  (num_genres.choose choose_genres) * books_per_genre * books_per_genre = 48 :=
by sorry

end NUMINAMATH_CALUDE_two_books_from_different_genres_l3870_387092


namespace NUMINAMATH_CALUDE_max_value_constraint_l3870_387088

theorem max_value_constraint (m : ℝ) : m > 1 →
  (∃ (x y : ℝ), y ≥ x ∧ y ≤ m * x ∧ x + y ≤ 1 ∧ x + m * y < 2) ↔ m < 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3870_387088


namespace NUMINAMATH_CALUDE_proof_by_contradiction_on_incorrect_statement_l3870_387075

-- Define a proposition
variable (P : Prop)

-- Define the property of being an incorrect statement
def is_incorrect (S : Prop) : Prop := ¬S

-- Define the process of attempting proof by contradiction
def attempt_proof_by_contradiction (S : Prop) : Prop :=
  ∃ (proof : ¬S → False), True

-- Define what it means for a proof method to fail to produce a useful conclusion
def fails_to_produce_useful_conclusion (S : Prop) : Prop :=
  ¬(S ∨ ¬S)

-- Theorem statement
theorem proof_by_contradiction_on_incorrect_statement
  (h : is_incorrect P) :
  attempt_proof_by_contradiction P →
  fails_to_produce_useful_conclusion P :=
by sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_on_incorrect_statement_l3870_387075


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3870_387096

theorem trigonometric_identity : 
  (Real.sqrt 3) / (Real.cos (10 * π / 180)) - 1 / (Real.sin (170 * π / 180)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3870_387096


namespace NUMINAMATH_CALUDE_cost_of_seven_sandwiches_six_sodas_l3870_387043

/-- Calculates the total cost of purchasing sandwiches and sodas at Sally's Snack Shop -/
def snack_shop_cost (sandwich_count : ℕ) (soda_count : ℕ) : ℕ :=
  let sandwich_price := 4
  let soda_price := 3
  let bulk_discount := 10
  let total_items := sandwich_count + soda_count
  let total_cost := sandwich_count * sandwich_price + soda_count * soda_price
  if total_items > 10 then total_cost - bulk_discount else total_cost

/-- Theorem stating that purchasing 7 sandwiches and 6 sodas costs $36 -/
theorem cost_of_seven_sandwiches_six_sodas :
  snack_shop_cost 7 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_seven_sandwiches_six_sodas_l3870_387043


namespace NUMINAMATH_CALUDE_sum_of_distinct_roots_l3870_387007

theorem sum_of_distinct_roots (x y : ℝ) (h1 : x ≠ y) (h2 : x^2 - 2000*x = y^2 - 2000*y) : x + y = 2000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_roots_l3870_387007


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l3870_387020

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sequence_is_arithmetic (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a (n + 1) - a n = 3) : 
  is_arithmetic_sequence a :=
sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l3870_387020


namespace NUMINAMATH_CALUDE_sqrt_144_div_6_l3870_387083

theorem sqrt_144_div_6 : Real.sqrt 144 / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_144_div_6_l3870_387083


namespace NUMINAMATH_CALUDE_sunflower_seed_contest_l3870_387076

theorem sunflower_seed_contest (total_seeds : ℕ) (first_player : ℕ) (second_player : ℕ) 
  (h1 : total_seeds = 214)
  (h2 : first_player = 78)
  (h3 : second_player = 53)
  (h4 : total_seeds = first_player + second_player + (total_seeds - first_player - second_player))
  (h5 : total_seeds - first_player - second_player > second_player) :
  (total_seeds - first_player - second_player) - second_player = 30 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_seed_contest_l3870_387076


namespace NUMINAMATH_CALUDE_first_year_interest_l3870_387062

theorem first_year_interest (initial_deposit : ℝ) (first_year_balance : ℝ) 
  (second_year_increase : ℝ) (total_increase : ℝ) :
  initial_deposit = 500 →
  first_year_balance = 600 →
  second_year_increase = 0.1 →
  total_increase = 0.32 →
  first_year_balance - initial_deposit = 100 := by
  sorry

end NUMINAMATH_CALUDE_first_year_interest_l3870_387062


namespace NUMINAMATH_CALUDE_probability_theorem_l3870_387067

/-- Represents a standard deck of cards with additional properties -/
structure Deck :=
  (total : ℕ)
  (kings : ℕ)
  (aces : ℕ)
  (others : ℕ)
  (h1 : total = kings + aces + others)

/-- The probability of drawing either two aces or at least one king -/
def probability_two_aces_or_king (d : Deck) : ℚ :=
  sorry

/-- The theorem statement -/
theorem probability_theorem (d : Deck) 
  (h2 : d.total = 54) 
  (h3 : d.kings = 4) 
  (h4 : d.aces = 6) 
  (h5 : d.others = 44) : 
  probability_two_aces_or_king d = 221 / 1431 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3870_387067


namespace NUMINAMATH_CALUDE_parabola_vertex_l3870_387048

/-- The vertex of a parabola defined by y = -2x^2 + 3 is (0, 3) -/
theorem parabola_vertex :
  let f : ℝ → ℝ := λ x => -2 * x^2 + 3
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = 3 ∧ ∀ x : ℝ, f x ≤ f p.1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3870_387048


namespace NUMINAMATH_CALUDE_problem_proof_l3870_387052

theorem problem_proof (k n : ℤ) : 
  (5 + k) * (5 - k) = n - (2^3) → k = 2 → n = 29 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l3870_387052


namespace NUMINAMATH_CALUDE_crane_folding_theorem_l3870_387060

/-- The number of cranes Hyerin folds per day -/
def hyerin_cranes_per_day : ℕ := 16

/-- The number of days Hyerin folds cranes -/
def hyerin_days : ℕ := 7

/-- The number of cranes Taeyeong folds per day -/
def taeyeong_cranes_per_day : ℕ := 25

/-- The number of days Taeyeong folds cranes -/
def taeyeong_days : ℕ := 6

/-- The total number of cranes folded by Hyerin and Taeyeong -/
def total_cranes : ℕ := hyerin_cranes_per_day * hyerin_days + taeyeong_cranes_per_day * taeyeong_days

theorem crane_folding_theorem : total_cranes = 262 := by
  sorry

end NUMINAMATH_CALUDE_crane_folding_theorem_l3870_387060


namespace NUMINAMATH_CALUDE_largest_fraction_l3870_387045

theorem largest_fraction :
  let a := 35 / 69
  let b := 7 / 15
  let c := 9 / 19
  let d := 399 / 799
  let e := 150 / 299
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l3870_387045


namespace NUMINAMATH_CALUDE_balls_in_boxes_count_l3870_387068

/-- The number of ways to place three distinct balls into three distinct boxes -/
def place_balls_in_boxes : ℕ := 27

/-- The number of choices for each ball -/
def choices_per_ball : ℕ := 3

/-- Theorem: The number of ways to place three distinct balls into three distinct boxes
    is equal to the cube of the number of choices for each ball -/
theorem balls_in_boxes_count :
  place_balls_in_boxes = choices_per_ball ^ 3 := by sorry

end NUMINAMATH_CALUDE_balls_in_boxes_count_l3870_387068


namespace NUMINAMATH_CALUDE_log_product_equality_l3870_387000

theorem log_product_equality : (Real.log 2 / Real.log 3 + Real.log 5 / Real.log 3) * (Real.log 9 / Real.log 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equality_l3870_387000


namespace NUMINAMATH_CALUDE_gain_percentage_calculation_l3870_387005

theorem gain_percentage_calculation (selling_price gain : ℝ) : 
  selling_price = 225 → gain = 75 → (gain / (selling_price - gain)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gain_percentage_calculation_l3870_387005


namespace NUMINAMATH_CALUDE_vitamin_d3_capsules_per_bottle_l3870_387073

/-- Calculates the number of capsules in each bottle given the total days, 
    daily serving size, and total number of bottles. -/
def capsules_per_bottle (total_days : ℕ) (daily_serving : ℕ) (total_bottles : ℕ) : ℕ :=
  (total_days * daily_serving) / total_bottles

/-- Theorem stating that given the specific conditions, the number of capsules
    per bottle is 60. -/
theorem vitamin_d3_capsules_per_bottle :
  capsules_per_bottle 180 2 6 = 60 := by
  sorry

#eval capsules_per_bottle 180 2 6

end NUMINAMATH_CALUDE_vitamin_d3_capsules_per_bottle_l3870_387073


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3870_387079

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = q * a n)

/-- The arithmetic sequence property for three terms -/
def ArithmeticSequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  ArithmeticSequence (3 * a 1) (2 * a 2) ((1/2) * a 3) →
  q = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3870_387079


namespace NUMINAMATH_CALUDE_test_score_after_5_hours_l3870_387058

/-- A student's test score is directly proportional to study time -/
structure TestScore where
  maxPoints : ℝ
  scoreAfter2Hours : ℝ
  hoursStudied : ℝ
  score : ℝ
  proportional : scoreAfter2Hours / 2 = score / hoursStudied

/-- The theorem to prove -/
theorem test_score_after_5_hours (test : TestScore) 
  (h1 : test.maxPoints = 150)
  (h2 : test.scoreAfter2Hours = 90)
  (h3 : test.hoursStudied = 5) : 
  test.score = 225 := by
sorry

end NUMINAMATH_CALUDE_test_score_after_5_hours_l3870_387058


namespace NUMINAMATH_CALUDE_sam_distance_l3870_387074

/-- Proves that Sam drove 200 miles given the conditions of the problem -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) : 
  (marguerite_distance / marguerite_time) * sam_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_sam_distance_l3870_387074


namespace NUMINAMATH_CALUDE_third_grade_agreement_l3870_387090

theorem third_grade_agreement (total_agreed : ℕ) (fourth_grade_agreed : ℕ) 
  (h1 : total_agreed = 391) (h2 : fourth_grade_agreed = 237) :
  total_agreed - fourth_grade_agreed = 154 := by
  sorry

end NUMINAMATH_CALUDE_third_grade_agreement_l3870_387090


namespace NUMINAMATH_CALUDE_marcus_points_l3870_387023

/-- Proves that Marcus scored 28 points in the basketball game -/
theorem marcus_points (total_points : ℕ) (other_players : ℕ) (avg_points : ℕ) : 
  total_points = 63 → other_players = 5 → avg_points = 7 → 
  total_points - (other_players * avg_points) = 28 := by
  sorry

end NUMINAMATH_CALUDE_marcus_points_l3870_387023


namespace NUMINAMATH_CALUDE_odd_sum_games_exists_l3870_387022

theorem odd_sum_games_exists (n : ℕ) (h : n = 15) : 
  ∃ (i j : ℕ) (games_played : ℕ → ℕ), 
    i < n ∧ j < n ∧ i ≠ j ∧ 
    (games_played i + games_played j) % 2 = 1 ∧
    ∀ k, k < n → games_played k ≤ n - 2 :=
by sorry

end NUMINAMATH_CALUDE_odd_sum_games_exists_l3870_387022


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3870_387037

-- Define the first equation
def equation1 (x : ℝ) : Prop := (x + 1) * (x + 3) = 15

-- Define the second equation
def equation2 (y : ℝ) : Prop := (y - 3)^2 + 3*(y - 3) + 2 = 0

-- Theorem for the first equation
theorem solution_equation1 : 
  (∃ x : ℝ, equation1 x) ∧ 
  (∀ x : ℝ, equation1 x ↔ (x = -6 ∨ x = 2)) :=
sorry

-- Theorem for the second equation
theorem solution_equation2 : 
  (∃ y : ℝ, equation2 y) ∧ 
  (∀ y : ℝ, equation2 y ↔ (y = 1 ∨ y = 2)) :=
sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3870_387037


namespace NUMINAMATH_CALUDE_max_sum_lcm_165_l3870_387028

theorem max_sum_lcm_165 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  Nat.lcm (Nat.lcm (Nat.lcm a.val b.val) c.val) d.val = 165 →
  a.val + b.val + c.val + d.val ≤ 268 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_lcm_165_l3870_387028


namespace NUMINAMATH_CALUDE_min_sum_on_parabola_l3870_387001

theorem min_sum_on_parabola :
  ∀ n m : ℕ,
  m = 19 * n^2 - 98 * n →
  102 ≤ m + n :=
by sorry

end NUMINAMATH_CALUDE_min_sum_on_parabola_l3870_387001


namespace NUMINAMATH_CALUDE_min_distance_proof_l3870_387026

/-- The distance between the graphs of y = 2x and y = -x^2 - 2x - 1 at a given x -/
def distance (x : ℝ) : ℝ := 2*x - (-x^2 - 2*x - 1)

/-- The minimum non-negative distance between the graphs -/
def min_distance : ℝ := 1

theorem min_distance_proof : 
  ∀ x : ℝ, distance x ≥ 0 → distance x ≥ min_distance :=
sorry

end NUMINAMATH_CALUDE_min_distance_proof_l3870_387026


namespace NUMINAMATH_CALUDE_check_payment_inequality_l3870_387010

theorem check_payment_inequality (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 → 
  10 ≤ y ∧ y ≤ 99 → 
  100 * y + x - (100 * x + y) = 2156 →
  100 * y + x < 2 * (100 * x + y) := by
  sorry

end NUMINAMATH_CALUDE_check_payment_inequality_l3870_387010


namespace NUMINAMATH_CALUDE_journey_duration_l3870_387056

-- Define the distance covered by the train
def distance : ℝ := 80

-- Define the average speed of the train
def average_speed : ℝ := 10

-- Theorem: The duration of the journey is 8 seconds
theorem journey_duration : (distance / average_speed) = 8 := by
  sorry

end NUMINAMATH_CALUDE_journey_duration_l3870_387056


namespace NUMINAMATH_CALUDE_equilateral_triangle_construction_l3870_387042

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def rotatePoint (p : ℝ × ℝ) (center : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ := sorry

theorem equilateral_triangle_construction 
  (A : ℝ × ℝ) (S₁ S₂ : Circle) : 
  ∃ (B C : ℝ × ℝ), 
    (∃ (t : ℝ), B = rotatePoint C A (-π/3)) ∧
    (∃ (t : ℝ), C = rotatePoint B A (π/3)) ∧
    (∃ (t : ℝ), B ∈ {p | (p.1 - S₁.center.1)^2 + (p.2 - S₁.center.2)^2 = S₁.radius^2}) ∧
    (∃ (t : ℝ), C ∈ {p | (p.1 - S₂.center.1)^2 + (p.2 - S₂.center.2)^2 = S₂.radius^2}) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_construction_l3870_387042


namespace NUMINAMATH_CALUDE_vhs_to_dvd_cost_l3870_387050

def replace_cost (num_movies : ℕ) (trade_in_price : ℕ) (dvd_price : ℕ) : ℕ :=
  num_movies * dvd_price - num_movies * trade_in_price

theorem vhs_to_dvd_cost :
  replace_cost 100 2 10 = 800 := by
  sorry

end NUMINAMATH_CALUDE_vhs_to_dvd_cost_l3870_387050


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3870_387081

theorem sum_of_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_sum_product : a + b = 6 * (a * b)) : 1 / a + 1 / b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3870_387081


namespace NUMINAMATH_CALUDE_cupcakes_theorem_l3870_387006

/-- The number of cupcakes when shared equally among children -/
def cupcakes_per_child : ℕ := 12

/-- The number of children sharing the cupcakes -/
def number_of_children : ℕ := 8

/-- The total number of cupcakes -/
def total_cupcakes : ℕ := cupcakes_per_child * number_of_children

theorem cupcakes_theorem : total_cupcakes = 96 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_theorem_l3870_387006


namespace NUMINAMATH_CALUDE_prob_three_red_marbles_l3870_387044

def total_marbles : ℕ := 5 + 6 + 7

def prob_all_red (red white blue : ℕ) : ℚ :=
  (red : ℚ) / total_marbles *
  ((red - 1) : ℚ) / (total_marbles - 1) *
  ((red - 2) : ℚ) / (total_marbles - 2)

theorem prob_three_red_marbles :
  prob_all_red 5 6 7 = 5 / 408 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_red_marbles_l3870_387044


namespace NUMINAMATH_CALUDE_circle_center_condition_l3870_387021

-- Define the circle equation
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

-- Define the condition for the center to be in the third quadrant
def center_in_third_quadrant (k : ℝ) : Prop :=
  k > 0 ∧ -2 < 0

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  k > 4

-- Theorem statement
theorem circle_center_condition (k : ℝ) :
  (∃ x y : ℝ, circle_equation x y k) ∧ 
  center_in_third_quadrant k →
  k_range k :=
by sorry

end NUMINAMATH_CALUDE_circle_center_condition_l3870_387021


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3870_387024

theorem inequality_solution_set (x : ℝ) :
  (3*x - 1) / (2 - x) ≥ 0 ↔ x ∈ {y : ℝ | 1/3 ≤ y ∧ y < 2} :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3870_387024


namespace NUMINAMATH_CALUDE_greatest_b_value_l3870_387013

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 14 ≥ 0 → x ≤ 7) ∧ 
  (-7^2 + 9*7 - 14 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_greatest_b_value_l3870_387013


namespace NUMINAMATH_CALUDE_greatest_multiple_of_five_cubed_less_than_3375_l3870_387095

theorem greatest_multiple_of_five_cubed_less_than_3375 :
  ∃ (x : ℕ), x > 0 ∧ 5 ∣ x ∧ x^3 < 3375 ∧ ∀ (y : ℕ), y > 0 → 5 ∣ y → y^3 < 3375 → y ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_five_cubed_less_than_3375_l3870_387095


namespace NUMINAMATH_CALUDE_savings_calculation_l3870_387071

/-- Represents a person's financial information -/
structure Person where
  income : ℕ
  expenditure : ℕ

/-- The problem setup -/
def financialProblem (p1 p2 : Person) : Prop :=
  p1.income = 3000 ∧
  4 * p1.income = 5 * p2.income ∧
  2 * p1.expenditure = 3 * p2.expenditure ∧
  p1.income - p1.expenditure = p2.income - p2.expenditure

/-- The theorem to prove -/
theorem savings_calculation (p1 p2 : Person) :
  financialProblem p1 p2 → p1.income - p1.expenditure = 1200 := by
  sorry

#check savings_calculation

end NUMINAMATH_CALUDE_savings_calculation_l3870_387071


namespace NUMINAMATH_CALUDE_range_of_sum_and_abs_l3870_387084

theorem range_of_sum_and_abs (a b : ℝ) (ha : 1 ≤ a ∧ a ≤ 3) (hb : -4 < b ∧ b < 2) :
  1 < a + |b| ∧ a + |b| < 7 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_and_abs_l3870_387084


namespace NUMINAMATH_CALUDE_area_AMDN_eq_area_ABC_l3870_387041

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define that ABC is an acute triangle
def is_acute_triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define points E and F on side BC
def E_on_BC (E B C : ℝ × ℝ) : Prop := sorry
def F_on_BC (F B C : ℝ × ℝ) : Prop := sorry

-- Define the angle equality
def angle_BAE_eq_CAF (A B C E F : ℝ × ℝ) : Prop := sorry

-- Define perpendicular lines
def FM_perp_AB (F M A B : ℝ × ℝ) : Prop := sorry
def FN_perp_AC (F N A C : ℝ × ℝ) : Prop := sorry

-- Define D as the intersection of extended AE and the circumcircle
def D_on_circumcircle (A B C D E : ℝ × ℝ) : Prop := sorry

-- Define area function
def area (points : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_AMDN_eq_area_ABC 
  (A B C D E F M N : ℝ × ℝ) 
  (h1 : is_acute_triangle A B C)
  (h2 : E_on_BC E B C)
  (h3 : F_on_BC F B C)
  (h4 : angle_BAE_eq_CAF A B C E F)
  (h5 : FM_perp_AB F M A B)
  (h6 : FN_perp_AC F N A C)
  (h7 : D_on_circumcircle A B C D E) :
  area [A, M, D, N] = area [A, B, C] := by sorry

end NUMINAMATH_CALUDE_area_AMDN_eq_area_ABC_l3870_387041


namespace NUMINAMATH_CALUDE_batsman_new_average_is_38_l3870_387018

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  totalRuns : ℕ
  innings : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (stats : BatsmanStats) (newInningScore : ℕ) : ℚ :=
  (stats.totalRuns + newInningScore) / (stats.innings + 1)

/-- Theorem: Given the conditions, the batsman's new average is 38 -/
theorem batsman_new_average_is_38 
  (stats : BatsmanStats)
  (h1 : stats.innings = 16)
  (h2 : newAverage stats 86 = stats.average + 3) :
  newAverage stats 86 = 38 := by
sorry

end NUMINAMATH_CALUDE_batsman_new_average_is_38_l3870_387018


namespace NUMINAMATH_CALUDE_product_evaluation_l3870_387038

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3870_387038


namespace NUMINAMATH_CALUDE_poster_count_l3870_387034

/-- The total number of posters made by Mario, Samantha, and Jonathan -/
def total_posters (mario_posters samantha_posters jonathan_posters : ℕ) : ℕ :=
  mario_posters + samantha_posters + jonathan_posters

/-- Theorem stating the total number of posters made by Mario, Samantha, and Jonathan -/
theorem poster_count : ∃ (mario_posters samantha_posters jonathan_posters : ℕ),
  mario_posters = 36 ∧
  samantha_posters = mario_posters + 45 ∧
  jonathan_posters = 2 * samantha_posters ∧
  total_posters mario_posters samantha_posters jonathan_posters = 279 :=
by
  sorry

end NUMINAMATH_CALUDE_poster_count_l3870_387034


namespace NUMINAMATH_CALUDE_equal_play_time_l3870_387065

theorem equal_play_time (team_size : ℕ) (field_players : ℕ) (match_duration : ℕ) 
  (h1 : team_size = 10)
  (h2 : field_players = 8)
  (h3 : match_duration = 45)
  (h4 : field_players < team_size) :
  (field_players * match_duration) / team_size = 36 := by
  sorry

end NUMINAMATH_CALUDE_equal_play_time_l3870_387065


namespace NUMINAMATH_CALUDE_quarterly_interest_rate_proof_l3870_387012

/-- Proves that the given annual interest payment for a loan with quarterly compounding
    is consistent with the calculated quarterly interest rate. -/
theorem quarterly_interest_rate_proof
  (principal : ℝ)
  (annual_interest : ℝ)
  (quarterly_rate : ℝ)
  (h_principal : principal = 10000)
  (h_annual_interest : annual_interest = 2155.06)
  (h_quarterly_rate : quarterly_rate = 0.05) :
  annual_interest = principal * ((1 + quarterly_rate) ^ 4 - 1) :=
by sorry

end NUMINAMATH_CALUDE_quarterly_interest_rate_proof_l3870_387012


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l3870_387036

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry

-- Theorem statement
theorem line_perp_parallel_planes 
  (α β : Plane) (l : Line) 
  (h1 : α ≠ β) 
  (h2 : perpendicular l β) 
  (h3 : parallel α β) : 
  perpendicular l α :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l3870_387036


namespace NUMINAMATH_CALUDE_min_value_implies_a_inequality_solution_l3870_387035

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Theorem for part (i)
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f x a ≥ 2) ∧ (∃ x, f x a = 2) → a = 0 ∨ a = -4 := by sorry

-- Theorem for part (ii)
theorem inequality_solution (x : ℝ) :
  f x 2 ≤ 6 ↔ x ∈ Set.Icc (-3) 3 := by sorry

-- Note: Set.Icc represents a closed interval [a, b]

end NUMINAMATH_CALUDE_min_value_implies_a_inequality_solution_l3870_387035


namespace NUMINAMATH_CALUDE_green_blue_difference_l3870_387059

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : ℕ
  ratio : Fin 3 → ℕ
  color_sum : ratio 0 + ratio 1 + ratio 2 = 18

theorem green_blue_difference (bag : DiskBag) 
  (h1 : bag.total = 108)
  (h2 : bag.ratio 0 = 3)  -- Blue
  (h3 : bag.ratio 1 = 7)  -- Yellow
  (h4 : bag.ratio 2 = 8)  -- Green
  : (bag.total / 18 * bag.ratio 2) - (bag.total / 18 * bag.ratio 0) = 30 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_l3870_387059


namespace NUMINAMATH_CALUDE_fashion_line_blend_pieces_l3870_387064

theorem fashion_line_blend_pieces (silk_pieces : ℕ) (cashmere_pieces : ℕ) (total_pieces : ℕ) : 
  silk_pieces = 10 →
  cashmere_pieces = silk_pieces / 2 →
  total_pieces = 13 →
  cashmere_pieces - (total_pieces - silk_pieces) = 2 :=
by sorry

end NUMINAMATH_CALUDE_fashion_line_blend_pieces_l3870_387064


namespace NUMINAMATH_CALUDE_ladybugs_with_spots_l3870_387053

theorem ladybugs_with_spots (total : ℕ) (without_spots : ℕ) (with_spots : ℕ) : 
  total = 67082 → without_spots = 54912 → total = with_spots + without_spots → 
  with_spots = 12170 := by
sorry

end NUMINAMATH_CALUDE_ladybugs_with_spots_l3870_387053


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_fourth_power_l3870_387085

def x : ℕ := 7 * 24 * 48

theorem smallest_y_for_perfect_fourth_power (y : ℕ) :
  y = 6174 ↔ (
    y > 0 ∧
    ∃ (n : ℕ), x * y = n^4 ∧
    ∀ (z : ℕ), 0 < z ∧ z < y → ¬∃ (m : ℕ), x * z = m^4
  ) := by sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_fourth_power_l3870_387085


namespace NUMINAMATH_CALUDE_fahrenheit_celsius_conversion_l3870_387049

theorem fahrenheit_celsius_conversion (F C : ℝ) : C = (5 / 9) * (F - 30) → C = 30 → F = 84 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_celsius_conversion_l3870_387049
