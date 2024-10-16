import Mathlib

namespace NUMINAMATH_CALUDE_shrimp_earnings_l2543_254302

/-- Calculates the earnings of each boy from catching and selling shrimp --/
theorem shrimp_earnings (victor_shrimp : ℕ) (austin_diff : ℕ) (price : ℚ) (price_per : ℕ) :
  victor_shrimp = 26 →
  austin_diff = 8 →
  price = 7 →
  price_per = 11 →
  let austin_shrimp := victor_shrimp - austin_diff
  let brian_shrimp := (victor_shrimp + austin_shrimp) / 2
  let total_shrimp := victor_shrimp + austin_shrimp + brian_shrimp
  let total_earnings := (total_shrimp / price_per : ℚ) * price
  total_earnings / 3 = 14 := by
sorry


end NUMINAMATH_CALUDE_shrimp_earnings_l2543_254302


namespace NUMINAMATH_CALUDE_single_point_conic_section_l2543_254347

theorem single_point_conic_section (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + p.2^2 + 6 * p.1 - 8 * p.2 + d = 0) → d = 19 := by
  sorry

end NUMINAMATH_CALUDE_single_point_conic_section_l2543_254347


namespace NUMINAMATH_CALUDE_beneficial_average_recording_l2543_254307

/-- Proves that recording the average of two new test grades is beneficial
    if the average of previous grades is higher than the average of the new grades -/
theorem beneficial_average_recording (n : ℕ) (x y : ℝ) (h : x / n > y / 2) :
  (x + y) / (n + 2) > (x + y / 2) / (n + 1) := by
  sorry

#check beneficial_average_recording

end NUMINAMATH_CALUDE_beneficial_average_recording_l2543_254307


namespace NUMINAMATH_CALUDE_log_expression_simplification_l2543_254335

theorem log_expression_simplification :
  (1/2) * Real.log (32/49) - (4/3) * Real.log (Real.sqrt 8) + Real.log (Real.sqrt 245) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_simplification_l2543_254335


namespace NUMINAMATH_CALUDE_largest_time_for_85_degrees_l2543_254372

/-- The temperature function in Denver, CO on a specific day -/
def temperature (t : ℝ) : ℝ := -t^2 + 10*t + 60

/-- The largest non-negative real solution to the equation temperature(t) = 85 is 15 -/
theorem largest_time_for_85_degrees :
  (∃ (t : ℝ), t ≥ 0 ∧ temperature t = 85) →
  (∀ (t : ℝ), t ≥ 0 ∧ temperature t = 85 → t ≤ 15) ∧
  (temperature 15 = 85) := by
sorry

end NUMINAMATH_CALUDE_largest_time_for_85_degrees_l2543_254372


namespace NUMINAMATH_CALUDE_combined_sticker_count_l2543_254361

theorem combined_sticker_count 
  (june_initial : ℕ) 
  (bonnie_initial : ℕ) 
  (birthday_gift : ℕ) : 
  june_initial + bonnie_initial + 2 * birthday_gift = 
    (june_initial + birthday_gift) + (bonnie_initial + birthday_gift) := by
  sorry

end NUMINAMATH_CALUDE_combined_sticker_count_l2543_254361


namespace NUMINAMATH_CALUDE_sum_remainder_mod_11_l2543_254310

theorem sum_remainder_mod_11 : 
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_11_l2543_254310


namespace NUMINAMATH_CALUDE_tan_function_property_l2543_254384

/-- 
Given a function f(x) = a * tan(b * x) where a and b are positive constants,
if the function has a period of 2π/5 and passes through the point (π/10, 1),
then the product ab equals 5/2.
-/
theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + 2 * π / 5))) → 
  (a * Real.tan (b * π / 10) = 1) → 
  a * b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_function_property_l2543_254384


namespace NUMINAMATH_CALUDE_ellipse_equation_l2543_254389

/-- The standard equation of an ellipse given its properties -/
theorem ellipse_equation (f1 f2 p : ℝ × ℝ) (other_ellipse : ℝ → ℝ → Prop) :
  f1 = (0, -4) →
  f2 = (0, 4) →
  p = (-3, 2) →
  (∀ x y, other_ellipse x y ↔ x^2/9 + y^2/4 = 1) →
  (∀ x y, (x^2/15 + y^2/10 = 1) ↔
    (∃ d1 d2 : ℝ,
      d1 + d2 = 10 ∧
      d1^2 = (x - f1.1)^2 + (y - f1.2)^2 ∧
      d2^2 = (x - f2.1)^2 + (y - f2.2)^2 ∧
      x^2/15 + y^2/10 = 1 ∧
      other_ellipse x y)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2543_254389


namespace NUMINAMATH_CALUDE_pond_length_l2543_254316

/-- Given a rectangular field and a square pond, prove the length of the pond --/
theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_area : ℝ) : 
  field_length = 32 →
  field_width = field_length / 2 →
  pond_area = (field_length * field_width) / 8 →
  ∃ (pond_length : ℝ), pond_length^2 = pond_area ∧ pond_length = 8 :=
by sorry

end NUMINAMATH_CALUDE_pond_length_l2543_254316


namespace NUMINAMATH_CALUDE_total_difference_across_age_groups_l2543_254325

/-- Represents the number of children in each category for an age group -/
structure AgeGroup where
  camp : ℕ
  home : ℕ

/-- Calculates the difference between camp and home for an age group -/
def difference (group : AgeGroup) : ℤ :=
  group.camp - group.home

/-- The given data for each age group -/
def group_5_10 : AgeGroup := ⟨245785, 197680⟩
def group_11_15 : AgeGroup := ⟨287279, 253425⟩
def group_16_18 : AgeGroup := ⟨285994, 217173⟩

/-- The theorem to be proved -/
theorem total_difference_across_age_groups :
  difference group_5_10 + difference group_11_15 + difference group_16_18 = 150780 := by
  sorry

end NUMINAMATH_CALUDE_total_difference_across_age_groups_l2543_254325


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2543_254362

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem rationalize_denominator :
  (1 : ℝ) / (cubeRoot 2 + cubeRoot 16) = cubeRoot 4 / 6 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2543_254362


namespace NUMINAMATH_CALUDE_q_necessary_not_sufficient_l2543_254305

-- Define the propositions
def p (b : ℝ) : Prop := ∃ r : ℝ, r ≠ 0 ∧ b = 1 * r ∧ 9 = b * r

def q (b : ℝ) : Prop := b = 3

-- State the theorem
theorem q_necessary_not_sufficient :
  (∀ b : ℝ, p b → q b) ∧ (∃ b : ℝ, p b ∧ ¬q b) := by
  sorry

end NUMINAMATH_CALUDE_q_necessary_not_sufficient_l2543_254305


namespace NUMINAMATH_CALUDE_f_minus_g_equals_one_l2543_254370

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- State the theorem
theorem f_minus_g_equals_one 
  (h_even : is_even f) 
  (h_odd : is_odd g) 
  (h_sum : ∀ x, f x + g x = x^3 + x^2 + 1) : 
  f 1 - g 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_f_minus_g_equals_one_l2543_254370


namespace NUMINAMATH_CALUDE_curve_composition_l2543_254344

-- Define the curve
def curve (x y : ℝ) : Prop := (3*x - y + 1) * (y - Real.sqrt (1 - x^2)) = 0

-- Define a semicircle
def semicircle (x y : ℝ) : Prop := y = Real.sqrt (1 - x^2) ∧ -1 ≤ x ∧ x ≤ 1

-- Define a line segment
def line_segment (x y : ℝ) : Prop := 3*x - y + 1 = 0 ∧ -1 ≤ x ∧ x ≤ 1

-- Theorem statement
theorem curve_composition :
  ∀ x y : ℝ, curve x y ↔ (semicircle x y ∨ line_segment x y) :=
sorry

end NUMINAMATH_CALUDE_curve_composition_l2543_254344


namespace NUMINAMATH_CALUDE_derivative_product_polynomial_l2543_254321

theorem derivative_product_polynomial (x : ℝ) :
  let f : ℝ → ℝ := λ x => (2*x^2 + 3)*(3*x - 1)
  let f' : ℝ → ℝ := λ x => 18*x^2 - 4*x + 9
  HasDerivAt f (f' x) x := by sorry

end NUMINAMATH_CALUDE_derivative_product_polynomial_l2543_254321


namespace NUMINAMATH_CALUDE_pool_filling_trips_l2543_254396

/-- The number of trips required to fill the pool -/
def trips_to_fill_pool (caleb_gallons cynthia_gallons pool_capacity : ℕ) : ℕ :=
  (pool_capacity + caleb_gallons + cynthia_gallons - 1) / (caleb_gallons + cynthia_gallons)

/-- Theorem stating that it takes 7 trips to fill the pool -/
theorem pool_filling_trips :
  trips_to_fill_pool 7 8 105 = 7 := by sorry

end NUMINAMATH_CALUDE_pool_filling_trips_l2543_254396


namespace NUMINAMATH_CALUDE_sport_water_amount_l2543_254393

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (cornSyrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standardRatio : DrinkRatio :=
  { flavoring := 1,
    cornSyrup := 12,
    water := 30 }

/-- The sport formulation ratio -/
def sportRatio : DrinkRatio :=
  { flavoring := standardRatio.flavoring,
    cornSyrup := standardRatio.cornSyrup / 3,
    water := standardRatio.water * 2 }

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sportCornSyrup : ℚ := 6

/-- Theorem: The amount of water in the sport formulation is 90 ounces -/
theorem sport_water_amount : 
  (sportRatio.water / sportRatio.cornSyrup) * sportCornSyrup = 90 := by
  sorry

end NUMINAMATH_CALUDE_sport_water_amount_l2543_254393


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l2543_254365

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a b : Line) (α : Plane) 
  (h1 : a ≠ b)
  (h2 : parallel a α) 
  (h3 : perpendicular b α) : 
  perpendicularLines a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l2543_254365


namespace NUMINAMATH_CALUDE_karens_order_cost_l2543_254345

/-- The cost of Karen's fast-food order -/
def fast_food_order_cost (burger_price sandwich_price smoothie_price : ℕ) 
  (burger_quantity sandwich_quantity smoothie_quantity : ℕ) : ℕ :=
  burger_price * burger_quantity + 
  sandwich_price * sandwich_quantity + 
  smoothie_price * smoothie_quantity

/-- Theorem stating that Karen's fast-food order costs $17 -/
theorem karens_order_cost : 
  fast_food_order_cost 5 4 4 1 1 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_karens_order_cost_l2543_254345


namespace NUMINAMATH_CALUDE_mary_crayons_left_l2543_254369

/-- The number of crayons Mary has left after giving some away and breaking some -/
def crayons_left (initial_green initial_blue initial_yellow : ℚ)
  (given_green given_blue given_yellow broken_yellow : ℚ) : ℚ :=
  (initial_green - given_green) + (initial_blue - given_blue) + (initial_yellow - given_yellow - broken_yellow)

/-- Theorem stating that Mary has 12 crayons left -/
theorem mary_crayons_left :
  crayons_left 5 8 7 3.5 1.25 2.75 0.5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_crayons_left_l2543_254369


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2543_254312

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (7^3 + 9^3 : ℝ) → 
  ∃ (n : ℤ), n = 10 ∧ 
  ∀ (m : ℤ), |x^(1/3) - (n : ℝ)| ≤ |x^(1/3) - (m : ℝ)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2543_254312


namespace NUMINAMATH_CALUDE_ball_probabilities_l2543_254353

def total_balls : ℕ := 6
def white_balls : ℕ := 2
def red_balls : ℕ := 2
def yellow_balls : ℕ := 2

def prob_two_red : ℚ := 1 / 15
def prob_same_color : ℚ := 1 / 5
def prob_one_white : ℚ := 2 / 3

theorem ball_probabilities :
  (total_balls = white_balls + red_balls + yellow_balls) →
  (prob_two_red = (red_balls.choose 2) / (total_balls.choose 2)) ∧
  (prob_same_color = (white_balls.choose 2 + red_balls.choose 2 + yellow_balls.choose 2) / (total_balls.choose 2)) ∧
  (prob_one_white = (white_balls * (total_balls - white_balls)) / (total_balls * (total_balls - 1))) :=
by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2543_254353


namespace NUMINAMATH_CALUDE_least_x_squared_divisible_by_240_l2543_254314

theorem least_x_squared_divisible_by_240 :
  ∀ x : ℕ, x > 0 → x^2 % 240 = 0 → x ≥ 60 :=
by
  sorry

end NUMINAMATH_CALUDE_least_x_squared_divisible_by_240_l2543_254314


namespace NUMINAMATH_CALUDE_min_value_a2_plus_b2_l2543_254324

theorem min_value_a2_plus_b2 (a b : ℝ) (h : (9 : ℝ) / a^2 + (4 : ℝ) / b^2 = 1) :
  ∀ x y : ℝ, (9 : ℝ) / x^2 + (4 : ℝ) / y^2 = 1 → x^2 + y^2 ≥ 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a2_plus_b2_l2543_254324


namespace NUMINAMATH_CALUDE_rachel_apple_tree_l2543_254326

/-- The number of apples on Rachel's tree after picking some and new ones growing. -/
def final_apples (initial : ℕ) (picked : ℕ) (new_grown : ℕ) : ℕ :=
  initial - picked + new_grown

/-- Theorem stating that the final number of apples is correct. -/
theorem rachel_apple_tree (initial : ℕ) (picked : ℕ) (new_grown : ℕ) 
    (h1 : initial = 4) (h2 : picked = 2) (h3 : new_grown = 3) : 
  final_apples initial picked new_grown = 5 := by
  sorry

end NUMINAMATH_CALUDE_rachel_apple_tree_l2543_254326


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2543_254311

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ (697 * n ≡ 1421 * n [ZMOD 36]) ∧ 
  ∀ (m : ℕ), m > 0 → (697 * m ≡ 1421 * m [ZMOD 36]) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2543_254311


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l2543_254394

def B_current_age : ℕ := 34
def A_current_age : ℕ := B_current_age + 4

def A_future_age : ℕ := A_current_age + 10
def B_past_age : ℕ := B_current_age - 10

theorem age_ratio_is_two_to_one :
  A_future_age / B_past_age = 2 ∧ A_future_age % B_past_age = 0 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l2543_254394


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l2543_254301

/-- Given a binomial expansion (2x + 1/√x)^n where n is a positive integer,
    if the ratio of binomial coefficients of the second term to the third term is 2:5,
    then n = 6, the coefficient of x^3 is 240, and the sum of binomial terms is 728 -/
theorem binomial_expansion_properties (n : ℕ+) :
  (Nat.choose n 1 : ℚ) / (Nat.choose n 2 : ℚ) = 2 / 5 →
  (n = 6 ∧
   (Nat.choose 6 2 : ℕ) * 2^4 = 240 ∧
   (2^6 * Nat.choose 6 0 + 2^5 * Nat.choose 6 1 + 2^4 * Nat.choose 6 2 +
    2^3 * Nat.choose 6 3 + 2^2 * Nat.choose 6 4 + 2 * Nat.choose 6 5) = 728) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l2543_254301


namespace NUMINAMATH_CALUDE_angle_of_inclination_slope_one_l2543_254329

/-- The angle of inclination of a line with slope 1 is π/4 --/
theorem angle_of_inclination_slope_one :
  let line : ℝ → ℝ := λ x ↦ x + 1
  let slope : ℝ := 1
  let angle_of_inclination : ℝ := Real.arctan slope
  angle_of_inclination = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_slope_one_l2543_254329


namespace NUMINAMATH_CALUDE_student_arrangement_count_l2543_254381

/-- The number of ways to arrange students from three grades in a row --/
def arrange_students (grade1 : ℕ) (grade2 : ℕ) (grade3 : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial grade2) * (Nat.factorial grade3)

/-- Theorem stating the number of arrangements for the specific case --/
theorem student_arrangement_count :
  arrange_students 1 2 3 = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l2543_254381


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l2543_254388

theorem unique_congruence_in_range : ∃! n : ℕ, 3 ≤ n ∧ n ≤ 8 ∧ n % 8 = 123456 % 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l2543_254388


namespace NUMINAMATH_CALUDE_simplify_expression_l2543_254343

theorem simplify_expression (x : ℝ) : 7*x + 8 - 3*x + 14 = 4*x + 22 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2543_254343


namespace NUMINAMATH_CALUDE_melanie_has_41_balloons_l2543_254387

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The total number of blue balloons Joan and Melanie have together -/
def total_balloons : ℕ := 81

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := total_balloons - joan_balloons

/-- Theorem stating that Melanie has 41 blue balloons -/
theorem melanie_has_41_balloons : melanie_balloons = 41 := by
  sorry

end NUMINAMATH_CALUDE_melanie_has_41_balloons_l2543_254387


namespace NUMINAMATH_CALUDE_de_moivres_formula_l2543_254386

theorem de_moivres_formula (n : ℕ) (φ : ℝ) :
  (Complex.cos φ + Complex.I * Complex.sin φ) ^ n = Complex.cos (n * φ) + Complex.I * Complex.sin (n * φ) := by
  sorry

end NUMINAMATH_CALUDE_de_moivres_formula_l2543_254386


namespace NUMINAMATH_CALUDE_asia_highest_population_l2543_254350

-- Define the structure for continent population data
structure ContinentPopulation where
  name : String
  population1950 : ℝ
  population2000 : ℝ

-- Define Asia's population data
def asia : ContinentPopulation := {
  name := "Asia",
  population1950 := 1.402,
  population2000 := 3.683
}

-- Define a function to check if a continent has the highest population
def hasHighestPopulation (continent : ContinentPopulation) (allContinents : List ContinentPopulation) (year : Nat) : Prop :=
  match year with
  | 1950 => ∀ c ∈ allContinents, continent.population1950 ≥ c.population1950
  | 2000 => ∀ c ∈ allContinents, continent.population2000 ≥ c.population2000
  | _ => False

-- Theorem statement
theorem asia_highest_population (allContinents : List ContinentPopulation) :
  asia ∈ allContinents →
  hasHighestPopulation asia allContinents 1950 ∧ hasHighestPopulation asia allContinents 2000 := by
  sorry

end NUMINAMATH_CALUDE_asia_highest_population_l2543_254350


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l2543_254373

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h : is_increasing f) : f (a^2 + 1) > f a := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l2543_254373


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2543_254398

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_prime_solution :
  ∀ p q r : ℕ,
    is_prime p ∧ is_prime q ∧ is_prime r ∧
    p * (q - r) = q + r →
    p = 5 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l2543_254398


namespace NUMINAMATH_CALUDE_green_beans_count_l2543_254300

def total_beans : ℕ := 572

def red_beans : ℕ := total_beans / 4

def remaining_after_red : ℕ := total_beans - red_beans

def white_beans : ℕ := remaining_after_red / 3

def remaining_after_white : ℕ := remaining_after_red - white_beans

def blue_beans : ℕ := remaining_after_white / 5

def remaining_after_blue : ℕ := remaining_after_white - blue_beans

def yellow_beans : ℕ := remaining_after_blue / 6

def remaining_after_yellow : ℕ := remaining_after_blue - yellow_beans

def green_beans : ℕ := remaining_after_yellow / 2

theorem green_beans_count : green_beans = 95 := by
  sorry

end NUMINAMATH_CALUDE_green_beans_count_l2543_254300


namespace NUMINAMATH_CALUDE_greatest_valid_integer_l2543_254349

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 24 = 4

theorem greatest_valid_integer : 
  (∀ m, is_valid m → m ≤ 140) ∧ is_valid 140 :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_integer_l2543_254349


namespace NUMINAMATH_CALUDE_bottom_right_not_divisible_by_2011_l2543_254357

/-- Represents a cell on the board -/
structure Cell where
  x : Nat
  y : Nat

/-- Represents the board configuration -/
structure Board where
  size : Nat
  markedCells : List Cell

/-- Counts the number of paths from (0,0) to (x,y) that don't pass through marked cells -/
def countPaths (board : Board) (x y : Nat) : Nat :=
  sorry

theorem bottom_right_not_divisible_by_2011 (board : Board) :
  board.size = 2012 →
  (∀ c ∈ board.markedCells, c.x = c.y ∧ c.x ≠ 0 ∧ c.x ≠ 2011) →
  ¬ (countPaths board 2011 2011 % 2011 = 0) :=
by sorry

end NUMINAMATH_CALUDE_bottom_right_not_divisible_by_2011_l2543_254357


namespace NUMINAMATH_CALUDE_watermelon_stand_problem_l2543_254323

/-- A watermelon stand problem -/
theorem watermelon_stand_problem (total_melons : ℕ) 
  (single_melon_customers : ℕ) (triple_melon_customers : ℕ) :
  total_melons = 46 →
  single_melon_customers = 17 →
  triple_melon_customers = 3 →
  total_melons - (single_melon_customers * 1 + triple_melon_customers * 3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_stand_problem_l2543_254323


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2543_254351

theorem cubic_equation_roots (p : ℝ) : 
  (p = 6 ∨ p = -6) → 
  ∃ x y : ℝ, x ≠ y ∧ y - x = 1 ∧ 
  x^3 - 7*x + p = 0 ∧ 
  y^3 - 7*y + p = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2543_254351


namespace NUMINAMATH_CALUDE_equal_integers_in_table_l2543_254382

theorem equal_integers_in_table (t : Fin 10 → Fin 10 → ℤ) 
  (h : ∀ i j i' j', (i = i' ∧ |j - j'| = 1) ∨ (j = j' ∧ |i - i'| = 1) → |t i j - t i' j'| ≤ 5) :
  ∃ i j i' j', (i, j) ≠ (i', j') ∧ t i j = t i' j' :=
sorry

end NUMINAMATH_CALUDE_equal_integers_in_table_l2543_254382


namespace NUMINAMATH_CALUDE_slope_range_l2543_254380

theorem slope_range (α : Real) (h : π/3 < α ∧ α < 5*π/6) :
  ∃ k : Real, (k < -Real.sqrt 3 / 3 ∨ k > Real.sqrt 3) ∧ k = Real.tan α :=
by
  sorry

end NUMINAMATH_CALUDE_slope_range_l2543_254380


namespace NUMINAMATH_CALUDE_green_peaches_count_l2543_254376

/-- Given a basket of peaches, prove that the number of green peaches is 3 -/
theorem green_peaches_count (total : ℕ) (red : ℕ) (h1 : total = 16) (h2 : red = 13) :
  total - red = 3 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l2543_254376


namespace NUMINAMATH_CALUDE_quadratic_other_root_l2543_254355

theorem quadratic_other_root 
  (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : a * 2^2 = b) : 
  a * (-2)^2 = b := by
sorry

end NUMINAMATH_CALUDE_quadratic_other_root_l2543_254355


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l2543_254315

/-- The set A defined by a quadratic inequality -/
def A (a₁ b₁ c₁ : ℝ) : Set ℝ := {x | a₁ * x^2 + b₁ * x + c₁ > 0}

/-- The set B defined by a quadratic inequality -/
def B (a₂ b₂ c₂ : ℝ) : Set ℝ := {x | a₂ * x^2 + b₂ * x + c₂ > 0}

/-- The condition for coefficient ratios -/
def ratio_condition (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂

theorem not_sufficient_nor_necessary
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (h₁ : a₁ * b₁ * c₁ ≠ 0) (h₂ : a₂ * b₂ * c₂ ≠ 0) :
  ¬(∀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, ratio_condition a₁ b₁ c₁ a₂ b₂ c₂ → A a₁ b₁ c₁ = B a₂ b₂ c₂) ∧
  ¬(∀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, A a₁ b₁ c₁ = B a₂ b₂ c₂ → ratio_condition a₁ b₁ c₁ a₂ b₂ c₂) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l2543_254315


namespace NUMINAMATH_CALUDE_cow_calf_total_cost_l2543_254368

theorem cow_calf_total_cost (cow_cost calf_cost : ℕ) 
  (h1 : cow_cost = 880)
  (h2 : calf_cost = 110)
  (h3 : cow_cost = 8 * calf_cost) : 
  cow_cost + calf_cost = 990 := by
  sorry

end NUMINAMATH_CALUDE_cow_calf_total_cost_l2543_254368


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2543_254390

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def yIntercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line (b : Line) :
  parallel b { slope := 3, point := (0, -6) } →
  b.point = (1, 2) →
  yIntercept b = -1 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2543_254390


namespace NUMINAMATH_CALUDE_two_correct_conclusions_l2543_254366

-- Define the type for analogical conclusions
inductive AnalogyConclusion
| ComplexRational
| VectorParallel
| PlanePlanar

-- Function to check if a conclusion is correct
def isCorrectConclusion (c : AnalogyConclusion) : Prop :=
  match c with
  | .ComplexRational => True
  | .VectorParallel => False
  | .PlanePlanar => True

-- Theorem statement
theorem two_correct_conclusions :
  (∃ (c1 c2 : AnalogyConclusion), c1 ≠ c2 ∧ 
    isCorrectConclusion c1 ∧ isCorrectConclusion c2 ∧
    (∀ (c3 : AnalogyConclusion), c3 ≠ c1 ∧ c3 ≠ c2 → ¬isCorrectConclusion c3)) :=
by sorry

end NUMINAMATH_CALUDE_two_correct_conclusions_l2543_254366


namespace NUMINAMATH_CALUDE_short_trees_planted_calculation_park_short_trees_planted_l2543_254331

/-- The number of short trees planted in a park -/
def short_trees_planted (initial_short_trees final_short_trees : ℕ) : ℕ :=
  final_short_trees - initial_short_trees

/-- Theorem stating that the number of short trees planted is the difference between the final and initial number of short trees -/
theorem short_trees_planted_calculation (initial_short_trees final_short_trees : ℕ) 
  (h : final_short_trees ≥ initial_short_trees) :
  short_trees_planted initial_short_trees final_short_trees = final_short_trees - initial_short_trees :=
by
  sorry

/-- The specific case for the park problem -/
theorem park_short_trees_planted :
  short_trees_planted 41 98 = 57 :=
by
  sorry

end NUMINAMATH_CALUDE_short_trees_planted_calculation_park_short_trees_planted_l2543_254331


namespace NUMINAMATH_CALUDE_shopping_mall_profit_l2543_254360

/-- Represents the cost and selling prices of items A and B, and the minimum number of type B items to purchase for a profit exceeding $380 -/
theorem shopping_mall_profit (cost_A cost_B sell_A sell_B : ℚ) (min_B : ℕ) : 
  cost_A = cost_B - 2 →
  80 / cost_A = 100 / cost_B →
  sell_A = 12 →
  sell_B = 15 →
  cost_A = 8 →
  cost_B = 10 →
  (∀ y : ℕ, y ≥ min_B → 
    (sell_A - cost_A) * (3 * y - 5 : ℚ) + (sell_B - cost_B) * y > 380) →
  min_B = 24 :=
by sorry

end NUMINAMATH_CALUDE_shopping_mall_profit_l2543_254360


namespace NUMINAMATH_CALUDE_exists_finite_harmonic_progression_no_infinite_harmonic_progression_l2543_254337

/-- A sequence of positive integers is in harmonic progression if their reciprocals form an arithmetic progression. -/
def IsHarmonicProgression (s : ℕ → ℕ) : Prop :=
  ∃ d : ℚ, ∀ i j : ℕ, (1 : ℚ) / s i - (1 : ℚ) / s j = d * (i - j)

/-- For any natural number N, there exists a strictly increasing sequence of N positive integers in harmonic progression. -/
theorem exists_finite_harmonic_progression (N : ℕ) :
    ∃ (s : ℕ → ℕ), (∀ i < N, s i < s (i + 1)) ∧ IsHarmonicProgression s :=
  sorry

/-- There does not exist a strictly increasing infinite sequence of positive integers in harmonic progression. -/
theorem no_infinite_harmonic_progression :
    ¬∃ (s : ℕ → ℕ), (∀ i : ℕ, s i < s (i + 1)) ∧ IsHarmonicProgression s :=
  sorry

end NUMINAMATH_CALUDE_exists_finite_harmonic_progression_no_infinite_harmonic_progression_l2543_254337


namespace NUMINAMATH_CALUDE_solve_equation_l2543_254342

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 2 / 3 → x = -27 / 23 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2543_254342


namespace NUMINAMATH_CALUDE_circle_angle_problem_l2543_254313

theorem circle_angle_problem (x y : ℝ) : 
  y = 2 * x → 7 * x + 6 * x + 3 * x + (2 * x + y) = 360 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_angle_problem_l2543_254313


namespace NUMINAMATH_CALUDE_max_automobile_weight_l2543_254392

/-- Represents the capacity of the ferry in tons -/
def ferry_capacity : ℝ := 50

/-- Represents the maximum number of automobiles the ferry can carry -/
def max_automobiles : ℝ := 62.5

/-- Represents the conversion factor from tons to pounds -/
def tons_to_pounds : ℝ := 2000

/-- Theorem stating that the maximum weight of an automobile is 1600 pounds -/
theorem max_automobile_weight :
  (ferry_capacity * tons_to_pounds) / max_automobiles = 1600 := by
  sorry

end NUMINAMATH_CALUDE_max_automobile_weight_l2543_254392


namespace NUMINAMATH_CALUDE_exists_odd_64digit_no_zeros_div_101_l2543_254363

/-- A 64-digit natural number -/
def Digit64 : Type := { n : ℕ // n ≥ 10^63 ∧ n < 10^64 }

/-- Predicate for numbers not containing zeros -/
def NoZeros (n : ℕ) : Prop := ∀ d : ℕ, d < 64 → (n / 10^d) % 10 ≠ 0

/-- Theorem stating the existence of an odd 64-digit number without zeros that is divisible by 101 -/
theorem exists_odd_64digit_no_zeros_div_101 :
  ∃ (n : Digit64), NoZeros n.val ∧ n.val % 101 = 0 ∧ n.val % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_exists_odd_64digit_no_zeros_div_101_l2543_254363


namespace NUMINAMATH_CALUDE_eventually_linear_closed_under_addition_l2543_254330

theorem eventually_linear_closed_under_addition (S : Set ℕ) 
  (h_closed : ∀ a b : ℕ, a ∈ S → b ∈ S → (a + b) ∈ S) :
  ∃ k N : ℕ, ∀ n : ℕ, n > N → (n ∈ S ↔ k ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_eventually_linear_closed_under_addition_l2543_254330


namespace NUMINAMATH_CALUDE_simplest_proper_fraction_with_7_numerator_simplest_improper_fraction_with_7_denominator_l2543_254333

-- Define a function to check if a fraction is in its simplest form
def isSimplestForm (n d : ℕ) : Prop :=
  n.gcd d = 1

-- Define a function to check if a fraction is proper
def isProper (n d : ℕ) : Prop :=
  n < d

-- Define a function to check if a fraction is improper
def isImproper (n d : ℕ) : Prop :=
  n ≥ d

-- Theorem for the simplest proper fraction with 7 as numerator
theorem simplest_proper_fraction_with_7_numerator :
  isSimplestForm 7 8 ∧ isProper 7 8 ∧
  ∀ d : ℕ, d > 7 → isSimplestForm 7 d → d ≥ 8 :=
sorry

-- Theorem for the simplest improper fraction with 7 as denominator
theorem simplest_improper_fraction_with_7_denominator :
  isSimplestForm 8 7 ∧ isImproper 8 7 ∧
  ∀ n : ℕ, n > 7 → isSimplestForm n 7 → n ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_simplest_proper_fraction_with_7_numerator_simplest_improper_fraction_with_7_denominator_l2543_254333


namespace NUMINAMATH_CALUDE_apartment_groceries_cost_l2543_254306

/-- Proves the cost of groceries for three roommates given their expenses -/
theorem apartment_groceries_cost 
  (rent : ℕ) 
  (utilities : ℕ) 
  (internet : ℕ) 
  (cleaning_supplies : ℕ) 
  (one_roommate_total : ℕ) 
  (h1 : rent = 1100)
  (h2 : utilities = 114)
  (h3 : internet = 60)
  (h4 : cleaning_supplies = 40)
  (h5 : one_roommate_total = 924) :
  (one_roommate_total - (rent + utilities + internet + cleaning_supplies) / 3) * 3 = 1458 :=
by sorry

end NUMINAMATH_CALUDE_apartment_groceries_cost_l2543_254306


namespace NUMINAMATH_CALUDE_regression_properties_l2543_254328

/-- Regression line equation -/
def regression_line (x : ℝ) : ℝ := 6 * x + 8

/-- Data points -/
def data_points : List (ℝ × ℝ) := [(2, 19), (3, 25), (4, 0), (5, 38), (6, 44)]

/-- The value of the unclear data point -/
def unclear_data : ℝ := 34

/-- Theorem stating the properties of the regression line and data points -/
theorem regression_properties :
  let third_point := (4, unclear_data)
  let residual := (third_point.2 - regression_line third_point.1)
  (unclear_data = 34) ∧
  (residual = 2) ∧
  (regression_line 7 = 50) := by sorry

end NUMINAMATH_CALUDE_regression_properties_l2543_254328


namespace NUMINAMATH_CALUDE_polygon_deformation_to_triangle_l2543_254318

/-- A planar polygon represented by its vertices -/
structure PlanarPolygon where
  vertices : List (ℝ × ℝ)
  is_planar : sorry
  is_closed : sorry

/-- A function that checks if a polygon can be deformed into a triangle -/
def can_deform_to_triangle (p : PlanarPolygon) : Prop :=
  sorry

/-- The main theorem stating that any planar polygon with more than 4 sides
    can be deformed into a triangle -/
theorem polygon_deformation_to_triangle 
  (p : PlanarPolygon) (h : p.vertices.length > 4) :
  can_deform_to_triangle p :=
sorry

end NUMINAMATH_CALUDE_polygon_deformation_to_triangle_l2543_254318


namespace NUMINAMATH_CALUDE_freeway_to_traffic_ratio_l2543_254385

def total_time : ℝ := 10
def traffic_time : ℝ := 2

theorem freeway_to_traffic_ratio :
  (total_time - traffic_time) / traffic_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_freeway_to_traffic_ratio_l2543_254385


namespace NUMINAMATH_CALUDE_original_group_size_l2543_254348

/-- Given a group of men working on a project, this theorem proves that the original number of men is 30, based on the given conditions. -/
theorem original_group_size (initial_days work_days : ℕ) (absent_men : ℕ) : 
  initial_days = 10 → 
  work_days = 12 → 
  absent_men = 5 → 
  ∃ (original_size : ℕ), 
    original_size * initial_days = (original_size - absent_men) * work_days ∧ 
    original_size = 30 := by
  sorry

end NUMINAMATH_CALUDE_original_group_size_l2543_254348


namespace NUMINAMATH_CALUDE_rosas_initial_flowers_l2543_254340

/-- The problem of finding Rosa's initial number of flowers -/
theorem rosas_initial_flowers :
  ∀ (initial_flowers additional_flowers total_flowers : ℕ),
    additional_flowers = 23 →
    total_flowers = 90 →
    total_flowers = initial_flowers + additional_flowers →
    initial_flowers = 67 := by
  sorry

end NUMINAMATH_CALUDE_rosas_initial_flowers_l2543_254340


namespace NUMINAMATH_CALUDE_quadrilateral_area_not_integer_l2543_254371

theorem quadrilateral_area_not_integer (n : ℕ) : 
  ¬ (∃ (m : ℕ), m^2 = n * (n + 1) * (n + 2) * (n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_not_integer_l2543_254371


namespace NUMINAMATH_CALUDE_sara_golf_balls_l2543_254354

def dozen : ℕ := 12

theorem sara_golf_balls (total_balls : ℕ) (h : total_balls = 192) :
  total_balls / dozen = 16 := by
  sorry

end NUMINAMATH_CALUDE_sara_golf_balls_l2543_254354


namespace NUMINAMATH_CALUDE_triangle_area_calculation_l2543_254341

theorem triangle_area_calculation (base : ℝ) (height_factor : ℝ) :
  base = 3.6 →
  height_factor = 2.5 →
  (base * (height_factor * base)) / 2 = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_calculation_l2543_254341


namespace NUMINAMATH_CALUDE_exponent_calculation_l2543_254379

theorem exponent_calculation (a : ℝ) : a^3 * (a^3)^2 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l2543_254379


namespace NUMINAMATH_CALUDE_tank_length_calculation_l2543_254358

/-- Calculates the length of a tank given its dimensions and plastering costs. -/
theorem tank_length_calculation (width depth cost_per_sqm total_cost : ℝ) 
  (h_width : width = 12)
  (h_depth : depth = 6)
  (h_cost_per_sqm : cost_per_sqm = 0.75)
  (h_total_cost : total_cost = 558) :
  ∃ length : ℝ, length = 25 ∧ 
  total_cost = (2 * (length * depth) + 2 * (width * depth) + (length * width)) * cost_per_sqm :=
by sorry

end NUMINAMATH_CALUDE_tank_length_calculation_l2543_254358


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l2543_254352

open Set Real

def M : Set ℝ := {x | ∃ y, y = log (x - 1)}
def N : Set ℝ := {y | ∃ x, y = x^2 + 1}

theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l2543_254352


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2543_254399

def U : Set Nat := {0, 1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {3, 4, 5, 6}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2543_254399


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2543_254317

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem set_intersection_theorem :
  (Set.univ \ A) ∩ B = {x | 2 < x ∧ x ≤ 3} :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2543_254317


namespace NUMINAMATH_CALUDE_number_problem_l2543_254319

theorem number_problem (x : ℚ) : x - (3/5) * x = 56 → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2543_254319


namespace NUMINAMATH_CALUDE_integer_sum_problem_l2543_254378

theorem integer_sum_problem (x y : ℕ) (h1 : x > y) (h2 : x - y = 16) (h3 : x * y = 63) :
  x + y = 2 * Real.sqrt 127 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l2543_254378


namespace NUMINAMATH_CALUDE_max_value_expression_l2543_254303

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 2) :
  a * b * Real.sqrt 3 + 3 * b * c ≤ 2 ∧ ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b^2 + c^2 = 2 ∧ a * b * Real.sqrt 3 + 3 * b * c = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2543_254303


namespace NUMINAMATH_CALUDE_cube_sum_l2543_254375

/-- The number of faces in a cube -/
def cube_faces : ℕ := 6

/-- The number of edges in a cube -/
def cube_edges : ℕ := 12

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The sum of faces, edges, and vertices of a cube is 26 -/
theorem cube_sum : cube_faces + cube_edges + cube_vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_l2543_254375


namespace NUMINAMATH_CALUDE_patricks_age_l2543_254308

/-- Given that Patrick is half the age of his elder brother Robert, and Robert will turn 30 after 2 years, prove that Patrick's current age is 14 years. -/
theorem patricks_age (robert_age_in_two_years : ℕ) (robert_current_age : ℕ) (patrick_age : ℕ) : 
  robert_age_in_two_years = 30 → 
  robert_current_age = robert_age_in_two_years - 2 →
  patrick_age = robert_current_age / 2 →
  patrick_age = 14 := by
sorry

end NUMINAMATH_CALUDE_patricks_age_l2543_254308


namespace NUMINAMATH_CALUDE_calculation_proof_l2543_254309

theorem calculation_proof :
  (∃ x : ℝ, x ^ 2 = 2 ∧
    (Real.sqrt 6 * Real.sqrt (1/3) - Real.sqrt 16 * Real.sqrt 18 = -11 * x) ∧
    ((2 - Real.sqrt 5) * (2 + Real.sqrt 5) + (2 - Real.sqrt 2) ^ 2 = 5 - 4 * x)) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2543_254309


namespace NUMINAMATH_CALUDE_assignments_for_40_points_l2543_254332

/-- Calculates the number of assignments needed for a given number of points and assignments per point -/
def assignmentsForPoints (points : ℕ) (assignmentsPerPoint : ℕ) : ℕ :=
  points * assignmentsPerPoint

/-- Calculates the total number of assignments needed for 40 homework points -/
def totalAssignments : ℕ :=
  assignmentsForPoints 7 3 +  -- First 7 points
  assignmentsForPoints 7 4 +  -- Next 7 points (8-14)
  assignmentsForPoints 7 5 +  -- Next 7 points (15-21)
  assignmentsForPoints 7 6 +  -- Next 7 points (22-28)
  assignmentsForPoints 7 7 +  -- Next 7 points (29-35)
  assignmentsForPoints 5 8    -- Last 5 points (36-40)

/-- The theorem stating that 215 assignments are needed for 40 homework points -/
theorem assignments_for_40_points : totalAssignments = 215 := by
  sorry


end NUMINAMATH_CALUDE_assignments_for_40_points_l2543_254332


namespace NUMINAMATH_CALUDE_sum_product_inequality_l2543_254364

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l2543_254364


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2543_254374

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating that in an arithmetic sequence {a_n} where a_1 + a_9 = 10, a_5 = 5 -/
theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 9 = 10) :
  a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2543_254374


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_sum_90_l2543_254334

theorem largest_of_five_consecutive_sum_90 :
  ∀ n : ℕ, (n + (n+1) + (n+2) + (n+3) + (n+4) = 90) → (n+4 = 20) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_sum_90_l2543_254334


namespace NUMINAMATH_CALUDE_probability_of_drawing_balls_l2543_254320

def total_balls : ℕ := 15
def black_balls : ℕ := 10
def white_balls : ℕ := 5
def drawn_balls : ℕ := 5
def drawn_black : ℕ := 3
def drawn_white : ℕ := 2

theorem probability_of_drawing_balls : 
  (Nat.choose black_balls drawn_black * Nat.choose white_balls drawn_white) / 
  Nat.choose total_balls drawn_balls = 400 / 1001 := by
sorry

end NUMINAMATH_CALUDE_probability_of_drawing_balls_l2543_254320


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2543_254304

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((2 + Complex.I) / (1 - 2 * Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2543_254304


namespace NUMINAMATH_CALUDE_domain_of_f_l2543_254359

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x^2 - 1)
def domain_f_squared (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | -Real.sqrt 3 ≤ x ∧ x ≤ Real.sqrt 3}

-- Theorem statement
theorem domain_of_f (f : ℝ → ℝ) :
  (∀ x, x ∈ domain_f_squared f → f (x^2 - 1) ≠ 0) →
  (∀ y, f y ≠ 0 → -1 ≤ y ∧ y ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l2543_254359


namespace NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l2543_254322

theorem cos_squared_alpha_minus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 1/3) :
  Real.cos (α - π/4)^2 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l2543_254322


namespace NUMINAMATH_CALUDE_five_students_three_locations_l2543_254395

/-- The number of ways for a given number of students to choose from a given number of locations. -/
def number_of_choices (num_students : ℕ) (num_locations : ℕ) : ℕ :=
  num_locations ^ num_students

/-- Theorem: The number of ways for 5 students to choose from 3 locations is equal to 243. -/
theorem five_students_three_locations :
  number_of_choices 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_students_three_locations_l2543_254395


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l2543_254346

theorem polynomial_functional_equation (p : ℝ → ℝ) (c : ℝ) :
  (∀ x, p (p x) = x * p x + c * x^2) →
  ((p = id ∧ c = 0) ∨ (∀ x, p x = -x ∧ c = -2)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l2543_254346


namespace NUMINAMATH_CALUDE_mechanic_job_hours_l2543_254327

theorem mechanic_job_hours (hourly_rate parts_cost total_bill : ℕ) : 
  hourly_rate = 45 → parts_cost = 225 → total_bill = 450 → 
  ∃ hours : ℕ, hours * hourly_rate + parts_cost = total_bill ∧ hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_job_hours_l2543_254327


namespace NUMINAMATH_CALUDE_total_rulers_problem_solution_l2543_254356

/-- Given an initial number of rulers and a number of rulers added, 
    the total number of rulers is equal to the sum of the initial number and the added number. -/
theorem total_rulers (initial_rulers added_rulers : ℕ) :
  initial_rulers + added_rulers = initial_rulers + added_rulers := by sorry

/-- The specific case mentioned in the problem -/
theorem problem_solution :
  let initial_rulers : ℕ := 11
  let added_rulers : ℕ := 14
  initial_rulers + added_rulers = 25 := by sorry

end NUMINAMATH_CALUDE_total_rulers_problem_solution_l2543_254356


namespace NUMINAMATH_CALUDE_coffee_shop_revenue_l2543_254391

theorem coffee_shop_revenue : 
  let coffee_orders : ℕ := 7
  let tea_orders : ℕ := 8
  let coffee_price : ℕ := 5
  let tea_price : ℕ := 4
  let total_revenue := coffee_orders * coffee_price + tea_orders * tea_price
  total_revenue = 67 := by sorry

end NUMINAMATH_CALUDE_coffee_shop_revenue_l2543_254391


namespace NUMINAMATH_CALUDE_final_balance_is_450_l2543_254336

/-- Calculates the final balance after withdrawal and deposit --/
def finalBalance (initialBalance : ℚ) : ℚ :=
  let remainingBalance := initialBalance - 200
  let depositAmount := remainingBalance / 2
  remainingBalance + depositAmount

/-- Theorem: The final balance is $450 given the conditions --/
theorem final_balance_is_450 :
  ∃ (initialBalance : ℚ),
    (initialBalance - 200 = initialBalance * (3/5)) ∧
    (finalBalance initialBalance = 450) :=
by sorry

end NUMINAMATH_CALUDE_final_balance_is_450_l2543_254336


namespace NUMINAMATH_CALUDE_solution_comparison_l2543_254397

theorem solution_comparison (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0) :
  (-q' / p > -q / p') ↔ (q' / p < q / p') :=
by sorry

end NUMINAMATH_CALUDE_solution_comparison_l2543_254397


namespace NUMINAMATH_CALUDE_factorization_proof_l2543_254338

theorem factorization_proof (a b : ℝ) : a * b^2 - 9 * a = a * (b + 3) * (b - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2543_254338


namespace NUMINAMATH_CALUDE_statement_A_necessary_not_sufficient_l2543_254383

theorem statement_A_necessary_not_sufficient :
  (∀ x y : ℝ, (x + y ≠ 5 → (x ≠ 2 ∨ y ≠ 3))) ∧
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 3) ∧ (x + y = 5)) := by
  sorry

end NUMINAMATH_CALUDE_statement_A_necessary_not_sufficient_l2543_254383


namespace NUMINAMATH_CALUDE_jony_stops_at_70_l2543_254377

/-- Represents the walking scenario of Jony along Sunrise Boulevard -/
structure WalkingScenario where
  start_block : ℕ
  turn_block : ℕ
  block_length : ℕ
  walking_speed : ℕ
  walking_time : ℕ

/-- Calculates the block where Jony stops walking -/
def stop_block (scenario : WalkingScenario) : ℕ :=
  sorry

/-- Theorem stating that Jony stops at block 70 given the specific scenario -/
theorem jony_stops_at_70 : 
  let scenario : WalkingScenario := {
    start_block := 10,
    turn_block := 90,
    block_length := 40,
    walking_speed := 100,
    walking_time := 40
  }
  stop_block scenario = 70 := by
  sorry

end NUMINAMATH_CALUDE_jony_stops_at_70_l2543_254377


namespace NUMINAMATH_CALUDE_enclosed_area_equals_eight_thirds_l2543_254339

-- Define the functions f and g
def f (x : ℝ) : ℝ := -2 * x^2 + 7 * x - 6
def g (x : ℝ) : ℝ := -x

-- Define the theorem
theorem enclosed_area_equals_eight_thirds :
  ∃ (a b : ℝ), a < b ∧
  (∫ x in a..b, f x - g x) = 8/3 :=
sorry

end NUMINAMATH_CALUDE_enclosed_area_equals_eight_thirds_l2543_254339


namespace NUMINAMATH_CALUDE_cookies_in_bag_l2543_254367

/-- The number of cookies that can fit in one paper bag given a total number of cookies and bags -/
def cookies_per_bag (total_cookies : ℕ) (total_bags : ℕ) : ℕ :=
  (total_cookies / total_bags : ℕ)

/-- Theorem stating that given 292 cookies and 19 paper bags, one bag can hold at most 15 cookies -/
theorem cookies_in_bag : cookies_per_bag 292 19 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_bag_l2543_254367
