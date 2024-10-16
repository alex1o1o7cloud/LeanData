import Mathlib

namespace NUMINAMATH_CALUDE_broken_line_circle_cover_l2660_266048

/-- A closed broken line in a metric space -/
structure ClosedBrokenLine (X : Type*) [MetricSpace X] where
  points : Set X
  is_closed : IsClosed points
  perimeter : ℝ

/-- Theorem: Any closed broken line with perimeter 1 can be covered by a circle of radius 1/4 -/
theorem broken_line_circle_cover {X : Type*} [MetricSpace X] (L : ClosedBrokenLine X) 
  (h_perimeter : L.perimeter = 1) :
  ∃ (center : X), ∀ (p : X), p ∈ L.points → dist center p ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_broken_line_circle_cover_l2660_266048


namespace NUMINAMATH_CALUDE_rug_overlap_problem_l2660_266004

/-- Given three rugs with specific overlapping conditions, prove the area covered by exactly two layers -/
theorem rug_overlap_problem (total_area : ℝ) (covered_area : ℝ) (triple_layer_area : ℝ) 
  (h1 : total_area = 204)
  (h2 : covered_area = 140)
  (h3 : triple_layer_area = 20) :
  total_area - 2 * triple_layer_area - covered_area = 24 := by
  sorry

end NUMINAMATH_CALUDE_rug_overlap_problem_l2660_266004


namespace NUMINAMATH_CALUDE_walters_coins_theorem_l2660_266090

/-- The value of a penny in cents -/
def penny : ℕ := 1

/-- The value of a nickel in cents -/
def nickel : ℕ := 5

/-- The value of a dime in cents -/
def dime : ℕ := 10

/-- The value of a quarter in cents -/
def quarter : ℕ := 25

/-- The number of cents in a dollar -/
def cents_in_dollar : ℕ := 100

/-- The percentage of a dollar represented by Walter's coins -/
def walters_coins_percentage : ℚ := (penny + nickel + dime + quarter : ℚ) / cents_in_dollar * 100

theorem walters_coins_theorem : walters_coins_percentage = 41 := by
  sorry

end NUMINAMATH_CALUDE_walters_coins_theorem_l2660_266090


namespace NUMINAMATH_CALUDE_product_and_square_calculation_l2660_266039

theorem product_and_square_calculation :
  (99 * 101 = 9999) ∧ (98^2 = 9604) := by
  sorry

end NUMINAMATH_CALUDE_product_and_square_calculation_l2660_266039


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2660_266069

theorem nested_fraction_equality : 
  (1 / (1 + 1 / (2 + 1 / 5))) = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2660_266069


namespace NUMINAMATH_CALUDE_max_xy_constraint_l2660_266034

theorem max_xy_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_constraint : 4 * x + 9 * y = 6) :
  x * y ≤ 1 / 4 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 4 * x₀ + 9 * y₀ = 6 ∧ x₀ * y₀ = 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_max_xy_constraint_l2660_266034


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2660_266037

theorem fractional_equation_solution :
  ∃ x : ℚ, (3 / x = 1 / (x - 1)) ∧ (x = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2660_266037


namespace NUMINAMATH_CALUDE_orange_pear_weight_equivalence_l2660_266036

/-- Given that 7 oranges weigh the same as 5 pears, 
    prove that 49 oranges weigh the same as 35 pears. -/
theorem orange_pear_weight_equivalence :
  ∀ (orange_weight pear_weight : ℝ),
  orange_weight > 0 → pear_weight > 0 →
  7 * orange_weight = 5 * pear_weight →
  49 * orange_weight = 35 * pear_weight :=
by
  sorry

#check orange_pear_weight_equivalence

end NUMINAMATH_CALUDE_orange_pear_weight_equivalence_l2660_266036


namespace NUMINAMATH_CALUDE_putnam_inequality_l2660_266092

theorem putnam_inequality (a x : ℝ) (h1 : 0 < x) (h2 : x < a) :
  (a - x)^6 - 3*a*(a - x)^5 + (5/2)*a^2*(a - x)^4 - (1/2)*a^4*(a - x)^2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_putnam_inequality_l2660_266092


namespace NUMINAMATH_CALUDE_division_remainder_l2660_266032

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 686) (h2 : divisor = 36) (h3 : quotient = 19) :
  dividend % divisor = 2 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l2660_266032


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2660_266031

/-- The repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 0.36363636

/-- The fraction 40/99 -/
def fraction : ℚ := 40 / 99

/-- Theorem stating that the repeating decimal 0.363636... equals 40/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2660_266031


namespace NUMINAMATH_CALUDE_two_digit_subtraction_pattern_l2660_266086

theorem two_digit_subtraction_pattern (a b : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) :
  (10 * a + b) - (10 * b + a) = 9 * (a - b) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_subtraction_pattern_l2660_266086


namespace NUMINAMATH_CALUDE_horner_v₂_equals_4_l2660_266023

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 1 + x + x^2 + x^3 + 2x^4 -/
def f : List ℝ := [1, 1, 1, 1, 2]

/-- v₂ in Horner's method for f(x) at x = 1 -/
def v₂ : ℝ :=
  let v₁ := 2 * 1 + 1  -- a₄x + a₃
  v₁ * 1 + 1           -- v₁x + a₂

theorem horner_v₂_equals_4 :
  v₂ = 4 := by sorry

end NUMINAMATH_CALUDE_horner_v₂_equals_4_l2660_266023


namespace NUMINAMATH_CALUDE_parabola_directrix_l2660_266053

/-- The equation of the directrix of the parabola y² = 8x is x = -2 -/
theorem parabola_directrix (x y : ℝ) : 
  (∀ x y, y^2 = 8*x → ∃ p, p = 4 ∧ x = -p/2) → 
  ∃ k, k = -2 ∧ (∀ x y, y^2 = 8*x → x = k) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2660_266053


namespace NUMINAMATH_CALUDE_total_accepted_cartons_is_990_l2660_266071

/-- Represents the number of cartons delivered to a customer -/
def delivered_cartons (customer : Fin 5) : ℕ :=
  if customer.val < 2 then 300 else 200

/-- Represents the number of damaged cartons for a customer -/
def damaged_cartons (customer : Fin 5) : ℕ :=
  match customer.val with
  | 0 => 70
  | 1 => 50
  | 2 => 40
  | 3 => 30
  | 4 => 20
  | _ => 0  -- This case should never occur due to Fin 5

/-- Calculates the number of accepted cartons for a customer -/
def accepted_cartons (customer : Fin 5) : ℕ :=
  delivered_cartons customer - damaged_cartons customer

/-- The main theorem stating that the total number of accepted cartons is 990 -/
theorem total_accepted_cartons_is_990 :
  (Finset.sum Finset.univ accepted_cartons) = 990 := by
  sorry


end NUMINAMATH_CALUDE_total_accepted_cartons_is_990_l2660_266071


namespace NUMINAMATH_CALUDE_basil_cookie_boxes_l2660_266024

/-- The number of cookies Basil gets in the morning and before bed -/
def morning_night_cookies : ℚ := 1/2 + 1/2

/-- The number of whole cookies Basil gets during the day -/
def day_cookies : ℕ := 2

/-- The number of cookies per box -/
def cookies_per_box : ℕ := 45

/-- The number of days Basil needs cookies for -/
def days : ℕ := 30

/-- Theorem stating the number of boxes Basil needs for 30 days -/
theorem basil_cookie_boxes : 
  ⌈(days * (morning_night_cookies + day_cookies)) / cookies_per_box⌉ = 2 := by
  sorry

end NUMINAMATH_CALUDE_basil_cookie_boxes_l2660_266024


namespace NUMINAMATH_CALUDE_cookie_radius_l2660_266042

/-- Given a circle described by the equation x^2 + y^2 + 2x - 4y - 7 = 0, its radius is 2√3 -/
theorem cookie_radius (x y : ℝ) : 
  (x^2 + y^2 + 2*x - 4*y - 7 = 0) → 
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cookie_radius_l2660_266042


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l2660_266017

theorem ceiling_floor_difference (x : ℤ) :
  let y : ℚ := 1/2
  (⌈(x : ℚ) + y⌉ - ⌊(x : ℚ) + y⌋ : ℤ) = 1 ∧ 
  (⌈(x : ℚ) + y⌉ - ((x : ℚ) + y) : ℚ) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l2660_266017


namespace NUMINAMATH_CALUDE_prob_one_of_three_wins_l2660_266056

/-- The probability that one of three mutually exclusive events occurs is the sum of their individual probabilities -/
theorem prob_one_of_three_wins (pX pY pZ : ℚ) 
  (hX : pX = 1/6) (hY : pY = 1/10) (hZ : pZ = 1/8) : 
  pX + pY + pZ = 47/120 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_of_three_wins_l2660_266056


namespace NUMINAMATH_CALUDE_suit_price_increase_l2660_266074

/-- Proves that the percentage increase in the price of a suit is 30% -/
theorem suit_price_increase (original_price : ℝ) (final_price : ℝ) : 
  original_price = 200 →
  final_price = 182 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 30 ∧
    final_price = original_price * (1 + increase_percentage / 100) * 0.7 :=
by sorry

end NUMINAMATH_CALUDE_suit_price_increase_l2660_266074


namespace NUMINAMATH_CALUDE_log_equation_solution_l2660_266065

-- Define the logarithm function for base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log_equation_solution :
  ∃ x : ℝ, x > 0 ∧ log5 x - 4 * log5 2 = -3 ∧ x = 16 / 125 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2660_266065


namespace NUMINAMATH_CALUDE_prob_rain_theorem_l2660_266040

/-- The probability of rain on at least one day during a three-day period -/
def prob_rain_at_least_once (p1 p2 p3 : ℝ) : ℝ :=
  1 - (1 - p1) * (1 - p2) * (1 - p3)

/-- Theorem stating the probability of rain on at least one day is 86% -/
theorem prob_rain_theorem :
  prob_rain_at_least_once 0.3 0.6 0.5 = 0.86 := by
  sorry

#eval prob_rain_at_least_once 0.3 0.6 0.5

end NUMINAMATH_CALUDE_prob_rain_theorem_l2660_266040


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2660_266014

theorem stratified_sample_size 
  (total_population : ℕ) 
  (elderly_population : ℕ) 
  (elderly_sample : ℕ) 
  (n : ℕ) 
  (h1 : total_population = 162) 
  (h2 : elderly_population = 27) 
  (h3 : elderly_sample = 6) 
  (h4 : elderly_population * n = total_population * elderly_sample) : 
  n = 36 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2660_266014


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2660_266046

theorem sqrt_equation_solution :
  let x : ℝ := 49
  Real.sqrt x + Real.sqrt (x + 3) = 12 - Real.sqrt (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2660_266046


namespace NUMINAMATH_CALUDE_min_value_product_l2660_266016

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x / y + y / z + z / x + y / x + z / y + x / z = 10) :
  (x / y + y / z + z / x) * (y / x + z / y + x / z) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l2660_266016


namespace NUMINAMATH_CALUDE_probability_of_specific_combination_l2660_266059

def shirts : ℕ := 6
def shorts : ℕ := 7
def socks : ℕ := 8
def hats : ℕ := 3
def total_items : ℕ := shirts + shorts + socks + hats
def items_chosen : ℕ := 4

theorem probability_of_specific_combination :
  (shirts.choose 1 * shorts.choose 1 * socks.choose 1 * hats.choose 1) / total_items.choose items_chosen = 144 / 1815 :=
sorry

end NUMINAMATH_CALUDE_probability_of_specific_combination_l2660_266059


namespace NUMINAMATH_CALUDE_max_points_at_distance_l2660_266025

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Whether a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop := sorry

/-- The number of points on a circle that are at a given distance from a point -/
def numPointsAtDistance (c : Circle) (p : Point) (d : ℝ) : ℕ := sorry

theorem max_points_at_distance (C : Circle) (P : Point) :
  isOutside P C →
  (∃ (n : ℕ), numPointsAtDistance C P 5 = n ∧ 
    ∀ (m : ℕ), numPointsAtDistance C P 5 ≤ m → n ≤ m) →
  numPointsAtDistance C P 5 = 2 := by sorry

end NUMINAMATH_CALUDE_max_points_at_distance_l2660_266025


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l2660_266079

theorem quadratic_roots_difference (r₁ r₂ : ℝ) : 
  r₁^2 - 7*r₁ + 12 = 0 → r₂^2 - 7*r₂ + 12 = 0 → |r₁ - r₂| = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l2660_266079


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l2660_266070

theorem min_value_of_function (x : ℝ) : 
  (x^2 + 3) / Real.sqrt (x^2 + 2) ≥ 3 * Real.sqrt 2 / 2 :=
by sorry

theorem min_value_achieved : 
  ∃ x : ℝ, (x^2 + 3) / Real.sqrt (x^2 + 2) = 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l2660_266070


namespace NUMINAMATH_CALUDE_sign_sum_theorem_l2660_266027

theorem sign_sum_theorem (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (x : ℤ), x ∈ ({5, 3, 2, 0, -3} : Set ℤ) ∧
  (a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d| = x) := by
  sorry

end NUMINAMATH_CALUDE_sign_sum_theorem_l2660_266027


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2660_266011

theorem complex_equation_solution (z : ℂ) :
  (Complex.I * Real.sqrt 3 + 3 * Complex.I) * z = Complex.I * Real.sqrt 3 →
  z = (Real.sqrt 3 / 4 : ℂ) + (Complex.I / 4) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2660_266011


namespace NUMINAMATH_CALUDE_least_integer_square_condition_l2660_266041

theorem least_integer_square_condition : ∃ x : ℤ, x^2 = 3*x + 12 ∧ ∀ y : ℤ, y^2 = 3*y + 12 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_condition_l2660_266041


namespace NUMINAMATH_CALUDE_movie_watching_time_l2660_266093

/-- The duration of Bret's train ride to Boston -/
def total_duration : ℕ := 9

/-- The time Bret spends reading a book -/
def reading_time : ℕ := 2

/-- The time Bret spends eating dinner -/
def eating_time : ℕ := 1

/-- The time Bret has left for a nap -/
def nap_time : ℕ := 3

/-- Theorem stating that the time spent watching movies is 3 hours -/
theorem movie_watching_time :
  total_duration - (reading_time + eating_time + nap_time) = 3 := by
  sorry

end NUMINAMATH_CALUDE_movie_watching_time_l2660_266093


namespace NUMINAMATH_CALUDE_mix_buyer_probability_l2660_266019

theorem mix_buyer_probability (total : ℕ) (cake muffin cookie : ℕ) 
  (cake_muffin cake_cookie muffin_cookie : ℕ) (all_three : ℕ) 
  (h_total : total = 150)
  (h_cake : cake = 70)
  (h_muffin : muffin = 60)
  (h_cookie : cookie = 50)
  (h_cake_muffin : cake_muffin = 25)
  (h_cake_cookie : cake_cookie = 15)
  (h_muffin_cookie : muffin_cookie = 10)
  (h_all_three : all_three = 5) : 
  (total - (cake + muffin + cookie - cake_muffin - cake_cookie - muffin_cookie + all_three)) / total = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_mix_buyer_probability_l2660_266019


namespace NUMINAMATH_CALUDE_equal_distribution_classroom_l2660_266028

/-- Proves that given 4 classrooms, 56 boys, and 44 girls, with an equal distribution of boys and girls across all classrooms, the total number of students in each classroom is 25. -/
theorem equal_distribution_classroom (num_classrooms : ℕ) (num_boys : ℕ) (num_girls : ℕ) 
  (h1 : num_classrooms = 4)
  (h2 : num_boys = 56)
  (h3 : num_girls = 44)
  (h4 : num_boys % num_classrooms = 0)
  (h5 : num_girls % num_classrooms = 0) :
  (num_boys / num_classrooms) + (num_girls / num_classrooms) = 25 :=
by sorry

end NUMINAMATH_CALUDE_equal_distribution_classroom_l2660_266028


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2660_266006

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (3 + 4 * x) = 7 ∧ x = 11.5) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2660_266006


namespace NUMINAMATH_CALUDE_dennis_rocks_theorem_l2660_266061

/-- Calculates the number of rocks Dennis made the fish spit out -/
def rocks_spit_out (initial_rocks : ℕ) (eaten_rocks : ℕ) (final_rocks : ℕ) : ℕ :=
  final_rocks - (initial_rocks - eaten_rocks)

/-- Proves that Dennis made the fish spit out 2 rocks -/
theorem dennis_rocks_theorem (initial_rocks eaten_rocks final_rocks : ℕ) 
  (h1 : initial_rocks = 10)
  (h2 : eaten_rocks = initial_rocks / 2)
  (h3 : final_rocks = 7) :
  rocks_spit_out initial_rocks eaten_rocks final_rocks = 2 := by
sorry

end NUMINAMATH_CALUDE_dennis_rocks_theorem_l2660_266061


namespace NUMINAMATH_CALUDE_expression_evaluation_l2660_266087

theorem expression_evaluation : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2660_266087


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2019_l2660_266076

theorem last_four_digits_of_5_pow_2019 (h5 : 5^5 % 10000 = 3125)
                                       (h6 : 5^6 % 10000 = 5625)
                                       (h7 : 5^7 % 10000 = 8125)
                                       (h8 : 5^8 % 10000 = 0625) :
  5^2019 % 10000 = 8125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2019_l2660_266076


namespace NUMINAMATH_CALUDE_mike_seashells_l2660_266082

/-- The total number of seashells Mike found -/
def total_seashells (initial : ℝ) (later : ℝ) : ℝ := initial + later

/-- Theorem stating that Mike found 10.75 seashells in total -/
theorem mike_seashells :
  let initial_seashells : ℝ := 6.5
  let later_seashells : ℝ := 4.25
  total_seashells initial_seashells later_seashells = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_mike_seashells_l2660_266082


namespace NUMINAMATH_CALUDE_knitting_productivity_comparison_l2660_266098

/-- Represents a knitter with their working time and break time -/
structure Knitter where
  workTime : ℕ
  breakTime : ℕ

/-- Calculates the total cycle time for a knitter -/
def cycleTime (k : Knitter) : ℕ := k.workTime + k.breakTime

/-- Calculates the number of complete cycles in a given time -/
def completeCycles (k : Knitter) (totalTime : ℕ) : ℕ :=
  totalTime / cycleTime k

/-- Calculates the total working time within a given time period -/
def totalWorkTime (k : Knitter) (totalTime : ℕ) : ℕ :=
  completeCycles k totalTime * k.workTime

theorem knitting_productivity_comparison : 
  let girl1 : Knitter := ⟨5, 1⟩
  let girl2 : Knitter := ⟨7, 1⟩
  let commonBreakTime := lcm (cycleTime girl1) (cycleTime girl2)
  totalWorkTime girl1 commonBreakTime * 21 = totalWorkTime girl2 commonBreakTime * 20 := by
  sorry

end NUMINAMATH_CALUDE_knitting_productivity_comparison_l2660_266098


namespace NUMINAMATH_CALUDE_average_weight_increase_l2660_266099

theorem average_weight_increase (initial_count : ℕ) (initial_weight : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 60 →
  new_weight = 80 →
  (new_weight - old_weight) / initial_count = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2660_266099


namespace NUMINAMATH_CALUDE_inequality_problem_l2660_266054

/-- Given an inequality and its solution set, prove the values of a and b and solve another inequality -/
theorem inequality_problem (a b c : ℝ) : 
  (∀ x, (a * x^2 - 3*x + 6 > 4) ↔ (x < 1 ∨ x > b)) →
  c > 2 →
  (a = 1 ∧ b = 2) ∧
  (∀ x, (a * x^2 - (a*c + b)*x + b*c < 0) ↔ (2 < x ∧ x < c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l2660_266054


namespace NUMINAMATH_CALUDE_third_quiz_score_is_92_l2660_266049

/-- Given the average score of three quizzes and the average score of the first two quizzes,
    calculates the score of the third quiz. -/
def third_quiz_score (avg_three : ℚ) (avg_two : ℚ) : ℚ :=
  3 * avg_three - 2 * avg_two

/-- Theorem stating that given the specific average scores,
    the third quiz score is 92. -/
theorem third_quiz_score_is_92 :
  third_quiz_score 94 95 = 92 := by
  sorry

end NUMINAMATH_CALUDE_third_quiz_score_is_92_l2660_266049


namespace NUMINAMATH_CALUDE_givenPoint_on_y_axis_l2660_266044

/-- A point in the Cartesian coordinate system. -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of a point lying on the y-axis. -/
def OnYAxis (p : CartesianPoint) : Prop :=
  p.x = 0

/-- The given point (0, -1) in the Cartesian coordinate system. -/
def givenPoint : CartesianPoint :=
  ⟨0, -1⟩

/-- Theorem stating that the given point lies on the y-axis. -/
theorem givenPoint_on_y_axis : OnYAxis givenPoint := by
  sorry

end NUMINAMATH_CALUDE_givenPoint_on_y_axis_l2660_266044


namespace NUMINAMATH_CALUDE_power_of_product_l2660_266075

theorem power_of_product (x : ℝ) : (2 * x)^3 = 8 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2660_266075


namespace NUMINAMATH_CALUDE_expression_simplification_l2660_266051

theorem expression_simplification (a b c x y z : ℝ) :
  (c * x * (b * x^2 + 3 * a^2 * y^2 + c^2 * z^2) + b * z * (b * x^2 + 3 * c^2 * x^2 + a^2 * y^2)) / (c * x + b * z) = 
  b * x^2 + a^2 * y^2 + c^2 * z^2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2660_266051


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2660_266095

theorem sum_of_four_numbers (a b c d : ℕ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : a > d)
  (h2 : a * b = c * d)
  (h3 : a + b + c + d = a * c) :
  a + b + c + d = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2660_266095


namespace NUMINAMATH_CALUDE_inequality_proof_l2660_266007

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 3 / 2) :
  x + 4 * y + 9 * z ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2660_266007


namespace NUMINAMATH_CALUDE_absent_workers_fraction_l2660_266009

theorem absent_workers_fraction (p : ℕ) (W : ℝ) (h : p > 0) :
  let work_per_person := W / p
  let absent_fraction : ℝ → ℝ := λ x => x
  let present_workers : ℝ → ℝ := λ x => p * (1 - x)
  let increased_work_per_person := work_per_person * 1.2
  increased_work_per_person = W / (present_workers (absent_fraction (1/6))) :=
by sorry

end NUMINAMATH_CALUDE_absent_workers_fraction_l2660_266009


namespace NUMINAMATH_CALUDE_bmw_sales_l2660_266072

def total_cars : ℕ := 300
def ford_percentage : ℚ := 18 / 100
def nissan_percentage : ℚ := 25 / 100
def chevrolet_percentage : ℚ := 20 / 100

theorem bmw_sales : 
  let other_brands_percentage := ford_percentage + nissan_percentage + chevrolet_percentage
  let bmw_percentage := 1 - other_brands_percentage
  ↑⌊bmw_percentage * total_cars⌋ = 111 := by sorry

end NUMINAMATH_CALUDE_bmw_sales_l2660_266072


namespace NUMINAMATH_CALUDE_square_side_length_equal_area_l2660_266013

theorem square_side_length_equal_area (rectangle_length rectangle_width : ℝ) :
  rectangle_length = 72 ∧ rectangle_width = 18 →
  ∃ (square_side : ℝ), square_side ^ 2 = rectangle_length * rectangle_width ∧ square_side = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_equal_area_l2660_266013


namespace NUMINAMATH_CALUDE_solve_equation_l2660_266060

theorem solve_equation (x : ℝ) (h : x + 1 = 2) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2660_266060


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_with_150_exterior_l2660_266081

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base_angle₁ : ℝ
  base_angle₂ : ℝ
  vertex_angle : ℝ
  is_isosceles : base_angle₁ = base_angle₂
  angle_sum : base_angle₁ + base_angle₂ + vertex_angle = 180

-- Define the exterior angle
def exterior_angle (t : IsoscelesTriangle) : ℝ := 180 - t.vertex_angle

-- Theorem statement
theorem isosceles_triangle_base_angle_with_150_exterior
  (t : IsoscelesTriangle)
  (h : exterior_angle t = 150) :
  t.base_angle₁ = 30 ∨ t.base_angle₁ = 75 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_with_150_exterior_l2660_266081


namespace NUMINAMATH_CALUDE_waiter_tips_sum_l2660_266020

/-- Represents the tips received by a waiter during a lunch shift -/
def WaiterTips : Type := List Float

/-- The number of customers served during the lunch shift -/
def totalCustomers : Nat := 10

/-- The number of customers who left a tip -/
def tippingCustomers : Nat := 5

/-- The list of tips received from the customers who left tips -/
def tipsList : WaiterTips := [1.50, 2.75, 3.25, 4.00, 5.00]

/-- Theorem stating that the sum of tips received by the waiter is $16.50 -/
theorem waiter_tips_sum :
  tipsList.length = tippingCustomers ∧
  totalCustomers = tippingCustomers + (totalCustomers - tippingCustomers) →
  tipsList.sum = 16.50 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_sum_l2660_266020


namespace NUMINAMATH_CALUDE_final_stamp_count_l2660_266022

def parkers_stamps (initial_stamps : ℕ) (addies_stamps : ℕ) : ℕ :=
  initial_stamps + (addies_stamps / 4)

theorem final_stamp_count : parkers_stamps 18 72 = 36 := by
  sorry

end NUMINAMATH_CALUDE_final_stamp_count_l2660_266022


namespace NUMINAMATH_CALUDE_power_of_power_equals_power_product_l2660_266091

theorem power_of_power_equals_power_product (x : ℝ) : (x^2)^4 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_equals_power_product_l2660_266091


namespace NUMINAMATH_CALUDE_code_cracking_probability_l2660_266045

theorem code_cracking_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/5) (h2 : p2 = 1/3) (h3 : p3 = 1/4) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_code_cracking_probability_l2660_266045


namespace NUMINAMATH_CALUDE_remainder_of_3_power_100_plus_5_mod_11_l2660_266097

theorem remainder_of_3_power_100_plus_5_mod_11 : (3^100 + 5) % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_power_100_plus_5_mod_11_l2660_266097


namespace NUMINAMATH_CALUDE_solutions_count_for_specific_n_l2660_266043

/-- Count of integer solutions for x^2 - y^2 = n^2 -/
def count_solutions (n : ℕ) : ℕ :=
  if n = 1992 then 90
  else if n = 1993 then 6
  else if n = 1994 then 6
  else 0

/-- Theorem stating the count of integer solutions for x^2 - y^2 = n^2 for specific n values -/
theorem solutions_count_for_specific_n :
  (count_solutions 1992 = 90) ∧
  (count_solutions 1993 = 6) ∧
  (count_solutions 1994 = 6) :=
by sorry

end NUMINAMATH_CALUDE_solutions_count_for_specific_n_l2660_266043


namespace NUMINAMATH_CALUDE_local_tax_deduction_l2660_266012

/-- Proves that given an hourly wage of 25 dollars and a 2% local tax rate, 
    the amount deducted for local taxes is 50 cents per hour. -/
theorem local_tax_deduction (hourly_wage : ℝ) (tax_rate : ℝ) :
  hourly_wage = 25 ∧ tax_rate = 0.02 →
  (hourly_wage * tax_rate * 100 : ℝ) = 50 := by
  sorry

#check local_tax_deduction

end NUMINAMATH_CALUDE_local_tax_deduction_l2660_266012


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2660_266068

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (a b : V)

-- State the theorem
theorem vector_difference_magnitude 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 1) 
  (h3 : ‖a + b‖ = 1) : 
  ‖a - b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2660_266068


namespace NUMINAMATH_CALUDE_louis_age_proof_l2660_266094

/-- Carla's age in 6 years -/
def carla_future_age : ℕ := 30

/-- Number of years until Carla reaches her future age -/
def years_until_future : ℕ := 6

/-- Sum of Carla's and Louis's current ages -/
def sum_of_ages : ℕ := 55

/-- Louis's current age -/
def louis_age : ℕ := 31

theorem louis_age_proof :
  louis_age = sum_of_ages - (carla_future_age - years_until_future) :=
by sorry

end NUMINAMATH_CALUDE_louis_age_proof_l2660_266094


namespace NUMINAMATH_CALUDE_certain_number_subtraction_l2660_266001

theorem certain_number_subtraction (x : ℤ) : 
  (3005 - x + 10 = 2705) → (x = 310) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_subtraction_l2660_266001


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2660_266047

/-- A line in the form y + 2 = k(x + 1) always passes through the point (-1, -2) -/
theorem line_passes_through_point (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ k * (x + 1) - 2
  f (-1) = -2 := by
  sorry


end NUMINAMATH_CALUDE_line_passes_through_point_l2660_266047


namespace NUMINAMATH_CALUDE_min_visible_sum_l2660_266057

/-- Represents a die in the cube -/
structure Die where
  faces : Fin 6 → Nat
  sum_opposite : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents the large 4x4x4 cube -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Calculates the sum of visible faces on the large cube -/
def visibleSum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the minimum sum of visible faces -/
theorem min_visible_sum (cube : LargeCube) : 
  visibleSum cube ≥ 144 :=
sorry

end NUMINAMATH_CALUDE_min_visible_sum_l2660_266057


namespace NUMINAMATH_CALUDE_fifth_term_is_81_l2660_266073

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n
  sum_1_3 : a 1 + a 3 = 10
  sum_2_4 : a 2 + a 4 = -30

/-- The fifth term of the geometric sequence is 81 -/
theorem fifth_term_is_81 (seq : GeometricSequence) : seq.a 5 = 81 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_81_l2660_266073


namespace NUMINAMATH_CALUDE_quadratic_function_range_l2660_266058

theorem quadratic_function_range (a b : ℝ) : 
  (∀ x ∈ Set.Ioo 2 5, a * x^2 + b * x + 2 > 0) →
  (a * 1^2 + b * 1 + 2 = 1) →
  a > 3 - 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l2660_266058


namespace NUMINAMATH_CALUDE_possible_m_values_l2660_266096

theorem possible_m_values (M N : Set ℝ) (m : ℝ) :
  M = {x : ℝ | 2 * x^2 - 5 * x - 3 = 0} →
  N = {x : ℝ | m * x = 1} →
  N ⊆ M →
  {m | ∃ (x : ℝ), x ∈ N} = {-2, 1/3} :=
by sorry

end NUMINAMATH_CALUDE_possible_m_values_l2660_266096


namespace NUMINAMATH_CALUDE_share_face_value_l2660_266029

theorem share_face_value (dividend_rate : ℝ) (desired_return : ℝ) (market_value : ℝ) :
  dividend_rate = 0.09 →
  desired_return = 0.12 →
  market_value = 36.00000000000001 →
  (desired_return / dividend_rate) * market_value = 48.00000000000001 :=
by
  sorry

end NUMINAMATH_CALUDE_share_face_value_l2660_266029


namespace NUMINAMATH_CALUDE_even_function_property_l2660_266088

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_prop : ∀ x, f (2 + x) = -f (2 - x)) :
  f 2010 = 0 := by sorry

end NUMINAMATH_CALUDE_even_function_property_l2660_266088


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2660_266033

/-- The equation of the line passing through the points of tangency of the tangents drawn from a point to a circle. -/
theorem tangent_line_equation (P : ℝ × ℝ) (c : ℝ × ℝ → Prop) :
  P = (2, 1) →
  (∀ x y, c (x, y) ↔ x^2 + y^2 = 4) →
  ∃ A B : ℝ × ℝ,
    (c A ∧ c B) ∧
    (∀ t : ℝ, c ((1 - t) * A.1 + t * B.1, (1 - t) * A.2 + t * B.2) → t = 0 ∨ t = 1) ∧
    (∀ x y, 2 * x + y - 4 = 0 ↔ ∃ t : ℝ, x = (1 - t) * A.1 + t * B.1 ∧ y = (1 - t) * A.2 + t * B.2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2660_266033


namespace NUMINAMATH_CALUDE_no_valid_box_dimensions_l2660_266015

theorem no_valid_box_dimensions : ¬∃ (b c : ℤ), b ≤ c ∧ 2 * b * c + 2 * (2 * b + 2 * c + b * c) = 120 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_box_dimensions_l2660_266015


namespace NUMINAMATH_CALUDE_comparison_theorem_l2660_266083

theorem comparison_theorem (n : ℕ) (h : n ≥ 2) :
  (2^(2^2) * n < 3^(3^(3^3)) * n - 1) ∧
  (3^(3^(3^3)) * n > 4^(4^(4^4)) * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l2660_266083


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l2660_266030

theorem no_positive_integer_solution :
  ¬∃ (x y z t : ℕ+), x^2 + 2*y^2 = z^2 ∧ 2*x^2 + y^2 = t^2 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l2660_266030


namespace NUMINAMATH_CALUDE_max_value_of_y_l2660_266055

noncomputable section

def angle_alpha : ℝ := Real.arctan (-Real.sqrt 3 / 3)

def point_P : ℝ × ℝ := (-3, Real.sqrt 3)

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

def f (x : ℝ) : ℝ := 
  determinant (Real.cos (x + angle_alpha)) (-Real.sin angle_alpha) (Real.sin (x + angle_alpha)) (Real.cos angle_alpha)

def y (x : ℝ) : ℝ := Real.sqrt 3 * f (Real.pi / 2 - 2 * x) + 2 * f x ^ 2

theorem max_value_of_y :
  ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) ∧ 
  y x = 3 ∧ 
  ∀ (z : ℝ), z ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) → y z ≤ y x :=
sorry

end

end NUMINAMATH_CALUDE_max_value_of_y_l2660_266055


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l2660_266050

theorem unique_solution_for_exponential_equation :
  ∀ n p : ℕ+,
    Nat.Prime p →
    3^(p : ℕ) - n * p = n + p →
    n = 6 ∧ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l2660_266050


namespace NUMINAMATH_CALUDE_largest_number_of_piles_l2660_266038

theorem largest_number_of_piles (apples : Nat) (apricots : Nat) (cherries : Nat)
  (h1 : apples = 42)
  (h2 : apricots = 60)
  (h3 : cherries = 90) :
  Nat.gcd apples (Nat.gcd apricots cherries) = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_of_piles_l2660_266038


namespace NUMINAMATH_CALUDE_BH_length_l2660_266026

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CA := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  AB = 5 ∧ BC = 7 ∧ CA = 8

-- Define points G and H on ray AB
def points_on_ray (A B G H : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, t₁ > 1 ∧ t₂ > t₁ ∧
  G = (A.1 + t₁ * (B.1 - A.1), A.2 + t₁ * (B.2 - A.2)) ∧
  H = (A.1 + t₂ * (B.1 - A.1), A.2 + t₂ * (B.2 - A.2))

-- Define point I on the intersection of circumcircles
def point_on_circumcircles (A B C G H I : ℝ × ℝ) : Prop :=
  I ≠ C ∧
  ∃ r₁ r₂ : ℝ,
    (I.1 - A.1)^2 + (I.2 - A.2)^2 = r₁^2 ∧
    (G.1 - A.1)^2 + (G.2 - A.2)^2 = r₁^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = r₁^2 ∧
    (I.1 - B.1)^2 + (I.2 - B.2)^2 = r₂^2 ∧
    (H.1 - B.1)^2 + (H.2 - B.2)^2 = r₂^2 ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = r₂^2

-- Define distances GI and HI
def distances (G H I : ℝ × ℝ) : Prop :=
  Real.sqrt ((G.1 - I.1)^2 + (G.2 - I.2)^2) = 3 ∧
  Real.sqrt ((H.1 - I.1)^2 + (H.2 - I.2)^2) = 8

-- Main theorem
theorem BH_length (A B C G H I : ℝ × ℝ) :
  triangle_ABC A B C →
  points_on_ray A B G H →
  point_on_circumcircles A B C G H I →
  distances G H I →
  Real.sqrt ((B.1 - H.1)^2 + (B.2 - H.2)^2) = (6 + 47 * Real.sqrt 2) / 9 := by
  sorry

end NUMINAMATH_CALUDE_BH_length_l2660_266026


namespace NUMINAMATH_CALUDE_smallest_sum_of_two_digit_numbers_l2660_266000

def NumberSet : Finset Nat := {5, 6, 7, 8, 9}

def is_valid_pair (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ a ≠ b ∧ a ≥ 10 ∧ a < 100 ∧ b ≥ 10 ∧ b < 100

def sum_of_pair (a b : Nat) : Nat := a + b

theorem smallest_sum_of_two_digit_numbers :
  ∃ (a b : Nat), is_valid_pair a b ∧
    sum_of_pair a b = 125 ∧
    (∀ (c d : Nat), is_valid_pair c d → sum_of_pair c d ≥ 125) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_two_digit_numbers_l2660_266000


namespace NUMINAMATH_CALUDE_ice_cream_cost_l2660_266066

/-- Proves that the cost of a scoop of ice cream is $5 given the problem conditions -/
theorem ice_cream_cost (people : ℕ) (meal_cost : ℕ) (total_money : ℕ) 
  (h1 : people = 3)
  (h2 : meal_cost = 10)
  (h3 : total_money = 45)
  (h4 : ∃ (ice_cream_cost : ℕ), total_money = people * meal_cost + people * ice_cream_cost) :
  ∃ (ice_cream_cost : ℕ), ice_cream_cost = 5 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l2660_266066


namespace NUMINAMATH_CALUDE_circles_A_B_intersect_l2660_266021

/-- Circle A is defined by the equation x^2 + y^2 + 4x + 2y + 1 = 0 -/
def circle_A (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- Circle B is defined by the equation x^2 + y^2 - 2x - 6y + 1 = 0 -/
def circle_B (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 1 = 0

/-- Two circles are intersecting if there exists a point that satisfies both circle equations -/
def circles_intersect (c1 c2 : ℝ → ℝ → Prop) : Prop :=
  ∃ x y : ℝ, c1 x y ∧ c2 x y

/-- Theorem stating that circle A and circle B are intersecting -/
theorem circles_A_B_intersect : circles_intersect circle_A circle_B := by
  sorry

end NUMINAMATH_CALUDE_circles_A_B_intersect_l2660_266021


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l2660_266002

theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → x ≠ 0 → 
  (10 * x + y) - (10 * y + x) = 72 →
  x - y = 8 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l2660_266002


namespace NUMINAMATH_CALUDE_factorial_sum_equals_5040_l2660_266084

theorem factorial_sum_equals_5040 : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_5040_l2660_266084


namespace NUMINAMATH_CALUDE_pet_store_gerbils_l2660_266010

/-- The initial number of gerbils in a pet store -/
def initial_gerbils : ℕ := 68

/-- The number of gerbils sold -/
def sold_gerbils : ℕ := 14

/-- The difference between the initial number and the number sold -/
def difference : ℕ := 54

theorem pet_store_gerbils : 
  initial_gerbils = sold_gerbils + difference := by sorry

end NUMINAMATH_CALUDE_pet_store_gerbils_l2660_266010


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2660_266089

theorem sufficient_not_necessary (a : ℝ) :
  (a = 1/8 → ∀ x : ℝ, x > 0 → 2*x + a/x ≥ 1) ∧
  (∃ a : ℝ, a > 1/8 ∧ ∀ x : ℝ, x > 0 → 2*x + a/x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2660_266089


namespace NUMINAMATH_CALUDE_curve_intersects_axes_l2660_266003

-- Define the parametric equations
def curve_x (t : ℝ) : ℝ := t - 1
def curve_y (t : ℝ) : ℝ := t + 2

-- Define the curve as a set of points
def curve : Set (ℝ × ℝ) := {(x, y) | ∃ t : ℝ, x = curve_x t ∧ y = curve_y t}

-- Define the coordinate axes
def x_axis : Set (ℝ × ℝ) := {(x, y) | y = 0}
def y_axis : Set (ℝ × ℝ) := {(x, y) | x = 0}

theorem curve_intersects_axes :
  (0, 3) ∈ curve ∩ y_axis ∧ (-3, 0) ∈ curve ∩ x_axis :=
sorry

end NUMINAMATH_CALUDE_curve_intersects_axes_l2660_266003


namespace NUMINAMATH_CALUDE_ellipse_properties_l2660_266085

noncomputable def ellipseC (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def pointA : ℝ × ℝ := (0, 1)

def arithmeticSequence (BF1 F1F2 BF2 : ℝ) : Prop :=
  2 * F1F2 = Real.sqrt 3 * (BF1 + BF2)

def lineL (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 + 2)}

def outsideCircle (A P Q : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (Q.1 - A.1) + (P.2 - A.2) * (Q.2 - A.2) > 0

theorem ellipse_properties (a b : ℝ) (h : a > b ∧ b > 0) :
  pointA ∈ ellipseC a b h →
  (∀ B ∈ ellipseC a b h, ∃ F1 F2 : ℝ × ℝ,
    arithmeticSequence (dist B F1) (dist F1 F2) (dist B F2)) →
  (ellipseC a b h = ellipseC 2 1 ⟨by norm_num, by norm_num⟩) ∧
  (∀ k : ℝ, (∀ P Q : ℝ × ℝ, P ∈ ellipseC 2 1 ⟨by norm_num, by norm_num⟩ →
                            Q ∈ ellipseC 2 1 ⟨by norm_num, by norm_num⟩ →
                            P ∈ lineL k → Q ∈ lineL k → P ≠ Q →
                            outsideCircle pointA P Q) ↔
             (k < -3/10 ∨ k > 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2660_266085


namespace NUMINAMATH_CALUDE_even_cube_plus_20_divisible_by_48_l2660_266067

theorem even_cube_plus_20_divisible_by_48 (k : ℤ) : 
  ∃ (n : ℤ), 8 * k * (k^2 + 5) = 48 * n := by
  sorry

end NUMINAMATH_CALUDE_even_cube_plus_20_divisible_by_48_l2660_266067


namespace NUMINAMATH_CALUDE_ellipse_ratio_l2660_266005

/-- An ellipse with equation mx² + ny² = 1, foci on the x-axis, and eccentricity 1/2 -/
structure Ellipse (m n : ℝ) : Prop where
  equation : ∀ x y : ℝ, m * x^2 + n * y^2 = 1
  foci_on_x_axis : True  -- We can't directly represent this geometrically, so we use True as a placeholder
  eccentricity : (1 : ℝ) / 2 = (1 - m / n).sqrt

/-- The ratio m/n for the given ellipse is 3/4 -/
theorem ellipse_ratio (m n : ℝ) (e : Ellipse m n) : m / n = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_ratio_l2660_266005


namespace NUMINAMATH_CALUDE_no_functions_satisfying_conditions_l2660_266052

theorem no_functions_satisfying_conditions :
  ¬ (∃ (f g : ℝ → ℝ), 
    (∀ (x y : ℝ), f (x^2 + g y) - f (x^2) + g y - g x ≤ 2 * y) ∧ 
    (∀ (x : ℝ), f x ≥ x^2)) := by
  sorry

end NUMINAMATH_CALUDE_no_functions_satisfying_conditions_l2660_266052


namespace NUMINAMATH_CALUDE_partitioned_triangle_area_l2660_266078

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first triangle -/
  area1 : ℝ
  /-- Area of the second triangle -/
  area2 : ℝ
  /-- Area of the third triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  area_quad : ℝ

/-- The theorem to be proved -/
theorem partitioned_triangle_area 
  (t : PartitionedTriangle) 
  (h1 : t.area1 = 4) 
  (h2 : t.area2 = 8) 
  (h3 : t.area3 = 12) : 
  t.area_quad = 16 := by
  sorry


end NUMINAMATH_CALUDE_partitioned_triangle_area_l2660_266078


namespace NUMINAMATH_CALUDE_sixth_term_before_three_l2660_266077

def fibonacci_like_sequence (a : ℤ → ℤ) : Prop :=
  ∀ n, a (n + 2) = a (n + 1) + a n

theorem sixth_term_before_three (a : ℤ → ℤ) :
  fibonacci_like_sequence a →
  a 0 = 3 ∧ a 1 = 5 ∧ a 2 = 8 ∧ a 3 = 13 ∧ a 4 = 21 →
  a (-6) = -1 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_before_three_l2660_266077


namespace NUMINAMATH_CALUDE_triangle_side_length_l2660_266064

theorem triangle_side_length (a : ℕ) : 
  (a % 2 = 1) → -- a is odd
  (2 + a > 3) ∧ (2 + 3 > a) ∧ (a + 3 > 2) → -- triangle inequality
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2660_266064


namespace NUMINAMATH_CALUDE_prob_not_six_in_six_rolls_l2660_266063

-- Define a fair six-sided die
def fair_die : Finset ℕ := Finset.range 6

-- Define the probability of an event for a fair die
def prob (event : Finset ℕ) : ℚ :=
  event.card / fair_die.card

-- Define the event of not rolling a 6
def not_six : Finset ℕ := Finset.range 5

-- Theorem statement
theorem prob_not_six_in_six_rolls :
  (prob not_six) ^ 6 = (5 : ℚ) / 6 ^ 6 :=
sorry

end NUMINAMATH_CALUDE_prob_not_six_in_six_rolls_l2660_266063


namespace NUMINAMATH_CALUDE_target_heart_rate_for_sprinting_is_156_l2660_266018

-- Define the athlete's age
def age : ℕ := 30

-- Define the maximum heart rate calculation
def max_heart_rate (a : ℕ) : ℕ := 225 - a

-- Define the target heart rate for jogging
def target_heart_rate_jogging (mhr : ℕ) : ℕ := (mhr * 3) / 4

-- Define the target heart rate for sprinting
def target_heart_rate_sprinting (thr_jogging : ℕ) : ℕ := thr_jogging + 10

-- Theorem to prove
theorem target_heart_rate_for_sprinting_is_156 : 
  target_heart_rate_sprinting (target_heart_rate_jogging (max_heart_rate age)) = 156 := by
  sorry

end NUMINAMATH_CALUDE_target_heart_rate_for_sprinting_is_156_l2660_266018


namespace NUMINAMATH_CALUDE_magnitude_of_BC_l2660_266080

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (1, -2)
def AC : ℝ × ℝ := (4, -1)

theorem magnitude_of_BC : 
  let C : ℝ × ℝ := (A.1 + AC.1, A.2 + AC.2)
  let BC : ℝ × ℝ := (B.1 - C.1, B.2 - C.2)
  Real.sqrt ((BC.1)^2 + (BC.2)^2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_BC_l2660_266080


namespace NUMINAMATH_CALUDE_binomial_cube_seven_l2660_266035

theorem binomial_cube_seven : 7^3 + 3*(7^2) + 3*7 + 1 = 512 := by
  sorry

end NUMINAMATH_CALUDE_binomial_cube_seven_l2660_266035


namespace NUMINAMATH_CALUDE_noah_total_capacity_l2660_266008

-- Define Ali's closet capacity
def ali_closet_capacity : ℕ := 200

-- Define the ratio of Noah's closet capacity to Ali's
def noah_closet_ratio : ℚ := 1 / 4

-- Define the number of Noah's closets
def noah_closet_count : ℕ := 2

-- Theorem statement
theorem noah_total_capacity :
  noah_closet_count * (noah_closet_ratio * ali_closet_capacity) = 100 := by
  sorry


end NUMINAMATH_CALUDE_noah_total_capacity_l2660_266008


namespace NUMINAMATH_CALUDE_number_of_girls_in_school_l2660_266062

/-- Given a school with more girls than boys, calculate the number of girls. -/
theorem number_of_girls_in_school 
  (total_pupils : ℕ) 
  (girl_boy_difference : ℕ) 
  (h1 : total_pupils = 926)
  (h2 : girl_boy_difference = 458) :
  ∃ (girls boys : ℕ), 
    girls = boys + girl_boy_difference ∧ 
    girls + boys = total_pupils ∧ 
    girls = 692 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_school_l2660_266062
