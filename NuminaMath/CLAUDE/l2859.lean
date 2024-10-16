import Mathlib

namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l2859_285962

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π

/-- The theorem stating that if a/cos(B) = b/cos(A) in a triangle, 
    then the triangle is either isosceles or right-angled. -/
theorem triangle_isosceles_or_right_angled (t : Triangle) 
  (h : t.a / Real.cos t.B = t.b / Real.cos t.A) : 
  (t.A = t.B) ∨ (t.C = π/2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l2859_285962


namespace NUMINAMATH_CALUDE_box_volume_l2859_285960

/-- The volume of a rectangular box with face areas 30, 20, and 12 square inches is 60 cubic inches. -/
theorem box_volume (l w h : ℝ) 
  (area1 : l * w = 30)
  (area2 : w * h = 20)
  (area3 : l * h = 12) :
  l * w * h = 60 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l2859_285960


namespace NUMINAMATH_CALUDE_minimum_value_inequality_l2859_285931

theorem minimum_value_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : 2 * m + n = 1) :
  1 / m + 2 / n ≥ 8 ∧ (1 / m + 2 / n = 8 ↔ n = 2 * m ∧ n = 1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_inequality_l2859_285931


namespace NUMINAMATH_CALUDE_hernandez_state_tax_l2859_285947

/-- Calculates the state tax for a resident given their taxable income and months of residence --/
def calculate_state_tax (taxable_income : ℕ) (months_of_residence : ℕ) : ℕ :=
  let adjusted_income := taxable_income - 5000
  let tax_bracket1 := min adjusted_income 10000
  let tax_bracket2 := min (max (adjusted_income - 10000) 0) 20000
  let tax_bracket3 := min (max (adjusted_income - 30000) 0) 30000
  let tax_bracket4 := max (adjusted_income - 60000) 0
  let total_tax := tax_bracket1 / 100 + tax_bracket2 * 3 / 100 + tax_bracket3 * 5 / 100 + tax_bracket4 * 7 / 100
  let tax_credit := if months_of_residence < 10 then 500 else 0
  total_tax - tax_credit

/-- The theorem stating that Mr. Hernandez's state tax is $575 --/
theorem hernandez_state_tax :
  calculate_state_tax 42500 9 = 575 :=
by sorry

end NUMINAMATH_CALUDE_hernandez_state_tax_l2859_285947


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2859_285987

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = 3 ∧ b = 5) ∨ (a = 3 ∧ c = 5) ∨ (b = 3 ∧ c = 5) →
  a^2 + b^2 = c^2 →
  c = Real.sqrt 34 ∨ c = 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2859_285987


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2859_285904

theorem inequality_system_solution (m : ℝ) : 
  (∀ x, (x + 5 < 4*x - 1 ∧ x > m) ↔ x > 2) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2859_285904


namespace NUMINAMATH_CALUDE_initial_tax_rate_proof_l2859_285990

def annual_income : ℝ := 48000
def new_tax_rate : ℝ := 30
def tax_savings : ℝ := 7200

theorem initial_tax_rate_proof :
  ∃ (initial_rate : ℝ),
    initial_rate > 0 ∧
    initial_rate < 100 ∧
    (initial_rate / 100 * annual_income) - (new_tax_rate / 100 * annual_income) = tax_savings ∧
    initial_rate = 45 := by
  sorry

end NUMINAMATH_CALUDE_initial_tax_rate_proof_l2859_285990


namespace NUMINAMATH_CALUDE_parabola_directrix_l2859_285967

/-- Given a parabola with equation y² = 2x, its directrix has the equation x = -1/2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y^2 = 2*x) → (∃ (a : ℝ), a = -1/2 ∧ (∀ (x₀ y₀ : ℝ), y₀^2 = 2*x₀ → x₀ = a)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2859_285967


namespace NUMINAMATH_CALUDE_cube_plus_three_square_plus_three_plus_one_l2859_285927

theorem cube_plus_three_square_plus_three_plus_one : 101^3 + 3*(101^2) + 3*101 + 1 = 1061208 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_three_square_plus_three_plus_one_l2859_285927


namespace NUMINAMATH_CALUDE_power_of_product_cubed_l2859_285966

theorem power_of_product_cubed (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by sorry

end NUMINAMATH_CALUDE_power_of_product_cubed_l2859_285966


namespace NUMINAMATH_CALUDE_min_value_on_interval_l2859_285998

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the theorem
theorem min_value_on_interval (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 6), f x ≥ 9) ∧ (∃ x ∈ Set.Icc a (a + 6), f x = 9) ↔ a = 2 ∨ a = -10 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l2859_285998


namespace NUMINAMATH_CALUDE_bird_percentage_problem_l2859_285930

theorem bird_percentage_problem (total : ℝ) (pigeons sparrows crows parakeets : ℝ) :
  pigeons = 0.4 * total →
  sparrows = 0.2 * total →
  crows = 0.15 * total →
  parakeets = total - (pigeons + sparrows + crows) →
  crows / (total - sparrows) = 0.1875 := by
sorry

end NUMINAMATH_CALUDE_bird_percentage_problem_l2859_285930


namespace NUMINAMATH_CALUDE_car_trip_average_speed_l2859_285982

/-- Calculates the average speed of a car trip given the following conditions:
  * The trip lasts for 15 hours
  * The car averages 30 miles per hour for the first 5 hours
  * The car averages 42 miles per hour for the remaining time
-/
theorem car_trip_average_speed :
  let total_time : ℝ := 15
  let initial_time : ℝ := 5
  let initial_speed : ℝ := 30
  let remaining_speed : ℝ := 42
  let total_distance : ℝ := initial_speed * initial_time + remaining_speed * (total_time - initial_time)
  total_distance / total_time = 38 := by
sorry


end NUMINAMATH_CALUDE_car_trip_average_speed_l2859_285982


namespace NUMINAMATH_CALUDE_women_per_table_l2859_285914

theorem women_per_table (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 7 →
  men_per_table = 2 →
  total_customers = 63 →
  ∃ women_per_table : ℕ, women_per_table * num_tables + men_per_table * num_tables = total_customers ∧ women_per_table = 7 :=
by sorry

end NUMINAMATH_CALUDE_women_per_table_l2859_285914


namespace NUMINAMATH_CALUDE_fourth_number_in_list_l2859_285978

theorem fourth_number_in_list (known_numbers : List ℕ) (average : ℕ) : 
  known_numbers = [55, 48, 507, 684, 42] →
  average = 223 →
  ∃ x : ℕ, (List.sum known_numbers + x) / 6 = average ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_fourth_number_in_list_l2859_285978


namespace NUMINAMATH_CALUDE_flag_design_count_l2859_285992

/-- The number of color choices for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem: The number of possible flag designs is 27 -/
theorem flag_design_count : total_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_design_count_l2859_285992


namespace NUMINAMATH_CALUDE_exercise_time_distribution_l2859_285993

theorem exercise_time_distribution (total_time : ℕ) (aerobics_ratio : ℕ) (weight_ratio : ℕ) 
  (h1 : total_time = 250)
  (h2 : aerobics_ratio = 3)
  (h3 : weight_ratio = 2) :
  ∃ (aerobics_time weight_time : ℕ),
    aerobics_time + weight_time = total_time ∧
    aerobics_time * weight_ratio = weight_time * aerobics_ratio ∧
    aerobics_time = 150 ∧
    weight_time = 100 := by
  sorry

end NUMINAMATH_CALUDE_exercise_time_distribution_l2859_285993


namespace NUMINAMATH_CALUDE_horner_v3_equals_108_l2859_285945

def horner_v (coeffs : List ℝ) (x : ℝ) : List ℝ :=
  coeffs.scanl (fun acc a => acc * x + a) 0

def f (x : ℝ) : ℝ := 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 6 * x + 7

theorem horner_v3_equals_108 :
  let coeffs := [2, -5, -4, 3, -6, 7]
  let x := 5
  let v := horner_v coeffs x
  v[3] = 108 := by sorry

end NUMINAMATH_CALUDE_horner_v3_equals_108_l2859_285945


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_dishonest_dealer_profit_result_l2859_285911

/-- Calculates the overall percent profit for a dishonest dealer selling two products. -/
theorem dishonest_dealer_profit (weight_A weight_B : ℝ) (cost_A cost_B : ℝ) 
  (discount_A discount_B : ℝ) (purchase_A purchase_B : ℝ) : ℝ :=
  let actual_weight_A := weight_A / 1000 * purchase_A
  let actual_weight_B := weight_B / 1000 * purchase_B
  let cost_price_A := actual_weight_A * cost_A
  let cost_price_B := actual_weight_B * cost_B
  let total_cost_price := cost_price_A + cost_price_B
  let selling_price_A := purchase_A * cost_A
  let selling_price_B := purchase_B * cost_B
  let discounted_price_A := selling_price_A * (1 - discount_A)
  let discounted_price_B := selling_price_B * (1 - discount_B)
  let total_selling_price := discounted_price_A + discounted_price_B
  let profit := total_selling_price - total_cost_price
  let percent_profit := profit / total_cost_price * 100
  percent_profit

/-- The overall percent profit is approximately 30.99% -/
theorem dishonest_dealer_profit_result : 
  ∃ ε > 0, |dishonest_dealer_profit 700 750 60 80 0.05 0.03 6 12 - 30.99| < ε :=
sorry

end NUMINAMATH_CALUDE_dishonest_dealer_profit_dishonest_dealer_profit_result_l2859_285911


namespace NUMINAMATH_CALUDE_two_equidistant_points_l2859_285912

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its normal vector and a point on the line -/
structure Line where
  normal : ℝ × ℝ
  point : ℝ × ℝ

/-- Configuration of a circle and two parallel tangent lines -/
structure CircleTangentConfig where
  circle : Circle
  tangent1 : Line
  tangent2 : Line
  d1 : ℝ  -- distance from circle center to tangent1
  d2 : ℝ  -- distance from circle center to tangent2

/-- Predicate to check if a point is equidistant from a circle and two lines -/
def isEquidistant (p : ℝ × ℝ) (c : Circle) (l1 l2 : Line) : Prop := sorry

/-- Main theorem: There are exactly two points equidistant from the circle and both tangents -/
theorem two_equidistant_points (config : CircleTangentConfig) 
  (h1 : config.d1 ≠ config.d2)
  (h2 : config.d1 > config.circle.radius)
  (h3 : config.d2 > config.circle.radius)
  (h4 : config.tangent1.normal = config.tangent2.normal) :  -- parallel tangents
  ∃! (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧ 
    isEquidistant p1 config.circle config.tangent1 config.tangent2 ∧ 
    isEquidistant p2 config.circle config.tangent1 config.tangent2 := by
  sorry

end NUMINAMATH_CALUDE_two_equidistant_points_l2859_285912


namespace NUMINAMATH_CALUDE_hill_climbing_speed_l2859_285976

/-- Proves that given a round trip journey where the total time is 6 hours (4 hours up, 2 hours down)
    and the average speed for the whole journey is 4 km/h, then the average speed while climbing
    to the top is 3 km/h. -/
theorem hill_climbing_speed
  (total_time : ℝ)
  (up_time : ℝ)
  (down_time : ℝ)
  (average_speed : ℝ)
  (h1 : total_time = up_time + down_time)
  (h2 : total_time = 6)
  (h3 : up_time = 4)
  (h4 : down_time = 2)
  (h5 : average_speed = 4) :
  (average_speed * total_time) / (2 * up_time) = 3 := by
  sorry

end NUMINAMATH_CALUDE_hill_climbing_speed_l2859_285976


namespace NUMINAMATH_CALUDE_balloon_arrangements_eq_36_l2859_285902

/-- The number of distinguishable arrangements of letters in "BALLOON" with vowels first -/
def balloon_arrangements : ℕ :=
  let vowels := ['A', 'O', 'O']
  let consonants := ['B', 'L', 'L', 'N']
  let vowel_arrangements := Nat.factorial 3 / Nat.factorial 2
  let consonant_arrangements := Nat.factorial 4 / Nat.factorial 2
  vowel_arrangements * consonant_arrangements

/-- Theorem stating that the number of distinguishable arrangements of "BALLOON" with vowels first is 36 -/
theorem balloon_arrangements_eq_36 : balloon_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_eq_36_l2859_285902


namespace NUMINAMATH_CALUDE_painters_work_days_l2859_285941

/-- Represents the number of work-days required for a given number of painters to complete a job -/
noncomputable def workDays (numPainters : ℕ) : ℝ :=
  sorry

theorem painters_work_days :
  (workDays 6 = 1.5) →
  (∀ (n m : ℕ), n * workDays n = m * workDays m) →
  workDays 4 = 2.25 :=
by
  sorry

end NUMINAMATH_CALUDE_painters_work_days_l2859_285941


namespace NUMINAMATH_CALUDE_monotonically_decreasing_quadratic_l2859_285988

/-- A function f is monotonically decreasing on an interval [a, b] if for all x, y in [a, b] with x ≤ y, we have f(x) ≥ f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y

/-- The theorem statement -/
theorem monotonically_decreasing_quadratic (a : ℝ) :
  MonotonicallyDecreasing (fun x => a * x^2 - 2 * x + 1) 1 10 ↔ a ≤ 1/10 :=
sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_quadratic_l2859_285988


namespace NUMINAMATH_CALUDE_quiz_ranking_l2859_285983

structure Student where
  name : String
  score : ℕ

def Hannah : Student := { name := "Hannah", score := 0 }
def Cassie : Student := { name := "Cassie", score := 0 }
def Bridget : Student := { name := "Bridget", score := 0 }

def is_not_highest (s : Student) (others : List Student) : Prop :=
  ∃ t ∈ others, t.score > s.score

def scored_better_than (s1 s2 : Student) : Prop :=
  s1.score > s2.score

def is_not_lowest (s : Student) (others : List Student) : Prop :=
  ∃ t ∈ others, s.score > t.score

theorem quiz_ranking :
  is_not_highest Hannah [Cassie, Bridget] →
  scored_better_than Bridget Cassie →
  is_not_lowest Cassie [Hannah, Bridget] →
  scored_better_than Bridget Cassie ∧ scored_better_than Cassie Hannah :=
by sorry

end NUMINAMATH_CALUDE_quiz_ranking_l2859_285983


namespace NUMINAMATH_CALUDE_solve_equation_l2859_285908

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.03) : x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2859_285908


namespace NUMINAMATH_CALUDE_solve_paint_problem_l2859_285977

def paint_problem (original_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) : Prop :=
  ∃ (cans_per_room : ℚ) (total_cans : ℕ),
    cans_per_room > 0 ∧
    total_cans * cans_per_room = original_rooms ∧
    (total_cans - lost_cans) * cans_per_room = remaining_rooms ∧
    remaining_rooms / cans_per_room = 17

theorem solve_paint_problem :
  paint_problem 42 4 34 := by
  sorry

end NUMINAMATH_CALUDE_solve_paint_problem_l2859_285977


namespace NUMINAMATH_CALUDE_base_seven_divisibility_l2859_285944

theorem base_seven_divisibility (y : ℕ) : 
  y ≤ 6 → (∃! y, (5 * 7^2 + y * 7 + 2) % 19 = 0 ∧ y ≤ 6) → y = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_divisibility_l2859_285944


namespace NUMINAMATH_CALUDE_ngo_employee_count_l2859_285907

/-- The number of illiterate employees -/
def illiterate_employees : ℕ := 20

/-- The decrease in total wages of illiterate employees in Rupees -/
def total_wage_decrease : ℕ := 300

/-- The decrease in average salary for all employees in Rupees -/
def average_salary_decrease : ℕ := 10

/-- The number of educated employees in the NGO -/
def educated_employees : ℕ := 10

theorem ngo_employee_count :
  educated_employees = total_wage_decrease / average_salary_decrease - illiterate_employees :=
by sorry

end NUMINAMATH_CALUDE_ngo_employee_count_l2859_285907


namespace NUMINAMATH_CALUDE_decimal_to_binary_111_octal_to_decimal_77_l2859_285943

/-- Converts a decimal number to its binary representation -/
def decimalToBinary (n : ℕ) : List Bool := sorry

/-- Converts an octal number to its decimal representation -/
def octalToDecimal (n : ℕ) : ℕ := sorry

theorem decimal_to_binary_111 :
  decimalToBinary 111 = [true, true, false, true, true, true, true] := by sorry

theorem octal_to_decimal_77 :
  octalToDecimal 77 = 63 := by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_111_octal_to_decimal_77_l2859_285943


namespace NUMINAMATH_CALUDE_power_two_2017_mod_7_l2859_285925

theorem power_two_2017_mod_7 : 2^2017 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_2017_mod_7_l2859_285925


namespace NUMINAMATH_CALUDE_animal_population_l2859_285937

theorem animal_population (lions leopards elephants : ℕ) : 
  lions = 200 →
  lions = 2 * leopards →
  elephants = (lions + leopards) / 2 →
  lions + leopards + elephants = 450 := by
sorry

end NUMINAMATH_CALUDE_animal_population_l2859_285937


namespace NUMINAMATH_CALUDE_michael_birdhouse_earnings_l2859_285969

/-- The amount of money Michael made from selling birdhouses -/
def michael_earnings (extra_large_price large_price medium_price small_price extra_small_price : ℕ)
  (extra_large_sold large_sold medium_sold small_sold extra_small_sold : ℕ) : ℕ :=
  extra_large_price * extra_large_sold +
  large_price * large_sold +
  medium_price * medium_sold +
  small_price * small_sold +
  extra_small_price * extra_small_sold

/-- Theorem stating that Michael's earnings from selling birdhouses is $487 -/
theorem michael_birdhouse_earnings :
  michael_earnings 45 22 16 10 5 3 5 7 8 10 = 487 := by sorry

end NUMINAMATH_CALUDE_michael_birdhouse_earnings_l2859_285969


namespace NUMINAMATH_CALUDE_book_sale_profit_l2859_285973

theorem book_sale_profit (total_cost selling_price_1 cost_1 : ℚ) : 
  total_cost = 500 →
  cost_1 = 291.67 →
  selling_price_1 = cost_1 * (1 - 15/100) →
  selling_price_1 = (total_cost - cost_1) * (1 + 19/100) →
  True := by sorry

end NUMINAMATH_CALUDE_book_sale_profit_l2859_285973


namespace NUMINAMATH_CALUDE_second_smallest_box_count_l2859_285926

theorem second_smallest_box_count : 
  (∃ n : ℕ, n > 0 ∧ n < 8 ∧ 12 * n % 10 = 6) ∧
  (∀ n : ℕ, n > 0 ∧ n < 8 → 12 * n % 10 ≠ 6) ∧
  12 * 8 % 10 = 6 := by
sorry

end NUMINAMATH_CALUDE_second_smallest_box_count_l2859_285926


namespace NUMINAMATH_CALUDE_locus_of_centers_l2859_285997

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₂ with equation (x - 3)² + y² = 25 -/
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

/-- A circle is externally tangent to C₁ if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₂ if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (5 - r)^2

/-- The locus of centers (a, b) of circles externally tangent to C₁ and
    internally tangent to C₂ satisfies the equation 4a² + 4b² - 52a - 169 = 0 -/
theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) →
  4 * a^2 + 4 * b^2 - 52 * a - 169 = 0 :=
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l2859_285997


namespace NUMINAMATH_CALUDE_paint_remaining_paint_problem_l2859_285970

theorem paint_remaining (initial_paint : ℝ) (first_day_usage : ℝ) (second_day_usage : ℝ) (spill_loss : ℝ) : ℝ :=
  let remaining_after_first_day := initial_paint - first_day_usage
  let remaining_after_second_day := remaining_after_first_day - second_day_usage
  let remaining_after_spill := remaining_after_second_day - spill_loss
  remaining_after_spill

theorem paint_problem : paint_remaining 1 (1/2) ((1/2)/2) ((1/4)/4) = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_paint_problem_l2859_285970


namespace NUMINAMATH_CALUDE_ice_cream_shop_problem_l2859_285934

/-- Ice cream shop problem -/
theorem ice_cream_shop_problem (total_revenue : ℕ) (cone_price : ℕ) (free_cones : ℕ) (n : ℕ) :
  total_revenue = 100 ∧ 
  cone_price = 2 ∧ 
  free_cones = 10 ∧
  (total_revenue / cone_price + free_cones) % n = 0 →
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_shop_problem_l2859_285934


namespace NUMINAMATH_CALUDE_probability_log3_is_integer_l2859_285946

/-- A three-digit number is a natural number between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The count of three-digit numbers that are powers of 3. -/
def CountPowersOfThree : ℕ := 2

/-- The total count of three-digit numbers. -/
def TotalThreeDigitNumbers : ℕ := 900

/-- The probability of a randomly chosen three-digit number being a power of 3. -/
def ProbabilityPowerOfThree : ℚ := CountPowersOfThree / TotalThreeDigitNumbers

theorem probability_log3_is_integer :
  ProbabilityPowerOfThree = 1 / 450 := by sorry

end NUMINAMATH_CALUDE_probability_log3_is_integer_l2859_285946


namespace NUMINAMATH_CALUDE_grandmother_age_l2859_285952

def cody_age : ℕ := 14
def grandmother_age_multiplier : ℕ := 6

theorem grandmother_age : 
  cody_age * grandmother_age_multiplier = 84 := by
  sorry

end NUMINAMATH_CALUDE_grandmother_age_l2859_285952


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l2859_285942

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l2859_285942


namespace NUMINAMATH_CALUDE_unique_number_between_9_and_9_1_cube_root_l2859_285957

theorem unique_number_between_9_and_9_1_cube_root (n : ℕ+) : 
  (∃ k : ℕ, n = 21 * k) ∧ 
  (9 < (n : ℝ) ^ (1/3) ∧ (n : ℝ) ^ (1/3) < 9.1) ↔ 
  n = 735 :=
sorry

end NUMINAMATH_CALUDE_unique_number_between_9_and_9_1_cube_root_l2859_285957


namespace NUMINAMATH_CALUDE_john_scores_42_points_l2859_285989

/-- The number of points John scores in a game, given the scoring pattern and game duration -/
def johnTotalPoints (pointsPer4Min : ℕ) (intervalsPer12Min : ℕ) (numPeriods : ℕ) : ℕ :=
  pointsPer4Min * intervalsPer12Min * numPeriods

/-- Theorem stating that John scores 42 points under the given conditions -/
theorem john_scores_42_points :
  let pointsPer4Min := 2 * 2 + 1 * 3  -- 2 shots worth 2 points and 1 shot worth 3 points
  let intervalsPer12Min := 12 / 4     -- Each period is 12 minutes, divided into 4-minute intervals
  let numPeriods := 2                 -- He plays for 2 periods
  johnTotalPoints pointsPer4Min intervalsPer12Min numPeriods = 42 := by
  sorry

#eval johnTotalPoints (2 * 2 + 1 * 3) (12 / 4) 2

end NUMINAMATH_CALUDE_john_scores_42_points_l2859_285989


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2859_285910

theorem trigonometric_identities (α : Real) 
  (h : 3 * Real.sin α - 2 * Real.cos α = 0) : 
  (((Real.cos α - Real.sin α) / (Real.cos α + Real.sin α)) + 
   ((Real.cos α + Real.sin α) / (Real.cos α - Real.sin α)) = 6) ∧ 
  ((Real.sin α)^2 - 2 * Real.sin α * Real.cos α + 4 * (Real.cos α)^2 = 28/13) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2859_285910


namespace NUMINAMATH_CALUDE_points_per_enemy_l2859_285971

theorem points_per_enemy (num_enemies : ℕ) (completion_bonus : ℕ) (total_points : ℕ) 
  (h1 : num_enemies = 6)
  (h2 : completion_bonus = 8)
  (h3 : total_points = 62) :
  (total_points - completion_bonus) / num_enemies = 9 := by
  sorry

end NUMINAMATH_CALUDE_points_per_enemy_l2859_285971


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2859_285950

theorem smallest_n_square_and_cube : 
  (∃ (n : ℕ), n > 0 ∧ 
   (∃ (k : ℕ), 5 * n = k^2) ∧ 
   (∃ (m : ℕ), 7 * n = m^3) ∧
   (∀ (x : ℕ), x > 0 → 
    (∃ (y : ℕ), 5 * x = y^2) → 
    (∃ (z : ℕ), 7 * x = z^3) → 
    x ≥ 1715)) ∧
  (∃ (k m : ℕ), 5 * 1715 = k^2 ∧ 7 * 1715 = m^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2859_285950


namespace NUMINAMATH_CALUDE_number_of_paths_l2859_285923

def grid_width : ℕ := 6
def grid_height : ℕ := 5
def path_length : ℕ := 8
def steps_right : ℕ := grid_width - 1
def steps_up : ℕ := grid_height - 1

theorem number_of_paths : 
  Nat.choose path_length steps_up = Nat.choose path_length (path_length - steps_right) := by
  sorry

end NUMINAMATH_CALUDE_number_of_paths_l2859_285923


namespace NUMINAMATH_CALUDE_jorge_goals_this_season_l2859_285940

/-- Given that Jorge scored 156 goals last season and his total goals are 343,
    prove that the number of goals he scored this season is 343 - 156. -/
theorem jorge_goals_this_season 
  (goals_last_season : ℕ) 
  (total_goals : ℕ) 
  (h1 : goals_last_season = 156) 
  (h2 : total_goals = 343) : 
  total_goals - goals_last_season = 343 - 156 :=
by sorry

end NUMINAMATH_CALUDE_jorge_goals_this_season_l2859_285940


namespace NUMINAMATH_CALUDE_exam_score_calculation_l2859_285965

/-- Given an examination with the following conditions:
  * Total number of questions is 120
  * Each correct answer scores 3 marks
  * Each wrong answer loses 1 mark
  * The total score is 180 marks
  This theorem proves that the number of correctly answered questions is 75. -/
theorem exam_score_calculation (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 120)
  (h2 : correct_score = 3)
  (h3 : wrong_score = -1)
  (h4 : total_score = 180) :
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_score + (total_questions - correct_answers) * wrong_score = total_score ∧ 
    correct_answers = 75 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l2859_285965


namespace NUMINAMATH_CALUDE_problem_solution_l2859_285984

theorem problem_solution : 
  ((-1 : ℝ) ^ 2023 + Real.sqrt 9 - 2022 ^ 0 = 1) ∧ 
  ((Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) + Real.sqrt 8 = 2 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2859_285984


namespace NUMINAMATH_CALUDE_tree_ratio_is_13_3_l2859_285975

/-- The ratio of trees planted to fallen Mahogany trees -/
def tree_ratio (initial_mahogany : ℕ) (initial_narra : ℕ) (total_fallen : ℕ) (final_count : ℕ) : ℚ :=
  let mahogany_fallen := (total_fallen + 1) / 2
  let trees_planted := final_count - (initial_mahogany + initial_narra - total_fallen)
  (trees_planted : ℚ) / mahogany_fallen

/-- The ratio of trees planted to fallen Mahogany trees is 13:3 -/
theorem tree_ratio_is_13_3 :
  tree_ratio 50 30 5 88 = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tree_ratio_is_13_3_l2859_285975


namespace NUMINAMATH_CALUDE_unanswered_test_completion_ways_l2859_285938

/-- Represents a multiple-choice test -/
structure MultipleChoiceTest where
  num_questions : ℕ
  choices_per_question : ℕ

/-- Calculates the number of ways to complete the test with all questions unanswered -/
def ways_to_complete_unanswered (test : MultipleChoiceTest) : ℕ := 1

/-- Theorem: For a 4-question test with 5 choices per question, there is only one way to complete it with all questions unanswered -/
theorem unanswered_test_completion_ways 
  (test : MultipleChoiceTest) 
  (h1 : test.num_questions = 4) 
  (h2 : test.choices_per_question = 5) : 
  ways_to_complete_unanswered test = 1 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_test_completion_ways_l2859_285938


namespace NUMINAMATH_CALUDE_cubic_polynomial_condition_l2859_285920

/-- A polynomial is cubic if its highest degree term is of degree 3 -/
def IsCubicPolynomial (p : Polynomial ℝ) : Prop :=
  p.degree = 3

theorem cubic_polynomial_condition (m n : ℕ) :
  IsCubicPolynomial (X * Y^(m-n) + (n-2) * X^2 * Y^2 + 1) →
  m + 2*n = 8 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_condition_l2859_285920


namespace NUMINAMATH_CALUDE_rotation_dilation_determinant_l2859_285924

theorem rotation_dilation_determinant :
  ∀ (E : Matrix (Fin 2) (Fin 2) ℝ),
  (∃ (R S : Matrix (Fin 2) (Fin 2) ℝ),
    R = !![0, -1; 1, 0] ∧
    S = !![5, 0; 0, 5] ∧
    E = S * R) →
  Matrix.det E = 25 := by
sorry

end NUMINAMATH_CALUDE_rotation_dilation_determinant_l2859_285924


namespace NUMINAMATH_CALUDE_identify_alkali_metal_l2859_285922

/-- Represents an alkali metal with its atomic mass -/
structure AlkaliMetal where
  atomic_mass : ℝ

/-- Represents a mixture of an alkali metal and its oxide -/
structure Mixture (R : AlkaliMetal) where
  initial_mass : ℝ
  final_mass : ℝ

/-- Theorem: If a mixture of alkali metal R and its oxide R₂O weighs 10.8 grams,
    and after reaction with water and drying, the resulting solid weighs 16 grams,
    then the atomic mass of R is 23. -/
theorem identify_alkali_metal (R : AlkaliMetal) (mix : Mixture R) :
  mix.initial_mass = 10.8 ∧ mix.final_mass = 16 → R.atomic_mass = 23 := by
  sorry

#check identify_alkali_metal

end NUMINAMATH_CALUDE_identify_alkali_metal_l2859_285922


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2859_285917

theorem inequality_solution_set :
  {x : ℝ | (4 : ℝ) / (x + 1) ≤ 1} = Set.Iic (-1) ∪ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2859_285917


namespace NUMINAMATH_CALUDE_games_missed_l2859_285995

/-- Given that Benny's high school played 39 baseball games and he attended 14 games,
    prove that the number of games Benny missed is 25. -/
theorem games_missed (total_games : ℕ) (games_attended : ℕ) (h1 : total_games = 39) (h2 : games_attended = 14) :
  total_games - games_attended = 25 := by
  sorry

end NUMINAMATH_CALUDE_games_missed_l2859_285995


namespace NUMINAMATH_CALUDE_min_value_a_min_value_sum_equality_condition_l2859_285906

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x - 2|

-- Theorem for part I
theorem min_value_a (a : ℝ) :
  (∃ x, f x a ≤ a) → a ≥ 1 :=
sorry

-- Define M as the minimum value of a
def M : ℝ := 1

-- Theorem for part II
theorem min_value_sum (m n p : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) 
  (h_sum : m + 2*n + 3*p = M) :
  3/m + 2/n + 1/p ≥ 6 + 2*Real.sqrt 6 + 2*Real.sqrt 2 :=
sorry

-- Theorem for equality condition
theorem equality_condition (m n p : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) 
  (h_sum : m + 2*n + 3*p = M) :
  3/m + 2/n + 1/p = 6 + 2*Real.sqrt 6 + 2*Real.sqrt 2 ↔ 
  m = 1/6 ∧ n = 1/4 ∧ p = 1/6 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_min_value_sum_equality_condition_l2859_285906


namespace NUMINAMATH_CALUDE_population_decrease_l2859_285919

/-- Proves that given an initial population of 15000, a 12% increase in the first year,
    and a final population of 14784 after two years, the percentage decrease in the second year is 12%. -/
theorem population_decrease (initial_population : ℝ) (first_year_increase : ℝ) (final_population : ℝ) :
  initial_population = 15000 →
  first_year_increase = 12 →
  final_population = 14784 →
  ∃ (second_year_decrease : ℝ),
    second_year_decrease = 12 ∧
    final_population = initial_population * (1 + first_year_increase / 100) * (1 - second_year_decrease / 100) :=
by sorry

end NUMINAMATH_CALUDE_population_decrease_l2859_285919


namespace NUMINAMATH_CALUDE_equation_and_inequalities_l2859_285932

theorem equation_and_inequalities (x a : ℝ) (hx : x ≠ 0) :
  (x⁻¹ + a * x = 1 ↔ a = (x - 1) / x^2) ∧
  (x⁻¹ + a * x > 1 ↔ (a > (x - 1) / x^2 ∧ x > 0) ∨ (a < (x - 1) / x^2 ∧ x < 0)) ∧
  (x⁻¹ + a * x < 1 ↔ (a < (x - 1) / x^2 ∧ x > 0) ∨ (a > (x - 1) / x^2 ∧ x < 0)) := by
  sorry

end NUMINAMATH_CALUDE_equation_and_inequalities_l2859_285932


namespace NUMINAMATH_CALUDE_union_A_B_when_m_2_range_m_for_A_subset_B_l2859_285964

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) / (x - 3/2) < 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - (m + 1)*x + m ≤ 0}

-- Theorem for part (1)
theorem union_A_B_when_m_2 : A ∪ B 2 = {x | -2 < x ∧ x ≤ 2} := by sorry

-- Theorem for part (2)
theorem range_m_for_A_subset_B : 
  {m | A ⊆ B m} = {m | -2 < m ∧ m < 3/2} := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_m_2_range_m_for_A_subset_B_l2859_285964


namespace NUMINAMATH_CALUDE_units_digit_factorial_product_squared_l2859_285903

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The main theorem -/
theorem units_digit_factorial_product_squared :
  unitsDigit ((factorial 1 * factorial 2 * factorial 3 * factorial 4) ^ 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_factorial_product_squared_l2859_285903


namespace NUMINAMATH_CALUDE_nora_game_probability_l2859_285921

theorem nora_game_probability (p_lose : ℚ) (h1 : p_lose = 5/8) (h2 : ¬ ∃ p_tie : ℚ, p_tie > 0) :
  ∃ p_win : ℚ, p_win = 3/8 ∧ p_win + p_lose = 1 := by
  sorry

end NUMINAMATH_CALUDE_nora_game_probability_l2859_285921


namespace NUMINAMATH_CALUDE_total_fruit_mass_is_7425_l2859_285963

/-- The number of apple trees in the orchard -/
def num_apple_trees : ℕ := 30

/-- The mass of apples produced by each apple tree (in kg) -/
def apple_yield_per_tree : ℕ := 150

/-- The number of peach trees in the orchard -/
def num_peach_trees : ℕ := 45

/-- The average mass of peaches produced by each peach tree (in kg) -/
def peach_yield_per_tree : ℕ := 65

/-- The total mass of fruit harvested in the orchard (in kg) -/
def total_fruit_mass : ℕ :=
  num_apple_trees * apple_yield_per_tree + num_peach_trees * peach_yield_per_tree

theorem total_fruit_mass_is_7425 : total_fruit_mass = 7425 := by
  sorry

end NUMINAMATH_CALUDE_total_fruit_mass_is_7425_l2859_285963


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_1001_l2859_285915

theorem modular_inverse_17_mod_1001 : ∃ x : ℕ, x ≤ 1000 ∧ (17 * x) % 1001 = 1 :=
by
  use 530
  sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_1001_l2859_285915


namespace NUMINAMATH_CALUDE_students_per_van_l2859_285913

theorem students_per_van (num_vans : ℕ) (num_minibusses : ℕ) (students_per_minibus : ℕ) (total_students : ℕ) :
  num_vans = 6 →
  num_minibusses = 4 →
  students_per_minibus = 24 →
  total_students = 156 →
  (total_students - num_minibusses * students_per_minibus) / num_vans = 10 :=
by sorry

end NUMINAMATH_CALUDE_students_per_van_l2859_285913


namespace NUMINAMATH_CALUDE_perfect_square_problem_l2859_285909

theorem perfect_square_problem (n : ℕ+) :
  ∃ k : ℕ, (n : ℤ)^2 + 19*(n : ℤ) + 48 = k^2 → n = 33 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_problem_l2859_285909


namespace NUMINAMATH_CALUDE_parallelogram_bisecting_line_slope_l2859_285955

/-- A parallelogram with vertices at (8,35), (8,90), (25,125), and (25,70) -/
structure Parallelogram where
  v1 : ℝ × ℝ := (8, 35)
  v2 : ℝ × ℝ := (8, 90)
  v3 : ℝ × ℝ := (25, 125)
  v4 : ℝ × ℝ := (25, 70)

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- The line cuts the parallelogram into two congruent polygons -/
def cuts_into_congruent_polygons (p : Parallelogram) (l : Line) : Prop := sorry

theorem parallelogram_bisecting_line_slope (p : Parallelogram) (l : Line) :
  cuts_into_congruent_polygons p l → l.slope = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_bisecting_line_slope_l2859_285955


namespace NUMINAMATH_CALUDE_problem_solution_l2859_285956

def f (k : ℝ) (x : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

theorem problem_solution :
  (∀ x : ℝ, x ≤ -1 → f 2 x = 0 ↔ x = (-1 - Real.sqrt 3) / 2) ∧
  (∀ k : ℝ, (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
    -7/2 < k ∧ k < -1) ∧
  (∀ k x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 →
    1/x₁ + 1/x₂ < 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2859_285956


namespace NUMINAMATH_CALUDE_range_of_f_l2859_285954

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc (-1 : ℝ) 2 = Set.Icc 3 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l2859_285954


namespace NUMINAMATH_CALUDE_card_problem_l2859_285948

theorem card_problem (w x y z : ℝ) : 
  x = w / 2 →
  y = w + x →
  z = 400 →
  w + x + y + z = 1000 →
  w = 200 := by
sorry

end NUMINAMATH_CALUDE_card_problem_l2859_285948


namespace NUMINAMATH_CALUDE_two_colonies_growth_time_l2859_285980

/-- Represents the number of days it takes for a colony to reach the habitat's limit -/
def daysToLimit : ℕ := 21

/-- Represents the daily growth factor of a bacteria colony -/
def growthFactor : ℕ := 2

/-- Represents the number of initial colonies -/
def initialColonies : ℕ := 2

theorem two_colonies_growth_time (daysToLimit : ℕ) (growthFactor : ℕ) (initialColonies : ℕ) :
  daysToLimit = 21 ∧ growthFactor = 2 ∧ initialColonies = 2 →
  daysToLimit = 21 :=
by sorry

end NUMINAMATH_CALUDE_two_colonies_growth_time_l2859_285980


namespace NUMINAMATH_CALUDE_inequality_proof_l2859_285994

theorem inequality_proof (a b : ℝ) : 
  (a^2 - 1) * (b^2 - 1) ≥ 0 → a^2 + b^2 - 1 - a^2*b^2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2859_285994


namespace NUMINAMATH_CALUDE_jade_transactions_l2859_285936

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 18 →
  jade = 84 :=
by sorry

end NUMINAMATH_CALUDE_jade_transactions_l2859_285936


namespace NUMINAMATH_CALUDE_garage_cleanup_l2859_285961

theorem garage_cleanup (total_trips : ℕ) (jean_extra_trips : ℕ) (total_capacity : ℝ) (actual_weight : ℝ) 
  (h1 : total_trips = 40)
  (h2 : jean_extra_trips = 6)
  (h3 : total_capacity = 8000)
  (h4 : actual_weight = 7850) : 
  let bill_trips := (total_trips - jean_extra_trips) / 2
  let jean_trips := bill_trips + jean_extra_trips
  let avg_weight := actual_weight / total_trips
  jean_trips = 23 ∧ avg_weight = 196.25 := by
  sorry

end NUMINAMATH_CALUDE_garage_cleanup_l2859_285961


namespace NUMINAMATH_CALUDE_sector_central_angle_l2859_285996

theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = 2/5 * Real.pi) :
  (2 * area) / (r^2) = π / 5 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2859_285996


namespace NUMINAMATH_CALUDE_polynomial_roots_l2859_285968

theorem polynomial_roots : ∃ (a b c : ℝ), 
  (a = -1 ∧ b = Real.sqrt 6 ∧ c = -Real.sqrt 6) ∧
  (∀ x : ℝ, x^3 + x^2 - 6*x - 6 = 0 ↔ (x = a ∨ x = b ∨ x = c)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2859_285968


namespace NUMINAMATH_CALUDE_chord_intersection_ratio_l2859_285933

-- Define a circle
variable (circle : Type) [AddCommGroup circle] [Module ℝ circle]

-- Define points on the circle
variable (E F G H Q : circle)

-- Define the lengths
variable (EQ FQ GQ HQ : ℝ)

-- State the theorem
theorem chord_intersection_ratio 
  (h1 : EQ = 5) 
  (h2 : GQ = 12) 
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_ratio_l2859_285933


namespace NUMINAMATH_CALUDE_walnut_trees_planted_park_walnut_trees_l2859_285928

/-- The number of walnut trees planted in a park -/
def trees_planted (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that the number of trees planted is the difference between final and initial counts -/
theorem walnut_trees_planted (initial : ℕ) (final : ℕ) (h : initial ≤ final) :
  trees_planted initial final = final - initial :=
by sorry

/-- The specific problem instance -/
theorem park_walnut_trees :
  trees_planted 22 55 = 33 :=
by sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_park_walnut_trees_l2859_285928


namespace NUMINAMATH_CALUDE_ellipse_intersection_range_l2859_285999

-- Define the line equation
def line (k : ℝ) (x y : ℝ) : Prop := y - k * x - 1 = 0

-- Define the ellipse equation
def ellipse (b : ℝ) (x y : ℝ) : Prop := x^2 / 5 + y^2 / b = 1

-- Define the condition that the line always intersects the ellipse
def always_intersects (b : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x y : ℝ, line k x y ∧ ellipse b x y

-- Theorem statement
theorem ellipse_intersection_range :
  ∀ b : ℝ, (always_intersects b) ↔ (b ∈ Set.Icc 1 5 ∪ Set.Ioi 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_range_l2859_285999


namespace NUMINAMATH_CALUDE_real_roots_range_not_p_and_q_implies_range_l2859_285951

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0
def q (m : ℝ) : Prop := m ∈ Set.Icc (-1 : ℝ) 5

-- Theorem 1
theorem real_roots_range (m : ℝ) : p m → m ∈ Set.Iic 1 := by sorry

-- Theorem 2
theorem not_p_and_q_implies_range (m : ℝ) : 
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) → m ∈ Set.Iio (-1) ∪ Set.Ioc 1 5 := by sorry

end NUMINAMATH_CALUDE_real_roots_range_not_p_and_q_implies_range_l2859_285951


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2859_285916

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 - 3 * Complex.I) = Complex.mk (-11/13) (29/13) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2859_285916


namespace NUMINAMATH_CALUDE_area_between_curves_l2859_285929

-- Define the functions
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := 2 - x^2

-- Define the intersection points
def x₁ : ℝ := -2
def x₂ : ℝ := 1

-- State the theorem
theorem area_between_curves : 
  (∫ (x : ℝ) in x₁..x₂, g x - f x) = 9/2 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l2859_285929


namespace NUMINAMATH_CALUDE_set_operations_l2859_285901

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3, 5}
def B : Set Nat := {3, 5, 6}

theorem set_operations :
  (A ∩ B = {3, 5}) ∧ ((U \ A) ∪ B = {3, 4, 5, 6}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2859_285901


namespace NUMINAMATH_CALUDE_total_orchestra_members_l2859_285918

/-- Represents the number of boys in the orchestra -/
def boys : ℕ := sorry

/-- Represents the number of girls in the orchestra -/
def girls : ℕ := sorry

/-- The number of girls is twice the number of boys -/
axiom girls_twice_boys : girls = 2 * boys

/-- If 24 girls are transferred, the number of boys will be twice the number of girls -/
axiom boys_twice_remaining_girls : boys = 2 * (girls - 24)

/-- The total number of boys and girls in the orchestra is 48 -/
theorem total_orchestra_members : boys + girls = 48 := by sorry

end NUMINAMATH_CALUDE_total_orchestra_members_l2859_285918


namespace NUMINAMATH_CALUDE_line_through_points_l2859_285972

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def on_line (a b x : V) : Prop := ∃ t : ℝ, x = a + t • (b - a)

theorem line_through_points (a b : V) (h : a ≠ b) :
  ∃ k m : ℝ, m = 5/8 ∧ on_line a b (k • a + m • b) → k = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2859_285972


namespace NUMINAMATH_CALUDE_number_problem_l2859_285959

theorem number_problem : ∃ x : ℝ, (0.10 * x + 0.08 * 24 = 5.92) ∧ (x = 40) := by sorry

end NUMINAMATH_CALUDE_number_problem_l2859_285959


namespace NUMINAMATH_CALUDE_square_value_l2859_285939

theorem square_value (square : ℚ) : 44 * 25 = square * 100 → square = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l2859_285939


namespace NUMINAMATH_CALUDE_pauls_allowance_l2859_285974

/-- Paul's savings in dollars -/
def savings : ℕ := 3

/-- Cost of one toy in dollars -/
def toy_cost : ℕ := 5

/-- Number of toys Paul wants to buy -/
def num_toys : ℕ := 2

/-- Paul's allowance in dollars -/
def allowance : ℕ := 7

theorem pauls_allowance :
  savings + allowance = num_toys * toy_cost :=
sorry

end NUMINAMATH_CALUDE_pauls_allowance_l2859_285974


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2859_285958

-- Define the complex number z as (m+1)(1-i)
def z (m : ℝ) : ℂ := (m + 1) * (1 - Complex.I)

-- Theorem statement
theorem pure_imaginary_condition (m : ℝ) :
  (z m).re = 0 → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2859_285958


namespace NUMINAMATH_CALUDE_physical_fitness_test_participation_l2859_285935

/-- The number of students who met the standards in the physical fitness test -/
def students_met_standards : ℕ := 900

/-- The percentage of students who took the test but did not meet the standards -/
def percentage_not_met_standards : ℚ := 25 / 100

/-- The percentage of students who did not participate in the test -/
def percentage_not_participated : ℚ := 4 / 100

/-- The total number of students in the sixth grade -/
def total_students : ℕ := 1200

/-- The number of students who did not participate in the physical fitness test -/
def students_not_participated : ℕ := 48

theorem physical_fitness_test_participation :
  (students_not_participated : ℚ) = percentage_not_participated * (total_students : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_physical_fitness_test_participation_l2859_285935


namespace NUMINAMATH_CALUDE_maximize_expected_score_l2859_285900

structure QuestionType where
  correct_prob : ℝ
  points : ℕ

def expected_score (first second : QuestionType) : ℝ :=
  first.correct_prob * (first.points + second.correct_prob * second.points) +
  first.correct_prob * (1 - second.correct_prob) * first.points

theorem maximize_expected_score (type_a type_b : QuestionType)
  (ha : type_a.correct_prob = 0.8)
  (hb : type_b.correct_prob = 0.6)
  (pa : type_a.points = 20)
  (pb : type_b.points = 80) :
  expected_score type_b type_a > expected_score type_a type_b :=
sorry

end NUMINAMATH_CALUDE_maximize_expected_score_l2859_285900


namespace NUMINAMATH_CALUDE_xi_eq_4_equiv_events_l2859_285986

/-- Represents the outcome of a single die roll -/
def DieRoll : Type := Fin 6

/-- Represents the outcome of rolling two dice -/
def TwoDiceRoll : Type := DieRoll × DieRoll

/-- The sum of the points obtained when rolling two dice -/
def ξ : TwoDiceRoll → Nat :=
  fun (d1, d2) => d1.val + 1 + d2.val + 1

/-- The event where one die shows 3 and the other shows 1 -/
def event_3_1 : Set TwoDiceRoll :=
  {roll | (roll.1.val = 2 ∧ roll.2.val = 0) ∨ (roll.1.val = 0 ∧ roll.2.val = 2)}

/-- The event where both dice show 2 -/
def event_2_2 : Set TwoDiceRoll :=
  {roll | roll.1.val = 1 ∧ roll.2.val = 1}

/-- The theorem stating that ξ = 4 is equivalent to the union of event_3_1 and event_2_2 -/
theorem xi_eq_4_equiv_events :
  {roll : TwoDiceRoll | ξ roll = 4} = event_3_1 ∪ event_2_2 := by
  sorry

end NUMINAMATH_CALUDE_xi_eq_4_equiv_events_l2859_285986


namespace NUMINAMATH_CALUDE_angles_in_range_l2859_285953

-- Define the set S
def S : Set ℝ := {x | ∃ k : ℤ, x = k * 360 + 370 + 23 / 60}

-- Define the range of angles
def inRange (x : ℝ) : Prop := -720 ≤ x ∧ x < 360

-- State the theorem
theorem angles_in_range :
  ∃! (a b c : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    inRange a ∧ inRange b ∧ inRange c ∧
    a = -709 - 37 / 60 ∧
    b = -349 - 37 / 60 ∧
    c = 10 + 23 / 60 :=
  sorry

end NUMINAMATH_CALUDE_angles_in_range_l2859_285953


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2859_285949

/-- A geometric sequence with positive terms and common ratio greater than 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : ∀ n, a n > 0
  h_ratio : q > 1
  h_geometric : ∀ n, a (n + 1) = q * a n

/-- Theorem statement for the geometric sequence problem -/
theorem geometric_sequence_problem (seq : GeometricSequence)
  (h1 : seq.a 3 + seq.a 5 = 20)
  (h2 : seq.a 2 * seq.a 6 = 64) :
  seq.a 6 = 32 := by
    sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2859_285949


namespace NUMINAMATH_CALUDE_circle_area_greater_than_five_times_triangle_area_l2859_285985

theorem circle_area_greater_than_five_times_triangle_area 
  (R r : ℝ) (S : ℝ) (h_R_positive : R > 0) (h_r_positive : r > 0) (h_S_positive : S > 0)
  (h_R_r : R ≥ 2 * r) -- Euler's inequality
  (h_S : S ≤ (3 * Real.sqrt 3 / 2) * R * r) -- Upper bound for triangle area
  : π * (R + r)^2 > 5 * S := by
  sorry

end NUMINAMATH_CALUDE_circle_area_greater_than_five_times_triangle_area_l2859_285985


namespace NUMINAMATH_CALUDE_angle_of_inclination_range_l2859_285981

theorem angle_of_inclination_range (θ : Real) (x y : Real) :
  x - y * Real.sin θ + 1 = 0 →
  ∃ α, α ∈ Set.Icc (π/4) (3*π/4) ∧
       (α = π/2 ∨ Real.tan α = 1 / Real.sin θ) :=
by sorry

end NUMINAMATH_CALUDE_angle_of_inclination_range_l2859_285981


namespace NUMINAMATH_CALUDE_gasoline_distribution_possible_l2859_285905

/-- Represents the state of the gasoline distribution system -/
structure GasolineState where
  barrel : ℕ
  bucket1 : ℕ
  bucket2 : ℕ
  scoop : ℕ

/-- Represents a single operation in the distribution process -/
inductive Operation
  | FillBucketFromBarrel (bucket : ℕ)
  | EmptyScoopToBarrel
  | TransferFromBucketToScoop (bucket : ℕ) (amount : ℕ)
  | TransferFromScoopToBucket (bucket : ℕ)

/-- Applies an operation to the current state -/
def applyOperation (state : GasolineState) (op : Operation) : GasolineState :=
  match op with
  | Operation.FillBucketFromBarrel bucket => sorry
  | Operation.EmptyScoopToBarrel => sorry
  | Operation.TransferFromBucketToScoop bucket amount => sorry
  | Operation.TransferFromScoopToBucket bucket => sorry

/-- Checks if the distribution goal has been achieved -/
def isGoalAchieved (state : GasolineState) : Prop :=
  state.bucket1 = 6 ∧ state.bucket2 = 6

/-- Theorem stating that the gasoline distribution is possible -/
theorem gasoline_distribution_possible : ∃ (ops : List Operation),
  let finalState := ops.foldl applyOperation ⟨28, 0, 0, 0⟩
  isGoalAchieved finalState :=
sorry

end NUMINAMATH_CALUDE_gasoline_distribution_possible_l2859_285905


namespace NUMINAMATH_CALUDE_sin_2x_value_l2859_285991

theorem sin_2x_value (x : ℝ) (h : Real.sin (x + π/4) = -3/5) : 
  Real.sin (2*x) = -7/25 := by sorry

end NUMINAMATH_CALUDE_sin_2x_value_l2859_285991


namespace NUMINAMATH_CALUDE_leah_lost_money_proof_l2859_285979

def leah_lost_money (initial_earnings : ℝ) (milkshake_fraction : ℝ) (comic_book_fraction : ℝ) (savings_fraction : ℝ) (not_shredded_fraction : ℝ) : ℝ :=
  let remaining_after_milkshake := initial_earnings - milkshake_fraction * initial_earnings
  let remaining_after_comic := remaining_after_milkshake - comic_book_fraction * remaining_after_milkshake
  let remaining_after_savings := remaining_after_comic - savings_fraction * remaining_after_comic
  let not_shredded := not_shredded_fraction * remaining_after_savings
  remaining_after_savings - not_shredded

theorem leah_lost_money_proof :
  leah_lost_money 28 (1/7) (1/5) (3/8) 0.1 = 10.80 := by
  sorry

end NUMINAMATH_CALUDE_leah_lost_money_proof_l2859_285979
