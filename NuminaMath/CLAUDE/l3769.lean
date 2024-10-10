import Mathlib

namespace eleven_team_league_games_l3769_376942

/-- The number of games played in a league where each team plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 11 teams, where each team plays every other team exactly once, 
    the total number of games played is 55. -/
theorem eleven_team_league_games : games_played 11 = 55 := by
  sorry

end eleven_team_league_games_l3769_376942


namespace equation_solution_l3769_376931

theorem equation_solution : ∃ x : ℝ, (3 / 4 - 1 / x = 1 / 2) ∧ (x = 4) := by
  sorry

end equation_solution_l3769_376931


namespace chord_length_concentric_circles_l3769_376973

/-- Given two concentric circles with radii R and r, where R > r, 
    and the area of the ring between them is 16π square inches,
    the length of a chord of the larger circle that is tangent to the smaller circle is 8 inches. -/
theorem chord_length_concentric_circles 
  (R r : ℝ) 
  (h1 : R > r) 
  (h2 : π * R^2 - π * r^2 = 16 * π) : 
  ∃ (c : ℝ), c = 8 ∧ c^2 = 4 * (R^2 - r^2) :=
sorry

end chord_length_concentric_circles_l3769_376973


namespace sum_of_digits_nine_times_ascending_l3769_376975

/-- A function that checks if a natural number has digits in ascending order -/
def has_ascending_digits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get i < (n.digits 10).get j

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem: For any number A with digits in ascending order, 
    the sum of digits of 9 * A is always 9 -/
theorem sum_of_digits_nine_times_ascending (A : ℕ) 
  (h : has_ascending_digits A) : sum_of_digits (9 * A) = 9 :=
by sorry

end sum_of_digits_nine_times_ascending_l3769_376975


namespace ingrid_income_proof_l3769_376921

/-- The annual income of John in dollars -/
def john_income : ℝ := 56000

/-- The tax rate for John as a decimal -/
def john_tax_rate : ℝ := 0.30

/-- The tax rate for Ingrid as a decimal -/
def ingrid_tax_rate : ℝ := 0.40

/-- The combined tax rate for John and Ingrid as a decimal -/
def combined_tax_rate : ℝ := 0.3569

/-- Ingrid's income in dollars -/
def ingrid_income : ℝ := 73924.13

/-- Theorem stating that given the conditions, Ingrid's income is correct -/
theorem ingrid_income_proof :
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) = combined_tax_rate :=
by sorry

end ingrid_income_proof_l3769_376921


namespace fraction_subtraction_property_l3769_376946

theorem fraction_subtraction_property (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b - c / d = (a - c) / (b + d)) ↔ (a / c = (b / d)^2) :=
sorry

end fraction_subtraction_property_l3769_376946


namespace distance_between_cars_l3769_376972

/-- The distance between two cars on a road after they travel towards each other -/
theorem distance_between_cars (initial_distance car1_distance car2_distance : ℝ) :
  initial_distance = 150 ∧ 
  car1_distance = 50 ∧ 
  car2_distance = 35 →
  initial_distance - (car1_distance + car2_distance) = 65 := by
  sorry


end distance_between_cars_l3769_376972


namespace probability_red_then_white_l3769_376953

/-- The probability of drawing a red ball followed by a white ball in two successive draws with replacement -/
theorem probability_red_then_white (total : ℕ) (red : ℕ) (white : ℕ) 
  (h_total : total = 9)
  (h_red : red = 3)
  (h_white : white = 2) :
  (red : ℚ) / total * (white : ℚ) / total = 2 / 27 := by
  sorry

end probability_red_then_white_l3769_376953


namespace triangle_circle_tangent_l3769_376902

theorem triangle_circle_tangent (a b c : ℝ) (x : ℝ) :
  -- Triangle ABC is a right triangle
  a^2 = b^2 + c^2 →
  -- Perimeter of triangle ABC is 190
  a + b + c = 190 →
  -- Circle with radius 23 centered at O on AB is tangent to BC
  (b - x) / b = 23 / a →
  -- AO = x (where O is the center of the circle)
  x^2 + (b - x)^2 = c^2 →
  -- The length of AO is 67
  x = 67 := by
    sorry

#eval 67 + 1  -- x + y = 68

end triangle_circle_tangent_l3769_376902


namespace jason_work_experience_l3769_376954

/-- Calculates the total work experience in months given years as bartender and years and months as manager -/
def total_work_experience (bartender_years : ℕ) (manager_years : ℕ) (manager_months : ℕ) : ℕ := 
  bartender_years * 12 + manager_years * 12 + manager_months

/-- Proves that Jason's total work experience is 150 months -/
theorem jason_work_experience : 
  total_work_experience 9 3 6 = 150 := by
  sorry

end jason_work_experience_l3769_376954


namespace fourth_power_sum_equals_51_to_fourth_l3769_376915

theorem fourth_power_sum_equals_51_to_fourth : 
  ∃! (n : ℕ+), 50^4 + 43^4 + 36^4 + 6^4 = n^4 :=
by sorry

end fourth_power_sum_equals_51_to_fourth_l3769_376915


namespace fifth_root_of_unity_sum_l3769_376945

theorem fifth_root_of_unity_sum (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^12 + ω^15 + ω^18 + ω^21 + ω^24 = 0 := by
  sorry

end fifth_root_of_unity_sum_l3769_376945


namespace arithmetic_sequence_scalar_multiple_l3769_376909

theorem arithmetic_sequence_scalar_multiple
  (a : ℕ → ℝ) (d c : ℝ) (h_arith : ∀ n, a (n + 1) - a n = d) (h_c : c ≠ 0) :
  ∃ (b : ℕ → ℝ), (∀ n, b n = c * a n) ∧ (∀ n, b (n + 1) - b n = c * d) :=
sorry

end arithmetic_sequence_scalar_multiple_l3769_376909


namespace inequality_proof_l3769_376918

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  2 * x + 1 / (x^2 - 2*x*y + y^2) ≥ 2 * y + 3 := by
  sorry

end inequality_proof_l3769_376918


namespace jills_number_satisfies_conditions_l3769_376991

def jills_favorite_number := 98

theorem jills_number_satisfies_conditions :
  -- 98 is even
  Even jills_favorite_number ∧
  -- 98 has repeating prime factors
  ∃ p : Nat, Prime p ∧ (jills_favorite_number % (p * p) = 0) ∧
  -- 7 is a prime factor of 98
  jills_favorite_number % 7 = 0 := by
  sorry

end jills_number_satisfies_conditions_l3769_376991


namespace buffet_dressing_cases_l3769_376905

/-- Represents the number of cases for each type of dressing -/
structure DressingCases where
  ranch : ℕ
  caesar : ℕ
  italian : ℕ
  thousandIsland : ℕ

/-- Checks if the ratios between dressing cases are correct -/
def correctRatios (cases : DressingCases) : Prop :=
  7 * cases.caesar = 2 * cases.ranch ∧
  cases.caesar * 3 = cases.italian ∧
  3 * cases.thousandIsland = 2 * cases.italian

/-- The theorem to be proved -/
theorem buffet_dressing_cases : 
  ∃ (cases : DressingCases), 
    cases.ranch = 28 ∧
    cases.caesar = 8 ∧
    cases.italian = 24 ∧
    cases.thousandIsland = 16 ∧
    correctRatios cases :=
by sorry

end buffet_dressing_cases_l3769_376905


namespace intersection_A_complement_B_when_m_3_m_value_for_given_intersection_l3769_376981

-- Define set A
def A : Set ℝ := {x | 6 / (x + 1) ≥ 1}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem 1
theorem intersection_A_complement_B_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem m_value_for_given_intersection :
  ∃ m : ℝ, A ∩ B m = {x | -1 < x ∧ x < 4} ∧ m = 8 := by sorry

end intersection_A_complement_B_when_m_3_m_value_for_given_intersection_l3769_376981


namespace shipment_arrival_time_l3769_376960

/-- Calculates the number of days until a shipment arrives at a warehouse -/
def daysUntilArrival (daysSinceDeparture : ℕ) (navigationDays : ℕ) (customsDays : ℕ) (warehouseArrivalDay : ℕ) : ℕ :=
  let daysInPort := daysSinceDeparture - navigationDays
  let daysAfterCustoms := daysInPort - customsDays
  warehouseArrivalDay - daysAfterCustoms

theorem shipment_arrival_time :
  daysUntilArrival 30 21 4 7 = 2 := by
  sorry

end shipment_arrival_time_l3769_376960


namespace solve_exponential_equation_l3769_376971

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ) ^ (3 * x) = 27 ^ (1/3) ∧ x = 1/3 := by
  sorry

end solve_exponential_equation_l3769_376971


namespace escalator_speed_l3769_376937

theorem escalator_speed (escalator_speed : ℝ) (escalator_length : ℝ) (time_taken : ℝ) 
  (h1 : escalator_speed = 11)
  (h2 : escalator_length = 126)
  (h3 : time_taken = 9) :
  escalator_speed + (escalator_length - escalator_speed * time_taken) / time_taken = 14 :=
by sorry

end escalator_speed_l3769_376937


namespace mechanic_average_earning_l3769_376955

/-- The average earning of a mechanic for a week, given specific conditions --/
theorem mechanic_average_earning (first_four_avg : ℚ) (last_four_avg : ℚ) (fourth_day : ℚ) :
  first_four_avg = 18 →
  last_four_avg = 22 →
  fourth_day = 13 →
  (4 * first_four_avg + 4 * last_four_avg - fourth_day) / 7 = 160 / 7 := by
  sorry

#eval (160 : ℚ) / 7

end mechanic_average_earning_l3769_376955


namespace white_balls_count_l3769_376933

theorem white_balls_count (x : ℕ) : 
  (3 : ℚ) / (3 + x) = 1/5 → x = 12 := by
  sorry

end white_balls_count_l3769_376933


namespace science_club_officer_selection_l3769_376993

def science_club_officers (n : ℕ) (k : ℕ) (special_members : ℕ) : ℕ :=
  (n - special_members).choose k + special_members * (special_members - 1) * (n - special_members)

theorem science_club_officer_selection :
  science_club_officers 25 3 2 = 10764 :=
by sorry

end science_club_officer_selection_l3769_376993


namespace triharmonic_properties_l3769_376919

-- Define a triharmonic quadruple
def is_triharmonic (A B C D : ℝ × ℝ) : Prop :=
  (dist A B) * (dist C D) = (dist A C) * (dist B D) ∧
  (dist A B) * (dist C D) = (dist A D) * (dist B C)

-- Define concyclicity
def are_concyclic (A B C D : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ) (r : ℝ), r > 0 ∧
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r

theorem triharmonic_properties
  (A B C D A1 B1 C1 D1 : ℝ × ℝ)
  (h1 : is_triharmonic A B C D)
  (h2 : is_triharmonic A1 B C D)
  (h3 : is_triharmonic A B1 C D)
  (h4 : is_triharmonic A B C1 D)
  (h5 : is_triharmonic A B C D1)
  (hA : A1 ≠ A) (hB : B1 ≠ B) (hC : C1 ≠ C) (hD : D1 ≠ D) :
  are_concyclic A B C1 D1 ∧ is_triharmonic A1 B1 C1 D1 := by
  sorry

end triharmonic_properties_l3769_376919


namespace droid_coffee_ratio_l3769_376963

/-- The ratio of afternoon to morning coffee bean usage in Droid's coffee shop --/
def afternoon_to_morning_ratio (morning_bags : ℕ) (total_weekly_bags : ℕ) : ℚ :=
  let afternoon_ratio := (total_weekly_bags / 7 - morning_bags - 2 * morning_bags) / morning_bags
  afternoon_ratio

/-- Theorem stating that the ratio of afternoon to morning coffee bean usage is 3 --/
theorem droid_coffee_ratio :
  afternoon_to_morning_ratio 3 126 = 3 := by sorry

end droid_coffee_ratio_l3769_376963


namespace part1_part2_l3769_376970

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x - 1

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ x - 3) ↔ (0 ≤ a ∧ a ≤ 8) :=
sorry

-- Part 2
theorem part2 (a : ℝ) (h : a < 0) :
  (∀ x : ℝ, f a x < 0 ↔ 
    (a = -1 ∧ x ≠ 1) ∨
    (-1 < a ∧ a < 0 ∧ (x < 1 ∨ x > -1/a)) ∨
    (a < -1 ∧ (x < -1/a ∨ x > 1))) :=
sorry

end part1_part2_l3769_376970


namespace black_region_area_l3769_376913

theorem black_region_area (large_square_side : ℝ) (small_square_side : ℝ) :
  large_square_side = 10 →
  small_square_side = 4 →
  (large_square_side ^ 2) - 2 * (small_square_side ^ 2) = 68 := by
  sorry

end black_region_area_l3769_376913


namespace square_sum_given_linear_equations_l3769_376901

theorem square_sum_given_linear_equations :
  ∀ x y : ℝ, 3 * x + 2 * y = 20 → 4 * x + 2 * y = 26 → x^2 + y^2 = 37 := by
sorry

end square_sum_given_linear_equations_l3769_376901


namespace reciprocal_roots_condition_l3769_376938

/-- The roots of the quadratic equation 2x^2 + 5x + k = 0 are reciprocal if and only if k = 2 -/
theorem reciprocal_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x * y = 1 ∧ 2 * x^2 + 5 * x + k = 0 ∧ 2 * y^2 + 5 * y + k = 0) ↔ 
  k = 2 :=
by sorry

end reciprocal_roots_condition_l3769_376938


namespace flower_shop_cost_l3769_376924

/-- The total cost of buying roses and lilies with given conditions -/
theorem flower_shop_cost : 
  let num_roses : ℕ := 20
  let num_lilies : ℕ := (3 * num_roses) / 4
  let cost_per_rose : ℕ := 5
  let cost_per_lily : ℕ := 2 * cost_per_rose
  let total_cost : ℕ := num_roses * cost_per_rose + num_lilies * cost_per_lily
  total_cost = 250 := by
  sorry

end flower_shop_cost_l3769_376924


namespace parallel_lines_condition_l3769_376979

/-- Two lines are parallel if their slopes are equal and they are not identical. -/
def are_parallel (m n : ℝ) : Prop :=
  (m = 1 ∧ n ≠ -1) ∨ (m = -1 ∧ n ≠ 1)

/-- The theorem states that two lines mx+y-n=0 and x+my+1=0 are parallel
    if and only if (m=1 and n≠-1) or (m=-1 and n≠1). -/
theorem parallel_lines_condition (m n : ℝ) :
  are_parallel m n ↔ ∀ x y : ℝ, (m * x + y - n = 0 ↔ x + m * y + 1 = 0) :=
sorry

end parallel_lines_condition_l3769_376979


namespace triangle_circumscribed_circle_diameter_l3769_376950

/-- Given a triangle with one side of 12 inches and the opposite angle of 30°,
    the diameter of the circumscribed circle is 24 inches. -/
theorem triangle_circumscribed_circle_diameter
  (side : ℝ) (angle : ℝ) :
  side = 12 →
  angle = 30 * π / 180 →
  side / Real.sin angle = 24 :=
by sorry

end triangle_circumscribed_circle_diameter_l3769_376950


namespace perpendicular_vectors_l3769_376910

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is perpendicular to c, then k = -3 -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) (k : ℝ) : 
  a = (Real.sqrt 3, 1) → 
  b = (0, 1) → 
  c = (k, Real.sqrt 3) → 
  (a.1 + 2 * b.1, a.2 + 2 * b.2) • c = 0 → 
  k = -3 := by
  sorry

end perpendicular_vectors_l3769_376910


namespace range_of_a_l3769_376900

-- Define a decreasing function on (0, +∞)
variable (f : ℝ → ℝ)
variable (h_decreasing : ∀ x y, 0 < x → 0 < y → x < y → f y < f x)

-- Define the domain of f
variable (h_domain : ∀ x, 0 < x → f x ∈ Set.range f)

-- Define the variable a
variable (a : ℝ)

-- State the theorem
theorem range_of_a (h_ineq : f (2*a^2 + a + 1) < f (3*a^2 - 4*a + 1)) :
  (0 < a ∧ a < 1/3) ∨ (1 < a ∧ a < 5) := by
  sorry

end range_of_a_l3769_376900


namespace smallest_five_digit_base3_palindrome_is_10001_l3769_376965

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from one base to another -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- Returns the number of digits of a number in a given base -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_five_digit_base3_palindrome_is_10001 :
  ∃ (otherBase : ℕ),
    otherBase ≠ 3 ∧
    otherBase > 1 ∧
    isPalindrome 10001 3 ∧
    numDigits 10001 3 = 5 ∧
    isPalindrome (baseConvert 10001 3 otherBase) otherBase ∧
    numDigits (baseConvert 10001 3 otherBase) otherBase = 3 ∧
    ∀ (n : ℕ),
      n < 10001 →
      (isPalindrome n 3 ∧ numDigits n 3 = 5) →
      ¬∃ (b : ℕ), b ≠ 3 ∧ b > 1 ∧
        isPalindrome (baseConvert n 3 b) b ∧
        numDigits (baseConvert n 3 b) b = 3 :=
by sorry

end smallest_five_digit_base3_palindrome_is_10001_l3769_376965


namespace parallel_lines_a_value_l3769_376969

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) → m₁ = m₂

/-- The value of a for which the lines x + ay - 1 = 0 and (a-1)x + ay + 1 = 0 are parallel -/
theorem parallel_lines_a_value : 
  (∀ x y, x + a * y - 1 = 0 ↔ (a - 1) * x + a * y + 1 = 0) → a = 2 :=
by sorry

end parallel_lines_a_value_l3769_376969


namespace divisibility_concatenation_l3769_376923

theorem divisibility_concatenation (a b : ℕ) : 
  100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 →  -- a and b are three-digit numbers
  ¬(37 ∣ a) →  -- a is not divisible by 37
  ¬(37 ∣ b) →  -- b is not divisible by 37
  (37 ∣ (a + b)) →  -- a + b is divisible by 37
  (37 ∣ (1000 * a + b))  -- 1000a + b is divisible by 37
  := by sorry

end divisibility_concatenation_l3769_376923


namespace max_value_quadratic_l3769_376951

theorem max_value_quadratic :
  ∃ (M : ℝ), M = 26 ∧ ∀ (x : ℝ), -3 * x^2 + 18 * x - 1 ≤ M :=
sorry

end max_value_quadratic_l3769_376951


namespace polynomial_division_remainder_l3769_376930

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^4 - 3•X + 1 : Polynomial ℝ) = (X^2 - X - 1) * q + 3 := by
  sorry

end polynomial_division_remainder_l3769_376930


namespace min_sum_positive_integers_l3769_376943

theorem min_sum_positive_integers (x y z w : ℕ+) 
  (h : (2 : ℕ) * x ^ 2 = (5 : ℕ) * y ^ 3 ∧ 
       (5 : ℕ) * y ^ 3 = (8 : ℕ) * z ^ 4 ∧ 
       (8 : ℕ) * z ^ 4 = (3 : ℕ) * w) : 
  x + y + z + w ≥ 54 := by
sorry

end min_sum_positive_integers_l3769_376943


namespace imaginary_part_of_complex_fraction_l3769_376988

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := i / (1 - i)
  Complex.im z = 1 / 2 := by sorry

end imaginary_part_of_complex_fraction_l3769_376988


namespace cubic_equation_solution_l3769_376982

theorem cubic_equation_solution (x : ℝ) (h : 9 / x^2 = x / 25) : x = (225 : ℝ)^(1/3) := by
  sorry

end cubic_equation_solution_l3769_376982


namespace three_digit_difference_divisible_by_nine_l3769_376967

theorem three_digit_difference_divisible_by_nine :
  ∀ (a b c : ℕ), 
    a ≤ 9 → b ≤ 9 → c ≤ 9 → a ≠ 0 →
    ∃ (k : ℤ), (100 * a + 10 * b + c) - (a + b + c) = 9 * k := by
  sorry

end three_digit_difference_divisible_by_nine_l3769_376967


namespace range_of_a_l3769_376983

theorem range_of_a (a : ℝ) : 
  (∀ x, x > a → x^2 + x - 2 > 0) ∧ 
  (∃ x, x^2 + x - 2 > 0 ∧ x ≤ a) → 
  a ≥ 1 := by
sorry

end range_of_a_l3769_376983


namespace horse_grain_consumption_l3769_376940

/-- Calculates the amount of grain each horse eats per day -/
theorem horse_grain_consumption
  (num_horses : ℕ)
  (oats_per_meal : ℕ)
  (oats_meals_per_day : ℕ)
  (total_days : ℕ)
  (total_food : ℕ)
  (h1 : num_horses = 4)
  (h2 : oats_per_meal = 4)
  (h3 : oats_meals_per_day = 2)
  (h4 : total_days = 3)
  (h5 : total_food = 132) :
  (total_food - num_horses * oats_per_meal * oats_meals_per_day * total_days) / (num_horses * total_days) = 3 := by
  sorry

end horse_grain_consumption_l3769_376940


namespace rectangular_enclosure_properties_l3769_376966

/-- Represents the area of a rectangular enclosure with perimeter 32 meters and side length x -/
def area (x : ℝ) : ℝ := -x^2 + 16*x

/-- Theorem stating the properties of the rectangular enclosure -/
theorem rectangular_enclosure_properties :
  ∀ x : ℝ, 0 < x → x < 16 →
  (∀ y : ℝ, y = area x → 
    (y = 60 → (x = 6 ∨ x = 10)) ∧
    (y ≤ 64) ∧
    (y = 64 ↔ x = 8)) :=
by sorry

end rectangular_enclosure_properties_l3769_376966


namespace waiter_tips_l3769_376952

/-- Calculates the total tips earned by a waiter --/
def calculate_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Proves that the waiter earned $27 in tips --/
theorem waiter_tips : calculate_tips 7 4 9 = 27 := by
  sorry

end waiter_tips_l3769_376952


namespace banana_count_l3769_376907

/-- The number of bananas Melissa had initially -/
def initial_bananas : ℕ := 88

/-- The number of bananas Melissa shared -/
def shared_bananas : ℕ := 4

/-- The number of bananas Melissa had left after sharing -/
def remaining_bananas : ℕ := 84

theorem banana_count : initial_bananas = shared_bananas + remaining_bananas := by
  sorry

end banana_count_l3769_376907


namespace equation_represents_parallel_lines_l3769_376922

theorem equation_represents_parallel_lines :
  ∃ (m c₁ c₂ : ℝ), m ≠ 0 ∧ c₁ ≠ c₂ ∧
  ∀ (x y : ℝ), x^2 - 9*y^2 + 3*x = 0 ↔ (y = m*x + c₁ ∨ y = m*x + c₂) :=
sorry

end equation_represents_parallel_lines_l3769_376922


namespace diophantine_equation_solutions_l3769_376944

theorem diophantine_equation_solutions :
  ∀ m n : ℕ+, 7^(m : ℕ) - 3 * 2^(n : ℕ) = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) := by
sorry

end diophantine_equation_solutions_l3769_376944


namespace part_1_part_2_l3769_376904

/-- Definition of set A -/
def A (a : ℝ) : Set ℝ := {a - 3, 2 * a - 1, a^2 + 1}

/-- Definition of set B -/
def B (x : ℝ) : Set ℝ := {0, 1, x}

/-- Theorem for part 1 -/
theorem part_1 (a : ℝ) : -3 ∈ A a → a = 0 ∨ a = -1 := by sorry

/-- Theorem for part 2 -/
theorem part_2 (x : ℝ) : x^2 ∈ B x → x = -1 := by sorry

end part_1_part_2_l3769_376904


namespace conditional_extremum_l3769_376926

/-- The objective function to be optimized -/
def f (x₁ x₂ : ℝ) : ℝ := x₁^2 + x₂^2 - x₁*x₂ + x₁ + x₂ - 6

/-- The constraint function -/
def g (x₁ x₂ : ℝ) : ℝ := x₁ + x₂ + 3

/-- Theorem stating the conditional extremum of f subject to g -/
theorem conditional_extremum :
  ∃ (x₁ x₂ : ℝ), g x₁ x₂ = 0 ∧ 
    (∀ (y₁ y₂ : ℝ), g y₁ y₂ = 0 → f x₁ x₂ ≤ f y₁ y₂) ∧
    x₁ = -3/2 ∧ x₂ = -3/2 ∧ f x₁ x₂ = -9/2 :=
sorry

end conditional_extremum_l3769_376926


namespace parallel_planes_theorem_l3769_376956

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations and operations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (intersect : Line → Line → Set Point)
variable (plane_parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_theorem 
  (l m : Line) (α β : Plane) (P : Point) :
  l ≠ m →
  α ≠ β →
  subset l α →
  subset m α →
  intersect l m = {P} →
  parallel l β →
  parallel m β →
  plane_parallel α β :=
sorry

end parallel_planes_theorem_l3769_376956


namespace optimal_pen_area_optimal_parallel_side_l3769_376994

/-- The length of the side parallel to the shed that maximizes the rectangular goat pen area -/
def optimal_parallel_side_length : ℝ := 50

/-- The total length of fence available -/
def total_fence_length : ℝ := 100

/-- The length of the shed -/
def shed_length : ℝ := 300

/-- The area of the pen as a function of the perpendicular side length -/
def pen_area (y : ℝ) : ℝ := y * (total_fence_length - 2 * y)

theorem optimal_pen_area :
  ∀ y : ℝ, 0 < y → y < total_fence_length / 2 →
  pen_area y ≤ pen_area (total_fence_length / 4) :=
sorry

theorem optimal_parallel_side :
  optimal_parallel_side_length = total_fence_length / 2 :=
sorry

end optimal_pen_area_optimal_parallel_side_l3769_376994


namespace quadratic_equation_root_l3769_376912

theorem quadratic_equation_root (b : ℝ) : 
  (2 : ℝ) ^ 2 * 2 + b * 2 - 4 = 0 → b = -2 := by
  sorry

end quadratic_equation_root_l3769_376912


namespace min_distance_sum_l3769_376980

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

/-- The left focus of the hyperbola -/
def F : ℝ × ℝ := sorry

/-- Point A -/
def A : ℝ × ℝ := (1, 4)

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- A point is on the right branch of the hyperbola -/
def on_right_branch (p : ℝ × ℝ) : Prop :=
  hyperbola p.1 p.2 ∧ p.1 > 0

theorem min_distance_sum :
  ∀ P : ℝ × ℝ, on_right_branch P →
    distance P F + distance P A ≥ 9 ∧
    ∃ Q : ℝ × ℝ, on_right_branch Q ∧ distance Q F + distance Q A = 9 :=
sorry

end min_distance_sum_l3769_376980


namespace lesser_fraction_l3769_376936

theorem lesser_fraction (x y : ℝ) (h_sum : x + y = 13/14) (h_prod : x * y = 1/8) :
  min x y = (13 - Real.sqrt 57) / 28 := by sorry

end lesser_fraction_l3769_376936


namespace min_value_theorem_l3769_376916

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 26 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ 2 / a₀ + 3 / b₀ = 26 := by
  sorry

end min_value_theorem_l3769_376916


namespace max_product_constraint_l3769_376984

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 6 * a + 8 * b = 48) :
  a * b ≤ 12 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 6 * a₀ + 8 * b₀ = 48 ∧ a₀ * b₀ = 12 := by
  sorry

end max_product_constraint_l3769_376984


namespace circle_equation_l3769_376985

/-- The equation of a circle with center (1, 2) passing through the origin (0, 0) -/
theorem circle_equation : ∀ x y : ℝ, 
  (x - 1)^2 + (y - 2)^2 = 5 ↔ 
  (x - 1)^2 + (y - 2)^2 = (0 - 1)^2 + (0 - 2)^2 := by sorry

end circle_equation_l3769_376985


namespace geometric_series_ratio_l3769_376914

/-- Given a geometric series with first term a and common ratio r -/
def geometric_series (a r : ℝ) : ℕ → ℝ := fun n => a * r^n

/-- Sum of the geometric series up to infinity -/
def series_sum (a r : ℝ) : ℝ := 24

/-- Sum of terms with odd powers of r -/
def odd_powers_sum (a r : ℝ) : ℝ := 9

/-- Theorem: If the sum of a geometric series is 24 and the sum of terms with odd powers of r is 9, then r = 3/5 -/
theorem geometric_series_ratio (a r : ℝ) (h1 : series_sum a r = 24) (h2 : odd_powers_sum a r = 9) :
  r = 3/5 := by sorry

end geometric_series_ratio_l3769_376914


namespace quadratic_inequality_solution_l3769_376974

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) → 
  a * b = 6 := by
  sorry

end quadratic_inequality_solution_l3769_376974


namespace reflect_P_across_y_axis_l3769_376976

/-- Reflects a point across the y-axis in a 2D Cartesian coordinate system -/
def reflect_across_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-2, 3)

/-- Theorem stating that reflecting P(-2,3) across the y-axis results in (2,3) -/
theorem reflect_P_across_y_axis :
  reflect_across_y_axis P = (2, 3) := by
  sorry

end reflect_P_across_y_axis_l3769_376976


namespace sum_of_squares_geq_twice_product_l3769_376917

theorem sum_of_squares_geq_twice_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by
  sorry

end sum_of_squares_geq_twice_product_l3769_376917


namespace animals_food_consumption_l3769_376998

/-- The total food consumption for a group of animals in one month -/
def total_food_consumption (num_animals : ℕ) (food_per_animal : ℕ) : ℕ :=
  num_animals * food_per_animal

/-- Theorem: Given 6 animals, with each animal eating 4 kg of food in one month,
    the total food consumption for all animals in one month is 24 kg -/
theorem animals_food_consumption :
  total_food_consumption 6 4 = 24 := by
  sorry

end animals_food_consumption_l3769_376998


namespace intersection_of_M_and_N_l3769_376964

open Set

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def N : Set ℝ := {x | Real.log x ≥ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 1 ≤ x ∧ x ≤ 4} := by
  sorry

end intersection_of_M_and_N_l3769_376964


namespace min_value_xy_l3769_376986

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + 4 * y + 5) :
  x * y ≥ 25 := by
  sorry

end min_value_xy_l3769_376986


namespace problem_solution_l3769_376957

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 10) 
  (h3 : x = 1) : 
  y = 13 := by
  sorry

end problem_solution_l3769_376957


namespace expression_evaluation_l3769_376997

-- Define the expression
def expression : ℕ → ℕ → ℕ := λ a b => (3^a + 7^b)^2 - (3^a - 7^b)^2

-- State the theorem
theorem expression_evaluation :
  expression 1003 1004 = 5292 * 441^500 :=
by sorry

end expression_evaluation_l3769_376997


namespace function_inequality_implies_range_l3769_376908

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem function_inequality_implies_range (f : ℝ → ℝ) (a : ℝ) :
  decreasing_function f →
  (∀ x, x > 0 → f x ≠ 0) →
  f (2 * a^2 + a + 1) < f (3 * a^2 - 4 * a + 1) →
  (0 < a ∧ a < 1/3) ∨ (1 < a ∧ a < 5) :=
by sorry

end function_inequality_implies_range_l3769_376908


namespace more_squirrels_than_nuts_l3769_376978

def num_squirrels : ℕ := 4
def num_nuts : ℕ := 2

theorem more_squirrels_than_nuts : num_squirrels - num_nuts = 2 := by
  sorry

end more_squirrels_than_nuts_l3769_376978


namespace car_average_speed_l3769_376987

theorem car_average_speed 
  (total_time : ℝ) 
  (first_interval : ℝ) 
  (first_speed : ℝ) 
  (second_speed : ℝ) 
  (h1 : total_time = 8) 
  (h2 : first_interval = 4) 
  (h3 : first_speed = 70) 
  (h4 : second_speed = 60) : 
  (first_speed * first_interval + second_speed * (total_time - first_interval)) / total_time = 65 := by
  sorry

end car_average_speed_l3769_376987


namespace quadratic_inequality_solution_l3769_376928

theorem quadratic_inequality_solution (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 →
  (∀ x, x^2 - 2*a*x - 8*a^2 < 0 ↔ x₁ < x ∧ x < x₂) →
  x₂ - x₁ = 15 →
  a = 5/2 := by
  sorry

end quadratic_inequality_solution_l3769_376928


namespace probability_all_yellow_apples_l3769_376939

def total_apples : ℕ := 8
def yellow_apples : ℕ := 3
def red_apples : ℕ := 5
def apples_chosen : ℕ := 3

theorem probability_all_yellow_apples :
  (Nat.choose yellow_apples apples_chosen) / (Nat.choose total_apples apples_chosen) = 1 / 56 :=
by sorry

end probability_all_yellow_apples_l3769_376939


namespace sampling_method_l3769_376920

/-- Represents a bag of milk powder with a three-digit number -/
def BagNumber := Fin 800

/-- The random number table row -/
def RandomRow : List ℕ := [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79]

/-- Selects valid numbers from the random row -/
def selectValidNumbers (row : List ℕ) : List BagNumber := sorry

/-- The sampling method -/
theorem sampling_method (randomRow : List ℕ) :
  randomRow = RandomRow →
  (selectValidNumbers randomRow).take 5 = [⟨785, sorry⟩, ⟨567, sorry⟩, ⟨199, sorry⟩, ⟨507, sorry⟩, ⟨175, sorry⟩] := by
  sorry

end sampling_method_l3769_376920


namespace golden_state_team_total_points_l3769_376927

def golden_state_team_points : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun draymond curry kelly durant klay =>
    draymond = 12 ∧
    curry = 2 * draymond ∧
    kelly = 9 ∧
    durant = 2 * kelly ∧
    klay = draymond / 2 ∧
    draymond + curry + kelly + durant + klay = 69

theorem golden_state_team_total_points :
  ∃ (draymond curry kelly durant klay : ℕ),
    golden_state_team_points draymond curry kelly durant klay :=
by
  sorry

end golden_state_team_total_points_l3769_376927


namespace sarah_stamp_collection_value_l3769_376934

/-- Calculates the total value of a stamp collection given the following conditions:
    - The total number of stamps in the collection
    - The number of stamps in a subset
    - The total value of the subset
    Assuming the price per stamp is constant. -/
def stamp_collection_value (total_stamps : ℕ) (subset_stamps : ℕ) (subset_value : ℚ) : ℚ :=
  (total_stamps : ℚ) * (subset_value / subset_stamps)

/-- Theorem stating that a collection of 20 stamps, where 4 stamps are worth $10,
    has a total value of $50. -/
theorem sarah_stamp_collection_value :
  stamp_collection_value 20 4 10 = 50 := by
  sorry

end sarah_stamp_collection_value_l3769_376934


namespace henry_margo_meeting_l3769_376949

/-- The time it takes Henry to complete one lap around the track -/
def henry_lap_time : ℕ := 7

/-- The time it takes Margo to complete one lap around the track -/
def margo_lap_time : ℕ := 12

/-- The time when Henry and Margo meet at the starting line -/
def meeting_time : ℕ := 84

theorem henry_margo_meeting :
  Nat.lcm henry_lap_time margo_lap_time = meeting_time := by
  sorry

end henry_margo_meeting_l3769_376949


namespace geometric_sequence_common_ratio_l3769_376903

theorem geometric_sequence_common_ratio 
  (b₁ : ℕ+) 
  (q : ℕ+) 
  (seq : ℕ → ℕ+) 
  (h_geometric : ∀ n, seq n = b₁ * q ^ (n - 1)) 
  (h_sum : seq 3 + seq 5 + seq 7 = 819 * 6^2016) :
  q = 1 ∨ q = 2 ∨ q = 3 ∨ q = 4 := by
sorry

end geometric_sequence_common_ratio_l3769_376903


namespace z_sixth_power_l3769_376925

theorem z_sixth_power (z : ℂ) : 
  z = (Real.sqrt 3 + Complex.I) / 2 → 
  z^6 = (1 + Real.sqrt 3) / 4 - ((Real.sqrt 3 + 1) / 8) * Complex.I :=
by sorry

end z_sixth_power_l3769_376925


namespace vlad_sister_height_difference_l3769_376992

/-- Converts feet and inches to total inches -/
def heightToInches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- Represents the height difference between two people -/
def heightDifference (person1_feet : ℕ) (person1_inches : ℕ) (person2_feet : ℕ) (person2_inches : ℕ) : ℕ :=
  heightToInches person1_feet person1_inches - heightToInches person2_feet person2_inches

theorem vlad_sister_height_difference :
  heightDifference 6 3 2 10 = 41 := by sorry

end vlad_sister_height_difference_l3769_376992


namespace arrangements_count_l3769_376932

/-- Represents the number of acts in the show -/
def total_acts : ℕ := 6

/-- Represents the possible positions for Act A -/
def act_a_positions : Finset ℕ := {1, 2}

/-- Represents the possible positions for Act B -/
def act_b_positions : Finset ℕ := {2, 3, 4, 5}

/-- Represents the position of Act C -/
def act_c_position : ℕ := total_acts

/-- A function that calculates the number of arrangements -/
def count_arrangements : ℕ := sorry

/-- The theorem stating that the number of arrangements is 42 -/
theorem arrangements_count : count_arrangements = 42 := by sorry

end arrangements_count_l3769_376932


namespace absolute_value_equation_l3769_376989

theorem absolute_value_equation (x : ℝ) : |x - 1| = 2*x → x = 1/3 := by
  sorry

end absolute_value_equation_l3769_376989


namespace rahul_work_days_l3769_376929

/-- The number of days it takes Rajesh to complete the work -/
def rajesh_days : ℝ := 2

/-- The total payment for the work -/
def total_payment : ℝ := 355

/-- Rahul's share of the payment -/
def rahul_share : ℝ := 142

/-- The number of days it takes Rahul to complete the work -/
def rahul_days : ℝ := 3

theorem rahul_work_days :
  rajesh_days = 2 ∧
  total_payment = 355 ∧
  rahul_share = 142 →
  rahul_days = 3 := by
  sorry

end rahul_work_days_l3769_376929


namespace minimum_value_of_expression_l3769_376977

theorem minimum_value_of_expression (x : ℝ) (h : x > 0) :
  x + 4 / x ≥ 4 ∧ (x + 4 / x = 4 ↔ x = 2) := by
  sorry

end minimum_value_of_expression_l3769_376977


namespace solution_set_when_a_is_3_range_of_a_when_sum_geq_5_l3769_376935

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 3|

-- Part 1
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≤ 6} = {x : ℝ | 0 ≤ x ∧ x ≤ 3} :=
by sorry

-- Part 2
theorem range_of_a_when_sum_geq_5 :
  ∀ x : ℝ, f a x + g x ≥ 5 → a ≥ 4 :=
by sorry

end solution_set_when_a_is_3_range_of_a_when_sum_geq_5_l3769_376935


namespace equation_unique_solution_l3769_376961

theorem equation_unique_solution :
  ∃! x : ℝ, (3 * x^2 - 18 * x) / (x^2 - 6 * x) = x^2 - 4 * x + 3 ∧ x ≠ 0 ∧ x ≠ 6 := by
  sorry

end equation_unique_solution_l3769_376961


namespace knight_moves_equality_on_7x7_l3769_376999

/-- Represents a position on a chessboard --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a knight's move on a chessboard --/
inductive KnightMove : Position → Position → Prop
  | move_1 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x + 1, y + 2⟩
  | move_2 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x + 2, y + 1⟩
  | move_3 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x + 2, y - 1⟩
  | move_4 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x + 1, y - 2⟩
  | move_5 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x - 1, y - 2⟩
  | move_6 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x - 2, y - 1⟩
  | move_7 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x - 2, y + 1⟩
  | move_8 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x - 1, y + 2⟩

/-- Minimum number of moves for a knight to reach a target position from a start position --/
def minMoves (start target : Position) : Nat :=
  sorry

theorem knight_moves_equality_on_7x7 :
  let start := ⟨0, 0⟩
  let topRight := ⟨6, 6⟩
  let bottomRight := ⟨6, 0⟩
  minMoves start topRight = minMoves start bottomRight :=
by sorry

end knight_moves_equality_on_7x7_l3769_376999


namespace triangle_area_triangle_area_proof_l3769_376948

/-- The area of a triangle with side lengths 9, 40, and 41 is 180 -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    let a : ℝ := 9
    let b : ℝ := 40
    let c : ℝ := 41
    (a^2 + b^2 = c^2) ∧ (area = (1/2) * a * b) ∧ (area = 180)

/-- Proof of the theorem -/
theorem triangle_area_proof : ∃ (area : ℝ), triangle_area area := by
  sorry

end triangle_area_triangle_area_proof_l3769_376948


namespace magnitude_of_z_plus_two_l3769_376911

/-- Given a complex number z = (1+i)/i, prove that the magnitude of z+2 is √10 -/
theorem magnitude_of_z_plus_two (z : ℂ) : z = (1 + Complex.I) / Complex.I → Complex.abs (z + 2) = Real.sqrt 10 := by
  sorry

end magnitude_of_z_plus_two_l3769_376911


namespace quadratic_solution_l3769_376990

theorem quadratic_solution (a b : ℝ) : 
  (1 : ℝ)^2 * a - (1 : ℝ) * b - 5 = 0 → 2023 + a - b = 2028 := by
  sorry

end quadratic_solution_l3769_376990


namespace arithmetic_sequence_common_difference_l3769_376962

/-- Given an arithmetic sequence {a_n} where a_1 = 13 and a_4 = 1,
    prove that the common difference d is -4. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)  -- The sequence a_n
  (h1 : a 1 = 13)  -- a_1 = 13
  (h4 : a 4 = 1)   -- a_4 = 1
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- Definition of arithmetic sequence
  : a 2 - a 1 = -4 :=
by sorry

end arithmetic_sequence_common_difference_l3769_376962


namespace find_C_l3769_376959

theorem find_C (A B C : ℝ) 
  (h_diff1 : A ≠ B) (h_diff2 : A ≠ C) (h_diff3 : B ≠ C)
  (h1 : 3 * A - A = 10)
  (h2 : B + A = 12)
  (h3 : C - B = 6) : 
  C = 13 := by
sorry

end find_C_l3769_376959


namespace binomial_coefficient_problem_l3769_376958

theorem binomial_coefficient_problem (m : ℤ) : 
  (Nat.choose 4 2 : ℤ) * m^2 = (Nat.choose 4 3 : ℤ) * m + 16 → m = 2 :=
by
  sorry

end binomial_coefficient_problem_l3769_376958


namespace certain_number_is_88_l3769_376906

theorem certain_number_is_88 (x : ℝ) (y : ℝ) : 
  x = y + 0.25 * y → x = 110 → y = 88 := by
sorry

end certain_number_is_88_l3769_376906


namespace sum_external_angles_hexagon_l3769_376968

/-- A regular hexagon is a polygon with 6 sides of equal length and 6 equal angles -/
def RegularHexagon : Type := Unit

/-- The external angle of a polygon is the angle between one side and the extension of an adjacent side -/
def ExternalAngle (p : RegularHexagon) : ℝ := sorry

/-- The sum of external angles of a regular hexagon -/
def SumExternalAngles (p : RegularHexagon) : ℝ := sorry

/-- Theorem: The sum of the external angles of a regular hexagon is 360° -/
theorem sum_external_angles_hexagon (p : RegularHexagon) :
  SumExternalAngles p = 360 := by sorry

end sum_external_angles_hexagon_l3769_376968


namespace milk_for_cookies_l3769_376941

/-- Given the ratio of milk to cookies and the conversion between quarts and pints,
    calculate the amount of milk needed for a different number of cookies. -/
theorem milk_for_cookies (cookies_base : ℕ) (quarts_base : ℕ) (cookies_target : ℕ) :
  cookies_base > 0 →
  quarts_base > 0 →
  cookies_target > 0 →
  (cookies_base = 18 ∧ quarts_base = 3 ∧ cookies_target = 15) →
  (∃ (pints_target : ℚ), pints_target = 5 ∧
    pints_target = (quarts_base * 2 : ℚ) * cookies_target / cookies_base) :=
by
  sorry

#check milk_for_cookies

end milk_for_cookies_l3769_376941


namespace mod_equivalence_l3769_376947

theorem mod_equivalence (m : ℕ) : 
  198 * 864 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 22 := by
  sorry

end mod_equivalence_l3769_376947


namespace jordan_machine_l3769_376995

theorem jordan_machine (x : ℚ) : ((3 * x - 6) / 2 + 9 = 27) → x = 14 := by
  sorry

end jordan_machine_l3769_376995


namespace earliest_meeting_time_is_440_l3769_376996

/-- Represents the lap time in minutes for each runner -/
structure LapTime where
  charlie : ℕ
  ben : ℕ
  laura : ℕ

/-- Calculates the earliest meeting time in minutes -/
def earliest_meeting_time (lt : LapTime) : ℕ :=
  Nat.lcm (Nat.lcm lt.charlie lt.ben) lt.laura

/-- Theorem: Given the specific lap times, the earliest meeting time is 440 minutes -/
theorem earliest_meeting_time_is_440 :
  earliest_meeting_time ⟨5, 8, 11⟩ = 440 := by
  sorry

end earliest_meeting_time_is_440_l3769_376996
