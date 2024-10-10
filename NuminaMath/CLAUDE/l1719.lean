import Mathlib

namespace P_greater_than_Q_l1719_171942

theorem P_greater_than_Q : ∀ x : ℝ, (x^2 + 2) > (2*x) := by
  sorry

end P_greater_than_Q_l1719_171942


namespace smallest_with_12_divisors_l1719_171987

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is the smallest positive integer satisfying property P -/
def is_smallest_satisfying (n : ℕ) (P : ℕ → Prop) : Prop :=
  P n ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬P m

theorem smallest_with_12_divisors :
  is_smallest_satisfying 288 (λ n => num_divisors n = 12) := by sorry

end smallest_with_12_divisors_l1719_171987


namespace remainder_three_power_twentyfour_mod_seven_l1719_171902

theorem remainder_three_power_twentyfour_mod_seven :
  3^24 % 7 = 1 := by
sorry

end remainder_three_power_twentyfour_mod_seven_l1719_171902


namespace class_average_mark_l1719_171978

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_average : ℝ) (remaining_average : ℝ) : 
  total_students = 35 →
  excluded_students = 5 →
  excluded_average = 20 →
  remaining_average = 90 →
  (total_students * (total_students * remaining_average - 
    excluded_students * excluded_average)) / 
    (total_students * (total_students - excluded_students)) = 80 := by
  sorry

end class_average_mark_l1719_171978


namespace triangle_properties_l1719_171948

-- Define the points A and B
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 3)

-- Define the properties of triangle ABC
def is_isosceles (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

def is_perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0

-- Define the possible coordinates of point C
def C1 : ℝ × ℝ := (3, -1)
def C2 : ℝ × ℝ := (-3, 7)

-- Define the equations of the median lines
def median_eq1 (x y : ℝ) : Prop := 7 * x - y + 3 = 0
def median_eq2 (x y : ℝ) : Prop := x + 7 * y - 21 = 0

-- Theorem statement
theorem triangle_properties :
  (is_isosceles A B C1 ∧ is_perpendicular A B C1 ∧
   median_eq1 ((A.1 + C1.1) / 2) ((A.2 + C1.2) / 2)) ∨
  (is_isosceles A B C2 ∧ is_perpendicular A B C2 ∧
   median_eq2 ((A.1 + C2.1) / 2) ((A.2 + C2.2) / 2)) := by sorry

end triangle_properties_l1719_171948


namespace about_set_S_l1719_171932

def S : Set ℤ := {x | ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2}

theorem about_set_S :
  (∀ x ∈ S, ¬(3 ∣ x)) ∧ (∃ x ∈ S, 11 ∣ x) := by
  sorry

end about_set_S_l1719_171932


namespace trig_identity_l1719_171922

theorem trig_identity (x : ℝ) : 
  (Real.sin x ^ 6 + Real.cos x ^ 6 - 1) ^ 3 + 27 * Real.sin x ^ 6 * Real.cos x ^ 6 = 0 := by
  sorry

end trig_identity_l1719_171922


namespace ore_without_alloy_percentage_l1719_171944

/-- Represents the composition of an ore -/
structure Ore where
  alloy_percentage : Real
  iron_in_alloy : Real
  total_ore : Real
  pure_iron : Real

/-- Theorem: The percentage of ore not containing the alloy with iron is 75% -/
theorem ore_without_alloy_percentage (ore : Ore)
  (h1 : ore.alloy_percentage = 0.25)
  (h2 : ore.iron_in_alloy = 0.90)
  (h3 : ore.total_ore = 266.6666666666667)
  (h4 : ore.pure_iron = 60) :
  1 - ore.alloy_percentage = 0.75 := by
  sorry

#check ore_without_alloy_percentage

end ore_without_alloy_percentage_l1719_171944


namespace min_sum_squared_distances_min_sum_squared_distances_achievable_l1719_171916

/-- The minimum sum of squared distances from a point on a circle to two fixed points -/
theorem min_sum_squared_distances (x y : ℝ) :
  (x - 3)^2 + (y - 4)^2 = 4 →
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 ≥ 26 := by
  sorry

/-- The minimum sum of squared distances is achievable -/
theorem min_sum_squared_distances_achievable :
  ∃ x y : ℝ, (x - 3)^2 + (y - 4)^2 = 4 ∧
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 26 := by
  sorry

end min_sum_squared_distances_min_sum_squared_distances_achievable_l1719_171916


namespace square_sum_ge_product_sum_l1719_171940

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := by
  sorry

end square_sum_ge_product_sum_l1719_171940


namespace restaurant_bill_theorem_l1719_171947

def bill_with_discount (bill : Float) (discount_rate : Float) : Float :=
  bill * (1 - discount_rate / 100)

def total_bill (bob_bill kate_bill john_bill sarah_bill : Float)
               (bob_discount kate_discount john_discount sarah_discount : Float) : Float :=
  bill_with_discount bob_bill bob_discount +
  bill_with_discount kate_bill kate_discount +
  bill_with_discount john_bill john_discount +
  bill_with_discount sarah_bill sarah_discount

theorem restaurant_bill_theorem :
  total_bill 35.50 29.75 43.20 27.35 5.75 2.35 3.95 9.45 = 128.76945 := by
  sorry

end restaurant_bill_theorem_l1719_171947


namespace sector_area_l1719_171950

theorem sector_area (circumference : Real) (central_angle : Real) :
  circumference = 8 * π / 9 + 4 →
  central_angle = 80 * π / 180 →
  (1 / 2) * (circumference - 2 * (circumference / (2 * π + central_angle))) ^ 2 * central_angle / (2 * π) = 8 * π / 9 := by
  sorry

end sector_area_l1719_171950


namespace exactly_two_successes_in_four_trials_l1719_171941

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 4

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The binomial probability mass function -/
def binomial_pmf (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p ^ k * (1 - p) ^ (n - k)

/-- The theorem to be proved -/
theorem exactly_two_successes_in_four_trials : 
  binomial_pmf n k p = 0.3456 := by sorry

end exactly_two_successes_in_four_trials_l1719_171941


namespace return_journey_speed_l1719_171991

/-- Given a round trip with the following conditions:
    - The distance between home and the retreat is 300 miles each way
    - The average speed to the retreat was 50 miles per hour
    - The round trip took 10 hours
    - The same route was taken both ways
    Prove that the average speed on the return journey is 75 mph. -/
theorem return_journey_speed (distance : ℝ) (speed_to : ℝ) (total_time : ℝ) :
  distance = 300 →
  speed_to = 50 →
  total_time = 10 →
  let time_to : ℝ := distance / speed_to
  let time_from : ℝ := total_time - time_to
  let speed_from : ℝ := distance / time_from
  speed_from = 75 := by sorry

end return_journey_speed_l1719_171991


namespace min_a_sqrt_sum_l1719_171996

theorem min_a_sqrt_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) →
  ∃ a_min : ℝ, a_min = Real.sqrt 2 ∧ ∀ a : ℝ, (∀ x y, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) → a ≥ a_min :=
by sorry

end min_a_sqrt_sum_l1719_171996


namespace complex_number_location_l1719_171921

theorem complex_number_location : ∃ (z : ℂ), 
  z = (1 : ℂ) / (2 + Complex.I) + Complex.I ^ 2018 ∧ 
  z.re < 0 ∧ z.im < 0 :=
by sorry

end complex_number_location_l1719_171921


namespace triangle_distance_set_l1719_171900

theorem triangle_distance_set (a b k : ℝ) (ha : 0 < a) (hb : 0 < b) (hk : k^2 > 2*a^2/3 + 2*b^2/3) :
  let S := {P : ℝ × ℝ | P.1^2 + P.2^2 + (P.1 - a)^2 + P.2^2 + P.1^2 + (P.2 - b)^2 < k^2}
  let C := {P : ℝ × ℝ | (P.1 - a/3)^2 + (P.2 - b/3)^2 < (k^2 - 2*a^2/3 - 2*b^2/3) / 3}
  S = C := by sorry

end triangle_distance_set_l1719_171900


namespace lucas_overall_accuracy_l1719_171938

theorem lucas_overall_accuracy 
  (emily_individual_accuracy : Real) 
  (emily_overall_accuracy : Real)
  (lucas_individual_accuracy : Real)
  (h1 : emily_individual_accuracy = 0.7)
  (h2 : emily_overall_accuracy = 0.82)
  (h3 : lucas_individual_accuracy = 0.85) :
  lucas_individual_accuracy * 0.5 + (emily_overall_accuracy - emily_individual_accuracy * 0.5) = 0.895 := by
  sorry

end lucas_overall_accuracy_l1719_171938


namespace total_ways_eq_600_l1719_171901

/-- Represents the number of cards in the left pocket -/
def left_cards : ℕ := 30

/-- Represents the number of cards in the right pocket -/
def right_cards : ℕ := 20

/-- Represents the total number of ways to select one card from each pocket -/
def total_ways : ℕ := left_cards * right_cards

/-- Theorem stating that the total number of ways to select one card from each pocket is 600 -/
theorem total_ways_eq_600 : total_ways = 600 := by sorry

end total_ways_eq_600_l1719_171901


namespace cartoon_length_missy_cartoon_length_l1719_171976

/-- The length of a cartoon given specific TV watching conditions -/
theorem cartoon_length (reality_shows : ℕ) (reality_show_length : ℕ) (total_time : ℕ) : ℕ :=
  let cartoon_length := total_time - reality_shows * reality_show_length
  by
    sorry

/-- The length of Missy's cartoon is 10 minutes -/
theorem missy_cartoon_length : cartoon_length 5 28 150 = 10 := by
  sorry

end cartoon_length_missy_cartoon_length_l1719_171976


namespace tens_place_of_first_ten_digit_number_l1719_171931

/-- Represents the sequence of grouped numbers -/
def groupedSequence : List (List Nat) := sorry

/-- The number of digits in the nth group -/
def groupDigits (n : Nat) : Nat := n

/-- The sum of digits in the first n groups -/
def sumDigitsUpTo (n : Nat) : Nat := sorry

/-- The first ten-digit number in the sequence -/
def firstTenDigitNumber : Nat := sorry

/-- Theorem: The tens place digit of the first ten-digit number is 2 -/
theorem tens_place_of_first_ten_digit_number :
  (firstTenDigitNumber / 1000000000) % 10 = 2 := by sorry

end tens_place_of_first_ten_digit_number_l1719_171931


namespace correct_calculation_l1719_171929

theorem correct_calculation (a b : ℝ) : 4 * a^2 * b - 3 * b * a^2 = a^2 * b := by
  sorry

end correct_calculation_l1719_171929


namespace log_inequality_l1719_171935

theorem log_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  let a : ℝ := (Real.sqrt 2 + 1) / 2
  let f : ℝ → ℝ := fun x ↦ Real.log x / Real.log a
  f m > f n → m > n := by sorry

end log_inequality_l1719_171935


namespace min_value_theorem_l1719_171962

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 2/y = 1 → a*(b - 1) ≤ x*(y - 1) ∧ 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 2/y = 1 ∧ x*(y - 1) = 3 + 2*Real.sqrt 2 :=
by sorry

end min_value_theorem_l1719_171962


namespace new_students_count_l1719_171913

theorem new_students_count : ∃! n : ℕ, n < 400 ∧ n % 17 = 16 ∧ n % 19 = 12 ∧ n = 288 := by
  sorry

end new_students_count_l1719_171913


namespace smallest_square_sum_of_consecutive_integers_l1719_171914

theorem smallest_square_sum_of_consecutive_integers :
  ∃ n : ℕ, 
    (n > 0) ∧ 
    (10 * (2 * n + 19) = 250) ∧ 
    (∀ m : ℕ, m > 0 → m < n → ¬∃ k : ℕ, 10 * (2 * m + 19) = k * k) := by
  sorry

end smallest_square_sum_of_consecutive_integers_l1719_171914


namespace sum_of_numbers_in_ratio_l1719_171972

theorem sum_of_numbers_in_ratio (x : ℝ) :
  x > 0 →
  x^2 + (2*x)^2 + (4*x)^2 = 1701 →
  x + 2*x + 4*x = 63 := by
sorry

end sum_of_numbers_in_ratio_l1719_171972


namespace point_on_line_l1719_171920

/-- Prove that for a point P(2, m) lying on the line 3x + y = 2, the value of m is -4. -/
theorem point_on_line (m : ℝ) : (3 * 2 + m = 2) → m = -4 := by
  sorry

end point_on_line_l1719_171920


namespace unique_solution_for_prime_equation_l1719_171908

theorem unique_solution_for_prime_equation :
  ∀ a b : ℕ,
  Prime a →
  b > 0 →
  9 * (2 * a + b)^2 = 509 * (4 * a + 511 * b) →
  a = 251 ∧ b = 7 :=
by sorry

end unique_solution_for_prime_equation_l1719_171908


namespace baseball_theorem_l1719_171969

def baseball_problem (team_scores : List Nat) (lost_games : Nat) : Prop :=
  let total_games := team_scores.length
  let won_games := total_games - lost_games
  let opponent_scores := team_scores.map (λ score =>
    if score ∈ [2, 4, 6, 8] then score + 2 else score / 3)
  
  (total_games = 8) ∧
  (team_scores = [2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (lost_games = 4) ∧
  (opponent_scores.sum = 36)

theorem baseball_theorem :
  baseball_problem [2, 3, 4, 5, 6, 7, 8, 9] 4 := by
  sorry

end baseball_theorem_l1719_171969


namespace skylar_current_age_l1719_171989

/-- Represents Skylar's donation history and age calculation -/
def skylar_age (start_age : ℕ) (annual_donation : ℕ) (total_donated : ℕ) : ℕ :=
  start_age + total_donated / annual_donation

/-- Theorem stating Skylar's current age -/
theorem skylar_current_age :
  skylar_age 13 5 105 = 34 := by
  sorry

end skylar_current_age_l1719_171989


namespace complex_equation_to_parabola_l1719_171994

/-- The set of points (x, y) satisfying the complex equation is equivalent to a parabola with two holes -/
theorem complex_equation_to_parabola (x y : ℝ) :
  (Complex.I + x^2 - 2*x + 2*y*Complex.I = 
   (y - 1 : ℂ) + ((4*y^2 - 1)/(2*y - 1) : ℝ)*Complex.I) ↔ 
  (y = (x - 1)^2 ∧ y ≠ (1/2 : ℝ)) :=
sorry

end complex_equation_to_parabola_l1719_171994


namespace students_at_start_l1719_171967

theorem students_at_start (initial_students final_students left_students new_students : ℕ) :
  final_students = 43 →
  left_students = 3 →
  new_students = 42 →
  initial_students + new_students - left_students = final_students →
  initial_students = 4 := by
sorry

end students_at_start_l1719_171967


namespace min_value_cubic_function_l1719_171917

theorem min_value_cubic_function (x : ℝ) (h : x > 0) :
  x^3 + 9*x + 81/x^4 ≥ 21 ∧ ∃ y > 0, y^3 + 9*y + 81/y^4 = 21 :=
sorry

end min_value_cubic_function_l1719_171917


namespace f_extrema_half_f_extrema_sum_gt_zero_l1719_171933

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + a * x) - 2 * x / (x + 2)

-- Theorem for part (1)
theorem f_extrema_half :
  let a : ℝ := 1/2
  ∃ (min_val : ℝ), (∀ x, x > -2 → f a x ≥ min_val) ∧
                   (∃ x, x > -2 ∧ f a x = min_val) ∧
                   min_val = Real.log 2 - 1 ∧
                   (∀ M, ∃ x, x > -2 ∧ f a x > M) :=
sorry

-- Theorem for part (2)
theorem f_extrema_sum_gt_zero (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : 1/2 < a ∧ a < 1) 
  (hx₁ : x₁ > -1/a ∧ (∀ y, y > -1/a → f a y ≤ f a x₁))
  (hx₂ : x₂ > -1/a ∧ (∀ y, y > -1/a → f a y ≤ f a x₂))
  (hd : x₁ ≠ x₂) :
  f a x₁ + f a x₂ > f a 0 :=
sorry

end f_extrema_half_f_extrema_sum_gt_zero_l1719_171933


namespace derivative_f_at_pi_div_2_l1719_171907

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.sin x

theorem derivative_f_at_pi_div_2 :
  deriv f (π / 2) = π := by sorry

end derivative_f_at_pi_div_2_l1719_171907


namespace hyperbola_equation_theorem_l1719_171968

/-- A hyperbola with vertex and center at (1, 0) and eccentricity 2 -/
structure Hyperbola where
  vertex : ℝ × ℝ
  center : ℝ × ℝ
  eccentricity : ℝ
  vertex_eq_center : vertex = center
  vertex_x : vertex.1 = 1
  vertex_y : vertex.2 = 0
  eccentricity_val : eccentricity = 2

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 - y^2/3 = 1

/-- Theorem stating that the given hyperbola has the equation x² - y²/3 = 1 -/
theorem hyperbola_equation_theorem (h : Hyperbola) :
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 - y^2/3 = 1 := by
  sorry

end hyperbola_equation_theorem_l1719_171968


namespace consecutive_integer_product_divisible_by_six_l1719_171956

theorem consecutive_integer_product_divisible_by_six (n : ℤ) : 
  ∃ k : ℤ, n * (n + 1) * (n + 2) = 6 * k := by
  sorry

#check consecutive_integer_product_divisible_by_six

end consecutive_integer_product_divisible_by_six_l1719_171956


namespace probability_green_given_no_red_l1719_171965

/-- The set of all possible colors for memories -/
inductive Color
| Red
| Green
| Blue
| Yellow
| Purple

/-- A memory coloring is a set of at most two distinct colors -/
def MemoryColoring := Finset Color

/-- The set of all valid memory colorings -/
def AllColorings : Finset MemoryColoring :=
  sorry

/-- The set of memory colorings without red -/
def ColoringsWithoutRed : Finset MemoryColoring :=
  sorry

/-- The set of memory colorings that are at least partly green and have no red -/
def GreenColoringsWithoutRed : Finset MemoryColoring :=
  sorry

/-- The probability of a memory being at least partly green given that it has no red -/
theorem probability_green_given_no_red :
  (Finset.card GreenColoringsWithoutRed) / (Finset.card ColoringsWithoutRed) = 2 / 5 :=
sorry

end probability_green_given_no_red_l1719_171965


namespace total_gold_stars_l1719_171930

def monday_stars : ℕ := 4
def tuesday_stars : ℕ := 7
def wednesday_stars : ℕ := 3
def thursday_stars : ℕ := 8
def friday_stars : ℕ := 2

theorem total_gold_stars : 
  monday_stars + tuesday_stars + wednesday_stars + thursday_stars + friday_stars = 24 := by
  sorry

end total_gold_stars_l1719_171930


namespace quadratic_root_sum_product_ratio_l1719_171997

theorem quadratic_root_sum_product_ratio : 
  ∀ x₁ x₂ : ℝ, x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → 
  (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by sorry

end quadratic_root_sum_product_ratio_l1719_171997


namespace exists_valid_triangle_l1719_171939

-- Define the necessary structures
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the given elements
variable (p q : Line)
variable (C : Point)
variable (c : ℝ)

-- Define a right triangle
structure RightTriangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (hypotenuse_length : ℝ)

-- Define the conditions for the desired triangle
def is_valid_triangle (t : RightTriangle) : Prop :=
  -- Right angle at C
  (t.A.x - t.C.x) * (t.B.x - t.C.x) + (t.A.y - t.C.y) * (t.B.y - t.C.y) = 0 ∧
  -- Vertex A on line p
  t.A.y = p.slope * t.A.x + p.intercept ∧
  -- Hypotenuse parallel to line q
  (t.A.y - t.C.y) / (t.A.x - t.C.x) = q.slope ∧
  -- Hypotenuse length is c
  t.hypotenuse_length = c ∧
  -- C is the given point
  t.C = C

-- Theorem statement
theorem exists_valid_triangle :
  ∃ (t : RightTriangle), is_valid_triangle p q C c t :=
sorry

end exists_valid_triangle_l1719_171939


namespace worker_r_earnings_l1719_171919

/-- Given the daily earnings of three workers p, q, and r, prove that r earns 50 per day. -/
theorem worker_r_earnings
  (p q r : ℚ)  -- Daily earnings of workers p, q, and r
  (h1 : 9 * (p + q + r) = 1800)  -- p, q, and r together earn 1800 in 9 days
  (h2 : 5 * (p + r) = 600)  -- p and r can earn 600 in 5 days
  (h3 : 7 * (q + r) = 910)  -- q and r can earn 910 in 7 days
  : r = 50 := by
  sorry


end worker_r_earnings_l1719_171919


namespace parallel_planes_from_common_perpendicular_l1719_171904

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_from_common_perpendicular 
  (m : Line) (α β : Plane) (h_diff : α ≠ β) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end parallel_planes_from_common_perpendicular_l1719_171904


namespace inequality_solution_set_l1719_171984

theorem inequality_solution_set (x : ℝ) :
  x ≠ 1 →
  ((x^2 + x - 6) / (x - 1) ≤ 0 ↔ x ∈ Set.Iic (-3) ∪ Set.Ioo 1 2) :=
by sorry

end inequality_solution_set_l1719_171984


namespace circus_ticket_cost_l1719_171936

/-- The cost of tickets at a circus --/
theorem circus_ticket_cost (cost_per_ticket : ℕ) (num_tickets : ℕ) (total_cost : ℕ) : 
  cost_per_ticket = 44 → num_tickets = 7 → total_cost = cost_per_ticket * num_tickets → total_cost = 308 := by
  sorry

end circus_ticket_cost_l1719_171936


namespace tangent_circles_area_ratio_l1719_171928

/-- Regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- Circle tangent to three sides of a regular hexagon -/
structure TangentCircle (h : RegularHexagon) :=
  (radius : ℝ)
  (tangent_to_parallel_sides : True)
  (tangent_to_other_side : True)

/-- The ratio of areas of two tangent circles to a regular hexagon is 1 -/
theorem tangent_circles_area_ratio (h : RegularHexagon) 
  (c1 c2 : TangentCircle h) : 
  (c1.radius^2) / (c2.radius^2) = 1 := by sorry

end tangent_circles_area_ratio_l1719_171928


namespace event_C_is_certain_l1719_171960

-- Define an enumeration for the events
inductive Event
  | A -- It will rain after thunder
  | B -- Tomorrow will be sunny
  | C -- 1 hour equals 60 minutes
  | D -- There will be a rainbow after the rain

-- Define a function to check if an event is certain
def isCertain (e : Event) : Prop :=
  match e with
  | Event.C => True
  | _ => False

-- Theorem stating that Event C is certain
theorem event_C_is_certain : isCertain Event.C := by
  sorry

end event_C_is_certain_l1719_171960


namespace stratified_sampling_theorem_l1719_171903

/-- Represents the population sizes for each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Represents the sample sizes for each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Checks if the sample is proportional to the population -/
def isProportionalSample (pop : Population) (sam : Sample) (totalSample : ℕ) : Prop :=
  sam.elderly * (pop.elderly + pop.middleAged + pop.young) = pop.elderly * totalSample ∧
  sam.middleAged * (pop.elderly + pop.middleAged + pop.young) = pop.middleAged * totalSample ∧
  sam.young * (pop.elderly + pop.middleAged + pop.young) = pop.young * totalSample

theorem stratified_sampling_theorem (pop : Population) (sam : Sample) :
  pop.elderly = 27 →
  pop.middleAged = 54 →
  pop.young = 81 →
  sam.elderly + sam.middleAged + sam.young = 36 →
  isProportionalSample pop sam 36 →
  sam.elderly = 6 ∧ sam.middleAged = 12 ∧ sam.young = 18 := by
  sorry

#check stratified_sampling_theorem

end stratified_sampling_theorem_l1719_171903


namespace rational_solutions_quadratic_l1719_171980

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + 3 * k = 0) ↔ k = 6 :=
sorry

end rational_solutions_quadratic_l1719_171980


namespace isosceles_triangle_perimeter_l1719_171985

/-- An isosceles triangle with two sides of length 3 and one side of length 1 -/
structure IsoscelesTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (base : ℝ)
  (isIsosceles : side1 = side2)
  (side1_eq_3 : side1 = 3)
  (base_eq_1 : base = 1)

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.side1 + t.side2 + t.base

/-- Theorem: The perimeter of the specified isosceles triangle is 7 -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, perimeter t = 7 := by
  sorry


end isosceles_triangle_perimeter_l1719_171985


namespace expression_simplification_and_evaluation_l1719_171946

theorem expression_simplification_and_evaluation (a : ℚ) (h : a = 1/2) :
  (a - 1) / (a - 2) * ((a^2 - 4) / (a^2 - 2*a + 1)) - 2 / (a - 1) = -1 := by
  sorry

end expression_simplification_and_evaluation_l1719_171946


namespace per_minute_charge_plan_a_l1719_171979

/-- Represents the per-minute charge after the first 4 minutes under plan A -/
def x : ℝ := sorry

/-- The cost of an 18-minute call under plan A -/
def cost_plan_a : ℝ := 0.60 + 14 * x

/-- The cost of an 18-minute call under plan B -/
def cost_plan_b : ℝ := 0.08 * 18

/-- Theorem stating that the per-minute charge after the first 4 minutes under plan A is $0.06 -/
theorem per_minute_charge_plan_a : x = 0.06 := by
  have h1 : cost_plan_a = cost_plan_b := by sorry
  -- The proof goes here
  sorry

end per_minute_charge_plan_a_l1719_171979


namespace coat_price_l1719_171971

/-- The original price of a coat given a specific price reduction and percentage decrease. -/
theorem coat_price (price_reduction : ℝ) (percent_decrease : ℝ) (original_price : ℝ) : 
  price_reduction = 300 ∧ 
  percent_decrease = 0.60 ∧ 
  price_reduction = percent_decrease * original_price → 
  original_price = 500 := by
sorry

end coat_price_l1719_171971


namespace crayons_left_l1719_171981

theorem crayons_left (initial : ℕ) (given_away : ℕ) (lost : ℕ) : 
  initial = 1453 → given_away = 563 → lost = 558 → 
  initial - given_away - lost = 332 := by
sorry

end crayons_left_l1719_171981


namespace total_soaking_time_l1719_171961

/-- Calculates the total soaking time for clothes with grass and marinara stains. -/
theorem total_soaking_time
  (grass_stain_time : ℕ)
  (marinara_stain_time : ℕ)
  (grass_stains : ℕ)
  (marinara_stains : ℕ)
  (h1 : grass_stain_time = 4)
  (h2 : marinara_stain_time = 7)
  (h3 : grass_stains = 3)
  (h4 : marinara_stains = 1) :
  grass_stain_time * grass_stains + marinara_stain_time * marinara_stains = 19 :=
by sorry

#check total_soaking_time

end total_soaking_time_l1719_171961


namespace point_coordinates_l1719_171925

/-- A point in the second quadrant with a specific distance from the x-axis -/
def SecondQuadrantPoint (m : ℝ) : Prop :=
  m - 3 < 0 ∧ m + 2 > 0 ∧ |m + 2| = 4

/-- The theorem stating that a point with the given properties has coordinates (-1, 4) -/
theorem point_coordinates (m : ℝ) (h : SecondQuadrantPoint m) : 
  (m - 3 = -1) ∧ (m + 2 = 4) :=
sorry

end point_coordinates_l1719_171925


namespace quadratic_inequality_solution_l1719_171905

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 9*x + 20 < 1 ↔ (9 - Real.sqrt 5) / 2 < x ∧ x < (9 + Real.sqrt 5) / 2 := by
  sorry

end quadratic_inequality_solution_l1719_171905


namespace original_hourly_wage_l1719_171957

/-- Given a worker's daily wage, increased wage, bonus, total new wage, and hours worked per day,
    calculate the original hourly wage. -/
theorem original_hourly_wage (W : ℝ) (h1 : 1.60 * W + 10 = 45) (h2 : 8 > 0) :
  W / 8 = (45 - 10) / (1.60 * 8) := by sorry

end original_hourly_wage_l1719_171957


namespace circle_area_sum_l1719_171999

/-- The sum of the areas of an infinite sequence of circles, where the radius of the first circle
    is 1 and each subsequent circle's radius is 2/3 of the previous one, is equal to 9π/5. -/
theorem circle_area_sum : 
  let radius : ℕ → ℝ := λ n => (2/3)^(n-1)
  let area : ℕ → ℝ := λ n => π * (radius n)^2
  (∑' n, area n) = 9*π/5 := by sorry

end circle_area_sum_l1719_171999


namespace ellipse_foci_l1719_171909

/-- Represents an ellipse with equation x²/a² + y²/b² = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The foci of an ellipse -/
structure Foci where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- Theorem: The foci of the ellipse x²/1 + y²/10 = 1 are (0, -3) and (0, 3) -/
theorem ellipse_foci (e : Ellipse) (h₁ : e.a = 1) (h₂ : e.b = 10) :
  ∃ f : Foci, f.x₁ = 0 ∧ f.y₁ = -3 ∧ f.x₂ = 0 ∧ f.y₂ = 3 := by
  sorry

end ellipse_foci_l1719_171909


namespace quadratic_roots_difference_l1719_171998

theorem quadratic_roots_difference (P : ℝ) : 
  (∃ α β : ℝ, α^2 - 2*α - P = 0 ∧ β^2 - 2*β - P = 0 ∧ α - β = 12) → P = 35 := by
  sorry

end quadratic_roots_difference_l1719_171998


namespace smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l1719_171918

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 24 ∣ n^2 ∧ 450 ∣ n^3 → n ≥ 60 :=
by sorry

theorem sixty_satisfies : 24 ∣ 60^2 ∧ 450 ∣ 60^3 :=
by sorry

theorem smallest_n_is_sixty : ∃! n : ℕ, n > 0 ∧ 24 ∣ n^2 ∧ 450 ∣ n^3 ∧ ∀ m : ℕ, (m > 0 ∧ 24 ∣ m^2 ∧ 450 ∣ m^3) → m ≥ n :=
by sorry

end smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l1719_171918


namespace eleventh_term_is_110_div_7_l1719_171927

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of the first six terms is 30
  sum_first_six : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 30
  -- Seventh term is 10
  seventh_term : a + 6*d = 10

/-- The eleventh term of the specific arithmetic sequence is 110/7 -/
theorem eleventh_term_is_110_div_7 (seq : ArithmeticSequence) :
  seq.a + 10*seq.d = 110/7 := by
  sorry

end eleventh_term_is_110_div_7_l1719_171927


namespace book_cost_price_l1719_171970

/-- The cost price of a book given specific pricing conditions -/
theorem book_cost_price (marked_price selling_price cost_price : ℝ) :
  selling_price = 1.25 * cost_price →
  0.95 * marked_price = selling_price →
  selling_price = 62.5 →
  cost_price = 50 := by
sorry

end book_cost_price_l1719_171970


namespace denominator_one_root_l1719_171949

theorem denominator_one_root (k : ℝ) : 
  (∃! x : ℝ, -2 * x^2 + 8 * x + k = 0) ↔ k = -8 := by sorry

end denominator_one_root_l1719_171949


namespace sales_tax_theorem_l1719_171958

/-- Calculates the sales tax paid given total purchase, tax rate, and cost of tax-free items -/
def calculate_sales_tax (total_purchase : ℝ) (tax_rate : ℝ) (tax_free_cost : ℝ) : ℝ :=
  let taxable_cost := total_purchase - tax_free_cost
  tax_rate * taxable_cost

/-- Theorem stating that under the given conditions, the sales tax paid is 0.3 -/
theorem sales_tax_theorem (total_purchase tax_rate tax_free_cost : ℝ) 
  (h1 : total_purchase = 25)
  (h2 : tax_rate = 0.06)
  (h3 : tax_free_cost = 19.7) :
  calculate_sales_tax total_purchase tax_rate tax_free_cost = 0.3 := by
  sorry

#eval calculate_sales_tax 25 0.06 19.7

end sales_tax_theorem_l1719_171958


namespace ticket_price_calculation_l1719_171966

def commission_rate : ℝ := 0.12
def desired_net_amount : ℝ := 22

theorem ticket_price_calculation :
  ∃ (price : ℝ), price * (1 - commission_rate) = desired_net_amount ∧ price = 25 := by
  sorry

end ticket_price_calculation_l1719_171966


namespace unknown_number_solution_l1719_171955

theorem unknown_number_solution (x : ℝ) : 
  4.7 * 13.26 + 4.7 * 9.43 + 4.7 * x = 470 ↔ x = 77.31 := by
sorry

end unknown_number_solution_l1719_171955


namespace arrangements_six_people_one_restricted_l1719_171983

def number_of_arrangements (n : ℕ) : ℕ :=
  (n - 1) * (Nat.factorial (n - 1))

theorem arrangements_six_people_one_restricted :
  number_of_arrangements 6 = 600 := by
  sorry

end arrangements_six_people_one_restricted_l1719_171983


namespace unique_prime_303509_l1719_171973

theorem unique_prime_303509 :
  ∃! (B : ℕ), B < 10 ∧ Nat.Prime (303500 + B) :=
by
  -- The proof would go here
  sorry

end unique_prime_303509_l1719_171973


namespace divisible_by_seven_l1719_171926

def repeated_digit (d : Nat) (n : Nat) : Nat :=
  d * (10^n - 1) / 9

theorem divisible_by_seven : ∃ k : Nat,
  (repeated_digit 8 50 * 10 + 5) * 10^50 + repeated_digit 9 50 = 7 * k := by
  sorry

end divisible_by_seven_l1719_171926


namespace meal_cost_calculation_l1719_171964

theorem meal_cost_calculation (adults children : ℕ) (total_bill : ℚ) :
  adults = 2 →
  children = 5 →
  total_bill = 21 →
  ∃ (meal_cost : ℚ), meal_cost * (adults + children) = total_bill ∧ meal_cost = 3 :=
by sorry

end meal_cost_calculation_l1719_171964


namespace equation_solution_l1719_171986

theorem equation_solution : ∃! x : ℝ, x + Real.sqrt (3 * x - 2) = 6 ∧ x = (15 - Real.sqrt 73) / 2 := by
  sorry

end equation_solution_l1719_171986


namespace set_conditions_l1719_171975

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem set_conditions (m : ℝ) :
  (B m = ∅ ↔ m < 2) ∧
  (A ∩ B m = ∅ ↔ m > 4 ∨ m < 2) := by
  sorry

end set_conditions_l1719_171975


namespace equation_solution_l1719_171982

theorem equation_solution : 
  ∃! x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x * (4 : ℝ)^(3*x) = (64 : ℝ)^(4*x) ∧ x = 0 := by
  sorry

end equation_solution_l1719_171982


namespace largest_perfect_square_factor_34020_l1719_171934

def largest_perfect_square_factor (n : ℕ) : ℕ := 
  sorry

theorem largest_perfect_square_factor_34020 :
  largest_perfect_square_factor 34020 = 324 := by
  sorry

end largest_perfect_square_factor_34020_l1719_171934


namespace impossible_perpendicular_intersection_l1719_171990

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (coincident : Line → Line → Prop)

-- Define the planes and lines
variable (α : Plane)
variable (a b : Line)

-- State the theorem
theorem impossible_perpendicular_intersection 
  (h1 : ¬ coincident a b)
  (h2 : perpendicular a α)
  (h3 : intersect a b) :
  ¬ (perpendicular b α) :=
sorry

end impossible_perpendicular_intersection_l1719_171990


namespace incentive_savings_l1719_171992

/-- Calculates the amount saved given an initial amount and spending percentages -/
def calculate_savings (initial_amount : ℝ) (food_percent : ℝ) (clothes_percent : ℝ) 
  (household_percent : ℝ) (savings_percent : ℝ) : ℝ :=
  let remaining_after_food := initial_amount * (1 - food_percent)
  let remaining_after_clothes := remaining_after_food * (1 - clothes_percent)
  let remaining_after_household := remaining_after_clothes * (1 - household_percent)
  remaining_after_household * savings_percent

/-- Theorem stating that given the specified spending pattern, 
    the amount saved from a $600 incentive is $171.36 -/
theorem incentive_savings : 
  calculate_savings 600 0.3 0.2 0.15 0.6 = 171.36 := by
  sorry

end incentive_savings_l1719_171992


namespace vector_magnitude_problem_l1719_171910

-- Define the vectors
def a (m : ℝ) : ℝ × ℝ := (m, 2)
def b (n : ℝ) : ℝ × ℝ := (-1, n)

-- Define the theorem
theorem vector_magnitude_problem (m n : ℝ) : 
  n > 0 ∧ 
  (a m) • (b n) = 0 ∧ 
  m^2 + n^2 = 5 → 
  ‖2 • (a m) + (b n)‖ = Real.sqrt 34 := by
  sorry

end vector_magnitude_problem_l1719_171910


namespace original_number_proof_l1719_171943

theorem original_number_proof :
  ∃ N : ℕ, 
    (∃ k : ℤ, Odd (N * k) ∧ (N * k) % 9 = 0) ∧
    N * 4 = 108 ∧
    N = 27 := by
  sorry

end original_number_proof_l1719_171943


namespace pie_crust_flour_calculation_l1719_171974

/-- Given the initial conditions of pie crust baking and a new number of crusts,
    calculate the amount of flour required for each new crust. -/
theorem pie_crust_flour_calculation (initial_crusts : ℕ) (initial_flour : ℚ) (new_crusts : ℕ) :
  initial_crusts > 0 →
  initial_flour > 0 →
  new_crusts > 0 →
  (initial_flour / initial_crusts) * new_crusts = initial_flour →
  initial_flour / new_crusts = 4 / 5 := by
  sorry

end pie_crust_flour_calculation_l1719_171974


namespace couples_after_dance_l1719_171988

/-- The number of initial couples at the ball. -/
def n : ℕ := 2018

/-- The function that determines the source area for a couple at minute i. -/
def s (i : ℕ) : ℕ := i % n + 1

/-- The function that determines the destination area for a couple at minute i. -/
def r (i : ℕ) : ℕ := (2 * i) % n + 1

/-- Predicate to determine if a couple in area k survives after t minutes. -/
def survives (k t : ℕ) : Prop := sorry

/-- The number of couples remaining after t minutes. -/
def remaining_couples (t : ℕ) : ℕ := sorry

/-- The main theorem stating that after n² minutes, 505 couples remain. -/
theorem couples_after_dance : remaining_couples (n^2) = 505 := by sorry

end couples_after_dance_l1719_171988


namespace simplify_trig_expression_l1719_171953

theorem simplify_trig_expression : 
  Real.sqrt (1 - Real.sin (160 * π / 180) ^ 2) = Real.cos (20 * π / 180) := by
  sorry

end simplify_trig_expression_l1719_171953


namespace summer_degrees_l1719_171915

/-- Given two people where one has five more degrees than the other, 
    and their combined degrees total 295, prove that the person with 
    more degrees has 150 degrees. -/
theorem summer_degrees (s j : ℕ) 
    (h1 : s = j + 5)
    (h2 : s + j = 295) : 
  s = 150 := by
  sorry

end summer_degrees_l1719_171915


namespace four_point_theorem_l1719_171977

-- Define a type for points in a plane
variable (Point : Type)

-- Define a predicate for collinearity
variable (collinear : Point → Point → Point → Point → Prop)

-- Define a predicate for concyclicity
variable (concyclic : Point → Point → Point → Point → Prop)

-- Define a predicate for circle intersection
variable (circle_intersect : Point → Point → Point → Point → Prop)

-- Define the theorem
theorem four_point_theorem 
  (A B C D : Point) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_intersect : circle_intersect A B C D) : 
  collinear A B C D ∨ concyclic A B C D :=
sorry

end four_point_theorem_l1719_171977


namespace roger_has_two_more_candies_l1719_171923

/-- The number of candy bags Sandra has -/
def sandra_bags : ℕ := 2

/-- The number of candy pieces in each of Sandra's bags -/
def sandra_pieces_per_bag : ℕ := 6

/-- The number of candy bags Roger has -/
def roger_bags : ℕ := 2

/-- The number of candy pieces in Roger's first bag -/
def roger_bag1_pieces : ℕ := 11

/-- The number of candy pieces in Roger's second bag -/
def roger_bag2_pieces : ℕ := 3

/-- Theorem stating that Roger has 2 more pieces of candy than Sandra -/
theorem roger_has_two_more_candies : 
  (roger_bag1_pieces + roger_bag2_pieces) - (sandra_bags * sandra_pieces_per_bag) = 2 := by
  sorry

end roger_has_two_more_candies_l1719_171923


namespace soccer_balls_per_basket_l1719_171911

theorem soccer_balls_per_basket
  (num_baskets : ℕ)
  (tennis_balls_per_basket : ℕ)
  (total_balls_removed : ℕ)
  (balls_remaining : ℕ)
  (h1 : num_baskets = 5)
  (h2 : tennis_balls_per_basket = 15)
  (h3 : total_balls_removed = 44)
  (h4 : balls_remaining = 56) :
  (num_baskets * tennis_balls_per_basket + num_baskets * 5 = balls_remaining + total_balls_removed) := by
sorry

end soccer_balls_per_basket_l1719_171911


namespace triangle_count_is_twenty_l1719_171952

/-- Represents a point on the 3x3 grid -/
structure GridPoint where
  x : Fin 3
  y : Fin 3

/-- Represents a triangle on the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The set of all possible triangles on the 3x3 grid -/
def allGridTriangles : Set GridTriangle := sorry

/-- Counts the number of triangles in the 3x3 grid -/
def countTriangles : ℕ := sorry

/-- Theorem stating that the number of triangles in the 3x3 grid is 20 -/
theorem triangle_count_is_twenty : countTriangles = 20 := by sorry

end triangle_count_is_twenty_l1719_171952


namespace intersection_of_M_and_N_l1719_171963

def M : Set Int := {1, 2, 3, 4}
def N : Set Int := {-2, 2}

theorem intersection_of_M_and_N : M ∩ N = {2} := by sorry

end intersection_of_M_and_N_l1719_171963


namespace sqrt_equation_solution_l1719_171912

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (5 + n) = 8 → n = 59 := by
  sorry

end sqrt_equation_solution_l1719_171912


namespace different_result_l1719_171959

theorem different_result : 
  (-2 - (-3) ≠ 2 - 3) ∧ 
  (-2 - (-3) ≠ -3 + 2) ∧ 
  (-2 - (-3) ≠ -3 - (-2)) ∧ 
  (2 - 3 = -3 + 2) ∧ 
  (2 - 3 = -3 - (-2)) := by
  sorry

end different_result_l1719_171959


namespace sequence_increasing_iff_a0_eq_one_fifth_l1719_171993

/-- The sequence defined by a(n+1) = 2^n - 3*a(n) -/
def a : ℕ → ℝ → ℝ 
  | 0, a₀ => a₀
  | n + 1, a₀ => 2^n - 3 * a n a₀

/-- The sequence is increasing -/
def is_increasing (a₀ : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) a₀ > a n a₀

/-- Theorem: The sequence is increasing if and only if a₀ = 1/5 -/
theorem sequence_increasing_iff_a0_eq_one_fifth :
  ∀ a₀ : ℝ, is_increasing a₀ ↔ a₀ = 1/5 := by sorry

end sequence_increasing_iff_a0_eq_one_fifth_l1719_171993


namespace stock_dividend_rate_l1719_171945

/-- Given a stock with a certain yield and price, calculate its dividend rate. -/
def dividend_rate (yield : ℝ) (price : ℝ) : ℝ :=
  yield * price

/-- Theorem: The dividend rate of a stock yielding 8% quoted at 150 is 12. -/
theorem stock_dividend_rate :
  let yield : ℝ := 0.08
  let price : ℝ := 150
  dividend_rate yield price = 12 := by
  sorry

end stock_dividend_rate_l1719_171945


namespace emmy_and_gerry_apples_l1719_171951

/-- The number of apples Emmy and Gerry can buy together -/
def total_apples (apple_price : ℕ) (emmy_money : ℕ) (gerry_money : ℕ) : ℕ :=
  (emmy_money + gerry_money) / apple_price

/-- Theorem: Emmy and Gerry can buy 150 apples altogether -/
theorem emmy_and_gerry_apples :
  total_apples 2 200 100 = 150 := by
  sorry

#eval total_apples 2 200 100

end emmy_and_gerry_apples_l1719_171951


namespace cube_less_than_triple_l1719_171995

theorem cube_less_than_triple (x : ℤ) : x^3 < 3*x ↔ x = -3 ∨ x = -2 ∨ x = 1 := by
  sorry

end cube_less_than_triple_l1719_171995


namespace pizza_sector_chord_length_squared_l1719_171924

theorem pizza_sector_chord_length_squared (r : ℝ) (h : r = 8) :
  let chord_length_squared := 2 * r^2
  chord_length_squared = 128 := by sorry

end pizza_sector_chord_length_squared_l1719_171924


namespace inequality_solution_l1719_171954

theorem inequality_solution : 
  {x : ℝ | (x - 2)^2 < 3*x + 4} = {x : ℝ | 0 ≤ x ∧ x < 7} := by sorry

end inequality_solution_l1719_171954


namespace one_plus_three_squared_l1719_171937

theorem one_plus_three_squared : 1 + 3^2 = 10 := by
  sorry

end one_plus_three_squared_l1719_171937


namespace cost_increase_percentage_l1719_171906

theorem cost_increase_percentage (cost selling_price : ℝ) (increase_factor : ℝ) : 
  cost > 0 →
  selling_price = cost * 2.6 →
  (selling_price - cost * (1 + increase_factor)) / selling_price = 0.5692307692307692 →
  increase_factor = 0.12 := by
sorry

end cost_increase_percentage_l1719_171906
