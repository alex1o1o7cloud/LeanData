import Mathlib

namespace midnight_temperature_l1074_107479

def morning_temp : ℝ := 30
def afternoon_rise : ℝ := 1
def midnight_drop : ℝ := 7

theorem midnight_temperature : 
  morning_temp + afternoon_rise - midnight_drop = 24 := by
  sorry

end midnight_temperature_l1074_107479


namespace existence_of_sequence_l1074_107498

theorem existence_of_sequence (α : ℝ) (n : ℕ) (h_α : 0 < α ∧ α < 1) (h_n : 0 < n) :
  ∃ (a : ℕ → ℕ), 
    (∀ i ∈ Finset.range n, 1 ≤ a i) ∧
    (∀ i ∈ Finset.range (n-1), a i < a (i+1)) ∧
    (∀ i ∈ Finset.range n, a i ≤ 2^(n-1)) ∧
    (∀ i ∈ Finset.range (n-1), ⌊(α^(i+1) : ℝ) * (a (i+1) : ℝ)⌋ ≥ ⌊(α^i : ℝ) * (a i : ℝ)⌋) :=
by sorry

end existence_of_sequence_l1074_107498


namespace monomial_properties_l1074_107483

def monomial_coefficient (a : ℤ) (b c : ℕ) : ℤ := -2

def monomial_degree (a : ℤ) (b c : ℕ) : ℕ := 1 + b + c

theorem monomial_properties :
  let m := monomial_coefficient (-2) 2 4
  let n := monomial_degree (-2) 2 4
  m = -2 ∧ n = 7 := by sorry

end monomial_properties_l1074_107483


namespace shaded_area_calculation_l1074_107455

/-- The area of the shaded regions in a figure with two rectangles and two semicircles removed -/
theorem shaded_area_calculation (small_radius : ℝ) (large_radius : ℝ)
  (h_small : small_radius = 3)
  (h_large : large_radius = 6) :
  let small_rect_area := small_radius * (2 * small_radius)
  let large_rect_area := large_radius * (2 * large_radius)
  let small_semicircle_area := π * small_radius^2 / 2
  let large_semicircle_area := π * large_radius^2 / 2
  small_rect_area + large_rect_area - small_semicircle_area - large_semicircle_area = 90 - 45 * π / 2 :=
by sorry

end shaded_area_calculation_l1074_107455


namespace vegetable_field_division_l1074_107436

theorem vegetable_field_division (total_area : ℚ) (num_parts : ℕ) 
  (h1 : total_area = 5)
  (h2 : num_parts = 8) :
  (1 : ℚ) / num_parts = 1 / 8 ∧ total_area / num_parts = 5 / 8 := by
  sorry

end vegetable_field_division_l1074_107436


namespace divisor_proof_l1074_107489

theorem divisor_proof : ∃ x : ℝ, (26.3 * 12 * 20) / x + 125 = 2229 ∧ x = 3 := by
  sorry

end divisor_proof_l1074_107489


namespace city_distance_proof_l1074_107453

theorem city_distance_proof : 
  ∃ S : ℕ+, 
    (∀ x : ℕ, x ≤ S → (Nat.gcd x (S - x) = 1 ∨ Nat.gcd x (S - x) = 3 ∨ Nat.gcd x (S - x) = 13)) ∧ 
    (∀ T : ℕ+, T < S → ∃ y : ℕ, y ≤ T ∧ Nat.gcd y (T - y) ≠ 1 ∧ Nat.gcd y (T - y) ≠ 3 ∧ Nat.gcd y (T - y) ≠ 13) ∧
    S = 39 :=
by sorry

end city_distance_proof_l1074_107453


namespace difference_of_squares_divisible_by_eight_l1074_107403

theorem difference_of_squares_divisible_by_eight (a b : ℤ) (h : a > b) :
  ∃ k : ℤ, (2 * a + 1)^2 - (2 * b + 1)^2 = 8 * k := by
  sorry

end difference_of_squares_divisible_by_eight_l1074_107403


namespace arithmetic_geometric_mean_inequality_l1074_107469

theorem arithmetic_geometric_mean_inequality 
  (a b c : ℝ) 
  (ha : a ≥ 0) 
  (hb : b ≥ 0) 
  (hc : c ≥ 0) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) :=
sorry

end arithmetic_geometric_mean_inequality_l1074_107469


namespace work_completion_time_l1074_107478

/-- Given that P persons can complete a work in 24 days, 
    prove that 2P persons can complete half of the work in 6 days. -/
theorem work_completion_time 
  (P : ℕ) -- number of persons
  (full_work : ℝ) -- amount of full work
  (h1 : P > 0) -- assumption that there's at least one person
  (h2 : full_work > 0) -- assumption that there's some work to be done
  (h3 : P * 24 * full_work = P * 24 * full_work) -- work completion condition
  : (2 * P) * 6 * (full_work / 2) = P * 24 * full_work := by
  sorry

end work_completion_time_l1074_107478


namespace sunday_visitors_theorem_l1074_107463

/-- Represents the average number of visitors on Sundays in a library -/
def average_sunday_visitors (
  total_days : ℕ)  -- Total number of days in the month
  (sunday_count : ℕ)  -- Number of Sundays in the month
  (non_sunday_average : ℕ)  -- Average number of visitors on non-Sundays
  (month_average : ℕ)  -- Average number of visitors per day for the entire month
  : ℕ :=
  ((month_average * total_days) - (non_sunday_average * (total_days - sunday_count))) / sunday_count

/-- Theorem stating that the average number of Sunday visitors is 510 given the problem conditions -/
theorem sunday_visitors_theorem :
  average_sunday_visitors 30 5 240 285 = 510 := by
  sorry

#eval average_sunday_visitors 30 5 240 285

end sunday_visitors_theorem_l1074_107463


namespace divisibility_by_29_fourth_power_l1074_107402

theorem divisibility_by_29_fourth_power (x y z : ℤ) (S : ℤ) 
  (h1 : S = x^4 + y^4 + z^4) 
  (h2 : 29 ∣ S) : 
  29^4 ∣ S := by
  sorry

end divisibility_by_29_fourth_power_l1074_107402


namespace interior_angles_sum_l1074_107429

/-- If the sum of the interior angles of a convex polygon with n sides is 1800°,
    then the sum of the interior angles of a convex polygon with n + 4 sides is 2520°. -/
theorem interior_angles_sum (n : ℕ) :
  (180 * (n - 2) = 1800) → (180 * ((n + 4) - 2) = 2520) := by
  sorry

end interior_angles_sum_l1074_107429


namespace league_face_count_l1074_107456

/-- The number of games in a single round-robin tournament with n teams -/
def roundRobinGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of times each team faces another in a league -/
def faceCount (totalTeams : ℕ) (totalGames : ℕ) : ℕ :=
  totalGames / roundRobinGames totalTeams

theorem league_face_count :
  faceCount 14 455 = 5 := by sorry

end league_face_count_l1074_107456


namespace perfect_square_consecutive_base_equation_l1074_107460

theorem perfect_square_consecutive_base_equation :
  ∀ (A B : ℕ),
    (∃ n : ℕ, A = n^2) →
    B = A + 1 →
    (1 * A^2 + 2 * A + 3) + (2 * B + 1) = 5 * (A + B) →
    (A : ℝ) + B = 7 + 4 * Real.sqrt 2 :=
by
  sorry

end perfect_square_consecutive_base_equation_l1074_107460


namespace common_chord_of_circles_l1074_107476

/-- Given two circles in the xy-plane, this theorem states that
    their common chord lies on a specific line. -/
theorem common_chord_of_circles (x y : ℝ) : 
  (x^2 + y^2 + 2*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) := by
  sorry

end common_chord_of_circles_l1074_107476


namespace sum_of_solutions_equation_l1074_107412

theorem sum_of_solutions_equation (x₁ x₂ : ℚ) : 
  (4 * x₁ + 7 = 0 ∨ 5 * x₁ - 8 = 0) ∧
  (4 * x₂ + 7 = 0 ∨ 5 * x₂ - 8 = 0) ∧
  x₁ ≠ x₂ →
  x₁ + x₂ = -3/20 := by sorry

end sum_of_solutions_equation_l1074_107412


namespace sticker_distribution_l1074_107481

theorem sticker_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  (Nat.choose (n + k - 1) (k - 1)) = 1001 := by
  sorry

end sticker_distribution_l1074_107481


namespace arnold_protein_consumption_l1074_107410

/-- Protein content of food items and consumption amounts -/
def collagen_protein : ℕ := 9
def protein_powder_protein : ℕ := 21
def steak_protein : ℕ := 56
def yogurt_protein : ℕ := 15
def almonds_protein : ℕ := 12

def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 2
def steak_count : ℕ := 1
def yogurt_servings : ℕ := 1
def almonds_cups : ℕ := 1

/-- Total protein consumed by Arnold -/
def total_protein : ℕ :=
  collagen_protein * collagen_scoops +
  protein_powder_protein * protein_powder_scoops +
  steak_protein * steak_count +
  yogurt_protein * yogurt_servings +
  almonds_protein * almonds_cups

/-- Theorem stating that the total protein consumed is 134 grams -/
theorem arnold_protein_consumption : total_protein = 134 := by
  sorry

end arnold_protein_consumption_l1074_107410


namespace chromium_percentage_in_mixed_alloy_l1074_107405

/-- Given two alloys with different chromium percentages and weights, 
    calculates the chromium percentage in the resulting alloy when mixed. -/
theorem chromium_percentage_in_mixed_alloy 
  (chromium_percent1 chromium_percent2 : ℝ)
  (weight1 weight2 : ℝ)
  (h1 : chromium_percent1 = 15)
  (h2 : chromium_percent2 = 8)
  (h3 : weight1 = 15)
  (h4 : weight2 = 35) :
  let total_chromium := (chromium_percent1 / 100 * weight1) + (chromium_percent2 / 100 * weight2)
  let total_weight := weight1 + weight2
  (total_chromium / total_weight) * 100 = 10.1 := by
sorry

end chromium_percentage_in_mixed_alloy_l1074_107405


namespace total_discount_calculation_l1074_107452

theorem total_discount_calculation (original_price : ℝ) (initial_discount : ℝ) (additional_discount : ℝ) :
  initial_discount = 0.5 →
  additional_discount = 0.25 →
  let sale_price := original_price * (1 - initial_discount)
  let final_price := sale_price * (1 - additional_discount)
  let total_discount := (original_price - final_price) / original_price
  total_discount = 0.625 :=
by sorry

end total_discount_calculation_l1074_107452


namespace rational_terms_count_l1074_107492

/-- The number of rational terms in the expansion of (√2 + ∛3)^100 -/
def rational_terms_a : ℕ := 26

/-- The number of rational terms in the expansion of (√2 + ∜3)^300 -/
def rational_terms_b : ℕ := 13

/-- Theorem stating the number of rational terms in the expansions -/
theorem rational_terms_count :
  (rational_terms_a = 26) ∧ (rational_terms_b = 13) := by sorry

end rational_terms_count_l1074_107492


namespace pure_imaginary_ratio_l1074_107417

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 5*I) * (a + b*I) = y*I) : a/b = -5/3 := by
  sorry

end pure_imaginary_ratio_l1074_107417


namespace harvest_duration_l1074_107477

theorem harvest_duration (total_earnings : ℕ) (weekly_earnings : ℕ) (h1 : total_earnings = 133) (h2 : weekly_earnings = 7) :
  total_earnings / weekly_earnings = 19 :=
by
  sorry

end harvest_duration_l1074_107477


namespace sqrt_3_times_sqrt_12_l1074_107499

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by sorry

end sqrt_3_times_sqrt_12_l1074_107499


namespace partition_product_ratio_l1074_107414

theorem partition_product_ratio (n : ℕ) (h : n > 2) :
  ∃ (A B : Finset ℕ), 
    A ∪ B = Finset.range n ∧ 
    A ∩ B = ∅ ∧ 
    max ((A.prod id) / (B.prod id)) ((B.prod id) / (A.prod id)) ≤ (n - 1) / (n - 2) := by
  sorry

end partition_product_ratio_l1074_107414


namespace max_value_expression_l1074_107415

theorem max_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y^2 * z^2 * (x^2 + y^2 + z^2)) / ((x + y)^3 * (y + z)^3) ≤ 1/24 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 * b^2 * c^2 * (a^2 + b^2 + c^2)) / ((a + b)^3 * (b + c)^3) = 1/24 :=
by sorry

end max_value_expression_l1074_107415


namespace max_y_coordinate_of_ellipse_l1074_107473

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the equation of the ellipse -/
def isOnEllipse (p : Point) : Prop :=
  (p.x - 3)^2 / 49 + (p.y - 4)^2 / 25 = 1

/-- Theorem: The maximum y-coordinate of any point on the given ellipse is 9 -/
theorem max_y_coordinate_of_ellipse :
  ∀ p : Point, isOnEllipse p → p.y ≤ 9 ∧ ∃ q : Point, isOnEllipse q ∧ q.y = 9 :=
by sorry

end max_y_coordinate_of_ellipse_l1074_107473


namespace concert_attendance_difference_l1074_107420

theorem concert_attendance_difference (first_concert : Nat) (second_concert : Nat)
  (h1 : first_concert = 65899)
  (h2 : second_concert = 66018) :
  second_concert - first_concert = 119 := by
  sorry

end concert_attendance_difference_l1074_107420


namespace largest_a_value_l1074_107408

/-- The equation has at least one integer root -/
def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, (x^2 - (a+7)*x + 7*a)^(1/3) + 3^(1/3) = 0

/-- 11 is the largest integer value of a for which the equation has at least one integer root -/
theorem largest_a_value : (has_integer_root 11 ∧ ∀ a : ℤ, a > 11 → ¬has_integer_root a) :=
sorry

end largest_a_value_l1074_107408


namespace impossible_equal_sum_arrangement_l1074_107470

theorem impossible_equal_sum_arrangement : ¬∃ (a b c d e f : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) ∧
  (∃ (s : ℕ), 
    a + b + c = s ∧
    a + d + e = s ∧
    b + d + f = s ∧
    c + e + f = s) :=
by sorry

end impossible_equal_sum_arrangement_l1074_107470


namespace range_of_a_l1074_107435

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → a ≥ x^2 - 2*x - 1) → 
  a ≥ 2 := by
sorry

end range_of_a_l1074_107435


namespace four_last_digit_fib_mod8_l1074_107406

/-- Fibonacci sequence modulo 8 -/
def fib_mod8 : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => (fib_mod8 n + fib_mod8 (n + 1)) % 8

/-- Set of digits that have appeared in the Fibonacci sequence modulo 8 up to n -/
def digits_appeared (n : ℕ) : Finset ℕ :=
  Finset.range (n + 1).succ
    |>.filter (fun i => fib_mod8 i ∈ Finset.range 8)
    |>.image fib_mod8

/-- The proposition that 4 is the last digit to appear in the Fibonacci sequence modulo 8 -/
theorem four_last_digit_fib_mod8 :
  ∃ n : ℕ, 4 ∈ digits_appeared n ∧ digits_appeared n = Finset.range 8 :=
sorry

end four_last_digit_fib_mod8_l1074_107406


namespace arithmetic_sequence_problem_l1074_107421

theorem arithmetic_sequence_problem :
  ∀ a b c : ℤ,
  (∃ d : ℤ, b = a + d ∧ c = b + d) →  -- arithmetic sequence condition
  a + b + c = 6 →                    -- sum condition
  a * b * c = -10 →                  -- product condition
  ((a = 5 ∧ b = 2 ∧ c = -1) ∨ (a = -1 ∧ b = 2 ∧ c = 5)) :=
by sorry

end arithmetic_sequence_problem_l1074_107421


namespace rectangle_triangle_equal_area_l1074_107419

/-- The width of a rectangle whose area is equal to the area of a triangle with base 16 and height equal to the rectangle's length -/
theorem rectangle_triangle_equal_area (x : ℝ) (y : ℝ) 
  (h : x * y = (1/2) * 16 * x) : y = 8 := by
  sorry

end rectangle_triangle_equal_area_l1074_107419


namespace not_divisible_by_169_l1074_107462

theorem not_divisible_by_169 (n : ℕ) : ¬(169 ∣ (n^2 + 5*n + 16)) := by
  sorry

end not_divisible_by_169_l1074_107462


namespace consecutive_integers_sum_30_l1074_107441

theorem consecutive_integers_sum_30 : ∃! a : ℕ, ∃ n : ℕ,
  n ≥ 3 ∧ (Finset.range n).sum (λ i => a + i) = 30 :=
by sorry

end consecutive_integers_sum_30_l1074_107441


namespace smallest_integer_with_remainder_one_sixty_one_satisfies_conditions_smallest_integer_is_sixty_one_l1074_107443

theorem smallest_integer_with_remainder_one (n : ℕ) : n > 1 ∧ 
  n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1 → n ≥ 61 :=
by
  sorry

theorem sixty_one_satisfies_conditions : 
  61 > 1 ∧ 61 % 4 = 1 ∧ 61 % 5 = 1 ∧ 61 % 6 = 1 :=
by
  sorry

theorem smallest_integer_is_sixty_one : 
  ∃ (n : ℕ), n > 1 ∧ n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1 ∧ 
  ∀ (m : ℕ), m > 1 ∧ m % 4 = 1 ∧ m % 5 = 1 ∧ m % 6 = 1 → m ≥ n :=
by
  sorry

end smallest_integer_with_remainder_one_sixty_one_satisfies_conditions_smallest_integer_is_sixty_one_l1074_107443


namespace total_distance_walked_l1074_107490

-- Define constants for conversion
def feet_per_mile : ℕ := 5280
def feet_per_yard : ℕ := 3

-- Define the distances walked by each person
def lionel_miles : ℕ := 4
def esther_yards : ℕ := 975
def niklaus_feet : ℕ := 1287

-- Theorem statement
theorem total_distance_walked :
  lionel_miles * feet_per_mile + esther_yards * feet_per_yard + niklaus_feet = 24332 :=
by sorry

end total_distance_walked_l1074_107490


namespace dividend_calculation_l1074_107449

/-- Calculates the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : face_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : dividend_rate = 0.07) :
  let share_cost := face_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := face_value * dividend_rate
  num_shares * dividend_per_share = 840 := by sorry

end dividend_calculation_l1074_107449


namespace emily_spent_twelve_dollars_l1074_107413

/-- The amount Emily spent on flowers -/
def emily_spent (price_per_flower : ℕ) (num_roses : ℕ) (num_daisies : ℕ) : ℕ :=
  price_per_flower * (num_roses + num_daisies)

/-- Theorem: Emily spent 12 dollars on flowers -/
theorem emily_spent_twelve_dollars :
  emily_spent 3 2 2 = 12 := by
  sorry

end emily_spent_twelve_dollars_l1074_107413


namespace stock_investment_fractions_l1074_107422

theorem stock_investment_fractions (initial_investment : ℝ) 
  (final_value : ℝ) (f : ℝ) : 
  initial_investment = 900 →
  final_value = 1350 →
  0 ≤ f →
  f ≤ 1/2 →
  2 * (2 * f * initial_investment) + (1/2 * (1 - 2*f) * initial_investment) = final_value →
  f = 1/3 := by
  sorry

end stock_investment_fractions_l1074_107422


namespace circle_center_l1074_107426

/-- The center of a circle given by the equation (x-h)^2 + (y-k)^2 = r^2 is (h,k) -/
theorem circle_center (h k r : ℝ) : 
  (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 ↔ ((x, y) ∈ {p : ℝ × ℝ | (p.1 - h)^2 + (p.2 - k)^2 = r^2})) → 
  (h, k) = (1, 1) → r^2 = 2 →
  (1, 1) ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 2} :=
by sorry

end circle_center_l1074_107426


namespace digit_129_in_n_or_3n_l1074_107454

/-- Given a natural number, returns true if it contains the digit 1, 2, or 9 in its base-ten representation -/
def containsDigit129 (n : ℕ) : Prop :=
  ∃ d, d ∈ [1, 2, 9] ∧ ∃ k m, n = k * 10 + d + m * 10

theorem digit_129_in_n_or_3n (n : ℕ+) : containsDigit129 n.val ∨ containsDigit129 (3 * n.val) := by
  sorry

end digit_129_in_n_or_3n_l1074_107454


namespace friends_bill_split_l1074_107448

-- Define the problem parameters
def num_friends : ℕ := 5
def original_bill : ℚ := 100
def discount_percentage : ℚ := 6

-- Define the theorem
theorem friends_bill_split :
  let discount := discount_percentage / 100 * original_bill
  let discounted_bill := original_bill - discount
  let individual_payment := discounted_bill / num_friends
  individual_payment = 18.8 := by sorry

end friends_bill_split_l1074_107448


namespace max_value_implies_m_l1074_107400

-- Define the variables
variable (x y m : ℝ)

-- Define the function z
def z (x y : ℝ) : ℝ := x - 3 * y

-- State the theorem
theorem max_value_implies_m (h1 : y ≥ x) (h2 : x + 3 * y ≤ 4) (h3 : x ≥ m)
  (h4 : ∀ x' y', y' ≥ x' → x' + 3 * y' ≤ 4 → x' ≥ m → z x' y' ≤ 8) 
  (h5 : ∃ x' y', y' ≥ x' ∧ x' + 3 * y' ≤ 4 ∧ x' ≥ m ∧ z x' y' = 8) : m = -4 := by
  sorry

end max_value_implies_m_l1074_107400


namespace log_inequality_l1074_107475

theorem log_inequality : 
  let m := Real.log 0.6 / Real.log 0.3
  let n := (1/2) * (Real.log 0.6 / Real.log 2)
  m + n > m * n := by sorry

end log_inequality_l1074_107475


namespace unique_modular_residue_l1074_107444

theorem unique_modular_residue :
  ∃! n : ℤ, 0 ≤ n ∧ n < 11 ∧ -1234 ≡ n [ZMOD 11] :=
by sorry

end unique_modular_residue_l1074_107444


namespace complement_A_intersect_B_l1074_107466

open Set Real

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

-- Define set B
def B : Set ℝ := {x | 2 * x - x^2 > 0}

-- State the theorem
theorem complement_A_intersect_B : (𝒰 \ A) ∩ B = Ioo 0 1 := by sorry

end complement_A_intersect_B_l1074_107466


namespace no_distributive_laws_hold_l1074_107424

-- Define the # operation
def hash (a b : ℝ) : ℝ := a + 2 * b

-- Theorem stating that none of the distributive laws hold
theorem no_distributive_laws_hold :
  (∃ x y z : ℝ, hash x (y + z) ≠ hash x y + hash x z) ∧
  (∃ x y z : ℝ, x + hash y z ≠ hash (x + y) (x + z)) ∧
  (∃ x y z : ℝ, hash x (hash y z) ≠ hash (hash x y) (hash x z)) :=
by
  sorry


end no_distributive_laws_hold_l1074_107424


namespace six_people_arrangement_l1074_107493

/-- The number of ways to arrange 6 people in a line with two specific people not adjacent -/
def line_arrangement (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial (n - k) * (Nat.choose (n - k + 1) k)

theorem six_people_arrangement :
  line_arrangement 6 2 = 480 := by
  sorry

end six_people_arrangement_l1074_107493


namespace function_and_range_l1074_107467

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 3 * x - 2

-- Define the function g
def g : ℝ → ℝ := fun x ↦ x * f x

-- Theorem statement
theorem function_and_range :
  (∀ x : ℝ, f x + 2 * f (-x) = -3 * x - 6) →
  (∀ x : ℝ, f x = 3 * x - 2) ∧
  (Set.Icc 0 3).image g = Set.Icc (-1/3) 21 :=
by sorry

end function_and_range_l1074_107467


namespace max_value_of_b_l1074_107416

theorem max_value_of_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2 * a * b = (2 * a - b) / (2 * a + 3 * b)) : 
  b ≤ 1/3 ∧ ∃ (x : ℝ), x > 0 ∧ 2 * x * (1/3) = (2 * x - 1/3) / (2 * x + 1) := by
sorry

end max_value_of_b_l1074_107416


namespace distance_to_park_is_five_l1074_107491

/-- The distance from Talia's house to the park -/
def distance_to_park : ℝ := sorry

/-- The distance from the park to the grocery store -/
def park_to_grocery : ℝ := 3

/-- The distance from the grocery store to Talia's house -/
def grocery_to_house : ℝ := 8

/-- The total distance Talia drives -/
def total_distance : ℝ := 16

theorem distance_to_park_is_five :
  distance_to_park = 5 :=
by
  have h1 : distance_to_park + park_to_grocery + grocery_to_house = total_distance := by sorry
  sorry

end distance_to_park_is_five_l1074_107491


namespace complex_magnitude_problem_l1074_107482

theorem complex_magnitude_problem (z : ℂ) (h : (z - Complex.I) * (1 + Complex.I) = 2 - Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_magnitude_problem_l1074_107482


namespace paint_room_time_l1074_107411

/-- The time (in hours) it takes Alice to paint the room alone -/
def alice_time : ℝ := 3

/-- The time (in hours) it takes Bob to paint the room alone -/
def bob_time : ℝ := 6

/-- The duration (in hours) of the break Alice and Bob take -/
def break_time : ℝ := 2

/-- The total time (in hours) it takes Alice and Bob to paint the room together, including the break -/
def total_time : ℝ := 4

theorem paint_room_time :
  (1 / alice_time + 1 / bob_time) * (total_time - break_time) = 1 :=
sorry

end paint_room_time_l1074_107411


namespace fathers_full_time_jobs_l1074_107459

theorem fathers_full_time_jobs (total_parents : ℝ) (h1 : total_parents > 0) : 
  let mothers := 0.4 * total_parents
  let fathers := 0.6 * total_parents
  let mothers_full_time := 0.9 * mothers
  let total_full_time := 0.81 * total_parents
  let fathers_full_time := total_full_time - mothers_full_time
  fathers_full_time / fathers = 3/4 := by sorry

end fathers_full_time_jobs_l1074_107459


namespace addition_subtraction_elimination_not_factorization_l1074_107474

/-- Represents a mathematical method --/
inductive Method
  | TakingOutCommonFactor
  | CrossMultiplication
  | Formula
  | AdditionSubtractionElimination

/-- Predicate to determine if a method is a factorization method --/
def IsFactorizationMethod (m : Method) : Prop :=
  m = Method.TakingOutCommonFactor ∨ 
  m = Method.CrossMultiplication ∨ 
  m = Method.Formula

theorem addition_subtraction_elimination_not_factorization :
  ¬(IsFactorizationMethod Method.AdditionSubtractionElimination) :=
by sorry

end addition_subtraction_elimination_not_factorization_l1074_107474


namespace log_equation_solution_l1074_107484

theorem log_equation_solution : ∃ x : ℝ, (Real.log x - Real.log 25) / 100 = -20 := by
  sorry

end log_equation_solution_l1074_107484


namespace ant_final_position_l1074_107461

/-- Represents a point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents the state of the ant -/
structure AntState where
  position : Point
  direction : Direction
  moveCount : Nat
  moveDistance : Nat

/-- Function to update the ant's state after a move -/
def move (state : AntState) : AntState :=
  match state.direction with
  | Direction.North => { state with position := ⟨state.position.x, state.position.y + state.moveDistance⟩, direction := Direction.East }
  | Direction.East => { state with position := ⟨state.position.x + state.moveDistance, state.position.y⟩, direction := Direction.South }
  | Direction.South => { state with position := ⟨state.position.x, state.position.y - state.moveDistance⟩, direction := Direction.West }
  | Direction.West => { state with position := ⟨state.position.x - state.moveDistance, state.position.y⟩, direction := Direction.North }

/-- Function to perform multiple moves -/
def multiMove (initialState : AntState) (n : Nat) : AntState :=
  match n with
  | 0 => initialState
  | m + 1 => 
    let newState := move initialState
    multiMove { newState with moveCount := newState.moveCount + 1, moveDistance := newState.moveDistance + 2 } m

/-- Theorem stating the final position of the ant -/
theorem ant_final_position :
  let initialState : AntState := {
    position := ⟨10, -10⟩,
    direction := Direction.North,
    moveCount := 0,
    moveDistance := 2
  }
  let finalState := multiMove initialState 10
  finalState.position = ⟨22, 0⟩ := by
  sorry


end ant_final_position_l1074_107461


namespace largest_k_inequality_l1074_107494

theorem largest_k_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (∀ k : ℕ+, (1 / (a - b) + 1 / (b - c) ≥ k / (a - c)) → k ≤ 4) ∧ 
  (∃ a b c : ℝ, a > b ∧ b > c ∧ 1 / (a - b) + 1 / (b - c) = 4 / (a - c)) := by
  sorry

end largest_k_inequality_l1074_107494


namespace concavity_and_inflection_point_l1074_107488

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 4

-- Define the second derivative of f
def f'' (x : ℝ) : ℝ := 6*x - 12

-- Theorem stating the concavity and inflection point properties
theorem concavity_and_inflection_point :
  (∀ x < 2, f'' x < 0) ∧
  (∀ x > 2, f'' x > 0) ∧
  f'' 2 = 0 ∧
  f 2 = -12 := by
  sorry

end concavity_and_inflection_point_l1074_107488


namespace meet_once_l1074_107465

/-- Represents the meeting scenario between Michael and the garbage truck --/
structure MeetingScenario where
  michael_speed : ℝ
  pail_distance : ℝ
  truck_speed : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def number_of_meetings (scenario : MeetingScenario) : ℕ :=
  sorry

/-- The theorem stating that Michael and the truck meet exactly once --/
theorem meet_once (scenario : MeetingScenario) 
  (h1 : scenario.michael_speed = 4)
  (h2 : scenario.pail_distance = 300)
  (h3 : scenario.truck_speed = 6)
  (h4 : scenario.truck_stop_time = 20)
  (h5 : scenario.initial_distance = 300) :
  number_of_meetings scenario = 1 :=
sorry

end meet_once_l1074_107465


namespace chess_group_players_l1074_107404

/-- The number of players in the chess group. -/
def n : ℕ := 20

/-- The total number of games played. -/
def total_games : ℕ := 190

/-- Theorem stating that the number of players is correct given the conditions. -/
theorem chess_group_players :
  (n * (n - 1) / 2 = total_games) ∧
  (∀ m : ℕ, m ≠ n → m * (m - 1) / 2 ≠ total_games) := by
  sorry

#check chess_group_players

end chess_group_players_l1074_107404


namespace magical_red_knights_fraction_l1074_107433

theorem magical_red_knights_fraction (total : ℕ) (red : ℕ) (blue : ℕ) (magical : ℕ) 
  (h1 : red = (3 * total) / 7)
  (h2 : blue = total - red)
  (h3 : magical = total / 4)
  (h4 : ∃ (r s : ℕ), (r * blue * 3 = s * red) ∧ (r * red + r * blue = s * magical)) :
  ∃ (r s : ℕ), (r * red = s * magical) ∧ (r = 21 ∧ s = 52) :=
sorry

end magical_red_knights_fraction_l1074_107433


namespace min_container_cost_l1074_107480

def container_cost (a b : ℝ) : ℝ := 20 * (a * b) + 10 * 2 * (a + b)

theorem min_container_cost :
  ∀ a b : ℝ,
  a > 0 → b > 0 →
  a * b = 4 →
  container_cost a b ≥ 160 :=
by
  sorry

end min_container_cost_l1074_107480


namespace sufficient_not_necessary_condition_l1074_107458

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x = 1 → x^3 = x) ∧ ¬(x^3 = x → x = 1) := by
  sorry

end sufficient_not_necessary_condition_l1074_107458


namespace zero_acceleration_in_quadrant_IV_l1074_107409

-- Define the disk and its properties
structure Disk where
  uniform : Bool
  rolling_smoothly : Bool
  pulled_by_force : Bool

-- Define the acceleration vectors
structure Acceleration where
  tangential : ℝ × ℝ
  centripetal : ℝ × ℝ
  horizontal : ℝ × ℝ

-- Define the quadrants of the disk
inductive Quadrant
  | I
  | II
  | III
  | IV

-- Function to check if a point in a given quadrant can have zero total acceleration
def can_have_zero_acceleration (d : Disk) (q : Quadrant) (a : Acceleration) : Prop :=
  d.uniform ∧ d.rolling_smoothly ∧ d.pulled_by_force ∧
  match q with
  | Quadrant.IV => ∃ (x y : ℝ), 
      x > 0 ∧ y < 0 ∧
      a.tangential.1 + a.centripetal.1 + a.horizontal.1 = 0 ∧
      a.tangential.2 + a.centripetal.2 + a.horizontal.2 = 0
  | _ => False

-- Theorem statement
theorem zero_acceleration_in_quadrant_IV (d : Disk) (a : Acceleration) :
  d.uniform ∧ d.rolling_smoothly ∧ d.pulled_by_force →
  ∃ (q : Quadrant), can_have_zero_acceleration d q a :=
sorry

end zero_acceleration_in_quadrant_IV_l1074_107409


namespace number_of_goats_l1074_107439

theorem number_of_goats (total_cost cow_price goat_price : ℕ) 
  (h1 : total_cost = 1500)
  (h2 : cow_price = 400)
  (h3 : goat_price = 70) : 
  ∃ (num_goats : ℕ), total_cost = 2 * cow_price + num_goats * goat_price ∧ num_goats = 10 :=
by sorry

end number_of_goats_l1074_107439


namespace jerry_zinc_consumption_l1074_107430

/-- The amount of zinc Jerry eats from antacids -/
def zinc_consumed (big_antacid_weight : ℝ) (big_antacid_count : ℕ) (big_antacid_zinc_percent : ℝ)
                  (small_antacid_weight : ℝ) (small_antacid_count : ℕ) (small_antacid_zinc_percent : ℝ) : ℝ :=
  (big_antacid_weight * big_antacid_count * big_antacid_zinc_percent +
   small_antacid_weight * small_antacid_count * small_antacid_zinc_percent) * 1000

/-- Theorem stating the amount of zinc Jerry consumes -/
theorem jerry_zinc_consumption :
  zinc_consumed 2 2 0.05 1 3 0.15 = 650 := by
  sorry

end jerry_zinc_consumption_l1074_107430


namespace lineup_arrangements_eq_960_l1074_107434

/-- The number of ways to arrange 5 volunteers and 2 elderly individuals in a row,
    where the elderly individuals must stand next to each other but not at the ends. -/
def lineup_arrangements : ℕ :=
  let n_volunteers : ℕ := 5
  let n_elderly : ℕ := 2
  let volunteer_arrangements : ℕ := Nat.factorial n_volunteers
  let elderly_pair_positions : ℕ := n_volunteers - 1
  let elderly_internal_arrangements : ℕ := Nat.factorial n_elderly
  volunteer_arrangements * (elderly_pair_positions - 1) * elderly_internal_arrangements

theorem lineup_arrangements_eq_960 : lineup_arrangements = 960 := by
  sorry

end lineup_arrangements_eq_960_l1074_107434


namespace teaching_years_difference_l1074_107472

theorem teaching_years_difference :
  ∀ (V A D : ℕ),
  V + A + D = 93 →
  V = A + 9 →
  D = 40 →
  V < D →
  D - V = 9 :=
by
  sorry

end teaching_years_difference_l1074_107472


namespace remaining_payment_l1074_107407

/-- Given a product with a 10% deposit of $140, prove that the remaining amount to be paid is $1260 -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (full_price : ℝ) : 
  deposit = 140 ∧ 
  deposit_percentage = 0.1 ∧ 
  deposit = deposit_percentage * full_price → 
  full_price - deposit = 1260 :=
by sorry

end remaining_payment_l1074_107407


namespace geometric_sequence_fourth_term_l1074_107442

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- A sequence is geometric if the ratio between consecutive terms is constant. -/
def IsGeometric (a : Sequence) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem to be proved. -/
theorem geometric_sequence_fourth_term
  (a : Sequence)
  (h1 : a 1 = 2)
  (h2 : IsGeometric (fun n => 1 + a n) 3) :
  a 4 = 80 := by
  sorry


end geometric_sequence_fourth_term_l1074_107442


namespace selection_problem_l1074_107446

theorem selection_problem (n : ℕ) (r : ℕ) (h1 : n = 10) (h2 : r = 4) :
  Nat.choose n r = 210 := by
  sorry

end selection_problem_l1074_107446


namespace square_root_equation_solution_l1074_107425

theorem square_root_equation_solution (P : ℝ) :
  Real.sqrt (3 - 2*P) + Real.sqrt (1 - 2*P) = 2 → P = 3/8 := by
  sorry

end square_root_equation_solution_l1074_107425


namespace integer_fraction_characterization_l1074_107438

theorem integer_fraction_characterization (p n : ℕ) :
  Nat.Prime p → n > 0 →
  (∃ k : ℕ, (n^p + 1 : ℕ) = k * (p^n + 1)) ↔
  ((p = 2 ∧ (n = 2 ∨ n = 4)) ∨ (p > 2 ∧ n = p)) := by
  sorry

end integer_fraction_characterization_l1074_107438


namespace tim_appetizers_l1074_107486

theorem tim_appetizers (total_spent : ℚ) (entree_percentage : ℚ) (appetizer_cost : ℚ) : 
  total_spent = 50 →
  entree_percentage = 80 / 100 →
  appetizer_cost = 5 →
  (total_spent * (1 - entree_percentage)) / appetizer_cost = 2 := by
sorry

end tim_appetizers_l1074_107486


namespace beatrice_prob_five_given_win_l1074_107423

-- Define the number of players and die sides
def num_players : ℕ := 5
def num_sides : ℕ := 8

-- Define the probability of rolling a specific number
def prob_roll (n : ℕ) : ℚ := 1 / num_sides

-- Define the probability of winning for any player
def prob_win : ℚ := 1 / num_players

-- Define the probability of other players rolling less than 5
def prob_others_less_than_5 : ℚ := (4 / 8) ^ (num_players - 1)

-- Define the probability of winning with a 5 (including tie-breaks)
def prob_win_with_5 : ℚ := prob_others_less_than_5 + 369 / 2048

-- State the theorem
theorem beatrice_prob_five_given_win :
  (prob_roll 5 * prob_win_with_5) / prob_win = 115 / 1024 := by
sorry

end beatrice_prob_five_given_win_l1074_107423


namespace historical_fiction_new_release_fraction_is_four_sevenths_l1074_107445

/-- Represents the inventory of a bookstore -/
structure BookstoreInventory where
  total_books : ℕ
  historical_fiction_ratio : ℚ
  historical_fiction_new_release_ratio : ℚ
  other_new_release_ratio : ℚ

/-- Calculates the fraction of new releases that are historical fiction -/
def historical_fiction_new_release_fraction (inventory : BookstoreInventory) : ℚ :=
  let historical_fiction := inventory.total_books * inventory.historical_fiction_ratio
  let other_books := inventory.total_books * (1 - inventory.historical_fiction_ratio)
  let historical_fiction_new_releases := historical_fiction * inventory.historical_fiction_new_release_ratio
  let other_new_releases := other_books * inventory.other_new_release_ratio
  historical_fiction_new_releases / (historical_fiction_new_releases + other_new_releases)

/-- Theorem stating that the fraction of new releases that are historical fiction is 4/7 -/
theorem historical_fiction_new_release_fraction_is_four_sevenths
  (inventory : BookstoreInventory)
  (h1 : inventory.historical_fiction_ratio = 2/5)
  (h2 : inventory.historical_fiction_new_release_ratio = 2/5)
  (h3 : inventory.other_new_release_ratio = 1/5) :
  historical_fiction_new_release_fraction inventory = 4/7 := by
  sorry

end historical_fiction_new_release_fraction_is_four_sevenths_l1074_107445


namespace doll_production_theorem_l1074_107432

/-- The number of non-defective dolls produced per day -/
def non_defective_dolls : ℕ := 4800

/-- The ratio of total dolls to non-defective dolls -/
def total_to_non_defective_ratio : ℚ := 133 / 100

/-- The total number of dolls produced per day -/
def total_dolls : ℕ := 6384

/-- Theorem stating the relationship between non-defective dolls, the ratio, and total dolls -/
theorem doll_production_theorem :
  (non_defective_dolls : ℚ) * total_to_non_defective_ratio = total_dolls := by
  sorry

end doll_production_theorem_l1074_107432


namespace blue_paint_calculation_l1074_107495

/-- Given a paint mixture with a ratio of blue to green paint and a total number of cans,
    calculate the number of cans of blue paint required. -/
def blue_paint_cans (blue_ratio green_ratio total_cans : ℕ) : ℕ :=
  (blue_ratio * total_cans) / (blue_ratio + green_ratio)

/-- Theorem stating that for a 4:3 ratio of blue to green paint and 42 total cans,
    24 cans of blue paint are required. -/
theorem blue_paint_calculation :
  blue_paint_cans 4 3 42 = 24 := by
  sorry

end blue_paint_calculation_l1074_107495


namespace garden_breadth_l1074_107418

/-- Given a rectangular garden with perimeter 680 m and length 258 m, its breadth is 82 m -/
theorem garden_breadth (perimeter length breadth : ℝ) : 
  perimeter = 680 ∧ length = 258 ∧ perimeter = 2 * (length + breadth) → breadth = 82 := by
  sorry

end garden_breadth_l1074_107418


namespace sqrt_x_div_sqrt_y_equals_five_halves_l1074_107487

theorem sqrt_x_div_sqrt_y_equals_five_halves (x y : ℝ) 
  (h : ((2/3)^2 + (1/6)^2) / ((1/2)^2 + (1/7)^2) = 28*x/(25*y)) : 
  Real.sqrt x / Real.sqrt y = 5/2 := by
  sorry

end sqrt_x_div_sqrt_y_equals_five_halves_l1074_107487


namespace pump_fill_time_l1074_107440

/-- The time it takes to fill the tank with the leak present -/
def fill_time_with_leak : ℝ := 10

/-- The time it takes for the leak to empty a full tank -/
def empty_time : ℝ := 10

/-- The time it takes for the pump to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 5

theorem pump_fill_time :
  (1 / fill_time_without_leak - 1 / empty_time = 1 / fill_time_with_leak) →
  fill_time_without_leak = 5 := by
  sorry

end pump_fill_time_l1074_107440


namespace fraction_evaluation_l1074_107451

theorem fraction_evaluation (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 1) :
  6 / (a + b + c) = 1 := by
  sorry

end fraction_evaluation_l1074_107451


namespace sum_of_selected_elements_ge_one_l1074_107485

/-- Definition of the table element at position (i, j) -/
def table_element (i j : ℕ) : ℚ := 1 / (i + j - 1)

/-- A selection of n elements from an n × n table, where no two elements are in the same row or column -/
def valid_selection (n : ℕ) : Type := 
  { s : Finset (ℕ × ℕ) // s.card = n ∧ 
    (∀ (a b : ℕ × ℕ), a ∈ s → b ∈ s → a ≠ b → a.1 ≠ b.1 ∧ a.2 ≠ b.2) ∧
    (∀ (a : ℕ × ℕ), a ∈ s → a.1 ≤ n ∧ a.2 ≤ n) }

/-- The main theorem: The sum of selected elements is not less than 1 -/
theorem sum_of_selected_elements_ge_one (n : ℕ) (h : n > 0) :
  ∀ (s : valid_selection n), (s.val.sum (λ (x : ℕ × ℕ) => table_element x.1 x.2)) ≥ 1 := by
  sorry


end sum_of_selected_elements_ge_one_l1074_107485


namespace inequality_implies_upper_bound_l1074_107496

theorem inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 3| > a) → a < 5 := by
  sorry

end inequality_implies_upper_bound_l1074_107496


namespace cut_to_square_l1074_107457

/-- Represents a shape on a checkered paper --/
structure Shape :=
  (area : ℕ)
  (has_hole : Bool)

/-- Represents a square --/
def is_square (s : Shape) : Prop :=
  ∃ (side : ℕ), s.area = side * side ∧ s.has_hole = false

/-- Represents the ability to cut a shape into two parts --/
def can_cut (s : Shape) : Prop :=
  ∃ (part1 part2 : Shape), part1.area + part2.area = s.area

/-- Represents the ability to form a square from two parts --/
def can_form_square (part1 part2 : Shape) : Prop :=
  is_square (Shape.mk (part1.area + part2.area) false)

/-- The main theorem: given a shape with a hole, it can be cut into two parts
    that can form a square --/
theorem cut_to_square (s : Shape) (h : s.has_hole = true) :
  ∃ (part1 part2 : Shape),
    can_cut s ∧
    can_form_square part1 part2 :=
sorry

end cut_to_square_l1074_107457


namespace select_president_and_vice_president_l1074_107464

/-- The number of students in the classroom --/
def num_students : ℕ := 4

/-- The number of positions to be filled (president and vice president) --/
def num_positions : ℕ := 2

/-- Theorem stating the number of ways to select a class president and vice president --/
theorem select_president_and_vice_president :
  (num_students * (num_students - 1)) = 12 := by
  sorry

end select_president_and_vice_president_l1074_107464


namespace thirteenth_fib_is_610_l1074_107428

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The 13th Fibonacci number is 610 -/
theorem thirteenth_fib_is_610 : fib 13 = 610 := by
  sorry

end thirteenth_fib_is_610_l1074_107428


namespace anna_overall_score_l1074_107401

/-- Represents a test with a number of problems and a score percentage -/
structure Test where
  problems : ℕ
  score : ℚ
  h_score_range : 0 ≤ score ∧ score ≤ 1

/-- Calculates the number of problems answered correctly in a test -/
def correctProblems (t : Test) : ℚ :=
  t.problems * t.score

/-- Theorem stating that Anna's overall score across three tests is 78% -/
theorem anna_overall_score (test1 test2 test3 : Test)
  (h1 : test1.problems = 30 ∧ test1.score = 3/4)
  (h2 : test2.problems = 50 ∧ test2.score = 17/20)
  (h3 : test3.problems = 20 ∧ test3.score = 13/20) :
  (correctProblems test1 + correctProblems test2 + correctProblems test3) /
  (test1.problems + test2.problems + test3.problems) = 39/50 := by
  sorry

end anna_overall_score_l1074_107401


namespace product_inequality_l1074_107450

theorem product_inequality (a b x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1)
  (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hx₃ : 0 < x₃) (hx₄ : 0 < x₄) (hx₅ : 0 < x₅)
  (hx_prod : x₁ * x₂ * x₃ * x₄ * x₅ = 1) :
  (a * x₁ + b) * (a * x₂ + b) * (a * x₃ + b) * (a * x₄ + b) * (a * x₅ + b) ≥ 1 := by
  sorry

end product_inequality_l1074_107450


namespace coefficient_of_x_fourth_power_is_zero_l1074_107431

def expression (x : ℝ) : ℝ := 3 * (x^2 - x^4) - 5 * (x^4 - x^6 + x^2) + 4 * (2*x^4 - x^8)

theorem coefficient_of_x_fourth_power_is_zero :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, expression x = f x + 0 * x^4 :=
sorry

end coefficient_of_x_fourth_power_is_zero_l1074_107431


namespace train_passing_time_l1074_107471

/-- Given a train of length l traveling at constant velocity v, if the time to pass a platform
    of length 3l is 4 times the time to pass a pole, then the time to pass the pole is l/v. -/
theorem train_passing_time
  (l v : ℝ) -- Length of train and velocity
  (h_pos_l : l > 0)
  (h_pos_v : v > 0)
  (t : ℝ) -- Time to pass the pole
  (T : ℝ) -- Time to pass the platform
  (h_platform_time : T = 4 * t) -- Time to pass platform is 4 times time to pass pole
  (h_platform_length : 4 * l = v * T) -- Distance-velocity-time equation for platform
  : t = l / v := by
  sorry

end train_passing_time_l1074_107471


namespace average_first_five_subjects_l1074_107468

/-- Given a student's average marks and marks in the last subject, calculate the average of the first 5 subjects -/
theorem average_first_five_subjects 
  (total_subjects : Nat) 
  (average_all : ℚ) 
  (marks_last : ℚ) 
  (h1 : total_subjects = 6) 
  (h2 : average_all = 79) 
  (h3 : marks_last = 104) : 
  (average_all * total_subjects - marks_last) / (total_subjects - 1) = 74 := by
sorry

end average_first_five_subjects_l1074_107468


namespace least_palindrome_addition_l1074_107497

/-- A function that checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The starting number in our problem -/
def startNumber : ℕ := 250000

/-- The least number to be added to create a palindrome -/
def leastAddition : ℕ := 52

/-- Theorem stating that 52 is the least natural number that,
    when added to 250000, results in a palindrome -/
theorem least_palindrome_addition :
  (∀ k : ℕ, k < leastAddition → ¬isPalindrome (startNumber + k)) ∧
  isPalindrome (startNumber + leastAddition) := by sorry

end least_palindrome_addition_l1074_107497


namespace abs_3x_plus_5_not_positive_l1074_107437

theorem abs_3x_plus_5_not_positive (x : ℚ) : ¬(|3*x + 5| > 0) ↔ x = -5/3 := by
  sorry

end abs_3x_plus_5_not_positive_l1074_107437


namespace geometric_roots_difference_l1074_107427

theorem geometric_roots_difference (m n : ℝ) : 
  (∃ a r : ℝ, a = 1/2 ∧ r > 0 ∧ 
    (∀ x : ℝ, (x^2 - m*x + 2)*(x^2 - n*x + 2) = 0 ↔ 
      x = a ∨ x = a*r ∨ x = a*r^2 ∨ x = a*r^3)) →
  |m - n| = 3/2 := by sorry

end geometric_roots_difference_l1074_107427


namespace midpoint_chain_l1074_107447

/-- Given a line segment XY with midpoints as described, prove that XY = 80 when XJ = 5 -/
theorem midpoint_chain (X Y G H I J : ℝ) : 
  (G = (X + Y) / 2) →  -- G is midpoint of XY
  (H = (X + G) / 2) →  -- H is midpoint of XG
  (I = (X + H) / 2) →  -- I is midpoint of XH
  (J = (X + I) / 2) →  -- J is midpoint of XI
  (J - X = 5) →        -- XJ = 5
  (Y - X = 80) :=      -- XY = 80
by sorry

end midpoint_chain_l1074_107447
