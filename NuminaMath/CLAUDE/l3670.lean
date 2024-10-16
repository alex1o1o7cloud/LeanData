import Mathlib

namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3670_367051

/-- Given two concentric circles where a chord is tangent to the smaller circle,
    this theorem calculates the area of the region between the two circles. -/
theorem area_between_concentric_circles
  (outer_radius inner_radius chord_length : ℝ)
  (h_positive : 0 < inner_radius ∧ inner_radius < outer_radius)
  (h_tangent : inner_radius ^ 2 + (chord_length / 2) ^ 2 = outer_radius ^ 2)
  (h_chord : chord_length = 100) :
  (π * (outer_radius ^ 2 - inner_radius ^ 2) : ℝ) = 2000 * π :=
sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3670_367051


namespace NUMINAMATH_CALUDE_set_intersection_and_subset_l3670_367060

def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}

def B (a : ℝ) : Set ℝ := {x | (x - a) / (x - (a^2 + 1)) < 0}

theorem set_intersection_and_subset (a : ℝ) :
  (a = 2 → A a ∩ B a = {x | 2 < x ∧ x < 5}) ∧
  (B a ⊆ A a ↔ a ∈ Set.Icc (-1) (-1/2) ∪ Set.Icc 2 3) :=
sorry

end NUMINAMATH_CALUDE_set_intersection_and_subset_l3670_367060


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l3670_367004

theorem unique_integer_satisfying_conditions (x : ℤ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : 0 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + 2 < 4) : 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l3670_367004


namespace NUMINAMATH_CALUDE_minimum_coins_for_purchase_l3670_367061

def quarter : ℕ := 25
def dime : ℕ := 10
def nickel : ℕ := 5

def candy_bar : ℕ := 45
def chewing_gum : ℕ := 35
def chocolate_bar : ℕ := 65
def juice_pack : ℕ := 70
def cookies : ℕ := 80

def total_cost : ℕ := 2 * candy_bar + 3 * chewing_gum + chocolate_bar + 2 * juice_pack + cookies

theorem minimum_coins_for_purchase :
  ∃ (q d n : ℕ), 
    q * quarter + d * dime + n * nickel = total_cost ∧ 
    q + d + n = 20 ∧ 
    q = 19 ∧ 
    d = 0 ∧ 
    n = 1 ∧
    ∀ (q' d' n' : ℕ), 
      q' * quarter + d' * dime + n' * nickel = total_cost → 
      q' + d' + n' ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_minimum_coins_for_purchase_l3670_367061


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3670_367031

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 3*x = 0) ∧
  (∃ x : ℝ, x^2 - 4*x - 1 = 0) ∧
  (∀ x : ℝ, x^2 - 3*x = 0 ↔ (x = 0 ∨ x = 3)) ∧
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3670_367031


namespace NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_two_equals_i_l3670_367071

theorem complex_exp_thirteen_pi_over_two_equals_i :
  Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_two_equals_i_l3670_367071


namespace NUMINAMATH_CALUDE_total_selection_methods_l3670_367069

-- Define the number of candidate schools
def total_schools : ℕ := 8

-- Define the number of schools to be selected
def selected_schools : ℕ := 4

-- Define the number of schools for session A
def schools_in_session_A : ℕ := 2

-- Define the number of remaining sessions (B and C)
def remaining_sessions : ℕ := 2

-- Theorem to prove
theorem total_selection_methods :
  (total_schools.choose selected_schools) *
  (selected_schools.choose schools_in_session_A) *
  (remaining_sessions!) = 840 := by
  sorry

end NUMINAMATH_CALUDE_total_selection_methods_l3670_367069


namespace NUMINAMATH_CALUDE_power_equality_l3670_367052

/-- Given n ∈ ℕ, x = (1 + 1/n)^n, and y = (1 + 1/n)^(n+1), prove that x^y = y^x -/
theorem power_equality (n : ℕ) (x y : ℝ) 
  (hx : x = (1 + 1/n)^n) 
  (hy : y = (1 + 1/n)^(n+1)) : 
  x^y = y^x := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3670_367052


namespace NUMINAMATH_CALUDE_sarahs_earnings_proof_l3670_367020

/-- Sarah's earnings for an 8-hour day, given Connor's hourly wage and their wage ratio -/
def sarahs_daily_earnings (connors_hourly_wage : ℝ) (wage_ratio : ℝ) (hours_worked : ℝ) : ℝ :=
  connors_hourly_wage * wage_ratio * hours_worked

/-- Theorem stating Sarah's earnings for an 8-hour day -/
theorem sarahs_earnings_proof (connors_hourly_wage : ℝ) (wage_ratio : ℝ) (hours_worked : ℝ)
    (h1 : connors_hourly_wage = 7.20)
    (h2 : wage_ratio = 5)
    (h3 : hours_worked = 8) :
    sarahs_daily_earnings connors_hourly_wage wage_ratio hours_worked = 288 := by
  sorry

#eval sarahs_daily_earnings 7.20 5 8

end NUMINAMATH_CALUDE_sarahs_earnings_proof_l3670_367020


namespace NUMINAMATH_CALUDE_toy_spending_ratio_l3670_367006

def Trevor_spending : ℕ := 80
def total_spending : ℕ := 680
def years : ℕ := 4

def spending_ratio (Reed Quinn : ℕ) : Prop :=
  Reed = 2 * Quinn

theorem toy_spending_ratio :
  ∀ Reed Quinn : ℕ,
  (Trevor_spending = Reed + 20) →
  (∃ k : ℕ, Reed = k * Quinn) →
  (years * (Trevor_spending + Reed + Quinn) = total_spending) →
  spending_ratio Reed Quinn :=
by
  sorry

#check toy_spending_ratio

end NUMINAMATH_CALUDE_toy_spending_ratio_l3670_367006


namespace NUMINAMATH_CALUDE_isosceles_triangle_circle_centers_distance_l3670_367077

/-- For an isosceles triangle with circumradius R and inradius r, 
    the distance d between the centers of the circumscribed and inscribed circles 
    is given by d = √(R(R-2r)). -/
theorem isosceles_triangle_circle_centers_distance 
  (R r d : ℝ) 
  (h_R_pos : R > 0) 
  (h_r_pos : r > 0) 
  (h_isosceles : IsIsosceles) : 
  d = Real.sqrt (R * (R - 2 * r)) := by
  sorry

/-- Represents an isosceles triangle -/
structure IsIsosceles : Prop where
  -- Add necessary fields to represent an isosceles triangle
  -- This is left abstract as the problem doesn't provide specific details

#check isosceles_triangle_circle_centers_distance

end NUMINAMATH_CALUDE_isosceles_triangle_circle_centers_distance_l3670_367077


namespace NUMINAMATH_CALUDE_cakes_served_during_lunch_l3670_367002

theorem cakes_served_during_lunch :
  ∀ (lunch_cakes dinner_cakes : ℕ),
    dinner_cakes = 9 →
    dinner_cakes = lunch_cakes + 3 →
    lunch_cakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_during_lunch_l3670_367002


namespace NUMINAMATH_CALUDE_erik_money_left_l3670_367021

theorem erik_money_left (initial_money : ℕ) (bread_quantity : ℕ) (juice_quantity : ℕ) 
  (bread_price : ℕ) (juice_price : ℕ) (h1 : initial_money = 86) 
  (h2 : bread_quantity = 3) (h3 : juice_quantity = 3) (h4 : bread_price = 3) 
  (h5 : juice_price = 6) : 
  initial_money - (bread_quantity * bread_price + juice_quantity * juice_price) = 59 := by
  sorry

end NUMINAMATH_CALUDE_erik_money_left_l3670_367021


namespace NUMINAMATH_CALUDE_percentage_of_pine_trees_l3670_367087

theorem percentage_of_pine_trees (total_trees : ℕ) (non_pine_trees : ℕ) : 
  total_trees = 350 → non_pine_trees = 105 → 
  (((total_trees - non_pine_trees) : ℚ) / total_trees) * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_pine_trees_l3670_367087


namespace NUMINAMATH_CALUDE_lucy_flour_purchase_l3670_367072

/-- Calculates the amount of flour needed to replenish stock --/
def flour_to_buy (initial : ℕ) (used : ℕ) (full_bag : ℕ) : ℕ :=
  let remaining := initial - used
  let after_spill := remaining / 2
  full_bag - after_spill

/-- Theorem: Given the initial conditions, Lucy needs to buy 370g of flour --/
theorem lucy_flour_purchase :
  flour_to_buy 500 240 500 = 370 := by
  sorry

end NUMINAMATH_CALUDE_lucy_flour_purchase_l3670_367072


namespace NUMINAMATH_CALUDE_newspaper_buying_percentage_l3670_367050

def newspapers_bought : ℕ := 500
def selling_price : ℚ := 2
def percentage_sold : ℚ := 80 / 100
def profit : ℚ := 550

theorem newspaper_buying_percentage : 
  ∀ (buying_price : ℚ),
    (newspapers_bought : ℚ) * percentage_sold * selling_price - 
    (newspapers_bought : ℚ) * buying_price = profit →
    (selling_price - buying_price) / selling_price = 75 / 100 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_buying_percentage_l3670_367050


namespace NUMINAMATH_CALUDE_a_minus_b_greater_than_one_l3670_367038

theorem a_minus_b_greater_than_one (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hf : ∃ (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    (∀ x, x^3 + a*x^2 + 2*b*x - 1 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃))
  (hg : ∀ x, 2*x^2 + 2*b*x + a ≠ 0) : 
  a - b > 1 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_greater_than_one_l3670_367038


namespace NUMINAMATH_CALUDE_sqrt_simplification_l3670_367047

theorem sqrt_simplification : (Real.sqrt 2 * Real.sqrt 20) / Real.sqrt 5 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l3670_367047


namespace NUMINAMATH_CALUDE_carlos_has_largest_result_l3670_367094

def starting_number : ℕ := 12

def alice_result : ℕ := ((starting_number - 2) * 3) + 3

def ben_result : ℕ := ((starting_number * 3) - 2) + 3

def carlos_result : ℕ := (starting_number - 2 + 3) * 3

theorem carlos_has_largest_result :
  carlos_result > alice_result ∧ carlos_result > ben_result :=
sorry

end NUMINAMATH_CALUDE_carlos_has_largest_result_l3670_367094


namespace NUMINAMATH_CALUDE_dryer_sheet_box_cost_l3670_367037

/-- The cost of a box of dryer sheets -/
def box_cost (loads_per_week : ℕ) (sheets_per_load : ℕ) (sheets_per_box : ℕ) (yearly_savings : ℚ) : ℚ :=
  yearly_savings / (loads_per_week * 52 / sheets_per_box)

/-- Theorem stating the cost of a box of dryer sheets -/
theorem dryer_sheet_box_cost :
  box_cost 4 1 104 11 = (11/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_dryer_sheet_box_cost_l3670_367037


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3670_367012

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_sum1 : a 1 + a 3 = 6)
  (h_sum2 : (a 1 + a 2 + a 3 + a 4) + a 2 = (a 1 + a 2 + a 3) + 3)
  : q = (1 : ℝ) / 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3670_367012


namespace NUMINAMATH_CALUDE_initial_birds_count_l3670_367024

def birds_problem (initial_birds : ℕ) (landed_birds : ℕ) (total_birds : ℕ) : Prop :=
  initial_birds + landed_birds = total_birds

theorem initial_birds_count : ∃ (initial_birds : ℕ), 
  birds_problem initial_birds 8 20 ∧ initial_birds = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_count_l3670_367024


namespace NUMINAMATH_CALUDE_ratio_reduction_l3670_367063

theorem ratio_reduction (x : ℕ) (h : x ≥ 3) :
  (∃ a b : ℕ, a < b ∧ (6 - x : ℚ) / (7 - x) < a / b) ∧
  (∀ a b : ℕ, a < b → (6 - x : ℚ) / (7 - x) < a / b → 4 ≤ a) :=
sorry

end NUMINAMATH_CALUDE_ratio_reduction_l3670_367063


namespace NUMINAMATH_CALUDE_custom_mult_comm_custom_mult_comm_complex_l3670_367034

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem stating the commutativity of the custom multiplication -/
theorem custom_mult_comm (a b : ℝ) : custom_mult a b = custom_mult b a := by
  sorry

/-- Theorem stating the commutativity of the custom multiplication with a complex expression -/
theorem custom_mult_comm_complex (a b c : ℝ) : custom_mult a (b - c) = custom_mult (b - c) a := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_comm_custom_mult_comm_complex_l3670_367034


namespace NUMINAMATH_CALUDE_triangle_properties_and_heron_l3670_367064

/-- Triangle properties and Heron's formula -/
theorem triangle_properties_and_heron (r r_a r_b r_c p a b c S : ℝ) 
  (hr : r > 0) (hr_a : r_a > 0) (hr_b : r_b > 0) (hr_c : r_c > 0)
  (hp : p > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hS : S > 0)
  (h_semi_perimeter : p = (a + b + c) / 2)
  (h_inradius : r = S / p)
  (h_exradius_a : r_a = S / (p - a))
  (h_exradius_b : r_b = S / (p - b))
  (h_exradius_c : r_c = S / (p - c)) : 
  (r * p = r_a * (p - a)) ∧ 
  (r * r_a = (p - b) * (p - c)) ∧
  (r_b * r_c = p * (p - a)) ∧
  (S^2 = p * (p - a) * (p - b) * (p - c)) ∧
  (S^2 = r * r_a * r_b * r_c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_and_heron_l3670_367064


namespace NUMINAMATH_CALUDE_hexagon_triangle_ratio_l3670_367058

/-- A regular hexagon divided into six equal triangles -/
structure RegularHexagon where
  /-- The area of one of the six triangles -/
  s : ℝ
  /-- The area of a region formed by two adjacent triangles -/
  r : ℝ
  /-- The hexagon is divided into six equal triangles -/
  triangle_count : ℕ
  triangle_count_eq : triangle_count = 6
  /-- r is the area of two adjacent triangles -/
  r_eq : r = 2 * s

/-- The ratio of the area of two adjacent triangles to the area of one triangle in a regular hexagon is 2 -/
theorem hexagon_triangle_ratio (h : RegularHexagon) : r / s = 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_triangle_ratio_l3670_367058


namespace NUMINAMATH_CALUDE_existence_of_non_divisible_a_l3670_367083

theorem existence_of_non_divisible_a (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧
    ¬(p^2 ∣ a^(p-1) - 1) ∧
    ¬(p^2 ∣ (a+1)^(p-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_divisible_a_l3670_367083


namespace NUMINAMATH_CALUDE_arc_length_quarter_circle_l3670_367033

/-- Given a circle D with circumference 72 feet and an arc EF subtended by a central angle of 90°,
    prove that the length of arc EF is 18 feet. -/
theorem arc_length_quarter_circle (D : Real) (EF : Real) :
  D = 72 → -- Circumference of circle D is 72 feet
  EF = D / 4 → -- Arc EF is subtended by a 90° angle (1/4 of the circle)
  EF = 18 := by sorry

end NUMINAMATH_CALUDE_arc_length_quarter_circle_l3670_367033


namespace NUMINAMATH_CALUDE_roller_coaster_cost_l3670_367091

/-- The cost of a roller coaster ride in tickets, given the total number of tickets needed,
    the cost of a Ferris wheel ride, and the cost of a log ride. -/
theorem roller_coaster_cost
  (total_tickets : ℕ)
  (ferris_wheel_cost : ℕ)
  (log_ride_cost : ℕ)
  (h1 : total_tickets = 10)
  (h2 : ferris_wheel_cost = 2)
  (h3 : log_ride_cost = 1)
  : total_tickets - (ferris_wheel_cost + log_ride_cost) = 7 := by
  sorry

#check roller_coaster_cost

end NUMINAMATH_CALUDE_roller_coaster_cost_l3670_367091


namespace NUMINAMATH_CALUDE_sqrt_30_simplest_l3670_367040

/-- Predicate to check if a number is a perfect square --/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Predicate to check if a square root is in its simplest form --/
def IsSimplestSquareRoot (n : ℝ) : Prop :=
  ∃ m : ℕ, n = Real.sqrt m ∧ m > 0 ∧ ¬∃ k : ℕ, k > 1 ∧ IsPerfectSquare k ∧ k ∣ m

/-- Theorem stating that √30 is the simplest square root among the given options --/
theorem sqrt_30_simplest :
  IsSimplestSquareRoot (Real.sqrt 30) ∧
  ¬IsSimplestSquareRoot (Real.sqrt 0.1) ∧
  ¬IsSimplestSquareRoot (1/2 : ℝ) ∧
  ¬IsSimplestSquareRoot (Real.sqrt 18) :=
by sorry


end NUMINAMATH_CALUDE_sqrt_30_simplest_l3670_367040


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l3670_367089

def ends_in (n : ℕ) (m : ℕ) : Prop := n % 100 = m

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem smallest_number_with_conditions : 
  ∀ n : ℕ, 
    ends_in n 56 ∧ 
    n % 56 = 0 ∧ 
    digit_sum n = 56 →
    n ≥ 29899856 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l3670_367089


namespace NUMINAMATH_CALUDE_range_of_x_l3670_367019

theorem range_of_x (x : ℝ) : (1 / x < 3) ∧ (1 / x > -4) → x > 1/3 ∨ x < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l3670_367019


namespace NUMINAMATH_CALUDE_boat_license_combinations_count_l3670_367008

/-- Represents the set of allowed letters for boat licenses -/
def AllowedLetters : Finset Char := {'A', 'M', 'F'}

/-- Represents the set of allowed digits for boat licenses -/
def AllowedDigits : Finset Nat := Finset.range 10

/-- Calculates the number of possible boat license combinations -/
def BoatLicenseCombinations : Nat :=
  (Finset.card AllowedLetters) * (Finset.card AllowedDigits) ^ 5

/-- Theorem stating the number of possible boat license combinations -/
theorem boat_license_combinations_count :
  BoatLicenseCombinations = 300000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_combinations_count_l3670_367008


namespace NUMINAMATH_CALUDE_simplify_expression_l3670_367088

theorem simplify_expression (x : ℝ) : 4*x + 6*x^3 + 8 - (3 - 6*x^3 - 4*x) = 12*x^3 + 8*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3670_367088


namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_non_five_digit_palindrome_product_l3670_367035

/-- A function that checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- A function that checks if a number is a five-digit palindrome -/
def isFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

/-- The theorem statement -/
theorem smallest_three_digit_palindrome_non_five_digit_palindrome_product :
  isThreeDigitPalindrome 131 ∧
  ¬(isFiveDigitPalindrome (131 * 103)) ∧
  ∀ n : ℕ, isThreeDigitPalindrome n ∧ n < 131 → isFiveDigitPalindrome (n * 103) :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_non_five_digit_palindrome_product_l3670_367035


namespace NUMINAMATH_CALUDE_medals_count_l3670_367043

/-- The total number of medals displayed in the sports center -/
def total_medals (gold silver bronze : ℕ) : ℕ :=
  gold + silver + bronze

/-- Theorem: The total number of medals is 67 -/
theorem medals_count : total_medals 19 32 16 = 67 := by
  sorry

end NUMINAMATH_CALUDE_medals_count_l3670_367043


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3670_367090

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the theorem
theorem triangle_ABC_properties 
  (A B C : ℝ) 
  (h_triangle : triangle_ABC A B C)
  (h_cos_A : Real.cos A = -5/13)
  (h_cos_B : Real.cos B = 3/5)
  (h_BC : 5 = 5) :
  Real.sin C = 16/65 ∧ 
  5 * 5 * Real.sin C / 2 = 8/3 :=
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3670_367090


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l3670_367062

theorem roots_sum_of_squares (a b c : ℝ) (r s : ℝ) : 
  r^2 - (a+b)*r + ab + c = 0 → 
  s^2 - (a+b)*s + ab + c = 0 → 
  r^2 + s^2 = a^2 + b^2 - 2*c := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l3670_367062


namespace NUMINAMATH_CALUDE_regular_decagon_interior_angle_l3670_367018

/-- The measure of each interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle : ℝ := by
  -- Define the number of sides of a decagon
  let n : ℕ := 10

  -- Define the sum of interior angles formula
  let sum_of_interior_angles (sides : ℕ) : ℝ := (sides - 2) * 180

  -- Calculate the sum of interior angles for a decagon
  let total_angle_sum : ℝ := sum_of_interior_angles n

  -- Calculate the measure of one interior angle
  let interior_angle : ℝ := total_angle_sum / n

  -- Prove that the interior angle is 144 degrees
  sorry

end NUMINAMATH_CALUDE_regular_decagon_interior_angle_l3670_367018


namespace NUMINAMATH_CALUDE_caleb_spent_66_50_l3670_367084

/-- The total amount spent on hamburgers -/
def total_spent (total_burgers : ℕ) (single_cost double_cost : ℚ) (double_count : ℕ) : ℚ :=
  let single_count := total_burgers - double_count
  double_count * double_cost + single_count * single_cost

/-- Theorem stating that Caleb spent $66.50 on hamburgers -/
theorem caleb_spent_66_50 :
  total_spent 50 1 (3/2) 33 = 133/2 := by
  sorry

end NUMINAMATH_CALUDE_caleb_spent_66_50_l3670_367084


namespace NUMINAMATH_CALUDE_justin_and_tim_games_l3670_367097

theorem justin_and_tim_games (total_players : ℕ) (h1 : total_players = 8) :
  Nat.choose (total_players - 2) 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_justin_and_tim_games_l3670_367097


namespace NUMINAMATH_CALUDE_digit_200_of_17_over_70_is_2_l3670_367000

/-- The 200th digit after the decimal point in the decimal representation of 17/70 -/
def digit_200_of_17_over_70 : ℕ := 2

/-- Theorem stating that the 200th digit after the decimal point in 17/70 is 2 -/
theorem digit_200_of_17_over_70_is_2 :
  digit_200_of_17_over_70 = 2 := by sorry

end NUMINAMATH_CALUDE_digit_200_of_17_over_70_is_2_l3670_367000


namespace NUMINAMATH_CALUDE_last_painted_cell_l3670_367036

/-- Represents a cell in a rectangular grid --/
structure Cell where
  row : ℕ
  col : ℕ

/-- Represents a rectangular grid --/
structure Rectangle where
  rows : ℕ
  cols : ℕ

/-- Defines the spiral painting process --/
def spiralPaint (rect : Rectangle) : Cell :=
  sorry

/-- Theorem statement for the last painted cell in a 333 × 444 rectangle --/
theorem last_painted_cell :
  let rect : Rectangle := { rows := 333, cols := 444 }
  spiralPaint rect = { row := 167, col := 278 } :=
sorry

end NUMINAMATH_CALUDE_last_painted_cell_l3670_367036


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3670_367039

/-- Two real numbers are inversely proportional if their product is constant. -/
def InverselyProportional (x y : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_problem (x y : ℝ → ℝ) 
  (h1 : InverselyProportional x y)
  (h2 : x 1 = 36 ∧ y 1 = 4)
  (h3 : y 2 = 12) :
  x 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3670_367039


namespace NUMINAMATH_CALUDE_festival_attendance_l3670_367025

theorem festival_attendance (total : ℕ) (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) 
  (h_total : total = 2700)
  (h_day2 : day2 = day1 / 2)
  (h_day3 : day3 = 3 * day1)
  (h_sum : day1 + day2 + day3 = total) :
  day2 = 300 := by
sorry

end NUMINAMATH_CALUDE_festival_attendance_l3670_367025


namespace NUMINAMATH_CALUDE_twenty_in_base_five_l3670_367057

/-- Converts a decimal number to its base-5 representation -/
def to_base_five (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid base-5 number -/
def is_valid_base_five (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d < 5

/-- Converts a list of base-5 digits to its decimal value -/
def from_base_five (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 5 + d) 0

theorem twenty_in_base_five :
  to_base_five 20 = [4, 0] ∧
  is_valid_base_five [4, 0] ∧
  from_base_five [4, 0] = 20 :=
sorry

end NUMINAMATH_CALUDE_twenty_in_base_five_l3670_367057


namespace NUMINAMATH_CALUDE_tetrahedron_circumscribed_sphere_area_l3670_367014

theorem tetrahedron_circumscribed_sphere_area (edge_length : ℝ) : 
  edge_length = 4 → 
  ∃ (sphere_area : ℝ), sphere_area = 24 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_circumscribed_sphere_area_l3670_367014


namespace NUMINAMATH_CALUDE_pie_chart_highlights_part_whole_l3670_367099

/-- Enumeration of statistical graph types --/
inductive StatisticalGraph
  | BarGraph
  | PieChart
  | LineGraph
  | FrequencyDistributionHistogram

/-- Function to determine if a graph type highlights part-whole relationships --/
def highlights_part_whole_relationship (graph : StatisticalGraph) : Prop :=
  match graph with
  | StatisticalGraph.PieChart => True
  | _ => False

/-- Theorem stating that the Pie chart is the graph that highlights part-whole relationships --/
theorem pie_chart_highlights_part_whole :
  ∀ (graph : StatisticalGraph),
    highlights_part_whole_relationship graph ↔ graph = StatisticalGraph.PieChart :=
by sorry

end NUMINAMATH_CALUDE_pie_chart_highlights_part_whole_l3670_367099


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l3670_367068

/-- Given a tank that requires different numbers of buckets to fill based on bucket capacity,
    this theorem proves the relationship between the original and reduced bucket capacities. -/
theorem bucket_capacity_reduction (original_buckets reduced_buckets : ℕ) 
  (h1 : original_buckets = 10)
  (h2 : reduced_buckets = 25)
  : (original_buckets : ℚ) / reduced_buckets = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l3670_367068


namespace NUMINAMATH_CALUDE_son_age_is_eleven_l3670_367054

/-- Represents the ages of a mother and son -/
structure FamilyAges where
  son : ℕ
  mother : ℕ

/-- The conditions of the age problem -/
def AgeProblemConditions (ages : FamilyAges) : Prop :=
  (ages.son + ages.mother = 55) ∧ 
  (ages.son - 3 + ages.mother - 3 = 49) ∧
  (ages.mother = 4 * ages.son)

/-- The theorem stating that under the given conditions, the son's age is 11 -/
theorem son_age_is_eleven (ages : FamilyAges) 
  (h : AgeProblemConditions ages) : ages.son = 11 := by
  sorry

end NUMINAMATH_CALUDE_son_age_is_eleven_l3670_367054


namespace NUMINAMATH_CALUDE_line_intersection_parameter_range_l3670_367065

/-- Given two points A and B, and a line that intersects the line segment AB,
    this theorem proves the range of the parameter m in the line equation. -/
theorem line_intersection_parameter_range :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (2, -1)
  let line (m : ℝ) (x y : ℝ) := x - 2*y + m = 0
  ∀ m : ℝ, (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    line m ((1-t)*A.1 + t*B.1) ((1-t)*A.2 + t*B.2)) ↔ 
  -4 ≤ m ∧ m ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_parameter_range_l3670_367065


namespace NUMINAMATH_CALUDE_probability_six_distinct_numbers_l3670_367030

theorem probability_six_distinct_numbers (n : ℕ) (h : n = 6) :
  (Nat.factorial n : ℚ) / (n ^ n : ℚ) = 5 / 324 := by
  sorry

end NUMINAMATH_CALUDE_probability_six_distinct_numbers_l3670_367030


namespace NUMINAMATH_CALUDE_area_units_order_l3670_367007

/-- An enumeration of area units -/
inductive AreaUnit
  | SquareKilometer
  | Hectare
  | SquareMeter
  | SquareDecimeter
  | SquareCentimeter

/-- A function to compare two area units -/
def areaUnitLarger (a b : AreaUnit) : Prop :=
  match a, b with
  | AreaUnit.SquareKilometer, _ => a ≠ b
  | AreaUnit.Hectare, AreaUnit.SquareKilometer => False
  | AreaUnit.Hectare, _ => a ≠ b
  | AreaUnit.SquareMeter, AreaUnit.SquareKilometer => False
  | AreaUnit.SquareMeter, AreaUnit.Hectare => False
  | AreaUnit.SquareMeter, _ => a ≠ b
  | AreaUnit.SquareDecimeter, AreaUnit.SquareCentimeter => True
  | AreaUnit.SquareDecimeter, _ => False
  | AreaUnit.SquareCentimeter, _ => False

/-- Theorem stating the correct order of area units from largest to smallest -/
theorem area_units_order :
  areaUnitLarger AreaUnit.SquareKilometer AreaUnit.Hectare ∧
  areaUnitLarger AreaUnit.Hectare AreaUnit.SquareMeter ∧
  areaUnitLarger AreaUnit.SquareMeter AreaUnit.SquareDecimeter ∧
  areaUnitLarger AreaUnit.SquareDecimeter AreaUnit.SquareCentimeter :=
sorry

end NUMINAMATH_CALUDE_area_units_order_l3670_367007


namespace NUMINAMATH_CALUDE_cart_distance_theorem_l3670_367048

/-- Represents a cart with two wheels -/
structure Cart where
  front_wheel_circumference : ℝ
  back_wheel_circumference : ℝ

/-- Calculates the distance traveled by the cart -/
def distance_traveled (c : Cart) (back_wheel_revolutions : ℝ) : ℝ :=
  c.back_wheel_circumference * back_wheel_revolutions

theorem cart_distance_theorem (c : Cart) (back_wheel_revolutions : ℝ) :
  c.front_wheel_circumference = 30 →
  c.back_wheel_circumference = 33 →
  c.front_wheel_circumference * (back_wheel_revolutions + 5) = c.back_wheel_circumference * back_wheel_revolutions →
  distance_traveled c back_wheel_revolutions = 1650 := by
  sorry

#check cart_distance_theorem

end NUMINAMATH_CALUDE_cart_distance_theorem_l3670_367048


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3670_367015

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n : ℝ) * (a 1 + a n) / 2) →
  (∀ k m, a (k + m) - a k = m * (a 2 - a 1)) →
  ((a 2 - 1)^3 + 5*(a 2 - 1) = 1) →
  ((a 2010 - 1)^3 + 5*(a 2010 - 1) = -1) →
  (a 2 + a 2010 = 2 ∧ S 2011 = 2011) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3670_367015


namespace NUMINAMATH_CALUDE_poly_properties_l3670_367066

/-- The polynomial under consideration -/
def p (x y : ℝ) : ℝ := 2*x*y - x^2*y + 3*x^3*y - 5

/-- The degree of a term in a polynomial of two variables -/
def term_degree (a b : ℕ) : ℕ := a + b

/-- The degree of the polynomial p -/
def poly_degree : ℕ := 4

/-- The number of terms in the polynomial p -/
def num_terms : ℕ := 4

theorem poly_properties :
  (∃ x y : ℝ, term_degree 3 1 = poly_degree ∧ p x y ≠ 0) ∧
  num_terms = 4 :=
sorry

end NUMINAMATH_CALUDE_poly_properties_l3670_367066


namespace NUMINAMATH_CALUDE_adjacent_semicircles_perimeter_l3670_367041

/-- The perimeter of a shape formed by two adjacent semicircles with radius 1 --/
theorem adjacent_semicircles_perimeter :
  ∀ (r : ℝ), r = 1 →
  ∃ (perimeter : ℝ), perimeter = 3 * r :=
by sorry

end NUMINAMATH_CALUDE_adjacent_semicircles_perimeter_l3670_367041


namespace NUMINAMATH_CALUDE_rosie_pies_calculation_l3670_367003

-- Define the function that calculates the number of pies
def pies_from_apples (apples_per_3_pies : ℕ) (available_apples : ℕ) : ℕ :=
  (available_apples * 3) / apples_per_3_pies

-- Theorem statement
theorem rosie_pies_calculation :
  pies_from_apples 12 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_calculation_l3670_367003


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l3670_367005

/-- Represents a cube with smaller cubes removed from its corners -/
structure ModifiedCube where
  originalSideLength : ℕ
  removedCubeSideLength : ℕ

/-- Calculates the number of edges in the modified cube -/
def edgeCount (cube : ModifiedCube) : ℕ :=
  12 * 3 -- Each original edge is divided into 3 segments

/-- Theorem stating that a cube of side length 4 with cubes of side length 2 removed from each corner has 36 edges -/
theorem modified_cube_edge_count :
  ∀ (cube : ModifiedCube),
    cube.originalSideLength = 4 →
    cube.removedCubeSideLength = 2 →
    edgeCount cube = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l3670_367005


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3670_367096

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  n : ℕ -- number of terms
  d : ℝ -- common difference
  a₁ : ℝ -- first term

/-- Sum of magnitudes of terms in an arithmetic sequence -/
def sumOfMagnitudes (seq : ArithmeticSequence) : ℝ := 
  sorry

/-- New sequence obtained by adding a constant to all terms -/
def addConstant (seq : ArithmeticSequence) (c : ℝ) : ArithmeticSequence :=
  sorry

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  sumOfMagnitudes seq = 250 ∧
  sumOfMagnitudes (addConstant seq 1) = 250 ∧
  sumOfMagnitudes (addConstant seq 2) = 250 →
  seq.n^2 * seq.d = 1000 ∨ seq.n^2 * seq.d = -1000 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3670_367096


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3670_367078

theorem arithmetic_sequence_common_difference
  (a b c : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_arithmetic : ∃ (d : ℝ), ∃ (m n k : ℤ), a = a + m * d ∧ b = a + n * d ∧ c = a + k * d)
  (h_equation : (c - b) / a + (a - c) / b + (b - a) / c = 0) :
  ∃ (d : ℝ), d = 0 ∧ ∃ (m n k : ℤ), a = a + m * d ∧ b = a + n * d ∧ c = a + k * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3670_367078


namespace NUMINAMATH_CALUDE_modular_congruence_existence_l3670_367046

theorem modular_congruence_existence (a c : ℕ+) (b : ℤ) :
  ∃ x : ℕ+, (c : ℤ) ∣ ((a : ℤ)^(x : ℕ) + x - b) := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_existence_l3670_367046


namespace NUMINAMATH_CALUDE_probability_of_winning_all_games_l3670_367027

def number_of_games : ℕ := 6
def probability_of_winning_single_game : ℚ := 3/5

theorem probability_of_winning_all_games :
  (probability_of_winning_single_game ^ number_of_games : ℚ) = 729/15625 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_winning_all_games_l3670_367027


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3670_367017

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 864 → volume = 1728 → 
  (∃ (side_length : ℝ), 
    surface_area = 6 * side_length^2 ∧ 
    volume = side_length^3) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3670_367017


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3670_367085

theorem rectangle_dimension_change (original_length original_width : ℝ) 
  (new_length new_width : ℝ) (h_positive : original_length > 0 ∧ original_width > 0) :
  new_width = 1.5 * original_width ∧ 
  original_length * original_width = new_length * new_width →
  (original_length - new_length) / original_length = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3670_367085


namespace NUMINAMATH_CALUDE_liars_count_l3670_367023

/-- Represents the type of inhabitant: Knight or Liar -/
inductive InhabitantType
| Knight
| Liar

/-- Represents an island in the Tenth Kingdom -/
structure Island where
  population : Nat
  knights : Nat

/-- Represents the Tenth Kingdom -/
structure TenthKingdom where
  islands : List Island
  total_islands : Nat
  inhabitants_per_island : Nat

/-- Predicate for islands where everyone answered "Yes" to the first question -/
def first_question_yes (i : Island) : Prop :=
  i.knights = i.population / 2

/-- Predicate for islands where everyone answered "No" to the first question -/
def first_question_no (i : Island) : Prop :=
  i.knights ≠ i.population / 2

/-- Predicate for islands where everyone answered "No" to the second question -/
def second_question_no (i : Island) : Prop :=
  i.knights ≥ i.population / 2

/-- Predicate for islands where everyone answered "Yes" to the second question -/
def second_question_yes (i : Island) : Prop :=
  i.knights < i.population / 2

/-- Main theorem: The number of liars in the Tenth Kingdom is 1013 -/
theorem liars_count (k : TenthKingdom) : Nat := by
  sorry

/-- The Tenth Kingdom setup -/
def tenth_kingdom : TenthKingdom := {
  islands := [],  -- Placeholder for the list of islands
  total_islands := 17,
  inhabitants_per_island := 119
}

#check liars_count tenth_kingdom

end NUMINAMATH_CALUDE_liars_count_l3670_367023


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3670_367067

-- Problem 1
theorem problem_1 : 
  2 * Real.cos (π / 4) + (3 - Real.pi) ^ 0 - |2 - Real.sqrt 8| - (-1/3)⁻¹ = 6 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : 
  let x : ℝ := Real.sqrt 27 + |-2| - 3 * Real.tan (π / 3)
  ((x^2 - 1) / (x^2 - 2*x + 1) - 1 / (x - 1)) / ((x + 2) / (x - 1)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3670_367067


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l3670_367055

/-- 
A parallelogram has side lengths of 10, 12, 10y-2, and 4x+6. 
This theorem proves that x+y = 2.7.
-/
theorem parallelogram_side_sum (x y : ℝ) : 
  (4*x + 6 = 12) → (10*y - 2 = 10) → x + y = 2.7 := by sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l3670_367055


namespace NUMINAMATH_CALUDE_angle_conversion_l3670_367042

def angle : Real := 54.12

theorem angle_conversion (ε : Real) (h : ε > 0) :
  ∃ (d : ℕ) (m : ℕ) (s : ℕ),
    d = 54 ∧ m = 7 ∧ s = 12 ∧ 
    abs (angle - (d : Real) - (m : Real) / 60 - (s : Real) / 3600) < ε :=
by sorry

end NUMINAMATH_CALUDE_angle_conversion_l3670_367042


namespace NUMINAMATH_CALUDE_first_half_speed_l3670_367032

/-- Proves that given a 6-hour journey where the second half is traveled at 48 kmph
    and the total distance is 324 km, the speed during the first half must be 60 kmph. -/
theorem first_half_speed (total_time : ℝ) (second_half_speed : ℝ) (total_distance : ℝ)
    (h1 : total_time = 6)
    (h2 : second_half_speed = 48)
    (h3 : total_distance = 324) :
    let first_half_time := total_time / 2
    let second_half_time := total_time / 2
    let second_half_distance := second_half_speed * second_half_time
    let first_half_distance := total_distance - second_half_distance
    let first_half_speed := first_half_distance / first_half_time
    first_half_speed = 60 := by
  sorry

#check first_half_speed

end NUMINAMATH_CALUDE_first_half_speed_l3670_367032


namespace NUMINAMATH_CALUDE_triangle_area_l3670_367098

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the area of the triangle is 3 under the given conditions. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  a = Real.sqrt 5 →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  (1 / 2) * a * c * Real.sin B = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3670_367098


namespace NUMINAMATH_CALUDE_bakery_boxes_l3670_367092

theorem bakery_boxes (total_muffins : ℕ) (muffins_per_box : ℕ) (additional_boxes : ℕ) 
  (h1 : total_muffins = 95)
  (h2 : muffins_per_box = 5)
  (h3 : additional_boxes = 9) :
  total_muffins / muffins_per_box - additional_boxes = 10 :=
by sorry

end NUMINAMATH_CALUDE_bakery_boxes_l3670_367092


namespace NUMINAMATH_CALUDE_time_spent_studying_l3670_367082

/-- Represents the time allocation in a day --/
structure DayTime where
  total : ℝ
  tv : ℝ
  exercise : ℝ
  socialMedia : ℝ
  study : ℝ

/-- Theorem stating the time spent studying given the conditions --/
theorem time_spent_studying (d : DayTime) : 
  d.total = 1440 ∧ 
  d.tv = d.total / 5 ∧ 
  d.exercise = d.total / 8 ∧ 
  d.socialMedia = (d.total - d.tv - d.exercise) / 6 ∧ 
  d.study = (d.total - d.tv - d.exercise - d.socialMedia) / 4 →
  d.study = 202.5 := by
  sorry


end NUMINAMATH_CALUDE_time_spent_studying_l3670_367082


namespace NUMINAMATH_CALUDE_count_odd_integers_between_fractions_l3670_367056

theorem count_odd_integers_between_fractions :
  let lower_bound : ℚ := 17 / 4
  let upper_bound : ℚ := 35 / 2
  (Finset.filter (fun n => n % 2 = 1)
    (Finset.Icc (Int.ceil lower_bound) (Int.floor upper_bound))).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_count_odd_integers_between_fractions_l3670_367056


namespace NUMINAMATH_CALUDE_square_sum_equals_b_times_ab_plus_two_l3670_367028

theorem square_sum_equals_b_times_ab_plus_two
  (x y a b : ℝ) 
  (h1 : x * y = b) 
  (h2 : 1 / (x^2) + 1 / (y^2) = a) : 
  (x + y)^2 = b * (a * b + 2) := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_b_times_ab_plus_two_l3670_367028


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3670_367026

/-- Given a quadratic function y = x^2 - 2bx + b^2 + c whose graph intersects
    the line y = 1 - x at only one point, and its vertex is on the graph of
    y = ax^2 (a ≠ 0), prove that the range of values for a is a ≥ -1/5 and a ≠ 0. -/
theorem quadratic_function_range (b c : ℝ) (a : ℝ) 
  (h1 : ∃! x, x^2 - 2*b*x + b^2 + c = 1 - x) 
  (h2 : c = a * b^2) 
  (h3 : a ≠ 0) : 
  a ≥ -1/5 ∧ a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3670_367026


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l3670_367053

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  students_per_group : ℕ
  interval : ℕ

/-- Creates a systematic sampling for the given problem. -/
def create_sampling : SystematicSampling :=
  { total_students := 50
  , num_groups := 10
  , students_per_group := 5
  , interval := 10
  }

/-- Calculates the number drawn from a specific group given the number drawn from another group. -/
def calculate_number (s : SystematicSampling) (known_group : ℕ) (known_number : ℕ) (target_group : ℕ) : ℕ :=
  known_number + (target_group - known_group) * s.interval

/-- Theorem stating that if the number drawn from the third group is 13, 
    then the number drawn from the seventh group is 53. -/
theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s = create_sampling) 
  (h2 : calculate_number s 3 13 7 = 53) : 
  calculate_number s 3 13 7 = 53 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l3670_367053


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3670_367049

theorem arithmetic_geometric_sequence (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (2 * b = a + c) →  -- arithmetic sequence condition
  (b ^ 2 = c * (a + 1)) →  -- geometric sequence condition when a is increased by 1
  (b ^ 2 = a * (c + 2)) →  -- geometric sequence condition when c is increased by 2
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3670_367049


namespace NUMINAMATH_CALUDE_constant_sum_implies_parallelogram_l3670_367029

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Additional condition to ensure convexity

-- Define a function to calculate the distance from a point to a line
def distanceToLine (p : ℝ × ℝ) (l : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

-- Define a function to check if a point is inside the quadrilateral
def isInsideQuadrilateral (q : ConvexQuadrilateral) (p : ℝ × ℝ) : Prop := sorry

-- Define a function to calculate the sum of distances from a point to all sides
def sumOfDistances (q : ConvexQuadrilateral) (p : ℝ × ℝ) : ℝ :=
  distanceToLine p (q.vertices 0, q.vertices 1) +
  distanceToLine p (q.vertices 1, q.vertices 2) +
  distanceToLine p (q.vertices 2, q.vertices 3) +
  distanceToLine p (q.vertices 3, q.vertices 0)

-- Define what it means for a quadrilateral to be a parallelogram
def isParallelogram (q : ConvexQuadrilateral) : Prop := sorry

-- The main theorem
theorem constant_sum_implies_parallelogram (q : ConvexQuadrilateral) :
  (∃ k : ℝ, ∀ p : ℝ × ℝ, isInsideQuadrilateral q p → sumOfDistances q p = k) →
  isParallelogram q := by sorry

end NUMINAMATH_CALUDE_constant_sum_implies_parallelogram_l3670_367029


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3670_367095

/-- The x-coordinates of the intersection points between a circle and a line -/
theorem circle_line_intersection
  (x1 y1 x2 y2 : ℝ)  -- Endpoints of the circle's diameter
  (m b : ℝ)  -- Line equation coefficients (y = mx + b)
  (h_distinct : (x1, y1) ≠ (x2, y2))  -- Ensure distinct endpoints
  (h_line : m = -1/2 ∧ b = 5)  -- Specific line equation
  (h_endpoints : x1 = 2 ∧ y1 = 4 ∧ x2 = 10 ∧ y2 = 8)  -- Specific endpoint coordinates
  : ∃ (x_left x_right : ℝ),
    x_left = 4.4 - 2.088 ∧
    x_right = 4.4 + 2.088 ∧
    (∀ (x y : ℝ),
      (x - (x1 + x2)/2)^2 + (y - (y1 + y2)/2)^2 = ((x2 - x1)^2 + (y2 - y1)^2)/4 ∧
      y = m * x + b →
      x = x_left ∨ x = x_right) :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3670_367095


namespace NUMINAMATH_CALUDE_max_value_expression_l3670_367076

theorem max_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, 3 * (a - x) * (x + Real.sqrt (x^2 + 2 * b^2)) ≤ a^2 + 3 * b^2) ∧
  (∃ x : ℝ, 3 * (a - x) * (x + Real.sqrt (x^2 + 2 * b^2)) = a^2 + 3 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3670_367076


namespace NUMINAMATH_CALUDE_final_b_value_l3670_367013

def program_execution (a b c : Int) : Int :=
  let a' := b
  let b' := c
  b'

theorem final_b_value :
  ∀ (a b c : Int),
  a = 3 →
  b = -5 →
  c = 8 →
  program_execution a b c = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_b_value_l3670_367013


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l3670_367022

/-- Given a quadratic expression 3x^2 + 9x + 17, when written in the form a(x-h)^2 + k, h = -3/2 -/
theorem quadratic_form_h_value : 
  ∃ (a k : ℝ), ∀ x : ℝ, 3*x^2 + 9*x + 17 = a*(x - (-3/2))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l3670_367022


namespace NUMINAMATH_CALUDE_min_value_theorem_l3670_367045

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (y / (2*x) + 1 / y) ≥ 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3670_367045


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l3670_367074

theorem binomial_expansion_example : 7^3 + 3*(7^2)*2 + 3*7*(2^2) + 2^3 = (7 + 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l3670_367074


namespace NUMINAMATH_CALUDE_quadratic_translation_problem_solution_l3670_367011

/-- Represents a horizontal and vertical translation of a quadratic function -/
structure Translation where
  horizontal : ℝ
  vertical : ℝ

/-- Applies a translation to a quadratic function -/
def apply_translation (f : ℝ → ℝ) (t : Translation) : ℝ → ℝ :=
  λ x => f (x + t.horizontal) - t.vertical

theorem quadratic_translation (a : ℝ) (t : Translation) :
  apply_translation (λ x => a * x^2) t =
  λ x => a * (x + t.horizontal)^2 - t.vertical := by
  sorry

/-- The specific translation in the problem -/
def problem_translation : Translation :=
  { horizontal := 3, vertical := 2 }

theorem problem_solution :
  apply_translation (λ x => 2 * x^2) problem_translation =
  λ x => 2 * (x + 3)^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_translation_problem_solution_l3670_367011


namespace NUMINAMATH_CALUDE_marias_stamp_collection_l3670_367001

/-- The problem of calculating Maria's stamp collection increase -/
theorem marias_stamp_collection 
  (current_stamps : ℕ) 
  (increase_percentage : ℚ) 
  (h1 : current_stamps = 40)
  (h2 : increase_percentage = 20 / 100) : 
  current_stamps + (increase_percentage * current_stamps).floor = 48 := by
  sorry

end NUMINAMATH_CALUDE_marias_stamp_collection_l3670_367001


namespace NUMINAMATH_CALUDE_distribute_four_balls_three_boxes_l3670_367009

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 3 ways to distribute 4 indistinguishable balls into 3 distinguishable boxes,
    with each box containing at least one ball -/
theorem distribute_four_balls_three_boxes :
  distribute_balls 4 3 = 3 := by sorry

end NUMINAMATH_CALUDE_distribute_four_balls_three_boxes_l3670_367009


namespace NUMINAMATH_CALUDE_average_weight_increase_l3670_367010

/-- Proves that replacing a person weighing 65 kg with a person weighing 77 kg
    in a group of 8 people increases the average weight by 1.5 kg. -/
theorem average_weight_increase (initial_average : ℝ) :
  let initial_total := 8 * initial_average
  let new_total := initial_total - 65 + 77
  let new_average := new_total / 8
  new_average - initial_average = 1.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3670_367010


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l3670_367075

/-- Given two concentric circles D and C, where C is inside D, 
    this theorem proves the diameter of C when the ratio of 
    the area between the circles to the area of C is 4:1 -/
theorem concentric_circles_area_ratio (d_diameter : ℝ) 
  (h_d_diameter : d_diameter = 24) 
  (c_diameter : ℝ) 
  (h_inside : c_diameter < d_diameter) 
  (h_ratio : (π * (d_diameter/2)^2 - π * (c_diameter/2)^2) / (π * (c_diameter/2)^2) = 4) :
  c_diameter = 24 * Real.sqrt 5 / 5 := by
  sorry

#check concentric_circles_area_ratio

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l3670_367075


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3670_367093

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_ab : a + b = 35)
  (sum_bc : b + c = 57)
  (sum_ca : c + a = 62) :
  a + b + c = 77 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3670_367093


namespace NUMINAMATH_CALUDE_banana_count_l3670_367070

theorem banana_count (bunches_of_eight : Nat) (bananas_per_bunch_eight : Nat)
                     (bunches_of_seven : Nat) (bananas_per_bunch_seven : Nat) :
  bunches_of_eight = 6 →
  bananas_per_bunch_eight = 8 →
  bunches_of_seven = 5 →
  bananas_per_bunch_seven = 7 →
  bunches_of_eight * bananas_per_bunch_eight + bunches_of_seven * bananas_per_bunch_seven = 83 :=
by sorry

end NUMINAMATH_CALUDE_banana_count_l3670_367070


namespace NUMINAMATH_CALUDE_sum_of_ages_l3670_367081

/-- Represents the ages of Markus, his son, and his grandson -/
structure FamilyAges where
  markus : ℕ
  son : ℕ
  grandson : ℕ

/-- Defines the conditions for the family ages -/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.markus = 2 * ages.son ∧
  ages.son = 2 * ages.grandson ∧
  ages.grandson = 20

/-- Theorem stating that the sum of ages is 140 for a valid family age structure -/
theorem sum_of_ages (ages : FamilyAges) (h : validFamilyAges ages) : 
  ages.markus + ages.son + ages.grandson = 140 := by
  sorry

#check sum_of_ages

end NUMINAMATH_CALUDE_sum_of_ages_l3670_367081


namespace NUMINAMATH_CALUDE_repeating_pattern_is_125_l3670_367080

def recurring_decimal : ℚ := 0.125125125125125

theorem repeating_pattern_is_125 : 
  ∃ (n : ℕ), recurring_decimal * 10^n - recurring_decimal * 10^(n-3) = 125 / 1000 :=
sorry

end NUMINAMATH_CALUDE_repeating_pattern_is_125_l3670_367080


namespace NUMINAMATH_CALUDE_f_value_at_3_l3670_367086

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 3

-- State the theorem
theorem f_value_at_3 (a b c : ℝ) : f a b c (-3) = 7 → f a b c 3 = -13 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l3670_367086


namespace NUMINAMATH_CALUDE_simplify_expression_l3670_367079

theorem simplify_expression (a b : ℝ) (h1 : a + b ≠ 0) (h2 : a - 2*b ≠ 0) (h3 : a^2 - b^2 ≠ 0) (h4 : a^2 - 4*a*b + 4*b^2 ≠ 0) :
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3670_367079


namespace NUMINAMATH_CALUDE_sqrt_2_simplest_l3670_367044

def is_simplest_sqrt (x : ℝ) (others : List ℝ) : Prop :=
  ∀ y ∈ others, ¬∃ (n : ℕ) (r : ℝ), n > 1 ∧ y = n * Real.sqrt r

theorem sqrt_2_simplest : is_simplest_sqrt (Real.sqrt 2) [Real.sqrt 8, Real.sqrt 12, Real.sqrt 18] := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_simplest_l3670_367044


namespace NUMINAMATH_CALUDE_apple_sales_theorem_l3670_367073

/-- Represents the sales of apples over three days in a store. -/
structure AppleSales where
  day1 : ℝ  -- Sales on day 1 in kg
  day2 : ℝ  -- Sales on day 2 in kg
  day3 : ℝ  -- Sales on day 3 in kg

/-- The conditions of the apple sales problem. -/
def appleSalesProblem (s : AppleSales) : Prop :=
  s.day2 = s.day1 / 4 + 8 ∧
  s.day3 = s.day2 / 4 + 8 ∧
  s.day3 = 18

/-- The theorem stating that if the conditions are met, 
    the sales on the first day were 128 kg. -/
theorem apple_sales_theorem (s : AppleSales) :
  appleSalesProblem s → s.day1 = 128 := by
  sorry

#check apple_sales_theorem

end NUMINAMATH_CALUDE_apple_sales_theorem_l3670_367073


namespace NUMINAMATH_CALUDE_tenth_power_sum_l3670_367016

theorem tenth_power_sum (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) : a^10 + b^10 = 93 := by
  sorry

end NUMINAMATH_CALUDE_tenth_power_sum_l3670_367016


namespace NUMINAMATH_CALUDE_identify_counterfeit_coin_l3670_367059

/-- Represents the result of a weighing -/
inductive WeighResult
  | Left  : WeighResult  -- Left pan is heavier
  | Right : WeighResult  -- Right pan is heavier
  | Equal : WeighResult  -- Pans are balanced

/-- Represents a coin -/
inductive Coin
  | A : Coin
  | B : Coin
  | C : Coin
  | D : Coin

/-- Represents the state of a coin -/
inductive CoinState
  | Genuine : CoinState
  | Counterfeit : CoinState

/-- Represents whether the counterfeit coin is heavier or lighter -/
inductive CounterfeitWeight
  | Heavier : CounterfeitWeight
  | Lighter : CounterfeitWeight

/-- Function to perform a weighing -/
def weigh (left : List Coin) (right : List Coin) : WeighResult := sorry

/-- Function to determine the state of a coin -/
def determineCoinState (c : Coin) : CoinState := sorry

/-- Function to determine if the counterfeit coin is heavier or lighter -/
def determineCounterfeitWeight : CounterfeitWeight := sorry

/-- Theorem stating that the counterfeit coin can be identified in at most 3 weighings -/
theorem identify_counterfeit_coin :
  ∃ (counterfeit : Coin) (weight : CounterfeitWeight),
    (∀ c : Coin, c ≠ counterfeit → determineCoinState c = CoinState.Genuine) ∧
    (determineCoinState counterfeit = CoinState.Counterfeit) ∧
    (weight = determineCounterfeitWeight) ∧
    (∃ (w₁ w₂ w₃ : WeighResult),
      w₁ = weigh [Coin.A, Coin.B] [Coin.C, Coin.D] ∧
      w₂ = weigh [Coin.A, Coin.C] [Coin.B, Coin.D] ∧
      w₃ = weigh [Coin.A, Coin.D] [Coin.B, Coin.C]) :=
by
  sorry

end NUMINAMATH_CALUDE_identify_counterfeit_coin_l3670_367059
