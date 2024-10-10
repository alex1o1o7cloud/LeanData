import Mathlib

namespace square_one_fifth_equals_point_zero_four_l4090_409089

theorem square_one_fifth_equals_point_zero_four (ε : ℝ) :
  ∃ ε > 0, (1 / 5 : ℝ)^2 = 0.04 + ε ∧ ε < 0.00000000000000001 := by
  sorry

end square_one_fifth_equals_point_zero_four_l4090_409089


namespace quadratic_equation_properties_l4090_409059

-- Define the coefficients of the quadratic equation -16x^2 + 72x - 90 = 0
def a : ℝ := -16
def b : ℝ := 72
def c : ℝ := -90

-- Theorem stating the sum of solutions and absence of positive real solutions
theorem quadratic_equation_properties :
  (let sum_of_solutions := -b / a
   sum_of_solutions = 4.5) ∧
  (∀ x : ℝ, -16 * x^2 + 72 * x - 90 ≠ 0 ∨ x ≤ 0) :=
by sorry

end quadratic_equation_properties_l4090_409059


namespace nails_per_station_l4090_409051

theorem nails_per_station (total_nails : ℕ) (num_stations : ℕ) 
  (h1 : total_nails = 140) (h2 : num_stations = 20) :
  total_nails / num_stations = 7 := by
  sorry

end nails_per_station_l4090_409051


namespace quadratic_inequality_condition_l4090_409065

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x, a * x^2 + 2 * a * x + 1 > 0) →
  (0 ≤ a ∧ a < 2) ∧
  ¬(0 ≤ a ∧ a < 2 → ∀ x, a * x^2 + 2 * a * x + 1 > 0) :=
by sorry

end quadratic_inequality_condition_l4090_409065


namespace x_axis_fixed_slope_two_invariant_l4090_409022

/-- Transformation that maps a point (x, y) to (x-y, -y) -/
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - p.2, -p.2)

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

theorem x_axis_fixed :
  ∀ (x : ℝ), transform (x, 0) = (x, 0) := by sorry

theorem slope_two_invariant (b : ℝ) :
  ∀ (x y : ℝ), 
    (Line.contains { slope := 2, intercept := b } (x, y)) →
    (Line.contains { slope := 2, intercept := b } (transform (x, y))) := by sorry

end x_axis_fixed_slope_two_invariant_l4090_409022


namespace negation_of_proposition_l4090_409092

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), Int.floor a = a → a^2 * b = 0) ↔
  (∃ (a b : ℝ), Int.floor a = a ∧ a^2 * b ≠ 0) := by sorry

end negation_of_proposition_l4090_409092


namespace specific_semicircle_chord_product_l4090_409015

/-- A structure representing a semicircle with equally spaced points -/
structure SemicircleWithPoints where
  radius : ℝ
  num_points : ℕ

/-- The product of chord lengths in a semicircle with equally spaced points -/
def chord_product (s : SemicircleWithPoints) : ℝ :=
  sorry

/-- Theorem stating the product of chord lengths for a specific semicircle configuration -/
theorem specific_semicircle_chord_product :
  let s : SemicircleWithPoints := { radius := 4, num_points := 8 }
  chord_product s = 4718592 := by
  sorry

end specific_semicircle_chord_product_l4090_409015


namespace smallest_n_congruence_four_satisfies_congruence_four_is_smallest_l4090_409067

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 7^n % 5 = n^4 % 5) → n ≥ 4 :=
by sorry

theorem four_satisfies_congruence : 
  7^4 % 5 = 4^4 % 5 :=
by sorry

theorem four_is_smallest : 
  ∀ m : ℕ, m > 0 ∧ m < 4 → 7^m % 5 ≠ m^4 % 5 :=
by sorry

end smallest_n_congruence_four_satisfies_congruence_four_is_smallest_l4090_409067


namespace broccoli_production_increase_l4090_409021

theorem broccoli_production_increase :
  ∀ (last_year_side : ℕ) (this_year_side : ℕ),
    last_year_side = 50 →
    this_year_side = 51 →
    this_year_side * this_year_side - last_year_side * last_year_side = 101 :=
by sorry

end broccoli_production_increase_l4090_409021


namespace imaginary_part_of_complex_number_l4090_409073

theorem imaginary_part_of_complex_number :
  let z : ℂ := -1/2 + (1/2) * Complex.I
  Complex.im z = 1/2 := by sorry

end imaginary_part_of_complex_number_l4090_409073


namespace largest_C_for_divisibility_by_4_l4090_409079

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem largest_C_for_divisibility_by_4 :
  ∃ (B : ℕ) (h_B : B < 10),
    ∀ (C : ℕ) (h_C : C < 10),
      is_divisible_by_4 (4000000 + 600000 + B * 100000 + 41800 + C) →
      C ≤ 8 ∧
      is_divisible_by_4 (4000000 + 600000 + B * 100000 + 41800 + 8) :=
by sorry

end largest_C_for_divisibility_by_4_l4090_409079


namespace total_cakes_served_l4090_409047

def cakes_served_lunch_today : ℕ := 5
def cakes_served_dinner_today : ℕ := 6
def cakes_served_yesterday : ℕ := 3

theorem total_cakes_served :
  cakes_served_lunch_today + cakes_served_dinner_today + cakes_served_yesterday = 14 :=
by sorry

end total_cakes_served_l4090_409047


namespace evaluate_expression_l4090_409093

theorem evaluate_expression : 3000 * (3000^1500) = 3000^1501 := by
  sorry

end evaluate_expression_l4090_409093


namespace smallest_c_value_l4090_409033

/-- Given a function y = a * cos(b * x + c), where a, b, and c are positive constants,
    and the graph reaches its maximum at x = 0, prove that the smallest possible value of c is 0. -/
theorem smallest_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.cos (b * x + c) ≤ a * Real.cos c) →
  c = 0 := by
  sorry

end smallest_c_value_l4090_409033


namespace hall_area_is_450_l4090_409098

/-- Represents a rectangular hall with specific properties. -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  width_half_length : width = length / 2
  length_width_diff : length - width = 15

/-- Calculates the area of a rectangular hall. -/
def area (hall : RectangularHall) : ℝ := hall.length * hall.width

/-- Theorem stating that a rectangular hall with the given properties has an area of 450 square units. -/
theorem hall_area_is_450 (hall : RectangularHall) : area hall = 450 := by
  sorry

end hall_area_is_450_l4090_409098


namespace function_has_max_and_min_l4090_409082

/-- The function f(x) = x^3 - ax^2 + ax has both a maximum and a minimum value 
    if and only if a is in the range (-∞, 0) ∪ (3, +∞) -/
theorem function_has_max_and_min (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^3 - a*x^2 + a*x ≤ x₁^3 - a*x₁^2 + a*x₁) ∧
    (∀ x : ℝ, x^3 - a*x^2 + a*x ≥ x₂^3 - a*x₂^2 + a*x₂)) ↔ 
  (a < 0 ∨ a > 3) := by
  sorry

#check function_has_max_and_min

end function_has_max_and_min_l4090_409082


namespace washer_dryer_price_ratio_l4090_409078

theorem washer_dryer_price_ratio :
  ∀ (washer_price dryer_price : ℕ),
    washer_price + dryer_price = 600 →
    ∃ k : ℕ, washer_price = k * dryer_price →
    dryer_price = 150 →
    washer_price / dryer_price = 3 := by
  sorry

end washer_dryer_price_ratio_l4090_409078


namespace pan_division_theorem_main_theorem_l4090_409036

/-- Represents the dimensions of a rectangular pan --/
structure PanDimensions where
  length : ℕ
  width : ℕ

/-- Represents the side length of a square piece of cake --/
def PieceSize : ℕ := 3

/-- Calculates the number of square pieces that can be cut from a rectangular pan --/
def numberOfPieces (pan : PanDimensions) (pieceSize : ℕ) : ℕ :=
  (pan.length * pan.width) / (pieceSize * pieceSize)

/-- Theorem stating that a 30x24 inch pan can be divided into 80 3-inch square pieces --/
theorem pan_division_theorem (pan : PanDimensions) (h1 : pan.length = 30) (h2 : pan.width = 24) :
  numberOfPieces pan PieceSize = 80 := by
  sorry

/-- Main theorem to be proved --/
theorem main_theorem : ∃ (pan : PanDimensions), 
  pan.length = 30 ∧ pan.width = 24 ∧ numberOfPieces pan PieceSize = 80 := by
  sorry

end pan_division_theorem_main_theorem_l4090_409036


namespace special_sequence_property_l4090_409081

/-- A sequence of natural numbers with specific properties -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ k, a (k + 1) - a k ∈ ({0, 1} : Set ℕ)

theorem special_sequence_property (a : ℕ → ℕ) (m : ℕ) :
  SpecialSequence a →
  (∃ m, a m = m / 1000) →
  ∃ n, a n = n / 500 := by
  sorry

end special_sequence_property_l4090_409081


namespace pauls_shopping_bill_l4090_409026

def dress_shirt_price : ℝ := 15.00
def pants_price : ℝ := 40.00
def suit_price : ℝ := 150.00
def sweater_price : ℝ := 30.00

def num_dress_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_suits : ℕ := 1
def num_sweaters : ℕ := 2

def store_discount : ℝ := 0.20
def coupon_discount : ℝ := 0.10

def total_before_discount : ℝ := 
  dress_shirt_price * num_dress_shirts +
  pants_price * num_pants +
  suit_price * num_suits +
  sweater_price * num_sweaters

def final_price : ℝ := 
  total_before_discount * (1 - store_discount) * (1 - coupon_discount)

theorem pauls_shopping_bill : final_price = 252.00 := by
  sorry

end pauls_shopping_bill_l4090_409026


namespace student_ticket_price_is_318_l4090_409061

/-- Calculates the price of a student ticket given the total number of tickets sold,
    total revenue, adult ticket price, number of adult tickets sold, and number of student tickets sold. -/
def student_ticket_price (total_tickets : ℕ) (total_revenue : ℚ) (adult_price : ℚ) 
                         (adult_tickets : ℕ) (student_tickets : ℕ) : ℚ :=
  (total_revenue - (adult_price * adult_tickets)) / student_tickets

/-- Proves that the student ticket price is $3.18 given the specified conditions. -/
theorem student_ticket_price_is_318 :
  student_ticket_price 846 3846 6 410 436 = 318/100 := by
  sorry

end student_ticket_price_is_318_l4090_409061


namespace cubic_polynomial_conditions_l4090_409007

def f (x : ℚ) : ℚ := 15 * x^3 - 37 * x^2 + 30 * x - 8

theorem cubic_polynomial_conditions :
  f 1 = 0 ∧ f (2/3) = -4 ∧ f (4/5) = -16/5 := by
  sorry

end cubic_polynomial_conditions_l4090_409007


namespace diophantine_equation_only_zero_solution_l4090_409009

theorem diophantine_equation_only_zero_solution (x y u t : ℤ) 
  (h : x^2 + y^2 = 1974 * (u^2 + t^2)) : x = 0 ∧ y = 0 ∧ u = 0 ∧ t = 0 := by
  sorry

end diophantine_equation_only_zero_solution_l4090_409009


namespace sin_135_degrees_l4090_409062

theorem sin_135_degrees :
  Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_135_degrees_l4090_409062


namespace trigonometric_identity_l4090_409074

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : a > 0) (k : b > 0) :
  (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b) →
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 :=
by sorry

end trigonometric_identity_l4090_409074


namespace parallelogram_base_length_l4090_409090

theorem parallelogram_base_length
  (area : ℝ) (base : ℝ) (altitude : ℝ)
  (h1 : area = 288)
  (h2 : altitude = 2 * base)
  (h3 : area = base * altitude) :
  base = 12 := by
sorry

end parallelogram_base_length_l4090_409090


namespace range_of_a_l4090_409041

theorem range_of_a (a : ℝ) : 
  (∀ x, 0 < x ∧ x < 1 → (x - a) * (x - (a + 2)) ≤ 0) ∧ 
  (∃ x, ¬(0 < x ∧ x < 1) ∧ (x - a) * (x - (a + 2)) ≤ 0) → 
  -1 ≤ a ∧ a ≤ 0 :=
by sorry

end range_of_a_l4090_409041


namespace same_grade_probability_l4090_409049

theorem same_grade_probability (total : ℕ) (first : ℕ) (second : ℕ) (third : ℕ) 
  (h_total : total = 10)
  (h_first : first = 4)
  (h_second : second = 3)
  (h_third : third = 3)
  (h_sum : first + second + third = total) :
  (Nat.choose first 2 + Nat.choose second 2 + Nat.choose third 2) / Nat.choose total 2 = 4 / 15 := by
sorry

end same_grade_probability_l4090_409049


namespace alice_paid_fifteen_per_acorn_l4090_409056

/-- The price Alice paid for each acorn -/
def alice_price_per_acorn (alice_acorn_count : ℕ) (alice_bob_price_ratio : ℕ) (bob_total_price : ℕ) : ℚ :=
  (alice_bob_price_ratio * bob_total_price : ℚ) / alice_acorn_count

/-- Theorem stating that Alice paid $15 for each acorn -/
theorem alice_paid_fifteen_per_acorn :
  alice_price_per_acorn 3600 9 6000 = 15 := by
  sorry

end alice_paid_fifteen_per_acorn_l4090_409056


namespace mom_has_one_eye_l4090_409006

/-- Represents the number of eyes for each family member -/
structure MonsterFamily where
  mom_eyes : ℕ
  dad_eyes : ℕ
  kids_eyes : ℕ
  num_kids : ℕ

/-- The total number of eyes in the monster family -/
def total_eyes (f : MonsterFamily) : ℕ :=
  f.mom_eyes + f.dad_eyes + f.kids_eyes * f.num_kids

/-- Theorem stating that the mom has 1 eye given the conditions -/
theorem mom_has_one_eye (f : MonsterFamily) 
  (h1 : f.dad_eyes = 3)
  (h2 : f.kids_eyes = 4)
  (h3 : f.num_kids = 3)
  (h4 : total_eyes f = 16) : 
  f.mom_eyes = 1 := by
  sorry


end mom_has_one_eye_l4090_409006


namespace sum_of_critical_slopes_l4090_409038

/-- The parabola y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- The point Q -/
def Q : ℝ × ℝ := (10, 5)

/-- The line through Q with slope m -/
def line (m : ℝ) (x : ℝ) : ℝ := m * (x - Q.1) + Q.2

/-- The quadratic equation representing the intersection of the line and parabola -/
def intersection_quadratic (m : ℝ) (x : ℝ) : ℝ := 
  parabola x - line m x

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := 
  m^2 - 4 * (10 * m - 5)

/-- The theorem stating that the sum of the critical slopes is 40 -/
theorem sum_of_critical_slopes : 
  ∃ (r s : ℝ), (∀ m, discriminant m < 0 ↔ r < m ∧ m < s) ∧ r + s = 40 :=
sorry

end sum_of_critical_slopes_l4090_409038


namespace house_to_library_distance_l4090_409032

/-- Represents the distances between locations in miles -/
structure Distances where
  total : ℝ
  library_to_post_office : ℝ
  post_office_to_home : ℝ

/-- Calculates the distance from house to library -/
def distance_house_to_library (d : Distances) : ℝ :=
  d.total - d.library_to_post_office - d.post_office_to_home

/-- Theorem stating the distance from house to library is 0.3 miles -/
theorem house_to_library_distance (d : Distances) 
  (h1 : d.total = 0.8)
  (h2 : d.library_to_post_office = 0.1)
  (h3 : d.post_office_to_home = 0.4) : 
  distance_house_to_library d = 0.3 := by
  sorry

end house_to_library_distance_l4090_409032


namespace complex_power_difference_l4090_409044

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : 
  (1 + i)^18 - (1 - i)^18 = 1024 * i :=
sorry

end complex_power_difference_l4090_409044


namespace no_prime_with_consecutive_squares_l4090_409085

theorem no_prime_with_consecutive_squares (n : ℕ) : 
  Prime n → ¬(∃ a b : ℕ, (2 * n + 1 = a^2) ∧ (3 * n + 1 = b^2)) :=
by sorry

end no_prime_with_consecutive_squares_l4090_409085


namespace congruence_problem_l4090_409066

theorem congruence_problem (x : ℤ) : 
  (5 * x + 3) % 16 = 4 → (4 * x + 5) % 16 = 9 := by
  sorry

end congruence_problem_l4090_409066


namespace vanaspati_percentage_after_addition_l4090_409077

/-- Calculates the percentage of vanaspati in a ghee mixture after adding pure ghee -/
theorem vanaspati_percentage_after_addition
  (original_quantity : ℝ)
  (original_pure_ghee_percentage : ℝ)
  (original_vanaspati_percentage : ℝ)
  (added_pure_ghee : ℝ)
  (h1 : original_quantity = 30)
  (h2 : original_pure_ghee_percentage = 50)
  (h3 : original_vanaspati_percentage = 50)
  (h4 : added_pure_ghee = 20)
  (h5 : original_pure_ghee_percentage + original_vanaspati_percentage = 100) :
  let original_vanaspati := original_quantity * (original_vanaspati_percentage / 100)
  let new_total_quantity := original_quantity + added_pure_ghee
  (original_vanaspati / new_total_quantity) * 100 = 30 := by
sorry

end vanaspati_percentage_after_addition_l4090_409077


namespace circle_diameter_from_area_l4090_409037

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end circle_diameter_from_area_l4090_409037


namespace cottage_build_time_l4090_409043

/-- Represents the time (in days) it takes to build a cottage -/
def build_time (num_builders : ℕ) (days : ℕ) : Prop :=
  num_builders * days = 24

theorem cottage_build_time :
  build_time 3 8 → build_time 6 4 := by sorry

end cottage_build_time_l4090_409043


namespace custom_op_7_neg3_custom_op_not_commutative_l4090_409071

-- Define the custom operation ※
def custom_op (a b : ℤ) : ℤ := (a + 2) * 2 - b

-- Theorem 1: 7 ※ (-3) = 21
theorem custom_op_7_neg3 : custom_op 7 (-3) = 21 := by sorry

-- Theorem 2: 7 ※ (-3) ≠ (-3) ※ 7
theorem custom_op_not_commutative : custom_op 7 (-3) ≠ custom_op (-3) 7 := by sorry

end custom_op_7_neg3_custom_op_not_commutative_l4090_409071


namespace fraction_equivalence_l4090_409057

theorem fraction_equivalence : 
  (20 / 16 : ℚ) = 10 / 8 ∧
  (1 + 6 / 24 : ℚ) = 10 / 8 ∧
  (1 + 2 / 8 : ℚ) = 10 / 8 ∧
  (1 + 40 / 160 : ℚ) = 10 / 8 ∧
  (1 + 4 / 8 : ℚ) ≠ 10 / 8 := by
  sorry

end fraction_equivalence_l4090_409057


namespace alice_ice_cream_count_l4090_409095

/-- The number of pints of ice cream Alice had on Wednesday -/
def ice_cream_on_wednesday (sunday_pints : ℕ) : ℕ :=
  let monday_pints := 3 * sunday_pints
  let tuesday_pints := monday_pints / 3
  let total_before_wednesday := sunday_pints + monday_pints + tuesday_pints
  let returned_pints := tuesday_pints / 2
  total_before_wednesday - returned_pints

/-- Theorem stating that Alice had 18 pints of ice cream on Wednesday -/
theorem alice_ice_cream_count : ice_cream_on_wednesday 4 = 18 := by
  sorry

#eval ice_cream_on_wednesday 4

end alice_ice_cream_count_l4090_409095


namespace A_obtuse_sufficient_not_necessary_l4090_409083

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define an obtuse angle
def is_obtuse (angle : Real) : Prop := angle > 90

-- Define an obtuse triangle
def is_obtuse_triangle (t : Triangle) : Prop :=
  is_obtuse t.A ∨ is_obtuse t.B ∨ is_obtuse t.C

-- Theorem statement
theorem A_obtuse_sufficient_not_necessary (t : Triangle) :
  (is_obtuse t.A → is_obtuse_triangle t) ∧
  ∃ (t' : Triangle), is_obtuse_triangle t' ∧ ¬is_obtuse t'.A :=
sorry

end A_obtuse_sufficient_not_necessary_l4090_409083


namespace repeating_decimal_sum_l4090_409011

theorem repeating_decimal_sum : 
  let x : ℚ := (23 : ℚ) / 99
  let y : ℚ := (14 : ℚ) / 999
  let z : ℚ := (6 : ℚ) / 9999
  x + y + z = (2469 : ℚ) / 9999 := by sorry

end repeating_decimal_sum_l4090_409011


namespace algebraic_expression_value_l4090_409001

theorem algebraic_expression_value (x : ℝ) : -2 * (2 - x) + (1 + x) = 0 → 2 * x^2 - 7 = -5 := by
  sorry

end algebraic_expression_value_l4090_409001


namespace lawrence_county_kids_at_camp_l4090_409000

def lawrence_county_kids_at_home : ℕ := 134867
def outside_county_kids_at_camp : ℕ := 424944
def total_kids_at_camp : ℕ := 458988

theorem lawrence_county_kids_at_camp :
  total_kids_at_camp - outside_county_kids_at_camp = 34044 := by
  sorry

end lawrence_county_kids_at_camp_l4090_409000


namespace power_value_l4090_409030

theorem power_value (m n : ℤ) (x : ℝ) (h1 : x^m = 3) (h2 : x = 2) : x^(2*m+n) = 18 := by
  sorry

end power_value_l4090_409030


namespace part1_part2_l4090_409016

-- Part 1
theorem part1 (f : ℝ → ℝ) :
  (∀ x ≥ 0, f (Real.sqrt x + 1) = x + 2 * Real.sqrt x) →
  (∀ x ≥ 1, f x = x^2 - 2*x) :=
sorry

-- Part 2
theorem part2 (f : ℝ → ℝ) :
  (∃ k b : ℝ, ∀ x, f x = k * x + b) →
  (∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) →
  (∀ x, f x = 2 * x + 7) :=
sorry

end part1_part2_l4090_409016


namespace sequence_fourth_term_l4090_409052

theorem sequence_fourth_term (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, S n = 3^n + 2*n + 1) →
  (∀ n : ℕ, n ≥ 1 → S n = S (n-1) + a n) →
  a 4 = 56 := by
sorry

end sequence_fourth_term_l4090_409052


namespace find_m_value_l4090_409028

theorem find_m_value (m : ℤ) : 
  (∃ (x : ℤ), x - m / 3 ≥ 0 ∧ 2 * x - 3 ≥ 3 * (x - 2)) ∧ 
  (∃! (a b : ℤ), a ≠ b ∧ 
    (a - m / 3 ≥ 0 ∧ 2 * a - 3 ≥ 3 * (a - 2)) ∧ 
    (b - m / 3 ≥ 0 ∧ 2 * b - 3 ≥ 3 * (b - 2))) ∧
  (∃ (k : ℤ), k > 0 ∧ 4 * (m + 1) = k * (m^2 - 1)) →
  m = 5 :=
sorry

end find_m_value_l4090_409028


namespace two_solutions_congruence_l4090_409053

theorem two_solutions_congruence (a : ℕ) (h_a : a < 2007) :
  (∃! u v : ℕ, u < 2007 ∧ v < 2007 ∧ u ≠ v ∧
    (u^2 + a) % 2007 = 0 ∧ (v^2 + a) % 2007 = 0) ↔
  (a % 9 = 0 ∨ a % 9 = 8 ∨ a % 9 = 5 ∨ a % 9 = 2) ∧
  ∃ x : ℕ, x < 223 ∧ (x^2 % 223 = (223 - a % 223) % 223) :=
by sorry

end two_solutions_congruence_l4090_409053


namespace max_value_of_product_sum_l4090_409010

theorem max_value_of_product_sum (x y z : ℝ) (h : x + 2*y + z = 7) :
  ∃ (max : ℝ), max = 7 ∧ ∀ (x' y' z' : ℝ), x' + 2*y' + z' = 7 → x'*y' + x'*z' + y'*z' ≤ max :=
sorry

end max_value_of_product_sum_l4090_409010


namespace coin_division_l4090_409087

theorem coin_division (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5)
  (h3 : ∀ m : ℕ, m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) :
  n % 9 = 2 := by
  sorry

end coin_division_l4090_409087


namespace weight_sequence_l4090_409064

theorem weight_sequence (a : ℕ → ℝ) : 
  (∀ n, a n < a (n + 1)) →  -- weights are in increasing order
  (∀ k, k ≤ 29 → a k + a (k + 3) = a (k + 1) + a (k + 2)) →  -- balancing condition
  a 3 = 9 →  -- third weight is 9 grams
  a 9 = 33 →  -- ninth weight is 33 grams
  a 33 = 257 :=  -- 33rd weight is 257 grams
by
  sorry


end weight_sequence_l4090_409064


namespace can_obtain_next_number_l4090_409045

/-- Represents the allowed operations on a number -/
inductive Operation
  | AddNine : Operation
  | DeleteOne : Operation

/-- Applies a sequence of operations to a number -/
def applyOperations (a : ℕ) (ops : List Operation) : ℕ := sorry

/-- Theorem stating that A+1 can always be obtained from A using the allowed operations -/
theorem can_obtain_next_number (A : ℕ) : 
  A > 0 → ∃ (ops : List Operation), applyOperations A ops = A + 1 := by sorry

end can_obtain_next_number_l4090_409045


namespace either_false_sufficient_not_necessary_for_not_p_true_l4090_409004

theorem either_false_sufficient_not_necessary_for_not_p_true (p q : Prop) :
  (((¬p ∧ ¬q) → ¬p) ∧ ∃ (r : Prop), (¬r ∧ ¬(¬r ∧ ¬q))) := by
  sorry

end either_false_sufficient_not_necessary_for_not_p_true_l4090_409004


namespace expression_equals_polynomial_l4090_409003

/-- The given expression is equal to the simplified polynomial for all real x -/
theorem expression_equals_polynomial (x : ℝ) :
  (3 * x^3 + 4 * x^2 - 5 * x + 8) * (x - 2) -
  (x - 2) * (2 * x^3 - 7 * x^2 + 10) +
  (7 * x - 15) * (x - 2) * (2 * x + 1) =
  x^4 + 23 * x^3 - 78 * x^2 + 39 * x + 34 := by
  sorry

end expression_equals_polynomial_l4090_409003


namespace monica_classes_count_l4090_409024

/-- Represents the number of students in each of Monica's classes -/
def class_sizes : List Nat := [20, 25, 25, 10, 28, 28]

/-- The total number of students Monica sees each day -/
def total_students : Nat := 136

/-- Theorem stating that Monica has 6 classes per day -/
theorem monica_classes_count : List.length class_sizes = 6 ∧ List.sum class_sizes = total_students := by
  sorry

end monica_classes_count_l4090_409024


namespace fraction_evaluation_l4090_409058

theorem fraction_evaluation : (2 + 3 + 4) / (2 * 3 * 4) = 3 / 8 := by
  sorry

end fraction_evaluation_l4090_409058


namespace problem_solution_l4090_409042

theorem problem_solution (x : ℝ) : (0.20 * x = 0.15 * 1500 - 15) → x = 1050 := by
  sorry

end problem_solution_l4090_409042


namespace select_books_result_l4090_409096

/-- The number of ways to select one book from each of two bags of science books -/
def select_books (bag1_count : ℕ) (bag2_count : ℕ) : ℕ :=
  bag1_count * bag2_count

/-- Theorem: The number of ways to select one book from each of two bags,
    where one bag contains 4 different books and the other contains 5 different books,
    is equal to 20. -/
theorem select_books_result : select_books 4 5 = 20 := by
  sorry

end select_books_result_l4090_409096


namespace geometric_progression_first_term_l4090_409048

/-- 
A geometric progression with 2 terms, where:
- The last term is 1/3
- The common ratio is 1/3
- The sum of terms is 40/3
Then, the first term is 10.
-/
theorem geometric_progression_first_term 
  (n : ℕ) 
  (last_term : ℚ) 
  (common_ratio : ℚ) 
  (sum : ℚ) : 
  n = 2 ∧ 
  last_term = 1/3 ∧ 
  common_ratio = 1/3 ∧ 
  sum = 40/3 → 
  ∃ (a : ℚ), a = 10 ∧ sum = a * (1 - common_ratio^n) / (1 - common_ratio) :=
by sorry

end geometric_progression_first_term_l4090_409048


namespace configurations_count_l4090_409097

/-- The number of squares in the set -/
def total_squares : ℕ := 8

/-- The number of squares to be placed -/
def squares_to_place : ℕ := 2

/-- The number of distinct sides on which squares can be placed -/
def distinct_sides : ℕ := 2

/-- The number of configurations that can be formed -/
def num_configurations : ℕ := total_squares * (total_squares - 1)

theorem configurations_count :
  num_configurations = 56 :=
sorry

end configurations_count_l4090_409097


namespace wuhan_spring_temp_difference_l4090_409019

/-- The average daily high temperature in spring in the Wuhan area -/
def average_high : ℝ := 15

/-- The lowest temperature in spring in the Wuhan area -/
def lowest_temp : ℝ := 7

/-- The difference between the average daily high temperature and the lowest temperature -/
def temp_difference : ℝ := average_high - lowest_temp

/-- Theorem stating that the temperature difference is 8°C -/
theorem wuhan_spring_temp_difference : temp_difference = 8 := by
  sorry

end wuhan_spring_temp_difference_l4090_409019


namespace archery_competition_scores_l4090_409084

/-- Represents an archer's score distribution --/
structure ArcherScore where
  bullseye : Nat
  ring39 : Nat
  ring24 : Nat
  ring23 : Nat
  ring17 : Nat
  ring16 : Nat

/-- Calculates the total score for an archer --/
def totalScore (score : ArcherScore) : Nat :=
  40 * score.bullseye + 39 * score.ring39 + 24 * score.ring24 +
  23 * score.ring23 + 17 * score.ring17 + 16 * score.ring16

/-- Calculates the total number of arrows used --/
def totalArrows (score : ArcherScore) : Nat :=
  score.bullseye + score.ring39 + score.ring24 + score.ring23 + score.ring17 + score.ring16

theorem archery_competition_scores :
  ∃ (dora reggie finch : ArcherScore),
    totalScore dora = 120 ∧
    totalScore reggie = 110 ∧
    totalScore finch = 100 ∧
    totalArrows dora = 6 ∧
    totalArrows reggie = 6 ∧
    totalArrows finch = 6 ∧
    dora.bullseye + reggie.bullseye + finch.bullseye = 1 ∧
    dora = { bullseye := 1, ring39 := 0, ring24 := 0, ring23 := 0, ring17 := 0, ring16 := 5 } ∧
    reggie = { bullseye := 0, ring39 := 0, ring24 := 0, ring23 := 2, ring17 := 0, ring16 := 4 } ∧
    finch = { bullseye := 0, ring39 := 0, ring24 := 0, ring23 := 0, ring17 := 4, ring16 := 2 } :=
by
  sorry


end archery_competition_scores_l4090_409084


namespace car_average_speed_l4090_409060

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 100) (h2 : speed2 = 80) :
  (speed1 + speed2) / 2 = 90 := by
  sorry

end car_average_speed_l4090_409060


namespace attendance_problem_l4090_409008

/-- Proves that the number of people who didn't show up is 12 --/
theorem attendance_problem (total_invited : ℕ) (tables_used : ℕ) (table_capacity : ℕ) : 
  total_invited - (tables_used * table_capacity) = 12 :=
by
  sorry

#check attendance_problem 18 2 3

end attendance_problem_l4090_409008


namespace delta_problem_l4090_409094

-- Define the Δ operation
def delta (a b : ℕ) : ℕ := a^2 + b

-- State the theorem
theorem delta_problem : delta (3^(delta 2 6)) (4^(delta 4 2)) = 72201960037 := by
  sorry

end delta_problem_l4090_409094


namespace cats_remaining_after_sale_l4090_409046

theorem cats_remaining_after_sale
  (initial_siamese : ℕ)
  (initial_persian : ℕ)
  (initial_house : ℕ)
  (sold_siamese : ℕ)
  (sold_persian : ℕ)
  (sold_house : ℕ)
  (h1 : initial_siamese = 20)
  (h2 : initial_persian = 12)
  (h3 : initial_house = 8)
  (h4 : sold_siamese = 8)
  (h5 : sold_persian = 5)
  (h6 : sold_house = 3) :
  initial_siamese + initial_persian + initial_house -
  (sold_siamese + sold_persian + sold_house) = 24 :=
by sorry

end cats_remaining_after_sale_l4090_409046


namespace sum_opposite_and_abs_l4090_409031

theorem sum_opposite_and_abs : -15 + |(-6)| = -9 := by
  sorry

end sum_opposite_and_abs_l4090_409031


namespace binomial_coefficient_floor_divisibility_l4090_409050

theorem binomial_coefficient_floor_divisibility (p n : ℕ) 
  (hp : Nat.Prime p) (hn : n ≥ p) : 
  (Nat.choose n p - n / p) % p = 0 :=
by sorry

end binomial_coefficient_floor_divisibility_l4090_409050


namespace smallest_bob_number_l4090_409035

def alice_number : ℕ := 36

def is_twice_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = 2 * p

def has_only_factors_of (n m : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p ∣ n → p ∣ m

theorem smallest_bob_number :
  ∃ n : ℕ, 
    n > 0 ∧
    is_twice_prime n ∧
    has_only_factors_of n alice_number ∧
    (∀ m : ℕ, m > 0 → is_twice_prime m → has_only_factors_of m alice_number → n ≤ m) ∧
    n = 4 :=
sorry

end smallest_bob_number_l4090_409035


namespace sum_of_a_and_b_l4090_409029

theorem sum_of_a_and_b (a b : ℝ) (ha : |a| = 5) (hb : |b| = 2) (ha_neg : a < 0) (hb_pos : b > 0) :
  a + b = -3 := by
  sorry

end sum_of_a_and_b_l4090_409029


namespace least_perfect_square_exponent_l4090_409013

theorem least_perfect_square_exponent : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (∃ (k : ℕ), 2^8 + 2^11 + 2^m = k^2) → m ≥ n) ∧
  (∃ (k : ℕ), 2^8 + 2^11 + 2^n = k^2) ∧
  n = 12 := by
sorry

end least_perfect_square_exponent_l4090_409013


namespace heaviest_tv_weight_difference_l4090_409080

/-- Represents the dimensions of a TV screen -/
structure TVDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the area of a TV screen given its dimensions -/
def screenArea (d : TVDimensions) : ℕ := d.width * d.height

/-- Calculates the weight of a TV in ounces given its screen area -/
def tvWeight (area : ℕ) : ℕ := area * 4

/-- Converts weight from ounces to pounds -/
def ouncesToPounds (oz : ℕ) : ℕ := oz / 16

theorem heaviest_tv_weight_difference (bill_tv bob_tv steve_tv : TVDimensions) 
    (h1 : bill_tv = ⟨48, 100⟩)
    (h2 : bob_tv = ⟨70, 60⟩)
    (h3 : steve_tv = ⟨84, 92⟩) :
  ouncesToPounds (tvWeight (screenArea steve_tv)) - 
  (ouncesToPounds (tvWeight (screenArea bill_tv)) + ouncesToPounds (tvWeight (screenArea bob_tv))) = 318 := by
  sorry


end heaviest_tv_weight_difference_l4090_409080


namespace james_huskies_count_l4090_409017

/-- The number of huskies James has -/
def num_huskies : ℕ := sorry

/-- The number of pitbulls James has -/
def num_pitbulls : ℕ := 2

/-- The number of golden retrievers James has -/
def num_golden_retrievers : ℕ := 4

/-- The number of pups each husky and pitbull has -/
def pups_per_husky_pitbull : ℕ := 3

/-- The additional number of pups each golden retriever has compared to huskies -/
def additional_pups_golden : ℕ := 2

/-- The difference between total pups and adult dogs -/
def pup_adult_difference : ℕ := 30

theorem james_huskies_count :
  num_huskies = 5 ∧
  num_huskies * pups_per_husky_pitbull +
  num_pitbulls * pups_per_husky_pitbull +
  num_golden_retrievers * (pups_per_husky_pitbull + additional_pups_golden) =
  num_huskies + num_pitbulls + num_golden_retrievers + pup_adult_difference :=
sorry

end james_huskies_count_l4090_409017


namespace mia_speed_theorem_l4090_409075

def eugene_speed : ℚ := 5
def carlos_ratio : ℚ := 3/4
def mia_ratio : ℚ := 4/3

theorem mia_speed_theorem : 
  mia_ratio * (carlos_ratio * eugene_speed) = eugene_speed := by
  sorry

end mia_speed_theorem_l4090_409075


namespace equation_system_solution_equality_l4090_409020

theorem equation_system_solution_equality (x y r s : ℝ) : 
  3 * x + 2 * y = 16 →
  5 * x + 3 * y = r →
  5 * x + 3 * y = s →
  r - s = 0 := by
sorry

end equation_system_solution_equality_l4090_409020


namespace rs_fraction_l4090_409088

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the altitude CH
def altitude (t : Triangle) : ℝ × ℝ := sorry

-- Define the points R and S
def R (t : Triangle) : ℝ × ℝ := sorry
def S (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem rs_fraction (t : Triangle) :
  distance (t.A) (t.B) = 2023 →
  distance (t.A) (t.C) = 2022 →
  distance (t.B) (t.C) = 2021 →
  distance (R t) (S t) = 2021 / 2023 := by
  sorry

end rs_fraction_l4090_409088


namespace allison_extra_glue_sticks_l4090_409072

/-- Represents the number of items bought by a person -/
structure Items where
  glue_sticks : ℕ
  construction_paper : ℕ

/-- The problem setup -/
def craft_store_problem (allison marie : Items) : Prop :=
  allison.glue_sticks > marie.glue_sticks ∧
  marie.construction_paper = 6 * allison.construction_paper ∧
  marie.glue_sticks = 15 ∧
  marie.construction_paper = 30 ∧
  allison.glue_sticks + allison.construction_paper = 28

/-- The theorem to prove -/
theorem allison_extra_glue_sticks (allison marie : Items) 
  (h : craft_store_problem allison marie) : 
  allison.glue_sticks - marie.glue_sticks = 8 := by
  sorry


end allison_extra_glue_sticks_l4090_409072


namespace days_to_finish_book_l4090_409005

theorem days_to_finish_book (total_pages book_chapters pages_per_day : ℕ) : 
  total_pages = 193 → book_chapters = 15 → pages_per_day = 44 → 
  (total_pages + pages_per_day - 1) / pages_per_day = 5 := by
sorry

end days_to_finish_book_l4090_409005


namespace sanoop_tshirts_l4090_409002

/-- The number of t-shirts Sanoop initially bought -/
def initial_tshirts : ℕ := 8

/-- The initial average price of t-shirts in Rs -/
def initial_avg_price : ℚ := 526

/-- The average price of t-shirts after returning one, in Rs -/
def new_avg_price : ℚ := 505

/-- The price of the returned t-shirt in Rs -/
def returned_price : ℚ := 673

theorem sanoop_tshirts :
  initial_tshirts = 8 ∧
  initial_avg_price * initial_tshirts = 
    new_avg_price * (initial_tshirts - 1) + returned_price :=
by sorry

end sanoop_tshirts_l4090_409002


namespace right_triangle_hypotenuse_l4090_409076

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- Sum of squares condition
  c = 25 * Real.sqrt 10 := by
sorry

end right_triangle_hypotenuse_l4090_409076


namespace simplify_and_evaluate_l4090_409054

theorem simplify_and_evaluate (a : ℤ) 
  (h1 : -1 < a) (h2 : a < Real.sqrt 5) (h3 : a ≠ 0) (h4 : a ≠ 1) (h5 : a ≠ -1) :
  (a - a^2 / (a^2 - 1)) / (a^2 / (a^2 - 1)) = 1/2 :=
sorry

end simplify_and_evaluate_l4090_409054


namespace cistern_wet_surface_area_l4090_409018

/-- Calculates the total wet surface area of a rectangular cistern --/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem stating the total wet surface area of a specific cistern --/
theorem cistern_wet_surface_area :
  let length : ℝ := 5
  let width : ℝ := 4
  let depth : ℝ := 1.25
  total_wet_surface_area length width depth = 42.5 := by
  sorry

end cistern_wet_surface_area_l4090_409018


namespace radical_conjugate_sum_product_l4090_409014

theorem radical_conjugate_sum_product (c d : ℝ) : 
  (c + Real.sqrt d) + (c - Real.sqrt d) = 6 → 
  (c + Real.sqrt d) * (c - Real.sqrt d) = 4 → 
  c + d = 8 := by
sorry

end radical_conjugate_sum_product_l4090_409014


namespace rectangle_area_l4090_409025

/-- Given a rectangle ABCD with the following properties:
  - Sides AB and CD have length 3x
  - Sides AD and BC have length x
  - A circle with radius r is tangent to side AB at its midpoint, AD, and CD
  - 2r = x
  Prove that the area of rectangle ABCD is 12r^2 -/
theorem rectangle_area (x r : ℝ) (h1 : 2 * r = x) : 3 * x * x = 12 * r^2 := by
  sorry

end rectangle_area_l4090_409025


namespace unique_solution_quadratic_positive_n_for_unique_solution_l4090_409055

theorem unique_solution_quadratic (n : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + n * x + 16 = 0) ↔ n = 24 ∨ n = -24 :=
by sorry

theorem positive_n_for_unique_solution :
  ∃ n : ℝ, n > 0 ∧ (∃! x : ℝ, 9 * x^2 + n * x + 16 = 0) ∧ n = 24 :=
by sorry

end unique_solution_quadratic_positive_n_for_unique_solution_l4090_409055


namespace distance_between_points_l4090_409091

/-- The distance between two points (2, -7) and (-8, 4) is √221. -/
theorem distance_between_points : Real.sqrt 221 = Real.sqrt ((2 - (-8))^2 + ((-7) - 4)^2) := by
  sorry

end distance_between_points_l4090_409091


namespace curve_transformation_l4090_409063

/-- Given a curve C: (x-y)^2 + y^2 = 1 transformed by matrix A = [[2, -2], [0, 1]],
    prove that the resulting curve C' has the equation x^2/4 + y^2 = 1 -/
theorem curve_transformation (x₀ y₀ x y : ℝ) : 
  (x₀ - y₀)^2 + y₀^2 = 1 →
  x = 2*x₀ - 2*y₀ →
  y = y₀ →
  x^2/4 + y^2 = 1 := by
sorry

end curve_transformation_l4090_409063


namespace parabola_focal_chord_property_l4090_409070

-- Define the parabola
def parabola (p : ℝ) : Set (ℝ × ℝ) := {xy | xy.2^2 = 2*p*xy.1}

-- Define the focal chord
def is_focal_chord (p : ℝ) (P Q : ℝ × ℝ) : Prop :=
  P ∈ parabola p ∧ Q ∈ parabola p

-- Define the directrix
def directrix (p : ℝ) : Set (ℝ × ℝ) := {xy | xy.1 = -p}

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) := {xy | xy.2 = 0}

-- Define perpendicularity
def perpendicular (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

theorem parabola_focal_chord_property (p : ℝ) (P Q M N : ℝ × ℝ) :
  p > 0 →
  is_focal_chord p P Q →
  N ∈ directrix p →
  N ∈ x_axis →
  perpendicular P Q N Q →
  perpendicular P M M (0, 0) →
  M.2 = 0 →
  abs (P.1 - M.1) = abs (M.1 - Q.1) :=
by sorry

end parabola_focal_chord_property_l4090_409070


namespace kylie_stamps_l4090_409012

theorem kylie_stamps (kylie_stamps : ℕ) (nelly_stamps : ℕ) : 
  nelly_stamps = kylie_stamps + 44 →
  kylie_stamps + nelly_stamps = 112 →
  kylie_stamps = 34 := by
sorry

end kylie_stamps_l4090_409012


namespace geometric_sequence_fifth_term_l4090_409069

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 2 + a 4 = 5) :
  a 5 = 1/2 := by
sorry

end geometric_sequence_fifth_term_l4090_409069


namespace infinite_series_sum_l4090_409086

/-- The infinite series ∑(n=1 to ∞) (n³ + 2n² - n) / (n+3)! converges to 1/6 -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (n^3 + 2*n^2 - n : ℚ) / (Nat.factorial (n+3)) = 1/6 := by sorry

end infinite_series_sum_l4090_409086


namespace book_arrangements_eq_1440_l4090_409068

/-- The number of ways to arrange 8 books (3 Russian, 2 French, and 3 Italian) on a shelf,
    keeping the Russian books together and the French books together. -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 8
  let russian_books : ℕ := 3
  let french_books : ℕ := 2
  let italian_books : ℕ := 3
  let russian_unit : ℕ := 1
  let french_unit : ℕ := 1
  let total_units : ℕ := russian_unit + french_unit + italian_books
  Nat.factorial total_units * Nat.factorial russian_books * Nat.factorial french_books

/-- Theorem stating that the number of book arrangements is 1440. -/
theorem book_arrangements_eq_1440 : book_arrangements = 1440 := by
  sorry

end book_arrangements_eq_1440_l4090_409068


namespace associate_professor_items_l4090_409099

def CommitteeMeeting (associate_count : ℕ) (assistant_count : ℕ) 
  (total_pencils : ℕ) (total_charts : ℕ) : Prop :=
  associate_count + assistant_count = 9 ∧
  assistant_count = 11 ∧
  2 * assistant_count = 16 ∧
  total_pencils = 11 ∧
  total_charts = 16

theorem associate_professor_items :
  ∃! (associate_count : ℕ), CommitteeMeeting associate_count (9 - associate_count) 11 16 ∧
  associate_count = 1 ∧
  11 = 9 - associate_count ∧
  16 = 2 * (9 - associate_count) :=
sorry

end associate_professor_items_l4090_409099


namespace paint_mixture_intensity_l4090_409027

theorem paint_mixture_intensity 
  (original_intensity : ℝ) 
  (added_intensity : ℝ) 
  (replaced_fraction : ℝ) 
  (h1 : original_intensity = 0.6) 
  (h2 : added_intensity = 0.3) 
  (h3 : replaced_fraction = 2/3) : 
  (1 - replaced_fraction) * original_intensity + replaced_fraction * added_intensity = 0.4 := by
  sorry

end paint_mixture_intensity_l4090_409027


namespace ellipse_foci_y_axis_l4090_409039

theorem ellipse_foci_y_axis (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (9 - m) + y^2 / (m - 5) = 1 → 
    ∃ c : ℝ, c > 0 ∧ ∀ p : ℝ × ℝ, p.1^2 + p.2^2 = c^2 → 
      (0, c) ∈ {f : ℝ × ℝ | (f.1 - x)^2 + (f.2 - y)^2 + (f.1 + x)^2 + (f.2 - y)^2 = 4 * c^2}) →
  7 < m ∧ m < 9 :=
by sorry

end ellipse_foci_y_axis_l4090_409039


namespace min_value_sum_of_squares_l4090_409023

theorem min_value_sum_of_squares (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hsum : a + b + c = 9) : 
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) ≥ 9 := by
  sorry

end min_value_sum_of_squares_l4090_409023


namespace circle_with_AB_diameter_l4090_409034

-- Define the points A and B
def A : ℝ × ℝ := (-3, -5)
def B : ℝ × ℝ := (5, 1)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 25

-- Theorem statement
theorem circle_with_AB_diameter :
  ∀ x y : ℝ,
  circle_equation x y ↔ 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    ((x = (1 - t) * A.1 + t * B.1) ∧
     (y = (1 - t) * A.2 + t * B.2))) :=
by sorry

end circle_with_AB_diameter_l4090_409034


namespace p_necessary_but_not_sufficient_for_q_l4090_409040

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- Define what it means for p to be necessary but not sufficient for q
def necessary_but_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x)

-- Theorem statement
theorem p_necessary_but_not_sufficient_for_q :
  necessary_but_not_sufficient p q :=
sorry

end p_necessary_but_not_sufficient_for_q_l4090_409040
