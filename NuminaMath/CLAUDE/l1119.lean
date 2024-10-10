import Mathlib

namespace sum_of_remainders_is_93_l1119_111939

def is_valid_number (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ,
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    a = b + 2 ∧ b = c + 1 ∧ c = d + 1 ∧ d = e + 1 ∧
    0 ≤ e ∧ e < 10 ∧ 2 ≤ a ∧ a ≤ 6

def valid_numbers : List ℕ :=
  [23456, 34567, 45678, 56789, 67890]

theorem sum_of_remainders_is_93 :
  (valid_numbers.map (· % 43)).sum = 93 :=
sorry

end sum_of_remainders_is_93_l1119_111939


namespace concurrent_lines_through_circumcenter_l1119_111900

-- Define the basic structures
structure Point :=
  (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Line :=
  (p1 p2 : Point)

-- Define the properties
def isAcuteAngled (t : Triangle) : Prop := sorry

def altitudeFoot (t : Triangle) (v : Point) : Point := sorry

def perpendicularFoot (p : Point) (l : Line) : Point := sorry

def isOn (p : Point) (l : Line) : Prop := sorry

def intersectionPoint (l1 l2 : Line) : Point := sorry

def circumcenter (t : Triangle) : Point := sorry

-- Main theorem
theorem concurrent_lines_through_circumcenter 
  (t : Triangle) 
  (hAcute : isAcuteAngled t)
  (D : Point) (hD : D = altitudeFoot t t.A)
  (E : Point) (hE : E = altitudeFoot t t.B)
  (F : Point) (hF : F = altitudeFoot t t.C)
  (P : Point) (hP : P = perpendicularFoot t.A (Line.mk E F))
  (Q : Point) (hQ : Q = perpendicularFoot t.B (Line.mk F D))
  (R : Point) (hR : R = perpendicularFoot t.C (Line.mk D E)) :
  ∃ O : Point, 
    isOn O (Line.mk t.A P) ∧ 
    isOn O (Line.mk t.B Q) ∧ 
    isOn O (Line.mk t.C R) ∧
    O = circumcenter t :=
sorry

end concurrent_lines_through_circumcenter_l1119_111900


namespace remainder_problem_l1119_111968

theorem remainder_problem (n : ℤ) (h : n % 11 = 3) : (5 * n - 9) % 11 = 6 := by
  sorry

end remainder_problem_l1119_111968


namespace subset_complement_iff_m_range_l1119_111931

open Set Real

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 28 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2*x^2 - (5+m)*x + 5 ≤ 0}

-- State the theorem
theorem subset_complement_iff_m_range (m : ℝ) :
  B m ⊆ (univ \ A) ↔ m < -5 - 2*Real.sqrt 10 ∨ m > -5 + 2*Real.sqrt 10 := by
  sorry

end subset_complement_iff_m_range_l1119_111931


namespace two_dressers_capacity_l1119_111998

/-- The total number of pieces of clothing that can be held by two dressers -/
def total_clothing_capacity (first_dresser_drawers : ℕ) (first_dresser_capacity : ℕ) 
  (second_dresser_drawers : ℕ) (second_dresser_capacity : ℕ) : ℕ :=
  first_dresser_drawers * first_dresser_capacity + second_dresser_drawers * second_dresser_capacity

/-- Theorem stating the total clothing capacity of two specific dressers -/
theorem two_dressers_capacity : 
  total_clothing_capacity 12 8 6 10 = 156 := by
  sorry

end two_dressers_capacity_l1119_111998


namespace prize_distribution_l1119_111945

theorem prize_distribution (total_winners : ℕ) (min_award : ℝ) (max_award : ℝ) : 
  total_winners = 20 →
  min_award = 20 →
  max_award = 340 →
  (∃ (prize : ℝ), 
    prize > 0 ∧
    (∀ (winner : ℕ), winner ≤ total_winners → ∃ (award : ℝ), min_award ≤ award ∧ award ≤ max_award) ∧
    (2/5 * prize = 3/5 * total_winners * max_award) ∧
    prize = 10200) :=
by sorry

end prize_distribution_l1119_111945


namespace two_over_x_is_inverse_proportion_l1119_111916

/-- A function f is an inverse proportion function if there exists a constant k such that f(x) = k/x for all non-zero x. -/
def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function f(x) = 2/x is an inverse proportion function. -/
theorem two_over_x_is_inverse_proportion :
  is_inverse_proportion (λ x : ℝ => 2 / x) := by
  sorry


end two_over_x_is_inverse_proportion_l1119_111916


namespace crackers_per_friend_l1119_111925

/-- Given that Matthew had 23 crackers initially, has 11 crackers left, and gave equal numbers of crackers to 2 friends, prove that each friend ate 6 crackers. -/
theorem crackers_per_friend (initial_crackers : ℕ) (remaining_crackers : ℕ) (num_friends : ℕ) :
  initial_crackers = 23 →
  remaining_crackers = 11 →
  num_friends = 2 →
  (initial_crackers - remaining_crackers) / num_friends = 6 :=
by sorry

end crackers_per_friend_l1119_111925


namespace power_24_mod_15_l1119_111926

theorem power_24_mod_15 : 24^2377 % 15 = 9 := by
  sorry

end power_24_mod_15_l1119_111926


namespace students_liking_food_l1119_111965

theorem students_liking_food (total : ℕ) (dislike : ℕ) (like : ℕ) : 
  total = 814 → dislike = 431 → like = total - dislike → like = 383 := by
sorry

end students_liking_food_l1119_111965


namespace books_written_proof_l1119_111919

/-- The number of books written by Zig -/
def zig_books : ℕ := 60

/-- The number of books written by Flo -/
def flo_books : ℕ := zig_books / 4

/-- The number of books written by Tim -/
def tim_books : ℕ := flo_books / 2

/-- The total number of books written by Zig, Flo, and Tim -/
def total_books : ℕ := zig_books + flo_books + tim_books

theorem books_written_proof : total_books = 82 := by
  sorry

end books_written_proof_l1119_111919


namespace remainder_polynomial_division_l1119_111962

theorem remainder_polynomial_division (z : ℂ) : 
  ∃ (Q R : ℂ → ℂ), 
    (∀ z, z^2023 - 1 = (z^3 - 1) * (Q z) + R z) ∧ 
    (∃ (a b c : ℂ), ∀ z, R z = a*z^2 + b*z + c) ∧
    R z = z^2 + z - 1 := by
  sorry

end remainder_polynomial_division_l1119_111962


namespace log_sum_equality_fraction_sum_equality_l1119_111985

-- Part 1
theorem log_sum_equality : 2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) + 2^(Real.log 3 / Real.log 2) = 5 := by sorry

-- Part 2
theorem fraction_sum_equality : (5 + 1/16)^(1/2) + (-1)^(-1) / 0.75^(-2) + (2 + 10/27)^(-2/3) = 9/4 := by sorry

end log_sum_equality_fraction_sum_equality_l1119_111985


namespace sin_negative_31pi_over_6_l1119_111986

theorem sin_negative_31pi_over_6 : Real.sin (-31 * Real.pi / 6) = 1 / 2 := by
  sorry

end sin_negative_31pi_over_6_l1119_111986


namespace petes_number_l1119_111948

theorem petes_number : ∃ x : ℚ, 5 * (3 * x + 15) = 200 ∧ x = 25 / 3 := by
  sorry

end petes_number_l1119_111948


namespace jenny_sweets_problem_l1119_111924

theorem jenny_sweets_problem : ∃ n : ℕ+, 
  5 ∣ n ∧ 6 ∣ n ∧ ¬(12 ∣ n) ∧ n = 90 := by
  sorry

end jenny_sweets_problem_l1119_111924


namespace intersection_line_equation_l1119_111922

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 27 = 0

-- Define the line AB
def line_AB (x y : ℝ) : Prop := 3*x - 3*y - 10 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ A B : ℝ × ℝ,
  (circle_C1 A.1 A.2 ∧ circle_C2 A.1 A.2) →
  (circle_C1 B.1 B.2 ∧ circle_C2 B.1 B.2) →
  A ≠ B →
  line_AB A.1 A.2 ∧ line_AB B.1 B.2 :=
by sorry

end intersection_line_equation_l1119_111922


namespace gcf_of_lcms_l1119_111996

theorem gcf_of_lcms : Nat.gcd (Nat.lcm 9 21) (Nat.lcm 14 15) = 21 := by
  sorry

end gcf_of_lcms_l1119_111996


namespace min_value_expression_min_value_achievable_l1119_111969

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (a^2 + 1) / a + (2*b^2 + 1) / b ≥ 4 + 2*Real.sqrt 2 :=
by sorry

theorem min_value_achievable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2*b = 1 ∧
  (a^2 + 1) / a + (2*b^2 + 1) / b = 4 + 2*Real.sqrt 2 :=
by sorry

end min_value_expression_min_value_achievable_l1119_111969


namespace park_length_l1119_111949

theorem park_length (perimeter breadth : ℝ) (h1 : perimeter = 1000) (h2 : breadth = 200) :
  let length := (perimeter - 2 * breadth) / 2
  length = 300 :=
by
  sorry

#check park_length

end park_length_l1119_111949


namespace pencil_cost_is_25_l1119_111977

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := sorry

/-- The cost of a pen in cents -/
def pen_cost : ℕ := 80

/-- The total number of items (pens and pencils) bought -/
def total_items : ℕ := 36

/-- The number of pencils bought -/
def pencils_bought : ℕ := 16

/-- The total amount spent in cents -/
def total_spent : ℕ := 2000  -- 20 dollars = 2000 cents

theorem pencil_cost_is_25 : 
  pencil_cost = 25 ∧ 
  pencil_cost * pencils_bought + pen_cost * (total_items - pencils_bought) = total_spent :=
sorry

end pencil_cost_is_25_l1119_111977


namespace nth_equation_solutions_l1119_111905

theorem nth_equation_solutions (n : ℕ+) :
  let eq := fun x : ℝ => x + (n^2 + n) / x + (2*n + 1)
  eq (-n : ℝ) = 0 ∧ eq (-(n + 1) : ℝ) = 0 :=
by sorry

end nth_equation_solutions_l1119_111905


namespace diagonal_crosses_24_tiles_l1119_111994

/-- The number of tiles crossed by a diagonal line on a rectangular grid --/
def tiles_crossed (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- Proof that a diagonal on a 12x15 rectangle crosses 24 tiles --/
theorem diagonal_crosses_24_tiles :
  tiles_crossed 12 15 = 24 := by
  sorry

end diagonal_crosses_24_tiles_l1119_111994


namespace factorization_equality_l1119_111990

theorem factorization_equality (a : ℝ) : -9 - a^2 + 6*a = -(a - 3)^2 := by
  sorry

end factorization_equality_l1119_111990


namespace point_coordinates_wrt_origin_l1119_111906

/-- Given a point A in a Cartesian coordinate system with coordinates (-2, -3),
    its coordinates with respect to the origin are also (-2, -3). -/
theorem point_coordinates_wrt_origin :
  ∀ (A : ℝ × ℝ), A = (-2, -3) → A = (-2, -3) :=
by sorry

end point_coordinates_wrt_origin_l1119_111906


namespace percentage_problem_l1119_111915

theorem percentage_problem (x : ℝ) : 
  (15 / 100 * 40 = x / 100 * 16 + 2) → x = 25 := by
  sorry

end percentage_problem_l1119_111915


namespace completing_square_equivalence_l1119_111902

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 8*x + 1 = 0 ↔ (x - 4)^2 = 15 := by
sorry

end completing_square_equivalence_l1119_111902


namespace range_of_y_over_x_for_unit_modulus_complex_l1119_111913

theorem range_of_y_over_x_for_unit_modulus_complex (x y : ℝ) :
  (x - 2)^2 + y^2 = 1 →
  y ≠ 0 →
  ∃ k : ℝ, y = k * x ∧ k ∈ Set.Ioo (-Real.sqrt 3 / 3) 0 ∪ Set.Ioo 0 (Real.sqrt 3 / 3) :=
by sorry

end range_of_y_over_x_for_unit_modulus_complex_l1119_111913


namespace symmetric_point_is_correct_l1119_111970

/-- The line of symmetry --/
def line_of_symmetry (x y : ℝ) : Prop := x + 2*y - 10 = 0

/-- The original point --/
def original_point : ℝ × ℝ := (1, 2)

/-- The symmetric point --/
def symmetric_point : ℝ × ℝ := (3, 6)

/-- Checks if two points are symmetric with respect to a line --/
def is_symmetric (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  line_of_symmetry midpoint.1 midpoint.2 ∧
  (p2.2 - p1.2) * (1 : ℝ) = (p2.1 - p1.1) * (-2 : ℝ)

theorem symmetric_point_is_correct : 
  is_symmetric original_point symmetric_point :=
sorry

end symmetric_point_is_correct_l1119_111970


namespace pure_imaginary_complex_number_l1119_111978

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, (a + 2 * Complex.I) / (2 - Complex.I) = b * Complex.I) → a = 1 := by
  sorry

end pure_imaginary_complex_number_l1119_111978


namespace inequality_solution_set_l1119_111967

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3/5) 1 :=
by sorry

end inequality_solution_set_l1119_111967


namespace hannah_total_spending_l1119_111917

def hannah_fair_spending (initial_amount : ℝ) (ride_percent : ℝ) (game_percent : ℝ)
  (dessert_cost : ℝ) (cotton_candy_cost : ℝ) (hotdog_cost : ℝ) (keychain_cost : ℝ) : ℝ :=
  (initial_amount * ride_percent) + (initial_amount * game_percent) +
  dessert_cost + cotton_candy_cost + hotdog_cost + keychain_cost

theorem hannah_total_spending :
  hannah_fair_spending 80 0.35 0.25 7 4 5 6 = 70 := by
  sorry

end hannah_total_spending_l1119_111917


namespace expression_value_l1119_111959

theorem expression_value :
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 1
  x^2 * y * z - x * y * z^2 = 6 := by
sorry

end expression_value_l1119_111959


namespace grace_earnings_l1119_111958

theorem grace_earnings (weekly_charge : ℕ) (payment_interval : ℕ) (total_weeks : ℕ) (total_earnings : ℕ) : 
  weekly_charge = 300 →
  payment_interval = 2 →
  total_weeks = 6 →
  total_earnings = 1800 →
  total_weeks * weekly_charge = total_earnings :=
by
  sorry

end grace_earnings_l1119_111958


namespace cousin_distribution_l1119_111929

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers --/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 cousins and 5 rooms --/
def cousins : ℕ := 5
def rooms : ℕ := 5

/-- The main theorem: there are 52 ways to distribute the cousins into the rooms --/
theorem cousin_distribution : distribute cousins rooms = 52 := by sorry

end cousin_distribution_l1119_111929


namespace movie_production_l1119_111987

theorem movie_production (x : ℝ) : 
  (∃ y : ℝ, y = 1.25 * x ∧ 5 * (x + y) = 2475) → x = 220 :=
by sorry

end movie_production_l1119_111987


namespace exists_term_with_nine_l1119_111953

/-- Represents an arithmetic progression with natural number first term and common difference -/
structure ArithmeticProgression where
  first_term : ℕ
  common_difference : ℕ

/-- Predicate to check if a natural number contains the digit 9 -/
def contains_digit_nine (n : ℕ) : Prop := sorry

/-- Theorem stating that there exists a term in the arithmetic progression containing the digit 9 -/
theorem exists_term_with_nine (ap : ArithmeticProgression) : 
  ∃ (k : ℕ), contains_digit_nine (ap.first_term + k * ap.common_difference) := by sorry

end exists_term_with_nine_l1119_111953


namespace nancy_soap_packs_l1119_111963

/-- Proves that Nancy bought 6 packs of soap given the conditions -/
theorem nancy_soap_packs : 
  ∀ (bars_per_pack total_bars : ℕ),
    bars_per_pack = 5 →
    total_bars = 30 →
    total_bars / bars_per_pack = 6 := by
  sorry

end nancy_soap_packs_l1119_111963


namespace intersection_implies_c_18_l1119_111971

-- Define the functions
def f (x : ℝ) : ℝ := |x - 20| + |x + 18|
def g (c x : ℝ) : ℝ := x + c

-- Define the intersection condition
def unique_intersection (c : ℝ) : Prop :=
  ∃! x, f x = g c x

-- Theorem statement
theorem intersection_implies_c_18 :
  ∀ c : ℝ, unique_intersection c → c = 18 := by
  sorry

end intersection_implies_c_18_l1119_111971


namespace molecular_weight_AlPO4_correct_l1119_111976

/-- The molecular weight of AlPO4 in grams per mole -/
def molecular_weight_AlPO4 : ℝ := 122

/-- The number of moles given in the problem -/
def moles : ℝ := 4

/-- The total weight of the given moles of AlPO4 in grams -/
def total_weight : ℝ := 488

/-- Theorem: The molecular weight of AlPO4 is correct given the total weight of 4 moles -/
theorem molecular_weight_AlPO4_correct : 
  molecular_weight_AlPO4 * moles = total_weight :=
sorry

end molecular_weight_AlPO4_correct_l1119_111976


namespace jennifer_remaining_money_l1119_111951

def initial_amount : ℚ := 360
def sandwich_proportion : ℚ := 3/10
def museum_proportion : ℚ := 1/4
def book_proportion : ℚ := 35/100
def charity_proportion : ℚ := 1/8

theorem jennifer_remaining_money :
  let sandwich_cost := initial_amount * sandwich_proportion
  let museum_cost := initial_amount * museum_proportion
  let book_cost := initial_amount * book_proportion
  let total_spent := sandwich_cost + museum_cost + book_cost
  let remaining_before_charity := initial_amount - total_spent
  let charity_donation := remaining_before_charity * charity_proportion
  let final_remaining := remaining_before_charity - charity_donation
  final_remaining = 63/2 := by
sorry

end jennifer_remaining_money_l1119_111951


namespace power_sum_is_integer_l1119_111937

theorem power_sum_is_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/(x^n) = m :=
sorry

end power_sum_is_integer_l1119_111937


namespace sleep_difference_l1119_111964

def sleep_pattern (x : ℝ) : Prop :=
  let first_night := 6
  let second_night := x
  let third_night := x / 2
  let fourth_night := 3 * (x / 2)
  first_night + second_night + third_night + fourth_night = 30

theorem sleep_difference : ∃ x : ℝ, sleep_pattern x ∧ x - 6 = 2 := by
  sorry

end sleep_difference_l1119_111964


namespace complex_fraction_real_l1119_111950

theorem complex_fraction_real (a : ℝ) : 
  ((-a + Complex.I) / (1 - Complex.I)).im = 0 → a = 1 := by
  sorry

end complex_fraction_real_l1119_111950


namespace susan_babysitting_earnings_l1119_111904

def susan_earnings (initial : ℝ) : Prop :=
  let after_clothes := initial / 2
  let after_books := after_clothes / 2
  after_books = 150

theorem susan_babysitting_earnings :
  ∃ (initial : ℝ), susan_earnings initial ∧ initial = 600 :=
sorry

end susan_babysitting_earnings_l1119_111904


namespace x_plus_y_value_l1119_111907

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 2010)
  (eq2 : x + 2010 * Real.sin y = 2009)
  (y_range : π / 2 ≤ y ∧ y ≤ π) :
  x + y = 2011 + π := by
  sorry

end x_plus_y_value_l1119_111907


namespace expression_evaluation_l1119_111940

theorem expression_evaluation : 200 * (200 + 5) - (200 * 200 + 5) = 995 := by
  sorry

end expression_evaluation_l1119_111940


namespace product_of_sums_equals_difference_of_powers_l1119_111943

theorem product_of_sums_equals_difference_of_powers : 
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * 
  (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) = 7^128 - 5^128 := by
sorry

end product_of_sums_equals_difference_of_powers_l1119_111943


namespace exists_ten_digit_number_divisible_by_11_with_all_digits_l1119_111928

def is_ten_digit_number (n : ℕ) : Prop :=
  10^9 ≤ n ∧ n < 10^10

def contains_all_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, (n / 10^k) % 10 = d

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem exists_ten_digit_number_divisible_by_11_with_all_digits :
  ∃ n : ℕ, is_ten_digit_number n ∧ contains_all_digits n ∧ is_divisible_by_11 n :=
sorry

end exists_ten_digit_number_divisible_by_11_with_all_digits_l1119_111928


namespace range_of_a_for_inequality_l1119_111954

theorem range_of_a_for_inequality (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ 2^x * (3*x + a) < 1) ↔ a < 1 :=
sorry

end range_of_a_for_inequality_l1119_111954


namespace fraction_equation_solution_l1119_111909

theorem fraction_equation_solution (a b : ℝ) (h : a / b = 5 / 4) :
  ∃ x : ℝ, (4 * a + x * b) / (4 * a - x * b) = 4 ∧ x = 3 := by
  sorry

end fraction_equation_solution_l1119_111909


namespace least_n_satisfying_conditions_l1119_111973

theorem least_n_satisfying_conditions : ∃ n : ℕ,
  n > 1 ∧
  2*n % 3 = 2 ∧
  3*n % 4 = 3 ∧
  4*n % 5 = 4 ∧
  5*n % 6 = 5 ∧
  (∀ m : ℕ, m > 1 ∧ 
    2*m % 3 = 2 ∧
    3*m % 4 = 3 ∧
    4*m % 5 = 4 ∧
    5*m % 6 = 5 → m ≥ n) ∧
  n = 61 :=
by sorry

end least_n_satisfying_conditions_l1119_111973


namespace z_in_first_quadrant_l1119_111908

/-- Given a complex number z such that z / (1 - z) = 2i, prove that z is in the first quadrant -/
theorem z_in_first_quadrant (z : ℂ) (h : z / (1 - z) = Complex.I * 2) : 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end z_in_first_quadrant_l1119_111908


namespace wilsons_theorem_l1119_111984

theorem wilsons_theorem (n : ℕ) (h : n ≥ 2) :
  Nat.Prime n ↔ (Nat.factorial (n - 1) % n = n - 1) := by
  sorry

end wilsons_theorem_l1119_111984


namespace contrapositive_example_l1119_111947

theorem contrapositive_example (a b : ℝ) : 
  (∀ a b, a = 0 → a * b = 0) ↔ (∀ a b, a * b ≠ 0 → a ≠ 0) := by sorry

end contrapositive_example_l1119_111947


namespace coin_difference_is_nine_l1119_111927

def coin_denominations : List Nat := [5, 10, 25, 50]

def amount_to_pay : Nat := 55

def min_coins (denominations : List Nat) (amount : Nat) : Nat :=
  sorry

def max_coins (denominations : List Nat) (amount : Nat) : Nat :=
  sorry

theorem coin_difference_is_nine :
  max_coins coin_denominations amount_to_pay - min_coins coin_denominations amount_to_pay = 9 :=
by sorry

end coin_difference_is_nine_l1119_111927


namespace probability_five_odd_in_six_rolls_l1119_111980

theorem probability_five_odd_in_six_rolls : 
  let n : ℕ := 6  -- number of rolls
  let k : ℕ := 5  -- number of desired odd rolls
  let p : ℚ := 1/2  -- probability of rolling an odd number on a single roll
  Nat.choose n k * p^k * (1-p)^(n-k) = 3/32 := by
sorry

end probability_five_odd_in_six_rolls_l1119_111980


namespace a_2k_minus_1_has_three_prime_factors_l1119_111993

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 6
  | (n + 2) => 4 * a (n + 1) - a n + 2

theorem a_2k_minus_1_has_three_prime_factors (k : ℕ) (h : k > 3) :
  ∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  (p * q * r : ℤ) ∣ a (2^k - 1) :=
sorry

end a_2k_minus_1_has_three_prime_factors_l1119_111993


namespace max_value_of_f_l1119_111946

def f (x : ℝ) : ℝ := -3 * x^2 + 18

theorem max_value_of_f :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M ∧ M = 18 := by
  sorry

end max_value_of_f_l1119_111946


namespace largest_n_for_product_2016_l1119_111961

/-- An arithmetic sequence with integer terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product_2016 :
  ∀ a b : ℕ → ℤ,
  ArithmeticSequence a →
  ArithmeticSequence b →
  a 1 = 1 →
  b 1 = 1 →
  a 2 ≤ b 2 →
  (∃ n : ℕ, a n * b n = 2016) →
  (∀ m : ℕ, (∃ n : ℕ, n > m ∧ a n * b n = 2016) → m ≤ 32) :=
sorry

end largest_n_for_product_2016_l1119_111961


namespace wizard_elixir_combinations_l1119_111955

/-- The number of herbs available for the wizard's elixir. -/
def num_herbs : ℕ := 4

/-- The number of crystals available for the wizard's elixir. -/
def num_crystals : ℕ := 6

/-- The number of incompatible combinations due to the first problematic crystal. -/
def incompatible_combinations_1 : ℕ := 2

/-- The number of incompatible combinations due to the second problematic crystal. -/
def incompatible_combinations_2 : ℕ := 1

/-- The total number of viable combinations for the wizard's elixir. -/
def viable_combinations : ℕ := num_herbs * num_crystals - (incompatible_combinations_1 + incompatible_combinations_2)

theorem wizard_elixir_combinations :
  viable_combinations = 21 :=
sorry

end wizard_elixir_combinations_l1119_111955


namespace linda_coloring_books_l1119_111997

/-- Represents Linda's purchase --/
structure Purchase where
  coloringBookPrice : ℝ
  coloringBookCount : ℕ
  peanutPackPrice : ℝ
  peanutPackCount : ℕ
  stuffedAnimalPrice : ℝ
  totalPaid : ℝ

/-- Theorem stating the number of coloring books Linda bought --/
theorem linda_coloring_books (p : Purchase) 
  (h1 : p.coloringBookPrice = 4)
  (h2 : p.peanutPackPrice = 1.5)
  (h3 : p.peanutPackCount = 4)
  (h4 : p.stuffedAnimalPrice = 11)
  (h5 : p.totalPaid = 25)
  (h6 : p.coloringBookPrice * p.coloringBookCount + 
        p.peanutPackPrice * p.peanutPackCount + 
        p.stuffedAnimalPrice = p.totalPaid) :
  p.coloringBookCount = 2 := by
  sorry

end linda_coloring_books_l1119_111997


namespace jessica_has_62_marbles_l1119_111935

-- Define the number of marbles each person has
def dennis_marbles : ℕ := 70
def kurt_marbles : ℕ := dennis_marbles - 45
def laurie_marbles : ℕ := kurt_marbles + 12
def jessica_marbles : ℕ := laurie_marbles + 25

-- Theorem to prove
theorem jessica_has_62_marbles : jessica_marbles = 62 := by
  sorry

end jessica_has_62_marbles_l1119_111935


namespace meaningful_expression_l1119_111923

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 3)) ↔ x > 3 :=
by sorry

end meaningful_expression_l1119_111923


namespace history_not_statistics_l1119_111933

theorem history_not_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ)
  (h_total : total = 90)
  (h_history : history = 36)
  (h_statistics : statistics = 32)
  (h_history_or_statistics : history_or_statistics = 57) :
  history - (history + statistics - history_or_statistics) = 25 := by
  sorry

end history_not_statistics_l1119_111933


namespace mod_nine_equivalence_l1119_111920

theorem mod_nine_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2222 ≡ n [ZMOD 9] ∧ n = 6 := by
  sorry

end mod_nine_equivalence_l1119_111920


namespace contrapositive_not_true_l1119_111972

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b ∨ b = k • a

/-- Two vectors have the same direction if they are positive scalar multiples of each other -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ (a = k • b ∨ b = k • a)

/-- The original proposition -/
def original_proposition : Prop :=
  ∀ a b : ℝ × ℝ, collinear a b → same_direction a b

/-- The contrapositive of the original proposition -/
def contrapositive : Prop :=
  ∀ a b : ℝ × ℝ, ¬ same_direction a b → ¬ collinear a b

theorem contrapositive_not_true : ¬ contrapositive := by
  sorry

end contrapositive_not_true_l1119_111972


namespace total_new_games_is_92_l1119_111983

/-- The number of new games Katie has -/
def katie_new_games : ℕ := 84

/-- The number of new games Katie's friends have -/
def friends_new_games : ℕ := 8

/-- The total number of new games Katie and her friends have together -/
def total_new_games : ℕ := katie_new_games + friends_new_games

/-- Theorem stating that the total number of new games is 92 -/
theorem total_new_games_is_92 : total_new_games = 92 := by sorry

end total_new_games_is_92_l1119_111983


namespace soccer_league_games_l1119_111999

/-- The number of teams in the soccer league -/
def num_teams : ℕ := 12

/-- The number of times each pair of teams plays against each other -/
def games_per_pair : ℕ := 4

/-- The total number of games played in the season -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2 * games_per_pair

theorem soccer_league_games :
  total_games = 264 :=
sorry

end soccer_league_games_l1119_111999


namespace increasing_function_implies_a_leq_neg_two_l1119_111941

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- State the theorem
theorem increasing_function_implies_a_leq_neg_two :
  ∀ a : ℝ, (∀ x y : ℝ, -2 < x ∧ x < y ∧ y < 2 → f a x < f a y) →
  a ≤ -2 := by
  sorry

end increasing_function_implies_a_leq_neg_two_l1119_111941


namespace intersection_with_complement_l1119_111932

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem intersection_with_complement :
  A ∩ (U \ B) = {1} := by sorry

end intersection_with_complement_l1119_111932


namespace tv_show_watch_time_l1119_111975

/-- Calculates the total watch time for a TV show with regular seasons and a final season -/
def total_watch_time (regular_seasons : ℕ) (episodes_per_regular_season : ℕ) 
  (extra_episodes_final_season : ℕ) (hours_per_episode : ℚ) : ℚ :=
  let total_episodes := regular_seasons * episodes_per_regular_season + 
    (episodes_per_regular_season + extra_episodes_final_season)
  total_episodes * hours_per_episode

/-- Theorem stating that the total watch time for the given TV show is 112 hours -/
theorem tv_show_watch_time : 
  total_watch_time 9 22 4 (1/2) = 112 := by sorry

end tv_show_watch_time_l1119_111975


namespace impossible_mixture_l1119_111918

/-- Represents the properties of an ingredient -/
structure Ingredient :=
  (volume : ℝ)
  (water_content : ℝ)

/-- Proves that it's impossible to create a mixture with exactly 20% water content
    using the given volumes of tomato juice, tomato paste, and secret sauce -/
theorem impossible_mixture
  (tomato_juice : Ingredient)
  (tomato_paste : Ingredient)
  (secret_sauce : Ingredient)
  (h1 : tomato_juice.volume = 40)
  (h2 : tomato_juice.water_content = 0.9)
  (h3 : tomato_paste.volume = 20)
  (h4 : tomato_paste.water_content = 0.45)
  (h5 : secret_sauce.volume = 10)
  (h6 : secret_sauce.water_content = 0.7)
  : ¬ ∃ (x y z : ℝ),
    0 ≤ x ∧ x ≤ tomato_juice.volume ∧
    0 ≤ y ∧ y ≤ tomato_paste.volume ∧
    0 ≤ z ∧ z ≤ secret_sauce.volume ∧
    (x * tomato_juice.water_content + y * tomato_paste.water_content + z * secret_sauce.water_content) / (x + y + z) = 0.2 :=
sorry


end impossible_mixture_l1119_111918


namespace prob_different_colors_bag_l1119_111910

/-- Represents the number of chips of each color in the bag -/
structure ChipCounts where
  blue : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of drawing two chips of different colors -/
def probDifferentColors (counts : ChipCounts) : ℚ :=
  let total := counts.blue + counts.red + counts.yellow + counts.green
  let probBlue := counts.blue / total
  let probRed := counts.red / total
  let probYellow := counts.yellow / total
  let probGreen := counts.green / total
  let probDiffAfterBlue := (total - counts.blue) / total
  let probDiffAfterRed := (total - counts.red) / total
  let probDiffAfterYellow := (total - counts.yellow) / total
  let probDiffAfterGreen := (total - counts.green) / total
  probBlue * probDiffAfterBlue + probRed * probDiffAfterRed +
  probYellow * probDiffAfterYellow + probGreen * probDiffAfterGreen

/-- The main theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors_bag :
  probDifferentColors { blue := 6, red := 5, yellow := 4, green := 3 } = 119 / 162 := by
  sorry


end prob_different_colors_bag_l1119_111910


namespace z_purely_imaginary_z_in_fourth_quadrant_l1119_111966

/-- Definition of the complex number z as a function of m -/
def z (m : ℝ) : ℂ := Complex.mk (2*m^2 - 7*m + 6) (m^2 - m - 2)

/-- z is purely imaginary iff m = 3/2 -/
theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * (z m).im ↔ m = 3/2 := by
  sorry

/-- z is in the fourth quadrant iff -1 < m < 3/2 -/
theorem z_in_fourth_quadrant (m : ℝ) : 
  (z m).re > 0 ∧ (z m).im < 0 ↔ -1 < m ∧ m < 3/2 := by
  sorry

end z_purely_imaginary_z_in_fourth_quadrant_l1119_111966


namespace expected_value_5X_plus_4_l1119_111981

/-- Distribution of random variable X -/
structure Distribution where
  p0 : ℝ
  p2 : ℝ
  p4 : ℝ
  sum_to_one : p0 + p2 + p4 = 1
  non_negative : p0 ≥ 0 ∧ p2 ≥ 0 ∧ p4 ≥ 0

/-- Expected value of a random variable -/
def expected_value (d : Distribution) : ℝ := 0 * d.p0 + 2 * d.p2 + 4 * d.p4

/-- Theorem: Expected value of 5X+4 equals 16 -/
theorem expected_value_5X_plus_4 (d : Distribution) 
  (h1 : d.p0 = 0.3) 
  (h2 : d.p4 = 0.5) : 
  5 * expected_value d + 4 = 16 := by
  sorry

end expected_value_5X_plus_4_l1119_111981


namespace min_a_for_quadratic_roots_in_unit_interval_l1119_111956

theorem min_a_for_quadratic_roots_in_unit_interval :
  ∀ (a b c : ℤ) (α β : ℝ),
    a > 0 →
    (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = α ∨ x = β) →
    0 < α →
    α < β →
    β < 1 →
    a ≥ 5 ∧ ∃ (a₀ b₀ c₀ : ℤ) (α₀ β₀ : ℝ),
      a₀ = 5 ∧
      a₀ > 0 ∧
      (∀ x : ℝ, a₀ * x^2 + b₀ * x + c₀ = 0 ↔ x = α₀ ∨ x = β₀) ∧
      0 < α₀ ∧
      α₀ < β₀ ∧
      β₀ < 1 :=
by sorry

end min_a_for_quadratic_roots_in_unit_interval_l1119_111956


namespace days_off_per_month_l1119_111912

def total_holidays : ℕ := 36
def months_in_year : ℕ := 12

theorem days_off_per_month :
  total_holidays / months_in_year = 3 := by sorry

end days_off_per_month_l1119_111912


namespace number_equation_solution_l1119_111960

theorem number_equation_solution : ∃ x : ℝ, (0.68 * x - 5) / 3 = 17 := by
  sorry

end number_equation_solution_l1119_111960


namespace closest_integer_to_largest_root_squared_l1119_111938

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 8*x^2 - 2*x + 3

-- State the theorem
theorem closest_integer_to_largest_root_squared : 
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ 
    (a > b ∧ a > c) ∧
    (abs (a^2 - 67) < 1) :=
by sorry

end closest_integer_to_largest_root_squared_l1119_111938


namespace geometric_sequence_common_ratio_l1119_111989

/-- A geometric sequence is defined by its first term and common ratio. -/
def GeometricSequence (a : ℚ) (r : ℚ) : ℕ → ℚ := fun n => a * r^(n - 1)

/-- The common ratio of a geometric sequence. -/
def CommonRatio (seq : ℕ → ℚ) : ℚ := seq 2 / seq 1

theorem geometric_sequence_common_ratio :
  let seq := GeometricSequence 16 (-3/2)
  (seq 1 = 16) ∧ (seq 2 = -24) ∧ (seq 3 = 36) ∧ (seq 4 = -54) →
  CommonRatio seq = -3/2 := by
sorry

end geometric_sequence_common_ratio_l1119_111989


namespace equal_vector_sums_implies_equilateral_or_equal_l1119_111934

-- Define the circle and points
def Circle := {p : ℂ | ∃ r : ℝ, r > 0 ∧ Complex.abs p = r}

-- Define the property of equal vector sums
def EqualVectorSums (A B C : ℂ) : Prop :=
  Complex.abs (A + B) = Complex.abs (B + C) ∧ 
  Complex.abs (B + C) = Complex.abs (C + A)

-- Define an equilateral triangle
def IsEquilateralTriangle (A B C : ℂ) : Prop :=
  Complex.abs (A - B) = Complex.abs (B - C) ∧
  Complex.abs (B - C) = Complex.abs (C - A)

-- State the theorem
theorem equal_vector_sums_implies_equilateral_or_equal 
  (A B C : ℂ) (hA : A ∈ Circle) (hB : B ∈ Circle) (hC : C ∈ Circle) 
  (hEqual : EqualVectorSums A B C) :
  A = B ∧ B = C ∨ IsEquilateralTriangle A B C := by
  sorry

end equal_vector_sums_implies_equilateral_or_equal_l1119_111934


namespace two_digit_sum_product_equality_l1119_111930

/-- P(n) is the product of the digits of n -/
def P (n : ℕ) : ℕ := sorry

/-- S(n) is the sum of the digits of n -/
def S (n : ℕ) : ℕ := sorry

/-- A two-digit number can be represented as 10a + b where a ≠ 0 -/
def isTwoDigit (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 10 * a + b

theorem two_digit_sum_product_equality :
  ∀ n : ℕ, isTwoDigit n → (n = P n + S n ↔ ∃ a : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ n = 10 * a + 9) :=
sorry

end two_digit_sum_product_equality_l1119_111930


namespace nesbitts_inequality_l1119_111974

theorem nesbitts_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 ∧
  (a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) := by
  sorry

end nesbitts_inequality_l1119_111974


namespace probability_red_second_given_white_first_l1119_111979

-- Define the total number of balls and the number of each color
def total_balls : ℕ := 5
def red_balls : ℕ := 3
def white_balls : ℕ := 2

-- Define the probability of drawing a red ball second, given the first is white
def prob_red_second_given_white_first : ℚ := 3 / 4

-- Theorem statement
theorem probability_red_second_given_white_first :
  (red_balls : ℚ) / (total_balls - 1) = prob_red_second_given_white_first :=
by sorry

end probability_red_second_given_white_first_l1119_111979


namespace students_with_both_fruits_l1119_111911

theorem students_with_both_fruits (apples bananas only_one : ℕ) 
  (h1 : apples = 12)
  (h2 : bananas = 8)
  (h3 : only_one = 10) :
  apples + bananas - only_one = 5 := by
  sorry

end students_with_both_fruits_l1119_111911


namespace arithmetic_sequence_length_example_l1119_111988

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ aₙ d : ℤ) : ℤ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The arithmetic sequence with first term 2, last term 2017, 
    and common difference 5 has 404 terms -/
theorem arithmetic_sequence_length_example : 
  arithmeticSequenceLength 2 2017 5 = 404 := by
  sorry

end arithmetic_sequence_length_example_l1119_111988


namespace probability_is_one_l1119_111901

def card_set : Finset ℕ := {1, 3, 4, 6, 7, 9}

def probability_less_than_or_equal_to_9 : ℚ :=
  (card_set.filter (λ x => x ≤ 9)).card / card_set.card

theorem probability_is_one :
  probability_less_than_or_equal_to_9 = 1 := by
  sorry

end probability_is_one_l1119_111901


namespace remaining_distance_to_grandma_l1119_111952

theorem remaining_distance_to_grandma (total_distance driven_first driven_second : ℕ) 
  (h1 : total_distance = 78)
  (h2 : driven_first = 35)
  (h3 : driven_second = 18) : 
  total_distance - (driven_first + driven_second) = 25 := by
  sorry

end remaining_distance_to_grandma_l1119_111952


namespace trigonometric_equality_l1119_111957

theorem trigonometric_equality : 
  (2 * Real.sin (47 * π / 180) - Real.sqrt 3 * Real.sin (17 * π / 180)) / Real.cos (17 * π / 180) = 1 := by
  sorry

end trigonometric_equality_l1119_111957


namespace complex_equation_solution_l1119_111944

theorem complex_equation_solution (b : ℂ) : (1 + b * Complex.I) * Complex.I = -1 + Complex.I → b = 1 := by
  sorry

end complex_equation_solution_l1119_111944


namespace find_a_l1119_111914

-- Define the sets U and A
def U (a : ℝ) : Set ℝ := {2, 3, a^2 + 2*a - 3}
def A (a : ℝ) : Set ℝ := {|2*a - 1|, 2}

-- Define the theorem
theorem find_a : ∃ (a : ℝ), 
  (U a \ A a = {5}) ∧ 
  (A a ⊆ U a) ∧
  (a = 2) := by sorry

end find_a_l1119_111914


namespace sphere_volume_from_surface_area_l1119_111991

/-- Given a sphere with surface area 16π cm², prove its volume is 32π/3 cm³ -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * π * r^2 = 16 * π → (4/3) * π * r^3 = (32 * π)/3 := by
sorry

end sphere_volume_from_surface_area_l1119_111991


namespace complementary_angles_ratio_l1119_111903

theorem complementary_angles_ratio (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Angles are positive
  a + b = 90 ∧     -- Angles are complementary
  a / b = 5 / 4 →  -- Ratio of angles is 5:4
  a = 50 :=        -- Larger angle is 50°
by sorry

end complementary_angles_ratio_l1119_111903


namespace quadratic_no_solution_b_range_l1119_111992

theorem quadratic_no_solution_b_range (b : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + 1 > 0) → -2 < b ∧ b < 2 := by
  sorry

end quadratic_no_solution_b_range_l1119_111992


namespace chord_intersection_parameter_l1119_111936

/-- Given a line and a circle, prove that the parameter a equals 1 when they intersect to form a chord of length √2. -/
theorem chord_intersection_parameter (a : ℝ) : a > 0 → ∃ (x y : ℝ),
  (x + y + a = 0) ∧ (x^2 + y^2 = a) ∧ 
  (∃ (x1 y1 x2 y2 : ℝ), (x1 + y1 + a = 0) ∧ (x2 + y2 + a = 0) ∧
                        (x1^2 + y1^2 = a) ∧ (x2^2 + y2^2 = a) ∧
                        ((x1 - x2)^2 + (y1 - y2)^2 = 2)) →
  a = 1 := by sorry

end chord_intersection_parameter_l1119_111936


namespace athlete_arrangements_l1119_111921

def male_athletes : ℕ := 7
def female_athletes : ℕ := 3

theorem athlete_arrangements :
  let total_athletes := male_athletes + female_athletes
  let arrangements_case1 := (male_athletes.factorial) * (male_athletes - 1) * (male_athletes - 2) * (male_athletes - 3)
  let arrangements_case2 := 2 * (female_athletes.factorial) * (male_athletes.factorial)
  let arrangements_case3 := (total_athletes + 1).factorial * (female_athletes.factorial)
  (arrangements_case1 = 604800) ∧
  (arrangements_case2 = 60480) ∧
  (arrangements_case3 = 241920) := by
  sorry

#eval male_athletes.factorial * (male_athletes - 1) * (male_athletes - 2) * (male_athletes - 3)
#eval 2 * female_athletes.factorial * male_athletes.factorial
#eval (male_athletes + female_athletes + 1).factorial * female_athletes.factorial

end athlete_arrangements_l1119_111921


namespace largest_multiple_of_45_with_8_and_0_l1119_111942

/-- A function that checks if a natural number consists only of digits 8 and 0 -/
def onlyEightAndZero (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 8 ∨ d = 0

/-- The largest positive multiple of 45 consisting only of digits 8 and 0 -/
def m : ℕ := sorry

theorem largest_multiple_of_45_with_8_and_0 :
  m % 45 = 0 ∧
  onlyEightAndZero m ∧
  (∀ k : ℕ, k > m → k % 45 = 0 → ¬onlyEightAndZero k) ∧
  m / 45 = 197530 :=
sorry

end largest_multiple_of_45_with_8_and_0_l1119_111942


namespace apartments_per_floor_l1119_111982

theorem apartments_per_floor (num_buildings : ℕ) (floors_per_building : ℕ) 
  (doors_per_apartment : ℕ) (total_doors : ℕ) :
  num_buildings = 2 →
  floors_per_building = 12 →
  doors_per_apartment = 7 →
  total_doors = 1008 →
  (total_doors / doors_per_apartment) / (num_buildings * floors_per_building) = 6 :=
by
  sorry

end apartments_per_floor_l1119_111982


namespace walk_distance_difference_l1119_111995

theorem walk_distance_difference (total_distance susan_distance : ℕ) 
  (h1 : total_distance = 15)
  (h2 : susan_distance = 9) :
  susan_distance - (total_distance - susan_distance) = 3 :=
by sorry

end walk_distance_difference_l1119_111995
