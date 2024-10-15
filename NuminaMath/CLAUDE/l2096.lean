import Mathlib

namespace NUMINAMATH_CALUDE_simplify_trig_fraction_l2096_209634

theorem simplify_trig_fraction (x : ℝ) :
  (3 + 2 * Real.sin x + 2 * Real.cos x) / (3 + 2 * Real.sin x - 2 * Real.cos x) = 
  3 / 5 + 2 / 5 * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_fraction_l2096_209634


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_l2096_209686

/-- The sum of distinct powers of 2 that equals 72 -/
def sum_of_powers (a b c : ℕ) : Prop :=
  2^a + 2^b + 2^c = 72 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The least sum of exponents when expressing 72 as a sum of at least three distinct powers of 2 -/
theorem least_sum_of_exponents :
  ∃ (a b c : ℕ), sum_of_powers a b c ∧
    ∀ (x y z : ℕ), sum_of_powers x y z → a + b + c ≤ x + y + z ∧ a + b + c = 9 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_l2096_209686


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2096_209620

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q)
  (h_pos : q > 0)
  (h_equality : a 3 * a 9 = (a 5)^2) :
  q = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2096_209620


namespace NUMINAMATH_CALUDE_black_card_probability_l2096_209615

theorem black_card_probability (total_cards : ℕ) (black_cards : ℕ) 
  (h_total : total_cards = 52) 
  (h_black : black_cards = 17) : 
  (black_cards * (black_cards - 1) * (black_cards - 2)) / 
  (total_cards * (total_cards - 1) * (total_cards - 2)) = 40 / 1301 := by
  sorry

end NUMINAMATH_CALUDE_black_card_probability_l2096_209615


namespace NUMINAMATH_CALUDE_three_points_collinear_l2096_209637

/-- Three points lie on the same line if and only if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem three_points_collinear (b : ℝ) :
  collinear (4, -7) (-2*b + 3, 5) (3*b + 4, 3) → b = -5/28 := by
  sorry

#check three_points_collinear

end NUMINAMATH_CALUDE_three_points_collinear_l2096_209637


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_range_l2096_209622

open Real

/-- The function f(x) = e^x - (a-1)x + 1 is monotonically decreasing on [0,1] -/
def is_monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ (Set.Icc 0 1) → y ∈ (Set.Icc 0 1) → x ≤ y → f x ≥ f y

/-- The main theorem -/
theorem monotone_decreasing_implies_a_range 
  (a : ℝ) 
  (f : ℝ → ℝ) 
  (h : f = fun x ↦ exp x - (a - 1) * x + 1) 
  (h_monotone : is_monotone_decreasing f) : 
  a ∈ Set.Ici (exp 1 + 1) := by
sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_range_l2096_209622


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2096_209630

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 5*x + 3) * (x^2 + 9*x + 20) + (x^2 + 7*x - 8) = 
  (x^2 + 7*x + 8) * (x^2 + 7*x + 14) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2096_209630


namespace NUMINAMATH_CALUDE_greatest_abcba_divisible_by_11_l2096_209678

/-- Represents a five-digit number in the form AB,CBA --/
structure ABCBA where
  a : Nat
  b : Nat
  c : Nat
  value : Nat := a * 10000 + b * 1000 + c * 100 + b * 10 + a

/-- Checks if the digits a, b, and c are valid for our problem --/
def valid_digits (a b c : Nat) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10

theorem greatest_abcba_divisible_by_11 :
  ∃ (n : ABCBA), 
    valid_digits n.a n.b n.c ∧ 
    n.value % 11 = 0 ∧
    n.value = 96569 ∧
    (∀ (m : ABCBA), valid_digits m.a m.b m.c → m.value % 11 = 0 → m.value ≤ n.value) := by
  sorry

end NUMINAMATH_CALUDE_greatest_abcba_divisible_by_11_l2096_209678


namespace NUMINAMATH_CALUDE_susie_babysitting_rate_l2096_209625

/-- Susie's babysitting scenario -/
theorem susie_babysitting_rate :
  ∀ (rate : ℚ),
  (∀ (day : ℕ), day ≤ 7 → day * (3 * rate) = day * (3 * rate)) →  -- She works 3 hours every day
  (3/10 + 2/5) * (7 * (3 * rate)) + 63 = 7 * (3 * rate) →  -- Spent fractions and remaining money
  rate = 10 := by
sorry

end NUMINAMATH_CALUDE_susie_babysitting_rate_l2096_209625


namespace NUMINAMATH_CALUDE_kates_hair_length_l2096_209633

theorem kates_hair_length (logan_hair emily_hair kate_hair : ℝ) : 
  logan_hair = 20 →
  emily_hair = logan_hair + 6 →
  kate_hair = emily_hair / 2 →
  kate_hair = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_kates_hair_length_l2096_209633


namespace NUMINAMATH_CALUDE_right_triangle_sin_A_l2096_209667

theorem right_triangle_sin_A (A B C : Real) :
  -- Right triangle ABC with ∠B = 90°
  0 < A ∧ A < Real.pi / 2 →
  0 < C ∧ C < Real.pi / 2 →
  A + C = Real.pi / 2 →
  -- 3 tan A = 4
  3 * Real.tan A = 4 →
  -- Conclusion: sin A = 4/5
  Real.sin A = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sin_A_l2096_209667


namespace NUMINAMATH_CALUDE_bus_system_daily_passengers_l2096_209636

def total_people : ℕ := 109200000
def num_weeks : ℕ := 13
def days_per_week : ℕ := 7

theorem bus_system_daily_passengers : 
  total_people / (num_weeks * days_per_week) = 1200000 := by
  sorry

end NUMINAMATH_CALUDE_bus_system_daily_passengers_l2096_209636


namespace NUMINAMATH_CALUDE_chord_length_l2096_209676

/-- The length of the chord intercepted by a line on a circle -/
theorem chord_length (x y : ℝ) : 
  let circle := {(x, y) | x^2 + y^2 - 2*x - 4*y = 0}
  let line := {(x, y) | x + 2*y - 5 + Real.sqrt 5 = 0}
  let chord := circle ∩ line
  (∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_chord_length_l2096_209676


namespace NUMINAMATH_CALUDE_fraction_simplification_l2096_209668

theorem fraction_simplification :
  (1 / 5 + 1 / 3) / (3 / 4 - 1 / 8) = 64 / 75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2096_209668


namespace NUMINAMATH_CALUDE_park_visitors_difference_l2096_209674

theorem park_visitors_difference (total : ℕ) (bikers : ℕ) (hikers : ℕ) :
  total = 676 →
  bikers = 249 →
  total = bikers + hikers →
  hikers > bikers →
  hikers - bikers = 178 := by
sorry

end NUMINAMATH_CALUDE_park_visitors_difference_l2096_209674


namespace NUMINAMATH_CALUDE_min_tablets_extracted_l2096_209627

theorem min_tablets_extracted (total_A : ℕ) (total_B : ℕ) : 
  total_A = 10 → total_B = 10 → 
  ∃ (min_extracted : ℕ), 
    (∀ (n : ℕ), n < min_extracted → 
      ∃ (a b : ℕ), a + b = n ∧ (a < 2 ∨ b < 2)) ∧
    (∀ (a b : ℕ), a + b = min_extracted → a ≥ 2 ∧ b ≥ 2) ∧
    min_extracted = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_tablets_extracted_l2096_209627


namespace NUMINAMATH_CALUDE_pyramid_multiplication_l2096_209618

theorem pyramid_multiplication (z x : ℕ) : z = 2 → x = 24 →
  (12 * x = 84 ∧ x * 7 = 168 ∧ 12 * z = x) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_multiplication_l2096_209618


namespace NUMINAMATH_CALUDE_village_population_l2096_209608

theorem village_population (P : ℝ) : 
  P > 0 →
  (P * 1.05 * 0.95 = 9975) →
  P = 10000 := by
sorry

end NUMINAMATH_CALUDE_village_population_l2096_209608


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2096_209698

theorem inequality_solution_set (x : ℝ) :
  x ≠ -7 →
  ((x^2 - 49) / (x + 7) < 0) ↔ (x < -7 ∨ (-7 < x ∧ x < 7)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2096_209698


namespace NUMINAMATH_CALUDE_range_of_f_l2096_209649

def f (x : ℝ) : ℝ := x^2 + 1

theorem range_of_f :
  Set.range f = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2096_209649


namespace NUMINAMATH_CALUDE_det_inequality_equiv_l2096_209616

/-- Definition of a second-order determinant -/
def secondOrderDet (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating the equivalence of the determinant inequality and the simplified inequality -/
theorem det_inequality_equiv (x : ℝ) :
  secondOrderDet 2 (3 - x) 1 x > 0 ↔ 3 * x - 3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_det_inequality_equiv_l2096_209616


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_ratio_l2096_209611

/-- Given vectors a and b are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_imply_ratio (x : ℝ) :
  let a : ℝ × ℝ := (Real.sin x, -1)
  let b : ℝ × ℝ := (Real.cos x, 2)
  are_parallel a b →
  (Real.cos x - Real.sin x) / (Real.cos x + Real.sin x) = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_ratio_l2096_209611


namespace NUMINAMATH_CALUDE_pentagon_properties_independent_l2096_209692

/-- A pentagon is a polygon with 5 sides --/
structure Pentagon where
  sides : Fin 5 → ℝ
  angles : Fin 5 → ℝ

/-- A pentagon is equilateral if all its sides have the same length --/
def Pentagon.isEquilateral (p : Pentagon) : Prop :=
  ∀ i j : Fin 5, p.sides i = p.sides j

/-- A pentagon is equiangular if all its angles are equal --/
def Pentagon.isEquiangular (p : Pentagon) : Prop :=
  ∀ i j : Fin 5, p.angles i = p.angles j

/-- The properties of equal angles and equal sides in a pentagon are independent --/
theorem pentagon_properties_independent :
  (∃ p : Pentagon, p.isEquiangular ∧ ¬p.isEquilateral) ∧
  (∃ q : Pentagon, q.isEquilateral ∧ ¬q.isEquiangular) := by
  sorry

end NUMINAMATH_CALUDE_pentagon_properties_independent_l2096_209692


namespace NUMINAMATH_CALUDE_cubic_function_not_monotonic_l2096_209612

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

def not_monotonic (f : ℝ → ℝ) : Prop :=
  ∃ x y z : ℝ, x < y ∧ y < z ∧ ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

theorem cubic_function_not_monotonic (a : ℝ) :
  not_monotonic (f a) → a ∈ Set.Iio (-Real.sqrt 3) ∪ Set.Ioi (Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_not_monotonic_l2096_209612


namespace NUMINAMATH_CALUDE_smallest_positive_angle_2014_l2096_209639

def same_terminal_side (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

theorem smallest_positive_angle_2014 :
  ∃! θ : ℝ, 0 ≤ θ ∧ θ < 360 ∧ same_terminal_side θ (-2014) ∧ θ = 146 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_2014_l2096_209639


namespace NUMINAMATH_CALUDE_spectators_count_l2096_209614

/-- The number of wristbands given to each spectator -/
def wristbands_per_person : ℕ := 2

/-- The total number of wristbands distributed -/
def total_wristbands : ℕ := 290

/-- The number of people who watched the game -/
def spectators : ℕ := total_wristbands / wristbands_per_person

theorem spectators_count : spectators = 145 := by
  sorry

end NUMINAMATH_CALUDE_spectators_count_l2096_209614


namespace NUMINAMATH_CALUDE_contribution_rate_of_random_error_l2096_209621

theorem contribution_rate_of_random_error 
  (sum_squared_residuals : ℝ) 
  (total_sum_squares : ℝ) 
  (h1 : sum_squared_residuals = 325) 
  (h2 : total_sum_squares = 923) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  abs (sum_squared_residuals / total_sum_squares - 0.352) < ε :=
sorry

end NUMINAMATH_CALUDE_contribution_rate_of_random_error_l2096_209621


namespace NUMINAMATH_CALUDE_donation_ratio_l2096_209650

def monthly_income : ℝ := 240
def groceries_expense : ℝ := 20
def remaining_amount : ℝ := 100

def donation : ℝ := monthly_income - groceries_expense - remaining_amount

theorem donation_ratio : donation / monthly_income = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_donation_ratio_l2096_209650


namespace NUMINAMATH_CALUDE_square_side_length_l2096_209631

/-- A square with perimeter 32 cm has sides of length 8 cm -/
theorem square_side_length (s : ℝ) (h₁ : s > 0) (h₂ : 4 * s = 32) : s = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2096_209631


namespace NUMINAMATH_CALUDE_smallest_integer_a_for_unique_solution_l2096_209603

-- Define the system of equations
def equation1 (x y a : ℝ) : Prop := y / (a - Real.sqrt x - 1) = 4
def equation2 (x y : ℝ) : Prop := y = (Real.sqrt x + 5) / (Real.sqrt x + 1)

-- Define the property of having a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! (x y : ℝ), equation1 x y a ∧ equation2 x y

-- State the theorem
theorem smallest_integer_a_for_unique_solution :
  (∀ a : ℤ, a < 3 → ¬(has_unique_solution (a : ℝ))) ∧
  has_unique_solution 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_a_for_unique_solution_l2096_209603


namespace NUMINAMATH_CALUDE_usb_drive_usage_percentage_l2096_209688

theorem usb_drive_usage_percentage (total_capacity : ℝ) (available_space : ℝ) 
  (h1 : total_capacity = 16) 
  (h2 : available_space = 8) : 
  (total_capacity - available_space) / total_capacity * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_usb_drive_usage_percentage_l2096_209688


namespace NUMINAMATH_CALUDE_intersecting_lines_length_l2096_209643

/-- Given a geometric configuration with two intersecting lines AC and BD, prove that AC = 3√19 -/
theorem intersecting_lines_length (O A B C D : ℝ × ℝ) (x : ℝ) : 
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist O A = 5 →
  dist O C = 11 →
  dist O D = 5 →
  dist O B = 6 →
  dist B D = 9 →
  x = dist A C →
  x = 3 * Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_intersecting_lines_length_l2096_209643


namespace NUMINAMATH_CALUDE_dubblefud_red_balls_l2096_209638

/-- The number of red balls in a Dubblefud game selection -/
def num_red_balls (r b g : ℕ) : Prop :=
  (2 ^ r) * (4 ^ b) * (5 ^ g) = 16000 ∧ b = g ∧ r = 0

/-- Theorem stating that the number of red balls is 0 given the conditions -/
theorem dubblefud_red_balls :
  ∃ (r b g : ℕ), num_red_balls r b g :=
sorry

end NUMINAMATH_CALUDE_dubblefud_red_balls_l2096_209638


namespace NUMINAMATH_CALUDE_coin_age_possibilities_l2096_209654

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def coin_digits : List ℕ := [3, 3, 3, 5, 1, 8]

def valid_first_digit (d : ℕ) : Prop := d ∈ coin_digits ∧ is_prime d

def count_valid_ages : ℕ := 40

theorem coin_age_possibilities :
  (∀ d ∈ coin_digits, d ≥ 0 ∧ d ≤ 9) →
  (∃ d ∈ coin_digits, valid_first_digit d) →
  count_valid_ages = 40 := by
  sorry

end NUMINAMATH_CALUDE_coin_age_possibilities_l2096_209654


namespace NUMINAMATH_CALUDE_spending_on_games_l2096_209696

theorem spending_on_games (total : ℚ) (movies burgers ice_cream music games : ℚ) : 
  total = 40 ∧ 
  movies = 1/4 ∧ 
  burgers = 1/8 ∧ 
  ice_cream = 1/5 ∧ 
  music = 1/4 ∧ 
  games = 3/20 ∧ 
  movies + burgers + ice_cream + music + games = 1 →
  total * games = 7 := by
sorry

end NUMINAMATH_CALUDE_spending_on_games_l2096_209696


namespace NUMINAMATH_CALUDE_hand_towels_per_set_l2096_209632

/-- The number of hand towels in a set -/
def h : ℕ := sorry

/-- The number of bath towels in a set -/
def bath_towels_per_set : ℕ := 6

/-- The smallest number of each type of towel sold -/
def min_towels_sold : ℕ := 102

theorem hand_towels_per_set :
  (∃ (n : ℕ), h * n = bath_towels_per_set * n ∧ h * n = min_towels_sold) →
  h = 17 := by sorry

end NUMINAMATH_CALUDE_hand_towels_per_set_l2096_209632


namespace NUMINAMATH_CALUDE_cubic_sum_greater_than_mixed_product_l2096_209607

theorem cubic_sum_greater_than_mixed_product (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_greater_than_mixed_product_l2096_209607


namespace NUMINAMATH_CALUDE_inequality_proof_l2096_209628

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + y * z + z * x = 6) : 
  1 / (2 * Real.sqrt 2 + x^2 * (y + z)) + 
  1 / (2 * Real.sqrt 2 + y^2 * (x + z)) + 
  1 / (2 * Real.sqrt 2 + z^2 * (x + y)) ≤ 
  1 / (x * y * z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2096_209628


namespace NUMINAMATH_CALUDE_common_number_in_list_l2096_209606

theorem common_number_in_list (l : List ℝ) : 
  l.length = 7 →
  (l.take 4).sum / 4 = 6 →
  (l.drop 3).sum / 4 = 9 →
  l.sum / 7 = 55 / 7 →
  ∃ x ∈ l.take 4 ∩ l.drop 3, x = 5 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_list_l2096_209606


namespace NUMINAMATH_CALUDE_find_other_number_l2096_209655

theorem find_other_number (x y : ℤ) : 
  ((x = 19 ∨ y = 19) ∧ 3 * x + 4 * y = 103) → 
  (x = 9 ∨ y = 9) := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l2096_209655


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l2096_209663

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  (x - 7)^2 = 3*(4*y + 2)^2 - 108

/-- A hyperbola is characterized by having coefficients of x^2 and y^2 with opposite signs
    when the equation is in standard form -/
def is_hyperbola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a*b < 0 ∧
    ∀ x y, eq x y ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0

/-- The given conic equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation := by
  sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l2096_209663


namespace NUMINAMATH_CALUDE_opposite_reciprocal_sum_l2096_209671

theorem opposite_reciprocal_sum (a b c : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c = 1/4)    -- the reciprocal of c is 4
  : 3*a + 3*b - 4*c = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_sum_l2096_209671


namespace NUMINAMATH_CALUDE_continuity_at_3_l2096_209691

def f (x : ℝ) := -2 * x^2 - 4

theorem continuity_at_3 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x - f 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_3_l2096_209691


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2096_209695

theorem inequality_equivalence (y : ℝ) :
  3/40 + |y - 17/80| < 1/8 ↔ 13/80 < y ∧ y < 21/80 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2096_209695


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2096_209626

theorem complex_equation_solution (i : ℂ) (z : ℂ) (h1 : i * i = -1) (h2 : i * z = 4 + 3 * i) : z = 3 - 4 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2096_209626


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2096_209659

theorem polynomial_division_remainder : ∀ (x : ℝ), ∃ (q : ℝ), 2*x^2 - 17*x + 47 = (x - 5) * q + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2096_209659


namespace NUMINAMATH_CALUDE_min_sum_floor_l2096_209687

theorem min_sum_floor (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ⌊(a + b) / c⌋ + ⌊(b + c) / d⌋ + ⌊(c + a) / b⌋ + ⌊(d + a) / c⌋ ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_floor_l2096_209687


namespace NUMINAMATH_CALUDE_g_of_2_eq_6_l2096_209601

def g (x : ℝ) : ℝ := x^3 - x

theorem g_of_2_eq_6 : g 2 = 6 := by sorry

end NUMINAMATH_CALUDE_g_of_2_eq_6_l2096_209601


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_origin_l2096_209665

/-- Given a line with equation y = 2x + 1, its symmetric line with respect to the origin
    has the equation y = 2x - 1. -/
theorem symmetric_line_wrt_origin :
  ∀ (x y : ℝ), y = 2*x + 1 → ∃ (x' y' : ℝ), y' = 2*x' - 1 ∧ x' = -x ∧ y' = -y :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_origin_l2096_209665


namespace NUMINAMATH_CALUDE_lunch_sales_calculation_l2096_209699

/-- Represents the number of hot dogs served by a restaurant -/
structure HotDogSales where
  total : ℕ
  dinner : ℕ
  lunch : ℕ

/-- Given the total number of hot dogs sold and the number sold during dinner,
    calculate the number of hot dogs sold during lunch -/
def lunchSales (sales : HotDogSales) : ℕ :=
  sales.total - sales.dinner

theorem lunch_sales_calculation (sales : HotDogSales) 
  (h1 : sales.total = 11)
  (h2 : sales.dinner = 2) :
  lunchSales sales = 9 := by
sorry

end NUMINAMATH_CALUDE_lunch_sales_calculation_l2096_209699


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extremum_l2096_209653

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem f_monotonicity_and_extremum :
  (∀ x, x > 0 → f x ≤ f (1 : ℝ)) ∧
  (∀ x y, 0 < x ∧ x < 1 ∧ 1 < y → f x > f 1 ∧ f y > f 1) ∧
  f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extremum_l2096_209653


namespace NUMINAMATH_CALUDE_power_of_two_equality_l2096_209666

theorem power_of_two_equality (x : ℤ) : (1 / 8 : ℚ) * (2 : ℚ)^40 = (2 : ℚ)^x → x = 37 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l2096_209666


namespace NUMINAMATH_CALUDE_expression_evaluation_l2096_209656

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 - z^2 + 2*x*y + 2*y*z = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2096_209656


namespace NUMINAMATH_CALUDE_log_sum_theorem_l2096_209693

theorem log_sum_theorem (a b : ℤ) : 
  a + 1 = b → 
  (a : ℝ) < Real.log 800 / Real.log 2 → 
  (Real.log 800 / Real.log 2 : ℝ) < b → 
  a + b = 19 := by
sorry

end NUMINAMATH_CALUDE_log_sum_theorem_l2096_209693


namespace NUMINAMATH_CALUDE_quadratic_vertex_l2096_209641

/-- The quadratic function f(x) = (x-3)^2 + 1 -/
def f (x : ℝ) : ℝ := (x - 3)^2 + 1

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (3, 1)

/-- Theorem: The vertex of the quadratic function f(x) = (x-3)^2 + 1 is at the point (3,1) -/
theorem quadratic_vertex : 
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l2096_209641


namespace NUMINAMATH_CALUDE_repetend_of_four_seventeenths_l2096_209684

/-- The decimal representation of 4/17 has a repeating block of 235294117647 -/
theorem repetend_of_four_seventeenths : 
  ∃ (n : ℕ), (4 : ℚ) / 17 = (n : ℚ) / 999999999999 ∧ n = 235294117647 := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_four_seventeenths_l2096_209684


namespace NUMINAMATH_CALUDE_streetlight_combinations_l2096_209635

/-- Represents the number of streetlights -/
def total_lights : ℕ := 12

/-- Represents the number of lights that can be turned off -/
def lights_off : ℕ := 3

/-- Represents the number of positions where lights can be turned off -/
def eligible_positions : ℕ := 8

/-- The number of ways to turn off lights under the given conditions -/
def ways_to_turn_off : ℕ := Nat.choose eligible_positions lights_off

theorem streetlight_combinations : ways_to_turn_off = 56 := by
  sorry

end NUMINAMATH_CALUDE_streetlight_combinations_l2096_209635


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2096_209613

/-- Given an arithmetic sequence with common difference 2 and where a₁, a₃, and a₄ form a geometric sequence, prove that a₂ = -6 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = 2) →  -- arithmetic sequence with common difference 2
  (∃ r, a 3 = r * a 1 ∧ a 4 = r * a 3) →  -- a₁, a₃, a₄ form a geometric sequence
  a 2 = -6 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2096_209613


namespace NUMINAMATH_CALUDE_geometric_series_product_l2096_209677

theorem geometric_series_product (y : ℝ) : 
  (∑' n : ℕ, (1/3:ℝ)^n) * (∑' n : ℕ, (-1/3:ℝ)^n) = ∑' n : ℕ, (1/y:ℝ)^n → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_product_l2096_209677


namespace NUMINAMATH_CALUDE_combined_sixth_grade_percent_l2096_209623

-- Define the schools
structure School where
  name : String
  total_students : ℕ
  sixth_grade_percent : ℚ

-- Define the given data
def pineview : School := ⟨"Pineview", 150, 15/100⟩
def oakridge : School := ⟨"Oakridge", 180, 17/100⟩
def maplewood : School := ⟨"Maplewood", 170, 15/100⟩

def schools : List School := [pineview, oakridge, maplewood]

-- Function to calculate the number of 6th graders in a school
def sixth_graders (s : School) : ℚ :=
  s.total_students * s.sixth_grade_percent

-- Function to calculate the total number of students
def total_students (schools : List School) : ℕ :=
  schools.foldl (fun acc s => acc + s.total_students) 0

-- Function to calculate the total number of 6th graders
def total_sixth_graders (schools : List School) : ℚ :=
  schools.foldl (fun acc s => acc + sixth_graders s) 0

-- Theorem statement
theorem combined_sixth_grade_percent :
  (total_sixth_graders schools) / (total_students schools : ℚ) = 1572 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_combined_sixth_grade_percent_l2096_209623


namespace NUMINAMATH_CALUDE_calculation_proof_l2096_209664

theorem calculation_proof : 99 * (5/8) - 0.625 * 68 + 6.25 * 0.1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2096_209664


namespace NUMINAMATH_CALUDE_larry_channels_l2096_209617

/-- Calculates the final number of channels Larry has after all changes --/
def final_channels (initial : ℕ) (removed : ℕ) (added : ℕ) (reduced : ℕ) (sports : ℕ) (supreme : ℕ) : ℕ :=
  initial - removed + added - reduced + sports + supreme

/-- Theorem stating that Larry's final number of channels is 147 --/
theorem larry_channels :
  final_channels 150 20 12 10 8 7 = 147 := by
  sorry

end NUMINAMATH_CALUDE_larry_channels_l2096_209617


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l2096_209670

theorem quadratic_equation_roots_condition (k : ℝ) : 
  (∃ p q : ℝ, 3 * p^2 + 6 * p + k = 0 ∧ 
              3 * q^2 + 6 * q + k = 0 ∧ 
              |p - q| = (1/2) * (p^2 + q^2)) ↔ 
  (k = -16 + 12 * Real.sqrt 2 ∨ k = -16 - 12 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l2096_209670


namespace NUMINAMATH_CALUDE_new_person_weight_l2096_209681

/-- The weight of the new person in a group, given:
  * The initial number of people in the group
  * The average weight increase when a new person replaces one person
  * The weight of the person being replaced
-/
def weight_of_new_person (initial_count : ℕ) (avg_increase : ℚ) (replaced_weight : ℚ) : ℚ :=
  replaced_weight + initial_count * avg_increase

theorem new_person_weight :
  weight_of_new_person 12 (37/10) 65 = 1094/10 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2096_209681


namespace NUMINAMATH_CALUDE_absolute_value_complex_power_l2096_209658

theorem absolute_value_complex_power : 
  Complex.abs ((5 : ℂ) + (Complex.I * Real.sqrt 11)) ^ 4 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_complex_power_l2096_209658


namespace NUMINAMATH_CALUDE_samuel_food_drinks_spending_l2096_209648

def total_budget : ℕ := 20
def ticket_cost : ℕ := 14
def kevin_drinks : ℕ := 2
def kevin_food : ℕ := 4

theorem samuel_food_drinks_spending :
  ∀ (samuel_food_drinks : ℕ),
    samuel_food_drinks = total_budget - ticket_cost →
    kevin_drinks + kevin_food + ticket_cost = total_budget →
    samuel_food_drinks = 6 := by
  sorry

end NUMINAMATH_CALUDE_samuel_food_drinks_spending_l2096_209648


namespace NUMINAMATH_CALUDE_xyz_sum_l2096_209657

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = y * z + x)
  (h2 : y * z + x = x * z + y)
  (h3 : x * y + z = 47) : 
  x + y + z = 48 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l2096_209657


namespace NUMINAMATH_CALUDE_binomial_max_and_expectation_l2096_209679

/-- The probability mass function for a binomial distribution with 20 trials and 2 successes -/
def f (p : ℝ) : ℝ := 190 * p^2 * (1 - p)^18

/-- The value of p that maximizes f(p) -/
def p₀ : ℝ := 0.1

/-- The number of items in a box -/
def box_size : ℕ := 200

/-- The number of items initially inspected -/
def initial_inspection : ℕ := 20

/-- The cost of inspecting one item -/
def inspection_cost : ℝ := 2

/-- The compensation fee for one defective item -/
def compensation_fee : ℝ := 25

/-- The expected number of defective items in the remaining items after initial inspection -/
def expected_defective : ℝ := 18

theorem binomial_max_and_expectation :
  (∀ p, p > 0 ∧ p < 1 → f p ≤ f p₀) ∧
  expected_defective = (box_size - initial_inspection : ℝ) * p₀ := by sorry

end NUMINAMATH_CALUDE_binomial_max_and_expectation_l2096_209679


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2096_209605

theorem algebraic_expression_value (x y : ℝ) 
  (h : Real.sqrt (x - 3) + y^2 - 4*y + 4 = 0) : 
  (x^2 - y^2) / (x*y) * (1 / (x^2 - 2*x*y + y^2)) / (x / (x^2*y - x*y^2)) - 1 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2096_209605


namespace NUMINAMATH_CALUDE_james_total_cost_l2096_209646

/-- Calculates the total amount James has to pay for adopting a puppy and a kitten -/
def total_cost (puppy_fee kitten_fee multiple_pet_discount friend1_contribution friend2_contribution sales_tax_rate pet_supplies : ℚ) : ℚ :=
  let total_adoption_fees := puppy_fee + kitten_fee
  let discounted_fees := total_adoption_fees * (1 - multiple_pet_discount)
  let friend_contributions := friend1_contribution * puppy_fee + friend2_contribution * kitten_fee
  let fees_after_contributions := discounted_fees - friend_contributions
  let sales_tax := fees_after_contributions * sales_tax_rate
  fees_after_contributions + sales_tax + pet_supplies

/-- The total cost James has to pay is $354.48 -/
theorem james_total_cost :
  total_cost 200 150 0.1 0.25 0.15 0.07 95 = 354.48 := by
  sorry

end NUMINAMATH_CALUDE_james_total_cost_l2096_209646


namespace NUMINAMATH_CALUDE_parallel_line_vector_l2096_209602

theorem parallel_line_vector (m : ℝ) : 
  (∀ x y : ℝ, m * x + 2 * y + 6 = 0 → (1 - m) * y = x) → 
  m = -1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_vector_l2096_209602


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2096_209604

theorem sum_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_prod : x * y = 16) (h_recip : 1 / x = 3 / y) : 
  x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2096_209604


namespace NUMINAMATH_CALUDE_library_crates_l2096_209629

theorem library_crates (novels : ℕ) (comics : ℕ) (documentaries : ℕ) (albums : ℕ) 
  (crate_capacity : ℕ) (h1 : novels = 145) (h2 : comics = 271) (h3 : documentaries = 419) 
  (h4 : albums = 209) (h5 : crate_capacity = 9) : 
  ((novels + comics + documentaries + albums) / crate_capacity : ℕ) = 116 := by
  sorry

end NUMINAMATH_CALUDE_library_crates_l2096_209629


namespace NUMINAMATH_CALUDE_probability_six_heads_ten_coins_l2096_209680

def num_coins : ℕ := 10
def num_heads : ℕ := 6

theorem probability_six_heads_ten_coins :
  (Nat.choose num_coins num_heads : ℚ) / (2 ^ num_coins) = 210 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_six_heads_ten_coins_l2096_209680


namespace NUMINAMATH_CALUDE_always_positive_l2096_209642

theorem always_positive (x : ℝ) : (-x)^2 + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_l2096_209642


namespace NUMINAMATH_CALUDE_max_volume_box_l2096_209689

/-- The volume function for the open-top box -/
def volume (a x : ℝ) : ℝ := x * (a - 2*x)^2

/-- The theorem stating the maximum volume and optimal cut length -/
theorem max_volume_box (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (a/2) ∧
    (∀ y ∈ Set.Ioo 0 (a/2), volume a x ≥ volume a y) ∧
    x = a/6 ∧
    volume a x = 2*a^3/27 :=
sorry

end NUMINAMATH_CALUDE_max_volume_box_l2096_209689


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l2096_209683

theorem min_sum_of_squares (x y : ℝ) (h : (x + 3) * (y - 3) = 0) :
  ∃ (m : ℝ), m = 18 ∧ ∀ (a b : ℝ), (a + 3) * (b - 3) = 0 → x^2 + y^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l2096_209683


namespace NUMINAMATH_CALUDE_happy_number_iff_multiple_of_eight_l2096_209610

/-- A number is "happy" if it is equal to the square difference of two consecutive odd numbers. -/
def is_happy_number (n : ℤ) : Prop :=
  ∃ k : ℤ, n = (2*k + 1)^2 - (2*k - 1)^2

/-- The theorem states that a number is a "happy number" if and only if it is a multiple of 8. -/
theorem happy_number_iff_multiple_of_eight (n : ℤ) :
  is_happy_number n ↔ ∃ m : ℤ, n = 8 * m :=
by sorry

end NUMINAMATH_CALUDE_happy_number_iff_multiple_of_eight_l2096_209610


namespace NUMINAMATH_CALUDE_card_ratio_proof_l2096_209673

theorem card_ratio_proof :
  let full_deck : ℕ := 52
  let num_partial_decks : ℕ := 3
  let num_full_decks : ℕ := 3
  let discarded_cards : ℕ := 34
  let remaining_cards : ℕ := 200
  let total_cards : ℕ := remaining_cards + discarded_cards
  let partial_deck_cards : ℕ := (total_cards - num_full_decks * full_deck) / num_partial_decks
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ partial_deck_cards * b = full_deck * a ∧ a = 1 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_card_ratio_proof_l2096_209673


namespace NUMINAMATH_CALUDE_five_digit_sum_contains_zero_l2096_209609

-- Define a five-digit number type
def FiveDigitNumber := { n : ℕ // n ≥ 10000 ∧ n < 100000 }

-- Define a function to check if a number contains 0
def containsZero (n : FiveDigitNumber) : Prop :=
  ∃ (a b c d : ℕ), n.val = 10000 * a + 1000 * b + 100 * c + 10 * d ∨
                    n.val = 10000 * a + 1000 * b + 100 * c + d ∨
                    n.val = 10000 * a + 1000 * b + 10 * c + d ∨
                    n.val = 10000 * a + 100 * b + 10 * c + d ∨
                    n.val = 1000 * a + 100 * b + 10 * c + d

-- Define a function to check if two numbers differ by switching two digits
def differByTwoDigits (n m : FiveDigitNumber) : Prop :=
  ∃ (a b c d e f : ℕ),
    (n.val = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
     m.val = 10000 * a + 1000 * b + 100 * f + 10 * d + e) ∨
    (n.val = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
     m.val = 10000 * a + 1000 * f + 100 * c + 10 * d + e) ∨
    (n.val = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
     m.val = 10000 * f + 1000 * b + 100 * c + 10 * d + e)

theorem five_digit_sum_contains_zero (n m : FiveDigitNumber)
  (h1 : differByTwoDigits n m)
  (h2 : n.val + m.val = 111111) :
  containsZero n ∨ containsZero m :=
sorry

end NUMINAMATH_CALUDE_five_digit_sum_contains_zero_l2096_209609


namespace NUMINAMATH_CALUDE_meeting_probability_approx_point_one_l2096_209640

/-- Object movement in a 2D plane -/
structure Object where
  x : ℤ
  y : ℤ

/-- Probability of movement in each direction -/
structure MoveProb where
  right : ℝ
  up : ℝ
  left : ℝ
  down : ℝ

/-- Calculate the probability of two objects meeting after n steps -/
def meetingProbability (a : Object) (c : Object) (aProb : MoveProb) (cProb : MoveProb) (n : ℕ) : ℝ :=
  sorry

/-- Theorem: The probability of A and C meeting after 7 steps is approximately 0.10 -/
theorem meeting_probability_approx_point_one :
  let a := Object.mk 0 0
  let c := Object.mk 6 8
  let aProb := MoveProb.mk 0.5 0.5 0 0
  let cProb := MoveProb.mk 0.1 0.1 0.4 0.4
  abs (meetingProbability a c aProb cProb 7 - 0.1) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_meeting_probability_approx_point_one_l2096_209640


namespace NUMINAMATH_CALUDE_max_area_inscribed_triangle_l2096_209644

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  is_convex : Convex ℝ (convexHull ℝ vertices)

/-- A triangle inscribed in a convex polygon -/
structure InscribedTriangle (M : ConvexPolygon) where
  points : Fin 3 → ℝ × ℝ
  inside : ∀ i, points i ∈ convexHull ℝ M.vertices

/-- The area of a triangle given by three points -/
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem max_area_inscribed_triangle (M : ConvexPolygon) :
  ∃ (t : InscribedTriangle M), 
    (∀ i, t.points i ∈ M.vertices) ∧
    (∀ (s : InscribedTriangle M), 
      triangleArea (t.points 0) (t.points 1) (t.points 2) ≥ 
      triangleArea (s.points 0) (s.points 1) (s.points 2)) :=
sorry

end NUMINAMATH_CALUDE_max_area_inscribed_triangle_l2096_209644


namespace NUMINAMATH_CALUDE_max_vector_sum_value_l2096_209660

/-- The maximum value of |OA + OB + OP| given the specified conditions -/
theorem max_vector_sum_value : ∃ (max : ℝ),
  max = 6 ∧
  ∀ (P : ℝ × ℝ),
  (P.1 - 3)^2 + P.2^2 = 1 →
  ‖(1, 0) + (0, 3) + P‖ ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_vector_sum_value_l2096_209660


namespace NUMINAMATH_CALUDE_third_grade_trees_l2096_209697

theorem third_grade_trees (second_grade_trees : ℕ) (third_grade_trees : ℕ) : 
  second_grade_trees = 15 →
  third_grade_trees < 3 * second_grade_trees →
  third_grade_trees = 42 →
  true :=
by sorry

end NUMINAMATH_CALUDE_third_grade_trees_l2096_209697


namespace NUMINAMATH_CALUDE_impossible_shot_l2096_209685

-- Define the elliptical billiard table
structure EllipticalTable where
  foci : Pointℝ × Pointℝ

-- Define the balls
structure Ball where
  position : Pointℝ

-- Define the properties of the problem
def is_on_edge (table : EllipticalTable) (ball : Ball) : Prop := sorry

def is_on_focal_segment (table : EllipticalTable) (ball : Ball) : Prop := sorry

def bounces_and_hits (table : EllipticalTable) (ball_A ball_B : Ball) : Prop := sorry

def crosses_focal_segment_before_bounce (table : EllipticalTable) (ball : Ball) : Prop := sorry

-- State the theorem
theorem impossible_shot (table : EllipticalTable) (ball_A ball_B : Ball) :
  is_on_edge table ball_A ∧
  is_on_focal_segment table ball_B ∧
  bounces_and_hits table ball_A ball_B ∧
  ¬crosses_focal_segment_before_bounce table ball_A →
  False := by sorry

end NUMINAMATH_CALUDE_impossible_shot_l2096_209685


namespace NUMINAMATH_CALUDE_rain_probability_tel_aviv_l2096_209675

theorem rain_probability_tel_aviv (p : ℝ) (n k : ℕ) (h_p : p = 1/2) (h_n : n = 6) (h_k : k = 4) :
  (n.choose k) * p^k * (1-p)^(n-k) = 15/64 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_tel_aviv_l2096_209675


namespace NUMINAMATH_CALUDE_probability_greater_than_400_probability_even_l2096_209600

def digits : List ℕ := [1, 5, 6]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (∀ d, d ∈ digits → (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d))

def valid_numbers : List ℕ := [156, 165, 516, 561, 615, 651]

theorem probability_greater_than_400 :
  (valid_numbers.filter (λ n => n > 400)).length / valid_numbers.length = 2 / 3 := by sorry

theorem probability_even :
  (valid_numbers.filter (λ n => n % 2 = 0)).length / valid_numbers.length = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_greater_than_400_probability_even_l2096_209600


namespace NUMINAMATH_CALUDE_center_of_symmetry_condition_l2096_209662

/-- A point A(a, b) is a center of symmetry for a function f if and only if
    for all x, f(a-x) + f(a+x) = 2b -/
theorem center_of_symmetry_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x y : ℝ, f x = y → f (2*a - x) = 2*b - y) ↔
  (∀ x : ℝ, f (a-x) + f (a+x) = 2*b) :=
sorry

end NUMINAMATH_CALUDE_center_of_symmetry_condition_l2096_209662


namespace NUMINAMATH_CALUDE_matrix_equation_holds_l2096_209619

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 2, 1; 2, 0, 2; 1, 2, 0]

theorem matrix_equation_holds :
  B^3 + (-8 : ℤ) • B^2 + (-12 : ℤ) • B + (-28 : ℤ) • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_holds_l2096_209619


namespace NUMINAMATH_CALUDE_prob_non_intersecting_chords_l2096_209647

/-- The probability of non-intersecting chords when pairing 2n points on a circle -/
theorem prob_non_intersecting_chords (n : ℕ) : 
  ∃ (P : ℚ), P = (2^n : ℚ) / (n + 1).factorial := by
  sorry

end NUMINAMATH_CALUDE_prob_non_intersecting_chords_l2096_209647


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2096_209645

theorem max_value_of_expression (w x y z : ℝ) : 
  w ≥ 0 → x ≥ 0 → y ≥ 0 → z ≥ 0 → w + x + y + z = 200 → 
  w * z + x * y + z * x ≤ 7500 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2096_209645


namespace NUMINAMATH_CALUDE_coefficient_of_x_fourth_l2096_209651

def expression (x : ℝ) : ℝ :=
  4 * (x^4 - 2*x^3 + x^2) + 2 * (3*x^4 + x^3 - 2*x^2 + x) - 6 * (2*x^2 - x^4 + 3*x^3)

theorem coefficient_of_x_fourth (x : ℝ) :
  ∃ (a b c d e : ℝ), expression x = 4*x^4 + a*x^3 + b*x^2 + c*x + d :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_fourth_l2096_209651


namespace NUMINAMATH_CALUDE_a_range_for_increasing_f_l2096_209669

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

-- State the theorem
theorem a_range_for_increasing_f :
  ∀ a : ℝ, 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ 
  (a ≥ 3/2 ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_a_range_for_increasing_f_l2096_209669


namespace NUMINAMATH_CALUDE_inequality_proof_l2096_209672

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  (((1 - a) ^ (1 / b) ≤ (1 - a) ^ b) ∧
   ((1 + a) ^ a ≤ (1 + b) ^ b) ∧
   ((1 - a) ^ b ≤ (1 - a) ^ (b / 2))) ∧
  ((1 - a) ^ a > (1 - b) ^ b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2096_209672


namespace NUMINAMATH_CALUDE_vector_equation_solution_l2096_209624

theorem vector_equation_solution (a b : ℝ × ℝ) (m n : ℝ) :
  a = (2, 1) →
  b = (1, -2) →
  m • a + n • b = (5, -5) →
  m - n = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l2096_209624


namespace NUMINAMATH_CALUDE_value_of_r_l2096_209690

theorem value_of_r (a b m p r : ℝ) : 
  (a^2 - m*a + 6 = 0) → 
  (b^2 - m*b + 6 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) → 
  r = 49/6 := by
sorry

end NUMINAMATH_CALUDE_value_of_r_l2096_209690


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l2096_209661

-- Define the slopes of the two lines
def slope1 : ℚ := 1/2
def slope2 (b : ℚ) : ℚ := -b/5

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_b_value :
  perpendicular slope1 (slope2 b) → b = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l2096_209661


namespace NUMINAMATH_CALUDE_bird_watching_problem_l2096_209682

theorem bird_watching_problem (total_watchers : Nat) (average_birds : Nat) 
  (first_watcher_birds : Nat) (second_watcher_birds : Nat) :
  total_watchers = 3 →
  average_birds = 9 →
  first_watcher_birds = 7 →
  second_watcher_birds = 11 →
  (total_watchers * average_birds - first_watcher_birds - second_watcher_birds) = 9 := by
  sorry

end NUMINAMATH_CALUDE_bird_watching_problem_l2096_209682


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l2096_209652

def is_valid_external_diagonals (d1 d2 d3 : ℝ) : Prop :=
  d1 > 0 ∧ d2 > 0 ∧ d3 > 0 ∧
  d1^2 + d2^2 > d3^2 ∧
  d1^2 + d3^2 > d2^2 ∧
  d2^2 + d3^2 > d1^2

theorem invalid_external_diagonals :
  ¬ (is_valid_external_diagonals 5 6 8) :=
by sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l2096_209652


namespace NUMINAMATH_CALUDE_four_inch_cube_value_l2096_209694

/-- Represents the properties of a gold cube -/
structure GoldCube where
  edge : ℝ  -- Edge length in inches
  weight : ℝ  -- Weight in pounds
  value : ℝ  -- Value in dollars

/-- The properties of gold cubes are directly proportional to their volume -/
axiom prop_proportional_to_volume (c1 c2 : GoldCube) :
  c2.weight = c1.weight * (c2.edge / c1.edge)^3 ∧
  c2.value = c1.value * (c2.edge / c1.edge)^3

/-- Given information about a one-inch gold cube -/
def one_inch_cube : GoldCube :=
  { edge := 1
  , weight := 0.5
  , value := 1000 }

/-- Theorem: A four-inch cube of gold is worth $64000 -/
theorem four_inch_cube_value :
  ∃ (c : GoldCube), c.edge = 4 ∧ c.value = 64000 :=
sorry

end NUMINAMATH_CALUDE_four_inch_cube_value_l2096_209694
