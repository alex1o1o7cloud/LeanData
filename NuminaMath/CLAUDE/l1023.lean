import Mathlib

namespace birth_rate_calculation_l1023_102394

/-- Represents the average birth rate in people per two seconds -/
def average_birth_rate : ℝ := 4

/-- Represents the death rate in people per two seconds -/
def death_rate : ℝ := 2

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- Represents the net population increase in one day -/
def net_increase_per_day : ℕ := 86400

theorem birth_rate_calculation :
  average_birth_rate = 4 :=
by
  sorry

#check birth_rate_calculation

end birth_rate_calculation_l1023_102394


namespace no_positive_integers_satisfy_divisibility_l1023_102358

theorem no_positive_integers_satisfy_divisibility : ¬ ∃ (a b c : ℕ+), (3 * (a * b + b * c + c * a)) ∣ (a^2 + b^2 + c^2) := by
  sorry

end no_positive_integers_satisfy_divisibility_l1023_102358


namespace solve_equation_l1023_102370

theorem solve_equation (m : ℝ) : (m - 6) ^ 4 = (1 / 16)⁻¹ ↔ m = 8 := by sorry

end solve_equation_l1023_102370


namespace overall_average_score_l1023_102342

theorem overall_average_score 
  (morning_avg : ℝ) 
  (evening_avg : ℝ) 
  (student_ratio : ℚ) 
  (h_morning_avg : morning_avg = 82) 
  (h_evening_avg : evening_avg = 75) 
  (h_student_ratio : student_ratio = 5 / 3) :
  let m := (student_ratio * evening_students : ℝ)
  let e := evening_students
  let total_students := m + e
  let total_score := morning_avg * m + evening_avg * e
  total_score / total_students = 79.375 :=
by
  sorry

#check overall_average_score

end overall_average_score_l1023_102342


namespace closest_integer_to_cube_root_l1023_102359

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (11^3 + 3^3 : ℝ)^(1/3) → 
  ∃ (n : ℤ), n = 11 ∧ ∀ (m : ℤ), |x - n| ≤ |x - m| :=
by sorry

end closest_integer_to_cube_root_l1023_102359


namespace multiply_monomials_l1023_102393

theorem multiply_monomials (a : ℝ) : 3 * a^3 * (-4 * a^2) = -12 * a^5 := by sorry

end multiply_monomials_l1023_102393


namespace expression_evaluation_l1023_102360

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  2 * (x^2 + 2*x*y) - 2*x^2 - x*y = 18 := by
  sorry

end expression_evaluation_l1023_102360


namespace bankers_discount_l1023_102353

/-- Banker's discount calculation -/
theorem bankers_discount (bankers_gain : ℝ) (rate : ℝ) (time : ℝ) : 
  bankers_gain = 270 → rate = 12 → time = 3 → 
  ∃ (bankers_discount : ℝ), abs (bankers_discount - 421.88) < 0.01 := by
  sorry

end bankers_discount_l1023_102353


namespace last_l_replaced_by_p_l1023_102382

-- Define the alphabet size
def alphabet_size : ℕ := 26

-- Define the position of 'l' in the alphabet (1-indexed)
def l_position : ℕ := 12

-- Define the occurrence of the last 'l' in the message
def l_occurrence : ℕ := 2

-- Define the shift function
def shift (n : ℕ) : ℕ := 2^n

-- Define the function to calculate the new position
def new_position (start : ℕ) (shift : ℕ) : ℕ :=
  (start + shift - 1) % alphabet_size + 1

-- Define the position of 'p' in the alphabet (1-indexed)
def p_position : ℕ := 16

-- The theorem to prove
theorem last_l_replaced_by_p :
  new_position l_position (shift l_occurrence) = p_position := by
  sorry

end last_l_replaced_by_p_l1023_102382


namespace fair_queue_l1023_102397

def queue_problem (initial_queue : ℕ) (net_change : ℕ) (interval : ℕ) (total_time : ℕ) : Prop :=
  let intervals := total_time / interval
  let final_queue := initial_queue + intervals * net_change
  final_queue = 24

theorem fair_queue : queue_problem 12 1 5 60 := by
  sorry

end fair_queue_l1023_102397


namespace direct_proportion_m_value_l1023_102336

/-- A function f: ℝ → ℝ is a direct proportion if there exists a constant k such that f(x) = k * x for all x ∈ ℝ -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The given function -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ -7 * x + 2 + m

theorem direct_proportion_m_value :
  (∃ m : ℝ, is_direct_proportion (f m)) → (∃ m : ℝ, m = -2 ∧ is_direct_proportion (f m)) :=
by sorry

end direct_proportion_m_value_l1023_102336


namespace sides_formula_l1023_102357

/-- The number of sides in the nth figure of a sequence starting with a hexagon,
    where each subsequent figure has 5 more sides than the previous one. -/
def sides (n : ℕ) : ℕ := 5 * n + 1

/-- Theorem stating that the number of sides in the nth figure is 5n + 1 -/
theorem sides_formula (n : ℕ) : sides n = 5 * n + 1 := by
  sorry

end sides_formula_l1023_102357


namespace quadratic_coefficient_l1023_102362

theorem quadratic_coefficient (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9) - 36 = 0) → b = 5 := by
  sorry

end quadratic_coefficient_l1023_102362


namespace least_sum_with_constraint_l1023_102395

theorem least_sum_with_constraint (x y z : ℕ+) : 
  (∀ a b c : ℕ+, x + y + z ≤ a + b + c) → 
  (x + y + z = 37) → 
  (5 * y = 6 * z) → 
  x = 21 := by
sorry

end least_sum_with_constraint_l1023_102395


namespace percentage_boys_playing_soccer_l1023_102351

theorem percentage_boys_playing_soccer 
  (total_students : ℕ) 
  (num_boys : ℕ) 
  (num_playing_soccer : ℕ) 
  (num_girls_not_playing : ℕ) 
  (h1 : total_students = 500) 
  (h2 : num_boys = 350) 
  (h3 : num_playing_soccer = 250) 
  (h4 : num_girls_not_playing = 115) :
  (num_boys - (total_students - num_boys - num_girls_not_playing)) / num_playing_soccer * 100 = 86 := by
sorry

end percentage_boys_playing_soccer_l1023_102351


namespace inequality_system_solution_l1023_102345

theorem inequality_system_solution (m n : ℝ) : 
  (∀ x, -1 < x ∧ x < 3 ↔ (x - 3*m < 0 ∧ n - 2*x < 0)) →
  (m + n)^2023 = -1 := by
  sorry

end inequality_system_solution_l1023_102345


namespace katie_juice_problem_l1023_102308

theorem katie_juice_problem (initial_juice : ℚ) (given_away : ℚ) (remaining : ℚ) :
  initial_juice = 5 →
  given_away = 18 / 7 →
  remaining = initial_juice - given_away →
  remaining = 17 / 7 := by
sorry

end katie_juice_problem_l1023_102308


namespace fraction_of_8000_l1023_102303

theorem fraction_of_8000 (x : ℝ) : x = 0.1 →
  x * 8000 - (1 / 20) * (1 / 100) * 8000 = 796 := by
  sorry

end fraction_of_8000_l1023_102303


namespace queenie_earnings_l1023_102341

/-- Calculates the total earnings for a part-time clerk with overtime -/
def total_earnings (daily_rate : ℕ) (overtime_rate : ℕ) (days_worked : ℕ) (overtime_hours : ℕ) : ℕ :=
  daily_rate * days_worked + overtime_rate * overtime_hours

/-- Proves that Queenie's total earnings are $770 -/
theorem queenie_earnings : total_earnings 150 5 5 4 = 770 := by
  sorry

end queenie_earnings_l1023_102341


namespace probability_all_colors_l1023_102323

/-- The probability of selecting 4 balls from a bag containing 2 red balls, 3 white balls, and 4 yellow balls, such that the selection includes balls of all three colors, is equal to 4/7. -/
theorem probability_all_colors (red : ℕ) (white : ℕ) (yellow : ℕ) (total_select : ℕ) :
  red = 2 →
  white = 3 →
  yellow = 4 →
  total_select = 4 →
  (Nat.choose (red + white + yellow) total_select : ℚ) ≠ 0 →
  (↑(Nat.choose red 2 * Nat.choose white 1 * Nat.choose yellow 1 +
     Nat.choose red 1 * Nat.choose white 2 * Nat.choose yellow 1 +
     Nat.choose red 1 * Nat.choose white 1 * Nat.choose yellow 2) /
   Nat.choose (red + white + yellow) total_select : ℚ) = 4 / 7 :=
by sorry

end probability_all_colors_l1023_102323


namespace min_effort_for_mop_l1023_102325

/-- Represents the effort and points for each exam --/
structure ExamEffort :=
  (effort : ℕ)
  (points : ℕ)

/-- Defines the problem of Alex making MOP --/
def MakeMOP (amc : ExamEffort) (aime : ExamEffort) (usamo : ExamEffort) : Prop :=
  let total_points := amc.points + aime.points
  let total_effort := amc.effort + aime.effort + usamo.effort
  total_points ≥ 200 ∧ usamo.points ≥ 21 ∧ total_effort = 320

/-- Theorem stating the minimum effort required for Alex to make MOP --/
theorem min_effort_for_mop :
  ∃ (amc aime usamo : ExamEffort),
    amc.effort = 3 * (amc.points / 6) ∧
    aime.effort = 7 * (aime.points / 10) ∧
    usamo.effort = 10 * usamo.points ∧
    MakeMOP amc aime usamo ∧
    ∀ (amc' aime' usamo' : ExamEffort),
      amc'.effort = 3 * (amc'.points / 6) →
      aime'.effort = 7 * (aime'.points / 10) →
      usamo'.effort = 10 * usamo'.points →
      MakeMOP amc' aime' usamo' →
      amc'.effort + aime'.effort + usamo'.effort ≥ 320 :=
by
  sorry

end min_effort_for_mop_l1023_102325


namespace min_value_sum_reciprocals_l1023_102361

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  1 / (a + 1) + 1 / (b + 3) ≥ 28 / 49 := by
  sorry

end min_value_sum_reciprocals_l1023_102361


namespace mistaken_division_correct_multiplication_l1023_102356

theorem mistaken_division_correct_multiplication : 
  ∀ n : ℕ, 
  (n / 96 = 5) → 
  (n % 96 = 17) → 
  (n * 69 = 34293) := by
sorry

end mistaken_division_correct_multiplication_l1023_102356


namespace min_coeff_x2_and_coeff_x7_l1023_102311

def f (m n : ℕ) (x : ℝ) : ℝ := (1 + x)^m + (1 + x)^n

theorem min_coeff_x2_and_coeff_x7 (m n : ℕ) :
  (∃ k : ℕ, k = m + n ∧ k = 19) →
  (∃ min_coeff_x2 : ℕ, 
    min_coeff_x2 = Nat.min (m * (m - 1) / 2 + n * (n - 1) / 2) 
                           ((m + 1) * m / 2 + (n - 1) * (n - 2) / 2) ∧
    min_coeff_x2 = 81) ∧
  (∃ coeff_x7 : ℕ, 
    (m = 10 ∧ n = 9 ∨ m = 9 ∧ n = 10) →
    coeff_x7 = Nat.choose 10 7 + Nat.choose 9 7 ∧
    coeff_x7 = 156) :=
by sorry

end min_coeff_x2_and_coeff_x7_l1023_102311


namespace complement_intersection_equals_set_l1023_102310

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2, 3}

-- Define set N
def N : Set Nat := {2, 3, 4}

-- Theorem statement
theorem complement_intersection_equals_set :
  (M ∩ N)ᶜ = {1, 4} :=
by
  sorry

end complement_intersection_equals_set_l1023_102310


namespace no_xy_term_implies_m_eq_neg_two_l1023_102374

/-- A polynomial in x and y with a parameter m -/
def polynomial (x y m : ℝ) : ℝ := 8 * x^2 + (m + 1) * x * y - 5 * y + x * y - 8

theorem no_xy_term_implies_m_eq_neg_two (m : ℝ) :
  (∀ x y : ℝ, polynomial x y m = 8 * x^2 - 5 * y - 8) →
  m = -2 := by
  sorry

end no_xy_term_implies_m_eq_neg_two_l1023_102374


namespace ellipse_intersection_property_l1023_102306

/-- An ellipse with semi-major axis 2 and semi-minor axis √3 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + (p.2^2 / 3) = 1}

/-- Check if two points are symmetric about the x-axis -/
def symmetric_about_x (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- The intersection point of two lines -/
def intersection (p₁ p₂ q₁ q₂ : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: For the given ellipse, if points satisfy the specified conditions,
    then the x-coordinates of A and B multiply to give 4 -/
theorem ellipse_intersection_property
  (d e : ℝ × ℝ)
  (h_d : d ∈ Ellipse)
  (h_e : e ∈ Ellipse)
  (h_sym : symmetric_about_x d e)
  (x₁ x₂ : ℝ)
  (h_not_tangent : ∀ y, (x₁, y) ≠ d)
  (c : ℝ × ℝ)
  (h_c_intersection : c = intersection d (x₁, 0) e (x₂, 0))
  (h_c_on_ellipse : c ∈ Ellipse) :
  x₁ * x₂ = 4 :=
sorry

end ellipse_intersection_property_l1023_102306


namespace interval_condition_l1023_102309

theorem interval_condition (x : ℝ) : 
  (2 < 3*x ∧ 3*x < 3 ∧ 2 < 4*x ∧ 4*x < 3) ↔ (2/3 < x ∧ x < 3/4) := by
  sorry

end interval_condition_l1023_102309


namespace bicycle_cost_is_150_l1023_102368

/-- The cost of the bicycle Patrick wants to buy. -/
def bicycle_cost : ℕ := 150

/-- The amount Patrick saved, which is half the price of the bicycle. -/
def patricks_savings : ℕ := bicycle_cost / 2

/-- The amount Patrick lent to his friend. -/
def lent_amount : ℕ := 50

/-- The amount Patrick has left after lending money to his friend. -/
def remaining_amount : ℕ := 25

/-- Theorem stating that the bicycle cost is 150, given the conditions. -/
theorem bicycle_cost_is_150 :
  patricks_savings - lent_amount = remaining_amount →
  bicycle_cost = 150 := by
  sorry

end bicycle_cost_is_150_l1023_102368


namespace jenna_stamps_problem_l1023_102319

theorem jenna_stamps_problem :
  Nat.gcd 1260 1470 = 210 := by
  sorry

end jenna_stamps_problem_l1023_102319


namespace billy_sleep_theorem_l1023_102335

def night1_sleep : ℕ := 6

def night2_sleep : ℕ := night1_sleep + 2

def night3_sleep : ℕ := night2_sleep / 2

def night4_sleep : ℕ := night3_sleep * 3

def total_sleep : ℕ := night1_sleep + night2_sleep + night3_sleep + night4_sleep

theorem billy_sleep_theorem : total_sleep = 30 := by
  sorry

end billy_sleep_theorem_l1023_102335


namespace probability_for_given_box_l1023_102338

/-- Represents the contents of the box -/
structure Box where
  blue : Nat
  red : Nat
  green : Nat

/-- The probability of drawing all blue chips before both green chips -/
def probability_all_blue_before_both_green (box : Box) : Rat :=
  17/36

/-- Theorem stating the probability for the given box configuration -/
theorem probability_for_given_box :
  let box : Box := { blue := 4, red := 3, green := 2 }
  probability_all_blue_before_both_green box = 17/36 := by
  sorry

end probability_for_given_box_l1023_102338


namespace completing_square_equivalence_l1023_102339

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) := by
sorry

end completing_square_equivalence_l1023_102339


namespace relationship_exists_l1023_102305

/-- Represents the contingency table --/
structure ContingencyTable where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  n : ℕ

/-- Calculates K^2 value --/
def calculate_k_squared (ct : ContingencyTable) : ℚ :=
  (ct.n * (ct.a * ct.d - ct.b * ct.c)^2 : ℚ) / 
  ((ct.a + ct.b) * (ct.c + ct.d) * (ct.a + ct.c) * (ct.b + ct.d) : ℚ)

/-- Theorem stating the conditions and the result --/
theorem relationship_exists (a : ℕ) : 
  5 < a ∧ 
  a < 10 ∧
  let ct := ContingencyTable.mk a (15 - a) (20 - a) (30 + a) 65
  calculate_k_squared ct ≥ (6635 : ℚ) / 1000 →
  a = 9 := by
  sorry

#check relationship_exists

end relationship_exists_l1023_102305


namespace isosceles_right_triangle_area_l1023_102387

/-- An isosceles right triangle with perimeter 3p has area (153 - 108√2) / 2 * p^2 -/
theorem isosceles_right_triangle_area (p : ℝ) (h : p > 0) :
  let perimeter := 3 * p
  let leg := (9 * p - 6 * p * Real.sqrt 2)
  let area := (1 / 2) * leg ^ 2
  area = (153 - 108 * Real.sqrt 2) / 2 * p ^ 2 := by
  sorry

end isosceles_right_triangle_area_l1023_102387


namespace largest_prime_to_check_primality_l1023_102302

theorem largest_prime_to_check_primality (n : ℕ) : 
  1000 ≤ n → n ≤ 1100 → 
  (∀ p : ℕ, p.Prime → p ≤ 31 → n % p ≠ 0) → 
  (∀ p : ℕ, p.Prime → p < n → n % p ≠ 0) :=
sorry

end largest_prime_to_check_primality_l1023_102302


namespace largest_prime_divisor_test_l1023_102301

theorem largest_prime_divisor_test (n : ℕ) (h1 : 1100 ≤ n) (h2 : n ≤ 1150) :
  Prime n → ∀ p, Prime p → p > 31 → ¬(p ∣ n) := by
  sorry

end largest_prime_divisor_test_l1023_102301


namespace cube_preserves_order_l1023_102343

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end cube_preserves_order_l1023_102343


namespace fraction_is_positive_integer_l1023_102328

theorem fraction_is_positive_integer (p : ℕ+) :
  (↑p : ℚ) = 3 ↔ (∃ (k : ℕ+), ((4 * p + 35) : ℚ) / ((3 * p - 8) : ℚ) = ↑k) := by
  sorry

end fraction_is_positive_integer_l1023_102328


namespace quadratic_root_range_l1023_102329

theorem quadratic_root_range (k : ℝ) (α β : ℝ) : 
  (∃ x, 7 * x^2 - (k + 13) * x + k^2 - k - 2 = 0) →
  (∀ x, 7 * x^2 - (k + 13) * x + k^2 - k - 2 = 0 → x = α ∨ x = β) →
  0 < α → α < 1 → 1 < β → β < 2 →
  (3 < k ∧ k < 4) ∨ (-2 < k ∧ k < -1) :=
by sorry

end quadratic_root_range_l1023_102329


namespace joes_bath_shop_soap_sales_l1023_102337

theorem joes_bath_shop_soap_sales : ∃ n : ℕ, n > 0 ∧ n % 7 = 0 ∧ n % 23 = 0 ∧ ∀ m : ℕ, m > 0 → m % 7 = 0 → m % 23 = 0 → m ≥ n :=
by sorry

end joes_bath_shop_soap_sales_l1023_102337


namespace point_relationship_l1023_102379

/-- Given points A(-2,a), B(-1,b), C(3,c) on the graph of y = 4/x, prove that b < a < c -/
theorem point_relationship (a b c : ℝ) : 
  (a = 4 / (-2)) → (b = 4 / (-1)) → (c = 4 / 3) → b < a ∧ a < c := by
  sorry

end point_relationship_l1023_102379


namespace joker_probability_l1023_102375

/-- A deck of cards with Jokers -/
structure DeckWithJokers where
  total_cards : ℕ
  joker_cards : ℕ
  unique_cards : Prop
  shuffled : Prop

/-- The probability of drawing a specific card type from a deck -/
def probability_of_draw (deck : DeckWithJokers) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / deck.total_cards

/-- Our specific deck configuration -/
def our_deck : DeckWithJokers := {
  total_cards := 54
  joker_cards := 2
  unique_cards := True
  shuffled := True
}

/-- Theorem: The probability of drawing a Joker from our deck is 1/27 -/
theorem joker_probability :
  probability_of_draw our_deck our_deck.joker_cards = 1 / 27 := by
  sorry

end joker_probability_l1023_102375


namespace expression_value_l1023_102322

theorem expression_value (x y z : ℝ) (hx : x = 1 + Real.sqrt 2) (hy : y = x + 1) (hz : z = x - 1) :
  y^2 * z^4 - 4 * y^3 * z^3 + 6 * y^2 * z^2 + 4 * y = -120 - 92 * Real.sqrt 2 := by
  sorry

end expression_value_l1023_102322


namespace arithmetic_sequence_sum_l1023_102354

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ + a₈ = 6,
    prove that 3a₂ + a₁₆ = 12 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 3 + a 8 = 6) : 
  3 * a 2 + a 16 = 12 := by
  sorry

end arithmetic_sequence_sum_l1023_102354


namespace equal_probability_sums_l1023_102364

def num_dice : ℕ := 8
def min_face_value : ℕ := 1
def max_face_value : ℕ := 6

def min_sum : ℕ := num_dice * min_face_value
def max_sum : ℕ := num_dice * max_face_value

def symmetric_sum (s : ℕ) : ℕ := 2 * ((min_sum + max_sum) / 2) - s

theorem equal_probability_sums :
  symmetric_sum 11 = 45 :=
sorry

end equal_probability_sums_l1023_102364


namespace tan_135_deg_l1023_102313

/-- Tangent of 135 degrees is -1 -/
theorem tan_135_deg : Real.tan (135 * π / 180) = -1 := by
  sorry

end tan_135_deg_l1023_102313


namespace jacks_lifetime_l1023_102324

theorem jacks_lifetime (L : ℝ) : 
  L > 0 → 
  (1/6 : ℝ) * L + (1/12 : ℝ) * L + (1/7 : ℝ) * L + 5 + (1/2 : ℝ) * L + 4 = L → 
  L = 84 := by
sorry

end jacks_lifetime_l1023_102324


namespace meal_cost_calculation_l1023_102391

/-- Proves that given a meal with a 12% sales tax, an 18% tip on the original price,
    and a total cost of $33.00, the original cost of the meal before tax and tip is $25.5. -/
theorem meal_cost_calculation (original_cost : ℝ) : 
  let tax_rate : ℝ := 0.12
  let tip_rate : ℝ := 0.18
  let total_cost : ℝ := 33.00
  (1 + tax_rate + tip_rate) * original_cost = total_cost → original_cost = 25.5 := by
sorry

end meal_cost_calculation_l1023_102391


namespace solution_set_for_a_equals_2_range_of_a_for_solutions_l1023_102376

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + a|

-- Theorem for part I
theorem solution_set_for_a_equals_2 :
  {x : ℝ | f x 2 > 6} = {x : ℝ | x < -3 ∨ x > 3} := by sorry

-- Theorem for part II
theorem range_of_a_for_solutions :
  {a : ℝ | ∃ x, f x a < a^2 - 1} = {a : ℝ | a < -1 - Real.sqrt 2 ∨ a > 1 + Real.sqrt 2} := by sorry

end solution_set_for_a_equals_2_range_of_a_for_solutions_l1023_102376


namespace not_always_achievable_all_plus_l1023_102326

/-- Represents a sign in a cell of the grid -/
inductive Sign
| Plus
| Minus

/-- Represents the grid -/
def Grid := Fin 8 → Fin 8 → Sign

/-- Represents a square subgrid -/
structure Square where
  size : Nat
  row : Fin 8
  col : Fin 8

/-- Checks if a square is valid (3x3 or 4x4) -/
def Square.isValid (s : Square) : Prop :=
  (s.size = 3 ∨ s.size = 4) ∧
  s.row + s.size ≤ 8 ∧
  s.col + s.size ≤ 8

/-- Applies an operation to the grid -/
def applyOperation (g : Grid) (s : Square) : Grid :=
  sorry

/-- Checks if a grid is filled with only Plus signs -/
def isAllPlus (g : Grid) : Prop :=
  ∀ i j, g i j = Sign.Plus

/-- Main theorem: It's not always possible to achieve all Plus signs -/
theorem not_always_achievable_all_plus :
  ∃ (initial : Grid), ¬∃ (operations : List Square),
    (∀ s ∈ operations, s.isValid) →
    isAllPlus (operations.foldl applyOperation initial) :=
  sorry

end not_always_achievable_all_plus_l1023_102326


namespace consecutive_nonprime_integers_l1023_102377

theorem consecutive_nonprime_integers : ∃ (a : ℕ),
  (25 < a) ∧
  (a + 4 < 50) ∧
  (¬ Nat.Prime a) ∧
  (¬ Nat.Prime (a + 1)) ∧
  (¬ Nat.Prime (a + 2)) ∧
  (¬ Nat.Prime (a + 3)) ∧
  (¬ Nat.Prime (a + 4)) ∧
  ((a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) % 10 = 0) ∧
  (a + 4 = 36) :=
by sorry

end consecutive_nonprime_integers_l1023_102377


namespace correct_marble_distribution_l1023_102333

/-- Represents the distribution of marbles among three boys -/
structure MarbleDistribution where
  x : ℕ
  first_boy : ℕ := 5 * x + 2
  second_boy : ℕ := 2 * x - 1
  third_boy : ℕ := x + 3

/-- The theorem stating the correct distribution of marbles -/
theorem correct_marble_distribution :
  ∃ (d : MarbleDistribution),
    d.first_boy + d.second_boy + d.third_boy = 60 ∧
    d.first_boy = 37 ∧
    d.second_boy = 13 ∧
    d.third_boy = 10 := by
  sorry

end correct_marble_distribution_l1023_102333


namespace quadratic_inequality_properties_l1023_102318

-- Define the quadratic function
def f (a c : ℝ) (x : ℝ) := a * x^2 + 2 * x + c

-- Define the solution set
def solution_set (a c : ℝ) := {x : ℝ | x < -1 ∨ x > 2}

-- State the theorem
theorem quadratic_inequality_properties
  (a c : ℝ)
  (h : ∀ x, f a c x < 0 ↔ x ∈ solution_set a c) :
  (a + c = 2) ∧
  (c^(1/a) = 1/2) ∧
  (∃! y, y ∈ {x : ℝ | x^2 - 2*a*x + c = 0}) :=
by sorry

end quadratic_inequality_properties_l1023_102318


namespace centroid_coincides_with_inscribed_sphere_center_l1023_102378

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents the centroids of faces opposite to vertices -/
structure FaceCentroids where
  SA : Point3D
  SB : Point3D
  SC : Point3D
  SD : Point3D

/-- Calculates the centroid of a system of homogeneous thin plates -/
def systemCentroid (t : Tetrahedron) (fc : FaceCentroids) : Point3D :=
  sorry

/-- Calculates the center of the inscribed sphere of a tetrahedron -/
def inscribedSphereCenter (t : Tetrahedron) : Point3D :=
  sorry

/-- Main theorem: The centroid of the system coincides with the center of the inscribed sphere -/
theorem centroid_coincides_with_inscribed_sphere_center 
  (t : Tetrahedron) (fc : FaceCentroids) :
  systemCentroid t fc = inscribedSphereCenter (Tetrahedron.mk fc.SA fc.SB fc.SC fc.SD) :=
by
  sorry

end centroid_coincides_with_inscribed_sphere_center_l1023_102378


namespace min_beta_delta_sum_l1023_102373

open Complex

/-- A complex-valued function g defined as g(z) = (5 + 3i)z^3 + βz + δ -/
def g (β δ : ℂ) (z : ℂ) : ℂ := (5 + 3*I)*z^3 + β*z + δ

/-- The theorem stating the minimum value of |β| + |δ| given the conditions -/
theorem min_beta_delta_sum :
  ∀ β δ : ℂ,
  (g β δ 1).im = 0 →
  (g β δ (-I)).im = 0 →
  (∃ β₀ δ₀ : ℂ, ∀ β δ : ℂ, (g β δ 1).im = 0 → (g β δ (-I)).im = 0 →
    Complex.abs β₀ + Complex.abs δ₀ ≤ Complex.abs β + Complex.abs δ) →
  ∃ β₀ δ₀ : ℂ, Complex.abs β₀ + Complex.abs δ₀ = Real.sqrt 73 :=
sorry

end min_beta_delta_sum_l1023_102373


namespace decimal_equals_scientific_l1023_102327

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_abs_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The given number in decimal form -/
def decimal_number : ℝ := -0.0000406

/-- The scientific notation representation of the number -/
def scientific_form : ScientificNotation := {
  coefficient := -4.06,
  exponent := -5,
  one_le_abs_coeff := by sorry
}

/-- Theorem stating that the decimal number is equal to its scientific notation representation -/
theorem decimal_equals_scientific : 
  decimal_number = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by sorry

end decimal_equals_scientific_l1023_102327


namespace picnic_attendance_l1023_102372

theorem picnic_attendance (total_students : ℕ) (picnic_attendees : ℕ) 
  (girls : ℕ) (boys : ℕ) : 
  total_students = 1500 →
  picnic_attendees = 975 →
  total_students = girls + boys →
  picnic_attendees = (3 * girls / 4) + (3 * boys / 5) →
  (3 * girls / 4 : ℕ) = 375 :=
by sorry

end picnic_attendance_l1023_102372


namespace gcd_of_specific_numbers_l1023_102392

theorem gcd_of_specific_numbers :
  let m : ℕ := 33333333
  let n : ℕ := 666666666
  Nat.gcd m n = 3 := by
sorry

end gcd_of_specific_numbers_l1023_102392


namespace drunk_driving_wait_time_l1023_102315

theorem drunk_driving_wait_time (p₀ r : ℝ) (h1 : p₀ = 89) (h2 : 61 = 89 * Real.exp (2 * r)) : 
  ⌈Real.log (20 / 89) / r⌉ = 8 := by
  sorry

end drunk_driving_wait_time_l1023_102315


namespace decimal_29_to_binary_l1023_102384

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_29_to_binary :
  decimal_to_binary 29 = [1, 1, 1, 0, 1] := by sorry

end decimal_29_to_binary_l1023_102384


namespace simple_interest_principal_l1023_102369

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : interest = 1000)
  (h2 : rate = 10)
  (h3 : time = 4)
  : interest = (2500 * rate * time) / 100 :=
by sorry

end simple_interest_principal_l1023_102369


namespace intersection_point_x_coordinate_l1023_102383

theorem intersection_point_x_coordinate (x y : ℝ) : 
  y = 4 * x - 19 ∧ 2 * x + y = 95 → x = 19 := by
  sorry

end intersection_point_x_coordinate_l1023_102383


namespace height_percentage_difference_l1023_102380

theorem height_percentage_difference (P Q : ℝ) (h : Q = P * (1 + 66.67 / 100)) :
  P = Q * (1 - 40 / 100) :=
sorry

end height_percentage_difference_l1023_102380


namespace inequality_implications_l1023_102352

theorem inequality_implications (a b : ℝ) :
  (b > 0 ∧ 0 > a → 1/a < 1/b) ∧
  (0 > a ∧ a > b → 1/a < 1/b) ∧
  (a > b ∧ b > 0 → 1/a < 1/b) ∧
  ¬(∀ a b : ℝ, a > 0 ∧ 0 > b → 1/a < 1/b) :=
by sorry

end inequality_implications_l1023_102352


namespace simon_red_stamps_count_l1023_102349

/-- The number of red stamps Simon has -/
def simon_red_stamps : ℕ := 34

/-- The number of white stamps Peter has -/
def peter_white_stamps : ℕ := 80

/-- The selling price of a red stamp in dollars -/
def red_stamp_price : ℚ := 1/2

/-- The selling price of a white stamp in dollars -/
def white_stamp_price : ℚ := 1/5

/-- The difference in the amount of money they make in dollars -/
def money_difference : ℚ := 1

theorem simon_red_stamps_count : 
  (simon_red_stamps : ℚ) * red_stamp_price - (peter_white_stamps : ℚ) * white_stamp_price = money_difference :=
by sorry

end simon_red_stamps_count_l1023_102349


namespace sin_cos_sixty_degrees_l1023_102320

theorem sin_cos_sixty_degrees :
  Real.sin (π / 3) = Real.sqrt 3 / 2 ∧ Real.cos (π / 3) = 1 / 2 := by
  sorry

end sin_cos_sixty_degrees_l1023_102320


namespace trash_can_prices_min_A_type_cans_l1023_102355

-- Define variables for trash can prices
variable (price_A : ℝ) (price_B : ℝ)

-- Define the cost equations
def cost_equation_1 (price_A price_B : ℝ) : Prop :=
  3 * price_A + 4 * price_B = 580

def cost_equation_2 (price_A price_B : ℝ) : Prop :=
  6 * price_A + 5 * price_B = 860

-- Define the total number of trash cans
def total_cans : ℕ := 200

-- Define the budget constraint
def budget : ℝ := 15000

-- Theorem for part 1
theorem trash_can_prices :
  cost_equation_1 price_A price_B ∧ cost_equation_2 price_A price_B →
  price_A = 60 ∧ price_B = 100 := by sorry

-- Theorem for part 2
theorem min_A_type_cans (num_A : ℕ) :
  num_A * price_A + (total_cans - num_A) * price_B ≤ budget →
  num_A ≥ 125 := by sorry

end trash_can_prices_min_A_type_cans_l1023_102355


namespace solution_set_f_gt_2_min_value_f_l1023_102346

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_gt_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -7 ∨ x > 5/3} := by sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (x : ℝ), f x = -9/2 ∧ ∀ (y : ℝ), f y ≥ -9/2 := by sorry

end solution_set_f_gt_2_min_value_f_l1023_102346


namespace quadratic_intersects_x_axis_l1023_102396

/-- A quadratic function f(x) = kx^2 - 7x - 7 intersects the x-axis if and only if
    k ≥ -7/4 and k ≠ 0 -/
theorem quadratic_intersects_x_axis (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 := by
  sorry

end quadratic_intersects_x_axis_l1023_102396


namespace power_product_equality_l1023_102331

theorem power_product_equality : 2000 * (2000 ^ 2000) = 2000 ^ 2001 := by
  sorry

end power_product_equality_l1023_102331


namespace arithmetic_sequence_sum_l1023_102371

theorem arithmetic_sequence_sum : ∃ (n : ℕ), 
  let a := 71  -- first term
  let d := 2   -- common difference
  let l := 99  -- last term
  n = (l - a) / d + 1 ∧ 
  3 * (n * (a + l) / 2) = 3825 := by
  sorry

end arithmetic_sequence_sum_l1023_102371


namespace derivative_f_at_negative_one_l1023_102317

def f (x : ℝ) : ℝ := x^6

theorem derivative_f_at_negative_one :
  deriv f (-1) = -6 := by sorry

end derivative_f_at_negative_one_l1023_102317


namespace smallest_value_l1023_102332

theorem smallest_value : 
  54 * Real.sqrt 3 < 144 ∧ 54 * Real.sqrt 3 < 108 * Real.sqrt 6 - 108 * Real.sqrt 2 := by
  sorry

end smallest_value_l1023_102332


namespace smallest_digit_correction_l1023_102307

def original_sum : ℕ := 356 + 781 + 492
def incorrect_sum : ℕ := 1529
def corrected_number : ℕ := 256

theorem smallest_digit_correction :
  (original_sum = incorrect_sum + 100) ∧
  (corrected_number + 781 + 492 = incorrect_sum) ∧
  (∀ n : ℕ, n < 356 → n > corrected_number → n + 781 + 492 ≠ incorrect_sum) := by
  sorry

end smallest_digit_correction_l1023_102307


namespace existence_of_n_l1023_102321

theorem existence_of_n (p : ℕ) (a k : ℕ+) (h_prime : Nat.Prime p) 
  (h_bound : p ^ a.val < k.val ∧ k.val < 2 * p ^ a.val) :
  ∃ n : ℕ, n < p ^ (2 * a.val) ∧ 
    (Nat.choose n k.val) % (p ^ a.val) = n % (p ^ a.val) ∧ 
    n % (p ^ a.val) = k.val % (p ^ a.val) := by
  sorry

end existence_of_n_l1023_102321


namespace route_down_length_for_given_conditions_l1023_102365

/-- Represents a hiking trip up and down a mountain -/
structure MountainHike where
  rate_up : ℝ
  time : ℝ
  rate_down_factor : ℝ

/-- Calculates the length of the route down the mountain -/
def route_down_length (hike : MountainHike) : ℝ :=
  hike.rate_up * hike.rate_down_factor * hike.time

/-- Theorem stating the length of the route down the mountain for the given conditions -/
theorem route_down_length_for_given_conditions :
  let hike : MountainHike := {
    rate_up := 3,
    time := 2,
    rate_down_factor := 1.5
  }
  route_down_length hike = 9 := by sorry

end route_down_length_for_given_conditions_l1023_102365


namespace mans_speed_with_stream_l1023_102386

/-- Given a man's rowing speed against the stream and his speed in still water,
    calculate his speed with the stream. -/
theorem mans_speed_with_stream
  (speed_against_stream : ℝ)
  (speed_still_water : ℝ)
  (h1 : speed_against_stream = 4)
  (h2 : speed_still_water = 5) :
  speed_still_water + (speed_still_water - speed_against_stream) = 6 :=
by sorry

end mans_speed_with_stream_l1023_102386


namespace theta_range_l1023_102348

theorem theta_range (θ : Real) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) →
  π / 12 < θ ∧ θ < 5 * π / 12 := by
  sorry

end theta_range_l1023_102348


namespace intersection_solution_l1023_102367

/-- Given two linear functions that intersect at x = 2, prove that the solution
    to their system of equations is (2, 2) -/
theorem intersection_solution (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ 2 * x - 2
  let g : ℝ → ℝ := fun x ↦ a * x + b
  (f 2 = g 2) →
  (∃! p : ℝ × ℝ, p.1 = 2 ∧ p.2 = 2 ∧ 2 * p.1 - p.2 = 2 ∧ p.2 = a * p.1 + b) :=
by sorry

end intersection_solution_l1023_102367


namespace derivative_f_l1023_102304

noncomputable def f (x : ℝ) : ℝ := (1/4) * Real.log ((x-1)/(x+1)) - (1/2) * Real.arctan x

theorem derivative_f (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  deriv f x = 1 / (x^4 - 1) := by sorry

end derivative_f_l1023_102304


namespace tangent_line_minimum_value_l1023_102385

theorem tangent_line_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : ∃ x₀ : ℝ, (2 * x₀ - a = Real.log (2 * x₀ + b)) ∧ 
    (∀ x : ℝ, 2 * x - a ≤ Real.log (2 * x + b))) :
  (4 / a + 1 / b) ≥ 9 := by
sorry

end tangent_line_minimum_value_l1023_102385


namespace smallest_integer_in_set_l1023_102350

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 4 < 3 * ((6 * n + 15) / 6)) → n ≥ -1 :=
by
  sorry

end smallest_integer_in_set_l1023_102350


namespace ellipse_equation_l1023_102344

/-- Given two ellipses C1 and C2, where C1 is defined by x²/4 + y² = 1,
    C2 has the same eccentricity as C1, and the minor axis of C2 is
    the same as the major axis of C1, prove that the equation of C2 is
    y²/16 + x²/4 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  let C1 := {(x, y) | x^2/4 + y^2 = 1}
  let e1 := Real.sqrt (1 - (2^2)/(4^2))  -- eccentricity of C1
  let C2 := {(x, y) | ∃ (a : ℝ), a > 2 ∧ y^2/a^2 + x^2/4 = 1 ∧ Real.sqrt (1 - (2^2)/(a^2)) = e1}
  ∀ (x y : ℝ), (x, y) ∈ C2 ↔ y^2/16 + x^2/4 = 1 :=
by sorry

end ellipse_equation_l1023_102344


namespace f_monotonic_iff_a_in_range_l1023_102388

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x - 7

-- Define the property of being monotonically increasing
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- State the theorem
theorem f_monotonic_iff_a_in_range (a : ℝ) :
  MonotonicallyIncreasing (f a) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end f_monotonic_iff_a_in_range_l1023_102388


namespace extreme_value_implies_a_equals_5_l1023_102316

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  (f' a (-3) = 0) → a = 5 := by
  sorry

end extreme_value_implies_a_equals_5_l1023_102316


namespace four_digit_to_two_digit_ratio_l1023_102300

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to its numerical value -/
def TwoDigitNumber.toNat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

/-- Converts a TwoDigitNumber to a four-digit number by repeating it -/
def TwoDigitNumber.toFourDigitNumber (n : TwoDigitNumber) : Nat :=
  1000 * n.tens + 100 * n.ones + 10 * n.tens + n.ones

/-- Theorem stating the ratio of the four-digit number to the original two-digit number is 101 -/
theorem four_digit_to_two_digit_ratio (n : TwoDigitNumber) :
    (n.toFourDigitNumber : ℚ) / (n.toNat : ℚ) = 101 := by
  sorry

end four_digit_to_two_digit_ratio_l1023_102300


namespace mutually_exclusive_events_l1023_102399

/-- Represents the outcome of selecting an item -/
inductive ItemSelection
  | Qualified
  | Defective

/-- Represents a batch of products -/
structure Batch where
  qualified : ℕ
  defective : ℕ
  qualified_exceeds_two : qualified > 2
  defective_exceeds_two : defective > 2

/-- Represents the selection of two items from a batch -/
def TwoItemSelection := Prod ItemSelection ItemSelection

/-- Event: At least one defective item -/
def AtLeastOneDefective (selection : TwoItemSelection) : Prop :=
  selection.1 = ItemSelection.Defective ∨ selection.2 = ItemSelection.Defective

/-- Event: All qualified items -/
def AllQualified (selection : TwoItemSelection) : Prop :=
  selection.1 = ItemSelection.Qualified ∧ selection.2 = ItemSelection.Qualified

/-- Theorem: AtLeastOneDefective and AllQualified are mutually exclusive -/
theorem mutually_exclusive_events (batch : Batch) (selection : TwoItemSelection) :
  ¬(AtLeastOneDefective selection ∧ AllQualified selection) :=
sorry

end mutually_exclusive_events_l1023_102399


namespace geometric_progression_first_term_l1023_102340

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricProgression (a : ℝ) (r : ℝ) := fun (n : ℕ) => a * r ^ (n - 1)

theorem geometric_progression_first_term
  (a : ℝ) (r : ℝ) (h1 : r ≠ 0)
  (h2 : GeometricProgression a r 2 = 5)
  (h3 : GeometricProgression a r 3 = 1) :
  a = 25 := by
  sorry

#check geometric_progression_first_term

end geometric_progression_first_term_l1023_102340


namespace fraction_sum_equality_l1023_102398

theorem fraction_sum_equality (m n p : ℝ) 
  (h : m / (140 - m) + n / (210 - n) + p / (180 - p) = 9) :
  10 / (140 - m) + 14 / (210 - n) + 12 / (180 - p) = 40 := by
  sorry

end fraction_sum_equality_l1023_102398


namespace function_property_l1023_102381

theorem function_property (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x y : ℝ, f (x + y) = f x + f y) 
  (h2 : f (-3) = a) : 
  f 12 = -4 * a := by
  sorry

end function_property_l1023_102381


namespace f_magnitude_relationship_l1023_102330

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing : ∀ x₁ x₂, x₁ ∈ Set.Ici (0 : ℝ) → x₂ ∈ Set.Ici (0 : ℝ) → x₁ ≠ x₂ → 
  (x₁ - x₂) * (f x₁ - f x₂) < 0

-- State the theorem to be proved
theorem f_magnitude_relationship : f 0 > f (-2) ∧ f (-2) > f 3 :=
sorry

end f_magnitude_relationship_l1023_102330


namespace students_per_group_l1023_102389

theorem students_per_group 
  (total_students : ℕ) 
  (num_teachers : ℕ) 
  (h1 : total_students = 256) 
  (h2 : num_teachers = 8) 
  (h3 : num_teachers > 0) :
  total_students / num_teachers = 32 := by
  sorry

end students_per_group_l1023_102389


namespace not_divisible_by_1955_l1023_102363

theorem not_divisible_by_1955 : ∀ n : ℕ, ¬(1955 ∣ (n^2 + n + 1)) := by
  sorry

end not_divisible_by_1955_l1023_102363


namespace max_weight_is_6250_l1023_102390

/-- The maximum number of crates the trailer can carry on a single trip -/
def max_crates : ℕ := 5

/-- The minimum weight of each crate in kilograms -/
def min_crate_weight : ℕ := 1250

/-- The maximum weight of crates on a single trip -/
def max_total_weight : ℕ := max_crates * min_crate_weight

/-- Theorem stating that the maximum weight of crates on a single trip is 6250 kg -/
theorem max_weight_is_6250 : max_total_weight = 6250 := by
  sorry

end max_weight_is_6250_l1023_102390


namespace courier_travel_times_l1023_102312

/-- Represents a courier with their travel times -/
structure Courier where
  meetingTime : ℝ
  remainingTime : ℝ

/-- Proves that given the conditions, the couriers' total travel times are 28 and 21 hours -/
theorem courier_travel_times (c1 c2 : Courier) 
  (h1 : c1.remainingTime = 16)
  (h2 : c2.remainingTime = 9)
  (h3 : c1.meetingTime = c2.meetingTime)
  (h4 : c1.meetingTime * (1 / c1.meetingTime + 1 / c2.meetingTime) = 1) :
  (c1.meetingTime + c1.remainingTime = 28) ∧ 
  (c2.meetingTime + c2.remainingTime = 21) := by
  sorry

#check courier_travel_times

end courier_travel_times_l1023_102312


namespace engineering_majors_consecutive_probability_l1023_102366

/-- The number of people sitting at the round table -/
def total_people : ℕ := 11

/-- The number of engineering majors -/
def engineering_majors : ℕ := 5

/-- The number of ways to arrange engineering majors consecutively after fixing one position -/
def consecutive_arrangements : ℕ := 7

/-- The number of ways to choose seats for engineering majors without restriction -/
def total_arrangements : ℕ := Nat.choose (total_people - 1) (engineering_majors - 1)

/-- The probability of engineering majors sitting consecutively -/
def probability : ℚ := consecutive_arrangements / total_arrangements

theorem engineering_majors_consecutive_probability :
  probability = 1 / 30 :=
sorry

end engineering_majors_consecutive_probability_l1023_102366


namespace renovation_constraint_l1023_102334

/-- Represents the constraint condition for hiring workers in a renovation project. -/
theorem renovation_constraint (x y : ℕ) : 
  (50 : ℝ) * x + (40 : ℝ) * y ≤ 2000 ↔ (5 : ℝ) * x + (4 : ℝ) * y ≤ 200 :=
by sorry

#check renovation_constraint

end renovation_constraint_l1023_102334


namespace motorcyclist_travel_l1023_102347

theorem motorcyclist_travel (total_distance : ℕ) (first_two_days : ℕ) (second_day_extra : ℕ)
  (h1 : total_distance = 980)
  (h2 : first_two_days = 725)
  (h3 : second_day_extra = 123) :
  ∃ (day1 day2 day3 : ℕ),
    day1 + day2 + day3 = total_distance ∧
    day1 + day2 = first_two_days ∧
    day2 = day3 + second_day_extra ∧
    day1 = 347 ∧
    day2 = 378 ∧
    day3 = 255 := by
  sorry

end motorcyclist_travel_l1023_102347


namespace hearty_blue_packages_l1023_102314

/-- The number of packages of red beads -/
def red_packages : ℕ := 5

/-- The number of beads in each package -/
def beads_per_package : ℕ := 40

/-- The total number of beads Hearty has -/
def total_beads : ℕ := 320

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := (total_beads - red_packages * beads_per_package) / beads_per_package

theorem hearty_blue_packages : blue_packages = 3 := by
  sorry

end hearty_blue_packages_l1023_102314
