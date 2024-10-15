import Mathlib

namespace NUMINAMATH_GPT_distinct_digit_sum_l785_78500

theorem distinct_digit_sum (a b c d : ℕ) (h1 : a + c = 10) (h2 : b + c = 9) (h3 : a + d = 1)
  (h4 : a ≠ b) (h5 : a ≠ c) (h6 : a ≠ d) (h7 : b ≠ c) (h8 : b ≠ d) (h9 : c ≠ d)
  (h10 : a < 10) (h11 : b < 10) (h12 : c < 10) (h13 : d < 10)
  (h14 : 0 ≤ a) (h15 : 0 ≤ b) (h16 : 0 ≤ c) (h17 : 0 ≤ d) :
  a + b + c + d = 18 :=
sorry

end NUMINAMATH_GPT_distinct_digit_sum_l785_78500


namespace NUMINAMATH_GPT_strawberry_cost_l785_78515

variables (S C : ℝ)

theorem strawberry_cost :
  (C = 6 * S) ∧ (5 * S + 5 * C = 77) → S = 2.2 :=
by
  sorry

end NUMINAMATH_GPT_strawberry_cost_l785_78515


namespace NUMINAMATH_GPT_angle_K_is_72_l785_78584

variables {J K L M : ℝ}

/-- Given that $JKLM$ is a trapezoid with parallel sides $\overline{JK}$ and $\overline{LM}$,
and given $\angle J = 3\angle M$, $\angle L = 2\angle K$, $\angle J + \angle K = 180^\circ$,
and $\angle L + \angle M = 180^\circ$, prove that $\angle K = 72^\circ$. -/
theorem angle_K_is_72 {J K L M : ℝ}
  (h1 : J = 3 * M)
  (h2 : L = 2 * K)
  (h3 : J + K = 180)
  (h4 : L + M = 180) :
  K = 72 :=
by
  sorry

end NUMINAMATH_GPT_angle_K_is_72_l785_78584


namespace NUMINAMATH_GPT_tanner_remaining_money_l785_78574
-- Import the entire Mathlib library

-- Define the conditions using constants
def s_Sep : ℕ := 17
def s_Oct : ℕ := 48
def s_Nov : ℕ := 25
def v_game : ℕ := 49

-- Define the total amount left and prove it equals 41
theorem tanner_remaining_money :
  (s_Sep + s_Oct + s_Nov - v_game) = 41 :=
by { sorry }

end NUMINAMATH_GPT_tanner_remaining_money_l785_78574


namespace NUMINAMATH_GPT_escalator_walk_rate_l785_78569

theorem escalator_walk_rate (v : ℝ) : (v + 15) * 10 = 200 → v = 5 := by
  sorry

end NUMINAMATH_GPT_escalator_walk_rate_l785_78569


namespace NUMINAMATH_GPT_no_zero_terms_in_arithmetic_progression_l785_78547

theorem no_zero_terms_in_arithmetic_progression (a d : ℤ) (h : ∃ (n : ℕ), 2 * a + (2 * n - 1) * d = ((3 * n - 1) * (2 * a + (3 * n - 2) * d)) / 2) :
  ∀ (m : ℕ), a + (m - 1) * d ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_no_zero_terms_in_arithmetic_progression_l785_78547


namespace NUMINAMATH_GPT_time_reduced_fraction_l785_78510

theorem time_reduced_fraction 
  (S : ℝ) (hs : S = 24.000000000000007) 
  (D : ℝ) : 
  1 - (D / (S + 12) / (D / S)) = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_time_reduced_fraction_l785_78510


namespace NUMINAMATH_GPT_everton_college_calculators_l785_78561

theorem everton_college_calculators (total_cost : ℤ) (num_scientific_calculators : ℤ) 
  (cost_per_scientific : ℤ) (cost_per_graphing : ℤ) (total_scientific_cost : ℤ) 
  (num_graphing_calculators : ℤ) (total_graphing_cost : ℤ) (total_calculators : ℤ) :
  total_cost = 1625 ∧
  num_scientific_calculators = 20 ∧
  cost_per_scientific = 10 ∧
  cost_per_graphing = 57 ∧
  total_scientific_cost = num_scientific_calculators * cost_per_scientific ∧
  total_graphing_cost = num_graphing_calculators * cost_per_graphing ∧
  total_cost = total_scientific_cost + total_graphing_cost ∧
  total_calculators = num_scientific_calculators + num_graphing_calculators → 
  total_calculators = 45 :=
by
  intros
  sorry

end NUMINAMATH_GPT_everton_college_calculators_l785_78561


namespace NUMINAMATH_GPT_krystian_total_books_borrowed_l785_78524

/-
Conditions:
1. Krystian starts on Monday by borrowing 40 books.
2. Each day from Tuesday to Thursday, he borrows 5% more books than he did the previous day.
3. On Friday, his number of borrowed books is 40% higher than on Thursday.
4. During weekends, Krystian borrows books for his friends, and he borrows 2 additional books for every 10 books borrowed during the weekdays.

Theorem: Given these conditions, Krystian borrows a total of 283 books from Monday to Sunday.
-/
theorem krystian_total_books_borrowed : 
  let mon := 40
  let tue := mon + (5 * mon / 100)
  let wed := tue + (5 * tue / 100)
  let thu := wed + (5 * wed / 100)
  let fri := thu + (40 * thu / 100)
  let weekday_total := mon + tue + wed + thu + fri
  let weekend := 2 * (weekday_total / 10)
  weekday_total + weekend = 283 := 
by
  sorry

end NUMINAMATH_GPT_krystian_total_books_borrowed_l785_78524


namespace NUMINAMATH_GPT_premium_rate_l785_78509

theorem premium_rate (P : ℝ) : (14400 / (100 + P)) * 5 = 600 → P = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_premium_rate_l785_78509


namespace NUMINAMATH_GPT_find_ab_l785_78575

noncomputable def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

theorem find_ab (a b : ℝ) :
  (f 1 a b = 10) ∧ ((3 * 1^2 - 2 * a * 1 - b = 0)) → (a, b) = (-4, 11) ∨ (a, b) = (3, -3) :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l785_78575


namespace NUMINAMATH_GPT_polycarp_error_l785_78571

def three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem polycarp_error (a b n : ℕ) (ha : three_digit a) (hb : three_digit b)
  (h : 10000 * a + b = n * a * b) : n = 73 :=
by
  sorry

end NUMINAMATH_GPT_polycarp_error_l785_78571


namespace NUMINAMATH_GPT_equal_real_roots_eq_one_l785_78533

theorem equal_real_roots_eq_one (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y * y = x) ∧ (∀ x y : ℝ, x^2 - 2 * x + m = 0 ↔ (x = y) → b^2 - 4 * a * c = 0) → m = 1 := 
sorry

end NUMINAMATH_GPT_equal_real_roots_eq_one_l785_78533


namespace NUMINAMATH_GPT_common_ratio_geometric_series_l785_78517

theorem common_ratio_geometric_series :
  let a := 2 / 3
  let b := 4 / 9
  let c := 8 / 27
  (b / a = 2 / 3) ∧ (c / b = 2 / 3) → 
  ∃ r : ℚ, r = 2 / 3 ∧ ∀ n : ℕ, (a * r^n) = (a * (2 / 3)^n) :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_series_l785_78517


namespace NUMINAMATH_GPT_weight_of_7_weights_l785_78587

theorem weight_of_7_weights :
  ∀ (w : ℝ), (16 * w + 0.6 = 17.88) → 7 * w = 7.56 :=
by
  intros w h
  sorry

end NUMINAMATH_GPT_weight_of_7_weights_l785_78587


namespace NUMINAMATH_GPT_alice_paid_percentage_of_srp_l785_78532

theorem alice_paid_percentage_of_srp
  (P : ℝ) -- Suggested Retail Price (SRP)
  (MP : ℝ := P * 0.60) -- Marked Price (MP) is 40% less than SRP
  (price_alice_paid : ℝ := MP * 0.60) -- Alice purchased the book for 40% off the marked price
  : (price_alice_paid / P) * 100 = 36 :=
by
  -- only the statement is required, so proof is omitted
  sorry

end NUMINAMATH_GPT_alice_paid_percentage_of_srp_l785_78532


namespace NUMINAMATH_GPT_domain_of_fraction_is_all_real_l785_78590

theorem domain_of_fraction_is_all_real (k : ℝ) :
  (∀ x : ℝ, -7 * x^2 + 3 * x + 4 * k ≠ 0) ↔ k < -9 / 112 :=
by sorry

end NUMINAMATH_GPT_domain_of_fraction_is_all_real_l785_78590


namespace NUMINAMATH_GPT_difference_of_squares_division_l785_78585

theorem difference_of_squares_division :
  let a := 121
  let b := 112
  (a^2 - b^2) / 3 = 699 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_division_l785_78585


namespace NUMINAMATH_GPT_grandchildren_ages_l785_78512

theorem grandchildren_ages (x : ℕ) (y : ℕ) :
  (x + y = 30) →
  (5 * (x * (x + 1) + (30 - x) * (31 - x)) = 2410) →
  (x = 16 ∧ y = 14) ∨ (x = 14 ∧ y = 16) :=
by
  sorry

end NUMINAMATH_GPT_grandchildren_ages_l785_78512


namespace NUMINAMATH_GPT_total_original_cost_l785_78529

theorem total_original_cost (discounted_price1 discounted_price2 discounted_price3 : ℕ) 
  (discount_rate1 discount_rate2 discount_rate3 : ℚ)
  (h1 : discounted_price1 = 4400)
  (h2 : discount_rate1 = 0.56)
  (h3 : discounted_price2 = 3900)
  (h4 : discount_rate2 = 0.35)
  (h5 : discounted_price3 = 2400)
  (h6 : discount_rate3 = 0.20) :
  (discounted_price1 / (1 - discount_rate1) + discounted_price2 / (1 - discount_rate2) 
    + discounted_price3 / (1 - discount_rate3) = 19000) :=
by
  sorry

end NUMINAMATH_GPT_total_original_cost_l785_78529


namespace NUMINAMATH_GPT_geometric_progression_common_ratio_l785_78563

theorem geometric_progression_common_ratio (a r : ℝ) (h_pos : a > 0)
  (h_eq : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)) :
  r = 1/2 :=
sorry

end NUMINAMATH_GPT_geometric_progression_common_ratio_l785_78563


namespace NUMINAMATH_GPT_find_number_eq_fifty_l785_78568

theorem find_number_eq_fifty (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 := by 
  sorry

end NUMINAMATH_GPT_find_number_eq_fifty_l785_78568


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l785_78556

variable (x : ℝ)

theorem quadratic_inequality_solution (hx : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l785_78556


namespace NUMINAMATH_GPT_gcd_3_pow_1007_minus_1_3_pow_1018_minus_1_l785_78516

theorem gcd_3_pow_1007_minus_1_3_pow_1018_minus_1 :
  Nat.gcd (3^1007 - 1) (3^1018 - 1) = 177146 :=
by
  -- Proof follows from the Euclidean algorithm and factoring, skipping the proof here.
  sorry

end NUMINAMATH_GPT_gcd_3_pow_1007_minus_1_3_pow_1018_minus_1_l785_78516


namespace NUMINAMATH_GPT_prove_triangle_inequality_l785_78536

def triangle_inequality (a b c a1 a2 b1 b2 c1 c2 : ℝ) : Prop := 
  a * a1 * a2 + b * b1 * b2 + c * c1 * c2 ≥ a * b * c

theorem prove_triangle_inequality 
  (a b c a1 a2 b1 b2 c1 c2 : ℝ)
  (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c)
  (h4: 0 ≤ a1) (h5: 0 ≤ a2) 
  (h6: 0 ≤ b1) (h7: 0 ≤ b2)
  (h8: 0 ≤ c1) (h9: 0 ≤ c2) : triangle_inequality a b c a1 a2 b1 b2 c1 c2 :=
sorry

end NUMINAMATH_GPT_prove_triangle_inequality_l785_78536


namespace NUMINAMATH_GPT_clearance_sale_total_earnings_l785_78513

-- Define the variables used in the problem
def total_jackets := 214
def price_before_noon := 31.95
def price_after_noon := 18.95
def jackets_sold_after_noon := 133

-- Calculate the total earnings
def total_earnings_from_clearance_sale : Prop :=
  (133 * 18.95 + (214 - 133) * 31.95) = 5107.30

-- State the theorem to be proven
theorem clearance_sale_total_earnings : total_earnings_from_clearance_sale :=
  by sorry

end NUMINAMATH_GPT_clearance_sale_total_earnings_l785_78513


namespace NUMINAMATH_GPT_tangent_line_circle_l785_78557

theorem tangent_line_circle : 
  ∃ (k : ℚ), (∀ x y : ℚ, ((x - 3) ^ 2 + (y - 4) ^ 2 = 25) 
               → (3 * x + 4 * y - 25 = 0)) :=
sorry

end NUMINAMATH_GPT_tangent_line_circle_l785_78557


namespace NUMINAMATH_GPT_quadratic_has_real_roots_l785_78580

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, x^2 - 4 * x - 2 * k + 8 = 0) ->
  k ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_l785_78580


namespace NUMINAMATH_GPT_find_a_l785_78538

-- Define the quadratic equation with the root condition
def quadratic_with_root_zero (a : ℝ) : Prop :=
  (a - 1) * 0^2 + 0 + a - 2 = 0

-- State the theorem to be proved
theorem find_a (a : ℝ) (h : quadratic_with_root_zero a) : a = 2 :=
by
  -- Statement placeholder, proof omitted
  sorry

end NUMINAMATH_GPT_find_a_l785_78538


namespace NUMINAMATH_GPT_total_cost_for_tickets_l785_78573

-- Definitions given in conditions
def num_students : ℕ := 20
def num_teachers : ℕ := 3
def ticket_cost : ℕ := 5

-- Proof Statement 
theorem total_cost_for_tickets : num_students + num_teachers * ticket_cost = 115 := by
  sorry

end NUMINAMATH_GPT_total_cost_for_tickets_l785_78573


namespace NUMINAMATH_GPT_correct_propositions_l785_78549

variables (a b : ℝ) (x : ℝ) (a_max : ℝ)

/-- Given propositions to analyze. -/
noncomputable def propositions :=
  ((a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3) ∧
  ((¬ ∀ x : ℝ, x^2 + x - 2 > 0) ↔ ∃ x : ℝ, x^2 + x - 2 ≤ 0) ∧
  (a_max = 2 ∧ ∀ x > 0, x + 1/x ≥ a_max))

/-- The main theorem stating which propositions are correct -/
theorem correct_propositions (h1 : a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3)
                            (h2 : (¬ ∀ x : ℝ, x^2 + x - 2 > 0) ↔ ∃ x : ℝ, x^2 + x - 2 ≤ 0)
                            (h3 : a_max = 2 ∧ ∀ x > 0, x + 1/x ≥ a_max) :
  propositions a b a_max :=
by
  sorry

end NUMINAMATH_GPT_correct_propositions_l785_78549


namespace NUMINAMATH_GPT_circumscribed_center_on_Ox_axis_l785_78502

-- Define the quadratic equation
noncomputable def quadratic_eq (p x : ℝ) : ℝ := 2^p * x^2 + 5 * p * x - 2^(p^2)

-- Define the conditions for the problem
def intersects_Ox (p : ℝ) : Prop := ∃ x1 x2 : ℝ, quadratic_eq p x1 = 0 ∧ quadratic_eq p x2 = 0 ∧ x1 ≠ x2

def intersects_Oy (p : ℝ) : Prop := quadratic_eq p 0 = -2^(p^2)

-- Define the problem statement
theorem circumscribed_center_on_Ox_axis :
  (∀ p : ℝ, intersects_Ox p ∧ intersects_Oy p → (p = 0 ∨ p = -1)) →
  (0 + (-1) = -1) :=
sorry

end NUMINAMATH_GPT_circumscribed_center_on_Ox_axis_l785_78502


namespace NUMINAMATH_GPT_integer_ratio_condition_l785_78583

variable (x y : ℝ)

theorem integer_ratio_condition 
  (h : 3 < (x - y) / (x + y) ∧ (x - y) / (x + y) < 6)
  (h_int : ∃ t : ℤ, x = t * y) :
  ∃ t : ℤ, t = -2 :=
by
  sorry

end NUMINAMATH_GPT_integer_ratio_condition_l785_78583


namespace NUMINAMATH_GPT_compound_interest_calculation_l785_78555

theorem compound_interest_calculation : 
  ∀ (x y T SI: ℝ), 
  x = 5000 → T = 2 → SI = 500 → 
  (y = SI * 100 / (x * T)) → 
  (5000 * (1 + (y / 100))^T - 5000 = 512.5) :=
by 
  intros x y T SI hx hT hSI hy
  sorry

end NUMINAMATH_GPT_compound_interest_calculation_l785_78555


namespace NUMINAMATH_GPT_greatest_possible_individual_award_l785_78519

variable (prize : ℕ)
variable (total_winners : ℕ)
variable (min_award : ℕ)
variable (fraction_prize : ℚ)
variable (fraction_winners : ℚ)

theorem greatest_possible_individual_award 
  (h1 : prize = 2500)
  (h2 : total_winners = 25)
  (h3 : min_award = 50)
  (h4 : fraction_prize = 3/5)
  (h5 : fraction_winners = 2/5) :
  ∃ award, award = 1300 := by
  sorry

end NUMINAMATH_GPT_greatest_possible_individual_award_l785_78519


namespace NUMINAMATH_GPT_face_value_of_each_ticket_without_tax_l785_78551

theorem face_value_of_each_ticket_without_tax (total_people : ℕ) (total_cost : ℝ) (sales_tax : ℝ) (face_value : ℝ)
  (h1 : total_people = 25)
  (h2 : total_cost = 945)
  (h3 : sales_tax = 0.05)
  (h4 : total_cost = (1 + sales_tax) * face_value * total_people) :
  face_value = 36 := by
  sorry

end NUMINAMATH_GPT_face_value_of_each_ticket_without_tax_l785_78551


namespace NUMINAMATH_GPT_floor_neg_seven_over_four_l785_78595

theorem floor_neg_seven_over_four : Int.floor (- 7 / 4 : ℝ) = -2 := 
by
  sorry

end NUMINAMATH_GPT_floor_neg_seven_over_four_l785_78595


namespace NUMINAMATH_GPT_paul_peaches_l785_78550

theorem paul_peaches (P : ℕ) (h1 : 26 - P = 22) : P = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_paul_peaches_l785_78550


namespace NUMINAMATH_GPT_a_3_and_a_4_sum_l785_78540

theorem a_3_and_a_4_sum (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℚ) :
  (1 - (1 / (2 * x))) ^ 6 = a_0 + a_1 * (1 / x) + a_2 * (1 / x) ^ 2 + a_3 * (1 / x) ^ 3 + 
  a_4 * (1 / x) ^ 4 + a_5 * (1 / x) ^ 5 + a_6 * (1 / x) ^ 6 →
  a_3 + a_4 = -25 / 16 :=
sorry

end NUMINAMATH_GPT_a_3_and_a_4_sum_l785_78540


namespace NUMINAMATH_GPT_determine_n_l785_78594

theorem determine_n (n : ℕ) (h1 : n > 2020) (h2 : ∃ m : ℤ, (n - 2020) = m^2 * (2120 - n)) : 
  n = 2070 ∨ n = 2100 ∨ n = 2110 := 
sorry

end NUMINAMATH_GPT_determine_n_l785_78594


namespace NUMINAMATH_GPT_arithmetic_sequence_num_terms_l785_78589

theorem arithmetic_sequence_num_terms (a d l : ℕ) (h1 : a = 15) (h2 : d = 4) (h3 : l = 159) :
  ∃ n : ℕ, l = a + (n-1) * d ∧ n = 37 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_num_terms_l785_78589


namespace NUMINAMATH_GPT_probability_same_color_set_l785_78558

theorem probability_same_color_set 
  (black_pairs blue_pairs : ℕ)
  (green_pairs : {g : Finset (ℕ × ℕ) // g.card = 3})
  (total_pairs := 15)
  (total_shoes := total_pairs * 2) :
  2 * black_pairs + 2 * blue_pairs + green_pairs.val.card * 2 = total_shoes →
  ∃ probability : ℚ, 
    probability = 89 / 435 :=
by
  intro h_total_shoes
  let black_shoes := black_pairs * 2
  let blue_shoes := blue_pairs * 2
  let green_shoes := green_pairs.val.card * 2
  
  have h_black_probability : ℚ := (black_shoes / total_shoes) * (black_pairs / (total_shoes - 1))
  have h_blue_probability : ℚ := (blue_shoes / total_shoes) * (blue_pairs / (total_shoes - 1))
  have h_green_probability : ℚ := (green_shoes / total_shoes) * (green_pairs.val.card / (total_shoes - 1))
  
  have h_total_probability : ℚ := h_black_probability + h_blue_probability + h_green_probability
  
  use h_total_probability
  sorry

end NUMINAMATH_GPT_probability_same_color_set_l785_78558


namespace NUMINAMATH_GPT_acute_angle_89_l785_78588

def is_acute_angle (angle : ℝ) : Prop := angle > 0 ∧ angle < 90

theorem acute_angle_89 :
  is_acute_angle 89 :=
by {
  -- proof details would go here, since only the statement is required
  sorry
}

end NUMINAMATH_GPT_acute_angle_89_l785_78588


namespace NUMINAMATH_GPT_no_such_natural_number_exists_l785_78581

theorem no_such_natural_number_exists :
  ¬ ∃ (n s : ℕ), n = 2014 * s + 2014 ∧ n % s = 2014 ∧ (n / s) = 2014 :=
by
  sorry

end NUMINAMATH_GPT_no_such_natural_number_exists_l785_78581


namespace NUMINAMATH_GPT_gcd_polynomials_l785_78503

theorem gcd_polynomials (b : ℕ) (hb : ∃ k : ℕ, b = 2 * 7771 * k) :
  Int.gcd (8 * b ^ 2 + 55 * b + 144) (4 * b + 19) = 8 :=
by sorry

end NUMINAMATH_GPT_gcd_polynomials_l785_78503


namespace NUMINAMATH_GPT_students_in_class_l785_78507

theorem students_in_class (total_spent: ℝ) (packs_per_student: ℝ) (sausages_per_student: ℝ) (cost_pack_noodles: ℝ) (cost_sausage: ℝ) (cost_per_student: ℝ) (num_students: ℝ):
  total_spent = 290 → 
  packs_per_student = 2 → 
  sausages_per_student = 1 → 
  cost_pack_noodles = 3.5 → 
  cost_sausage = 7.5 → 
  cost_per_student = packs_per_student * cost_pack_noodles + sausages_per_student * cost_sausage →
  total_spent = cost_per_student * num_students →
  num_students = 20 := 
by
  sorry

end NUMINAMATH_GPT_students_in_class_l785_78507


namespace NUMINAMATH_GPT_correctly_transformed_equation_l785_78530

theorem correctly_transformed_equation (s a b x y : ℝ) :
  (s = a * b → a = s / b ∧ b ≠ 0) ∧
  (1/2 * x = 8 → x = 16) ∧
  (-x - 1 = y - 1 → x = -y) ∧
  (a = b → a + 3 = b + 3) :=
by
  sorry

end NUMINAMATH_GPT_correctly_transformed_equation_l785_78530


namespace NUMINAMATH_GPT_B_completion_time_l785_78554

theorem B_completion_time (A_days : ℕ) (A_efficiency_multiple : ℝ) (B_days_correct : ℝ) :
  A_days = 15 →
  A_efficiency_multiple = 1.8 →
  B_days_correct = 4 + 1 / 6 →
  B_days_correct = 25 / 6 :=
sorry

end NUMINAMATH_GPT_B_completion_time_l785_78554


namespace NUMINAMATH_GPT_total_canvas_area_l785_78592

theorem total_canvas_area (rect_length rect_width tri1_base tri1_height tri2_base tri2_height : ℕ)
    (h1 : rect_length = 5) (h2 : rect_width = 8)
    (h3 : tri1_base = 3) (h4 : tri1_height = 4)
    (h5 : tri2_base = 4) (h6 : tri2_height = 6) :
    (rect_length * rect_width) + ((tri1_base * tri1_height) / 2) + ((tri2_base * tri2_height) / 2) = 58 := by
  sorry

end NUMINAMATH_GPT_total_canvas_area_l785_78592


namespace NUMINAMATH_GPT_eliot_account_balance_l785_78598

variable (A E F : ℝ)

theorem eliot_account_balance
  (h1 : A > E)
  (h2 : F > A)
  (h3 : A - E = (1 : ℝ) / 12 * (A + E))
  (h4 : F - A = (1 : ℝ) / 8 * (F + A))
  (h5 : 1.1 * A = 1.2 * E + 21)
  (h6 : 1.05 * F = 1.1 * A + 40) :
  E = 210 := 
sorry

end NUMINAMATH_GPT_eliot_account_balance_l785_78598


namespace NUMINAMATH_GPT_find_n_l785_78520

theorem find_n (n : ℕ) (h : 2 * 2^2 * 2^n = 2^10) : n = 7 :=
sorry

end NUMINAMATH_GPT_find_n_l785_78520


namespace NUMINAMATH_GPT_jim_saves_by_buying_gallon_l785_78504

-- Define the conditions as variables
def cost_per_gallon_costco : ℕ := 8
def ounces_per_gallon : ℕ := 128
def cost_per_16oz_bottle_store : ℕ := 3
def ounces_per_bottle : ℕ := 16

-- Define the theorem that needs to be proven
theorem jim_saves_by_buying_gallon (h1 : cost_per_gallon_costco = 8)
                                    (h2 : ounces_per_gallon = 128)
                                    (h3 : cost_per_16oz_bottle_store = 3)
                                    (h4 : ounces_per_bottle = 16) : 
  (8 * 3 - 8) = 16 :=
by sorry

end NUMINAMATH_GPT_jim_saves_by_buying_gallon_l785_78504


namespace NUMINAMATH_GPT_emily_total_beads_l785_78566

-- Let's define the given conditions
def necklaces : ℕ := 11
def beads_per_necklace : ℕ := 28

-- The statement to prove
theorem emily_total_beads : (necklaces * beads_per_necklace) = 308 := by
  sorry

end NUMINAMATH_GPT_emily_total_beads_l785_78566


namespace NUMINAMATH_GPT_find_w_l785_78544

theorem find_w (k : ℝ) (h1 : z * Real.sqrt w = k)
  (z_w3 : z = 6) (w3 : w = 3) :
  z = 3 / 2 → w = 48 := sorry

end NUMINAMATH_GPT_find_w_l785_78544


namespace NUMINAMATH_GPT_max_rectangles_3x5_in_17x22_l785_78523

theorem max_rectangles_3x5_in_17x22 : ∃ n : ℕ, n = 24 ∧ 
  (∀ (cut_3x5_pieces : ℤ), cut_3x5_pieces ≤ n) :=
by
  sorry

end NUMINAMATH_GPT_max_rectangles_3x5_in_17x22_l785_78523


namespace NUMINAMATH_GPT_max_value_abs_expression_l785_78593

noncomputable def circle_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

theorem max_value_abs_expression (x y : ℝ) (h : circle_eq x y) : 
  ∃ t : ℝ, |3 * x + 4 * y - 3| = t ∧ t ≤ 8 :=
sorry

end NUMINAMATH_GPT_max_value_abs_expression_l785_78593


namespace NUMINAMATH_GPT_x_value_l785_78565

theorem x_value :
  ∀ (x y : ℝ), x = y - 0.1 * y ∧ y = 125 + 0.1 * 125 → x = 123.75 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_x_value_l785_78565


namespace NUMINAMATH_GPT_product_sequence_l785_78582

theorem product_sequence : 
  let seq := [1/3, 9/1, 1/27, 81/1, 1/243, 729/1, 1/2187, 6561/1, 1/19683, 59049/1]
  ((seq[0] * seq[1]) * (seq[2] * seq[3]) * (seq[4] * seq[5]) * (seq[6] * seq[7]) * (seq[8] * seq[9])) = 243 :=
by
  sorry

end NUMINAMATH_GPT_product_sequence_l785_78582


namespace NUMINAMATH_GPT_inverse_function_correct_inequality_solution_l785_78535

noncomputable def f (x : ℝ) : ℝ := 1 - (2 / (2^x + 1))

noncomputable def f_inv (y : ℝ) : ℝ := Real.log (1 + y) / Real.log (1 - y)

theorem inverse_function_correct (x : ℝ) (hx : -1 < x ∧ x < 1) :
  f_inv (f x) = x :=
sorry

theorem inequality_solution :
  ∀ x, (1 / 2 < x ∧ x < 1) ↔ (f_inv x > Real.log (1 + x) + 1) :=
sorry

end NUMINAMATH_GPT_inverse_function_correct_inequality_solution_l785_78535


namespace NUMINAMATH_GPT_constraint_condition_2000_yuan_wage_l785_78518

-- Definitions based on the given conditions
def wage_carpenter : ℕ := 50
def wage_bricklayer : ℕ := 40
def total_wage : ℕ := 2000

-- Let x be the number of carpenters and y be the number of bricklayers
variable (x y : ℕ)

-- The proof problem statement
theorem constraint_condition_2000_yuan_wage (x y : ℕ) : 
  wage_carpenter * x + wage_bricklayer * y = total_wage → 5 * x + 4 * y = 200 :=
by
  intro h
  -- Simplification step will be placed here
  sorry

end NUMINAMATH_GPT_constraint_condition_2000_yuan_wage_l785_78518


namespace NUMINAMATH_GPT_determine_x_l785_78596

theorem determine_x (y : ℚ) (h : y = (36 + 249 / 999) / 100) :
  ∃ x : ℕ, y = x / 99900 ∧ x = 36189 :=
by
  sorry

end NUMINAMATH_GPT_determine_x_l785_78596


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l785_78559

theorem arithmetic_sequence_sum 
    (a : ℕ → ℤ)
    (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) -- Arithmetic sequence condition
    (h2 : a 5 = 3)
    (h3 : a 6 = -2) :
    (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l785_78559


namespace NUMINAMATH_GPT_cos_pi_div_4_add_alpha_l785_78560

variable (α : ℝ)

theorem cos_pi_div_4_add_alpha (h : Real.sin (Real.pi / 4 - α) = Real.sqrt 2 / 2) :
  Real.cos (Real.pi / 4 + α) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_div_4_add_alpha_l785_78560


namespace NUMINAMATH_GPT_total_shoes_l785_78542

theorem total_shoes (Brian_shoes : ℕ) (Edward_shoes : ℕ) (Jacob_shoes : ℕ)
  (hBrian : Brian_shoes = 22)
  (hEdward : Edward_shoes = 3 * Brian_shoes)
  (hJacob : Jacob_shoes = Edward_shoes / 2) :
  Brian_shoes + Edward_shoes + Jacob_shoes = 121 :=
by 
  sorry

end NUMINAMATH_GPT_total_shoes_l785_78542


namespace NUMINAMATH_GPT_proof_problem_l785_78576

theorem proof_problem (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x^2 + 2 * |y| = 2 * x * y) :
  (x > 0 → x + y > 3) ∧ (x < 0 → x + y < -3) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l785_78576


namespace NUMINAMATH_GPT_quadratic_root_c_l785_78579

theorem quadratic_root_c (c : ℝ) :
  (∀ x : ℝ, x^2 + 3 * x + c = (x + (3/2))^2 - 7/4) → c = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_c_l785_78579


namespace NUMINAMATH_GPT_average_tickets_sold_by_female_l785_78545

-- Define the conditions as Lean expressions.

def totalMembers (M : ℕ) : ℕ := M + 2 * M
def totalTickets (F : ℕ) (M : ℕ) : ℕ := 58 * M + F * 2 * M
def averageTicketsPerMember (F : ℕ) (M : ℕ) : ℕ := (totalTickets F M) / (totalMembers M)

theorem average_tickets_sold_by_female (F M : ℕ) 
  (h1 : 66 * (totalMembers M) = totalTickets F M) :
  F = 70 :=
by
  sorry

end NUMINAMATH_GPT_average_tickets_sold_by_female_l785_78545


namespace NUMINAMATH_GPT_sin_cos_product_l785_78578

theorem sin_cos_product (ϕ : ℝ) (h : Real.tan (ϕ + Real.pi / 4) = 5) : 
  1 / (Real.sin ϕ * Real.cos ϕ) = 13 / 6 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_product_l785_78578


namespace NUMINAMATH_GPT_zack_initial_marbles_l785_78525

theorem zack_initial_marbles :
  let a1 := 20
  let a2 := 30
  let a3 := 35
  let a4 := 25
  let a5 := 28
  let a6 := 40
  let r := 7
  let T := a1 + a2 + a3 + a4 + a5 + a6 + r
  T = 185 :=
by
  sorry

end NUMINAMATH_GPT_zack_initial_marbles_l785_78525


namespace NUMINAMATH_GPT_tangent_expression_equals_two_l785_78562

noncomputable def eval_tangent_expression : ℝ :=
  (1 + Real.tan (3 * Real.pi / 180)) * (1 + Real.tan (42 * Real.pi / 180))

theorem tangent_expression_equals_two :
  eval_tangent_expression = 2 :=
by sorry

end NUMINAMATH_GPT_tangent_expression_equals_two_l785_78562


namespace NUMINAMATH_GPT_minimum_distance_focus_to_circle_point_l785_78511

def focus_of_parabola : ℝ × ℝ := (1, 0)
def center_of_circle : ℝ × ℝ := (4, 4)
def radius_of_circle : ℝ := 4
def circle_equation (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 4)^2 = 16

theorem minimum_distance_focus_to_circle_point :
  ∃ P : ℝ × ℝ, circle_equation P.1 P.2 ∧ dist focus_of_parabola P = 5 :=
sorry

end NUMINAMATH_GPT_minimum_distance_focus_to_circle_point_l785_78511


namespace NUMINAMATH_GPT_total_cost_six_years_l785_78505

variable {fees : ℕ → ℝ}

-- Conditions
def fee_first_year : fees 1 = 80 := sorry

def fee_increase (n : ℕ) : fees (n + 1) = fees n + (10 + 2 * (n - 1)) := 
sorry

-- Proof problem: Prove that the total cost is 670
theorem total_cost_six_years : (fees 1 + fees 2 + fees 3 + fees 4 + fees 5 + fees 6) = 670 :=
by sorry

end NUMINAMATH_GPT_total_cost_six_years_l785_78505


namespace NUMINAMATH_GPT_find_w_squared_l785_78586

theorem find_w_squared (w : ℝ) (h : (2 * w + 19) ^ 2 = (4 * w + 9) * (3 * w + 13)) :
  w ^ 2 = ((6 + Real.sqrt 524) / 4) ^ 2 :=
sorry

end NUMINAMATH_GPT_find_w_squared_l785_78586


namespace NUMINAMATH_GPT_jason_initial_cards_l785_78522

-- Conditions
def cards_given_away : ℕ := 9
def cards_left : ℕ := 4

-- Theorem to prove
theorem jason_initial_cards : cards_given_away + cards_left = 13 :=
by
  sorry

end NUMINAMATH_GPT_jason_initial_cards_l785_78522


namespace NUMINAMATH_GPT_walking_distance_l785_78553

-- Define the pace in miles per hour.
def pace : ℝ := 2

-- Define the duration in hours.
def duration : ℝ := 8

-- Define the total distance walked.
def total_distance (pace : ℝ) (duration : ℝ) : ℝ := pace * duration

-- Define the theorem we need to prove.
theorem walking_distance :
  total_distance pace duration = 16 := by
  sorry

end NUMINAMATH_GPT_walking_distance_l785_78553


namespace NUMINAMATH_GPT_circle_radius_l785_78508

theorem circle_radius (r x y : ℝ) (h1 : x = π * r^2) (h2 : y = 2 * π * r) (h3 : x + y = 120 * π) : r = 10 :=
sorry

end NUMINAMATH_GPT_circle_radius_l785_78508


namespace NUMINAMATH_GPT_super_rare_snake_cost_multiple_l785_78591

noncomputable def price_of_regular_snake : ℕ := 250
noncomputable def total_money_obtained : ℕ := 2250
noncomputable def number_of_snakes : ℕ := 3
noncomputable def eggs_per_snake : ℕ := 2

theorem super_rare_snake_cost_multiple :
  (total_money_obtained - (number_of_snakes * eggs_per_snake - 1) * price_of_regular_snake) / price_of_regular_snake = 4 :=
by
  sorry

end NUMINAMATH_GPT_super_rare_snake_cost_multiple_l785_78591


namespace NUMINAMATH_GPT_perimeter_of_triangle_ABC_l785_78577

-- Define the focal points and their radius
def radius : ℝ := 2

-- Define the distances between centers of the tangent circles
def center_distance : ℝ := 2 * radius

-- Define the lengths of the sides of the triangle ABC based on the problem constraints
def AB : ℝ := 2 * radius + 2 * center_distance
def BC : ℝ := 2 * radius + center_distance
def CA : ℝ := 2 * radius + center_distance

-- Define the perimeter calculation
def perimeter : ℝ := AB + BC + CA

-- Theorem stating the actual perimeter of the triangle ABC
theorem perimeter_of_triangle_ABC : perimeter = 28 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_ABC_l785_78577


namespace NUMINAMATH_GPT_difference_of_cubes_l785_78572

theorem difference_of_cubes (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 - n^2 = 43) : m^3 - n^3 = 1387 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_cubes_l785_78572


namespace NUMINAMATH_GPT_find_m_l785_78521

theorem find_m (x1 x2 m : ℝ) (h1 : 2 * x1^2 - 3 * x1 + m = 0) (h2 : 2 * x2^2 - 3 * x2 + m = 0) (h3 : 8 * x1 - 2 * x2 = 7) :
  m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_l785_78521


namespace NUMINAMATH_GPT_minimize_acme_cost_l785_78539

theorem minimize_acme_cost (x : ℕ) : 75 + 12 * x < 16 * x → x = 19 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_minimize_acme_cost_l785_78539


namespace NUMINAMATH_GPT_triangle_equilateral_from_condition_l785_78537

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
a = b ∧ b = c

theorem triangle_equilateral_from_condition (a b c h_a h_b h_c : ℝ)
  (h : a + h_a = b + h_b ∧ b + h_b = c + h_c) :
  is_equilateral a b c :=
sorry

end NUMINAMATH_GPT_triangle_equilateral_from_condition_l785_78537


namespace NUMINAMATH_GPT_graphs_relative_position_and_intersection_l785_78564

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
noncomputable def g (x : ℝ) : ℝ := x^2 + 3 * x + 5

theorem graphs_relative_position_and_intersection :
  (1 > -1.5) ∧ ( ∃ y, f 0 = y ∧ g 0 = y ) ∧ f 0 = 5 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_graphs_relative_position_and_intersection_l785_78564


namespace NUMINAMATH_GPT_factor_polynomial_l785_78531

theorem factor_polynomial (y : ℝ) :
  y^8 - 4 * y^6 + 6 * y^4 - 4 * y^2 + 1 = ((y - 1) * (y + 1))^4 :=
sorry

end NUMINAMATH_GPT_factor_polynomial_l785_78531


namespace NUMINAMATH_GPT_max_k_value_l785_78501

noncomputable def max_k : ℝ := sorry 

theorem max_k_value :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), (x - 4)^2 + y^2 = 1 ∧ y = k * x - 2 ∧  (x - 4)^2 + y^2 ≤ 4) ↔ 
  k ≤ 4 / 3 := sorry

end NUMINAMATH_GPT_max_k_value_l785_78501


namespace NUMINAMATH_GPT_smallest_number_of_coins_l785_78541

theorem smallest_number_of_coins (p n d q h: ℕ) (total: ℕ) 
  (coin_value: ℕ → ℕ)
  (h_p: coin_value 1 = 1) 
  (h_n: coin_value 5 = 5) 
  (h_d: coin_value 10 = 10) 
  (h_q: coin_value 25 = 25) 
  (h_h: coin_value 50 = 50)
  (total_def: total = p * (coin_value 1) + n * (coin_value 5) +
                     d * (coin_value 10) + q * (coin_value 25) + 
                     h * (coin_value 50))
  (h_total: total = 100): 
  p + n + d + q + h = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_coins_l785_78541


namespace NUMINAMATH_GPT_percentage_seeds_germinated_l785_78534

theorem percentage_seeds_germinated :
  let S1 := 300
  let S2 := 200
  let S3 := 150
  let S4 := 250
  let S5 := 100
  let G1 := 0.20
  let G2 := 0.35
  let G3 := 0.45
  let G4 := 0.25
  let G5 := 0.60
  (G1 * S1 + G2 * S2 + G3 * S3 + G4 * S4 + G5 * S5) / (S1 + S2 + S3 + S4 + S5) * 100 = 32 := 
by
  sorry

end NUMINAMATH_GPT_percentage_seeds_germinated_l785_78534


namespace NUMINAMATH_GPT_don_eats_80_pizzas_l785_78570

variable (D Daria : ℝ)

-- Condition 1: Daria consumes 2.5 times the amount of pizza that Don does.
def condition1 : Prop := Daria = 2.5 * D

-- Condition 2: Together, they eat 280 pizzas.
def condition2 : Prop := D + Daria = 280

-- Conclusion: The number of pizzas Don eats is 80.
theorem don_eats_80_pizzas (h1 : condition1 D Daria) (h2 : condition2 D Daria) : D = 80 :=
by
  sorry

end NUMINAMATH_GPT_don_eats_80_pizzas_l785_78570


namespace NUMINAMATH_GPT_find_b_l785_78599

theorem find_b (a b : ℝ) (h1 : (1 : ℝ)^3 + a*(1)^2 + b*1 + a^2 = 10)
    (h2 : 3*(1 : ℝ)^2 + 2*a*(1) + b = 0) : b = -11 :=
sorry

end NUMINAMATH_GPT_find_b_l785_78599


namespace NUMINAMATH_GPT_nonagon_diagonals_count_l785_78506

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_nonagon_diagonals_count_l785_78506


namespace NUMINAMATH_GPT_find_q_l785_78546

def polynomial_q (x p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_q (p q r : ℝ) (h₀ : r = 3)
  (h₁ : (-p / 3) = -r)
  (h₂ : (-r) = 1 + p + q + r) :
  q = -16 :=
by
  -- h₀ implies r = 3
  -- h₁ becomes (-p / 3) = -3
  -- which results in p = 9
  -- h₂ becomes -3 = 1 + 9 + q + 3
  -- leading to q = -16
  sorry

end NUMINAMATH_GPT_find_q_l785_78546


namespace NUMINAMATH_GPT_length_of_each_piece_is_correct_l785_78514

noncomputable def rod_length : ℝ := 38.25
noncomputable def num_pieces : ℕ := 45
noncomputable def length_each_piece_cm : ℝ := 85

theorem length_of_each_piece_is_correct : (rod_length / num_pieces) * 100 = length_each_piece_cm :=
by
  sorry

end NUMINAMATH_GPT_length_of_each_piece_is_correct_l785_78514


namespace NUMINAMATH_GPT_segment_area_l785_78543

noncomputable def area_segment_above_triangle (a b c : ℝ) (triangle_area : ℝ) (y : ℝ) :=
  let ellipse_area := Real.pi * a * b
  ellipse_area - triangle_area

theorem segment_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 1) :
  let y := (4 * Real.sqrt 2) / 3
  let triangle_area := (1 / 2) * (2 * (b - y))
  area_segment_above_triangle a b c triangle_area y = 6 * Real.pi - 2 + (4 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_GPT_segment_area_l785_78543


namespace NUMINAMATH_GPT_time_spent_on_marketing_posts_l785_78548

-- Bryan's conditions
def hours_customer_outreach : ℕ := 4
def hours_advertisement : ℕ := hours_customer_outreach / 2
def total_hours_worked : ℕ := 8

-- Proof statement: Bryan spends 2 hours each day on marketing posts
theorem time_spent_on_marketing_posts : 
  total_hours_worked - (hours_customer_outreach + hours_advertisement) = 2 := by
  sorry

end NUMINAMATH_GPT_time_spent_on_marketing_posts_l785_78548


namespace NUMINAMATH_GPT_press_x_squared_three_times_to_exceed_10000_l785_78567

theorem press_x_squared_three_times_to_exceed_10000 :
  ∃ (n : ℕ), n = 3 ∧ (5^(2^n) > 10000) :=
by
  sorry

end NUMINAMATH_GPT_press_x_squared_three_times_to_exceed_10000_l785_78567


namespace NUMINAMATH_GPT_sum_of_fractions_l785_78552

theorem sum_of_fractions :
  (7:ℚ) / 12 + (11:ℚ) / 15 = 79 / 60 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l785_78552


namespace NUMINAMATH_GPT_min_value_abs_ab_l785_78528

theorem min_value_abs_ab (a b : ℝ) (hab : a ≠ 0 ∧ b ≠ 0) 
(h_perpendicular : - 1 / (a^2) * (a^2 + 1) / b = -1) :
|a * b| = 2 :=
sorry

end NUMINAMATH_GPT_min_value_abs_ab_l785_78528


namespace NUMINAMATH_GPT_benzene_molecular_weight_l785_78597

theorem benzene_molecular_weight (w: ℝ) (h: 4 * w = 312) : w = 78 :=
by
  sorry

end NUMINAMATH_GPT_benzene_molecular_weight_l785_78597


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l785_78527

theorem necessary_not_sufficient_condition (x : ℝ) :
  ((-6 ≤ x ∧ x ≤ 3) → (-5 ≤ x ∧ x ≤ 3)) ∧
  (¬ ((-5 ≤ x ∧ x ≤ 3) → (-6 ≤ x ∧ x ≤ 3))) :=
by
  -- Need proof steps here
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l785_78527


namespace NUMINAMATH_GPT_no_solution_inequalities_l785_78526

theorem no_solution_inequalities (m : ℝ) : 
  (∀ x : ℝ, (2 * x - 1 < 3) → (x > m) → false) ↔ (m ≥ 2) :=
by 
  sorry

end NUMINAMATH_GPT_no_solution_inequalities_l785_78526
