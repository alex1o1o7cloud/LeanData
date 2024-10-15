import Mathlib

namespace NUMINAMATH_GPT_fraction_equiv_ratio_equiv_percentage_equiv_l296_29685

-- Define the problem's components and conditions.
def frac_1 : ℚ := 3 / 5
def frac_2 (a b : ℚ) : Prop := 3 / 5 = a / b
def ratio_1 (a b : ℚ) : Prop := 10 / a = b / 100
def percentage_1 (a b : ℚ) : Prop := (a / b) * 100 = 60

-- Problem statement 1: Fraction equality
theorem fraction_equiv : frac_2 12 20 := 
by sorry

-- Problem statement 2: Ratio equality
theorem ratio_equiv : ratio_1 (50 / 3) 60 := 
by sorry

-- Problem statement 3: Percentage equality
theorem percentage_equiv : percentage_1 60 100 := 
by sorry

end NUMINAMATH_GPT_fraction_equiv_ratio_equiv_percentage_equiv_l296_29685


namespace NUMINAMATH_GPT_largest_angle_right_triangle_l296_29669

theorem largest_angle_right_triangle
  (a b c : ℝ)
  (h₁ : ∃ x : ℝ, x^2 + 4 * (c + 2) = (c + 4) * x)
  (h₂ : a + b = c + 4)
  (h₃ : a * b = 4 * (c + 2))
  : ∃ x : ℝ, x = 90 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_angle_right_triangle_l296_29669


namespace NUMINAMATH_GPT_unique_nonneg_sequence_l296_29659

theorem unique_nonneg_sequence (a : List ℝ) (h_sum : 0 < a.sum) :
  ∃ b : List ℝ, (∀ x ∈ b, 0 ≤ x) ∧ 
                (∃ f : List ℝ → List ℝ, (f a = b) ∧ (∀ x y z, f (x :: y :: z :: tl) = (x + y) :: (-y) :: (z + y) :: tl)) :=
sorry

end NUMINAMATH_GPT_unique_nonneg_sequence_l296_29659


namespace NUMINAMATH_GPT_dividend_calculation_l296_29635

theorem dividend_calculation (q d r x : ℝ) 
  (hq : q = -427.86) (hd : d = 52.7) (hr : r = -14.5)
  (hx : x = q * d + r) : 
  x = -22571.002 :=
by 
  sorry

end NUMINAMATH_GPT_dividend_calculation_l296_29635


namespace NUMINAMATH_GPT_find_initial_tomatoes_l296_29630

-- Define the initial number of tomatoes
def initial_tomatoes (T : ℕ) : Prop :=
  T + 77 - 172 = 80

-- Theorem statement to prove the initial number of tomatoes is 175
theorem find_initial_tomatoes : ∃ T : ℕ, initial_tomatoes T ∧ T = 175 :=
sorry

end NUMINAMATH_GPT_find_initial_tomatoes_l296_29630


namespace NUMINAMATH_GPT_part_a_part_b_l296_29649

-- Part (a)
theorem part_a
  (initial_deposit : ℝ)
  (initial_exchange_rate : ℝ)
  (annual_return_rate : ℝ)
  (final_exchange_rate : ℝ)
  (conversion_fee_rate : ℝ)
  (broker_commission_rate : ℝ) :
  initial_deposit = 12000 →
  initial_exchange_rate = 60 →
  annual_return_rate = 0.12 →
  final_exchange_rate = 80 →
  conversion_fee_rate = 0.04 →
  broker_commission_rate = 0.25 →
  let deposit_in_dollars := 12000 / 60
  let profit_in_dollars := deposit_in_dollars * 0.12
  let total_in_dollars := deposit_in_dollars + profit_in_dollars
  let broker_commission := profit_in_dollars * 0.25
  let amount_before_conversion := total_in_dollars - broker_commission
  let amount_in_rubles := amount_before_conversion * 80
  let conversion_fee := amount_in_rubles * 0.04
  let final_amount := amount_in_rubles - conversion_fee
  final_amount = 16742.4 := sorry

-- Part (b)
theorem part_b
  (initial_deposit : ℝ)
  (final_amount : ℝ) :
  initial_deposit = 12000 →
  final_amount = 16742.4 →
  let effective_return := (16742.4 / 12000) - 1
  effective_return * 100 = 39.52 := sorry

end NUMINAMATH_GPT_part_a_part_b_l296_29649


namespace NUMINAMATH_GPT_dane_daughters_initial_flowers_l296_29642

theorem dane_daughters_initial_flowers :
  (exists (x y : ℕ), x = y ∧ 5 * 4 = 20 ∧ x + y = 30) →
  (exists f : ℕ, f = 5 ∧ 10 = 30 - 20 + 10 ∧ x = f * 2) :=
by
  -- Lean proof needs to go here
  sorry

end NUMINAMATH_GPT_dane_daughters_initial_flowers_l296_29642


namespace NUMINAMATH_GPT_average_marks_l296_29667

/-- Given that the total marks in physics, chemistry, and mathematics is 110 more than the marks obtained in physics. -/
theorem average_marks (P C M : ℕ) (h : P + C + M = P + 110) : (C + M) / 2 = 55 :=
by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_average_marks_l296_29667


namespace NUMINAMATH_GPT_part_a_part_b_l296_29639

-- Part (a): Prove that if 2^n - 1 divides m^2 + 9 for positive integers m and n, then n must be a power of 2.
theorem part_a (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : (2^n - 1) ∣ (m^2 + 9)) : ∃ k : ℕ, n = 2^k := 
sorry

-- Part (b): Prove that if n is a power of 2, then there exists a positive integer m such that 2^n - 1 divides m^2 + 9.
theorem part_b (n : ℕ) (hn : ∃ k : ℕ, n = 2^k) : ∃ m : ℕ, 0 < m ∧ (2^n - 1) ∣ (m^2 + 9) := 
sorry

end NUMINAMATH_GPT_part_a_part_b_l296_29639


namespace NUMINAMATH_GPT_partI_partII_l296_29666

noncomputable def f (x m : ℝ) : ℝ := Real.log x - m * x
noncomputable def f' (x m : ℝ) : ℝ := (1 / x) - m

theorem partI (m : ℝ) : (∃ x : ℝ, x > 0 ∧ f x m = -1) → m = 1 := by
  sorry

theorem partII (x1 x2 : ℝ) (h1 : e ^ x1 ≤ x2) (h2 : f x1 1 = 0) (h3 : f x2 1 = 0) :
  ∃ y : ℝ, y = (x1 - x2) * f' (x1 + x2) 1 ∧ y = 2 / (1 + Real.exp 1) := by
  sorry

end NUMINAMATH_GPT_partI_partII_l296_29666


namespace NUMINAMATH_GPT_range_of_a3_plus_a9_l296_29647

variable {a_n : ℕ → ℝ}

-- Given condition: in a geometric sequence, a4 * a8 = 9
def geom_seq_condition (a_n : ℕ → ℝ) : Prop :=
  a_n 4 * a_n 8 = 9

-- Theorem statement
theorem range_of_a3_plus_a9 (a_n : ℕ → ℝ) (h : geom_seq_condition a_n) :
  ∃ x y, (x + y = a_n 3 + a_n 9) ∧ (x ≥ 0 ∧ y ≥ 0 ∧ x + y ≥ 6) ∨ (x ≤ 0 ∧ y ≤ 0 ∧ x + y ≤ -6) ∨ (x = 0 ∧ y = 0 ∧ a_n 3 + a_n 9 ∈ (Set.Ici 6 ∪ Set.Iic (-6))) :=
sorry

end NUMINAMATH_GPT_range_of_a3_plus_a9_l296_29647


namespace NUMINAMATH_GPT_melted_ice_cream_depth_l296_29628

theorem melted_ice_cream_depth
  (r_sphere : ℝ) (r_cylinder : ℝ) (V_sphere : ℝ) (V_cylinder : ℝ)
  (h : ℝ)
  (hr_sphere : r_sphere = 3)
  (hr_cylinder : r_cylinder = 10)
  (hV_sphere : V_sphere = 4 / 3 * Real.pi * r_sphere^3)
  (hV_cylinder : V_cylinder = Real.pi * r_cylinder^2 * h)
  (volume_conservation : V_sphere = V_cylinder) :
  h = 9 / 25 :=
by
  sorry

end NUMINAMATH_GPT_melted_ice_cream_depth_l296_29628


namespace NUMINAMATH_GPT_total_value_is_155_l296_29673

def coin_count := 20
def silver_coin_count := 10
def silver_coin_value_total := 30
def gold_coin_count := 5
def regular_coin_value := 1

def silver_coin_value := silver_coin_value_total / 4
def gold_coin_value := 2 * silver_coin_value

def total_silver_value := silver_coin_count * silver_coin_value
def total_gold_value := gold_coin_count * gold_coin_value
def regular_coin_count := coin_count - (silver_coin_count + gold_coin_count)
def total_regular_value := regular_coin_count * regular_coin_value

def total_collection_value := total_silver_value + total_gold_value + total_regular_value

theorem total_value_is_155 : total_collection_value = 155 := 
by
  sorry

end NUMINAMATH_GPT_total_value_is_155_l296_29673


namespace NUMINAMATH_GPT_least_multiple_greater_than_500_l296_29663

theorem least_multiple_greater_than_500 : ∃ n : ℕ, n > 0 ∧ 35 * n > 500 ∧ 35 * n = 525 :=
by
  sorry

end NUMINAMATH_GPT_least_multiple_greater_than_500_l296_29663


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l296_29627

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 - 5*x + 4 < 0) → (|x - 2| < 1) ∧ ¬( |x - 2| < 1 → x^2 - 5*x + 4 < 0) :=
by 
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l296_29627


namespace NUMINAMATH_GPT_total_customers_l296_29664

namespace math_proof

-- Definitions based on the problem's conditions.
def tables : ℕ := 9
def women_per_table : ℕ := 7
def men_per_table : ℕ := 3

-- The theorem stating the problem's question and correct answer.
theorem total_customers : tables * (women_per_table + men_per_table) = 90 := 
by
  -- This would be expanded into a proof, but we use sorry to bypass it here.
  sorry

end math_proof

end NUMINAMATH_GPT_total_customers_l296_29664


namespace NUMINAMATH_GPT_sin_identity_l296_29684

theorem sin_identity (α : ℝ) (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) :
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 := sorry

end NUMINAMATH_GPT_sin_identity_l296_29684


namespace NUMINAMATH_GPT_Jane_shopping_oranges_l296_29605

theorem Jane_shopping_oranges 
  (o a : ℕ)
  (h1 : a + o = 5)
  (h2 : 30 * a + 45 * o + 20 = n)
  (h3 : ∃ k : ℕ, n = 100 * k) : 
  o = 2 :=
by
  sorry

end NUMINAMATH_GPT_Jane_shopping_oranges_l296_29605


namespace NUMINAMATH_GPT_conjugate_in_fourth_quadrant_l296_29610

def complex_conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

-- Given complex number
def z : ℂ := ⟨5, 3⟩

-- Conjugate of z
def z_conjugate : ℂ := complex_conjugate z

-- Cartesian coordinates of the conjugate
def z_conjugate_coordinates : ℝ × ℝ := (z_conjugate.re, z_conjugate.im)

-- Definition of the Fourth Quadrant
def is_in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem conjugate_in_fourth_quadrant :
  is_in_fourth_quadrant z_conjugate_coordinates :=
by sorry

end NUMINAMATH_GPT_conjugate_in_fourth_quadrant_l296_29610


namespace NUMINAMATH_GPT_yellow_balls_count_l296_29655

theorem yellow_balls_count (R B G Y : ℕ) 
  (h1 : R = 2 * B) 
  (h2 : B = 2 * G) 
  (h3 : Y > 7) 
  (h4 : R + B + G + Y = 27) : 
  Y = 20 := by
  sorry

end NUMINAMATH_GPT_yellow_balls_count_l296_29655


namespace NUMINAMATH_GPT_minimum_four_sum_multiple_of_four_l296_29620

theorem minimum_four_sum_multiple_of_four (n : ℕ) (h : n = 7) (s : Fin n → ℤ) :
  ∃ (a b c d : Fin n), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (s a + s b + s c + s d) % 4 = 0 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_minimum_four_sum_multiple_of_four_l296_29620


namespace NUMINAMATH_GPT_right_triangle_third_side_l296_29603

theorem right_triangle_third_side (a b c : ℝ) (ha : a = 8) (hb : b = 6) (h_right_triangle : a^2 + b^2 = c^2) :
  c = 10 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_third_side_l296_29603


namespace NUMINAMATH_GPT_afternoon_shells_eq_l296_29643

def morning_shells : ℕ := 292
def total_shells : ℕ := 616

theorem afternoon_shells_eq :
  total_shells - morning_shells = 324 := by
  sorry

end NUMINAMATH_GPT_afternoon_shells_eq_l296_29643


namespace NUMINAMATH_GPT_base_eight_to_base_ten_l296_29680

theorem base_eight_to_base_ten {d1 d2 d3 : ℕ} (h1 : d1 = 1) (h2 : d2 = 5) (h3 : d3 = 7) :
  d3 * 8^0 + d2 * 8^1 + d1 * 8^2 = 111 := 
by
  sorry

end NUMINAMATH_GPT_base_eight_to_base_ten_l296_29680


namespace NUMINAMATH_GPT_tap_filling_time_l296_29641

theorem tap_filling_time (T : ℝ) (hT1 : T > 0) 
  (h_fill_with_one_tap : ∀ (t : ℝ), t = T → t > 0)
  (h_fill_with_second_tap : ∀ (s : ℝ), s = 60 → s > 0)
  (both_open_first_10_minutes : 10 * (1 / T + 1 / 60) + 20 * (1 / 60) = 1) :
    T = 20 := 
sorry

end NUMINAMATH_GPT_tap_filling_time_l296_29641


namespace NUMINAMATH_GPT_min_value_expr_l296_29640

theorem min_value_expr (a : ℝ) (ha : a > 0) : 
  ∃ (x : ℝ), x = (a-1)*(4*a-1)/a ∧ ∀ (y : ℝ), y = (a-1)*(4*a-1)/a → y ≥ -1 :=
by sorry

end NUMINAMATH_GPT_min_value_expr_l296_29640


namespace NUMINAMATH_GPT_sara_grew_4_onions_l296_29656

def onions_sally := 5
def onions_fred := 9
def total_onions := 18

def onions_sara : ℕ := total_onions - (onions_sally + onions_fred)

theorem sara_grew_4_onions : onions_sara = 4 := by
  -- proof here
  sorry

end NUMINAMATH_GPT_sara_grew_4_onions_l296_29656


namespace NUMINAMATH_GPT_boys_less_than_two_fifths_total_l296_29697

theorem boys_less_than_two_fifths_total
  (n b g n1 n2 b1 b2 : ℕ)
  (h_total: n = b + g)
  (h_first_trip: b1 < 2 * n1 / 5)
  (h_second_trip: b2 < 2 * n2 / 5)
  (h_participation: b ≤ b1 + b2)
  (h_total_participants: n ≤ n1 + n2) :
  b < 2 * n / 5 := 
sorry

end NUMINAMATH_GPT_boys_less_than_two_fifths_total_l296_29697


namespace NUMINAMATH_GPT_cos_pi_minus_alpha_l296_29633

theorem cos_pi_minus_alpha (α : ℝ) (hα : α > π ∧ α < 3 * π / 2) (h : Real.sin α = -5/13) :
  Real.cos (π - α) = 12 / 13 := 
by
  sorry

end NUMINAMATH_GPT_cos_pi_minus_alpha_l296_29633


namespace NUMINAMATH_GPT_max_value_m_l296_29671

theorem max_value_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 2 * x - 8 > 0) -> (x < m)) -> m = -2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_m_l296_29671


namespace NUMINAMATH_GPT_assignment_statement_increases_l296_29690

theorem assignment_statement_increases (N : ℕ) : (N + 1 = N + 1) :=
sorry

end NUMINAMATH_GPT_assignment_statement_increases_l296_29690


namespace NUMINAMATH_GPT_expression_value_l296_29609

theorem expression_value (x y : ℝ) (h : x + y = -1) : x^4 + 5 * x^3 * y + x^2 * y + 8 * x^2 * y^2 + x * y^2 + 5 * x * y^3 + y^4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l296_29609


namespace NUMINAMATH_GPT_angle_ABD_l296_29629

theorem angle_ABD (A B C D E F : Type)
  (quadrilateral : Prop)
  (angle_ABC : ℝ)
  (angle_BDE : ℝ)
  (angle_BDF : ℝ)
  (h1 : quadrilateral)
  (h2 : angle_ABC = 120)
  (h3 : angle_BDE = 30)
  (h4 : angle_BDF = 28) :
  (180 - angle_ABC = 60) :=
by
  sorry

end NUMINAMATH_GPT_angle_ABD_l296_29629


namespace NUMINAMATH_GPT_percentage_disliked_by_both_l296_29698

theorem percentage_disliked_by_both (total_comics liked_by_females liked_by_males disliked_by_both : ℕ) 
  (total_comics_eq : total_comics = 300)
  (liked_by_females_eq : liked_by_females = 30 * total_comics / 100)
  (liked_by_males_eq : liked_by_males = 120)
  (disliked_by_both_eq : disliked_by_both = total_comics - (liked_by_females + liked_by_males)) :
  (disliked_by_both * 100 / total_comics) = 30 := by
  sorry

end NUMINAMATH_GPT_percentage_disliked_by_both_l296_29698


namespace NUMINAMATH_GPT_max_a4b2c_l296_29686

-- Define the conditions and required statement
theorem max_a4b2c (a b c : ℝ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a + b + c = 1) :
    a^4 * b^2 * c ≤ 1024 / 117649 :=
sorry

end NUMINAMATH_GPT_max_a4b2c_l296_29686


namespace NUMINAMATH_GPT_total_sum_lent_l296_29607

noncomputable def interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem total_sum_lent 
  (x y : ℝ)
  (h1 : interest x (3 / 100) 5 = interest y (5 / 100) 3) 
  (h2 : y = 1332.5) : 
  x + y = 2665 :=
by
  -- We would continue the proof steps here.
  sorry

end NUMINAMATH_GPT_total_sum_lent_l296_29607


namespace NUMINAMATH_GPT_largest_visits_l296_29677

theorem largest_visits (stores : ℕ) (total_visits : ℕ) (unique_visitors : ℕ) 
  (visits_two_stores : ℕ) (remaining_visitors : ℕ) : 
  stores = 7 ∧ total_visits = 21 ∧ unique_visitors = 11 ∧ visits_two_stores = 7 ∧ remaining_visitors = (unique_visitors - visits_two_stores) →
  (remaining_visitors * 2 <= total_visits - visits_two_stores * 2) → (∀ v : ℕ, v * unique_visitors = total_visits) →
  (∃ v_max : ℕ, v_max = 4) :=
by
  sorry

end NUMINAMATH_GPT_largest_visits_l296_29677


namespace NUMINAMATH_GPT_equilateral_triangle_l296_29672

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 0 < α ∧ α < π)
  (h5 : 0 < β ∧ β < π)
  (h6 : 0 < γ ∧ γ < π)
  (h7 : α + β + γ = π)
  (h8 : a * (1 - 2 * Real.cos α) + b * (1 - 2 * Real.cos β) + c * (1 - 2 * Real.cos γ) = 0) :
  α = β ∧ β = γ ∧ γ = α :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_l296_29672


namespace NUMINAMATH_GPT_simplify_vectors_l296_29617

variable (α : Type*) [AddCommGroup α]

variables (CE AC DE AD : α)

theorem simplify_vectors : CE + AC - DE - AD = (0 : α) := 
by sorry

end NUMINAMATH_GPT_simplify_vectors_l296_29617


namespace NUMINAMATH_GPT_gcd_bc_minimum_l296_29660

theorem gcd_bc_minimum
  (a b c : ℕ)
  (h1 : Nat.gcd a b = 360)
  (h2 : Nat.gcd a c = 1170)
  (h3 : ∃ k1 : ℕ, b = 5 * k1)
  (h4 : ∃ k2 : ℕ, c = 13 * k2) : Nat.gcd b c = 90 :=
by
  sorry

end NUMINAMATH_GPT_gcd_bc_minimum_l296_29660


namespace NUMINAMATH_GPT_man_rowing_speed_l296_29637

noncomputable def rowing_speed_in_still_water : ℝ :=
  let distance := 0.1   -- kilometers
  let time := 20 / 3600 -- hours
  let current_speed := 3 -- km/hr
  let downstream_speed := distance / time
  downstream_speed - current_speed

theorem man_rowing_speed :
  rowing_speed_in_still_water = 15 :=
  by
    -- Proof comes here
    sorry

end NUMINAMATH_GPT_man_rowing_speed_l296_29637


namespace NUMINAMATH_GPT_series_converges_l296_29675

theorem series_converges :
  ∑' n, (2^n) / (3^(2^n) + 1) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_series_converges_l296_29675


namespace NUMINAMATH_GPT_probability_same_color_is_correct_l296_29611

-- Define the total number of each color marbles
def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def green_marbles : ℕ := 4

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

-- Define the probability calculation function
def probability_all_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))) +
  (white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))) +
  (blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))) +
  (green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)))

-- Define the theorem to prove the computed probability
theorem probability_same_color_is_correct :
  probability_all_same_color = 106 / 109725 := sorry

end NUMINAMATH_GPT_probability_same_color_is_correct_l296_29611


namespace NUMINAMATH_GPT_sqrt_64_eq_pm_8_l296_29602

theorem sqrt_64_eq_pm_8 : ∃x : ℤ, x^2 = 64 ∧ (x = 8 ∨ x = -8) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_64_eq_pm_8_l296_29602


namespace NUMINAMATH_GPT_yesterday_tomorrow_is_friday_l296_29696

-- Defining the days of the week
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to go to the next day
def next_day : Day → Day
| Sunday    => Monday
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday

-- Function to go to the previous day
def previous_day : Day → Day
| Sunday    => Saturday
| Monday    => Sunday
| Tuesday   => Monday
| Wednesday => Tuesday
| Thursday  => Wednesday
| Friday    => Thursday
| Saturday  => Friday

-- Proving the statement
theorem yesterday_tomorrow_is_friday (T : Day) (H : next_day (previous_day T) = Thursday) : previous_day (next_day (next_day T)) = Friday :=
by
  sorry

end NUMINAMATH_GPT_yesterday_tomorrow_is_friday_l296_29696


namespace NUMINAMATH_GPT_number_of_pairs_is_2_pow_14_l296_29668

noncomputable def number_of_pairs_satisfying_conditions : ℕ :=
  let fact5 := Nat.factorial 5
  let fact50 := Nat.factorial 50
  Nat.card {p : ℕ × ℕ | Nat.gcd p.1 p.2 = fact5 ∧ Nat.lcm p.1 p.2 = fact50}

theorem number_of_pairs_is_2_pow_14 :
  number_of_pairs_satisfying_conditions = 2^14 := by
  sorry

end NUMINAMATH_GPT_number_of_pairs_is_2_pow_14_l296_29668


namespace NUMINAMATH_GPT_problem_solution_l296_29676

noncomputable def proof_problem : Prop :=
∀ x y : ℝ, y = (x + 1)^2 ∧ (x * y^2 + y = 1) → false

theorem problem_solution : proof_problem :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l296_29676


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l296_29622

theorem boat_speed_in_still_water
  (v c : ℝ)
  (h1 : v + c = 10)
  (h2 : v - c = 4) :
  v = 7 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l296_29622


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l296_29653

   variable (a_n : ℕ → ℝ)
   variable (a_5 : ℝ := 13)
   variable (S_5 : ℝ := 35)
   variable (d : ℝ)

   theorem arithmetic_sequence_common_difference {a_1 : ℝ} :
     (a_1 + 4 * d = a_5) ∧ (5 * a_1 + 10 * d = S_5) → d = 3 :=
   by
     sorry
   
end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l296_29653


namespace NUMINAMATH_GPT_students_came_to_school_l296_29699

theorem students_came_to_school (F M T A : ℕ) 
    (hF : F = 658)
    (hM : M = F - 38)
    (hA : A = 17)
    (hT : T = M + F - A) :
    T = 1261 := by 
sorry

end NUMINAMATH_GPT_students_came_to_school_l296_29699


namespace NUMINAMATH_GPT_solution_set_f_x_minus_2_pos_l296_29687

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then 2 * x - 4 else 2 * (-x) - 4

theorem solution_set_f_x_minus_2_pos :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_f_x_minus_2_pos_l296_29687


namespace NUMINAMATH_GPT_evening_temperature_l296_29689

-- Definitions based on conditions
def noon_temperature : ℤ := 2
def temperature_drop : ℤ := 3

-- The theorem statement
theorem evening_temperature : noon_temperature - temperature_drop = -1 := 
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_evening_temperature_l296_29689


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_for_absolute_inequality_l296_29646

theorem necessary_and_sufficient_condition_for_absolute_inequality (a : ℝ) :
  (a < 3) ↔ (∀ x : ℝ, |x + 2| + |x - 1| > a) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_for_absolute_inequality_l296_29646


namespace NUMINAMATH_GPT_number_of_classes_l296_29606

theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_number_of_classes_l296_29606


namespace NUMINAMATH_GPT_football_team_matches_l296_29652

theorem football_team_matches (total_matches loses total_points: ℕ) 
  (points_win points_draw points_lose wins draws: ℕ)
  (h1: total_matches = 15)
  (h2: loses = 4)
  (h3: total_points = 29)
  (h4: points_win = 3)
  (h5: points_draw = 1)
  (h6: points_lose = 0)
  (h7: wins + draws + loses = total_matches)
  (h8: points_win * wins + points_draw * draws = total_points) :
  wins = 9 ∧ draws = 2 :=
sorry


end NUMINAMATH_GPT_football_team_matches_l296_29652


namespace NUMINAMATH_GPT_penguin_permutations_correct_l296_29600

def num_permutations_of_multiset (total : ℕ) (freqs : List ℕ) : ℕ :=
  Nat.factorial total / (freqs.foldl (λ acc x => acc * Nat.factorial x) 1)

def penguin_permutations : ℕ := num_permutations_of_multiset 7 [2, 1, 1, 1, 1, 1]

theorem penguin_permutations_correct : penguin_permutations = 2520 := by
  sorry

end NUMINAMATH_GPT_penguin_permutations_correct_l296_29600


namespace NUMINAMATH_GPT_seating_arrangement_count_l296_29625

-- Define the conditions.
def chairs : ℕ := 7
def people : ℕ := 5
def end_chairs : ℕ := 3

-- Define the main theorem to prove the number of arrangements.
theorem seating_arrangement_count :
  (end_chairs * 2) * (6 * 5 * 4 * 3) = 2160 := by
  sorry

end NUMINAMATH_GPT_seating_arrangement_count_l296_29625


namespace NUMINAMATH_GPT_percentage_increase_l296_29636

theorem percentage_increase (initial final : ℝ) (h_initial : initial = 200) (h_final : final = 250) :
  ((final - initial) / initial) * 100 = 25 := 
sorry

end NUMINAMATH_GPT_percentage_increase_l296_29636


namespace NUMINAMATH_GPT_min_value_of_expression_l296_29644

theorem min_value_of_expression {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : 
  (1 / a) + (2 / b) >= 8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l296_29644


namespace NUMINAMATH_GPT_second_cube_surface_area_l296_29615

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_second_cube_surface_area_l296_29615


namespace NUMINAMATH_GPT_example_solution_l296_29651

variable (x y θ : Real)
variable (h1 : 0 < x) (h2 : 0 < y)
variable (h3 : θ ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2))
variable (h4 : Real.sin θ / x = Real.cos θ / y)
variable (h5 : Real.cos θ ^ 2 / x ^ 2 + Real.sin θ ^ 2 / y ^ 2 = 10 / (3 * (x ^ 2 + y ^ 2)))

theorem example_solution : x / y = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_example_solution_l296_29651


namespace NUMINAMATH_GPT_appears_every_number_smallest_triplicate_number_l296_29694

open Nat

/-- Pascal's triangle is constructed such that each number 
    is the sum of the two numbers directly above it in the 
    previous row -/
def pascal (r k : ℕ) : ℕ :=
  if k > r then 0 else Nat.choose r k

/-- Every positive integer does appear at least once, but not 
    necessarily more than once for smaller numbers -/
theorem appears_every_number (n : ℕ) : ∃ r k : ℕ, pascal r k = n := sorry

/-- The smallest three-digit number in Pascal's triangle 
    that appears more than once is 102 -/
theorem smallest_triplicate_number : ∃ r1 k1 r2 k2 : ℕ, 
  100 ≤ pascal r1 k1 ∧ pascal r1 k1 < 1000 ∧ 
  pascal r1 k1 = 102 ∧ 
  r1 ≠ r2 ∧ k1 ≠ k2 ∧ 
  pascal r1 k1 = pascal r2 k2 := sorry

end NUMINAMATH_GPT_appears_every_number_smallest_triplicate_number_l296_29694


namespace NUMINAMATH_GPT_cos_double_angle_l296_29624

theorem cos_double_angle (a : ℝ) (h : Real.sin a = 1 / 3) : Real.cos (2 * a) = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l296_29624


namespace NUMINAMATH_GPT_half_angle_quadrant_l296_29691

theorem half_angle_quadrant
  (α : ℝ) (k : ℤ)
  (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (∃ m : ℤ, m * π < α / 2 ∧ α / 2 < m * π + π / 2) :=
by
  sorry

end NUMINAMATH_GPT_half_angle_quadrant_l296_29691


namespace NUMINAMATH_GPT_system_of_equations_solution_l296_29619

/-- Integer solutions to the system of equations:
    \begin{cases}
        xz - 2yt = 3 \\
        xt + yz = 1
    \end{cases}
-/
theorem system_of_equations_solution :
  ∃ (x y z t : ℤ), 
    x * z - 2 * y * t = 3 ∧ 
    x * t + y * z = 1 ∧
    ((x = 1 ∧ y = 0 ∧ z = 3 ∧ t = 1) ∨
     (x = -1 ∧ y = 0 ∧ z = -3 ∧ t = -1) ∨
     (x = 3 ∧ y = 1 ∧ z = 1 ∧ t = 0) ∨
     (x = -3 ∧ y = -1 ∧ z = -1 ∧ t = 0)) :=
by {
  sorry
}

end NUMINAMATH_GPT_system_of_equations_solution_l296_29619


namespace NUMINAMATH_GPT_sale_in_third_month_l296_29604

theorem sale_in_third_month (sale1 sale2 sale4 sale5 sale6 avg_sale : ℝ) (n_months : ℝ) (sale3 : ℝ):
  sale1 = 5400 →
  sale2 = 9000 →
  sale4 = 7200 →
  sale5 = 4500 →
  sale6 = 1200 →
  avg_sale = 5600 →
  n_months = 6 →
  (n_months * avg_sale) - (sale1 + sale2 + sale4 + sale5 + sale6) = sale3 →
  sale3 = 6300 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sale_in_third_month_l296_29604


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l296_29608

theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x, 1 < x ∧ x < 3 → x^2 < ax + b) : b^a = 81 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l296_29608


namespace NUMINAMATH_GPT_largest_possible_percent_error_l296_29626

theorem largest_possible_percent_error 
  (r : ℝ) (delta : ℝ) (h_r : r = 15) (h_delta : delta = 0.1) : 
  ∃(error : ℝ), error = 0.21 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_largest_possible_percent_error_l296_29626


namespace NUMINAMATH_GPT_exists_integer_coordinates_l296_29632

theorem exists_integer_coordinates :
  ∃ (x y : ℤ), (x^2 + y^2) = 2 * 2017^2 + 2 * 2018^2 :=
by
  sorry

end NUMINAMATH_GPT_exists_integer_coordinates_l296_29632


namespace NUMINAMATH_GPT_increasing_function_range_l296_29631

theorem increasing_function_range (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x = x^3 - a * x - 1) :
  (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) ↔ a ≤ 0 :=
sorry

end NUMINAMATH_GPT_increasing_function_range_l296_29631


namespace NUMINAMATH_GPT_selected_people_take_B_l296_29681

def arithmetic_sequence (a d n : Nat) : Nat := a + (n - 1) * d

theorem selected_people_take_B (a d total sampleCount start n_upper n_lower : Nat) :
  a = 9 →
  d = 30 →
  total = 960 →
  sampleCount = 32 →
  start = 451 →
  n_upper = 25 →
  n_lower = 16 →
  (960 / 32) = d → 
  (10 = n_upper - n_lower + 1) ∧ 
  ∀ n, (n_lower ≤ n ∧ n ≤ n_upper) → (start ≤ arithmetic_sequence a d n ∧ arithmetic_sequence a d n ≤ 750) :=
by sorry

end NUMINAMATH_GPT_selected_people_take_B_l296_29681


namespace NUMINAMATH_GPT_negation_of_proposition_l296_29648

noncomputable def negation_proposition (f : ℝ → Prop) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ ¬ f x

theorem negation_of_proposition :
  (∀ x : ℝ, x ≥ 0 → x^2 + x - 1 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 + x - 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l296_29648


namespace NUMINAMATH_GPT_parabola_focus_to_equation_l296_29661

-- Define the focus of the parabola
def F : (ℝ × ℝ) := (5, 0)

-- Define the standard equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 20 * x

-- State the problem in Lean
theorem parabola_focus_to_equation : 
  (F = (5, 0)) → ∀ x y, parabola_equation x y :=
by
  intro h_focus_eq
  sorry

end NUMINAMATH_GPT_parabola_focus_to_equation_l296_29661


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l296_29665

noncomputable def problem1 : ℝ :=
  (Real.sqrt (1 / 3) + Real.sqrt 6) / Real.sqrt 3

noncomputable def problem2 : ℝ :=
  (Real.sqrt 3)^2 - Real.sqrt 4 + Real.sqrt ((-2)^2)

theorem problem1_solution :
  problem1 = 1 + 3 * Real.sqrt 2 :=
by
  sorry

theorem problem2_solution :
  problem2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l296_29665


namespace NUMINAMATH_GPT_train_length_calculation_l296_29695

theorem train_length_calculation (L : ℝ) (t : ℝ) (v_faster : ℝ) (v_slower : ℝ) (relative_speed : ℝ) (total_distance : ℝ) :
  (v_faster = 60) →
  (v_slower = 40) →
  (relative_speed = (v_faster - v_slower) * 1000 / 3600) →
  (t = 48) →
  (total_distance = relative_speed * t) →
  (2 * L = total_distance) →
  L = 133.44 :=
by
  intros
  sorry

end NUMINAMATH_GPT_train_length_calculation_l296_29695


namespace NUMINAMATH_GPT_find_theta_l296_29682

noncomputable def P := (Real.sin (3 * Real.pi / 4), Real.cos (3 * Real.pi / 4))

theorem find_theta
  (theta : ℝ)
  (h_theta_range : 0 ≤ theta ∧ theta < 2 * Real.pi)
  (h_P_theta : P = (Real.sin theta, Real.cos theta)) :
  theta = 7 * Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_find_theta_l296_29682


namespace NUMINAMATH_GPT_mr_william_land_percentage_l296_29638

-- Define the conditions
def farm_tax_percentage : ℝ := 0.5
def total_tax_collected : ℝ := 3840
def mr_william_tax : ℝ := 480

-- Theorem statement proving the question == answer
theorem mr_william_land_percentage : 
  (mr_william_tax / total_tax_collected) * 100 = 12.5 := 
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_mr_william_land_percentage_l296_29638


namespace NUMINAMATH_GPT_cones_slant_height_angle_l296_29670

theorem cones_slant_height_angle :
  ∀ (α: ℝ),
  α = 2 * Real.arccos (Real.sqrt (2 / (2 + Real.sqrt 2))) :=
by
  sorry

end NUMINAMATH_GPT_cones_slant_height_angle_l296_29670


namespace NUMINAMATH_GPT_find_k_l296_29634

theorem find_k
  (AB AC : ℝ)
  (k : ℝ)
  (h1 : AB = AC)
  (h2 : AB = 8)
  (h3 : AC = 5 - k) : k = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l296_29634


namespace NUMINAMATH_GPT_HCF_is_five_l296_29601

noncomputable def HCF_of_numbers (a b : ℕ) : ℕ := Nat.gcd a b

theorem HCF_is_five :
  ∃ (a b : ℕ),
    a + b = 55 ∧
    Nat.lcm a b = 120 ∧
    (1 / (a : ℝ) + 1 / (b : ℝ) = 0.09166666666666666) →
    HCF_of_numbers a b = 5 :=
by 
  sorry

end NUMINAMATH_GPT_HCF_is_five_l296_29601


namespace NUMINAMATH_GPT_monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value_l296_29658

-- Proof Problem I
noncomputable def f1 (x : ℝ) := x^2 + x - Real.log x

theorem monotonic_intervals_a1 : 
  (∀ x, 0 < x ∧ x < 1 / 2 → f1 x < 0) ∧ (∀ x, 1 / 2 < x → f1 x > 0) := 
sorry

-- Proof Problem II
noncomputable def f2 (x : ℝ) (a : ℝ) := x^2 + a * x - Real.log x

theorem decreasing_on_1_to_2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f2 x a ≤ 0) → a ≤ -7 / 2 :=
sorry

-- Proof Problem III
noncomputable def g (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem exists_a_for_minimum_value :
  ∃ a : ℝ, (∀ x, 0 < x ∧ x ≤ Real.exp 1 → g x a = 3) ∧ a = Real.exp 2 :=
sorry

end NUMINAMATH_GPT_monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value_l296_29658


namespace NUMINAMATH_GPT_binom_10_8_equals_45_l296_29657

theorem binom_10_8_equals_45 : Nat.choose 10 8 = 45 := 
by
  sorry

end NUMINAMATH_GPT_binom_10_8_equals_45_l296_29657


namespace NUMINAMATH_GPT_four_diff_digits_per_day_l296_29678

def valid_time_period (start_hour : ℕ) (end_hour : ℕ) : ℕ :=
  let total_minutes := (end_hour - start_hour + 1) * 60
  let valid_combinations :=
    match start_hour with
    | 0 => 0  -- start with appropriate calculation logic
    | 2 => 0  -- start with appropriate calculation logic
    | _ => 0  -- for general case, replace with correct logic
  total_minutes + valid_combinations  -- use proper aggregation

theorem four_diff_digits_per_day :
  valid_time_period 0 19 + valid_time_period 20 23 = 588 :=
by
  sorry

end NUMINAMATH_GPT_four_diff_digits_per_day_l296_29678


namespace NUMINAMATH_GPT_find_third_number_l296_29612

theorem find_third_number :
  let total_sum := 121526
  let first_addend := 88888
  let second_addend := 1111
  (total_sum = first_addend + second_addend + 31527) :=
by
  sorry

end NUMINAMATH_GPT_find_third_number_l296_29612


namespace NUMINAMATH_GPT_kims_total_points_l296_29683

theorem kims_total_points :
  let points_easy := 2
  let points_average := 3
  let points_hard := 5
  let answers_easy := 6
  let answers_average := 2
  let answers_hard := 4
  let total_points := (answers_easy * points_easy) + (answers_average * points_average) + (answers_hard * points_hard)
  total_points = 38 :=
by
  -- This is a placeholder to indicate that the proof is not included.
  sorry

end NUMINAMATH_GPT_kims_total_points_l296_29683


namespace NUMINAMATH_GPT_y_capital_l296_29693

theorem y_capital (X Y Z : ℕ) (Pz : ℕ) (Z_months_after_start : ℕ) (total_profit Z_share : ℕ)
    (hx : X = 20000)
    (hz : Z = 30000)
    (hz_profit : Z_share = 14000)
    (htotal_profit : total_profit = 50000)
    (hZ_months : Z_months_after_start = 5)
  : Y = 25000 := 
by
  -- Here we would have a proof, skipped with sorry for now
  sorry

end NUMINAMATH_GPT_y_capital_l296_29693


namespace NUMINAMATH_GPT_walnuts_left_in_burrow_l296_29616

-- Definitions of conditions
def boy_gathers : ℕ := 15
def originally_in_burrow : ℕ := 25
def boy_drops : ℕ := 3
def boy_hides : ℕ := 5
def girl_brings : ℕ := 12
def girl_eats : ℕ := 4
def girl_gives_away : ℕ := 3
def girl_loses : ℕ := 2

-- Theorem statement
theorem walnuts_left_in_burrow : 
  originally_in_burrow + (boy_gathers - boy_drops - boy_hides) + 
  (girl_brings - girl_eats - girl_gives_away - girl_loses) = 35 := 
sorry

end NUMINAMATH_GPT_walnuts_left_in_burrow_l296_29616


namespace NUMINAMATH_GPT_suzy_twice_mary_l296_29679

def suzy_current_age : ℕ := 20
def mary_current_age : ℕ := 8

theorem suzy_twice_mary (x : ℕ) : suzy_current_age + x = 2 * (mary_current_age + x) ↔ x = 4 := by
  sorry

end NUMINAMATH_GPT_suzy_twice_mary_l296_29679


namespace NUMINAMATH_GPT_inequalities_not_simultaneous_l296_29688

theorem inequalities_not_simultaneous (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (ineq1 : a + b < c + d) (ineq2 : (a + b) * (c + d) < a * b + c * d) (ineq3 : (a + b) * c * d < (c + d) * a * b) :
  false := 
sorry

end NUMINAMATH_GPT_inequalities_not_simultaneous_l296_29688


namespace NUMINAMATH_GPT_probability_correct_l296_29614

noncomputable def probability_of_getting_number_greater_than_4 : ℚ :=
  let favorable_outcomes := 2
  let total_outcomes := 6
  favorable_outcomes / total_outcomes

theorem probability_correct :
  probability_of_getting_number_greater_than_4 = 1 / 3 := by sorry

end NUMINAMATH_GPT_probability_correct_l296_29614


namespace NUMINAMATH_GPT_determine_b_eq_l296_29621

theorem determine_b_eq (b : ℝ) : (∃! (x : ℝ), |x^2 + 3 * b * x + 4 * b| ≤ 3) ↔ b = 4 / 3 ∨ b = 1 := 
by sorry

end NUMINAMATH_GPT_determine_b_eq_l296_29621


namespace NUMINAMATH_GPT_evaluate_F_of_4_and_f_of_5_l296_29623

def f (a : ℤ) : ℤ := 2 * a - 2
def F (a b : ℤ) : ℤ := b^2 + a + 1

theorem evaluate_F_of_4_and_f_of_5 : F 4 (f 5) = 69 := by
  -- Definitions and intermediate steps are not included in the statement, proof is omitted.
  sorry

end NUMINAMATH_GPT_evaluate_F_of_4_and_f_of_5_l296_29623


namespace NUMINAMATH_GPT_function_C_is_quadratic_l296_29692

def isQuadratic (f : ℝ → ℝ) :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

def function_C (x : ℝ) : ℝ := (x + 1)^2 - 5

theorem function_C_is_quadratic : isQuadratic function_C :=
by
  sorry

end NUMINAMATH_GPT_function_C_is_quadratic_l296_29692


namespace NUMINAMATH_GPT_problem_curves_l296_29645

theorem problem_curves (x y : ℝ) : 
  ((x * (x^2 + y^2 - 4) = 0 → (x = 0 ∨ x^2 + y^2 = 4)) ∧
  (x^2 + (x^2 + y^2 - 4)^2 = 0 → ((x = 0 ∧ y = -2) ∨ (x = 0 ∧ y = 2)))) :=
by
  sorry -- proof to be filled in later

end NUMINAMATH_GPT_problem_curves_l296_29645


namespace NUMINAMATH_GPT_polynomial_divisibility_l296_29674

open Polynomial

variables {R : Type*} [CommRing R]
variables {f g h k : R[X]}

theorem polynomial_divisibility (h1 : (X^2 + 1) * h + (X - 1) * f + (X - 2) * g = 0)
    (h2 : (X^2 + 1) * k + (X + 1) * f + (X + 2) * g = 0) :
    (X^2 + 1) ∣ (f * g) :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_l296_29674


namespace NUMINAMATH_GPT_proof_A2_less_than_3A1_plus_n_l296_29650

-- Define the conditions in terms of n, A1, and A2.
variables (n : ℕ)

-- A1 and A2 are the numbers of selections to select two students
-- such that their weight difference is ≤ 1 kg and ≤ 2 kg respectively.
variables (A1 A2 : ℕ)

-- The main theorem needs to prove that A2 < 3 * A1 + n.
theorem proof_A2_less_than_3A1_plus_n (h : A2 < 3 * A1 + n) : A2 < 3 * A1 + n :=
by {
  sorry -- proof goes here, but it's not required for the Lean statement.
}

end NUMINAMATH_GPT_proof_A2_less_than_3A1_plus_n_l296_29650


namespace NUMINAMATH_GPT_min_value_exponential_sub_l296_29654

theorem min_value_exponential_sub (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h : x + 2 * y = x * y) : ∃ y₀ > 0, ∀ y > 1, e^y - 8 / x ≥ e :=
by
  sorry

end NUMINAMATH_GPT_min_value_exponential_sub_l296_29654


namespace NUMINAMATH_GPT_f_decreasing_ln_inequality_limit_inequality_l296_29662

-- Definitions of the given conditions
noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) / x

-- Statements we need to prove

-- (I) Prove that f(x) is decreasing on (0, +∞)
theorem f_decreasing : ∀ x y : ℝ, 0 < x → x < y → f y < f x := sorry

-- (II) Prove that for the inequality ln(1 + x) < ax to hold for all x in (0, +∞), a must be at least 1
theorem ln_inequality (a : ℝ) : (∀ x : ℝ, 0 < x → Real.log (1 + x) < a * x) ↔ 1 ≤ a := sorry

-- (III) Prove that (1 + 1/n)^n < e for all n in ℕ*
theorem limit_inequality (n : ℕ) (h : n ≠ 0) : (1 + 1 / n) ^ n < Real.exp 1 := sorry

end NUMINAMATH_GPT_f_decreasing_ln_inequality_limit_inequality_l296_29662


namespace NUMINAMATH_GPT_helens_mother_brought_101_l296_29613

-- Define the conditions
def total_hotdogs : ℕ := 480
def dylan_mother_hotdogs : ℕ := 379
def helens_mother_hotdogs := total_hotdogs - dylan_mother_hotdogs

-- Theorem statement: Prove that the number of hotdogs Helen's mother brought is 101
theorem helens_mother_brought_101 : helens_mother_hotdogs = 101 :=
by
  sorry

end NUMINAMATH_GPT_helens_mother_brought_101_l296_29613


namespace NUMINAMATH_GPT_border_area_l296_29618

theorem border_area (h_photo : ℕ) (w_photo : ℕ) (border : ℕ) (h : h_photo = 8) (w : w_photo = 10) (b : border = 2) :
  (2 * (border + h_photo) * (border + w_photo) - h_photo * w_photo) = 88 :=
by
  rw [h, w, b]
  sorry

end NUMINAMATH_GPT_border_area_l296_29618
