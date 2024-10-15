import Mathlib

namespace NUMINAMATH_GPT_distributor_B_lower_avg_price_l1461_146124

theorem distributor_B_lower_avg_price (p_1 p_2 : ℝ) (h : p_1 < p_2) :
  (p_1 + p_2) / 2 > (2 * p_1 * p_2) / (p_1 + p_2) :=
by {
  sorry
}

end NUMINAMATH_GPT_distributor_B_lower_avg_price_l1461_146124


namespace NUMINAMATH_GPT_min_photographs_42_tourists_3_monuments_l1461_146158

noncomputable def min_photos_taken (num_tourists : ℕ) (num_monuments : ℕ) : ℕ :=
  if num_tourists = 42 ∧ num_monuments = 3 then 123 else 0

-- Main statement:
theorem min_photographs_42_tourists_3_monuments : 
  (∀ (num_tourists num_monuments : ℕ), 
    num_tourists = 42 ∧ num_monuments = 3 → min_photos_taken num_tourists num_monuments = 123)
  := by
    sorry

end NUMINAMATH_GPT_min_photographs_42_tourists_3_monuments_l1461_146158


namespace NUMINAMATH_GPT_solve_problem_l1461_146104

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem solve_problem :
  is_prime 2017 :=
by
  have h1 : 2017 > 1 := by linarith
  have h2 : ∀ m : ℕ, m ∣ 2017 → m = 1 ∨ m = 2017 :=
    sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_solve_problem_l1461_146104


namespace NUMINAMATH_GPT_present_age_of_A_l1461_146145

theorem present_age_of_A {x : ℕ} (h₁ : ∃ (x : ℕ), 5 * x = A ∧ 3 * x = B)
                         (h₂ : ∀ (A B : ℕ), (A + 6) / (B + 6) = 7 / 5) : A = 15 :=
by sorry

end NUMINAMATH_GPT_present_age_of_A_l1461_146145


namespace NUMINAMATH_GPT_donna_received_total_interest_l1461_146120

-- Donna's investment conditions
def totalInvestment : ℝ := 33000
def investmentAt4Percent : ℝ := 13000
def investmentAt225Percent : ℝ := totalInvestment - investmentAt4Percent
def rate4Percent : ℝ := 0.04
def rate225Percent : ℝ := 0.0225

-- The interest calculation
def interestFrom4PercentInvestment : ℝ := investmentAt4Percent * rate4Percent
def interestFrom225PercentInvestment : ℝ := investmentAt225Percent * rate225Percent
def totalInterest : ℝ := interestFrom4PercentInvestment + interestFrom225PercentInvestment

-- The proof statement
theorem donna_received_total_interest :
  totalInterest = 970 := by
sorry

end NUMINAMATH_GPT_donna_received_total_interest_l1461_146120


namespace NUMINAMATH_GPT_arcsin_sqrt_3_div_2_is_pi_div_3_l1461_146133

noncomputable def arcsin_sqrt_3_div_2 : ℝ := Real.arcsin (Real.sqrt 3 / 2)

theorem arcsin_sqrt_3_div_2_is_pi_div_3 : arcsin_sqrt_3_div_2 = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_sqrt_3_div_2_is_pi_div_3_l1461_146133


namespace NUMINAMATH_GPT_largest_multiple_of_15_less_than_400_l1461_146130

theorem largest_multiple_of_15_less_than_400 (x : ℕ) (k : ℕ) (h : x = 15 * k) (h1 : x < 400) (h2 : ∀ m : ℕ, (15 * m < 400) → m ≤ k) : x = 390 :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_15_less_than_400_l1461_146130


namespace NUMINAMATH_GPT_avg_cost_is_12_cents_l1461_146177

noncomputable def avg_cost_per_pencil 
    (price_per_package : ℝ)
    (num_pencils : ℕ)
    (shipping_cost : ℝ)
    (discount_rate : ℝ) : ℝ :=
  let price_after_discount := price_per_package - (discount_rate * price_per_package)
  let total_cost := price_after_discount + shipping_cost
  let total_cost_cents := total_cost * 100
  total_cost_cents / num_pencils

theorem avg_cost_is_12_cents :
  avg_cost_per_pencil 29.70 300 8.50 0.10 = 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_avg_cost_is_12_cents_l1461_146177


namespace NUMINAMATH_GPT_total_pages_in_book_l1461_146163

-- Conditions
def hours_reading := 5
def pages_read := 2323
def increase_per_hour := 10
def extra_pages_read := 90

-- Main statement to prove
theorem total_pages_in_book (T : ℕ) :
  (∃ P : ℕ, P + (P + increase_per_hour) + (P + 2 * increase_per_hour) + 
   (P + 3 * increase_per_hour) + (P + 4 * increase_per_hour) = pages_read) ∧
  (pages_read = T - pages_read + extra_pages_read) →
  T = 4556 :=
by { sorry }

end NUMINAMATH_GPT_total_pages_in_book_l1461_146163


namespace NUMINAMATH_GPT_equations_not_equivalent_l1461_146192

theorem equations_not_equivalent :
  ∀ x : ℝ, (x + 7 + 10 / (2 * x - 1) = 8 - x + 10 / (2 * x - 1)) ↔ false :=
by
  intro x
  sorry

end NUMINAMATH_GPT_equations_not_equivalent_l1461_146192


namespace NUMINAMATH_GPT_price_of_small_bags_l1461_146135

theorem price_of_small_bags (price_medium_bag : ℤ) (price_large_bag : ℤ) 
  (money_mark_has : ℤ) (balloons_in_small_bag : ℤ) 
  (balloons_in_medium_bag : ℤ) (balloons_in_large_bag : ℤ) 
  (total_balloons : ℤ) : 
  price_medium_bag = 6 → 
  price_large_bag = 12 → 
  money_mark_has = 24 → 
  balloons_in_small_bag = 50 → 
  balloons_in_medium_bag = 75 → 
  balloons_in_large_bag = 200 → 
  total_balloons = 400 → 
  (money_mark_has / (total_balloons / balloons_in_small_bag)) = 3 :=
by 
  sorry

end NUMINAMATH_GPT_price_of_small_bags_l1461_146135


namespace NUMINAMATH_GPT_bridgette_total_baths_l1461_146118

def bridgette_baths (dogs baths_per_dog_per_month cats baths_per_cat_per_month birds baths_per_bird_per_month : ℕ) : ℕ :=
  (dogs * baths_per_dog_per_month * 12) + (cats * baths_per_cat_per_month * 12) + (birds * (12 / baths_per_bird_per_month))

theorem bridgette_total_baths :
  bridgette_baths 2 2 3 1 4 4 = 96 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_bridgette_total_baths_l1461_146118


namespace NUMINAMATH_GPT_gemma_amount_given_l1461_146164

theorem gemma_amount_given
  (cost_per_pizza : ℕ)
  (number_of_pizzas : ℕ)
  (tip : ℕ)
  (change_back : ℕ)
  (h1 : cost_per_pizza = 10)
  (h2 : number_of_pizzas = 4)
  (h3 : tip = 5)
  (h4 : change_back = 5) :
  number_of_pizzas * cost_per_pizza + tip + change_back = 50 := sorry

end NUMINAMATH_GPT_gemma_amount_given_l1461_146164


namespace NUMINAMATH_GPT_perpendicular_slope_l1461_146129

theorem perpendicular_slope (x y : ℝ) (h : 5 * x - 4 * y = 20) : 
  ∃ m : ℝ, m = -4 / 5 :=
sorry

end NUMINAMATH_GPT_perpendicular_slope_l1461_146129


namespace NUMINAMATH_GPT_hyperbola_with_foci_on_y_axis_l1461_146162

variable (m n : ℝ)

-- condition stating that mn < 0
def mn_neg : Prop := m * n < 0

-- the main theorem statement
theorem hyperbola_with_foci_on_y_axis (h : mn_neg m n) : 
  (∃ a : ℝ, a > 0 ∧ ∀ x y : ℝ, m * x^2 - m * y^2 = n ↔ y^2 - x^2 = a) :=
sorry

end NUMINAMATH_GPT_hyperbola_with_foci_on_y_axis_l1461_146162


namespace NUMINAMATH_GPT_sum_of_perimeters_l1461_146101

theorem sum_of_perimeters (x y : ℝ) (h₁ : x^2 + y^2 = 125) (h₂ : x^2 - y^2 = 65) : 4 * x + 4 * y = 60 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_perimeters_l1461_146101


namespace NUMINAMATH_GPT_olivia_pays_in_dollars_l1461_146136

theorem olivia_pays_in_dollars (q_chips q_soda : ℕ) 
  (h_chips : q_chips = 4) (h_soda : q_soda = 12) : (q_chips + q_soda) / 4 = 4 := by
  sorry

end NUMINAMATH_GPT_olivia_pays_in_dollars_l1461_146136


namespace NUMINAMATH_GPT_find_divisor_l1461_146140

theorem find_divisor (d : ℕ) (h1 : 127 = d * 5 + 2) : d = 25 :=
sorry

end NUMINAMATH_GPT_find_divisor_l1461_146140


namespace NUMINAMATH_GPT_number_halfway_l1461_146178

theorem number_halfway (a b : ℚ) (h1 : a = 1/12) (h2 : b = 1/10) : (a + b) / 2 = 11 / 120 := by
  sorry

end NUMINAMATH_GPT_number_halfway_l1461_146178


namespace NUMINAMATH_GPT_emily_age_proof_l1461_146155

theorem emily_age_proof (e m : ℕ) (h1 : e = m - 18) (h2 : e + m = 54) : e = 18 :=
by
  sorry

end NUMINAMATH_GPT_emily_age_proof_l1461_146155


namespace NUMINAMATH_GPT_medians_sum_le_circumradius_l1461_146194

-- Definition of the problem
variable (a b c R : ℝ) (m_a m_b m_c : ℝ)

-- Conditions: medians of triangle ABC, and R is the circumradius
def is_median (m : ℝ) (a b c : ℝ) : Prop :=
  m^2 = (2*b^2 + 2*c^2 - a^2) / 4

-- Main theorem to prove
theorem medians_sum_le_circumradius (h_ma : is_median m_a a b c)
  (h_mb : is_median m_b b a c) (h_mc : is_median m_c c a b) 
  (h_R : a^2 + b^2 + c^2 ≤ 9 * R^2) :
  m_a + m_b + m_c ≤ 9 / 2 * R :=
sorry

end NUMINAMATH_GPT_medians_sum_le_circumradius_l1461_146194


namespace NUMINAMATH_GPT_descent_phase_duration_l1461_146191

noncomputable def start_time_in_seconds : ℕ := 45 * 60 + 39
noncomputable def end_time_in_seconds : ℕ := 47 * 60 + 33

theorem descent_phase_duration :
  end_time_in_seconds - start_time_in_seconds = 114 := by
  sorry

end NUMINAMATH_GPT_descent_phase_duration_l1461_146191


namespace NUMINAMATH_GPT_count_six_digit_numbers_with_at_least_one_zero_l1461_146112

theorem count_six_digit_numbers_with_at_least_one_zero : 
  900000 - 531441 = 368559 :=
by
  sorry

end NUMINAMATH_GPT_count_six_digit_numbers_with_at_least_one_zero_l1461_146112


namespace NUMINAMATH_GPT_part1_part2_l1461_146197

-- Part 1
theorem part1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := sorry

-- Part 2
theorem part2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a * b + b * c + c * a ≤ 1 / 3 := sorry

end NUMINAMATH_GPT_part1_part2_l1461_146197


namespace NUMINAMATH_GPT_second_divisor_l1461_146172

theorem second_divisor (N k D m : ℤ) (h1 : N = 35 * k + 25) (h2 : N = D * m + 4) : D = 17 := by
  -- Follow conditions from problem
  sorry

end NUMINAMATH_GPT_second_divisor_l1461_146172


namespace NUMINAMATH_GPT_find_unknown_rate_l1461_146128

-- Define the known quantities
def num_blankets1 := 4
def price1 := 100

def num_blankets2 := 5
def price2 := 150

def num_blankets3 := 3
def price3 := 200

def num_blankets4 := 6
def price4 := 75

def num_blankets_unknown := 2

def avg_price := 150
def total_blankets := num_blankets1 + num_blankets2 + num_blankets3 + num_blankets4 + num_blankets_unknown -- 20 blankets in total

-- Hypotheses
def total_known_cost := num_blankets1 * price1 + num_blankets2 * price2 + num_blankets3 * price3 + num_blankets4 * price4
-- 2200 Rs.

def total_cost := total_blankets * avg_price -- 3000 Rs.

theorem find_unknown_rate :
  (total_cost - total_known_cost) / num_blankets_unknown = 400 :=
by sorry

end NUMINAMATH_GPT_find_unknown_rate_l1461_146128


namespace NUMINAMATH_GPT_find_other_discount_l1461_146126

def other_discount (list_price final_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) : Prop :=
  let price_after_first_discount := list_price - (first_discount / 100) * list_price
  final_price = price_after_first_discount - (second_discount / 100) * price_after_first_discount

theorem find_other_discount : 
  other_discount 70 59.22 10 6 :=
by
  sorry

end NUMINAMATH_GPT_find_other_discount_l1461_146126


namespace NUMINAMATH_GPT_geom_inequality_l1461_146168

variables {Point : Type} [MetricSpace Point] {O A B C K L H M : Point}

/-- Conditions -/
def circumcenter_of_triangle (O A B C : Point) : Prop := 
 -- Definition that O is the circumcenter of triangle ABC
 sorry 

def midpoint_of_arc (K B C A : Point) : Prop := 
 -- Definition that K is the midpoint of the arc BC not containing A
 sorry

def lies_on_line (K L A : Point) : Prop := 
 -- Definition that K lies on line AL
 sorry

def similar_triangles (A H L K M : Point) : Prop := 
 -- Definition that triangles AHL and KML are similar
 sorry 

def segment_inequality (AL KL : ℝ) : Prop := 
 -- Definition that AL < KL
 sorry 

/-- Proof Problem -/
theorem geom_inequality (h1 : circumcenter_of_triangle O A B C) 
                       (h2: midpoint_of_arc K B C A)
                       (h3: lies_on_line K L A)
                       (h4: similar_triangles A H L K M)
                       (h5: segment_inequality (dist A L) (dist K L)) : 
  dist A K < dist B C := 
sorry

end NUMINAMATH_GPT_geom_inequality_l1461_146168


namespace NUMINAMATH_GPT_line_intersects_y_axis_at_eight_l1461_146176

theorem line_intersects_y_axis_at_eight :
  ∃ b : ℝ, ∃ f : ℝ → ℝ, (∀ x, f x = 2 * x + b) ∧ f 1 = 10 ∧ f (-9) = -10 ∧ f 0 = 8 :=
by
  -- Definitions and calculations leading to verify the theorem
  sorry

end NUMINAMATH_GPT_line_intersects_y_axis_at_eight_l1461_146176


namespace NUMINAMATH_GPT_cost_of_siding_l1461_146144

def area_of_wall (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def area_of_roof (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length * width)

def area_of_sheet (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def sheets_needed (total_area : ℕ) (sheet_area : ℕ) : ℕ :=
  (total_area + sheet_area - 1) / sheet_area  -- Cooling the ceiling with integer arithmetic

def total_cost (sheets : ℕ) (price_per_sheet : ℕ) : ℕ :=
  sheets * price_per_sheet

theorem cost_of_siding : 
  ∀ (length_wall width_wall length_roof width_roof length_sheet width_sheet price_per_sheet : ℕ),
  length_wall = 10 → width_wall = 7 →
  length_roof = 10 → width_roof = 6 →
  length_sheet = 10 → width_sheet = 14 →
  price_per_sheet = 50 →
  total_cost (sheets_needed (area_of_wall length_wall width_wall + area_of_roof length_roof width_roof) (area_of_sheet length_sheet width_sheet)) price_per_sheet = 100 :=
by
  intros
  simp [area_of_wall, area_of_roof, area_of_sheet, sheets_needed, total_cost]
  sorry

end NUMINAMATH_GPT_cost_of_siding_l1461_146144


namespace NUMINAMATH_GPT_least_clock_equivalent_l1461_146169

theorem least_clock_equivalent (t : ℕ) (h : t > 5) : 
  (t^2 - t) % 24 = 0 → t = 9 :=
by
  sorry

end NUMINAMATH_GPT_least_clock_equivalent_l1461_146169


namespace NUMINAMATH_GPT_cubic_polynomial_greater_than_zero_l1461_146149

theorem cubic_polynomial_greater_than_zero (x : ℝ) : x^3 + 3*x^2 + x - 5 > 0 → x > 1 :=
sorry

end NUMINAMATH_GPT_cubic_polynomial_greater_than_zero_l1461_146149


namespace NUMINAMATH_GPT_proportion_of_bike_riders_is_correct_l1461_146148

-- Define the given conditions as constants
def total_students : ℕ := 92
def bus_riders : ℕ := 20
def walkers : ℕ := 27

-- Define the remaining students after bus riders and after walkers
def remaining_after_bus_riders : ℕ := total_students - bus_riders
def bike_riders : ℕ := remaining_after_bus_riders - walkers

-- Define the expected proportion
def expected_proportion : ℚ := 45 / 72

-- State the theorem to be proved
theorem proportion_of_bike_riders_is_correct :
  (↑bike_riders / ↑remaining_after_bus_riders : ℚ) = expected_proportion := 
by
  sorry

end NUMINAMATH_GPT_proportion_of_bike_riders_is_correct_l1461_146148


namespace NUMINAMATH_GPT_jackie_walks_daily_l1461_146105

theorem jackie_walks_daily (x : ℝ) :
  (∀ t : ℕ, t = 6 →
    6 * x = 6 * 1.5 + 3) →
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_jackie_walks_daily_l1461_146105


namespace NUMINAMATH_GPT_translate_line_up_l1461_146166

-- Define the original line equation as a function
def original_line (x : ℝ) : ℝ := 2 * x - 4

-- Define the new line equation after translating upwards by 5 units
def new_line (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement to prove the translation result
theorem translate_line_up (x : ℝ) : original_line x + 5 = new_line x :=
by
  -- This would normally be where the proof goes, but we'll insert a placeholder
  sorry

end NUMINAMATH_GPT_translate_line_up_l1461_146166


namespace NUMINAMATH_GPT_bicycles_difference_on_october_1_l1461_146184

def initial_inventory : Nat := 200
def february_decrease : Nat := 4
def march_decrease : Nat := 6
def april_decrease : Nat := 8
def may_decrease : Nat := 10
def june_decrease : Nat := 12
def july_decrease : Nat := 14
def august_decrease : Nat := 16 + 20
def september_decrease : Nat := 18
def shipment : Nat := 50

def total_decrease : Nat := february_decrease + march_decrease + april_decrease + may_decrease + june_decrease + july_decrease + august_decrease + september_decrease
def stock_increase : Nat := shipment
def net_decrease : Nat := total_decrease - stock_increase

theorem bicycles_difference_on_october_1 : initial_inventory - net_decrease = 58 := by
  sorry

end NUMINAMATH_GPT_bicycles_difference_on_october_1_l1461_146184


namespace NUMINAMATH_GPT_no_a_where_A_eq_B_singleton_l1461_146153

def f (a x : ℝ) := x^2 + 4 * x - 2 * a
def g (a x : ℝ) := x^2 - a * x + a + 3

theorem no_a_where_A_eq_B_singleton :
  ∀ a : ℝ,
    (∃ x₁ : ℝ, (f a x₁ ≤ 0 ∧ ∀ x₂, f a x₂ ≤ 0 → x₂ = x₁)) ∧
    (∃ y₁ : ℝ, (g a y₁ ≤ 0 ∧ ∀ y₂, g a y₂ ≤ 0 → y₂ = y₁)) →
    (¬ ∃ z : ℝ, (f a z ≤ 0) ∧ (g a z ≤ 0)) := 
by
  sorry

end NUMINAMATH_GPT_no_a_where_A_eq_B_singleton_l1461_146153


namespace NUMINAMATH_GPT_simplify_expression_l1461_146115

variable (a : Real)

theorem simplify_expression : (-2 * a) * a - (-2 * a)^2 = -6 * a^2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1461_146115


namespace NUMINAMATH_GPT_volume_of_tetrahedron_eq_20_l1461_146117

noncomputable def volume_tetrahedron (a b c : ℝ) : ℝ :=
  1 / 3 * a * b * c

theorem volume_of_tetrahedron_eq_20 {x y z : ℝ} (h1 : x^2 + y^2 = 25) (h2 : y^2 + z^2 = 41) (h3 : z^2 + x^2 = 34) :
  volume_tetrahedron 3 4 5 = 20 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_tetrahedron_eq_20_l1461_146117


namespace NUMINAMATH_GPT_find_876_last_three_digits_l1461_146167

noncomputable def has_same_last_three_digits (N : ℕ) : Prop :=
  (N^2 - N) % 1000 = 0

theorem find_876_last_three_digits (N : ℕ) (h1 : has_same_last_three_digits N) (h2 : N > 99) (h3 : N < 1000) : 
  N % 1000 = 876 :=
sorry

end NUMINAMATH_GPT_find_876_last_three_digits_l1461_146167


namespace NUMINAMATH_GPT_largest_a_value_l1461_146187

theorem largest_a_value (a b c : ℝ) (h1 : a + b + c = 7) (h2 : ab + ac + bc = 12) : 
  a ≤ (7 + Real.sqrt 46) / 3 :=
sorry

end NUMINAMATH_GPT_largest_a_value_l1461_146187


namespace NUMINAMATH_GPT_find_n_from_sum_of_coeffs_l1461_146102

-- The mathematical conditions and question translated to Lean

def sum_of_coefficients (n : ℕ) : ℕ := 6 ^ n
def binomial_coefficients_sum (n : ℕ) : ℕ := 2 ^ n

theorem find_n_from_sum_of_coeffs (n : ℕ) (M N : ℕ) (hM : M = sum_of_coefficients n) (hN : N = binomial_coefficients_sum n) (condition : M - N = 240) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_n_from_sum_of_coeffs_l1461_146102


namespace NUMINAMATH_GPT_suraj_innings_l1461_146142

theorem suraj_innings (n A : ℕ) (h1 : A + 6 = 16) (h2 : (n * A + 112) / (n + 1) = 16) : n = 16 :=
by
  sorry

end NUMINAMATH_GPT_suraj_innings_l1461_146142


namespace NUMINAMATH_GPT_george_total_socks_l1461_146175

-- Define the initial number of socks George had
def initial_socks : ℝ := 28.0

-- Define the number of socks he bought
def bought_socks : ℝ := 36.0

-- Define the number of socks his Dad gave him
def given_socks : ℝ := 4.0

-- Define the number of total socks
def total_socks : ℝ := initial_socks + bought_socks + given_socks

-- State the theorem we want to prove
theorem george_total_socks : total_socks = 68.0 :=
by
  sorry

end NUMINAMATH_GPT_george_total_socks_l1461_146175


namespace NUMINAMATH_GPT_beyonce_total_songs_l1461_146173

theorem beyonce_total_songs (s a b t : ℕ) (h_s : s = 5) (h_a : a = 2 * 15) (h_b : b = 20) (h_t : t = s + a + b) : t = 55 := by
  rw [h_s, h_a, h_b] at h_t
  exact h_t

end NUMINAMATH_GPT_beyonce_total_songs_l1461_146173


namespace NUMINAMATH_GPT_shorter_side_length_l1461_146199

theorem shorter_side_length (L W : ℝ) (h₁ : L * W = 120) (h₂ : 2 * L + 2 * W = 46) : L = 8 ∨ W = 8 := 
by 
  sorry

end NUMINAMATH_GPT_shorter_side_length_l1461_146199


namespace NUMINAMATH_GPT_log_squared_sum_eq_one_l1461_146121

open Real

theorem log_squared_sum_eq_one :
  (log 2)^2 * log 250 + (log 5)^2 * log 40 = 1 := by
  sorry

end NUMINAMATH_GPT_log_squared_sum_eq_one_l1461_146121


namespace NUMINAMATH_GPT_spent_amount_l1461_146125

def initial_amount : ℕ := 15
def final_amount : ℕ := 11

theorem spent_amount : initial_amount - final_amount = 4 :=
by
  sorry

end NUMINAMATH_GPT_spent_amount_l1461_146125


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_55_l1461_146143

theorem smallest_four_digit_divisible_by_55 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 55 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 55 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_55_l1461_146143


namespace NUMINAMATH_GPT_ram_money_l1461_146193

theorem ram_money (R G K : ℝ) (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : K = 3468) :
  R = 588 := by
  sorry

end NUMINAMATH_GPT_ram_money_l1461_146193


namespace NUMINAMATH_GPT_sum_of_numbers_l1461_146139

theorem sum_of_numbers (x y z : ℝ) (h1 : x ≤ y) (h2 : y ≤ z) 
  (h3 : y = 5) (h4 : (x + y + z) / 3 = x + 10) (h5 : (x + y + z) / 3 = z - 15) : 
  x + y + z = 30 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1461_146139


namespace NUMINAMATH_GPT_leo_average_speed_last_segment_l1461_146195

theorem leo_average_speed_last_segment :
  let total_distance := 135
  let total_time_hr := 135 / 60.0
  let segment_time_hr := 45 / 60.0
  let first_segment_distance := 55 * segment_time_hr
  let second_segment_distance := 70 * segment_time_hr
  let last_segment_distance := total_distance - (first_segment_distance + second_segment_distance)
  last_segment_distance / segment_time_hr = 55 :=
by
  sorry

end NUMINAMATH_GPT_leo_average_speed_last_segment_l1461_146195


namespace NUMINAMATH_GPT_dogs_not_liking_any_l1461_146182

variables (totalDogs : ℕ) (dogsLikeWatermelon : ℕ) (dogsLikeSalmon : ℕ) (dogsLikeBothSalmonWatermelon : ℕ)
          (dogsLikeChicken : ℕ) (dogsLikeWatermelonNotSalmon : ℕ) (dogsLikeSalmonChickenNotWatermelon : ℕ)

theorem dogs_not_liking_any : totalDogs = 80 → dogsLikeWatermelon = 21 → dogsLikeSalmon = 58 →
  dogsLikeBothSalmonWatermelon = 12 → dogsLikeChicken = 15 →
  dogsLikeWatermelonNotSalmon = 7 → dogsLikeSalmonChickenNotWatermelon = 10 →
  (totalDogs - ((dogsLikeSalmon - (dogsLikeBothSalmonWatermelon + dogsLikeSalmonChickenNotWatermelon)) +
                (dogsLikeWatermelon - (dogsLikeBothSalmonWatermelon + dogsLikeWatermelonNotSalmon)) +
                (dogsLikeChicken - (dogsLikeWatermelonNotSalmon + dogsLikeSalmonChickenNotWatermelon)) +
                dogsLikeBothSalmonWatermelon + dogsLikeWatermelonNotSalmon + dogsLikeSalmonChickenNotWatermelon)) = 13 :=
by
  intros h_totalDogs h_dogsLikeWatermelon h_dogsLikeSalmon h_dogsLikeBothSalmonWatermelon 
         h_dogsLikeChicken h_dogsLikeWatermelonNotSalmon h_dogsLikeSalmonChickenNotWatermelon
  sorry

end NUMINAMATH_GPT_dogs_not_liking_any_l1461_146182


namespace NUMINAMATH_GPT_certain_number_is_213_l1461_146103

theorem certain_number_is_213 (x : ℝ) (h1 : x * 16 = 3408) (h2 : x * 1.6 = 340.8) : x = 213 :=
sorry

end NUMINAMATH_GPT_certain_number_is_213_l1461_146103


namespace NUMINAMATH_GPT_sheets_in_set_l1461_146160

-- Definitions of the conditions
def John_sheets_left (S E : ℕ) : Prop := S - E = 80
def Mary_sheets_used (S E : ℕ) : Prop := S = 4 * E

-- Theorems to prove the number of sheets
theorem sheets_in_set (S E : ℕ) (hJohn : John_sheets_left S E) (hMary : Mary_sheets_used S E) : S = 320 :=
by { 
  sorry 
}

end NUMINAMATH_GPT_sheets_in_set_l1461_146160


namespace NUMINAMATH_GPT_find_a_l1461_146152

theorem find_a : 
  (∃ (a : ℝ), ∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 4 * y = 0 → (3 * x + y + a = 0))) → a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l1461_146152


namespace NUMINAMATH_GPT_smallest_prime_less_than_square_l1461_146131

theorem smallest_prime_less_than_square : 
  ∃ (p : ℕ) (n : ℕ), p = 13 ∧ Prime p ∧ p = n^2 - 12 ∧ 0 < p ∧ ∀ q, (Prime q ∧ ∃ m, q = m^2 - 12 ∧ 0 < q  → q ≥ p) := by
  sorry

end NUMINAMATH_GPT_smallest_prime_less_than_square_l1461_146131


namespace NUMINAMATH_GPT_andy_wrong_questions_l1461_146179

theorem andy_wrong_questions (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 6) (h3 : c = 3) : a = 6 := by
  sorry

end NUMINAMATH_GPT_andy_wrong_questions_l1461_146179


namespace NUMINAMATH_GPT_abc_relationship_l1461_146161

noncomputable def a : ℝ := Real.log 5 - Real.log 3
noncomputable def b : ℝ := (2/5) * Real.exp (2/3)
noncomputable def c : ℝ := 2/3

theorem abc_relationship : b > c ∧ c > a :=
by
  sorry

end NUMINAMATH_GPT_abc_relationship_l1461_146161


namespace NUMINAMATH_GPT_salem_size_comparison_l1461_146174

theorem salem_size_comparison (S L : ℕ) (hL: L = 58940)
  (hSalem: S - 130000 = 2 * 377050) :
  (S / L = 15) :=
sorry

end NUMINAMATH_GPT_salem_size_comparison_l1461_146174


namespace NUMINAMATH_GPT_train_speed_l1461_146106

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 3500) (h_time : time = 80) : 
  length / time = 43.75 := 
by 
  sorry

end NUMINAMATH_GPT_train_speed_l1461_146106


namespace NUMINAMATH_GPT_cars_in_north_america_correct_l1461_146154

def total_cars_produced : ℕ := 6755
def cars_produced_in_europe : ℕ := 2871

def cars_produced_in_north_america : ℕ := total_cars_produced - cars_produced_in_europe

theorem cars_in_north_america_correct : cars_produced_in_north_america = 3884 :=
by sorry

end NUMINAMATH_GPT_cars_in_north_america_correct_l1461_146154


namespace NUMINAMATH_GPT_strips_overlap_area_l1461_146183

theorem strips_overlap_area :
  ∀ (length_left length_right area_only_left area_only_right : ℕ) (S : ℚ),
    length_left = 9 →
    length_right = 7 →
    area_only_left = 27 →
    area_only_right = 18 →
    (area_only_left + S) / (area_only_right + S) = 9 / 7 →
    S = 13.5 :=
by
  intros length_left length_right area_only_left area_only_right S
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_strips_overlap_area_l1461_146183


namespace NUMINAMATH_GPT_first_term_of_geometric_sequence_l1461_146156

-- Define a geometric sequence
def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r^n

-- Initialize conditions
variable (a r : ℝ)

-- Provided that the 3rd term and the 6th term
def third_term : Prop := geometric_sequence a r 2 = 5
def sixth_term : Prop := geometric_sequence a r 5 = 40

-- The theorem to prove that a == 5/4 given the conditions
theorem first_term_of_geometric_sequence : third_term a r ∧ sixth_term a r → a = 5 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_first_term_of_geometric_sequence_l1461_146156


namespace NUMINAMATH_GPT_money_per_postcard_l1461_146108

def postcards_per_day : ℕ := 30
def days : ℕ := 6
def total_earning : ℕ := 900
def total_postcards := postcards_per_day * days
def price_per_postcard := total_earning / total_postcards

theorem money_per_postcard :
  price_per_postcard = 5 := 
sorry

end NUMINAMATH_GPT_money_per_postcard_l1461_146108


namespace NUMINAMATH_GPT_son_l1461_146147

theorem son's_age (S M : ℕ) (h₁ : M = S + 25) (h₂ : M + 2 = 2 * (S + 2)) : S = 23 := by
  sorry

end NUMINAMATH_GPT_son_l1461_146147


namespace NUMINAMATH_GPT_ratio_matt_fem_4_1_l1461_146150

-- Define Fem's current age
def FemCurrentAge : ℕ := 11

-- Define the condition about the sum of their ages in two years
def AgeSumInTwoYears (MattCurrentAge : ℕ) : Prop :=
  (FemCurrentAge + 2) + (MattCurrentAge + 2) = 59

-- Define the desired ratio as a property
def DesiredRatio (MattCurrentAge : ℕ) : Prop :=
  MattCurrentAge / FemCurrentAge = 4

-- Create the theorem statement
theorem ratio_matt_fem_4_1 (M : ℕ) (h : AgeSumInTwoYears M) : DesiredRatio M :=
  sorry

end NUMINAMATH_GPT_ratio_matt_fem_4_1_l1461_146150


namespace NUMINAMATH_GPT_part1_part2_l1461_146189

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

theorem part1 (a : ℝ) (h : A ∩ B a = {2}) : a = -1 ∨ a = -3 := by
  sorry

theorem part2 (a : ℝ) (h : A ∪ B a = A) : a ≤ -3 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1461_146189


namespace NUMINAMATH_GPT_road_signs_count_l1461_146111

theorem road_signs_count (n1 n2 n3 n4 : ℕ) (h1 : n1 = 40) (h2 : n2 = n1 + n1 / 4) (h3 : n3 = 2 * n2) (h4 : n4 = n3 - 20) : 
  n1 + n2 + n3 + n4 = 270 := 
by
  sorry

end NUMINAMATH_GPT_road_signs_count_l1461_146111


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1461_146180

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ) (h₁ : -1 ≤ x ∧ x ≤ 2) : a > 4 → x^2 - a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1461_146180


namespace NUMINAMATH_GPT_condition_necessary_but_not_sufficient_l1461_146141

-- Definitions based on given conditions
variables {a b c : ℝ}

-- The condition that needs to be qualified
def condition (a b c : ℝ) := a > 0 ∧ b^2 - 4 * a * c < 0

-- The statement to be verified
def statement (a b c : ℝ) := ∀ x : ℝ, a * x^2 + b * x + c > 0

-- Prove that the condition is a necessary but not sufficient condition for the statement
theorem condition_necessary_but_not_sufficient :
  condition a b c → (¬ (condition a b c ↔ statement a b c)) :=
by
  sorry

end NUMINAMATH_GPT_condition_necessary_but_not_sufficient_l1461_146141


namespace NUMINAMATH_GPT_marked_price_is_300_max_discount_is_50_l1461_146134

-- Definition of the conditions given in the problem:
def loss_condition (x : ℝ) : Prop := 0.4 * x - 30 = 0.7 * x - 60
def profit_condition (x : ℝ) : Prop := 0.7 * x - 60 - (0.4 * x - 30) = 90

-- Statement for the first problem: Prove the marked price is 300 yuan.
theorem marked_price_is_300 : ∃ x : ℝ, loss_condition x ∧ profit_condition x ∧ x = 300 := by
  exists 300
  simp [loss_condition, profit_condition]
  sorry

noncomputable def max_discount (x : ℝ) : ℝ := 100 - (30 + 0.4 * x) / x * 100

def no_loss_max_discount (d : ℝ) : Prop := d = 50

-- Statement for the second problem: Prove the maximum discount is 50%.
theorem max_discount_is_50 (x : ℝ) (h_loss : loss_condition x) (h_profit : profit_condition x) : no_loss_max_discount (max_discount x) := by
  simp [max_discount, no_loss_max_discount]
  sorry

end NUMINAMATH_GPT_marked_price_is_300_max_discount_is_50_l1461_146134


namespace NUMINAMATH_GPT_trig_identity_l1461_146100

open Real

theorem trig_identity (α β : ℝ) (h : cos α * cos β - sin α * sin β = 0) : sin α * cos β + cos α * sin β = 1 ∨ sin α * cos β + cos α * sin β = -1 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1461_146100


namespace NUMINAMATH_GPT_contractor_job_completion_l1461_146181

theorem contractor_job_completion 
  (total_days : ℕ := 100) 
  (initial_workers : ℕ := 10) 
  (days_worked_initial : ℕ := 20) 
  (fraction_completed_initial : ℚ := 1/4) 
  (fired_workers : ℕ := 2) 
  : ∀ (remaining_days : ℕ), remaining_days = 75 → (remaining_days + days_worked_initial = 95) :=
by
  sorry

end NUMINAMATH_GPT_contractor_job_completion_l1461_146181


namespace NUMINAMATH_GPT_coprime_exists_pow_divisible_l1461_146123

theorem coprime_exists_pow_divisible (a n : ℕ) (h_coprime : Nat.gcd a n = 1) : 
  ∃ m : ℕ, n ∣ a^m - 1 :=
by
  sorry

end NUMINAMATH_GPT_coprime_exists_pow_divisible_l1461_146123


namespace NUMINAMATH_GPT_exists_sum_of_squares_form_l1461_146146

theorem exists_sum_of_squares_form (n : ℕ) (h : n % 25 = 9) :
  ∃ (a b c : ℕ), n = (a * (a + 1)) / 2 + (b * (b + 1)) / 2 + (c * (c + 1)) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_exists_sum_of_squares_form_l1461_146146


namespace NUMINAMATH_GPT_even_function_behavior_l1461_146122

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def condition (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 ≠ x2 → (x2 - x1) * (f x2 - f x1) > 0

theorem even_function_behavior (f : ℝ → ℝ) (h_even : is_even_function f) (h_condition : condition f) 
  (n : ℕ) (h_n : n > 0) : 
  f (n+1) < f (-n) ∧ f (-n) < f (n-1) :=
sorry

end NUMINAMATH_GPT_even_function_behavior_l1461_146122


namespace NUMINAMATH_GPT_changfei_class_l1461_146116

theorem changfei_class (m n : ℕ) (h : m * (m - 1) + m * n + n = 51) : m + n = 9 :=
sorry

end NUMINAMATH_GPT_changfei_class_l1461_146116


namespace NUMINAMATH_GPT_tangent_line_of_circle_l1461_146138
-- Import the required libraries

-- Define the given condition of the circle in polar coordinates
def polar_circle (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta

-- Define the property of the tangent line in polar coordinates
def tangent_line (rho theta : ℝ) : Prop :=
  rho * Real.cos theta = 4

-- State the theorem to be proven
theorem tangent_line_of_circle (rho theta : ℝ) (h : polar_circle rho theta) :
  tangent_line rho theta :=
sorry

end NUMINAMATH_GPT_tangent_line_of_circle_l1461_146138


namespace NUMINAMATH_GPT_abs_difference_l1461_146127

theorem abs_difference (a b : ℝ) (h₁ : a * b = 9) (h₂ : a + b = 10) : |a - b| = 8 :=
sorry

end NUMINAMATH_GPT_abs_difference_l1461_146127


namespace NUMINAMATH_GPT_sum_of_coefficients_l1461_146185

theorem sum_of_coefficients (a₅ a₄ a₃ a₂ a₁ a₀ : ℤ)
  (h₀ : (x - 2)^5 = a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀)
  (h₁ : a₅ + a₄ + a₃ + a₂ + a₁ + a₀ = -1)
  (h₂ : a₀ = -32) :
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1461_146185


namespace NUMINAMATH_GPT_sample_capacity_n_l1461_146186

theorem sample_capacity_n
  (n : ℕ) 
  (engineers technicians craftsmen : ℕ) 
  (total_population : ℕ)
  (stratified_interval systematic_interval : ℕ) :
  engineers = 6 →
  technicians = 12 →
  craftsmen = 18 →
  total_population = engineers + technicians + craftsmen →
  total_population = 36 →
  (∃ n : ℕ, n ∣ total_population ∧ 6 ∣ n ∧ 35 % (n + 1) = 0) →
  n = 6 :=
by
  sorry

end NUMINAMATH_GPT_sample_capacity_n_l1461_146186


namespace NUMINAMATH_GPT_arrange_logs_in_order_l1461_146165

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.8 / Real.log 1.2
noncomputable def c : ℝ := Real.sqrt 1.5

theorem arrange_logs_in_order : b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_arrange_logs_in_order_l1461_146165


namespace NUMINAMATH_GPT_distinct_real_roots_c_l1461_146137

theorem distinct_real_roots_c (c : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ + c = 0 ∧ x₂^2 - 4*x₂ + c = 0) ↔ c < 4 := by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_c_l1461_146137


namespace NUMINAMATH_GPT_solution_set_inequality_l1461_146113

noncomputable def solution_set := {x : ℝ | (x + 1) * (x - 2) ≤ 0 ∧ x ≠ -1}

theorem solution_set_inequality :
  solution_set = {x : ℝ | -1 < x ∧ x ≤ 2} :=
by {
-- Insert proof here
sorry
}

end NUMINAMATH_GPT_solution_set_inequality_l1461_146113


namespace NUMINAMATH_GPT_train_speed_is_correct_l1461_146196

noncomputable def speed_of_train (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed_is_correct :
  speed_of_train 200 19.99840012798976 = 36.00287976960864 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_correct_l1461_146196


namespace NUMINAMATH_GPT_abs_sum_less_b_l1461_146132

theorem abs_sum_less_b (x : ℝ) (b : ℝ) (h : |2 * x - 8| + |2 * x - 6| < b) (hb : b > 0) : b > 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_sum_less_b_l1461_146132


namespace NUMINAMATH_GPT_cos_sub_eq_five_over_eight_l1461_146198

theorem cos_sub_eq_five_over_eight (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5 / 8 := 
by sorry

end NUMINAMATH_GPT_cos_sub_eq_five_over_eight_l1461_146198


namespace NUMINAMATH_GPT_five_coins_total_cannot_be_30_cents_l1461_146114

theorem five_coins_total_cannot_be_30_cents :
  ¬ ∃ (a b c d e : ℕ), 
  a + b + c + d + e = 5 ∧ 
  (a * 1 + b * 5 + c * 10 + d * 25 + e * 50) = 30 := 
sorry

end NUMINAMATH_GPT_five_coins_total_cannot_be_30_cents_l1461_146114


namespace NUMINAMATH_GPT_roberts_test_score_l1461_146109

structure ClassState where
  num_students : ℕ
  avg_19_students : ℕ
  class_avg_20_students : ℕ

def calculate_roberts_score (s : ClassState) : ℕ :=
  let total_19_students := s.num_students * s.avg_19_students
  let total_20_students := (s.num_students + 1) * s.class_avg_20_students
  total_20_students - total_19_students

theorem roberts_test_score 
  (state : ClassState) 
  (h1 : state.num_students = 19) 
  (h2 : state.avg_19_students = 74)
  (h3 : state.class_avg_20_students = 75) : 
  calculate_roberts_score state = 94 := by
  sorry

end NUMINAMATH_GPT_roberts_test_score_l1461_146109


namespace NUMINAMATH_GPT_pizza_slices_with_both_toppings_l1461_146159

theorem pizza_slices_with_both_toppings (total_slices ham_slices pineapple_slices slices_with_both : ℕ)
  (h_total: total_slices = 15)
  (h_ham: ham_slices = 8)
  (h_pineapple: pineapple_slices = 12)
  (h_slices_with_both: slices_with_both + (ham_slices - slices_with_both) + (pineapple_slices - slices_with_both) = total_slices)
  : slices_with_both = 5 :=
by
  -- the proof would go here, but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_pizza_slices_with_both_toppings_l1461_146159


namespace NUMINAMATH_GPT_reciprocal_opposite_neg_two_thirds_l1461_146110

noncomputable def opposite (a : ℚ) : ℚ := -a
noncomputable def reciprocal (a : ℚ) : ℚ := 1 / a

theorem reciprocal_opposite_neg_two_thirds : reciprocal (opposite (-2 / 3)) = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_reciprocal_opposite_neg_two_thirds_l1461_146110


namespace NUMINAMATH_GPT_isosceles_triangle_bisector_properties_l1461_146157

theorem isosceles_triangle_bisector_properties:
  ∀ (T : Type) (triangle : T)
  (is_isosceles : Prop) (vertex_angle_bisector_bisects_base : Prop) (vertex_angle_bisector_perpendicular_to_base : Prop),
  is_isosceles 
  → (vertex_angle_bisector_bisects_base ∧ vertex_angle_bisector_perpendicular_to_base) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_bisector_properties_l1461_146157


namespace NUMINAMATH_GPT_complement_intersection_l1461_146170

-- Define sets P and Q.
def P : Set ℝ := {x | x ≥ 2}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Define the complement of P.
def complement_P : Set ℝ := {x | x < 2}

-- The theorem we need to prove.
theorem complement_intersection : complement_P ∩ Q = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1461_146170


namespace NUMINAMATH_GPT_find_other_endpoint_l1461_146107

theorem find_other_endpoint (x_m y_m : ℤ) (x1 y1 : ℤ) 
(m_cond : x_m = (x1 + (-1)) / 2) (m_cond' : y_m = (y1 + (-4)) / 2) : 
(x_m, y_m) = (3, -1) ∧ (x1, y1) = (7, 2) → (-1, -4) = (-1, -4) :=
by
  sorry

end NUMINAMATH_GPT_find_other_endpoint_l1461_146107


namespace NUMINAMATH_GPT_isosceles_triangle_length_l1461_146188

theorem isosceles_triangle_length (a : ℝ) (h_graph_A : ∃ y, (a, y) ∈ {p : ℝ × ℝ | p.snd = -p.fst^2})
  (h_graph_B : ∃ y, (-a, y) ∈ {p : ℝ × ℝ | p.snd = -p.fst^2}) 
  (h_isosceles : ∃ O : ℝ × ℝ, O = (0, 0) ∧ 
    dist (a, -a^2) O = dist (-a, -a^2) O ∧ dist (a, -a^2) (-a, -a^2) = dist (-a, -a^2) O) :
  dist (a, -a^2) (0, 0) = 2 * Real.sqrt 3 := sorry

end NUMINAMATH_GPT_isosceles_triangle_length_l1461_146188


namespace NUMINAMATH_GPT_min_value_expression_l1461_146119

open Real

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_condition : a * b * c = 1) :
  a^2 + 8 * a * b + 32 * b^2 + 24 * b * c + 8 * c^2 ≥ 36 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1461_146119


namespace NUMINAMATH_GPT_totalPeaches_l1461_146190

-- Define the number of red, yellow, and green peaches
def redPeaches := 7
def yellowPeaches := 15
def greenPeaches := 8

-- Define the total number of peaches and the proof statement
theorem totalPeaches : redPeaches + yellowPeaches + greenPeaches = 30 := by
  sorry

end NUMINAMATH_GPT_totalPeaches_l1461_146190


namespace NUMINAMATH_GPT_difference_between_numbers_l1461_146171

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 27630) (h2 : a = 5 * b + 5) : a - b = 18421 :=
  sorry

end NUMINAMATH_GPT_difference_between_numbers_l1461_146171


namespace NUMINAMATH_GPT_smallest_number_of_ducks_l1461_146151

theorem smallest_number_of_ducks (n_ducks n_cranes : ℕ) (h1 : n_ducks = n_cranes) : 
  ∃ n, n_ducks = n ∧ n_cranes = n ∧ n = Nat.lcm 13 17 := by
  use 221
  sorry

end NUMINAMATH_GPT_smallest_number_of_ducks_l1461_146151
