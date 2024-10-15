import Mathlib

namespace NUMINAMATH_GPT_value_of_a_l1307_130717

/--
Given that x = 3 is a solution to the equation 3x - 2a = 5,
prove that a = 2.
-/
theorem value_of_a (x a : ℤ) (h : 3 * x - 2 * a = 5) (hx : x = 3) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1307_130717


namespace NUMINAMATH_GPT_nitrogen_mass_percentage_in_ammonium_phosphate_l1307_130738

def nitrogen_mass_percentage
  (molar_mass_N : ℚ)
  (molar_mass_H : ℚ)
  (molar_mass_P : ℚ)
  (molar_mass_O : ℚ)
  : ℚ :=
  let molar_mass_NH4 := molar_mass_N + 4 * molar_mass_H
  let molar_mass_PO4 := molar_mass_P + 4 * molar_mass_O
  let molar_mass_NH4_3_PO4 := 3 * molar_mass_NH4 + molar_mass_PO4
  let mass_N_in_NH4_3_PO4 := 3 * molar_mass_N
  (mass_N_in_NH4_3_PO4 / molar_mass_NH4_3_PO4) * 100

theorem nitrogen_mass_percentage_in_ammonium_phosphate
  (molar_mass_N : ℚ := 14.01)
  (molar_mass_H : ℚ := 1.01)
  (molar_mass_P : ℚ := 30.97)
  (molar_mass_O : ℚ := 16.00)
  : nitrogen_mass_percentage molar_mass_N molar_mass_H molar_mass_P molar_mass_O = 28.19 :=
by
  sorry

end NUMINAMATH_GPT_nitrogen_mass_percentage_in_ammonium_phosphate_l1307_130738


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1307_130718

open Classical

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a > 1 ∧ b > 3) → (a + b > 4) ∧ ¬((a + b > 4) → (a > 1 ∧ b > 3)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1307_130718


namespace NUMINAMATH_GPT_expression_value_l1307_130745

noncomputable def givenExpression : ℝ :=
  -2^2 + Real.sqrt 8 - 3 + 1/3

theorem expression_value : givenExpression = -20/3 + 2 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l1307_130745


namespace NUMINAMATH_GPT_sum_of_perimeters_correct_l1307_130768

noncomputable def sum_of_perimeters (s w : ℝ) : ℝ :=
  let l := 2 * w
  let square_area := s^2
  let rectangle_area := l * w
  let sq_perimeter := 4 * s
  let rect_perimeter := 2 * l + 2 * w
  sq_perimeter + rect_perimeter

theorem sum_of_perimeters_correct (s w : ℝ) (h1 : s^2 + 2 * w^2 = 130) (h2 : s^2 - 2 * w^2 = 50) :
  sum_of_perimeters s w = 12 * Real.sqrt 10 + 12 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_GPT_sum_of_perimeters_correct_l1307_130768


namespace NUMINAMATH_GPT_largest_possible_value_of_s_l1307_130759

theorem largest_possible_value_of_s (p q r s : ℝ)
  (h₁ : p + q + r + s = 12)
  (h₂ : pq + pr + ps + qr + qs + rs = 24) : 
  s ≤ 3 + 3 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_largest_possible_value_of_s_l1307_130759


namespace NUMINAMATH_GPT_hyperbola_center_coordinates_l1307_130729

-- Defining the equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  (3 * y + 6)^2 / 16 - (2 * x - 1)^2 / 9 = 1

-- Stating the theorem to verify the center of the hyperbola
theorem hyperbola_center_coordinates :
  ∃ (h k : ℝ), (h = 1/2) ∧ (k = -2) ∧ 
    ∀ x y, hyperbola_eq x y ↔ ((y + 2)^2 / (4 / 3)^2 - (x - 1/2)^2 / (3 / 2)^2 = 1) :=
by sorry

end NUMINAMATH_GPT_hyperbola_center_coordinates_l1307_130729


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l1307_130774

variable (α : ℝ)

theorem trigonometric_identity_proof :
  3 + 4 * (Real.sin (4 * α + (3 / 2) * Real.pi)) +
  Real.sin (8 * α + (5 / 2) * Real.pi) = 
  8 * (Real.sin (2 * α))^4 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l1307_130774


namespace NUMINAMATH_GPT_value_of_a_squared_plus_b_squared_l1307_130721

variable (a b : ℝ)

theorem value_of_a_squared_plus_b_squared (h1 : a - b = 10) (h2 : a * b = 55) : a^2 + b^2 = 210 := 
by 
sorry

end NUMINAMATH_GPT_value_of_a_squared_plus_b_squared_l1307_130721


namespace NUMINAMATH_GPT_lana_extra_flowers_l1307_130770

theorem lana_extra_flowers :
  ∀ (tulips roses used total_extra : ℕ),
    tulips = 36 →
    roses = 37 →
    used = 70 →
    total_extra = (tulips + roses - used) →
    total_extra = 3 :=
by
  intros tulips roses used total_extra ht hr hu hte
  rw [ht, hr, hu] at hte
  sorry

end NUMINAMATH_GPT_lana_extra_flowers_l1307_130770


namespace NUMINAMATH_GPT_inequality_solution_l1307_130788

theorem inequality_solution (x : ℝ) : 
  x^3 - 10 * x^2 + 28 * x > 0 ↔ (0 < x ∧ x < 4) ∨ (6 < x)
:= sorry

end NUMINAMATH_GPT_inequality_solution_l1307_130788


namespace NUMINAMATH_GPT_rectangular_solid_surface_area_l1307_130748

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (h_volume : a * b * c = 1001) :
  2 * (a * b + b * c + c * a) = 622 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_solid_surface_area_l1307_130748


namespace NUMINAMATH_GPT_worker_late_by_10_minutes_l1307_130790

def usual_time : ℕ := 40
def speed_ratio : ℚ := 4 / 5
def time_new := (usual_time : ℚ) * (5 / 4) -- This is the equation derived from solving

theorem worker_late_by_10_minutes : 
  ((time_new : ℚ) - usual_time) = 10 :=
by
  sorry -- proof is skipped

end NUMINAMATH_GPT_worker_late_by_10_minutes_l1307_130790


namespace NUMINAMATH_GPT_problem_l1307_130735

noncomputable def fx (a b c : ℝ) (x : ℝ) : ℝ := a * x + b / x + c

theorem problem 
  (a b c : ℝ) 
  (h_odd : ∀ x, fx a b c x = -fx a b c (-x))
  (h_f1 : fx a b c 1 = 5 / 2)
  (h_f2 : fx a b c 2 = 17 / 4) :
  (a = 2) ∧ (b = 1 / 2) ∧ (c = 0) ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 / 2 → fx a b c x₁ > fx a b c x₂) := 
sorry

end NUMINAMATH_GPT_problem_l1307_130735


namespace NUMINAMATH_GPT_geometric_sum_l1307_130715

def S10 : ℕ := 36
def S20 : ℕ := 48

theorem geometric_sum (S30 : ℕ) (h1 : S10 = 36) (h2 : S20 = 48) : S30 = 52 :=
by
  have h3 : (S20 - S10) ^ 2 = S10 * (S30 - S20) :=
    sorry -- This is based on the properties of the geometric sequence
  sorry  -- Solve the equation to show S30 = 52

end NUMINAMATH_GPT_geometric_sum_l1307_130715


namespace NUMINAMATH_GPT_jacob_hours_l1307_130737

theorem jacob_hours (J : ℕ) (H1 : ∃ (G P : ℕ),
    G = J - 6 ∧
    P = 2 * G - 4 ∧
    J + G + P = 50) : J = 18 :=
by
  sorry

end NUMINAMATH_GPT_jacob_hours_l1307_130737


namespace NUMINAMATH_GPT_man_born_in_1936_l1307_130756

noncomputable def year_of_birth (x : ℕ) : ℕ :=
  x^2 - 2 * x

theorem man_born_in_1936 :
  ∃ x : ℕ, x < 50 ∧ year_of_birth x < 1950 ∧ year_of_birth x = 1892 :=
by
  sorry

end NUMINAMATH_GPT_man_born_in_1936_l1307_130756


namespace NUMINAMATH_GPT_smallest_number_jungkook_l1307_130742

theorem smallest_number_jungkook (jungkook yoongi yuna : ℕ) 
  (hj : jungkook = 6 - 3) (hy : yoongi = 4) (hu : yuna = 5) : 
  jungkook < yoongi ∧ jungkook < yuna :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_jungkook_l1307_130742


namespace NUMINAMATH_GPT_find_constant_k_l1307_130796

theorem find_constant_k (k : ℝ) :
    -x^2 - (k + 9) * x - 8 = -(x - 2) * (x - 4) → k = -15 := by
  sorry

end NUMINAMATH_GPT_find_constant_k_l1307_130796


namespace NUMINAMATH_GPT_sum_eq_two_l1307_130743

theorem sum_eq_two (x y : ℝ) (h : x^2 + y^2 = 10 * x - 6 * y - 34) : x + y = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_eq_two_l1307_130743


namespace NUMINAMATH_GPT_annual_interest_rate_is_6_percent_l1307_130736

-- Definitions from the conditions
def principal : ℕ := 150
def total_amount_paid : ℕ := 159
def interest := total_amount_paid - principal
def interest_rate := (interest * 100) / principal

-- The theorem to prove
theorem annual_interest_rate_is_6_percent :
  interest_rate = 6 := by sorry

end NUMINAMATH_GPT_annual_interest_rate_is_6_percent_l1307_130736


namespace NUMINAMATH_GPT_winter_melon_ratio_l1307_130789

theorem winter_melon_ratio (T Ok_sales Choc_sales : ℕ) (hT : T = 50) 
  (hOk : Ok_sales = 3 * T / 10) (hChoc : Choc_sales = 15) :
  (T - (Ok_sales + Choc_sales)) / T = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_winter_melon_ratio_l1307_130789


namespace NUMINAMATH_GPT_jason_picked_7_pears_l1307_130740

def pears_picked_by_jason (total_pears mike_pears : ℕ) : ℕ :=
  total_pears - mike_pears

theorem jason_picked_7_pears :
  pears_picked_by_jason 15 8 = 7 :=
by
  -- Proof is required but we can insert sorry here to skip it for now
  sorry

end NUMINAMATH_GPT_jason_picked_7_pears_l1307_130740


namespace NUMINAMATH_GPT_number_of_dimes_l1307_130755

theorem number_of_dimes (k : ℕ) (dimes quarters : ℕ) (value : ℕ)
  (h1 : 3 * k = dimes)
  (h2 : 2 * k = quarters)
  (h3 : value = (10 * dimes) + (25 * quarters))
  (h4 : value = 400) :
  dimes = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_dimes_l1307_130755


namespace NUMINAMATH_GPT_calculate_total_cost_l1307_130747

noncomputable def sandwich_cost : ℕ := 4
noncomputable def soda_cost : ℕ := 3
noncomputable def num_sandwiches : ℕ := 7
noncomputable def num_sodas : ℕ := 8
noncomputable def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem calculate_total_cost : total_cost = 52 := by
  sorry

end NUMINAMATH_GPT_calculate_total_cost_l1307_130747


namespace NUMINAMATH_GPT_value_of_a_l1307_130728

theorem value_of_a {a : ℝ} (h : ∀ x y : ℝ, (a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) → x = y) : a = 0 ∨ a = 1 := 
  sorry

end NUMINAMATH_GPT_value_of_a_l1307_130728


namespace NUMINAMATH_GPT_possible_values_of_m_l1307_130775

theorem possible_values_of_m (m : ℝ) (h1 : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_m_l1307_130775


namespace NUMINAMATH_GPT_total_visitors_count_l1307_130781

def initial_morning_visitors : ℕ := 500
def noon_departures : ℕ := 119
def additional_afternoon_arrivals : ℕ := 138

def afternoon_arrivals : ℕ := noon_departures + additional_afternoon_arrivals
def total_visitors : ℕ := initial_morning_visitors + afternoon_arrivals

theorem total_visitors_count : total_visitors = 757 := 
by sorry

end NUMINAMATH_GPT_total_visitors_count_l1307_130781


namespace NUMINAMATH_GPT_conditions_for_unique_solution_l1307_130703

noncomputable def is_solution (n p x y z : ℕ) : Prop :=
x + p * y = n ∧ x + y = p^z

def unique_positive_integer_solution (n p : ℕ) : Prop :=
∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ is_solution n p x y z

theorem conditions_for_unique_solution {n p : ℕ} :
  (1 < p) ∧ ((n - 1) % (p - 1) = 0) ∧ ∀ k : ℕ, n ≠ p^k ↔ unique_positive_integer_solution n p :=
sorry

end NUMINAMATH_GPT_conditions_for_unique_solution_l1307_130703


namespace NUMINAMATH_GPT_correct_statements_l1307_130776

/-- The line (3+m)x+4y-3+3m=0 (m ∈ ℝ) always passes through the fixed point (-3, 3) -/
def statement1 (m : ℝ) : Prop :=
  ∀ x y : ℝ, (3 + m) * x + 4 * y - 3 + 3 * m = 0 → (x = -3 ∧ y = 3)

/-- For segment AB with endpoint B at (3,4) and A moving on the circle x²+y²=4,
    the trajectory equation of the midpoint M of segment AB is (x - 3/2)²+(y - 2)²=1 -/
def statement2 : Prop :=
  ∀ x y x1 y1 : ℝ, ((x1, y1) : ℝ × ℝ) ∈ {p | p.1^2 + p.2^2 = 4} → x = (x1 + 3) / 2 → y = (y1 + 4) / 2 → 
    (x - 3 / 2)^2 + (y - 2)^2 = 1

/-- Given M = {(x, y) | y = √(1 - x²)} and N = {(x, y) | y = x + b},
    if M ∩ N ≠ ∅, then b ∈ [-√2, √2] -/
def statement3 (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = Real.sqrt (1 - x^2) ∧ y = x + b → b ∈ [-Real.sqrt 2, Real.sqrt 2]

/-- Given the circle C: (x - b)² + (y - c)² = a² (a > 0, b > 0, c > 0) intersects the x-axis and is
    separate from the y-axis, then the intersection point of the line ax + by + c = 0 and the line
    x + y + 1 = 0 is in the second quadrant -/
def statement4 (a b c : ℝ) : Prop :=
  a > 0 → b > 0 → c > 0 → b > a → a > c →
  ∃ x y : ℝ, (a * x + b * y + c = 0 ∧ x + y + 1 = 0) ∧ x < 0 ∧ y > 0

/-- Among the statements, the correct ones are 1, 2, and 4 -/
theorem correct_statements : 
  (∀ m : ℝ, statement1 m) ∧ statement2 ∧ (∀ b : ℝ, ¬ statement3 b) ∧ 
  (∀ a b c : ℝ, statement4 a b c) :=
by sorry

end NUMINAMATH_GPT_correct_statements_l1307_130776


namespace NUMINAMATH_GPT_find_sum_A_B_l1307_130760

-- Definitions based on conditions
def A : ℤ := -3 - (-5)
def B : ℤ := 2 + (-2)

-- Theorem statement matching the problem
theorem find_sum_A_B : A + B = 2 :=
sorry

end NUMINAMATH_GPT_find_sum_A_B_l1307_130760


namespace NUMINAMATH_GPT_range_of_a_if_intersection_empty_range_of_a_if_union_equal_B_l1307_130780

-- Definitions for the sets A and B
def setA (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < a + 1}
def setB : Set ℝ := {x : ℝ | x < -1 ∨ x > 2}

-- Question (1): Proof statement for A ∩ B = ∅ implying 0 ≤ a ≤ 1
theorem range_of_a_if_intersection_empty (a : ℝ) :
  (setA a ∩ setB = ∅) → (0 ≤ a ∧ a ≤ 1) := 
sorry

-- Question (2): Proof statement for A ∪ B = B implying a ≤ -2 or a ≥ 3
theorem range_of_a_if_union_equal_B (a : ℝ) :
  (setA a ∪ setB = setB) → (a ≤ -2 ∨ 3 ≤ a) := 
sorry

end NUMINAMATH_GPT_range_of_a_if_intersection_empty_range_of_a_if_union_equal_B_l1307_130780


namespace NUMINAMATH_GPT_ducks_in_garden_l1307_130719

theorem ducks_in_garden (num_rabbits : ℕ) (num_ducks : ℕ) 
  (total_legs : ℕ)
  (rabbit_legs : ℕ) (duck_legs : ℕ) 
  (H1 : num_rabbits = 9)
  (H2 : rabbit_legs = 4)
  (H3 : duck_legs = 2)
  (H4 : total_legs = 48)
  (H5 : num_rabbits * rabbit_legs + num_ducks * duck_legs = total_legs) :
  num_ducks = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_ducks_in_garden_l1307_130719


namespace NUMINAMATH_GPT_find_general_term_a_l1307_130730

-- Define the sequence and conditions
noncomputable def S (n : ℕ) : ℚ :=
  if n = 0 then 0 else (n - 1) / (n * (n + 1))

-- General term to prove
def a (n : ℕ) : ℚ := 1 / (2^n) - 1 / (n * (n + 1))

theorem find_general_term_a :
  ∀ n : ℕ, n > 0 → S n + a n = (n - 1) / (n * (n + 1)) :=
by
  intro n hn
  sorry -- Proof omitted

end NUMINAMATH_GPT_find_general_term_a_l1307_130730


namespace NUMINAMATH_GPT_emptying_rate_l1307_130757

theorem emptying_rate (fill_time1 : ℝ) (total_fill_time : ℝ) (T : ℝ) 
  (h1 : fill_time1 = 4) 
  (h2 : total_fill_time = 20) 
  (h3 : 1 / fill_time1 - 1 / T = 1 / total_fill_time) :
  T = 5 :=
by
  sorry

end NUMINAMATH_GPT_emptying_rate_l1307_130757


namespace NUMINAMATH_GPT_avg_weight_b_c_l1307_130702

theorem avg_weight_b_c
  (a b c : ℝ)
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : b = 31) :
  (b + c) / 2 = 43 := 
by {
  sorry
}

end NUMINAMATH_GPT_avg_weight_b_c_l1307_130702


namespace NUMINAMATH_GPT_simplify_power_expression_l1307_130720

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^5 = 243 * x^20 :=
by
  sorry

end NUMINAMATH_GPT_simplify_power_expression_l1307_130720


namespace NUMINAMATH_GPT_total_cost_l1307_130792

-- Define conditions as variables
def n_b : ℕ := 3    -- number of bedroom doors
def n_o : ℕ := 2    -- number of outside doors
def c_o : ℕ := 20   -- cost per outside door
def c_b : ℕ := c_o / 2  -- cost per bedroom door

-- Define the total cost using the conditions
def c_total : ℕ := (n_o * c_o) + (n_b * c_b)

-- State the theorem to be proven
theorem total_cost :
  c_total = 70 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l1307_130792


namespace NUMINAMATH_GPT_proof_ac_plus_bd_l1307_130723

theorem proof_ac_plus_bd (a b c d : ℝ)
  (h1 : a + b + c = 10)
  (h2 : a + b + d = -6)
  (h3 : a + c + d = 0)
  (h4 : b + c + d = 15) :
  ac + bd = -130.111 := 
by
  sorry

end NUMINAMATH_GPT_proof_ac_plus_bd_l1307_130723


namespace NUMINAMATH_GPT_expression_value_zero_l1307_130771

theorem expression_value_zero (a b c : ℝ) (h : a^2 + b = b^2 + c ∧ b^2 + c = c^2 + a) : 
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 := by
  sorry

end NUMINAMATH_GPT_expression_value_zero_l1307_130771


namespace NUMINAMATH_GPT_pow_mod_eq_l1307_130701

theorem pow_mod_eq :
  (13 ^ 7) % 11 = 7 :=
by
  sorry

end NUMINAMATH_GPT_pow_mod_eq_l1307_130701


namespace NUMINAMATH_GPT_positive_integer_solutions_l1307_130766

theorem positive_integer_solutions
  (x : ℤ) :
  (5 + 3 * x < 13) ∧ ((x + 2) / 3 - (x - 1) / 2 <= 2) →
  (x = 1 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_l1307_130766


namespace NUMINAMATH_GPT_perpendicular_lines_a_value_l1307_130785

theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ m1 m2 : ℝ, (m1 = -a / 2 ∧ m2 = -1 / (a * (a + 1)) ∧ m1 * m2 = -1) ∨
   (a = 0 ∧ ax + 2 * y + 6 = 0 ∧ x + a * (a + 1) * y + (a^2 - 1) = 0)) →
  (a = -3 / 2 ∨ a = 0) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_a_value_l1307_130785


namespace NUMINAMATH_GPT_retirement_savings_l1307_130714

/-- Define the initial deposit amount -/
def P : ℕ := 800000

/-- Define the annual interest rate as a rational number -/
def r : ℚ := 7/100

/-- Define the number of years the money is invested for -/
def t : ℕ := 15

/-- Simple interest formula to calculate the accumulated amount -/
noncomputable def A : ℚ := P * (1 + r * t)

theorem retirement_savings :
  A = 1640000 := 
by
  sorry

end NUMINAMATH_GPT_retirement_savings_l1307_130714


namespace NUMINAMATH_GPT_figure_50_unit_squares_l1307_130799

-- Definitions reflecting the conditions from step A
def f (n : ℕ) := (1/2 : ℚ) * n^3 + (7/2 : ℚ) * n + 1

theorem figure_50_unit_squares : f 50 = 62676 := by
  sorry

end NUMINAMATH_GPT_figure_50_unit_squares_l1307_130799


namespace NUMINAMATH_GPT_continuity_at_4_l1307_130793

def f (x : ℝ) : ℝ := -2 * x^2 + 9

theorem continuity_at_4 : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 4| < δ → |f x + 23| < ε := by
  sorry

end NUMINAMATH_GPT_continuity_at_4_l1307_130793


namespace NUMINAMATH_GPT_arithmetic_sequence_50th_term_l1307_130711

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 5
  let n := 50
  let a_n := a_1 + (n - 1) * d
  a_n = 248 :=
by
  let a_1 := 3
  let d := 5
  let n := 50
  let a_n := a_1 + (n - 1) * d
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_50th_term_l1307_130711


namespace NUMINAMATH_GPT_log_diff_l1307_130786

theorem log_diff : (Real.log (12:ℝ) / Real.log (2:ℝ)) - (Real.log (3:ℝ) / Real.log (2:ℝ)) = 2 := 
by
  sorry

end NUMINAMATH_GPT_log_diff_l1307_130786


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1307_130706

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) : ((x - 2) / (x - 1)) / ((x + 1) - (3 / (x - 1))) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1307_130706


namespace NUMINAMATH_GPT_x_eq_3_minus_2t_and_y_eq_3t_plus_6_l1307_130758

theorem x_eq_3_minus_2t_and_y_eq_3t_plus_6 (t : ℝ) (x : ℝ) (y : ℝ) : x = 3 - 2 * t → y = 3 * t + 6 → x = 0 → y = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_x_eq_3_minus_2t_and_y_eq_3t_plus_6_l1307_130758


namespace NUMINAMATH_GPT_jean_business_hours_l1307_130712

-- Definitions of the conditions
def weekday_hours : ℕ := 10 - 16 -- from 4 pm to 10 pm
def weekend_hours : ℕ := 10 - 18 -- from 6 pm to 10 pm
def weekdays : ℕ := 5 -- Monday through Friday
def weekends : ℕ := 2 -- Saturday and Sunday

-- Total weekly hours
def total_weekly_hours : ℕ :=
  (weekday_hours * weekdays) + (weekend_hours * weekends)

-- Proof statement
theorem jean_business_hours : total_weekly_hours = 38 :=
by
  sorry

end NUMINAMATH_GPT_jean_business_hours_l1307_130712


namespace NUMINAMATH_GPT_greatest_a_no_integral_solution_l1307_130779

theorem greatest_a_no_integral_solution (a : ℤ) :
  (∀ x : ℤ, |x + 1| ≥ a - 3 / 2) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_greatest_a_no_integral_solution_l1307_130779


namespace NUMINAMATH_GPT_rob_has_24_cards_l1307_130750

theorem rob_has_24_cards 
  (r : ℕ) -- total number of baseball cards Rob has
  (dr : ℕ) -- number of doubles Rob has
  (hj: dr = 1 / 3 * r) -- one third of Rob's cards are doubles
  (jess_doubles : ℕ) -- number of doubles Jess has
  (hj_mult : jess_doubles = 5 * dr) -- Jess has 5 times as many doubles as Rob
  (jess_doubles_40 : jess_doubles = 40) -- Jess has 40 doubles baseball cards
: r = 24 :=
by
  sorry

end NUMINAMATH_GPT_rob_has_24_cards_l1307_130750


namespace NUMINAMATH_GPT_light_flashes_in_three_quarters_hour_l1307_130764

theorem light_flashes_in_three_quarters_hour (flash_interval seconds_in_three_quarters_hour : ℕ) 
  (h1 : flash_interval = 15) (h2 : seconds_in_three_quarters_hour = 2700) : 
  (seconds_in_three_quarters_hour / flash_interval = 180) :=
by
  sorry

end NUMINAMATH_GPT_light_flashes_in_three_quarters_hour_l1307_130764


namespace NUMINAMATH_GPT_find_m_prove_inequality_l1307_130710

-- Using noncomputable to handle real numbers where needed
noncomputable def f (x m : ℝ) := m - |x - 1|

-- First proof: Find m given conditions on f(x)
theorem find_m (m : ℝ) :
  (∀ x, f (x + 2) m + f (x - 2) m ≥ 0 ↔ -2 ≤ x ∧ x ≤ 4) → m = 3 :=
sorry

-- Second proof: Prove the inequality given m = 3
theorem prove_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / a + 1 / (2 * b) + 1 / (3 * c) = 3) → a + 2 * b + 3 * c ≥ 3 :=
sorry

end NUMINAMATH_GPT_find_m_prove_inequality_l1307_130710


namespace NUMINAMATH_GPT_unit_digit_seven_power_500_l1307_130787

def unit_digit (x : ℕ) : ℕ := x % 10

theorem unit_digit_seven_power_500 :
  unit_digit (7 ^ 500) = 1 := 
by
  sorry

end NUMINAMATH_GPT_unit_digit_seven_power_500_l1307_130787


namespace NUMINAMATH_GPT_pow_addition_l1307_130761

theorem pow_addition : (-2)^2 + 2^2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_pow_addition_l1307_130761


namespace NUMINAMATH_GPT_min_radius_cylinder_proof_l1307_130705

-- Defining the radius of the hemisphere
def radius_hemisphere : ℝ := 10

-- Defining the angle alpha which is less than or equal to 30 degrees
def angle_alpha_leq_30 (α : ℝ) : Prop := α ≤ 30 * Real.pi / 180

-- Minimum radius of the cylinder given alpha <= 30 degrees
noncomputable def min_radius_cylinder : ℝ :=
  10 * (2 / Real.sqrt 3 - 1)

theorem min_radius_cylinder_proof (α : ℝ) (hα : angle_alpha_leq_30 α) :
  min_radius_cylinder = 10 * (2 / Real.sqrt 3 - 1) :=
by
  -- Here would go the detailed proof steps
  sorry

end NUMINAMATH_GPT_min_radius_cylinder_proof_l1307_130705


namespace NUMINAMATH_GPT_xinxin_nights_at_seaside_l1307_130753

-- Definitions from conditions
def arrival_day : ℕ := 30
def may_days : ℕ := 31
def departure_day : ℕ := 4
def nights_spent : ℕ := (departure_day + (may_days - arrival_day))

-- Theorem to prove the number of nights spent
theorem xinxin_nights_at_seaside : nights_spent = 5 := 
by
  -- Include proof steps here in actual Lean proof
  sorry

end NUMINAMATH_GPT_xinxin_nights_at_seaside_l1307_130753


namespace NUMINAMATH_GPT_sum_of_areas_l1307_130769

theorem sum_of_areas (r s t : ℝ)
  (h1 : r + s = 13)
  (h2 : s + t = 5)
  (h3 : r + t = 12)
  (h4 : t = r / 2) : 
  π * (r ^ 2 + s ^ 2 + t ^ 2) = 105 * π := 
by
  sorry

end NUMINAMATH_GPT_sum_of_areas_l1307_130769


namespace NUMINAMATH_GPT_stone_radius_l1307_130726

theorem stone_radius (hole_diameter hole_depth : ℝ) (r : ℝ) :
  hole_diameter = 30 → hole_depth = 10 → (r - 10)^2 + 15^2 = r^2 → r = 16.25 :=
by
  intros h_diam h_depth hyp_eq
  sorry

end NUMINAMATH_GPT_stone_radius_l1307_130726


namespace NUMINAMATH_GPT_ratio_of_surface_areas_l1307_130791

theorem ratio_of_surface_areas {r R : ℝ} 
  (h : (4/3) * Real.pi * r^3 / ((4/3) * Real.pi * R^3) = 1 / 8) :
  (4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 1 / 4 := 
sorry

end NUMINAMATH_GPT_ratio_of_surface_areas_l1307_130791


namespace NUMINAMATH_GPT_magnitude_difference_l1307_130795

open Complex

noncomputable def c1 : ℂ := 18 - 5 * I
noncomputable def c2 : ℂ := 14 + 6 * I
noncomputable def c3 : ℂ := 3 - 12 * I
noncomputable def c4 : ℂ := 4 + 9 * I

theorem magnitude_difference : 
  Complex.abs ((c1 * c2) - (c3 * c4)) = Real.sqrt 146365 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_difference_l1307_130795


namespace NUMINAMATH_GPT_vasya_birthday_was_thursday_l1307_130731

variable (today : String)
variable (tomorrow : String)
variable (day_after_tomorrow : String)
variable (birthday : String)

-- Conditions given in the problem
axiom birthday_not_sunday : birthday ≠ "Sunday"
axiom sunday_day_after_tomorrow : day_after_tomorrow = "Sunday"

-- From the conditions, we need to prove that Vasya's birthday was on Thursday.
theorem vasya_birthday_was_thursday : birthday = "Thursday" :=
by
  -- Fill in the proof here
  sorry

end NUMINAMATH_GPT_vasya_birthday_was_thursday_l1307_130731


namespace NUMINAMATH_GPT_find_f_inv_8_l1307_130746

variable (f : ℝ → ℝ)

-- Given conditions
axiom h1 : f 5 = 1
axiom h2 : ∀ x, f (2 * x) = 2 * f x

-- Theorem to prove
theorem find_f_inv_8 : f ⁻¹' {8} = {40} :=
by sorry

end NUMINAMATH_GPT_find_f_inv_8_l1307_130746


namespace NUMINAMATH_GPT_pieces_left_l1307_130752

def pieces_initial : ℕ := 900
def pieces_used : ℕ := 156

theorem pieces_left : pieces_initial - pieces_used = 744 := by
  sorry

end NUMINAMATH_GPT_pieces_left_l1307_130752


namespace NUMINAMATH_GPT_measure_of_angle_A_l1307_130707

-- Define the given conditions
variables (A B : ℝ)
axiom supplementary : A + B = 180
axiom measure_rel : A = 7 * B

-- The theorem statement to prove
theorem measure_of_angle_A : A = 157.5 :=
by
  -- proof steps would go here, but are omitted
  sorry

end NUMINAMATH_GPT_measure_of_angle_A_l1307_130707


namespace NUMINAMATH_GPT_max_quarters_l1307_130777

theorem max_quarters (a b c : ℕ) (h1 : a + b + c = 120) (h2 : 5 * a + 10 * b + 25 * c = 1000) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) : c ≤ 19 :=
sorry

example : ∃ a b c : ℕ, a + b + c = 120 ∧ 5 * a + 10 * b + 25 * c = 1000 ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ c = 19 :=
sorry

end NUMINAMATH_GPT_max_quarters_l1307_130777


namespace NUMINAMATH_GPT_height_of_right_triangle_l1307_130739

theorem height_of_right_triangle (a b c : ℝ) (h : ℝ) (h_right : a^2 + b^2 = c^2) (h_area : h = (a * b) / c) : h = (a * b) / c := 
by
  sorry

end NUMINAMATH_GPT_height_of_right_triangle_l1307_130739


namespace NUMINAMATH_GPT_greatest_ribbon_length_l1307_130797

-- Define lengths of ribbons
def ribbon_lengths : List ℕ := [8, 16, 20, 28]

-- Condition ensures gcd and prime check
def gcd_is_prime (n : ℕ) : Prop :=
  ∃ d : ℕ, (∀ l ∈ ribbon_lengths, d ∣ l) ∧ Prime d ∧ n = d

-- Prove the greatest length that can make the ribbon pieces, with no ribbon left over, is 2
theorem greatest_ribbon_length : ∃ d, gcd_is_prime d ∧ ∀ m, gcd_is_prime m → m ≤ 2 := 
sorry

end NUMINAMATH_GPT_greatest_ribbon_length_l1307_130797


namespace NUMINAMATH_GPT_infinite_sum_converges_to_3_l1307_130767

theorem infinite_sum_converges_to_3 :
  (∑' k : ℕ, (7 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))) = 3 :=
by
  sorry

end NUMINAMATH_GPT_infinite_sum_converges_to_3_l1307_130767


namespace NUMINAMATH_GPT_Kelly_weight_is_M_l1307_130773

variable (M : ℝ) -- Megan's weight
variable (K : ℝ) -- Kelly's weight
variable (Mike : ℝ) -- Mike's weight

-- Conditions based on the problem statement
def Kelly_less_than_Megan (M K : ℝ) : Prop := K = 0.85 * M
def Mike_greater_than_Megan (M Mike : ℝ) : Prop := Mike = M + 5
def Total_weight_exceeds_bridge (total_weight : ℝ) : Prop := total_weight = 100 + 19
def Total_weight_of_children (M K Mike total_weight : ℝ) : Prop := total_weight = M + K + Mike

theorem Kelly_weight_is_M : (M = 40) → (Total_weight_exceeds_bridge 119) → (Kelly_less_than_Megan M K) → (Mike_greater_than_Megan M Mike) → K = 34 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_Kelly_weight_is_M_l1307_130773


namespace NUMINAMATH_GPT_range_of_m_l1307_130724

def is_ellipse (m : ℝ) : Prop :=
  ∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

theorem range_of_m (m : ℝ) (h : is_ellipse m) : m > 5 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1307_130724


namespace NUMINAMATH_GPT_sum_of_squares_inequality_l1307_130749

theorem sum_of_squares_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ (1/3)*(a + b + c)^2 := sorry

end NUMINAMATH_GPT_sum_of_squares_inequality_l1307_130749


namespace NUMINAMATH_GPT_pow_mul_eq_add_l1307_130727

variable (a : ℝ)

theorem pow_mul_eq_add : a^2 * a^3 = a^5 := 
by 
  sorry

end NUMINAMATH_GPT_pow_mul_eq_add_l1307_130727


namespace NUMINAMATH_GPT_proof_problem_l1307_130782

open Real

-- Definitions of curves and transformations
def C1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }
def C2 := { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1 }

-- Parametric equation of C2
def parametric_C2 := ∃ α : ℝ, (0 ≤ α ∧ α ≤ 2*π) ∧
  (C2 = { p : ℝ × ℝ | p.1 = 2 * cos α ∧ p.2 = (1/2) * sin α })

-- Equation of line l1 maximizing the perimeter of ABCD
def line_l1 (p : ℝ × ℝ): Prop :=
  p.2 = (1/4) * p.1

theorem proof_problem : parametric_C2 ∧
  ∀ (A B C D : ℝ × ℝ),
    (A ∈ C2 ∧ B ∈ C2 ∧ C ∈ C2 ∧ D ∈ C2) →
    (line_l1 A ∧ line_l1 B) → 
    (line_l1 A ∧ line_l1 B) ∧
    (line_l1 C ∧ line_l1 D) →
    y = (1 / 4) * x :=
sorry

end NUMINAMATH_GPT_proof_problem_l1307_130782


namespace NUMINAMATH_GPT_red_card_events_l1307_130732

-- Definitions based on the conditions
inductive Person
| A | B | C | D

inductive Card
| Red | Black | Blue | White

-- Definition of the events
def event_A_receives_red (distribution : Person → Card) : Prop :=
  distribution Person.A = Card.Red

def event_B_receives_red (distribution : Person → Card) : Prop :=
  distribution Person.B = Card.Red

-- The relationship between the two events
def mutually_exclusive_but_not_opposite (distribution : Person → Card) : Prop :=
  (event_A_receives_red distribution → ¬ event_B_receives_red distribution) ∧
  (event_B_receives_red distribution → ¬ event_A_receives_red distribution)

-- The formal theorem statement
theorem red_card_events (distribution : Person → Card) :
  mutually_exclusive_but_not_opposite distribution :=
sorry

end NUMINAMATH_GPT_red_card_events_l1307_130732


namespace NUMINAMATH_GPT_minimum_value_l1307_130751

theorem minimum_value :
  ∀ (m n : ℝ), m > 0 → n > 0 → (3 * m + n = 1) → (3 / m + 1 / n) ≥ 16 :=
by
  intros m n hm hn hline
  sorry

end NUMINAMATH_GPT_minimum_value_l1307_130751


namespace NUMINAMATH_GPT_quadratic_coeffs_l1307_130700

theorem quadratic_coeffs (x : ℝ) :
  (x - 1)^2 = 3 * x - 2 → ∃ b c, (x^2 + b * x + c = 0 ∧ b = -5 ∧ c = 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_coeffs_l1307_130700


namespace NUMINAMATH_GPT_polynomial_behavior_l1307_130713

noncomputable def Q (x : ℝ) : ℝ := x^6 - 6 * x^5 + 10 * x^4 - x^3 - x + 12

theorem polynomial_behavior : 
  (∀ x : ℝ, x < 0 → Q x > 0) ∧ (∃ x : ℝ, x > 0 ∧ Q x = 0) := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_behavior_l1307_130713


namespace NUMINAMATH_GPT_rate_per_kg_grapes_is_70_l1307_130754

-- Let G be the rate per kg for the grapes
def rate_per_kg_grapes (G : ℕ) := G

-- Bruce purchased 8 kg of grapes at rate G per kg
def grapes_cost (G : ℕ) := 8 * G

-- Bruce purchased 11 kg of mangoes at the rate of 55 per kg
def mangoes_cost := 11 * 55

-- Bruce paid a total of 1165 to the shopkeeper
def total_paid := 1165

-- The problem: Prove that the rate per kg for the grapes is 70
theorem rate_per_kg_grapes_is_70 : rate_per_kg_grapes 70 = 70 ∧ grapes_cost 70 + mangoes_cost = total_paid := by
  sorry

end NUMINAMATH_GPT_rate_per_kg_grapes_is_70_l1307_130754


namespace NUMINAMATH_GPT_possible_values_f_one_l1307_130772

noncomputable def f (x : ℝ) : ℝ := sorry

variables (a b : ℝ)
axiom f_equation : ∀ x y : ℝ, 
  f ((x - y) ^ 2) = a * (f x)^2 - 2 * x * f y + b * y^2

theorem possible_values_f_one : f 1 = 1 ∨ f 1 = 2 :=
sorry

end NUMINAMATH_GPT_possible_values_f_one_l1307_130772


namespace NUMINAMATH_GPT_fred_added_nine_l1307_130744

def onions_in_basket (initial_onions : ℕ) (added_by_sara : ℕ) (taken_by_sally : ℕ) (added_by_fred : ℕ) : ℕ :=
  initial_onions + added_by_sara - taken_by_sally + added_by_fred

theorem fred_added_nine : ∀ (S F : ℕ), onions_in_basket S 4 5 F = S + 8 → F = 9 :=
by
  intros S F h
  sorry

end NUMINAMATH_GPT_fred_added_nine_l1307_130744


namespace NUMINAMATH_GPT_marble_ratio_l1307_130778

theorem marble_ratio (total_marbles red_marbles dark_blue_marbles : ℕ) (h_total : total_marbles = 63) (h_red : red_marbles = 38) (h_blue : dark_blue_marbles = 6) :
  (total_marbles - red_marbles - dark_blue_marbles) / red_marbles = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_marble_ratio_l1307_130778


namespace NUMINAMATH_GPT_solve_linear_system_l1307_130794

-- Given conditions
def matrix : Matrix (Fin 2) (Fin 3) ℚ :=
  ![![1, -1, 1], ![1, 1, 3]]

def system_of_equations (x y : ℚ) : Prop :=
  (x - y = 1) ∧ (x + y = 3)

-- Desired solution
def solution (x y : ℚ) : Prop :=
  x = 2 ∧ y = 1

-- Proof problem statement
theorem solve_linear_system : ∃ x y : ℚ, system_of_equations x y ∧ solution x y := by
  sorry

end NUMINAMATH_GPT_solve_linear_system_l1307_130794


namespace NUMINAMATH_GPT_leonardo_needs_more_money_l1307_130763

-- Defining the problem
def cost_of_chocolate : ℕ := 500 -- 5 dollars in cents
def leonardo_own_money : ℕ := 400 -- 4 dollars in cents
def borrowed_money : ℕ := 59 -- borrowed cents

-- Prove that Leonardo needs 41 more cents
theorem leonardo_needs_more_money : (cost_of_chocolate - (leonardo_own_money + borrowed_money) = 41) :=
by
  sorry

end NUMINAMATH_GPT_leonardo_needs_more_money_l1307_130763


namespace NUMINAMATH_GPT_triangle_centroid_altitude_l1307_130784

/-- In triangle XYZ with side lengths XY = 7, XZ = 24, and YZ = 25, the length of GQ where Q 
    is the foot of the altitude from the centroid G to the side YZ is 56/25. -/
theorem triangle_centroid_altitude :
  let XY := 7
  let XZ := 24
  let YZ := 25
  let GQ := 56 / 25
  GQ = (56 : ℝ) / 25 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_triangle_centroid_altitude_l1307_130784


namespace NUMINAMATH_GPT_find_triangle_with_properties_l1307_130762

-- Define the angles forming an arithmetic progression
def angles_arithmetic_progression (α β γ : ℝ) : Prop :=
  β - α = γ - β

-- Define the sides forming an arithmetic progression
def sides_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

-- Define the sides forming a geometric progression
def sides_geometric_progression (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Define the sum of angles in a triangle
def sum_of_angles (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- The problem statement:
theorem find_triangle_with_properties 
    (α β γ a b c : ℝ)
    (h1 : angles_arithmetic_progression α β γ)
    (h2 : sum_of_angles α β γ)
    (h3 : sides_arithmetic_progression a b c ∨ sides_geometric_progression a b c) :
  α = 60 ∧ β = 60 ∧ γ = 60 :=
by 
  sorry

end NUMINAMATH_GPT_find_triangle_with_properties_l1307_130762


namespace NUMINAMATH_GPT_exponent_fraction_simplification_l1307_130798

theorem exponent_fraction_simplification :
  (2 ^ 2020 + 2 ^ 2016) / (2 ^ 2020 - 2 ^ 2016) = 17 / 15 :=
by
  sorry

end NUMINAMATH_GPT_exponent_fraction_simplification_l1307_130798


namespace NUMINAMATH_GPT_problem1_problem2_problem3_general_conjecture_l1307_130734

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

-- Prove f(0) + f(1) = sqrt(2) / 2
theorem problem1 : f 0 + f 1 = Real.sqrt 2 / 2 := by
  sorry

-- Prove f(-1) + f(2) = sqrt(2) / 2
theorem problem2 : f (-1) + f 2 = Real.sqrt 2 / 2 := by
  sorry

-- Prove f(-2) + f(3) = sqrt(2) / 2
theorem problem3 : f (-2) + f 3 = Real.sqrt 2 / 2 := by
  sorry

-- Prove ∀ x, f(-x) + f(x+1) = sqrt(2) / 2
theorem general_conjecture (x : ℝ) : f (-x) + f (x + 1) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_general_conjecture_l1307_130734


namespace NUMINAMATH_GPT_max_area_of_rectangle_l1307_130709

-- Question: Prove the largest possible area of a rectangle given the conditions
theorem max_area_of_rectangle :
  ∀ (x : ℝ), (2 * x + 2 * (x + 5) = 60) → x * (x + 5) ≤ 218.75 :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_rectangle_l1307_130709


namespace NUMINAMATH_GPT_line_intersects_axes_l1307_130765

theorem line_intersects_axes (a b : ℝ) (x1 y1 x2 y2 : ℝ) (h_points : (x1, y1) = (8, 2) ∧ (x2, y2) = (4, 6)) :
  (∃ x_intercept : ℝ, (x_intercept, 0) = (10, 0)) ∧ (∃ y_intercept : ℝ, (0, y_intercept) = (0, 10)) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_axes_l1307_130765


namespace NUMINAMATH_GPT_parabola_focus_distance_l1307_130733

theorem parabola_focus_distance (C : Set (ℝ × ℝ))
  (hC : ∀ x y, (y^2 = x) → (x, y) ∈ C)
  (F : ℝ × ℝ)
  (hF : F = (1/4, 0))
  (A : ℝ × ℝ)
  (hA : A = (x0, y0) ∧ (y0^2 = x0 ∧ (x0, y0) ∈ C))
  (hAF : dist A F = (5/4) * x0) :
  x0 = 1 :=
sorry

end NUMINAMATH_GPT_parabola_focus_distance_l1307_130733


namespace NUMINAMATH_GPT_find_fruit_juice_amount_l1307_130741

def total_punch : ℕ := 14 * 10
def mountain_dew : ℕ := 6 * 12
def ice : ℕ := 28
def fruit_juice : ℕ := total_punch - mountain_dew - ice

theorem find_fruit_juice_amount : fruit_juice = 40 := by
  sorry

end NUMINAMATH_GPT_find_fruit_juice_amount_l1307_130741


namespace NUMINAMATH_GPT_car_speed_l1307_130725

theorem car_speed 
  (d : ℝ) (t : ℝ) 
  (hd : d = 520) (ht : t = 8) : 
  d / t = 65 := 
by 
  sorry

end NUMINAMATH_GPT_car_speed_l1307_130725


namespace NUMINAMATH_GPT_division_of_floats_l1307_130704

theorem division_of_floats : 4.036 / 0.04 = 100.9 :=
by
  sorry

end NUMINAMATH_GPT_division_of_floats_l1307_130704


namespace NUMINAMATH_GPT_k_range_l1307_130708

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  (Real.log x) - x - x * Real.exp (-x) - k

theorem k_range (k : ℝ) : (∀ x > 0, ∃ x > 0, f x k = 0) ↔ k ≤ -1 - (1 / Real.exp 1) :=
sorry

end NUMINAMATH_GPT_k_range_l1307_130708


namespace NUMINAMATH_GPT_minimum_value_of_a_l1307_130783

-- Define the given condition
axiom a_pos : ℝ → Prop
axiom positive : ∀ (x : ℝ), 0 < x

-- Definition of the equation
def equation (x y a : ℝ) : Prop :=
  (2 * x - y / Real.exp 1) * Real.log (y / x) = x / (a * Real.exp 1)

-- The mathematical statement we need to prove
theorem minimum_value_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) (h_eq : equation x y a) : 
  a ≥ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_a_l1307_130783


namespace NUMINAMATH_GPT_column_of_2023_l1307_130716

theorem column_of_2023 : 
  let columns := ["G", "H", "I", "J", "K", "L", "M"]
  let pattern := ["H", "I", "J", "K", "L", "M", "L", "K", "J", "I", "H", "G"]
  let n := 2023
  (pattern.get! ((n - 2) % 12)) = "I" :=
by
  -- Sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_column_of_2023_l1307_130716


namespace NUMINAMATH_GPT_q_is_false_of_pq_false_and_notp_false_l1307_130722

variables (p q : Prop)

theorem q_is_false_of_pq_false_and_notp_false (hpq_false : ¬(p ∧ q)) (hnotp_false : ¬(¬p)) : ¬q := 
by 
  sorry

end NUMINAMATH_GPT_q_is_false_of_pq_false_and_notp_false_l1307_130722
