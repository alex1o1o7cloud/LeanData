import Mathlib

namespace NUMINAMATH_GPT_minimize_d_and_distance_l1323_132315

-- Define point and geometric shapes
structure Point :=
  (x : ℝ)
  (y : ℝ)

def Parabola (P : Point) : Prop := P.x^2 = 4 * P.y
def Circle (P1 : Point) : Prop := (P1.x - 2)^2 + (P1.y + 1)^2 = 1

-- Define the point P and point P1
variable (P : Point)
variable (P1 : Point)

-- Condition: P is on the parabola
axiom on_parabola : Parabola P

-- Condition: P1 is on the circle
axiom on_circle : Circle P1

-- Theorem: coordinates of P when the function d + distance(P, P1) is minimized
theorem minimize_d_and_distance :
  P = { x := 2 * Real.sqrt 2 - 2, y := 3 - 2 * Real.sqrt 2 } :=
sorry

end NUMINAMATH_GPT_minimize_d_and_distance_l1323_132315


namespace NUMINAMATH_GPT_quadratic_has_two_real_roots_l1323_132387

-- Define the condition that the discriminant must be non-negative
def discriminant_nonneg (a b c : ℝ) : Prop := b * b - 4 * a * c ≥ 0

-- Define our specific quadratic equation conditions: x^2 - 2x + m = 0
theorem quadratic_has_two_real_roots (m : ℝ) :
  discriminant_nonneg 1 (-2) m → m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_real_roots_l1323_132387


namespace NUMINAMATH_GPT_parabola_focus_l1323_132382

open Real

theorem parabola_focus (a : ℝ) (h k : ℝ) (x y : ℝ) (f : ℝ) :
  (a = -1/4) → (h = 0) → (k = 0) → 
  (f = (1 / (4 * a))) →
  (y = a * (x - h) ^ 2 + k) → 
  (y = -1 / 4 * x ^ 2) → f = -1 := by
  intros h_a h_h h_k h_f parabola_eq _
  rw [h_a, h_h, h_k] at *
  sorry

end NUMINAMATH_GPT_parabola_focus_l1323_132382


namespace NUMINAMATH_GPT_sandbox_volume_l1323_132361

def length : ℕ := 312
def width : ℕ := 146
def depth : ℕ := 75
def volume (l w d : ℕ) : ℕ := l * w * d

theorem sandbox_volume : volume length width depth = 3429000 := by
  sorry

end NUMINAMATH_GPT_sandbox_volume_l1323_132361


namespace NUMINAMATH_GPT_f_2016_is_1_l1323_132343

noncomputable def f : ℤ → ℤ := sorry

axiom h1 : f 1 = 1
axiom h2 : f 2015 ≠ 1
axiom h3 : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)
axiom h4 : ∀ x : ℤ, f x = f (-x)

theorem f_2016_is_1 : f 2016 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_f_2016_is_1_l1323_132343


namespace NUMINAMATH_GPT_total_soda_consumption_l1323_132356

variables (c_soda b_soda c_consumed b_consumed b_remaining carol_final bob_final total_consumed : ℕ)

-- Define the conditions
def carol_soda_size : ℕ := 20
def bob_soda_25_percent_more : ℕ := carol_soda_size + carol_soda_size * 25 / 100
def carol_consumed : ℕ := carol_soda_size * 80 / 100
def bob_consumed : ℕ := bob_soda_25_percent_more * 80 / 100
def carol_remaining : ℕ := carol_soda_size - carol_consumed
def bob_remaining : ℕ := bob_soda_25_percent_more - bob_consumed
def bob_gives_carol : ℕ := bob_remaining / 2 + 3
def carol_final_consumption : ℕ := carol_consumed + bob_gives_carol
def bob_final_consumption : ℕ := bob_consumed - bob_gives_carol
def total_soda_consumed : ℕ := carol_final_consumption + bob_final_consumption

-- The theorem to prove the total amount of soda consumed by Carol and Bob together is 36 ounces
theorem total_soda_consumption : total_soda_consumed = 36 := by {
  sorry
}

end NUMINAMATH_GPT_total_soda_consumption_l1323_132356


namespace NUMINAMATH_GPT_cost_price_6500_l1323_132324

variable (CP SP : ℝ)

-- Condition 1: The selling price is 30% more than the cost price.
def selling_price (CP : ℝ) : ℝ := CP * 1.3

-- Condition 2: The selling price is Rs. 8450.
axiom selling_price_8450 : selling_price CP = 8450

-- Prove that the cost price of the computer table is Rs. 6500.
theorem cost_price_6500 : CP = 6500 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_6500_l1323_132324


namespace NUMINAMATH_GPT_find_value_of_a2_plus_b2_plus_c2_l1323_132359

variables (a b c : ℝ)

-- Define the conditions
def conditions := (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (a + b + c = 0) ∧ (a^3 + b^3 + c^3 = a^5 + b^5 + c^5)

-- State the theorem we need to prove
theorem find_value_of_a2_plus_b2_plus_c2 (h : conditions a b c) : a^2 + b^2 + c^2 = 6 / 5 :=
  sorry

end NUMINAMATH_GPT_find_value_of_a2_plus_b2_plus_c2_l1323_132359


namespace NUMINAMATH_GPT_problem1_problem2_l1323_132357

-- Proof Problem 1: Prove that when \( k = 5 \), \( x^2 - 5x + 4 > 0 \) holds for \( \{x \mid x < 1 \text{ or } x > 4\} \).
theorem problem1 (x : ℝ) (h : x^2 - 5 * x + 4 > 0) : x < 1 ∨ x > 4 :=
sorry

-- Proof Problem 2: Prove that the range of values for \( k \) such that \( x^2 - kx + 4 > 0 \) holds for all real numbers \( x \) is \( (-4, 4) \).
theorem problem2 (k : ℝ) : (∀ x : ℝ, x^2 - k * x + 4 > 0) ↔ -4 < k ∧ k < 4 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1323_132357


namespace NUMINAMATH_GPT_sum_of_coefficients_l1323_132331

theorem sum_of_coefficients (a b c d : ℝ) (f : ℝ → ℝ)
    (h1 : ∀ x, f (x + 2) = 2*x^3 + 5*x^2 + 3*x + 6)
    (h2 : ∀ x, f x = a*x^3 + b*x^2 + c*x + d) :
  a + b + c + d = 6 :=
by sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1323_132331


namespace NUMINAMATH_GPT_area_of_triangle_arithmetic_sides_l1323_132318

theorem area_of_triangle_arithmetic_sides 
  (a : ℝ) (h : a > 0) (h_sin : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2) :
  let s₁ := a - 2
  let s₂ := a
  let s₃ := a + 2
  ∃ (a b c : ℝ), 
    a = s₁ ∧ b = s₂ ∧ c = s₃ ∧ 
    Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 → 
    (1/2 * s₁ * s₂ * Real.sin (2 * Real.pi / 3) = 15 * Real.sqrt 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_arithmetic_sides_l1323_132318


namespace NUMINAMATH_GPT_area_triangle_ABF_proof_area_triangle_AFD_proof_l1323_132351

variable (A B C D M F : Type)
variable (area_square : Real) (midpoint_D_CM : Prop) (lies_on_line_BC : Prop)

-- Given conditions
axiom area_ABCD_300 : area_square = 300
axiom M_midpoint_DC : midpoint_D_CM
axiom F_on_line_BC : lies_on_line_BC

-- Define areas for the triangles
def area_triangle_ABF : Real := 300
def area_triangle_AFD : Real := 150

-- Prove that given the conditions, the area of triangle ABF is 300 cm²
theorem area_triangle_ABF_proof : area_square = 300 ∧ midpoint_D_CM ∧ lies_on_line_BC → area_triangle_ABF = 300 :=
by
  intro h
  sorry

-- Prove that given the conditions, the area of triangle AFD is 150 cm²
theorem area_triangle_AFD_proof : area_square = 300 ∧ midpoint_D_CM ∧ lies_on_line_BC → area_triangle_AFD = 150 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_area_triangle_ABF_proof_area_triangle_AFD_proof_l1323_132351


namespace NUMINAMATH_GPT_first_three_workers_dig_time_l1323_132309

variables 
  (a b c d : ℝ) -- work rates of the four workers
  (hours : ℝ) -- time to dig the trench

def work_together (a b c d hours : ℝ) := (a + b + c + d) * hours = 1

def scenario1 (a b c d : ℝ) := (2 * a + (1/2) * b + c + d) * 6 = 1

def scenario2 (a b c d : ℝ) := (a/2 + 2 * b + c + d) * 4 = 1

theorem first_three_workers_dig_time
  (h1 : work_together a b c d 6)
  (h2 : scenario1 a b c d)
  (h3 : scenario2 a b c d) :
  hours = 6 := 
sorry

end NUMINAMATH_GPT_first_three_workers_dig_time_l1323_132309


namespace NUMINAMATH_GPT_width_of_room_l1323_132388

theorem width_of_room (C r l : ℝ) (hC : C = 18700) (hr : r = 850) (hl : l = 5.5) : 
  ∃ w, C / r / l = w ∧ w = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_width_of_room_l1323_132388


namespace NUMINAMATH_GPT_powerjet_pumps_250_gallons_in_30_minutes_l1323_132335

theorem powerjet_pumps_250_gallons_in_30_minutes :
  let rate : ℝ := 500
  let time_in_hours : ℝ := 1 / 2
  rate * time_in_hours = 250 :=
by
  sorry

end NUMINAMATH_GPT_powerjet_pumps_250_gallons_in_30_minutes_l1323_132335


namespace NUMINAMATH_GPT_inequality_solution_l1323_132342

theorem inequality_solution (a x : ℝ) (h : |a + 1| < 3) :
  (-4 < a ∧ a < -2 ∧ (x > -1 ∨ x < 1 + a)) ∨ 
  (a = -2 ∧ (x ∈ Set.univ \ {-1})) ∨ 
  (-2 < a ∧ a < 2 ∧ (x > 1 + a ∨ x < -1)) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l1323_132342


namespace NUMINAMATH_GPT_equilateral_triangle_sum_l1323_132310

theorem equilateral_triangle_sum (side_length : ℚ) (h_eq : side_length = 13 / 12) :
  3 * side_length = 13 / 4 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_equilateral_triangle_sum_l1323_132310


namespace NUMINAMATH_GPT_lily_catches_up_mary_in_60_minutes_l1323_132394

theorem lily_catches_up_mary_in_60_minutes
  (mary_speed : ℝ) (lily_speed : ℝ) (initial_distance : ℝ)
  (h_mary_speed : mary_speed = 4)
  (h_lily_speed : lily_speed = 6)
  (h_initial_distance : initial_distance = 2) :
  ∃ t : ℝ, t = 60 := by
  sorry

end NUMINAMATH_GPT_lily_catches_up_mary_in_60_minutes_l1323_132394


namespace NUMINAMATH_GPT_ratio_of_bases_l1323_132338

-- Definitions for an isosceles trapezoid
def isosceles_trapezoid (s t : ℝ) := ∃ (a b c d : ℝ), s = d ∧ s = a ∧ t = b ∧ (a + c = b + d)

-- Main theorem statement based on conditions and required ratio
theorem ratio_of_bases (s t : ℝ) (h1 : isosceles_trapezoid s t)
  (h2 : s = s) (h3 : t = t) : s / t = 3 / 5 :=
by { sorry }

end NUMINAMATH_GPT_ratio_of_bases_l1323_132338


namespace NUMINAMATH_GPT_mary_flour_requirement_l1323_132368

theorem mary_flour_requirement (total_flour : ℕ) (added_flour : ℕ) (remaining_flour : ℕ) 
  (h1 : total_flour = 7) 
  (h2 : added_flour = 2) 
  (h3 : remaining_flour = total_flour - added_flour) : 
  remaining_flour = 5 :=
sorry

end NUMINAMATH_GPT_mary_flour_requirement_l1323_132368


namespace NUMINAMATH_GPT_stratified_sampling_correct_l1323_132371

variables (total_employees senior_employees mid_level_employees junior_employees sample_size : ℕ)
          (sampling_ratio : ℚ)
          (senior_sample mid_sample junior_sample : ℕ)

-- Conditions
def company_conditions := 
  total_employees = 450 ∧ 
  senior_employees = 45 ∧ 
  mid_level_employees = 135 ∧ 
  junior_employees = 270 ∧ 
  sample_size = 30 ∧ 
  sampling_ratio = 1 / 15

-- Proof goal
theorem stratified_sampling_correct : 
  company_conditions total_employees senior_employees mid_level_employees junior_employees sample_size sampling_ratio →
  senior_sample = senior_employees * sampling_ratio ∧ 
  mid_sample = mid_level_employees * sampling_ratio ∧ 
  junior_sample = junior_employees * sampling_ratio ∧
  senior_sample + mid_sample + junior_sample = sample_size :=
by sorry

end NUMINAMATH_GPT_stratified_sampling_correct_l1323_132371


namespace NUMINAMATH_GPT_min_value_a_4b_l1323_132316

theorem min_value_a_4b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = a + b) :
  a + 4 * b = 9 :=
sorry

end NUMINAMATH_GPT_min_value_a_4b_l1323_132316


namespace NUMINAMATH_GPT_extremum_and_equal_values_l1323_132367

theorem extremum_and_equal_values {f : ℝ → ℝ} {a b x_0 x_1 : ℝ} 
    (hf : ∀ x, f x = (x - 1)^3 - a * x + b)
    (h'x0 : deriv f x_0 = 0)
    (hfx1_eq_fx0 : f x_1 = f x_0)
    (hx1_ne_x0 : x_1 ≠ x_0) :
  x_1 + 2 * x_0 = 3 := sorry

end NUMINAMATH_GPT_extremum_and_equal_values_l1323_132367


namespace NUMINAMATH_GPT_cost_effectiveness_l1323_132348

-- Define the variables and conditions
def num_employees : ℕ := 30
def ticket_price : ℝ := 80
def group_discount_rate : ℝ := 0.8
def women_discount_rate : ℝ := 0.5

-- Define the costs for each scenario
def cost_with_group_discount : ℝ := num_employees * ticket_price * group_discount_rate

def cost_with_women_discount (x : ℕ) : ℝ :=
  ticket_price * women_discount_rate * x + ticket_price * (num_employees - x)

-- Formalize the equivalence of cost and comparison logic
theorem cost_effectiveness (x : ℕ) (h : 0 ≤ x ∧ x ≤ num_employees) :
  if x < 12 then cost_with_women_discount x > cost_with_group_discount
  else if x = 12 then cost_with_women_discount x = cost_with_group_discount
  else cost_with_women_discount x < cost_with_group_discount :=
by sorry

end NUMINAMATH_GPT_cost_effectiveness_l1323_132348


namespace NUMINAMATH_GPT_max_points_in_equilateral_property_set_l1323_132358

theorem max_points_in_equilateral_property_set (Γ : Finset (ℝ × ℝ)) :
  (∀ (A B : (ℝ × ℝ)), A ∈ Γ → B ∈ Γ → 
    ∃ C : (ℝ × ℝ), C ∈ Γ ∧ 
    dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B) → Γ.card ≤ 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_max_points_in_equilateral_property_set_l1323_132358


namespace NUMINAMATH_GPT_least_five_digit_perfect_square_and_cube_l1323_132365

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end NUMINAMATH_GPT_least_five_digit_perfect_square_and_cube_l1323_132365


namespace NUMINAMATH_GPT_trailing_zeros_50_factorial_l1323_132327

def factorial_trailing_zeros (n : Nat) : Nat :=
  n / 5 + n / 25 -- Count the number of trailing zeros given the algorithm used in solution steps

theorem trailing_zeros_50_factorial : factorial_trailing_zeros 50 = 12 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_trailing_zeros_50_factorial_l1323_132327


namespace NUMINAMATH_GPT_area_PQR_is_4_5_l1323_132300

noncomputable def point := (ℝ × ℝ)

def P : point := (2, 1)
def Q : point := (1, 4)
def R_line (x: ℝ) : point := (x, 6 - x)

def area_triangle (A B C : point) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem area_PQR_is_4_5 (x : ℝ) (h : R_line x ∈ {p : point | p.1 + p.2 = 6}) : 
  area_triangle P Q (R_line x) = 4.5 :=
    sorry

end NUMINAMATH_GPT_area_PQR_is_4_5_l1323_132300


namespace NUMINAMATH_GPT_maximum_pairwise_sum_is_maximal_l1323_132305

noncomputable def maximum_pairwise_sum (set_sums : List ℝ) (x y z w : ℝ) : Prop :=
  ∃ (a b c d e : ℝ), set_sums = [400, 500, 600, 700, 800, 900, x, y, z, w] ∧  
  ((2 / 5) * (400 + 500 + 600 + 700 + 800 + 900 + x + y + z + w)) = 
    (a + b + c + d + e) ∧ 
  5 * (a + b + c + d + e) - (400 + 500 + 600 + 700 + 800 + 900) = 1966.67

theorem maximum_pairwise_sum_is_maximal :
  maximum_pairwise_sum [400, 500, 600, 700, 800, 900] 1966.67 (1966.67 / 4) 
(1966.67 / 3) (1966.67 / 2) :=
sorry

end NUMINAMATH_GPT_maximum_pairwise_sum_is_maximal_l1323_132305


namespace NUMINAMATH_GPT_find_triplets_find_triplets_non_negative_l1323_132396

theorem find_triplets :
  ∀ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) →
    x^2 + y^2 + 1 = 2^z →
    (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 1) :=
by
  sorry

theorem find_triplets_non_negative :
  ∀ (x y z : ℕ), x^2 + y^2 + 1 = 2^z →
    (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_triplets_find_triplets_non_negative_l1323_132396


namespace NUMINAMATH_GPT_ratio_of_numbers_l1323_132354

theorem ratio_of_numbers (A B D M : ℕ) 
  (h1 : A + B + D = M)
  (h2 : Nat.gcd A B = D)
  (h3 : Nat.lcm A B = M)
  (h4 : A ≥ B) : A / B = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l1323_132354


namespace NUMINAMATH_GPT_div_by_5_l1323_132353

theorem div_by_5 (n : ℕ) (hn : 0 < n) : (2^(4*n+1) + 3) % 5 = 0 := 
by sorry

end NUMINAMATH_GPT_div_by_5_l1323_132353


namespace NUMINAMATH_GPT_algebraic_expression_decrease_l1323_132391

theorem algebraic_expression_decrease (x y : ℝ) :
  let original_expr := 2 * x^2 * y
  let new_expr := 2 * ((1 / 2) * x) ^ 2 * ((1 / 2) * y)
  let decrease := ((original_expr - new_expr) / original_expr) * 100
  decrease = 87.5 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_decrease_l1323_132391


namespace NUMINAMATH_GPT_fixed_monthly_costs_l1323_132372

theorem fixed_monthly_costs
  (production_cost_per_component : ℕ)
  (shipping_cost_per_component : ℕ)
  (components_per_month : ℕ)
  (lowest_price_per_component : ℕ)
  (total_revenue : ℕ)
  (total_variable_cost : ℕ)
  (F : ℕ) :
  production_cost_per_component = 80 →
  shipping_cost_per_component = 5 →
  components_per_month = 150 →
  lowest_price_per_component = 195 →
  total_variable_cost = components_per_month * (production_cost_per_component + shipping_cost_per_component) →
  total_revenue = components_per_month * lowest_price_per_component →
  total_revenue = total_variable_cost + F →
  F = 16500 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_fixed_monthly_costs_l1323_132372


namespace NUMINAMATH_GPT_expected_red_pairs_correct_l1323_132313

-- Define the number of red cards and the total number of cards
def red_cards : ℕ := 25
def total_cards : ℕ := 50

-- Calculate the probability that one red card is followed by another red card in a circle of total_cards
def prob_adj_red : ℚ := (red_cards - 1) / (total_cards - 1)

-- The expected number of pairs of adjacent red cards
def expected_adj_red_pairs : ℚ := red_cards * prob_adj_red

-- The theorem to be proved: the expected number of adjacent red pairs is 600/49
theorem expected_red_pairs_correct : expected_adj_red_pairs = 600 / 49 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_expected_red_pairs_correct_l1323_132313


namespace NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l1323_132352

-- Theorem follows the tuple of (question, conditions, correct answer)
theorem smallest_n_for_terminating_decimal (n : ℕ) (h : ∃ k : ℕ, n + 75 = 2^k ∨ n + 75 = 5^k ∨ n + 75 = (2^k * 5^k)) :
  n = 50 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l1323_132352


namespace NUMINAMATH_GPT_at_least_one_inequality_holds_l1323_132373

theorem at_least_one_inequality_holds (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y > 2) : 
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_inequality_holds_l1323_132373


namespace NUMINAMATH_GPT_packets_of_chips_l1323_132390

variable (P R M : ℕ)

theorem packets_of_chips (h1: P > 0) (h2: R > 0) (h3: M > 0) :
  ((10 * M * P) / R) = (10 * M * P) / R :=
sorry

end NUMINAMATH_GPT_packets_of_chips_l1323_132390


namespace NUMINAMATH_GPT_rachel_class_choices_l1323_132380

theorem rachel_class_choices : (Nat.choose 8 3) = 56 :=
by
  sorry

end NUMINAMATH_GPT_rachel_class_choices_l1323_132380


namespace NUMINAMATH_GPT_sum_of_consecutive_multiples_of_4_l1323_132374

theorem sum_of_consecutive_multiples_of_4 (n : ℝ) (h : 4 * n + (4 * n + 8) = 140) :
  4 * n + (4 * n + 4) + (4 * n + 8) = 210 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_multiples_of_4_l1323_132374


namespace NUMINAMATH_GPT_distance_to_place_l1323_132322

variables {r c1 c2 t D : ℝ}

theorem distance_to_place (h : t = (D / (r - c1)) + (D / (r + c2))) :
  D = t * (r^2 - c1 * c2) / (2 * r + c2 - c1) :=
by
  have h1 : D * (r + c2) / (r - c1) * (r - c1) = D * (r + c2) := by sorry
  have h2 : D * (r - c1) / (r + c2) * (r + c2) = D * (r - c1) := by sorry
  have h3 : D * (r + c2) = D * (r + c2) := by sorry
  have h4 : D * (r - c1) = D * (r - c1) := by sorry
  have h5 : t * (r - c1) * (r + c2) = D * (r + c2) + D * (r - c1) := by sorry
  have h6 : t * (r^2 - c1 * c2) = D * (2 * r + c2 - c1) := by sorry
  have h7 : D = t * (r^2 - c1 * c2) / (2 * r + c2 - c1) := by sorry
  exact h7

end NUMINAMATH_GPT_distance_to_place_l1323_132322


namespace NUMINAMATH_GPT_evaluate_star_property_l1323_132326

noncomputable def star (a b : ℕ) : ℕ := b ^ a

theorem evaluate_star_property (a b c m : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hm : 0 < m) :
  (star a b ≠ star b a) ∧
  (star a (star b c) ≠ star (star a b) c) ∧
  (star a (b ^ m) ≠ star (star a m) b) ∧
  ((star a b) ^ m ≠ star a (m * b)) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_star_property_l1323_132326


namespace NUMINAMATH_GPT_perimeter_of_shaded_region_l1323_132378

noncomputable def circle_center : Type := sorry -- Define the object type for circle's center
noncomputable def radius_length : ℝ := 10 -- Define the radius length as 10
noncomputable def central_angle : ℝ := 270 -- Define the central angle corresponding to the arc RS

-- Function to calculate the perimeter of the shaded region
noncomputable def perimeter_shaded_region (radius : ℝ) (angle : ℝ) : ℝ :=
  2 * radius + (angle / 360) * 2 * Real.pi * radius

-- Theorem stating that the perimeter of the shaded region is 20 + 15π given the conditions
theorem perimeter_of_shaded_region : 
  perimeter_shaded_region radius_length central_angle = 20 + 15 * Real.pi :=
by
  -- skipping the actual proof
  sorry

end NUMINAMATH_GPT_perimeter_of_shaded_region_l1323_132378


namespace NUMINAMATH_GPT_intersect_complement_A_B_eq_l1323_132345

open Set

variable (U : Set ℝ)
variable (A : Set ℝ)
variable (B : Set ℝ)

noncomputable def complement_A : Set ℝ := U \ A
noncomputable def intersection_complement_A_B : Set ℝ := complement_A U A ∩ B

theorem intersect_complement_A_B_eq : 
  U = univ ∧ A = {x : ℝ | x + 1 < 0} ∧ B = {x : ℝ | x - 3 < 0} →
  intersection_complement_A_B U A B = Icc (-1 : ℝ) 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_intersect_complement_A_B_eq_l1323_132345


namespace NUMINAMATH_GPT_diamond_associative_l1323_132330

def diamond (a b : ℕ) : ℕ := a ^ (b / a)

theorem diamond_associative (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  diamond a (diamond b c) = diamond (diamond a b) c :=
sorry

end NUMINAMATH_GPT_diamond_associative_l1323_132330


namespace NUMINAMATH_GPT_line_through_point_equal_intercepts_l1323_132385

-- Definitions based on conditions
def passes_through (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l p.1 p.2

def equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a, a ≠ 0 ∧ (∀ x y, l x y ↔ x + y = a) ∨ (∀ x y, l x y ↔ y = 2 * x)

-- Theorem statement based on the problem
theorem line_through_point_equal_intercepts :
  ∃ l, passes_through (1, 2) l ∧ equal_intercepts l ∧
  (∀ x y, l x y ↔ 2 * x - y = 0) ∨ (∀ x y, l x y ↔ x + y - 3 = 0) :=
sorry

end NUMINAMATH_GPT_line_through_point_equal_intercepts_l1323_132385


namespace NUMINAMATH_GPT_range_of_a_l1323_132346

variable {x a : ℝ}

def p (x : ℝ) := 2*x^2 - 3*x + 1 ≤ 0
def q (x : ℝ) (a : ℝ) := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a (h : ¬ p x → ¬ q x a) : 0 ≤ a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1323_132346


namespace NUMINAMATH_GPT_gcd_consecutive_digits_l1323_132339

theorem gcd_consecutive_digits (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) 
  (h₁ : b = a + 1) (h₂ : c = a + 2) (h₃ : d = a + 3) :
  ∃ g, g = gcd (1000 * a + 100 * b + 10 * c + d - (1000 * d + 100 * c + 10 * b + a)) 3096 :=
by {
  sorry
}

end NUMINAMATH_GPT_gcd_consecutive_digits_l1323_132339


namespace NUMINAMATH_GPT_jordan_machine_solution_l1323_132334

theorem jordan_machine_solution (x : ℝ) (h : 2 * x + 3 - 5 = 27) : x = 14.5 :=
sorry

end NUMINAMATH_GPT_jordan_machine_solution_l1323_132334


namespace NUMINAMATH_GPT_solve_quadratic_equation_solve_linear_equation_l1323_132321

-- Equation (1)
theorem solve_quadratic_equation :
  ∀ x : ℝ, x^2 - 8 * x + 1 = 0 → (x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15) :=
by
  sorry

-- Equation (2)
theorem solve_linear_equation :
  ∀ x : ℝ, 3 * x * (x - 1) = 2 - 2 * x → (x = 1 ∨ x = -2/3) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_solve_linear_equation_l1323_132321


namespace NUMINAMATH_GPT_find_remainder_l1323_132317

theorem find_remainder (x y P Q : ℕ) (hx : 0 < x) (hy : 0 < y) (h : x^4 + y^4 = (P + 13) * (x + y) + Q) : Q = 8 :=
sorry

end NUMINAMATH_GPT_find_remainder_l1323_132317


namespace NUMINAMATH_GPT_composite_expression_l1323_132320

theorem composite_expression (n : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (a * b = 6 * 2^(2^(4 * n)) + 1) :=
by
  sorry

end NUMINAMATH_GPT_composite_expression_l1323_132320


namespace NUMINAMATH_GPT_student_missed_20_l1323_132364

theorem student_missed_20 {n : ℕ} (S_correct : ℕ) (S_incorrect : ℕ) 
    (h1 : S_correct = n * (n + 1) / 2)
    (h2 : S_incorrect = S_correct - 20) : 
    S_incorrect = n * (n + 1) / 2 - 20 := 
sorry

end NUMINAMATH_GPT_student_missed_20_l1323_132364


namespace NUMINAMATH_GPT_sum_of_squares_l1323_132392

theorem sum_of_squares (x y : ℝ) (h₁ : x + y = 40) (h₂ : x * y = 120) : x^2 + y^2 = 1360 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1323_132392


namespace NUMINAMATH_GPT_find_c_squared_ab_l1323_132369

theorem find_c_squared_ab (a b c : ℝ) (h1 : a^2 * (b + c) = 2008) (h2 : b^2 * (a + c) = 2008) (h3 : a ≠ b) : 
  c^2 * (a + b) = 2008 :=
sorry

end NUMINAMATH_GPT_find_c_squared_ab_l1323_132369


namespace NUMINAMATH_GPT_average_speed_of_bus_trip_l1323_132379

theorem average_speed_of_bus_trip 
  (v d : ℝ) 
  (h1 : d = 560)
  (h2 : ∀ v > 0, ∀ Δv > 0, (d / v) - (d / (v + Δv)) = 2)
  (h3 : Δv = 10): 
  v = 50 := 
by 
  sorry

end NUMINAMATH_GPT_average_speed_of_bus_trip_l1323_132379


namespace NUMINAMATH_GPT_analogical_reasoning_l1323_132381

theorem analogical_reasoning {a b c : ℝ} (h1 : c ≠ 0) : 
  (a + b) * c = a * c + b * c → (a + b) / c = a / c + b / c := 
by 
  sorry

end NUMINAMATH_GPT_analogical_reasoning_l1323_132381


namespace NUMINAMATH_GPT_find_first_term_l1323_132399

theorem find_first_term (a : ℚ) (n : ℕ) (T : ℕ → ℚ)
  (hT : ∀ n, T n = n * (2 * a + 5 * (n - 1)) / 2)
  (h_const : ∃ c : ℚ, ∀ n > 0, T (4 * n) / T n = c) :
  a = 5 / 2 := 
sorry

end NUMINAMATH_GPT_find_first_term_l1323_132399


namespace NUMINAMATH_GPT_find_original_number_l1323_132377

theorem find_original_number (x : ℚ) (h : 5 * ((3 * x + 6) / 2) = 100) : x = 34 / 3 := sorry

end NUMINAMATH_GPT_find_original_number_l1323_132377


namespace NUMINAMATH_GPT_first_discount_percentage_l1323_132341

theorem first_discount_percentage (x : ℝ) :
  let initial_price := 26.67
  let final_price := 15.0
  let second_discount := 0.25
  (initial_price * (1 - x / 100) * (1 - second_discount) = final_price) → x = 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_first_discount_percentage_l1323_132341


namespace NUMINAMATH_GPT_dorothy_profit_l1323_132398

-- Define the conditions
def expense := 53
def number_of_doughnuts := 25
def price_per_doughnut := 3

-- Define revenue and profit calculations
def revenue := number_of_doughnuts * price_per_doughnut
def profit := revenue - expense

-- Prove the profit calculation
theorem dorothy_profit : profit = 22 := by
  sorry

end NUMINAMATH_GPT_dorothy_profit_l1323_132398


namespace NUMINAMATH_GPT_find_coefficient_y_l1323_132304

theorem find_coefficient_y (a b c : ℕ) (h1 : 100 * a + 10 * b + c - 7 * (a + b + c) = 100) (h2 : a + b + c ≠ 0) :
  100 * c + 10 * b + a = 43 * (a + b + c) :=
by
  sorry

end NUMINAMATH_GPT_find_coefficient_y_l1323_132304


namespace NUMINAMATH_GPT_remainder_of_sum_of_powers_div_2_l1323_132383

theorem remainder_of_sum_of_powers_div_2 : 
  (1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7 + 8^8 + 9^9) % 2 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_powers_div_2_l1323_132383


namespace NUMINAMATH_GPT_smallest_whole_number_gt_total_sum_l1323_132329

-- Declarations of the fractions involved
def term1 : ℚ := 3 + 1/3
def term2 : ℚ := 4 + 1/6
def term3 : ℚ := 5 + 1/12
def term4 : ℚ := 6 + 1/8

-- Definition of the entire sum
def total_sum : ℚ := term1 + term2 + term3 + term4

-- Statement of the theorem
theorem smallest_whole_number_gt_total_sum : 
  ∀ n : ℕ, (n > total_sum) → (∀ m : ℕ, (m >= 0) → (m > total_sum) → (n ≤ m)) → n = 19 := by
  sorry -- the proof is omitted

end NUMINAMATH_GPT_smallest_whole_number_gt_total_sum_l1323_132329


namespace NUMINAMATH_GPT_n_cube_plus_5n_divisible_by_6_l1323_132336

theorem n_cube_plus_5n_divisible_by_6 (n : ℤ) : 6 ∣ (n^3 + 5 * n) := 
sorry

end NUMINAMATH_GPT_n_cube_plus_5n_divisible_by_6_l1323_132336


namespace NUMINAMATH_GPT_range_of_ab_l1323_132349

noncomputable def circle_equation (x y : ℝ) : Prop := (x^2 + y^2 + 2*x - 4*y + 1 = 0)

noncomputable def line_equation (a b x y : ℝ) : Prop := (2*a*x - b*y - 2 = 0)

def symmetric_with_respect_to (center_x center_y a b : ℝ) : Prop :=
  line_equation a b center_x center_y  -- check if the line passes through the center

theorem range_of_ab (a b : ℝ) (h_symm : symmetric_with_respect_to (-1) 2 a b) : 
  ∃ ab_max : ℝ, ab_max = 1/4 ∧ ∀ ab : ℝ, ab = (a * b) → ab ≤ ab_max :=
sorry

end NUMINAMATH_GPT_range_of_ab_l1323_132349


namespace NUMINAMATH_GPT_part1_part2_part3a_part3b_l1323_132363

open Real

variable (θ : ℝ) (m : ℝ)

-- Conditions
axiom theta_domain : 0 < θ ∧ θ < 2 * π
axiom quadratic_eq : ∀ x : ℝ, 2 * x^2 - (sqrt 3 + 1) * x + m = 0
axiom roots_eq_theta : ∀ x : ℝ, (x = sin θ ∨ x = cos θ)

-- Proof statements
theorem part1 : 1 - cos θ ≠ 0 → 1 - tan θ ≠ 0 → 
  (sin θ / (1 - cos θ) + cos θ / (1 - tan θ)) = (3 + 5 * sqrt 3) / 4 := sorry

theorem part2 : sin θ * cos θ = m / 2 → m = sqrt 3 / 4 := sorry

theorem part3a : sin θ = sqrt 3 / 2 ∧ cos θ = 1 / 2 → θ = π / 3 := sorry

theorem part3b : sin θ = 1 / 2 ∧ cos θ = sqrt 3 / 2 → θ = π / 6 := sorry

end NUMINAMATH_GPT_part1_part2_part3a_part3b_l1323_132363


namespace NUMINAMATH_GPT_ribbon_arrangement_count_correct_l1323_132340

-- Definitions for the problem conditions
inductive Color
| red
| yellow
| blue

-- The color sequence from top to bottom
def color_sequence : List Color := [Color.red, Color.blue, Color.yellow, Color.yellow]

-- A function to count the valid arrangements
def count_valid_arrangements (sequence : List Color) : Nat :=
  -- Since we need to prove, we're bypassing the actual implementation with sorry
  sorry

-- The proof statement
theorem ribbon_arrangement_count_correct : count_valid_arrangements color_sequence = 12 :=
by
  sorry

end NUMINAMATH_GPT_ribbon_arrangement_count_correct_l1323_132340


namespace NUMINAMATH_GPT_min_floodgates_to_reduce_level_l1323_132389

-- Definitions for the conditions given in the problem
def num_floodgates : ℕ := 10
def a (v : ℝ) := 30 * v
def w (v : ℝ) := 2 * v

def time_one_gate : ℝ := 30
def time_two_gates : ℝ := 10
def time_target : ℝ := 3

-- Prove that the minimum number of floodgates \(n\) that must be opened to achieve the goal
theorem min_floodgates_to_reduce_level (v : ℝ) (n : ℕ) :
  (a v + time_target * v) ≤ (n * time_target * w v) → n ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_min_floodgates_to_reduce_level_l1323_132389


namespace NUMINAMATH_GPT_perpendicular_bisector_c_value_l1323_132350

theorem perpendicular_bisector_c_value :
  (∃ c : ℝ, ∀ x y : ℝ, 
    2 * x - y = c ↔ x = 5 ∧ y = 8) → c = 2 := 
by
  sorry

end NUMINAMATH_GPT_perpendicular_bisector_c_value_l1323_132350


namespace NUMINAMATH_GPT_minimum_value_proof_l1323_132308

noncomputable def min_value (x y : ℝ) : ℝ :=
  (x^2 / (x + 2)) + (y^2 / (y + 1))

theorem minimum_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  min_value x y = 1 / 4 :=
  sorry

end NUMINAMATH_GPT_minimum_value_proof_l1323_132308


namespace NUMINAMATH_GPT_find_e1_l1323_132301

-- Definitions related to the problem statement
variable (P F1 F2 : Type)
variable (cos_angle : ℝ)
variable (e1 e2 : ℝ)

-- Conditions
def cosine_angle_condition := cos_angle = 3 / 5
def eccentricity_relation := e2 = 2 * e1

-- Theorem that needs to be proved
theorem find_e1 (h_cos : cosine_angle_condition cos_angle)
                (h_ecc_rel : eccentricity_relation e1 e2) :
  e1 = Real.sqrt 10 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_e1_l1323_132301


namespace NUMINAMATH_GPT_comm_add_comm_mul_distrib_l1323_132306

variable {α : Type*} [AddCommMonoid α] [Mul α] [Distrib α]

theorem comm_add (a b : α) : a + b = b + a :=
by sorry

theorem comm_mul (a b : α) : a * b = b * a :=
by sorry

theorem distrib (a b c : α) : (a + b) * c = a * c + b * c :=
by sorry

end NUMINAMATH_GPT_comm_add_comm_mul_distrib_l1323_132306


namespace NUMINAMATH_GPT_quadratic_coefficient_conversion_l1323_132355

theorem quadratic_coefficient_conversion :
  ∀ x : ℝ, (3 * x^2 - 1 = 5 * x) → (3 * x^2 - 5 * x - 1 = 0) :=
by
  intros x h
  rw [←sub_eq_zero, ←h]
  ring

end NUMINAMATH_GPT_quadratic_coefficient_conversion_l1323_132355


namespace NUMINAMATH_GPT_maximum_ratio_l1323_132312

-- Define the conditions
def is_two_digit_positive_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def mean_is_45 (x y : ℕ) : Prop :=
  (x + y) / 2 = 45

-- State the theorem
theorem maximum_ratio (x y : ℕ) (hx : is_two_digit_positive_integer x) (hy : is_two_digit_positive_integer y) (h_mean : mean_is_45 x y) : 
  ∃ (k: ℕ), (x / y = k) ∧ k = 8 :=
sorry

end NUMINAMATH_GPT_maximum_ratio_l1323_132312


namespace NUMINAMATH_GPT_average_waiting_time_l1323_132328

-- Define the problem conditions
def light_period : ℕ := 3  -- Total cycle time in minutes
def green_time : ℕ := 1    -- Green light duration in minutes
def red_time : ℕ := 2      -- Red light duration in minutes

-- Define the probabilities of each light state
def P_G : ℚ := green_time / light_period
def P_R : ℚ := red_time / light_period

-- Define the expected waiting times given each state
def E_T_G : ℚ := 0
def E_T_R : ℚ := red_time / 2

-- Calculate the expected waiting time using the law of total expectation
def E_T : ℚ := E_T_G * P_G + E_T_R * P_R

-- Convert the expected waiting time to seconds
def E_T_seconds : ℚ := E_T * 60

-- Prove that the expected waiting time in seconds is 40 seconds
theorem average_waiting_time : E_T_seconds = 40 := by
  sorry

end NUMINAMATH_GPT_average_waiting_time_l1323_132328


namespace NUMINAMATH_GPT_determine_abcd_l1323_132360

theorem determine_abcd (a b c d : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) (h₁ : 0 ≤ b ∧ b ≤ 9) 
    (h₂ : 0 ≤ c ∧ c ≤ 9) (h₃ : 0 ≤ d ∧ d ≤ 9) 
    (h₄ : (10 * a + b) / 99 + (1000 * a + 100 * b + 10 * c + d) / 9999 = 27 / 37) :
    1000 * a + 100 * b + 10 * c + d = 3644 :=
by
  sorry

end NUMINAMATH_GPT_determine_abcd_l1323_132360


namespace NUMINAMATH_GPT_variance_is_0_02_l1323_132366

def data_points : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

theorem variance_is_0_02 : variance data_points = 0.02 :=
by
  sorry

end NUMINAMATH_GPT_variance_is_0_02_l1323_132366


namespace NUMINAMATH_GPT_avg_megabyte_usage_per_hour_l1323_132347

theorem avg_megabyte_usage_per_hour (megabytes : ℕ) (days : ℕ) (hours : ℕ) (avg_mbps : ℕ)
  (h1 : megabytes = 27000)
  (h2 : days = 15)
  (h3 : hours = days * 24)
  (h4 : avg_mbps = megabytes / hours) : 
  avg_mbps = 75 := by
  sorry

end NUMINAMATH_GPT_avg_megabyte_usage_per_hour_l1323_132347


namespace NUMINAMATH_GPT_obtain_2015_in_4_operations_obtain_2015_in_3_operations_l1323_132344

-- Define what an operation is
def operation (cards : List ℕ) : List ℕ :=
  sorry  -- Implementation of this is unnecessary for the statement

-- Check if 2015 can be obtained in 4 operations
def can_obtain_2015_in_4_operations (initial_cards : List ℕ) : Prop :=
  ∃ cards, (operation^[4] initial_cards) = cards ∧ 2015 ∈ cards

-- Check if 2015 can be obtained in 3 operations
def can_obtain_2015_in_3_operations (initial_cards : List ℕ) : Prop :=
  ∃ cards, (operation^[3] initial_cards) = cards ∧ 2015 ∈ cards

theorem obtain_2015_in_4_operations :
  can_obtain_2015_in_4_operations [1, 2] :=
sorry

theorem obtain_2015_in_3_operations :
  can_obtain_2015_in_3_operations [1, 2] :=
sorry

end NUMINAMATH_GPT_obtain_2015_in_4_operations_obtain_2015_in_3_operations_l1323_132344


namespace NUMINAMATH_GPT_true_proposition_among_ABCD_l1323_132307

theorem true_proposition_among_ABCD : 
  (∀ x : ℝ, x^2 < x + 1) = false ∧
  (∀ x : ℝ, x^2 ≥ x + 1) = false ∧
  (∃ x : ℝ, ∀ y : ℝ, x * y^2 ≠ y^2) = true ∧
  (∀ x : ℝ, ∃ y : ℝ, x > y^2) = false :=
by 
  sorry

end NUMINAMATH_GPT_true_proposition_among_ABCD_l1323_132307


namespace NUMINAMATH_GPT_find_middle_number_l1323_132303

theorem find_middle_number (x y z : ℤ) (h1 : x + y = 22) (h2 : x + z = 29) (h3 : y + z = 37) (h4 : x < y) (h5 : y < z) : y = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_middle_number_l1323_132303


namespace NUMINAMATH_GPT_mean_of_remaining_three_numbers_l1323_132319

theorem mean_of_remaining_three_numbers 
    (a b c d : ℝ)
    (h₁ : (a + b + c + d) / 4 = 92)
    (h₂ : d = 120)
    (h₃ : b = 60) : 
    (a + b + c) / 3 = 82.6666666666 := 
by 
    -- This state suggests adding the constraints added so far for the proof:
    sorry

end NUMINAMATH_GPT_mean_of_remaining_three_numbers_l1323_132319


namespace NUMINAMATH_GPT_relationship_a_b_c_l1323_132362

noncomputable def a := Real.log 3 / Real.log (1/2)
noncomputable def b := Real.log (1/2) / Real.log 3
noncomputable def c := Real.exp (0.3 * Real.log 2)

theorem relationship_a_b_c : 
  a < b ∧ b < c := 
by {
  sorry
}

end NUMINAMATH_GPT_relationship_a_b_c_l1323_132362


namespace NUMINAMATH_GPT_nails_to_buy_l1323_132311

-- Define the initial number of nails Tom has
def initial_nails : ℝ := 247

-- Define the number of nails found in the toolshed
def toolshed_nails : ℝ := 144

-- Define the number of nails found in a drawer
def drawer_nails : ℝ := 0.5

-- Define the number of nails given by the neighbor
def neighbor_nails : ℝ := 58.75

-- Define the total number of nails needed for the project
def total_needed_nails : ℝ := 625.25

-- Define the total number of nails Tom already has
def total_existing_nails : ℝ := 
  initial_nails + toolshed_nails + drawer_nails + neighbor_nails

-- Prove that Tom needs to buy 175 more nails
theorem nails_to_buy :
  total_needed_nails - total_existing_nails = 175 := by
  sorry

end NUMINAMATH_GPT_nails_to_buy_l1323_132311


namespace NUMINAMATH_GPT_cyclic_sum_fraction_ge_one_l1323_132384

theorem cyclic_sum_fraction_ge_one (a b c : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (hineq : (a/(b+c+1) + b/(c+a+1) + c/(a+b+1)) ≤ 1) :
  (1/(b+c+1) + 1/(c+a+1) + 1/(a+b+1)) ≥ 1 :=
by sorry

end NUMINAMATH_GPT_cyclic_sum_fraction_ge_one_l1323_132384


namespace NUMINAMATH_GPT_rectangular_solid_length_l1323_132337

theorem rectangular_solid_length (w h : ℕ) (surface_area : ℕ) (l : ℕ) 
  (hw : w = 4) (hh : h = 1) (hsa : surface_area = 58) 
  (h_surface_area_formula : surface_area = 2 * l * w + 2 * l * h + 2 * w * h) : 
  l = 5 :=
by
  rw [hw, hh, hsa] at h_surface_area_formula
  sorry

end NUMINAMATH_GPT_rectangular_solid_length_l1323_132337


namespace NUMINAMATH_GPT_smallest_c_l1323_132302

variable {f : ℝ → ℝ}

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 = 0) ∧ (f 1 = 1) ∧ (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) ∧ (∀ x1 x2, 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2)

theorem smallest_c (f : ℝ → ℝ) (h : satisfies_conditions f) : (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ 2 * x) ∧ (∀ c, c < 2 → ∃ x, 0 < x ∧ x ≤ 1 ∧ ¬ (f x ≤ c * x)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_c_l1323_132302


namespace NUMINAMATH_GPT_prime_numbers_satisfy_equation_l1323_132333

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_satisfy_equation :
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ (p + q^2 = r^4) ∧ 
  (p = 7) ∧ (q = 3) ∧ (r = 2) :=
by
  sorry

end NUMINAMATH_GPT_prime_numbers_satisfy_equation_l1323_132333


namespace NUMINAMATH_GPT_sum_a_b_l1323_132314

variable {a b : ℝ}

theorem sum_a_b (hab : a * b = 5) (hrecip : 1 / (a^2) + 1 / (b^2) = 0.6) : a + b = 5 ∨ a + b = -5 :=
sorry

end NUMINAMATH_GPT_sum_a_b_l1323_132314


namespace NUMINAMATH_GPT_min_value_expression_l1323_132376

open Real

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x^2 + y^2 + z^2 = 1) : 
  (∃ (c : ℝ), c = 3 * sqrt 3 / 2 ∧ c ≤ (x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2))) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1323_132376


namespace NUMINAMATH_GPT_transformed_parabolas_combined_l1323_132397

theorem transformed_parabolas_combined (a b c : ℝ) :
  let f (x : ℝ) := a * (x - 3) ^ 2 + b * (x - 3) + c
  let g (x : ℝ) := -a * (x + 4) ^ 2 - b * (x + 4) - c
  ∀ x, (f x + g x) = -14 * a * x - 19 * a - 7 * b :=
by
  -- This is a placeholder for the actual proof using the conditions
  sorry

end NUMINAMATH_GPT_transformed_parabolas_combined_l1323_132397


namespace NUMINAMATH_GPT_Alice_fills_needed_l1323_132323

def cups_needed : ℚ := 15/4
def cup_capacity : ℚ := 1/3
def fills_needed : ℚ := 12

theorem Alice_fills_needed : (cups_needed / cup_capacity).ceil = fills_needed := by
  -- Proof is omitted with sorry
  sorry

end NUMINAMATH_GPT_Alice_fills_needed_l1323_132323


namespace NUMINAMATH_GPT_always_two_real_roots_find_m_l1323_132370

-- Given quadratic equation: x^2 - 4mx + 3m^2 = 0
-- Definitions for the problem
def quadratic_eq (m x : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 = 0

-- Q1: Prove that this equation always has two real roots.
theorem always_two_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ :=
by
  sorry

-- Q2: If m > 0 and the difference between the two real roots is 2, find the value of m.
theorem find_m (m : ℝ) (h₁ : m > 0) (h₂ : ∃ x₁ x₂ : ℝ, quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ |x₁ - x₂| = 2) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_always_two_real_roots_find_m_l1323_132370


namespace NUMINAMATH_GPT_exists_infinite_irregular_set_l1323_132332

def is_irregular (A : Set ℤ) :=
  ∀ ⦃x y : ℤ⦄, x ∈ A → y ∈ A → x ≠ y → ∀ ⦃k : ℤ⦄, x + k * (y - x) ≠ x ∧ x + k * (y - x) ≠ y

theorem exists_infinite_irregular_set : ∃ A : Set ℤ, Set.Infinite A ∧ is_irregular A :=
sorry

end NUMINAMATH_GPT_exists_infinite_irregular_set_l1323_132332


namespace NUMINAMATH_GPT_license_plate_count_l1323_132395

def license_plate_combinations : Nat :=
  26 * Nat.choose 25 2 * Nat.choose 4 2 * 720

theorem license_plate_count :
  license_plate_combinations = 33696000 :=
by
  unfold license_plate_combinations
  sorry

end NUMINAMATH_GPT_license_plate_count_l1323_132395


namespace NUMINAMATH_GPT_bridge_weight_excess_l1323_132325

theorem bridge_weight_excess :
  ∀ (Kelly_weight Megan_weight Mike_weight : ℕ),
  Kelly_weight = 34 →
  Kelly_weight = 85 * Megan_weight / 100 →
  Mike_weight = Megan_weight + 5 →
  (Kelly_weight + Megan_weight + Mike_weight) - 100 = 19 :=
by
  intros Kelly_weight Megan_weight Mike_weight
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_bridge_weight_excess_l1323_132325


namespace NUMINAMATH_GPT_tan_sum_eq_tan_prod_l1323_132375

noncomputable def tan (x : Real) : Real :=
  Real.sin x / Real.cos x

theorem tan_sum_eq_tan_prod (α β γ : Real) (h : tan α + tan β + tan γ = tan α * tan β * tan γ) :
  ∃ k : Int, α + β + γ = k * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_eq_tan_prod_l1323_132375


namespace NUMINAMATH_GPT_calc1_calc2_calc3_calc4_calc5_calc6_l1323_132393

theorem calc1 : 320 + 16 * 27 = 752 :=
by
  -- Proof goes here
  sorry

theorem calc2 : 1500 - 125 * 8 = 500 :=
by
  -- Proof goes here
  sorry

theorem calc3 : 22 * 22 - 84 = 400 :=
by
  -- Proof goes here
  sorry

theorem calc4 : 25 * 8 * 9 = 1800 :=
by
  -- Proof goes here
  sorry

theorem calc5 : (25 + 38) * 15 = 945 :=
by
  -- Proof goes here
  sorry

theorem calc6 : (62 + 12) * 38 = 2812 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_calc1_calc2_calc3_calc4_calc5_calc6_l1323_132393


namespace NUMINAMATH_GPT_birds_in_store_l1323_132386

/-- 
A pet store had a total of 180 animals, consisting of birds, dogs, and cats. 
Among the birds, 64 talked, and 13 didn't. If there were 40 dogs in the store 
and the number of birds that talked was four times the number of cats, 
prove that there were 124 birds in total.
-/
theorem birds_in_store (total_animals : ℕ) (talking_birds : ℕ) (non_talking_birds : ℕ) 
  (dogs : ℕ) (cats : ℕ) 
  (h1 : total_animals = 180)
  (h2 : talking_birds = 64)
  (h3 : non_talking_birds = 13)
  (h4 : dogs = 40)
  (h5 : talking_birds = 4 * cats) : 
  talking_birds + non_talking_birds + dogs + cats = 180 ∧ 
  talking_birds + non_talking_birds = 124 :=
by
  -- We are skipping the proof itself and focusing on the theorem statement
  sorry

end NUMINAMATH_GPT_birds_in_store_l1323_132386
