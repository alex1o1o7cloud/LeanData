import Mathlib

namespace common_difference_is_3_l2320_232062

theorem common_difference_is_3 (a : ℕ → ℤ) (d : ℤ) (h1 : a 2 = 4) (h2 : 1 + a 3 = 5 + d)
  (h3 : a 6 = 4 + 4 * d) (h4 : 4 + a 10 = 8 + 8 * d) :
  (5 + d) * (8 + 8 * d) = (4 + 4 * d) ^ 2 → d = 3 := 
by
  intros hg
  sorry

end common_difference_is_3_l2320_232062


namespace find_number_l2320_232034

theorem find_number (n : ℕ) : (n / 2) + 5 = 15 → n = 20 :=
by
  intro h
  sorry

end find_number_l2320_232034


namespace side_length_of_S2_l2320_232026

variable (r s : ℝ)

theorem side_length_of_S2 (h1 : 2 * r + s = 2100) (h2 : 2 * r + 3 * s = 3400) : s = 650 := by
  sorry

end side_length_of_S2_l2320_232026


namespace initial_total_balls_l2320_232093

theorem initial_total_balls (B T : Nat) (h1 : B = 9) (h2 : ∀ (n : Nat), (T - 5) * 1/5 = 4) :
  T = 25 := sorry

end initial_total_balls_l2320_232093


namespace harriet_speed_l2320_232044

/-- Harriet drove back from B-town to A-ville at a constant speed of 145 km/hr.
    The entire trip took 5 hours, and it took Harriet 2.9 hours to drive from A-ville to B-town.
    Prove that Harriet's speed while driving from A-ville to B-town was 105 km/hr. -/
theorem harriet_speed (v_return : ℝ) (T_total : ℝ) (t_AB : ℝ) (v_AB : ℝ) :
  v_return = 145 →
  T_total = 5 →
  t_AB = 2.9 →
  v_AB = 105 :=
by
  intros
  sorry

end harriet_speed_l2320_232044


namespace cost_price_of_ball_l2320_232058

variable (C : ℝ)

theorem cost_price_of_ball (h : 15 * C - 720 = 5 * C) : C = 72 :=
by
  sorry

end cost_price_of_ball_l2320_232058


namespace remainder_when_divided_by_multiple_of_10_l2320_232023

theorem remainder_when_divided_by_multiple_of_10 (N : ℕ) (hN : ∃ k : ℕ, N = 10 * k) (hrem : (19 ^ 19 + 19) % N = 18) : N = 10 := by
  sorry

end remainder_when_divided_by_multiple_of_10_l2320_232023


namespace quadratic_z_and_u_l2320_232079

variables (a b c α β γ : ℝ)
variable (d : ℝ)
variable (δ : ℝ)
variables (x₁ x₂ y₁ y₂ z₁ z₂ u₁ u₂ : ℝ)

-- Given conditions
variable (h_nonzero : a * α ≠ 0)
variable (h_discriminant1 : b^2 - 4 * a * c ≥ 0)
variable (h_discriminant2 : β^2 - 4 * α * γ ≥ 0)
variable (hx_roots_order : x₁ ≤ x₂)
variable (hy_roots_order : y₁ ≤ y₂)
variable (h_eq_discriminant1 : b^2 - 4 * a * c = d^2)
variable (h_eq_discriminant2 : β^2 - 4 * α * γ = δ^2)

-- Translate into mathematical constraints for the roots
variable (hx1 : x₁ = (-b - d) / (2 * a))
variable (hx2 : x₂ = (-b + d) / (2 * a))
variable (hy1 : y₁ = (-β - δ) / (2 * α))
variable (hy2 : y₂ = (-β + δ) / (2 * α))

-- Variables for polynomial equations roots
axiom h_z1 : z₁ = x₁ + y₁
axiom h_z2 : z₂ = x₂ + y₂
axiom h_u1 : u₁ = x₁ + y₂
axiom h_u2 : u₂ = x₂ + y₁

theorem quadratic_z_and_u :
  (2 * a * α) * z₂ * z₂ + 2 * (a * β + α * b) * z₁ + (2 * a * γ + 2 * α * c + b * β - d * δ) = 0 ∧
  (2 * a * α) * u₂ * u₂ + 2 * (a * β + α * b) * u₁ + (2 * a * γ + 2 * α * c + b * β + d * δ) = 0 := sorry

end quadratic_z_and_u_l2320_232079


namespace micheal_item_count_l2320_232020

theorem micheal_item_count : ∃ a b c : ℕ, a + b + c = 50 ∧ 60 * a + 500 * b + 400 * c = 10000 ∧ a = 30 :=
  by
    sorry

end micheal_item_count_l2320_232020


namespace solve_for_x_l2320_232007

theorem solve_for_x :
  ∃ x : ℤ, (225 - 4209520 / ((1000795 + (250 + x) * 50) / 27)) = 113 ∧ x = 40 := 
by
  sorry

end solve_for_x_l2320_232007


namespace find_solutions_l2320_232094

theorem find_solutions :
  {x : ℝ | 1 / (x^2 + 12 * x - 9) + 1 / (x^2 + 3 * x - 9) + 1 / (x^2 - 14 * x - 9) = 0} = {1, -9, 3, -3} :=
by
  sorry

end find_solutions_l2320_232094


namespace draw_13_cards_no_straight_flush_l2320_232015

theorem draw_13_cards_no_straight_flush :
  let deck_size := 52
  let suit_count := 4
  let rank_count := 13
  let non_straight_flush_draws (n : ℕ) := 3^n - 3
  n = rank_count →
  ∀ (draw : ℕ), draw = non_straight_flush_draws n :=
by
-- Proof would be here
sorry

end draw_13_cards_no_straight_flush_l2320_232015


namespace units_digit_of_7_pow_3_l2320_232028

theorem units_digit_of_7_pow_3 : (7 ^ 3) % 10 = 3 :=
by
  sorry

end units_digit_of_7_pow_3_l2320_232028


namespace income_expenditure_ratio_l2320_232036

variable (I S E : ℕ)
variable (hI : I = 16000)
variable (hS : S = 3200)
variable (hExp : S = I - E)

theorem income_expenditure_ratio (I S E : ℕ) (hI : I = 16000) (hS : S = 3200) (hExp : S = I - E) : I / Nat.gcd I E = 5 ∧ E / Nat.gcd I E = 4 := by
  sorry

end income_expenditure_ratio_l2320_232036


namespace tanya_bought_11_pears_l2320_232001

variable (P : ℕ)

-- Define the given conditions about the number of different fruits Tanya bought
def apples : ℕ := 4
def pineapples : ℕ := 2
def basket_of_plums : ℕ := 1

-- Define the total number of fruits initially and the remaining fruits
def initial_fruit_total : ℕ := 18
def remaining_fruit_total : ℕ := 9
def half_fell_out_of_bag : ℕ := remaining_fruit_total * 2

-- The main theorem to prove
theorem tanya_bought_11_pears (h : P + apples + pineapples + basket_of_plums = initial_fruit_total) : P = 11 := by
  -- providing a placeholder for the proof
  sorry

end tanya_bought_11_pears_l2320_232001


namespace prime_divides_product_of_divisors_l2320_232010

theorem prime_divides_product_of_divisors (p : ℕ) (n : ℕ) (a : Fin n → ℕ) 
(Hp : Nat.Prime p) (Hdiv : p ∣ (Finset.univ.prod a)) : 
∃ i : Fin n, p ∣ a i :=
sorry

end prime_divides_product_of_divisors_l2320_232010


namespace fraction_operation_l2320_232075

theorem fraction_operation : (3 / 5 - 1 / 10 + 2 / 15 = 19 / 30) :=
by
  sorry

end fraction_operation_l2320_232075


namespace opera_house_earnings_l2320_232057

theorem opera_house_earnings :
  let rows := 150
  let seats_per_row := 10
  let ticket_cost := 10
  let total_seats := rows * seats_per_row
  let seats_not_taken := total_seats * 20 / 100
  let seats_taken := total_seats - seats_not_taken
  let total_earnings := ticket_cost * seats_taken
  total_earnings = 12000 := by
sorry

end opera_house_earnings_l2320_232057


namespace mr_green_garden_yield_l2320_232003

noncomputable def garden_yield (steps_length steps_width step_length yield_per_sqft : ℝ) : ℝ :=
  let length_ft := steps_length * step_length
  let width_ft := steps_width * step_length
  let area := length_ft * width_ft
  area * yield_per_sqft

theorem mr_green_garden_yield :
  garden_yield 18 25 2.5 0.5 = 1406.25 :=
by
  sorry

end mr_green_garden_yield_l2320_232003


namespace thread_length_l2320_232059

theorem thread_length (initial_length : ℝ) (fraction : ℝ) (additional_length : ℝ) (total_length : ℝ) 
  (h1 : initial_length = 12) 
  (h2 : fraction = 3 / 4) 
  (h3 : additional_length = initial_length * fraction)
  (h4 : total_length = initial_length + additional_length) : 
  total_length = 21 := 
by
  -- proof steps would go here
  sorry

end thread_length_l2320_232059


namespace johns_raise_percentage_increase_l2320_232052

def initial_earnings : ℚ := 65
def new_earnings : ℚ := 70
def percentage_increase (initial new : ℚ) : ℚ := ((new - initial) / initial) * 100

theorem johns_raise_percentage_increase : percentage_increase initial_earnings new_earnings = 7.692307692 :=
by
  sorry

end johns_raise_percentage_increase_l2320_232052


namespace possible_values_of_m_l2320_232078

theorem possible_values_of_m
  (m : ℕ)
  (h1 : ∃ (m' : ℕ), m = m' ∧ 0 < m)            -- m is a positive integer
  (h2 : 2 * (m - 1) + 3 * (m + 2) > 4 * (m - 5))    -- AB + AC > BC
  (h3 : 2 * (m - 1) + 4 * (m + 5) > 3 * (m + 2))    -- AB + BC > AC
  (h4 : 3 * (m + 2) + 4 * (m + 5) > 2 * (m - 1))    -- AC + BC > AB
  (h5 : 3 * (m + 2) > 2 * (m - 1))                  -- AC > AB
  (h6 : 4 * (m + 5) > 3 * (m + 2))                  -- BC > AC
  : m ≥ 7 := 
sorry

end possible_values_of_m_l2320_232078


namespace sum_of_squares_of_roots_eq_21_l2320_232030

theorem sum_of_squares_of_roots_eq_21 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 + x2^2 = 21 ∧ x1 + x2 = -a ∧ x1 * x2 = 2*a) ↔ a = -3 :=
by
  sorry

end sum_of_squares_of_roots_eq_21_l2320_232030


namespace poly_div_l2320_232033

theorem poly_div (A B : ℂ) :
  (∀ x : ℂ, x^3 + x^2 + 1 = 0 → x^202 + A * x + B = 0) → A + B = 0 :=
by
  intros h
  sorry

end poly_div_l2320_232033


namespace part_I_part_II_l2320_232053

noncomputable def curve_M (theta : ℝ) : ℝ := 4 * Real.cos theta

noncomputable def line_l (t m alpha : ℝ) : ℝ × ℝ :=
  let x := m + t * Real.cos alpha
  let y := t * Real.sin alpha
  (x, y)

theorem part_I (varphi : ℝ) :
  let OB := curve_M (varphi + π / 4)
  let OC := curve_M (varphi - π / 4)
  let OA := curve_M varphi
  OB + OC = Real.sqrt 2 * OA := by
  sorry

theorem part_II (m alpha : ℝ) :
  let varphi := π / 12
  let B := (1, Real.sqrt 3)
  let C := (3, -Real.sqrt 3)
  exists t1 t2, line_l t1 m alpha = B ∧ line_l t2 m alpha = C :=
  have hα : alpha = 2 * π / 3 := by sorry
  have hm : m = 2 := by sorry
  sorry

end part_I_part_II_l2320_232053


namespace exponentiation_calculation_l2320_232027

theorem exponentiation_calculation : 3000 * (3000 ^ 3000) ^ 2 = 3000 ^ 6001 := by
  sorry

end exponentiation_calculation_l2320_232027


namespace apples_handed_out_l2320_232002

theorem apples_handed_out 
  (initial_apples : ℕ)
  (pies_made : ℕ)
  (apples_per_pie : ℕ)
  (H : initial_apples = 50)
  (H1 : pies_made = 9)
  (H2 : apples_per_pie = 5) :
  initial_apples - (pies_made * apples_per_pie) = 5 := 
by
  sorry

end apples_handed_out_l2320_232002


namespace number_of_semesters_l2320_232072

-- Define the given conditions
def units_per_semester : ℕ := 20
def cost_per_unit : ℕ := 50
def total_cost : ℕ := 2000

-- Define the cost per semester using the conditions
def cost_per_semester := units_per_semester * cost_per_unit

-- Prove the number of semesters is 2 given the conditions
theorem number_of_semesters : total_cost / cost_per_semester = 2 := by
  -- Add a placeholder "sorry" to skip the actual proof
  sorry

end number_of_semesters_l2320_232072


namespace order_of_numbers_l2320_232005

def base16_to_dec (s : String) : ℕ := sorry
def base6_to_dec (s : String) : ℕ := sorry
def base4_to_dec (s : String) : ℕ := sorry
def base2_to_dec (s : String) : ℕ := sorry

theorem order_of_numbers:
  let a := base16_to_dec "3E"
  let b := base6_to_dec "210"
  let c := base4_to_dec "1000"
  let d := base2_to_dec "111011"
  a = 62 ∧ b = 78 ∧ c = 64 ∧ d = 59 →
  b > c ∧ c > a ∧ a > d :=
by
  intros
  sorry

end order_of_numbers_l2320_232005


namespace min_abs_E_value_l2320_232068

theorem min_abs_E_value (x E : ℝ) (h : |x - 4| + |E| + |x - 5| = 10) : |E| = 9 :=
sorry

end min_abs_E_value_l2320_232068


namespace find_angle_C_find_area_of_triangle_l2320_232067

variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Sides of the triangle

-- Proof 1: Prove \(C = \frac{\pi}{3}\) given \(a \cos B \cos C + b \cos A \cos C = \frac{c}{2}\).

theorem find_angle_C 
  (h : a * Real.cos B * Real.cos C + b * Real.cos A * Real.cos C = c / 2) : C = π / 3 :=
sorry

-- Proof 2: Prove the area of triangle \(ABC = \frac{3\sqrt{3}}{2}\) given \(c = \sqrt{7}\), \(a + b = 5\), and \(C = \frac{\pi}{3}\).

theorem find_area_of_triangle 
  (h1 : c = Real.sqrt 7) (h2 : a + b = 5) (h3 : C = π / 3) : 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
sorry

end find_angle_C_find_area_of_triangle_l2320_232067


namespace ship_length_in_steps_l2320_232012

theorem ship_length_in_steps (E S L : ℝ) (H1 : L + 300 * S = 300 * E) (H2 : L - 60 * S = 60 * E) :
  L = 100 * E :=
by sorry

end ship_length_in_steps_l2320_232012


namespace harmonic_mean_ordered_pairs_l2320_232076

theorem harmonic_mean_ordered_pairs :
  ∃ n : ℕ, n = 23 ∧ ∀ (a b : ℕ), 
    0 < a ∧ 0 < b ∧ a < b ∧ (2 * a * b = 2 ^ 24 * (a + b)) → n = 23 :=
by sorry

end harmonic_mean_ordered_pairs_l2320_232076


namespace cruzs_marbles_l2320_232098

theorem cruzs_marbles (Atticus Jensen Cruz : ℕ) 
  (h1 : 3 * (Atticus + Jensen + Cruz) = 60) 
  (h2 : Atticus = Jensen / 2) 
  (h3 : Atticus = 4) : 
  Cruz = 8 := 
sorry

end cruzs_marbles_l2320_232098


namespace carbonated_water_solution_l2320_232065

variable (V V_1 V_2 : ℝ)
variable (C2 : ℝ)

def carbonated_water_percent (V V1 V2 C2 : ℝ) : Prop :=
  0.8 * V1 + C2 * V2 = 0.6 * V

theorem carbonated_water_solution :
  ∀ (V : ℝ),
  (V1 = 0.1999999999999997 * V) →
  (V2 = 0.8000000000000003 * V) →
  carbonated_water_percent V V1 V2 C2 →
  C2 = 0.55 :=
by
  intros V V1_eq V2_eq carbonated_eq
  sorry

end carbonated_water_solution_l2320_232065


namespace rowing_time_from_A_to_B_and_back_l2320_232040

-- Define the problem parameters and conditions
def rowing_speed_still_water : ℝ := 5
def distance_AB : ℝ := 12
def stream_speed : ℝ := 1

-- Define the problem to prove
theorem rowing_time_from_A_to_B_and_back :
  let downstream_speed := rowing_speed_still_water + stream_speed
  let upstream_speed := rowing_speed_still_water - stream_speed
  let time_downstream := distance_AB / downstream_speed
  let time_upstream := distance_AB / upstream_speed
  let total_time := time_downstream + time_upstream
  total_time = 5 :=
by
  sorry

end rowing_time_from_A_to_B_and_back_l2320_232040


namespace sum_of_a_and_b_l2320_232047

theorem sum_of_a_and_b (a b : ℝ) (h1 : abs a = 5) (h2 : b = -2) (h3 : a * b > 0) : a + b = -7 := by
  sorry

end sum_of_a_and_b_l2320_232047


namespace apples_remaining_in_each_basket_l2320_232046

-- Definition of conditions
def total_apples : ℕ := 128
def number_of_baskets : ℕ := 8
def apples_taken_per_basket : ℕ := 7

-- Definition of the problem
theorem apples_remaining_in_each_basket :
  (total_apples / number_of_baskets) - apples_taken_per_basket = 9 := 
by 
  sorry

end apples_remaining_in_each_basket_l2320_232046


namespace larger_number_is_37_point_435_l2320_232004

theorem larger_number_is_37_point_435 (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 96) (h3 : x > y) : x = 37.435 :=
by
  sorry

end larger_number_is_37_point_435_l2320_232004


namespace greater_expected_area_vasya_l2320_232042

noncomputable def expected_area_vasya : ℚ :=
  (1/6) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

noncomputable def expected_area_asya : ℚ :=
  ((1/6) * (1 + 2 + 3 + 4 + 5 + 6)) * ((1/6) * (1 + 2 + 3 + 4 + 5 + 6))

theorem greater_expected_area_vasya : expected_area_vasya > expected_area_asya :=
  by
  -- We've provided the expected area values as definitions
  -- expected_area_vasya = 91/6
  -- vs. expected_area_asya = 12.25 = (21/6)^2 = 441/36 = 12.25
  sorry

end greater_expected_area_vasya_l2320_232042


namespace problem_min_a2_area_l2320_232095

noncomputable def area (a b c : ℝ) (A B C : ℝ) : ℝ := 
  0.5 * b * c * Real.sin A

noncomputable def min_a2_area (a b c : ℝ) (A B C : ℝ): ℝ := 
  let S := area a b c A B C
  a^2 / S

theorem problem_min_a2_area :
  ∀ (a b c A B C : ℝ), 
    a > 0 → b > 0 → c > 0 → 
    A + B + C = Real.pi →
    a / Real.sin A = b / Real.sin B ∧ a / Real.sin A = c / Real.sin C →
    b * Real.cos C + c * Real.cos B = 3 * a * Real.cos A →
    min_a2_area a b c A B C ≥ 2 * Real.sqrt 2 :=
by
  sorry

end problem_min_a2_area_l2320_232095


namespace largest_n_fact_product_of_four_consecutive_integers_l2320_232069

theorem largest_n_fact_product_of_four_consecutive_integers :
  ∀ (n : ℕ), (∃ x : ℕ, n.factorial = x * (x + 1) * (x + 2) * (x + 3)) → n ≤ 6 :=
by
  sorry

end largest_n_fact_product_of_four_consecutive_integers_l2320_232069


namespace find_income_l2320_232083

-- Define the conditions
def income_and_expenditure (income expenditure : ℕ) : Prop :=
  5 * expenditure = 3 * income

def savings (income expenditure : ℕ) (saving : ℕ) : Prop :=
  income - expenditure = saving

-- State the theorem
theorem find_income (expenditure : ℕ) (saving : ℕ) (h1 : income_and_expenditure 5 3) (h2 : savings (5 * expenditure) (3 * expenditure) saving) :
  5 * expenditure = 10000 :=
by
  -- Use the provided hint or conditions
  sorry

end find_income_l2320_232083


namespace fifth_pile_magazines_l2320_232060

theorem fifth_pile_magazines :
  let first_pile := 3
  let second_pile := first_pile + 1
  let third_pile := second_pile + 2
  let fourth_pile := third_pile + 3
  let fifth_pile := fourth_pile + (3 + 1)
  fifth_pile = 13 :=
by
  let first_pile := 3
  let second_pile := first_pile + 1
  let third_pile := second_pile + 2
  let fourth_pile := third_pile + 3
  let fifth_pile := fourth_pile + (3 + 1)
  show fifth_pile = 13
  sorry

end fifth_pile_magazines_l2320_232060


namespace seashells_solution_l2320_232087

def seashells_problem (T : ℕ) : Prop :=
  T + 13 = 50 → T = 37

theorem seashells_solution : seashells_problem 37 :=
by
  intro h
  sorry

end seashells_solution_l2320_232087


namespace ratio_of_metals_l2320_232041

theorem ratio_of_metals (G C S : ℝ) (h1 : 11 * G + 5 * C + 7 * S = 9 * (G + C + S)) : 
  G / C = 1 / 2 ∧ G / S = 1 :=
by
  sorry

end ratio_of_metals_l2320_232041


namespace factorize_expression_l2320_232061

theorem factorize_expression (x : ℝ) : -2 * x^2 + 2 * x - (1 / 2) = -2 * (x - (1 / 2))^2 :=
by
  sorry

end factorize_expression_l2320_232061


namespace eggs_in_basket_l2320_232008

theorem eggs_in_basket (x : ℕ) (h₁ : 600 / x + 1 = 600 / (x - 20)) : x = 120 :=
sorry

end eggs_in_basket_l2320_232008


namespace larry_stickers_l2320_232031

theorem larry_stickers (initial_stickers : ℕ) (lost_stickers : ℕ) (final_stickers : ℕ) 
  (initial_eq_93 : initial_stickers = 93) 
  (lost_eq_6 : lost_stickers = 6) 
  (final_eq : final_stickers = initial_stickers - lost_stickers) : 
  final_stickers = 87 := 
  by 
  -- proof goes here
  sorry

end larry_stickers_l2320_232031


namespace cost_of_machines_max_type_A_machines_l2320_232038

-- Defining the cost equations for type A and type B machines
theorem cost_of_machines (x y : ℝ) (h1 : 3 * x + 2 * y = 31) (h2 : x - y = 2) : x = 7 ∧ y = 5 :=
sorry

-- Defining the budget constraint and computing the maximum number of type A machines purchasable
theorem max_type_A_machines (m : ℕ) (h : 7 * m + 5 * (6 - m) ≤ 34) : m ≤ 2 :=
sorry

end cost_of_machines_max_type_A_machines_l2320_232038


namespace identity_true_for_any_abc_l2320_232013

theorem identity_true_for_any_abc : 
  ∀ (a b c : ℝ), (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c :=
by
  sorry

end identity_true_for_any_abc_l2320_232013


namespace find_expression_l2320_232037

theorem find_expression (a b : ℝ) (h₁ : a - b = 5) (h₂ : a * b = 2) :
  a^2 - a * b + b^2 = 27 := 
by
  sorry

end find_expression_l2320_232037


namespace part1_part2_l2320_232019

noncomputable def f (x a : ℝ) := |x - a|

theorem part1 (a m : ℝ) :
  (∀ x, f x a ≤ m ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 ∧ m = 3 :=
by
  sorry

theorem part2 (t x : ℝ) (h_t : 0 ≤ t ∧ t < 2) :
  f x 2 + t ≥ f (x + 2) 2 ↔ x ≤ (t + 2) / 2 :=
by
  sorry

end part1_part2_l2320_232019


namespace at_most_one_negative_l2320_232014

theorem at_most_one_negative (a b c : ℝ) (h1 : a + b + c ≥ 0) (h2 : abc ≤ 0) : 
  (a < 0 ∧ b >= 0 ∧ c >= 0) ∨ (a >= 0 ∧ b < 0 ∧ c >= 0) ∨ (a >= 0 ∧ b >= 0 ∧ c < 0) ∨ 
  (a >= 0 ∧ b >= 0 ∧ c >= 0) :=
sorry

end at_most_one_negative_l2320_232014


namespace hyperbola_asymptote_eq_l2320_232017

-- Define the given hyperbola equation and its asymptote
def hyperbola_eq (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / a^2) - (y^2 / 4) = 1

def asymptote_eq (a : ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, y = (1/2) * x

-- State the main theorem
theorem hyperbola_asymptote_eq :
  (∃ a : ℝ, hyperbola_eq a ∧ asymptote_eq a) →
  (∃ x y : ℝ, (x^2 / 16) - (y^2 / 4) = 1) := 
by
  sorry

end hyperbola_asymptote_eq_l2320_232017


namespace one_prime_p_10_14_l2320_232055

theorem one_prime_p_10_14 :
  ∃! (p : ℕ), Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) :=
sorry

end one_prime_p_10_14_l2320_232055


namespace hyperbola_equation_l2320_232050

theorem hyperbola_equation (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h_hyperbola : ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) 
  (h_focus : ∃ (p : ℝ × ℝ), p = (1, 0))
  (h_line_passing_focus : ∀ y, ∃ (m c : ℝ), y = -b * y + c)
  (h_parallel : ∀ x y : ℝ, b/a = -b)
  (h_perpendicular : ∀ x y : ℝ, b/a * (-b) = -1) : 
  ∀ x y : ℝ, x^2 - y^2 = 1 :=
by
  sorry

end hyperbola_equation_l2320_232050


namespace exists_f_with_f3_eq_9_forall_f_f3_le_9_l2320_232071

-- Define the real-valued function f satisfying the given conditions
variable (f : ℝ → ℝ)
variable (f_real : ∀ x : ℝ, true)  -- f is real-valued and defined for all real numbers
variable (f_mul : ∀ x y : ℝ, f (x * y) = f x * f y)  -- f(xy) = f(x)f(y)
variable (f_add : ∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y))  -- f(x+y) ≤ 2(f(x) + f(y))
variable (f_2 : f 2 = 4)  -- f(2) = 4

-- Part a
theorem exists_f_with_f3_eq_9 : ∃ f : ℝ → ℝ, (∀ x : ℝ, true) ∧ 
                              (∀ x y : ℝ, f (x * y) = f x * f y) ∧ 
                              (∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y)) ∧ 
                              (f 2 = 4) ∧ 
                              (f 3 = 9) := 
sorry

-- Part b
theorem forall_f_f3_le_9 : ∀ f : ℝ → ℝ, 
                        (∀ x : ℝ, true) → 
                        (∀ x y : ℝ, f (x * y) = f x * f y) → 
                        (∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y)) → 
                        (f 2 = 4) → 
                        (f 3 ≤ 9) := 
sorry

end exists_f_with_f3_eq_9_forall_f_f3_le_9_l2320_232071


namespace typing_speed_equation_l2320_232049

theorem typing_speed_equation (x : ℕ) (h_pos : x > 0) :
  120 / x = 180 / (x + 6) :=
sorry

end typing_speed_equation_l2320_232049


namespace completing_the_square_l2320_232097

theorem completing_the_square (x : ℝ) :
  4 * x^2 - 2 * x - 1 = 0 → (x - 1/4)^2 = 5/16 := 
by
  sorry

end completing_the_square_l2320_232097


namespace math_proof_problem_l2320_232099

/-- Given three real numbers a, b, and c such that a ≥ b ≥ 1 ≥ c ≥ 0 and a + b + c = 3.

Part (a): Prove that 2 ≤ ab + bc + ca ≤ 3.
Part (b): Prove that (24 / (a^3 + b^3 + c^3)) + (25 / (ab + bc + ca)) ≥ 14.
--/
theorem math_proof_problem (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ 1) (h3 : 1 ≥ c)
  (h4 : c ≥ 0) (h5 : a + b + c = 3) :
  (2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 3) ∧ 
  (24 / (a^3 + b^3 + c^3) + 25 / (a * b + b * c + c * a) ≥ 14) 
  :=
by
  sorry

end math_proof_problem_l2320_232099


namespace solve_system_l2320_232070

def x : ℚ := 2.7 / 13
def y : ℚ := 1.0769

theorem solve_system :
  (∃ (x' y' : ℚ), 4 * x' - 3 * y' = -2.4 ∧ 5 * x' + 6 * y' = 7.5) ↔
  (x = 2.7 / 13 ∧ y = 1.0769) :=
by
  sorry

end solve_system_l2320_232070


namespace derivative_f_l2320_232024

noncomputable def f (x : ℝ) := x * Real.cos x - Real.sin x

theorem derivative_f :
  ∀ x : ℝ, deriv f x = -x * Real.sin x :=
by
  sorry

end derivative_f_l2320_232024


namespace day_of_100th_day_of_2005_l2320_232022

-- Define the days of the week
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

open Weekday

-- Define a function to add days to a given weekday
def add_days (d: Weekday) (n: ℕ) : Weekday :=
  match d with
  | Sunday => [Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday].get? (n % 7) |>.getD Sunday
  | Monday => [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday].get? (n % 7) |>.getD Monday
  | Tuesday => [Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, Monday].get? (n % 7) |>.getD Tuesday
  | Wednesday => [Wednesday, Thursday, Friday, Saturday, Sunday, Monday, Tuesday].get? (n % 7) |>.getD Wednesday
  | Thursday => [Thursday, Friday, Saturday, Sunday, Monday, Tuesday, Wednesday].get? (n % 7) |>.getD Thursday
  | Friday => [Friday, Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday].get? (n % 7) |>.getD Friday
  | Saturday => [Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday, Friday].get? (n % 7) |>.getD Saturday

-- State the theorem
theorem day_of_100th_day_of_2005 :
  add_days Tuesday 55 = Monday :=
by sorry

end day_of_100th_day_of_2005_l2320_232022


namespace chlorine_discount_l2320_232016

theorem chlorine_discount
  (cost_chlorine : ℕ)
  (cost_soap : ℕ)
  (num_chlorine : ℕ)
  (num_soap : ℕ)
  (discount_soap : ℤ)
  (total_savings : ℤ)
  (price_chlorine : ℤ)
  (price_soap_after_discount : ℤ)
  (total_price_before_discount : ℤ)
  (total_price_after_discount : ℤ)
  (goal_discount : ℤ) :
  cost_chlorine = 10 →
  cost_soap = 16 →
  num_chlorine = 3 →
  num_soap = 5 →
  discount_soap = 25 →
  total_savings = 26 →
  price_soap_after_discount = (1 - (discount_soap / 100)) * 16 →
  total_price_before_discount = (num_chlorine * cost_chlorine) + (num_soap * cost_soap) →
  total_price_after_discount = (num_chlorine * ((100 - goal_discount) / 100) * cost_chlorine) + (num_soap * 12) →
  total_price_before_discount - total_price_after_discount = total_savings →
  goal_discount = 20 :=
by
  intros
  sorry

end chlorine_discount_l2320_232016


namespace real_part_of_z_l2320_232021

theorem real_part_of_z (z : ℂ) (h : ∃ (r : ℝ), z^2 + z = r) : z.re = -1 / 2 :=
by
  sorry

end real_part_of_z_l2320_232021


namespace lawrence_worked_hours_l2320_232073

-- Let h_M, h_T, h_F be the hours worked on Monday, Tuesday, and Friday respectively
-- Let h_W be the hours worked on Wednesday (h_W = 5.5)
-- Let h_R be the hours worked on Thursday (h_R = 5.5)
-- Let total hours worked in 5 days be 25
-- Prove that h_M + h_T + h_F = 14

theorem lawrence_worked_hours :
  ∀ (h_M h_T h_F : ℝ), h_W = 5.5 → h_R = 5.5 → (5 * 5 = 25) → 
  h_M + h_T + h_F + h_W + h_R = 25 → h_M + h_T + h_F = 14 :=
by
  intros h_M h_T h_F h_W h_R h_total h_sum
  sorry

end lawrence_worked_hours_l2320_232073


namespace pet_food_total_weight_l2320_232054

theorem pet_food_total_weight:
  let cat_food_bags := 3
  let weight_per_cat_food_bag := 3 -- pounds
  let dog_food_bags := 4 
  let weight_per_dog_food_bag := 5 -- pounds
  let bird_food_bags := 5
  let weight_per_bird_food_bag := 2 -- pounds
  let total_weight_pounds := (cat_food_bags * weight_per_cat_food_bag) + (dog_food_bags * weight_per_dog_food_bag) + (bird_food_bags * weight_per_bird_food_bag)
  let total_weight_ounces := total_weight_pounds * 16
  total_weight_ounces = 624 :=
by
  let cat_food_bags := 3
  let weight_per_cat_food_bag := 3
  let dog_food_bags := 4
  let weight_per_dog_food_bag := 5
  let bird_food_bags := 5
  let weight_per_bird_food_bag := 2
  let total_weight_pounds := (cat_food_bags * weight_per_cat_food_bag) + (dog_food_bags * weight_per_dog_food_bag) + (bird_food_bags * weight_per_bird_food_bag)
  let total_weight_ounces := total_weight_pounds * 16
  show total_weight_ounces = 624
  sorry

end pet_food_total_weight_l2320_232054


namespace sample_size_eq_100_l2320_232092

variables (frequency : ℕ) (frequency_rate : ℚ)

theorem sample_size_eq_100 (h1 : frequency = 50) (h2 : frequency_rate = 0.5) :
  frequency / frequency_rate = 100 :=
by
  sorry

end sample_size_eq_100_l2320_232092


namespace exponent_comparison_l2320_232063

theorem exponent_comparison : 1.7 ^ 0.3 > 0.9 ^ 11 := 
by sorry

end exponent_comparison_l2320_232063


namespace sum_of_reciprocals_l2320_232032

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  1/x + 1/y = 3/8 := by
  sorry

end sum_of_reciprocals_l2320_232032


namespace train_passing_time_l2320_232045

noncomputable def length_of_train : ℝ := 450
noncomputable def speed_kmh : ℝ := 80
noncomputable def length_of_station : ℝ := 300
noncomputable def speed_m_per_s : ℝ := speed_kmh * 1000 / 3600 -- Convert km/hour to m/second
noncomputable def total_distance : ℝ := length_of_train + length_of_station
noncomputable def passing_time : ℝ := total_distance / speed_m_per_s

theorem train_passing_time : abs (passing_time - 33.75) < 0.01 :=
by
  sorry

end train_passing_time_l2320_232045


namespace y_intercept_of_line_l2320_232074

theorem y_intercept_of_line (m x y b : ℝ) (h_slope : m = 4) (h_point : (x, y) = (199, 800)) (h_line : y = m * x + b) :
    b = 4 :=
by
  sorry

end y_intercept_of_line_l2320_232074


namespace multiple_of_a_age_l2320_232006

theorem multiple_of_a_age (A B M : ℝ) (h1 : A = B + 5) (h2 : A + B = 13) (h3 : M * (A + 7) = 4 * (B + 7)) : M = 2.75 :=
sorry

end multiple_of_a_age_l2320_232006


namespace range_of_k_l2320_232064

theorem range_of_k (k : ℝ) :
  ∃ x : ℝ, k * x^2 - 2 * x - 1 = 0 ↔ k ≥ -1 :=
by
  sorry

end range_of_k_l2320_232064


namespace tetrahedron_volume_formula_l2320_232011

-- Definitions used directly in the conditions
variable (a b d : ℝ) (φ : ℝ)

-- Tetrahedron volume formula theorem statement
theorem tetrahedron_volume_formula 
  (ha_pos : 0 < a) 
  (hb_pos : 0 < b) 
  (hd_pos : 0 < d) 
  (hφ_pos : 0 < φ) 
  (hφ_le_pi : φ ≤ Real.pi) :
  (∀ V : ℝ, V = 1 / 6 * a * b * d * Real.sin φ) :=
sorry

end tetrahedron_volume_formula_l2320_232011


namespace symmetric_difference_card_l2320_232000

variable (x y : Finset ℤ)
variable (h1 : x.card = 16)
variable (h2 : y.card = 18)
variable (h3 : (x ∩ y).card = 6)

theorem symmetric_difference_card :
  (x \ y ∪ y \ x).card = 22 := by sorry

end symmetric_difference_card_l2320_232000


namespace max_value_of_XYZ_XY_YZ_ZX_l2320_232086

theorem max_value_of_XYZ_XY_YZ_ZX (X Y Z : ℕ) (h : X + Y + Z = 15) : 
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 200 := 
sorry

end max_value_of_XYZ_XY_YZ_ZX_l2320_232086


namespace problem_solution_l2320_232048

noncomputable def question (x y z : ℝ) : Prop := 
  (x ≠ y ∧ y ≠ z ∧ z ≠ x) → 
  ((x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∨ 
   (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ∨ 
   (z + x)/(z^2 + z*x + x^2) = (x + y)/(x^2 + x*y + y^2)) → 
  ( (x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∧ 
    (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) )

theorem problem_solution (x y z : ℝ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ((x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∨ 
   (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ∨ 
   (z + x)/(z^2 + z*x + x^2) = (x + y)/(x^2 + x*y + y^2)) →
  ( (x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∧ 
    (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ) :=
sorry

end problem_solution_l2320_232048


namespace consecutive_integers_l2320_232009

theorem consecutive_integers (a b c : ℝ)
  (h1 : ∃ k : ℤ, a + b = k ∧ b + c = k + 1 ∧ c + a = k + 2)
  (h2 : ∃ k : ℤ, b + c = 2 * k + 1) :
  ∃ n : ℤ, a = n + 2 ∧ b = n + 1 ∧ c = n := 
sorry

end consecutive_integers_l2320_232009


namespace ratio_of_areas_of_squares_l2320_232066

theorem ratio_of_areas_of_squares (side_C side_D : ℕ) 
  (hC : side_C = 48) (hD : side_D = 60) : 
  (side_C^2 : ℚ)/(side_D^2 : ℚ) = 16/25 :=
by
  -- sorry, proof omitted
  sorry

end ratio_of_areas_of_squares_l2320_232066


namespace koschei_never_equal_l2320_232039

-- Define the problem setup 
def coins_at_vertices (n1 n2 n3 n4 n5 n6 : ℕ) : Prop := 
  ∃ k : ℕ, n1 = k ∧ n2 = k ∧ n3 = k ∧ n4 = k ∧ n5 = k ∧ n6 = k

-- Define the operation condition
def operation_condition (n1 n2 n3 n4 n5 n6 : ℕ) : Prop :=
  ∃ x : ℕ, (n1 - x = x ∧ n2 + 6 * x = x) ∨ (n2 - x = x ∧ n3 + 6 * x = x) ∨ 
  (n3 - x = x ∧ n4 + 6 * x = x) ∨ (n4 - x = x ∧ n5 + 6 * x = x) ∨ 
  (n5 - x = x ∧ n6 + 6 * x = x) ∨ (n6 - x = x ∧ n1 + 6 * x = x)

-- The main theorem 
theorem koschei_never_equal (n1 n2 n3 n4 n5 n6 : ℕ) : 
  (∃ x : ℕ, coins_at_vertices n1 n2 n3 n4 n5 n6) → False :=
by
  sorry

end koschei_never_equal_l2320_232039


namespace additional_carpet_needed_l2320_232088

-- Define the given conditions as part of the hypothesis:
def carpetArea : ℕ := 18
def roomLength : ℕ := 4
def roomWidth : ℕ := 20

-- The theorem we want to prove:
theorem additional_carpet_needed : (roomLength * roomWidth - carpetArea) = 62 := by
  sorry

end additional_carpet_needed_l2320_232088


namespace sum_product_le_four_l2320_232089

theorem sum_product_le_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := 
sorry

end sum_product_le_four_l2320_232089


namespace tax_deduction_cents_l2320_232077

def bob_hourly_wage : ℝ := 25
def tax_rate : ℝ := 0.025

theorem tax_deduction_cents :
  (bob_hourly_wage * 100 * tax_rate) = 62.5 :=
by
  -- This is the statement that needs to be proven.
  sorry

end tax_deduction_cents_l2320_232077


namespace find_starting_number_l2320_232025

-- Define that there are 15 even integers between a starting number and 40
def even_integers_range (n : ℕ) : Prop :=
  ∃ k : ℕ, (1 ≤ k) ∧ (k = 15) ∧ (n + 2*(k-1) = 40)

-- Proof statement
theorem find_starting_number : ∃ n : ℕ, even_integers_range n ∧ n = 12 :=
by
  sorry

end find_starting_number_l2320_232025


namespace value_of_smaller_denom_l2320_232085

-- We are setting up the conditions given in the problem.
variables (x : ℕ) -- The value of the smaller denomination bill.

-- Condition 1: She has 4 bills of denomination x.
def value_smaller_denomination : ℕ := 4 * x

-- Condition 2: She has 8 bills of $10 denomination.
def value_ten_bills : ℕ := 8 * 10

-- Condition 3: The total value of the bills is $100.
def total_value : ℕ := 100

-- Prove that x = 5 using the given conditions.
theorem value_of_smaller_denom : value_smaller_denomination x + value_ten_bills = total_value → x = 5 :=
by
  intro h
  -- Proof steps would go here
  sorry

end value_of_smaller_denom_l2320_232085


namespace parabola_intersection_min_y1_y2_sqr_l2320_232091

theorem parabola_intersection_min_y1_y2_sqr :
  ∀ (x1 x2 y1 y2 : ℝ)
    (h1 : y1 ^ 2 = 4 * x1)
    (h2 : y2 ^ 2 = 4 * x2)
    (h3 : (∃ k : ℝ, x1 = 4 ∧ y1 = k * (4 - 4)) ∨ x1 = 4 ∧ y1 ≠ x2),
    ∃ m : ℝ, (y1^2 + y2^2) = m ∧ m = 32 := 
sorry

end parabola_intersection_min_y1_y2_sqr_l2320_232091


namespace fred_found_43_seashells_l2320_232084

-- Define the conditions
def tom_seashells : ℕ := 15
def additional_seashells : ℕ := 28

-- Define Fred's total seashells based on the conditions
def fred_seashells : ℕ := tom_seashells + additional_seashells

-- The theorem to prove that Fred found 43 seashells
theorem fred_found_43_seashells : fred_seashells = 43 :=
by
  -- Proof goes here
  sorry

end fred_found_43_seashells_l2320_232084


namespace rooks_same_distance_l2320_232081

theorem rooks_same_distance (rooks : Fin 8 → (ℕ × ℕ)) 
    (h_non_attacking : ∀ i j, i ≠ j → Prod.fst (rooks i) ≠ Prod.fst (rooks j) ∧ Prod.snd (rooks i) ≠ Prod.snd (rooks j)) 
    : ∃ i j k l, i ≠ j ∧ k ≠ l ∧ (Prod.fst (rooks i) - Prod.fst (rooks k))^2 + (Prod.snd (rooks i) - Prod.snd (rooks k))^2 = (Prod.fst (rooks j) - Prod.fst (rooks l))^2 + (Prod.snd (rooks j) - Prod.snd (rooks l))^2 :=
by 
  -- Proof goes here
  sorry

end rooks_same_distance_l2320_232081


namespace hall_of_mirrors_l2320_232051

theorem hall_of_mirrors (h : ℝ) 
    (condition1 : 2 * (30 * h) + (20 * h) = 960) :
  h = 12 :=
by
  sorry

end hall_of_mirrors_l2320_232051


namespace least_four_digit_divisible_by_15_25_40_75_is_1200_l2320_232080

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

def divisible_by_25 (n : ℕ) : Prop :=
  n % 25 = 0

def divisible_by_40 (n : ℕ) : Prop :=
  n % 40 = 0

def divisible_by_75 (n : ℕ) : Prop :=
  n % 75 = 0

theorem least_four_digit_divisible_by_15_25_40_75_is_1200 :
  ∃ n : ℕ, is_four_digit n ∧ divisible_by_15 n ∧ divisible_by_25 n ∧ divisible_by_40 n ∧ divisible_by_75 n ∧
  (∀ m : ℕ, is_four_digit m ∧ divisible_by_15 m ∧ divisible_by_25 m ∧ divisible_by_40 m ∧ divisible_by_75 m → n ≤ m) ∧
  n = 1200 := 
sorry

end least_four_digit_divisible_by_15_25_40_75_is_1200_l2320_232080


namespace three_squares_not_divisible_by_three_l2320_232043

theorem three_squares_not_divisible_by_three 
  (N : ℕ) (a b c : ℤ) 
  (h₁ : N = 9 * (a^2 + b^2 + c^2)) :
  ∃ x y z : ℤ, N = x^2 + y^2 + z^2 ∧ ¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z) := 
sorry

end three_squares_not_divisible_by_three_l2320_232043


namespace distinct_solutions_subtract_eight_l2320_232090

noncomputable def f (x : ℝ) : ℝ := (6 * x - 18) / (x^2 + 2 * x - 15)
noncomputable def equation := ∀ x, f x = x + 3

noncomputable def r_solutions (r s : ℝ) := (r > s) ∧ (f r = r + 3) ∧ (f s = s + 3)

theorem distinct_solutions_subtract_eight
  (r s : ℝ) (h : r_solutions r s) : r - s = 8 :=
sorry

end distinct_solutions_subtract_eight_l2320_232090


namespace find_boxes_l2320_232035

variable (John Jules Joseph Stan : ℕ)

-- Conditions
axiom h1 : John = 30
axiom h2 : John = 6 * Jules / 5 -- Equivalent to John having 20% more boxes than Jules
axiom h3 : Jules = Joseph + 5
axiom h4 : Joseph = Stan / 5 -- Equivalent to Joseph having 80% fewer boxes than Stan

-- Theorem to prove
theorem find_boxes (h1 : John = 30) (h2 : John = 6 * Jules / 5) (h3 : Jules = Joseph + 5) (h4 : Joseph = Stan / 5) : Stan = 100 :=
sorry

end find_boxes_l2320_232035


namespace sufficient_not_necessary_l2320_232082

theorem sufficient_not_necessary (x y : ℝ) : (x > |y|) → (x > y ∧ ¬ (x > y → x > |y|)) :=
by
  sorry

end sufficient_not_necessary_l2320_232082


namespace regular_seminar_fee_l2320_232018

-- Define the main problem statement
theorem regular_seminar_fee 
  (F : ℝ) 
  (discount_per_teacher : ℝ) 
  (number_of_teachers : ℕ)
  (food_allowance_per_teacher : ℝ)
  (total_spent : ℝ) :
  discount_per_teacher = 0.95 * F →
  number_of_teachers = 10 →
  food_allowance_per_teacher = 10 →
  total_spent = 1525 →
  (number_of_teachers * discount_per_teacher + number_of_teachers * food_allowance_per_teacher = total_spent) →
  F = 150 := 
  by sorry

end regular_seminar_fee_l2320_232018


namespace max_operations_l2320_232056

def arithmetic_mean (a b : ℕ) := (a + b) / 2

theorem max_operations (b : ℕ) (hb : b < 2002) (heven : (2002 + b) % 2 = 0) :
  ∃ n, n = 10 ∧ (2002 - b) / 2^n = 1 :=
by
  sorry

end max_operations_l2320_232056


namespace total_rattlesnakes_l2320_232029

-- Definitions based on the problem's conditions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def other_snakes : ℕ := total_snakes - (pythons + boa_constrictors)

-- Statement to be proved
theorem total_rattlesnakes : other_snakes = 40 := 
by 
  -- Skipping the proof
  sorry

end total_rattlesnakes_l2320_232029


namespace intersection_of_M_and_N_l2320_232096

-- Define the sets M and N with the given conditions
def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem that the intersection of M and N is as described
theorem intersection_of_M_and_N : (M ∩ N) = {x : ℝ | -1 < x ∧ x < 1} :=
by
  -- the proof will go here
  sorry

end intersection_of_M_and_N_l2320_232096
