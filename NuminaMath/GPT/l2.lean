import Mathlib

namespace jerry_birthday_games_l2_2973

def jerry_original_games : ℕ := 7
def jerry_total_games_after_birthday : ℕ := 9
def games_jerry_got_for_birthday (original total : ℕ) : ℕ := total - original

theorem jerry_birthday_games :
  games_jerry_got_for_birthday jerry_original_games jerry_total_games_after_birthday = 2 := by
  sorry

end jerry_birthday_games_l2_2973


namespace value_of_g_at_3_l2_2766

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := 5 * x^3 - 6 * x^2 - 3 * x + 5

-- The theorem statement
theorem value_of_g_at_3 : g 3 = 77 := by
  -- This would require a proof, but we put sorry as instructed
  sorry

end value_of_g_at_3_l2_2766


namespace literature_books_cost_more_l2_2673

theorem literature_books_cost_more :
  let num_books := 45
  let literature_cost_per_book := 7
  let technology_cost_per_book := 5
  (num_books * literature_cost_per_book) - (num_books * technology_cost_per_book) = 90 :=
by
  sorry

end literature_books_cost_more_l2_2673


namespace max_slope_avoiding_lattice_points_l2_2284

theorem max_slope_avoiding_lattice_points :
  ∃ a : ℝ, (1 < a ∧ ∀ m : ℝ, (1 < m ∧ m < a) → (∀ x : ℤ, (10 < x ∧ x ≤ 200) → ∃ k : ℝ, y = m * x + 5 ∧ (m * x + 5 ≠ k))) ∧ a = 101 / 100 :=
sorry

end max_slope_avoiding_lattice_points_l2_2284


namespace scientific_notation_of_population_l2_2504

theorem scientific_notation_of_population :
  (141260 : ℝ) = 1.4126 * 10^5 :=
sorry

end scientific_notation_of_population_l2_2504


namespace convert_quadratic_to_general_form_l2_2285

theorem convert_quadratic_to_general_form
  (x : ℝ)
  (h : 3 * x * (x - 3) = 4) :
  3 * x ^ 2 - 9 * x - 4 = 0 :=
by
  sorry

end convert_quadratic_to_general_form_l2_2285


namespace distance_traveled_l2_2113

-- Definition of the velocity function
def velocity (t : ℝ) : ℝ := 2 * t - 3

-- Prove the integral statement
theorem distance_traveled : 
  (∫ t in (0 : ℝ)..(5 : ℝ), abs (velocity t)) = 29 / 2 := by 
{ sorry }

end distance_traveled_l2_2113


namespace smallest_four_digit_multiple_of_17_l2_2376

theorem smallest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n ∧ n < 10000 ∧ n % 17 = 0) ∧ ∀ m, (1000 ≤ m ∧ m < 10000 ∧ m % 17 = 0 → n ≤ m) ∧ n = 1013 :=
by
  sorry

end smallest_four_digit_multiple_of_17_l2_2376


namespace distance_from_point_to_hyperbola_asymptote_l2_2997

noncomputable def distance_to_asymptote (x1 y1 a b : ℝ) : ℝ :=
  abs (a * x1 + b * y1) / real.sqrt (a ^ 2 + b ^ 2)

theorem distance_from_point_to_hyperbola_asymptote :
  distance_to_asymptote 3 0 3 (-4) = 9 / 5 :=
by
  sorry

end distance_from_point_to_hyperbola_asymptote_l2_2997


namespace cost_D_to_E_l2_2775

def distance_DF (DF DE EF : ℝ) : Prop :=
  DE^2 = DF^2 + EF^2

def cost_to_fly (distance : ℝ) (per_kilometer_cost booking_fee : ℝ) : ℝ :=
  distance * per_kilometer_cost + booking_fee

noncomputable def total_cost_to_fly_from_D_to_E : ℝ :=
  let DE := 3750 -- Distance from D to E (km)
  let booking_fee := 120 -- Booking fee in dollars
  let per_kilometer_cost := 0.12 -- Cost per kilometer in dollars
  cost_to_fly DE per_kilometer_cost booking_fee

theorem cost_D_to_E : total_cost_to_fly_from_D_to_E = 570 := by
  sorry

end cost_D_to_E_l2_2775


namespace solutions_periodic_with_same_period_l2_2173

variable {y z : ℝ → ℝ}
variable (f g : ℝ → ℝ)

-- defining the conditions
variable (h1 : ∀ x, deriv y x = - (z x)^3)
variable (h2 : ∀ x, deriv z x = (y x)^3)
variable (h3 : y 0 = 1)
variable (h4 : z 0 = 0)
variable (h5 : ∀ x, y x = f x)
variable (h6 : ∀ x, z x = g x)

-- proving periodicity
theorem solutions_periodic_with_same_period : ∃ k > 0, (∀ x, f (x + k) = f x ∧ g (x + k) = g x) := by
  sorry

end solutions_periodic_with_same_period_l2_2173


namespace cos_17pi_over_4_l2_2583

theorem cos_17pi_over_4 : Real.cos (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_17pi_over_4_l2_2583


namespace opposite_of_3_l2_2849

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l2_2849


namespace problem1_solution_set_problem2_range_of_m_l2_2321

def f (x : ℝ) : ℝ := |x - 3| - 5
def g (x : ℝ) : ℝ := |x + 2| - 2

theorem problem1_solution_set :
  {x : ℝ | f x ≤ 2} = {x : ℝ | -4 ≤ x ∧ x ≤ 10} := 
sorry

theorem problem2_range_of_m (m : ℝ) (h : ∃ x : ℝ, f x - g x ≥ m - 3) :
  m ≤ 5 :=
sorry

end problem1_solution_set_problem2_range_of_m_l2_2321


namespace quadrilateral_diagonal_length_l2_2934

theorem quadrilateral_diagonal_length (d : ℝ) 
  (h_offsets : true) 
  (area_quadrilateral : 195 = ((1 / 2) * d * 9) + ((1 / 2) * d * 6)) : 
  d = 26 :=
by 
  sorry

end quadrilateral_diagonal_length_l2_2934


namespace Seulgi_second_round_need_l2_2181

def Hohyeon_first_round := 23
def Hohyeon_second_round := 28
def Hyunjeong_first_round := 32
def Hyunjeong_second_round := 17
def Seulgi_first_round := 27

def Hohyeon_total := Hohyeon_first_round + Hohyeon_second_round
def Hyunjeong_total := Hyunjeong_first_round + Hyunjeong_second_round

def required_total_for_Seulgi := Hohyeon_total + 1

theorem Seulgi_second_round_need (Seulgi_second_round: ℕ) :
  Seulgi_first_round + Seulgi_second_round ≥ required_total_for_Seulgi → Seulgi_second_round ≥ 25 :=
by
  sorry

end Seulgi_second_round_need_l2_2181


namespace degree_meas_supp_compl_35_l2_2093

noncomputable def degree_meas_supplement_complement (θ : ℝ) : ℝ :=
  180 - (90 - θ)

theorem degree_meas_supp_compl_35 : degree_meas_supplement_complement 35 = 125 :=
by
  unfold degree_meas_supplement_complement
  norm_num
  sorry

end degree_meas_supp_compl_35_l2_2093


namespace first_tier_price_level_is_10000_l2_2255

noncomputable def first_tier_price_level (P : ℝ) : Prop :=
  ∀ (car_price : ℝ), car_price = 30000 → (P ≤ car_price ∧ 
    (0.25 * P + 0.15 * (car_price - P)) = 5500)

theorem first_tier_price_level_is_10000 :
  first_tier_price_level 10000 :=
by
  sorry

end first_tier_price_level_is_10000_l2_2255


namespace problem1_monotonic_decreasing_problem2_monotonic_decreasing_pos_problem2_monotonic_decreasing_neg_l2_2617

-- Problem 1: Monotonicity of f(x) = 1 - 3x on ℝ
theorem problem1_monotonic_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → (1 - 3 * x1) > (1 - 3 * x2) :=
by
  -- Proof (skipped)
  sorry

-- Problem 2: Monotonicity of g(x) = 1/x + 2 on (0, ∞) and (-∞, 0)
theorem problem2_monotonic_decreasing_pos : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → (1 / x1 + 2) > (1 / x2 + 2) :=
by
  -- Proof (skipped)
  sorry

theorem problem2_monotonic_decreasing_neg : ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → (1 / x1 + 2) > (1 / x2 + 2) :=
by
  -- Proof (skipped)
  sorry

end problem1_monotonic_decreasing_problem2_monotonic_decreasing_pos_problem2_monotonic_decreasing_neg_l2_2617


namespace billboard_shorter_side_length_l2_2332

theorem billboard_shorter_side_length
  (L W : ℝ)
  (h1 : L * W = 120)
  (h2 : 2 * L + 2 * W = 46) :
  min L W = 8 :=
by
  sorry

end billboard_shorter_side_length_l2_2332


namespace complex_cube_root_identity_l2_2043

theorem complex_cube_root_identity (a b c : ℂ) (ω : ℂ)
  (h1 : ω^3 = 1)
  (h2 : 1 + ω + ω^2 = 0) :
  (a + b * ω + c * ω^2) * (a + b * ω^2 + c * ω) = a^2 + b^2 + c^2 - ab - ac - bc :=
by
  sorry

end complex_cube_root_identity_l2_2043


namespace opposite_of_3_is_neg3_l2_2827

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2827


namespace car_speed_reduction_and_increase_l2_2880

theorem car_speed_reduction_and_increase (V x : ℝ)
  (h1 : V > 0) -- V is positive
  (h2 : V * (1 - x / 100) * (1 + 0.5 * x / 100) = V * (1 - 0.6 * x / 100)) :
  x = 20 :=
sorry

end car_speed_reduction_and_increase_l2_2880


namespace ratio_water_duck_to_pig_l2_2417

theorem ratio_water_duck_to_pig :
  let gallons_per_minute := 3
  let pumping_minutes := 25
  let total_gallons := gallons_per_minute * pumping_minutes
  let corn_rows := 4
  let plants_per_row := 15
  let gallons_per_corn_plant := 0.5
  let total_corn_plants := corn_rows * plants_per_row
  let total_corn_water := total_corn_plants * gallons_per_corn_plant
  let pig_count := 10
  let gallons_per_pig := 4
  let total_pig_water := pig_count * gallons_per_pig
  let duck_count := 20
  let total_duck_water := total_gallons - total_corn_water - total_pig_water
  let gallons_per_duck := total_duck_water / duck_count
  let ratio := gallons_per_duck / gallons_per_pig
  ratio = 1 / 16 := 
by
  sorry

end ratio_water_duck_to_pig_l2_2417


namespace proof_problem_l2_2950

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

theorem proof_problem :
  (∃ A ω φ, (A = 2) ∧ (ω = 2) ∧ (φ = Real.pi / 4) ∧
  f (3 * Real.pi / 8) = 0 ∧
  f (Real.pi / 8) = 2 ∧
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → f x ≤ 2) ∧
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → f x ≥ -Real.sqrt 2) ∧
  f (-Real.pi / 4) = -Real.sqrt 2) :=
sorry

end proof_problem_l2_2950


namespace club_truncator_more_wins_than_losses_l2_2283

noncomputable def club_truncator_probability : ℚ :=
  let total_games := 8
  let win_prob := 1/3
  let lose_prob := 1/3
  let tie_prob := 1/3
  -- The probability given by the solution
  let final_probability := 2741 / 6561
  final_probability

theorem club_truncator_more_wins_than_losses :
  club_truncator_probability = 2741 / 6561 :=
sorry

end club_truncator_more_wins_than_losses_l2_2283


namespace reflect_over_x_axis_l2_2531

def coords (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem reflect_over_x_axis :
  coords (-6, -9) = (-6, 9) :=
by
  sorry

end reflect_over_x_axis_l2_2531


namespace tan_7pi_over_4_eq_neg1_l2_2296

theorem tan_7pi_over_4_eq_neg1 : Real.tan (7 * Real.pi / 4) = -1 :=
  sorry

end tan_7pi_over_4_eq_neg1_l2_2296


namespace initial_water_amount_l2_2266

theorem initial_water_amount (W : ℝ) (h1 : ∀ t, t = 50 -> 0.008 * t = 0.4) (h2 : 0.04 * W = 0.4) : W = 10 :=
by
  sorry

end initial_water_amount_l2_2266


namespace determinant_real_root_unique_l2_2042

theorem determinant_real_root_unique {a b c : ℝ} (ha : 0 < a ∧ a ≠ 1) (hb : 0 < b ∧ b ≠ 1) (hc : 0 < c ∧ c ≠ 1) :
  ∃! x : ℝ, (Matrix.det ![
    ![x - 1, c - 1, -(b - 1)],
    ![-(c - 1), x - 1, a - 1],
    ![b - 1, -(a - 1), x - 1]
  ]) = 0 :=
by
  sorry

end determinant_real_root_unique_l2_2042


namespace shares_owned_l2_2396

theorem shares_owned (expected_earnings dividend_ratio additional_per_10c actual_earnings total_dividend : ℝ)
  ( h1 : expected_earnings = 0.80 )
  ( h2 : dividend_ratio = 0.50 )
  ( h3 : additional_per_10c = 0.04 )
  ( h4 : actual_earnings = 1.10 )
  ( h5 : total_dividend = 156.0 ) :
  ∃ shares : ℝ, shares = total_dividend / (expected_earnings * dividend_ratio + (max ((actual_earnings - expected_earnings) / 0.10) 0) * additional_per_10c) ∧ shares = 300 := 
sorry

end shares_owned_l2_2396


namespace set_C_is_correct_l2_2187

open Set

noncomputable def set_A : Set ℝ := {x | x ^ 2 - x - 12 ≤ 0}
noncomputable def set_B : Set ℝ := {x | (x + 1) / (x - 1) < 0}
noncomputable def set_C : Set ℝ := {x | x ∈ set_A ∧ x ∉ set_B}

theorem set_C_is_correct : set_C = {x | -3 ≤ x ∧ x ≤ -1} ∪ {x | 1 ≤ x ∧ x ≤ 4} :=
by
  sorry

end set_C_is_correct_l2_2187


namespace perpendicular_k_parallel_k_l2_2017

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Define the scalar multiple operations and vector operations
def smul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ := (v₁.1 + v₂.1, v₂.2 + v₂.2)
def sub (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ := (v₁.1 - v₂.1, v₂.2 - v₂.2)
def dot (v₁ v₂ : ℝ × ℝ) : ℝ := (v₁.1 * v₂.1 + v₁.2 * v₂.2)

-- Problem 1: If k*a + b is perpendicular to a - 3*b, then k = 19
theorem perpendicular_k (k : ℝ) :
  let vak := add (smul k a) b
  let amb := sub a (smul 3 b)
  dot vak amb = 0 → k = 19 := sorry

-- Problem 2: If k*a + b is parallel to a - 3*b, then k = -1/3 and they are in opposite directions
theorem parallel_k (k : ℝ) :
  let vak := add (smul k a) b
  let amb := sub a (smul 3 b)
  ∃ m : ℝ, vak = smul m amb ∧ m < 0 → k = -1/3 := sorry

end perpendicular_k_parallel_k_l2_2017


namespace juice_left_l2_2881

theorem juice_left (total consumed : ℚ) (h_total : total = 1) (h_consumed : consumed = 4 / 6) :
  total - consumed = 2 / 6 ∨ total - consumed = 1 / 3 :=
by
  sorry

end juice_left_l2_2881


namespace range_of_x_l2_2014

theorem range_of_x (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : x + y + z = 1) (h4 : x^2 + y^2 + z^2 = 3) : 1 ≤ x ∧ x ≤ 5 / 3 :=
by
  sorry

end range_of_x_l2_2014


namespace symmetric_circle_equation_l2_2532

-- Define the original circle and the line of symmetry
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2
def line_of_symmetry (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Proving the equation of the symmetric circle
theorem symmetric_circle_equation :
  (∀ x y : ℝ, original_circle x y ↔ (x + 3)^2 + (y - 2)^2 = 2) :=
by
  sorry

end symmetric_circle_equation_l2_2532


namespace cost_of_blue_cap_l2_2282

theorem cost_of_blue_cap (cost_tshirt cost_backpack cost_cap total_spent discount: ℝ) 
  (h1 : cost_tshirt = 30) 
  (h2 : cost_backpack = 10) 
  (h3 : discount = 2)
  (h4 : total_spent = 43) 
  (h5 : total_spent = cost_tshirt + cost_backpack + cost_cap - discount) : 
  cost_cap = 5 :=
by sorry

end cost_of_blue_cap_l2_2282


namespace number_of_neighborhoods_l2_2570

def street_lights_per_side : ℕ := 250
def roads_per_neighborhood : ℕ := 4
def total_street_lights : ℕ := 20000

theorem number_of_neighborhoods : 
  (total_street_lights / (2 * street_lights_per_side * roads_per_neighborhood)) = 10 :=
by
  -- proof to show that the number of neighborhoods is 10
  sorry

end number_of_neighborhoods_l2_2570


namespace cos_pi_over_6_minus_a_eq_5_over_12_l2_2441

theorem cos_pi_over_6_minus_a_eq_5_over_12 (a : ℝ) (h : Real.sin (Real.pi / 3 + a) = 5 / 12) :
  Real.cos (Real.pi / 6 - a) = 5 / 12 :=
by
  sorry

end cos_pi_over_6_minus_a_eq_5_over_12_l2_2441


namespace polar_conversion_equiv_l2_2030

noncomputable def polar_convert (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
if r < 0 then (-r, θ + Real.pi) else (r, θ)

theorem polar_conversion_equiv : polar_convert (-3) (Real.pi / 4) = (3, 5 * Real.pi / 4) :=
by
  sorry

end polar_conversion_equiv_l2_2030


namespace investment_of_D_l2_2894

/--
Given C and D started a business where C invested Rs. 1000 and D invested some amount.
They made a total profit of Rs. 500, and D's share of the profit is Rs. 100.
So, how much did D invest in the business?
-/
theorem investment_of_D 
  (C_invested : ℕ) (D_share : ℕ) (total_profit : ℕ) 
  (H1 : C_invested = 1000) 
  (H2 : D_share = 100) 
  (H3 : total_profit = 500) 
  : ∃ D : ℕ, D = 250 :=
by
  sorry

end investment_of_D_l2_2894


namespace ellipse_problem_l2_2597

theorem ellipse_problem : 
  let a := 2
  let b := sqrt 3
  let e := 1 / 2
  let M := {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}
  ∃ P : ℝ × ℝ, P = (4, 0) ∧ 
  ∀ A B C : ℝ × ℝ, 
    A ∉ {v : ℝ × ℝ | v.1^2 / a^2 + v.2^2 / b^2 = 1 ∧ v.1 ≠ 0 ∧ v.2 ≠ 0}
    ∧ C = (A.1, -A.2)
    ∧ B ∈ {v : ℝ × ℝ | (C.1 - 2) * (v.2 + C.2) = (C.2 + 2) * (v.1 - 2)}
    → 
      let PA := (A.1 - 4, A.2)
      let F2C := (A.1 - 1, -A.2)
      (range x1 : ℝ, -2 < x1 ∧ x1 < 2) 
      (range x1, (7/4 * (x1 - 10/7)^2 - 18/7)) :=
sorry

end ellipse_problem_l2_2597


namespace isosceles_triangle_altitude_l2_2776

open Real

theorem isosceles_triangle_altitude (DE DF DG EG GF EF : ℝ) (h1 : DE = 5) (h2 : DF = 5) (h3 : EG = 2 * GF)
(h4 : DG = sqrt (DE^2 - GF^2)) (h5 : EF = EG + GF) (h6 : EF = 3 * GF) : EF = 5 :=
by
  -- Proof would go here
  sorry

end isosceles_triangle_altitude_l2_2776


namespace opposite_of_3_l2_2838

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l2_2838


namespace vacation_costs_l2_2725

variable (Anne_paid Beth_paid Carlos_paid : ℕ) (a b : ℕ)

theorem vacation_costs (hAnne : Anne_paid = 120) (hBeth : Beth_paid = 180) (hCarlos : Carlos_paid = 150)
  (h_a : a = 30) (h_b : b = 30) :
  a - b = 0 := sorry

end vacation_costs_l2_2725


namespace solution_to_system_l2_2053

theorem solution_to_system :
  ∀ (x y z : ℝ), 
  x * (3 * y^2 + 1) = y * (y^2 + 3) →
  y * (3 * z^2 + 1) = z * (z^2 + 3) →
  z * (3 * x^2 + 1) = x * (x^2 + 3) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = -1 ∧ y = -1 ∧ z = -1) :=
by
  sorry

end solution_to_system_l2_2053


namespace sin_half_angle_identity_l2_2747

theorem sin_half_angle_identity (theta : ℝ) (h : Real.sin (Real.pi / 2 + theta) = - 1 / 2) :
  2 * Real.sin (theta / 2) ^ 2 - 1 = 1 / 2 := 
by
  sorry

end sin_half_angle_identity_l2_2747


namespace total_cost_second_set_l2_2263

variable (A V : ℝ)

-- Condition declarations
axiom cost_video_cassette : V = 300
axiom cost_second_set : 7 * A + 3 * V = 1110

-- Proof goal
theorem total_cost_second_set :
  7 * A + 3 * V = 1110 :=
by
  sorry

end total_cost_second_set_l2_2263


namespace solve_circle_tangent_and_intercept_l2_2175

namespace CircleProblems

-- Condition: Circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + 3 = 0

-- Problem 1: Equations of tangent lines with equal intercepts
def tangent_lines_with_equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, l x y ↔ x + y + 1 = 0) ∨ (∀ x y : ℝ, l x y ↔ x + y - 3 = 0)

-- Problem 2: Equations of lines passing through origin and intercepted by the circle with a segment length of 2
def lines_intercepted_by_circle (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, l x y ↔ x = 0) ∨ (∀ x y : ℝ, l x y ↔ y = - (3 / 4) * x)

theorem solve_circle_tangent_and_intercept (l_tangent l_origin : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, circle_eq x y → l_tangent x y) →
  tangent_lines_with_equal_intercepts l_tangent ∧ lines_intercepted_by_circle l_origin :=
by
  sorry

end CircleProblems

end solve_circle_tangent_and_intercept_l2_2175


namespace scientific_notation_of_population_l2_2506

theorem scientific_notation_of_population :
  (141260 : ℝ) = 1.4126 * 10^5 :=
sorry

end scientific_notation_of_population_l2_2506


namespace find_base_k_l2_2701

theorem find_base_k : ∃ k : ℕ, 6 * k^2 + 6 * k + 4 = 340 ∧ k = 7 := 
by 
  sorry

end find_base_k_l2_2701


namespace cylinder_height_relation_l2_2317

theorem cylinder_height_relation (r1 r2 h1 h2 V1 V2 : ℝ) 
  (h_volumes_equal : V1 = V2)
  (h_r2_gt_r1 : r2 = 1.1 * r1)
  (h_volume_first : V1 = π * r1^2 * h1)
  (h_volume_second : V2 = π * r2^2 * h2) : 
  h1 = 1.21 * h2 :=
by 
  sorry

end cylinder_height_relation_l2_2317


namespace find_a10_of_arithmetic_sequence_l2_2600

theorem find_a10_of_arithmetic_sequence (a : ℕ → ℚ)
  (h_seq : ∀ n : ℕ, ∃ d : ℚ, ∀ m : ℕ, a (n + m + 1) = a (n + m) + d)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 4) :
  a 10 = -4 / 5 :=
sorry

end find_a10_of_arithmetic_sequence_l2_2600


namespace determine_m_range_l2_2023

variable {R : Type} [OrderedCommGroup R]

-- Define the odd function f: ℝ → ℝ
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define the increasing function f: ℝ → ℝ
def increasing_function (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y

-- Define the main theorem
theorem determine_m_range (f : ℝ → ℝ) (odd_f : odd_function f) (inc_f : increasing_function f) :
    (∀ θ : ℝ, f (Real.cos (2 * θ) - 5) + f (2 * m + 4 * Real.sin θ) > 0) → m > 5 :=
by
  sorry

end determine_m_range_l2_2023


namespace brian_total_commission_l2_2579

theorem brian_total_commission :
  let commission_rate := 0.02
  let house1 := 157000
  let house2 := 499000
  let house3 := 125000
  let total_sales := house1 + house2 + house3
  let total_commission := total_sales * commission_rate
  total_commission = 15620 := by
{
  sorry
}

end brian_total_commission_l2_2579


namespace max_5x_plus_3y_l2_2172

theorem max_5x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 8 * y + 10) : 5 * x + 3 * y ≤ 105 :=
sorry

end max_5x_plus_3y_l2_2172


namespace final_number_is_correct_l2_2565

def initial_number := 9
def doubled_number (x : ℕ) := x * 2
def added_number (x : ℕ) := x + 13
def trebled_number (x : ℕ) := x * 3

theorem final_number_is_correct : trebled_number (added_number (doubled_number initial_number)) = 93 := by
  sorry

end final_number_is_correct_l2_2565


namespace expression_simplification_l2_2914

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l2_2914


namespace find_value_of_expression_l2_2201

theorem find_value_of_expression (x y z : ℝ)
  (h1 : 12 * x - 9 * y^2 = 7)
  (h2 : 6 * y - 9 * z^2 = -2)
  (h3 : 12 * z - 9 * x^2 = 4) : 
  6 * x^2 + 9 * y^2 + 12 * z^2 = 9 :=
  sorry

end find_value_of_expression_l2_2201


namespace k_value_range_l2_2606

-- Definitions
def f (x : ℝ) (k : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- The theorem we are interested in
theorem k_value_range (k : ℝ) (h : ∀ x₁ x₂ : ℝ, (x₁ > 5 → x₂ > 5 → f x₁ k ≤ f x₂ k) ∨ (x₁ > 5 → x₂ > 5 → f x₁ k ≥ f x₂ k)) :
  k ≥ 40 :=
sorry

end k_value_range_l2_2606


namespace bake_cookies_l2_2721

noncomputable def scale_factor (original_cookies target_cookies : ℕ) : ℕ :=
  target_cookies / original_cookies

noncomputable def required_flour (original_flour : ℕ) (scale : ℕ) : ℕ :=
  original_flour * scale

noncomputable def adjusted_sugar (original_sugar : ℕ) (scale : ℕ) (reduction_percent : ℚ) : ℚ :=
  original_sugar * scale * (1 - reduction_percent)

theorem bake_cookies 
  (original_cookies : ℕ)
  (target_cookies : ℕ)
  (original_flour : ℕ)
  (original_sugar : ℕ)
  (reduction_percent : ℚ)
  (h_original_cookies : original_cookies = 40)
  (h_target_cookies : target_cookies = 80)
  (h_original_flour : original_flour = 3)
  (h_original_sugar : original_sugar = 1)
  (h_reduction_percent : reduction_percent = 0.25) :
  required_flour original_flour (scale_factor original_cookies target_cookies) = 6 ∧ 
  adjusted_sugar original_sugar (scale_factor original_cookies target_cookies) reduction_percent = 1.5 := by
    sorry

end bake_cookies_l2_2721


namespace sin_2alpha_value_l2_2325

theorem sin_2alpha_value (α a : ℝ)
  (h : sin (a + π / 4) = sqrt 2 * (sin α + 2 * cos α)) : 
  sin (2 * α) = -3 / 5 := 
by 
  sorry

end sin_2alpha_value_l2_2325


namespace percentage_profits_revenues_previous_year_l2_2338

noncomputable def companyProfits (R P R2009 P2009 : ℝ) : Prop :=
  (R2009 = 0.8 * R) ∧ (P2009 = 0.15 * R2009) ∧ (P2009 = 1.5 * P)

theorem percentage_profits_revenues_previous_year (R P : ℝ) (h : companyProfits R P (0.8 * R) (0.12 * R)) : 
  (P / R * 100) = 8 :=
by 
  sorry

end percentage_profits_revenues_previous_year_l2_2338


namespace arithmetic_sequence_index_l2_2675

theorem arithmetic_sequence_index {a : ℕ → ℕ} (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 3) (h₃ : a n = 2014) : n = 672 :=
by
  sorry

end arithmetic_sequence_index_l2_2675


namespace scientific_notation_population_l2_2500

theorem scientific_notation_population :
    ∃ (a b : ℝ), (b = 5 ∧ 1412.60 * 10 ^ 6 = a * 10 ^ b ∧ a = 1.4126) :=
sorry

end scientific_notation_population_l2_2500


namespace tan_7pi_over_4_eq_neg1_l2_2297

theorem tan_7pi_over_4_eq_neg1 : Real.tan (7 * Real.pi / 4) = -1 :=
  sorry

end tan_7pi_over_4_eq_neg1_l2_2297


namespace find_first_number_l2_2863

theorem find_first_number (sum_is_33 : ∃ x y : ℕ, x + y = 33) (second_is_twice_first : ∃ x y : ℕ, y = 2 * x) (second_is_22 : ∃ y : ℕ, y = 22) : ∃ x : ℕ, x = 11 :=
by
  sorry

end find_first_number_l2_2863


namespace average_rainfall_is_4_l2_2680

namespace VirginiaRainfall

def march_rainfall : ℝ := 3.79
def april_rainfall : ℝ := 4.5
def may_rainfall : ℝ := 3.95
def june_rainfall : ℝ := 3.09
def july_rainfall : ℝ := 4.67

theorem average_rainfall_is_4 :
  (march_rainfall + april_rainfall + may_rainfall + june_rainfall + july_rainfall) / 5 = 4 := by
  sorry

end VirginiaRainfall

end average_rainfall_is_4_l2_2680


namespace mrs_sheridan_final_cats_l2_2651

def initial_cats : ℝ := 17.5
def given_away_cats : ℝ := 6.2
def returned_cats : ℝ := 2.8
def additional_given_away_cats : ℝ := 1.3

theorem mrs_sheridan_final_cats : 
  initial_cats - given_away_cats + returned_cats - additional_given_away_cats = 12.8 :=
by
  sorry

end mrs_sheridan_final_cats_l2_2651


namespace seven_circle_divisors_exists_non_adjacent_divisors_l2_2993

theorem seven_circle_divisors_exists_non_adjacent_divisors (a : Fin 7 → ℕ)
  (h_adj : ∀ i : Fin 7, a i ∣ a (i + 1) % 7 ∨ a (i + 1) % 7 ∣ a i) :
  ∃ (i j : Fin 7), i ≠ j ∧ j ≠ i + 1 % 7 ∧ j ≠ i + 6 % 7 ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
  sorry

end seven_circle_divisors_exists_non_adjacent_divisors_l2_2993


namespace roots_polynomial_d_l2_2513

theorem roots_polynomial_d (c d u v : ℝ) (ru rpush rv rpush2 : ℝ) :
    (u + v + ru = 0) ∧ (u+3 + v-2 + rpush2 = 0) ∧
    (d + 153 = -(u + 3) * (v - 2) * (ru)) ∧ (d + 153 = s) ∧ (s = -(u + 3) * (v - 2) * (rpush2 - 1)) →
    d = 0 :=
by
  sorry

end roots_polynomial_d_l2_2513


namespace geometric_sequence_when_k_is_neg_one_l2_2002

noncomputable def S (n : ℕ) (k : ℝ) : ℝ := 3^n + k

noncomputable def a (n : ℕ) (k : ℝ) : ℝ :=
  if n = 1 then S 1 k else S n k - S (n-1) k

theorem geometric_sequence_when_k_is_neg_one :
  ∀ n : ℕ, n ≥ 1 → ∃ r : ℝ, ∀ m : ℕ, m ≥ 1 → a m (-1) = a 1 (-1) * r^(m-1) :=
by
  sorry

end geometric_sequence_when_k_is_neg_one_l2_2002


namespace probability_have_all_letters_l2_2782

noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_select_letters (word : String) (k : ℕ) (letters : Finset Char) : ℚ :=
  if letters ⊆ word.to_finset && letters.card = k then 1 / binom word.length k else 0

theorem probability_have_all_letters :
  let CAKE := "CAKE".to_finset
  let SHORE := "SHORE".to_finset
  let FLOW := "FLOW".to_finset
  let COFFEE := "COFFEE".to_finset
  let p₁ := probability_select_letters "CAKE" 2 (Finset.of_list ['C', 'E'])
  let p₂ := probability_select_letters "SHORE" 4 (Finset.of_list ['O', 'E', 'F'])
  let p₃ := probability_select_letters "FLOW" 3 (Finset.of_list ['F', 'E'])
  p₁ * p₂ * p₃ = 1 / 120 :=
by
  sorry

end probability_have_all_letters_l2_2782


namespace total_cookies_l2_2190

theorem total_cookies (x y : Nat) (h1 : x = 137) (h2 : y = 251) : x * y = 34387 := by
  sorry

end total_cookies_l2_2190


namespace crayons_count_l2_2508

theorem crayons_count 
  (initial_crayons erasers : ℕ) 
  (erasers_count end_crayons : ℕ) 
  (initial_erasers : erasers = 38) 
  (end_crayons_more_erasers : end_crayons = erasers + 353) : 
  initial_crayons = end_crayons := 
by 
  sorry

end crayons_count_l2_2508


namespace flower_bed_area_l2_2034

theorem flower_bed_area (total_posts : ℕ) (corner_posts : ℕ) (spacing : ℕ) (long_side_multiplier : ℕ)
  (h1 : total_posts = 24)
  (h2 : corner_posts = 4)
  (h3 : spacing = 3)
  (h4 : long_side_multiplier = 3) :
  ∃ (area : ℕ), area = 144 := 
sorry

end flower_bed_area_l2_2034


namespace sum_ab_equals_five_l2_2009

-- Definitions for conditions
variables {a b : ℝ}

-- Assumption that establishes the solution set for the quadratic inequality
axiom quadratic_solution_set : ∀ x : ℝ, -2 < x ∧ x < 3 ↔ x^2 + b * x - a < 0

-- Statement to be proved
theorem sum_ab_equals_five : a + b = 5 :=
sorry

end sum_ab_equals_five_l2_2009


namespace least_number_of_homeowners_l2_2631

theorem least_number_of_homeowners (total_members : ℕ) 
(num_men : ℕ) (num_women : ℕ) 
(homeowners_men : ℕ) (homeowners_women : ℕ) 
(h_total : total_members = 5000)
(h_men_women : num_men + num_women = total_members) 
(h_percentage_men : homeowners_men = 15 * num_men / 100)
(h_percentage_women : homeowners_women = 25 * num_women / 100):
  homeowners_men + homeowners_women = 4 :=
sorry

end least_number_of_homeowners_l2_2631


namespace expand_and_simplify_l2_2238

noncomputable def expanded_expr (a : ℝ) : ℝ :=
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6)

theorem expand_and_simplify (a : ℝ) :
  expanded_expr a = a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 :=
by {
  -- Proof goes here
  sorry
}

end expand_and_simplify_l2_2238


namespace catch_up_time_l2_2354

-- Define the speeds of Person A and Person B.
def speed_A : ℝ := 10 -- kilometers per hour
def speed_B : ℝ := 7  -- kilometers per hour

-- Define the initial distance between Person A and Person B.
def initial_distance : ℝ := 15 -- kilometers

-- Prove the time it takes for person A to catch up with person B is 5 hours.
theorem catch_up_time :
  initial_distance / (speed_A - speed_B) = 5 :=
by
  -- Proof can be added here
  sorry

end catch_up_time_l2_2354


namespace degree_measure_of_supplement_of_complement_of_35_degree_angle_l2_2082

def complement (α : ℝ) : ℝ := 90 - α
def supplement (β : ℝ) : ℝ := 180 - β

theorem degree_measure_of_supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 :=
by
  sorry

end degree_measure_of_supplement_of_complement_of_35_degree_angle_l2_2082


namespace value_of_x2017_l2_2604

-- Definitions and conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f (x) < f (y)

def arithmetic_sequence (x : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, x (n + 1) = x n + d

variables (f : ℝ → ℝ) (x : ℕ → ℝ)
variables (d : ℝ)
variable (h_odd : is_odd_function f)
variable (h_increasing : is_increasing_function f)
variable (h_arithmetic : arithmetic_sequence x 2)
variable (h_condition : f (x 7) + f (x 8) = 0)

-- Define the proof goal
theorem value_of_x2017 : x 2017 = 4019 :=
by
  sorry

end value_of_x2017_l2_2604


namespace allan_balloons_count_l2_2723

-- Definition of the conditions
def Total_balloons : ℕ := 3
def Jake_balloons : ℕ := 1

-- The theorem that corresponds to the problem statement
theorem allan_balloons_count (Allan_balloons : ℕ) (h : Allan_balloons + Jake_balloons = Total_balloons) : Allan_balloons = 2 := 
by
  sorry

end allan_balloons_count_l2_2723


namespace douglas_votes_in_county_y_l2_2107

variable (V : ℝ) -- Number of voters in County Y
variable (A B : ℝ) -- Votes won by Douglas in County X and County Y respectively

-- Conditions
axiom h1 : A = 0.74 * 2 * V
axiom h2 : A + B = 0.66 * 3 * V
axiom ratio : (2 * V) / V = 2

-- Proof Statement
theorem douglas_votes_in_county_y :
  (B / V) * 100 = 50 := by
sorry

end douglas_votes_in_county_y_l2_2107


namespace tan_seven_pi_over_four_l2_2291

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by sorry

end tan_seven_pi_over_four_l2_2291


namespace three_digit_perfect_squares_div_by_4_count_l2_2323

theorem three_digit_perfect_squares_div_by_4_count : 
  (∃ count : ℕ, count = 11 ∧ (∀ n : ℕ, 10 ≤ n ∧ n ≤ 31 → n^2 ≥ 100 ∧ n^2 ≤ 999 ∧ n^2 % 4 = 0)) :=
by
  sorry

end three_digit_perfect_squares_div_by_4_count_l2_2323


namespace simplify_expression_l2_2922

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l2_2922


namespace avg_price_pen_is_correct_l2_2879

-- Definitions for the total numbers and expenses:
def number_of_pens : ℕ := 30
def number_of_pencils : ℕ := 75
def total_cost : ℕ := 630
def avg_price_pencil : ℝ := 2.00

-- Calculation of total cost for pencils and pens
def total_cost_pencils : ℝ := number_of_pencils * avg_price_pencil
def total_cost_pens : ℝ := total_cost - total_cost_pencils

-- Statement to prove:
theorem avg_price_pen_is_correct :
  total_cost_pens / number_of_pens = 16 :=
by
  sorry

end avg_price_pen_is_correct_l2_2879


namespace total_interest_proof_l2_2108

open Real

def initial_investment : ℝ := 10000
def interest_6_months : ℝ := 0.02 * initial_investment
def reinvested_amount_6_months : ℝ := initial_investment + interest_6_months
def interest_10_months : ℝ := 0.03 * reinvested_amount_6_months
def reinvested_amount_10_months : ℝ := reinvested_amount_6_months + interest_10_months
def interest_18_months : ℝ := 0.04 * reinvested_amount_10_months

def total_interest : ℝ := interest_6_months + interest_10_months + interest_18_months

theorem total_interest_proof : total_interest = 926.24 := by
    sorry

end total_interest_proof_l2_2108


namespace first_term_of_geometric_series_l2_2276

theorem first_term_of_geometric_series (r a S : ℝ) (h_r : r = 1 / 4) (h_S : S = 40) 
  (h_geometric_sum : S = a / (1 - r)) : a = 30 :=
by
  -- The proof would go here, but we place a sorry to skip the proof.
  sorry

end first_term_of_geometric_series_l2_2276


namespace kite_area_correct_l2_2941

open Real

structure Point where
  x : ℝ
  y : ℝ

def Kite (p1 p2 p3 p4 : Point) : Prop :=
  let triangle_area (a b c : Point) : ℝ :=
    abs (0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)))
  triangle_area p1 p2 p4 + triangle_area p1 p3 p4 = 102

theorem kite_area_correct : ∃ (p1 p2 p3 p4 : Point), 
  p1 = Point.mk 0 10 ∧ 
  p2 = Point.mk 6 14 ∧ 
  p3 = Point.mk 12 10 ∧ 
  p4 = Point.mk 6 0 ∧ 
  Kite p1 p2 p3 p4 :=
by
  sorry

end kite_area_correct_l2_2941


namespace athlete_A_most_stable_l2_2527

noncomputable def athlete_A_variance : ℝ := 0.019
noncomputable def athlete_B_variance : ℝ := 0.021
noncomputable def athlete_C_variance : ℝ := 0.020
noncomputable def athlete_D_variance : ℝ := 0.022

theorem athlete_A_most_stable :
  athlete_A_variance < athlete_B_variance ∧
  athlete_A_variance < athlete_C_variance ∧
  athlete_A_variance < athlete_D_variance :=
by {
  sorry
}

end athlete_A_most_stable_l2_2527


namespace opposite_of_three_l2_2815

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l2_2815


namespace find_b_l2_2205

theorem find_b (a b c : ℕ) (h1 : a * b + b * c - c * a = 0) (h2 : a - c = 101) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) : b = 2550 :=
sorry

end find_b_l2_2205


namespace opposite_of_three_l2_2860

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l2_2860


namespace problem1_problem2_l2_2705

-- Problem 1 Lean Statement
theorem problem1 (m n : ℕ) (h1 : 3 ^ m = 6) (h2 : 9 ^ n = 2) : 3 ^ (m - 2 * n) = 3 :=
by
  sorry

-- Problem 2 Lean Statement
theorem problem2 (x : ℝ) (n : ℕ) (h : x ^ (2 * n) = 3) : (x ^ (3 * n)) ^ 2 - (x ^ 2) ^ (2 * n) = 18 :=
by
  sorry

end problem1_problem2_l2_2705


namespace degree_meas_supp_compl_35_l2_2095

noncomputable def degree_meas_supplement_complement (θ : ℝ) : ℝ :=
  180 - (90 - θ)

theorem degree_meas_supp_compl_35 : degree_meas_supplement_complement 35 = 125 :=
by
  unfold degree_meas_supplement_complement
  norm_num
  sorry

end degree_meas_supp_compl_35_l2_2095


namespace hyperbola_representation_iff_l2_2188

theorem hyperbola_representation_iff (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (2 + m) - (y^2) / (m + 1) = 1) ↔ (m > -1 ∨ m < -2) :=
by
  sorry

end hyperbola_representation_iff_l2_2188


namespace opposite_of_three_l2_2842

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l2_2842


namespace union_A_B_intersection_A_complement_B_l2_2953

def setA (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2
def setB (x : ℝ) : Prop := x * (x - 4) ≤ 0

theorem union_A_B : {x : ℝ | setA x} ∪ {x : ℝ | setB x} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem intersection_A_complement_B : {x : ℝ | setA x} ∩ {x : ℝ | ¬ setB x} = {x : ℝ | -1 ≤ x ∧ x < 0} :=
by
  sorry

end union_A_B_intersection_A_complement_B_l2_2953


namespace files_to_organize_in_afternoon_l2_2740

-- Defining the given conditions.
def initial_files : ℕ := 60
def files_organized_in_the_morning : ℕ := initial_files / 2
def missing_files_in_the_afternoon : ℕ := 15

-- The theorem to prove:
theorem files_to_organize_in_afternoon : 
  files_organized_in_the_morning + missing_files_in_the_afternoon = initial_files / 2 →
  ∃ afternoon_files : ℕ, 
    afternoon_files = (initial_files - files_organized_in_the_morning) - missing_files_in_the_afternoon :=
by
  -- Proof will go here, skipping with sorry for now.
  sorry

end files_to_organize_in_afternoon_l2_2740


namespace power_function_value_l2_2951

theorem power_function_value {α : ℝ} (h : 3^α = Real.sqrt 3) : (9 : ℝ)^α = 3 :=
by sorry

end power_function_value_l2_2951


namespace probability_of_success_l2_2545

theorem probability_of_success 
  (pA : ℚ) (pB : ℚ) 
  (hA : pA = 2 / 3) 
  (hB : pB = 3 / 5) :
  1 - ((1 - pA) * (1 - pB)) = 13 / 15 :=
by
  sorry

end probability_of_success_l2_2545


namespace candy_eaten_l2_2727

theorem candy_eaten (x : ℕ) (initial_candy eaten_more remaining : ℕ) (h₁ : initial_candy = 22) (h₂ : eaten_more = 5) (h₃ : remaining = 8) (h₄ : initial_candy - x - eaten_more = remaining) : x = 9 :=
by
  -- proof
  sorry

end candy_eaten_l2_2727


namespace probability_third_smallest_is_five_l2_2992

theorem probability_third_smallest_is_five :
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 7 4)
  let probability := favorable_ways / total_ways
  probability = Rat.ofInt 35 / 132 :=
by
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 7 4)
  let probability := favorable_ways / total_ways
  show probability = Rat.ofInt 35 / 132
  sorry

end probability_third_smallest_is_five_l2_2992


namespace tower_total_surface_area_l2_2659

/-- Given seven cubes with volumes 1, 8, 27, 64, 125, 216, and 343 cubic units each, stacked vertically
    with volumes decreasing from bottom to top, compute their total surface area including the bottom. -/
theorem tower_total_surface_area :
  let volumes := [1, 8, 27, 64, 125, 216, 343]
  let side_lengths := volumes.map (fun v => v ^ (1 / 3))
  let surface_area (n : ℝ) (visible_faces : ℕ) := visible_faces * (n ^ 2)
  let total_surface_area := surface_area 7 5 + surface_area 6 4 + surface_area 5 4 + surface_area 4 4
                            + surface_area 3 4 + surface_area 2 4 + surface_area 1 5
  total_surface_area = 610 := sorry

end tower_total_surface_area_l2_2659


namespace g_value_at_2002_l2_2603

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Conditions given in the problem
axiom f_one : f 1 = 1
axiom f_inequality_5 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_inequality_1 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

-- Define the function g based on f
def g (x : ℝ) : ℝ := f x + 1 - x

-- The goal is to prove that g 2002 = 1
theorem g_value_at_2002 : g 2002 = 1 :=
sorry

end g_value_at_2002_l2_2603


namespace bulbs_arrangement_l2_2686

theorem bulbs_arrangement :
  let blue_bulbs := 5
  let red_bulbs := 8
  let white_bulbs := 11
  let total_non_white_bulbs := blue_bulbs + red_bulbs
  let total_gaps := total_non_white_bulbs + 1
  (Nat.choose 13 5) * (Nat.choose total_gaps white_bulbs) = 468468 :=
by
  sorry

end bulbs_arrangement_l2_2686


namespace opposite_of_3_is_neg3_l2_2819

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2819


namespace f_positive_l2_2947

variable (f : ℝ → ℝ)

-- f is a differentiable function on ℝ
variable (hf : differentiable ℝ f)

-- Condition: (x+1)f(x) + x f''(x) > 0
variable (H : ∀ x, (x + 1) * f x + x * (deriv^[2]) f x > 0)

-- Prove: ∀ x, f x > 0
theorem f_positive : ∀ x, f x > 0 := 
by
  sorry

end f_positive_l2_2947


namespace how_many_three_digit_numbers_without_5s_and_8s_l2_2455

def is_valid_hundreds_digit (d : ℕ) : Prop := d ≠ 0 ∧ d ≠ 5 ∧ d ≠ 8
def is_valid_digit (d : ℕ) : Prop := d ≠ 5 ∧ d ≠ 8

theorem how_many_three_digit_numbers_without_5s_and_8s : 
  (∃ count : ℕ, count = 
    (∑ d1 in (finset.range 10).filter is_valid_hundreds_digit, 
      ∑ d2 in (finset.range 10).filter is_valid_digit, 
        ∑ d3 in (finset.range 10).filter is_valid_digit, 1)) = 448 :=
by
  sorry

end how_many_three_digit_numbers_without_5s_and_8s_l2_2455


namespace smallest_product_of_set_l2_2301

noncomputable def smallest_product_set : Set ℤ := { -10, -3, 0, 4, 6 }

theorem smallest_product_of_set :
  ∃ (a b : ℤ), a ∈ smallest_product_set ∧ b ∈ smallest_product_set ∧ a ≠ b ∧ a * b = -60 ∧
  ∀ (x y : ℤ), x ∈ smallest_product_set ∧ y ∈ smallest_product_set ∧ x ≠ y → x * y ≥ -60 := 
sorry

end smallest_product_of_set_l2_2301


namespace expand_and_simplify_l2_2236

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l2_2236


namespace circular_permutations_count_l2_2300

noncomputable def α : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def β : ℝ := (1 - Real.sqrt 5) / 2

noncomputable def b (n : ℕ) : ℝ := 
if n = 1 then 1 else 
if n = 2 then 2 else 
2 + α^(n - 1) + β^(n - 1)

theorem circular_permutations_count (n : ℕ) : b n = α^n + β^n + 2 := 
by {
  sorry
}

end circular_permutations_count_l2_2300


namespace mutually_exclusive_pairs_l2_2178

/-- Define the events for shooting rings and drawing balls. -/
inductive ShootEvent
| hits_7th_ring : ShootEvent
| hits_8th_ring : ShootEvent

inductive PersonEvent
| at_least_one_hits : PersonEvent
| A_hits_B_does_not : PersonEvent

inductive BallEvent
| at_least_one_black : BallEvent
| both_red : BallEvent
| no_black : BallEvent
| one_red : BallEvent

/-- Define mutually exclusive events. -/
def mutually_exclusive (e1 e2 : Prop) : Prop := e1 ∧ e2 → False

/-- Prove the pairs of events that are mutually exclusive. -/
theorem mutually_exclusive_pairs :
  mutually_exclusive (ShootEvent.hits_7th_ring = ShootEvent.hits_7th_ring) (ShootEvent.hits_8th_ring = ShootEvent.hits_8th_ring) ∧
  ¬mutually_exclusive (PersonEvent.at_least_one_hits = PersonEvent.at_least_one_hits) (PersonEvent.A_hits_B_does_not = PersonEvent.A_hits_B_does_not) ∧
  mutually_exclusive (BallEvent.at_least_one_black = BallEvent.at_least_one_black) (BallEvent.both_red = BallEvent.both_red) ∧
  mutually_exclusive (BallEvent.no_black = BallEvent.no_black) (BallEvent.one_red = BallEvent.one_red) :=
by {
  sorry
}

end mutually_exclusive_pairs_l2_2178


namespace equal_wear_tires_l2_2563

theorem equal_wear_tires (t D d : ℕ) (h1 : t = 7) (h2 : D = 42000) (h3 : t * d = 6 * D) : d = 36000 :=
by
  sorry

end equal_wear_tires_l2_2563


namespace factorize_expression_l2_2588

theorem factorize_expression : 989 * 1001 * 1007 + 320 = 991 * 997 * 1009 := 
by sorry

end factorize_expression_l2_2588


namespace min_value_of_linear_combination_of_variables_l2_2629

-- Define the conditions that x and y are positive numbers and satisfy the equation x + 3y = 5xy
def conditions (x y : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y

-- State the theorem that the minimum value of 3x + 4y given the conditions is 5
theorem min_value_of_linear_combination_of_variables (x y : ℝ) (h: conditions x y) : 3 * x + 4 * y ≥ 5 :=
by 
  sorry

end min_value_of_linear_combination_of_variables_l2_2629


namespace opposite_of_three_l2_2855

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l2_2855


namespace difference_of_squares_l2_2256

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) : x^2 - y^2 = 160 :=
by
  sorry

end difference_of_squares_l2_2256


namespace percentage_apples_sold_l2_2399

theorem percentage_apples_sold (A P : ℕ) (h1 : A = 600) (h2 : A * (100 - P) / 100 = 420) : P = 30 := 
by {
  sorry
}

end percentage_apples_sold_l2_2399


namespace find_M_l2_2538

theorem find_M (a b c M : ℝ) (h1 : a + b + c = 120) (h2 : a - 9 = M) (h3 : b + 9 = M) (h4 : 9 * c = M) : 
  M = 1080 / 19 :=
by sorry

end find_M_l2_2538


namespace max_objective_value_l2_2962

theorem max_objective_value (x y : ℝ) (h1 : x - y - 2 ≥ 0) (h2 : 2 * x + y - 2 ≤ 0) (h3 : y + 4 ≥ 0) :
  ∃ (z : ℝ), z = 4 * x + 3 * y ∧ z ≤ 8 :=
sorry

end max_objective_value_l2_2962


namespace Ann_age_is_39_l2_2578

def current_ages (A B : ℕ) : Prop :=
  A + B = 52 ∧ (B = 2 * B - A / 3) ∧ (A = 3 * B)

theorem Ann_age_is_39 : ∃ A B : ℕ, current_ages A B ∧ A = 39 :=
by
  sorry

end Ann_age_is_39_l2_2578


namespace other_five_say_equal_numbers_l2_2048

noncomputable def knights_and_liars_problem : Prop :=
  ∃ (K L : ℕ), K + L = 10 ∧
  ∀ (x : ℕ), (x < 5 → "There are more liars" = true) ∨ (x >= 5 → "There are equal numbers of knights and liars" = true)

theorem other_five_say_equal_numbers :
  knights_and_liars_problem :=
sorry

end other_five_say_equal_numbers_l2_2048


namespace N_subset_M_l2_2795

open Set

def M : Set (ℝ × ℝ) := { p | ∃ x, p = (x, 2*x + 1) }
def N : Set (ℝ × ℝ) := { p | ∃ x, p = (x, -x^2) }

theorem N_subset_M : N ⊆ M :=
by
  sorry

end N_subset_M_l2_2795


namespace kelsey_remaining_half_speed_l2_2786

variable (total_hours : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) (remaining_half_time : ℝ) (remaining_half_distance : ℝ)

axiom h1 : total_hours = 10
axiom h2 : first_half_speed = 25
axiom h3 : total_distance = 400
axiom h4 : remaining_half_time = total_hours - total_distance / (2 * first_half_speed)
axiom h5 : remaining_half_distance = total_distance / 2

theorem kelsey_remaining_half_speed :
  remaining_half_distance / remaining_half_time = 100
:=
by
  sorry

end kelsey_remaining_half_speed_l2_2786


namespace sum_of_possible_values_of_p_s_l2_2645

theorem sum_of_possible_values_of_p_s (p q r s : ℝ) 
  (h1 : |p - q| = 3) 
  (h2 : |q - r| = 4) 
  (h3 : |r - s| = 5) : 
  (finite (λ x, ∃ (s : ℝ), (|p - s| = x ∧ |q - p| = 3 ∧ |r - q| = 4 ∧ |s - r| = 5)).sum = 24) :=
sorry

end sum_of_possible_values_of_p_s_l2_2645


namespace whatsapp_messages_total_l2_2112

-- Define conditions
def messages_monday : ℕ := 300
def messages_tuesday : ℕ := 200
def messages_wednesday : ℕ := messages_tuesday + 300
def messages_thursday : ℕ := 2 * messages_wednesday
def messages_friday : ℕ := messages_thursday + (20 * messages_thursday) / 100
def messages_saturday : ℕ := messages_friday - (10 * messages_friday) / 100

-- Theorem statement to be proved
theorem whatsapp_messages_total :
  messages_monday + messages_tuesday + messages_wednesday + messages_thursday + messages_friday + messages_saturday = 4280 :=
by 
  sorry

end whatsapp_messages_total_l2_2112


namespace swimmer_upstream_distance_l2_2130

theorem swimmer_upstream_distance (v : ℝ) (c : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) 
                                   (downstream_speed : ℝ) (upstream_time : ℝ) : 
  c = 4.5 →
  downstream_distance = 55 →
  downstream_time = 5 →
  downstream_speed = downstream_distance / downstream_time →
  v + c = downstream_speed →
  upstream_time = 5 →
  (v - c) * upstream_time = 10 := 
by
  intro h_c
  intro h_downstream_distance
  intro h_downstream_time
  intro h_downstream_speed
  intro h_effective_downstream
  intro h_upstream_time
  sorry

end swimmer_upstream_distance_l2_2130


namespace roster_method_A_l2_2861

def A : Set ℤ := {x | 0 < x ∧ x ≤ 2}

theorem roster_method_A :
  A = {1, 2} :=
by
  sorry

end roster_method_A_l2_2861


namespace Kylie_uses_3_towels_in_one_month_l2_2195

-- Define the necessary variables and conditions
variable (daughters_towels : Nat) (husband_towels : Nat) (loads : Nat) (towels_per_load : Nat)
variable (K : Nat) -- number of bath towels Kylie uses

-- Given conditions
axiom h1 : daughters_towels = 6
axiom h2 : husband_towels = 3
axiom h3 : loads = 3
axiom h4 : towels_per_load = 4
axiom h5 : (K + daughters_towels + husband_towels) = (loads * towels_per_load)

-- Prove that K = 3
theorem Kylie_uses_3_towels_in_one_month : K = 3 :=
by
  sorry

end Kylie_uses_3_towels_in_one_month_l2_2195


namespace opposite_of_three_l2_2856

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l2_2856


namespace glass_ball_radius_l2_2362

theorem glass_ball_radius (x y r : ℝ) (h_parabola : x^2 = 2 * y) (h_touch : y = r) (h_range : 0 ≤ y ∧ y ≤ 20) : 0 < r ∧ r ≤ 1 :=
sorry

end glass_ball_radius_l2_2362


namespace volume_CO2_is_7_l2_2878

-- Definitions based on conditions
def Avogadro_law (V1 V2 : ℝ) : Prop := V1 = V2
def molar_ratio (V_CO2 V_O2 : ℝ) : Prop := V_CO2 = 1 / 2 * V_O2
def volume_O2 : ℝ := 14

-- Statement to be proved
theorem volume_CO2_is_7 : ∃ V_CO2 : ℝ, molar_ratio V_CO2 volume_O2 ∧ V_CO2 = 7 := by
  sorry

end volume_CO2_is_7_l2_2878


namespace find_larger_number_l2_2684

-- Definitions based on the conditions
variables (x y : ℕ)

-- Main theorem
theorem find_larger_number (h1 : x + y = 50) (h2 : x - y = 10) : x = 30 :=
by
  sorry

end find_larger_number_l2_2684


namespace expression_simplification_l2_2911

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l2_2911


namespace simplify_expression_l2_2906

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l2_2906


namespace sphere_radius_ratio_l2_2867

theorem sphere_radius_ratio (R1 R2 : ℝ) (m n : ℝ) (hm : 1 < m) (hn : 1 < n) 
  (h_ratio1 : (2 * π * R1 * ((2 * R1) / (m + 1))) / (4 * π * R1 * R1) = 1 / (m + 1))
  (h_ratio2 : (2 * π * R2 * ((2 * R2) / (n + 1))) / (4 * π * R2 * R2) = 1 / (n + 1)): 
  R2 / R1 = ((m - 1) * (n + 1)) / ((m + 1) * (n - 1)) := 
by
  sorry

end sphere_radius_ratio_l2_2867


namespace sarah_age_l2_2803

variable (s m : ℕ)

theorem sarah_age (h1 : s = m - 18) (h2 : s + m = 50) : s = 16 :=
by {
  -- The proof will go here
  sorry
}

end sarah_age_l2_2803


namespace compounded_rate_of_growth_l2_2136

theorem compounded_rate_of_growth (k m : ℝ) :
  (1 + k / 100) * (1 + m / 100) - 1 = ((k + m + (k * m / 100)) / 100) :=
by
  sorry

end compounded_rate_of_growth_l2_2136


namespace angle_AFE_is_80_degrees_l2_2197

-- Defining the setup and given conditions
def point := ℝ × ℝ  -- defining a 2D point
noncomputable def A : point := (0, 0)
noncomputable def B : point := (1, 0)
noncomputable def C : point := (1, 1)
noncomputable def D : point := (0, 1)
noncomputable def E : point := (-1, 1.732)  -- Place E such that angle CDE ≈ 130 degrees

-- Conditions
def angle_CDE := 130
def DF_over_DE := 2  -- DF = 2 * DE
noncomputable def F : point := (0.5, 1)  -- This is an example position; real positioning depends on more details

-- Proving that the angle AFE is 80 degrees
theorem angle_AFE_is_80_degrees :
  ∃ (AFE : ℝ), AFE = 80 := sorry

end angle_AFE_is_80_degrees_l2_2197


namespace mouse_away_from_cheese_l2_2566

theorem mouse_away_from_cheese:
  ∃ a b : ℝ, a = 3 ∧ b = 3 ∧ (a + b = 6) ∧
  ∀ x y : ℝ, (y = -3 * x + 12) → 
  ∀ (a y₀ : ℝ), y₀ = (1/3) * a + 11 →
  (a, b) = (3, 3) :=
by
  sorry

end mouse_away_from_cheese_l2_2566


namespace inheritance_amount_l2_2546

theorem inheritance_amount (x : ℝ) 
  (federal_tax : ℝ := 0.25 * x) 
  (state_tax : ℝ := 0.15 * (x - federal_tax)) 
  (city_tax : ℝ := 0.05 * (x - federal_tax - state_tax)) 
  (total_tax : ℝ := 20000) :
  (federal_tax + state_tax + city_tax = total_tax) → 
  x = 50704 :=
by
  intros h
  sorry

end inheritance_amount_l2_2546


namespace opposite_of_3_is_neg3_l2_2823

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2823


namespace points_product_l2_2134

def f (n : ℕ) : ℕ :=
  if n % 6 == 0 then 6
  else if n % 2 == 0 then 2
  else 0

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

def allie_rolls := [5, 4, 1, 2]
def betty_rolls := [6, 3, 3, 2]

def allie_points := total_points allie_rolls
def betty_points := total_points betty_rolls

theorem points_product : allie_points * betty_points = 32 := by
  sorry

end points_product_l2_2134


namespace find_f_8_6_l2_2253

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem find_f_8_6 (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_symmetry : ∀ x, f (1 + x) = f (1 - x))
  (h_def : ∀ x, -1 ≤ x ∧ x ≤ 0 → f x = - (1 / 2) * x) :
  f 8.6 = 0.3 :=
sorry

end find_f_8_6_l2_2253


namespace train_length_l2_2131

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (speed_conversion : speed_kmh = 40) 
  (time_condition : time_s = 27) : 
  (speed_kmh * 1000 / 3600 * time_s = 300) := 
by
  sorry

end train_length_l2_2131


namespace frustum_radius_l2_2060

theorem frustum_radius (r : ℝ) (h1 : ∃ r1 r2, r1 = r 
                                  ∧ r2 = 3 * r 
                                  ∧ r1 * 2 * π * 3 = r2 * 2 * π
                                  ∧ (lateral_area = 84 * π)) (h2 : slant_height = 3) : 
  r = 7 :=
sorry

end frustum_radius_l2_2060


namespace pencil_and_eraser_cost_l2_2045

theorem pencil_and_eraser_cost (p e : ℕ) :
  2 * p + e = 40 →
  p > e →
  e ≥ 3 →
  p + e = 22 :=
by
  sorry

end pencil_and_eraser_cost_l2_2045


namespace range_of_a_l2_2445

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≥ 0) ↔ -1 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l2_2445


namespace solve_abs_linear_eq_l2_2020

theorem solve_abs_linear_eq (x : ℝ) : (|x - 1| + x - 1 = 0) ↔ (x ≤ 1) :=
sorry

end solve_abs_linear_eq_l2_2020


namespace complex_mul_l2_2979

theorem complex_mul (i : ℂ) (h : i^2 = -1) :
    (1 - i) * (1 + 2 * i) = 3 + i :=
by
  sorry

end complex_mul_l2_2979


namespace original_profit_percentage_l2_2719

theorem original_profit_percentage
  (C : ℝ) -- original cost
  (S : ℝ) -- selling price
  (y : ℝ) -- original profit percentage
  (hS : S = C * (1 + 0.01 * y)) -- condition for selling price based on original cost
  (hC' : S = 0.85 * C * (1 + 0.01 * (y + 20))) -- condition for selling price based on reduced cost
  : y = -89 :=
by
  sorry

end original_profit_percentage_l2_2719


namespace geun_bae_fourth_day_jumps_l2_2472

-- Define a function for number of jump ropes Geun-bae does on each day
def jump_ropes (n : ℕ) : ℕ :=
  match n with
  | 0     => 15
  | n + 1 => 2 * jump_ropes n

-- Theorem stating the number of jump ropes Geun-bae does on the fourth day
theorem geun_bae_fourth_day_jumps : jump_ropes 3 = 120 := 
by {
  sorry
}

end geun_bae_fourth_day_jumps_l2_2472


namespace boxes_filled_l2_2653

theorem boxes_filled (total_toys toys_per_box : ℕ) (h1 : toys_per_box = 8) (h2 : total_toys = 32) :
  total_toys / toys_per_box = 4 :=
by
  sorry

end boxes_filled_l2_2653


namespace son_l2_2405

def woman's_age (W S : ℕ) : Prop := W = 2 * S + 3
def sum_of_ages (W S : ℕ) : Prop := W + S = 84

theorem son's_age_is_27 (W S : ℕ) (h1: woman's_age W S) (h2: sum_of_ages W S) : S = 27 :=
by
  sorry

end son_l2_2405


namespace opposite_of_3_is_neg3_l2_2829

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2829


namespace rake_yard_alone_time_l2_2473

-- Definitions for the conditions
def brother_time := 45 -- Brother takes 45 minutes
def together_time := 18 -- Together it takes 18 minutes

-- Define and prove the time it takes you to rake the yard alone based on given conditions
theorem rake_yard_alone_time : 
  ∃ (x : ℕ), (1 / (x : ℚ) + 1 / (brother_time : ℚ) = 1 / (together_time : ℚ)) ∧ x = 30 :=
by
  sorry

end rake_yard_alone_time_l2_2473


namespace simplify_expression_eq_square_l2_2897

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l2_2897


namespace inequality_1_inequality_2_l2_2663

theorem inequality_1 (x : ℝ) : (2 * x^2 - 3 * x + 1 < 0) ↔ (1 / 2 < x ∧ x < 1) := 
by sorry

theorem inequality_2 (x : ℝ) (h : x ≠ -1) : (2 * x / (x + 1) ≥ 1) ↔ (x < -1 ∨ x ≥ 1) := 
by sorry

end inequality_1_inequality_2_l2_2663


namespace expression_eq_16x_l2_2022

variable (x y z w : ℝ)

theorem expression_eq_16x
  (h1 : y = 2 * x)
  (h2 : z = 3 * y)
  (h3 : w = z + x) :
  x + y + z + w = 16 * x :=
sorry

end expression_eq_16x_l2_2022


namespace perimeter_of_unshaded_rectangle_l2_2402

theorem perimeter_of_unshaded_rectangle (length width height base area shaded_area perimeter : ℝ)
  (h1 : length = 12)
  (h2 : width = 9)
  (h3 : height = 3)
  (h4 : base = (2 * shaded_area) / height)
  (h5 : shaded_area = 18)
  (h6 : perimeter = 2 * ((length - base) + width))
  : perimeter = 24 := by
  sorry

end perimeter_of_unshaded_rectangle_l2_2402


namespace ratio_initial_to_doubled_l2_2884

theorem ratio_initial_to_doubled (x : ℕ) (h : 3 * (2 * x + 9) = 63) : x / (2 * x) = 1 / 2 := 
by
  sorry

end ratio_initial_to_doubled_l2_2884


namespace work_rate_c_l2_2385

theorem work_rate_c (A B C : ℝ) 
  (h1 : A + B = 1 / 15) 
  (h2 : A + B + C = 1 / 5) :
  (1 / C) = 7.5 :=
by 
  sorry

end work_rate_c_l2_2385


namespace range_of_a_l2_2862

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x < 2 → (x + a < 0))) → (a ≤ -2) :=
sorry

end range_of_a_l2_2862


namespace initial_bananas_on_tree_l2_2711

-- Definitions of given conditions
def bananas_left_on_tree : ℕ := 100
def bananas_eaten : ℕ := 70
def bananas_in_basket : ℕ := 2 * bananas_eaten

-- Statement to prove the initial number of bananas on the tree
theorem initial_bananas_on_tree : bananas_left_on_tree + (bananas_in_basket + bananas_eaten) = 310 :=
by
  sorry

end initial_bananas_on_tree_l2_2711


namespace joseph_cards_percentage_left_l2_2783

theorem joseph_cards_percentage_left (h1 : ℕ := 16) (h2 : ℚ := 3/8) (h3 : ℕ := 2) :
  ((h1 - (h2 * h1 + h3)) / h1 * 100) = 50 :=
by
  sorry

end joseph_cards_percentage_left_l2_2783


namespace refreshment_stand_distance_l2_2129

theorem refreshment_stand_distance 
  (A B S : ℝ) -- Positions of the camps and refreshment stand
  (dist_A_highway : A = 400) -- Distance from the first camp to the highway
  (dist_B_A : B = 700) -- Distance from the second camp directly across the highway
  (equidistant : ∀ x, S = x ∧ dist (S, A) = dist (S, B)) : 
  S = 500 := -- Distance from the refreshment stand to each camp is 500 meters
sorry

end refreshment_stand_distance_l2_2129


namespace part1_part2_l2_2040

-- Definitions and conditions
def prop_p (a : ℝ) : Prop := 
  let Δ := -4 * a^2 + 4 * a + 24 
  Δ ≥ 0

def neg_prop_p (a : ℝ) : Prop := ¬ prop_p a

def prop_q (m a : ℝ) : Prop := 
  (m - 1 ≤ a ∧ a ≤ m + 3)

-- Part 1 theorem statement
theorem part1 (a : ℝ) : neg_prop_p a → (a < -2 ∨ a > 3) :=
by sorry

-- Part 2 theorem statement
theorem part2 (m : ℝ) : 
  (∀ a : ℝ, prop_q m a → prop_p a) ∧ (∃ a : ℝ, prop_p a ∧ ¬ prop_q m a) → (-1 ≤ m ∧ m < 0) :=
by sorry

end part1_part2_l2_2040


namespace hyperbola_eccentricity_l2_2809

theorem hyperbola_eccentricity (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : ∀ x : ℝ, y = (3 / 4) * x → y = (b / a) * x) : 
  (b = (3 / 4) * a) → (e = 5 / 4) := 
by
  sorry

end hyperbola_eccentricity_l2_2809


namespace even_digit_number_division_l2_2298

theorem even_digit_number_division (N : ℕ) (n : ℕ) :
  (N % 2 = 0) ∧
  (∃ a b : ℕ, (∀ k : ℕ, N = a * 10^n + b → N = k * (a * b)) ∧
  ((N = (1000^(2*n - 1) + 1)^2 / 7) ∨
   (N = 12) ∨
   (N = (10^n + 2)^2 / 6) ∨
   (N = 1352) ∨
   (N = 15))) :=
sorry

end even_digit_number_division_l2_2298


namespace ratio_of_juice_to_bread_l2_2217

variable (total_money : ℕ) (money_left : ℕ) (cost_bread : ℕ) (cost_butter : ℕ) (cost_juice : ℕ)

def compute_ratio (total_money money_left cost_bread cost_butter cost_juice : ℕ) : ℕ :=
  cost_juice / cost_bread

theorem ratio_of_juice_to_bread :
  total_money = 15 →
  money_left = 6 →
  cost_bread = 2 →
  cost_butter = 3 →
  total_money - money_left - (cost_bread + cost_butter) = cost_juice →
  compute_ratio total_money money_left cost_bread cost_butter cost_juice = 2 :=
by
  intros
  sorry

end ratio_of_juice_to_bread_l2_2217


namespace question_l2_2654

-- Let x and y be real numbers.
variables (x y : ℝ)

-- Proposition A: x + y ≠ 8
def PropA : Prop := x + y ≠ 8

-- Proposition B: x ≠ 2 ∨ y ≠ 6
def PropB : Prop := x ≠ 2 ∨ y ≠ 6

-- We need to prove that PropA is a sufficient but not necessary condition for PropB.
theorem question : (PropA x y → PropB x y) ∧ ¬ (PropB x y → PropA x y) :=
sorry

end question_l2_2654


namespace max_volume_small_cube_l2_2432

theorem max_volume_small_cube (a : ℝ) (h : a = 2) : (a^3 = 8) := by
  sorry

end max_volume_small_cube_l2_2432


namespace tan_seven_pi_over_four_l2_2289

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by sorry

end tan_seven_pi_over_four_l2_2289


namespace ned_time_left_to_diffuse_bomb_l2_2209

-- Conditions
def building_flights : Nat := 20
def time_per_flight : Nat := 11
def bomb_timer : Nat := 72
def time_spent_running : Nat := 165

-- Main statement
theorem ned_time_left_to_diffuse_bomb : 
  (bomb_timer - (building_flights - (time_spent_running / time_per_flight)) * time_per_flight) = 17 :=
by
  sorry

end ned_time_left_to_diffuse_bomb_l2_2209


namespace find_k_l2_2966

theorem find_k
  (k : ℝ)
  (AB : ℝ × ℝ := (3, 1))
  (AC : ℝ × ℝ := (2, k))
  (BC : ℝ × ℝ := (2 - 3, k - 1))
  (h_perpendicular : AB.1 * BC.1 + AB.2 * BC.2 = 0)
  : k = 4 :=
sorry

end find_k_l2_2966


namespace triangle_angle_measure_l2_2639

theorem triangle_angle_measure
  (D E F : ℝ)
  (hD : D = 70)
  (hE : E = 2 * F + 18)
  (h_sum : D + E + F = 180) :
  F = 92 / 3 :=
by
  sorry

end triangle_angle_measure_l2_2639


namespace find_a_l2_2605

-- Definitions for the hyperbola and its eccentricity
def hyperbola_eq (a : ℝ) : Prop := a > 0 ∧ ∃ b : ℝ, b^2 = 3 ∧ ∃ e : ℝ, e = 2 ∧ 
  e = Real.sqrt (1 + b^2 / a^2)

-- The main theorem stating the value of 'a' given the conditions
theorem find_a (a : ℝ) (h : hyperbola_eq a) : a = 1 := 
by {
  sorry
}

end find_a_l2_2605


namespace like_terms_sum_l2_2067

theorem like_terms_sum (m n : ℕ) (h1 : 6 * x ^ 5 * y ^ (2 * n) = 6 * x ^ m * y ^ 4) : m + n = 7 := by
  sorry

end like_terms_sum_l2_2067


namespace number_of_terms_in_sequence_l2_2018

def arithmetic_sequence_terms (a d l : ℕ) : ℕ :=
  (l - a) / d + 1

theorem number_of_terms_in_sequence : arithmetic_sequence_terms 1 4 57 = 15 :=
by {
  sorry
}

end number_of_terms_in_sequence_l2_2018


namespace other_number_l2_2685

theorem other_number (x : ℕ) (h : 27 + x = 62) : x = 35 :=
by
  sorry

end other_number_l2_2685


namespace sequence_properties_l2_2436

theorem sequence_properties (a : ℕ → ℝ)
  (h1 : a 1 = 1 / 5)
  (h2 : ∀ n : ℕ, n > 1 → a (n - 1) / a n = (2 * a (n - 1) + 1) / (1 - 2 * a n)) :
  (∀ n : ℕ, n > 0 → (1 / a n) - (1 / a (n - 1)) = 4) ∧
  (∀ m k : ℕ, m > 0 ∧ k > 0 → a m * a k = a (4 * m * k + m + k)) :=
by
  sorry

end sequence_properties_l2_2436


namespace ellipse_major_axis_focal_distance_l2_2756

theorem ellipse_major_axis_focal_distance (m : ℝ) (h1 : 10 - m > 0) (h2 : m - 2 > 0) 
  (h3 : ∀ x y, x^2 / (10 - m) + y^2 / (m - 2) = 1) 
  (h4 : ∃ c, 2 * c = 4 ∧ c^2 = (m - 2) - (10 - m)) : m = 8 :=
by
  sorry

end ellipse_major_axis_focal_distance_l2_2756


namespace problem1_problem2_l2_2054

-- Define the first problem as a proof statement in Lean
theorem problem1 (x : ℝ) : (x - 2) ^ 2 = 25 → (x = 7 ∨ x = -3) := sorry

-- Define the second problem as a proof statement in Lean
theorem problem2 (x : ℝ) : (x - 5) ^ 2 = 2 * (5 - x) → (x = 5 ∨ x = 3) := sorry

end problem1_problem2_l2_2054


namespace expand_and_simplify_l2_2241

noncomputable def expanded_expr (a : ℝ) : ℝ :=
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6)

theorem expand_and_simplify (a : ℝ) :
  expanded_expr a = a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 :=
by {
  -- Proof goes here
  sorry
}

end expand_and_simplify_l2_2241


namespace arrangement_A_head_is_720_arrangement_ABC_together_is_720_arrangement_ABC_not_together_is_1440_arrangement_A_not_head_B_not_middle_is_3720_l2_2763

noncomputable def arrangements_A_head : ℕ := 720
noncomputable def arrangements_ABC_together : ℕ := 720
noncomputable def arrangements_ABC_not_together : ℕ := 1440
noncomputable def arrangements_A_not_head_B_not_middle : ℕ := 3720

theorem arrangement_A_head_is_720 :
  arrangements_A_head = 720 := 
  by sorry

theorem arrangement_ABC_together_is_720 :
  arrangements_ABC_together = 720 := 
  by sorry

theorem arrangement_ABC_not_together_is_1440 :
  arrangements_ABC_not_together = 1440 := 
  by sorry

theorem arrangement_A_not_head_B_not_middle_is_3720 :
  arrangements_A_not_head_B_not_middle = 3720 := 
  by sorry

end arrangement_A_head_is_720_arrangement_ABC_together_is_720_arrangement_ABC_not_together_is_1440_arrangement_A_not_head_B_not_middle_is_3720_l2_2763


namespace running_speed_l2_2691

variables (w t_w t_r : ℝ)

-- Given conditions
def walking_speed : w = 8 := sorry
def walking_time_hours : t_w = 4.75 := sorry
def running_time_hours : t_r = 2 := sorry

-- Prove the man's running speed
theorem running_speed (w t_w t_r : ℝ) 
  (H1 : w = 8) 
  (H2 : t_w = 4.75) 
  (H3 : t_r = 2) : 
  (w * t_w) / t_r = 19 := 
sorry

end running_speed_l2_2691


namespace train_speed_including_stoppages_l2_2931

theorem train_speed_including_stoppages (s : ℝ) (t : ℝ) (running_time_fraction : ℝ) :
  s = 48 ∧ t = 1/4 ∧ running_time_fraction = (1 - t) → (s * running_time_fraction = 36) :=
by
  sorry

end train_speed_including_stoppages_l2_2931


namespace max_minute_hands_l2_2103

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
  sorry

end max_minute_hands_l2_2103


namespace quadratic_has_distinct_real_roots_l2_2627

theorem quadratic_has_distinct_real_roots (a : ℝ) (h : a = -2) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 2 * x1 + 3 = 0 ∧ a * x2^2 + 2 * x2 + 3 = 0) :=
by
  sorry

end quadratic_has_distinct_real_roots_l2_2627


namespace colorful_family_children_count_l2_2773

theorem colorful_family_children_count 
    (B W S x : ℕ)
    (h1 : B = W) (h2 : W = S)
    (h3 : (B - x) + W = 10)
    (h4 : W + (S + x) = 18) :
    B + W + S = 21 :=
by
  sorry

end colorful_family_children_count_l2_2773


namespace opposite_of_three_l2_2845

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l2_2845


namespace appropriate_weight_design_l2_2275

def weight_design (w_l w_s w_r w_w : ℕ) : Prop :=
  w_l > w_s ∧ w_l > w_w ∧ w_w > w_r ∧ w_s = w_w

theorem appropriate_weight_design :
  weight_design 5 2 1 2 :=
by {
  sorry -- skipped proof
}

end appropriate_weight_design_l2_2275


namespace expression_equals_41_l2_2957

theorem expression_equals_41 (x : ℝ) (h : 3*x^2 + 9*x + 5 ≠ 0) : 
  (3*x^2 + 9*x + 15) / (3*x^2 + 9*x + 5) = 41 :=
by
  sorry

end expression_equals_41_l2_2957


namespace distance_to_asymptote_l2_2999

noncomputable def distance_from_asymptote : ℝ :=
  let x0 := 3
  let y0 := 0
  let A := 3
  let B := -4
  let C := 0
  @Real.sqrt ((A^2) + (B^2))^{-1} * abs (A * x0 + B * y0 + C)

theorem distance_to_asymptote : distance_from_asymptote = (9 / 5) := sorry

end distance_to_asymptote_l2_2999


namespace contradictory_goldbach_l2_2063

theorem contradictory_goldbach : ¬ (∀ n : ℕ, 2 < n ∧ Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q) :=
sorry

end contradictory_goldbach_l2_2063


namespace expand_and_simplify_l2_2242

noncomputable def expanded_expr (a : ℝ) : ℝ :=
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6)

theorem expand_and_simplify (a : ℝ) :
  expanded_expr a = a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 :=
by {
  -- Proof goes here
  sorry
}

end expand_and_simplify_l2_2242


namespace seats_selection_l2_2488

theorem seats_selection (n k d : ℕ) (hn : n ≥ 4) (hk : k ≥ 2) (hd : d ≥ 2) (hkd : k * d ≤ n) :
  ∃ ways : ℕ, ways = (n / k) * Nat.choose (n - k * d + k - 1) (k - 1) :=
sorry

end seats_selection_l2_2488


namespace reformulate_and_find_product_l2_2223

theorem reformulate_and_find_product (a b x y : ℝ)
  (h : a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 2)) :
  ∃ m' n' p' : ℤ, (a^m' * x - a^n') * (a^p' * y - a^3) = a^5 * b^5 ∧ m' * n' * p' = 48 :=
by
  sorry

end reformulate_and_find_product_l2_2223


namespace sum_of_sides_le_twice_third_side_l2_2638

theorem sum_of_sides_le_twice_third_side 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = 180)
  (h3 : a / (Real.sin A) = b / (Real.sin B))
  (h4 : a / (Real.sin A) = c / (Real.sin C))
  (h5 : b / (Real.sin B) = c / (Real.sin C)) : 
  a + c ≤ 2 * b := 
by 
  sorry

end sum_of_sides_le_twice_third_side_l2_2638


namespace find_n_sequence_sum_l2_2450

theorem find_n_sequence_sum 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₀ : ∀ n, a n = (2^n - 1) / 2^n)
  (h₁ : S 6 = 321 / 64) :
  ∃ n, S n = 321 / 64 ∧ n = 6 := 
by 
  sorry

end find_n_sequence_sum_l2_2450


namespace least_positive_integer_x_l2_2550

theorem least_positive_integer_x
  (n : ℕ)
  (h : (3 * n)^2 + 2 * 43 * (3 * n) + 43^2) % 53 = 0
  (h_pos : 0 < n)
  : n = 21 :=
sorry

end least_positive_integer_x_l2_2550


namespace remaining_plants_after_bugs_l2_2409

theorem remaining_plants_after_bugs (initial_plants first_day_eaten second_day_fraction third_day_eaten remaining_plants : ℕ) : 
  initial_plants = 30 →
  first_day_eaten = 20 →
  second_day_fraction = 2 →
  third_day_eaten = 1 →
  remaining_plants = initial_plants - first_day_eaten - (initial_plants - first_day_eaten) / second_day_fraction - third_day_eaten →
  remaining_plants = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end remaining_plants_after_bugs_l2_2409


namespace difference_between_local_and_face_value_l2_2557

def numeral := 657903

def local_value (n : ℕ) : ℕ :=
  if n = 7 then 70000 else 0

def face_value (n : ℕ) : ℕ :=
  n

theorem difference_between_local_and_face_value :
  local_value 7 - face_value 7 = 69993 :=
by
  sorry

end difference_between_local_and_face_value_l2_2557


namespace total_amount_in_bank_l2_2726

-- Definition of the checks and their values
def checks_1mil : Nat := 25
def checks_100k : Nat := 8
def value_1mil : Nat := 1000000
def value_100k : Nat := 100000

-- The proof statement
theorem total_amount_in_bank 
  (total : Nat) 
  (h1 : checks_1mil * value_1mil = 25000000)
  (h2 : checks_100k * value_100k = 800000):
  total = 25000000 + 800000 :=
sorry

end total_amount_in_bank_l2_2726


namespace opposite_of_3_l2_2851

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l2_2851


namespace ratio_S3_S9_l2_2748

noncomputable def Sn (a r : ℝ) (n : ℕ) : ℝ := (a * (1 - r ^ n)) / (1 - r)

theorem ratio_S3_S9 (a r : ℝ) (h1 : r ≠ 1) (h2 : Sn a r 6 = 3 * Sn a r 3) :
  Sn a r 3 / Sn a r 9 = 1 / 7 :=
by
  sorry

end ratio_S3_S9_l2_2748


namespace problem_statement_l2_2150

def assoc_number (x : ℚ) : ℚ :=
  if x >= 0 then 2 * x - 1 else -2 * x + 1

theorem problem_statement (a b : ℚ) (ha : a > 0) (hb : b < 0) (hab : assoc_number a = assoc_number b) :
  (a + b)^2 - 2 * a - 2 * b = -1 :=
sorry

end problem_statement_l2_2150


namespace find_C_l2_2982

theorem find_C (A B C D E : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10) (h5 : E < 10) 
  (h : 4 * (10 * (10000 * A + 1000 * B + 100 * C + 10 * D + E) + 4) = 400000 + (10000 * A + 1000 * B + 100 * C + 10 * D + E)) : 
  C = 2 :=
sorry

end find_C_l2_2982


namespace max_xyz_value_l2_2309

theorem max_xyz_value : 
  ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 1 ∧ 
    (1 / x + 1 / y + 1 / z = 10) ∧ xyz(x, y, z) ≤ 4 / 125 := 
begin
  sorry
end

def xyz (x y z : ℝ) := x * y * z

end max_xyz_value_l2_2309


namespace opposite_of_3_is_neg3_l2_2826

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2826


namespace distance_between_A_and_B_l2_2387

def rowing_speed_still_water : ℝ := 10
def round_trip_time : ℝ := 5
def stream_speed : ℝ := 2

theorem distance_between_A_and_B : 
  ∃ x : ℝ, 
    (x / (rowing_speed_still_water - stream_speed) + x / (rowing_speed_still_water + stream_speed) = round_trip_time) 
    ∧ x = 24 :=
sorry

end distance_between_A_and_B_l2_2387


namespace simplify_expression_l2_2693

theorem simplify_expression : 
  (1 / (64^(1/3))^9) * 8^6 = 1 := by 
  have h1 : 64 = 2^6 := by rfl
  have h2 : 8 = 2^3 := by rfl
  sorry

end simplify_expression_l2_2693


namespace molecular_weight_of_one_mole_l2_2096

-- Definitions as Conditions
def total_molecular_weight := 960
def number_of_moles := 5

-- The theorem statement
theorem molecular_weight_of_one_mole :
  total_molecular_weight / number_of_moles = 192 :=
by
  sorry

end molecular_weight_of_one_mole_l2_2096


namespace opposite_of_3_l2_2833

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l2_2833


namespace num_winning_scenarios_l2_2230

-- Define the problem conditions
def total_tickets : ℕ := 8
def prize_tickets : ℕ := 3
def no_prize_tickets : ℕ := 5
def total_people : ℕ := 4
def tickets_per_person : ℕ := 2

-- Definition to capture the total number of different winning scenarios
theorem num_winning_scenarios : 
  ∃ n : ℕ, n = 60 ∧ 
    (total_tickets = 8) ∧ 
    (prize_tickets = 3) ∧ 
    (no_prize_tickets = 5) ∧ 
    (total_people = 4) ∧ 
    (tickets_per_person = 2) := by
  -- sorry allows us to skip the proof
  sorry

end num_winning_scenarios_l2_2230


namespace identify_quadratic_equation_l2_2890

-- Definitions of the equations
def eqA : Prop := ∀ x : ℝ, x^2 + 1/x^2 = 4
def eqB : Prop := ∀ (a b x : ℝ), a*x^2 + b*x - 3 = 0
def eqC : Prop := ∀ x : ℝ, (x - 1)*(x + 2) = 1
def eqD : Prop := ∀ (x y : ℝ), 3*x^2 - 2*x*y - 5*y^2 = 0

-- Definition that identifies whether a given equation is a quadratic equation in one variable
def isQuadraticInOneVariable (eq : Prop) : Prop := 
  ∃ (a b c : ℝ) (a0 : a ≠ 0), ∀ x : ℝ, eq = (a * x^2 + b * x + c = 0)

theorem identify_quadratic_equation :
  isQuadraticInOneVariable eqC :=
by
  sorry

end identify_quadratic_equation_l2_2890


namespace degree_of_expression_l2_2608

open Polynomial

noncomputable def expr1 : Polynomial ℤ := (monomial 5 3 - monomial 3 2 + 4) * (monomial 12 2 - monomial 8 1 + monomial 6 5 - 15)
noncomputable def expr2 : Polynomial ℤ := (monomial 3 2 - 4) ^ 6
noncomputable def final_expr : Polynomial ℤ := expr1 - expr2

theorem degree_of_expression : degree final_expr = 18 := by
  sorry

end degree_of_expression_l2_2608


namespace sum_abc_of_quadrilateral_l2_2123

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem sum_abc_of_quadrilateral :
  let p1 := (0, 0)
  let p2 := (4, 3)
  let p3 := (5, 2)
  let p4 := (4, -1)
  let perimeter := 
    distance p1 p2 + distance p2 p3 + distance p3 p4 + distance p4 p1
  let a : ℤ := 1    -- corresponding to the equivalent simplified distances to √5 parts
  let b : ℤ := 2    -- corresponding to the equivalent simplified distances to √2 parts
  let c : ℤ := 9    -- rest constant integer simplified part
  a + b + c = 12 :=
by
  sorry

end sum_abc_of_quadrilateral_l2_2123


namespace triangle_properties_l2_2640

open Real

variables (A B C a b c : ℝ) (triangle_obtuse triangle_right triangle_acute : Prop)

-- Declaration of properties 
def sin_gt (A B : ℝ) := sin A > sin B
def tan_product_lt (A C : ℝ) := tan A * tan C < 1
def cos_squared_eq (A B C : ℝ) := cos A ^ 2 + cos B ^ 2 - cos C ^ 2 = 1

theorem triangle_properties :
  (sin_gt A B → A > B) ∧
  (triangle_obtuse → tan_product_lt A C) ∧
  (cos_squared_eq A B C → triangle_right) :=
  by sorry

end triangle_properties_l2_2640


namespace find_f_of_2_l2_2610

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x - b

theorem find_f_of_2 (a b : ℝ) (h_pos : 0 < a)
  (h1 : ∀ x : ℝ, a * f x a b - b = 4 * x - 3)
  : f 2 a b = 3 := 
sorry

end find_f_of_2_l2_2610


namespace even_function_phi_l2_2609

noncomputable def phi := (3 * Real.pi) / 2

theorem even_function_phi (phi_val : Real) (hphi : 0 ≤ phi_val ∧ phi_val ≤ 2 * Real.pi) :
  (∀ x, Real.sin ((x + phi) / 3) = Real.sin ((-x + phi) / 3)) ↔ phi_val = phi := by
  sorry

end even_function_phi_l2_2609


namespace arithmetic_sequence_diff_l2_2342

theorem arithmetic_sequence_diff (a : ℕ → ℝ)
  (h1 : a 5 * a 7 = 6)
  (h2 : a 2 + a 10 = 5) :
  a 10 - a 6 = 2 ∨ a 10 - a 6 = -2 := by
  sorry

end arithmetic_sequence_diff_l2_2342


namespace solve_system_of_equations_l2_2524

theorem solve_system_of_equations :
  ∃ (x y : ℤ), (x - y = 2) ∧ (2 * x + y = 7) ∧ (x = 3) ∧ (y = 1) :=
by
  sorry

end solve_system_of_equations_l2_2524


namespace sum_of_integers_l2_2062

theorem sum_of_integers (x y : ℤ) (h_pos : 0 < y) (h_gt : x > y) (h_diff : x - y = 14) (h_prod : x * y = 48) : x + y = 20 :=
sorry

end sum_of_integers_l2_2062


namespace find_multiple_of_diff_l2_2400

theorem find_multiple_of_diff (n sum diff remainder k : ℕ) 
  (hn : n = 220070) 
  (hs : sum = 555 + 445) 
  (hd : diff = 555 - 445)
  (hr : remainder = 70)
  (hmod : n % sum = remainder) 
  (hquot : n / sum = k) :
  ∃ k, k = 2 ∧ k * diff = n / sum := 
by 
  sorry

end find_multiple_of_diff_l2_2400


namespace simplify_expression_eq_square_l2_2896

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l2_2896


namespace draw_at_least_one_even_ball_l2_2561

theorem draw_at_least_one_even_ball:
  -- Let the total number of ordered draws of 4 balls from 15 balls
  let total_draws := 15 * 14 * 13 * 12
  -- Let the total number of ordered draws of 4 balls where all balls are odd (balls 1, 3, ..., 15)
  let odd_draws := 8 * 7 * 6 * 5
  -- The number of valid draws containing at least one even ball
  total_draws - odd_draws = 31080 :=
by
  sorry

end draw_at_least_one_even_ball_l2_2561


namespace probability_at_most_one_red_light_l2_2633

def probability_of_no_red_light (p : ℚ) (n : ℕ) : ℚ := (1 - p) ^ n

def probability_of_exactly_one_red_light (p : ℚ) (n : ℕ) : ℚ :=
  (n.choose 1) * p ^ 1 * (1 - p) ^ (n - 1)

theorem probability_at_most_one_red_light (p : ℚ) (n : ℕ) (h : p = 1/3 ∧ n = 4) :
  probability_of_no_red_light p n + probability_of_exactly_one_red_light p n = 16 / 27 :=
by
  rw [h.1, h.2]
  sorry

end probability_at_most_one_red_light_l2_2633


namespace least_positive_integer_l2_2871

theorem least_positive_integer (x : ℕ) :
  (∃ k : ℤ, (3 * x + 41) ^ 2 = 53 * k) ↔ x = 4 :=
by
  sorry

end least_positive_integer_l2_2871


namespace sin_squared_plus_sin_double_eq_one_l2_2431

variable (α : ℝ)
variable (h : Real.tan α = 1 / 2)

theorem sin_squared_plus_sin_double_eq_one : Real.sin α ^ 2 + Real.sin (2 * α) = 1 :=
by
  -- sorry to indicate the proof is skipped
  sorry

end sin_squared_plus_sin_double_eq_one_l2_2431


namespace car_speed_problem_l2_2370

theorem car_speed_problem (x : ℝ) (h1 : ∀ x, x + 30 / 2 = 65) : x = 100 :=
by
  sorry

end car_speed_problem_l2_2370


namespace a1_a2_a3_sum_l2_2961

-- Given conditions and hypothesis
variables (a0 a1 a2 a3 : ℝ)
axiom H : ∀ x : ℝ, 1 + x + x^2 + x^3 = a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3

-- Goal statement to be proven
theorem a1_a2_a3_sum : a1 + a2 + a3 = -3 :=
sorry

end a1_a2_a3_sum_l2_2961


namespace opposite_of_3_is_neg3_l2_2832

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2832


namespace minimum_area_for_rectangle_l2_2575

theorem minimum_area_for_rectangle 
(length width : ℝ) 
(h_length_min : length = 4 - 0.5) 
(h_width_min : width = 5 - 1) :
length * width = 14 := 
by 
  simp [h_length_min, h_width_min]
  sorry

end minimum_area_for_rectangle_l2_2575


namespace number_of_refills_l2_2620

variable (totalSpent costPerRefill : ℕ)
variable (h1 : totalSpent = 40)
variable (h2 : costPerRefill = 10)

theorem number_of_refills (h1 h2 : totalSpent = 40) (h2 : costPerRefill = 10) :
  totalSpent / costPerRefill = 4 := by
  sorry

end number_of_refills_l2_2620


namespace remove_max_rooks_l2_2199

-- Defines the problem of removing the maximum number of rooks under given conditions
theorem remove_max_rooks (n : ℕ) (attacks_odd : (ℕ × ℕ) → ℕ) :
  (∀ p : ℕ × ℕ, (attacks_odd p) % 2 = 1 → true) →
  n = 8 →
  (∃ m, m = 59) :=
by
  intros _ _
  existsi 59
  sorry

end remove_max_rooks_l2_2199


namespace opposite_of_3_l2_2850

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l2_2850


namespace shpuntik_can_form_triangle_l2_2373

-- Define lengths of the sticks before swap
variables {a b c d e f : ℝ}

-- Conditions before the swap
-- Both sets of sticks can form a triangle
-- The lengths of Vintik's sticks are a, b, c
-- The lengths of Shpuntik's sticks are d, e, f
axiom triangle_ineq_vintik : a + b > c ∧ b + c > a ∧ c + a > b
axiom triangle_ineq_shpuntik : d + e > f ∧ e + f > d ∧ f + d > e
axiom sum_lengths_vintik : a + b + c = 1
axiom sum_lengths_shpuntik : d + e + f = 1

-- Define lengths of the sticks after swap
-- x1, x2, x3 are Vintik's new sticks; y1, y2, y3 are Shpuntik's new sticks
variables {x1 x2 x3 y1 y2 y3 : ℝ}

-- Neznaika's swap
axiom swap_stick_vintik : x1 = a ∧ x2 = b ∧ x3 = f ∨ x1 = a ∧ x2 = d ∧ x3 = c ∨ x1 = e ∧ x2 = b ∧ x3 = c
axiom swap_stick_shpuntik : y1 = d ∧ y2 = e ∧ y3 = c ∨ y1 = e ∧ y2 = b ∧ y3 = f ∨ y1 = a ∧ y2 = b ∧ y3 = f 

-- Total length after the swap remains unchanged
axiom sum_lengths_after_swap : x1 + x2 + x3 + y1 + y2 + y3 = 2

-- Vintik cannot form a triangle with the current lengths
axiom no_triangle_vintik : x1 >= x2 + x3

-- Prove that Shpuntik can still form a triangle
theorem shpuntik_can_form_triangle : y1 + y2 > y3 ∧ y2 + y3 > y1 ∧ y3 + y1 > y2 := sorry

end shpuntik_can_form_triangle_l2_2373


namespace charlie_ride_distance_l2_2216

-- Define the known values
def oscar_ride : ℝ := 0.75
def difference : ℝ := 0.5

-- Define Charlie's bus ride distance
def charlie_ride : ℝ := oscar_ride - difference

-- The theorem to be proven
theorem charlie_ride_distance : charlie_ride = 0.25 := 
by sorry

end charlie_ride_distance_l2_2216


namespace pathway_bricks_total_is_280_l2_2479

def total_bricks (n : ℕ) : ℕ :=
  let odd_bricks := 2 * (1 + 1 + ((n / 2) - 1) * 2)
  let even_bricks := 4 * (1 + 2 + (n / 2 - 1) * 2)
  odd_bricks + even_bricks
   
theorem pathway_bricks_total_is_280 (n : ℕ) (h : total_bricks n = 280) : n = 10 :=
sorry

end pathway_bricks_total_is_280_l2_2479


namespace overall_average_output_l2_2388

theorem overall_average_output 
  (initial_cogs : ℕ := 60) 
  (rate_1 : ℕ := 36) 
  (rate_2 : ℕ := 60) 
  (second_batch_cogs : ℕ := 60) :
  (initial_cogs + second_batch_cogs) / ((initial_cogs / rate_1) + (second_batch_cogs / rate_2)) = 45 := 
  sorry

end overall_average_output_l2_2388


namespace factorial_division_l2_2729

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem factorial_division :
  (factorial 17) / (factorial 7 * factorial 10) = 408408 := by
  sorry

end factorial_division_l2_2729


namespace problem_statement_l2_2607

open Real

noncomputable def f (x : ℝ) : ℝ := 10^x

theorem problem_statement : f (log 2) * f (log 5) = 10 :=
by {
  -- Note: Proof is omitted as indicated in the procedure.
  sorry
}

end problem_statement_l2_2607


namespace find_abc_sum_l2_2056

theorem find_abc_sum (a b c : ℕ) (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 294) : a + b + c = 8 :=
sorry

end find_abc_sum_l2_2056


namespace perpendicular_parallel_l2_2439

variables {a b : Line} {α : Plane}

-- Definition of perpendicular and parallel relations should be available
-- since their exact details were not provided, placeholder functions will be used for demonstration

-- Placeholder definitions for perpendicular and parallel (they should be accurately defined elsewhere)
def perp (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry

theorem perpendicular_parallel {a b : Line} {α : Plane}
    (a_perp_alpha : perp a α)
    (b_perp_alpha : perp b α)
    : parallel a b :=
sorry

end perpendicular_parallel_l2_2439


namespace daily_evaporation_rate_l2_2394

theorem daily_evaporation_rate (initial_amount : ℝ) (period : ℕ) (percentage_evaporated : ℝ) (h_initial : initial_amount = 10) (h_period : period = 50) (h_percentage : percentage_evaporated = 4) : 
  (percentage_evaporated / 100 * initial_amount) / period = 0.008 :=
by
  -- Ensures that the conditions translate directly into the Lean theorem statement
  rw [h_initial, h_period, h_percentage]
  -- Insert the required logical proof here
  sorry

end daily_evaporation_rate_l2_2394


namespace number_of_student_tickets_sold_l2_2543

variable (A S : ℝ)

theorem number_of_student_tickets_sold
  (h1 : A + S = 59)
  (h2 : 4 * A + 2.5 * S = 222.50) :
  S = 9 :=
by sorry

end number_of_student_tickets_sold_l2_2543


namespace total_commission_l2_2581

-- Define the commission rate
def commission_rate : ℝ := 0.02

-- Define the sale prices of the three houses
def sale_price1 : ℝ := 157000
def sale_price2 : ℝ := 499000
def sale_price3 : ℝ := 125000

-- Total commission calculation
theorem total_commission :
  (commission_rate * sale_price1 + commission_rate * sale_price2 + commission_rate * sale_price3) = 15620 := 
by
  sorry

end total_commission_l2_2581


namespace virginia_avg_rainfall_l2_2683

theorem virginia_avg_rainfall:
  let march := 3.79
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let total_rainfall := march + april + may + june + july
  let avg_rainfall := total_rainfall / 5
  avg_rainfall = 4 := by sorry

end virginia_avg_rainfall_l2_2683


namespace total_amount_l2_2797

noncomputable def mark_amount : ℝ := 5 / 8

noncomputable def carolyn_amount : ℝ := 7 / 20

theorem total_amount : mark_amount + carolyn_amount = 0.975 := by
  sorry

end total_amount_l2_2797


namespace smallest_prime_divides_polynomial_l2_2162

theorem smallest_prime_divides_polynomial : 
  ∃ n : ℤ, n^2 + 5 * n + 23 = 17 := 
sorry

end smallest_prime_divides_polynomial_l2_2162


namespace minimum_value_of_f_l2_2938

def f (x : ℝ) : ℝ := |3 - x| + |x - 2|

theorem minimum_value_of_f : ∃ x0 : ℝ, (∀ x : ℝ, f x0 ≤ f x) ∧ f x0 = 1 := 
by
  sorry

end minimum_value_of_f_l2_2938


namespace scientific_notation_population_l2_2501

theorem scientific_notation_population :
    ∃ (a b : ℝ), (b = 5 ∧ 1412.60 * 10 ^ 6 = a * 10 ^ b ∧ a = 1.4126) :=
sorry

end scientific_notation_population_l2_2501


namespace expand_and_simplify_l2_2235

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l2_2235


namespace length_of_AB_l2_2356

theorem length_of_AB
  (P Q : ℝ) (AB : ℝ)
  (hP : P = 3 / 7 * AB)
  (hQ : Q = 4 / 9 * AB)
  (hPQ : abs (Q - P) = 3) :
  AB = 189 :=
by
  sorry

end length_of_AB_l2_2356


namespace option_c_opposites_l2_2251

theorem option_c_opposites : -|3| = -3 ∧ 3 = 3 → ( ∃ x y : ℝ, x = -3 ∧ y = 3 ∧ x = -y) :=
by
  sorry

end option_c_opposites_l2_2251


namespace cubes_divisible_by_9_l2_2231

theorem cubes_divisible_by_9 (n: ℕ) (h: n > 0) : 9 ∣ n^3 + (n + 1)^3 + (n + 2)^3 :=
by 
  sorry

end cubes_divisible_by_9_l2_2231


namespace parallelogram_base_length_l2_2703

theorem parallelogram_base_length 
  (area : ℝ)
  (b h : ℝ)
  (h_area : area = 128)
  (h_altitude : h = 2 * b) 
  (h_area_eq : area = b * h) : 
  b = 8 :=
by
  -- Proof goes here
  sorry

end parallelogram_base_length_l2_2703


namespace p_necessary_for_q_l2_2202

-- Definitions
def p (a b : ℝ) : Prop := (a + b = 2) ∨ (a + b = -2)
def q (a b : ℝ) : Prop := a + b = 2

-- Statement of the problem
theorem p_necessary_for_q (a b : ℝ) : (p a b → q a b) ∧ ¬(q a b → p a b) := 
sorry

end p_necessary_for_q_l2_2202


namespace opposite_of_3_l2_2834

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l2_2834


namespace tan_seven_pi_over_four_l2_2294

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by
  sorry

end tan_seven_pi_over_four_l2_2294


namespace min_value_problem_l2_2596

theorem min_value_problem (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 57 * a + 88 * b + 125 * c ≥ 1148) :
  240 ≤ a^3 + b^3 + c^3 + 5 * a^2 + 5 * b^2 + 5 * c^2 :=
sorry

end min_value_problem_l2_2596


namespace f_4_1981_eq_l2_2533

noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| (x+1), 0 => f x 1
| (x+1), (y+1) => f x (f (x+1) y)

theorem f_4_1981_eq : f 4 1981 = 2^1984 - 3 := 
by
  sorry

end f_4_1981_eq_l2_2533


namespace arithmetic_arrangement_result_l2_2142

theorem arithmetic_arrangement_result :
    (1 / 8) * (1 / 9) * (1 / 28) = 1 / 2016 ∨ ((1 / 8) - (1 / 9)) * (1 / 28) = 1 / 2016 :=
by {
    sorry
}

end arithmetic_arrangement_result_l2_2142


namespace alcohol_percentage_l2_2132

theorem alcohol_percentage (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 100) 
(h3 : (0.6 + (x / 100) * 6 = 2.4)) : x = 30 :=
by sorry

end alcohol_percentage_l2_2132


namespace negation_of_at_most_one_odd_l2_2381

variable (a b c : ℕ)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def at_most_one_odd (a b c : ℕ) : Prop :=
  (is_odd a ∧ ¬is_odd b ∧ ¬is_odd c) ∨
  (¬is_odd a ∧ is_odd b ∧ ¬is_odd c) ∨
  (¬is_odd a ∧ ¬is_odd b ∧ is_odd c) ∨
  (¬is_odd a ∧ ¬is_odd b ∧ ¬is_odd c)

theorem negation_of_at_most_one_odd :
  ¬ at_most_one_odd a b c ↔
  ∃ x y, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ x ≠ y ∧ is_odd x ∧ is_odd y :=
sorry

end negation_of_at_most_one_odd_l2_2381


namespace find_function_that_satisfies_eq_l2_2981

theorem find_function_that_satisfies_eq :
  ∀ (f : ℕ → ℕ), (∀ (m n : ℕ), f (m + f n) = f (f m) + f n) → (∀ n : ℕ, f n = n) :=
by
  intro f
  intro h
  sorry

end find_function_that_satisfies_eq_l2_2981


namespace rotation_transforms_and_sums_l2_2196

theorem rotation_transforms_and_sums 
    (D E F D' E' F' : (ℝ × ℝ))
    (hD : D = (0, 0)) (hE : E = (0, 20)) (hF : F = (30, 0)) 
    (hD' : D' = (-26, 23)) (hE' : E' = (-46, 23)) (hF' : F' = (-26, -7))
    (n : ℝ) (x y : ℝ)
    (rotation_condition : 0 < n ∧ n < 180)
    (angle_condition : n = 90) :
    n + x + y = 60.5 :=
by
  have hx : x = -49 := sorry
  have hy : y = 19.5 := sorry
  have hn : n = 90 := sorry
  sorry

end rotation_transforms_and_sums_l2_2196


namespace center_of_circle_eq_minus_two_four_l2_2299

theorem center_of_circle_eq_minus_two_four : 
  ∀ (x y : ℝ), x^2 + 4 * x + y^2 - 8 * y + 16 = 0 → (x, y) = (-2, 4) :=
by {
  sorry
}

end center_of_circle_eq_minus_two_four_l2_2299


namespace cut_piece_ratio_l2_2642

noncomputable def original_log_length : ℕ := 20
noncomputable def weight_per_foot : ℕ := 150
noncomputable def cut_piece_weight : ℕ := 1500

theorem cut_piece_ratio :
  (cut_piece_weight / weight_per_foot / original_log_length) = (1 / 2) := by
  sorry

end cut_piece_ratio_l2_2642


namespace range_of_y_when_x_3_l2_2001

variable (a c : ℝ)

theorem range_of_y_when_x_3 (h1 : -4 ≤ a + c ∧ a + c ≤ -1) (h2 : -1 ≤ 4 * a + c ∧ 4 * a + c ≤ 5) :
  -1 ≤ 9 * a + c ∧ 9 * a + c ≤ 20 :=
sorry

end range_of_y_when_x_3_l2_2001


namespace properties_of_f_l2_2320

open Real 

def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem properties_of_f :
  (∀ x : ℝ, f x = x / (x^2 + 1)) ∧
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < 1 → 0 < x2 → x2 < 1 → x1 < x2 → f x1 < f x2) ∧
  (range f = Set.Icc (-(1/2 : ℝ)) (1/2)) := 
by
  sorry

end properties_of_f_l2_2320


namespace one_integral_root_exists_l2_2927

theorem one_integral_root_exists :
    ∃! x : ℤ, x - 8 / (x - 3) = 2 - 8 / (x - 3) :=
by
  sorry

end one_integral_root_exists_l2_2927


namespace least_positive_angle_l2_2935

theorem least_positive_angle (θ : ℝ) (h : Real.cos (10 * Real.pi / 180) = Real.sin (15 * Real.pi / 180) + Real.sin θ) :
  θ = 32.5 * Real.pi / 180 := 
sorry

end least_positive_angle_l2_2935


namespace required_remaining_speed_l2_2144

-- Definitions for the given problem
variables (D T : ℝ) 

-- Given conditions from the problem
def speed_first_part (D T : ℝ) : Prop := 
  40 = (2 * D / 3) / (T / 3)

def remaining_distance_time (D T : ℝ) : Prop :=
  10 = (D / 3) / (2 * (2 * D / 3) / 40 / 3)

-- Theorem to be proved
theorem required_remaining_speed (D T : ℝ) 
  (h1 : speed_first_part D T)
  (h2 : remaining_distance_time D T) :
  10 = (D / 3) / (2 * (T / 3)) :=
  sorry  -- Proof is skipped

end required_remaining_speed_l2_2144


namespace speed_of_current_l2_2679

-- Definitions
def speed_boat_still_water := 60
def speed_downstream := 77
def speed_upstream := 43

-- Theorem statement
theorem speed_of_current : ∃ x, speed_boat_still_water + x = speed_downstream ∧ speed_boat_still_water - x = speed_upstream ∧ x = 17 :=
by
  unfold speed_boat_still_water speed_downstream speed_upstream
  sorry

end speed_of_current_l2_2679


namespace factory_profit_l2_2398

def cost_per_unit : ℝ := 2.00
def fixed_cost : ℝ := 500.00
def selling_price_per_unit : ℝ := 2.50

theorem factory_profit (x : ℕ) (hx : x > 1000) :
  selling_price_per_unit * x > fixed_cost + cost_per_unit * x :=
by
  sorry

end factory_profit_l2_2398


namespace isosceles_trapezoid_area_l2_2407

theorem isosceles_trapezoid_area (x y : ℝ)
  (h1 : 0.8 = real.sin (real.arcsin (0.8)))
  (h2 : 16 = y + 1.2 * x)
  (h3 : 2 * y + 1.2 * x = 2 * x) 
  (h4 : x = 10)
  (h5 : y = 4)
  : 1 / 2 * (4 + 16) * (0.8 * 10) = 80 :=
by
  sorry


end isosceles_trapezoid_area_l2_2407


namespace sale_in_fifth_month_l2_2883

theorem sale_in_fifth_month (Sale1 Sale2 Sale3 Sale4 Sale6 AvgSale : ℤ) 
(h1 : Sale1 = 6435) (h2 : Sale2 = 6927) (h3 : Sale3 = 6855) (h4 : Sale4 = 7230) 
(h5 : Sale6 = 4991) (h6 : AvgSale = 6500) : (39000 - (Sale1 + Sale2 + Sale3 + Sale4 + Sale6)) = 6562 :=
by
  sorry

end sale_in_fifth_month_l2_2883


namespace value_of_a_l2_2335

theorem value_of_a (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x - 1 = 0) ∧ (∀ x y : ℝ, a * x^2 - 2 * x - 1 = 0 ∧ a * y^2 - 2 * y - 1 = 0 → x = y) ↔ (a = 0 ∨ a = -1) :=
by
  sorry

end value_of_a_l2_2335


namespace arithmetic_sequence_num_terms_l2_2182

theorem arithmetic_sequence_num_terms 
  (a : ℕ) (d : ℕ) (l : ℕ) (n : ℕ)
  (h1 : a = 20)
  (h2 : d = 5)
  (h3 : l = 150)
  (h4 : 150 = 20 + (n-1) * 5) :
  n = 27 :=
by sorry

end arithmetic_sequence_num_terms_l2_2182


namespace max_minute_hands_l2_2101

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
sorry

end max_minute_hands_l2_2101


namespace average_weight_increase_l2_2059

theorem average_weight_increase 
  (A : ℝ) (X : ℝ)
  (h1 : 8 * (A + X) = 8 * A + 36) :
  X = 4.5 := 
sorry

end average_weight_increase_l2_2059


namespace geometric_sequence_sum_l2_2970

def a (n : ℕ) : ℕ := 3 * (2 ^ (n - 1))

theorem geometric_sequence_sum :
  a 1 = 3 → a 4 = 24 → (a 3 + a 4 + a 5) = 84 :=
by
  intros h1 h4
  sorry

end geometric_sequence_sum_l2_2970


namespace least_positive_angle_l2_2937

theorem least_positive_angle (θ : ℝ) (h : θ > 0 ∧ θ ≤ 360) : 
  (cos 10 = sin 15 + sin θ) → θ = 32.5 :=
by 
  sorry

end least_positive_angle_l2_2937


namespace arithmetic_sequence_sum_geometric_sequence_ratio_l2_2437

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a (n + 1) = a n + d

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℕ) (q : ℚ) :=
  ∀ n, a (n + 1) = a n * q
  
-- Prove the sum of the first n terms for an arithmetic sequence
theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 3 ∧ (∀ n, S n = (n * (3 + a (n + 1) - 1)) / 2) ∧ is_arithmetic_sequence a 4 → 
  S n = 2 * n^2 + n :=
sorry

-- Prove the range of the common ratio for a geometric sequence
theorem geometric_sequence_ratio (a : ℕ → ℕ) (S : ℕ → ℚ) (q : ℚ) :
  a 1 = 3 ∧ is_geometric_sequence a q ∧ ∃ lim : ℚ, (∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) ∧ lim < 12 → 
  -1 < q ∧ q < 1 ∧ q ≠ 0 ∧ q < 3/4 :=
sorry

end arithmetic_sequence_sum_geometric_sequence_ratio_l2_2437


namespace find_ac_find_a_and_c_l2_2336

variables (A B C a b c : ℝ)

-- Condition: Angles A, B, C form an arithmetic sequence.
def arithmetic_sequence := 2 * B = A + C

-- Condition: Area of the triangle is sqrt(3)/2.
def area_triangle := (1/2) * a * c * (Real.sin B) = (Real.sqrt 3) / 2

-- Condition: b = sqrt(3)
def b_sqrt3 := b = Real.sqrt 3

-- Goal 1: To prove that ac = 2.
theorem find_ac (h1 : arithmetic_sequence A B C) (h2 : area_triangle a c B) : a * c = 2 :=
sorry

-- Goal 2: To prove a = 2 and c = 1 given the additional condition.
theorem find_a_and_c (h1 : arithmetic_sequence A B C) (h2 : area_triangle a c B) (h3 : b_sqrt3 b) (h4 : a > c) : a = 2 ∧ c = 1 :=
sorry

end find_ac_find_a_and_c_l2_2336


namespace stickers_left_correct_l2_2800

-- Define the initial number of stickers and number of stickers given away
def n_initial : ℝ := 39.0
def n_given_away : ℝ := 22.0

-- Proof statement: The number of stickers left at the end is 17.0
theorem stickers_left_correct : n_initial - n_given_away = 17.0 := by
  sorry

end stickers_left_correct_l2_2800


namespace expression_simplification_l2_2916

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l2_2916


namespace heaviest_person_is_Vanya_l2_2279

variables (A D T V M : ℕ)

-- conditions
def condition1 : Prop := A + D = 82
def condition2 : Prop := D + T = 74
def condition3 : Prop := T + V = 75
def condition4 : Prop := V + M = 65
def condition5 : Prop := M + A = 62

theorem heaviest_person_is_Vanya (h1 : condition1 A D) (h2 : condition2 D T) (h3 : condition3 T V) (h4 : condition4 V M) (h5 : condition5 M A) :
  V = 43 :=
sorry

end heaviest_person_is_Vanya_l2_2279


namespace frac_pow_eq_l2_2413

theorem frac_pow_eq : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by 
  sorry

end frac_pow_eq_l2_2413


namespace ratio_of_boys_to_girls_l2_2337

def boys_girls_ratio (b g : ℕ) : ℚ := b / g

theorem ratio_of_boys_to_girls (b g : ℕ) (h1 : b = g + 6) (h2 : g + b = 40) :
  boys_girls_ratio b g = 23 / 17 :=
by
  sorry

end ratio_of_boys_to_girls_l2_2337


namespace balls_in_boxes_l2_2802

theorem balls_in_boxes :
  ∃ (f : Fin 5 → Fin 3), 
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ b : Fin 3, ∃ i, f i = b) ∧
    f 0 ≠ f 1 :=
  sorry

end balls_in_boxes_l2_2802


namespace coefficient_of_x2_in_expansion_is_neg3_l2_2222

noncomputable def coefficient_x2_in_expansion : ℤ :=
  let f := (1 - X)^6 * (1 + X)^4
  in coeff (f.expand) 2

theorem coefficient_of_x2_in_expansion_is_neg3 :
  coefficient_x2_in_expansion = -3 := sorry

end coefficient_of_x2_in_expansion_is_neg3_l2_2222


namespace exists_line_through_ellipse_diameter_circle_origin_l2_2448

theorem exists_line_through_ellipse_diameter_circle_origin :
  ∃ m : ℝ, (m = (4 * Real.sqrt 3) / 3 ∨ m = -(4 * Real.sqrt 3) / 3) ∧
  ∀ (x y : ℝ), (x^2 + 2 * y^2 = 8) → (y = x + m) → (x^2 + (x + m)^2 = 8) :=
by
  sorry

end exists_line_through_ellipse_diameter_circle_origin_l2_2448


namespace repeating_decimal_fraction_l2_2157

theorem repeating_decimal_fraction :
  (5 + 341 / 999) = (5336 / 999) :=
by
  sorry

end repeating_decimal_fraction_l2_2157


namespace expand_simplify_expression_l2_2248

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l2_2248


namespace scientific_notation_141260_million_l2_2494

theorem scientific_notation_141260_million :
  ∃ (a : ℝ) (n : ℤ), 141260 * 10^6 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4126 ∧ n = 5 :=
by
  sorry

end scientific_notation_141260_million_l2_2494


namespace opposite_of_3_is_neg3_l2_2830

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2830


namespace product_213_16_l2_2329

theorem product_213_16 :
  (213 * 16 = 3408) :=
by
  have h1 : (0.16 * 2.13 = 0.3408) := by sorry
  sorry

end product_213_16_l2_2329


namespace bounds_for_f3_l2_2760

variable (a c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 - c

theorem bounds_for_f3 (h1 : -4 ≤ f a c 1 ∧ f a c 1 ≤ -1)
                      (h2 : -1 ≤ f a c 2 ∧ f a c 2 ≤ 5) :
  -1 ≤ f a c 3 ∧ f a c 3 ≤ 20 := 
sorry

end bounds_for_f3_l2_2760


namespace find_y_l2_2459

theorem find_y (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -5) : y = 18 := by
  -- Proof can go here
  sorry

end find_y_l2_2459


namespace minimize_abs_expression_l2_2695

theorem minimize_abs_expression {x : ℝ} : 
  ((|x - 2|) + 3) ≥ ((|2 - 2|) + 3) := 
sorry

end minimize_abs_expression_l2_2695


namespace scooter_price_and_installment_l2_2347

variable {P : ℝ} -- price of the scooter
variable {m : ℝ} -- monthly installment

theorem scooter_price_and_installment (h1 : 0.2 * P = 240) (h2 : (0.8 * P) = 12 * m) : 
  P = 1200 ∧ m = 80 := by
  sorry

end scooter_price_and_installment_l2_2347


namespace face_opposite_of_E_l2_2924

-- Definitions of faces and their relationships
inductive Face : Type
| A | B | C | D | E | F | x

open Face

-- Adjacency relationship
def is_adjacent_to (f1 f2 : Face) : Prop :=
(f1 = x ∧ (f2 = A ∨ f2 = B ∨ f2 = C ∨ f2 = D)) ∨
(f2 = x ∧ (f1 = A ∨ f1 = B ∨ f1 = C ∨ f1 = D)) ∨
(f1 = E ∧ (f2 = A ∨ f2 = B ∨ f2 = C ∨ f2 = D)) ∨
(f2 = E ∧ (f1 = A ∨ f1 = B ∨ f1 = C ∨ f1 = D))

-- Non-adjacency relationship
def is_opposite (f1 f2 : Face) : Prop :=
∀ f : Face, is_adjacent_to f1 f → ¬ is_adjacent_to f2 f

-- Theorem to prove that F is opposite of E
theorem face_opposite_of_E : is_opposite E F :=
sorry

end face_opposite_of_E_l2_2924


namespace pq_eq_neg72_l2_2644

theorem pq_eq_neg72 {p q : ℝ} (h : ∀ x, (x - 7) * (3 * x + 11) = x ^ 2 - 20 * x + 63 →
(p = x ∨ q = x) ∧ p ≠ q) : 
(p + 2) * (q + 2) = -72 :=
sorry

end pq_eq_neg72_l2_2644


namespace set_equality_proof_l2_2985

theorem set_equality_proof :
  (∃ (u : ℤ), ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l) ↔
  (∃ (u : ℤ), ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r) :=
sorry

end set_equality_proof_l2_2985


namespace rectangle_equation_l2_2637

-- Given points in the problem, we define the coordinates
def A : ℝ × ℝ := (5, 5)
def B : ℝ × ℝ := (9, 2)
def C (a : ℝ) : ℝ × ℝ := (a, 13)
def D (b : ℝ) : ℝ × ℝ := (15, b)

-- We need to prove that a - b = 1 given the conditions
theorem rectangle_equation (a b : ℝ) (h1 : C a = (a, 13)) (h2 : D b = (15, b)) (h3 : 15 - a = 4) (h4 : 13 - b = 3) : 
     a - b = 1 := 
sorry

end rectangle_equation_l2_2637


namespace opposite_of_3_l2_2847

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l2_2847


namespace arithmetic_expression_equals_fraction_l2_2140

theorem arithmetic_expression_equals_fraction (a b c : ℚ) :
  a = 1/8 → b = 1/9 → c = 1/28 →
  (a * b * c = 1/2016) ∨ ((a - b) * c = 1/2016) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  left
  sorry

end arithmetic_expression_equals_fraction_l2_2140


namespace subtraction_correct_l2_2664

theorem subtraction_correct : 900000009000 - 123456789123 = 776543220777 :=
by
  -- Placeholder proof to ensure it compiles
  sorry

end subtraction_correct_l2_2664


namespace circle_center_l2_2624

theorem circle_center (x y : ℝ) :
  x^2 + y^2 - 2*x + 4*y + 1 = 0 → (1, -2) = (1, -2) :=
by
  sorry

end circle_center_l2_2624


namespace tank_capacity_l2_2460

theorem tank_capacity (T : ℝ) (h : (3 / 4) * T + 7 = (7 / 8) * T) : T = 56 := 
sorry

end tank_capacity_l2_2460


namespace max_additional_plates_l2_2221

def initial_plates_count : ℕ := 5 * 3 * 4 * 2
def new_second_set_size : ℕ := 5  -- second set after adding two letters
def new_fourth_set_size : ℕ := 3 -- fourth set after adding one letter
def new_plates_count : ℕ := 5 * new_second_set_size * 4 * new_fourth_set_size

theorem max_additional_plates :
  new_plates_count - initial_plates_count = 180 := by
  sorry

end max_additional_plates_l2_2221


namespace A_worked_alone_after_B_left_l2_2393

/-- A and B can together finish a work in 40 days. They worked together for 10 days and then B left.
    A alone can finish the job in 80 days. We need to find out how many days did A work alone after B left. -/
theorem A_worked_alone_after_B_left
  (W : ℝ)
  (A_work_rate : ℝ := W / 80)
  (B_work_rate : ℝ := W / 80)
  (AB_work_rate : ℝ := W / 40)
  (work_done_together_in_10_days : ℝ := 10 * (W / 40))
  (remaining_work : ℝ := W - work_done_together_in_10_days)
  (A_rate_alone : ℝ := W / 80) :
  ∃ D : ℝ, D * (W / 80) = remaining_work → D = 60 :=
by
  sorry

end A_worked_alone_after_B_left_l2_2393


namespace tournament_games_count_l2_2264

-- We define the conditions
def number_of_players : ℕ := 6

-- Function to calculate the number of games played in a tournament where each player plays twice with each opponent
def total_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- Now we state the theorem
theorem tournament_games_count : total_games number_of_players = 60 := by
  -- Proof goes here
  sorry

end tournament_games_count_l2_2264


namespace determine_number_l2_2267

theorem determine_number (x : ℝ) (number : ℝ) (h1 : number / x = 0.03) (h2 : x = 0.3) : number = 0.009 := by
  sorry

end determine_number_l2_2267


namespace opposite_of_3_l2_2848

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l2_2848


namespace max_area_of_2m_wire_l2_2621

theorem max_area_of_2m_wire (P : ℝ) (l w : ℝ) (a : ℝ) :
  P = 2 → 2 * (l + w) = 2 → (a = l * w → l + w = 1) → max l w = (1 / 4) :=
by
  sorry

end max_area_of_2m_wire_l2_2621


namespace seat_arrangement_l2_2341

theorem seat_arrangement :
  ∃ (arrangement : Fin 7 → String), 
  (arrangement 6 = "Diane") ∧
  (∃ (i j : Fin 7), i < j ∧ arrangement i = "Carla" ∧ arrangement j = "Adam" ∧ j = (i + 1)) ∧
  (∃ (i j k : Fin 7), i < j ∧ j < k ∧ arrangement i = "Brian" ∧ arrangement j = "Ellie" ∧ (k - i) ≥ 3) ∧
  arrangement 3 = "Carla" := 
sorry

end seat_arrangement_l2_2341


namespace percentage_of_sikh_boys_is_10_l2_2465

theorem percentage_of_sikh_boys_is_10 (total_boys : ℕ)
  (perc_muslim : ℝ) (perc_hindu : ℝ) (other_comm_boys : ℕ)
  (H_total_boys : total_boys = 850)
  (H_perc_muslim : perc_muslim = 0.40)
  (H_perc_hindu : perc_hindu = 0.28)
  (H_other_comm_boys : other_comm_boys = 187) :
  ((total_boys - ( (perc_muslim * total_boys) + (perc_hindu * total_boys) + other_comm_boys)) / total_boys) * 100 = 10 :=
by
  sorry

end percentage_of_sikh_boys_is_10_l2_2465


namespace plants_remaining_l2_2412

theorem plants_remaining (plants_initial plants_first_day plants_second_day_eaten plants_third_day_eaten : ℕ)
  (h1 : plants_initial = 30)
  (h2 : plants_first_day = 20)
  (h3 : plants_second_day_eaten = (plants_initial - plants_first_day) / 2)
  (h4 : plants_third_day_eaten = 1)
  : (plants_initial - plants_first_day - plants_second_day_eaten - plants_third_day_eaten) = 4 := 
by
  sorry

end plants_remaining_l2_2412


namespace isosceles_triangle_apex_angle_l2_2314

theorem isosceles_triangle_apex_angle (a b c : ℝ) (ha : a = 40) (hb : b = 40) (hc : b = c) :
  (a + b + c = 180) → (c = 100 ∨ a = 40) :=
by
-- We start the proof and provide the conditions.
  sorry  -- Lean expects the proof here.

end isosceles_triangle_apex_angle_l2_2314


namespace investment_in_scheme_B_l2_2137

theorem investment_in_scheme_B 
    (yieldA : ℝ) (yieldB : ℝ) (investmentA : ℝ) (difference : ℝ) (totalA : ℝ) (totalB : ℝ):
    yieldA = 0.30 → yieldB = 0.50 → investmentA = 300 → difference = 90 
    → totalA = investmentA + (yieldA * investmentA) 
    → totalB = (1 + yieldB) * totalB 
    → totalA = totalB + difference 
    → totalB = 200 :=
by sorry

end investment_in_scheme_B_l2_2137


namespace polynomial_divisibility_l2_2219

theorem polynomial_divisibility (n : ℕ) (h : 0 < n) : 
  ∃ g : Polynomial ℚ, 
    (Polynomial.X + 1)^(2*n + 1) + Polynomial.X^(n + 2) = g * (Polynomial.X^2 + Polynomial.X + 1) := 
by
  sorry

end polynomial_divisibility_l2_2219


namespace positive_integer_satisfies_condition_l2_2806

def num_satisfying_pos_integers : ℕ :=
  1

theorem positive_integer_satisfies_condition :
  ∃ (n : ℕ), 16 - 4 * n > 10 ∧ n = num_satisfying_pos_integers := by
  sorry

end positive_integer_satisfies_condition_l2_2806


namespace surface_area_of_large_cube_l2_2397

theorem surface_area_of_large_cube (l w h : ℕ) (cube_side : ℕ) 
  (volume_cuboid : ℕ := l * w * h) 
  (n_cubes := volume_cuboid / (cube_side ^ 3))
  (side_length_large_cube : ℕ := cube_side * (n_cubes^(1/3 : ℕ))) 
  (surface_area_large_cube : ℕ := 6 * (side_length_large_cube ^ 2)) :
  l = 25 → w = 10 → h = 4 → cube_side = 1 → surface_area_large_cube = 600 :=
by
  intros hl hw hh hcs
  subst hl
  subst hw
  subst hh
  subst hcs
  sorry

end surface_area_of_large_cube_l2_2397


namespace abs_sub_nonneg_l2_2327

theorem abs_sub_nonneg (a : ℝ) : |a| - a ≥ 0 :=
sorry

end abs_sub_nonneg_l2_2327


namespace degree_measure_of_supplement_of_complement_of_35_degree_angle_l2_2083

def complement (α : ℝ) : ℝ := 90 - α
def supplement (β : ℝ) : ℝ := 180 - β

theorem degree_measure_of_supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 :=
by
  sorry

end degree_measure_of_supplement_of_complement_of_35_degree_angle_l2_2083


namespace zero_in_interval_l2_2372

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 9

theorem zero_in_interval :
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) → -- f(x) is increasing on (0, +∞)
  f 2 < 0 → -- f(2) < 0
  f 3 > 0 → -- f(3) > 0
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  intros h_increasing h_f2_lt_0 h_f3_gt_0
  sorry

end zero_in_interval_l2_2372


namespace line_PQ_passes_through_fixed_point_l2_2611

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 2 = 1

-- Define the conditions for points P and Q on the hyperbola
def on_hyperbola (P Q : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2

-- Define the condition for perpendicular lines, given points A, P, and Q
def perpendicular (A P Q : ℝ × ℝ) : Prop :=
  ((P.2 - A.2) / (P.1 - A.1)) * ((Q.2 - A.2) / (Q.1 - A.1)) = -1

-- Define the main theorem to prove
theorem line_PQ_passes_through_fixed_point :
  ∀ (P Q : ℝ × ℝ), on_hyperbola P Q → perpendicular ⟨-1, 0⟩ P Q →
    ∃ (b : ℝ), ∀ (y : ℝ), (P.1 = y * P.2 + b ∨ Q.1 = y * Q.2 + b) → (b = 3) :=
by
  sorry

end line_PQ_passes_through_fixed_point_l2_2611


namespace reciprocal_of_mixed_num_l2_2536

-- Define the fraction representation of the mixed number -1 1/2
def mixed_num_to_improper (a : ℚ) : ℚ := -3/2

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Prove the statement
theorem reciprocal_of_mixed_num : reciprocal (mixed_num_to_improper (-1.5)) = -2/3 :=
by
  -- skip proof
  sorry

end reciprocal_of_mixed_num_l2_2536


namespace simplify_expression_l2_2923

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l2_2923


namespace find_m_l2_2960

theorem find_m (m : ℝ) : (∀ x > 0, x^2 - 2 * (m^2 + m + 1) * Real.log x ≥ 1) ↔ (m = 0 ∨ m = -1) :=
by
  sorry

end find_m_l2_2960


namespace maximum_distance_correct_l2_2307

noncomputable def maximum_distance 
  (m : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (intersection : (x y : ℝ) → (x + m * y = 0) ∧ (m * x - y - 2 * m + 4 = 0) → P = (x, y)) 
  (distance : (x y : ℝ) → (x - 1) * Real.cos θ + (y - 2) * Real.sin θ = 3) : 
  ℝ :=
3 + Real.sqrt 5

theorem maximum_distance_correct 
  (m : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (intersection : (x y : ℝ) → (x + m * y = 0) ∧ (m * x - y - 2 * m + 4 = 0) → P = (x, y)) 
  (distance : (x y : ℝ) → (x - 1) * Real.cos θ + (y - 2) * Real.sin θ = 3) : 
  maximum_distance m θ P intersection distance = 3 + Real.sqrt 5 := 
sorry

end maximum_distance_correct_l2_2307


namespace min_distance_curves_l2_2434

theorem min_distance_curves (P Q : ℝ × ℝ) (h1 : P.2 = (1/3) * Real.exp P.1) (h2 : Q.2 = Real.log (3 * Q.1)) :
  ∃ d : ℝ, d = Real.sqrt 2 * (Real.log 3 - 1) ∧ d = |P.1 - Q.1| := sorry

end min_distance_curves_l2_2434


namespace find_angle_CBO_l2_2892

theorem find_angle_CBO :
  ∀ (BAO CAO CBO ABO ACO BCO AOC : ℝ), 
  BAO = CAO → 
  CBO = ABO → 
  ACO = BCO → 
  AOC = 110 →
  CBO = 20 :=
by
  intros BAO CAO CBO ABO ACO BCO AOC hBAO_CAOC hCBO_ABO hACO_BCO hAOC
  sorry

end find_angle_CBO_l2_2892


namespace graduation_ceremony_l2_2464

theorem graduation_ceremony (teachers administrators graduates chairs : ℕ) 
  (h1 : teachers = 20) 
  (h2 : administrators = teachers / 2) 
  (h3 : graduates = 50) 
  (h4 : chairs = 180) :
  (chairs - (teachers + administrators + graduates)) / graduates = 2 :=
by 
  sorry

end graduation_ceremony_l2_2464


namespace smallest_integer_in_set_l2_2226

theorem smallest_integer_in_set (median : ℤ) (greatest : ℤ) (h1 : median = 144) (h2 : greatest = 153) : ∃ x : ℤ, x = 135 :=
by
  sorry

end smallest_integer_in_set_l2_2226


namespace prob_both_students_female_l2_2135

-- Define the conditions
def total_students : ℕ := 5
def male_students : ℕ := 2
def female_students : ℕ := 3
def selected_students : ℕ := 2

-- Define the function to compute binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function to select 2 female students
def probability_both_female : ℚ := 
  (binomial female_students selected_students : ℚ) / (binomial total_students selected_students : ℚ)

-- The actual theorem to be proved
theorem prob_both_students_female : probability_both_female = 0.3 := by
  sorry

end prob_both_students_female_l2_2135


namespace sum_of_consecutive_integers_l2_2864

theorem sum_of_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 14) : a + b + c = 39 := 
by 
  sorry

end sum_of_consecutive_integers_l2_2864


namespace exponential_grows_faster_than_polynomial_l2_2449

theorem exponential_grows_faster_than_polynomial (x : ℝ) (h : 0 < x) : 2^x > x^2 := sorry

end exponential_grows_faster_than_polynomial_l2_2449


namespace cost_comparison_l2_2366

def full_ticket_price : ℝ := 240

def cost_agency_A (x : ℕ) : ℝ :=
  full_ticket_price + 0.5 * full_ticket_price * x

def cost_agency_B (x : ℕ) : ℝ :=
  0.6 * full_ticket_price * (x + 1)

theorem cost_comparison (x : ℕ) :
  (x = 4 → cost_agency_A x = cost_agency_B x) ∧
  (x > 4 → cost_agency_A x < cost_agency_B x) ∧
  (x < 4 → cost_agency_A x > cost_agency_B x) :=
by
  sorry

end cost_comparison_l2_2366


namespace smallest_positive_period_2pi_range_of_f_intervals_monotonically_increasing_l2_2012

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin x - (Real.sqrt 3 / 2) * Real.cos x

theorem smallest_positive_period_2pi : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

theorem range_of_f : ∀ y : ℝ, y ∈ Set.range f ↔ -1 ≤ y ∧ y ≤ 1 := by
  sorry

theorem intervals_monotonically_increasing : 
  ∀ k : ℤ, 
  ∀ x : ℝ, 
  (2 * k * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ 2 * k * Real.pi + 5 * Real.pi / 6) → 
  (f (x + Real.pi / 6) - f x) ≥ 0 := by
  sorry

end smallest_positive_period_2pi_range_of_f_intervals_monotonically_increasing_l2_2012


namespace max_value_of_inverse_l2_2483

noncomputable def f (x y z : ℝ) : ℝ := (1/4) * x^2 + 2 * y^2 + 16 * z^2

theorem max_value_of_inverse (x y z a b c : ℝ) (h : a + b + c = 1) (pos_intercepts : a > 0 ∧ b > 0 ∧ c > 0)
  (point_on_plane : (x/a + y/b + z/c = 1)) (pos_points : x > 0 ∧ y > 0 ∧ z > 0) :
  ∀ (k : ℕ), 21 ≤ k → k < (f x y z)⁻¹ :=
sorry

end max_value_of_inverse_l2_2483


namespace percentage_sum_of_v_and_w_l2_2767

variable {x y z v w : ℝ} 

theorem percentage_sum_of_v_and_w (h1 : 0.45 * z = 0.39 * y) (h2 : y = 0.75 * x) 
                                  (h3 : v = 0.80 * z) (h4 : w = 0.60 * y) :
                                  v + w = 0.97 * x :=
by 
  sorry

end percentage_sum_of_v_and_w_l2_2767


namespace balance_difference_l2_2893

def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

theorem balance_difference :
  let angela_balance := compound_interest 12000 0.05 15
  let bob_balance := simple_interest 15000 0.06 15
  let diff := abs (bob_balance - angela_balance)
  (round diff = 3553) :=
by 
  let angela_balance := compound_interest 12000 0.05 15
  let bob_balance := simple_interest 15000 0.06 15
  let diff := abs (bob_balance - angela_balance)
  have h : round diff = 3553 := sorry
  assumption

end balance_difference_l2_2893


namespace find_p_l2_2330

theorem find_p (A B C p q r s : ℝ) (h₀ : A ≠ 0)
  (h₁ : r + s = -B / A)
  (h₂ : r * s = C / A)
  (h₃ : r^3 + s^3 = -p) :
  p = (B^3 - 3 * A * B * C + 2 * A^2 * C^2) / A^3 :=
sorry

end find_p_l2_2330


namespace conic_is_parabola_l2_2151

-- Define the main equation
def main_equation (x y : ℝ) : Prop :=
  y^4 - 6 * x^2 = 3 * y^2 - 2

-- Definition of parabola condition
def is_parabola (x y : ℝ) : Prop :=
  ∃ a b c : ℝ, y^2 = a * x + b ∧ a ≠ 0

-- The theorem statement.
theorem conic_is_parabola :
  ∀ x y : ℝ, main_equation x y → is_parabola x y :=
by
  intros x y h
  sorry

end conic_is_parabola_l2_2151


namespace smaller_square_perimeter_l2_2403

theorem smaller_square_perimeter (s : ℕ) (h1 : 4 * s = 144) : 
  let smaller_s := s / 3 
  let smaller_perimeter := 4 * smaller_s 
  smaller_perimeter = 48 :=
by
  let smaller_s := s / 3
  let smaller_perimeter := 4 * smaller_s 
  sorry

end smaller_square_perimeter_l2_2403


namespace average_payment_l2_2698

-- Each condition from part a) is used as a definition here
variable (n : Nat) (p1 p2 first_payment remaining_payment : Nat)

-- Conditions given in natural language
def payments_every_year : Prop :=
  n = 52 ∧
  first_payment = 410 ∧
  remaining_payment = first_payment + 65 ∧
  p1 = 8 * first_payment ∧
  p2 = 44 * remaining_payment ∧
  p2 = 44 * (first_payment + 65) ∧
  p1 + p2 = 24180

-- The theorem to prove based on the conditions
theorem average_payment 
  (h : payments_every_year n p1 p2 first_payment remaining_payment) 
  : (p1 + p2) / n = 465 := 
sorry  -- Proof is omitted intentionally

end average_payment_l2_2698


namespace scientific_notation_correct_l2_2507

def num_people : ℝ := 2580000
def scientific_notation_form : ℝ := 2.58 * 10^6

theorem scientific_notation_correct : num_people = scientific_notation_form :=
by
  sorry

end scientific_notation_correct_l2_2507


namespace opposite_of_3_is_neg3_l2_2822

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2822


namespace complement_intersect_A_B_range_of_a_l2_2945

-- Definitions for sets A and B
def setA : Set ℝ := {x | -2 < x ∧ x < 0}
def setB : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}

-- First statement to prove
theorem complement_intersect_A_B : (setAᶜ ∩ setB) = {x | x ≥ 0} :=
  sorry

-- Definition for set C
def setC (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 1}

-- Second statement to prove
theorem range_of_a (a : ℝ) : (setC a ⊆ setA) ↔ (a ≤ -1) ∨ (-1 ≤ a ∧ a ≤ -1 / 2) :=
  sorry

end complement_intersect_A_B_range_of_a_l2_2945


namespace opposite_of_three_l2_2859

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l2_2859


namespace find_ratio_l2_2007

open Real

-- Definitions and conditions
variables (b1 b2 : ℝ) (F1 F2 : ℝ × ℝ)
noncomputable def ellipse_eq (Q : ℝ × ℝ) : Prop := (Q.1^2 / 49) + (Q.2^2 / b1^2) = 1
noncomputable def hyperbola_eq (Q : ℝ × ℝ) : Prop := (Q.1^2 / 16) - (Q.2^2 / b2^2) = 1
noncomputable def same_foci (Q : ℝ × ℝ) : Prop := true  -- Placeholder: Representing that both shapes have the same foci F1 and F2

-- The main theorem
theorem find_ratio (Q : ℝ × ℝ) (h1 : ellipse_eq b1 Q) (h2 : hyperbola_eq b2 Q) (h3 : same_foci Q) : 
  abs ((dist Q F1) - (dist Q F2)) / ((dist Q F1) + (dist Q F2)) = 4 / 7 := 
sorry

end find_ratio_l2_2007


namespace initial_paintings_l2_2510

theorem initial_paintings (paintings_per_day : ℕ) (days : ℕ) (total_paintings : ℕ) (initial_paintings : ℕ) 
  (h1 : paintings_per_day = 2) 
  (h2 : days = 30) 
  (h3 : total_paintings = 80) 
  (h4 : total_paintings = initial_paintings + paintings_per_day * days) : 
  initial_paintings = 20 := by
  sorry

end initial_paintings_l2_2510


namespace work_together_days_l2_2395

theorem work_together_days (A_rate B_rate x total_work B_days_worked : ℚ)
  (hA : A_rate = 1/4)
  (hB : B_rate = 1/8)
  (hCombined : (A_rate + B_rate) * x + B_rate * B_days_worked = total_work)
  (hTotalWork : total_work = 1)
  (hBDays : B_days_worked = 2) : x = 2 :=
by
  sorry

end work_together_days_l2_2395


namespace find_k_l2_2790

variables {r k : ℝ}
variables {O A B C D : EuclideanSpace ℝ (Fin 3)}

-- Points A, B, C, and D lie on a sphere centered at O with radius r
variables (hA : dist O A = r) (hB : dist O B = r) (hC : dist O C = r) (hD : dist O D = r)
-- The given vector equation
variables (h_eq : 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = (0 : EuclideanSpace ℝ (Fin 3)))

theorem find_k (hA : dist O A = r) (hB : dist O B = r) (hC : dist O C = r) (hD : dist O D = r)
(h_eq : 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = (0 : EuclideanSpace ℝ (Fin 3))) : 
k = -7 :=
sorry

end find_k_l2_2790


namespace chord_length_l2_2058

/-- Given two concentric circles with radii R and r, where the area of the annulus between them is 16π,
    a chord of the larger circle that is tangent to the smaller circle has a length of 8. -/
theorem chord_length {R r c : ℝ} 
  (h1 : R^2 - r^2 = 16)
  (h2 : (c / 2)^2 + r^2 = R^2) :
  c = 8 :=
by
  sorry

end chord_length_l2_2058


namespace evaluate_expression_l2_2288

theorem evaluate_expression :
  (42 / (9 - 3 * 2)) * 4 = 56 :=
by
  sorry

end evaluate_expression_l2_2288


namespace watch_correct_time_l2_2574

-- Conditions
def initial_time_slow : ℕ := 4 -- minutes slow at 8:00 AM
def final_time_fast : ℕ := 6 -- minutes fast at 4:00 PM
def total_time_interval : ℕ := 480 -- total time interval in minutes from 8:00 AM to 4:00 PM
def rate_of_time_gain : ℚ := (initial_time_slow + final_time_fast) / total_time_interval

-- Statement to prove
theorem watch_correct_time : 
  ∃ t : ℕ, t = 11 * 60 + 12 ∧ 
  ((8 * 60 + t) * rate_of_time_gain = 4) := 
sorry

end watch_correct_time_l2_2574


namespace range_of_a_l2_2489

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * a * x + a + 2 ≤ 0 → 1 ≤ x ∧ x ≤ 4) ↔ a ∈ Set.Ioo (-1 : ℝ) (18 / 7) ∨ a = 18 / 7 := 
by
  sorry

end range_of_a_l2_2489


namespace find_number_l2_2270

theorem find_number (number : ℤ) (h : number + 7 = 6) : number = -1 :=
by
  sorry

end find_number_l2_2270


namespace find_ad_l2_2189

-- Defining the two-digit and three-digit numbers
def two_digit (a b : ℕ) : ℕ := 10 * a + b
def three_digit (a b : ℕ) : ℕ := 100 + two_digit a b

def two_digit' (c d : ℕ) : ℕ := 10 * c + d
def three_digit' (c d : ℕ) : ℕ := 100 * c + 10 * d + 1

-- The main problem
theorem find_ad (a b c d : ℕ) (h1 : three_digit a b = three_digit' c d + 15) (h2 : two_digit a b = two_digit' c d + 24) :
    two_digit a d = 32 := by
  sorry

end find_ad_l2_2189


namespace false_statement_l2_2286

noncomputable def heartsuit (x y : ℝ) := abs (x - y)
noncomputable def diamondsuit (z w : ℝ) := (z + w) ^ 2

theorem false_statement : ∃ (x y : ℝ), (heartsuit x y) ^ 2 ≠ diamondsuit x y := by
  sorry

end false_statement_l2_2286


namespace minimum_value_expression_l2_2351

theorem minimum_value_expression (p q r s t u v w : ℝ) (h1 : p > 0) (h2 : q > 0) 
    (h3 : r > 0) (h4 : s > 0) (h5 : t > 0) (h6 : u > 0) (h7 : v > 0) (h8 : w > 0)
    (hpqrs : p * q * r * s = 16) (htuvw : t * u * v * w = 25) 
    (hptqu : p * t = q * u ∧ q * u = r * v ∧ r * v = s * w) : 
    (p * t) ^ 2 + (q * u) ^ 2 + (r * v) ^ 2 + (s * w) ^ 2 = 80 := sorry

end minimum_value_expression_l2_2351


namespace range_of_x_inequality_l2_2044

theorem range_of_x_inequality (x : ℝ) (h : |2 * x - 1| + x + 3 ≤ 5) : -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end range_of_x_inequality_l2_2044


namespace ryan_hours_difference_l2_2930

theorem ryan_hours_difference :
  let hours_english := 6
  let hours_chinese := 7
  hours_chinese - hours_english = 1 := 
by
  -- this is where the proof steps would go
  sorry

end ryan_hours_difference_l2_2930


namespace lucy_age_l2_2987

theorem lucy_age (Inez_age : ℕ) (Zack_age : ℕ) (Jose_age : ℕ) (Lucy_age : ℕ) 
  (h1 : Inez_age = 18) 
  (h2 : Zack_age = Inez_age + 4) 
  (h3 : Jose_age = Zack_age - 6) 
  (h4 : Lucy_age = Jose_age + 2) : 
  Lucy_age = 18 := by
sorry

end lucy_age_l2_2987


namespace nat_square_iff_divisibility_l2_2519

theorem nat_square_iff_divisibility (A : ℕ) :
  (∃ k : ℕ, A = k^2) ↔ (∀ n : ℕ, ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ n ∣ ((A + i) * (A + i) - A)) :=
sorry

end nat_square_iff_divisibility_l2_2519


namespace gcd_of_ratio_and_lcm_l2_2770

theorem gcd_of_ratio_and_lcm (A B : ℕ) (k : ℕ) (hA : A = 5 * k) (hB : B = 6 * k) (hlcm : Nat.lcm A B = 180) : Nat.gcd A B = 6 :=
by
  sorry

end gcd_of_ratio_and_lcm_l2_2770


namespace maximum_value_of_function_l2_2810

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - 2 * Real.sin x - 2

theorem maximum_value_of_function :
  ∃ x : ℝ, f x = 1 ∧ ∀ y : ℝ, -1 ≤ Real.sin y ∧ Real.sin y ≤ 1 → f y ≤ 1 :=
by
  sorry

end maximum_value_of_function_l2_2810


namespace not_possible_d_count_l2_2671

open Real

theorem not_possible_d_count (t s d : ℝ) (h1 : 3 * t - 4 * s = 1989) (h2 : t - s = d) (h3 : 4 * s > 0) :
  ∃ k : ℕ, k = 663 ∧ ∀ n : ℕ, 1 ≤ n ∧ n ≤ k → d ≠ n :=
by
  sorry

end not_possible_d_count_l2_2671


namespace complex_expression_l2_2736

-- The condition: n is a positive integer
variable (n : ℕ) (hn : 0 < n)

-- Definition of the problem to be proved
theorem complex_expression (n : ℕ) (hn : 0 < n) : 
  (Complex.I ^ (4 * n) + Complex.I ^ (4 * n + 1) + Complex.I ^ (4 * n + 2) + Complex.I ^ (4 * n + 3)) = 0 :=
sorry

end complex_expression_l2_2736


namespace correct_answer_of_john_l2_2643

theorem correct_answer_of_john (x : ℝ) (h : 5 * x + 4 = 104) : (x + 5) / 4 = 6.25 :=
by
  sorry

end correct_answer_of_john_l2_2643


namespace largest_class_students_l2_2466

theorem largest_class_students :
  ∃ x : ℕ, (x + (x - 4) + (x - 8) + (x - 12) + (x - 16) + (x - 20) + (x - 24) +
  (x - 28) + (x - 32) + (x - 36) = 100) ∧ x = 28 :=
by
  sorry

end largest_class_students_l2_2466


namespace sum_a_b_eq_negative_one_l2_2630

theorem sum_a_b_eq_negative_one 
  (a b : ℝ) 
  (h1 : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - a * x - b < 0)
  (h2 : ∀ x : ℝ, x^2 - a * x - b = 0 → x = 2 ∨ x = 3) :
  a + b = -1 := 
sorry

end sum_a_b_eq_negative_one_l2_2630


namespace opposite_of_three_l2_2816

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l2_2816


namespace find_common_tangent_sum_constant_l2_2792

theorem find_common_tangent_sum_constant :
  ∃ (a b c : ℕ), (∀ x y : ℚ, y = x^2 + 169/100 → x = y^2 + 49/4 → a * x + b * y = c) ∧
  (Int.gcd (Int.gcd a b) c = 1) ∧
  (a + b + c = 52) :=
sorry

end find_common_tangent_sum_constant_l2_2792


namespace second_bus_percentage_full_l2_2866

noncomputable def bus_capacity : ℕ := 150
noncomputable def employees_in_buses : ℕ := 195
noncomputable def first_bus_percentage : ℚ := 0.60

theorem second_bus_percentage_full :
  let employees_first_bus := first_bus_percentage * bus_capacity
  let employees_second_bus := (employees_in_buses : ℚ) - employees_first_bus
  let second_bus_percentage := (employees_second_bus / bus_capacity) * 100
  second_bus_percentage = 70 :=
by
  sorry

end second_bus_percentage_full_l2_2866


namespace cube_sum_equal_one_l2_2055

theorem cube_sum_equal_one (x y z : ℝ) (h1 : x + y + z = 3) (h2 : xy + xz + yz = 1) (h3 : xyz = 1) :
  x^3 + y^3 + z^3 = 1 := 
sorry

end cube_sum_equal_one_l2_2055


namespace equivalent_annual_rate_l2_2738

theorem equivalent_annual_rate :
  ∀ (annual_rate compounding_periods: ℝ), annual_rate = 0.08 → compounding_periods = 4 → 
  ((1 + (annual_rate / compounding_periods)) ^ compounding_periods - 1) * 100 = 8.24 :=
by
  intros annual_rate compounding_periods h_rate h_periods
  sorry

end equivalent_annual_rate_l2_2738


namespace total_students_at_gathering_l2_2154

theorem total_students_at_gathering (x : ℕ) 
  (h1 : ∃ x : ℕ, 0 < x)
  (h2 : (x + 6) / (2 * x + 6) = 2 / 3) : 
  (2 * x + 6) = 18 := 
  sorry

end total_students_at_gathering_l2_2154


namespace increasing_function_proof_l2_2364

theorem increasing_function_proof {f : ℝ → ℝ} (h1 : ∀ x, 0 < x → f x > 0) (h2 : ∀ x, 0 < x → deriv f x > 0) :
  ∀ x, 0 < x → deriv (λ x, x * f x) x > 0 :=
by
  intro x hx
  rw [deriv_mul, deriv_id, one_mul]
  exact add_pos (h1 x hx) (mul_pos hx (h2 x hx))

end increasing_function_proof_l2_2364


namespace NoahClosetsFit_l2_2989

-- Declare the conditions as Lean variables and proofs
variable (AliClosetCapacity : ℕ) (NoahClosetsRatio : ℕ) (NoahClosetsCount : ℕ)
variable (H1 : AliClosetCapacity = 200)
variable (H2 : NoahClosetsRatio = 1 / 4)
variable (H3 : NoahClosetsCount = 2)

-- Define the total number of jeans both of Noah's closets can fit
noncomputable def NoahTotalJeans : ℕ := (AliClosetCapacity * NoahClosetsRatio) * NoahClosetsCount

-- Theorem to prove
theorem NoahClosetsFit (AliClosetCapacity : ℕ) (NoahClosetsRatio : ℕ) (NoahClosetsCount : ℕ)
  (H1 : AliClosetCapacity = 200) 
  (H2 : NoahClosetsRatio = 1 / 4) 
  (H3 : NoahClosetsCount = 2) 
  : NoahTotalJeans AliClosetCapacity NoahClosetsRatio NoahClosetsCount = 100 := 
  by 
    sorry

end NoahClosetsFit_l2_2989


namespace simplify_expression_l2_2907

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l2_2907


namespace scientific_notation_conversion_l2_2497

theorem scientific_notation_conversion (x : ℝ) (h_population : x = 141260000) :
  x = 1.4126 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l2_2497


namespace restaurant_total_glasses_l2_2877

theorem restaurant_total_glasses (x y t : ℕ) 
  (h1 : y = x + 16)
  (h2 : (12 * x + 16 * y) / (x + y) = 15)
  (h3 : t = 12 * x + 16 * y) : 
  t = 480 :=
by 
  -- Proof omitted
  sorry

end restaurant_total_glasses_l2_2877


namespace scientific_notation_141260_million_l2_2492

theorem scientific_notation_141260_million :
  ∃ (a : ℝ) (n : ℤ), 141260 * 10^6 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4126 ∧ n = 5 :=
by
  sorry

end scientific_notation_141260_million_l2_2492


namespace expand_and_simplify_l2_2232

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l2_2232


namespace sum_x_coordinates_common_points_l2_2046

-- Definition of the equivalence relation modulo 9
def equiv_mod (a b n : ℤ) : Prop := ∃ k : ℤ, a = b + n * k

-- Definitions of the given conditions
def graph1 (x y : ℤ) : Prop := equiv_mod y (3 * x + 6) 9
def graph2 (x y : ℤ) : Prop := equiv_mod y (7 * x + 3) 9

-- Definition of when two graphs intersect
def points_in_common (x y : ℤ) : Prop := graph1 x y ∧ graph2 x y

-- Proof that the sum of the x-coordinates of the points in common is 3
theorem sum_x_coordinates_common_points : 
  ∃ x y, points_in_common x y ∧ (x = 3) := 
sorry

end sum_x_coordinates_common_points_l2_2046


namespace A_inter_B_empty_iff_l2_2983

variable (m : ℝ)

def A : Set ℝ := {x | x^2 - 3 * x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem A_inter_B_empty_iff : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by
  sorry

end A_inter_B_empty_iff_l2_2983


namespace matrix_problem_l2_2485

def A : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![20 / 3, 4 / 3],
  ![-8 / 3, 8 / 3]
]
def B : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![0, 0], -- Correct values for B can be computed from conditions if needed
  ![0, 0]
]

theorem matrix_problem (A B : Matrix (Fin 2) (Fin 2) ℚ)
  (h1 : A + B = A * B)
  (h2 : A * B = ![
  ![20 / 3, 4 / 3],
  ![-8 / 3, 8 / 3]
]) :
  B * A = ![
    ![20 / 3, 4 / 3],
    ![-8 / 3, 8 / 3]
  ] :=
sorry

end matrix_problem_l2_2485


namespace factorial_division_result_l2_2872

-- Define the inputs and expected output
def n : ℕ := 9
def k : ℕ := 3

-- Condition of factorial
def factorial (m : ℕ) : ℕ := Nat.factorial m

theorem factorial_division_result : (factorial n) / (factorial (n - k)) = 504 :=
by
  sorry

end factorial_division_result_l2_2872


namespace probability_both_blue_l2_2474

-- Conditions defined as assumptions
def jarC_red := 6
def jarC_blue := 10
def total_buttons_in_C := jarC_red + jarC_blue

def after_transfer_buttons_in_C := (3 / 4) * total_buttons_in_C

-- Carla removes the same number of red and blue buttons
-- and after transfer, 12 buttons remain in Jar C
def removed_buttons := total_buttons_in_C - after_transfer_buttons_in_C
def removed_red_buttons := removed_buttons / 2
def removed_blue_buttons := removed_buttons / 2

def remaining_red_in_C := jarC_red - removed_red_buttons
def remaining_blue_in_C := jarC_blue - removed_blue_buttons
def remaining_buttons_in_C := remaining_red_in_C + remaining_blue_in_C

def total_buttons_in_D := removed_buttons
def transferred_blue_buttons := removed_blue_buttons

-- Probability calculations
def probability_blue_in_C := remaining_blue_in_C / remaining_buttons_in_C
def probability_blue_in_D := transferred_blue_buttons / total_buttons_in_D

-- Proof
theorem probability_both_blue :
  (probability_blue_in_C * probability_blue_in_D) = (1 / 3) := 
by
  -- sorry is used here to skip the actual proof
  sorry

end probability_both_blue_l2_2474


namespace construction_paper_initial_count_l2_2965

theorem construction_paper_initial_count 
    (b r d : ℕ)
    (ratio_cond : b = 2 * r)
    (daily_usage : ∀ n : ℕ, n ≤ d → n * 1 = b ∧ n * 3 = r)
    (last_day_cond : 0 = b ∧ 15 = r):
    b + r = 135 :=
sorry

end construction_paper_initial_count_l2_2965


namespace f_1987_eq_5_l2_2958

noncomputable def f : ℕ → ℝ := sorry

axiom f_def : ∀ x : ℕ, x ≥ 0 → ∃ y : ℝ, f x = y
axiom f_one : f 1 = 2
axiom functional_eq : ∀ a b : ℕ, a ≥ 0 → b ≥ 0 → f (a + b) = f a + f b - 3 * f (a * b) + 1

theorem f_1987_eq_5 : f 1987 = 5 := sorry

end f_1987_eq_5_l2_2958


namespace probability_all_yellow_l2_2033

-- Definitions and conditions
def total_apples : ℕ := 8
def red_apples : ℕ := 5
def yellow_apples : ℕ := 3
def chosen_apples : ℕ := 3

-- Theorem to prove
theorem probability_all_yellow :
  (yellow_apples.choose chosen_apples : ℚ) / (total_apples.choose chosen_apples) = 1 / 56 := sorry

end probability_all_yellow_l2_2033


namespace cindy_correct_method_l2_2585

theorem cindy_correct_method (x : ℝ) (h : (x - 7) / 5 = 15) : (x - 5) / 7 = 11 := 
by
  sorry

end cindy_correct_method_l2_2585


namespace probability_at_least_one_odd_l2_2716

def rolls : List (ℕ) := [1, 2, 3, 4, 5, 6]

def fairDie (n : ℕ) : Prop := n ∈ rolls

def evenOutcome : Set ℕ := {n : ℕ | n % 2 = 0}
def oddOutcome : Set ℕ := {n : ℕ | n % 2 = 1}

noncomputable def probability_of_even : ℚ := 1 / 2
noncomputable def probability_of_all_even : ℚ := (1 / 2) ^ 8

theorem probability_at_least_one_odd :
  let p_even := probability_of_even in let p_all_even := probability_of_all_even in
  1 - p_all_even = 255 / 256 := by
  sorry

end probability_at_least_one_odd_l2_2716


namespace volume_of_sphere_l2_2636

theorem volume_of_sphere
  (r : ℝ) (V : ℝ)
  (h₁ : r = 1/3)
  (h₂ : 2 * r = (16/9 * V)^(1/3)) :
  V = 1/6 :=
  sorry

end volume_of_sphere_l2_2636


namespace cosQ_is_0_point_4_QP_is_12_prove_QR_30_l2_2665

noncomputable def find_QR (Q : Real) (QP : Real) : Real :=
  let cosQ := 0.4
  let QR := QP / cosQ
  QR

theorem cosQ_is_0_point_4_QP_is_12_prove_QR_30 :
  find_QR 0.4 12 = 30 :=
by
  sorry

end cosQ_is_0_point_4_QP_is_12_prove_QR_30_l2_2665


namespace parts_of_cut_square_l2_2714

theorem parts_of_cut_square (folds_to_one_by_one : ℕ) : folds_to_one_by_one = 9 :=
  sorry

end parts_of_cut_square_l2_2714


namespace block_path_length_l2_2925

theorem block_path_length
  (length width height : ℝ) 
  (dot_distance : ℝ) 
  (rolls_to_return : ℕ) 
  (π : ℝ) 
  (k : ℝ)
  (H1 : length = 2) 
  (H2 : width = 1) 
  (H3 : height = 1)
  (H4 : dot_distance = 1)
  (H5 : rolls_to_return = 2) 
  (H6 : k = 4) 
  : (2 * rolls_to_return * length * π = k * π) :=
by sorry

end block_path_length_l2_2925


namespace supplement_of_complement_of_35_degree_angle_l2_2089

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 := 
by sorry

end supplement_of_complement_of_35_degree_angle_l2_2089


namespace granger_total_payment_proof_l2_2322

-- Conditions
def cost_per_can_spam := 3
def cost_per_jar_peanut_butter := 5
def cost_per_loaf_bread := 2
def quantity_spam := 12
def quantity_peanut_butter := 3
def quantity_bread := 4

-- Calculation
def total_cost_spam := quantity_spam * cost_per_can_spam
def total_cost_peanut_butter := quantity_peanut_butter * cost_per_jar_peanut_butter
def total_cost_bread := quantity_bread * cost_per_loaf_bread

-- Total amount paid
def total_amount_paid := total_cost_spam + total_cost_peanut_butter + total_cost_bread

-- Theorem to be proven
theorem granger_total_payment_proof : total_amount_paid = 59 :=
by
  sorry

end granger_total_payment_proof_l2_2322


namespace n_value_l2_2254

theorem n_value (n : ℕ) (h1 : ∃ a b : ℕ, a = (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7 ∧ b = 2 * n ∧ a ^ 2 - b ^ 2 = 0) : n = 10 := 
  by sorry

end n_value_l2_2254


namespace drone_height_l2_2592

theorem drone_height (r s h : ℝ) 
  (h_distance_RS : r^2 + s^2 = 160^2)
  (h_DR : h^2 + r^2 = 170^2) 
  (h_DS : h^2 + s^2 = 150^2) : 
  h = 30 * Real.sqrt 43 :=
by 
  sorry

end drone_height_l2_2592


namespace find_a_l2_2369

noncomputable def quadratic_inequality_solution (a b : ℝ) : Prop :=
  a * ((-1/2) * (1/3)) * 20 = 20 ∧
  a < 0 ∧
  (-b / (2 * a)) = (-1 / 2 + 1 / 3)

theorem find_a (a b : ℝ) (h : quadratic_inequality_solution a b) : a = -12 :=
  sorry

end find_a_l2_2369


namespace fraction_torn_off_l2_2401

theorem fraction_torn_off (P: ℝ) (A_remaining: ℝ) (fraction: ℝ):
  P = 32 → 
  A_remaining = 48 → 
  fraction = 1 / 4 :=
by 
  sorry

end fraction_torn_off_l2_2401


namespace expand_and_simplify_l2_2234

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l2_2234


namespace part1_solution_set_part2_min_value_l2_2013

-- Part 1
noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |3 * x|

theorem part1_solution_set :
  {x : ℝ | f x ≥ 3 * |x| + 1} = {x : ℝ | x ≥ -1/2} ∪ {x : ℝ | x ≤ -3/2} :=
by
  sorry

-- Part 2
noncomputable def f_min (x a b : ℝ) : ℝ := 2 * |x + a| + |3 * x - b|

theorem part2_min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∃ x, f_min x a b = 2) :
  3 * a + b = 3 :=
by
  sorry

end part1_solution_set_part2_min_value_l2_2013


namespace paint_time_for_two_people_l2_2660

/-- 
Proof Problem Statement: Prove that it would take 12 hours for two people to paint the house
given that six people can paint it in 4 hours, assuming everyone works at the same rate.
--/
theorem paint_time_for_two_people 
  (h1 : 6 * 4 = 24) 
  (h2 : ∀ (n : ℕ) (t : ℕ), n * t = 24 → t = 24 / n) : 
  2 * 12 = 24 :=
sorry

end paint_time_for_two_people_l2_2660


namespace number_of_rows_l2_2779

theorem number_of_rows (n : ℕ) (h : ∑ i in finset.range n, 53 - 2 * i = 405) : n = 9 :=
sorry

end number_of_rows_l2_2779


namespace solve_system_of_equations_l2_2525

theorem solve_system_of_equations :
  ∃ (x y : ℤ), (x - y = 2) ∧ (2 * x + y = 7) ∧ (x = 3) ∧ (y = 1) :=
by
  sorry

end solve_system_of_equations_l2_2525


namespace sin_C_l2_2780

variable {A B C : ℝ}

theorem sin_C (hA : A = 90) (hcosB : Real.cos B = 3/5) : Real.sin (90 - B) = 3/5 :=
by
  sorry

end sin_C_l2_2780


namespace arithmetic_arrangement_result_l2_2141

theorem arithmetic_arrangement_result :
    (1 / 8) * (1 / 9) * (1 / 28) = 1 / 2016 ∨ ((1 / 8) - (1 / 9)) * (1 / 28) = 1 / 2016 :=
by {
    sorry
}

end arithmetic_arrangement_result_l2_2141


namespace three_digit_no_5_no_8_l2_2454

theorem three_digit_no_5_no_8 : 
  let valid_digits := {0, 1, 2, 3, 4, 6, 7, 9}
  let valid_hundreds := valid_digits \ {0}
  (set.card valid_hundreds) * (set.card valid_digits) * (set.card valid_digits) = 448 :=
by
  let valid_digits := {0, 1, 2, 3, 4, 6, 7, 9}
  let valid_hundreds := valid_digits \ {0}
  have h1 : set.card valid_digits = 8 := by norm_num
  have h2 : set.card valid_hundreds = 7 := by norm_num
  suffices h : (7 : ℕ) * 8 * 8 = 448 by exact h
  norm_num

end three_digit_no_5_no_8_l2_2454


namespace solution_set_l2_2616

theorem solution_set (x y : ℝ) : 
  x^5 - 10 * x^3 * y^2 + 5 * x * y^4 = 0 ↔ 
  x = 0 
  ∨ y = x / Real.sqrt (5 + 2 * Real.sqrt 5) 
  ∨ y = x / Real.sqrt (5 - 2 * Real.sqrt 5) 
  ∨ y = -x / Real.sqrt (5 + 2 * Real.sqrt 5) 
  ∨ y = -x / Real.sqrt (5 - 2 * Real.sqrt 5) := 
by
  sorry

end solution_set_l2_2616


namespace total_commission_l2_2582

-- Define the commission rate
def commission_rate : ℝ := 0.02

-- Define the sale prices of the three houses
def sale_price1 : ℝ := 157000
def sale_price2 : ℝ := 499000
def sale_price3 : ℝ := 125000

-- Total commission calculation
theorem total_commission :
  (commission_rate * sale_price1 + commission_rate * sale_price2 + commission_rate * sale_price3) = 15620 := 
by
  sorry

end total_commission_l2_2582


namespace minimum_sum_sequence_l2_2978

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 49

noncomputable def S_n (n : ℕ) : ℤ := (n * (a_n 1 + a_n n)) / 2

theorem minimum_sum_sequence : ∃ n : ℕ, S_n n = (n - 24) * (n - 24) - 24 * 24 ∧ (∀ m : ℕ, S_n m ≥ S_n n) ∧ n = 24 := 
by {
  sorry -- Proof omitted
}

end minimum_sum_sequence_l2_2978


namespace supplement_of_complement_of_35_degree_angle_l2_2087

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 := 
by sorry

end supplement_of_complement_of_35_degree_angle_l2_2087


namespace range_of_omega_l2_2759

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f ω x = 0 → 
      (∃ x₁ x₂, x₁ ≠ x₂ ∧ 0 ≤ x₁ ∧ x₁ ≤ Real.pi / 2 ∧ 
        0 ≤ x₂ ∧ x₂ ≤ Real.pi / 2 ∧ f ω x₁ = 0 ∧ f ω x₂ = 0)) ↔ 2 ≤ ω ∧ ω < 4 :=
sorry

end range_of_omega_l2_2759


namespace medium_sized_fir_trees_count_l2_2540

theorem medium_sized_fir_trees_count 
  (total_trees : ℕ) (ancient_oaks : ℕ) (saplings : ℕ)
  (h1 : total_trees = 96)
  (h2 : ancient_oaks = 15)
  (h3 : saplings = 58) :
  total_trees - ancient_oaks - saplings = 23 :=
by 
  sorry

end medium_sized_fir_trees_count_l2_2540


namespace find_common_students_l2_2463

theorem find_common_students
  (total_english : ℕ)
  (total_math : ℕ)
  (difference_only_english_math : ℕ)
  (both_english_math : ℕ) :
  total_english = both_english_math + (both_english_math + 10) →
  total_math = both_english_math + both_english_math →
  difference_only_english_math = 10 →
  total_english = 30 →
  total_math = 20 →
  both_english_math = 10 :=
by
  intros
  sorry

end find_common_students_l2_2463


namespace ratio_proof_l2_2019

variable (a b c d : ℚ)

theorem ratio_proof 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 7) :
  d / a = 2 / 35 := by
  sorry

end ratio_proof_l2_2019


namespace min_marked_cells_l2_2551

-- Define the dimensions of the grid
def grid_width : ℕ := 50
def grid_height : ℕ := 50
def strip_width : ℕ := 6

-- Define the total number of strips
def total_strips : ℕ := (grid_width * (grid_height / strip_width)) + (grid_height * (grid_width / strip_width))

-- Statement of the theorem
theorem min_marked_cells : total_strips = 416 :=
by
  Sorry -- Proof goes here 

end min_marked_cells_l2_2551


namespace weeks_in_semester_l2_2408

-- Define the conditions and the question as a hypothesis
def annie_club_hours : Nat := 13

theorem weeks_in_semester (w : Nat) (h : 13 * (w - 2) = 52) : w = 6 := by
  sorry

end weeks_in_semester_l2_2408


namespace polygon_interior_angles_540_implies_5_sides_l2_2537

theorem polygon_interior_angles_540_implies_5_sides (n : ℕ) :
  (n - 2) * 180 = 540 → n = 5 :=
by
  sorry

end polygon_interior_angles_540_implies_5_sides_l2_2537


namespace expand_simplify_expression_l2_2249

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l2_2249


namespace music_player_and_concert_tickets_l2_2252

theorem music_player_and_concert_tickets (n : ℕ) (h1 : 35 % 5 = 0) (h2 : 35 % n = 0) (h3 : ∀ m : ℕ, m < 35 → (m % 5 ≠ 0 ∨ m % n ≠ 0)) : n = 7 :=
sorry

end music_player_and_concert_tickets_l2_2252


namespace right_triangle_area_l2_2287

/-- Given a right triangle where one leg is 18 cm and the hypotenuse is 30 cm,
    prove that the area of the triangle is 216 square centimeters. -/
theorem right_triangle_area (a b c : ℝ) 
    (ha : a = 18) 
    (hc : c = 30) 
    (h_right : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 216 :=
by
  -- Substitute the values given and solve the area.
  sorry

end right_triangle_area_l2_2287


namespace simplify_expression_l2_2920

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l2_2920


namespace simplify_expression_l2_2918

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l2_2918


namespace zahra_kimmie_money_ratio_l2_2346

theorem zahra_kimmie_money_ratio (KimmieMoney ZahraMoney : ℕ) (hKimmie : KimmieMoney = 450)
  (totalSavings : ℕ) (hSaving : totalSavings = 375)
  (h : KimmieMoney / 2 + ZahraMoney / 2 = totalSavings) :
  ZahraMoney / KimmieMoney = 2 / 3 :=
by
  -- Conditions to be used in the proof, but skipped for now
  sorry

end zahra_kimmie_money_ratio_l2_2346


namespace determine_point_T_l2_2425

noncomputable def point : Type := ℝ × ℝ

def is_square (O P Q R : point) : Prop :=
  O.1 = 0 ∧ O.2 = 0 ∧
  Q.1 = 3 ∧ Q.2 = 3 ∧
  P.1 = 3 ∧ P.2 = 0 ∧
  R.1 = 0 ∧ R.2 = 3

def twice_area_square_eq_area_triangle (O P Q T : point) : Prop :=
  2 * (3 * 3) = abs ((P.1 * Q.2 + Q.1 * T.2 + T.1 * P.2 - P.2 * Q.1 - Q.2 * T.1 - T.2 * P.1) / 2)

theorem determine_point_T (O P Q R T : point) (h1 : is_square O P Q R) : 
  twice_area_square_eq_area_triangle O P Q T ↔ T = (3, 12) :=
sorry

end determine_point_T_l2_2425


namespace journey_total_distance_l2_2389

theorem journey_total_distance (D : ℝ) (h_train : D * (3 / 5) = t) (h_bus : D * (7 / 20) = b) (h_walk : D * (1 - ((3 / 5) + (7 / 20))) = 6.5) : D = 130 :=
by
  sorry

end journey_total_distance_l2_2389


namespace virginia_avg_rainfall_l2_2682

theorem virginia_avg_rainfall:
  let march := 3.79
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let total_rainfall := march + april + may + june + july
  let avg_rainfall := total_rainfall / 5
  avg_rainfall = 4 := by sorry

end virginia_avg_rainfall_l2_2682


namespace quadratic_sum_l2_2535

theorem quadratic_sum (x : ℝ) :
  (∃ a b c : ℝ, 6 * x^2 + 48 * x + 162 = a * (x + b) ^ 2 + c ∧ a + b + c = 76) :=
by
  sorry

end quadratic_sum_l2_2535


namespace ratio_of_population_is_correct_l2_2967

noncomputable def ratio_of_population (M W C : ℝ) : ℝ :=
  (M / (W + C)) * 100

theorem ratio_of_population_is_correct
  (M W C : ℝ) 
  (hW: W = 0.9 * M)
  (hC: C = 0.6 * (M + W)) :
  ratio_of_population M W C = 49.02 := 
by
  sorry

end ratio_of_population_is_correct_l2_2967


namespace tan_seven_pi_over_four_l2_2290

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by sorry

end tan_seven_pi_over_four_l2_2290


namespace exists_natural_m_l2_2646

def n (m : ℕ) : ℕ := (Nat.factors m!).count 2

theorem exists_natural_m :
  ∃ (m : ℕ), m > 1990^(1990) ∧ m = 3^(1990) + n m := sorry

end exists_natural_m_l2_2646


namespace brian_total_commission_l2_2580

theorem brian_total_commission :
  let commission_rate := 0.02
  let house1 := 157000
  let house2 := 499000
  let house3 := 125000
  let total_sales := house1 + house2 + house3
  let total_commission := total_sales * commission_rate
  total_commission = 15620 := by
{
  sorry
}

end brian_total_commission_l2_2580


namespace total_workers_in_workshop_l2_2390

theorem total_workers_in_workshop 
  (W : ℕ)
  (T : ℕ := 5)
  (avg_all : ℕ := 700)
  (avg_technicians : ℕ := 800)
  (avg_rest : ℕ := 650) 
  (total_salary_all : ℕ := W * avg_all)
  (total_salary_technicians : ℕ := T * avg_technicians)
  (total_salary_rest : ℕ := (W - T) * avg_rest) :
  total_salary_all = total_salary_technicians + total_salary_rest →
  W = 15 :=
by
  sorry

end total_workers_in_workshop_l2_2390


namespace range_func_l2_2421

noncomputable def func (x : ℝ) : ℝ := x + 4 / x

theorem range_func (x : ℝ) (hx : x ≠ 0) : func x ≤ -4 ∨ func x ≥ 4 := by
  sorry

end range_func_l2_2421


namespace total_coughs_after_20_minutes_l2_2303

theorem total_coughs_after_20_minutes (georgia_rate robert_rate : ℕ) (coughs_per_minute : ℕ) :
  georgia_rate = 5 →
  robert_rate = 2 * georgia_rate →
  coughs_per_minute = georgia_rate + robert_rate →
  (20 * coughs_per_minute) = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_coughs_after_20_minutes_l2_2303


namespace travel_options_l2_2229

-- Define the conditions
def trains_from_A_to_B := 3
def ferries_from_B_to_C := 2

-- State the proof problem
theorem travel_options (t : ℕ) (f : ℕ) (h1 : t = trains_from_A_to_B) (h2 : f = ferries_from_B_to_C) : t * f = 6 :=
by
  rewrite [h1, h2]
  sorry

end travel_options_l2_2229


namespace athlete_stable_performance_l2_2529

theorem athlete_stable_performance 
  (A_var : ℝ) (B_var : ℝ) (C_var : ℝ) (D_var : ℝ)
  (avg_score : ℝ)
  (hA_var : A_var = 0.019)
  (hB_var : B_var = 0.021)
  (hC_var : C_var = 0.020)
  (hD_var : D_var = 0.022)
  (havg : avg_score = 13.2) :
  A_var < B_var ∧ A_var < C_var ∧ A_var < D_var :=
by {
  sorry
}

end athlete_stable_performance_l2_2529


namespace seventeen_in_base_three_l2_2158

theorem seventeen_in_base_three : (17 : ℕ) = 1 * 3^2 + 2 * 3^1 + 2 * 3^0 :=
by
  -- This is the arithmetic representation of the conversion,
  -- proving that 17 in base 10 equals 122 in base 3
  sorry

end seventeen_in_base_three_l2_2158


namespace sphere_radius_ratio_l2_2717

theorem sphere_radius_ratio (R r : ℝ) (h₁ : (4 / 3) * Real.pi * R ^ 3 = 450 * Real.pi) (h₂ : (4 / 3) * Real.pi * r ^ 3 = 0.25 * 450 * Real.pi) :
  r / R = 1 / 2 :=
sorry

end sphere_radius_ratio_l2_2717


namespace total_coughs_after_20_minutes_l2_2304

def coughs_in_n_minutes (rate_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  rate_per_minute * minutes

def total_coughs (georgia_rate_per_minute : ℕ) (minutes : ℕ) (multiplier : ℕ) : ℕ :=
  let georgia_coughs := coughs_in_n_minutes georgia_rate_per_minute minutes
  let robert_rate_per_minute := georgia_rate_per_minute * multiplier
  let robert_coughs := coughs_in_n_minutes robert_rate_per_minute minutes
  georgia_coughs + robert_coughs

theorem total_coughs_after_20_minutes :
  total_coughs 5 20 2 = 300 :=
by
  sorry

end total_coughs_after_20_minutes_l2_2304


namespace solve_for_y_l2_2359

noncomputable def solve_quadratic := {y : ℂ // 4 + 3 * y^2 = 0.7 * y - 40}

theorem solve_for_y : 
  ∃ y : ℂ, (y = 0.1167 + 3.8273 * Complex.I ∨ y = 0.1167 - 3.8273 * Complex.I) ∧
            (4 + 3 * y^2 = 0.7 * y - 40) :=
by
  sorry

end solve_for_y_l2_2359


namespace total_students_l2_2122

theorem total_students (x : ℝ) :
  (x - (1/2)*x - (1/4)*x - (1/8)*x = 3) → x = 24 :=
by
  intro h
  sorry

end total_students_l2_2122


namespace expression_simplification_l2_2915

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l2_2915


namespace min_n_for_constant_term_l2_2769

theorem min_n_for_constant_term (n : ℕ) (h : n > 0) :
  ∃ (r : ℕ), (2 * n = 5 * r) → n = 5 :=
by
  sorry

end min_n_for_constant_term_l2_2769


namespace shape_is_cone_l2_2164

-- Define the spherical coordinate system and the condition
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

def shape (c : ℝ) (p : SphericalCoord) : Prop := p.φ ≤ c

-- The shape described by \(\exists c, \forall p \in SphericalCoord, shape c p\) is a cone
theorem shape_is_cone (c : ℝ) (p : SphericalCoord) : shape c p → (c ≥ 0 ∧ c ≤ π → shape c p = Cone) :=
by
  sorry

end shape_is_cone_l2_2164


namespace adam_first_year_students_l2_2889

theorem adam_first_year_students (X : ℕ) 
  (remaining_years_students : ℕ := 9 * 50)
  (total_students : ℕ := 490) 
  (total_years_students : X + remaining_years_students = total_students) : X = 40 :=
by { sorry }

end adam_first_year_students_l2_2889


namespace rate_of_fencing_is_4_90_l2_2805

noncomputable def rate_of_fencing_per_meter : ℝ :=
  let area_hectares := 13.86
  let cost := 6466.70
  let area_m2 := area_hectares * 10000
  let radius := Real.sqrt (area_m2 / Real.pi)
  let circumference := 2 * Real.pi * radius
  cost / circumference

theorem rate_of_fencing_is_4_90 :
  rate_of_fencing_per_meter = 4.90 := sorry

end rate_of_fencing_is_4_90_l2_2805


namespace rate_times_base_eq_9000_l2_2374

noncomputable def Rate : ℝ := 0.00015
noncomputable def BaseAmount : ℝ := 60000000

theorem rate_times_base_eq_9000 :
  Rate * BaseAmount = 9000 := 
  sorry

end rate_times_base_eq_9000_l2_2374


namespace odd_function_property_l2_2165

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Lean 4 statement of the problem
theorem odd_function_property (f : ℝ → ℝ) (h : is_odd f) : ∀ x : ℝ, f x + f (-x) = 0 := 
  by sorry

end odd_function_property_l2_2165


namespace correct_number_of_three_digit_numbers_l2_2456

def count_valid_three_digit_numbers : Nat :=
  let hundreds := [1, 2, 3, 4, 6, 7, 9].length
  let tens_units := [0, 1, 2, 3, 4, 6, 7, 9].length
  hundreds * tens_units * tens_units

theorem correct_number_of_three_digit_numbers :
  count_valid_three_digit_numbers = 448 :=
by
  unfold count_valid_three_digit_numbers
  sorry

end correct_number_of_three_digit_numbers_l2_2456


namespace max_contribution_l2_2106

theorem max_contribution (total_contribution : ℝ) (num_people : ℕ) (min_contribution_each : ℝ) (h1 : total_contribution = 45.00) (h2 : num_people = 25) (h3 : min_contribution_each = 1.00) : 
  ∃ max_cont : ℝ, max_cont = 21.00 :=
by
  sorry

end max_contribution_l2_2106


namespace solve_system_of_equations_l2_2522

theorem solve_system_of_equations :
  ∃ x y : ℝ, (x - y = 2) ∧ (2 * x + y = 7) ∧ x = 3 ∧ y = 1 :=
by
  sorry

end solve_system_of_equations_l2_2522


namespace common_factor_l2_2742

theorem common_factor (x y a b : ℤ) : 
  3 * x * (a - b) - 9 * y * (b - a) = 3 * (a - b) * (x + 3 * y) :=
by {
  sorry
}

end common_factor_l2_2742


namespace limit_of_nested_radical_l2_2146

theorem limit_of_nested_radical :
  ∃ F : ℝ, F = 43 ∧ F = Real.sqrt (86 + 41 * F) :=
sorry

end limit_of_nested_radical_l2_2146


namespace distinct_triplet_inequality_l2_2939

theorem distinct_triplet_inequality (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  abs (a / (b - c)) + abs (b / (c - a)) + abs (c / (a - b)) ≥ 2 := 
sorry

end distinct_triplet_inequality_l2_2939


namespace intersection_eq_l2_2946

def setM (x : ℝ) : Prop := x > -1
def setN (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem intersection_eq : {x : ℝ | setM x} ∩ {x | setN x} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end intersection_eq_l2_2946


namespace find_m_l2_2954

variable (m : ℝ)
def vector_a : ℝ × ℝ := (1, 3)
def vector_b : ℝ × ℝ := (m, -2)

theorem find_m (h : (1 + m) + 3 = 0) : m = -4 := by
  sorry

end find_m_l2_2954


namespace simplify_expression_l2_2919

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l2_2919


namespace math_competition_question_1_math_competition_question_2_l2_2666

noncomputable def participant_score_probabilities : Prop :=
  let P1 := (3 / 5)^2 * (2 / 5)^2
  let P2 := 2 * (3 / 5) * (2 / 5)
  let P3 := 2 * (3 / 5) * (2 / 5)^2
  let P4 := (3 / 5)^2
  P1 + P2 + P3 + P4 = 208 / 625

noncomputable def winning_probabilities : Prop :=
  let P_100_or_more := (4 / 5)^8 * (3 / 5)^3 + 3 * (4 / 5)^8 * (3 / 5)^2 * (2 / 5) + 
                      (8 * (4 / 5)^7 * (1/5) * (3 / 5)^3 + 
                      28 * (4 / 5)^6 * (1/5)^2 * (3 / 5)^3)
  let winning_if_100_or_more := P_100_or_more * (9 / 10)
  let winning_if_less_100 := (1 - P_100_or_more) * (2 / 5)
  winning_if_100_or_more + winning_if_less_100 ≥ 1 / 2

theorem math_competition_question_1 : participant_score_probabilities :=
by sorry

theorem math_competition_question_2 : winning_probabilities :=
by sorry

end math_competition_question_1_math_competition_question_2_l2_2666


namespace supplement_of_complement_is_125_l2_2091

-- Definition of the initial angle
def initial_angle : ℝ := 35

-- Definition of the complement of the angle
def complement_angle (θ : ℝ) : ℝ := 90 - θ

-- Definition of the supplement of an angle
def supplement_angle (θ : ℝ) : ℝ := 180 - θ

-- Main theorem statement
theorem supplement_of_complement_is_125 : 
  supplement_angle (complement_angle initial_angle) = 125 := 
by
  sorry

end supplement_of_complement_is_125_l2_2091


namespace integers_between_sqrt7_and_sqrt77_l2_2765

theorem integers_between_sqrt7_and_sqrt77 : 
  2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3 ∧ 8 < Real.sqrt 77 ∧ Real.sqrt 77 < 9 →
  ∃ (n : ℕ), n = 6 ∧ ∀ (k : ℕ), (3 ≤ k ∧ k ≤ 8) ↔ (2 < Real.sqrt 7 ∧ Real.sqrt 77 < 9) :=
by sorry

end integers_between_sqrt7_and_sqrt77_l2_2765


namespace coefficient_of_term_in_expansion_l2_2078

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

noncomputable def power_rat (r : ℚ) (n : ℕ) : ℚ :=
  r ^ n
  
theorem coefficient_of_term_in_expansion :
  ∃ (c : ℚ), 
  (∃ (x y : ℚ), (x = 3 ∧ y = 7) → 
  x^3 * y^7 = c * ((binomial_coefficient 10 3) * 
  (power_rat (2/3) 3) * (power_rat (-3/4) 7))) ∧
  c = -3645/7656 :=
begin
  sorry
end

end coefficient_of_term_in_expansion_l2_2078


namespace trajectory_of_point_P_l2_2176

open Real

theorem trajectory_of_point_P (a : ℝ) (ha : a > 0) :
  (∀ x y : ℝ, (a = 1 → x = 0) ∧ 
    (a ≠ 1 → (x - (a^2 + 1) / (a^2 - 1))^2 + y^2 = 4 * a^2 / (a^2 - 1)^2)) := 
by 
  sorry

end trajectory_of_point_P_l2_2176


namespace determine_teeth_l2_2076

theorem determine_teeth (x V : ℝ) (h1 : V = 63 * x / (x + 10)) (h2 : V = 28 * (x + 10)) :
  x = 20 ∧ (x + 10) = 30 :=
by
  sorry

end determine_teeth_l2_2076


namespace factor_x6_minus_64_l2_2159

theorem factor_x6_minus_64 :
  ∀ x : ℝ, (x^6 - 64) = (x-2) * (x+2) * (x^4 + 4*x^2 + 16) :=
by
  sorry

end factor_x6_minus_64_l2_2159


namespace dealer_decision_is_mode_l2_2559

noncomputable def sales_A := 15
noncomputable def sales_B := 22
noncomputable def sales_C := 18
noncomputable def sales_D := 10

def is_mode (sales: List ℕ) (mode_value: ℕ) : Prop :=
  mode_value ∈ sales ∧ ∀ x ∈ sales, x ≤ mode_value

theorem dealer_decision_is_mode : 
  is_mode [sales_A, sales_B, sales_C, sales_D] sales_B :=
by
  sorry

end dealer_decision_is_mode_l2_2559


namespace fifth_equation_l2_2440

noncomputable def equation_1 : Prop := 2 * 1 = 2
noncomputable def equation_2 : Prop := 2 ^ 2 * 1 * 3 = 3 * 4
noncomputable def equation_3 : Prop := 2 ^ 3 * 1 * 3 * 5 = 4 * 5 * 6

theorem fifth_equation
  (h1 : equation_1)
  (h2 : equation_2)
  (h3 : equation_3) :
  2 ^ 5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 :=
by {
  sorry
}

end fifth_equation_l2_2440


namespace sum_of_squares_and_cubes_l2_2994

theorem sum_of_squares_and_cubes (a b : ℤ) (h : ∃ k : ℤ, a^2 - 4*b = k^2) :
  ∃ x1 x2 : ℤ, a^2 - 2*b = x1^2 + x2^2 ∧ 3*a*b - a^3 = x1^3 + x2^3 :=
by
  sorry

end sum_of_squares_and_cubes_l2_2994


namespace each_persons_final_share_l2_2071

theorem each_persons_final_share
  (total_dining_bill : ℝ)
  (number_of_people : ℕ)
  (tip_percentage : ℝ) :
  total_dining_bill = 211.00 →
  tip_percentage = 0.15 →
  number_of_people = 5 →
  ((total_dining_bill + total_dining_bill * tip_percentage) / number_of_people) = 48.53 :=
by
  intros
  sorry

end each_persons_final_share_l2_2071


namespace simplify_expression_l2_2905

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l2_2905


namespace jerusha_earnings_l2_2974

variable (L : ℝ) 

theorem jerusha_earnings (h1 : L + 4 * L = 85) : 4 * L = 68 := 
by
  sorry

end jerusha_earnings_l2_2974


namespace fettuccine_to_tortellini_ratio_l2_2265

-- Definitions based on the problem conditions
def total_students := 800
def preferred_spaghetti := 320
def preferred_fettuccine := 200
def preferred_tortellini := 160
def preferred_penne := 120

-- Theorem to prove that the ratio is 5/4
theorem fettuccine_to_tortellini_ratio :
  (preferred_fettuccine : ℚ) / (preferred_tortellini : ℚ) = 5 / 4 :=
sorry

end fettuccine_to_tortellini_ratio_l2_2265


namespace smaller_successive_number_l2_2227

theorem smaller_successive_number (n : ℕ) (h : n * (n + 1) = 9506) : n = 97 :=
sorry

end smaller_successive_number_l2_2227


namespace robert_ate_more_chocolates_l2_2514

-- Define the number of chocolates eaten by Robert and Nickel
def robert_chocolates : ℕ := 12
def nickel_chocolates : ℕ := 3

-- State the problem as a theorem to prove
theorem robert_ate_more_chocolates :
  robert_chocolates - nickel_chocolates = 9 :=
by
  sorry

end robert_ate_more_chocolates_l2_2514


namespace frequency_of_scoring_l2_2875

def shots : ℕ := 80
def goals : ℕ := 50
def frequency : ℚ := goals / shots

theorem frequency_of_scoring : frequency = 0.625 := by
  sorry

end frequency_of_scoring_l2_2875


namespace simplify_expression_evaluate_expression_with_values_l2_2358

-- Problem 1: Simplify the expression to -xy
theorem simplify_expression (x y : ℤ) : 
  3 * x^2 + 2 * x * y - 4 * y^2 - 3 * x * y + 4 * y^2 - 3 * x^2 = - x * y :=
  sorry

-- Problem 2: Evaluate the expression with given values
theorem evaluate_expression_with_values (a b : ℤ) (ha : a = 2) (hb : b = -3) :
  a + (5 * a - 3 * b) - 2 * (a - 2 * b) = 5 :=
  sorry

end simplify_expression_evaluate_expression_with_values_l2_2358


namespace theta_in_second_quadrant_l2_2324

theorem theta_in_second_quadrant (θ : ℝ) (h₁ : Real.sin θ > 0) (h₂ : Real.cos θ < 0) : 
  π / 2 < θ ∧ θ < π := 
sorry

end theta_in_second_quadrant_l2_2324


namespace remaining_plants_after_bugs_l2_2410

theorem remaining_plants_after_bugs (initial_plants first_day_eaten second_day_fraction third_day_eaten remaining_plants : ℕ) : 
  initial_plants = 30 →
  first_day_eaten = 20 →
  second_day_fraction = 2 →
  third_day_eaten = 1 →
  remaining_plants = initial_plants - first_day_eaten - (initial_plants - first_day_eaten) / second_day_fraction - third_day_eaten →
  remaining_plants = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end remaining_plants_after_bugs_l2_2410


namespace athlete_stable_performance_l2_2530

theorem athlete_stable_performance 
  (A_var : ℝ) (B_var : ℝ) (C_var : ℝ) (D_var : ℝ)
  (avg_score : ℝ)
  (hA_var : A_var = 0.019)
  (hB_var : B_var = 0.021)
  (hC_var : C_var = 0.020)
  (hD_var : D_var = 0.022)
  (havg : avg_score = 13.2) :
  A_var < B_var ∧ A_var < C_var ∧ A_var < D_var :=
by {
  sorry
}

end athlete_stable_performance_l2_2530


namespace distance_eq_3_implies_points_l2_2353

-- Definition of the distance of point A to the origin
def distance_to_origin (x : ℝ) : ℝ := |x|

-- Theorem statement translating the problem
theorem distance_eq_3_implies_points (x : ℝ) (h : distance_to_origin x = 3) :
  x = 3 ∨ x = -3 :=
sorry

end distance_eq_3_implies_points_l2_2353


namespace find_white_balls_l2_2194

-- Define the number of red balls
def red_balls : ℕ := 4

-- Define the probability of drawing a red ball
def prob_red : ℚ := 1 / 4

-- Define the number of white balls
def white_balls : ℕ := 12

theorem find_white_balls (x : ℕ) (h1 : (red_balls : ℚ) / (red_balls + x) = prob_red) : x = white_balls :=
by
  -- Proof is omitted
  sorry

end find_white_balls_l2_2194


namespace negation_of_existence_l2_2368

theorem negation_of_existence :
  ¬(∃ x : ℝ, x^2 + 2 * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 1 ≥ 0 :=
by
  sorry

end negation_of_existence_l2_2368


namespace factoring_sum_of_coefficients_l2_2224

theorem factoring_sum_of_coefficients 
  (a b c d e f g h j k : ℤ)
  (h1 : 64 * x^6 - 729 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) :
  a + b + c + d + e + f + g + h + j + k = 30 :=
sorry

end factoring_sum_of_coefficients_l2_2224


namespace cost_price_l2_2128

namespace ClothingDiscount

variables (x : ℝ)

def loss_condition (x : ℝ) : ℝ := 0.5 * x + 20
def profit_condition (x : ℝ) : ℝ := 0.8 * x - 40

def marked_price := { x : ℝ // loss_condition x = profit_condition x }

noncomputable def clothing_price : marked_price := 
    ⟨200, sorry⟩

theorem cost_price : loss_condition 200 = 120 :=
sorry

end ClothingDiscount

end cost_price_l2_2128


namespace probability_at_least_one_head_and_die_3_l2_2262

-- Define the probability of an event happening
noncomputable def probability_of_event (total_outcomes : ℕ) (successful_outcomes : ℕ) : ℚ :=
  successful_outcomes / total_outcomes

-- Define the problem specific values
def total_coin_outcomes : ℕ := 4
def successful_coin_outcomes : ℕ := 3
def total_die_outcomes : ℕ := 8
def successful_die_outcome : ℕ := 1
def total_outcomes : ℕ := total_coin_outcomes * total_die_outcomes
def successful_outcomes : ℕ := successful_coin_outcomes * successful_die_outcome

-- Prove that the probability of at least one head in two coin flips and die showing a 3 is 3/32
theorem probability_at_least_one_head_and_die_3 : 
  probability_of_event total_outcomes successful_outcomes = 3 / 32 := by
  sorry

end probability_at_least_one_head_and_die_3_l2_2262


namespace length_of_EF_l2_2593

variables {A B C D E F : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables (triangle_ABC : A) (triangle_DEF : D)
variables {AB DE BC EF : ℝ}

-- Given conditions
variable (sim_ABC_DEF : ∃ (A B C D E F : Type*) 
                          [metric_space A] [metric_space B] [metric_space C] 
                          [metric_space D] [metric_space E] [metric_space F],
                          ∀ (triangle_ABC : A) (triangle_DEF : D),
                          ∃ (AB DE BC EF : ℝ),
                          (triangle_ABC = triangle_DEF)
                        )
variable (ratio_AB_DE : AB / DE = 1 / 2)
variable (BC_length : BC = 2)

-- Prove that EF = 4
theorem length_of_EF (h1 : sim_ABC_DEF) (h2 : ratio_AB_DE) (h3 : BC_length) : EF = 4 :=
sorry

end length_of_EF_l2_2593


namespace circle_equation_l2_2808

theorem circle_equation : 
  ∃ (b : ℝ), (∀ (x y : ℝ), (x^2 + (y - b)^2 = 1 ↔ (x = 1 ∧ y = 2) → b = 2)) :=
sorry

end circle_equation_l2_2808


namespace golden_section_AC_length_l2_2305

namespace GoldenSection

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def AC_length (AB : ℝ) : ℝ :=
  let φ := golden_ratio
  AB / φ

theorem golden_section_AC_length (AB : ℝ) (C_gold : Prop) (hAB : AB = 2) (A_gt_B : AC_length AB > AB - AC_length AB) :
  AC_length AB = Real.sqrt 5 - 1 :=
  sorry

end GoldenSection

end golden_section_AC_length_l2_2305


namespace highlighter_count_l2_2191

-- Define the quantities of highlighters.
def pinkHighlighters := 3
def yellowHighlighters := 7
def blueHighlighters := 5

-- Define the total number of highlighters.
def totalHighlighters := pinkHighlighters + yellowHighlighters + blueHighlighters

-- The theorem states that the total number of highlighters is 15.
theorem highlighter_count : totalHighlighters = 15 := by
  -- Proof skipped for now.
  sorry

end highlighter_count_l2_2191


namespace rate_of_current_l2_2699

variable (c : ℝ)
def effective_speed_downstream (c : ℝ) : ℝ := 4.5 + c
def effective_speed_upstream (c : ℝ) : ℝ := 4.5 - c

theorem rate_of_current
  (h1 : ∀ d : ℝ, d / (4.5 - c) = 2 * (d / (4.5 + c)))
  : c = 1.5 :=
by
  sorry

end rate_of_current_l2_2699


namespace sharks_at_other_beach_is_12_l2_2415

-- Define the conditions
def cape_may_sharks := 32
def sharks_other_beach (S : ℕ) := 2 * S + 8

-- Statement to prove
theorem sharks_at_other_beach_is_12 (S : ℕ) (h : cape_may_sharks = sharks_other_beach S) : S = 12 :=
by
  -- Sorry statement to skip the proof part
  sorry

end sharks_at_other_beach_is_12_l2_2415


namespace simplify_expression_l2_2995

theorem simplify_expression (a : ℤ) (ha : a = -2) : 
  3 * a^2 + (a^2 + (5 * a^2 - 2 * a) - 3 * (a^2 - 3 * a)) = 10 := 
by 
  sorry

end simplify_expression_l2_2995


namespace no_such_b_exists_l2_2426

theorem no_such_b_exists (b : ℝ) (hb : 0 < b) :
  ¬(∃ k : ℝ, 0 < k ∧ ∀ n : ℕ, 0 < n → (n - k ≤ (⌊b * n⌋ : ℤ) ∧ (⌊b * n⌋ : ℤ) < n)) :=
by
  sorry

end no_such_b_exists_l2_2426


namespace triangle_is_right_l2_2462

theorem triangle_is_right (A B C a b c : ℝ) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
    (h₄ : A + B + C = π) (h_eq : a * (Real.cos C) + c * (Real.cos A) = b * (Real.sin B)) : B = π / 2 :=
by
  sorry

end triangle_is_right_l2_2462


namespace machine_value_after_two_years_l2_2066

noncomputable def machine_market_value (initial_value : ℝ) (years : ℕ) (decrease_rate : ℝ) : ℝ :=
  initial_value * (1 - decrease_rate) ^ years

theorem machine_value_after_two_years :
  machine_market_value 8000 2 0.2 = 5120 := by
  sorry

end machine_value_after_two_years_l2_2066


namespace part1_part2_part3_l2_2198

-- Defining the quadratic function
def quadratic (t : ℝ) (x : ℝ) : ℝ := x^2 - 2 * t * x + 3

-- Part (1)
theorem part1 (t : ℝ) (h : quadratic t 2 = 1) : t = 3 / 2 :=
by sorry

-- Part (2)
theorem part2 (t : ℝ) (h : ∀x, 0 ≤ x → x ≤ 3 → (quadratic t x) ≥ -2) : t = Real.sqrt 5 :=
by sorry

-- Part (3)
theorem part3 (m a b : ℝ) (hA : quadratic t (m - 2) = a) (hB : quadratic t 4 = b) 
              (hC : quadratic t m = a) (ha : a < b) (hb : b < 3) (ht : t > 0) : 
              (3 < m ∧ m < 4) ∨ (m > 6) :=
by sorry

end part1_part2_part3_l2_2198


namespace simplify_expression_l2_2917

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l2_2917


namespace election_winner_votes_l2_2865

theorem election_winner_votes :
  ∃ V W : ℝ, (V = (71.42857142857143 / 100) * V + 3000 + 5000) ∧
            (W = (71.42857142857143 / 100) * V) ∧
            W = 20000 := by
  sorry

end election_winner_votes_l2_2865


namespace expand_and_simplify_l2_2243

noncomputable def expanded_expr (a : ℝ) : ℝ :=
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6)

theorem expand_and_simplify (a : ℝ) :
  expanded_expr a = a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 :=
by {
  -- Proof goes here
  sorry
}

end expand_and_simplify_l2_2243


namespace odd_even_shift_composition_l2_2807

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_even_function_shifted (f : ℝ → ℝ) (shift : ℝ) : Prop :=
  ∀ x : ℝ, f (x + shift) = f (-x + shift)

theorem odd_even_shift_composition
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_even_shift : is_even_function_shifted f 3)
  (h_f1 : f 1 = 1) :
  f 6 + f 11 = -1 := by
  sorry

end odd_even_shift_composition_l2_2807


namespace kevin_ends_with_cards_l2_2787

def cards_found : ℝ := 47.0
def cards_lost : ℝ := 7.0

theorem kevin_ends_with_cards : cards_found - cards_lost = 40.0 := by
  sorry

end kevin_ends_with_cards_l2_2787


namespace sin_of_angle_l2_2010

theorem sin_of_angle (α : ℝ) (x y : ℝ) (h1 : x = -3) (h2 : y = -4) (r : ℝ) (hr : r = Real.sqrt (x^2 + y^2)) : 
  Real.sin α = -4 / r := 
by
  -- Definitions
  let y := -4
  let x := -3
  let r := Real.sqrt (x^2 + y^2)
  -- Proof
  sorry

end sin_of_angle_l2_2010


namespace least_positive_integer_x_multiple_53_l2_2870

theorem least_positive_integer_x_multiple_53 :
  ∃ x : ℕ, (x > 0) ∧ ((3 * x + 41)^2) % 53 = 0 ∧ ∀ y : ℕ, (y > 0) ∧ ((3 * y + 41)^2) % 53 = 0 → x ≤ y := 
begin
  use 4,
  split,
  { -- 4 > 0
    exact dec_trivial },
  split,
  { -- (3 * 4 + 41)^2 % 53 = 0
    calc (3 * 4 + 41)^2 % 53 = (53)^2 % 53 : by norm_num
    ... = 0 : by norm_num },
  { -- smallest positive integer solution
    assume y hy,
    cases hy with hy_gt0 hy_multiple,
    by_contradiction hxy,
    have x_val : 4 = 1,
      by linarith,
    norm_num at x_val,
    cases x_val
  }
end

end least_positive_integer_x_multiple_53_l2_2870


namespace determine_n_l2_2484

noncomputable def S : ℕ → ℝ := sorry -- define arithmetic series sum
noncomputable def a_1 : ℝ := sorry -- define first term
noncomputable def d : ℝ := sorry -- define common difference

axiom S_6 : S 6 = 36
axiom S_n {n : ℕ} (h : n > 0) : S n = 324
axiom S_n_minus_6 {n : ℕ} (h : n > 6) : S (n - 6) = 144

theorem determine_n (n : ℕ) (h : n > 0) : n = 18 := by {
  sorry
}

end determine_n_l2_2484


namespace scientific_notation_141260_million_l2_2491

theorem scientific_notation_141260_million :
  ∃ (a : ℝ) (n : ℤ), 141260 * 10^6 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4126 ∧ n = 5 :=
by
  sorry

end scientific_notation_141260_million_l2_2491


namespace ab_product_l2_2024

theorem ab_product (a b : ℝ) (h_sol : ∀ x, -1 < x ∧ x < 4 → x^2 + a * x + b < 0) 
  (h_roots : ∀ x, x^2 + a * x + b = 0 ↔ x = -1 ∨ x = 4) : 
  a * b = 12 :=
sorry

end ab_product_l2_2024


namespace horse_revolutions_l2_2564

noncomputable def carousel_revolutions (r1 r2 d1 : ℝ) : ℝ :=
  (d1 * r1) / r2

theorem horse_revolutions :
  carousel_revolutions 30 10 40 = 120 :=
by
  sorry

end horse_revolutions_l2_2564


namespace shaded_triangle_probability_l2_2969

noncomputable def total_triangles : ℕ := 5
noncomputable def shaded_triangles : ℕ := 2
noncomputable def probability_shaded : ℚ := shaded_triangles / total_triangles

theorem shaded_triangle_probability : probability_shaded = 2 / 5 :=
by
  sorry

end shaded_triangle_probability_l2_2969


namespace solve_for_a_l2_2334

noncomputable def line_slope_parallels (a : ℝ) : Prop :=
  (a^2 - a) = 6

theorem solve_for_a : { a : ℝ // line_slope_parallels a } → (a = -2 ∨ a = 3) := by
  sorry

end solve_for_a_l2_2334


namespace base_number_is_three_l2_2768

theorem base_number_is_three (some_number : ℝ) (y : ℕ) (h1 : 9^y = some_number^14) (h2 : y = 7) : some_number = 3 :=
by { sorry }

end base_number_is_three_l2_2768


namespace prob_A1_selected_prob_neither_B1_C1_selected_l2_2539

-- Conditions
def volunteers : List (String × String × String) :=
  [("A1", "B1", "C1"), ("A1", "B1", "C2"), ("A1", "B2", "C1"), ("A1", "B2", "C2"),
   ("A1", "B3", "C1"), ("A1", "B3", "C2"), ("A2", "B1", "C1"), ("A2", "B1", "C2"),
   ("A2", "B2", "C1"), ("A2", "B2", "C2"), ("A2", "B3", "C1"), ("A2", "B3", "C2"),
   ("A3", "B1", "C1"), ("A3", "B1", "C2"), ("A3", "B2", "C1"), ("A3", "B2", "C2"),
   ("A3", "B3", "C1"), ("A3", "B3", "C2")]

def selected_event_a1 : List (String × String × String) :=
  [("A1", "B1", "C1"), ("A1", "B1", "C2"), ("A1", "B2", "C1"), ("A1", "B2", "C2"),
   ("A1", "B3", "C1"), ("A1", "B3", "C2")]

def selected_event_neither_b1_c1 : List (String × String × String) :=
  [("A1", "B2", "C2"), ("A1", "B3", "C2"), ("A2", "B2", "C2"), ("A2", "B3", "C2"),
   ("A3", "B2", "C2"), ("A3", "B3", "C2"), ("A1", "B2", "C1"), ("A1", "B3", "C1"),
   ("A2", "B2", "C1"), ("A2", "B3", "C1"), ("A3", "B2", "C1"), ("A3", "B3", "C1"),
   ("A1", "B2", "C2"), ("A1", "B3", "C2"), ("A2", "B2", "C2")]

-- Lean 4 statements for proofs
theorem prob_A1_selected : 
  (selected_event_a1.length : ℚ) / (volunteers.length : ℚ) = 1 / 3 :=
  by
    sorry

theorem prob_neither_B1_C1_selected :
  (selected_event_neither_b1_c1.length : ℚ) / (volunteers.length : ℚ) = 5 / 6 :=
  by
    sorry

end prob_A1_selected_prob_neither_B1_C1_selected_l2_2539


namespace find_weight_of_B_l2_2391

theorem find_weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 44) : B = 33 :=
by 
  sorry

end find_weight_of_B_l2_2391


namespace arithmetic_sequence_sum_l2_2468

-- Define the arithmetic sequence {a_n}
noncomputable def a_n (n : ℕ) : ℝ := sorry

-- Given condition
axiom h1 : a_n 3 + a_n 7 = 37

-- Proof statement
theorem arithmetic_sequence_sum : a_n 2 + a_n 4 + a_n 6 + a_n 8 = 74 :=
by
  sorry

end arithmetic_sequence_sum_l2_2468


namespace petya_vasya_equal_again_l2_2051

theorem petya_vasya_equal_again (n : ℤ) (hn : n ≠ 0) :
  ∃ (k m : ℕ), (∃ P V : ℤ, P = n + 10 * k ∧ V = n - 10 * k ∧ 2014 * P * V = n) :=
sorry

end petya_vasya_equal_again_l2_2051


namespace yarn_length_proof_l2_2669

def green_length := 156
def total_length := 632

noncomputable def red_length (x : ℕ) := green_length * x + 8

theorem yarn_length_proof (x : ℕ) (green_length_eq : green_length = 156)
  (total_length_eq : green_length + red_length x = 632) : x = 3 :=
by {
  sorry
}

end yarn_length_proof_l2_2669


namespace woody_writing_time_l2_2641

open Real

theorem woody_writing_time (W : ℝ) 
  (h1 : ∃ n : ℝ, n * 12 = W * 12 + 3) 
  (h2 : 12 * W + (12 * W + 3) = 39) :
  W = 1.5 :=
by sorry

end woody_writing_time_l2_2641


namespace lee_sold_action_figures_l2_2348

-- Defining variables and conditions based on the problem
def sneaker_cost : ℕ := 90
def saved_money : ℕ := 15
def price_per_action_figure : ℕ := 10
def remaining_money : ℕ := 25

-- Theorem statement asserting that Lee sold 10 action figures
theorem lee_sold_action_figures : 
  (sneaker_cost - saved_money + remaining_money) / price_per_action_figure = 10  :=
by
  sorry

end lee_sold_action_figures_l2_2348


namespace sum_of_square_areas_l2_2032

theorem sum_of_square_areas (a b : ℝ)
  (h1 : a + b = 14)
  (h2 : a - b = 2) :
  a^2 + b^2 = 100 := by
  sorry

end sum_of_square_areas_l2_2032


namespace remaining_people_statement_l2_2049

-- Definitions of conditions
def number_of_people : Nat := 10
def number_of_knights (K : Nat) : Prop := K ≤ number_of_people
def number_of_liars (L : Nat) : Prop := L ≤ number_of_people
def statement (s : String) : Prop := s = "There are more liars" ∨ s = "There are equal numbers"

-- Main theorem
theorem remaining_people_statement (K L : Nat) (h_total : K + L = number_of_people) 
  (h_knights_behavior : ∀ k, k < K → statement "There are equal numbers") 
  (h_liars_behavior : ∀ l, l < L → statement "There are more liars") :
  K = 5 → L = 5 → ∀ i, i < number_of_people → (i < 5 → statement "There are more liars") ∧ (i >= 5 → statement "There are equal numbers") := 
by
  sorry

end remaining_people_statement_l2_2049


namespace wilma_garden_rows_l2_2874

theorem wilma_garden_rows :
  ∃ (rows : ℕ),
    (∃ (yellow green red total : ℕ),
      yellow = 12 ∧
      green = 2 * yellow ∧
      red = 42 ∧
      total = yellow + green + red ∧
      total / 13 = rows ∧
      rows = 6) :=
sorry

end wilma_garden_rows_l2_2874


namespace average_of_distinct_numbers_l2_2932

theorem average_of_distinct_numbers (A B C D : ℕ) (hA : A = 1 ∨ A = 3 ∨ A = 5 ∨ A = 7)
                                   (hB : B = 1 ∨ B = 3 ∨ B = 5 ∨ B = 7)
                                   (hC : C = 1 ∨ C = 3 ∨ C = 5 ∨ C = 7)
                                   (hD : D = 1 ∨ D = 3 ∨ D = 5 ∨ D = 7)
                                   (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
    (A + B + C + D) / 4 = 4 := by
  sorry

end average_of_distinct_numbers_l2_2932


namespace distinct_domino_paths_l2_2798

/-- Matt will arrange five identical, dotless dominoes (1 by 2 rectangles) 
on a 6 by 4 grid so that a path is formed from the upper left-hand corner 
(0, 0) to the lower right-hand corner (4, 5). Prove that the number of 
distinct arrangements is 126. -/
theorem distinct_domino_paths : 
  let m := 4
  let n := 5
  let total_moves := m + n
  let right_moves := m
  let down_moves := n
  (total_moves.choose right_moves) = 126 := by
{ 
  sorry 
}

end distinct_domino_paths_l2_2798


namespace overall_percentage_badminton_l2_2061

theorem overall_percentage_badminton (N S : ℕ) (pN pS : ℝ) :
  N = 1500 → S = 1800 → pN = 0.30 → pS = 0.35 → 
  ( (N * pN + S * pS) / (N + S) ) * 100 = 33 := 
by
  intros hN hS hpN hpS
  sorry

end overall_percentage_badminton_l2_2061


namespace quadrilateral_equal_area_division_l2_2649

noncomputable def intersection_point (A B C D : ℝ × ℝ) :=
  (23 / 6 : ℝ, 7.5)

theorem quadrilateral_equal_area_division :
  let A := (1,1) : ℝ × ℝ in
  let B := (2,4) : ℝ × ℝ in
  let C := (5,4) : ℝ × ℝ in
  let D := (6,1) : ℝ × ℝ in
  let I := intersection_point A B C D in
  I = ⟨ 23/6, 7.5 ⟩ ∧ (23 + 6 + 15 + 2 = 46) :=
by sorry

end quadrilateral_equal_area_division_l2_2649


namespace triangle_perimeter_l2_2702

-- Define the side lengths
def a : ℕ := 7
def b : ℕ := 10
def c : ℕ := 15

-- Define the perimeter
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Statement of the proof problem
theorem triangle_perimeter : perimeter 7 10 15 = 32 := by
  sorry

end triangle_perimeter_l2_2702


namespace heather_ends_up_with_45_blocks_l2_2956

-- Conditions
def initialBlocks (Heather : Type) : ℕ := 86
def sharedBlocks (Heather : Type) : ℕ := 41

-- The theorem to prove
theorem heather_ends_up_with_45_blocks (Heather : Type) :
  (initialBlocks Heather) - (sharedBlocks Heather) = 45 :=
by
  sorry

end heather_ends_up_with_45_blocks_l2_2956


namespace lindy_total_distance_traveled_l2_2258

theorem lindy_total_distance_traveled 
    (initial_distance : ℕ)
    (jack_speed : ℕ)
    (christina_speed : ℕ)
    (lindy_speed : ℕ) 
    (meet_time : ℕ)
    (distance : ℕ) :
    initial_distance = 150 →
    jack_speed = 7 →
    christina_speed = 8 →
    lindy_speed = 10 →
    meet_time = initial_distance / (jack_speed + christina_speed) →
    distance = lindy_speed * meet_time →
    distance = 100 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end lindy_total_distance_traveled_l2_2258


namespace simplify_expression_l2_2908

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l2_2908


namespace both_shots_hit_target_exactly_one_shot_hits_target_l2_2028

variable (p q : Prop)

theorem both_shots_hit_target : (p ∧ q) := sorry

theorem exactly_one_shot_hits_target : ((p ∧ ¬ q) ∨ (¬ p ∧ q)) := sorry

end both_shots_hit_target_exactly_one_shot_hits_target_l2_2028


namespace range_of_f_l2_2672

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 2

theorem range_of_f : Set.range f = Set.Ici 3 :=
by
  sorry

end range_of_f_l2_2672


namespace solution_of_inequality_system_l2_2228

theorem solution_of_inequality_system (x : ℝ) : 
  (x + 1 > 0 ∧ x + 3 ≤ 4) ↔ (-1 < x ∧ x ≤ 1) := 
by
  sorry

end solution_of_inequality_system_l2_2228


namespace angle_equivalence_modulo_l2_2971

-- Defining the given angles
def theta1 : ℤ := -510
def theta2 : ℤ := 210

-- Proving that the angles are equivalent modulo 360
theorem angle_equivalence_modulo : theta1 % 360 = theta2 % 360 :=
by sorry

end angle_equivalence_modulo_l2_2971


namespace x_intercept_of_parabola_l2_2302

theorem x_intercept_of_parabola (a b c : ℝ)
    (h_vertex : ∀ x, (a * (x - 5)^2 + 9 = y) → (x, y) = (5, 9))
    (h_intercept : ∀ x, (a * x^2 + b * x + c = 0) → x = 0 ∨ y = 0) :
    ∃ x0 : ℝ, x0 = 10 :=
by
  sorry

end x_intercept_of_parabola_l2_2302


namespace smallest_possible_value_of_N_l2_2273

-- Define the dimensions of the block
variables (l m n : ℕ) 

-- Define the condition that the product of dimensions minus one is 143
def hidden_cubes_count (l m n : ℕ) : Prop := (l - 1) * (m - 1) * (n - 1) = 143

-- Define the total number of cubes in the outer block
def total_cubes (l m n : ℕ) : ℕ := l * m * n

-- The final proof statement
theorem smallest_possible_value_of_N : 
  ∃ (l m n : ℕ), hidden_cubes_count l m n → N = total_cubes l m n → N = 336 :=
sorry

end smallest_possible_value_of_N_l2_2273


namespace part1_solution_part2_solution_l2_2739

-- Definitions for costs
variables (x y : ℝ)
variables (cost_A cost_B : ℝ)

-- Conditions
def condition1 : 80 * x + 35 * y = 2250 :=
  sorry

def condition2 : x = y - 15 :=
  sorry

-- Part 1: Cost of one bottle of each disinfectant
theorem part1_solution : x = cost_A ∧ y = cost_B :=
  sorry

-- Additional conditions for part 2
variables (m : ℕ)
variables (total_bottles : ℕ := 50)
variables (budget : ℝ := 1200)

-- Conditions for part 2
def condition3 : m + (total_bottles - m) = total_bottles :=
  sorry

def condition4 : 15 * m + 30 * (total_bottles - m) ≤ budget :=
  sorry

-- Part 2: Minimum number of bottles of Class A disinfectant
theorem part2_solution : m ≥ 20 :=
  sorry

end part1_solution_part2_solution_l2_2739


namespace number_of_persons_l2_2891

theorem number_of_persons
    (total_amount : ℕ) 
    (amount_per_person : ℕ) 
    (h1 : total_amount = 42900) 
    (h2 : amount_per_person = 1950) :
    total_amount / amount_per_person = 22 :=
by
  sorry

end number_of_persons_l2_2891


namespace expand_and_simplify_l2_2240

noncomputable def expanded_expr (a : ℝ) : ℝ :=
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6)

theorem expand_and_simplify (a : ℝ) :
  expanded_expr a = a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 :=
by {
  -- Proof goes here
  sorry
}

end expand_and_simplify_l2_2240


namespace max_d_6_digit_multiple_33_l2_2160

theorem max_d_6_digit_multiple_33 (x d e : ℕ) 
  (hx : 1 ≤ x ∧ x ≤ 9) 
  (hd : 0 ≤ d ∧ d ≤ 9) 
  (he : 0 ≤ e ∧ e ≤ 9)
  (h1 : (x * 100000 + 50000 + d * 1000 + 300 + 30 + e) ≥ 100000) 
  (h2 : (x + d + e + 11) % 3 = 0)
  (h3 : ((x + d - e - 5 + 11) % 11 = 0)) :
  d = 9 := 
sorry

end max_d_6_digit_multiple_33_l2_2160


namespace alicia_candies_problem_l2_2573

theorem alicia_candies_problem :
  ∃ (n : ℕ), n >= 100 ∧ n < 1000 ∧ (n % 9 = 7) ∧ (n % 7 = 5) ∧ n = 124 :=
by
  sorry

end alicia_candies_problem_l2_2573


namespace min_square_distance_l2_2512

theorem min_square_distance (x y z w : ℝ) (h1 : x * y = 4) (h2 : z^2 + 4 * w^2 = 4) : (x - z)^2 + (y - w)^2 ≥ 1.6 :=
sorry

end min_square_distance_l2_2512


namespace numbers_at_distance_1_from_neg2_l2_2215

theorem numbers_at_distance_1_from_neg2 : 
  ∃ x : ℤ, (|x + 2| = 1) ∧ (x = -1 ∨ x = -3) :=
by
  sorry

end numbers_at_distance_1_from_neg2_l2_2215


namespace find_a_l2_2015

def setA (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }
def setB : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }

theorem find_a (a : ℝ) : setA a ⊆ setB ↔ a = 0 ∨ a = 1 ∨ a = 1 / 2 :=
by
  sorry

end find_a_l2_2015


namespace parking_space_area_l2_2700

theorem parking_space_area (L W : ℕ) (h1 : L = 9) (h2 : 2 * W + L = 37) : L * W = 126 :=
by
  -- Proof omitted.
  sorry

end parking_space_area_l2_2700


namespace real_solutions_in_interval_l2_2589

noncomputable def problem_statement (x : ℝ) : Prop :=
  (x + 1 > 0) ∧ 
  (x ≠ -1) ∧
  (x^2 / (x + 1 - Real.sqrt (x + 1))^2 < (x^2 + 3 * x + 18) / (x + 1)^2)
  
theorem real_solutions_in_interval (x : ℝ) (h : problem_statement x) : -1 < x ∧ x < 3 :=
sorry

end real_solutions_in_interval_l2_2589


namespace find_missing_dimension_l2_2363

-- Definitions based on conditions
def is_dimension_greatest_area (x : ℝ) : Prop :=
  max (2 * x) (max (3 * x) 6) = 15

-- The final statement to prove
theorem find_missing_dimension (x : ℝ) (h1 : is_dimension_greatest_area x) : x = 5 :=
sorry

end find_missing_dimension_l2_2363


namespace percentage_of_female_students_25_or_older_l2_2774

theorem percentage_of_female_students_25_or_older
  (T : ℝ) (M F : ℝ) (P : ℝ)
  (h1 : M = 0.40 * T)
  (h2 : F = 0.60 * T)
  (h3 : 0.56 = (0.20 * T) + (0.60 * (1 - P) * T)) :
  P = 0.40 :=
by
  sorry

end percentage_of_female_students_25_or_older_l2_2774


namespace total_price_of_order_l2_2562

theorem total_price_of_order :
  let num_ice_cream_bars := 225
  let price_per_ice_cream_bar := 0.60
  let num_sundaes := 125
  let price_per_sundae := 0.52
  (num_ice_cream_bars * price_per_ice_cream_bar + num_sundaes * price_per_sundae) = 200 := 
by
  -- The proof steps go here
  sorry

end total_price_of_order_l2_2562


namespace circle_through_A_B_C_l2_2163

-- Definitions of points A, B, and C
def A : ℝ × ℝ := (1, 12)
def B : ℝ × ℝ := (7, 10)
def C : ℝ × ℝ := (-9, 2)

-- Definition of the expected standard equation of the circle
def circle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 100

-- Theorem stating that the expected equation is the equation of the circle through points A, B, and C
theorem circle_through_A_B_C : 
  ∀ (x y : ℝ),
  (x, y) = A ∨ (x, y) = B ∨ (x, y) = C → 
  circle_eq x y := sorry

end circle_through_A_B_C_l2_2163


namespace max_value_condition_l2_2206

noncomputable def maximumValue (a b c : ℝ) : ℝ :=
  a + 2 * Real.sqrt (a * b) + Real.cbrt (a * b * c)

theorem max_value_condition (a b c : ℝ) (h : a + 2 * b + c = 2) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) :
  maximumValue a b c ≤ 2.545 := 
sorry

end max_value_condition_l2_2206


namespace opposite_of_3_is_neg3_l2_2824

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2824


namespace fraction_of_girls_l2_2340

variable (total_students : ℕ) (number_of_boys : ℕ)

theorem fraction_of_girls (h1 : total_students = 160) (h2 : number_of_boys = 60) :
    (total_students - number_of_boys) / total_students = 5 / 8 := by
  sorry

end fraction_of_girls_l2_2340


namespace find_interest_rate_l2_2963

-- Conditions
def principal1 : ℝ := 100
def rate1 : ℝ := 0.05
def time1 : ℕ := 48

def principal2 : ℝ := 600
def time2 : ℕ := 4

-- The given interest produced by the first amount
def interest1 : ℝ := principal1 * rate1 * time1

-- The interest produced by the second amount should be the same
def interest2 (rate2 : ℝ) : ℝ := principal2 * rate2 * time2

-- The interest rate to prove
def rate2_correct : ℝ := 0.1

theorem find_interest_rate :
  ∃ rate2 : ℝ, interest2 rate2 = interest1 ∧ rate2 = rate2_correct :=
by
  sorry

end find_interest_rate_l2_2963


namespace smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square_l2_2694

theorem smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square :
  ∃ p : ℕ, Prime p ∧ (∃ k m : ℤ, k^2 = p - 6 ∧ m^2 = p + 9 ∧ m^2 - k^2 = 15) ∧ p = 127 :=
sorry

end smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square_l2_2694


namespace romeo_total_profit_is_55_l2_2516

-- Defining the conditions
def number_of_bars : ℕ := 5
def cost_per_bar : ℕ := 5
def packaging_cost_per_bar : ℕ := 2
def total_selling_price : ℕ := 90

-- Defining the profit calculation
def total_cost_per_bar := cost_per_bar + packaging_cost_per_bar
def selling_price_per_bar := total_selling_price / number_of_bars
def profit_per_bar := selling_price_per_bar - total_cost_per_bar
def total_profit := profit_per_bar * number_of_bars

-- Proving the total profit
theorem romeo_total_profit_is_55 : total_profit = 55 :=
by
  sorry

end romeo_total_profit_is_55_l2_2516


namespace expand_and_simplify_l2_2237

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l2_2237


namespace at_least_one_not_less_than_two_l2_2306

open Real

theorem at_least_one_not_less_than_two (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  (a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) → false := 
sorry

end at_least_one_not_less_than_two_l2_2306


namespace max_profit_l2_2718

noncomputable def profit (x : ℝ) : ℝ :=
  10 * (x - 40) * (100 - x)

theorem max_profit (x : ℝ) (hx : x > 40) :
  (profit 70 = 9000) ∧ ∀ y > 40, profit y ≤ 9000 := by
  sorry

end max_profit_l2_2718


namespace value_of_a_l2_2601

noncomputable def f (x : ℝ) : ℝ := x^2 + 10
noncomputable def g (x : ℝ) : ℝ := x^2 - 5

theorem value_of_a (a : ℝ) (h₁ : a > 0) (h₂ : f (g a) = 18) :
  a = Real.sqrt (5 + 2 * Real.sqrt 2) ∨ a = Real.sqrt (5 - 2 * Real.sqrt 2) := 
by
  sorry

end value_of_a_l2_2601


namespace vertices_form_parabola_l2_2204

variables (a c d : ℝ) (h_a : 0 < a) (h_c : 0 < c) (h_d : 0 < d)

/-- Given fixed positive numbers a, c, and d, for any real number b, 
the set of vertices (x_t, y_t) of the parabolas y = a*x^2 + (b+c)*x + d 
forms a new parabola y = -a*x^2 + d. -/
theorem vertices_form_parabola (b : ℝ) :
  let x_v := - (b + c) / (2 * a),
      y_v := - (b + c)^2 / (4 * a) + d in
  y_v = -a * x_v^2 + d := 
begin
  sorry
end

end vertices_form_parabola_l2_2204


namespace two_digit_numbers_satisfying_l2_2041

def P (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a * b

def S (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a + b

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_numbers_satisfying (n : ℕ) : 
  is_two_digit n → n = P n + S n ↔ (n % 10 = 9) :=
by
  sorry

end two_digit_numbers_satisfying_l2_2041


namespace initial_average_weight_l2_2687

theorem initial_average_weight
  (A : ℚ) -- Define A as a rational number since we are dealing with division 
  (h1 : 6 * A + 133 = 7 * 151) : -- Condition from the problem translated into an equation
  A = 154 := -- Statement we need to prove
by
  sorry -- Placeholder for the proof

end initial_average_weight_l2_2687


namespace equilateral_triangle_of_condition_l2_2442

theorem equilateral_triangle_of_condition (a b c : ℝ) (h : a^2 + 2 * b^2 + c^2 - 2 * b * (a + c) = 0) : a = b ∧ b = c :=
by
  /- Proof goes here -/
  sorry

end equilateral_triangle_of_condition_l2_2442


namespace first_term_of_geometric_series_l2_2277

theorem first_term_of_geometric_series (r a S : ℝ) (h_r : r = 1 / 4) (h_S : S = 40) 
  (h_geometric_sum : S = a / (1 - r)) : a = 30 :=
by
  -- The proof would go here, but we place a sorry to skip the proof.
  sorry

end first_term_of_geometric_series_l2_2277


namespace regular_price_of_fish_l2_2271

theorem regular_price_of_fish (discounted_price_per_quarter_pound : ℝ)
  (discount : ℝ) (hp1 : discounted_price_per_quarter_pound = 2) (hp2 : discount = 0.4) :
  ∃ x : ℝ, x = (40 / 3) :=
by
  sorry

end regular_price_of_fish_l2_2271


namespace math_problem_l2_2311

variable {f : ℝ → ℝ}
variable {g : ℝ → ℝ}

noncomputable def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
noncomputable def odd_function (g : ℝ → ℝ) := ∀ x : ℝ, g x = -g (-x)

theorem math_problem
  (hf_even : even_function f)
  (hf_0 : f 0 = 1)
  (hg_odd : odd_function g)
  (hgf : ∀ x : ℝ, g x = f (x - 1)) :
  f 2011 + f 2012 + f 2013 = 1 := sorry

end math_problem_l2_2311


namespace least_positive_angle_is_75_l2_2936

noncomputable def least_positive_angle (θ : ℝ) : Prop :=
  cos (10 * Real.pi / 180) = sin (15 * Real.pi / 180) + sin θ

theorem least_positive_angle_is_75 :
  least_positive_angle (75 * Real.pi / 180) :=
by
  sorry

end least_positive_angle_is_75_l2_2936


namespace degree_of_divisor_l2_2885

theorem degree_of_divisor (f q r d : Polynomial ℝ)
  (h_f : f.degree = 15)
  (h_q : q.degree = 9)
  (h_r : r = Polynomial.C 5 * X^4 + Polynomial.C 3 * X^3 - Polynomial.C 2 * X^2 + Polynomial.C 9 * X - Polynomial.C 7)
  (h_div : f = d * q + r) :
  d.degree = 6 :=
by sorry

end degree_of_divisor_l2_2885


namespace required_tiles_0_4m_l2_2509

-- Defining given conditions
def num_tiles_0_3m : ℕ := 720
def side_length_0_3m : ℝ := 0.3
def side_length_0_4m : ℝ := 0.4

-- The problem statement translated to Lean 4
theorem required_tiles_0_4m : (side_length_0_4m ^ 2) * (405 : ℝ) = (side_length_0_3m ^ 2) * (num_tiles_0_3m : ℝ) := 
by
  -- Skipping the proof
  sorry

end required_tiles_0_4m_l2_2509


namespace bounds_for_a_l2_2153

theorem bounds_for_a (a : ℝ) (h_a : a > 0) :
  ∀ x : ℝ, 0 < x ∧ x < 17 → (3 / 4) * x = (5 / 6) * (17 - x) + a → a < (153 / 12) := 
sorry

end bounds_for_a_l2_2153


namespace a_range_l2_2333

noncomputable def f (a x : ℝ) : ℝ := x^3 + a*x^2 - 2*x + 5

noncomputable def f' (a x : ℝ) : ℝ := 3*x^2 + 2*a*x - 2

theorem a_range (a : ℝ) :
  (∃ x y : ℝ, (1/3 < x ∧ x < 1/2) ∧ (1/3 < y ∧ y < 1/2) ∧ f' a x = 0 ∧ f' a y = 0) ↔
  a ∈ Set.Ioo (5/4) (5/2) :=
by
  sorry

end a_range_l2_2333


namespace compute_product_l2_2648

theorem compute_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 10 :=
sorry

end compute_product_l2_2648


namespace nature_of_roots_Q_l2_2451

noncomputable def Q (x : ℝ) : ℝ := x^6 - 4 * x^5 + 3 * x^4 - 7 * x^3 - x^2 + x + 10

theorem nature_of_roots_Q : 
  ∃ (negative_roots positive_roots : Finset ℝ),
    (∀ r ∈ negative_roots, r < 0) ∧
    (∀ r ∈ positive_roots, r > 0) ∧
    negative_roots.card = 1 ∧
    positive_roots.card > 1 ∧
    ∀ r, r ∈ negative_roots ∨ r ∈ positive_roots → Q r = 0 :=
sorry

end nature_of_roots_Q_l2_2451


namespace greatest_possible_median_l2_2361

theorem greatest_possible_median {k m r s t : ℕ} 
  (h_mean : (k + m + r + s + t) / 5 = 18) 
  (h_order : k < m ∧ m < r ∧ r < s ∧ s < t) 
  (h_t : t = 40) :
  r = 23 := sorry

end greatest_possible_median_l2_2361


namespace g_at_3_l2_2458

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem g_at_3 : g 3 = 147 :=
by
  -- Proof omitted for brevity
  sorry

end g_at_3_l2_2458


namespace minimum_value_of_linear_expression_l2_2959

theorem minimum_value_of_linear_expression :
  ∀ (x y : ℝ), |y| ≤ 2 - x ∧ x ≥ -1 → 2 * x + y ≥ -5 :=
by
  sorry

end minimum_value_of_linear_expression_l2_2959


namespace infinite_geometric_subsequence_exists_l2_2349

theorem infinite_geometric_subsequence_exists
  (a : ℕ) (d : ℕ) (h_d_pos : d > 0)
  (a_n : ℕ → ℕ)
  (h_arith_prog : ∀ n, a_n n = a + n * d) :
  ∃ (g : ℕ → ℕ), (∀ m n, m < n → g m < g n) ∧ (∃ r : ℕ, ∀ n, g (n+1) = g n * r) ∧ (∀ n, ∃ m, a_n m = g n) :=
sorry

end infinite_geometric_subsequence_exists_l2_2349


namespace equation_of_perpendicular_line_l2_2006

theorem equation_of_perpendicular_line (x y : ℝ) (l1 : 2*x - 3*y + 4 = 0) (pt : x = -2 ∧ y = -3) :
  3*(-2) + 2*(-3) + 12 = 0 := by
  sorry

end equation_of_perpendicular_line_l2_2006


namespace students_at_start_of_year_l2_2635

theorem students_at_start_of_year (S : ℝ) (h1 : S + 46.0 = 56) : S = 10 :=
sorry

end students_at_start_of_year_l2_2635


namespace rectangle_perimeter_inscribed_l2_2124

noncomputable def circle_area : ℝ := 32 * Real.pi
noncomputable def rectangle_area : ℝ := 34
noncomputable def rectangle_perimeter : ℝ := 28

theorem rectangle_perimeter_inscribed (area_circle : ℝ := 32 * Real.pi)
  (area_rectangle : ℝ := 34) : ∃ (P : ℝ), P = 28 :=
by
  use rectangle_perimeter
  sorry

end rectangle_perimeter_inscribed_l2_2124


namespace ab_cd_eq_zero_l2_2328

theorem ab_cd_eq_zero  
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1)
  (h2 : c^2 + d^2 = 1)
  (h3 : ad - bc = -1) :
  ab + cd = 0 :=
by
  sorry

end ab_cd_eq_zero_l2_2328


namespace min_value_of_expression_l2_2752

theorem min_value_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  25 ≤ (4 / a) + (9 / b) :=
sorry

end min_value_of_expression_l2_2752


namespace supplement_of_complement_of_35_degree_angle_l2_2088

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 := 
by sorry

end supplement_of_complement_of_35_degree_angle_l2_2088


namespace swap_numbers_l2_2360

theorem swap_numbers (a b : ℕ) (hc: b = 17) (ha : a = 8) : 
  ∃ c, c = b ∧ b = a ∧ a = c := 
by
  sorry

end swap_numbers_l2_2360


namespace sum_abc_equals_33_l2_2073

theorem sum_abc_equals_33 (a b c N : ℕ) (h_neq : ∀ x y, x ≠ y → x ≠ y → x ≠ y → x ≠ y ) 
(hN1 : N = 5 * a + 3 * b + 5 * c) (hN2 : N = 4 * a + 5 * b + 4 * c)
(h_range : 131 < N ∧ N < 150) : a + b + c = 33 :=
sorry

end sum_abc_equals_33_l2_2073


namespace scientific_notation_population_l2_2499

theorem scientific_notation_population :
    ∃ (a b : ℝ), (b = 5 ∧ 1412.60 * 10 ^ 6 = a * 10 ^ b ∧ a = 1.4126) :=
sorry

end scientific_notation_population_l2_2499


namespace percentage_increase_l2_2534

theorem percentage_increase (lowest_price highest_price : ℝ) (h_low : lowest_price = 15) (h_high : highest_price = 25) :
  ((highest_price - lowest_price) / lowest_price) * 100 = 66.67 :=
by
  sorry

end percentage_increase_l2_2534


namespace expression_simplification_l2_2913

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l2_2913


namespace tetrahedron_perpendicular_distances_inequalities_l2_2260

section Tetrahedron

variables {R : Type*} [LinearOrderedField R]

variables {S_A S_B S_C S_D V d_A d_B d_C d_D h_A h_B h_C h_D : R}

/-- Given areas and perpendicular distances of a tetrahedron, prove inequalities involving these parameters. -/
theorem tetrahedron_perpendicular_distances_inequalities 
  (h1 : S_A * d_A + S_B * d_B + S_C * d_C + S_D * d_D = 3 * V) : 
  (min h_A (min h_B (min h_C h_D)) ≤ d_A + d_B + d_C + d_D) ∧ 
  (d_A + d_B + d_C + d_D ≤ max h_A (max h_B (max h_C h_D))) ∧ 
  (d_A * d_B * d_C * d_D ≤ 81 * V ^ 4 / (256 * S_A * S_B * S_C * S_D)) :=
sorry

end Tetrahedron

end tetrahedron_perpendicular_distances_inequalities_l2_2260


namespace sin_double_angle_l2_2942

theorem sin_double_angle (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (2 * α) = 4 / 5 :=
sorry

end sin_double_angle_l2_2942


namespace ted_age_l2_2281

theorem ted_age (t s : ℝ) 
  (h1 : t = 3 * s - 20) 
  (h2: t + s = 70) : 
  t = 47.5 := 
by
  sorry

end ted_age_l2_2281


namespace opposite_of_three_l2_2857

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l2_2857


namespace arithmetic_expression_equals_fraction_l2_2139

theorem arithmetic_expression_equals_fraction (a b c : ℚ) :
  a = 1/8 → b = 1/9 → c = 1/28 →
  (a * b * c = 1/2016) ∨ ((a - b) * c = 1/2016) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  left
  sorry

end arithmetic_expression_equals_fraction_l2_2139


namespace solve_for_x_l2_2661

theorem solve_for_x (x: ℝ) (h: (x-3)^4 = 16): x = 5 := 
by
  sorry

end solve_for_x_l2_2661


namespace coeff_x2_product_l2_2077

open Polynomial

noncomputable def poly1 : Polynomial ℤ := -5 * X^3 - 5 * X^2 - 7 * X + 1
noncomputable def poly2 : Polynomial ℤ := -X^2 - 6 * X + 1

theorem coeff_x2_product : (poly1 * poly2).coeff 2 = 36 := by
  sorry

end coeff_x2_product_l2_2077


namespace rhombus_angles_l2_2149

-- Define the conditions for the proof
variables (a e f : ℝ) (α β : ℝ)

-- Using the geometric mean condition
def geometric_mean_condition := a^2 = e * f

-- Using the condition that diagonals of a rhombus intersect at right angles and bisect each other
def diagonals_intersect_perpendicularly := α + β = 180 ∧ α = 30 ∧ β = 150

-- Prove the question assuming the given conditions
theorem rhombus_angles (h1 : geometric_mean_condition a e f) (h2 : diagonals_intersect_perpendicularly α β) : 
  (α = 30) ∧ (β = 150) :=
sorry

end rhombus_angles_l2_2149


namespace possible_perimeters_l2_2316

-- Define the condition that the side lengths satisfy the equation
def sides_satisfy_eqn (x : ℝ) : Prop := x^2 - 6 * x + 8 = 0

-- Theorem to prove the possible perimeters
theorem possible_perimeters (x y z : ℝ) (h1 : sides_satisfy_eqn x) (h2 : sides_satisfy_eqn y) (h3 : sides_satisfy_eqn z) :
  (x + y + z = 10) ∨ (x + y + z = 6) ∨ (x + y + z = 12) := by
  sorry

end possible_perimeters_l2_2316


namespace remainder_is_3_l2_2873

theorem remainder_is_3 (x y r : ℕ) (h1 : x = 7 * y + r) (h2 : 2 * x = 18 * y + 2) (h3 : 11 * y - x = 1)
  (hrange : 0 ≤ r ∧ r < 7) : r = 3 := 
sorry

end remainder_is_3_l2_2873


namespace find_alpha_l2_2599

open Real

def alpha_is_acute (α : ℝ) : Prop := 0 < α ∧ α < π / 2

theorem find_alpha (α : ℝ) (h1 : alpha_is_acute α) (h2 : sin (α - 10 * (pi / 180)) = sqrt 3 / 2) : α = 70 * (pi / 180) :=
sorry

end find_alpha_l2_2599


namespace girls_count_l2_2339

-- Definition of the conditions
variables (B G : ℕ)

def college_conditions (B G : ℕ) : Prop :=
  (B + G = 416) ∧ (B = (8 * G) / 5)

-- Statement to prove
theorem girls_count (B G : ℕ) (h : college_conditions B G) : G = 160 :=
by
  sorry

end girls_count_l2_2339


namespace expand_simplify_expression_l2_2244

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l2_2244


namespace opposite_of_three_l2_2846

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l2_2846


namespace divisibility_of_polynomial_l2_2793

theorem divisibility_of_polynomial (n : ℕ) (h : n ≥ 1) : 
  ∃ primes : Finset ℕ, primes.card = n ∧ ∀ p ∈ primes, p.Prime ∧ p ∣ (2^(2^n) + 2^(2^(n-1)) + 1) :=
sorry

end divisibility_of_polynomial_l2_2793


namespace sqrt_nested_eq_l2_2587

theorem sqrt_nested_eq (y : ℝ) (hy : 0 ≤ y) :
  Real.sqrt (y * Real.sqrt (y * Real.sqrt (y * Real.sqrt y))) = y ^ (9 / 4) :=
by
  sorry

end sqrt_nested_eq_l2_2587


namespace fixed_point_exists_find_ellipse_equation_l2_2438

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

def tangent_line (x y : ℝ) : Prop :=
  y = x + sqrt 6

def line_passing_through_fixed_point (x y : ℝ) (k m : ℝ) (h : k ≠ 0) : Prop :=
  ∃ x1 x2 y1 y2 : ℝ,
    ellipse x1 y1 ∧ ellipse x2 y2 ∧
    y1 = k * x1 + m ∧
    y2 = k * x2 + m ∧
    (x1 - 2) * (x2 - 2) + y1 * y2 = 0 ∧
    sqrt (7 * (m / k)^2 + 16 * (m / k) + 4) = 0

theorem fixed_point_exists (k : ℝ) (h : k ≠ 0) :
  ∀ m : ℝ, ∃ x y : ℝ, line_passing_through_fixed_point x y k m h → (x, y) = (2 / 7, 0) :=
sorry

theorem find_ellipse_equation :
  ∃ a b : ℝ, 0 < b ∧ b < a ∧ a = 2 ∧ b = sqrt 3 ∧ ellipse = (λ x y, (x^2 / (a^2)) + (y^2 / (b^2)) = 1) :=
sorry

end fixed_point_exists_find_ellipse_equation_l2_2438


namespace min_value_SN64_by_aN_is_17_over_2_l2_2750

noncomputable def a_n (n : ℕ) : ℕ := 2 * n
noncomputable def S_n (n : ℕ) : ℕ := n^2 + n

theorem min_value_SN64_by_aN_is_17_over_2 :
  ∃ (n : ℕ), 2 ≤ n ∧ (a_2 = 4 ∧ S_10 = 110) →
  ((S_n n + 64) / a_n n) = 17 / 2 :=
by
  sorry

end min_value_SN64_by_aN_is_17_over_2_l2_2750


namespace cost_of_supplies_l2_2075

theorem cost_of_supplies (x y z : ℝ) 
  (h1 : 3 * x + 7 * y + z = 3.15) 
  (h2 : 4 * x + 10 * y + z = 4.2) :
  (x + y + z = 1.05) :=
by 
  sorry

end cost_of_supplies_l2_2075


namespace Lee_surpasses_Hernandez_in_May_l2_2193

def monthly_totals_Hernandez : List ℕ :=
  [4, 8, 9, 5, 7, 6]

def monthly_totals_Lee : List ℕ :=
  [3, 9, 10, 6, 8, 8]

def cumulative_sum (lst : List ℕ) : List ℕ :=
  List.scanl (· + ·) 0 lst

noncomputable def cumulative_Hernandez := cumulative_sum monthly_totals_Hernandez
noncomputable def cumulative_Lee := cumulative_sum monthly_totals_Lee

-- Lean 4 statement asserting when Lee surpasses Hernandez in cumulative home runs
theorem Lee_surpasses_Hernandez_in_May :
  cumulative_Hernandez[3] < cumulative_Lee[3] :=
sorry

end Lee_surpasses_Hernandez_in_May_l2_2193


namespace simplify_expression_eq_square_l2_2899

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l2_2899


namespace axis_of_symmetry_circle_l2_2625

theorem axis_of_symmetry_circle (a : ℝ) : 
  (2 * a + 0 - 1 = 0) ↔ (a = 1 / 2) :=
by
  sorry

end axis_of_symmetry_circle_l2_2625


namespace opposite_of_three_l2_2844

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l2_2844


namespace lara_gives_betty_l2_2480

variables (X Y : ℝ)

-- Conditions
-- Lara has spent X dollars
-- Betty has spent Y dollars
-- Y is greater than X
theorem lara_gives_betty (h : Y > X) : (Y - X) / 2 = (X + Y) / 2 - X :=
by
  sorry

end lara_gives_betty_l2_2480


namespace minimum_people_who_like_both_l2_2467

theorem minimum_people_who_like_both
    (total_people : ℕ)
    (vivaldi_likers : ℕ)
    (chopin_likers : ℕ)
    (people_surveyed : total_people = 150)
    (like_vivaldi : vivaldi_likers = 120)
    (like_chopin : chopin_likers = 90) :
    ∃ (both_likers : ℕ), both_likers = 60 ∧
                            vivaldi_likers + chopin_likers - both_likers ≤ total_people :=
by 
  sorry

end minimum_people_who_like_both_l2_2467


namespace cost_of_magazine_l2_2704

theorem cost_of_magazine (B M : ℝ) 
  (h1 : 2 * B + 2 * M = 26) 
  (h2 : B + 3 * M = 27) : 
  M = 7 := 
by 
  sorry

end cost_of_magazine_l2_2704


namespace jill_and_emily_total_peaches_l2_2220

-- Define each person and their conditions
variables (Steven Jake Jill Maria Emily : ℕ)

-- Given conditions
def steven_has_peaches : Steven = 14 := sorry
def jake_has_fewer_than_steven : Jake = Steven - 6 := sorry
def jake_has_more_than_jill : Jake = Jill + 3 := sorry
def maria_has_twice_jake : Maria = 2 * Jake := sorry
def emily_has_fewer_than_maria : Emily = Maria - 9 := sorry

-- The theorem statement combining the conditions and the required result
theorem jill_and_emily_total_peaches (Steven Jake Jill Maria Emily : ℕ)
  (h1 : Steven = 14) 
  (h2 : Jake = Steven - 6) 
  (h3 : Jake = Jill + 3) 
  (h4 : Maria = 2 * Jake) 
  (h5 : Emily = Maria - 9) : 
  Jill + Emily = 12 := 
sorry

end jill_and_emily_total_peaches_l2_2220


namespace initial_amount_spent_l2_2099

theorem initial_amount_spent (X : ℝ) 
    (h_bread : X - 3 ≥ 0) 
    (h_candy : X - 3 - 2 ≥ 0) 
    (h_turkey : X - 3 - 2 - (1/3) * (X - 3 - 2) ≥ 0) 
    (h_remaining : X - 3 - 2 - (1/3) * (X - 3 - 2) = 18) : X = 32 := 
sorry

end initial_amount_spent_l2_2099


namespace sarah_math_homework_pages_l2_2657

theorem sarah_math_homework_pages (x : ℕ) 
  (h1 : ∀ page, 4 * page = 4 * 6 + 4 * x)
  (h2 : 40 = 4 * 6 + 4 * x) : 
  x = 4 :=
by 
  sorry

end sarah_math_homework_pages_l2_2657


namespace matrix_power_101_l2_2200

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![1, 0, 0],
  ![0, 0, 1],
  ![0, 1, 0]
]

theorem matrix_power_101 :
  B ^ (101 : ℕ) = B := sorry

end matrix_power_101_l2_2200


namespace initial_bananas_l2_2708

theorem initial_bananas (bananas_left: ℕ) (eaten: ℕ) (basket: ℕ) 
                        (h_left: bananas_left = 100) 
                        (h_eaten: eaten = 70) 
                        (h_basket: basket = 2 * eaten): 
  bananas_left + eaten + basket = 310 :=
by
  sorry

end initial_bananas_l2_2708


namespace simplify_expression_l2_2903

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l2_2903


namespace john_writing_time_l2_2035

def pages_per_day : ℕ := 20
def pages_per_book : ℕ := 400
def number_of_books : ℕ := 3

theorem john_writing_time : (pages_per_book / pages_per_day) * number_of_books = 60 :=
by
  -- The proof should be placed here.
  sorry

end john_writing_time_l2_2035


namespace sum_of_values_of_M_l2_2068

theorem sum_of_values_of_M (M : ℝ) (h : M * (M - 8) = 12) :
  (∃ M1 M2 : ℝ, M^2 - 8 * M - 12 = 0 ∧ M1 + M2 = 8) :=
sorry

end sum_of_values_of_M_l2_2068


namespace find_m_containing_2015_l2_2352

theorem find_m_containing_2015 : 
  ∃ n : ℕ, ∀ k, 0 ≤ k ∧ k < n → 2015 = n^3 → (1979 + 2*k < 2015 ∧ 2015 < 1979 + 2*k + 2*n) :=
by
  sorry

end find_m_containing_2015_l2_2352


namespace quadratic_translation_l2_2225

theorem quadratic_translation (b c : ℝ) :
  (∀ x : ℝ, (x^2 + b * x + c = (x - 3)^2 - 2)) →
  b = 4 ∧ c = 6 :=
by
  sorry

end quadratic_translation_l2_2225


namespace athlete_A_most_stable_l2_2528

noncomputable def athlete_A_variance : ℝ := 0.019
noncomputable def athlete_B_variance : ℝ := 0.021
noncomputable def athlete_C_variance : ℝ := 0.020
noncomputable def athlete_D_variance : ℝ := 0.022

theorem athlete_A_most_stable :
  athlete_A_variance < athlete_B_variance ∧
  athlete_A_variance < athlete_C_variance ∧
  athlete_A_variance < athlete_D_variance :=
by {
  sorry
}

end athlete_A_most_stable_l2_2528


namespace range_of_sum_l2_2461

theorem range_of_sum (x y : ℝ) (h : x^2 + x + y^2 + y = 0) : 
  -2 ≤ x + y ∧ x + y ≤ 0 :=
sorry

end range_of_sum_l2_2461


namespace lydia_age_when_planted_l2_2972

-- Definition of the conditions
def years_to_bear_fruit : ℕ := 7
def lydia_age_when_fruit_bears : ℕ := 11

-- Lean 4 statement to prove Lydia's age when she planted the tree
theorem lydia_age_when_planted (a : ℕ) : a = lydia_age_when_fruit_bears - years_to_bear_fruit :=
by
  have : a = 4 := by sorry
  exact this

end lydia_age_when_planted_l2_2972


namespace probability_A2_l2_2143

-- Define events and their probabilities
variable (A1 : Prop) (A2 : Prop) (B1 : Prop)
variable (P : Prop → ℝ)
variable [MeasureTheory.MeasureSpace ℝ]

-- Conditions given in the problem
axiom P_A1 : P A1 = 0.5
axiom P_B1 : P B1 = 0.5
axiom P_A2_given_A1 : P (A2 ∧ A1) / P A1 = 0.7
axiom P_A2_given_B1 : P (A2 ∧ B1) / P B1 = 0.8

-- Theorem statement to prove
theorem probability_A2 : P A2 = 0.75 :=
by
  -- Skipping the proof as per instructions
  sorry

end probability_A2_l2_2143


namespace variance_male_greater_than_female_l2_2192

noncomputable def male_scores : List ℝ := [87, 95, 89, 93, 91]
noncomputable def female_scores : List ℝ := [89, 94, 94, 89, 94]

-- Function to calculate the variance of scores
noncomputable def variance (scores : List ℝ) : ℝ :=
  let n := scores.length
  let mean := scores.sum / n
  (scores.map (λ x => (x - mean) ^ 2)).sum / n

-- We assert the problem statement
theorem variance_male_greater_than_female :
  variance male_scores > variance female_scores :=
by
  sorry

end variance_male_greater_than_female_l2_2192


namespace least_common_multiple_of_5_to_10_is_2520_l2_2556

-- Definitions of the numbers
def numbers : List ℤ := [5, 6, 7, 8, 9, 10]

-- Definition of prime factorization for verification (optional, keeping it simple)
def prime_factors (n : ℤ) : List ℤ :=
  if n = 5 then [5]
  else if n = 6 then [2, 3]
  else if n = 7 then [7]
  else if n = 8 then [2, 2, 2]
  else if n = 9 then [3, 3]
  else if n = 10 then [2, 5]
  else []

-- The property to be proved: The least common multiple of numbers is 2520
theorem least_common_multiple_of_5_to_10_is_2520 : ∃ n : ℕ, (∀ m ∈ numbers, m ∣ n) ∧ n = 2520 := by
  use 2520
  sorry

end least_common_multiple_of_5_to_10_is_2520_l2_2556


namespace simplify_expression_l2_2921

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l2_2921


namespace WidgetsPerHour_l2_2475

theorem WidgetsPerHour 
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (widgets_per_week : ℕ) 
  (H1 : hours_per_day = 8)
  (H2 : days_per_week = 5)
  (H3 : widgets_per_week = 800) : 
  widgets_per_week / (hours_per_day * days_per_week) = 20 := 
sorry

end WidgetsPerHour_l2_2475


namespace sum_of_digits_of_m_l2_2707

theorem sum_of_digits_of_m (k m : ℕ) : 
  1 ≤ k ∧ k ≤ 3 ∧ 10000 ≤ 11131 * k + 1203 ∧ 11131 * k + 1203 < 100000 ∧ 
  11131 * k + 1203 = m * m ∧ 3 * k < 10 → 
  (m.digits 10).sum = 15 :=
by 
  sorry

end sum_of_digits_of_m_l2_2707


namespace walking_speed_is_4_l2_2720

def distance : ℝ := 20
def total_time : ℝ := 3.75
def running_distance : ℝ := 10
def running_speed : ℝ := 8
def walking_distance : ℝ := 10

theorem walking_speed_is_4 (W : ℝ) 
  (H1 : running_distance + walking_distance = distance)
  (H2 : running_speed > 0)
  (H3 : walking_distance > 0)
  (H4 : W > 0)
  (H5 : walking_distance / W + running_distance / running_speed = total_time) :
  W = 4 :=
by sorry

end walking_speed_is_4_l2_2720


namespace polynomial_roots_and_coefficients_l2_2542

theorem polynomial_roots_and_coefficients 
  (a b c d e : ℝ)
  (h1 : a = 2)
  (h2 : 256 * a + 64 * b + 16 * c + 4 * d + e = 0)
  (h3 : -81 * a + 27 * b - 9 * c + 3 * d + e = 0)
  (h4 : 625 * a + 125 * b + 25 * c + 5 * d + e = 0) :
  (b + c + d) / a = 151 := 
by
  sorry

end polynomial_roots_and_coefficients_l2_2542


namespace count_even_numbers_between_500_and_800_l2_2764

theorem count_even_numbers_between_500_and_800 :
  let a := 502
  let d := 2
  let last_term := 798
  ∃ n, a + (n - 1) * d = last_term ∧ n = 149 :=
by
  sorry

end count_even_numbers_between_500_and_800_l2_2764


namespace sandy_savings_l2_2481

theorem sandy_savings (S : ℝ) :
  let last_year_savings := 0.10 * S
  let this_year_salary := 1.10 * S
  let this_year_savings := 1.65 * last_year_savings
  let P := this_year_savings / this_year_salary
  P * 100 = 15 :=
by
  let last_year_savings := 0.10 * S
  let this_year_salary := 1.10 * S
  let this_year_savings := 1.65 * last_year_savings
  let P := this_year_savings / this_year_salary
  have hP : P = 0.165 / 1.10 := by sorry
  have hP_percent : P * 100 = 15 := by sorry
  exact hP_percent

end sandy_savings_l2_2481


namespace rate_of_grapes_calculation_l2_2179

theorem rate_of_grapes_calculation (total_cost cost_mangoes cost_grapes : ℕ) (rate_grapes : ℕ):
  total_cost = 1125 →
  cost_mangoes = 9 * 55 →
  cost_grapes = 9 * rate_grapes →
  total_cost = cost_grapes + cost_mangoes →
  rate_grapes = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end rate_of_grapes_calculation_l2_2179


namespace car_R_average_speed_l2_2731

theorem car_R_average_speed 
  (R P S: ℝ)
  (h1: S = 2 * P)
  (h2: P + 2 = R)
  (h3: P = R + 10)
  (h4: S = R + 20) :
  R = 25 :=
by 
  sorry

end car_R_average_speed_l2_2731


namespace abs_equation_solution_l2_2427

theorem abs_equation_solution (x : ℝ) (h : |x - 3| = 2 * x + 4) : x = -1 / 3 :=
by
  sorry

end abs_equation_solution_l2_2427


namespace walkways_area_l2_2138

-- Define the conditions and prove the total walkway area is 416 square feet
theorem walkways_area (rows : ℕ) (columns : ℕ) (bed_width : ℝ) (bed_height : ℝ) (walkway_width : ℝ) 
  (h_rows : rows = 4) (h_columns : columns = 3) (h_bed_width : bed_width = 8) (h_bed_height : bed_height = 3) (h_walkway_width : walkway_width = 2) : 
  (rows * (bed_height + walkway_width) + walkway_width) * (columns * (bed_width + walkway_width) + walkway_width) - rows * columns * bed_width * bed_height = 416 := 
by 
  sorry

end walkways_area_l2_2138


namespace expand_simplify_expression_l2_2246

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l2_2246


namespace coplanar_vectors_x_value_l2_2613

open Matrix

-- Definitions based on conditions
def vec_a (x : ℝ) : Vector := ![1, x, 2]
def vec_b : Vector := ![0, 1, 2]
def vec_c : Vector := ![1, 0, 0]

-- Statement of the problem in Lean
theorem coplanar_vectors_x_value (x : ℝ) 
  (coplanar : det ![vec_a x, vec_b, vec_c] = 0) : x = -1 :=
  by sorry

end coplanar_vectors_x_value_l2_2613


namespace car_preference_related_to_gender_l2_2115

-- Definitions related to the given problem conditions
def total_survey_size (n : ℕ) : ℕ := 20 * n

def chi_square (n : ℕ) : ℝ := 5.556

def contingency_table (n : ℕ) : Bool → Bool → ℕ
| true, true  => 10 * n  -- Male, Like
| true, false => 2 * n   -- Male, Dislike
| false, true => 5 * n   -- Female, Like
| false, false => 3 * n  -- Female, Dislike

-- Statement of the problem to prove
-- Assuming n is a positive natural number
theorem car_preference_related_to_gender (n : ℕ) (h_pos : n > 0) :
  chi_square n ≈ 5.556 ∧
  (n = 5) ∧
  (0:ℝ) × (14 / 55 : ℝ) + 1 × (28 / 55) + 2 × (12 / 55) + 3 × (1 / 55) = 1 :=
by
  sorry

end car_preference_related_to_gender_l2_2115


namespace line_intersection_points_l2_2577

def line_intersects_axes (x y : ℝ) : Prop :=
  (4 * y - 5 * x = 20)

theorem line_intersection_points :
  ∃ p1 p2, line_intersects_axes p1.1 p1.2 ∧ line_intersects_axes p2.1 p2.2 ∧
    (p1 = (-4, 0) ∧ p2 = (0, 5)) :=
by
  sorry

end line_intersection_points_l2_2577


namespace solutions_of_system_l2_2615

theorem solutions_of_system :
  ∀ (x y : ℝ), (x - 2 * y = 1) ∧ (x^3 - 8 * y^3 - 6 * x * y = 1) ↔ y = (x - 1) / 2 :=
by
  -- Since this is a statement-only task, the detailed proof is omitted.
  -- Insert actual proof here.
  sorry

end solutions_of_system_l2_2615


namespace percentage_passed_l2_2116

def swim_club_members := 100
def not_passed_course_taken := 40
def not_passed_course_not_taken := 30
def not_passed := not_passed_course_taken + not_passed_course_not_taken

theorem percentage_passed :
  ((swim_club_members - not_passed).toFloat / swim_club_members.toFloat * 100) = 30 := by
  sorry

end percentage_passed_l2_2116


namespace least_pos_int_x_l2_2549

theorem least_pos_int_x (x : ℕ) (h1 : ∃ k : ℤ, (3 * x + 43) = 53 * k) 
  : x = 21 :=
sorry

end least_pos_int_x_l2_2549


namespace overtime_hours_correct_l2_2105

def regular_pay_rate : ℕ := 3
def max_regular_hours : ℕ := 40
def total_pay_received : ℕ := 192
def overtime_pay_rate : ℕ := 2 * regular_pay_rate
def regular_earnings : ℕ := regular_pay_rate * max_regular_hours
def additional_earnings : ℕ := total_pay_received - regular_earnings
def calculated_overtime_hours : ℕ := additional_earnings / overtime_pay_rate

theorem overtime_hours_correct :
  calculated_overtime_hours = 12 :=
by
  sorry

end overtime_hours_correct_l2_2105


namespace scientific_notation_conversion_l2_2498

theorem scientific_notation_conversion (x : ℝ) (h_population : x = 141260000) :
  x = 1.4126 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l2_2498


namespace shirt_original_price_l2_2345

theorem shirt_original_price (original_price final_price : ℝ) (h1 : final_price = 0.5625 * original_price) 
  (h2 : final_price = 19) : original_price = 33.78 :=
by
  sorry

end shirt_original_price_l2_2345


namespace opposite_of_three_l2_2841

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l2_2841


namespace average_rainfall_is_4_l2_2681

namespace VirginiaRainfall

def march_rainfall : ℝ := 3.79
def april_rainfall : ℝ := 4.5
def may_rainfall : ℝ := 3.95
def june_rainfall : ℝ := 3.09
def july_rainfall : ℝ := 4.67

theorem average_rainfall_is_4 :
  (march_rainfall + april_rainfall + may_rainfall + june_rainfall + july_rainfall) / 5 = 4 := by
  sorry

end VirginiaRainfall

end average_rainfall_is_4_l2_2681


namespace find_a_b_and_tangent_line_l2_2796

noncomputable def f (a b x : ℝ) := x^3 + 2 * a * x^2 + b * x + a
noncomputable def g (x : ℝ) := x^2 - 3 * x + 2
noncomputable def f' (a b x : ℝ) := 3 * x^2 + 4 * a * x + b
noncomputable def g' (x : ℝ) := 2 * x - 3

theorem find_a_b_and_tangent_line (a b : ℝ) :
  f a b 2 = 0 ∧ g 2 = 0 ∧ f' a b 2 = 1 ∧ g' 2 = 1 → (a = -2 ∧ b = 5 ∧ ∀ x y : ℝ, y = x - 2 ↔ x - y - 2 = 0) :=
by
  intro h
  sorry

end find_a_b_and_tangent_line_l2_2796


namespace rental_difference_l2_2110

variable (C K : ℕ)

theorem rental_difference
  (hc : 15 * C + 18 * K = 405)
  (hr : 3 * K = 2 * C) :
  C - K = 5 :=
sorry

end rental_difference_l2_2110


namespace supplement_of_complement_of_35_degree_angle_l2_2086

theorem supplement_of_complement_of_35_degree_angle : 
  ∀ α β : ℝ,  α = 35 ∧ β = (90 - α) → (180 - β) = 125 :=
by
  intros α β h
  rcases h with ⟨h1, h2⟩
  rw h1 at h2
  rw h2
  linarith

end supplement_of_complement_of_35_degree_angle_l2_2086


namespace expand_simplify_expression_l2_2245

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l2_2245


namespace find_r_l2_2791

noncomputable def f (r a : ℝ) (x : ℝ) : ℝ := (x - r - 1) * (x - r - 8) * (x - a)
noncomputable def g (r b : ℝ) (x : ℝ) : ℝ := (x - r - 2) * (x - r - 9) * (x - b)

theorem find_r
  (r a b : ℝ)
  (h_condition1 : ∀ x, f r a x - g r b x = r)
  (h_condition2 : f r a (r + 2) = r)
  (h_condition3 : f r a (r + 9) = r)
  : r = -264 / 7 := sorry

end find_r_l2_2791


namespace cyclic_inequality_l2_2021

theorem cyclic_inequality
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (a^3 + b^3 + a * b * c) + 1 / (b^3 + c^3 + a * b * c) + 1 / (c^3 + a^3 + a * b * c)) ≤ 1 / (a * b * c) :=
by
  sorry

end cyclic_inequality_l2_2021


namespace not_possible_in_five_trips_possible_in_six_trips_l2_2868

def truck_capacity := 2000
def rice_sacks := 150
def corn_sacks := 100
def rice_weight_per_sack := 60
def corn_weight_per_sack := 25

def total_rice_weight := rice_sacks * rice_weight_per_sack
def total_corn_weight := corn_sacks * corn_weight_per_sack
def total_weight := total_rice_weight + total_corn_weight

theorem not_possible_in_five_trips : total_weight > 5 * truck_capacity :=
by
  sorry

theorem possible_in_six_trips : total_weight <= 6 * truck_capacity :=
by
  sorry

#print axioms not_possible_in_five_trips
#print axioms possible_in_six_trips

end not_possible_in_five_trips_possible_in_six_trips_l2_2868


namespace total_minutes_of_game_and_ceremony_l2_2886

-- Define the components of the problem
def game_hours : ℕ := 2
def game_additional_minutes : ℕ := 35
def ceremony_minutes : ℕ := 25

-- Prove the total minutes is 180
theorem total_minutes_of_game_and_ceremony (h: game_hours = 2) (ga: game_additional_minutes = 35) (c: ceremony_minutes = 25) :
  (game_hours * 60 + game_additional_minutes + ceremony_minutes) = 180 :=
  sorry

end total_minutes_of_game_and_ceremony_l2_2886


namespace points_on_same_side_after_25_seconds_l2_2511

def movement_time (side_length : ℕ) (perimeter : ℕ)
  (speed_A speed_B : ℕ) (start_mid_B : ℕ) : ℕ :=
  25

theorem points_on_same_side_after_25_seconds (side_length : ℕ) (perimeter : ℕ)
  (speed_A speed_B : ℕ) (start_mid_B : ℕ) :
  side_length = 100 ∧ perimeter = 400 ∧ speed_A = 5 ∧ speed_B = 10 ∧ start_mid_B = 50 →
  movement_time side_length perimeter speed_A speed_B start_mid_B = 25 :=
by
  intros h
  sorry

end points_on_same_side_after_25_seconds_l2_2511


namespace simplify_expression_l2_2520

theorem simplify_expression : 8 * (15 / 9) * (-45 / 40) = -1 :=
  by
  sorry

end simplify_expression_l2_2520


namespace supplement_of_complement_of_35_degree_angle_l2_2084

theorem supplement_of_complement_of_35_degree_angle : 
  ∀ α β : ℝ,  α = 35 ∧ β = (90 - α) → (180 - β) = 125 :=
by
  intros α β h
  rcases h with ⟨h1, h2⟩
  rw h1 at h2
  rw h2
  linarith

end supplement_of_complement_of_35_degree_angle_l2_2084


namespace pq_true_l2_2171

open Real

def p : Prop := ∃ x0 : ℝ, tan x0 = sqrt 3

def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem pq_true : p ∧ q :=
by
  sorry

end pq_true_l2_2171


namespace initial_marbles_count_l2_2772

-- Definitions as per conditions in the problem
variables (x y z : ℕ)

-- Condition 1: Removing one black marble results in one-eighth of the remaining marbles being black
def condition1 : Prop := (x - 1) * 8 = (x + y - 1)

-- Condition 2: Removing three white marbles results in one-sixth of the remaining marbles being black
def condition2 : Prop := x * 6 = (x + y - 3)

-- Proof that initial total number of marbles is 9 given conditions
theorem initial_marbles_count (h1 : condition1 x y) (h2 : condition2 x y) : x + y = 9 :=
by 
  sorry

end initial_marbles_count_l2_2772


namespace exists_solution_in_interval_l2_2928

noncomputable def f (x : ℝ) : ℝ := x^3 - 2^x

theorem exists_solution_in_interval : ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ f x = 0 :=
by {
  -- Use the Intermediate Value Theorem, given f is continuous on [1, 2]
  sorry
}

end exists_solution_in_interval_l2_2928


namespace opposite_of_three_l2_2817

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l2_2817


namespace opposite_of_three_l2_2840

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l2_2840


namespace range_of_3t_plus_s_l2_2420

noncomputable def f : ℝ → ℝ := sorry

def is_increasing (f : ℝ → ℝ) := ∀ x y, x ≤ y → f x ≤ f y

def symmetric_about (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x, f (x - a) = b - f (a - x)

def satisfies_inequality (s t : ℝ) (f : ℝ → ℝ) := 
  f (s^2 - 2*s) ≥ -f (2*t - t^2)

def in_interval (s : ℝ) := 1 ≤ s ∧ s ≤ 4

theorem range_of_3t_plus_s (f : ℝ → ℝ) :
  is_increasing f ∧ symmetric_about f 3 0 →
  (∀ s t, satisfies_inequality s t f → in_interval s → -2 ≤ 3 * t + s ∧ 3 * t + s ≤ 16) :=
sorry

end range_of_3t_plus_s_l2_2420


namespace metallic_sheet_dimension_l2_2121

theorem metallic_sheet_dimension :
  ∃ w : ℝ, (∀ (h := 8) (l := 40) (v := 2688),
    v = (w - 2 * h) * (l - 2 * h) * h) → w = 30 :=
by sorry

end metallic_sheet_dimension_l2_2121


namespace average_payment_52_installments_l2_2876

theorem average_payment_52_installments :
  let first_payment : ℕ := 500
  let remaining_payment : ℕ := first_payment + 100
  let num_first_payments : ℕ := 25
  let num_remaining_payments : ℕ := 27
  let total_payments : ℕ := num_first_payments + num_remaining_payments
  let total_paid_first : ℕ := num_first_payments * first_payment
  let total_paid_remaining : ℕ := num_remaining_payments * remaining_payment
  let total_paid : ℕ := total_paid_first + total_paid_remaining
  let average_payment : ℚ := total_paid / total_payments
  average_payment = 551.92 :=
by
  sorry

end average_payment_52_installments_l2_2876


namespace fourth_proportional_segment_l2_2952

theorem fourth_proportional_segment 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  : ∃ x : ℝ, x = (b * c) / a := 
by
  sorry

end fourth_proportional_segment_l2_2952


namespace solve_system_of_equations_l2_2523

theorem solve_system_of_equations :
  ∃ x y : ℝ, (x - y = 2) ∧ (2 * x + y = 7) ∧ x = 3 ∧ y = 1 :=
by
  sorry

end solve_system_of_equations_l2_2523


namespace degree_meas_supp_compl_35_l2_2094

noncomputable def degree_meas_supplement_complement (θ : ℝ) : ℝ :=
  180 - (90 - θ)

theorem degree_meas_supp_compl_35 : degree_meas_supplement_complement 35 = 125 :=
by
  unfold degree_meas_supplement_complement
  norm_num
  sorry

end degree_meas_supp_compl_35_l2_2094


namespace least_number_subtracted_l2_2590

theorem least_number_subtracted (n : ℕ) (h : n = 2361) : 
  ∃ k, (n - k) % 23 = 0 ∧ k = 15 := 
by
  sorry

end least_number_subtracted_l2_2590


namespace most_stable_performance_l2_2430

-- Given variances for the students' scores
def variance_A : ℝ := 2.1
def variance_B : ℝ := 3.5
def variance_C : ℝ := 9
def variance_D : ℝ := 0.7

-- Prove that student D has the most stable performance
theorem most_stable_performance : 
  variance_D < variance_A ∧ variance_D < variance_B ∧ variance_D < variance_C := 
  by 
    sorry

end most_stable_performance_l2_2430


namespace opposite_of_3_l2_2835

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l2_2835


namespace degree_measure_of_supplement_of_complement_of_35_degree_angle_l2_2081

def complement (α : ℝ) : ℝ := 90 - α
def supplement (β : ℝ) : ℝ := 180 - β

theorem degree_measure_of_supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 :=
by
  sorry

end degree_measure_of_supplement_of_complement_of_35_degree_angle_l2_2081


namespace opposite_of_three_l2_2813

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l2_2813


namespace complement_of_A_in_U_l2_2986

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U : U \ A = {2, 4} := 
by
  sorry

end complement_of_A_in_U_l2_2986


namespace smallest_four_digit_multiple_of_17_is_1013_l2_2379

-- Lean definition to state the problem
def smallest_four_digit_multiple_of_17 : ℕ :=
  1013

-- Main Lean theorem to assert the correctness
theorem smallest_four_digit_multiple_of_17_is_1013 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = smallest_four_digit_multiple_of_17 :=
by
  -- proof here
  sorry

end smallest_four_digit_multiple_of_17_is_1013_l2_2379


namespace contradiction_proof_l2_2869

theorem contradiction_proof (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
sorry

end contradiction_proof_l2_2869


namespace sum_of_abc_is_33_l2_2074

theorem sum_of_abc_is_33 (a b c N : ℕ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hN1 : N = 5 * a + 3 * b + 5 * c)
    (hN2 : N = 4 * a + 5 * b + 4 * c) (hN_range : 131 < N ∧ N < 150) :
  a + b + c = 33 := 
sorry

end sum_of_abc_is_33_l2_2074


namespace sum_modulo_9_l2_2745

theorem sum_modulo_9 : 
  (88000 + 88002 + 87999 + 88001 + 88003 + 87998) % 9 = 0 := 
by
  sorry

end sum_modulo_9_l2_2745


namespace gcd_m_n_l2_2486

   -- Define m and n according to the problem statement
   def m : ℕ := 33333333
   def n : ℕ := 666666666

   -- State the theorem we want to prove
   theorem gcd_m_n : Int.gcd m n = 3 := by
     -- put proof here
     sorry
   
end gcd_m_n_l2_2486


namespace number_of_students_in_course_l2_2212

-- Define the conditions
def total (T : ℕ) :=
  (1/5 : ℚ) * T + (1/4 : ℚ) * T + (1/2 : ℚ) * T + 20 = T

-- Formalize the problem statement
theorem number_of_students_in_course : ∃ T : ℕ, total T ∧ T = 400 := 
sorry

end number_of_students_in_course_l2_2212


namespace find_n_in_arithmetic_sequence_l2_2778

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 4 then 7 else
  if n = 5 then 16 - 7 else sorry

-- Define the arithmetic sequence and the given conditions
theorem find_n_in_arithmetic_sequence (n : ℕ) (a : ℕ → ℕ) 
  (h1 : a 4 = 7) 
  (h2 : a 3 + a 6 = 16) 
  (h3 : a n = 31) :
  n = 16 :=
by
  sorry

end find_n_in_arithmetic_sequence_l2_2778


namespace distance_center_to_point_l2_2895

noncomputable def circle_center (x y : ℝ) (h : x^2 + y^2 = 6 * x - 2 * y - 15) : ℝ × ℝ :=
  (3, -1)

noncomputable def distance_to_point (cx cy px py : ℝ) : ℝ :=
  real.sqrt ((px - cx)^2 + (py - cy)^2)

theorem distance_center_to_point : 
  (distance_to_point 3 (-1) (-2) 5) = real.sqrt 61 := by
  sorry

end distance_center_to_point_l2_2895


namespace conic_sections_l2_2735

theorem conic_sections (x y : ℝ) (h : y^4 - 6 * x^4 = 3 * y^2 - 2) :
  (∃ a b : ℝ, y^2 = a + b * x^2) ∨ (∃ c d : ℝ, y^2 = c - d * x^2) :=
sorry

end conic_sections_l2_2735


namespace sum_of_a_b_c_l2_2457

theorem sum_of_a_b_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc1 : a * b + c = 47) (habc2 : b * c + a = 47) (habc3 : a * c + b = 47) : a + b + c = 48 := 
sorry

end sum_of_a_b_c_l2_2457


namespace car_drive_highway_distance_l2_2114

theorem car_drive_highway_distance
  (d_local : ℝ)
  (s_local : ℝ)
  (s_highway : ℝ)
  (s_avg : ℝ)
  (d_total := d_local + s_avg * (d_local / s_local + d_local / s_highway))
  (t_local := d_local / s_local)
  (t_highway : ℝ := (d_total - d_local) / s_highway)
  (t_total := t_local + t_highway)
  (avg_speed := (d_total) / t_total)
  : d_local = 60 → s_local = 20 → s_highway = 60 → s_avg = 36 → avg_speed = 36 → d_total - d_local = 120 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4]
  sorry

end car_drive_highway_distance_l2_2114


namespace scientific_notation_of_population_l2_2503

theorem scientific_notation_of_population :
  (141260 : ℝ) = 1.4126 * 10^5 :=
sorry

end scientific_notation_of_population_l2_2503


namespace leap_year_53_sundays_and_february_5_sundays_l2_2161

theorem leap_year_53_sundays_and_february_5_sundays :
  let Y := 366
  let W := 52
  ∃ (p : ℚ), p = (2/7) * (1/7) → p = 2/49
:=
by
  sorry

end leap_year_53_sundays_and_february_5_sundays_l2_2161


namespace inequality_abc_equality_condition_abc_l2_2350

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  (a / (2 * a + 1)) + (b / (3 * b + 1)) + (c / (6 * c + 1)) ≤ 1 / 2 :=
sorry

theorem equality_condition_abc (a b c : ℝ) :
  (a / (2 * a + 1)) + (b / (3 * b + 1)) + (c / (6 * c + 1)) = 1 / 2 ↔ 
  a = 1 / 2 ∧ b = 1 / 3 ∧ c = 1 / 6 :=
sorry

end inequality_abc_equality_condition_abc_l2_2350


namespace expression_simplification_l2_2912

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l2_2912


namespace Lee_charge_per_lawn_l2_2482

theorem Lee_charge_per_lawn
  (x : ℝ)
  (mowed_lawns : ℕ)
  (total_earned : ℝ)
  (tips : ℝ)
  (tip_amount : ℝ)
  (num_customers_tipped : ℕ)
  (earnings_from_mowing : ℝ)
  (total_earning_with_tips : ℝ) :
  mowed_lawns = 16 →
  total_earned = 558 →
  num_customers_tipped = 3 →
  tip_amount = 10 →
  tips = num_customers_tipped * tip_amount →
  earnings_from_mowing = mowed_lawns * x →
  total_earning_with_tips = earnings_from_mowing + tips →
  total_earning_with_tips = total_earned →
  x = 33 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end Lee_charge_per_lawn_l2_2482


namespace distance_to_asymptote_l2_2998

/-- Define the hyperbola equation as a predicate --/
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1

/-- Define the asymptote equations as predicates --/
def asymptote1 (x y : ℝ) : Prop := 3 * x - 4 * y = 0
def asymptote2 (x y : ℝ) : Prop := 3 * x + 4 * y = 0

/-- Define the distance formula from a point to a line --/
def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / (Real.sqrt (A^2 + B^2))

/-- Proof statement that the distance from (3,0) to asymptote is 9/5 --/
theorem distance_to_asymptote : 
  distance_from_point_to_line 3 0 3 (-4) 0 = 9 / 5 :=
by
  -- the main proof computation goes here
  sorry

end distance_to_asymptote_l2_2998


namespace kay_weight_training_time_l2_2785

variables (total_minutes : ℕ) (aerobic_ratio weight_ratio : ℕ)
-- Conditions
def kay_exercise := total_minutes = 250
def ratio_cond := aerobic_ratio = 3 ∧ weight_ratio = 2
def total_ratio_parts := aerobic_ratio + weight_ratio

-- Question and proof goal
theorem kay_weight_training_time (h1 : kay_exercise total_minutes) (h2 : ratio_cond aerobic_ratio weight_ratio) :
  (total_minutes / total_ratio_parts * weight_ratio) = 100 :=
by
  sorry

end kay_weight_training_time_l2_2785


namespace number_of_symmetric_subsets_l2_2612

def has_integer_solutions (m : ℤ) : Prop :=
  ∃ x y : ℤ, x * y = -36 ∧ x + y = -m

def M : Set ℤ :=
  {m | has_integer_solutions m}

def is_symmetric_subset (A : Set ℤ) : Prop :=
  A ⊆ M ∧ ∀ a ∈ A, -a ∈ A

theorem number_of_symmetric_subsets :
  (∃ A : Set ℤ, is_symmetric_subset A ∧ A ≠ ∅) →
  (∃ n : ℕ, n = 31) :=
by
  sorry

end number_of_symmetric_subsets_l2_2612


namespace combined_molecular_weight_l2_2080

theorem combined_molecular_weight {m1 m2 : ℕ} 
  (MW_C : ℝ) (MW_H : ℝ) (MW_O : ℝ)
  (Butanoic_acid : ℕ × ℕ × ℕ)
  (Propanoic_acid : ℕ × ℕ × ℕ)
  (MW_Butanoic_acid : ℝ)
  (MW_Propanoic_acid : ℝ)
  (weight_Butanoic_acid : ℝ)
  (weight_Propanoic_acid : ℝ)
  (total_weight : ℝ) :
MW_C = 12.01 → MW_H = 1.008 → MW_O = 16.00 →
Butanoic_acid = (4, 8, 2) → MW_Butanoic_acid = (4 * MW_C) + (8 * MW_H) + (2 * MW_O) →
Propanoic_acid = (3, 6, 2) → MW_Propanoic_acid = (3 * MW_C) + (6 * MW_H) + (2 * MW_O) →
m1 = 9 → weight_Butanoic_acid = m1 * MW_Butanoic_acid →
m2 = 5 → weight_Propanoic_acid = m2 * MW_Propanoic_acid →
total_weight = weight_Butanoic_acid + weight_Propanoic_acid →
total_weight = 1163.326 :=
by {
  intros;
  sorry
}

end combined_molecular_weight_l2_2080


namespace discount_difference_l2_2280

open Real

noncomputable def single_discount (B : ℝ) (d1 : ℝ) : ℝ :=
  B * (1 - d1)

noncomputable def successive_discounts (B : ℝ) (d2 : ℝ) (d3 : ℝ) : ℝ :=
  (B * (1 - d2)) * (1 - d3)

theorem discount_difference (B : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) :
  B = 12000 →
  d1 = 0.30 →
  d2 = 0.25 →
  d3 = 0.05 →
  abs (single_discount B d1 - successive_discounts B d2 d3) = 150 := by
  intros h_B h_d1 h_d2 h_d3
  rw [h_B, h_d1, h_d2, h_d3]
  rw [single_discount, successive_discounts]
  sorry

end discount_difference_l2_2280


namespace profit_is_55_l2_2518

-- Define the given conditions:
def cost_of_chocolates (bars: ℕ) (price_per_bar: ℕ) : ℕ :=
  bars * price_per_bar

def cost_of_packaging (bars: ℕ) (cost_per_bar: ℕ) : ℕ :=
  bars * cost_per_bar

def total_sales : ℕ :=
  90

def total_cost (cost_of_chocolates cost_of_packaging: ℕ) : ℕ :=
  cost_of_chocolates + cost_of_packaging

def profit (total_sales total_cost: ℕ) : ℕ :=
  total_sales - total_cost

-- Given values:
def bars: ℕ := 5
def price_per_bar: ℕ := 5
def cost_per_packaging_bar: ℕ := 2

-- Define the profit calculation theorem:
theorem profit_is_55 : 
  profit total_sales (total_cost (cost_of_chocolates bars price_per_bar) (cost_of_packaging bars cost_per_packaging_bar)) = 55 :=
by {
  -- The proof will be inserted here
  sorry
}

end profit_is_55_l2_2518


namespace train_length_l2_2571

theorem train_length 
  (bridge_length train_length time_seconds v : ℝ)
  (h1 : bridge_length = 300)
  (h2 : time_seconds = 36)
  (h3 : v = 40) :
  (train_length = v * time_seconds - bridge_length) →
  (train_length = 1140) := by
  -- solve in a few lines
  -- This proof is omitted for the purpose of this task
  sorry

end train_length_l2_2571


namespace trigonometric_expression_l2_2168

noncomputable def cosθ (θ : ℝ) := 1 / Real.sqrt 10
noncomputable def sinθ (θ : ℝ) := 3 / Real.sqrt 10
noncomputable def tanθ (θ : ℝ) := 3

theorem trigonometric_expression (θ : ℝ) (h : tanθ θ = 3) :
  (1 + cosθ θ) / sinθ θ + sinθ θ / (1 - cosθ θ) = (10 * Real.sqrt 10 + 10) / 9 := 
  sorry

end trigonometric_expression_l2_2168


namespace exists_multiple_with_odd_digit_sum_l2_2308

theorem exists_multiple_with_odd_digit_sum (M : Nat) :
  ∃ N : Nat, N % M = 0 ∧ (Nat.digits 10 N).sum % 2 = 1 :=
by
  sorry

end exists_multiple_with_odd_digit_sum_l2_2308


namespace scientific_notation_of_population_l2_2505

theorem scientific_notation_of_population :
  (141260 : ℝ) = 1.4126 * 10^5 :=
sorry

end scientific_notation_of_population_l2_2505


namespace seq_general_term_l2_2602

-- Define the sequence according to the given conditions
def seq (a : ℕ+ → ℚ) : Prop := 
  a 1 = 1 ∧ ∀ n : ℕ+, a (n + 1) = n / (n + 1 : ℕ) * a n

-- The theorem statement: proving the general term
theorem seq_general_term (a : ℕ+ → ℚ) (h : seq a) : ∀ n : ℕ+, a n = 1 / n :=
by {
  sorry
}

end seq_general_term_l2_2602


namespace num_workers_l2_2558

-- Define the number of workers (n) and the initial contribution per worker (x)
variable (n x : ℕ)

-- Condition 1: The total contribution is Rs. 3 lacs
axiom h1 : n * x = 300000

-- Condition 2: If each worker contributed Rs. 50 more, the total would be Rs. 3.75 lacs
axiom h2 : n * (x + 50) = 375000

-- Proof Problem: Prove that the number of workers (n) is 1500
theorem num_workers : n = 1500 :=
by
  -- The proof will go here
  sorry

end num_workers_l2_2558


namespace inscribed_circle_radius_l2_2097

theorem inscribed_circle_radius (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 30) : 
  ∃ r : ℝ, r = (105 * Real.sqrt 274) / 274 := 
by 
  sorry

end inscribed_circle_radius_l2_2097


namespace alpha_beta_value_l2_2446

theorem alpha_beta_value :
  ∃ α β : ℝ, (α^2 - 2 * α - 4 = 0) ∧ (β^2 - 2 * β - 4 = 0) ∧ (α + β = 2) ∧ (α^3 + 8 * β + 6 = 30) :=
by
  sorry

end alpha_beta_value_l2_2446


namespace initial_bananas_l2_2709

theorem initial_bananas (bananas_left: ℕ) (eaten: ℕ) (basket: ℕ) 
                        (h_left: bananas_left = 100) 
                        (h_eaten: eaten = 70) 
                        (h_basket: basket = 2 * eaten): 
  bananas_left + eaten + basket = 310 :=
by
  sorry

end initial_bananas_l2_2709


namespace opposite_of_three_l2_2854

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l2_2854


namespace opposite_of_3_is_neg3_l2_2821

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2821


namespace ordered_triples_count_l2_2453

open Nat

theorem ordered_triples_count :
  ∃ (count : ℕ), (count = 10) ∧
  (∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 →
   lcm x y = 180 → lcm x z = 800 → lcm y z = 1200 →
   count = 10) :=
sorry

end ordered_triples_count_l2_2453


namespace math_problem_l2_2185

variable (x y : ℝ)

theorem math_problem (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by sorry

end math_problem_l2_2185


namespace sum_of_dihedral_angles_leq_90_l2_2315
noncomputable section

-- Let θ1 and θ2 be angles formed by a line with two perpendicular planes
variable (θ1 θ2 : ℝ)

-- Define the condition stating the planes are perpendicular, and the line forms dihedral angles
def dihedral_angle_condition (θ1 θ2 : ℝ) : Prop := 
  θ1 ≥ 0 ∧ θ1 ≤ 90 ∧ θ2 ≥ 0 ∧ θ2 ≤ 90

-- The theorem statement capturing the problem
theorem sum_of_dihedral_angles_leq_90 
  (θ1 θ2 : ℝ) 
  (h : dihedral_angle_condition θ1 θ2) : 
  θ1 + θ2 ≤ 90 :=
sorry

end sum_of_dihedral_angles_leq_90_l2_2315


namespace math_books_count_l2_2392

theorem math_books_count (M H : ℕ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 396) : M = 54 :=
sorry

end math_books_count_l2_2392


namespace angle_C_in_triangle_l2_2470

theorem angle_C_in_triangle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A + B = 115) : C = 65 := 
by 
  sorry

end angle_C_in_triangle_l2_2470


namespace asymptotes_of_hyperbola_l2_2754

variable (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
variable (h3 : (1 + b^2 / a^2) = (6 / 4))

theorem asymptotes_of_hyperbola :
  ∃ (m : ℝ), m = b / a ∧ (m = Real.sqrt 2 / 2) ∧ ∀ x : ℝ, (y = m*x) ∨ (y = -m*x) :=
by
  sorry

end asymptotes_of_hyperbola_l2_2754


namespace percentage_of_cash_is_20_l2_2477

theorem percentage_of_cash_is_20
  (raw_materials : ℕ)
  (machinery : ℕ)
  (total_amount : ℕ)
  (h_raw_materials : raw_materials = 35000)
  (h_machinery : machinery = 40000)
  (h_total_amount : total_amount = 93750) :
  (total_amount - (raw_materials + machinery)) * 100 / total_amount = 20 :=
by
  sorry

end percentage_of_cash_is_20_l2_2477


namespace simplify_expression_l2_2904

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l2_2904


namespace prime_pairs_l2_2933

theorem prime_pairs (p q : ℕ) : 
  p < 2005 → q < 2005 → 
  Prime p → Prime q → 
  (q ∣ p^2 + 4) → 
  (p ∣ q^2 + 4) → 
  (p = 2 ∧ q = 2) :=
by sorry

end prime_pairs_l2_2933


namespace fourth_arithmetic_sequence_equation_l2_2948

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables (a : ℕ → ℝ) (h : is_arithmetic_sequence a)
variable (h1 : a 1 - 2 * a 2 + a 3 = 0)
variable (h2 : a 1 - 3 * a 2 + 3 * a 3 - a 4 = 0)
variable (h3 : a 1 - 4 * a 2 + 6 * a 3 - 4 * a 4 + a 5 = 0)

-- Theorem statement to be proven
theorem fourth_arithmetic_sequence_equation : a 1 - 5 * a 2 + 10 * a 3 - 10 * a 4 + 5 * a 5 - a 6 = 0 :=
by
  sorry

end fourth_arithmetic_sequence_equation_l2_2948


namespace all_rationals_as_sum_of_q_n_l2_2991

theorem all_rationals_as_sum_of_q_n :
  ∀ (r : ℚ), ∃ (n : ℕ) (a : ℕ → ℚ), (∀ i, a i = (i - 1) / (i + 2) ∧ n ≥ 0) ∧ 
            r = ∑ i in Finset.range n, a i :=
by
  sorry

end all_rationals_as_sum_of_q_n_l2_2991


namespace students_calculation_l2_2344

variable (students_boys students_playing_soccer students_not_playing_soccer girls_not_playing_soccer : ℕ)
variable (percentage_boys_play_soccer : ℚ)

def students_not_playing_sum (students_boys_not_playing : ℕ) : ℕ :=
  students_boys_not_playing + girls_not_playing_soccer

def total_students (students_not_playing_sum students_playing_soccer : ℕ) : ℕ :=
  students_not_playing_sum + students_playing_soccer

theorem students_calculation 
  (H1 : students_boys = 312)
  (H2 : students_playing_soccer = 250)
  (H3 : percentage_boys_play_soccer = 0.86)
  (H4 : girls_not_playing_soccer = 73)
  (H5 : percentage_boys_play_soccer * students_playing_soccer = 215)
  (H6 : students_boys - 215 = 97)
  (H7 : students_not_playing_sum 97 = 170)
  (H8 : total_students 170 250 = 420) : ∃ total, total = 420 :=
by 
  existsi total_students 170 250
  exact H8

end students_calculation_l2_2344


namespace percent_calculation_l2_2619

theorem percent_calculation (x : ℝ) (h : 0.40 * x = 160) : 0.30 * x = 120 :=
by
  sorry

end percent_calculation_l2_2619


namespace simplify_expression_eq_square_l2_2898

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l2_2898


namespace Ryan_dig_time_alone_l2_2554

theorem Ryan_dig_time_alone :
  ∃ R : ℝ, ∀ Castel_time together_time,
    Castel_time = 6 ∧ together_time = 30 / 11 →
    (1 / R + 1 / Castel_time = 11 / 30) →
    R = 5 :=
by 
  sorry

end Ryan_dig_time_alone_l2_2554


namespace card_area_after_shortening_l2_2156

/-- Given a card with dimensions 3 inches by 7 inches, prove that 
  if the length is shortened by 1 inch and the width is shortened by 2 inches, 
  then the resulting area is 10 square inches. -/
theorem card_area_after_shortening :
  let length := 3
  let width := 7
  let new_length := length - 1
  let new_width := width - 2
  new_length * new_width = 10 :=
by
  let length := 3
  let width := 7
  let new_length := length - 1
  let new_width := width - 2
  show new_length * new_width = 10
  sorry

end card_area_after_shortening_l2_2156


namespace find_locus_of_P_l2_2598

theorem find_locus_of_P:
  ∃ x y: ℝ, (x - 1)^2 + y^2 = 9 ∧ y ≠ 0 ∧
          ((x + 2)^2 + y^2 + (x - 4)^2 + y^2 = 36) :=
sorry

end find_locus_of_P_l2_2598


namespace scientific_notation_conversion_l2_2495

theorem scientific_notation_conversion (x : ℝ) (h_population : x = 141260000) :
  x = 1.4126 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l2_2495


namespace angle_sum_property_l2_2343

theorem angle_sum_property 
  (P Q R S : Type) 
  (alpha beta : ℝ)
  (h1 : alpha = 3 * x)
  (h2 : beta = 2 * x)
  (h3 : alpha + beta = 90) :
  x = 18 :=
by
  sorry

end angle_sum_property_l2_2343


namespace value_of_a_minus_b_l2_2447

theorem value_of_a_minus_b
  (a b : ℝ)
  (h1 : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1 / 2 < x ∧ x < 1 / 3) :
  a - b = -10 :=
by sorry

end value_of_a_minus_b_l2_2447


namespace neg_exists_le_eq_forall_gt_l2_2762

open Classical

variable {n : ℕ}

theorem neg_exists_le_eq_forall_gt :
  (¬ ∃ (n : ℕ), n > 0 ∧ 2^n ≤ 2 * n + 1) ↔
  (∀ (n : ℕ), n > 0 → 2^n > 2 * n + 1) :=
by 
  sorry

end neg_exists_le_eq_forall_gt_l2_2762


namespace expression_simplification_l2_2910

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l2_2910


namespace kevin_total_distance_l2_2037

def v1 : ℝ := 10
def t1 : ℝ := 0.5
def v2 : ℝ := 20
def t2 : ℝ := 0.5
def v3 : ℝ := 8
def t3 : ℝ := 0.25

theorem kevin_total_distance : v1 * t1 + v2 * t2 + v3 * t3 = 17 := by
  sorry

end kevin_total_distance_l2_2037


namespace min_value_of_expression_l2_2943

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) : x^2 + (1 / 4) * y^2 ≥ 1 / 8 :=
sorry

end min_value_of_expression_l2_2943


namespace tangent_line_slope_l2_2025

theorem tangent_line_slope (x₀ y₀ k : ℝ)
    (h_tangent_point : y₀ = x₀ + Real.exp (-x₀))
    (h_tangent_line : y₀ = k * x₀) :
    k = 1 - Real.exp 1 := 
sorry

end tangent_line_slope_l2_2025


namespace angle_bisectors_meet_on_segment_l2_2789

theorem angle_bisectors_meet_on_segment 
  (A B C D : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB AC BC : ℝ)  -- sides of the triangle
  (ABD_angle ACD_angle : ℝ)  -- angles at ABD and ACD
  (h_AB_AC : AB = AC) 
  (h_AB_AC_not : AB ≠ BC)
  (h_angle_ABD : ABD_angle = 30)
  (h_angle_ACD : ACD_angle = 30) :
  ∃ P : Type, IsAngleBisector P (Angle A C B) (Angle A D B) (Segment A B) :=
sorry

end angle_bisectors_meet_on_segment_l2_2789


namespace opposite_of_3_is_neg3_l2_2828

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2828


namespace alt_fib_factorial_seq_last_two_digits_eq_85_l2_2418

noncomputable def alt_fib_factorial_seq_last_two_digits : ℕ :=
  let f0 := 1   -- 0!
  let f1 := 1   -- 1!
  let f2 := 2   -- 2!
  let f3 := 6   -- 3!
  let f5 := 120 -- 5! (last two digits 20)
  (f0 - f1 + f1 - f2 + f3 - (f5 % 100)) % 100

theorem alt_fib_factorial_seq_last_two_digits_eq_85 :
  alt_fib_factorial_seq_last_two_digits = 85 :=
by 
  sorry

end alt_fib_factorial_seq_last_two_digits_eq_85_l2_2418


namespace distance_between_parallel_lines_l2_2016

theorem distance_between_parallel_lines :
  let a := 4
  let b := -3
  let c1 := 2
  let c2 := -1
  let d := (abs (c1 - c2)) / (Real.sqrt (a^2 + b^2))
  d = 3 / 5 :=
by
  sorry

end distance_between_parallel_lines_l2_2016


namespace value_2_std_devs_below_mean_l2_2259

theorem value_2_std_devs_below_mean {μ σ : ℝ} (h_mean : μ = 10.5) (h_std_dev : σ = 1) : μ - 2 * σ = 8.5 :=
by
  sorry

end value_2_std_devs_below_mean_l2_2259


namespace first_year_fee_correct_l2_2133

noncomputable def first_year_fee (n : ℕ) (annual_increase : ℕ) (sixth_year_fee : ℕ) : ℕ :=
  sixth_year_fee - (n - 1) * annual_increase

theorem first_year_fee_correct (n annual_increase sixth_year_fee value : ℕ) 
  (h_n : n = 6) (h_annual_increase : annual_increase = 10) 
  (h_sixth_year_fee : sixth_year_fee = 130) (h_value : value = 80) :
  first_year_fee n annual_increase sixth_year_fee = value :=
by {
  sorry
}

end first_year_fee_correct_l2_2133


namespace first_place_team_ties_l2_2027

noncomputable def teamPoints (wins ties: ℕ) : ℕ := 2 * wins + ties

theorem first_place_team_ties {T : ℕ} : 
  teamPoints 13 1 + teamPoints 8 10 + teamPoints 12 T = 81 → T = 4 :=
by
  sorry

end first_place_team_ties_l2_2027


namespace maximum_value_problem_l2_2261

theorem maximum_value_problem (x : ℝ) (h : 0 < x ∧ x < 4/3) : ∃ M, M = (4 / 3) ∧ ∀ y, 0 < y ∧ y < 4/3 → x * (4 - 3 * x) ≤ M :=
sorry

end maximum_value_problem_l2_2261


namespace ned_time_left_to_diffuse_bomb_l2_2208

-- Conditions
def building_flights : Nat := 20
def time_per_flight : Nat := 11
def bomb_timer : Nat := 72
def time_spent_running : Nat := 165

-- Main statement
theorem ned_time_left_to_diffuse_bomb : 
  (bomb_timer - (building_flights - (time_spent_running / time_per_flight)) * time_per_flight) = 17 :=
by
  sorry

end ned_time_left_to_diffuse_bomb_l2_2208


namespace part1_l2_2706

theorem part1 (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^2005 + z^2006 + z^2008 + z^2009 = -2 :=
  sorry

end part1_l2_2706


namespace integer_roots_if_q_positive_no_integer_roots_if_q_negative_l2_2414

theorem integer_roots_if_q_positive (p q : ℤ) (hq : q > 0) :
  (∃ x1 x2 : ℤ, x1 * x2 = q ∧ x1 + x2 = p) ∧
  (∃ y1 y2 : ℤ, y1 * y2 = q ∧ y1 + y2 = p + 1) :=
sorry

theorem no_integer_roots_if_q_negative (p q : ℤ) (hq : q < 0) :
  ¬ ((∃ x1 x2 : ℤ, x1 * x2 = q ∧ x1 + x2 = p) ∧
  (∃ y1 y2 : ℤ, y1 * y2 = q ∧ y1 + y2 = p + 1)) :=
sorry

end integer_roots_if_q_positive_no_integer_roots_if_q_negative_l2_2414


namespace find_f_100_l2_2668

theorem find_f_100 (f : ℝ → ℝ) (k : ℝ) (h_nonzero : k ≠ 0) 
(h_func : ∀ x y : ℝ, 0 < x → 0 < y → k * (x * f y - y * f x) = f (x / y)) : 
f 100 = 0 := 
by
  sorry

end find_f_100_l2_2668


namespace sum_of_squares_of_rates_l2_2422

variable (b j s : ℤ) -- rates in km/h
-- conditions
def ed_condition : Prop := 3 * b + 4 * j + 2 * s = 86
def sue_condition : Prop := 5 * b + 2 * j + 4 * s = 110

theorem sum_of_squares_of_rates (b j s : ℤ) (hEd : ed_condition b j s) (hSue : sue_condition b j s) : 
  b^2 + j^2 + s^2 = 3349 := 
sorry

end sum_of_squares_of_rates_l2_2422


namespace smallest_four_digit_multiple_of_17_is_1013_l2_2378

-- Lean definition to state the problem
def smallest_four_digit_multiple_of_17 : ℕ :=
  1013

-- Main Lean theorem to assert the correctness
theorem smallest_four_digit_multiple_of_17_is_1013 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = smallest_four_digit_multiple_of_17 :=
by
  -- proof here
  sorry

end smallest_four_digit_multiple_of_17_is_1013_l2_2378


namespace isosceles_triangle_base_angle_l2_2029

theorem isosceles_triangle_base_angle (x : ℝ) 
  (h1 : ∀ (a b : ℝ), a + b + (20 + 2 * b) = 180)
  (h2 : 20 + 2 * x = 180 - 2 * x - x) : x = 40 :=
by sorry

end isosceles_triangle_base_angle_l2_2029


namespace no_nat_solutions_no_int_solutions_l2_2757

theorem no_nat_solutions (x y : ℕ) : x^3 + 5 * y = y^3 + 5 * x → x = y :=
by sorry

theorem no_int_solutions (x y : ℤ) : x^3 + 5 * y = y^3 + 5 * x → x = y :=
by sorry

end no_nat_solutions_no_int_solutions_l2_2757


namespace a_perp_a_minus_b_l2_2614

noncomputable def a : ℝ × ℝ := (-2, 1)
noncomputable def b : ℝ × ℝ := (-1, 3)
noncomputable def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

theorem a_perp_a_minus_b : (a.1 * a_minus_b.1 + a.2 * a_minus_b.2) = 0 := by
  sorry

end a_perp_a_minus_b_l2_2614


namespace smallest_y_l2_2269

noncomputable def x : ℕ := 3 * 40 * 75

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ (k : ℕ), k^3 = n

theorem smallest_y (y : ℕ) (hy : y = 3) :
  ∀ (x : ℕ), x = 3 * 40 * 75 → is_perfect_cube (x * y) :=
by
  intro x hx
  unfold is_perfect_cube
  exists 5 -- This is just a placeholder value; the proof would find the correct k
  sorry

end smallest_y_l2_2269


namespace pictures_per_album_l2_2692

theorem pictures_per_album (phone_pics camera_pics albums : ℕ) (h_phone : phone_pics = 22) (h_camera : camera_pics = 2) (h_albums : albums = 4) (h_total_pics : phone_pics + camera_pics = 24) : (phone_pics + camera_pics) / albums = 6 :=
by
  sorry

end pictures_per_album_l2_2692


namespace a8_value_l2_2003

def sequence_sum (n : ℕ) : ℕ := 2^n - 1

def nth_term (S : ℕ → ℕ) (n : ℕ) : ℕ :=
  S n - S (n - 1)

theorem a8_value : nth_term sequence_sum 8 = 128 :=
by
  -- Proof goes here
  sorry

end a8_value_l2_2003


namespace truck_speed_kmph_l2_2404

theorem truck_speed_kmph (d : ℕ) (t : ℕ) (km_m : ℕ) (hr_s : ℕ) 
  (h1 : d = 600) (h2 : t = 20) (h3 : km_m = 1000) (h4 : hr_s = 3600) : 
  (d / t) * (hr_s / km_m) = 108 := by
  sorry

end truck_speed_kmph_l2_2404


namespace next_in_sequence_is_80_l2_2365

def seq (n : ℕ) : ℕ := n^2 - 1

theorem next_in_sequence_is_80 :
  seq 9 = 80 :=
by
  sorry

end next_in_sequence_is_80_l2_2365


namespace freight_cost_minimization_l2_2072

-- Define the main parameters: tonnage and costs for the trucks.
def freight_cost (num_seven_ton_trucks : ℕ) (num_five_ton_trucks : ℕ) : ℕ :=
  65 * num_seven_ton_trucks + 50 * num_five_ton_trucks

-- Define the total transported capacity by the two types of trucks.
def total_capacity (num_seven_ton_trucks : ℕ) (num_five_ton_trucks : ℕ) : ℕ :=
  7 * num_seven_ton_trucks + 5 * num_five_ton_trucks

-- Define the minimum freight cost given the conditions.
def minimum_freight_cost := 685

-- The theorem we want to prove.
theorem freight_cost_minimization : ∃ x y : ℕ, total_capacity x y ≥ 73 ∧
  (freight_cost x y = minimum_freight_cost) :=
by
  sorry

end freight_cost_minimization_l2_2072


namespace infinite_solutions_eq_l2_2655

/-
Proving that the equation x - y + z = 1 has infinite solutions under the conditions:
1. x, y, z are distinct positive integers.
2. The product of any two numbers is divisible by the third one.
-/
theorem infinite_solutions_eq (x y z : ℕ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) 
(h4 : ∃ m n k : ℕ, x = m * n ∧ y = n * k ∧ z = m * k)
(h5 : (x*y) % z = 0) (h6 : (y*z) % x = 0) (h7 : (z*x) % y = 0) : 
∃ (m : ℕ), x - y + z = 1 ∧ x > 0 ∧ y > 0 ∧ z > 0 :=
by sorry

end infinite_solutions_eq_l2_2655


namespace length_of_platform_l2_2697

-- Definitions for conditions
def train_length : ℕ := 300
def time_cross_platform : ℕ := 39
def time_cross_signal : ℕ := 12

-- Speed calculation
def train_speed := train_length / time_cross_signal

-- Total distance calculation while crossing the platform
def total_distance := train_speed * time_cross_platform

-- Length of the platform
def platform_length : ℕ := total_distance - train_length

-- Theorem stating the length of the platform
theorem length_of_platform :
  platform_length = 675 := by
  sorry

end length_of_platform_l2_2697


namespace increased_consumption_5_percent_l2_2371

theorem increased_consumption_5_percent (T C : ℕ) (h1 : ¬ (T = 0)) (h2 : ¬ (C = 0)) :
  (0.80 * (1 + x/100) = 0.84) → (x = 5) :=
by
  sorry

end increased_consumption_5_percent_l2_2371


namespace sum_of_next_five_even_integers_l2_2257

theorem sum_of_next_five_even_integers (a : ℕ) (x : ℕ) 
  (h : a = x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) : 
  (x + 10) + (x + 12) + (x + 14) + (x + 16) + (x + 18) = a + 50 := by
  sorry

end sum_of_next_five_even_integers_l2_2257


namespace g_expression_f_expression_l2_2310

-- Given functions f and g that satisfy the conditions
variable {f g : ℝ → ℝ}

-- Conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom sum_eq : ∀ x, f x + g x = 2^x + 2 * x

-- Theorem statements to prove
theorem g_expression : g = fun x => 2^x := by sorry
theorem f_expression : f = fun x => 2 * x := by sorry

end g_expression_f_expression_l2_2310


namespace number_of_ways_to_choose_subsets_l2_2887

open Finset

def setT : Finset ℕ := {0, 1, 2, 3, 4, 5}

def valid_subsets (T : Finset ℕ) (A B : Finset ℕ) : Prop :=
(A ∪ B = T) ∧ (A ∩ B).card = 3

theorem number_of_ways_to_choose_subsets :
  let S := setT in ∃ (n : ℕ), n = 80 :=
by
  sorry

end number_of_ways_to_choose_subsets_l2_2887


namespace min_cells_marked_l2_2552

/-- The minimum number of cells that need to be marked in a 50x50 grid so
each 1x6 vertical or horizontal strip has at least one marked cell is 416. -/
theorem min_cells_marked {n : ℕ} : n = 416 → 
  (∀ grid : Fin 50 × Fin 50, ∃ cells : Finset (Fin 50 × Fin 50), 
    (∀ (r c : Fin 50), (r = 6 * i + k ∨ c = 6 * i + k) →
      (∃ (cell : Fin 50 × Fin 50), cell ∈ cells)) →
    cells.card = n) := 
sorry

end min_cells_marked_l2_2552


namespace billboard_dimensions_l2_2126

theorem billboard_dimensions (photo_width_cm : ℕ) (photo_length_dm : ℕ) (billboard_area_m2 : ℕ)
  (h1 : photo_width_cm = 30) (h2 : photo_length_dm = 4) (h3 : billboard_area_m2 = 48) :
  ∃ photo_length_cm : ℕ, photo_length_cm = 40 ∧
  ∃ k : ℕ, k = 20 ∧
  ∃ billboard_width_m billboard_length_m : ℕ,
    billboard_width_m = photo_width_cm * k / 100 ∧ 
    billboard_length_m = photo_length_cm * k / 100 ∧ 
    billboard_width_m = 6 ∧ 
    billboard_length_m = 8 := by
  sorry

end billboard_dimensions_l2_2126


namespace carter_lucy_ratio_l2_2147

-- Define the number of pages Oliver can read in 1 hour
def oliver_pages : ℕ := 40

-- Define the number of additional pages Lucy can read compared to Oliver
def additional_pages : ℕ := 20

-- Define the number of pages Carter can read in 1 hour
def carter_pages : ℕ := 30

-- Calculate the number of pages Lucy can read in 1 hour
def lucy_pages : ℕ := oliver_pages + additional_pages

-- Prove the ratio of the number of pages Carter can read to the number of pages Lucy can read is 1/2
theorem carter_lucy_ratio : (carter_pages : ℚ) / (lucy_pages : ℚ) = 1 / 2 := by
  sorry

end carter_lucy_ratio_l2_2147


namespace sum_of_possible_values_l2_2794

theorem sum_of_possible_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 2) :
  (x - 2) * (y - 2) = 6 ∨ (x - 2) * (y - 2) = 9 →
  (if (x - 2) * (y - 2) = 6 then 6 else 0) + (if (x - 2) * (y - 2) = 9 then 9 else 0) = 15 :=
by
  sorry

end sum_of_possible_values_l2_2794


namespace Harriet_siblings_product_l2_2955

variable (Harry_sisters : Nat)
variable (Harry_brothers : Nat)
variable (Harriet_sisters : Nat)
variable (Harriet_brothers : Nat)

theorem Harriet_siblings_product:
  Harry_sisters = 4 -> 
  Harry_brothers = 6 ->
  Harriet_sisters = Harry_sisters -> 
  Harriet_brothers = Harry_brothers ->
  Harriet_sisters * Harriet_brothers = 24 :=
by
  intro hs hb hhs hhb
  rw [hhs, hhb]
  sorry

end Harriet_siblings_product_l2_2955


namespace arithmetic_sum_l2_2944

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n * d)

def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sum :
  ∀ (a d : ℕ),
  arithmetic_sequence a d 2 + arithmetic_sequence a d 3 + arithmetic_sequence a d 4 = 12 →
  sum_first_n_terms a d 7 = 28 :=
by
  sorry

end arithmetic_sum_l2_2944


namespace sequence_ratio_l2_2753

variable (a : ℕ → ℝ) -- Define the sequence a_n
variable (q : ℝ) (h_q : q > 0) -- q is the common ratio and it is positive

-- Define the conditions
axiom geom_seq_pos : ∀ n : ℕ, 0 < a n
axiom geom_seq_def : ∀ n : ℕ, a (n + 1) = q * a n
axiom arith_seq_def : 2 * (1 / 2 * a 3) = 3 * a 1 + 2 * a 2

theorem sequence_ratio : (a 11 + a 13) / (a 8 + a 10) = 27 := 
by
  sorry

end sequence_ratio_l2_2753


namespace eggs_per_chicken_per_day_l2_2478

-- Define the conditions
def chickens : ℕ := 8
def price_per_dozen : ℕ := 5
def total_revenue : ℕ := 280
def weeks : ℕ := 4
def eggs_per_dozen : ℕ := 12
def days_per_week : ℕ := 7

-- Theorem statement on how many eggs each chicken lays per day
theorem eggs_per_chicken_per_day :
  (chickens * ((total_revenue / price_per_dozen * eggs_per_dozen) / (weeks * days_per_week))) / chickens = 3 :=
by
  sorry

end eggs_per_chicken_per_day_l2_2478


namespace scientific_notation_population_l2_2502

theorem scientific_notation_population :
    ∃ (a b : ℝ), (b = 5 ∧ 1412.60 * 10 ^ 6 = a * 10 ^ b ∧ a = 1.4126) :=
sorry

end scientific_notation_population_l2_2502


namespace smaller_third_angle_l2_2690

theorem smaller_third_angle (x y : ℕ) (h₁ : x = 64) 
  (h₂ : 2 * x + (x - y) = 180) : y = 12 :=
by
  sorry

end smaller_third_angle_l2_2690


namespace expand_and_simplify_l2_2233

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l2_2233


namespace max_value_frac_inv_sum_l2_2980

theorem max_value_frac_inv_sum (x y : ℝ) (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b)
  (h3 : a^x = 6) (h4 : b^y = 6) (h5 : a + b = 2 * Real.sqrt 6) :
  ∃ m, m = 1 ∧ (∀ x y a b, (1 < a) → (1 < b) → (a^x = 6) → (b^y = 6) → (a + b = 2 * Real.sqrt 6) → 
  (∃ n, (n = (1/x + 1/y)) → n ≤ m)) :=
by
  sorry

end max_value_frac_inv_sum_l2_2980


namespace opposite_of_three_l2_2812

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l2_2812


namespace power_eq_45_l2_2167

theorem power_eq_45 (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 5) : a^(2*m + n) = 45 := by
  sorry

end power_eq_45_l2_2167


namespace similar_triangles_proportionalities_l2_2594

-- Definitions of the conditions as hypotheses
variables (A B C D E F : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (triangle_ABC_sim_triangle_DEF : SimilarTriangles A B C D E F)
variables (AB_DE_ratio : AB / DE = 1 / 2)
variables (BC_length : BC = 2)

-- Defining the hypothesis of similarity
def SimilarTriangles (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Prop :=
  ∀ (AB BC CA DE EF FD : ℝ), (AB / DE = BC / EF) ∧ (BC / EF = CA / FD) ∧ (CA / FD = AB / DE)

-- The proof statement
theorem similar_triangles_proportionalities (A B C D E F : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (triangle_ABC_sim_triangle_DEF : SimilarTriangles A B C D E F)
  (AB_DE_ratio : AB / DE = 1 / 2)
  (BC_length : BC = 2) : 
  EF = 4 := 
by sorry

end similar_triangles_proportionalities_l2_2594


namespace problem_statement_l2_2984

def setA : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def setB : Set ℝ := {x : ℝ | x ≥ 2}
def setC (a : ℝ) : Set ℝ := {x : ℝ | 2 * x + a ≥ 0}

theorem problem_statement (a : ℝ):
  (setA ∩ setB = {x : ℝ | 2 ≤ x ∧ x < 3}) ∧ 
  (setA ∪ setB = {x : ℝ | x ≥ -1}) ∧ 
  (setB ⊆ setC a → a > -4) :=
by
  sorry

end problem_statement_l2_2984


namespace relationship_among_a_b_c_l2_2000

noncomputable def a : ℝ := (1 / 3) ^ 3
noncomputable def b (x : ℝ) : ℝ := x ^ 3
noncomputable def c (x : ℝ) : ℝ := Real.log x

theorem relationship_among_a_b_c (x : ℝ) (h : x > 2) : a < c x ∧ c x < b x :=
by {
  -- proof steps are skipped
  sorry
}

end relationship_among_a_b_c_l2_2000


namespace t_is_perfect_square_l2_2435

variable (n : ℕ) (hpos : 0 < n)
variable (t : ℕ) (ht : t = 2 + 2 * Nat.sqrt (1 + 12 * n^2))

theorem t_is_perfect_square (n : ℕ) (hpos : 0 < n) (t : ℕ) (ht : t = 2 + 2 * Nat.sqrt (1 + 12 * n^2)) : 
  ∃ k : ℕ, t = k * k := 
sorry

end t_is_perfect_square_l2_2435


namespace number_of_permutations_l2_2487

theorem number_of_permutations (n : ℕ) (h : n > 1) :
  (∃ p : (list ℕ) → Prop, 
     (∀ (a : list ℕ), list.perm (list.range n) a → p a → 
        ∃ i, i ∈ list.range (n - 1) ∧ a.nth_le i sorry > a.nth_le (i + 1) sorry) ∧
      (∀ i, i ∈ list.range (n - 1) → 
        ∃ a : list ℕ, list.perm (list.range n) a ∧ p a ∧ a.nth_le i sorry > a.nth_le (i + 1) sorry) ∧
      ∀ j, j ∈ list.range (n - 1) → 
        ∃! (a : list ℕ), list.perm (list.range n) a ∧ p a ∧ a.nth_le j sorry > a.nth_le (j + 1) sorry
  ) :=
  2^n - n - 1 := sorry

end number_of_permutations_l2_2487


namespace g_crosses_horizontal_asymptote_at_minus_four_l2_2166

noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 8 * x - 10) / (x^2 - 5 * x + 6)

theorem g_crosses_horizontal_asymptote_at_minus_four : g (-4) = 3 := 
by
  sorry

end g_crosses_horizontal_asymptote_at_minus_four_l2_2166


namespace five_p_squared_plus_two_q_squared_odd_p_squared_plus_pq_plus_q_squared_odd_l2_2312

variable (p q : ℕ)
variable (hp : p % 2 = 1)  -- p is odd
variable (hq : q % 2 = 1)  -- q is odd

theorem five_p_squared_plus_two_q_squared_odd 
    (hp : p % 2 = 1) 
    (hq : q % 2 = 1) : 
    (5 * p^2 + 2 * q^2) % 2 = 1 := 
sorry

theorem p_squared_plus_pq_plus_q_squared_odd 
    (hp : p % 2 = 1) 
    (hq : q % 2 = 1) : 
    (p^2 + p * q + q^2) % 2 = 1 := 
sorry

end five_p_squared_plus_two_q_squared_odd_p_squared_plus_pq_plus_q_squared_odd_l2_2312


namespace lunch_break_duration_l2_2218

theorem lunch_break_duration :
  ∃ (L : ℝ), 
    (∃ (p a : ℝ),
      (6 - L) * (p + a) = 0.4 ∧
      (4 - L) * a = 0.15 ∧
      (10 - L) * p = 0.45) ∧
    291 = L * 60 := 
by
  sorry

end lunch_break_duration_l2_2218


namespace checkerboard_probability_l2_2214

def total_squares (n : ℕ) : ℕ :=
  n * n

def perimeter_squares (n : ℕ) : ℕ :=
  4 * n - 4

def non_perimeter_squares (n : ℕ) : ℕ :=
  total_squares n - perimeter_squares n

def probability_non_perimeter_square (n : ℕ) : ℚ :=
  non_perimeter_squares n / total_squares n

theorem checkerboard_probability :
  probability_non_perimeter_square 10 = 16 / 25 :=
by
  sorry

end checkerboard_probability_l2_2214


namespace opposite_of_three_l2_2843

theorem opposite_of_three : -3 = opposite(3) := 
by
  sorry

end opposite_of_three_l2_2843


namespace katie_miles_l2_2788

theorem katie_miles (x : ℕ) (h1 : ∀ y, y = 3 * x → y ≤ 240) (h2 : x + 3 * x = 240) : x = 60 :=
sorry

end katie_miles_l2_2788


namespace find_v4_l2_2730

noncomputable def horner_method (x : ℤ) : ℤ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 20
  let v4 := v3 * x - 8
  v4

theorem find_v4 : horner_method (-2) = -16 :=
  by {
    -- Proof goes here, but we are only required to write the statement.
    sorry
  }

end find_v4_l2_2730


namespace ned_defuse_time_l2_2210

theorem ned_defuse_time (flights_total time_per_flight bomb_time time_spent : ℕ) (h1 : flights_total = 20) (h2 : time_per_flight = 11) (h3 : bomb_time = 72) (h4 : time_spent = 165) :
  bomb_time - (flights_total * time_per_flight - time_spent) / time_per_flight * time_per_flight = 17 := by
  sorry

end ned_defuse_time_l2_2210


namespace curve_properties_l2_2743

noncomputable def curve : (ℝ → ℝ) := λ x, Real.exp x - 3

theorem curve_properties : 
  (curve 0 = -2) ∧ (∀ x, deriv curve x = curve x + 3) :=
by
  sorry

end curve_properties_l2_2743


namespace value_of_f_2018_l2_2038

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 3) * f x = -1
axiom initial_condition : f (-1) = 2

theorem value_of_f_2018 : f 2018 = -1 / 2 :=
by
  sorry

end value_of_f_2018_l2_2038


namespace corrected_mean_l2_2367

theorem corrected_mean (n : ℕ) (mean : ℝ) (obs1 obs2 : ℝ) (inc1 inc2 cor1 cor2 : ℝ)
    (h_num_obs : n = 50)
    (h_initial_mean : mean = 36)
    (h_incorrect1 : inc1 = 23) (h_correct1 : cor1 = 34)
    (h_incorrect2 : inc2 = 55) (h_correct2 : cor2 = 45)
    : (mean * n + (cor1 - inc1) + (cor2 - inc2)) / n = 36.02 := 
by 
  -- Insert steps to prove the theorem here
  sorry

end corrected_mean_l2_2367


namespace plants_remaining_l2_2411

theorem plants_remaining (plants_initial plants_first_day plants_second_day_eaten plants_third_day_eaten : ℕ)
  (h1 : plants_initial = 30)
  (h2 : plants_first_day = 20)
  (h3 : plants_second_day_eaten = (plants_initial - plants_first_day) / 2)
  (h4 : plants_third_day_eaten = 1)
  : (plants_initial - plants_first_day - plants_second_day_eaten - plants_third_day_eaten) = 4 := 
by
  sorry

end plants_remaining_l2_2411


namespace diamond_value_l2_2811

def diamond (a b : Int) : Int :=
  a * b^2 - b + 1

theorem diamond_value : diamond (-1) 6 = -41 := by
  sorry

end diamond_value_l2_2811


namespace smallest_distance_zero_l2_2357

theorem smallest_distance_zero :
  let r_track (t : ℝ) := (Real.cos t, Real.sin t)
  let i_track (t : ℝ) := (Real.cos (t / 2), Real.sin (t / 2))
  ∀ t₁ t₂ : ℝ, dist (r_track t₁) (i_track t₂) = 0 := by
  sorry

end smallest_distance_zero_l2_2357


namespace eccentricity_of_ellipse_l2_2177

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

theorem eccentricity_of_ellipse {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : a > 0)
                                 (h_angle : Real.cos (Real.pi / 6) = b / a) :
    eccentricity a b = (Real.sqrt 6) / 3 := by
  sorry

end eccentricity_of_ellipse_l2_2177


namespace distance_foci_to_line_l2_2471

noncomputable def ellipse_equation (rho theta : ℝ) : Prop :=
  rho^2 = 12 / (3 * (Real.cos theta)^2 + 4 * (Real.sin theta)^2)

def line_parametric (t : ℝ) : ℝ × ℝ :=
  (2 + Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t)

def line_standard (x y : ℝ) : Prop :=
  x - y - 2 = 0

def ellipse_cartesian (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

def point_to_line_distance (x y a b c : ℝ) : ℝ :=
  Real.abs (a*x + b*y + c) / Real.sqrt (a^2 + b^2)

def sum_of_distances_to_line (F1 F2 : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  point_to_line_distance F1.1 F1.2 a b c + point_to_line_distance F2.1 F2.2 a b c

theorem distance_foci_to_line :
  let f1 := (-1 : ℝ, 0 : ℝ)
  let f2 := (1 : ℝ, 0 : ℝ)
  let a := 1
  let b := -1
  let c := -2
  sum_of_distances_to_line f1 f2 a b c = 2 * Real.sqrt 2 :=
by
  sorry

end distance_foci_to_line_l2_2471


namespace derivative_at_2_l2_2949

def f (x : ℝ) : ℝ := (x + 3) * (x + 2) * (x + 1) * x * (x - 1) * (x - 2) * (x - 3)

theorem derivative_at_2 : (deriv f 2) = -120 :=
by
  sorry

end derivative_at_2_l2_2949


namespace area_in_square_yards_l2_2125

/-
  Given:
  - length of the classroom in feet
  - width of the classroom in feet

  Prove that the area required to cover the classroom in square yards is 30. 
-/

def classroom_length_feet : ℕ := 15
def classroom_width_feet : ℕ := 18
def feet_to_yard (feet : ℕ) : ℕ := feet / 3

theorem area_in_square_yards :
  let length_yards := feet_to_yard classroom_length_feet
  let width_yards := feet_to_yard classroom_width_feet
  length_yards * width_yards = 30 :=
by
  sorry

end area_in_square_yards_l2_2125


namespace breakEvenBooks_l2_2568

theorem breakEvenBooks (FC VC_per_book SP : ℝ) (hFC : FC = 56430) (hVC : VC_per_book = 8.25) (hSP : SP = 21.75) :
  ∃ x : ℕ, FC + (VC_per_book * x) = SP * x ∧ x = 4180 :=
by {
  sorry
}

end breakEvenBooks_l2_2568


namespace amusement_park_admission_l2_2526

def number_of_children (children_fee : ℤ) (adults_fee : ℤ) (total_people : ℤ) (total_fees : ℤ) : ℤ :=
  let y := (total_fees - total_people * children_fee) / (adults_fee - children_fee)
  total_people - y

theorem amusement_park_admission :
  number_of_children 15 40 315 8100 = 180 :=
by
  -- Fees in cents to avoid decimals
  sorry  -- Placeholder for the proof

end amusement_park_admission_l2_2526


namespace tan_seven_pi_over_four_l2_2293

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by
  sorry

end tan_seven_pi_over_four_l2_2293


namespace birthday_count_l2_2674

theorem birthday_count (N : ℕ) (P : ℝ) (days : ℕ) (hN : N = 1200) (hP1 : P = 1 / 365 ∨ P = 1 / 366) 
  (hdays : days = 365 ∨ days = 366) : 
  N * P = 4 :=
by
  sorry

end birthday_count_l2_2674


namespace opposite_of_three_l2_2818

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l2_2818


namespace prove_a_value_l2_2278

theorem prove_a_value (a : ℝ) (h : (a - 2) * 0^2 + 0 + a^2 - 4 = 0) : a = -2 := 
by
  sorry

end prove_a_value_l2_2278


namespace performance_stability_l2_2383

theorem performance_stability (avg_score : ℝ) (num_shots : ℕ) (S_A S_B : ℝ) 
  (h_avg : num_shots = 10)
  (h_same_avg : avg_score = avg_score) 
  (h_SA : S_A^2 = 0.4) 
  (h_SB : S_B^2 = 2) : 
  (S_A < S_B) :=
by
  sorry

end performance_stability_l2_2383


namespace range_of_f_l2_2011

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem range_of_f 
  (x : ℝ) : f (x - 1) + f (x + 1) > 0 ↔ x ∈ Set.Ioi 0 :=
by
  sorry

end range_of_f_l2_2011


namespace mixed_candy_price_l2_2569

noncomputable def price_per_pound (a b c : ℕ) (pa pb pc : ℝ) : ℝ :=
  (a * pa + b * pb + c * pc) / (a + b + c)

theorem mixed_candy_price :
  let a := 30
  let b := 15
  let c := 20
  let pa := 10.0
  let pb := 12.0
  let pc := 15.0
  price_per_pound a b c pa pb pc * 0.9 = 10.8 := by
  sorry

end mixed_candy_price_l2_2569


namespace nat_pairs_exp_eq_l2_2926

theorem nat_pairs_exp_eq (a b : ℕ) : a^b = b^a ↔ (a = b) ∨ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2) := 
by
  sorry

end nat_pairs_exp_eq_l2_2926


namespace jerry_expected_candies_l2_2964

noncomputable def blue_eggs := rat.mk 4 10
noncomputable def purple_eggs := rat.mk 3 10
noncomputable def red_eggs := rat.mk 2 10
noncomputable def green_eggs := rat.mk 1 10

-- Expected number of candies for each egg color
noncomputable def E_blue :=
  (rat.mk 1 3) * 3 + (rat.mk 1 2) * 2 + (rat.mk 1 6) * 0

noncomputable def E_purple :=
  (rat.mk 1 2) * 5 + (rat.mk 1 2) * 0

noncomputable def E_red :=
  (rat.mk 3 4) * 1 + (rat.mk 1 4) * 4

noncomputable def E_green :=
  (rat.mk 1 2) * 6 + (rat.mk 1 2) * 8

-- Overall expected number of candies
noncomputable def expected_candies :=
  blue_eggs * E_blue + purple_eggs * E_purple +
  red_eggs * E_red + green_eggs * E_green

theorem jerry_expected_candies : expected_candies = rat.mk 26 10 :=
  by 
    -- Verification of expected candies calculation
    have h1 : E_blue = 2 := sorry
    have h2 : E_purple = 2.5 := sorry
    have h3 : E_red = 1.75 := sorry
    have h4 : E_green = 7 := sorry
    sorry

end jerry_expected_candies_l2_2964


namespace expand_simplify_expression_l2_2247

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l2_2247


namespace rectangular_box_in_sphere_radius_l2_2567

theorem rectangular_box_in_sphere_radius (a b c s : ℝ) 
  (h1 : a + b + c = 40) 
  (h2 : 2 * a * b + 2 * b * c + 2 * a * c = 608) 
  (h3 : (2 * s)^2 = a^2 + b^2 + c^2) : 
  s = 16 * Real.sqrt 2 :=
by
  sorry

end rectangular_box_in_sphere_radius_l2_2567


namespace dan_initial_money_l2_2419

def initial_amount (spent_candy : ℕ) (spent_chocolate : ℕ) (remaining : ℕ) : ℕ :=
  spent_candy + spent_chocolate + remaining

theorem dan_initial_money 
  (spent_candy : ℕ) (spent_chocolate : ℕ) (remaining : ℕ)
  (h_candy : spent_candy = 2)
  (h_chocolate : spent_chocolate = 3)
  (h_remaining : remaining = 2) :
  initial_amount spent_candy spent_chocolate remaining = 7 :=
by
  rw [h_candy, h_chocolate, h_remaining]
  unfold initial_amount
  rfl

end dan_initial_money_l2_2419


namespace ellipse_equation_hyperbola_vertices_and_foci_exists_point_P_on_x_axis_angles_complementary_l2_2755

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - y^2 / 2 = 1

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 2 = 1

def point_on_x_axis (P : ℝ × ℝ) : Prop :=
  P.snd = 0

def angles_complementary (P A B : ℝ × ℝ) : Prop :=
  let kPA := (A.snd - P.snd) / (A.fst - P.fst)
  let kPB := (B.snd - P.snd) / (B.fst - P.fst)
  kPA + kPB = 0

theorem ellipse_equation_hyperbola_vertices_and_foci :
  (∀ x y : ℝ, hyperbola_eq x y → ellipse_eq x y) :=
sorry

theorem exists_point_P_on_x_axis_angles_complementary (F2 A B : ℝ × ℝ) :
  F2 = (1, 0) → (∃ P : ℝ × ℝ, point_on_x_axis P ∧ angles_complementary P A B) :=
sorry

end ellipse_equation_hyperbola_vertices_and_foci_exists_point_P_on_x_axis_angles_complementary_l2_2755


namespace f_neg_one_l2_2667

-- Assume the function f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- Conditions
-- 1. f(x) is odd: f(-x) = -f(x) for all x ∈ ℝ
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- 2. f(x) = 2^x for all x > 0
axiom f_pos : ∀ x : ℝ, x > 0 → f x = 2^x

-- Proof statement to be filled
theorem f_neg_one : f (-1) = -2 := 
by
  sorry

end f_neg_one_l2_2667


namespace arithmetic_sequence_probability_l2_2799

theorem arithmetic_sequence_probability (n p : ℕ) (h_cond : n + p = 2008) (h_neg : n = 161) (h_pos : p = 2008 - 161) :
  ∃ a b : ℕ, (a = 1715261 ∧ b = 2016024 ∧ a + b = 3731285) ∧ (a / b = 1715261 / 2016024) := by
  sorry

end arithmetic_sequence_probability_l2_2799


namespace intersection_is_correct_l2_2207

noncomputable def setA : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
noncomputable def setB : Set ℝ := {x | Real.log x / Real.log 2 ≤ 2}

theorem intersection_is_correct : setA ∩ setB = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_is_correct_l2_2207


namespace employee_payment_l2_2109

theorem employee_payment (X Y : ℝ) (h1 : X + Y = 528) (h2 : X = 1.2 * Y) : Y = 240 :=
by
  sorry

end employee_payment_l2_2109


namespace transform_graph_of_g_to_f_l2_2689

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - Real.sqrt 3 * Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x) + 1

theorem transform_graph_of_g_to_f :
  ∀ (x : ℝ), f x = g (x + (5 * Real.pi) / 12) :=
by
  sorry

end transform_graph_of_g_to_f_l2_2689


namespace evaluate_expression_l2_2586

theorem evaluate_expression (a : ℚ) (h : a = 4 / 3) : 
  (6 * a ^ 2 - 15 * a + 5) * (3 * a - 4) = 0 := by
  sorry

end evaluate_expression_l2_2586


namespace solve_equation_l2_2996

theorem solve_equation (x : ℝ) (h : (x - 60) / 3 = (4 - 3 * x) / 6) : x = 124 / 5 := by
  sorry

end solve_equation_l2_2996


namespace scientific_notation_conversion_l2_2496

theorem scientific_notation_conversion (x : ℝ) (h_population : x = 141260000) :
  x = 1.4126 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l2_2496


namespace supplement_of_complement_of_35_degree_angle_l2_2085

theorem supplement_of_complement_of_35_degree_angle : 
  ∀ α β : ℝ,  α = 35 ∧ β = (90 - α) → (180 - β) = 125 :=
by
  intros α β h
  rcases h with ⟨h1, h2⟩
  rw h1 at h2
  rw h2
  linarith

end supplement_of_complement_of_35_degree_angle_l2_2085


namespace balloon_height_per_ounce_l2_2180

theorem balloon_height_per_ounce
    (total_money : ℕ)
    (sheet_cost : ℕ)
    (rope_cost : ℕ)
    (propane_cost : ℕ)
    (helium_price : ℕ)
    (max_height : ℕ)
    :
    total_money = 200 →
    sheet_cost = 42 →
    rope_cost = 18 →
    propane_cost = 14 →
    helium_price = 150 →
    max_height = 9492 →
    max_height / ((total_money - (sheet_cost + rope_cost + propane_cost)) / helium_price) = 113 :=
by
  intros
  sorry

end balloon_height_per_ounce_l2_2180


namespace opposite_of_3_l2_2836

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l2_2836


namespace julia_played_with_34_kids_l2_2976

-- Define the number of kids Julia played with on each day
def kidsMonday : Nat := 17
def kidsTuesday : Nat := 15
def kidsWednesday : Nat := 2

-- Define the total number of kids Julia played with
def totalKids : Nat := kidsMonday + kidsTuesday + kidsWednesday

-- Prove given conditions
theorem julia_played_with_34_kids :
  totalKids = 34 :=
by
  sorry

end julia_played_with_34_kids_l2_2976


namespace total_amount_paid_l2_2728

-- Define the given conditions
def q_g : ℕ := 9        -- Quantity of grapes
def r_g : ℕ := 70       -- Rate per kg of grapes
def q_m : ℕ := 9        -- Quantity of mangoes
def r_m : ℕ := 55       -- Rate per kg of mangoes

-- Define the total amount paid calculation and prove it equals 1125
theorem total_amount_paid : (q_g * r_g + q_m * r_m) = 1125 :=
by
  -- Proof will be provided here. Currently using 'sorry' to skip it.
  sorry

end total_amount_paid_l2_2728


namespace simplify_expression_l2_2521

theorem simplify_expression : 8 * (15 / 9) * (-45 / 40) = -1 :=
  by
  sorry

end simplify_expression_l2_2521


namespace opposite_of_3_l2_2839

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l2_2839


namespace inequality_solution_l2_2064

-- We define the problem
def interval_of_inequality : Set ℝ := { x : ℝ | (x + 1) * (2 - x) > 0 }

-- We define the expected solution set
def expected_solution_set : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

-- The theorem to be proved
theorem inequality_solution :
  interval_of_inequality = expected_solution_set := by 
  sorry

end inequality_solution_l2_2064


namespace opposite_of_3_l2_2853

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l2_2853


namespace student_tickets_count_l2_2544

-- Defining the parameters and conditions
variables (A S : ℕ)
variables (h1 : A + S = 59) (h2 : 4 * A + 5 * S / 2 = 222.50)

-- The statement to prove
theorem student_tickets_count : S = 9 :=
by
  sorry

end student_tickets_count_l2_2544


namespace problem1_problem2_l2_2560

variable (a : ℝ) -- Declaring a as a real number

-- Proof statement for Problem 1
theorem problem1 : (a + 2) * (a - 2) = a^2 - 4 :=
sorry

-- Proof statement for Problem 2
theorem problem2 (h : a ≠ -2) : (a^2 - 4) / (a + 2) + 2 = a :=
sorry

end problem1_problem2_l2_2560


namespace sufficient_but_not_necessary_l2_2104

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 < x ∧ x < 2) : x < 2 ∧ ∀ y, (y < 2 → y ≤ 1 ∨ y ≥ 2) :=
by
  sorry

end sufficient_but_not_necessary_l2_2104


namespace largest_number_formed_l2_2186

-- Define the digits
def digit1 : ℕ := 2
def digit2 : ℕ := 6
def digit3 : ℕ := 9

-- Define the function to form the largest number using the given digits
def largest_three_digit_number (a b c : ℕ) : ℕ :=
  if a > b ∧ a > c then
    if b > c then 100 * a + 10 * b + c
    else 100 * a + 10 * c + b
  else if b > a ∧ b > c then
    if a > c then 100 * b + 10 * a + c
    else 100 * b + 10 * c + a
  else
    if a > b then 100 * c + 10 * a + b
    else 100 * c + 10 * b + a

-- Statement that this function correctly computes the largest number
theorem largest_number_formed :
  largest_three_digit_number digit1 digit2 digit3 = 962 :=
by
  sorry

end largest_number_formed_l2_2186


namespace opposite_of_3_is_neg3_l2_2831

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2831


namespace train_length_l2_2572

noncomputable def length_of_train (t : ℝ) (v_train_kmh : ℝ) (v_man_kmh : ℝ) : ℝ :=
  let v_relative_kmh := v_train_kmh - v_man_kmh
  let v_relative_ms := v_relative_kmh * 1000 / 3600
  v_relative_ms * t

theorem train_length : length_of_train 30.99752019838413 80 8 = 619.9504039676826 := 
  by simp [length_of_train]; sorry

end train_length_l2_2572


namespace supplement_of_complement_is_125_l2_2092

-- Definition of the initial angle
def initial_angle : ℝ := 35

-- Definition of the complement of the angle
def complement_angle (θ : ℝ) : ℝ := 90 - θ

-- Definition of the supplement of an angle
def supplement_angle (θ : ℝ) : ℝ := 180 - θ

-- Main theorem statement
theorem supplement_of_complement_is_125 : 
  supplement_angle (complement_angle initial_angle) = 125 := 
by
  sorry

end supplement_of_complement_is_125_l2_2092


namespace max_min_distance_inequality_l2_2595

theorem max_min_distance_inequality (n : ℕ) (D d : ℝ) (h1 : d > 0) 
    (exists_points : ∃ (points : Fin n → ℝ × ℝ), 
      (∀ i j : Fin n, i ≠ j → dist (points i) (points j) ≥ d) 
      ∧ (∀ i j : Fin n, dist (points i) (points j) ≤ D)) : 
    D / d > (Real.sqrt (n * Real.pi)) / 2 - 1 := 
  sorry

end max_min_distance_inequality_l2_2595


namespace parametric_eqn_and_max_sum_l2_2469

noncomputable def polar_eq (ρ θ : ℝ) := ρ^2 = 4 * ρ * (Real.cos θ + Real.sin θ) - 6

theorem parametric_eqn_and_max_sum (θ : ℝ):
  (∃ (x y : ℝ), (2 + Real.sqrt 2 * Real.cos θ, 2 + Real.sqrt 2 * Real.sin θ) = (x, y)) ∧
  (∃ (θ : ℝ), θ = Real.pi / 4 → (3, 3) = (3, 3) ∧ 6 = 6) :=
by {
  sorry
}

end parametric_eqn_and_max_sum_l2_2469


namespace jesse_mia_total_miles_per_week_l2_2781

noncomputable def jesse_miles_per_day_first_three := 2 / 3
noncomputable def jesse_miles_day_four := 10
noncomputable def mia_miles_per_day_first_four := 3
noncomputable def average_final_three_days := 6

theorem jesse_mia_total_miles_per_week :
  let jesse_total_first_four_days := 3 * jesse_miles_per_day_first_three + jesse_miles_day_four
  let mia_total_first_four_days := 4 * mia_miles_per_day_first_four
  let total_miles_needed_final_three_days := 3 * average_final_three_days * 2
  jesse_total_first_four_days + total_miles_needed_final_three_days = 48 ∧
  mia_total_first_four_days + total_miles_needed_final_three_days = 48 :=
by
  sorry

end jesse_mia_total_miles_per_week_l2_2781


namespace find_c_value_l2_2117

theorem find_c_value :
  ∃ c : ℝ, (∀ x y : ℝ, (x + 10) ^ 2 + (y + 4) ^ 2 = 169 ∧ (x - 3) ^ 2 + (y - 9) ^ 2 = 65 → x + y = c) ∧ c = 3 :=
sorry

end find_c_value_l2_2117


namespace factorize_quadratic_l2_2424

theorem factorize_quadratic : ∀ x : ℝ, x^2 - 7*x + 10 = (x - 2)*(x - 5) :=
by
  sorry

end factorize_quadratic_l2_2424


namespace find_extra_lives_first_level_l2_2026

-- Conditions as definitions
def initial_lives : ℕ := 2
def extra_lives_second_level : ℕ := 11
def total_lives_after_second_level : ℕ := 19

-- Definition representing the extra lives in the first level
def extra_lives_first_level (x : ℕ) : Prop :=
  initial_lives + x + extra_lives_second_level = total_lives_after_second_level

-- The theorem we need to prove
theorem find_extra_lives_first_level : ∃ x : ℕ, extra_lives_first_level x ∧ x = 6 :=
by
  sorry  -- Placeholder for the proof

end find_extra_lives_first_level_l2_2026


namespace sufficient_but_not_necessary_condition_l2_2170

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 1 → 1 / a < 1) ∧ ((1 / a < 1) → (a > 1 ∨ a < 0)) → 
  (∀ (P Q : Prop), (P → Q) → (Q → P ∨ False) → P ∧ ¬Q → False) :=
by
  sorry

end sufficient_but_not_necessary_condition_l2_2170


namespace tan_7pi_over_4_eq_neg1_l2_2295

theorem tan_7pi_over_4_eq_neg1 : Real.tan (7 * Real.pi / 4) = -1 :=
  sorry

end tan_7pi_over_4_eq_neg1_l2_2295


namespace coefficient_of_x3y7_in_expansion_l2_2079

-- Definitions based on the conditions in the problem
def a : ℚ := (2 / 3)
def b : ℚ := - (3 / 4)
def n : ℕ := 10
def k1 : ℕ := 3
def k2 : ℕ := 7

-- Statement of the math proof problem
theorem coefficient_of_x3y7_in_expansion :
  (a * x ^ k1 + b * y ^ k2) ^ n = x3y7_coeff * x ^ k1 * y ^ k2  :=
sorry

end coefficient_of_x3y7_in_expansion_l2_2079


namespace annual_interest_payment_l2_2416

def principal : ℝ := 10000
def quarterly_rate : ℝ := 0.05

theorem annual_interest_payment :
  (principal * quarterly_rate * 4) = 2000 :=
by sorry

end annual_interest_payment_l2_2416


namespace expand_and_simplify_l2_2239

noncomputable def expanded_expr (a : ℝ) : ℝ :=
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6)

theorem expand_and_simplify (a : ℝ) :
  expanded_expr a = a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 :=
by {
  -- Proof goes here
  sorry
}

end expand_and_simplify_l2_2239


namespace find_square_tiles_l2_2712

theorem find_square_tiles (t s p : ℕ) (h1 : t + s + p = 35) (h2 : 3 * t + 4 * s + 5 * p = 140) (hp0 : p = 0) : s = 35 := by
  sorry

end find_square_tiles_l2_2712


namespace fraction_of_white_surface_area_is_11_16_l2_2120

theorem fraction_of_white_surface_area_is_11_16 :
  let cube_surface_area := 6 * 4^2
  let total_surface_faces := 96
  let corner_black_faces := 8 * 3
  let center_black_faces := 6 * 1
  let total_black_faces := corner_black_faces + center_black_faces
  let white_faces := total_surface_faces - total_black_faces
  (white_faces : ℚ) / total_surface_faces = 11 / 16 := 
by sorry

end fraction_of_white_surface_area_is_11_16_l2_2120


namespace find_x_value_l2_2618

theorem find_x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^3) (h2 : x / 6 = 3 * y) : x = 18 * Real.sqrt 6 := 
by 
  sorry

end find_x_value_l2_2618


namespace JerushaEarnings_is_68_l2_2975

-- Define the conditions
def LottiesEarnings := ℝ
def JerushaEarnings (L : LottiesEarnings) := 4 * L

-- Condition 1: Jerusha's earning is 4 times Lottie's earnings
def condition1 (L : LottiesEarnings) : Prop := JerushaEarnings L = 4 * L

-- Condition 2: The total earnings of Jerusha and Lottie is $85
def condition2 (L : LottiesEarnings) : Prop := JerushaEarnings L + L = 85

-- The theorem to prove Jerusha's earnings is $68
theorem JerushaEarnings_is_68 (L : LottiesEarnings) (h1 : condition1 L) (h2 : condition2 L) : JerushaEarnings L = 68 := 
by 
  sorry

end JerushaEarnings_is_68_l2_2975


namespace scientific_notation_141260_million_l2_2493

theorem scientific_notation_141260_million :
  ∃ (a : ℝ) (n : ℤ), 141260 * 10^6 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4126 ∧ n = 5 :=
by
  sorry

end scientific_notation_141260_million_l2_2493


namespace cooking_time_l2_2715

theorem cooking_time
  (total_potatoes : ℕ) (cooked_potatoes : ℕ) (remaining_time : ℕ) (remaining_potatoes : ℕ)
  (h_total : total_potatoes = 15)
  (h_cooked : cooked_potatoes = 8)
  (h_remaining_time : remaining_time = 63)
  (h_remaining_potatoes : remaining_potatoes = total_potatoes - cooked_potatoes) :
  remaining_time / remaining_potatoes = 9 :=
by
  sorry

end cooking_time_l2_2715


namespace probability_factor_lt_10_l2_2553

theorem probability_factor_lt_10 (n : ℕ) (h : n = 90) :
  (∃ factors_lt_10 : ℕ, ∃ total_factors : ℕ,
    factors_lt_10 = 7 ∧ total_factors = 12 ∧ (factors_lt_10 / total_factors : ℚ) = 7 / 12) :=
by sorry

end probability_factor_lt_10_l2_2553


namespace perfect_square_condition_l2_2547

theorem perfect_square_condition (x y : ℕ) :
  ∃ k : ℕ, (x + y)^2 + 3*x + y + 1 = k^2 ↔ x = y := 
by 
  sorry

end perfect_square_condition_l2_2547


namespace max_minute_hands_l2_2100

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
sorry

end max_minute_hands_l2_2100


namespace line_parallel_to_y_axis_l2_2626

theorem line_parallel_to_y_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x + b * y + 1 = 0 → b = 0):
  a ≠ 0 ∧ b = 0 :=
sorry

end line_parallel_to_y_axis_l2_2626


namespace geometric_sum_common_ratios_l2_2111

theorem geometric_sum_common_ratios (k p r : ℝ) 
  (hp : p ≠ r) (h_seq : p ≠ 1 ∧ r ≠ 1 ∧ p ≠ 0 ∧ r ≠ 0) 
  (h : k * p^4 - k * r^4 = 4 * (k * p^2 - k * r^2)) : 
  p + r = 3 :=
by
  -- Details omitted as requested
  sorry

end geometric_sum_common_ratios_l2_2111


namespace hyperbola_satisfies_m_l2_2761

theorem hyperbola_satisfies_m (m : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 - m * y^2 = 1)
  (h2 : ∀ a b : ℝ, (a^2 = 1) ∧ (b^2 = 1/m) ∧ (2*a = 2 * 2*b)) : 
  m = 4 := 
sorry

end hyperbola_satisfies_m_l2_2761


namespace max_minute_hands_l2_2102

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
  sorry

end max_minute_hands_l2_2102


namespace find_two_digit_number_t_l2_2591

theorem find_two_digit_number_t (t : ℕ) (ht1 : 10 ≤ t) (ht2 : t ≤ 99) (ht3 : 13 * t % 100 = 52) : t = 12 := 
sorry

end find_two_digit_number_t_l2_2591


namespace fraction_to_decimal_l2_2423

theorem fraction_to_decimal :
  (45 : ℚ) / (5 ^ 3) = 0.360 :=
by
  sorry

end fraction_to_decimal_l2_2423


namespace initial_bananas_on_tree_l2_2710

-- Definitions of given conditions
def bananas_left_on_tree : ℕ := 100
def bananas_eaten : ℕ := 70
def bananas_in_basket : ℕ := 2 * bananas_eaten

-- Statement to prove the initial number of bananas on the tree
theorem initial_bananas_on_tree : bananas_left_on_tree + (bananas_in_basket + bananas_eaten) = 310 :=
by
  sorry

end initial_bananas_on_tree_l2_2710


namespace profit_is_55_l2_2517

-- Define the given conditions:
def cost_of_chocolates (bars: ℕ) (price_per_bar: ℕ) : ℕ :=
  bars * price_per_bar

def cost_of_packaging (bars: ℕ) (cost_per_bar: ℕ) : ℕ :=
  bars * cost_per_bar

def total_sales : ℕ :=
  90

def total_cost (cost_of_chocolates cost_of_packaging: ℕ) : ℕ :=
  cost_of_chocolates + cost_of_packaging

def profit (total_sales total_cost: ℕ) : ℕ :=
  total_sales - total_cost

-- Given values:
def bars: ℕ := 5
def price_per_bar: ℕ := 5
def cost_per_packaging_bar: ℕ := 2

-- Define the profit calculation theorem:
theorem profit_is_55 : 
  profit total_sales (total_cost (cost_of_chocolates bars price_per_bar) (cost_of_packaging bars cost_per_packaging_bar)) = 55 :=
by {
  -- The proof will be inserted here
  sorry
}

end profit_is_55_l2_2517


namespace half_angle_quadrant_l2_2444

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π + π < α ∧ α < 2 * k * π + (3 * π / 2)) : 
  (∃ j : ℤ, j * π + (π / 2) < (α / 2) ∧ (α / 2) < j * π + (3 * π / 4)) :=
  by sorry

end half_angle_quadrant_l2_2444


namespace apple_cost_l2_2988

theorem apple_cost (A : ℝ) (h_discount : ∃ (n : ℕ), 15 = (5 * (5: ℝ) * A + 3 * 2 + 2 * 3 - n)) : A = 1 :=
by
  sorry

end apple_cost_l2_2988


namespace compare_exponents_l2_2005

def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

theorem compare_exponents : b < a ∧ a < c :=
by
  have h1 : a = 2^(4/3) := rfl
  have h2 : b = 4^(2/5) := rfl
  have h3 : c = 25^(1/3) := rfl
  -- These are used to indicate the definitions, not the proof steps
  sorry

end compare_exponents_l2_2005


namespace smallest_positive_integer_in_linear_combination_l2_2152

theorem smallest_positive_integer_in_linear_combination :
  ∃ m n : ℤ, 2016 * m + 43200 * n = 24 :=
by
  sorry

end smallest_positive_integer_in_linear_combination_l2_2152


namespace supplement_of_complement_is_125_l2_2090

-- Definition of the initial angle
def initial_angle : ℝ := 35

-- Definition of the complement of the angle
def complement_angle (θ : ℝ) : ℝ := 90 - θ

-- Definition of the supplement of an angle
def supplement_angle (θ : ℝ) : ℝ := 180 - θ

-- Main theorem statement
theorem supplement_of_complement_is_125 : 
  supplement_angle (complement_angle initial_angle) = 125 := 
by
  sorry

end supplement_of_complement_is_125_l2_2090


namespace find_y_l2_2031

noncomputable def angle_ABC := 75
noncomputable def angle_BAC := 70
noncomputable def angle_CDE := 90
noncomputable def angle_BCA : ℝ := 180 - (angle_ABC + angle_BAC)
noncomputable def y : ℝ := 90 - angle_BCA

theorem find_y : y = 55 :=
by
  have h1: angle_BCA = 180 - (75 + 70) := rfl
  have h2: y = 90 - angle_BCA := rfl
  rw [h1] at h2
  exact h2.trans (by norm_num)

end find_y_l2_2031


namespace copper_zinc_ratio_l2_2555

theorem copper_zinc_ratio (total_weight : ℝ) (zinc_weight : ℝ) 
  (h_total_weight : total_weight = 70) (h_zinc_weight : zinc_weight = 31.5) : 
  (70 - 31.5) / 31.5 = 77 / 63 :=
by
  have h_copper_weight : total_weight - zinc_weight = 38.5 :=
    by rw [h_total_weight, h_zinc_weight]; norm_num
  sorry

end copper_zinc_ratio_l2_2555


namespace chess_club_not_playing_any_l2_2632

theorem chess_club_not_playing_any (total_members : ℕ) (chess_players : ℕ) (checkers_players : ℕ) (both_players : ℕ) 
  (h1 : total_members = 70) (h2 : chess_players = 45) (h3 : checkers_players = 38) (h4 : both_players = 25) : 
  total_members - (chess_players - both_players + checkers_players - both_players + both_players) = 12 := 
by sorry

end chess_club_not_playing_any_l2_2632


namespace find_fraction_l2_2331

theorem find_fraction (f n : ℝ) (h1 : f * n - 5 = 5) (h2 : n = 50) : f = 1 / 5 :=
by
  -- skipping the proof as requested
  sorry

end find_fraction_l2_2331


namespace find_hourly_charge_computer_B_l2_2272

noncomputable def hourly_charge_computer_B (B : ℝ) :=
  ∃ (A h : ℝ),
    A = 1.4 * B ∧
    B * (h + 20) = 550 ∧
    A * h = 550 ∧
    B = 7.86

theorem find_hourly_charge_computer_B : ∃ B : ℝ, hourly_charge_computer_B B :=
  sorry

end find_hourly_charge_computer_B_l2_2272


namespace percentage_of_the_stock_l2_2713

noncomputable def faceValue : ℝ := 100
noncomputable def yield : ℝ := 0.10
noncomputable def quotedPrice : ℝ := 160

theorem percentage_of_the_stock : 
  (yield * faceValue / quotedPrice * 100 = 6.25) :=
by
  sorry

end percentage_of_the_stock_l2_2713


namespace isosceles_triangle_angle_condition_l2_2622

theorem isosceles_triangle_angle_condition (A B C : ℝ) (h_iso : A = B) (h_angle_eq : A = 2 * C ∨ C = 2 * A) :
    (A = 45 ∨ A = 72) ∧ (B = 45 ∨ B = 72) :=
by
  -- Given isosceles triangle properties.
  sorry

end isosceles_triangle_angle_condition_l2_2622


namespace find_m_l2_2647

def numFactorsOf2 (k : ℕ) : ℕ :=
  k / 2 + k / 4 + k / 8 + k / 16 + k / 32 + k / 64 + k / 128 + k / 256

theorem find_m : ∃ m : ℕ, m > 1990 ^ 1990 ∧ m = 3 ^ 1990 + numFactorsOf2 m :=
by
  sorry

end find_m_l2_2647


namespace opposite_of_3_is_neg3_l2_2825

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2825


namespace find_costs_l2_2777

theorem find_costs (a b : ℝ) (h1 : a - b = 3) (h2 : 3 * b - 2 * a = 3) : a = 12 ∧ b = 9 :=
sorry

end find_costs_l2_2777


namespace inequality_non_empty_solution_l2_2677

theorem inequality_non_empty_solution (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x + 1 < 0) → a ≤ 1 := sorry

end inequality_non_empty_solution_l2_2677


namespace smallest_four_digit_multiple_of_17_l2_2377

theorem smallest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n ∧ n < 10000 ∧ n % 17 = 0) ∧ ∀ m, (1000 ≤ m ∧ m < 10000 ∧ m % 17 = 0 → n ≤ m) ∧ n = 1013 :=
by
  sorry

end smallest_four_digit_multiple_of_17_l2_2377


namespace interest_rate_l2_2119

-- Definitions based on given conditions
def SumLent : ℝ := 1500
def InterestTime : ℝ := 4
def InterestAmount : ℝ := SumLent - 1260

-- Main theorem to prove the interest rate r is 4%
theorem interest_rate (r : ℝ) : (InterestAmount = SumLent * r / 100 * InterestTime) → r = 4 :=
by
  sorry

end interest_rate_l2_2119


namespace equation_one_solutions_equation_two_solutions_l2_2662

theorem equation_one_solutions (x : ℝ) : x^2 + 2 * x - 8 = 0 ↔ x = -4 ∨ x = 2 := 
by {
  sorry
}

theorem equation_two_solutions (x : ℝ) : x * (x - 2) = x - 2 ↔ x = 2 ∨ x = 1 := 
by {
  sorry
}

end equation_one_solutions_equation_two_solutions_l2_2662


namespace smallest_base10_integer_l2_2375

theorem smallest_base10_integer :
  ∃ (n A B : ℕ), 
    (A < 5) ∧ (B < 7) ∧ 
    (n = 6 * A) ∧ 
    (n = 8 * B) ∧ 
    n = 24 := 
sorry

end smallest_base10_integer_l2_2375


namespace inequality_system_solution_l2_2678

theorem inequality_system_solution (x : ℝ) : x + 1 > 0 → x - 3 > 0 → x > 3 :=
by
  intros h1 h2
  sorry

end inequality_system_solution_l2_2678


namespace max_three_cards_l2_2490

theorem max_three_cards (n m p : ℕ) (h : n + m + p = 8) (sum : 3 * n + 4 * m + 5 * p = 33) 
  (n_le_10 : n ≤ 10) (m_le_10 : m ≤ 10) (p_le_10 : p ≤ 10) : n ≤ 3 := 
sorry

end max_three_cards_l2_2490


namespace probability_of_selecting_girl_l2_2155

theorem probability_of_selecting_girl (boys girls : ℕ) (total_students : ℕ) (prob : ℚ) 
  (h1 : boys = 3) 
  (h2 : girls = 2) 
  (h3 : total_students = boys + girls) 
  (h4 : prob = girls / total_students) : 
  prob = 2 / 5 := 
sorry

end probability_of_selecting_girl_l2_2155


namespace simplify_expression_eq_square_l2_2902

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l2_2902


namespace new_light_wattage_is_143_l2_2386

-- Define the original wattage and the percentage increase
def original_wattage : ℕ := 110
def percentage_increase : ℕ := 30

-- Compute the increase in wattage
noncomputable def increase : ℕ := (percentage_increase * original_wattage) / 100

-- The new wattage should be the original wattage plus the increase
noncomputable def new_wattage : ℕ := original_wattage + increase

-- State the theorem that proves the new wattage is 143 watts
theorem new_light_wattage_is_143 : new_wattage = 143 := by
  unfold new_wattage
  unfold increase
  sorry

end new_light_wattage_is_143_l2_2386


namespace johns_profit_l2_2476

-- Definitions based on Conditions
def original_price_per_bag : ℝ := 4
def discount_percentage : ℝ := 0.10
def discounted_price_per_bag := original_price_per_bag * (1 - discount_percentage)
def bags_bought : ℕ := 30
def cost_per_bag : ℝ := if bags_bought >= 20 then discounted_price_per_bag else original_price_per_bag
def total_cost := bags_bought * cost_per_bag
def bags_sold_to_adults : ℕ := 20
def bags_sold_to_children : ℕ := 10
def price_per_bag_for_adults : ℝ := 8
def price_per_bag_for_children : ℝ := 6
def revenue_from_adults := bags_sold_to_adults * price_per_bag_for_adults
def revenue_from_children := bags_sold_to_children * price_per_bag_for_children
def total_revenue := revenue_from_adults + revenue_from_children
def profit := total_revenue - total_cost

-- Lean Statement to be Proven
theorem johns_profit : profit = 112 :=
by
  sorry

end johns_profit_l2_2476


namespace geometric_sequence_solution_l2_2433

variables (a : ℕ → ℝ) (q : ℝ)
-- Given conditions
def condition1 : Prop := abs (a 1) = 1
def condition2 : Prop := a 5 = -8 * a 2
def condition3 : Prop := a 5 > a 2
-- Proof statement
theorem geometric_sequence_solution :
  condition1 a → condition2 a → condition3 a → ∀ n, a n = (-2)^(n - 1) :=
sorry

end geometric_sequence_solution_l2_2433


namespace correct_statement_l2_2382

-- conditions
def condition_a : Prop := ∀(population : Type), ¬(comprehensive_survey population)
def data_set := [3, 5, 4, 1, -2]
def condition_b : Prop := median data_set = 4
def winning_probability : ℚ := 1 / 20
def condition_c : Prop := (∀n : ℕ, winning_probability * n = 1) → (∃i, i = 20)
def average_score (scores : List ℝ) : ℝ := scores.sum / scores.length
def variance (scores : List ℝ) : ℝ := (scores.map (λ x, (x - average_score scores)^2)).sum / scores.length
def scores_a := replicate 10 (average_score (replicate 10 10)) -- assumed scores for simplification
def scores_b := scores_a ++ [n + 1 for n in scores_a] -- assumed different scores to reflect variance
def condition_d : Prop := (average_score scores_a = average_score scores_b) ∧ (variance scores_a = 0.4) ∧ (variance scores_b = 2)

-- statement of the problem
theorem correct_statement : 
  condition_a ∧ condition_b ∧ condition_c ∧ condition_d → (∃D : Prop, D = condition_d) :=
by
  sorry

end correct_statement_l2_2382


namespace opposite_of_three_l2_2814

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l2_2814


namespace length_at_4kg_length_increases_by_2_relationship_linear_length_at_12kg_l2_2968

noncomputable def spring_length (x : ℝ) : ℝ :=
  2 * x + 18

-- Problem (1)
theorem length_at_4kg : (spring_length 4) = 26 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (2)
theorem length_increases_by_2 : ∀ (x y : ℝ), y = x + 1 → (spring_length y) = (spring_length x) + 2 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (3)
theorem relationship_linear : ∃ (k b : ℝ), (∀ x, spring_length x = k * x + b) ∧ k = 2 ∧ b = 18 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (4)
theorem length_at_12kg : (spring_length 12) = 42 :=
  by
    -- The complete proof is omitted.
    sorry

end length_at_4kg_length_increases_by_2_relationship_linear_length_at_12kg_l2_2968


namespace max_students_l2_2541

theorem max_students 
  (x : ℕ) 
  (h_lt : x < 100)
  (h_mod8 : x % 8 = 5) 
  (h_mod5 : x % 5 = 3) 
  : x = 93 := 
sorry

end max_students_l2_2541


namespace simplify_expression_l2_2696

theorem simplify_expression (a : ℝ) : 
  ( (a^(16 / 8))^(1 / 4) )^3 * ( (a^(16 / 4))^(1 / 8) )^3 = a^3 := by
  sorry

end simplify_expression_l2_2696


namespace opposite_of_3_is_neg3_l2_2820

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l2_2820


namespace supermarket_flour_import_l2_2274

theorem supermarket_flour_import :
  let long_grain_rice := (9 : ℚ) / 20
  let glutinous_rice := (7 : ℚ) / 20
  let combined_rice := long_grain_rice + glutinous_rice
  let less_amount := (3 : ℚ) / 20
  let flour : ℚ := combined_rice - less_amount
  flour = (13 : ℚ) / 20 :=
by
  sorry

end supermarket_flour_import_l2_2274


namespace evaluate_expression_l2_2929

theorem evaluate_expression : 5 - 7 * (8 - 3^2) * 4 = 33 :=
by
  sorry

end evaluate_expression_l2_2929


namespace minimum_pencils_needed_l2_2882

theorem minimum_pencils_needed (red_pencils blue_pencils : ℕ) (total_pencils : ℕ) 
  (h_red : red_pencils = 7) (h_blue : blue_pencils = 4) (h_total : total_pencils = red_pencils + blue_pencils) :
  (∃ n : ℕ, n = 8 ∧ n ≤ total_pencils ∧ (∀ m : ℕ, m < 8 → (m < red_pencils ∨ m < blue_pencils))) :=
by
  sorry

end minimum_pencils_needed_l2_2882


namespace hyperbola_center_l2_2744

theorem hyperbola_center : ∃ c : ℝ × ℝ, c = (3, 5) ∧
  ∀ x y : ℝ, 9 * x ^ 2 - 54 * x - 36 * y ^ 2 + 360 * y - 891 = 0 → (c.1 = 3 ∧ c.2 = 5) :=
by
  use (3, 5)
  sorry

end hyperbola_center_l2_2744


namespace cannot_form_set_l2_2250

/-- Define the set of non-negative real numbers not exceeding 20 --/
def setA : Set ℝ := {x | 0 ≤ x ∧ x ≤ 20}

/-- Define the set of solutions of the equation x^2 - 9 = 0 within the real numbers --/
def setB : Set ℝ := {x | x^2 - 9 = 0}

/-- Define the set of all students taller than 170 cm enrolled in a certain school in the year 2013 --/
def setC : Type := sorry

/-- Define the (pseudo) set of all approximate values of sqrt(3) --/
def pseudoSetD : Set ℝ := {x | x = Real.sqrt 3}

/-- Main theorem stating that setD cannot form a mathematically valid set --/
theorem cannot_form_set (x : ℝ) : x ∈ pseudoSetD → False := sorry

end cannot_form_set_l2_2250


namespace d_2_is_zero_l2_2039

noncomputable def d_4 : ℤ := 0
noncomputable def d_3 : ℤ := 1
noncomputable def d_2 : ℤ := 0
noncomputable def d_1 : ℤ := 0
noncomputable def d_0 : ℤ := 0

def p (x : ℤ) : ℤ := d_4 * x^4 + d_3 * x^3 + d_2 * x^2 + d_1 * x + d_0

theorem d_2_is_zero :
  ∀ m : ℤ, m ≥ 3 → E(m) = p(m) → d_2 = 0 :=
by
  intros m hm heq
  -- reasonable proof steps would follow here
  simp at heq
  sorry

end d_2_is_zero_l2_2039


namespace sum_of_cubes_l2_2203

def cubic_eq (x : ℝ) : Prop := x^3 - 2 * x^2 + 3 * x - 4 = 0

variables (a b c : ℝ)

axiom a_root : cubic_eq a
axiom b_root : cubic_eq b
axiom c_root : cubic_eq c

axiom sum_roots : a + b + c = 2
axiom sum_products_roots : a * b + a * c + b * c = 3
axiom product_roots : a * b * c = 4

theorem sum_of_cubes : a^3 + b^3 + c^3 = 2 :=
by
  sorry

end sum_of_cubes_l2_2203


namespace total_distance_travelled_l2_2050

theorem total_distance_travelled (D : ℝ) (h1 : (D / 2) / 30 + (D / 2) / 25 = 11) : D = 150 :=
sorry

end total_distance_travelled_l2_2050


namespace min_value_a_plus_b_l2_2169

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + b = 2 * a * b) : a + b ≥ 2 + Real.sqrt 3 :=
sorry

end min_value_a_plus_b_l2_2169


namespace calculate_product_sum_l2_2584

theorem calculate_product_sum :
  17 * (17/18) + 35 * (35/36) = 50 + 1/12 :=
by sorry

end calculate_product_sum_l2_2584


namespace percent_of_a_is_b_l2_2623

theorem percent_of_a_is_b (a b c : ℝ) (h1 : c = 0.30 * a) (h2 : c = 0.25 * b) : b = 1.2 * a :=
by
  -- proof 
  sorry

end percent_of_a_is_b_l2_2623


namespace chelsea_sugar_bags_l2_2732

variable (n : ℕ)

-- Defining the conditions as hypotheses
def initial_sugar : ℕ := 24
def remaining_sugar : ℕ := 21
def sugar_lost : ℕ := initial_sugar - remaining_sugar
def torn_bag_sugar : ℕ := 2 * sugar_lost

-- Define the statement to prove
theorem chelsea_sugar_bags :
  n = initial_sugar / torn_bag_sugar → n = 4 :=
by
  sorry

end chelsea_sugar_bags_l2_2732


namespace smallest_n_l2_2380

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 4 * n = k^2) (h2 : ∃ l : ℕ, 5 * n = l^3) : n = 100 :=
sorry

end smallest_n_l2_2380


namespace conor_total_vegetables_l2_2733

-- Definitions for each day of the week
def vegetables_per_day_mon_wed : Nat := 12 + 9 + 8 + 15 + 7
def vegetables_per_day_thu_sat : Nat := 7 + 5 + 4 + 10 + 4
def total_vegetables : Nat := 3 * vegetables_per_day_mon_wed + 3 * vegetables_per_day_thu_sat

-- Lean statement for the proof problem
theorem conor_total_vegetables : total_vegetables = 243 := by
  sorry

end conor_total_vegetables_l2_2733


namespace probability_red_or_black_probability_red_black_or_white_l2_2634

-- We define the probabilities of events A, B, and C
def P_A : ℚ := 5 / 12
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 6

-- Define the probability of event D for completeness
def P_D : ℚ := 1 / 12

-- 1. Statement for the probability of drawing a red or black ball (P(A ⋃ B))
theorem probability_red_or_black :
  (P_A + P_B = 3 / 4) :=
by
  sorry

-- 2. Statement for the probability of drawing a red, black, or white ball (P(A ⋃ B ⋃ C))
theorem probability_red_black_or_white :
  (P_A + P_B + P_C = 11 / 12) :=
by
  sorry

end probability_red_or_black_probability_red_black_or_white_l2_2634


namespace rug_area_is_180_l2_2127

variables (w l : ℕ)

def length_eq_width_plus_eight (l w : ℕ) : Prop :=
  l = w + 8

def uniform_width_between_rug_and_room (d : ℕ) : Prop :=
  d = 8

def area_uncovered_by_rug (area : ℕ) : Prop :=
  area = 704

def area_of_rug (w l : ℕ) : ℕ :=
  l * w

theorem rug_area_is_180 (w l : ℕ) (hwld : length_eq_width_plus_eight l w)
  (huw : uniform_width_between_rug_and_room 8)
  (huar : area_uncovered_by_rug 704) :
  area_of_rug w l = 180 :=
sorry

end rug_area_is_180_l2_2127


namespace interval_contains_root_l2_2429

-- Define the function in question
def func (x : ℝ) : ℝ := 2^x + x

theorem interval_contains_root :
  ∃ x ∈ Ioo (-1 : ℝ) (-1/2 : ℝ), func x = 0 :=
by
  -- Here, we would use the Intermediate Value Theorem and the evaluations at the end points
  -- Since it's just a statement, we write sorry for the proof
  sorry

end interval_contains_root_l2_2429


namespace opposite_of_3_l2_2852

-- Define the concept of opposite of a number and the logic for positive numbers
def opposite (x : Int) : Int := 
  if x > 0 then -x
  else if x < 0 then -x
  else 0

-- Statement to prove that the opposite of 3 is -3
theorem opposite_of_3 : opposite 3 = -3 :=
by 
  -- Using the definition of opposite
  unfold opposite
  -- Simplify the expression for x = 3
  simp [lt_irrefl, int.coe_nat_lt]
  -- Conclude proof
  rfl

end opposite_of_3_l2_2852


namespace tan_seven_pi_over_four_l2_2292

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by
  sorry

end tan_seven_pi_over_four_l2_2292


namespace four_distinct_real_roots_l2_2318

theorem four_distinct_real_roots (m : ℝ) : 
  (∀ x : ℝ, |(x-1)*(x-3)| = m*x → ∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ↔ 
  0 < m ∧ m < 4 - 2 * Real.sqrt 3 :=
by
  sorry

end four_distinct_real_roots_l2_2318


namespace problem_statement_l2_2008

open Complex

theorem problem_statement :
  (3 - I) / (2 + I) = 1 - I :=
by
  sorry

end problem_statement_l2_2008


namespace four_ab_eq_four_l2_2326

theorem four_ab_eq_four {a b : ℝ} (h : a * b = 1) : 4 * a * b = 4 :=
by
  sorry

end four_ab_eq_four_l2_2326


namespace cubic_roots_quadratic_l2_2452

theorem cubic_roots_quadratic (A B C p : ℚ)
  (hA : A ≠ 0)
  (h1 : (∀ x : ℚ, A * x^2 + B * x + C = 0 ↔ x = (root1) ∨ x = (root2)))
  (h2 : root1 + root2 = - B / A)
  (h3 : root1 * root2 = C / A)
  (new_eq : ∀ x : ℚ, x^2 + p*x + q = 0 ↔ x = root1^3 ∨ x = root2^3) :
  p = (B^3 - 3 * A * B * C) / A^3 :=
by
  sorry

end cubic_roots_quadratic_l2_2452


namespace measure_of_angle_A_l2_2724

-- Define the given conditions
variables (A B : ℝ)
axiom supplementary : A + B = 180
axiom measure_rel : A = 7 * B

-- The theorem statement to prove
theorem measure_of_angle_A : A = 157.5 :=
by
  -- proof steps would go here, but are omitted
  sorry

end measure_of_angle_A_l2_2724


namespace increasing_interval_l2_2065

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval : {x : ℝ | 2 < x} = { x : ℝ | (x - 3) * Real.exp x > 0 } :=
by
  sorry

end increasing_interval_l2_2065


namespace fraction_simplification_l2_2148

theorem fraction_simplification : 
  (1877^2 - 1862^2) / (1880^2 - 1859^2) = 5 / 7 := 
by 
  sorry

end fraction_simplification_l2_2148


namespace fourth_person_height_l2_2688

theorem fourth_person_height 
  (H : ℕ) 
  (h_avg : (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 79) : 
  H + 10 = 85 :=
by
  sorry

end fourth_person_height_l2_2688


namespace exists_special_number_divisible_by_1991_l2_2052

theorem exists_special_number_divisible_by_1991 :
  ∃ (N : ℤ) (n : ℕ), n > 2 ∧ (N % 1991 = 0) ∧ 
  (∃ a b x : ℕ, N = 10 ^ (n + 1) * a + 10 ^ n * x + 9 * 10 ^ (n - 1) + b) :=
sorry

end exists_special_number_divisible_by_1991_l2_2052


namespace number_of_possible_values_of_a_l2_2804

def is_factor (m n : ℕ) : Prop := n % m = 0

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem number_of_possible_values_of_a :
  (∃ a : ℕ, is_factor 3 a ∧ is_factor a 18 ∧ even (sum_of_digits a)) →
  (π : ℕ) : ℕ :=
begin
  cases h with a,
  exact 1,
end

end number_of_possible_values_of_a_l2_2804


namespace reduced_price_per_kg_l2_2722

variable (P : ℝ)
variable (R : ℝ)
variable (Q : ℝ)

theorem reduced_price_per_kg
  (h1 : R = 0.75 * P)
  (h2 : 500 = Q * P)
  (h3 : 500 = (Q + 5) * R)
  (h4 : Q = 500 / P) :
  R = 25 :=
by
  sorry

end reduced_price_per_kg_l2_2722


namespace binary_subtraction_l2_2145

theorem binary_subtraction : ∀ (x y : ℕ), x = 0b11011 → y = 0b101 → x - y = 0b10110 :=
by
  sorry

end binary_subtraction_l2_2145


namespace ned_defuse_time_l2_2211

theorem ned_defuse_time (flights_total time_per_flight bomb_time time_spent : ℕ) (h1 : flights_total = 20) (h2 : time_per_flight = 11) (h3 : bomb_time = 72) (h4 : time_spent = 165) :
  bomb_time - (flights_total * time_per_flight - time_spent) / time_per_flight * time_per_flight = 17 := by
  sorry

end ned_defuse_time_l2_2211


namespace function_relationship_selling_price_for_profit_max_profit_l2_2888

-- Step (1): Prove the function relationship between y and x
theorem function_relationship (x y: ℝ) (h1 : ∀ x, y = -2*x + 80)
  (h2 : x = 22 ∧ y = 36 ∨ x = 24 ∧ y = 32) :
  y = -2*x + 80 := by
  sorry

-- Step (2): Selling price per book for a 150 yuan profit per week
theorem selling_price_for_profit (x: ℝ) (hx : 20 ≤ x ∧ x ≤ 28) (profit : ℝ)
  (h_profit : profit = (x - 20) * (-2*x + 80)) (h2 : profit = 150) : 
  x = 25 := by
  sorry

-- Step (3): Maximizing the weekly profit
theorem max_profit (x w: ℝ) (hx : 20 ≤ x ∧ x ≤ 28) 
  (profit : ∀ x, w = (x - 20) * (-2*x + 80)) :
  w = 192 ∧ x = 28 := by
  sorry

end function_relationship_selling_price_for_profit_max_profit_l2_2888


namespace peter_completes_remaining_work_in_14_days_l2_2650

-- Define the conditions and the theorem
variable (W : ℕ) (work_done : ℕ) (remaining_work : ℕ)

theorem peter_completes_remaining_work_in_14_days
  (h1 : Matt_and_Peter_rate = (W/20))
  (h2 : Peter_rate = (W/35))
  (h3 : Work_done_in_12_days = (12 * (W/20)))
  (h4 : Remaining_work = (W - (12 * (W/20))))
  : (remaining_work / Peter_rate)  = 14 := sorry

end peter_completes_remaining_work_in_14_days_l2_2650


namespace page_shoes_l2_2801

/-- Page's initial collection of shoes -/
def initial_collection : ℕ := 80

/-- Page donates 30% of her collection -/
def donation (n : ℕ) : ℕ := n * 30 / 100

/-- Page buys additional shoes -/
def additional_shoes : ℕ := 6

/-- Page's final collection after donation and purchase -/
def final_collection (n : ℕ) : ℕ := (n - donation n) + additional_shoes

/-- Proof that the final collection of shoes is 62 given the initial collection of 80 pairs -/
theorem page_shoes : (final_collection initial_collection) = 62 := 
by sorry

end page_shoes_l2_2801


namespace temperature_difference_on_day_xianning_l2_2213

theorem temperature_difference_on_day_xianning 
  (highest_temp : ℝ) (lowest_temp : ℝ) 
  (h_highest : highest_temp = 2) (h_lowest : lowest_temp = -3) : 
  highest_temp - lowest_temp = 5 := 
by
  sorry

end temperature_difference_on_day_xianning_l2_2213


namespace simplify_expression_eq_square_l2_2900

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l2_2900


namespace sum_of_coefficients_l2_2183

theorem sum_of_coefficients (a_5 a_4 a_3 a_2 a_1 a_0 : ℤ) :
  (x-2)^5 = a_5*x^5 + a_4*x^4 + a_3*x^3 + a_2*x^2 + a_1*x + a_0 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  sorry

end sum_of_coefficients_l2_2183


namespace find_julios_bonus_l2_2784

def commission (customers: ℕ) : ℕ :=
  customers * 1

def total_commission (week1: ℕ) (week2: ℕ) (week3: ℕ) : ℕ :=
  commission week1 + commission week2 + commission week3

noncomputable def julios_bonus (total_earnings salary total_commission: ℕ) : ℕ :=
  total_earnings - salary - total_commission

theorem find_julios_bonus :
  let week1 := 35
  let week2 := 2 * week1
  let week3 := 3 * week1
  let salary := 500
  let total_earnings := 760
  let total_comm := total_commission week1 week2 week3
  julios_bonus total_earnings salary total_comm = 50 :=
by
  sorry

end find_julios_bonus_l2_2784


namespace initial_bananas_proof_l2_2047

noncomputable def initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) : ℕ :=
  (extra_bananas * (total_children - absent_children)) / (total_children - extra_bananas)

theorem initial_bananas_proof
  (total_children : ℕ)
  (absent_children : ℕ)
  (extra_bananas : ℕ)
  (h_total : total_children = 640)
  (h_absent : absent_children = 320)
  (h_extra : extra_bananas = 2) : initial_bananas_per_child total_children absent_children extra_bananas = 2 :=
by
  sorry

end initial_bananas_proof_l2_2047


namespace alpha_beta_diff_l2_2443

theorem alpha_beta_diff 
  (α β : ℝ)
  (h1 : α + β = 17)
  (h2 : α * β = 70) : |α - β| = 3 :=
by
  sorry

end alpha_beta_diff_l2_2443


namespace brownies_each_l2_2940

theorem brownies_each (num_columns : ℕ) (num_rows : ℕ) (total_people : ℕ) (total_brownies : ℕ) 
(h1 : num_columns = 6) (h2 : num_rows = 3) (h3 : total_people = 6) 
(h4 : total_brownies = num_columns * num_rows) : 
total_brownies / total_people = 3 := 
by
  -- Placeholder for the actual proof
  sorry

end brownies_each_l2_2940


namespace quadratic_equation_solutions_l2_2070

theorem quadratic_equation_solutions : ∀ x : ℝ, x^2 - 2 * x = 0 ↔ (x = 0 ∨ x = 2) := 
by sorry

end quadratic_equation_solutions_l2_2070


namespace gcd_is_3_l2_2428

def gcd_6273_14593 : ℕ := Nat.gcd 6273 14593

theorem gcd_is_3 : gcd_6273_14593 = 3 :=
by
  sorry

end gcd_is_3_l2_2428


namespace kevin_total_distance_l2_2036

theorem kevin_total_distance :
  let d1 := 10 * 0.5,
      d2 := 20 * 0.5,
      d3 := 8 * 0.25 in
  d1 + d2 + d3 = 17 := by
  sorry

end kevin_total_distance_l2_2036


namespace dime_probability_l2_2268

theorem dime_probability (dime_value quarter_value : ℝ) (dime_worth quarter_worth total_coins: ℕ) :
  dime_value = 0.10 ∧
  quarter_value = 0.25 ∧
  dime_worth = 10 ∧
  quarter_worth = 4 ∧
  total_coins = 14 →
  (dime_worth / total_coins : ℝ) = 5 / 7 :=
by
  sorry

end dime_probability_l2_2268


namespace largest_perfect_square_factor_1760_l2_2548

theorem largest_perfect_square_factor_1760 :
  ∃ n, (∃ k, n = k^2) ∧ n ∣ 1760 ∧ ∀ m, (∃ j, m = j^2) ∧ m ∣ 1760 → m ≤ n := by
  sorry

end largest_perfect_square_factor_1760_l2_2548


namespace female_students_proportion_and_count_l2_2652

noncomputable def num_students : ℕ := 30
noncomputable def num_male_students : ℕ := 8
noncomputable def overall_avg_score : ℚ := 90
noncomputable def male_avg_scores : (ℚ × ℚ × ℚ) := (87, 95, 89)
noncomputable def female_avg_scores : (ℚ × ℚ × ℚ) := (92, 94, 91)
noncomputable def avg_attendance_alg_geom : ℚ := 0.85
noncomputable def avg_attendance_calc : ℚ := 0.89

theorem female_students_proportion_and_count :
  ∃ (F : ℕ), F = num_students - num_male_students ∧ (F / num_students : ℚ) = 11 / 15 :=
by
  sorry

end female_students_proportion_and_count_l2_2652


namespace masha_problem_l2_2737

noncomputable def sum_arithmetic_series (a l n : ℕ) : ℕ :=
  (n * (a + l)) / 2

theorem masha_problem : 
  let a_even := 372
  let l_even := 506
  let n_even := 67
  let a_odd := 373
  let l_odd := 505
  let n_odd := 68
  let S_even := sum_arithmetic_series a_even l_even n_even
  let S_odd := sum_arithmetic_series a_odd l_odd n_odd
  S_odd - S_even = 439 := 
by sorry

end masha_problem_l2_2737


namespace line_slope_is_neg_half_l2_2676

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := x + 2 * y - 4 = 0

-- The main theorem to be proved
theorem line_slope_is_neg_half : ∀ (x y : ℝ), line_eq x y → (∃ m b : ℝ, y = m * x + b ∧ m = -1/2) := by
  sorry

end line_slope_is_neg_half_l2_2676


namespace irreducible_fraction_l2_2656

-- Definition of gcd
def my_gcd (m n : Int) : Int :=
  gcd m n

-- Statement of the problem
theorem irreducible_fraction (a : Int) : my_gcd (a^3 + 2 * a) (a^4 + 3 * a^2 + 1) = 1 :=
by
  sorry

end irreducible_fraction_l2_2656


namespace range_of_m_l2_2004

noncomputable def isEllipse (m : ℝ) : Prop := (m^2 > 2 * m + 8) ∧ (2 * m + 8 > 0)
noncomputable def intersectsXAxisAtTwoPoints (m : ℝ) : Prop := (2 * m - 3)^2 - 1 > 0

theorem range_of_m (m : ℝ) :
  ((m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0 ∨ (2 * m - 3)^2 - 1 > 0) ∧
  ¬ (m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0 ∧ (2 * m - 3)^2 - 1 > 0)) →
  (m ≤ -4 ∨ (-2 ≤ m ∧ m < 1) ∨ (2 < m ∧ m ≤ 4)) :=
by sorry

end range_of_m_l2_2004


namespace balance_five_diamonds_bullets_l2_2746

variables (a b c : ℝ)

-- Conditions
def condition1 : Prop := 4 * a + 2 * b = 12 * c
def condition2 : Prop := 2 * a = b + 4 * c

-- Theorem statement
theorem balance_five_diamonds_bullets (h1 : condition1 a b c) (h2 : condition2 a b c) : 5 * b = 5 * c :=
by
  sorry

end balance_five_diamonds_bullets_l2_2746


namespace simplify_expression_eq_square_l2_2901

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l2_2901


namespace min_value_f_l2_2319

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (exp x - 1) + x

theorem min_value_f {x0 : ℝ} (hx0 : 0 < x0) (hx0_min : ∀ x > 0, f x ≥ f x0) :
  f x0 = x0 + 1 ∧ f x0 < 3 :=
by sorry

end min_value_f_l2_2319


namespace marble_problem_l2_2098

-- Define the initial number of marbles
def initial_marbles : Prop :=
  ∃ (x y : ℕ), (y - 4 = 2 * (x + 4)) ∧ (y + 2 = 11 * (x - 2)) ∧ (y = 20) ∧ (x = 4)

-- The main theorem to prove the initial number of marbles
theorem marble_problem (x y : ℕ) (cond1 : y - 4 = 2 * (x + 4)) (cond2 : y + 2 = 11 * (x - 2)) :
  y = 20 ∧ x = 4 :=
sorry

end marble_problem_l2_2098


namespace eliza_is_shorter_by_2_inch_l2_2741

theorem eliza_is_shorter_by_2_inch
  (total_height : ℕ)
  (height_sibling1 height_sibling2 height_sibling3 height_eliza : ℕ) :
  total_height = 330 →
  height_sibling1 = 66 →
  height_sibling2 = 66 →
  height_sibling3 = 60 →
  height_eliza = 68 →
  total_height - (height_sibling1 + height_sibling2 + height_sibling3 + height_eliza) - height_eliza = 2 :=
by
  sorry

end eliza_is_shorter_by_2_inch_l2_2741


namespace rectangle_area_l2_2670

theorem rectangle_area (L B r s : ℝ) (h1 : L = 5 * r)
                       (h2 : r = s)
                       (h3 : s^2 = 16)
                       (h4 : B = 11) :
  (L * B = 220) :=
by
  sorry

end rectangle_area_l2_2670


namespace min_workers_to_make_profit_l2_2118

theorem min_workers_to_make_profit :
  ∃ n : ℕ, 500 + 8 * 15 * n < 124 * n ∧ n = 126 :=
by
  sorry

end min_workers_to_make_profit_l2_2118


namespace chrysanthemums_arrangement_l2_2576

theorem chrysanthemums_arrangement :
  let varieties := ['A', 'B', 'C', 'D', 'E', 'F']
  let is_same_side (A B C : char) (perm : List char) := 
    (perm.indexOf A < perm.indexOf C ∧ perm.indexOf B < perm.indexOf C) ∨ 
    (perm.indexOf A > perm.indexOf C ∧ perm.indexOf B > perm.indexOf C)
  (list.length 
    (list.filter 
      (λ perm, is_same_side 'A' 'B' 'C' perm) 
      (list.perm.quotient varieties)))
  = 480 := sorry

end chrysanthemums_arrangement_l2_2576


namespace find_notebooks_l2_2384

theorem find_notebooks (S N : ℕ) (h1 : N = 4 * S + 3) (h2 : N + 6 = 5 * S) : N = 39 := 
by
  sorry 

end find_notebooks_l2_2384


namespace triangle_obtuse_l2_2771

def is_obtuse_triangle (A B C : ℝ) : Prop := A > 90 ∨ B > 90 ∨ C > 90

theorem triangle_obtuse (A B C : ℝ) (h1 : A > 3 * B) (h2 : C < 2 * B) (h3 : A + B + C = 180) : is_obtuse_triangle A B C :=
by sorry

end triangle_obtuse_l2_2771


namespace solve_x_l2_2184

theorem solve_x (x: ℝ) (h: -4 * x - 15 = 12 * x + 5) : x = -5 / 4 :=
sorry

end solve_x_l2_2184


namespace find_m_l2_2628

theorem find_m (m : ℕ) (h : 10^(m-1) < 2^512 ∧ 2^512 < 10^m): 
  m = 155 :=
sorry

end find_m_l2_2628


namespace opposite_of_three_l2_2858

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l2_2858


namespace min_x_y_l2_2313

open Real

theorem min_x_y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : x + y ≥ 16 := 
sorry

end min_x_y_l2_2313


namespace arithmetic_sequence_general_term_l2_2749

theorem arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h₁ : a 1 = 3)
  (h₂ : ∀ n ≥ 2, 2 * a n = S n * S (n - 1)) :
  (∃ d : ℚ, d = -1/2 ∧ ∀ n ≥ 2, (1 / S n) - (1 / S (n - 1)) = d) :=
sorry

theorem general_term (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h₁ : a 1 = 3)
  (h₂ : ∀ n ≥ 2, 2 * a n = S n * S (n - 1)) :
  ∀ n, a n = if n = 1 then 3 else 18 / ((8 - 3 * n) * (5 - 3 * n)) :=
sorry

end arithmetic_sequence_general_term_l2_2749


namespace calculate_AH_l2_2355

def square (a : ℝ) := a ^ 2
def area_square (s : ℝ) := s ^ 2
def area_triangle (b h : ℝ) := 0.5 * b * h

theorem calculate_AH (s DG DH AH : ℝ) 
  (h_square : area_square s = 144) 
  (h_area_triangle : area_triangle DG DH = 63)
  (h_perpendicular : DG = DH)
  (h_hypotenuse : square AH = square s + square DH) :
  AH = 3 * Real.sqrt 30 :=
by
  -- Proof would be provided here
  sorry

end calculate_AH_l2_2355


namespace problem_solution_l2_2758

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x - a

theorem problem_solution (x₀ x₁ a : ℝ) (h₁ : 3 * x₀^2 - 2 * x₀ + a = 0) (h₂ : f x₁ a = f x₀ a) (h₃ : x₁ ≠ x₀) : x₁ + 2 * x₀ = 1 :=
by
  sorry

end problem_solution_l2_2758


namespace ratio_Smax_Smin_l2_2057

-- Define the area of a cube's diagonal cross-section through BD1
def cross_section_area (a : ℝ) : ℝ := sorry

theorem ratio_Smax_Smin (a : ℝ) (S S_min S_max : ℝ) :
  cross_section_area a = S →
  S_min = (a^2 * Real.sqrt 6) / 2 →
  S_max = a^2 * Real.sqrt 6 →
  S_max / S_min = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end ratio_Smax_Smin_l2_2057


namespace true_proposition_among_ABCD_l2_2406

theorem true_proposition_among_ABCD : 
  (∀ x : ℝ, x^2 < x + 1) = false ∧
  (∀ x : ℝ, x^2 ≥ x + 1) = false ∧
  (∃ x : ℝ, ∀ y : ℝ, x * y^2 ≠ y^2) = true ∧
  (∀ x : ℝ, ∃ y : ℝ, x > y^2) = false :=
by 
  sorry

end true_proposition_among_ABCD_l2_2406


namespace bucket_ratio_l2_2977

theorem bucket_ratio :
  ∀ (leak_rate : ℚ) (duration : ℚ) (bucket_capacity : ℚ),
  leak_rate = 1.5 ∧ duration = 12 ∧ bucket_capacity = 36 →
  bucket_capacity / (leak_rate * duration) = 2 :=
by
  intros leak_rate duration bucket_capacity h
  have h_rate := h.1
  have h_duration := h.2.1
  have h_capacity := h.2.2
  rw [h_rate, h_duration, h_capacity]
  norm_num
  sorry

end bucket_ratio_l2_2977


namespace find_S_l2_2990

theorem find_S :
  (1/4 : ℝ) * (1/6 : ℝ) * S = (1/5 : ℝ) * (1/8 : ℝ) * 160 → S = 96 :=
by
  intro h
  -- Proof is omitted
  sorry 

end find_S_l2_2990


namespace simplify_expression_l2_2909

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l2_2909


namespace opposite_of_3_l2_2837

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l2_2837


namespace range_of_f_l2_2069

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (Real.cos x) + Real.arccos (Real.sin x)

theorem range_of_f : Set.range f = Set.Icc 0 Real.pi :=
sorry

end range_of_f_l2_2069


namespace maximum_perimeter_l2_2734

noncomputable def triangle_base : ℝ := 10.0
noncomputable def triangle_height : ℝ := 12.0
noncomputable def segment_length : ℝ := 1.25

def distance (x y : ℝ) : ℝ := sqrt (x^2 + y^2)

def perimeter (k : ℕ) : ℝ :=
  segment_length + distance 12 (k : ℝ) + distance 12 (k + 1)

theorem maximum_perimeter : ∃ k : ℕ, k < 8 ∧ perimeter k = 26.27 :=
  sorry

end maximum_perimeter_l2_2734


namespace total_surface_area_of_tower_l2_2658

def volume_to_side_length (v : ℕ) : ℕ :=
  nat.cbrt v

def surface_area (s : ℕ) : ℕ :=
  6 * s^2

def adjusted_surface_area (s : ℕ) : ℕ :=
  if s > 1 then surface_area s - s^2 else surface_area s

theorem total_surface_area_of_tower :
  let side_lengths := [7, 6, 5, 4, 3, 2, 1].map volume_to_side_length in
  let surface_areas := side_lengths.map adjusted_surface_area in
  surface_areas.sum = 701 :=
by
  sorry

end total_surface_area_of_tower_l2_2658


namespace romeo_total_profit_is_55_l2_2515

-- Defining the conditions
def number_of_bars : ℕ := 5
def cost_per_bar : ℕ := 5
def packaging_cost_per_bar : ℕ := 2
def total_selling_price : ℕ := 90

-- Defining the profit calculation
def total_cost_per_bar := cost_per_bar + packaging_cost_per_bar
def selling_price_per_bar := total_selling_price / number_of_bars
def profit_per_bar := selling_price_per_bar - total_cost_per_bar
def total_profit := profit_per_bar * number_of_bars

-- Proving the total profit
theorem romeo_total_profit_is_55 : total_profit = 55 :=
by
  sorry

end romeo_total_profit_is_55_l2_2515


namespace minimum_y_squared_l2_2174

theorem minimum_y_squared :
  let consecutive_sum (x : ℤ) := (x-5)^2 + (x-4)^2 + (x-3)^2 + (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2
  ∃ y : ℤ, y^2 = 11 * (1^2 + 10) ∧ ∀ z : ℤ, z^2 = 11 * consecutive_sum z → y^2 ≤ z^2 := by
sorry

end minimum_y_squared_l2_2174


namespace trigonometric_identity_l2_2751

theorem trigonometric_identity (α : ℝ) (h : Real.sin (3 * Real.pi - α) = 2 * Real.sin (Real.pi / 2 + α)) : 
  (Real.sin (Real.pi - α) ^ 3 - Real.sin (Real.pi / 2 - α)) / 
  (3 * Real.cos (Real.pi / 2 + α) + 2 * Real.cos (Real.pi + α)) = -3/40 :=
by
  sorry

end trigonometric_identity_l2_2751
