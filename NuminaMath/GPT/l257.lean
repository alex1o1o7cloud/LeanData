import Mathlib

namespace gcf_270_108_150_l257_25778

theorem gcf_270_108_150 : Nat.gcd (Nat.gcd 270 108) 150 = 30 := 
  sorry

end gcf_270_108_150_l257_25778


namespace parabola_vertex_l257_25791

-- Define the parabola equation
def parabola_equation (x : ℝ) : ℝ := (x - 2)^2 + 5

-- State the theorem to find the vertex
theorem parabola_vertex : ∃ h k : ℝ, ∀ x : ℝ, parabola_equation x = (x - h)^2 + k ∧ h = 2 ∧ k = 5 :=
by
  sorry

end parabola_vertex_l257_25791


namespace percent_profit_l257_25711

variable (C S : ℝ)

theorem percent_profit (h : 72 * C = 60 * S) : ((S - C) / C) * 100 = 20 := by
  sorry

end percent_profit_l257_25711


namespace total_number_of_coins_l257_25777

theorem total_number_of_coins (n : ℕ) (h : 4 * n - 4 = 240) : n^2 = 3721 :=
by
  sorry

end total_number_of_coins_l257_25777


namespace sean_total_apples_l257_25789

-- Define initial apples
def initial_apples : Nat := 9

-- Define the number of apples Susan gives each day
def apples_per_day : Nat := 8

-- Define the number of days Susan gives apples
def number_of_days : Nat := 5

-- Calculate total apples given by Susan
def total_apples_given : Nat := apples_per_day * number_of_days

-- Define the final total apples
def total_apples : Nat := initial_apples + total_apples_given

-- Prove the number of total apples is 49
theorem sean_total_apples : total_apples = 49 := by
  sorry

end sean_total_apples_l257_25789


namespace time_diff_is_6_l257_25704

-- Define the speeds for the different sails
def speed_of_large_sail : ℕ := 50
def speed_of_small_sail : ℕ := 20

-- Define the distance of the trip
def trip_distance : ℕ := 200

-- Calculate the time for each sail
def time_large_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed
def time_small_sail (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Define the time difference
def time_difference (distance : ℕ) (speed_large : ℕ) (speed_small : ℕ) : ℕ := 
  (distance / speed_small) - (distance / speed_large)

-- Prove that the time difference between the large and small sails is 6 hours
theorem time_diff_is_6 : time_difference trip_distance speed_of_large_sail speed_of_small_sail = 6 := by
  -- useful := time_difference trip_distance speed_of_large_sail speed_of_small_sail,
  -- change useful with 6,
  sorry

end time_diff_is_6_l257_25704


namespace number_of_integers_l257_25744

theorem number_of_integers (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 2020) (h3 : ∃ k : ℕ, n^n = k^2) : n = 1032 :=
sorry

end number_of_integers_l257_25744


namespace quadratic_inequality_solution_l257_25732

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end quadratic_inequality_solution_l257_25732


namespace louis_age_l257_25784

variable (L J M : ℕ) -- L for Louis, J for Jerica, and M for Matilda

theorem louis_age : 
  (M = 35) ∧ (M = J + 7) ∧ (J = 2 * L) → L = 14 := 
by 
  intro h 
  sorry

end louis_age_l257_25784


namespace vasya_kolya_difference_impossible_l257_25710

theorem vasya_kolya_difference_impossible : 
  ∀ k v : ℕ, (∃ q₁ q₂ : ℕ, 14400 = q₁ * 2 + q₂ * 2 + 1 + 1) → ¬ ∃ k, ∃ v, (v - k = 11 ∧ 14400 = k * q₁ + v * q₂) :=
by sorry

end vasya_kolya_difference_impossible_l257_25710


namespace perimeter_of_square_l257_25781

-- Defining the square with area
structure Square where
  side_length : ℝ
  area : ℝ

-- Defining a constant square with given area 625
def givenSquare : Square := 
  { side_length := 25, -- will square root the area of 625
    area := 625 }

-- Defining the function to calculate the perimeter of the square
noncomputable def perimeter (s : Square) : ℝ :=
  4 * s.side_length

-- The theorem stating that the perimeter of the given square with area 625 is 100
theorem perimeter_of_square : perimeter givenSquare = 100 := 
sorry

end perimeter_of_square_l257_25781


namespace bridge_length_l257_25706

-- Defining the problem based on the given conditions and proof goal
theorem bridge_length (L : ℝ) 
  (h1 : L / 4 + L / 3 + 120 = L) :
  L = 288 :=
sorry

end bridge_length_l257_25706


namespace Emma_investment_l257_25793

-- Define the necessary context and variables
variable (E : ℝ) -- Emma's investment
variable (B : ℝ := 500) -- Briana's investment which is a known constant
variable (ROI_Emma : ℝ := 0.30 * E) -- Emma's return on investment after 2 years
variable (ROI_Briana : ℝ := 0.20 * B) -- Briana's return on investment after 2 years
variable (ROI_difference : ℝ := ROI_Emma - ROI_Briana) -- The difference in their ROI

theorem Emma_investment :
  ROI_difference = 10 → E = 366.67 :=
by
  intros h
  sorry

end Emma_investment_l257_25793


namespace platform_length_proof_l257_25752

noncomputable def train_length : ℝ := 480

noncomputable def speed_kmph : ℝ := 55

noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600

noncomputable def crossing_time : ℝ := 71.99424046076314

noncomputable def total_distance_covered : ℝ := speed_mps * crossing_time

noncomputable def platform_length : ℝ := total_distance_covered - train_length

theorem platform_length_proof : platform_length = 620 := by
  sorry

end platform_length_proof_l257_25752


namespace evaluate_given_condition_l257_25792

noncomputable def evaluate_expression (b : ℚ) : ℚ :=
  (7 * b^2 - 15 * b + 5) * (3 * b - 4)

theorem evaluate_given_condition (b : ℚ) (h : b = 4 / 3) : evaluate_expression b = 0 := by
  sorry

end evaluate_given_condition_l257_25792


namespace mul_eight_neg_half_l257_25715

theorem mul_eight_neg_half : 8 * (- (1/2: ℚ)) = -4 := 
by 
  sorry

end mul_eight_neg_half_l257_25715


namespace slope_perpendicular_l257_25719

theorem slope_perpendicular (x1 y1 x2 y2 m : ℚ) 
  (hx1 : x1 = 3) (hy1 : y1 = -4) (hx2 : x2 = -6) (hy2 : y2 = 2) 
  (hm : m = (y2 - y1) / (x2 - x1)) :
  ∀ m_perpendicular: ℚ, m_perpendicular = (-1 / m) → m_perpendicular = 3/2 := 
sorry

end slope_perpendicular_l257_25719


namespace find_m_l257_25749

theorem find_m (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((1/3 : ℝ) * x1^3 - 3 * x1 + m = 0) ∧ ((1/3 : ℝ) * x2^3 - 3 * x2 + m = 0)) ↔ (m = -2 * Real.sqrt 3 ∨ m = 2 * Real.sqrt 3) :=
sorry

end find_m_l257_25749


namespace probability_of_divisor_of_6_is_two_thirds_l257_25717

noncomputable def probability_divisor_of_6 : ℚ :=
  have divisors_of_6 : Finset ℕ := {1, 2, 3, 6}
  have total_possible_outcomes : ℕ := 6
  have favorable_outcomes : ℕ := 4
  have probability_event : ℚ := favorable_outcomes / total_possible_outcomes
  2 / 3

theorem probability_of_divisor_of_6_is_two_thirds :
  probability_divisor_of_6 = 2 / 3 :=
sorry

end probability_of_divisor_of_6_is_two_thirds_l257_25717


namespace smallest_part_of_division_l257_25796

theorem smallest_part_of_division (x : ℝ) (h : 2 * x + (1/2) * x + (1/4) * x = 105) : 
  (1/4) * x = 10.5 :=
sorry

end smallest_part_of_division_l257_25796


namespace amusement_park_l257_25702

theorem amusement_park
  (A : ℕ)
  (adult_ticket_cost : ℕ := 22)
  (child_ticket_cost : ℕ := 7)
  (num_children : ℕ := 2)
  (total_cost : ℕ := 58)
  (cost_eq : adult_ticket_cost * A + child_ticket_cost * num_children = total_cost) :
  A = 2 :=
by {
  sorry
}

end amusement_park_l257_25702


namespace problem_solution_l257_25754

theorem problem_solution :
  ∀ p q : ℝ, (3 * p ^ 2 - 5 * p - 21 = 0) → (3 * q ^ 2 - 5 * q - 21 = 0) →
  (9 * p ^ 3 - 9 * q ^ 3) * (p - q)⁻¹ = 88 :=
by 
  sorry

end problem_solution_l257_25754


namespace first_digit_base5_of_312_is_2_l257_25738

theorem first_digit_base5_of_312_is_2 :
  ∃ d : ℕ, d = 2 ∧ (∀ n : ℕ, d * 5 ^ n ≤ 312 ∧ 312 < (d + 1) * 5 ^ n) :=
by
  sorry

end first_digit_base5_of_312_is_2_l257_25738


namespace find_covered_number_l257_25751

theorem find_covered_number (a x : ℤ) (h : (x - a) / 2 = x + 3) (hx : x = -7) : a = 1 := by
  sorry

end find_covered_number_l257_25751


namespace point_on_y_axis_l257_25798

theorem point_on_y_axis (m : ℝ) (M : ℝ × ℝ) (hM : M = (m + 1, m + 3)) (h_on_y_axis : M.1 = 0) : M = (0, 2) :=
by
  -- Proof omitted
  sorry

end point_on_y_axis_l257_25798


namespace abs_div_one_add_i_by_i_l257_25724

noncomputable def imaginary_unit : ℂ := Complex.I

/-- The absolute value of the complex number (1 + i)/i is √2. -/
theorem abs_div_one_add_i_by_i : Complex.abs ((1 + imaginary_unit) / imaginary_unit) = Real.sqrt 2 := by
  sorry

end abs_div_one_add_i_by_i_l257_25724


namespace triangle_equilateral_if_abs_eq_zero_l257_25771

theorem triangle_equilateral_if_abs_eq_zero (a b c : ℝ) (h : abs (a - b) + abs (b - c) = 0) : a = b ∧ b = c :=
by
  sorry

end triangle_equilateral_if_abs_eq_zero_l257_25771


namespace correct_operation_l257_25705

theorem correct_operation :
  ¬(a^2 * a^3 = a^6) ∧ ¬(6 * a / (3 * a) = 2 * a) ∧ ¬(2 * a^2 + 3 * a^3 = 5 * a^5) ∧ (-a * b^2)^2 = a^2 * b^4 :=
by
  sorry

end correct_operation_l257_25705


namespace correct_equation_l257_25769

theorem correct_equation (x : ℕ) (h : x ≤ 26) :
    let a_parts := 2100
    let b_parts := 1200
    let total_workers := 26
    let a_rate := 30
    let b_rate := 20
    let type_a_time := (a_parts : ℚ) / (a_rate * x)
    let type_b_time := (b_parts : ℚ) / (b_rate * (total_workers - x))
    type_a_time = type_b_time :=
by
    sorry

end correct_equation_l257_25769


namespace Seokgi_candies_l257_25742

theorem Seokgi_candies (C : ℕ) 
  (h1 : C / 2 + (C - C / 2) / 3 + 12 = C)
  (h2 : ∃ x, x = 12) :
  C = 36 := 
by 
  sorry

end Seokgi_candies_l257_25742


namespace ratio_of_areas_l257_25716

theorem ratio_of_areas (r : ℝ) (s1 s2 : ℝ) 
  (h1 : s1^2 = 4 / 5 * r^2)
  (h2 : s2^2 = 2 * r^2) :
  (s1^2 / s2^2) = 2 / 5 := by
  sorry

end ratio_of_areas_l257_25716


namespace reflection_image_l257_25757

theorem reflection_image (m b : ℝ) 
  (h1 : ∀ x y : ℝ, (x, y) = (0, 1) → (4, 5) = (2 * ((x + (m * y - y + b))/ (1 + m^2)) - x, 2 * ((y + (m * x - x + b)) / (1 + m^2)) - y))
  : m + b = 4 :=
sorry

end reflection_image_l257_25757


namespace mike_eggs_basket_l257_25731

theorem mike_eggs_basket : ∃ k : ℕ, (30 % k = 0) ∧ (42 % k = 0) ∧ k ≥ 4 ∧ (30 / k) ≥ 3 ∧ (42 / k) ≥ 3 ∧ k = 6 := 
by
  -- skipping the proof
  sorry

end mike_eggs_basket_l257_25731


namespace a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b_l257_25795

theorem a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b
  (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a ^ 2 < 0 → a < b) ∧ 
  (¬∀ a b : ℝ, a < b → (a - b) * a ^ 2 < 0) :=
sorry

end a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b_l257_25795


namespace parametric_equations_solution_l257_25721

theorem parametric_equations_solution (t₁ t₂ : ℝ) : 
  (1 = 1 + 2 * t₁ ∧ 2 = 2 - 3 * t₁) ∧
  (-1 = 1 + 2 * t₂ ∧ 5 = 2 - 3 * t₂) ↔
  (t₁ = 0 ∧ t₂ = -1) :=
by
  sorry

end parametric_equations_solution_l257_25721


namespace geometric_series_sum_l257_25725

theorem geometric_series_sum :
  let a := 3
  let r := 2
  let n := 8
  let S := (a * (1 - r^n)) / (1 - r)
  (3 + 6 + 12 + 24 + 48 + 96 + 192 + 384 = S) → S = 765 :=
by
  -- conditions
  let a := 3
  let r := 2
  let n := 8
  let S := (a * (1 - r^n)) / (1 - r)
  have h : 3 * (1 - 2^n) / (1 - 2) = 765 := sorry
  sorry

end geometric_series_sum_l257_25725


namespace zeros_of_quadratic_l257_25790

theorem zeros_of_quadratic (a b : ℝ) (h : a + b = 0) : 
  ∀ x, (b * x^2 - a * x = 0) ↔ (x = 0 ∨ x = -1) :=
by
  intro x
  sorry

end zeros_of_quadratic_l257_25790


namespace binomial_identity_l257_25713

theorem binomial_identity (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  k * (n.choose k) = n * ((n - 1).choose (k - 1)) :=
sorry

end binomial_identity_l257_25713


namespace find_a1_in_arithmetic_sequence_l257_25770

theorem find_a1_in_arithmetic_sequence (d n a_n : ℤ) (h_d : d = 2) (h_n : n = 15) (h_a_n : a_n = -10) :
  ∃ a1 : ℤ, a1 = -38 :=
by
  sorry

end find_a1_in_arithmetic_sequence_l257_25770


namespace pirate_ship_minimum_speed_l257_25772

noncomputable def minimum_speed (initial_distance : ℝ) (caravel_speed : ℝ) (caravel_direction : ℝ) : ℝ :=
  let caravel_velocity_x := -caravel_speed * Real.cos caravel_direction
  let caravel_velocity_y := -caravel_speed * Real.sin caravel_direction
  let t := initial_distance / (caravel_speed * (1 + Real.sqrt 3))
  let v_p := Real.sqrt ((initial_distance / t - caravel_velocity_x)^2 + (caravel_velocity_y)^2)
  v_p

theorem pirate_ship_minimum_speed : 
  minimum_speed 10 12 (Real.pi / 3) = 6 * Real.sqrt 6 :=
by
  sorry

end pirate_ship_minimum_speed_l257_25772


namespace find_number_l257_25767

-- Statement of the problem in Lean 4
theorem find_number (n : ℝ) (h : n / 3000 = 0.008416666666666666) : n = 25.25 :=
sorry

end find_number_l257_25767


namespace length_of_diagonal_EG_l257_25722

theorem length_of_diagonal_EG (EF FG GH HE : ℕ) (hEF : EF = 7) (hFG : FG = 15) 
  (hGH : GH = 7) (hHE : HE = 7) (primeEG : Prime EG) : EG = 11 ∨ EG = 13 :=
by
  -- Apply conditions and proof steps here
  sorry

end length_of_diagonal_EG_l257_25722


namespace row_time_to_100_yards_l257_25727

theorem row_time_to_100_yards :
  let init_width_yd := 50
  let final_width_yd := 100
  let increase_width_yd_per_10m := 2
  let rowing_speed_mps := 5
  let current_speed_mps := 1
  let yard_to_meter := 0.9144
  let init_width_m := init_width_yd * yard_to_meter
  let final_width_m := final_width_yd * yard_to_meter
  let width_increase_m_per_10m := increase_width_yd_per_10m * yard_to_meter
  let total_width_increase := (final_width_m - init_width_m)
  let num_segments := total_width_increase / width_increase_m_per_10m
  let total_distance := num_segments * 10
  let effective_speed := rowing_speed_mps + current_speed_mps
  let time := total_distance / effective_speed
  time = 41.67 := by
  sorry

end row_time_to_100_yards_l257_25727


namespace find_cos_E_floor_l257_25773

theorem find_cos_E_floor (EF GH EH FG : ℝ) (E G : ℝ) 
  (h1 : EF = 200) 
  (h2 : GH = 200) 
  (h3 : EH ≠ FG) 
  (h4 : EF + GH + EH + FG = 800) 
  (h5 : E = G) : 
  (⌊1000 * Real.cos E⌋ = 1000) := 
by 
  sorry

end find_cos_E_floor_l257_25773


namespace cube_of_99999_is_correct_l257_25748

theorem cube_of_99999_is_correct : (99999 : ℕ)^3 = 999970000299999 :=
by
  sorry

end cube_of_99999_is_correct_l257_25748


namespace patty_heavier_before_losing_weight_l257_25735

theorem patty_heavier_before_losing_weight {w_R w_P w_P' x : ℝ}
  (h1 : w_R = 100)
  (h2 : w_P = 100 * x)
  (h3 : w_P' = w_P - 235)
  (h4 : w_P' = w_R + 115) :
  x = 4.5 :=
by
  sorry

end patty_heavier_before_losing_weight_l257_25735


namespace difference_of_M_and_m_l257_25733

-- Define the variables and conditions
def total_students : ℕ := 2500
def min_G : ℕ := 1750
def max_G : ℕ := 1875
def min_R : ℕ := 1000
def max_R : ℕ := 1125

-- The statement to prove
theorem difference_of_M_and_m : 
  ∃ G R m M, 
  (G = total_students - R + m) ∧ 
  (min_G ≤ G ∧ G ≤ max_G) ∧
  (min_R ≤ R ∧ R ≤ max_R) ∧
  (m = min_G + min_R - total_students) ∧
  (M = max_G + max_R - total_students) ∧
  (M - m = 250) :=
sorry

end difference_of_M_and_m_l257_25733


namespace distance_to_bus_stand_l257_25764

theorem distance_to_bus_stand :
  ∀ D : ℝ, (D / 5 - 0.2 = D / 6 + 0.25) → D = 13.5 :=
by
  intros D h
  sorry

end distance_to_bus_stand_l257_25764


namespace daniel_total_worth_l257_25760

theorem daniel_total_worth
    (sales_tax_paid : ℝ)
    (sales_tax_rate : ℝ)
    (cost_tax_free_items : ℝ)
    (tax_rate_pos : 0 < sales_tax_rate) :
    sales_tax_paid = 0.30 →
    sales_tax_rate = 0.05 →
    cost_tax_free_items = 18.7 →
    ∃ (x : ℝ), 0.05 * x = 0.30 ∧ (x + cost_tax_free_items = 24.7) := by
    sorry

end daniel_total_worth_l257_25760


namespace banana_cost_l257_25701

/-- If 4 bananas cost $20, then the cost of one banana is $5. -/
theorem banana_cost (total_cost num_bananas : ℕ) (cost_per_banana : ℕ) 
  (h : total_cost = 20 ∧ num_bananas = 4) : cost_per_banana = 5 := by
  sorry

end banana_cost_l257_25701


namespace clock_hands_form_right_angle_at_180_over_11_l257_25723

-- Define the angular speeds as constants
def ω_hour : ℝ := 0.5  -- Degrees per minute
def ω_minute : ℝ := 6  -- Degrees per minute

-- Function to calculate the angle of the hour hand after t minutes
def angle_hour (t : ℝ) : ℝ := ω_hour * t

-- Function to calculate the angle of the minute hand after t minutes
def angle_minute (t : ℝ) : ℝ := ω_minute * t

-- Theorem: Prove the two hands form a right angle at the given time
theorem clock_hands_form_right_angle_at_180_over_11 : 
  ∃ t : ℝ, (6 * t - 0.5 * t = 90) ∧ t = 180 / 11 :=
by 
  -- This is where the proof would go, but we skip it with sorry
  sorry

end clock_hands_form_right_angle_at_180_over_11_l257_25723


namespace distance_between_vertices_hyperbola_l257_25762

theorem distance_between_vertices_hyperbola : 
  ∀ {x y : ℝ}, (x^2 / 121 - y^2 / 49 = 1) → (11 * 2 = 22) :=
by
  sorry

end distance_between_vertices_hyperbola_l257_25762


namespace john_investment_in_bank_a_l257_25700

theorem john_investment_in_bank_a :
  ∃ x : ℝ, 
    0 ≤ x ∧ x ≤ 1500 ∧
    x * (1 + 0.04)^3 + (1500 - x) * (1 + 0.06)^3 = 1740.54 ∧
    x = 695 := sorry

end john_investment_in_bank_a_l257_25700


namespace angle_triple_complement_l257_25774

-- Let x be the angle in degrees.
-- The angle is triple its complement.
-- We need to prove that x = 67.5.

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := 
by
  sorry

end angle_triple_complement_l257_25774


namespace zero_of_f_l257_25780

noncomputable def f (x : ℝ) : ℝ := (|Real.log x - Real.log 2|) - (1 / 3) ^ x

theorem zero_of_f :
  ∃ x1 x2 : ℝ, x1 < x2 ∧ (f x1 = 0) ∧ (f x2 = 0) ∧
  (1 < x1 ∧ x1 < 2) ∧ (2 < x2) := 
sorry

end zero_of_f_l257_25780


namespace bottom_right_corner_value_l257_25737

variable (a b c x : ℕ)

/--
Conditions:
- The sums of the numbers in each of the four 2x2 grids forming part of the 3x3 grid are equal.
- Known values for corners: a, b, and c.
Conclusion:
- The bottom right corner value x must be 0.
-/

theorem bottom_right_corner_value (S: ℕ) (A B C D E: ℕ) :
  S = a + A + B + C →
  S = A + b + C + D →
  S = B + C + c + E →
  S = C + D + E + x →
  x = 0 :=
by
  sorry

end bottom_right_corner_value_l257_25737


namespace leonard_younger_than_nina_by_4_l257_25785

variable (L N J : ℕ)

-- Conditions based on conditions from the problem
axiom h1 : L = 6
axiom h2 : N = 1 / 2 * J
axiom h3 : L + N + J = 36

-- Statement to prove
theorem leonard_younger_than_nina_by_4 : N - L = 4 :=
by 
  sorry

end leonard_younger_than_nina_by_4_l257_25785


namespace problem_f_2011_2012_l257_25775

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_2011_2012 :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f (1-x) = f (1+x)) →
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x = 2^x - 1) →
  f 2011 + f 2012 = -1 :=
by
  intros h1 h2 h3
  sorry

end problem_f_2011_2012_l257_25775


namespace num_terms_arith_seq_l257_25703

theorem num_terms_arith_seq {a d t : ℕ} (h_a : a = 5) (h_d : d = 3) (h_t : t = 140) :
  ∃ n : ℕ, t = a + (n-1) * d ∧ n = 46 :=
by
  sorry

end num_terms_arith_seq_l257_25703


namespace mens_wages_l257_25745

variable (M : ℕ) (wages_of_men : ℕ)

-- Conditions based on the problem
axiom eq1 : 15 * M = 90
axiom def_wages_of_men : wages_of_men = 5 * M

-- Prove that the total wages of the men are Rs. 30
theorem mens_wages : wages_of_men = 30 :=
by
  -- The proof would go here
  sorry

end mens_wages_l257_25745


namespace part1_part2_l257_25779

open Real

theorem part1 (m : ℝ) (h : ∀ x : ℝ, abs (x - 2) + abs (x - 3) ≥ m) : m ≤ 1 := 
sorry

theorem part2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1 = 1 / a + 1 / (2 * b) + 1 / (3 * c)) : a + 2 * b + 3 * c ≥ 9 := 
sorry

end part1_part2_l257_25779


namespace bad_carrots_count_l257_25753

def total_carrots (vanessa_carrots : ℕ) (mother_carrots : ℕ) : ℕ := 
vanessa_carrots + mother_carrots

def bad_carrots (total_carrots : ℕ) (good_carrots : ℕ) : ℕ := 
total_carrots - good_carrots

theorem bad_carrots_count : 
  ∀ (vanessa_carrots mother_carrots good_carrots : ℕ), 
  vanessa_carrots = 17 → 
  mother_carrots = 14 → 
  good_carrots = 24 → 
  bad_carrots (total_carrots vanessa_carrots mother_carrots) good_carrots = 7 := 
by 
  intros; 
  sorry

end bad_carrots_count_l257_25753


namespace x_squared_plus_y_squared_l257_25734

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : x^2 + y^2 = 25 := by
  sorry

end x_squared_plus_y_squared_l257_25734


namespace correct_graph_for_race_l257_25746

-- Define the conditions for the race.
def tortoise_constant_speed (d t : ℝ) := 
  ∃ k : ℝ, k > 0 ∧ d = k * t

def hare_behavior (d t t_nap t_end d_nap : ℝ) :=
  ∃ k1 k2 : ℝ, k1 > 0 ∧ k2 > 0 ∧ t_nap > 0 ∧ t_end > t_nap ∧
  (d = k1 * t ∨ (t_nap < t ∧ t < t_end ∧ d = d_nap) ∨ (t_end ≥ t ∧ d = d_nap + k2 * (t - t_end)))

-- Define the competition outcome.
def tortoise_wins (d_tortoise d_hare : ℝ) :=
  d_tortoise > d_hare

-- Proof that the graph which describes the race is Option (B).
theorem correct_graph_for_race :
  ∃ d_t d_h t t_nap t_end d_nap, 
    tortoise_constant_speed d_t t ∧ hare_behavior d_h t t_nap t_end d_nap ∧ tortoise_wins d_t d_h → "Option B" = "correct" :=
sorry -- Proof omitted.

end correct_graph_for_race_l257_25746


namespace solve_fractional_equation_l257_25794

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x + 1 ≠ 0) :
  (1 / x = 2 / (x + 1)) → x = 1 := 
by
  sorry

end solve_fractional_equation_l257_25794


namespace function_does_not_have_property_P_l257_25729

-- Definition of property P
def hasPropertyP (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ≠ x2 → f ((x1 + x2) / 2) = (f x1 + f x2) / 2

-- Function in question
def f (x : ℝ) : ℝ :=
  x^2

-- Statement that function f does not have property P
theorem function_does_not_have_property_P : ¬hasPropertyP f :=
  sorry

end function_does_not_have_property_P_l257_25729


namespace one_sixth_of_x_l257_25747

theorem one_sixth_of_x (x : ℝ) (h : x / 3 = 4) : x / 6 = 2 :=
sorry

end one_sixth_of_x_l257_25747


namespace inequality_solution_l257_25783

-- Define the problem statement formally
theorem inequality_solution (x : ℝ)
  (h1 : 2 * x > x + 1)
  (h2 : 4 * x - 1 > 7) :
  x > 2 :=
sorry

end inequality_solution_l257_25783


namespace lines_condition_l257_25756

-- Assume x and y are real numbers representing coordinates on the lines l1 and l2
variables (x y : ℝ)

-- Points on the lines l1 and l2 satisfy the condition |x| - |y| = 0.
theorem lines_condition (x y : ℝ) (h : abs x = abs y) : abs x - abs y = 0 :=
by
  sorry

end lines_condition_l257_25756


namespace inequality_correctness_l257_25787

variable (a b : ℝ)
variable (h1 : a < b) (h2 : b < 0)

theorem inequality_correctness : a^2 > ab ∧ ab > b^2 := by
  sorry

end inequality_correctness_l257_25787


namespace updated_mean_l257_25750

-- Definitions
def initial_mean := 200
def number_of_observations := 50
def decrement_per_observation := 9

-- Theorem stating the updated mean after decrementing each observation
theorem updated_mean : 
  (initial_mean * number_of_observations - decrement_per_observation * number_of_observations) / number_of_observations = 191 :=
by
  -- Placeholder for the proof
  sorry

end updated_mean_l257_25750


namespace perpendicular_lines_m_value_l257_25708

-- Define the first line
def line1 (x y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the second line
def line2 (x y : ℝ) (m : ℝ) : Prop := 6 * x - m * y - 3 = 0

-- Define the perpendicular condition for slopes of two lines
def perpendicular_slopes (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Prove the value of m for perpendicular lines
theorem perpendicular_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, line1 x y → ∃ y', line2 x y' m) →
  (∀ x y : ℝ, ∃ x', line1 x y ∧ line2 x' y m) →
  perpendicular_slopes 3 (6 / m) →
  m = -18 :=
by
  sorry

end perpendicular_lines_m_value_l257_25708


namespace Zoe_siblings_l257_25743

structure Child where
  eyeColor : String
  hairColor : String
  height : String

def Emma : Child := { eyeColor := "Green", hairColor := "Red", height := "Tall" }
def Zoe : Child := { eyeColor := "Gray", hairColor := "Brown", height := "Short" }
def Liam : Child := { eyeColor := "Green", hairColor := "Brown", height := "Short" }
def Noah : Child := { eyeColor := "Gray", hairColor := "Red", height := "Tall" }
def Mia : Child := { eyeColor := "Green", hairColor := "Red", height := "Short" }
def Lucas : Child := { eyeColor := "Gray", hairColor := "Brown", height := "Tall" }

def sibling (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor ∨ c1.height = c2.height

theorem Zoe_siblings : sibling Zoe Noah ∧ sibling Zoe Lucas ∧ ∃ x, sibling Noah x ∧ sibling Lucas x :=
by
  sorry

end Zoe_siblings_l257_25743


namespace value_of_a_l257_25720

theorem value_of_a (a : ℝ) (x : ℝ) (h : (a - 1) * x^2 + x + a^2 - 1 = 0) : a = -1 :=
sorry

end value_of_a_l257_25720


namespace third_wins_against_seventh_l257_25761

-- Define the participants and their distinct points 
variables (p : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → p i ≠ p j)
-- descending order condition
variables (h_order : ∀ i j, i < j → p i > p j)
-- second place points equals sum of last four places
variables (h_second : p 2 = p 5 + p 6 + p 7 + p 8)

-- Theorem stating the third place player won against the seventh place player
theorem third_wins_against_seventh :
  p 3 > p 7 :=
sorry

end third_wins_against_seventh_l257_25761


namespace sum_of_numbers_l257_25765

theorem sum_of_numbers (x y : ℝ) (h1 : y = 4 * x) (h2 : x + y = 45) : x + y = 45 := 
by
  sorry

end sum_of_numbers_l257_25765


namespace largest_b_value_l257_25799

open Real

structure Triangle :=
(side_a side_b side_c : ℝ)
(a_pos : 0 < side_a)
(b_pos : 0 < side_b)
(c_pos : 0 < side_c)
(tri_ineq_a : side_a + side_b > side_c)
(tri_ineq_b : side_b + side_c > side_a)
(tri_ineq_c : side_c + side_a > side_b)

noncomputable def inradius (T : Triangle) : ℝ :=
  let s := (T.side_a + T.side_b + T.side_c) / 2
  let A := sqrt (s * (s - T.side_a) * (s - T.side_b) * (s - T.side_c))
  A / s

noncomputable def circumradius (T : Triangle) : ℝ :=
  let A := sqrt (((T.side_a + T.side_b + T.side_c) / 2) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_a) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_b) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_c))
  (T.side_a * T.side_b * T.side_c) / (4 * A)

noncomputable def condition_met (T1 T2 : Triangle) : Prop :=
  (inradius T1 / circumradius T1) = (inradius T2 / circumradius T2)

theorem largest_b_value :
  let T1 := Triangle.mk 8 11 11 (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  ∃ b > 0, ∃ T2 : Triangle, T2.side_a = b ∧ T2.side_b = 1 ∧ T2.side_c = 1 ∧ b = 14 / 11 ∧ condition_met T1 T2 :=
  sorry

end largest_b_value_l257_25799


namespace unique_handshakes_count_l257_25776

-- Definitions from the conditions
def teams : Nat := 4
def players_per_team : Nat := 2
def total_players : Nat := teams * players_per_team

def handshakes_per_player : Nat := total_players - players_per_team

-- The Lean statement to prove the total number of unique handshakes
theorem unique_handshakes_count : (total_players * handshakes_per_player) / 2 = 24 := 
by
  -- Proof steps would go here
  sorry

end unique_handshakes_count_l257_25776


namespace contractor_fine_per_absent_day_l257_25718

theorem contractor_fine_per_absent_day :
  ∀ (total_days absent_days wage_per_day total_receipt fine_per_absent_day : ℝ),
    total_days = 30 →
    wage_per_day = 25 →
    absent_days = 4 →
    total_receipt = 620 →
    (total_days - absent_days) * wage_per_day - absent_days * fine_per_absent_day = total_receipt →
    fine_per_absent_day = 7.50 :=
by
  intros total_days absent_days wage_per_day total_receipt fine_per_absent_day
  intro h1 h2 h3 h4 h5
  sorry

end contractor_fine_per_absent_day_l257_25718


namespace exists_three_digit_numbers_with_property_l257_25726

open Nat

def is_three_digit_number (n : ℕ) : Prop := (100 ≤ n ∧ n < 1000)

def distinct_digits (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def inserts_zeros_and_is_square (n : ℕ) (k : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  let transformed_number := a * 10^(2*k + 2) + b * 10^(k + 1) + c
  ∃ x : ℕ, transformed_number = x * x

theorem exists_three_digit_numbers_with_property:
  ∃ n1 n2 : ℕ, 
    is_three_digit_number n1 ∧ 
    is_three_digit_number n2 ∧ 
    distinct_digits n1 ∧ 
    distinct_digits n2 ∧ 
    ( ∀ k, inserts_zeros_and_is_square n1 k ) ∧ 
    ( ∀ k, inserts_zeros_and_is_square n2 k ) ∧ 
    n1 ≠ n2 := 
sorry

end exists_three_digit_numbers_with_property_l257_25726


namespace number_of_square_tiles_l257_25782

-- A box contains a mix of triangular and square tiles.
-- There are 30 tiles in total with 100 edges altogether.
variable (x y : ℕ) -- where x is the number of triangular tiles and y is the number of square tiles, both must be natural numbers
-- Each triangular tile has 3 edges, and each square tile has 4 edges.

-- Define the conditions
def tile_condition_1 : Prop := x + y = 30
def tile_condition_2 : Prop := 3 * x + 4 * y = 100

-- The goal is to prove the number of square tiles y is 10.
theorem number_of_square_tiles : tile_condition_1 x y → tile_condition_2 x y → y = 10 :=
  by
    intros h1 h2
    sorry

end number_of_square_tiles_l257_25782


namespace solve_equation_l257_25763

noncomputable def f (x : ℝ) : ℝ :=
  abs (abs (abs (abs (abs x - 8) - 4) - 2) - 1)

noncomputable def g (x : ℝ) : ℝ :=
  abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs x - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1)

theorem solve_equation : ∀ (x : ℝ), f x = g x :=
by
  sorry -- The proof will be inserted here

end solve_equation_l257_25763


namespace truck_capacity_rental_plan_l257_25741

-- Define the variables for the number of boxes each type of truck can carry
variables {x y : ℕ}

-- Define the conditions for the number of boxes carried by trucks
axiom cond1 : 15 * x + 25 * y = 750
axiom cond2 : 10 * x + 30 * y = 700

-- Problem 1: Prove x = 25 and y = 15
theorem truck_capacity : x = 25 ∧ y = 15 :=
by
  sorry

-- Define the variables for the number of each type of truck
variables {m : ℕ}

-- Define the conditions for the total number of trucks and boxes to be carried
axiom cond3 : 25 * m + 15 * (70 - m) ≤ 1245
axiom cond4 : 70 - m ≤ 3 * m

-- Problem 2: Prove there is one valid rental plan with m = 18 and 70-m = 52
theorem rental_plan : 17 ≤ m ∧ m ≤ 19 ∧ 70 - m ≤ 3 * m ∧ (70-m = 52 → m = 18) :=
by
  sorry

end truck_capacity_rental_plan_l257_25741


namespace fruit_costs_l257_25759

theorem fruit_costs (
    A O B : ℝ
) (h1 : O = A + 0.28)
  (h2 : B = A - 0.15)
  (h3 : 3 * A + 7 * O + 5 * B = 7.84) :
  A = 0.442 ∧ O = 0.722 ∧ B = 0.292 :=
by
  -- The proof is omitted here; replacing with sorry for now
  sorry

end fruit_costs_l257_25759


namespace equation_of_line_l257_25712

theorem equation_of_line (x y : ℝ) :
  (∃ (x1 y1 : ℝ), (x1 = 0) ∧ (y1= 2) ∧ (y - y1 = 2 * (x - x1))) → (y = 2 * x + 2) :=
by
  sorry

end equation_of_line_l257_25712


namespace James_average_speed_l257_25714

theorem James_average_speed (TotalDistance : ℝ) (BreakTime : ℝ) (TotalTripTime : ℝ) (h1 : TotalDistance = 42) (h2 : BreakTime = 1) (h3 : TotalTripTime = 9) :
  (TotalDistance / (TotalTripTime - BreakTime)) = 5.25 :=
by
  sorry

end James_average_speed_l257_25714


namespace probability_neither_prime_nor_composite_lemma_l257_25740

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

def neither_prime_nor_composite (n : ℕ) : Prop :=
  ¬ is_prime n ∧ ¬ is_composite n

def probability_of_neither_prime_nor_composite (n : ℕ) : ℚ :=
  if 1 ≤ n ∧ n ≤ 97 then 1 / 97 else 0

theorem probability_neither_prime_nor_composite_lemma :
  probability_of_neither_prime_nor_composite 1 = 1 / 97 := by
  sorry

end probability_neither_prime_nor_composite_lemma_l257_25740


namespace four_digit_numbers_with_one_digit_as_average_l257_25797

noncomputable def count_valid_four_digit_numbers : Nat := 80

theorem four_digit_numbers_with_one_digit_as_average :
  ∃ n : Nat, n = count_valid_four_digit_numbers ∧ n = 80 := by
  use count_valid_four_digit_numbers
  constructor
  · rfl
  · rfl

end four_digit_numbers_with_one_digit_as_average_l257_25797


namespace A_subset_B_l257_25707

def inA (n : ℕ) : Prop := ∃ x y : ℕ, n = x^2 + 2 * y^2 ∧ x > y
def inB (n : ℕ) : Prop := ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ n = (a^3 + b^3 + c^3) / (a + b + c)

theorem A_subset_B : ∀ (n : ℕ), inA n → inB n := 
sorry

end A_subset_B_l257_25707


namespace f_2009_is_one_l257_25736

   -- Define the properties of the function f
   variables (f : ℤ → ℤ)
   variable (h_even : ∀ x : ℤ, f x = f (-x))
   variable (h1 : f 1 = 1)
   variable (h2008 : f 2008 ≠ 1)
   variable (h_max : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b))

   -- Prove that f(2009) = 1
   theorem f_2009_is_one : f 2009 = 1 :=
   sorry
   
end f_2009_is_one_l257_25736


namespace cost_of_tax_free_items_l257_25728

theorem cost_of_tax_free_items (total_cost : ℝ) (tax_40_percent : ℝ) 
  (tax_30_percent : ℝ) (discount : ℝ) : 
  (total_cost = 120) →
  (tax_40_percent = 0.4 * total_cost) →
  (tax_30_percent = 0.3 * total_cost) →
  (discount = 0.05 * tax_30_percent) →
  (tax-free_items = total_cost - (tax_40_percent + (tax_30_percent - discount))) → 
  tax_free_items = 36 :=
by sorry

end cost_of_tax_free_items_l257_25728


namespace cricket_player_average_l257_25755

theorem cricket_player_average
  (A : ℕ)
  (h1 : 8 * A + 96 = 9 * (A + 8)) :
  A = 24 :=
by
  sorry

end cricket_player_average_l257_25755


namespace eggs_left_in_jar_l257_25786

def eggs_after_removal (original removed : Nat) : Nat :=
  original - removed

theorem eggs_left_in_jar : eggs_after_removal 27 7 = 20 :=
by
  sorry

end eggs_left_in_jar_l257_25786


namespace solve_for_x_l257_25739

theorem solve_for_x (x : ℝ) (h : (x / 3) / 3 = 9 / (x / 3)) : x = 3 ^ (5 / 2) ∨ x = -3 ^ (5 / 2) :=
by
  sorry

end solve_for_x_l257_25739


namespace dividend_calculation_l257_25788

theorem dividend_calculation :
  let divisor := 17
  let quotient := 9
  let remainder := 6
  let dividend := 159
  (divisor * quotient) + remainder = dividend :=
by
  sorry

end dividend_calculation_l257_25788


namespace quadratic_inequality_range_l257_25768

theorem quadratic_inequality_range (a x : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) ↔ (-1/2 < a ∧ a < 3/2) :=
by
  sorry

end quadratic_inequality_range_l257_25768


namespace count_triangles_in_figure_l257_25730

/-- 
The figure is a rectangle divided into 8 columns and 2 rows with additional diagonal and vertical lines.
We need to prove that there are 76 triangles in total in the figure.
-/
theorem count_triangles_in_figure : 
  let columns := 8 
  let rows := 2 
  let num_triangles := 76 
  ∃ total_triangles, total_triangles = num_triangles :=
by
  sorry

end count_triangles_in_figure_l257_25730


namespace no_fraternity_member_is_club_member_thm_l257_25709

-- Definitions from the conditions
variable (Person : Type)
variable (Club : Person → Prop)
variable (Honest : Person → Prop)
variable (Student : Person → Prop)
variable (Fraternity : Person → Prop)

-- Hypotheses from the problem statements
axiom all_club_members_honest (p : Person) : Club p → Honest p
axiom some_students_not_honest : ∃ p : Person, Student p ∧ ¬ Honest p
axiom no_fraternity_member_is_club_member (p : Person) : Fraternity p → ¬ Club p

-- The theorem to be proven
theorem no_fraternity_member_is_club_member_thm : 
  ∀ p : Person, Fraternity p → ¬ Club p := 
by 
  sorry

end no_fraternity_member_is_club_member_thm_l257_25709


namespace next_special_year_after_2009_l257_25758

def is_special_year (n : ℕ) : Prop :=
  ∃ d1 d2 d3 d4 : ℕ,
    (2000 ≤ n) ∧ (n < 10000) ∧
    (d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n) ∧
    (d1 ≠ 0) ∧
    ∀ (p q r s : ℕ),
    (p * 1000 + q * 100 + r * 10 + s < n) →
    (p ≠ d1 ∨ q ≠ d2 ∨ r ≠ d3 ∨ s ≠ d4)

theorem next_special_year_after_2009 : ∃ y : ℕ, is_special_year y ∧ y > 2009 ∧ y = 2022 :=
  sorry

end next_special_year_after_2009_l257_25758


namespace evaluate_expression_correct_l257_25766

noncomputable def evaluate_expression :=
  abs (-1) - ((-3.14 + Real.pi) ^ 0) + (2 ^ (-1 : ℤ)) + (Real.cos (Real.pi / 6)) ^ 2

theorem evaluate_expression_correct : evaluate_expression = 5 / 4 := by sorry

end evaluate_expression_correct_l257_25766
