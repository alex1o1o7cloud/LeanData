import Mathlib

namespace rowing_time_l861_86191

theorem rowing_time (rowing_speed : ℕ) (current_speed : ℕ) (distance : ℕ) 
  (h_rowing_speed : rowing_speed = 10)
  (h_current_speed : current_speed = 2)
  (h_distance : distance = 24) : 
  2 * distance / (rowing_speed + current_speed) + 2 * distance / (rowing_speed - current_speed) = 5 :=
by
  rw [h_rowing_speed, h_current_speed, h_distance]
  norm_num
  sorry

end rowing_time_l861_86191


namespace maria_nickels_l861_86155

theorem maria_nickels (dimes quarters_initial quarters_additional : ℕ) (total_amount : ℚ) 
  (Hd : dimes = 4) (Hqi : quarters_initial = 4) (Hqa : quarters_additional = 5) (Htotal : total_amount = 3) : 
  (dimes * 0.10 + quarters_initial * 0.25 + quarters_additional * 0.25 + n/20) = total_amount → n = 7 :=
  sorry

end maria_nickels_l861_86155


namespace romanov_family_savings_l861_86120

theorem romanov_family_savings :
  let cost_multi_tariff_meter := 3500
  let installation_cost := 1100
  let monthly_consumption := 300
  let night_consumption := 230
  let day_consumption := monthly_consumption - night_consumption
  let night_rate := 3.4
  let day_rate := 5.2
  let standard_rate := 4.6
  let yearly_cost_multi_tariff :=
    (night_consumption * night_rate * 12) +
    (day_consumption * day_rate * 12)
  let total_cost_multi_tariff :=
    cost_multi_tariff_meter + installation_cost + (yearly_cost_multi_tariff * 3)
  let yearly_cost_standard :=
    monthly_consumption * standard_rate * 12
  let total_cost_standard :=
    yearly_cost_standard * 3
  total_cost_standard - total_cost_multi_tariff = 3824 := 
by {
  sorry -- Proof goes here
}

end romanov_family_savings_l861_86120


namespace gcd_example_l861_86133

-- Define the two numbers
def a : ℕ := 102
def b : ℕ := 238

-- Define the GCD of a and b
def gcd_ab : ℕ :=
  Nat.gcd a b

-- The expected result of the GCD
def expected_gcd : ℕ := 34

-- Prove that the GCD of a and b is equal to the expected GCD
theorem gcd_example : gcd_ab = expected_gcd := by
  sorry

end gcd_example_l861_86133


namespace max_value_of_y_over_x_l861_86175

theorem max_value_of_y_over_x {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 2 * y = 3) :
  y / x ≤ 9 / 8 :=
sorry

end max_value_of_y_over_x_l861_86175


namespace maximum_candy_leftover_l861_86164

theorem maximum_candy_leftover (x : ℕ) 
  (h1 : ∀ (bags : ℕ), bags = 12 → x ≥ bags * 10)
  (h2 : ∃ (leftover : ℕ), leftover < 12 ∧ leftover = (x - 120) % 12) : 
  ∃ (leftover : ℕ), leftover = 11 :=
by
  sorry

end maximum_candy_leftover_l861_86164


namespace find_m_l861_86121

variable {α : Type*} [DecidableEq α]

-- Definitions and conditions
def A (m : ℤ) : Set ℤ := {-1, 3, m ^ 2}
def B : Set ℤ := {3, 4}

theorem find_m (m : ℤ) (h : B ⊆ A m) : m = 2 ∨ m = -2 := by
  sorry

end find_m_l861_86121


namespace ratio_sheep_to_horses_l861_86183

theorem ratio_sheep_to_horses 
  (horse_food_per_day : ℕ) 
  (total_horse_food : ℕ) 
  (num_sheep : ℕ) 
  (H1 : horse_food_per_day = 230) 
  (H2 : total_horse_food = 12880) 
  (H3 : num_sheep = 48) 
  : (num_sheep : ℚ) / (total_horse_food / horse_food_per_day : ℚ) = 6 / 7
  :=
by
  sorry

end ratio_sheep_to_horses_l861_86183


namespace arithmetic_seq_sum_2017_l861_86135

theorem arithmetic_seq_sum_2017 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (a1 : a 1 = -2017) 
  (h1 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1))
  (h2 : (S 2014) / 2014 - (S 2008) / 2008 = 6) : 
  S 2017 = -2017 :=
by
  sorry

end arithmetic_seq_sum_2017_l861_86135


namespace each_person_bid_count_l861_86134

-- Define the conditions and initial values
noncomputable def auctioneer_price_increase : ℕ := 5
noncomputable def initial_price : ℕ := 15
noncomputable def final_price : ℕ := 65
noncomputable def number_of_bidders : ℕ := 2

-- Define the proof statement
theorem each_person_bid_count : 
  (final_price - initial_price) / auctioneer_price_increase / number_of_bidders = 5 :=
by sorry

end each_person_bid_count_l861_86134


namespace percent_research_and_development_is_9_l861_86156

-- Define given percentages
def percent_transportation := 20
def percent_utilities := 5
def percent_equipment := 4
def percent_supplies := 2

-- Define degree representation and calculate percent for salaries
def degrees_in_circle := 360
def degrees_salaries := 216
def percent_salaries := (degrees_salaries * 100) / degrees_in_circle

-- Define the total percentage representation
def total_percent := 100
def known_percent := percent_transportation + percent_utilities + percent_equipment + percent_supplies + percent_salaries

-- Calculate the percent for research and development
def percent_research_and_development := total_percent - known_percent

-- Theorem statement
theorem percent_research_and_development_is_9 : percent_research_and_development = 9 :=
by 
  -- Placeholder for actual proof
  sorry

end percent_research_and_development_is_9_l861_86156


namespace polynomial_divisible_exists_l861_86182

theorem polynomial_divisible_exists (p : Polynomial ℤ) (a : ℕ → ℤ) (k : ℕ) 
  (h_inc : ∀ i j, i < j → a i < a j) (h_nonzero : ∀ i, i < k → p.eval (a i) ≠ 0) :
  ∃ a_0 : ℤ, ∀ i, i < k → p.eval (a i) ∣ p.eval a_0 := 
by
  sorry

end polynomial_divisible_exists_l861_86182


namespace point_coordinates_l861_86115

-- We assume that the point P has coordinates (2, 4) and prove that the coordinates with respect to the origin in Cartesian system are indeed (2, 4).
theorem point_coordinates (x y : ℝ) (h : x = 2 ∧ y = 4) : (x, y) = (2, 4) :=
by
  sorry

end point_coordinates_l861_86115


namespace ellipse_equation_midpoint_coordinates_l861_86145

noncomputable def ellipse_c := {x : ℝ × ℝ | (x.1^2 / 25) + (x.2^2 / 16) = 1}

theorem ellipse_equation (a b : ℝ) (h1 : a = 5) (h2 : b = 4) :
    ∀ x y : ℝ, x = 0 → y = 4 → (y^2 / b^2 = 1) ∧ (e = 3 / 5) → 
      (a > b ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) := 
sorry

theorem midpoint_coordinates (a b : ℝ) (h1 : a = 5) (h2 : b = 4) :
    ∀ x y x1 x2 y1 y2 : ℝ, 
    (y = 4 / 5 * (x - 3)) → 
    (y1 = 4 / 5 * (x1 - 3)) ∧ (y2 = 4 / 5 * (x2 - 3)) ∧ 
    (x1^2 / a^2) + ((y1 - 3)^2 / b^2) = 1 ∧ (x2^2 / a^2) + ((y2 - 3)^2 / b^2) = 1 ∧ 
    (x1 + x2 = 3) → 
    ((x1 + x2) / 2 = 3 / 2) ∧ ((y1 + y2) / 2 = -6 / 5) := 
sorry

end ellipse_equation_midpoint_coordinates_l861_86145


namespace initial_amount_invested_l861_86109

-- Conditions
def initial_investment : ℝ := 367.36
def annual_interest_rate : ℝ := 0.08
def accumulated_amount : ℝ := 500
def years : ℕ := 4

-- Required to prove that the initial investment satisfies the given equation
theorem initial_amount_invested :
  initial_investment * (1 + annual_interest_rate) ^ years = accumulated_amount :=
by
  sorry

end initial_amount_invested_l861_86109


namespace total_questions_needed_l861_86127

def m_total : ℕ := 35
def p_total : ℕ := 15
def t_total : ℕ := 20

def m_written : ℕ := (3 * m_total) / 7
def p_written : ℕ := p_total / 5
def t_written : ℕ := t_total / 4

def m_remaining : ℕ := m_total - m_written
def p_remaining : ℕ := p_total - p_written
def t_remaining : ℕ := t_total - t_written

def total_remaining : ℕ := m_remaining + p_remaining + t_remaining

theorem total_questions_needed : total_remaining = 47 := by
  sorry

end total_questions_needed_l861_86127


namespace least_subtract_for_divisibility_l861_86168

theorem least_subtract_for_divisibility (n : ℕ) (hn : n = 427398) : 
  (∃ m : ℕ, n - m % 10 = 0 ∧ m = 2) :=
by
  sorry

end least_subtract_for_divisibility_l861_86168


namespace internal_angles_and_area_of_grey_triangle_l861_86188

/-- Given three identical grey triangles, 
    three identical squares, and an equilateral 
    center triangle with area 2 cm^2,
    the internal angles of the grey triangles 
    are 120 degrees and 30 degrees, and the 
    total grey area is 6 cm^2. -/
theorem internal_angles_and_area_of_grey_triangle 
  (triangle_area : ℝ)
  (α β : ℝ)
  (grey_area : ℝ) :
  triangle_area = 2 →  
  α = 120 ∧ β = 30 ∧ grey_area = 6 :=
by
  sorry

end internal_angles_and_area_of_grey_triangle_l861_86188


namespace initial_parking_hours_proof_l861_86172

noncomputable def initial_parking_hours (total_cost : ℝ) (excess_hourly_rate : ℝ) (average_cost : ℝ) (total_hours : ℕ) : ℝ :=
  let h := (total_hours * average_cost - total_cost) / excess_hourly_rate
  h

theorem initial_parking_hours_proof : initial_parking_hours 21.25 1.75 2.361111111111111 9 = 2 :=
by
  sorry

end initial_parking_hours_proof_l861_86172


namespace exponentiation_properties_l861_86126

theorem exponentiation_properties:
  (10^6) * (10^2)^3 / 10^4 = 10^8 :=
by
  sorry

end exponentiation_properties_l861_86126


namespace erwin_chocolates_weeks_l861_86114

-- Define weekdays chocolates and weekends chocolates
def weekdays_chocolates := 2
def weekends_chocolates := 1

-- Define the total chocolates Erwin ate
def total_chocolates := 24

-- Define the number of weekdays and weekend days in a week
def weekdays := 5
def weekends := 2

-- Define the total chocolates Erwin eats in a week
def chocolates_per_week : Nat := (weekdays * weekdays_chocolates) + (weekends * weekends_chocolates)

-- Prove that Erwin finishes all chocolates in 2 weeks
theorem erwin_chocolates_weeks : (total_chocolates / chocolates_per_week) = 2 := by
  sorry

end erwin_chocolates_weeks_l861_86114


namespace distance_from_P_to_x_axis_l861_86180

-- Define the point P with coordinates (4, -3)
def P : ℝ × ℝ := (4, -3)

-- Define the distance from a point to the x-axis as the absolute value of the y-coordinate
def distance_to_x_axis (point : ℝ × ℝ) : ℝ :=
  abs point.snd

-- State the theorem to be proved
theorem distance_from_P_to_x_axis : distance_to_x_axis P = 3 :=
by
  -- The proof is not required; we can use sorry to skip it
  sorry

end distance_from_P_to_x_axis_l861_86180


namespace exists_nat_with_digit_sum_l861_86169

-- Definitions of the necessary functions
def digit_sum (n : ℕ) : ℕ := sorry -- Assume this is the sum of the digits of n

theorem exists_nat_with_digit_sum :
  ∃ n : ℕ, digit_sum n = 1000 ∧ digit_sum (n^2) = 1000000 :=
by
  sorry

end exists_nat_with_digit_sum_l861_86169


namespace no_integer_solution_l861_86177

theorem no_integer_solution (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
by
  -- Proof omitted
  sorry

end no_integer_solution_l861_86177


namespace problem_1_problem_2_l861_86149

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
axiom h1 : ∀ n : ℕ, 2 * S n = a (n + 1) - 2^(n + 1) + 1
axiom h2 : a 2 + 5 = a 1 + (a 3 - a 2)

-- Problem 1: Prove the value of a₁
theorem problem_1 : a 1 = 1 := sorry

-- Problem 2: Find the general term formula for the sequence {aₙ}
theorem problem_2 : ∀ n : ℕ, a n = 3^n - 2^n := sorry

end problem_1_problem_2_l861_86149


namespace intersect_xz_plane_at_point_l861_86150

-- Define points and vectors in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the points A and B
def A : Point3D := ⟨2, -1, 3⟩
def B : Point3D := ⟨6, 7, -2⟩

-- Define the direction vector as the difference between points A and B
def direction_vector (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

-- Function to parameterize the line given a point and direction vector
def parametric_line (P : Point3D) (v : Point3D) (t : ℝ) : Point3D :=
  ⟨P.x + t * v.x, P.y + t * v.y, P.z + t * v.z⟩

-- Define the xz-plane intersection condition (y coordinate should be 0)
def intersects_xz_plane (P : Point3D) (v : Point3D) (t : ℝ) : Prop :=
  (parametric_line P v t).y = 0

-- Define the intersection point as a Point3D
def intersection_point : Point3D := ⟨2.5, 0, 2.375⟩

-- Statement to prove the intersection
theorem intersect_xz_plane_at_point : 
  ∃ t : ℝ, intersects_xz_plane A (direction_vector A B) t ∧ parametric_line A (direction_vector A B) t = intersection_point :=
by
  sorry

end intersect_xz_plane_at_point_l861_86150


namespace log_product_l861_86184

open Real

theorem log_product : log 9 / log 2 * (log 5 / log 3) * (log 8 / log (sqrt 5)) = 12 :=
by
  sorry

end log_product_l861_86184


namespace pushing_car_effort_l861_86162

theorem pushing_car_effort (effort constant : ℕ) (people1 people2 : ℕ) 
  (h1 : constant = people1 * effort)
  (h2 : people1 = 4)
  (h3 : effort = 120)
  (h4 : people2 = 6) :
  effort * people1 = constant → constant = people2 * 80 :=
by
  sorry

end pushing_car_effort_l861_86162


namespace complement_union_eq_l861_86159

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem complement_union_eq : (U \ (S ∪ T)) = {2, 4, 7, 8} :=
by {
  sorry
}

end complement_union_eq_l861_86159


namespace math_problem_l861_86166

theorem math_problem (p q : ℕ) (hp : p % 13 = 7) (hq : q % 13 = 7) (hp_lower : 1000 ≤ p) (hp_upper : p < 10000) (hq_lower : 10000 ≤ q) (min_p : ∀ n, n % 13 = 7 → 1000 ≤ n → n < 10000 → p ≤ n) (min_q : ∀ n, n % 13 = 7 → 10000 ≤ n → q ≤ n) : 
  q - p = 8996 := 
sorry

end math_problem_l861_86166


namespace combustion_moles_l861_86131

-- Chemical reaction definitions
def balanced_equation : Prop :=
  ∀ (CH4 Cl2 O2 CO2 HCl H2O : ℝ),
  1 * CH4 + 4 * Cl2 + 4 * O2 = 1 * CO2 + 4 * HCl + 2 * H2O

-- Moles of substances
def moles_CH4 := 24
def moles_Cl2 := 48
def moles_O2 := 96
def moles_CO2 := 24
def moles_HCl := 48
def moles_H2O := 48

-- Prove the conditions based on the balanced equation
theorem combustion_moles :
  balanced_equation →
  (moles_O2 = 4 * moles_CH4) ∧
  (moles_H2O = 2 * moles_CH4) :=
by {
  sorry
}

end combustion_moles_l861_86131


namespace number_of_leap_years_l861_86148

noncomputable def is_leap_year (year : ℕ) : Prop :=
  (year % 1300 = 300 ∨ year % 1300 = 700) ∧ 2000 ≤ year ∧ year ≤ 5000

noncomputable def leap_years : List ℕ :=
  [2900, 4200, 3300, 4600]

theorem number_of_leap_years : leap_years.length = 4 ∧ ∀ y ∈ leap_years, is_leap_year y := by
  sorry

end number_of_leap_years_l861_86148


namespace four_digit_palindrome_square_count_l861_86107

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l861_86107


namespace find_g_inverse_75_l861_86142

noncomputable def g (x : ℝ) : ℝ := 3 * x^3 - 6

theorem find_g_inverse_75 : g⁻¹ 75 = 3 := sorry

end find_g_inverse_75_l861_86142


namespace hexagon_sequences_l861_86130

theorem hexagon_sequences : ∃ n : ℕ, n = 7 ∧ 
  ∀ (x d : ℕ), 6 * x + 15 * d = 720 ∧ (2 * x + 5 * d = 240) ∧ 
  (x + 5 * d < 160) ∧ (0 < x) ∧ (0 < d) ∧ (d % 2 = 0) ↔ (∃ k < n, (∃ x, ∃ d, x = 85 - 2*k ∧ d = 2 + 2*k)) :=
by
  sorry

end hexagon_sequences_l861_86130


namespace solve_equation_1_solve_equation_2_solve_equation_3_l861_86189

theorem solve_equation_1 (x : ℝ) : (x^2 - 3 * x = 0) ↔ (x = 0 ∨ x = 3) := sorry

theorem solve_equation_2 (x : ℝ) : (4 * x^2 - x - 5 = 0) ↔ (x = 5/4 ∨ x = -1) := sorry

theorem solve_equation_3 (x : ℝ) : (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2/3) := sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l861_86189


namespace find_m_l861_86108

-- Define the function and conditions
def power_function (x : ℝ) (m : ℕ) : ℝ := x^(m - 2)

theorem find_m (m : ℕ) (x : ℝ) (h1 : 0 < m) (h2 : power_function 0 m = 0 → false) : m = 1 ∨ m = 2 :=
by
  sorry -- Skip the proof

end find_m_l861_86108


namespace number_of_ways_to_choose_bases_l861_86132

theorem number_of_ways_to_choose_bases : ∀ (students bases : ℕ), students = 4 → bases = 4 → (bases^students) = 256 :=
by
  intros students bases h_students h_bases
  rw [h_students, h_bases]
  exact pow_succ' 4 3

end number_of_ways_to_choose_bases_l861_86132


namespace two_leq_one_add_one_div_n_pow_n_lt_three_l861_86152

theorem two_leq_one_add_one_div_n_pow_n_lt_three :
  ∀ (n : ℕ), 2 ≤ (1 + (1 : ℝ) / n) ^ n ∧ (1 + (1 : ℝ) / n) ^ n < 3 := 
by 
  sorry

end two_leq_one_add_one_div_n_pow_n_lt_three_l861_86152


namespace set_intersection_complement_l861_86122

def setA : Set ℝ := {-2, -1, 0, 1, 2}
def setB : Set ℝ := { x : ℝ | x^2 + 2*x < 0 }
def complementB : Set ℝ := { x : ℝ | x ≥ 0 ∨ x ≤ -2 }

theorem set_intersection_complement :
  setA ∩ complementB = {-2, 0, 1, 2} :=
by
  sorry

end set_intersection_complement_l861_86122


namespace total_students_l861_86128

-- Lean statement: Prove the number of students given the conditions.
theorem total_students (num_classrooms : ℕ) (num_buses : ℕ) (seats_per_bus : ℕ) 
  (students : ℕ) (h1 : num_classrooms = 87) (h2 : num_buses = 29) 
  (h3 : seats_per_bus = 2) (h4 : students = num_classrooms * num_buses * seats_per_bus) :
  students = 5046 :=
by
  sorry

end total_students_l861_86128


namespace cosine_triangle_ABC_l861_86190

noncomputable def triangle_cosine_proof (a b : ℝ) (A : ℝ) (cosB : ℝ) : Prop :=
  let sinA := Real.sin A
  let sinB := b * sinA / a
  let cosB_expr := Real.sqrt (1 - sinB^2)
  cosB = cosB_expr

theorem cosine_triangle_ABC : triangle_cosine_proof (Real.sqrt 7) 2 (Real.pi / 4) (Real.sqrt 35 / 7) :=
by
  sorry

end cosine_triangle_ABC_l861_86190


namespace find_fx_plus_1_l861_86100

theorem find_fx_plus_1 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x - 1) = x^2 + 4 * x - 5) : 
  ∀ x : ℤ, f (x + 1) = x^2 + 8 * x + 7 :=
sorry

end find_fx_plus_1_l861_86100


namespace time_to_fill_pot_l861_86161

def pot_volume : ℕ := 3000  -- in ml
def rate_of_entry : ℕ := 60 -- in ml/minute

-- Statement: Prove that the time required for the pot to be full is 50 minutes.
theorem time_to_fill_pot : (pot_volume / rate_of_entry) = 50 := by
  sorry

end time_to_fill_pot_l861_86161


namespace installation_quantities_l861_86153

theorem installation_quantities :
  ∃ x1 x2 x3 : ℕ, x1 = 22 ∧ x2 = 88 ∧ x3 = 22 ∧
  (x1 + x2 + x3 ≥ 100) ∧
  (x2 = 4 * x1) ∧
  (∃ k : ℕ, x3 = k * x1) ∧
  (5 * x3 = x2 + 22) :=
  by {
    -- We are simply stating the equivalence and supporting conditions.
    -- Here, we will use 'sorry' as a placeholder.
    sorry
  }

end installation_quantities_l861_86153


namespace grazing_months_l861_86144

theorem grazing_months
    (total_rent : ℝ)
    (c_rent : ℝ)
    (a_oxen : ℕ)
    (a_months : ℕ)
    (b_oxen : ℕ)
    (c_oxen : ℕ)
    (c_months : ℕ)
    (b_months : ℝ)
    (total_oxen_months : ℝ) :
    total_rent = 140 ∧
    c_rent = 36 ∧
    a_oxen = 10 ∧
    a_months = 7 ∧
    b_oxen = 12 ∧
    c_oxen = 15 ∧
    c_months = 3 ∧
    c_rent / total_rent = (c_oxen * c_months) / total_oxen_months ∧
    total_oxen_months = (a_oxen * a_months) + (b_oxen * b_months) + (c_oxen * c_months)
    → b_months = 5 := by
    sorry

end grazing_months_l861_86144


namespace brenda_has_eight_l861_86117

-- Define the amounts each friend has
def emma_money : ℕ := 8
def daya_money : ℕ := emma_money + (emma_money / 4)
def jeff_money : ℕ := (2 * daya_money) / 5
def brenda_money : ℕ := jeff_money + 4

-- Define the theorem to prove Brenda's money is 8
theorem brenda_has_eight : brenda_money = 8 := by
  sorry

end brenda_has_eight_l861_86117


namespace line_through_point_l861_86105

-- Definitions for conditions
def point : (ℝ × ℝ) := (1, 2)

-- Function to check if a line equation holds for the given form 
def is_line_eq (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Main Lean theorem statement
theorem line_through_point (a b c : ℝ) :
  (∃ a b c, (is_line_eq a b c 1 2) ∧ 
           ((a = 1 ∧ b = 1 ∧ c = -3) ∨ (a = 2 ∧ b = -1 ∧ c = 0))) :=
sorry

end line_through_point_l861_86105


namespace probability_sum_less_than_product_l861_86171

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l861_86171


namespace multiplication_as_sum_of_squares_l861_86198

theorem multiplication_as_sum_of_squares :
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 := by
  sorry

end multiplication_as_sum_of_squares_l861_86198


namespace smallest_prime_perimeter_l861_86193

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_scalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_triple_prime (a b c : ℕ) : Prop := is_prime a ∧ is_prime b ∧ is_prime c

def is_triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

theorem smallest_prime_perimeter :
  ∃ a b c : ℕ, is_scalene a b c ∧ is_triple_prime a b c ∧ is_prime (a + b + c) ∧ a + b + c = 23 :=
sorry

end smallest_prime_perimeter_l861_86193


namespace eccentricity_of_hyperbola_l861_86124

theorem eccentricity_of_hyperbola {a b c e : ℝ} (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : b = 2 * a)
  (h₄ : c^2 = a^2 + b^2) :
  e = Real.sqrt 5 :=
by
  sorry

end eccentricity_of_hyperbola_l861_86124


namespace steps_from_center_to_square_l861_86179

-- Define the conditions and question in Lean 4
def steps_to_center := 354
def total_steps := 582

-- Prove that the steps from Rockefeller Center to Times Square is 228
theorem steps_from_center_to_square : (total_steps - steps_to_center) = 228 := by
  sorry

end steps_from_center_to_square_l861_86179


namespace probability_x_plus_2y_lt_6_l861_86139

noncomputable def prob_x_plus_2y_lt_6 : ℚ :=
  let rect_area : ℚ := (4 : ℚ) * 3
  let quad_area : ℚ := (4 : ℚ) * 1 + (1 / 2 : ℚ) * 4 * 2
  quad_area / rect_area

theorem probability_x_plus_2y_lt_6 :
  prob_x_plus_2y_lt_6 = 2 / 3 :=
by
  sorry

end probability_x_plus_2y_lt_6_l861_86139


namespace find_lost_card_number_l861_86116

theorem find_lost_card_number (n : ℕ) (S : ℕ) (x : ℕ) 
  (h1 : S = n * (n + 1) / 2) 
  (h2 : S - x = 101) 
  (h3 : n = 14) : 
  x = 4 := 
by sorry

end find_lost_card_number_l861_86116


namespace find_smaller_number_l861_86104

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 45) (h2 : b = 4 * a) : a = 9 :=
by
  sorry

end find_smaller_number_l861_86104


namespace parabola_directrix_tangent_circle_l861_86138

theorem parabola_directrix_tangent_circle (p : ℝ) (h_pos : 0 < p) (h_tangent: ∃ x : ℝ, (x = p/2) ∧ (x-5)^2 + (0:ℝ)^2 = 25) : p = 20 :=
sorry

end parabola_directrix_tangent_circle_l861_86138


namespace gcd_of_repeated_six_digit_integers_l861_86186

-- Given condition
def is_repeated_six_digit_integer (n : ℕ) : Prop :=
  100 ≤ n / 1000 ∧ n / 1000 < 1000 ∧ n = 1001 * (n / 1000)

-- Theorem to prove
theorem gcd_of_repeated_six_digit_integers :
  ∀ n : ℕ, is_repeated_six_digit_integer n → gcd n 1001 = 1001 :=
by sorry

end gcd_of_repeated_six_digit_integers_l861_86186


namespace establish_model_steps_correct_l861_86170

-- Define each step as a unique identifier
inductive Step : Type
| observe_pose_questions
| propose_assumptions
| express_properties
| test_or_revise

open Step

-- The sequence of steps to establish a mathematical model for population change
def correct_model_steps : List Step :=
  [observe_pose_questions, propose_assumptions, express_properties, test_or_revise]

-- The correct answer is the sequence of steps in the correct order
theorem establish_model_steps_correct :
  correct_model_steps = [observe_pose_questions, propose_assumptions, express_properties, test_or_revise] :=
  by sorry

end establish_model_steps_correct_l861_86170


namespace range_of_set_l861_86194

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l861_86194


namespace permutation_sum_eq_744_l861_86111

open Nat

theorem permutation_sum_eq_744 (n : ℕ) (h1 : n ≠ 0) (h2 : n + 3 ≤ 2 * n) (h3 : n + 1 ≤ 4) :
  choose (2 * n) (n + 3) + choose 4 (n + 1) = 744 := by
  sorry

end permutation_sum_eq_744_l861_86111


namespace sally_received_quarters_l861_86140

theorem sally_received_quarters : 
  ∀ (original_quarters total_quarters received_quarters : ℕ), 
  original_quarters = 760 → 
  total_quarters = 1178 → 
  received_quarters = total_quarters - original_quarters → 
  received_quarters = 418 :=
by 
  intros original_quarters total_quarters received_quarters h_original h_total h_received
  rw [h_original, h_total] at h_received
  exact h_received

end sally_received_quarters_l861_86140


namespace intersection_complement_l861_86174

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement : A ∩ (U \ B) = {1, 3} :=
by {
  sorry
}

end intersection_complement_l861_86174


namespace total_number_of_books_l861_86158

theorem total_number_of_books (history_books geography_books math_books : ℕ)
  (h1 : history_books = 32) (h2 : geography_books = 25) (h3 : math_books = 43) :
  history_books + geography_books + math_books = 100 :=
by
  -- the proof would go here but we use sorry to skip it
  sorry

end total_number_of_books_l861_86158


namespace cos_angle_between_vectors_l861_86196

theorem cos_angle_between_vectors :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (1, 3)
  let dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
  let magnitude (x : ℝ × ℝ) : ℝ := Real.sqrt (x.1 ^ 2 + x.2 ^ 2)
  let cos_theta := dot_product a b / (magnitude a * magnitude b)
  cos_theta = -Real.sqrt 2 / 10 :=
by
  sorry

end cos_angle_between_vectors_l861_86196


namespace abs_inequality_l861_86173

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end abs_inequality_l861_86173


namespace molecular_weight_of_N2O5_l861_86110

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def num_atoms_N : ℕ := 2
def num_atoms_O : ℕ := 5
def molecular_weight_N2O5 : ℝ := (num_atoms_N * atomic_weight_N) + (num_atoms_O * atomic_weight_O)

theorem molecular_weight_of_N2O5 : molecular_weight_N2O5 = 108.02 :=
by
  sorry

end molecular_weight_of_N2O5_l861_86110


namespace prime_transformation_l861_86141

theorem prime_transformation (p : ℕ) (prime_p : Nat.Prime p) (h : p = 3) : ∃ q : ℕ, q = 13 * p + 2 ∧ Nat.Prime q :=
by
  use 41
  sorry

end prime_transformation_l861_86141


namespace quadrilateral_offset_l861_86147

theorem quadrilateral_offset (d A h₂ x : ℝ)
  (h_da: d = 40)
  (h_A: A = 400)
  (h_h2 : h₂ = 9)
  (h_area : A = 1/2 * d * (x + h₂)) : 
  x = 11 :=
by sorry

end quadrilateral_offset_l861_86147


namespace mean_second_set_l861_86157

theorem mean_second_set (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 62) :
  (48 + 62 + 98 + 124 + x) / 5 = 78 :=
sorry

end mean_second_set_l861_86157


namespace quadratic_root_in_l861_86113

variable (a b c m : ℝ)

theorem quadratic_root_in (ha : a > 0) (hm : m > 0) 
  (h : a / (m + 2) + b / (m + 1) + c / m = 0) : 
  ∃ x, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 := 
by
  sorry

end quadratic_root_in_l861_86113


namespace find_abc_l861_86101

theorem find_abc (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h_eq : 10 * a + 11 * b + c = 25) : a = 0 ∧ b = 2 ∧ c = 3 := 
sorry

end find_abc_l861_86101


namespace find_k_value_l861_86154

theorem find_k_value (x y k : ℝ) 
  (h1 : x - 3 * y = k + 2) 
  (h2 : x - y = 4) 
  (h3 : 3 * x + y = -8) : 
  k = 12 := 
  by {
    sorry
  }

end find_k_value_l861_86154


namespace chord_count_l861_86178

theorem chord_count {n : ℕ} (h : n = 2024) : 
  ∃ k : ℕ, k ≥ 1024732 ∧ ∀ (i j : ℕ), (i < n → j < n → i ≠ j → true) := sorry

end chord_count_l861_86178


namespace find_smallest_angle_l861_86187

open Real

theorem find_smallest_angle :
  ∃ x : ℝ, (x > 0 ∧ sin (4 * x * (π / 180)) * sin (6 * x * (π / 180)) = cos (4 * x * (π / 180)) * cos (6 * x * (π / 180))) ∧ x = 9 :=
by
  sorry

end find_smallest_angle_l861_86187


namespace count_numbers_with_remainder_7_dividing_65_l861_86123

theorem count_numbers_with_remainder_7_dividing_65 : 
  (∃ n : ℕ, n > 7 ∧ n ∣ 58 ∧ 65 % n = 7) ∧ 
  (∀ m : ℕ, m > 7 ∧ m ∣ 58 ∧ 65 % m = 7 → m = 29 ∨ m = 58) :=
sorry

end count_numbers_with_remainder_7_dividing_65_l861_86123


namespace nearest_integer_to_power_sum_l861_86103

theorem nearest_integer_to_power_sum :
  let x := (3 + Real.sqrt 5)
  Int.floor ((x ^ 4) + 1 / 2) = 752 :=
by
  sorry

end nearest_integer_to_power_sum_l861_86103


namespace men_work_equivalence_l861_86185

theorem men_work_equivalence : 
  ∀ (M : ℕ) (m w : ℕ),
  (3 * w = 2 * m) ∧ 
  (M * 21 * 8 * m = 21 * 60 * 3 * w) →
  M = 15 := by
  intro M m w
  intro h
  sorry

end men_work_equivalence_l861_86185


namespace Olivia_steps_l861_86106

def round_to_nearest_ten (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

theorem Olivia_steps :
  let x := 57 + 68
  let y := x - 15
  round_to_nearest_ten y = 110 := 
by
  sorry

end Olivia_steps_l861_86106


namespace missing_fraction_l861_86163

-- Defining all the given fractions
def f1 : ℚ := 1 / 3
def f2 : ℚ := 1 / 2
def f3 : ℚ := 1 / 5
def f4 : ℚ := 1 / 4
def f5 : ℚ := -9 / 20
def f6 : ℚ := -5 / 6

-- Defining the total sum in decimal form
def total_sum : ℚ := 5 / 6  -- Since 0.8333333333333334 is equivalent to 5/6

-- Defining the sum of the given fractions
def given_sum : ℚ := f1 + f2 + f3 + f4 + f5 + f6

-- The Lean 4 statement to prove the missing fraction
theorem missing_fraction : ∃ x : ℚ, (given_sum + x = total_sum) ∧ x = 5 / 6 :=
by
  use 5 / 6
  constructor
  . sorry
  . rfl

end missing_fraction_l861_86163


namespace building_height_l861_86197

theorem building_height (flagpole_height : ℝ) (flagpole_shadow : ℝ) 
  (building_shadow : ℝ) (building_height : ℝ)
  (h_flagpole : flagpole_height = 18)
  (s_flagpole : flagpole_shadow = 45)
  (s_building : building_shadow = 70)
  (ratio_eq : flagpole_height / flagpole_shadow = building_height / building_shadow) :
  building_height = 28 :=
by
  have h_flagpole_shadow := ratio_eq ▸ h_flagpole ▸ s_flagpole ▸ s_building
  sorry

end building_height_l861_86197


namespace largest_common_number_in_sequences_from_1_to_200_l861_86195

theorem largest_common_number_in_sequences_from_1_to_200 :
  ∃ a, a ≤ 200 ∧ a % 8 = 3 ∧ a % 9 = 5 ∧ ∀ b, (b ≤ 200 ∧ b % 8 = 3 ∧ b % 9 = 5) → b ≤ a :=
sorry

end largest_common_number_in_sequences_from_1_to_200_l861_86195


namespace competition_score_l861_86137

theorem competition_score
    (x : ℕ)
    (h1 : 20 ≥ x)
    (h2 : 5 * x - (20 - x) = 70) :
    x = 15 :=
sorry

end competition_score_l861_86137


namespace triangle_ab_length_triangle_roots_quadratic_l861_86160

open Real

noncomputable def right_angled_triangle_length_ab (p s : ℝ) : ℝ :=
  (p / 2) - sqrt ((p / 2)^2 - 2 * s)

noncomputable def right_angled_triangle_quadratic (p s : ℝ) : Polynomial ℝ :=
  Polynomial.X^2 - Polynomial.C ((p / 2) + sqrt ((p / 2)^2 - 2 * s)) * Polynomial.X
    + Polynomial.C (2 * s)

theorem triangle_ab_length (p s : ℝ) :
  ∃ (AB : ℝ), AB = right_angled_triangle_length_ab p s ∧
    ∃ (AC BC : ℝ), (AC + BC + AB = p) ∧ (1 / 2 * BC * AC = s) :=
by
  use right_angled_triangle_length_ab p s
  sorry

theorem triangle_roots_quadratic (p s : ℝ) :
  ∃ (AC BC : ℝ), AC + BC = (p / 2) + sqrt ((p / 2)^2 - 2 * s) ∧
    AC * BC = 2 * s ∧
    (Polynomial.aeval AC (right_angled_triangle_quadratic p s) = 0) ∧
    (Polynomial.aeval BC (right_angled_triangle_quadratic p s) = 0) :=
by
  sorry

end triangle_ab_length_triangle_roots_quadratic_l861_86160


namespace problem_solution_l861_86102

variable {x y z : ℝ}

/-- Suppose that x, y, and z are three positive numbers that satisfy the given conditions.
    Prove that z + 1/y = 13/77. --/
theorem problem_solution (h1 : x * y * z = 1)
                         (h2 : x + 1 / z = 8)
                         (h3 : y + 1 / x = 29) :
  z + 1 / y = 13 / 77 := 
  sorry

end problem_solution_l861_86102


namespace charity_dinner_cost_l861_86143

def cost_of_rice_per_plate : ℝ := 0.10
def cost_of_chicken_per_plate : ℝ := 0.40
def number_of_plates : ℕ := 100

theorem charity_dinner_cost : 
  cost_of_rice_per_plate + cost_of_chicken_per_plate * number_of_plates = 50 :=
by
  sorry

end charity_dinner_cost_l861_86143


namespace expansion_coeff_l861_86125

theorem expansion_coeff (a b : ℝ) (x : ℝ) (h : (1 + a * x) ^ 5 = 1 + 10 * x + b * x^2 + a^5 * x^5) :
  b = 40 :=
sorry

end expansion_coeff_l861_86125


namespace initial_erasers_count_l861_86136

noncomputable def erasers_lost := 42
noncomputable def erasers_ended_up_with := 53

theorem initial_erasers_count (initial_erasers : ℕ) : 
  initial_erasers_ended_up_with = initial_erasers - erasers_lost → initial_erasers = 95 :=
by
  sorry

end initial_erasers_count_l861_86136


namespace gcd_1151_3079_l861_86151

def a : ℕ := 1151
def b : ℕ := 3079

theorem gcd_1151_3079 : gcd a b = 1 := by
  sorry

end gcd_1151_3079_l861_86151


namespace apples_for_48_oranges_l861_86112

theorem apples_for_48_oranges (o a : ℕ) (h : 8 * o = 6 * a) (ho : o = 48) : a = 36 :=
by
  sorry

end apples_for_48_oranges_l861_86112


namespace eight_in_M_nine_in_M_ten_not_in_M_l861_86118

def M (a : ℤ) : Prop := ∃ b c : ℤ, a = b^2 - c^2

theorem eight_in_M : M 8 := by
  sorry

theorem nine_in_M : M 9 := by
  sorry

theorem ten_not_in_M : ¬ M 10 := by
  sorry

end eight_in_M_nine_in_M_ten_not_in_M_l861_86118


namespace sqrt_x_div_sqrt_y_l861_86119

theorem sqrt_x_div_sqrt_y (x y : ℝ)
  (h : ( ( (2/3)^2 + (1/6)^2 ) / ( (1/2)^2 + (1/7)^2 ) ) = 28 * x / (25 * y)) :
  (Real.sqrt x) / (Real.sqrt y) = 5 / 2 :=
sorry

end sqrt_x_div_sqrt_y_l861_86119


namespace parallel_line_dividing_triangle_l861_86199

theorem parallel_line_dividing_triangle (base : ℝ) (length_parallel_line : ℝ) 
    (h_base : base = 24) 
    (h_parallel : (length_parallel_line / base)^2 = 1/2) : 
    length_parallel_line = 12 * Real.sqrt 2 :=
sorry

end parallel_line_dividing_triangle_l861_86199


namespace no_such_set_exists_l861_86165

open Nat Set

theorem no_such_set_exists (M : Set ℕ) : 
  (∀ m : ℕ, m > 1 → ∃ a b : ℕ, a ∈ M ∧ b ∈ M ∧ a + b = m) →
  (∀ a b c d : ℕ, a ∈ M → b ∈ M → c ∈ M → d ∈ M → 
    a > 10 → b > 10 → c > 10 → d > 10 → a + b = c + d → a = c ∨ a = d) → 
  False := by
  sorry

end no_such_set_exists_l861_86165


namespace tank_fill_time_l861_86129

theorem tank_fill_time :
  let fill_rate_A := 1 / 8
  let empty_rate_B := 1 / 24
  let combined_rate := fill_rate_A - empty_rate_B
  let time_with_both_pipes := 66
  let partial_fill := time_with_both_pipes * combined_rate
  let remaining_fill := 1 - (partial_fill % 1)
  let additional_time_A := remaining_fill / fill_rate_A
  time_with_both_pipes + additional_time_A = 70 :=
by
  let fill_rate_A := 1 / 8
  let empty_rate_B := 1 / 24
  let combined_rate := fill_rate_A - empty_rate_B
  let time_with_both_pipes := 66
  let partial_fill := time_with_both_pipes * combined_rate
  let remaining_fill := 1 - (partial_fill % 1)
  let additional_time_A := remaining_fill / fill_rate_A
  have h : time_with_both_pipes + additional_time_A = 70 := sorry
  exact h

end tank_fill_time_l861_86129


namespace fractions_comparison_l861_86181

theorem fractions_comparison : 
  (99 / 100 < 100 / 101) ∧ (100 / 101 > 199 / 201) ∧ (99 / 100 < 199 / 201) :=
by sorry

end fractions_comparison_l861_86181


namespace second_smallest_odd_number_l861_86176

-- Define the conditions
def four_consecutive_odd_numbers_sum (n : ℕ) : Prop := 
  n % 2 = 1 ∧ (n + (n + 2) + (n + 4) + (n + 6) = 112)

-- State the theorem
theorem second_smallest_odd_number (n : ℕ) (h : four_consecutive_odd_numbers_sum n) : n + 2 = 27 :=
sorry

end second_smallest_odd_number_l861_86176


namespace optimal_washing_effect_l861_86167

noncomputable def total_capacity : ℝ := 20 -- kilograms
noncomputable def weight_clothes : ℝ := 5 -- kilograms
noncomputable def weight_detergent_existing : ℝ := 2 * 0.02 -- kilograms
noncomputable def optimal_concentration : ℝ := 0.004 -- kilograms per kilogram of water

theorem optimal_washing_effect :
  ∃ (additional_detergent additional_water : ℝ),
    additional_detergent = 0.02 ∧ additional_water = 14.94 ∧
    weight_clothes + additional_water + weight_detergent_existing + additional_detergent = total_capacity ∧
    weight_detergent_existing + additional_detergent = optimal_concentration * additional_water :=
by
  sorry

end optimal_washing_effect_l861_86167


namespace cube_edge_length_surface_area_equals_volume_l861_86192

theorem cube_edge_length_surface_area_equals_volume (a : ℝ) (h : 6 * a ^ 2 = a ^ 3) : a = 6 := 
by {
  sorry
}

end cube_edge_length_surface_area_equals_volume_l861_86192


namespace range_of_m_l861_86146

theorem range_of_m {m : ℝ} (h : ∃ x : ℝ, 2 < x ∧ x < 3 ∧ x^2 + 2 * x - m = 0) : 8 < m ∧ m < 15 :=
sorry

end range_of_m_l861_86146
