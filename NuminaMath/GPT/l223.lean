import Mathlib

namespace completing_the_square_correct_l223_22372

theorem completing_the_square_correct :
  (∃ x : ℝ, x^2 - 6 * x + 5 = 0) →
  (∃ x : ℝ, (x - 3)^2 = 4) :=
by
  sorry

end completing_the_square_correct_l223_22372


namespace find_ac_pair_l223_22330

theorem find_ac_pair (a c : ℤ) (h1 : a + c = 37) (h2 : a < c) (h3 : 36^2 - 4 * a * c = 0) : a = 12 ∧ c = 25 :=
by
  sorry

end find_ac_pair_l223_22330


namespace solve_linear_system_l223_22381

variable {a b : ℝ}
variables {m n : ℝ}

theorem solve_linear_system
  (h1 : a * 2 - b * 1 = 3)
  (h2 : a * 2 + b * 1 = 5)
  (h3 : a * (m + 2 * n) - 2 * b * n = 6)
  (h4 : a * (m + 2 * n) + 2 * b * n = 10) :
  m = 2 ∧ n = 1 := 
sorry

end solve_linear_system_l223_22381


namespace fraction_simplification_l223_22357

theorem fraction_simplification :
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16) /
  (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 :=
by
  sorry

end fraction_simplification_l223_22357


namespace part1_part2_l223_22373

theorem part1 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (1 / a^2 + 1 / b^2 ≥ 1 / 2) :=
sorry

theorem part2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = a * b ∧ (|2 * a - 1| + |3 * b - 1| = 2 * Real.sqrt 6 + 3)) :=
sorry

end part1_part2_l223_22373


namespace find_mean_of_two_l223_22323

-- Define the set of numbers
def numbers : List ℕ := [1879, 1997, 2023, 2029, 2113, 2125]

-- Define the mean of the four selected numbers
def mean_of_four : ℕ := 2018

-- Define the sum of all numbers
def total_sum : ℕ := numbers.sum

-- Define the sum of the four numbers with a given mean
def sum_of_four : ℕ := 4 * mean_of_four

-- Define the sum of the remaining two numbers
def sum_of_two (total sum_of_four : ℕ) : ℕ := total - sum_of_four

-- Define the mean of the remaining two numbers
def mean_of_two (sum_two : ℕ) : ℕ := sum_two / 2

-- Define the condition theorem to be proven
theorem find_mean_of_two : mean_of_two (sum_of_two total_sum sum_of_four) = 2047 := 
by
  sorry

end find_mean_of_two_l223_22323


namespace degree_of_g_l223_22337

theorem degree_of_g (f g : Polynomial ℝ) (h : Polynomial ℝ) (H1 : h = f.comp g + g) 
  (H2 : h.natDegree = 6) (H3 : f.natDegree = 3) : g.natDegree = 2 := 
sorry

end degree_of_g_l223_22337


namespace original_number_l223_22343

theorem original_number (x : ℤ) (h : (x - 5) / 4 = (x - 4) / 5) : x = 9 :=
sorry

end original_number_l223_22343


namespace third_factor_of_product_l223_22369

theorem third_factor_of_product (w : ℕ) (h_w_pos : w > 0) (h_w_168 : w = 168)
  (w_factors : (936 * w) = 2^5 * 3^3 * x)
  (h36_factors : 2^5 ∣ (936 * w)) (h33_factors : 3^3 ∣ (936 * w)) : 
  (936 * w) / (2^5 * 3^3) = 182 :=
by {
  -- This is a placeholder. The actual proof is omitted.
  sorry
}

end third_factor_of_product_l223_22369


namespace factor_expression_l223_22361

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) :=
by
  sorry

end factor_expression_l223_22361


namespace regular_polygon_sides_l223_22365

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) 
(h_interior : (n - 2) * 180 / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l223_22365


namespace find_first_train_length_l223_22352

namespace TrainProblem

-- Define conditions
def speed_first_train_kmph := 42
def speed_second_train_kmph := 48
def length_second_train_m := 163
def time_clear_s := 12
def relative_speed_kmph := speed_first_train_kmph + speed_second_train_kmph

-- Convert kmph to m/s
def kmph_to_mps(kmph : ℕ) : ℕ := kmph * 5 / 18
def relative_speed_mps := kmph_to_mps relative_speed_kmph

-- Calculate total distance covered by the trains in meters
def total_distance_m := relative_speed_mps * time_clear_s

-- Define the length of the first train to be proved
def length_first_train_m := 137

-- Theorem statement
theorem find_first_train_length :
  total_distance_m = length_first_train_m + length_second_train_m :=
sorry

end TrainProblem

end find_first_train_length_l223_22352


namespace negation_example_l223_22364

theorem negation_example (p : ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) :
  ∃ x0 : ℝ, x0 > 0 ∧ (x0 + 1) * Real.exp x0 ≤ 1 :=
sorry

end negation_example_l223_22364


namespace delta_y_over_delta_x_l223_22398

def curve (x : ℝ) : ℝ := x^2 + x

theorem delta_y_over_delta_x (Δx Δy : ℝ) 
  (hQ : (2 + Δx, 6 + Δy) = (2 + Δx, curve (2 + Δx)))
  (hP : 6 = curve 2) : 
  (Δy / Δx) = Δx + 5 :=
by
  sorry

end delta_y_over_delta_x_l223_22398


namespace find_value_of_m_l223_22326

/-- Given the parabola y = 4x^2 + 4x + 5 and the line y = 8mx + 8m intersect at exactly one point,
    prove the value of m^{36} + 1155 / m^{12} is 39236. -/
theorem find_value_of_m (m : ℝ) (h: ∃ x, 4 * x^2 + 4 * x + 5 = 8 * m * x + 8 * m ∧
  ∀ x₁ x₂, 4 * x₁^2 + 4 * x₁ + 5 = 8 * m * x₁ + 8 * m →
  4 * x₂^2 + 4 * x₂ + 5 = 8 * m * x₂ + 8 * m → x₁ = x₂) :
  m^36 + 1155 / m^12 = 39236 := 
sorry

end find_value_of_m_l223_22326


namespace beta_greater_than_alpha_l223_22328

theorem beta_greater_than_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2) (h5 : Real.sin (α + β) = 2 * Real.sin α) : β > α := 
sorry

end beta_greater_than_alpha_l223_22328


namespace tan_trig_identity_l223_22329

noncomputable def given_condition (α : ℝ) : Prop :=
  Real.tan (α + Real.pi / 3) = 2

theorem tan_trig_identity (α : ℝ) (h : given_condition α) :
  (Real.sin (α + (4 * Real.pi / 3)) + Real.cos ((2 * Real.pi / 3) - α)) /
  (Real.cos ((Real.pi / 6) - α) - Real.sin (α + (5 * Real.pi / 6))) = -3 :=
sorry

end tan_trig_identity_l223_22329


namespace _l223_22378

noncomputable def polynomial_divides (x : ℂ) (n : ℕ) : Prop :=
  (x - 1) ^ 3 ∣ x ^ (2 * n + 1) - (2 * n + 1) * x ^ (n + 1) + (2 * n + 1) * x ^ n - 1

lemma polynomial_division_theorem : ∀ (n : ℕ), n ≥ 1 → ∀ (x : ℂ), polynomial_divides x n :=
by
  intros n hn x
  unfold polynomial_divides
  sorry

end _l223_22378


namespace solution_set_quadratic_inequality_l223_22316

theorem solution_set_quadratic_inequality :
  {x : ℝ | -x^2 + 5*x + 6 > 0} = {x : ℝ | -1 < x ∧ x < 6} :=
sorry

end solution_set_quadratic_inequality_l223_22316


namespace max_buses_in_city_l223_22367

theorem max_buses_in_city (num_stops stops_per_bus shared_stops : ℕ) (h_stops : num_stops = 9) (h_stops_per_bus : stops_per_bus = 3) (h_shared_stops : shared_stops = 1) : 
  ∃ (max_buses : ℕ), max_buses = 12 :=
by
  sorry

end max_buses_in_city_l223_22367


namespace parabola_equation_l223_22320

noncomputable def parabola_focus : (ℝ × ℝ) := (5, -2)

noncomputable def parabola_directrix (x y : ℝ) : Prop := 4 * x - 5 * y = 20

theorem parabola_equation (x y : ℝ) :
  (parabola_focus = (5, -2)) →
  (parabola_directrix x y) →
  25 * x^2 + 40 * x * y + 16 * y^2 - 650 * x + 184 * y + 1009 = 0 :=
by
  sorry

end parabola_equation_l223_22320


namespace vehicle_distribution_l223_22399

theorem vehicle_distribution :
  ∃ B T U : ℕ, 2 * B + 3 * T + U = 18 ∧ ∀ n : ℕ, n ≤ 18 → ∃ t : ℕ, ∃ (u : ℕ), 2 * (n - t) + u = 18 ∧ 2 * Nat.gcd t u + 3 * t + u = 18 ∧
  10 + 8 + 7 + 5 + 4 + 2 + 1 = 37 := by
  sorry

end vehicle_distribution_l223_22399


namespace chessboard_polygon_l223_22370

-- Conditions
variable (A B a b : ℕ)

-- Statement of the theorem
theorem chessboard_polygon (A B a b : ℕ) : A - B = 4 * (a - b) :=
sorry

end chessboard_polygon_l223_22370


namespace club_additional_members_l223_22303

theorem club_additional_members (current_members additional_members future_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 15) 
  (h3 : future_members = current_members + additional_members) : 
  future_members - current_members = 15 :=
by
  sorry

end club_additional_members_l223_22303


namespace probability_black_white_l223_22348

structure Jar :=
  (black_balls : ℕ)
  (white_balls : ℕ)
  (green_balls : ℕ)

def total_balls (j : Jar) : ℕ :=
  j.black_balls + j.white_balls + j.green_balls

def choose (n k : ℕ) : ℕ := n.choose k

theorem probability_black_white (j : Jar) (h_black : j.black_balls = 3) (h_white : j.white_balls = 3) (h_green : j.green_balls = 1) :
  (choose 3 1 * choose 3 1) / (choose (total_balls j) 2) = 3 / 7 :=
by
  sorry

end probability_black_white_l223_22348


namespace g_of_10_l223_22322

noncomputable def g : ℕ → ℝ := sorry

axiom g_initial : g 1 = 2

axiom g_condition : ∀ (m n : ℕ), m ≥ n → g (m + n) + g (m - n) = 2 * g m + 3 * g n

theorem g_of_10 : g 10 = 496 :=
by
  sorry

end g_of_10_l223_22322


namespace find_d_l223_22347

theorem find_d (c : ℝ) (d : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ) (ω : ℝ)  
  (h1 : α = c) 
  (h2 : β = 43)
  (h3 : γ = 59)
  (h4 : ω = d)
  (h5 : α + d + β + γ = 180) :
  d = 42 :=
by
  sorry

end find_d_l223_22347


namespace arithmetic_series_first_term_l223_22306

theorem arithmetic_series_first_term :
  ∃ (a d : ℝ), (25 * (2 * a + 49 * d) = 200) ∧ (25 * (2 * a + 149 * d) = 2700) ∧ (a = -20.5) :=
by
  sorry

end arithmetic_series_first_term_l223_22306


namespace cistern_water_depth_l223_22346

theorem cistern_water_depth:
  ∀ h: ℝ,
  (4 * 4 + 4 * h * 4 + 4 * h * 4 = 36) → h = 1.25 := by
    sorry

end cistern_water_depth_l223_22346


namespace negation_prop_l223_22395

theorem negation_prop : (¬(∃ x : ℝ, x + 2 ≤ 0)) ↔ (∀ x : ℝ, x + 2 > 0) := 
  sorry

end negation_prop_l223_22395


namespace emily_second_round_points_l223_22300

theorem emily_second_round_points (P : ℤ)
  (first_round_points : ℤ := 16)
  (last_round_points_lost : ℤ := 48)
  (end_points : ℤ := 1)
  (points_equation : first_round_points + P - last_round_points_lost = end_points) :
  P = 33 :=
  by {
    sorry
  }

end emily_second_round_points_l223_22300


namespace mary_max_earnings_l223_22341

theorem mary_max_earnings
  (max_hours : ℕ)
  (regular_rate : ℕ)
  (overtime_rate_increase_percent : ℕ)
  (first_hours : ℕ)
  (total_max_hours : ℕ)
  (total_hours_payable : ℕ) :
  max_hours = 60 →
  regular_rate = 8 →
  overtime_rate_increase_percent = 25 →
  first_hours = 20 →
  total_max_hours = 60 →
  total_hours_payable = 560 →
  ((first_hours * regular_rate) + ((total_max_hours - first_hours) * (regular_rate + (regular_rate * overtime_rate_increase_percent / 100)))) = total_hours_payable :=
by
  intros
  sorry

end mary_max_earnings_l223_22341


namespace number_of_roots_in_right_half_plane_is_one_l223_22342

def Q5 (z : ℂ) : ℂ := z^5 + z^4 + 2*z^3 - 8*z - 1

theorem number_of_roots_in_right_half_plane_is_one :
  (∃ n, ∀ z, Q5 z = 0 ∧ z.re > 0 ↔ n = 1) := 
sorry

end number_of_roots_in_right_half_plane_is_one_l223_22342


namespace position_of_2010_is_correct_l223_22335

-- Definition of the arithmetic sequence and row starting points
def first_term : Nat := 1
def common_difference : Nat := 2
def S (n : Nat) : Nat := (n * (2 * first_term + (n - 1) * common_difference)) / 2

-- Definition of the position where number 2010 appears
def row_of_number (x : Nat) : Nat :=
  let n := (Nat.sqrt x) + 1
  if (n - 1) * (n - 1) < x && x <= n * n then n else n - 1

def column_of_number (x : Nat) : Nat :=
  let row := row_of_number x
  x - (S (row - 1)) + 1

-- Main theorem
theorem position_of_2010_is_correct :
  row_of_number 2010 = 45 ∧ column_of_number 2010 = 74 :=
by
  sorry

end position_of_2010_is_correct_l223_22335


namespace car_speed_is_104_mph_l223_22319

noncomputable def speed_of_car_in_mph
  (fuel_efficiency_km_per_liter : ℝ) -- car travels 64 km per liter
  (fuel_consumption_gallons : ℝ) -- fuel tank decreases by 3.9 gallons
  (time_hours : ℝ) -- period of 5.7 hours
  (gallon_to_liter : ℝ) -- 1 gallon is 3.8 liters
  (km_to_mile : ℝ) -- 1 mile is 1.6 km
  : ℝ :=
  let fuel_consumption_liters := fuel_consumption_gallons * gallon_to_liter
  let distance_km := fuel_efficiency_km_per_liter * fuel_consumption_liters
  let distance_miles := distance_km / km_to_mile
  let speed_mph := distance_miles / time_hours
  speed_mph

theorem car_speed_is_104_mph 
  (fuel_efficiency_km_per_liter : ℝ := 64)
  (fuel_consumption_gallons : ℝ := 3.9)
  (time_hours : ℝ := 5.7)
  (gallon_to_liter : ℝ := 3.8)
  (km_to_mile : ℝ := 1.6)
  : speed_of_car_in_mph fuel_efficiency_km_per_liter fuel_consumption_gallons time_hours gallon_to_liter km_to_mile = 104 :=
  by
    sorry

end car_speed_is_104_mph_l223_22319


namespace intersection_point_l223_22334

/-- Coordinates of points A, B, C, and D -/
def pointA : Fin 3 → ℝ := ![3, -2, 4]
def pointB : Fin 3 → ℝ := ![13, -12, 9]
def pointC : Fin 3 → ℝ := ![1, 6, -8]
def pointD : Fin 3 → ℝ := ![3, -1, 2]

/-- Prove the intersection point of the lines AB and CD is (-7, 8, -1) -/
theorem intersection_point :
  let lineAB (t : ℝ) := pointA + t • (pointB - pointA)
  let lineCD (s : ℝ) := pointC + s • (pointD - pointC)
  ∃ t s : ℝ, lineAB t = lineCD s ∧ lineAB t = ![-7, 8, -1] :=
sorry

end intersection_point_l223_22334


namespace find_difference_condition_l223_22324

variable (a b c : ℝ)

theorem find_difference_condition (h1 : (a + b) / 2 = 40) (h2 : (b + c) / 2 = 60) : c - a = 40 := by
  sorry

end find_difference_condition_l223_22324


namespace trajectory_equation_l223_22344

noncomputable def circle1_center := (-3, 0)
noncomputable def circle2_center := (3, 0)

def circle1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

def is_tangent_internally (x y : ℝ) : Prop := 
  ∃ (P : ℝ × ℝ), circle1 P.1 P.2 ∧ circle2 P.1 P.2

theorem trajectory_equation :
  ∀ (x y : ℝ), is_tangent_internally x y → (x^2 / 16 + y^2 / 7 = 1) :=
sorry

end trajectory_equation_l223_22344


namespace equal_roots_quadratic_l223_22317

theorem equal_roots_quadratic {k : ℝ} 
  (h : (∃ x : ℝ, x^2 - 6 * x + k = 0 ∧ x^2 - 6 * x + k = 0)) : 
  k = 9 :=
sorry

end equal_roots_quadratic_l223_22317


namespace sum_series_eq_half_l223_22358

theorem sum_series_eq_half :
  ∑' n : ℕ, (3^(n+1) / (9^(n+1) - 1)) = 1/2 := 
sorry

end sum_series_eq_half_l223_22358


namespace enhanced_inequality_l223_22384

theorem enhanced_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * a^2 / (b + c) + 2 * b^2 / (c + a) + 2 * c^2 / (a + b) ≥ a + b + c + (2 * a - b - c)^2 / (a + b + c)) :=
sorry

end enhanced_inequality_l223_22384


namespace no_integer_solution_l223_22391

theorem no_integer_solution (x y : ℤ) : 2 * x + 6 * y ≠ 91 :=
by
  sorry

end no_integer_solution_l223_22391


namespace arithmetic_sequence_a13_l223_22350

theorem arithmetic_sequence_a13 (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : a 5 = 3) (h2 : a 9 = 6) 
  (h3 : ∀ n, a n = a1 + (n - 1) * d) : 
  a 13 = 9 :=
sorry

end arithmetic_sequence_a13_l223_22350


namespace problem_l223_22376

variable (a b c : ℝ)

theorem problem (h : a^2 * b^2 + 18 * a * b * c > 4 * b^3 + 4 * a^3 * c + 27 * c^2) : a^2 > 3 * b :=
by
  sorry

end problem_l223_22376


namespace least_clock_equivalent_l223_22333

theorem least_clock_equivalent (x : ℕ) : 
  x > 3 ∧ x % 12 = (x * x) % 12 → x = 12 := 
by
  sorry

end least_clock_equivalent_l223_22333


namespace diagonals_of_angle_bisectors_l223_22339

theorem diagonals_of_angle_bisectors (a b : ℝ) (BAD ABC : ℝ) (hBAD : BAD = ABC) :
  ∃ d : ℝ, d = |a - b| :=
by
  sorry

end diagonals_of_angle_bisectors_l223_22339


namespace perpendicular_lines_k_value_l223_22321

theorem perpendicular_lines_k_value :
  ∀ (k : ℝ), (∀ (x y : ℝ), x + 4 * y - 1 = 0) →
             (∀ (x y : ℝ), k * x + y + 2 = 0) →
             (-1 / 4 * -k = -1) →
             k = -4 :=
by
  intros k h1 h2 h3
  sorry

end perpendicular_lines_k_value_l223_22321


namespace base_length_of_parallelogram_l223_22353

theorem base_length_of_parallelogram 
  (area : ℝ) (base altitude : ℝ) 
  (h_area : area = 242)
  (h_altitude : altitude = 2 * base) :
  base = 11 :=
by
  sorry

end base_length_of_parallelogram_l223_22353


namespace cylindrical_to_rectangular_conversion_l223_22394

theorem cylindrical_to_rectangular_conversion 
  (r θ z : ℝ) 
  (h1 : r = 10) 
  (h2 : θ = Real.pi / 3) 
  (h3 : z = -2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (5, 5 * Real.sqrt 3, -2) :=
by
  sorry

end cylindrical_to_rectangular_conversion_l223_22394


namespace max_value_of_3x_plus_4y_on_curve_C_l223_22387

theorem max_value_of_3x_plus_4y_on_curve_C :
  ∀ (x y : ℝ),
  (∃ (ρ θ : ℝ), ρ^2 = 36 / (4 * (Real.cos θ)^2 + 9 * (Real.sin θ)^2) ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (P : ℝ × ℝ) →
  (P = (x, y)) →
  3 * x + 4 * y ≤ Real.sqrt 145 ∧ ∃ (α : ℝ), 0 ≤ α ∧ α < 2 * Real.pi ∧ 3 * x + 4 * y = Real.sqrt 145 := 
by
  intros x y h_exists P hP
  sorry

end max_value_of_3x_plus_4y_on_curve_C_l223_22387


namespace total_votes_l223_22315

variable (V : ℝ)

theorem total_votes (h1 : 0.34 * V + 640 = 0.66 * V) : V = 2000 :=
by 
  sorry

end total_votes_l223_22315


namespace one_add_i_cubed_eq_one_sub_i_l223_22397

theorem one_add_i_cubed_eq_one_sub_i (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end one_add_i_cubed_eq_one_sub_i_l223_22397


namespace handshake_count_l223_22356

-- Define the number of team members, referees, and the total number of handshakes
def num_team_members := 7
def num_referees := 3
def num_coaches := 2

-- Calculate the handshakes
def team_handshakes := num_team_members * num_team_members
def player_refhandshakes := (2 * num_team_members) * num_referees
def coach_handshakes := num_coaches * (2 * num_team_members + num_referees)

-- The total number of handshakes
def total_handshakes := team_handshakes + player_refhandshakes + coach_handshakes

-- The proof statement
theorem handshake_count : total_handshakes = 125 := 
by
  -- Placeholder for proof
  sorry

end handshake_count_l223_22356


namespace find_constants_l223_22393

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x + 2)

theorem find_constants (a : ℝ) (x : ℝ) (h : x ≠ -2) :
  f a (f a x) = x ∧ a = -4 :=
by
  sorry

end find_constants_l223_22393


namespace solve_for_a_l223_22390

open Set

theorem solve_for_a (a : ℝ) :
  let M := ({a^2, a + 1, -3} : Set ℝ)
  let P := ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ)
  M ∩ P = {-3} →
  a = -1 :=
by
  intros M P h
  have hM : M = {a^2, a + 1, -3} := rfl
  have hP : P = {a - 3, 2 * a - 1, a^2 + 1} := rfl
  rw [hM, hP] at h
  sorry

end solve_for_a_l223_22390


namespace parallel_transitive_l223_22396

-- Definition of parallel lines
def are_parallel (l1 l2 : Line) : Prop :=
  ∃ (P : Line), l1 = P ∧ l2 = P

-- Theorem stating that if two lines are parallel to the same line, then they are parallel to each other
theorem parallel_transitive (l1 l2 l3 : Line) (h1 : are_parallel l1 l3) (h2 : are_parallel l2 l3) :
  are_parallel l1 l2 :=
by
  sorry

end parallel_transitive_l223_22396


namespace mean_height_is_68_l223_22371

/-
Given the heights of the volleyball players:
  heights_50s = [58, 59]
  heights_60s = [60, 61, 62, 65, 65, 66, 67]
  heights_70s = [70, 71, 71, 72, 74, 75, 79, 79]

We need to prove that the mean height of the players is 68 inches.
-/
def heights_50s : List ℕ := [58, 59]
def heights_60s : List ℕ := [60, 61, 62, 65, 65, 66, 67]
def heights_70s : List ℕ := [70, 71, 71, 72, 74, 75, 79, 79]

def total_heights : List ℕ := heights_50s ++ heights_60s ++ heights_70s
def number_of_players : ℕ := total_heights.length
def total_height : ℕ := total_heights.sum
def mean_height : ℕ := total_height / number_of_players

theorem mean_height_is_68 : mean_height = 68 := by
  sorry

end mean_height_is_68_l223_22371


namespace probability_not_passing_l223_22331

theorem probability_not_passing (P_passing : ℚ) (h : P_passing = 4/7) : (1 - P_passing = 3/7) :=
by
  rw [h]
  norm_num

end probability_not_passing_l223_22331


namespace centroid_of_triangle_l223_22385

theorem centroid_of_triangle :
  let x1 := 9
  let y1 := -8
  let x2 := -5
  let y2 := 6
  let x3 := 4
  let y3 := -3
  ( (x1 + x2 + x3) / 3 = 8 / 3 ∧ (y1 + y2 + y3) / 3 = -5 / 3 ) :=
by
  let x1 := 9
  let y1 := -8
  let x2 := -5
  let y2 := 6
  let x3 := 4
  let y3 := -3
  have centroid_x : (x1 + x2 + x3) / 3 = 8 / 3 := sorry
  have centroid_y : (y1 + y2 + y3) / 3 = -5 / 3 := sorry
  exact ⟨centroid_x, centroid_y⟩

end centroid_of_triangle_l223_22385


namespace rolling_circle_trace_eq_envelope_l223_22332

-- Definitions for the geometrical setup
variable {a : ℝ} (C : ℝ → ℝ → Prop)

-- The main statement to prove
theorem rolling_circle_trace_eq_envelope (hC : ∀ t : ℝ, C (a * t) a) :
  ∃ P : ℝ × ℝ → Prop, ∀ t : ℝ, C (a/2 * t + a/2 * Real.sin t) (a/2 + a/2 * Real.cos t) :=
by
  sorry

end rolling_circle_trace_eq_envelope_l223_22332


namespace jessica_withdrew_200_l223_22360

noncomputable def initial_balance (final_balance : ℝ) : ℝ :=
  (final_balance * 25 / 18)

noncomputable def withdrawn_amount (initial_balance : ℝ) : ℝ :=
  (initial_balance * 2 / 5)

theorem jessica_withdrew_200 :
  ∀ (final_balance : ℝ), final_balance = 360 → withdrawn_amount (initial_balance final_balance) = 200 :=
by
  intros final_balance h
  rw [h]
  unfold initial_balance withdrawn_amount
  sorry

end jessica_withdrew_200_l223_22360


namespace contrapositive_prop_l223_22325

theorem contrapositive_prop {α : Type} [Mul α] [Zero α] (a b : α) : 
  (a = 0 → a * b = 0) ↔ (a * b ≠ 0 → a ≠ 0) :=
by sorry

end contrapositive_prop_l223_22325


namespace lcm_gcd_product_12_15_l223_22305

theorem lcm_gcd_product_12_15 : 
  let a := 12
  let b := 15
  lcm a b * gcd a b = 180 :=
by
  sorry

end lcm_gcd_product_12_15_l223_22305


namespace solve_for_a_l223_22311

theorem solve_for_a (a : ℚ) (h : a + a/3 + a/4 = 11/4) : a = 33/19 :=
sorry

end solve_for_a_l223_22311


namespace center_of_circle_l223_22345

theorem center_of_circle (x y : ℝ) : x^2 - 8 * x + y^2 - 4 * y = 4 → (x, y) = (4, 2) :=
by
  sorry

end center_of_circle_l223_22345


namespace beavers_still_working_is_one_l223_22380

def initial_beavers : Nat := 2
def beavers_swimming : Nat := 1
def still_working_beavers : Nat := initial_beavers - beavers_swimming

theorem beavers_still_working_is_one : still_working_beavers = 1 :=
by
  sorry

end beavers_still_working_is_one_l223_22380


namespace sin_alpha_plus_pi_over_4_tan_double_alpha_l223_22354

-- Definitions of sin and tan 
open Real

variable (α : ℝ)

-- Given conditions
axiom α_in_interval : 0 < α ∧ α < π / 2
axiom sin_alpha_def : sin α = sqrt 5 / 5

-- Statement to prove
theorem sin_alpha_plus_pi_over_4 : sin (α + π / 4) = 3 * sqrt 10 / 10 :=
by
  sorry

theorem tan_double_alpha : tan (2 * α) = 4 / 3 :=
by
  sorry

end sin_alpha_plus_pi_over_4_tan_double_alpha_l223_22354


namespace find_number_l223_22386

theorem find_number (x : ℝ) :
  9 * (((x + 1.4) / 3) - 0.7) = 5.4 ↔ x = 2.5 :=
by sorry

end find_number_l223_22386


namespace option_d_necessary_sufficient_l223_22368

theorem option_d_necessary_sufficient (a : ℝ) : (a ≠ 0) ↔ (∃! x : ℝ, a * x = 1) := 
sorry

end option_d_necessary_sufficient_l223_22368


namespace bruce_paid_correct_amount_l223_22310

-- Define the conditions
def kg_grapes : ℕ := 8
def cost_per_kg_grapes : ℕ := 70
def kg_mangoes : ℕ := 8
def cost_per_kg_mangoes : ℕ := 55

-- Calculate partial costs
def cost_grapes := kg_grapes * cost_per_kg_grapes
def cost_mangoes := kg_mangoes * cost_per_kg_mangoes
def total_paid := cost_grapes + cost_mangoes

-- The theorem to prove
theorem bruce_paid_correct_amount : total_paid = 1000 := 
by 
  -- Merge several logical steps into one
  -- sorry can be used for incomplete proof
  sorry

end bruce_paid_correct_amount_l223_22310


namespace allan_correct_answers_l223_22351

theorem allan_correct_answers (x y : ℕ) (h1 : x + y = 120) (h2 : x - (0.25 : ℝ) * y = 100) : x = 104 :=
by
  sorry

end allan_correct_answers_l223_22351


namespace work_problem_l223_22377

-- Definition of the conditions and the problem statement
theorem work_problem (P D : ℕ)
  (h1 : ∀ (P : ℕ), ∀ (D : ℕ), (2 * P) * 6 = P * D * 1 / 2) : 
  D = 24 :=
by
  sorry

end work_problem_l223_22377


namespace sue_charge_per_dog_l223_22314

def amount_saved_christian : ℝ := 5
def amount_saved_sue : ℝ := 7
def charge_per_yard : ℝ := 5
def yards_mowed_christian : ℝ := 4
def total_cost_perfume : ℝ := 50
def additional_amount_needed : ℝ := 6
def dogs_walked_sue : ℝ := 6

theorem sue_charge_per_dog :
  (amount_saved_christian + (charge_per_yard * yards_mowed_christian) + amount_saved_sue + (dogs_walked_sue * x) + additional_amount_needed = total_cost_perfume) → x = 2 :=
by
  sorry

end sue_charge_per_dog_l223_22314


namespace square_tiles_count_l223_22336

theorem square_tiles_count (t s p : ℕ) (h1 : t + s + p = 30) (h2 : 3 * t + 4 * s + 5 * p = 108) : s = 6 := by
  sorry

end square_tiles_count_l223_22336


namespace sufficient_conditions_for_quadratic_l223_22366

theorem sufficient_conditions_for_quadratic (x : ℝ) : 
  (0 < x ∧ x < 4) ∨ (-2 < x ∧ x < 4) ∨ (-2 < x ∧ x < 3) → x^2 - 2*x - 8 < 0 :=
by
  sorry

end sufficient_conditions_for_quadratic_l223_22366


namespace slices_per_friend_l223_22363

theorem slices_per_friend (n : ℕ) (h1 : n > 0)
    (h2 : ∀ i : ℕ, i < n → (15 + 18 + 20 + 25) = 78 * n) :
    78 = (15 + 18 + 20 + 25) / n := 
by
  sorry

end slices_per_friend_l223_22363


namespace simplify_expr_l223_22318

theorem simplify_expr : (1 / (1 - Real.sqrt 3)) * (1 / (1 + Real.sqrt 3)) = -1 / 2 := by
  sorry

end simplify_expr_l223_22318


namespace walking_speed_l223_22309

theorem walking_speed 
  (D : ℝ) 
  (V_w : ℝ) 
  (h1 : D = V_w * 8) 
  (h2 : D = 36 * 2) : 
  V_w = 9 :=
by
  sorry

end walking_speed_l223_22309


namespace gcd_8164_2937_l223_22374

/-- Define the two integers a and b -/
def a : ℕ := 8164
def b : ℕ := 2937

/-- Prove that the greatest common divisor of a and b is 1 -/
theorem gcd_8164_2937 : Nat.gcd a b = 1 :=
  by
  sorry

end gcd_8164_2937_l223_22374


namespace stuart_segments_to_start_point_l223_22302

-- Definitions of given conditions
def concentric_circles {C : Type} (large small : Set C) (center : C) : Prop :=
  ∀ (x y : C), x ∈ large → y ∈ large → x ≠ y → (x = center ∨ y = center)

def tangent_to_small_circle {C : Type} (chord : Set C) (small : Set C) : Prop :=
  ∀ (x y : C), x ∈ chord → y ∈ chord → x ≠ y → (∀ z ∈ small, x ≠ z ∧ y ≠ z)

def measure_angle (ABC : Type) (θ : ℝ) : Prop :=
  θ = 60

-- The theorem to solve the problem
theorem stuart_segments_to_start_point 
    (C : Type)
    {large small : Set C} 
    {center : C} 
    {chords : List (Set C)}
    (h_concentric : concentric_circles large small center)
    (h_tangent : ∀ chord ∈ chords, tangent_to_small_circle chord small)
    (h_angle : ∀ ABC ∈ chords, measure_angle ABC 60)
    : ∃ n : ℕ, n = 3 := 
  sorry

end stuart_segments_to_start_point_l223_22302


namespace john_needs_60_bags_l223_22312

theorem john_needs_60_bags
  (horses : ℕ)
  (feeding_per_day : ℕ)
  (food_per_feeding : ℕ)
  (bag_weight : ℕ)
  (days : ℕ)
  (tons_in_pounds : ℕ)
  (half : ℕ)
  (h1 : horses = 25)
  (h2 : feeding_per_day = 2)
  (h3 : food_per_feeding = 20)
  (h4 : bag_weight = 1000)
  (h5 : days = 60)
  (h6 : tons_in_pounds = 2000)
  (h7 : half = 1 / 2) :
  ((horses * feeding_per_day * food_per_feeding * days) / (tons_in_pounds * half)) = 60 := by
  sorry

end john_needs_60_bags_l223_22312


namespace calc1_calc2_l223_22383

-- Problem 1
theorem calc1 : 2 * Real.sqrt 3 - 3 * Real.sqrt 12 + 5 * Real.sqrt 27 = 11 * Real.sqrt 3 := 
by sorry

-- Problem 2
theorem calc2 : (1 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 6) - (2 * Real.sqrt 3 - 1)^2 
              = -2 * Real.sqrt 2 + 4 * Real.sqrt 3 - 13 := 
by sorry

end calc1_calc2_l223_22383


namespace cafe_table_count_l223_22308

theorem cafe_table_count (cafe_seats_base7 : ℕ) (seats_per_table : ℕ) (cafe_seats_base10 : ℕ)
    (h1 : cafe_seats_base7 = 3 * 7^2 + 1 * 7^1 + 2 * 7^0) 
    (h2 : seats_per_table = 3) : cafe_seats_base10 = 156 ∧ (cafe_seats_base10 / seats_per_table) = 52 := 
by {
  sorry
}

end cafe_table_count_l223_22308


namespace arithmetic_sequence_seventy_fifth_term_l223_22388

theorem arithmetic_sequence_seventy_fifth_term:
  ∀ (a₁ a₂ d : ℕ), a₁ = 3 → a₂ = 51 → a₂ = a₁ + 24 * d → (3 + 74 * d) = 151 := by
  sorry

end arithmetic_sequence_seventy_fifth_term_l223_22388


namespace largest_study_only_Biology_l223_22392

-- Let's define the total number of students
def total_students : ℕ := 500

-- Define the given conditions
def S : ℕ := 65 * total_students / 100
def M : ℕ := 55 * total_students / 100
def B : ℕ := 50 * total_students / 100
def P : ℕ := 15 * total_students / 100

def MS : ℕ := 35 * total_students / 100
def MB : ℕ := 25 * total_students / 100
def BS : ℕ := 20 * total_students / 100
def MSB : ℕ := 10 * total_students / 100

-- Required to prove that the largest number of students who study only Biology is 75
theorem largest_study_only_Biology : 
  (B - MB - BS + MSB) = 75 :=
by 
  sorry

end largest_study_only_Biology_l223_22392


namespace sheila_weekly_earnings_l223_22340

-- Defining the conditions
def hourly_wage : ℕ := 12
def hours_mwf : ℕ := 8
def days_mwf : ℕ := 3
def hours_tt : ℕ := 6
def days_tt : ℕ := 2

-- Defining Sheila's total weekly earnings
noncomputable def weekly_earnings := (hours_mwf * hourly_wage * days_mwf) + (hours_tt * hourly_wage * days_tt)

-- The statement of the proof
theorem sheila_weekly_earnings : weekly_earnings = 432 :=
by
  sorry

end sheila_weekly_earnings_l223_22340


namespace geometric_sequence_general_term_l223_22375

theorem geometric_sequence_general_term (a : ℕ → ℕ) (q : ℕ) (h_q : q = 4) (h_sum : a 0 + a 1 + a 2 = 21)
  (h_geo : ∀ n, a (n + 1) = a n * q) : ∀ n, a n = 4 ^ n :=
by {
  sorry
}

end geometric_sequence_general_term_l223_22375


namespace find_b_15_l223_22301

variable {a : ℕ → ℤ} (b : ℕ → ℤ) (S : ℕ → ℤ)

/-- An arithmetic sequence where S_n is the sum of the first n terms, with S_9 = -18 and S_13 = -52
   and a geometric sequence where b_5 = a_5 and b_7 = a_7. -/
theorem find_b_15 
  (h1 : S 9 = -18) 
  (h2 : S 13 = -52) 
  (h3 : b 5 = a 5) 
  (h4 : b 7 = a 7) 
  : b 15 = -64 := 
sorry

end find_b_15_l223_22301


namespace problem_statement_l223_22382

noncomputable def distance_from_line_to_point (a b : ℝ) : ℝ :=
  abs (1 / 2) / (Real.sqrt (a ^ 2 + b ^ 2))

theorem problem_statement (a b : ℝ) (h1 : a = (1 - 2 * b) / 2) (h2 : b = 1 / 2 - a) :
  distance_from_line_to_point a b ≤ Real.sqrt 2 := 
sorry

end problem_statement_l223_22382


namespace walk_direction_east_l223_22307

theorem walk_direction_east (m : ℤ) (h : m = -2023) : m = -(-2023) :=
by
  sorry

end walk_direction_east_l223_22307


namespace BURN_maps_to_8615_l223_22379

open List Function

def tenLetterMapping : List (Char × Nat) := 
  [('G', 0), ('R', 1), ('E', 2), ('A', 3), ('T', 4), ('N', 5), ('U', 6), ('M', 7), ('B', 8), ('S', 9)]

def charToDigit (c : Char) : Option Nat :=
  tenLetterMapping.lookup c

def wordToNumber (word : List Char) : Option (List Nat) :=
  word.mapM charToDigit 

theorem BURN_maps_to_8615 :
  wordToNumber ['B', 'U', 'R', 'N'] = some [8, 6, 1, 5] :=
by
  sorry

end BURN_maps_to_8615_l223_22379


namespace inequality_proof_l223_22304

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c)) ≥ (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) :=
by
  sorry

end inequality_proof_l223_22304


namespace small_pizza_slices_l223_22355

-- Definitions based on conditions
def large_pizza_slices : ℕ := 16
def num_large_pizzas : ℕ := 2
def num_small_pizzas : ℕ := 2
def total_slices_eaten : ℕ := 48

-- Statement to prove
theorem small_pizza_slices (S : ℕ) (H : num_large_pizzas * large_pizza_slices + num_small_pizzas * S = total_slices_eaten) : S = 8 :=
by
  sorry

end small_pizza_slices_l223_22355


namespace exponents_multiplication_exponents_power_exponents_distributive_l223_22313

variables (x y m : ℝ)

theorem exponents_multiplication (x : ℝ) : (x^5) * (x^2) = x^7 :=
by sorry

theorem exponents_power (m : ℝ) : (m^2)^4 = m^8 :=
by sorry

theorem exponents_distributive (x y : ℝ) : (-2 * x * y^2)^3 = -8 * x^3 * y^6 :=
by sorry

end exponents_multiplication_exponents_power_exponents_distributive_l223_22313


namespace right_triangle_hypotenuse_l223_22359

theorem right_triangle_hypotenuse (a b : ℕ) (a_val : a = 4) (b_val : b = 5) :
    ∃ c : ℝ, c^2 = (a:ℝ)^2 + (b:ℝ)^2 ∧ c = Real.sqrt 41 :=
by
  sorry

end right_triangle_hypotenuse_l223_22359


namespace find_x_l223_22389

theorem find_x :
  let a := 0.15
  let b := 0.06
  let c := 0.003375
  let d := 0.000216
  let e := 0.0225
  let f := 0.0036
  let g := 0.08999999999999998
  ∃ x, c - (d / e) + x + f = g →
  x = 0.092625 :=
by
  sorry

end find_x_l223_22389


namespace original_price_of_sarees_l223_22362

theorem original_price_of_sarees (P : ℝ) (h : 0.75 * 0.85 * P = 306) : P = 480 :=
by
  sorry

end original_price_of_sarees_l223_22362


namespace average_tickets_sold_by_male_members_l223_22349

theorem average_tickets_sold_by_male_members 
  (M F : ℕ)
  (total_average : ℕ)
  (female_average : ℕ)
  (ratio : ℕ × ℕ)
  (h1 : total_average = 66)
  (h2 : female_average = 70)
  (h3 : ratio = (1, 2))
  (h4 : F = 2 * M)
  (h5 : (M + F) * total_average = M * r + F * female_average) :
  r = 58 :=
sorry

end average_tickets_sold_by_male_members_l223_22349


namespace shopkeeper_profit_percent_l223_22327

noncomputable def profit_percent : ℚ := 
let cp_each := 1       -- Cost price of each article
let sp_each := 1.2     -- Selling price of each article without discount
let discount := 0.05   -- 5% discount
let tax := 0.10        -- 10% sales tax
let articles := 30     -- Number of articles
let cp_total := articles * cp_each      -- Total cost price
let sp_after_discount := sp_each * (1 - discount)    -- Selling price after discount
let revenue_before_tax := articles * sp_after_discount   -- Total revenue before tax
let tax_amount := revenue_before_tax * tax   -- Sales tax amount
let revenue_after_tax := revenue_before_tax + tax_amount -- Total revenue after tax
let profit := revenue_after_tax - cp_total -- Profit
(profit / cp_total) * 100 -- Profit percent

theorem shopkeeper_profit_percent : profit_percent = 25.4 :=
by
  -- Here follows the proof based on the conditions and steps above
  sorry

end shopkeeper_profit_percent_l223_22327


namespace same_number_of_friends_l223_22338

-- Definitions and conditions
def num_people (n : ℕ) := true   -- Placeholder definition to indicate the number of people
def num_friends (person : ℕ) (n : ℕ) : ℕ := sorry -- The number of friends a given person has (needs to be defined)
def friends_range (n : ℕ) := ∀ person, 0 ≤ num_friends person n ∧ num_friends person n < n

-- Theorem statement
theorem same_number_of_friends (n : ℕ) (h1 : num_people n) (h2 : friends_range n) : 
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ num_friends p1 n = num_friends p2 n :=
by
  sorry

end same_number_of_friends_l223_22338
