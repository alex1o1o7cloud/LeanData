import Mathlib

namespace max_area_rectangle_max_area_achievable_l697_69732

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

/-- The theorem stating the maximum area of a rectangle with given conditions -/
theorem max_area_rectangle (l w : ℕ) : 
  (l + w = 60) →  -- Perimeter condition: 2(l + w) = 120
  (isPrime l ∨ isPrime w) →  -- One dimension is prime
  (l * w ≤ 899) :=  -- The area is at most 899
by sorry

/-- The theorem stating that the maximum area of 899 is achievable -/
theorem max_area_achievable : 
  ∃ l w : ℕ, (l + w = 60) ∧ (isPrime l ∨ isPrime w) ∧ (l * w = 899) :=
by sorry

end max_area_rectangle_max_area_achievable_l697_69732


namespace greatest_two_digit_multiple_of_seven_l697_69774

theorem greatest_two_digit_multiple_of_seven : ∃ n : ℕ, n = 98 ∧ 
  (∀ m : ℕ, m < 100 ∧ 7 ∣ m → m ≤ n) := by
  sorry

end greatest_two_digit_multiple_of_seven_l697_69774


namespace g_is_correct_l697_69746

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := -2*x^5 + 7*x^4 + 4*x^3 - 2*x^2 - 8*x + 4

-- Theorem statement
theorem g_is_correct :
  ∀ x : ℝ, 2*x^5 - 4*x^3 + 3*x + g x = 7*x^4 - 2*x^2 - 5*x + 4 :=
by
  sorry

end g_is_correct_l697_69746


namespace ball_count_l697_69750

theorem ball_count (white green yellow red purple : ℕ) 
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 5)
  (h4 : red = 6)
  (h5 : purple = 9)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 3/4) :
  white + green + yellow + red + purple = 60 := by sorry

end ball_count_l697_69750


namespace existence_of_point_N_l697_69714

theorem existence_of_point_N (a m : ℝ) (ha : a > 0) (hm : m ∈ Set.union (Set.Ioo (-1) 0) (Set.Ioi 0)) :
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = a^2 ∧ |y₀| = (|m| * a) / Real.sqrt (1 + m) := by
  sorry

end existence_of_point_N_l697_69714


namespace dot_product_equals_negative_49_l697_69781

def vector1 : Fin 4 → ℝ := ![4, -5, 2, -1]
def vector2 : Fin 4 → ℝ := ![-6, 3, -4, 2]

theorem dot_product_equals_negative_49 :
  (Finset.sum Finset.univ (λ i => vector1 i * vector2 i)) = -49 := by
  sorry

end dot_product_equals_negative_49_l697_69781


namespace equation_solution_l697_69734

theorem equation_solution (k : ℤ) : 
  (∃ x : ℤ, x > 0 ∧ 9 * x - 3 = k * x + 14) ↔ (k = 8 ∨ k = -8) :=
by sorry

end equation_solution_l697_69734


namespace joan_balloons_l697_69763

def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2

theorem joan_balloons : initial_balloons - lost_balloons = 7 := by
  sorry

end joan_balloons_l697_69763


namespace larger_number_proof_l697_69718

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1335)
  (h2 : L = 6 * S + 15) :
  L = 1599 := by
  sorry

end larger_number_proof_l697_69718


namespace range_of_m_l697_69764

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 + 1) * (x^2 - 8*x - 20) ≤ 0 → -2 ≤ x ∧ x ≤ 10) ∧
  (∀ x : ℝ, x^2 - 2*x + (1 - m^2) ≤ 0 → 1 - m ≤ x ∧ x ≤ 1 + m) ∧
  (m > 0) ∧
  (∀ x : ℝ, (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 10 ∧ (x < 1 - m ∨ x > 1 + m)) →
  m ≥ 9 :=
by sorry

end range_of_m_l697_69764


namespace triangle_side_ratio_l697_69754

/-- In a triangle ABC, if angle A is 2π/3 and side a is √3 times side c, then the ratio of side a to side b is √3. -/
theorem triangle_side_ratio (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  A + B + C = π →  -- angle sum property
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- valid angle measures
  A = 2 * π / 3 →  -- given angle A
  a = Real.sqrt 3 * c →  -- given side relation
  a / b = Real.sqrt 3 := by
sorry

end triangle_side_ratio_l697_69754


namespace connie_total_markers_l697_69789

/-- The number of red markers Connie has -/
def red_markers : ℕ := 5230

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 4052

/-- The number of green markers Connie has -/
def green_markers : ℕ := 3180

/-- The number of purple markers Connie has -/
def purple_markers : ℕ := 2763

/-- The total number of markers Connie has -/
def total_markers : ℕ := red_markers + blue_markers + green_markers + purple_markers

theorem connie_total_markers : total_markers = 15225 := by
  sorry

end connie_total_markers_l697_69789


namespace complement_of_M_in_U_l697_69771

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def M : Finset Nat := {1, 2, 4}

theorem complement_of_M_in_U :
  (U \ M) = {3, 5, 6} := by sorry

end complement_of_M_in_U_l697_69771


namespace abc_zero_l697_69755

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) : 
  a * b * c = 0 := by
sorry

end abc_zero_l697_69755


namespace find_n_l697_69761

theorem find_n (x y : ℝ) (h1 : x = 3) (h2 : y = 2) : x - y^(x-y) * (x+y) = -7 := by
  sorry

end find_n_l697_69761


namespace intersection_of_A_and_B_l697_69711

-- Define sets A and B
def A : Set ℝ := {x | x^2 + x - 12 < 0}
def B : Set ℝ := {x | 2 - x < 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l697_69711


namespace logarithmic_equality_implies_zero_product_l697_69794

theorem logarithmic_equality_implies_zero_product (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (a - b) * Real.log c + (b - c) * Real.log a + (c - a) * Real.log b = 0) :
  (a - b) * (b - c) * (c - a) = 0 := by
  sorry

end logarithmic_equality_implies_zero_product_l697_69794


namespace max_value_sqrt_sum_l697_69716

theorem max_value_sqrt_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (a_ge : a ≥ -1/2)
  (b_ge : b ≥ -2)
  (c_ge : c ≥ -7/3) :
  ∃ (x y z : ℝ), x + y + z = 3 ∧ 
    x ≥ -1/2 ∧ y ≥ -2 ∧ z ≥ -7/3 ∧
    Real.sqrt (4*x + 2) + Real.sqrt (4*y + 8) + Real.sqrt (4*z + 10) = 4 * Real.sqrt 6 ∧
    ∀ (a b c : ℝ), a + b + c = 3 → a ≥ -1/2 → b ≥ -2 → c ≥ -7/3 →
      Real.sqrt (4*a + 2) + Real.sqrt (4*b + 8) + Real.sqrt (4*c + 10) ≤ 4 * Real.sqrt 6 :=
by
  sorry

end max_value_sqrt_sum_l697_69716


namespace absolute_fraction_inequality_l697_69720

theorem absolute_fraction_inequality (x : ℝ) : 
  |((3 * x - 2) / (x + 1))| > 3 ↔ x < -1 ∨ (-1 < x ∧ x < 1/6) :=
by sorry

end absolute_fraction_inequality_l697_69720


namespace triangle_side_length_l697_69765

theorem triangle_side_length 
  (A B C : ℝ) 
  (hBC : Real.cos C = -Real.sqrt 2 / 2) 
  (hAC : Real.sin A / Real.sin B = 1 / (2 * Real.cos (A + B))) 
  (hBA : B * A = 2 * Real.sqrt 2) : 
  Real.sqrt ((Real.sin A)^2 + (Real.sin B)^2 - 2 * Real.sin A * Real.sin B * Real.cos C) = Real.sqrt 10 := by
sorry

end triangle_side_length_l697_69765


namespace probability_red_before_green_l697_69772

def num_red : ℕ := 4
def num_green : ℕ := 3
def num_blue : ℕ := 1

def total_chips : ℕ := num_red + num_green + num_blue

theorem probability_red_before_green :
  let favorable_arrangements := (total_chips - 1).choose num_green
  let total_arrangements := total_chips.choose num_green * total_chips.choose num_blue
  (favorable_arrangements * total_chips) / total_arrangements = 3 / 5 := by
sorry

end probability_red_before_green_l697_69772


namespace house_size_multiple_l697_69701

theorem house_size_multiple (sara_house : ℝ) (nada_house : ℝ) (extra_size : ℝ) :
  sara_house = 1000 →
  nada_house = 450 →
  sara_house = nada_house * (sara_house - extra_size) / nada_house + extra_size →
  extra_size = 100 →
  (sara_house - extra_size) / nada_house = 2 :=
by
  sorry

end house_size_multiple_l697_69701


namespace circle_C_equation_l697_69739

-- Define the circles and points
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def point_A : ℝ × ℝ := (1, 0)

-- Define the properties of circle C
structure Circle_C where
  center : ℝ × ℝ
  tangent_to_x_axis : center.2 > 0
  tangent_at_A : (center.1 - point_A.1)^2 + (center.2 - point_A.2)^2 = center.2^2
  intersects_O : ∃ P Q : ℝ × ℝ, P ∈ circle_O ∧ Q ∈ circle_O ∧
    (P.1 - center.1)^2 + (P.2 - center.2)^2 = center.2^2 ∧
    (Q.1 - center.1)^2 + (Q.2 - center.2)^2 = center.2^2
  PQ_length : ∃ P Q : ℝ × ℝ, P ∈ circle_O ∧ Q ∈ circle_O ∧
    (P.1 - center.1)^2 + (P.2 - center.2)^2 = center.2^2 ∧
    (Q.1 - center.1)^2 + (Q.2 - center.2)^2 = center.2^2 ∧
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 14/4

-- Theorem stating the standard equation of circle C
theorem circle_C_equation (c : Circle_C) :
  ∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.center.2^2 ↔
  (x - 1)^2 + (y - 1)^2 = 1 :=
sorry

end circle_C_equation_l697_69739


namespace cubic_root_sum_l697_69780

theorem cubic_root_sum (α β γ : ℂ) : 
  α^3 - α - 1 = 0 → β^3 - β - 1 = 0 → γ^3 - γ - 1 = 0 →
  (1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = -7 := by
sorry

end cubic_root_sum_l697_69780


namespace mixed_number_multiplication_problem_solution_l697_69762

theorem mixed_number_multiplication (a b c d : ℚ) :
  (a + b / c) * (1 / d) = (a * c + b) / (c * d) :=
by sorry

theorem problem_solution : 2 + 4/5 * (1/5) = 14/25 :=
by sorry

end mixed_number_multiplication_problem_solution_l697_69762


namespace log_equation_solution_l697_69787

theorem log_equation_solution (a : ℝ) (h : a > 0) :
  Real.log a / Real.log 2 - 2 * Real.log 2 / Real.log a = 1 →
  a = 4 ∨ a = 1/2 := by
sorry

end log_equation_solution_l697_69787


namespace roller_coaster_problem_l697_69724

def roller_coaster_rides (people_in_line : ℕ) (cars : ℕ) (people_per_car : ℕ) : ℕ :=
  (people_in_line + cars * people_per_car - 1) / (cars * people_per_car)

theorem roller_coaster_problem :
  roller_coaster_rides 84 7 2 = 6 := by
  sorry

end roller_coaster_problem_l697_69724


namespace alice_bob_calculation_l697_69785

theorem alice_bob_calculation (x : ℕ) : 
  let alice_result := ((x + 2) * 2 + 3)
  2 * (alice_result + 3) = 4 * x + 16 := by
  sorry

end alice_bob_calculation_l697_69785


namespace complex_modulus_one_l697_69744

theorem complex_modulus_one (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) :
  Complex.abs z = 1 := by
  sorry

end complex_modulus_one_l697_69744


namespace weight_of_B_l697_69710

theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 := by
sorry

end weight_of_B_l697_69710


namespace probability_specific_case_l697_69730

/-- The probability of drawing a white marble first and a red marble second -/
def probability_white_then_red (total_marbles : ℕ) (red_marbles : ℕ) (white_marbles : ℕ) : ℚ :=
  (white_marbles : ℚ) / (total_marbles : ℚ) * (red_marbles : ℚ) / ((total_marbles - 1) : ℚ)

theorem probability_specific_case :
  probability_white_then_red 10 4 6 = 4 / 15 := by
  sorry

#eval probability_white_then_red 10 4 6

end probability_specific_case_l697_69730


namespace pyramid_slice_height_l697_69791

-- Define the pyramid P
structure Pyramid :=
  (base_length : ℝ)
  (base_width : ℝ)
  (height : ℝ)

-- Define the main theorem
theorem pyramid_slice_height (P : Pyramid) (volume_ratio : ℝ) :
  P.base_length = 15 →
  P.base_width = 20 →
  P.height = 30 →
  volume_ratio = 9 →
  (P.height - (P.height / (volume_ratio ^ (1/3 : ℝ)))) = 20 := by
  sorry


end pyramid_slice_height_l697_69791


namespace equation_solution_l697_69722

theorem equation_solution : ∃ x : ℝ, 
  (x^2 - 7*x + 12) / (x^2 - 9*x + 20) = (x^2 - 4*x - 21) / (x^2 - 5*x - 24) ∧ x = 11 := by
  sorry

end equation_solution_l697_69722


namespace abs_eq_sqrt_sq_l697_69737

theorem abs_eq_sqrt_sq (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end abs_eq_sqrt_sq_l697_69737


namespace triangle_altitude_angle_relation_l697_69728

theorem triangle_altitude_angle_relation (A B C : Real) (C₁ C₂ : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = 180 →
  -- A is 60° and greater than B
  A = 60 ∧ A > B →
  -- C₁ and C₂ are parts of angle C divided by the altitude
  C = C₁ + C₂ →
  -- C₁ is adjacent to side b (opposite to angle B)
  C₁ > 0 ∧ C₂ > 0 →
  -- The altitude creates right angles
  B + C₁ = 90 ∧ A + C₂ = 90 →
  -- Conclusion
  C₁ - C₂ = 0 := by
sorry

end triangle_altitude_angle_relation_l697_69728


namespace dropped_student_score_l697_69788

theorem dropped_student_score
  (initial_students : ℕ)
  (initial_average : ℚ)
  (remaining_students : ℕ)
  (new_average : ℚ)
  (h1 : initial_students = 16)
  (h2 : initial_average = 62.5)
  (h3 : remaining_students = 15)
  (h4 : new_average = 63)
  (h5 : remaining_students = initial_students - 1) :
  (initial_students : ℚ) * initial_average - (remaining_students : ℚ) * new_average = 55 :=
by
  sorry

end dropped_student_score_l697_69788


namespace simplify_product_of_radicals_l697_69799

theorem simplify_product_of_radicals (x : ℝ) (hx : x > 0) :
  Real.sqrt (48 * x) * Real.sqrt (27 * x) * Real.sqrt (32 * x) = 144 * x * Real.sqrt (2 * x) := by
  sorry

end simplify_product_of_radicals_l697_69799


namespace mr_slinkums_order_l697_69747

theorem mr_slinkums_order (on_shelves_percent : ℚ) (in_storage : ℕ) : 
  on_shelves_percent = 1/5 ∧ in_storage = 120 → 
  (1 - on_shelves_percent) * 150 = in_storage :=
by
  sorry

end mr_slinkums_order_l697_69747


namespace min_value_of_function_l697_69708

theorem min_value_of_function (x y : ℝ) : x^2 + y^2 - 8*x + 6*y + 26 ≥ 1 := by
  sorry

end min_value_of_function_l697_69708


namespace f_derivative_at_2_l697_69726

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- State the theorem
theorem f_derivative_at_2 : 
  (deriv f) 2 = 6 := by sorry

end f_derivative_at_2_l697_69726


namespace arevalo_dinner_bill_l697_69738

/-- The Arevalo family's dinner bill problem -/
theorem arevalo_dinner_bill (salmon_price black_burger_price chicken_katsu_price : ℝ)
  (service_charge_rate : ℝ) (paid_amount change_received : ℝ) :
  salmon_price = 40 ∧
  black_burger_price = 15 ∧
  chicken_katsu_price = 25 ∧
  service_charge_rate = 0.1 ∧
  paid_amount = 100 ∧
  change_received = 8 →
  let total_food_cost := salmon_price + black_burger_price + chicken_katsu_price
  let service_charge := service_charge_rate * total_food_cost
  let subtotal := total_food_cost + service_charge
  let amount_paid := paid_amount - change_received
  let tip := amount_paid - subtotal
  tip / total_food_cost = 0.05 := by
  sorry

end arevalo_dinner_bill_l697_69738


namespace logical_equivalences_l697_69721

theorem logical_equivalences (p q : Prop) : 
  ((p ∧ q) ↔ ¬(¬p ∨ ¬q)) ∧
  ((p ∨ q) ↔ ¬(¬p ∧ ¬q)) ∧
  ((p → q) ↔ (¬q → ¬p)) ∧
  ((p ↔ q) ↔ ((p → q) ∧ (q → p))) :=
by sorry

end logical_equivalences_l697_69721


namespace leak_drain_time_l697_69715

/-- Given a pump that can fill a tank in 2 hours, and with a leak it takes 2 1/3 hours to fill the tank,
    prove that the time it takes for the leak to drain all the water of the tank is 14 hours. -/
theorem leak_drain_time (pump_fill_time leak_fill_time : ℚ) : 
  pump_fill_time = 2 →
  leak_fill_time = 7/3 →
  (1 / (1 / pump_fill_time - 1 / leak_fill_time)) = 14 := by
  sorry

end leak_drain_time_l697_69715


namespace cricket_team_average_age_l697_69706

theorem cricket_team_average_age : 
  ∀ (team_size : ℕ) (captain_age wicket_keeper_age : ℕ) (team_average : ℚ),
    team_size = 11 →
    captain_age = 27 →
    wicket_keeper_age = 28 →
    (team_size : ℚ) * team_average = 
      (captain_age : ℚ) + (wicket_keeper_age : ℚ) + 
      ((team_size - 2) : ℚ) * (team_average - 1) →
    team_average = 23 := by
  sorry

end cricket_team_average_age_l697_69706


namespace absolute_value_ratio_l697_69784

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 12*a*b) :
  |((a + b) / (a - b))| = Real.sqrt (7/5) := by
  sorry

end absolute_value_ratio_l697_69784


namespace eventual_shot_probability_l697_69777

def basketball_game (make_probability : ℝ) (get_ball_back_probability : ℝ) : Prop :=
  (0 ≤ make_probability ∧ make_probability ≤ 1) ∧
  (0 ≤ get_ball_back_probability ∧ get_ball_back_probability ≤ 1)

theorem eventual_shot_probability
  (make_prob : ℝ)
  (get_ball_back_prob : ℝ)
  (h_game : basketball_game make_prob get_ball_back_prob)
  (h_make_prob : make_prob = 1/10)
  (h_get_ball_back_prob : get_ball_back_prob = 9/10) :
  (1 - (1 - make_prob) * get_ball_back_prob / (1 - (1 - make_prob) * (1 - get_ball_back_prob))) = 10/19 :=
by sorry


end eventual_shot_probability_l697_69777


namespace third_episode_duration_l697_69704

/-- Given a series of four episodes with known durations for three episodes
    and a total duration, this theorem proves the duration of the third episode. -/
theorem third_episode_duration
  (total_duration : ℕ)
  (first_episode : ℕ)
  (second_episode : ℕ)
  (fourth_episode : ℕ)
  (h1 : total_duration = 240)  -- 4 hours in minutes
  (h2 : first_episode = 58)
  (h3 : second_episode = 62)
  (h4 : fourth_episode = 55)
  : total_duration - (first_episode + second_episode + fourth_episode) = 65 := by
  sorry

#check third_episode_duration

end third_episode_duration_l697_69704


namespace sqrt_two_thirds_irrational_l697_69760

theorem sqrt_two_thirds_irrational (h : Irrational (Real.sqrt 6)) : Irrational (Real.sqrt (2/3)) := by
  sorry

end sqrt_two_thirds_irrational_l697_69760


namespace problem_solution_l697_69735

theorem problem_solution (x : ℝ) (h : x = 13 / Real.sqrt (19 + 8 * Real.sqrt 3)) :
  (x^4 - 6*x^3 - 2*x^2 + 18*x + 23) / (x^2 - 8*x + 15) = 5 := by
  sorry

end problem_solution_l697_69735


namespace number_percentage_l697_69796

theorem number_percentage (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → (40/100 : ℝ) * N = 192 := by
  sorry

end number_percentage_l697_69796


namespace combined_distance_theorem_l697_69798

/-- Represents the four lakes in the migration sequence -/
inductive Lake : Type
| Jim : Lake
| Disney : Lake
| London : Lake
| Everest : Lake

/-- The number of birds in the group -/
def num_birds : ℕ := 25

/-- The number of migration sequences completed in a year -/
def sequences_per_year : ℕ := 2

/-- The distance between two lakes in miles -/
def distance (a b : Lake) : ℕ :=
  match a, b with
  | Lake.Jim, Lake.Disney => 42
  | Lake.Disney, Lake.London => 57
  | Lake.London, Lake.Everest => 65
  | Lake.Everest, Lake.Jim => 70
  | _, _ => 0  -- For other combinations, return 0

/-- The total distance of one migration sequence -/
def sequence_distance : ℕ :=
  distance Lake.Jim Lake.Disney +
  distance Lake.Disney Lake.London +
  distance Lake.London Lake.Everest +
  distance Lake.Everest Lake.Jim

/-- Theorem: The combined distance traveled by all birds in a year is 11,700 miles -/
theorem combined_distance_theorem :
  num_birds * sequences_per_year * sequence_distance = 11700 := by
  sorry

end combined_distance_theorem_l697_69798


namespace alpha_set_property_l697_69736

theorem alpha_set_property (r s : ℕ) (hr : r > s) (hgcd : Nat.gcd r s = 1) :
  let α : ℚ := r / s
  let N_α : Set ℕ := {m | ∃ n : ℕ, m = ⌊n * α⌋}
  ∀ m ∈ N_α, ¬(r ∣ (m + 1)) := by
  sorry

end alpha_set_property_l697_69736


namespace percentage_decrease_of_b_l697_69700

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- a and b are positive
  a / b = 4 / 5 ∧  -- ratio of a to b is 4 to 5
  x = a * 1.25 ∧  -- x equals a increased by 25 percent
  m = b * (1 - p / 100) ∧  -- m equals b decreased by p percent
  m / x = 0.8  -- ratio of m to x is 0.8
  → p = 20 := by  -- prove that p (percentage decrease) is 20
sorry

end percentage_decrease_of_b_l697_69700


namespace train_length_l697_69723

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed : Real) (time : Real) :
  speed = 108 →
  time = 1.4998800095992322 →
  ∃ (length : Real), abs (length - (speed * 1000 / 3600 * time)) < 0.001 ∧ abs (length - 44.996) < 0.001 := by
  sorry

end train_length_l697_69723


namespace evie_shells_left_l697_69712

/-- The number of shells Evie collects per day -/
def shells_per_day : ℕ := 10

/-- The number of days Evie collects shells -/
def collection_days : ℕ := 6

/-- The number of shells Evie gives to her brother -/
def shells_given : ℕ := 2

/-- The number of shells Evie has left after collecting and giving some away -/
def shells_left : ℕ := shells_per_day * collection_days - shells_given

/-- Theorem stating that Evie has 58 shells left -/
theorem evie_shells_left : shells_left = 58 := by sorry

end evie_shells_left_l697_69712


namespace batsman_average_l697_69797

theorem batsman_average (total_matches : ℕ) (first_set_matches : ℕ) (first_set_average : ℝ) (total_average : ℝ) :
  total_matches = 30 →
  first_set_matches = 20 →
  first_set_average = 30 →
  total_average = 25 →
  let second_set_matches := total_matches - first_set_matches
  let second_set_average := (total_average * total_matches - first_set_average * first_set_matches) / second_set_matches
  second_set_average = 15 := by sorry

end batsman_average_l697_69797


namespace probability_three_unused_theorem_expected_hits_nine_targets_theorem_l697_69768

/-- Represents a rocket artillery system on a missile cruiser -/
structure RocketSystem where
  total_rockets : ℕ
  hit_probability : ℝ

/-- Calculates the probability of exactly three unused rockets remaining after firing at five targets -/
def probability_three_unused (system : RocketSystem) : ℝ :=
  10 * system.hit_probability^3 * (1 - system.hit_probability)^2

/-- Calculates the expected number of targets hit when firing at nine targets -/
def expected_hits_nine_targets (system : RocketSystem) : ℝ :=
  10 * system.hit_probability - system.hit_probability^10

/-- Theorem stating the probability of exactly three unused rockets remaining after firing at five targets -/
theorem probability_three_unused_theorem (system : RocketSystem) :
  probability_three_unused system = 10 * system.hit_probability^3 * (1 - system.hit_probability)^2 := by
  sorry

/-- Theorem stating the expected number of targets hit when firing at nine targets -/
theorem expected_hits_nine_targets_theorem (system : RocketSystem) :
  expected_hits_nine_targets system = 10 * system.hit_probability - system.hit_probability^10 := by
  sorry

end probability_three_unused_theorem_expected_hits_nine_targets_theorem_l697_69768


namespace number_of_possible_values_l697_69709

theorem number_of_possible_values (m n k a b : ℕ+) :
  ((1 + a.val : ℕ) * n.val^2 - 4 * (m.val + a.val) * n.val + 4 * m.val^2 + 4 * a.val + b.val * (k.val - 1)^2 < 3) →
  (∃ (s : Finset ℕ), s = {x | ∃ (m' n' k' : ℕ+), 
    ((1 + a.val : ℕ) * n'.val^2 - 4 * (m'.val + a.val) * n'.val + 4 * m'.val^2 + 4 * a.val + b.val * (k'.val - 1)^2 < 3) ∧
    x = m'.val + n'.val + k'.val} ∧ 
  s.card = 4) :=
sorry

end number_of_possible_values_l697_69709


namespace infinite_squares_sum_cube_l697_69727

theorem infinite_squares_sum_cube :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, ∃ m a : ℕ,
    m > f n ∧ m > 1 ∧ 3 * (2 * a + m + 1)^2 = 11 * m^2 + 1 :=
sorry

end infinite_squares_sum_cube_l697_69727


namespace amount_after_two_years_l697_69795

/-- The amount after n years given an initial amount and yearly increase rate -/
def amount_after_years (initial_amount : ℝ) (increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + increase_rate) ^ years

/-- Theorem stating the amount after two years -/
theorem amount_after_two_years :
  let initial_amount : ℝ := 62000
  let increase_rate : ℝ := 1/8
  let years : ℕ := 2
  amount_after_years initial_amount increase_rate years = 78468.75 := by
sorry

end amount_after_two_years_l697_69795


namespace power_multiplication_l697_69770

theorem power_multiplication (x : ℝ) (h : x = 5) : x^3 * x^4 = 78125 := by
  sorry

end power_multiplication_l697_69770


namespace inscribed_square_area_l697_69792

/-- The area of a square inscribed in an isosceles right triangle -/
theorem inscribed_square_area (leg_length : ℝ) (h : leg_length = 28 * Real.sqrt 2) :
  let diagonal := leg_length
  let side := diagonal / Real.sqrt 2
  side ^ 2 = 784 := by sorry

end inscribed_square_area_l697_69792


namespace square_area_from_rectangle_circle_l697_69758

theorem square_area_from_rectangle_circle (rectangle_length : ℝ) (circle_radius : ℝ) (square_side : ℝ) : 
  rectangle_length = (2 / 5) * circle_radius →
  circle_radius = square_side →
  rectangle_length * 10 = 180 →
  square_side ^ 2 = 2025 := by
  sorry

end square_area_from_rectangle_circle_l697_69758


namespace projection_a_onto_b_l697_69766

def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (-2, 1)

theorem projection_a_onto_b :
  let proj_magnitude := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)
  proj_magnitude = 2 * Real.sqrt 5 := by sorry

end projection_a_onto_b_l697_69766


namespace hyperbola_parabola_intersection_l697_69767

/-- Given a hyperbola and a parabola with specific properties, prove that p = 1 -/
theorem hyperbola_parabola_intersection (a b p : ℝ) : 
  a > 0 → b > 0 → p > 0 →
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y, y^2 = 2 * p * x) →
  (a^2 + b^2) / a^2 = 4 →
  1/2 * (p/2) * (b*p/a) = Real.sqrt 3 / 4 →
  p = 1 := by
sorry

end hyperbola_parabola_intersection_l697_69767


namespace smallest_n_satisfying_conditions_l697_69757

theorem smallest_n_satisfying_conditions : 
  ∃ n : ℕ, 
    n > 2021 ∧ 
    Nat.gcd 63 (n + 120) = 21 ∧ 
    Nat.gcd (n + 63) 120 = 60 ∧
    (∀ m : ℕ, m > 2021 → Nat.gcd 63 (m + 120) = 21 → Nat.gcd (m + 63) 120 = 60 → m ≥ n) ∧
    n = 2337 :=
by sorry

end smallest_n_satisfying_conditions_l697_69757


namespace trigonometric_simplification_l697_69741

theorem trigonometric_simplification :
  (Real.tan (12 * π / 180) - Real.sqrt 3) / (Real.sin (12 * π / 180) * Real.cos (24 * π / 180)) = -8 := by
  sorry

end trigonometric_simplification_l697_69741


namespace santa_candy_problem_l697_69773

theorem santa_candy_problem (total : ℕ) (chocolate : ℕ) (gummy : ℕ) :
  total = 2023 →
  chocolate + gummy = total →
  chocolate = (75 * gummy) / 100 →
  chocolate = 867 := by
sorry

end santa_candy_problem_l697_69773


namespace geometric_sequence_common_ratio_l697_69729

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_geometric : is_geometric_sequence a q)
  (h_arithmetic : is_arithmetic_sequence (λ n => match n with
    | 0 => a 3
    | 1 => 3 * a 2
    | 2 => 5 * a 1
    | _ => 0))
  (h_increasing : ∀ n : ℕ, a n < a (n + 1)) :
  q = 5 := by sorry

end geometric_sequence_common_ratio_l697_69729


namespace common_internal_tangent_length_l697_69769

theorem common_internal_tangent_length 
  (center_distance : ℝ) 
  (radius1 : ℝ) 
  (radius2 : ℝ) 
  (h1 : center_distance = 50)
  (h2 : radius1 = 7)
  (h3 : radius2 = 10) : 
  Real.sqrt (center_distance^2 - (radius1 + radius2)^2) = Real.sqrt 2211 :=
sorry

end common_internal_tangent_length_l697_69769


namespace intersection_and_union_when_a_is_neg_one_intersection_equals_B_iff_l697_69790

-- Define sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 5}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 2}

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_neg_one :
  (A ∩ B (-1) = {x | -2 ≤ x ∧ x ≤ -1}) ∧
  (A ∪ B (-1) = {x | x ≤ 1 ∨ x ≥ 5}) := by sorry

-- Theorem for part (2)
theorem intersection_equals_B_iff :
  ∀ a : ℝ, A ∩ B a = B a ↔ a ≤ -3 ∨ a > 2 := by sorry

end intersection_and_union_when_a_is_neg_one_intersection_equals_B_iff_l697_69790


namespace sheet_width_sheet_width_proof_l697_69705

/-- The width of a rectangular metallic sheet, given specific conditions --/
theorem sheet_width : ℝ :=
  let length : ℝ := 100
  let cut_size : ℝ := 10
  let box_volume : ℝ := 24000
  let width : ℝ := 50

  have h1 : box_volume = (length - 2 * cut_size) * (width - 2 * cut_size) * cut_size :=
    by sorry

  50

theorem sheet_width_proof (length : ℝ) (cut_size : ℝ) (box_volume : ℝ) :
  length = 100 →
  cut_size = 10 →
  box_volume = 24000 →
  box_volume = (length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size →
  sheet_width = 50 :=
by sorry

end sheet_width_sheet_width_proof_l697_69705


namespace mrs_sheridan_fish_count_l697_69753

theorem mrs_sheridan_fish_count (initial_fish : Nat) (fish_from_sister : Nat) : 
  initial_fish = 22 → fish_from_sister = 47 → initial_fish + fish_from_sister = 69 := by
  sorry

end mrs_sheridan_fish_count_l697_69753


namespace rectangular_field_length_l697_69778

theorem rectangular_field_length (width : ℝ) (pond_side : ℝ) : 
  pond_side = 8 →
  (pond_side ^ 2) = (1 / 2) * (2 * width * width) →
  2 * width = 16 :=
by
  sorry

end rectangular_field_length_l697_69778


namespace full_price_store_a_is_125_l697_69749

/-- The full price of a smartphone at Store A, given discount information for two stores. -/
def full_price_store_a : ℝ :=
  let discount_a : ℝ := 0.08
  let price_b : ℝ := 130
  let discount_b : ℝ := 0.10
  let price_difference : ℝ := 2

  -- Define the equation based on the given conditions
  let equation : ℝ → Prop := fun p =>
    p * (1 - discount_a) = price_b * (1 - discount_b) - price_difference

  -- Assert that 125 satisfies the equation
  125

theorem full_price_store_a_is_125 :
  full_price_store_a = 125 := by sorry

end full_price_store_a_is_125_l697_69749


namespace birthday_celebration_men_count_l697_69733

/-- Proves that the number of men at a birthday celebration was 15 given the specified conditions. -/
theorem birthday_celebration_men_count :
  ∀ (total_guests women men children : ℕ),
    total_guests = 60 →
    women = total_guests / 2 →
    total_guests = women + men + children →
    50 = women + (men - men / 3) + (children - 5) →
    men = 15 :=
by
  sorry

end birthday_celebration_men_count_l697_69733


namespace die_roll_outcomes_l697_69731

/-- The number of faces on a standard die -/
def numDieFaces : ℕ := 6

/-- The number of rolls before stopping -/
def numRolls : ℕ := 5

/-- The number of different outcomes when rolling a die continuously and stopping
    after exactly 5 rolls, with the condition that three different numbers appear
    on the fifth roll -/
def numOutcomes : ℕ := 840

/-- Theorem stating that the number of different outcomes is 840 -/
theorem die_roll_outcomes :
  (numDieFaces.choose 2) * ((numDieFaces - 2).choose 1) * (4 + 6 + 4) = numOutcomes := by
  sorry

end die_roll_outcomes_l697_69731


namespace specific_case_general_case_l697_69745

-- Define the theorem for the specific case n = 4
theorem specific_case :
  Real.sqrt (4 + 4/15) = 8 * Real.sqrt 15 / 15 := by sorry

-- Define the theorem for the general case
theorem general_case (n : ℕ) (h : n ≥ 2) :
  Real.sqrt (n + n/(n^2 - 1)) = n * Real.sqrt (n/(n^2 - 1)) := by sorry

end specific_case_general_case_l697_69745


namespace symmetric_circle_equation_l697_69779

/-- Given a circle and a line of symmetry, this theorem proves the equation of the symmetric circle. -/
theorem symmetric_circle_equation (x y : ℝ) :
  (x - 3)^2 + (y + 4)^2 = 2 →  -- Original circle equation
  x + y = 0 →               -- Line of symmetry
  (x - 4)^2 + (y + 3)^2 = 2 -- Symmetric circle equation
:= by sorry

end symmetric_circle_equation_l697_69779


namespace solve_equation_l697_69707

theorem solve_equation (y : ℝ) : 7 - y = 10 → y = -3 := by
  sorry

end solve_equation_l697_69707


namespace study_time_for_average_75_l697_69752

/-- Represents the relationship between study time and test score -/
structure StudyScoreRelation where
  studyTime : ℝ
  score : ℝ
  ratio : ℝ
  rel : score = ratio * studyTime

/-- Proves that 4.5 hours of study will result in a score of 90, given the initial condition -/
theorem study_time_for_average_75 
  (initial : StudyScoreRelation) 
  (h_initial : initial.studyTime = 3 ∧ initial.score = 60) :
  ∃ (second : StudyScoreRelation), 
    second.studyTime = 4.5 ∧ 
    second.score = 90 ∧ 
    (initial.score + second.score) / 2 = 75 ∧
    second.ratio = initial.ratio := by
  sorry

end study_time_for_average_75_l697_69752


namespace biotech_job_count_l697_69782

/-- Represents the class of 2000 biotechnology graduates --/
structure BiotechClass :=
  (total : ℕ)
  (secondDegree : ℕ)
  (bothJobAndDegree : ℕ)
  (neither : ℕ)

/-- Calculates the number of graduates who found a job --/
def graduatesWithJob (c : BiotechClass) : ℕ :=
  c.total - c.neither - (c.secondDegree - c.bothJobAndDegree)

/-- Theorem: In the given biotech class, 32 graduates found a job --/
theorem biotech_job_count (c : BiotechClass) 
  (h1 : c.total = 73)
  (h2 : c.secondDegree = 45)
  (h3 : c.bothJobAndDegree = 13)
  (h4 : c.neither = 9) :
  graduatesWithJob c = 32 := by
sorry

end biotech_job_count_l697_69782


namespace no_real_roots_l697_69725

theorem no_real_roots :
  ¬∃ x : ℝ, x^2 = 2*x - 3 := by
sorry

end no_real_roots_l697_69725


namespace four_star_three_equals_nineteen_l697_69740

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := a^2 + a*b - b^2

-- State the theorem
theorem four_star_three_equals_nineteen :
  customOp 4 3 = 19 := by sorry

end four_star_three_equals_nineteen_l697_69740


namespace sum_always_negative_l697_69713

def f (x : ℝ) : ℝ := -x - x^3

theorem sum_always_negative (α β γ : ℝ) 
  (h1 : α + β > 0) (h2 : β + γ > 0) (h3 : γ + α > 0) : 
  f α + f β + f γ < 0 := by
  sorry

end sum_always_negative_l697_69713


namespace students_have_two_hands_l697_69759

/-- Given a class with the following properties:
  * There are 11 students including Peter
  * The total number of hands excluding Peter's is 20
  * Every student has the same number of hands
  Prove that each student has 2 hands. -/
theorem students_have_two_hands
  (total_students : ℕ)
  (hands_excluding_peter : ℕ)
  (h_total_students : total_students = 11)
  (h_hands_excluding_peter : hands_excluding_peter = 20) :
  hands_excluding_peter + 2 = total_students * 2 :=
sorry

end students_have_two_hands_l697_69759


namespace xy_value_l697_69786

theorem xy_value (x y : ℝ) (h : (x + 22) / y + 290 / (x * y) = (26 - y) / x) :
  x * y = -143 := by sorry

end xy_value_l697_69786


namespace even_increasing_ordering_l697_69703

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

theorem even_increasing_ordering (f : ℝ → ℝ) 
  (h_even : is_even f) (h_incr : increasing_on_pos f) : 
  f 3 < f (-Real.pi) ∧ f (-Real.pi) < f (-4) := by sorry

end even_increasing_ordering_l697_69703


namespace solution_interval_l697_69776

theorem solution_interval (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9 ↔ 63 / 26 < x ∧ x ≤ 14 / 5 := by
  sorry

end solution_interval_l697_69776


namespace parallel_vectors_imply_x_value_l697_69719

/-- Given two 2D vectors a and b, if a + b is parallel to 2a - b, then the x-coordinate of b is -4. -/
theorem parallel_vectors_imply_x_value (a b : ℝ × ℝ) (h : a = (2, 1)) (h' : b.2 = -2) :
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (2 • a - b)) → b.1 = -4 := by
  sorry

end parallel_vectors_imply_x_value_l697_69719


namespace largest_angle_of_specific_triangle_l697_69743

/-- Given a triangle with sides 3√2, 6, and 3√10, its largest interior angle is 135°. -/
theorem largest_angle_of_specific_triangle : 
  ∀ (a b c θ : ℝ), 
  a = 3 * Real.sqrt 2 → 
  b = 6 → 
  c = 3 * Real.sqrt 10 → 
  c > a ∧ c > b → 
  θ = Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) → 
  θ = 135 * (π / 180) := by sorry

end largest_angle_of_specific_triangle_l697_69743


namespace factor_values_l697_69748

def polynomial (x : ℝ) : ℝ := 8 * x^2 + 18 * x - 5

theorem factor_values (t : ℝ) : 
  (∀ x, polynomial x = 0 → x = t) ↔ t = 1/4 ∨ t = -5 := by
  sorry

end factor_values_l697_69748


namespace password_is_5949_l697_69756

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_ambiguous_for_alice (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ 
    ((5000 + x * 100 + y * 10) % 9 = 0 ∨ (5000 + x * 100 + y * 10 + 9) % 9 = 0)

def is_ambiguous_for_bob (n : ℕ) : Prop :=
  ∃ (y z : ℕ), y < 10 ∧ z < 10 ∧ 
    ((5000 + y * 10 + z) % 9 = 0 ∨ (5000 + 900 + y * 10 + z) % 9 = 0)

theorem password_is_5949 :
  ∀ n : ℕ,
  5000 ≤ n ∧ n < 6000 →
  is_multiple_of_9 n →
  is_ambiguous_for_alice n →
  is_ambiguous_for_bob n →
  n ≤ 5949 :=
sorry

end password_is_5949_l697_69756


namespace polynomial_division_theorem_l697_69751

theorem polynomial_division_theorem (x : ℝ) :
  ∃ r : ℝ, (5 * x^2 - 5 * x + 3) * (2 * x + 4) + r = 10 * x^3 + 20 * x^2 - 9 * x + 6 ∧ 
  (∃ c : ℝ, r = c) := by
  sorry

end polynomial_division_theorem_l697_69751


namespace sams_remaining_pennies_l697_69775

/-- Given an initial amount of pennies and an amount spent, calculate the remaining pennies -/
def remaining_pennies (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem: Sam's remaining pennies -/
theorem sams_remaining_pennies :
  remaining_pennies 98 93 = 5 := by
  sorry

end sams_remaining_pennies_l697_69775


namespace no_nine_diagonals_intersection_l697_69742

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- A diagonal of a polygon -/
def Diagonal (n : ℕ) (p : RegularPolygon n) (i j : Fin n) : Set (ℝ × ℝ) :=
  sorry

/-- The set of all diagonals in a polygon -/
def AllDiagonals (n : ℕ) (p : RegularPolygon n) : Set (Set (ℝ × ℝ)) :=
  sorry

/-- A point is internal to a polygon if it's inside the polygon -/
def IsInternal (n : ℕ) (p : RegularPolygon n) (point : ℝ × ℝ) : Prop :=
  sorry

/-- The number of diagonals passing through a point -/
def DiagonalsThroughPoint (n : ℕ) (p : RegularPolygon n) (point : ℝ × ℝ) : ℕ :=
  sorry

theorem no_nine_diagonals_intersection (p : RegularPolygon 25) 
  (diags : AllDiagonals 25 p) :
  ¬ ∃ (point : ℝ × ℝ), IsInternal 25 p point ∧ DiagonalsThroughPoint 25 p point = 9 :=
sorry

end no_nine_diagonals_intersection_l697_69742


namespace jacob_age_2005_l697_69793

/-- Given that Jacob was one-third as old as his grandfather at the end of 2000,
    and the sum of the years in which they were born is 3858,
    prove that Jacob will be 40.5 years old at the end of 2005. -/
theorem jacob_age_2005 (jacob_age_2000 : ℝ) (grandfather_age_2000 : ℝ) :
  jacob_age_2000 = (1 / 3) * grandfather_age_2000 →
  (2000 - jacob_age_2000) + (2000 - grandfather_age_2000) = 3858 →
  jacob_age_2000 + 5 = 40.5 := by
sorry

end jacob_age_2005_l697_69793


namespace two_players_goals_l697_69702

theorem two_players_goals (total_goals : ℕ) (players : ℕ) (percentage : ℚ) 
  (h1 : total_goals = 300)
  (h2 : players = 2)
  (h3 : percentage = 1/5) : 
  (↑total_goals * percentage) / players = 30 := by
  sorry

end two_players_goals_l697_69702


namespace matt_jump_time_l697_69717

/-- Given that Matt skips rope 3 times per second and gets 1800 skips in total,
    prove that he jumped for 10 minutes. -/
theorem matt_jump_time (skips_per_second : ℕ) (total_skips : ℕ) (jump_time : ℕ) :
  skips_per_second = 3 →
  total_skips = 1800 →
  jump_time * 60 * skips_per_second = total_skips →
  jump_time = 10 :=
by sorry

end matt_jump_time_l697_69717


namespace simplify_expression_l697_69783

theorem simplify_expression (t : ℝ) (h : t ≠ 0) :
  (t^5 * t^3) / t^4 = t^4 := by
  sorry

end simplify_expression_l697_69783
