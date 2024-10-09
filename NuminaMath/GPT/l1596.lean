import Mathlib

namespace problem_l1596_159605

def a : ℝ := (-2)^2002
def b : ℝ := (-2)^2003

theorem problem : a + b = -2^2002 := by
  sorry

end problem_l1596_159605


namespace find_larger_number_l1596_159684

theorem find_larger_number :
  ∃ (x y : ℝ), (y = x + 10) ∧ (x = y / 2) ∧ (x + y = 34) → y = 20 :=
by
  sorry

end find_larger_number_l1596_159684


namespace quadratic_real_solution_l1596_159626

theorem quadratic_real_solution (m : ℝ) (i : ℂ) (h_i : i * i = -1)
  (h_quad : ∃ z : ℝ, z^2 + (i * z) + m = 0) : m = 0 :=
sorry

end quadratic_real_solution_l1596_159626


namespace age_problem_l1596_159675

theorem age_problem (age x : ℕ) (h : age = 64) :
  (1 / 2 : ℝ) * (8 * (age + x) - 8 * (age - 8)) = age → x = 8 :=
by
  sorry

end age_problem_l1596_159675


namespace models_kirsty_can_buy_l1596_159698

def savings := 30 * 0.45
def new_price := 0.50

theorem models_kirsty_can_buy : savings / new_price = 27 := by
  sorry

end models_kirsty_can_buy_l1596_159698


namespace find_ab_l1596_159609

variable (a b m n : ℝ)

theorem find_ab (h1 : (a + b)^2 = m) (h2 : (a - b)^2 = n) : 
  a * b = (m - n) / 4 :=
by
  sorry

end find_ab_l1596_159609


namespace value_of_f_at_6_l1596_159674

variable {R : Type*} [LinearOrderedField R]

noncomputable def f : R → R := sorry

-- Conditions
axiom odd_function (x : R) : f (-x) = -f x
axiom periodicity (x : R) : f (x + 2) = -f x

-- Theorem to prove
theorem value_of_f_at_6 : f 6 = 0 := by sorry

end value_of_f_at_6_l1596_159674


namespace symmetry_x_y_axis_symmetry_line_y_neg1_l1596_159663

-- Define point P
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P : Point := { x := 1, y := 2 }

-- Condition for symmetry with respect to x-axis
def symmetric_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Condition for symmetry with respect to the line y = -1
def symmetric_line_y_neg1 (p : Point) : Point :=
  { x := p.x, y := 2 * 1 - p.y - 1 }

-- Theorem statements
theorem symmetry_x_y_axis : symmetric_x P = { x := 1, y := -2 } := sorry
theorem symmetry_line_y_neg1 : symmetric_line_y_neg1 { x := 1, y := -2 } = { x := 1, y := 3 } := sorry

end symmetry_x_y_axis_symmetry_line_y_neg1_l1596_159663


namespace max_volume_is_correct_l1596_159668

noncomputable def max_volume_of_inscribed_sphere (AB BC AA₁ : ℝ) (h₁ : AB = 6) (h₂ : BC = 8) (h₃ : AA₁ = 3) : ℝ :=
  let AC := Real.sqrt ((6 : ℝ) ^ 2 + (8 : ℝ) ^ 2)
  let r := (AB + BC - AC) / 2
  let sphere_radius := AA₁ / 2
  (4/3) * Real.pi * sphere_radius ^ 3

theorem max_volume_is_correct : max_volume_of_inscribed_sphere 6 8 3 (by rfl) (by rfl) (by rfl) = 9 * Real.pi / 2 := by
  sorry

end max_volume_is_correct_l1596_159668


namespace tom_beach_days_l1596_159621

theorem tom_beach_days (total_seashells days_seashells : ℕ) (found_each_day total_found : ℕ) 
    (h1 : found_each_day = 7) (h2 : total_found = 35) : total_found / found_each_day = 5 := 
by 
  sorry

end tom_beach_days_l1596_159621


namespace stork_count_l1596_159618

theorem stork_count (B S : ℕ) (h1 : B = 7) (h2 : B = S + 3) : S = 4 := 
by 
  sorry -- Proof to be filled in


end stork_count_l1596_159618


namespace average_fuel_efficiency_round_trip_l1596_159616

noncomputable def average_fuel_efficiency (d1 d2 mpg1 mpg2 : ℝ) : ℝ :=
  let total_distance := d1 + d2
  let fuel_used := (d1 / mpg1) + (d2 / mpg2)
  total_distance / fuel_used

theorem average_fuel_efficiency_round_trip :
  average_fuel_efficiency 180 180 36 24 = 28.8 :=
by 
  sorry

end average_fuel_efficiency_round_trip_l1596_159616


namespace no_real_roots_of_quadratic_l1596_159688

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

theorem no_real_roots_of_quadratic :
  let a := 2
  let b := -5
  let c := 6
  discriminant a b c < 0 → ¬∃ x : ℝ, 2 * x ^ 2 - 5 * x + 6 = 0 :=
by {
  -- Proof skipped
  sorry
}

end no_real_roots_of_quadratic_l1596_159688


namespace eval_expression_l1596_159693

-- Define the redefined operation
def red_op (a b : ℝ) : ℝ := (a + b)^2

-- Define the target expression to be evaluated
def expr (x y : ℝ) : ℝ := red_op ((x + y)^2) ((x - y)^2)

-- State the theorem
theorem eval_expression (x y : ℝ) : expr x y = 4 * (x^2 + y^2)^2 := by
  sorry

end eval_expression_l1596_159693


namespace length_of_bridge_l1596_159603

noncomputable def speed_kmh_to_mps (speed_kmh : ℕ) : ℝ := speed_kmh * 1000 / 3600

def total_distance_covered (speed_mps : ℝ) (time_s : ℕ) : ℝ := speed_mps * time_s

def bridge_length (total_distance : ℝ) (train_length : ℝ) : ℝ := total_distance - train_length

theorem length_of_bridge (train_length : ℝ) (time_s : ℕ) (speed_kmh : ℕ) :
  bridge_length (total_distance_covered (speed_kmh_to_mps speed_kmh) time_s) train_length = 299.9 :=
by
  have speed_mps := speed_kmh_to_mps speed_kmh
  have total_distance := total_distance_covered speed_mps time_s
  have length_of_bridge := bridge_length total_distance train_length
  sorry

end length_of_bridge_l1596_159603


namespace binary_to_decimal_correct_l1596_159628

def binary_to_decimal : ℕ := 110011

theorem binary_to_decimal_correct : 
  binary_to_decimal = 51 := sorry

end binary_to_decimal_correct_l1596_159628


namespace p_sufficient_not_necessary_for_q_l1596_159672

-- Define the propositions p and q based on the given conditions
def p (α : ℝ) : Prop := α = Real.pi / 4
def q (α : ℝ) : Prop := Real.sin α = Real.cos α

-- Theorem that states p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q (α : ℝ) : p α → (q α) ∧ ¬(q α → p α) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l1596_159672


namespace solve_for_x_l1596_159614

theorem solve_for_x (x y : ℝ) (h1 : 2 * x - 3 * y = 18) (h2 : x + 2 * y = 8) : x = 60 / 7 := sorry

end solve_for_x_l1596_159614


namespace range_of_a_l1596_159632

/-- 
For the system of inequalities in terms of x 
    \begin{cases} 
    x - a < 0 
    ax < 1 
    \end{cases}
the range of values for the real number a such that the solution set is not empty is [-1, ∞).
-/
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x - a < 0 ∧ a * x < 1) ↔ -1 ≤ a :=
by sorry

end range_of_a_l1596_159632


namespace find_x_collinear_l1596_159633

theorem find_x_collinear (x : ℝ) (a b : ℝ × ℝ) (h_a : a = (2, 1)) (h_b : b = (x, -1)) 
  (h_collinear : ∃ k : ℝ, (a.1 - b.1, a.2 - b.2) = (k * b.1, k * b.2)) : x = -2 :=
by 
  -- the proof would go here
  sorry

end find_x_collinear_l1596_159633


namespace initial_card_count_l1596_159660

theorem initial_card_count (x : ℕ) (h1 : (3 * (1/2) * ((x / 3) + (4 / 3))) = 34) : x = 64 :=
  sorry

end initial_card_count_l1596_159660


namespace closest_whole_number_l1596_159638

theorem closest_whole_number :
  let x := (10^2001 + 10^2003) / (10^2002 + 10^2002)
  abs ((x : ℝ) - 5) < 1 :=
by 
  sorry

end closest_whole_number_l1596_159638


namespace alice_forest_walks_l1596_159697

theorem alice_forest_walks
  (morning_distance : ℕ)
  (total_distance : ℕ)
  (days_per_week : ℕ)
  (forest_distance : ℕ) :
  morning_distance = 10 →
  total_distance = 110 →
  days_per_week = 5 →
  (total_distance - morning_distance * days_per_week) / days_per_week = forest_distance →
  forest_distance = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_forest_walks_l1596_159697


namespace mean_temperature_l1596_159694

def temperatures : List ℚ := [80, 79, 81, 85, 87, 89, 87, 90, 89, 88]

theorem mean_temperature :
  let n := temperatures.length
  let sum := List.sum temperatures
  (sum / n : ℚ) = 85.5 :=
by
  sorry

end mean_temperature_l1596_159694


namespace base9_addition_correct_l1596_159692

-- Definition of base 9 addition problem.
def add_base9 (a b c : ℕ) : ℕ :=
  let sum := a + b + c -- Sum in base 10
  let d0 := sum % 9 -- Least significant digit in base 9
  let carry1 := sum / 9
  (carry1 + carry1 / 9 * 9 + carry1 % 9) + d0 -- Sum in base 9 considering carry

-- The specific values converted to base 9 integers
def n1 := 3 * 9^2 + 4 * 9 + 6
def n2 := 8 * 9^2 + 0 * 9 + 2
def n3 := 1 * 9^2 + 5 * 9 + 7

-- The expected result converted to base 9 integer
def expected_sum := 1 * 9^3 + 4 * 9^2 + 1 * 9 + 6

theorem base9_addition_correct : add_base9 n1 n2 n3 = expected_sum := by
  -- Proof will be provided here
  sorry

end base9_addition_correct_l1596_159692


namespace rectangle_length_width_ratio_l1596_159625

-- Define the side lengths of the small squares and the large square
variables (s : ℝ)

-- Define the dimensions of the large square and the rectangle
def large_square_side : ℝ := 5 * s
def rectangle_length : ℝ := 5 * s
def rectangle_width : ℝ := s

-- State and prove the theorem
theorem rectangle_length_width_ratio : rectangle_length s / rectangle_width s = 5 :=
by sorry

end rectangle_length_width_ratio_l1596_159625


namespace total_charging_time_l1596_159695

def charge_smartphone_full : ℕ := 26
def charge_tablet_full : ℕ := 53
def charge_phone_half : ℕ := charge_smartphone_full / 2
def charge_tablet : ℕ := charge_tablet_full

theorem total_charging_time : 
  charge_phone_half + charge_tablet = 66 := by
  sorry

end total_charging_time_l1596_159695


namespace simplify_polynomial_l1596_159654

theorem simplify_polynomial (x : ℝ) (A B C D : ℝ) :
  (y = (x^3 + 12 * x^2 + 47 * x + 60) / (x + 3)) →
  (y = A * x^2 + B * x + C) →
  x ≠ D →
  A = 1 ∧ B = 9 ∧ C = 20 ∧ D = -3 :=
by
  sorry

end simplify_polynomial_l1596_159654


namespace harry_james_payment_l1596_159604

theorem harry_james_payment (x y H : ℝ) (h1 : H - 12 = 44 / y) (h2 : y > 1) (h3 : H != 12 + 44/3) : H = 23 ∧ y = 4 :=
by
  sorry

end harry_james_payment_l1596_159604


namespace find_integers_for_perfect_square_l1596_159678

theorem find_integers_for_perfect_square (x : ℤ) :
  (∃ k : ℤ, x * (x + 1) * (x + 7) * (x + 8) = k^2) ↔ 
  x = -9 ∨ x = -8 ∨ x = -7 ∨ x = -4 ∨ x = -1 ∨ x = 0 ∨ x = 1 :=
sorry

end find_integers_for_perfect_square_l1596_159678


namespace sin_C_of_arith_prog_angles_l1596_159649

theorem sin_C_of_arith_prog_angles (A B C a b : ℝ) (h_abc : A + B + C = Real.pi)
  (h_arith_prog : 2 * B = A + C) (h_a : a = Real.sqrt 2) (h_b : b = Real.sqrt 3) :
  Real.sin C = (Real.sqrt 2 + Real.sqrt 6) / 4 :=
sorry

end sin_C_of_arith_prog_angles_l1596_159649


namespace full_house_plus_two_probability_l1596_159685

def total_ways_to_choose_7_cards_from_52 : ℕ :=
  Nat.choose 52 7

def ways_for_full_house_plus_two : ℕ :=
  13 * 4 * 12 * 6 * 55 * 16

def probability_full_house_plus_two : ℚ :=
  (ways_for_full_house_plus_two : ℚ) / (total_ways_to_choose_7_cards_from_52 : ℚ)

theorem full_house_plus_two_probability :
  probability_full_house_plus_two = 13732 / 3344614 :=
by
  sorry

end full_house_plus_two_probability_l1596_159685


namespace sum_of_coefficients_l1596_159615

theorem sum_of_coefficients (a b : ℝ) (h : ∀ x : ℝ, (x > 1 ∧ x < 4) ↔ (ax^2 + bx - 2 > 0)) :
  a + b = 2 :=
by
  sorry

end sum_of_coefficients_l1596_159615


namespace snowball_game_l1596_159642

theorem snowball_game (x y z : ℕ) (h : 5 * x + 4 * y + 3 * z = 12) : 
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end snowball_game_l1596_159642


namespace find_certain_number_l1596_159670

noncomputable def certain_number_is_square (n : ℕ) (x : ℕ) : Prop :=
  ∃ (y : ℕ), x * n = y * y

theorem find_certain_number : ∃ x, certain_number_is_square 3 x :=
by 
  use 1
  unfold certain_number_is_square
  use 3
  sorry

end find_certain_number_l1596_159670


namespace differential_solution_l1596_159696

theorem differential_solution (C : ℝ) : 
  ∃ y : ℝ → ℝ, (∀ x : ℝ, y x = C * (1 + x^2)) := 
by
  sorry

end differential_solution_l1596_159696


namespace final_problem_l1596_159606

-- Define the function f
def f (x p q : ℝ) : ℝ := x * abs x + p * x + q

-- Proposition ①: When q=0, f(x) is an odd function
def prop1 (p : ℝ) : Prop :=
  ∀ x : ℝ, f x p 0 = - f (-x) p 0

-- Proposition ②: The graph of y=f(x) is symmetric with respect to the point (0,q)
def prop2 (p q : ℝ) : Prop :=
  ∀ x : ℝ, f x p q = f (-x) p q + 2 * q

-- Proposition ③: When p=0 and q > 0, the equation f(x)=0 has exactly one real root
def prop3 (q : ℝ) : Prop :=
  q > 0 → ∃! x : ℝ, f x 0 q = 0

-- Proposition ④: The equation f(x)=0 has at most two real roots
def prop4 (p q : ℝ) : Prop :=
  ∀ x1 x2 x3 : ℝ, f x1 p q = 0 ∧ f x2 p q = 0 ∧ f x3 p q = 0 → x1 = x2 ∨ x1 = x3 ∨ x2 = x3

-- The final problem to prove that propositions ①, ②, and ③ are true and proposition ④ is false
theorem final_problem (p q : ℝ) :
  prop1 p ∧ prop2 p q ∧ prop3 q ∧ ¬prop4 p q :=
sorry

end final_problem_l1596_159606


namespace area_relation_l1596_159653

-- Define the areas of the triangles
variables (a b c : ℝ)

-- Define the condition that triangles T_a and T_c are similar (i.e., homothetic)
-- which implies the relationship between their areas.
theorem area_relation (ha : 0 < a) (hc : 0 < c) (habc : b = Real.sqrt (a * c)) : b = Real.sqrt (a * c) := by
  sorry

end area_relation_l1596_159653


namespace curve_crossing_self_l1596_159679

theorem curve_crossing_self (t t' : ℝ) :
  (t^3 - t - 2 = t'^3 - t' - 2) ∧ (t ≠ t') ∧ 
  (t^3 - t^2 - 9 * t + 5 = t'^3 - t'^2 - 9 * t' + 5) → 
  (t = 3 ∧ t' = -3) ∨ (t = -3 ∧ t' = 3) →
  (t^3 - t - 2 = 22) ∧ (t^3 - t^2 - 9 * t + 5 = -4) :=
by
  sorry

end curve_crossing_self_l1596_159679


namespace inequality_holds_l1596_159657

variables {a b c : ℝ}

theorem inequality_holds (h1 : c < b) (h2 : b < a) (h3 : ac < 0) : ab > ac :=
sorry

end inequality_holds_l1596_159657


namespace parabola_intersection_points_l1596_159667

theorem parabola_intersection_points :
  let parabola1 := λ x : ℝ => 4*x^2 + 3*x - 1
  let parabola2 := λ x : ℝ => x^2 + 8*x + 7
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ = -4/3 ∧ y₁ = -17/9 ∧
                        x₂ = 2 ∧ y₂ = 27 ∧
                        parabola1 x₁ = y₁ ∧ 
                        parabola2 x₁ = y₁ ∧
                        parabola1 x₂ = y₂ ∧
                        parabola2 x₂ = y₂ :=
by {
  sorry
}

end parabola_intersection_points_l1596_159667


namespace total_cost_proof_l1596_159627

-- Definitions for the problem conditions
def basketball_cost : ℕ := 48
def volleyball_cost : ℕ := basketball_cost - 18
def basketball_quantity : ℕ := 3
def volleyball_quantity : ℕ := 5
def total_basketball_cost : ℕ := basketball_cost * basketball_quantity
def total_volleyball_cost : ℕ := volleyball_cost * volleyball_quantity
def total_cost : ℕ := total_basketball_cost + total_volleyball_cost

-- Theorem to be proved
theorem total_cost_proof : total_cost = 294 :=
by
  sorry

end total_cost_proof_l1596_159627


namespace is_not_prime_390629_l1596_159645

theorem is_not_prime_390629 : ¬ Prime 390629 :=
sorry

end is_not_prime_390629_l1596_159645


namespace bus_stop_time_l1596_159687

theorem bus_stop_time (v_exclude_stop v_include_stop : ℕ) (h1 : v_exclude_stop = 54) (h2 : v_include_stop = 36) : 
  ∃ t: ℕ, t = 20 :=
by
  sorry

end bus_stop_time_l1596_159687


namespace lowest_number_in_range_l1596_159661

theorem lowest_number_in_range (y : ℕ) (h : ∀ x y : ℕ, 0 < x ∧ x < y) : ∃ x : ℕ, x = 999 :=
by
  existsi 999
  sorry

end lowest_number_in_range_l1596_159661


namespace grace_apples_after_6_weeks_l1596_159662

def apples_per_day_bella : ℕ := 6

def days_per_week : ℕ := 7

def fraction_apples_bella_consumes : ℚ := 1/3

def weeks : ℕ := 6

theorem grace_apples_after_6_weeks :
  let apples_per_week_bella := apples_per_day_bella * days_per_week
  let apples_per_week_grace := apples_per_week_bella / fraction_apples_bella_consumes
  let remaining_apples_week := apples_per_week_grace - apples_per_week_bella
  let total_apples := remaining_apples_week * weeks
  total_apples = 504 := by
  sorry

end grace_apples_after_6_weeks_l1596_159662


namespace find_original_number_l1596_159677

theorem find_original_number (x : ℤ) (h : 3 * (2 * x + 9) = 57) : x = 5 := by
  sorry

end find_original_number_l1596_159677


namespace sum_of_roots_of_f_l1596_159682

noncomputable def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = - f x

noncomputable def f_increasing_on (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x < y → f x < f y

theorem sum_of_roots_of_f (f : ℝ → ℝ) (m : ℝ) (x1 x2 x3 x4 : ℝ)
  (h1 : odd_function f)
  (h2 : ∀ x, f (x - 4) = - f x)
  (h3 : f_increasing_on f 0 2)
  (h4 : m > 0)
  (h5 : f x1 = m)
  (h6 : f x2 = m)
  (h7 : f x3 = m)
  (h8 : f x4 = m)
  (h9 : x1 ≠ x2)
  (h10 : x1 ≠ x3)
  (h11 : x1 ≠ x4)
  (h12 : x2 ≠ x3)
  (h13 : x2 ≠ x4)
  (h14 : x3 ≠ x4)
  (h15 : ∀ x, -8 ≤ x ∧ x ≤ 8 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4) :
  x1 + x2 + x3 + x4 = -8 :=
sorry

end sum_of_roots_of_f_l1596_159682


namespace sequence_general_formula_l1596_159634

theorem sequence_general_formula (a : ℕ → ℕ)
    (h1 : a 1 = 3) 
    (h2 : a 2 = 4) 
    (h3 : a 3 = 6) 
    (h4 : a 4 = 10) 
    (h5 : a 5 = 18) :
    ∀ n : ℕ, a n = 2^(n-1) + 2 :=
sorry

end sequence_general_formula_l1596_159634


namespace total_balloons_cost_is_91_l1596_159648

-- Define the number of balloons and their costs for Fred, Sam, and Dan
def fred_balloons : ℕ := 10
def fred_cost_per_balloon : ℝ := 1

def sam_balloons : ℕ := 46
def sam_cost_per_balloon : ℝ := 1.5

def dan_balloons : ℕ := 16
def dan_cost_per_balloon : ℝ := 0.75

-- Calculate the total cost for each person’s balloons
def fred_total_cost : ℝ := fred_balloons * fred_cost_per_balloon
def sam_total_cost : ℝ := sam_balloons * sam_cost_per_balloon
def dan_total_cost : ℝ := dan_balloons * dan_cost_per_balloon

-- Calculate the total cost of all the balloons combined
def total_cost : ℝ := fred_total_cost + sam_total_cost + dan_total_cost

-- The main statement to be proved
theorem total_balloons_cost_is_91 : total_cost = 91 :=
by
  -- Recall that the previous individual costs can be worked out and added
  -- But for the sake of this statement, we use sorry to skip details
  sorry

end total_balloons_cost_is_91_l1596_159648


namespace find_whole_number_M_l1596_159636

-- Define the conditions
def condition (M : ℕ) : Prop :=
  21 < M ∧ M < 23

-- Define the main theorem to be proven
theorem find_whole_number_M (M : ℕ) (h : condition M) : M = 22 := by
  sorry

end find_whole_number_M_l1596_159636


namespace clarinet_cost_correct_l1596_159665

noncomputable def total_spent : ℝ := 141.54
noncomputable def song_book_cost : ℝ := 11.24
noncomputable def clarinet_cost : ℝ := total_spent - song_book_cost

theorem clarinet_cost_correct : clarinet_cost = 130.30 :=
by
  sorry

end clarinet_cost_correct_l1596_159665


namespace range_of_x_l1596_159699

theorem range_of_x (x : ℝ) : (∃ y : ℝ, y = (2 / (Real.sqrt (x - 1)))) → (x > 1) :=
by
  sorry

end range_of_x_l1596_159699


namespace infection_equation_l1596_159639

-- Given conditions
def initially_infected : Nat := 1
def total_after_two_rounds : ℕ := 81
def avg_infect_per_round (x : ℕ) : ℕ := x

-- Mathematically equivalent proof problem
theorem infection_equation (x : ℕ) 
  (h1 : initially_infected = 1)
  (h2 : total_after_two_rounds = 81)
  (h3 : ∀ (y : ℕ), initially_infected + avg_infect_per_round y + (avg_infect_per_round y)^2 = total_after_two_rounds):
  (1 + x)^2 = 81 :=
by
  sorry

end infection_equation_l1596_159639


namespace initial_plants_count_l1596_159650

theorem initial_plants_count (p : ℕ) 
    (h1 : p - 20 > 0)
    (h2 : (p - 20) / 2 > 0)
    (h3 : ((p - 20) / 2) - 1 > 0)
    (h4 : ((p - 20) / 2) - 1 = 4) : 
    p = 30 :=
by
  sorry

end initial_plants_count_l1596_159650


namespace k_l_m_n_values_l1596_159619

theorem k_l_m_n_values (k l m n : ℕ) (hk : 0 < k) (hl : 0 < l) (hm : 0 < m) (hn : 0 < n)
  (hklmn : k + l + m + n = k * m) (hln : k + l + m + n = l * n) :
  k + l + m + n = 16 ∨ k + l + m + n = 18 ∨ k + l + m + n = 24 ∨ k + l + m + n = 30 :=
sorry

end k_l_m_n_values_l1596_159619


namespace speed_difference_is_zero_l1596_159686

theorem speed_difference_is_zero :
  let distance_bike := 72
  let time_bike := 9
  let distance_truck := 72
  let time_truck := 9
  let speed_bike := distance_bike / time_bike
  let speed_truck := distance_truck / time_truck
  (speed_truck - speed_bike) = 0 := by
  sorry

end speed_difference_is_zero_l1596_159686


namespace range_of_k_l1596_159602

-- Define the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + k

-- Define the condition that the function does not pass through the third quadrant
def does_not_pass_third_quadrant (k : ℝ) : Prop :=
  ∀ x : ℝ, (x < 0 ∧ linear_function k x < 0) → false

-- Theorem statement proving the range of k
theorem range_of_k (k : ℝ) : does_not_pass_third_quadrant k ↔ (0 ≤ k ∧ k < 2) :=
by
  sorry

end range_of_k_l1596_159602


namespace solve_ab_cd_l1596_159624

theorem solve_ab_cd (a b c d : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a + b + d = -2) 
  (h3 : a + c + d = 5) 
  (h4 : b + c + d = 4) 
  : a * b + c * d = 26 / 9 := 
by {
  sorry
}

end solve_ab_cd_l1596_159624


namespace find_x_l1596_159631

theorem find_x (x : ℝ) (h : 1 / 7 + 7 / x = 15 / x + 1 / 15) : x = 105 := 
by 
  sorry

end find_x_l1596_159631


namespace tom_seashells_found_l1596_159676

/-- 
Given:
- sally_seashells = 9 (number of seashells Sally found)
- jessica_seashells = 5 (number of seashells Jessica found)
- total_seashells = 21 (number of seashells found together)

Prove that the number of seashells that Tom found (tom_seashells) is 7.
-/
theorem tom_seashells_found (sally_seashells jessica_seashells total_seashells tom_seashells : ℕ)
  (h₁ : sally_seashells = 9) (h₂ : jessica_seashells = 5) (h₃ : total_seashells = 21) :
  tom_seashells = 7 :=
by
  sorry

end tom_seashells_found_l1596_159676


namespace value_of_v3_at_neg4_l1596_159610

def poly (x : ℤ) : ℤ := (((((2 * x + 5) * x + 6) * x + 23) * x - 8) * x + 10) * x - 3

theorem value_of_v3_at_neg4 : poly (-4) = -49 := 
by
  sorry

end value_of_v3_at_neg4_l1596_159610


namespace sum_of_first_three_tests_l1596_159601

variable (A B C: ℕ)

def scores (A B C test4 : ℕ) : Prop := (A + B + C + test4) / 4 = 85

theorem sum_of_first_three_tests (h : scores A B C 100) : A + B + C = 240 :=
by
  -- Proof goes here
  sorry

end sum_of_first_three_tests_l1596_159601


namespace one_fourth_more_than_x_equals_twenty_percent_less_than_80_l1596_159690

theorem one_fourth_more_than_x_equals_twenty_percent_less_than_80 :
  ∃ n : ℝ, (80 - 0.30 * 80 = 56) ∧ (5 / 4 * n = 56) ∧ (n = 45) :=
by
  sorry

end one_fourth_more_than_x_equals_twenty_percent_less_than_80_l1596_159690


namespace maximum_value_conditions_l1596_159671

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem maximum_value_conditions (x_0 : ℝ) (h_max : ∀ x : ℝ, f x ≤ f x_0) :
    f x_0 = x_0 ∧ f x_0 < 1 / 2 :=
by
  sorry

end maximum_value_conditions_l1596_159671


namespace participants_l1596_159608

variable {A B C D : Prop}

theorem participants (h1 : A → B) (h2 : ¬C → ¬B) (h3 : C → ¬D) :
  (¬A ∧ C ∧ B ∧ ¬D) ∨ ¬B :=
by
  -- The proof is not provided
  sorry

end participants_l1596_159608


namespace divide_19_degree_angle_into_19_equal_parts_l1596_159689

/-- Divide a 19° angle into 19 equal parts, resulting in each part being 1° -/
theorem divide_19_degree_angle_into_19_equal_parts
  (α : ℝ) (hα : α = 19) :
  α / 19 = 1 :=
by
  sorry

end divide_19_degree_angle_into_19_equal_parts_l1596_159689


namespace second_piece_weight_l1596_159647

theorem second_piece_weight (w1 : ℝ) (s1 : ℝ) (s2 : ℝ) (w2 : ℝ) :
  (s1 = 4) → (w1 = 16) → (s2 = 6) → w2 = w1 * (s2^2 / s1^2) → w2 = 36 :=
by
  intro h_s1 h_w1 h_s2 h_w2
  rw [h_s1, h_w1, h_s2] at h_w2
  norm_num at h_w2
  exact h_w2

end second_piece_weight_l1596_159647


namespace river_road_cars_l1596_159680

theorem river_road_cars
  (B C : ℕ)
  (h1 : B * 17 = C)
  (h2 : C = B + 80) :
  C = 85 := by
  sorry

end river_road_cars_l1596_159680


namespace customer_paid_amount_l1596_159612

theorem customer_paid_amount 
  (cost_price : ℝ) 
  (markup_percent : ℝ) 
  (customer_payment : ℝ)
  (h1 : cost_price = 1250) 
  (h2 : markup_percent = 0.60)
  (h3 : customer_payment = cost_price + (markup_percent * cost_price)) :
  customer_payment = 2000 :=
sorry

end customer_paid_amount_l1596_159612


namespace no_positive_real_solution_l1596_159635

open Real

theorem no_positive_real_solution (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  ¬(∀ n : ℕ, 0 < n → (n - 2) / a ≤ ⌊b * n⌋ ∧ ⌊b * n⌋ < (n - 1) / a) :=
by sorry

end no_positive_real_solution_l1596_159635


namespace inequality_proof_l1596_159683

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem inequality_proof :
  (a / (a + b)) * ((a + 2 * b) / (a + 3 * b)) < Real.sqrt (a / (a + 4 * b)) :=
sorry

end inequality_proof_l1596_159683


namespace production_units_l1596_159611

-- Define the production function U
def U (women hours days : ℕ) : ℕ := women * hours * days

-- State the theorem
theorem production_units (x z : ℕ) (hx : ¬ x = 0) :
  U z z z = (z^3 / x) :=
  sorry

end production_units_l1596_159611


namespace time_A_to_complete_race_l1596_159617

noncomputable def km_race_time (V_B : ℕ) : ℚ :=
  940 / V_B

theorem time_A_to_complete_race : km_race_time 6 = 156.67 := by
  sorry

end time_A_to_complete_race_l1596_159617


namespace no_positive_integer_has_product_as_perfect_square_l1596_159643

theorem no_positive_integer_has_product_as_perfect_square:
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, n * (n + 1) = k * k :=
by
  sorry

end no_positive_integer_has_product_as_perfect_square_l1596_159643


namespace max_non_managers_l1596_159607

theorem max_non_managers (N : ℕ) : (8 / N : ℚ) > 7 / 32 → N ≤ 36 :=
by sorry

end max_non_managers_l1596_159607


namespace problem_statement_l1596_159669

variable {a b c d k : ℝ}

theorem problem_statement (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h_pos : 0 < k)
    (h_sum_ab : a + b = k)
    (h_sum_cd : c + d = k^2)
    (h_roots1 : ∀ x, x^2 - 4*a*x - 5*b = 0 → x = c ∨ x = d)
    (h_roots2 : ∀ x, x^2 - 4*c*x - 5*d = 0 → x = a ∨ x = b) : 
    a + b + c + d = k + k^2 :=
sorry

end problem_statement_l1596_159669


namespace negation_of_statement_6_l1596_159681

variable (Teenager Adult : Type)
variable (CanCookWell : Teenager → Prop)
variable (CanCookWell' : Adult → Prop)

-- Conditions from the problem
def all_teenagers_can_cook_well : Prop :=
  ∀ t : Teenager, CanCookWell t

def some_teenagers_can_cook_well : Prop :=
  ∃ t : Teenager, CanCookWell t

def no_adults_can_cook_well : Prop :=
  ∀ a : Adult, ¬CanCookWell' a

def all_adults_cannot_cook_well : Prop :=
  ∀ a : Adult, ¬CanCookWell' a

def at_least_one_adult_cannot_cook_well : Prop :=
  ∃ a : Adult, ¬CanCookWell' a

def all_adults_can_cook_well : Prop :=
  ∀ a : Adult, CanCookWell' a

-- Theorem to prove
theorem negation_of_statement_6 :
  at_least_one_adult_cannot_cook_well Adult CanCookWell' = ¬ all_adults_can_cook_well Adult CanCookWell' :=
sorry

end negation_of_statement_6_l1596_159681


namespace problem_statement_l1596_159659

noncomputable def floor_T (u v w x : ℝ) : ℤ :=
  ⌊u + v + w + x⌋

theorem problem_statement (u v w x : ℝ) (T : ℝ) (h₁: u^2 + v^2 = 3005) (h₂: w^2 + x^2 = 3005) (h₃: u * w = 1729) (h₄: v * x = 1729) :
  floor_T u v w x = 155 :=
by
  sorry

end problem_statement_l1596_159659


namespace geese_count_l1596_159637

theorem geese_count (initial : ℕ) (flown_away : ℕ) (left : ℕ) 
  (h₁ : initial = 51) (h₂ : flown_away = 28) : 
  left = initial - flown_away → left = 23 := 
by
  sorry

end geese_count_l1596_159637


namespace archer_probability_less_than_8_l1596_159641

-- Define the conditions as probabilities for hitting the 10-ring, 9-ring, and 8-ring.
def p_10 : ℝ := 0.24
def p_9 : ℝ := 0.28
def p_8 : ℝ := 0.19

-- Define the probability that the archer scores at least 8.
def p_at_least_8 : ℝ := p_10 + p_9 + p_8

-- Calculate the probability of the archer scoring less than 8.
def p_less_than_8 : ℝ := 1 - p_at_least_8

-- Now, state the theorem to prove that this probability is equal to 0.29.
theorem archer_probability_less_than_8 : p_less_than_8 = 0.29 := by sorry

end archer_probability_less_than_8_l1596_159641


namespace gear_angular_speeds_ratio_l1596_159664

noncomputable def gear_ratio (x y z w : ℕ) (ω_A ω_B ω_C ω_D : ℝ) :=
  x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C ∧ z * ω_C = w * ω_D

theorem gear_angular_speeds_ratio (x y z w : ℕ) (ω_A ω_B ω_C ω_D : ℝ) 
  (h : gear_ratio x y z w ω_A ω_B ω_C ω_D) :
  ω_A / ω_B = y / x ∧ ω_B / ω_C = z / y ∧ ω_C / ω_D = w / z :=
by sorry

end gear_angular_speeds_ratio_l1596_159664


namespace pizza_shared_cost_l1596_159640

theorem pizza_shared_cost (total_price : ℕ) (num_people : ℕ) (share: ℕ)
  (h1 : total_price = 40) (h2 : num_people = 5) : share = 8 :=
by
  sorry

end pizza_shared_cost_l1596_159640


namespace Carter_reads_30_pages_in_1_hour_l1596_159651

variables (C L O : ℕ)

def Carter_reads_half_as_many_pages_as_Lucy_in_1_hour (C L : ℕ) : Prop :=
  C = L / 2

def Lucy_reads_20_more_pages_than_Oliver_in_1_hour (L O : ℕ) : Prop :=
  L = O + 20

def Oliver_reads_40_pages_in_1_hour (O : ℕ) : Prop :=
  O = 40

theorem Carter_reads_30_pages_in_1_hour
  (C L O : ℕ)
  (h1 : Carter_reads_half_as_many_pages_as_Lucy_in_1_hour C L)
  (h2 : Lucy_reads_20_more_pages_than_Oliver_in_1_hour L O)
  (h3 : Oliver_reads_40_pages_in_1_hour O) : 
  C = 30 :=
by
  sorry

end Carter_reads_30_pages_in_1_hour_l1596_159651


namespace sqrt_meaningful_range_l1596_159655

theorem sqrt_meaningful_range (x : ℝ) : 
  (x + 4) ≥ 0 ↔ x ≥ -4 :=
by sorry

end sqrt_meaningful_range_l1596_159655


namespace max_value_of_a_l1596_159600

theorem max_value_of_a :
  ∀ (m : ℚ) (x : ℤ),
    (0 < x ∧ x ≤ 50) →
    (1 / 2 < m ∧ m < 25 / 49) →
    (∀ k : ℤ, m * x + 3 ≠ k) →
  m < 25 / 49 :=
sorry

end max_value_of_a_l1596_159600


namespace Karl_max_score_l1596_159620

def max_possible_score : ℕ :=
  69

theorem Karl_max_score (minutes problems : ℕ) (n_points : ℕ → ℕ) (time_1_5 : ℕ) (time_6_10 : ℕ) (time_11_15 : ℕ)
    (h1 : minutes = 15) (h2 : problems = 15)
    (h3 : ∀ n, n = n_points n)
    (h4 : ∀ i, 1 ≤ i ∧ i ≤ 5 → time_1_5 = 1)
    (h5 : ∀ i, 6 ≤ i ∧ i ≤ 10 → time_6_10 = 2)
    (h6 : ∀ i, 11 ≤ i ∧ i ≤ 15 → time_11_15 = 3) : 
    max_possible_score = 69 :=
  by
  sorry

end Karl_max_score_l1596_159620


namespace investment2_rate_l1596_159629

-- Define the initial conditions
def total_investment : ℝ := 10000
def investment1 : ℝ := 4000
def rate1 : ℝ := 0.05
def investment2 : ℝ := 3500
def income1 : ℝ := investment1 * rate1
def yearly_income_goal : ℝ := 500
def remaining_investment : ℝ := total_investment - investment1 - investment2
def rate3 : ℝ := 0.064
def income3 : ℝ := remaining_investment * rate3

-- The main theorem
theorem investment2_rate (rate2 : ℝ) : 
  income1 + income3 + investment2 * (rate2 / 100) = yearly_income_goal → rate2 = 4 := 
by 
  sorry

end investment2_rate_l1596_159629


namespace rhombus_area_eq_54_l1596_159656

theorem rhombus_area_eq_54
  (a b : ℝ) (eq_long_side : a = 4 * Real.sqrt 3) (eq_short_side : b = 3 * Real.sqrt 3)
  (rhombus_diagonal1 : ℝ := 9 * Real.sqrt 3) (rhombus_diagonal2 : ℝ := 4 * Real.sqrt 3) :
  (1 / 2) * rhombus_diagonal1 * rhombus_diagonal2 = 54 := by
  sorry

end rhombus_area_eq_54_l1596_159656


namespace lucy_last_10_shots_l1596_159691

variable (shots_30 : ℕ) (percentage_30 : ℚ) (total_shots : ℕ) (percentage_40 : ℚ)
variable (shots_made_30 : ℕ) (shots_made_40 : ℕ) (shots_made_last_10 : ℕ)

theorem lucy_last_10_shots 
    (h1 : shots_30 = 30) 
    (h2 : percentage_30 = 0.60) 
    (h3 : total_shots = 40) 
    (h4 : percentage_40 = 0.62 )
    (h5 : shots_made_30 = Nat.floor (percentage_30 * shots_30)) 
    (h6 : shots_made_40 = Nat.floor (percentage_40 * total_shots))
    (h7 : shots_made_last_10 = shots_made_40 - shots_made_30) 
    : shots_made_last_10 = 7 := sorry

end lucy_last_10_shots_l1596_159691


namespace quadratic_solution_l1596_159622

theorem quadratic_solution (x : ℝ) :
  (x^2 + 2 * x = 0) ↔ (x = 0 ∨ x = -2) :=
by
  sorry

end quadratic_solution_l1596_159622


namespace remainder_of_product_mod_5_l1596_159673

theorem remainder_of_product_mod_5 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := by
  sorry

end remainder_of_product_mod_5_l1596_159673


namespace find_fraction_l1596_159623

-- Variables and Definitions
variables (x : ℚ)

-- Conditions
def condition1 := (2 / 3) / x = (3 / 5) / (7 / 15)

-- Theorem to prove the certain fraction
theorem find_fraction (h : condition1 x) : x = 14 / 27 :=
by sorry

end find_fraction_l1596_159623


namespace largest_element_lg11_l1596_159666

variable (x y : ℝ)
variable (A : Set ℝ)  (B : Set ℝ)

-- Conditions
def condition1 : A = Set.insert (Real.log x) (Set.insert (Real.log y) (Set.insert (Real.log (x + y / x)) ∅)) := sorry
def condition2 : B = Set.insert 0 (Set.insert 1 ∅) := sorry
def condition3 : B ⊆ A := sorry

-- Statement
theorem largest_element_lg11 (x y : ℝ)

  (Aeq : A = Set.insert (Real.log x) (Set.insert (Real.log y) (Set.insert (Real.log (x + y / x)) ∅)))
  (Beq : B = Set.insert 0 (Set.insert 1 ∅))
  (subset : B ⊆ A) :
  ∃ M ∈ A, ∀ a ∈ A, a ≤ M ∧ M = Real.log 11 :=
sorry

end largest_element_lg11_l1596_159666


namespace soccer_league_points_l1596_159658

structure Team :=
  (name : String)
  (regular_wins : ℕ)
  (losses : ℕ)
  (draws : ℕ)
  (bonus_wins : ℕ)

def total_points (t : Team) : ℕ :=
  3 * t.regular_wins + t.draws + 2 * t.bonus_wins

def Team_Soccer_Stars : Team :=
  { name := "Team Soccer Stars", regular_wins := 18, losses := 5, draws := 7, bonus_wins := 6 }

def Lightning_Strikers : Team :=
  { name := "Lightning Strikers", regular_wins := 15, losses := 8, draws := 7, bonus_wins := 5 }

def Goal_Grabbers : Team :=
  { name := "Goal Grabbers", regular_wins := 21, losses := 5, draws := 4, bonus_wins := 4 }

def Clever_Kickers : Team :=
  { name := "Clever Kickers", regular_wins := 11, losses := 10, draws := 9, bonus_wins := 2 }

theorem soccer_league_points :
  total_points Team_Soccer_Stars = 73 ∧
  total_points Lightning_Strikers = 62 ∧
  total_points Goal_Grabbers = 75 ∧
  total_points Clever_Kickers = 46 ∧
  [Goal_Grabbers, Team_Soccer_Stars, Lightning_Strikers, Clever_Kickers].map total_points =
  [75, 73, 62, 46] := 
by
  sorry

end soccer_league_points_l1596_159658


namespace value_of_s_in_base_b_l1596_159630

noncomputable def b : ℕ :=
  10

def fourteen_in_b (b : ℕ) : ℕ :=
  b + 4

def seventeen_in_b (b : ℕ) : ℕ :=
  b + 7

def eighteen_in_b (b : ℕ) : ℕ :=
  b + 8

def five_thousand_four_and_four_in_b (b : ℕ) : ℕ :=
  5 * b ^ 3 + 4 * b ^ 2 + 4

def product_in_base_b_equals (b : ℕ) : Prop :=
  (fourteen_in_b b) * (seventeen_in_b b) * (eighteen_in_b b) = five_thousand_four_and_four_in_b b

def s_in_base_b (b : ℕ) : ℕ :=
  fourteen_in_b b + seventeen_in_b b + eighteen_in_b b

theorem value_of_s_in_base_b (b : ℕ) (h : product_in_base_b_equals b) : s_in_base_b b = 49 := by
  sorry

end value_of_s_in_base_b_l1596_159630


namespace range_of_a_l1596_159652

theorem range_of_a
  (a : ℝ)
  (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 ≥ a)
  (h2 : ∃ x0 : ℝ, x0^2 + 2*a*x0 + 2 - a = 0) :
  a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l1596_159652


namespace expected_value_winnings_l1596_159646

def probability_heads : ℚ := 2 / 5
def probability_tails : ℚ := 3 / 5
def winnings_heads : ℚ := 4
def loss_tails : ℚ := -3

theorem expected_value_winnings : 
  (probability_heads * winnings_heads + probability_tails * loss_tails) = -1 / 5 := 
by
  -- calculation steps and proof would go here
  sorry

end expected_value_winnings_l1596_159646


namespace order_of_reading_amounts_l1596_159644

variable (a b c d : ℝ)

theorem order_of_reading_amounts (h1 : a + c = b + d) (h2 : a + b > c + d) (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c :=
by
  sorry

end order_of_reading_amounts_l1596_159644


namespace total_surfers_is_60_l1596_159613

-- Define the number of surfers in Santa Monica beach
def surfers_santa_monica : ℕ := 20

-- Define the number of surfers in Malibu beach as twice the number of surfers in Santa Monica beach
def surfers_malibu : ℕ := 2 * surfers_santa_monica

-- Define the total number of surfers on both beaches
def total_surfers : ℕ := surfers_santa_monica + surfers_malibu

-- Prove that the total number of surfers is 60
theorem total_surfers_is_60 : total_surfers = 60 := by
  sorry

end total_surfers_is_60_l1596_159613
