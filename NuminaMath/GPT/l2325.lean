import Mathlib

namespace NUMINAMATH_GPT_solve_system_of_equations_l2325_232596

theorem solve_system_of_equations (x y : ℤ) (h1 : x + y = 8) (h2 : x - 3 * y = 4) : x = 7 ∧ y = 1 :=
by {
    -- Proof would go here
    sorry
}

end NUMINAMATH_GPT_solve_system_of_equations_l2325_232596


namespace NUMINAMATH_GPT_smallest_a_b_sum_l2325_232534

theorem smallest_a_b_sum :
∀ (a b : ℕ), 
  (5 * a + 6 = 6 * b + 5) ∧ 
  (∀ d : ℕ, d < 10 → d < a) ∧ 
  (∀ d : ℕ, d < 10 → d < b) ∧ 
  (0 < a) ∧ 
  (0 < b) 
  → a + b = 13 :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_b_sum_l2325_232534


namespace NUMINAMATH_GPT_evaluate_expression_l2325_232513

theorem evaluate_expression :
  -1^2008 + 3*(-1)^2007 + 1^2008 - 2*(-1)^2009 = -5 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2325_232513


namespace NUMINAMATH_GPT_range_of_x_l2325_232506

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_x (f_even : ∀ x : ℝ, f x = f (-x))
                   (f_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
                   (f_at_one_third : f (1/3) = 0) :
  {x : ℝ | f (Real.log x / Real.log (1/8)) > 0} = {x : ℝ | (0 < x ∧ x < 1/2) ∨ 2 < x} :=
sorry

end NUMINAMATH_GPT_range_of_x_l2325_232506


namespace NUMINAMATH_GPT_parallelepiped_volume_k_l2325_232508

theorem parallelepiped_volume_k (k : ℝ) : 
    abs (3 * k^2 - 13 * k + 27) = 20 ↔ k = (13 + Real.sqrt 85) / 6 ∨ k = (13 - Real.sqrt 85) / 6 := 
by sorry

end NUMINAMATH_GPT_parallelepiped_volume_k_l2325_232508


namespace NUMINAMATH_GPT_determine_quadrant_l2325_232503

def pointInWhichQuadrant (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On axis or origin"

theorem determine_quadrant : pointInWhichQuadrant (-7) 3 = "Second quadrant" :=
by
  sorry

end NUMINAMATH_GPT_determine_quadrant_l2325_232503


namespace NUMINAMATH_GPT_triangle_area_correct_l2325_232510

def point := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_correct :
  let A : point := (3, -1)
  let B : point := (3, 6)
  let C : point := (8, 6)
  triangle_area A B C = 17.5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_correct_l2325_232510


namespace NUMINAMATH_GPT_find_num_round_balloons_l2325_232529

variable (R : ℕ) -- Number of bags of round balloons that Janeth bought
variable (RoundBalloonsPerBag : ℕ := 20)
variable (LongBalloonsPerBag : ℕ := 30)
variable (BagsLongBalloons : ℕ := 4)
variable (BurstRoundBalloons : ℕ := 5)
variable (BalloonsLeft : ℕ := 215)

def total_long_balloons : ℕ := BagsLongBalloons * LongBalloonsPerBag
def total_balloons : ℕ := R * RoundBalloonsPerBag + total_long_balloons - BurstRoundBalloons

theorem find_num_round_balloons :
  BalloonsLeft = total_balloons → R = 5 := by
  sorry

end NUMINAMATH_GPT_find_num_round_balloons_l2325_232529


namespace NUMINAMATH_GPT_minimum_value_xy_minimum_value_x_plus_2y_l2325_232586

-- (1) Prove that the minimum value of \(xy\) given \(x, y > 0\) and \(\frac{1}{x} + \frac{9}{y} = 1\) is \(36\).
theorem minimum_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 9 / y = 1) : x * y ≥ 36 := 
sorry

-- (2) Prove that the minimum value of \(x + 2y\) given \(x, y > 0\) and \(\frac{1}{x} + \frac{9}{y} = 1\) is \(19 + 6\sqrt{2}\).
theorem minimum_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 9 / y = 1) : x + 2 * y ≥ 19 + 6 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_minimum_value_xy_minimum_value_x_plus_2y_l2325_232586


namespace NUMINAMATH_GPT_number_of_pebbles_l2325_232567

theorem number_of_pebbles (P : ℕ) : 
  (P * (1/4 : ℝ) + 3 * (1/2 : ℝ) + 2 * 2 = 7) → P = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_pebbles_l2325_232567


namespace NUMINAMATH_GPT_difference_shares_l2325_232576

-- Given conditions in the problem
variable (V : ℕ) (F R : ℕ)
variable (hV : V = 1500)
variable (hRatioF : F = 3 * (V / 5))
variable (hRatioR : R = 11 * (V / 5))

-- The statement we need to prove
theorem difference_shares : R - F = 2400 :=
by
  -- Using the conditions to derive the result.
  sorry

end NUMINAMATH_GPT_difference_shares_l2325_232576


namespace NUMINAMATH_GPT_number_of_solutions_l2325_232515

theorem number_of_solutions :
  ∃ (x y z : ℝ), 
    (x = 4036 - 4037 * Real.sign (y - z)) ∧ 
    (y = 4036 - 4037 * Real.sign (z - x)) ∧ 
    (z = 4036 - 4037 * Real.sign (x - y)) :=
sorry

end NUMINAMATH_GPT_number_of_solutions_l2325_232515


namespace NUMINAMATH_GPT_successive_numbers_product_2652_l2325_232580

theorem successive_numbers_product_2652 (n : ℕ) (h : n * (n + 1) = 2652) : n = 51 :=
sorry

end NUMINAMATH_GPT_successive_numbers_product_2652_l2325_232580


namespace NUMINAMATH_GPT_arctan_tan_equiv_l2325_232579

theorem arctan_tan_equiv (h1 : Real.tan (Real.pi / 4 + Real.pi / 12) = 1 / Real.tan (Real.pi / 4 - Real.pi / 3))
  (h2 : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3):
  Real.arctan (Real.tan (5 * Real.pi / 12) - 2 * Real.tan (Real.pi / 6)) = 5 * Real.pi / 12 := 
sorry

end NUMINAMATH_GPT_arctan_tan_equiv_l2325_232579


namespace NUMINAMATH_GPT_there_exists_l_l2325_232556

theorem there_exists_l (m n : ℕ) (h1 : m ≠ 0) (h2 : n ≠ 0) 
  (h3 : ∀ k : ℕ, 0 < k → Nat.gcd (17 * k - 1) m = Nat.gcd (17 * k - 1) n) :
  ∃ l : ℤ, m = (17 : ℕ) ^ l.natAbs * n := 
sorry

end NUMINAMATH_GPT_there_exists_l_l2325_232556


namespace NUMINAMATH_GPT_sector_area_is_nine_l2325_232558

-- Defining the given conditions
def arc_length (r θ : ℝ) : ℝ := r * θ
def sector_area (r θ : ℝ) : ℝ := 0.5 * r^2 * θ

-- Given conditions
variables (r : ℝ) (θ : ℝ)
variable (h1 : arc_length r θ = 6)
variable (h2 : θ = 2)

-- Goal: Prove that the area of the sector is 9
theorem sector_area_is_nine : sector_area r θ = 9 := by
  sorry

end NUMINAMATH_GPT_sector_area_is_nine_l2325_232558


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l2325_232519

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95) 
  (h_avg_ab : (a + b) / 2 = 3.8) 
  (h_avg_cd : (c + d) / 2 = 3.85) :
  ((e + f) / 2) = 4.2 := 
by 
  sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l2325_232519


namespace NUMINAMATH_GPT_who_is_who_l2325_232501

-- Define the types for inhabitants
inductive Inhabitant
| A : Inhabitant
| B : Inhabitant

-- Define the property of being a liar
def is_liar (x : Inhabitant) : Prop := 
  match x with
  | Inhabitant.A  => false -- Initial assumption, to be refined
  | Inhabitant.B  => false -- Initial assumption, to be refined

-- Define the statement made by A
def statement_by_A : Prop :=
  (is_liar Inhabitant.A ∧ ¬ is_liar Inhabitant.B)

-- The main theorem to prove
theorem who_is_who (h : ¬statement_by_A) :
  is_liar Inhabitant.A ∧ is_liar Inhabitant.B :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_who_is_who_l2325_232501


namespace NUMINAMATH_GPT_net_increase_proof_l2325_232504

def initial_cars := 50
def initial_motorcycles := 75
def initial_vans := 25

def car_arrival_rate : ℝ := 70
def car_departure_rate : ℝ := 40
def motorcycle_arrival_rate : ℝ := 120
def motorcycle_departure_rate : ℝ := 60
def van_arrival_rate : ℝ := 30
def van_departure_rate : ℝ := 20

def play_duration : ℝ := 2.5

def net_increase_car : ℝ := play_duration * (car_arrival_rate - car_departure_rate)
def net_increase_motorcycle : ℝ := play_duration * (motorcycle_arrival_rate - motorcycle_departure_rate)
def net_increase_van : ℝ := play_duration * (van_arrival_rate - van_departure_rate)

theorem net_increase_proof :
  net_increase_car = 75 ∧
  net_increase_motorcycle = 150 ∧
  net_increase_van = 25 :=
by
  -- Proof would go here.
  sorry

end NUMINAMATH_GPT_net_increase_proof_l2325_232504


namespace NUMINAMATH_GPT_find_a_l2325_232582

theorem find_a (a : ℝ) (A B : Set ℝ)
    (hA : A = {a^2, a + 1, -3})
    (hB : B = {a - 3, 2 * a - 1, a^2 + 1}) 
    (h : A ∩ B = {-3}) : a = -1 := by
  sorry

end NUMINAMATH_GPT_find_a_l2325_232582


namespace NUMINAMATH_GPT_find_third_root_l2325_232578

noncomputable def P (a b x : ℝ) : ℝ := a * x^3 + (a + 4 * b) * x^2 + (b - 5 * a) * x + (10 - a)

theorem find_third_root (a b : ℝ) (h1 : P a b (-1) = 0) (h2 : P a b 4 = 0) : 
 ∃ c : ℝ, c ≠ -1 ∧ c ≠ 4 ∧ P a b c = 0 ∧ c = 8 / 3 :=
 sorry

end NUMINAMATH_GPT_find_third_root_l2325_232578


namespace NUMINAMATH_GPT_lowest_value_meter_can_record_l2325_232516

theorem lowest_value_meter_can_record (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 6) (h2 : A = 2) : A = 2 :=
by sorry

end NUMINAMATH_GPT_lowest_value_meter_can_record_l2325_232516


namespace NUMINAMATH_GPT_sequence_arithmetic_l2325_232562

theorem sequence_arithmetic (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = 2 * n^2 - 2 * n) →
  (∀ n, a n = S n - S (n - 1)) →
  (∀ n, a n - a (n - 1) = 4) :=
by
  intros hS ha
  sorry

end NUMINAMATH_GPT_sequence_arithmetic_l2325_232562


namespace NUMINAMATH_GPT_remainder_div_9_l2325_232509

theorem remainder_div_9 (x y : ℤ) (h : 9 ∣ (x + 2 * y)) : (2 * (5 * x - 8 * y - 4)) % 9 = -8 ∨ (2 * (5 * x - 8 * y - 4)) % 9 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_9_l2325_232509


namespace NUMINAMATH_GPT_derivative_of_sin_squared_minus_cos_squared_l2325_232545

noncomputable def func (x : ℝ) : ℝ := (Real.sin x)^2 - (Real.cos x)^2

theorem derivative_of_sin_squared_minus_cos_squared (x : ℝ) :
  deriv func x = 2 * Real.sin (2 * x) :=
sorry

end NUMINAMATH_GPT_derivative_of_sin_squared_minus_cos_squared_l2325_232545


namespace NUMINAMATH_GPT_c_sub_a_eq_60_l2325_232520

theorem c_sub_a_eq_60 (a b c : ℝ) 
  (h1 : (a + b) / 2 = 30) 
  (h2 : (b + c) / 2 = 60) : 
  c - a = 60 := 
by 
  sorry

end NUMINAMATH_GPT_c_sub_a_eq_60_l2325_232520


namespace NUMINAMATH_GPT_monkey_slips_2_feet_each_hour_l2325_232560

/-- 
  A monkey climbs a 17 ft tree, hopping 3 ft and slipping back a certain distance each hour.
  The monkey takes 15 hours to reach the top. Prove that the monkey slips back 2 feet each hour.
-/
def monkey_slips_back_distance (s : ℝ) : Prop :=
  ∃ s : ℝ, (14 * (3 - s) + 3 = 17) ∧ s = 2

theorem monkey_slips_2_feet_each_hour : monkey_slips_back_distance 2 := by
  -- Sorry, proof omitted
  sorry

end NUMINAMATH_GPT_monkey_slips_2_feet_each_hour_l2325_232560


namespace NUMINAMATH_GPT_particular_solution_satisfies_l2325_232536

noncomputable def particular_solution (x : ℝ) : ℝ :=
  (1/3) * Real.exp (-4 * x) - (1/3) * Real.exp (2 * x) + (x ^ 2 + 3 * x) * Real.exp (2 * x)

def initial_conditions (f df : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧ df 0 = 1

def differential_equation (f df ddf : ℝ → ℝ) : Prop :=
  ∀ x, ddf x + 2 * df x - 8 * f x = (12 * x + 20) * Real.exp (2 * x)

theorem particular_solution_satisfies :
  ∃ C1 C2 : ℝ, initial_conditions (λ x => C1 * Real.exp (-4 * x) + C2 * Real.exp (2 * x) + particular_solution x) 
              (λ x => -4 * C1 * Real.exp (-4 * x) + 2 * C2 * Real.exp (2 * x) + (2 * x^2 + 8 * x + 3) * Real.exp (2 * x)) ∧ 
              differential_equation (λ x => C1 * Real.exp (-4 * x) + C2 * Real.exp (2 * x) + particular_solution x) 
                                  (λ x => -4 * C1 * Real.exp (-4 * x) + 2 * C2 * Real.exp (2 * x) + (2 * x^2 + 8 * x + 3) * Real.exp (2 * x)) 
                                  (λ x => 16 * C1 * Real.exp (-4 * x) + 4 * C2 * Real.exp (2 * x) + (4 * x^2 + 12 * x + 1) * Real.exp (2 * x)) :=
sorry

end NUMINAMATH_GPT_particular_solution_satisfies_l2325_232536


namespace NUMINAMATH_GPT_paco_ate_more_salty_than_sweet_l2325_232577

-- Define the initial conditions
def sweet_start := 8
def salty_start := 6
def sweet_ate := 20
def salty_ate := 34

-- Define the statement to prove
theorem paco_ate_more_salty_than_sweet : (salty_ate - sweet_ate) = 14 := by
    sorry

end NUMINAMATH_GPT_paco_ate_more_salty_than_sweet_l2325_232577


namespace NUMINAMATH_GPT_weekly_goal_cans_l2325_232511

theorem weekly_goal_cans (c₁ c₂ c₃ c₄ c₅ : ℕ) (h₁ : c₁ = 20) (h₂ : c₂ = c₁ + 5) (h₃ : c₃ = c₂ + 5) 
  (h₄ : c₄ = c₃ + 5) (h₅ : c₅ = c₄ + 5) : 
  c₁ + c₂ + c₃ + c₄ + c₅ = 150 :=
by
  sorry

end NUMINAMATH_GPT_weekly_goal_cans_l2325_232511


namespace NUMINAMATH_GPT_range_of_a_l2325_232502

variable (a : ℝ) (x : ℝ) (x₀ : ℝ)

def p (a : ℝ) : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ (x₀ : ℝ), ∃ (x : ℝ), x + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2325_232502


namespace NUMINAMATH_GPT_work_increase_percentage_l2325_232547

theorem work_increase_percentage (p : ℕ) (hp : p > 0) : 
  let absent_fraction := 1 / 6
  let work_per_person_original := 1 / p
  let present_people := p - p * absent_fraction
  let work_per_person_new := 1 / present_people
  let work_increase := work_per_person_new - work_per_person_original
  let percentage_increase := (work_increase / work_per_person_original) * 100
  percentage_increase = 20 :=
by
  sorry

end NUMINAMATH_GPT_work_increase_percentage_l2325_232547


namespace NUMINAMATH_GPT_tan_alpha_value_l2325_232583

open Real

variable (α : ℝ)

/- Conditions -/
def alpha_interval : Prop := (0 < α) ∧ (α < π)
def sine_cosine_sum : Prop := sin α + cos α = -7 / 13

/- Statement -/
theorem tan_alpha_value 
  (h1 : alpha_interval α)
  (h2 : sine_cosine_sum α) : 
  tan α = -5 / 12 :=
sorry

end NUMINAMATH_GPT_tan_alpha_value_l2325_232583


namespace NUMINAMATH_GPT_solve_x_l2325_232585

theorem solve_x :
  (2 / 3 - 1 / 4) = 1 / (12 / 5) :=
by
  sorry

end NUMINAMATH_GPT_solve_x_l2325_232585


namespace NUMINAMATH_GPT_f_expression_f_odd_l2325_232588

noncomputable def f (x : ℝ) (a b : ℝ) := (2^x + b) / (2^x + a)

theorem f_expression :
  ∃ a b, f 1 a b = 1 / 3 ∧ f 0 a b = 0 ∧ (∀ x, f x a b = (2^x - 1) / (2^x + 1)) :=
by
  sorry

theorem f_odd :
  ∀ x, f x 1 (-1) = (2^x - 1) / (2^x + 1) ∧ f (-x) 1 (-1) = -f x 1 (-1) :=
by
  sorry

end NUMINAMATH_GPT_f_expression_f_odd_l2325_232588


namespace NUMINAMATH_GPT_find_numbers_with_lcm_gcd_l2325_232550

theorem find_numbers_with_lcm_gcd :
  ∃ a b : ℕ, lcm a b = 90 ∧ gcd a b = 6 ∧ ((a = 18 ∧ b = 30) ∨ (a = 30 ∧ b = 18)) :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_with_lcm_gcd_l2325_232550


namespace NUMINAMATH_GPT_buses_passed_on_highway_l2325_232563

/-- Problem statement:
     Buses from Dallas to Austin leave every hour on the hour.
     Buses from Austin to Dallas leave every two hours, starting at 7:00 AM.
     The trip from one city to the other takes 6 hours.
     Assuming the buses travel on the same highway,
     how many Dallas-bound buses does an Austin-bound bus pass on the highway?
-/
theorem buses_passed_on_highway :
  ∀ (t_depart_A2D : ℕ) (trip_time : ℕ) (buses_departures_D2A : ℕ → ℕ),
  (∀ n, buses_departures_D2A n = n) →
  trip_time = 6 →
  ∃ n, t_depart_A2D = 7 ∧ 
    (∀ t, t_depart_A2D ≤ t ∧ t < t_depart_A2D + trip_time →
      ∃ m, m + 1 = t ∧ buses_departures_D2A (m - 6) ≤ t ∧ t < buses_departures_D2A (m - 6) + 6) ↔ n + 1 = 7 := 
sorry

end NUMINAMATH_GPT_buses_passed_on_highway_l2325_232563


namespace NUMINAMATH_GPT_a8_eq_128_l2325_232518

-- Definitions of conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions
axiom a2_eq_2 : a 2 = 2
axiom a3_mul_a4_eq_32 : a 3 * a 4 = 32
axiom is_geometric : is_geometric_sequence a q

-- Statement to prove
theorem a8_eq_128 : a 8 = 128 :=
sorry

end NUMINAMATH_GPT_a8_eq_128_l2325_232518


namespace NUMINAMATH_GPT_find_number_l2325_232507

theorem find_number (N : ℝ) (h : (5/4 : ℝ) * N = (4/5 : ℝ) * N + 27) : N = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2325_232507


namespace NUMINAMATH_GPT_volume_of_prism_l2325_232552

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 48) (h3 : b * c = 72) : a * b * c = 168 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l2325_232552


namespace NUMINAMATH_GPT_line_passes_through_point_l2325_232589

theorem line_passes_through_point (k : ℝ) :
  (1 + 4 * k) * 2 - (2 - 3 * k) * 2 + 2 - 14 * k = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_point_l2325_232589


namespace NUMINAMATH_GPT_subtract_correctly_l2325_232548

theorem subtract_correctly (x : ℕ) (h : x + 35 = 77) : x - 35 = 7 :=
sorry

end NUMINAMATH_GPT_subtract_correctly_l2325_232548


namespace NUMINAMATH_GPT_vector_subtraction_l2325_232594

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def vec_smul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

def vec_a : ℝ × ℝ := (3, 5)
def vec_b : ℝ × ℝ := (-2, 1)

theorem vector_subtraction : vec_sub vec_a (vec_smul 2 vec_b) = (7, 3) :=
by
  sorry

end NUMINAMATH_GPT_vector_subtraction_l2325_232594


namespace NUMINAMATH_GPT_distance_inequality_solution_l2325_232549

theorem distance_inequality_solution (x : ℝ) (h : |x| > |x + 1|) : x < -1 / 2 :=
sorry

end NUMINAMATH_GPT_distance_inequality_solution_l2325_232549


namespace NUMINAMATH_GPT_maximum_value_of_2x_plus_y_l2325_232557

noncomputable def max_value_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) : ℝ :=
  (2 * x + y)

theorem maximum_value_of_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 1) :
  max_value_2x_plus_y x y h ≤ (2 * Real.sqrt 10) / 5 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_2x_plus_y_l2325_232557


namespace NUMINAMATH_GPT_tv_price_increase_percentage_l2325_232528

theorem tv_price_increase_percentage (P Q : ℝ) (x : ℝ) :
  (P * (1 + x / 100) * Q * 0.8 = P * Q * 1.28) → x = 60 :=
by sorry

end NUMINAMATH_GPT_tv_price_increase_percentage_l2325_232528


namespace NUMINAMATH_GPT_intersection_eq_l2325_232523

def setA : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def setB : Set ℝ := {x : ℝ | x > 2 ∨ x < -1}

theorem intersection_eq : (setA ∩ setB) = {x : ℝ | 2 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l2325_232523


namespace NUMINAMATH_GPT_find_base_b_l2325_232568

theorem find_base_b : ∃ b : ℕ, b > 4 ∧ (b + 2)^2 = b^2 + 4 * b + 4 ∧ b = 5 := 
sorry

end NUMINAMATH_GPT_find_base_b_l2325_232568


namespace NUMINAMATH_GPT_balloons_left_after_distribution_l2325_232554

theorem balloons_left_after_distribution :
  (22 + 40 + 70 + 90) % 10 = 2 := by
  sorry

end NUMINAMATH_GPT_balloons_left_after_distribution_l2325_232554


namespace NUMINAMATH_GPT_number_of_buses_proof_l2325_232566

-- Define the conditions
def columns_per_bus : ℕ := 4
def rows_per_bus : ℕ := 10
def total_students : ℕ := 240
def seats_per_bus (c : ℕ) (r : ℕ) : ℕ := c * r
def number_of_buses (total : ℕ) (seats : ℕ) : ℕ := total / seats

-- State the theorem we want to prove
theorem number_of_buses_proof :
  number_of_buses total_students (seats_per_bus columns_per_bus rows_per_bus) = 6 := 
sorry

end NUMINAMATH_GPT_number_of_buses_proof_l2325_232566


namespace NUMINAMATH_GPT_shirt_cost_l2325_232531

theorem shirt_cost (J S : ℕ) 
  (h₁ : 3 * J + 2 * S = 69) 
  (h₂ : 2 * J + 3 * S = 61) :
  S = 9 :=
by 
  sorry

end NUMINAMATH_GPT_shirt_cost_l2325_232531


namespace NUMINAMATH_GPT_smaller_angle_at_7_15_l2325_232551

theorem smaller_angle_at_7_15 (h_angle : ℝ) (m_angle : ℝ) : 
  h_angle = 210 + 0.5 * 15 →
  m_angle = 90 →
  min (abs (h_angle - m_angle)) (360 - abs (h_angle - m_angle)) = 127.5 :=
  by
    intros h_eq m_eq
    rw [h_eq, m_eq]
    sorry

end NUMINAMATH_GPT_smaller_angle_at_7_15_l2325_232551


namespace NUMINAMATH_GPT_sector_max_area_l2325_232537

theorem sector_max_area (P : ℝ) (R l S : ℝ) :
  (P > 0) → (2 * R + l = P) → (S = 1/2 * R * l) →
  (R = P / 4) ∧ (S = P^2 / 16) :=
by
  sorry

end NUMINAMATH_GPT_sector_max_area_l2325_232537


namespace NUMINAMATH_GPT_min_red_chips_l2325_232599

theorem min_red_chips (w b r : ℕ) 
  (h1 : b ≥ (1 / 3) * w)
  (h2 : b ≤ (1 / 4) * r)
  (h3 : w + b ≥ 70) : r ≥ 72 :=
by
  sorry

end NUMINAMATH_GPT_min_red_chips_l2325_232599


namespace NUMINAMATH_GPT_num_pairs_with_math_book_l2325_232521

theorem num_pairs_with_math_book (books : Finset String) (h : books = {"Chinese", "Mathematics", "English", "Biology", "History"}):
  (∃ pairs : Finset (Finset String), pairs.card = 4 ∧ ∀ pair ∈ pairs, "Mathematics" ∈ pair) :=
by
  sorry

end NUMINAMATH_GPT_num_pairs_with_math_book_l2325_232521


namespace NUMINAMATH_GPT_fraction_eq_zero_has_solution_l2325_232526

theorem fraction_eq_zero_has_solution :
  ∀ (x : ℝ), x^2 - x - 2 = 0 ∧ x + 1 ≠ 0 → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_eq_zero_has_solution_l2325_232526


namespace NUMINAMATH_GPT_roots_mul_shift_eq_neg_2018_l2325_232597

theorem roots_mul_shift_eq_neg_2018 {a b : ℝ}
  (h1 : a + b = -1)
  (h2 : a * b = -2020) :
  (a - 1) * (b - 1) = -2018 :=
sorry

end NUMINAMATH_GPT_roots_mul_shift_eq_neg_2018_l2325_232597


namespace NUMINAMATH_GPT_positive_difference_between_loans_l2325_232533

noncomputable def loan_amount : ℝ := 12000

noncomputable def option1_interest_rate : ℝ := 0.08
noncomputable def option1_years_1 : ℕ := 3
noncomputable def option1_years_2 : ℕ := 9

noncomputable def option2_interest_rate : ℝ := 0.09
noncomputable def option2_years : ℕ := 12

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate)^years

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal + principal * rate * years

noncomputable def payment_at_year_3 : ℝ :=
  compound_interest loan_amount option1_interest_rate option1_years_1 / 3

noncomputable def remaining_balance_after_3_years : ℝ :=
  compound_interest loan_amount option1_interest_rate option1_years_1 - payment_at_year_3

noncomputable def total_payment_option1 : ℝ :=
  payment_at_year_3 + compound_interest remaining_balance_after_3_years option1_interest_rate option1_years_2

noncomputable def total_payment_option2 : ℝ :=
  simple_interest loan_amount option2_interest_rate option2_years

noncomputable def positive_difference : ℝ :=
  abs (total_payment_option1 - total_payment_option2)

theorem positive_difference_between_loans : positive_difference = 1731 := by
  sorry

end NUMINAMATH_GPT_positive_difference_between_loans_l2325_232533


namespace NUMINAMATH_GPT_function_defined_for_all_reals_l2325_232592

theorem function_defined_for_all_reals (m : ℝ) :
  (∀ x : ℝ, 7 * x ^ 2 + m - 6 ≠ 0) → m > 6 :=
by
  sorry

end NUMINAMATH_GPT_function_defined_for_all_reals_l2325_232592


namespace NUMINAMATH_GPT_set_star_result_l2325_232542

-- Define the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Define the operation ∗ between sets A and B
def set_star (A B : Set ℕ) : Set ℕ := {x | ∃ x1 ∈ A, ∃ x2 ∈ B, x = x1 + x2}

-- Rewrite the main theorem to be proven
theorem set_star_result : set_star A B = {2, 3, 4, 5} :=
  sorry

end NUMINAMATH_GPT_set_star_result_l2325_232542


namespace NUMINAMATH_GPT_library_table_count_l2325_232524

def base6_to_base10 (n : Nat) : Nat :=
  let d0 := (n % 10)
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 * 36 + d1 * 6 + d0 

theorem library_table_count (chairs people_per_table : Nat) (h1 : chairs = 231) (h2 : people_per_table = 3) :
    Nat.ceil ((base6_to_base10 chairs) / people_per_table) = 31 :=
by
  sorry

end NUMINAMATH_GPT_library_table_count_l2325_232524


namespace NUMINAMATH_GPT_claire_hours_cleaning_l2325_232591

-- Definitions of given conditions
def total_hours_in_day : ℕ := 24
def hours_sleeping : ℕ := 8
def hours_cooking : ℕ := 2
def hours_crafting : ℕ := 5
def total_working_hours : ℕ := total_hours_in_day - hours_sleeping

-- Definition of the question
def hours_cleaning := total_working_hours - (hours_cooking + hours_crafting + hours_crafting)

-- The proof goal
theorem claire_hours_cleaning : hours_cleaning = 4 := by
  sorry

end NUMINAMATH_GPT_claire_hours_cleaning_l2325_232591


namespace NUMINAMATH_GPT_no_such_integers_x_y_l2325_232584

theorem no_such_integers_x_y (x y : ℤ) : x^2 + 1974 ≠ y^2 := by
  sorry

end NUMINAMATH_GPT_no_such_integers_x_y_l2325_232584


namespace NUMINAMATH_GPT_total_money_spent_l2325_232517

def time_in_minutes_at_arcade : ℕ := 3 * 60
def cost_per_interval : ℕ := 50 -- in cents
def interval_duration : ℕ := 6 -- in minutes
def total_intervals : ℕ := time_in_minutes_at_arcade / interval_duration

theorem total_money_spent :
  ((total_intervals * cost_per_interval) = 1500) := 
by
  sorry

end NUMINAMATH_GPT_total_money_spent_l2325_232517


namespace NUMINAMATH_GPT_tan_arithmetic_geometric_l2325_232500

noncomputable def a_seq : ℕ → ℝ := sorry -- Define a_n as an arithmetic sequence (details abstracted)
noncomputable def b_seq : ℕ → ℝ := sorry -- Define b_n as a geometric sequence (details abstracted)

axiom a_seq_is_arithmetic : ∀ n m : ℕ, a_seq (n + 1) - a_seq n = a_seq (m + 1) - a_seq m
axiom b_seq_is_geometric : ∀ n : ℕ, ∃ r : ℝ, b_seq (n + 1) = b_seq n * r
axiom a_seq_sum : a_seq 2017 + a_seq 2018 = Real.pi
axiom b_seq_square : b_seq 20 ^ 2 = 4

theorem tan_arithmetic_geometric : 
  (Real.tan ((a_seq 2 + a_seq 4033) / (b_seq 1 * b_seq 39)) = 1) :=
sorry

end NUMINAMATH_GPT_tan_arithmetic_geometric_l2325_232500


namespace NUMINAMATH_GPT_probability_of_condition1_before_condition2_l2325_232539

-- Definitions for conditions
def condition1 (draw_counts : List ℕ) : Prop :=
  ∃ count ∈ draw_counts, count ≥ 3

def condition2 (draw_counts : List ℕ) : Prop :=
  ∀ count ∈ draw_counts, count ≥ 1

-- Probability function
def probability_condition1_before_condition2 : ℚ :=
  13 / 27

-- The proof statement
theorem probability_of_condition1_before_condition2 :
  (∃ draw_counts : List ℕ, (condition1 draw_counts) ∧  ¬(condition2 draw_counts)) →
  probability_condition1_before_condition2 = 13 / 27 :=
sorry

end NUMINAMATH_GPT_probability_of_condition1_before_condition2_l2325_232539


namespace NUMINAMATH_GPT_calc_g_g_neg3_l2325_232569

def g (x : ℚ) : ℚ :=
x⁻¹ + x⁻¹ / (2 + x⁻¹)

theorem calc_g_g_neg3 : g (g (-3)) = -135 / 8 := 
by
  sorry

end NUMINAMATH_GPT_calc_g_g_neg3_l2325_232569


namespace NUMINAMATH_GPT_coefficient_a_for_factor_l2325_232561

noncomputable def P (a : ℚ) (x : ℚ) : ℚ := x^3 + 2 * x^2 + a * x + 20

theorem coefficient_a_for_factor (a : ℚ) :
  (∀ x : ℚ, (x - 3) ∣ P a x) → a = -65/3 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_a_for_factor_l2325_232561


namespace NUMINAMATH_GPT_problem_statement_l2325_232587

noncomputable def ellipse_equation (t : ℝ) (ht : t > 0) : String :=
  if h : t = 2 then "x^2/9 + y^2/2 = 1"
  else "invalid equation"

theorem problem_statement (m : ℝ) (t : ℝ) (ht : t > 0) (ha : t = 2) 
  (A E F B : ℝ × ℝ) (hA : A = (-3, 0)) (hB : B = (1, 0))
  (hl : ∀ x y, x = m * y + 1) (area : ℝ) (har : area = 16/3) :
  ((ellipse_equation t ht) = "x^2/9 + y^2/2 = 1") ∧
  (∃ M N : ℝ × ℝ, 
    (M.1 = 3 ∧ N.1 = 3) ∧
    ((M.1 - B.1) * (N.1 - B.1) + (M.2 - B.2) * (N.2 - B.2) = 0)) := 
sorry

end NUMINAMATH_GPT_problem_statement_l2325_232587


namespace NUMINAMATH_GPT_sunset_time_correct_l2325_232565

theorem sunset_time_correct : 
  let sunrise := (6 * 60 + 43)       -- Sunrise time in minutes (6:43 AM)
  let daylight := (11 * 60 + 56)     -- Length of daylight in minutes (11:56)
  let sunset := (sunrise + daylight) % (24 * 60) -- Calculate sunset time considering 24-hour cycle
  let sunset_hour := sunset / 60     -- Convert sunset time back into hours
  let sunset_minute := sunset % 60   -- Calculate remaining minutes
  (sunset_hour - 12, sunset_minute) = (6, 39)    -- Convert to 12-hour format and check against 6:39 PM
:= by
  sorry

end NUMINAMATH_GPT_sunset_time_correct_l2325_232565


namespace NUMINAMATH_GPT_pencils_in_each_box_l2325_232564

theorem pencils_in_each_box (n : ℕ) (h : 10 * n - 10 = 40) : n = 5 := by
  sorry

end NUMINAMATH_GPT_pencils_in_each_box_l2325_232564


namespace NUMINAMATH_GPT_part_I_part_II_l2325_232593

def S_n (n : ℕ) : ℕ := sorry
def a_n (n : ℕ) : ℕ := sorry

theorem part_I (n : ℕ) (h1 : 2 * S_n n = 3^n + 3) :
  a_n n = if n = 1 then 3 else 3^(n-1) :=
sorry

theorem part_II (n : ℕ) (h1 : a_n 1 = 1) (h2 : ∀ n : ℕ, a_n (n + 1) - a_n n = 2^n) :
  S_n n = 2^(n + 1) - n - 2 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l2325_232593


namespace NUMINAMATH_GPT_part1_part2_l2325_232532

variables {A B C : ℝ} {a b c : ℝ}

-- conditions of the problem
def condition_1 (a b c : ℝ) (C : ℝ) : Prop :=
  a * Real.cos C + Real.sqrt 3 * Real.sin C - b - c = 0

def condition_2 (C : ℝ) : Prop :=
  0 < C ∧ C < Real.pi

-- Part 1: Proving the value of angle A
theorem part1 (a b c C : ℝ) (h1 : condition_1 a b c C) (h2 : condition_2 C) : 
  A = Real.pi / 3 :=
sorry

-- Part 2: Range of possible values for the perimeter, given c = 3
def is_acute_triangle (A B C : ℝ) : Prop :=
  A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2

theorem part2 (a b A B C : ℝ) (h1 : condition_1 a b 3 C) (h2 : condition_2 C) 
           (h3 : A = Real.pi / 3) (h4 : is_acute_triangle A B C) :
  ∃ p, p ∈ Set.Ioo ((3 * Real.sqrt 3 + 9) / 2) (9 + 3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2325_232532


namespace NUMINAMATH_GPT_power_function_value_at_4_l2325_232546

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

theorem power_function_value_at_4 :
  ∃ a : ℝ, power_function a 2 = (Real.sqrt 2) / 2 → power_function a 4 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_power_function_value_at_4_l2325_232546


namespace NUMINAMATH_GPT_least_n_div_mod_l2325_232514

theorem least_n_div_mod (n : ℕ) (h_pos : n > 1) (h_mod25 : n % 25 = 1) (h_mod7 : n % 7 = 1) : n = 176 :=
by
  sorry

end NUMINAMATH_GPT_least_n_div_mod_l2325_232514


namespace NUMINAMATH_GPT_greatest_x_l2325_232598

theorem greatest_x (x : ℕ) : (x^6 / x^3 ≤ 27) → x ≤ 3 :=
by sorry

end NUMINAMATH_GPT_greatest_x_l2325_232598


namespace NUMINAMATH_GPT_red_pens_count_l2325_232570

theorem red_pens_count (R : ℕ) : 
  (∃ (black_pens blue_pens : ℕ), 
  black_pens = R + 10 ∧ 
  blue_pens = R + 7 ∧ 
  R + black_pens + blue_pens = 41) → 
  R = 8 := by
  sorry

end NUMINAMATH_GPT_red_pens_count_l2325_232570


namespace NUMINAMATH_GPT_soccer_ball_cost_l2325_232544

theorem soccer_ball_cost (x : ℕ) (h : 5 * x + 4 * 65 = 980) : x = 144 :=
by
  sorry

end NUMINAMATH_GPT_soccer_ball_cost_l2325_232544


namespace NUMINAMATH_GPT_factorize_expression_l2325_232559

theorem factorize_expression (x : ℝ) : (x + 3) ^ 2 - (x + 3) = (x + 3) * (x + 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2325_232559


namespace NUMINAMATH_GPT_simplify_sqrt_of_square_l2325_232535

-- The given condition
def x : ℤ := -9

-- The theorem stating the simplified form
theorem simplify_sqrt_of_square : (Real.sqrt ((x : ℝ) ^ 2) = 9) := by    
    sorry

end NUMINAMATH_GPT_simplify_sqrt_of_square_l2325_232535


namespace NUMINAMATH_GPT_university_cost_per_box_l2325_232538

def box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def num_boxes (total_volume box_volume : ℕ) : ℕ :=
  total_volume / box_volume

def cost_per_box (total_cost num_boxes : ℚ) : ℚ :=
  total_cost / num_boxes

theorem university_cost_per_box :
  let length := 20
  let width := 20
  let height := 15
  let total_volume := 3060000
  let total_cost := 459
  let box_vol := box_volume length width height
  let boxes := num_boxes total_volume box_vol
  cost_per_box total_cost boxes = 0.90 :=
by
  sorry

end NUMINAMATH_GPT_university_cost_per_box_l2325_232538


namespace NUMINAMATH_GPT_min_k_valid_l2325_232575

def S : Set ℕ := {1, 2, 3, 4}

def valid_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ b : Fin 4 → ℕ,
    (∀ i : Fin 4, b i ∈ S) ∧ b 3 ≠ 1 →
    ∃ i1 i2 i3 i4 : Fin (k + 1), i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧
      (a i1 = b 0 ∧ a i2 = b 1 ∧ a i3 = b 2 ∧ a i4 = b 3)

def min_k := 11

theorem min_k_valid : ∀ a : ℕ → ℕ,
  valid_sequence a min_k → 
  min_k = 11 :=
sorry

end NUMINAMATH_GPT_min_k_valid_l2325_232575


namespace NUMINAMATH_GPT_cost_of_fencing_per_meter_l2325_232571

theorem cost_of_fencing_per_meter
  (length breadth : ℕ)
  (total_cost : ℝ)
  (h1 : length = breadth + 20)
  (h2 : length = 60)
  (h3 : total_cost = 5300) :
  (total_cost / (2 * length + 2 * breadth)) = 26.5 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_per_meter_l2325_232571


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2325_232530

theorem solution_set_of_inequality :
  {x : ℝ | (x-1)*(2-x) ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2325_232530


namespace NUMINAMATH_GPT_impossible_coins_l2325_232574

theorem impossible_coins : ∀ (p1 p2 : ℝ), 
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 →
  False :=
by 
  sorry

end NUMINAMATH_GPT_impossible_coins_l2325_232574


namespace NUMINAMATH_GPT_intersection_x_value_l2325_232572

theorem intersection_x_value:
  ∃ x y : ℝ, y = 4 * x - 29 ∧ 3 * x + y = 105 ∧ x = 134 / 7 :=
by
  sorry

end NUMINAMATH_GPT_intersection_x_value_l2325_232572


namespace NUMINAMATH_GPT_max_value_xyz_l2325_232543

theorem max_value_xyz (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : 2 * x + 3 * x * y^2 + 2 * z = 36) : 
  x^2 * y^2 * z ≤ 144 :=
sorry

end NUMINAMATH_GPT_max_value_xyz_l2325_232543


namespace NUMINAMATH_GPT_length_of_bridge_l2325_232512

theorem length_of_bridge 
  (lenA : ℝ) (speedA : ℝ) (lenB : ℝ) (speedB : ℝ) (timeA : ℝ) (timeB : ℝ) (startAtSameTime : Prop)
  (h1 : lenA = 120) (h2 : speedA = 12.5) (h3 : lenB = 150) (h4 : speedB = 15.28) 
  (h5 : timeA = 30) (h6 : timeB = 25) : 
  (∃ X : ℝ, X = 757) :=
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l2325_232512


namespace NUMINAMATH_GPT_union_set_when_m_neg3_range_of_m_for_intersection_l2325_232541

def setA (x : ℝ) : Prop := x^2 - x - 12 ≤ 0
def setB (x m : ℝ) : Prop := 2*m - 1 ≤ x ∧ x ≤ m + 1

theorem union_set_when_m_neg3 : 
  (∀ x, setA x ∨ setB x (-3) ↔ -7 ≤ x ∧ x ≤ 4) := 
by sorry

theorem range_of_m_for_intersection :
  (∀ m x, (setA x ∧ setB x m ↔ setB x m) → m ≥ -1) := 
by sorry

end NUMINAMATH_GPT_union_set_when_m_neg3_range_of_m_for_intersection_l2325_232541


namespace NUMINAMATH_GPT_root_expression_value_l2325_232527

theorem root_expression_value 
  (m : ℝ) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2021 = 2024 := 
by 
  sorry

end NUMINAMATH_GPT_root_expression_value_l2325_232527


namespace NUMINAMATH_GPT_correct_expression_l2325_232573

-- Definitions for the problem options.
def optionA (m n : ℕ) : ℕ := 2 * m + n
def optionB (m n : ℕ) : ℕ := m + 2 * n
def optionC (m n : ℕ) : ℕ := 2 * (m + n)
def optionD (m n : ℕ) : ℕ := (m + n) ^ 2

-- Statement for the proof problem.
theorem correct_expression (m n : ℕ) : optionB m n = m + 2 * n :=
by sorry

end NUMINAMATH_GPT_correct_expression_l2325_232573


namespace NUMINAMATH_GPT_tank_empty_time_l2325_232540

theorem tank_empty_time (V : ℝ) (r_inlet r_outlet1 r_outlet2 : ℝ) (I : V = 20 * 12^3)
  (r_inlet_val : r_inlet = 5) (r_outlet1_val : r_outlet1 = 9) 
  (r_outlet2_val : r_outlet2 = 8) : 
  (V / ((r_outlet1 + r_outlet2) - r_inlet) = 2880) :=
by
  sorry

end NUMINAMATH_GPT_tank_empty_time_l2325_232540


namespace NUMINAMATH_GPT_find_13_numbers_l2325_232505

theorem find_13_numbers :
  ∃ (a : Fin 13 → ℕ),
    (∀ i, a i % 21 = 0) ∧
    (∀ i j, i ≠ j → ¬(a i ∣ a j) ∧ ¬(a j ∣ a i)) ∧
    (∀ i j, i ≠ j → (a i ^ 5) % (a j ^ 4) = 0) :=
sorry

end NUMINAMATH_GPT_find_13_numbers_l2325_232505


namespace NUMINAMATH_GPT_binomial_sum_equal_36_l2325_232595

theorem binomial_sum_equal_36 (n : ℕ) (h : n > 0) :
  (n + n * (n - 1) / 2 = 36) → n = 8 :=
by
  sorry

end NUMINAMATH_GPT_binomial_sum_equal_36_l2325_232595


namespace NUMINAMATH_GPT_AlyssaBottleCaps_l2325_232581

def bottleCapsKatherine := 34
def bottleCapsGivenAway (bottleCaps: ℕ) := bottleCaps / 2
def bottleCapsLost (bottleCaps: ℕ) := bottleCaps - 8

theorem AlyssaBottleCaps : bottleCapsLost (bottleCapsGivenAway bottleCapsKatherine) = 9 := 
  by 
  sorry

end NUMINAMATH_GPT_AlyssaBottleCaps_l2325_232581


namespace NUMINAMATH_GPT_max_marks_l2325_232553

theorem max_marks (M : ℝ) (h_pass : 0.33 * M = 165) : M = 500 := 
by
  sorry

end NUMINAMATH_GPT_max_marks_l2325_232553


namespace NUMINAMATH_GPT_maximum_marks_l2325_232555

theorem maximum_marks (passing_percentage : ℝ) (score : ℝ) (shortfall : ℝ) (total_marks : ℝ) : 
  passing_percentage = 30 → 
  score = 212 → 
  shortfall = 16 → 
  total_marks = (score + shortfall) * 100 / passing_percentage → 
  total_marks = 760 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  assumption

end NUMINAMATH_GPT_maximum_marks_l2325_232555


namespace NUMINAMATH_GPT_complex_number_solution_l2325_232525

theorem complex_number_solution (a b : ℝ) (z : ℂ) :
  z = a + b * I →
  (a - 2) ^ 2 + b ^ 2 = 25 →
  (a + 4) ^ 2 + b ^ 2 = 25 →
  a ^ 2 + (b - 2) ^ 2 = 25 →
  z = -1 - 4 * I :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l2325_232525


namespace NUMINAMATH_GPT_unique_t_digit_l2325_232522

theorem unique_t_digit (t : ℕ) (ht : t < 100) (ht2 : 10 ≤ t) (h : 13 * t ≡ 42 [MOD 100]) : t = 34 := 
by
-- Proof is omitted
sorry

end NUMINAMATH_GPT_unique_t_digit_l2325_232522


namespace NUMINAMATH_GPT_find_k_l2325_232590

theorem find_k (x y k : ℝ) (h1 : 2 * x + y = 4 * k) (h2 : x - y = k) (h3 : x + 2 * y = 12) : k = 4 :=
sorry

end NUMINAMATH_GPT_find_k_l2325_232590
