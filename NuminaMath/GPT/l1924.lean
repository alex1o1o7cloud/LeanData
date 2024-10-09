import Mathlib

namespace distance_between_stations_l1924_192480

theorem distance_between_stations (x y t : ℝ) 
(start_same_hour : t > 0)
(speed_slow_train : ∀ t, x = 16 * t)
(speed_fast_train : ∀ t, y = 21 * t)
(distance_difference : y = x + 60) : 
  x + y = 444 := 
sorry

end distance_between_stations_l1924_192480


namespace find_n_l1924_192493

   theorem find_n (n : ℕ) : 
     (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 3 * y + z = n) → 
     (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 3 * y + z = n) → 
     (n = 34 ∨ n = 37) :=
   by
     intros
     sorry
   
end find_n_l1924_192493


namespace slab_cost_l1924_192429

-- Define the conditions
def cubes_per_stick : ℕ := 4
def cubes_per_slab : ℕ := 80
def total_kabob_cost : ℕ := 50
def kabob_sticks_made : ℕ := 40
def total_cubes_needed := kabob_sticks_made * cubes_per_stick
def slabs_needed := total_cubes_needed / cubes_per_slab

-- Final proof problem statement in Lean 4
theorem slab_cost : (total_kabob_cost / slabs_needed) = 25 := by
  sorry

end slab_cost_l1924_192429


namespace count_paths_l1924_192421

-- Define the lattice points and paths
def isLatticePoint (P : ℤ × ℤ) : Prop := true
def isLatticePath (P : ℕ → ℤ × ℤ) (n : ℕ) : Prop :=
  (∀ i, 0 < i → i ≤ n → abs ((P i).1 - (P (i - 1)).1) + abs ((P i).2 - (P (i - 1)).2) = 1)

-- Define F(n) with the given constraints
def numberOfPaths (n : ℕ) : ℕ :=
  -- Placeholder for the actual complex counting logic, which is not detailed here
  sorry

-- Identify F(n) from the initial conditions and the correct result
theorem count_paths (n : ℕ) :
  numberOfPaths n = Nat.choose (2 * n) n :=
sorry

end count_paths_l1924_192421


namespace complete_the_square_d_l1924_192419

theorem complete_the_square_d (x : ℝ) (h : x^2 + 6 * x + 5 = 0) : ∃ d : ℝ, (x + 3)^2 = d ∧ d = 4 :=
by
  sorry

end complete_the_square_d_l1924_192419


namespace rate_of_decrease_l1924_192447

theorem rate_of_decrease (x : ℝ) (h : 400 * (1 - x) ^ 2 = 361) : x = 0.05 :=
by {
  sorry -- The proof is omitted as requested.
}

end rate_of_decrease_l1924_192447


namespace find_x_l1924_192498

theorem find_x (x : ℕ) (h : 5 * x + 4 * x + x + 2 * x = 360) : x = 30 :=
by
  sorry

end find_x_l1924_192498


namespace tank_capacity_l1924_192402

theorem tank_capacity (V : ℝ) (initial_fraction final_fraction : ℝ) (added_water : ℝ)
  (h1 : initial_fraction = 1 / 4)
  (h2 : final_fraction = 3 / 4)
  (h3 : added_water = 208)
  (h4 : final_fraction - initial_fraction = 1 / 2)
  (h5 : (1 / 2) * V = added_water) :
  V = 416 :=
by
  -- Given: initial_fraction = 1/4, final_fraction = 3/4, added_water = 208
  -- Difference in fullness: 1/2
  -- Equation for volume: 1/2 * V = 208
  -- Hence, V = 416
  sorry

end tank_capacity_l1924_192402


namespace greatest_number_of_balloons_l1924_192443

-- Let p be the regular price of one balloon, and M be the total amount of money Orvin has
variable (p M : ℝ)

-- Initial condition: Orvin can buy 45 balloons at the regular price.
-- Thus, he has money M = 45 * p
def orvin_has_enough_money : Prop :=
  M = 45 * p

-- Special Sale condition: The first balloon costs p and the second balloon costs p/2,
-- so total cost for 2 balloons = 1.5 * p
def special_sale_condition : Prop :=
  ∀ pairs : ℝ, M / (1.5 * p) = pairs ∧ pairs * 2 = 60

-- Given the initial condition and the special sale condition, prove the greatest 
-- number of balloons Orvin could purchase is 60
theorem greatest_number_of_balloons (p : ℝ) (M : ℝ) (h1 : orvin_has_enough_money p M) (h2 : special_sale_condition p M) : 
∀ N : ℝ, N = 60 :=
sorry

end greatest_number_of_balloons_l1924_192443


namespace isosceles_triangle_perimeter_correct_l1924_192499

-- Definitions based on conditions
def equilateral_triangle_side_length (perimeter : ℕ) : ℕ :=
  perimeter / 3

def isosceles_triangle_perimeter (side1 side2 base : ℕ) : ℕ :=
  side1 + side2 + base

-- Given conditions
def equilateral_triangle_perimeter : ℕ := 45
def equilateral_triangle_side : ℕ := equilateral_triangle_side_length equilateral_triangle_perimeter

-- The side of the equilateral triangle is also a leg of the isosceles triangle
def isosceles_triangle_leg : ℕ := equilateral_triangle_side
def isosceles_triangle_base : ℕ := 10

-- The problem to prove
theorem isosceles_triangle_perimeter_correct : 
  isosceles_triangle_perimeter isosceles_triangle_leg isosceles_triangle_leg isosceles_triangle_base = 40 :=
by
  sorry

end isosceles_triangle_perimeter_correct_l1924_192499


namespace ages_correct_l1924_192496

variables (Son Daughter Wife Man Father : ℕ)

theorem ages_correct :
  (Man = Son + 20) ∧
  (Man = Daughter + 15) ∧
  (Man + 2 = 2 * (Son + 2)) ∧
  (Man + 2 = 3 * (Daughter + 2)) ∧
  (Wife = Man - 5) ∧
  (Wife + 6 = 2 * (Daughter + 6)) ∧
  (Father = Man + 32) →
  (Son = 7 ∧ Daughter = 12 ∧ Wife = 22 ∧ Man = 27 ∧ Father = 59) :=
by
  intros h
  sorry

end ages_correct_l1924_192496


namespace a_4_value_l1924_192494

def seq (n : ℕ) : ℚ :=
  if n = 0 then 0 -- To handle ℕ index starting from 0.
  else if n = 1 then 1
  else seq (n - 1) + 1 / ((n:ℚ) * (n-1))

noncomputable def a_4 : ℚ := seq 4

theorem a_4_value : a_4 = 7 / 4 := 
  by sorry

end a_4_value_l1924_192494


namespace aaron_and_carson_scoops_l1924_192462

def initial_savings (a c : ℕ) : Prop :=
  a = 150 ∧ c = 150

def total_savings (t a c : ℕ) : Prop :=
  t = a + c

def restaurant_expense (r t : ℕ) : Prop :=
  r = 3 * t / 4

def service_charge_inclusive (r sc : ℕ) : Prop :=
  r = sc * 115 / 100

def remaining_money (t r rm : ℕ) : Prop :=
  rm = t - r

def money_left (al cl : ℕ) : Prop :=
  al = 4 ∧ cl = 4

def ice_cream_scoop_cost (s : ℕ) : Prop :=
  s = 4

def total_scoops (rm ml s scoop_total : ℕ) : Prop :=
  scoop_total = (rm - (ml - 4 - 4)) / s

theorem aaron_and_carson_scoops :
  ∃ a c t r sc rm al cl s scoop_total, initial_savings a c ∧
  total_savings t a c ∧
  restaurant_expense r t ∧
  service_charge_inclusive r sc ∧
  remaining_money t r rm ∧
  money_left al cl ∧
  ice_cream_scoop_cost s ∧
  total_scoops rm (al + cl) s scoop_total ∧
  scoop_total = 16 :=
sorry

end aaron_and_carson_scoops_l1924_192462


namespace min_reciprocal_sum_l1924_192461

theorem min_reciprocal_sum (a b x y : ℝ) (h1 : 8 * x - y - 4 ≤ 0) (h2 : x + y + 1 ≥ 0) (h3 : y - 4 * x ≤ 0) 
    (ha : a > 0) (hb : b > 0) (hz : a * x + b * y = 2) : 
    1 / a + 1 / b = 9 / 2 := 
    sorry

end min_reciprocal_sum_l1924_192461


namespace total_seashells_l1924_192446

theorem total_seashells :
  let initial_seashells : ℝ := 6.5
  let more_seashells : ℝ := 4.25
  initial_seashells + more_seashells = 10.75 :=
by
  sorry

end total_seashells_l1924_192446


namespace find_x_in_triangle_l1924_192466

theorem find_x_in_triangle 
  (P Q R S: Type) 
  (PQS_is_straight: PQS) 
  (angle_PQR: ℝ)
  (h1: angle_PQR = 110) 
  (angle_RQS : ℝ)
  (h2: angle_RQS = 70)
  (angle_QRS : ℝ)
  (h3: angle_QRS = 3 * angle_x)
  (angle_QSR : ℝ)
  (h4: angle_QSR = angle_x + 14) 
  (triangle_angles_sum : ∀ (a b c: ℝ), a + b + c = 180) : 
  angle_x = 24 :=
by
  sorry

end find_x_in_triangle_l1924_192466


namespace find_a_b_find_solution_set_l1924_192452

-- Conditions
variable {a b c x : ℝ}

-- Given inequality condition
def given_inequality (x : ℝ) (a b : ℝ) : Prop := a * x^2 + x + b > 0

-- Define the solution set
def solution_set (x : ℝ) (a b : ℝ) : Prop :=
  (x < -2 ∨ x > 1) ↔ given_inequality x a b

-- Part I: Prove values of a and b
theorem find_a_b
  (H : ∀ x, solution_set x a b) :
  a = 1 ∧ b = -2 := by sorry

-- Define the second inequality
def second_inequality (x : ℝ) (c : ℝ) : Prop := x^2 - (c - 2) * x - 2 * c < 0

-- Solution set for the second inequality
def second_solution_set (x : ℝ) (c : ℝ) : Prop :=
  (c = -2 → False) ∧
  (c > -2 → -2 < x ∧ x < c) ∧
  (c < -2 → c < x ∧ x < -2)

-- Part II: Prove the solution set
theorem find_solution_set
  (H : a = 1)
  (H1 : b = -2) :
  ∀ x, second_solution_set x c ↔ second_inequality x c := by sorry

end find_a_b_find_solution_set_l1924_192452


namespace inequality_holds_l1924_192450

theorem inequality_holds (x y : ℝ) : (y - x^2 < abs x) ↔ (y < x^2 + abs x) := by
  sorry

end inequality_holds_l1924_192450


namespace snail_kite_eats_35_snails_l1924_192470

theorem snail_kite_eats_35_snails : 
  let day1 := 3
  let day2 := day1 + 2
  let day3 := day2 + 2
  let day4 := day3 + 2
  let day5 := day4 + 2
  day1 + day2 + day3 + day4 + day5 = 35 := 
by
  sorry

end snail_kite_eats_35_snails_l1924_192470


namespace total_distance_walked_l1924_192471

-- Define the conditions
def walking_rate : ℝ := 4
def time_before_break : ℝ := 2
def time_after_break : ℝ := 0.5

-- Define the required theorem
theorem total_distance_walked : 
  walking_rate * time_before_break + walking_rate * time_after_break = 10 := 
sorry

end total_distance_walked_l1924_192471


namespace total_students_l1924_192433

theorem total_students (total_students_with_brown_eyes total_students_with_black_hair: ℕ)
    (h1: ∀ (total_students : ℕ), (2 * total_students_with_brown_eyes) = 3 * total_students)
    (h2: (2 * total_students_with_black_hair) = total_students_with_brown_eyes)
    (h3: total_students_with_black_hair = 6) : 
    ∃ total_students : ℕ, total_students = 18 :=
by
  sorry

end total_students_l1924_192433


namespace vector_scalar_operations_l1924_192445

-- Define the vectors
def v1 : ℤ × ℤ := (2, -9)
def v2 : ℤ × ℤ := (-1, -6)

-- Define the scalars
def c1 : ℤ := 4
def c2 : ℤ := 3

-- Define the scalar multiplication of vectors
def scale (c : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (c * v.1, c * v.2)

-- Define the vector subtraction
def sub (v w : ℤ × ℤ) : ℤ × ℤ := (v.1 - w.1, v.2 - w.2)

-- State the theorem
theorem vector_scalar_operations :
  sub (scale c1 v1) (scale c2 v2) = (11, -18) :=
by
  sorry

end vector_scalar_operations_l1924_192445


namespace Jonathan_typing_time_l1924_192490

theorem Jonathan_typing_time
  (J : ℝ)
  (HJ : 0 < J)
  (rate_Jonathan : ℝ := 1 / J)
  (rate_Susan : ℝ := 1 / 30)
  (rate_Jack : ℝ := 1 / 24)
  (combined_rate : ℝ := 1 / 10)
  (combined_rate_eq : rate_Jonathan + rate_Susan + rate_Jack = combined_rate)
  : J = 40 :=
sorry

end Jonathan_typing_time_l1924_192490


namespace intersection_complementA_setB_l1924_192484

noncomputable def setA : Set ℝ := { x | abs x > 1 }

noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

noncomputable def complementA : Set ℝ := { x | abs x ≤ 1 }

theorem intersection_complementA_setB : 
  (complementA ∩ setB) = { x | 0 ≤ x ∧ x ≤ 1 } := by
  sorry

end intersection_complementA_setB_l1924_192484


namespace proof_problem_l1924_192420

variables {a b c d e : ℝ}

theorem proof_problem (h1 : a * b^2 * c^3 * d^4 * e^5 < 0) (h2 : b^2 ≥ 0) (h3 : d^4 ≥ 0) :
  a * b^2 * c * d^4 * e < 0 :=
sorry

end proof_problem_l1924_192420


namespace fibonacci_recurrence_l1924_192409

def F : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => F (n + 1) + F n

theorem fibonacci_recurrence (n : ℕ) (h: n ≥ 2) : 
  F n = F (n-1) + F (n-2) := by
 {
 sorry
 }

end fibonacci_recurrence_l1924_192409


namespace max_a_condition_slope_condition_exponential_inequality_l1924_192426

noncomputable def f (x a : ℝ) := Real.exp x - a * (x + 1)
noncomputable def g (x a : ℝ) := f x a + a / Real.exp x

theorem max_a_condition (a : ℝ) (h_pos : a > 0) 
  (h_nonneg : ∀ x : ℝ, f x a ≥ 0) : a ≤ 1 := sorry

theorem slope_condition (a m : ℝ) 
  (ha : a ≤ -1) 
  (h_slope : ∀ x1 x2 : ℝ, x1 ≠ x2 → 
    (g x2 a - g x1 a) / (x2 - x1) > m) : m ≤ 3 := sorry

theorem exponential_inequality (n : ℕ) (hn : n > 0) : 
  (2 * (Real.exp n - 1)) / (Real.exp 1 - 1) ≥ n * (n + 1) := sorry

end max_a_condition_slope_condition_exponential_inequality_l1924_192426


namespace min_value_cx_plus_dy_squared_l1924_192428

theorem min_value_cx_plus_dy_squared
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (∃ (x y : ℝ), a * x^2 + b * y^2 = 1 ∧ ∀ (x y : ℝ), a * x^2 + b * y^2 = 1 → c * x + d * y^2 ≥ -c / a.sqrt) :=
sorry

end min_value_cx_plus_dy_squared_l1924_192428


namespace Tahir_contribution_l1924_192468

theorem Tahir_contribution
  (headphone_cost : ℕ := 200)
  (kenji_yen : ℕ := 15000)
  (exchange_rate : ℕ := 100)
  (kenji_contribution : ℕ := kenji_yen / exchange_rate)
  (tahir_contribution : ℕ := headphone_cost - kenji_contribution) :
  tahir_contribution = 50 := 
  by sorry

end Tahir_contribution_l1924_192468


namespace profit_sharing_l1924_192441

-- Define constants and conditions
def Tom_investment : ℕ := 30000
def Tom_share : ℝ := 0.40

def Jose_investment : ℕ := 45000
def Jose_start_month : ℕ := 2
def Jose_share : ℝ := 0.30

def Sarah_investment : ℕ := 60000
def Sarah_start_month : ℕ := 5
def Sarah_share : ℝ := 0.20

def Ravi_investment : ℕ := 75000
def Ravi_start_month : ℕ := 8
def Ravi_share : ℝ := 0.10

def total_profit : ℕ := 120000

-- Define expected shares
def Tom_expected_share : ℕ := 48000
def Jose_expected_share : ℕ := 36000
def Sarah_expected_share : ℕ := 24000
def Ravi_expected_share : ℕ := 12000

-- Theorem statement
theorem profit_sharing :
  let Tom_contribution := Tom_investment * 12
  let Jose_contribution := Jose_investment * (12 - Jose_start_month)
  let Sarah_contribution := Sarah_investment * (12 - Sarah_start_month)
  let Ravi_contribution := Ravi_investment * (12 - Ravi_start_month)
  Tom_share * total_profit = Tom_expected_share ∧
  Jose_share * total_profit = Jose_expected_share ∧
  Sarah_share * total_profit = Sarah_expected_share ∧
  Ravi_share * total_profit = Ravi_expected_share := by {
    sorry
  }

end profit_sharing_l1924_192441


namespace percent_of_total_l1924_192404

theorem percent_of_total (p n : ℝ) (h1 : p = 35 / 100) (h2 : n = 360) : p * n = 126 := by
  sorry

end percent_of_total_l1924_192404


namespace flight_duration_l1924_192459

theorem flight_duration (takeoff landing : Nat)
  (h m : Nat) (h_pos : 0 < m) (m_lt_60 : m < 60)
  (time_takeoff : takeoff = 9 * 60 + 27)
  (time_landing : landing = 11 * 60 + 56)
  (flight_duration : (landing - takeoff) = h * 60 + m) :
  h + m = 31 :=
sorry

end flight_duration_l1924_192459


namespace trapezoidal_park_no_solution_l1924_192406

theorem trapezoidal_park_no_solution :
  (∃ b1 b2 : ℕ, 2 * 1800 = 40 * (b1 + b2) ∧ (∃ m : ℕ, b1 = 5 * (2 * m + 1)) ∧ (∃ n : ℕ, b2 = 2 * n)) → false :=
by
  sorry

end trapezoidal_park_no_solution_l1924_192406


namespace total_apples_l1924_192456

-- Define the number of apples given to each person
def apples_per_person : ℝ := 15.0

-- Define the number of people
def number_of_people : ℝ := 3.0

-- Goal: Prove that the total number of apples is 45.0
theorem total_apples : apples_per_person * number_of_people = 45.0 := by
  sorry

end total_apples_l1924_192456


namespace slope_of_line_l1924_192425

theorem slope_of_line : 
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 1) ∧ (y1 = 3) ∧ (x2 = 7) ∧ (y2 = -9)
  → (y2 - y1) / (x2 - x1) = -2 := by
  sorry

end slope_of_line_l1924_192425


namespace find_other_number_l1924_192487

theorem find_other_number
  (x y lcm hcf : ℕ)
  (h_lcm : Nat.lcm x y = lcm)
  (h_hcf : Nat.gcd x y = hcf)
  (h_x : x = 462)
  (h_lcm_value : lcm = 2310)
  (h_hcf_value : hcf = 30) :
  y = 150 :=
by
  sorry

end find_other_number_l1924_192487


namespace decreasing_implies_bound_l1924_192434

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  - (1 / 2) * x ^ 2 + b * Real.log x

theorem decreasing_implies_bound (b : ℝ) :
  (∀ x > 2, -x + b / x ≤ 0) → b ≤ 4 :=
  sorry

end decreasing_implies_bound_l1924_192434


namespace side_length_of_base_l1924_192479

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l1924_192479


namespace perimeter_of_triangle_AF2B_l1924_192481

theorem perimeter_of_triangle_AF2B (a : ℝ) (m n : ℝ) (F1 F2 A B : ℝ × ℝ) 
  (h_hyperbola : ∀ x y : ℝ, (x^2 - 4*y^2 = 4) ↔ (x^2 / 4 - y^2 = 1)) 
  (h_mn : m + n = 3) 
  (h_AF1 : dist A F1 = m) 
  (h_BF1 : dist B F1 = n) 
  (h_AF2 : dist A F2 = 4 + m) 
  (h_BF2 : dist B F2 = 4 + n) 
  : dist A F1 + dist A F2 + dist B F2 + dist B F1 = 14 :=
by
  sorry

end perimeter_of_triangle_AF2B_l1924_192481


namespace minimum_value_fraction_l1924_192415

theorem minimum_value_fraction (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 2) : 
  (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 2 → 
    ((1 / (1 + x)) + (1 / (2 + 2 * y)) ≥ 4 / 5)) :=
by sorry

end minimum_value_fraction_l1924_192415


namespace values_of_xyz_l1924_192451

theorem values_of_xyz (x y z : ℝ) (h1 : 2 * x - y + z = 14) (h2 : y = 2) (h3 : x + z = 3 * y + 5) : 
  x = 5 ∧ y = 2 ∧ z = 6 := 
by
  sorry

end values_of_xyz_l1924_192451


namespace percentage_both_correct_l1924_192448

variable (A B : Type) 

noncomputable def percentage_of_test_takers_correct_first : ℝ := 0.85
noncomputable def percentage_of_test_takers_correct_second : ℝ := 0.70
noncomputable def percentage_of_test_takers_neither_correct : ℝ := 0.05

theorem percentage_both_correct :
  percentage_of_test_takers_correct_first + 
  percentage_of_test_takers_correct_second - 
  (1 - percentage_of_test_takers_neither_correct) = 0.60 := by
  sorry

end percentage_both_correct_l1924_192448


namespace sum_series_eq_seven_twelve_l1924_192412

noncomputable def sum_series : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * (n:ℝ)^2 + 2 * (n:ℝ) + 1) / ((n:ℝ) * (n + 1) * (n + 2) * (n + 3)) else 0

theorem sum_series_eq_seven_twelve : sum_series = 7 / 12 :=
by
  sorry

end sum_series_eq_seven_twelve_l1924_192412


namespace farmer_brown_additional_cost_l1924_192432

-- Definitions for the conditions
def originalQuantity : ℕ := 10
def originalPricePerBale : ℕ := 15
def newPricePerBale : ℕ := 18
def newQuantity : ℕ := 2 * originalQuantity

-- Definition for the target equation (additional cost)
def additionalCost : ℕ := (newQuantity * newPricePerBale) - (originalQuantity * originalPricePerBale)

-- Theorem stating the problem voiced in Lean 4
theorem farmer_brown_additional_cost : additionalCost = 210 :=
by {
  sorry
}

end farmer_brown_additional_cost_l1924_192432


namespace length_width_difference_l1924_192440

theorem length_width_difference (L W : ℝ) 
  (h1 : W = 1/2 * L) 
  (h2 : L * W = 578) : L - W = 17 :=
sorry

end length_width_difference_l1924_192440


namespace increasing_function_shape_implies_number_l1924_192437

variable {I : Set ℝ} {f : ℝ → ℝ}

theorem increasing_function_shape_implies_number (h : ∀ (x₁ x₂ : ℝ), x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ < x₂ → f x₁ < f x₂) 
: ∀ (x₁ x₂ : ℝ), x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ < x₂ → f x₁ < f x₂ :=
sorry

end increasing_function_shape_implies_number_l1924_192437


namespace mowing_lawn_time_l1924_192475

theorem mowing_lawn_time (mary_time tom_time tom_solo_work : ℝ) 
  (mary_rate tom_rate : ℝ)
  (combined_rate remaining_lawn total_time : ℝ) :
  mary_time = 3 → 
  tom_time = 6 → 
  tom_solo_work = 3 → 
  mary_rate = 1 / mary_time → 
  tom_rate = 1 / tom_time → 
  combined_rate = mary_rate + tom_rate →
  remaining_lawn = 1 - (tom_solo_work * tom_rate) →
  total_time = tom_solo_work + (remaining_lawn / combined_rate) →
  total_time = 4 :=
by sorry

end mowing_lawn_time_l1924_192475


namespace ratio_h_r_bounds_l1924_192401

theorem ratio_h_r_bounds
  {a b c h r : ℝ}
  (h_right_angle : a^2 + b^2 = c^2)
  (h_area1 : 1/2 * a * b = 1/2 * c * h)
  (h_area2 : 1/2 * (a + b + c) * r = 1/2 * a * b) :
  2 < h / r ∧ h / r ≤ 2.41 :=
by
  sorry

end ratio_h_r_bounds_l1924_192401


namespace consecutive_odd_integers_sum_l1924_192431

theorem consecutive_odd_integers_sum (x : ℤ) (h : x + (x + 4) = 138) :
  x + (x + 2) + (x + 4) = 207 :=
sorry

end consecutive_odd_integers_sum_l1924_192431


namespace triangle_angle_bisector_sum_l1924_192472

theorem triangle_angle_bisector_sum (P Q R : ℝ × ℝ)
  (hP : P = (-8, 5)) (hQ : Q = (-15, -19)) (hR : R = (1, -7)) 
  (a b c : ℕ) (h : a + c = 89) 
  (gcd_abc : Int.gcd (Int.gcd a b) c = 1) :
  a + c = 89 :=
by
  sorry

end triangle_angle_bisector_sum_l1924_192472


namespace ratio_of_sums_l1924_192410

noncomputable def sum_upto (n : ℕ) : ℕ := n * (n + 1) / 2

theorem ratio_of_sums (n : ℕ) (S1 S2 : ℕ) 
  (hn_even : n % 2 = 0)
  (hn_pos : 0 < n)
  (h_sum : sum_upto (n^2) = n^2 * (n^2 + 1) / 2)
  (h_S1S2_sum : S1 + S2 = n^2 * (n^2 + 1) / 2)
  (h_ratio : 64 * S1 = 39 * S2) :
  ∃ k : ℕ, n = 103 * k :=
sorry

end ratio_of_sums_l1924_192410


namespace compute_f_at_5_l1924_192474

def f : ℝ → ℝ := sorry

axiom f_property : ∀ x : ℝ, f (10 ^ x) = x

theorem compute_f_at_5 : f 5 = Real.log 5 / Real.log 10 :=
by
  sorry

end compute_f_at_5_l1924_192474


namespace highest_degree_divisibility_l1924_192482

-- Definition of the problem settings
def prime_number := 1991
def number_1 := 1990 ^ (1991 ^ 1002)
def number_2 := 1992 ^ (1501 ^ 1901)
def combined_number := number_1 + number_2

-- Statement of the proof to be formalized
theorem highest_degree_divisibility (k : ℕ) : k = 1001 ∧ prime_number ^ k ∣ combined_number := by
  sorry

end highest_degree_divisibility_l1924_192482


namespace abs_diff_x_y_l1924_192457

variables {x y : ℝ}

noncomputable def floor (z : ℝ) : ℤ := Int.floor z
noncomputable def fract (z : ℝ) : ℝ := z - floor z

theorem abs_diff_x_y 
  (h1 : floor x + fract y = 3.7) 
  (h2 : fract x + floor y = 4.6) : 
  |x - y| = 1.1 :=
by
  sorry

end abs_diff_x_y_l1924_192457


namespace mixed_feed_cost_l1924_192400

/-- Tim and Judy mix two kinds of feed for pedigreed dogs. They made 35 pounds of feed by mixing 
    one kind worth $0.18 per pound with another worth $0.53 per pound. They used 17 pounds of the cheaper kind in the mix.
    We are to prove that the cost per pound of the mixed feed is $0.36 per pound. -/
theorem mixed_feed_cost
  (total_weight : ℝ) (cheaper_cost : ℝ) (expensive_cost : ℝ) (cheaper_weight : ℝ)
  (total_weight_eq : total_weight = 35)
  (cheaper_cost_eq : cheaper_cost = 0.18)
  (expensive_cost_eq : expensive_cost = 0.53)
  (cheaper_weight_eq : cheaper_weight = 17) :
  ((cheaper_weight * cheaper_cost + (total_weight - cheaper_weight) * expensive_cost) / total_weight) = 0.36 :=
by
  sorry

end mixed_feed_cost_l1924_192400


namespace maximum_n_l1924_192442

noncomputable def a1 : ℝ := sorry -- define a1 solving a_5 equations
noncomputable def q : ℝ := sorry -- define q solving a_5 and a_6 + a_7 equations
noncomputable def sn (n : ℕ) : ℝ := (a1 * (1 - q^n)) / (1 - q)  -- S_n of geometric series with a1 and q
noncomputable def pin (n : ℕ) : ℝ := (a1 * (q^((1 + n) * n / 2 - (11 * n) / 2 + 19 / 2)))  -- Pi solely in terms of n, a1, and q

theorem maximum_n (n : ℕ) (h1 : (a1 : ℝ) > 0) (h2 : q > 0) (h3 : q ≠ 1)
(h4 : a1 * q^4 = 1 / 4) (h5 : a1 * q^5 + a1 * q^6 = 3 / 2) :
  ∃ n : ℕ, sn n > pin n ∧ ∀ m : ℕ, m > 13 → sn m ≤ pin m := sorry

end maximum_n_l1924_192442


namespace two_digit_number_reversed_l1924_192467

theorem two_digit_number_reversed :
  ∃ (x y : ℕ), (10 * x + y = 73) ∧ (10 * x + y = 2 * (10 * y + x) - 1) ∧ (x < 10) ∧ (y < 10) := 
by
  sorry

end two_digit_number_reversed_l1924_192467


namespace triangle_is_right_l1924_192403

variable {n : ℕ}

theorem triangle_is_right 
  (h1 : n > 1) 
  (h2 : a = 2 * n) 
  (h3 : b = n^2 - 1) 
  (h4 : c = n^2 + 1)
  : a^2 + b^2 = c^2 := 
by
  -- skipping the proof
  sorry

end triangle_is_right_l1924_192403


namespace banana_nn_together_count_l1924_192427

open Finset

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def arrangements_banana_with_nn_together : ℕ :=
  (factorial 4) / (factorial 3)

theorem banana_nn_together_count : arrangements_banana_with_nn_together = 4 := by
  sorry

end banana_nn_together_count_l1924_192427


namespace carrots_as_potatoes_l1924_192424

variable (G O C P : ℕ)

theorem carrots_as_potatoes :
  G = 8 →
  G = (1 / 3 : ℚ) * O →
  O = 2 * C →
  P = 2 →
  (C / P : ℚ) = 6 :=
by intros hG1 hG2 hO hP; sorry

end carrots_as_potatoes_l1924_192424


namespace sequence_general_formula_l1924_192495

-- Definitions according to conditions in a)
def seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n + 1

def S (n : ℕ) (seq : ℕ → ℕ) : ℕ :=
  n * seq (n + 1) - 3 * n^2 - 4 * n

-- The proof goal
theorem sequence_general_formula (n : ℕ) (h : 0 < n) :
  seq n = 2 * n + 1 :=
by
  sorry

end sequence_general_formula_l1924_192495


namespace sector_max_angle_l1924_192492

variables (r l : ℝ)

theorem sector_max_angle (h : 2 * r + l = 40) : (l / r) = 2 :=
sorry

end sector_max_angle_l1924_192492


namespace no_solutions_exist_l1924_192485

theorem no_solutions_exist : ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 2 :=
by sorry

end no_solutions_exist_l1924_192485


namespace power_function_not_pass_origin_l1924_192458

noncomputable def does_not_pass_through_origin (m : ℝ) : Prop :=
  ∀ x:ℝ, (m^2 - 3 * m + 3) * x^(m^2 - m - 2) ≠ 0

theorem power_function_not_pass_origin (m : ℝ) :
  does_not_pass_through_origin m ↔ (m = 1 ∨ m = 2) :=
sorry

end power_function_not_pass_origin_l1924_192458


namespace kids_at_camp_l1924_192423

theorem kids_at_camp (total_stayed_home : ℕ) (difference : ℕ) (x : ℕ) 
  (h1 : total_stayed_home = 777622) 
  (h2 : difference = 574664) 
  (h3 : total_stayed_home = x + difference) : 
  x = 202958 :=
by
  sorry

end kids_at_camp_l1924_192423


namespace lowest_possible_number_of_students_l1924_192465

theorem lowest_possible_number_of_students : ∃ n : ℕ, (n > 0) ∧ (∃ k1 : ℕ, n = 10 * k1) ∧ (∃ k2 : ℕ, n = 24 * k2) ∧ n = 120 :=
by
  sorry

end lowest_possible_number_of_students_l1924_192465


namespace car_travel_distance_l1924_192418

theorem car_travel_distance :
  ∀ (train_speed : ℝ) (fraction : ℝ) (time_minutes : ℝ) (car_speed : ℝ) (distance : ℝ),
  train_speed = 90 →
  fraction = 5 / 6 →
  time_minutes = 30 →
  car_speed = fraction * train_speed →
  distance = car_speed * (time_minutes / 60) →
  distance = 37.5 :=
by
  intros train_speed fraction time_minutes car_speed distance
  intros h_train_speed h_fraction h_time_minutes h_car_speed h_distance
  sorry

end car_travel_distance_l1924_192418


namespace parabola_equation_l1924_192483

theorem parabola_equation {p : ℝ} (hp : 0 < p)
  (h_cond : ∃ A B : ℝ × ℝ, (A.1^2 = 2 * A.2 * p) ∧ (B.1^2 = 2 * B.2 * p) ∧ (A.2 = A.1 - p / 2) ∧ (B.2 = B.1 - p / 2) ∧ (|A.1 - B.1|^2 + |A.2 - B.2|^2 = 4))
  : y^2 = 2 * x := sorry

end parabola_equation_l1924_192483


namespace complement_U_A_l1924_192439

def U := {x : ℝ | x < 2}
def A := {x : ℝ | x^2 < x}

theorem complement_U_A :
  (U \ A) = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} :=
sorry

end complement_U_A_l1924_192439


namespace an_general_term_sum_bn_l1924_192463

open Nat

variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (T : ℕ → ℕ)

-- Conditions
axiom a3 : a 3 = 3
axiom S6 : S 6 = 21
axiom Sn : ∀ n, S n = n * (a 1 + a n) / 2

-- Define bn based on the given condition for bn = an + 2^n
def bn (n : ℕ) : ℕ := a n + 2^n

-- Define Tn based on the given condition for Tn.
def Tn (n : ℕ) : ℕ := (n * (n + 1)) / 2 + (2^(n + 1) - 2)

-- Prove the general term formula of the arithmetic sequence an
theorem an_general_term (n : ℕ) : a n = n :=
by
  sorry

-- Prove the sum of the first n terms of the sequence bn
theorem sum_bn (n : ℕ) : T n = Tn n :=
by
  sorry

end an_general_term_sum_bn_l1924_192463


namespace sequence_values_l1924_192407

theorem sequence_values (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
    (h_arith : 2 + (a - 2) = a + (b - a)) (h_geom : a * a = b * (9 / b)) : a = 4 ∧ b = 6 :=
by
  -- insert proof here
  sorry

end sequence_values_l1924_192407


namespace rotated_point_l1924_192460

def point := (ℝ × ℝ × ℝ)

def rotate_point (A P : point) (θ : ℝ) : point :=
  -- Function implementing the rotation (the full definition would normally be placed here)
  sorry

def A : point := (1, 1, 1)
def P : point := (1, 1, 0)

theorem rotated_point (θ : ℝ) (hθ : θ = 60) :
  rotate_point A P θ = (1/3, 4/3, 1/3) :=
sorry

end rotated_point_l1924_192460


namespace initial_alcohol_percentage_l1924_192476

theorem initial_alcohol_percentage (P : ℚ) (initial_volume : ℚ) (added_alcohol : ℚ) (added_water : ℚ)
  (final_percentage : ℚ) (final_volume : ℚ) (alcohol_volume_in_initial_solution : ℚ) :
  initial_volume = 40 ∧ 
  added_alcohol = 3.5 ∧ 
  added_water = 6.5 ∧ 
  final_percentage = 0.11 ∧ 
  final_volume = 50 ∧ 
  alcohol_volume_in_initial_solution = (P / 100) * initial_volume ∧ 
  alcohol_volume_in_initial_solution + added_alcohol = final_percentage * final_volume
  → P = 5 :=
by
  sorry

end initial_alcohol_percentage_l1924_192476


namespace algebra_sum_l1924_192438

-- Given conditions
def letterValue (ch : Char) : Int :=
  let pos := ch.toNat - 'a'.toNat + 1
  match pos % 6 with
  | 1 => 1
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | 0 => -2
  | _ => 0  -- This case is actually unreachable.

def wordValue (w : List Char) : Int :=
  w.foldl (fun acc ch => acc + letterValue ch) 0

theorem algebra_sum : wordValue ['a', 'l', 'g', 'e', 'b', 'r', 'a'] = 0 :=
  sorry

end algebra_sum_l1924_192438


namespace profit_last_month_l1924_192489

variable (gas_expenses earnings_per_lawn lawns_mowed extra_income profit : ℤ)

def toms_profit (gas_expenses earnings_per_lawn lawns_mowed extra_income : ℤ) : ℤ :=
  (lawns_mowed * earnings_per_lawn + extra_income) - gas_expenses

theorem profit_last_month :
  toms_profit 17 12 3 10 = 29 :=
by
  rw [toms_profit]
  sorry

end profit_last_month_l1924_192489


namespace monica_total_savings_l1924_192444

noncomputable def weekly_savings (week: ℕ) : ℕ :=
  if week < 6 then 15 + 5 * week
  else if week < 11 then 40 - 5 * (week - 5)
  else weekly_savings (week % 10)

theorem monica_total_savings : 
  let cycle_savings := (15 + 20 + 25 + 30 + 35 + 40) + (40 + 35 + 30 + 25 + 20 + 15) - 40 
  let total_savings := 5 * cycle_savings
  total_savings = 1450 := by
  sorry

end monica_total_savings_l1924_192444


namespace campers_afternoon_l1924_192430

def morning_campers : ℕ := 52
def additional_campers : ℕ := 9
def total_campers_afternoon : ℕ := morning_campers + additional_campers

theorem campers_afternoon : total_campers_afternoon = 61 :=
by
  sorry

end campers_afternoon_l1924_192430


namespace none_of_the_above_option_l1924_192454

-- Define integers m and n
variables (m n: ℕ)

-- Define P and R in terms of m and n
def P : ℕ := 2^m
def R : ℕ := 5^n

-- Define the statement to prove
theorem none_of_the_above_option : ∀ (m n : ℕ), 15^(m + n) ≠ P^(m + n) * R ∧ 15^(m + n) ≠ (3^m * 3^n * 5^m) ∧ 15^(m + n) ≠ (3^m * P^n) ∧ 15^(m + n) ≠ (2^m * 5^n * 5^m) :=
by sorry

end none_of_the_above_option_l1924_192454


namespace anne_ben_charlie_difference_l1924_192449

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def charlie_discount_rate : ℝ := 0.15

def anne_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)
def ben_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)
def charlie_total : ℝ := (original_price * (1 - charlie_discount_rate)) * (1 + sales_tax_rate)

def anne_minus_ben_minus_charlie : ℝ := anne_total - ben_total - charlie_total

theorem anne_ben_charlie_difference : anne_minus_ben_minus_charlie = -12.96 :=
by
  sorry

end anne_ben_charlie_difference_l1924_192449


namespace original_number_l1924_192469

theorem original_number (x : ℝ) (h : 1.50 * x = 165) : x = 110 :=
sorry

end original_number_l1924_192469


namespace general_term_an_l1924_192497

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 2
noncomputable def S_n (n : ℕ) : ℕ := n^2 + 3 * n

theorem general_term_an (n : ℕ) (h : 1 ≤ n) : a_n n = (S_n n) - (S_n (n-1)) :=
by sorry

end general_term_an_l1924_192497


namespace sequence_equals_identity_l1924_192422

theorem sequence_equals_identity (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → gcd (a i) (a j) = gcd i j) : 
  ∀ i : ℕ, a i = i := 
by 
  sorry

end sequence_equals_identity_l1924_192422


namespace remainder_zero_by_68_l1924_192405

theorem remainder_zero_by_68 (N R1 Q2 : ℕ) (h1 : N = 68 * 269 + R1) (h2 : N % 67 = 1) : R1 = 0 := by
  sorry

end remainder_zero_by_68_l1924_192405


namespace selection_of_hexagonal_shape_l1924_192408

-- Lean 4 Statement: Prove that there are 78 distinct ways to select diagram b from the hexagonal grid of diagram a, considering rotations.

theorem selection_of_hexagonal_shape :
  let center_positions := 1
  let first_ring_positions := 6
  let second_ring_positions := 12
  let third_ring_positions := 6
  let fourth_ring_positions := 1
  let total_positions := center_positions + first_ring_positions + second_ring_positions + third_ring_positions + fourth_ring_positions
  let rotations := 3
  total_positions * rotations = 78 := by
  -- You can skip the explicit proof body here, replace with sorry
  sorry

end selection_of_hexagonal_shape_l1924_192408


namespace chromium_percentage_l1924_192478

theorem chromium_percentage (x : ℝ) : 
  (15 * x / 100 + 35 * 8 / 100 = 50 * 8.6 / 100) → 
  x = 10 := 
sorry

end chromium_percentage_l1924_192478


namespace parallelogram_count_l1924_192416

theorem parallelogram_count (m n : ℕ) : 
  ∃ p : ℕ, p = (m.choose 2) * (n.choose 2) :=
by
  sorry

end parallelogram_count_l1924_192416


namespace max_common_ratio_arithmetic_geometric_sequence_l1924_192453

open Nat

theorem max_common_ratio_arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (k : ℕ) (q : ℝ) 
  (hk : k ≥ 2) (ha : ∀ n, a (n + 1) = a n + d)
  (hg : (a 1) * (a (2 * k)) = (a k) ^ 2) :
  q ≤ 2 :=
by
  sorry

end max_common_ratio_arithmetic_geometric_sequence_l1924_192453


namespace employees_cycle_l1924_192486

theorem employees_cycle (total_employees : ℕ) (drivers_percentage walkers_percentage cyclers_percentage: ℕ) (walk_cycle_ratio_walk walk_cycle_ratio_cycle: ℕ)
    (h_total : total_employees = 500)
    (h_drivers_perc : drivers_percentage = 35)
    (h_transit_perc : walkers_percentage = 25)
    (h_walkers_cyclers_ratio_walk : walk_cycle_ratio_walk = 3)
    (h_walkers_cyclers_ratio_cycle : walk_cycle_ratio_cycle = 7) :
    cyclers_percentage = 140 :=
by
  sorry

end employees_cycle_l1924_192486


namespace find_a1_l1924_192414

noncomputable def a_n : ℕ → ℝ := sorry
noncomputable def S_n : ℕ → ℝ := sorry

theorem find_a1
  (h1 : ∀ n : ℕ, a_n 2 * a_n 8 = 2 * a_n 3 * a_n 6)
  (h2 : S_n 5 = -62) :
  a_n 1 = -2 :=
sorry

end find_a1_l1924_192414


namespace cos_sq_sub_sin_sq_pi_div_12_l1924_192455

theorem cos_sq_sub_sin_sq_pi_div_12 : 
  (Real.cos (π / 12))^2 - (Real.sin (π / 12))^2 = Real.cos (π / 6) :=
by
  sorry

end cos_sq_sub_sin_sq_pi_div_12_l1924_192455


namespace max_L_shaped_figures_in_5x7_rectangle_l1924_192417

def L_shaped_figure : Type := ℕ

def rectangle_area := 5 * 7

def l_shape_area := 3

def max_l_shapes_in_rectangle (rect_area : ℕ) (l_area : ℕ) : ℕ := rect_area / l_area

theorem max_L_shaped_figures_in_5x7_rectangle : max_l_shapes_in_rectangle rectangle_area l_shape_area = 11 :=
by
  sorry

end max_L_shaped_figures_in_5x7_rectangle_l1924_192417


namespace arithmetic_sequence_problem_l1924_192435

variable {α : Type*} [LinearOrderedRing α]

theorem arithmetic_sequence_problem
  (a : ℕ → α)
  (h : ∀ n, a (n + 1) = a n + (a 1 - a 0))
  (h_seq : a 5 + a 6 + a 7 + a 8 + a 9 = 450) :
  a 3 + a 11 = 180 :=
sorry

end arithmetic_sequence_problem_l1924_192435


namespace non_neg_scalar_product_l1924_192413

theorem non_neg_scalar_product (a b c d e f g h : ℝ) : 
  (0 ≤ ac + bd) ∨ (0 ≤ ae + bf) ∨ (0 ≤ ag + bh) ∨ (0 ≤ ce + df) ∨ (0 ≤ cg + dh) ∨ (0 ≤ eg + fh) :=
  sorry

end non_neg_scalar_product_l1924_192413


namespace fencing_cost_l1924_192411

def total_cost_of_fencing 
  (length breadth cost_per_meter : ℝ)
  (h1 : length = 62)
  (h2 : length = breadth + 24)
  (h3 : cost_per_meter = 26.50) : ℝ :=
  2 * (length + breadth) * cost_per_meter

theorem fencing_cost : total_cost_of_fencing 62 38 26.50 (by rfl) (by norm_num) (by norm_num) = 5300 := 
by 
  sorry

end fencing_cost_l1924_192411


namespace f_is_even_f_monotonic_increase_range_of_a_for_solutions_l1924_192488

-- Define the function f(x) = x^2 - 2a|x|
def f (a x : ℝ) : ℝ := x^2 - 2 * a * |x|

-- Given a > 0
variable (a : ℝ) (ha : a > 0)

-- 1. Prove that f(x) is an even function.
theorem f_is_even : ∀ x : ℝ, f a x = f a (-x) := sorry

-- 2. Prove the interval of monotonic increase for f(x) when x > 0 is [a, +∞).
theorem f_monotonic_increase (x : ℝ) (hx : x > 0) : a ≤ x → ∃ c : ℝ, x ≤ c := sorry

-- 3. Prove the range of values for a for which the equation f(x) = -1 has solutions is a ≥ 1.
theorem range_of_a_for_solutions : (∃ x : ℝ, f a x = -1) ↔ 1 ≤ a := sorry

end f_is_even_f_monotonic_increase_range_of_a_for_solutions_l1924_192488


namespace longest_side_of_triangle_l1924_192464

-- Definitions of the conditions in a)
def side1 : ℝ := 9
def side2 (x : ℝ) : ℝ := x + 5
def side3 (x : ℝ) : ℝ := 2 * x + 3
def perimeter : ℝ := 40

-- Statement of the mathematically equivalent proof problem.
theorem longest_side_of_triangle (x : ℝ) (h : side1 + side2 x + side3 x = perimeter) : 
  max side1 (max (side2 x) (side3 x)) = side3 x := 
sorry

end longest_side_of_triangle_l1924_192464


namespace find_functions_satisfying_lcm_gcd_eq_l1924_192436

noncomputable def satisfies_functional_equation (f : ℕ → ℕ) : Prop := 
  ∀ m n : ℕ, m > 0 ∧ n > 0 → f (m * n) = Nat.lcm m n * Nat.gcd (f m) (f n)

noncomputable def solution_form (f : ℕ → ℕ) : Prop := 
  ∃ k : ℕ, ∀ x : ℕ, f x = k * x

theorem find_functions_satisfying_lcm_gcd_eq (f : ℕ → ℕ) : 
  satisfies_functional_equation f ↔ solution_form f := 
sorry

end find_functions_satisfying_lcm_gcd_eq_l1924_192436


namespace percent_nurses_with_neither_l1924_192473

-- Define the number of nurses in each category
def total_nurses : ℕ := 150
def nurses_with_hbp : ℕ := 90
def nurses_with_ht : ℕ := 50
def nurses_with_both : ℕ := 30

-- Define a predicate that checks the conditions of the problem
theorem percent_nurses_with_neither :
  ((total_nurses - (nurses_with_hbp + nurses_with_ht - nurses_with_both)) * 100 : ℚ) / total_nurses = 2667 / 100 :=
by sorry

end percent_nurses_with_neither_l1924_192473


namespace suraj_average_after_17th_innings_l1924_192491

theorem suraj_average_after_17th_innings (A : ℕ) :
  (16 * A + 92) / 17 = A + 4 -> A + 4 = 28 := 
by 
  sorry

end suraj_average_after_17th_innings_l1924_192491


namespace function_zero_interval_l1924_192477

noncomputable def f (x : ℝ) : ℝ := 1 / 4^x - Real.log x / Real.log 4

theorem function_zero_interval :
  ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 := by
  sorry

end function_zero_interval_l1924_192477
