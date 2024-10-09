import Mathlib

namespace fractions_expressible_iff_prime_l1383_138395

noncomputable def is_good_fraction (a b n : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = n

theorem fractions_expressible_iff_prime (n : ℕ) (hn : n > 1) :
  (∀ (a b : ℕ), b < n → ∃ (k l : ℤ), k * a + l * n = b) ↔ Prime n :=
sorry

end fractions_expressible_iff_prime_l1383_138395


namespace simplify_expression_l1383_138345

theorem simplify_expression (x : ℝ) : 5 * x + 6 - x + 12 = 4 * x + 18 :=
by sorry

end simplify_expression_l1383_138345


namespace find_b_l1383_138372

noncomputable def p (x : ℕ) := 3 * x + 5
noncomputable def q (x : ℕ) (b : ℕ) := 4 * x - b

theorem find_b : ∃ (b : ℕ), p (q 3 b) = 29 ∧ b = 4 := sorry

end find_b_l1383_138372


namespace min_value_range_of_x_l1383_138335

variables (a b x : ℝ)

-- Problem 1: Prove the minimum value of 1/a + 4/b given a + b = 1, a > 0, b > 0
theorem min_value (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : 
  ∃ c, c = 9 ∧ ∀ y, ∃ (a b : ℝ), a + b = 1 ∧ a > 0 ∧ b > 0 → (1/a + 4/b) ≥ y :=
sorry

-- Problem 2: Prove the range of x for which 1/a + 4/b ≥ |2x - 1| - |x + 1|
theorem range_of_x (h : ∀ (a b : ℝ), a + b = 1 → a > 0 → b > 0 → (1/a + 4/b) ≥ (|2*x - 1| - |x + 1|)) :
  -7 ≤ x ∧ x ≤ 11 :=
sorry

end min_value_range_of_x_l1383_138335


namespace minimum_flour_cost_l1383_138351

-- Definitions based on conditions provided
def loaves : ℕ := 12
def flour_per_loaf : ℕ := 4
def flour_needed : ℕ := loaves * flour_per_loaf

def ten_pound_bag_weight : ℕ := 10
def ten_pound_bag_cost : ℕ := 10

def twelve_pound_bag_weight : ℕ := 12
def twelve_pound_bag_cost : ℕ := 13

def cost_10_pound_bags : ℕ := (flour_needed + ten_pound_bag_weight - 1) / ten_pound_bag_weight * ten_pound_bag_cost
def cost_12_pound_bags : ℕ := (flour_needed + twelve_pound_bag_weight - 1) / twelve_pound_bag_weight * twelve_pound_bag_cost

theorem minimum_flour_cost : min cost_10_pound_bags cost_12_pound_bags = 50 := by
  sorry

end minimum_flour_cost_l1383_138351


namespace factorize_ax2_minus_a_l1383_138317

theorem factorize_ax2_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_ax2_minus_a_l1383_138317


namespace work_rate_a_b_l1383_138393

/-- a and b can do a piece of work in some days, b and c in 5 days, c and a in 15 days. If c takes 12 days to do the work, 
    prove that a and b together can complete the work in 10 days.
-/
theorem work_rate_a_b
  (A B C : ℚ) 
  (h1 : B + C = 1 / 5)
  (h2 : C + A = 1 / 15)
  (h3 : C = 1 / 12) :
  (A + B = 1 / 10) := 
sorry

end work_rate_a_b_l1383_138393


namespace arithmetic_sequence_sum_first_nine_terms_l1383_138368

variable (a_n : ℕ → ℤ)
variable (S_n : ℕ → ℤ)
variable (d : ℤ)

-- The sequence {a_n} is an arithmetic sequence.
def arithmetic_sequence := ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- The sum of the first n terms of the sequence.
def sum_first_n_terms := ∀ n : ℕ, S_n n = (n * (a_n 1 + a_n n)) / 2

-- Given condition: a_2 = 3 * a_4 - 6
def given_condition := a_n 2 = 3 * a_n 4 - 6

-- The main theorem to prove S_9 = 27
theorem arithmetic_sequence_sum_first_nine_terms (h_arith : arithmetic_sequence a_n d) (h_sum : sum_first_n_terms a_n S_n) (h_condition : given_condition a_n) : 
  S_n 9 = 27 := 
by
  sorry

end arithmetic_sequence_sum_first_nine_terms_l1383_138368


namespace slope_equal_angles_l1383_138331

-- Define the problem
theorem slope_equal_angles (k : ℝ) :
  (∀ (l1 l2 : ℝ), l1 = 1 ∧ l2 = 2 → (abs ((k - l1) / (1 + k * l1)) = abs ((l2 - k) / (1 + l2 * k)))) →
  (k = (1 + Real.sqrt 10) / 3 ∨ k = (1 - Real.sqrt 10) / 3) :=
by
  intros
  sorry

end slope_equal_angles_l1383_138331


namespace composition_points_value_l1383_138373

theorem composition_points_value (f g : ℕ → ℕ) (ab cd : ℕ) 
  (h₁ : f 2 = 6) 
  (h₂ : f 3 = 4) 
  (h₃ : f 4 = 2)
  (h₄ : g 2 = 4) 
  (h₅ : g 3 = 2) 
  (h₆ : g 5 = 6) :
  let (a, b) := (2, 6)
  let (c, d) := (3, 4)
  ab + cd = (a * b) + (c * d) :=
by {
  sorry
}

end composition_points_value_l1383_138373


namespace relationship_l1383_138381

-- Given definitions
def S : ℕ := 31
def L : ℕ := 124 - S

-- Proving the relationship
theorem relationship: S + L = 124 ∧ S = 31 → L = S + 62 := by
  sorry

end relationship_l1383_138381


namespace max_value_of_ab_expression_l1383_138324

noncomputable def max_ab_expression : ℝ :=
  let a := 4
  let b := 20 / 3
  a * b * (60 - 5 * a - 3 * b)

theorem max_value_of_ab_expression :
  ∀ (a b : ℝ), 0 < a → 0 < b → 5 * a + 3 * b < 60 →
  ab * (60 - 5 * a - 3 * b) ≤ max_ab_expression :=
sorry

end max_value_of_ab_expression_l1383_138324


namespace part1_part2_l1383_138336

open Real

-- Condition: tan(alpha) = 3
variable {α : ℝ} (h : tan α = 3)

-- Proof of first part
theorem part1 : (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 :=
by
  sorry

-- Proof of second part
theorem part2 : 1 - 4 * sin α * cos α + 2 * cos α ^ 2 = 0 :=
by
  sorry

end part1_part2_l1383_138336


namespace property_holds_for_1_and_4_l1383_138300

theorem property_holds_for_1_and_4 (n : ℕ) : 
  (∀ q : ℕ, n % q^2 < q^(q^2) / 2) ↔ (n = 1 ∨ n = 4) :=
by sorry

end property_holds_for_1_and_4_l1383_138300


namespace quadratic_roots_vieta_l1383_138349

theorem quadratic_roots_vieta :
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → 
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 2 ∧ x₁ * x₂ = -8) → 
  (x₁ + x₂) / (x₁ * x₂) = -1 / 4 :=
by
  intros x₁ x₂ h₁ h₂
  sorry

end quadratic_roots_vieta_l1383_138349


namespace factory_profit_l1383_138357

def cost_per_unit : ℝ := 2.00
def fixed_cost : ℝ := 500.00
def selling_price_per_unit : ℝ := 2.50

theorem factory_profit (x : ℕ) (hx : x > 1000) :
  selling_price_per_unit * x > fixed_cost + cost_per_unit * x :=
by
  sorry

end factory_profit_l1383_138357


namespace max_angle_in_hexagon_l1383_138347

-- Definition of the problem
theorem max_angle_in_hexagon :
  ∃ (a d : ℕ), a + a + d + a + 2*d + a + 3*d + a + 4*d + a + 5*d = 720 ∧ 
               a + 5 * d < 180 ∧ 
               (∀ a d : ℕ, a + a + d + a + 2*d + a + 3*d + a + 4*d + a + 5*d = 720 → 
               a + 5*d < 180 → m <= 175) :=
sorry

end max_angle_in_hexagon_l1383_138347


namespace least_integer_value_l1383_138343

theorem least_integer_value 
  (x : ℤ) (h : |3 * x - 5| ≤ 22) : x = -5 ↔ ∃ (k : ℤ), k = -5 ∧ |3 * k - 5| ≤ 22 :=
by
  sorry

end least_integer_value_l1383_138343


namespace c_work_rate_l1383_138350

theorem c_work_rate (x : ℝ) : 
  (1 / 7 + 1 / 14 + 1 / x = 1 / 4) → x = 28 :=
by
  sorry

end c_work_rate_l1383_138350


namespace isosceles_with_60_eq_angle_is_equilateral_l1383_138341

open Real

noncomputable def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) :=
  A = 60 ∧ B = 60 ∧ C = 60

noncomputable def is_isosceles_triangle (a b c : ℝ) (A B C : ℝ) :=
  (a = b ∨ b = c ∨ c = a) ∧ (A + B + C = 180)

theorem isosceles_with_60_eq_angle_is_equilateral
  (a b c A B C : ℝ)
  (h_iso : is_isosceles_triangle a b c A B C)
  (h_angle : A = 60 ∨ B = 60 ∨ C = 60) :
  is_equilateral_triangle a b c A B C :=
sorry

end isosceles_with_60_eq_angle_is_equilateral_l1383_138341


namespace sum_of_inserted_numbers_l1383_138398

variable {x y : ℝ} -- Variables x and y are real numbers

-- Conditions
axiom geometric_sequence_condition : x^2 = 3 * y
axiom arithmetic_sequence_condition : 2 * y = x + 9

-- Goal: Prove that x + y = 45 / 4 (which is 11 1/4)
theorem sum_of_inserted_numbers : x + y = 45 / 4 :=
by
  -- Utilize axioms and conditions
  sorry

end sum_of_inserted_numbers_l1383_138398


namespace remainder_of_sum_l1383_138344

theorem remainder_of_sum (n1 n2 n3 n4 n5 : ℕ) (h1 : n1 = 7145) (h2 : n2 = 7146)
  (h3 : n3 = 7147) (h4 : n4 = 7148) (h5 : n5 = 7149) :
  ((n1 + n2 + n3 + n4 + n5) % 8) = 7 :=
by sorry

end remainder_of_sum_l1383_138344


namespace remainder_div_x_plus_2_l1383_138359

def f (x : ℤ) : ℤ := x^15 + 3

theorem remainder_div_x_plus_2 : f (-2) = -32765 := by
  sorry

end remainder_div_x_plus_2_l1383_138359


namespace expected_heads_value_in_cents_l1383_138339

open ProbabilityTheory

-- Define the coins and their respective values
def penny_value := 1
def nickel_value := 5
def half_dollar_value := 50
def dollar_value := 100

-- Define the probability of landing heads for each coin
def heads_prob := 1 / 2

-- Define the expected value function
noncomputable def expected_value_of_heads : ℝ :=
  heads_prob * (penny_value + nickel_value + half_dollar_value + dollar_value)

theorem expected_heads_value_in_cents : expected_value_of_heads = 78 := by
  sorry

end expected_heads_value_in_cents_l1383_138339


namespace cos_square_theta_plus_pi_over_4_eq_one_fourth_l1383_138304

variable (θ : ℝ)

theorem cos_square_theta_plus_pi_over_4_eq_one_fourth
  (h : Real.tan θ + 1 / Real.tan θ = 4) :
  Real.cos (θ + Real.pi / 4) ^ 2 = 1 / 4 :=
sorry

end cos_square_theta_plus_pi_over_4_eq_one_fourth_l1383_138304


namespace construct_convex_hexagon_l1383_138375

-- Definitions of the sides and their lengths
variables {A B C D E F : Type} -- Points of the hexagon
variables {AB BC CD DE EF FA : ℝ}  -- Lengths of the sides
variables (convex_hexagon : Prop) -- the hexagon is convex

-- Hypotheses of parallel and equal opposite sides
variables (H_AB_DE : AB = DE)
variables (H_BC_EF : BC = EF)
variables (H_CD_AF : CD = AF)

-- Define the construction of the hexagon under the given conditions
theorem construct_convex_hexagon
  (convex_hexagon : Prop)
  (H_AB_DE : AB = DE)
  (H_BC_EF : BC = EF)
  (H_CD_AF : CD = AF) : 
  ∃ (A B C D E F : Type), 
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A ∧ convex_hexagon ∧ 
    (AB = FA) ∧ (AF = CD) ∧ (BC = EF) ∧ (AB = DE) := 
sorry -- Proof omitted

end construct_convex_hexagon_l1383_138375


namespace Joann_lollipop_theorem_l1383_138322

noncomputable def Joann_lollipops (a : ℝ) : ℝ := a + 9

theorem Joann_lollipop_theorem (a : ℝ) (total_lollipops : ℝ) 
  (h1 : a + (a + 3) + (a + 6) + (a + 9) + (a + 12) + (a + 15) = 150) 
  (h2 : total_lollipops = 150) : 
  Joann_lollipops a = 26.5 :=
by
  sorry

end Joann_lollipop_theorem_l1383_138322


namespace no_solution_x_to_2n_plus_y_to_2n_eq_z_sq_l1383_138319

theorem no_solution_x_to_2n_plus_y_to_2n_eq_z_sq (n : ℕ) (h : ∀ (x y z : ℕ), x^n + y^n ≠ z^n) : ∀ (x y z : ℕ), x^(2*n) + y^(2*n) ≠ z^2 :=
by 
  intro x y z
  sorry

end no_solution_x_to_2n_plus_y_to_2n_eq_z_sq_l1383_138319


namespace range_of_a_l1383_138387

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end range_of_a_l1383_138387


namespace time_to_cover_length_l1383_138364

-- Define the conditions
def speed_escalator : ℝ := 12
def length_escalator : ℝ := 150
def speed_person : ℝ := 3

-- State the theorem to be proved
theorem time_to_cover_length : (length_escalator / (speed_escalator + speed_person)) = 10 := by
  sorry

end time_to_cover_length_l1383_138364


namespace product_of_distinct_numbers_l1383_138355

theorem product_of_distinct_numbers (x y : ℝ) (h1 : x ≠ y)
  (h2 : 1 / (1 + x^2) + 1 / (1 + y^2) = 2 / (1 + x * y)) :
  x * y = 1 := 
sorry

end product_of_distinct_numbers_l1383_138355


namespace proof_problem_l1383_138396

open Set

noncomputable def U : Set ℝ := Icc (-5 : ℝ) 4

noncomputable def A : Set ℝ := {x : ℝ | -3 ≤ 2 * x + 1 ∧ 2 * x + 1 < 1}

noncomputable def B : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}

-- Definition of the complement of A in U
noncomputable def complement_U_A : Set ℝ := U \ A

-- The final proof statement
theorem proof_problem : (complement_U_A ∩ B) = Icc 0 2 :=
by
  sorry

end proof_problem_l1383_138396


namespace value_of_x_l1383_138399

theorem value_of_x (v w z y x : ℤ) 
  (h1 : v = 90)
  (h2 : w = v + 30)
  (h3 : z = w + 21)
  (h4 : y = z + 11)
  (h5 : x = y + 6) : 
  x = 158 :=
by 
  sorry

end value_of_x_l1383_138399


namespace students_going_on_field_trip_l1383_138365

-- Define conditions
def van_capacity : Nat := 7
def number_of_vans : Nat := 6
def number_of_adults : Nat := 9

-- Define the total capacity
def total_people_capacity : Nat := number_of_vans * van_capacity

-- Define the number of students
def number_of_students : Nat := total_people_capacity - number_of_adults

-- Prove the number of students is 33
theorem students_going_on_field_trip : number_of_students = 33 := by
  sorry

end students_going_on_field_trip_l1383_138365


namespace f_monotonicity_l1383_138394

noncomputable def f : ℝ → ℝ := sorry -- Definition of the function f(x)

axiom f_symm (x : ℝ) : f (1 - x) = f x

axiom f_derivative (x : ℝ) : (x - 1 / 2) * (deriv f x) > 0

theorem f_monotonicity (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : x1 + x2 > 1) : f x1 < f x2 :=
sorry

end f_monotonicity_l1383_138394


namespace rectangle_ratio_l1383_138391

theorem rectangle_ratio (s y x : ℝ)
  (h1 : 4 * y * x + s * s = 9 * s * s)
  (h2 : s + y + y = 3 * s)
  (h3 : y = s)
  (h4 : x + s = 3 * s) : 
  (x / y = 2) :=
sorry

end rectangle_ratio_l1383_138391


namespace time_boarding_in_London_l1383_138306

open Nat

def time_in_ET_to_London_time (time_et: ℕ) : ℕ :=
  (time_et + 5) % 24

def subtract_hours (time: ℕ) (hours: ℕ) : ℕ :=
  (time + 24 * (hours / 24) - (hours % 24)) % 24

theorem time_boarding_in_London :
  let cape_town_arrival_time_et := 10
  let flight_duration_ny_to_cape := 10
  let ny_departure_time := subtract_hours cape_town_arrival_time_et flight_duration_ny_to_cape
  let flight_duration_london_to_ny := 18
  let ny_arrival_time := subtract_hours ny_departure_time flight_duration_london_to_ny
  let london_time := time_in_ET_to_London_time ny_arrival_time
  let london_departure_time := subtract_hours london_time flight_duration_london_to_ny
  london_departure_time = 17 :=
by
  -- Proof omitted
  sorry

end time_boarding_in_London_l1383_138306


namespace min_value_x_4_over_x_min_value_x_4_over_x_eq_l1383_138303

theorem min_value_x_4_over_x (x : ℝ) (h : x > 0) : x + 4 / x ≥ 4 :=
sorry

theorem min_value_x_4_over_x_eq (x : ℝ) (h : x > 0) : (x + 4 / x = 4) ↔ (x = 2) :=
sorry

end min_value_x_4_over_x_min_value_x_4_over_x_eq_l1383_138303


namespace find_x_l1383_138307

theorem find_x (x : ℚ) : x * 9999 = 724827405 → x = 72492.75 :=
by
  sorry

end find_x_l1383_138307


namespace polynomial_factorization_l1383_138361

theorem polynomial_factorization (x : ℝ) :
  x^6 + 6*x^5 + 15*x^4 + 20*x^3 + 15*x^2 + 6*x + 1 = (x + 1)^6 :=
by {
  -- proof goes here
  sorry
}

end polynomial_factorization_l1383_138361


namespace prism_volume_l1383_138340

noncomputable def volume_of_prism (l w h : ℝ) : ℝ :=
l * w * h

theorem prism_volume (l w h : ℝ) (h1 : l = 2 * w) (h2 : l * w = 10) (h3 : w * h = 18) (h4 : l * h = 36) :
  volume_of_prism l w h = 36 * Real.sqrt 5 :=
by
  -- Proof goes here
  sorry

end prism_volume_l1383_138340


namespace find_xy_l1383_138312

theorem find_xy (p x y : ℕ) (hp : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  p * (x - y) = x * y ↔ (x, y) = (p^2 - p, p + 1) := by
  sorry

end find_xy_l1383_138312


namespace sum_of_two_numbers_l1383_138356

theorem sum_of_two_numbers (a b S : ℤ) (h : a + b = S) : 
  3 * (a + 5) + 3 * (b + 7) = 3 * S + 36 :=
by
  sorry

end sum_of_two_numbers_l1383_138356


namespace cost_price_of_watch_l1383_138392

theorem cost_price_of_watch 
  (CP : ℝ)
  (h1 : 0.88 * CP = SP_loss)
  (h2 : 1.04 * CP = SP_gain)
  (h3 : SP_gain - SP_loss = 140) :
  CP = 875 := 
sorry

end cost_price_of_watch_l1383_138392


namespace part_one_part_two_l1383_138323

-- Given that tan α = 2, prove that the following expressions are correct:

theorem part_one (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (Real.pi - α) + Real.cos (α - Real.pi / 2) - Real.cos (3 * Real.pi + α)) / 
  (Real.cos (Real.pi / 2 + α) - Real.sin (2 * Real.pi + α) + 2 * Real.sin (α - Real.pi / 2)) = 
  -5 / 6 := 
by
  -- Proof skipped
  sorry

theorem part_two (α : ℝ) (h : Real.tan α = 2) :
  Real.cos (2 * α) + Real.sin α * Real.cos α = -1 / 5 := 
by
  -- Proof skipped
  sorry

end part_one_part_two_l1383_138323


namespace shortest_side_length_l1383_138321

theorem shortest_side_length (perimeter : ℝ) (shortest : ℝ) (side1 side2 side3 : ℝ) 
  (h1 : side1 + side2 + side3 = perimeter)
  (h2 : side1 = 2 * shortest)
  (h3 : side2 = 2 * shortest) :
  shortest = 3 := by
  sorry

end shortest_side_length_l1383_138321


namespace sqrt_expression_l1383_138332

theorem sqrt_expression :
  (Real.sqrt (2 ^ 4 * 3 ^ 6 * 5 ^ 2)) = 540 := sorry

end sqrt_expression_l1383_138332


namespace bob_got_15_candies_l1383_138318

-- Define the problem conditions
def bob_neighbor_sam : Prop := true -- Bob is Sam's next door neighbor
def bob_accompany_sam_home : Prop := true -- Bob decided to accompany Sam home

def bob_share_chewing_gums : ℕ := 15 -- Bob's share of chewing gums
def bob_share_chocolate_bars : ℕ := 20 -- Bob's share of chocolate bars
def bob_share_candies : ℕ := 15 -- Bob's share of assorted candies

-- Define the main assertion
theorem bob_got_15_candies : bob_share_candies = 15 := 
by sorry

end bob_got_15_candies_l1383_138318


namespace tax_percentage_l1383_138333

theorem tax_percentage (car_price tax_paid first_tier_price : ℝ) (first_tier_tax_rate : ℝ) (tax_second_tier : ℝ) :
  car_price = 30000 ∧
  tax_paid = 5500 ∧
  first_tier_price = 10000 ∧
  first_tier_tax_rate = 0.25 ∧
  tax_second_tier = 0.15
  → (tax_second_tier) = 0.15 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4, h5⟩
  sorry

end tax_percentage_l1383_138333


namespace simplify_expression_l1383_138363

theorem simplify_expression (x : ℝ) : (x + 3) * (x - 3) = x^2 - 9 :=
by
  -- We acknowledge this is the placeholder for the proof.
  -- This statement follows directly from the difference of squares identity.
  sorry

end simplify_expression_l1383_138363


namespace saree_final_sale_price_in_inr_l1383_138362

noncomputable def finalSalePrice (initialPrice: ℝ) (discounts: List ℝ) (conversionRate: ℝ) : ℝ :=
  let finalUSDPrice := discounts.foldl (fun acc discount => acc * (1 - discount)) initialPrice
  finalUSDPrice * conversionRate

theorem saree_final_sale_price_in_inr
  (initialPrice : ℝ := 150)
  (discounts : List ℝ := [0.20, 0.15, 0.05])
  (conversionRate : ℝ := 75)
  : finalSalePrice initialPrice discounts conversionRate = 7267.5 :=
by
  sorry

end saree_final_sale_price_in_inr_l1383_138362


namespace races_to_champion_l1383_138309

theorem races_to_champion (num_sprinters : ℕ) (sprinters_per_race : ℕ) (advancing_per_race : ℕ)
  (eliminated_per_race : ℕ) (initial_races : ℕ) (total_races : ℕ):
  num_sprinters = 360 ∧ sprinters_per_race = 8 ∧ advancing_per_race = 2 ∧ 
  eliminated_per_race = 6 ∧ initial_races = 45 ∧ total_races = 62 →
  initial_races + (initial_races / sprinters_per_race +
  ((initial_races / sprinters_per_race) / sprinters_per_race +
  (((initial_races / sprinters_per_race) / sprinters_per_race) / sprinters_per_race + 1))) = total_races :=
sorry

end races_to_champion_l1383_138309


namespace bumper_car_rides_l1383_138311

-- Define the conditions
def rides_on_ferris_wheel : ℕ := 7
def cost_per_ride : ℕ := 5
def total_tickets : ℕ := 50

-- Formulate the statement to be proved
theorem bumper_car_rides : ∃ n : ℕ, 
  total_tickets = (rides_on_ferris_wheel * cost_per_ride) + (n * cost_per_ride) ∧ n = 3 :=
sorry

end bumper_car_rides_l1383_138311


namespace A_square_or_cube_neg_identity_l1383_138386

open Matrix

theorem A_square_or_cube_neg_identity (A : Matrix (Fin 2) (Fin 2) ℚ)
  (n : ℕ) (hn_nonzero : n ≠ 0) (hA_pow_n : A ^ n = -(1 : Matrix (Fin 2) (Fin 2) ℚ)) :
  A ^ 2 = -(1 : Matrix (Fin 2) (Fin 2) ℚ) ∨ A ^ 3 = -(1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

end A_square_or_cube_neg_identity_l1383_138386


namespace fencing_cost_correct_l1383_138384

noncomputable def length : ℝ := 80
noncomputable def diff : ℝ := 60
noncomputable def cost_per_meter : ℝ := 26.50

-- Let's calculate the breadth first
noncomputable def breadth : ℝ := length - diff

-- Calculate the perimeter
noncomputable def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost
noncomputable def total_cost : ℝ := perimeter * cost_per_meter

theorem fencing_cost_correct : total_cost = 5300 := 
by 
  sorry

end fencing_cost_correct_l1383_138384


namespace mice_path_count_l1383_138358

theorem mice_path_count
  (x y : ℕ)
  (left_house_yesterday top_house_yesterday right_house_yesterday : ℕ)
  (left_house_today top_house_today right_house_today : ℕ)
  (h_left_yesterday : left_house_yesterday = 8)
  (h_top_yesterday : top_house_yesterday = 4)
  (h_right_yesterday : right_house_yesterday = 7)
  (h_left_today : left_house_today = 4)
  (h_top_today : top_house_today = 4)
  (h_right_today : right_house_today = 7)
  (h_eq : (left_house_yesterday - left_house_today) + 
          (right_house_yesterday - right_house_today) = 
          top_house_today - top_house_yesterday) :
  x + y = 11 :=
by
  sorry

end mice_path_count_l1383_138358


namespace total_money_collected_is_140_l1383_138377

def total_attendees : ℕ := 280
def child_attendees : ℕ := 80
def adult_attendees : ℕ := total_attendees - child_attendees
def adult_ticket_cost : ℝ := 0.60
def child_ticket_cost : ℝ := 0.25

def money_collected_from_adults : ℝ := adult_attendees * adult_ticket_cost
def money_collected_from_children : ℝ := child_attendees * child_ticket_cost
def total_money_collected : ℝ := money_collected_from_adults + money_collected_from_children

theorem total_money_collected_is_140 : total_money_collected = 140 := by
  sorry

end total_money_collected_is_140_l1383_138377


namespace value_range_of_f_l1383_138366

noncomputable def f (x : ℝ) : ℝ := 2 + Real.logb 5 (x + 3)

theorem value_range_of_f :
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ∈ Set.Icc (2 : ℝ) 3 := 
by
  sorry

end value_range_of_f_l1383_138366


namespace sin_alpha_value_l1383_138346

open Real


theorem sin_alpha_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : π / 2 < β ∧ β < π)
  (h_sin_alpha_beta : sin (α + β) = 3 / 5) (h_cos_beta : cos β = -5 / 13) :
  sin α = 33 / 65 := 
by
  sorry

end sin_alpha_value_l1383_138346


namespace min_troublemakers_l1383_138329

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l1383_138329


namespace rectangle_length_l1383_138338

theorem rectangle_length (P L B : ℕ) (hP : P = 500) (hB : B = 100) (hP_eq : P = 2 * (L + B)) : L = 150 :=
by
  sorry

end rectangle_length_l1383_138338


namespace trains_cross_time_l1383_138371

noncomputable def time_to_cross (len1 len2 speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let speed1_ms := speed1_kmh * (5 / 18)
  let speed2_ms := speed2_kmh * (5 / 18)
  let relative_speed_ms := speed1_ms + speed2_ms
  let total_distance := len1 + len2
  total_distance / relative_speed_ms

theorem trains_cross_time :
  time_to_cross 1500 1000 90 75 = 54.55 := by
  sorry

end trains_cross_time_l1383_138371


namespace determine_k_l1383_138301

noncomputable def f (x k : ℝ) : ℝ := -4 * x^3 + k * x

theorem determine_k : ∀ k : ℝ, (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x k ≤ 1) → k = 3 :=
by
  sorry

end determine_k_l1383_138301


namespace inequality_holds_iff_x_in_interval_l1383_138316

theorem inequality_holds_iff_x_in_interval (x : ℝ) :
  (∀ n : ℕ, 0 < n → (1 + x)^n ≤ 1 + (2^n - 1) * x) ↔ (0 ≤ x ∧ x ≤ 1) :=
sorry

end inequality_holds_iff_x_in_interval_l1383_138316


namespace exists_arithmetic_progression_2004_perfect_powers_perfect_powers_not_infinite_arithmetic_progression_l1383_138352

-- Definition: A positive integer n is a perfect power if n = a ^ b for some integers a, b with b > 1.
def isPerfectPower (n : ℕ) : Prop :=
  ∃ a b : ℕ, b > 1 ∧ n = a^b

-- Part (a): Prove the existence of an arithmetic progression of 2004 perfect powers.
theorem exists_arithmetic_progression_2004_perfect_powers :
  ∃ (x r : ℕ), (∀ n : ℕ, n < 2004 → ∃ a b : ℕ, b > 1 ∧ (x + n * r) = a^b) :=
sorry

-- Part (b): Prove that perfect powers cannot form an infinite arithmetic progression.
theorem perfect_powers_not_infinite_arithmetic_progression :
  ¬ ∃ (x r : ℕ), (∀ n : ℕ, ∃ a b : ℕ, b > 1 ∧ (x + n * r) = a^b) :=
sorry

end exists_arithmetic_progression_2004_perfect_powers_perfect_powers_not_infinite_arithmetic_progression_l1383_138352


namespace michael_total_cost_l1383_138374

def peach_pies : ℕ := 5
def apple_pies : ℕ := 4
def blueberry_pies : ℕ := 3

def pounds_per_pie : ℕ := 3

def price_per_pound_peaches : ℝ := 2.0
def price_per_pound_apples : ℝ := 1.0
def price_per_pound_blueberries : ℝ := 1.0

def total_peach_pounds : ℕ := peach_pies * pounds_per_pie
def total_apple_pounds : ℕ := apple_pies * pounds_per_pie
def total_blueberry_pounds : ℕ := blueberry_pies * pounds_per_pie

def cost_peaches : ℝ := total_peach_pounds * price_per_pound_peaches
def cost_apples : ℝ := total_apple_pounds * price_per_pound_apples
def cost_blueberries : ℝ := total_blueberry_pounds * price_per_pound_blueberries

def total_cost : ℝ := cost_peaches + cost_apples + cost_blueberries

theorem michael_total_cost :
  total_cost = 51.0 := by
  sorry

end michael_total_cost_l1383_138374


namespace james_total_riding_time_including_rest_stop_l1383_138314

theorem james_total_riding_time_including_rest_stop :
  let distance1 := 40 -- miles
  let speed1 := 16 -- miles per hour
  let distance2 := 40 -- miles
  let speed2 := 20 -- miles per hour
  let rest_stop := 20 -- minutes
  let rest_stop_in_hours := rest_stop / 60 -- convert to hours
  let time1 := distance1 / speed1 -- time for the first part
  let time2 := distance2 / speed2 -- time for the second part
  let total_time := time1 + rest_stop_in_hours + time2 -- total time including rest
  total_time = 4.83 :=
by
  sorry

end james_total_riding_time_including_rest_stop_l1383_138314


namespace right_triangle_inequality_l1383_138378

theorem right_triangle_inequality {a b c : ℝ} (h : c^2 = a^2 + b^2) : 
  a + b ≤ c * Real.sqrt 2 :=
sorry

end right_triangle_inequality_l1383_138378


namespace div_by_73_l1383_138385

theorem div_by_73 (n : ℕ) (h : 0 < n) : (2^(3*n + 6) + 3^(4*n + 2)) % 73 = 0 := sorry

end div_by_73_l1383_138385


namespace no_sum_2015_l1383_138376

theorem no_sum_2015 (x a : ℤ) : 3 * x + 3 * a ≠ 2015 := by
  sorry

end no_sum_2015_l1383_138376


namespace bacon_calories_percentage_l1383_138330

-- Mathematical statement based on the problem
theorem bacon_calories_percentage :
  ∀ (total_sandwich_calories : ℕ) (number_of_bacon_strips : ℕ) (calories_per_strip : ℕ),
    total_sandwich_calories = 1250 →
    number_of_bacon_strips = 2 →
    calories_per_strip = 125 →
    (number_of_bacon_strips * calories_per_strip) * 100 / total_sandwich_calories = 20 :=
by
  intros total_sandwich_calories number_of_bacon_strips calories_per_strip h1 h2 h3 
  sorry

end bacon_calories_percentage_l1383_138330


namespace candy_bar_cost_correct_l1383_138325

def initial_amount : ℕ := 4
def remaining_amount : ℕ := 3
def candy_bar_cost : ℕ := initial_amount - remaining_amount

theorem candy_bar_cost_correct : candy_bar_cost = 1 := by
  unfold candy_bar_cost
  sorry

end candy_bar_cost_correct_l1383_138325


namespace two_real_roots_opposite_signs_l1383_138379

theorem two_real_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, (a * x^2 - (a + 3) * x + 2 = 0) ∧ (a * y^2 - (a + 3) * y + 2 = 0) ∧ (x * y < 0)) ↔ (a < 0) :=
by
  sorry

end two_real_roots_opposite_signs_l1383_138379


namespace hot_peppers_percentage_correct_l1383_138308

def sunday_peppers : ℕ := 7
def monday_peppers : ℕ := 12
def tuesday_peppers : ℕ := 14
def wednesday_peppers : ℕ := 12
def thursday_peppers : ℕ := 5
def friday_peppers : ℕ := 18
def saturday_peppers : ℕ := 12
def non_hot_peppers : ℕ := 64

def total_peppers : ℕ := sunday_peppers + monday_peppers + tuesday_peppers + wednesday_peppers + thursday_peppers + friday_peppers + saturday_peppers
def hot_peppers : ℕ := total_peppers - non_hot_peppers
def hot_peppers_percentage : ℕ := (hot_peppers * 100) / total_peppers

theorem hot_peppers_percentage_correct : hot_peppers_percentage = 20 := 
by 
  sorry

end hot_peppers_percentage_correct_l1383_138308


namespace people_eat_only_vegetarian_l1383_138337

def number_of_people_eat_only_veg (total_veg : ℕ) (both_veg_nonveg : ℕ) : ℕ :=
  total_veg - both_veg_nonveg

theorem people_eat_only_vegetarian
  (total_veg : ℕ) (both_veg_nonveg : ℕ)
  (h1 : total_veg = 28)
  (h2 : both_veg_nonveg = 12)
  : number_of_people_eat_only_veg total_veg both_veg_nonveg = 16 := by
  sorry

end people_eat_only_vegetarian_l1383_138337


namespace problem_theorem_l1383_138397

theorem problem_theorem (x y z : ℤ) 
  (h1 : x = 10 * y + 3)
  (h2 : 2 * x = 21 * y + 1)
  (h3 : 3 * x = 5 * z + 2) : 
  11 * y - x + 7 * z = 219 := 
by
  sorry

end problem_theorem_l1383_138397


namespace smallest_next_divisor_l1383_138390

theorem smallest_next_divisor (n : ℕ) (hn : n % 2 = 0) (h4d : 1000 ≤ n ∧ n < 10000) (hdiv : 221 ∣ n) : 
  ∃ (d : ℕ), d = 238 ∧ 221 < d ∧ d ∣ n :=
by
  sorry

end smallest_next_divisor_l1383_138390


namespace quadratic_ineq_solutions_l1383_138327

theorem quadratic_ineq_solutions (c : ℝ) (h : c > 0) : c < 16 ↔ ∀ x : ℝ, x^2 - 8 * x + c < 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x = x1 ∨ x = x2 :=
by
  sorry

end quadratic_ineq_solutions_l1383_138327


namespace no_linear_factor_with_integer_coefficients_l1383_138342

def expression (x y z : ℤ) : ℤ :=
  x^2 - y^2 - z^2 + 3 * y * z + x + 2 * y - z

theorem no_linear_factor_with_integer_coefficients:
  ¬ ∃ (a b c d : ℤ), a ≠ 0 ∧ 
                      ∀ (x y z : ℤ), 
                        expression x y z = a * x + b * y + c * z + d := by
  sorry

end no_linear_factor_with_integer_coefficients_l1383_138342


namespace max_yellow_apples_can_take_max_total_apples_can_take_l1383_138315

structure Basket :=
  (total_apples : ℕ)
  (green_apples : ℕ)
  (yellow_apples : ℕ)
  (red_apples : ℕ)
  (green_lt_yellow : green_apples < yellow_apples)
  (yellow_lt_red : yellow_apples < red_apples)

def basket_conditions : Basket :=
  { total_apples := 44,
    green_apples := 11,
    yellow_apples := 14,
    red_apples := 19,
    green_lt_yellow := sorry,  -- 11 < 14
    yellow_lt_red := sorry }   -- 14 < 19

theorem max_yellow_apples_can_take : basket_conditions.yellow_apples = 14 := sorry

theorem max_total_apples_can_take : basket_conditions.green_apples 
                                     + basket_conditions.yellow_apples 
                                     + (basket_conditions.red_apples - 2) = 42 := sorry

end max_yellow_apples_can_take_max_total_apples_can_take_l1383_138315


namespace largest_possible_k_satisfies_triangle_condition_l1383_138380

theorem largest_possible_k_satisfies_triangle_condition :
  ∃ k : ℕ, 
    k = 2009 ∧ 
    ∀ (b r w : Fin 2009 → ℝ), 
    (∀ i : Fin 2009, i ≤ i.succ → b i ≤ b i.succ ∧ r i ≤ r i.succ ∧ w i ≤ w i.succ) → 
    (∃ (j : Fin 2009), 
      b j + r j > w j ∧ b j + w j > r j ∧ r j + w j > b j) :=
sorry

end largest_possible_k_satisfies_triangle_condition_l1383_138380


namespace determine_digit_square_l1383_138310

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_palindrome (n : ℕ) : Prop :=
  let d1 := (n / 100000) % 10
  let d2 := (n / 10000) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 100) % 10
  let d5 := (n / 10) % 10
  let d6 := n % 10
  d1 = d6 ∧ d2 = d5 ∧ d3 = d4

def is_multiple_of_6 (n : ℕ) : Prop := is_even (n % 10) ∧ is_divisible_by_3 (List.sum (Nat.digits 10 n))

theorem determine_digit_square :
  ∃ (square : ℕ),
  (is_palindrome (53700000 + square * 10 + 735) ∧ is_multiple_of_6 (53700000 + square * 10 + 735)) ∧ square = 6 := by
  sorry

end determine_digit_square_l1383_138310


namespace distance_upstream_l1383_138388

/-- Proof that the distance a man swims upstream is 18 km given certain conditions. -/
theorem distance_upstream (c : ℝ) (h1 : 54 / (12 + c) = 3) (h2 : 12 - c = 6) : (12 - c) * 3 = 18 :=
by
  sorry

end distance_upstream_l1383_138388


namespace remainder_of_2n_l1383_138334

theorem remainder_of_2n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := 
sorry

end remainder_of_2n_l1383_138334


namespace base_number_is_two_l1383_138382

theorem base_number_is_two (a : ℝ) (x : ℕ) (h1 : x = 14) (h2 : a^x - a^(x - 2) = 3 * a^12) : a = 2 := by
  sorry

end base_number_is_two_l1383_138382


namespace max_S_possible_l1383_138360

theorem max_S_possible (nums : List ℝ) (h_nums_in_bound : ∀ n ∈ nums, 0 ≤ n ∧ n ≤ 1) (h_sum_leq_253_div_12 : nums.sum ≤ 253 / 12) :
  ∃ (A B : List ℝ), (∀ x ∈ A, x ∈ nums) ∧ (∀ y ∈ B, y ∈ nums) ∧ A.union B = nums ∧ A.sum ≤ 11 ∧ B.sum ≤ 11 :=
sorry

end max_S_possible_l1383_138360


namespace expansion_coefficient_l1383_138305

theorem expansion_coefficient (x : ℝ) (h : x ≠ 0): 
  (∃ r : ℕ, (7 - (3 / 2 : ℝ) * r = 1) ∧ Nat.choose 7 r = 35) := 
  sorry

end expansion_coefficient_l1383_138305


namespace find_x_value_l1383_138389

noncomputable def floor_plus_2x_eq_33 (x : ℝ) : Prop :=
  ∃ n : ℤ, ⌊x⌋ = n ∧ n + 2 * x = 33 ∧  (0 : ℝ) ≤ x - n ∧ x - n < 1

theorem find_x_value : ∀ x : ℝ, floor_plus_2x_eq_33 x → x = 11 :=
by
  intro x
  intro h
  -- Proof skipped, included as 'sorry' to compile successfully.
  sorry

end find_x_value_l1383_138389


namespace seq_integer_l1383_138302

theorem seq_integer (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 1) (h3 : a 3 = 249)
(h_rec : ∀ n, a (n + 3) = (1991 + a (n + 2) * a (n + 1)) / a n) :
∀ n, ∃ b : ℤ, a n = b :=
by
  sorry

end seq_integer_l1383_138302


namespace case1_equiv_case2_equiv_determine_case_l1383_138367

theorem case1_equiv (a c x : ℝ) (hc : c ≠ 0) (hx : x ≠ 0) : 
  ((x + a) / (x + c) = a / c) ↔ (a = c) :=
by sorry

theorem case2_equiv (b d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) : 
  (b / d = b / d) :=
by sorry

theorem determine_case (a b c d x : ℝ) (hc : c ≠ 0) (hx : x ≠ 0) (hb : b ≠ 0) (hd : d ≠ 0) :
  ¬((x + a) / (x + c) = a / c) ∧ (b / d = b / d) :=
by sorry

end case1_equiv_case2_equiv_determine_case_l1383_138367


namespace matrix_subtraction_l1383_138354

-- Define matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 4, -3 ],
  ![ 2,  8 ]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 1,  5 ],
  ![ -3,  6 ]
]

-- Define the result matrix as given in the problem
def result : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![ 3, -8 ],
  ![ 5,  2 ]
]

-- The theorem to prove
theorem matrix_subtraction : A - B = result := 
by 
  sorry

end matrix_subtraction_l1383_138354


namespace composite_infinitely_many_l1383_138326

theorem composite_infinitely_many (t : ℕ) (ht : t ≥ 2) :
  ∃ n : ℕ, n = 3 ^ (2 ^ t) - 2 ^ (2 ^ t) ∧ (3 ^ (n - 1) - 2 ^ (n - 1)) % n = 0 :=
by
  use 3 ^ (2 ^ t) - 2 ^ (2 ^ t)
  sorry 

end composite_infinitely_many_l1383_138326


namespace fraction_is_meaningful_l1383_138353

theorem fraction_is_meaningful (x : ℝ) : x ≠ 1 ↔ ∃ y : ℝ, y = 8 / (x - 1) :=
by
  sorry

end fraction_is_meaningful_l1383_138353


namespace intersection_M_N_l1383_138320

noncomputable def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def N : Set ℝ := {x | abs x < 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l1383_138320


namespace gwen_money_difference_l1383_138383

theorem gwen_money_difference:
  let money_from_grandparents : ℕ := 15
  let money_from_uncle : ℕ := 8
  money_from_grandparents - money_from_uncle = 7 :=
by
  sorry

end gwen_money_difference_l1383_138383


namespace exists_powers_of_7_difference_div_by_2021_l1383_138328

theorem exists_powers_of_7_difference_div_by_2021 :
  ∃ n m : ℕ, n > m ∧ 2021 ∣ (7^n - 7^m) := 
by
  sorry

end exists_powers_of_7_difference_div_by_2021_l1383_138328


namespace intersection_A_B_l1383_138313

def A : Set ℝ := {x | x * (x - 4) < 0}
def B : Set ℝ := {0, 1, 5}

theorem intersection_A_B : (A ∩ B) = {1} := by
  sorry

end intersection_A_B_l1383_138313


namespace std_deviation_above_l1383_138370

variable (mean : ℝ) (std_dev : ℝ) (score1 : ℝ) (score2 : ℝ)
variable (n1 : ℝ) (n2 : ℝ)

axiom h_mean : mean = 74
axiom h_std1 : score1 = 58
axiom h_std2 : score2 = 98
axiom h_cond1 : score1 = mean - n1 * std_dev
axiom h_cond2 : n1 = 2

theorem std_deviation_above (mean std_dev score1 score2 n1 n2 : ℝ)
  (h_mean : mean = 74)
  (h_std1 : score1 = 58)
  (h_std2 : score2 = 98)
  (h_cond1 : score1 = mean - n1 * std_dev)
  (h_cond2 : n1 = 2) :
  n2 = (score2 - mean) / std_dev :=
sorry

end std_deviation_above_l1383_138370


namespace lcm_of_9_12_15_is_180_l1383_138369

theorem lcm_of_9_12_15_is_180 :
  Nat.lcm 9 (Nat.lcm 12 15) = 180 :=
by
  sorry

end lcm_of_9_12_15_is_180_l1383_138369


namespace find_b_l1383_138348

theorem find_b (a b : ℤ) (h1 : 3 * a + 1 = 1) (h2 : b - a = 2) : b = 2 := by
  sorry

end find_b_l1383_138348
