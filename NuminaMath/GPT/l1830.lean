import Mathlib

namespace sqrt_inequality_l1830_183084

theorem sqrt_inequality : (Real.sqrt 6 + Real.sqrt 7) > (2 * Real.sqrt 2 + Real.sqrt 5) :=
by {
  sorry
}

end sqrt_inequality_l1830_183084


namespace no_family_of_lines_exists_l1830_183038

theorem no_family_of_lines_exists (k : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (∀ n, (1 : ℝ) = k n * (1 : ℝ) + (1 - k n)) ∧
  (∀ n, k (n + 1) = a n - b n ∧ a n = 1 - 1 / k n ∧ b n = 1 - k n) ∧
  (∀ n, k n * k (n + 1) ≥ 0) →
  False :=
by
  sorry

end no_family_of_lines_exists_l1830_183038


namespace tiles_needed_l1830_183033

def floor9ₓ12_ft : Type := {l : ℕ × ℕ // l = (9, 12)}
def tile4ₓ6_inch : Type := {l : ℕ × ℕ // l = (4, 6)}

theorem tiles_needed (floor : floor9ₓ12_ft) (tile : tile4ₓ6_inch) : 
  ∃ tiles : ℕ, tiles = 648 :=
sorry

end tiles_needed_l1830_183033


namespace moles_of_CO2_formed_l1830_183055

-- Definitions based on the conditions provided
def moles_HNO3 := 2
def moles_NaHCO3 := 2
def balanced_eq (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) : Prop :=
  HNO3 = NaHCO3 ∧ NaNO3 = NaHCO3 ∧ CO2 = NaHCO3 ∧ H2O = NaHCO3

-- Lean Proposition: Prove that 2 moles of CO2 are formed
theorem moles_of_CO2_formed :
  balanced_eq moles_HNO3 moles_NaHCO3 moles_HNO3 moles_HNO3 moles_HNO3 →
  ∃ CO2, CO2 = 2 :=
by
  sorry

end moles_of_CO2_formed_l1830_183055


namespace boat_travel_time_l1830_183012

noncomputable def total_travel_time (stream_speed boat_speed distance_AB : ℝ) : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let distance_BC := distance_AB / 2
  (distance_AB / downstream_speed) + (distance_BC / upstream_speed)

theorem boat_travel_time :
  total_travel_time 4 14 180 = 19 :=
by
  sorry

end boat_travel_time_l1830_183012


namespace car_travel_speed_l1830_183025

theorem car_travel_speed (v : ℝ) : 
  (1 / 60) * 3600 + 5 = (1 / v) * 3600 → v = 65 := 
by
  intros h
  sorry

end car_travel_speed_l1830_183025


namespace pair_with_15_is_47_l1830_183091

theorem pair_with_15_is_47 (numbers : Set ℕ) (k : ℕ) 
  (h : numbers = {49, 29, 9, 40, 22, 15, 53, 33, 13, 47}) 
  (pair_sum_eq_k : ∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → (a, b) ≠ (15, 15) → a + b = k) : 
  ∃ (k : ℕ), 15 + 47 = k := 
sorry

end pair_with_15_is_47_l1830_183091


namespace monomial_k_add_n_l1830_183054

variable (k n : ℤ)

-- Conditions
def is_monomial_coefficient (k : ℤ) : Prop := -k = 5
def is_monomial_degree (n : ℤ) : Prop := n + 1 = 7

-- Theorem to prove
theorem monomial_k_add_n (hk : is_monomial_coefficient k) (hn : is_monomial_degree n) : k + n = 1 :=
by
  sorry

end monomial_k_add_n_l1830_183054


namespace fraction_defined_iff_l1830_183065

theorem fraction_defined_iff (x : ℝ) : (x + 2 ≠ 0) ↔ (x ≠ -2) :=
by sorry

end fraction_defined_iff_l1830_183065


namespace find_number_l1830_183095

theorem find_number (x : ℤ) (h1 : x - 2 + 4 = 9) : x = 7 :=
by
  sorry

end find_number_l1830_183095


namespace total_men_employed_l1830_183074

/--
A work which could be finished in 11 days was finished 3 days earlier 
after 10 more men joined. Prove that the total number of men employed 
to finish the work earlier is 37.
-/
theorem total_men_employed (x : ℕ) (h1 : 11 * x = 8 * (x + 10)) : x = 27 ∧ 27 + 10 = 37 := by
  sorry

end total_men_employed_l1830_183074


namespace min_trams_spy_sees_l1830_183017

/-- 
   Vasya stood at a bus stop for some time and saw 1 bus and 2 trams.
   Buses run every hour.
   After Vasya left, a spy stood at the bus stop for 10 hours and saw 10 buses.
   Given these conditions, the minimum number of trams that the spy could have seen is 5.
-/
theorem min_trams_spy_sees (bus_interval tram_interval : ℕ) 
  (vasya_buses vasya_trams spy_buses spy_hours min_trams : ℕ) 
  (h1 : bus_interval = 1)
  (h2 : vasya_buses = 1)
  (h3 : vasya_trams = 2)
  (h4 : spy_buses = spy_hours)
  (h5 : spy_buses = 10)
  (h6 : spy_hours = 10)
  (h7 : ∀ t : ℕ, t * tram_interval ≤ 2 → 2 * bus_interval ≤ 2)
  (h8 : min_trams = 5) :
  min_trams = 5 := 
sorry

end min_trams_spy_sees_l1830_183017


namespace find_second_number_l1830_183019

theorem find_second_number (x y : ℤ) (h1 : x = -63) (h2 : (2 + y + x) / 3 = 5) : y = 76 :=
sorry

end find_second_number_l1830_183019


namespace hydrogen_atoms_in_compound_l1830_183088

-- Define atoms and their weights
def C_weight : ℕ := 12
def H_weight : ℕ := 1
def O_weight : ℕ := 16

-- Number of each atom in the compound and total molecular weight
def num_C : ℕ := 4
def num_O : ℕ := 1
def total_weight : ℕ := 65

-- Total mass of carbon and oxygen in the compound
def mass_C_O : ℕ := (num_C * C_weight) + (num_O * O_weight)

-- Mass and number of hydrogen atoms in the compound
def mass_H : ℕ := total_weight - mass_C_O
def num_H : ℕ := mass_H / H_weight

theorem hydrogen_atoms_in_compound : num_H = 1 := by
  sorry

end hydrogen_atoms_in_compound_l1830_183088


namespace wood_blocks_after_days_l1830_183097

-- Defining the known conditions
def blocks_per_tree : Nat := 3
def trees_per_day : Nat := 2
def days : Nat := 5

-- Stating the theorem to prove the total number of blocks of wood after 5 days
theorem wood_blocks_after_days : blocks_per_tree * trees_per_day * days = 30 :=
by
  sorry

end wood_blocks_after_days_l1830_183097


namespace smallest_three_digit_multiple_of_17_l1830_183083

theorem smallest_three_digit_multiple_of_17 : 
    ∃ (n : ℤ), n ≥ 100 ∧ n < 1000 ∧ (∃ (k : ℤ), n = 17 * k) ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l1830_183083


namespace opposite_of_neg_abs_is_positive_two_l1830_183048

theorem opposite_of_neg_abs_is_positive_two : -(abs (-2)) = -2 :=
by sorry

end opposite_of_neg_abs_is_positive_two_l1830_183048


namespace find_remainder_l1830_183043

theorem find_remainder (G : ℕ) (Q1 Q2 R1 : ℕ) (hG : G = 127) (h1 : 1661 = G * Q1 + R1) (h2 : 2045 = G * Q2 + 13) : R1 = 10 :=
by
  sorry

end find_remainder_l1830_183043


namespace smallest_addition_to_make_multiple_of_5_l1830_183002

theorem smallest_addition_to_make_multiple_of_5 : ∃ k : ℕ, k > 0 ∧ (729 + k) % 5 = 0 ∧ k = 1 := sorry

end smallest_addition_to_make_multiple_of_5_l1830_183002


namespace abs_eq_zero_sum_is_neg_two_l1830_183090

theorem abs_eq_zero_sum_is_neg_two (x y : ℝ) (h : |x - 1| + |y + 3| = 0) : x + y = -2 := 
by 
  sorry

end abs_eq_zero_sum_is_neg_two_l1830_183090


namespace village_current_population_l1830_183003

theorem village_current_population (initial_population : ℕ) (ten_percent_die : ℕ)
  (twenty_percent_leave : ℕ) : 
  initial_population = 4399 →
  ten_percent_die = initial_population / 10 →
  twenty_percent_leave = (initial_population - ten_percent_die) / 5 →
  (initial_population - ten_percent_die) - twenty_percent_leave = 3167 :=
sorry

end village_current_population_l1830_183003


namespace value_range_neg_x_squared_l1830_183007

theorem value_range_neg_x_squared:
  (∀ y, (-9 ≤ y ∧ y ≤ 0) ↔ ∃ x, (-3 ≤ x ∧ x ≤ 1) ∧ y = -x^2) :=
by
  sorry

end value_range_neg_x_squared_l1830_183007


namespace zero_function_unique_l1830_183022

theorem zero_function_unique 
  (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), f (x ^ (42 ^ 42) + y) = f (x ^ 3 + 2 * y) + f (x ^ 12)) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end zero_function_unique_l1830_183022


namespace transformation_correct_l1830_183073

theorem transformation_correct (a b : ℝ) (h₁ : 3 * a = 2 * b) (h₂ : a ≠ 0) (h₃ : b ≠ 0) :
  a / 2 = b / 3 :=
sorry

end transformation_correct_l1830_183073


namespace soccer_most_students_l1830_183075

def sports := ["hockey", "basketball", "soccer", "volleyball", "badminton"]
def num_students (sport : String) : Nat :=
  match sport with
  | "hockey" => 30
  | "basketball" => 35
  | "soccer" => 50
  | "volleyball" => 20
  | "badminton" => 25
  | _ => 0

theorem soccer_most_students : ∀ sport ∈ sports, num_students "soccer" ≥ num_students sport := by
  sorry

end soccer_most_students_l1830_183075


namespace triangle_side_length_l1830_183004

theorem triangle_side_length {A B C : Type*} 
  (a b : ℝ) (S : ℝ) (ha : a = 4) (hb : b = 5) (hS : S = 5 * Real.sqrt 3) :
  ∃ c : ℝ, c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
by
  sorry

end triangle_side_length_l1830_183004


namespace tangent_line_eq_l1830_183057

open Real

noncomputable def f (x : ℝ) : ℝ := exp x * log x

theorem tangent_line_eq (x y : ℝ) (h : x = 1 ∧ y = 0) :
  ∃ m b, (∀ t, y = m * (t - 1) + b) ∧ (f x = y) ∧ (m = exp 1) ∧ (b = -exp 1) :=
by
  sorry

end tangent_line_eq_l1830_183057


namespace arrangement_count_27_arrangement_count_26_l1830_183052

open Int

def valid_arrangement_count (n : ℕ) : ℕ :=
  if n = 27 then 14 else if n = 26 then 105 else 0

theorem arrangement_count_27 : valid_arrangement_count 27 = 14 :=
  by
    sorry

theorem arrangement_count_26 : valid_arrangement_count 26 = 105 :=
  by
    sorry

end arrangement_count_27_arrangement_count_26_l1830_183052


namespace max_value_of_f_on_interval_l1830_183069

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x - 2

theorem max_value_of_f_on_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ 1 :=
by
  sorry

end max_value_of_f_on_interval_l1830_183069


namespace remainder_of_m_div_1000_l1830_183039

   -- Define the set T
   def T : Set ℕ := {n | 1 ≤ n ∧ n ≤ 12}

   -- Define the computation of m
   noncomputable def m : ℕ := (3^12 - 2 * 2^12 + 1) / 2

   -- Statement for the proof problem
   theorem remainder_of_m_div_1000 : m % 1000 = 625 := by
     sorry
   
end remainder_of_m_div_1000_l1830_183039


namespace negation_of_existence_proposition_l1830_183040

theorem negation_of_existence_proposition :
  ¬ (∃ x : ℝ, x^2 + 2*x - 8 = 0) ↔ ∀ x : ℝ, x^2 + 2*x - 8 ≠ 0 := by
  sorry

end negation_of_existence_proposition_l1830_183040


namespace line_intersection_x_value_l1830_183056

theorem line_intersection_x_value :
  let line1 (x : ℝ) := 3 * x + 14
  let line2 (x : ℝ) (y : ℝ) := 5 * x - 2 * y = 40
  ∃ x : ℝ, ∃ y : ℝ, (line1 x = y) ∧ (line2 x y) ∧ (x = -68) :=
by
  sorry

end line_intersection_x_value_l1830_183056


namespace totalLemonProductionIn5Years_l1830_183047

-- Definition of a normal lemon tree's production rate
def normalLemonProduction : ℕ := 60

-- Definition of the percentage increase for Jim's lemon trees (50%)
def percentageIncrease : ℕ := 50

-- Calculate Jim's lemon tree production per year
def jimLemonProduction : ℕ := normalLemonProduction * (100 + percentageIncrease) / 100

-- Calculate the total number of trees in Jim's grove
def treesInGrove : ℕ := 50 * 30

-- Calculate the total lemon production by Jim's grove in one year
def annualLemonProduction : ℕ := treesInGrove * jimLemonProduction

-- Calculate the total lemon production by Jim's grove in 5 years
def fiveYearLemonProduction : ℕ := 5 * annualLemonProduction

-- Theorem: Prove that the total lemon production in 5 years is 675000
theorem totalLemonProductionIn5Years : fiveYearLemonProduction = 675000 := by
  -- Proof needs to be filled in
  sorry

end totalLemonProductionIn5Years_l1830_183047


namespace total_matches_correct_total_points_earthlings_correct_total_players_is_square_l1830_183087

-- Definitions
variables (t a : ℕ)

-- Part (a): Total number of matches
def total_matches : ℕ := (t + a) * (t + a - 1) / 2

-- Part (b): Total points of the Earthlings
def total_points_earthlings : ℕ := (t * (t - 1)) / 2 + (a * (a - 1)) / 2

-- Part (c): Total number of players is a perfect square
def is_total_players_square : Prop := ∃ k : ℕ, (t + a) = k * k

-- Lean statements
theorem total_matches_correct : total_matches t a = (t + a) * (t + a - 1) / 2 := 
by sorry

theorem total_points_earthlings_correct : total_points_earthlings t a = (t * (t - 1)) / 2 + (a * (a - 1)) / 2 := 
by sorry

theorem total_players_is_square : is_total_players_square t a := by sorry

end total_matches_correct_total_points_earthlings_correct_total_players_is_square_l1830_183087


namespace students_shared_cost_l1830_183020

theorem students_shared_cost (P n : ℕ) (h_price_range: 100 ≤ P ∧ P ≤ 120)
  (h_div1: P % n = 0) (h_div2: P % (n - 2) = 0) (h_extra_cost: P / n + 1 = P / (n - 2)) : n = 14 := by
  sorry

end students_shared_cost_l1830_183020


namespace realNumbersGreaterThan8IsSet_l1830_183070

-- Definitions based on conditions:
def verySmallNumbers : Type := {x : ℝ // sorry} -- Need to define what very small numbers would be
def interestingBooks : Type := sorry -- Need to define what interesting books would be
def realNumbersGreaterThan8 : Set ℝ := { x : ℝ | x > 8 }
def tallPeople : Type := sorry -- Need to define what tall people would be

-- Main theorem: Real numbers greater than 8 can form a set
theorem realNumbersGreaterThan8IsSet : Set ℝ :=
  realNumbersGreaterThan8

end realNumbersGreaterThan8IsSet_l1830_183070


namespace product_simplification_l1830_183068

variables {a b c : ℝ}

theorem product_simplification (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (ab + bc + ac) * ((ab)⁻¹ + (bc)⁻¹ + (ac)⁻¹)) = 
  ((ab + bc + ac)^2) / (abc) := 
sorry

end product_simplification_l1830_183068


namespace rectangle_cut_l1830_183036

def dimensions_ratio (x y : ℕ) : Prop := ∃ (r : ℚ), x = r * y

theorem rectangle_cut (k m n : ℕ) (hk : ℝ) (hm : ℝ) (hn : ℝ) 
  (h1 : k + m + n = 10) 
  (h2 : k * 9 / 10 = hk)
  (h3 : m * 9 / 10 = hm)
  (h4 : n * 9 / 10 = hn)
  (h5 : hk + hm + hn = 9) :
  ∃ (k' m' n' : ℕ), 
    dimensions_ratio k k' ∧ 
    dimensions_ratio m m' ∧
    dimensions_ratio n n' ∧
    k ≠ m ∧ m ≠ n ∧ k ≠ n :=
sorry

end rectangle_cut_l1830_183036


namespace problem_statement_l1830_183086

def digit_sum (k : ℕ) : ℕ :=
  k.digits 10 |>.sum

theorem problem_statement :
  ∀ n : ℕ, (∃ a b : ℕ, n = digit_sum a ∧ n = digit_sum b ∧ n = digit_sum (a + b)) ↔ (∃ k : ℕ, n = 9 * k) :=
by
  sorry

end problem_statement_l1830_183086


namespace cylinder_original_radius_inch_l1830_183066

theorem cylinder_original_radius_inch (r : ℝ) :
  (∃ r : ℝ, (π * (r + 4)^2 * 3 = π * r^2 * 15) ∧ (r > 0)) →
  r = 1 + Real.sqrt 5 :=
by 
  sorry

end cylinder_original_radius_inch_l1830_183066


namespace asymptote_of_hyperbola_l1830_183000

theorem asymptote_of_hyperbola (x y : ℝ) :
  (x^2 - (y^2 / 4) = 1) → (y = 2 * x ∨ y = -2 * x) := sorry

end asymptote_of_hyperbola_l1830_183000


namespace range_of_a_l1830_183078

open Set

noncomputable def setA (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + a ≥ 0}

theorem range_of_a (a : ℝ) : (1 ∉ setA a) → a < 1 :=
sorry

end range_of_a_l1830_183078


namespace minimum_value_of_g_l1830_183034

noncomputable def g (a b x : ℝ) : ℝ :=
  max (|x + a|) (|x + b|)

theorem minimum_value_of_g (a b : ℝ) (h : a < b) :
  ∃ x : ℝ, g a b x = (b - a) / 2 :=
by
  use - (a + b) / 2
  sorry

end minimum_value_of_g_l1830_183034


namespace quad_root_magnitude_l1830_183011

theorem quad_root_magnitude (m : ℝ) :
  (∃ x : ℝ, x^2 - x + m^2 - 4 = 0 ∧ x = 1) → m = 2 ∨ m = -2 :=
by
  sorry

end quad_root_magnitude_l1830_183011


namespace quadratic_equation_l1830_183018

theorem quadratic_equation (a b c x1 x2 : ℝ) (hx1 : a * x1^2 + b * x1 + c = 0) (hx2 : a * x2^2 + b * x2 + c = 0) :
  ∃ y : ℝ, c * y^2 + b * y + a = 0 := 
sorry

end quadratic_equation_l1830_183018


namespace problem_condition_l1830_183015

theorem problem_condition (x y : ℝ) (h : x^2 + y^2 - x * y = 1) : 
  x + y ≥ -2 ∧ x^2 + y^2 ≤ 2 :=
by
  sorry

end problem_condition_l1830_183015


namespace dice_sum_prob_l1830_183006

theorem dice_sum_prob :
  (3 / 6) * (3 / 6) * (2 / 5) * (1 / 6) * 2 = 13 / 216 :=
by sorry

end dice_sum_prob_l1830_183006


namespace part1_part2_l1830_183016

noncomputable def f : ℝ → ℝ := sorry

variable (x y : ℝ)
variable (hx0 : 0 < x)
variable (hy0 : 0 < y)
variable (hx12 : x < 1 → f x > 0)
variable (hf_half : f (1 / 2) = 1)
variable (hf_mul : f (x * y) = f x + f y)

theorem part1 : (∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2) := sorry

theorem part2 : (∀ x, 3 < x → x < 4 → f (x - 3) > f (1 / x) - 2) := sorry

end part1_part2_l1830_183016


namespace angle_in_second_quadrant_l1830_183009

-- Definitions of conditions
def sin2_pos : Prop := Real.sin 2 > 0
def cos2_neg : Prop := Real.cos 2 < 0

-- Statement of the problem
theorem angle_in_second_quadrant (h1 : sin2_pos) (h2 : cos2_neg) : 
    (∃ α, 0 < α ∧ α < π ∧ P = (Real.sin α, Real.cos α)) :=
by
  sorry

end angle_in_second_quadrant_l1830_183009


namespace area_square_15_cm_l1830_183072

-- Define the side length of the square
def side_length : ℝ := 15

-- Define the area calculation for a square given the side length
def area_of_square (s : ℝ) : ℝ := s * s

-- The theorem statement translating the problem to Lean
theorem area_square_15_cm :
  area_of_square side_length = 225 :=
by
  -- We need to provide proof here, but 'sorry' is used to skip the proof as per instructions
  sorry

end area_square_15_cm_l1830_183072


namespace words_per_page_l1830_183037

theorem words_per_page (p : ℕ) :
  (136 * p) % 203 = 184 % 203 ∧ p ≤ 100 → p = 73 :=
sorry

end words_per_page_l1830_183037


namespace num_apartments_per_floor_l1830_183051

-- Definitions used in the proof
def num_buildings : ℕ := 2
def floors_per_building : ℕ := 12
def doors_per_apartment : ℕ := 7
def total_doors_needed : ℕ := 1008

-- Lean statement to proof the number of apartments per floor
theorem num_apartments_per_floor : 
  (total_doors_needed / (doors_per_apartment * num_buildings * floors_per_building)) = 6 :=
by
  sorry

end num_apartments_per_floor_l1830_183051


namespace combinedTotalSandcastlesAndTowers_l1830_183098

def markSandcastles : Nat := 20
def towersPerMarkSandcastle : Nat := 10
def jeffSandcastles : Nat := 3 * markSandcastles
def towersPerJeffSandcastle : Nat := 5

theorem combinedTotalSandcastlesAndTowers :
  (markSandcastles + markSandcastles * towersPerMarkSandcastle) +
  (jeffSandcastles + jeffSandcastles * towersPerJeffSandcastle) = 580 :=
by
  sorry

end combinedTotalSandcastlesAndTowers_l1830_183098


namespace parabola_directrix_l1830_183079

theorem parabola_directrix (x y : ℝ) (h : y = 16 * x^2) : y = -1/64 :=
sorry

end parabola_directrix_l1830_183079


namespace find_a_l1830_183050

open Set

-- Define set A
def A : Set ℝ := {-1, 1, 3}

-- Define set B in terms of a
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

-- State the theorem
theorem find_a (a : ℝ) (h : A ∩ B a = {3}) : a = 1 :=
sorry

end find_a_l1830_183050


namespace bird_stork_difference_l1830_183094

theorem bird_stork_difference :
  let initial_birds := 3
  let initial_storks := 4
  let additional_birds := 2
  let total_birds := initial_birds + additional_birds
  total_birds - initial_storks = 1 := 
by
  let initial_birds := 3
  let initial_storks := 4
  let additional_birds := 2
  let total_birds := initial_birds + additional_birds
  show total_birds - initial_storks = 1
  sorry

end bird_stork_difference_l1830_183094


namespace girls_in_wind_band_not_string_band_l1830_183031

def M_G : ℕ := 100
def F_G : ℕ := 80
def M_O : ℕ := 80
def F_O : ℕ := 100
def total_students : ℕ := 230
def boys_in_both : ℕ := 60

theorem girls_in_wind_band_not_string_band : (F_G - (total_students - (M_G + F_G + M_O + F_O - boys_in_both - boys_in_both))) = 10 :=
by
  sorry

end girls_in_wind_band_not_string_band_l1830_183031


namespace even_n_of_even_Omega_P_l1830_183024

-- Define the Omega function
def Omega (N : ℕ) : ℕ := 
  N.factors.length

-- Define the polynomial function P
def P (x : ℕ) (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  List.prod (List.map (λ i => x + a i) (List.range n))

theorem even_n_of_even_Omega_P (a : ℕ → ℕ) (n : ℕ)
  (H : ∀ k > 0, Even (Omega (P k a n))) : Even n :=
by
  sorry

end even_n_of_even_Omega_P_l1830_183024


namespace heather_distance_l1830_183010

-- Definitions based on conditions
def distance_from_car_to_entrance (x : ℝ) : ℝ := x
def distance_from_entrance_to_rides (x : ℝ) : ℝ := x
def distance_from_rides_to_car : ℝ := 0.08333333333333333
def total_distance_walked : ℝ := 0.75

-- Lean statement to prove
theorem heather_distance (x : ℝ) (h : distance_from_car_to_entrance x + distance_from_entrance_to_rides x + distance_from_rides_to_car = total_distance_walked) :
  x = 0.33333333333333335 :=
by
  sorry

end heather_distance_l1830_183010


namespace pieces_left_to_place_l1830_183026

noncomputable def total_pieces : ℕ := 300
noncomputable def reyn_pieces : ℕ := 25
noncomputable def rhys_pieces : ℕ := 2 * reyn_pieces
noncomputable def rory_pieces : ℕ := 3 * reyn_pieces
noncomputable def placed_pieces : ℕ := reyn_pieces + rhys_pieces + rory_pieces
noncomputable def remaining_pieces : ℕ := total_pieces - placed_pieces

theorem pieces_left_to_place : remaining_pieces = 150 :=
by sorry

end pieces_left_to_place_l1830_183026


namespace friend_owns_10_bikes_l1830_183021

theorem friend_owns_10_bikes (ignatius_bikes : ℕ) (tires_per_bike : ℕ) (unicycle_tires : ℕ) (tricycle_tires : ℕ) (friend_total_tires : ℕ) :
  ignatius_bikes = 4 →
  tires_per_bike = 2 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_total_tires = 3 * (ignatius_bikes * tires_per_bike) →
  (friend_total_tires - (unicycle_tires + tricycle_tires)) / tires_per_bike = 10 :=
by
  sorry

end friend_owns_10_bikes_l1830_183021


namespace divisibility_of_3_pow_p_minus_2_pow_p_minus_1_l1830_183099

theorem divisibility_of_3_pow_p_minus_2_pow_p_minus_1 (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) : 
  (3^p - 2^p - 1) % (42 * p) = 0 := 
by
  sorry

end divisibility_of_3_pow_p_minus_2_pow_p_minus_1_l1830_183099


namespace proof_problem_l1830_183053

-- Defining lines l1, l2, l3
def l1 (x y : ℝ) : Prop := 3 * x + 4 * y = 2
def l2 (x y : ℝ) : Prop := 2 * x + y = -2
def l3 (x y : ℝ) : Prop := x - 2 * y = 1

-- Point P being the intersection of l1 and l2
def P : ℝ × ℝ := (-2, 2)

-- Definition of the first required line passing through P and the origin
def line_through_P_and_origin (x y : ℝ) : Prop := x + y = 0

-- Definition of the second required line passing through P and perpendicular to l3
def required_line (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- The theorem to prove
theorem proof_problem :
  (∃ x y, l1 x y ∧ l2 x y ∧ (x, y) = P) →
  (∀ x y, (x, y) = P → line_through_P_and_origin x y) ∧
  (∀ x y, (x, y) = P → required_line x y) :=
by
  sorry

end proof_problem_l1830_183053


namespace correct_operations_result_l1830_183071

theorem correct_operations_result (n : ℕ) 
  (h1 : n / 8 - 12 = 32) : (n * 8 + 12 = 2828) :=
sorry

end correct_operations_result_l1830_183071


namespace probability_girls_same_color_l1830_183049

open Classical

noncomputable def probability_same_color_marbles : ℚ :=
(3/6) * (2/5) * (1/4) + (3/6) * (2/5) * (1/4)

theorem probability_girls_same_color :
  probability_same_color_marbles = 1/20 := by
  sorry

end probability_girls_same_color_l1830_183049


namespace arithmetic_progression_x_value_l1830_183093

theorem arithmetic_progression_x_value :
  ∃ x : ℝ, (2 * x - 1) + ((5 * x + 6) - (3 * x + 4)) = (3 * x + 4) + ((3 * x + 4) - (2 * x - 1)) ∧ x = 3 :=
by
  sorry

end arithmetic_progression_x_value_l1830_183093


namespace notAlwaysTriangleInSecondQuadrantAfterReflection_l1830_183059

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  P : Point
  Q : Point
  R : Point

def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

def reflectionOverYEqualsX (p : Point) : Point :=
  { x := p.y, y := p.x }

def reflectTriangleOverYEqualsX (T : Triangle) : Triangle :=
  { P := reflectionOverYEqualsX T.P,
    Q := reflectionOverYEqualsX T.Q,
    R := reflectionOverYEqualsX T.R }

def triangleInSecondQuadrant (T : Triangle) : Prop :=
  isInSecondQuadrant T.P ∧ isInSecondQuadrant T.Q ∧ isInSecondQuadrant T.R

theorem notAlwaysTriangleInSecondQuadrantAfterReflection
  (T : Triangle)
  (h : triangleInSecondQuadrant T)
  : ¬ (triangleInSecondQuadrant (reflectTriangleOverYEqualsX T)) := 
sorry -- Proof not required

end notAlwaysTriangleInSecondQuadrantAfterReflection_l1830_183059


namespace students_in_class_l1830_183046

theorem students_in_class (b g : ℕ) 
  (h1 : b + g = 20)
  (h2 : (b : ℚ) / 20 = (3 : ℚ) / 4 * (g : ℚ) / 20) : 
  b = 12 ∧ g = 8 :=
by
  sorry

end students_in_class_l1830_183046


namespace simple_interest_years_l1830_183005

theorem simple_interest_years (P : ℝ) (difference : ℝ) (N : ℝ) : 
  P = 2300 → difference = 69 → (23 * N = 69) → N = 3 :=
by
  intros hP hdifference heq
  sorry

end simple_interest_years_l1830_183005


namespace ninety_eight_times_ninety_eight_l1830_183060

theorem ninety_eight_times_ninety_eight : 98 * 98 = 9604 :=
by
  sorry

end ninety_eight_times_ninety_eight_l1830_183060


namespace min_value_of_a2_b2_l1830_183077

theorem min_value_of_a2_b2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 4) : 
  ∃ m : ℝ, (∀ x y, x > 0 → y > 0 → x + y = 4 → x^2 + y^2 ≥ m) ∧ m = 8 :=
by
  sorry

end min_value_of_a2_b2_l1830_183077


namespace Aiyanna_more_than_Alyssa_Brady_fewer_than_Aiyanna_Brady_more_than_Alyssa_l1830_183032

-- Defining the number of cookies each person had
def Alyssa_cookies : ℕ := 1523
def Aiyanna_cookies : ℕ := 3720
def Brady_cookies : ℕ := 2265

-- Proving the statements
theorem Aiyanna_more_than_Alyssa : Aiyanna_cookies - Alyssa_cookies = 2197 := by
  sorry

theorem Brady_fewer_than_Aiyanna : Aiyanna_cookies - Brady_cookies = 1455 := by
  sorry

theorem Brady_more_than_Alyssa : Brady_cookies - Alyssa_cookies = 742 := by
  sorry

end Aiyanna_more_than_Alyssa_Brady_fewer_than_Aiyanna_Brady_more_than_Alyssa_l1830_183032


namespace option_D_correct_option_A_incorrect_option_B_incorrect_option_C_incorrect_l1830_183064

-- Define the variables
variables (m : ℤ)

-- State the conditions as hypotheses
theorem option_D_correct (m : ℤ) : 
  (m * (m - 1) = m^2 - m) :=
by {
    -- Proof sketch (not implemented):
    -- Use distributive property to demonstrate that both sides are equal.
    sorry
}

theorem option_A_incorrect (m : ℤ) : 
  ¬ (m^4 + m^3 = m^7) :=
by {
    -- Proof sketch (not implemented):
    -- Demonstrate that exponents can't be added this way when bases are added.
    sorry
}

theorem option_B_incorrect (m : ℤ) : 
  ¬ ((m^4)^3 = m^7) :=
by {
    -- Proof sketch (not implemented):
    -- Show that raising m^4 to the power of 3 results in m^12.
    sorry
}

theorem option_C_incorrect (m : ℤ) : 
  ¬ (2 * m^5 / m^3 = m^2) :=
by {
    -- Proof sketch (not implemented):
    -- Show that dividing results in 2m^2.
    sorry
}

end option_D_correct_option_A_incorrect_option_B_incorrect_option_C_incorrect_l1830_183064


namespace workman_problem_l1830_183041

theorem workman_problem 
  {A B : Type}
  (W : ℕ)
  (RA RB : ℝ)
  (h1 : RA = (1/2) * RB)
  (h2 : RA + RB = W / 14)
  : W / RB = 21 :=
by
  sorry

end workman_problem_l1830_183041


namespace number_of_solutions_is_zero_l1830_183044

theorem number_of_solutions_is_zero : 
  ∀ x : ℝ, (x ≠ 0 ∧ x ≠ 5) → (3 * x^2 - 15 * x) / (x^2 - 5 * x) ≠ x - 2 :=
by
  sorry

end number_of_solutions_is_zero_l1830_183044


namespace option_A_is_correct_l1830_183023

-- Define propositions p and q
variables (p q : Prop)

-- Option A
def isOptionACorrect: Prop := (¬p ∨ ¬q) → (¬p ∧ ¬q)

theorem option_A_is_correct: isOptionACorrect p q := sorry

end option_A_is_correct_l1830_183023


namespace find_unknown_number_l1830_183080

-- Definitions

-- Declaring that we have an inserted number 'a' between 3 and unknown number 'b'
variable (a b : ℕ)

-- Conditions provided in the problem
def arithmetic_sequence_condition (a b : ℕ) : Prop := 
  a - 3 = b - a

def geometric_sequence_condition (a b : ℕ) : Prop :=
  (a - 6) / 3 = b / (a - 6)

-- The theorem statement equivalent to the problem
theorem find_unknown_number (h1 : arithmetic_sequence_condition a b) (h2 : geometric_sequence_condition a b) : b = 27 :=
sorry

end find_unknown_number_l1830_183080


namespace trajectory_of_moving_circle_l1830_183035

-- Define the conditions
def passes_through (M : ℝ × ℝ) (A : ℝ × ℝ) : Prop :=
  M = A

def tangent_to_line (M : ℝ × ℝ) (l : ℝ) : Prop :=
  M.1 = -l

noncomputable def equation_of_trajectory (M : ℝ × ℝ) : Prop :=
  M.2 ^ 2 = 12 * M.1

theorem trajectory_of_moving_circle 
  (M : ℝ × ℝ)
  (A : ℝ × ℝ)
  (l : ℝ)
  (h1 : passes_through M (3, 0))
  (h2 : tangent_to_line M 3)
  : equation_of_trajectory M := 
sorry

end trajectory_of_moving_circle_l1830_183035


namespace systematic_sampling_method_l1830_183014

theorem systematic_sampling_method (k : ℕ) (n : ℕ) 
  (invoice_stubs : ℕ → ℕ) : 
  (k > 0) → 
  (n > 0) → 
  (invoice_stubs 15 = k) → 
  (∀ i : ℕ, invoice_stubs (15 + i * 50) = k + i * 50)
  → (sampling_method = "systematic") :=
by 
  intro h1 h2 h3 h4
  sorry

end systematic_sampling_method_l1830_183014


namespace a_n_geometric_sequence_b_n_general_term_l1830_183061

theorem a_n_geometric_sequence (t : ℝ) (h : t ≠ 0 ∧ t ≠ 1) :
  (∀ n, ∃ r : ℝ, a_n = t^n) :=
sorry

theorem b_n_general_term (t : ℝ) (h1 : t ≠ 0 ∧ t ≠ 1) (h2 : ∀ n, a_n = t^n)
  (h3 : ∃ q : ℝ, q = (2 * t^2 + t) / 2) :
  (∀ n, b_n = (t^(n + 1) * (2 * t + 1)^(n - 1)) / 2^(n - 2)) :=
sorry

end a_n_geometric_sequence_b_n_general_term_l1830_183061


namespace probability_of_black_given_not_white_l1830_183008

variable (total_balls white_balls black_balls red_balls : ℕ)
variable (ball_is_not_white : Prop)

theorem probability_of_black_given_not_white 
  (h1 : total_balls = 10)
  (h2 : white_balls = 5)
  (h3 : black_balls = 3)
  (h4 : red_balls = 2)
  (h5 : ball_is_not_white) :
  (3 : ℚ) / 5 = (black_balls : ℚ) / (total_balls - white_balls) :=
by
  simp only [h1, h2, h3, h4]
  sorry

end probability_of_black_given_not_white_l1830_183008


namespace stone_reaches_bottom_l1830_183058

structure StoneInWater where
  σ : ℝ   -- Density of stone in g/cm³
  d : ℝ   -- Depth of lake in cm
  g : ℝ   -- Acceleration due to gravity in cm/sec²
  σ₁ : ℝ  -- Density of water in g/cm³

noncomputable def time_and_velocity (siw : StoneInWater) : ℝ × ℝ :=
  let g₁ := ((siw.σ - siw.σ₁) / siw.σ) * siw.g
  let t := Real.sqrt ((2 * siw.d) / g₁)
  let v := g₁ * t
  (t, v)

theorem stone_reaches_bottom (siw : StoneInWater)
  (hσ : siw.σ = 2.1)
  (hd : siw.d = 850)
  (hg : siw.g = 980.8)
  (hσ₁ : siw.σ₁ = 1.0) :
  time_and_velocity siw = (1.82, 935) :=
by
  sorry

end stone_reaches_bottom_l1830_183058


namespace original_five_digit_number_l1830_183089

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end original_five_digit_number_l1830_183089


namespace new_person_weight_is_90_l1830_183062

-- Define the weight of the replaced person
def replaced_person_weight : ℝ := 40

-- Define the increase in average weight when the new person replaces the replaced person
def increase_in_average_weight : ℝ := 10

-- Define the increase in total weight as 5 times the increase in average weight
def increase_in_total_weight (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase

-- Define the weight of the new person
def new_person_weight (replaced_w : ℝ) (total_increase : ℝ) : ℝ := replaced_w + total_increase

-- Prove that the weight of the new person is 90 kg
theorem new_person_weight_is_90 :
  new_person_weight replaced_person_weight (increase_in_total_weight 5 increase_in_average_weight) = 90 := 
by 
  -- sorry will skip the proof, as required
  sorry

end new_person_weight_is_90_l1830_183062


namespace paula_paint_coverage_l1830_183029

-- Define the initial conditions
def initial_capacity : ℕ := 36
def lost_cans : ℕ := 4
def reduced_capacity : ℕ := 28

-- Define the proof problem
theorem paula_paint_coverage :
  (initial_capacity - reduced_capacity = lost_cans * (initial_capacity / reduced_capacity)) →
  (reduced_capacity / (initial_capacity / reduced_capacity) = 14) :=
by
  sorry

end paula_paint_coverage_l1830_183029


namespace only_solution_l1830_183030

def phi : ℕ → ℕ := sorry  -- Euler's totient function
def d : ℕ → ℕ := sorry    -- Divisor function

theorem only_solution (n : ℕ) (h1 : n ∣ (phi n)^(d n) + 1) (h2 : ¬ d n ^ 5 ∣ n ^ (phi n) - 1) : n = 2 :=
sorry

end only_solution_l1830_183030


namespace lg_sum_eq_lg_double_diff_l1830_183076

theorem lg_sum_eq_lg_double_diff (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_harmonic : 2 / y = 1 / x + 1 / z) : 
  Real.log (x + z) + Real.log (x - 2 * y + z) = 2 * Real.log (x - z) := 
by
  sorry

end lg_sum_eq_lg_double_diff_l1830_183076


namespace negation_proposition_l1830_183085

theorem negation_proposition :
  (¬ (x ≠ 3 ∧ x ≠ 2) → ¬ (x ^ 2 - 5 * x + 6 ≠ 0)) =
  ((x = 3 ∨ x = 2) → (x ^ 2 - 5 * x + 6 = 0)) :=
by
  sorry

end negation_proposition_l1830_183085


namespace quadratic_real_roots_k_leq_one_l1830_183096

theorem quadratic_real_roots_k_leq_one (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ↔ k ≤ 1 :=
by
  sorry

end quadratic_real_roots_k_leq_one_l1830_183096


namespace simplify_exponentiation_l1830_183045

-- Define the exponents and the base
variables (t : ℕ)

-- Define the expression and expected result
def expr := t^5 * t^2
def expected := t^7

-- State the proof goal
theorem simplify_exponentiation : expr = expected := 
by sorry

end simplify_exponentiation_l1830_183045


namespace factorization_correct_l1830_183067

theorem factorization_correct (x y : ℝ) : x^2 - 4 * y^2 = (x - 2 * y) * (x + 2 * y) :=
by sorry

end factorization_correct_l1830_183067


namespace choose_starters_with_twins_l1830_183092

theorem choose_starters_with_twins :
  let total_players := 12
  let num_starters := 5
  let twins_num := 2
  let total_ways := Nat.choose total_players num_starters
  let without_twins := Nat.choose (total_players - twins_num) num_starters
  total_ways - without_twins = 540 := 
by
  let total_players := 12
  let num_starters := 5
  let twins_num := 2
  let total_ways := Nat.choose total_players num_starters
  let without_twins := Nat.choose (total_players - twins_num) num_starters
  exact Nat.sub_eq_of_eq_add sorry -- here we will need the exact proof steps which we skip

end choose_starters_with_twins_l1830_183092


namespace min_area_of_B_l1830_183082

noncomputable def setA := { p : ℝ × ℝ | abs (p.1 - 2) + abs (p.2 - 3) ≤ 1 }

noncomputable def setB (D E F : ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 + D * p.1 + E * p.2 + F ≤ 0 ∧ D^2 + E^2 - 4 * F > 0 }

theorem min_area_of_B (D E F : ℝ) (h : setA ⊆ setB D E F) : 
  ∃ r : ℝ, (∀ p ∈ setB D E F, p.1^2 + p.2^2 ≤ r^2) ∧ (π * r^2 = 2 * π) :=
sorry

end min_area_of_B_l1830_183082


namespace no_solution_fractional_eq_l1830_183013

theorem no_solution_fractional_eq :
  ¬∃ x : ℝ, (1 - x) / (x - 2) = 1 / (2 - x) + 1 :=
by
  -- The proof is intentionally omitted.
  sorry

end no_solution_fractional_eq_l1830_183013


namespace range_of_m_l1830_183081

open Set

theorem range_of_m (m : ℝ) : 
  (∀ x, (m + 1 ≤ x ∧ x ≤ 2 * m - 1) → (-2 < x ∧ x ≤ 5)) → 
  m ∈ Iic (3 : ℝ) :=
by
  intros h
  sorry

end range_of_m_l1830_183081


namespace mass_percentage_C_in_C6HxO6_indeterminate_l1830_183063

-- Definition of conditions
def mass_percentage_C_in_C6H8O6 : ℚ := 40.91 / 100
def molar_mass_C : ℚ := 12.01
def molar_mass_H : ℚ := 1.01
def molar_mass_O : ℚ := 16.00

-- Formula for molar mass of C6H8O6
def molar_mass_C6H8O6 : ℚ := 6 * molar_mass_C + 8 * molar_mass_H + 6 * molar_mass_O

-- Mass of carbon in C6H8O6 is 40.91% of the total molar mass
def mass_of_C_in_C6H8O6 : ℚ := mass_percentage_C_in_C6H8O6 * molar_mass_C6H8O6

-- Hypothesis: mass percentage of carbon in C6H8O6 is given
axiom hyp_mass_percentage_C_in_C6H8O6 : mass_of_C_in_C6H8O6 = 72.06

-- Proof that we need the value of x to determine the mass percentage of C in C6HxO6
theorem mass_percentage_C_in_C6HxO6_indeterminate (x : ℚ) :
  (molar_mass_C6H8O6 = 176.14) → (mass_of_C_in_C6H8O6 = 72.06) → False :=
by
  sorry

end mass_percentage_C_in_C6HxO6_indeterminate_l1830_183063


namespace Rachel_plant_arrangement_l1830_183027

-- We define Rachel's plants and lamps
inductive Plant : Type
| basil1
| basil2
| aloe
| cactus

inductive Lamp : Type
| white1
| white2
| red1
| red2

def arrangements (plants : List Plant) (lamps : List Lamp) : Nat :=
  -- This would be the function counting all valid arrangements
  -- I'm skipping the implementation
  sorry

def Rachel_arrangement_count : Nat :=
  arrangements [Plant.basil1, Plant.basil2, Plant.aloe, Plant.cactus]
                [Lamp.white1, Lamp.white2, Lamp.red1, Lamp.red2]

theorem Rachel_plant_arrangement : Rachel_arrangement_count = 22 := by
  sorry

end Rachel_plant_arrangement_l1830_183027


namespace solutions_to_h_eq_1_l1830_183001

noncomputable def h (x : ℝ) : ℝ :=
if x ≤ 0 then 5 * x + 10 else 3 * x - 5

theorem solutions_to_h_eq_1 : {x : ℝ | h x = 1} = {-9/5, 2} :=
by
  sorry

end solutions_to_h_eq_1_l1830_183001


namespace min_sum_xyz_l1830_183028

theorem min_sum_xyz (x y z : ℝ) 
  (hx : x ≥ 4) (hy : y ≥ 5) (hz : z ≥ 6) 
  (hxyz : x^2 + y^2 + z^2 ≥ 90) : 
  x + y + z ≥ 16 := 
sorry

end min_sum_xyz_l1830_183028


namespace total_spent_l1830_183042

theorem total_spent (cost_other_toys : ℕ) (cost_lightsaber : ℕ) 
  (h1 : cost_other_toys = 1000) 
  (h2 : cost_lightsaber = 2 * cost_other_toys) : 
  cost_lightsaber + cost_other_toys = 3000 :=
  by
    sorry

end total_spent_l1830_183042
