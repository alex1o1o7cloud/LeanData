import Mathlib

namespace NUMINAMATH_GPT_rhombus_area_l1201_120173

theorem rhombus_area (side d1 : ℝ) (h_side : side = 28) (h_d1 : d1 = 12) : 
  (side = 28 ∧ d1 = 12) →
  ∃ area : ℝ, area = 328.32 := 
by 
  sorry

end NUMINAMATH_GPT_rhombus_area_l1201_120173


namespace NUMINAMATH_GPT_circumference_of_base_of_cone_l1201_120150

theorem circumference_of_base_of_cone (V : ℝ) (h : ℝ) (C : ℝ) (r : ℝ) 
  (h1 : V = 24 * Real.pi) (h2 : h = 6) (h3 : V = (1/3) * Real.pi * r^2 * h) 
  (h4 : r = Real.sqrt 12) : C = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circumference_of_base_of_cone_l1201_120150


namespace NUMINAMATH_GPT_first_person_days_l1201_120149

theorem first_person_days (x : ℝ) (hp : 30 ≥ 0) (ht : 10 ≥ 0) (h_work : 1/x + 1/30 = 1/10) : x = 15 :=
by
  -- Begin by acknowledging the assumptions: hp, ht, and h_work
  sorry

end NUMINAMATH_GPT_first_person_days_l1201_120149


namespace NUMINAMATH_GPT_amy_7_mile_run_time_l1201_120122

-- Define the conditions
variable (rachel_time_per_9_miles : ℕ) (amy_time_per_4_miles : ℕ) (amy_time_per_mile : ℕ) (amy_time_per_7_miles: ℕ)

-- State the conditions
def conditions : Prop :=
  rachel_time_per_9_miles = 36 ∧
  amy_time_per_4_miles = 1 / 3 * rachel_time_per_9_miles ∧
  amy_time_per_mile = amy_time_per_4_miles / 4 ∧
  amy_time_per_7_miles = amy_time_per_mile * 7

-- The main statement to prove
theorem amy_7_mile_run_time (rachel_time_per_9_miles : ℕ) (amy_time_per_4_miles : ℕ) (amy_time_per_mile : ℕ) (amy_time_per_7_miles: ℕ) :
  conditions rachel_time_per_9_miles amy_time_per_4_miles amy_time_per_mile amy_time_per_7_miles → 
  amy_time_per_7_miles = 21 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_amy_7_mile_run_time_l1201_120122


namespace NUMINAMATH_GPT_sum_of_reversed_integers_l1201_120177

-- Definitions of properties and conditions
def reverse_digits (m n : ℕ) : Prop :=
  let to_digits (x : ℕ) : List ℕ := x.digits 10
  to_digits m = (to_digits n).reverse

-- The main theorem statement
theorem sum_of_reversed_integers
  (m n : ℕ)
  (h_rev: reverse_digits m n)
  (h_prod: m * n = 1446921630) :
  m + n = 79497 :=
sorry

end NUMINAMATH_GPT_sum_of_reversed_integers_l1201_120177


namespace NUMINAMATH_GPT_find_a_l1201_120187

theorem find_a (a b : ℤ) (h1 : 4181 * a + 2584 * b = 0) (h2 : 2584 * a + 1597 * b = -1) : a = 1597 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1201_120187


namespace NUMINAMATH_GPT_find_blue_balloons_l1201_120146

theorem find_blue_balloons (purple_balloons : ℕ) (left_balloons : ℕ) (total_balloons : ℕ) (blue_balloons : ℕ) :
  purple_balloons = 453 →
  left_balloons = 378 →
  total_balloons = left_balloons * 2 →
  total_balloons = purple_balloons + blue_balloons →
  blue_balloons = 303 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_blue_balloons_l1201_120146


namespace NUMINAMATH_GPT_prime_power_implies_one_l1201_120144

theorem prime_power_implies_one (p : ℕ) (a : ℤ) (n : ℕ) (h_prime : Nat.Prime p) (h_eq : 2^p + 3^p = a^n) :
  n = 1 :=
sorry

end NUMINAMATH_GPT_prime_power_implies_one_l1201_120144


namespace NUMINAMATH_GPT_spring_length_relationship_maximum_mass_l1201_120136

theorem spring_length_relationship (x y : ℝ) : 
  (y = 0.5 * x + 12) ↔ y = 12 + 0.5 * x := 
by sorry

theorem maximum_mass (x y : ℝ) : 
  (y = 0.5 * x + 12) → (y ≤ 20) → (x ≤ 16) :=
by sorry

end NUMINAMATH_GPT_spring_length_relationship_maximum_mass_l1201_120136


namespace NUMINAMATH_GPT_right_triangle_of_condition_l1201_120196

theorem right_triangle_of_condition
  (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h_trig : Real.sin γ - Real.cos α = Real.cos β) :
  (α = 90) ∨ (β = 90) :=
sorry

end NUMINAMATH_GPT_right_triangle_of_condition_l1201_120196


namespace NUMINAMATH_GPT_sum_of_smallest_two_consecutive_numbers_l1201_120180

theorem sum_of_smallest_two_consecutive_numbers (n : ℕ) (h : n * (n + 1) * (n + 2) = 210) : n + (n + 1) = 11 :=
sorry

end NUMINAMATH_GPT_sum_of_smallest_two_consecutive_numbers_l1201_120180


namespace NUMINAMATH_GPT_jenny_change_l1201_120197

/-!
## Problem statement

Jenny is printing 7 copies of her 25-page essay. It costs $0.10 to print one page.
She also buys 7 pens, each costing $1.50. If she pays with $40, calculate the change she should get.
-/

def cost_per_page : ℝ := 0.10
def pages_per_copy : ℕ := 25
def num_copies : ℕ := 7
def cost_per_pen : ℝ := 1.50
def num_pens : ℕ := 7
def amount_paid : ℝ := 40.0

def total_pages : ℕ := num_copies * pages_per_copy

def cost_printing : ℝ := total_pages * cost_per_page
def cost_pens : ℝ := num_pens * cost_per_pen

def total_cost : ℝ := cost_printing + cost_pens

theorem jenny_change : amount_paid - total_cost = 12 := by
  -- proof here
  sorry

end NUMINAMATH_GPT_jenny_change_l1201_120197


namespace NUMINAMATH_GPT_ladder_length_l1201_120118

/-- The length of the ladder leaning against a wall when it forms
    a 60 degree angle with the ground and the foot of the ladder 
    is 9.493063650744542 m from the wall is 18.986127301489084 m. -/
theorem ladder_length (L : ℝ) (adjacent : ℝ) (θ : ℝ) (cosθ : ℝ) :
  θ = Real.pi / 3 ∧ adjacent = 9.493063650744542 ∧ cosθ = Real.cos θ →
  L = 18.986127301489084 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ladder_length_l1201_120118


namespace NUMINAMATH_GPT_ratio_of_speeds_l1201_120130

theorem ratio_of_speeds (a b v1 v2 S : ℝ)
  (h1 : S = a * (v1 + v2))
  (h2 : S = b * (v1 - v2)) :
  v2 / v1 = (a + b) / (b - a) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l1201_120130


namespace NUMINAMATH_GPT_length_of_field_l1201_120141

theorem length_of_field (width : ℕ) (distance_covered : ℕ) (n : ℕ) (L : ℕ) 
  (h1 : width = 15) 
  (h2 : distance_covered = 540) 
  (h3 : n = 3) 
  (h4 : 2 * (L + width) = perimeter)
  (h5 : n * perimeter = distance_covered) : 
  L = 75 :=
by 
  sorry

end NUMINAMATH_GPT_length_of_field_l1201_120141


namespace NUMINAMATH_GPT_fraction_of_yard_occupied_l1201_120162

/-
Proof Problem: Given a rectangular yard that measures 30 meters by 8 meters and contains
an isosceles trapezoid-shaped flower bed with parallel sides measuring 14 meters and 24 meters,
and a height of 6 meters, prove that the fraction of the yard occupied by the flower bed is 19/40.
-/

theorem fraction_of_yard_occupied (length_yard width_yard b1 b2 h area_trapezoid area_yard : ℝ) 
  (h_length_yard : length_yard = 30) 
  (h_width_yard : width_yard = 8) 
  (h_b1 : b1 = 14) 
  (h_b2 : b2 = 24) 
  (h_height_trapezoid : h = 6) 
  (h_area_trapezoid : area_trapezoid = (1/2) * (b1 + b2) * h) 
  (h_area_yard : area_yard = length_yard * width_yard) : 
  area_trapezoid / area_yard = 19 / 40 := 
by {
  -- Follow-up steps to prove the statement would go here
  sorry
}

end NUMINAMATH_GPT_fraction_of_yard_occupied_l1201_120162


namespace NUMINAMATH_GPT_height_of_smaller_cone_removed_l1201_120145

noncomputable def frustum_area_lower_base : ℝ := 196 * Real.pi
noncomputable def frustum_area_upper_base : ℝ := 16 * Real.pi
def frustum_height : ℝ := 30

theorem height_of_smaller_cone_removed (r1 r2 H : ℝ)
  (h1 : r1 = Real.sqrt (frustum_area_lower_base / Real.pi))
  (h2 : r2 = Real.sqrt (frustum_area_upper_base / Real.pi))
  (h3 : r2 / r1 = 2 / 7)
  (h4 : frustum_height = (5 / 7) * H) :
  H - frustum_height = 12 :=
by 
  sorry

end NUMINAMATH_GPT_height_of_smaller_cone_removed_l1201_120145


namespace NUMINAMATH_GPT_greatest_possible_gcd_value_l1201_120153

noncomputable def sn (n : ℕ) := n ^ 2
noncomputable def expression (n : ℕ) := 2 * sn n + 10 * n
noncomputable def gcd_value (a b : ℕ) := Nat.gcd a b 

theorem greatest_possible_gcd_value :
  ∃ n : ℕ, gcd_value (expression n) (n - 3) = 42 :=
sorry

end NUMINAMATH_GPT_greatest_possible_gcd_value_l1201_120153


namespace NUMINAMATH_GPT_find_missing_surface_area_l1201_120121

noncomputable def total_surface_area (areas : List ℕ) : ℕ :=
  areas.sum

def known_areas : List ℕ := [148, 46, 72, 28, 88, 126, 58]

def missing_surface_area : ℕ := 22

theorem find_missing_surface_area (areas : List ℕ) (total : ℕ) (missing : ℕ) :
  total_surface_area areas + missing = total →
  missing = 22 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_surface_area_l1201_120121


namespace NUMINAMATH_GPT_total_cost_l1201_120186

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 4
def num_sodas : ℕ := 5

theorem total_cost : (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) = 31 := by
  sorry

end NUMINAMATH_GPT_total_cost_l1201_120186


namespace NUMINAMATH_GPT_simplify_fraction_l1201_120143

variables {x y : ℝ}

theorem simplify_fraction (h : x / y = 2 / 5) : (3 * y - 2 * x) / (3 * y + 2 * x) = 11 / 19 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1201_120143


namespace NUMINAMATH_GPT_steve_halfway_time_longer_l1201_120161

theorem steve_halfway_time_longer :
  ∀ (Td: ℝ) (Ts: ℝ),
  Td = 33 →
  Ts = 2 * Td →
  (Ts / 2) - (Td / 2) = 16.5 :=
by
  intros Td Ts hTd hTs
  rw [hTd, hTs]
  sorry

end NUMINAMATH_GPT_steve_halfway_time_longer_l1201_120161


namespace NUMINAMATH_GPT_linear_regression_solution_l1201_120168

theorem linear_regression_solution :
  let barx := 5
  let bary := 50
  let sum_xi_squared := 145
  let sum_xiyi := 1380
  let n := 5
  let b := (sum_xiyi - barx * bary) / (sum_xi_squared - n * barx^2)
  let a := bary - b * barx
  let predicted_y := 6.5 * 10 + 17.5
  b = 6.5 ∧ a = 17.5 ∧ predicted_y = 82.5 := 
by
  intros
  sorry

end NUMINAMATH_GPT_linear_regression_solution_l1201_120168


namespace NUMINAMATH_GPT_linear_function_decreasing_y_l1201_120120

theorem linear_function_decreasing_y (x1 y1 y2 : ℝ) :
  y1 = -2 * x1 - 7 → y2 = -2 * (x1 - 1) - 7 → y1 < y2 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_linear_function_decreasing_y_l1201_120120


namespace NUMINAMATH_GPT_simplify_fraction_l1201_120184

theorem simplify_fraction :
  (30 / 35) * (21 / 45) * (70 / 63) - (2 / 3) = - (8 / 15) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1201_120184


namespace NUMINAMATH_GPT_jack_shoes_time_l1201_120195

theorem jack_shoes_time (J : ℝ) (h : J + 2 * (J + 3) = 18) : J = 4 :=
by
  sorry

end NUMINAMATH_GPT_jack_shoes_time_l1201_120195


namespace NUMINAMATH_GPT_total_handshakes_at_convention_l1201_120176

theorem total_handshakes_at_convention :
  let gremlins := 25
  let imps := 18
  let specific_gremlins := 5
  let friendly_gremlins := gremlins - specific_gremlins
  let handshakes_among_gremlins := (friendly_gremlins * (friendly_gremlins - 1)) / 2
  let handshakes_between_imps_and_gremlins := imps * gremlins
  handshakes_among_gremlins + handshakes_between_imps_and_gremlins = 640 := by
  sorry

end NUMINAMATH_GPT_total_handshakes_at_convention_l1201_120176


namespace NUMINAMATH_GPT_melted_mixture_weight_l1201_120171

theorem melted_mixture_weight
    (Z C : ℝ)
    (ratio_eq : Z / C = 9 / 11)
    (zinc_weight : Z = 33.3) :
    Z + C = 74 :=
by
  sorry

end NUMINAMATH_GPT_melted_mixture_weight_l1201_120171


namespace NUMINAMATH_GPT_sum_of_coefficients_l1201_120164

def P (x : ℝ) : ℝ := 3 * (x^8 - 2 * x^5 + x^3 - 7) - 5 * (x^6 + 3 * x^2 - 6) + 2 * (x^4 - 5)

theorem sum_of_coefficients : P 1 = -19 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1201_120164


namespace NUMINAMATH_GPT_total_population_l1201_120198

theorem total_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) : 
  b + g + t = 26 * t :=
by
  -- We state our theorem including assumptions and goal
  sorry -- placeholder for the proof

end NUMINAMATH_GPT_total_population_l1201_120198


namespace NUMINAMATH_GPT_remainder_div2_l1201_120151

   theorem remainder_div2 :
     ∀ z x : ℕ, (∃ k : ℕ, z = 4 * k) → (∃ n : ℕ, x = 2 * n) → (z + x + 4 + z + 3) % 2 = 1 :=
   by
     intros z x h1 h2
     sorry
   
end NUMINAMATH_GPT_remainder_div2_l1201_120151


namespace NUMINAMATH_GPT_joey_route_length_l1201_120139

-- Definitions
def time_one_way : ℝ := 1
def avg_speed : ℝ := 8
def return_speed : ℝ := 12

-- Theorem to prove
theorem joey_route_length : (∃ D : ℝ, D = 6 ∧ (D / avg_speed = time_one_way + D / return_speed)) :=
sorry

end NUMINAMATH_GPT_joey_route_length_l1201_120139


namespace NUMINAMATH_GPT_range_of_a_l1201_120188

def A := {x : ℝ | |x| >= 3}
def B (a : ℝ) := {x : ℝ | x >= a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a <= -3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1201_120188


namespace NUMINAMATH_GPT_problem_inequality_l1201_120191

variable {α : Type*} [LinearOrder α]

def M (x y : α) : α := max x y
def m (x y : α) : α := min x y

theorem problem_inequality (a b c d e : α) (h : a < b) (h1 : b < c) (h2 : c < d) (h3 : d < e) : 
  M (M a (m b c)) (m d (m a e)) = b := sorry

end NUMINAMATH_GPT_problem_inequality_l1201_120191


namespace NUMINAMATH_GPT_proof_problem_l1201_120102

variables {x y z w : ℝ}

-- Condition given in the problem
def condition (x y z w : ℝ) : Prop :=
  (x - y) * (z - w) / ((y - z) * (w - x)) = 1 / 3

-- The statement to be proven
theorem proof_problem (h : condition x y z w) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1201_120102


namespace NUMINAMATH_GPT_no_valid_pair_for_tangential_quadrilateral_l1201_120148

theorem no_valid_pair_for_tangential_quadrilateral (a d : ℝ) (h : d > 0) :
  ¬((∃ a d, a + (a + 2 * d) = (a + d) + (a + 3 * d))) :=
by
  sorry

end NUMINAMATH_GPT_no_valid_pair_for_tangential_quadrilateral_l1201_120148


namespace NUMINAMATH_GPT_minimize_total_time_l1201_120123

def exercise_time (s : ℕ → ℕ) : Prop :=
  ∀ i, s i < 45

def total_exercises (a : ℕ → ℕ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 25

def minimize_time (a : ℕ → ℕ) (s : ℕ → ℕ) : Prop :=
  ∃ (j : ℕ), (1 ≤ j ∧ j ≤ 7 ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ 7 → if i = j then a i = 25 else a i = 0) ∧
  ∀ i, 1 ≤ i ∧ i ≤ 7 → s i ≥ s j)

theorem minimize_total_time
  (a : ℕ → ℕ) (s : ℕ → ℕ) 
  (h_exercise_time : exercise_time s)
  (h_total_exercises : total_exercises a) :
  minimize_time a s := by
  sorry

end NUMINAMATH_GPT_minimize_total_time_l1201_120123


namespace NUMINAMATH_GPT_min_value_quadratic_function_l1201_120163

def f (a b c x : ℝ) : ℝ := a * (x - b) * (x - c)

theorem min_value_quadratic_function :
  ∃ a b c : ℝ, 
    (1 ≤ a ∧ a < 10) ∧
    (1 ≤ b ∧ b < 10) ∧
    (1 ≤ c ∧ c < 10) ∧
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (∀ x : ℝ, f a b c x ≥ -128) :=
sorry

end NUMINAMATH_GPT_min_value_quadratic_function_l1201_120163


namespace NUMINAMATH_GPT_red_to_blue_ratio_l1201_120185

theorem red_to_blue_ratio
    (total_balls : ℕ)
    (num_white_balls : ℕ)
    (num_blue_balls : ℕ)
    (num_red_balls : ℕ) :
    total_balls = 100 →
    num_white_balls = 16 →
    num_blue_balls = num_white_balls + 12 →
    num_red_balls = total_balls - (num_white_balls + num_blue_balls) →
    (num_red_balls / num_blue_balls : ℚ) = 2 :=
by
  intro h1 h2 h3 h4
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_red_to_blue_ratio_l1201_120185


namespace NUMINAMATH_GPT_total_money_l1201_120113

theorem total_money (m c : ℝ) (hm : m = 5 / 8) (hc : c = 7 / 20) : m + c = 0.975 := sorry

end NUMINAMATH_GPT_total_money_l1201_120113


namespace NUMINAMATH_GPT_value_of_x_l1201_120109

def condition (x : ℝ) : Prop :=
  3 * x = (20 - x) + 20

theorem value_of_x : ∃ x : ℝ, condition x ∧ x = 10 := 
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1201_120109


namespace NUMINAMATH_GPT_parabola_relationship_l1201_120152

noncomputable def parabola (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem parabola_relationship (a b m n t : ℝ) (ha : a ≠ 0)
  (h1 : 3 * a + b > 0) (h2 : a + b < 0)
  (hm : parabola a b (-3) = m)
  (hn : parabola a b 2 = n)
  (ht : parabola a b 4 = t) :
  n < t ∧ t < m :=
by
  sorry

end NUMINAMATH_GPT_parabola_relationship_l1201_120152


namespace NUMINAMATH_GPT_division_quotient_l1201_120103

theorem division_quotient (x : ℤ) (y : ℤ) (r : ℝ) (h1 : x > 0) (h2 : y = 96) (h3 : r = 11.52) :
  ∃ q : ℝ, q = (x - r) / y := 
sorry

end NUMINAMATH_GPT_division_quotient_l1201_120103


namespace NUMINAMATH_GPT_range_of_a_l1201_120137

variable {x a : ℝ}

def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := |x| > a

theorem range_of_a (h : ¬p x → ¬q x a) : a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1201_120137


namespace NUMINAMATH_GPT_new_concentration_is_37_percent_l1201_120111

-- Conditions
def capacity_vessel_1 : ℝ := 2 -- litres
def alcohol_concentration_vessel_1 : ℝ := 0.35

def capacity_vessel_2 : ℝ := 6 -- litres
def alcohol_concentration_vessel_2 : ℝ := 0.50

def total_poured_liquid : ℝ := 8 -- litres
def final_vessel_capacity : ℝ := 10 -- litres

-- Question: Prove the new concentration of the mixture
theorem new_concentration_is_37_percent :
  (alcohol_concentration_vessel_1 * capacity_vessel_1 + alcohol_concentration_vessel_2 * capacity_vessel_2) / final_vessel_capacity = 0.37 := by
  sorry

end NUMINAMATH_GPT_new_concentration_is_37_percent_l1201_120111


namespace NUMINAMATH_GPT_problem_statement_l1201_120181

theorem problem_statement : 2017 - (1 / 2017) = (2018 * 2016) / 2017 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1201_120181


namespace NUMINAMATH_GPT_perfect_squares_50_to_200_l1201_120172

theorem perfect_squares_50_to_200 : 
  ∃ (k : ℕ), k = 7 ∧ ∀ n : ℤ, 50 < n^2 ∧ n^2 < 200 -> (8 ≤ n ∧ n ≤ 14) := 
by
  sorry

end NUMINAMATH_GPT_perfect_squares_50_to_200_l1201_120172


namespace NUMINAMATH_GPT_smallest_possible_value_l1201_120175

theorem smallest_possible_value (x : ℝ) (hx : 11 = x^2 + 1 / x^2) :
  x + 1 / x = -Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_l1201_120175


namespace NUMINAMATH_GPT_shirts_per_minute_l1201_120192

theorem shirts_per_minute (shirts_in_6_minutes : ℕ) (time_minutes : ℕ) (h1 : shirts_in_6_minutes = 36) (h2 : time_minutes = 6) : 
  ((shirts_in_6_minutes / time_minutes) = 6) :=
by
  sorry

end NUMINAMATH_GPT_shirts_per_minute_l1201_120192


namespace NUMINAMATH_GPT_correct_eq_count_l1201_120189

-- Define the correctness of each expression
def eq1 := (∀ x : ℤ, (-2 * x)^3 = 2 * x^3 = false)
def eq2 := (∀ a : ℤ, a^2 * a^3 = a^3 = false)
def eq3 := (∀ x : ℤ, (-x)^9 / (-x)^3 = x^6 = true)
def eq4 := (∀ a : ℤ, (-3 * a^2)^3 = -9 * a^6 = false)

-- Define the condition that there are exactly one correct equation
def num_correct_eqs := (1 = 1)

-- The theorem statement, proving the count of correct equations is 1
theorem correct_eq_count : eq1 → eq2 → eq3 → eq4 → num_correct_eqs :=
  by intros; sorry

end NUMINAMATH_GPT_correct_eq_count_l1201_120189


namespace NUMINAMATH_GPT_initial_money_l1201_120167

theorem initial_money (cost_of_candy_bar : ℕ) (change_received : ℕ) (initial_money : ℕ) 
  (h_cost : cost_of_candy_bar = 45) (h_change : change_received = 5) :
  initial_money = cost_of_candy_bar + change_received :=
by
  -- here is the place for the proof which is not needed
  sorry

end NUMINAMATH_GPT_initial_money_l1201_120167


namespace NUMINAMATH_GPT_sin_angle_calculation_l1201_120138

theorem sin_angle_calculation (α : ℝ) (h : α = 240) : Real.sin (150 - α) = -1 :=
by
  rw [h]
  norm_num
  sorry

end NUMINAMATH_GPT_sin_angle_calculation_l1201_120138


namespace NUMINAMATH_GPT_almond_walnut_ratio_is_5_to_2_l1201_120116

-- Definitions based on conditions
variables (A W : ℕ)
def almond_ratio_to_walnut_ratio := A / (2 * W)
def weight_of_almonds := 250
def total_weight := 350
def weight_of_walnuts := total_weight - weight_of_almonds

-- Theorem to prove
theorem almond_walnut_ratio_is_5_to_2
  (h_ratio : almond_ratio_to_walnut_ratio A W = 250 / 100)
  (h_weights : weight_of_walnuts = 100) :
  A = 5 ∧ 2 * W = 2 := by
  sorry

end NUMINAMATH_GPT_almond_walnut_ratio_is_5_to_2_l1201_120116


namespace NUMINAMATH_GPT_value_of_expression_l1201_120101

theorem value_of_expression : 1 + 3^2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1201_120101


namespace NUMINAMATH_GPT_exists_fixed_point_subset_l1201_120140

-- Definitions of set and function f with the required properties
variable {α : Type} [DecidableEq α]
variable (H : Finset α)
variable (f : Finset α → Finset α)

-- Conditions
axiom increasing_mapping (X Y : Finset α) : X ⊆ Y → f X ⊆ f Y
axiom range_in_H (X : Finset α) : f X ⊆ H

-- Statement to prove
theorem exists_fixed_point_subset : ∃ H₀ ⊆ H, f H₀ = H₀ :=
sorry

end NUMINAMATH_GPT_exists_fixed_point_subset_l1201_120140


namespace NUMINAMATH_GPT_divisible_by_n_sequence_l1201_120170

theorem divisible_by_n_sequence (n : ℕ) (h1 : n > 1) (h2 : n % 2 = 1) : 
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 ∧ n ∣ (2^k - 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_divisible_by_n_sequence_l1201_120170


namespace NUMINAMATH_GPT_invitations_per_package_l1201_120156

theorem invitations_per_package (total_friends : ℕ) (total_packs : ℕ) (invitations_per_pack : ℕ) 
  (h1 : total_friends = 10) (h2 : total_packs = 5)
  (h3 : invitations_per_pack * total_packs = total_friends) : 
  invitations_per_pack = 2 :=
by
  sorry

end NUMINAMATH_GPT_invitations_per_package_l1201_120156


namespace NUMINAMATH_GPT_alberto_more_than_bjorn_and_charlie_l1201_120160

theorem alberto_more_than_bjorn_and_charlie (time : ℕ) 
  (alberto_speed bjorn_speed charlie_speed: ℕ) 
  (alberto_distance bjorn_distance charlie_distance : ℕ) :
  time = 6 ∧ alberto_speed = 10 ∧ bjorn_speed = 8 ∧ charlie_speed = 9
  ∧ alberto_distance = alberto_speed * time
  ∧ bjorn_distance = bjorn_speed * time
  ∧ charlie_distance = charlie_speed * time
  → (alberto_distance - bjorn_distance = 12) ∧ (alberto_distance - charlie_distance = 6) :=
by
  sorry

end NUMINAMATH_GPT_alberto_more_than_bjorn_and_charlie_l1201_120160


namespace NUMINAMATH_GPT_multiple_of_9_is_multiple_of_3_l1201_120193

theorem multiple_of_9_is_multiple_of_3 (n : ℤ) (h : ∃ k : ℤ, n = 9 * k) : ∃ m : ℤ, n = 3 * m :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_9_is_multiple_of_3_l1201_120193


namespace NUMINAMATH_GPT_proof_problem_l1201_120126

-- Definitions based on the conditions
def x := 70 + 0.11 * 70
def y := x + 0.15 * x
def z := y - 0.2 * y

-- The statement to prove
theorem proof_problem : 3 * z - 2 * x + y = 148.407 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1201_120126


namespace NUMINAMATH_GPT_find_initial_marbles_l1201_120131

def initial_marbles (W Y H : ℕ) : Prop :=
  (W + 2 = 20) ∧ (Y - 5 = 20) ∧ (H + 3 = 20)

theorem find_initial_marbles (W Y H : ℕ) (h : initial_marbles W Y H) : W = 18 :=
  by
    sorry

end NUMINAMATH_GPT_find_initial_marbles_l1201_120131


namespace NUMINAMATH_GPT_probability_value_l1201_120117

noncomputable def P (k : ℕ) (c : ℚ) : ℚ := c / (k * (k + 1))

theorem probability_value (c : ℚ) (h : P 1 c + P 2 c + P 3 c + P 4 c = 1) : P 1 c + P 2 c = 5 / 6 := 
by
  sorry

end NUMINAMATH_GPT_probability_value_l1201_120117


namespace NUMINAMATH_GPT_trajectory_midpoint_l1201_120169

-- Define the hyperbola equation
def hyperbola (x y : ℝ) := x^2 - (y^2 / 4) = 1

-- Define the condition that a line passes through the point (0, 1)
def line_through_fixed_point (k x y : ℝ) := y = k * x + 1

-- Define the theorem to prove the trajectory of the midpoint of the chord
theorem trajectory_midpoint (x y k : ℝ) (h : ∃ x y, hyperbola x y ∧ line_through_fixed_point k x y) : 
    4 * x^2 - y^2 + y = 0 := 
sorry

end NUMINAMATH_GPT_trajectory_midpoint_l1201_120169


namespace NUMINAMATH_GPT_dasha_strip_problem_l1201_120166

theorem dasha_strip_problem (a b c : ℕ) (h : a * (2 * b + 2 * c - a) = 43) :
  a = 1 ∧ b + c = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_dasha_strip_problem_l1201_120166


namespace NUMINAMATH_GPT_intersection_complement_eq_three_l1201_120183

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_eq_three : N ∩ (U \ M) = {3} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_three_l1201_120183


namespace NUMINAMATH_GPT_charge_per_action_figure_l1201_120107

-- Definitions according to given conditions
def cost_of_sneakers : ℕ := 90
def saved_amount : ℕ := 15
def num_action_figures : ℕ := 10
def left_after_purchase : ℕ := 25

-- Theorem to prove the charge per action figure
theorem charge_per_action_figure : 
  (cost_of_sneakers - saved_amount + left_after_purchase) / num_action_figures = 10 :=
by 
  sorry

end NUMINAMATH_GPT_charge_per_action_figure_l1201_120107


namespace NUMINAMATH_GPT_all_are_knights_l1201_120127

-- Definitions for inhabitants as either knights or knaves
inductive Inhabitant
| Knight : Inhabitant
| Knave : Inhabitant

open Inhabitant

-- Functions that determine if an inhabitant is a knight or a knave
def is_knight (x : Inhabitant) : Prop :=
  x = Knight

def is_knave (x : Inhabitant) : Prop :=
  x = Knave

-- Given conditions
axiom A : Inhabitant
axiom B : Inhabitant
axiom C : Inhabitant

axiom statement_A : is_knight A → is_knight B
axiom statement_B : is_knight B → (is_knight A → is_knight C)

-- The proof goal
theorem all_are_knights : is_knight A ∧ is_knight B ∧ is_knight C := by
  sorry

end NUMINAMATH_GPT_all_are_knights_l1201_120127


namespace NUMINAMATH_GPT_valid_k_for_triangle_l1201_120178

theorem valid_k_for_triangle (k : ℕ) :
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ b + c > a ∧ c + a > b)) → k ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_valid_k_for_triangle_l1201_120178


namespace NUMINAMATH_GPT_Joey_age_l1201_120133

theorem Joey_age (J B : ℕ) (h1 : J + 5 = B) (h2 : J - 4 = B - J) : J = 9 :=
by 
  sorry

end NUMINAMATH_GPT_Joey_age_l1201_120133


namespace NUMINAMATH_GPT_initial_printing_presses_l1201_120108

theorem initial_printing_presses (P : ℕ) 
  (h1 : 500000 / (9 * P) = 500000 / (12 * 30)) : 
  P = 40 :=
by
  sorry

end NUMINAMATH_GPT_initial_printing_presses_l1201_120108


namespace NUMINAMATH_GPT_remaining_sugar_l1201_120119

-- Conditions as definitions
def total_sugar : ℝ := 9.8
def spilled_sugar : ℝ := 5.2

-- Theorem to prove the remaining sugar
theorem remaining_sugar : total_sugar - spilled_sugar = 4.6 := by
  sorry

end NUMINAMATH_GPT_remaining_sugar_l1201_120119


namespace NUMINAMATH_GPT_max_cone_cross_section_area_l1201_120179

theorem max_cone_cross_section_area
  (V A B : Type)
  (E : Type)
  (l : ℝ)
  (α : ℝ) :
  0 < l ∧ 0 < α ∧ α < 180 → 
  ∃ (area : ℝ), area = (1 / 2) * l^2 :=
by
  sorry

end NUMINAMATH_GPT_max_cone_cross_section_area_l1201_120179


namespace NUMINAMATH_GPT_find_x_of_product_eq_72_l1201_120129

theorem find_x_of_product_eq_72 (x : ℝ) (h : 0 < x) (hx : x * ⌊x⌋₊ = 72) : x = 9 :=
sorry

end NUMINAMATH_GPT_find_x_of_product_eq_72_l1201_120129


namespace NUMINAMATH_GPT_cookie_recipe_total_cups_l1201_120110

theorem cookie_recipe_total_cups (r_butter : ℕ) (r_flour : ℕ) (r_sugar : ℕ) (sugar_cups : ℕ) 
  (h_ratio : r_butter = 1 ∧ r_flour = 2 ∧ r_sugar = 3) (h_sugar : sugar_cups = 9) : 
  r_butter * (sugar_cups / r_sugar) + r_flour * (sugar_cups / r_sugar) + sugar_cups = 18 := 
by 
  sorry

end NUMINAMATH_GPT_cookie_recipe_total_cups_l1201_120110


namespace NUMINAMATH_GPT_ratio_two_to_three_nights_ago_l1201_120165

def question (x : ℕ) (k : ℕ) : (ℕ × ℕ) := (x, k)

def pages_three_nights_ago := 15
def additional_pages_last_night (x : ℕ) := x + 5
def total_pages := 100
def pages_tonight := 20

theorem ratio_two_to_three_nights_ago :
  ∃ (x : ℕ), 
    (x + additional_pages_last_night x = total_pages - (pages_three_nights_ago + pages_tonight)) 
    ∧ (x / pages_three_nights_ago = 2 / 1) :=
by
  sorry

end NUMINAMATH_GPT_ratio_two_to_three_nights_ago_l1201_120165


namespace NUMINAMATH_GPT_forty_percent_of_number_l1201_120100

theorem forty_percent_of_number (N : ℝ) 
  (h : (1/4) * (1/3) * (2/5) * N = 35) : 0.4 * N = 420 :=
by
  sorry

end NUMINAMATH_GPT_forty_percent_of_number_l1201_120100


namespace NUMINAMATH_GPT_test_end_time_l1201_120158

def start_time := 12 * 60 + 35  -- 12 hours 35 minutes in minutes
def duration := 4 * 60 + 50     -- 4 hours 50 minutes in minutes

theorem test_end_time : (start_time + duration) = 17 * 60 + 25 := by
  sorry

end NUMINAMATH_GPT_test_end_time_l1201_120158


namespace NUMINAMATH_GPT_two_pow_n_minus_one_div_by_seven_iff_l1201_120132

theorem two_pow_n_minus_one_div_by_seven_iff (n : ℕ) : (7 ∣ (2^n - 1)) ↔ (∃ k : ℕ, n = 3 * k) := by
  sorry

end NUMINAMATH_GPT_two_pow_n_minus_one_div_by_seven_iff_l1201_120132


namespace NUMINAMATH_GPT_find_value_of_expression_l1201_120190

noncomputable def x1 : ℝ := sorry
noncomputable def x2 : ℝ := sorry
noncomputable def x3 : ℝ := sorry
noncomputable def x4 : ℝ := sorry
noncomputable def x5 : ℝ := sorry
noncomputable def x6 : ℝ := sorry

def condition1 : Prop := x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 + 11 * x6 = 2
def condition2 : Prop := 3 * x1 + 5 * x2 + 7 * x3 + 9 * x4 + 11 * x5 + 13 * x6 = 15
def condition3 : Prop := 5 * x1 + 7 * x2 + 9 * x3 + 11 * x4 + 13 * x5 + 15 * x6 = 52

theorem find_value_of_expression : condition1 → condition2 → condition3 → (7 * x1 + 9 * x2 + 11 * x3 + 13 * x4 + 15 * x5 + 17 * x6 = 65) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l1201_120190


namespace NUMINAMATH_GPT_people_in_rooms_l1201_120157

theorem people_in_rooms (x y : ℕ) (h1 : x + y = 76) (h2 : x - 30 = y - 40) : x = 33 ∧ y = 43 := by
  sorry

end NUMINAMATH_GPT_people_in_rooms_l1201_120157


namespace NUMINAMATH_GPT_segments_either_disjoint_or_common_point_l1201_120147

theorem segments_either_disjoint_or_common_point (n : ℕ) (segments : List (ℝ × ℝ)) 
  (h_len : segments.length = n^2 + 1) : 
  (∃ (disjoint_segments : List (ℝ × ℝ)), disjoint_segments.length ≥ n + 1 ∧ 
    (∀ (s1 s2 : (ℝ × ℝ)), s1 ∈ disjoint_segments → s2 ∈ disjoint_segments 
    → s1 ≠ s2 → ¬ (s1.1 ≤ s2.2 ∧ s2.1 ≤ s1.2))) 
  ∨ 
  (∃ (common_point_segments : List (ℝ × ℝ)), common_point_segments.length ≥ n + 1 ∧ 
    (∃ (p : ℝ), ∀ (s : (ℝ × ℝ)), s ∈ common_point_segments → s.1 ≤ p ∧ p ≤ s.2)) :=
sorry

end NUMINAMATH_GPT_segments_either_disjoint_or_common_point_l1201_120147


namespace NUMINAMATH_GPT_find_r_s_l1201_120106

def is_orthogonal (v₁ v₂ : ℝ × ℝ × ℝ) : Prop :=
  v₁.1 * v₂.1 + v₁.2.1 * v₂.2.1 + v₁.2.2 * v₂.2.2 = 0

def have_equal_magnitudes (v₁ v₂ : ℝ × ℝ × ℝ) : Prop :=
  v₁.1^2 + v₁.2.1^2 + v₁.2.2^2 = v₂.1^2 + v₂.2.1^2 + v₂.2.2^2

theorem find_r_s (r s : ℝ) :
  is_orthogonal (4, r, -2) (-1, 2, s) ∧
  have_equal_magnitudes (4, r, -2) (-1, 2, s) →
  r = -11 / 4 ∧ s = -19 / 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_r_s_l1201_120106


namespace NUMINAMATH_GPT_find_divisor_l1201_120155

variable {N : ℤ} (k q : ℤ) {D : ℤ}

theorem find_divisor (h1 : N = 158 * k + 50) (h2 : N = D * q + 13) (h3 : D > 13) (h4 : D < 158) :
  D = 37 :=
by 
  sorry

end NUMINAMATH_GPT_find_divisor_l1201_120155


namespace NUMINAMATH_GPT_add_fractions_l1201_120114

theorem add_fractions : (1 / 6 : ℚ) + (5 / 12) = 7 / 12 := 
by
  sorry

end NUMINAMATH_GPT_add_fractions_l1201_120114


namespace NUMINAMATH_GPT_largest_integer_solution_l1201_120142

theorem largest_integer_solution :
  ∀ (x : ℤ), x - 5 > 3 * x - 1 → x ≤ -3 := by
  sorry

end NUMINAMATH_GPT_largest_integer_solution_l1201_120142


namespace NUMINAMATH_GPT_solve_for_x_l1201_120104

theorem solve_for_x (x : ℝ) (h1 : x^2 - 9 ≠ 0) (h2 : x + 3 ≠ 0) :
  (20 / (x^2 - 9) - 3 / (x + 3) = 2) ↔ (x = (-3 + Real.sqrt 385) / 4 ∨ x = (-3 - Real.sqrt 385) / 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1201_120104


namespace NUMINAMATH_GPT_Conor_can_chop_116_vegetables_in_a_week_l1201_120174

-- Define the conditions
def eggplants_per_day : ℕ := 12
def carrots_per_day : ℕ := 9
def potatoes_per_day : ℕ := 8
def work_days_per_week : ℕ := 4

-- Define the total vegetables per day
def vegetables_per_day : ℕ := eggplants_per_day + carrots_per_day + potatoes_per_day

-- Define the total vegetables per week
def vegetables_per_week : ℕ := vegetables_per_day * work_days_per_week

-- The proof statement
theorem Conor_can_chop_116_vegetables_in_a_week : vegetables_per_week = 116 :=
by
  sorry  -- The proof step is omitted with sorry

end NUMINAMATH_GPT_Conor_can_chop_116_vegetables_in_a_week_l1201_120174


namespace NUMINAMATH_GPT_rope_length_total_l1201_120135

theorem rope_length_total :
  let length1 := 24
  let length2 := 20
  let length3 := 14
  let length4 := 12
  length1 + length2 + length3 + length4 = 70 :=
by
  sorry

end NUMINAMATH_GPT_rope_length_total_l1201_120135


namespace NUMINAMATH_GPT_prime_power_value_l1201_120182

theorem prime_power_value (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h1 : Nat.Prime (7 * p + q)) (h2 : Nat.Prime (p * q + 11)) : 
  p ^ q = 8 ∨ p ^ q = 9 := 
sorry

end NUMINAMATH_GPT_prime_power_value_l1201_120182


namespace NUMINAMATH_GPT_part_I_part_II_l1201_120112

-- Define the conditions given in the problem
def set_A : Set ℝ := { x | -1 < x ∧ x < 3 }
def set_B (a b : ℝ) : Set ℝ := { x | x^2 - a * x + b < 0 }

-- Part I: Prove that if A = B, then a = 2 and b = -3
theorem part_I (a b : ℝ) (h : set_A = set_B a b) : a = 2 ∧ b = -3 :=
sorry

-- Part II: Prove that if b = 3 and A ∩ B ⊇ B, then the range of a is [-2√3, 4]
theorem part_II (a : ℝ) (b : ℝ := 3) (h : set_A ∩ set_B a b ⊇ set_B a b) : -2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1201_120112


namespace NUMINAMATH_GPT_susie_total_earnings_l1201_120199

def pizza_prices (type : String) (is_whole : Bool) : ℝ :=
  match type, is_whole with
  | "Margherita", false => 3
  | "Margherita", true  => 15
  | "Pepperoni", false  => 4
  | "Pepperoni", true   => 18
  | "Veggie Supreme", false => 5
  | "Veggie Supreme", true  => 22
  | "Meat Lovers", false => 6
  | "Meat Lovers", true  => 25
  | "Hawaiian", false   => 4.5
  | "Hawaiian", true    => 20
  | _, _                => 0

def topping_price (is_weekend : Bool) : ℝ :=
  if is_weekend then 1 else 2

def happy_hour_price : ℝ := 3

noncomputable def susie_earnings : ℝ :=
  let margherita_slices := 12 * happy_hour_price + 12 * pizza_prices "Margherita" false
  let pepperoni_slices := 8 * happy_hour_price + 8 * pizza_prices "Pepperoni" false + 6 * topping_price true
  let veggie_supreme_pizzas := 4 * pizza_prices "Veggie Supreme" true + 8 * topping_price true
  let margherita_whole_discounted := 3 * pizza_prices "Margherita" true - (3 * pizza_prices "Margherita" true) * 0.1
  let meat_lovers_slices := 10 * happy_hour_price + 10 * pizza_prices "Meat Lovers" false
  let hawaiian_slices := 12 * pizza_prices "Hawaiian" false + 4 * topping_price true
  let pepperoni_whole := pizza_prices "Pepperoni" true + 3 * topping_price true
  margherita_slices + pepperoni_slices + veggie_supreme_pizzas + margherita_whole_discounted + meat_lovers_slices + hawaiian_slices + pepperoni_whole

theorem susie_total_earnings : susie_earnings = 439.5 := by
  sorry

end NUMINAMATH_GPT_susie_total_earnings_l1201_120199


namespace NUMINAMATH_GPT_jeremy_watermelons_l1201_120134

theorem jeremy_watermelons :
  ∀ (total_watermelons : ℕ) (weeks : ℕ) (consumption_per_week : ℕ) (eaten_per_week : ℕ),
  total_watermelons = 30 →
  weeks = 6 →
  eaten_per_week = 3 →
  consumption_per_week = total_watermelons / weeks →
  (consumption_per_week - eaten_per_week) = 2 :=
by
  intros total_watermelons weeks consumption_per_week eaten_per_week h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_jeremy_watermelons_l1201_120134


namespace NUMINAMATH_GPT_min_value_expression_l1201_120105

theorem min_value_expression (x y z : ℝ) (h : x - 2 * y + 2 * z = 5) : (x + 5) ^ 2 + (y - 1) ^ 2 + (z + 3) ^ 2 ≥ 36 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1201_120105


namespace NUMINAMATH_GPT_max_a_value_l1201_120194

-- Variables representing the real numbers a, b, c, and d
variables (a b c d : ℝ)

-- Real number hypothesis conditions
-- 1. a + b + c + d = 10
-- 2. ab + ac + ad + bc + bd + cd = 20

theorem max_a_value
  (h1 : a + b + c + d = 10)
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  a ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_GPT_max_a_value_l1201_120194


namespace NUMINAMATH_GPT_jebb_take_home_pay_is_4620_l1201_120124

noncomputable def gross_salary : ℤ := 6500
noncomputable def federal_tax (income : ℤ) : ℤ :=
  let tax1 := min income 2000 * 10 / 100
  let tax2 := min (max (income - 2000) 0) 2000 * 15 / 100
  let tax3 := max (income - 4000) 0 * 25 / 100
  tax1 + tax2 + tax3

noncomputable def health_insurance : ℤ := 300
noncomputable def retirement_contribution (income : ℤ) : ℤ := income * 7 / 100

noncomputable def total_deductions (income : ℤ) : ℤ :=
  federal_tax income + health_insurance + retirement_contribution income

noncomputable def take_home_pay (income : ℤ) : ℤ :=
  income - total_deductions income

theorem jebb_take_home_pay_is_4620 : take_home_pay gross_salary = 4620 := by
  sorry

end NUMINAMATH_GPT_jebb_take_home_pay_is_4620_l1201_120124


namespace NUMINAMATH_GPT_cubic_sum_l1201_120154

theorem cubic_sum (x : ℝ) (h : x + 1/x = 4) : x^3 + 1/x^3 = 52 :=
by 
  sorry

end NUMINAMATH_GPT_cubic_sum_l1201_120154


namespace NUMINAMATH_GPT_must_hold_inequality_l1201_120125

variable (f : ℝ → ℝ)

noncomputable def condition : Prop := ∀ x > 0, x * (deriv^[2] f) x < 1

theorem must_hold_inequality (h : condition f) : f (Real.exp 1) < f 1 + 1 := 
sorry

end NUMINAMATH_GPT_must_hold_inequality_l1201_120125


namespace NUMINAMATH_GPT_volleyball_team_selection_l1201_120159

noncomputable def numberOfWaysToChooseStarters : ℕ :=
  (Nat.choose 13 4 * 3) + (Nat.choose 14 4 * 1)

theorem volleyball_team_selection :
  numberOfWaysToChooseStarters = 3146 := by
  sorry

end NUMINAMATH_GPT_volleyball_team_selection_l1201_120159


namespace NUMINAMATH_GPT_reciprocal_is_1_or_neg1_self_square_is_0_or_1_l1201_120115

theorem reciprocal_is_1_or_neg1 (x : ℝ) (hx : x = 1 / x) :
  x = 1 ∨ x = -1 :=
sorry

theorem self_square_is_0_or_1 (x : ℝ) (hx : x = x^2) :
  x = 0 ∨ x = 1 :=
sorry

end NUMINAMATH_GPT_reciprocal_is_1_or_neg1_self_square_is_0_or_1_l1201_120115


namespace NUMINAMATH_GPT_T_bisects_broken_line_l1201_120128

def midpoint_arc {α : Type*} [LinearOrderedField α] (A B C : α) : α := (A + B + C) / 2
def projection_perpendicular {α : Type*} [LinearOrderedField α] (F A B C : α) : α := sorry -- Define perpendicular projection T

theorem T_bisects_broken_line {α : Type*} [LinearOrderedField α]
  (A B C : α) (F := midpoint_arc A B C) (T := projection_perpendicular F A B C) :
  T = (A + B + C) / 2 :=
sorry

end NUMINAMATH_GPT_T_bisects_broken_line_l1201_120128
