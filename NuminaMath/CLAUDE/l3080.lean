import Mathlib

namespace james_brothers_count_l3080_308059

def market_value : ℝ := 500000
def selling_price : ℝ := market_value * 1.2
def revenue_after_taxes : ℝ := selling_price * 0.9
def share_per_person : ℝ := 135000

theorem james_brothers_count :
  ∃ (n : ℕ), (revenue_after_taxes / (n + 1 : ℝ) = share_per_person) ∧ n = 3 :=
by sorry

end james_brothers_count_l3080_308059


namespace particle_prob_origin_prob_form_l3080_308060

/-- A particle starts at (4,4) and moves randomly until it hits a coordinate axis. 
    At each step, it moves to one of (a-1, b), (a, b-1), or (a-1, b-1) with equal probability. -/
def particle_movement (a b : ℕ) : Fin 3 → ℕ × ℕ
| 0 => (a - 1, b)
| 1 => (a, b - 1)
| 2 => (a - 1, b - 1)

/-- The probability of the particle reaching (0,0) when starting from (4,4) -/
def prob_reach_origin : ℚ :=
  63 / 3^8

/-- Theorem stating that the probability of reaching (0,0) is 63/3^8 -/
theorem particle_prob_origin : 
  prob_reach_origin = 63 / 3^8 := by sorry

/-- The probability can be expressed as m/3^n where m is not divisible by 3 -/
theorem prob_form (m n : ℕ) (h : m % 3 ≠ 0) : 
  prob_reach_origin = m / 3^n := by sorry

end particle_prob_origin_prob_form_l3080_308060


namespace expression_value_l3080_308039

theorem expression_value : 3^(1^(2^8)) + ((3^1)^2)^8 = 43046724 := by
  sorry

end expression_value_l3080_308039


namespace number_composition_proof_l3080_308010

theorem number_composition_proof : 
  let ones : ℕ := 5
  let tenths : ℕ := 7
  let hundredths : ℕ := 21
  let thousandths : ℕ := 53
  let composed_number := 
    (ones : ℝ) + 
    (tenths : ℝ) * 0.1 + 
    (hundredths : ℝ) * 0.01 + 
    (thousandths : ℝ) * 0.001
  10 * composed_number = 59.63 := by
sorry

end number_composition_proof_l3080_308010


namespace elsa_final_marbles_l3080_308079

/-- Calculates the final number of marbles Elsa has at the end of the day. -/
def elsas_marbles (initial : ℕ) (lost_breakfast : ℕ) (given_to_susie : ℕ) (received_from_mom : ℕ) : ℕ :=
  initial - lost_breakfast - given_to_susie + received_from_mom + 2 * given_to_susie

/-- Theorem stating that Elsa ends up with 54 marbles given the conditions of the problem. -/
theorem elsa_final_marbles :
  elsas_marbles 40 3 5 12 = 54 :=
by sorry

end elsa_final_marbles_l3080_308079


namespace circle_reconstruction_uniqueness_l3080_308007

-- Define the types for lines and circles
def Line : Type := ℝ × ℝ → Prop
def Circle : Type := ℝ × ℝ → Prop

-- Define the property of two lines being parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the property of two lines being perpendicular
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define the property of a line being tangent to a circle
def tangent_to (l : Line) (c : Circle) : Prop := sorry

-- Define the distance between two parallel lines
def distance_between_parallel_lines (l1 l2 : Line) : ℝ := sorry

-- Main theorem
theorem circle_reconstruction_uniqueness 
  (e1 e2 f1 f2 : Line) 
  (h_parallel_e : parallel e1 e2) 
  (h_parallel_f : parallel f1 f2) 
  (h_not_perp_e1f1 : ¬ perpendicular e1 f1) 
  (h_not_perp_e2f2 : ¬ perpendicular e2 f2) :
  (∃! (k1 k2 : Circle), 
    tangent_to e1 k1 ∧ tangent_to e2 k2 ∧ 
    tangent_to f1 k1 ∧ tangent_to f2 k2 ∧ 
    (∃ (e f : Line), tangent_to e k1 ∧ tangent_to e k2 ∧ 
                     tangent_to f k1 ∧ tangent_to f k2)) ↔ 
  distance_between_parallel_lines e1 e2 ≠ distance_between_parallel_lines f1 f2 :=
sorry

end circle_reconstruction_uniqueness_l3080_308007


namespace bulb_cost_difference_l3080_308027

theorem bulb_cost_difference (lamp_cost : ℝ) (total_cost : ℝ) (bulb_cost : ℝ) : 
  lamp_cost = 7 → 
  2 * lamp_cost + 6 * bulb_cost = 32 → 
  bulb_cost < lamp_cost →
  lamp_cost - bulb_cost = 4 := by
sorry

end bulb_cost_difference_l3080_308027


namespace smallest_solution_quartic_equation_l3080_308019

theorem smallest_solution_quartic_equation :
  ∃ x : ℝ, x^4 - 14*x^2 + 49 = 0 ∧ 
  (∀ y : ℝ, y^4 - 14*y^2 + 49 = 0 → x ≤ y) ∧
  x = -Real.sqrt 7 :=
sorry

end smallest_solution_quartic_equation_l3080_308019


namespace sum_RS_ST_l3080_308029

/-- Represents a polygon PQRSTU -/
structure Polygon :=
  (area : ℝ)
  (PQ : ℝ)
  (QR : ℝ)
  (TU : ℝ)

/-- Theorem stating the sum of RS and ST in the polygon PQRSTU -/
theorem sum_RS_ST (poly : Polygon) (h1 : poly.area = 70) (h2 : poly.PQ = 10) 
  (h3 : poly.QR = 7) (h4 : poly.TU = 6) : ∃ (RS ST : ℝ), RS + ST = 80 := by
  sorry

#check sum_RS_ST

end sum_RS_ST_l3080_308029


namespace parallel_lines_distance_l3080_308075

/-- Given a circle intersected by three equally spaced parallel lines creating chords of lengths 36, 36, and 40, the distance between two adjacent parallel lines is 4√19/3 -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (36 * r^2 = 648 + (9/4) * d^2) ∧ 
  (40 * r^2 = 800 + (45/4) * d^2) →
  d = (4 * Real.sqrt 19) / 3 :=
by sorry

end parallel_lines_distance_l3080_308075


namespace monochromatic_state_reachable_final_color_independent_l3080_308044

/-- Represents the three possible colors of glass pieces -/
inductive Color
  | Red
  | Yellow
  | Blue

/-- Represents the state of glass pieces -/
structure GlassState :=
  (red : Nat)
  (yellow : Nat)
  (blue : Nat)
  (total : Nat)
  (total_eq : red + yellow + blue = total)

/-- Represents an operation on glass pieces -/
def perform_operation (state : GlassState) : GlassState :=
  sorry

/-- Theorem stating that it's always possible to reach a monochromatic state -/
theorem monochromatic_state_reachable (initial_state : GlassState) 
  (h : initial_state.total = 1987) :
  ∃ (final_state : GlassState) (c : Color), 
    (final_state.red = initial_state.total ∧ c = Color.Red) ∨
    (final_state.yellow = initial_state.total ∧ c = Color.Yellow) ∨
    (final_state.blue = initial_state.total ∧ c = Color.Blue) :=
  sorry

/-- Theorem stating that the final color is independent of operation order -/
theorem final_color_independent (initial_state : GlassState) 
  (h : initial_state.total = 1987) :
  ∀ (final_state1 final_state2 : GlassState) (c1 c2 : Color),
    ((final_state1.red = initial_state.total ∧ c1 = Color.Red) ∨
     (final_state1.yellow = initial_state.total ∧ c1 = Color.Yellow) ∨
     (final_state1.blue = initial_state.total ∧ c1 = Color.Blue)) →
    ((final_state2.red = initial_state.total ∧ c2 = Color.Red) ∨
     (final_state2.yellow = initial_state.total ∧ c2 = Color.Yellow) ∨
     (final_state2.blue = initial_state.total ∧ c2 = Color.Blue)) →
    c1 = c2 :=
  sorry

end monochromatic_state_reachable_final_color_independent_l3080_308044


namespace expression_simplification_l3080_308042

theorem expression_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 - 3*x^3 =
  -x^3 - x^2 + 23*x - 3 := by
  sorry

end expression_simplification_l3080_308042


namespace oreo_cheesecake_problem_l3080_308043

theorem oreo_cheesecake_problem (graham_boxes_initial : ℕ) (graham_boxes_per_cake : ℕ) (oreo_packets_per_cake : ℕ) (graham_boxes_leftover : ℕ) :
  graham_boxes_initial = 14 →
  graham_boxes_per_cake = 2 →
  oreo_packets_per_cake = 3 →
  graham_boxes_leftover = 4 →
  let cakes_made := (graham_boxes_initial - graham_boxes_leftover) / graham_boxes_per_cake
  ∃ oreo_packets_bought : ℕ, oreo_packets_bought = cakes_made * oreo_packets_per_cake :=
by sorry

end oreo_cheesecake_problem_l3080_308043


namespace f_deriv_l3080_308089

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (2 * x - 3 + Real.sqrt (4 * x^2 - 12 * x + 10)) -
  Real.sqrt (4 * x^2 - 12 * x + 10) * Real.arctan (2 * x - 3)

theorem f_deriv :
  ∀ x : ℝ, DifferentiableAt ℝ f x →
    deriv f x = - Real.arctan (2 * x - 3) / Real.sqrt (4 * x^2 - 12 * x + 10) :=
by sorry

end f_deriv_l3080_308089


namespace power_division_rule_l3080_308082

theorem power_division_rule (a : ℝ) : a^7 / a^5 = a^2 := by
  sorry

end power_division_rule_l3080_308082


namespace f_properties_l3080_308098

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (2*a + 1)*x + a * log x

theorem f_properties (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) ((2*x - (2*a + 1) + a/x) : ℝ) x) ∧
  (HasDerivAt (f a) 0 1 ↔ a = 1) ∧
  (∀ x > 1, f a x > 0 ↔ a ≤ 0) :=
sorry

end f_properties_l3080_308098


namespace pentadecagon_triangles_l3080_308040

/-- The number of vertices in a regular pentadecagon -/
def n : ℕ := 15

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem pentadecagon_triangles : num_triangles = 455 := by sorry

end pentadecagon_triangles_l3080_308040


namespace quadratic_equation_solution_l3080_308084

theorem quadratic_equation_solution (x y z t : ℝ) :
  x^2 + y^2 + z^2 + t^2 = x*(y + z + t) → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end quadratic_equation_solution_l3080_308084


namespace partner_investment_time_l3080_308047

/-- Given two partners p and q with investment and profit ratios, prove q's investment time -/
theorem partner_investment_time
  (investment_ratio_p investment_ratio_q : ℚ)
  (profit_ratio_p profit_ratio_q : ℚ)
  (investment_time_p : ℚ) :
  investment_ratio_p = 7 →
  investment_ratio_q = 5 →
  profit_ratio_p = 7 →
  profit_ratio_q = 10 →
  investment_time_p = 2 →
  ∃ (investment_time_q : ℚ),
    investment_time_q = 4 ∧
    (profit_ratio_p / profit_ratio_q) =
    ((investment_ratio_p * investment_time_p) /
     (investment_ratio_q * investment_time_q)) :=
by sorry


end partner_investment_time_l3080_308047


namespace jack_gerald_notebook_difference_l3080_308033

theorem jack_gerald_notebook_difference :
  ∀ (jack_initial gerald : ℕ),
    jack_initial > gerald →
    gerald = 8 →
    jack_initial - 5 - 6 = 10 →
    jack_initial - gerald = 13 := by
  sorry

end jack_gerald_notebook_difference_l3080_308033


namespace cubic_equation_natural_roots_l3080_308071

/-- The cubic equation has three natural number roots if and only if p = 76 -/
theorem cubic_equation_natural_roots (p : ℝ) : 
  (∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (5 * (x : ℝ)^3 - 5*(p+1)*(x : ℝ)^2 + (71*p-1)*(x : ℝ) + 1 = 66*p) ∧
    (5 * (y : ℝ)^3 - 5*(p+1)*(y : ℝ)^2 + (71*p-1)*(y : ℝ) + 1 = 66*p) ∧
    (5 * (z : ℝ)^3 - 5*(p+1)*(z : ℝ)^2 + (71*p-1)*(z : ℝ) + 1 = 66*p)) ↔
  p = 76 :=
by sorry

end cubic_equation_natural_roots_l3080_308071


namespace chocolate_division_l3080_308045

/-- Represents the number of chocolate pieces Maria has after a given number of days -/
def chocolatePieces (days : ℕ) : ℕ :=
  9 + 8 * days

theorem chocolate_division :
  (chocolatePieces 3 = 25) ∧
  (∀ n : ℕ, chocolatePieces n ≠ 2014) :=
by sorry

end chocolate_division_l3080_308045


namespace orange_juice_mixture_fraction_l3080_308090

/-- Represents the fraction of orange juice in a mixture of two pitchers -/
def orange_juice_fraction (capacity1 capacity2 : ℚ) (fraction1 fraction2 : ℚ) : ℚ :=
  (capacity1 * fraction1 + capacity2 * fraction2) / (capacity1 + capacity2)

/-- Theorem stating that the fraction of orange juice in the given mixture is 17/52 -/
theorem orange_juice_mixture_fraction :
  orange_juice_fraction 500 800 (1/4) (3/8) = 17/52 := by
  sorry

end orange_juice_mixture_fraction_l3080_308090


namespace james_beef_pork_ratio_l3080_308048

/-- Proves that the ratio of beef to pork James bought is 2:1 given the problem conditions --/
theorem james_beef_pork_ratio :
  ∀ (beef pork : ℝ) (meals : ℕ),
    beef = 20 →
    meals * 20 = 400 →
    meals * 1.5 = beef + pork →
    beef / pork = 2 := by
  sorry

end james_beef_pork_ratio_l3080_308048


namespace correct_proposition_l3080_308036

-- Define proposition p
def p : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

-- Theorem to prove
theorem correct_proposition : ¬p ∧ q := by
  sorry

end correct_proposition_l3080_308036


namespace joes_lift_ratio_l3080_308000

/-- Joe's weight-lifting competition results -/
def JoesLifts (first second : ℕ) : Prop :=
  first + second = 600 ∧ first = 300 ∧ 2 * first = second + 300

theorem joes_lift_ratio :
  ∀ first second : ℕ, JoesLifts first second → first = second :=
by
  sorry

end joes_lift_ratio_l3080_308000


namespace geometric_progression_values_l3080_308031

theorem geometric_progression_values (p : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (3*p + 1) = |p - 3| * r ∧ (9*p + 10) = (3*p + 1) * r) ↔ 
  (p = -1 ∨ p = 29/18) :=
sorry

end geometric_progression_values_l3080_308031


namespace coin_flip_probability_difference_l3080_308061

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

def prob_heads (n k : ℕ) : ℚ :=
  (binomial n k : ℚ) * (1 / 2) ^ n

theorem coin_flip_probability_difference : 
  |prob_heads 5 2 - prob_heads 5 4| = 5 / 32 := by sorry

end coin_flip_probability_difference_l3080_308061


namespace hyperbola_dimensions_l3080_308041

/-- Proves that for a hyperbola with given conditions, a = 3 and b = 4 -/
theorem hyperbola_dimensions (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_conjugate : 2 * b = 8)
  (h_distance : a * b / Real.sqrt (a^2 + b^2) = 12/5) :
  a = 3 ∧ b = 4 := by
  sorry

end hyperbola_dimensions_l3080_308041


namespace square_area_from_adjacent_points_l3080_308070

/-- The area of a square with adjacent vertices at (1,3) and (4,7) is 25 square units. -/
theorem square_area_from_adjacent_points : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (4, 7)
  let distance := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := distance^2
  area = 25 := by sorry

end square_area_from_adjacent_points_l3080_308070


namespace prop_A_sufficient_not_necessary_for_prop_B_l3080_308009

theorem prop_A_sufficient_not_necessary_for_prop_B :
  (∀ a b : ℝ, a < b ∧ b < 0 → a * b > b^2) ∧
  (∃ a b : ℝ, a * b > b^2 ∧ ¬(a < b ∧ b < 0)) :=
by sorry

end prop_A_sufficient_not_necessary_for_prop_B_l3080_308009


namespace characterize_valid_functions_l3080_308012

def is_valid_function (f : ℤ → ℤ) : Prop :=
  f 1 ≠ f (-1) ∧ ∀ m n : ℤ, (f (m + n))^2 ∣ (f m - f n)

theorem characterize_valid_functions :
  ∀ f : ℤ → ℤ, is_valid_function f →
    (∀ x : ℤ, f x = 1 ∨ f x = -1) ∨
    (∀ x : ℤ, f x = 2 ∨ f x = -2) ∧ f 1 = -f (-1) :=
by sorry

end characterize_valid_functions_l3080_308012


namespace unique_solution_exponential_equation_l3080_308096

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (4 : ℝ)^x + (2 : ℝ)^x - 2 = 0 :=
by sorry

end unique_solution_exponential_equation_l3080_308096


namespace gilbert_basil_bushes_l3080_308085

/-- The number of basil bushes Gilbert planted initially -/
def initial_basil_bushes : ℕ := 1

/-- The total number of herb plants at the end of spring -/
def total_plants : ℕ := 5

/-- The number of mint types (which were eaten) -/
def mint_types : ℕ := 2

/-- The number of parsley plants -/
def parsley_plants : ℕ := 1

/-- The number of extra basil plants that grew during spring -/
def extra_basil : ℕ := 1

theorem gilbert_basil_bushes :
  initial_basil_bushes = total_plants - mint_types - parsley_plants - extra_basil :=
by sorry

end gilbert_basil_bushes_l3080_308085


namespace bottles_theorem_l3080_308037

/-- The number of ways to take out 24 bottles, where each time either 3 or 4 bottles are taken -/
def ways_to_take_bottles : ℕ :=
  -- Number of ways to take out 4 bottles 6 times
  1 +
  -- Number of ways to take out 3 bottles 8 times
  1 +
  -- Number of ways to take out 3 bottles 4 times and 4 bottles 3 times
  (Nat.choose 7 3)

/-- Theorem stating that the number of ways to take out the bottles is 37 -/
theorem bottles_theorem : ways_to_take_bottles = 37 := by
  sorry

#eval ways_to_take_bottles

end bottles_theorem_l3080_308037


namespace expression_simplification_l3080_308035

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 + 1) :
  (x + 1) / (x + 2) / (x - 2 + 3 / (x + 2)) = Real.sqrt 5 / 5 := by
  sorry

end expression_simplification_l3080_308035


namespace f_properties_g_inequality_l3080_308001

/-- The function f(x) = a ln x + 1/x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1 / x

/-- The function g(x) = f(x) - 1/x -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 1 / x

theorem f_properties (a : ℝ) :
  (a > 0 → (∃ (x : ℝ), x > 0 ∧ f a x = a - a * Real.log a ∧ ∀ y > 0, f a y ≥ f a x) ∧
            (¬∃ (M : ℝ), ∀ x > 0, f a x ≤ M)) ∧
  (a ≤ 0 → ¬∃ (x : ℝ), x > 0 ∧ (∀ y > 0, f a y ≥ f a x ∨ ∀ y > 0, f a y ≤ f a x)) :=
sorry

theorem g_inequality (m n : ℝ) (h1 : 0 < m) (h2 : m < n) :
  (g 1 n - g 1 m) / 2 > (n - m) / (n + m) :=
sorry

end f_properties_g_inequality_l3080_308001


namespace intersection_M_N_l3080_308073

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ 0}
def N : Set ℝ := {y | -Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = {y | 0 ≤ y ∧ y ≤ Real.sqrt 2} := by sorry

end intersection_M_N_l3080_308073


namespace father_total_spending_l3080_308099

def heaven_spending : ℕ := 2 * 5 + 4 * 5
def brother_eraser_spending : ℕ := 10 * 4
def brother_highlighter_spending : ℕ := 30

theorem father_total_spending :
  heaven_spending + brother_eraser_spending + brother_highlighter_spending = 100 := by
  sorry

end father_total_spending_l3080_308099


namespace product_remainder_by_five_l3080_308023

theorem product_remainder_by_five : 
  (2685 * 4932 * 91406) % 5 = 0 := by
  sorry

end product_remainder_by_five_l3080_308023


namespace perpendicular_unit_vector_l3080_308058

def a : ℝ × ℝ := (2, 1)

theorem perpendicular_unit_vector :
  let v : ℝ × ℝ := (Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5)
  (v.1 * v.1 + v.2 * v.2 = 1) ∧ (a.1 * v.1 + a.2 * v.2 = 0) := by
  sorry

end perpendicular_unit_vector_l3080_308058


namespace circle_of_students_l3080_308078

theorem circle_of_students (n : ℕ) (h : n > 0) :
  (∃ (a b : ℕ), a < n ∧ b < n ∧ a = 6 ∧ b = 16 ∧ (b - a) * 2 + 2 = n) →
  n = 22 :=
by sorry

end circle_of_students_l3080_308078


namespace largest_increase_2007_2008_l3080_308066

/-- Represents the number of students taking AMC 10 for each year from 2002 to 2008 -/
def students : Fin 7 → ℕ
  | 0 => 50  -- 2002
  | 1 => 55  -- 2003
  | 2 => 60  -- 2004
  | 3 => 65  -- 2005
  | 4 => 72  -- 2006
  | 5 => 80  -- 2007
  | 6 => 90  -- 2008

/-- Calculates the percentage increase between two consecutive years -/
def percentageIncrease (year : Fin 6) : ℚ :=
  (students (year.succ) - students year) / students year * 100

/-- Theorem stating that the percentage increase between 2007 and 2008 is the largest -/
theorem largest_increase_2007_2008 :
  ∀ year : Fin 6, percentageIncrease 5 ≥ percentageIncrease year :=
by sorry

end largest_increase_2007_2008_l3080_308066


namespace circle_ratio_new_circumference_to_area_increase_l3080_308053

/-- The ratio of new circumference to increase in area when a circle's radius is increased -/
theorem circle_ratio_new_circumference_to_area_increase 
  (r k : ℝ) (h : k > 0) : 
  (2 * Real.pi * (r + k)) / (Real.pi * ((r + k)^2 - r^2)) = 2 * (r + k) / (2 * r * k + k^2) :=
sorry

end circle_ratio_new_circumference_to_area_increase_l3080_308053


namespace absolute_value_equation_solution_l3080_308077

theorem absolute_value_equation_solution :
  {y : ℝ | |4 * y - 5| = 39} = {11, -8.5} := by
  sorry

end absolute_value_equation_solution_l3080_308077


namespace permutations_of_six_distinct_objects_l3080_308092

theorem permutations_of_six_distinct_objects : Nat.factorial 6 = 720 := by
  sorry

end permutations_of_six_distinct_objects_l3080_308092


namespace return_amount_calculation_l3080_308015

-- Define the borrowed amount
def borrowed_amount : ℝ := 100

-- Define the interest rate
def interest_rate : ℝ := 0.10

-- Theorem to prove
theorem return_amount_calculation :
  borrowed_amount * (1 + interest_rate) = 110 := by
  sorry

end return_amount_calculation_l3080_308015


namespace set_A_equivalent_range_of_a_l3080_308093

-- Define set A
def A : Set ℝ := {x | (3*x - 5)/(x + 1) ≤ 1}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}

-- Theorem for part 1
theorem set_A_equivalent : A = {x : ℝ | -1 < x ∧ x ≤ 3} := by sorry

-- Theorem for part 2
theorem range_of_a (a : ℝ) : B a ∩ (Set.univ \ A) = B a → a ≤ -2 ∨ a > 4 := by sorry

end set_A_equivalent_range_of_a_l3080_308093


namespace central_angle_invariant_under_doubling_l3080_308017

theorem central_angle_invariant_under_doubling 
  (r : ℝ) (l : ℝ) (h_r : r > 0) (h_l : l > 0) :
  l / r = (2 * l) / (2 * r) :=
by sorry

end central_angle_invariant_under_doubling_l3080_308017


namespace fermat_quotient_perfect_square_no_fermat_quotient_perfect_square_l3080_308018

theorem fermat_quotient_perfect_square (p : ℕ) (h : Prime p) :
  (∃ (x : ℕ), (7^(p-1) - 1) / p = x^2) ↔ p = 3 :=
sorry

theorem no_fermat_quotient_perfect_square (p : ℕ) (h : Prime p) :
  ¬∃ (x : ℕ), (11^(p-1) - 1) / p = x^2 :=
sorry

end fermat_quotient_perfect_square_no_fermat_quotient_perfect_square_l3080_308018


namespace parabola_properties_l3080_308095

-- Define the parabola function
def f (x : ℝ) : ℝ := (x - 1)^2 - 3

theorem parabola_properties :
  -- 1. The parabola opens upwards
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f ((x₁ + x₂) / 2) < (f x₁ + f x₂) / 2) ∧
  -- 2. The parabola intersects the x-axis at two distinct points
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  -- 3. The minimum value of y is -3 and occurs when x = 1
  (∀ x : ℝ, f x ≥ -3) ∧ (f 1 = -3) ∧
  -- 4. There exists an x > 1 such that y ≤ 0
  (∃ x : ℝ, x > 1 ∧ f x ≤ 0) :=
by sorry

end parabola_properties_l3080_308095


namespace purely_imaginary_complex_number_l3080_308062

theorem purely_imaginary_complex_number (x : ℝ) :
  let z : ℂ := 2 + Complex.I + (1 - Complex.I) * x
  (∃ (y : ℝ), z = Complex.I * y) → x = -2 :=
by
  sorry

end purely_imaginary_complex_number_l3080_308062


namespace alpha_value_l3080_308008

theorem alpha_value (α : Real) 
  (h1 : (1 - 4 * Real.sin α) / Real.tan α = Real.sqrt 3)
  (h2 : α ∈ Set.Ioo 0 (Real.pi / 2)) :
  α = Real.pi / 18 := by
  sorry

end alpha_value_l3080_308008


namespace labeling_existence_condition_l3080_308013

/-- A labeling of lattice points in Z^2 with positive integers -/
def Labeling := ℤ × ℤ → ℕ+

/-- The property that only finitely many distinct labels occur -/
def FiniteLabels (l : Labeling) : Prop :=
  ∃ (n : ℕ), ∀ (p : ℤ × ℤ), l p ≤ n

/-- The distance condition for a given c > 0 -/
def DistanceCondition (c : ℝ) (l : Labeling) : Prop :=
  ∀ (i : ℕ+) (p q : ℤ × ℤ), l p = i ∧ l q = i → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 : ℝ) ≥ c^(i : ℝ)

/-- The main theorem -/
theorem labeling_existence_condition (c : ℝ) :
  (c > 0 ∧
   ∃ (l : Labeling), FiniteLabels l ∧ DistanceCondition c l) ↔
  (c > 0 ∧ c < Real.sqrt 2) :=
sorry

end labeling_existence_condition_l3080_308013


namespace equation_solutions_l3080_308005

theorem equation_solutions :
  ∀ n m : ℕ, m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) :=
by sorry

end equation_solutions_l3080_308005


namespace complex_subtraction_l3080_308051

theorem complex_subtraction : (5 - 3*I) - (2 + 7*I) = 3 - 10*I := by sorry

end complex_subtraction_l3080_308051


namespace unique_digits_for_divisibility_l3080_308028

-- Define the number 13xy45z as a function of x, y, z
def number (x y z : ℕ) : ℕ := 13000000 + x * 100000 + y * 10000 + 4500 + z

-- Define the divisibility condition
def is_divisible_by_792 (n : ℕ) : Prop := n % 792 = 0

-- Theorem statement
theorem unique_digits_for_divisibility :
  ∃! (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ is_divisible_by_792 (number x y z) ∧ x = 2 ∧ y = 3 ∧ z = 6 := by
  sorry

end unique_digits_for_divisibility_l3080_308028


namespace fine_on_fifth_day_l3080_308064

/-- Calculates the fine for a given day -/
def dailyFine (previousFine : ℚ) : ℚ :=
  min (previousFine * 2) (previousFine + 0.15)

/-- Calculates the total fine up to a given day -/
def totalFine (day : ℕ) : ℚ :=
  match day with
  | 0 => 0
  | 1 => 0.05
  | n + 1 => totalFine n + dailyFine (dailyFine (totalFine n))

/-- The theorem to be proved -/
theorem fine_on_fifth_day :
  totalFine 5 = 1.35 := by
  sorry

end fine_on_fifth_day_l3080_308064


namespace min_sum_of_product_l3080_308014

theorem min_sum_of_product (a b c : ℕ+) (h : a * b * c = 2450) :
  ∃ (x y z : ℕ+), x * y * z = 2450 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 76 :=
sorry

end min_sum_of_product_l3080_308014


namespace minimal_sum_of_squares_l3080_308086

theorem minimal_sum_of_squares (a b c : ℕ+) : 
  a ≠ b → b ≠ c → a ≠ c →
  ∃ p q r : ℕ+, (a + b : ℕ) = p^2 ∧ (b + c : ℕ) = q^2 ∧ (a + c : ℕ) = r^2 →
  (a : ℕ) + b + c ≥ 55 :=
sorry

end minimal_sum_of_squares_l3080_308086


namespace special_triangle_unique_values_l3080_308072

/-- An isosceles triangle with a specific internal point -/
structure SpecialTriangle where
  -- The side length of the two equal sides
  s : ℝ
  -- The base length
  t : ℝ
  -- Coordinates of the internal point P
  px : ℝ
  py : ℝ
  -- Assertion that the triangle is isosceles
  h_isosceles : s > 0
  -- Assertion that P is inside the triangle
  h_inside : 0 < px ∧ px < t ∧ 0 < py ∧ py < s
  -- Distance from A to P is 2
  h_ap : px^2 + py^2 = 4
  -- Distance from B to P is 2√2
  h_bp : (t - px)^2 + py^2 = 8
  -- Distance from C to P is 3
  h_cp : px^2 + (s - py)^2 = 9

/-- The theorem stating the unique values of s and t -/
theorem special_triangle_unique_values (tri : SpecialTriangle) : 
  tri.s = 2 * Real.sqrt 3 ∧ tri.t = 6 := by sorry

end special_triangle_unique_values_l3080_308072


namespace lap_time_six_minutes_l3080_308056

/-- Represents a circular track with two photographers -/
structure CircularTrack :=
  (length : ℝ)
  (photographer1_position : ℝ)
  (photographer2_position : ℝ)

/-- Represents a runner on the circular track -/
structure Runner :=
  (speed : ℝ)
  (start_position : ℝ)

/-- Calculates the time spent closer to each photographer -/
def time_closer_to_photographer (track : CircularTrack) (runner : Runner) : ℝ × ℝ := sorry

/-- The main theorem to prove -/
theorem lap_time_six_minutes 
  (track : CircularTrack) 
  (runner : Runner) 
  (h1 : (time_closer_to_photographer track runner).1 = 2)
  (h2 : (time_closer_to_photographer track runner).2 = 3) :
  runner.speed * track.length = 6 * runner.speed := by sorry

end lap_time_six_minutes_l3080_308056


namespace f_composition_of_three_l3080_308055

def f (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem f_composition_of_three : f (f (f (f 3))) = 187 := by
  sorry

end f_composition_of_three_l3080_308055


namespace imaginary_part_of_z_l3080_308021

theorem imaginary_part_of_z (θ : ℝ) :
  let z : ℂ := Complex.mk (Real.sin (2 * θ) - 1) (Real.sqrt 2 * Real.cos θ - 1)
  (z.re = 0) → z.im = -2 := by
  sorry

end imaginary_part_of_z_l3080_308021


namespace equilateral_triangle_on_lines_l3080_308032

/-- Represents a line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle --/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Check if three lines are coplanar and equidistant --/
def are_coplanar_equidistant (l1 l2 l3 : Line) : Prop :=
  l1.slope = l2.slope ∧ l2.slope = l3.slope ∧
  |l2.intercept - l1.intercept| = |l3.intercept - l2.intercept|

/-- Check if a triangle is equilateral --/
def is_equilateral (t : Triangle) : Prop :=
  let d1 := ((t.b.x - t.a.x)^2 + (t.b.y - t.a.y)^2).sqrt
  let d2 := ((t.c.x - t.b.x)^2 + (t.c.y - t.b.y)^2).sqrt
  let d3 := ((t.a.x - t.c.x)^2 + (t.a.y - t.c.y)^2).sqrt
  d1 = d2 ∧ d2 = d3

/-- Check if a point lies on a line --/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Theorem: Given three coplanar and equidistant lines, 
    it is possible to construct an equilateral triangle 
    with its vertices lying on these lines --/
theorem equilateral_triangle_on_lines 
  (l1 l2 l3 : Line) 
  (h : are_coplanar_equidistant l1 l2 l3) :
  ∃ (t : Triangle), 
    is_equilateral t ∧ 
    point_on_line t.a l1 ∧ 
    point_on_line t.b l2 ∧ 
    point_on_line t.c l3 := by
  sorry

end equilateral_triangle_on_lines_l3080_308032


namespace marquita_garden_width_marquita_garden_width_proof_l3080_308097

/-- The width of Marquita's gardens given the conditions of the problem -/
theorem marquita_garden_width : ℝ :=
  let mancino_garden_count : ℕ := 3
  let mancino_garden_length : ℝ := 16
  let mancino_garden_width : ℝ := 5
  let marquita_garden_count : ℕ := 2
  let marquita_garden_length : ℝ := 8
  let total_area : ℝ := 304

  let mancino_total_area := mancino_garden_count * mancino_garden_length * mancino_garden_width
  let marquita_total_area := total_area - mancino_total_area
  let marquita_garden_area := marquita_total_area / marquita_garden_count
  let marquita_garden_width := marquita_garden_area / marquita_garden_length

  4

theorem marquita_garden_width_proof : marquita_garden_width = 4 := by
  sorry

end marquita_garden_width_marquita_garden_width_proof_l3080_308097


namespace complex_modulus_evaluation_l3080_308091

theorem complex_modulus_evaluation :
  Complex.abs (3 / 4 - 5 * Complex.I + (1 + 3 * Complex.I)) = Real.sqrt 113 / 4 := by
  sorry

end complex_modulus_evaluation_l3080_308091


namespace sample_size_proof_l3080_308025

theorem sample_size_proof (n : ℕ) 
  (h1 : ∃ k : ℕ, 2*k + 3*k + 4*k = 27) 
  (h2 : ∃ k : ℕ, n = 2*k + 3*k + 4*k + 6*k + 4*k + k) : n = 60 := by
  sorry

end sample_size_proof_l3080_308025


namespace journey_mpg_is_28_l3080_308006

/-- Calculates the average miles per gallon for a car journey -/
def average_mpg (initial_odometer final_odometer : ℕ) 
                (initial_fill first_refill second_refill : ℕ) : ℚ :=
  let total_distance := final_odometer - initial_odometer
  let total_gas := initial_fill + first_refill + second_refill
  (total_distance : ℚ) / total_gas

/-- Theorem stating that the average MPG for the given journey is 28 -/
theorem journey_mpg_is_28 :
  let initial_odometer := 56100
  let final_odometer := 57500
  let initial_fill := 10
  let first_refill := 15
  let second_refill := 25
  average_mpg initial_odometer final_odometer initial_fill first_refill second_refill = 28 := by
  sorry

#eval average_mpg 56100 57500 10 15 25

end journey_mpg_is_28_l3080_308006


namespace fraction_equation_solution_l3080_308069

theorem fraction_equation_solution (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -5 ∧ x ≠ 0 ∧ x ≠ 6 →
    (P / (x + 5) + Q / (x * (x - 6)) : ℝ) = (x^2 - 4*x + 20) / (x^3 + x^2 - 30*x)) →
  (Q : ℚ) / P = 4 := by
sorry

end fraction_equation_solution_l3080_308069


namespace events_mutually_exclusive_and_complementary_l3080_308004

def S : Set Nat := {1, 2, 3, 4, 5}

def A : Set Nat := {x ∈ S | x % 2 = 0}

def B : Set Nat := {x ∈ S | x % 2 = 1}

theorem events_mutually_exclusive_and_complementary :
  (A ∩ B = ∅) ∧ (A ∪ B = S) := by
  sorry

end events_mutually_exclusive_and_complementary_l3080_308004


namespace quadratic_max_value_l3080_308088

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_max_value 
  (a b c : ℝ) 
  (h1 : f a b c 1 = -40)
  (h2 : f a b c (-1) = -8)
  (h3 : f a b c (-3) = 8)
  (h4 : -b / (2 * a) = -4)
  (h5 : ∃ x₁ x₂, x₁ = -1 ∧ x₂ = -7 ∧ f a b c x₁ = -8 ∧ f a b c x₂ = -8)
  (h6 : a + b + c = -40) :
  ∃ x_max, ∀ x, f a b c x ≤ f a b c x_max ∧ f a b c x_max = 10 :=
sorry

end quadratic_max_value_l3080_308088


namespace vector_sum_magnitude_l3080_308020

/-- Given vectors a and b, prove that |a + 2b| = √7 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  a = (Real.cos (5 * π / 180), Real.sin (5 * π / 180)) →
  b = (Real.cos (65 * π / 180), Real.sin (65 * π / 180)) →
  ‖a + 2 • b‖ = Real.sqrt 7 := by
  sorry

end vector_sum_magnitude_l3080_308020


namespace three_parallel_lines_planes_l3080_308074

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a placeholder and may need to be adjusted based on Lean's geometry libraries

-- Define a predicate for parallel lines
def parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of parallel lines

-- Define a predicate for coplanar lines
def coplanar (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Definition of coplanar lines

-- Define a function to count planes through two lines
def count_planes_through_two_lines (l1 l2 : Line3D) : ℕ :=
  sorry -- Definition to count planes through two lines

-- Theorem statement
theorem three_parallel_lines_planes (a b c : Line3D) :
  parallel a b ∧ parallel b c ∧ parallel a c ∧ ¬coplanar a b c →
  (count_planes_through_two_lines a b +
   count_planes_through_two_lines b c +
   count_planes_through_two_lines a c) = 3 :=
by sorry

end three_parallel_lines_planes_l3080_308074


namespace f_monotonicity_and_maximum_l3080_308038

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - k * x^2

theorem f_monotonicity_and_maximum (k : ℝ) :
  (k = 1 →
    (∀ x y, x < y ∧ y < 0 → f k x < f k y) ∧
    (∀ x y, 0 < x ∧ x < y ∧ y < Real.log 2 → f k x > f k y) ∧
    (∀ x y, Real.log 2 < x ∧ x < y → f k x < f k y)) ∧
  (1/2 < k ∧ k ≤ 1 →
    ∀ x, x ∈ Set.Icc 0 k → f k x ≤ (k - 1) * Real.exp k - k^3) := by
  sorry

end f_monotonicity_and_maximum_l3080_308038


namespace special_arithmetic_progression_all_integer_l3080_308063

/-- An arithmetic progression with the property that the product of any two distinct terms is also a term. -/
structure SpecialArithmeticProgression where
  seq : ℕ → ℤ
  is_arithmetic : ∃ d : ℤ, ∀ n : ℕ, seq (n + 1) = seq n + d
  is_increasing : ∀ n : ℕ, seq (n + 1) > seq n
  product_property : ∀ m n : ℕ, m ≠ n → ∃ k : ℕ, seq m * seq n = seq k

/-- All terms in a SpecialArithmeticProgression are integers. -/
theorem special_arithmetic_progression_all_integer (ap : SpecialArithmeticProgression) : 
  ∀ n : ℕ, ∃ k : ℤ, ap.seq n = k :=
sorry

end special_arithmetic_progression_all_integer_l3080_308063


namespace sid_shopping_l3080_308068

def shopping_problem (initial_amount : ℕ) (snack_cost : ℕ) (remaining_amount_extra : ℕ) : Prop :=
  let computer_accessories_cost := initial_amount - snack_cost - (initial_amount / 2 + remaining_amount_extra)
  computer_accessories_cost = 12

theorem sid_shopping :
  shopping_problem 48 8 4 :=
sorry

end sid_shopping_l3080_308068


namespace quadratic_root_one_iff_sum_coeffs_zero_l3080_308034

theorem quadratic_root_one_iff_sum_coeffs_zero (a b c : ℝ) :
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = 1) ↔ a + b + c = 0 := by
  sorry

end quadratic_root_one_iff_sum_coeffs_zero_l3080_308034


namespace a_plus_b_value_l3080_308046

theorem a_plus_b_value (a b : ℝ) : 
  ((a + 2)^2 = 1 ∧ 3^3 = b - 3) → (a + b = 29 ∨ a + b = 27) := by
  sorry

end a_plus_b_value_l3080_308046


namespace beths_underwater_time_l3080_308011

/-- Calculates the total underwater time for a scuba diver -/
def total_underwater_time (primary_tank_time : ℕ) (supplemental_tanks : ℕ) (time_per_supplemental_tank : ℕ) : ℕ :=
  primary_tank_time + supplemental_tanks * time_per_supplemental_tank

/-- Proves that Beth's total underwater time is 8 hours -/
theorem beths_underwater_time :
  let primary_tank_time : ℕ := 2
  let supplemental_tanks : ℕ := 6
  let time_per_supplemental_tank : ℕ := 1
  total_underwater_time primary_tank_time supplemental_tanks time_per_supplemental_tank = 8 := by
  sorry

#eval total_underwater_time 2 6 1

end beths_underwater_time_l3080_308011


namespace l2_passes_through_point_perpendicular_implies_a_value_max_distance_to_l1_l3080_308054

-- Define the lines l1 and l2
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := 3 * x + (a - 1) * y + 3 - a = 0

-- Define the point P
def P : ℝ × ℝ := (1, 3)

-- Statement 1
theorem l2_passes_through_point : ∀ a : ℝ, l2 a (-2/3) 1 := by sorry

-- Statement 2
theorem perpendicular_implies_a_value : 
  ∀ a : ℝ, (∀ x y : ℝ, l1 a x y → l2 a x y → (a * 3 + 2 * (a - 1) = 0)) → a = 2/5 := by sorry

-- Statement 3
theorem max_distance_to_l1 : 
  ∀ a : ℝ, ∃ x y : ℝ, l1 a x y ∧ Real.sqrt ((x - P.1)^2 + (y - P.2)^2) = 5 := by sorry

end l2_passes_through_point_perpendicular_implies_a_value_max_distance_to_l1_l3080_308054


namespace parallel_vectors_k_value_l3080_308081

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k_value :
  let a : ℝ × ℝ := (6, 2)
  let b : ℝ × ℝ := (-2, k)
  parallel a b → k = -2/3 := by
  sorry

end parallel_vectors_k_value_l3080_308081


namespace fifty_cent_items_count_l3080_308016

def total_items : ℕ := 35
def total_price : ℕ := 4000  -- in cents

def is_valid_purchase (x y z : ℕ) : Prop :=
  x + y + z = total_items ∧
  50 * x + 300 * y + 400 * z = total_price

theorem fifty_cent_items_count : ∃ (x y z : ℕ), is_valid_purchase x y z ∧ x = 30 := by
  sorry

end fifty_cent_items_count_l3080_308016


namespace probability_mixed_selection_l3080_308024

/- Define the number of male and female students -/
def num_male : ℕ := 3
def num_female : ℕ := 4

/- Define the total number of students -/
def total_students : ℕ := num_male + num_female

/- Define the number of volunteers to be selected -/
def num_volunteers : ℕ := 3

/- Theorem stating the probability of selecting both male and female students -/
theorem probability_mixed_selection :
  (1 : ℚ) - (Nat.choose num_male num_volunteers + Nat.choose num_female num_volunteers : ℚ) / 
  (Nat.choose total_students num_volunteers : ℚ) = 6/7 := by
  sorry

end probability_mixed_selection_l3080_308024


namespace f_range_l3080_308094

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 1) / (x + 2)

theorem f_range :
  Set.range f = {y : ℝ | y < 1 ∨ y > 1} :=
sorry

end f_range_l3080_308094


namespace ajay_dal_gain_l3080_308065

/-- Calculates the total gain from a dal transaction -/
def calculate_gain (quantity1 : ℕ) (price1 : ℚ) (quantity2 : ℕ) (price2 : ℚ) (selling_price : ℚ) : ℚ :=
  let total_cost := quantity1 * price1 + quantity2 * price2
  let total_quantity := quantity1 + quantity2
  let total_revenue := total_quantity * selling_price
  total_revenue - total_cost

/-- Proves that Ajay's total gain in the dal transaction is Rs 27.50 -/
theorem ajay_dal_gain : calculate_gain 15 (14.5) 10 13 15 = (27.5) := by
  sorry

end ajay_dal_gain_l3080_308065


namespace janes_number_l3080_308049

theorem janes_number : ∃ x : ℚ, 5 * (3 * x + 16) = 250 ∧ x = 34/3 := by
  sorry

end janes_number_l3080_308049


namespace third_row_sum_is_401_l3080_308067

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the grid -/
def Grid := ℕ → ℕ → ℕ

/-- The size of the grid -/
def gridSize : ℕ := 16

/-- The starting position (centermost) -/
def startPos : Position :=
  { row := 9, col := 9 }

/-- Fills the grid in a clockwise spiral pattern -/
def fillGrid : Grid :=
  sorry

/-- Gets the numbers in a specific row -/
def getRowNumbers (g : Grid) (row : ℕ) : List ℕ :=
  sorry

/-- Theorem: The sum of the greatest and least number in the third row from the top is 401 -/
theorem third_row_sum_is_401 :
  let g := fillGrid
  let thirdRow := getRowNumbers g 3
  (List.maximum thirdRow).getD 0 + (List.minimum thirdRow).getD 0 = 401 := by
  sorry

end third_row_sum_is_401_l3080_308067


namespace probability_same_activity_l3080_308057

/-- The probability that two specific students participate in the same activity
    when four students are divided into two groups. -/
theorem probability_same_activity (n : ℕ) (m : ℕ) : 
  n = 4 → m = 2 → (m : ℚ) / (Nat.choose n 2) = 1 / 3 := by sorry

end probability_same_activity_l3080_308057


namespace exactly_seventeen_solutions_l3080_308002

/-- The number of ordered pairs of complex numbers satisfying the given equations -/
def num_solutions : ℕ := 17

/-- The property that a pair of complex numbers satisfies the given equations -/
def satisfies_equations (a b : ℂ) : Prop :=
  a^5 * b^3 = 1 ∧ a^9 * b^2 = 1

/-- The theorem stating that there are exactly 17 solutions -/
theorem exactly_seventeen_solutions :
  ∃! (s : Set (ℂ × ℂ)), 
    (∀ (p : ℂ × ℂ), p ∈ s ↔ satisfies_equations p.1 p.2) ∧
    Finite s ∧
    Nat.card s = num_solutions :=
sorry

end exactly_seventeen_solutions_l3080_308002


namespace root_product_equals_eight_l3080_308087

theorem root_product_equals_eight :
  (32 : ℝ) ^ (1/5 : ℝ) * (8 : ℝ) ^ (1/3 : ℝ) * (4 : ℝ) ^ (1/2 : ℝ) = 8 := by
  sorry

end root_product_equals_eight_l3080_308087


namespace cultivation_equation_correct_l3080_308052

/-- Represents the cultivation problem of a farmer --/
structure CultivationProblem where
  paddy_area : ℝ
  dry_area : ℝ
  dry_rate_difference : ℝ
  time_ratio : ℝ

/-- The equation representing the cultivation problem --/
def cultivation_equation (p : CultivationProblem) (x : ℝ) : Prop :=
  p.paddy_area / x = 2 * (p.dry_area / (x + p.dry_rate_difference))

/-- Theorem stating that the given equation correctly represents the cultivation problem --/
theorem cultivation_equation_correct (p : CultivationProblem) :
  p.paddy_area = 36 ∧ 
  p.dry_area = 30 ∧ 
  p.dry_rate_difference = 4 ∧ 
  p.time_ratio = 2 →
  ∃ x : ℝ, cultivation_equation p x :=
by sorry

end cultivation_equation_correct_l3080_308052


namespace complex_equation_sum_l3080_308026

theorem complex_equation_sum (a b : ℝ) : (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (2 + Complex.I) * (1 - b * Complex.I) = a + Complex.I →
  a + b = 2 := by sorry

end complex_equation_sum_l3080_308026


namespace binary_to_octal_equivalence_l3080_308030

-- Define the binary number
def binary_num : ℕ := 11011

-- Define the octal number
def octal_num : ℕ := 33

-- Theorem stating the equivalence of the binary and octal representations
theorem binary_to_octal_equivalence :
  (binary_num.digits 2).foldl (· + 2 * ·) 0 = (octal_num.digits 8).foldl (· + 8 * ·) 0 :=
by sorry

end binary_to_octal_equivalence_l3080_308030


namespace best_fit_model_l3080_308083

/-- Represents a regression model with a correlation coefficient -/
structure RegressionModel where
  R : ℝ
  h_R_range : R ≥ 0 ∧ R ≤ 1

/-- Defines when one model has a better fit than another -/
def better_fit (m1 m2 : RegressionModel) : Prop := m1.R > m2.R

theorem best_fit_model (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.R = 0.98)
  (h2 : model2.R = 0.80)
  (h3 : model3.R = 0.50)
  (h4 : model4.R = 0.25) :
  better_fit model1 model2 ∧ better_fit model1 model3 ∧ better_fit model1 model4 := by
  sorry


end best_fit_model_l3080_308083


namespace geometric_sequence_problem_l3080_308080

/-- Given that -l, a, b, c, and -9 form a geometric sequence, prove that b = -3 and ac = 9 -/
theorem geometric_sequence_problem (l a b c : ℝ) 
  (h1 : ∃ (r : ℝ), a / (-l) = r ∧ b / a = r ∧ c / b = r ∧ (-9) / c = r) : 
  b = -3 ∧ a * c = 9 := by
  sorry


end geometric_sequence_problem_l3080_308080


namespace divisibility_condition_l3080_308076

theorem divisibility_condition (a b : ℕ) (ha : a ≥ 3) (hb : b ≥ 3) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) →
  ∃ k : ℕ, k ≥ 1 ∧ a = 7 * k^2 ∧ b = 7 * k :=
by sorry

end divisibility_condition_l3080_308076


namespace fourth_term_of_geometric_progression_l3080_308022

/-- Given a geometric progression with the first three terms, find the fourth term -/
theorem fourth_term_of_geometric_progression (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3^(1/4 : ℝ)) 
    (h₂ : a₂ = 3^(1/6 : ℝ)) (h₃ : a₃ = 3^(1/12 : ℝ)) : 
  ∃ (a₄ : ℝ), a₄ = (a₃ * a₂) / a₁ ∧ a₄ = 1 := by
sorry


end fourth_term_of_geometric_progression_l3080_308022


namespace total_earnings_proof_l3080_308003

def total_earnings (jermaine_earnings terrence_earnings emilee_earnings : ℕ) : ℕ :=
  jermaine_earnings + terrence_earnings + emilee_earnings

theorem total_earnings_proof (terrence_earnings emilee_earnings : ℕ) 
  (h1 : terrence_earnings = 30)
  (h2 : emilee_earnings = 25) :
  total_earnings (terrence_earnings + 5) terrence_earnings emilee_earnings = 90 :=
by
  sorry

end total_earnings_proof_l3080_308003


namespace decreasing_function_inequality_l3080_308050

/-- A decreasing function on (0, +∞) -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f y < f x

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f (2 * a^2 + a + 1) < f (3 * a^2 - 4 * a + 1)) :
  (0 < a ∧ a < 1/3) ∨ (1 < a ∧ a < 5) :=
by
  sorry

end decreasing_function_inequality_l3080_308050
