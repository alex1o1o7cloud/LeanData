import Mathlib

namespace jeff_total_jars_l1111_111142

theorem jeff_total_jars (x : ℕ) : 
  16 * x + 28 * x + 40 * x + 52 * x = 2032 → 4 * x = 56 :=
by
  intro h
  -- additional steps to solve the problem would go here.
  sorry

end jeff_total_jars_l1111_111142


namespace combine_exponent_remains_unchanged_l1111_111137

-- Define combining like terms condition
def combining_like_terms (terms : List (ℕ × String)) : List (ℕ × String) := sorry

-- Define the problem statement
theorem combine_exponent_remains_unchanged (terms : List (ℕ × String)) : 
  (combining_like_terms terms).map Prod.snd = terms.map Prod.snd :=
sorry

end combine_exponent_remains_unchanged_l1111_111137


namespace new_average_after_exclusion_l1111_111127

theorem new_average_after_exclusion (S : ℕ) (h1 : S = 27 * 5) (excluded : ℕ) (h2 : excluded = 35) : (S - excluded) / 4 = 25 :=
by
  sorry

end new_average_after_exclusion_l1111_111127


namespace find_a_l1111_111185

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = 3 * x^(a-2) - 2) (h_cond : f 2 = 4) : a = 3 :=
by
  sorry

end find_a_l1111_111185


namespace new_temperature_l1111_111143

-- Define the initial temperature
variable (t : ℝ)

-- Define the temperature drop
def temperature_drop : ℝ := 2

-- State the theorem
theorem new_temperature (t : ℝ) (temperature_drop : ℝ) : t - temperature_drop = t - 2 :=
by
  sorry

end new_temperature_l1111_111143


namespace mark_has_3_tanks_l1111_111162

-- Define conditions
def pregnant_fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20
def total_young : ℕ := 240

-- Theorem statement that Mark has 3 tanks
theorem mark_has_3_tanks : (total_young / (pregnant_fish_per_tank * young_per_fish)) = 3 :=
by
  sorry

end mark_has_3_tanks_l1111_111162


namespace knights_on_red_chairs_l1111_111169

theorem knights_on_red_chairs (K L K_r L_b : ℕ) (h1: K + L = 20)
  (h2: K - K_r + L_b = 10) (h3: K_r + L - L_b = 10) (h4: K_r = L_b) : K_r = 5 := by
  sorry

end knights_on_red_chairs_l1111_111169


namespace total_trip_cost_l1111_111123

def distance_AC : ℝ := 4000
def distance_AB : ℝ := 4250
def bus_rate : ℝ := 0.10
def plane_rate : ℝ := 0.15
def boarding_fee : ℝ := 150

theorem total_trip_cost :
  let distance_BC := Real.sqrt (distance_AB ^ 2 - distance_AC ^ 2)
  let flight_cost := distance_AB * plane_rate + boarding_fee
  let bus_cost := distance_BC * bus_rate
  flight_cost + bus_cost = 931.15 :=
by
  sorry

end total_trip_cost_l1111_111123


namespace find_a_l1111_111181

theorem find_a (a x : ℝ) (h1 : 3 * x + 2 * a = 2) (h2 : x = 1) : a = -1/2 :=
by
  sorry

end find_a_l1111_111181


namespace bamboo_break_height_l1111_111136

theorem bamboo_break_height (x : ℝ) (h₁ : 0 < x) (h₂ : x < 9) (h₃ : x^2 + 3^2 = (9 - x)^2) : x = 4 :=
by
  sorry

end bamboo_break_height_l1111_111136


namespace y_coord_vertex_C_l1111_111151

/-- The coordinates of vertices A, B, and D are given as A(0,0), B(0,1), and D(3,1).
 Vertex C is directly above vertex B. The quadrilateral ABCD has a vertical line of symmetry 
 and the area of quadrilateral ABCD is 18 square units.
 Prove that the y-coordinate of vertex C is 11. -/
theorem y_coord_vertex_C (h : ℝ) 
  (A : ℝ × ℝ := (0, 0)) 
  (B : ℝ × ℝ := (0, 1)) 
  (D : ℝ × ℝ := (3, 1)) 
  (C : ℝ × ℝ := (0, h)) 
  (symmetry : C.fst = B.fst) 
  (area : 18 = 3 * 1 + (1 / 2) * 3 * (h - 1)) :
  h = 11 := 
by
  sorry

end y_coord_vertex_C_l1111_111151


namespace boat_downstream_distance_l1111_111101

theorem boat_downstream_distance (V_b V_s : ℝ) (t_downstream t_upstream : ℝ) (d_upstream : ℝ) 
  (h1 : t_downstream = 8) (h2 : t_upstream = 15) (h3 : d_upstream = 75) (h4 : V_s = 3.75) 
  (h5 : V_b - V_s = (d_upstream / t_upstream)) : (V_b + V_s) * t_downstream = 100 :=
by
  sorry

end boat_downstream_distance_l1111_111101


namespace quadratic_inequality_range_l1111_111125

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2 : ℝ) 2 :=
sorry

end quadratic_inequality_range_l1111_111125


namespace degree_of_expression_l1111_111160

open Polynomial

noncomputable def expr1 : Polynomial ℤ := (monomial 5 3 - monomial 3 2 + 4) * (monomial 12 2 - monomial 8 1 + monomial 6 5 - 15)
noncomputable def expr2 : Polynomial ℤ := (monomial 3 2 - 4) ^ 6
noncomputable def final_expr : Polynomial ℤ := expr1 - expr2

theorem degree_of_expression : degree final_expr = 18 := by
  sorry

end degree_of_expression_l1111_111160


namespace mutually_exclusive_but_not_complementary_l1111_111166

open Classical

namespace CardDistribution

inductive Card
| red | yellow | blue | white

inductive Person
| A | B | C | D

def Event_A_gets_red (distrib: Person → Card) : Prop :=
  distrib Person.A = Card.red

def Event_D_gets_red (distrib: Person → Card) : Prop :=
  distrib Person.D = Card.red

theorem mutually_exclusive_but_not_complementary :
  ∀ (distrib: Person → Card),
  (Event_A_gets_red distrib → ¬Event_D_gets_red distrib) ∧
  ¬(∀ (distrib: Person → Card), Event_A_gets_red distrib ∨ Event_D_gets_red distrib) := 
by
  sorry

end CardDistribution

end mutually_exclusive_but_not_complementary_l1111_111166


namespace find_interest_rate_l1111_111105

theorem find_interest_rate
  (P : ℝ) (CI : ℝ) (T : ℝ) (n : ℕ)
  (comp_int_formula : CI = P * ((1 + (r / (n : ℝ))) ^ (n * T)) - P) :
  r = 0.099 :=
by
  have h : CI = 788.13 := sorry
  have hP : P = 5000 := sorry
  have hT : T = 1.5 := sorry
  have hn : (n : ℝ) = 2 := sorry
  sorry

end find_interest_rate_l1111_111105


namespace denomination_of_remaining_notes_eq_500_l1111_111140

-- Definitions of the given conditions:
def total_money : ℕ := 10350
def total_notes : ℕ := 126
def n_50_notes : ℕ := 117

-- The theorem stating what we need to prove
theorem denomination_of_remaining_notes_eq_500 :
  ∃ (X : ℕ), X = 500 ∧ total_money = (n_50_notes * 50 + (total_notes - n_50_notes) * X) :=
by
sorry

end denomination_of_remaining_notes_eq_500_l1111_111140


namespace tangent_circle_radius_l1111_111121

theorem tangent_circle_radius (r1 r2 d : ℝ) (h1 : r2 = 2) (h2 : d = 5) (tangent : abs (r1 - r2) = d ∨ r1 + r2 = d) :
  r1 = 3 ∨ r1 = 7 :=
by
  sorry

end tangent_circle_radius_l1111_111121


namespace multiple_of_shorter_piece_l1111_111163

theorem multiple_of_shorter_piece :
  ∃ (m : ℕ), 
  (35 + (m * 35 + 15) = 120) ∧
  (m = 2) :=
by
  sorry

end multiple_of_shorter_piece_l1111_111163


namespace product_positivity_l1111_111145

theorem product_positivity (a b c d e f : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (h : ∀ {x y z u v w : ℝ}, x * y * z > 0 → ¬(u * v * w > 0) ∧ ¬(x * v * u > 0) ∧ ¬(y * z * u > 0) ∧ ¬(y * v * x > 0)) :
  b * d * e > 0 ∧ a * c * d < 0 ∧ a * c * e < 0 ∧ b * d * f < 0 ∧ b * e * f < 0 := 
by
  sorry

end product_positivity_l1111_111145


namespace real_solutions_of_equation_l1111_111170

theorem real_solutions_of_equation :
  (∃! x : ℝ, (5 * x) / (x^2 + 2 * x + 4) + (6 * x) / (x^2 - 6 * x + 4) = -4 / 3) :=
sorry

end real_solutions_of_equation_l1111_111170


namespace tank_capacity_l1111_111103

variable (C : ℝ)  -- total capacity of the tank

-- The tank is 5/8 full initially
axiom h1 : (5/8) * C + 15 = (19/24) * C

theorem tank_capacity : C = 90 :=
by
  sorry

end tank_capacity_l1111_111103


namespace square_side_length_eq_8_over_pi_l1111_111141

noncomputable def side_length_square : ℝ := 8 / Real.pi

theorem square_side_length_eq_8_over_pi :
  ∀ (s : ℝ),
  (4 * s = (Real.pi * (s / Real.sqrt 2) ^ 2) / 2) →
  s = side_length_square :=
by
  intro s h
  sorry

end square_side_length_eq_8_over_pi_l1111_111141


namespace least_three_digit_multiple_of_3_4_7_l1111_111180

theorem least_three_digit_multiple_of_3_4_7 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 7 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 → m % 3 = 0 ∧ m % 4 = 0 ∧ m % 7 = 0 → n ≤ m :=
  sorry

end least_three_digit_multiple_of_3_4_7_l1111_111180


namespace unique_solution_positive_n_l1111_111191

theorem unique_solution_positive_n (n : ℝ) : 
  ( ∃ x : ℝ, 4 * x^2 + n * x + 16 = 0 ∧ ∀ y : ℝ, 4 * y^2 + n * y + 16 = 0 → y = x ) → n = 16 := 
by {
  sorry
}

end unique_solution_positive_n_l1111_111191


namespace value_of_V3_l1111_111184

-- Define the polynomial function using Horner's rule
def f (x : ℤ) := (((((2 * x + 0) * x - 3) * x + 2) * x + 1) * x - 3)

-- Define the value of x
def x : ℤ := 2

-- Prove the value of V_3 when x = 2
theorem value_of_V3 : f x = 12 := by
  sorry

end value_of_V3_l1111_111184


namespace cost_per_meter_of_fencing_l1111_111175

/-- A rectangular farm has area 1200 m², a short side of 30 m, and total job cost 1560 Rs.
    Prove that the cost of fencing per meter is 13 Rs. -/
theorem cost_per_meter_of_fencing
  (A : ℝ := 1200)
  (W : ℝ := 30)
  (job_cost : ℝ := 1560)
  (L : ℝ := A / W)
  (D : ℝ := Real.sqrt (L^2 + W^2))
  (total_length : ℝ := L + W + D) :
  job_cost / total_length = 13 := 
sorry

end cost_per_meter_of_fencing_l1111_111175


namespace ceil_floor_difference_l1111_111153

theorem ceil_floor_difference : 
  (Int.ceil ((15 : ℚ) / 8 * ((-34 : ℚ) / 4)) - Int.floor (((15 : ℚ) / 8) * Int.ceil ((-34 : ℚ) / 4)) = 0) :=
by 
  sorry

end ceil_floor_difference_l1111_111153


namespace cost_of_tax_free_items_l1111_111147

theorem cost_of_tax_free_items : 
  ∀ (total_spent : ℝ) (sales_tax : ℝ) (tax_rate : ℝ) (taxable_cost : ℝ),
  total_spent = 25 ∧ sales_tax = 0.30 ∧ tax_rate = 0.05 ∧ sales_tax = tax_rate * taxable_cost → 
  total_spent - taxable_cost = 19 :=
by
  intros total_spent sales_tax tax_rate taxable_cost
  intro h
  sorry

end cost_of_tax_free_items_l1111_111147


namespace max_area_rectangle_l1111_111139

theorem max_area_rectangle (l w : ℕ) (h : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
sorry

end max_area_rectangle_l1111_111139


namespace smallest_nine_digit_times_smallest_seven_digit_l1111_111172

theorem smallest_nine_digit_times_smallest_seven_digit :
  let smallest_nine_digit := 100000000
  let smallest_seven_digit := 1000000
  smallest_nine_digit = 100 * smallest_seven_digit :=
by
  sorry

end smallest_nine_digit_times_smallest_seven_digit_l1111_111172


namespace base8_to_base10_problem_l1111_111152

theorem base8_to_base10_problem (c d : ℕ) (h : 543 = 3*8^2 + c*8 + d) : (c * d) / 12 = 5 / 4 :=
by 
  sorry

end base8_to_base10_problem_l1111_111152


namespace carnations_in_last_three_bouquets_l1111_111118

/--
Trevor buys six bouquets of carnations.
In the first bouquet, there are 9.5 carnations.
In the second bouquet, there are 14.25 carnations.
In the third bouquet, there are 18.75 carnations.
The average number of carnations in all six bouquets is 16.
Prove that the total number of carnations in the fourth, fifth, and sixth bouquets combined is 53.5.
-/
theorem carnations_in_last_three_bouquets:
  let bouquet1 := 9.5
  let bouquet2 := 14.25
  let bouquet3 := 18.75
  let total_bouquets := 6
  let average_per_bouquet := 16
  let total_carnations := average_per_bouquet * total_bouquets
  let remaining_carnations := total_carnations - (bouquet1 + bouquet2 + bouquet3)
  remaining_carnations = 53.5 :=
by
  sorry

end carnations_in_last_three_bouquets_l1111_111118


namespace g_at_4_l1111_111157

def g (x : ℝ) : ℝ := 5 * x + 6

theorem g_at_4 : g 4 = 26 :=
by
  sorry

end g_at_4_l1111_111157


namespace current_tree_height_in_inches_l1111_111110

-- Constants
def initial_height_ft : ℝ := 10
def growth_percentage : ℝ := 0.50
def feet_to_inches : ℝ := 12

-- Conditions
def growth_ft : ℝ := growth_percentage * initial_height_ft
def current_height_ft : ℝ := initial_height_ft + growth_ft

-- Question/Answer equivalence
theorem current_tree_height_in_inches :
  (current_height_ft * feet_to_inches) = 180 :=
by 
  sorry

end current_tree_height_in_inches_l1111_111110


namespace domain_of_sqrt_expr_l1111_111119

theorem domain_of_sqrt_expr (x : ℝ) : x ≥ 3 ∧ x < 8 ↔ x ∈ Set.Ico 3 8 :=
by
  sorry

end domain_of_sqrt_expr_l1111_111119


namespace johns_age_l1111_111100

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l1111_111100


namespace necessary_but_not_sufficient_condition_l1111_111130

noncomputable def condition_sufficiency (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + m*x + 1 > 0

theorem necessary_but_not_sufficient_condition (m : ℝ) : m < 2 → (¬ condition_sufficiency m ∨ condition_sufficiency m) :=
by
  sorry

end necessary_but_not_sufficient_condition_l1111_111130


namespace necessary_but_not_sufficient_condition_l1111_111144

open Real

-- Define α as an internal angle of a triangle
def is_internal_angle (α : ℝ) : Prop := (0 < α ∧ α < π)

-- Given conditions
axiom α : ℝ
axiom h1 : is_internal_angle α

-- Prove: if (α ≠ π / 6) then (sin α ≠ 1 / 2) is a necessary but not sufficient condition 
theorem necessary_but_not_sufficient_condition : 
  (α ≠ π / 6) ∧ ¬((α ≠ π / 6) → (sin α ≠ 1 / 2)) ∧ ((sin α ≠ 1 / 2) → (α ≠ π / 6)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l1111_111144


namespace find_a_l1111_111113

-- Definitions for conditions
def line_equation (a : ℝ) (x y : ℝ) := a * x - y - 1 = 0
def angle_of_inclination (θ : ℝ) := θ = Real.pi / 3

-- The main theorem statement
theorem find_a (a : ℝ) (θ : ℝ) (h1 : angle_of_inclination θ) (h2 : a = Real.tan θ) : a = Real.sqrt 3 :=
 by
   -- skipping the proof
   sorry

end find_a_l1111_111113


namespace lines_intersection_points_l1111_111124

theorem lines_intersection_points :
  let line1 (x y : ℝ) := 2 * y - 3 * x = 4
  let line2 (x y : ℝ) := 3 * x + y = 5
  let line3 (x y : ℝ) := 6 * x - 4 * y = 8
  ∃ p1 p2 : (ℝ × ℝ),
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
    (p1 = (2, 5)) ∧ (p2 = (14/9, 1/3)) :=
by
  sorry

end lines_intersection_points_l1111_111124


namespace curves_intersect_at_four_points_l1111_111150

theorem curves_intersect_at_four_points (a : ℝ) :
  (∃ x y : ℝ, (x^2 + y^2 = a^2 ∧ y = -x^2 + a ) ∧ 
   (0 = x ∧ y = a) ∧ 
   (∃ t : ℝ, x = t ∧ (y = 1 ∧ x^2 = a - 1))) ↔ a = 2 := 
by
  sorry

end curves_intersect_at_four_points_l1111_111150


namespace solve_for_m_l1111_111199

theorem solve_for_m (m : ℝ) (f g : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^2 - 2 * x + m) →
  (∀ x : ℝ, g x = x^2 - 2 * x + 9 * m) →
  f 2 = 2 * g 2 →
  m = 0 :=
  by
    intros hf hg hs
    sorry

end solve_for_m_l1111_111199


namespace algebraic_expression_value_l1111_111126

theorem algebraic_expression_value (x : ℝ) (h : x ^ 2 - 3 * x = 4) : 2 * x ^ 2 - 6 * x - 3 = 5 :=
by
  sorry

end algebraic_expression_value_l1111_111126


namespace plane_speeds_l1111_111111

-- Define the speeds of the planes
def speed_slower (x : ℕ) := x
def speed_faster (x : ℕ) := 2 * x

-- Define the distances each plane travels in 3 hours
def distance_slower (x : ℕ) := 3 * speed_slower x
def distance_faster (x : ℕ) := 3 * speed_faster x

-- Define the total distance
def total_distance (x : ℕ) := distance_slower x + distance_faster x

-- Prove the speeds given the total distance
theorem plane_speeds (x : ℕ) (h : total_distance x = 2700) : speed_slower x = 300 ∧ speed_faster x = 600 :=
by {
  sorry
}

end plane_speeds_l1111_111111


namespace doritos_ratio_l1111_111104

noncomputable def bags_of_chips : ℕ := 80
noncomputable def bags_per_pile : ℕ := 5
noncomputable def piles : ℕ := 4

theorem doritos_ratio (D T : ℕ) (h1 : T = bags_of_chips)
  (h2 : D = piles * bags_per_pile) :
  (D : ℚ) / T = 1 / 4 := by
  sorry

end doritos_ratio_l1111_111104


namespace find_a_equiv_l1111_111193

noncomputable def A (a : ℝ) : Set ℝ := {1, 3, a^2}
noncomputable def B (a : ℝ) : Set ℝ := {1, 2 + a}

theorem find_a_equiv (a : ℝ) (h : A a ∪ B a = A a) : a = 2 :=
by
  sorry

end find_a_equiv_l1111_111193


namespace minimum_value_fraction_l1111_111132

theorem minimum_value_fraction (a b : ℝ) (h1 : a > 1) (h2 : b > 2) (h3 : 2 * a + b - 6 = 0) :
  (1 / (a - 1) + 2 / (b - 2)) = 4 := 
  sorry

end minimum_value_fraction_l1111_111132


namespace percentage_of_left_handed_women_l1111_111167

variable (x y : Nat) (h_ratio_rh_lh : 3 * x = 1 * x)
variable (h_ratio_men_women : 3 * y = 2 * y)
variable (h_rh_men_max : True)

theorem percentage_of_left_handed_women :
  (x / (4 * x)) * 100 = 25 :=
by sorry

end percentage_of_left_handed_women_l1111_111167


namespace batsman_average_increase_l1111_111177

theorem batsman_average_increase (A : ℕ) (H1 : 16 * A + 85 = 17 * (A + 3)) : A + 3 = 37 :=
by {
  sorry
}

end batsman_average_increase_l1111_111177


namespace marble_count_l1111_111107

theorem marble_count (x : ℕ) 
  (h1 : ∀ (Liam Mia Noah Olivia: ℕ), Mia = 3 * Liam ∧ Noah = 4 * Mia ∧ Olivia = 2 * Noah)
  (h2 : Liam + Mia + Noah + Olivia = 156)
  : x = 4 :=
by sorry

end marble_count_l1111_111107


namespace gcd_1443_999_l1111_111116

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by
  sorry

end gcd_1443_999_l1111_111116


namespace athlete_difference_l1111_111112

-- Define the conditions
def initial_athletes : ℕ := 300
def rate_of_leaving : ℕ := 28
def time_of_leaving : ℕ := 4
def rate_of_arriving : ℕ := 15
def time_of_arriving : ℕ := 7

-- Define intermediary calculations
def number_leaving : ℕ := rate_of_leaving * time_of_leaving
def remaining_athletes : ℕ := initial_athletes - number_leaving
def number_arriving : ℕ := rate_of_arriving * time_of_arriving
def total_sunday_night : ℕ := remaining_athletes + number_arriving

-- Theorem statement
theorem athlete_difference : initial_athletes - total_sunday_night = 7 :=
by
  sorry

end athlete_difference_l1111_111112


namespace minimum_unused_area_for_given_shapes_l1111_111192

def remaining_area (side_length : ℕ) (total_area used_area : ℕ) : ℕ :=
  total_area - used_area

theorem minimum_unused_area_for_given_shapes : (remaining_area 5 (5 * 5) (2 * 2 + 1 * 3 + 2 * 1) = 16) :=
by
  -- We skip the proof here, as instructed.
  sorry

end minimum_unused_area_for_given_shapes_l1111_111192


namespace equal_roots_quadratic_eq_l1111_111138

theorem equal_roots_quadratic_eq (m n : ℝ) (h : m^2 - 4 * n = 0) : m = 2 ∧ n = 1 :=
by
  sorry

end equal_roots_quadratic_eq_l1111_111138


namespace change_correct_l1111_111154

def cost_gum : ℕ := 350
def cost_protractor : ℕ := 500
def amount_paid : ℕ := 1000

theorem change_correct : amount_paid - (cost_gum + cost_protractor) = 150 := by
  sorry

end change_correct_l1111_111154


namespace smaller_bills_denomination_correct_l1111_111173

noncomputable def denomination_of_smaller_bills : ℕ :=
  let total_money := 1000
  let part_smaller_bills := 3 / 10
  let smaller_bills_amount := part_smaller_bills * total_money
  let rest_of_money := total_money - smaller_bills_amount
  let bill_100_denomination := 100
  let total_bills := 13
  let num_100_bills := rest_of_money / bill_100_denomination
  let num_smaller_bills := total_bills - num_100_bills
  let denomination := smaller_bills_amount / num_smaller_bills
  denomination

theorem smaller_bills_denomination_correct : denomination_of_smaller_bills = 50 := by
  sorry

end smaller_bills_denomination_correct_l1111_111173


namespace discount_received_l1111_111159

theorem discount_received (original_cost : ℝ) (amt_spent : ℝ) (discount : ℝ) 
  (h1 : original_cost = 467) (h2 : amt_spent = 68) : 
  discount = 399 :=
by
  sorry

end discount_received_l1111_111159


namespace arithmetic_sequence_75th_term_diff_l1111_111186

noncomputable def sum_arith_sequence (n : ℕ) (a d : ℚ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_75th_term_diff {n : ℕ} {a d : ℚ}
  (hn : n = 150)
  (sum_seq : sum_arith_sequence n a d = 15000)
  (term_range : ∀ k, 0 ≤ k ∧ k < n → 20 ≤ a + k * d ∧ a + k * d ≤ 150)
  (t75th : ∃ L G, L = a + 74 * d ∧ G = a + 74 * d) :
  G - L = (7500 / 149) :=
sorry

end arithmetic_sequence_75th_term_diff_l1111_111186


namespace problem_statement_l1111_111133

theorem problem_statement (m n : ℤ) (h : 3 * m - n = 1) : 9 * m ^ 2 - n ^ 2 - 2 * n = 1 := 
by sorry

end problem_statement_l1111_111133


namespace non_divisible_by_twenty_l1111_111135

theorem non_divisible_by_twenty (k : ℤ) (h : ∃ m : ℤ, k * (k + 1) * (k + 2) = 5 * m) :
  ¬ (∃ l : ℤ, k * (k + 1) * (k + 2) = 20 * l) := sorry

end non_divisible_by_twenty_l1111_111135


namespace tank_breadth_l1111_111198

/-
  We need to define the conditions:
  1. The field dimensions.
  2. The tank dimensions (length and depth), and the unknown breadth.
  3. The relationship after the tank is dug.
-/

noncomputable def field_length : ℝ := 90
noncomputable def field_breadth : ℝ := 50
noncomputable def tank_length : ℝ := 25
noncomputable def tank_depth : ℝ := 4
noncomputable def rise_in_level : ℝ := 0.5

theorem tank_breadth (B : ℝ) (h : 100 * B = (field_length * field_breadth - tank_length * B) * rise_in_level) : B = 20 :=
by sorry

end tank_breadth_l1111_111198


namespace greatest_x_l1111_111149

theorem greatest_x (x : ℕ) (h : x^2 < 32) : x ≤ 5 := 
sorry

end greatest_x_l1111_111149


namespace final_price_correct_l1111_111195

open BigOperators

-- Define the constants used in the problem
def original_price : ℝ := 500
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.10
def state_tax : ℝ := 0.05

-- Define the calculation steps
def price_after_first_discount : ℝ := original_price * (1 - first_discount)
def price_after_second_discount : ℝ := price_after_first_discount * (1 - second_discount)
def final_price : ℝ := price_after_second_discount * (1 + state_tax)

-- Prove that the final price is 354.375
theorem final_price_correct : final_price = 354.375 :=
by
  sorry

end final_price_correct_l1111_111195


namespace find_abc_solutions_l1111_111197

theorem find_abc_solutions :
  ∀ (a b c : ℕ),
    (2^(a) * 3^(b) = 7^(c) - 1) ↔
    ((a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 4 ∧ b = 1 ∧ c = 2)) :=
by
  sorry

end find_abc_solutions_l1111_111197


namespace complex_number_real_imaginary_opposite_l1111_111102

theorem complex_number_real_imaginary_opposite (a : ℝ) (i : ℂ) (comp : z = (1 - a * i) * i):
  (z.re = -z.im) → a = 1 :=
by 
  sorry

end complex_number_real_imaginary_opposite_l1111_111102


namespace tom_driving_speed_l1111_111179

theorem tom_driving_speed
  (v : ℝ)
  (hKarenSpeed : 60 = 60) -- Karen drives at an average speed of 60 mph
  (hKarenLateStart: 4 / 60 = 1 / 15) -- Karen starts 4 minutes late, which is 1/15 hours
  (hTomDistance : 24 = 24) -- Tom drives 24 miles before Karen wins the bet
  (hTimeEquation: 24 / v = 8 / 15): -- The equation derived from given conditions
  v = 45 := 
by
  sorry

end tom_driving_speed_l1111_111179


namespace probability_at_least_one_hit_l1111_111108

theorem probability_at_least_one_hit (pA pB pC : ℝ) (hA : pA = 0.7) (hB : pB = 0.5) (hC : pC = 0.4) : 
  (1 - ((1 - pA) * (1 - pB) * (1 - pC))) = 0.91 :=
by
  sorry

end probability_at_least_one_hit_l1111_111108


namespace cloves_needed_l1111_111168

theorem cloves_needed (cv_fp : 3 / 2 = 1.5) (cw_fp : 3 / 3 = 1) (vc_fp : 3 / 8 = 0.375) : 
  let cloves_for_vampires := 45
  let cloves_for_wights := 12
  let cloves_for_bats := 15
  30 * (3 / 2) + 12 * (3 / 3) + 40 * (3 / 8) = 72 := by
  sorry

end cloves_needed_l1111_111168


namespace parabola_directrix_l1111_111189

theorem parabola_directrix (a : ℝ) :
  (∃ y : ℝ, y = ax^2 ∧ y = -2) → a = 1/8 :=
by
  -- Solution steps are omitted.
  sorry

end parabola_directrix_l1111_111189


namespace abc_greater_than_n_l1111_111196

theorem abc_greater_than_n
  (a b c n : ℕ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : 1 < n)
  (h5 : a ^ n + b ^ n = c ^ n) :
  a > n ∧ b > n ∧ c > n :=
sorry

end abc_greater_than_n_l1111_111196


namespace tom_hockey_games_l1111_111115

theorem tom_hockey_games (g_this_year g_last_year : ℕ) 
  (h1 : g_this_year = 4)
  (h2 : g_last_year = 9) 
  : g_this_year + g_last_year = 13 := 
by
  sorry

end tom_hockey_games_l1111_111115


namespace minimum_of_quadratic_l1111_111182

theorem minimum_of_quadratic : ∀ x : ℝ, 1 ≤ x^2 - 6 * x + 10 :=
by
  intro x
  have h : x^2 - 6 * x + 10 = (x - 3)^2 + 1 := by ring
  rw [h]
  have h_nonneg : (x - 3)^2 ≥ 0 := by apply sq_nonneg
  linarith

end minimum_of_quadratic_l1111_111182


namespace percentage_increase_direct_proportionality_l1111_111187

variable (x y k q : ℝ)
variable (h1 : x = k * y)
variable (h2 : x' = x * (1 + q / 100))

theorem percentage_increase_direct_proportionality :
  ∃ q_percent : ℝ, y' = y * (1 + q_percent / 100) ∧ q_percent = q := sorry

end percentage_increase_direct_proportionality_l1111_111187


namespace sum_of_corners_of_9x9_grid_l1111_111165

theorem sum_of_corners_of_9x9_grid : 
    let topLeft := 1
    let topRight := 9
    let bottomLeft := 73
    let bottomRight := 81
    topLeft + topRight + bottomLeft + bottomRight = 164 :=
by {
  sorry
}

end sum_of_corners_of_9x9_grid_l1111_111165


namespace simplified_polynomial_l1111_111156

theorem simplified_polynomial : ∀ (x : ℝ), (3 * x + 2) * (3 * x - 2) - (3 * x - 1) ^ 2 = 6 * x - 5 := by
  sorry

end simplified_polynomial_l1111_111156


namespace infinite_fractions_2_over_odd_l1111_111183

theorem infinite_fractions_2_over_odd (a b : ℕ) (n : ℕ) : 
  (a = 2 → 2 * b + 1 ≠ 0) ∧ ((b = 2 * n + 1) → (2 + 2) / (2 * (2 * n + 1)) = 2 / (2 * n + 1)) ∧ (a / b = 2 / (2 * n + 1)) :=
by
  sorry

end infinite_fractions_2_over_odd_l1111_111183


namespace no_rain_either_day_l1111_111161

noncomputable def P_A := 0.62
noncomputable def P_B := 0.54
noncomputable def P_A_and_B := 0.44
noncomputable def P_A_or_B := P_A + P_B - P_A_and_B -- Applying Inclusion-Exclusion principle.
noncomputable def P_A_and_B_complement := 1 - P_A_or_B -- Complement of P(A ∪ B).

theorem no_rain_either_day :
  P_A_and_B_complement = 0.28 :=
by
  unfold P_A_and_B_complement P_A_or_B
  unfold P_A P_B P_A_and_B
  simp
  sorry

end no_rain_either_day_l1111_111161


namespace tyler_common_ratio_l1111_111134

theorem tyler_common_ratio (a r : ℝ) 
  (h1 : a / (1 - r) = 10)
  (h2 : (a + 4) / (1 - r) = 15) : 
  r = 1 / 5 :=
by
  sorry

end tyler_common_ratio_l1111_111134


namespace area_fraction_of_rhombus_in_square_l1111_111155

theorem area_fraction_of_rhombus_in_square :
  let n := 7                 -- grid size
  let side_length := n - 1   -- side length of the square
  let square_area := side_length^2 -- area of the square
  let rhombus_side := Real.sqrt 2 -- side length of the rhombus
  let rhombus_area := 2      -- area of the rhombus
  (rhombus_area / square_area) = 1 / 18 := sorry

end area_fraction_of_rhombus_in_square_l1111_111155


namespace andy_demerits_for_joke_l1111_111106

def max_demerits := 50
def demerits_late_per_instance := 2
def instances_late := 6
def remaining_demerits := 23
def total_demerits := max_demerits - remaining_demerits
def demerits_late := demerits_late_per_instance * instances_late
def demerits_joke := total_demerits - demerits_late

theorem andy_demerits_for_joke : demerits_joke = 15 := by
  sorry

end andy_demerits_for_joke_l1111_111106


namespace max_brownie_pieces_l1111_111128

theorem max_brownie_pieces (base height piece_width piece_height : ℕ) 
    (h_base : base = 30) (h_height : height = 24)
    (h_piece_width : piece_width = 3) (h_piece_height : piece_height = 4) :
  (base / piece_width) * (height / piece_height) = 60 :=
by sorry

end max_brownie_pieces_l1111_111128


namespace primes_ge_3_are_4k_pm1_infinitely_many_primes_4k_minus1_l1111_111178

-- Part 1: Prove that every prime number >= 3 is of the form 4k-1 or 4k+1
theorem primes_ge_3_are_4k_pm1 (p : ℕ) (hp_prime: Nat.Prime p) (hp_ge_3: p ≥ 3) : 
  ∃ k : ℕ, p = 4 * k + 1 ∨ p = 4 * k - 1 :=
by
  sorry

-- Part 2: Prove that there are infinitely many primes of the form 4k-1
theorem infinitely_many_primes_4k_minus1 : 
  ∀ (n : ℕ), ∃ (p : ℕ), Nat.Prime p ∧ p = 4 * k - 1 ∧ p > n :=
by
  sorry

end primes_ge_3_are_4k_pm1_infinitely_many_primes_4k_minus1_l1111_111178


namespace B_finishes_in_4_days_l1111_111109

theorem B_finishes_in_4_days
  (A_days : ℕ) (B_days : ℕ) (working_days_together : ℕ) 
  (A_rate : ℝ) (B_rate : ℝ) (combined_rate : ℝ) (work_done : ℝ) (remaining_work : ℝ)
  (B_rate_alone : ℝ) (days_B: ℝ) :
  A_days = 5 →
  B_days = 10 →
  working_days_together = 2 →
  A_rate = 1 / A_days →
  B_rate = 1 / B_days →
  combined_rate = A_rate + B_rate →
  work_done = combined_rate * working_days_together →
  remaining_work = 1 - work_done →
  B_rate_alone = 1 / B_days →
  days_B = remaining_work / B_rate_alone →
  days_B = 4 := 
by
  intros
  sorry

end B_finishes_in_4_days_l1111_111109


namespace problem_statement_l1111_111158

-- Define complex number i
noncomputable def i : ℂ := Complex.I

-- Define x as per the problem statement
noncomputable def x : ℂ := (1 + i * Real.sqrt 3) / 2

-- The main proposition to prove
theorem problem_statement : (1 / (x^2 - x)) = -1 :=
  sorry

end problem_statement_l1111_111158


namespace janice_bottle_caps_l1111_111171

-- Define the conditions
def num_boxes : ℕ := 79
def caps_per_box : ℕ := 4

-- Define the question as a theorem to prove
theorem janice_bottle_caps : num_boxes * caps_per_box = 316 :=
by
  sorry

end janice_bottle_caps_l1111_111171


namespace system_of_equations_solution_l1111_111148

theorem system_of_equations_solution :
  ∀ (x : Fin 100 → ℝ), 
  (x 0 + x 1 + x 2 = 0) ∧ 
  (x 1 + x 2 + x 3 = 0) ∧ 
  -- Continue for all other equations up to
  (x 98 + x 99 + x 0 = 0) ∧ 
  (x 99 + x 0 + x 1 = 0)
  → ∀ (i : Fin 100), x i = 0 :=
by
  intros x h
  -- We can insert the detailed solving steps here
  sorry

end system_of_equations_solution_l1111_111148


namespace endangered_animal_population_after_3_years_l1111_111131

-- Given conditions and definitions
def population (m : ℕ) (n : ℕ) : ℝ := m * (0.90 ^ n)

theorem endangered_animal_population_after_3_years :
  population 8000 3 = 5832 :=
by
  sorry

end endangered_animal_population_after_3_years_l1111_111131


namespace polynomial_at_one_l1111_111190

def f (x : ℝ) : ℝ := x^4 - 7*x^3 - 9*x^2 + 11*x + 7

theorem polynomial_at_one :
  f 1 = 3 := 
by
  sorry

end polynomial_at_one_l1111_111190


namespace problem_part1_problem_part2_l1111_111120

variable (a : ℝ)

def quadratic_solution_set_1 := {x : ℝ | x^2 + 2*x + a = 0}
def quadratic_solution_set_2 := {x : ℝ | a*x^2 + 2*x + 2 = 0}

theorem problem_part1 :
  (quadratic_solution_set_1 a = ∅ ∨ quadratic_solution_set_2 a = ∅) ∧ ¬ (quadratic_solution_set_1 a = ∅ ∧ quadratic_solution_set_2 a = ∅) →
  (1/2 < a ∧ a ≤ 1) :=
sorry

theorem problem_part2 :
  quadratic_solution_set_1 a ∪ quadratic_solution_set_2 a ≠ ∅ →
  a ≤ 1 :=
sorry

end problem_part1_problem_part2_l1111_111120


namespace megan_roles_other_than_lead_l1111_111122

def total_projects : ℕ := 800

def theater_percentage : ℚ := 50 / 100
def films_percentage : ℚ := 30 / 100
def television_percentage : ℚ := 20 / 100

def theater_lead_percentage : ℚ := 55 / 100
def theater_support_percentage : ℚ := 30 / 100
def theater_ensemble_percentage : ℚ := 10 / 100
def theater_cameo_percentage : ℚ := 5 / 100

def films_lead_percentage : ℚ := 70 / 100
def films_support_percentage : ℚ := 20 / 100
def films_minor_percentage : ℚ := 7 / 100
def films_cameo_percentage : ℚ := 3 / 100

def television_lead_percentage : ℚ := 60 / 100
def television_support_percentage : ℚ := 25 / 100
def television_recurring_percentage : ℚ := 10 / 100
def television_guest_percentage : ℚ := 5 / 100

theorem megan_roles_other_than_lead :
  let theater_projects := total_projects * theater_percentage
  let films_projects := total_projects * films_percentage
  let television_projects := total_projects * television_percentage

  let theater_other_roles := (theater_projects * theater_support_percentage) + 
                             (theater_projects * theater_ensemble_percentage) + 
                             (theater_projects * theater_cameo_percentage)

  let films_other_roles := (films_projects * films_support_percentage) + 
                           (films_projects * films_minor_percentage) + 
                           (films_projects * films_cameo_percentage)

  let television_other_roles := (television_projects * television_support_percentage) + 
                                (television_projects * television_recurring_percentage) + 
                                (television_projects * television_guest_percentage)
  
  theater_other_roles + films_other_roles + television_other_roles = 316 :=
by
  sorry

end megan_roles_other_than_lead_l1111_111122


namespace find_f_half_l1111_111114

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_half (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ Real.pi / 2) (h₁ : f (Real.sin x) = x) : 
  f (1 / 2) = Real.pi / 6 :=
sorry

end find_f_half_l1111_111114


namespace diameter_of_lid_is_2_inches_l1111_111117

noncomputable def π : ℝ := 3.14
def C : ℝ := 6.28

theorem diameter_of_lid_is_2_inches (d : ℝ) : d = C / π → d = 2 :=
by
  intro h
  sorry

end diameter_of_lid_is_2_inches_l1111_111117


namespace a2_add_a8_l1111_111188

variable (a : ℕ → ℝ) -- a_n is an arithmetic sequence
variable (d : ℝ) -- common difference

-- Condition stating that a_n is an arithmetic sequence with common difference d
axiom arithmetic_sequence : ∀ n, a (n + 1) = a n + d

-- Given condition a_3 + a_4 + a_5 + a_6 + a_7 = 450
axiom given_condition : a 3 + a 4 + a 5 + a 6 + a 7 = 450

theorem a2_add_a8 : a 2 + a 8 = 180 :=
by
  sorry

end a2_add_a8_l1111_111188


namespace ratio_of_part_diminished_by_10_to_whole_number_l1111_111129

theorem ratio_of_part_diminished_by_10_to_whole_number (N : ℝ) (x : ℝ) (h1 : 1/5 * N + 4 = x * N - 10) (h2 : N = 280) :
  x = 1 / 4 :=
by
  rw [h2] at h1
  sorry

end ratio_of_part_diminished_by_10_to_whole_number_l1111_111129


namespace first_problem_solution_set_second_problem_a_range_l1111_111194

-- Define the function f(x) = |2x - a| + |x - 1|
def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + abs (x - 1)

-- First problem: When a = 3, the solution set of the inequality f(x) ≥ 2
theorem first_problem_solution_set (x : ℝ) : (f x 3 ≥ 2) ↔ (x ≤ 2 / 3 ∨ x ≥ 2) :=
by sorry

-- Second problem: If f(x) ≥ 5 - x for ∀ x ∈ ℝ, find the range of the real number a
theorem second_problem_a_range (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 5 - x) ↔ (6 ≤ a) :=
by sorry

end first_problem_solution_set_second_problem_a_range_l1111_111194


namespace count_non_squares_or_cubes_l1111_111164

theorem count_non_squares_or_cubes (n : ℕ) (h₀ : 1 ≤ n ∧ n ≤ 200) : 
  ∃ c, c = 182 ∧ 
  (∃ k, k^2 = n ∨ ∃ m, m^3 = n) → false :=
by
  sorry

end count_non_squares_or_cubes_l1111_111164


namespace resistance_per_band_is_10_l1111_111146

noncomputable def resistance_per_band := 10
def total_squat_weight := 30
def dumbbell_weight := 10
def number_of_bands := 2

theorem resistance_per_band_is_10 :
  (total_squat_weight - dumbbell_weight) / number_of_bands = resistance_per_band := 
by
  sorry

end resistance_per_band_is_10_l1111_111146


namespace remaining_regular_toenails_l1111_111174

def big_toenail_space := 2
def total_capacity := 100
def big_toenails_count := 20
def regular_toenails_count := 40

theorem remaining_regular_toenails : 
  total_capacity - (big_toenails_count * big_toenail_space + regular_toenails_count) = 20 := by
  sorry

end remaining_regular_toenails_l1111_111174


namespace distance_between_trees_l1111_111176

theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) (h1 : yard_length = 255) (h2 : num_trees = 18) : yard_length / (num_trees - 1) = 15 := by
  sorry

end distance_between_trees_l1111_111176
