import Mathlib

namespace conference_min_duration_l2051_205149

theorem conference_min_duration : Nat.gcd 9 11 = 1 ∧ Nat.gcd 9 12 = 3 ∧ Nat.gcd 11 12 = 1 ∧ Nat.lcm 9 (Nat.lcm 11 12) = 396 := by
  sorry

end conference_min_duration_l2051_205149


namespace parabola_directrix_l2051_205184

theorem parabola_directrix (x : ℝ) : ∃ d : ℝ, (∀ x : ℝ, 4 * x ^ 2 - 3 = d) → d = -49 / 16 :=
by
  sorry

end parabola_directrix_l2051_205184


namespace four_digit_integer_existence_l2051_205191

theorem four_digit_integer_existence :
  ∃ (a b c d : ℕ), 
    (1000 * a + 100 * b + 10 * c + d = 4522) ∧
    (a + b + c + d = 16) ∧
    (b + c = 10) ∧
    (a - d = 3) ∧
    (1000 * a + 100 * b + 10 * c + d) % 9 = 0 :=
by sorry

end four_digit_integer_existence_l2051_205191


namespace trapezoid_perimeter_l2051_205198

theorem trapezoid_perimeter (a b : ℝ) (h : ∃ c : ℝ, a * b = c^2) :
  ∃ K : ℝ, K = 2 * (a + b + Real.sqrt (a * b)) :=
by
  sorry

end trapezoid_perimeter_l2051_205198


namespace average_weight_of_arun_l2051_205121

variable (weight : ℝ)

def arun_constraint := 61 < weight ∧ weight < 72
def brother_constraint := 60 < weight ∧ weight < 70
def mother_constraint := weight ≤ 64
def father_constraint := 62 < weight ∧ weight < 73
def sister_constraint := 59 < weight ∧ weight < 68

theorem average_weight_of_arun : 
  (∃ w : ℝ, arun_constraint w ∧ brother_constraint w ∧ mother_constraint w ∧ father_constraint w ∧ sister_constraint w) →
  (63.5 = (63 + 64) / 2) := 
by
  sorry

end average_weight_of_arun_l2051_205121


namespace jade_pieces_left_l2051_205188

-- Define the initial number of pieces Jade has
def initial_pieces : Nat := 100

-- Define the number of pieces per level
def pieces_per_level : Nat := 7

-- Define the number of levels in the tower
def levels : Nat := 11

-- Define the resulting number of pieces Jade has left after building the tower
def pieces_left : Nat := initial_pieces - (pieces_per_level * levels)

-- The theorem stating that after building the tower, Jade has 23 pieces left
theorem jade_pieces_left : pieces_left = 23 := by
  -- Proof omitted
  sorry

end jade_pieces_left_l2051_205188


namespace remainder_3_pow_100_plus_5_mod_8_l2051_205197

theorem remainder_3_pow_100_plus_5_mod_8 : (3^100 + 5) % 8 = 6 := by
  sorry

end remainder_3_pow_100_plus_5_mod_8_l2051_205197


namespace seq_solution_l2051_205175

-- Definitions: Define the sequence {a_n} according to the given conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n ≥ 2, a n - 2 * a (n - 1) = n ^ 2 - 3

-- Main statement: Prove that for all n, the sequence satisfies the derived formula
theorem seq_solution (a : ℕ → ℤ) (h : seq a) : ∀ n, a n = 2 ^ (n + 2) - n ^ 2 - 4 * n - 3 :=
sorry

end seq_solution_l2051_205175


namespace fractions_not_equal_to_seven_over_five_l2051_205141

theorem fractions_not_equal_to_seven_over_five :
  (7 / 5 ≠ 1 + (4 / 20)) ∧ (7 / 5 ≠ 1 + (3 / 15)) ∧ (7 / 5 ≠ 1 + (2 / 6)) :=
by
  sorry

end fractions_not_equal_to_seven_over_five_l2051_205141


namespace sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1_l2051_205161

theorem sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1 (t : ℝ) : 
  Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) :=
sorry

end sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1_l2051_205161


namespace least_three_digit_multiple_of_3_4_9_is_108_l2051_205153

theorem least_three_digit_multiple_of_3_4_9_is_108 :
  ∃ (n : ℕ), (100 ≤ n) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 9 = 0) ∧ (n = 108) :=
by
  sorry

end least_three_digit_multiple_of_3_4_9_is_108_l2051_205153


namespace range_of_a_l2051_205108

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l2051_205108


namespace complex_third_quadrant_l2051_205112

-- Define the imaginary unit i.
def i : ℂ := Complex.I 

-- Define the complex number z = i * (1 + i).
def z : ℂ := i * (1 + i)

-- Prove that z lies in the third quadrant.
theorem complex_third_quadrant : z.re < 0 ∧ z.im < 0 := 
by
  sorry

end complex_third_quadrant_l2051_205112


namespace period_of_cos_3x_l2051_205182

theorem period_of_cos_3x :
  ∃ T : ℝ, (∀ x : ℝ, (Real.cos (3 * (x + T))) = Real.cos (3 * x)) ∧ (T = (2 * Real.pi) / 3) :=
sorry

end period_of_cos_3x_l2051_205182


namespace election_winner_votes_difference_l2051_205178

theorem election_winner_votes_difference :
  ∃ W S T F, F = 199 ∧ W = S + 53 ∧ W = T + 79 ∧ W + S + T + F = 979 ∧ (W - F = 105) :=
by
  sorry

end election_winner_votes_difference_l2051_205178


namespace max_value_E_X_E_Y_l2051_205110

open MeasureTheory

-- Defining the random variables and their ranges
variables {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
variable (X : Ω → ℝ) (Y : Ω → ℝ)

-- Condition: 2 ≤ X ≤ 3
def condition1 : Prop := ∀ ω, 2 ≤ X ω ∧ X ω ≤ 3

-- Condition: XY = 1
def condition2 : Prop := ∀ ω, X ω * Y ω = 1

-- The theorem statement
theorem max_value_E_X_E_Y (h1 : condition1 X) (h2 : condition2 X Y) : 
  ∃ E_X E_Y, (E_X = ∫ ω, X ω ∂μ) ∧ (E_Y = ∫ ω, Y ω ∂μ) ∧ (E_X * E_Y = 25 / 24) := 
sorry

end max_value_E_X_E_Y_l2051_205110


namespace intersection_of_A_and_B_l2051_205144

open Set

variable {α : Type} [PartialOrder α]

noncomputable def A := { x : ℝ | -1 < x ∧ x < 1 }
noncomputable def B := { x : ℝ | 0 < x }

theorem intersection_of_A_and_B :
  A ∩ B = { x : ℝ | 0 < x ∧ x < 1 } :=
by sorry

end intersection_of_A_and_B_l2051_205144


namespace no_integer_solutions_l2051_205154

theorem no_integer_solutions :
   ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2 * x^2 * y^2 + 2 * y^2 * z^2 + 2 * z^2 * x^2 + 24 :=
by
  sorry

end no_integer_solutions_l2051_205154


namespace cos_2pi_minus_alpha_tan_alpha_minus_7pi_l2051_205118

open Real

variables (α : ℝ)
variables (h1 : sin (π + α) = -1 / 3) (h2 : π / 2 < α ∧ α < π)

-- Statement for the problem (Ⅰ)
theorem cos_2pi_minus_alpha :
  cos (2 * π - α) = -2 * sqrt 2 / 3 :=
sorry

-- Statement for the problem (Ⅱ)
theorem tan_alpha_minus_7pi :
  tan (α - 7 * π) = -sqrt 2 / 4 :=
sorry

end cos_2pi_minus_alpha_tan_alpha_minus_7pi_l2051_205118


namespace palm_trees_total_l2051_205105

theorem palm_trees_total
  (forest_palm_trees : ℕ := 5000)
  (desert_palm_trees : ℕ := forest_palm_trees - (3 * forest_palm_trees / 5)) :
  desert_palm_trees + forest_palm_trees = 7000 :=
by
  sorry

end palm_trees_total_l2051_205105


namespace cone_base_circumference_l2051_205152

theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) (C : ℝ) 
  (hV : V = 18 * Real.pi)
  (hh : h = 6) 
  (hV_cone : V = (1/3) * Real.pi * r^2 * h) :
  C = 2 * Real.pi * r → C = 6 * Real.pi :=
by 
  -- We assume as conditions are only mentioned
  sorry

end cone_base_circumference_l2051_205152


namespace ramesh_paid_price_l2051_205192

-- Define the variables based on the conditions
variable (labelledPrice transportCost installationCost sellingPrice paidPrice : ℝ)

-- Define the specific values given in the problem
def discount : ℝ := 0.20 
def profitRate : ℝ := 0.10 
def actualSellingPrice : ℝ := 24475
def transportAmount : ℝ := 125
def installationAmount : ℝ := 250

-- Define the conditions given in the problem as Lean definitions
def selling_price_no_discount (P : ℝ) : ℝ := (1 + profitRate) * P
def discounted_price (P : ℝ) : ℝ := P * (1 - discount)
def total_cost (P : ℝ) : ℝ :=  discounted_price P + transportAmount + installationAmount

-- The problem is to prove that the price Ramesh paid for the refrigerator is Rs. 18175
theorem ramesh_paid_price : 
  ∀ (labelledPrice : ℝ), 
  selling_price_no_discount labelledPrice = actualSellingPrice → 
  paidPrice = total_cost labelledPrice → 
  paidPrice = 18175 := 
by
  intros labelledPrice h1 h2 
  sorry

end ramesh_paid_price_l2051_205192


namespace original_quantity_of_ghee_l2051_205106

theorem original_quantity_of_ghee (Q : ℝ) (h1 : 0.6 * Q = 9) (h2 : 0.4 * Q = 6) (h3 : 0.4 * Q = 0.2 * (Q + 10)) : Q = 10 :=
by sorry

end original_quantity_of_ghee_l2051_205106


namespace number_of_terms_in_arithmetic_sequence_is_20_l2051_205116

theorem number_of_terms_in_arithmetic_sequence_is_20
  (a : ℕ → ℤ)
  (common_difference : ℤ)
  (h1 : common_difference = 2)
  (even_num_terms : ℕ)
  (h2 : ∃ k, even_num_terms = 2 * k)
  (sum_odd_terms sum_even_terms : ℤ)
  (h3 : sum_odd_terms = 15)
  (h4 : sum_even_terms = 35)
  (h5 : ∀ n, a n = a 0 + n * common_difference) :
  even_num_terms = 20 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_is_20_l2051_205116


namespace length_of_bridge_l2051_205168

theorem length_of_bridge (L_train : ℕ) (v_km_hr : ℕ) (t : ℕ) 
  (h_L_train : L_train = 150)
  (h_v_km_hr : v_km_hr = 45)
  (h_t : t = 30) : 
  ∃ L_bridge : ℕ, L_bridge = 225 :=
by 
  sorry

end length_of_bridge_l2051_205168


namespace directrix_of_parabola_l2051_205172

theorem directrix_of_parabola :
  (∀ x y : ℝ, x^2 = 4 * y → ∃ y₀ : ℝ, y₀ = -1 ∧ ∀ y' : ℝ, y' = y₀) :=
by
  sorry

end directrix_of_parabola_l2051_205172


namespace solve_pair_N_n_l2051_205129

def is_solution_pair (N n : ℕ) : Prop :=
  N ^ 2 = 1 + n * (N + n)

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

theorem solve_pair_N_n (N n : ℕ) (i : ℕ) :
  is_solution_pair N n ↔ N = fibonacci (i + 1) ∧ n = fibonacci i := sorry

end solve_pair_N_n_l2051_205129


namespace rotation_90_ccw_l2051_205100

-- Define the complex number before the rotation
def initial_complex : ℂ := -4 - 2 * Complex.I

-- Define the resulting complex number after a 90-degree counter-clockwise    rotation
def result_complex : ℂ := 2 - 4 * Complex.I

-- State the theorem to be proved
theorem rotation_90_ccw (z : ℂ) (h : z = initial_complex) :
  Complex.I * z = result_complex :=
by sorry

end rotation_90_ccw_l2051_205100


namespace u2008_is_5898_l2051_205113

-- Define the sequence as given in the problem.
def u (n : ℕ) : ℕ := sorry  -- The nth term of the sequence defined in the problem.

-- The main theorem stating u_{2008} = 5898.
theorem u2008_is_5898 : u 2008 = 5898 := sorry

end u2008_is_5898_l2051_205113


namespace chord_length_proof_tangent_lines_through_M_l2051_205122

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

noncomputable def line_l (x y : ℝ) : Prop := 2*x - y + 4 = 0

noncomputable def point_M : (ℝ × ℝ) := (3, 1)

noncomputable def chord_length : ℝ := 4 * Real.sqrt (5) / 5

noncomputable def tangent_line_1 (x y : ℝ) : Prop := 3*x - 4*y - 5 = 0
noncomputable def tangent_line_2 (x : ℝ) : Prop := x = 3

theorem chord_length_proof :
  ∀ x y : ℝ, circle_C x y → line_l x y → chord_length = 4 * Real.sqrt (5) / 5 :=
by sorry

theorem tangent_lines_through_M :
  ∀ x y : ℝ, circle_C x y → (tangent_line_1 x y ∨ tangent_line_2 x) :=
by sorry

end chord_length_proof_tangent_lines_through_M_l2051_205122


namespace quadratic_roots_difference_l2051_205131

theorem quadratic_roots_difference (a b : ℝ) :
  (5 * a^2 - 30 * a + 45 = 0) ∧ (5 * b^2 - 30 * b + 45 = 0) → (a - b)^2 = 0 :=
by
  sorry

end quadratic_roots_difference_l2051_205131


namespace movement_of_hands_of_clock_involves_rotation_l2051_205187

theorem movement_of_hands_of_clock_involves_rotation (A B C D : Prop) :
  (A ↔ (∃ p : ℝ, ∃ θ : ℝ, p ≠ θ)) → -- A condition: exists a fixed point and rotation around it
  (B ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- B condition: does not rotate around a fixed point
  (C ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- C condition: does not rotate around a fixed point
  (D ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- D condition: does not rotate around a fixed point
  A :=
by
  intros hA hB hC hD
  sorry

end movement_of_hands_of_clock_involves_rotation_l2051_205187


namespace Rebecca_group_count_l2051_205174

def groupEggs (total_eggs number_of_eggs_per_group total_groups : Nat) : Prop :=
  total_groups = total_eggs / number_of_eggs_per_group

theorem Rebecca_group_count :
  groupEggs 8 2 4 :=
by
  sorry

end Rebecca_group_count_l2051_205174


namespace hcf_of_two_numbers_l2051_205156

theorem hcf_of_two_numbers 
  (x y : ℕ) 
  (h1 : x + y = 45)
  (h2 : Nat.lcm x y = 120)
  (h3 : (1/x : ℚ) + (1/y : ℚ) = 11/120) : 
  Nat.gcd x y = 1 := 
sorry

end hcf_of_two_numbers_l2051_205156


namespace least_five_digit_perfect_square_and_cube_l2051_205199

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l2051_205199


namespace chord_ratio_l2051_205127

theorem chord_ratio (A B C D P : Type) (AP BP CP DP : ℝ)
  (h1 : AP = 4) (h2 : CP = 9)
  (h3 : AP * BP = CP * DP) : BP / DP = 9 / 4 := 
by 
  sorry

end chord_ratio_l2051_205127


namespace objects_meeting_time_l2051_205142

theorem objects_meeting_time 
  (initial_velocity : ℝ) (g : ℝ) (t_delay : ℕ) (t_meet : ℝ) 
  (hv : initial_velocity = 120) 
  (hg : g = 9.8) 
  (ht : t_delay = 5)
  : t_meet = 14.74 :=
sorry

end objects_meeting_time_l2051_205142


namespace circle_center_coordinates_l2051_205138

theorem circle_center_coordinates :
  ∃ c : ℝ × ℝ, (c = (1, -2)) ∧ 
  (∀ x y : ℝ, (x^2 + y^2 - 2*x + 4*y - 4 = 0 ↔ (x - 1)^2 + (y + 2)^2 = 9)) :=
by
  sorry

end circle_center_coordinates_l2051_205138


namespace three_digit_number_div_by_11_l2051_205179

theorem three_digit_number_div_by_11 (x y z n : ℕ) 
  (hx : 0 < x ∧ x < 10) 
  (hy : 0 ≤ y ∧ y < 10) 
  (hz : 0 ≤ z ∧ z < 10) 
  (hn : n = 100 * x + 10 * y + z) 
  (hq : (n / 11) = x + y + z) : 
  n = 198 :=
by
  sorry

end three_digit_number_div_by_11_l2051_205179


namespace find_x_average_is_3_l2051_205132

theorem find_x_average_is_3 (x : ℝ) (h : (2 + 4 + 1 + 3 + x) / 5 = 3) : x = 5 :=
sorry

end find_x_average_is_3_l2051_205132


namespace ferris_wheel_rides_l2051_205151

theorem ferris_wheel_rides :
  let people_per_20_minutes := 70
  let operation_duration_hours := 6
  let minutes_per_hour := 60
  let operation_duration_minutes := operation_duration_hours * minutes_per_hour
  let times_per_hour := minutes_per_hour / 20
  let total_people_per_hour := times_per_hour * people_per_20_minutes
  let total_people := total_people_per_hour * operation_duration_hours
  total_people = 1260 :=
by
  let people_per_20_minutes := 70
  let operation_duration_hours := 6
  let minutes_per_hour := 60
  let operation_duration_minutes := operation_duration_hours * minutes_per_hour
  let times_per_hour := minutes_per_hour / 20
  let total_people_per_hour := times_per_hour * people_per_20_minutes
  let total_people := total_people_per_hour * operation_duration_hours
  have : total_people = 1260 := by sorry
  exact this

end ferris_wheel_rides_l2051_205151


namespace vector_parallel_l2051_205115

variables {t : ℝ}

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, t)

theorem vector_parallel (h : (1 : ℝ) / (3 : ℝ) = (3 : ℝ) / t) : t = 9 :=
by 
  sorry

end vector_parallel_l2051_205115


namespace function_form_l2051_205189

theorem function_form (f : ℕ → ℕ) (H : ∀ (x y z : ℕ), x ≠ y → y ≠ z → z ≠ x → (∃ k : ℕ, x + y + z = k^2 ↔ ∃ m : ℕ, f x + f y + f z = m^2)) : ∃ k : ℕ, ∀ n : ℕ, f n = k^2 * n :=
by
  sorry

end function_form_l2051_205189


namespace min_distance_PS_l2051_205195

-- Definitions of the distances given in the problem
def PQ : ℝ := 12
def QR : ℝ := 7
def RS : ℝ := 5

-- Hypotheses for the problem
axiom h1 : PQ = 12
axiom h2 : QR = 7
axiom h3 : RS = 5

-- The goal is to prove that the minimum distance between P and S is 0.
theorem min_distance_PS : ∃ PS : ℝ, PS = 0 :=
by
  -- The proof is omitted
  sorry

end min_distance_PS_l2051_205195


namespace students_per_bus_correct_l2051_205181

def total_students : ℝ := 28
def number_of_buses : ℝ := 2.0
def students_per_bus : ℝ := 14

theorem students_per_bus_correct :
  total_students / number_of_buses = students_per_bus := 
by
  -- Proof should go here
  sorry

end students_per_bus_correct_l2051_205181


namespace deformable_to_triangle_l2051_205148

-- We define a planar polygon with n rods connected by hinges
structure PlanarPolygon (n : ℕ) :=
  (rods : Fin n → ℝ)
  (connections : Fin n → Fin n → Prop)

-- Define the conditions for the rods being rigid and connections (hinges)
def rigid_rod (n : ℕ) : PlanarPolygon n → Prop := λ poly => 
  ∀ i j, poly.connections i j → poly.rods i = poly.rods j

-- Defining the theorem for deformation into a triangle
theorem deformable_to_triangle (n : ℕ) (p : PlanarPolygon n) : 
  (n > 4) ↔ ∃ q : PlanarPolygon 3, true :=
by
  sorry

end deformable_to_triangle_l2051_205148


namespace find_n_l2051_205135

theorem find_n (n : ℕ) (M : ℕ) (A : ℕ) 
  (hM : M = n - 11) 
  (hA : A = n - 2) 
  (hM_ge_one : M ≥ 1) 
  (hA_ge_one : A ≥ 1) 
  (hM_plus_A_lt_n : M + A < n) : 
  n = 12 := 
by 
  sorry

end find_n_l2051_205135


namespace tickets_sold_l2051_205120

theorem tickets_sold (student_tickets non_student_tickets student_ticket_price non_student_ticket_price total_revenue : ℕ)
  (h1 : student_ticket_price = 5)
  (h2 : non_student_ticket_price = 8)
  (h3 : total_revenue = 930)
  (h4 : student_tickets = 90)
  (h5 : non_student_tickets = 60) :
  student_tickets + non_student_tickets = 150 := 
by 
  sorry

end tickets_sold_l2051_205120


namespace solution_set_inequality_k_l2051_205130

theorem solution_set_inequality_k (k : ℚ) :
  (∀ x : ℚ, 3 * x - (2 * k - 3) < 4 * x + 3 * k + 6 ↔ x > 1) → k = -4/5 :=
by
  sorry

end solution_set_inequality_k_l2051_205130


namespace vertical_coordinate_intersection_l2051_205157

def original_function (x : ℝ) := x^2 + 2 * x + 1

def shifted_function (x : ℝ) := (x + 3)^2 + 3

theorem vertical_coordinate_intersection :
  shifted_function 0 = 12 :=
by
  sorry

end vertical_coordinate_intersection_l2051_205157


namespace shaded_triangle_area_l2051_205128

theorem shaded_triangle_area (b h : ℝ) (hb : b = 2) (hh : h = 3) : 
  (1 / 2 * b * h) = 3 := 
by
  rw [hb, hh]
  norm_num

end shaded_triangle_area_l2051_205128


namespace supplement_of_angle_with_given_complement_l2051_205124

theorem supplement_of_angle_with_given_complement (θ : ℝ) (h : 90 - θ = 50) : 180 - θ = 140 :=
by sorry

end supplement_of_angle_with_given_complement_l2051_205124


namespace paul_books_left_l2051_205136
-- Add the necessary imports

-- Define the initial conditions
def initial_books : ℕ := 115
def books_sold : ℕ := 78

-- Statement of the problem as a theorem
theorem paul_books_left : (initial_books - books_sold) = 37 := by
  -- Proof omitted
  sorry

end paul_books_left_l2051_205136


namespace equilateral_triangle_of_equal_heights_and_inradius_l2051_205139

theorem equilateral_triangle_of_equal_heights_and_inradius 
  {a b c h1 h2 h3 r : ℝ} (h1_eq : h1 = 2 * r * (a * b * c) / a) 
  (h2_eq : h2 = 2 * r * (a * b * c) / b) 
  (h3_eq : h3 = 2 * r * (a * b * c) / c) 
  (sum_heights_eq : h1 + h2 + h3 = 9 * r) : a = b ∧ b = c ∧ c = a :=
by
  sorry

end equilateral_triangle_of_equal_heights_and_inradius_l2051_205139


namespace symmetric_line_eq_l2051_205193

theorem symmetric_line_eq (x y: ℝ) :
    (∃ (a b: ℝ), 3 * a - b + 2 = 0 ∧ a = 2 - x ∧ b = 2 - y) → 3 * x - y - 6 = 0 :=
by
    intro h
    sorry

end symmetric_line_eq_l2051_205193


namespace number_of_days_l2051_205103

theorem number_of_days (d : ℝ) (h : 2 * d = 1.5 * d + 3) : d = 6 :=
by
  sorry

end number_of_days_l2051_205103


namespace new_mean_l2051_205119

-- Define the conditions
def mean_of_numbers (n : ℕ) (mean : ℝ) : ℝ := n * mean
def added_to_each (n : ℕ) (addend : ℝ) : ℝ := n * addend

-- The proof problem
theorem new_mean (n : ℕ) (mean addend : ℝ) (h1 : mean_of_numbers n mean = 600) (h2 : added_to_each n addend = 150) (h3 : n = 15) (h4 : mean = 40) (h5 : addend = 10) :
  (mean_of_numbers n mean + added_to_each n addend) / n = 50 :=
by
  sorry

end new_mean_l2051_205119


namespace smallest_y_value_l2051_205109

theorem smallest_y_value (y : ℚ) (h : y / 7 + 2 / (7 * y) = 1 / 3) : y = 2 / 3 :=
sorry

end smallest_y_value_l2051_205109


namespace smallest_integer_greater_than_20_l2051_205177

noncomputable def smallest_integer_greater_than_A : ℕ :=
  let a (n : ℕ) := 4 * n - 3
  let A := Real.sqrt (a 1580) - 1 / 4
  Nat.ceil A

theorem smallest_integer_greater_than_20 :
  smallest_integer_greater_than_A = 20 :=
sorry

end smallest_integer_greater_than_20_l2051_205177


namespace lizzy_loan_amount_l2051_205186

noncomputable def interest_rate : ℝ := 0.20
noncomputable def initial_amount : ℝ := 30
noncomputable def final_amount : ℝ := 33

theorem lizzy_loan_amount (X : ℝ) (h : initial_amount + (1 + interest_rate) * X = final_amount) : X = 2.5 := 
by
  sorry

end lizzy_loan_amount_l2051_205186


namespace age_of_third_boy_l2051_205111

theorem age_of_third_boy (a b c : ℕ) (h1 : a = 9) (h2 : b = 9) (h_sum : a + b + c = 29) : c = 11 :=
by
  sorry

end age_of_third_boy_l2051_205111


namespace Math_Proof_Problem_l2051_205164

noncomputable def problem : ℝ := (1005^3) / (1003 * 1004) - (1003^3) / (1004 * 1005)

theorem Math_Proof_Problem : ⌊ problem ⌋ = 8 :=
by
  sorry

end Math_Proof_Problem_l2051_205164


namespace polynomial_in_y_l2051_205165

theorem polynomial_in_y {x y : ℝ} (h₁ : x^3 - 6 * x^2 + 11 * x - 6 = 0) (h₂ : y = x + 1/x) :
  x^2 * (y^2 + y - 6) = 0 :=
sorry

end polynomial_in_y_l2051_205165


namespace quadratic_function_points_l2051_205169

theorem quadratic_function_points:
  (∀ x y, (y = x^2 + x - 1) → ((x = -2 → y = 1) ∧ (x = 0 → y = -1) ∧ (x = 2 → y = 5))) →
  (-1 < 1 ∧ 1 < 5) :=
by
  intro h
  have h1 := h (-2) 1 (by ring)
  have h2 := h 0 (-1) (by ring)
  have h3 := h 2 5 (by ring)
  exact And.intro (by linarith) (by linarith)

end quadratic_function_points_l2051_205169


namespace sum_of_digits_l2051_205183

theorem sum_of_digits (A B C D : ℕ) (H1: A < B) (H2: B < C) (H3: C < D)
  (H4: A > 0) (H5: B > 0) (H6: C > 0) (H7: D > 0)
  (H8: 1000 * A + 100 * B + 10 * C + D + 1000 * D + 100 * C + 10 * B + A = 11990) : 
  (A, B, C, D) = (1, 9, 9, 9) :=
sorry

end sum_of_digits_l2051_205183


namespace age_difference_is_13_l2051_205196

variables (A B C X : ℕ)
variables (total_age_A_B total_age_B_C : ℕ)

-- Conditions
def condition1 : Prop := total_age_A_B = total_age_B_C + X
def condition2 : Prop := C = A - 13

-- Theorem statement
theorem age_difference_is_13 (h1: condition1 total_age_A_B total_age_B_C X)
                             (h2: condition2 A C) :
  X = 13 :=
sorry

end age_difference_is_13_l2051_205196


namespace number_of_penny_piles_l2051_205126

theorem number_of_penny_piles
    (piles_of_quarters : ℕ := 4) 
    (piles_of_dimes : ℕ := 6)
    (piles_of_nickels : ℕ := 9)
    (total_value_in_dollars : ℝ := 21)
    (coins_per_pile : ℕ := 10)
    (quarter_value : ℝ := 0.25)
    (dime_value : ℝ := 0.10)
    (nickel_value : ℝ := 0.05)
    (penny_value : ℝ := 0.01) :
    (total_value_in_dollars - ((piles_of_quarters * coins_per_pile * quarter_value) +
                               (piles_of_dimes * coins_per_pile * dime_value) +
                               (piles_of_nickels * coins_per_pile * nickel_value))) /
                               (coins_per_pile * penny_value) = 5 := 
by
  sorry

end number_of_penny_piles_l2051_205126


namespace inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l2051_205140

theorem inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
sorry

end inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l2051_205140


namespace regular_dodecahedron_has_12_faces_l2051_205180

-- Define a structure to represent a regular dodecahedron
structure RegularDodecahedron where

-- The main theorem to state that a regular dodecahedron has 12 faces
theorem regular_dodecahedron_has_12_faces (D : RegularDodecahedron) : ∃ faces : ℕ, faces = 12 := by
  sorry

end regular_dodecahedron_has_12_faces_l2051_205180


namespace gcd_55555555_111111111_l2051_205159

/-- Let \( m = 55555555 \) and \( n = 111111111 \).
We want to prove that the greatest common divisor (gcd) of \( m \) and \( n \) is 1. -/
theorem gcd_55555555_111111111 :
  let m := 55555555
  let n := 111111111
  Nat.gcd m n = 1 :=
by
  sorry

end gcd_55555555_111111111_l2051_205159


namespace lines_intersect_ellipse_at_2_or_4_points_l2051_205171

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 9 = 1

def line_intersects_ellipse (line : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  ellipse_eq x y ∧ line x y

def number_of_intersections (line1 line2 : ℝ → ℝ → Prop) (n : ℕ) : Prop :=
  ∃ pts : Finset (ℝ × ℝ), (∀ pt ∈ pts, (line_intersects_ellipse line1 pt.1 pt.2 ∨
                                        line_intersects_ellipse line2 pt.1 pt.2)) ∧
                           pts.card = n ∧ 
                           (∀ pt ∈ pts, line1 pt.1 pt.2 ∨ line2 pt.1 pt.2) ∧
                           (∀ (pt1 pt2 : ℝ × ℝ), pt1 ∈ pts → pt2 ∈ pts → pt1 ≠ pt2 → pt1 ≠ pt2)

theorem lines_intersect_ellipse_at_2_or_4_points 
  (line1 line2 : ℝ → ℝ → Prop)
  (h1 : ∃ x1 y1, line1 x1 y1 ∧ ellipse_eq x1 y1)
  (h2 : ∃ x2 y2, line2 x2 y2 ∧ ellipse_eq x2 y2)
  (h3: ¬ ∀ x y, line1 x y ∧ ellipse_eq x y → false)
  (h4: ¬ ∀ x y, line2 x y ∧ ellipse_eq x y → false) :
  ∃ n : ℕ, (n = 2 ∨ n = 4) ∧ number_of_intersections line1 line2 n := sorry

end lines_intersect_ellipse_at_2_or_4_points_l2051_205171


namespace oranges_to_pears_l2051_205162

-- Define the equivalence relation between oranges and pears
def equivalent_weight (orange pear : ℕ) : Prop := 4 * pear = 3 * orange

-- Given:
-- 1. 4 oranges weigh the same as 3 pears
-- 2. Jimmy has 36 oranges
-- Prove that 27 pears are required to balance the weight of 36 oranges
theorem oranges_to_pears (orange pear : ℕ) (h : equivalent_weight 1 1) :
  (4 * pear = 3 * orange) → equivalent_weight 36 27 :=
by
  sorry

end oranges_to_pears_l2051_205162


namespace original_price_of_article_l2051_205150

theorem original_price_of_article
  (P S : ℝ) 
  (h1 : S = 1.4 * P) 
  (h2 : S - P = 560) 
  : P = 1400 :=
by
  sorry

end original_price_of_article_l2051_205150


namespace workers_combined_time_l2051_205104

theorem workers_combined_time (g_rate a_rate c_rate : ℝ)
  (hg : g_rate = 1 / 70)
  (ha : a_rate = 1 / 30)
  (hc : c_rate = 1 / 42) :
  1 / (g_rate + a_rate + c_rate) = 14 :=
by
  sorry

end workers_combined_time_l2051_205104


namespace max_possible_n_l2051_205194

theorem max_possible_n :
  ∃ (n : ℕ), (n < 150) ∧ (∃ (k l : ℤ), n = 9 * k - 1 ∧ n = 6 * l - 5 ∧ n = 125) :=
by 
  sorry

end max_possible_n_l2051_205194


namespace jason_cutting_hours_l2051_205146

-- Definitions derived from conditions
def time_to_cut_one_lawn : ℕ := 30  -- minutes
def lawns_per_day := 8 -- number of lawns Jason cuts each day
def days := 2 -- number of days (Saturday and Sunday)
def minutes_in_an_hour := 60 -- conversion factor from minutes to hours

-- The proof problem
theorem jason_cutting_hours : 
  (time_to_cut_one_lawn * lawns_per_day * days) / minutes_in_an_hour = 8 := sorry

end jason_cutting_hours_l2051_205146


namespace primes_divisible_by_3_percentage_is_12_5_l2051_205147

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end primes_divisible_by_3_percentage_is_12_5_l2051_205147


namespace find_second_group_of_men_l2051_205166

noncomputable def work_rate_of_man := ℝ
noncomputable def work_rate_of_woman := ℝ

variables (m w : ℝ)

-- Condition 1: 3 men and 8 women complete the task in the same time as x men and 2 women.
axiom condition1 (x : ℝ) : 3 * m + 8 * w = x * m + 2 * w

-- Condition 2: 2 men and 3 women complete half the task in the same time as 3 men and 8 women completing the whole task.
axiom condition2 : 2 * m + 3 * w = 0.5 * (3 * m + 8 * w)

theorem find_second_group_of_men (x : ℝ) (m w : ℝ) (h1 : 0.5 * m = w)
  (h2 : 3 * m + 8 * w = x * m + 2 * w) : x = 6 :=
by {
  sorry
}

end find_second_group_of_men_l2051_205166


namespace exponential_ordering_l2051_205158

noncomputable def a := (0.4:ℝ)^(0.3:ℝ)
noncomputable def b := (0.3:ℝ)^(0.4:ℝ)
noncomputable def c := (0.3:ℝ)^(-0.2:ℝ)

theorem exponential_ordering : b < a ∧ a < c := by
  sorry

end exponential_ordering_l2051_205158


namespace similar_pentagon_area_l2051_205176

theorem similar_pentagon_area
  (K1 K2 : ℝ) (L1 L2 : ℝ)
  (h_similar : true)  -- simplifying the similarity condition as true for the purpose of this example
  (h_K1 : K1 = 18)
  (h_K2 : K2 = 24)
  (h_L1 : L1 = 8.4375) :
  L2 = 15 :=
by
  sorry

end similar_pentagon_area_l2051_205176


namespace expr_value_at_neg2_l2051_205137

variable (a b : ℝ)

def expr (x : ℝ) : ℝ := a * x^3 + b * x - 7

theorem expr_value_at_neg2 :
  (expr a b 2 = -19) → (expr a b (-2) = 5) :=
by 
  intro h
  sorry

end expr_value_at_neg2_l2051_205137


namespace shifted_function_expression_l2051_205160

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x + Real.pi / 3)

theorem shifted_function_expression (ω : ℝ) (h : ℝ) (x : ℝ) (h_positive : ω > 0) (h_period : Real.pi = 2 * Real.pi / ω) :
  f ω (x + h) = Real.cos (2 * x) :=
by
  -- We assume h = π/12, ω = 2
  have ω_val : ω = 2 := by sorry
  have h_val : h = Real.pi / 12 := by sorry
  rw [ω_val, h_val]
  sorry

end shifted_function_expression_l2051_205160


namespace consecutive_integers_divisor_l2051_205163

theorem consecutive_integers_divisor {m n : ℕ} (hm : m < n) (a : ℕ) :
  ∃ i j : ℕ, i ≠ j ∧ (a + i) * (a + j) % (m * n) = 0 :=
by
  sorry

end consecutive_integers_divisor_l2051_205163


namespace money_first_day_l2051_205185

-- Define the total mushrooms
def total_mushrooms : ℕ := 65

-- Define the mushrooms picked on the second day
def mushrooms_day2 : ℕ := 12

-- Define the mushrooms picked on the third day
def mushrooms_day3 : ℕ := 2 * mushrooms_day2

-- Define the price per mushroom
def price_per_mushroom : ℕ := 2

-- Prove that the amount of money made on the first day is $58
theorem money_first_day : (total_mushrooms - mushrooms_day2 - mushrooms_day3) * price_per_mushroom = 58 := 
by
  -- Skip the proof
  sorry

end money_first_day_l2051_205185


namespace extremum_range_a_l2051_205190

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - a * x^2 + x

theorem extremum_range_a :
  (∀ x : ℝ, -1 < x ∧ x < 0 → (f a x = 0 → ∃ x0 : ℝ, f a x0 = 0 ∧ -1 < x0 ∧ x0 < 0)) →
  a < -1/5 ∨ a = -1 :=
sorry

end extremum_range_a_l2051_205190


namespace six_star_three_l2051_205133

-- Define the mathematical operation.
def operation (r t : ℝ) : ℝ := sorry

axiom condition_1 (r : ℝ) : operation r 0 = r^2
axiom condition_2 (r t : ℝ) : operation r t = operation t r
axiom condition_3 (r t : ℝ) : operation (r + 1) t = operation r t + 2 * t + 1

-- Prove that 6 * 3 = 75 given the conditions.
theorem six_star_three : operation 6 3 = 75 := by
  sorry

end six_star_three_l2051_205133


namespace molecular_weight_of_9_moles_l2051_205155

theorem molecular_weight_of_9_moles (molecular_weight : ℕ) (moles : ℕ) (h₁ : molecular_weight = 1098) (h₂ : moles = 9) :
  molecular_weight * moles = 9882 :=
by {
  sorry
}

end molecular_weight_of_9_moles_l2051_205155


namespace tangent_line_x_squared_l2051_205101

theorem tangent_line_x_squared (P : ℝ × ℝ) (hP : P = (1, -1)) :
  ∃ (a : ℝ), a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2 ∧
    ((∀ x : ℝ, (2 * (1 + Real.sqrt 2) * x - (3 + 2 * Real.sqrt 2)) = P.2 → 
      P.2 = (2 * (1 + Real.sqrt 2) * P.1 - (3 + 2 * Real.sqrt 2))) ∨
    (∀ x : ℝ, (2 * (1 - Real.sqrt 2) * x - (3 - 2 * Real.sqrt 2)) = P.2 → 
      P.2 = (2 * (1 - Real.sqrt 2) * P.1 - (3 - 2 * Real.sqrt 2)))) := by
  sorry

end tangent_line_x_squared_l2051_205101


namespace geom_series_correct_sum_l2051_205123

-- Define the geometric series sum
noncomputable def geom_series_sum (a r : ℚ) (n : ℕ) :=
  a * (1 - r ^ n) / (1 - r)

-- Given conditions
def a := (1 : ℚ) / 4
def r := (1 : ℚ) / 4
def n := 8

-- Correct answer sum
def correct_sum := (65535 : ℚ) / 196608

-- Proof problem statement
theorem geom_series_correct_sum : geom_series_sum a r n = correct_sum := 
  sorry

end geom_series_correct_sum_l2051_205123


namespace problem1_problem2_l2051_205134

theorem problem1 (n : ℕ) (hn : 0 < n) : (3^(2*n+1) + 2^(n+2)) % 7 = 0 := 
sorry

theorem problem2 (n : ℕ) (hn : 0 < n) : (3^(2*n+2) + 2^(6*n+1)) % 11 = 0 := 
sorry

end problem1_problem2_l2051_205134


namespace sum_of_primes_between_30_and_50_l2051_205107

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- List of prime numbers between 30 and 50
def prime_numbers_between_30_and_50 : List ℕ := [31, 37, 41, 43, 47]

-- Sum of prime numbers between 30 and 50
def sum_prime_numbers_between_30_and_50 : ℕ :=
  prime_numbers_between_30_and_50.sum

-- Theorem: The sum of prime numbers between 30 and 50 is 199
theorem sum_of_primes_between_30_and_50 :
  sum_prime_numbers_between_30_and_50 = 199 := by
    sorry

end sum_of_primes_between_30_and_50_l2051_205107


namespace ratio_of_triangle_to_square_l2051_205173

theorem ratio_of_triangle_to_square (s : ℝ) (hs : 0 < s) :
  let A_square := s^2
  let A_triangle := (1/2) * s * (s/2)
  A_triangle / A_square = 1/4 :=
by
  sorry

end ratio_of_triangle_to_square_l2051_205173


namespace rooms_equation_l2051_205143

theorem rooms_equation (x : ℕ) (h₁ : ∃ n, n = 6 * (x - 1)) (h₂ : ∃ m, m = 5 * x + 4) :
  6 * (x - 1) = 5 * x + 4 :=
sorry

end rooms_equation_l2051_205143


namespace cubical_block_weight_l2051_205117

-- Given conditions
variables (s : ℝ) (volume_ratio : ℝ) (weight2 : ℝ)
variable (h : volume_ratio = 8)
variable (h_weight : weight2 = 40)

-- The problem statement
theorem cubical_block_weight (weight1 : ℝ) :
  volume_ratio * weight1 = weight2 → weight1 = 5 :=
by
  -- Assume volume ratio as 8, weight of the second cube as 40 pounds
  have h1 : volume_ratio = 8 := h
  have h2 : weight2 = 40 := h_weight
  -- sorry is here to indicate we are skipping the proof
  sorry

end cubical_block_weight_l2051_205117


namespace idiom_describes_random_event_l2051_205114

-- Define the idioms as propositions.
def FishingForMoonInWater : Prop := ∀ (x : Type), x -> False
def CastlesInTheAir : Prop := ∀ (y : Type), y -> False
def WaitingByStumpForHare : Prop := ∃ (z : Type), True
def CatchingTurtleInJar : Prop := ∀ (w : Type), w -> False

-- Define the main theorem to state that WaitingByStumpForHare describes a random event.
theorem idiom_describes_random_event : WaitingByStumpForHare :=
  sorry

end idiom_describes_random_event_l2051_205114


namespace find_a2015_l2051_205145

variable (a : ℕ → ℝ)

-- Conditions
axiom h1 : a 1 = 1
axiom h2 : a 2 = 3
axiom h3 : ∀ n : ℕ, n > 0 → a (n + 1) - a n ≤ 2 ^ n
axiom h4 : ∀ n : ℕ, n > 0 → a (n + 2) - a n ≥ 3 * 2 ^ n

-- Theorem stating the solution
theorem find_a2015 : a 2015 = 2 ^ 2015 - 1 :=
by sorry

end find_a2015_l2051_205145


namespace janet_initial_number_l2051_205125

-- Define the conditions using Lean definitions
def janetProcess (x : ℕ) : ℕ :=
  (2 * (x + 7)) - 4

-- The theorem that expresses the statement of the problem: If the final result of the process is 28, then x = 9
theorem janet_initial_number (x : ℕ) (h : janetProcess x = 28) : x = 9 :=
sorry

end janet_initial_number_l2051_205125


namespace find_precy_age_l2051_205170

-- Defining the given conditions as Lean definitions
def alex_current_age : ℕ := 15
def alex_age_in_3_years : ℕ := alex_current_age + 3
def alex_age_a_year_ago : ℕ := alex_current_age - 1
axiom precy_current_age : ℕ
axiom in_3_years : alex_age_in_3_years = 3 * (precy_current_age + 3)
axiom a_year_ago : alex_age_a_year_ago = 7 * (precy_current_age - 1)

-- Stating the equivalent proof problem
theorem find_precy_age : precy_current_age = 3 :=
by
  sorry

end find_precy_age_l2051_205170


namespace value_of_M_in_equation_l2051_205102

theorem value_of_M_in_equation :
  ∀ {M : ℕ}, (32 = 2^5) ∧ (8 = 2^3) → (32^3 * 8^4 = 2^M) → M = 27 :=
by
  intros M h1 h2
  sorry

end value_of_M_in_equation_l2051_205102


namespace total_feet_l2051_205167

theorem total_feet (heads hens : ℕ) (h1 : heads = 46) (h2 : hens = 22) : 
  ∃ feet : ℕ, feet = 140 := 
by 
  sorry

end total_feet_l2051_205167
