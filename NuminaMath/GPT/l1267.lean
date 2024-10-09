import Mathlib

namespace guest_bedroom_area_l1267_126765

theorem guest_bedroom_area 
  (master_bedroom_bath_area : ℝ)
  (kitchen_guest_bath_living_area : ℝ)
  (total_rent : ℝ)
  (rate_per_sqft : ℝ)
  (num_guest_bedrooms : ℕ)
  (area_guest_bedroom : ℝ) :
  master_bedroom_bath_area = 500 →
  kitchen_guest_bath_living_area = 600 →
  total_rent = 3000 →
  rate_per_sqft = 2 →
  num_guest_bedrooms = 2 →
  (total_rent / rate_per_sqft) - (master_bedroom_bath_area + kitchen_guest_bath_living_area) / num_guest_bedrooms = area_guest_bedroom → 
  area_guest_bedroom = 200 := by
  sorry

end guest_bedroom_area_l1267_126765


namespace common_chord_equation_l1267_126786

-- Definitions of the given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 12 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 14*y + 15 = 0

-- Definition of the common chord line
def common_chord_line (x y : ℝ) : Prop := 6*x + 8*y - 3 = 0

-- The theorem to be proved
theorem common_chord_equation :
  (∀ x y, circle1 x y → circle2 x y → common_chord_line x y) :=
by sorry

end common_chord_equation_l1267_126786


namespace find_a_from_expansion_l1267_126754

theorem find_a_from_expansion :
  (∃ a : ℝ, (∃ c : ℝ, (∃ d : ℝ, (∃ e : ℝ, (20 - 30 * a + 6 * a^2 = -16 ∧ (a = 2 ∨ a = 3))))))
:= sorry

end find_a_from_expansion_l1267_126754


namespace ratio_a_over_b_l1267_126735

-- Definitions of conditions
def func (a b x : ℝ) : ℝ := a * x^2 + b
def derivative (a b x : ℝ) : ℝ := 2 * a * x

-- Given conditions
variables (a b : ℝ)
axiom tangent_slope : derivative a b 1 = 2
axiom point_on_graph : func a b 1 = 3

-- Statement to prove
theorem ratio_a_over_b : a / b = 1 / 2 :=
by sorry

end ratio_a_over_b_l1267_126735


namespace compound_interest_doubling_time_l1267_126726

theorem compound_interest_doubling_time :
  ∃ (t : ℕ), (0.15 : ℝ) = 0.15 ∧ ∀ (n : ℕ), (n = 1) →
               (2 : ℝ) < (1 + 0.15) ^ t ∧ t = 5 :=
by
  sorry

end compound_interest_doubling_time_l1267_126726


namespace solve_for_q_l1267_126779

theorem solve_for_q :
  ∀ (k l q : ℚ),
    (3 / 4 = k / 108) →
    (3 / 4 = (l + k) / 126) →
    (3 / 4 = (q - l) / 180) →
    q = 148.5 :=
by
  intros k l q hk hl hq
  sorry

end solve_for_q_l1267_126779


namespace simplify_fraction_l1267_126729

theorem simplify_fraction : (3 / 462 + 17 / 42) = 95 / 231 :=
by 
  sorry

end simplify_fraction_l1267_126729


namespace some_number_value_l1267_126724

theorem some_number_value (a : ℤ) (x1 x2 : ℤ)
  (h1 : x1 + a = 10) (h2 : x2 + a = -10) (h_sum : x1 + x2 = 20) : a = -10 :=
by
  sorry

end some_number_value_l1267_126724


namespace sum_of_coefficients_l1267_126720

theorem sum_of_coefficients (a b c : ℤ) (h : a - b + c = -1) : a + b + c = -1 := sorry

end sum_of_coefficients_l1267_126720


namespace reciprocal_of_neg_three_l1267_126775

theorem reciprocal_of_neg_three : -3 * (-1 / 3) = 1 := 
by
  sorry

end reciprocal_of_neg_three_l1267_126775


namespace total_students_in_school_district_l1267_126791

def CampusA_students : Nat :=
  let students_per_grade : Nat := 100
  let num_grades : Nat := 5
  let special_education : Nat := 30
  (students_per_grade * num_grades) + special_education

def CampusB_students : Nat :=
  let students_per_grade : Nat := 120
  let num_grades : Nat := 5
  students_per_grade * num_grades

def CampusC_students : Nat :=
  let students_per_grade : Nat := 150
  let num_grades : Nat := 2
  let international_program : Nat := 50
  (students_per_grade * num_grades) + international_program

def total_students : Nat :=
  CampusA_students + CampusB_students + CampusC_students

theorem total_students_in_school_district : total_students = 1480 := by
  sorry

end total_students_in_school_district_l1267_126791


namespace k_value_of_polynomial_square_l1267_126715

theorem k_value_of_polynomial_square (k : ℤ) :
  (∃ (f : ℤ → ℤ), ∀ x, f x = x^2 + 6 * x + k^2) → (k = 3 ∨ k = -3) :=
by
  sorry

end k_value_of_polynomial_square_l1267_126715


namespace greatest_possible_remainder_l1267_126723

theorem greatest_possible_remainder (x : ℕ) : ∃ r, r < 11 ∧ x % 11 = r ∧ r = 10 :=
by
  exists 10
  sorry

end greatest_possible_remainder_l1267_126723


namespace part1_part2_l1267_126713

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 5 * Real.log x + a * x^2 - 6 * x
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := 5 / x + 2 * a * x - 6

theorem part1 (a : ℝ) (h_tangent : f_prime 1 a = 0) : a = 1 / 2 :=
by {
  sorry
}

theorem part2 (a : ℝ) (h_a : a = 1/2) :
  (∀ x, 0 < x → x < 1 → f_prime x a > 0) ∧
  (∀ x, 5 < x → f_prime x a > 0) ∧
  (∀ x, 1 < x → x < 5 → f_prime x a < 0) :=
by {
  sorry
}

end part1_part2_l1267_126713


namespace three_same_colored_balls_l1267_126788

theorem three_same_colored_balls (balls : ℕ) (color_count : ℕ) (balls_per_color : ℕ) (h1 : balls = 60) (h2 : color_count = balls / balls_per_color) (h3 : balls_per_color = 6) :
  ∃ n, n = 21 ∧ (∀ picks : ℕ, picks ≥ n → ∃ c, ∃ k ≥ 3, k ≤ balls_per_color ∧ (c < color_count) ∧ (picks / c = k)) :=
sorry

end three_same_colored_balls_l1267_126788


namespace find_decreased_amount_l1267_126785

variables (x y : ℝ)

axiom h1 : 0.20 * x - y = 6
axiom h2 : x = 50.0

theorem find_decreased_amount : y = 4 :=
by
  sorry

end find_decreased_amount_l1267_126785


namespace exists_right_triangle_area_eq_perimeter_l1267_126711

theorem exists_right_triangle_area_eq_perimeter :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a + b + c = (a * b) / 2 ∧ a ≠ b ∧ 
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 12 ∧ b = 5 ∧ c = 13) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10)) :=
by
  sorry

end exists_right_triangle_area_eq_perimeter_l1267_126711


namespace prime_eq_sol_l1267_126750

theorem prime_eq_sol {p q x y z : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
  sorry

end prime_eq_sol_l1267_126750


namespace age_relation_l1267_126759

variable (x y z : ℕ)

theorem age_relation (h1 : x > y) : (z > y) ↔ (∃ w, w > 0 ∧ y + z > 2 * x) :=
sorry

end age_relation_l1267_126759


namespace calc1_calc2_l1267_126712

noncomputable def calculation1 := -4^2

theorem calc1 : calculation1 = -16 := by
  sorry

noncomputable def calculation2 := (-3) - (-6)

theorem calc2 : calculation2 = 3 := by
  sorry

end calc1_calc2_l1267_126712


namespace fraction_to_decimal_l1267_126731

theorem fraction_to_decimal :
  (58 / 200 : ℝ) = 1.16 := by
  sorry

end fraction_to_decimal_l1267_126731


namespace min_calls_required_l1267_126792

-- Define the set of people involved in the communication
inductive Person
| A | B | C | D | E | F

-- Function to calculate the minimum number of calls for everyone to know all pieces of gossip
def minCalls : ℕ :=
  9

-- Theorem stating the minimum number of calls required
theorem min_calls_required : minCalls = 9 := by
  sorry

end min_calls_required_l1267_126792


namespace sum_powers_of_i_l1267_126764

-- Define the conditions
def i : ℂ := Complex.I -- Complex.I is the imaginary unit in ℂ (ℂ is the set of complex numbers)

-- The theorem statement
theorem sum_powers_of_i : (i + i^2 + i^3 + i^4) * 150 + 1 + i + i^2 + i^3 = 0 := by
  sorry

end sum_powers_of_i_l1267_126764


namespace time_to_fill_cistern_l1267_126751

def pipe_p_rate := (1: ℚ) / 10
def pipe_q_rate := (1: ℚ) / 15
def pipe_r_rate := - (1: ℚ) / 30
def combined_rate_p_q := pipe_p_rate + pipe_q_rate
def combined_rate_q_r := pipe_q_rate + pipe_r_rate
def initial_fill := 2 * combined_rate_p_q
def remaining_fill := 1 - initial_fill
def remaining_time := remaining_fill / combined_rate_q_r

theorem time_to_fill_cistern :
  remaining_time = 20 := by sorry

end time_to_fill_cistern_l1267_126751


namespace solution_set_of_quadratic_inequality_l1267_126793

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x ≥ 1} :=
by
  sorry

end solution_set_of_quadratic_inequality_l1267_126793


namespace extra_bananas_each_child_gets_l1267_126783

theorem extra_bananas_each_child_gets
  (total_children : ℕ)
  (bananas_per_child : ℕ)
  (absent_children : ℕ)
  (present_children : ℕ)
  (total_bananas : ℕ)
  (bananas_each_present_child_gets : ℕ)
  (extra_bananas : ℕ) :
  total_children = 840 ∧
  bananas_per_child = 2 ∧
  absent_children = 420 ∧
  present_children = total_children - absent_children ∧
  total_bananas = total_children * bananas_per_child ∧
  bananas_each_present_child_gets = total_bananas / present_children ∧
  extra_bananas = bananas_each_present_child_gets - bananas_per_child →
  extra_bananas = 2 :=
by
  sorry

end extra_bananas_each_child_gets_l1267_126783


namespace point_D_eq_1_2_l1267_126787

-- Definitions and conditions
def point : Type := ℝ × ℝ

def A : point := (-1, 4)
def B : point := (-4, -1)
def C : point := (4, 7)

-- Translate function
def translate (p : point) (dx dy : ℝ) := (p.1 + dx, p.2 + dy)

-- The translation distances found from A to C
def dx := C.1 - A.1
def dy := C.2 - A.2

-- The point D
def D : point := translate B dx dy

-- Proof objective
theorem point_D_eq_1_2 : D = (1, 2) := by
  sorry

end point_D_eq_1_2_l1267_126787


namespace one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1_l1267_126701

-- Definition of the conditions.
variable (x : ℝ)

-- Statement of the problem in Lean.
theorem one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1 (x : ℝ) :
    (1 / 3) * (9 * x - 3) = 3 * x - 1 :=
by sorry

end one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1_l1267_126701


namespace boat_speed_is_13_l1267_126773

noncomputable def boatSpeedStillWater : ℝ := 
  let Vs := 6 -- Speed of the stream in km/hr
  let time := 3.6315789473684212 -- Time taken in hours to travel 69 km downstream
  let distance := 69 -- Distance traveled in km
  (distance - Vs * time) / time

theorem boat_speed_is_13 : boatSpeedStillWater = 13 := by
  sorry

end boat_speed_is_13_l1267_126773


namespace inequality_proof_l1267_126742

theorem inequality_proof (a b c d : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
  (a_geq_1 : 1 ≤ a) (b_geq_1 : 1 ≤ b) (c_geq_1 : 1 ≤ c)
  (abcd_eq_1 : a * b * c * d = 1)
  : 
  1 / (a^2 - a + 1)^2 + 1 / (b^2 - b + 1)^2 + 1 / (c^2 - c + 1)^2 + 1 / (d^2 - d + 1)^2 ≤ 4
  := sorry

end inequality_proof_l1267_126742


namespace function_properties_l1267_126776

theorem function_properties (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 25 ≠ 0 ∧ x^2 - (k - 6) * x + 16 ≠ 0) → 
  (-2 < k ∧ k < 10) :=
by
  intros h
  sorry

end function_properties_l1267_126776


namespace solve_problem_l1267_126745

def problem_statement (x y : ℕ) : Prop :=
  (x = 3) ∧ (y = 2) → (x^8 + 2 * x^4 * y^2 + y^4) / (x^4 + y^2) = 85

theorem solve_problem : problem_statement 3 2 :=
  by sorry

end solve_problem_l1267_126745


namespace sawing_time_determination_l1267_126710

variable (totalLength pieceLength sawTime : Nat)

theorem sawing_time_determination
  (h1 : totalLength = 10)
  (h2 : pieceLength = 2)
  (h3 : sawTime = 10) :
  (totalLength / pieceLength - 1) * sawTime = 40 := by
  sorry

end sawing_time_determination_l1267_126710


namespace highest_daily_profit_and_total_profit_l1267_126718

def cost_price : ℕ := 6
def standard_price : ℕ := 10

def price_relative (day : ℕ) : ℤ := 
  match day with
  | 1 => 3
  | 2 => 2
  | 3 => 1
  | 4 => -1
  | 5 => -2
  | _ => 0

def quantity_sold (day : ℕ) : ℕ :=
  match day with
  | 1 => 7
  | 2 => 12
  | 3 => 15
  | 4 => 32
  | 5 => 34
  | _ => 0

noncomputable def selling_price (day : ℕ) : ℤ := standard_price + price_relative day

noncomputable def profit_per_pen (day : ℕ) : ℤ := (selling_price day) - cost_price

noncomputable def daily_profit (day : ℕ) : ℤ := (profit_per_pen day) * (quantity_sold day)

theorem highest_daily_profit_and_total_profit 
  (h_highest_profit: daily_profit 4 = 96) 
  (h_total_profit: daily_profit 1 + daily_profit 2 + daily_profit 3 + daily_profit 4 + daily_profit 5 = 360) : 
  True :=
by
  sorry

end highest_daily_profit_and_total_profit_l1267_126718


namespace compl_union_eq_l1267_126709

-- Definitions
def U : Set ℤ := {x | 1 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {1, 3, 4}
def B : Set ℤ := {2, 4}

-- The statement
theorem compl_union_eq : (Aᶜ ∩ U) ∪ B = {2, 4, 5, 6} :=
by sorry

end compl_union_eq_l1267_126709


namespace rate_is_correct_l1267_126737

noncomputable def rate_of_interest (P A T : ℝ) : ℝ :=
  let SI := A - P
  (SI * 100) / (P * T)

theorem rate_is_correct :
  rate_of_interest 10000 18500 8 = 10.625 := 
by
  sorry

end rate_is_correct_l1267_126737


namespace trivia_team_l1267_126771

theorem trivia_team (total_students groups students_per_group students_not_picked : ℕ) (h1 : total_students = 65)
  (h2 : groups = 8) (h3 : students_per_group = 6) (h4 : students_not_picked = total_students - groups * students_per_group) :
  students_not_picked = 17 :=
sorry

end trivia_team_l1267_126771


namespace quadratic_vertex_l1267_126756

theorem quadratic_vertex (x y : ℝ) (h : y = -3 * x^2 + 2) : (x, y) = (0, 2) :=
sorry

end quadratic_vertex_l1267_126756


namespace correct_option_is_D_l1267_126782

def p : Prop := 3 ≥ 3
def q : Prop := 3 > 4

theorem correct_option_is_D (hp : p) (hq : ¬ q) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬ ¬ p :=
by
  sorry

end correct_option_is_D_l1267_126782


namespace tan_monotone_increasing_interval_l1267_126707

theorem tan_monotone_increasing_interval :
  ∀ k : ℤ, ∀ x : ℝ, 
  (-π / 2 + k * π < x + π / 4 ∧ x + π / 4 < π / 2 + k * π) ↔
  (k * π - 3 * π / 4 < x ∧ x < k * π + π / 4) :=
by sorry

end tan_monotone_increasing_interval_l1267_126707


namespace camera_pictures_olivia_camera_pictures_l1267_126796

theorem camera_pictures (phone_pics : Nat) (albums : Nat) (pics_per_album : Nat) (total_pics : Nat) : Prop :=
  phone_pics = 5 →
  albums = 8 →
  pics_per_album = 5 →
  total_pics = albums * pics_per_album →
  total_pics - phone_pics = 35

-- Here's the statement of the theorem followed by a sorry to indicate that the proof is not provided
theorem olivia_camera_pictures (phone_pics albums pics_per_album total_pics : Nat) (h1 : phone_pics = 5) (h2 : albums = 8) (h3 : pics_per_album = 5) (h4 : total_pics = albums * pics_per_album) : total_pics - phone_pics = 35 :=
by
  sorry

end camera_pictures_olivia_camera_pictures_l1267_126796


namespace new_salary_correct_l1267_126716

-- Define the initial salary and percentage increase as given in the conditions
def initial_salary : ℝ := 10000
def percentage_increase : ℝ := 0.02

-- Define the function that calculates the new salary after a percentage increase
def new_salary (initial_salary : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_salary + (initial_salary * percentage_increase)

-- The theorem statement that proves the new salary is €10,200
theorem new_salary_correct :
  new_salary initial_salary percentage_increase = 10200 := by
  sorry

end new_salary_correct_l1267_126716


namespace smallest_x_l1267_126700

theorem smallest_x (M x : ℕ) (h : 720 * x = M^3) : x = 300 :=
by
  sorry

end smallest_x_l1267_126700


namespace retailer_marked_price_percentage_above_cost_l1267_126767

noncomputable def cost_price : ℝ := 100
noncomputable def discount_rate : ℝ := 0.15
noncomputable def sales_profit_rate : ℝ := 0.275

theorem retailer_marked_price_percentage_above_cost :
  ∃ (MP : ℝ), ((MP - cost_price) / cost_price = 0.5) ∧ (((MP * (1 - discount_rate)) - cost_price) / cost_price = sales_profit_rate) :=
sorry

end retailer_marked_price_percentage_above_cost_l1267_126767


namespace fraction_calculation_l1267_126719

theorem fraction_calculation : ( ( (1/2 : ℚ) + (1/5) ) / ( (3/7) - (1/14) ) * (2/3) ) = 98/75 :=
by
  sorry

end fraction_calculation_l1267_126719


namespace sum_of_coefficients_shifted_function_l1267_126708

def original_function (x : ℝ) : ℝ :=
  3*x^2 - 2*x + 6

def shifted_function (x : ℝ) : ℝ :=
  original_function (x + 5)

theorem sum_of_coefficients_shifted_function : 
  let a := 3
  let b := 28
  let c := 71
  a + b + c = 102 :=
by
  -- Placeholder for the proof
  sorry

end sum_of_coefficients_shifted_function_l1267_126708


namespace f_g_2_eq_36_l1267_126766

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 4 * x - 2

theorem f_g_2_eq_36 : f (g 2) = 36 :=
by
  sorry

end f_g_2_eq_36_l1267_126766


namespace trig_identity_l1267_126725

variable (α : ℝ)

theorem trig_identity (h : Real.sin (α - 70 * Real.pi / 180) = α) : 
  Real.cos (α + 20 * Real.pi / 180) = -α := by
  sorry

end trig_identity_l1267_126725


namespace intersection_nonempty_iff_l1267_126705

/-- Define sets A and B as described in the problem. -/
def A (x : ℝ) : Prop := -2 < x ∧ x ≤ 1
def B (x : ℝ) (k : ℝ) : Prop := x ≥ k

/-- The main theorem to prove the range of k where the intersection of A and B is non-empty. -/
theorem intersection_nonempty_iff (k : ℝ) : (∃ x, A x ∧ B x k) ↔ k ≤ 1 :=
by
  sorry

end intersection_nonempty_iff_l1267_126705


namespace exponent_inequality_l1267_126795

theorem exponent_inequality (a b c : ℝ) (h1 : a ≠ 1) (h2 : b ≠ 1) (h3 : c ≠ 1) (h4 : a > b) (h5 : b > c) (h6 : c > 0) : a ^ b > c ^ b :=
  sorry

end exponent_inequality_l1267_126795


namespace math_problem_l1267_126717

theorem math_problem
  (a b c d m : ℝ)
  (h1 : a = -b)
  (h2 : c = (1 / d) ∨ d = (1 / c))
  (h3 : |m| = 4) :
  (a + b = 0) ∧ (c * d = 1) ∧ (m = 4 ∨ m = -4) ∧
  ((a + b) / 3 + m^2 - 5 * (c * d) = 11) := by
  sorry

end math_problem_l1267_126717


namespace sequence_sum_equality_l1267_126794

theorem sequence_sum_equality {a_n : ℕ → ℕ} (S_n : ℕ → ℕ) (n : ℕ) (h : n > 0) 
  (h1 : ∀ n, 3 * a_n n = 2 * S_n n + n) : 
  S_n n = (3^((n:ℕ)+1) - 2 * n) / 4 := 
sorry

end sequence_sum_equality_l1267_126794


namespace radius_excircle_ABC_l1267_126774

variables (A B C P Q : Point)
variables (r_ABP r_APQ r_AQC : ℝ) (re_ABP re_APQ re_AQC : ℝ)
variable (r_ABC : ℝ)

-- Conditions
-- Radii of the incircles of triangles ABP, APQ, and AQC are all equal to 1
axiom incircle_ABP : r_ABP = 1
axiom incircle_APQ : r_APQ = 1
axiom incircle_AQC : r_AQC = 1

-- Radii of the corresponding excircles opposite A for ABP, APQ, and AQC are 3, 6, and 5 respectively
axiom excircle_ABP : re_ABP = 3
axiom excircle_APQ : re_APQ = 6
axiom excircle_AQC : re_AQC = 5

-- Radius of the incircle of triangle ABC is 3/2
axiom incircle_ABC : r_ABC = 3 / 2

-- Theorem stating the radius of the excircle of triangle ABC opposite A is 135
theorem radius_excircle_ABC (r_ABC : ℝ) : r_ABC = 3 / 2 → ∀ (re_ABC : ℝ), re_ABC = 135 := 
by
  intros 
  sorry

end radius_excircle_ABC_l1267_126774


namespace jerry_games_before_birthday_l1267_126777

def num_games_before (current received : ℕ) : ℕ :=
  current - received

theorem jerry_games_before_birthday : 
  ∀ (current received before : ℕ), current = 9 → received = 2 → before = num_games_before current received → before = 7 :=
by
  intros current received before h_current h_received h_before
  rw [h_current, h_received] at h_before
  exact h_before

end jerry_games_before_birthday_l1267_126777


namespace cat_litter_container_weight_l1267_126703

theorem cat_litter_container_weight :
  (∀ (cost_container : ℕ) (pounds_per_litterbox : ℕ) (cost_total : ℕ) (days : ℕ),
    cost_container = 21 ∧ pounds_per_litterbox = 15 ∧ cost_total = 210 ∧ days = 210 → 
    ∀ (weeks : ℕ), weeks = days / 7 →
    ∀ (containers : ℕ), containers = cost_total / cost_container →
    ∀ (cost_per_container : ℕ), cost_per_container = cost_total / containers →
    (∃ (pounds_per_container : ℕ), pounds_per_container = cost_container / cost_per_container ∧ pounds_per_container = 3)) :=
by
  intros cost_container pounds_per_litterbox cost_total days
  intros h weeks hw containers hc containers_cost hc_cost
  sorry

end cat_litter_container_weight_l1267_126703


namespace alloy_problem_solution_l1267_126755

theorem alloy_problem_solution (x y k n : ℝ) (H_weight : k * 4 * x + n * 3 * y = 10)
    (H_ratio : (kx + ny)/(k * 3 * x + n * 2 * y) = 3/7) :
    k * 4 * x = 4 :=
by
  -- Proof to be provided
  sorry

end alloy_problem_solution_l1267_126755


namespace find_y_minus_x_l1267_126722

theorem find_y_minus_x (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) 
  (h4 : Real.sqrt x + Real.sqrt y = 1) 
  (h5 : Real.sqrt (x / y) + Real.sqrt (y / x) = 10 / 3) : 
  y - x = 1 / 2 :=
sorry

end find_y_minus_x_l1267_126722


namespace gcd_of_1237_and_1849_l1267_126733

def gcd_1237_1849 : ℕ := 1

theorem gcd_of_1237_and_1849 : Nat.gcd 1237 1849 = gcd_1237_1849 := by
  sorry

end gcd_of_1237_and_1849_l1267_126733


namespace regression_line_estimate_l1267_126741

theorem regression_line_estimate:
  (∀ (x y : ℝ), y = 1.23 * x + a ↔ a = 5 - 1.23 * 4) →
  ∃ (y : ℝ), y = 1.23 * 2 + 0.08 :=
by
  intro h
  use 2.54
  simp
  sorry

end regression_line_estimate_l1267_126741


namespace degree_to_radian_l1267_126772

theorem degree_to_radian (h : 1 = (π / 180)) : 60 = π * (1 / 3) := 
sorry

end degree_to_radian_l1267_126772


namespace min_deg_q_l1267_126746

-- Definitions of polynomials requirements
variables (p q r : Polynomial ℝ)

-- Given Conditions
def polynomials_relation : Prop := 5 * p + 6 * q = r
def deg_p : Prop := p.degree = 10
def deg_r : Prop := r.degree = 12

-- The main theorem we want to prove
theorem min_deg_q (h1 : polynomials_relation p q r) (h2 : deg_p p) (h3 : deg_r r) : q.degree ≥ 12 :=
sorry

end min_deg_q_l1267_126746


namespace first_term_of_arithmetic_sequence_l1267_126778

theorem first_term_of_arithmetic_sequence (T : ℕ → ℝ) (b : ℝ) 
  (h1 : ∀ n : ℕ, T n = (n * (2 * b + (n - 1) * 4)) / 2) 
  (h2 : ∃ d : ℝ, ∀ n : ℕ, T (4 * n) / T n = d) :
  b = 2 :=
by
  sorry

end first_term_of_arithmetic_sequence_l1267_126778


namespace triangle_perimeter_l1267_126744

theorem triangle_perimeter
  (x : ℝ) 
  (h : x^2 - 6 * x + 8 = 0)
  (a b c : ℝ)
  (ha : a = 2)
  (hb : b = 4)
  (hc : c = x)
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 10 := 
sorry

end triangle_perimeter_l1267_126744


namespace probability_of_two_one_color_and_one_other_color_l1267_126738

theorem probability_of_two_one_color_and_one_other_color
    (black_balls white_balls : ℕ)
    (total_drawn : ℕ)
    (draw_two_black_one_white : ℕ)
    (draw_one_black_two_white : ℕ)
    (total_ways : ℕ)
    (favorable_ways : ℕ)
    (probability : ℚ) :
    black_balls = 8 →
    white_balls = 7 →
    total_drawn = 3 →
    draw_two_black_one_white = 196 →
    draw_one_black_two_white = 168 →
    total_ways = 455 →
    favorable_ways = draw_two_black_one_white + draw_one_black_two_white →
    probability = favorable_ways / total_ways →
    probability = 4 / 5 :=
by sorry

end probability_of_two_one_color_and_one_other_color_l1267_126738


namespace verify_probabilities_l1267_126702

/-- A bag contains 2 red balls, 3 black balls, and 4 white balls, all of the same size.
    A ball is drawn from the bag at a time, and once drawn, it is not replaced. -/
def total_balls := 9
def red_balls := 2
def black_balls := 3
def white_balls := 4

/-- Calculate the probability that the first ball is black and the second ball is white. -/
def prob_first_black_second_white :=
  (black_balls / total_balls) * (white_balls / (total_balls - 1))

/-- Calculate the probability that the number of draws does not exceed 3, 
    given that drawing a red ball means stopping. -/
def prob_draws_not_exceed_3 :=
  (red_balls / total_balls) +
  ((total_balls - red_balls) / total_balls) * (red_balls / (total_balls - 1)) +
  ((total_balls - red_balls - 1) / total_balls) *
  ((total_balls - red_balls) / (total_balls - 1)) *
  (red_balls / (total_balls - 2))

/-- Theorem that verifies the probabilities based on the given conditions. -/
theorem verify_probabilities :
  prob_first_black_second_white = 1 / 6 ∧
  prob_draws_not_exceed_3 = 7 / 12 :=
by
  sorry

end verify_probabilities_l1267_126702


namespace circles_touch_each_other_l1267_126768

-- Define the radii of the two circles and the distance between their centers.
variables (R r d : ℝ)

-- Hypotheses: the condition and the relationships derived from the solution.
variables (x y t : ℝ)

-- The core relationships as conditions based on the problem and the solution.
axiom h1 : x + y = t
axiom h2 : x / y = R / r
axiom h3 : t / d = x / R

-- The proof statement
theorem circles_touch_each_other 
  (h1 : x + y = t) 
  (h2 : x / y = R / r) 
  (h3 : t / d = x / R) : 
  d = R + r := 
by 
  sorry

end circles_touch_each_other_l1267_126768


namespace polynomial_factorization_l1267_126734

theorem polynomial_factorization (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^2 + ab + ac + b^2 + bc + c^2) :=
sorry

end polynomial_factorization_l1267_126734


namespace num_of_terms_in_arithmetic_sequence_l1267_126727

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the first term, common difference, and last term of the sequence
def a : ℕ := 15
def d : ℕ := 4
def last_term : ℕ := 99

-- Define the number of terms in the sequence
def n : ℕ := 22

-- State the theorem
theorem num_of_terms_in_arithmetic_sequence : arithmetic_seq a d n = last_term :=
by
  sorry

end num_of_terms_in_arithmetic_sequence_l1267_126727


namespace everett_weeks_worked_l1267_126749

theorem everett_weeks_worked (daily_hours : ℕ) (total_hours : ℕ) (days_in_week : ℕ) 
  (h1 : daily_hours = 5) (h2 : total_hours = 140) (h3 : days_in_week = 7) : 
  (total_hours / (daily_hours * days_in_week) = 4) :=
by
  sorry

end everett_weeks_worked_l1267_126749


namespace sum_place_values_of_7s_l1267_126704

theorem sum_place_values_of_7s (n : ℝ) (h : n = 87953.0727) : 
  let a := 7000
  let b := 0.07
  let c := 0.0007
  a + b + c = 7000.0707 :=
by
  sorry

end sum_place_values_of_7s_l1267_126704


namespace simplify_and_evaluate_l1267_126743

theorem simplify_and_evaluate : 
  (1 / (3 - 2) - 1 / (3 + 1)) / (3 / (3^2 - 1)) = 2 :=
by
  sorry

end simplify_and_evaluate_l1267_126743


namespace remainder_of_polynomial_division_l1267_126732

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * x^3 - 18 * x^2 + 31 * x - 40

-- Define the divisor D(x)
def D (x : ℝ) : ℝ := 5 * x - 10

-- Prove that the remainder when P(x) is divided by D(x) is -10
theorem remainder_of_polynomial_division : (P 2) = -10 := by
  sorry

end remainder_of_polynomial_division_l1267_126732


namespace caitlinAgeIsCorrect_l1267_126760

-- Define Aunt Anna's age
def auntAnnAge : Nat := 48

-- Define the difference between Aunt Anna's age and 18
def ageDifference : Nat := auntAnnAge - 18

-- Define Brianna's age as twice the difference
def briannaAge : Nat := 2 * ageDifference

-- Define Caitlin's age as 6 years younger than Brianna
def caitlinAge : Nat := briannaAge - 6

-- Theorem to prove Caitlin's age
theorem caitlinAgeIsCorrect : caitlinAge = 54 := by
  sorry -- Proof to be filled in

end caitlinAgeIsCorrect_l1267_126760


namespace num_supermarkets_in_US_l1267_126797

theorem num_supermarkets_in_US (U C : ℕ) (h1 : U + C = 420) (h2 : U = C + 56) : U = 238 :=
by
  sorry

end num_supermarkets_in_US_l1267_126797


namespace intersection_result_complement_union_result_l1267_126799

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x > 0}

theorem intersection_result : A ∩ B = {x | 0 < x ∧ x < 2} :=
by
  sorry

theorem complement_union_result : (compl B) ∪ A = {x | x < 2} :=
by
  sorry

end intersection_result_complement_union_result_l1267_126799


namespace sin_70_eq_1_minus_2k_squared_l1267_126780

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : Real.sin (10 * Real.pi / 180) = k) :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * k^2 :=
by
  sorry

end sin_70_eq_1_minus_2k_squared_l1267_126780


namespace arun_weight_average_l1267_126739

theorem arun_weight_average (w : ℝ) 
  (h1 : 64 < w ∧ w < 72) 
  (h2 : 60 < w ∧ w < 70) 
  (h3 : w ≤ 67) : 
  (64 + 67) / 2 = 65.5 := 
  by sorry

end arun_weight_average_l1267_126739


namespace dealer_profit_percentage_l1267_126789

-- Definitions of conditions
def cost_price (C : ℝ) : ℝ := C
def list_price (C : ℝ) : ℝ := 1.5 * C
def discount_rate : ℝ := 0.1
def discounted_price (C : ℝ) : ℝ := (1 - discount_rate) * list_price C
def price_for_45_articles (C : ℝ) : ℝ := 45 * discounted_price C
def cost_for_40_articles (C : ℝ) : ℝ := 40 * cost_price C

-- Statement of the problem
theorem dealer_profit_percentage (C : ℝ) (h₀ : C > 0) :
  (price_for_45_articles C - cost_for_40_articles C) / cost_for_40_articles C * 100 = 35 :=  
sorry

end dealer_profit_percentage_l1267_126789


namespace flagpole_breaking_height_l1267_126753

theorem flagpole_breaking_height (x : ℝ) (h_pos : 0 < x) (h_ineq : x < 6)
    (h_pythagoras : (x^2 + 2^2 = 6^2)) : x = Real.sqrt 10 :=
by sorry

end flagpole_breaking_height_l1267_126753


namespace factorial_inequality_l1267_126757

theorem factorial_inequality (n : ℕ) : 2^n * n! < (n+1)^n :=
by
  sorry

end factorial_inequality_l1267_126757


namespace modulo_residue_l1267_126758

theorem modulo_residue : 
  ∃ (x : ℤ), 0 ≤ x ∧ x < 31 ∧ (-1237 % 31) = x := 
  sorry

end modulo_residue_l1267_126758


namespace determine_y_l1267_126784

theorem determine_y (x y : ℤ) (h1 : x^2 + 4 * x - 1 = y - 2) (h2 : x = -3) : y = -2 := by
  intros
  sorry

end determine_y_l1267_126784


namespace triangle_areas_l1267_126730

-- Define points based on the conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Triangle DEF vertices
def D : Point := { x := 0, y := 4 }
def E : Point := { x := 6, y := 0 }
def F : Point := { x := 6, y := 5 }

-- Triangle GHI vertices
def G : Point := { x := 0, y := 8 }
def H : Point := { x := 0, y := 6 }
def I : Point := F  -- I and F are the same point

-- Auxiliary function to calculate area of a triangle given its vertices
def area (A B C : Point) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

-- Prove that the areas are correct
theorem triangle_areas :
  area D E F = 15 ∧ area G H I = 6 :=
by
  sorry

end triangle_areas_l1267_126730


namespace problem_l1267_126706

variable {x y : ℝ}

theorem problem (h : x < y) : 3 - x > 3 - y :=
sorry

end problem_l1267_126706


namespace inequality_for_pos_a_b_c_d_l1267_126748

theorem inequality_for_pos_a_b_c_d
  (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) * (b + c) * (c + d) * (d + a) * (1 + (abcd ^ (1/4)))^4
  ≥ 16 * abcd * (1 + a) * (1 + b) * (1 + c) * (1 + d) :=
by
  sorry

end inequality_for_pos_a_b_c_d_l1267_126748


namespace proof_problem_l1267_126769

variable (a b c : ℝ)
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_prod : a * b * c = 1)
variable (h_ineq : a^2011 + b^2011 + c^2011 < (1 / a)^2011 + (1 / b)^2011 + (1 / c)^2011)

theorem proof_problem : a + b + c < 1 / a + 1 / b + 1 / c := 
  sorry

end proof_problem_l1267_126769


namespace dan_gave_marbles_l1267_126761

-- Conditions as definitions in Lean 4
def original_marbles : ℕ := 64
def marbles_left : ℕ := 50
def marbles_given : ℕ := original_marbles - marbles_left

-- Theorem statement proving the question == answer given the conditions.
theorem dan_gave_marbles : marbles_given = 14 := by
  sorry

end dan_gave_marbles_l1267_126761


namespace alphazia_lost_words_l1267_126728

def alphazia_letters := 128
def forbidden_letters := 2
def total_forbidden_pairs := forbidden_letters * alphazia_letters

theorem alphazia_lost_words :
  let one_letter_lost := forbidden_letters
  let two_letter_lost := 2 * alphazia_letters
  one_letter_lost + two_letter_lost = 258 :=
by
  sorry

end alphazia_lost_words_l1267_126728


namespace find_number_l1267_126763

def valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ((n % 10 = 6 ∧ ¬ (n % 7 = 0)) ∨ (¬ (n % 10 = 6) ∧ n % 7 = 0)) ∧
  ((n > 26 ∧ ¬ (n % 10 = 8)) ∨ (¬ (n > 26) ∧ n % 10 = 8)) ∧
  ((n % 13 = 0 ∧ ¬ (n < 27)) ∨ (¬ (n % 13 = 0) ∧ n < 27))

theorem find_number : ∃ n : ℕ, valid_number n ∧ n = 91 := by
  sorry

end find_number_l1267_126763


namespace max_value_in_interval_l1267_126747

variable {R : Type*} [OrderedCommRing R]

variables (f : R → R)
variables (odd_f : ∀ x, f (-x) = -f (x))
variables (f_increasing : ∀ x y, 0 < x → x < y → f x < f y)
variables (additive_f : ∀ x y, f (x + y) = f x + f y)
variables (f1_eq_2 : f 1 = 2)

theorem max_value_in_interval : ∀ x ∈ Set.Icc (-3 : R) (-2 : R), f x ≤ f (-2) ∧ f (-2) = -4 :=
by
  sorry

end max_value_in_interval_l1267_126747


namespace range_of_a_l1267_126721

noncomputable def quadratic_inequality_holds (a : ℝ) : Prop :=
  ∀ (x : ℝ), a * x^2 - a * x - 1 < 0 

theorem range_of_a (a : ℝ) : quadratic_inequality_holds a ↔ -4 < a ∧ a ≤ 0 := 
sorry

end range_of_a_l1267_126721


namespace range_of_a_l1267_126752

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (hf_odd : ∀ x, f (-x) = -f x)
  (hf_expr_pos : ∀ x, x > 0 → f x = -x^2 + ax - 1 - a)
  (hf_monotone : ∀ x y, x < y → f y ≤ f x) :
  -1 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l1267_126752


namespace triangle_area_interval_l1267_126770

theorem triangle_area_interval (s : ℝ) :
  10 ≤ (s - 1)^(3 / 2) ∧ (s - 1)^(3 / 2) ≤ 50 → (5.64 ≤ s ∧ s ≤ 18.32) :=
by
  sorry

end triangle_area_interval_l1267_126770


namespace smallest_integer_representable_l1267_126798

theorem smallest_integer_representable (a b : ℕ) (h₁ : 3 < a) (h₂ : 3 < b)
    (h₃ : a + 3 = 3 * b + 1) : 13 = min (a + 3) (3 * b + 1) :=
by
  sorry

end smallest_integer_representable_l1267_126798


namespace right_triangle_legs_l1267_126714

theorem right_triangle_legs (a b : ℕ) (hypotenuse : ℕ) (h : hypotenuse = 39) : a^2 + b^2 = 39^2 → (a = 15 ∧ b = 36) ∨ (a = 36 ∧ b = 15) :=
by
  sorry

end right_triangle_legs_l1267_126714


namespace tangerines_left_proof_l1267_126790

-- Define the number of tangerines Jimin ate
def tangerinesJiminAte : ℕ := 7

-- Define the total number of tangerines
def totalTangerines : ℕ := 12

-- Define the number of tangerines left
def tangerinesLeft : ℕ := totalTangerines - tangerinesJiminAte

-- Theorem stating the number of tangerines left equals 5
theorem tangerines_left_proof : tangerinesLeft = 5 := 
by
  sorry

end tangerines_left_proof_l1267_126790


namespace minimum_tenth_game_score_l1267_126762

theorem minimum_tenth_game_score (S5 : ℕ) (score10 : ℕ) 
  (h1 : 18 + 15 + 16 + 19 = 68)
  (h2 : S5 ≤ 85)
  (h3 : (S5 + 68 + score10) / 10 > 17) : 
  score10 ≥ 18 := sorry

end minimum_tenth_game_score_l1267_126762


namespace sum_of_angles_l1267_126736

theorem sum_of_angles (x y : ℝ) (n : ℕ) :
  n = 16 →
  (∃ k l : ℕ, k = 3 ∧ l = 5 ∧ 
  x = (k * (360 / n)) / 2 ∧ y = (l * (360 / n)) / 2) →
  x + y = 90 :=
by
  intros
  sorry

end sum_of_angles_l1267_126736


namespace rectangle_side_l1267_126740

theorem rectangle_side (x : ℝ) (w : ℝ) (P : ℝ) (hP : P = 30) (h : 2 * (x + w) = P) : w = 15 - x :=
by
  -- Proof goes here
  sorry

end rectangle_side_l1267_126740


namespace zoey_finished_on_monday_l1267_126781

def total_days_read (n : ℕ) : ℕ :=
  2 * ((2^n) - 1)

def day_of_week_finished (start_day : ℕ) (total_days : ℕ) : ℕ :=
  (start_day + total_days) % 7

theorem zoey_finished_on_monday :
  day_of_week_finished 1 (total_days_read 18) = 1 :=
by
  sorry

end zoey_finished_on_monday_l1267_126781
