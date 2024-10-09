import Mathlib

namespace arrangement_with_one_ball_per_box_arrangement_with_one_empty_box_l1117_111711

-- Definition of the first proof problem
theorem arrangement_with_one_ball_per_box:
  ∃ n : ℕ, n = 24 ∧ 
    -- Number of ways to arrange 4 different balls in 4 boxes such that each box has exactly one ball
    n = Nat.factorial 4 :=
by sorry

-- Definition of the second proof problem
theorem arrangement_with_one_empty_box:
  ∃ n : ℕ, n = 144 ∧ 
    -- Number of ways to arrange 4 different balls in 4 boxes such that exactly one box is empty
    n = Nat.choose 4 2 * Nat.factorial 3 :=
by sorry

end arrangement_with_one_ball_per_box_arrangement_with_one_empty_box_l1117_111711


namespace find_d_k_l1117_111717

open Matrix

noncomputable def matrix_A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![6, d]]

noncomputable def inv_matrix_A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let detA := 3 * d - 24
  (1 / detA) • ![![d, -4], ![-6, 3]]

theorem find_d_k (d k : ℝ) (h : inv_matrix_A d = k • matrix_A d) :
    (d, k) = (-3, 1/33) := by
  sorry

end find_d_k_l1117_111717


namespace marissa_lunch_calories_l1117_111737

theorem marissa_lunch_calories :
  (1 * 400) + (5 * 20) + (5 * 50) = 750 :=
by
  sorry

end marissa_lunch_calories_l1117_111737


namespace number_of_square_free_odds_l1117_111768

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → false

theorem number_of_square_free_odds (n : ℕ) (h1 : 1 < n) (h2 : n < 200) (h3 : n % 2 = 1) :
  (is_square_free n) ↔ (n = 79) := by
  sorry

end number_of_square_free_odds_l1117_111768


namespace trigonometric_relationship_l1117_111723

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < (π / 2))
variable (hβ : 0 < β ∧ β < π)
variable (h : Real.tan α = Real.cos β / (1 - Real.sin β))

theorem trigonometric_relationship : 
    2 * α - β = π / 2 :=
sorry

end trigonometric_relationship_l1117_111723


namespace solution_set_of_inequality_l1117_111786

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3*x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l1117_111786


namespace total_weekly_cost_correct_l1117_111761

def daily_consumption (cups_per_day : ℕ) (ounces_per_cup : ℝ) : ℝ :=
  cups_per_day * ounces_per_cup

def weekly_consumption (cups_per_day : ℕ) (ounces_per_cup : ℝ) (days_per_week : ℕ) : ℝ :=
  daily_consumption cups_per_day ounces_per_cup * days_per_week

def weekly_cost (weekly_ounces : ℝ) (cost_per_ounce : ℝ) : ℝ :=
  weekly_ounces * cost_per_ounce

def person_A_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 3 0.4 7) 1.40

def person_B_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 1 0.6 7) 1.20

def person_C_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 2 0.5 5) 1.35

def james_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 2 0.5 7) 1.25

def total_weekly_cost : ℝ :=
  person_A_weekly_cost + person_B_weekly_cost + person_C_weekly_cost + james_weekly_cost

theorem total_weekly_cost_correct : total_weekly_cost = 32.30 := by
  unfold total_weekly_cost person_A_weekly_cost person_B_weekly_cost person_C_weekly_cost james_weekly_cost
  unfold weekly_cost weekly_consumption daily_consumption
  sorry

end total_weekly_cost_correct_l1117_111761


namespace hyperbola_asymptotes_correct_l1117_111759

noncomputable def asymptotes_for_hyperbola : Prop :=
  ∀ (x y : ℂ),
    9 * (x : ℂ) ^ 2 - 4 * (y : ℂ) ^ 2 = -36 → 
    (y = (3 / 2) * (-Complex.I) * x) ∨ (y = -(3 / 2) * (-Complex.I) * x)

theorem hyperbola_asymptotes_correct :
  asymptotes_for_hyperbola := 
sorry

end hyperbola_asymptotes_correct_l1117_111759


namespace mary_age_l1117_111744

theorem mary_age :
  ∃ M R : ℕ, (R = M + 30) ∧ (R + 20 = 2 * (M + 20)) ∧ (M = 10) :=
by
  sorry

end mary_age_l1117_111744


namespace coordinates_of_B_l1117_111714

noncomputable def B_coordinates := 
  let A : ℝ × ℝ := (-1, -5)
  let a : ℝ × ℝ := (2, 3)
  let AB := (3 * a.1, 3 * a.2)
  let B := (A.1 + AB.1, A.2 + AB.2)
  B

theorem coordinates_of_B : B_coordinates = (5, 4) := 
by 
  sorry

end coordinates_of_B_l1117_111714


namespace range_of_real_number_a_l1117_111769

theorem range_of_real_number_a (a : ℝ) : (∀ (x : ℝ), 0 < x → a < x + 1/x) → a < 2 := 
by
  sorry

end range_of_real_number_a_l1117_111769


namespace find_x_l1117_111731

-- Define conditions
def simple_interest (x y : ℝ) : Prop :=
  x * y * 2 / 100 = 800

def compound_interest (x y : ℝ) : Prop :=
  x * ((1 + y / 100)^2 - 1) = 820

-- Prove x = 8000 given the conditions
theorem find_x (x y : ℝ) (h1 : simple_interest x y) (h2 : compound_interest x y) : x = 8000 :=
  sorry

end find_x_l1117_111731


namespace verify_statements_l1117_111788

def line1 (a x y : ℝ) : Prop := a * x - (a + 2) * y - 2 = 0
def line2 (a x y : ℝ) : Prop := (a - 2) * x + 3 * a * y + 2 = 0

theorem verify_statements (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ x = -1 ∧ y = -1) ∧
  (∀ x y : ℝ, (line1 a x y ∧ line2 a x y) → (a = 0 ∨ a = -4)) :=
by sorry

end verify_statements_l1117_111788


namespace find_interest_rate_l1117_111780

theorem find_interest_rate
  (P : ℝ)  -- Principal amount
  (A : ℝ)  -- Final amount
  (T : ℝ)  -- Time period in years
  (H1 : P = 1000)
  (H2 : A = 1120)
  (H3 : T = 2.4)
  : ∃ R : ℝ, (A - P) = (P * R * T) / 100 ∧ R = 5 :=
by
  -- Proof with calculations to be provided here
  sorry

end find_interest_rate_l1117_111780


namespace number_of_full_boxes_l1117_111764

theorem number_of_full_boxes (peaches_in_basket baskets_eaten_peaches box_capacity : ℕ) (h1 : peaches_in_basket = 23) (h2 : baskets = 7) (h3 : eaten_peaches = 7) (h4 : box_capacity = 13) :
  (peaches_in_basket * baskets - eaten_peaches) / box_capacity = 11 :=
by
  sorry

end number_of_full_boxes_l1117_111764


namespace second_largest_subtract_smallest_correct_l1117_111762

-- Definition of the elements
def numbers : List ℕ := [10, 11, 12, 13, 14]

-- Conditions derived from the problem
def smallest_number : ℕ := 10
def second_largest_number : ℕ := 13

-- Lean theorem statement representing the problem
theorem second_largest_subtract_smallest_correct :
  (second_largest_number - smallest_number) = 3 := 
by
  sorry

end second_largest_subtract_smallest_correct_l1117_111762


namespace volumes_of_rotated_solids_l1117_111754

theorem volumes_of_rotated_solids
  (π : ℝ)
  (b c a : ℝ)
  (h₁ : a^2 = b^2 + c^2)
  (v v₁ v₂ : ℝ)
  (hv : v = (1/3) * π * (b^2 * c^2) / a)
  (hv₁ : v₁ = (1/3) * π * c^2 * b)
  (hv₂ : v₂ = (1/3) * π * b^2 * c) :
  (1 / v^2) = (1 / v₁^2) + (1 / v₂^2) := 
by sorry

end volumes_of_rotated_solids_l1117_111754


namespace original_number_l1117_111709

theorem original_number (x : ℝ) (h1 : 1.5 * x = 135) : x = 90 :=
by
  sorry

end original_number_l1117_111709


namespace part1_monotonicity_part2_range_a_l1117_111749

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x + 1

theorem part1_monotonicity (a : ℝ) :
  (∀ x > 0, (0 : ℝ) < x → 0 < 1 / x - a) ∨
  (a > 0 → ∀ x > 0, (0 : ℝ) < x ∧ x < 1 / a → 0 < 1 / x - a ∧ 1 / a < x → 1 / x - a < 0) := sorry

theorem part2_range_a (a : ℝ) :
  (∀ x > 0, Real.log x - a * x + 1 ≤ 0) → 1 ≤ a := sorry

end part1_monotonicity_part2_range_a_l1117_111749


namespace sum_every_second_term_is_1010_l1117_111772

def sequence_sum (n : ℕ) (a d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_every_second_term_is_1010 :
  ∃ (x1 : ℤ) (d : ℤ) (S : ℤ), 
  (sequence_sum 2020 x1 d = 6060) ∧
  (d = 2) ∧
  (S = (1010 : ℤ)) ∧ 
  (2 * S + 4040 = 6060) :=
  sorry

end sum_every_second_term_is_1010_l1117_111772


namespace find_y_l1117_111775

theorem find_y (a b : ℝ) (y : ℝ) (h0 : b ≠ 0) (h1 : (3 * a)^(2 * b) = a^b * y^b) : y = 9 * a := by
  sorry

end find_y_l1117_111775


namespace investment_difference_l1117_111733

noncomputable def future_value_semi_annual (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + annual_rate / 2)^((years * 2))

noncomputable def future_value_monthly (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + annual_rate / 12)^((years * 12))

theorem investment_difference :
  let jose_investment := future_value_semi_annual 30000 0.03 3
  let patricia_investment := future_value_monthly 30000 0.025 3
  round (jose_investment) - round (patricia_investment) = 317 :=
by
  sorry

end investment_difference_l1117_111733


namespace cube_problem_l1117_111776

theorem cube_problem (n : ℕ) (h1 : n > 3) :
  (12 * (n - 4) = (n - 2)^3) → n = 5 :=
by {
  sorry
}

end cube_problem_l1117_111776


namespace workers_complete_time_l1117_111771

theorem workers_complete_time 
  (time_A time_B time_C : ℕ) 
  (hA : time_A = 10)
  (hB : time_B = 12) 
  (hC : time_C = 15) : 
  let rate_A := (1: ℚ) / time_A
  let rate_B := (1: ℚ) / time_B
  let rate_C := (1: ℚ) / time_C
  let total_rate := rate_A + rate_B + rate_C
  1 / total_rate = 4 := 
by
  sorry

end workers_complete_time_l1117_111771


namespace same_terminal_side_angle_l1117_111738

theorem same_terminal_side_angle (θ : ℤ) : θ = -390 → ∃ k : ℤ, 0 ≤ θ + k * 360 ∧ θ + k * 360 < 360 ∧ θ + k * 360 = 330 :=
  by
    sorry

end same_terminal_side_angle_l1117_111738


namespace pentagon_edges_same_color_l1117_111782

theorem pentagon_edges_same_color
  (A B : Fin 5 → Fin 5)
  (C : (Fin 5 → Fin 5) × (Fin 5 → Fin 5) → Bool)
  (condition : ∀ (i j : Fin 5), ∀ (k l m : Fin 5), (C (i, j) = C (k, l) → C (i, j) ≠ C (k, m))) :
  (∀ (x : Fin 5), C (A x, A ((x + 1) % 5)) = C (B x, B ((x + 1) % 5))) :=
by
sorry

end pentagon_edges_same_color_l1117_111782


namespace inequality_relation_l1117_111779

theorem inequality_relation (a b : ℝ) :
  (∃ a b : ℝ, a > b ∧ ¬(1/a < 1/b)) ∧ (∃ a b : ℝ, (1/a < 1/b) ∧ ¬(a > b)) :=
by {
  sorry
}

end inequality_relation_l1117_111779


namespace count_total_wheels_l1117_111794

theorem count_total_wheels (trucks : ℕ) (cars : ℕ) (truck_wheels : ℕ) (car_wheels : ℕ) :
  trucks = 12 → cars = 13 → truck_wheels = 4 → car_wheels = 4 →
  (trucks * truck_wheels + cars * car_wheels) = 100 :=
by
  intros h_trucks h_cars h_truck_wheels h_car_wheels
  sorry

end count_total_wheels_l1117_111794


namespace average_speed_l1117_111728

-- Define the conditions as constants and theorems
def distance1 : ℝ := 240
def distance2 : ℝ := 420
def time_diff : ℝ := 3

theorem average_speed : ∃ v t : ℝ, distance1 = v * t ∧ distance2 = v * (t + time_diff) → v = 60 := 
by
  sorry

end average_speed_l1117_111728


namespace non_empty_solution_set_inequality_l1117_111751

theorem non_empty_solution_set_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 3| < a) ↔ a > -4 := 
sorry

end non_empty_solution_set_inequality_l1117_111751


namespace find_n_l1117_111719

theorem find_n (n : ℝ) (h : (2 : ℂ) / (1 - I) = (1 : ℂ) + n * I) : n = 1 := by
  sorry

end find_n_l1117_111719


namespace find_phi_l1117_111785

theorem find_phi 
  (f : ℝ → ℝ)
  (phi : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, f x = Real.sin (2 * x + Real.pi / 6))
  (h2 : 0 < phi ∧ phi < Real.pi / 2)
  (h3 : ∀ x, y x = f (x - phi) ∧ y x = y (-x)) :
  phi = Real.pi / 3 :=
by
  sorry

end find_phi_l1117_111785


namespace find_x_l1117_111736

theorem find_x (x : ℚ) (h : x ≠ 2 ∧ x ≠ 4/5) :
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 22*x - 48)/(5*x - 4) = -7 → x = -4/3 :=
by
  intro h1
  sorry

end find_x_l1117_111736


namespace multiple_of_spending_on_wednesday_l1117_111712

-- Definitions based on the conditions
def monday_spending : ℤ := 60
def tuesday_spending : ℤ := 4 * monday_spending
def total_spending : ℤ := 600

-- Problem to prove
theorem multiple_of_spending_on_wednesday (x : ℤ) : 
  monday_spending + tuesday_spending + x * monday_spending = total_spending → 
  x = 5 := by
  sorry

end multiple_of_spending_on_wednesday_l1117_111712


namespace tea_garden_problem_pruned_to_wild_conversion_l1117_111793

-- Definitions and conditions as per the problem statement
def total_area : ℕ := 16
def total_yield : ℕ := 660
def wild_yield_per_mu : ℕ := 30
def pruned_yield_per_mu : ℕ := 50

-- Lean 4 statement as per the proof problem
theorem tea_garden_problem :
  ∃ (x y : ℕ), (x + y = total_area) ∧ (wild_yield_per_mu * x + pruned_yield_per_mu * y = total_yield) ∧
  x = 7 ∧ y = 9 :=
sorry

-- Additional theorem for the conversion condition
theorem pruned_to_wild_conversion :
  ∀ (a : ℕ), (wild_yield_per_mu * (7 + a) ≥ pruned_yield_per_mu * (9 - a)) → a ≥ 3 :=
sorry

end tea_garden_problem_pruned_to_wild_conversion_l1117_111793


namespace combined_tickets_l1117_111743

-- Definitions from the conditions
def dave_spent : Nat := 43
def dave_left : Nat := 55
def alex_spent : Nat := 65
def alex_left : Nat := 42

-- Theorem to prove that the combined starting tickets of Dave and Alex is 205
theorem combined_tickets : dave_spent + dave_left + alex_spent + alex_left = 205 := 
by
  sorry

end combined_tickets_l1117_111743


namespace equation_of_line_through_point_with_given_slope_l1117_111721

-- Define the condition that line L passes through point P(-2, 5) and has slope -3/4
def line_through_point_with_slope (x1 y1 m : ℚ) (x y : ℚ) : Prop :=
  y - y1 = m * (x - x1)

-- Define the specific point (-2, 5) and slope -3/4
def P : ℚ × ℚ := (-2, 5)
def m : ℚ := -3 / 4

-- The standard form equation of the line as the target
def standard_form (x y : ℚ) : Prop :=
  3 * x + 4 * y - 14 = 0

-- The theorem to prove
theorem equation_of_line_through_point_with_given_slope :
  ∀ x y : ℚ, line_through_point_with_slope (-2) 5 (-3 / 4) x y → standard_form x y :=
  by
    intros x y h
    sorry

end equation_of_line_through_point_with_given_slope_l1117_111721


namespace ratio_of_volumes_of_cones_l1117_111732

theorem ratio_of_volumes_of_cones (r θ h1 h2 : ℝ) (hθ : 3 * θ + 4 * θ = 2 * π)
    (hr1 : r₁ = 3 * r / 7) (hr2 : r₂ = 4 * r / 7) :
    let V₁ := (1 / 3) * π * r₁^2 * h1
    let V₂ := (1 / 3) * π * r₂^2 * h2
    V₁ / V₂ = (9 : ℝ) / 16 := by
  sorry

end ratio_of_volumes_of_cones_l1117_111732


namespace greatest_int_less_than_200_gcd_30_is_5_l1117_111730

theorem greatest_int_less_than_200_gcd_30_is_5 : ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_int_less_than_200_gcd_30_is_5_l1117_111730


namespace book_loss_percentage_l1117_111783

theorem book_loss_percentage 
  (C S : ℝ) 
  (h : 15 * C = 20 * S) : 
  (C - S) / C * 100 = 25 := 
by 
  sorry

end book_loss_percentage_l1117_111783


namespace certain_number_l1117_111742

theorem certain_number (x certain_number : ℕ) (h1 : x = 3327) (h2 : 9873 + x = certain_number) : 
  certain_number = 13200 := 
by
  sorry

end certain_number_l1117_111742


namespace complement_intersection_l1117_111798

-- Definitions for the sets
def U : Set ℕ := {1, 2, 3, 4, 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Definition of the complement of a set with respect to a universal set
def complement (U A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Theorem to prove
theorem complement_intersection :
  complement U (A ∩ B) = {1, 4, 6} :=
by
  sorry

end complement_intersection_l1117_111798


namespace arithmetic_seq_a7_constant_l1117_111791

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

-- Definition of an arithmetic sequence
def is_arithmetic_seq (a : ℕ → α) : Prop :=
∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

-- Given arithmetic sequence {a_n}
variable (a : ℕ → α)
-- Given the property that a_2 + a_4 + a_{15} is a constant
variable (C : α)
variable (h : is_arithmetic_seq a)
variable (h_constant : a 2 + a 4 + a 15 = C)

-- Prove that a_7 is a constant
theorem arithmetic_seq_a7_constant (h : is_arithmetic_seq a) (h_constant : a 2 + a 4 + a 15 = C) : ∃ k : α, a 7 = k :=
by
  sorry

end arithmetic_seq_a7_constant_l1117_111791


namespace students_left_correct_l1117_111787

-- Define the initial number of students
def initial_students : ℕ := 8

-- Define the number of new students
def new_students : ℕ := 8

-- Define the final number of students
def final_students : ℕ := 11

-- Define the number of students who left during the year
def students_who_left : ℕ :=
  (initial_students + new_students) - final_students

theorem students_left_correct : students_who_left = 5 :=
by
  -- Instantiating the definitions
  let initial := initial_students
  let new := new_students
  let final := final_students

  -- Calculation of students who left
  let L := (initial + new) - final

  -- Asserting the result
  show L = 5
  sorry

end students_left_correct_l1117_111787


namespace valid_triangle_inequality_l1117_111773

theorem valid_triangle_inequality (a : ℝ) 
  (h1 : 4 + 6 > a) 
  (h2 : 4 + a > 6) 
  (h3 : 6 + a > 4) : 
  a = 5 :=
sorry

end valid_triangle_inequality_l1117_111773


namespace find_line_equation_l1117_111774

noncomputable def perpendicular_origin_foot := 
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ y = 2 * x + 5) ∧
    l (-2) 1

theorem find_line_equation : 
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ 2 * x - y + 5 = 0) ∧
    l (-2) 1 ∧
    ∀ p q : ℝ, p = 0 → q = 0 → ¬ (l p q)
:= sorry

end find_line_equation_l1117_111774


namespace portion_to_joe_and_darcy_eq_half_l1117_111781

open Int

noncomputable def portion_given_to_joe_and_darcy : ℚ := 
let total_slices := 8
let portion_to_carl := 1 / 4
let slices_to_carl := portion_to_carl * total_slices
let slices_left := 2
let slices_given_to_joe_and_darcy := total_slices - slices_to_carl - slices_left
let portion_to_joe_and_darcy := slices_given_to_joe_and_darcy / total_slices
portion_to_joe_and_darcy

theorem portion_to_joe_and_darcy_eq_half :
  portion_given_to_joe_and_darcy = 1 / 2 :=
sorry

end portion_to_joe_and_darcy_eq_half_l1117_111781


namespace BaSO4_molecular_weight_l1117_111722

noncomputable def Ba : ℝ := 137.327
noncomputable def S : ℝ := 32.065
noncomputable def O : ℝ := 15.999
noncomputable def BaSO4 : ℝ := Ba + S + 4 * O

theorem BaSO4_molecular_weight : BaSO4 = 233.388 := by
  sorry

end BaSO4_molecular_weight_l1117_111722


namespace potato_bag_weight_l1117_111765

-- Defining the weight of the bag of potatoes as a variable W
variable (W : ℝ)

-- Given condition: The weight of the bag is described by the equation
def weight_condition (W : ℝ) := W = 12 / (W / 2)

-- Proving the weight of the bag of potatoes is 12 lbs:
theorem potato_bag_weight : weight_condition W → W = 12 :=
by
  sorry

end potato_bag_weight_l1117_111765


namespace unique_intersection_value_l1117_111702

theorem unique_intersection_value :
  (∀ (x y : ℝ), y = x^2 → y = 4 * x + k) → (k = -4) := 
by
  sorry

end unique_intersection_value_l1117_111702


namespace jana_distance_l1117_111767

theorem jana_distance (time_to_walk_one_mile : ℝ) (time_to_walk : ℝ) :
  (time_to_walk_one_mile = 18) → (time_to_walk = 15) →
  ((time_to_walk / time_to_walk_one_mile) * 1 = 0.8) :=
  by
    intros h1 h2
    rw [h1, h2]
    -- Here goes the proof, but it is skipped as per requirements
    sorry

end jana_distance_l1117_111767


namespace min_value_fracs_l1117_111753

-- Define the problem and its conditions in Lean.
theorem min_value_fracs (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : 
  (2 / a + 3 / b) ≥ 8 + 4 * Real.sqrt 3 :=
  sorry

end min_value_fracs_l1117_111753


namespace cistern_length_l1117_111758

def cistern_conditions (L : ℝ) : Prop := 
  let width := 4
  let depth := 1.25
  let wet_surface_area := 42.5
  (L * width) + (2 * (L * depth)) + (2 * (width * depth)) = wet_surface_area

theorem cistern_length : 
  ∃ L : ℝ, cistern_conditions L ∧ L = 5 := sorry

end cistern_length_l1117_111758


namespace martha_black_butterflies_l1117_111756

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ) 
    (h1 : total_butterflies = 19)
    (h2 : blue_butterflies = 2 * yellow_butterflies)
    (h3 : blue_butterflies = 6) :
    black_butterflies = 10 :=
by
  -- Prove the theorem assuming the conditions are met
  sorry

end martha_black_butterflies_l1117_111756


namespace f_properties_l1117_111734

noncomputable def f : ℕ → ℕ := sorry

theorem f_properties (f : ℕ → ℕ) :
  (∀ x y : ℕ, x > 0 → y > 0 → f (x * y) = f x + f y) →
  (f 10 = 16) →
  (f 40 = 24) →
  (f 3 = 5) →
  (f 800 = 44) :=
by
  intros h1 h2 h3 h4
  sorry

end f_properties_l1117_111734


namespace gcd_polynomial_multiple_528_l1117_111705

-- Definition of the problem
theorem gcd_polynomial_multiple_528 (k : ℕ) : 
  gcd (3 * (528 * k) ^ 3 + (528 * k) ^ 2 + 4 * (528 * k) + 66) (528 * k) = 66 :=
by
  sorry

end gcd_polynomial_multiple_528_l1117_111705


namespace remainder_of_3_pow_100_mod_7_is_4_l1117_111700

theorem remainder_of_3_pow_100_mod_7_is_4
  (h1 : 3^1 ≡ 3 [MOD 7])
  (h2 : 3^2 ≡ 2 [MOD 7])
  (h3 : 3^3 ≡ 6 [MOD 7])
  (h4 : 3^4 ≡ 4 [MOD 7])
  (h5 : 3^5 ≡ 5 [MOD 7])
  (h6 : 3^6 ≡ 1 [MOD 7]) :
  3^100 ≡ 4 [MOD 7] :=
by
  sorry

end remainder_of_3_pow_100_mod_7_is_4_l1117_111700


namespace sufficient_but_not_necessary_condition_l1117_111726

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_even_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

-- This definition states that both f and g are either odd or even functions
def is_odd_or_even (f g : ℝ → ℝ) : Prop := 
  (is_odd f ∧ is_odd g) ∨ (is_even f ∧ is_even g)

theorem sufficient_but_not_necessary_condition (f g : ℝ → ℝ)
  (h : is_odd_or_even f g) : 
  ¬(is_odd f ∧ is_odd g) → is_even_function (f * g) :=
sorry

end sufficient_but_not_necessary_condition_l1117_111726


namespace problem_l1117_111757

theorem problem
  (a b : ℝ)
  (h : a^2 + b^2 - 2 * a + 6 * b + 10 = 0) :
  2 * a^100 - 3 * b⁻¹ = 3 := 
by {
  -- Proof steps go here
  sorry
}

end problem_l1117_111757


namespace joe_paint_fraction_l1117_111799

theorem joe_paint_fraction :
  let total_paint := 360
  let fraction_first_week := 1 / 9
  let used_first_week := (fraction_first_week * total_paint)
  let remaining_after_first_week := total_paint - used_first_week
  let total_used := 104
  let used_second_week := total_used - used_first_week
  let fraction_second_week := used_second_week / remaining_after_first_week
  fraction_second_week = 1 / 5 :=
by
  sorry

end joe_paint_fraction_l1117_111799


namespace cube_face_sum_l1117_111715

theorem cube_face_sum (a d b e c f g : ℕ)
    (h1 : g = 2)
    (h2 : 2310 = 2 * 3 * 5 * 7 * 11)
    (h3 : (a + d) * (b + e) * (c + f) = 3 * 5 * 7 * 11):
    (a + d) + (b + e) + (c + f) = 47 :=
by
    sorry

end cube_face_sum_l1117_111715


namespace number_of_students_playing_soccer_l1117_111748

variables (T B girls_total soccer_total G no_girls_soccer perc_boys_soccer : ℕ)

-- Conditions:
def total_students := T = 420
def boys_students := B = 312
def girls_students := G = 420 - 312
def girls_not_playing_soccer := no_girls_soccer = 63
def perc_boys_play_soccer := perc_boys_soccer = 82
def girls_playing_soccer := G - no_girls_soccer = 45

-- Proof Problem:
theorem number_of_students_playing_soccer (h1 : total_students T) (h2 : boys_students B) (h3 : girls_students G) (h4 : girls_not_playing_soccer no_girls_soccer) (h5 : girls_playing_soccer G no_girls_soccer) (h6 : perc_boys_play_soccer perc_boys_soccer) : soccer_total = 250 :=
by {
  -- The proof would be inserted here.
  sorry
}

end number_of_students_playing_soccer_l1117_111748


namespace combined_original_price_l1117_111789

theorem combined_original_price (S P : ℝ) 
  (hS : 0.25 * S = 6) 
  (hP : 0.60 * P = 12) :
  S + P = 44 :=
by
  sorry

end combined_original_price_l1117_111789


namespace expression_equiv_l1117_111703

theorem expression_equiv :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by
  sorry

end expression_equiv_l1117_111703


namespace domain_of_function_l1117_111701

theorem domain_of_function :
  {x : ℝ | x ≥ -1} \ {0} = {x : ℝ | (x ≥ -1 ∧ x < 0) ∨ x > 0} :=
by
  sorry

end domain_of_function_l1117_111701


namespace charlie_coins_worth_44_cents_l1117_111755

-- Definitions based on the given conditions
def total_coins := 17
def p_eq_n_plus_2 (p n : ℕ) := p = n + 2

-- The main theorem stating the problem and the expected answer
theorem charlie_coins_worth_44_cents (p n : ℕ) (h1 : p + n = total_coins) (h2 : p_eq_n_plus_2 p n) :
  (7 * 5 + p * 1 = 44) :=
sorry

end charlie_coins_worth_44_cents_l1117_111755


namespace points_per_round_l1117_111746

-- Definitions based on conditions
def final_points (jane_points : ℕ) : Prop := jane_points = 60
def lost_points (jane_lost : ℕ) : Prop := jane_lost = 20
def rounds_played (jane_rounds : ℕ) : Prop := jane_rounds = 8

-- The theorem we want to prove
theorem points_per_round (jane_points jane_lost jane_rounds points_per_round : ℕ) 
  (h1 : final_points jane_points) 
  (h2 : lost_points jane_lost) 
  (h3 : rounds_played jane_rounds) : 
  points_per_round = ((jane_points + jane_lost) / jane_rounds) := 
sorry

end points_per_round_l1117_111746


namespace solve_equation_l1117_111795

theorem solve_equation (x : ℝ) : x*(x-3)^2*(5+x) = 0 ↔ x = 0 ∨ x = 3 ∨ x = -5 := 
by 
  sorry

end solve_equation_l1117_111795


namespace triangle_sine_equality_l1117_111735

theorem triangle_sine_equality {a b c : ℝ} {α β γ : ℝ} 
  (cos_rule : c^2 = a^2 + b^2 - 2 * a * b * Real.cos γ)
  (area : ∃ T : ℝ, T = (1 / 2) * a * b * Real.sin γ)
  (sin_addition_γ : Real.sin (γ + Real.pi / 6) = Real.sin γ * (Real.sqrt 3 / 2) + Real.cos γ * (1 / 2))
  (sin_addition_β : Real.sin (β + Real.pi / 6) = Real.sin β * (Real.sqrt 3 / 2) + Real.cos β * (1 / 2))
  (sin_addition_α : Real.sin (α + Real.pi / 6) = Real.sin α * (Real.sqrt 3 / 2) + Real.cos α * (1 / 2)) :
  c^2 + 2 * a * b * Real.sin (γ + Real.pi / 6) = b^2 + 2 * a * c * Real.sin (β + Real.pi / 6) ∧
  b^2 + 2 * a * c * Real.sin (β + Real.pi / 6) = a^2 + 2 * b * c * Real.sin (α + Real.pi / 6) :=
sorry

end triangle_sine_equality_l1117_111735


namespace max_band_members_l1117_111720

theorem max_band_members (n : ℤ) (h1 : 30 * n % 21 = 9) (h2 : 30 * n < 1500) : 30 * n ≤ 1470 :=
by
  -- Proof to be filled in later
  sorry

end max_band_members_l1117_111720


namespace p_q_2r_value_l1117_111739

variable (p q r : ℝ) (f : ℝ → ℝ)

-- The conditions as definitions
def f_def : f = fun x => p * x^2 + q * x + r := by sorry
def f_at_0 : f 0 = 9 := by sorry
def f_at_1 : f 1 = 6 := by sorry

-- The theorem statement
theorem p_q_2r_value : p + q + 2 * r = 15 :=
by
  -- utilizing the given definitions 
  have h₁ : r = 9 := by sorry
  have h₂ : p + q + r = 6 := by sorry
  -- substitute into p + q + 2r
  sorry

end p_q_2r_value_l1117_111739


namespace arcsin_neg_sqrt_two_over_two_l1117_111745

theorem arcsin_neg_sqrt_two_over_two : Real.arcsin (-Real.sqrt 2 / 2) = -Real.pi / 4 :=
  sorry

end arcsin_neg_sqrt_two_over_two_l1117_111745


namespace largest_4_digit_number_divisible_by_12_l1117_111707

theorem largest_4_digit_number_divisible_by_12 : ∃ (n : ℕ), 1000 ≤ n ∧ n ≤ 9999 ∧ n % 12 = 0 ∧ ∀ m, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 12 = 0 → m ≤ n := 
sorry

end largest_4_digit_number_divisible_by_12_l1117_111707


namespace fermat_little_theorem_l1117_111718

theorem fermat_little_theorem (p : ℕ) (hp : Nat.Prime p) (a : ℕ) : a^p ≡ a [MOD p] :=
sorry

end fermat_little_theorem_l1117_111718


namespace subset_neg1_of_leq3_l1117_111790

theorem subset_neg1_of_leq3 :
  {x | x = -1} ⊆ {x | x ≤ 3} :=
sorry

end subset_neg1_of_leq3_l1117_111790


namespace find_x3_y3_l1117_111796

noncomputable def x_y_conditions (x y : ℝ) :=
  x - y = 3 ∧
  x^2 + y^2 = 27

theorem find_x3_y3 (x y : ℝ) (h : x_y_conditions x y) : x^3 - y^3 = 108 :=
  sorry

end find_x3_y3_l1117_111796


namespace expression_equiv_l1117_111729

theorem expression_equiv (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^4 + 1) / x^2) * ((y^4 + 1) / y^2) + ((x^4 - 1) / y^2) * ((y^4 - 1) / x^2) =
  2*x^2*y^2 + 2/(x^2*y^2) :=
by 
  sorry

end expression_equiv_l1117_111729


namespace equation_c_is_linear_l1117_111706

-- Define the condition for being a linear equation with one variable
def is_linear_equation_with_one_variable (eq : ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ (a * x + b = 0)

-- The given equation to check is (x - 1) / 2 = 1, which simplifies to x = 3
def equation_c (x : ℝ) : Prop := (x - 1) / 2 = 1

-- Prove that the given equation is a linear equation with one variable
theorem equation_c_is_linear :
  is_linear_equation_with_one_variable equation_c :=
sorry

end equation_c_is_linear_l1117_111706


namespace vector_parallel_find_k_l1117_111760

theorem vector_parallel_find_k (k : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h₁ : a = (3 * k + 1, 2)) 
  (h₂ : b = (k, 1)) 
  (h₃ : ∃ c : ℝ, a = c • b) : k = -1 := 
by 
  sorry

end vector_parallel_find_k_l1117_111760


namespace sequence_inequality_l1117_111777

variable (a : ℕ → ℝ) (b : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

noncomputable def is_geometric (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) - b n = b 1 - b 0

theorem sequence_inequality
  (ha : ∀ n, 0 < a n)
  (hg : is_geometric a q)
  (ha6_eq_b7 : a 6 = b 7)
  (hb : is_arithmetic b) :
  a 3 + a 9 ≥ b 4 + b 10 :=
by
  sorry

end sequence_inequality_l1117_111777


namespace find_number_l1117_111716

theorem find_number (x : ℝ) (h : 0.2 * x = 0.3 * 120 + 80) : x = 580 :=
by
  sorry

end find_number_l1117_111716


namespace intersection_point_of_y_eq_4x_minus_2_with_x_axis_l1117_111792

theorem intersection_point_of_y_eq_4x_minus_2_with_x_axis :
  ∃ x, (4 * x - 2 = 0 ∧ (x, 0) = (1 / 2, 0)) :=
by
  sorry

end intersection_point_of_y_eq_4x_minus_2_with_x_axis_l1117_111792


namespace A_equals_k_with_conditions_l1117_111727

theorem A_equals_k_with_conditions (n m : ℕ) (h_n : 1 < n) (h_m : 1 < m) :
  ∃ k : ℤ, (1 : ℝ) < k ∧ (( (n + Real.sqrt (n^2 - 4)) / 2 ) ^ m = (k + Real.sqrt (k^2 - 4)) / 2) :=
sorry

end A_equals_k_with_conditions_l1117_111727


namespace percentage_of_women_picnic_l1117_111747

theorem percentage_of_women_picnic (E : ℝ) (h1 : 0.20 * 0.55 * E + W * 0.45 * E = 0.29 * E) : 
  W = 0.4 := 
  sorry

end percentage_of_women_picnic_l1117_111747


namespace prove_range_of_m_prove_m_value_l1117_111724

def quadratic_roots (m : ℝ) (x1 x2 : ℝ) : Prop := 
  x1 * x1 - (2 * m - 3) * x1 + m * m = 0 ∧ 
  x2 * x2 - (2 * m - 3) * x2 + m * m = 0

def range_of_m (m : ℝ) : Prop := 
  m <= 3/4

def condition_on_m (m : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 + x2 = -(x1 * x2)

theorem prove_range_of_m (m : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_roots m x1 x2) → range_of_m m :=
sorry

theorem prove_m_value (m : ℝ) (x1 x2 : ℝ) :
  quadratic_roots m x1 x2 → condition_on_m m x1 x2 → m = -3 :=
sorry

end prove_range_of_m_prove_m_value_l1117_111724


namespace first_comparison_second_comparison_l1117_111741

theorem first_comparison (x y : ℕ) (h1 : x = 2^40) (h2 : y = 3^28) : x < y := 
by sorry

theorem second_comparison (a b : ℕ) (h3 : a = 31^11) (h4 : b = 17^14) : a < b := 
by sorry

end first_comparison_second_comparison_l1117_111741


namespace sum_is_945_l1117_111710

def sum_of_integers_from_90_to_99 : ℕ :=
  90 + 91 + 92 + 93 + 94 + 95 + 96 + 97 + 98 + 99

theorem sum_is_945 : sum_of_integers_from_90_to_99 = 945 := 
by
  sorry

end sum_is_945_l1117_111710


namespace correct_statements_l1117_111750

-- Definitions for each statement
def statement_1 := ∀ p q : ℤ, q ≠ 0 → (∃ n : ℤ, ∃ d : ℤ, p = n ∧ q = d ∧ (n, d) = (p, q))
def statement_2 := ∀ r : ℚ, (r > 0 ∨ r < 0) ∨ (∃ d : ℚ, d ≥ 0)
def statement_3 := ∀ x y : ℚ, abs x = abs y → x = y
def statement_4 := ∀ x : ℚ, (-x = x ∧ abs x = x) → x = 0
def statement_5 := ∀ x y : ℚ, abs x > abs y → x > y
def statement_6 := (∃ n : ℕ, n > 0) ∧ (∀ r : ℚ, r > 0 → ∃ q : ℚ, q > 0 ∧ q < r)

-- Main theorem: Prove that exactly 3 statements are correct
theorem correct_statements : 
  (statement_1 ∧ statement_4 ∧ statement_6) ∧ 
  (¬ statement_2 ∧ ¬ statement_3 ∧ ¬ statement_5) :=
by
  sorry

end correct_statements_l1117_111750


namespace no_entangled_two_digit_numbers_l1117_111763

theorem no_entangled_two_digit_numbers :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 → 10 * a + b ≠ 2 * (a + b ^ 3) :=
by
  intros a b h
  rcases h with ⟨ha1, ha9, hb9⟩
  sorry

end no_entangled_two_digit_numbers_l1117_111763


namespace grandma_red_bacon_bits_l1117_111725

def mushrooms := 3
def cherry_tomatoes := 2 * mushrooms
def pickles := 4 * cherry_tomatoes
def bacon_bits := 4 * pickles
def red_bacon_bits := bacon_bits / 3

theorem grandma_red_bacon_bits : red_bacon_bits = 32 := by
  sorry

end grandma_red_bacon_bits_l1117_111725


namespace min_value_expression_l1117_111778

theorem min_value_expression (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 1 / (x - 2) ∧ y = 4 := 
sorry

end min_value_expression_l1117_111778


namespace SallyCarrots_l1117_111708

-- Definitions of the conditions
def FredGrew (F : ℕ) := F = 4
def TotalGrew (T : ℕ) := T = 10
def SallyGrew (S : ℕ) (F T : ℕ) := S + F = T

-- The theorem to be proved
theorem SallyCarrots : ∃ S : ℕ, FredGrew 4 ∧ TotalGrew 10 ∧ SallyGrew S 4 10 ∧ S = 6 :=
  sorry

end SallyCarrots_l1117_111708


namespace find_x_plus_y_l1117_111766

theorem find_x_plus_y (x y : ℝ) 
  (h1 : x + Real.cos y = 1004)
  (h2 : x + 1004 * Real.sin y = 1003)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 1003 :=
sorry

end find_x_plus_y_l1117_111766


namespace f_is_even_if_g_is_odd_l1117_111740

variable (g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) :=
  ∀ x : ℝ, h (-x) = -h x

def is_even (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = f x

theorem f_is_even_if_g_is_odd (hg : is_odd g) :
  is_even (fun x => |g (x^4)|) :=
by
  sorry

end f_is_even_if_g_is_odd_l1117_111740


namespace number_of_girls_on_playground_l1117_111713

theorem number_of_girls_on_playground (boys girls total : ℕ) 
  (h1 : boys = 44) (h2 : total = 97) (h3 : total = boys + girls) : 
  girls = 53 :=
by sorry

end number_of_girls_on_playground_l1117_111713


namespace greatest_integer_with_gcd_6_l1117_111784

theorem greatest_integer_with_gcd_6 (x : ℕ) :
  x < 150 ∧ gcd x 12 = 6 → x = 138 :=
by
  sorry

end greatest_integer_with_gcd_6_l1117_111784


namespace rectangle_area_l1117_111770

theorem rectangle_area (w l : ℕ) (h1 : l = 15) (h2 : (2 * l + 2 * w) / w = 5) : l * w = 150 :=
by
  -- We provide the conditions in the theorem's signature:
  -- l is the length which is 15 cm, given by h1
  -- The ratio of the perimeter to the width is 5:1, given by h2
  sorry

end rectangle_area_l1117_111770


namespace sin_210_l1117_111704

theorem sin_210 : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end sin_210_l1117_111704


namespace largest_measureable_quantity_is_1_l1117_111797

theorem largest_measureable_quantity_is_1 : 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd 496 403) 713) 824) 1171 = 1 :=
  sorry

end largest_measureable_quantity_is_1_l1117_111797


namespace triangle_ABC_proof_l1117_111752

noncomputable def sin2C_eq_sqrt3sinC (C : ℝ) : Prop := Real.sin (2 * C) = Real.sqrt 3 * Real.sin C

theorem triangle_ABC_proof (C a b c : ℝ) 
  (H1 : sin2C_eq_sqrt3sinC C) 
  (H2 : 0 < Real.sin C)
  (H3 : b = 6) 
  (H4 : a + b + c = 6*Real.sqrt 3 + 6) :
  (C = π/6) ∧ (1/2 * a * b * Real.sin C = 6*Real.sqrt 3) :=
sorry

end triangle_ABC_proof_l1117_111752
