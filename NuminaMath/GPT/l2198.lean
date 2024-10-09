import Mathlib

namespace g_49_l2198_219847

noncomputable def g : ℝ → ℝ := sorry

axiom g_func_eqn (x y : ℝ) : g (x^2 * y) = x * g y
axiom g_one_val : g 1 = 6

theorem g_49 : g 49 = 42 := by
  sorry

end g_49_l2198_219847


namespace main_theorem_l2198_219851

-- Define even functions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define odd functions
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : is_even_function f)
variable (h2 : is_odd_function g)
variable (h3 : ∀ x, g x = f (x - 1))

-- Theorem to prove
theorem main_theorem : f 2017 + f 2019 = 0 := sorry

end main_theorem_l2198_219851


namespace quilt_patch_cost_l2198_219879

-- Definitions of the conditions
def length : ℕ := 16
def width : ℕ := 20
def patch_area : ℕ := 4
def cost_first_10 : ℕ := 10
def cost_after_10 : ℕ := 5
def num_first_patches : ℕ := 10

-- Define the calculations based on the problem conditions
def quilt_area : ℕ := length * width
def total_patches : ℕ := quilt_area / patch_area
def cost_first : ℕ := num_first_patches * cost_first_10
def remaining_patches : ℕ := total_patches - num_first_patches
def cost_remaining : ℕ := remaining_patches * cost_after_10
def total_cost : ℕ := cost_first + cost_remaining

-- Statement of the proof problem
theorem quilt_patch_cost : total_cost = 450 := by
  -- Placeholder for the proof
  sorry

end quilt_patch_cost_l2198_219879


namespace initial_cookies_count_l2198_219890

theorem initial_cookies_count (x : ℕ) (h_ate : ℕ) (h_left : ℕ) :
  h_ate = 2 → h_left = 5 → (x - h_ate = h_left) → x = 7 :=
by
  intros
  sorry

end initial_cookies_count_l2198_219890


namespace cost_price_one_metre_l2198_219837

noncomputable def selling_price : ℤ := 18000
noncomputable def total_metres : ℕ := 600
noncomputable def loss_per_metre : ℤ := 5

noncomputable def total_loss : ℤ := loss_per_metre * (total_metres : ℤ) -- Note the cast to ℤ for multiplication
noncomputable def cost_price : ℤ := selling_price + total_loss
noncomputable def cost_price_per_metre : ℚ := cost_price / (total_metres : ℤ)

theorem cost_price_one_metre : cost_price_per_metre = 35 := by
  sorry

end cost_price_one_metre_l2198_219837


namespace count_f_compositions_l2198_219883

noncomputable def count_special_functions : Nat :=
  let A := Finset.range 6
  let f := (Set.univ : Set (A → A))
  sorry

theorem count_f_compositions (f : Fin 6 → Fin 6) 
  (h : ∀ x : Fin 6, (f ∘ f ∘ f) x = x) :
  count_special_functions = 81 :=
sorry

end count_f_compositions_l2198_219883


namespace product_increase_l2198_219850

variable (x : ℤ)

theorem product_increase (h : 53 * x = 1585) : 1585 - (35 * x) = 535 :=
by sorry

end product_increase_l2198_219850


namespace phi_eq_pi_div_two_l2198_219836

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.cos (x + ϕ)

theorem phi_eq_pi_div_two (ϕ : ℝ) (h1 : 0 ≤ ϕ) (h2 : ϕ ≤ π)
  (h3 : ∀ x : ℝ, f x ϕ = -f (-x) ϕ) : ϕ = π / 2 :=
sorry

end phi_eq_pi_div_two_l2198_219836


namespace find_x_value_l2198_219832

theorem find_x_value (a b x : ℤ) (h : a * b = (a - 1) * (b - 1)) (h2 : x * 9 = 160) :
  x = 21 :=
sorry

end find_x_value_l2198_219832


namespace triathlete_average_speed_is_approx_3_5_l2198_219831

noncomputable def triathlete_average_speed : ℝ :=
  let x : ℝ := 1; -- This represents the distance of biking/running segment
  let swimming_speed := 2; -- km/h
  let biking_speed := 25; -- km/h
  let running_speed := 12; -- km/h
  let swimming_distance := 2 * x; -- 2x km
  let biking_distance := x; -- x km
  let running_distance := x; -- x km
  let total_distance := swimming_distance + biking_distance + running_distance; -- 4x km
  let swimming_time := swimming_distance / swimming_speed; -- x hours
  let biking_time := biking_distance / biking_speed; -- x/25 hours
  let running_time := running_distance / running_speed; -- x/12 hours
  let total_time := swimming_time + biking_time + running_time; -- 1.12333x hours
  total_distance / total_time -- This should be the average speed

theorem triathlete_average_speed_is_approx_3_5 :
  abs (triathlete_average_speed - 3.5) < 0.1 := 
by
  sorry

end triathlete_average_speed_is_approx_3_5_l2198_219831


namespace housewife_spending_l2198_219860

theorem housewife_spending (P R M : ℝ) (h1 : R = 65) (h2 : R = 0.75 * P) (h3 : M / R - M / P = 5) :
  M = 1300 :=
by
  -- Proof steps will be added here.
  sorry

end housewife_spending_l2198_219860


namespace symmetric_line_equation_l2198_219853

theorem symmetric_line_equation :
  (∃ line : ℝ → ℝ, ∀ x y, x + 2 * y - 3 = 0 → line 1 = 1 ∧ (∃ b, line 0 = b → x - 2 * y + 1 = 0)) :=
sorry

end symmetric_line_equation_l2198_219853


namespace find_range_of_values_l2198_219880

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem find_range_of_values (f : ℝ → ℝ) (h_even : is_even f)
  (h_increasing : is_increasing_on_nonneg f) (h_f1_zero : f 1 = 0) :
  { x : ℝ | f (Real.log x / Real.log (1/2)) > 0 } = 
  { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | x > 2 } :=
by 
  sorry

end find_range_of_values_l2198_219880


namespace basketball_game_first_half_points_l2198_219820

theorem basketball_game_first_half_points (a b r d : ℕ) (H1 : a = b)
  (H2 : a * (1 + r + r^2 + r^3) = 4 * a + 6 * d + 1) 
  (H3 : 15 * a ≤ 100) (H4 : b + (b + d) + b + 2 * d + b + 3 * d < 100) : 
  (a + a * r + b + b + d) = 34 :=
by sorry

end basketball_game_first_half_points_l2198_219820


namespace number_of_students_surveyed_l2198_219852

noncomputable def M : ℕ := 60
noncomputable def N : ℕ := 90
noncomputable def B : ℕ := M / 3

theorem number_of_students_surveyed : M + B + N = 170 := by
  rw [M, N, B]
  norm_num
  sorry

end number_of_students_surveyed_l2198_219852


namespace total_distance_traveled_is_correct_l2198_219865

-- Definitions of given conditions
def Vm : ℕ := 8
def Vr : ℕ := 2
def round_trip_time : ℝ := 1

-- Definitions needed for intermediate calculations (speed computations)
def upstream_speed (Vm Vr : ℕ) : ℕ := Vm - Vr
def downstream_speed (Vm Vr : ℕ) : ℕ := Vm + Vr

-- The equation representing the total time for the round trip
def time_equation (D : ℝ) (Vm Vr : ℕ) : Prop :=
  D / upstream_speed Vm Vr + D / downstream_speed Vm Vr = round_trip_time

-- Prove that the total distance traveled by the man is 7.5 km
theorem total_distance_traveled_is_correct : ∃ D : ℝ, D / upstream_speed Vm Vr + D / downstream_speed Vm Vr = round_trip_time ∧ 2 * D = 7.5 :=
by
  sorry

end total_distance_traveled_is_correct_l2198_219865


namespace f_periodic_if_is_bounded_and_satisfies_fe_l2198_219875

variable {f : ℝ → ℝ}

-- Condition 1: f is a bounded real function, i.e., it is bounded above and below
def is_bounded (f : ℝ → ℝ) : Prop := ∃ M, ∀ x, |f x| ≤ M

-- Condition 2: The functional equation given for all x.
def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1/3) + f (x + 1/2) = f x + f (x + 5/6)

-- We need to show that f is periodic with period 1.
theorem f_periodic_if_is_bounded_and_satisfies_fe (h_bounded : is_bounded f) (h_fe : functional_eq f) : 
  ∀ x, f (x + 1) = f x :=
sorry

end f_periodic_if_is_bounded_and_satisfies_fe_l2198_219875


namespace solveForN_l2198_219829

-- Define the condition that sqrt(8 + n) = 9
def condition (n : ℝ) : Prop := Real.sqrt (8 + n) = 9

-- State the main theorem that given the condition, n must be 73
theorem solveForN (n : ℝ) (h : condition n) : n = 73 := by
  sorry

end solveForN_l2198_219829


namespace goat_age_l2198_219810

theorem goat_age : 26 + 42 = 68 := 
by 
  -- Since we only need the statement,
  -- we add sorry to skip the proof.
  sorry

end goat_age_l2198_219810


namespace rectangle_width_l2198_219870

theorem rectangle_width (w : ℝ) 
  (h1 : ∃ w : ℝ, w > 0 ∧ (2 * w + 2 * (w - 2)) = 16) 
  (h2 : ∀ w, w > 0 → 2 * w + 2 * (w - 2) = 16 → w = 5) : 
  w = 5 := 
sorry

end rectangle_width_l2198_219870


namespace num_right_triangles_with_incenter_origin_l2198_219834

theorem num_right_triangles_with_incenter_origin (p : ℕ) (hp : Nat.Prime p) :
  let M : ℤ × ℤ := (p * 1994, 7 * p * 1994)
  let is_lattice_point (x : ℤ × ℤ) : Prop := True  -- All points considered are lattice points
  let is_right_angle_vertex (M : ℤ × ℤ) : Prop := True
  let is_incenter_origin (M : ℤ × ℤ) : Prop := True
  let num_triangles (p : ℕ) : ℕ :=
    if p = 2 then 18
    else if p = 997 then 20
    else 36
  num_triangles p = if p = 2 then 18 else if p = 997 then 20 else 36 := (

  by sorry

 )

end num_right_triangles_with_incenter_origin_l2198_219834


namespace solve_for_x_l2198_219881

theorem solve_for_x (x : ℚ) : x^2 + 125 = (x - 15)^2 → x = 10 / 3 := by
  sorry

end solve_for_x_l2198_219881


namespace evaluate_expression_l2198_219815

noncomputable def x : ℚ := 4 / 7
noncomputable def y : ℚ := 6 / 8

theorem evaluate_expression : (7 * x + 8 * y) / (56 * x * y) = 5 / 12 := by
  sorry

end evaluate_expression_l2198_219815


namespace jonah_poured_total_pitchers_l2198_219895

theorem jonah_poured_total_pitchers :
  (0.25 + 0.125) + (0.16666666666666666 + 0.08333333333333333 + 0.16666666666666666) + 
  (0.25 + 0.125) + (0.3333333333333333 + 0.08333333333333333 + 0.16666666666666666) = 1.75 :=
by
  sorry

end jonah_poured_total_pitchers_l2198_219895


namespace non_negative_real_sum_expressions_l2198_219823

theorem non_negative_real_sum_expressions (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) (h_sum : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end non_negative_real_sum_expressions_l2198_219823


namespace intersection_points_l2198_219844

theorem intersection_points (a : ℝ) (h : 2 < a) :
  (∃ n : ℕ, (n = 1 ∨ n = 2) ∧ (∃ x1 x2 : ℝ, y = (a-3)*x^2 - x - 1/4 ∧ x1 ≠ x2)) :=
sorry

end intersection_points_l2198_219844


namespace central_angle_radian_l2198_219869

-- Define the context of the sector and conditions
def sector (r θ : ℝ) :=
  θ = r * 6 ∧ 1/2 * r^2 * θ = 6

-- Define the radian measure of the central angle
theorem central_angle_radian (r : ℝ) (θ : ℝ) (h : sector r θ) : θ = 3 :=
by
  sorry

end central_angle_radian_l2198_219869


namespace five_students_in_a_row_five_students_with_constraints_five_students_into_three_classes_l2198_219862

-- Definition: Number of ways to arrange n items in a row
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Question (1)
theorem five_students_in_a_row : factorial 5 = 120 :=
by sorry

-- Question (2) - Rather than performing combinatorial steps directly, we'll assume a function to calculate the specific arrangement
def specific_arrangement (students: ℕ) : ℕ :=
  if students = 5 then 24 else 0

theorem five_students_with_constraints : specific_arrangement 5 = 24 :=
by sorry

-- Question (3) - Number of ways to divide n students into k classes with at least one student in each class
def number_of_ways_to_divide (students: ℕ) (classes: ℕ) : ℕ :=
  if students = 5 ∧ classes = 3 then 150 else 0

theorem five_students_into_three_classes : number_of_ways_to_divide 5 3 = 150 :=
by sorry

end five_students_in_a_row_five_students_with_constraints_five_students_into_three_classes_l2198_219862


namespace well_defined_interval_l2198_219874

def is_well_defined (x : ℝ) : Prop :=
  (5 - x > 0) ∧ (x ≠ 2)

theorem well_defined_interval : 
  ∀ x : ℝ, (is_well_defined x) ↔ (x < 5 ∧ x ≠ 2) :=
by 
  sorry

end well_defined_interval_l2198_219874


namespace bird_watcher_total_l2198_219861

theorem bird_watcher_total
  (M : ℕ) (T : ℕ) (W : ℕ)
  (h1 : M = 70)
  (h2 : T = M / 2)
  (h3 : W = T + 8) :
  M + T + W = 148 :=
by
  -- proof omitted
  sorry

end bird_watcher_total_l2198_219861


namespace sin_zero_necessary_not_sufficient_l2198_219814

theorem sin_zero_necessary_not_sufficient:
  (∀ α : ℝ, (∃ k : ℤ, α = 2 * k * Real.pi) → (Real.sin α = 0)) ∧
  ¬ (∀ α : ℝ, (Real.sin α = 0) → (∃ k : ℤ, α = 2 * k * Real.pi)) :=
by
  sorry

end sin_zero_necessary_not_sufficient_l2198_219814


namespace solution_set_f_ge_0_l2198_219893

variables {f : ℝ → ℝ}

-- Conditions
axiom h1 : ∀ x : ℝ, f (-x) = -f x  -- f is odd function
axiom h2 : ∀ x y : ℝ, 0 < x → x < y → f x < f y  -- f is monotonically increasing on (0, +∞)
axiom h3 : f 3 = 0  -- f(3) = 0

theorem solution_set_f_ge_0 : { x : ℝ | f x ≥ 0 } = { x : ℝ | -3 ≤ x ∧ x ≤ 0 } ∪ { x : ℝ | 3 ≤ x } :=
by
  sorry

end solution_set_f_ge_0_l2198_219893


namespace total_number_of_fish_l2198_219805

theorem total_number_of_fish :
  let goldfish := 8
  let angelfish := goldfish + 4
  let guppies := 2 * angelfish
  let tetras := goldfish - 3
  let bettas := tetras + 5
  goldfish + angelfish + guppies + tetras + bettas = 59 := by
  -- Provide the proof here.
  sorry

end total_number_of_fish_l2198_219805


namespace probability_of_rolling_number_less_than_5_is_correct_l2198_219825

noncomputable def probability_of_rolling_number_less_than_5 : ℚ :=
  let total_outcomes := 8
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_of_rolling_number_less_than_5_is_correct :
  probability_of_rolling_number_less_than_5 = 1 / 2 := by
  sorry

end probability_of_rolling_number_less_than_5_is_correct_l2198_219825


namespace actual_time_is_1240pm_l2198_219857

def kitchen_and_cellphone_start (t : ℕ) : Prop := t = 8 * 60  -- 8:00 AM in minutes
def kitchen_clock_after_breakfast (t : ℕ) : Prop := t = 8 * 60 + 30  -- 8:30 AM in minutes
def cellphone_after_breakfast (t : ℕ) : Prop := t = 8 * 60 + 20  -- 8:20 AM in minutes
def kitchen_clock_at_3pm (t : ℕ) : Prop := t = 15 * 60  -- 3:00 PM in minutes

theorem actual_time_is_1240pm : 
  (kitchen_and_cellphone_start 480) ∧ 
  (kitchen_clock_after_breakfast 510) ∧ 
  (cellphone_after_breakfast 500) ∧
  (kitchen_clock_at_3pm 900) → 
  real_time_at_kitchen_clock_time_3pm = 12 * 60 + 40 :=
by
  sorry

end actual_time_is_1240pm_l2198_219857


namespace remainder_when_sum_divided_by_40_l2198_219855

theorem remainder_when_sum_divided_by_40 (x y : ℤ) 
  (h1 : x % 80 = 75) 
  (h2 : y % 120 = 115) : 
  (x + y) % 40 = 30 := 
  sorry

end remainder_when_sum_divided_by_40_l2198_219855


namespace cups_of_sugar_l2198_219871

theorem cups_of_sugar (flour_total flour_added sugar : ℕ) (h₁ : flour_total = 10) (h₂ : flour_added = 7) (h₃ : flour_total - flour_added = sugar + 1) :
  sugar = 2 :=
by
  sorry

end cups_of_sugar_l2198_219871


namespace pq_work_together_in_10_days_l2198_219838

theorem pq_work_together_in_10_days 
  (p q r : ℝ)
  (hq : 1/q = 1/28)
  (hr : 1/r = 1/35)
  (hp : 1/p = 1/q + 1/r) :
  1/p + 1/q = 1/10 :=
by sorry

end pq_work_together_in_10_days_l2198_219838


namespace radius_distance_relation_l2198_219802

variables {A B C : Point} (Γ₁ Γ₂ ω₀ : Circle)
variables (ω : ℕ → Circle)
variables (r d : ℕ → ℝ)

def diam_circle (P Q : Point) : Circle := sorry  -- This is to define a circle with diameter PQ
def tangent (κ κ' κ'' : Circle) : Prop := sorry  -- This is to define that three circles are mutually tangent

-- Defining the properties as given in the conditions
axiom Γ₁_def : Γ₁ = diam_circle A B
axiom Γ₂_def : Γ₂ = diam_circle A C
axiom ω₀_def : ω₀ = diam_circle B C
axiom ω_def : ∀ n : ℕ, tangent (if n = 0 then ω₀ else ω (n - 1)) Γ₁ (ω n) ∧ tangent (if n = 0 then ω₀ else ω (n - 1)) Γ₂ (ω n) -- ωₙ is tangent to previous circle, Γ₁ and Γ₂

-- The main proof statement
theorem radius_distance_relation (n : ℕ) : r n = 2 * n * d n :=
sorry

end radius_distance_relation_l2198_219802


namespace national_education_fund_expenditure_l2198_219888

theorem national_education_fund_expenditure (gdp_2012 : ℝ) (h : gdp_2012 = 43.5 * 10^12) : 
  (0.04 * gdp_2012) = 1.74 * 10^13 := 
by sorry

end national_education_fund_expenditure_l2198_219888


namespace inverse_proportional_k_value_l2198_219801

theorem inverse_proportional_k_value (k : ℝ) :
  (∃ x y : ℝ, y = k / x ∧ x = - (Real.sqrt 2) / 2 ∧ y = Real.sqrt 2) → 
  k = -1 :=
by
  sorry

end inverse_proportional_k_value_l2198_219801


namespace value_of_x_l2198_219806

theorem value_of_x (x : ℝ) (h : ∃ k < 0, (x, 1) = k • (4, x)) : x = -2 :=
sorry

end value_of_x_l2198_219806


namespace find_fraction_l2198_219807

def number : ℕ := 16

theorem find_fraction (f : ℚ) : f * number + 5 = 13 → f = 1 / 2 :=
by
  sorry

end find_fraction_l2198_219807


namespace find_constant_C_l2198_219804

def polynomial_remainder (C : ℝ) (x : ℝ) : ℝ :=
  C * x^3 - 3 * x^2 + x - 1

theorem find_constant_C :
  (polynomial_remainder 2 (-1) = -7) → 2 = 2 :=
by
  sorry

end find_constant_C_l2198_219804


namespace sufficient_condition_l2198_219833

theorem sufficient_condition (a b : ℝ) (h1 : a > 1) (h2 : b > 1) : ab > 1 :=
sorry

end sufficient_condition_l2198_219833


namespace find_star_1993_1935_l2198_219824

axiom star (x y : ℕ) : ℕ
axiom star_idempotent (x : ℕ) : star x x = 0
axiom star_assoc (x y z : ℕ) : star x (star y z) = star x y + z

theorem find_star_1993_1935 : star 1993 1935 = 58 :=
by
  sorry

end find_star_1993_1935_l2198_219824


namespace find_distance_CD_l2198_219854

-- Define the ellipse and the required points
def ellipse (x y : ℝ) : Prop := 16 * (x-3)^2 + 4 * (y+2)^2 = 64

-- Define the center and the semi-axes lengths
noncomputable def center : (ℝ × ℝ) := (3, -2)
noncomputable def semi_major_axis_length : ℝ := 4
noncomputable def semi_minor_axis_length : ℝ := 2

-- Define the points C and D on the ellipse
def point_C (x y : ℝ) : Prop := ellipse x y ∧ (x = 3 + semi_major_axis_length ∨ x = 3 - semi_major_axis_length) ∧ y = -2
def point_D (x y : ℝ) : Prop := ellipse x y ∧ x = 3 ∧ (y = -2 + semi_minor_axis_length ∨ y = -2 - semi_minor_axis_length)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Main theorem to prove
theorem find_distance_CD : 
  ∃ C D : ℝ × ℝ, 
    (point_C C.1 C.2 ∧ point_D D.1 D.2) → 
    distance C D = 2 * Real.sqrt 5 := 
sorry

end find_distance_CD_l2198_219854


namespace intersection_of_A_and_B_is_B_implies_m_leq_4_over_3_l2198_219830

noncomputable def f (x : ℝ) : ℝ := (1 / (Real.sqrt (x + 2))) + Real.log (3 - x)
def A : Set ℝ := { x | -2 < x ∧ x < 3 }
def B (m : ℝ) : Set ℝ := { x | 1 - m < x ∧ x < 3 * m - 1 }

theorem intersection_of_A_and_B_is_B_implies_m_leq_4_over_3 (m : ℝ) 
    (h : A ∩ B m = B m) : m ≤ 4 / 3 := by
  sorry

end intersection_of_A_and_B_is_B_implies_m_leq_4_over_3_l2198_219830


namespace problem_solution_l2198_219898

theorem problem_solution (n : Real) (h : 0.04 * n + 0.1 * (30 + n) = 15.2) : n = 89.09 := 
sorry

end problem_solution_l2198_219898


namespace digit_place_value_ratio_l2198_219817

theorem digit_place_value_ratio :
  let number := 86304.2957
  let digit_6_value := 1000
  let digit_5_value := 0.1
  digit_6_value / digit_5_value = 10000 :=
by
  let number := 86304.2957
  let digit_6_value := 1000
  let digit_5_value := 0.1
  sorry

end digit_place_value_ratio_l2198_219817


namespace div_1947_l2198_219867

theorem div_1947 (n : ℕ) (hn : n % 2 = 1) : 1947 ∣ (46^n + 296 * 13^n) :=
by
  sorry

end div_1947_l2198_219867


namespace soccer_team_starters_l2198_219809

open Nat

-- Definitions representing the conditions
def total_players : ℕ := 18
def twins_included : ℕ := 2
def remaining_players : ℕ := total_players - twins_included
def starters_to_choose : ℕ := 7 - twins_included

-- Theorem statement to assert the solution
theorem soccer_team_starters :
  Nat.choose remaining_players starters_to_choose = 4368 :=
by
  -- Placeholder for proof
  sorry

end soccer_team_starters_l2198_219809


namespace unique_solution_t_interval_l2198_219873

theorem unique_solution_t_interval (x y z v t : ℝ) :
  (x + y + z + v = 0) →
  ((x * y + y * z + z * v) + t * (x * z + x * v + y * v) = 0) →
  (t > (3 - Real.sqrt 5) / 2) ∧ (t < (3 + Real.sqrt 5) / 2) :=
by
  intro h1 h2
  sorry

end unique_solution_t_interval_l2198_219873


namespace tangent_line_eq_l2198_219892

noncomputable def f (x : ℝ) := x / (2 * x - 1)

def tangentLineAtPoint (x : ℝ) : ℝ := -x + 2

theorem tangent_line_eq {x y : ℝ} (hxy : y = f 1) (f_deriv : deriv f 1 = -1) :
  y = 1 → tangentLineAtPoint x = -x + 2 :=
by
  intros
  sorry

end tangent_line_eq_l2198_219892


namespace bottom_left_square_side_length_l2198_219889

theorem bottom_left_square_side_length (x y : ℕ) 
  (h1 : 1 + (x - 1) = 1) 
  (h2 : 2 * x - 1 = (x - 2) + (x - 3) + y) :
  y = 4 :=
sorry

end bottom_left_square_side_length_l2198_219889


namespace longest_side_of_similar_triangle_l2198_219811

theorem longest_side_of_similar_triangle :
  ∀ (x : ℝ),
    let a := 8
    let b := 10
    let c := 12
    let s₁ := a * x
    let s₂ := b * x
    let s₃ := c * x
    a + b + c = 30 → 
    30 * x = 150 → 
    s₁ > 30 → 
    max s₁ (max s₂ s₃) = 60 :=
by
  intros x a b c s₁ s₂ s₃ h₁ h₂ h₃
  sorry

end longest_side_of_similar_triangle_l2198_219811


namespace number_of_boys_l2198_219841

theorem number_of_boys (x g : ℕ) 
  (h1 : x + g = 150) 
  (h2 : g = (x * 150) / 100) 
  : x = 60 := 
by 
  sorry

end number_of_boys_l2198_219841


namespace cos_alpha_is_negative_four_fifths_l2198_219887

variable (α : ℝ)
variable (H1 : Real.sin α = 3 / 5)
variable (H2 : π / 2 < α ∧ α < π)

theorem cos_alpha_is_negative_four_fifths (H1 : Real.sin α = 3 / 5) (H2 : π / 2 < α ∧ α < π) :
  Real.cos α = -4 / 5 :=
sorry

end cos_alpha_is_negative_four_fifths_l2198_219887


namespace pugs_cleaning_time_l2198_219818

theorem pugs_cleaning_time : 
  (∀ (p t: ℕ), 15 * 12 = p * t ↔ 15 * 12 = 4 * 45) :=
by
  sorry

end pugs_cleaning_time_l2198_219818


namespace transformation_thinking_reflected_in_solution_of_quadratic_l2198_219868

theorem transformation_thinking_reflected_in_solution_of_quadratic :
  ∀ (x : ℝ), (x - 3)^2 - 5 * (x - 3) = 0 → (x = 3 ∨ x = 8) →
  transformation_thinking :=
by
  intros x h_eq h_solutions
  sorry

end transformation_thinking_reflected_in_solution_of_quadratic_l2198_219868


namespace find_number_l2198_219858

theorem find_number (x : ℝ) (h : x / 0.025 = 40) : x = 1 := 
by sorry

end find_number_l2198_219858


namespace weight_range_correct_l2198_219896

noncomputable def combined_weight : ℕ := 158
noncomputable def tracy_weight : ℕ := 52
noncomputable def jake_weight : ℕ := tracy_weight + 8
noncomputable def john_weight : ℕ := combined_weight - (tracy_weight + jake_weight)
noncomputable def weight_range : ℕ := jake_weight - john_weight

theorem weight_range_correct : weight_range = 14 := 
by
  sorry

end weight_range_correct_l2198_219896


namespace total_oranges_l2198_219812

theorem total_oranges :
  let capacity_box1 := 80
  let capacity_box2 := 50
  let fullness_box1 := (3/4 : ℚ)
  let fullness_box2 := (3/5 : ℚ)
  let oranges_box1 := fullness_box1 * capacity_box1
  let oranges_box2 := fullness_box2 * capacity_box2
  oranges_box1 + oranges_box2 = 90 := 
by
  sorry

end total_oranges_l2198_219812


namespace sequence_properties_l2198_219877

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 3 - 2^n

-- Prove the statements
theorem sequence_properties (n : ℕ) :
  (a (2 * n) = 3 - 4^n) ∧ (a 2 / a 3 = 1 / 5) :=
by
  sorry

end sequence_properties_l2198_219877


namespace fraction_combination_l2198_219813

theorem fraction_combination (x y : ℝ) (h : y / x = 3 / 4) : (x + y) / x = 7 / 4 :=
by
  -- Proof steps will be inserted here (for now using sorry)
  sorry

end fraction_combination_l2198_219813


namespace cakes_served_yesterday_l2198_219843

theorem cakes_served_yesterday (cakes_today_lunch : ℕ) (cakes_today_dinner : ℕ) (total_cakes : ℕ)
  (h1 : cakes_today_lunch = 5) (h2 : cakes_today_dinner = 6) (h3 : total_cakes = 14) :
  total_cakes - (cakes_today_lunch + cakes_today_dinner) = 3 :=
by
  -- Import necessary libraries
  sorry

end cakes_served_yesterday_l2198_219843


namespace system_of_equations_a_solution_l2198_219846

theorem system_of_equations_a_solution (x y a : ℝ) (h1 : 4 * x + y = a) (h2 : 3 * x + 4 * y^2 = 3 * a) (hx : x = 3) : a = 15 ∨ a = 9.75 :=
by
  sorry

end system_of_equations_a_solution_l2198_219846


namespace dean_marathon_time_l2198_219839

/-- 
Micah runs 2/3 times as fast as Dean, and it takes Jake 1/3 times more time to finish the marathon
than it takes Micah. The total time the three take to complete the marathon is 23 hours.
Prove that the time it takes Dean to finish the marathon is approximately 7.67 hours.
-/
theorem dean_marathon_time (D M J : ℝ)
  (h1 : M = D * (3 / 2))
  (h2 : J = M + (1 / 3) * M)
  (h3 : D + M + J = 23) : 
  D = 23 / 3 :=
by
  sorry

end dean_marathon_time_l2198_219839


namespace arithmetic_progression_common_difference_l2198_219885

theorem arithmetic_progression_common_difference 
  (x y : ℤ) 
  (h1 : 280 * x^2 - 61 * x * y + 3 * y^2 - 13 = 0) 
  (h2 : ∃ a d : ℤ, x = a + 3 * d ∧ y = a + 8 * d) : 
  ∃ d : ℤ, d = -5 := 
sorry

end arithmetic_progression_common_difference_l2198_219885


namespace part_I_l2198_219803

variable (a b c n p q : ℝ)

theorem part_I (hne0 : a ≠ 0) (bne0 : b ≠ 0) (cne0 : c ≠ 0)
    (h1 : a^2 + b^2 + c^2 = 2) (h2 : n^2 + p^2 + q^2 = 2) :
    (n^4 / a^2 + p^4 / b^2 + q^4 / c^2) ≥ 2 := 
sorry

end part_I_l2198_219803


namespace initial_weight_of_solution_Y_is_8_l2198_219842

theorem initial_weight_of_solution_Y_is_8
  (W : ℝ)
  (hw1 : 0.25 * W = 0.20 * W + 0.4)
  (hw2 : W ≠ 0) : W = 8 :=
by
  sorry

end initial_weight_of_solution_Y_is_8_l2198_219842


namespace arithmetic_sequence_8th_term_l2198_219821

theorem arithmetic_sequence_8th_term 
  (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 41) : 
  a + 7 * d = 59 := 
by 
  sorry

end arithmetic_sequence_8th_term_l2198_219821


namespace tangent_line_equation_is_correct_l2198_219849

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_equation_is_correct :
  let p : ℝ × ℝ := (0, 1)
  let f' := fun x => x * Real.exp x + Real.exp x
  let slope := f' 0
  let tangent_line := fun x y => slope * (x - p.1) - (y - p.2)
  tangent_line = (fun x y => x - y + 1) :=
by
  intros
  sorry

end tangent_line_equation_is_correct_l2198_219849


namespace cos_B_and_area_of_triangle_l2198_219886

theorem cos_B_and_area_of_triangle (A B C : ℝ) (a b c : ℝ)
  (h_sin_A : Real.sin A = Real.sin (2 * B))
  (h_a : a = 4) (h_b : b = 6) :
  Real.cos B = 1 / 3 ∧ ∃ (area : ℝ), area = 8 * Real.sqrt 2 :=
by
  sorry  -- Proof goes here

end cos_B_and_area_of_triangle_l2198_219886


namespace sum_of_b_for_unique_solution_l2198_219840

theorem sum_of_b_for_unique_solution :
  (∃ b1 b2, (3 * (0:ℝ)^2 + (b1 + 6) * 0 + 7 = 0 ∧ 3 * (0:ℝ)^2 + (b2 + 6) * 0 + 7 = 0) ∧ 
   ((b1 + 6)^2 - 4 * 3 * 7 = 0) ∧ ((b2 + 6)^2 - 4 * 3 * 7 = 0) ∧ 
   b1 + b2 = -12)  :=
by
  sorry

end sum_of_b_for_unique_solution_l2198_219840


namespace rancher_problem_l2198_219864

theorem rancher_problem (s c : ℕ) (h : 30 * s + 35 * c = 1500) : (s = 1 ∧ c = 42) ∨ (s = 36 ∧ c = 12) := 
by
  sorry

end rancher_problem_l2198_219864


namespace min_value_expression_l2198_219859

theorem min_value_expression (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) : 
  ∃(x : ℝ), x ≤ (a - b) * (b - c) * (c - d) * (d - a) ∧ x = -1/8 :=
sorry

end min_value_expression_l2198_219859


namespace clara_boxes_l2198_219884

theorem clara_boxes (x : ℕ)
  (h1 : 12 * x + 20 * 80 + 16 * 70 = 3320) : x = 50 := by
  sorry

end clara_boxes_l2198_219884


namespace hannah_quarters_l2198_219882

theorem hannah_quarters :
  ∃ n : ℕ, 40 < n ∧ n < 400 ∧
  n % 6 = 3 ∧ n % 7 = 3 ∧ n % 8 = 3 ∧ 
  (n = 171 ∨ n = 339) :=
by
  sorry

end hannah_quarters_l2198_219882


namespace circle_constant_ratio_l2198_219819

theorem circle_constant_ratio (b : ℝ) :
  (∀ (x y : ℝ), (x + 4)^2 + (y + b)^2 = 16 → 
    ∃ k : ℝ, 
      ∀ P : ℝ × ℝ, 
        P = (x, y) → 
        dist P (-2, 0) / dist P (4, 0) = k)
  → b = 0 :=
by
  intros h
  sorry

end circle_constant_ratio_l2198_219819


namespace binary_sum_eq_669_l2198_219856

def binary111111111 : ℕ := 511
def binary1111111 : ℕ := 127
def binary11111 : ℕ := 31

theorem binary_sum_eq_669 :
  binary111111111 + binary1111111 + binary11111 = 669 :=
by
  sorry

end binary_sum_eq_669_l2198_219856


namespace ocean_depth_at_base_of_cone_l2198_219891

noncomputable def cone_volume (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

noncomputable def submerged_height_fraction (total_height volume_fraction : ℝ) : ℝ :=
  total_height * (volume_fraction)^(1/3)

theorem ocean_depth_at_base_of_cone (total_height radius : ℝ) 
  (above_water_volume_fraction : ℝ) : ℝ :=
  let above_water_height := submerged_height_fraction total_height above_water_volume_fraction
  total_height - above_water_height

example : ocean_depth_at_base_of_cone 10000 2000 (3 / 5) = 1566 := by
  sorry

end ocean_depth_at_base_of_cone_l2198_219891


namespace satisfies_differential_equation_l2198_219827

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) / x

theorem satisfies_differential_equation (x : ℝ) (hx : x ≠ 0) : 
  x * (deriv (fun x => (Real.sin x) / x) x) + (Real.sin x) / x = Real.cos x := 
by
  -- the proof goes here
  sorry

end satisfies_differential_equation_l2198_219827


namespace trigonometric_identity_l2198_219866

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 1 / 2) : 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 :=
by
  sorry

end trigonometric_identity_l2198_219866


namespace sum_of_possible_N_values_l2198_219872

theorem sum_of_possible_N_values (a b c N : ℕ) (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = 2 * (a + b)) :
  ∃ sum_N : ℕ, sum_N = 672 :=
by
  sorry

end sum_of_possible_N_values_l2198_219872


namespace min_garden_cost_l2198_219863

theorem min_garden_cost : 
  let flower_cost (flower : String) : Real :=
    if flower = "Asters" then 1 else
    if flower = "Begonias" then 2 else
    if flower = "Cannas" then 2 else
    if flower = "Dahlias" then 3 else
    if flower = "Easter lilies" then 2.5 else
    0
  let region_area (region : String) : Nat :=
    if region = "Bottom left" then 10 else
    if region = "Top left" then 9 else
    if region = "Bottom right" then 20 else
    if region = "Top middle" then 2 else
    if region = "Top right" then 7 else
    0
  let min_cost : Real :=
    (flower_cost "Dahlias" * region_area "Top middle") + 
    (flower_cost "Easter lilies" * region_area "Top right") + 
    (flower_cost "Cannas" * region_area "Top left") + 
    (flower_cost "Begonias" * region_area "Bottom left") + 
    (flower_cost "Asters" * region_area "Bottom right")
  min_cost = 81.5 :=
by
  sorry

end min_garden_cost_l2198_219863


namespace john_weekly_earnings_l2198_219897

/-- John takes 3 days off of streaming per week. 
    John streams for 4 hours at a time on the days he does stream.
    John makes $10 an hour.
    Prove that John makes $160 a week. -/

theorem john_weekly_earnings (days_off : ℕ) (hours_per_day : ℕ) (wage_per_hour : ℕ) 
  (h_days_off : days_off = 3) (h_hours_per_day : hours_per_day = 4) 
  (h_wage_per_hour : wage_per_hour = 10) : 
  7 - days_off * hours_per_day * wage_per_hour = 160 := by
  sorry

end john_weekly_earnings_l2198_219897


namespace delta_value_l2198_219899

theorem delta_value (Δ : ℤ) (h : 4 * (-3) = Δ - 3) : Δ = -9 :=
by {
  sorry
}

end delta_value_l2198_219899


namespace find_m_l2198_219848

variable (m x1 x2 : ℝ)

def quadratic_eqn (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m * x + 2 * m - 1 = 0

def roots_condition (m x1 x2 : ℝ) : Prop :=
  x1^2 + x2^2 = 23 ∧
  x1 + x2 = m ∧
  x1 * x2 = 2 * m - 1

theorem find_m (m x1 x2 : ℝ) : 
  quadratic_eqn m → 
  roots_condition m x1 x2 → 
  m = -3 :=
by
  intro hQ hR
  sorry

end find_m_l2198_219848


namespace probability_inside_octahedron_l2198_219828

noncomputable def probability_of_octahedron : ℝ := 
  let cube_volume := 8
  let octahedron_volume := 4 / 3
  octahedron_volume / cube_volume

theorem probability_inside_octahedron :
  probability_of_octahedron = 1 / 6 :=
  by
    sorry

end probability_inside_octahedron_l2198_219828


namespace solve_for_a_l2198_219878
-- Additional imports might be necessary depending on specifics of the proof

theorem solve_for_a (a x y : ℝ) (h1 : ax - y = 3) (h2 : x = 1) (h3 : y = 2) : a = 5 :=
by
  sorry

end solve_for_a_l2198_219878


namespace cinema_meeting_day_l2198_219835

-- Define the cycles for Kolya, Seryozha, and Vanya.
def kolya_cycle : ℕ := 4
def seryozha_cycle : ℕ := 5
def vanya_cycle : ℕ := 6

-- The problem statement requiring proof.
theorem cinema_meeting_day : ∃ n : ℕ, n > 0 ∧ n % kolya_cycle = 0 ∧ n % seryozha_cycle = 0 ∧ n % vanya_cycle = 0 ∧ n = 60 := 
  sorry

end cinema_meeting_day_l2198_219835


namespace solve_for_x_l2198_219808

theorem solve_for_x (x : ℝ) :
    (1 / 3 * ((x + 8) + (7 * x + 3) + (3 * x + 9)) = 5 * x - 10) → x = 12.5 :=
by
  intro h
  sorry

end solve_for_x_l2198_219808


namespace sarah_speed_for_rest_of_trip_l2198_219826

def initial_speed : ℝ := 15  -- miles per hour
def initial_time : ℝ := 1  -- hour
def total_distance : ℝ := 45  -- miles
def extra_time_if_same_speed : ℝ := 1  -- hour (late)
def arrival_early_time : ℝ := 0.5  -- hour (early)

theorem sarah_speed_for_rest_of_trip (remaining_distance remaining_time : ℝ) :
  remaining_distance = total_distance - initial_speed * initial_time →
  remaining_time = (remaining_distance / initial_speed - extra_time_if_same_speed) + arrival_early_time →
  remaining_distance / remaining_time = 20 :=
by
  intros h1 h2
  sorry

end sarah_speed_for_rest_of_trip_l2198_219826


namespace cost_per_ice_cream_l2198_219822

theorem cost_per_ice_cream (chapati_count : ℕ)
                           (rice_plate_count : ℕ)
                           (mixed_vegetable_plate_count : ℕ)
                           (ice_cream_cup_count : ℕ)
                           (cost_per_chapati : ℕ)
                           (cost_per_rice_plate : ℕ)
                           (cost_per_mixed_vegetable : ℕ)
                           (amount_paid : ℕ)
                           (total_cost_chapatis : ℕ)
                           (total_cost_rice : ℕ)
                           (total_cost_mixed_vegetable : ℕ)
                           (total_non_ice_cream_cost : ℕ)
                           (total_ice_cream_cost : ℕ)
                           (cost_per_ice_cream_cup : ℕ) :
    chapati_count = 16 →
    rice_plate_count = 5 →
    mixed_vegetable_plate_count = 7 →
    ice_cream_cup_count = 6 →
    cost_per_chapati = 6 →
    cost_per_rice_plate = 45 →
    cost_per_mixed_vegetable = 70 →
    amount_paid = 961 →
    total_cost_chapatis = chapati_count * cost_per_chapati →
    total_cost_rice = rice_plate_count * cost_per_rice_plate →
    total_cost_mixed_vegetable = mixed_vegetable_plate_count * cost_per_mixed_vegetable →
    total_non_ice_cream_cost = total_cost_chapatis + total_cost_rice + total_cost_mixed_vegetable →
    total_ice_cream_cost = amount_paid - total_non_ice_cream_cost →
    cost_per_ice_cream_cup = total_ice_cream_cost / ice_cream_cup_count →
    cost_per_ice_cream_cup = 25 :=
by
    intros; sorry

end cost_per_ice_cream_l2198_219822


namespace games_played_l2198_219845

theorem games_played (x : ℕ) (h1 : x * 26 + 42 * (20 - x) = 600) : x = 15 :=
by {
  sorry
}

end games_played_l2198_219845


namespace divisor_of_7_l2198_219816

theorem divisor_of_7 (a n : ℤ) (h1 : a ≥ 1) (h2 : a ∣ (n + 2)) (h3 : a ∣ (n^2 + n + 5)) : a = 1 ∨ a = 7 :=
by
  sorry

end divisor_of_7_l2198_219816


namespace unique_last_digit_divisible_by_7_l2198_219800

theorem unique_last_digit_divisible_by_7 :
  ∃! d : ℕ, (∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d) :=
sorry

end unique_last_digit_divisible_by_7_l2198_219800


namespace pure_imaginary_solution_l2198_219894

theorem pure_imaginary_solution (m : ℝ) (h₁ : m^2 - m - 4 = 0) (h₂ : m^2 - 5 * m - 6 ≠ 0) :
  m = (1 + Real.sqrt 17) / 2 ∨ m = (1 - Real.sqrt 17) / 2 :=
sorry

end pure_imaginary_solution_l2198_219894


namespace sufficient_but_not_necessary_condition_l2198_219876

theorem sufficient_but_not_necessary_condition (x y m : ℝ) (h: x^2 + y^2 - 4 * x + 2 * y + m = 0):
  (m = 0) → (5 > m) ∧ ((5 > m) → (m ≠ 0)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l2198_219876
