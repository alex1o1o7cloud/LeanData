import Mathlib

namespace remainder_of_sum_div_11_is_9_l2068_206893

def seven_times_ten_pow_twenty : ℕ := 7 * 10 ^ 20
def two_pow_twenty : ℕ := 2 ^ 20
def sum : ℕ := seven_times_ten_pow_twenty + two_pow_twenty

theorem remainder_of_sum_div_11_is_9 : sum % 11 = 9 := by
  sorry

end remainder_of_sum_div_11_is_9_l2068_206893


namespace final_score_l2068_206801

-- Definitions based on the conditions
def bullseye_points : ℕ := 50
def miss_points : ℕ := 0
def half_bullseye_points : ℕ := bullseye_points / 2

-- Statement to prove
theorem final_score : bullseye_points + miss_points + half_bullseye_points = 75 :=
by
  sorry

end final_score_l2068_206801


namespace james_muffins_correct_l2068_206857

-- Arthur baked 115 muffins
def arthur_muffins : ℕ := 115

-- James baked 12 times as many muffins as Arthur
def james_multiplier : ℕ := 12

-- The number of muffins James baked
def james_muffins : ℕ := arthur_muffins * james_multiplier

-- The expected result
def expected_james_muffins : ℕ := 1380

-- The statement we want to prove
theorem james_muffins_correct : james_muffins = expected_james_muffins := by
  sorry

end james_muffins_correct_l2068_206857


namespace integral_2x_plus_3_squared_l2068_206817

open Real

-- Define the function to be integrated
def f (x : ℝ) := (2 * x + 3) ^ 2

-- State the theorem for the indefinite integral
theorem integral_2x_plus_3_squared :
  ∃ C : ℝ, ∫ x, f x = (1 / 6) * (2 * x + 3) ^ 3 + C :=
by
  sorry

end integral_2x_plus_3_squared_l2068_206817


namespace ordered_pair_correct_l2068_206808

def find_ordered_pair (s m : ℚ) : Prop :=
  (∀ t : ℚ, (∃ x y : ℚ, x = -3 + t * m ∧ y = s + t * (-7) ∧ y = (3/4) * x + 5))
  ∧ s = 11/4 ∧ m = -28/3

theorem ordered_pair_correct :
  find_ordered_pair (11/4) (-28/3) :=
by
  sorry

end ordered_pair_correct_l2068_206808


namespace total_monthly_sales_l2068_206898

-- Definitions and conditions
def num_customers_per_month : ℕ := 500
def lettuce_per_customer : ℕ := 2
def price_per_lettuce : ℕ := 1
def tomatoes_per_customer : ℕ := 4
def price_per_tomato : ℕ := 1 / 2

-- Statement to prove
theorem total_monthly_sales : num_customers_per_month * (lettuce_per_customer * price_per_lettuce + tomatoes_per_customer * price_per_tomato) = 2000 := 
by 
  sorry

end total_monthly_sales_l2068_206898


namespace complement_intersection_l2068_206841

open Set

-- Define the universal set I, and sets M and N
def I : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {3, 4, 5}

-- Lean statement to prove the desired result
theorem complement_intersection : (I \ N) ∩ M = {1, 2} := by
  sorry

end complement_intersection_l2068_206841


namespace neg_exists_le_zero_iff_forall_gt_zero_l2068_206899

variable (m : ℝ)

theorem neg_exists_le_zero_iff_forall_gt_zero :
  (¬ ∃ x : ℤ, (x:ℝ)^2 + 2 * x + m ≤ 0) ↔ ∀ x : ℤ, (x:ℝ)^2 + 2 * x + m > 0 :=
by
  sorry

end neg_exists_le_zero_iff_forall_gt_zero_l2068_206899


namespace nine_points_unit_square_l2068_206829

theorem nine_points_unit_square :
  ∀ (points : List (ℝ × ℝ)), points.length = 9 → 
  (∀ (x : ℝ × ℝ), x ∈ points → 0 ≤ x.1 ∧ x.1 ≤ 1 ∧ 0 ≤ x.2 ∧ x.2 ≤ 1) → 
  ∃ (A B C : ℝ × ℝ), A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ 
  (1 / 8 : ℝ) ≤ abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 :=
by
  sorry

end nine_points_unit_square_l2068_206829


namespace joshua_needs_more_cents_l2068_206869

-- Definitions of inputs
def cost_of_pen_dollars : ℕ := 6
def joshua_money_dollars : ℕ := 5
def borrowed_cents : ℕ := 68

-- Convert dollar amounts to cents
def dollar_to_cents (d : ℕ) : ℕ := d * 100

def cost_of_pen_cents := dollar_to_cents cost_of_pen_dollars
def joshua_money_cents := dollar_to_cents joshua_money_dollars

-- Total amount Joshua has in cents
def total_cents := joshua_money_cents + borrowed_cents

-- Calculation of the required amount
def needed_cents := cost_of_pen_cents - total_cents

theorem joshua_needs_more_cents : needed_cents = 32 := by 
  sorry

end joshua_needs_more_cents_l2068_206869


namespace AM_GM_Inequality_equality_condition_l2068_206831

-- Given conditions
variables (n : ℕ) (a b : ℝ)

-- Assumptions
lemma condition_n : 0 < n := sorry
lemma condition_a : 0 < a := sorry
lemma condition_b : 0 < b := sorry

-- Statement
theorem AM_GM_Inequality :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) :=
sorry

-- Equality condition
theorem equality_condition :
  (1 + a / b) ^ n + (1 + b / a) ^ n = 2 ^ (n + 1) ↔ a = b :=
sorry

end AM_GM_Inequality_equality_condition_l2068_206831


namespace polyhedron_volume_l2068_206885

-- Define the polyhedron and its properties
def polyhedron (P : Type) : Prop :=
∃ (C : Type), 
  (∀ (p : P) (e : ℝ), e = 2) ∧ 
  (∃ (octFaces triFaces : ℕ), octFaces = 6 ∧ triFaces = 8) ∧
  (∀ (vol : ℝ), vol = (56 + (112 * Real.sqrt 2) / 3))
  
-- A theorem stating the volume of the polyhedron
theorem polyhedron_volume : ∀ (P : Type), polyhedron P → ∃ (vol : ℝ), vol = 56 + (112 * Real.sqrt 2) / 3 :=
by
  intros P hP
  sorry

end polyhedron_volume_l2068_206885


namespace prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29_l2068_206881

theorem prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29 
  (n : ℕ) (h1 : Prime n) (h2 : 20 < n) (h3 : n < 30) (h4 : n % 8 = 5) : n = 29 := 
by
  sorry

end prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29_l2068_206881


namespace emily_spending_l2068_206820

theorem emily_spending : ∀ {x : ℝ}, (x + 2 * x + 3 * x = 120) → (x = 20) :=
by
  intros x h
  sorry

end emily_spending_l2068_206820


namespace vehicle_distance_traveled_l2068_206813

theorem vehicle_distance_traveled 
  (perimeter_back : ℕ) (perimeter_front : ℕ) (revolution_difference : ℕ)
  (R : ℕ)
  (h1 : perimeter_back = 9)
  (h2 : perimeter_front = 7)
  (h3 : revolution_difference = 10)
  (h4 : (R * perimeter_back) = ((R + revolution_difference) * perimeter_front)) :
  (R * perimeter_back) = 315 :=
by
  -- Prove that the distance traveled by the vehicle is 315 feet
  -- given the conditions and the hypothesis.
  sorry

end vehicle_distance_traveled_l2068_206813


namespace fill_in_square_l2068_206826

theorem fill_in_square (x y : ℝ) (h : 4 * x^2 * (81 / 4 * x * y) = 81 * x^3 * y) : (81 / 4 * x * y) = (81 / 4 * x * y) :=
by
  sorry

end fill_in_square_l2068_206826


namespace shinyoung_initial_candies_l2068_206851

theorem shinyoung_initial_candies : 
  ∀ (C : ℕ), 
    (C / 2) - ((C / 6) + 5) = 5 → 
    C = 30 := by
  intros C h
  sorry

end shinyoung_initial_candies_l2068_206851


namespace convert_neg_300_deg_to_rad_l2068_206823

theorem convert_neg_300_deg_to_rad :
  -300 * (Real.pi / 180) = - (5 / 3) * Real.pi :=
by
  sorry

end convert_neg_300_deg_to_rad_l2068_206823


namespace correct_operation_l2068_206854

theorem correct_operation (a b : ℝ) : 
  (a+2)*(a-2) = a^2 - 4 :=
by
  sorry

end correct_operation_l2068_206854


namespace parabola_distance_to_focus_l2068_206884

theorem parabola_distance_to_focus :
  ∀ (P : ℝ × ℝ), P.1 = 2 ∧ P.2^2 = 4 * P.1 → dist P (1, 0) = 3 :=
by
  intro P h
  have h₁ : P.1 = 2 := h.1
  have h₂ : P.2^2 = 4 * P.1 := h.2
  sorry

end parabola_distance_to_focus_l2068_206884


namespace find_x_plus_y_l2068_206852

-- Define the initial assumptions and conditions
variables {x y : ℝ}
axiom geom_sequence : 1 > 0 ∧ x > 0 ∧ y > 0 ∧ 3 > 0 ∧ 1 * x = y
axiom arith_sequence : 2 * y = x + 3

-- Prove that x + y = 15 / 4
theorem find_x_plus_y : x + y = 15 / 4 := sorry

end find_x_plus_y_l2068_206852


namespace division_quotient_less_dividend_l2068_206890

theorem division_quotient_less_dividend
  (a1 : (6 : ℝ) > 0)
  (a2 : (5 / 7 : ℝ) > 0)
  (a3 : (3 / 8 : ℝ) > 0)
  (h1 : (3 / 5 : ℝ) < 1)
  (h2 : (5 / 4 : ℝ) > 1)
  (h3 : (5 / 12 : ℝ) < 1):
  (6 / (3 / 5) > 6) ∧ (5 / 7 / (5 / 4) < 5 / 7) ∧ (3 / 8 / (5 / 12) > 3 / 8) :=
by
  sorry

end division_quotient_less_dividend_l2068_206890


namespace inequality_not_true_l2068_206834

variable (a b c : ℝ)

theorem inequality_not_true (h : a < b) : ¬ (-3 * a < -3 * b) :=
by
  sorry

end inequality_not_true_l2068_206834


namespace quotient_base6_division_l2068_206811

theorem quotient_base6_division :
  let a := 2045
  let b := 14
  let base := 6
  a / b = 51 :=
by
  sorry

end quotient_base6_division_l2068_206811


namespace johnny_years_ago_l2068_206859

theorem johnny_years_ago 
  (J : ℕ) (hJ : J = 8) (X : ℕ) 
  (h : J + 2 = 2 * (J - X)) : 
  X = 3 := by
  sorry

end johnny_years_ago_l2068_206859


namespace tree_ratio_l2068_206897

theorem tree_ratio (A P C : ℕ) 
  (hA : A = 58)
  (hP : P = 3 * A)
  (hC : C = 5 * P) : (A, P, C) = (1, 3 * 58, 15 * 58) :=
by
  sorry

end tree_ratio_l2068_206897


namespace simplify_expression_l2068_206803

theorem simplify_expression : 4 * (8 - 2 + 3) - 7 = 29 := 
by {
  sorry
}

end simplify_expression_l2068_206803


namespace parallelepiped_volume_l2068_206807

noncomputable def volume_of_parallelepiped (a : ℝ) : ℝ :=
  (a^3 * Real.sqrt 2) / 2

theorem parallelepiped_volume (a : ℝ) (h_pos : 0 < a) :
  volume_of_parallelepiped a = (a^3 * Real.sqrt 2) / 2 :=
by
  sorry

end parallelepiped_volume_l2068_206807


namespace clock_malfunction_fraction_correct_l2068_206875

theorem clock_malfunction_fraction_correct : 
  let hours_total := 24
  let hours_incorrect := 6
  let minutes_total := 60
  let minutes_incorrect := 6
  let fraction_correct_hours := (hours_total - hours_incorrect) / hours_total
  let fraction_correct_minutes := (minutes_total - minutes_incorrect) / minutes_total
  (fraction_correct_hours * fraction_correct_minutes) = 27 / 40
:= 
by
  sorry

end clock_malfunction_fraction_correct_l2068_206875


namespace area_of_rectangle_l2068_206805

def length : ℕ := 4
def width : ℕ := 2

theorem area_of_rectangle : length * width = 8 :=
by
  sorry

end area_of_rectangle_l2068_206805


namespace chord_square_length_l2068_206814

/-- Given three circles with radii 4, 8, and 16, such that the first two are externally tangent to each other and both are internally tangent to the third, if a chord in the circle with radius 16 is a common external tangent to the other two circles, then the square of the length of this chord is 7616/9. -/
theorem chord_square_length (r1 r2 r3 : ℝ) (h1 : r1 = 4) (h2 : r2 = 8) (h3 : r3 = 16)
  (tangent_condition : ∀ (O4 O8 O16 : ℝ), O4 = r1 + r2 ∧ O8 = r2 + r3 ∧ O16 = r1 + r3) :
  (16^2 - (20/3)^2) * 4 = 7616 / 9 :=
by
  sorry

end chord_square_length_l2068_206814


namespace find_b_in_geometric_sequence_l2068_206848

theorem find_b_in_geometric_sequence (a_1 : ℤ) :
  ∀ (n : ℕ), ∃ (b : ℤ), (3^n - b = (a_1 * (3^n - 1)) / 2) :=
by
  sorry

example (a_1 : ℤ) :
  ∃ (b : ℤ), ∀ (n : ℕ), 3^n - b = (a_1 * (3^n - 1)) / 2 :=
by
  use 1
  sorry

end find_b_in_geometric_sequence_l2068_206848


namespace platform_length_is_correct_l2068_206896

def speed_kmph : ℝ := 72
def seconds_to_cross_platform : ℝ := 26
def train_length_m : ℝ := 270.0416

noncomputable def length_of_platform : ℝ :=
  let speed_mps := speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * seconds_to_cross_platform
  total_distance - train_length_m

theorem platform_length_is_correct : 
  length_of_platform = 249.9584 := 
by
  sorry

end platform_length_is_correct_l2068_206896


namespace quadratic_inequality_solution_set_l2068_206872

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < -2) : 
  { x : ℝ | ax^2 + (a - 2)*x - 2 ≥ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 2/a } := 
by
  sorry

end quadratic_inequality_solution_set_l2068_206872


namespace cube_root_of_neg_125_l2068_206816

theorem cube_root_of_neg_125 : (-5)^3 = -125 := 
by sorry

end cube_root_of_neg_125_l2068_206816


namespace sequence_a1_l2068_206883

variable (S : ℕ → ℤ) (a : ℕ → ℤ)

def Sn_formula (n : ℕ) (a₁ : ℤ) : ℤ := (a₁ * (4^n - 1)) / 3

theorem sequence_a1 (h1 : ∀ n : ℕ, S n = Sn_formula n (a 1))
                    (h2 : a 4 = 32) :
  a 1 = 1 / 2 :=
by
  sorry

end sequence_a1_l2068_206883


namespace new_person_weight_l2068_206847

theorem new_person_weight (avg_increase : ℝ) (num_people : ℕ) (weight_replaced : ℝ) (new_weight : ℝ) : 
    num_people = 8 → avg_increase = 1.5 → weight_replaced = 65 → 
    new_weight = weight_replaced + num_people * avg_increase → 
    new_weight = 77 :=
by
  intros h1 h2 h3 h4
  sorry

end new_person_weight_l2068_206847


namespace linear_function_is_C_l2068_206812

theorem linear_function_is_C :
  ∀ (f : ℤ → ℤ), (f = (λ x => 2 * x^2 - 1) ∨ f = (λ x => -1/x) ∨ f = (λ x => (x+1)/3) ∨ f = (λ x => 3 * x + 2 * x^2 - 1)) →
  (f = (λ x => (x+1)/3)) ↔ 
  (∃ (m b : ℤ), ∀ x : ℤ, f x = m * x + b) :=
by
  sorry

end linear_function_is_C_l2068_206812


namespace real_roots_if_and_only_if_m_leq_5_l2068_206866

theorem real_roots_if_and_only_if_m_leq_5 (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x + 1 = 0) ↔ m ≤ 5 :=
by
  sorry

end real_roots_if_and_only_if_m_leq_5_l2068_206866


namespace number_of_integers_between_cubed_values_l2068_206861

theorem number_of_integers_between_cubed_values :
  ∃ n : ℕ, n = (1278 - 1122 + 1) ∧ 
  ∀ x : ℤ, (1122 < x ∧ x < 1278) → (1123 ≤ x ∧ x ≤ 1277) := 
by
  sorry

end number_of_integers_between_cubed_values_l2068_206861


namespace range_of_a_l2068_206839

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x >= 2 then (a - 1 / 2) * x 
  else a^x - 4

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) ↔ (1 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l2068_206839


namespace functional_equation_solution_l2068_206891

-- Define the conditions of the problem.
variable (f : ℝ → ℝ) 
variable (h : ∀ x y u v : ℝ, (f x + f y) * (f u + f v) = f (x * u - y * v) + f (x * v + y * u))

-- Formalize the statement that no other functions satisfy the conditions except f(x) = x^2.
theorem functional_equation_solution : (∀ x : ℝ, f x = x^2) :=
by
  -- The proof goes here, but since the proof is not required, we skip it.
  sorry

end functional_equation_solution_l2068_206891


namespace rate_of_interest_l2068_206806

theorem rate_of_interest (P T SI: ℝ) (h1 : P = 2500) (h2 : T = 5) (h3 : SI = P - 2000) (h4 : SI = (P * R * T) / 100):
  R = 4 :=
by
  sorry

end rate_of_interest_l2068_206806


namespace polynomial_division_l2068_206849

theorem polynomial_division (a b c : ℤ) :
  (∀ x : ℝ, (17 * x^2 - 3 * x + 4) - (a * x^2 + b * x + c) = (5 * x + 6) * (2 * x + 1)) →
  a - b - c = 29 := by
  sorry

end polynomial_division_l2068_206849


namespace C_work_completion_l2068_206877

theorem C_work_completion (A_completion_days B_completion_days AB_completion_days : ℕ)
  (A_cond : A_completion_days = 8)
  (B_cond : B_completion_days = 12)
  (AB_cond : AB_completion_days = 4) :
  ∃ (C_completion_days : ℕ), C_completion_days = 24 := 
by
  sorry

end C_work_completion_l2068_206877


namespace city_map_distance_example_l2068_206838

variable (distance_on_map : ℝ)
variable (scale : ℝ)
variable (actual_distance : ℝ)

theorem city_map_distance_example
  (h1 : distance_on_map = 16)
  (h2 : scale = 1 / 10000)
  (h3 : actual_distance = distance_on_map / scale) :
  actual_distance = 1.6 * 10^3 :=
by
  sorry

end city_map_distance_example_l2068_206838


namespace sector_central_angle_l2068_206865

theorem sector_central_angle (r θ : ℝ) (h1 : 2 * r + r * θ = 6) (h2 : 0.5 * r * r * θ = 2) : θ = 1 ∨ θ = 4 :=
sorry

end sector_central_angle_l2068_206865


namespace graph_intersect_points_l2068_206855

-- Define f as a function defined on all real numbers and invertible
variable (f : ℝ → ℝ) (hf : Function.Injective f)

-- Define the theorem to find the number of intersection points
theorem graph_intersect_points : 
  ∃ (n : ℕ), n = 3 ∧ ∃ (x : ℝ), (f (x^2) = f (x^6)) :=
  by
    -- Outline sketch: We aim to show there are 3 real solutions satisfying the equation
    -- The proof here is skipped, hence we put sorry
    sorry

end graph_intersect_points_l2068_206855


namespace decreasing_geometric_sequence_l2068_206887

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * q ^ n

theorem decreasing_geometric_sequence (a₁ q : ℝ) (aₙ : ℕ → ℝ) (hₙ : ∀ n, aₙ n = geometric_sequence a₁ q n) 
  (h_condition : 0 < q ∧ q < 1) : ¬(0 < q ∧ q < 1 ↔ ∀ n, aₙ n > aₙ (n + 1)) :=
sorry

end decreasing_geometric_sequence_l2068_206887


namespace volume_of_wedge_l2068_206860

theorem volume_of_wedge (d : ℝ) (angle : ℝ) (V : ℝ) (n : ℕ) 
  (h_d : d = 18) 
  (h_angle : angle = 60)
  (h_radius_height : ∀ r h, r = d / 2 ∧ h = d) 
  (h_volume_cylinder : V = π * (d / 2) ^ 2 * d) 
  : n = 729 ↔ V / 2 = n * π :=
by
  sorry

end volume_of_wedge_l2068_206860


namespace quadratic_graph_y1_lt_y2_l2068_206880

theorem quadratic_graph_y1_lt_y2 (x1 x2 : ℝ) (h1 : -x1^2 = y1) (h2 : -x2^2 = y2) (h3 : x1 * x2 > x2^2) : y1 < y2 :=
  sorry

end quadratic_graph_y1_lt_y2_l2068_206880


namespace lizard_eye_difference_l2068_206892

def jan_eye : ℕ := 3
def jan_wrinkle : ℕ := 3 * jan_eye
def jan_spot : ℕ := 7 * jan_wrinkle

def cousin_eye : ℕ := 3
def cousin_wrinkle : ℕ := 2 * cousin_eye
def cousin_spot : ℕ := 5 * cousin_wrinkle

def total_eyes : ℕ := jan_eye + cousin_eye
def total_wrinkles : ℕ := jan_wrinkle + cousin_wrinkle
def total_spots : ℕ := jan_spot + cousin_spot
def total_spots_and_wrinkles : ℕ := total_wrinkles + total_spots

theorem lizard_eye_difference : total_spots_and_wrinkles - total_eyes = 102 := by
  sorry

end lizard_eye_difference_l2068_206892


namespace man_work_days_l2068_206830

variable (W : ℝ) -- Denoting the amount of work by W

-- Defining the work rate variables
variables (M Wm B : ℝ)

-- Conditions from the problem:
-- Combined work rate of man, woman, and boy together completes the work in 3 days
axiom combined_work_rate : M + Wm + B = W / 3
-- Woman completes the work alone in 18 days
axiom woman_work_rate : Wm = W / 18
-- Boy completes the work alone in 9 days
axiom boy_work_rate : B = W / 9

-- The goal is to prove the man takes 6 days to complete the work alone
theorem man_work_days : (W / M) = 6 :=
by
  sorry

end man_work_days_l2068_206830


namespace total_decorations_l2068_206850

theorem total_decorations 
  (skulls : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) (pumpkins : ℕ) 
  (cauldron : ℕ) (budget_decorations : ℕ) (left_decorations : ℕ)
  (h_skulls : skulls = 12)
  (h_broomsticks : broomsticks = 4)
  (h_spiderwebs : spiderwebs = 12)
  (h_pumpkins : pumpkins = 2 * spiderwebs)
  (h_cauldron : cauldron = 1)
  (h_budget_decorations : budget_decorations = 20)
  (h_left_decorations : left_decorations = 10) : 
  skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget_decorations + left_decorations = 83 := 
by 
  sorry

end total_decorations_l2068_206850


namespace union_A_B_inter_A_B_inter_compA_B_l2068_206856

-- Extend the universal set U to be the set of all real numbers ℝ
def U : Set ℝ := Set.univ

-- Define set A as the set of all real numbers x such that -3 ≤ x ≤ 4
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}

-- Define set B as the set of all real numbers x such that -1 < x < 5
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}

-- Prove that A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 5}
theorem union_A_B : A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 5} := by
  sorry

-- Prove that A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 4}
theorem inter_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 4} := by
  sorry

-- Define the complement of A in U
def comp_A : Set ℝ := {x : ℝ | x < -3 ∨ x > 4}

-- Prove that (complement_U A) ∩ B = {x : ℝ | 4 < x ∧ x < 5}
theorem inter_compA_B : comp_A ∩ B = {x : ℝ | 4 < x ∧ x < 5} := by
  sorry

end union_A_B_inter_A_B_inter_compA_B_l2068_206856


namespace find_integers_l2068_206832

theorem find_integers (x : ℤ) : x^2 < 3 * x → x = 1 ∨ x = 2 := by
  sorry

end find_integers_l2068_206832


namespace find_second_sum_l2068_206827

theorem find_second_sum (x : ℝ) (total_sum : ℝ) (h : total_sum = 2691) 
  (h1 : (24 * x) / 100 = 15 * (total_sum - x) / 100) : total_sum - x = 1656 :=
by
  sorry

end find_second_sum_l2068_206827


namespace b_contribution_is_correct_l2068_206845

-- Definitions based on the conditions
def A_investment : ℕ := 35000
def B_join_after_months : ℕ := 5
def profit_ratio_A_B : ℕ := 2
def profit_ratio_B_A : ℕ := 3
def A_total_months : ℕ := 12
def B_total_months : ℕ := 7
def profit_ratio := (profit_ratio_A_B, profit_ratio_B_A)
def total_investment_time_ratio : ℕ := 12 * 35000 / 7

-- The property to be proven
theorem b_contribution_is_correct (X : ℕ) (h : 35000 * 12 / (X * 7) = 2 / 3) : X = 90000 :=
by
  sorry

end b_contribution_is_correct_l2068_206845


namespace total_stops_traveled_l2068_206889

-- Definitions based on the conditions provided
def yoojeong_stops : ℕ := 3
def namjoon_stops : ℕ := 2

-- Theorem statement to prove the total number of stops
theorem total_stops_traveled : yoojeong_stops + namjoon_stops = 5 := by
  -- Proof omitted
  sorry

end total_stops_traveled_l2068_206889


namespace pool_water_left_l2068_206874

theorem pool_water_left 
  (h1_rate: ℝ) (h1_time: ℝ)
  (h2_rate: ℝ) (h2_time: ℝ)
  (h4_rate: ℝ) (h4_time: ℝ)
  (leak_loss: ℝ)
  (h1_rate_eq: h1_rate = 8)
  (h1_time_eq: h1_time = 1)
  (h2_rate_eq: h2_rate = 10)
  (h2_time_eq: h2_time = 2)
  (h4_rate_eq: h4_rate = 14)
  (h4_time_eq: h4_time = 1)
  (leak_loss_eq: leak_loss = 8) :
  (h1_rate * h1_time) + (h2_rate * h2_time) + (h2_rate * h2_time) + (h4_rate * h4_time) - leak_loss = 34 :=
by
  rw [h1_rate_eq, h1_time_eq, h2_rate_eq, h2_time_eq, h4_rate_eq, h4_time_eq, leak_loss_eq]
  norm_num
  sorry

end pool_water_left_l2068_206874


namespace football_field_width_l2068_206824

theorem football_field_width (length : ℕ) (total_distance : ℕ) (laps : ℕ) (width : ℕ) 
  (h1 : length = 100) (h2 : total_distance = 1800) (h3 : laps = 6) :
  width = 50 :=
by 
  -- Proof omitted
  sorry

end football_field_width_l2068_206824


namespace exp_inequality_solution_l2068_206804

theorem exp_inequality_solution (x : ℝ) (h : 1 < Real.exp x ∧ Real.exp x < 2) : 0 < x ∧ x < Real.log 2 :=
by
  sorry

end exp_inequality_solution_l2068_206804


namespace max_abs_sum_l2068_206818

theorem max_abs_sum (a b c : ℝ) (h : ∀ x, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  |a| + |b| + |c| ≤ 3 :=
sorry

end max_abs_sum_l2068_206818


namespace dot_product_calculation_l2068_206840

def vector := (ℤ × ℤ)

def dot_product (v1 v2 : vector) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : vector := (1, 3)
def b : vector := (-1, 2)

def scalar_mult (c : ℤ) (v : vector) : vector :=
  (c * v.1, c * v.2)

def vector_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

theorem dot_product_calculation :
  dot_product (vector_add (scalar_mult 2 a) b) b = 15 := by
  sorry

end dot_product_calculation_l2068_206840


namespace gain_is_rs_150_l2068_206843

noncomputable def P : ℝ := 5000
noncomputable def R_borrow : ℝ := 4
noncomputable def R_lend : ℝ := 7
noncomputable def T : ℝ := 2

noncomputable def SI (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (P * R * T) / 100

noncomputable def interest_paid := SI P R_borrow T
noncomputable def interest_earned := SI P R_lend T

noncomputable def gain_per_year : ℝ :=
  (interest_earned / T) - (interest_paid / T)

theorem gain_is_rs_150 : gain_per_year = 150 :=
by
  sorry

end gain_is_rs_150_l2068_206843


namespace solve_inequality_l2068_206878

variable {a x : ℝ}

theorem solve_inequality (h : a > 0) : 
  (ax^2 - (a + 1)*x + 1 < 0) ↔ 
    (if 0 < a ∧ a < 1 then 1 < x ∧ x < 1/a else 
     if a = 1 then false else 
     if a > 1 then 1/a < x ∧ x < 1 else true) :=
  sorry

end solve_inequality_l2068_206878


namespace cos_double_angle_l2068_206819

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 1 / 2) : Real.cos (2 * α) = 3 / 5 :=
by
  sorry

end cos_double_angle_l2068_206819


namespace find_original_number_l2068_206868

theorem find_original_number (r : ℝ) (h : 1.15 * r - 0.7 * r = 40) : r = 88.88888888888889 :=
by
  sorry

end find_original_number_l2068_206868


namespace value_of_x_plus_y_l2068_206800

theorem value_of_x_plus_y 
  (x y : ℝ) 
  (h1 : -x = 3) 
  (h2 : |y| = 5) : 
  x + y = 2 ∨ x + y = -8 := 
  sorry

end value_of_x_plus_y_l2068_206800


namespace range_of_c_l2068_206822

theorem range_of_c (c : ℝ) :
  (c^2 - 5 * c + 7 > 1 ∧ (|2 * c - 1| ≤ 1)) ∨ ((c^2 - 5 * c + 7 ≤ 1) ∧ |2 * c - 1| > 1) ↔ (0 ≤ c ∧ c ≤ 1) ∨ (2 ≤ c ∧ c ≤ 3) :=
sorry

end range_of_c_l2068_206822


namespace find_j_value_l2068_206802

variable {R : Type*} [LinearOrderedField R]

-- Definitions based on conditions
def polynomial_has_four_distinct_real_roots_in_arithmetic_progression
(p : Polynomial R) : Prop :=
∃ a d : R, p.roots.toFinset = {a, a + d, a + 2*d, a + 3*d} ∧
a ≠ a + d ∧ a ≠ a + 2*d ∧ a ≠ a + 3*d ∧ a + d ≠ a + 2*d ∧
a + d ≠ a + 3*d ∧ a + 2*d ≠ a + 3*d

-- The main theorem statement
theorem find_j_value (k : R) 
  (h : polynomial_has_four_distinct_real_roots_in_arithmetic_progression 
  (Polynomial.X^4 + Polynomial.C j * Polynomial.X^2 + Polynomial.C k * Polynomial.X + Polynomial.C 900)) :
  j = -900 :=
sorry

end find_j_value_l2068_206802


namespace number_of_white_balls_l2068_206862

-- Definition of conditions
def red_balls : ℕ := 4
def frequency_of_red_balls : ℝ := 0.25
def total_balls (white_balls : ℕ) : ℕ := red_balls + white_balls

-- Proving the number of white balls given the conditions
theorem number_of_white_balls (x : ℕ) :
  (red_balls : ℝ) / total_balls x = frequency_of_red_balls → x = 12 :=
by
  sorry

end number_of_white_balls_l2068_206862


namespace exponential_function_decreasing_l2068_206879

theorem exponential_function_decreasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (0 < a ∧ a < 1) → ¬ (∀ x : ℝ, x > 0 → a ^ x > 0) :=
by
  sorry

end exponential_function_decreasing_l2068_206879


namespace graph_of_equation_is_two_lines_l2068_206873

theorem graph_of_equation_is_two_lines : 
  ∀ (x y : ℝ), (x - y)^2 = x^2 - y^2 ↔ (x = 0 ∨ y = 0) := 
by
  sorry

end graph_of_equation_is_two_lines_l2068_206873


namespace value_of_a10_l2068_206888

/-- Define arithmetic sequence and properties -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = (n * (a 1 + a n) / 2)

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}
axiom arith_seq : arithmetic_sequence a d
axiom sum_formula : sum_of_first_n_terms a 5 S
axiom sum_condition : S 5 = 60
axiom term_condition : a 1 + a 2 + a 3 = a 4 + a 5

theorem value_of_a10 : a 10 = 26 :=
sorry

end value_of_a10_l2068_206888


namespace original_pencil_count_l2068_206846

-- Defining relevant constants and assumptions based on the problem conditions
def pencilsRemoved : ℕ := 4
def pencilsLeft : ℕ := 83

-- Theorem to prove the original number of pencils is 87
theorem original_pencil_count : pencilsLeft + pencilsRemoved = 87 := by
  sorry

end original_pencil_count_l2068_206846


namespace gcf_lcm_60_72_l2068_206870

def gcf_lcm_problem (a b : ℕ) : Prop :=
  gcd a b = 12 ∧ lcm a b = 360

theorem gcf_lcm_60_72 : gcf_lcm_problem 60 72 :=
by {
  sorry
}

end gcf_lcm_60_72_l2068_206870


namespace add_ten_to_certain_number_l2068_206894

theorem add_ten_to_certain_number (x : ℤ) (h : x + 36 = 71) : x + 10 = 45 :=
by
  sorry

end add_ten_to_certain_number_l2068_206894


namespace difference_in_pages_l2068_206895

def purple_pages_per_book : ℕ := 230
def orange_pages_per_book : ℕ := 510
def purple_books_read : ℕ := 5
def orange_books_read : ℕ := 4

theorem difference_in_pages : 
  orange_books_read * orange_pages_per_book - purple_books_read * purple_pages_per_book = 890 :=
by
  sorry

end difference_in_pages_l2068_206895


namespace one_in_set_A_l2068_206810

theorem one_in_set_A : 1 ∈ {x | x ≥ -1} :=
sorry

end one_in_set_A_l2068_206810


namespace tan_alpha_eq_one_l2068_206864

open Real

theorem tan_alpha_eq_one (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (h_cos_sin_eq : cos (α + β) = sin (α - β)) : tan α = 1 :=
by
  sorry

end tan_alpha_eq_one_l2068_206864


namespace polynomial_value_given_cond_l2068_206825

variable (x : ℝ)
theorem polynomial_value_given_cond :
  (x^2 - (5/2) * x = 6) →
  2 * x^2 - 5 * x + 6 = 18 :=
by
  sorry

end polynomial_value_given_cond_l2068_206825


namespace measure_of_angle_f_l2068_206809

theorem measure_of_angle_f (angle_D angle_E angle_F : ℝ)
  (h1 : angle_D = 75)
  (h2 : angle_E = 4 * angle_F + 30)
  (h3 : angle_D + angle_E + angle_F = 180) : 
  angle_F = 15 :=
by
  sorry

end measure_of_angle_f_l2068_206809


namespace area_of_rectangle_l2068_206886

-- Definitions and conditions
def side_of_square : ℕ := 50
def radius_of_circle : ℕ := side_of_square
def length_of_rectangle : ℕ := (2 * radius_of_circle) / 5
def breadth_of_rectangle : ℕ := 10

-- Theorem statement
theorem area_of_rectangle :
  (length_of_rectangle * breadth_of_rectangle = 200) := by
  sorry

end area_of_rectangle_l2068_206886


namespace number_of_true_statements_l2068_206837

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m
def is_odd (n : ℕ) : Prop := ∃ m : ℕ, n = 2 * m + 1
def is_even (n : ℕ) : Prop := ∃ m : ℕ, n = 2 * m

theorem number_of_true_statements : 3 = (ite ((∀ p q : ℕ, is_prime p → is_prime q → is_prime (p * q)) = false) 0 1) +
                                     (ite ((∀ a b : ℕ, is_square a → is_square b → is_square (a * b)) = true) 1 0) +
                                     (ite ((∀ x y : ℕ, is_odd x → is_odd y → is_odd (x * y)) = true) 1 0) +
                                     (ite ((∀ u v : ℕ, is_even u → is_even v → is_even (u * v)) = true) 1 0) :=
by
  sorry

end number_of_true_statements_l2068_206837


namespace remaining_amount_is_9_l2068_206871

-- Define the original prices of the books
def book1_price : ℝ := 13.00
def book2_price : ℝ := 15.00
def book3_price : ℝ := 10.00
def book4_price : ℝ := 10.00

-- Define the discount rate for the first two books
def discount_rate : ℝ := 0.25

-- Define the total cost without discount
def total_cost_without_discount := book1_price + book2_price + book3_price + book4_price

-- Calculate the discounts for the first two books
def book1_discount := book1_price * discount_rate
def book2_discount := book2_price * discount_rate

-- Calculate the discounted prices for the first two books
def discounted_book1_price := book1_price - book1_discount
def discounted_book2_price := book2_price - book2_discount

-- Calculate the total cost of the books with discounts applied
def total_cost_with_discount := discounted_book1_price + discounted_book2_price + book3_price + book4_price

-- Define the free shipping threshold
def free_shipping_threshold : ℝ := 50.00

-- Calculate the remaining amount Connor needs to spend
def remaining_amount_to_spend := free_shipping_threshold - total_cost_with_discount

-- State the theorem
theorem remaining_amount_is_9 : remaining_amount_to_spend = 9.00 := by
  -- we would provide the proof here
  sorry

end remaining_amount_is_9_l2068_206871


namespace gcd_765432_654321_l2068_206815

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end gcd_765432_654321_l2068_206815


namespace tessa_needs_more_apples_l2068_206828

/-- Tessa starts with 4 apples.
    Anita gives her 5 more apples.
    She needs 10 apples to make a pie.
    Prove that she needs 1 more apple to make the pie.
-/
theorem tessa_needs_more_apples:
  ∀ initial_apples extra_apples total_needed extra_needed: ℕ,
    initial_apples = 4 → extra_apples = 5 → total_needed = 10 →
    extra_needed = total_needed - (initial_apples + extra_apples) →
    extra_needed = 1 :=
by
  intros initial_apples extra_apples total_needed extra_needed hi he ht heq
  rw [hi, he, ht] at heq
  simp at heq
  assumption

end tessa_needs_more_apples_l2068_206828


namespace largest_int_mod_6_less_than_100_l2068_206882

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l2068_206882


namespace robotics_club_students_l2068_206863

theorem robotics_club_students (total cs e both neither : ℕ) 
  (h1 : total = 80)
  (h2 : cs = 52)
  (h3 : e = 38)
  (h4 : both = 25)
  (h5 : neither = total - (cs - both + e - both + both)) :
  neither = 15 :=
by
  sorry

end robotics_club_students_l2068_206863


namespace soja_book_page_count_l2068_206858

theorem soja_book_page_count (P : ℕ) (h1 : P > 0) (h2 : (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 100) : P = 300 :=
by
  -- The Lean proof is not required, so we just add sorry to skip the proof
  sorry

end soja_book_page_count_l2068_206858


namespace determine_a_l2068_206867

-- Define the sets A and B
def A : Set ℝ := { -1, 0, 2 }
def B (a : ℝ) : Set ℝ := { 2^a }

-- State the main theorem
theorem determine_a (a : ℝ) (h : B a ⊆ A) : a = 1 :=
by
  sorry

end determine_a_l2068_206867


namespace annie_journey_time_l2068_206844

noncomputable def total_time_journey (walk_speed1 bus_speed train_speed walk_speed2 blocks_walk1 blocks_bus blocks_train blocks_walk2 : ℝ) : ℝ :=
  let time_walk1 := blocks_walk1 / walk_speed1
  let time_bus := blocks_bus / bus_speed
  let time_train := blocks_train / train_speed
  let time_walk2 := blocks_walk2 / walk_speed2
  let time_back := time_walk2
  time_walk1 + time_bus + time_train + time_walk2 + time_back + time_train + time_bus + time_walk1

theorem annie_journey_time :
  total_time_journey 2 4 5 2 5 7 10 4 = 16.5 := by 
  sorry

end annie_journey_time_l2068_206844


namespace math_problem_l2068_206821

theorem math_problem 
  (x y z : ℚ)
  (h1 : 4 * x - 5 * y - z = 0)
  (h2 : x + 5 * y - 18 * z = 0)
  (hz : z ≠ 0) :
  (x^2 + 4 * x * y) / (y^2 + z^2) = 3622 / 9256 := 
sorry

end math_problem_l2068_206821


namespace tan_identity_proof_l2068_206835

noncomputable def tan_add_pi_over_3 (α β : ℝ) : ℝ :=
  Real.tan (α + Real.pi / 3)

theorem tan_identity_proof 
  (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3 / 5)
  (h2 : Real.tan (β - Real.pi / 3) = 1 / 4) :
  tan_add_pi_over_3 α β = 7 / 23 := 
sorry

end tan_identity_proof_l2068_206835


namespace num_students_third_section_l2068_206833

-- Define the conditions
def num_students_first_section : ℕ := 65
def num_students_second_section : ℕ := 35
def num_students_fourth_section : ℕ := 42
def mean_marks_first_section : ℝ := 50
def mean_marks_second_section : ℝ := 60
def mean_marks_third_section : ℝ := 55
def mean_marks_fourth_section : ℝ := 45
def overall_average_marks : ℝ := 51.95

-- Theorem stating the number of students in the third section
theorem num_students_third_section
  (x : ℝ)
  (h : (num_students_first_section * mean_marks_first_section
       + num_students_second_section * mean_marks_second_section
       + x * mean_marks_third_section
       + num_students_fourth_section * mean_marks_fourth_section)
       = overall_average_marks * (num_students_first_section + num_students_second_section + x + num_students_fourth_section)) :
  x = 45 :=
by
  -- Proof will go here
  sorry

end num_students_third_section_l2068_206833


namespace initial_people_count_l2068_206836

theorem initial_people_count (x : ℕ) 
  (h1 : (x + 15) % 5 = 0)
  (h2 : (x + 15) / 5 = 12) : 
  x = 45 := 
by
  sorry

end initial_people_count_l2068_206836


namespace delacroix_band_max_members_l2068_206876

theorem delacroix_band_max_members :
  ∃ n : ℕ, 30 * n % 28 = 6 ∧ 30 * n < 1200 ∧ 30 * n = 930 :=
by
  sorry

end delacroix_band_max_members_l2068_206876


namespace find_m_from_decomposition_l2068_206853

theorem find_m_from_decomposition (m : ℕ) (h : m > 0) : (m^2 - m + 1 = 73) → (m = 9) :=
by
  sorry

end find_m_from_decomposition_l2068_206853


namespace bob_age_l2068_206842

theorem bob_age (a b : ℝ) 
    (h1 : b = 3 * a - 20)
    (h2 : b + a = 70) : 
    b = 47.5 := by
    sorry

end bob_age_l2068_206842
