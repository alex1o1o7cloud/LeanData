import Mathlib

namespace largest_integral_ratio_l248_24852

theorem largest_integral_ratio (P A : ℕ) (rel_prime_sides : ∃ (a b c : ℕ), gcd a b = 1 ∧ gcd b c = 1 ∧ gcd c a = 1 ∧ a^2 + b^2 = c^2 ∧ P = a + b + c ∧ A = a * b / 2) :
  (∃ (k : ℕ), k = 45 ∧ ∀ l, l < 45 → l ≠ (P^2 / A)) :=
sorry

end largest_integral_ratio_l248_24852


namespace problem1_problem2_l248_24897

-- Problem statement 1: Prove (a-2)(a-6) < (a-3)(a-5)
theorem problem1 (a : ℝ) : (a - 2) * (a - 6) < (a - 3) * (a - 5) :=
by
  sorry

-- Problem statement 2: Prove the range of values for 2x - y given -2 < x < 1 and 1 < y < 2 is (-6, 1)
theorem problem2 (x y : ℝ) (hx : -2 < x) (hx1 : x < 1) (hy : 1 < y) (hy1 : y < 2) : -6 < 2 * x - y ∧ 2 * x - y < 1 :=
by
  sorry

end problem1_problem2_l248_24897


namespace total_raining_time_correct_l248_24845

-- Define individual durations based on given conditions
def duration_day1 : ℕ := 10        -- 17:00 - 07:00 = 10 hours
def duration_day2 : ℕ := duration_day1 + 2    -- Second day: 10 hours + 2 hours = 12 hours
def duration_day3 : ℕ := duration_day2 * 2    -- Third day: 12 hours * 2 = 24 hours

-- Define the total raining time over three days
def total_raining_time : ℕ := duration_day1 + duration_day2 + duration_day3

-- Formally state the theorem to prove the total rain time is 46 hours
theorem total_raining_time_correct : total_raining_time = 46 := by
  sorry

end total_raining_time_correct_l248_24845


namespace total_reading_materials_l248_24836

def reading_materials (magazines newspapers books pamphlets : ℕ) : ℕ :=
  magazines + newspapers + books + pamphlets

theorem total_reading_materials:
  reading_materials 425 275 150 75 = 925 := by
  sorry

end total_reading_materials_l248_24836


namespace cone_radius_l248_24805

theorem cone_radius (h : ℝ) (V : ℝ) (π : ℝ) (r : ℝ)
    (h_def : h = 21)
    (V_def : V = 2199.114857512855)
    (volume_formula : V = (1/3) * π * r^2 * h) : r = 10 :=
by {
  sorry
}

end cone_radius_l248_24805


namespace avg_speed_4_2_l248_24822

noncomputable def avg_speed_round_trip (D : ℝ) : ℝ :=
  let speed_up := 3
  let speed_down := 7
  let total_distance := 2 * D
  let total_time := D / speed_up + D / speed_down
  total_distance / total_time

theorem avg_speed_4_2 (D : ℝ) (hD : D > 0) : avg_speed_round_trip D = 4.2 := by
  sorry

end avg_speed_4_2_l248_24822


namespace shortest_part_length_l248_24870

theorem shortest_part_length (total_length : ℝ) (r1 r2 r3 : ℝ) (shortest_length : ℝ) :
  total_length = 196.85 → r1 = 3.6 → r2 = 8.4 → r3 = 12 → shortest_length = 29.5275 :=
by
  sorry

end shortest_part_length_l248_24870


namespace total_students_l248_24858

theorem total_students (N : ℕ)
  (h1 : (84 + 128 + 13 = 15 * N))
  : N = 15 :=
sorry

end total_students_l248_24858


namespace boys_in_school_l248_24813

theorem boys_in_school (x : ℕ) (boys girls : ℕ) (h1 : boys = 5 * x) 
  (h2 : girls = 13 * x) (h3 : girls - boys = 128) : boys = 80 :=
by
  sorry

end boys_in_school_l248_24813


namespace value_of_b_plus_c_l248_24827

variable {a b c d : ℝ}

theorem value_of_b_plus_c (h1 : a + b = 4) (h2 : c + d = 5) (h3 : a + d = 2) : b + c = 7 :=
sorry

end value_of_b_plus_c_l248_24827


namespace range_of_h_l248_24821

noncomputable def h : ℝ → ℝ
| x => if x = -7 then 0 else 2 * (x - 3)

theorem range_of_h :
  (Set.range h) = Set.univ \ {-20} :=
sorry

end range_of_h_l248_24821


namespace geom_prog_common_ratio_l248_24869

variable {α : Type*} [Field α]

theorem geom_prog_common_ratio (x y z r : α) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) 
  (h1 : x * (y + z) = a) (h2 : y * (z + x) = a * r) (h3 : z * (x + y) = a * r^2) :
  r^2 + r + 1 = 0 :=
by
  sorry

end geom_prog_common_ratio_l248_24869


namespace factorization_2109_two_digit_l248_24895

theorem factorization_2109_two_digit (a b: ℕ) : 
  2109 = a * b ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 → false :=
by
  sorry

end factorization_2109_two_digit_l248_24895


namespace relationship_y1_y2_l248_24877

theorem relationship_y1_y2 :
  ∀ (b y1 y2 : ℝ), 
  (∃ b y1 y2, y1 = -2023 * (-2) + b ∧ y2 = -2023 * (-1) + b) → y1 > y2 :=
by
  intro b y1 y2 h
  sorry

end relationship_y1_y2_l248_24877


namespace chris_did_not_get_A_l248_24853

variable (A : Prop) (MC_correct : Prop) (Essay80 : Prop)

-- The condition provided by professor
axiom condition : A ↔ (MC_correct ∧ Essay80)

-- The theorem we need to prove based on the statement (B) from the solution
theorem chris_did_not_get_A 
    (h : ¬ A) : ¬ MC_correct ∨ ¬ Essay80 :=
by sorry

end chris_did_not_get_A_l248_24853


namespace sin_cos_sum_l248_24804

theorem sin_cos_sum (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π)
  (h : Real.tan (θ + Real.pi / 4) = 1 / 7) : Real.sin θ + Real.cos θ = -1 / 5 := 
by
  sorry

end sin_cos_sum_l248_24804


namespace zero_in_interval_l248_24800

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem zero_in_interval : ∃ x0, f x0 = 0 ∧ 2 < x0 ∧ x0 < 3 :=
by
  have h_cont : Continuous f := sorry -- f is continuous (can be proven using the continuity of log and linear functions)
  have h_eval1 : f 2 < 0 := sorry -- f(2) = ln(2) - 6 + 4 < 0
  have h_eval2 : f 3 > 0 := sorry -- f(3) = ln(3) - 6 + 6 > 0
  -- By the Intermediate Value Theorem, since f is continuous and changes signs between (2, 3), there exists a zero x0 in (2, 3).
  exact sorry

end zero_in_interval_l248_24800


namespace radio_loss_percentage_l248_24830

theorem radio_loss_percentage (cost_price selling_price : ℕ) (h1 : cost_price = 1500) (h2 : selling_price = 1305) : 
  (cost_price - selling_price) * 100 / cost_price = 13 := by
  sorry

end radio_loss_percentage_l248_24830


namespace add_decimals_l248_24891

theorem add_decimals : 5.763 + 2.489 = 8.252 := 
by
  sorry

end add_decimals_l248_24891


namespace eggs_remainder_and_full_cartons_l248_24884

def abigail_eggs := 48
def beatrice_eggs := 63
def carson_eggs := 27
def carton_size := 15

theorem eggs_remainder_and_full_cartons :
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  ∃ (full_cartons left_over : ℕ),
    total_eggs = full_cartons * carton_size + left_over ∧
    left_over = 3 ∧
    full_cartons = 9 :=
by
  sorry

end eggs_remainder_and_full_cartons_l248_24884


namespace sum_of_modified_numbers_l248_24824

theorem sum_of_modified_numbers (x y R : ℝ) (h : x + y = R) : 
  2 * (x + 4) + 2 * (y + 5) = 2 * R + 18 :=
by
  sorry

end sum_of_modified_numbers_l248_24824


namespace vasya_numbers_l248_24828

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l248_24828


namespace gross_profit_percentage_l248_24848

theorem gross_profit_percentage :
  ∀ (selling_price wholesale_cost : ℝ),
  selling_price = 28 →
  wholesale_cost = 24.14 →
  (selling_price - wholesale_cost) / wholesale_cost * 100 = 15.99 :=
by
  intros selling_price wholesale_cost h1 h2
  rw [h1, h2]
  norm_num
  sorry

end gross_profit_percentage_l248_24848


namespace max_triangles_in_graph_l248_24875

def points : Finset Point := sorry
def no_coplanar (points : Finset Point) : Prop := sorry
def no_tetrahedron (points : Finset Point) : Prop := sorry
def triangles (points : Finset Point) : ℕ := sorry

theorem max_triangles_in_graph (points : Finset Point) 
  (H1 : points.card = 9) 
  (H2 : no_coplanar points) 
  (H3 : no_tetrahedron points) : 
  triangles points ≤ 27 := 
sorry

end max_triangles_in_graph_l248_24875


namespace sports_popularity_order_l248_24887

theorem sports_popularity_order :
  let soccer := (13 : ℚ) / 40
  let baseball := (9 : ℚ) / 30
  let basketball := (7 : ℚ) / 20
  let volleyball := (3 : ℚ) / 10
  basketball > soccer ∧ soccer > baseball ∧ baseball = volleyball :=
by
  sorry

end sports_popularity_order_l248_24887


namespace smallest_common_multiple_of_9_and_6_l248_24803

theorem smallest_common_multiple_of_9_and_6 : ∃ (n : ℕ), n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m := 
sorry

end smallest_common_multiple_of_9_and_6_l248_24803


namespace hyperbola_center_l248_24839

theorem hyperbola_center :
  ∃ (c : ℝ × ℝ), c = (3, 5) ∧
  (9 * (x - c.1)^2 - 36 * (y - c.2)^2 - (1244 - 243 - 1001) = 0) :=
sorry

end hyperbola_center_l248_24839


namespace length_of_train_l248_24885

-- declare constants
variables (L S : ℝ)

-- state conditions
def condition1 : Prop := L = S * 50
def condition2 : Prop := L + 500 = S * 100

-- state the theorem to prove
theorem length_of_train (h1 : condition1 L S) (h2 : condition2 L S) : L = 500 :=
by sorry

end length_of_train_l248_24885


namespace overall_percent_change_l248_24862

theorem overall_percent_change (W : ℝ) : 
  (W * 0.6 * 1.3 * 0.8 * 1.1) / W = 0.624 :=
by {
  sorry
}

end overall_percent_change_l248_24862


namespace eliminate_denominator_correctness_l248_24844

-- Define the initial equality with fractions
def initial_equation (x : ℝ) := (2 * x - 3) / 5 = (2 * x) / 3 - 3

-- Define the resulting expression after eliminating the denominators
def eliminated_denominators (x : ℝ) := 3 * (2 * x - 3) = 5 * 2 * x - 3 * 15

-- The theorem states that given the initial equation, the eliminated denomination expression holds true
theorem eliminate_denominator_correctness (x : ℝ) :
  initial_equation x → eliminated_denominators x := by
  sorry

end eliminate_denominator_correctness_l248_24844


namespace infinite_double_perfect_squares_l248_24882

def is_double_number (n : ℕ) : Prop :=
  ∃ k m : ℕ, m > 0 ∧ n = m * 10^k + m

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem infinite_double_perfect_squares : ∀ n : ℕ, ∃ m, n < m ∧ is_double_number m ∧ is_perfect_square m :=
  sorry

end infinite_double_perfect_squares_l248_24882


namespace sum_f_alpha_beta_gamma_neg_l248_24819

theorem sum_f_alpha_beta_gamma_neg (f : ℝ → ℝ)
  (h_f : ∀ x, f x = -x - x^3)
  (α β γ : ℝ)
  (h1 : α + β > 0)
  (h2 : β + γ > 0)
  (h3 : γ + α > 0) :
  f α + f β + f γ < 0 := 
sorry

end sum_f_alpha_beta_gamma_neg_l248_24819


namespace sufficient_but_not_necessary_l248_24851

variable (p q : Prop)

theorem sufficient_but_not_necessary : (¬p → ¬(p ∧ q)) ∧ (¬(¬p) → ¬(p ∧ q) → False) :=
by {
  sorry
}

end sufficient_but_not_necessary_l248_24851


namespace evaluate_expression_l248_24818

/-
  Define the expressions from the conditions.
  We define the numerator and denominator separately.
-/
def expr_numerator : ℚ := 1 - (1 / 4)
def expr_denominator : ℚ := 1 - (1 / 3)

/-
  Define the original expression to be proven.
  This is our main expression to evaluate.
-/
def expr : ℚ := expr_numerator / expr_denominator

/-
  State the final proof problem that the expression is equal to 9/8.
-/
theorem evaluate_expression : expr = 9 / 8 := sorry

end evaluate_expression_l248_24818


namespace remainder_expression_l248_24808

theorem remainder_expression (x y u v : ℕ) (hy_pos : y > 0) (h : x = u * y + v) (hv : 0 ≤ v) (hv_lt : v < y) :
  (x + 4 * u * y) % y = v :=
by
  sorry

end remainder_expression_l248_24808


namespace increasing_interval_of_f_l248_24890

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

theorem increasing_interval_of_f :
  ∀ x, x ∈ Set.Icc (-3 * Real.pi / 4) (Real.pi / 4) → MonotoneOn f (Set.Icc (-3 * Real.pi / 4) (Real.pi / 4)) :=
by
  sorry

end increasing_interval_of_f_l248_24890


namespace find_k_for_circle_radius_l248_24831

theorem find_k_for_circle_radius (k : ℝ) :
  (∃ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ∧ (x + 7)^2 + (y + 4)^2 = 10^2) ↔ k = 35 :=
by
  sorry

end find_k_for_circle_radius_l248_24831


namespace area_of_inscribed_triangle_l248_24855

noncomputable def area_of_triangle_inscribed_in_circle_with_arcs (a b c : ℕ) := 
  let circum := a + b + c
  let r := circum / (2 * Real.pi)
  let θ := 360 / (a + b + c)
  let angle1 := 4 * θ
  let angle2 := 6 * θ
  let angle3 := 8 * θ
  let sin80 := Real.sin (80 * Real.pi / 180)
  let sin120 := Real.sin (120 * Real.pi / 180)
  let sin160 := Real.sin (160 * Real.pi / 180)
  let approx_vals := sin80 + sin120 + sin160
  (1 / 2) * r^2 * approx_vals

theorem area_of_inscribed_triangle : 
  area_of_triangle_inscribed_in_circle_with_arcs 4 6 8 = 90.33 / Real.pi^2 :=
by sorry

end area_of_inscribed_triangle_l248_24855


namespace painting_clock_57_painting_clock_1913_l248_24856

-- Part (a)
theorem painting_clock_57 (h : ∀ n : ℕ, (n = 12 ∨ (exists k : ℕ, n = (12 + k * 57) % 12))) :
  ∃ m : ℕ, m = 4 :=
by { sorry }

-- Part (b)
theorem painting_clock_1913 (h : ∀ n : ℕ, (n = 12 ∨ (exists k : ℕ, n = (12 + k * 1913) % 12))) :
  ∃ m : ℕ, m = 12 :=
by { sorry }

end painting_clock_57_painting_clock_1913_l248_24856


namespace point_coordinates_l248_24893

-- Definitions based on conditions
def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0
def dist_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop := abs P.1 = d

-- Lean 4 statement
theorem point_coordinates {P : ℝ × ℝ} (h1 : on_x_axis P) (h2 : dist_to_y_axis P 3) :
  P = (3, 0) ∨ P = (-3, 0) :=
by sorry

end point_coordinates_l248_24893


namespace chosen_number_is_30_l248_24892

theorem chosen_number_is_30 (x : ℤ) 
  (h1 : 8 * x - 138 = 102) : x = 30 := 
sorry

end chosen_number_is_30_l248_24892


namespace common_factor_polynomials_l248_24854

theorem common_factor_polynomials (a : ℝ) :
  (∀ p : ℝ, p ≠ 0 ∧ 
           (p^3 - p - a = 0) ∧ 
           (p^2 + p - a = 0)) → 
  (a = 0 ∨ a = 10 ∨ a = -2) := by
  sorry

end common_factor_polynomials_l248_24854


namespace battery_life_in_standby_l248_24841

noncomputable def remaining_battery_life (b_s : ℝ) (b_a : ℝ) (t_total : ℝ) (t_active : ℝ) : ℝ :=
  let standby_rate := 1 / b_s
  let active_rate := 1 / b_a
  let standby_time := t_total - t_active
  let consumption_active := t_active * active_rate
  let consumption_standby := standby_time * standby_rate
  let total_consumption := consumption_active + consumption_standby
  let remaining_battery := 1 - total_consumption
  remaining_battery * b_s

theorem battery_life_in_standby :
  remaining_battery_life 30 4 10 1.5 = 10.25 := sorry

end battery_life_in_standby_l248_24841


namespace three_lines_pass_through_point_and_intersect_parabola_l248_24898

-- Define the point (0,1)
def point : ℝ × ℝ := (0, 1)

-- Define the parabola y^2 = 4x as a set of points
def parabola (p : ℝ × ℝ) : Prop :=
  (p.snd)^2 = 4 * (p.fst)

-- Define the condition for the line passing through (0,1)
def line_through_point (line_eq : ℝ → ℝ) : Prop :=
  line_eq 0 = 1

-- Define the condition for the line intersecting the parabola at only one point
def intersects_once (line_eq : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, parabola (x, line_eq x)

-- The main theorem statement
theorem three_lines_pass_through_point_and_intersect_parabola :
  ∃ (f1 f2 f3 : ℝ → ℝ), 
    line_through_point f1 ∧ line_through_point f2 ∧ line_through_point f3 ∧
    intersects_once f1 ∧ intersects_once f2 ∧ intersects_once f3 ∧
    (∀ (f : ℝ → ℝ), (line_through_point f ∧ intersects_once f) ->
      (f = f1 ∨ f = f2 ∨ f = f3)) :=
sorry

end three_lines_pass_through_point_and_intersect_parabola_l248_24898


namespace car_first_hour_speed_l248_24886

theorem car_first_hour_speed
  (x speed2 : ℝ)
  (avgSpeed : ℝ)
  (h_speed2 : speed2 = 60)
  (h_avgSpeed : avgSpeed = 35) :
  (avgSpeed = (x + speed2) / 2) → x = 10 :=
by
  sorry

end car_first_hour_speed_l248_24886


namespace set_theorem_l248_24833

noncomputable def set_A : Set ℕ := {1, 2}
noncomputable def set_B : Set ℕ := {1, 2, 3}
noncomputable def set_C : Set ℕ := {2, 3, 4}

theorem set_theorem : (set_A ∩ set_B) ∪ set_C = {1, 2, 3, 4} := by
  sorry

end set_theorem_l248_24833


namespace sin_double_alpha_l248_24815

variable (α β : ℝ)

theorem sin_double_alpha (h1 : Real.pi / 2 < β ∧ β < α ∧ α < 3 * Real.pi / 4)
        (h2 : Real.cos (α - β) = 12 / 13) 
        (h3 : Real.sin (α + β) = -3 / 5) : 
        Real.sin (2 * α) = -56 / 65 := by
  sorry

end sin_double_alpha_l248_24815


namespace integer_solutions_equation_l248_24868

theorem integer_solutions_equation : 
  (∃ x y : ℤ, (1 / (2022 : ℚ) = 1 / (x : ℚ) + 1 / (y : ℚ))) → 
  ∃! (n : ℕ), n = 53 :=
by
  sorry

end integer_solutions_equation_l248_24868


namespace max_rect_area_with_given_perimeter_l248_24826

-- Define the variables used in the problem
def length_of_wire := 12
def max_area (x : ℝ) := -(x - 3)^2 + 9

-- Lean Statement for the problem
theorem max_rect_area_with_given_perimeter : ∃ (A : ℝ), (∀ (x : ℝ), 0 < x ∧ x < 6 → (x * (6 - x) ≤ A)) ∧ A = 9 :=
by
  sorry

end max_rect_area_with_given_perimeter_l248_24826


namespace find_LN_l248_24829

noncomputable def LM : ℝ := 25
noncomputable def sinN : ℝ := 4 / 5

theorem find_LN (LN : ℝ) (h_sin : sinN = LM / LN) : LN = 125 / 4 :=
by
  sorry

end find_LN_l248_24829


namespace find_a_if_f_even_l248_24860

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (((2 * x) - 1) / ((2 * x) + 1))

theorem find_a_if_f_even (a : ℝ) :
  (∀ x : ℝ, (x > 1/2 ∨ x < -1/2) → f x a = f (-x) a) → a = 0 :=
by
  intro h1
  -- This is where the mathematical proof would go, but it's omitted as per the requirements.
  sorry

end find_a_if_f_even_l248_24860


namespace smallest_three_digit_number_with_property_l248_24873

theorem smallest_three_digit_number_with_property :
  ∃ a : ℕ, 100 ≤ a ∧ a < 1000 ∧ ∃ n : ℕ, 1000 * a + (a + 1) = n^2 ∧ a = 183 :=
sorry

end smallest_three_digit_number_with_property_l248_24873


namespace cricketer_average_increase_l248_24837

theorem cricketer_average_increase (A : ℝ) (H1 : 18 * A + 98 = 19 * 26) :
  26 - A = 4 :=
by
  sorry

end cricketer_average_increase_l248_24837


namespace find_F_l248_24849

theorem find_F (F C : ℝ) (hC_eq : C = (4/7) * (F - 40)) (hC_val : C = 35) : F = 101.25 :=
by
  sorry

end find_F_l248_24849


namespace necessary_but_not_sufficient_condition_l248_24812

variable (A B C : Set α) (a : α)
variable [Nonempty α]
variable (H1 : ∀ a, a ∈ A ↔ (a ∈ B ∧ a ∈ C))

theorem necessary_but_not_sufficient_condition :
  (a ∈ B → a ∈ A) ∧ ¬(a ∈ A → a ∈ B) :=
by
  sorry

end necessary_but_not_sufficient_condition_l248_24812


namespace pillows_from_feathers_l248_24883

def feathers_per_pound : ℕ := 300
def feathers_total : ℕ := 3600
def pounds_per_pillow : ℕ := 2

theorem pillows_from_feathers :
  (feathers_total / feathers_per_pound / pounds_per_pillow) = 6 :=
by
  sorry

end pillows_from_feathers_l248_24883


namespace smallest_N_for_abs_x_squared_minus_4_condition_l248_24888

theorem smallest_N_for_abs_x_squared_minus_4_condition (x : ℝ) 
  (h : abs (x - 2) < 0.01) : abs (x^2 - 4) < 0.0401 := 
sorry

end smallest_N_for_abs_x_squared_minus_4_condition_l248_24888


namespace ab_greater_than_a_plus_b_l248_24810

variable {a b : ℝ}
variables (pos_a : 0 < a) (pos_b : 0 < b) (h : a - b = a / b)

theorem ab_greater_than_a_plus_b : a * b > a + b :=
sorry

end ab_greater_than_a_plus_b_l248_24810


namespace no_consecutive_even_square_and_three_times_square_no_consecutive_square_and_seven_times_square_l248_24881

-- Problem 1: Square of an even number followed by three times a square number
theorem no_consecutive_even_square_and_three_times_square :
  ∀ (k n : ℕ), ¬(3 * n ^ 2 = 4 * k ^ 2 + 1) :=
by sorry

-- Problem 2: Square number followed by seven times another square number
theorem no_consecutive_square_and_seven_times_square :
  ∀ (r s : ℕ), ¬(7 * s ^ 2 = r ^ 2 + 1) :=
by sorry

end no_consecutive_even_square_and_three_times_square_no_consecutive_square_and_seven_times_square_l248_24881


namespace find_speed_grocery_to_gym_l248_24896

variables (v : ℝ) (speed_grocery_to_gym : ℝ)
variables (d_home_to_grocery : ℝ) (d_grocery_to_gym : ℝ)
variables (time_diff : ℝ)

def problem_conditions : Prop :=
  d_home_to_grocery = 840 ∧
  d_grocery_to_gym = 480 ∧
  time_diff = 40 ∧
  speed_grocery_to_gym = 2 * v

def correct_answer : Prop :=
  speed_grocery_to_gym = 30

theorem find_speed_grocery_to_gym :
  problem_conditions v speed_grocery_to_gym d_home_to_grocery d_grocery_to_gym time_diff →
  correct_answer speed_grocery_to_gym :=
by
  sorry

end find_speed_grocery_to_gym_l248_24896


namespace intersection_points_on_circle_l248_24861

theorem intersection_points_on_circle
  (x y : ℝ)
  (h1 : y = (x + 2)^2)
  (h2 : x + 2 = (y - 1)^2) :
  (x + 2)^2 + (y - 1)^2 = 2 :=
sorry

end intersection_points_on_circle_l248_24861


namespace number_of_arrangements_l248_24801

def basil_plants := 2
def aloe_plants := 1
def cactus_plants := 1
def white_lamps := 2
def red_lamps := 2
def total_plants := basil_plants + aloe_plants + cactus_plants
def total_lamps := white_lamps + red_lamps

theorem number_of_arrangements : total_plants = 4 ∧ total_lamps = 4 →
  ∃ n : ℕ, n = 28 :=
by
  intro h
  sorry

end number_of_arrangements_l248_24801


namespace points_scored_fourth_game_l248_24809

-- Define the conditions
def avg_score_3_games := 18
def avg_score_4_games := 17
def games_played_3 := 3
def games_played_4 := 4

-- Calculate total points after 3 games
def total_points_3_games := avg_score_3_games * games_played_3

-- Calculate total points after 4 games
def total_points_4_games := avg_score_4_games * games_played_4

-- Define a theorem to prove the points scored in the fourth game
theorem points_scored_fourth_game :
  total_points_4_games - total_points_3_games = 14 :=
by
  sorry

end points_scored_fourth_game_l248_24809


namespace shaded_area_l248_24864

theorem shaded_area (area_large : ℝ) (area_small : ℝ) (n_small_squares : ℕ) 
  (n_triangles: ℕ) (area_total : ℝ) : 
  area_large = 16 → 
  area_small = 1 → 
  n_small_squares = 4 → 
  n_triangles = 4 → 
  area_total = 4 → 
  4 * area_small = 4 →
  area_large - (area_total + (n_small_squares * area_small)) = 4 :=
by
  intros
  sorry

end shaded_area_l248_24864


namespace quadrilateral_area_l248_24866

structure Point :=
  (x : ℝ)
  (y : ℝ)

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def area_of_quadrilateral (A B C D : Point) : ℝ :=
  area_of_triangle A B C + area_of_triangle A C D

def A : Point := ⟨2, 2⟩
def B : Point := ⟨2, -1⟩
def C : Point := ⟨3, -1⟩
def D : Point := ⟨2007, 2008⟩

theorem quadrilateral_area :
  area_of_quadrilateral A B C D = 2008006.5 :=
by
  sorry

end quadrilateral_area_l248_24866


namespace sampling_methods_correct_l248_24899

def condition1 : Prop :=
  ∃ yogurt_boxes : ℕ, yogurt_boxes = 10 ∧ ∃ sample_boxes : ℕ, sample_boxes = 3

def condition2 : Prop :=
  ∃ rows seats_per_row attendees sample_size : ℕ,
    rows = 32 ∧ seats_per_row = 40 ∧ attendees = rows * seats_per_row ∧ sample_size = 32

def condition3 : Prop :=
  ∃ liberal_arts_classes science_classes total_classes sample_size : ℕ,
    liberal_arts_classes = 4 ∧ science_classes = 8 ∧ total_classes = liberal_arts_classes + science_classes ∧ sample_size = 50

def simple_random_sampling (s : Prop) : Prop := sorry -- definition for simple random sampling
def systematic_sampling (s : Prop) : Prop := sorry -- definition for systematic sampling
def stratified_sampling (s : Prop) : Prop := sorry -- definition for stratified sampling

theorem sampling_methods_correct :
  (condition1 → simple_random_sampling condition1) ∧
  (condition2 → systematic_sampling condition2) ∧
  (condition3 → stratified_sampling condition3) :=
by {
  sorry
}

end sampling_methods_correct_l248_24899


namespace intersection_of_M_and_N_l248_24807

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | x > 2 ∨ x < -2}
def expected_intersection : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_of_M_and_N : M ∩ N = expected_intersection := by
  sorry

end intersection_of_M_and_N_l248_24807


namespace find_amount_l248_24872

theorem find_amount (x : ℝ) (h1 : 0.25 * x = 0.15 * 1500 - 30) (h2 : x = 780) : 30 = 30 :=
by
  sorry

end find_amount_l248_24872


namespace probability_two_same_number_l248_24835

theorem probability_two_same_number :
  let rolls := 5
  let sides := 8
  let total_outcomes := sides ^ rolls
  let favorable_outcomes := 8 * 7 * 6 * 5 * 4
  let probability_all_different := (favorable_outcomes : ℚ) / total_outcomes
  let probability_at_least_two_same := 1 - probability_all_different
  probability_at_least_two_same = (3256 : ℚ) / 4096 :=
by 
  sorry

end probability_two_same_number_l248_24835


namespace percent_increase_equilateral_triangles_l248_24843

theorem percent_increase_equilateral_triangles :
  let s₁ := 3
  let s₂ := 2 * s₁
  let s₃ := 2 * s₂
  let s₄ := 2 * s₃
  let P₁ := 3 * s₁
  let P₄ := 3 * s₄
  (P₄ - P₁) / P₁ * 100 = 700 :=
by
  sorry

end percent_increase_equilateral_triangles_l248_24843


namespace intersection_point_l248_24838

theorem intersection_point (k : ℚ) :
  (∃ x y : ℚ, x + k * y = 0 ∧ 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0) ↔ (k = -1/2) :=
by sorry

end intersection_point_l248_24838


namespace ab_value_in_triangle_l248_24811

theorem ab_value_in_triangle (a b c : ℝ) (C : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : C = 60) :
  a * b = 4 / 3 :=
by sorry

end ab_value_in_triangle_l248_24811


namespace polynomial_degree_rational_coefficients_l248_24857

theorem polynomial_degree_rational_coefficients :
  ∃ p : Polynomial ℚ,
    (Polynomial.aeval (2 - 3 * Real.sqrt 3) p = 0) ∧
    (Polynomial.aeval (-2 - 3 * Real.sqrt 3) p = 0) ∧
    (Polynomial.aeval (3 + Real.sqrt 11) p = 0) ∧
    (Polynomial.aeval (3 - Real.sqrt 11) p = 0) ∧
    p.degree = 6 :=
sorry

end polynomial_degree_rational_coefficients_l248_24857


namespace min_value_a2_b2_l248_24874

theorem min_value_a2_b2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (h : a^2 - 2015 * a = b^2 - 2015 * b) : 
  a^2 + b^2 ≥ 2015^2 / 2 := 
sorry

end min_value_a2_b2_l248_24874


namespace sum_A_B_l248_24840

theorem sum_A_B (A B : ℕ) 
  (h1 : (1 / 4 : ℚ) * (1 / 8) = 1 / (4 * A))
  (h2 : 1 / (4 * A) = 1 / B) : A + B = 40 := 
by
  sorry

end sum_A_B_l248_24840


namespace watermelons_left_l248_24802

theorem watermelons_left (initial : ℕ) (eaten : ℕ) (remaining : ℕ) (h1 : initial = 4) (h2 : eaten = 3) : remaining = 1 :=
by
  sorry

end watermelons_left_l248_24802


namespace compounded_rate_of_growth_l248_24814

theorem compounded_rate_of_growth (k m : ℝ) :
  (1 + k / 100) * (1 + m / 100) - 1 = ((k + m + (k * m / 100)) / 100) :=
by
  sorry

end compounded_rate_of_growth_l248_24814


namespace sum_of_digits_of_x_l248_24889

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_of_x (x : ℕ) (h1 : 100 ≤ x) (h2 : x ≤ 949)
  (h3 : is_palindrome x) (h4 : is_palindrome (x + 50)) :
  sum_of_digits x = 19 :=
sorry

end sum_of_digits_of_x_l248_24889


namespace sum_arithmetic_series_base8_l248_24825

theorem sum_arithmetic_series_base8 : 
  let n := 36
  let a := 1
  let l := 30 -- 36_8 in base 10 is 30
  let S := (n * (a + l)) / 2
  let sum_base10 := 558
  let sum_base8 := 1056 -- 558 in base 8 is 1056
  S = sum_base10 ∧ sum_base10 = 1056 :=
by
  sorry

end sum_arithmetic_series_base8_l248_24825


namespace no_solution_xyz_l248_24878

theorem no_solution_xyz : ∀ (x y z : Nat), (1 ≤ x) → (x ≤ 9) → (0 ≤ y) → (y ≤ 9) → (0 ≤ z) → (z ≤ 9) →
    100 * x + 10 * y + z ≠ 10 * x * y + x * z :=
by
  intros x y z hx1 hx9 hy1 hy9 hz1 hz9
  sorry

end no_solution_xyz_l248_24878


namespace max_and_min_A_l248_24823

noncomputable def B := {B : ℕ // B > 22222222 ∧ gcd B 18 = 1}
noncomputable def A (B : B) : ℕ := 10^8 * ((B.val % 10)) + (B.val / 10)

noncomputable def A_max := 999999998
noncomputable def A_min := 122222224

theorem max_and_min_A : 
  (∃ B : B, A B = A_max) ∧ (∃ B : B, A B = A_min) := sorry

end max_and_min_A_l248_24823


namespace dimes_left_l248_24832

-- Definitions based on the conditions
def Initial_dimes : ℕ := 8
def Sister_borrowed : ℕ := 4
def Friend_borrowed : ℕ := 2

-- The proof problem statement (without the proof)
theorem dimes_left (Initial_dimes Sister_borrowed Friend_borrowed : ℕ) : 
  Initial_dimes = 8 → Sister_borrowed = 4 → Friend_borrowed = 2 →
  Initial_dimes - (Sister_borrowed + Friend_borrowed) = 2 :=
by
  intros
  sorry

end dimes_left_l248_24832


namespace lake_width_l248_24850

theorem lake_width
  (W : ℝ)
  (janet_speed : ℝ) (sister_speed : ℝ) (wait_time : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : wait_time = 3)
  (h4 : W / sister_speed = W / janet_speed + wait_time) :
  W = 60 := 
sorry

end lake_width_l248_24850


namespace kamal_chemistry_marks_l248_24816

variables (english math physics biology average total numSubjects : ℕ)

theorem kamal_chemistry_marks 
  (marks_in_english : english = 66)
  (marks_in_math : math = 65)
  (marks_in_physics : physics = 77)
  (marks_in_biology : biology = 75)
  (avg_marks : average = 69)
  (number_of_subjects : numSubjects = 5)
  (total_marks_known : total = 283) :
  ∃ chemistry : ℕ, chemistry = 62 := 
by 
  sorry

end kamal_chemistry_marks_l248_24816


namespace jose_share_is_correct_l248_24842

noncomputable def total_profit : ℝ := 
  5000 - 2000 + 7000 + 1000 - 3000 + 10000 + 500 + 4000 - 2500 + 6000 + 8000 - 1000

noncomputable def tom_investment_ratio : ℝ := 30000 * 12
noncomputable def jose_investment_ratio : ℝ := 45000 * 10
noncomputable def maria_investment_ratio : ℝ := 60000 * 8

noncomputable def total_investment_ratio : ℝ := tom_investment_ratio + jose_investment_ratio + maria_investment_ratio

noncomputable def jose_share : ℝ := (jose_investment_ratio / total_investment_ratio) * total_profit

theorem jose_share_is_correct : jose_share = 14658 := 
by 
  sorry

end jose_share_is_correct_l248_24842


namespace yellow_pill_cost_22_5_l248_24846

-- Definitions based on conditions
def number_of_days := 3 * 7
def total_cost := 903
def daily_cost := total_cost / number_of_days
def blue_pill_cost (yellow_pill_cost : ℝ) := yellow_pill_cost - 2

-- Prove that the cost of one yellow pill is 22.5 dollars
theorem yellow_pill_cost_22_5 : 
  ∃ (yellow_pill_cost : ℝ), 
    number_of_days = 21 ∧
    total_cost = 903 ∧ 
    (∀ yellow_pill_cost, daily_cost = yellow_pill_cost + blue_pill_cost yellow_pill_cost → yellow_pill_cost = 22.5) :=
by 
  sorry

end yellow_pill_cost_22_5_l248_24846


namespace geometric_seq_a9_l248_24863

theorem geometric_seq_a9 
  (a : ℕ → ℤ)  -- The sequence definition
  (h_geometric : ∀ n : ℕ, a (n+1) = a 1 * (a 2 ^ n) / a 1 ^ n)  -- Geometric sequence property
  (h_a1 : a 1 = 2)  -- Given a₁ = 2
  (h_a5 : a 5 = 18)  -- Given a₅ = 18
: a 9 = 162 := sorry

end geometric_seq_a9_l248_24863


namespace sum_of_repeating_decimals_l248_24876

noncomputable def x := (2 : ℚ) / (3 : ℚ)
noncomputable def y := (5 : ℚ) / (11 : ℚ)

theorem sum_of_repeating_decimals : x + y = (37 : ℚ) / (33 : ℚ) :=
by {
  sorry
}

end sum_of_repeating_decimals_l248_24876


namespace find_a_l248_24817

-- Define the conditions of the problem
def line1 (a : ℝ) : ℝ × ℝ × ℝ := (a + 3, 1, -3) -- Coefficients of line1: (a+3)x + y - 3 = 0
def line2 (a : ℝ) : ℝ × ℝ × ℝ := (5, a - 3, 4)  -- Coefficients of line2: 5x + (a-3)y + 4 = 0

-- Definition of direction vector and normal vector
def direction_vector (a : ℝ) : ℝ × ℝ := (1, -(a + 3))
def normal_vector (a : ℝ) : ℝ × ℝ := (5, a - 3)

-- Proof statement
theorem find_a (a : ℝ) : (direction_vector a = normal_vector a) → a = -2 :=
by {
  -- Insert proof here
  sorry
}

end find_a_l248_24817


namespace terminating_decimals_nat_l248_24865

theorem terminating_decimals_nat (n : ℕ) (h1 : ∃ a b : ℕ, n = 2^a * 5^b)
  (h2 : ∃ c d : ℕ, n + 1 = 2^c * 5^d) : n = 1 ∨ n = 4 :=
by
  sorry

end terminating_decimals_nat_l248_24865


namespace elves_closed_eyes_l248_24806

theorem elves_closed_eyes :
  ∃ (age: ℕ → ℕ), -- Function assigning each position an age
  (∀ n, 1 ≤ n ∧ n ≤ 100 → (age n < age ((n % 100) + 1) ∧ age n < age (n - 1 % 100 + 1)) ∨
                          (age n > age ((n % 100) + 1) ∧ age n > age (n - 1 % 100 + 1))) :=
by
  sorry

end elves_closed_eyes_l248_24806


namespace smallest_n_l248_24847

theorem smallest_n (n : ℕ) (h1 : n > 2016) (h2 : (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0) : n = 2020 :=
sorry

end smallest_n_l248_24847


namespace additional_chair_frequency_l248_24879

theorem additional_chair_frequency 
  (workers : ℕ)
  (chairs_per_worker_per_hour : ℕ)
  (hours : ℕ)
  (total_chairs : ℕ) 
  (additional_chairs_rate : ℕ)
  (h_workers : workers = 3) 
  (h_chairs_per_worker : chairs_per_worker_per_hour = 4) 
  (h_hours : hours = 6 ) 
  (h_total_chairs : total_chairs = 73) :
  additional_chairs_rate = 6 :=
by
  sorry

end additional_chair_frequency_l248_24879


namespace geometric_sequence_problem_l248_24867

variable {a : ℕ → ℝ}

-- Given conditions
def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q r, (∀ n, a (n + 1) = q * a n ∧ a 0 = r)

-- Define the conditions from the problem
def condition1 (a : ℕ → ℝ) :=
  a 3 + a 6 = 6

def condition2 (a : ℕ → ℝ) :=
  a 5 + a 8 = 9

-- Theorem to be proved
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (hgeom : geometric_sequence a)
  (h1 : condition1 a)
  (h2 : condition2 a) :
  a 7 + a 10 = 27 / 2 :=
sorry

end geometric_sequence_problem_l248_24867


namespace sale_in_first_month_l248_24894

theorem sale_in_first_month (sale1 sale2 sale3 sale4 sale5 : ℕ) 
  (h1 : sale1 = 5660) (h2 : sale2 = 6200) (h3 : sale3 = 6350) (h4 : sale4 = 6500) 
  (h_avg : (sale1 + sale2 + sale3 + sale4 + sale5) / 5 = 6000) : 
  sale5 = 5290 := 
by
  sorry

end sale_in_first_month_l248_24894


namespace evaluate_expression_l248_24880

theorem evaluate_expression (x y z : ℚ) (hx : x = 1 / 2) (hy : y = 1 / 3) (hz : z = -3) :
  (2 * x)^2 * (y^2)^3 * z^2 = 1 / 81 :=
by
  -- Proof omitted
  sorry

end evaluate_expression_l248_24880


namespace largest_among_options_l248_24859

theorem largest_among_options :
  let A := 15679 + (1 / 3579)
  let B := 15679 - (1 / 3579)
  let C := 15679 * (1 / 3579)
  let D := 15679 / (1 / 3579)
  let E := 15679 * 1.03
  D > A ∧ D > B ∧ D > C ∧ D > E := by
{
  let A := 15679 + (1 / 3579)
  let B := 15679 - (1 / 3579)
  let C := 15679 * (1 / 3579)
  let D := 15679 / (1 / 3579)
  let E := 15679 * 1.03
  sorry
}

end largest_among_options_l248_24859


namespace circle_outside_hexagon_area_l248_24834

theorem circle_outside_hexagon_area :
  let r := (Real.sqrt 2) / 2
  let s := 1
  let area_circle := π * r^2
  let area_hexagon := 3 * Real.sqrt 3 / 2 * s^2
  area_circle - area_hexagon = (π / 2) - (3 * Real.sqrt 3 / 2) :=
by
  sorry

end circle_outside_hexagon_area_l248_24834


namespace find_unique_positive_integer_pair_l248_24820

theorem find_unique_positive_integer_pair :
  ∃! (b c : ℕ), b > 0 ∧ c > 0 ∧ c > b^2 ∧ b > c^2 :=
sorry

end find_unique_positive_integer_pair_l248_24820


namespace sum_of_decimals_l248_24871

theorem sum_of_decimals :
  let a := 0.35
  let b := 0.048
  let c := 0.0072
  a + b + c = 0.4052 := by
  sorry

end sum_of_decimals_l248_24871
