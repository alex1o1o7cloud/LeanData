import Mathlib

namespace NUMINAMATH_GPT_total_questions_needed_l2352_235201

def m_total : ℕ := 35
def p_total : ℕ := 15
def t_total : ℕ := 20

def m_written : ℕ := (3 * m_total) / 7
def p_written : ℕ := p_total / 5
def t_written : ℕ := t_total / 4

def m_remaining : ℕ := m_total - m_written
def p_remaining : ℕ := p_total - p_written
def t_remaining : ℕ := t_total - t_written

def total_remaining : ℕ := m_remaining + p_remaining + t_remaining

theorem total_questions_needed : total_remaining = 47 := by
  sorry

end NUMINAMATH_GPT_total_questions_needed_l2352_235201


namespace NUMINAMATH_GPT_ticket_is_five_times_soda_l2352_235210

variable (p_i p_r : ℝ)

theorem ticket_is_five_times_soda
  (h1 : 6 * p_i + 20 * p_r = 50)
  (h2 : 6 * p_r = p_i + p_r) : p_i = 5 * p_r :=
sorry

end NUMINAMATH_GPT_ticket_is_five_times_soda_l2352_235210


namespace NUMINAMATH_GPT_perfect_square_l2352_235252

-- Define natural numbers m and n and the condition mn ∣ m^2 + n^2 + m
variables (m n : ℕ)

-- Define the condition as a hypothesis
def condition (m n : ℕ) : Prop := (m * n) ∣ (m ^ 2 + n ^ 2 + m)

-- The main theorem statement: if the condition holds, then m is a perfect square
theorem perfect_square (m n : ℕ) (h : condition m n) : ∃ k : ℕ, m = k ^ 2 :=
sorry

end NUMINAMATH_GPT_perfect_square_l2352_235252


namespace NUMINAMATH_GPT_marbles_total_l2352_235275

def marbles_initial := 22
def marbles_given := 20

theorem marbles_total : marbles_initial + marbles_given = 42 := by
  sorry

end NUMINAMATH_GPT_marbles_total_l2352_235275


namespace NUMINAMATH_GPT_men_work_equivalence_l2352_235230

theorem men_work_equivalence : 
  ∀ (M : ℕ) (m w : ℕ),
  (3 * w = 2 * m) ∧ 
  (M * 21 * 8 * m = 21 * 60 * 3 * w) →
  M = 15 := by
  intro M m w
  intro h
  sorry

end NUMINAMATH_GPT_men_work_equivalence_l2352_235230


namespace NUMINAMATH_GPT_line_equation_parametric_to_implicit_l2352_235287

theorem line_equation_parametric_to_implicit (t : ℝ) :
  ∀ x y : ℝ, (x = 3 * t + 6 ∧ y = 5 * t - 7) → y = (5 / 3) * x - 17 :=
by
  intros x y h
  obtain ⟨hx, hy⟩ := h
  sorry

end NUMINAMATH_GPT_line_equation_parametric_to_implicit_l2352_235287


namespace NUMINAMATH_GPT_grazing_months_l2352_235202

theorem grazing_months
    (total_rent : ℝ)
    (c_rent : ℝ)
    (a_oxen : ℕ)
    (a_months : ℕ)
    (b_oxen : ℕ)
    (c_oxen : ℕ)
    (c_months : ℕ)
    (b_months : ℝ)
    (total_oxen_months : ℝ) :
    total_rent = 140 ∧
    c_rent = 36 ∧
    a_oxen = 10 ∧
    a_months = 7 ∧
    b_oxen = 12 ∧
    c_oxen = 15 ∧
    c_months = 3 ∧
    c_rent / total_rent = (c_oxen * c_months) / total_oxen_months ∧
    total_oxen_months = (a_oxen * a_months) + (b_oxen * b_months) + (c_oxen * c_months)
    → b_months = 5 := by
    sorry

end NUMINAMATH_GPT_grazing_months_l2352_235202


namespace NUMINAMATH_GPT_burt_net_profit_l2352_235261

theorem burt_net_profit
  (cost_seeds : ℝ := 2.00)
  (cost_soil : ℝ := 8.00)
  (num_plants : ℕ := 20)
  (price_per_plant : ℝ := 5.00) :
  let total_cost := cost_seeds + cost_soil
  let total_revenue := num_plants * price_per_plant
  let net_profit := total_revenue - total_cost
  net_profit = 90.00 :=
by sorry

end NUMINAMATH_GPT_burt_net_profit_l2352_235261


namespace NUMINAMATH_GPT_rank_from_right_l2352_235254

theorem rank_from_right (rank_from_left total_students : ℕ) (h1 : rank_from_left = 5) (h2 : total_students = 10) :
  total_students - rank_from_left + 1 = 6 :=
by 
  -- Placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_rank_from_right_l2352_235254


namespace NUMINAMATH_GPT_smallest_prime_perimeter_l2352_235229

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_scalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_triple_prime (a b c : ℕ) : Prop := is_prime a ∧ is_prime b ∧ is_prime c

def is_triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

theorem smallest_prime_perimeter :
  ∃ a b c : ℕ, is_scalene a b c ∧ is_triple_prime a b c ∧ is_prime (a + b + c) ∧ a + b + c = 23 :=
sorry

end NUMINAMATH_GPT_smallest_prime_perimeter_l2352_235229


namespace NUMINAMATH_GPT_rowing_time_l2352_235224

theorem rowing_time (rowing_speed : ℕ) (current_speed : ℕ) (distance : ℕ) 
  (h_rowing_speed : rowing_speed = 10)
  (h_current_speed : current_speed = 2)
  (h_distance : distance = 24) : 
  2 * distance / (rowing_speed + current_speed) + 2 * distance / (rowing_speed - current_speed) = 5 :=
by
  rw [h_rowing_speed, h_current_speed, h_distance]
  norm_num
  sorry

end NUMINAMATH_GPT_rowing_time_l2352_235224


namespace NUMINAMATH_GPT_basis_vetors_correct_options_l2352_235273

def is_basis (e1 e2 : ℝ × ℝ) : Prop :=
  e1 ≠ (0, 0) ∧ e2 ≠ (0, 0) ∧ e1.1 * e2.2 - e1.2 * e2.1 ≠ 0

def option_A : ℝ × ℝ := (0, 0)
def option_A' : ℝ × ℝ := (1, 2)

def option_B : ℝ × ℝ := (2, -1)
def option_B' : ℝ × ℝ := (1, 2)

def option_C : ℝ × ℝ := (-1, -2)
def option_C' : ℝ × ℝ := (1, 2)

def option_D : ℝ × ℝ := (1, 1)
def option_D' : ℝ × ℝ := (1, 2)

theorem basis_vetors_correct_options:
  ¬ is_basis option_A option_A' ∧ ¬ is_basis option_C option_C' ∧ 
  is_basis option_B option_B' ∧ is_basis option_D option_D' := 
by
  sorry

end NUMINAMATH_GPT_basis_vetors_correct_options_l2352_235273


namespace NUMINAMATH_GPT_find_s5_l2352_235299

noncomputable def s (a b x y : ℝ) (n : ℕ) : ℝ :=
if n = 1 then (a * x + b * y) else
if n = 2 then (a * x^2 + b * y^2) else
if n = 3 then (a * x^3 + b * y^3) else
if n = 4 then (a * x^4 + b * y^4) else
if n = 5 then (a * x^5 + b * y^5) else 0

theorem find_s5 
  (a b x y : ℝ) :
  s a b x y 1 = 5 →
  s a b x y 2 = 11 →
  s a b x y 3 = 24 →
  s a b x y 4 = 58 →
  s a b x y 5 = 262.88 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_s5_l2352_235299


namespace NUMINAMATH_GPT_tangent_line_at_1_tangent_line_through_2_3_l2352_235277

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- Derivative of the function
def f' (x : ℝ) : ℝ := 3 * x^2

-- Problem 1: Prove that the tangent line at point (1, 1) is y = 3x - 2
theorem tangent_line_at_1 (x y : ℝ) (h : y = f 1 + f' 1 * (x - 1)) : y = 3 * x - 2 := 
sorry

-- Problem 2: Prove that the tangent line passing through (2/3, 0) is either y = 0 or y = 3x - 2
theorem tangent_line_through_2_3 (x y x0 : ℝ) 
  (hx0 : y = f x0 + f' x0 * (x - x0))
  (hp : 0 = f' x0 * (2/3 - x0)) :
  y = 0 ∨ y = 3 * x - 2 := 
sorry

end NUMINAMATH_GPT_tangent_line_at_1_tangent_line_through_2_3_l2352_235277


namespace NUMINAMATH_GPT_range_of_set_l2352_235227

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end NUMINAMATH_GPT_range_of_set_l2352_235227


namespace NUMINAMATH_GPT_intersection_is_correct_l2352_235294

def setA := {x : ℝ | 3 * x - x^2 > 0}
def setB := {x : ℝ | x ≤ 1}

theorem intersection_is_correct : 
  setA ∩ setB = {x | 0 < x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_GPT_intersection_is_correct_l2352_235294


namespace NUMINAMATH_GPT_parabola_directrix_tangent_circle_l2352_235200

theorem parabola_directrix_tangent_circle (p : ℝ) (h_pos : 0 < p) (h_tangent: ∃ x : ℝ, (x = p/2) ∧ (x-5)^2 + (0:ℝ)^2 = 25) : p = 20 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_tangent_circle_l2352_235200


namespace NUMINAMATH_GPT_cosine_triangle_ABC_l2352_235228

noncomputable def triangle_cosine_proof (a b : ℝ) (A : ℝ) (cosB : ℝ) : Prop :=
  let sinA := Real.sin A
  let sinB := b * sinA / a
  let cosB_expr := Real.sqrt (1 - sinB^2)
  cosB = cosB_expr

theorem cosine_triangle_ABC : triangle_cosine_proof (Real.sqrt 7) 2 (Real.pi / 4) (Real.sqrt 35 / 7) :=
by
  sorry

end NUMINAMATH_GPT_cosine_triangle_ABC_l2352_235228


namespace NUMINAMATH_GPT_full_day_students_l2352_235219

def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

theorem full_day_students : 
  (total_students - (total_students * percentage_half_day_students / 100)) = 60 := by
  sorry

end NUMINAMATH_GPT_full_day_students_l2352_235219


namespace NUMINAMATH_GPT_extreme_values_l2352_235222

-- Define the function f(x) with symbolic constants a and b
def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x

-- Given conditions
def intersects_at_1_0 (a b : ℝ) : Prop := (f 1 a b = 0)
def derivative_at_1_0 (a b : ℝ) : Prop := (3 - 2 * a - b = 0)

-- Main theorem statement
theorem extreme_values (a b : ℝ) (h1 : intersects_at_1_0 a b) (h2 : derivative_at_1_0 a b) :
  (∀ x, f x a b ≤ 4 / 27) ∧ (∀ x, 0 ≤ f x a b) :=
sorry

end NUMINAMATH_GPT_extreme_values_l2352_235222


namespace NUMINAMATH_GPT_davonte_ran_further_than_mercedes_l2352_235291

-- Conditions
variable (jonathan_distance : ℝ) (mercedes_distance : ℝ) (davonte_distance : ℝ)

-- Given conditions
def jonathan_ran := jonathan_distance = 7.5
def mercedes_ran_twice_jonathan := mercedes_distance = 2 * jonathan_distance
def mercedes_and_davonte_total := mercedes_distance + davonte_distance = 32

-- Prove the distance Davonte ran farther than Mercedes is 2 kilometers
theorem davonte_ran_further_than_mercedes :
  jonathan_ran jonathan_distance ∧
  mercedes_ran_twice_jonathan jonathan_distance mercedes_distance ∧
  mercedes_and_davonte_total mercedes_distance davonte_distance →
  davonte_distance - mercedes_distance = 2 :=
by
  sorry

end NUMINAMATH_GPT_davonte_ran_further_than_mercedes_l2352_235291


namespace NUMINAMATH_GPT_find_N_l2352_235262

theorem find_N : ∃ (N : ℕ), (1000 ≤ N ∧ N < 10000) ∧ (N^2 % 10000 = N) ∧ (N % 16 = 7) ∧ N = 3751 := 
by sorry

end NUMINAMATH_GPT_find_N_l2352_235262


namespace NUMINAMATH_GPT_glove_selection_l2352_235247

theorem glove_selection :
  let n := 6                -- Number of pairs
  let k := 4                -- Number of selected gloves
  let m := 1                -- Number of matching pairs
  let total_ways := n * 10 * 8 / 2  -- Calculation based on solution steps
  total_ways = 240 := by
  sorry

end NUMINAMATH_GPT_glove_selection_l2352_235247


namespace NUMINAMATH_GPT_part_I_part_II_l2352_235297

open Real

noncomputable def f (x : ℝ) : ℝ := log ((2 / (x + 1)) - 1)

def g (x a : ℝ) : ℝ := -x^2 + 2 * x + a

-- Domain of function f
def A : Set ℝ := {x | -1 < x ∧ x < 1}

-- Range of function g with a given condition on x
def B (a : ℝ) : Set ℝ := {y | ∃ x, 0 ≤ x ∧ x ≤ 3 ∧ y = g x a}

theorem part_I : f (1 / 2015) + f (-1 / 2015) = 0 := sorry

theorem part_II (a : ℝ) : (A ∩ B a) = ∅ ↔ a ≤ -2 ∨ a ≥ 4 := sorry

end NUMINAMATH_GPT_part_I_part_II_l2352_235297


namespace NUMINAMATH_GPT_steps_from_center_to_square_l2352_235244

-- Define the conditions and question in Lean 4
def steps_to_center := 354
def total_steps := 582

-- Prove that the steps from Rockefeller Center to Times Square is 228
theorem steps_from_center_to_square : (total_steps - steps_to_center) = 228 := by
  sorry

end NUMINAMATH_GPT_steps_from_center_to_square_l2352_235244


namespace NUMINAMATH_GPT_right_triangle_sin_sum_l2352_235289

/--
In a right triangle ABC with ∠A = 90°, prove that sin A + sin^2 B + sin^2 C = 2.
-/
theorem right_triangle_sin_sum (A B C : ℝ) (hA : A = 90) (hABC : A + B + C = 180) :
  Real.sin (A * π / 180) + Real.sin (B * π / 180) ^ 2 + Real.sin (C * π / 180) ^ 2 = 2 :=
sorry

end NUMINAMATH_GPT_right_triangle_sin_sum_l2352_235289


namespace NUMINAMATH_GPT_smallest_value_of_y_l2352_235243

theorem smallest_value_of_y (x y z d : ℝ) (h1 : x = y - d) (h2 : z = y + d) (h3 : x * y * z = 125) (h4 : 0 < x ∧ 0 < y ∧ 0 < z) : y ≥ 5 :=
by
  -- Officially, the user should navigate through the proof, but we conclude with 'sorry' as placeholder
  sorry

end NUMINAMATH_GPT_smallest_value_of_y_l2352_235243


namespace NUMINAMATH_GPT_pages_per_chapter_l2352_235246

-- Definitions based on conditions
def chapters_in_book : ℕ := 2
def days_to_finish : ℕ := 664
def chapters_per_day : ℕ := 332
def total_chapters_read : ℕ := chapters_per_day * days_to_finish

-- Theorem that states the problem
theorem pages_per_chapter : total_chapters_read / chapters_in_book = 110224 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_pages_per_chapter_l2352_235246


namespace NUMINAMATH_GPT_inequality_relations_l2352_235264

variable {R : Type} [OrderedAddCommGroup R]
variables (x y z : R)

theorem inequality_relations (h1 : x - y > x + z) (h2 : x + y < y + z) : y < -z ∧ x < z :=
by
  sorry

end NUMINAMATH_GPT_inequality_relations_l2352_235264


namespace NUMINAMATH_GPT_one_meter_to_leaps_l2352_235288

theorem one_meter_to_leaps 
  (x y z w u v : ℕ)
  (h1 : x * leaps = y * strides) 
  (h2 : z * bounds = w * leaps) 
  (h3 : u * bounds = v * meters) :
  1 * meters = (uw / vz) * leaps :=
sorry

end NUMINAMATH_GPT_one_meter_to_leaps_l2352_235288


namespace NUMINAMATH_GPT_multiplication_as_sum_of_squares_l2352_235209

theorem multiplication_as_sum_of_squares :
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 := by
  sorry

end NUMINAMATH_GPT_multiplication_as_sum_of_squares_l2352_235209


namespace NUMINAMATH_GPT_bottle_cap_cost_l2352_235282

-- Define the conditions given in the problem.
def caps_cost (n : ℕ) (cost : ℝ) : Prop := n * cost = 12

-- Prove that the cost of each bottle cap is $2 given 6 bottle caps cost $12.
theorem bottle_cap_cost (h : caps_cost 6 cost) : cost = 2 :=
sorry

end NUMINAMATH_GPT_bottle_cap_cost_l2352_235282


namespace NUMINAMATH_GPT_intersection_of_sets_l2352_235240

open Set Real

theorem intersection_of_sets :
  let A := {x : ℝ | x^2 - 2*x - 3 < 0}
  let B := {y : ℝ | ∃ (x : ℝ), y = sin x}
  A ∩ B = Ioc (-1) 1 := by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l2352_235240


namespace NUMINAMATH_GPT_average_visitors_per_day_is_276_l2352_235295

-- Define the number of days in the month
def num_days_in_month : ℕ := 30

-- Define the number of Sundays in the month
def num_sundays_in_month : ℕ := 4

-- Define the number of other days in the month
def num_other_days_in_month : ℕ := num_days_in_month - num_sundays_in_month * 7 / 7 + 2

-- Define the average visitors on Sundays
def avg_visitors_sunday : ℕ := 510

-- Define the average visitors on other days
def avg_visitors_other_days : ℕ := 240

-- Calculate total visitors on Sundays
def total_visitors_sundays : ℕ := num_sundays_in_month * avg_visitors_sunday

-- Calculate total visitors on other days
def total_visitors_other_days : ℕ := num_other_days_in_month * avg_visitors_other_days

-- Calculate total visitors in the month
def total_visitors_in_month : ℕ := total_visitors_sundays + total_visitors_other_days

-- Given conditions, prove average visitors per day in a month
theorem average_visitors_per_day_is_276 :
  total_visitors_in_month / num_days_in_month = 276 := by
  sorry

end NUMINAMATH_GPT_average_visitors_per_day_is_276_l2352_235295


namespace NUMINAMATH_GPT_cagr_decline_l2352_235236

theorem cagr_decline 
  (EV BV : ℝ) (n : ℕ) 
  (h_ev : EV = 52)
  (h_bv : BV = 89)
  (h_n : n = 3)
: ((EV / BV) ^ (1 / n) - 1) = -0.1678 := 
by
  rw [h_ev, h_bv, h_n]
  sorry

end NUMINAMATH_GPT_cagr_decline_l2352_235236


namespace NUMINAMATH_GPT_min_time_to_complete_tasks_l2352_235239

-- Define the conditions as individual time durations for each task in minutes
def bed_making_time : ℕ := 3
def teeth_washing_time : ℕ := 4
def water_boiling_time : ℕ := 10
def breakfast_time : ℕ := 7
def dish_washing_time : ℕ := 1
def backpack_organizing_time : ℕ := 2
def milk_making_time : ℕ := 1

-- Define the total minimum time required to complete all tasks
def min_completion_time : ℕ := 18

-- A theorem stating that given the times for each task, the minimum completion time is 18 minutes
theorem min_time_to_complete_tasks :
  bed_making_time + teeth_washing_time + water_boiling_time + 
  breakfast_time + dish_washing_time + backpack_organizing_time + milk_making_time - 
  (bed_making_time + teeth_washing_time + backpack_organizing_time + milk_making_time) <=
  min_completion_time := by
  sorry

end NUMINAMATH_GPT_min_time_to_complete_tasks_l2352_235239


namespace NUMINAMATH_GPT_arrange_abc_l2352_235293

noncomputable def a : ℝ := Real.log (4) / Real.log (0.3)
noncomputable def b : ℝ := Real.log (0.2) / Real.log (0.3)
noncomputable def c : ℝ := (1 / Real.exp 1) ^ Real.pi

theorem arrange_abc (a := a) (b := b) (c := c) : b > c ∧ c > a := by
  sorry

end NUMINAMATH_GPT_arrange_abc_l2352_235293


namespace NUMINAMATH_GPT_quadratic_solutions_l2352_235208

theorem quadratic_solutions : ∀ x : ℝ, x^2 - 25 = 0 → (x = 5 ∨ x = -5) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solutions_l2352_235208


namespace NUMINAMATH_GPT_simplify_fraction_l2352_235285

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : ((x^2 - y^2) / (x - y)) = x + y :=
by
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2352_235285


namespace NUMINAMATH_GPT_cos_angle_between_vectors_l2352_235231

theorem cos_angle_between_vectors :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (1, 3)
  let dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
  let magnitude (x : ℝ × ℝ) : ℝ := Real.sqrt (x.1 ^ 2 + x.2 ^ 2)
  let cos_theta := dot_product a b / (magnitude a * magnitude b)
  cos_theta = -Real.sqrt 2 / 10 :=
by
  sorry

end NUMINAMATH_GPT_cos_angle_between_vectors_l2352_235231


namespace NUMINAMATH_GPT_sets_equal_l2352_235232

def M := { u | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l }
def N := { u | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r }

theorem sets_equal : M = N :=
by sorry

end NUMINAMATH_GPT_sets_equal_l2352_235232


namespace NUMINAMATH_GPT_largest_difference_l2352_235274

def A := 3 * 1005^1006
def B := 1005^1006
def C := 1004 * 1005^1005
def D := 3 * 1005^1005
def E := 1005^1005
def F := 1005^1004

theorem largest_difference : 
  A - B > B - C ∧ 
  A - B > C - D ∧ 
  A - B > D - E ∧ 
  A - B > E - F :=
by
  sorry

end NUMINAMATH_GPT_largest_difference_l2352_235274


namespace NUMINAMATH_GPT_fractions_comparison_l2352_235257

theorem fractions_comparison : 
  (99 / 100 < 100 / 101) ∧ (100 / 101 > 199 / 201) ∧ (99 / 100 < 199 / 201) :=
by sorry

end NUMINAMATH_GPT_fractions_comparison_l2352_235257


namespace NUMINAMATH_GPT_range_of_a_l2352_235255

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (3 * a - 2) * x + 6 * a - 1 else a ^ x

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (3 / 8 ≤ a ∧ a < 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2352_235255


namespace NUMINAMATH_GPT_tan_alpha_value_l2352_235249

theorem tan_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α * Real.cos α = 1 / 4) :
  Real.tan α = 2 - Real.sqrt 3 ∨ Real.tan α = 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_alpha_value_l2352_235249


namespace NUMINAMATH_GPT_area_of_circle_below_line_l2352_235270

theorem area_of_circle_below_line (x y : ℝ) :
  (x - 3)^2 + (y - 5)^2 = 9 →
  y ≤ 8 →
  ∃ (A : ℝ), A = 9 * Real.pi :=
sorry

end NUMINAMATH_GPT_area_of_circle_below_line_l2352_235270


namespace NUMINAMATH_GPT_building_height_l2352_235258

theorem building_height (flagpole_height : ℝ) (flagpole_shadow : ℝ) 
  (building_shadow : ℝ) (building_height : ℝ)
  (h_flagpole : flagpole_height = 18)
  (s_flagpole : flagpole_shadow = 45)
  (s_building : building_shadow = 70)
  (ratio_eq : flagpole_height / flagpole_shadow = building_height / building_shadow) :
  building_height = 28 :=
by
  have h_flagpole_shadow := ratio_eq ▸ h_flagpole ▸ s_flagpole ▸ s_building
  sorry

end NUMINAMATH_GPT_building_height_l2352_235258


namespace NUMINAMATH_GPT_problem_statement_l2352_235226

def P := {x : ℤ | ∃ k : ℤ, x = 2 * k - 1}
def Q := {y : ℤ | ∃ n : ℤ, y = 2 * n}

theorem problem_statement (x y : ℤ) (hx : x ∈ P) (hy : y ∈ Q) :
  (x + y ∈ P) ∧ (x * y ∈ Q) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2352_235226


namespace NUMINAMATH_GPT_karen_wins_in_race_l2352_235248

theorem karen_wins_in_race (w : ℝ) (h1 : w / 45 > 1 / 15) 
    (h2 : 60 * (w / 45 - 1 / 15) = w + 4) : 
    w = 8 / 3 := 
sorry

end NUMINAMATH_GPT_karen_wins_in_race_l2352_235248


namespace NUMINAMATH_GPT_brushes_cost_l2352_235271

-- Define the conditions
def canvas_cost (B : ℝ) : ℝ := 3 * B
def paint_cost : ℝ := 5 * 8
def total_material_cost (B : ℝ) : ℝ := B + canvas_cost B + paint_cost
def earning_from_sale : ℝ := 200 - 80

-- State the question as a theorem in Lean
theorem brushes_cost (B : ℝ) (h : total_material_cost B = earning_from_sale) : B = 20 :=
sorry

end NUMINAMATH_GPT_brushes_cost_l2352_235271


namespace NUMINAMATH_GPT_nine_a_minus_six_b_l2352_235265

-- Define the variables and conditions.
variables (a b : ℚ)

-- Assume the given conditions.
def condition1 : Prop := 3 * a + 4 * b = 0
def condition2 : Prop := a = 2 * b - 3

-- Formalize the statement to prove.
theorem nine_a_minus_six_b (h1 : condition1 a b) (h2 : condition2 a b) : 9 * a - 6 * b = -81 / 5 :=
sorry

end NUMINAMATH_GPT_nine_a_minus_six_b_l2352_235265


namespace NUMINAMATH_GPT_polynomial_divisible_exists_l2352_235211

theorem polynomial_divisible_exists (p : Polynomial ℤ) (a : ℕ → ℤ) (k : ℕ) 
  (h_inc : ∀ i j, i < j → a i < a j) (h_nonzero : ∀ i, i < k → p.eval (a i) ≠ 0) :
  ∃ a_0 : ℤ, ∀ i, i < k → p.eval (a i) ∣ p.eval a_0 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_divisible_exists_l2352_235211


namespace NUMINAMATH_GPT_find_side_a_find_area_l2352_235280

-- Definitions from the conditions
variables {A B C : ℝ} 
variables {a b c : ℝ}
variable (angle_B: B = 120 * Real.pi / 180)
variable (side_b: b = Real.sqrt 7)
variable (side_c: c = 1)

-- The first proof problem: Prove that a = 2 given the above conditions
theorem find_side_a (h_angle_B: B = 120 * Real.pi / 180)
  (h_side_b: b = Real.sqrt 7) (h_side_c: c = 1)
  (h_cos_formula: b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) : a = 2 :=
  by
  sorry

-- The second proof problem: Prove that the area is sqrt(3)/2 given the above conditions
theorem find_area (h_angle_B: B = 120 * Real.pi / 180)
  (h_side_b: b = Real.sqrt 7) (h_side_c: c = 1)
  (h_side_a: a = 2) : (1 / 2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
  by
  sorry

end NUMINAMATH_GPT_find_side_a_find_area_l2352_235280


namespace NUMINAMATH_GPT_magnitude_of_difference_between_roots_l2352_235215

variable (α β m : ℝ)

theorem magnitude_of_difference_between_roots
    (hαβ_root : ∀ x, x^2 - 2 * m * x + m^2 - 4 = 0 → (x = α ∨ x = β)) :
    |α - β| = 4 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_difference_between_roots_l2352_235215


namespace NUMINAMATH_GPT_parallel_line_dividing_triangle_l2352_235237

theorem parallel_line_dividing_triangle (base : ℝ) (length_parallel_line : ℝ) 
    (h_base : base = 24) 
    (h_parallel : (length_parallel_line / base)^2 = 1/2) : 
    length_parallel_line = 12 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_parallel_line_dividing_triangle_l2352_235237


namespace NUMINAMATH_GPT_pythagorean_triple_divisibility_l2352_235281

theorem pythagorean_triple_divisibility (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (∃ k₃, k₃ ∣ a ∨ k₃ ∣ b) ∧
  (∃ k₄, k₄ ∣ a ∨ k₄ ∣ b ∧ 2 ∣ k₄) ∧
  (∃ k₅, k₅ ∣ a ∨ k₅ ∣ b ∨ k₅ ∣ c) :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_triple_divisibility_l2352_235281


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_solve_equation_3_l2352_235223

theorem solve_equation_1 (x : ℝ) : (x^2 - 3 * x = 0) ↔ (x = 0 ∨ x = 3) := sorry

theorem solve_equation_2 (x : ℝ) : (4 * x^2 - x - 5 = 0) ↔ (x = 5/4 ∨ x = -1) := sorry

theorem solve_equation_3 (x : ℝ) : (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2/3) := sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_solve_equation_3_l2352_235223


namespace NUMINAMATH_GPT_circle_chord_segments_l2352_235256

theorem circle_chord_segments (r : ℝ) (ch : ℝ) (a : ℝ) :
  (r = 8) ∧ (ch = 12) ∧ (r^2 - a^2 = 36) →
  a = 2 * Real.sqrt 7 → ∃ (ak bk : ℝ), ak = 8 - 2 * Real.sqrt 7 ∧ bk = 8 + 2 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_circle_chord_segments_l2352_235256


namespace NUMINAMATH_GPT_log_product_l2352_235213

open Real

theorem log_product : log 9 / log 2 * (log 5 / log 3) * (log 8 / log (sqrt 5)) = 12 :=
by
  sorry

end NUMINAMATH_GPT_log_product_l2352_235213


namespace NUMINAMATH_GPT_Grant_score_is_100_l2352_235296

/-- Definition of scores --/
def Hunter_score : ℕ := 45

def John_score (H : ℕ) : ℕ := 2 * H

def Grant_score (J : ℕ) : ℕ := J + 10

/-- Theorem to prove Grant's score --/
theorem Grant_score_is_100 : Grant_score (John_score Hunter_score) = 100 := 
  sorry

end NUMINAMATH_GPT_Grant_score_is_100_l2352_235296


namespace NUMINAMATH_GPT_wombat_clawing_l2352_235276

variable (W : ℕ)
variable (R : ℕ := 1)

theorem wombat_clawing :
    (9 * W + 3 * R = 39) → (W = 4) :=
by 
  sorry

end NUMINAMATH_GPT_wombat_clawing_l2352_235276


namespace NUMINAMATH_GPT_students_more_than_pets_l2352_235214

theorem students_more_than_pets
  (students_per_classroom : ℕ)
  (rabbits_per_classroom : ℕ)
  (birds_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (total_students : ℕ)
  (total_rabbits : ℕ)
  (total_birds : ℕ)
  (total_pets : ℕ)
  (difference : ℕ)
  : students_per_classroom = 22 → 
    rabbits_per_classroom = 3 → 
    birds_per_classroom = 2 → 
    number_of_classrooms = 5 → 
    total_students = students_per_classroom * number_of_classrooms → 
    total_rabbits = rabbits_per_classroom * number_of_classrooms → 
    total_birds = birds_per_classroom * number_of_classrooms → 
    total_pets = total_rabbits + total_birds → 
    difference = total_students - total_pets →
    difference = 85 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_students_more_than_pets_l2352_235214


namespace NUMINAMATH_GPT_no_three_segments_form_triangle_l2352_235217

theorem no_three_segments_form_triangle :
  ∃ (a : Fin 10 → ℕ), ∀ {i j k : Fin 10}, i < j → j < k → a i + a j ≤ a k :=
by
  sorry

end NUMINAMATH_GPT_no_three_segments_form_triangle_l2352_235217


namespace NUMINAMATH_GPT_number_of_ways_to_choose_bases_l2352_235205

theorem number_of_ways_to_choose_bases : ∀ (students bases : ℕ), students = 4 → bases = 4 → (bases^students) = 256 :=
by
  intros students bases h_students h_bases
  rw [h_students, h_bases]
  exact pow_succ' 4 3

end NUMINAMATH_GPT_number_of_ways_to_choose_bases_l2352_235205


namespace NUMINAMATH_GPT_intersect_xz_plane_at_point_l2352_235204

-- Define points and vectors in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the points A and B
def A : Point3D := ⟨2, -1, 3⟩
def B : Point3D := ⟨6, 7, -2⟩

-- Define the direction vector as the difference between points A and B
def direction_vector (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

-- Function to parameterize the line given a point and direction vector
def parametric_line (P : Point3D) (v : Point3D) (t : ℝ) : Point3D :=
  ⟨P.x + t * v.x, P.y + t * v.y, P.z + t * v.z⟩

-- Define the xz-plane intersection condition (y coordinate should be 0)
def intersects_xz_plane (P : Point3D) (v : Point3D) (t : ℝ) : Prop :=
  (parametric_line P v t).y = 0

-- Define the intersection point as a Point3D
def intersection_point : Point3D := ⟨2.5, 0, 2.375⟩

-- Statement to prove the intersection
theorem intersect_xz_plane_at_point : 
  ∃ t : ℝ, intersects_xz_plane A (direction_vector A B) t ∧ parametric_line A (direction_vector A B) t = intersection_point :=
by
  sorry

end NUMINAMATH_GPT_intersect_xz_plane_at_point_l2352_235204


namespace NUMINAMATH_GPT_aisha_probability_l2352_235279

noncomputable def prob_one_head (prob_tail : ℝ) (num_coins : ℕ) : ℝ :=
  1 - (prob_tail ^ num_coins)

theorem aisha_probability : 
  prob_one_head (1/2) 4 = 15 / 16 := 
by 
  sorry

end NUMINAMATH_GPT_aisha_probability_l2352_235279


namespace NUMINAMATH_GPT_tangent_slope_is_4_l2352_235206

theorem tangent_slope_is_4 (x y : ℝ) (h_curve : y = x^4) (h_slope : (deriv (fun x => x^4) x) = 4) :
    (x, y) = (1, 1) :=
by
  -- Place proof here
  sorry

end NUMINAMATH_GPT_tangent_slope_is_4_l2352_235206


namespace NUMINAMATH_GPT_expression_simplification_l2352_235207

theorem expression_simplification :
  (2 ^ 2 / 3 + (-(3 ^ 2) + 5) + (-(3) ^ 2) * ((2 / 3) ^ 2)) = 4 / 3 :=
sorry

end NUMINAMATH_GPT_expression_simplification_l2352_235207


namespace NUMINAMATH_GPT_inverse_proposition_l2352_235267

   theorem inverse_proposition (x a b : ℝ) :
     (x ≥ a^2 + b^2 → x ≥ 2 * a * b) →
     (x ≥ 2 * a * b → x ≥ a^2 + b^2) :=
   sorry
   
end NUMINAMATH_GPT_inverse_proposition_l2352_235267


namespace NUMINAMATH_GPT_cube_edge_length_surface_area_equals_volume_l2352_235225

theorem cube_edge_length_surface_area_equals_volume (a : ℝ) (h : 6 * a ^ 2 = a ^ 3) : a = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_cube_edge_length_surface_area_equals_volume_l2352_235225


namespace NUMINAMATH_GPT_tickets_needed_l2352_235292

def tickets_per_roller_coaster : ℕ := 5
def tickets_per_giant_slide : ℕ := 3
def roller_coaster_rides : ℕ := 7
def giant_slide_rides : ℕ := 4

theorem tickets_needed : tickets_per_roller_coaster * roller_coaster_rides + tickets_per_giant_slide * giant_slide_rides = 47 := 
by
  sorry

end NUMINAMATH_GPT_tickets_needed_l2352_235292


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2352_235266

variables {p q : Prop}

theorem sufficient_but_not_necessary :
  (p → q) ∧ (¬q → ¬p) ∧ ¬(q → p) → (¬q → ¬p) ∧ (¬(q → p)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2352_235266


namespace NUMINAMATH_GPT_count_divisible_by_90_four_digit_numbers_l2352_235272

theorem count_divisible_by_90_four_digit_numbers :
  ∃ (n : ℕ), (n = 10) ∧ (∀ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ x % 90 = 0 ∧ x % 100 = 90 → (x = 1890 ∨ x = 2790 ∨ x = 3690 ∨ x = 4590 ∨ x = 5490 ∨ x = 6390 ∨ x = 7290 ∨ x = 8190 ∨ x = 9090 ∨ x = 9990)) :=
by
  sorry

end NUMINAMATH_GPT_count_divisible_by_90_four_digit_numbers_l2352_235272


namespace NUMINAMATH_GPT_principal_amount_l2352_235263

theorem principal_amount (SI : ℝ) (R : ℝ) (T : ℕ) (P : ℝ) :
  SI = 3.45 → R = 0.05 → T = 3 → SI = P * R * T → P = 23 :=
by
  -- The proof steps would go here but are omitted as specified.
  sorry

end NUMINAMATH_GPT_principal_amount_l2352_235263


namespace NUMINAMATH_GPT_num_sheets_in_stack_l2352_235259

-- Definitions coming directly from the conditions
def thickness_ream := 4 -- cm
def num_sheets_ream := 400
def height_stack := 10 -- cm

-- The final proof statement
theorem num_sheets_in_stack : (height_stack / (thickness_ream / num_sheets_ream)) = 1000 :=
by
  sorry

end NUMINAMATH_GPT_num_sheets_in_stack_l2352_235259


namespace NUMINAMATH_GPT_tan_theta_is_sqrt3_div_5_l2352_235283

open Real

theorem tan_theta_is_sqrt3_div_5 (theta : ℝ) (h : 2 * sin (theta + π / 3) = 3 * sin (π / 3 - theta)) :
  tan theta = sqrt 3 / 5 :=
sorry

end NUMINAMATH_GPT_tan_theta_is_sqrt3_div_5_l2352_235283


namespace NUMINAMATH_GPT_ratio_sheep_to_horses_l2352_235212

theorem ratio_sheep_to_horses 
  (horse_food_per_day : ℕ) 
  (total_horse_food : ℕ) 
  (num_sheep : ℕ) 
  (H1 : horse_food_per_day = 230) 
  (H2 : total_horse_food = 12880) 
  (H3 : num_sheep = 48) 
  : (num_sheep : ℚ) / (total_horse_food / horse_food_per_day : ℚ) = 6 / 7
  :=
by
  sorry

end NUMINAMATH_GPT_ratio_sheep_to_horses_l2352_235212


namespace NUMINAMATH_GPT_find_smallest_angle_l2352_235235

open Real

theorem find_smallest_angle :
  ∃ x : ℝ, (x > 0 ∧ sin (4 * x * (π / 180)) * sin (6 * x * (π / 180)) = cos (4 * x * (π / 180)) * cos (6 * x * (π / 180))) ∧ x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_angle_l2352_235235


namespace NUMINAMATH_GPT_melted_ice_cream_depth_l2352_235269

noncomputable def radius_sphere : ℝ := 3
noncomputable def radius_cylinder : ℝ := 10
noncomputable def height_cylinder : ℝ := 36 / 100

theorem melted_ice_cream_depth :
  (4 / 3) * Real.pi * radius_sphere^3 = Real.pi * radius_cylinder^2 * height_cylinder :=
by
  sorry

end NUMINAMATH_GPT_melted_ice_cream_depth_l2352_235269


namespace NUMINAMATH_GPT_total_profit_l2352_235253

theorem total_profit (P Q R : ℝ) (profit : ℝ) 
  (h1 : 4 * P = 6 * Q) 
  (h2 : 6 * Q = 10 * R) 
  (h3 : R = 840 / 6) : 
  profit = 4340 :=
sorry

end NUMINAMATH_GPT_total_profit_l2352_235253


namespace NUMINAMATH_GPT_largest_consecutive_odd_numbers_l2352_235216

theorem largest_consecutive_odd_numbers (x : ℤ)
  (h : (x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) : 
  x + 6 = 27 :=
  sorry

end NUMINAMATH_GPT_largest_consecutive_odd_numbers_l2352_235216


namespace NUMINAMATH_GPT_total_steps_needed_l2352_235278

def cycles_needed (dist : ℕ) : ℕ := dist
def steps_per_cycle : ℕ := 5
def effective_steps_per_pattern : ℕ := 1

theorem total_steps_needed (dist : ℕ) (h : dist = 66) : 
  steps_per_cycle * cycles_needed dist = 330 :=
by 
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_total_steps_needed_l2352_235278


namespace NUMINAMATH_GPT_karl_sticker_count_l2352_235260

theorem karl_sticker_count : 
  ∀ (K R B : ℕ), 
    (R = K + 20) → 
    (B = R - 10) → 
    (K + R + B = 105) → 
    K = 25 := 
by
  intros K R B hR hB hSum
  sorry

end NUMINAMATH_GPT_karl_sticker_count_l2352_235260


namespace NUMINAMATH_GPT_fourth_machine_works_for_12_hours_daily_l2352_235250

noncomputable def hours_fourth_machine_works (m1_hours m1_production_rate: ℕ) (m2_hours m2_production_rate: ℕ) (price_per_kg: ℕ) (total_earning: ℕ) :=
  let m1_total_production := m1_hours * m1_production_rate
  let m1_total_output := 3 * m1_total_production
  let m1_revenue := m1_total_output * price_per_kg
  let remaining_revenue := total_earning - m1_revenue
  let m2_total_production := remaining_revenue / price_per_kg
  m2_total_production / m2_production_rate

theorem fourth_machine_works_for_12_hours_daily : hours_fourth_machine_works 23 2 (sorry) (sorry) 50 8100 = 12 := by
  sorry

end NUMINAMATH_GPT_fourth_machine_works_for_12_hours_daily_l2352_235250


namespace NUMINAMATH_GPT_product_of_numerator_and_denominator_l2352_235233

-- Defining the repeating decimal as a fraction in lowest terms
def repeating_decimal_as_fraction_in_lowest_terms : ℚ :=
  1 / 37

-- Theorem to prove the product of the numerator and the denominator
theorem product_of_numerator_and_denominator :
  (repeating_decimal_as_fraction_in_lowest_terms.num.natAbs *
   repeating_decimal_as_fraction_in_lowest_terms.den) = 37 :=
by
  -- declaration of the needed fact and its direct consequence
  sorry

end NUMINAMATH_GPT_product_of_numerator_and_denominator_l2352_235233


namespace NUMINAMATH_GPT_width_of_deck_l2352_235241

noncomputable def length : ℝ := 30
noncomputable def cost_per_sqft_construction : ℝ := 3
noncomputable def cost_per_sqft_sealant : ℝ := 1
noncomputable def total_cost : ℝ := 4800
noncomputable def total_cost_per_sqft : ℝ := cost_per_sqft_construction + cost_per_sqft_sealant

theorem width_of_deck (w : ℝ) 
  (h1 : length * w * total_cost_per_sqft = total_cost) : 
  w = 40 := 
sorry

end NUMINAMATH_GPT_width_of_deck_l2352_235241


namespace NUMINAMATH_GPT_total_tiles_number_l2352_235268

-- Define the conditions based on the problem statement
def square_floor_tiles (s : ℕ) : ℕ := s * s

def black_tiles_count (s : ℕ) : ℕ := 3 * s - 3

-- The main theorem statement: given the number of black tiles as 201,
-- prove that the total number of tiles is 4624
theorem total_tiles_number (s : ℕ) (h₁ : black_tiles_count s = 201) : 
  square_floor_tiles s = 4624 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_total_tiles_number_l2352_235268


namespace NUMINAMATH_GPT_math_problem_l2352_235242

theorem math_problem : (-1: ℝ)^2 + (1/3: ℝ)^0 = 2 := by
  sorry

end NUMINAMATH_GPT_math_problem_l2352_235242


namespace NUMINAMATH_GPT_power_addition_l2352_235298

theorem power_addition {a m n : ℝ} (h1 : a^m = 2) (h2 : a^n = 8) : a^(m + n) = 16 :=
sorry

end NUMINAMATH_GPT_power_addition_l2352_235298


namespace NUMINAMATH_GPT_repeating_decimals_sum_l2352_235286

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end NUMINAMATH_GPT_repeating_decimals_sum_l2352_235286


namespace NUMINAMATH_GPT_nuts_consumed_range_l2352_235220

def diet_day_nuts : Nat := 1
def normal_day_nuts : Nat := diet_day_nuts + 2

def total_nuts_consumed (start_with_diet_day : Bool) : Nat :=
  if start_with_diet_day then
    (10 * diet_day_nuts) + (9 * normal_day_nuts)
  else
    (10 * normal_day_nuts) + (9 * diet_day_nuts)

def min_nuts_consumed : Nat :=
  Nat.min (total_nuts_consumed true) (total_nuts_consumed false)

def max_nuts_consumed : Nat :=
  Nat.max (total_nuts_consumed true) (total_nuts_consumed false)

theorem nuts_consumed_range :
  min_nuts_consumed = 37 ∧ max_nuts_consumed = 39 := by
  sorry

end NUMINAMATH_GPT_nuts_consumed_range_l2352_235220


namespace NUMINAMATH_GPT_number_of_real_roots_eq_3_eq_m_l2352_235218

theorem number_of_real_roots_eq_3_eq_m {x m : ℝ} (h : ∀ x, x^2 - 2 * |x| + 2 = m) : m = 2 :=
sorry

end NUMINAMATH_GPT_number_of_real_roots_eq_3_eq_m_l2352_235218


namespace NUMINAMATH_GPT_gcd_of_repeated_six_digit_integers_l2352_235234

-- Given condition
def is_repeated_six_digit_integer (n : ℕ) : Prop :=
  100 ≤ n / 1000 ∧ n / 1000 < 1000 ∧ n = 1001 * (n / 1000)

-- Theorem to prove
theorem gcd_of_repeated_six_digit_integers :
  ∀ n : ℕ, is_repeated_six_digit_integer n → gcd n 1001 = 1001 :=
by sorry

end NUMINAMATH_GPT_gcd_of_repeated_six_digit_integers_l2352_235234


namespace NUMINAMATH_GPT_dave_spent_on_books_l2352_235221

-- Define the cost of books in each category without any discounts or taxes
def cost_animal_books : ℝ := 8 * 10
def cost_outer_space_books : ℝ := 6 * 12
def cost_train_books : ℝ := 9 * 8
def cost_history_books : ℝ := 4 * 15
def cost_science_books : ℝ := 5 * 18

-- Define the discount and tax rates
def discount_animal_books : ℝ := 0.10
def tax_science_books : ℝ := 0.15

-- Apply the discount to animal books
def discounted_cost_animal_books : ℝ := cost_animal_books * (1 - discount_animal_books)

-- Apply the tax to science books
def final_cost_science_books : ℝ := cost_science_books * (1 + tax_science_books)

-- Calculate the total cost of all books after discounts and taxes
def total_cost : ℝ := discounted_cost_animal_books 
                  + cost_outer_space_books
                  + cost_train_books
                  + cost_history_books
                  + final_cost_science_books

theorem dave_spent_on_books : total_cost = 379.5 := by
  sorry

end NUMINAMATH_GPT_dave_spent_on_books_l2352_235221


namespace NUMINAMATH_GPT_sqrt_x_div_sqrt_y_l2352_235203

theorem sqrt_x_div_sqrt_y (x y : ℝ)
  (h : ( ( (2/3)^2 + (1/6)^2 ) / ( (1/2)^2 + (1/7)^2 ) ) = 28 * x / (25 * y)) :
  (Real.sqrt x) / (Real.sqrt y) = 5 / 2 :=
sorry

end NUMINAMATH_GPT_sqrt_x_div_sqrt_y_l2352_235203


namespace NUMINAMATH_GPT_largest_common_number_in_sequences_from_1_to_200_l2352_235251

theorem largest_common_number_in_sequences_from_1_to_200 :
  ∃ a, a ≤ 200 ∧ a % 8 = 3 ∧ a % 9 = 5 ∧ ∀ b, (b ≤ 200 ∧ b % 8 = 3 ∧ b % 9 = 5) → b ≤ a :=
sorry

end NUMINAMATH_GPT_largest_common_number_in_sequences_from_1_to_200_l2352_235251


namespace NUMINAMATH_GPT_point_in_or_on_circle_l2352_235284

theorem point_in_or_on_circle (θ : Real) :
  let P := (5 * Real.cos θ, 4 * Real.sin θ)
  let C_eq := ∀ (x y : Real), x^2 + y^2 = 25
  25 * Real.cos θ ^ 2 + 16 * Real.sin θ ^ 2 ≤ 25 := 
by 
  sorry

end NUMINAMATH_GPT_point_in_or_on_circle_l2352_235284


namespace NUMINAMATH_GPT_power_function_through_point_l2352_235290

-- Define the condition that the power function passes through the point (2, 8)
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) (h : ∀ x, f x = x^α) (h₂ : f 2 = 8) :
  α = 3 ∧ ∀ x, f x = x^3 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_power_function_through_point_l2352_235290


namespace NUMINAMATH_GPT_internal_angles_and_area_of_grey_triangle_l2352_235238

/-- Given three identical grey triangles, 
    three identical squares, and an equilateral 
    center triangle with area 2 cm^2,
    the internal angles of the grey triangles 
    are 120 degrees and 30 degrees, and the 
    total grey area is 6 cm^2. -/
theorem internal_angles_and_area_of_grey_triangle 
  (triangle_area : ℝ)
  (α β : ℝ)
  (grey_area : ℝ) :
  triangle_area = 2 →  
  α = 120 ∧ β = 30 ∧ grey_area = 6 :=
by
  sorry

end NUMINAMATH_GPT_internal_angles_and_area_of_grey_triangle_l2352_235238


namespace NUMINAMATH_GPT_distance_from_P_to_x_axis_l2352_235245

-- Define the point P with coordinates (4, -3)
def P : ℝ × ℝ := (4, -3)

-- Define the distance from a point to the x-axis as the absolute value of the y-coordinate
def distance_to_x_axis (point : ℝ × ℝ) : ℝ :=
  abs point.snd

-- State the theorem to be proved
theorem distance_from_P_to_x_axis : distance_to_x_axis P = 3 :=
by
  -- The proof is not required; we can use sorry to skip it
  sorry

end NUMINAMATH_GPT_distance_from_P_to_x_axis_l2352_235245
