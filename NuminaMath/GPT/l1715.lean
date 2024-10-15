import Mathlib

namespace NUMINAMATH_GPT_slower_pipe_filling_time_l1715_171548

theorem slower_pipe_filling_time
  (t : ℝ)
  (H1 : ∀ (time_slow : ℝ), time_slow = t)
  (H2 : ∀ (time_fast : ℝ), time_fast = t / 3)
  (H3 : 1 / t + 1 / (t / 3) = 1 / 40) :
  t = 160 :=
sorry

end NUMINAMATH_GPT_slower_pipe_filling_time_l1715_171548


namespace NUMINAMATH_GPT_pizza_slice_volume_l1715_171515

-- Define the parameters given in the conditions
def pizza_thickness : ℝ := 0.5
def pizza_diameter : ℝ := 16.0
def num_slices : ℝ := 16.0

-- Define the volume of one slice
theorem pizza_slice_volume : (π * (pizza_diameter / 2) ^ 2 * pizza_thickness / num_slices) = 2 * π := by
  sorry

end NUMINAMATH_GPT_pizza_slice_volume_l1715_171515


namespace NUMINAMATH_GPT_jane_total_drying_time_l1715_171541

theorem jane_total_drying_time :
  let base_coat := 4
  let color_coat_1 := 5
  let color_coat_2 := 6
  let color_coat_3 := 7
  let nail_art_1 := 8
  let nail_art_2 := 10
  let top_coat := 9
  base_coat + color_coat_1 + color_coat_2 + color_coat_3 + nail_art_1 + nail_art_2 + top_coat = 49 :=
by 
  sorry

end NUMINAMATH_GPT_jane_total_drying_time_l1715_171541


namespace NUMINAMATH_GPT_TV_cost_is_1700_l1715_171540

def hourlyRate : ℝ := 10
def workHoursPerWeek : ℝ := 30
def weeksPerMonth : ℝ := 4
def additionalHours : ℝ := 50

def weeklyEarnings : ℝ := hourlyRate * workHoursPerWeek
def monthlyEarnings : ℝ := weeklyEarnings * weeksPerMonth
def additionalEarnings : ℝ := hourlyRate * additionalHours

def TVCost : ℝ := monthlyEarnings + additionalEarnings

theorem TV_cost_is_1700 : TVCost = 1700 := sorry

end NUMINAMATH_GPT_TV_cost_is_1700_l1715_171540


namespace NUMINAMATH_GPT_mary_total_nickels_l1715_171531

def mary_initial_nickels : ℕ := 7
def mary_dad_nickels : ℕ := 5

theorem mary_total_nickels : mary_initial_nickels + mary_dad_nickels = 12 := by
  sorry

end NUMINAMATH_GPT_mary_total_nickels_l1715_171531


namespace NUMINAMATH_GPT_total_pages_correct_l1715_171589

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def sum_history_geography_pages : ℕ := history_pages + geography_pages
def math_pages : ℕ := sum_history_geography_pages / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_correct : total_pages = 905 := by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_total_pages_correct_l1715_171589


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_18_with_digits_9_or_0_l1715_171577

noncomputable def m : ℕ := 90
theorem smallest_positive_multiple_of_18_with_digits_9_or_0 : m = 90 ∧ (∀ d ∈ m.digits 10, d = 0 ∨ d = 9) ∧ m % 18 = 0 → m / 18 = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_18_with_digits_9_or_0_l1715_171577


namespace NUMINAMATH_GPT_prove_intersection_area_is_correct_l1715_171578

noncomputable def octahedron_intersection_area 
  (side_length : ℝ) (cut_height_factor : ℝ) : ℝ :=
  have height_triangular_face := Real.sqrt (side_length^2 - (side_length / 2)^2)
  have plane_height := cut_height_factor * height_triangular_face
  have proportional_height := plane_height / height_triangular_face
  let new_side_length := proportional_height * side_length
  have hexagon_area := (3 * Real.sqrt 3 / 2) * (new_side_length^2) / 2 
  (3 * Real.sqrt 3 / 2) * (new_side_length^2)

theorem prove_intersection_area_is_correct 
  : 
  octahedron_intersection_area 2 (3 / 4) = 9 * Real.sqrt 3 / 8 :=
  sorry 

example : 9 + 3 + 8 = 20 := 
  by rfl

end NUMINAMATH_GPT_prove_intersection_area_is_correct_l1715_171578


namespace NUMINAMATH_GPT_average_percentage_decrease_l1715_171508

theorem average_percentage_decrease :
  ∃ (x : ℝ), (5000 * (1 - x / 100)^3 = 2560) ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_average_percentage_decrease_l1715_171508


namespace NUMINAMATH_GPT_find_length_BF_l1715_171501

-- Define the conditions
structure Rectangle :=
  (short_side : ℝ)
  (long_side : ℝ)

def folded_paper (rect : Rectangle) : Prop :=
  rect.short_side = 12

def congruent_triangles (rect : Rectangle) : Prop :=
  rect.short_side = 12

-- Define the length of BF to prove
def length_BF (rect : Rectangle) : ℝ := 10

-- The theorem statement
theorem find_length_BF (rect : Rectangle) (h1 : folded_paper rect) (h2 : congruent_triangles rect) :
  length_BF rect = 10 := 
  sorry

end NUMINAMATH_GPT_find_length_BF_l1715_171501


namespace NUMINAMATH_GPT_part1_part2_l1715_171500

def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (x - 3)
noncomputable def M := 3 / 2

theorem part1 (x : ℝ) (m : ℝ) : (∀ x, f x ≥ abs (m + 1)) → m ≤ M := sorry

theorem part2 (a b c : ℝ) : a > 0 → b > 0 → c > 0 → a + b + c = M →  (b^2 / a + c^2 / b + a^2 / c) ≥ M := sorry

end NUMINAMATH_GPT_part1_part2_l1715_171500


namespace NUMINAMATH_GPT_ones_digit_of_p_l1715_171559

theorem ones_digit_of_p (p q r s : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (hseq : q = p + 4 ∧ r = p + 8 ∧ s = p + 12) (hpg : p > 5) : (p % 10) = 9 :=
by
  sorry

end NUMINAMATH_GPT_ones_digit_of_p_l1715_171559


namespace NUMINAMATH_GPT_valid_sequences_count_l1715_171537

noncomputable def number_of_valid_sequences
(strings : List (List Nat))
(ball_A_shot : Nat)
(ball_B_shot : Nat) : Nat := 144

theorem valid_sequences_count :
  let strings := [[1, 2], [3, 4, 5], [6, 7, 8, 9]];
  let ball_A := 1;  -- Assuming A is the first ball in the first string
  let ball_B := 3;  -- Assuming B is the first ball in the second string
  ball_A = 1 →
  ball_B = 3 →
  ball_A_shot = 5 →
  ball_B_shot = 6 →
  number_of_valid_sequences strings ball_A_shot ball_B_shot = 144 :=
by
  intros strings ball_A ball_B hA hB hAShot hBShot
  sorry

end NUMINAMATH_GPT_valid_sequences_count_l1715_171537


namespace NUMINAMATH_GPT_square_of_999_l1715_171562

theorem square_of_999 : 999 * 999 = 998001 := by
  sorry

end NUMINAMATH_GPT_square_of_999_l1715_171562


namespace NUMINAMATH_GPT_water_dispenser_capacity_l1715_171523

theorem water_dispenser_capacity :
  ∀ (x : ℝ), (0.25 * x = 60) → x = 240 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_water_dispenser_capacity_l1715_171523


namespace NUMINAMATH_GPT_radius_of_circle_l1715_171575

theorem radius_of_circle : 
  ∀ (r : ℝ), 3 * (2 * Real.pi * r) = 2 * Real.pi * r ^ 2 → r = 3 :=
by
  intro r
  intro h
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1715_171575


namespace NUMINAMATH_GPT_c_work_rate_l1715_171570

theorem c_work_rate {A B C : ℚ} (h1 : A + B = 1/6) (h2 : B + C = 1/8) (h3 : C + A = 1/12) : C = 1/48 :=
by
  sorry

end NUMINAMATH_GPT_c_work_rate_l1715_171570


namespace NUMINAMATH_GPT_smallest_four_digit_2_mod_11_l1715_171582

theorem smallest_four_digit_2_mod_11 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 11 = 2 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 11 = 2 → n ≤ m) := 
by 
  use 1003
  sorry

end NUMINAMATH_GPT_smallest_four_digit_2_mod_11_l1715_171582


namespace NUMINAMATH_GPT_cubic_equation_roots_l1715_171550

theorem cubic_equation_roots (a b c d : ℝ) (h_a : a ≠ 0) 
(h_root1 : a * 4^3 + b * 4^2 + c * 4 + d = 0)
(h_root2 : a * (-3)^3 + b * (-3)^2 - 3 * c + d = 0) :
 (b + c) / a = -13 :=
by sorry

end NUMINAMATH_GPT_cubic_equation_roots_l1715_171550


namespace NUMINAMATH_GPT_probability_blue_is_approx_50_42_l1715_171597

noncomputable def probability_blue_second_pick : ℚ :=
  let yellow := 30
  let green := yellow / 3
  let red := 2 * green
  let total_marbles := 120
  let blue := total_marbles - (yellow + green + red)
  let total_after_first_pick := total_marbles - 1
  let blue_probability := (blue : ℚ) / total_after_first_pick
  blue_probability * 100

theorem probability_blue_is_approx_50_42 :
  abs (probability_blue_second_pick - 50.42) < 0.005 := -- Approximately checking for equality due to possible floating-point precision issues
sorry

end NUMINAMATH_GPT_probability_blue_is_approx_50_42_l1715_171597


namespace NUMINAMATH_GPT_quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l1715_171556

-- Part 1: Expression of the quadratic function
theorem quadratic_function_expression (a : ℝ) (h : a = 0) : 
  ∀ x, (x^2 + (a-2)*x + 3) = x^2 - 2*x + 3 :=
by sorry

-- Part 2: Range of y for 0 < x < 3
theorem quadratic_function_range (x y : ℝ) (h : ∀ x, y = x^2 - 2*x + 3) (hx : 0 < x ∧ x < 3) :
  2 ≤ y ∧ y < 6 :=
by sorry

-- Part 3: Range of m for y1 > y2
theorem quadratic_function_m_range (m y1 y2 : ℝ) (P Q : ℝ × ℝ)
  (h1 : P = (m - 1, y1)) (h2 : Q = (m, y2)) (h3 : y1 > y2) :
  m < 3 / 2 :=
by sorry

end NUMINAMATH_GPT_quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l1715_171556


namespace NUMINAMATH_GPT_inequality_proof_l1715_171551

theorem inequality_proof
  (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + d^2) * (d^2 + a^2) ≥ 
  64 * a * b * c * d * abs ((a - b) * (b - c) * (c - d) * (d - a)) := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1715_171551


namespace NUMINAMATH_GPT_inequality_1_system_of_inequalities_l1715_171594

-- Statement for inequality (1)
theorem inequality_1 (x : ℝ) : 2 - x ≥ (x - 1) / 3 - 1 → x ≤ 2.5 := 
sorry

-- Statement for system of inequalities (2)
theorem system_of_inequalities (x : ℝ) : 
  (5 * x + 1 < 3 * (x - 1)) ∧ ((x + 8) / 5 < (2 * x - 5) / 3 - 1) → false := 
sorry

end NUMINAMATH_GPT_inequality_1_system_of_inequalities_l1715_171594


namespace NUMINAMATH_GPT_coin_stack_count_l1715_171513

theorem coin_stack_count
  (TN : ℝ := 1.95)
  (TQ : ℝ := 1.75)
  (SH : ℝ := 20)
  (n q : ℕ) :
  (n*Tℕ + q*TQ = SH) → (n + q = 10) :=
sorry

end NUMINAMATH_GPT_coin_stack_count_l1715_171513


namespace NUMINAMATH_GPT_andrew_spent_total_amount_l1715_171553

/-- Conditions:
1. Andrew played a total of 7 games.
2. Cost distribution for games:
   - 3 games cost $9.00 each
   - 2 games cost $12.50 each
   - 2 games cost $15.00 each
3. Additional expenses:
   - $25.00 on snacks
   - $20.00 on drinks
-/
def total_cost_games : ℝ :=
  (3 * 9) + (2 * 12.5) + (2 * 15)

def cost_snacks : ℝ := 25
def cost_drinks : ℝ := 20

def total_spent (cost_games cost_snacks cost_drinks : ℝ) : ℝ :=
  cost_games + cost_snacks + cost_drinks

theorem andrew_spent_total_amount :
  total_spent total_cost_games 25 20 = 127 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_andrew_spent_total_amount_l1715_171553


namespace NUMINAMATH_GPT_village_food_sales_l1715_171536

theorem village_food_sales :
  ∀ (customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
      price_per_head_of_lettuce price_per_tomato : ℕ) 
    (H1 : customers_per_month = 500)
    (H2 : heads_of_lettuce_per_person = 2)
    (H3 : tomatoes_per_person = 4)
    (H4 : price_per_head_of_lettuce = 1)
    (H5 : price_per_tomato = 1 / 2), 
  customers_per_month * ((heads_of_lettuce_per_person * price_per_head_of_lettuce) 
    + (tomatoes_per_person * (price_per_tomato : ℝ))) = 2000 := 
by 
  intros customers_per_month heads_of_lettuce_per_person tomatoes_per_person 
         price_per_head_of_lettuce price_per_tomato 
         H1 H2 H3 H4 H5
  sorry

end NUMINAMATH_GPT_village_food_sales_l1715_171536


namespace NUMINAMATH_GPT_f_4_1981_eq_l1715_171596

noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| (x+1), 0 => f x 1
| (x+1), (y+1) => f x (f (x+1) y)

theorem f_4_1981_eq : f 4 1981 = 2^1984 - 3 := 
by
  sorry

end NUMINAMATH_GPT_f_4_1981_eq_l1715_171596


namespace NUMINAMATH_GPT_line_plane_intersection_l1715_171520

theorem line_plane_intersection :
  ∃ (x y z : ℝ), (∃ t : ℝ, x = -3 + 2 * t ∧ y = 1 + 3 * t ∧ z = 1 + 5 * t) ∧ (2 * x + 3 * y + 7 * z - 52 = 0) ∧ (x = -1) ∧ (y = 4) ∧ (z = 6) :=
sorry

end NUMINAMATH_GPT_line_plane_intersection_l1715_171520


namespace NUMINAMATH_GPT_lindsey_saved_in_november_l1715_171512

def savings_sept : ℕ := 50
def savings_oct : ℕ := 37
def additional_money : ℕ := 25
def spent_on_video_game : ℕ := 87
def money_left : ℕ := 36

def total_savings_before_november := savings_sept + savings_oct
def total_savings_after_november (N : ℕ) := total_savings_before_november + N + additional_money

theorem lindsey_saved_in_november : ∃ N : ℕ, total_savings_after_november N - spent_on_video_game = money_left ∧ N = 11 :=
by
  sorry

end NUMINAMATH_GPT_lindsey_saved_in_november_l1715_171512


namespace NUMINAMATH_GPT_fraction_problem_l1715_171524

-- Definitions of x and y based on the given conditions
def x : ℚ := 3 / 5
def y : ℚ := 7 / 9

-- The theorem stating the mathematical equivalence to be proven
theorem fraction_problem : (5 * x + 9 * y) / (45 * x * y) = 10 / 21 :=
by
  sorry

end NUMINAMATH_GPT_fraction_problem_l1715_171524


namespace NUMINAMATH_GPT_area_of_square_with_perimeter_l1715_171580

def perimeter_of_square (s : ℝ) : ℝ := 4 * s

def area_of_square (s : ℝ) : ℝ := s * s

theorem area_of_square_with_perimeter (p : ℝ) (h : perimeter_of_square (3 * p) = 12 * p) : area_of_square (3 * p) = 9 * p^2 := by
  sorry

end NUMINAMATH_GPT_area_of_square_with_perimeter_l1715_171580


namespace NUMINAMATH_GPT_find_slope_angle_l1715_171591

theorem find_slope_angle (α : ℝ) :
    (∃ x y : ℝ, x * Real.sin (2 * Real.pi / 5) + y * Real.cos (2 * Real.pi / 5) = 0) →
    α = 3 * Real.pi / 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_slope_angle_l1715_171591


namespace NUMINAMATH_GPT_cost_per_book_l1715_171552

theorem cost_per_book (initial_amount : ℤ) (remaining_amount : ℤ) (num_books : ℤ) (cost_per_book : ℤ) :
  initial_amount = 79 →
  remaining_amount = 16 →
  num_books = 9 →
  cost_per_book = (initial_amount - remaining_amount) / num_books →
  cost_per_book = 7 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_book_l1715_171552


namespace NUMINAMATH_GPT_machines_work_together_time_l1715_171583

theorem machines_work_together_time (rate1 rate2 : ℚ) (h1 : rate1 = 1 / 20) (h2 : rate2 = 1 / 30) :
  (1 / (rate1 + rate2)) = 12 :=
by
  sorry

end NUMINAMATH_GPT_machines_work_together_time_l1715_171583


namespace NUMINAMATH_GPT_stones_max_value_50_l1715_171521

-- Define the problem conditions in Lean
def value_of_stones (x y z : ℕ) : ℕ := 14 * x + 11 * y + 2 * z

def weight_of_stones (x y z : ℕ) : ℕ := 5 * x + 4 * y + z

def max_value_stones {x y z : ℕ} (h_w : weight_of_stones x y z ≤ 18) (h_x : x ≥ 0) (h_y : y ≥ 0) (h_z : z ≥ 0) : Prop :=
  value_of_stones x y z ≤ 50

theorem stones_max_value_50 : ∃ (x y z : ℕ), weight_of_stones x y z ≤ 18 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ value_of_stones x y z = 50 :=
by
  sorry

end NUMINAMATH_GPT_stones_max_value_50_l1715_171521


namespace NUMINAMATH_GPT_green_pen_count_l1715_171572

theorem green_pen_count 
  (blue_pens green_pens : ℕ)
  (h_ratio : blue_pens = 5 * green_pens / 3)
  (h_blue_pens : blue_pens = 20)
  : green_pens = 12 :=
by
  sorry

end NUMINAMATH_GPT_green_pen_count_l1715_171572


namespace NUMINAMATH_GPT_find_n_l1715_171503

theorem find_n (n : ℕ) (hn : n * n! - n! = 5040 - n!) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1715_171503


namespace NUMINAMATH_GPT_opposite_of_neg_3_l1715_171574

theorem opposite_of_neg_3 : (-(-3) = 3) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_3_l1715_171574


namespace NUMINAMATH_GPT_evaluate_g_at_4_l1715_171588

def g (x : ℕ) : ℕ := 5 * x - 2

theorem evaluate_g_at_4 : g 4 = 18 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_4_l1715_171588


namespace NUMINAMATH_GPT_evaluate_operation_l1715_171595

def operation (x : ℝ) : ℝ := 9 - x

theorem evaluate_operation : operation (operation 15) = 15 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_evaluate_operation_l1715_171595


namespace NUMINAMATH_GPT_termite_ridden_fraction_l1715_171505

theorem termite_ridden_fraction:
  ∀ T: ℝ, (3 / 4) * T = 1 / 4 → T = 1 / 3 :=
by
  intro T
  intro h
  sorry

end NUMINAMATH_GPT_termite_ridden_fraction_l1715_171505


namespace NUMINAMATH_GPT_gcd_of_powers_of_two_minus_one_l1715_171547

theorem gcd_of_powers_of_two_minus_one : 
  gcd (2^1015 - 1) (2^1020 - 1) = 1 :=
sorry

end NUMINAMATH_GPT_gcd_of_powers_of_two_minus_one_l1715_171547


namespace NUMINAMATH_GPT_tan_neg_3900_eq_sqrt3_l1715_171571

theorem tan_neg_3900_eq_sqrt3 : Real.tan (-3900 * Real.pi / 180) = Real.sqrt 3 := by
  -- Definitions of trigonometric values at 60 degrees
  have h_cos : Real.cos (60 * Real.pi / 180) = 1 / 2 := sorry
  have h_sin : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Using periodicity of the tangent function
  sorry

end NUMINAMATH_GPT_tan_neg_3900_eq_sqrt3_l1715_171571


namespace NUMINAMATH_GPT_fraction_exponentiation_example_l1715_171522

theorem fraction_exponentiation_example :
  (5/3)^4 = 625/81 :=
by
  sorry

end NUMINAMATH_GPT_fraction_exponentiation_example_l1715_171522


namespace NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l1715_171593

noncomputable
def convert_to_cartesian (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  let ρ1 := 10
  let θ1 := Real.pi / 4
  let φ1 := Real.pi / 6
  let ρ2 := 15
  let θ2 := 5 * Real.pi / 4
  let φ2 := Real.pi / 3
  convert_to_cartesian ρ1 θ1 φ1 = (5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 5 * Real.sqrt 3)
  ∧ convert_to_cartesian ρ2 θ2 φ2 = (-15 * Real.sqrt 6 / 4, -15 * Real.sqrt 6 / 4, 7.5) := 
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l1715_171593


namespace NUMINAMATH_GPT_parabola_directrix_p_l1715_171519

/-- Given a parabola with equation y^2 = 2px and directrix x = -2, prove that p = 4 -/
theorem parabola_directrix_p (p : ℝ) :
  (∀ y x : ℝ, y^2 = 2 * p * x) ∧ (∀ x : ℝ, x = -2 → True) → p = 4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_p_l1715_171519


namespace NUMINAMATH_GPT_lea_notebooks_count_l1715_171538

theorem lea_notebooks_count
  (cost_book : ℕ)
  (cost_binder : ℕ)
  (num_binders : ℕ)
  (cost_notebook : ℕ)
  (total_cost : ℕ)
  (h_book : cost_book = 16)
  (h_binder : cost_binder = 2)
  (h_num_binders : num_binders = 3)
  (h_notebook : cost_notebook = 1)
  (h_total : total_cost = 28) :
  ∃ num_notebooks : ℕ, num_notebooks = 6 ∧
    total_cost = cost_book + num_binders * cost_binder + num_notebooks * cost_notebook := 
by
  sorry

end NUMINAMATH_GPT_lea_notebooks_count_l1715_171538


namespace NUMINAMATH_GPT_additional_donation_l1715_171558

theorem additional_donation
  (t : ℕ) (c d₁ d₂ T a : ℝ)
  (h1 : t = 25)
  (h2 : c = 2.00)
  (h3 : d₁ = 15.00) 
  (h4 : d₂ = 15.00)
  (h5 : T = 100.00)
  (h6 : t * c + d₁ + d₂ + a = T) :
  a = 20.00 :=
by
  sorry

end NUMINAMATH_GPT_additional_donation_l1715_171558


namespace NUMINAMATH_GPT_circle_center_l1715_171565

theorem circle_center (x y : ℝ) : 
    (∃ x y : ℝ, x^2 - 8*x + y^2 - 4*y = 16) → (x, y) = (4, 2) := by
  sorry

end NUMINAMATH_GPT_circle_center_l1715_171565


namespace NUMINAMATH_GPT_y_coordinates_difference_l1715_171560

theorem y_coordinates_difference {m n k : ℤ}
  (h1 : m = 2 * n + 5)
  (h2 : m + 4 = 2 * (n + k) + 5) :
  k = 2 :=
by
  sorry

end NUMINAMATH_GPT_y_coordinates_difference_l1715_171560


namespace NUMINAMATH_GPT_quadratic_real_roots_range_l1715_171509

theorem quadratic_real_roots_range (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 2) :=
by
-- Proof outline:
-- Case 1: when a = 1, the equation simplifies to -2x + 1 = 0, which has a real solution x = 1/2.
-- Case 2: when a ≠ 1, the quadratic equation has real roots if the discriminant 8 - 4a ≥ 0, i.e., 2 ≥ a.
sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_l1715_171509


namespace NUMINAMATH_GPT_urea_formation_l1715_171545

theorem urea_formation
  (CO2 NH3 Urea : ℕ) 
  (h_CO2 : CO2 = 1)
  (h_NH3 : NH3 = 2) :
  Urea = 1 := by
  sorry

end NUMINAMATH_GPT_urea_formation_l1715_171545


namespace NUMINAMATH_GPT_coleFenceCostCorrect_l1715_171511

noncomputable def coleFenceCost : ℕ := 455

def woodenFenceCost : ℕ := 15 * 6
def woodenFenceNeighborContribution : ℕ := woodenFenceCost / 3
def coleWoodenFenceCost : ℕ := woodenFenceCost - woodenFenceNeighborContribution

def metalFenceCost : ℕ := 15 * 8
def coleMetalFenceCost : ℕ := metalFenceCost

def hedgeCost : ℕ := 30 * 10
def hedgeNeighborContribution : ℕ := hedgeCost / 2
def coleHedgeCost : ℕ := hedgeCost - hedgeNeighborContribution

def installationFee : ℕ := 75
def soilPreparationFee : ℕ := 50

def totalCost : ℕ := coleWoodenFenceCost + coleMetalFenceCost + coleHedgeCost + installationFee + soilPreparationFee

theorem coleFenceCostCorrect : totalCost = coleFenceCost := by
  -- Skipping the proof steps with sorry
  sorry

end NUMINAMATH_GPT_coleFenceCostCorrect_l1715_171511


namespace NUMINAMATH_GPT_single_solution_inequality_l1715_171599

theorem single_solution_inequality (a : ℝ) :
  (∃! (x : ℝ), abs (x^2 + 2 * a * x + 3 * a) ≤ 2) ↔ a = 1 ∨ a = 2 := 
sorry

end NUMINAMATH_GPT_single_solution_inequality_l1715_171599


namespace NUMINAMATH_GPT_cindy_hit_section_8_l1715_171542

inductive Player : Type
| Alice | Ben | Cindy | Dave | Ellen
deriving DecidableEq

structure DartContest :=
(player : Player)
(score : ℕ)

def ContestConditions (dc : DartContest) : Prop :=
  match dc with
  | ⟨Player.Alice, 10⟩ => True
  | ⟨Player.Ben, 6⟩ => True
  | ⟨Player.Cindy, 9⟩ => True
  | ⟨Player.Dave, 15⟩ => True
  | ⟨Player.Ellen, 19⟩ => True
  | _ => False

def isScoreSection8 (dc : DartContest) : Prop :=
  dc.player = Player.Cindy ∧ dc.score = 8

theorem cindy_hit_section_8 
  (cond : ∀ (dc : DartContest), ContestConditions dc) : 
  ∃ (dc : DartContest), isScoreSection8 dc := by
  sorry

end NUMINAMATH_GPT_cindy_hit_section_8_l1715_171542


namespace NUMINAMATH_GPT_trapezoid_is_proposition_l1715_171517

-- Define what it means to be a proposition
def is_proposition (s : String) : Prop := ∃ b : Bool, (s = "A trapezoid is a quadrilateral" ∨ s = "Construct line AB" ∨ s = "x is an integer" ∨ s = "Will it snow today?") ∧ 
  (b → s = "A trapezoid is a quadrilateral") 

-- Main proof statement
theorem trapezoid_is_proposition : is_proposition "A trapezoid is a quadrilateral" :=
  sorry

end NUMINAMATH_GPT_trapezoid_is_proposition_l1715_171517


namespace NUMINAMATH_GPT_maximize_x5y3_l1715_171592

theorem maximize_x5y3 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 30) :
  x = 18.75 ∧ y = 11.25 → (x^5 * y^3) = (18.75^5 * 11.25^3) :=
sorry

end NUMINAMATH_GPT_maximize_x5y3_l1715_171592


namespace NUMINAMATH_GPT_sin_minus_cos_sqrt_l1715_171586

theorem sin_minus_cos_sqrt (θ : ℝ) (b : ℝ) (h₁ : 0 < θ ∧ θ < π / 2) (h₂ : Real.cos (2 * θ) = b) :
  Real.sin θ - Real.cos θ = Real.sqrt (1 - b) :=
sorry

end NUMINAMATH_GPT_sin_minus_cos_sqrt_l1715_171586


namespace NUMINAMATH_GPT_pasha_encoded_expression_l1715_171514

theorem pasha_encoded_expression :
  2065 + 5 - 47 = 2023 :=
by
  sorry

end NUMINAMATH_GPT_pasha_encoded_expression_l1715_171514


namespace NUMINAMATH_GPT_fractional_part_wall_in_12_minutes_l1715_171526

-- Definitions based on given conditions
def time_to_paint_wall : ℕ := 60
def time_spent_painting : ℕ := 12

-- The goal is to prove that the fraction of the wall Mark can paint in 12 minutes is 1/5
theorem fractional_part_wall_in_12_minutes (t_pw: ℕ) (t_sp: ℕ) (h1: t_pw = 60) (h2: t_sp = 12) : 
  (t_sp : ℚ) / (t_pw : ℚ) = 1 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_fractional_part_wall_in_12_minutes_l1715_171526


namespace NUMINAMATH_GPT_number_of_shelves_l1715_171581

def initial_bears : ℕ := 17
def shipment_bears : ℕ := 10
def bears_per_shelf : ℕ := 9

theorem number_of_shelves : (initial_bears + shipment_bears) / bears_per_shelf = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_shelves_l1715_171581


namespace NUMINAMATH_GPT_unique_solution_l1715_171518

theorem unique_solution (k n : ℕ) (hk : k > 0) (hn : n > 0) (h : (7^k - 3^n) ∣ (k^4 + n^2)) : (k = 2 ∧ n = 4) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l1715_171518


namespace NUMINAMATH_GPT_whitney_money_left_over_l1715_171598

def total_cost (posters_cost : ℝ) (notebooks_cost : ℝ) (bookmarks_cost : ℝ) (pencils_cost : ℝ) (tax_rate : ℝ) :=
  let pre_tax := (3 * posters_cost) + (4 * notebooks_cost) + (5 * bookmarks_cost) + (2 * pencils_cost)
  let tax := pre_tax * tax_rate
  pre_tax + tax

def money_left_over (initial_money : ℝ) (total_cost : ℝ) :=
  initial_money - total_cost

theorem whitney_money_left_over :
  let initial_money := 40
  let posters_cost := 7.50
  let notebooks_cost := 5.25
  let bookmarks_cost := 3.10
  let pencils_cost := 1.15
  let tax_rate := 0.08
  money_left_over initial_money (total_cost posters_cost notebooks_cost bookmarks_cost pencils_cost tax_rate) = -26.20 :=
by
  sorry

end NUMINAMATH_GPT_whitney_money_left_over_l1715_171598


namespace NUMINAMATH_GPT_alcohol_quantity_in_mixture_l1715_171566

theorem alcohol_quantity_in_mixture : 
  ∃ (A W : ℕ), (A = 8) ∧ (A * 3 = 4 * W) ∧ (A * 5 = 4 * (W + 4)) :=
by
  sorry -- This is a placeholder; the proof itself is not required.

end NUMINAMATH_GPT_alcohol_quantity_in_mixture_l1715_171566


namespace NUMINAMATH_GPT_solve_for_x_l1715_171530

theorem solve_for_x : (∃ x : ℝ, 5 * x + 4 = -6) → x = -2 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1715_171530


namespace NUMINAMATH_GPT_total_glasses_l1715_171527

theorem total_glasses
  (x y : ℕ)
  (h1 : y = x + 16)
  (h2 : (12 * x + 16 * y) / (x + y) = 15) :
  12 * x + 16 * y = 480 :=
by
  sorry

end NUMINAMATH_GPT_total_glasses_l1715_171527


namespace NUMINAMATH_GPT_find_XY_base10_l1715_171585

theorem find_XY_base10 (X Y : ℕ) (h₁ : Y + 2 = X) (h₂ : X + 5 = 11) : X + Y = 10 := 
by 
  sorry

end NUMINAMATH_GPT_find_XY_base10_l1715_171585


namespace NUMINAMATH_GPT_find_numbers_l1715_171502

theorem find_numbers (x y z : ℝ) 
  (h1 : x = 280)
  (h2 : y = 200)
  (h3 : z = 220) :
  (x = 1.4 * y) ∧
  (x / z = 14 / 11) ∧
  (z - y = 0.125 * (x + y) - 40) :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l1715_171502


namespace NUMINAMATH_GPT_simplify_expansion_l1715_171506

theorem simplify_expansion (x : ℝ) : 
  (3 * x - 6) * (x + 8) - (x + 6) * (3 * x + 2) = -2 * x - 60 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expansion_l1715_171506


namespace NUMINAMATH_GPT_minimum_value_l1715_171529

def f (x : ℝ) : ℝ := |x - 4| + |x + 7| + |x - 5|

theorem minimum_value : ∃ x : ℝ, ∀ y : ℝ, f y ≥ f x ∧ f x = 4 :=
by
  -- Sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_minimum_value_l1715_171529


namespace NUMINAMATH_GPT_digits_of_number_l1715_171557

theorem digits_of_number (d : ℕ) (h1 : 0 ≤ d ∧ d ≤ 9) (h2 : (10 * (50 + d) + 2) % 6 = 0) : (5 * 10 + d) * 10 + 2 = 522 :=
by sorry

end NUMINAMATH_GPT_digits_of_number_l1715_171557


namespace NUMINAMATH_GPT_total_cases_of_candy_correct_l1715_171568

-- Define the number of cases of chocolate bars and lollipops
def cases_of_chocolate_bars : ℕ := 25
def cases_of_lollipops : ℕ := 55

-- Define the total number of cases of candy
def total_cases_of_candy : ℕ := cases_of_chocolate_bars + cases_of_lollipops

-- Prove that the total number of cases of candy is 80
theorem total_cases_of_candy_correct : total_cases_of_candy = 80 := by
  sorry

end NUMINAMATH_GPT_total_cases_of_candy_correct_l1715_171568


namespace NUMINAMATH_GPT_yellow_shirts_count_l1715_171543

theorem yellow_shirts_count (total_shirts blue_shirts green_shirts red_shirts yellow_shirts : ℕ) 
  (h1 : total_shirts = 36) 
  (h2 : blue_shirts = 8) 
  (h3 : green_shirts = 11) 
  (h4 : red_shirts = 6) 
  (h5 : yellow_shirts = total_shirts - (blue_shirts + green_shirts + red_shirts)) :
  yellow_shirts = 11 :=
by
  sorry

end NUMINAMATH_GPT_yellow_shirts_count_l1715_171543


namespace NUMINAMATH_GPT_total_earnings_correct_l1715_171539

section
  -- Define the conditions
  def wage : ℕ := 8
  def hours_Monday : ℕ := 8
  def hours_Tuesday : ℕ := 2

  -- Define the calculation for the total earnings
  def earnings (hours : ℕ) (rate : ℕ) : ℕ := hours * rate

  -- State the total earnings
  def total_earnings : ℕ := earnings hours_Monday wage + earnings hours_Tuesday wage

  -- Theorem: Prove that Will's total earnings in those two days is $80
  theorem total_earnings_correct : total_earnings = 80 := by
    sorry
end

end NUMINAMATH_GPT_total_earnings_correct_l1715_171539


namespace NUMINAMATH_GPT_amy_total_tickets_l1715_171534

def amy_initial_tickets : ℕ := 33
def amy_additional_tickets : ℕ := 21

theorem amy_total_tickets : amy_initial_tickets + amy_additional_tickets = 54 := by
  sorry

end NUMINAMATH_GPT_amy_total_tickets_l1715_171534


namespace NUMINAMATH_GPT_maximum_people_shaked_hands_l1715_171528

-- Given conditions
variables (N : ℕ) (hN : N > 4)
def has_not_shaken_hands_with (a b : ℕ) : Prop := sorry -- This should define the shaking hand condition

-- Main statement
theorem maximum_people_shaked_hands (h : ∃ i, has_not_shaken_hands_with i 2) :
  ∃ k, k = N - 3 := 
sorry

end NUMINAMATH_GPT_maximum_people_shaked_hands_l1715_171528


namespace NUMINAMATH_GPT_probability_exactly_2_boys_1_girl_probability_at_least_1_girl_l1715_171561

theorem probability_exactly_2_boys_1_girl 
  (boys girls : ℕ) 
  (total_group : ℕ) 
  (select : ℕ)
  (h_boys : boys = 4) 
  (h_girls : girls = 2) 
  (h_total_group : total_group = boys + girls) 
  (h_select : select = 3) : 
  (Nat.choose boys 2 * Nat.choose girls 1 / (Nat.choose total_group select) : ℚ) = 3 / 5 :=
by sorry

theorem probability_at_least_1_girl
  (boys girls : ℕ) 
  (total_group : ℕ) 
  (select : ℕ)
  (h_boys : boys = 4) 
  (h_girls : girls = 2) 
  (h_total_group : total_group = boys + girls) 
  (h_select : select = 3) : 
  (1 - (Nat.choose boys select / Nat.choose total_group select : ℚ)) = 4 / 5 :=
by sorry

end NUMINAMATH_GPT_probability_exactly_2_boys_1_girl_probability_at_least_1_girl_l1715_171561


namespace NUMINAMATH_GPT_interval_of_monotonic_decrease_minimum_value_in_interval_l1715_171532

noncomputable def f (x a : ℝ) : ℝ := 1 / x + a * Real.log x

-- Define the derivative of f
noncomputable def f_prime (x a : ℝ) : ℝ := (a * x - 1) / x^2

-- Prove that the interval of monotonic decrease is as specified
theorem interval_of_monotonic_decrease (a : ℝ) :
  if a ≤ 0 then ∀ x ∈ Set.Ioi (0 : ℝ), f_prime x a < 0
  else ∀ x ∈ Set.Ioo 0 (1/a), f_prime x a < 0 := sorry

-- Prove that, given x in [1/2, 1], the minimum value of f(x) is 0 when a = 2 / log 2
theorem minimum_value_in_interval :
  ∃ a : ℝ, (a = 2 / Real.log 2) ∧ ∀ x ∈ Set.Icc (1/2 : ℝ) 1, f x a ≥ 0 ∧ (∃ y ∈ Set.Icc (1/2 : ℝ) 1, f y a = 0) := sorry

end NUMINAMATH_GPT_interval_of_monotonic_decrease_minimum_value_in_interval_l1715_171532


namespace NUMINAMATH_GPT_percentage_difference_l1715_171584

theorem percentage_difference (water_yesterday : ℕ) (water_two_days_ago : ℕ) (h1 : water_yesterday = 48) (h2 : water_two_days_ago = 50) : 
  (water_two_days_ago - water_yesterday) / water_two_days_ago * 100 = 4 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1715_171584


namespace NUMINAMATH_GPT_no_such_increasing_seq_exists_l1715_171510

theorem no_such_increasing_seq_exists :
  ¬(∃ (a : ℕ → ℕ), (∀ m n : ℕ, a (m * n) = a m + a n) ∧ (∀ n : ℕ, a n < a (n + 1))) :=
by
  sorry

end NUMINAMATH_GPT_no_such_increasing_seq_exists_l1715_171510


namespace NUMINAMATH_GPT_conic_section_hyperbola_l1715_171576

theorem conic_section_hyperbola (x y : ℝ) : 
  (2 * x - 7)^2 - 4 * (y + 3)^2 = 169 → 
  -- Explain that this equation is of a hyperbola
  true := 
sorry

end NUMINAMATH_GPT_conic_section_hyperbola_l1715_171576


namespace NUMINAMATH_GPT_n_power_of_3_l1715_171555

theorem n_power_of_3 (n : ℕ) (h_prime : Prime (4^n + 2^n + 1)) : ∃ k : ℕ, n = 3^k := 
sorry

end NUMINAMATH_GPT_n_power_of_3_l1715_171555


namespace NUMINAMATH_GPT_mom_chicken_cost_l1715_171563

def cost_bananas : ℝ := 2 * 4 -- bananas cost
def cost_pears : ℝ := 2 -- pears cost
def cost_asparagus : ℝ := 6 -- asparagus cost
def total_expenses_other_than_chicken : ℝ := cost_bananas + cost_pears + cost_asparagus -- total cost of other items
def initial_money : ℝ := 55 -- initial amount of money
def remaining_money_after_other_purchases : ℝ := initial_money - total_expenses_other_than_chicken -- money left after covering other items

theorem mom_chicken_cost : 
  (remaining_money_after_other_purchases - 28 = 11) := 
by
  sorry

end NUMINAMATH_GPT_mom_chicken_cost_l1715_171563


namespace NUMINAMATH_GPT_solve_abs_equation_l1715_171569

theorem solve_abs_equation (y : ℝ) (h8 : y < 8) (h_eq : |y - 8| + 2 * y = 12) : y = 4 :=
sorry

end NUMINAMATH_GPT_solve_abs_equation_l1715_171569


namespace NUMINAMATH_GPT_sniper_B_has_greater_chance_of_winning_l1715_171525

def pA (n : ℕ) : ℝ :=
  if n = 1 then 0.4 else if n = 2 then 0.1 else if n = 3 then 0.5 else 0

def pB (n : ℕ) : ℝ :=
  if n = 1 then 0.1 else if n = 2 then 0.6 else if n = 3 then 0.3 else 0

noncomputable def expected_score (p : ℕ → ℝ) : ℝ :=
  (1 * p 1) + (2 * p 2) + (3 * p 3)

theorem sniper_B_has_greater_chance_of_winning :
  expected_score pB > expected_score pA :=
by
  sorry

end NUMINAMATH_GPT_sniper_B_has_greater_chance_of_winning_l1715_171525


namespace NUMINAMATH_GPT_division_remainder_l1715_171579

theorem division_remainder : 
  ∃ q r, 1234567 = 123 * q + r ∧ r < 123 ∧ r = 41 := 
by
  sorry

end NUMINAMATH_GPT_division_remainder_l1715_171579


namespace NUMINAMATH_GPT_shaded_area_of_modified_design_l1715_171544

noncomputable def radius_of_circles (side_length : ℝ) (grid_size : ℕ) : ℝ :=
  (side_length / grid_size) / 2

noncomputable def area_of_circle (radius : ℝ) : ℝ :=
  Real.pi * radius^2

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length^2

noncomputable def shaded_area (side_length : ℝ) (grid_size : ℕ) : ℝ :=
  let r := radius_of_circles side_length grid_size
  let total_circle_area := 9 * area_of_circle r
  area_of_square side_length - total_circle_area

theorem shaded_area_of_modified_design :
  shaded_area 24 3 = (576 - 144 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_of_modified_design_l1715_171544


namespace NUMINAMATH_GPT_carol_first_six_probability_l1715_171546

theorem carol_first_six_probability :
  let p := 1 / 6
  let q := 5 / 6
  let prob_cycle := q^4
  (p * q^3) / (1 - prob_cycle) = 125 / 671 :=
by
  sorry

end NUMINAMATH_GPT_carol_first_six_probability_l1715_171546


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1715_171535

-- Given conditions for the sequence
axiom pos_seq {a : ℕ → ℝ} : (∀ n : ℕ, 0 < a n)
axiom relation1 {a : ℕ → ℝ} (t : ℝ) : (∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
axiom relation2 {a : ℕ → ℝ} : 2 * (a 3) = (a 2) + (a 4)

-- Proof Requirements

-- (1) Find the value of (a1 + a3) / a2
theorem problem1 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) :
  (a 1 + a 3) / a 2 = 2 :=
sorry

-- (2) Prove that the sequence is an arithmetic sequence
theorem problem2 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) :
  ∀ n : ℕ, a (n+2) - a (n+1) = a (n+1) - a n :=
sorry

-- (3) Show p and r such that (1/a_k), (1/a_p), (1/a_r) form an arithmetic sequence
theorem problem3 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) (k : ℕ) (hk : k ≠ 0) :
  (k = 1 → ∀ p r : ℕ, ¬((k < p ∧ p < r) ∧ (1 / a k) + (1 / a r) = 2 * (1 / a p))) ∧ 
  (k ≥ 2 → ∃ p r : ℕ, (k < p ∧ p < r) ∧ (1 / a k) + (1 / a r) = 2 * (1 / a p) ∧ p = 2 * k - 1 ∧ r = k * (2 * k - 1)) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1715_171535


namespace NUMINAMATH_GPT_find_parabola_eq_l1715_171590

noncomputable def parabola_equation (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, y = -3 * x ^ 2 + 18 * x - 22 ↔ (x - 3) ^ 2 + 5 = y

theorem find_parabola_eq :
  ∃ a b c : ℝ, (vertex = (3, 5) ∧ axis_of_symmetry ∧ point_on_parabola = (2, 2)) →
  parabola_equation a b c :=
sorry

end NUMINAMATH_GPT_find_parabola_eq_l1715_171590


namespace NUMINAMATH_GPT_recurring_decimal_sum_as_fraction_l1715_171504

theorem recurring_decimal_sum_as_fraction :
  (0.2 + 0.03 + 0.0004) = 281 / 1111 := by
  sorry

end NUMINAMATH_GPT_recurring_decimal_sum_as_fraction_l1715_171504


namespace NUMINAMATH_GPT_giraffe_ratio_l1715_171516

theorem giraffe_ratio (g ng : ℕ) (h1 : g = 300) (h2 : g = ng + 290) : g / ng = 30 :=
by
  sorry

end NUMINAMATH_GPT_giraffe_ratio_l1715_171516


namespace NUMINAMATH_GPT_correct_exponentiation_l1715_171587

theorem correct_exponentiation (a : ℝ) : a^5 / a = a^4 := 
  sorry

end NUMINAMATH_GPT_correct_exponentiation_l1715_171587


namespace NUMINAMATH_GPT_buoy_radius_proof_l1715_171554

/-
We will define the conditions:
- width: 30 cm
- radius_ice_hole: 15 cm (half of width)
- depth: 12 cm
Then prove the radius of the buoy (r) equals 15.375 cm.
-/
noncomputable def radius_of_buoy : ℝ :=
  let width : ℝ := 30
  let depth : ℝ := 12
  let radius_ice_hole : ℝ := width / 2
  let r : ℝ := (369 / 24)
  r    -- the radius of the buoy

theorem buoy_radius_proof : radius_of_buoy = 15.375 :=
by 
  -- We assert that the above definition correctly computes the radius.
  sorry   -- Actual proof omitted

end NUMINAMATH_GPT_buoy_radius_proof_l1715_171554


namespace NUMINAMATH_GPT_flour_needed_l1715_171549

theorem flour_needed (flour_per_40_cookies : ℝ) (cookies : ℕ) (desired_cookies : ℕ) (flour_needed : ℝ) 
  (h1 : flour_per_40_cookies = 3) (h2 : cookies = 40) (h3 : desired_cookies = 100) :
  flour_needed = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_flour_needed_l1715_171549


namespace NUMINAMATH_GPT_total_number_of_workers_is_49_l1715_171564

-- Definitions based on the conditions
def avg_salary_all_workers := 8000
def num_technicians := 7
def avg_salary_technicians := 20000
def avg_salary_non_technicians := 6000

-- Prove that the total number of workers in the workshop is 49
theorem total_number_of_workers_is_49 :
  ∃ W, (avg_salary_all_workers * W = avg_salary_technicians * num_technicians + avg_salary_non_technicians * (W - num_technicians)) ∧ W = 49 := 
sorry

end NUMINAMATH_GPT_total_number_of_workers_is_49_l1715_171564


namespace NUMINAMATH_GPT_find_a_if_parallel_l1715_171507

-- Definitions of the vectors and the scalar a
def vector_m : ℝ × ℝ := (2, 1)
def vector_n (a : ℝ) : ℝ × ℝ := (4, a)

-- Condition for parallel vectors
def are_parallel (m n : ℝ × ℝ) : Prop :=
  m.1 / n.1 = m.2 / n.2

-- Lean 4 statement
theorem find_a_if_parallel (a : ℝ) (h : are_parallel vector_m (vector_n a)) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_if_parallel_l1715_171507


namespace NUMINAMATH_GPT_smallest_whole_number_larger_than_perimeter_l1715_171533

theorem smallest_whole_number_larger_than_perimeter {s : ℝ} (h1 : 16 < s) (h2 : s < 30) :
  61 > 7 + 23 + s :=
by
  sorry

end NUMINAMATH_GPT_smallest_whole_number_larger_than_perimeter_l1715_171533


namespace NUMINAMATH_GPT_line_y_axis_intersection_l1715_171567

-- Conditions: Line contains points (3, 20) and (-9, -6)
def line_contains_points : Prop :=
  ∃ m b : ℚ, ∀ (x y : ℚ), ((x = 3 ∧ y = 20) ∨ (x = -9 ∧ y = -6)) → (y = m * x + b)

-- Question: Prove that the line intersects the y-axis at (0, 27/2)
theorem line_y_axis_intersection :
  line_contains_points → (∃ (y : ℚ), y = 27/2) :=
by
  sorry

end NUMINAMATH_GPT_line_y_axis_intersection_l1715_171567


namespace NUMINAMATH_GPT_count_special_positive_integers_l1715_171573

theorem count_special_positive_integers : 
  ∃! n : ℕ, n < 10^6 ∧ 
  ∃ a b : ℕ, n = 2 * a^2 ∧ n = 3 * b^3 ∧ 
  ((n = 2592) ∨ (n = 165888)) :=
by
  sorry

end NUMINAMATH_GPT_count_special_positive_integers_l1715_171573
