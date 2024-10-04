import Mathlib

namespace intersection_sets_l89_89840

theorem intersection_sets (x : ℝ) :
  let M := {x | 2 * x - x^2 ≥ 0 }
  let N := {x | -1 < x ∧ x < 1}
  M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_sets_l89_89840


namespace sum_of_numbers_l89_89478

theorem sum_of_numbers (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : ab + bc + ca = 100) :
  a + b + c = 21 :=
sorry

end sum_of_numbers_l89_89478


namespace appended_number_divisible_by_12_l89_89646

theorem appended_number_divisible_by_12 :
  ∃ N, (N = 88) ∧ (∀ n, n ∈ finset.range N \ 71 → (let large_number := (list.range (N + 1)).filter (λ x, 71 ≤ x ∧ x ≤ N) in
       (list.foldr (λ a b, a * 100 + b) 0 large_number) % 12 = 0)) :=
by
  sorry

end appended_number_divisible_by_12_l89_89646


namespace dice_even_odd_equal_probability_l89_89954

theorem dice_even_odd_equal_probability :
  let p : ℚ := 35 / 128 in
  ∀ n : ℕ, n = 8 →
  ∀ k : ℕ, k = 4 →
  (∃ (binom : ℚ), binom = (Nat.choose n k)) →
  (∃ (prob : ℚ), prob = 1 / (2 ^ n)) →
  (∃ (total_prob : ℚ), total_prob = binom * prob) →
  total_prob = p :=
by
  intros n hn k hk binom hbinom prob hprob total_prob htotal_prob
  rw [hn, hk] at *
  cases hbinom with binom_val hbinom_val
  cases hprob with prob_val hprob_val
  rw hbinom_val at htotal_prob
  rw hprob_val at htotal_prob
  sorry

end dice_even_odd_equal_probability_l89_89954


namespace L_shaped_figure_perimeter_is_14_l89_89440

-- Define the side length of each square as a constant
def side_length : ℕ := 2

-- Define the horizontal base length
def base_length : ℕ := 3 * side_length

-- Define the height of the vertical stack
def vertical_stack_height : ℕ := 2 * side_length

-- Define the total perimeter of the "L" shaped figure
def L_shaped_figure_perimeter : ℕ :=
  base_length + side_length + vertical_stack_height + side_length + side_length + vertical_stack_height

-- The theorem that states the perimeter of the L-shaped figure is 14 units
theorem L_shaped_figure_perimeter_is_14 : L_shaped_figure_perimeter = 14 := sorry

end L_shaped_figure_perimeter_is_14_l89_89440


namespace polygon_interior_angles_sum_l89_89409

theorem polygon_interior_angles_sum {n : ℕ} 
  (h1 : ∀ (k : ℕ), k > 2 → (360 = k * 40)) :
  180 * (9 - 2) = 1260 :=
by
  sorry

end polygon_interior_angles_sum_l89_89409


namespace max_x_satisfying_ineq_l89_89463

theorem max_x_satisfying_ineq : ∃ (x : ℤ), (x ≤ 1 ∧ ∀ (y : ℤ), (y > x → y > 1) ∧ (y ≤ 1 → (y : ℚ) / 3 + 7 / 4 < 9 / 4)) := 
by
  sorry

end max_x_satisfying_ineq_l89_89463


namespace eggs_needed_per_month_l89_89271

def saly_needs : ℕ := 10
def ben_needs : ℕ := 14
def ked_needs : ℕ := ben_needs / 2
def weeks_in_month : ℕ := 4

def total_weekly_need : ℕ := saly_needs + ben_needs + ked_needs
def total_monthly_need : ℕ := total_weekly_need * weeks_in_month

theorem eggs_needed_per_month : total_monthly_need = 124 := by
  sorry

end eggs_needed_per_month_l89_89271


namespace woodworker_tables_l89_89932

theorem woodworker_tables
  (total_legs : ℕ)
  (legs_per_chair : ℕ)
  (legs_per_table : ℕ)
  (num_chairs : ℕ)
  (legs_needed_chairs : total_legs = legs_per_chair * num_chairs)
  (total_leg_equation : total_legs = 40)
  (num_chairs_equation : num_chairs = 6)
  (legs_per_chair_equation : legs_per_chair = 4)
  (legs_per_table_equation : legs_per_table = 4)
  : nat.div (total_legs - (legs_per_chair * num_chairs)) legs_per_table = 4 := 
by
  sorry

end woodworker_tables_l89_89932


namespace wine_with_cork_cost_is_2_10_l89_89923

noncomputable def cork_cost : ℝ := 0.05
noncomputable def wine_without_cork_cost : ℝ := cork_cost + 2.00
noncomputable def wine_with_cork_cost : ℝ := wine_without_cork_cost + cork_cost

theorem wine_with_cork_cost_is_2_10 : wine_with_cork_cost = 2.10 :=
by
  -- skipped proof
  sorry

end wine_with_cork_cost_is_2_10_l89_89923


namespace complex_norm_example_l89_89371

theorem complex_norm_example : 
  abs (-3 - (9 / 4 : ℝ) * I) = 15 / 4 := 
by
  sorry

end complex_norm_example_l89_89371


namespace car_speed_l89_89326

theorem car_speed (distance time : ℝ) (h_distance : distance = 275) (h_time : time = 5) : (distance / time = 55) :=
by
  sorry

end car_speed_l89_89326


namespace smallest_sum_a_b_l89_89298

theorem smallest_sum_a_b :
  ∃ (a b : ℕ), (7 * b - 4 * a = 3) ∧ a > 7 ∧ b > 7 ∧ a + b = 24 :=
by
  sorry

end smallest_sum_a_b_l89_89298


namespace general_formula_sum_formula_l89_89556

-- Define the geometric sequence
def geoseq (n : ℕ) : ℕ := 2^n

-- Define the sum of the first n terms of the geometric sequence
def sum_first_n_terms (n : ℕ) : ℕ := 2^(n+1) - 2

-- Given conditions
def a1 : ℕ := 2
def a4 : ℕ := 16

-- Theorem statements
theorem general_formula (n : ℕ) : 
  (geoseq 1 = a1) → (geoseq 4 = a4) → geoseq n = 2^n := sorry

theorem sum_formula (n : ℕ) : 
  (geoseq 1 = a1) → (geoseq 4 = a4) → sum_first_n_terms n = 2^(n+1) - 2 := sorry

end general_formula_sum_formula_l89_89556


namespace veranda_area_correct_l89_89015

-- Define the dimensions of the room.
def room_length : ℕ := 20
def room_width : ℕ := 12

-- Define the width of the veranda.
def veranda_width : ℕ := 2

-- Calculate the total dimensions with the veranda.
def total_length : ℕ := room_length + 2 * veranda_width
def total_width : ℕ := room_width + 2 * veranda_width

-- Calculate the area of the room and the total area including the veranda.
def room_area : ℕ := room_length * room_width
def total_area : ℕ := total_length * total_width

-- Prove that the area of the veranda is 144 m².
theorem veranda_area_correct : total_area - room_area = 144 := by
  sorry

end veranda_area_correct_l89_89015


namespace inverse_100_mod_101_l89_89052

theorem inverse_100_mod_101 : (100 * 100) % 101 = 1 :=
by
  -- Proof can be provided here.
  sorry

end inverse_100_mod_101_l89_89052


namespace point_lies_on_graph_l89_89394

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 2|

theorem point_lies_on_graph (a : ℝ) : f (-a) = f (a) :=
by
  sorry

end point_lies_on_graph_l89_89394


namespace percentage_blue_shirts_l89_89254

theorem percentage_blue_shirts (total_students := 600) 
 (percent_red := 23)
 (percent_green := 15)
 (students_other := 102)
 : (100 - (percent_red + percent_green + (students_other / total_students) * 100)) = 45 := by
  sorry

end percentage_blue_shirts_l89_89254


namespace right_triangle_area_inscribed_circle_l89_89021

theorem right_triangle_area_inscribed_circle (r a b c : ℝ)
  (h_c : c = 6 + 7)
  (h_a : a = 6 + r)
  (h_b : b = 7 + r)
  (h_pyth : (6 + r)^2 + (7 + r)^2 = 13^2):
  (1 / 2) * (a * b) = 42 :=
by
  -- The necessary calculations have already been derived and verified
  sorry

end right_triangle_area_inscribed_circle_l89_89021


namespace parabola_conditions_l89_89380

-- Definitions based on conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - 4*x - 3 + a

def passes_through (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f x = y

def intersects_at_2_points (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Proof Problem Statement
theorem parabola_conditions (a : ℝ) :
  (passes_through (quadratic_function a) 0 1 → a = 4) ∧
  (intersects_at_2_points (quadratic_function a) → (a = 3 ∨ a = 7)) :=
by
  sorry

end parabola_conditions_l89_89380


namespace dice_even_odd_equal_probability_l89_89952

noncomputable def probability_equal_even_odd_dice : ℚ :=
  let p : ℚ := 1 / 2 in
  let choose_8_4 : ℕ := Nat.choose 8 4 in
  choose_8_4 * (p^8)

theorem dice_even_odd_equal_probability :
  (probability_equal_even_odd_dice = 35 / 128) :=
by
  -- Formal proof goes here
  sorry

end dice_even_odd_equal_probability_l89_89952


namespace range_of_a_l89_89430

noncomputable def P (X : ℕ → ℝ) (i : ℕ) : ℝ := i / 10

theorem range_of_a (a : ℝ) :
  (P (λ i, P i) 1 + P (λ i, P i) 2 + P (λ i, P i) 3 = 3/5) →
  3 < a ∧ a ≤ 4 :=
by
  assume h : P (λ i, P i) 1 + P (λ i, P i) 2 + P (λ i, P i) 3 = 3/5
  sorry

end range_of_a_l89_89430


namespace not_all_yellow_l89_89792

def color := ℕ
def green : color := 0
def red : color := 1
def yellow : color := 2

noncomputable def next_color (left: color) (right: color) : color :=
  if left = right then left
  else if (left = green ∧ right = red) ∨ (left = red ∧ right = green) then yellow
  else if (left = green ∧ right = yellow) ∨ (left = yellow ∧ right = green) then red
  else green

noncomputable def next_colors (lights: list color) : list color := 
  list.map₂ next_color (lights.insert_nth (lights.length - 1) (lights.head_nth 0)) lights

theorem not_all_yellow : 
  ∃ (n : ℕ) (next_colors : ℕ → list color) (current : list color),
    n = 1998 ∧ 
    current.head_nth 0 = red ∧ 
    current.tail.all (= green) ∧ 
    ∀ i : ℕ, current = next_colors i → ∃ j : ℕ, next_colors j ↔ next_colors j.head_nth 0 ≠ yellow 
    :=
by
  sorry

end not_all_yellow_l89_89792


namespace inverse_100_mod_101_l89_89054

theorem inverse_100_mod_101 :
  ∃ x, (x : ℤ) ≡ 100 [MOD 101] ∧ 100 * x ≡ 1 [MOD 101] :=
by {
  use 100,
  split,
  { exact rfl },
  { norm_num }
}

end inverse_100_mod_101_l89_89054


namespace combined_work_time_l89_89770

def Worker_A_time : ℝ := 10
def Worker_B_time : ℝ := 15

theorem combined_work_time :
  (1 / Worker_A_time + 1 / Worker_B_time)⁻¹ = 6 := by
  sorry

end combined_work_time_l89_89770


namespace mean_age_of_children_l89_89889

theorem mean_age_of_children :
  let ages := [8, 8, 12, 12, 10, 14]
  let n := ages.length
  let sum_ages := ages.foldr (· + ·) 0
  let mean_age := sum_ages / n
  mean_age = 10 + 2 / 3 :=
by
  sorry

end mean_age_of_children_l89_89889


namespace area_OMVK_l89_89204

def AreaOfQuadrilateral (S_OKSL S_ONAM S_OMVK : ℝ) : ℝ :=
  let S_ABCD := 4 * (S_OKSL + S_ONAM)
  S_ABCD - S_OKSL - 24 - S_ONAM

theorem area_OMVK {S_OKSL S_ONAM : ℝ} (h_OKSL : S_OKSL = 6) (h_ONAM : S_ONAM = 12) : 
  AreaOfQuadrilateral S_OKSL S_ONAM 30 = 30 :=
by
  sorry

end area_OMVK_l89_89204


namespace inequality_solution_set_l89_89208

theorem inequality_solution_set :
  {x : ℝ | x ≠ 0 ∧ x ≠ 2 ∧ (x / (x - 2) + (x + 3) / (3 * x) ≥ 4)} =
  {x : ℝ | 0 < x ∧ x ≤ 1 / 8} ∪ {x : ℝ | 2 < x ∧ x ≤ 6} :=
by
  -- Proof will go here
  sorry

end inequality_solution_set_l89_89208


namespace percentage_reduction_l89_89629

theorem percentage_reduction :
  let P := 60
  let R := 45
  (900 / R) - (900 / P) = 5 →
  (P - R) / P * 100 = 25 :=
by 
  intros P R h
  have h1 : R = 45 := rfl
  have h2 : P = 60 := sorry
  rw [h1] at h
  rw [h2]
  sorry -- detailed steps to be filled in the proof

end percentage_reduction_l89_89629


namespace find_abc_sum_l89_89744

theorem find_abc_sum (A B C : ℤ) (h : ∀ x : ℝ, x^3 + A * x^2 + B * x + C = (x + 1) * (x - 3) * (x - 4)) : A + B + C = 11 :=
by {
  -- This statement asserts that, given the conditions, the sum A + B + C equals 11
  sorry
}

end find_abc_sum_l89_89744


namespace last_appended_number_is_84_l89_89645

theorem last_appended_number_is_84 : 
  ∃ N : ℕ, 
    let s := "7172737475767778798081" ++ (String.intercalate "" (List.map toString [82, 83, 84])) in
    (N = 84) ∧ (s.toNat % 12 = 0) :=
by
  sorry

end last_appended_number_is_84_l89_89645


namespace number_of_pencils_l89_89402

-- Define the given conditions
def circle_radius : ℝ := 14 -- 14 feet radius
def pencil_length_inches : ℝ := 6 -- 6-inch pencil

noncomputable def pencil_length_feet : ℝ := pencil_length_inches / 12 -- convert 6 inches to feet

-- Statement of the problem in Lean
theorem number_of_pencils (r : ℝ) (p_len_inch : ℝ) (d : ℝ) (p_len_feet : ℝ) :
  r = circle_radius →
  p_len_inch = pencil_length_inches →
  d = 2 * r →
  p_len_feet = pencil_length_feet →
  d / p_len_feet = 56 :=
by
  intros hr hp hd hpl
  sorry

end number_of_pencils_l89_89402


namespace no_solution_to_system_l89_89348

theorem no_solution_to_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 8) ∧ (6 * x - 8 * y = 18) :=
by
  sorry

end no_solution_to_system_l89_89348


namespace inverse_100_mod_101_l89_89053

theorem inverse_100_mod_101 :
  ∃ x, (x : ℤ) ≡ 100 [MOD 101] ∧ 100 * x ≡ 1 [MOD 101] :=
by {
  use 100,
  split,
  { exact rfl },
  { norm_num }
}

end inverse_100_mod_101_l89_89053


namespace solve_cubic_eq_l89_89672

theorem solve_cubic_eq (z : ℂ) : z^3 = 27 ↔ (z = 3 ∨ z = - (3 / 2) + (3 / 2) * Complex.I * Real.sqrt 3 ∨ z = - (3 / 2) - (3 / 2) * Complex.I * Real.sqrt 3) :=
by
  sorry

end solve_cubic_eq_l89_89672


namespace cheapest_lamp_cost_l89_89825

/--
Frank wants to buy a new lamp for his bedroom. The cost of the cheapest lamp is some amount, and the most expensive in the store is 3 times more expensive. Frank has $90, and if he buys the most expensive lamp available, he would have $30 remaining. Prove that the cost of the cheapest lamp is $20.
-/
theorem cheapest_lamp_cost (c most_expensive : ℝ) (h_cheapest_lamp : most_expensive = 3 * c) 
(h_frank_money : 90 - most_expensive = 30) : c = 20 := 
sorry

end cheapest_lamp_cost_l89_89825


namespace monotonically_increasing_range_of_a_l89_89839

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4 * x - 5)

theorem monotonically_increasing_range_of_a :
  ∀ (a : ℝ), (∀ x : ℝ, x > a → f x > f a) ↔ a ≥ 5 :=
by
  intro a
  unfold f
  sorry

end monotonically_increasing_range_of_a_l89_89839


namespace percent_of_y_equal_to_30_percent_of_60_percent_l89_89159

variable (y : ℝ)

theorem percent_of_y_equal_to_30_percent_of_60_percent (hy : y ≠ 0) :
  ((0.18 * y) / y) * 100 = 18 := by
  have hneq : y ≠ 0 := hy
  field_simp [hneq]
  norm_num
  sorry

end percent_of_y_equal_to_30_percent_of_60_percent_l89_89159


namespace sum_of_coordinates_A_l89_89425

-- Define the points A, B, and C and the given conditions
variables (A B C : ℝ × ℝ)
variables (h_ratio1 : dist A C / dist A B = 1 / 3)
variables (h_ratio2 : dist B C / dist A B = 1 / 3)
variables (h_B : B = (2, 8))
variables (h_C : C = (0, 2))

-- Lean 4 statement to prove the sum of the coordinates of A is -14
theorem sum_of_coordinates_A : (A.1 + A.2) = -14 :=
sorry

end sum_of_coordinates_A_l89_89425


namespace negation_of_exists_proposition_l89_89296

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 - 2 * x > 2) ↔ (∀ x : ℝ, x^2 - 2 * x ≤ 2) :=
by
  sorry

end negation_of_exists_proposition_l89_89296


namespace sum_of_last_two_digits_l89_89913

theorem sum_of_last_two_digits (x y : ℕ) : 
  x = 8 → y = 12 → (x^25 + y^25) % 100 = 0 := 
by
  intros hx hy
  sorry

end sum_of_last_two_digits_l89_89913


namespace find_eagle_feathers_times_l89_89103

theorem find_eagle_feathers_times (x : ℕ) (hawk_feathers : ℕ) (total_feathers_before_give : ℕ) (total_feathers : ℕ) (left_after_selling : ℕ) :
  hawk_feathers = 6 →
  total_feathers_before_give = 6 + 6 * x →
  total_feathers = total_feathers_before_give - 10 →
  left_after_selling = total_feathers / 2 →
  left_after_selling = 49 →
  x = 17 :=
by
  intros h_hawk h_total_before_give h_total h_left h_after_selling
  sorry

end find_eagle_feathers_times_l89_89103


namespace tangent_line_equation_at_point_l89_89289

-- Define the function y = f(x) = (2x - 1) / (x + 2)
def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

-- Define the point at which the tangent is evaluated
def point : ℝ × ℝ := (-1, -3)

-- Proof statement for the equation of the tangent line at the given point
theorem tangent_line_equation_at_point : 
  (∃ (a b c : ℝ), a * point.1 + b * point.2 + c = 0 ∧ a = 5 ∧ b = -1 ∧ c = 2) :=
sorry

end tangent_line_equation_at_point_l89_89289


namespace find_A_and_B_l89_89806

theorem find_A_and_B : 
  ∃ A B : ℝ, 
    (A = 6.5 ∧ B = 0.5) ∧
    (∀ x : ℝ, (8 * x - 17) / ((3 * x + 5) * (x - 3)) = A / (3 * x + 5) + B / (x - 3)) :=
by
  sorry

end find_A_and_B_l89_89806


namespace market_value_of_stock_l89_89176

-- Define the given conditions.
def face_value : ℝ := 100
def dividend_per_share : ℝ := 0.09 * face_value
def yield : ℝ := 0.08

-- State the problem: proving the market value of the stock.
theorem market_value_of_stock : (dividend_per_share / yield) * 100 = 112.50 := by
  -- Placeholder for the proof
  sorry

end market_value_of_stock_l89_89176


namespace train_car_count_l89_89369

theorem train_car_count
    (cars_first_15_sec : ℕ)
    (time_first_15_sec : ℕ)
    (total_time_minutes : ℕ)
    (total_additional_seconds : ℕ)
    (constant_speed : Prop)
    (h1 : cars_first_15_sec = 9)
    (h2 : time_first_15_sec = 15)
    (h3 : total_time_minutes = 3)
    (h4 : total_additional_seconds = 30)
    (h5 : constant_speed) :
    0.6 * (3 * 60 + 30) = 126 := by
  sorry

end train_car_count_l89_89369


namespace arthur_walks_distance_l89_89196

theorem arthur_walks_distance :
  ∀ (blocks_east blocks_north blocks_first blocks_other distance_first distance_other : ℕ)
  (fraction_first fraction_other : ℚ),
    blocks_east = 8 →
    blocks_north = 16 →
    blocks_first = 10 →
    blocks_other = (blocks_east + blocks_north) - blocks_first →
    fraction_first = 1 / 3 →
    fraction_other = 1 / 4 →
    distance_first = blocks_first * fraction_first →
    distance_other = blocks_other * fraction_other →
    (distance_first + distance_other) = 41 / 6 :=
by
  intros blocks_east blocks_north blocks_first blocks_other distance_first distance_other fraction_first fraction_other
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end arthur_walks_distance_l89_89196


namespace least_three_digit_7_heavy_l89_89340

/-- A number is 7-heavy if the remainder when the number is divided by 7 is greater than 4. -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- The statement to be proved: The least three-digit 7-heavy number is 104. -/
theorem least_three_digit_7_heavy : ∃ n, 100 ≤ n ∧ n < 1000 ∧ is_7_heavy(n) ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ is_7_heavy(m) → n ≤ m :=
begin
    use 104,
    split,
    { exact dec_trivial, },
    split,
    { exact dec_trivial, },
    split,
    { change 104 % 7 > 4,
      exact dec_trivial, },
    { intros m h1 h2,
      sorry
    }
end

end least_three_digit_7_heavy_l89_89340


namespace solve_for_b_l89_89698

theorem solve_for_b (b : ℚ) (h : b + 2 * b / 5 = 22 / 5) : b = 22 / 7 :=
sorry

end solve_for_b_l89_89698


namespace weeks_per_month_l89_89182

-- Define the given conditions
def num_employees_initial : Nat := 500
def additional_employees : Nat := 200
def hourly_wage : Nat := 12
def daily_work_hours : Nat := 10
def weekly_work_days : Nat := 5
def total_monthly_pay : Nat := 1680000

-- Calculate the total number of employees after hiring
def total_employees : Nat := num_employees_initial + additional_employees

-- Calculate the pay rates
def daily_pay_per_employee : Nat := hourly_wage * daily_work_hours
def weekly_pay_per_employee : Nat := daily_pay_per_employee * weekly_work_days

-- Calculate the total weekly pay for all employees
def total_weekly_pay : Nat := weekly_pay_per_employee * total_employees

-- Define the statement to be proved
theorem weeks_per_month
  (h1 : total_employees = num_employees_initial + additional_employees)
  (h2 : daily_pay_per_employee = hourly_wage * daily_work_hours)
  (h3 : weekly_pay_per_employee = daily_pay_per_employee * weekly_work_days)
  (h4 : total_weekly_pay = weekly_pay_per_employee * total_employees)
  (h5 : total_monthly_pay = 1680000) :
  total_monthly_pay / total_weekly_pay = 4 :=
by sorry

end weeks_per_month_l89_89182


namespace rectangular_prism_lateral_edge_length_l89_89406

-- Definition of the problem conditions
def is_rectangular_prism (v : ℕ) : Prop := v = 8
def sum_lateral_edges (l : ℕ) : ℕ := 4 * l

-- Theorem stating the problem to prove
theorem rectangular_prism_lateral_edge_length :
  ∀ (v l : ℕ), is_rectangular_prism v → sum_lateral_edges l = 56 → l = 14 :=
by
  intros v l h1 h2
  sorry

end rectangular_prism_lateral_edge_length_l89_89406


namespace tree_initial_leaves_l89_89555

theorem tree_initial_leaves (L : ℝ) (h1 : ∀ n : ℤ, 1 ≤ n ∧ n ≤ 4 → ∃ k : ℝ, L = k * (9/10)^n + k / 10^n)
                            (h2 : L * (9/10)^4 = 204) :
  L = 311 :=
by
  sorry

end tree_initial_leaves_l89_89555


namespace solve_for_k_l89_89973

theorem solve_for_k (x y k : ℤ) (h1 : x = -3) (h2 : y = 2) (h3 : 2 * x + k * y = 0) : k = 3 :=
by
  sorry

end solve_for_k_l89_89973


namespace condition_I_condition_II_l89_89984

noncomputable def f (x a : ℝ) : ℝ := |x - a|

-- Condition (I) proof problem
theorem condition_I (x : ℝ) (a : ℝ) (h : a = 1) :
  f x a ≥ 4 - |x - 1| ↔ (x ≤ -1 ∨ x ≥ 3) :=
by sorry

-- Condition (II) proof problem
theorem condition_II (a : ℝ) (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_f : ∀ x, f x a ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2)
    (h_eq : 1/m + 1/(2*n) = a) : mn ≥ 2 :=
by sorry

end condition_I_condition_II_l89_89984


namespace perception_arrangements_l89_89664

theorem perception_arrangements : 
  let n := 10
  let p := 2
  let e := 2
  let i := 2
  let r := 1
  let c := 1
  let t := 1
  let o := 1
  let n' := 1 
  (n.factorial / (p.factorial * e.factorial * i.factorial * r.factorial * c.factorial * t.factorial * o.factorial * n'.factorial)) = 453600 := 
by
  sorry

end perception_arrangements_l89_89664


namespace total_employees_l89_89860

variable (E : ℕ) -- E is the total number of employees

-- Conditions given in the problem
variable (male_fraction : ℚ := 0.45) -- 45% of the total employees are males
variable (males_below_50 : ℕ := 1170) -- 1170 males are below 50 years old
variable (males_total : ℕ := 2340) -- Total number of male employees

-- Condition derived from the problem (calculation of total males)
lemma male_employees_equiv (h : males_total = 2 * males_below_50) : males_total = 2340 :=
  by sorry

-- Main theorem
theorem total_employees (h : male_fraction * E = males_total) : E = 5200 :=
  by sorry

end total_employees_l89_89860


namespace items_left_in_store_l89_89652

theorem items_left_in_store: (4458 - 1561) + 575 = 3472 :=
by 
  sorry

end items_left_in_store_l89_89652


namespace no_supporters_l89_89949

theorem no_supporters (total_attendees : ℕ) (pct_first_team : ℕ) (pct_second_team : ℕ)
  (h1 : total_attendees = 50) (h2 : pct_first_team = 40) (h3 : pct_second_team = 34) :
  let supporters_first_team := (pct_first_team * total_attendees) / 100,
      supporters_second_team := (pct_second_team * total_attendees) / 100,
      total_supporters := supporters_first_team + supporters_second_team,
      no_support_count := total_attendees - total_supporters
  in no_support_count = 13 :=
by
  -- Definitions extracted from conditions
  let supporters_first_team := (pct_first_team * total_attendees) / 100
  let supporters_second_team := (pct_second_team * total_attendees) / 100
  let total_supporters := supporters_first_team + supporters_second_team
  let no_support_count := total_attendees - total_supporters
  
  -- Assume the conditions are already true
  have h1 : total_attendees = 50 := by sorry
  have h2 : pct_first_team = 40 := by sorry
  have h3 : pct_second_team = 34 := by sorry

  -- Start the proof
  calc
    no_support_count
        = 50 - (supporters_first_team + supporters_second_team) : by sorry
    ... = 50 - ((40 * 50) / 100 + (34 * 50) / 100) : by sorry
    ... = 50 - (20 + 17) : by sorry
    ... = 50 - 37 : by sorry
    ... = 13 : by sorry

end no_supporters_l89_89949


namespace fraction_exponentiation_l89_89967

theorem fraction_exponentiation :
  (1 / 3) ^ 5 = 1 / 243 :=
sorry

end fraction_exponentiation_l89_89967


namespace administrators_in_sample_l89_89774

theorem administrators_in_sample :
  let total_employees := 160
  let salespeople := 104
  let administrators := 32
  let logistics := 24
  let sample_size := 20
  let proportion_admin := administrators / total_employees
  let admin_in_sample := sample_size * proportion_admin
  admin_in_sample = 4 :=
by
  intros
  sorry

end administrators_in_sample_l89_89774


namespace fountain_area_l89_89493

theorem fountain_area (A B D C : ℝ) (h₁ : B - A = 20) (h₂ : D = (A + B) / 2) (h₃ : C - D = 12) :
  ∃ R : ℝ, R^2 = 244 ∧ π * R^2 = 244 * π :=
by
  sorry

end fountain_area_l89_89493


namespace cost_per_liter_of_gas_today_l89_89755

-- Definition of the conditions
def oil_price_rollback : ℝ := 0.4
def liters_today : ℝ := 10
def liters_friday : ℝ := 25
def total_liters := liters_today + liters_friday
def total_cost : ℝ := 39

-- The theorem to prove
theorem cost_per_liter_of_gas_today (C : ℝ) :
  (liters_today * C) + (liters_friday * (C - oil_price_rollback)) = total_cost →
  C = 1.4 := 
by 
  sorry

end cost_per_liter_of_gas_today_l89_89755


namespace solve_quadratic_equation_l89_89886

theorem solve_quadratic_equation (x : ℝ) : 
  2 * x^2 - 4 * x = 6 - 3 * x ↔ (x = -3/2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l89_89886


namespace num_arithmetic_sequences_l89_89805

theorem num_arithmetic_sequences (d : ℕ) (x : ℕ)
  (h_sum : 8 * x + 28 * d = 1080)
  (h_no180 : ∀ i, x + i * d ≠ 180)
  (h_pos : ∀ i, 0 < x + i * d)
  (h_less160 : ∀ i, x + i * d < 160)
  (h_not_equiangular : d ≠ 0) :
  ∃ n : ℕ, n = 3 :=
by sorry

end num_arithmetic_sequences_l89_89805


namespace average_age_when_youngest_born_l89_89305

theorem average_age_when_youngest_born (n : ℕ) (avg_age current_y : ℕ) (total_yr : ℕ) (reduction_yr yr_older : ℕ) (avg_age_older : ℕ) 
  (h1 : n = 7)
  (h2 : avg_age = 30)
  (h3 : current_y = 7)
  (h4 : total_yr = n * avg_age)
  (h5 : reduction_yr = (n - 1) * current_y)
  (h6 : yr_older = total_yr - reduction_yr)
  (h7 : avg_age_older = yr_older / (n - 1)) :
  avg_age_older = 28 :=
by 
  sorry

end average_age_when_youngest_born_l89_89305


namespace find_m_l89_89846

theorem find_m (x y m : ℤ) (h1 : x = 3) (h2 : y = 1) (h3 : x - m * y = 1) : m = 2 :=
by
  -- Proof goes here
  sorry

end find_m_l89_89846


namespace first_group_checked_correctly_l89_89810

-- Define the given conditions
def total_factories : ℕ := 169
def checked_by_second_group : ℕ := 52
def remaining_unchecked : ℕ := 48

-- Define the number of factories checked by the first group
def checked_by_first_group : ℕ := total_factories - checked_by_second_group - remaining_unchecked

-- State the theorem to be proved
theorem first_group_checked_correctly : checked_by_first_group = 69 :=
by
  -- The proof is not provided, use sorry to skip the proof steps
  sorry

end first_group_checked_correctly_l89_89810


namespace total_swordfish_catch_l89_89569

-- Definitions
def S_c : ℝ := 5 - 2
def S_m : ℝ := S_c - 1
def S_a : ℝ := 2 * S_m

def W_s : ℕ := 3  -- Number of sunny days
def W_r : ℕ := 2  -- Number of rainy days

-- Sunny and rainy day adjustments
def Shelly_sunny_catch : ℝ := S_c + 0.20 * S_c
def Sam_sunny_catch : ℝ := S_m + 0.20 * S_m
def Sara_sunny_catch : ℝ := S_a + 0.20 * S_a

def Shelly_rainy_catch : ℝ := S_c - 0.10 * S_c
def Sam_rainy_catch : ℝ := S_m - 0.10 * S_m
def Sara_rainy_catch : ℝ := S_a - 0.10 * S_a

-- Total catch calculations
def Shelly_total_catch : ℝ := W_s * Shelly_sunny_catch + W_r * Shelly_rainy_catch
def Sam_total_catch : ℝ := W_s * Sam_sunny_catch + W_r * Sam_rainy_catch
def Sara_total_catch : ℝ := W_s * Sara_sunny_catch + W_r * Sara_rainy_catch

def Total_catch : ℝ := Shelly_total_catch + Sam_total_catch + Sara_total_catch

-- Proof statement
theorem total_swordfish_catch : ⌊Total_catch⌋ = 48 := 
  by sorry

end total_swordfish_catch_l89_89569


namespace hyperbola_real_axis_length_l89_89817

theorem hyperbola_real_axis_length (x y : ℝ) :
  x^2 - y^2 / 9 = 1 → 2 = 2 :=
by
  sorry

end hyperbola_real_axis_length_l89_89817


namespace credit_card_more_beneficial_l89_89617

def gift_cost : ℝ := 8000
def credit_card_cashback_rate : ℝ := 0.005
def debit_card_cashback_rate : ℝ := 0.0075
def debit_card_interest_rate : ℝ := 0.005

def credit_card_total_income : ℝ := gift_cost * (credit_card_cashback_rate + debit_card_interest_rate)
def debit_card_total_income : ℝ := gift_cost * debit_card_cashback_rate

theorem credit_card_more_beneficial :
  credit_card_total_income > debit_card_total_income :=
by
  sorry

end credit_card_more_beneficial_l89_89617


namespace james_marbles_left_l89_89102

theorem james_marbles_left :
  ∀ (initial_marbles bags remaining_bags marbles_per_bag left_marbles : ℕ),
  initial_marbles = 28 →
  bags = 4 →
  marbles_per_bag = initial_marbles / bags →
  remaining_bags = bags - 1 →
  left_marbles = remaining_bags * marbles_per_bag →
  left_marbles = 21 :=
by
  intros initial_marbles bags remaining_bags marbles_per_bag left_marbles
  sorry

end james_marbles_left_l89_89102


namespace sum_consecutive_even_integers_l89_89585

theorem sum_consecutive_even_integers (m : ℤ) :
  (m + (m + 2) + (m + 4) + (m + 6) + (m + 8)) = 5 * m + 20 := by
  sorry

end sum_consecutive_even_integers_l89_89585


namespace exists_three_digit_numbers_with_property_l89_89192

open Nat

def is_three_digit_number (n : ℕ) : Prop := (100 ≤ n ∧ n < 1000)

def distinct_digits (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def inserts_zeros_and_is_square (n : ℕ) (k : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  let transformed_number := a * 10^(2*k + 2) + b * 10^(k + 1) + c
  ∃ x : ℕ, transformed_number = x * x

theorem exists_three_digit_numbers_with_property:
  ∃ n1 n2 : ℕ, 
    is_three_digit_number n1 ∧ 
    is_three_digit_number n2 ∧ 
    distinct_digits n1 ∧ 
    distinct_digits n2 ∧ 
    ( ∀ k, inserts_zeros_and_is_square n1 k ) ∧ 
    ( ∀ k, inserts_zeros_and_is_square n2 k ) ∧ 
    n1 ≠ n2 := 
sorry

end exists_three_digit_numbers_with_property_l89_89192


namespace range_a_sub_b_mul_c_l89_89529

theorem range_a_sub_b_mul_c (a b c : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) (h4 : 2 < c) (h5 : c < 3) :
  -6 < (a - b) * c ∧ (a - b) * c < 0 :=
by
  -- We need to prove the range of (a - b) * c is within (-6, 0)
  sorry

end range_a_sub_b_mul_c_l89_89529


namespace compute_f_at_919_l89_89684

-- Given conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) : Prop :=
∀ x, f (x + 4) = f (x - 2)

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ [-3, 0] then 6^(-x) else sorry

-- Lean statement for the proof problem
theorem compute_f_at_919 (f : ℝ → ℝ)
    (h_even : is_even_function f)
    (h_periodic : periodic_function f)
    (h_defined : ∀ x ∈ [-3, 0], f x = 6^(-x)) :
    f 919 = 6 := sorry

end compute_f_at_919_l89_89684


namespace total_number_of_animals_l89_89310

-- Prove that the total number of animals is 300 given the conditions described.
theorem total_number_of_animals (A : ℕ) (H₁ : 4 * (A / 3) = 400) : A = 300 :=
sorry

end total_number_of_animals_l89_89310


namespace joe_egg_count_l89_89560

theorem joe_egg_count : 
  let clubhouse : ℕ := 12
  let park : ℕ := 5
  let townhall : ℕ := 3
  clubhouse + park + townhall = 20 :=
by
  sorry

end joe_egg_count_l89_89560


namespace probability_two_or_more_women_l89_89086

-- Definitions based on the conditions
def men : ℕ := 8
def women : ℕ := 4
def total_people : ℕ := men + women
def chosen_people : ℕ := 4

-- Function to calculate the probability of a specific event
noncomputable def probability_event (event_count : ℕ) (total_count : ℕ) : ℚ :=
  event_count / total_count

-- Function to calculate the combination (binomial coefficient)
noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probability calculations based on steps given in the solution:
noncomputable def prob_no_women : ℚ :=
  probability_event ((men - 0) * (men - 1) * (men - 2) * (men - 3)) (total_people * (total_people - 1) * (total_people - 2) * (total_people - 3))

noncomputable def prob_exactly_one_woman : ℚ :=
  probability_event (binom women 1 * binom men 3) (binom total_people chosen_people)

noncomputable def prob_fewer_than_two_women : ℚ :=
  prob_no_women + prob_exactly_one_woman

noncomputable def prob_at_least_two_women : ℚ :=
  1 - prob_fewer_than_two_women

-- The main theorem to be proved
theorem probability_two_or_more_women :
  prob_at_least_two_women = 67 / 165 :=
sorry

end probability_two_or_more_women_l89_89086


namespace trees_died_in_typhoon_imply_all_died_l89_89933

-- Given conditions
def trees_initial := 3
def survived_trees (x : Int) := x
def died_trees (x : Int) := x + 23

-- Prove that the number of died trees is 3
theorem trees_died_in_typhoon_imply_all_died : ∀ x, 2 * survived_trees x + 23 = trees_initial → trees_initial = died_trees x := 
by
  intro x h
  sorry

end trees_died_in_typhoon_imply_all_died_l89_89933


namespace num_distinct_combinations_l89_89332

-- Define the conditions
def num_dials : Nat := 4
def digits : List Nat := List.range 10  -- Digits from 0 to 9

-- Define what it means for a combination to have distinct digits
def distinct_digits (comb : List Nat) : Prop :=
  comb.length = num_dials ∧ comb.Nodup

-- The main statement for the theorem
theorem num_distinct_combinations : 
  ∃ (n : Nat), n = 5040 ∧ ∀ comb : List Nat, distinct_digits comb → comb.length = num_dials →
  (List.permutations digits).length = n :=
by
  sorry

end num_distinct_combinations_l89_89332


namespace train_length_is_300_l89_89789

noncomputable def length_of_train (V L : ℝ) : Prop :=
  (L = V * 18) ∧ (L + 500 = V * 48)

theorem train_length_is_300
  (V : ℝ) (L : ℝ) (h : length_of_train V L) : L = 300 :=
by
  sorry

end train_length_is_300_l89_89789


namespace prime_in_range_l89_89848

theorem prime_in_range (p: ℕ) (h_prime: Nat.Prime p) (h_int_roots: ∃ a b: ℤ, a ≠ b ∧ a + b = -p ∧ a * b = -520 * p) : 11 < p ∧ p ≤ 21 := 
by
  sorry

end prime_in_range_l89_89848


namespace find_function_perfect_square_condition_l89_89225

theorem find_function_perfect_square_condition (g : ℕ → ℕ)
  (h : ∀ m n : ℕ, ∃ k : ℕ, (g m + n) * (g n + m) = k * k) :
  ∃ c : ℕ, ∀ m : ℕ, g m = m + c :=
sorry

end find_function_perfect_square_condition_l89_89225


namespace billy_videos_within_limit_l89_89796

def total_videos_watched_within_time_limit (time_limit : ℕ) (video_time : ℕ) (search_time : ℕ) (break_time : ℕ) (num_trials : ℕ) (videos_per_trial : ℕ) (categories : ℕ) (videos_per_category : ℕ) : ℕ :=
  let total_trial_time := videos_per_trial * video_time + search_time + break_time
  let total_category_time := videos_per_category * video_time
  let full_trial_time := num_trials * total_trial_time
  let full_category_time := categories * total_category_time
  let total_time := full_trial_time + full_category_time
  let non_watching_time := search_time * num_trials + break_time * (num_trials - 1)
  let available_time := time_limit - non_watching_time
  let max_videos := available_time / video_time
  max_videos

theorem billy_videos_within_limit : total_videos_watched_within_time_limit 90 4 3 5 5 15 2 10 = 13 := by
  sorry

end billy_videos_within_limit_l89_89796


namespace find_first_discount_l89_89033

-- Definitions for the given conditions
def list_price : ℝ := 150
def final_price : ℝ := 105
def second_discount : ℝ := 12.5

-- Statement representing the mathematical proof problem
theorem find_first_discount (x : ℝ) : 
  list_price * ((100 - x) / 100) * ((100 - second_discount) / 100) = final_price → x = 20 :=
by
  sorry

end find_first_discount_l89_89033


namespace negative_double_inequality_l89_89316

theorem negative_double_inequality (a : ℝ) (h : a < 0) : 2 * a < a :=
by { sorry }

end negative_double_inequality_l89_89316


namespace permutations_PERCEPTION_l89_89359

-- Define the word "PERCEPTION" and its letter frequencies
def word : String := "PERCEPTION"

def freq_P : Nat := 2
def freq_E : Nat := 2
def freq_R : Nat := 1
def freq_C : Nat := 1
def freq_T : Nat := 1
def freq_I : Nat := 1
def freq_O : Nat := 1
def freq_N : Nat := 1

-- Define the total number of letters in the word
def total_letters : Nat := 10

-- Calculate the number of permutations for the multiset
def permutations : Nat :=
  total_letters.factorial / (freq_P.factorial * freq_E.factorial)

-- Proof problem
theorem permutations_PERCEPTION :
  permutations = 907200 :=
by
  sorry

end permutations_PERCEPTION_l89_89359


namespace four_digit_property_l89_89707

-- Define the problem conditions and statement
theorem four_digit_property (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 0 ≤ y ∧ y < 100) :
  (100 * x + y = (x + y) ^ 2) ↔ (100 * x + y = 3025 ∨ 100 * x + y = 2025 ∨ 100 * x + y = 9801) := by
sorry

end four_digit_property_l89_89707


namespace f_at_3_l89_89851

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x - 1

-- The theorem to prove
theorem f_at_3 : f 3 = 5 := sorry

end f_at_3_l89_89851


namespace range_of_a_l89_89975

def p (a : ℝ) := 0 < a ∧ a < 1
def q (a : ℝ) := a > 5 / 2 ∨ 0 < a ∧ a < 1 / 2

theorem range_of_a (a : ℝ) :
  (a > 0) ∧ (a ≠ 1) ∧ (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ (1 / 2 ≤ a ∧ a < 1) ∨ (a > 5 / 2) :=
sorry

end range_of_a_l89_89975


namespace find_integer_solution_of_equations_l89_89662

theorem find_integer_solution_of_equations :
  ∃ (s : Finset (ℤ × ℤ × ℤ)),
    s = {⟨3, 8, 5⟩, ⟨8, 3, 5⟩, ⟨3, -5, -8⟩, ⟨-5, 8, -3⟩, ⟨-5, 3, -8⟩, ⟨8, -5, -3⟩} ∧
    ∀ (x y z : ℤ), 
    (⟨x, y, z⟩ ∈ s ↔ x + y - z = 6 ∧ x^3 + y^3 - z^3 = 414) := by
  sorry

end find_integer_solution_of_equations_l89_89662


namespace division_multiplication_l89_89613

-- Given a number x, we want to prove that (x / 6) * 12 = 2 * x under basic arithmetic operations.

theorem division_multiplication (x : ℝ) : (x / 6) * 12 = 2 * x := 
by
  sorry

end division_multiplication_l89_89613


namespace find_x_plus_y_l89_89454

theorem find_x_plus_y
  (x y : ℝ)
  (hx : x^3 - 3 * x^2 + 5 * x - 17 = 0)
  (hy : y^3 - 3 * y^2 + 5 * y + 11 = 0) :
  x + y = 2 := 
sorry

end find_x_plus_y_l89_89454


namespace find_window_width_on_second_wall_l89_89266

noncomputable def total_wall_area (width length height: ℝ) : ℝ :=
  4 * width * height

noncomputable def doorway_area (width height : ℝ) : ℝ :=
  width * height

noncomputable def window_area (width height : ℝ) : ℝ :=
  width * height

theorem find_window_width_on_second_wall :
  let room_width := 20
  let room_length := 20
  let room_height := 8
  let first_doorway_width := 3
  let first_doorway_height := 7
  let second_doorway_width := 5
  let second_doorway_height := 7
  let window_height := 4
  let area_to_paint := 560
  let total_area := total_wall_area room_width room_length room_height
  let first_doorway := doorway_area first_doorway_width first_doorway_height
  let second_doorway := doorway_area second_doorway_width second_doorway_height
  total_area - first_doorway - second_doorway - window_area w window_height = area_to_paint
  → w = 6 :=
by
  let room_width := 20
  let room_length := 20
  let room_height := 8
  let first_doorway_width := 3
  let first_doorway_height := 7
  let second_doorway_width := 5
  let second_doorway_height := 7
  let window_height := 4
  let area_to_paint := 560
  let total_area := total_wall_area room_width room_length room_height
  let first_doorway := doorway_area first_doorway_width first_doorway_height
  let second_doorway := doorway_area second_doorway_width second_doorway_height
  sorry

end find_window_width_on_second_wall_l89_89266


namespace minimum_value_of_3x_plus_4y_exists_x_y_for_minimum_value_l89_89542

theorem minimum_value_of_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
sorry

theorem exists_x_y_for_minimum_value : ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y ∧ 3 * x + 4 * y = 5 :=
sorry

end minimum_value_of_3x_plus_4y_exists_x_y_for_minimum_value_l89_89542


namespace range_of_m_l89_89985

/-- Define the domain set A where the function f(x) = 1 / sqrt(4 + 3x - x^2) is defined. -/
def A : Set ℝ := {x | -1 < x ∧ x < 4}

/-- Define the range set B where the function g(x) = - x^2 - 2x + 2, with x in [-1, 1], is defined. -/
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

/-- Define the set C in terms of m. -/
def C (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 2}

/-- Prove the range of the real number m such that C ∩ (A ∪ B) = C. -/
theorem range_of_m : {m : ℝ | C m ⊆ A ∪ B} = {m | -1 ≤ m ∧ m < 2} :=
by
  sorry

end range_of_m_l89_89985


namespace Ms_Rush_Speed_to_be_on_time_l89_89275

noncomputable def required_speed (d t r : ℝ) :=
  d = 50 * (t + 1/12) ∧ 
  d = 70 * (t - 1/9) →
  r = d / t →
  r = 74

theorem Ms_Rush_Speed_to_be_on_time 
  (d t r : ℝ) 
  (h1 : d = 50 * (t + 1/12)) 
  (h2 : d = 70 * (t - 1/9)) 
  (h3 : r = d / t) : 
  r = 74 :=
sorry

end Ms_Rush_Speed_to_be_on_time_l89_89275


namespace jamies_score_l89_89857

def quiz_score (correct incorrect unanswered : ℕ) : ℚ :=
  (correct * 2) + (incorrect * (-0.5)) + (unanswered * 0.25)

theorem jamies_score :
  quiz_score 16 10 4 = 28 :=
by
  sorry

end jamies_score_l89_89857


namespace smallest_diff_mod_13_l89_89567

theorem smallest_diff_mod_13 : 
  let m := Nat.find (λ k, 100 ≤ 13 * k + 7)
  let n := Nat.find (λ k, 1000 ≤ 13 * k + 7)
  (13 * n + 7) - (13 * m + 7) = 895 :=
by
  sorry

end smallest_diff_mod_13_l89_89567


namespace scientific_notation_of_6_ronna_l89_89035

def "ronna" := 27

theorem scientific_notation_of_6_ronna :
  "ronna" = 27 → 6 * 10^27 = 6 * 10^27 := 
by
  intro ronna_def
  rw ronna_def
  exact rfl

end scientific_notation_of_6_ronna_l89_89035


namespace find_unique_f_l89_89811

theorem find_unique_f (f : ℝ → ℝ) (h : ∀ x y z : ℝ, f (x * y) + f (x * z) ≥ f (x) * f (y * z) + 1) : 
    ∀ x : ℝ, f x = 1 :=
by
  sorry

end find_unique_f_l89_89811


namespace fraction_product_l89_89940

theorem fraction_product :
  ((1: ℚ) / 2) * (3 / 5) * (7 / 11) = 21 / 110 :=
by {
  sorry
}

end fraction_product_l89_89940


namespace distance_between_parallel_lines_l89_89814

theorem distance_between_parallel_lines : 
  ∀ (x y : ℝ), 
  (3 * x - 4 * y - 3 = 0) ∧ (6 * x - 8 * y + 5 = 0) → 
  ∃ d : ℝ, d = 11 / 10 :=
by
  sorry

end distance_between_parallel_lines_l89_89814


namespace tom_teaching_years_l89_89607

theorem tom_teaching_years :
  ∃ T D : ℕ, T + D = 70 ∧ D = (1 / 2) * T - 5 ∧ T = 50 :=
by
  sorry

end tom_teaching_years_l89_89607


namespace find_months_contributed_l89_89922

theorem find_months_contributed (x : ℕ) (profit_A profit_total : ℝ)
  (contrib_A : ℝ) (contrib_B : ℝ) (months_B : ℕ) :
  profit_A / profit_total = (contrib_A * x) / (contrib_A * x + contrib_B * months_B) →
  profit_A = 4800 →
  profit_total = 8400 →
  contrib_A = 5000 →
  contrib_B = 6000 →
  months_B = 5 →
  x = 8 :=
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  sorry

end find_months_contributed_l89_89922


namespace range_of_p_l89_89106

def h (x : ℝ) : ℝ := 4 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_of_p : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → -1 ≤ p x ∧ p x ≤ 1023 :=
by
  sorry

end range_of_p_l89_89106


namespace part1_part2_l89_89563

open Set

noncomputable def A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1: ∀ a : ℝ, A = B a → a = 1 := by
  intros a h
  sorry

theorem part2: ∀ a : ℝ, B a ⊆ A ∧ a > 0 → a = 1 := by
  intros a h
  sorry

end part1_part2_l89_89563


namespace bowl_capacity_percentage_l89_89626

theorem bowl_capacity_percentage
    (initial_half_full : ℕ)
    (added_water : ℕ)
    (total_water : ℕ)
    (full_capacity : ℕ)
    (percentage_filled : ℚ) :
    initial_half_full * 2 = full_capacity →
    initial_half_full + added_water = total_water →
    added_water = 4 →
    total_water = 14 →
    percentage_filled = (total_water * 100) / full_capacity →
    percentage_filled = 70 := 
by
    intros h1 h2 h3 h4 h5
    sorry

end bowl_capacity_percentage_l89_89626


namespace yield_is_eight_percent_l89_89779

noncomputable def par_value : ℝ := 100
noncomputable def annual_dividend : ℝ := 0.12 * par_value
noncomputable def market_value : ℝ := 150
noncomputable def yield_percentage : ℝ := (annual_dividend / market_value) * 100

theorem yield_is_eight_percent : yield_percentage = 8 := 
by 
  sorry

end yield_is_eight_percent_l89_89779


namespace fraction_equals_seven_twentyfive_l89_89043

theorem fraction_equals_seven_twentyfive :
  (1722^2 - 1715^2) / (1731^2 - 1706^2) = (7 / 25) :=
by
  sorry

end fraction_equals_seven_twentyfive_l89_89043


namespace expand_product_l89_89218

theorem expand_product (x : ℝ) : 5 * (x + 2) * (x + 6) * (x - 1) = 5 * x^3 + 35 * x^2 + 20 * x - 60 := 
by
  sorry

end expand_product_l89_89218


namespace inequality_solution_l89_89231

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem inequality_solution (k : ℝ) (h_pos : 0 < k) :
  (0 < k ∧ k < 1 ∧ (1 : ℝ) < x ∧ x < (1 / k)) ∨
  (k = 1 ∧ False) ∨
  (1 < k ∧ (1 / k) < x ∧ x < 1)
  ∨ False :=
sorry

end inequality_solution_l89_89231


namespace tory_toys_sold_is_7_l89_89036

-- Define the conditions as Lean definitions
def bert_toy_phones_sold : Nat := 8
def price_per_toy_phone : Nat := 18
def bert_earnings : Nat := bert_toy_phones_sold * price_per_toy_phone
def tory_earnings : Nat := bert_earnings - 4
def price_per_toy_gun : Nat := 20
def tory_toys_sold := tory_earnings / price_per_toy_gun

-- Prove that the number of toy guns Tory sold is 7
theorem tory_toys_sold_is_7 : tory_toys_sold = 7 :=
by
  sorry

end tory_toys_sold_is_7_l89_89036


namespace condition_1_condition_2_l89_89467

theorem condition_1 (m : ℝ) : (m^2 - 2*m - 15 > 0) ↔ (m < -3 ∨ m > 5) :=
sorry

theorem condition_2 (m : ℝ) : (2*m^2 + 3*m - 9 = 0) ∧ (7*m + 21 ≠ 0) ↔ (m = 3/2) :=
sorry

end condition_1_condition_2_l89_89467


namespace polygon_area_is_nine_l89_89214

-- Definitions of vertices and coordinates.
def vertexA := (0, 0)
def vertexD := (3, 0)
def vertexP := (3, 3)
def vertexM := (0, 3)

-- Area of the polygon formed by the vertices A, D, P, M.
def polygonArea (A D P M : ℕ × ℕ) : ℕ :=
  (D.1 - A.1) * (P.2 - A.2)

-- Statement of the theorem.
theorem polygon_area_is_nine : polygonArea vertexA vertexD vertexP vertexM = 9 := by
  sorry

end polygon_area_is_nine_l89_89214


namespace interest_rate_l89_89089

noncomputable def simple_interest (P r t: ℝ) : ℝ := P * r * t / 100

noncomputable def compound_interest (P r t: ℝ) : ℝ := P * (1 + r / 100) ^ t - P

theorem interest_rate (P r: ℝ) (h1: simple_interest P r 2 = 50) (h2: compound_interest P r 2 = 51.25) : r = 5 :=
by
  sorry

end interest_rate_l89_89089


namespace xiao_hua_seat_correct_l89_89705

-- Define the classroom setup
def classroom : Type := ℤ × ℤ

-- Define the total number of rows and columns in the classroom.
def total_rows : ℤ := 7
def total_columns : ℤ := 8

-- Define the position of Xiao Ming's seat.
def xiao_ming_seat : classroom := (3, 7)

-- Define the position of Xiao Hua's seat.
def xiao_hua_seat : classroom := (5, 2)

-- Prove that Xiao Hua's seat is designated as (5, 2)
theorem xiao_hua_seat_correct : xiao_hua_seat = (5, 2) := by
  -- The proof would go here
  sorry

end xiao_hua_seat_correct_l89_89705


namespace instantaneous_velocity_at_t4_l89_89323

def position (t : ℝ) : ℝ := t^2 - t + 2

theorem instantaneous_velocity_at_t4 : 
  (deriv position 4) = 7 := 
by
  sorry

end instantaneous_velocity_at_t4_l89_89323


namespace arithmetic_sequence_common_difference_l89_89688

theorem arithmetic_sequence_common_difference 
    (a_2 : ℕ → ℕ) (S_4 : ℕ) (a_n : ℕ → ℕ → ℕ) (S_n : ℕ → ℕ → ℕ → ℕ)
    (h1 : a_2 2 = 3) (h2 : S_4 = 16) 
    (h3 : ∀ n a_1 d, a_n a_1 n = a_1 + (n-1)*d)
    (h4 : ∀ n a_1 d, S_n n a_1 d = n / 2 * (2*a_1 + (n-1)*d)) : ∃ d, d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l89_89688


namespace allison_greater_prob_l89_89193

noncomputable def prob_allison_greater (p_brian : ℝ) (p_noah : ℝ) : ℝ :=
  p_brian * p_noah

theorem allison_greater_prob : prob_allison_greater (2/3) (1/2) = 1/3 :=
by {
  -- Calculate the combined probability
  sorry
}

end allison_greater_prob_l89_89193


namespace is_condition_B_an_algorithm_l89_89917

-- Definitions of conditions A, B, C, D
def condition_A := "At home, it is generally the mother who cooks"
def condition_B := "The steps to cook rice include washing the pot, rinsing the rice, adding water, and heating"
def condition_C := "Cooking outdoors is called camping cooking"
def condition_D := "Rice is necessary for cooking"

-- Definition of being considered an algorithm
def is_algorithm (s : String) : Prop :=
  s = condition_B  -- Based on the analysis that condition_B meets the criteria of an algorithm

-- The proof statement to show that condition_B can be considered an algorithm
theorem is_condition_B_an_algorithm : is_algorithm condition_B :=
by
  sorry

end is_condition_B_an_algorithm_l89_89917


namespace final_number_appended_is_84_l89_89649

noncomputable def arina_sequence := "7172737475767778798081"

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_divisible_by_12 (n : ℕ) : Prop := n % 12 = 0

-- Define adding numbers to the sequence
def append_number (seq : String) (n : ℕ) : String := seq ++ n.repr

-- Create the full sequence up to 84 and check if it's divisible by 12
def generate_full_sequence : String :=
  let base_seq := arina_sequence
  let full_seq := append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number arina_sequence 82) 83) 84))) 85) 86) 87) 88 
  full_seq

theorem final_number_appended_is_84 : (∃ seq : String, is_divisible_by_12(seq.to_nat) ∧ seq.ends_with "84") := 
by
  sorry

end final_number_appended_is_84_l89_89649


namespace bob_hair_length_l89_89899

theorem bob_hair_length (h_0 : ℝ) (r : ℝ) (t : ℝ) (months_per_year : ℝ) (h : ℝ) :
  h_0 = 6 ∧ r = 0.5 ∧ t = 5 ∧ months_per_year = 12 → h = h_0 + r * months_per_year * t :=
sorry

end bob_hair_length_l89_89899


namespace limit_S1_minus_S2_div_ln_t_l89_89205

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (Real.tan θ, 1 / Real.cos θ)

noncomputable def Q_point (t : ℝ) : ℝ × ℝ :=
  (t, Real.sqrt (t^2 + 1))

noncomputable def S1 (t : ℝ) : ℝ :=
  ∫ x in 0..t, Real.sqrt (x^2 + 1)

noncomputable def S2 (t : ℝ) : ℝ :=
  (1 / 2) * t * Real.sqrt (t^2 + 1)

theorem limit_S1_minus_S2_div_ln_t (t : ℝ) (ht : t > 0) :
  tendsto (λ t, (S1 t - S2 t) / Real.log t) at_top (𝓝 (1 / 2)) :=
  sorry

end limit_S1_minus_S2_div_ln_t_l89_89205


namespace lim_to_infinity_of_arith_geo_seq_l89_89692

noncomputable def a: ℝ
noncomputable def c: ℝ

theorem lim_to_infinity_of_arith_geo_seq (h1 : a + c = 2) (h2 : (a^2 * c^2 = 1)) 
(h3 : a ≠ c) :
  (Real.lim (λ n, (↑n : ℕ) → (a + c) / (a^2 + c^2)) = 0) := 
by 
  sorry

end lim_to_infinity_of_arith_geo_seq_l89_89692


namespace similar_right_triangles_l89_89494

open Real

theorem similar_right_triangles (x : ℝ) (h : ℝ)
  (h₁: 12^2 + 9^2 = (12^2 + 9^2))
  (similarity : (12 / x) = (9 / 6))
  (p : hypotenuse = 12*12) :
  x = 8 ∧ h = 10 := by
  sorry

end similar_right_triangles_l89_89494


namespace not_all_positive_l89_89262

theorem not_all_positive (a b c : ℝ) (h1 : a + b + c = 4) (h2 : a^2 + b^2 + c^2 = 12) (h3 : a * b * c = 1) : a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0 :=
sorry

end not_all_positive_l89_89262


namespace determinant_range_l89_89726

theorem determinant_range (x : ℝ) : 
  (2 * x - (3 - x) > 0) ↔ (x > 1) :=
by
  sorry

end determinant_range_l89_89726


namespace segment_lengths_l89_89702

noncomputable def radius : ℝ := 5
noncomputable def diameter : ℝ := 2 * radius
noncomputable def chord_length : ℝ := 8

-- The lengths of the segments AK and KB
theorem segment_lengths (x : ℝ) (y : ℝ) 
  (hx : 0 < x ∧ x < diameter) 
  (hy : 0 < y ∧ y < diameter) 
  (h1 : x + y = diameter) 
  (h2 : x * y = (diameter^2) / 4 - 16 / 4) : 
  x = 2.5 ∧ y = 7.5 := 
sorry

end segment_lengths_l89_89702


namespace curve_transformation_l89_89416

def matrix_transform (a : ℝ) (x y : ℝ) : ℝ × ℝ :=
  (0 * x + 1 * y, a * x + 0 * y)

def curve_eq (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 = 1

def transformed_curve_eq (x y : ℝ) : Prop :=
  x ^ 2 + (y ^ 2) / 4 = 1

theorem curve_transformation (a : ℝ) 
  (h₁ : matrix_transform a 2 (-2) = (-2, 4))
  (h₂ : ∀ x y, curve_eq x y → transformed_curve_eq (matrix_transform a x y).fst (matrix_transform a x y).snd) :
  a = 2 ∧ ∀ x y, curve_eq x y → transformed_curve_eq (0 * x + 1 * y) (2 * x + 0 * y) :=
by
  sorry

end curve_transformation_l89_89416


namespace exp_arbitrarily_large_l89_89261

theorem exp_arbitrarily_large (a : ℝ) (h : a > 1) : ∀ y > 0, ∃ x > 0, a^x > y := by
  sorry

end exp_arbitrarily_large_l89_89261


namespace solution_set_transformation_l89_89687

noncomputable def solution_set_of_first_inequality (a b : ℝ) : Set ℝ :=
  {x | a * x^2 - 5 * x + b > 0}

noncomputable def solution_set_of_second_inequality (a b : ℝ) : Set ℝ :=
  {x | b * x^2 - 5 * x + a > 0}

theorem solution_set_transformation (a b : ℝ)
  (h : solution_set_of_first_inequality a b = {x | -3 < x ∧ x < 2}) :
  solution_set_of_second_inequality a b = {x | x < -3 ∨ x > 2} :=
by
  sorry

end solution_set_transformation_l89_89687


namespace wall_height_l89_89946

theorem wall_height (length width depth total_bricks: ℕ) (h: ℕ) (H_length: length = 20) (H_width: width = 4) (H_depth: depth = 2) (H_total_bricks: total_bricks = 800) :
  80 * depth * h = total_bricks → h = 5 :=
by
  intros H_eq
  sorry

end wall_height_l89_89946


namespace quadratic_integer_roots_l89_89882

theorem quadratic_integer_roots (a b x : ℤ) :
  (∀ x₁ x₂ : ℤ, x₁ + x₂ = -b / a ∧ x₁ * x₂ = b / a → (x₁ = x₂ ∧ x₁ = -2 ∧ b = 4 * a) ∨ (x = -1 ∧ a = 0 ∧ b ≠ 0) ∨ (x = 0 ∧ a ≠ 0 ∧ b = 0)) :=
sorry

end quadratic_integer_roots_l89_89882


namespace parking_garage_savings_l89_89188

theorem parking_garage_savings :
  let weekly_cost := 10
  let monthly_cost := 35
  let weeks_per_year := 52
  let months_per_year := 12
  let annual_weekly_cost := weekly_cost * weeks_per_year
  let annual_monthly_cost := monthly_cost * months_per_year
  let annual_savings := annual_weekly_cost - annual_monthly_cost
  annual_savings = 100 := 
by
  sorry

end parking_garage_savings_l89_89188


namespace unique_7tuple_solution_l89_89509

theorem unique_7tuple_solution : 
  ∃! x : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ, 
  let (x1, x2, x3, x4, x5, x6, x7) := x in
  (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + (x4 - x5)^2 + (x5 - x6)^2 + (x6 - x7)^2 + x7^2 = 1 / 8 :=
by 
  sorry

end unique_7tuple_solution_l89_89509


namespace proof_statement_d_is_proposition_l89_89934

-- Define the conditions
def statement_a := "Do two points determine a line?"
def statement_b := "Take a point M on line AB"
def statement_c := "In the same plane, two lines do not intersect"
def statement_d := "The sum of two acute angles is greater than a right angle"

-- Define the property of being a proposition
def is_proposition (s : String) : Prop :=
  s ≠ "Do two points determine a line?" ∧
  s ≠ "Take a point M on line AB" ∧
  s ≠ "In the same plane, two lines do not intersect"

-- The equivalence proof that statement_d is the only proposition
theorem proof_statement_d_is_proposition :
  is_proposition statement_d ∧
  ¬is_proposition statement_a ∧
  ¬is_proposition statement_b ∧
  ¬is_proposition statement_c := by
  sorry

end proof_statement_d_is_proposition_l89_89934


namespace find_sin_2alpha_l89_89694

theorem find_sin_2alpha (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) 
    (h2 : 3 * Real.cos (2 * α) = Real.sqrt 2 * Real.sin (π / 4 - α)) : 
  Real.sin (2 * α) = -8 / 9 := 
sorry

end find_sin_2alpha_l89_89694


namespace percent_of_y_equal_to_30_percent_of_60_percent_l89_89158

variable (y : ℝ)

theorem percent_of_y_equal_to_30_percent_of_60_percent (hy : y ≠ 0) :
  ((0.18 * y) / y) * 100 = 18 := by
  have hneq : y ≠ 0 := hy
  field_simp [hneq]
  norm_num
  sorry

end percent_of_y_equal_to_30_percent_of_60_percent_l89_89158


namespace determine_c_l89_89285

theorem determine_c {f : ℝ → ℝ} (c : ℝ) (h : ∀ x, f x = 2 / (3 * x + c))
  (hf_inv : ∀ x, (f⁻¹ x) = (3 - 6 * x) / x) : c = 18 :=
by sorry

end determine_c_l89_89285


namespace geometric_sequence_proof_l89_89487

theorem geometric_sequence_proof (a : ℕ → ℝ) (q : ℝ) (h1 : q > 1) (h2 : a 1 > 0)
    (h3 : a 2 * a 4 + a 4 * a 10 - a 4 * a 6 - (a 5)^2 = 9) :
  a 3 - a 7 = -3 :=
by sorry

end geometric_sequence_proof_l89_89487


namespace no_intersection_of_ellipses_l89_89804

theorem no_intersection_of_ellipses :
  (∀ (x y : ℝ), (9*x^2 + y^2 = 9) ∧ (x^2 + 16*y^2 = 16) → false) :=
sorry

end no_intersection_of_ellipses_l89_89804


namespace mod_calculation_l89_89466

theorem mod_calculation : (9^7 + 8^8 + 7^9) % 5 = 2 := by
  sorry

end mod_calculation_l89_89466


namespace max_value_of_P_l89_89263

noncomputable def P (a b c : ℝ) : ℝ :=
  (2 / (a^2 + 1)) - (2 / (b^2 + 1)) + (3 / (c^2 + 1))

theorem max_value_of_P (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c + a + c = b) :
  ∃ x, x = 1 ∧ ∀ y, (y = P a b c) → y ≤ x :=
sorry

end max_value_of_P_l89_89263


namespace problem_inequality_l89_89576

theorem problem_inequality 
  (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) 
  (h8 : a + b + c + d ≥ 1) : 
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := 
sorry

end problem_inequality_l89_89576


namespace sum_of_smallest_natural_numbers_l89_89512

-- Define the problem statement
def satisfies_eq (A B : ℕ) := 360 / (A^3 / B) = 5

-- Prove that there exist natural numbers A and B such that 
-- satisfies_eq A B is true, and their sum is 9
theorem sum_of_smallest_natural_numbers :
  ∃ (A B : ℕ), satisfies_eq A B ∧ A + B = 9 :=
by
  -- Sorry is used here to indicate the proof is not given
  sorry

end sum_of_smallest_natural_numbers_l89_89512


namespace problem_statement_l89_89842

variable (a b c : ℝ)

theorem problem_statement 
  (h1 : ab / (a + b) = 1 / 3)
  (h2 : bc / (b + c) = 1 / 4)
  (h3 : ca / (c + a) = 1 / 5) :
  abc / (ab + bc + ca) = 1 / 6 := 
sorry

end problem_statement_l89_89842


namespace find_m_l89_89391

variable {S : ℕ → ℤ}
variable {m : ℕ}

/-- Given the sequences conditions, the value of m is 5 --/
theorem find_m (h1 : S (m - 1) = -2) (h2 : S m = 0) (h3 : S (m + 1) = 3) (h4 : 2 ≤ m) : m = 5 :=
sorry

end find_m_l89_89391


namespace sphere_radius_is_16_25_l89_89190

def sphere_in_cylinder_radius (r : ℝ) : Prop := 
  ∃ (x : ℝ), (x ^ 2 + 15 ^ 2 = r ^ 2) ∧ ((x + 10) ^ 2 = r ^ 2) ∧ (r = 16.25)

theorem sphere_radius_is_16_25 : 
  sphere_in_cylinder_radius 16.25 :=
sorry

end sphere_radius_is_16_25_l89_89190


namespace possible_values_for_t_l89_89683

noncomputable def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def b (a1 d : ℝ) (n : ℕ) : ℝ :=
  Real.sin (arithmetic_sequence a1 d n)

theorem possible_values_for_t :
  ∃ t : ℕ, t ≤ 8 ∧ (∀ n : ℕ, b 0 (real.pi/2) n = b 0 (real.pi/2) (n + t)) ∧ (∃ S : set ℝ, S = {b 0 (real.pi/2) n | n : ℕ} ∧ S.card = 4) →
  t = 4 :=
by
  sorry

end possible_values_for_t_l89_89683


namespace total_students_in_high_school_l89_89925

-- Definitions based on the problem conditions
def freshman_students : ℕ := 400
def sample_students : ℕ := 45
def sophomore_sample_students : ℕ := 15
def senior_sample_students : ℕ := 10

-- The theorem to be proved
theorem total_students_in_high_school : (sample_students = 45) → (freshman_students = 400) → (sophomore_sample_students = 15) → (senior_sample_students = 10) → ∃ total_students : ℕ, total_students = 900 :=
by
  sorry

end total_students_in_high_school_l89_89925


namespace bucket_water_total_l89_89179

theorem bucket_water_total (initial_gallons : ℝ) (added_gallons : ℝ) (total_gallons : ℝ) : 
  initial_gallons = 3 ∧ added_gallons = 6.8 → total_gallons = 9.8 :=
by
  { sorry }

end bucket_water_total_l89_89179


namespace sum_first_four_terms_eq_12_l89_89682

noncomputable def a : ℕ → ℤ := sorry -- An arithmetic sequence aₙ

-- Given conditions
axiom h1 : a 2 = 4
axiom h2 : a 1 + a 5 = 4 * a 3 - 4

theorem sum_first_four_terms_eq_12 : (a 1 + a 2 + a 3 + a 4) = 12 := 
by {
  sorry
}

end sum_first_four_terms_eq_12_l89_89682


namespace sum_partition_ominous_years_l89_89499

def is_ominous (n : ℕ) : Prop :=
  n = 1 ∨ Nat.Prime n

theorem sum_partition_ominous_years :
  ∀ n : ℕ, (¬ ∃ (A B : Finset ℕ), A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅ ∧ 
    (A.sum id = B.sum id ∧ A.card = B.card)) ↔ is_ominous n := 
sorry

end sum_partition_ominous_years_l89_89499


namespace coplanar_condition_l89_89104

-- Definitions representing points A, B, C, D and the origin O in a vector space over the reals
variables {V : Type*} [AddCommGroup V] [Module ℝ V] (O A B C D : V)

-- The main statement of the problem
theorem coplanar_condition (h : (2 : ℝ) • (A - O) - (3 : ℝ) • (B - O) + (7 : ℝ) • (C - O) + k • (D - O) = 0) :
  k = -6 :=
sorry

end coplanar_condition_l89_89104


namespace count_permutations_perception_l89_89360

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def num_permutations (word : String) : ℕ :=
  let total_letters := word.length
  let freq_map := word.to_list.groupBy id
  let fact_chars := freq_map.toList.map (λ (c, l) => factorial l.length)
  factorial total_letters / fact_chars.foldl (*) 1

theorem count_permutations_perception :
  num_permutations "PERCEPTION" = 907200 := by
  sorry

end count_permutations_perception_l89_89360


namespace arrange_PERCEPTION_l89_89663

theorem arrange_PERCEPTION : 
  let n := 10 
  let k_E := 2
  let k_P := 2
  let k_I := 2
  nat.factorial n / (nat.factorial k_E * nat.factorial k_P * nat.factorial k_I) = 453600 :=
by
  sorry

end arrange_PERCEPTION_l89_89663


namespace ratio_of_erasers_l89_89195

theorem ratio_of_erasers (a n : ℕ) (ha : a = 4) (hn : n = a + 12) :
  n / a = 4 :=
by
  sorry

end ratio_of_erasers_l89_89195


namespace problem_tiles_count_l89_89308

theorem problem_tiles_count (T B : ℕ) (h: 2 * T + 3 * B = 301) (hB: B = 3) : T = 146 := 
by
  sorry

end problem_tiles_count_l89_89308


namespace car_speed_l89_89180

variable (D : ℝ) (V : ℝ)

theorem car_speed
  (h1 : 1 / ((D / 3) / 80) + (D / 3) / 15 + (D / 3) / V = D / 30) :
  V = 35.625 :=
by 
  sorry

end car_speed_l89_89180


namespace number_of_men_in_first_group_l89_89926

-- Definitions based on the conditions provided
def work_done (men : ℕ) (days : ℕ) (work_rate : ℝ) : ℝ :=
  men * days * work_rate

-- Given conditions
def condition1 (M : ℕ) : Prop :=
  ∃ work_rate : ℝ, work_done M 12 work_rate = 66

def condition2 : Prop :=
  ∃ work_rate : ℝ, work_done 86 8 work_rate = 189.2

-- Proof goal
theorem number_of_men_in_first_group : 
  ∀ M : ℕ, condition1 M → condition2 → M = 57 := by
  sorry

end number_of_men_in_first_group_l89_89926


namespace figure_50_unit_squares_l89_89870

-- Definitions reflecting the conditions from step A
def f (n : ℕ) := (1/2 : ℚ) * n^3 + (7/2 : ℚ) * n + 1

theorem figure_50_unit_squares : f 50 = 62676 := by
  sorry

end figure_50_unit_squares_l89_89870


namespace arithmetic_sequence_general_formula_is_not_term_l89_89255

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 2) (h17 : a 17 = 66) :
  ∀ n : ℕ, a n = 4 * n - 2 := sorry

theorem is_not_term (a : ℕ → ℤ) 
  (ha : ∀ n : ℕ, a n = 4 * n - 2) :
  ∀ k : ℤ, k = 88 → ¬ ∃ n : ℕ, a n = k := sorry

end arithmetic_sequence_general_formula_is_not_term_l89_89255


namespace worth_of_used_car_l89_89038

theorem worth_of_used_car (earnings remaining : ℝ) (earnings_eq : earnings = 5000) (remaining_eq : remaining = 1000) : 
  ∃ worth : ℝ, worth = earnings - remaining ∧ worth = 4000 :=
by
  sorry

end worth_of_used_car_l89_89038


namespace find_last_num_divisible_by_12_stopping_at_84_l89_89644

theorem find_last_num_divisible_by_12_stopping_at_84 :
  ∃ N, (N = 84) ∧ (71 ≤ N) ∧ (let concatenated := string.join ((list.range (N - 70)).map (λ i, (string.of_nat (i + 71)))) in 
    (nat.divisible (int.of_nat (string.to_nat concatenated)) 12)) :=
begin
  sorry
end

end find_last_num_divisible_by_12_stopping_at_84_l89_89644


namespace largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative_l89_89226

theorem largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative :
  ∃ (n : ℤ), (4 < n) ∧ (n < 7) ∧ (n = 6) :=
by
  sorry

end largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative_l89_89226


namespace wholesale_cost_is_200_l89_89788

variable (W R E : ℝ)

def retail_price (W : ℝ) : ℝ := 1.20 * W

def employee_price (R : ℝ) : ℝ := 0.75 * R

-- Main theorem stating that given the retail and employee price formulas and the employee paid amount,
-- the wholesale cost W is equal to 200.
theorem wholesale_cost_is_200
  (hR : R = retail_price W)
  (hE : E = employee_price R)
  (heq : E = 180) :
  W = 200 :=
by
  sorry

end wholesale_cost_is_200_l89_89788


namespace find_q_l89_89137

def Q (x p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q (p q d : ℝ) (h: d = 3) (h1: -p / 3 = -d) (h2: -p / 3 = 1 + p + q + d) : q = -16 :=
by
  sorry

end find_q_l89_89137


namespace man_speed_in_still_water_l89_89473

theorem man_speed_in_still_water :
  ∃ (V_m V_s : ℝ), 
  V_m + V_s = 14 ∧ 
  V_m - V_s = 6 ∧ 
  V_m = 10 :=
by
  sorry

end man_speed_in_still_water_l89_89473


namespace factorial_trailing_zeros_500_l89_89209

theorem factorial_trailing_zeros_500 :
  let count_factors_of_five (n : ℕ) : ℕ := n / 5 + n / 25 + n / 125
  count_factors_of_five 500 = 124 :=
by
  sorry  -- The proof is not required as per the instructions.

end factorial_trailing_zeros_500_l89_89209


namespace equal_even_odd_probability_l89_89950

theorem equal_even_odd_probability : 
  let total_dice := 8
  let even_odd_combinations := Nat.choose total_dice 4
  let single_arrangement_probability := (1 / 2) ^ total_dice
  even_odd_combinations * single_arrangement_probability = 35 / 128 := by
  sorry

end equal_even_odd_probability_l89_89950


namespace number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72_l89_89382

def five_digit_number_count : Nat :=
  -- Number of ways to select and arrange odd digits in two groups
  let group_odd_digits := (Nat.choose 3 2) * (Nat.factorial 2)
  -- Number of ways to arrange the even digits
  let arrange_even_digits := Nat.factorial 2
  -- Number of ways to insert two groups of odd digits into the gaps among even digits
  let insert_odd_groups := (Nat.factorial 3)
  -- Total ways
  group_odd_digits * arrange_even_digits * arrange_even_digits * insert_odd_groups

theorem number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72 :
  five_digit_number_count = 72 :=
by
  -- Placeholder for proof
  sorry

end number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72_l89_89382


namespace therapy_hours_l89_89927

theorem therapy_hours (x n : ℕ) : 
  (x + 30) + 2 * x = 252 → 
  104 + (n - 1) * x = 400 → 
  x = 74 → 
  n = 5 := 
by
  sorry

end therapy_hours_l89_89927


namespace length_of_second_platform_l89_89639

-- Definitions
def length_train : ℝ := 230
def time_first_platform : ℝ := 15
def length_first_platform : ℝ := 130
def total_distance_first_platform : ℝ := length_train + length_first_platform
def time_second_platform : ℝ := 20

-- Statement to prove
theorem length_of_second_platform : 
  ∃ L : ℝ, (total_distance_first_platform / time_first_platform) = ((length_train + L) / time_second_platform) ∧ L = 250 :=
by
  sorry

end length_of_second_platform_l89_89639


namespace analytical_expression_of_C3_l89_89449

def C1 (x : ℝ) : ℝ := x^2 - 2*x + 3
def C2 (x : ℝ) : ℝ := C1 (x + 1)
def C3 (x : ℝ) : ℝ := C2 (-x)

theorem analytical_expression_of_C3 :
  ∀ x, C3 x = x^2 + 2 := by
  sorry

end analytical_expression_of_C3_l89_89449


namespace problem_1_problem_2_problem_3_l89_89278

noncomputable def area_triangle (a b C : ℝ) : ℝ := (1/2) * a * b * Real.sin C
noncomputable def area_quadrilateral (e f φ : ℝ) : ℝ := (1/2) * e * f * Real.sin φ

theorem problem_1 (a b C : ℝ) (hC : Real.sin C ≤ 1) : 
  area_triangle a b C ≤ (a^2 + b^2) / 4 :=
sorry

theorem problem_2 (e f φ : ℝ) (hφ : Real.sin φ ≤ 1) : 
  area_quadrilateral e f φ ≤ (e^2 + f^2) / 4 :=
sorry

theorem problem_3 (a b C c d D : ℝ) 
  (hC : Real.sin C ≤ 1) 
  (hD : Real.sin D ≤ 1) :
  area_triangle a b C + area_triangle c d D ≤ (a^2 + b^2 + c^2 + d^2) / 4 :=
sorry

end problem_1_problem_2_problem_3_l89_89278


namespace mean_of_remaining_students_l89_89412

theorem mean_of_remaining_students
  (n : ℕ) (h : n > 20)
  (mean_score_first_15 : ℝ)
  (mean_score_next_5 : ℝ)
  (overall_mean_score : ℝ) :
  mean_score_first_15 = 10 →
  mean_score_next_5 = 16 →
  overall_mean_score = 11 →
  ∀ a, a = (11 * n - 230) / (n - 20) := by
sorry

end mean_of_remaining_students_l89_89412


namespace rectangle_length_l89_89474

theorem rectangle_length
  (side_length_square : ℝ)
  (width_rectangle : ℝ)
  (area_equiv : side_length_square ^ 2 = width_rectangle * l)
  : l = 24 := by
  sorry

end rectangle_length_l89_89474


namespace quadrilateral_area_correct_l89_89333

open Real
open Function
open Classical

noncomputable def quadrilateral_area : ℝ :=
  let A := (0, 0)
  let B := (2, 3)
  let C := (5, 0)
  let D := (3, -2)
  let vector_cross_product (u v : ℝ × ℝ) : ℝ := u.1 * v.2 - u.2 * v.1
  let area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := 0.5 * abs (vector_cross_product (p2 - p1) (p3 - p1))
  area_triangle A B D + area_triangle B C D

theorem quadrilateral_area_correct : quadrilateral_area = 17 / 2 :=
  sorry

end quadrilateral_area_correct_l89_89333


namespace solve_fraction_eq_l89_89582

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (2 / (x - 2) = 3 / (x + 2)) → x = 10 :=
by
  sorry

end solve_fraction_eq_l89_89582


namespace geometric_sequence_problem_l89_89700

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

def condition (a : ℕ → ℝ) : Prop :=
a 4 + a 8 = -3

theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : condition a) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 :=
sorry

end geometric_sequence_problem_l89_89700


namespace martha_apples_l89_89267

theorem martha_apples (initial_apples jane_apples extra_apples final_apples : ℕ)
  (h1 : initial_apples = 20)
  (h2 : jane_apples = 5)
  (h3 : extra_apples = 2)
  (h4 : final_apples = 4) :
  initial_apples - jane_apples - (jane_apples + extra_apples) - final_apples = final_apples := 
by
  sorry

end martha_apples_l89_89267


namespace solution_set_of_inequality_system_l89_89300

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by
  sorry

end solution_set_of_inequality_system_l89_89300


namespace area_enclosed_by_curve_l89_89007

open Function Real

def enclosed_area : ℝ :=
  let A : real := 2 * π + 5 in A

theorem area_enclosed_by_curve :
  ∀ (x y : ℝ), x^2 + y^2 = 2 * (|x| + |y|) → enclosed_area = 2 * π + 5 :=
by
  sorry

end area_enclosed_by_curve_l89_89007


namespace value_ab_plus_a_plus_b_l89_89867

noncomputable def polynomial : Polynomial ℝ := Polynomial.C (-1) + Polynomial.X * Polynomial.C (-1) + Polynomial.X^2 * Polynomial.C (-4) + Polynomial.X^4

theorem value_ab_plus_a_plus_b {a b : ℝ} (h : polynomial.eval a = 0 ∧ polynomial.eval b = 0) : a * b + a + b = -1 / 2 :=
sorry

end value_ab_plus_a_plus_b_l89_89867


namespace pizza_area_difference_l89_89014

def hueys_hip_pizza (small_size : ℕ) (small_cost : ℕ) (large_size : ℕ) (large_cost : ℕ) : ℕ :=
  let small_area := small_size * small_size
  let large_area := large_size * large_size
  let individual_money := 30
  let pooled_money := 2 * individual_money

  let individual_small_total_area := (individual_money / small_cost) * small_area * 2
  let pooled_large_total_area := (pooled_money / large_cost) * large_area

  pooled_large_total_area - individual_small_total_area

theorem pizza_area_difference :
  hueys_hip_pizza 6 10 9 20 = 27 :=
by
  sorry

end pizza_area_difference_l89_89014


namespace cos_value_given_sin_l89_89515

theorem cos_value_given_sin (α : ℝ) (h : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5) :
  Real.cos (π / 3 - α) = 2 * Real.sqrt 5 / 5 := by
  sorry

end cos_value_given_sin_l89_89515


namespace work_together_10_days_l89_89013

noncomputable def rate_A (W : ℝ) : ℝ := W / 20
noncomputable def rate_B (W : ℝ) : ℝ := W / 20

theorem work_together_10_days (W : ℝ) (hW : W > 0) :
  let A := rate_A W
  let B := rate_B W
  let combined_rate := A + B
  W / combined_rate = 10 :=
by
  sorry

end work_together_10_days_l89_89013


namespace two_is_four_percent_of_fifty_l89_89482

theorem two_is_four_percent_of_fifty : (2 / 50) * 100 = 4 := 
by
  sorry

end two_is_four_percent_of_fifty_l89_89482


namespace positive_difference_of_two_numbers_l89_89595

theorem positive_difference_of_two_numbers :
  ∀ (x y : ℝ), x + y = 8 → x^2 - y^2 = 24 → abs (x - y) = 3 :=
by
  intros x y h1 h2
  sorry

end positive_difference_of_two_numbers_l89_89595


namespace simplify_fraction_l89_89659

theorem simplify_fraction :
  (3 - 6 + 12 - 24 + 48 - 96) / (6 - 12 + 24 - 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end simplify_fraction_l89_89659


namespace Monet_paintings_consecutively_l89_89113

noncomputable def probability_Monet_paintings_consecutively (total_art_pieces Monet_paintings : ℕ) : ℚ :=
  let numerator := 9 * Nat.factorial (total_art_pieces - Monet_paintings) * Nat.factorial Monet_paintings
  let denominator := Nat.factorial total_art_pieces
  numerator / denominator

theorem Monet_paintings_consecutively :
  probability_Monet_paintings_consecutively 12 4 = 18 / 95 := by
  sorry

end Monet_paintings_consecutively_l89_89113


namespace line_through_points_decreasing_direct_proportion_function_m_l89_89557

theorem line_through_points_decreasing (x₁ x₂ y₁ y₂ k b : ℝ) (h1 : x₁ < x₂) (h2 : y₁ = k * x₁ + b) (h3 : y₂ = k * x₂ + b) (h4 : k < 0) : y₁ > y₂ :=
sorry

theorem direct_proportion_function_m (x₁ x₂ y₁ y₂ m : ℝ) (h1 : x₁ < x₂) (h2 : y₁ = (1 - 2 * m) * x₁) (h3 : y₂ = (1 - 2 * m) * x₂) (h4 : y₁ > y₂) : m > 1/2 :=
sorry

end line_through_points_decreasing_direct_proportion_function_m_l89_89557


namespace greatest_possible_difference_in_rectangles_area_l89_89143

theorem greatest_possible_difference_in_rectangles_area :
  ∃ (l1 w1 l2 w2 l3 w3 : ℤ),
    2 * l1 + 2 * w1 = 148 ∧
    2 * l2 + 2 * w2 = 150 ∧
    2 * l3 + 2 * w3 = 152 ∧
    (∃ (A1 A2 A3 : ℤ),
      A1 = l1 * w1 ∧
      A2 = l2 * w2 ∧
      A3 = l3 * w3 ∧
      (max (abs (A1 - A2)) (max (abs (A1 - A3)) (abs (A2 - A3))) = 1372)) :=
by
  sorry

end greatest_possible_difference_in_rectangles_area_l89_89143


namespace number_of_breaks_l89_89101

theorem number_of_breaks (pushups_per_10_sec : ℕ) (pushups_per_min_with_breaks : ℕ) 
  (time_per_break : ℕ) (total_time_sec : ℕ) : 
  pushups_per_10_sec = 5 → pushups_per_min_with_breaks = 22 → time_per_break = 8 → total_time_sec = 60 →
  (total_time_sec / (pushups_per_10_sec * (total_time_sec / 10))) - pushups_per_min_with_breaks = 
  (8 / 5) * 10 / time_per_break :=
begin
  intros h1 h2 h3 h4,
  have max_pushups_per_min : ℕ := 5 * (total_time_sec / 10),
  have missed_pushups : ℕ := max_pushups_per_min - pushups_per_min_with_breaks,
  have total_break_time : ℕ := (missed_pushups * 10) / 5,
  have total_breaks := total_break_time / time_per_break,
  exact total_breaks
end

end number_of_breaks_l89_89101


namespace tunnel_length_proof_l89_89175

variable (train_length : ℝ) (train_speed : ℝ) (time_in_tunnel : ℝ)

noncomputable def tunnel_length (train_length train_speed time_in_tunnel : ℝ) : ℝ :=
  (train_speed / 60) * time_in_tunnel - train_length

theorem tunnel_length_proof 
  (h_train_length : train_length = 2) 
  (h_train_speed : train_speed = 30) 
  (h_time_in_tunnel : time_in_tunnel = 4) : 
  tunnel_length 2 30 4 = 2 := by
    simp [tunnel_length, h_train_length, h_train_speed, h_time_in_tunnel]
    norm_num
    sorry

end tunnel_length_proof_l89_89175


namespace sufficient_condition_of_implications_l89_89448

variables (P1 P2 θ : Prop)

theorem sufficient_condition_of_implications
  (h1 : P1 → θ)
  (h2 : P2 → P1) :
  P2 → θ :=
by sorry

end sufficient_condition_of_implications_l89_89448


namespace perception_permutations_count_l89_89365

theorem perception_permutations_count :
  let n := 10
  let freq_P := 2
  let freq_E := 2
  let factorial := λ x : ℕ, (Nat.factorial x)
  factorial n / (factorial freq_P * factorial freq_E) = 907200 :=
by sorry

end perception_permutations_count_l89_89365


namespace aiyanna_cookies_l89_89194

theorem aiyanna_cookies (a b : ℕ) (h₁ : a = 129) (h₂ : b = a + 11) : b = 140 := by
  sorry

end aiyanna_cookies_l89_89194


namespace percent_singles_l89_89807

theorem percent_singles (total_hits home_runs triples doubles : ℕ) 
  (h_total: total_hits = 50) 
  (h_hr: home_runs = 3) 
  (h_tr: triples = 2) 
  (h_double: doubles = 8) : 
  100 * (total_hits - (home_runs + triples + doubles)) / total_hits = 74 := 
by
  -- proofs
  sorry

end percent_singles_l89_89807


namespace prob_equal_even_odd_dice_l89_89956

def even_number_probability : ℚ := 1 / 2
def odd_number_probability : ℚ := 1 / 2

def probability_equal_even_odd (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k) * (even_number_probability) ^ n

theorem prob_equal_even_odd_dice : 
  probability_equal_even_odd 8 4 = 35 / 128 :=
by
  sorry

end prob_equal_even_odd_dice_l89_89956


namespace fraction_zero_l89_89468

theorem fraction_zero (x : ℝ) (h : x ≠ 3) : (2 * x^2 - 6 * x) / (x - 3) = 0 ↔ x = 0 := 
by
  sorry

end fraction_zero_l89_89468


namespace find_triples_l89_89056

theorem find_triples (x y z : ℝ) 
  (h1 : (1/3 : ℝ) * min x y + (2/3 : ℝ) * max x y = 2017)
  (h2 : (1/3 : ℝ) * min y z + (2/3 : ℝ) * max y z = 2018)
  (h3 : (1/3 : ℝ) * min z x + (2/3 : ℝ) * max z x = 2019) :
  (x = 2019) ∧ (y = 2016) ∧ (z = 2019) :=
sorry

end find_triples_l89_89056


namespace largest_number_of_square_plots_l89_89331

/-- A rectangular field measures 30 meters by 60 meters with 2268 meters of internal fencing to partition into congruent, square plots. The entire field must be partitioned with sides of squares parallel to the edges. Prove the largest number of square plots is 722. -/
theorem largest_number_of_square_plots (s n : ℕ) (h_length : 60 = n * s) (h_width : 30 = s * 2 * n) (h_fence : 120 * n - 90 ≤ 2268) :
(s * 2 * n) = 722 :=
sorry

end largest_number_of_square_plots_l89_89331


namespace tom_teaching_years_l89_89602

theorem tom_teaching_years (T D : ℝ) (h1 : T + D = 70) (h2 : D = (1/2) * T - 5) : T = 50 :=
by
  -- This is where the proof would normally go if it were required.
  sorry

end tom_teaching_years_l89_89602


namespace prime_in_choices_l89_89615

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def twenty := 20
def twenty_one := 21
def twenty_three := 23
def twenty_five := 25
def twenty_seven := 27

theorem prime_in_choices :
  is_prime twenty_three ∧ ¬ is_prime twenty ∧ ¬ is_prime twenty_one ∧ ¬ is_prime twenty_five ∧ ¬ is_prime twenty_seven :=
by
  sorry

end prime_in_choices_l89_89615


namespace divisibility_by_5_l89_89055

theorem divisibility_by_5 (x y : ℤ) : (x^2 - 2 * x * y + 2 * y^2) % 5 = 0 ∨ (x^2 + 2 * x * y + 2 * y^2) % 5 = 0 ↔ (x % 5 = 0 ∧ y % 5 = 0) ∨ (x % 5 ≠ 0 ∧ y % 5 ≠ 0) := 
by
  sorry

end divisibility_by_5_l89_89055


namespace tom_teaching_years_l89_89599

def years_tom_has_been_teaching (x : ℝ) : Prop :=
  x + (1/2 * x - 5) = 70

theorem tom_teaching_years:
  ∃ x : ℝ, years_tom_has_been_teaching x ∧ x = 50 :=
by
  sorry

end tom_teaching_years_l89_89599


namespace problem1_problem2_l89_89066

-- Define Sn as given
def S (n : ℕ) : ℕ := (n ^ 2 + n) / 2

-- Define a sequence a_n
def a (n : ℕ) : ℕ := if n = 1 then 1 else S n - S (n - 1)

-- Define b_n using a_n = log_2 b_n
def b (n : ℕ) : ℕ := 2 ^ n

-- Define the sum of first n terms of sequence b_n
def T (n : ℕ) : ℕ := (2 ^ (n + 1)) - 2

-- Our main theorem statements
theorem problem1 (n : ℕ) : a n = n := by
  sorry

theorem problem2 (n : ℕ) : (Finset.range n).sum b = T n := by
  sorry

end problem1_problem2_l89_89066


namespace range_of_a_l89_89071

theorem range_of_a (a b c : ℝ) (h₁ : a + b + c = 2) (h₂ : a^2 + b^2 + c^2 = 4) (h₃ : a > b ∧ b > c) :
  (2 / 3 < a ∧ a < 2) :=
sorry

end range_of_a_l89_89071


namespace transformed_curve_l89_89608

theorem transformed_curve (x y : ℝ) :
  (y * Real.cos x + 2 * y - 1 = 0) →
  (y - 1) * Real.sin x + 2 * y - 3 = 0 :=
by
  intro h
  sorry

end transformed_curve_l89_89608


namespace menu_choices_l89_89822

theorem menu_choices :
  let lunchChinese := 5 
  let lunchJapanese := 4 
  let dinnerChinese := 3 
  let dinnerJapanese := 5 
  let lunchOptions := lunchChinese + lunchJapanese
  let dinnerOptions := dinnerChinese + dinnerJapanese
  lunchOptions * dinnerOptions = 72 :=
by
  let lunchChinese := 5
  let lunchJapanese := 4
  let dinnerChinese := 3
  let dinnerJapanese := 5
  let lunchOptions := lunchChinese + lunchJapanese
  let dinnerOptions := dinnerChinese + dinnerJapanese
  have h : lunchOptions * dinnerOptions = 72 :=
    by 
      sorry
  exact h

end menu_choices_l89_89822


namespace tom_teaching_years_l89_89604

theorem tom_teaching_years (T D : ℝ) (h1 : T + D = 70) (h2 : D = (1/2) * T - 5) : T = 50 :=
by
  -- This is where the proof would normally go if it were required.
  sorry

end tom_teaching_years_l89_89604


namespace tan_alpha_eq_2_implies_sin_2alpha_inverse_l89_89974

theorem tan_alpha_eq_2_implies_sin_2alpha_inverse (α : ℝ) (h : Real.tan α = 2) :
  1 / Real.sin (2 * α) = 5 / 4 :=
sorry

end tan_alpha_eq_2_implies_sin_2alpha_inverse_l89_89974


namespace repeating_decimal_base4_sum_l89_89126

theorem repeating_decimal_base4_sum (a b : ℕ) (hrelprime : Int.gcd a b = 1)
  (h4_rep : ((12 : ℚ) / (44 : ℚ)) = (a : ℚ) / (b : ℚ)) : a + b = 7 :=
sorry

end repeating_decimal_base4_sum_l89_89126


namespace kaleb_candy_problem_l89_89420

-- Define the initial problem with given conditions

theorem kaleb_candy_problem :
  ∀ (total_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ),
    total_boxes = 14 →
    given_away_boxes = 5 →
    pieces_per_box = 6 →
    (total_boxes - given_away_boxes) * pieces_per_box = 54 :=
by
  intros total_boxes given_away_boxes pieces_per_box
  intros h1 h2 h3
  -- Use assumptions
  sorry

end kaleb_candy_problem_l89_89420


namespace woodworker_tables_count_l89_89931

/-- A woodworker made a total of 40 furniture legs and has built 6 chairs.
    Each chair requires 4 legs. Prove that the number of tables made is 4,
    assuming each table also requires 4 legs. -/
theorem woodworker_tables_count (total_legs chairs tables : ℕ)
  (legs_per_chair legs_per_table : ℕ)
  (H1 : total_legs = 40)
  (H2 : chairs = 6)
  (H3 : legs_per_chair = 4)
  (H4 : legs_per_table = 4)
  (H5 : total_legs = chairs * legs_per_chair + tables * legs_per_table) :
  tables = 4 := 
  sorry

end woodworker_tables_count_l89_89931


namespace modulo_17_residue_l89_89464

theorem modulo_17_residue : (3^4 + 6 * 49 + 8 * 137 + 7 * 34) % 17 = 5 := 
by
  sorry

end modulo_17_residue_l89_89464


namespace number_of_integers_between_sqrt10_and_sqrt100_l89_89077

theorem number_of_integers_between_sqrt10_and_sqrt100 :
  (∃ n : ℕ, n = 7) :=
sorry

end number_of_integers_between_sqrt10_and_sqrt100_l89_89077


namespace dogs_in_pet_shop_l89_89918

variable (D C B : ℕ) (x : ℕ)

theorem dogs_in_pet_shop
  (h1 : D = 3 * x)
  (h2 : C = 7 * x)
  (h3 : B = 12 * x)
  (h4 : D + B = 375) :
  D = 75 :=
by
  sorry

end dogs_in_pet_shop_l89_89918


namespace Robert_GRE_exam_l89_89437

/-- Robert started preparation for GRE entrance examination in the month of January and prepared for 5 months. Prove that he could write the examination any date after the end of May.-/
theorem Robert_GRE_exam (start_month : ℕ) (prep_duration : ℕ) : 
  start_month = 1 → prep_duration = 5 → ∃ exam_date, exam_date > 5 :=
by
  sorry

end Robert_GRE_exam_l89_89437


namespace find_x_minus_4y_l89_89399

theorem find_x_minus_4y (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - 3 * y = 10) : x - 4 * y = 5 :=
by 
  sorry

end find_x_minus_4y_l89_89399


namespace tangent_line_equation_l89_89685

theorem tangent_line_equation :
  (∀ x, f (-x) = -f x) → (∀ x, x < 0 → f x = log (-x) + 2 * x) →
  (∃ a b c : ℝ, (∀ x y : ℝ, y = f x → a * x + b * y + c = 0) ∧
    (∀ x y : ℕ, (x, y) = (1, f 1) → a * x + b * y + c = 0)) :=
  sorry

end tangent_line_equation_l89_89685


namespace Buffy_whiskers_is_40_l89_89969

def number_of_whiskers (Puffy Scruffy Buffy Juniper : ℕ) : Prop :=
  Puffy = 3 * Juniper ∧
  Puffy = Scruffy / 2 ∧
  Buffy = (Puffy + Scruffy + Juniper) / 3 ∧
  Juniper = 12

theorem Buffy_whiskers_is_40 :
  ∃ (Puffy Scruffy Buffy Juniper : ℕ), 
    number_of_whiskers Puffy Scruffy Buffy Juniper ∧ Buffy = 40 := 
by
  sorry

end Buffy_whiskers_is_40_l89_89969


namespace b_negative_l89_89237

variable {R : Type*} [LinearOrderedField R]

theorem b_negative (a b : R) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∀ x : R, 0 ≤ x → (x - a) * (x - b) * (x - (2*a + b)) ≥ 0) : b < 0 := 
sorry

end b_negative_l89_89237


namespace find_m_l89_89132

variable (m : ℝ)

theorem find_m (h1 : 3 * (-7.5) - y = m) (h2 : -0.4 * (-7.5) + y = 3) : m = -22.5 :=
by
  sorry

end find_m_l89_89132


namespace mabel_visits_helen_l89_89718

-- Define the number of steps Mabel lives from Lake High school
def MabelSteps : ℕ := 4500

-- Define the number of steps Helen lives from the school
def HelenSteps : ℕ := (3 * MabelSteps) / 4

-- Define the total number of steps Mabel will walk to visit Helen
def TotalSteps : ℕ := MabelSteps + HelenSteps

-- Prove that the total number of steps Mabel walks to visit Helen is 7875
theorem mabel_visits_helen :
  TotalSteps = 7875 :=
sorry

end mabel_visits_helen_l89_89718


namespace division_simplification_l89_89657

theorem division_simplification : 180 / (12 + 13 * 3) = 60 / 17 := by
  sorry

end division_simplification_l89_89657


namespace bond_paper_cost_l89_89793

/-!
# Bond Paper Cost Calculation

This theorem calculates the total cost to buy the required amount of each type of bond paper, given the specified conditions.
-/

def cost_of_ream (sheets_per_ream : ℤ) (cost_per_ream : ℤ) (required_sheets : ℤ) : ℤ :=
  let reams_needed := (required_sheets + sheets_per_ream - 1) / sheets_per_ream
  reams_needed * cost_per_ream

theorem bond_paper_cost :
  let total_sheets := 5000
  let required_A := 2500
  let required_B := 1500
  let remaining_sheets := total_sheets - required_A - required_B
  let cost_A := cost_of_ream 500 27 required_A
  let cost_B := cost_of_ream 400 24 required_B
  let cost_C := cost_of_ream 300 18 remaining_sheets
  cost_A + cost_B + cost_C = 303 := 
by
  sorry

end bond_paper_cost_l89_89793


namespace experiment_variance_l89_89004

noncomputable def probability_of_success : ℚ := 5/9

noncomputable def variance_of_binomial (n : ℕ) (p : ℚ) : ℚ :=
  n * p * (1 - p)

def number_of_experiments : ℕ := 30

theorem experiment_variance :
  variance_of_binomial number_of_experiments probability_of_success = 200/27 :=
by
  sorry

end experiment_variance_l89_89004


namespace estimate_production_in_March_l89_89330

theorem estimate_production_in_March 
  (monthly_production : ℕ → ℝ)
  (x y : ℝ)
  (hx : x = 3)
  (hy : y = x + 1) : y = 4 :=
by
  sorry

end estimate_production_in_March_l89_89330


namespace common_ratio_of_geometric_sequence_l89_89256

theorem common_ratio_of_geometric_sequence (a_1 a_2 a_3 a_4 q : ℝ)
  (h1 : a_1 * a_2 * a_3 = 27)
  (h2 : a_2 + a_4 = 30)
  (geometric_sequence : a_2 = a_1 * q ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3) :
  q = 3 ∨ q = -3 :=
sorry

end common_ratio_of_geometric_sequence_l89_89256


namespace violates_properties_l89_89164

-- Definitions from conditions
variables {a b c m : ℝ}

-- Conclusion to prove
theorem violates_properties :
  (∀ c : ℝ, ac = bc → (c ≠ 0 → a = b)) ∧ (c = 0 → ac = bc) → False :=
sorry

end violates_properties_l89_89164


namespace B_catches_up_with_A_l89_89168

-- Define the conditions
def speed_A : ℝ := 10 -- A's speed in kmph
def speed_B : ℝ := 20 -- B's speed in kmph
def delay : ℝ := 6 -- Delay in hours after A's start

-- Define the total distance where B catches up with A
def distance_catch_up : ℝ := 120

-- Statement to prove B catches up with A at 120 km from the start
theorem B_catches_up_with_A :
  (speed_A * delay + speed_A * (distance_catch_up / speed_B - delay)) = distance_catch_up :=
by
  sorry

end B_catches_up_with_A_l89_89168


namespace problem_part_one_problem_part_two_l89_89797

theorem problem_part_one : 23 - 17 - (-6) + (-16) = -4 :=
by
  sorry

theorem problem_part_two : 0 - 32 / ((-2)^3 - (-4)) = 8 :=
by
  sorry

end problem_part_one_problem_part_two_l89_89797


namespace find_t_squared_l89_89631
noncomputable section

-- Definitions of the given conditions
def hyperbola_opens_vertically (x y : ℝ) : Prop :=
  (y^2 / 4 - 5 * x^2 / 16 = 1)

-- Statement of the problem
theorem find_t_squared (t : ℝ) 
  (h1 : hyperbola_opens_vertically 4 (-3))
  (h2 : hyperbola_opens_vertically 0 (-2))
  (h3 : hyperbola_opens_vertically 2 t) : 
  t^2 = 8 := 
sorry -- Proof is omitted, it's just the statement

end find_t_squared_l89_89631


namespace inv_100_mod_101_l89_89050

theorem inv_100_mod_101 : (100 : ℤ) * (100 : ℤ) % 101 = 1 := by
  sorry

end inv_100_mod_101_l89_89050


namespace problem_statement_l89_89236

noncomputable def A : Set ℝ := {x | 2 < x ∧ x ≤ 6}
noncomputable def B : Set ℝ := {x | x^2 - 4 * x < 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m - 1}

theorem problem_statement (m : ℝ) :
    (A ∩ B = {x | 2 < x ∧ x < 4}) ∧
    (¬(A ∪ B) = {x | x ≤ 0 ∨ x > 6}) ∧
    (C m ⊆ B → m ∈ Set.Iic (5/2)) := 
by
  sorry

end problem_statement_l89_89236


namespace fixed_monthly_fee_l89_89281

theorem fixed_monthly_fee (x y : ℝ)
  (h1 : x + y = 15.80)
  (h2 : x + 3 * y = 28.62) :
  x = 9.39 :=
sorry

end fixed_monthly_fee_l89_89281


namespace evaluate_f_at_4_l89_89588

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem evaluate_f_at_4 : f 4 = 9 := by
  sorry

end evaluate_f_at_4_l89_89588


namespace max_a_inequality_l89_89994

theorem max_a_inequality (a : ℝ) :
  (∀ x : ℝ, x * a ≤ Real.exp (x - 1) + x^2 + 1) → a ≤ 3 := 
sorry

end max_a_inequality_l89_89994


namespace percent_of_percent_l89_89153

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l89_89153


namespace value_of_f2_l89_89993

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b * x + 3

theorem value_of_f2 (a b : ℝ) (h1 : f 1 a b = 7) (h2 : f 3 a b = 15) : f 2 a b = 11 :=
by
  sorry

end value_of_f2_l89_89993


namespace percent_of_percent_l89_89155

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l89_89155


namespace total_blood_cells_correct_l89_89790

-- Define the number of blood cells in the first and second samples.
def sample_1_blood_cells : ℕ := 4221
def sample_2_blood_cells : ℕ := 3120

-- Define the total number of blood cells.
def total_blood_cells : ℕ := sample_1_blood_cells + sample_2_blood_cells

-- Theorem stating the total number of blood cells based on the conditions.
theorem total_blood_cells_correct : total_blood_cells = 7341 :=
by
  -- Proof is omitted
  sorry

end total_blood_cells_correct_l89_89790


namespace find_f1_verify_function_l89_89069

theorem find_f1 (f : ℝ → ℝ) (h_mono : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 > f x2)
    (h1_pos : ∀ x : ℝ, 0 < x → f x > 1 / x^2)
    (h_eq : ∀ x : ℝ, 0 < x → (f x)^2 * f (f x - 1 / x^2) = (f 1)^3) :
    f 1 = 2 := sorry

theorem verify_function (f : ℝ → ℝ) (h_def : ∀ x : ℝ, 0 < x → f x = 2 / x^2) :
    (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 > f x2) ∧ (∀ x : ℝ, 0 < x → f x > 1 / x^2) ∧
    (∀ x : ℝ, 0 < x → (f x)^2 * f (f x - 1 / x^2) = (f 1)^3) := sorry

end find_f1_verify_function_l89_89069


namespace function_range_cosine_identity_l89_89589

theorem function_range_cosine_identity
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h₀ : 0 < ω)
  (h₁ : ∀ x, f x = (1/2) * Real.cos (ω * x) - (Real.sqrt 3 / 2) * Real.sin (ω * x))
  (h₂ : ∀ x, f (x + π / ω) = f x) :
  Set.Icc (f (-π / 3)) (f (π / 6)) = Set.Icc (-1 / 2) 1 :=
by
  sorry

end function_range_cosine_identity_l89_89589


namespace total_weight_correct_total_money_earned_correct_l89_89138

variable (records : List Int) (std_weight : Int)

-- Conditions
def deviation_sum (records : List Int) : Int := records.foldl (· + ·) 0

def batch_weight (std_weight : Int) (n : Int) (deviation_sum : Int) : Int :=
  deviation_sum + std_weight * n

def first_day_sales (total_weight : Int) (price_per_kg : Int) : Int :=
  price_per_kg * (total_weight / 2)

def second_day_sales (total_weight : Int) (first_day_sales_weight : Int) (discounted_price_per_kg : Int) : Int :=
  discounted_price_per_kg * (total_weight - first_day_sales_weight)

def total_earnings (first_day_sales : Int) (second_day_sales : Int) : Int :=
  first_day_sales + second_day_sales

-- Proof statements
theorem total_weight_correct : 
  deviation_sum records = 4 ∧ std_weight = 30 ∧ records.length = 8 → 
  batch_weight std_weight records.length (deviation_sum records) = 244 :=
by
  intro h
  sorry

theorem total_money_earned_correct :
  first_day_sales (batch_weight std_weight records.length (deviation_sum records)) 10 = 1220 ∧
  second_day_sales (batch_weight std_weight records.length (deviation_sum records)) (batch_weight std_weight records.length (deviation_sum records) / 2) (10 * 9 / 10) = 1098 →
  total_earnings 1220 1098 = 2318 :=
by
  intro h
  sorry

end total_weight_correct_total_money_earned_correct_l89_89138


namespace pure_imaginary_condition_l89_89480

theorem pure_imaginary_condition (a b : ℝ) : 
  (a = 0) ↔ (∃ b : ℝ, b ≠ 0 ∧ z = a + b * I) :=
sorry

end pure_imaginary_condition_l89_89480


namespace probability_even_equals_odd_when_eight_dice_rolled_l89_89962

theorem probability_even_equals_odd_when_eight_dice_rolled :
  let diceRollOutcome := {1, 2, 3, 4, 5, 6}
  let evenNumbers := {2, 4, 6}
  let oddNumbers := {1, 3, 5}
  let totalDice := 8
  ∀ numberEven numberOdd : ℕ, numberEven = 4 → numberOdd = 4 →
  let prob_even_odd := (Nat.choose totalDice numberEven) * (1/2)^totalDice
  prob_even_odd = 35 / 128 := sorry

end probability_even_equals_odd_when_eight_dice_rolled_l89_89962


namespace value_of_kaftan_l89_89025

theorem value_of_kaftan (K : ℝ) (h : (7 / 12) * (12 + K) = 5 + K) : K = 4.8 :=
by
  sorry

end value_of_kaftan_l89_89025


namespace johnsonville_max_members_l89_89304

theorem johnsonville_max_members 
  (n : ℤ) 
  (h1 : 15 * n % 30 = 6) 
  (h2 : 15 * n < 900) 
  : 15 * n ≤ 810 :=
sorry

end johnsonville_max_members_l89_89304


namespace additional_chair_frequency_l89_89906

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

end additional_chair_frequency_l89_89906


namespace molecular_weight_of_7_moles_AlPO4_is_correct_l89_89465

def atomic_weight_Al : Float := 26.98
def atomic_weight_P : Float := 30.97
def atomic_weight_O : Float := 16.00

def molecular_weight_AlPO4 : Float :=
  (1 * atomic_weight_Al) + (1 * atomic_weight_P) + (4 * atomic_weight_O)

noncomputable def weight_of_7_moles_AlPO4 : Float :=
  7 * molecular_weight_AlPO4

theorem molecular_weight_of_7_moles_AlPO4_is_correct :
  weight_of_7_moles_AlPO4 = 853.65 := by
  -- computation goes here
  sorry

end molecular_weight_of_7_moles_AlPO4_is_correct_l89_89465


namespace solution_set_of_inequality_system_l89_89301

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0) ∧ (x + 1 > 0) ↔ (-1 < x ∧ x < 1) :=
by
  -- proof to be filled in
  sorry

end solution_set_of_inequality_system_l89_89301


namespace smallest_f_for_perfect_square_l89_89407

theorem smallest_f_for_perfect_square (f : ℕ) (h₁: 3150 = 2 * 3 * 5^2 * 7) (h₂: ∃ m : ℕ, 3150 * f = m^2) :
  f = 14 :=
sorry

end smallest_f_for_perfect_square_l89_89407


namespace number_of_integers_between_sqrt10_and_sqrt100_l89_89078

theorem number_of_integers_between_sqrt10_and_sqrt100 :
  (∃ n : ℕ, n = 7) :=
sorry

end number_of_integers_between_sqrt10_and_sqrt100_l89_89078


namespace determinant_range_l89_89727

theorem determinant_range (x : ℝ) : 
  (2 * x - (3 - x) > 0) ↔ (x > 1) :=
by
  sorry

end determinant_range_l89_89727


namespace determinant_inequality_l89_89729

variable (x : ℝ)

def determinant (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem determinant_inequality (h : determinant 2 (3 - x) 1 x > 0) : x > 1 := by
  sorry

end determinant_inequality_l89_89729


namespace find_n_l89_89966

theorem find_n
    (h : Real.arctan (1 / 2) + Real.arctan (1 / 3) + Real.arctan (1 / 7) + Real.arctan (1 / n) = Real.pi / 2) :
    n = 46 :=
sorry

end find_n_l89_89966


namespace hcf_of_three_numbers_l89_89594

theorem hcf_of_three_numbers (a b c : ℕ) (h1 : a + b + c = 60)
  (h2 : Nat.lcm (Nat.lcm a b) c = 180)
  (h3 : (1:ℚ)/a + 1/b + 1/c = 11/120)
  (h4 : a * b * c = 900) :
  Nat.gcd (Nat.gcd a b) c = 5 :=
by
  sorry

end hcf_of_three_numbers_l89_89594


namespace probability_even_equals_odd_l89_89961

/-- Given eight 6-sided dice, prove that the probability 
that the number of dice showing even numbers equals 
the number of dice showing odd numbers is 35 / 128. -/
theorem probability_even_equals_odd (n : ℕ) (hn : n = 8) : 
  (∑ k in finset.range (n+1), 
    if k = 4 then (nat.choose n k) * ((1:ℝ/ℕ).nat_cast ^ k) * ((1:ℝ/ℕ).nat_cast ^ (n - k)) else 0) = 35 / 128 :=
by { sorry }

end probability_even_equals_odd_l89_89961


namespace A_receives_more_than_B_l89_89181

variable (A B C : ℝ)

axiom h₁ : A = 1/3 * (B + C)
axiom h₂ : B = 2/7 * (A + C)
axiom h₃ : A + B + C = 720

theorem A_receives_more_than_B : A - B = 20 :=
by
  sorry

end A_receives_more_than_B_l89_89181


namespace minimum_quadratic_expression_l89_89912

theorem minimum_quadratic_expression : ∃ (x : ℝ), (∀ y : ℝ, y^2 - 6*y + 5 ≥ -4) ∧ (x^2 - 6*x + 5 = -4) :=
by
  sorry

end minimum_quadratic_expression_l89_89912


namespace percent_of_y_equal_to_30_percent_of_60_percent_l89_89157

variable (y : ℝ)

theorem percent_of_y_equal_to_30_percent_of_60_percent (hy : y ≠ 0) :
  ((0.18 * y) / y) * 100 = 18 := by
  have hneq : y ≠ 0 := hy
  field_simp [hneq]
  norm_num
  sorry

end percent_of_y_equal_to_30_percent_of_60_percent_l89_89157


namespace total_protest_days_l89_89257

theorem total_protest_days (d1 : ℕ) (increase_percent : ℕ) (d2 : ℕ) (total_days : ℕ) (h1 : d1 = 4) (h2 : increase_percent = 25) (h3 : d2 = d1 + (d1 * increase_percent / 100)) : total_days = d1 + d2 → total_days = 9 :=
by
  intros
  sorry

end total_protest_days_l89_89257


namespace eggs_needed_per_month_l89_89273

def weekly_eggs_needed : ℕ := 10 + 14 + (14 / 2)

def weeks_in_month : ℕ := 4

def monthly_eggs_needed (weekly_eggs : ℕ) (weeks : ℕ) : ℕ :=
  weekly_eggs * weeks

theorem eggs_needed_per_month : 
  monthly_eggs_needed weekly_eggs_needed weeks_in_month = 124 :=
by {
  -- calculation details go here, but we leave it as sorry
  sorry
}

end eggs_needed_per_month_l89_89273


namespace kids_stay_home_lawrence_county_l89_89215

def total_kids_lawrence_county : ℕ := 1201565
def kids_camp_lawrence_county : ℕ := 610769

theorem kids_stay_home_lawrence_county : total_kids_lawrence_county - kids_camp_lawrence_county = 590796 := by
  sorry

end kids_stay_home_lawrence_county_l89_89215


namespace cube_edge_length_proof_l89_89929

-- Define the edge length of the cube
def edge_length_of_cube := 15

-- Define the volume of the cube
def volume_of_cube (a : ℕ) := a^3

-- Define the volume of the displaced water
def volume_of_displaced_water := 20 * 15 * 11.25

-- The theorem to prove
theorem cube_edge_length_proof : ∃ a : ℕ, volume_of_cube a = 3375 ∧ a = edge_length_of_cube := 
by {
  sorry
}

end cube_edge_length_proof_l89_89929


namespace arithmetic_sequence_m_l89_89978

theorem arithmetic_sequence_m (m : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, a n = 2 * n - 1) →
  (∀ n, S n = n * (2 * n - 1) / 2) →
  S m = (a m + a (m + 1)) / 2 →
  m = 2 :=
by
  sorry

end arithmetic_sequence_m_l89_89978


namespace a_pow_b_iff_a_minus_1_b_positive_l89_89516

theorem a_pow_b_iff_a_minus_1_b_positive (a b : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : 
  (a^b > 1) ↔ ((a - 1) * b > 0) := 
sorry

end a_pow_b_iff_a_minus_1_b_positive_l89_89516


namespace total_items_left_in_store_l89_89654

noncomputable def items_ordered : ℕ := 4458
noncomputable def items_sold : ℕ := 1561
noncomputable def items_in_storeroom : ℕ := 575

theorem total_items_left_in_store : 
  (items_ordered - items_sold) + items_in_storeroom = 3472 := 
by 
  sorry

end total_items_left_in_store_l89_89654


namespace volume_of_prism_l89_89442

theorem volume_of_prism (x y z : ℝ) (hx : x * y = 28) (hy : x * z = 45) (hz : y * z = 63) : x * y * z = 282 := by
  sorry

end volume_of_prism_l89_89442


namespace S10_value_l89_89139

def sequence_sum (n : ℕ) : ℕ :=
  (2^(n+1)) - 2 - n

theorem S10_value : sequence_sum 10 = 2036 := by
  sorry

end S10_value_l89_89139


namespace students_not_enrolled_in_either_l89_89413

variable (total_students french_students german_students both_students : ℕ)

theorem students_not_enrolled_in_either (h1 : total_students = 60)
                                        (h2 : french_students = 41)
                                        (h3 : german_students = 22)
                                        (h4 : both_students = 9) :
    total_students - (french_students + german_students - both_students) = 6 := by
  sorry

end students_not_enrolled_in_either_l89_89413


namespace symmetric_point_y_axis_l89_89893

theorem symmetric_point_y_axis (x y : ℝ) (hx : x = -3) (hy : y = 2) :
  (-x, y) = (3, 2) :=
by
  sorry

end symmetric_point_y_axis_l89_89893


namespace car_speed_constant_l89_89485

theorem car_speed_constant (v : ℝ) : 
  (1 / (v / 3600) - 1 / (80 / 3600) = 2) → v = 3600 / 47 := 
by
  sorry

end car_speed_constant_l89_89485


namespace min_value_expression_l89_89818

theorem min_value_expression : ∃ x y z : ℝ, (3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + z^2 + 6 * z + 10) = -7 / 2 :=
by sorry

end min_value_expression_l89_89818


namespace count_three_digit_numbers_between_l89_89845

theorem count_three_digit_numbers_between 
  (a b : ℕ) 
  (ha : a = 137) 
  (hb : b = 285) : 
  ∃ n, n = (b - a - 1) + 1 := 
sorry

end count_three_digit_numbers_between_l89_89845


namespace sufficient_condition_x_gt_2_l89_89128

theorem sufficient_condition_x_gt_2 (x : ℝ) (h : x > 2) : x^2 - 2 * x > 0 := by
  sorry

end sufficient_condition_x_gt_2_l89_89128


namespace train_platform_time_l89_89174

theorem train_platform_time :
  ∀ (L_train L_platform T_tree S D T_platform : ℝ),
    L_train = 1200 ∧ 
    T_tree = 120 ∧ 
    L_platform = 1100 ∧ 
    S = L_train / T_tree ∧ 
    D = L_train + L_platform ∧ 
    T_platform = D / S →
    T_platform = 230 :=
by
  intros
  sorry

end train_platform_time_l89_89174


namespace solution_set_inequality_l89_89903

theorem solution_set_inequality (x : ℝ) : 
  (2 < 1 / (x - 1) ∧ 1 / (x - 1) < 3) ↔ (4 / 3 < x ∧ x < 3 / 2) := 
by
  sorry

end solution_set_inequality_l89_89903


namespace grocery_delivery_amount_l89_89641

theorem grocery_delivery_amount (initial_savings final_price trips : ℕ) 
(fixed_charge : ℝ) (percent_charge : ℝ) (total_saved : ℝ) : 
  initial_savings = 14500 →
  final_price = 14600 →
  trips = 40 →
  fixed_charge = 1.5 →
  percent_charge = 0.05 →
  total_saved = final_price - initial_savings →
  60 + percent_charge * G = total_saved →
  G = 800 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end grocery_delivery_amount_l89_89641


namespace concatenated_number_not_power_of_two_l89_89345

theorem concatenated_number_not_power_of_two :
  ∀ (N : ℕ), (∀ i, 11111 ≤ i ∧ i ≤ 99999) →
  (N ≡ 0 [MOD 11111]) → ¬ ∃ k, N = 2^k :=
by
  sorry

end concatenated_number_not_power_of_two_l89_89345


namespace calculate_difference_l89_89568

theorem calculate_difference :
  let m := Nat.find (λ m, m > 99 ∧ m < 1000 ∧ m % 13 = 7)
  let n := Nat.find (λ n, n > 999 ∧ n < 10000 ∧ n % 13 = 7)
  n - m = 895 :=
by
  sorry

end calculate_difference_l89_89568


namespace percentage_of_smoking_teens_l89_89999

theorem percentage_of_smoking_teens (total_students : ℕ) (hospitalized_percentage : ℝ) (non_hospitalized_count : ℕ) 
  (h_total_students : total_students = 300)
  (h_hospitalized_percentage : hospitalized_percentage = 0.70)
  (h_non_hospitalized_count : non_hospitalized_count = 36) : 
  (non_hospitalized_count / (total_students * (1 - hospitalized_percentage))) * 100 = 40 := 
by 
  sorry

end percentage_of_smoking_teens_l89_89999


namespace solve_for_x_l89_89173

theorem solve_for_x (x : ℤ) (h : 3 * x + 36 = 48) : x = 4 := by
  sorry

end solve_for_x_l89_89173


namespace must_divisor_of_a_l89_89868

-- The statement
theorem must_divisor_of_a (a b c d : ℕ) (h1 : Nat.gcd a b = 18)
    (h2 : Nat.gcd b c = 45) (h3 : Nat.gcd c d = 60) (h4 : 90 < Nat.gcd d a ∧ Nat.gcd d a < 120) : 
    5 ∣ a := 
sorry

end must_divisor_of_a_l89_89868


namespace find_k_l89_89223

theorem find_k (θ : ℝ) :
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 = k + 2 * (tan θ^2 + (1 / tan θ)^2) →
  k = 5 := by
  sorry

end find_k_l89_89223


namespace math_proof_l89_89801

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def binom (n k : Nat) : Nat :=
  (factorial n) / ((factorial k) * (factorial (n - k)))

theorem math_proof :
  binom 20 6 * factorial 6 = 27907200 :=
by
  sorry

end math_proof_l89_89801


namespace no_solution_abs_eq_l89_89881

theorem no_solution_abs_eq (x : ℝ) (h : x > 0) : |x + 4| = 3 - x → false :=
by
  sorry

end no_solution_abs_eq_l89_89881


namespace problem_l89_89986

theorem problem (d r : ℕ) (a b c : ℕ) (ha : a = 1059) (hb : b = 1417) (hc : c = 2312)
  (h1 : d ∣ (b - a)) (h2 : d ∣ (c - a)) (h3 : d ∣ (c - b)) (hd : d > 1)
  (hr : r = a % d):
  d - r = 15 := sorry

end problem_l89_89986


namespace initial_bags_of_rice_l89_89638

theorem initial_bags_of_rice (sold restocked final initial : Int) 
  (h1 : sold = 23)
  (h2 : restocked = 132)
  (h3 : final = 164) 
  : ((initial - sold) + restocked = final) ↔ initial = 55 :=
by 
  have eq1 : ((initial - sold) + restocked = final) ↔ initial - 23 + 132 = 164 := by rw [h1, h2, h3]
  simp [eq1]
  sorry

end initial_bags_of_rice_l89_89638


namespace solve_mod_equation_l89_89211

theorem solve_mod_equation (y b n : ℤ) (h1 : 15 * y + 4 ≡ 7 [ZMOD 18]) (h2 : y ≡ b [ZMOD n]) (h3 : 2 ≤ n) (h4 : b < n) : b + n = 11 :=
sorry

end solve_mod_equation_l89_89211


namespace noodles_initial_count_l89_89503

theorem noodles_initial_count (noodles_given : ℕ) (noodles_now : ℕ) (initial_noodles : ℕ) 
  (h_given : noodles_given = 12) (h_now : noodles_now = 54) (h_initial_noodles : initial_noodles = noodles_now + noodles_given) : 
  initial_noodles = 66 :=
by 
  rw [h_now, h_given] at h_initial_noodles
  exact h_initial_noodles

-- Adding 'sorry' since the solution steps are not required

end noodles_initial_count_l89_89503


namespace time_after_increment_l89_89408

-- Define the current time in minutes
def current_time_minutes : ℕ := 15 * 60  -- 3:00 p.m. in minutes

-- Define the time increment in minutes
def time_increment : ℕ := 1567

-- Calculate the total time in minutes after the increment
def total_time_minutes : ℕ := current_time_minutes + time_increment

-- Convert total time back to hours and minutes
def calculated_hours : ℕ := total_time_minutes / 60
def calculated_minutes : ℕ := total_time_minutes % 60

-- The expected hours and minutes after the increment
def expected_hours : ℕ := 17 -- 17:00 hours which is 5:00 p.m.
def expected_minutes : ℕ := 7 -- 7 minutes

theorem time_after_increment :
  (calculated_hours - 24 * (calculated_hours / 24) = expected_hours) ∧ (calculated_minutes = expected_minutes) :=
by
  sorry

end time_after_increment_l89_89408


namespace find_constants_l89_89373

theorem find_constants (a b : ℚ) (h1 : 3 * a + b = 7) (h2 : a + 4 * b = 5) :
  a = 61 / 33 ∧ b = 8 / 11 :=
by
  sorry

end find_constants_l89_89373


namespace intersection_point_exists_correct_line_l89_89374

noncomputable def line1 (x y : ℝ) : Prop := 2 * x - 3 * y + 2 = 0
noncomputable def line2 (x y : ℝ) : Prop := 3 * x - 4 * y - 2 = 0
noncomputable def parallel_line (x y : ℝ) : Prop := 4 * x - 2 * y + 7 = 0
noncomputable def target_line (x y : ℝ) : Prop := 2 * x - y - 18 = 0

theorem intersection_point_exists (x y : ℝ) : line1 x y ∧ line2 x y → (x = 14 ∧ y = 10) := 
by sorry

theorem correct_line (x y : ℝ) : 
  (∃ (x y : ℝ), line1 x y ∧ line2 x y) ∧ parallel_line x y 
  → target_line x y :=
by sorry

end intersection_point_exists_correct_line_l89_89374


namespace cone_lateral_area_l89_89699

theorem cone_lateral_area (r l S: ℝ) (h1: r = 1 / 2) (h2: l = 1) (h3: S = π * r * l) : 
  S = π / 2 :=
by
  sorry

end cone_lateral_area_l89_89699


namespace percent_of_y_equal_to_30_percent_of_60_percent_l89_89160

variable (y : ℝ)

theorem percent_of_y_equal_to_30_percent_of_60_percent (hy : y ≠ 0) :
  ((0.18 * y) / y) * 100 = 18 := by
  have hneq : y ≠ 0 := hy
  field_simp [hneq]
  norm_num
  sorry

end percent_of_y_equal_to_30_percent_of_60_percent_l89_89160


namespace stories_in_building_l89_89782

-- Definitions of the conditions
def apartments_per_floor := 4
def people_per_apartment := 2
def total_people := 200

-- Definition of people per floor
def people_per_floor := apartments_per_floor * people_per_apartment

-- The theorem stating the desired conclusion
theorem stories_in_building :
  total_people / people_per_floor = 25 :=
by
  -- Insert the proof here
  sorry

end stories_in_building_l89_89782


namespace arithmetic_and_geometric_sequence_l89_89681

theorem arithmetic_and_geometric_sequence (a : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + 2) 
  (h_geom_seq : (a 2)^2 = a 0 * a 3) : 
  a 1 + a 2 = -10 := 
sorry

end arithmetic_and_geometric_sequence_l89_89681


namespace top_black_second_red_probability_l89_89634

-- Define the problem conditions in Lean
def num_standard_cards : ℕ := 52
def num_jokers : ℕ := 2
def num_total_cards : ℕ := num_standard_cards + num_jokers

def num_black_cards : ℕ := 26
def num_red_cards : ℕ := 26

-- Lean statement
theorem top_black_second_red_probability :
  (num_black_cards / num_total_cards * num_red_cards / (num_total_cards - 1)) = 338 / 1431 := by
  sorry

end top_black_second_red_probability_l89_89634


namespace intercepts_equal_l89_89519

theorem intercepts_equal (a : ℝ) :
  (∃ x y : ℝ, ax + y - 2 - a = 0 ∧
              y = 0 ∧ x = (a + 2) / a ∧
              x = 0 ∧ y = 2 + a) →
  (a = 1 ∨ a = -2) :=
by
  sorry

end intercepts_equal_l89_89519


namespace crocodiles_count_l89_89858

-- Definitions of constants
def alligators : Nat := 23
def vipers : Nat := 5
def total_dangerous_animals : Nat := 50

-- Theorem statement
theorem crocodiles_count :
  total_dangerous_animals - alligators - vipers = 22 :=
by
  sorry

end crocodiles_count_l89_89858


namespace estimated_survival_probability_l89_89574

-- Definitions of the given data
def number_of_trees_transplanted : List ℕ := [100, 1000, 5000, 8000, 10000, 15000, 20000]
def number_of_trees_survived : List ℕ := [87, 893, 4485, 7224, 8983, 13443, 18044]
def survival_rates : List ℝ := [0.870, 0.893, 0.897, 0.903, 0.898, 0.896, 0.902]

-- Question: Prove that the probability of survival of this type of young tree under these conditions is 0.9.
theorem estimated_survival_probability : 
  (1 / List.length number_of_trees_transplanted.to_real) * 
  (List.sum survival_rates) >= 0.9 ∧ 
  (1 / List.length number_of_trees_transplanted.to_real) * 
  (List.sum survival_rates) < 1 :=
  by sorry

end estimated_survival_probability_l89_89574


namespace find_a_l89_89836
open Real

noncomputable def f (a x : ℝ) := x * sin x + a * x

theorem find_a (a : ℝ) : (deriv (f a) (π / 2) = 1) → a = 0 := by
  sorry

end find_a_l89_89836


namespace city_tax_problem_l89_89852

theorem city_tax_problem :
  ∃ (x y : ℕ), 
    ((x + 3000) * (y - 10) = x * y) ∧
    ((x - 1000) * (y + 10) = x * y) ∧
    (x = 3000) ∧
    (y = 20) ∧
    (x * y = 60000) :=
by
  sorry

end city_tax_problem_l89_89852


namespace properties_of_dataset_l89_89327

def ordered_data : List ℕ := [30, 31, 31, 37, 40, 46, 47, 57, 62, 67]

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => if acc.count x > l.count acc then x else acc) 0

def median (l : List ℕ) : ℕ :=
  let n := l.length
  if n % 2 = 1 then
    l.get! (n / 2)
  else
    (l.get! (n / 2 - 1) + l.get! (n / 2)) / 2

def range (l : List ℕ) : ℕ :=
  l.maximum.getD 0 - l.minimum.getD 0

def quantile (l : List ℕ) (q : ℚ) : ℕ :=
  let idx := (q * rat.ofInt l.length).floor.toNat
  (l.get! idx + l.get! (idx + 1)) / 2

theorem properties_of_dataset :
  let dataset := [67, 57, 37, 40, 46, 62, 31, 47, 31, 30].sort
  mode dataset = 31 ∧
  median dataset ≠ 40 ∧
  range dataset = 37 ∧
  quantile dataset (10 / 100) = 30.5 := by
  sorry

end properties_of_dataset_l89_89327


namespace Roe_saved_15_per_month_aug_nov_l89_89733

-- Step 1: Define the given conditions
def savings_per_month_jan_jul : ℕ := 10
def months_jan_jul : ℕ := 7
def savings_dec : ℕ := 20
def total_savings_needed : ℕ := 150
def months_aug_nov : ℕ := 4

-- Step 2: Define the intermediary calculations based on the conditions
def total_saved_jan_jul := savings_per_month_jan_jul * months_jan_jul
def total_savings_aug_nov := total_savings_needed - total_saved_jan_jul - savings_dec

-- Step 3: Define what we need to prove
def savings_per_month_aug_nov : ℕ := total_savings_aug_nov / months_aug_nov

-- Step 4: State the proof goal
theorem Roe_saved_15_per_month_aug_nov :
  savings_per_month_aug_nov = 15 :=
by
  sorry

end Roe_saved_15_per_month_aug_nov_l89_89733


namespace translate_parabola_up_one_unit_l89_89344

theorem translate_parabola_up_one_unit (x : ℝ) :
  let y := 3 * x^2
  (y + 1) = 3 * x^2 + 1 :=
by
  -- Proof omitted
  sorry

end translate_parabola_up_one_unit_l89_89344


namespace number_of_people_per_taxi_l89_89297

def num_people_in_each_taxi (x : ℕ) (cars taxis vans total : ℕ) : Prop :=
  (cars = 3 * 4) ∧ (vans = 2 * 5) ∧ (total = 58) ∧ (taxis = 6 * x) ∧ (cars + vans + taxis = total)

theorem number_of_people_per_taxi
  (x cars taxis vans total : ℕ)
  (h1 : cars = 3 * 4)
  (h2 : vans = 2 * 5)
  (h3 : total = 58)
  (h4 : taxis = 6 * x)
  (h5 : cars + vans + taxis = total) :
  x = 6 :=
by
  sorry

end number_of_people_per_taxi_l89_89297


namespace tangent_line_equation_l89_89288

noncomputable def y (x : ℝ) := (2 * x - 1) / (x + 2)
def point : ℝ × ℝ := (-1, -3)
def tangent_eq (x y : ℝ) : Prop := 5 * x - y + 2 = 0

theorem tangent_line_equation :
  tangent_eq (-1) (-3) := 
sorry

end tangent_line_equation_l89_89288


namespace number_of_sequences_l89_89096

-- Define the number of targets and their columns
def targetSequence := ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']

-- Define our problem statement
theorem number_of_sequences :
  (List.permutations targetSequence).length = 4200 := by
  sorry

end number_of_sequences_l89_89096


namespace find_ratio_b_over_a_l89_89244

theorem find_ratio_b_over_a (a b : ℝ)
  (h1 : ∀ x, deriv (fun x => a * x^2 + b) x = 2 * a * x)
  (h2 : deriv (fun x => a * x^2 + b) 1 = 2)
  (h3 : a * 1^2 + b = 3) : b / a = 2 := 
sorry

end find_ratio_b_over_a_l89_89244


namespace valid_arrangements_count_is_20_l89_89438

noncomputable def count_valid_arrangements : ℕ :=
  sorry

theorem valid_arrangements_count_is_20 :
  count_valid_arrangements = 20 :=
  by
    sorry

end valid_arrangements_count_is_20_l89_89438


namespace perfect_square_trinomial_l89_89536

theorem perfect_square_trinomial (b : ℝ) : 
  (∃ (x : ℝ), 4 * x^2 + b * x + 1 = (2 * x + 1) ^ 2) ↔ (b = 4 ∨ b = -4) := 
by 
  sorry

end perfect_square_trinomial_l89_89536


namespace card_drawn_greater_probability_l89_89058

theorem card_drawn_greater_probability :
  let cards := {1, 2, 3, 4, 5}
  let total_draws := 5 * 5
  let favorable_events := 10
  (favorable_events.toRat / total_draws.toRat) = 2 / 5 :=
by
  sorry

end card_drawn_greater_probability_l89_89058


namespace nine_odot_three_l89_89907

-- Defining the operation based on the given conditions
axiom odot_def (a b : ℕ) : ℕ

axiom odot_eq_1 : odot_def 2 4 = 8
axiom odot_eq_2 : odot_def 4 6 = 14
axiom odot_eq_3 : odot_def 5 3 = 13
axiom odot_eq_4 : odot_def 8 7 = 23

-- Proving that 9 ⊙ 3 = 21
theorem nine_odot_three : odot_def 9 3 = 21 := 
by
  sorry

end nine_odot_three_l89_89907


namespace rowing_speed_upstream_l89_89632

theorem rowing_speed_upstream (V_s V_downstream : ℝ) (V_s_eq : V_s = 28) (V_downstream_eq : V_downstream = 31) : 
  V_s - (V_downstream - V_s) = 25 := 
by
  sorry

end rowing_speed_upstream_l89_89632


namespace expected_value_is_correct_l89_89020

def probability_of_rolling_one : ℚ := 1 / 4

def probability_of_other_numbers : ℚ := 3 / 4 / 5

def win_amount : ℚ := 8

def loss_amount : ℚ := -3

def expected_value : ℚ := (probability_of_rolling_one * win_amount) + 
                          (probability_of_other_numbers * 5 * loss_amount)

theorem expected_value_is_correct : expected_value = -0.25 :=
by 
  unfold expected_value probability_of_rolling_one probability_of_other_numbers win_amount loss_amount
  sorry

end expected_value_is_correct_l89_89020


namespace chef_meals_prepared_for_dinner_l89_89552

theorem chef_meals_prepared_for_dinner (lunch_meals_prepared lunch_meals_sold dinner_meals_total : ℕ) 
  (h1 : lunch_meals_prepared = 17)
  (h2 : lunch_meals_sold = 12)
  (h3 : dinner_meals_total = 10) :
  (dinner_meals_total - (lunch_meals_prepared - lunch_meals_sold)) = 5 :=
by
  -- Lean proof code to proceed from here
  sorry

end chef_meals_prepared_for_dinner_l89_89552


namespace vehicle_speed_l89_89034

theorem vehicle_speed (distance : ℝ) (time : ℝ) (h_dist : distance = 150) (h_time : time = 0.75) : distance / time = 200 :=
  by
    sorry

end vehicle_speed_l89_89034


namespace distance_to_destination_l89_89490

-- Conditions
def Speed : ℝ := 65 -- speed in km/hr
def Time : ℝ := 3   -- time in hours

-- Question to prove
theorem distance_to_destination : Speed * Time = 195 := by
  sorry

end distance_to_destination_l89_89490


namespace xiao_ming_final_score_correct_l89_89318

/-- Xiao Ming's scores in image, content, and effect are 9, 8, and 8 points, respectively.
    The weights (ratios) for these scores are 3:4:3.
    Prove that Xiao Ming's final competition score is 8.3 points. -/
def xiao_ming_final_score : Prop :=
  let image_score := 9
  let content_score := 8
  let effect_score := 8
  let image_weight := 3
  let content_weight := 4
  let effect_weight := 3
  let total_weight := image_weight + content_weight + effect_weight
  let weighted_score := (image_score * image_weight) + (content_score * content_weight) + (effect_score * effect_weight)
  weighted_score / total_weight = 8.3

theorem xiao_ming_final_score_correct : xiao_ming_final_score := by
  sorry

end xiao_ming_final_score_correct_l89_89318


namespace solve_for_k_l89_89367

theorem solve_for_k : ∃ k : ℤ, 2^5 - 10 = 5^2 + k ∧ k = -3 := by
  sorry

end solve_for_k_l89_89367


namespace correct_calculation_D_l89_89469

theorem correct_calculation_D (m : ℕ) : 
  (2 * m ^ 3) * (3 * m ^ 2) = 6 * m ^ 5 :=
by
  sorry

end correct_calculation_D_l89_89469


namespace mabel_total_walk_l89_89717

theorem mabel_total_walk (steps_mabel : ℝ) (ratio_helen : ℝ) (steps_helen : ℝ) (total_steps : ℝ) :
  steps_mabel = 4500 →
  ratio_helen = 3 / 4 →
  steps_helen = ratio_helen * steps_mabel →
  total_steps = steps_mabel + steps_helen →
  total_steps = 7875 := 
by
  intros h_steps_mabel h_ratio_helen h_steps_helen h_total_steps
  rw [h_steps_mabel, h_ratio_helen] at h_steps_helen
  rw [h_steps_helen, h_steps_mabel] at h_total_steps
  rw h_total_steps
  rw [h_steps_mabel, h_ratio_helen]
  linarith

end mabel_total_walk_l89_89717


namespace find_f_m_l89_89838

noncomputable def f (x : ℝ) := x^5 + Real.tan x - 3

theorem find_f_m (m : ℝ) (h : f (-m) = -2) : f m = -4 :=
sorry

end find_f_m_l89_89838


namespace ratio_size12_to_size6_l89_89892

-- Definitions based on conditions
def cheerleaders_size2 : ℕ := 4
def cheerleaders_size6 : ℕ := 10
def total_cheerleaders : ℕ := 19
def cheerleaders_size12 : ℕ := total_cheerleaders - (cheerleaders_size2 + cheerleaders_size6)

-- Proof statement
theorem ratio_size12_to_size6 : cheerleaders_size12.toFloat / cheerleaders_size6.toFloat = 1 / 2 := sorry

end ratio_size12_to_size6_l89_89892


namespace problem1_l89_89798

theorem problem1 : 2 * Real.sin (Real.pi / 3) - 3 * Real.tan (Real.pi / 6) = 0 := by
  sorry

end problem1_l89_89798


namespace chi_square_test_probability_distribution_expectation_value_l89_89307

variables (a b c d : ℕ)
variables (n : ℕ := a + b + c + d)

-- Chi-Square Calculation
noncomputable def K_squared : ℝ :=
  (n * ((a * d - b * c) ^ 2)) / ((a + b) * (c + d) * (a + c) * (b + d))

def crit_value_85 : ℝ := 2.072

theorem chi_square_test (a b c d : ℕ) (h1 : a = 12) (h2 : b = 4)
  (h3 : c = 9) (h4 : d = 5) (K_sq : ℝ := K_squared a b c d)
  (crit_val : ℝ := crit_value_85) :
  K_sq < crit_val := by
  sorry

-- Probability Distribution and Mathematical Expectation
def prob (n k : ℕ) : ℚ := nat.choose n k / nat.choose 9 3

noncomputable def expectation : ℚ :=
  (0 * prob 5 3) + (1 * prob 4 1 * prob 5 2) + (2 * prob 4 2 * prob 5 1) + (3 * prob 4 3)

theorem probability_distribution (h : 4 = 4) :
  prob 5 3 = 5 / 42 ∧ prob 4 1 * prob 5 2 = 10 / 21 ∧ prob 4 2 * prob 5 1 = 5 / 14 ∧ prob 4 3 = 1 / 21 := by
  sorry

theorem expectation_value : expectation = 4 / 3 := by
  sorry

end chi_square_test_probability_distribution_expectation_value_l89_89307


namespace surveys_completed_total_l89_89032

variable (regular_rate cellphone_rate total_earnings cellphone_surveys total_surveys : ℕ)
variable (h_regular_rate : regular_rate = 10)
variable (h_cellphone_rate : cellphone_rate = 13) -- 30% higher than regular_rate
variable (h_total_earnings : total_earnings = 1180)
variable (h_cellphone_surveys : cellphone_surveys = 60)
variable (h_total_surveys : total_surveys = cellphone_surveys + (total_earnings - (cellphone_surveys * cellphone_rate)) / regular_rate)

theorem surveys_completed_total :
  total_surveys = 100 :=
by
  sorry

end surveys_completed_total_l89_89032


namespace delta_max_success_ratio_l89_89415

theorem delta_max_success_ratio :
  ∃ (x y z w : ℕ),
  (0 < x ∧ x < (7 * y) / 12) ∧
  (0 < z ∧ z < (5 * w) / 8) ∧
  (y + w = 600) ∧
  (35 * x + 28 * z < 4200) ∧
  (x + z = 150) ∧ 
  (x + z) / 600 = 1 / 4 :=
by sorry

end delta_max_success_ratio_l89_89415


namespace euler_totient_problem_l89_89423

open Nat

def is_odd (n : ℕ) := n % 2 = 1

def is_power_of_2 (m : ℕ) := ∃ k : ℕ, m = 2^k

theorem euler_totient_problem (n : ℕ) (h1 : n > 0) (h2 : is_odd n) (h3 : is_power_of_2 (φ n)) (h4 : is_power_of_2 (φ (n + 1))) :
  is_power_of_2 (n + 1) ∨ n = 5 := 
sorry

end euler_totient_problem_l89_89423


namespace people_in_club_M_l89_89764

theorem people_in_club_M (m s z n : ℕ) (h1 : s = 18) (h2 : z = 11) (h3 : m + s + z + n = 60) (h4 : n ≤ 26) : m = 5 :=
sorry

end people_in_club_M_l89_89764


namespace ratio_t_q_l89_89534

theorem ratio_t_q (q r s t : ℚ) (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) : 
  t / q = 3 / 2 :=
by
  sorry

end ratio_t_q_l89_89534


namespace solution_1_solution_2_l89_89977

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem solution_1 :
  (∀ x : ℝ, f x = 2 * Real.sin (2 * x + Real.pi / 3)) :=
by sorry

theorem solution_2 (x0 : ℝ) (hx0 : x0 ∈ Set.Icc (Real.pi / 2) Real.pi) :
  f (x0 / 2) = -3 / 8 → 
  Real.cos (x0 + Real.pi / 6) = - Real.sqrt 741 / 32 - 3 / 32 :=
by sorry

end solution_1_solution_2_l89_89977


namespace max_value_of_f_l89_89815

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem max_value_of_f : ∀ x : ℝ, f x ≤ Real.sin 1 + 1 ∧ (f 0 = Real.sin 1 + 1) :=
by
  intro x
  sorry

end max_value_of_f_l89_89815


namespace solve_for_k_l89_89972

theorem solve_for_k (x y k : ℤ) (h1 : x = -3) (h2 : y = 2) (h3 : 2 * x + k * y = 0) : k = 3 :=
by
  sorry

end solve_for_k_l89_89972


namespace evaluate_expression_l89_89217

theorem evaluate_expression (c : ℕ) (hc : c = 4) : 
  ((c^c - 2 * c * (c-2)^c + c^2)^c) = 431441456 :=
by
  rw [hc]
  sorry

end evaluate_expression_l89_89217


namespace factorization_correct_l89_89669

-- Define the input expression
def expr (x y : ℝ) : ℝ := 2 * x^3 - 18 * x * y^2

-- Define the factorized form
def factorized_expr (x y : ℝ) : ℝ := 2 * x * (x + 3*y) * (x - 3*y)

-- Prove that the original expression is equal to the factorized form
theorem factorization_correct (x y : ℝ) : expr x y = factorized_expr x y := 
by sorry

end factorization_correct_l89_89669


namespace no_such_two_digit_number_exists_l89_89213

theorem no_such_two_digit_number_exists :
  ¬ ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
                 (10 * x + y = 2 * (x^2 + y^2) + 6) ∧
                 (10 * x + y = 4 * (x * y) + 6) := by
  -- We need to prove that no two-digit number satisfies
  -- both conditions.
  sorry

end no_such_two_digit_number_exists_l89_89213


namespace arina_sophia_divisible_l89_89648

theorem arina_sophia_divisible (N: ℕ) (k: ℕ) (large_seq: list ℕ): 
  (k = 81) → 
  (large_seq = (list.range' 71 (k + 1)).append (list.range' 82 (N + 1))) → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).nat_sum % 3 = 0 → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).last_digits 2 % 4 = 0 →
  (list.foldl (λ n d, 10 * n + d) 0 large_seq) % 12 = 0 → 
  N = 84 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end arina_sophia_divisible_l89_89648


namespace solution_set_l89_89238

-- Define the function f with given properties
axiom f : ℝ → ℝ

axiom domain_f : ∀ x : ℝ, f x = f x
axiom f_pos : ∀ x > 0, f x > 1
axiom f_mul : ∀ x y : ℝ, f (x + y) = f x * f y

-- Define the function log base 1/2
noncomputable def log_half (x : ℝ) : ℝ := log x / log (1/2)

-- The goal is to prove the inequality solution set
theorem solution_set : { x : ℝ | f (log_half x) ≤ 1 / f (log_half x + 1) } = { x : ℝ | x ≥ 4 } :=
by
  sorry

end solution_set_l89_89238


namespace tangent_line_eq_l89_89291

theorem tangent_line_eq {x y : ℝ} (h : y = (2 * x - 1) / (x + 2)) (hx1 : x = -1) (hy1 : y = -3) : 
  ∃ m b, 5 * x - y + 2 = 0 :=
by
  sorry

end tangent_line_eq_l89_89291


namespace height_of_tree_in_kilmer_park_l89_89897

-- Define the initial conditions
def initial_height_ft := 52
def growth_per_year_ft := 5
def years := 8
def ft_to_inch := 12

-- Define the expected result in inches
def expected_height_inch := 1104

-- State the problem as a theorem
theorem height_of_tree_in_kilmer_park :
  (initial_height_ft + growth_per_year_ft * years) * ft_to_inch = expected_height_inch :=
by
  sorry

end height_of_tree_in_kilmer_park_l89_89897


namespace count_permutations_perception_l89_89361

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def num_permutations (word : String) : ℕ :=
  let total_letters := word.length
  let freq_map := word.to_list.groupBy id
  let fact_chars := freq_map.toList.map (λ (c, l) => factorial l.length)
  factorial total_letters / fact_chars.foldl (*) 1

theorem count_permutations_perception :
  num_permutations "PERCEPTION" = 907200 := by
  sorry

end count_permutations_perception_l89_89361


namespace ratio_problem_l89_89532

theorem ratio_problem {q r s t : ℚ} (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) :
  t / q = 3 / 2 :=
sorry

end ratio_problem_l89_89532


namespace solve_quadratic_equation_l89_89885

theorem solve_quadratic_equation (x : ℝ) : 
  2 * x^2 - 4 * x = 6 - 3 * x ↔ (x = -3/2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l89_89885


namespace prime_digit_one_l89_89384

theorem prime_digit_one (p q r s : ℕ) (h1 : p > 10) (h2 : nat.prime p) (h3 : nat.prime q) (h4 : nat.prime r) (h5 : nat.prime s) 
                        (h6 : p < q) (h7 : q < r) (h8 : r < s) (h9 : q = p + 10) (h10 : r = q + 10) (h11 : s = r + 10) :
  (p % 10) = 1 := 
sorry

end prime_digit_one_l89_89384


namespace find_b_l89_89518

theorem find_b (a b c : ℝ) (A B C : ℝ) (h1 : a = 10) (h2 : c = 20) (h3 : B = 120) :
  b = 10 * Real.sqrt 7 :=
sorry

end find_b_l89_89518


namespace num_integer_values_satisfying_condition_l89_89965

theorem num_integer_values_satisfying_condition : 
  ∃ s : Finset ℤ, (∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ∧ s.card = 3 :=
by
  sorry

end num_integer_values_satisfying_condition_l89_89965


namespace modules_count_l89_89319

theorem modules_count (x y: ℤ) (hx: 10 * x + 35 * y = 450) (hy: x + y = 11) : y = 10 :=
by
  sorry

end modules_count_l89_89319


namespace cos_alpha_minus_beta_cos_alpha_plus_beta_l89_89776

variables (α β : Real) (h1 : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2)
           (h2 : Real.tan α * Real.tan β = 13/7)
           (h3 : Real.sin (α - β) = sqrt 5 / 3)

-- Part (1): Prove that cos (α - β) = 2/3
theorem cos_alpha_minus_beta : Real.cos (α - β) = 2 / 3 := by
  have h := h1
  have h := h2
  have h := h3
  sorry

-- Part (2): Prove that cos (α + β) = -1/5
theorem cos_alpha_plus_beta : Real.cos (α + β) = -1 / 5 := by
  have h := h1
  have h := h2
  have h := h3
  sorry

end cos_alpha_minus_beta_cos_alpha_plus_beta_l89_89776


namespace common_difference_divisible_by_6_l89_89003

theorem common_difference_divisible_by_6 (p q r d : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hp3 : p > 3) (hq3 : q > 3) (hr3 : r > 3) (h1 : q = p + d) (h2 : r = p + 2 * d) : d % 6 = 0 := 
sorry

end common_difference_divisible_by_6_l89_89003


namespace number_of_elements_less_than_2004_l89_89427

theorem number_of_elements_less_than_2004 (f : ℕ → ℕ) 
    (h0 : f 0 = 0) 
    (h1 : ∀ n : ℕ, (f (2 * n + 1)) ^ 2 - (f (2 * n)) ^ 2 = 6 * f n + 1) 
    (h2 : ∀ n : ℕ, f (2 * n) > f n) 
  : ∃ m : ℕ,  m = 128 ∧ ∀ x : ℕ, f x < 2004 → x < m := sorry

end number_of_elements_less_than_2004_l89_89427


namespace min_sum_of_primes_l89_89197

open Classical

theorem min_sum_of_primes (k m n p : ℕ) (h1 : 47 + m = k) (h2 : 53 + n = k) (h3 : 71 + p = k)
  (pm : Prime m) (pn : Prime n) (pp : Prime p) :
  m + n + p = 57 ↔ (k = 76 ∧ m = 29 ∧ n = 23 ∧ p = 5) :=
by {
  sorry
}

end min_sum_of_primes_l89_89197


namespace three_digit_number_divisible_by_7_l89_89162

theorem three_digit_number_divisible_by_7 (t : ℕ) :
  (n : ℕ) = 600 + 10 * t + 5 →
  n ≥ 100 ∧ n < 1000 →
  n % 10 = 5 →
  (n / 100) % 10 = 6 →
  n % 7 = 0 →
  n = 665 :=
by
  sorry

end three_digit_number_divisible_by_7_l89_89162


namespace max_packages_delivered_l89_89720

/-- Max's performance conditions and delivery problem -/
theorem max_packages_delivered
  (max_daily_capacity : ℕ) (num_days : ℕ)
  (days_max_performance : ℕ) (max_deliveries_days1 : ℕ)
  (days_half_performance : ℕ) (half_deliveries_days2 : ℕ)
  (days_fraction_performance : ℕ) (fraction_deliveries_days3 : ℕ)
  (last_two_days_fraction : ℕ) (fraction_last_two_days : ℕ):
  ∀ (remaining_capacity : ℕ), remaining_capacity = 
  max_daily_capacity * num_days - 
  (days_max_performance * max_deliveries_days1 + 
  days_half_performance * half_deliveries_days2 + 
  days_fraction_performance * fraction_deliveries_days3 * (1/7) + 
  last_two_days_fraction * fraction_last_two_days * (4/5)) := sorry

#eval max_packages_delivered 35 7 2 35 2 50 1 35 (2 * 28)

end max_packages_delivered_l89_89720


namespace sum_of_midpoint_coordinates_l89_89894

theorem sum_of_midpoint_coordinates : 
  let (x1, y1) := (4, 7)
  let (x2, y2) := (10, 19)
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  sum_of_coordinates = 20 := sorry

end sum_of_midpoint_coordinates_l89_89894


namespace find_ax5_plus_by5_l89_89284

theorem find_ax5_plus_by5 (a b x y : ℝ)
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := 
sorry

end find_ax5_plus_by5_l89_89284


namespace isosceles_trapezoid_fewest_axes_l89_89002

def equilateral_triangle_axes : Nat := 3
def isosceles_trapezoid_axes : Nat := 1
def rectangle_axes : Nat := 2
def regular_pentagon_axes : Nat := 5

theorem isosceles_trapezoid_fewest_axes :
  isosceles_trapezoid_axes < equilateral_triangle_axes ∧
  isosceles_trapezoid_axes < rectangle_axes ∧
  isosceles_trapezoid_axes < regular_pentagon_axes :=
by
  sorry

end isosceles_trapezoid_fewest_axes_l89_89002


namespace granola_bars_relation_l89_89434

theorem granola_bars_relation (x y z : ℕ) (h1 : z = x / (3 * y)) : z = x / (3 * y) :=
by {
    sorry
}

end granola_bars_relation_l89_89434


namespace find_fraction_l89_89541

theorem find_fraction (x y : ℝ) (h1 : (1/3) * (1/4) * x = 18) (h2 : y * x = 64.8) : y = 0.3 :=
sorry

end find_fraction_l89_89541


namespace region_area_l89_89910

theorem region_area : 
  (∃ (x y : ℝ), abs (4 * x - 16) + abs (3 * y + 9) ≤ 6) →
  (∀ (A : ℝ), (∀ x y : ℝ, abs (4 * x - 16) + abs (3 * y + 9) ≤ 6 → 0 ≤ A ∧ A = 6)) :=
by
  intro h exist_condtion
  sorry

end region_area_l89_89910


namespace line_through_points_l89_89229

theorem line_through_points (x1 y1 x2 y2 : ℝ) (h1 : x1 ≠ x2) (hx1 : x1 = -3) (hy1 : y1 = 1) (hx2 : x2 = 1) (hy2 : y2 = 5) :
  ∃ (m b : ℝ), (m + b = 5) ∧ (y1 = m * x1 + b) ∧ (y2 = m * x2 + b) :=
by
  sorry

end line_through_points_l89_89229


namespace solve_quadratic_equation_l89_89884

theorem solve_quadratic_equation (x : ℝ) : 
  (2 * x^2 - 4 * x = 6 - 3 * x) ↔ 
  (x = -3/2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l89_89884


namespace martha_apples_l89_89268

theorem martha_apples (initial_apples jane_apples extra_apples final_apples : ℕ)
  (h1 : initial_apples = 20)
  (h2 : jane_apples = 5)
  (h3 : extra_apples = 2)
  (h4 : final_apples = 4) :
  initial_apples - jane_apples - (jane_apples + extra_apples) - final_apples = final_apples := 
by
  sorry

end martha_apples_l89_89268


namespace fraction_multiplication_l89_89039

theorem fraction_multiplication :
  (2 / (3 : ℚ)) * (4 / 7) * (5 / 9) * (11 / 13) = 440 / 2457 :=
by
  sorry

end fraction_multiplication_l89_89039


namespace average_marks_of_failed_boys_l89_89859

def total_boys : ℕ := 120
def average_marks_all_boys : ℝ := 35
def number_of_passed_boys : ℕ := 100
def average_marks_passed_boys : ℝ := 39
def number_of_failed_boys : ℕ := total_boys - number_of_passed_boys

noncomputable def total_marks_all_boys : ℝ := average_marks_all_boys * total_boys
noncomputable def total_marks_passed_boys : ℝ := average_marks_passed_boys * number_of_passed_boys
noncomputable def total_marks_failed_boys : ℝ := total_marks_all_boys - total_marks_passed_boys
noncomputable def average_marks_failed_boys : ℝ := total_marks_failed_boys / number_of_failed_boys

theorem average_marks_of_failed_boys :
  average_marks_failed_boys = 15 :=
by
  -- The proof can be filled in here
  sorry

end average_marks_of_failed_boys_l89_89859


namespace intersect_not_A_B_l89_89073

open Set

-- Define the universal set U
def U := ℝ

-- Define set A
def A := {x : ℝ | x ≤ 3}

-- Define set B
def B := {x : ℝ | x ≤ 6}

-- Define the complement of A in U
def not_A := {x : ℝ | x > 3}

-- The proof problem
theorem intersect_not_A_B :
  (not_A ∩ B) = {x : ℝ | 3 < x ∧ x ≤ 6} :=
sorry

end intersect_not_A_B_l89_89073


namespace man_walking_time_l89_89619

theorem man_walking_time
  (T : ℕ) -- Let T be the time (in minutes) the man usually arrives at the station.
  (usual_arrival_home : ℕ) -- The time (in minutes) they usually arrive home, which is T + 30.
  (early_arrival : ℕ) (walking_start_time : ℕ) (early_home_arrival : ℕ)
  (usual_arrival_home_eq : usual_arrival_home = T + 30)
  (early_arrival_eq : early_arrival = T - 60)
  (walking_start_time_eq : walking_start_time = early_arrival)
  (early_home_arrival_eq : early_home_arrival = T)
  (time_saved : ℕ) (half_time_walk : ℕ)
  (time_saved_eq : time_saved = 30)
  (half_time_walk_eq : half_time_walk = time_saved / 2) :
  walking_start_time = half_time_walk := by
  sorry

end man_walking_time_l89_89619


namespace max_tiles_to_spell_CMWMC_l89_89019

theorem max_tiles_to_spell_CMWMC {Cs Ms Ws : ℕ} (hC : Cs = 8) (hM : Ms = 8) (hW : Ws = 8) : 
  ∃ (max_draws : ℕ), max_draws = 18 :=
by
  -- Assuming we have 8 C's, 8 M's, and 8 W's in the bag
  sorry

end max_tiles_to_spell_CMWMC_l89_89019


namespace tim_movie_marathon_l89_89460

variables (first_movie second_movie third_movie fourth_movie fifth_movie sixth_movie seventh_movie : ℝ)

/-- Tim's movie marathon --/
theorem tim_movie_marathon
  (first_movie_duration : first_movie = 2)
  (second_movie_duration : second_movie = 1.5 * first_movie)
  (third_movie_duration : third_movie = 0.8 * (first_movie + second_movie))
  (fourth_movie_duration : fourth_movie = 2 * second_movie)
  (fifth_movie_duration : fifth_movie = third_movie - 0.5)
  (sixth_movie_duration : sixth_movie = (second_movie + fourth_movie) / 2)
  (seventh_movie_duration : seventh_movie = 45 / fifth_movie) :
  first_movie + second_movie + third_movie + fourth_movie + fifth_movie + sixth_movie + seventh_movie = 35.8571 :=
sorry

end tim_movie_marathon_l89_89460


namespace parallelogram_base_length_l89_89622

theorem parallelogram_base_length (A : ℕ) (h b : ℕ) (h1 : A = b * h) (h2 : h = 2 * b) (h3 : A = 200) : b = 10 :=
by {
  sorry
}

end parallelogram_base_length_l89_89622


namespace probabilistic_dice_problem_l89_89799

noncomputable def fair_die : ℙ (fin 6) :=
  λ x, if x.1 = 5 then (1 / 6 : ℚ) else (1 / 6 : ℚ)

noncomputable def biased_die : ℙ (fin 6) :=
  λ x, if x.1 = 5 then (1 / 2 : ℚ) else (1 / 10 : ℚ)

theorem probabilistic_dice_problem :
  let p := 325
  let q := 656
  let prob_six_fifth_roll_given_two_sixes_in_first_four := (p : ℚ) / (q : ℚ)
  in p + q = 981 :=
begin
  -- Proof goes here.
  sorry,
end

end probabilistic_dice_problem_l89_89799


namespace workers_l89_89616

theorem workers (N C : ℕ) (h1 : N * C = 300000) (h2 : N * (C + 50) = 315000) : N = 300 :=
by
  sorry

end workers_l89_89616


namespace D_is_quadratic_l89_89470

-- Define the equations
def eq_A (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def eq_B (x : ℝ) : Prop := 2 * x^2 - 3 * x = 2 * (x^2 - 2)
def eq_C (x : ℝ) : Prop := x^3 - 2 * x + 7 = 0
def eq_D (x : ℝ) : Prop := (x - 2)^2 - 4 = 0

-- Define what it means to be a quadratic equation
def is_quadratic (f : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x ↔ a * x^2 + b * x + c = 0)

theorem D_is_quadratic : is_quadratic eq_D :=
sorry

end D_is_quadratic_l89_89470


namespace even_quadratic_increasing_l89_89996

theorem even_quadratic_increasing (m : ℝ) (h : ∀ x : ℝ, (m-1)*x^2 + 2*m*x + 1 = (m-1)*(-x)^2 + 2*m*(-x) + 1) :
  ∀ x1 x2 : ℝ, x1 < x2 ∧ x2 ≤ 0 → ((m-1)*x1^2 + 2*m*x1 + 1) < ((m-1)*x2^2 + 2*m*x2 + 1) :=
sorry

end even_quadratic_increasing_l89_89996


namespace solution_set_of_inequality_l89_89757

theorem solution_set_of_inequality : { x : ℝ | x^2 - 2 * x + 1 ≤ 0 } = {1} :=
sorry

end solution_set_of_inequality_l89_89757


namespace probability_even_equals_odd_l89_89960

/-- Given eight 6-sided dice, prove that the probability 
that the number of dice showing even numbers equals 
the number of dice showing odd numbers is 35 / 128. -/
theorem probability_even_equals_odd (n : ℕ) (hn : n = 8) : 
  (∑ k in finset.range (n+1), 
    if k = 4 then (nat.choose n k) * ((1:ℝ/ℕ).nat_cast ^ k) * ((1:ℝ/ℕ).nat_cast ^ (n - k)) else 0) = 35 / 128 :=
by { sorry }

end probability_even_equals_odd_l89_89960


namespace problem_statement_l89_89504

theorem problem_statement (p : ℕ) (hprime : Prime p) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ p = m^2 + n^2 ∧ p ∣ (m^3 + n^3 + 8 * m * n)) → p = 5 :=
by
  sorry

end problem_statement_l89_89504


namespace geometric_progression_digits_l89_89292

theorem geometric_progression_digits (a b N : ℕ) (q : ℚ)
  (h1 : b = a * q)
  (h2 : 10 * a + b = 3 * (a * q^2))
  (h3 : 10 ≤ N) (h4 : N ≤ 99) :
  N = 12 ∨ N = 24 ∨ N = 36 ∨ N = 48 :=
by
  sorry

end geometric_progression_digits_l89_89292


namespace probability_two_sixes_l89_89011

theorem probability_two_sixes (h1: ∀ i : ℕ, 1 ≤ i ∧ i ≤ 6 → prob_event (λ ω, ω = i) = 1/6)
  (h2 : @independent ℕ _ ℕ _ _ _ (λ i, (prob_event (λ ω, ω = i)))):
  (prob_event (λ ω1, ω1 = 6) * prob_event (λ ω2, ω2 = 6)) = 1 / 36 :=
by {
  sorry
}

end probability_two_sixes_l89_89011


namespace total_sample_any_candy_42_percent_l89_89090

-- Define percentages as rational numbers to avoid dealing with decimals directly
def percent_of_caught_A : ℚ := 12 / 100
def percent_of_not_caught_A : ℚ := 7 / 100
def percent_of_caught_B : ℚ := 5 / 100
def percent_of_not_caught_B : ℚ := 6 / 100
def percent_of_caught_C : ℚ := 9 / 100
def percent_of_not_caught_C : ℚ := 3 / 100

-- Sum up the percentages for those caught and not caught for each type of candy
def total_percent_A : ℚ := percent_of_caught_A + percent_of_not_caught_A
def total_percent_B : ℚ := percent_of_caught_B + percent_of_not_caught_B
def total_percent_C : ℚ := percent_of_caught_C + percent_of_not_caught_C

-- Sum of the total percentages for all types
def total_percent_sample_any_candy : ℚ := total_percent_A + total_percent_B + total_percent_C

theorem total_sample_any_candy_42_percent :
  total_percent_sample_any_candy = 42 / 100 :=
by
  sorry

end total_sample_any_candy_42_percent_l89_89090


namespace total_amount_shared_l89_89352

theorem total_amount_shared (total_amount : ℝ) 
  (h_debby : total_amount * 0.25 = (total_amount - 4500))
  (h_maggie : total_amount * 0.75 = 4500) : total_amount = 6000 :=
begin
  sorry
end

end total_amount_shared_l89_89352


namespace solve_equation_l89_89439

-- Definitions for the variables and the main equation
def equation (x y z : ℤ) : Prop :=
  5 * x^2 + y^2 + 3 * z^2 - 2 * y * z = 30

-- The statement that needs to be proved
theorem solve_equation (x y z : ℤ) :
  equation x y z ↔ (x, y, z) = (1, 5, 0) ∨ (x, y, z) = (1, -5, 0) ∨ (x, y, z) = (-1, 5, 0) ∨ (x, y, z) = (-1, -5, 0) :=
by
  sorry

end solve_equation_l89_89439


namespace problem_1_problem_2_l89_89110

-- Definition f
def f (a x : ℝ) : ℝ := a * x + 3 - abs (2 * x - 1)

-- Problem 1: If a = 1, prove ∀ x, f(1, x) ≤ 2
theorem problem_1 : (∀ x : ℝ, f 1 x ≤ 2) :=
sorry

-- Problem 2: The range of a for which f has a maximum value is -2 ≤ a ≤ 2
theorem problem_2 : (∀ a : ℝ, (∀ x : ℝ, (2 * x - 1 > 0 -> (f a x) ≤ (f a ((4 - a) / (2 * (4 - a))))) 
                        ∧ (2 * x - 1 ≤ 0 -> (f a x) ≤ (f a (1 - 2 / (1 - a))))) 
                        ↔ -2 ≤ a ∧ a ≤ 2) :=
sorry

end problem_1_problem_2_l89_89110


namespace mixed_bag_cost_l89_89658

def cost_per_pound_colombian : ℝ := 5.5
def cost_per_pound_peruvian : ℝ := 4.25
def total_weight : ℝ := 40
def weight_colombian : ℝ := 28.8

noncomputable def cost_per_pound_mixed_bag : ℝ :=
  (weight_colombian * cost_per_pound_colombian + (total_weight - weight_colombian) * cost_per_pound_peruvian) / total_weight

theorem mixed_bag_cost :
  cost_per_pound_mixed_bag = 5.15 :=
  sorry

end mixed_bag_cost_l89_89658


namespace hyperbola_equation_l89_89981

noncomputable def focal_distance : ℝ := 10
noncomputable def c : ℝ := 5
noncomputable def point_P : (ℝ × ℝ) := (2, 1)
noncomputable def eq1 : Prop := ∀ (x y : ℝ), (x^2) / 20 - (y^2) / 5 = 1 ↔ c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1
noncomputable def eq2 : Prop := ∀ (x y : ℝ), (y^2) / 5 - (x^2) / 20 = 1 ↔ c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1

theorem hyperbola_equation :
  (∃ a b : ℝ, c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1 ∧ 
    (∀ x y : ℝ, (x^2) / a^2 - (y^2) / b^2 = 1) ∨ 
    (∃ a' b' : ℝ, c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1 ∧ 
      (∀ x y : ℝ, (y^2) / a'^2 - (x^2) / b'^2 = 1))) :=
by sorry

end hyperbola_equation_l89_89981


namespace smallest_positive_integer_divides_l89_89671

theorem smallest_positive_integer_divides (m : ℕ) : 
  (∀ z : ℂ, z ≠ 0 → (z^11 + z^10 + z^8 + z^7 + z^5 + z^4 + z^2 + 1) ∣ (z^m - 1)) →
  (m = 88) :=
sorry

end smallest_positive_integer_divides_l89_89671


namespace number_of_balls_sold_l89_89276

theorem number_of_balls_sold 
  (selling_price : ℤ) (loss_per_5_balls : ℤ) (cost_price_per_ball : ℤ) (n : ℤ) 
  (h1 : selling_price = 720)
  (h2 : loss_per_5_balls = 5 * cost_price_per_ball)
  (h3 : cost_price_per_ball = 48)
  (h4 : (n * cost_price_per_ball) - selling_price = loss_per_5_balls) :
  n = 20 := 
by
  sorry

end number_of_balls_sold_l89_89276


namespace total_cost_is_346_l89_89418

-- Definitions of the given conditions
def total_people : ℕ := 35 + 5 + 1
def total_lunches : ℕ := total_people + 3
def vegetarian_lunches : ℕ := 10
def gluten_free_lunches : ℕ := 5
def nut_free_lunches : ℕ := 3
def halal_lunches : ℕ := 4
def veg_and_gluten_free_lunches : ℕ := 2
def regular_cost : ℕ := 7
def special_cost : ℕ := 8
def veg_and_gluten_free_cost : ℕ := 9

-- Calculate regular lunches considering dietary overlaps
def regular_lunches : ℕ := 
  total_lunches - vegetarian_lunches - gluten_free_lunches - nut_free_lunches - halal_lunches + veg_and_gluten_free_lunches

-- Calculate costs per category of lunches
def total_regular_cost : ℕ := regular_lunches * regular_cost
def total_vegetarian_cost : ℕ := (vegetarian_lunches - veg_and_gluten_free_lunches) * special_cost
def total_gluten_free_cost : ℕ := gluten_free_lunches * special_cost
def total_nut_free_cost : ℕ := nut_free_lunches * special_cost
def total_halal_cost : ℕ := halal_lunches * special_cost
def total_veg_and_gluten_free_cost : ℕ := veg_and_gluten_free_lunches * veg_and_gluten_free_cost

-- Calculate total cost
def total_cost : ℕ :=
  total_regular_cost + total_vegetarian_cost + total_gluten_free_cost + total_nut_free_cost + total_halal_cost + total_veg_and_gluten_free_cost

-- Theorem stating the main question
theorem total_cost_is_346 : total_cost = 346 :=
  by
    -- This is where the proof would go
    sorry

end total_cost_is_346_l89_89418


namespace toll_for_18_wheel_truck_l89_89000

theorem toll_for_18_wheel_truck : 
  let x := 5 
  let w := 15 
  let y := 2 
  let T := 3.50 + 0.50 * (x - 2) + 0.10 * w + y 
  T = 8.50 := 
by 
  -- let x := 5 
  -- let w := 15 
  -- let y := 2 
  -- let T := 3.50 + 0.50 * (x - 2) + 0.10 * w + y 
  -- Note: the let statements within the brackets above
  sorry

end toll_for_18_wheel_truck_l89_89000


namespace problem_statement_l89_89100

-- Definitions of parallel and perpendicular predicates (should be axioms or definitions in the context)
-- For simplification, assume we have a space with lines and planes, with corresponding relations.

axiom Line : Type
axiom Plane : Type
axiom parallel : Line → Line → Prop
axiom perpendicular : Line → Plane → Prop
axiom subset : Line → Plane → Prop

-- Assume the necessary conditions: m and n are lines, a and b are planes, with given relationships.
variables (m n : Line) (a b : Plane)

-- The conditions given.
variables (m_parallel_n : parallel m n)
variables (m_perpendicular_a : perpendicular m a)

-- The proposition to prove: If m parallel n and m perpendicular to a, then n is perpendicular to a.
theorem problem_statement : perpendicular n a :=
sorry

end problem_statement_l89_89100


namespace count_100_digit_numbers_divisible_by_3_l89_89943

def num_100_digit_numbers_divisible_by_3 : ℕ := (4^50 + 2) / 3

theorem count_100_digit_numbers_divisible_by_3 :
  ∃ n : ℕ, n = num_100_digit_numbers_divisible_by_3 :=
by
  use (4^50 + 2) / 3
  sorry

end count_100_digit_numbers_divisible_by_3_l89_89943


namespace no_nat_triplet_square_l89_89667

theorem no_nat_triplet_square (m n k : ℕ) : ¬ (∃ a b c : ℕ, m^2 + n + k = a^2 ∧ n^2 + k + m = b^2 ∧ k^2 + m + n = c^2) :=
by sorry

end no_nat_triplet_square_l89_89667


namespace exercise_books_purchasing_methods_l89_89005

theorem exercise_books_purchasing_methods :
  ∃ (ways : ℕ), ways = 5 ∧
  (∃ (x y z : ℕ), 2 * x + 5 * y + 11 * z = 40 ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1) ∧
  (∀ (x₁ y₁ z₁ x₂ y₂ z₂ : ℕ),
    2 * x₁ + 5 * y₁ + 11 * z₂ = 40 ∧ x₁ ≥ 1 ∧ y₁ ≥ 1 ∧ z₁ ≥ 1 →
    2 * x₂ + 5 * y₂ + 11 * z₂ = 40 ∧ x₂ ≥ 1 ∧ y₂ ≥ 1 ∧ z₂ ≥ 1 →
    (x₁, y₁, z₁) = (x₂, y₂, z₂)) := sorry

end exercise_books_purchasing_methods_l89_89005


namespace negation_proposition_l89_89295

open Set

theorem negation_proposition :
  ¬ (∀ x : ℝ, x^2 + 2 * x + 5 > 0) → (∃ x : ℝ, x^2 + 2 * x + 5 ≤ 0) :=
sorry

end negation_proposition_l89_89295


namespace number_of_k_combinations_with_repetition_l89_89970

noncomputable def combinations_with_repetition (n k : ℕ) : ℕ :=
  nat.choose (n + k - 1) k

theorem number_of_k_combinations_with_repetition (n k : ℕ) :
  combinations_with_repetition n k = nat.choose (n + k - 1) k := 
by
  sorry

end number_of_k_combinations_with_repetition_l89_89970


namespace find_a_and_other_root_l89_89686

theorem find_a_and_other_root (a : ℝ) (h : (2 : ℝ) ^ 2 - 3 * (2 : ℝ) + a = 0) :
  a = 2 ∧ ∃ x : ℝ, x ^ 2 - 3 * x + a = 0 ∧ x ≠ 2 ∧ x = 1 := 
by
  sorry

end find_a_and_other_root_l89_89686


namespace positive_difference_of_two_numbers_l89_89596

theorem positive_difference_of_two_numbers :
  ∀ (x y : ℝ), x + y = 8 → x^2 - y^2 = 24 → abs (x - y) = 3 :=
by
  intros x y h1 h2
  sorry

end positive_difference_of_two_numbers_l89_89596


namespace sheena_completes_in_37_weeks_l89_89580

-- Definitions based on the conditions
def hours_per_dress : List Nat := [15, 18, 20, 22, 24, 26, 28]
def hours_cycle : List Nat := [5, 3, 6, 4]
def finalize_hours : Nat := 10

-- The total hours needed to sew all dresses
def total_dress_hours : Nat := hours_per_dress.sum

-- The total hours needed including finalizing hours
def total_hours : Nat := total_dress_hours + finalize_hours

-- Total hours sewed in each 4-week cycle
def hours_per_cycle : Nat := hours_cycle.sum

-- Total number of weeks it will take to complete all dresses
def weeks_needed : Nat := 4 * ((total_hours + hours_per_cycle - 1) / hours_per_cycle)
def additional_weeks : Nat := if total_hours % hours_per_cycle == 0 then 0 else 1

theorem sheena_completes_in_37_weeks : weeks_needed + additional_weeks = 37 := by
  sorry

end sheena_completes_in_37_weeks_l89_89580


namespace acute_triangles_no_more_than_three_quarters_l89_89855

theorem acute_triangles_no_more_than_three_quarters 
  {n : ℕ} (h : n > 3) 
  (no_three_collinear : ∀ (p₁ p₂ p₃ : ℝ × ℝ), p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₃ ≠ p₁ → ¬ collinear ℝ (λ i, [p₁, p₂, p₃]) i) :
  let points := fin n → ℝ × ℝ in
  (number_of_acute_triangles points) ≤ 3 / 4 * (number_of_triangles points) := 
sorry

end acute_triangles_no_more_than_three_quarters_l89_89855


namespace expressions_equal_when_a_plus_b_plus_c_eq_1_l89_89743

theorem expressions_equal_when_a_plus_b_plus_c_eq_1
  (a b c : ℝ) (h : a + b + c = 1) :
  a + b * c = (a + b) * (a + c) :=
sorry

end expressions_equal_when_a_plus_b_plus_c_eq_1_l89_89743


namespace parabola_directrix_l89_89446

theorem parabola_directrix (x y : ℝ) (h : y = - (1/8) * x^2) : y = 2 :=
sorry

end parabola_directrix_l89_89446


namespace cyclist_traveled_18_miles_l89_89329

noncomputable def cyclist_distance (v t d : ℕ) : Prop :=
  (d = v * t) ∧ 
  (d = (v + 1) * (3 * t / 4)) ∧ 
  (d = (v - 1) * (t + 3))

theorem cyclist_traveled_18_miles : ∃ (d : ℕ), cyclist_distance 3 6 d ∧ d = 18 :=
by
  sorry

end cyclist_traveled_18_miles_l89_89329


namespace y_is_less_than_x_by_9444_percent_l89_89786

theorem y_is_less_than_x_by_9444_percent (x y : ℝ) (h : x = 18 * y) : (x - y) / x * 100 = 94.44 :=
by
  sorry

end y_is_less_than_x_by_9444_percent_l89_89786


namespace invitations_per_package_l89_89346

-- Definitions based on conditions in the problem.
def numPackages : Nat := 5
def totalInvitations : Nat := 45

-- Definition of the problem and proof statement.
theorem invitations_per_package :
  totalInvitations / numPackages = 9 :=
by
  sorry

end invitations_per_package_l89_89346


namespace greatest_value_a_maximum_value_a_l89_89816

-- Define the quadratic polynomial
def quadratic (a : ℝ) : ℝ := -a^2 + 9 * a - 20

-- The statement to be proven:
theorem greatest_value_a : ∀ a : ℝ, (quadratic a ≥ 0) → a ≤ 5 := 
sorry

theorem maximum_value_a : quadratic 5 = 0 :=
sorry

end greatest_value_a_maximum_value_a_l89_89816


namespace intercept_sum_modulo_l89_89919

theorem intercept_sum_modulo (x_0 y_0 : ℤ) (h1 : 0 ≤ x_0) (h2 : x_0 < 17) (h3 : 0 ≤ y_0) (h4 : y_0 < 17)
                       (hx : 5 * x_0 ≡ 2 [ZMOD 17])
                       (hy : 3 * y_0 ≡ 15 [ZMOD 17]) :
    x_0 + y_0 = 19 := 
by
  sorry

end intercept_sum_modulo_l89_89919


namespace number_of_triangles_l89_89259

theorem number_of_triangles (n : ℕ) (hn : 0 < n) :
  ∃ t, t = (n + 2) ^ 2 - 2 * (⌊ (n : ℝ) / 2 ⌋) / 4 :=
by
  sorry

end number_of_triangles_l89_89259


namespace max_cylinder_volume_in_cone_l89_89376

theorem max_cylinder_volume_in_cone :
  ∃ x, (0 < x ∧ x < 1) ∧ ∀ y, (0 < y ∧ y < 1 → y ≠ x → ((π * (-2 * y^3 + 2 * y^2)) ≤ (π * (-2 * x^3 + 2 * x^2)))) ∧ 
  (π * (-2 * x^3 + 2 * x^2) = 8 * π / 27) := sorry

end max_cylinder_volume_in_cone_l89_89376


namespace eqD_is_linear_l89_89916

-- Definitions for the given equations
def eqA (x y : ℝ) : Prop := 3 * x - 2 * y = 1
def eqB (x : ℝ) : Prop := 1 + (1 / x) = x
def eqC (x : ℝ) : Prop := x^2 = 9
def eqD (x : ℝ) : Prop := 2 * x - 3 = 5

-- Definition of a linear equation in one variable
def isLinear (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, (∀ x : ℝ, eq x ↔ a * x + b = c)

-- Theorem stating that eqD is a linear equation
theorem eqD_is_linear : isLinear eqD :=
  sorry

end eqD_is_linear_l89_89916


namespace fly_distance_to_ceiling_l89_89909

theorem fly_distance_to_ceiling :
  ∀ (x y z : ℝ), 
  (x = 3) → 
  (y = 4) → 
  (z * z + 25 = 49) →
  z = 2 * Real.sqrt 6 :=
by
  sorry

end fly_distance_to_ceiling_l89_89909


namespace problem_statement_l89_89108

noncomputable def alpha : ℝ := 3 + Real.sqrt 8
noncomputable def beta : ℝ := 3 - Real.sqrt 8
noncomputable def x : ℝ := alpha^(500)
noncomputable def N : ℝ := alpha^(500) + beta^(500)
noncomputable def n : ℝ := N - 1
noncomputable def f : ℝ := x - n
noncomputable def one_minus_f : ℝ := 1 - f

theorem problem_statement : x * one_minus_f = 1 :=
by
  -- Insert the proof here
  sorry

end problem_statement_l89_89108


namespace third_even_number_sequence_l89_89821

theorem third_even_number_sequence (x : ℕ) (h_even : x % 2 = 0) (h_sum : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) = 180) : x + 4 = 30 :=
by
  sorry

end third_even_number_sequence_l89_89821


namespace sphere_wedge_volume_l89_89496

theorem sphere_wedge_volume (circumference : ℝ) (num_wedges : ℕ) (V : ℝ) (wedge_volume : ℝ) :
  circumference = 18 * Real.pi → num_wedges = 6 → V = (4 / 3) * Real.pi * (9^3) → wedge_volume = V / 6 → 
  wedge_volume = 162 * Real.pi :=
by
  intros h1 h2 h3 h4
  rw h3 at h4
  rw [←Real.pi_mul, ←mul_assoc, Nat.cast_bit1, Nat.cast_bit0, Nat.cast_one, pow_succ, pow_one, ←mul_assoc] at h4
  rw [mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc] at h4
  sorry

end sphere_wedge_volume_l89_89496


namespace base_number_exponent_l89_89084

theorem base_number_exponent (x : ℝ) (h : ((x^4) * 3.456789) ^ 12 = y) (has_24_digits : true) : x = 10^12 :=
  sorry

end base_number_exponent_l89_89084


namespace Dave_pays_4_more_than_Doug_l89_89048

-- Define the conditions
def pizza_cost : ℝ := 8
def anchovy_cost : ℝ := 2
def number_of_slices : ℕ := 8
def Dave_slices_with_anchovies : ℕ := 4
def Dave_plain_slices : ℕ := 1
def Doug_plain_slices : ℕ := 3

-- Calculate total cost
def total_pizza_cost : ℝ := pizza_cost + anchovy_cost

-- Calculate cost per slice
def cost_per_slice : ℝ := total_pizza_cost / number_of_slices

-- Calculate Dave's total cost
def Dave_total_cost : ℝ := (cost_per_slice * (Dave_slices_with_anchovies + Dave_plain_slices))

-- Calculate Doug's total cost
def Doug_total_cost : ℝ := (cost_per_slice * Doug_plain_slices)

-- Calculate the difference in payment
def difference_in_payment : ℝ := Dave_total_cost - Doug_total_cost

-- State the theorem
theorem Dave_pays_4_more_than_Doug : difference_in_payment = 4 := by
  sorry

end Dave_pays_4_more_than_Doug_l89_89048


namespace count_integers_between_sqrts_l89_89081

theorem count_integers_between_sqrts (a b : ℝ) (h1 : a = 10) (h2 : b = 100) :
  let lower_bound := Int.ceil (Real.sqrt a),
      upper_bound := Int.floor (Real.sqrt b) in
  (upper_bound - lower_bound + 1) = 7 := 
by
  rw [h1, h2]
  let lower_bound := Int.ceil (Real.sqrt 10)
  let upper_bound := Int.floor (Real.sqrt 100)
  have h_lower : lower_bound = 4 := by sorry
  have h_upper : upper_bound = 10 := by sorry
  rw [h_lower, h_upper]
  norm_num
  sorry

end count_integers_between_sqrts_l89_89081


namespace find_range_of_a_l89_89111

noncomputable def range_of_a (a : ℝ) : Prop :=
∀ (x : ℝ) (θ : ℝ), (0 ≤ θ ∧ θ ≤ (Real.pi / 2)) → 
  let α := (x + 3, x)
  let β := (2 * Real.sin θ * Real.cos θ, a * Real.sin θ + a * Real.cos θ)
  let sum := (α.1 + β.1, α.2 + β.2)
  (sum.1^2 + sum.2^2)^(1/2) ≥ Real.sqrt 2

theorem find_range_of_a : range_of_a a ↔ (a ≤ 1 ∨ a ≥ 5) :=
sorry

end find_range_of_a_l89_89111


namespace x_solves_quadratic_and_sum_is_75_l89_89445

theorem x_solves_quadratic_and_sum_is_75
  (x a b : ℕ) (h : x^2 + 10 * x = 45) (hx_pos : 0 < x) (hx_form : x = Nat.sqrt a - b) 
  (ha_pos : 0 < a) (hb_pos : 0 < b)
  : a + b = 75 := 
sorry

end x_solves_quadratic_and_sum_is_75_l89_89445


namespace line_circle_interaction_l89_89253

theorem line_circle_interaction (a : ℝ) :
  let r := 10
  let d := |a| / 5
  let intersects := -50 < a ∧ a < 50 
  let tangent := a = 50 ∨ a = -50 
  let separate := a < -50 ∨ a > 50 
  (d < r ↔ intersects) ∧ (d = r ↔ tangent) ∧ (d > r ↔ separate) :=
by sorry

end line_circle_interaction_l89_89253


namespace intersection_l89_89247

def setA : Set ℝ := { x : ℝ | x^2 - 2*x - 3 < 0 }
def setB : Set ℝ := { x : ℝ | x > 1 }

theorem intersection (x : ℝ) : x ∈ setA ∧ x ∈ setB ↔ 1 < x ∧ x < 3 := by
  sorry

end intersection_l89_89247


namespace base_conversion_to_zero_l89_89127

theorem base_conversion_to_zero (A B : ℕ) (hA : 0 ≤ A ∧ A < 12) (hB : 0 ≤ B ∧ B < 5) 
    (h1 : 12 * A + B = 5 * B + A) : 12 * A + B = 0 :=
by
  sorry

end base_conversion_to_zero_l89_89127


namespace eggs_needed_per_month_l89_89274

def weekly_eggs_needed : ℕ := 10 + 14 + (14 / 2)

def weeks_in_month : ℕ := 4

def monthly_eggs_needed (weekly_eggs : ℕ) (weeks : ℕ) : ℕ :=
  weekly_eggs * weeks

theorem eggs_needed_per_month : 
  monthly_eggs_needed weekly_eggs_needed weeks_in_month = 124 :=
by {
  -- calculation details go here, but we leave it as sorry
  sorry
}

end eggs_needed_per_month_l89_89274


namespace total_protest_days_l89_89258

theorem total_protest_days (d1 : ℕ) (increase_percent : ℕ) (d2 : ℕ) (total_days : ℕ) (h1 : d1 = 4) (h2 : increase_percent = 25) (h3 : d2 = d1 + (d1 * increase_percent / 100)) : total_days = d1 + d2 → total_days = 9 :=
by
  intros
  sorry

end total_protest_days_l89_89258


namespace monthly_income_of_P_l89_89443

-- Define variables and assumptions
variables (P Q R : ℝ)
axiom avg_P_Q : (P + Q) / 2 = 5050
axiom avg_Q_R : (Q + R) / 2 = 6250
axiom avg_P_R : (P + R) / 2 = 5200

-- Prove that the monthly income of P is 4000
theorem monthly_income_of_P : P = 4000 :=
by
  sorry

end monthly_income_of_P_l89_89443


namespace harry_sandy_meet_point_l89_89988

theorem harry_sandy_meet_point :
  let H : ℝ × ℝ := (10, -3)
  let S : ℝ × ℝ := (2, 7)
  let t : ℝ := 2 / 3
  let meet_point : ℝ × ℝ := (H.1 + t * (S.1 - H.1), H.2 + t * (S.2 - H.2))
  meet_point = (14 / 3, 11 / 3) := 
by
  sorry

end harry_sandy_meet_point_l89_89988


namespace proof_problem_l89_89714

-- Given definitions and conditions
def ellipse (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 16 = 1

def focal_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

def foci_positions (c : ℝ) : (ℝ × ℝ) := (-c, c)

def point_P_on_ellipse (x y : ℝ) : Prop := ellipse x y

def angle_F1_P_F2_half_pi_over_three (F1 P F2 : ℝ × ℝ) : Prop := 
  ∠ (F1 - P) (F2 - P) = Real.pi / 3

-- Goal is to prove the dot product
def dot_product_correct (F1 F2 P : ℝ × ℝ) : Prop :=
  let (x1, y1) := F1 in
  let (x2, y2) := F2 in
  let (px, py) := P in
  (px - x1) * (px - x2) + (py - y1) * (py - y2) = 32 / 3

-- Putting it all together to state the theorem
theorem proof_problem (x y : ℝ) (h1 : point_P_on_ellipse x y)
  (a b c : ℝ) (h2 : a = 5) (h3 : b = 4) (h4 : c = focal_distance a b)
  (h5 : F1 F2 := foci_positions c)
  (h6 : angle_F1_P_F2_half_pi_over_three F1 (x, y) F2) :
  dot_product_correct F1 F2 (x, y) :=
sorry

end proof_problem_l89_89714


namespace sequence_a4_value_l89_89072

theorem sequence_a4_value :
  ∃ (a : ℕ → ℕ), (a 1 = 1) ∧ (∀ n, a (n+1) = 2 * a n + 1) ∧ (a 4 = 15) :=
by
  sorry

end sequence_a4_value_l89_89072


namespace base7_to_base5_l89_89945

theorem base7_to_base5 (n : ℕ) (h : n = 305) : 
    3 * 7 ^ 2 + 0 * 7 ^ 1 + 5 = 152 → 152 = 1 * 5 ^ 3 + 1 * 5 ^ 2 + 0 * 5 ^ 1 + 2 * 5 ^ 0 → 305 = 1102 :=
by
  intros h1 h2
  sorry

end base7_to_base5_l89_89945


namespace molecular_weight_al_fluoride_l89_89313

/-- Proving the molecular weight of Aluminum fluoride calculation -/
theorem molecular_weight_al_fluoride (x : ℕ) (h : 10 * x = 840) : x = 84 :=
by sorry

end molecular_weight_al_fluoride_l89_89313


namespace fraction_simplification_l89_89221

noncomputable def x : ℚ := 0.714714714 -- Repeating decimal representation for x
noncomputable def y : ℚ := 2.857857857 -- Repeating decimal representation for y

theorem fraction_simplification :
  (x / y) = (714 / 2855) :=
by
  sorry

end fraction_simplification_l89_89221


namespace arrange_PERCEPTION_l89_89665

theorem arrange_PERCEPTION :
  ∀ n k1 k2 k3 : ℕ, n = 10 → k1 = 2 → k2 = 2 → k3 = 2 →
  (nat.factorial n) / (nat.factorial k1 * nat.factorial k2 * nat.factorial k3) = 453600 := 
by
  intros n k1 k2 k3 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end arrange_PERCEPTION_l89_89665


namespace exists_five_digit_with_product_10080_l89_89146

noncomputable def has_digit_product_10080 (n : ℕ) : Prop :=
  (n.digits 10).prod = 10080

theorem exists_five_digit_with_product_10080 : 
  ∃ n : ℕ, n < 100000 ∧ 10000 ≤ n ∧ has_digit_product_10080 n ∧ n = 98754 :=
by
  sorry

end exists_five_digit_with_product_10080_l89_89146


namespace percentOfNonUnionWomenIs90_l89_89703

variable (totalEmployees : ℕ) (percentMen : ℚ) (percentUnionized : ℚ) (percentUnionizedMen : ℚ)

noncomputable def percentNonUnionWomen : ℚ :=
  let numberOfMen := percentMen * totalEmployees
  let numberOfUnionEmployees := percentUnionized * totalEmployees
  let numberOfUnionMen := percentUnionizedMen * numberOfUnionEmployees
  let numberOfNonUnionEmployees := totalEmployees - numberOfUnionEmployees
  let numberOfNonUnionMen := numberOfMen - numberOfUnionMen
  let numberOfNonUnionWomen := numberOfNonUnionEmployees - numberOfNonUnionMen
  (numberOfNonUnionWomen / numberOfNonUnionEmployees) * 100

theorem percentOfNonUnionWomenIs90
  (h1 : percentMen = 46 / 100)
  (h2 : percentUnionized = 60 / 100)
  (h3 : percentUnionizedMen = 70 / 100) : percentNonUnionWomen 100 46 60 70 = 90 :=
sorry

end percentOfNonUnionWomenIs90_l89_89703


namespace four_digit_numbers_divisible_by_90_l89_89397

theorem four_digit_numbers_divisible_by_90 : 
  let is_valid a b := (a + b) % 9 = 0 ∧ b % 2 = 0
  let nums := { (a, b) // is_valid a b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 }
  (Finset.card nums) = 5 :=
by
  sorry

end four_digit_numbers_divisible_by_90_l89_89397


namespace greatest_int_less_than_200_with_gcd_18_eq_9_l89_89911

theorem greatest_int_less_than_200_with_gcd_18_eq_9 :
  ∃ n, n < 200 ∧ Int.gcd n 18 = 9 ∧ ∀ m, m < 200 ∧ Int.gcd m 18 = 9 → m ≤ n :=
sorry

end greatest_int_less_than_200_with_gcd_18_eq_9_l89_89911


namespace fraction_of_shaded_area_l89_89311

theorem fraction_of_shaded_area (total_length total_width : ℕ) (total_area : ℕ)
  (quarter_fraction half_fraction : ℚ)
  (h1 : total_length = 15) 
  (h2 : total_width = 20)
  (h3 : total_area = total_length * total_width)
  (h4 : quarter_fraction = 1 / 4)
  (h5 : half_fraction = 1 / 2) :
  (half_fraction * quarter_fraction * total_area) / total_area = 1 / 8 :=
by
  sorry

end fraction_of_shaded_area_l89_89311


namespace paul_mowing_lawns_l89_89722

theorem paul_mowing_lawns : 
  ∃ M : ℕ, 
    (∃ money_made_weeating : ℕ, money_made_weeating = 13) ∧
    (∃ spending_per_week : ℕ, spending_per_week = 9) ∧
    (∃ weeks_last : ℕ, weeks_last = 9) ∧
    (M + 13 = 9 * 9) → 
    M = 68 := by
sorry

end paul_mowing_lawns_l89_89722


namespace isosceles_triangle_base_angle_l89_89097

theorem isosceles_triangle_base_angle (a b h θ : ℝ)
  (h1 : a^2 = 4 * b^2 * h)
  (h_b : b = 2 * a * Real.cos θ)
  (h_h : h = a * Real.sin θ) :
  θ = Real.arccos (1/4) :=
by
  sorry

end isosceles_triangle_base_angle_l89_89097


namespace non_vegan_gluten_cupcakes_eq_28_l89_89027

def total_cupcakes : ℕ := 80
def gluten_free_cupcakes : ℕ := total_cupcakes / 2
def vegan_cupcakes : ℕ := 24
def vegan_gluten_free_cupcakes : ℕ := vegan_cupcakes / 2
def non_vegan_cupcakes : ℕ := total_cupcakes - vegan_cupcakes
def gluten_cupcakes : ℕ := total_cupcakes - gluten_free_cupcakes
def non_vegan_gluten_cupcakes : ℕ := gluten_cupcakes - vegan_gluten_free_cupcakes

theorem non_vegan_gluten_cupcakes_eq_28 :
  non_vegan_gluten_cupcakes = 28 := by
  sorry

end non_vegan_gluten_cupcakes_eq_28_l89_89027


namespace a8_eq_128_l89_89678

-- Definitions of conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions
axiom a2_eq_2 : a 2 = 2
axiom a3_mul_a4_eq_32 : a 3 * a 4 = 32
axiom is_geometric : is_geometric_sequence a q

-- Statement to prove
theorem a8_eq_128 : a 8 = 128 :=
sorry

end a8_eq_128_l89_89678


namespace largest_expr_is_a_squared_plus_b_squared_l89_89676

noncomputable def largest_expression (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a ≠ b) : Prop :=
  (a^2 + b^2 > a - b) ∧ (a^2 + b^2 > a + b) ∧ (a^2 + b^2 > 2 * a * b)

theorem largest_expr_is_a_squared_plus_b_squared (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a ≠ b) : 
  largest_expression a b h₁ h₂ h₃ :=
by
  sorry

end largest_expr_is_a_squared_plus_b_squared_l89_89676


namespace polygon_diagonals_l89_89635

theorem polygon_diagonals (n : ℕ) (k_0 k_1 k_2 : ℕ)
  (h1 : 2 * k_2 + k_1 = n)
  (h2 : k_2 + k_1 + k_0 = n - 2) :
  k_2 ≥ 2 :=
sorry

end polygon_diagonals_l89_89635


namespace original_price_of_sarees_l89_89593

theorem original_price_of_sarees (P : ℝ):
  (0.80 * P) * 0.95 = 152 → P = 200 :=
by
  intro h1
  -- You can omit the proof here because the task requires only the statement.
  sorry

end original_price_of_sarees_l89_89593


namespace cos_sum_to_product_l89_89131

theorem cos_sum_to_product (x : ℝ) : 
  (∃ a b c d : ℕ, a * Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x) =
  Real.cos (2 * x) + Real.cos (6 * x) + Real.cos (10 * x) + Real.cos (14 * x) 
  ∧ a + b + c + d = 18) :=
sorry

end cos_sum_to_product_l89_89131


namespace speed_of_current_l89_89185

theorem speed_of_current (d : ℝ) (c : ℝ) : 
  ∀ (h1 : ∀ (t : ℝ), d = (30 - c) * (40 / 60)) (h2 : ∀ (t : ℝ), d = (30 + c) * (25 / 60)), 
  c = 90 / 13 := by
  sorry

end speed_of_current_l89_89185


namespace value_of_coat_l89_89026

noncomputable def coat_value := 4.8

theorem value_of_coat 
  (annual_rubles : ℝ) (months_worked : ℝ) (rubles_received : ℝ) (actual_payment : ℝ) (value_of_coat : ℝ) :
  annual_rubles = 12 →
  months_worked = 7 →
  rubles_received = 5 →
  actual_payment = 5 + value_of_coat →
  (7 / 12 * (12 + value_of_coat)) = actual_payment →
  value_of_coat = 4.8 :=
by {
  intros h1 h2 h3 h4 h5,
  assume h5,
  sorry
}

end value_of_coat_l89_89026


namespace value_of_F_l89_89095

   variables (B G P Q F : ℕ)

   -- Define the main hypothesis stating that the total lengths of the books are equal.
   def fill_shelf := 
     (∃ d a : ℕ, d = B * a + 2 * G * a ∧ d = P * a + 2 * Q * a ∧ d = F * a)

   -- Prove that F equals B + 2G and P + 2Q under the hypothesis.
   theorem value_of_F (h : fill_shelf B G P Q F) : F = B + 2 * G ∧ F = P + 2 * Q :=
   sorry
   
end value_of_F_l89_89095


namespace jaylen_bell_peppers_ratio_l89_89710

theorem jaylen_bell_peppers_ratio :
  ∃ j_bell_p, ∃ k_bell_p, ∃ j_green_b, ∃ k_green_b, ∃ j_carrots, ∃ j_cucumbers, ∃ j_total_veg,
  j_carrots = 5 ∧
  j_cucumbers = 2 ∧
  k_bell_p = 2 ∧
  k_green_b = 20 ∧
  j_green_b = 20 / 2 - 3 ∧
  j_total_veg = 18 ∧
  j_carrots + j_cucumbers + j_green_b + j_bell_p = j_total_veg ∧
  j_bell_p / k_bell_p = 2 :=
sorry

end jaylen_bell_peppers_ratio_l89_89710


namespace small_pump_filling_time_l89_89031

theorem small_pump_filling_time :
  ∃ S : ℝ, (L = 2) → 
         (1 / 0.4444444444444444 = S + L) → 
         (1 / S = 4) :=
by 
  sorry

end small_pump_filling_time_l89_89031


namespace inverse_100_mod_101_l89_89051

theorem inverse_100_mod_101 : (100 * 100) % 101 = 1 :=
by
  -- Proof can be provided here.
  sorry

end inverse_100_mod_101_l89_89051


namespace tingting_solution_correct_l89_89461

noncomputable def product_of_square_roots : ℝ :=
  (Real.sqrt 8) * (Real.sqrt 18)

theorem tingting_solution_correct : product_of_square_roots = 12 := by
  sorry

end tingting_solution_correct_l89_89461


namespace max_extra_packages_l89_89719

/-- Max's delivery performance --/
def max_daily_packages : Nat := 35

/-- (1) Max delivered the maximum number of packages on two days --/
def max_2_days : Nat := 2 * max_daily_packages

/-- (2) On two other days, Max unloaded a total of 50 packages --/
def two_days_50 : Nat := 50

/-- (3) On one day, Max delivered one-seventh of the maximum possible daily performance --/
def one_seventh_day : Nat := max_daily_packages / 7

/-- (4) On the last two days, the sum of packages was four-fifths of the maximum daily performance --/
def last_2_days : Nat := 2 * (4 * max_daily_packages / 5)

/-- (5) Total packages delivered in the week --/
def total_delivered : Nat := max_2_days + two_days_50 + one_seventh_day + last_2_days

/-- (6) Total possible packages in a week if worked at maximum performance --/
def total_possible : Nat := 7 * max_daily_packages

/-- (7) Difference between total possible and total delivered packages --/
def difference : Nat := total_possible - total_delivered

/-- Proof problem: Prove the difference is 64 --/
theorem max_extra_packages : difference = 64 := by
  sorry

end max_extra_packages_l89_89719


namespace martha_apples_l89_89269

theorem martha_apples (martha_initial_apples : ℕ) (jane_apples : ℕ) 
  (james_additional_apples : ℕ) (target_martha_apples : ℕ) :
  martha_initial_apples = 20 →
  jane_apples = 5 →
  james_additional_apples = 2 →
  target_martha_apples = 4 →
  (let james_apples := jane_apples + james_additional_apples in
   let martha_remaining_apples := martha_initial_apples - jane_apples - james_apples in
   martha_remaining_apples - target_martha_apples = 4) :=
begin
  sorry
end

end martha_apples_l89_89269


namespace math_problem_l89_89166

theorem math_problem :
  (∃ n : ℕ, 28 = 4 * n) ∧
  ((∃ n1 : ℕ, 361 = 19 * n1) ∧ ¬(∃ n2 : ℕ, 63 = 19 * n2)) ∧
  (¬((∃ n3 : ℕ, 90 = 30 * n3) ∧ ¬(∃ n4 : ℕ, 65 = 30 * n4))) ∧
  ((∃ n5 : ℕ, 45 = 15 * n5) ∧ (∃ n6 : ℕ, 30 = 15 * n6)) ∧
  (∃ n7 : ℕ, 144 = 12 * n7) :=
by {
  -- We need to prove each condition to be true and then prove the statements A, B, D, E are true.
  sorry
}

end math_problem_l89_89166


namespace value_of_g_at_2_l89_89713

def g (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

theorem value_of_g_at_2 : g 2 = 11 := 
by
  sorry

end value_of_g_at_2_l89_89713


namespace sum_of_three_integers_l89_89453

theorem sum_of_three_integers (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) (h7 : a * b * c = 5^3) : a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_l89_89453


namespace a2_a8_sum_l89_89835

variable {a : ℕ → ℝ}  -- Define the arithmetic sequence a

-- Conditions:
axiom arithmetic_sequence (n : ℕ) : a (n + 1) - a n = a 1 - a 0
axiom a1_a9_sum : a 1 + a 9 = 8

-- Theorem stating the question and the answer
theorem a2_a8_sum : a 2 + a 8 = 8 :=
by
  sorry

end a2_a8_sum_l89_89835


namespace value_of_m_l89_89251

theorem value_of_m (m : ℤ) : 
  (∃ f : ℤ → ℤ, ∀ x : ℤ, x^2 - (m+1)*x + 1 = (f x)^2) → (m = 1 ∨ m = -3) := 
by
  sorry

end value_of_m_l89_89251


namespace point_B_is_south_of_A_total_distance_traveled_fuel_consumed_l89_89506

-- Define the travel records and the fuel consumption rate
def travel_records : List Int := [18, -9, 7, -14, -6, 13, -6, -8]
def fuel_consumption_rate : Float := 0.4

-- Question 1: Proof that point B is 5 km south of point A
theorem point_B_is_south_of_A : (travel_records.sum = -5) :=
  by sorry

-- Question 2: Proof that total distance traveled is 81 km
theorem total_distance_traveled : (travel_records.map Int.natAbs).sum = 81 :=
  by sorry

-- Question 3: Proof that the fuel consumed is 32 liters (Rounded)
theorem fuel_consumed : Float.floor (81 * fuel_consumption_rate) = 32 :=
  by sorry

end point_B_is_south_of_A_total_distance_traveled_fuel_consumed_l89_89506


namespace Daniella_savings_l89_89936

def initial_savings_of_Daniella (D : ℤ) := D
def initial_savings_of_Ariella (D : ℤ) := D + 200
def interest_rate : ℚ := 0.10
def time_years : ℚ := 2
def total_amount_after_two_years (initial_amount : ℤ) : ℚ :=
  initial_amount + initial_amount * interest_rate * time_years
def final_amount_of_Ariella : ℚ := 720

theorem Daniella_savings :
  ∃ D : ℤ, total_amount_after_two_years (initial_savings_of_Ariella D) = final_amount_of_Ariella ∧ initial_savings_of_Daniella D = 400 :=
by
  sorry

end Daniella_savings_l89_89936


namespace tan_2016_l89_89521

-- Define the given condition
def sin_36 (a : ℝ) : Prop := Real.sin (36 * Real.pi / 180) = a

-- Prove the required statement given the condition
theorem tan_2016 (a : ℝ) (h : sin_36 a) : Real.tan (2016 * Real.pi / 180) = a / Real.sqrt (1 - a^2) :=
sorry

end tan_2016_l89_89521


namespace trapezoid_area_eq_c_l89_89338

theorem trapezoid_area_eq_c (b c : ℝ) (hb : b = Real.sqrt c) (hc : 0 < c) :
    let shorter_base := b - 3
    let altitude := b
    let longer_base := b + 3
    let K := (1/2) * (shorter_base + longer_base) * altitude
    K = c :=
by
    sorry

end trapezoid_area_eq_c_l89_89338


namespace total_watermelons_l89_89709

def watermelons_grown_by_jason : ℕ := 37
def watermelons_grown_by_sandy : ℕ := 11

theorem total_watermelons : watermelons_grown_by_jason + watermelons_grown_by_sandy = 48 := by
  sorry

end total_watermelons_l89_89709


namespace find_complex_Z_l89_89241

open Complex

theorem find_complex_Z (Z : ℂ) (h : (2 + 4 * I) / Z = 1 - I) : 
  Z = -1 + 3 * I :=
by
  sorry

end find_complex_Z_l89_89241


namespace recurring_decimal_fraction_l89_89220

theorem recurring_decimal_fraction :
  let a := 0.714714714...
  let b := 2.857857857...
  (a / b) = (119 / 476) :=
by
  let a := (714 / (999 : ℝ))
  let b := (2856 / (999 : ℝ))
  sorry

end recurring_decimal_fraction_l89_89220


namespace red_balloon_count_l89_89938

theorem red_balloon_count (total_balloons : ℕ) (green_balloons : ℕ) (red_balloons : ℕ) :
  total_balloons = 17 →
  green_balloons = 9 →
  red_balloons = total_balloons - green_balloons →
  red_balloons = 8 := by
  sorry

end red_balloon_count_l89_89938


namespace estimatedSurvivalProbability_l89_89573

-- Definitions specific to the problem
def numYoungTreesTransplanted : ℕ := 20000
def numYoungTreesSurvived : ℕ := 18044

def survivalRate : ℝ := numYoungTreesSurvived / numYoungTreesTransplanted

theorem estimatedSurvivalProbability :
  Real.round (survivalRate * 10) / 10 = 0.9 :=
by
  sorry

end estimatedSurvivalProbability_l89_89573


namespace cubic_polynomial_roots_l89_89428

noncomputable def polynomial := fun x : ℝ => x^3 - 2*x - 2

theorem cubic_polynomial_roots
  (x y z : ℝ) 
  (h1: polynomial x = 0)
  (h2: polynomial y = 0)
  (h3: polynomial z = 0):
  x * (y - z)^2 + y * (z - x)^2 + z * (x - y)^2 = 0 :=
by
  -- Solution steps will be filled here to prove the theorem
  sorry

end cubic_polynomial_roots_l89_89428


namespace find_a1_l89_89680

variable (a : ℕ → ℕ)
variable (q : ℕ)
variable (h_q_pos : 0 < q)
variable (h_a2a6 : a 2 * a 6 = 8 * a 4)
variable (h_a2 : a 2 = 2)

theorem find_a1 :
  a 1 = 1 :=
by
  sorry

end find_a1_l89_89680


namespace polygon_sides_given_ratio_l89_89393

theorem polygon_sides_given_ratio (n : ℕ) 
  (h : (n - 2) * 180 / 360 = 9 / 2) : n = 11 :=
sorry

end polygon_sides_given_ratio_l89_89393


namespace not_support_either_l89_89948

theorem not_support_either (total_attendance supporters_first supporters_second : ℕ) 
  (h1 : total_attendance = 50) 
  (h2 : supporters_first = 50 * 40 / 100) 
  (h3 : supporters_second = 50 * 34 / 100) : 
  total_attendance - (supporters_first + supporters_second) = 13 :=
by
  sorry

end not_support_either_l89_89948


namespace equal_even_odd_probability_l89_89951

theorem equal_even_odd_probability : 
  let total_dice := 8
  let even_odd_combinations := Nat.choose total_dice 4
  let single_arrangement_probability := (1 / 2) ^ total_dice
  even_odd_combinations * single_arrangement_probability = 35 / 128 := by
  sorry

end equal_even_odd_probability_l89_89951


namespace interest_years_calculation_l89_89191

theorem interest_years_calculation 
  (total_sum : ℝ)
  (second_sum : ℝ)
  (interest_rate_first : ℝ)
  (interest_rate_second : ℝ)
  (time_second : ℝ)
  (interest_second : ℝ)
  (x : ℝ)
  (y : ℝ)
  (h1 : total_sum = 2795)
  (h2 : second_sum = 1720)
  (h3 : interest_rate_first = 3)
  (h4 : interest_rate_second = 5)
  (h5 : time_second = 3)
  (h6 : interest_second = (second_sum * interest_rate_second * time_second) / 100)
  (h7 : interest_second = 258)
  (h8 : x = (total_sum - second_sum))
  (h9 : (interest_rate_first * x * y) / 100 = interest_second)
  : y = 8 := sorry

end interest_years_calculation_l89_89191


namespace no_rational_roots_l89_89424

theorem no_rational_roots (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = p) (h2 : Prime p) (h3: Nat.digits 10 p = [a, b, c, d]) : 
  ¬ ∃ x : ℚ, a * x^3 + b * x^2 + c * x + d = 0 :=
by
  sorry

end no_rational_roots_l89_89424


namespace find_range_of_m_l89_89547

-- Statements of the conditions given in the problem
axiom positive_real_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : (1 / x + 4 / y = 1)

-- Main statement of the proof problem
theorem find_range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 / x + 4 / y = 1) :
  (∃ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ (1 / x + 4 / y = 1) ∧ (x + y / 4 < m^2 - 3 * m)) ↔ (m < -1 ∨ m > 4) := 
sorry

end find_range_of_m_l89_89547


namespace find_stream_speed_l89_89030

variable (r w : ℝ)

noncomputable def stream_speed:
    Prop := 
    (21 / (r + w) + 4 = 21 / (r - w)) ∧ 
    (21 / (3 * r + w) + 0.5 = 21 / (3 * r - w)) ∧ 
    w = 3 

theorem find_stream_speed : ∃ w, stream_speed r w := 
by
  sorry

end find_stream_speed_l89_89030


namespace tom_teaching_years_l89_89605

theorem tom_teaching_years :
  ∃ T D : ℕ, T + D = 70 ∧ D = (1 / 2) * T - 5 ∧ T = 50 :=
by
  sorry

end tom_teaching_years_l89_89605


namespace smallest_x_abs_eq_15_l89_89377

theorem smallest_x_abs_eq_15 :
  ∃ x : ℝ, (|x - 8| = 15) ∧ ∀ y : ℝ, (|y - 8| = 15) → y ≥ x :=
sorry

end smallest_x_abs_eq_15_l89_89377


namespace total_units_is_34_l89_89872

-- Define the number of units on the first floor
def first_floor_units : Nat := 2

-- Define the number of units on the remaining floors (each floor) and number of such floors
def other_floors_units : Nat := 5
def number_of_other_floors : Nat := 3

-- Define the total number of floors per building
def total_floors : Nat := 4

-- Calculate the total units in one building
def units_in_one_building : Nat := first_floor_units + other_floors_units * number_of_other_floors

-- The number of buildings
def number_of_buildings : Nat := 2

-- Calculate the total number of units in both buildings
def total_units : Nat := units_in_one_building * number_of_buildings

-- Prove the total units is 34
theorem total_units_is_34 : total_units = 34 := by
  sorry

end total_units_is_34_l89_89872


namespace problem_equivalent_l89_89458

theorem problem_equivalent (a c : ℕ) (h : (3 * 100 + a * 10 + 7) + 214 = 5 * 100 + c * 10 + 1) (h5c1_div3 : (5 + c + 1) % 3 = 0) : a + c = 4 :=
sorry

end problem_equivalent_l89_89458


namespace parallel_lines_slope_l89_89315

theorem parallel_lines_slope (b : ℝ) 
  (h₁ : ∀ x y : ℝ, 3 * y - 3 * b = 9 * x → (b = 3 - 9)) 
  (h₂ : ∀ x y : ℝ, y + 2 = (b + 9) * x → (b = 3 - 9)) : b = -6 :=
by
  sorry

end parallel_lines_slope_l89_89315


namespace rational_cubes_rational_values_l89_89562

theorem rational_cubes_rational_values {a b : ℝ} (ha : 0 < a) (hb : 0 < b) 
  (hab : a + b = 1) (ha3 : ∃ r : ℚ, a^3 = r) (hb3 : ∃ s : ℚ, b^3 = s) : 
  ∃ r s : ℚ, a = r ∧ b = s :=
sorry

end rational_cubes_rational_values_l89_89562


namespace eggs_needed_per_month_l89_89272

def saly_needs : ℕ := 10
def ben_needs : ℕ := 14
def ked_needs : ℕ := ben_needs / 2
def weeks_in_month : ℕ := 4

def total_weekly_need : ℕ := saly_needs + ben_needs + ked_needs
def total_monthly_need : ℕ := total_weekly_need * weeks_in_month

theorem eggs_needed_per_month : total_monthly_need = 124 := by
  sorry

end eggs_needed_per_month_l89_89272


namespace g_at_3_l89_89895

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_at_3 (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 2) : g 3 = 0 := by
  sorry

end g_at_3_l89_89895


namespace length_of_side_l89_89856

theorem length_of_side {r : ℝ} (h1 : r = 8) (h2 : ∀ (A B C : ℝ), is_right_triangle A B C) :
  ∃ (AB : ℝ), AB = 16 + 16 * Real.sqrt 3 :=
by
  use 16 + 16 * Real.sqrt 3
  sorry

end length_of_side_l89_89856


namespace range_of_k_l89_89517

theorem range_of_k (k : ℝ) :
  (∀ (x1 : ℝ), x1 ∈ Set.Icc (-1 : ℝ) 3 →
    ∃ (x0 : ℝ), x0 ∈ Set.Icc (-1 : ℝ) 3 ∧ (2 * x1^2 + x1 - k) ≤ (x0^3 - 3 * x0)) →
  k ≥ 3 :=
by
  -- This is the place for the proof. 'sorry' is used to indicate that the proof is omitted.
  sorry

end range_of_k_l89_89517


namespace shared_total_l89_89351

theorem shared_total (total_amount : ℝ) (maggie_share : ℝ) (debby_percentage : ℝ)
  (h1 : debby_percentage = 0.25)
  (h2 : maggie_share = 4500)
  (h3 : maggie_share = (1 - debby_percentage) * total_amount) :
  total_amount = 6000 :=
by
  sorry

end shared_total_l89_89351


namespace guitar_price_proof_l89_89651

def total_guitar_price (x : ℝ) : Prop :=
  0.20 * x = 240 → x = 1200

theorem guitar_price_proof (x : ℝ) (h : 0.20 * x = 240) : x = 1200 :=
by
  sorry

end guitar_price_proof_l89_89651


namespace triangle_inequality_l89_89339

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
sorry

end triangle_inequality_l89_89339


namespace solve_for_x_l89_89738

theorem solve_for_x (x : ℝ) (h : (2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4)) : x = -14 := 
by 
  sorry

end solve_for_x_l89_89738


namespace exists_permutation_satisfies_eq_l89_89321

theorem exists_permutation_satisfies_eq : 
  ∃ (a b c d e f : ℕ), 
    {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6} ∧ 
    (a-1)*(b-2)*(c-3)*(d-4)*(e-5)*(f-6) = 75 := 
by
  sorry

end exists_permutation_satisfies_eq_l89_89321


namespace trigonometric_identities_l89_89833

noncomputable def tan (θ : ℝ) : ℝ := Real.tan θ
noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ
noncomputable def cos (θ : ℝ) : ℝ := Real.cos θ

theorem trigonometric_identities (θ : ℝ) (h_tan : tan θ = 2) (h_identity : sin θ ^ 2 + cos θ ^ 2 = 1) :
    ((sin θ = 2 * Real.sqrt 5 / 5 ∧ cos θ = Real.sqrt 5 / 5) ∨ (sin θ = -2 * Real.sqrt 5 / 5 ∧ cos θ = -Real.sqrt 5 / 5)) ∧
    ((4 * sin θ - 3 * cos θ) / (6 * cos θ + 2 * sin θ) = 1 / 2) :=
by
  sorry

end trigonometric_identities_l89_89833


namespace simplify_expression_correct_l89_89581

noncomputable def simplify_expression (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : ℝ :=
  let expr1 := (a^2 - b^2) / (a^2 + 2 * a * b + b^2)
  let expr2 := (2 : ℝ) / (a * b)
  let expr3 := ((1 : ℝ) / a + (1 : ℝ) / b)^2
  let expr4 := (2 : ℝ) / (a^2 - b^2 + 2 * a * b)
  expr1 + expr2 / expr3 * expr4

theorem simplify_expression_correct (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  simplify_expression a b h = 2 / (a + b)^2 := by
  sorry

end simplify_expression_correct_l89_89581


namespace largest_number_is_correct_l89_89759

theorem largest_number_is_correct (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 3) : c = 33.25 :=
by
  sorry

end largest_number_is_correct_l89_89759


namespace same_grade_percentage_is_correct_l89_89550

def total_students : ℕ := 40

def grade_distribution : ℕ × ℕ × ℕ × ℕ :=
  (17, 40, 100)

def same_grade_percentage (total_students : ℕ) (same_grade_students : ℕ) : ℚ :=
  (same_grade_students / total_students) * 100

theorem same_grade_percentage_is_correct :
  let same_grade_students := 3 + 5 + 6 + 3
  same_grade_percentage total_students same_grade_students = 42.5 :=
by 
let same_grade_students := 3 + 5 + 6 + 3
show same_grade_percentage total_students same_grade_students = 42.5
sorry

end same_grade_percentage_is_correct_l89_89550


namespace functional_equation_identity_l89_89670

def f : ℝ → ℝ := sorry

theorem functional_equation_identity (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x * f y + y) = f (f (x * y)) + y) : 
  ∀ y : ℝ, f y = y :=
sorry

end functional_equation_identity_l89_89670


namespace segments_interior_proof_l89_89177

noncomputable def count_internal_segments (squares hexagons octagons : Nat) : Nat := 
  let vertices := (squares * 4 + hexagons * 6 + octagons * 8) / 3
  let total_segments := (vertices * (vertices - 1)) / 2
  let edges_along_faces := 3 * vertices
  (total_segments - edges_along_faces) / 2

theorem segments_interior_proof : count_internal_segments 12 8 6 = 840 := 
  by sorry

end segments_interior_proof_l89_89177


namespace remainder_division_l89_89314

-- Definition of the number in terms of its components
def num : ℤ := 98 * 10^6 + 76 * 10^4 + 54 * 10^2 + 32

-- The modulus
def m : ℤ := 25

-- The given problem restated as a hypothesis and goal
theorem remainder_division : num % m = 7 :=
by
  sorry

end remainder_division_l89_89314


namespace center_of_circle_polar_coords_l89_89861

theorem center_of_circle_polar_coords :
  ∀ (θ : ℝ), ∃ (ρ : ℝ), (ρ, θ) = (2, Real.pi) ∧ ρ = - 4 * Real.cos θ := 
sorry

end center_of_circle_polar_coords_l89_89861


namespace slope_angle_bisector_l89_89210

open Real

theorem slope_angle_bisector (m1 m2 : ℝ) (h1 : m1 = 2) (h2 : m2 = -3) : 
    let k := (m1 + m2 + sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2) in
    m1 = 2 ∧ m2 = -3 → k = (-1 + sqrt 14) / 7 := 
by
  intros
  simp [h1, h2]
  sorry

end slope_angle_bisector_l89_89210


namespace total_money_shared_l89_89354

theorem total_money_shared (T : ℝ) (h : 0.75 * T = 4500) : T = 6000 :=
by
  sorry

end total_money_shared_l89_89354


namespace marble_count_l89_89094

theorem marble_count (r g b : ℕ) (h1 : g + b = 6) (h2 : r + b = 8) (h3 : r + g = 4) : r + g + b = 9 :=
sorry

end marble_count_l89_89094


namespace length_of_bridge_l89_89016

theorem length_of_bridge (length_of_train : ℕ) (speed_of_train_kmh : ℕ) (time_to_cross : ℕ)
  (lt : length_of_train = 140)
  (st : speed_of_train_kmh = 45)
  (tc : time_to_cross = 30) : 
  ∃ length_of_bridge, length_of_bridge = 235 := 
by 
  sorry

end length_of_bridge_l89_89016


namespace ratio_t_q_l89_89535

theorem ratio_t_q (q r s t : ℚ) (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) : 
  t / q = 3 / 2 :=
by
  sorry

end ratio_t_q_l89_89535


namespace problem1_problem2_l89_89042

-- Problem 1
theorem problem1 (a b : ℝ) : 
  a^2 * (2 * a * b - 1) + (a - 3 * b) * (a + b) = 2 * a^3 * b - 2 * a * b - 3 * b^2 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (2 * x - 3)^2 - (x + 2)^2 = 3 * x^2 - 16 * x + 5 :=
by sorry

end problem1_problem2_l89_89042


namespace math_problem_l89_89201

theorem math_problem :
  |(-3 : ℝ)| - Real.sqrt 8 - (1/2 : ℝ)⁻¹ + 2 * Real.cos (Real.pi / 4) = 1 - Real.sqrt 2 :=
by
  sorry

end math_problem_l89_89201


namespace imaginary_part_of_z_l89_89242

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : z = 2 / (-1 + I)) : z.im = -1 := 
by
  sorry

end imaginary_part_of_z_l89_89242


namespace ram_efficiency_eq_27_l89_89725

theorem ram_efficiency_eq_27 (R : ℕ) (h1 : ∀ Krish, 2 * (1 / (R : ℝ)) = 1 / Krish) 
  (h2 : ∀ s, 3 * (1 / (R : ℝ)) * s = 1 ↔ s = (9 : ℝ)) : R = 27 :=
sorry

end ram_efficiency_eq_27_l89_89725


namespace mohan_least_cookies_l89_89871

theorem mohan_least_cookies :
  ∃ b : ℕ, 
    b % 6 = 5 ∧
    b % 8 = 3 ∧
    b % 9 = 6 ∧
    b = 59 :=
by
  sorry

end mohan_least_cookies_l89_89871


namespace num_distinct_log_values_l89_89989

-- Defining the set of numbers
def number_set : Set ℕ := {1, 2, 3, 4, 6, 9}

-- Define a function to count distinct logarithmic values
noncomputable def distinct_log_values (s : Set ℕ) : ℕ := 
  -- skipped, assume the implementation is done correctly
  sorry 

theorem num_distinct_log_values : distinct_log_values number_set = 17 :=
by
  sorry

end num_distinct_log_values_l89_89989


namespace conclusion_1_conclusion_2_conclusion_3_conclusion_4_l89_89830

variable (a : ℕ → ℝ)

-- Conditions
def sequence_positive : Prop :=
  ∀ n, a n > 0

def recurrence_relation : Prop :=
  ∀ n, a (n + 1) ^ 2 - a (n + 1) = a n

-- Correct conclusions to prove:

-- Conclusion ①
theorem conclusion_1 (h1 : sequence_positive a) (h2 : recurrence_relation a) :
  ∀ n ≥ 2, a n > 1 := 
sorry

-- Conclusion ②
theorem conclusion_2 (h1 : sequence_positive a) (h2 : recurrence_relation a) :
  ¬∀ n, a n = a (n + 1) := 
sorry

-- Conclusion ③
theorem conclusion_3 (h1 : sequence_positive a) (h2 : recurrence_relation a) (h3 : 0 < a 1 ∧ a 1 < 2) :
  ∀ n, a (n + 1) > a n :=
sorry

-- Conclusion ④
theorem conclusion_4 (h1 : sequence_positive a) (h2 : recurrence_relation a) (h4 : a 1 > 2) :
  ∀ n ≥ 2, 2 < a n ∧ a n < a 1 :=
sorry

end conclusion_1_conclusion_2_conclusion_3_conclusion_4_l89_89830


namespace quadratic_solution_l89_89134

theorem quadratic_solution (x : ℝ) (h : x^2 - 4 * x + 2 = 0) : x + 2 / x = 4 :=
by sorry

end quadratic_solution_l89_89134


namespace infinite_solutions_xyz_l89_89571

theorem infinite_solutions_xyz : ∀ k : ℕ, 
  (∃ n : ℕ, n > k ∧ ∃ x y z : ℕ, x^2 + y^2 + z^2 - x*y*z + 10 = 0 ∧ x > 2008 ∧ y > 2008 ∧ z > 2008) →
  ∃ x y z : ℕ, x^2 + y^2 + z^2 - x*y*z + 10 = 0 ∧ x > 2008 ∧ y > 2008 ∧ z > 2008 := 
sorry

end infinite_solutions_xyz_l89_89571


namespace class_groups_l89_89854

open Nat

def combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem class_groups (boys girls : ℕ) (group_size : ℕ) :
  boys = 9 → girls = 12 → group_size = 3 →
  (combinations boys 1 * combinations girls 2) + (combinations boys 2 * combinations girls 1) = 1026 :=
by
  intros
  sorry

end class_groups_l89_89854


namespace probability_even_sum_l89_89309

-- Definitions of probabilities for the first wheel
def P_even1 : ℚ := 1 / 2
def P_odd1 : ℚ := 1 / 2

-- Definitions of probabilities for the second wheel
def P_even2 : ℚ := 1 / 5
def P_odd2 : ℚ := 4 / 5

-- Probability that the sum of numbers from both wheels is even
def P_even_sum : ℚ := P_even1 * P_even2 + P_odd1 * P_odd2

-- Theorem statement
theorem probability_even_sum : P_even_sum = 1 / 2 :=
by {
  sorry -- The proof is not required
}

end probability_even_sum_l89_89309


namespace outer_boundary_diameter_l89_89022

-- Define the given conditions
def fountain_diameter : ℝ := 12
def walking_path_width : ℝ := 6
def garden_ring_width : ℝ := 10

-- Define what we need to prove
theorem outer_boundary_diameter :
  2 * (fountain_diameter / 2 + garden_ring_width + walking_path_width) = 44 :=
by
  sorry

end outer_boundary_diameter_l89_89022


namespace function_range_is_interval_l89_89067

theorem function_range_is_interval :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ (256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x) ∧ 
  (256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x) ≤ 1 := 
by
  sorry

end function_range_is_interval_l89_89067


namespace plus_one_eq_next_plus_l89_89378

theorem plus_one_eq_next_plus (m : ℕ) (h : m > 1) : (m^2 + m) + 1 = ((m + 1)^2 + (m + 1)) := by
  sorry

end plus_one_eq_next_plus_l89_89378


namespace ones_digit_of_prime_in_arithmetic_sequence_is_one_l89_89383

theorem ones_digit_of_prime_in_arithmetic_sequence_is_one 
  (p q r s : ℕ) 
  (hp : Prime p) 
  (hq : Prime q) 
  (hr : Prime r) 
  (hs : Prime s) 
  (h₁ : p > 10) 
  (h₂ : q = p + 10) 
  (h₃ : r = q + 10) 
  (h₄ : s = r + 10) 
  (h₅ : s > r) 
  (h₆ : r > q) 
  (h₇ : q > p) : 
  p % 10 = 1 :=
sorry

end ones_digit_of_prime_in_arithmetic_sequence_is_one_l89_89383


namespace tom_teaching_years_l89_89600

def years_tom_has_been_teaching (x : ℝ) : Prop :=
  x + (1/2 * x - 5) = 70

theorem tom_teaching_years:
  ∃ x : ℝ, years_tom_has_been_teaching x ∧ x = 50 :=
by
  sorry

end tom_teaching_years_l89_89600


namespace triangle_area_PQR_l89_89129

section TriangleArea

variables {a b c d : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
variables (hOppositeSides : (0 - c) * b - (a - 0) * d < 0)

theorem triangle_area_PQR :
  let P := (0, a)
  let Q := (b, 0)
  let R := (c, d)
  let area := (1 / 2) * (a * c + b * d - a * b)
  area = (1 / 2) * (a * c + b * d - a * b) := 
by
  sorry

end TriangleArea

end triangle_area_PQR_l89_89129


namespace machine_work_rate_l89_89306

theorem machine_work_rate (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -6 ∧ x ≠ -1) : 
  (1 / (x + 6) + 1 / (x + 1) + 1 / (2 * x) = 1 / x) → x = 2 / 3 :=
by
  sorry

end machine_work_rate_l89_89306


namespace expression_simplification_l89_89661

theorem expression_simplification :
  (2 + 3) * (2^3 + 3^3) * (2^9 + 3^9) * (2^27 + 3^27) = 3^41 - 2^41 := 
sorry

end expression_simplification_l89_89661


namespace find_number_l89_89186

theorem find_number (n : ℝ) (h : n / 0.06 = 16.666666666666668) : n = 1 :=
by
  sorry

end find_number_l89_89186


namespace time_to_see_each_other_again_l89_89711

variable (t : ℝ) (t_frac : ℚ)
variable (kenny_speed jenny_speed : ℝ)
variable (kenny_initial jenny_initial : ℝ)
variable (building_side distance_between_paths : ℝ)

def kenny_position (t : ℝ) : ℝ := kenny_initial + kenny_speed * t
def jenny_position (t : ℝ) : ℝ := jenny_initial + jenny_speed * t

theorem time_to_see_each_other_again
  (kenny_speed_eq : kenny_speed = 4)
  (jenny_speed_eq : jenny_speed = 2)
  (kenny_initial_eq : kenny_initial = -50)
  (jenny_initial_eq : jenny_initial = -50)
  (building_side_eq : building_side = 100)
  (distance_between_paths_eq : distance_between_paths = 300)
  (t_gt_50 : t > 50)
  (t_frac_eq : t_frac = 50) :
  (t == t_frac) :=
  sorry

end time_to_see_each_other_again_l89_89711


namespace sum_of_seven_consecutive_integers_l89_89887

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
  sorry

end sum_of_seven_consecutive_integers_l89_89887


namespace units_digit_of_expression_l89_89046

theorem units_digit_of_expression :
  let A := 12 + Real.sqrt 245 in
  (A ^ 21 + A ^ 84) % 10 = 6 :=
by
  sorry

end units_digit_of_expression_l89_89046


namespace find_principal_l89_89637

-- Definitions based on conditions
def simple_interest (P R T : ℚ) : ℚ := (P * R * T) / 100

-- Given conditions
def SI : ℚ := 6016.75
def R : ℚ := 8
def T : ℚ := 5

-- Stating the proof problem
theorem find_principal : 
  ∃ P : ℚ, simple_interest P R T = SI ∧ P = 15041.875 :=
by {
  sorry
}

end find_principal_l89_89637


namespace height_of_brick_l89_89627

-- Definitions of wall dimensions
def L_w : ℝ := 700
def W_w : ℝ := 600
def H_w : ℝ := 22.5

-- Number of bricks
def n : ℝ := 5600

-- Definitions of brick dimensions (length and width)
def L_b : ℝ := 25
def W_b : ℝ := 11.25

-- Main theorem: Prove the height of each brick
theorem height_of_brick : ∃ h : ℝ, h = 6 :=
by
  -- Will add the proof steps here eventually
  sorry

end height_of_brick_l89_89627


namespace quadrilateral_area_l89_89386

noncomputable def area_of_quadrilateral (a : ℝ) : ℝ :=
  let sqrt3 := Real.sqrt 3
  let num := a^2 * (9 - 5 * sqrt3)
  let denom := 12
  num / denom

theorem quadrilateral_area (a : ℝ) : area_of_quadrilateral a = (a^2 * (9 - 5 * Real.sqrt 3)) / 12 := by
  sorry

end quadrilateral_area_l89_89386


namespace product_of_real_roots_of_equation_l89_89666

theorem product_of_real_roots_of_equation : 
  ∀ x : ℝ, (x^4 + (x - 4)^4 = 32) → x = 2 :=
sorry

end product_of_real_roots_of_equation_l89_89666


namespace find_a1_plus_b1_l89_89334

noncomputable def series (n : ℕ) : ℝ × ℝ :=
nat.rec_on n (a₁, b₁)
(λ n rec, (2 * real.sqrt 3 * rec.1 + rec.2, real.sqrt 3 * rec.2 - 2 * rec.1))

theorem find_a1_plus_b1 (a₁ b₁ : ℝ) 
(h1 : series 150 = (-1, 3)) : 
  a₁ + b₁ = - (1 / 4^149) := 
by sorry

#eval find_a1_plus_b1 (-1 / 4^149) -- Example to evaluate the theorem

end find_a1_plus_b1_l89_89334


namespace min_value_of_fraction_l89_89979

theorem min_value_of_fraction (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 2 * m + n = 1) : 
  (1 / m + 2 / n) = 8 := 
by 
  sorry

end min_value_of_fraction_l89_89979


namespace geometric_mean_eq_6_l89_89240

theorem geometric_mean_eq_6 (b c : ℝ) (hb : b = 3) (hc : c = 12) :
  (b * c) ^ (1/2 : ℝ) = 6 := 
by
  sorry

end geometric_mean_eq_6_l89_89240


namespace negation_P_l89_89133

-- Define the condition that x is a real number
variable (x : ℝ)

-- Define the proposition P
def P := ∀ (x : ℝ), x ≥ 2

-- Define the negation of P
def not_P := ∃ (x : ℝ), x < 2

-- Theorem stating the equivalence of the negation of P
theorem negation_P : ¬P ↔ not_P := by
  sorry

end negation_P_l89_89133


namespace least_possible_perimeter_l89_89750

/-- Proof that the least possible perimeter of a triangle with two sides of length 24 and 51 units,
    and the third side being an integer, is 103 units. -/
theorem least_possible_perimeter (a b : ℕ) (c : ℕ) (h1 : a = 24) (h2 : b = 51) (h3 : c > 27) (h4 : c < 75) :
    a + b + c = 103 :=
by
  sorry

end least_possible_perimeter_l89_89750


namespace pencils_across_diameter_l89_89405

theorem pencils_across_diameter (r : ℝ) (pencil_length_inch : ℕ) (pencils : ℕ) :
  r = 14 ∧ pencil_length_inch = 6 ∧ pencils = 56 → 
  let d := 2 * r in
  let pencil_length_feet := pencil_length_inch / 12 in
  pencils = d / pencil_length_feet :=
begin
  sorry -- Proof is skipped
end

end pencils_across_diameter_l89_89405


namespace total_apartment_units_l89_89875

-- Define the number of apartment units on different floors
def units_first_floor := 2
def units_other_floors := 5
def num_other_floors := 3
def num_buildings := 2

-- Calculation of total units in one building
def units_one_building := units_first_floor + num_other_floors * units_other_floors

-- Calculation of total units in all buildings
def total_units := num_buildings * units_one_building

-- The theorem to prove
theorem total_apartment_units : total_units = 34 :=
by
  sorry

end total_apartment_units_l89_89875


namespace area_of_woods_l89_89905

def width := 8 -- the width in miles
def length := 3 -- the length in miles
def area (w : Nat) (l : Nat) : Nat := w * l -- the area function for a rectangle

theorem area_of_woods : area width length = 24 := by
  sorry

end area_of_woods_l89_89905


namespace median_length_YN_perimeter_triangle_XYZ_l89_89554

-- Definitions for conditions
noncomputable def length_XY : ℝ := 5
noncomputable def length_XZ : ℝ := 12
noncomputable def is_right_angle_XYZ : Prop := true
noncomputable def midpoint_N : ℝ := length_XZ / 2

-- Theorem statement for the length of the median YN
theorem median_length_YN (XY XZ : ℝ) (right_angle : is_right_angle_XYZ) :
  XY = 5 ∧ XZ = 12 → (XY^2 + XZ^2) = 169 → (13 / 2) = 6.5 := by
  sorry

-- Theorem statement for the perimeter of triangle XYZ
theorem perimeter_triangle_XYZ (XY XZ : ℝ) (right_angle : is_right_angle_XYZ) :
  XY = 5 ∧ XZ = 12 → (XY^2 + XZ^2) = 169 → (XY + XZ + 13) = 30 := by
  sorry

end median_length_YN_perimeter_triangle_XYZ_l89_89554


namespace constant_a_value_l89_89065

theorem constant_a_value (S : ℕ → ℝ)
  (a : ℝ)
  (h : ∀ n : ℕ, S n = 3 ^ (n + 1) + a) :
  a = -3 :=
sorry

end constant_a_value_l89_89065


namespace beneficial_card_l89_89618

theorem beneficial_card
  (P : ℕ) (r_c r_d r_i : ℚ) :
  let credit_income := (r_c * P + r_i * P)
  let debit_income := r_d * P
  P = 8000 ∧ r_c = 0.005 ∧ r_d = 0.0075 ∧ r_i = 0.005 →
  credit_income > debit_income :=
by
  intro h
  cases h with hP hr
  cases hr with hrc hrd_ri
  cases hrd_ri with hrd hri
  rw [hP, hrc, hrd, hri]
  sorry

end beneficial_card_l89_89618


namespace hyperbola_range_m_l89_89586

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (m - 2) ≠ 0 ∧ (m + 3) ≠ 0 ∧ (x^2 / (m - 2) + y^2 / (m + 3) = 1)) ↔ (-3 < m ∧ m < 2) :=
by
  sorry

end hyperbola_range_m_l89_89586


namespace num_ways_to_arrange_PERCEPTION_l89_89362

open Finset

def word := "PERCEPTION"

def num_letters : ℕ := 10

def occurrences : List (Char × ℕ) :=
  [('P', 2), ('E', 2), ('R', 1), ('C', 1), ('E', 2), ('P', 2), ('T', 1), ('I', 2), ('O', 1), ('N', 1)]

def factorial (n : ℕ) : ℕ := List.range n.succ.foldl (· * ·) 1

noncomputable def num_distinct_arrangements (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / ks.foldl (λ acc k => acc * factorial k) 1

theorem num_ways_to_arrange_PERCEPTION :
  num_distinct_arrangements num_letters [2, 2, 2, 1, 1, 1, 1, 1] = 453600 := 
by sorry

end num_ways_to_arrange_PERCEPTION_l89_89362


namespace area_of_polygon_ABCDEF_l89_89679

-- Definitions based on conditions
def AB : ℕ := 8
def BC : ℕ := 10
def DC : ℕ := 5
def FA : ℕ := 7
def GF : ℕ := 3
def ED : ℕ := 7
def height_GF_ED : ℕ := 2

-- Area calculations based on given conditions
def area_ABCG : ℕ := AB * BC
def area_trapezoid_GFED : ℕ := (GF + ED) * height_GF_ED / 2

-- Proof statement
theorem area_of_polygon_ABCDEF :
  area_ABCG - area_trapezoid_GFED = 70 :=
by
  simp [area_ABCG, area_trapezoid_GFED]
  sorry

end area_of_polygon_ABCDEF_l89_89679


namespace trigonometric_expression_evaluation_l89_89366

theorem trigonometric_expression_evaluation :
  (Real.cos (-585 * Real.pi / 180)) / 
  (Real.tan (495 * Real.pi / 180) + Real.sin (-690 * Real.pi / 180)) = Real.sqrt 2 :=
  sorry

end trigonometric_expression_evaluation_l89_89366


namespace factors_of_48_are_multiples_of_6_l89_89693

theorem factors_of_48_are_multiples_of_6:
  let factors := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48] in
  let multiples_of_6 := [6, 12, 24, 48] in
  count (λ x, x ∈ multiples_of_6) factors = 4 :=
by
  -- We first state the factors of 48
  have factors_of_48 : list ℕ := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48],
  -- Let multiples_of_6 be [6, 12, 24, 48]
  have multiples_6 : list ℕ := [6, 12, 24, 48],
  -- Verify the counts and other details later directly
  exact sorry

end factors_of_48_are_multiples_of_6_l89_89693


namespace pencils_across_diameter_l89_89401

def radius_feet : ℝ := 14
def pencil_length_inches : ℝ := 6

theorem pencils_across_diameter : 
  (2 * radius_feet * 12 / pencil_length_inches) = 56 := 
by
  sorry

end pencils_across_diameter_l89_89401


namespace next_performance_together_in_90_days_l89_89809

theorem next_performance_together_in_90_days :
  Nat.lcm (Nat.lcm 5 6) (Nat.lcm 9 10) = 90 := by
  sorry

end next_performance_together_in_90_days_l89_89809


namespace find_c_l89_89417

theorem find_c (c : ℝ) (h : (-c / 4) + (-c / 7) = 22) : c = -56 :=
by
  sorry

end find_c_l89_89417


namespace star_number_of_intersections_2018_25_l89_89206

-- Definitions for the conditions
def rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def star_intersections (n k : ℕ) : ℕ := 
  n * (k - 1)

-- The main theorem
theorem star_number_of_intersections_2018_25 :
  2018 ≥ 5 ∧ 25 < 2018 / 2 ∧ rel_prime 2018 25 → 
  star_intersections 2018 25 = 48432 :=
by
  intros h
  sorry

end star_number_of_intersections_2018_25_l89_89206


namespace journey_ratio_l89_89592

/-- Given a full-circle journey broken into parts,
  including paths through the Zoo Park (Z), the Circus (C), and the Park (P), 
  prove that the journey avoiding the Zoo Park is 11 times shorter. -/
theorem journey_ratio (Z C P : ℝ) (h1 : C = (3 / 4) * Z) 
                      (h2 : P = (1 / 4) * Z) : 
  Z = 11 * P := 
sorry

end journey_ratio_l89_89592


namespace number_of_ears_pierced_l89_89863

-- Definitions for the conditions
def nosePiercingPrice : ℝ := 20
def earPiercingPrice := nosePiercingPrice + 0.5 * nosePiercingPrice
def totalAmountMade : ℝ := 390
def nosesPierced : ℕ := 6
def totalFromNoses := nosesPierced * nosePiercingPrice
def totalFromEars := totalAmountMade - totalFromNoses

-- The proof statement
theorem number_of_ears_pierced : totalFromEars / earPiercingPrice = 9 := by
  sorry

end number_of_ears_pierced_l89_89863


namespace error_percent_in_area_l89_89476

theorem error_percent_in_area 
    (L W : ℝ) 
    (measured_length : ℝ := 1.09 * L) 
    (measured_width : ℝ := 0.92 * W) 
    (correct_area : ℝ := L * W) 
    (incorrect_area : ℝ := measured_length * measured_width) :
    100 * (incorrect_area - correct_area) / correct_area = 0.28 :=
by
  sorry

end error_percent_in_area_l89_89476


namespace find_f_pi_l89_89395

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.tan (ω * x + Real.pi / 3)

theorem find_f_pi (ω : ℝ) (h_positive : ω > 0) (h_period : Real.pi / ω = 3 * Real.pi) :
  f (ω := ω) Real.pi = -Real.sqrt 3 :=
by
  -- ω is given to be 1/3 by the condition h_period, substituting that 
  -- directly might be clearer for stating the problem accurately
  have h_omega : ω = 1 / 3 := by
    sorry
  rw [h_omega]
  sorry


end find_f_pi_l89_89395


namespace range_of_a_l89_89429

variable {Ω : Type*} [MeasurableSpace Ω]

noncomputable def X : Ω → ℕ := sorry -- Define X as a random variable
axiom X_distrib : ∀ i ∈ {1, 2, 3, 4}, ℙ(X = i) = i / 10

theorem range_of_a (a : ℝ) : 
  (ℙ(1 ≤ X ∧ X < a) = 3 / 5) ↔ (3 < a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l89_89429


namespace large_number_divisible_by_12_l89_89647

theorem large_number_divisible_by_12 : (Nat.digits 10 ((71 * 10^168) + (72 * 10^166) + (73 * 10^164) + (74 * 10^162) + (75 * 10^160) + (76 * 10^158) + (77 * 10^156) + (78 * 10^154) + (79 * 10^152) + (80 * 10^150) + (81 * 10^148) + (82 * 10^146) + (83 * 10^144) + 84)).foldl (λ x y => x * 10 + y) 0 % 12 = 0 := 
sorry

end large_number_divisible_by_12_l89_89647


namespace star_property_l89_89823

-- Define the custom operation
def star (a b : ℝ) : ℝ := a^2 - b

-- Define the property to prove
theorem star_property (x y : ℝ) : star (x - y) (x + y) = x^2 - x - 2 * x * y + y^2 - y :=
by sorry

end star_property_l89_89823


namespace right_triangle_sets_l89_89587

theorem right_triangle_sets :
  ∃! (a b c : ℕ), 
    ((a = 5 ∧ b = 12 ∧ c = 13) ∧ a * a + b * b = c * c) ∧ 
    ¬(∃ a b c, (a = 3 ∧ b = 4 ∧ c = 6) ∧ a * a + b * b = c * c) ∧
    ¬(∃ a b c, (a = 4 ∧ b = 5 ∧ c = 6) ∧ a * a + b * b = c * c) ∧
    ¬(∃ a b c, (a = 5 ∧ b = 7 ∧ c = 9) ∧ a * a + b * b = c * c) :=
by {
  --- proof needed
  sorry
}

end right_triangle_sets_l89_89587


namespace dice_even_odd_equal_probability_l89_89953

noncomputable def probability_equal_even_odd_dice : ℚ :=
  let p : ℚ := 1 / 2 in
  let choose_8_4 : ℕ := Nat.choose 8 4 in
  choose_8_4 * (p^8)

theorem dice_even_odd_equal_probability :
  (probability_equal_even_odd_dice = 35 / 128) :=
by
  -- Formal proof goes here
  sorry

end dice_even_odd_equal_probability_l89_89953


namespace exponentiation_rule_l89_89040

theorem exponentiation_rule (m n : ℤ) : (-2 * m^3 * n^2)^2 = 4 * m^6 * n^4 :=
by
  sorry

end exponentiation_rule_l89_89040


namespace polynomial_coeffs_l89_89246

theorem polynomial_coeffs :
  ( ∃ (a1 a2 a3 a4 a5 : ℕ), (∀ (x : ℝ), (x + 1) ^ 3 * (x + 2) ^ 2 = x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5) ∧ a4 = 16 ∧ a5 = 4) := 
by
  sorry

end polynomial_coeffs_l89_89246


namespace determine_digit_l89_89044

theorem determine_digit (Θ : ℚ) (h : 312 / Θ = 40 + 2 * Θ) : Θ = 6 :=
sorry

end determine_digit_l89_89044


namespace sum_of_r_s_l89_89564

theorem sum_of_r_s (m : ℝ) (x : ℝ) (y : ℝ) (r s : ℝ) 
  (parabola_eqn : y = x^2 + 4) 
  (point_Q : (x, y) = (10, 5)) 
  (roots_rs : ∀ (m : ℝ), m^2 - 40*m + 4 = 0 → r < m → m < s)
  : r + s = 40 := 
sorry

end sum_of_r_s_l89_89564


namespace volume_tetrahedron_375sqrt2_l89_89668

noncomputable def tetrahedronVolume (area_ABC : ℝ) (area_BCD : ℝ) (BC : ℝ) (angle_ABC_BCD : ℝ) : ℝ :=
  let h_BCD := (2 * area_BCD) / BC
  let h_D_ABD := h_BCD * Real.sin angle_ABC_BCD
  (1 / 3) * area_ABC * h_D_ABD

theorem volume_tetrahedron_375sqrt2 :
  tetrahedronVolume 150 90 12 (Real.pi / 4) = 375 * Real.sqrt 2 := by
  sorry

end volume_tetrahedron_375sqrt2_l89_89668


namespace prob_equal_even_odd_dice_l89_89957

def even_number_probability : ℚ := 1 / 2
def odd_number_probability : ℚ := 1 / 2

def probability_equal_even_odd (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k) * (even_number_probability) ^ n

theorem prob_equal_even_odd_dice : 
  probability_equal_even_odd 8 4 = 35 / 128 :=
by
  sorry

end prob_equal_even_odd_dice_l89_89957


namespace jasmine_max_stickers_l89_89708

-- Given conditions and data
def sticker_cost : ℝ := 0.75
def jasmine_budget : ℝ := 10.0

-- Proof statement
theorem jasmine_max_stickers : ∃ n : ℕ, (n : ℝ) * sticker_cost ≤ jasmine_budget ∧ (∀ m : ℕ, (m > n) → (m : ℝ) * sticker_cost > jasmine_budget) :=
sorry

end jasmine_max_stickers_l89_89708


namespace recurring_decimal_fraction_l89_89219

theorem recurring_decimal_fraction :
  let a := 0.714714714...
  let b := 2.857857857...
  (a / b) = (119 / 476) :=
by
  let a := (714 / (999 : ℝ))
  let b := (2856 / (999 : ℝ))
  sorry

end recurring_decimal_fraction_l89_89219


namespace playgroup_count_l89_89761

-- Definitions based on the conditions
def total_people (girls boys parents : ℕ) := girls + boys + parents
def playgroups (total size_per_group : ℕ) := total / size_per_group

-- Statement of the problem
theorem playgroup_count (girls boys parents size_per_group : ℕ)
  (h_girls : girls = 14)
  (h_boys : boys = 11)
  (h_parents : parents = 50)
  (h_size_per_group : size_per_group = 25) :
  playgroups (total_people girls boys parents) size_per_group = 3 :=
by {
  -- This is just the statement, the proof is skipped with sorry
  sorry
}

end playgroup_count_l89_89761


namespace boat_speed_still_water_l89_89780

theorem boat_speed_still_water (V_b V_c : ℝ) (h1 : 45 / (V_b - V_c) = t) (h2 : V_b = 12)
(h3 : V_b + V_c = 15):
  V_b = 12 :=
by
  sorry

end boat_speed_still_water_l89_89780


namespace product_of_two_numbers_ratio_l89_89766

theorem product_of_two_numbers_ratio (x y : ℝ)
  (h1 : x - y ≠ 0)
  (h2 : x + y = 4 * (x - y))
  (h3 : x * y = 18 * (x - y)) :
  x * y = 86.4 :=
by
  sorry

end product_of_two_numbers_ratio_l89_89766


namespace repeating_decimal_fraction_l89_89303

theorem repeating_decimal_fraction : (0.363636363636 : ℚ) = 4 / 11 := 
sorry

end repeating_decimal_fraction_l89_89303


namespace car_travel_distance_l89_89325

variable (b t : Real)
variable (h1 : b > 0)
variable (h2 : t > 0)

theorem car_travel_distance (b t : Real) (h1 : b > 0) (h2 : t > 0) :
  let rate := b / 4
  let inches_in_yard := 36
  let time_in_seconds := 5 * 60
  let distance_in_inches := (rate / t) * time_in_seconds
  let distance_in_yards := distance_in_inches / inches_in_yard
  distance_in_yards = (25 * b) / (12 * t) := by
  sorry

end car_travel_distance_l89_89325


namespace range_fraction_l89_89730

theorem range_fraction {x y : ℝ} (h : x^2 + y^2 + 2 * x = 0) :
  ∃ a b : ℝ, a = -1 ∧ b = 1 / 3 ∧ ∀ z, z = (y - x) / (x - 1) → a ≤ z ∧ z ≤ b :=
by 
  sorry

end range_fraction_l89_89730


namespace pencils_in_stock_at_end_of_week_l89_89723

def pencils_per_day : ℕ := 100
def days_per_week : ℕ := 5
def initial_pencils : ℕ := 80
def sold_pencils : ℕ := 350

theorem pencils_in_stock_at_end_of_week :
  (pencils_per_day * days_per_week + initial_pencils - sold_pencils) = 230 :=
by sorry  -- Proof will be filled in later

end pencils_in_stock_at_end_of_week_l89_89723


namespace inequality_abc_lt_l89_89233

variable (a b c : ℝ)

theorem inequality_abc_lt:
  c > b → b > a → a^2 * b + b^2 * c + c^2 * a < a * b^2 + b * c^2 + c * a^2 :=
by
  intros h1 h2
  sorry

end inequality_abc_lt_l89_89233


namespace initial_men_work_count_l89_89122

-- Define conditions given in the problem
def work_rate (M : ℕ) := 1 / (40 * M)
def initial_men_can_complete_work_in_40_days (M : ℕ) : Prop := M * work_rate M * 40 = 1
def work_done_by_initial_men_in_16_days (M : ℕ) := (M * 16) * work_rate M
def remaining_work_done_by_remaining_men_in_40_days (M : ℕ) := ((M - 14) * 40) * work_rate M

-- Define the main theorem to prove
theorem initial_men_work_count (M : ℕ) :
  initial_men_can_complete_work_in_40_days M →
  work_done_by_initial_men_in_16_days M = 2 / 5 →
  3 / 5 = (remaining_work_done_by_remaining_men_in_40_days M) →
  M = 15 :=
by
  intros h_initial h_16_days h_remaining
  have rate := h_initial
  sorry

end initial_men_work_count_l89_89122


namespace inequality_holds_for_positive_vars_l89_89578

theorem inequality_holds_for_positive_vars (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
    x^2 + y^2 + 1 ≥ x * y + x + y :=
sorry

end inequality_holds_for_positive_vars_l89_89578


namespace hyperbola_representation_l89_89915

variable (x y : ℝ)

/--
Given the equation (x - y)^2 = 3(x^2 - y^2), we prove that
the resulting graph represents a hyperbola.
-/
theorem hyperbola_representation :
  (x - y)^2 = 3 * (x^2 - y^2) →
  ∃ A B C : ℝ, A ≠ 0 ∧ (x^2 + x * y - 2 * y^2 = 0) ∧ (A = 1) ∧ (B = 1) ∧ (C = -2) ∧ (B^2 - 4*A*C > 0) :=
by
  sorry

end hyperbola_representation_l89_89915


namespace sufficient_but_not_necessary_condition_l89_89826

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2) :=
begin
  sorry
end

end sufficient_but_not_necessary_condition_l89_89826


namespace arrasta_um_proof_l89_89017

variable (n : ℕ)

def arrasta_um_possible_moves (n : ℕ) : ℕ :=
  6 * n - 8

theorem arrasta_um_proof (n : ℕ) (h : n ≥ 2) : arrasta_um_possible_moves n =
6 * n - 8 := by
  sorry

end arrasta_um_proof_l89_89017


namespace min_time_to_cook_noodles_l89_89432

/-- 
Li Ming needs to cook noodles, following these steps: 
① Boil the noodles for 4 minutes; 
② Wash vegetables for 5 minutes; 
③ Prepare the noodles and condiments for 2 minutes; 
④ Boil the water in the pot for 10 minutes; 
⑤ Wash the pot and add water for 2 minutes. 
Apart from step ④, only one step can be performed at a time. 
Prove that the minimum number of minutes needed to complete these tasks is 16.
-/
def total_time : Nat :=
  let t5 := 2 -- Wash the pot and add water
  let t4 := 10 -- Boil the water in the pot
  let t2 := 5 -- Wash vegetables
  let t3 := 2 -- Prepare the noodles and condiments
  let t1 := 4 -- Boil the noodles
  t5 + t4.max (t2 + t3) + t1

theorem min_time_to_cook_noodles : total_time = 16 :=
by
  sorry

end min_time_to_cook_noodles_l89_89432


namespace min_value_is_neg_500000_l89_89426

noncomputable def min_expression_value (a b : ℝ) : ℝ :=
  let term1 := a + 1/b
  let term2 := b + 1/a
  (term1 * (term1 - 1000) + term2 * (term2 - 1000))

theorem min_value_is_neg_500000 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_expression_value a b ≥ -500000 :=
sorry

end min_value_is_neg_500000_l89_89426


namespace positive_difference_of_two_numbers_l89_89597

variable {x y : ℝ}

theorem positive_difference_of_two_numbers (h₁ : x + y = 8) (h₂ : x^2 - y^2 = 24) : |x - y| = 3 :=
by
  sorry

end positive_difference_of_two_numbers_l89_89597


namespace wedge_volume_cylinder_l89_89548

theorem wedge_volume_cylinder (r h : ℝ) (theta : ℝ) (V : ℝ) 
  (hr : r = 6) (hh : h = 6) (htheta : theta = 60) (hV : V = 113) : 
  V = (theta / 360) * π * r^2 * h :=
by
  sorry

end wedge_volume_cylinder_l89_89548


namespace sum_of_numbers_given_average_l89_89088

variable (average : ℝ) (n : ℕ) (sum : ℝ)

theorem sum_of_numbers_given_average (h1 : average = 4.1) (h2 : n = 6) (h3 : average = sum / n) :
  sum = 24.6 :=
by
  sorry

end sum_of_numbers_given_average_l89_89088


namespace ratio_problem_l89_89533

theorem ratio_problem {q r s t : ℚ} (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) :
  t / q = 3 / 2 :=
sorry

end ratio_problem_l89_89533


namespace problem_solution_l89_89248

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {1, 2}

-- Define set B
def B : Set ℕ := {2}

-- Define the complement function specific to our universal set U
def complement (U A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Lean theorem to prove the given problem's correctness
theorem problem_solution : complement U (A ∪ B) = {3, 4} :=
by
  sorry -- Proof is omitted as per the instructions

end problem_solution_l89_89248


namespace tom_teaching_years_l89_89606

theorem tom_teaching_years :
  ∃ T D : ℕ, T + D = 70 ∧ D = (1 / 2) * T - 5 ∧ T = 50 :=
by
  sorry

end tom_teaching_years_l89_89606


namespace minimum_value_of_quadratic_l89_89754

theorem minimum_value_of_quadratic : ∀ x : ℝ, (∃ y : ℝ, y = (x-2)^2 - 3) → ∃ m : ℝ, (∀ x : ℝ, (x-2)^2 - 3 ≥ m) ∧ m = -3 :=
by
  sorry

end minimum_value_of_quadratic_l89_89754


namespace elvins_fixed_monthly_charge_l89_89370

-- Definition of the conditions
def january_bill (F C_J : ℝ) : Prop := F + C_J = 48
def february_bill (F C_J : ℝ) : Prop := F + 2 * C_J = 90

theorem elvins_fixed_monthly_charge (F C_J : ℝ) (h_jan : january_bill F C_J) (h_feb : february_bill F C_J) : F = 6 :=
by
  sorry

end elvins_fixed_monthly_charge_l89_89370


namespace total_items_left_in_store_l89_89655

noncomputable def items_ordered : ℕ := 4458
noncomputable def items_sold : ℕ := 1561
noncomputable def items_in_storeroom : ℕ := 575

theorem total_items_left_in_store : 
  (items_ordered - items_sold) + items_in_storeroom = 3472 := 
by 
  sorry

end total_items_left_in_store_l89_89655


namespace total_apartment_units_l89_89874

-- Define the number of apartment units on different floors
def units_first_floor := 2
def units_other_floors := 5
def num_other_floors := 3
def num_buildings := 2

-- Calculation of total units in one building
def units_one_building := units_first_floor + num_other_floors * units_other_floors

-- Calculation of total units in all buildings
def total_units := num_buildings * units_one_building

-- The theorem to prove
theorem total_apartment_units : total_units = 34 :=
by
  sorry

end total_apartment_units_l89_89874


namespace sum_even_ints_between_200_and_600_l89_89009

theorem sum_even_ints_between_200_and_600 
  : (Finset.sum (Finset.filter (λ n, n % 2 = 0) (Finset.Icc 200 600))) = 79600 :=
by
  sorry

end sum_even_ints_between_200_and_600_l89_89009


namespace quadratic_average_of_roots_l89_89944

theorem quadratic_average_of_roots (a b c : ℝ) (h_eq : a ≠ 0) (h_b : b = -6) (h_c : c = 3) 
  (discriminant : (b^2 - 4 * a * c) = 12) : 
  (b^2 - 4 * a * c = 12) → ((-b / (2 * a)) / 2 = 1.5) :=
by
  have a_val : a = 2 := sorry
  sorry

end quadratic_average_of_roots_l89_89944


namespace sixth_term_sequence_l89_89553

theorem sixth_term_sequence (a b c d : ℚ)
  (h1 : a = 1/4 * (3 + b))
  (h2 : b = 1/4 * (a + c))
  (h3 : c = 1/4 * (b + 48))
  (h4 : 48 = 1/4 * (c + d)) :
  d = 2001 / 14 :=
sorry

end sixth_term_sequence_l89_89553


namespace quadratic_ineq_solution_set_l89_89690

theorem quadratic_ineq_solution_set (a b c : ℝ) (h1 : a < 0) (h2 : ∀ x : ℝ, 3 < x → x < 6 → ax^2 + bx + c > 0) :
  ∀ x : ℝ, x < (1 / 6) ∨ x > (1 / 3) → cx^2 + bx + a < 0 := by 
  sorry

end quadratic_ineq_solution_set_l89_89690


namespace student_chose_number_l89_89335

theorem student_chose_number (x : ℕ) (h : 2 * x - 138 = 112) : x = 125 :=
by
  sorry

end student_chose_number_l89_89335


namespace total_strings_correct_l89_89514

-- Definitions based on conditions
def num_ukuleles : ℕ := 2
def num_guitars : ℕ := 4
def num_violins : ℕ := 2
def strings_per_ukulele : ℕ := 4
def strings_per_guitar : ℕ := 6
def strings_per_violin : ℕ := 4

-- Total number of strings
def total_strings : ℕ := num_ukuleles * strings_per_ukulele +
                         num_guitars * strings_per_guitar +
                         num_violins * strings_per_violin

-- The proof statement
theorem total_strings_correct : total_strings = 40 :=
by
  -- Proof omitted.
  sorry

end total_strings_correct_l89_89514


namespace wanda_can_eat_100000_numbers_l89_89767

-- Define the main theorem
theorem wanda_can_eat_100000_numbers :
  ∃ (n : ℕ), n ≤ 2011 ∧ ∃ (S : Finset (ℕ × ℕ)), S.card ≥ 100000 ∧
  (∀ ⟨i, j⟩ ∈ S, i ≤ n ∧ j ≤ i) ∧
  (∀ ⟨a, b⟩ ⟨c, d⟩ ⟨e, f⟩ ∈ S, (Nat.choose a b) + (Nat.choose c d) ≠ (Nat.choose e f)) :=
sorry

end wanda_can_eat_100000_numbers_l89_89767


namespace inv_100_mod_101_l89_89049

theorem inv_100_mod_101 : (100 : ℤ) * (100 : ℤ) % 101 = 1 := by
  sorry

end inv_100_mod_101_l89_89049


namespace find_number_l89_89085

theorem find_number (N Q : ℕ) (h1 : N = 5 * Q) (h2 : Q + N + 5 = 65) : N = 50 :=
by
  sorry

end find_number_l89_89085


namespace necessary_not_sufficient_x2_minus_3x_plus_2_l89_89990

theorem necessary_not_sufficient_x2_minus_3x_plus_2 (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ m → x^2 - 3 * x + 2 ≤ 0) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ m ∧ ¬(x^2 - 3 * x + 2 ≤ 0)) →
  m ≥ 2 :=
sorry

end necessary_not_sufficient_x2_minus_3x_plus_2_l89_89990


namespace volume_of_wedge_l89_89497

theorem volume_of_wedge (c : ℝ) (h : c = 18 * Real.pi) : 
  let r := c / (2 * Real.pi) in
  let V := (4 / 3) * Real.pi * r^3 in
  (V / 6) = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l89_89497


namespace balloons_problem_l89_89763

theorem balloons_problem :
  ∃ (b y : ℕ), y = 3414 ∧ b + y = 8590 ∧ b - y = 1762 := 
by
  sorry

end balloons_problem_l89_89763


namespace javiers_household_legs_l89_89419

-- Definitions given the problem conditions
def humans : ℕ := 6
def human_legs : ℕ := 2

def dogs : ℕ := 2
def dog_legs : ℕ := 4

def cats : ℕ := 1
def cat_legs : ℕ := 4

def parrots : ℕ := 1
def parrot_legs : ℕ := 2

def lizards : ℕ := 1
def lizard_legs : ℕ := 4

def stool_legs : ℕ := 3
def table_legs : ℕ := 4
def cabinet_legs : ℕ := 6

-- Problem statement
theorem javiers_household_legs :
  (humans * human_legs) + (dogs * dog_legs) + (cats * cat_legs) + (parrots * parrot_legs) +
  (lizards * lizard_legs) + stool_legs + table_legs + cabinet_legs = 43 := by
  -- We leave the proof as an exercise for the reader
  sorry

end javiers_household_legs_l89_89419


namespace find_a_if_even_function_l89_89544

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * 2^x + 2^(-x)

-- Theorem statement
theorem find_a_if_even_function (a : ℝ) (h : is_even_function (f a)) : a = 1 :=
sorry

end find_a_if_even_function_l89_89544


namespace tom_teaching_years_l89_89603

theorem tom_teaching_years (T D : ℝ) (h1 : T + D = 70) (h2 : D = (1/2) * T - 5) : T = 50 :=
by
  -- This is where the proof would normally go if it were required.
  sorry

end tom_teaching_years_l89_89603


namespace total_pieces_of_junk_mail_l89_89785

def pieces_per_block : ℕ := 48
def num_blocks : ℕ := 4

theorem total_pieces_of_junk_mail : (pieces_per_block * num_blocks) = 192 := by
  sorry

end total_pieces_of_junk_mail_l89_89785


namespace leak_empty_time_l89_89169

/-- 
The time taken for a leak to empty a full tank, given that an electric pump can fill a tank in 7 hours and it takes 14 hours to fill the tank with the leak present, is 14 hours.
 -/
theorem leak_empty_time (P L : ℝ) (hP : P = 1 / 7) (hCombined : P - L = 1 / 14) : L = 1 / 14 ∧ 1 / L = 14 :=
by
  sorry

end leak_empty_time_l89_89169


namespace count_red_balls_l89_89486

/-- Given conditions:
  - The total number of balls in the bag is 100.
  - There are 50 white, 20 green, 10 yellow, and 3 purple balls.
  - The probability that a ball will be neither red nor purple is 0.8.
  Prove that the number of red balls is 17. -/
theorem count_red_balls (total_balls white_balls green_balls yellow_balls purple_balls red_balls : ℕ)
  (h1 : total_balls = 100)
  (h2 : white_balls = 50)
  (h3 : green_balls = 20)
  (h4 : yellow_balls = 10)
  (h5 : purple_balls = 3)
  (h6 : (white_balls + green_balls + yellow_balls) = 80)
  (h7 : (white_balls + green_balls + yellow_balls) / (total_balls : ℝ) = 0.8) :
  red_balls = 17 :=
by
  sorry

end count_red_balls_l89_89486


namespace subtraction_equality_l89_89006

theorem subtraction_equality : 3.56 - 2.15 = 1.41 :=
by
  sorry

end subtraction_equality_l89_89006


namespace min_value_x_y_l89_89850

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : x + y ≥ 16 :=
sorry

end min_value_x_y_l89_89850


namespace fraction_of_phone_numbers_begin_with_8_and_end_with_5_l89_89650

theorem fraction_of_phone_numbers_begin_with_8_and_end_with_5 :
  let total_numbers := 7 * 10^7
  let specific_numbers := 10^6
  specific_numbers / total_numbers = 1 / 70 := by
  sorry

end fraction_of_phone_numbers_begin_with_8_and_end_with_5_l89_89650


namespace james_goals_product_l89_89099

theorem james_goals_product :
  ∃ (g7 g8 : ℕ), g7 < 7 ∧ g8 < 7 ∧ 
  (22 + g7) % 7 = 0 ∧ (22 + g7 + g8) % 8 = 0 ∧ 
  g7 * g8 = 24 :=
by
  sorry

end james_goals_product_l89_89099


namespace least_positive_int_satisfies_congruence_l89_89057

theorem least_positive_int_satisfies_congruence :
  ∃ x : ℕ, (x + 3001) % 15 = 1723 % 15 ∧ x = 12 :=
by
  sorry

end least_positive_int_satisfies_congruence_l89_89057


namespace lizard_ratio_l89_89559

def lizard_problem (W S : ℕ) : Prop :=
  (S = 7 * W) ∧ (3 = S + W - 69) ∧ (W / 3 = 3)

theorem lizard_ratio (W S : ℕ) (h : lizard_problem W S) : W / 3 = 3 :=
  by
    rcases h with ⟨h1, h2, h3⟩
    exact h3

end lizard_ratio_l89_89559


namespace perception_permutations_count_l89_89364

theorem perception_permutations_count :
  let n := 10
  let freq_P := 2
  let freq_E := 2
  let factorial := λ x : ℕ, (Nat.factorial x)
  factorial n / (factorial freq_P * factorial freq_E) = 907200 :=
by sorry

end perception_permutations_count_l89_89364


namespace percent_of_percent_l89_89154

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l89_89154


namespace hyperbola_slopes_l89_89070

variables {x1 y1 x2 y2 x y k1 k2 : ℝ}

theorem hyperbola_slopes (h1 : y1^2 - (x1^2 / 2) = 1)
  (h2 : y2^2 - (x2^2 / 2) = 1)
  (hx : x1 + x2 = 2 * x)
  (hy : y1 + y2 = 2 * y)
  (hk1 : k1 = (y2 - y1) / (x2 - x1))
  (hk2 : k2 = y / x) :
  k1 * k2 = 1 / 2 :=
sorry

end hyperbola_slopes_l89_89070


namespace dice_even_odd_probability_l89_89958

theorem dice_even_odd_probability : 
  let p : ℚ := (nat.choose 8 4) * (1 / 2) ^ 8 in
  p = 35 / 128 :=
by
  -- proof steps would go here
  sorry

end dice_even_odd_probability_l89_89958


namespace average_trees_planted_l89_89937

theorem average_trees_planted 
  (A : ℕ) 
  (B : ℕ) 
  (C : ℕ) 
  (h1 : A = 35) 
  (h2 : B = A + 6) 
  (h3 : C = A - 3) : 
  (A + B + C) / 3 = 36 :=
  by
  sorry

end average_trees_planted_l89_89937


namespace nathalie_cake_fraction_l89_89324

theorem nathalie_cake_fraction
    (cake_weight : ℕ)
    (pierre_ate : ℕ)
    (double_what_nathalie_ate : pierre_ate = 2 * (pierre_ate / 2))
    (pierre_ate_correct : pierre_ate = 100) :
    (pierre_ate / 2) / cake_weight = 1 / 8 :=
by
  sorry

end nathalie_cake_fraction_l89_89324


namespace minimum_value_proof_l89_89545

noncomputable def minimum_value : ℝ :=
  3 + 2 * Real.sqrt 2

theorem minimum_value_proof (a b : ℝ) (h_line_eq : ∀ x y : ℝ, a * x + b * y = 1)
  (h_ab_pos : a * b > 0)
  (h_center_bisect : ∃ x y : ℝ, (x - 1)^2 + (y - 2)^2 <= x^2 + y^2) :
  (1 / a + 1 / b) ≥ minimum_value :=
by
  -- Sorry placeholder for the proof
  sorry

end minimum_value_proof_l89_89545


namespace consecutive_product_not_mth_power_l89_89202

theorem consecutive_product_not_mth_power (n m k : ℕ) :
  ¬ ∃ k, (n - 1) * n * (n + 1) = k^m := 
sorry

end consecutive_product_not_mth_power_l89_89202


namespace order_of_fractions_l89_89392

theorem order_of_fractions (a b c d : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (hpos_d : d > 0)
(hab : a > b) : (b / a) < (b + c) / (a + c) ∧ (b + c) / (a + c) < (a + d) / (b + d) ∧ (a + d) / (b + d) < (a / b) :=
by
  sorry

end order_of_fractions_l89_89392


namespace smallest_value_A_plus_B_plus_C_plus_D_l89_89511

variable (A B C D : ℤ)

-- Given conditions in Lean statement form
def isArithmeticSequence (A B C : ℤ) : Prop :=
  B - A = C - B

def isGeometricSequence (B C D : ℤ) : Prop :=
  (C / B : ℚ) = 4 / 3 ∧ (D / C : ℚ) = C / B

def givenConditions (A B C D : ℤ) : Prop :=
  isArithmeticSequence A B C ∧ isGeometricSequence B C D

-- The proof problem to validate the smallest possible value
theorem smallest_value_A_plus_B_plus_C_plus_D (h : givenConditions A B C D) :
  A + B + C + D = 43 :=
sorry

end smallest_value_A_plus_B_plus_C_plus_D_l89_89511


namespace ThaboRatio_l89_89888

-- Define the variables
variables (P_f P_nf H_nf : ℕ)

-- Define the conditions as hypotheses
def ThaboConditions := P_f + P_nf + H_nf = 280 ∧ P_nf = H_nf + 20 ∧ H_nf = 55

-- State the theorem we want to prove
theorem ThaboRatio (h : ThaboConditions P_f P_nf H_nf) : (P_f / P_nf) = 2 :=
by sorry

end ThaboRatio_l89_89888


namespace initial_bees_l89_89142

variable (B : ℕ)

theorem initial_bees (h : B + 10 = 26) : B = 16 :=
by sorry

end initial_bees_l89_89142


namespace total_length_proof_l89_89561

noncomputable def total_length_climbed (keaton_ladder_height : ℕ) (keaton_times : ℕ) (shortening : ℕ) (reece_times : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - shortening
  let keaton_total := keaton_ladder_height * keaton_times
  let reece_total := reece_ladder_height * reece_times
  (keaton_total + reece_total) * 100

theorem total_length_proof :
  total_length_climbed 60 40 8 35 = 422000 := by
  sorry

end total_length_proof_l89_89561


namespace wedge_volume_formula_l89_89498

noncomputable def sphere_wedge_volume : ℝ :=
let r := 9 in
let volume_of_sphere := (4 / 3) * Real.pi * r^3 in
let volume_of_one_wedge := volume_of_sphere / 6 in
volume_of_one_wedge

theorem wedge_volume_formula
  (circumference : ℝ)
  (h1 : circumference = 18 * Real.pi)
  (num_wedges : ℕ)
  (h2 : num_wedges = 6) :
  sphere_wedge_volume = 162 * Real.pi :=
by
  sorry

end wedge_volume_formula_l89_89498


namespace number_of_companies_l89_89198

theorem number_of_companies (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
by
  sorry

end number_of_companies_l89_89198


namespace probability_even_sum_includes_ball_15_l89_89924

-- Definition of the conditions in Lean
def balls : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

def odd_balls : Set ℕ := {n ∈ balls | n % 2 = 1}
def even_balls : Set ℕ := {n ∈ balls | n % 2 = 0}
def ball_15 : ℕ := 15

-- The number of ways to choose k elements from a set of n elements
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Number of ways to draw 7 balls ensuring the sum is even and ball 15 is included
def favorable_outcomes : ℕ :=
  choose 6 5 * choose 8 1 +   -- 5 other odd and 1 even
  choose 6 3 * choose 8 3 +   -- 3 other odd and 3 even
  choose 6 1 * choose 8 5     -- 1 other odd and 5 even

-- Total number of ways to choose 7 balls including ball 15:
def total_outcomes : ℕ := choose 14 6

-- Probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- The proof we require
theorem probability_even_sum_includes_ball_15 :
  probability = 1504 / 3003 :=
by
  -- proof omitted for brevity
  sorry

end probability_even_sum_includes_ball_15_l89_89924


namespace fifty_new_edges_l89_89092

-- Definition: Tree with 100 vertices, each vertex having exactly one outgoing road.
def initial_tree (V : Type) [Fintype V] [DecidableEq V] : SimpleGraph V :=
{ Edge := λ x y, ∃ (p : Path x y), p.edges.length = 1,
  symm := by finish,
  loopless := by finish }

-- Problem statement: Proving the existence of 50 new edges under given conditions.
theorem fifty_new_edges (V : Type) [Fintype V] [DecidableEq V] (G : SimpleGraph V)
    (h_tree : G.IsTree) (h_vertices : Fintype.card V = 100)
    (h_leaves : ∀ v, G.degree v = 1) :
  ∃ (new_edges : Finset (Sym2 V)), new_edges.card = 50 ∧
    ∀ e ∈ G.edgeFinset ∪ new_edges, (G ⧸ e).IsConnected :=
sorry

end fifty_new_edges_l89_89092


namespace problem1_l89_89481

def f (x : ℝ) := (1 - 3 * x) * (1 + x) ^ 5

theorem problem1 :
  let a : ℝ := f (1 / 3)
  a = 0 :=
by
  let a := f (1 / 3)
  sorry

end problem1_l89_89481


namespace probability_even_equals_odd_when_eight_dice_rolled_l89_89963

theorem probability_even_equals_odd_when_eight_dice_rolled :
  let diceRollOutcome := {1, 2, 3, 4, 5, 6}
  let evenNumbers := {2, 4, 6}
  let oddNumbers := {1, 3, 5}
  let totalDice := 8
  ∀ numberEven numberOdd : ℕ, numberEven = 4 → numberOdd = 4 →
  let prob_even_odd := (Nat.choose totalDice numberEven) * (1/2)^totalDice
  prob_even_odd = 35 / 128 := sorry

end probability_even_equals_odd_when_eight_dice_rolled_l89_89963


namespace least_three_digit_7_heavy_l89_89342

-- Define what it means to be a 7-heavy number
def is_7_heavy (n : ℕ) : Prop :=
  n % 7 > 4

-- Define the property of being three-digit
def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

-- The statement to prove
theorem least_three_digit_7_heavy : ∃ n, is_7_heavy n ∧ is_three_digit n ∧ ∀ m, is_7_heavy m ∧ is_three_digit m → n ≤ m :=
begin
  use [104],
  split,
  { -- Proof that 104 is 7-heavy
    show is_7_heavy 104,
    simp [is_7_heavy], -- Calculation: 104 % 7 = 6 which is > 4
    norm_num,
  },
  split,
  { -- Proof that 104 is a three-digit number
    show is_three_digit 104,
    simp [is_three_digit],
    norm_num,
  },
  { -- Proof that 104 is the smallest 7-heavy three-digit number
    intros m hm,
    cases hm with hm1 hm2,
    suffices : 104 ≤ m,
    exact this,
    calc 104 ≤ 100 + 7 - 1 : by norm_num
        ... ≤ m            : by linarith [hm2.left, hm2.right],
    sorry,
  }
sorry

end least_three_digit_7_heavy_l89_89342


namespace solve_perimeter_l89_89900

noncomputable def ellipse_perimeter_proof : Prop :=
  let a := 4
  let b := Real.sqrt 7
  let c := 3
  let F1 := (-c, 0)
  let F2 := (c, 0)
  let ellipse_eq (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 7) = 1
  ∀ (A B : ℝ×ℝ), 
    (ellipse_eq A.1 A.2) ∧ (ellipse_eq B.1 B.2) ∧ (∃ l : ℝ, l ≠ 0 ∧ ∀ t : ℝ, (A = (F1.1 + t * l, F1.2 + t * l)) ∨ (B = (F1.1 + t * l, F1.2 + t * l))) 
    → ∃ P : ℝ, P = 16

theorem solve_perimeter : ellipse_perimeter_proof := sorry

end solve_perimeter_l89_89900


namespace last_four_digits_5_pow_2017_l89_89435

theorem last_four_digits_5_pow_2017 : (5 ^ 2017) % 10000 = 3125 :=
by sorry

end last_four_digits_5_pow_2017_l89_89435


namespace books_sold_l89_89118

def initial_books : ℕ := 134
def given_books : ℕ := 39
def books_left : ℕ := 68

theorem books_sold : (initial_books - given_books - books_left = 27) := 
by 
  sorry

end books_sold_l89_89118


namespace complete_square_eq_l89_89500

theorem complete_square_eq (x : ℝ) :
  x^2 - 8 * x + 15 = 0 →
  (x - 4)^2 = 1 :=
by sorry

end complete_square_eq_l89_89500


namespace sum_of_three_distinct_integers_l89_89451

theorem sum_of_three_distinct_integers (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c) 
  (h₄ : a * b * c = 5^3) (h₅ : a > 0) (h₆ : b > 0) (h₇ : c > 0) : a + b + c = 31 :=
by
  sorry

end sum_of_three_distinct_integers_l89_89451


namespace greatest_multiple_of_4_l89_89740

theorem greatest_multiple_of_4 (x : ℕ) (h1 : x % 4 = 0) (h2 : x > 0) (h3 : x^3 < 500) : x ≤ 4 :=
by sorry

end greatest_multiple_of_4_l89_89740


namespace jeans_and_shirts_l89_89037

-- Let's define the necessary variables and conditions.
variables (J S X : ℝ)

-- Given conditions
def condition1 := 3 * J + 2 * S = X
def condition2 := 2 * J + 3 * S = 61

-- Given the price of one shirt
def price_of_shirt := S = 9

-- The problem we need to prove
theorem jeans_and_shirts : condition1 J S X ∧ condition2 J S ∧ price_of_shirt S →
  X = 69 :=
by
  sorry

end jeans_and_shirts_l89_89037


namespace two_pow_a_plus_two_pow_neg_a_l89_89696

theorem two_pow_a_plus_two_pow_neg_a (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 1) :
  2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end two_pow_a_plus_two_pow_neg_a_l89_89696


namespace problem_I5_1_l89_89249

theorem problem_I5_1 (a : ℝ) (h : a^2 - 8^2 = 12^2 + 9^2) : a = 17 := 
sorry

end problem_I5_1_l89_89249


namespace max_black_cells_in_101x101_grid_l89_89921

theorem max_black_cells_in_101x101_grid :
  ∀ (k : ℕ), k ≤ 101 → 2 * k * (101 - k) ≤ 5100 :=
by
  sorry

end max_black_cells_in_101x101_grid_l89_89921


namespace negation_of_universal_statement_l89_89264

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2 ≤ 0 :=
sorry

end negation_of_universal_statement_l89_89264


namespace greatest_integer_gcd_3_l89_89610

theorem greatest_integer_gcd_3 : ∃ n, n < 100 ∧ gcd n 18 = 3 ∧ ∀ m, m < 100 ∧ gcd m 18 = 3 → m ≤ n := by
  sorry

end greatest_integer_gcd_3_l89_89610


namespace min_xy_min_x_plus_y_l89_89976

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 := 
sorry

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 := 
sorry

end min_xy_min_x_plus_y_l89_89976


namespace eval_exponents_l89_89760

theorem eval_exponents : (2^3)^2 - 4^3 = 0 := by
  sorry

end eval_exponents_l89_89760


namespace parabola_through_point_intersects_axes_at_two_points_l89_89379

open Real

def quadratic_function_a := λ a x : ℝ, x^2 - 4 * x - 3 + a

theorem parabola_through_point (a : ℝ) :
  (∃ y, quadratic_function_a a 0 = y ∧ y = 1) → a = 4 := by
  sorry

theorem intersects_axes_at_two_points (a : ℝ) :
  (∀ x y : ℝ, quadratic_function_a a x = 0 → quadratic_function_a a y = 0) →
  (∃ b, b^2 - 4 * 1 * (-3 + a) = 0) → a = 7 := by
  sorry

end parabola_through_point_intersects_axes_at_two_points_l89_89379


namespace sum_first_five_arithmetic_l89_89060

theorem sum_first_five_arithmetic (a : ℕ → ℝ) (h₁ : ∀ n, a n = a 0 + n * (a 1 - a 0)) (h₂ : a 1 = -1) (h₃ : a 3 = -5) :
  (a 0 + a 1 + a 2 + a 3 + a 4) = -15 :=
by
  sorry

end sum_first_five_arithmetic_l89_89060


namespace exponent_on_right_side_l89_89538

theorem exponent_on_right_side (n : ℕ) (k : ℕ) (h : n = 25) :
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^k → k = 26 :=
by
  sorry

end exponent_on_right_side_l89_89538


namespace perception_num_permutations_l89_89357

def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def perception_arrangements : ℕ :=
  let total_letters := 10
  let repetitions_P := 2
  let repetitions_E := 2
  factorial total_letters / (factorial repetitions_P * factorial repetitions_E)

theorem perception_num_permutations :
  perception_arrangements = 907200 :=
by sorry

end perception_num_permutations_l89_89357


namespace max_L_shaped_figures_in_5x7_rectangle_l89_89312

def L_shaped_figure : Type := ℕ

def rectangle_area := 5 * 7

def l_shape_area := 3

def max_l_shapes_in_rectangle (rect_area : ℕ) (l_area : ℕ) : ℕ := rect_area / l_area

theorem max_L_shaped_figures_in_5x7_rectangle : max_l_shapes_in_rectangle rectangle_area l_shape_area = 11 :=
by
  sorry

end max_L_shaped_figures_in_5x7_rectangle_l89_89312


namespace duration_of_call_l89_89422

def initial_credit : ℝ := 30
def remaining_credit : ℝ := 26.48
def cost_per_minute : ℝ := 0.16
def credit_used : ℝ := initial_credit - remaining_credit
def minutes_of_call : ℕ := (credit_used / cost_per_minute).toInt

theorem duration_of_call :
  minutes_of_call = 22 :=
by
  -- Placeholder proof statement
  sorry

end duration_of_call_l89_89422


namespace magnitude_of_root_of_quadratic_eq_l89_89834

open Complex

theorem magnitude_of_root_of_quadratic_eq (z : ℂ) 
  (h : z^2 - (2 : ℂ) * z + 2 = 0) : abs z = Real.sqrt 2 :=
by 
  sorry

end magnitude_of_root_of_quadratic_eq_l89_89834


namespace number_of_cows_on_boat_l89_89322

-- Definitions based on conditions
def number_of_sheep := 20
def number_of_dogs := 14
def sheep_drowned := 3
def cows_drowned := 2 * sheep_drowned  -- Twice as many cows drowned as did sheep.
def dogs_made_it_shore := number_of_dogs  -- All dogs made it to shore.
def total_animals_shore := 35
def total_sheep_shore := number_of_sheep - sheep_drowned
def total_sheep_cows_shore := total_animals_shore - dogs_made_it_shore
def cows_made_it_shore := total_sheep_cows_shore - total_sheep_shore

-- Theorem stating the problem
theorem number_of_cows_on_boat : 
  (cows_made_it_shore + cows_drowned) = 10 := by
  sorry

end number_of_cows_on_boat_l89_89322


namespace count_integers_between_sqrts_l89_89082

theorem count_integers_between_sqrts (a b : ℝ) (h1 : a = 10) (h2 : b = 100) :
  let lower_bound := Int.ceil (Real.sqrt a),
      upper_bound := Int.floor (Real.sqrt b) in
  (upper_bound - lower_bound + 1) = 7 := 
by
  rw [h1, h2]
  let lower_bound := Int.ceil (Real.sqrt 10)
  let upper_bound := Int.floor (Real.sqrt 100)
  have h_lower : lower_bound = 4 := by sorry
  have h_upper : upper_bound = 10 := by sorry
  rw [h_lower, h_upper]
  norm_num
  sorry

end count_integers_between_sqrts_l89_89082


namespace part1_part2_l89_89520

def setA : Set ℝ := {x | (x - 2) / (x + 1) < 0}
def setB (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 - k}

theorem part1 : (setB (-1)).union setA = {x : ℝ | -1 < x ∧ x < 3 } := by
  sorry

theorem part2 (k : ℝ) : (setA ∩ setB k = setB k ↔ 0 ≤ k) := by
  sorry

end part1_part2_l89_89520


namespace value_of_f_log_half_24_l89_89849

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_log_half_24 :
  (∀ x : ℝ, f x * -1 = f (-x)) → -- Condition 1: f(x) is an odd function.
  (∀ x : ℝ, f (x + 1) = f (x - 1)) → -- Condition 2: f(x + 1) = f(x - 1).
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2^x - 2) → -- Condition 3: For 0 < x < 1, f(x) = 2^x - 2.
  f (Real.logb 0.5 24) = 1 / 2 := 
sorry

end value_of_f_log_half_24_l89_89849


namespace triangle_B_is_right_triangle_l89_89396

theorem triangle_B_is_right_triangle :
  let a := 1
  let b := 2
  let c := Real.sqrt 3
  a^2 + c^2 = b^2 :=
by
  sorry

end triangle_B_is_right_triangle_l89_89396


namespace dan_has_more_balloons_l89_89207

-- Constants representing the number of balloons Dan and Tim have
def dans_balloons : ℝ := 29.0
def tims_balloons : ℝ := 4.142857143

-- Theorem: The ratio of Dan's balloons to Tim's balloons is 7
theorem dan_has_more_balloons : dans_balloons / tims_balloons = 7 := 
by
  sorry

end dan_has_more_balloons_l89_89207


namespace problem_solution_exists_l89_89803

theorem problem_solution_exists (x : ℝ) (h : ∃ x, 2 * (3 * 5 - x) - x = -8) : x = 10 :=
sorry

end problem_solution_exists_l89_89803


namespace number_of_black_balls_l89_89098

theorem number_of_black_balls
  (total_balls : ℕ)  -- define the total number of balls
  (B : ℕ)            -- define B as the number of black balls
  (prob_red : ℚ := 1/4) -- define the probability of drawing a red ball as 1/4
  (red_balls : ℕ := 3)  -- define the number of red balls as 3
  (h1 : total_balls = red_balls + B) -- total balls is the sum of red and black balls
  (h2 : red_balls / total_balls = prob_red) -- given probability
  : B = 9 :=              -- we need to prove that B is 9
by
  sorry

end number_of_black_balls_l89_89098


namespace ratio_xy_l89_89991

theorem ratio_xy (x y : ℝ) (h : 2*y - 5*x = 0) : x / y = 2 / 5 :=
by sorry

end ratio_xy_l89_89991


namespace running_race_total_students_l89_89523

theorem running_race_total_students 
  (number_of_first_grade_students number_of_second_grade_students : ℕ)
  (h1 : number_of_first_grade_students = 8)
  (h2 : number_of_second_grade_students = 5 * number_of_first_grade_students) :
  number_of_first_grade_students + number_of_second_grade_students = 48 := 
by
  -- we will leave the proof empty
  sorry

end running_race_total_students_l89_89523


namespace total_steps_walked_l89_89716

theorem total_steps_walked (d_mabel : ℕ) (d_helen : ℕ) (h1 : d_mabel = 4500) (h2 : d_helen = 3 * d_mabel / 4) : 
  d_mabel + d_helen = 7875 :=
by
  rw [h1, h2]
  have : 3 * 4500 / 4 = 3375 := by norm_num
  rw this
  norm_num
  sorry

end total_steps_walked_l89_89716


namespace simplify_fraction_l89_89880

theorem simplify_fraction (a b : ℤ) (h : a = 2^6 + 2^4) (h1 : b = 2^5 - 2^2) : 
  (a / b : ℚ) = 20 / 7 := by
  sorry

end simplify_fraction_l89_89880


namespace percent_calculation_l89_89150

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l89_89150


namespace amy_required_hours_per_week_l89_89642

variable (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_pay : ℕ) 
variable (pay_raise_percent : ℕ) (school_year_weeks : ℕ) (required_school_year_pay : ℕ)

def summer_hours_total := summer_hours_per_week * summer_weeks
def summer_hourly_pay := summer_pay / summer_hours_total
def new_hourly_pay := summer_hourly_pay + (summer_hourly_pay / 10)  -- 10% pay raise
def total_needed_hours := required_school_year_pay / new_hourly_pay
def required_hours_per_week := total_needed_hours / school_year_weeks

theorem amy_required_hours_per_week :
  summer_hours_per_week = 40 →
  summer_weeks = 12 →
  summer_pay = 4800 →
  pay_raise_percent = 10 →
  school_year_weeks = 36 →
  required_school_year_pay = 7200 →
  required_hours_per_week = 18 := sorry

end amy_required_hours_per_week_l89_89642


namespace sum_of_two_numbers_l89_89901

variables {x y : ℝ}

theorem sum_of_two_numbers (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end sum_of_two_numbers_l89_89901


namespace distance_post_office_l89_89633

theorem distance_post_office 
  (D : ℝ)
  (speed_to_post_office : ℝ := 25)
  (speed_back : ℝ := 4)
  (total_time : ℝ := 5 + (48 / 60)) :
  (D / speed_to_post_office + D / speed_back = total_time) → D = 20 :=
by
  sorry

end distance_post_office_l89_89633


namespace number_of_satisfying_subsets_l89_89061

theorem number_of_satisfying_subsets (A B C : Finset ℕ) (hAB : A ⊆ B) (hAC : A ⊆ C) (hB : B = {0, 1, 2, 3, 4}) (hC : C = {0, 2, 4, 8}) : 
  (∃ n, n = 8) :=
by
  sorry

end number_of_satisfying_subsets_l89_89061


namespace min_value_l89_89539

theorem min_value (a b c : ℤ) (h : a > b ∧ b > c) :
  ∃ x, x = (a + b + c) / (a - b - c) ∧ 
       x + (a - b - c) / (a + b + c) = 2 := sorry

end min_value_l89_89539


namespace binom_15_13_eq_105_l89_89800

theorem binom_15_13_eq_105 : Nat.choose 15 13 = 105 := by
  sorry

end binom_15_13_eq_105_l89_89800


namespace f_inequality_l89_89023

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f(x+3) = -1 / f(x)
axiom f_prop1 : ∀ x : ℝ, f (x + 3) = -1 / f x

-- Condition 2: ∀ 3 ≤ x_1 < x_2 ≤ 6, f(x_1) < f(x_2)
axiom f_prop2 : ∀ x1 x2 : ℝ, 3 ≤ x1 → x1 < x2 → x2 ≤ 6 → f x1 < f x2

-- Condition 3: The graph of y = f(x + 3) is symmetric about the y-axis
axiom f_prop3 : ∀ x : ℝ, f (3 - x) = f (3 + x)

-- Theorem: f(3) < f(4.5) < f(7)
theorem f_inequality : f 3 < f 4.5 ∧ f 4.5 < f 7 := by
  sorry

end f_inequality_l89_89023


namespace part_a_part_b_part_c_l89_89172

-- Part (a) Lean Statement
theorem part_a (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ k : ℝ, k = 2 * p / (p + 1)) :=
by
  -- Definitions and conditions would go here
  sorry

-- Part (b) Lean Statement
theorem part_b (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∃ q : ℝ, q = 1 - p ∧ ∃ r : ℝ, r = 2 * p / (2 * p + (1 - p) ^ 2)) :=
by
  -- Definitions and conditions would go here
  sorry

-- Part (c) Lean Statement
theorem part_c (N : ℕ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ S : ℝ, S = N * p / (p + 1)) :=
by
  -- Definitions and conditions would go here
  sorry

end part_a_part_b_part_c_l89_89172


namespace intersection_M_N_l89_89869

def M : Set ℝ := {x | x < 2016}

def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l89_89869


namespace percent_calculation_l89_89151

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l89_89151


namespace tan_double_alpha_l89_89239

theorem tan_double_alpha (α : ℝ) (h : ∀ x : ℝ, (3 * Real.sin x + Real.cos x) ≤ (3 * Real.sin α + Real.cos α)) :
  Real.tan (2 * α) = -3 / 4 :=
sorry

end tan_double_alpha_l89_89239


namespace sum_of_numbers_in_third_column_is_96_l89_89112

theorem sum_of_numbers_in_third_column_is_96 :
  ∃ (a : ℕ), (136 = a + 16 * a) ∧ (272 = 2 * a + 32 * a) ∧ (12 * a = 96) :=
by
  let a := 8
  have h1 : 136 = a + 16 * a := by sorry  -- Proof here that 136 = 8 + 16 * 8
  have h2 : 272 = 2 * a + 32 * a := by sorry  -- Proof here that 272 = 2 * 8 + 32 * 8
  have h3 : 12 * a = 96 := by sorry  -- Proof here that 12 * 8 = 96
  existsi a
  exact ⟨h1, h2, h3⟩

end sum_of_numbers_in_third_column_is_96_l89_89112


namespace find_c2_given_d4_l89_89739

theorem find_c2_given_d4 (c d k : ℝ) (h : c^2 * d^4 = k) (hc8 : c = 8) (hd2 : d = 2) (hd4 : d = 4):
  c^2 = 4 :=
by
  sorry

end find_c2_given_d4_l89_89739


namespace polynomial_coefficients_l89_89971

theorem polynomial_coefficients (x a₄ a₃ a₂ a₁ a₀ : ℝ) (h : (x - 1)^4 = a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀)
  : a₄ - a₃ + a₂ - a₁ = 15 := by
  sorry

end polynomial_coefficients_l89_89971


namespace value_depletion_rate_l89_89488

theorem value_depletion_rate (V_initial V_final : ℝ) (t : ℝ) (r : ℝ) :
  V_initial = 900 → V_final = 729 → t = 2 → V_final = V_initial * (1 - r)^t → r = 0.1 :=
by sorry

end value_depletion_rate_l89_89488


namespace value_of_c_l89_89983

-- Define the function f(x)
def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem value_of_c (a b c m : ℝ) (h₀ : ∀ x : ℝ, 0 ≤ f x a b)
  (h₁ : ∀ x : ℝ, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
sorry

end value_of_c_l89_89983


namespace quadratic_real_roots_l89_89673

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ k ≥ -9 / 4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_real_roots_l89_89673


namespace minimum_value_l89_89235

noncomputable def min_value (a b c d : ℝ) : ℝ :=
(a - c) ^ 2 + (b - d) ^ 2

theorem minimum_value (a b c d : ℝ) (hab : a * b = 3) (hcd : c + 3 * d = 0) :
  min_value a b c d ≥ (18 / 5) :=
by
  sorry

end minimum_value_l89_89235


namespace volume_of_each_cube_is_correct_l89_89024

def box_length : ℕ := 12
def box_width : ℕ := 16
def box_height : ℕ := 6
def total_volume : ℕ := 1152
def number_of_cubes : ℕ := 384

theorem volume_of_each_cube_is_correct :
  (total_volume / number_of_cubes = 3) :=
by
  sorry

end volume_of_each_cube_is_correct_l89_89024


namespace books_loaned_out_l89_89489

theorem books_loaned_out (initial_books : ℕ) (returned_percentage : ℝ) (end_books : ℕ) (x : ℝ) :
    initial_books = 75 →
    returned_percentage = 0.70 →
    end_books = 63 →
    0.30 * x = (initial_books - end_books) →
    x = 40 := by
  sorry

end books_loaned_out_l89_89489


namespace find_a_and_union_set_l89_89866

theorem find_a_and_union_set (a : ℝ) 
  (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-3, a + 1}) 
  (hB : B = {2 * a - 1, a ^ 2 + 1}) 
  (h_inter : A ∩ B = {3}) : 
  a = 2 ∧ A ∪ B = {-3, 3, 5} :=
by
  sorry

end find_a_and_union_set_l89_89866


namespace period_ends_at_5_pm_l89_89674

open Time

def rain_duration : TimeSpan := Time.of_seconds (2 * 3600) -- 2 hours
def no_rain_duration : TimeSpan := Time.of_seconds (6 * 3600) -- 6 hours
def start_time : Time := Time.mk 9 0 0 -- 9:00 am

def end_time := start_time + rain_duration + no_rain_duration

theorem period_ends_at_5_pm : end_time = Time.mk 17 0 0 := begin
  sorry
end

end period_ends_at_5_pm_l89_89674


namespace find_f3_l89_89390

variable (f : ℕ → ℕ)

axiom h : ∀ x : ℕ, f (x + 1) = x ^ 2

theorem find_f3 : f 3 = 4 :=
by
  sorry

end find_f3_l89_89390


namespace number_of_small_slices_l89_89794

-- Define the given conditions
variables (S L : ℕ)
axiom total_slices : S + L = 5000
axiom total_revenue : 150 * S + 250 * L = 1050000

-- State the problem we need to prove
theorem number_of_small_slices : S = 1500 :=
by sorry

end number_of_small_slices_l89_89794


namespace general_term_formula_l89_89777

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d, ∀ n, a (n + 1) = a n + d

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 5 = (2 / 7) * (a 3) ^ 2) (h2 : S 7 = 63) :
  ∀ n, a n = 2 * n + 1 := by
  sorry

end general_term_formula_l89_89777


namespace mass_percentage_H_in_C4H8O2_l89_89227

theorem mass_percentage_H_in_C4H8O2 (molar_mass_C : Real := 12.01) 
                                    (molar_mass_H : Real := 1.008) 
                                    (molar_mass_O : Real := 16.00) 
                                    (num_C_atoms : Nat := 4)
                                    (num_H_atoms : Nat := 8)
                                    (num_O_atoms : Nat := 2) :
    (num_H_atoms * molar_mass_H) / ((num_C_atoms * molar_mass_C) + (num_H_atoms * molar_mass_H) + (num_O_atoms * molar_mass_O)) * 100 = 9.15 :=
by
  sorry

end mass_percentage_H_in_C4H8O2_l89_89227


namespace least_possible_perimeter_l89_89751

theorem least_possible_perimeter (x : ℕ) (h1 : 27 < x) (h2 : x < 75) :
  24 + 51 + x = 103 :=
by
  sorry

end least_possible_perimeter_l89_89751


namespace find_a8_l89_89265

noncomputable def arithmetic_sequence : Type := sorry

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (n d : ℝ)
variables (h1 : a 6 = 12) (h2 : S 3 = 12)

theorem find_a8 : 
  let d := 2,
  let a1 := 2 in
  (a 8 = a1 + 7 * d) →
  a 8 = 16 :=
by
  sorry

end find_a8_l89_89265


namespace opposite_sqrt_4_l89_89591

theorem opposite_sqrt_4 : - (Real.sqrt 4) = -2 := sorry

end opposite_sqrt_4_l89_89591


namespace wilted_flowers_are_18_l89_89721

def picked_flowers := 53
def flowers_per_bouquet := 7
def bouquets_after_wilted := 5

def flowers_left := bouquets_after_wilted * flowers_per_bouquet
def flowers_wilted : ℕ := picked_flowers - flowers_left

theorem wilted_flowers_are_18 : flowers_wilted = 18 := by
  sorry

end wilted_flowers_are_18_l89_89721


namespace sum_of_odds_square_l89_89115

theorem sum_of_odds_square (n : ℕ) (h : 0 < n) : (Finset.range n).sum (λ i => 2 * i + 1) = n ^ 2 :=
sorry

end sum_of_odds_square_l89_89115


namespace number_of_boys_at_reunion_l89_89087

theorem number_of_boys_at_reunion (n : ℕ) (h : n * (n - 1) / 2 = 66) : n = 12 :=
sorry

end number_of_boys_at_reunion_l89_89087


namespace sum_of_sequences_contains_repetition_l89_89431

open Finset

/-- Define the alphabet positions from 1 to 26 -/
def alphabet_positions : Finset ℕ := range 27

noncomputable def sum_sequences (seq1 seq2 : Fin 27 → ℕ) : Fin 27 → ℕ :=
  λ i => ((seq1 i) + (seq2 i)) % 26+1

theorem sum_of_sequences_contains_repetition (seq1 seq2 : Fin 27 → ℕ)
  (h_seq1 : ∀ i, seq1 i ∈ alphabet_positions) 
  (h_seq2 : ∀ i, seq2 i ∈ alphabet_positions) 
  (h_distinct : injective seq1) :
  ¬ injective (sum_sequences seq1 seq2) :=
sorry

end sum_of_sequences_contains_repetition_l89_89431


namespace bar_graph_represents_circle_graph_l89_89928

theorem bar_graph_represents_circle_graph (r b g : ℕ) 
  (h1 : r = g) 
  (h2 : b = 3 * r) : 
  (r = 1 ∧ b = 3 ∧ g = 1) :=
sorry

end bar_graph_represents_circle_graph_l89_89928


namespace expression_value_at_2_l89_89768

theorem expression_value_at_2 : (2^2 - 3 * 2 + 2) = 0 :=
by
  sorry

end expression_value_at_2_l89_89768


namespace fraction_identity_l89_89540

theorem fraction_identity (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : y^2 - 1/x ≠ 0) :
  (x^2 - 1/y) / (y^2 - 1/x) = x / y := 
by {
  sorry
}

end fraction_identity_l89_89540


namespace derivative_given_limit_l89_89105

open Real

variable {f : ℝ → ℝ} {x₀ : ℝ}

theorem derivative_given_limit (h : ∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (x₀ - 2 * Δx) - f x₀) / Δx + 2) < ε) :
  deriv f x₀ = -1 := by
  sorry

end derivative_given_limit_l89_89105


namespace sum_of_first_and_third_l89_89140

theorem sum_of_first_and_third :
  ∀ (A B C : ℕ),
  A + B + C = 330 →
  A = 2 * B →
  C = A / 3 →
  B = 90 →
  A + C = 240 :=
by
  intros A B C h1 h2 h3 h4
  sorry

end sum_of_first_and_third_l89_89140


namespace absolute_value_and_power_sum_l89_89041

theorem absolute_value_and_power_sum :
  |(-4 : ℤ)| + (3 - Real.pi)^0 = 5 := by
  sorry

end absolute_value_and_power_sum_l89_89041


namespace tan_alpha_neg_four_thirds_cos2alpha_plus_cos_alpha_add_pi_over_2_l89_89062

variable (α : ℝ)
variable (h1 : π / 2 < α)
variable (h2 : α < π)
variable (h3 : Real.sin α = 4 / 5)

theorem tan_alpha_neg_four_thirds (h1 : π / 2 < α) (h2 : α < π) (h3 : Real.sin α = 4 / 5) : Real.tan α = -4 / 3 := 
by sorry

theorem cos2alpha_plus_cos_alpha_add_pi_over_2 (h1 : π / 2 < α) (h2 : α < π) (h3 : Real.sin α = 4 / 5) : 
  Real.cos (2 * α) + Real.cos (α + π / 2) = -27 / 25 := 
by sorry

end tan_alpha_neg_four_thirds_cos2alpha_plus_cos_alpha_add_pi_over_2_l89_89062


namespace reflection_eqn_l89_89753

theorem reflection_eqn 
  (x y : ℝ)
  (h : y = 2 * x + 3) : 
  -y = 2 * x + 3 :=
sorry

end reflection_eqn_l89_89753


namespace count_different_numerators_in_T_l89_89712

theorem count_different_numerators_in_T : 
  let T := { r : ℚ | ∃ (a b c d : ℕ), r = (1000 * a + 100 * b + 10 * c + d) / 9999 ∧ 0 < r ∧ r < 1 ∧ r.denom = 9999 ∧ Nat.gcd (1000 * a + 100 * b + 10 * c + d) 9999 = 1 } in
  T.count ≈ 5800 :=
by
  sorry

end count_different_numerators_in_T_l89_89712


namespace find_number_of_cups_l89_89876

theorem find_number_of_cups (a C B : ℝ) (h1 : a * C + 2 * B = 12.75) (h2 : 2 * C + 5 * B = 14.00) (h3 : B = 1.5) : a = 3 :=
by
  sorry

end find_number_of_cups_l89_89876


namespace percent_calculation_l89_89148

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l89_89148


namespace badgers_win_at_least_three_l89_89441

noncomputable def probability_badgers_win_at_least_three : ℝ :=
  ∑ k in Finset.range (5 + 1), if k ≥ 3 then Nat.choose 5 k * (0.5)^k * (0.5)^(5 - k) else 0

theorem badgers_win_at_least_three :
  probability_badgers_win_at_least_three = 1 / 2 :=
by sorry

end badgers_win_at_least_three_l89_89441


namespace lambda_sum_ellipse_l89_89524

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

noncomputable def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 4)

noncomputable def intersects_y_axis (k : ℝ) : ℝ × ℝ :=
  (0, -4 * k)

noncomputable def lambda1 (x1 : ℝ) : ℝ :=
  x1 / (4 - x1)

noncomputable def lambda2 (x2 : ℝ) : ℝ :=
  x2 / (4 - x2)

theorem lambda_sum_ellipse {k x1 x2 : ℝ}
  (h1 : ellipse x1 (k * (x1 - 4)))
  (h2 : ellipse x2 (k * (x2 - 4)))
  (h3 : line_through_focus k x1 (k * (x1 - 4)))
  (h4 : line_through_focus k x2 (k * (x2 - 4))) :
  lambda1 x1 + lambda2 x2 = -50 / 9 := 
sorry

end lambda_sum_ellipse_l89_89524


namespace min_value_of_f_l89_89623

-- Define the problem domain: positive real numbers
variables (a b c x y z : ℝ)
variables (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0)
variables (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0)

-- Define the given equations
variables (h1 : c * y + b * z = a)
variables (h2 : a * z + c * x = b)
variables (h3 : b * x + a * y = c)

-- Define the function f(x, y, z)
noncomputable def f (x y z : ℝ) : ℝ :=
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)

-- The theorem statement: under the given conditions the minimum value of f(x, y, z) is 1/2
theorem min_value_of_f :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    c * y + b * z = a →
    a * z + c * x = b →
    b * x + a * y = c →
    f x y z = 1 / 2) :=
sorry

end min_value_of_f_l89_89623


namespace fraction_simplification_l89_89222

noncomputable def x : ℚ := 0.714714714 -- Repeating decimal representation for x
noncomputable def y : ℚ := 2.857857857 -- Repeating decimal representation for y

theorem fraction_simplification :
  (x / y) = (714 / 2855) :=
by
  sorry

end fraction_simplification_l89_89222


namespace solution_set_of_inequality_system_l89_89299

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by
  sorry

end solution_set_of_inequality_system_l89_89299


namespace probability_P8_equals_P_l89_89930

theorem probability_P8_equals_P : 
  let k_even_combinations := [Finset.card (Finset.filter (λ k, k % 2 = 0) (Finset.range 9))],
  let total_sequences := 4 ^ 8,
  P_8_eq_P := (Finset.sum k_even_combinations (λ k, Nat.binomial 8 k ^ 2)),
  P_8_eq_P / total_sequences = 1225 / 16384 :=
by
  let k_even_combinations := List.map (λ k, Nat.choose 8 k * Nat.choose 8 k) [0, 2, 4, 6, 8]
  let valid_sequences := List.sum k_even_combinations
  let total_sequences := 4 ^ 8
  have h : valid_sequences = 6470 := by norm_num -- sum of squared binomials
  have h2 : total_sequences = 65536 := by norm_num -- 4^8
  have P_8_eq_P : (valid_sequences : ℚ) / (total_sequences : ℚ) = 1225 / 16384 := by
    rw [h, h2]
    norm_num
  exact P_8_eq_P

end probability_P8_equals_P_l89_89930


namespace cameras_not_in_both_l89_89879

-- Definitions for the given conditions
def shared_cameras : ℕ := 12
def sarah_cameras : ℕ := 24
def mike_unique_cameras : ℕ := 9

-- The proof statement
theorem cameras_not_in_both : (sarah_cameras - shared_cameras) + mike_unique_cameras = 21 := by
  sorry

end cameras_not_in_both_l89_89879


namespace find_functions_satisfying_lcm_gcd_eq_l89_89508

noncomputable def satisfies_functional_equation (f : ℕ → ℕ) : Prop := 
  ∀ m n : ℕ, m > 0 ∧ n > 0 → f (m * n) = Nat.lcm m n * Nat.gcd (f m) (f n)

noncomputable def solution_form (f : ℕ → ℕ) : Prop := 
  ∃ k : ℕ, ∀ x : ℕ, f x = k * x

theorem find_functions_satisfying_lcm_gcd_eq (f : ℕ → ℕ) : 
  satisfies_functional_equation f ↔ solution_form f := 
sorry

end find_functions_satisfying_lcm_gcd_eq_l89_89508


namespace unique_solution_7tuples_l89_89510

theorem unique_solution_7tuples : 
  ∃! (x : Fin 7 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1/8 :=
sorry

end unique_solution_7tuples_l89_89510


namespace increase_in_average_weight_l89_89549

variable (A : ℝ)

theorem increase_in_average_weight (h1 : ∀ (A : ℝ), 4 * A - 65 + 71 = 4 * (A + 1.5)) :
  (71 - 65) / 4 = 1.5 :=
by
  sorry

end increase_in_average_weight_l89_89549


namespace mph_to_fps_l89_89628

theorem mph_to_fps (C G : ℝ) (x : ℝ) (hC : C = 60 * x) (hG : G = 40 * x) (h1 : 7 * C - 7 * G = 210) :
  x = 1.5 :=
by {
  -- Math proof here, but we insert sorry for now
  sorry
}

end mph_to_fps_l89_89628


namespace trader_sells_cloth_l89_89337

theorem trader_sells_cloth
  (total_SP : ℝ := 4950)
  (profit_per_meter : ℝ := 15)
  (cost_price_per_meter : ℝ := 51)
  (SP_per_meter : ℝ := cost_price_per_meter + profit_per_meter)
  (x : ℝ := total_SP / SP_per_meter) :
  x = 75 :=
by
  sorry

end trader_sells_cloth_l89_89337


namespace product_range_l89_89091

theorem product_range (m b : ℚ) (h₀ : m = 3 / 4) (h₁ : b = 6 / 5) : 0 < m * b ∧ m * b < 1 :=
by
  sorry

end product_range_l89_89091


namespace sum_coefficients_l89_89387

theorem sum_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℚ) :
  (1 - 2 * (1 : ℚ))^5 = a_0 + a_1 * (1 : ℚ) + a_2 * (1 : ℚ)^2 + a_3 * (1 : ℚ)^3 + a_4 * (1 : ℚ)^4 + a_5 * (1 : ℚ)^5 →
  (1 - 2 * (0 : ℚ))^5 = a_0 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -2 :=
by
  sorry

end sum_coefficients_l89_89387


namespace xyz_sum_neg1_l89_89528

theorem xyz_sum_neg1 (x y z : ℝ) (h : (x + 1)^2 + |y - 2| = -(2 * x - z)^2) : x + y + z = -1 :=
sorry

end xyz_sum_neg1_l89_89528


namespace prob_white_or_black_l89_89483

-- Defining the problem conditions
def total_balls := 5
def white_and_black := 2
def draw_balls := 3

theorem prob_white_or_black:
  let total_combinations := Nat.choose total_balls draw_balls
  let favorable_combinations := total_combinations - 1  -- all three drawn balls are of the remaining three colors
  let probability := favorable_combinations / total_combinations
  probability = 9 / 10 :=
by
  -- skip the proof
  sorry

end prob_white_or_black_l89_89483


namespace sufficient_but_not_necessary_l89_89827

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2) :=
by
  sorry

end sufficient_but_not_necessary_l89_89827


namespace nickels_left_l89_89734

theorem nickels_left (n b : ℕ) (h₁ : n = 31) (h₂ : b = 20) : n - b = 11 :=
by
  sorry

end nickels_left_l89_89734


namespace total_rings_is_19_l89_89939

-- Definitions based on the problem conditions
def rings_on_first_day : Nat := 8
def rings_on_second_day : Nat := 6
def rings_on_third_day : Nat := 5

-- Total rings calculation
def total_rings : Nat := rings_on_first_day + rings_on_second_day + rings_on_third_day

-- Proof statement
theorem total_rings_is_19 : total_rings = 19 := by
  -- Proof goes here
  sorry

end total_rings_is_19_l89_89939


namespace olivia_savings_l89_89163

noncomputable def compound_amount 
  (P : ℝ) -- Initial principal
  (r : ℝ) -- Annual interest rate
  (n : ℕ) -- Number of times interest is compounded per year
  (t : ℕ) -- Number of years
  : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem olivia_savings :
  compound_amount 2500 0.045 2 21 = 5077.14 :=
by
  sorry

end olivia_savings_l89_89163


namespace volume_of_reservoir_proof_relationship_Q_t_proof_min_hourly_drainage_proof_min_time_to_drain_proof_l89_89787

noncomputable def volume_of_reservoir (drain_rate : ℝ) (time_to_drain : ℝ) : ℝ :=
  drain_rate * time_to_drain

theorem volume_of_reservoir_proof :
  volume_of_reservoir 8 6 = 48 :=
by
  sorry

noncomputable def relationship_Q_t (volume : ℝ) (t : ℝ) : ℝ :=
  volume / t

theorem relationship_Q_t_proof :
  ∀ (t : ℝ), relationship_Q_t 48 t = 48 / t :=
by
  intro t
  sorry

noncomputable def min_hourly_drainage (volume : ℝ) (time : ℝ) : ℝ :=
  volume / time

theorem min_hourly_drainage_proof :
  min_hourly_drainage 48 5 = 9.6 :=
by
  sorry

theorem min_time_to_drain_proof :
  ∀ (max_capacity : ℝ), relationship_Q_t 48 max_capacity = 12 → 48 / 12 = 4 :=
by
  intro max_capacity h
  sorry

end volume_of_reservoir_proof_relationship_Q_t_proof_min_hourly_drainage_proof_min_time_to_drain_proof_l89_89787


namespace cos_C_value_l89_89982

-- Definitions for the perimeter and sine ratios
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (perimeter : ℝ) (sin_ratio_A sin_ratio_B sin_ratio_C : ℚ)

-- Given conditions
axiom perimeter_condition : perimeter = a + b + c
axiom sine_ratio_condition : (sin_ratio_A / sin_ratio_B / sin_ratio_C) = (3 / 2 / 4)
axiom side_lengths : a = 3 ∧ b = 2 ∧ c = 4

-- To prove

theorem cos_C_value (h1 : sine_ratio_A = 3) (h2 : sine_ratio_B = 2) (h3 : sin_ratio_C = 4) :
  (3^2 + 2^2 - 4^2) / (2 * 3 * 2) = -1 / 4 :=
sorry

end cos_C_value_l89_89982


namespace triangle_min_perimeter_l89_89748

theorem triangle_min_perimeter:
  ∃ x : ℤ, 27 < x ∧ x < 75 ∧ (24 + 51 + x) = 103 :=
begin
  sorry
end

end triangle_min_perimeter_l89_89748


namespace planes_perpendicular_of_line_conditions_l89_89280

variables (a b l : Line) (M N : Plane)

-- Definitions of lines and planes and their relations
def parallel_to_plane (a : Line) (M : Plane) : Prop := sorry
def perpendicular_to_plane (a : Line) (M : Plane) : Prop := sorry
def subset_of_plane (a : Line) (M : Plane) : Prop := sorry

-- Statement of the main theorem to be proved
theorem planes_perpendicular_of_line_conditions (a b l : Line) (M N : Plane) :
  (perpendicular_to_plane a M) → (parallel_to_plane a N) → (perpendicular_to_plane N M) :=
  by
  sorry

end planes_perpendicular_of_line_conditions_l89_89280


namespace pet_store_satisfaction_l89_89189

theorem pet_store_satisfaction :
  let puppies := 15
  let kittens := 6
  let hamsters := 8
  let friends := 3
  puppies * kittens * hamsters * friends.factorial = 4320 := by
  sorry

end pet_store_satisfaction_l89_89189


namespace cos_pi_minus_alpha_l89_89832

open Real

variable (α : ℝ)

theorem cos_pi_minus_alpha (h1 : 0 < α ∧ α < π / 2) (h2 : sin α = 4 / 5) : cos (π - α) = -3 / 5 := by
  sorry

end cos_pi_minus_alpha_l89_89832


namespace tom_teaching_years_l89_89601

def years_tom_has_been_teaching (x : ℝ) : Prop :=
  x + (1/2 * x - 5) = 70

theorem tom_teaching_years:
  ∃ x : ℝ, years_tom_has_been_teaching x ∧ x = 50 :=
by
  sorry

end tom_teaching_years_l89_89601


namespace union_A_B_l89_89715

def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 2}

theorem union_A_B : A ∪ B = {1, 2, 3} := 
by
  sorry

end union_A_B_l89_89715


namespace total_stops_traveled_l89_89117

-- Definitions based on the conditions provided
def yoojeong_stops : ℕ := 3
def namjoon_stops : ℕ := 2

-- Theorem statement to prove the total number of stops
theorem total_stops_traveled : yoojeong_stops + namjoon_stops = 5 := by
  -- Proof omitted
  sorry

end total_stops_traveled_l89_89117


namespace awards_distribution_count_l89_89230

-- Define the problem conditions
def num_awards : Nat := 5
def num_students : Nat := 3

-- Verify each student gets at least one award
def each_student_gets_at_least_one (distributions : List (List Nat)) : Prop :=
  ∀ (dist : List Nat), dist ∈ distributions → (∀ (d : Nat), d > 0)

-- Define the main theorem to be proved
theorem awards_distribution_count :
  ∃ (distributions : List (List Nat)), each_student_gets_at_least_one distributions ∧ distributions.length = 150 :=
sorry

end awards_distribution_count_l89_89230


namespace shaniqua_income_per_haircut_l89_89736

theorem shaniqua_income_per_haircut (H : ℝ) :
  (8 * H + 5 * 25 = 221) → (H = 12) :=
by
  intro h
  sorry

end shaniqua_income_per_haircut_l89_89736


namespace area_triangle_PTS_l89_89120

theorem area_triangle_PTS {PQ QR PS QT PT TS : ℝ} 
  (hPQ : PQ = 4) 
  (hQR : QR = 6) 
  (hPS : PS = 2 * Real.sqrt 13) 
  (hQT : QT = 12 * Real.sqrt 13 / 13) 
  (hPT : PT = 4) 
  (hTS : TS = (2 * Real.sqrt 13) - 4) : 
  (1 / 2) * PT * QT = 24 * Real.sqrt 13 / 13 := 
by 
  sorry

end area_triangle_PTS_l89_89120


namespace parallel_lines_slope_equality_l89_89411

theorem parallel_lines_slope_equality (m : ℝ) : (∀ x y : ℝ, 3 * x + y - 3 = 0) ∧ (∀ x y : ℝ, 6 * x + m * y + 1 = 0) → m = 2 :=
by 
  sorry

end parallel_lines_slope_equality_l89_89411


namespace height_of_tree_in_8_years_in_inches_l89_89896

theorem height_of_tree_in_8_years_in_inches 
  (initial_height : ℕ) (annual_growth : ℕ) (years : ℕ) (feet_to_inches : ℕ) 
  (h_initial_height : initial_height = 52) 
  (h_annual_growth : annual_growth = 5) 
  (h_years : years = 8) 
  (h_feet_to_inches : feet_to_inches = 12) : 
  let total_growth := annual_growth * years in
  let final_height_in_feet := initial_height + total_growth in
  final_height_in_feet * feet_to_inches = 1104 :=
by
  sorry

end height_of_tree_in_8_years_in_inches_l89_89896


namespace find_a_l89_89695

theorem find_a (a : ℝ) (h : ∃ (b : ℝ), (16 * (x : ℝ) * x) + 40 * x + a = (4 * x + b) ^ 2) : a = 25 := sorry

end find_a_l89_89695


namespace original_cost_price_l89_89433

-- Define the conditions
def selling_price : ℝ := 24000
def discount_rate : ℝ := 0.15
def tax_rate : ℝ := 0.02
def profit_rate : ℝ := 0.12

-- Define the necessary calculations
def discounted_price (sp : ℝ) (dr : ℝ) : ℝ := sp * (1 - dr)
def total_tax (sp : ℝ) (tr : ℝ) : ℝ := sp * tr
def profit (c : ℝ) (pr : ℝ) : ℝ := c * (1 + pr)

-- The problem is to prove that the original cost price is $17,785.71
theorem original_cost_price : 
  ∃ (C : ℝ), C = 17785.71 ∧ 
  selling_price * (1 - discount_rate - tax_rate) = (1 + profit_rate) * C :=
sorry

end original_cost_price_l89_89433


namespace mean_equality_l89_89294

-- Define average calculation function
def average (a b c : ℕ) : ℕ :=
  (a + b + c) / 3

def average_two (a b : ℕ) : ℕ :=
  (a + b) / 2

theorem mean_equality (x : ℕ) 
  (h : average 8 16 24 = average_two 10 x) : 
  x = 22 :=
by {
  -- The actual proof is here
  sorry
}

end mean_equality_l89_89294


namespace solve_x_l89_89124

theorem solve_x (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 8 * x ^ 2 + 16 * x * y = x ^ 3 + 3 * x ^ 2 * y) (h₄ : y = 2 * x) : x = 40 / 7 :=
by
  sorry

end solve_x_l89_89124


namespace magnitude_correct_l89_89742

open Real

noncomputable def magnitude_of_vector_addition
  (a b : ℝ × ℝ)
  (theta : ℝ)
  (ha : a = (1, 1))
  (hb : ‖b‖ = 2)
  (h_angle : theta = π / 4) : ℝ :=
  ‖3 • a + b‖

theorem magnitude_correct 
  (a b : ℝ × ℝ)
  (theta : ℝ)
  (ha : a = (1, 1))
  (hb : ‖b‖ = 2)
  (h_angle : theta = π / 4) :
  magnitude_of_vector_addition a b theta ha hb h_angle = sqrt 34 :=
sorry

end magnitude_correct_l89_89742


namespace mango_coconut_ratio_l89_89877

open Function

theorem mango_coconut_ratio
  (mango_trees : ℕ)
  (coconut_trees : ℕ)
  (total_trees : ℕ)
  (R : ℚ)
  (H1 : mango_trees = 60)
  (H2 : coconut_trees = R * 60 - 5)
  (H3 : total_trees = 85)
  (H4 : total_trees = mango_trees + coconut_trees) :
  R = 1/2 :=
by
  sorry

end mango_coconut_ratio_l89_89877


namespace algebra_problem_l89_89847

theorem algebra_problem (a b c d x : ℝ) (h1 : a = -b) (h2 : c * d = 1) (h3 : |x| = 3) : 
  (a + b) / 2023 + c * d - x^2 = -8 := by
  sorry

end algebra_problem_l89_89847


namespace teacher_drank_milk_false_l89_89457

-- Define the condition that the volume of milk a teacher can reasonably drink in a day is more appropriately measured in milliliters rather than liters.
def reasonable_volume_units := "milliliters"

-- Define the statement to be judged
def teacher_milk_intake := 250

-- Define the unit of the statement
def unit_of_statement := "liters"

-- The proof goal is to conclude that the statement "The teacher drank 250 liters of milk today" is false, given the condition on volume units.
theorem teacher_drank_milk_false (vol : ℕ) (unit : String) (reasonable_units : String) :
  vol = 250 ∧ unit = "liters" ∧ reasonable_units = "milliliters" → false :=
by
  sorry

end teacher_drank_milk_false_l89_89457


namespace cover_points_with_circles_l89_89491

theorem cover_points_with_circles (n : ℕ) (points : Fin n → ℝ × ℝ)
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → min (dist (points i) (points j)) (min (dist (points j) (points k)) (dist (points i) (points k))) ≤ 1) :
  ∃ (a b : Fin n), ∀ (p : Fin n), dist (points p) (points a) ≤ 1 ∨ dist (points p) (points b) ≤ 1 := 
sorry

end cover_points_with_circles_l89_89491


namespace min_value_of_expression_l89_89546

theorem min_value_of_expression (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y = 5 := 
sorry

end min_value_of_expression_l89_89546


namespace absolute_value_inequality_solution_l89_89283

theorem absolute_value_inequality_solution (x : ℝ) :
  abs ((3 * x + 2) / (x + 2)) > 3 ↔ (x < -2) ∨ (-2 < x ∧ x < -4 / 3) :=
by
  sorry

end absolute_value_inequality_solution_l89_89283


namespace num_pairs_satisfying_eq_l89_89819

theorem num_pairs_satisfying_eq :
  ∃ n : ℕ, (n = 256) ∧ (∀ x y : ℤ, x^2 + x * y = 30000000 → true) :=
sorry

end num_pairs_satisfying_eq_l89_89819


namespace pounds_of_oranges_l89_89184

noncomputable def price_of_pounds_oranges (E O : ℝ) (P : ℕ) : Prop :=
  let current_total_price := E
  let increased_total_price := 1.09 * E + 1.06 * (O * P)
  (increased_total_price - current_total_price) = 15

theorem pounds_of_oranges (E O : ℝ) (P : ℕ): 
  E = O * P ∧ 
  (price_of_pounds_oranges E O P) → 
  P = 100 := 
by
  sorry

end pounds_of_oranges_l89_89184


namespace least_three_digit_7_heavy_l89_89341

-- Define what it means for a number to be "7-heavy"
def is_7_heavy(n : ℕ) : Prop := n % 7 > 4

-- Smallest three-digit number
def smallest_three_digit_number : ℕ := 100

-- Least three-digit 7-heavy whole number
theorem least_three_digit_7_heavy : ∃ n, smallest_three_digit_number ≤ n ∧ is_7_heavy(n) ∧ ∀ m, smallest_three_digit_number ≤ m ∧ is_7_heavy(m) → n ≤ m := 
  sorry

end least_three_digit_7_heavy_l89_89341


namespace pipe_tank_overflow_l89_89724

theorem pipe_tank_overflow (t : ℕ) :
  let rateA := 1 / 30
  let rateB := 1 / 60
  let combined_rate := rateA + rateB
  let workA := rateA * (t - 15)
  let workB := rateB * t
  (workA + workB = 1) ↔ (t = 25) := by
  sorry

end pipe_tank_overflow_l89_89724


namespace perception_num_permutations_l89_89356

def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def perception_arrangements : ℕ :=
  let total_letters := 10
  let repetitions_P := 2
  let repetitions_E := 2
  factorial total_letters / (factorial repetitions_P * factorial repetitions_E)

theorem perception_num_permutations :
  perception_arrangements = 907200 :=
by sorry

end perception_num_permutations_l89_89356


namespace least_possible_perimeter_l89_89749

/-- Proof that the least possible perimeter of a triangle with two sides of length 24 and 51 units,
    and the third side being an integer, is 103 units. -/
theorem least_possible_perimeter (a b : ℕ) (c : ℕ) (h1 : a = 24) (h2 : b = 51) (h3 : c > 27) (h4 : c < 75) :
    a + b + c = 103 :=
by
  sorry

end least_possible_perimeter_l89_89749


namespace trig_expression_l89_89059

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
  (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7 :=
by
  sorry

end trig_expression_l89_89059


namespace intersection_in_fourth_quadrant_l89_89010

variable {a : ℝ} {x : ℝ}

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x / Real.log a
noncomputable def g (x : ℝ) (a : ℝ) := (1 - a) * x

theorem intersection_in_fourth_quadrant (h : a > 1) :
  ∃ x : ℝ, x > 0 ∧ f x a < 0 ∧ f x a = g x a :=
sorry

end intersection_in_fourth_quadrant_l89_89010


namespace line_equation_l89_89447

-- Define the conditions: point (2,1) on the line and slope is 2
def point_on_line (x y : ℝ) (m b : ℝ) : Prop := y = m * x + b

def slope_of_line (m : ℝ) : Prop := m = 2

-- Prove the equation of the line is 2x - y - 3 = 0
theorem line_equation (b : ℝ) (h1 : point_on_line 2 1 2 b) : 2 * 2 - 1 - 3 = 0 := by
  sorry

end line_equation_l89_89447


namespace exists_k_with_three_different_real_roots_exists_k_with_two_different_real_roots_l89_89824

noncomputable def equation (x : ℝ) (k : ℝ) := x^2 - 2 * |x| - (2 * k + 1)^2

theorem exists_k_with_three_different_real_roots :
  ∃ k : ℝ, ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ equation x1 k = 0 ∧ equation x2 k = 0 ∧ equation x3 k = 0 :=
sorry

theorem exists_k_with_two_different_real_roots :
  ∃ k : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ equation x1 k = 0 ∧ equation x2 k = 0 :=
sorry

end exists_k_with_three_different_real_roots_exists_k_with_two_different_real_roots_l89_89824


namespace apples_given_by_nathan_l89_89643

theorem apples_given_by_nathan (initial_apples : ℕ) (total_apples : ℕ) (given_by_nathan : ℕ) :
  initial_apples = 6 → total_apples = 12 → given_by_nathan = (total_apples - initial_apples) → given_by_nathan = 6 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end apples_given_by_nathan_l89_89643


namespace riding_owners_ratio_l89_89784

theorem riding_owners_ratio :
  ∃ (R W : ℕ), (R + W = 16) ∧ (4 * R + 6 * W = 80) ∧ (R : ℚ) / 16 = 1/2 :=
by
  sorry

end riding_owners_ratio_l89_89784


namespace person_a_age_l89_89456

theorem person_a_age (A B : ℕ) (h1 : A + B = 43) (h2 : A + 4 = B + 7) : A = 23 :=
by sorry

end person_a_age_l89_89456


namespace sum_problem3_equals_50_l89_89203

-- Assume problem3_condition is a placeholder for the actual conditions described in problem 3
-- and sum_problem3 is a placeholder for the sum of elements described in problem 3.

axiom problem3_condition : Prop
axiom sum_problem3 : ℕ

theorem sum_problem3_equals_50 (h : problem3_condition) : sum_problem3 = 50 :=
sorry

end sum_problem3_equals_50_l89_89203


namespace fractional_part_wall_in_12_minutes_l89_89252

-- Definitions based on given conditions
def time_to_paint_wall : ℕ := 60
def time_spent_painting : ℕ := 12

-- The goal is to prove that the fraction of the wall Mark can paint in 12 minutes is 1/5
theorem fractional_part_wall_in_12_minutes (t_pw: ℕ) (t_sp: ℕ) (h1: t_pw = 60) (h2: t_sp = 12) : 
  (t_sp : ℚ) / (t_pw : ℚ) = 1 / 5 :=
by 
  sorry

end fractional_part_wall_in_12_minutes_l89_89252


namespace beavers_working_on_home_l89_89778

noncomputable def initial_beavers : ℝ := 2.0
noncomputable def additional_beavers : ℝ := 1.0

theorem beavers_working_on_home : initial_beavers + additional_beavers = 3.0 :=
by
  sorry

end beavers_working_on_home_l89_89778


namespace percent_of_y_equal_to_30_percent_of_60_percent_l89_89161

variable (y : ℝ)

theorem percent_of_y_equal_to_30_percent_of_60_percent (hy : y ≠ 0) :
  ((0.18 * y) / y) * 100 = 18 := by
  have hneq : y ≠ 0 := hy
  field_simp [hneq]
  norm_num
  sorry

end percent_of_y_equal_to_30_percent_of_60_percent_l89_89161


namespace solve_sqrt_equation_l89_89813

theorem solve_sqrt_equation (x : ℝ) (hx : x ≥ 2) :
  (\sqrt(x + 5 - 6 * \sqrt(x - 2)) + \sqrt(x + 12 - 8 * \sqrt(x - 2)) = 2) ↔ (x = 11 ∨ x = 27) :=
by sorry

end solve_sqrt_equation_l89_89813


namespace even_and_period_pi_l89_89243

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem even_and_period_pi :
  (∀ x : ℝ, f (-x) = f x) ∧ (∃ T > 0, ∀ x : ℝ, f (x + T) = f x) ∧ T = Real.pi :=
by
  -- First, prove that f(x) is an even function: ∀ x, f(-x) = f(x)
  -- Next, find the smallest positive period T: ∃ T > 0, ∀ x, f(x + T) = f(x)
  -- Finally, show that this period is pi: T = π
  sorry

end even_and_period_pi_l89_89243


namespace sequence_value_x_l89_89706

theorem sequence_value_x (a1 a2 a3 a4 a5 a6 : ℕ) 
  (h1 : a1 = 2) 
  (h2 : a2 = 5) 
  (h3 : a3 = 11) 
  (h4 : a4 = 20) 
  (h5 : a6 = 47)
  (h6 : a2 - a1 = 3) 
  (h7 : a3 - a2 = 6) 
  (h8 : a4 - a3 = 9) 
  (h9 : a6 - a5 = 15) : 
  a5 = 32 :=
sorry

end sequence_value_x_l89_89706


namespace simplify_expression_correct_l89_89737

def simplify_expression (x : ℝ) : Prop :=
  (5 - 2 * x) - (7 + 3 * x) = -2 - 5 * x

theorem simplify_expression_correct (x : ℝ) : simplify_expression x :=
  by
    sorry

end simplify_expression_correct_l89_89737


namespace discount_correct_l89_89444

variable {a : ℝ} (discount_percent : ℝ) (profit_percent : ℝ → ℝ)

noncomputable def calc_discount : ℝ :=
  discount_percent

theorem discount_correct :
  (discount_percent / 100) = (33 + 1 / 3) / 100 →
  profit_percent (discount_percent / 100) = (3 / 2) * (discount_percent / 100) →
  a * (1 - discount_percent / 100) * (1 + profit_percent (discount_percent / 100)) = a →
  discount_percent = 33 + 1 / 3 :=
by sorry

end discount_correct_l89_89444


namespace gas_cost_per_gallon_is_4_l89_89349

noncomputable def cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (total_miles / miles_per_gallon)

theorem gas_cost_per_gallon_is_4 :
  cost_per_gallon 32 432 54 = 4 := by
  sorry

end gas_cost_per_gallon_is_4_l89_89349


namespace novel_to_history_ratio_l89_89590

-- Define the conditions
def history_book_pages : ℕ := 300
def science_book_pages : ℕ := 600
def novel_pages := science_book_pages / 4

-- Define the target ratio to prove
def target_ratio := (novel_pages : ℚ) / (history_book_pages : ℚ)

theorem novel_to_history_ratio :
  target_ratio = (1 : ℚ) / (2 : ℚ) :=
by
  sorry

end novel_to_history_ratio_l89_89590


namespace dodecahedron_path_count_l89_89636

/-- A regular dodecahedron with constraints on movement between faces. -/
def num_ways_dodecahedron_move : Nat := 810

/-- Proving the number of different ways to move from the top face to the bottom face of a regular dodecahedron via a series of adjacent faces, such that each face is visited at most once, and movement from the lower ring to the upper ring is not allowed is 810. -/
theorem dodecahedron_path_count :
  num_ways_dodecahedron_move = 810 :=
by
  -- Proof goes here
  sorry

end dodecahedron_path_count_l89_89636


namespace percent_calculation_l89_89149

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l89_89149


namespace sum_of_positive_differences_eq_787484_l89_89260

open Finset

def S : Finset ℕ := (range 11).image (λ x, 3^x)

theorem sum_of_positive_differences_eq_787484 : 
  let differences := S.product S.filter (λ p, p.1 < p.2) in
  let positive_diffs := differences.map (λ p, p.2 - p.1) in
  positive_diffs.sum = 787484 :=
by
  sorry

end sum_of_positive_differences_eq_787484_l89_89260


namespace functional_inequality_solution_l89_89372

theorem functional_inequality_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, 2 + f x * f y ≤ x * y + 2 * f (x + y + 1)) ↔ (∀ x : ℝ, f x = x + 2) :=
by
  sorry

end functional_inequality_solution_l89_89372


namespace minimum_number_of_apples_l89_89502

-- Define the problem conditions and the proof statement
theorem minimum_number_of_apples :
  ∃ p : Fin 6 → ℕ, (∀ i, p i > 0) ∧ (Function.Injective p) ∧ (Finset.univ.sum p * 4 = 100) ∧ (Finset.univ.sum p = 25 / 4) := 
sorry

end minimum_number_of_apples_l89_89502


namespace find_x_l89_89914

theorem find_x (x : ℝ) (h : 3 * x = 36 - x + 16) : x = 13 :=
by
  sorry

end find_x_l89_89914


namespace total_amount_shared_l89_89353

theorem total_amount_shared (total_amount : ℝ) 
  (h_debby : total_amount * 0.25 = (total_amount - 4500))
  (h_maggie : total_amount * 0.75 = 4500) : total_amount = 6000 :=
begin
  sorry
end

end total_amount_shared_l89_89353


namespace total_fish_correct_l89_89656

def Billy_fish : ℕ := 10
def Tony_fish : ℕ := 3 * Billy_fish
def Sarah_fish : ℕ := Tony_fish + 5
def Bobby_fish : ℕ := 2 * Sarah_fish
def Jenny_fish : ℕ := Bobby_fish - 4
def total_fish : ℕ := Billy_fish + Tony_fish + Sarah_fish + Bobby_fish + Jenny_fish

theorem total_fish_correct : total_fish = 211 := by
  sorry

end total_fish_correct_l89_89656


namespace sum_consecutive_even_integers_l89_89584

theorem sum_consecutive_even_integers (m : ℤ) :
  (m + (m + 2) + (m + 4) + (m + 6) + (m + 8)) = 5 * m + 20 := by
  sorry

end sum_consecutive_even_integers_l89_89584


namespace number_of_cups_needed_to_fill_container_l89_89183

theorem number_of_cups_needed_to_fill_container (container_capacity cup_capacity : ℕ) (h1 : container_capacity = 640) (h2 : cup_capacity = 120) : 
  (container_capacity + cup_capacity - 1) / cup_capacity = 6 :=
by
  sorry

end number_of_cups_needed_to_fill_container_l89_89183


namespace find_f_inv_value_l89_89063

noncomputable def f (x : ℝ) : ℝ := 8^x
noncomputable def f_inv (y : ℝ) : ℝ := Real.logb 8 y

theorem find_f_inv_value (a : ℝ) (h : a = 8^(1/3)) : f_inv (a + 2) = Real.logb 8 (8^(1/3) + 2) := by
  sorry

end find_f_inv_value_l89_89063


namespace yuna_has_most_apples_l89_89864

def apples_count_jungkook : ℕ :=
  6 / 3

def apples_count_yoongi : ℕ :=
  4

def apples_count_yuna : ℕ :=
  5

theorem yuna_has_most_apples : apples_count_yuna > apples_count_yoongi ∧ apples_count_yuna > apples_count_jungkook :=
by
  sorry

end yuna_has_most_apples_l89_89864


namespace coin_value_difference_l89_89731

theorem coin_value_difference (p n d : ℕ) (h : p + n + d = 3000) (hp : p ≥ 1) (hn : n ≥ 1) (hd : d ≥ 1) : 
  (p + 5 * n + 10 * d).max - (p + 5 * n + 10 * d).min = 26973 := 
sorry

end coin_value_difference_l89_89731


namespace number_of_swaps_independent_l89_89459

theorem number_of_swaps_independent (n : ℕ) (hn : n = 20) (p : Fin n → Fin n) :
    (∀ i, p i ≠ i → ∃ j, p j ≠ j ∧ p (p j) = j) →
    ∃ s : List (Fin n × Fin n), List.length s ≤ n ∧
    (∀ σ : List (Fin n × Fin n), (∀ (i j : Fin n), (i, j) ∈ σ → p i ≠ i → ∃ p', σ = (i, p') :: (p', j) :: σ) →
     List.length σ = List.length s) :=
  sorry

end number_of_swaps_independent_l89_89459


namespace solve_for_w_l89_89282

theorem solve_for_w (w : ℂ) (i : ℂ) (i_squared : i^2 = -1) 
  (h : 3 - i * w = 1 + 2 * i * w) : 
  w = -2 * i / 3 := 
sorry

end solve_for_w_l89_89282


namespace no_solutions_in_naturals_l89_89583

theorem no_solutions_in_naturals (n k : ℕ) : ¬ (n ≤ n! - k^n ∧ n! - k^n ≤ k * n) :=
sorry

end no_solutions_in_naturals_l89_89583


namespace bacteria_growth_l89_89891

theorem bacteria_growth (d : ℕ) (t : ℕ) (initial final : ℕ) 
  (h_doubling : d = 4) 
  (h_initial : initial = 500) 
  (h_final : final = 32000) 
  (h_ratio : final / initial = 2^6) :
  t = d * 6 → t = 24 :=
by
  sorry

end bacteria_growth_l89_89891


namespace kiera_fruit_cups_l89_89385

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def total_cost : ℕ := 17

theorem kiera_fruit_cups : ∃ kiera_fruit_cups : ℕ, muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cups = total_cost - (muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups) :=
by
  let francis_cost := muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  let remaining_cost := total_cost - francis_cost
  let kiera_fruit_cups := remaining_cost / fruit_cup_cost
  exact ⟨kiera_fruit_cups, by sorry⟩

end kiera_fruit_cups_l89_89385


namespace smallest_x_l89_89455

theorem smallest_x (M x : ℕ) (h : 720 * x = M^3) : x = 300 :=
by
  sorry

end smallest_x_l89_89455


namespace sufficient_but_not_necessary_l89_89522

theorem sufficient_but_not_necessary (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : a + b > 0 :=
by
  sorry

end sufficient_but_not_necessary_l89_89522


namespace sum_of_coeffs_l89_89130

theorem sum_of_coeffs 
  (a b c d e x : ℝ)
  (h : (729 * x ^ 3 + 8) = (a * x + b) * (c * x ^ 2 + d * x + e)) :
  a + b + c + d + e = 78 :=
sorry

end sum_of_coeffs_l89_89130


namespace square_field_area_l89_89375

/-- 
  Statement: Prove that the area of the square field is 69696 square meters 
  given that the wire goes around the square field 15 times and the total 
  length of the wire is 15840 meters.
-/
theorem square_field_area (rounds : ℕ) (total_length : ℕ) (area : ℕ) 
  (h1 : rounds = 15) (h2 : total_length = 15840) : 
  area = 69696 := 
by 
  sorry

end square_field_area_l89_89375


namespace aunt_gemma_dog_food_l89_89199

theorem aunt_gemma_dog_food :
  ∀ (dogs : ℕ) (grams_per_meal : ℕ) (meals_per_day : ℕ) (sack_kg : ℕ) (days : ℕ), 
    dogs = 4 →
    grams_per_meal = 250 →
    meals_per_day = 2 →
    sack_kg = 50 →
    days = 50 →
    (dogs * meals_per_day * grams_per_meal * days) / (1000 * sack_kg) = 2 :=
by
  intros dogs grams_per_meal meals_per_day sack_kg days
  intros h_dogs h_grams_per_meal h_meals_per_day h_sack_kg h_days
  sorry

end aunt_gemma_dog_food_l89_89199


namespace original_expenditure_l89_89320

theorem original_expenditure (initial_students new_students : ℕ) (increment_expense : ℝ) (decrement_avg_expense : ℝ) (original_avg_expense : ℝ) (new_avg_expense : ℝ) 
  (total_initial_expense original_expenditure : ℝ)
  (h1 : initial_students = 35) 
  (h2 : new_students = 7) 
  (h3 : increment_expense = 42)
  (h4 : decrement_avg_expense = 1)
  (h5 : new_avg_expense = original_avg_expense - decrement_avg_expense)
  (h6 : total_initial_expense = initial_students * original_avg_expense)
  (h7 : original_expenditure = total_initial_expense)
  (h8 : 42 * new_avg_expense - original_students * original_avg_expense = increment_expense) :
  original_expenditure = 420 := 
by
  sorry

end original_expenditure_l89_89320


namespace volleyball_not_basketball_l89_89997

def class_size : ℕ := 40
def basketball_enjoyers : ℕ := 15
def volleyball_enjoyers : ℕ := 20
def neither_sport : ℕ := 10

theorem volleyball_not_basketball :
  (volleyball_enjoyers - (basketball_enjoyers + volleyball_enjoyers - (class_size - neither_sport))) = 15 :=
by
  sorry

end volleyball_not_basketball_l89_89997


namespace solution_set_l89_89812

open Real

noncomputable def condition (x : ℝ) := x ≥ 2

noncomputable def eq_1 (x : ℝ) := sqrt (x + 5 - 6 * sqrt (x - 2)) + sqrt (x + 12 - 8 * sqrt (x - 2)) = 2

theorem solution_set :
  {x : ℝ | condition x ∧ eq_1 x} = {x : ℝ | 11 ≤ x ∧ x ≤ 18} :=
by sorry

end solution_set_l89_89812


namespace xiaoming_interview_pass_probability_l89_89783

theorem xiaoming_interview_pass_probability :
  let p_correct := 0.7
  let p_fail_per_attempt := 1 - p_correct
  let p_fail_all_attempts := p_fail_per_attempt ^ 3
  let p_pass_interview := 1 - p_fail_all_attempts
  p_pass_interview = 0.973 := by
    let p_correct := 0.7
    let p_fail_per_attempt := 1 - p_correct
    let p_fail_all_attempts := p_fail_per_attempt ^ 3
    let p_pass_interview := 1 - p_fail_all_attempts
    sorry

end xiaoming_interview_pass_probability_l89_89783


namespace probability_of_gui_field_in_za_field_l89_89125

noncomputable def area_gui_field (base height : ℕ) : ℚ :=
  (1 / 2 : ℚ) * base * height

noncomputable def area_za_field (small_base large_base height : ℕ) : ℚ :=
  (1 / 2 : ℚ) * (small_base + large_base) * height

theorem probability_of_gui_field_in_za_field :
  let b1 := 10
  let b2 := 20
  let h1 := 10
  let base_gui := 8
  let height_gui := 5
  let za_area := area_za_field b1 b2 h1
  let gui_area := area_gui_field base_gui height_gui
  (gui_area / za_area) = (2 / 15 : ℚ) := by
    sorry

end probability_of_gui_field_in_za_field_l89_89125


namespace painted_cube_faces_same_color_probability_eq_one_l89_89808

noncomputable def cube_face_color_probability : ℚ := 2 / 3

noncomputable def painted_cube_problem : ℚ :=
  let p_red := 2 / 3 in
  let p_blue := 1 / 3 in
  let p_three_faces_same_color := 1 in
  p_three_faces_same_color

theorem painted_cube_faces_same_color_probability_eq_one :
  painted_cube_problem = 1 :=
  sorry

end painted_cube_faces_same_color_probability_eq_one_l89_89808


namespace number_of_pencils_l89_89403

-- Define the given conditions
def circle_radius : ℝ := 14 -- 14 feet radius
def pencil_length_inches : ℝ := 6 -- 6-inch pencil

noncomputable def pencil_length_feet : ℝ := pencil_length_inches / 12 -- convert 6 inches to feet

-- Statement of the problem in Lean
theorem number_of_pencils (r : ℝ) (p_len_inch : ℝ) (d : ℝ) (p_len_feet : ℝ) :
  r = circle_radius →
  p_len_inch = pencil_length_inches →
  d = 2 * r →
  p_len_feet = pencil_length_feet →
  d / p_len_feet = 56 :=
by
  intros hr hp hd hpl
  sorry

end number_of_pencils_l89_89403


namespace tangent_line_equation_l89_89287

noncomputable def curve (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

def point := (-1 : ℝ, -3 : ℝ)

theorem tangent_line_equation :
  ∃ m b : ℝ, (m = 5 ∧ b = 2) ∧ (∀ x : ℝ, ∀ y : ℝ, (y = curve x → point = (x, y) → 5 * x - y + 2 = 0)) :=
sorry

end tangent_line_equation_l89_89287


namespace range_a_sub_b_mul_c_l89_89530

theorem range_a_sub_b_mul_c (a b c : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) (h4 : 2 < c) (h5 : c < 3) :
  -6 < (a - b) * c ∧ (a - b) * c < 0 :=
by
  -- We need to prove the range of (a - b) * c is within (-6, 0)
  sorry

end range_a_sub_b_mul_c_l89_89530


namespace log_27_gt_point_53_l89_89775

open Real

theorem log_27_gt_point_53 :
  log 27 > 0.53 :=
by
  sorry

end log_27_gt_point_53_l89_89775


namespace martha_apples_l89_89270

theorem martha_apples (martha_initial_apples : ℕ) (jane_apples : ℕ) 
  (james_additional_apples : ℕ) (target_martha_apples : ℕ) :
  martha_initial_apples = 20 →
  jane_apples = 5 →
  james_additional_apples = 2 →
  target_martha_apples = 4 →
  (let james_apples := jane_apples + james_additional_apples in
   let martha_remaining_apples := martha_initial_apples - jane_apples - james_apples in
   martha_remaining_apples - target_martha_apples = 4) :=
begin
  sorry
end

end martha_apples_l89_89270


namespace no_solution_for_inequalities_l89_89691

theorem no_solution_for_inequalities (m : ℝ) :
  (∀ x : ℝ, x - m ≤ 2 * m + 3 ∧ (x - 1) / 2 ≥ m → false) ↔ m < -2 :=
by
  sorry

end no_solution_for_inequalities_l89_89691


namespace breadth_increase_25_percent_l89_89745

variable (L B : ℝ) 

-- Conditions
def original_area := L * B
def increased_length := 1.10 * L
def increased_area := 1.375 * (original_area L B)

-- The breadth increase percentage (to be proven as 25)
def percentage_increase_breadth (p : ℝ) := 
  increased_area L B = increased_length L * (B * (1 + p/100))

-- The statement to be proven
theorem breadth_increase_25_percent : 
  percentage_increase_breadth L B 25 := 
sorry

end breadth_increase_25_percent_l89_89745


namespace range_of_x_l89_89828

theorem range_of_x (a b x : ℝ) (h1 : a + b = 1) (h2 : 0 < a) (h3 : 0 < b) :
  (1 / a + 4 / b ≥ |2 * x - 1| - |x + 1|) → (-7 ≤ x ∧ x ≤ 11) :=
by
  -- we provide the exact statement we aim to prove.
  sorry

end range_of_x_l89_89828


namespace least_possible_perimeter_l89_89752

theorem least_possible_perimeter (x : ℕ) (h1 : 27 < x) (h2 : x < 75) :
  24 + 51 + x = 103 :=
by
  sorry

end least_possible_perimeter_l89_89752


namespace transformation_thinking_reflected_in_solution_of_quadratic_l89_89614

theorem transformation_thinking_reflected_in_solution_of_quadratic :
  ∀ (x : ℝ), (x - 3)^2 - 5 * (x - 3) = 0 → (x = 3 ∨ x = 8) →
  transformation_thinking :=
by
  intros x h_eq h_solutions
  sorry

end transformation_thinking_reflected_in_solution_of_quadratic_l89_89614


namespace sequence_general_term_l89_89987

theorem sequence_general_term (a : ℕ → ℚ) (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n+1) = (n * a n + 2 * (n+1)^2) / (n+2)) :
  ∀ n : ℕ, a n = (1 / 2 : ℚ) * n * (n + 1) := by
  sorry

end sequence_general_term_l89_89987


namespace shared_total_l89_89350

theorem shared_total (total_amount : ℝ) (maggie_share : ℝ) (debby_percentage : ℝ)
  (h1 : debby_percentage = 0.25)
  (h2 : maggie_share = 4500)
  (h3 : maggie_share = (1 - debby_percentage) * total_amount) :
  total_amount = 6000 :=
by
  sorry

end shared_total_l89_89350


namespace number_of_female_officers_l89_89477

theorem number_of_female_officers (h1 : 0.19 * T = 76) (h2 : T = 152 / 2) : T = 400 :=
by
  sorry

end number_of_female_officers_l89_89477


namespace triangle_area_from_curve_l89_89843

-- Definition of the curve
def curve (x : ℝ) : ℝ := (x - 2)^2 * (x + 3)

-- Area calculation based on intercepts
theorem triangle_area_from_curve : 
  (1 / 2) * (2 - (-3)) * (curve 0) = 30 :=
by
  sorry

end triangle_area_from_curve_l89_89843


namespace calculate_expr_l89_89968

noncomputable theory
open Real

def given_expr (x y z u : ℝ) : ℝ :=
  (log (x) / log (10)) * (0.625 * sin y) * sqrt 0.0729 * cos z * 28.9 / (0.0017 * 0.025 * 8.1 * tan u)

theorem calculate_expr :
  abs (given_expr 23 (58 * (π / 180)) (19 * (π / 180)) (33 * (π / 180)) - 1472.8) < 0.001 :=
by
  sorry

end calculate_expr_l89_89968


namespace transformed_curve_eq_l89_89609

theorem transformed_curve_eq :
  ∀ (y x : ℝ), (y * cos x + 2 * y - 1 = 0) →
    ((y - 1) * sin x + 2 * y - 3 = 0) :=
by
  intros y x h
  sorry

end transformed_curve_eq_l89_89609


namespace siblings_total_weight_l89_89501

/-- Given conditions:
Antonio's weight: 50 kilograms.
Antonio's sister weighs 12 kilograms less than Antonio.
Antonio's backpack weight: 5 kilograms.
Antonio's sister's backpack weight: 3 kilograms.
Marco's weight: 30 kilograms.
Marco's stuffed animal weight: 2 kilograms.
Prove that the total weight of the three siblings including additional weights is 128 kilograms.
-/
theorem siblings_total_weight :
  let antonio_weight := 50
  let antonio_sister_weight := antonio_weight - 12
  let antonio_backpack_weight := 5
  let antonio_sister_backpack_weight := 3
  let marco_weight := 30
  let marco_stuffed_animal_weight := 2
  antonio_weight + antonio_backpack_weight +
  antonio_sister_weight + antonio_sister_backpack_weight +
  marco_weight + marco_stuffed_animal_weight = 128 :=
by
  sorry

end siblings_total_weight_l89_89501


namespace permutations_PERCEPTION_l89_89358

-- Define the word "PERCEPTION" and its letter frequencies
def word : String := "PERCEPTION"

def freq_P : Nat := 2
def freq_E : Nat := 2
def freq_R : Nat := 1
def freq_C : Nat := 1
def freq_T : Nat := 1
def freq_I : Nat := 1
def freq_O : Nat := 1
def freq_N : Nat := 1

-- Define the total number of letters in the word
def total_letters : Nat := 10

-- Calculate the number of permutations for the multiset
def permutations : Nat :=
  total_letters.factorial / (freq_P.factorial * freq_E.factorial)

-- Proof problem
theorem permutations_PERCEPTION :
  permutations = 907200 :=
by
  sorry

end permutations_PERCEPTION_l89_89358


namespace no_real_roots_range_of_a_l89_89245

theorem no_real_roots_range_of_a (a : ℝ) : ¬ ∃ x : ℝ, a * x^2 + 6 * x + 1 = 0 ↔ a > 9 :=
by
  sorry

end no_real_roots_range_of_a_l89_89245


namespace dave_paid_4_more_than_doug_l89_89047

theorem dave_paid_4_more_than_doug :
  let slices := 8
  let plain_cost := 8
  let anchovy_additional_cost := 2
  let total_cost := plain_cost + anchovy_additional_cost
  let cost_per_slice := total_cost / slices
  let dave_slices := 5
  let doug_slices := slices - dave_slices
  -- Calculate payments
  let dave_payment := dave_slices * cost_per_slice
  let doug_payment := doug_slices * cost_per_slice
  dave_payment - doug_payment = 4 :=
by
  sorry

end dave_paid_4_more_than_doug_l89_89047


namespace latoya_call_duration_l89_89421

theorem latoya_call_duration
  (initial_credit remaining_credit : ℝ) (cost_per_minute : ℝ) (t : ℝ)
  (h1 : initial_credit = 30)
  (h2 : remaining_credit = 26.48)
  (h3 : cost_per_minute = 0.16)
  (h4 : initial_credit - remaining_credit = t * cost_per_minute) :
  t = 22 := 
sorry

end latoya_call_duration_l89_89421


namespace eighth_arithmetic_term_l89_89902

theorem eighth_arithmetic_term (a₂ a₁₄ a₈ : ℚ) 
  (h2 : a₂ = 8 / 11)
  (h14 : a₁₄ = 9 / 13) :
  a₈ = 203 / 286 :=
by
  sorry

end eighth_arithmetic_term_l89_89902


namespace more_apples_than_pears_l89_89001

-- Definitions based on conditions
def total_fruits : ℕ := 85
def apples : ℕ := 48

-- Statement to prove
theorem more_apples_than_pears : (apples - (total_fruits - apples)) = 11 := by
  -- proof steps
  sorry

end more_apples_than_pears_l89_89001


namespace opposite_of_x_abs_of_x_recip_of_x_l89_89135

noncomputable def x : ℝ := 1 - Real.sqrt 2

theorem opposite_of_x : -x = Real.sqrt 2 - 1 := 
by sorry

theorem abs_of_x : |x| = Real.sqrt 2 - 1 :=
by sorry

theorem recip_of_x : 1/x = -1 - Real.sqrt 2 :=
by sorry

end opposite_of_x_abs_of_x_recip_of_x_l89_89135


namespace max_value_of_a_l89_89980

theorem max_value_of_a (a b c : ℕ) (h : a + b + c = Nat.gcd a b + Nat.gcd b c + Nat.gcd c a + 120) : a ≤ 240 :=
by
  sorry

end max_value_of_a_l89_89980


namespace matrix_multiplication_correct_l89_89802

-- Define the matrices
def A : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![2, 0, -3],
    ![1, 3, -2],
    ![0, 2, 4]
  ]

def B : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![1, -1, 0],
    ![0, 2, -1],
    ![3, 0, 1]
  ]

def C : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![-7, -2, -3],
    ![-5, 5, -5],
    ![12, 4, 2]
  ]

-- Proof statement that multiplication of A and B gives C
theorem matrix_multiplication_correct : A * B = C := 
by
  sorry

end matrix_multiplication_correct_l89_89802


namespace tangent_line_to_curve_l89_89290

noncomputable def tangent_line_eq_at_point : Prop :=
  let f := λ (x : ℝ), (2 * x - 1) / (x + 2)
  tangent_line_eq f (-1 : ℝ, -3 : ℝ) = 5 * (λ (x : ℝ), x) - λ (y : ℝ), y + 2

theorem tangent_line_to_curve :
  tangent_line_eq_at_point :=
begin
  sorry
end

end tangent_line_to_curve_l89_89290


namespace ferry_q_more_time_l89_89620

variables (speed_ferry_p speed_ferry_q distance_ferry_p distance_ferry_q time_ferry_p time_ferry_q : ℕ)
  -- Conditions given in the problem
  (h1 : speed_ferry_p = 8)
  (h2 : time_ferry_p = 2)
  (h3 : distance_ferry_p = speed_ferry_p * time_ferry_p)
  (h4 : distance_ferry_q = 3 * distance_ferry_p)
  (h5 : speed_ferry_q = speed_ferry_p + 4)
  (h6 : time_ferry_q = distance_ferry_q / speed_ferry_q)

theorem ferry_q_more_time : time_ferry_q - time_ferry_p = 2 :=
by
  sorry

end ferry_q_more_time_l89_89620


namespace percent_of_percent_l89_89156

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l89_89156


namespace solve_equation_l89_89471

theorem solve_equation (m x : ℝ) (hm_pos : m > 0) (hm_ne_one : m ≠ 1) :
  7.320 * m^(1 + Real.log x / Real.log 3) + m^(1 - Real.log x / Real.log 3) = m^2 + 1 ↔ x = 3 ∨ x = 1/3 :=
by
  sorry

end solve_equation_l89_89471


namespace breadth_increase_l89_89746

theorem breadth_increase (L B : ℝ) (A : ℝ := L * B) 
  (L' : ℝ := 1.10 * L) (A' : ℝ := 1.375 * A) 
  (B' : ℝ := B * (1 + p / 100)) 
  (h1 : A = L * B)
  (h2 : A' = L' * B')
  (h3 : A' = 1.375 * A) 
  (h4 : L' = 1.10 * L) :
  p = 25 := 
begin 
  sorry 
end

end breadth_increase_l89_89746


namespace line_through_points_l89_89841

theorem line_through_points 
  (A1 B1 A2 B2 : ℝ) 
  (h₁ : A1 * -7 + B1 * 9 = 1) 
  (h₂ : A2 * -7 + B2 * 9 = 1) :
  ∃ (k : ℝ), ∀ (x y : ℝ), (A1, B1) ≠ (A2, B2) → y = k * x + (B1 - k * A1) → -7 * x + 9 * y = 1 :=
sorry

end line_through_points_l89_89841


namespace car_travel_distance_l89_89484

theorem car_travel_distance :
  ∃ S : ℝ, 
    (S > 0) ∧ 
    (∃ v1 v2 t1 t2 t3 t4 : ℝ, 
      (S / 2 = v1 * t1) ∧ (26.25 = v2 * t2) ∧ 
      (S / 2 = v2 * t3) ∧ (31.2 = v1 * t4) ∧ 
      (∃ k : ℝ, k = (S - 31.2) / (v1 + v2) ∧ k > 0 ∧ 
        (S = 58))) := sorry

end car_travel_distance_l89_89484


namespace hyperbola_equation_l89_89525

theorem hyperbola_equation 
  (x y : ℝ)
  (h_ellipse : x^2 / 10 + y^2 / 5 = 1)
  (h_asymptote : 3 * x + 4 * y = 0)
  (h_hyperbola : ∃ k ≠ 0, 9 * x^2 - 16 * y^2 = k) :
  ∃ k : ℝ, k = 45 ∧ (x^2 / 5 - 16 * y^2 / 45 = 1) :=
sorry

end hyperbola_equation_l89_89525


namespace find_min_value_of_quadratic_l89_89505

theorem find_min_value_of_quadratic : ∀ x : ℝ, ∃ c : ℝ, (∃ a b : ℝ, (y = 2*x^2 + 8*x + 7 ∧ (∀ x : ℝ, y ≥ c)) ∧ c = -1) :=
by
  sorry

end find_min_value_of_quadratic_l89_89505


namespace sachin_rahul_age_ratio_l89_89121

theorem sachin_rahul_age_ratio :
  ∀ (Sachin_age Rahul_age: ℕ),
    Sachin_age = 49 →
    Rahul_age = Sachin_age + 14 →
    Nat.gcd Sachin_age Rahul_age = 7 →
    (Sachin_age / Nat.gcd Sachin_age Rahul_age) = 7 ∧ (Rahul_age / Nat.gcd Sachin_age Rahul_age) = 9 :=
by
  intros Sachin_age Rahul_age h1 h2 h3
  rw [h1, h2]
  sorry

end sachin_rahul_age_ratio_l89_89121


namespace question_a_question_b_l89_89479

-- Definitions
def isSolutionA (a b : ℤ) : Prop :=
  1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 7

def isSolutionB (a b : ℤ) : Prop :=
  1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 25

-- Statements
theorem question_a (a b : ℤ) : isSolutionA a b ↔ (a, b) ∈ [(6, -42), (-42, 6), (8, 56), (56, 8), (14, 14)] :=
sorry

theorem question_b (a b : ℤ) : isSolutionB a b ↔ (a, b) ∈ [(24, -600), (-600, 24), (26, 650), (650, 26), (50, 50)] :=
sorry

end question_a_question_b_l89_89479


namespace eight_xyz_le_one_equality_conditions_l89_89123

theorem eight_xyz_le_one (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z ≤ 1 :=
sorry

theorem equality_conditions (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z = 1 ↔ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨
                   (x = -1/2 ∧ y = -1/2 ∧ z = 1/2) ∨
                   (x = -1/2 ∧ y = 1/2 ∧ z = -1/2) ∨
                   (x = 1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end eight_xyz_le_one_equality_conditions_l89_89123


namespace find_number_l89_89543

theorem find_number (N x : ℝ) (h1 : x = 1) (h2 : N / (4 + 1 / x) = 1) : N = 5 := 
by 
  sorry

end find_number_l89_89543


namespace ajay_income_l89_89772

theorem ajay_income
  (I : ℝ)
  (h₁ : I * 0.45 + I * 0.25 + I * 0.075 + 9000 = I) :
  I = 40000 :=
by
  sorry

end ajay_income_l89_89772


namespace Tim_bottle_quarts_l89_89144

theorem Tim_bottle_quarts (ounces_per_week : ℕ) (ounces_per_quart : ℕ) (days_per_week : ℕ) (additional_ounces_per_day : ℕ) (bottles_per_day : ℕ) : 
  ounces_per_week = 812 → ounces_per_quart = 32 → days_per_week = 7 → additional_ounces_per_day = 20 → bottles_per_day = 2 → 
  ∃ quarts_per_bottle : ℝ, quarts_per_bottle = 1.5 := 
by
  intros hw ho hd ha hb
  let total_quarts_per_week := (812 : ℝ) / 32 
  let total_quarts_per_day := total_quarts_per_week / 7 
  let additional_quarts_per_day := 20 / 32 
  let quarts_from_bottles := total_quarts_per_day - additional_quarts_per_day 
  let quarts_per_bottle := quarts_from_bottles / 2 
  use quarts_per_bottle 
  sorry

end Tim_bottle_quarts_l89_89144


namespace find_number_l89_89920

theorem find_number (x : ℝ) (h : 120 = 1.5 * x) : x = 80 :=
by
  sorry

end find_number_l89_89920


namespace prove_fraction_eq_zero_l89_89565

noncomputable theory

variables {R : Type*} [CommRing R]

def A : Matrix (Fin 2) (Fin 2) R :=
  ![![1, 2],
    ![3, 4]]

def B (a b c d : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![a, b],
    ![c, d]]

theorem prove_fraction_eq_zero (a b c d : R) (h₁ : A.mul (B a b c d) = (B a b c d).mul A) (h₂ : 4 * b ≠ c) : 
  (a - d) / (c - 4 * b) = 0 :=
by
  sorry

end prove_fraction_eq_zero_l89_89565


namespace sequences_count_eq_fib_l89_89107

open Nat

def num_special_sequences (n : ℕ) : ℕ :=
  (fibonacci (n + 2)) - 1

theorem sequences_count_eq_fib (n : ℕ) (h : 0 < n) :
  ∀ seq : List ℕ, 
    (∀ i, i < seq.length → odd i → odd (seq.nthLe i h)) ∧ 
    (∀ i, i < seq.length → even i → even (seq.nthLe i h)) → 
    seq.length ≤ n →
  seq.count <| λ x, (1 ≤ x ∧ x ≤ n) ∧ List.sorted (≤) seq =
  num_special_sequences n := 
sorry

end sequences_count_eq_fib_l89_89107


namespace min_value_f_at_3_f_increasing_for_k_neg4_l89_89068

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x + k / (x - 1)

-- Problem (1): If k = 4, find the minimum value of f(x) and the corresponding value of x.
theorem min_value_f_at_3 : ∃ x > 1, @f x 4 = 5 ∧ x = 3 :=
  sorry

-- Problem (2): If k = -4, prove that f(x) is an increasing function for x > 1.
theorem f_increasing_for_k_neg4 : ∀ ⦃x y : ℝ⦄, 1 < x → x < y → f x (-4) < f y (-4) :=
  sorry

end min_value_f_at_3_f_increasing_for_k_neg4_l89_89068


namespace solve_equation_l89_89771

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_equation:
    (7.331 * ((log_base 3 x - 1) / (log_base 3 (x / 3))) - 
    2 * (log_base 3 (Real.sqrt x)) + (log_base 3 x)^2 = 3) → 
    (x = 1 / 3 ∨ x = 9) := by
  sorry

end solve_equation_l89_89771


namespace meeting_time_l89_89941

def time_Cassie_leaves : ℕ := 495 -- 8:15 AM in minutes past midnight
def speed_Cassie : ℕ := 12 -- mph
def break_Cassie : ℚ := 0.25 -- hours
def time_Brian_leaves : ℕ := 540 -- 9:00 AM in minutes past midnight
def speed_Brian : ℕ := 14 -- mph
def total_distance : ℕ := 74 -- miles

def time_in_minutes (h m : ℕ) : ℕ := h * 60 + m

theorem meeting_time : time_Cassie_leaves + (87 : ℚ) / 26 * 60 = time_in_minutes 11 37 := 
by sorry

end meeting_time_l89_89941


namespace find_positive_number_l89_89492

theorem find_positive_number (x : ℕ) (h_pos : 0 < x) (h_equation : x * x / 100 + 6 = 10) : x = 20 :=
by
  sorry

end find_positive_number_l89_89492


namespace ratio_condition_equivalence_l89_89992

variable (a b c d : ℝ)

theorem ratio_condition_equivalence
  (h : (2 * a + 3 * b) / (b + 2 * c) = (3 * c + 2 * d) / (d + 2 * a)) :
  2 * a = 3 * c ∨ 2 * a + 3 * b + d + 2 * c = 0 :=
by
  sorry

end ratio_condition_equivalence_l89_89992


namespace num_ways_to_arrange_PERCEPTION_l89_89363

open Finset

def word := "PERCEPTION"

def num_letters : ℕ := 10

def occurrences : List (Char × ℕ) :=
  [('P', 2), ('E', 2), ('R', 1), ('C', 1), ('E', 2), ('P', 2), ('T', 1), ('I', 2), ('O', 1), ('N', 1)]

def factorial (n : ℕ) : ℕ := List.range n.succ.foldl (· * ·) 1

noncomputable def num_distinct_arrangements (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / ks.foldl (λ acc k => acc * factorial k) 1

theorem num_ways_to_arrange_PERCEPTION :
  num_distinct_arrangements num_letters [2, 2, 2, 1, 1, 1, 1, 1] = 453600 := 
by sorry

end num_ways_to_arrange_PERCEPTION_l89_89363


namespace hyperbola_center_origin_opens_vertically_l89_89630

noncomputable def t_squared : ℝ :=
  let a_sq := (64 / 5 : ℝ) in
  let y := 2 in
  let x := 2 in
  let frac := (frac := y^2 / 4 - 5 * x^2 / a_sq) in
  (frac + 5 / 16 - 1) in
  frac * 4 / 16

theorem hyperbola_center_origin_opens_vertically
  (a_sq : ℝ := 64 / 5)
  (y : ℝ := 2)
  (x : ℝ := 2) : t_squared = 21 / 4 :=
by
  sorry

end hyperbola_center_origin_opens_vertically_l89_89630


namespace sqrt_of_n_is_integer_l89_89998

theorem sqrt_of_n_is_integer (n : ℕ) (h : ∀ p, (0 ≤ p ∧ p < n) → ∃ m g, m + g = n ∧ (m - g) * (m - g) = n) :
  ∃ k : ℕ, k * k = n :=
by 
  sorry

end sqrt_of_n_is_integer_l89_89998


namespace thirteenth_result_is_878_l89_89171

-- Definitions based on the conditions
def avg_25_results : ℕ := 50
def num_25_results : ℕ := 25

def avg_first_12_results : ℕ := 14
def num_first_12_results : ℕ := 12

def avg_last_12_results : ℕ := 17
def num_last_12_results : ℕ := 12

-- Prove the 13th result is 878 given the above conditions.
theorem thirteenth_result_is_878 : 
  ((avg_25_results * num_25_results) - ((avg_first_12_results * num_first_12_results) + (avg_last_12_results * num_last_12_results))) = 878 :=
by
  sorry

end thirteenth_result_is_878_l89_89171


namespace smallest_x_solution_l89_89820

theorem smallest_x_solution :
  ∃ x : ℝ, x * |x| + 3 * x = 5 * x + 2 ∧ (∀ y : ℝ, y * |y| + 3 * y = 5 * y + 2 → x ≤ y)
:=
sorry

end smallest_x_solution_l89_89820


namespace total_money_shared_l89_89355

theorem total_money_shared (T : ℝ) (h : 0.75 * T = 4500) : T = 6000 :=
by
  sorry

end total_money_shared_l89_89355


namespace find_a_for_parallel_lines_l89_89228

def direction_vector_1 (a : ℝ) : ℝ × ℝ × ℝ :=
  (2 * a, 3, 2)

def direction_vector_2 : ℝ × ℝ × ℝ :=
  (2, 3, 2)

theorem find_a_for_parallel_lines : ∃ a : ℝ, direction_vector_1 a = direction_vector_2 :=
by
  use 1
  unfold direction_vector_1
  sorry  -- proof omitted

end find_a_for_parallel_lines_l89_89228


namespace arithmetic_sequence_contains_term_l89_89293

theorem arithmetic_sequence_contains_term (a1 : ℤ) (d : ℤ) (k : ℕ) (h1 : a1 = 3) (h2 : d = 9) :
  ∃ n : ℕ, (a1 + (n - 1) * d) = 3 * 4 ^ k := by
  sorry

end arithmetic_sequence_contains_term_l89_89293


namespace parabola_equation_origin_l89_89904

theorem parabola_equation_origin (x0 : ℝ) :
  ∃ (p : ℝ), (p > 0) ∧ (x0^2 = 2 * p * 2) ∧ (p = 2) ∧ (x0^2 = 4 * 2) := 
by 
  sorry

end parabola_equation_origin_l89_89904


namespace average_speed_of_car_l89_89758

/-- The car's average speed given it travels 65 km in the first hour and 45 km in the second hour. -/
theorem average_speed_of_car (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 65) (h2 : d2 = 45) (h3 : t = 2) :
  (d1 + d2) / t = 55 :=
by
  sorry

end average_speed_of_car_l89_89758


namespace max_length_PQ_l89_89018

-- Define the curve in polar coordinates
def curve (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Definition of points P and Q lying on the curve
def point_on_curve (ρ θ : ℝ) (P : ℝ × ℝ) : Prop :=
  curve ρ θ ∧ P = (ρ * Real.cos θ, ρ * Real.sin θ)

def points_on_curve (P Q : ℝ × ℝ) : Prop :=
  ∃ θ₁ θ₂ ρ₁ ρ₂, point_on_curve ρ₁ θ₁ P ∧ point_on_curve ρ₂ θ₂ Q

-- The theorem stating the maximum length of PQ
theorem max_length_PQ {P Q : ℝ × ℝ} (h : points_on_curve P Q) : dist P Q ≤ 4 :=
sorry

end max_length_PQ_l89_89018


namespace abs_eq_k_solution_l89_89381

theorem abs_eq_k_solution (k : ℝ) (h : k > 4014) :
  {x : ℝ | |x - 2007| + |x + 2007| = k} = (Set.Iio (-2007)) ∪ (Set.Ioi (2007)) :=
by
  sorry

end abs_eq_k_solution_l89_89381


namespace game_cost_l89_89579

theorem game_cost (initial_money : ℕ) (toys_count : ℕ) (toy_price : ℕ) (left_money : ℕ) : 
  initial_money = 63 ∧ toys_count = 5 ∧ toy_price = 3 ∧ left_money = 15 → 
  (initial_money - left_money = 48) :=
by
  sorry

end game_cost_l89_89579


namespace number_of_solutions_fractional_equation_l89_89735

open Real

def fractional_part (x : ℝ) : ℝ := x - floor x

theorem number_of_solutions_fractional_equation : 
  (∃ sols : ℕ, sols = 181 ∧ ∀ x : ℝ, |x| ≤ 10 → fractional_part x + fractional_part (x * x) = 1 → sols = 181) :=
by
  -- Define the fractional part function
  let f := λ x : ℝ, x - floor x
  -- Given conditions
  sorry

end number_of_solutions_fractional_equation_l89_89735


namespace Jaco_total_gift_budget_l89_89558

theorem Jaco_total_gift_budget :
  let friends_gifts := 8 * 9
  let parents_gifts := 2 * 14
  friends_gifts + parents_gifts = 100 :=
by
  let friends_gifts := 8 * 9
  let parents_gifts := 2 * 14
  show friends_gifts + parents_gifts = 100
  sorry

end Jaco_total_gift_budget_l89_89558


namespace find_k_l89_89224

open Real

noncomputable def k_value (θ : ℝ) : ℝ :=
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 - 2 * (tan θ ^ 2 + 1 / tan θ ^ 2) 

theorem find_k (θ : ℝ) (h : θ ≠ 0 ∧ θ ≠ π / 2 ∧ θ ≠ π) :
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 = k_value θ → k_value θ = 6 :=
by
  sorry

end find_k_l89_89224


namespace percentage_wax_left_eq_10_l89_89527

def total_amount_wax : ℕ := 
  let wax20 := 5 * 20
  let wax5 := 5 * 5
  let wax1 := 25 * 1
  wax20 + wax5 + wax1

def wax_used_for_new_candles : ℕ := 
  3 * 5

def percentage_wax_used (total_wax : ℕ) (wax_used : ℕ) : ℕ := 
  (wax_used * 100) / total_wax

theorem percentage_wax_left_eq_10 :
  percentage_wax_used total_amount_wax wax_used_for_new_candles = 10 :=
by
  sorry

end percentage_wax_left_eq_10_l89_89527


namespace fraction_sum_eq_l89_89621

-- Given conditions
variables (w x y : ℝ)
axiom hx : w / x = 1 / 6
axiom hy : w / y = 1 / 5

-- Proof goal
theorem fraction_sum_eq : (x + y) / y = 11 / 5 :=
by sorry

end fraction_sum_eq_l89_89621


namespace Harriet_sibling_product_l89_89076

-- Definition of the family structure
def Harry : Prop := 
  let sisters := 4
  let brothers := 4
  true

-- Harriet being one of Harry's sisters and calculating her siblings
def Harriet : Prop :=
  let S := 4 - 1 -- Number of Harriet's sisters
  let B := 4 -- Number of Harriet's brothers
  S * B = 12

theorem Harriet_sibling_product : Harry → Harriet := by
  intro h
  let S := 3
  let B := 4
  have : S * B = 12 := by norm_num
  exact this

end Harriet_sibling_product_l89_89076


namespace inequality_proof_l89_89829

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2 / 3 :=
sorry

end inequality_proof_l89_89829


namespace percentage_selected_in_state_B_l89_89414

theorem percentage_selected_in_state_B (appeared: ℕ) (selectedA: ℕ) (selected_diff: ℕ)
  (percentage_selectedA: ℝ)
  (h1: appeared = 8100)
  (h2: percentage_selectedA = 6.0)
  (h3: selectedA = appeared * (percentage_selectedA / 100))
  (h4: selected_diff = 81)
  (h5: selectedB = selectedA + selected_diff) :
  ((selectedB : ℝ) / appeared) * 100 = 7 := 
  sorry

end percentage_selected_in_state_B_l89_89414


namespace floor_eq_solution_l89_89507

theorem floor_eq_solution (x : ℝ) : 2.5 ≤ x ∧ x < 3.5 → (⌊2 * x + 0.5⌋ = ⌊x + 3⌋) :=
by
  sorry

end floor_eq_solution_l89_89507


namespace sum_even_integers_between_200_and_600_is_80200_l89_89008

noncomputable def sum_even_integers_between_200_and_600 (a d n : ℕ) : ℕ :=
  n / 2 * (a + (a + (n - 1) * d))

theorem sum_even_integers_between_200_and_600_is_80200 :
  sum_even_integers_between_200_and_600 202 2 200 = 80200 :=
by
  -- proof would go here
  sorry

end sum_even_integers_between_200_and_600_is_80200_l89_89008


namespace simplify_expression_l89_89012

open Real

-- Assume that x, y, z are non-zero real numbers
variables (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)

theorem simplify_expression : (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ := 
by
  -- Proof would go here.
  sorry

end simplify_expression_l89_89012


namespace triangle_min_perimeter_l89_89747

theorem triangle_min_perimeter:
  ∃ x : ℤ, 27 < x ∧ x < 75 ∧ (24 + 51 + x) = 103 :=
begin
  sorry
end

end triangle_min_perimeter_l89_89747


namespace geometric_sequence_fourth_term_l89_89660

theorem geometric_sequence_fourth_term :
  let a₁ := 3^(3/4)
  let a₂ := 3^(2/4)
  let a₃ := 3^(1/4)
  ∃ a₄, a₄ = 1 ∧ a₂ = a₁ * (a₃ / a₂) ∧ a₃ = a₂ * (a₄ / a₃) :=
by
  sorry

end geometric_sequence_fourth_term_l89_89660


namespace min_value_of_expression_min_value_achieved_l89_89677

theorem min_value_of_expression (x : ℝ) (h : x > 0) : 
  (x + 3 / (x + 1)) ≥ 2 * Real.sqrt 3 - 1 := 
sorry

theorem min_value_achieved (x : ℝ) (h : x = Real.sqrt 3 - 1) : 
  (x + 3 / (x + 1)) = 2 * Real.sqrt 3 - 1 := 
sorry

end min_value_of_expression_min_value_achieved_l89_89677


namespace watermelon_sales_correct_l89_89640

def total_watermelons_sold 
  (customers_one_melon : ℕ) 
  (customers_three_melons : ℕ) 
  (customers_two_melons : ℕ) : ℕ :=
  (customers_one_melon * 1) + (customers_three_melons * 3) + (customers_two_melons * 2)

theorem watermelon_sales_correct :
  total_watermelons_sold 17 3 10 = 46 := by
  sorry

end watermelon_sales_correct_l89_89640


namespace new_numbers_are_reciprocals_l89_89765

variable {x y : ℝ}

theorem new_numbers_are_reciprocals (h : (1 / x) + (1 / y) = 1) : 
  (x - 1 = 1 / (y - 1)) ∧ (y - 1 = 1 / (x - 1)) := 
by
  sorry

end new_numbers_are_reciprocals_l89_89765


namespace odd_pair_exists_k_l89_89075

theorem odd_pair_exists_k (a b : ℕ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) : 
  ∃ k : ℕ, (2^2018 ∣ b^k - a^2) ∨ (2^2018 ∣ a^k - b^2) := 
sorry

end odd_pair_exists_k_l89_89075


namespace correct_multiplication_result_l89_89398

theorem correct_multiplication_result (x : ℕ) (h : x - 6 = 51) : x * 6 = 342 :=
  by
  sorry

end correct_multiplication_result_l89_89398


namespace find_f_m_l89_89837

noncomputable def f (x : ℝ) := x^5 + Real.tan x - 3

theorem find_f_m (m : ℝ) (h : f (-m) = -2) : f m = -4 :=
sorry

end find_f_m_l89_89837


namespace nina_money_l89_89114

theorem nina_money (W : ℝ) (P: ℝ) (Q : ℝ) 
  (h1 : P = 6 * W)
  (h2 : Q = 8 * (W - 1))
  (h3 : P = Q) 
  : P = 24 := 
by 
  sorry

end nina_money_l89_89114


namespace pencils_across_diameter_l89_89404

theorem pencils_across_diameter (r : ℝ) (pencil_length_inch : ℕ) (pencils : ℕ) :
  r = 14 ∧ pencil_length_inch = 6 ∧ pencils = 56 → 
  let d := 2 * r in
  let pencil_length_feet := pencil_length_inch / 12 in
  pencils = d / pencil_length_feet :=
begin
  sorry -- Proof is skipped
end

end pencils_across_diameter_l89_89404


namespace part1_l89_89689

theorem part1 (a x0 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a ^ x0 = 2) : a ^ (3 * x0) = 8 := by
  sorry

end part1_l89_89689


namespace solution_y_amount_l89_89170

theorem solution_y_amount :
  ∀ (y : ℝ) (volume_x volume_y : ℝ),
    volume_x = 200 ∧
    volume_y = y ∧
    10 / 100 * volume_x = 20 ∧
    30 / 100 * volume_y = 0.3 * y ∧
    (20 + 0.3 * y) / (volume_x + y) = 0.25 →
    y = 600 :=
by 
  intros y volume_x volume_y
  intros H
  sorry

end solution_y_amount_l89_89170


namespace inheritance_calculation_l89_89865

theorem inheritance_calculation
  (x : ℝ)
  (h1 : 0.25 * x + 0.15 * (0.75 * x) = 14000) :
  x = 38600 := by
  sorry

end inheritance_calculation_l89_89865


namespace compare_sqrt_differences_l89_89675

theorem compare_sqrt_differences :
  let a := (Real.sqrt 7) - (Real.sqrt 6)
  let b := (Real.sqrt 3) - (Real.sqrt 2)
  a < b :=
by
  sorry -- Proof goes here

end compare_sqrt_differences_l89_89675


namespace league_games_count_l89_89741

theorem league_games_count :
  let num_divisions := 2
  let teams_per_division := 9
  let intra_division_games (teams_per_div : ℕ) := (teams_per_div * (teams_per_div - 1) / 2) * 3
  let inter_division_games (teams_per_div : ℕ) (num_div : ℕ) := teams_per_div * teams_per_div * 2
  intra_division_games teams_per_division * num_divisions + inter_division_games teams_per_division num_divisions = 378 :=
by
  sorry

end league_games_count_l89_89741


namespace range_of_x_function_l89_89756

open Real

theorem range_of_x_function : 
  ∀ x : ℝ, (x + 1 >= 0) ∧ (x - 3 ≠ 0) ↔ (x >= -1) ∧ (x ≠ 3) := 
by 
  sorry 

end range_of_x_function_l89_89756


namespace sequence_length_arithmetic_sequence_l89_89200

theorem sequence_length_arithmetic_sequence :
  ∃ n : ℕ, ∀ (a d : ℕ), a = 2 → d = 3 → a + (n - 1) * d = 2014 ∧ n = 671 :=
by {
  sorry
}

end sequence_length_arithmetic_sequence_l89_89200


namespace line_tangent_to_ellipse_l89_89212

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = mx + 2 → x^2 + 9 * y^2 = 9 → ∃ u, y = u) → m^2 = 1 / 3 := 
by
  intro h
  sorry

end line_tangent_to_ellipse_l89_89212


namespace max_gold_coins_l89_89167

theorem max_gold_coins (k : ℤ) (h1 : ∃ k : ℤ, 15 * k + 3 < 120) : 
  ∃ n : ℤ, n = 15 * k + 3 ∧ n < 120 ∧ n = 108 :=
by
  sorry

end max_gold_coins_l89_89167


namespace none_of_these_l89_89697

-- Problem Statement:
theorem none_of_these (r x y : ℝ) (h1 : r > 0) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : x^2 + y^2 > x^2 * y^2) :
  ¬ (-x > -y) ∧ ¬ (-x > y) ∧ ¬ (1 > -y / x) ∧ ¬ (1 < x / y) :=
by
  sorry

end none_of_these_l89_89697


namespace remainder_2753_div_98_l89_89611

theorem remainder_2753_div_98 : (2753 % 98) = 9 := 
by sorry

end remainder_2753_div_98_l89_89611


namespace randy_trip_total_distance_l89_89279

-- Definition of the problem condition
def randy_trip_length (x : ℝ) : Prop :=
  x / 3 + 20 + x / 5 = x

-- The total length of Randy's trip
theorem randy_trip_total_distance : ∃ x : ℝ, randy_trip_length x ∧ x = 300 / 7 :=
by
  sorry

end randy_trip_total_distance_l89_89279


namespace at_least_six_on_circle_l89_89074

-- Defining the types for point and circle
variable (Point : Type)
variable (Circle : Type)

-- Assuming the existence of a well-defined predicate that checks whether points lie on the same circle
variable (lies_on_circle : Circle → Point → Prop)
variable (exists_circle : Point → Point → Point → Point → Circle)
variable (five_points_condition : ∀ (p1 p2 p3 p4 p5 : Point), 
  ∃ (c : Circle), lies_on_circle c p1 ∧ lies_on_circle c p2 ∧ 
                   lies_on_circle c p3 ∧ lies_on_circle c p4)

-- Given 13 points on a plane
variables (P : List Point)
variable (length_P : P.length = 13)

-- The main theorem statement
theorem at_least_six_on_circle : 
  (∀ (P : List Point) (h : P.length = 13),
    (∀ p1 p2 p3 p4 p5 : Point, ∃ (c : Circle), lies_on_circle c p1 ∧ lies_on_circle c p2 ∧ 
                               lies_on_circle c p3 ∧ lies_on_circle c p4)) →
    (∃ (c : Circle), ∃ (l : List Point), l.length ≥ 6 ∧ ∀ p ∈ l, lies_on_circle c p) :=
sorry

end at_least_six_on_circle_l89_89074


namespace carbon_emission_l89_89317

theorem carbon_emission (x y : ℕ) (h1 : x + y = 70) (h2 : x = 5 * y - 8) : y = 13 ∧ x = 57 := by
  sorry

end carbon_emission_l89_89317


namespace total_units_is_34_l89_89873

-- Define the number of units on the first floor
def first_floor_units : Nat := 2

-- Define the number of units on the remaining floors (each floor) and number of such floors
def other_floors_units : Nat := 5
def number_of_other_floors : Nat := 3

-- Define the total number of floors per building
def total_floors : Nat := 4

-- Calculate the total units in one building
def units_in_one_building : Nat := first_floor_units + other_floors_units * number_of_other_floors

-- The number of buildings
def number_of_buildings : Nat := 2

-- Calculate the total number of units in both buildings
def total_units : Nat := units_in_one_building * number_of_buildings

-- Prove the total units is 34
theorem total_units_is_34 : total_units = 34 := by
  sorry

end total_units_is_34_l89_89873


namespace physics_kit_prices_l89_89436

theorem physics_kit_prices :
  ∃ (price_A price_B : ℝ), price_A = 180 ∧ price_B = 150 ∧
    price_A = 1.2 * price_B ∧
    9900 / price_A = 7500 / price_B + 5 :=
by
  use 180, 150
  sorry

end physics_kit_prices_l89_89436


namespace simplify_fraction_l89_89250

theorem simplify_fraction (a b : ℕ) (h : a ≠ b) : (2 * a) / (2 * b) = a / b :=
sorry

end simplify_fraction_l89_89250


namespace triangle_area_from_curve_l89_89844

-- Definition of the curve
def curve (x : ℝ) : ℝ := (x - 2)^2 * (x + 3)

-- Area calculation based on intercepts
theorem triangle_area_from_curve : 
  (1 / 2) * (2 - (-3)) * (curve 0) = 30 :=
by
  sorry

end triangle_area_from_curve_l89_89844


namespace book_has_125_pages_l89_89781

-- Define the number of pages in each chapter
def chapter1_pages : ℕ := 66
def chapter2_pages : ℕ := 35
def chapter3_pages : ℕ := 24

-- Define the total number of pages in the book
def total_pages : ℕ := chapter1_pages + chapter2_pages + chapter3_pages

-- State the theorem to prove that the total number of pages is 125
theorem book_has_125_pages : total_pages = 125 := 
by 
  -- The proof is omitted for the purpose of this task
  sorry

end book_has_125_pages_l89_89781


namespace percent_calculation_l89_89147

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l89_89147


namespace min_colors_5x5_grid_l89_89575

def is_valid_coloring (grid : Fin 5 × Fin 5 → ℕ) (k : ℕ) : Prop :=
  ∀ i j : Fin 5, ∀ di dj : Fin 2, ∀ c : ℕ,
    (di ≠ 0 ∨ dj ≠ 0) →
    (grid (i, j) = c ∧ grid (i + di, j + dj) = c ∧ grid (i + 2 * di, j + 2 * dj) = c) → 
    False

theorem min_colors_5x5_grid : 
  ∀ (grid : Fin 5 × Fin 5 → ℕ), (∀ i j, grid (i, j) < 3) → is_valid_coloring grid 3 := 
by
  sorry

end min_colors_5x5_grid_l89_89575


namespace tangent_line_at_point_l89_89286

noncomputable def f : ℝ → ℝ := λ x, (2 * x - 1) / (x + 2)

def point : ℝ × ℝ := (-1, -3)

def tangent_line_eq (m x₁ y₁ : ℝ) := 
  λ x y, y - y₁ = m * (x - x₁)

theorem tangent_line_at_point :
  let slope_at_point := 5 in
  let x₀ := point.fst in
  let y₀ := point.snd in
  ∀ x y, (tangent_line_eq slope_at_point x₀ y₀ x y) ↔ (5 * x - y + 2 = 0) := 
by {
  sorry
}

end tangent_line_at_point_l89_89286


namespace find_salary_month_l89_89890

variable (J F M A May : ℝ)

def condition_1 : Prop := (J + F + M + A) / 4 = 8000
def condition_2 : Prop := (F + M + A + May) / 4 = 8450
def condition_3 : Prop := J = 4700
def condition_4 (X : ℝ) : Prop := X = 6500

theorem find_salary_month (J F M A May : ℝ) 
  (h1 : condition_1 J F M A) 
  (h2 : condition_2 F M A May) 
  (h3 : condition_3 J) 
  : ∃ M : ℝ, condition_4 May :=
by sorry

end find_salary_month_l89_89890


namespace no_integer_solutions_quadratic_l89_89119

theorem no_integer_solutions_quadratic (n : ℤ) (s : ℕ) (pos_odd_s : s % 2 = 1) :
  ¬ ∃ x : ℤ, x^2 - 16 * n * x + 7^s = 0 :=
sorry

end no_integer_solutions_quadratic_l89_89119


namespace necessary_not_sufficient_l89_89624

variable (a b : ℝ)

theorem necessary_not_sufficient : 
  (a > b) -> ¬ (a > b+1) ∨ (a > b+1 ∧ a > b) :=
by
  intro h
  have h1 : ¬ (a > b+1) := sorry
  have h2 : (a > b+1 -> a > b) := sorry
  exact Or.inl h1

end necessary_not_sufficient_l89_89624


namespace percentage_of_females_l89_89116

theorem percentage_of_females (total_passengers : ℕ)
  (first_class_percentage : ℝ) (male_fraction_first_class : ℝ)
  (females_coach_class : ℕ) (h1 : total_passengers = 120)
  (h2 : first_class_percentage = 0.10)
  (h3 : male_fraction_first_class = 1/3)
  (h4 : females_coach_class = 40) :
  (females_coach_class + (first_class_percentage * total_passengers - male_fraction_first_class * (first_class_percentage * total_passengers))) / total_passengers * 100 = 40 :=
by
  sorry

end percentage_of_females_l89_89116


namespace find_a_l89_89232

theorem find_a (A B : Real) (b a : Real) (hA : A = 45) (hB : B = 60) (hb : b = Real.sqrt 3) : 
  a = Real.sqrt 2 :=
sorry

end find_a_l89_89232


namespace items_left_in_store_l89_89653

theorem items_left_in_store: (4458 - 1561) + 575 = 3472 :=
by 
  sorry

end items_left_in_store_l89_89653


namespace solve_for_diamond_l89_89083

theorem solve_for_diamond (d : ℕ) (h1 : d * 9 + 6 = d * 10 + 3) (h2 : d < 10) : d = 3 :=
by
  sorry

end solve_for_diamond_l89_89083


namespace volume_of_wedge_l89_89495

theorem volume_of_wedge (r : ℝ) (V : ℝ) (sphere_wedges : ℝ) 
  (h_circumference : 2 * Real.pi * r = 18 * Real.pi)
  (h_volume : V = (4 / 3) * Real.pi * r ^ 3) 
  (h_sphere_wedges : sphere_wedges = 6) : 
  V / sphere_wedges = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l89_89495


namespace number_of_integers_between_sqrt10_and_sqrt100_l89_89080

theorem number_of_integers_between_sqrt10_and_sqrt100 : 
  let a := Real.sqrt 10
  let b := Real.sqrt 100
  ∃ (n : ℕ), n = 6 ∧ (∀ x : ℕ, (x > a ∧ x < b) → (4 ≤ x ∧ x ≤ 9)) :=
by
  sorry

end number_of_integers_between_sqrt10_and_sqrt100_l89_89080


namespace batsman_average_after_11th_inning_l89_89472

theorem batsman_average_after_11th_inning
  (x : ℝ)  -- the average score of the batsman before the 11th inning
  (h1 : 10 * x + 85 = 11 * (x + 5))  -- given condition from the problem
  : x + 5 = 35 :=   -- goal statement proving the new average
by
  -- We need to prove that new average after the 11th inning is 35
  sorry

end batsman_average_after_11th_inning_l89_89472


namespace sequence_fiftieth_term_l89_89572

theorem sequence_fiftieth_term :
  ∀ (n : ℕ), 0 < n → (a : ℤ) (a = (-1)^(n-1) * (4 * n - 1)) → a = -199 :=
by {
  intros n hn a ha,
  sorry
}

end sequence_fiftieth_term_l89_89572


namespace min_value_of_expression_l89_89045

theorem min_value_of_expression (n : ℕ) (h : n > 0) : (n / 3 + 27 / n) ≥ 6 :=
by {
  -- Proof goes here but is not required in the statement
  sorry
}

end min_value_of_expression_l89_89045


namespace george_final_score_l89_89368

-- Definitions for points in the first half
def first_half_odd_points (questions : Nat) := 5 * 2
def first_half_even_points (questions : Nat) := 5 * 4
def first_half_bonus_points (questions : Nat) := 3 * 5
def first_half_points := first_half_odd_points 5 + first_half_even_points 5 + first_half_bonus_points 3

-- Definitions for points in the second half
def second_half_odd_points (questions : Nat) := 6 * 3
def second_half_even_points (questions : Nat) := 6 * 5
def second_half_bonus_points (questions : Nat) := 4 * 5
def second_half_points := second_half_odd_points 6 + second_half_even_points 6 + second_half_bonus_points 4

-- Definition of the total points
def total_points := first_half_points + second_half_points

-- The theorem statement to prove the total points
theorem george_final_score : total_points = 113 := by
  unfold total_points
  unfold first_half_points
  unfold second_half_points
  unfold first_half_odd_points first_half_even_points first_half_bonus_points
  unfold second_half_odd_points second_half_even_points second_half_bonus_points
  sorry

end george_final_score_l89_89368


namespace trig_identity_l89_89388

-- Given conditions
variables (α : ℝ) (h_tan : Real.tan (Real.pi - α) = -2)

-- The goal is to prove the desired equality.
theorem trig_identity :
  1 / (Real.cos (2 * α) + Real.cos α * Real.cos α) = -5 / 2 :=
by
  sorry

end trig_identity_l89_89388


namespace arithmetic_sequence_a6_value_l89_89704

theorem arithmetic_sequence_a6_value (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_roots : ∀ x, x^2 + 12 * x - 8 = 0 → (x = a 2 ∨ x = a 10)) :
  a 6 = -6 :=
by
  -- Definitions and given conditions would go here in a fully elaborated proof.
  sorry

end arithmetic_sequence_a6_value_l89_89704


namespace transformed_parabolas_combined_l89_89187

theorem transformed_parabolas_combined (a b c : ℝ) :
  let f (x : ℝ) := a * (x - 3) ^ 2 + b * (x - 3) + c
  let g (x : ℝ) := -a * (x + 4) ^ 2 - b * (x + 4) - c
  ∀ x, (f x + g x) = -14 * a * x - 19 * a - 7 * b :=
by
  -- This is a placeholder for the actual proof using the conditions
  sorry

end transformed_parabolas_combined_l89_89187


namespace part1_part2_part3_l89_89234

def A (x y : ℝ) := 2*x^2 + 3*x*y + 2*y
def B (x y : ℝ) := x^2 - x*y + x

theorem part1 (x y : ℝ) : A x y - 2 * B x y = 5*x*y - 2*x + 2*y := by
  sorry

theorem part2 (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 2) :
  A x y - 2 * B x y = 28 ∨ A x y - 2 * B x y = -40 ∨ A x y - 2 * B x y = -20 ∨ A x y - 2 * B x y = 32 := by
  sorry

theorem part3 (y : ℝ) : (∀ x : ℝ, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2/5 := by
  sorry

end part1_part2_part3_l89_89234


namespace Toby_second_part_distance_l89_89908

noncomputable def total_time_journey (distance_unloaded_second: ℝ) : ℝ :=
  18 + (distance_unloaded_second / 20) + 8 + 7

theorem Toby_second_part_distance:
  ∃ d : ℝ, total_time_journey d = 39 ∧ d = 120 :=
by
  use 120
  unfold total_time_journey
  sorry

end Toby_second_part_distance_l89_89908


namespace cyclic_quadrilateral_angle_D_l89_89093

theorem cyclic_quadrilateral_angle_D (A B C D : ℝ) (h₁ : A + B + C + D = 360) (h₂ : ∃ x, A = 3 * x ∧ B = 4 * x ∧ C = 6 * x) :
  D = 100 :=
by
  sorry

end cyclic_quadrilateral_angle_D_l89_89093


namespace catFinishesOnMondayNextWeek_l89_89878

def morningConsumptionDaily (day : String) : ℚ := if day = "Wednesday" then 1 / 3 else 1 / 4
def eveningConsumptionDaily : ℚ := 1 / 6

def totalDailyConsumption (day : String) : ℚ :=
  morningConsumptionDaily day + eveningConsumptionDaily

-- List of days in order
def week : List String := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

-- Total food available initially
def totalInitialFood : ℚ := 8

-- Function to calculate total food consumed until a given day
def foodConsumedUntil (day : String) : ℚ :=
  week.takeWhile (· != day) |>.foldl (λ acc d => acc + totalDailyConsumption d) 0

-- Function to determine the day when 8 cans are completely consumed
def finishingDay : String :=
  match week.find? (λ day => foodConsumedUntil day + totalDailyConsumption day = totalInitialFood) with
  | some day => day
  | none => "Monday"  -- If no exact match is found in the first week, it is Monday of the next week

theorem catFinishesOnMondayNextWeek :
  finishingDay = "Monday" := by
  sorry

end catFinishesOnMondayNextWeek_l89_89878


namespace multiply_24_99_l89_89773

theorem multiply_24_99 : 24 * 99 = 2376 :=
by
  sorry

end multiply_24_99_l89_89773


namespace probability_two_different_colors_l89_89178

noncomputable def probability_different_colors (total_balls red_balls black_balls : ℕ) : ℚ :=
  let total_ways := (Finset.range total_balls).card.choose 2
  let diff_color_ways := (Finset.range black_balls).card.choose 1 * (Finset.range red_balls).card.choose 1
  diff_color_ways / total_ways

theorem probability_two_different_colors (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ)
  (h_total : total_balls = 5) (h_red : red_balls = 2) (h_black : black_balls = 3) :
  probability_different_colors total_balls red_balls black_balls = 3 / 5 :=
by
  subst h_total
  subst h_red
  subst h_black
  -- Here the proof would follow using the above definitions and reasoning
  sorry

end probability_two_different_colors_l89_89178


namespace like_terms_sum_l89_89531

theorem like_terms_sum (m n : ℕ) (a b : ℝ) :
  (∀ c d : ℝ, -4 * a^(2 * m) * b^(3) = c * a^(6) * b^(n + 1)) →
  m + n = 5 :=
by 
  intro h
  sorry

end like_terms_sum_l89_89531


namespace inequality_selection_l89_89625

theorem inequality_selection (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 := 
by sorry

end inequality_selection_l89_89625


namespace quadratic_root_sum_m_n_l89_89995

theorem quadratic_root_sum_m_n (m n : ℤ) :
  (∃ x : ℤ, x^2 + m * x + 2 * n = 0 ∧ x = 2) → m + n = -2 :=
by
  sorry

end quadratic_root_sum_m_n_l89_89995


namespace value_of_pq_s_l89_89537

-- Definitions for the problem
def polynomial_divisible (p q s : ℚ) : Prop :=
  ∀ x : ℚ, (x^3 + 4 * x^2 + 16 * x + 8) ∣ (x^4 + 6 * x^3 + 8 * p * x^2 + 6 * q * x + s)

-- The main theorem statement to prove
theorem value_of_pq_s (p q s : ℚ) (h : polynomial_divisible p q s) : (p + q) * s = 332 / 3 :=
sorry -- Proof omitted

end value_of_pq_s_l89_89537


namespace determinant_inequality_l89_89728

variable (x : ℝ)

def determinant (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem determinant_inequality (h : determinant 2 (3 - x) 1 x > 0) : x > 1 := by
  sorry

end determinant_inequality_l89_89728


namespace forgot_to_mow_l89_89343

-- Definitions
def earning_per_lawn : ℕ := 9
def lawns_to_mow : ℕ := 12
def actual_earning : ℕ := 36

-- Statement to prove
theorem forgot_to_mow : (lawns_to_mow - (actual_earning / earning_per_lawn)) = 8 := by
  sorry

end forgot_to_mow_l89_89343


namespace smallest_k_divisibility_l89_89612

theorem smallest_k_divisibility : ∃ (k : ℕ), k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_divisibility_l89_89612


namespace triangular_number_difference_30_28_l89_89795

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_number_difference_30_28 : triangular_number 30 - triangular_number 28 = 59 := 
by
  sorry

end triangular_number_difference_30_28_l89_89795


namespace tea_drinking_proof_l89_89028

theorem tea_drinking_proof :
  ∃ (k : ℝ), 
    (∃ (c_sunday t_sunday c_wednesday t_wednesday : ℝ),
      c_sunday = 8.5 ∧ 
      t_sunday = 4 ∧ 
      c_wednesday = 5 ∧ 
      t_sunday * c_sunday = k ∧ 
      t_wednesday * c_wednesday = k ∧ 
      t_wednesday = 6.8) :=
sorry

end tea_drinking_proof_l89_89028


namespace curve_is_ellipse_perpendicular_intersects_l89_89064

noncomputable def curve (P : ℝ × ℝ) : Prop :=
  dist P (-⟨sqrt 3, 0⟩) + dist P (⟨sqrt 3, 0⟩) = 4

noncomputable def ellipse_eqn (x y : ℝ) : Prop :=
  (x^2) / 4 + y^2 = 1

theorem curve_is_ellipse : ∀ P, curve P ↔ ∃ x y, P = (x, y) ∧ ellipse_eqn x y :=
by sorry

noncomputable def line (k : ℝ) : ℝ × ℝ → Prop :=
λ (x, y), y = k * x - 2

theorem perpendicular_intersects (x1 y1 x2 y2 k : ℝ)
  (h1 : ellipse_eqn x1 y1)
  (h2 : ellipse_eqn x2 y2)
  (h3 : line k (x1, y1))
  (h4 : line k (x2, y2))
  (h5 : x1 * x2 + y1 * y2 = 0) :
  k = 2 ∨ k = -2 :=
by sorry

end curve_is_ellipse_perpendicular_intersects_l89_89064


namespace complement_intersection_eq_l89_89526

def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem complement_intersection_eq :
  (U \ A) ∩ B = {3} :=
by
  sorry

end complement_intersection_eq_l89_89526


namespace determine_k_value_l89_89947

theorem determine_k_value : (5 ^ 1002 + 6 ^ 1001) ^ 2 - (5 ^ 1002 - 6 ^ 1001) ^ 2 = 24 * 30 ^ 1001 :=
by
  sorry

end determine_k_value_l89_89947


namespace proof_problem_l89_89566

theorem proof_problem (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a > b) (h5 : a^2 - a * c + b * c = 7) :
  a - c = 0 ∨ a - c = 1 :=
 sorry

end proof_problem_l89_89566


namespace identify_parrots_l89_89277

-- Definitions of parrots
inductive Parrot
| gosha : Parrot
| kesha : Parrot
| roma : Parrot

open Parrot

-- Properties of each parrot
def always_honest (p : Parrot) : Prop :=
  p = gosha

def always_liar (p : Parrot) : Prop :=
  p = kesha

def sometimes_honest (p : Parrot) : Prop :=
  p = roma

-- Statements given by each parrot
def Gosha_statement : Prop :=
  always_liar kesha

def Kesha_statement : Prop :=
  sometimes_honest kesha

def Roma_statement : Prop :=
  always_honest kesha

-- Final statement to prove the identities
theorem identify_parrots (p : Parrot) :
  Gosha_statement ∧ Kesha_statement ∧ Roma_statement → (always_liar Parrot.kesha ∧ sometimes_honest Parrot.roma) :=
by
  intro h
  exact sorry

end identify_parrots_l89_89277


namespace solve_quadratic_equation_l89_89883

theorem solve_quadratic_equation (x : ℝ) : 
  (2 * x^2 - 4 * x = 6 - 3 * x) ↔ 
  (x = -3/2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l89_89883


namespace maximum_obtuse_dihedral_angles_l89_89336

-- condition: define what a tetrahedron is and its properties
structure Tetrahedron :=
  (edges : Fin 6 → ℝ)   -- represents the 6 edges
  (dihedral_angles : Fin 6 → ℝ) -- represents the 6 dihedral angles

-- Define obtuse angle in degrees
def is_obtuse (angle : ℝ) : Prop := angle > 90 ∧ angle < 180

-- Theorem statement
theorem maximum_obtuse_dihedral_angles (T : Tetrahedron) : 
  (∃ count : ℕ, count = 3 ∧ (∀ i, is_obtuse (T.dihedral_angles i) → count <= 3)) := sorry

end maximum_obtuse_dihedral_angles_l89_89336


namespace percent_of_percent_l89_89152

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l89_89152


namespace angle_ACD_l89_89862

theorem angle_ACD {α β δ : Type*} [LinearOrderedField α] [CharZero α] (ABC DAB DBA : α)
  (h1 : ABC = 60) (h2 : BAC = 80) (h3 : DAB = 10) (h4 : DBA = 20):
  ACD = 30 := by
  sorry

end angle_ACD_l89_89862


namespace probability_no_defective_pencils_l89_89475

theorem probability_no_defective_pencils :
  let total_pencils := 6
  let defective_pencils := 2
  let pencils_chosen := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_ways := Nat.choose total_pencils pencils_chosen
  let non_defective_ways := Nat.choose non_defective_pencils pencils_chosen
  (non_defective_ways / total_ways : ℚ) = 1 / 5 :=
by
  sorry

end probability_no_defective_pencils_l89_89475


namespace angle_D_measure_l89_89551

theorem angle_D_measure 
  (A B C D : Type)
  (angleA : ℝ)
  (angleB : ℝ)
  (angleC : ℝ)
  (angleD : ℝ)
  (BD_bisector : ℝ → ℝ) :
  angleA = 85 ∧ angleB = 50 ∧ angleC = 25 ∧ BD_bisector angleB = 25 →
  angleD = 130 :=
by
  intro h
  have hA := h.1
  have hB := h.2.1
  have hC := h.2.2.1
  have hBD := h.2.2.2
  sorry

end angle_D_measure_l89_89551


namespace sum_of_three_distinct_integers_l89_89450

theorem sum_of_three_distinct_integers (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c) 
  (h₄ : a * b * c = 5^3) (h₅ : a > 0) (h₆ : b > 0) (h₇ : c > 0) : a + b + c = 31 :=
by
  sorry

end sum_of_three_distinct_integers_l89_89450


namespace find_c_l89_89769

-- Define the polynomial P(x)
def P (c : ℚ) (x : ℚ) : ℚ := x^3 + 4 * x^2 + c * x + 20

-- Given that x - 3 is a factor of P(x), prove that c = -83/3
theorem find_c (c : ℚ) (h : P c 3 = 0) : c = -83 / 3 :=
by
  sorry

end find_c_l89_89769


namespace angela_problems_l89_89935

theorem angela_problems (M J S K A : ℕ) :
  M = 3 →
  J = (M * M - 5) + ((M * M - 5) / 3) →
  S = 50 / 10 →
  K = (J + S) / 2 →
  A = 50 - (M + J + S + K) →
  A = 32 :=
by
  intros hM hJ hS hK hA
  sorry

end angela_problems_l89_89935


namespace log_inequality_l89_89389

variable (a b : ℝ)

theorem log_inequality (h1 : a > b) (h2 : b > 1) : a * Real.log a > b * Real.log b :=
sorry

end log_inequality_l89_89389


namespace find_precy_age_l89_89853

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

end find_precy_age_l89_89853


namespace acme_corp_five_letter_words_l89_89791

open Finset

theorem acme_corp_five_letter_words (A E I O U Y : Nat) (A_limit : A = 3) (other_limit : E = 5 ∧ I = 5 ∧ O = 5 ∧ U = 5 ∧ Y = 5) :
  (A + E + I + O + U + Y = 28) →
  (number_of_valid_words : Nat := (5 ^ 5) + (5 * 5 ^ 4) + (10 * 5 ^ 3) + (10 * 5 ^ 2)) = 7750 :=
by
  sorry

end acme_corp_five_letter_words_l89_89791


namespace colored_paper_distribution_l89_89762

theorem colored_paper_distribution (F M : ℕ) (h1 : F + M = 24) (h2 : M = 2 * F) (total_sheets : ℕ) (distributed_sheets : total_sheets = 48) : 
  (48 / F) = 6 := by
  sorry

end colored_paper_distribution_l89_89762


namespace vans_capacity_l89_89732

-- Definitions based on the conditions
def num_students : ℕ := 22
def num_adults : ℕ := 2
def num_vans : ℕ := 3

-- The Lean statement (theorem to be proved)
theorem vans_capacity :
  (num_students + num_adults) / num_vans = 8 := 
by
  sorry

end vans_capacity_l89_89732


namespace calculate_area_of_triangle_l89_89347

theorem calculate_area_of_triangle :
  let p1 := (5, -2)
  let p2 := (5, 8)
  let p3 := (12, 8)
  let area := (1 / 2) * ((p2.2 - p1.2) * (p3.1 - p2.1))
  area = 35 := 
by
  sorry

end calculate_area_of_triangle_l89_89347


namespace probability_sum_le_4_of_two_dice_tossed_l89_89462

open ProbabilityTheory

def diceSumLE4 : MeasureTheory.ProbabilityMeasure (Set (Fin 6 × Fin 6)) :=
  sorry

theorem probability_sum_le_4_of_two_dice_tossed :
  diceSumLE4 { p : Fin 6 × Fin 6 | p.1 + p.2 + 2 ≤ 4 } = 1 / 6 :=
sorry

end probability_sum_le_4_of_two_dice_tossed_l89_89462


namespace number_of_integers_between_sqrt10_and_sqrt100_l89_89079

theorem number_of_integers_between_sqrt10_and_sqrt100 : 
  let a := Real.sqrt 10
  let b := Real.sqrt 100
  ∃ (n : ℕ), n = 6 ∧ (∀ x : ℕ, (x > a ∧ x < b) → (4 ≤ x ∧ x ≤ 9)) :=
by
  sorry

end number_of_integers_between_sqrt10_and_sqrt100_l89_89079


namespace properties_of_data_set_l89_89328

def data_set : List ℕ := [67, 57, 37, 40, 46, 62, 31, 47, 31, 30]

def sorted_data_set : List ℕ := [30, 31, 31, 37, 40, 46, 47, 57, 62, 67]

def mode (l : List ℕ) : ℕ :=
(d : ℕ × ℕ) ← list.maximumBy (λ d, l.count d) l
(_, x) := d
x

def range (l : List ℕ) : ℕ :=
(list.maximum l - list.minimum l)

def quantile (l : List ℕ) (q : ℕ) : ℝ :=
let pos := (q * l.length) / 100
let sorted := l.sort λ a b => a < b
let lower := sorted[pos]
let upper := sorted[pos + 1]
(lower + upper) / 2

theorem properties_of_data_set :
  mode data_set = 31 ∧
  range data_set = 37 ∧
  quantile data_set 10 = 30.5 :=
by sorry

end properties_of_data_set_l89_89328


namespace polynomial_inequality_l89_89570

theorem polynomial_inequality (f : ℝ → ℝ) (h1 : f 0 = 1)
    (h2 : ∀ (x y : ℝ), f (x - y) + f x ≥ 2 * x^2 - 2 * x * y + y^2 + 2 * x - y + 2) :
    f = λ x => x^2 + x + 1 := by
  sorry

end polynomial_inequality_l89_89570


namespace sum_of_three_integers_l89_89452

theorem sum_of_three_integers (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) (h7 : a * b * c = 5^3) : a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_l89_89452


namespace no_int_x_divisible_by_169_l89_89577

theorem no_int_x_divisible_by_169 (x : ℤ) : ¬ (169 ∣ (x^2 + 5 * x + 16)) := by
  sorry

end no_int_x_divisible_by_169_l89_89577


namespace diagonal_entries_cover_set_l89_89109

-- Variables and definitions used in the problem
variable {n : ℕ} (N : Finset ℕ) (f : ℕ → ℕ → ℕ)

-- Problem conditions
def isOdd (n : ℕ) : Prop := n % 2 = 1
def isSymmetric (f : ℕ → ℕ → ℕ) : Prop := ∀ r s, f(r,s) = f(s,r)
def isSurjectiveInRows (f : ℕ → ℕ → ℕ) (N : Finset ℕ) : Prop := 
  ∀ r ∈ N, (Finset.image (λ (s : ℕ), f r s) N) = N

-- Proof goal
theorem diagonal_entries_cover_set
  (hn : isOdd n)
  (hN : N = Finset.range (n+1))
  (hf_symm : isSymmetric f)
  (hf_surr : isSurjectiveInRows f N) :
  (Finset.image (λ r, f r r) N) = N :=
sorry

end diagonal_entries_cover_set_l89_89109


namespace factor_polynomial_l89_89964

theorem factor_polynomial (y : ℝ) : 
  y^6 - 64 = (y - 2) * (y + 2) * (y^2 + 2 * y + 4) * (y^2 - 2 * y + 4) :=
by
  sorry

end factor_polynomial_l89_89964


namespace min_odd_integers_is_zero_l89_89145

noncomputable def minOddIntegers (a b c d e f : ℤ) : ℕ :=
  if h₁ : a + b = 22 ∧ a + b + c + d = 36 ∧ a + b + c + d + e + f = 50 then
    0
  else
    6 -- default, just to match type expectations

theorem min_odd_integers_is_zero (a b c d e f : ℤ)
  (h₁ : a + b = 22)
  (h₂ : a + b + c + d = 36)
  (h₃ : a + b + c + d + e + f = 50) :
  minOddIntegers a b c d e f = 0 :=
  sorry

end min_odd_integers_is_zero_l89_89145


namespace proof_problem_l89_89410

variables (p q : Prop)

theorem proof_problem (hpq : p ∨ q) (hnp : ¬p) : q :=
by
  sorry

end proof_problem_l89_89410


namespace dice_even_odd_probability_l89_89959

theorem dice_even_odd_probability : 
  let p : ℚ := (nat.choose 8 4) * (1 / 2) ^ 8 in
  p = 35 / 128 :=
by
  -- proof steps would go here
  sorry

end dice_even_odd_probability_l89_89959


namespace part1_part2_l89_89831

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1 + Real.log x) / x
noncomputable def g (x : ℝ) : ℝ := 3 * Real.exp (1 - x) + 1

theorem part1 (a : ℝ) (x : ℝ) (h : 0 < x) : (∀ x > 0, f a x ≤ Real.exp 1) → a ≤ 1 := 
sorry

theorem part2 (a : ℝ) : (∃! x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 = g x1 ∧ f a x2 = g x2 ∧ f a x3 = g x3) → a = 3 :=
sorry

end part1_part2_l89_89831


namespace positive_difference_of_two_numbers_l89_89598

variable {x y : ℝ}

theorem positive_difference_of_two_numbers (h₁ : x + y = 8) (h₂ : x^2 - y^2 = 24) : |x - y| = 3 :=
by
  sorry

end positive_difference_of_two_numbers_l89_89598


namespace find_angle_BEC_l89_89029

theorem find_angle_BEC (A B C D E : Type) (angle_A angle_B angle_D angle_DEC angle_C angle_CED angle_BEC : ℝ) 
  (hA : angle_A = 50) (hB : angle_B = 90) (hD : angle_D = 70) (hDEC : angle_DEC = 20)
  (h_quadrilateral_sum: angle_A + angle_B + angle_C + angle_D = 360)
  (h_C : angle_C = 150)
  (h_CED : angle_CED = angle_C - angle_DEC)
  (h_BEC: angle_BEC = 180 - angle_B - angle_CED) : angle_BEC = 110 :=
by
  -- Definitions according to the given problem
  have h1 : angle_C = 360 - (angle_A + angle_B + angle_D) := by sorry
  have h2 : angle_CED = angle_C - angle_DEC := by sorry
  have h3 : angle_BEC = 180 - angle_B - angle_CED := by sorry

  -- Proving the required angle
  have h_goal : angle_BEC = 110 := by
    sorry  -- Actual proof steps go here

  exact h_goal

end find_angle_BEC_l89_89029


namespace ellen_smoothie_l89_89216

theorem ellen_smoothie :
  let yogurt := 0.1
  let orange_juice := 0.2
  let total_ingredients := 0.5
  let strawberries_used := total_ingredients - (yogurt + orange_juice)
  strawberries_used = 0.2 := by
  sorry

end ellen_smoothie_l89_89216


namespace value_of_a_l89_89136

theorem value_of_a
    (a b : ℝ)
    (h₁ : 0 < a ∧ 0 < b)
    (h₂ : a + b = 1)
    (h₃ : 21 * a^5 * b^2 = 35 * a^4 * b^3) :
    a = 5 / 8 :=
by
  sorry

end value_of_a_l89_89136


namespace hyperbola_focus_coordinates_l89_89898

theorem hyperbola_focus_coordinates :
  let a := 7
  let b := 11
  let h := 5
  let k := -3
  let c := Real.sqrt (a^2 + b^2)
  (∃ x y : ℝ, (x = h + c ∧ y = k) ∧ (∀ x' y', (x' = h + c ∧ y' = k) ↔ (x = x' ∧ y = y'))) :=
by
  sorry

end hyperbola_focus_coordinates_l89_89898


namespace solution_set_of_inequality_system_l89_89302

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0) ∧ (x + 1 > 0) ↔ (-1 < x ∧ x < 1) :=
by
  -- proof to be filled in
  sorry

end solution_set_of_inequality_system_l89_89302


namespace largest_result_is_0_point_1_l89_89165

theorem largest_result_is_0_point_1 : 
  max (5 * Real.sqrt 2 - 7) (max (7 - 5 * Real.sqrt 2) (max (|1 - 1|) 0.1)) = 0.1 := 
by
  -- We will prove this by comparing each value to 0.1
  sorry

end largest_result_is_0_point_1_l89_89165


namespace parallel_segments_have_equal_slopes_l89_89141

theorem parallel_segments_have_equal_slopes
  (A B X : ℝ × ℝ)
  (Y : ℝ × ℝ)
  (hA : A = (-5, -1))
  (hB : B = (2, -8))
  (hX : X = (2, 10))
  (hY1 : Y.1 = 20)
  (h_parallel : (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1)) :
  Y.2 = -8 :=
by
  sorry

end parallel_segments_have_equal_slopes_l89_89141


namespace erica_time_is_65_l89_89942

-- Definitions for the conditions
def dave_time : ℕ := 10
def chuck_time : ℕ := 5 * dave_time
def erica_time : ℕ := chuck_time + 3 * chuck_time / 10

-- The proof statement
theorem erica_time_is_65 : erica_time = 65 := by
  sorry

end erica_time_is_65_l89_89942


namespace dice_even_odd_equal_probability_l89_89955

theorem dice_even_odd_equal_probability :
  let p : ℚ := 35 / 128 in
  ∀ n : ℕ, n = 8 →
  ∀ k : ℕ, k = 4 →
  (∃ (binom : ℚ), binom = (Nat.choose n k)) →
  (∃ (prob : ℚ), prob = 1 / (2 ^ n)) →
  (∃ (total_prob : ℚ), total_prob = binom * prob) →
  total_prob = p :=
by
  intros n hn k hk binom hbinom prob hprob total_prob htotal_prob
  rw [hn, hk] at *
  cases hbinom with binom_val hbinom_val
  cases hprob with prob_val hprob_val
  rw hbinom_val at htotal_prob
  rw hprob_val at htotal_prob
  sorry

end dice_even_odd_equal_probability_l89_89955


namespace pencils_across_diameter_l89_89400

def radius_feet : ℝ := 14
def pencil_length_inches : ℝ := 6

theorem pencils_across_diameter : 
  (2 * radius_feet * 12 / pencil_length_inches) = 56 := 
by
  sorry

end pencils_across_diameter_l89_89400


namespace value_of_a_l89_89701

theorem value_of_a (a b : ℝ) (h1 : b = 2 * a) (h2 : b = 15 - 4 * a) : a = 5 / 2 :=
by
  sorry

end value_of_a_l89_89701


namespace quadratic_opposite_roots_l89_89513

theorem quadratic_opposite_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 + x2 = 0 ∧ x1 * x2 = k + 1) ↔ k = -2 :=
by
  sorry

end quadratic_opposite_roots_l89_89513
