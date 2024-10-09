import Mathlib

namespace positive_solution_is_perfect_square_l2376_237660

theorem positive_solution_is_perfect_square
  (t : ℤ)
  (n : ℕ)
  (h : n > 0)
  (root_cond : (n : ℤ)^2 + (4 * t - 1) * n + 4 * t^2 = 0) :
  ∃ k : ℕ, n = k^2 :=
sorry

end positive_solution_is_perfect_square_l2376_237660


namespace four_point_questions_l2376_237678

theorem four_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : y = 10 := 
sorry

end four_point_questions_l2376_237678


namespace payment_to_C_l2376_237676

/-- 
If A can complete a work in 6 days, B can complete the same work in 8 days, 
they signed to do the work for Rs. 2400 and completed the work in 3 days with 
the help of C, then the payment to C should be Rs. 300.
-/
theorem payment_to_C (total_payment : ℝ) (days_A : ℝ) (days_B : ℝ) (days_worked : ℝ) (portion_C : ℝ) :
   total_payment = 2400 ∧ days_A = 6 ∧ days_B = 8 ∧ days_worked = 3 ∧ portion_C = 1 / 8 →
   (portion_C * total_payment) = 300 := 
by 
  intros h
  cases h
  sorry

end payment_to_C_l2376_237676


namespace cricket_team_throwers_l2376_237618

def cricket_equation (T N : ℕ) := 
  (2 * N / 3 = 51 - T) ∧ (T + N = 58)

theorem cricket_team_throwers : 
  ∃ T : ℕ, ∃ N : ℕ, cricket_equation T N ∧ T = 37 :=
by
  sorry

end cricket_team_throwers_l2376_237618


namespace common_ratio_of_geometric_series_l2376_237642

theorem common_ratio_of_geometric_series (a1 a2 a3 : ℚ) (h1 : a1 = -4 / 7)
                                         (h2 : a2 = 14 / 3) (h3 : a3 = -98 / 9) :
  ∃ r : ℚ, r = a2 / a1 ∧ r = a3 / a2 ∧ r = -49 / 6 :=
by
  use -49 / 6
  sorry

end common_ratio_of_geometric_series_l2376_237642


namespace range_of_x_l2376_237628

theorem range_of_x (x : ℝ) : (4 : ℝ)^(2 * x - 1) > (1 / 2) ^ (-x - 4) → x > 2 := by
  sorry

end range_of_x_l2376_237628


namespace axis_of_symmetry_l2376_237680

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (4 - x)) : 
  ∀ x : ℝ, f x = f (4 - x) := 
  by sorry

end axis_of_symmetry_l2376_237680


namespace remainder_of_sum_mod_11_l2376_237699

theorem remainder_of_sum_mod_11 :
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 :=
by
  sorry

end remainder_of_sum_mod_11_l2376_237699


namespace pie_count_correct_l2376_237682

structure Berries :=
  (strawberries : ℕ)
  (blueberries : ℕ)
  (raspberries : ℕ)

def christine_picking : Berries := {strawberries := 10, blueberries := 8, raspberries := 20}

def rachel_picking : Berries :=
  let c := christine_picking
  {strawberries := 2 * c.strawberries,
   blueberries := 2 * c.blueberries,
   raspberries := c.raspberries / 2}

def total_berries (b1 b2 : Berries) : Berries :=
  {strawberries := b1.strawberries + b2.strawberries,
   blueberries := b1.blueberries + b2.blueberries,
   raspberries := b1.raspberries + b2.raspberries}

def pie_requirements : Berries := {strawberries := 3, blueberries := 2, raspberries := 4}

def max_pies (total : Berries) (requirements : Berries) : Berries :=
  {strawberries := total.strawberries / requirements.strawberries,
   blueberries := total.blueberries / requirements.blueberries,
   raspberries := total.raspberries / requirements.raspberries}

def correct_pies : Berries := {strawberries := 10, blueberries := 12, raspberries := 7}

theorem pie_count_correct :
  let total := total_berries christine_picking rachel_picking;
  max_pies total pie_requirements = correct_pies :=
by {
  sorry
}

end pie_count_correct_l2376_237682


namespace chris_raisins_nuts_l2376_237640

theorem chris_raisins_nuts (R N x : ℝ) 
  (hN : N = 4 * R) 
  (hxR : x * R = 0.15789473684210525 * (x * R + 4 * N)) :
  x = 3 :=
by
  sorry

end chris_raisins_nuts_l2376_237640


namespace find_value_of_a_3m_2n_l2376_237620

variable {a : ℝ} {m n : ℕ}
axiom h1 : a ^ m = 2
axiom h2 : a ^ n = 5

theorem find_value_of_a_3m_2n : a ^ (3 * m - 2 * n) = 8 / 25 := by
  sorry

end find_value_of_a_3m_2n_l2376_237620


namespace number_of_cannoneers_l2376_237641

-- Define the variables for cannoneers, women, and men respectively
variables (C W M : ℕ)

-- Define the conditions as assumptions
def conditions : Prop :=
  W = 2 * C ∧
  M = 2 * W ∧
  M + W = 378

-- Prove that the number of cannoneers is 63
theorem number_of_cannoneers (h : conditions C W M) : C = 63 :=
by sorry

end number_of_cannoneers_l2376_237641


namespace number_of_participants_l2376_237668

theorem number_of_participants (n : ℕ) (h : n - 1 = 25) : n = 26 := 
by sorry

end number_of_participants_l2376_237668


namespace parabola_equation_l2376_237658

-- Define the constants and the conditions
def parabola_focus : ℝ × ℝ := (3, 3)
def directrix : ℝ × ℝ × ℝ := (3, 7, -21)

theorem parabola_equation :
  ∃ a b c d e f : ℤ,
  a > 0 ∧
  Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd a b) c) d) e) f = 1 ∧
  (a : ℝ) * x^2 + (b : ℝ) * x * y + (c : ℝ) * y^2 + (d : ℝ) * x + (e : ℝ) * y + (f : ℝ) = 
  49 * x^2 - 42 * x * y + 9 * y^2 - 222 * x - 54 * y + 603 := sorry

end parabola_equation_l2376_237658


namespace ratio_population_A_to_F_l2376_237626

variable (F : ℕ)

def population_E := 6 * F
def population_D := 2 * population_E
def population_C := 8 * population_D
def population_B := 3 * population_C
def population_A := 5 * population_B

theorem ratio_population_A_to_F (F_pos : F > 0) :
  population_A F / F = 1440 := by
sorry

end ratio_population_A_to_F_l2376_237626


namespace quadratic_equation_roots_l2376_237621

-- Define the two numbers α and β such that their arithmetic and geometric means are given.
variables (α β : ℝ)

-- Arithmetic mean condition
def arithmetic_mean_condition : Prop := (α + β = 16)

-- Geometric mean condition
def geometric_mean_condition : Prop := (α * β = 225)

-- The quadratic equation with roots α and β
def quadratic_equation (x : ℝ) : ℝ := x^2 - 16 * x + 225

-- The proof statement
theorem quadratic_equation_roots (α β : ℝ) (h1 : arithmetic_mean_condition α β) (h2 : geometric_mean_condition α β) :
  ∃ x : ℝ, quadratic_equation x = 0 :=
sorry

end quadratic_equation_roots_l2376_237621


namespace factor_expression_l2376_237665

theorem factor_expression (a b c : ℝ) :
    (a + b) * (b + c) * (c + a) = 
    ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) :=
by
  sorry

end factor_expression_l2376_237665


namespace gcd_888_1147_l2376_237634

theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end gcd_888_1147_l2376_237634


namespace bowling_ball_volume_l2376_237645

open Real

noncomputable def remaining_volume (d_bowling_ball d1 d2 d3 d4 h1 h2 h3 h4 : ℝ) : ℝ :=
  let r_bowling_ball := d_bowling_ball / 2
  let v_bowling_ball := (4/3) * π * (r_bowling_ball ^ 3)
  let v_hole1 := π * ((d1 / 2) ^ 2) * h1
  let v_hole2 := π * ((d2 / 2) ^ 2) * h2
  let v_hole3 := π * ((d3 / 2) ^ 2) * h3
  let v_hole4 := π * ((d4 / 2) ^ 2) * h4
  v_bowling_ball - (v_hole1 + v_hole2 + v_hole3 + v_hole4)

theorem bowling_ball_volume :
  remaining_volume 40 3 3 4 5 10 10 12 8 = 10523.67 * π :=
by
  sorry

end bowling_ball_volume_l2376_237645


namespace third_diff_n_cube_is_const_6_third_diff_general_form_is_6_l2376_237613

-- Define the first finite difference function
def delta (f : ℕ → ℤ) (n : ℕ) : ℤ := f (n + 1) - f n

-- Define the second finite difference using the first
def delta2 (f : ℕ → ℤ) (n : ℕ) : ℤ := delta (delta f) n

-- Define the third finite difference using the second
def delta3 (f : ℕ → ℤ) (n : ℕ) : ℤ := delta (delta2 f) n

-- Prove the third finite difference of n^3 is 6
theorem third_diff_n_cube_is_const_6 :
  delta3 (fun (n : ℕ) => (n : ℤ)^3) = fun _ => 6 := 
by
  sorry

-- Prove the third finite difference of the general form function is 6
theorem third_diff_general_form_is_6 (a b c : ℤ) :
  delta3 (fun (n : ℕ) => (n : ℤ)^3 + a * (n : ℤ)^2 + b * (n : ℤ) + c) = fun _ => 6 := 
by
  sorry

end third_diff_n_cube_is_const_6_third_diff_general_form_is_6_l2376_237613


namespace right_angles_in_2_days_l2376_237654

-- Definitions
def hands_right_angle_twice_a_day (n : ℕ) : Prop :=
  n = 22

def right_angle_12_hour_frequency : Nat := 22
def hours_per_day : Nat := 24
def days : Nat := 2

-- Theorem to prove
theorem right_angles_in_2_days :
  hands_right_angle_twice_a_day right_angle_12_hour_frequency →
  right_angle_12_hour_frequency * (hours_per_day / 12) * days = 88 :=
by
  unfold hands_right_angle_twice_a_day
  intros 
  sorry

end right_angles_in_2_days_l2376_237654


namespace find_y_l2376_237627

theorem find_y (y : ℕ) : (8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10) = 2 ^ y → y = 33 := 
by 
  sorry

end find_y_l2376_237627


namespace shark_sightings_in_Daytona_Beach_l2376_237616

def CM : ℕ := 7

def DB : ℕ := 3 * CM + 5

theorem shark_sightings_in_Daytona_Beach : DB = 26 := by
  sorry

end shark_sightings_in_Daytona_Beach_l2376_237616


namespace three_digit_numbers_l2376_237672

theorem three_digit_numbers (n : ℕ) (a b c : ℕ) (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : n = 100 * a + 10 * b + c)
  (h3 : b^2 = a * c)
  (h4 : (10 * b + c) % 4 = 0) :
  n = 124 ∨ n = 248 ∨ n = 444 ∨ n = 964 ∨ n = 888 :=
sorry

end three_digit_numbers_l2376_237672


namespace mall_b_better_for_fewer_than_6_mall_equal_for_6_mall_a_better_for_more_than_6_l2376_237635

-- Definitions
def original_price : ℕ := 80
def discount_mallA (n : ℕ) : ℕ := min ((4 * n) * n) (80 * n / 2)
def discount_mallB (n : ℕ) : ℕ := (80 * n * 3) / 10

def total_cost_mallA (n : ℕ) : ℕ := (original_price * n) - discount_mallA n
def total_cost_mallB (n : ℕ) : ℕ := (original_price * n) - discount_mallB n

-- Theorem statements
theorem mall_b_better_for_fewer_than_6 (n : ℕ) (h : n < 6) : total_cost_mallA n > total_cost_mallB n := sorry
theorem mall_equal_for_6 (n : ℕ) (h : n = 6) : total_cost_mallA n = total_cost_mallB n := sorry
theorem mall_a_better_for_more_than_6 (n : ℕ) (h : n > 6) : total_cost_mallA n < total_cost_mallB n := sorry

end mall_b_better_for_fewer_than_6_mall_equal_for_6_mall_a_better_for_more_than_6_l2376_237635


namespace grandfather_older_than_grandmother_l2376_237669

noncomputable def Milena_age : ℕ := 7

noncomputable def Grandmother_age : ℕ := Milena_age * 9

noncomputable def Grandfather_age : ℕ := Milena_age + 58

theorem grandfather_older_than_grandmother :
  Grandfather_age - Grandmother_age = 2 := by
  sorry

end grandfather_older_than_grandmother_l2376_237669


namespace smallest_x_value_l2376_237655

theorem smallest_x_value (x : ℝ) (h : |4 * x + 9| = 37) : x = -11.5 :=
sorry

end smallest_x_value_l2376_237655


namespace find_m_of_quadratic_function_l2376_237656

theorem find_m_of_quadratic_function :
  ∀ (m : ℝ), (m + 1 ≠ 0) → ((m + 1) * x ^ (m^2 + 1) + 5 = a * x^2 + b * x + c) → m = 1 :=
by
  intro m h h_quad
  -- Proof Here
  sorry

end find_m_of_quadratic_function_l2376_237656


namespace complex_problem_l2376_237695

theorem complex_problem (a b : ℝ) (i : ℂ) (hi : i^2 = -1) 
  (h : (a - 2 * i) * i = b - i) : a + b = 1 :=
by
  sorry

end complex_problem_l2376_237695


namespace weight_ratio_l2376_237643

noncomputable def students_weight : ℕ := 79
noncomputable def siblings_total_weight : ℕ := 116

theorem weight_ratio (S W : ℕ) (h1 : siblings_total_weight = S + W) (h2 : students_weight = S):
  (S - 5) / (siblings_total_weight - S) = 2 :=
by
  sorry

end weight_ratio_l2376_237643


namespace angle_of_isosceles_trapezoid_in_monument_l2376_237667

-- Define the larger interior angle x of an isosceles trapezoid in the monument
def larger_interior_angle_of_trapezoid (x : ℝ) : Prop :=
  ∃ n : ℕ, 
    n = 12 ∧
    ∃ α : ℝ, 
      α = 360 / (2 * n) ∧
      ∃ θ : ℝ, 
        θ = (180 - α) / 2 ∧
        x = 180 - θ

-- The theorem stating the larger interior angle x is 97.5 degrees
theorem angle_of_isosceles_trapezoid_in_monument : larger_interior_angle_of_trapezoid 97.5 :=
by 
  sorry

end angle_of_isosceles_trapezoid_in_monument_l2376_237667


namespace gcd_840_1764_l2376_237663

theorem gcd_840_1764 : Int.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l2376_237663


namespace unit_A_saplings_l2376_237609

theorem unit_A_saplings 
  (Y B D J : ℕ)
  (h1 : J = 2 * Y + 20)
  (h2 : J = 3 * B + 24)
  (h3 : J = 5 * D - 45)
  (h4 : J + Y + B + D = 2126) :
  J = 1050 :=
by sorry

end unit_A_saplings_l2376_237609


namespace solve_for_n_l2376_237687

-- Define the problem statement
theorem solve_for_n : ∃ n : ℕ, (3 * n^2 + n = 219) ∧ (n = 9) := 
sorry

end solve_for_n_l2376_237687


namespace average_score_of_remaining_students_correct_l2376_237679

noncomputable def average_score_remaining_students (n : ℕ) (h_n : n > 15) (avg_all : ℚ) (avg_subgroup : ℚ) : ℚ :=
if h_avg_all : avg_all = 10 ∧ avg_subgroup = 16 then
  (10 * n - 240) / (n - 15)
else
  0

theorem average_score_of_remaining_students_correct (n : ℕ) (h_n : n > 15) :
  (average_score_remaining_students n h_n 10 16) = (10 * n - 240) / (n - 15) :=
by
  dsimp [average_score_remaining_students]
  split_ifs with h_avg
  · sorry
  · sorry

end average_score_of_remaining_students_correct_l2376_237679


namespace inequality_solution_l2376_237650

open Set Real

theorem inequality_solution (x : ℝ) :
  (1 / (x + 1) + 3 / (x + 7) ≥ 2 / 3) ↔ (x ∈ Ioo (-7 : ℝ) (-4) ∪ Ioo (-1) (2) ∪ {(-4 : ℝ), 2}) :=
by sorry

end inequality_solution_l2376_237650


namespace t50_mod_7_l2376_237698

def T (n : ℕ) : ℕ :=
  match n with
  | 0     => 3
  | n + 1 => 3 ^ T n

theorem t50_mod_7 : T 50 % 7 = 6 := sorry

end t50_mod_7_l2376_237698


namespace find_sample_size_l2376_237622

def sample_size (sample : List ℕ) : ℕ :=
  sample.length

theorem find_sample_size :
  sample_size (List.replicate 500 0) = 500 :=
by
  sorry

end find_sample_size_l2376_237622


namespace total_time_of_four_sets_of_stairs_l2376_237664

def time_first : ℕ := 15
def time_increment : ℕ := 10
def num_sets : ℕ := 4

theorem total_time_of_four_sets_of_stairs :
  let a := time_first
  let d := time_increment
  let n := num_sets
  let l := a + (n - 1) * d
  let S := n / 2 * (a + l)
  S = 120 :=
by
  sorry

end total_time_of_four_sets_of_stairs_l2376_237664


namespace matrix_inverse_l2376_237629

-- Define the given matrix
def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![5, 4], ![-2, 8]]

-- Define the expected inverse matrix
def A_inv_expected : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1/6, -1/12], ![1/24, 5/48]]

-- The main statement: Prove that the inverse of A is equal to the expected inverse
theorem matrix_inverse :
  A⁻¹ = A_inv_expected := sorry

end matrix_inverse_l2376_237629


namespace find_parabola_constant_l2376_237670

theorem find_parabola_constant (a b c : ℝ) (h_vertex : ∀ y, (4:ℝ) = -5 / 4 * y * y + 5 / 2 * y + c)
  (h_point : (-1:ℝ) = -5 / 4 * (3:ℝ) ^ 2 + 5 / 2 * (3:ℝ) + c ) :
  c = 11 / 4 :=
sorry

end find_parabola_constant_l2376_237670


namespace largest_increase_between_2006_and_2007_l2376_237619

-- Define the number of students taking the AMC in each year
def students_2002 := 50
def students_2003 := 55
def students_2004 := 63
def students_2005 := 70
def students_2006 := 75
def students_2007_AMC10 := 90
def students_2007_AMC12 := 15

-- Define the total number of students participating in any AMC contest each year
def total_students_2002 := students_2002
def total_students_2003 := students_2003
def total_students_2004 := students_2004
def total_students_2005 := students_2005
def total_students_2006 := students_2006
def total_students_2007 := students_2007_AMC10 + students_2007_AMC12

-- Function to calculate percentage increase
def percentage_increase (old new : ℕ) : ℚ :=
  ((new - old : ℕ) : ℚ) / old * 100

-- Calculate percentage increases between the years
def inc_2002_2003 := percentage_increase total_students_2002 total_students_2003
def inc_2003_2004 := percentage_increase total_students_2003 total_students_2004
def inc_2004_2005 := percentage_increase total_students_2004 total_students_2005
def inc_2005_2006 := percentage_increase total_students_2005 total_students_2006
def inc_2006_2007 := percentage_increase total_students_2006 total_students_2007

-- Prove that the largest percentage increase is between 2006 and 2007
theorem largest_increase_between_2006_and_2007 :
  inc_2006_2007 > inc_2005_2006 ∧
  inc_2006_2007 > inc_2004_2005 ∧
  inc_2006_2007 > inc_2003_2004 ∧
  inc_2006_2007 > inc_2002_2003 := 
by {
  sorry
}

end largest_increase_between_2006_and_2007_l2376_237619


namespace problem1_problem2_problem3_problem4_l2376_237690

-- Problem 1
theorem problem1 (x : ℝ) : 0.75 * x = (1 / 2) * 12 → x = 8 := 
by 
  intro h
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (0.7 / x) = (14 / 5) → x = 0.25 := 
by 
  intro h
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (1 / 6) * x = (2 / 15) * (2 / 3) → x = (8 / 15) := 
by 
  intro h
  sorry

-- Problem 4
theorem problem4 (x : ℝ) : 4.5 * x = 4 * 27 → x = 24 := 
by 
  intro h
  sorry

end problem1_problem2_problem3_problem4_l2376_237690


namespace trace_bag_weight_is_two_l2376_237608

-- Define the weights of Gordon's shopping bags
def weight_gordon1 : ℕ := 3
def weight_gordon2 : ℕ := 7

-- Summarize Gordon's total weight
def total_weight_gordon : ℕ := weight_gordon1 + weight_gordon2

-- Provide necessary conditions from problem statement
def trace_bags_count : ℕ := 5
def trace_total_weight : ℕ := total_weight_gordon
def trace_one_bag_weight : ℕ := trace_total_weight / trace_bags_count

theorem trace_bag_weight_is_two : trace_one_bag_weight = 2 :=
by 
  -- Placeholder for proof
  sorry

end trace_bag_weight_is_two_l2376_237608


namespace value_of_x_l2376_237659

theorem value_of_x (x : ℝ) : (1 / 8) * (2 : ℝ) ^ 32 = (4 : ℝ) ^ x → x = 29 / 2 :=
by
  sorry

end value_of_x_l2376_237659


namespace g_increasing_in_interval_l2376_237603

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + a * x + 2
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * x + a
noncomputable def f'' (a : ℝ) (x : ℝ) : ℝ := 2 * x - 2 * a

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f'' a x / x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 1 - a / (x^2)

theorem g_increasing_in_interval (a : ℝ) (h : a < 1) :
  ∀ x : ℝ, 1 < x → 0 < g' a x := by
  sorry

end g_increasing_in_interval_l2376_237603


namespace point_B_in_fourth_quadrant_l2376_237607

theorem point_B_in_fourth_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : (b > 0 ∧ a < 0) :=
by {
    sorry
}

end point_B_in_fourth_quadrant_l2376_237607


namespace range_of_m_l2376_237696

theorem range_of_m (p_false : ¬ (∀ x : ℝ, ∃ m : ℝ, 2 * x + 1 + m = 0)) : ∀ m : ℝ, m ≤ 1 :=
sorry

end range_of_m_l2376_237696


namespace lcm_36_90_eq_180_l2376_237686

theorem lcm_36_90_eq_180 : Nat.lcm 36 90 = 180 := 
by 
  sorry

end lcm_36_90_eq_180_l2376_237686


namespace average_age_of_women_l2376_237611

variable {A W : ℝ}

theorem average_age_of_women (A : ℝ) (h : 12 * (A + 3) = 12 * A - 90 + W) : 
  W / 3 = 42 := by
  sorry

end average_age_of_women_l2376_237611


namespace cid_earnings_l2376_237636

variable (x : ℕ)
variable (oil_change_price repair_price car_wash_price : ℕ)
variable (cars_repaired cars_washed total_earnings : ℕ)

theorem cid_earnings :
  (oil_change_price = 20) →
  (repair_price = 30) →
  (car_wash_price = 5) →
  (cars_repaired = 10) →
  (cars_washed = 15) →
  (total_earnings = 475) →
  (oil_change_price * x + repair_price * cars_repaired + car_wash_price * cars_washed = total_earnings) →
  x = 5 := by sorry

end cid_earnings_l2376_237636


namespace shaded_fraction_in_fourth_square_l2376_237646

theorem shaded_fraction_in_fourth_square : 
  ∀ (f : ℕ → ℕ), (f 1 = 1)
  ∧ (f 2 = 3)
  ∧ (f 3 = 5)
  ∧ (f 4 = f 3 + (3 - 1) + (5 - 3))
  ∧ (f 4 * 2 = 14)
  → (f 4 = 7)
  → (f 4 / 16 = 7 / 16) :=
sorry

end shaded_fraction_in_fourth_square_l2376_237646


namespace train_speed_in_kmh_l2376_237689

theorem train_speed_in_kmh (length_of_train : ℕ) (time_to_cross : ℕ) (speed_in_m_per_s : ℕ) (speed_in_km_per_h : ℕ) :
  length_of_train = 300 →
  time_to_cross = 12 →
  speed_in_m_per_s = length_of_train / time_to_cross →
  speed_in_km_per_h = speed_in_m_per_s * 3600 / 1000 →
  speed_in_km_per_h = 90 :=
by
  sorry

end train_speed_in_kmh_l2376_237689


namespace select_representatives_l2376_237610

theorem select_representatives
  (female_count : ℕ) (male_count : ℕ)
  (female_count_eq : female_count = 4)
  (male_count_eq : male_count = 6) :
  female_count * male_count = 24 := by
  sorry

end select_representatives_l2376_237610


namespace value_of_x_squared_plus_reciprocal_squared_l2376_237691

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 47 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = 7 :=
by sorry

end value_of_x_squared_plus_reciprocal_squared_l2376_237691


namespace geometric_sequence_S6_l2376_237648

variable (a : ℕ → ℝ) -- represents the geometric sequence

noncomputable def S (n : ℕ) : ℝ :=
if n = 0 then 0 else ((a 0) * (1 - (a 1 / a 0) ^ n)) / (1 - a 1 / a 0)

theorem geometric_sequence_S6 (h : ∀ n, a n = (a 0) * (a 1 / a 0) ^ n) :
  S a 2 = 6 ∧ S a 4 = 18 → S a 6 = 42 := 
by 
  intros h1
  sorry

end geometric_sequence_S6_l2376_237648


namespace no_valid_n_l2376_237675

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

def greatest_prime_factor (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n.minFac

theorem no_valid_n (n : ℕ) (h1 : n > 1)
  (h2 : is_prime (greatest_prime_factor n))
  (h3 : greatest_prime_factor n = Nat.sqrt n)
  (h4 : is_prime (greatest_prime_factor (n + 36)))
  (h5 : greatest_prime_factor (n + 36) = Nat.sqrt (n + 36)) :
  false :=
sorry

end no_valid_n_l2376_237675


namespace fraction_of_marbles_taken_away_l2376_237694

theorem fraction_of_marbles_taken_away (Chris_marbles Ryan_marbles remaining_marbles total_marbles taken_away_marbles : ℕ) 
    (hChris : Chris_marbles = 12) 
    (hRyan : Ryan_marbles = 28) 
    (hremaining : remaining_marbles = 20) 
    (htotal : total_marbles = Chris_marbles + Ryan_marbles) 
    (htaken_away : taken_away_marbles = total_marbles - remaining_marbles) : 
    (taken_away_marbles : ℚ) / total_marbles = 1 / 2 := 
by 
  sorry

end fraction_of_marbles_taken_away_l2376_237694


namespace matrix_multiplication_l2376_237684

variable (A B : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_multiplication :
  (A - B = A * B) →
  (A * B = ![![7, -2], ![4, -3]]) →
  (B * A = ![![6, -2], ![4, -4]]) :=
by
  intros h₁ h₂
  sorry

end matrix_multiplication_l2376_237684


namespace smallest_n_mod_equiv_l2376_237688

theorem smallest_n_mod_equiv (n : ℕ) (h : 5 * n ≡ 4960 [MOD 31]) : n = 31 := by 
  sorry

end smallest_n_mod_equiv_l2376_237688


namespace C_share_correct_l2376_237612

def investment_A := 27000
def investment_B := 72000
def investment_C := 81000
def total_profit := 80000

def gcd_investment : ℕ := Nat.gcd investment_A (Nat.gcd investment_B investment_C)
def ratio_A : ℕ := investment_A / gcd_investment
def ratio_B : ℕ := investment_B / gcd_investment
def ratio_C : ℕ := investment_C / gcd_investment
def total_parts : ℕ := ratio_A + ratio_B + ratio_C

def C_share : ℕ := (ratio_C / total_parts) * total_profit

theorem C_share_correct : C_share = 36000 := 
by sorry

end C_share_correct_l2376_237612


namespace work_ratio_l2376_237652

theorem work_ratio (m b : ℝ) (h1 : 12 * m + 16 * b = 1 / 5) (h2 : 13 * m + 24 * b = 1 / 4) : m = 2 * b :=
by sorry

end work_ratio_l2376_237652


namespace red_marbles_count_l2376_237606

variable (n : ℕ)

-- Conditions
def ratio_green_yellow_red := (3 * n, 4 * n, 2 * n)
def not_red_marbles := 3 * n + 4 * n = 63

-- Goal
theorem red_marbles_count (hn : not_red_marbles n) : 2 * n = 18 :=
by
  sorry

end red_marbles_count_l2376_237606


namespace maximize_z_l2376_237615

open Real

theorem maximize_z (x y : ℝ) (h1 : x + y ≤ 10) (h2 : 3 * x + y ≤ 18) (h3 : 0 ≤ x) (h4 : 0 ≤ y) :
  (∀ x y, x + y ≤ 10 ∧ 3 * x + y ≤ 18 ∧ 0 ≤ x ∧ 0 ≤ y → x + y / 2 ≤ 7) :=
by
  sorry

end maximize_z_l2376_237615


namespace fraction_product_l2376_237631

theorem fraction_product :
  (7 / 4 : ℚ) * (14 / 35) * (21 / 12) * (28 / 56) * (49 / 28) * (42 / 84) * (63 / 36) * (56 / 112) = (1201 / 12800) := 
by
  sorry

end fraction_product_l2376_237631


namespace value_of_Y_l2376_237604

theorem value_of_Y :
  let part1 := 15 * 180 / 100  -- 15% of 180
  let part2 := part1 - part1 / 3  -- one-third less than 15% of 180
  let part3 := 24.5 * (2 * 270 / 3) / 100  -- 24.5% of (2/3 * 270)
  let part4 := (5.4 * 2) / (0.25 * 0.25)  -- (5.4 * 2) / (0.25)^2
  let Y := part2 + part3 - part4
  Y = -110.7 := by
    -- proof skipped
    sorry

end value_of_Y_l2376_237604


namespace geometric_arithmetic_sum_l2376_237605

theorem geometric_arithmetic_sum {a : Nat → ℝ} {b : Nat → ℝ} 
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d)
  (h_condition : a 3 * a 11 = 4 * a 7)
  (h_equal : a 7 = b 7) :
  b 5 + b 9 = 8 :=
sorry

end geometric_arithmetic_sum_l2376_237605


namespace teams_working_together_l2376_237661

theorem teams_working_together
    (m n : ℕ) 
    (hA : ∀ t : ℕ, t = m → (t ≥ 0)) 
    (hB : ∀ t : ℕ, t = n → (t ≥ 0)) : 
  ∃ t : ℕ, t = (m * n) / (m + n) :=
by
  sorry

end teams_working_together_l2376_237661


namespace imaginary_part_of_z_l2376_237693

-- Define the imaginary unit i where i^2 = -1
def imaginary_unit : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := (2 + imaginary_unit) * (1 - imaginary_unit)

-- State the theorem to prove the imaginary part of z
theorem imaginary_part_of_z : Complex.im z = -1 := by
  sorry

end imaginary_part_of_z_l2376_237693


namespace find_a_from_binomial_l2376_237623

variable (x : ℝ) (a : ℝ)

def binomial_term (r : ℕ) : ℝ :=
  (Nat.choose 5 r) * ((-a)^r) * x^(5 - 2 * r)

theorem find_a_from_binomial :
  (∃ x : ℝ, ∃ a : ℝ, (binomial_term x a 1 = 10)) → a = -2 :=
by 
  sorry

end find_a_from_binomial_l2376_237623


namespace circle_incircle_tangent_radius_l2376_237600

theorem circle_incircle_tangent_radius (r1 r2 r3 : ℕ) (k : ℕ) (h1 : r1 = 1) (h2 : r2 = 4) (h3 : r3 = 9) : 
  k = 11 :=
by
  -- Definitions according to the problem
  let k₁ := r1
  let k₂ := r2
  let k₃ := r3
  -- Hypotheses given by the problem
  have h₁ : k₁ = 1 := h1
  have h₂ : k₂ = 4 := h2
  have h₃ : k₃ = 9 := h3
  -- Prove the radius of the incircle k
  sorry

end circle_incircle_tangent_radius_l2376_237600


namespace fractions_equiv_l2376_237666

theorem fractions_equiv:
  (8 : ℝ) / (7 * 67) = (0.8 : ℝ) / (0.7 * 67) :=
by
  sorry

end fractions_equiv_l2376_237666


namespace circular_paper_pieces_needed_l2376_237632

-- Definition of the problem conditions
def side_length_dm := 10
def side_length_cm := side_length_dm * 10
def perimeter_cm := 4 * side_length_cm
def number_of_sides := 4
def semicircles_per_side := 1
def total_semicircles := number_of_sides * semicircles_per_side
def semicircles_to_circles := 2
def total_circles := total_semicircles / semicircles_to_circles
def paper_pieces_per_circle := 20

-- Main theorem stating the problem and the answer.
theorem circular_paper_pieces_needed : (total_circles * paper_pieces_per_circle) = 40 :=
by sorry

end circular_paper_pieces_needed_l2376_237632


namespace chess_tournament_games_l2376_237647

theorem chess_tournament_games (n : ℕ) (h : n = 16) :
  (n * (n - 1) * 2) / 2 = 480 :=
by
  rw [h]
  simp
  norm_num
  sorry

end chess_tournament_games_l2376_237647


namespace a_gt_abs_b_suff_not_necc_l2376_237625

theorem a_gt_abs_b_suff_not_necc (a b : ℝ) (h : a > |b|) : 
  a^2 > b^2 ∧ ∀ a b : ℝ, (a^2 > b^2 → |a| > |b|) → ¬ (a < -|b|) := 
by
  sorry

end a_gt_abs_b_suff_not_necc_l2376_237625


namespace martin_walk_distance_l2376_237644

-- Define the conditions
def time : ℝ := 6 -- Martin's walking time in hours
def speed : ℝ := 2 -- Martin's walking speed in miles per hour

-- Define the target distance
noncomputable def distance : ℝ := 12 -- Distance from Martin's house to Lawrence's house

-- The theorem to prove the target distance given the conditions
theorem martin_walk_distance : (speed * time = distance) :=
by
  sorry

end martin_walk_distance_l2376_237644


namespace find_a_l2376_237657

theorem find_a (a : ℝ) (h : ∃ (b : ℝ), (16 * (x : ℝ) * x) + 40 * x + a = (4 * x + b) ^ 2) : a = 25 := sorry

end find_a_l2376_237657


namespace solution_set_of_f_inequality_l2376_237662

variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_deriv : ∀ x, f' x < f x)
variable (h_even : ∀ x, f (x + 2) = f (-x + 2))
variable (h_initial : f 0 = Real.exp 4)

theorem solution_set_of_f_inequality :
  {x : ℝ | f x < Real.exp x} = {x : ℝ | x > 4} := 
sorry

end solution_set_of_f_inequality_l2376_237662


namespace speed_of_stream_l2376_237683

variable (D : ℝ) -- The distance rowed in both directions
variable (vs : ℝ) -- The speed of the stream
variable (Vb : ℝ := 78) -- The speed of the boat in still water

theorem speed_of_stream (h : (D / (Vb - vs) = 2 * (D / (Vb + vs)))) : vs = 26 := by
    sorry

end speed_of_stream_l2376_237683


namespace unknown_subtraction_problem_l2376_237602

theorem unknown_subtraction_problem (x y : ℝ) (h1 : x = 40) (h2 : x / 4 * 5 + 10 - y = 48) : y = 12 :=
by
  sorry

end unknown_subtraction_problem_l2376_237602


namespace find_n_l2376_237674

theorem find_n (m n : ℕ) (h1: m = 34)
               (h2: (1^(m+1) / 5^(m+1)) * (1^n / 4^n) = 1 / (2 * 10^35)) : 
               n = 18 :=
by
  sorry

end find_n_l2376_237674


namespace find_length_of_GH_l2376_237639

variable {A B C F G H : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
          [MetricSpace F] [MetricSpace G] [MetricSpace H]

variables (AB BC Res : ℝ)
variables (ratio1 ratio2 : ℝ)
variable (similar : SimilarTriangles A B C F G H)

def length_of_GH (GH : ℝ) : Prop :=
  GH = 15

theorem find_length_of_GH (h1 : AB = 15) (h2 : BC = 25) (h3 : ratio1 = 5) (h4 : ratio2 = 3)
  (h5 : similar) : ∃ GH, length_of_GH GH :=
by
  have ratio : ratio2 / ratio1 = 3 / 5 := by assumption
  sorry

end find_length_of_GH_l2376_237639


namespace algebraic_expression_value_l2376_237673

theorem algebraic_expression_value (a : ℝ) (h : a^2 + 2023 * a - 1 = 0) : 
  a * (a + 1) * (a - 1) + 2023 * a^2 + 1 = 1 :=
by
  sorry

end algebraic_expression_value_l2376_237673


namespace cannot_fit_all_pictures_l2376_237624

theorem cannot_fit_all_pictures 
  (typeA_capacity : Nat) (typeB_capacity : Nat) (typeC_capacity : Nat)
  (typeA_count : Nat) (typeB_count : Nat) (typeC_count : Nat)
  (total_pictures : Nat)
  (h1 : typeA_capacity = 12)
  (h2 : typeB_capacity = 18)
  (h3 : typeC_capacity = 24)
  (h4 : typeA_count = 6)
  (h5 : typeB_count = 4)
  (h6 : typeC_count = 3)
  (h7 : total_pictures = 480) :
  (typeA_capacity * typeA_count + typeB_capacity * typeB_count + typeC_capacity * typeC_count < total_pictures) :=
  by sorry

end cannot_fit_all_pictures_l2376_237624


namespace tank_capacity_l2376_237601

variable (x : ℝ) -- Total capacity of the tank

theorem tank_capacity (h1 : x / 8 = 120 / (1 / 2 - 1 / 8)) :
  x = 320 :=
by
  sorry

end tank_capacity_l2376_237601


namespace three_character_license_plates_l2376_237685

theorem three_character_license_plates :
  let consonants := 20
  let vowels := 6
  (consonants * consonants * vowels = 2400) :=
by
  sorry

end three_character_license_plates_l2376_237685


namespace students_with_grade_B_and_above_l2376_237633

theorem students_with_grade_B_and_above (total_students : ℕ) (percent_below_B : ℕ) 
(h1 : total_students = 60) (h2 : percent_below_B = 40) : 
(total_students * (100 - percent_below_B) / 100) = 36 := by
  sorry

end students_with_grade_B_and_above_l2376_237633


namespace alvin_age_l2376_237681

theorem alvin_age (A S : ℕ) (h_s : S = 10) (h_cond : S = 1/2 * A - 5) : A = 30 := by
  sorry

end alvin_age_l2376_237681


namespace geometric_sequence_s4_l2376_237677

noncomputable def geometric_sequence_sum : ℕ → ℝ → ℝ → ℝ
| 0, a1, q => 0
| (n+1), a1, q => a1 * (1 - q^(n+1)) / (1 - q)

variable (a1 q : ℝ) (n : ℕ)

theorem geometric_sequence_s4  (h1 : a1 * (q^1) * (q^3) = 16) (h2 : geometric_sequence_sum 2 a1 q + a1 * (q^2) = 7) :
  geometric_sequence_sum 3 a1 q = 15 :=
sorry

end geometric_sequence_s4_l2376_237677


namespace standard_ellipse_eq_l2376_237692

def ellipse_standard_eq (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

theorem standard_ellipse_eq (P: ℝ × ℝ) (Q: ℝ × ℝ) (a b : ℝ) (h1 : P = (-3, 0)) (h2 : Q = (0, -2)) :
  ellipse_standard_eq 3 2 x y :=
by
  sorry

end standard_ellipse_eq_l2376_237692


namespace yolk_count_proof_l2376_237637

-- Define the conditions of the problem
def eggs_in_carton : ℕ := 12
def double_yolk_eggs : ℕ := 5
def single_yolk_eggs : ℕ := eggs_in_carton - double_yolk_eggs
def yolks_in_double_yolk_eggs : ℕ := double_yolk_eggs * 2
def yolks_in_single_yolk_eggs : ℕ := single_yolk_eggs
def total_yolks : ℕ := yolks_in_single_yolk_eggs + yolks_in_double_yolk_eggs

-- Stating the theorem to prove the total number of yolks is 17
theorem yolk_count_proof : total_yolks = 17 := 
by
  sorry

end yolk_count_proof_l2376_237637


namespace sum_of_mnp_l2376_237653

theorem sum_of_mnp (m n p : ℕ) (h_gcd : gcd m (gcd n p) = 1)
  (h : ∀ x : ℝ, 5 * x^2 - 11 * x + 6 = 0 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) :
  m + n + p = 22 :=
by
  sorry

end sum_of_mnp_l2376_237653


namespace count_divisibles_in_range_l2376_237671

theorem count_divisibles_in_range :
  let lower_bound := (2:ℤ)^10
  let upper_bound := (2:ℤ)^18
  let divisor := (2:ℤ)^9 
  (upper_bound - lower_bound) / divisor + 1 = 511 :=
by 
  sorry

end count_divisibles_in_range_l2376_237671


namespace sqrt_sum_eq_pow_l2376_237638

/-- 
For the value \( k = 3/2 \), the expression \( \sqrt{2016} + \sqrt{56} \) equals \( 14^k \)
-/
theorem sqrt_sum_eq_pow (k : ℝ) (h : k = 3 / 2) : 
  (Real.sqrt 2016 + Real.sqrt 56) = 14 ^ k := 
by 
  sorry

end sqrt_sum_eq_pow_l2376_237638


namespace problem1_problem2_l2376_237617

-- Problem 1
theorem problem1 :
  2 * Real.cos (Real.pi / 4) + (Real.pi - Real.sqrt 3)^0 - Real.sqrt 8 = 1 - Real.sqrt 2 := 
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) (h : m ≠ 1) :
  (2 / (m - 1) + 1) / ((2 * m + 2) / (m^2 - 2 * m + 1)) = (m - 1) / 2 :=
by
  sorry

end problem1_problem2_l2376_237617


namespace positive_difference_median_mode_l2376_237697

-- Definition of the data set
def data : List ℕ := [12, 13, 14, 15, 15, 22, 22, 22, 31, 31, 38, 39, 40, 41, 42, 43, 52, 58, 59]

-- Definition of the mode
def mode (l : List ℕ) : ℕ := 22  -- Specific to the data set provided

-- Definition of the median
def median (l : List ℕ) : ℕ := 31  -- Specific to the data set provided

-- Proof statement
theorem positive_difference_median_mode : 
  (median data - mode data) = 9 := by 
  sorry

end positive_difference_median_mode_l2376_237697


namespace fraction_of_orange_juice_in_mixture_l2376_237614

theorem fraction_of_orange_juice_in_mixture
  (capacity_pitcher : ℕ)
  (fraction_first_pitcher : ℚ)
  (fraction_second_pitcher : ℚ)
  (condition1 : capacity_pitcher = 500)
  (condition2 : fraction_first_pitcher = 1/4)
  (condition3 : fraction_second_pitcher = 3/7) :
  (125 + 500 * (3/7)) / (2 * 500) = 95 / 280 :=
by
  sorry

end fraction_of_orange_juice_in_mixture_l2376_237614


namespace measure_of_B_l2376_237651

-- Define the conditions (angles and their relationships)
variable (angle_P angle_R angle_O angle_B angle_L angle_S : ℝ)
variable (sum_of_angles : angle_P + angle_R + angle_O + angle_B + angle_L + angle_S = 720)
variable (supplementary_O_S : angle_O + angle_S = 180)
variable (right_angle_L : angle_L = 90)
variable (congruent_angles : angle_P = angle_R ∧ angle_R = angle_B)

-- Prove the measure of angle B
theorem measure_of_B : angle_B = 150 := by
  sorry

end measure_of_B_l2376_237651


namespace liliane_has_44_44_more_cookies_l2376_237649

variables (J : ℕ) (L O : ℕ) (totalCookies : ℕ)

def liliane_has_more_30_percent (J L : ℕ) : Prop :=
  L = J + (3 * J / 10)

def oliver_has_less_10_percent (J O : ℕ) : Prop :=
  O = J - (J / 10)

def total_cookies (J L O totalCookies : ℕ) : Prop :=
  J + L + O = totalCookies

theorem liliane_has_44_44_more_cookies
  (h1 : liliane_has_more_30_percent J L)
  (h2 : oliver_has_less_10_percent J O)
  (h3 : total_cookies J L O totalCookies)
  (h4 : totalCookies = 120) :
  (L - O) * 100 / O = 4444 / 100 := sorry

end liliane_has_44_44_more_cookies_l2376_237649


namespace unique_number_not_in_range_of_g_l2376_237630

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range_of_g 
  (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : g 5 a b c d = 5) (h6 : g 25 a b c d = 25) 
  (h7 : ∀ x, x ≠ -d/c → g (g x a b c d) a b c d = x) :
  ∃ r, r = 15 ∧ ∀ y, g y a b c d ≠ r := 
by
  sorry

end unique_number_not_in_range_of_g_l2376_237630
