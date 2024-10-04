import Mathlib

namespace quadratic_coefficients_l676_676542

theorem quadratic_coefficients :
  ∀ x : ℝ, 3 * x^2 = 5 * x - 1 → (∃ a b c : ℝ, a = 3 ∧ b = -5 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x h
  use 3, -5, 1
  sorry

end quadratic_coefficients_l676_676542


namespace solve_equation_l676_676428

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) : (2 / x = 1 / (x + 1)) → x = -2 :=
begin
  sorry
end

end solve_equation_l676_676428


namespace cross_product_zero_l676_676417

theorem cross_product_zero (x y : ℝ) (u : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) 
(h1 : u = (3, x, -9))
(h2 : v = (4, 6, y))
(h3 : u ×ₓ v = (0, 0, 0)) :
(x = 9/2 ∧ y = -12) := by
sorry

end cross_product_zero_l676_676417


namespace find_multiple_l676_676038

theorem find_multiple (m : ℤ) : 38 + m * 43 = 124 → m = 2 := by
  intro h
  sorry

end find_multiple_l676_676038


namespace die_rolls_divisible_by_4_l676_676537

theorem die_rolls_divisible_by_4 :
  (∃ p : ℚ, 
     (∀ n p, binomial_prob n p 8 (1/2) ≥ 2) →
     p = 247/256) :=
begin
  sorry
end

end die_rolls_divisible_by_4_l676_676537


namespace proof_problem_l676_676554

noncomputable def control_group_weights : List ℝ := [15.2, 18.8, 20.2, 21.3, 22.5, 23.2, 25.8, 26.5, 27.5, 30.1, 32.6, 34.3, 34.8, 35.6, 35.6, 35.8, 36.2, 37.3, 40.5, 43.2]
noncomputable def experimental_group_weights : List ℝ := [7.8, 9.2, 11.4, 12.4, 13.2, 15.5, 16.5, 18.0, 18.8, 19.2, 19.8, 20.2, 21.6, 22.8, 23.6, 23.9, 25.1, 28.2, 32.3, 36.5]

theorem proof_problem :
  (1) (List.sum experimental_group_weights) / 20 = 19.8 ∧
  (2) (let combined_weights := control_group_weights ++ experimental_group_weights;
          m := (combined_weights.sorted.nth 19 + combined_weights.sorted.nth 20) / 2;
          control_less_than_m := control_group_weights.filter (λ x => x < m).length;
          control_geq_m := control_group_weights.filter (λ x => x >= m).length;
          experimental_less_than_m := experimental_group_weights.filter (λ x => x < m).length;
          experimental_geq_m := experimental_group_weights.filter (λ x => x >= m).length;
        m = 23.4 ∧
        control_less_than_m = 6 ∧ control_geq_m = 14 ∧
        experimental_less_than_m = 14 ∧ experimental_geq_m = 6 ) ∧
  (3) (let ad := control_less_than_m * experimental_geq_m;
          bc := control_geq_m * experimental_less_than_m;
          k_square := 40 * (ad - bc) ^ 2 / (20 * 20 * 20 * 20);
        k_square = 6.4 ∧ k_square > 3.841) := sorry

end proof_problem_l676_676554


namespace find_number_l676_676491

theorem find_number (N : ℚ) (h : (5 / 6) * N = (5 / 16) * N + 150) : N = 288 := by
  sorry

end find_number_l676_676491


namespace probability_not_hearing_favorite_song_l676_676571

noncomputable def num_ways_favor_not_heard_first_5_min : ℕ :=
  12! - (11! + 10!)

theorem probability_not_hearing_favorite_song :
  (num_ways_favor_not_heard_first_5_min : ℚ) / 12! = 10 / 11 := by
sorry

end probability_not_hearing_favorite_song_l676_676571


namespace polynomial_is_quadratic_l676_676928

noncomputable def required_polynomial (P : ℝ[X]) :=
  ∀ a b c : ℝ,
    P.eval (a + b - 2 * c) + P.eval (b + c - 2 * a) + P.eval (c + a - 2 * b) = 3 * P.eval (a - b) + 3 * P.eval (b - c) + 3 * P.eval (c - a)

theorem polynomial_is_quadratic (P : ℝ[X]) :
  required_polynomial P → ∃ a b : ℝ, P = Polynomial.C a * X^2 + Polynomial.C b * X :=
sorry

end polynomial_is_quadratic_l676_676928


namespace water_flow_speed_correct_l676_676407

noncomputable def V := 
  let a := 44 -- Speed of the boat from city A in km/h
  let b := (V ^ 2) -- Speed of the boat from city B in km/h, where V is the water flow speed in km/h
  let meeting_time_diff := 3 / 4 -- 60 minutes - 45 minutes, converted to hours
  let delay := 2 / 3 -- 40 minutes, converted to hours
  
  -- Equation setup based on meeting times and speeds
  have h : delay * (b - V) = meeting_time_diff * (a + V + b - V) := sorry

  -- Solving the equation
  have h := by
    -- Simplifying the expression
    simp at h
    sorry
  -- Solving 5V^2 - 8V - 132 = 0
  solve_by_elim
    sorry

-- Final proof of the correct value of V
theorem water_flow_speed_correct : V = 6 :=
by
  show V = 6
  sorry

end water_flow_speed_correct_l676_676407


namespace center_of_circle_correct_l676_676507

noncomputable def center_of_circle : ℝ × ℝ :=
  let (a, b) := (-2, 15/2) in
  if (0,1) ∈ set_of (λ p, (p.1 - a)^2 + (p.2 - b)^2 = (3 - a)^2 + (4 - b)^2 ∧
     (p.1 - 1)^2 = 4 * (p.1 - 3) + 4 ∧
     (b - 4) / (a - 3) = -1/4) ∧
     (b - 5/2)/(a - 3/2) = -1 then (a, b) else (0, 0)

theorem center_of_circle_correct : center_of_circle = (-2, 15 / 2) :=
  sorry

end center_of_circle_correct_l676_676507


namespace theater_seat_count_l676_676713

theorem theater_seat_count :
  ∃ n : ℕ, n < 60 ∧ n % 9 = 5 ∧ n % 6 = 3 ∧ n = 41 :=
sorry

end theater_seat_count_l676_676713


namespace average_of_N_between_fractions_l676_676838

theorem average_of_N_between_fractions :
  ((2 / 9 : ℚ) < (N : ℚ) / 72) ∧ ((N : ℚ) / 72 < (1 / 3 : ℚ)) → 
  (∃ N_set : set ℤ, 
    N_set = {n | (2 / 9 : ℚ) < (n : ℚ) / 72 ∧ (n : ℚ) / 72 < (1 / 3 : ℚ)} ∧ 
    ∃ avg : ℚ, 
      avg = (N_set.to_finset.sum id) / N_set.card ∧ 
      avg = 20) :=
sorry

end average_of_N_between_fractions_l676_676838


namespace min_S_l676_676313

-- Define the arithmetic sequence
def a (n : ℕ) (a1 d : ℤ) : ℤ :=
  a1 + (n - 1) * d

-- Define the sum of the first n terms
def S (n : ℕ) (a1 : ℤ) (d : ℤ) : ℤ :=
  (n * (a1 + a n a1 d)) / 2

-- Conditions
def a4 : ℤ := -15
def d : ℤ := 3

-- Found a1 from a4 and d
def a1 : ℤ := -24

-- Theorem stating the minimum value of the sum
theorem min_S : ∃ n, S n a1 d = -108 :=
  sorry

end min_S_l676_676313


namespace garden_area_l676_676874

theorem garden_area (radius_ground : ℝ) (width_garden : ℝ) (π : ℝ) : 
  radius_ground = 15 → 
  width_garden = 1.2 → 
  π = 3.14159 → 
  let radius_large := radius_ground + width_garden in
  let area_large := π * radius_large^2 in
  let area_small := π * radius_ground^2 in
  let area_garden := area_large - area_small in
  area_garden ≈ 117.68 :=
begin
  intros h1 h2 h3,
  let r_large := radius_ground + width_garden,
  let a_large := π * r_large ^ 2,
  let a_small := π * radius_ground ^ 2,
  let a_garden := a_large - a_small,
  have h_r_large: r_large = 16.2, by { rw [h1, h2], linarith },
  have h_a_large: a_large = π * 262.44, by { rw h_r_large, norm_num },
  have h_a_small: a_small = π * 225, by { rw h1, norm_num },
  have h_a_garden: a_garden = π * 37.44, by { rw [←sub_eq_add_neg, h_a_large, h_a_small], norm_num },
  have h_approx: a_garden = π * 37.44, from h_a_garden,
  have h_approx_val: 3.14159 * 37.44 ≈ 117.68, by { norm_num },
  rw h3 at h_approx_val,
  exact h_approx_val,
end

end garden_area_l676_676874


namespace find_the_number_l676_676857

theorem find_the_number :
  ∃ x : ℕ, (x + 720) / 125 = 7392 / 462 ∧ x = 1280 := 
  sorry

end find_the_number_l676_676857


namespace find_n_l676_676320

noncomputable def a_n (n : ℕ) : ℝ :=
  1 / (Real.sqrt n + Real.sqrt (n + 1))

noncomputable def S_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), a_n k

theorem find_n (n : ℕ) (h : S_n n = 9) : n = 99 := 
  sorry

end find_n_l676_676320


namespace plush_toys_high_quality_probability_l676_676539

theorem plush_toys_high_quality_probability :
  let frequencies := [0.950, 0.940, 0.910, 0.920, 0.924, 0.921, 0.919, 0.923] in
  abs ((frequencies.sum / frequencies.length) - 0.92) < 0.01 :=
by
  sorry

end plush_toys_high_quality_probability_l676_676539


namespace number_of_possible_integers_l676_676274

theorem number_of_possible_integers (x: ℤ) (h: ⌈real.sqrt ↑x⌉ = 20) : 39 :=
  sorry

end number_of_possible_integers_l676_676274


namespace solve_equation_l676_676438

theorem solve_equation : ∃ x : ℝ, (2 / x) = (1 / (x + 1)) ∧ x = -2 :=
by
  use -2
  split
  { -- Proving the equality part
    show (2 / -2) = (1 / (-2 + 1)),
    simp,
    norm_num
  }
  { -- Proving the equation part
    refl
  }

end solve_equation_l676_676438


namespace max_single_painted_faces_l676_676519

theorem max_single_painted_faces (n : ℕ) (hn : n = 64) :
  ∃ max_cubes : ℕ, max_cubes = 32 := 
sorry

end max_single_painted_faces_l676_676519


namespace harvey_final_number_l676_676594

-- Conditions involving multiple students counting with specific rules
def harvey_says (students : List String) (total_count : Nat) : Nat :=
  let rec skip_numbers (count_from : Nat) (skip_every : Nat) (total: Nat) : List Nat :=
    if count_from > total then [] else
    if skip_every = 4 then count_from :: skip_numbers (count_from + 1) 1 total
    else skip_numbers (count_from + 1) (skip_every + 1) total
  in 
  let rec helper (remaining_students : List String) (remaining_numbers : List Nat) (skip_every : Nat) : Nat :=
    match remaining_students, remaining_numbers with
    | [last_student], [last_number] => last_number
    | _::students, numbers => 
      let new_numbers := numbers.filter (λ x, (numbers.indexOf x) % 4 != 1)
      helper students new_numbers skip_every
    | _, _ => 0
  in
  helper students (List.range total_count |>.map (1+)) 4

-- Lean 4 statement to prove Harvey eventually says the number 365
theorem harvey_final_number : harvey_says ["Alfred", "Bonnie", "Clara", "Daphne", "Eli", "Franco", "Gina", "Harvey"] 1100 = 365 :=
by
  sorry

end harvey_final_number_l676_676594


namespace region_of_polynomial_roots_l676_676191

noncomputable def polynomial_roots_in_unit_circle 
  (a b : ℝ) : Prop :=
  let Δ := a^2 - 4 * b in
  let z₁ := (-a + Real.sqrt Δ) / 2 in
  let z₂ := (-a - Real.sqrt Δ) / 2 in
  |z₁| < 1 ∧ |z₂| < 1

theorem region_of_polynomial_roots 
  (a b : ℝ) : 
  polynomial_roots_in_unit_circle a b ↔ 
    ∃ (x₁ x₂ x₃ x₄ x₅ x₆ y₁ y₂ y₃ y₄ y₅ y₆ : ℝ), 
    (x₁, y₁) = (2, 1) ∧
    (x₂, y₂) = (-2, 1) ∧
    (x₃, y₃) = (0, -1) ∧ 
    a > x₄ ∧ a < x₁ ∧ b > y₃ ∧ b < y₁ ∧
    y₂ = y₄ ∧ y₄ = y₆ ∧ x₅ = 0 ∧ x₆ = 0 ∧
    b < 1 :=
sorry

end region_of_polynomial_roots_l676_676191


namespace incorrect_option_D_l676_676509

-- Definitions based on conditions
def cumulative_progress (days : ℕ) : ℕ :=
  30 * days

-- The Lean statement representing the mathematically equivalent proof problem
theorem incorrect_option_D : cumulative_progress 11 = 330 ∧ ¬ (cumulative_progress 10 = 330) :=
by {
  sorry
}

end incorrect_option_D_l676_676509


namespace smallest_lambda_inequality_l676_676179

theorem smallest_lambda_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x * y * (x^2 + y^2) + y * z * (y^2 + z^2) + z * x * (z^2 + x^2) ≤ (1 / 8) * (x + y + z)^4 :=
sorry

end smallest_lambda_inequality_l676_676179


namespace quadratic_coefficients_l676_676543

theorem quadratic_coefficients :
  ∀ x : ℝ, 3 * x^2 = 5 * x - 1 → (∃ a b c : ℝ, a = 3 ∧ b = -5 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x h
  use 3, -5, 1
  sorry

end quadratic_coefficients_l676_676543


namespace parabola_focus_coordinates_l676_676703

theorem parabola_focus_coordinates (h : ∀ y, y^2 = 4 * x) : ∃ x, x = 1 ∧ y = 0 := 
sorry

end parabola_focus_coordinates_l676_676703


namespace phase_shift_cos_l676_676178

theorem phase_shift_cos (b c : ℝ) (h_b : b = 2) (h_c : c = π / 2) :
  (-c / b) = -π / 4 :=
by
  rw [h_b, h_c]
  sorry

end phase_shift_cos_l676_676178


namespace circle_range_of_m_l676_676411

theorem circle_range_of_m (m : ℝ) : (-∞, 1/2) := sorry

end circle_range_of_m_l676_676411


namespace probability_two_diff_fruits_l676_676736

-- Defining the types of fruits
inductive Fruit
| apple
| orange
| banana
| pear

-- Defining a function that represents the probability of Joe eating the same fruit in all meals
def same_fruit_prob : ℚ := (1/4)^4

-- Defining the probability that Joe eats at least two different kinds of fruit
def prob_two_different_fruits : ℚ := 1 - 4 * same_fruit_prob

-- The proof statement that needs to be proven
theorem probability_two_diff_fruits (h: same_fruit_prob = 1/256) : prob_two_different_fruits = 63/64 := 
by 
  rw [same_fruit_prob, h]
  sorry

end probability_two_diff_fruits_l676_676736


namespace unique_real_solution_l676_676658

theorem unique_real_solution (m : ℝ) (h : m > 0) :
  (∃! x : ℝ, m * log x - (1/2) * x^2 + m * x = 0) ↔ m = 1/2 := by
  sorry

end unique_real_solution_l676_676658


namespace rank_of_matrix_A_is_2_l676_676610

def matrix_A : Matrix (Fin 4) (Fin 5) ℚ :=
  ![![3, -1, 1, 2, -8],
    ![7, -1, 2, 1, -12],
    ![11, -1, 3, 0, -16],
    ![10, -2, 3, 3, -20]]

theorem rank_of_matrix_A_is_2 : Matrix.rank matrix_A = 2 := by
  sorry

end rank_of_matrix_A_is_2_l676_676610


namespace exam_inequality_l676_676624

-- Given conditions
variable {f : ℝ → ℝ} (h : ∀ x : ℝ, f x + deriv f x < 0)

-- Prove the required inequality
theorem exam_inequality : exp 2 * f 2 > exp 3 * f 3 :=
sorry

end exam_inequality_l676_676624


namespace part1_part2_l676_676204

-- Define the given vectors
def vec_a : ℝ × ℝ × ℝ := (3, 2, -1)
def vec_b : ℝ × ℝ × ℝ := (2, 1, 2)

-- Define dot product for 3D vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define vector operations
def vec_add (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2, u.3 + v.3)

def vec_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def scalar_mul (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (k * v.1, k * v.2, k * v.3)

-- Part 1: Prove the dot product result
theorem part1 : dot_product (vec_add vec_a vec_b) (vec_sub vec_a (scalar_mul 2 vec_b)) = -10 := 
  by sorry

-- Part 2: Prove the values of k for orthogonality
theorem part2 (k : ℝ) : (dot_product (vec_add (scalar_mul k vec_a) vec_b) (vec_sub vec_a (scalar_mul k vec_b)) = 0) ↔ (k = 3 / 2 ∨ k = -2 / 3) :=
  by sorry

end part1_part2_l676_676204


namespace abs_diff_x_y_l676_676348

variables {x y : ℝ}

noncomputable def floor (z : ℝ) : ℤ := Int.floor z
noncomputable def fract (z : ℝ) : ℝ := z - floor z

theorem abs_diff_x_y 
  (h1 : floor x + fract y = 3.7) 
  (h2 : fract x + floor y = 4.6) : 
  |x - y| = 1.1 :=
by
  sorry

end abs_diff_x_y_l676_676348


namespace statement_I_statement_II_counterexample_statement_III_counterexample_l676_676617

def floor (x : ℝ) := int.floor x

theorem statement_I (x : ℝ) (hx : ∀ k : ℤ, x ≠ k) : floor (x + 0.5) = floor x + 1 := 
sorry

theorem statement_II_counterexample (x y : ℝ) (hxy : ∀ k : ℤ, x + y ≠ k) : floor (x + y - 0.5) ≠ floor x + floor y - 1 := 
sorry

theorem statement_III_counterexample (x : ℝ) : floor (1.5 * x) ≠ floor 1.5 * floor x := 
sorry

end statement_I_statement_II_counterexample_statement_III_counterexample_l676_676617


namespace shaded_cube_count_l676_676415

theorem shaded_cube_count (n : ℕ) (m : ℕ) : 
  n = 4 → m = 64 → 
  (∀ (x y : ℕ), (x = 4 ∧ y = 4) → ∃ (shaded : ℕ), shaded = 32) :=
begin
  intros hn hm hxy,
  rcases hxy with ⟨hx, hy⟩,
  sorry
end

end shaded_cube_count_l676_676415


namespace solve_equation_l676_676435

theorem solve_equation : ∃ x : ℝ, (2 / x) = (1 / (x + 1)) ∧ x = -2 :=
by
  use -2
  split
  { -- Proving the equality part
    show (2 / -2) = (1 / (-2 + 1)),
    simp,
    norm_num
  }
  { -- Proving the equation part
    refl
  }

end solve_equation_l676_676435


namespace units_digit_product_composites_l676_676469

theorem units_digit_product_composites :
  (4 * 6 * 8 * 9 * 10) % 10 = 0 :=
sorry

end units_digit_product_composites_l676_676469


namespace complex_power_six_l676_676572

theorem complex_power_six :
  (1 + complex.i)^6 = -8 * complex.i :=
by sorry

end complex_power_six_l676_676572


namespace square_area_from_diagonal_l676_676464

theorem square_area_from_diagonal
  (d : ℝ) (h : d = 10) : ∃ (A : ℝ), A = 50 :=
by {
  -- here goes the proof
  sorry
}

end square_area_from_diagonal_l676_676464


namespace domain_of_sqrt_function_l676_676010

noncomputable def domain_of_function : set ℝ :=
  { x : ℝ | x^2 - 5 * x + 6 ≥ 0}

theorem domain_of_sqrt_function :
  domain_of_function = { x : ℝ | x ∈ (set.Iic 2 ∪ set.Ici 3) } :=
by
  sorry

end domain_of_sqrt_function_l676_676010


namespace two_digit_number_property_l676_676456

noncomputable def sum_of_digits (n: ℕ) : ℕ :=
  let tens := n / 10 in
  let units := n % 10 in
  tens + units

noncomputable def sum_of_units_digits (n: ℕ) : ℕ :=
  (n * 3) % 10 + (n * 5) % 10 + (n * 7) % 10 + (n * 9) % 10

theorem two_digit_number_property :
  ∃ (n1 n2 n3: ℕ), 
  (9 < n1 ∧ n1 < 100 ∧
   9 < n2 ∧ n2 < 100 ∧
   9 < n3 ∧ n3 < 100 ∧
   n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ∧
   sum_of_units_digits n1 = sum_of_digits n1 ∧
   sum_of_units_digits n2 = sum_of_digits n2 ∧
   sum_of_units_digits n3 = sum_of_digits n3) ∧
  ¬(∃ n4, (9 < n4 ∧ n4 < 100) ∧ sum_of_units_digits n4 = sum_of_digits n4 ∧ n4 ≠ n1 ∧ n4 ≠ n2 ∧ n4 ≠ n3) :=
sorry

end two_digit_number_property_l676_676456


namespace minimum_Sn_l676_676353

noncomputable def sequence (n : ℕ) : ℕ → ℝ
| 0 => 2  -- Define accordingly
| (m + 1) => 2 * (m + 1)  -- Define accordingly

def excellence_value (n : ℕ) (a : ℕ → ℝ) : ℝ :=
(a 1 + (Finset.range (n - 1)).sum (λ k => 2^k * a (k + 2))) / n

def Sn (n : ℕ) (a : ℕ → ℝ) : ℝ :=
Finset.range n).sum (λ k => a (k + 1) - 20)

theorem minimum_Sn (n : ℕ) (a : ℕ → ℝ) :
  excellence_value n a = 2^(n + 1) →
  (∃ n, Sn n a = -72) :=
begin
  sorry
end

end minimum_Sn_l676_676353


namespace coyote_time_lemma_l676_676918

theorem coyote_time_lemma (coyote_speed darrel_speed : ℝ) (catch_up_time t : ℝ) 
  (h1 : coyote_speed = 15) (h2 : darrel_speed = 30) (h3 : catch_up_time = 1) (h4 : darrel_speed * catch_up_time = coyote_speed * t) :
  t = 2 :=
by
  sorry

end coyote_time_lemma_l676_676918


namespace greatest_x_lcm_max_x_value_l676_676805

theorem greatest_x_lcm (x : ℕ) : lcm x (lcm 12 18) = 180 → x ≤ 180 :=
by {
  sorry
}

theorem max_x_value (x : ℕ) : lcm x (lcm 12 18) = 180 → x = 180 :=
by {
  have h_lcm := greatest_x_lcm x,
  sorry
}

end greatest_x_lcm_max_x_value_l676_676805


namespace probability_at_least_40_cents_heads_l676_676001

theorem probability_at_least_40_cents_heads :
  let coins := {50, 25, 10, 5, 1}
  3 / 8 =
    (∑ H in {x ∈ (Finset.powerset coins) | (x.sum ≥ 40)}, (1 / 2) ^ x.card) :=
sorry

end probability_at_least_40_cents_heads_l676_676001


namespace prob_each_student_gets_each_snack_l676_676884

-- Define the total number of snacks and their types
def total_snacks := 16
def snack_types := 4

-- Define the conditions for the problem
def students := 4
def snacks_per_type := 4

-- Define the probability calculation.
-- We would typically use combinatorial functions here, but for simplicity, use predefined values from the solution.
def prob_student_1 := 64 / 455
def prob_student_2 := 9 / 55
def prob_student_3 := 8 / 35
def prob_student_4 := 1 -- Always 1 for the final student's remaining snacks

-- Calculate the total probability
def total_prob := prob_student_1 * prob_student_2 * prob_student_3 * prob_student_4

-- The statement to prove the desired probability outcome
theorem prob_each_student_gets_each_snack : total_prob = (64 / 1225) :=
by
  sorry

end prob_each_student_gets_each_snack_l676_676884


namespace parabola_points_count_l676_676177

theorem parabola_points_count :
  ∃ n : ℕ, n = 8 ∧ 
    (∀ x y : ℕ, (y = -((x^2 : ℤ) / 3) + 7 * (x : ℤ) + 54) → 1 ≤ x ∧ x ≤ 26 ∧ x % 3 = 0) :=
by
  sorry

end parabola_points_count_l676_676177


namespace sum_of_digits_of_steps_l676_676404

theorem sum_of_digits_of_steps 
  (n : ℕ) -- Total number of steps must be a natural number
  (hSylvs_jumps : ℕ → ℕ := λ n, (n + 2) / 3) -- Sylvester's jumps
  (hPenny_jumps : ℕ → ℕ := λ n, (n + 3) / 4) -- Penny's jumps
  (hCondition : hSylvs_jumps n + 11 = hPenny_jumps n) -- Given condition
  (hDigitSum : (1 + 3 + 2) = 6) : -- The sum of the digits of 132
  (1 + 3 + 2 = 6) := 
  by 
    have h1 : hSylvs_jumps 132 = 44 := by sorry
    have h2 : hPenny_jumps 132 = 33 := by sorry
    have h3 : hCondition 132 := by sorry
    exact hDigitSum

end sum_of_digits_of_steps_l676_676404


namespace cos_seven_theta_l676_676681

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end cos_seven_theta_l676_676681


namespace num_distinct_factors_of_36_l676_676662

/-- Definition of the number 36. -/
def n : ℕ := 36

/-- Prime factorization of 36 is 2^2 * 3^2. -/
def prime_factors : List (ℕ × ℕ) := [(2, 2), (3, 2)]

/-- The number of distinct positive factors of 36 is 9. -/
theorem num_distinct_factors_of_36 : ∃ k : ℕ, k = 9 ∧ 
  ∀ d : ℕ, d ∣ n → d > 0 → List.mem d (List.range (n + 1)) :=
by
  sorry

end num_distinct_factors_of_36_l676_676662


namespace beads_to_remove_l676_676773

-- Definitions for the conditions given in the problem
def initial_blue_beads : Nat := 49
def initial_red_bead : Nat := 1
def total_initial_beads : Nat := initial_blue_beads + initial_red_bead
def target_blue_percentage : Nat := 90 -- percentage

-- The goal to prove
theorem beads_to_remove (initial_blue_beads : Nat) (initial_red_bead : Nat)
    (target_blue_percentage : Nat) : Nat :=
    let target_total_beads := (initial_red_bead * 100) / target_blue_percentage
    total_initial_beads - target_total_beads
-- Expected: beads_to_remove 49 1 90 = 40

example : beads_to_remove initial_blue_beads initial_red_bead target_blue_percentage = 40 := by 
    sorry

end beads_to_remove_l676_676773


namespace necessary_but_not_sufficient_condition_l676_676356

noncomputable theory

def geometric_sequence (a b c d : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c)

theorem necessary_but_not_sufficient_condition (a b c d : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) :
  (geometric_sequence a b c d → a * d = b * c) ∧ (¬ (a * d = b * c → geometric_sequence a b c d)) :=
sorry

end necessary_but_not_sufficient_condition_l676_676356


namespace log_evaluation_l676_676169

theorem log_evaluation : log 60 / log 10 + log 80 / log 10 - log 15 / log 10 = 2.505 :=
by
  -- The proof goes here.
  sorry

end log_evaluation_l676_676169


namespace cos_theta_seven_l676_676686

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end cos_theta_seven_l676_676686


namespace num_possible_x_l676_676268

theorem num_possible_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 20) : {y : ℕ | 361 < y ∧ y ≤ 400}.card = 39 :=
by
  sorry

end num_possible_x_l676_676268


namespace distance_between_points_l676_676942

theorem distance_between_points :
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  (Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 * Real.sqrt 2) :=
by
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  sorry

end distance_between_points_l676_676942


namespace minimum_surface_area_l676_676536

noncomputable def surface_area (r : ℝ) : ℝ :=
  π * (2 * r^4) / (r^2 - 1)

theorem minimum_surface_area : ∃ r : ℝ, r > 0 ∧ surface_area r = 8 * π :=
by
  use sqrt 2
  split
  · exact real.sqrt_pos.mpr (show (0 : ℝ) < 2 from by norm_num)
  · have r_squared : sqrt 2 ^ 2 = 2, from by norm_num
    simp [surface_area, pow_succ', r_squared]
    field_simp
    norm_num
    sorry

end minimum_surface_area_l676_676536


namespace solve_equation_l676_676422

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end solve_equation_l676_676422


namespace triangle_ABC_area_l676_676307

open Real EuclideanGeometry

variable (A B C D : Point)
variable (h : coplanar {A, B, C, D})
variable (h₁ : ∠ B = 90)
variable (h₂ : dist A C = 10)
variable (h₃ : dist B C = 6)
variable (h₄ : dist B D = 8)
variable (h₅ : angle B D = 90)

theorem triangle_ABC_area : area_triangle A B C = 18 :=
by sorry


end triangle_ABC_area_l676_676307


namespace find_lambda_l676_676209

variables {R : Type*} [field R]

def A (λ : R) : matrix (fin 3) (fin 3) R :=
  ![
    ![1, 2, -2],
    ![2, -1, λ],
    ![3, 1, -1]
  ]

theorem find_lambda (λ : R) (h : det (A λ) = 0) : λ = 7 / 5 :=
by
  sorry

end find_lambda_l676_676209


namespace opposite_teal_face_is_blue_l676_676511

noncomputable def cube_faces : Type :=
  { top   : String,
    front : String,
    right : String,
    left  : String,
    back  : String,
    bottom: String 
    -- faces with unique color terms
  }

def views (v : cube_faces) : Prop :=
  (v.top = "yellow") ∧ (v.front = "blue") ∧ (v.right = "orange")
  ∨ (v.top = "yellow") ∧ (v.front = "black") ∧ (v.right = "orange")
  ∨ (v.top = "yellow") ∧ (v.front = "violet") ∧ (v.right = "orange")

theorem opposite_teal_face_is_blue (v : cube_faces) (h : views v) : 
  v.back = "teal" -> v.front = "blue" := 
sorry

end opposite_teal_face_is_blue_l676_676511


namespace intersection_eq_l676_676376

noncomputable def setM : Set ℝ := {x : ℝ | 2^(x - 1) < 1}
noncomputable def setN : Set ℝ := {x : ℝ | Real.log x < 1}

theorem intersection_eq : (setM ∩ setN) = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l676_676376


namespace sum_of_ages_now_l676_676844

variable (D A Al B : ℝ)

noncomputable def age_condition (D : ℝ) : Prop :=
  D = 16

noncomputable def alex_age_condition (A : ℝ) : Prop :=
  A = 60 - (30 - 16)

noncomputable def allison_age_condition (Al : ℝ) : Prop :=
  Al = 15 - (30 - 16)

noncomputable def bernard_age_condition (B A Al : ℝ) : Prop :=
  B = (A + Al) / 2

noncomputable def sum_of_ages (A Al B : ℝ) : ℝ :=
  A + Al + B

theorem sum_of_ages_now :
  age_condition D →
  alex_age_condition A →
  allison_age_condition Al →
  bernard_age_condition B A Al →
  sum_of_ages A Al B = 70.5 := by
  sorry

end sum_of_ages_now_l676_676844


namespace average_transformed_is_2_l676_676702

-- Define the list of elements and their average
variables (k : Fin 8 → ℝ)
variable (h : (∑ i in Finset.univ, k i) / 8 = 4)

-- Define the transformation
def transform (k : Fin 8 → ℝ) : Fin 8 → ℝ := λ i, 2 * (k i - 3)

-- Assert that the average of the transformed list is 2
theorem average_transformed_is_2 (k : Fin 8 → ℝ) (h : (∑ i in Finset.univ, k i) / 8 = 4) :
  (∑ i in Finset.univ, transform k i) / 8 = 2 := 
sorry

end average_transformed_is_2_l676_676702


namespace width_of_margin_l676_676893

-- Given conditions as definitions
def total_area : ℝ := 20 * 30
def percentage_used : ℝ := 0.64
def used_area : ℝ := percentage_used * total_area

-- Definition of the width of the typing area
def width_after_margin (x : ℝ) : ℝ := 20 - 2 * x

-- Definition of the length after top and bottom margins
def length_after_margin : ℝ := 30 - 6

-- Calculate the area used considering the margins
def typing_area (x : ℝ) : ℝ := (width_after_margin x) * length_after_margin

-- Statement to prove
theorem width_of_margin : ∃ x : ℝ, typing_area x = used_area ∧ x = 2 := by
  -- We give the prompt to eventually prove the theorem with the correct value
  sorry

end width_of_margin_l676_676893


namespace triangle_side_lengths_l676_676584

theorem triangle_side_lengths:
  {x : ℤ} → (3 < x ∧ x < 11) → (x % 2 = 1) → 
  ({x | (3 < x) ∧ (x < 11) ∧ (x % 2 = 1)}.card = 3) :=
by
  sorry

end triangle_side_lengths_l676_676584


namespace cartons_packed_l676_676455

/-
  There are 768 cups, with every 12 cups packed into one box, and every 8 boxes packed into one carton.
  Prove that the number of cartons that can be packed is 8.
-/

theorem cartons_packed : 
  let cups := 768
  let cups_per_box := 12
  let boxes_per_carton := 8
  (768 ÷ (12 * 8)) = 8 :=
by
  let cups := 768
  let cups_per_box := 12
  let boxes_per_carton := 8
  have h1 : cups ÷ (cups_per_box * boxes_per_carton) = cups ÷ 96 := by
    rw [cups_per_box, boxes_per_carton]
  have h2 : 768 ÷ 96 = 8 := by
    sorry -- here you fill in the proof that 768 ÷ 96 = 8
  exact h2

end cartons_packed_l676_676455


namespace sum_lent_eq_1100_l676_676854

def interest_rate : ℚ := 6 / 100

def period : ℕ := 8

def interest_amount (P : ℚ) : ℚ :=
  period * interest_rate * P

def total_interest_eq_principal_minus_572 (P: ℚ) : Prop :=
  interest_amount P = P - 572

theorem sum_lent_eq_1100 : ∃ P : ℚ, total_interest_eq_principal_minus_572 P ∧ P = 1100 :=
by
  use 1100
  sorry

end sum_lent_eq_1100_l676_676854


namespace distance_between_points_l676_676951

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 3 3 (-2) (-2) = 5 * real.sqrt 2 :=
by
  sorry

end distance_between_points_l676_676951


namespace algebraic_identity_l676_676205

theorem algebraic_identity (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 2001 = -2000 :=
by
  sorry

end algebraic_identity_l676_676205


namespace cos_double_angle_l676_676228

theorem cos_double_angle (α : ℝ) (h1 : sin (α + π / 2) = - sqrt 5 / 5) (h2 : 0 < α ∧ α < π) : 
  cos (2 * α) = -3 / 5 :=
sorry

end cos_double_angle_l676_676228


namespace find_n_l676_676964

theorem find_n (a b c : ℤ) (m n p : ℕ)
  (h1 : a = 3)
  (h2 : b = -7)
  (h3 : c = -6)
  (h4 : m > 0)
  (h5 : n > 0)
  (h6 : p > 0)
  (h7 : Nat.gcd m p = 1)
  (h8 : Nat.gcd m n = 1)
  (h9 : Nat.gcd n p = 1)
  (h10 : ∃ x1 x2 : ℤ, x1 = (m + Int.sqrt n) / p ∧ x2 = (m - Int.sqrt n) / p)
  : n = 121 :=
sorry

end find_n_l676_676964


namespace complex_power_l676_676575

theorem complex_power : (1 + complex.i)^6 = -8 * complex.i :=
by
  sorry

end complex_power_l676_676575


namespace ratio_of_lemons_l676_676760

theorem ratio_of_lemons :
  ∃ (L J E I : ℕ), 
  L = 5 ∧ 
  J = L + 6 ∧ 
  J = E / 3 ∧ 
  E = I / 2 ∧ 
  L + J + E + I = 115 ∧ 
  J / E = 1 / 3 :=
by
  sorry

end ratio_of_lemons_l676_676760


namespace part_I_part_II_l676_676208

-- Define the function f
def f (x a : ℝ) := |x - a| + |x - 2|

-- Statement for part (I)
theorem part_I (a : ℝ) (h : ∃ x : ℝ, f x a ≤ a) : a ≥ 1 := sorry

-- Statement for part (II)
theorem part_II (m n p : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : p > 0) (h4 : m + 2 * n + 3 * p = 1) : 
  (3 / m) + (2 / n) + (1 / p) ≥ 6 + 2 * Real.sqrt 6 + 2 * Real.sqrt 2 := sorry

end part_I_part_II_l676_676208


namespace zero_integers_satisfy_conditions_l676_676961

noncomputable def satisfies_conditions (n : ℤ) : Prop :=
  ∃ k : ℤ, n * (25 - n) = k^2 * (25 - n)^2 ∧ n % 3 = 0

theorem zero_integers_satisfy_conditions :
  (∃ n : ℤ, satisfies_conditions n) → False := by
  sorry

end zero_integers_satisfy_conditions_l676_676961


namespace sum_of_roots_l676_676181

theorem sum_of_roots : 
  (∀ (x : ℝ), (3 * x^3 + 2 * x^2 - 6 * x + 15 = 0) ∨ (4 * x^3 - 16 * x^2 + 12 = 0)) →
  (∑ r in {r : ℝ | (3 * r^3 + 2 * r^2 - 6 * r + 15 = 0) ∨ (4 * r^3 - 16 * r^2 + 12 = 0)}, r) = 10 / 3  :=
by
  sorry

end sum_of_roots_l676_676181


namespace measure_B44_B45_B43_l676_676696

-- Definitions based on the problem conditions
def isosceles_right_triangle (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] 
                             (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) : Prop :=
angle_A = 45 ∧ angle_B = 45 ∧ angle_C = 90

def B_next (B_n B_n1 B_rec: ℕ → ℕ) (n: ℕ): ℕ :=
B_rec(n + 3) = (B_rec(n) + B_rec(n + 1)) / 2

-- Prove statement
theorem measure_B44_B45_B43 (B : ℕ → ℕ) (triangle_B1_B2_B3_isosceles : isosceles_right_triangle B 1 2)
                            (B_is_midpoint : ∀ n, B_next B n) :
  ∃ angle : ℝ, angle = 45 ∧ angle = ∡ B 44 45 43 := sorry

end measure_B44_B45_B43_l676_676696


namespace basketball_game_probability_l676_676295

def probability_at_least_seven_stayed (prob_unsure : ℚ) (unsure_total sure_total : ℕ) : ℚ :=
  let cases_seven := (Nat.choose unsure_total 3) * (prob_unsure ^ 3) * ((1 - prob_unsure) ^ (unsure_total - 3)) in
  let cases_eight := prob_unsure ^ unsure_total in
  cases_seven + cases_eight

theorem basketball_game_probability
  (unsure_prob : ℚ := 1/3)
  (unsure_total : ℕ := 4)
  (sure_total : ℕ := 4) -- Summarizing the conditions given
  : probability_at_least_seven_stayed unsure_prob unsure_total sure_total = 1/9 := 
by
  sorry

end basketball_game_probability_l676_676295


namespace baseball_distribution_remainder_l676_676059

theorem baseball_distribution_remainder :
  let total_baseballs := 43
  let number_of_classes := 6
  (total_baseballs % number_of_classes) = 1 :=
by
  let total_baseballs := 43
  let number_of_classes := 6
  have h1 : total_baseballs % number_of_classes = 43 % 6 := rfl
  have h2 : 43 % 6 = 1 := by norm_num
  exact eq.trans h1 h2

end baseball_distribution_remainder_l676_676059


namespace sin_alpha_plus_beta_l676_676645

theorem sin_alpha_plus_beta {α β : ℝ} {a b c : ℝ} (hα : a * sin α + b * cos α + c = 0) (hβ : a * sin β + b * cos β + c = 0) (distinct : α ≠ β) (α_in : 0 ≤ α ∧ α ≤ π) (β_in : 0 ≤ β ∧ β ≤ π) : 
  sin (α + β) = (2 * a * b) / (a^2 + b^2) :=
sorry

end sin_alpha_plus_beta_l676_676645


namespace correct_option_c_l676_676068

theorem correct_option_c :
  (\forall a b c : ℝ, a * b ≠ c -> 2 * (sqrt 3) * 3 * (sqrt 3) ≠ 6 * (sqrt 3))
  ∧ (\forall d e : ℝ, d ≠ e -> real.cbrt (-64) ≠ 4)
  ∧ (\forall f g : ℝ, f ≠ g -> ±(sqrt 4) ≠ 2)
  → (real.cbrt (-8))^2 = 4 :=
begin
  sorry
end

end correct_option_c_l676_676068


namespace sin_alpha_minus_pi_over_3_cos_double_alpha_l676_676222

noncomputable def alpha : ℝ := sorry  -- Since we don't know the actual value, we declare it noncomputable

axiom cos_alpha_value : cos alpha = -4 / 5
axiom alpha_interval : (π / 2) < alpha ∧ alpha < π

theorem sin_alpha_minus_pi_over_3 : sin (alpha - π / 3) = (3 + 4 * Real.sqrt 3) / 10 :=
by
  sorry

theorem cos_double_alpha : cos (2 * alpha) = 7 / 25 :=
by
  sorry

end sin_alpha_minus_pi_over_3_cos_double_alpha_l676_676222


namespace complex_solution_l676_676644

theorem complex_solution (z : Complex) (h : z^2 + 2 * Complex.i * z + 3 = 0) :
  z = Complex.i ∨ z = -3 * Complex.i :=
sorry

end complex_solution_l676_676644


namespace count_integers_satisfying_sqrt_condition_l676_676290

theorem count_integers_satisfying_sqrt_condition :
  ∀ x : ℕ, (⌈(Real.sqrt x)⌉ = 20) → (∃ n : ℕ, n = 39) :=
by {
  intro x,
  intro sqrt_x_ceil_eq_twenty,
  sorry
}

end count_integers_satisfying_sqrt_condition_l676_676290


namespace find_cos_7theta_l676_676678

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end find_cos_7theta_l676_676678


namespace solve_for_b_perp_lines_l676_676804

noncomputable def perp_slope_b (b : ℝ) : Prop :=
  let slope1 := - (2 / 3 : ℝ)
  let slope2 := - (b / 4 : ℝ)
  slope1 * slope2 = -1

theorem solve_for_b_perp_lines : perp_slope_b (-6) :=
by
  -- Definition of slopes
  let slope1 := - (2 / 3 : ℝ)
  let slope2 := - (-6 / 4 : ℝ)
  -- Check if the product of slopes is -1
  have slope_prod : slope1 * slope2 = (2 / 3) * (6 / 4) := by
    simp [slope1, slope2]
  have slope_prod_eq : slope1 * slope2 = 1 / 2 * 1 / 2 := by
    rw slope_prod
  calc
    (2 / 3) * (6 / 4) = (-1 : ℝ) := by simp [slope1 * slope2, slope_prod_eq]
  sorry

end solve_for_b_perp_lines_l676_676804


namespace pump_filling_time_without_leak_l676_676889

theorem pump_filling_time_without_leak (P : ℝ) (h1 : 1 / P - 1 / 14 = 3 / 7) : P = 2 :=
sorry

end pump_filling_time_without_leak_l676_676889


namespace value_of_f_l676_676236

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = f(x)

noncomputable def not_identically_zero (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f(x) ≠ 0

noncomputable def satisfies_functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x * f(x + 1) = (1 + x) * f(x)

theorem value_of_f (f : ℝ → ℝ) (h1 : even_function f) (h2 : not_identically_zero f) (h3 : satisfies_functional_equation f) :
  ∀ x : ℝ, f(x) = 0 :=
sorry

end value_of_f_l676_676236


namespace sum_possible_y_l676_676830

theorem sum_possible_y (y : ℝ) (h1 : ∃ y, (∀ (A B C : ℝ), A = 40 ∧ B = 40 → A + B + y = 180) ∨ (∀ (A B C : ℝ), A = 40 ∧ B = y → A + A + y = 180))
: ∑ y = 140 :=
by
  sorry

end sum_possible_y_l676_676830


namespace distance_between_points_l676_676956

noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem distance_between_points :
  dist (3, 3) (-2, -2) = 5 * Real.sqrt 2 := 
by
  sorry

end distance_between_points_l676_676956


namespace min_value_l676_676202

-- Given points A, B, and C and their specific coordinates
def A : (ℝ × ℝ) := (1, 3)
def B (a : ℝ) : (ℝ × ℝ) := (a, 1)
def C (b : ℝ) : (ℝ × ℝ) := (-b, 0)

-- Conditions
axiom a_pos (a : ℝ) : a > 0
axiom b_pos (b : ℝ) : b > 0
axiom collinear (a b : ℝ) : 3 * a + 2 * b = 1

-- The theorem to prove
theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hcollinear : 3 * a + 2 * b = 1) : 
  ∃ z, z = 11 + 6 * Real.sqrt 2 ∧ ∀ (x y : ℝ), (x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 1) -> (3 / x + 1 / y) ≥ z :=
by sorry -- Proof to be provided

end min_value_l676_676202


namespace minimum_moves_required_l676_676905

theorem minimum_moves_required (initial_marbles : ℕ) (boxes : ℕ) (min_moves : ℕ) :
  initial_marbles = 2017 →
  boxes = 1000 →
  min_moves = 176 →
  ∀ n, n ≥ min_moves → 
  (∀ i j : ℕ, i ≠ j → n > 0 → (∀ m, m < boxes → (initial_marbles * boxes - ((n * initial_marbles - (n * (n - 1)) / 2))) - m > 0)) → 
  (∃ (k : ℕ), k < boxes → initial_marbles - (k - n) ≠ initial_marbles - (k - n + 1)) :=
begin 
  sorry 
end

end minimum_moves_required_l676_676905


namespace largest_rational_under_one_fourth_with_rank_three_l676_676811

theorem largest_rational_under_one_fourth_with_rank_three :
  (∃ a1 a2 a3 : ℕ, a1 = 5 ∧ a2 = 21 ∧ a3 = 421 ∧ 
                    ∃ q : ℚ, q < 1/4 ∧ 
                            q = 1/a1 + 1/a2 + 1/a3 ∧
                            (∀ k : ℕ, (k < 3 → ∀ (b : ℕ → ℕ), q ≠ 1/b 1 + 1/b 2 +...+ 1/b k))) :=
by introv, sorry

end largest_rational_under_one_fourth_with_rank_three_l676_676811


namespace minimum_cost_for_13_bottles_l676_676504

def cost_per_bottle_shop_A := 200 -- in cents
def discount_shop_B := 15 / 100 -- discount
def promotion_B_threshold := 4
def promotion_A_threshold := 4

-- Function to calculate the cost in Shop A for given number of bottles
def shop_A_cost (bottles : ℕ) : ℕ :=
  let batches := bottles / 5
  let remainder := bottles % 5
  (batches * 4 + remainder) * cost_per_bottle_shop_A

-- Function to calculate the cost in Shop B for given number of bottles
def shop_B_cost (bottles : ℕ) : ℕ :=
  if bottles >= promotion_B_threshold then
    (bottles * cost_per_bottle_shop_A) * (1 - discount_shop_B)
  else
    bottles * cost_per_bottle_shop_A

-- Function to calculate combined cost for given numbers of bottles from Shops A and B
def combined_cost (bottles_A bottles_B : ℕ) : ℕ :=
  shop_A_cost bottles_A + shop_B_cost bottles_B

theorem minimum_cost_for_13_bottles : ∃ a b, a + b = 13 ∧ combined_cost a b = 2000 := 
sorry

end minimum_cost_for_13_bottles_l676_676504


namespace integer_values_x_l676_676284

theorem integer_values_x {x : ℝ} (h : ⌈real.sqrt x⌉ = 20) : ∃ n : ℕ, n = 39 :=
by
  have h1 : 19 < real.sqrt x ∧ real.sqrt x ≤ 20 := sorry
  have h2 : 361 < x ∧ x ≤ 400 := sorry
  have x_values : set.Icc 362 400 := sorry
  have n_values : ∃ n : ℕ, n = set.finite.to_finset x_values.card := sorry
  exact n_values

end integer_values_x_l676_676284


namespace maximize_profit_l676_676235

def profit (x : ℝ) : ℝ := -(1/3)*x^3 + 81*x - 234

theorem maximize_profit : ∃ x : ℝ, x = 9 ∧ ∀ y : ℝ, profit y ≤ profit x := 
by
  use 9
  sorry

end maximize_profit_l676_676235


namespace Chloe_final_points_l676_676913

-- Define the points scored (or lost) in each round
def round1_points : ℤ := 40
def round2_points : ℤ := 50
def round3_points : ℤ := 60
def round4_points : ℤ := 70
def round5_points : ℤ := -4
def round6_points : ℤ := 80
def round7_points : ℤ := -6

-- Statement to prove: Chloe's total points at the end of the game
theorem Chloe_final_points : 
  round1_points + round2_points + round3_points + round4_points + round5_points + round6_points + round7_points = 290 :=
by
  sorry

end Chloe_final_points_l676_676913


namespace purple_sum_expr_purple_sum_109_final_solution_l676_676524

def isPurple (n : ℚ) : Prop :=
  ∃ a b : ℕ+, n = (1 : ℚ) / (2^a * 5^b) ∧ a > b

theorem purple_sum_expr (S : ℚ) (a b : ℕ) (hrelprime : Nat.coprime a b) :
  S = ∑' n : ℕ+, (if isPurple (1/ (2^n * 5^(n:ℕ+))) then (1/ (2^n * 5^(n:ℕ+))) else 0) →
  S = 1 / 9 :=
sorry

theorem purple_sum_109 :
  100 * 1 + 9 = (100 * 1 + 9 : ℕ) :=
by
  rfl

theorem final_solution :
  ∃ a b : ℕ, Nat.coprime a b ∧ 100 * a + b = 109 ∧
  ∑' n : ℕ+, (if isPurple (1/ (2^n * 5^(n:ℕ+))) then (1/ (2^n * 5^(n:ℕ+))) else 0) = 1 / 9 :=
exists.intro 1 (exists.intro 9 ⟨Nat.coprime_one_right _, ⟨eq.refl 109, (purple_sum_expr (1 / 9) 1 9 Nat.coprime_one_right).2⟩⟩)

end purple_sum_expr_purple_sum_109_final_solution_l676_676524


namespace real_when_specific_pairs_l676_676371

open Complex

noncomputable def rational_pairs :=
  [(1, 1), (1, 3), (3, 1) : ℕ × ℕ]

theorem real_when_specific_pairs (a b n : ℕ) (h : Nat.gcd a b = 1) :
  (∃ k : ℤ, (sqrt a + i * sqrt b) ^ n = k) →
  (a, b) ∈ rational_pairs :=
by
  sorry

end real_when_specific_pairs_l676_676371


namespace W_k_two_lower_bound_l676_676400

-- Define W(k, 2)
def W (k : ℕ) (c : ℕ) : ℕ := -- smallest number such that for every n >= W(k, 2), 
  -- any 2-coloring of the set {1, 2, ..., n} contains a monochromatic arithmetic progression of length k
  sorry 

-- Define the statement to prove
theorem W_k_two_lower_bound (k : ℕ) : ∃ C > 0, W k 2 ≥ C * 2^(k / 2) :=
by
  sorry

end W_k_two_lower_bound_l676_676400


namespace total_caffeine_consumed_l676_676338

def caffeine_per_ounce (caffeine : ℝ) (ounces : ℝ) : ℝ :=
  caffeine / ounces

def caffeine_second_drink (caffeine_first_per_ounce : ℝ) : ℝ :=
  3 * caffeine_first_per_ounce * 2

def total_caffeine_drinks (caffeine_first : ℝ) (caffeine_second : ℝ) : ℝ :=
  caffeine_first + caffeine_second

theorem total_caffeine_consumed (caffeine_first : ℝ) (ounces_first : ℝ) (factor : ℝ) (ounces_second : ℝ) : 
  total_caffeine_drinks caffeine_first ((factor * caffeine_per_ounce caffeine_first ounces_first) * ounces_second) * 2 = 750 :=
by
  let caffeine_first := 250
  let ounces_first := 12
  let factor := 3
  let ounces_second := 2
  let caffeine_first_per_ounce := caffeine_per_ounce caffeine_first ounces_first
  let caffeine_second := caffeine_second_drink caffeine_first_per_ounce
  let caffeine_pill := total_caffeine_drinks caffeine_first caffeine_second
  have total_caffeine := caffeine_pill + total_caffeine_drinks caffeine_first caffeine_second
  show total_caffeine = 750
  sorry

end total_caffeine_consumed_l676_676338


namespace collinear_if_concylic_l676_676133

-- The construction: Given an acute triangle ABC, AB > AC, 
-- with O as the circumcenter, I as the incenter,
-- points D, X, Y, L as described, and line PQ through I touching circumcircle.

noncomputable theory

variables (A B C D L P Q X Y : Type) 

def collinear (A B C : Type) : Prop := 
  ∃ (l : Set Type), A ∈ l ∧ B ∈ l ∧ C ∈ l

def concyclic (P Q X Y : Type) : Prop := 
  ∃ (c : Set Type), P ∈ c ∧ Q ∈ c ∧ X ∈ c ∧ Y ∈ c

axiom acute_triangle (ABC : Type) (A B C: Type) : Prop
axiom circumcircle (ABC : Type) (O: Type) : Prop
axiom incircle (ABC : Type) (I: Type) : Prop
axiom tangents_intersection (B C O L : Type) : Prop
axiom altitude_from_A (ABC XYZ: Type) : Prop
axiom line_through_I_touching_circumcircle (PQ I O: Type) : Prop
axiom line_intersects_side (A O X BC : Type) : Prop
axiom side (D BC: Type) : Prop
axiom equality_condition (A L O I PX Y D Q B C X: Type) : Prop 

theorem collinear_if_concylic :
  (acute_triangle ABC A B C) ∧ (circumcircle ABC O) ∧ (incircle ABC I) ∧
  (side D BC) ∧ (altitude_from_A ABC AY) ∧ (tangents_intersection B C O L) ∧
  (line_intersects_side A O X BC) ∧ (line_through_I_touching_circumcircle PQ I O) ∧
  (collinear A D L ↔ concyclic P X Y Q) :=
begin
  sorry
end

end collinear_if_concylic_l676_676133


namespace fraction_bad_teams_leq_l676_676317

variable (teams total_teams : ℕ) (b : ℝ)

-- Given conditions
variable (cond₁ : total_teams = 18)
variable (cond₂ : teams = total_teams / 2)
variable (cond₃ : ∀ (rb_teams : ℕ), rb_teams ≠ 10 → rb_teams ≤ teams)

theorem fraction_bad_teams_leq (H : 18 * b ≤ teams) : b ≤ 1 / 2 :=
sorry

end fraction_bad_teams_leq_l676_676317


namespace no_solution_sin_eq_l676_676192

theorem no_solution_sin_eq (t : ℝ) (h : 0 ≤ t ∧ t ≤ π) :
  (∀ x : ℝ, ¬ (sin (x + t) = 1 - sin x)) ↔ (2 * π / 3 < t ∧ t ≤ π) :=
by
  sorry

end no_solution_sin_eq_l676_676192


namespace weight_difference_l676_676795

variables (W_A W_B W_C W_D W_E : ℝ)

def condition1 : Prop := (W_A + W_B + W_C) / 3 = 84
def condition2 : Prop := (W_A + W_B + W_C + W_D) / 4 = 80
def condition3 : Prop := (W_B + W_C + W_D + W_E) / 4 = 79
def condition4 : Prop := W_A = 80

theorem weight_difference (h1 : condition1 W_A W_B W_C) 
                          (h2 : condition2 W_A W_B W_C W_D) 
                          (h3 : condition3 W_B W_C W_D W_E) 
                          (h4 : condition4 W_A) : 
                          W_E - W_D = 8 :=
by
  sorry

end weight_difference_l676_676795


namespace mass_percentage_I_l676_676175

noncomputable def molarMassCa : ℝ := 40.08
noncomputable def molarMassI : ℝ := 126.90
noncomputable def molarMassCaI2 : ℝ := molarMassCa + 2 * molarMassI

theorem mass_percentage_I (h1 : molarMassCa = 40.08) 
                          (h2 : molarMassI = 126.90) 
                          (h3 : molarMassCaI2 = molarMassCa + 2 * molarMassI) :
   (2 * molarMassI / molarMassCaI2) * 100 ≈ 86.35 := 
by
  sorry

end mass_percentage_I_l676_676175


namespace penny_distinct_species_l676_676167

theorem penny_distinct_species :
  let sharks := 35
  let eels := 15
  let whales := 5
  let dolphins := 12
  let rays := 8
  let octopuses := 25
  let init_species := sharks + eels + whales + dolphins + rays + octopuses
  let unique_species := 6
  let categorized_pairs := 3
  init_species - categorized_pairs = 97 :=
by
  let sharks := 35
  let eels := 15
  let whales := 5
  let dolphins := 12
  let rays := 8
  let octopuses := 25
  let init_species := sharks + eels + whales + dolphins + rays + octopuses
  let unique_species := 6
  let categorized_pairs := 3
  have h1 : init_species = 35 + 15 + 5 + 12 + 8 + 25 := rfl
  have h2 : init_species = 100 := by rw h1
  have h3 : init_species - categorized_pairs = 100 - 3 := by rw h2
  have h4 : 100 - 3 = 97 := rfl
  show 100 - 3 = 97 from h4
  show init_species - categorized_pairs = 97 from h3
  sorry

end penny_distinct_species_l676_676167


namespace distance_between_points_l676_676945

theorem distance_between_points :
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  (Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 * Real.sqrt 2) :=
by
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  sorry

end distance_between_points_l676_676945


namespace correct_statements_l676_676350

variables {α : Type*} [AddCommGroup α] [VectorSpace ℝ α] (O A B C : α)
variables (P : α) (λ : ℝ)

-- Condition
def condition_2 := (P - O) = A + λ * ((B - A) / ∥B - A∥ + (C - A) / ∥C - A∥)
def condition_3 := (P - O) = A + λ * ((B - A) / (∥B - A∥ * Real.sin (angle B C A)) + (C - A) / (∥C - A∥ * Real.sin (angle C A B)))
def condition_4 := (P - O) = A + λ * ((B - A) / (∥B - A∥ * Real.cos (angle B C A)) + (C - A) / (∥C - A∥ * Real.cos (angle C A B)))
def condition_5 := (P - O) = (B + C) / 2 + λ * ((B - A) / (∥B - A∥ * Real.cos (angle B C A)) + (C - A) / (∥C - A∥ * Real.cos (angle C A B)))

-- Results
theorem correct_statements : 
  (condition_2 O A B C P λ →
   condition_3 O A B C P λ →
   condition_4 O A B C P λ →
   condition_5 O A B C P λ →
   true) :=
by sorry

end correct_statements_l676_676350


namespace number_of_parallelogram_conditions_l676_676718

variables (A B C D O : Type) 

-- Definitions of the conditions
def condition1 (AB CD AD BC : Type) :=
  parallel AB CD ∧ parallel AD BC

def condition2 (AB CD AD BC : Type) :=
  equal AB CD ∧ equal AD BC

def condition3 (AO CO BO DO : Type) :=
  equal AO CO ∧ equal BO DO

def condition4 (AB CD AD BC : Type) :=
  parallel AB CD ∧ equal AD BC

-- Definition to check if quadrilateral is parallelogram given conditions
def parallelogram (ABCD : Type) : Prop := sorry -- This would be the actual definition of a parallelogram which involves the properties of its sides, diagonals, etc.

-- Main theorem statement
theorem number_of_parallelogram_conditions :
  (condition1 AB CD AD BC → parallelogram ABCD)
  ∧ (condition2 AB CD AD BC → parallelogram ABCD)
  ∧ (condition3 AO CO BO DO → parallelogram ABCD)
  ∧ ¬(condition4 AB CD AD BC → parallelogram ABCD) :=
sorry

end number_of_parallelogram_conditions_l676_676718


namespace monotonically_increasing_iff_l676_676652

open Nat

noncomputable def sequence (n : ℕ) : ℝ → ℝ :=
  λ k, n^2 - k * n

theorem monotonically_increasing_iff (k : ℝ) : (∀ n : ℕ, n > 0 → sequence (n + 1) k - sequence n k > 0) ↔ k < 3 := by
  sorry

end monotonically_increasing_iff_l676_676652


namespace solve_equation_l676_676432

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) : (2 / x = 1 / (x + 1)) → x = -2 :=
begin
  sorry
end

end solve_equation_l676_676432


namespace max_piles_correct_l676_676484

def card_range := {x : ℕ // 1 ≤ x ∧ x ≤ 100}

def is_pile (a b c : card_range) : Prop := (c.1 = a.1 * b.1) ∧ (a ≠ b ∧ a ≠ c ∧ b ≠ c)

noncomputable def max_piles : ℕ :=
  8

theorem max_piles_correct :
  ∃ (piles : finset (finset card_range)), piles.card = max_piles ∧
    (∀ pile ∈ piles, ∃ a b c : card_range, pile = {a, b, c} ∧ is_pile a b c) ∧
    (∀ x : card_range, ∀ pile1 pile2 ∈ piles, pile1 ≠ pile2 → x ∈ pile1 → x ∉ pile2) :=
sorry

end max_piles_correct_l676_676484


namespace cos_theta_seven_l676_676688

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end cos_theta_seven_l676_676688


namespace solve_fractional_eq_l676_676443

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end solve_fractional_eq_l676_676443


namespace vertex_farthest_from_origin_l676_676789

/-- Given a square with center at (5, 3) and an area of 16 square units,
  the top side of which is horizontal, and that is then dilated 
  with the dilation center at (0, 0) and a scale factor of 3, 
  prove that the coordinates of the vertex of the image 
  of the square that is farthest from the origin are (21, 15). -/
theorem vertex_farthest_from_origin :
  let center := (5, 3)
  let area := 16
  let side_length := Real.sqrt area
  let dilation_center := (0, 0)
  let scale_factor := 3
  let original_vertices := [
    (center.1 - side_length / 2, center.2 - side_length / 2),
    (center.1 + side_length / 2, center.2 - side_length / 2),
    (center.1 + side_length / 2, center.2 + side_length / 2),
    (center.1 - side_length / 2, center.2 + side_length / 2)
  ]
  let dilated_vertices := original_vertices.map (λ ⟨x, y⟩, (scale_factor * x, scale_factor * y))
  ∃ vertex : (ℝ × ℝ), vertex ∈ dilated_vertices ∧
    vertex = (21, 15)
:= sorry

end vertex_farthest_from_origin_l676_676789


namespace initial_number_of_kids_l676_676826

theorem initial_number_of_kids (joined kids_total initial : ℕ) (h1 : joined = 22) (h2 : kids_total = 36) (h3 : kids_total = initial + joined) : initial = 14 :=
by 
  -- Proof goes here
  sorry

end initial_number_of_kids_l676_676826


namespace count_integers_satisfying_sqrt_condition_l676_676292

theorem count_integers_satisfying_sqrt_condition :
  ∀ x : ℕ, (⌈(Real.sqrt x)⌉ = 20) → (∃ n : ℕ, n = 39) :=
by {
  intro x,
  intro sqrt_x_ceil_eq_twenty,
  sorry
}

end count_integers_satisfying_sqrt_condition_l676_676292


namespace cherries_cost_l676_676619

def cost_per_kg (total_cost kilograms : ℕ) : ℕ :=
  total_cost / kilograms

theorem cherries_cost 
  (genevieve_amount : ℕ) 
  (short_amount : ℕ)
  (total_kilograms : ℕ) 
  (total_cost : ℕ := genevieve_amount + short_amount) 
  (cost : ℕ := cost_per_kg total_cost total_kilograms) : 
  cost = 8 :=
by
  have h1 : genevieve_amount = 1600 := by sorry
  have h2 : short_amount = 400 := by sorry
  have h3 : total_kilograms = 250 := by sorry
  sorry

end cherries_cost_l676_676619


namespace problem_solution_l676_676515

noncomputable def probability_at_least_one_three (prob : ℚ) : Prop :=
  ∃ (X1 X2 X3 : ℕ) (h1 : X1 ∈ finset.range 1 9)
    (h2 : X2 ∈ finset.range 1 9)
    (h3 : X3 ∈ finset.range 1 5),
    X1 + X2 = 2 * X3 ∧
    (X1 = 3 ∨ X2 = 3 ∨ X3 = 3) ∧
    prob = 3 / 16

theorem problem_solution : probability_at_least_one_three 3/16 :=
sorry

end problem_solution_l676_676515


namespace probability_both_divisible_by_4_l676_676054

def face_values := {1, 2, 3, 4, 5, 6}

def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

theorem probability_both_divisible_by_4 :
  let prob_one_die_is_4 := 1 / 6
  in prob_one_die_is_4 * prob_one_die_is_4 = 1 / 36 :=
by
  sorry

end probability_both_divisible_by_4_l676_676054


namespace factorize_xy_l676_676925

theorem factorize_xy (x y : ℕ): xy - x + y - 1 = (x + 1) * (y - 1) :=
by
  sorry

end factorize_xy_l676_676925


namespace correct_polynomial_l676_676375

noncomputable def p : Polynomial ℤ :=
  Polynomial.C 1 * Polynomial.X^6 - Polynomial.C 8 * Polynomial.X^4 - Polynomial.C 2 * Polynomial.X^3 + Polynomial.C 13 * Polynomial.X^2 - Polynomial.C 10 * Polynomial.X - Polynomial.C 1

theorem correct_polynomial (r t : ℝ) :
  (r^3 - r - 1 = 0) → (t = r + Real.sqrt 2) → Polynomial.aeval t p = 0 :=
by
  sorry

end correct_polynomial_l676_676375


namespace solve_equation_l676_676448

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end solve_equation_l676_676448


namespace sequence_sum_eq_zero_l676_676989

variable {n : ℕ}
variable {a : ℕ → ℝ}
variables (a_neq : ∀ i : ℕ, a (i+1) ≠ a (i-1))
variable (a_relation : ∀ i : ℕ, a (i+1)^2 - a i * a (i+1) + a i^2 = 0)
variable (a1 : a 1 = 1)
variable (anp1 : a (n+1) = 1)
variable (non_constant : ∃ i : ℕ, a (i) ≠ a (i+1))

theorem sequence_sum_eq_zero (h1 : a_relation) (h2 : a_neq) (h3 : a1) (h4 : anp1) (h5 : non_constant) : 
  ∑ i in Finset.range n, a i = 0 :=
by
  sorry

end sequence_sum_eq_zero_l676_676989


namespace quadrilateral_inequality_l676_676135

variable {a b c d e f : ℝ}

-- Condition: ABCD is a convex quadrilateral with given side lengths and diagonal lengths
def is_convex_quadrilateral (a b c d e f : ℝ) : Prop :=
  ∃ A B C D : ℝ, 
    -- Define whatever necessary properties pertain here, e.g., sides and diagonal lengths.

theorem quadrilateral_inequality (h : is_convex_quadrilateral a b c d e f) :
  a^2 + b^2 + c^2 + d^2 - e^2 - f^2 ≥ 0 := 
sorry

end quadrilateral_inequality_l676_676135


namespace sum_EO_1_to_150_l676_676581

def is_even (d : ℕ) : Prop := d % 2 = 0
def is_odd (d : ℕ) : Prop := d % 2 = 1

def E (n : ℕ) : ℕ :=
  (n.digits 10).filter is_even |>.sum

def O (n : ℕ) : ℕ :=
  (n.digits 10).filter is_odd |>.sum

theorem sum_EO_1_to_150 : ∑ n in finset.range 151, (E n + O n) = 1350 :=
by
  sorry

end sum_EO_1_to_150_l676_676581


namespace distance_between_points_l676_676958

noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem distance_between_points :
  dist (3, 3) (-2, -2) = 5 * Real.sqrt 2 := 
by
  sorry

end distance_between_points_l676_676958


namespace evaluate_polynomial_at_3_l676_676833

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 - 3*x + 2

theorem evaluate_polynomial_at_3 : f 3 = 2 :=
by
  sorry

end evaluate_polynomial_at_3_l676_676833


namespace price_of_case_l676_676660

variables (bottles_per_day : ℚ) (days : ℕ) (bottles_per_case : ℕ) (total_spent : ℚ)

def total_bottles_consumed (bottles_per_day : ℚ) (days : ℕ) : ℚ :=
  bottles_per_day * days

def cases_needed (total_bottles : ℚ) (bottles_per_case : ℕ) : ℚ :=
  total_bottles / bottles_per_case

def price_per_case (total_spent : ℚ) (cases : ℚ) : ℚ :=
  total_spent / cases

theorem price_of_case (h1 : bottles_per_day = 1/2)
                      (h2 : days = 240)
                      (h3 : bottles_per_case = 24)
                      (h4 : total_spent = 60) :
  price_per_case total_spent (cases_needed (total_bottles_consumed bottles_per_day days) bottles_per_case) = 12 := 
sorry

end price_of_case_l676_676660


namespace area_of_quadrilateral_BEGF_l676_676902

theorem area_of_quadrilateral_BEGF
  (ABCD : Type) [square ABCD]
  (side_length_ABCD : real)
  (E F : point)
  (midpoint_E_A_B : E = midpoint A B)
  (midpoint_F_B_C : F = midpoint B C)
  (side_length : side_length_ABCD = 20) :
  area (quadrilateral BEGF) = 80 :=
sorry

end area_of_quadrilateral_BEGF_l676_676902


namespace minimum_value_of_C_l676_676969

theorem minimum_value_of_C :
  ∃ C : ℝ, (∀ n : ℕ, n > 0 → (∑ k in finset.range (n - 1), (k + 1)^n) < C * n^n) ∧
  ∀ C' : ℝ, (∀ n : ℕ, n > 0 → (∑ k in finset.range (n - 1), (k + 1)^n) < C' * n^n) → C ≤ C' :=
begin
  use 1 / (exp 1 - 1),
  split,
  { -- Prove ∀ n : ℕ, n > 0 → (∑ k in finset.range (n - 1), (k + 1)^n) < (1 / (exp 1 - 1)) * n^n
    sorry },
  { -- Prove ∀ C' : ℝ, (∀ n : ℕ, n > 0 → (∑ k in finset.range (n - 1), (k + 1)^n) < C' * n^n) → 1 / (exp 1 - 1) ≤ C'
    sorry }
end

end minimum_value_of_C_l676_676969


namespace smallest_number_of_cookies_l676_676381

theorem smallest_number_of_cookies : ∃ x : ℤ, 
  (x % 6 = 5) ∧ 
  (x % 8 = 7) ∧ 
  (x % 9 = 2) ∧ 
  (∀ y, (y % 6 = 5) ∧ (y % 8 = 7) ∧ (y % 9 = 2) → y ≥ x) :=
by
  have exist_x : ∃ x : ℤ, (x % 6 = 5) ∧ (x % 8 = 7) ∧ (x % 9 = 2) := sorry,
  obtain ⟨x, hx⟩ := exist_x,
  existsi x,
  split, { exact hx.1 },
  split, { exact hx.2.1 },
  split, { exact hx.2.2.1 },
  sorry

end smallest_number_of_cookies_l676_676381


namespace max_area_triangle_PAB_l676_676650

theorem max_area_triangle_PAB {m x y : ℝ} :
  let line := m * x - y + m + 2 = 0
  let circle1 := (x + 1)^2 + (y - 2)^2 = 1
  let circle2 := (x - 3)^2 + y^2 = 5
  ∃ P A B : ℝ × ℝ,
    (P ∈ circle2) ∧
    (A ≠ B) ∧
    (A ∈ circle1) ∧ 
    (B ∈ circle1) ∧
    (A ∈ line) ∧ 
    (B ∈ line) ∧
    (P, A, and B are not colinear) ∧
    ((area_of_triangle P A B)  = 3 * real.sqrt 5)
  :=
sorry

end max_area_triangle_PAB_l676_676650


namespace total_solar_systems_and_planets_l676_676051

theorem total_solar_systems_and_planets (planets : ℕ) (solar_systems_per_planet : ℕ) (h1 : solar_systems_per_planet = 8) (h2 : planets = 20) : (planets + planets * solar_systems_per_planet) = 180 :=
by
  -- translate conditions to definitions
  let solar_systems := planets * solar_systems_per_planet
  -- sum solar systems and planets
  let total := planets + solar_systems
  -- exact proof goal
  exact calc
    (planets + solar_systems) = planets + planets * solar_systems_per_planet : by rfl
                        ... = 20 + 20 * 8                       : by rw [h1, h2]
                        ... = 180                                : by norm_num

end total_solar_systems_and_planets_l676_676051


namespace z1_z2_product_l676_676215

def z1 : ℂ := 2 / (1 + I)
def z2 : ℂ := -1 + I

theorem z1_z2_product : z1 * z2 = 2 * I :=
by
  sorry

end z1_z2_product_l676_676215


namespace pounds_per_pie_l676_676782

-- Define the conditions
def total_weight : ℕ := 120
def applesauce_weight := total_weight / 2
def pies_weight := total_weight - applesauce_weight
def number_of_pies := 15

-- Define the required proof for pounds per pie
theorem pounds_per_pie :
  pies_weight / number_of_pies = 4 := by
  sorry

end pounds_per_pie_l676_676782


namespace office_speed_l676_676520

variable (d v : ℝ)

theorem office_speed (h1 : v > 0) (h2 : ∀ t : ℕ, t = 30) (h3 : (2 * d) / (d / v + d / 30) = 24) : v = 20 := 
sorry

end office_speed_l676_676520


namespace calculate_grand_total_l676_676659

noncomputable def calculate_total_cost : ℝ :=
let original_price := 20.00 in
let discount := 0.20 in
let discounted_price := original_price * (1 - discount) in
let monogram_cost := 12.00 in
let pre_tax_cost := discounted_price + monogram_cost in
let tax_A := 0.06 in
let tax_B := 0.08 in
let tax_C := 0.055 in
let tax_D := 0.0725 in
let tax_E := 0.04 in
let shipping_A := 3.50 in
let shipping_B := 4.25 in
let shipping_C := 2.75 in
let shipping_D := 3.75 in
let shipping_E := 2.25 in
let total_A := pre_tax_cost * (1 + tax_A) + shipping_A in
let total_B := pre_tax_cost * (1 + tax_B) + shipping_B in
let total_C := pre_tax_cost * (1 + tax_C) + shipping_C in
let total_D := pre_tax_cost * (1 + tax_D) + shipping_D in
let total_E := pre_tax_cost * (1 + tax_E) + shipping_E in
let grand_total := total_A + total_B + total_C + total_D + total_E in
let coupon := 5.00 in
grand_total - coupon

theorem calculate_grand_total : calculate_total_cost = 160.11 :=
by
  sorry

end calculate_grand_total_l676_676659


namespace can_transform_to_l676_676052

variable (numbers : List ℤ)

def transformation : List ℤ → List ℤ 
| [] => []
| [x] => [x]
| [x, y] => [x, y]
| x :: y :: z :: rest => (x + y) :: -y :: (z + y) :: rest

def is_possible (initial_set target_set : List ℤ) : Prop :=
  ∃ (steps : ℕ) (op : (List ℤ → List ℤ) → List ℤ),
    op transformation initial_set = target_set

theorem can_transform_to (initial_set target_set : List ℤ) :
  initial_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10] →
  target_set = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1] →
  is_possible initial_set target_set :=
by
  sorry

end can_transform_to_l676_676052


namespace odd_function_f_f_at_1_l676_676233

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then -x * Real.log (3 - x) else -(−x * Real.log (3 + x))

theorem odd_function_f (x : ℝ) : f(x) = -f(-x) :=
by
  rw f
  rw f (-x)
  split_ifs
  case inl _ h₂ => congr; rw [Real.log_sub_eq_neg_log_add, neg_neg, h₂]
  case inr h₁' _ => congr; rw [Real.log_sub_eq_neg_log_add, neg_neg, not_le.1 h₁'] 

theorem f_at_1 : f 1 = -Real.log 4 :=
by
  have : f(-1) = Real.log 4 := by
    rw f
    norm_num
    split_ifs
    case inl => ring
  rw this
  exact odd_function_f 1

end odd_function_f_f_at_1_l676_676233


namespace propositions_correct_l676_676151

-- Define the propositions
def prop1 (f : ℝ → ℝ) := 
  (∀ x : ℝ, f (-x + 2) = f (x - 2))

def prop2 := 
  (∀ x₁ x₂ : ℝ, Real.exp ((x₁ + x₂) / 2) ≤ (Real.exp x₁ + Real.exp x₂) / 2)

noncomputable def prop3 (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) := 
  (∀ x : ℝ, (x > 0) → Real.log (a) x = a * x) ∧ (Real.log (a) 2 > Real.log (a) (-2) )

def prop4 := 
  (∀ x : ℝ, ((x-2014)^2 - 2) = (x + 2013)^2 - 2 * x - 1) ∧ (∀ x : ℝ, ∃ m : ℝ, m = -2)

-- The main statement
theorem propositions_correct :
  prop2 ∧ prop4 := by
  sorry

end propositions_correct_l676_676151


namespace solve_equation_l676_676434

theorem solve_equation : ∃ x : ℝ, (2 / x) = (1 / (x + 1)) ∧ x = -2 :=
by
  use -2
  split
  { -- Proving the equality part
    show (2 / -2) = (1 / (-2 + 1)),
    simp,
    norm_num
  }
  { -- Proving the equation part
    refl
  }

end solve_equation_l676_676434


namespace num_possible_x_l676_676269

theorem num_possible_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 20) : {y : ℕ | 361 < y ∧ y ≤ 400}.card = 39 :=
by
  sorry

end num_possible_x_l676_676269


namespace ways_to_fill_grid_l676_676715

/-- In how many ways can you fill a 3x3 table with the numbers 1 through 9 (each used once)
such that all pairs of adjacent numbers (sharing one side) are relatively prime? -/
theorem ways_to_fill_grid : 
  ∃ n : ℕ, n = 2016 ∧
    ∀ (table : Fin 3 × Fin 3 → ℕ),
    (∀ i j, i ≠ j → table i ≠ table j) ∧
    (∀ i, table i ∈ Finset.range 1 10) ∧
    (∀ i j, adjacent i j → Nat.coprime (table i) (table j)) →
    n = 2016 :=
by
  sorry

end ways_to_fill_grid_l676_676715


namespace sum_of_coordinates_l676_676298

theorem sum_of_coordinates {g h : ℝ → ℝ} 
  (h₁ : g 4 = 5)
  (h₂ : ∀ x, h x = (g x)^2) :
  4 + h 4 = 29 := by
  sorry

end sum_of_coordinates_l676_676298


namespace eight_digit_numbers_with_consecutive_digits_l676_676661

theorem eight_digit_numbers_with_consecutive_digits : 
  let num_digits := 8 in
  let valid_count := 230 in 
  ∃ n: ℕ, (num_digits = 8 ∧ valid_count = 230) ∧
  (n = valid_count) :=
sorry

end eight_digit_numbers_with_consecutive_digits_l676_676661


namespace solve_for_a_l676_676227

open Complex

noncomputable def problem_statement (a : ℝ) : Prop :=
  let z := (a : ℂ) - Complex.i
  (z / (1 + Complex.i)).re = 0

theorem solve_for_a (a : ℝ) (h : problem_statement a) : a = 1 := 
by sorry

end solve_for_a_l676_676227


namespace canteen_consumption_l676_676505

theorem canteen_consumption :
  ∀ (x : ℕ),
    (x + (500 - x) + (200 - x)) = 700 → 
    (500 - x) = 7 * (200 - x) →
    x = 150 :=
by
  sorry

end canteen_consumption_l676_676505


namespace rect_solution_proof_l676_676990

noncomputable def rect_solution_exists : Prop :=
  ∃ (l2 w2 : ℝ), 2 * (l2 + w2) = 12 ∧ l2 * w2 = 4 ∧
               l2 = 3 + Real.sqrt 5 ∧ w2 = 3 - Real.sqrt 5

theorem rect_solution_proof : rect_solution_exists :=
  by
    sorry

end rect_solution_proof_l676_676990


namespace union_sets_l676_676377

theorem union_sets (p q: ℝ) 
(h1: (∀ x : ℝ, ¬(x^2 + p * x + 12 = 0) → (x^2 - 5 * x + q = 0)) ∧ ∃ x : ℝ, x = 2) 
(h2: (∀ x : ℝ, (x^2 + p * x + 12 = 0) → ¬(x^2 - 5 * x + q = 0)) ∧ ∃ x : ℝ, x = 4) : 
({x : ℝ | x^2 + p * x + 12 = 0} ∪ {x : ℝ | x^2 - 5 * x + q = 0} = ({2, 3, 4} : set ℝ)) := by 
sorry

end union_sets_l676_676377


namespace count_integers_satisfying_sqrt_condition_l676_676289

theorem count_integers_satisfying_sqrt_condition :
  ∀ x : ℕ, (⌈(Real.sqrt x)⌉ = 20) → (∃ n : ℕ, n = 39) :=
by {
  intro x,
  intro sqrt_x_ceil_eq_twenty,
  sorry
}

end count_integers_satisfying_sqrt_condition_l676_676289


namespace find_bases_l676_676058

def is_valid_equation (n k : ℕ) : Prop :=
  n^2 + 1 = k^4 + k^3 + k + 1

theorem find_bases : ∃ n k : ℕ, n = 18 ∧ k = 4 ∧ is_valid_equation n k :=
by {
  use [18, 4],
  split,
  rfl,
  split,
  rfl,
  dsimp [is_valid_equation],
  rw [pow_two, pow_four],
  norm_num,
  sorry
}

end find_bases_l676_676058


namespace divisible_by_120_l676_676775

theorem divisible_by_120 (n : ℕ) : 120 ∣ n * (n^2 - 1) * (n^2 - 5 * n + 26) := sorry

end divisible_by_120_l676_676775


namespace sheets_of_paper_in_each_box_l676_676738

theorem sheets_of_paper_in_each_box (E S : ℕ) (h1 : 2 * E + 40 = S) (h2 : 4 * (E - 40) = S) : S = 240 :=
by
  sorry

end sheets_of_paper_in_each_box_l676_676738


namespace profit_maximization_l676_676101

noncomputable def revenue : ℝ → ℝ
| x := if x ≤ 400 then 400 * x - 0.5 * x ^ 2 else 80000

noncomputable def profit : ℝ → ℝ
| x := if 0 ≤ x ∧ x ≤ 400 then -0.5 * x ^ 2 + 300 * x - 20000 else -100 * x + 60000

theorem profit_maximization :
  ( ∀ x : ℝ, 0 ≤ x ∧ x ≤ 400 ∧ profit x = -0.5 * x ^ 2 + 300 * x - 20000 ) ∧
  ( ∀ x : ℝ, x > 400 ∧ profit x = -100 * x + 60000 ) ∧
  ( ∃ x : ℝ, x = 300 ∧ profit x = 25000 ) := by
  sorry

end profit_maximization_l676_676101


namespace distance_between_points_l676_676936

theorem distance_between_points :
  ∀ (P Q : ℝ × ℝ), P = (3, 3) ∧ Q = (-2, -2) → dist P Q = 5 * real.sqrt 2 :=
begin
  sorry
end

end distance_between_points_l676_676936


namespace union_of_sets_l676_676218

def A : Set ℝ := { x : ℝ | x * (x + 1) ≤ 0 }
def B : Set ℝ := { x : ℝ | -1 < x ∧ x < 1 }
def C : Set ℝ := { x : ℝ | -1 ≤ x ∧ x < 1 }

theorem union_of_sets :
  A ∪ B = C := 
sorry

end union_of_sets_l676_676218


namespace find_f_2024_l676_676985

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_f_2024 (a b c : ℝ)
  (h1 : f 2021 a b c = 2021)
  (h2 : f 2022 a b c = 2022)
  (h3 : f 2023 a b c = 2023) :
  f 2024 a b c = 2030 := sorry

end find_f_2024_l676_676985


namespace probability_yellow_balls_twice_correct_l676_676823

noncomputable def probability_draw_yellow_balls_twice (initial_white : ℕ) (initial_yellow : ℕ) : ℚ :=
  let first_draw_prob := initial_yellow / (initial_white + initial_yellow) in
  let second_draw_prob := (initial_yellow - 1) / (initial_white + initial_yellow - 1) in
  first_draw_prob * second_draw_prob

theorem probability_yellow_balls_twice_correct :
  probability_draw_yellow_balls_twice 1 2 = 1 / 3 :=
by sorry

end probability_yellow_balls_twice_correct_l676_676823


namespace smallest_positive_period_of_f_max_min_values_of_f_on_interval_l676_676243

noncomputable section

def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x - Real.cos x) + 1

theorem smallest_positive_period_of_f : ∃ p > 0, ∀ x, f (x + p) = f x ∧ (∀ q > 0, (∀ x, f (x + q) = f x) → p ≤ q) :=
sorry

theorem max_min_values_of_f_on_interval :
  let a := Real.pi / 8
  let b := 3 * Real.pi / 4
  ∃ m M, (∀ x ∈ set.Icc a b, m ≤ f x ∧ f x ≤ M) ∧
         (∀ y, y ∈ set.Icc a b → y ≠ a → y ≠ b → (f y = m ∨ f y = M)) ∧
         M = sqrt 2 ∧ m = -1 :=
sorry

end smallest_positive_period_of_f_max_min_values_of_f_on_interval_l676_676243


namespace cos_theta_seven_l676_676690

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end cos_theta_seven_l676_676690


namespace solve_equation_l676_676426

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end solve_equation_l676_676426


namespace olympic_high_school_amc10_l676_676138

/-- At Olympic High School, 2/5 of the freshmen and 4/5 of the sophomores took the AMC-10.
    Given that the number of freshmen and sophomore contestants was the same, there are twice as many freshmen as sophomores. -/
theorem olympic_high_school_amc10 (f s : ℕ) (hf : f > 0) (hs : s > 0)
  (contest_equal : (2 / 5 : ℚ)*f = (4 / 5 : ℚ)*s) : f = 2 * s :=
by
  sorry

end olympic_high_school_amc10_l676_676138


namespace integer_values_x_l676_676283

theorem integer_values_x {x : ℝ} (h : ⌈real.sqrt x⌉ = 20) : ∃ n : ℕ, n = 39 :=
by
  have h1 : 19 < real.sqrt x ∧ real.sqrt x ≤ 20 := sorry
  have h2 : 361 < x ∧ x ≤ 400 := sorry
  have x_values : set.Icc 362 400 := sorry
  have n_values : ∃ n : ℕ, n = set.finite.to_finset x_values.card := sorry
  exact n_values

end integer_values_x_l676_676283


namespace cos_seven_theta_l676_676691

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end cos_seven_theta_l676_676691


namespace number_of_sodas_in_pack_l676_676906

/-- Billy has twice as many brothers as sisters -/
def twice_as_many_brothers_as_sisters (brothers sisters : ℕ) : Prop :=
  brothers = 2 * sisters

/-- Billy has 2 sisters -/
def billy_has_2_sisters : Prop :=
  ∃ sisters : ℕ, sisters = 2

/-- Billy can give 2 sodas to each of his siblings if he wants to give out the entire pack while giving each sibling the same number of sodas -/
def divide_sodas_evenly (total_sodas siblings sodas_per_sibling : ℕ) : Prop :=
  total_sodas = siblings * sodas_per_sibling

/-- Determine the total number of sodas in the pack given the conditions -/
theorem number_of_sodas_in_pack : 
  ∃ (sisters brothers total_sodas : ℕ), 
    (twice_as_many_brothers_as_sisters brothers sisters) ∧ 
    (billy_has_2_sisters) ∧ 
    (divide_sodas_evenly total_sodas (sisters + brothers + 1) 2) ∧
    (total_sodas = 12) :=
by
  sorry

end number_of_sodas_in_pack_l676_676906


namespace dot_product_is_correct_l676_676220

def vector_a : ℝ × ℝ × ℝ := (-1, 1, -2)
def vector_b : ℝ × ℝ × ℝ := (1, -2, -1)
def vector_n (k : ℝ) : ℝ × ℝ × ℝ := (k, -2 * k, -2)

theorem dot_product_is_correct (k : ℝ) (hk : k = 2) :
  let n := vector_n k in
  let dot_product := (vector_a.1 * n.1) + (vector_a.2 * n.2) + (vector_a.3 * n.3) in
  dot_product = -2 :=
by
  sorry

end dot_product_is_correct_l676_676220


namespace coeff_x5_in_expansion_l676_676315

theorem coeff_x5_in_expansion :
  (x : ℝ) → polynomial.eval x ((x^2 + 1)^2 * (x - 1)^6) = -52 * x^5 + (other_terms : polynomial ℝ) := by
  sorry

end coeff_x5_in_expansion_l676_676315


namespace james_total_carrot_sticks_l676_676733

theorem james_total_carrot_sticks : 
  ∀ (before_dinner after_dinner : Nat), before_dinner = 22 → after_dinner = 15 → 
  before_dinner + after_dinner = 37 := 
by
  intros before_dinner after_dinner h1 h2
  rw [h1, h2]
  rfl

end james_total_carrot_sticks_l676_676733


namespace num_possible_integers_l676_676279

theorem num_possible_integers (x : ℕ) (h : ⌈Real.sqrt x⌉ = 20) : ∃ n : ℕ, n = 39 :=
by
  have h1 : 19 < Real.sqrt x ∧ Real.sqrt x ≤ 20 := sorry
  have h2 : 361 < x ∧ x ≤ 400 := sorry
  have h3 : ∃ (a b : ℕ), 361 < a ∧ a ≤ 400 ∧ b = a - 361 ∧ b + 1 = 39 := sorry
  use 39
  exact h3.right.right
  sorry

end num_possible_integers_l676_676279


namespace probability_at_least_one_diamond_l676_676871

theorem probability_at_least_one_diamond :
  ∃ p : ℚ, p = 15 / 34 ∧ 
  let no_replacement := true,
      total_cards := 52,
      diamonds := 13,
      non_diamonds := 39 in
  ∀ first_card second_card : ℕ,
  first_card ∈ {0...total_cards - 1} ∧ second_card ∈ {0...total_cards - 2} ∧ first_card ≠ second_card →
  p = 1 - (non_diamonds / (total_cards : ℚ)) * ((non_diamonds - 1) / (total_cards - 1 : ℚ)) :=
sorry

end probability_at_least_one_diamond_l676_676871


namespace degree_greater_than_2_l676_676343

variable (P Q : ℤ[X]) -- P and Q are polynomials with integer coefficients

theorem degree_greater_than_2 (P_nonconstant : ¬(P.degree = 0))
  (Q_nonconstant : ¬(Q.degree = 0))
  (h : ∃ S : Finset ℤ, S.card ≥ 25 ∧ ∀ x ∈ S, (P.eval x) * (Q.eval x) = 2009) :
  P.degree > 2 ∧ Q.degree > 2 :=
by
  sorry

end degree_greater_than_2_l676_676343


namespace calc_expression_l676_676840

theorem calc_expression : (3.242 * 14) / 100 = 0.45388 :=
by
  sorry

end calc_expression_l676_676840


namespace field_trip_buses_l676_676522

theorem field_trip_buses (x y : ℕ) (h1 : 30 * x + 42 * y ≥ 300) (h2 : 300 * x + 400 * y ≤ 3100) (h3 : x + y ≤ 8) :
  ∃! (x y : ℕ), 30 * x + 42 * y ≥ 300 ∧ 300 * x + 400 * y ≤ 3100 ∧ x + y ≤ 8 ∧ 300 * x + 400 * y = 2900 :=
begin
  sorry
end

end field_trip_buses_l676_676522


namespace range_of_angle_C_l676_676302

variable {A B C: ℝ}

theorem range_of_angle_C 
  (hA: A = π / 4)
  (hSinCondition : sin B > sqrt 2 * cos C)
  (hSumAngles : A + B + C = π) :
  (π / 4) < C ∧ C < (3 * π / 4) :=
by
  sorry

end range_of_angle_C_l676_676302


namespace frank_initial_money_l676_676196

theorem frank_initial_money (X : ℝ) (h1 : X * (4 / 5) * (3 / 4) * (6 / 7) * (2 / 3) = 600) : X = 2333.33 :=
sorry

end frank_initial_money_l676_676196


namespace hyperbola_asymptotic_lines_l676_676794

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := 3 * x^2 - y^2 = 3

-- Define the asymptotic lines
def are_asymptotic_lines (x y : ℝ) : Prop := y = sqrt 3 * x ∨ y = -sqrt 3 * x

-- Theorem statement
theorem hyperbola_asymptotic_lines : 
  (∀ x y : ℝ, hyperbola_equation x y → are_asymptotic_lines x y) :=
  sorry

end hyperbola_asymptotic_lines_l676_676794


namespace total_people_on_boats_l676_676498

theorem total_people_on_boats (boats : ℕ) (people_per_boat : ℕ) (h_boats : boats = 5) (h_people : people_per_boat = 3) : boats * people_per_boat = 15 :=
by
  sorry

end total_people_on_boats_l676_676498


namespace find_multiple_l676_676039

theorem find_multiple (m : ℤ) : 38 + m * 43 = 124 → m = 2 := by
  intro h
  sorry

end find_multiple_l676_676039


namespace triangle_area_l676_676726

theorem triangle_area (a c : ℝ) (B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : B = π / 3) : 
  (1 / 2 * a * c * Real.sin B) = sqrt 3 / 2 :=
by
  -- Conditions for the problem
  rw [h1, h2, h3]
  -- Solve by simplifying the expression
  sorry

end triangle_area_l676_676726


namespace probability_at_least_40_cents_heads_l676_676002

theorem probability_at_least_40_cents_heads :
  let coins := {50, 25, 10, 5, 1}
  3 / 8 =
    (∑ H in {x ∈ (Finset.powerset coins) | (x.sum ≥ 40)}, (1 / 2) ^ x.card) :=
sorry

end probability_at_least_40_cents_heads_l676_676002


namespace conjugate_in_first_quadrant_l676_676314

noncomputable def conjugate_quadrant : Prop :=
  let z := (2 - complex.i) / (1 + complex.i)
  let conj_z := complex.conj z
  conj_z.re > 0 ∧ conj_z.im > 0

theorem conjugate_in_first_quadrant : conjugate_quadrant :=
by
  sorry

end conjugate_in_first_quadrant_l676_676314


namespace trajectory_of_circle_l676_676885

theorem trajectory_of_circle (x y : ℝ) (r : ℝ) (h1 : (x, y) = (0, 1) → r = sqrt (x^2 + (y - 1)^2)) (h2 : r = abs (y + 1)) : x^2 = 4 * y :=
by
  sorry

end trajectory_of_circle_l676_676885


namespace shadow_of_sphere_is_parabola_l676_676213

theorem shadow_of_sphere_is_parabola
  (center : ℝ^3)
  (radius : ℝ)
  (luminous_point : ℝ^3)
  (projection_plane : ℝ^2)
  (is_sphere : ∀ p : ℝ^3, (p - center).norm = radius)
  (is_on_plane : ∀ p : ℝ^3, (p.2 = 0) → (∃ q : ℝ^2, p.1 = q.1 ∧ p.3 = q.2)) :
  ∃ parabola : ℝ^2 → ℝ, 
    ∀ p : ℝ^2, (project_shadow luminous_point center radius).p.1 = parabola p.1 :=
sorry

end shadow_of_sphere_is_parabola_l676_676213


namespace ellipse_inradius_circumradius_inequality_l676_676374

-- Let \(\Gamma\) be an ellipse with foci \(F_1\) and \(F_2\) and eccentricity \(e\).
-- Let \(P\) be any point on the ellipse, excluding the vertices of the major axis.
-- Let \(r\) be the inradius and \(R\) be the circumradius of \(\triangle P F_1 F_2\).
-- Prove that \(\frac{r}{R} \leq 2 e(1-e)\).

theorem ellipse_inradius_circumradius_inequality
  (e : ℝ)
  (F1 F2 P : ℝ → ℝ)
  (r R : ℝ)
  (h1 : e ∈ [0, 1]) -- e is the eccentricity of the ellipse and is between 0 and 1
  (h2 : P is_on_ellipse_with_foci F1 F2 e) -- P is a point on ellipse Γ with foci F1, F2 and eccentricity e
  (h3 : r = inradius_of_triangle (P, F1, F2)) -- r is the inradius of triangle P F1 F2
  (h4 : R = circumradius_of_triangle (P, F1, F2)) -- R is the circumradius of triangle P F1 F2
  : (r / R) ≤ 2 * e * (1 - e) :=
sorry

end ellipse_inradius_circumradius_inequality_l676_676374


namespace distance_points_l676_676930

-- Definition of distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Points
def point1 : ℝ × ℝ := (3, 3)
def point2 : ℝ × ℝ := (-2, -2)

-- Main theorem
theorem distance_points : distance point1 point2 = 5 * Real.sqrt 2 :=
by
  sorry

end distance_points_l676_676930


namespace remainder_is_one_l676_676063

theorem remainder_is_one (dividend divisor quotient remainder : ℕ) 
  (h1 : dividend = 222) 
  (h2 : divisor = 13)
  (h3 : quotient = 17)
  (h4 : dividend = divisor * quotient + remainder) : remainder = 1 :=
sorry

end remainder_is_one_l676_676063


namespace arg_sum_l676_676754

noncomputable def z (k : ℕ) : ℂ := complex.mk (2 * k^2 : ℝ) (1 : ℝ)

theorem arg_sum :
  (∑ k in finset.range 100, complex.arg (z (k + 1))) = (real.pi / 4) - real.arccot 201 := 
sorry

end arg_sum_l676_676754


namespace length_of_FQ_l676_676150

theorem length_of_FQ (DE DF EF FQ : ℝ) 
  (h_right_angle_E : right_triangle DEG DEF E)
  (h_DE : DE = 3)
  (h_DF : DF = Real.sqrt 34)
  (h_circle_tangent : ∃ (C : point), C ∈ line DE ∧ tangent C DF ∧ tangent C EF) :
  FQ = 5 :=
sorry

end length_of_FQ_l676_676150


namespace num_possible_integers_l676_676281

theorem num_possible_integers (x : ℕ) (h : ⌈Real.sqrt x⌉ = 20) : ∃ n : ℕ, n = 39 :=
by
  have h1 : 19 < Real.sqrt x ∧ Real.sqrt x ≤ 20 := sorry
  have h2 : 361 < x ∧ x ≤ 400 := sorry
  have h3 : ∃ (a b : ℕ), 361 < a ∧ a ≤ 400 ∧ b = a - 361 ∧ b + 1 = 39 := sorry
  use 39
  exact h3.right.right
  sorry

end num_possible_integers_l676_676281


namespace probability_two_odd_numbers_l676_676197

theorem probability_two_odd_numbers {s : Finset ℕ} (h : s = {1, 2, 3, 4}) :
  let draws_wo_replacement := ((s.erase 1).erase 3) ∪ ((s.erase 3).erase 1)
  let total_pairs := s.product s
  Finset.card draws_wo_replacement.to_finset / Finset.card total_pairs.to_finset = 1 / 6 :=
by {
  sorry
}

end probability_two_odd_numbers_l676_676197


namespace sum_seq_eq_zero_l676_676987

theorem sum_seq_eq_zero {a : ℕ → ℂ} (n : ℕ)
  (h₀ : ∀ i, (a (i + 1)) ^ 2 - a (i + 1) * a i + (a i) ^ 2 = 0)
  (h₁ : ∀ i, 1 ≤ i → i ≤ n → a (i + 1) ≠ a (i - 1))
  (h₂ : a 1 = 1)
  (h₃ : a (n + 1) = 1) :
  ∑ i in finset.range n, a i = 0 :=
sorry

end sum_seq_eq_zero_l676_676987


namespace root_of_equation_inequality_solution_set_l676_676246

-- Define the function
def f (x : ℝ) : ℝ := 2^x - 2 / x

-- Prove root of the equation f(x) = 0 is x = 1
theorem root_of_equation : f 1 = 0 :=
by 
  sorry

-- Prove the solution set of the inequality f(x) > 0 is (-∞, 0) ∪ (1, ∞)
theorem inequality_solution_set (x : ℝ) (hx : x ≠ 0) : f x > 0 ↔ (x < 0 ∨ x > 1) :=
by
  sorry

end root_of_equation_inequality_solution_set_l676_676246


namespace min_value_of_quadratic_l676_676966

theorem min_value_of_quadratic (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a^2 ≠ b^2) : 
  ∃ (x : ℝ), (∃ (y_min : ℝ), y_min = -( (abs (a - b)/2)^2 ) 
  ∧ ∀ (x : ℝ), (x - a)*(x - b) ≥ y_min) :=
sorry

end min_value_of_quadratic_l676_676966


namespace find_teachers_and_students_l676_676094

-- Mathematical statements corresponding to the problem conditions
def teachers_and_students_found (x y : ℕ) : Prop :=
  (y = 30 * x + 7) ∧ (31 * x = y + 1)

-- The theorem we need to prove
theorem find_teachers_and_students : ∃ x y, teachers_and_students_found x y ∧ x = 8 ∧ y = 247 :=
  by
    sorry

end find_teachers_and_students_l676_676094


namespace num_factors_36_l676_676664

theorem num_factors_36 : ∀ (n : ℕ), n = 36 → (∃ (a b : ℕ), 36 = 2^a * 3^b ∧ a = 2 ∧ b = 2 ∧ (a + 1) * (b + 1) = 9) :=
by
  sorry

end num_factors_36_l676_676664


namespace player_A_wins_if_m_gt_2n_player_A_wins_if_m_gt_alpha_n_l676_676045

-- Define the setup for the matchstick game with given conditions.
def matchsticks_game : Type :=
{ m n : ℕ // m > n }

-- Define the condition for player A to win for part 1.
theorem player_A_wins_if_m_gt_2n (m n : ℕ) (h : m > n) (hp : m > 2 * n) : true :=
sorry

-- Define the value of alpha as the golden ratio.
noncomputable def alpha : ℝ := (1 + Real.sqrt 5) / 2

-- Define the condition for player A to win for part 2.
theorem player_A_wins_if_m_gt_alpha_n (m n : ℕ) (h : m > n) (ha : (m : ℝ) > alpha * n) : true :=
sorry

end player_A_wins_if_m_gt_2n_player_A_wins_if_m_gt_alpha_n_l676_676045


namespace find_a_and_tangent_line_l676_676250

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2 * x^2 + a * x - 1

theorem find_a_and_tangent_line (a x y : ℝ) (h1 : f a 1 = x^3 - 2 * x^2 + 2 * x - 1)
  (h2 : (deriv (f a) 1 = 1)) :
  a = 2 ∧ (exists y, (y + 6 = 9 * (x + 1)))) :=
by
sry

end find_a_and_tangent_line_l676_676250


namespace triangle_side_lengths_l676_676294

noncomputable def valid_triangle (x : ℝ) : Prop :=
  sqrt (3 * x - 1) + sqrt (3 * x + 1) > 2 * sqrt x ∧
  sqrt (3 * x - 1) + 2 * sqrt x > sqrt (3 * x + 1) ∧
  sqrt (3 * x + 1) + 2 * sqrt x > sqrt (3 * x - 1)

noncomputable def not_right_triangle (x : ℝ) : Prop :=
  (sqrt (3 * x - 1))^2 + (sqrt (3 * x + 1))^2 ≠ (2 * sqrt x)^2 →
  (sqrt (3 * x - 1))^2 + (2 * sqrt x)^2 ≠ (sqrt (3 * x + 1))^2 →
  (sqrt (3 * x + 1))^2 + (2 * sqrt x)^2 ≠ (sqrt (3 * x - 1))^2

theorem triangle_side_lengths (x : ℝ) (h : x > 1/3) : valid_triangle x ∧ not_right_triangle x :=
sorry

end triangle_side_lengths_l676_676294


namespace number_of_possible_integers_l676_676273

theorem number_of_possible_integers (x: ℤ) (h: ⌈real.sqrt ↑x⌉ = 20) : 39 :=
  sorry

end number_of_possible_integers_l676_676273


namespace num_arrangements_l676_676454

-- Define the constraints and conditions as given in the problem
def student : Type := { A, B, C, D, E, F }

def dormitory : Type := { 1, 2, 3 }

def condition_student_A_in_dorm1 (assignment : student → dormitory) : Prop :=
  assignment A = 1

def condition_students_B_C_not_in_dorm3 (assignment : student → dormitory) : Prop :=
  assignment B ≠ 3 ∧ assignment C ≠ 3

def condition_each_dorm_has_2_students (assignment : student → dormitory) : Prop :=
  ∀ d : dormitory, (assignment.to_list).count d = 2

-- The proof problem
theorem num_arrangements : ∃ (assignment : student → dormitory), 
  condition_student_A_in_dorm1 assignment ∧
  condition_students_B_C_not_in_dorm3 assignment ∧
  condition_each_dorm_has_2_students assignment ∧
  (number_of_arrangements assignment = 25) :=
sorry

end num_arrangements_l676_676454


namespace problem_1_problem_2_l676_676633

variable (a b c : ℝ)

-- Conditions for the first proof
variables (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)

-- Statement for the first proof
theorem problem_1 (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  sqrt (a * b) + sqrt (b * c) + sqrt (c * a) ≤ a + b + c := 
sorry

-- Additional condition for the second proof
variable (h_sum : a + b + c = 1)

-- Statement for the second proof
theorem problem_2 (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 1) : 
  2 * a * b / (a + b) + 2 * b * c / (b + c) + 2 * c * a / (c + a) ≤ 1 := 
sorry

end problem_1_problem_2_l676_676633


namespace total_bricks_needed_equals_l676_676486

noncomputable def area_courtyard_m2 (length: ℝ) (width: ℝ) : ℝ :=
  length * width

noncomputable def area_courtyard_cm2 (length: ℝ) (width: ℝ) : ℝ :=
  area_courtyard_m2 length width * 10000

noncomputable def area_brick_cm2 (length: ℝ) (width: ℝ) : ℝ :=
  length * width

noncomputable def total_bricks_required (courtyard_area: ℝ) (brick_area: ℝ) : ℕ :=
  ⌈courtyard_area / brick_area⌉.to_nat

theorem total_bricks_needed_equals :
  total_bricks_required (area_courtyard_cm2 28 13) (area_brick_cm2 22 12) = 13788 :=
by
  sorry

end total_bricks_needed_equals_l676_676486


namespace find_multiple_l676_676036

theorem find_multiple (m : ℤ) (h : 38 + m * 43 = 124) : m = 2 := by
    sorry

end find_multiple_l676_676036


namespace range_of_a_l676_676980

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 1 → log_a a (2 - a * x) < log_a a (2 - a * (x / 2))) →
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l676_676980


namespace spinner_three_digit_prob_div_by_4_l676_676116

theorem spinner_three_digit_prob_div_by_4 :
  (({x // x ∈ {1, 2, 4, 8} }) × ({y // y ∈ {1, 2, 4, 8} }) × ({z // z ∈ {1, 2, 4, 8} })).cardinal.to_rat ≠ 0 →
  (∃! p : ℚ, p = 11/16) :=
by
   intro h
   have total := Nat.cast 64 
   have favorable := Nat.cast 44
   have prob := favorable / total
   exists (11/16 : ℚ)
   split
   · exact prob
   · intro q hq
     simp [*] at *
#align spinner_three_digit_prob_div_by_4 spinner_three_digit_prob_div_by_4

end spinner_three_digit_prob_div_by_4_l676_676116


namespace average_of_trees_l676_676767

theorem average_of_trees (numbers : List ℕ) (h : numbers = [10, 8, 9, 9]) : list.average numbers = 9 :=
by
  sorry

end average_of_trees_l676_676767


namespace n_squared_plus_m_squared_odd_implies_n_plus_m_not_even_l676_676697

theorem n_squared_plus_m_squared_odd_implies_n_plus_m_not_even (n m : ℤ) (h : (n^2 + m^2) % 2 = 1) : (n + m) % 2 ≠ 0 := by
  sorry

end n_squared_plus_m_squared_odd_implies_n_plus_m_not_even_l676_676697


namespace num_possible_integers_l676_676280

theorem num_possible_integers (x : ℕ) (h : ⌈Real.sqrt x⌉ = 20) : ∃ n : ℕ, n = 39 :=
by
  have h1 : 19 < Real.sqrt x ∧ Real.sqrt x ≤ 20 := sorry
  have h2 : 361 < x ∧ x ≤ 400 := sorry
  have h3 : ∃ (a b : ℕ), 361 < a ∧ a ≤ 400 ∧ b = a - 361 ∧ b + 1 = 39 := sorry
  use 39
  exact h3.right.right
  sorry

end num_possible_integers_l676_676280


namespace avg_score_boys_combined_l676_676136

variables {C c D d : ℕ}
variables (CHS_boys_avg CHS_girls_avg CHS_combined_avg
           DHS_boys_avg DHS_girls_avg DHS_combined_avg both_schools_avg : ℚ)

-- Given conditions
axiom CHS_boys : CHS_boys_avg = 68
axiom CHS_girls : CHS_girls_avg = 73
axiom CHS_combined : CHS_combined_avg = 70
axiom DHS_boys : DHS_boys_avg = 75
axiom DHS_girls : DHS_girls_avg = 85
axiom DHS_combined : DHS_combined_avg = 80

def avg_boys_combined : ℚ :=
  (68 * C + 75 * D) / (C + D)

theorem avg_score_boys_combined :
  avg_boys_combined = 70.8 :=
by {
  -- CHS average equation
  have CHS_eq : 68 * C + 73 * c = 70 * (C + c) := by sorry,
  -- DHS average equation
  have DHS_eq : 75 * D + 85 * d = 80 * (D + d) := by sorry,
  -- Relating the number of boys and girls
  have C_eq : C = 3 * (c / 2) := by sorry,
  have D_eq : D = d := by sorry,
  
  -- Calculation of average score
  sorry
}

end avg_score_boys_combined_l676_676136


namespace P_linear_P_rank_P_nullity_l676_676387

-- Define the projection operator onto the XY-plane
def P : ℝ^3 → ℝ^3 := λ v, ⟨v.1, v.2, 0⟩

-- 1. Prove linearity
theorem P_linear : linear_map ℝ ℝ^3 ℝ^3 P :=
sorry

-- 2. Matrix representation of P in the basis {i, j, k}
def P_matrix : matrix (fin 3) (fin 3) ℝ :=
  ![
    ![1, 0, 0],
    ![0, 1, 0],
    ![0, 0, 0]
  ]

-- 3. Image of P
def P_image : set ℝ^3 :=
  {v | ∃ α β : ℝ, v = ⟨α, β, 0⟩}

-- 4. Kernel of P
def P_kernel : set ℝ^3 :=
  {v | ∃ γ : ℝ, v = ⟨0, 0, γ⟩}

-- 5. Rank and nullity of P
theorem P_rank : finrank ℝ P_image = 2 :=
sorry

theorem P_nullity : finrank ℝ P_kernel = 1 :=
sorry

end P_linear_P_rank_P_nullity_l676_676387


namespace net_profit_correct_l676_676384

-- Define charges per kilo/item
def regClothesRate : ℝ := 3
def delClothesRate : ℝ := 4
def busClothesRate : ℝ := 5
def bulkyItemRate : ℝ := 6
def discountRateWed : ℝ := 0.10
def overheadCosts : ℝ := 150

-- Define the laundry amounts per day
def day1_regClothes : ℝ := 7
def day1_delClothes : ℝ := 4
def day1_busClothes : ℝ := 3
def day1_bulkyItems : ℝ := 2

def day2_regClothes : ℝ := 10
def day2_delClothes : ℝ := 6
def day2_busClothes : ℝ := 4
def day2_bulkyItems : ℝ := 3

def day3_regClothes : ℝ := 20
def day3_delClothes : ℝ := 4
def day3_busClothes : ℝ := 5
def day3_bulkyItems : ℝ := 2

-- Calculate total earnings for each day
def earnings_day1 : ℝ :=
    (day1_regClothes * regClothesRate) +
    (day1_delClothes * delClothesRate) +
    (day1_busClothes * busClothesRate) +
    (day1_bulkyItems * bulkyItemRate)

def earnings_day2 : ℝ :=
    (day2_regClothes * regClothesRate) +
    (day2_delClothes * delClothesRate) +
    (day2_busClothes * busClothesRate) +
    (day2_bulkyItems * bulkyItemRate)

def earnings_day3 : ℝ :=
    let total := 
        (day3_regClothes * regClothesRate) +
        (day3_delClothes * delClothesRate) +
        (day3_busClothes * busClothesRate) +
        (day3_bulkyItems * bulkyItemRate)
    in total * (1 - discountRateWed)

-- Total earnings and net profit calculations
def totalEarnings : ℝ := earnings_day1 + earnings_day2 + earnings_day3
def netProfit : ℝ := totalEarnings - overheadCosts

theorem net_profit_correct : netProfit = 107.70 := by
  -- Proof should go here
  sorry

end net_profit_correct_l676_676384


namespace log_sum_geometric_sequence_l676_676643

theorem log_sum_geometric_sequence : 
  (∀ n, a n > 0) → (a 8 * a 13 + a 9 * a 12 = 2^6) →
  ∑ i in (Finset.range 20).map (Finset.univ : Finset (Fin 20)), log 2 (a i) = 50 := 
by
  intros _ _
  sorry

end log_sum_geometric_sequence_l676_676643


namespace journey_time_l676_676329

theorem journey_time 
  (d1 d2 T : ℝ)
  (h1 : d1 / 30 + (150 - d1) / 10 = T)
  (h2 : d1 / 30 + d2 / 30 + (150 - (d1 - d2)) / 30 = T)
  (h3 : (d1 - d2) / 10 + (150 - (d1 - d2)) / 30 = T) :
  T = 5 := 
sorry

end journey_time_l676_676329


namespace sum_S_eq_l676_676189

def f (n : ℕ) : ℕ :=
  (n.digits 10).count 0

def n : ℕ := 9999999999

def S (n : ℕ) : ℕ :=
  (Finset.range n.succ).sum (λ k, 2 ^ (f k))

theorem sum_S_eq : S n = (9 * (11 ^ 10 - 1)) / 10 :=
by
  sorry

end sum_S_eq_l676_676189


namespace problem_statement_l676_676977

def has_solutions (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - m * x - 1 = 0

def p : Prop := ∀ m : ℝ, has_solutions m

def q : Prop := ∃ x_0 : ℕ, x_0^2 - 2 * x_0 - 1 ≤ 0

theorem problem_statement : ¬ (p ∧ ¬ q) := 
sorry

end problem_statement_l676_676977


namespace max_true_propositions_max_true_count_l676_676131

theorem max_true_propositions: 
  ∀ (P Q: Prop), (P ↔ ¬Q) → (P ≡ ¬(P ∧ Q))

-- Given the relationships, prove that the maximum number of true propositions is 4.
theorem max_true_count (P Q R S: Prop): 
  (P ↔ R) → (Q ↔ ¬P) → (max (set.count {P, Q, R, S}) ≤ 4) :=
sorry

end max_true_propositions_max_true_count_l676_676131


namespace percentage_increase_first_year_l676_676030

-- Assume the original price of the painting is P and the percentage increase during the first year is X
variable {P : ℝ} (X : ℝ)

-- Condition: The price decreases by 15% during the second year
def condition_decrease (price : ℝ) : ℝ := price * 0.85

-- Condition: The price at the end of the 2-year period was 93.5% of the original price
axiom condition_end_price : ∀ (P : ℝ), (P + (X/100) * P) * 0.85 = 0.935 * P

-- Proof problem: What was the percentage increase during the first year?
theorem percentage_increase_first_year : X = 10 :=
by 
  sorry

end percentage_increase_first_year_l676_676030


namespace prism_surface_area_of_sphere_l676_676532

theorem prism_surface_area_of_sphere (a b c : ℝ) (h1 : 2 * (a * b + b * c + c * a) = 88) (h2 : a + b + c = 12) :
  4 * π * (√(a^2 + b^2 + c^2))^2 = 56 * π := 
sorry

end prism_surface_area_of_sphere_l676_676532


namespace price_rollback_is_correct_l676_676025

-- Define the conditions
def liters_today : ℕ := 10
def cost_per_liter_today : ℝ := 1.4
def liters_friday : ℕ := 25
def total_liters : ℕ := 35
def total_cost : ℝ := 39

-- Define the price rollback calculation
noncomputable def price_rollback : ℝ :=
  (cost_per_liter_today - (total_cost - (liters_today * cost_per_liter_today)) / liters_friday)

-- The theorem stating the rollback per liter is $0.4
theorem price_rollback_is_correct : price_rollback = 0.4 := by
  sorry

end price_rollback_is_correct_l676_676025


namespace particle_position_after_72_moves_l676_676527

def ω : Complex := Complex.exp (Complex.I * Real.pi / 6)
def move (z : Complex) : Complex := ω * z + 12

def initial_position : Complex := 6

def position_after_n_moves(n : Nat) : Complex :=
  Nat.fold move initial_position n

theorem particle_position_after_72_moves : position_after_n_moves 72 = 6 :=
by sorry

end particle_position_after_72_moves_l676_676527


namespace cosine_angle_l676_676259

open Real -- Opening the Real number namespace

variables (c d : EuclideanSpace ℝ (Fin 3)) -- Assumption: Working in 3-dimensional Euclidean space

/-- Define the norms of c and d, and the vector sum norm condition --/
variables (h1 : ∥c∥ = 5) (h2 : ∥d∥ = 7) (h3 : ∥c + d∥ = 10)

-- Define the cosine of the angle between two vectors
noncomputable def cos_phi (c d : EuclideanSpace ℝ (Fin 3)) : ℝ := 
  inner c d / ((∥c∥) * (∥d∥))

/-- The main theorem stating that the cosine of the angle φ is 13/35 given the conditions --/
theorem cosine_angle (h1 : ∥c∥ = 5) (h2 : ∥d∥ = 7) (h3 : ∥c + d∥ = 10) : 
  cos_phi c d = 13 / 35 :=
sorry

end cosine_angle_l676_676259


namespace intersection_complement_B_C_subset_A_range_a_l676_676653

-- Define sets A, B, and C
def A : set ℝ := {x : ℝ | -3 < x ∧ x < 4}

noncomputable def B : set ℝ := {x : ℝ | x^2 + 2 * x - 8 > 0}

noncomputable def C (a : ℝ) (h : a ≠ 0) : set ℝ := {x : ℝ | x^2 - 4 * a * x + 3 * a^2 < 0}

-- Define complement of B in ℝ
noncomputable def not_B : set ℝ := {x : ℝ | -4 ≤ x ∧ x ≤ 2}

-- Question (I): Prove the intersection
theorem intersection_complement_B :
  A ∩ not_B = {x : ℝ | -3 < x ∧ x ≤ 2} := sorry

-- Question (II): Prove the range of a
theorem C_subset_A_range_a (a : ℝ) (h : a ≠ 0) :
  C a h ⊆ A ↔ (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a ≤ 4/3) := sorry

end intersection_complement_B_C_subset_A_range_a_l676_676653


namespace daisy_milk_cow_problem_l676_676158

theorem daisy_milk_cow_problem (M : ℝ) 
  (kid_consumed : 0.75 * M) 
  (cooking_use : 0.50 * (M - 0.75 * M))
  (remaining_milk : (M - 0.75 * M) - 0.50 * (M - 0.75 * M) = 2) : M = 16 := 
sorry

end daisy_milk_cow_problem_l676_676158


namespace probability_abc_216_l676_676847

theorem probability_abc_216 :
  let probability_of_six := 1 / 6
  in
  ((probability_of_six) ^ 3) = (1 / 216) :=
by
  sorry

end probability_abc_216_l676_676847


namespace math_problem_l676_676999

noncomputable def triangle_A_measure (a b c : ℝ) (A B C : ℝ) : Prop :=
  (b^2 + c^2 = a^2 + b * c) → A = π / 3

noncomputable def function_f_max_value (A : ℝ) : Prop :=
  (∀ x : ℝ, ∃ M : ℝ, M = 1 ∧ ∀ y : ℝ, (∃ (f : ℝ → ℝ), f y = sin (y - A) + sqrt 3 * cos y) → f x ≤ M)

theorem math_problem (a b c A B C x : ℝ) :
  (b^2 + c^2 = a^2 + b * c) →
  (triangle_A_measure a b c A B C) ∧ (function_f_max_value (π / 3)) :=
by
  sorry

end math_problem_l676_676999


namespace probability_x_lt_2y_l676_676529

/-- 
Statement of the problem: 
A point (x, y) is randomly selected from inside the rectangle with vertices (0, 0), (6, 0), (6, 2), and (0, 2). 
Prove that the probability that x < 2y is 1/3.
-/
theorem probability_x_lt_2y :
  let A := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 }
  let B := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 ∧ p.1 < 2 * p.2 }
  (measure_theory.measure_space.volume B / measure_theory.measure_space.volume A = (1 : ℝ) / 3) :=
sorry

end probability_x_lt_2y_l676_676529


namespace solve_equation_l676_676450

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end solve_equation_l676_676450


namespace product_of_integers_P_Q_R_S_l676_676194

theorem product_of_integers_P_Q_R_S (P Q R S : ℤ)
  (h1 : 0 < P) (h2 : 0 < Q) (h3 : 0 < R) (h4 : 0 < S)
  (h_sum : P + Q + R + S = 50)
  (h_rel : P + 4 = Q - 4 ∧ P + 4 = R * 3 ∧ P + 4 = S / 3) :
  P * Q * R * S = 43 * 107 * 75 * 225 / 1536 := 
by { sorry }

end product_of_integers_P_Q_R_S_l676_676194


namespace trains_crossing_time_l676_676493

noncomputable def relative_speed_km_per_hr (S1 S2 : ℕ) : ℕ := S1 + S2

noncomputable def relative_speed_m_per_s (rel_speed : ℕ) : ℝ :=
  (rel_speed * 1000) / 3600.0

noncomputable def total_distance (L1 L2 : ℕ) : ℕ := L1 + L2

noncomputable def crossing_time (L1 L2 S1 S2 : ℕ) : ℝ :=
  let rel_speed := relative_speed_km_per_hr S1 S2
  let rel_speed_m := relative_speed_m_per_s rel_speed
  (total_distance L1 L2) / rel_speed_m

theorem trains_crossing_time : crossing_time 140 190 60 40 ≈ 11.88 := by
  sorry

end trains_crossing_time_l676_676493


namespace cos_seven_theta_l676_676684

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end cos_seven_theta_l676_676684


namespace matrix_calc_l676_676915

variable {d b a c : ℝ}

def matA : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [0, d, -b],
    [-d, 0, a],
    [b, -a, 0]
  ]

def matB : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 2]
  ]

def matC : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [d^2, d * b, d * c],
    [d * b, b^2, b * c],
    [d * c, b * c, c^2]
  ]

def matAnswer : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [2 * d^2 - d * b * c, d * b + d * b^2 - c^2 * b, d * c + d * b * c - b * c^2],
    [-d^2 + 2 * d * b + a * d * c, -d^2 * b + 2 * b^2 + a * b * c, -d^2 * c + 2 * b * c + a * c^2],
    [b * d^2 - a * d * b + 2 * d * c, b^2 * d - a * b^2 + 2 * b * c, b * d * c - a * b * c + 2 * c^2]
  ]

theorem matrix_calc : (matA + matB) ⬝ matC = matAnswer := by
  sorry

end matrix_calc_l676_676915


namespace part_a_part_b_l676_676785

theorem part_a (n : ℕ) (hn : n = 17) (buds : Fin n → ℕ) :
  ∃ i : Fin n, (buds i + buds (i + 1)) % 2 = 0 :=
by sorry

theorem part_b (n : ℕ) (hn : n = 17) (buds : Fin n → ℕ) :
  ¬(∀ i : Fin n, (buds i + buds (i + 1)) % 3 = 0) :=
by sorry

end part_a_part_b_l676_676785


namespace num_possible_x_l676_676272

theorem num_possible_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 20) : {y : ℕ | 361 < y ∧ y ≤ 400}.card = 39 :=
by
  sorry

end num_possible_x_l676_676272


namespace base_ten_to_base_three_l676_676061

def base_three_representation (n : ℕ) : string :=
  -- Function to convert number to base 3 string
  sorry -- This would be an actual implementation of conversion to base 3

theorem base_ten_to_base_three :
  base_three_representation 172 = "20101" :=
by
  sorry -- Proof of the equivalence

end base_ten_to_base_three_l676_676061


namespace sum_of_digits_10pow91_plus_100_l676_676472

theorem sum_of_digits_10pow91_plus_100 : 
  ∑ d in (10^91 + 100).digits, d = 2 := 
sorry

end sum_of_digits_10pow91_plus_100_l676_676472


namespace problem_I_monotonicity_and_extremum_problem_II_range_of_k_l676_676647

noncomputable def f (x k : ℝ) : ℝ := 2 * Real.log x - (x - 1)^2 - 2 * k * (x - 1)

theorem problem_I_monotonicity_and_extremum : 
  (∀ x : ℝ, x > 0 → ∃ (I : Set ℝ) (m : ℝ), x ∈ I ∧ 
    (k = 1 → 
      (∀ y ∈ I, (0 < y ∧ y < 1 → f y 1 < 0 ∧ y = x) 
      ∧ (1 < y → f y 1 > 0 ∧ y = x))) 
  ∧ f 1 1 = 0 ∧  ∃! x, (f x 1 = 0 ∧ x = 1)) :=
sorry

theorem problem_II_range_of_k :
  {k : ℝ | ∃ x0 > 1, ∀ x : ℝ, 1 < x ∧ x < x0 → f x k > 0} = set_of (λ k, k < 1) :=
sorry

end problem_I_monotonicity_and_extremum_problem_II_range_of_k_l676_676647


namespace first_class_rate_is_correct_event_b_prob_is_correct_l676_676873

open Finset

def product_data : List (ℕ × ℕ × ℕ) := 
    [(1, 1, 2), (2, 1, 1), (2, 2, 2), (1, 1, 1), (1, 2, 1), 
     (1, 2, 2), (2, 1, 1), (2, 2, 1), (1, 1, 1), (2, 1, 2)]

def indicator_sum (p : ℕ × ℕ × ℕ) : ℕ := p.1 + p.2 + p.3

def eligible_products : List (ℕ × ℕ × ℕ) :=
    product_data.filter (λ p => indicator_sum p ≤ 4)

def first_class_rate : ℚ := (eligible_products.length : ℚ) / (product_data.length : ℚ)

theorem first_class_rate_is_correct : first_class_rate = 0.6 := 
by 
  rw [first_class_rate, product_data.length, eligible_products.length] 
  norm_num 
  sorry

example : Finset (ℕ × ℕ) := (eligible_products.map (λ p => p.1)).to_finset.powerset

def count_event_b : Finset (ℕ × ℕ) := 
    (eligible_products.to_finset).powerset.filter 
    (λ s => s.card = 2 ∧ ∀ p ∈ s, indicator_sum p = 4)

def event_b_prob : ℚ := 
    (count_event_b.card : ℚ) / 
    (eligible_products.to_finset.powerset.filter (λ s => s.card = 2)).card

theorem event_b_prob_is_correct : event_b_prob = 2 / 5 := 
by 
  rw [event_b_prob, count_event_b.card] 
  norm_num 
  sorry

end first_class_rate_is_correct_event_b_prob_is_correct_l676_676873


namespace remaining_volume_correct_l676_676408

-- Define the given conditions
def side_length : ℝ := 4
def radius : ℝ := 2
def height : ℝ := side_length

-- Define the volumes
def volume_cube := side_length ^ 3
def volume_cylinder := Real.pi * radius^2 * height
def remaining_volume := volume_cube - volume_cylinder

-- State the theorem
theorem remaining_volume_correct : remaining_volume = 64 - 16 * Real.pi := by
  sorry

end remaining_volume_correct_l676_676408


namespace total_dangerous_animals_l676_676560

theorem total_dangerous_animals
  (num_crocodiles : ℕ) (num_alligators : ℕ) (num_vipers : ℕ) :
  num_crocodiles = 22 →
  num_alligators = 23 →
  num_vipers = 5 →
  num_crocodiles + num_alligators + num_vipers = 50 :=
by {
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end total_dangerous_animals_l676_676560


namespace min_sum_arithmetic_seq_l676_676225

theorem min_sum_arithmetic_seq {a : ℕ → ℤ} (d a_1 a_2 : ℤ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 = -9)
  (h3 : 3 * (2 * a 1 + 2 * d) = 7 * (2 * a 1 + 6 * d)) :
  ∃ n, S n = a 1 * n ∧ (n = 5 ∨ n = other valid option) := 
by
  -- Define S_n
  let S := λ n, n * a 1
  -- sorry is added here to skip the proof
  sorry

end min_sum_arithmetic_seq_l676_676225


namespace die_vanishing_probability_and_floor_value_l676_676800

/-
Given conditions:
1. The die has four faces labeled 0, 1, 2, 3.
2. When the die lands on a face labeled:
   - 0: the die vanishes.
   - 1: nothing happens (one die remains).
   - 2: the die replicates into 2 dice.
   - 3: the die replicates into 3 dice.
3. All dice (original and replicas) will continuously be rolled.
Prove:
  The value of ⌊10/p⌋ is 24, where p is the probability that all dice will eventually disappear.
-/

theorem die_vanishing_probability_and_floor_value : 
  ∃ (p : ℝ), 
  (p^3 + p^2 - 3 * p + 1 = 0 ∧ 0 ≤ p ∧ p ≤ 1 ∧ p = Real.sqrt 2 - 1) 
  ∧ ⌊10 / p⌋ = 24 := 
    sorry

end die_vanishing_probability_and_floor_value_l676_676800


namespace sum_of_positive_integers_satisfying_inequality_l676_676613

theorem sum_of_positive_integers_satisfying_inequality :
  (∑ n in Finset.filter (λ n: ℕ, 1.5 * n - 6.3 < 7.2) (Finset.range 9)) = 36 :=
by
  sorry

end sum_of_positive_integers_satisfying_inequality_l676_676613


namespace diophantus_age_l676_676588

theorem diophantus_age :
  ∃ (x : ℕ),
    (x > 0) ∧
    (x / 6 + x / 12 + x / 7 + 5 + x / 2 + 4 = x) ∧
    (x = 84) :=
begin
  use 84,
  split,
  { norm_num, },
  split,
  { norm_num,
    linarith, },
  refl,
end

end diophantus_age_l676_676588


namespace solve_equation_l676_676398

theorem solve_equation :
  let x := (140 * Real.sqrt 2 - 140) / 9 in
  216 + Real.sqrt 41472 - 18 * x - Real.sqrt (648 * x^2) = 0 :=
by
  let x := (140 * Real.sqrt 2 - 140) / 9
  have : (18 * Real.sqrt 2 * x) = Real.sqrt (648 * x^2) := sorry
  rw [this]
  exact sorry

end solve_equation_l676_676398


namespace soaking_time_l676_676140

theorem soaking_time (time_per_grass_stain : ℕ) (time_per_marinara_stain : ℕ) 
    (number_of_grass_stains : ℕ) (number_of_marinara_stains : ℕ) : 
    time_per_grass_stain = 4 ∧ time_per_marinara_stain = 7 ∧ 
    number_of_grass_stains = 3 ∧ number_of_marinara_stains = 1 →
    (time_per_grass_stain * number_of_grass_stains + time_per_marinara_stain * number_of_marinara_stains) = 19 :=
by
  sorry

end soaking_time_l676_676140


namespace distance_between_points_l676_676939

theorem distance_between_points :
  ∀ (P Q : ℝ × ℝ), P = (3, 3) ∧ Q = (-2, -2) → dist P Q = 5 * real.sqrt 2 :=
begin
  sorry
end

end distance_between_points_l676_676939


namespace prince_cd_total_spent_l676_676561

theorem prince_cd_total_spent (total_cds : ℕ)
    (pct_20 : ℕ) (pct_15 : ℕ) (pct_10 : ℕ)
    (bought_20_pct : ℕ) (bought_15_pct : ℕ)
    (bought_10_pct : ℕ) (bought_6_pct : ℕ)
    (discount_cnt_4 : ℕ) (discount_amount_4 : ℕ)
    (discount_cnt_5 : ℕ) (discount_amount_5 : ℕ)
    (total_cost_no_discount : ℕ) (total_discount : ℕ) (total_spent : ℕ) :
    total_cds = 400 ∧
    pct_20 = 25 ∧ pct_15 = 30 ∧ pct_10 = 20 ∧
    bought_20_pct = 70 ∧ bought_15_pct = 40 ∧
    bought_10_pct = 80 ∧ bought_6_pct = 100 ∧
    discount_cnt_4 = 4 ∧ discount_amount_4 = 5 ∧
    discount_cnt_5 = 5 ∧ discount_amount_5 = 3 ∧
    total_cost_no_discount - total_discount = total_spent ∧
    total_spent = 3119 := by
  sorry

end prince_cd_total_spent_l676_676561


namespace james_total_carrot_sticks_l676_676734

theorem james_total_carrot_sticks : 
  ∀ (before_dinner after_dinner : Nat), before_dinner = 22 → after_dinner = 15 → 
  before_dinner + after_dinner = 37 := 
by
  intros before_dinner after_dinner h1 h2
  rw [h1, h2]
  rfl

end james_total_carrot_sticks_l676_676734


namespace sum_seq_eq_zero_l676_676986

theorem sum_seq_eq_zero {a : ℕ → ℂ} (n : ℕ)
  (h₀ : ∀ i, (a (i + 1)) ^ 2 - a (i + 1) * a i + (a i) ^ 2 = 0)
  (h₁ : ∀ i, 1 ≤ i → i ≤ n → a (i + 1) ≠ a (i - 1))
  (h₂ : a 1 = 1)
  (h₃ : a (n + 1) = 1) :
  ∑ i in finset.range n, a i = 0 :=
sorry

end sum_seq_eq_zero_l676_676986


namespace no_such_natural_number_exists_l676_676590

theorem no_such_natural_number_exists :
  ¬ ∃ (n s : ℕ), n = 2014 * s + 2014 ∧ n % s = 2014 ∧ (n / s) = 2014 :=
by
  sorry

end no_such_natural_number_exists_l676_676590


namespace factorial_fraction_evaluation_l676_676065

theorem factorial_fraction_evaluation :
  (10! * 6! * 3!) / (9! * 7! * 2!) = 30 / 7 :=
by
  sorry

end factorial_fraction_evaluation_l676_676065


namespace probability_of_collinear_dots_l676_676322

theorem probability_of_collinear_dots (dots : ℕ) (rows : ℕ) (columns : ℕ) (choose : ℕ → ℕ → ℕ) :
  dots = 20 ∧ rows = 5 ∧ columns = 4 ∧ choose 20 4 = 4845 → 
  (∃ sets_of_collinear_dots : ℕ, sets_of_collinear_dots = 20 ∧ 
   ∃ probability : ℚ,  probability = 4 / 969) :=
by
  sorry

end probability_of_collinear_dots_l676_676322


namespace rectangle_area_in_ellipse_l676_676890

theorem rectangle_area_in_ellipse :
  ∃ a b : ℝ, 2 * a = b ∧ (a^2 / 4 + b^2 / 8 = 1) ∧ 2 * a * b = 16 :=
by
  sorry

end rectangle_area_in_ellipse_l676_676890


namespace alcohol_solution_contradiction_l676_676265

theorem alcohol_solution_contradiction (initial_volume : ℕ) (added_water : ℕ) 
                                        (final_volume : ℕ) (final_concentration : ℕ) 
                                        (initial_concentration : ℕ) : 
                                        initial_volume = 75 → added_water = 50 → 
                                        final_volume = initial_volume + added_water → 
                                        final_concentration = 45 → 
                                        ¬ (initial_concentration * initial_volume = final_concentration * final_volume) :=
by 
  intro h_initial_volume h_added_water h_final_volume h_final_concentration
  sorry

end alcohol_solution_contradiction_l676_676265


namespace john_makes_l676_676337

theorem john_makes (initial_cost : ℝ) (discount : ℝ) (prize : ℝ) (keep_percentage : ℝ) (x : ℝ) :
  initial_cost = 20000 →
  discount = 0.20 →
  prize = 70000 →
  keep_percentage = 0.90 →
  (keep_percentage * prize - (initial_cost - discount * initial_cost) - x) = (47000 - x) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end john_makes_l676_676337


namespace expenditure_increase_l676_676414

theorem expenditure_increase
  (current_expenditure : ℝ)
  (future_expenditure : ℝ)
  (years : ℕ)
  (r : ℝ)
  (h₁ : current_expenditure = 1000)
  (h₂ : future_expenditure = 2197)
  (h₃ : years = 3)
  (h₄ : future_expenditure = current_expenditure * (1 + r / 100) ^ years) :
  r = 30 :=
sorry

end expenditure_increase_l676_676414


namespace modulus_subtraction_l676_676751

open Complex

-- Define the complex number z
def z : ℂ := 1 + I

-- Define the conjugate of z
def z_conj : ℂ := conj z

-- State the theorem to be proved
theorem modulus_subtraction (z : ℂ) (hz : z = 1 + I) : |conj z - 3| = Real.sqrt 5 := by
  sorry

end modulus_subtraction_l676_676751


namespace average_disk_space_per_minute_l676_676878

theorem average_disk_space_per_minute
  (days : ℕ)
  (total_disk_space : ℕ)
  (total_minutes : ℕ := days * 24 * 60)
  (average_disk_space : ℕ := (total_disk_space : ℚ) / total_minutes) :
  ((15 : ℕ) = days) →
  ((24000 : ℕ) = total_disk_space) →
  average_disk_space ≈ (1 : ℚ) := sorry

end average_disk_space_per_minute_l676_676878


namespace six_digit_square_number_cases_l676_676535

theorem six_digit_square_number_cases :
  ∃ n : ℕ, 316 ≤ n ∧ n < 1000 ∧ (n^2 = 232324 ∨ n^2 = 595984 ∨ n^2 = 929296) :=
by {
  sorry
}

end six_digit_square_number_cases_l676_676535


namespace each_person_gets_4_roses_l676_676781

def ricky_roses_total : Nat := 40
def roses_stolen : Nat := 4
def people : Nat := 9
def remaining_roses : Nat := ricky_roses_total - roses_stolen
def roses_per_person : Nat := remaining_roses / people

theorem each_person_gets_4_roses : roses_per_person = 4 := by
  sorry

end each_person_gets_4_roses_l676_676781


namespace diameter_of_outer_circle_l676_676879

theorem diameter_of_outer_circle (D d : ℝ) 
  (h1 : d = 24) 
  (h2 : π * (D / 2) ^ 2 - π * (d / 2) ^ 2 = 0.36 * π * (D / 2) ^ 2) : D = 30 := 
by 
  sorry

end diameter_of_outer_circle_l676_676879


namespace quadratic_root_m_l676_676705

theorem quadratic_root_m (m : ℝ) : (∃ x : ℝ, x^2 + x + m^2 - 1 = 0 ∧ x = 0) → (m = 1 ∨ m = -1) :=
by 
  sorry

end quadratic_root_m_l676_676705


namespace general_formula_max_sum_value_l676_676627

open BigOperators

def a : ℕ → ℤ
| 0       := 23
| 1       := -9
| (n + 2) := a n + 6 * (-1) ^ (n + 1) - 2

def S (n : ℕ) : ℤ := ∑ i in (Finset.range n).image Nat.succ, a i

theorem general_formula (n : ℕ) :
  (∀ n, nat.odd n → a n = 2 * n + 21) ∧
  (∀ n, nat.even n → a n = -4 * n - 1) :=
by sorry

theorem max_sum_value (n k : ℕ) :
  S 11 = 73 ∧ S n ≤ 73 :=
by sorry

end general_formula_max_sum_value_l676_676627


namespace number_of_teachers_students_possible_rental_plans_economical_plan_l676_676096

-- Definitions of conditions

def condition1 (x y : ℕ) : Prop := y - 30 * x = 7
def condition2 (x y : ℕ) : Prop := 31 * x - y = 1
def capacity_condition (m : ℕ) : Prop := 35 * m + 30 * (8 - m) ≥ 255
def rental_fee_condition (m : ℕ) : Prop := 400 * m + 320 * (8 - m) ≤ 3000

-- Problems to prove

theorem number_of_teachers_students (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 8 ∧ y = 247 := 
by sorry

theorem possible_rental_plans (m : ℕ) (h_cap : capacity_condition m) (h_fee : rental_fee_condition m) : m = 3 ∨ m = 4 ∨ m = 5 := 
by sorry

theorem economical_plan (m : ℕ) (h_fee : rental_fee_condition 3) (h_fee_alt1 : rental_fee_condition 4) (h_fee_alt2 : rental_fee_condition 5) : m = 3 := 
by sorry

end number_of_teachers_students_possible_rental_plans_economical_plan_l676_676096


namespace circle_area_l676_676774

noncomputable def point := (ℝ × ℝ)

def diameter (C D : point) := real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)

def radius (C D : point) := diameter C D / 2

noncomputable def area (C D : point) := real.pi * (radius C D)^2

theorem circle_area (C D : point) (hC: C = (-2, 3)) (hD: D = (2, -1)) :
  area C D = 8 * real.pi :=
by
  sorry

end circle_area_l676_676774


namespace prove_a_pow_minus_b_l676_676421

-- Definitions of conditions
variables (x a b : ℝ)

def condition_1 : Prop := x - a > 2
def condition_2 : Prop := 2 * x - b < 0
def solution_set_condition : Prop := -1 < x ∧ x < 1
def derived_a : Prop := a + 2 = -1
def derived_b : Prop := b / 2 = 1

-- The main theorem to prove
theorem prove_a_pow_minus_b (h1 : condition_1 x a) (h2 : condition_2 x b) (h3 : solution_set_condition x) (ha : derived_a a) (hb : derived_b b) : a^(-b) = (1 / 9) :=
by
  sorry

end prove_a_pow_minus_b_l676_676421


namespace Jesse_remaining_money_l676_676331

-- Define the conditions
def initial_money := 50
def novel_cost := 7
def lunch_cost := 2 * novel_cost
def total_spent := novel_cost + lunch_cost

-- Define the remaining money after spending
def remaining_money := initial_money - total_spent

-- Prove that the remaining money is $29
theorem Jesse_remaining_money : remaining_money = 29 := 
by
  sorry

end Jesse_remaining_money_l676_676331


namespace ratio_PA_AB_l676_676708

theorem ratio_PA_AB (A B C P : Type) [linear_ordered_field ℝ] 
  (triangle_ABC : triangle A B C) (ratio_AC_CB : AC/CB = 1/2)
  (bisector_ext_angle_C : bisector (exterior_angle C) B P) (between_A_B_P : between A B P) :
  ratio PA AB = 2 :=
  sorry

end ratio_PA_AB_l676_676708


namespace donut_holes_covered_by_lidia_l676_676562

noncomputable def surface_area (r: ℕ) : ℕ := 4 * Nat.pi * (r * r)

def radius_lidia := 5
def radius_marco := 7
def radius_priya := 9

def surface_area_lidia := surface_area radius_lidia
def surface_area_marco := surface_area radius_marco
def surface_area_priya := surface_area radius_priya

open Nat

theorem donut_holes_covered_by_lidia :
  let lcm_area := Nat.lcm (Nat.lcm surface_area_lidia surface_area_marco) surface_area_priya
  ∃ n : ℕ, lcm_area / surface_area_lidia = n ∧ n = 3528 :=
  by
    sorry

end donut_holes_covered_by_lidia_l676_676562


namespace distance_from_point_to_line_l676_676642

noncomputable def slope_line1 : ℝ := -2
noncomputable def slope_line2 (m : ℝ) : ℝ := -1 / m

noncomputable def are_perpendicular (m : ℝ) : Prop := slope_line1 * slope_line2 m = -1

theorem distance_from_point_to_line (m : ℝ) (h : are_perpendicular m) :
  let P := (m, m)
  let dist := (λ (x1 y1 A B C : ℝ), (|A * x1 + B * y1 + C| / sqrt (A * A + B * B)))
  dist (-2) (-2) 1 1 3 = real.sqrt 2 / 2 :=
by
  sorry

end distance_from_point_to_line_l676_676642


namespace distance_between_points_l676_676941

theorem distance_between_points :
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  (Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 * Real.sqrt 2) :=
by
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  sorry

end distance_between_points_l676_676941


namespace distance_between_points_l676_676937

theorem distance_between_points :
  ∀ (P Q : ℝ × ℝ), P = (3, 3) ∧ Q = (-2, -2) → dist P Q = 5 * real.sqrt 2 :=
begin
  sorry
end

end distance_between_points_l676_676937


namespace annual_expenditure_l676_676385

theorem annual_expenditure (x y : ℝ) (h1 : y = 0.8 * x + 0.1) (h2 : x = 15) : y = 12.1 :=
by
  sorry

end annual_expenditure_l676_676385


namespace find_multiple_l676_676037

theorem find_multiple (m : ℤ) (h : 38 + m * 43 = 124) : m = 2 := by
    sorry

end find_multiple_l676_676037


namespace find_y_l676_676188

theorem find_y (y: ℕ) (h1: y > 0) (h2: y ≤ 100)
  (h3: (43 + 69 + 87 + y + y) / 5 = 2 * y): 
  y = 25 :=
sorry

end find_y_l676_676188


namespace range_of_perimeter_l676_676996

theorem range_of_perimeter (A B C : ℝ) (a b c : ℝ) (h₁ : a = 1) (h₂ : 2 * Real.cos C + c = 2 * b) (h₃ : A + B + C = π) 
(h₄ : 0 < A) (h₅ : A < π / 2) (h₆ : 0 < B) (h₇ : B < π / 2) (h₈ : 0 < C) (h₉ : C < π / 2) :
∃ p, p = a + b + c ∧ (sqrt 3 + 1 < p ∧ p < 3) :=
by sorry

end range_of_perimeter_l676_676996


namespace intersection_M_N_l676_676219

def set_M : Set ℝ := { x : ℝ | -3 ≤ x ∧ x < 4 }
def set_N : Set ℝ := { x : ℝ | x^2 - 2 * x - 8 ≤ 0 }

theorem intersection_M_N : (set_M ∩ set_N) = { x : ℝ | -2 ≤ x ∧ x < 4 } :=
sorry

end intersection_M_N_l676_676219


namespace correct_operation_is_C_l676_676475

-- Defining the conditions (statements of operations)
def option_a : Prop := sqrt 2 + sqrt 3 = sqrt 5
def option_b : Prop := (-2)⁻¹ = 1 / 2
def option_c : Prop := -a^2 * a^2 = -a^4
def option_d : Prop := (a + b)^2 = a^2 + b^2

-- The statement we are to prove is that among options A, B, C, and D, only option C is correct.
theorem correct_operation_is_C : ¬option_a ∧ ¬option_b ∧ option_c ∧ ¬option_d := by
  sorry

end correct_operation_is_C_l676_676475


namespace polygon_sides_l676_676706

theorem polygon_sides (n : ℕ) : 
  (180 * (n - 2) / 360 = 5 / 2) → n = 7 :=
by
  sorry

end polygon_sides_l676_676706


namespace integrate_inequality_l676_676867

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

noncomputable def has_continuous_derivative (f' : ℝ → ℝ) : Prop := ∀ x ∈ set.Icc 0 1, has_deriv_at f (f' x) x

theorem integrate_inequality
  (H1 : has_continuous_derivative f') 
  (H2 : ∀ x ∈ set.Icc 0 1, 0 < f' x ∧ f' x ≤ 1)
  (H3 : f 0 = 0) :
  (∫ x in 0..1, f x)^2 ≥ ∫ x in 0..1, (f x)^3 := 
sorry

example : (∫ x in 0..1, (x : ℝ))^2 = ∫ x in 0..1, (x : ℝ)^3 := by
  rw [interval_integral.integral_of_le zero_le_one, interval_integral.integral_of_le zero_le_one]
  norm_num

end integrate_inequality_l676_676867


namespace option_C_is_x_plus_1_l676_676070

noncomputable def expression_A (x : ℝ) : ℝ := (x^2 - 1) / x * (x / (x^2 + 1))
noncomputable def expression_B (x : ℝ) : ℝ := 1 - 1 / x
noncomputable def expression_C (x : ℝ) : ℝ := (x^2 + 2x + 1) / (x + 1)
noncomputable def expression_D (x : ℝ) : ℝ := (x + 1) / x / (1 / (x - 1))

theorem option_C_is_x_plus_1 (x : ℝ) (hx1 : x ≠ -1) (hx2 : x ≠ 0) :
  expression_C x = x + 1 :=
by {
  unfold expression_C,
  have hx3 : x^2 + 2 * x + 1 = (x + 1)^2 := by ring,
  rw [hx3, (div_eq_of_eq_mul (x + 1) (ne_of_gt (lt_of_le_of_ne (real.rpow_nonneg_of_nonneg zero_le_two x) (ne.symm hx1))) (ne_of_ne (ne.symm hx2)) (x + 1) (one_mul (x + 1)).symm)],
  rw div_self (ne_of_gt (lt_of_le_of_ne (real.rpow_nonneg_of_nonneg zero_le_two x) (ne.symm hx1))),
}

end option_C_is_x_plus_1_l676_676070


namespace angle_between_vectors_is_pi_over_3_l676_676252

open Real

noncomputable def a : ℝ × ℝ := (1, 0)
noncomputable def b : ℝ × ℝ := (-1/2, sqrt 3 / 2)

-- Function to compute vector addition
def vector_add (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ :=
 (v₁.1 + v₂.1, v₁.2 + v₂.2)

-- Function to compute dot product of two vectors
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
 (v₁.1 * v₂.1) + (v₁.2 * v₂.2)

-- Function to compute the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
 Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem angle_between_vectors_is_pi_over_3 :
  let a := (1, 0) in
  let b := (-1/2, sqrt 3 / 2) in
  let c := vector_add a b in
  let cos_theta := dot_product a c / (magnitude a * magnitude c) in
  acos (Real.sqrt(cos_theta)) = π / 3 :=
sorry

end angle_between_vectors_is_pi_over_3_l676_676252


namespace problem_sequence_a2016_l676_676034

def sequence (a : ℕ → ℝ) : Prop :=
  a 0 = Real.sqrt 3 ∧ ∀ n, a (n + 1) = Real.floor (a n) + 1 / (a n - Real.floor (a n))

theorem problem_sequence_a2016 :
  ∃ a : ℕ → ℝ, sequence a ∧ a 2016 = 3024 + Real.sqrt 3 :=
by {
  sorry
}

end problem_sequence_a2016_l676_676034


namespace equilateral_triangle_OAB_length_l676_676319

noncomputable def C1_polar_equation (θ : ℝ) : ℝ := 2 * real.cos θ
noncomputable def C2_polar_equation (θ : ℝ) : ℝ := -4 * real.cos θ

theorem equilateral_triangle_OAB_length (θ : ℝ) (hθ : 0 < θ ∧ θ < real.pi / 2) :
  let ρA := C1_polar_equation θ in
  let ρB := C2_polar_equation (θ + real.pi / 3) in
  2 * real.cos θ = real.sqrt 3 * real.sin θ →
  ρA = 2 * real.cos θ →
  ρA = real.sqrt 21 / 7 →
  |ρA| = (2 * real.sqrt 21) / 7 :=
sorry

end equilateral_triangle_OAB_length_l676_676319


namespace distance_between_points_l676_676953

noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem distance_between_points :
  dist (3, 3) (-2, -2) = 5 * Real.sqrt 2 := 
by
  sorry

end distance_between_points_l676_676953


namespace min_value_y_l676_676806

theorem min_value_y : ∀ (x : ℝ), (x > 3) → (∃ (y : ℝ), y = (1 / (x - 3) + x) ∧ y ≥ 5) ∧ (∃ (x : ℝ), x = 4 ∧ (1 / (x - 3) + x) = 5) :=
by
  intros x hx
  suffices h : ((1 / (x - 3) + x) = 5 ↔ x = 4), from
  ⟨fun _ => ⟨5, by
    field_simp [ne_of_gt hx, ne_of_gt (sub_pos.mpr hx)]
    suffices 1 / (x - 3) + (x - 3) ≥ 2, from
    by
      rw ← add_assoc
      linarith
    exact calc
      1 / (x - 3) + (x - 3) = 2 : by sorry, h⟩,
      ⟨4, by linear_algebra sorry⟩⟩

end min_value_y_l676_676806


namespace cube_volume_surface_area_l676_676841

theorem cube_volume_surface_area 
    (V1 : ℝ) 
    (hV1 : V1 = 8)
    (A2 : ℝ)
    (hA2 : A2 = 3 * 6 * (2^2))
    : ∃ V2 : ℝ, V2 = (2 * real.sqrt 3)^3 := 
sorry

end cube_volume_surface_area_l676_676841


namespace find_abc_l676_676759

theorem find_abc (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a ≠ b) (h5 : f a = a^3) (h6 : f b = b^3) : 
  a + b + c = 16 :=
by
  let f := λ x : ℝ, x^3 + a * x^2 + b * x + c
  sorry

end find_abc_l676_676759


namespace initial_kids_count_l676_676825

-- Define the initial number of kids as a variable
def init_kids (current_kids join_kids : Nat) : Nat :=
  current_kids - join_kids

-- Define the total current kids and kids joined
def current_kids : Nat := 36
def join_kids : Nat := 22

-- Prove that the initial number of kids was 14
theorem initial_kids_count : init_kids current_kids join_kids = 14 :=
by
  -- Proof skipped
  sorry

end initial_kids_count_l676_676825


namespace marks_fathers_gift_l676_676379

noncomputable def total_spent (books : ℕ) (cost_per_book : ℕ) : ℕ :=
  books * cost_per_book

noncomputable def total_money_given (spent : ℕ) (left_over : ℕ) : ℕ :=
  spent + left_over

theorem marks_fathers_gift :
  total_money_given (total_spent 10 5) 35 = 85 := by
  sorry

end marks_fathers_gift_l676_676379


namespace line_containing_AB_area_of_triangle_ABC_l676_676656

-- Define the vertices of the triangle
def A := (1 : ℝ, 3 : ℝ)
def B := (3 : ℝ, 1 : ℝ)
def C := (-1 : ℝ, 0 : ℝ)

-- Prove the equation of the line containing side AB
theorem line_containing_AB : (∀ (x y : ℝ), x + y - 4 = 0 ↔ (∃ t : ℝ, x = 1 + 2 * t ∧ y = 3 - 2 * t)) :=
by
  sorry

-- Prove the area of triangle ABC is 5
theorem area_of_triangle_ABC : abs ((1/2) * ((fst A * (snd B - snd C)) + (fst B * (snd C - snd A)) + (fst C * (snd A - snd B)))) = 5 :=
by
  sorry

end line_containing_AB_area_of_triangle_ABC_l676_676656


namespace find_cos_7theta_l676_676680

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end find_cos_7theta_l676_676680


namespace solve_equation_l676_676425

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end solve_equation_l676_676425


namespace ratio_after_girls_leave_l676_676508

-- Define the initial conditions
def initial_conditions (B G : ℕ) : Prop :=
  B = G ∧ B + G = 32

-- Define the event of girls leaving
def girls_leave (G : ℕ) : ℕ :=
  G - 8

-- Define the final ratio of boys to girls
def final_ratio (B G : ℕ) : ℕ :=
  B / (girls_leave G)

-- Prove the final ratio is 2:1
theorem ratio_after_girls_leave (B G : ℕ) (h : initial_conditions B G) :
  final_ratio B G = 2 :=
by
  sorry

end ratio_after_girls_leave_l676_676508


namespace least_value_sum_log_l676_676296

theorem least_value_sum_log (a b : ℝ) (h1 : log 3 a + log 3 b ≥ 5) (h2 : a ≥ b) :
  a + b ≥ 18 * real.sqrt 3 :=
sorry

end least_value_sum_log_l676_676296


namespace first_player_can_score_at_least_55_l676_676086

-- Define the sequence of numbers and the concept of "striking out" numbers
def sequence : List ℕ := List.range' 1 101

-- Define the game rules
structure Game :=
(players : ℕ)
(turns : ℕ)
(strikes : ℕ)
(sequence : List ℕ)
(score : ℕ)

-- Define the problem conditions
def problem_conditions : Prop :=
  ∀ (s : List ℕ), s.length = 2 → (∃ x y : ℕ, x, y ∈ s ∧ abs (x - y) ≥ 55)

-- Define the main theorem
theorem first_player_can_score_at_least_55 : problem_conditions :=
by
  -- Placeholder for the actual proof
  sorry

end first_player_can_score_at_least_55_l676_676086


namespace weight_of_new_person_l676_676492

theorem weight_of_new_person {avg_increase : ℝ} (n : ℕ) (p : ℝ) (w : ℝ) (h : n = 8) (h1 : avg_increase = 2.5) (h2 : w = 67):
  p = 87 :=
by
  sorry

end weight_of_new_person_l676_676492


namespace butterflies_count_l676_676821

theorem butterflies_count (total_black_dots : ℕ) (black_dots_per_butterfly : ℕ) 
                          (h1 : total_black_dots = 4764) 
                          (h2 : black_dots_per_butterfly = 12) :
                          total_black_dots / black_dots_per_butterfly = 397 :=
by
  sorry

end butterflies_count_l676_676821


namespace union_complement_B_l676_676203

def U := Set ℝ
def A := {x : ℝ | x^2 ≥ 9}
def B := {x : ℝ | -1 < x ∧ x ≤ 7}
def compl_U_B := {x : ℝ | x ≤ -1 ∨ x > 7}

theorem union_complement_B :
  A ∪ compl_U_B = {x : ℝ | x ≥ 3 ∨ x ≤ -1} := by
sory

end union_complement_B_l676_676203


namespace opposite_sqrt5_minus_2_abs_sqrt2_minus_3_l676_676028

-- Definition of the opposite of an expression
def opposite (a b : ℝ) : ℝ := -1 * (a - b)

-- Definition of the absolute value given a condition
def abs_of_less (a b : ℝ) (h : a < b) : ℝ := -(a - b)

-- Statements to be proved
theorem opposite_sqrt5_minus_2 : opposite (√5) 2 = 2 - √5 := sorry

theorem abs_sqrt2_minus_3 : abs_of_less (√2) 3 (by norm_num) = 3 - √2 := sorry

end opposite_sqrt5_minus_2_abs_sqrt2_minus_3_l676_676028


namespace ratio_SRNM_to_ABC_valid_l676_676325

open Lean.Meta

noncomputable def ratio_SRNM_to_ABC (A B C M N X Y S R : Point) [is_triangle A B C] :=
  (M ∈ Trisection AC) ∧ 
  (N ∈ Trisection AC) ∧ 
  (X ∈ Trisection BC) ∧ 
  (Y ∈ Trisection BC) ∧ 
  (S ∈ LineIntersect AY) ∧ 
  (S ∈ Line BM) ∧ 
  (R ∈ LineIntersect BN) ∧ 
  (R ∈ Line BN) →
  ratio (Area (Quad SRNM)) (Area (Tri ABC)) = 0.06

axioms
  (A B C M N X Y S R : Point)
  (ht: is_triangle A B C)
  (h1 : M ∈ Trisection (LineSegment A C))
  (h2 : N ∈ Trisection (LineSegment A C))
  (h3 : X ∈ Trisection (LineSegment B C))
  (h4 : Y ∈ Trisection (LineSegment B C))
  (h5 : S ∈ LineIntersect (LineSegment A Y))
  (h6 : S ∈ Line (LineSegment B M))
  (h7 : R ∈ LineIntersect (LineSegment B N))
  (h8 : R ∈ Line (LineSegment B N))

theorem ratio_SRNM_to_ABC_valid:
    ∀ (A B C M N X Y S R : Point) [is_triangle A B C], 
      (M ∈ Trisection (LineSegment A C)) → 
      (N ∈ Trisection (LineSegment A C)) → 
      (X ∈ Trisection (LineSegment B C)) → 
      (Y ∈ Trisection (LineSegment B C)) → 
      (S ∈ LineIntersect (LineSegment A Y)) → 
      (S ∈ Line (LineSegment B M)) → 
      (R ∈ LineIntersect (LineSegment B N)) → 
      (R ∈ Line (LineSegment B N)) → 
      (ratio (Area (Quad SRNM)) (Area (Tri ABC)) = 0.06) :=
        by
    sorry

end ratio_SRNM_to_ABC_valid_l676_676325


namespace cartesian_from_polar_length_intersection_AB_l676_676566

-- Define the polar equation as a function in Lean
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.cos θ

-- Define the Cartesian equation as a function in Lean
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 8 * x

-- Define the parametric equation of the line in Lean
def param_line (t x y : ℝ) : Prop :=
  x = t + 2 ∧ y = t

-- Prove the Cartesian equation given the polar equation
theorem cartesian_from_polar (x y θ : ℝ) (h : polar_equation (Real.sqrt (x^2 + y^2)) θ) : cartesian_equation x y :=
by sorry

-- Prove the length of the intersection points AB
theorem length_intersection_AB (x y t : ℝ) (h_line : param_line t x y) (h_curve : cartesian_equation x y) : 
  2 * Real.sqrt 14 =
  by sorry

end cartesian_from_polar_length_intersection_AB_l676_676566


namespace soaking_time_l676_676141

theorem soaking_time (time_per_grass_stain : ℕ) (time_per_marinara_stain : ℕ) 
    (number_of_grass_stains : ℕ) (number_of_marinara_stains : ℕ) : 
    time_per_grass_stain = 4 ∧ time_per_marinara_stain = 7 ∧ 
    number_of_grass_stains = 3 ∧ number_of_marinara_stains = 1 →
    (time_per_grass_stain * number_of_grass_stains + time_per_marinara_stain * number_of_marinara_stains) = 19 :=
by
  sorry

end soaking_time_l676_676141


namespace minimum_average_label_l676_676410

theorem minimum_average_label {K : Type*} [fintype K] (hV : fintype.card K = 2017) 
  (label : (K → K → ℕ)) 
  (label_range : ∀ (x y : K), label x y ∈ {1, 2, 3})
  (triangle_condition : ∀ (x y z : K), x ≠ y → y ≠ z → x ≠ z → 
    label x y + label y z + label z x ≥ 5) : 
  ∃ (avg : ℚ), avg = (4033 : ℚ) / 2017 := 
sorry

end minimum_average_label_l676_676410


namespace Allen_age_difference_l676_676895

theorem Allen_age_difference (M A : ℕ) (h1 : M = 30) (h2 : (A + 3) + (M + 3) = 41) : M - A = 25 :=
by
  sorry

end Allen_age_difference_l676_676895


namespace problem1_l676_676087

theorem problem1 (x : ℝ) : abs (2 * x - 3) < 1 ↔ 1 < x ∧ x < 2 := sorry

end problem1_l676_676087


namespace median_after_adding_l676_676099

theorem median_after_adding (mean_mode_median : ∃ s : Multiset ℕ, (card s = 5) ∧ (s.sum = 28) ∧ (s.nodup = false) ∧ (s.mode = {4}) ∧ (s.median = 5) :
  let s' := (s + [9, 11])
  (s'.median = 6) :=
sorry

end median_after_adding_l676_676099


namespace find_greatest_individual_award_l676_676888

def prize_distribution : Type := sorry

noncomputable def greatest_individual_award
  (total_prize : ℝ)
  (total_winners : ℕ)
  (min_award : ℝ)
  (fraction_prize : ℝ)
  (fraction_winners : ℝ)
  (greatest_award : ℝ) : Prop :=
  total_prize = 400 ∧ total_winners = 20 ∧ min_award = 20 ∧ 
  fraction_prize = 2/5 ∧ fraction_winners = 3/5 ∧ 
  greatest_award = 100

theorem find_greatest_individual_award : 
  ∃ (total_prize total_winners min_award greatest_award : ℝ) 
    (fraction_prize fraction_winners : ℝ),
  prize_distribution ∧ greatest_individual_award total_prize total_winners 
  min_award fraction_prize fraction_winners greatest_award :=
sorry

end find_greatest_individual_award_l676_676888


namespace bad_arrangements_count_is_three_l676_676027

def is_bad_arrangement (arr : List ℕ) : Prop :=
  let sums := List.range 20 |>.map (λ n => List.any (List.sublists' arr |>.map List.sum) (· = n + 1))
  List.any (· = false) sums

def equivalent_under_rotation_reflection (arr1 arr2 : List ℕ) : Prop :=
  List.eq_or_mem {arr1, arr1.reverse} (List.range arr1.length).map (List.rotate arr1 ·)

def bad_arrangements_count : ℕ :=
  let all_arrangements := List.permutations [1, 2, 3, 4, 5, 6]
  let unique_arrangements := List.quotient.all all_arrangements (equivalent_under_rotation_reflection)
  (unique_arrangements.filter is_bad_arrangement).length

theorem bad_arrangements_count_is_three : bad_arrangements_count = 3 := by
  sorry

end bad_arrangements_count_is_three_l676_676027


namespace secretary_took_4_donuts_l676_676563

def donuts_taken_by_secretary (total_donuts eaten_by_bill remaining_after_coworkers : ℕ) : ℕ :=
  total_donuts - (remaining_after_coworkers * 2 + eaten_by_bill)

theorem secretary_took_4_donuts :
  donuts_taken_by_secretary 50 2 22 = 4 :=
by {
  unfold donuts_taken_by_secretary,
  norm_num,
}

end secretary_took_4_donuts_l676_676563


namespace final_score_l676_676073

theorem final_score (questions_first_half questions_second_half points_per_question : ℕ) (h1 : questions_first_half = 5) (h2 : questions_second_half = 5) (h3 : points_per_question = 5) : 
  (questions_first_half + questions_second_half) * points_per_question = 50 :=
by
  sorry

end final_score_l676_676073


namespace range_of_a_l676_676979

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 1 → log_a a (2 - a * x) < log_a a (2 - a * (x / 2))) →
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l676_676979


namespace quadrant_of_theta_l676_676224

theorem quadrant_of_theta
  (θ : ℝ)
  (htan : Real.tan θ < 0)
  (hcos : Real.cos θ > 0) :
  θ ∈ set.Ioo (3 * Real.pi / 2) (2 * Real.pi) :=
sorry

end quadrant_of_theta_l676_676224


namespace exists_bicolored_angles_l676_676370

-- Definitions for the conditions
def bicolored_angle (vertex : Point) (side1 side2 : Segment) (colors : Segment → Color) : Prop :=
  side1.vertex1 = vertex ∧ side1.vertex2 ≠ vertex ∧
  side2.vertex1 = vertex ∧ side2.vertex2 ≠ vertex ∧
  colors side1 ≠ colors side2

structure Configuration where
  k n : ℕ
  h_cond : n ≥ k ∧ k ≥ 3
  points : Fin (n+1) → Point
  h_ncollinear : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬Collinear (points i) (points j) (points k)
  colors : Segment → Fin k

noncomputable def number_of_bicolored_angles (config : Configuration) : ℕ :=
  ∑ p in set.univ (Fin (n+1)), 
    ∑ s1 in set.univ (Segment config.points), 
    ∑ s2 in set.univ (Segment config.points), 
      if p = s1.vertex1 ∧ p = s2.vertex1 ∧ config.colors s1 ≠ config.colors s2 then 1 else 0

theorem exists_bicolored_angles (config : Configuration) :
  ∃ coloring, number_of_bicolored_angles config > config.n * (config.n / config.k) ^ 2 * (choose config.k 2) :=
sorry  -- Proof goes here

end exists_bicolored_angles_l676_676370


namespace tomatoes_ready_for_sale_on_tuesday_l676_676380

-- Let's define the key quantities and conditions mentioned in the problem.
def initial_shipment := 1000.0 -- kg of tomatoes on Friday
def saturday_sales_rate := 0.60 -- 60% sales on Saturday
def sunday_spoilage_rate := 0.20 -- 20% spoilage on Sunday
def monday_shipment_rate := 1.5 -- Monday's shipment as multiple of Friday's shipment
def monday_sales_rate := 0.40 -- 40% sales on Monday
def tuesday_spoilage_rate := 0.15 -- 15% spoilage by Tuesday

-- The goal is to prove that Marta had 928.2 kg of tomatoes ready for sale on Tuesday.
theorem tomatoes_ready_for_sale_on_tuesday :
  let saturday_sales := initial_shipment * saturday_sales_rate,
      saturday_remaining := initial_shipment - saturday_sales,
      sunday_spoilage := saturday_remaining * sunday_spoilage_rate,
      sunday_remaining := saturday_remaining - sunday_spoilage,
      monday_shipment := initial_shipment * monday_shipment_rate,
      monday_total := sunday_remaining + monday_shipment,
      monday_sales := monday_total * monday_sales_rate,
      monday_remaining := monday_total - monday_sales,
      tuesday_spoilage := monday_remaining * tuesday_spoilage_rate,
      tuesday_remaining := monday_remaining - tuesday_spoilage
  in
  tuesday_remaining = 928.2 := by sorry

end tomatoes_ready_for_sale_on_tuesday_l676_676380


namespace tangent_sum_problem_l676_676634

theorem tangent_sum_problem
  (α β : ℝ)
  (h_eq_root : ∃ (x y : ℝ), (x = Real.tan α) ∧ (y = Real.tan β) ∧ (6*x^2 - 5*x + 1 = 0) ∧ (6*y^2 - 5*y + 1 = 0))
  (h_range_α : 0 < α ∧ α < π/2)
  (h_range_β : π < β ∧ β < 3*π/2) :
  (Real.tan (α + β) = 1) ∧ (α + β = 5*π/4) := 
sorry

end tangent_sum_problem_l676_676634


namespace integer_values_x_l676_676286

theorem integer_values_x {x : ℝ} (h : ⌈real.sqrt x⌉ = 20) : ∃ n : ℕ, n = 39 :=
by
  have h1 : 19 < real.sqrt x ∧ real.sqrt x ≤ 20 := sorry
  have h2 : 361 < x ∧ x ≤ 400 := sorry
  have x_values : set.Icc 362 400 := sorry
  have n_values : ∃ n : ℕ, n = set.finite.to_finset x_values.card := sorry
  exact n_values

end integer_values_x_l676_676286


namespace max_value_sqrt5_l676_676746

noncomputable def max_modulus (α β : ℂ) : ℝ :=
  |(β - α) / (1 - conj α * β)|

theorem max_value_sqrt5 (α β : ℂ)
  (h1 : |β| = 2)
  (h2 : arg β = arg α + π / 2)
  (h3 : conj α * β ≠ 1) :
  max_modulus α β = √5 :=
  sorry

end max_value_sqrt5_l676_676746


namespace proof_problem_l676_676801

noncomputable def g : ℝ → ℝ := sorry

lemma functional_equation (c d : ℝ) : (c^2) * g(d) = (d^2) * g(c) := sorry

lemma g4_nonzero : g 4 ≠ 0 := sorry

theorem proof_problem : (g(7) - g(3)) / g(4) = 2.5 := sorry

end proof_problem_l676_676801


namespace find_a_l676_676256

theorem find_a (a : ℝ) : (real.sqrt ((4 - 1)^2 + (2 - 2)^2 + (a - 3)^2) = real.sqrt 10) -> (a = 2 ∨ a = 4) := 
by {
    sorry
}

end find_a_l676_676256


namespace distance_between_points_l676_676948

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 3 3 (-2) (-2) = 5 * real.sqrt 2 :=
by
  sorry

end distance_between_points_l676_676948


namespace percentage_less_l676_676080

theorem percentage_less (P T J : ℝ) (hT : T = 0.9375 * P) (hJ : J = 0.8 * T) : (P - J) / P * 100 = 25 :=
by
  sorry

end percentage_less_l676_676080


namespace function_decreasing_l676_676012

noncomputable def f (a : ℝ) (x : ℝ) := a * x^3 - 2 * x

theorem function_decreasing (a : ℝ) :
  (∀ x, deriv (f a) x ≤ 0) → a ≤ 0 := 
begin
  -- Define the derivative of f
  have h_deriv : ∀ x, deriv (f a) x = 3 * a * x^2 - 2,
  {
    intro x,
    calc deriv (f a) x = deriv (λ x, a * x^3 - 2 * x) x : rfl
                    ... = a * deriv (λ x, x^3) x - 2 * deriv (λ x, x) x : by simp
                    ... = a * (3 * x^2) - 2 * 1 : by simp
                    ... = 3 * a * x^2 - 2 : by ring,
  },
  intro h,
  -- Consider the derivative at x = 0
  specialize h 0,
  rw h_deriv 0 at h,
  linarith,
end

end function_decreasing_l676_676012


namespace plane_coloring_l676_676463

-- Define a type for colors to represent red and blue
inductive Color
| red
| blue

-- The main statement
theorem plane_coloring (x : ℝ) (h_pos : 0 < x) (coloring : ℝ × ℝ → Color) :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ coloring p1 = coloring p2 ∧ dist p1 p2 = x :=
sorry

end plane_coloring_l676_676463


namespace jason_egg_consumption_l676_676600

theorem jason_egg_consumption :
  (∀ (days : ℕ), (weeks : ℕ), days = 7 ∧ weeks = 2 → 3 * days * weeks = 42) :=
by
  sorry

end jason_egg_consumption_l676_676600


namespace arithmetic_sequence_20th_term_l676_676004

-- Definitions for the first term and common difference
def first_term : ℤ := 8
def common_difference : ℤ := -3

-- Define the general term for an arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ := first_term + (n - 1) * common_difference

-- The specific property we seek to prove: the 20th term is -49
theorem arithmetic_sequence_20th_term : arithmetic_sequence 20 = -49 := by
  -- Proof is omitted, filled with sorry
  sorry

end arithmetic_sequence_20th_term_l676_676004


namespace prob_area_lt_circumference_l676_676886

-- Defining the problem parameters
def d : ℕ := 5
def k_values := {k | ∃ (i : ℕ) (j : ℕ), (2 ≤ i ∧ i ≤ 4) ∧ (2 ≤ j ∧ j ≤ 8) ∧ k = i + j}

-- Representing the Circumference formula
def circumference (k : ℕ) (d : ℕ) : ℝ := Real.pi * k * d

-- Representing the Area formula
def area (k : ℕ) (d : ℕ) : ℝ := Real.pi * k * (d^2) / 4

-- Statement to prove
theorem prob_area_lt_circumference : 
  (∀ k ∈ k_values, area k d < circumference k d) →
  (1 : ℝ) = 1 :=
by
  intros h
  sorry

end prob_area_lt_circumference_l676_676886


namespace non_qualified_pieces_l676_676808

theorem non_qualified_pieces (total_products : ℕ) (pass_rate : ℝ) (not_qualified_rate : ℝ) (num_not_qualified : ℕ) :
  total_products = 400 →
  pass_rate = 0.98 →
  not_qualified_rate = 1 - pass_rate →
  num_not_qualified = total_products * not_qualified_rate →
  num_not_qualified = 8 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end non_qualified_pieces_l676_676808


namespace triangle_inequality_l676_676994

variables {a b c Δ : ℝ} (h_triangle : ∃A B C : ℝ, A + B + C = π ∧ a = 2 * sin (A/2) ∧ b = 2 * sin (B/2) ∧ c = 2 * sin (C/2) ∧ Δ = 1/2 * a * b * sin C / 2)

theorem triangle_inequality (h : a^2 + b^2 + c^2 = 4 * sqrt 3 * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2) : 
  a = b ∧ b = c :=
sorry

end triangle_inequality_l676_676994


namespace minimum_distinct_coeffs_l676_676564

noncomputable def polynomial_deg_eight (a b c d e f g h i : ℝ) : polynomial ℝ :=
polynomial.C a * polynomial.X ^ 8 +
polynomial.C b * polynomial.X ^ 7 +
polynomial.C c * polynomial.X ^ 6 +
polynomial.C d * polynomial.X ^ 5 +
polynomial.C e * polynomial.X ^ 4 +
polynomial.C f * polynomial.X ^ 3 +
polynomial.C g * polynomial.X ^ 2 +
polynomial.C h * polynomial.X +
polynomial.C i

theorem minimum_distinct_coeffs (a b c d e f g h i : ℝ) (h_a : a ≠ 0) :
  let P := polynomial_deg_eight a b c d e f g h i in
  let distinct_coeffs := {a, 8 * a, 8 * 7 * a, 8 * 7 * 6 * a, 8 * 7 * 6 * 5 * a, 8 * 7 * 6 * 5 * 4 * a, 8 * 7 * 6 * 5 * 4 * 3 * a, 8 * 7 * 6 * 5 * 4 * 3 * 2 * a, 8! * a} in
  distinct_coeffs.card = 8 :=
sorry

end minimum_distinct_coeffs_l676_676564


namespace number_of_pieces_after_sawing_l676_676770

noncomputable def pieces_after_sawing (lcm : ℕ) : ℕ :=
  let marks_10 := list.range (lcm / 10) |>.map (λ n, 10 * (n + 1))
  let marks_12 := list.range (lcm / 12) |>.map (λ n, 12 * (n + 1))
  let marks_15 := list.range (lcm / 15) |>.map (λ n, 15 * (n + 1))
  let all_marks := (marks_10 ++ marks_12 ++ marks_15).eraseDups
  all_marks.length + 1

theorem number_of_pieces_after_sawing : pieces_after_sawing 60 = 28 := 
by sorry

end number_of_pieces_after_sawing_l676_676770


namespace closest_product_l676_676843

theorem closest_product 
  (a : ℝ := 0.001532) 
  (b : ℤ := 2134672) 
  (choices : Set ℝ := {3100, 3150, 3200, 3500, 4000}) : 
  (3150 ∈ choices) ∧ abs (a * b.toℝ - 3150) = min (abs (a * b.toℝ - c)) c ∈ choices :=
sorry

end closest_product_l676_676843


namespace distance_points_l676_676929

-- Definition of distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Points
def point1 : ℝ × ℝ := (3, 3)
def point2 : ℝ × ℝ := (-2, -2)

-- Main theorem
theorem distance_points : distance point1 point2 = 5 * Real.sqrt 2 :=
by
  sorry

end distance_points_l676_676929


namespace problem_statement_l676_676910

open Real

noncomputable def cubicRoot (x : ℝ) : ℝ := x^(1/3)

theorem problem_statement : cubicRoot (7 + 2 * sqrt 19) + cubicRoot (7 - 2 * sqrt 19) + 4 = 5 :=
by
  sorry

end problem_statement_l676_676910


namespace convex_polyhedron_circumscribed_sphere_l676_676894

theorem convex_polyhedron_circumscribed_sphere
  (P : Type) [polyhedron P] 
  (faces_inscribed : ∀ face : face P, inscribed_polygon face)
  (vertices_trivalent : ∀ v : vertex P, trivalent v) :
  ∃ sphere : Sphere, circumscribed sphere P :=
by
  sorry

end convex_polyhedron_circumscribed_sphere_l676_676894


namespace payroll_threshold_l676_676115

theorem payroll_threshold
  (payroll tax threshold : ℝ)
  (h1 : tax = 0.2 / 100 * (payroll - threshold))
  (h2 : payroll = 300000)
  (h3 : tax = 200) :
  threshold = 200000 := by
  have h4 : 0.2 / 100 = 0.002, by norm_num,
  rw [h4] at h1,
  linarith
  sorry

end payroll_threshold_l676_676115


namespace product_of_numbers_in_row_is_neg_one_l676_676212

theorem product_of_numbers_in_row_is_neg_one
  (a : Fin 100 → ℝ) (b : Fin 100 → ℝ)
  (h_distinct : Function.Injective a ∧ Function.Injective b)
  (h_col_product : ∀ j : Fin 100, (∏ i, a i + b j) = 1) :
  ∀ i : Fin 100, (∏ j, a i + b j) = -1 :=
sorry

end product_of_numbers_in_row_is_neg_one_l676_676212


namespace fraction_spent_first_week_l676_676074

theorem fraction_spent_first_week
  (S : ℝ) (F : ℝ)
  (h1 : S > 0)
  (h2 : F * S + 3 * (0.20 * S) + 0.15 * S = S) : 
  F = 0.25 := 
sorry

end fraction_spent_first_week_l676_676074


namespace limit_of_u_l676_676755

noncomputable def u (n : ℕ) : ℝ :=
  if n = 0 then 0 else (2 * n + Real.cos n) / (n * Real.sin (1 / n) + Real.sqrt ((n + 1) * (n + 2)))

theorem limit_of_u (h : ℕ+ → ℝ) (h_def : ∀ n : ℕ+, h n = (2 * n + Real.cos n) / (n * Real.sin (1 / n) + Real.sqrt ((n + 1) * (n + 2)))) :
  Filter.Tendsto h Filter.atTop (Filter.tendsto_const_nhds 2) :=
sorry

end limit_of_u_l676_676755


namespace centroid_moves_straight_line_l676_676255

theorem centroid_moves_straight_line (A B C : ℝ × ℝ) (AB_fixed : dist A B = d) :
  ∃ G : ℝ × ℝ, (is_centroid A B C G) ∧ C_moves_on_line C → G_moves_on_straight_line_parallel_to_AB G A B :=
by
  sorry

end centroid_moves_straight_line_l676_676255


namespace percentage_less_than_l676_676082

theorem percentage_less_than (P T J : ℝ) 
  (h1 : T = 0.9375 * P) 
  (h2 : J = 0.8 * T) 
  : (P - J) / P * 100 = 25 := 
by
  sorry

end percentage_less_than_l676_676082


namespace random_event_is_eventA_l676_676848

-- Definitions of conditions
def eventA : Prop := true  -- Tossing a coin and it lands either heads up or tails up is a random event
def eventB : Prop := (∀ (a b : ℝ), (b * a = b * a))  -- The area of a rectangle with sides of length a and b is ab is a certain event
def eventC : Prop := ∃ (defective_items : ℕ), (defective_items / 100 = 10 / 100)  -- Drawing 2 defective items from 100 parts with 10% defective parts is uncertain
def eventD : Prop := false -- Scoring 105 points in a regular 100-point system exam is an impossible event

-- The proof problem statement
theorem random_event_is_eventA : eventA ∧ ¬eventB ∧ ¬eventC ∧ ¬eventD := 
sorry

end random_event_is_eventA_l676_676848


namespace valid_starting_days_l676_676108

theorem valid_starting_days (days_in_month : ℕ) (days_in_week : ℕ) : 
  days_in_month = 30 ∧ days_in_week = 7 →
  (∃ n, n = 3) ↔ ∀ start_day, (days_in_month + start_day) % days_in_week ∈ {4, 5} := 
begin
  sorry
end

end valid_starting_days_l676_676108


namespace largest_A_l676_676477

def A : ℝ := (2010 / 2009) + (2010 / 2011)
def B : ℝ := (2010 / 2011) + (2012 / 2011)
def C : ℝ := (2011 / 2010) + (2011 / 2012)

theorem largest_A : A > B ∧ A > C := by sorry

end largest_A_l676_676477


namespace sign_of_ac_l676_676190

theorem sign_of_ac (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h : (a / b) + (c / d) = (a + c) / (b + d)) : a * c < 0 :=
by
  sorry

end sign_of_ac_l676_676190


namespace inequality_holds_iff_a_in_range_l676_676704

theorem inequality_holds_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, 3^(x^2 - 2*a*x) > (1/3)^(x+1)) ↔ (-1/2 < a ∧ a < 3/2) :=
by
  sorry

end inequality_holds_iff_a_in_range_l676_676704


namespace andrew_paid_in_dollars_l676_676900

def local_currency_to_dollars (units : ℝ) : ℝ := units * 0.25

def cost_of_fruits : ℝ :=
  let cost_grapes := 7 * 68
  let cost_mangoes := 9 * 48
  let cost_apples := 5 * 55
  let cost_oranges := 4 * 38
  let total_cost_grapes_mangoes := cost_grapes + cost_mangoes
  let total_cost_apples_oranges := cost_apples + cost_oranges
  let discount_grapes_mangoes := 0.10 * total_cost_grapes_mangoes
  let discounted_grapes_mangoes := total_cost_grapes_mangoes - discount_grapes_mangoes
  let discounted_apples_oranges := total_cost_apples_oranges - 25
  let total_discounted_cost := discounted_grapes_mangoes + discounted_apples_oranges
  let sales_tax := 0.05 * total_discounted_cost
  let total_tax := sales_tax + 15
  let total_amount_with_taxes := total_discounted_cost + total_tax
  total_amount_with_taxes

theorem andrew_paid_in_dollars : local_currency_to_dollars cost_of_fruits = 323.79 :=
  by
  sorry

end andrew_paid_in_dollars_l676_676900


namespace right_triangle_PQR_tan_pr_l676_676309

theorem right_triangle_PQR_tan_pr (PR QR : ℕ) (tanP : ℚ) (h_tanP : tanP = 3 / 4) (h_PR : PR = 12) (h_angleR : ∠PRQ = 90) :
  let PQ := Math.sqrt (QR ^ 2 + PR ^ 2)
  PQ = 15
:= sorry

end right_triangle_PQR_tan_pr_l676_676309


namespace find_m_plus_n_l676_676168

def probability_no_collection (total_pairs: Nat) : Prop :=
  let good_permutations : ℕ := (Nat.factorial (total_pairs - 1)) + 
                                (binomial (2 * total_pairs) total_pairs * (Nat.factorial (total_pairs - 1))^2 / 2)
  let total_permutations : ℕ := Nat.factorial (2 * total_pairs)
  let gcd_mn := Nat.gcd 27 192
  (good_permutations / total_permutations = 27 / 192) ∧ (27 / gcd_mn = 27) ∧ (192 / gcd_mn = 64) ∧ (27 + 64 = 73)

theorem find_m_plus_n (h : probability_no_collection 8): 
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ 
               ((m + n) = 73) := by
  { 
    sorry 
  }

end find_m_plus_n_l676_676168


namespace painting_time_l676_676378

def Linda_rate := 1 / 3
def Tom_rate := 1 / 4
def Jerry_rate := 1 / 6

def total_time_painting (T : ℝ) : ℝ :=
  (Linda_rate * T) + (Tom_rate * (T - 2)) + (Jerry_rate * (T - 1))

theorem painting_time :
  ∃ T : ℝ, total_time_painting T = 1 ∧ T = 18 / 17 :=
begin
  use 18 / 17,
  split,
  {
    unfold total_time_painting,
    norm_num,
  },
  {
    norm_num,
  }
end

end painting_time_l676_676378


namespace trajectory_equation_l676_676216

variables (OA OB OP PA AB : ℝ → ℝ) -- Non-zero, non-collinear vectors (points in ℝ)

variables (x y λ : ℝ)

-- Conditions
def condition1 := ¬collinear OA OB
def condition2 := ∀ t : ℝ, 2 * OP t = x * OA t + y * OB t
def condition3 := ∀ t : ℝ, PA t = λ * AB t

-- Prove the trajectory equation x + y - 2 = 0
theorem trajectory_equation (h1 : condition1) (h2 : condition2) (h3 : condition3) : x + y - 2 = 0 := by
  sorry

end trajectory_equation_l676_676216


namespace similar_triangles_l676_676134

structure Circle (point : Type) :=
(center : point)
(radius : ℝ)

variables (point : Type) [EuclideanGeometry point]

variables (O1 O2 B C E P A : point)
variables (r1 r2 : ℝ)

-- Definitions based on the conditions
def C1 : Circle point := Circle.mk O1 r1
def C2 : Circle point := Circle.mk O2 r2

-- External tangency condition
axiom external_tangent (h1 : Tangent point P O1 B) (h2 : Tangent point P O2 C) :
  tangent_at_point C1 C2 A

-- Ratio condition
axiom ratio_condition : (segment_length P B) / (segment_length P C) = r1 / r2

-- Intersect condition
axiom intersect_condition : intersects P A C2 E

-- Prove that the triangles are similar
theorem similar_triangles :
  similar_triangles (triangle P A B) (triangle P E C) := by 
  sorry

end similar_triangles_l676_676134


namespace num_possible_x_l676_676271

theorem num_possible_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 20) : {y : ℕ | 361 < y ∧ y ≤ 400}.card = 39 :=
by
  sorry

end num_possible_x_l676_676271


namespace distance_points_l676_676934

-- Definition of distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Points
def point1 : ℝ × ℝ := (3, 3)
def point2 : ℝ × ℝ := (-2, -2)

-- Main theorem
theorem distance_points : distance point1 point2 = 5 * Real.sqrt 2 :=
by
  sorry

end distance_points_l676_676934


namespace set_contains_all_integers_l676_676753

theorem set_contains_all_integers
  (n : ℕ)
  (a : ℕ → ℤ) -- sequence of integers a₁, a₂, ..., aₙ
  (h_gcd : Int.gcd (a 1) (Int.gcd (a 2) ... (a n)) = 1) -- gcd(a₁, a₂, ..., aₙ) = 1
  (S : Set ℤ) -- set of integers S
  (cond1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ∈ S) -- condition 1
  (cond2 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → a i - a j ∈ S) -- condition 2
  (cond3 : ∀ x y : ℤ, x ∈ S ∧ y ∈ S ∧ (x + y ∈ S) → (x - y ∈ S)) -- condition 3
  : S = Set.univ -- conclusion that S = ℤ
:= by
  sorry

end set_contains_all_integers_l676_676753


namespace min_x_prime_factors_sum_l676_676361

theorem min_x_prime_factors_sum
  (x y a b c d : ℕ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : 3 * x^12 = 5 * y^17)
  (hx : nat.factors x = [a, b])
  (prime_a : nat.prime a)
  (prime_b : nat.prime b)
  (ha : nat.multiplicity a x = c)
  (hb : nat.multiplicity b x = d)
  (h4 : a ≠ b) :
  a + b + c + d = 30 :=
sorry

end min_x_prime_factors_sum_l676_676361


namespace count_valid_integers_l676_676667

-- Definitions representing the conditions
def is_valid_integer (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 2023 ∧ (n % 10 = 2 * ((n / 10) % 10 + (n / 100) % 10 + (n / 1000) % 10))

-- The proof statement
theorem count_valid_integers : 
  (Finset.card (Finset.filter is_valid_integer (Finset.range 2024 \ Finset.range 1000))) = 16 :=
by
  sorry

end count_valid_integers_l676_676667


namespace Jesse_remaining_money_l676_676332

-- Define the conditions
def initial_money := 50
def novel_cost := 7
def lunch_cost := 2 * novel_cost
def total_spent := novel_cost + lunch_cost

-- Define the remaining money after spending
def remaining_money := initial_money - total_spent

-- Prove that the remaining money is $29
theorem Jesse_remaining_money : remaining_money = 29 := 
by
  sorry

end Jesse_remaining_money_l676_676332


namespace rhombus_area_l676_676308

-- Conditions translated to definitions
def pointA : ℝ × ℝ := (0, 3.5)
def pointB : ℝ × ℝ := (6, 0)
def pointC : ℝ × ℝ := (0, -3.5)
def pointD : ℝ × ℝ := (-6, 0)

-- Function to calculate distance between two points
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  (real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2))

-- Calculate lengths of diagonals
def d1 : ℝ := dist pointA pointC
def d2 : ℝ := dist pointB pointD

-- Proof problem statement
theorem rhombus_area : (d1 * d2) / 2 = 42 := by
  -- Proof goes here
  sorry

end rhombus_area_l676_676308


namespace b_n_arithmetic_max_min_terms_l676_676211

open Nat

-- Define the sequence a_n
def seq_a : ℕ → ℝ
| 0       := 3 / 5
| (n + 1) := 2 - 1 / (seq_a n)

-- Define the sequence b_n based on a_n
def seq_b (n : ℕ) : ℝ := 1 / (seq_a n - 1)

-- Prove that b_n is an arithmetic sequence with common difference 1
theorem b_n_arithmetic :
  ∀ n : ℕ, seq_b (n + 1) - seq_b n = 1 :=
sorry

-- Prove the maximum and minimum terms in the sequence a_n
theorem max_min_terms :
  seq_a 3 = 3 ∧ seq_a 2 = -1 :=
sorry

end b_n_arithmetic_max_min_terms_l676_676211


namespace remaining_land_area_l676_676113

/-- A proof to demonstrate the remaining land area after accounting for the pond
    in a rectangular field with specified area and perimeter, where the diameter
    of the pond is half the width of the rectangle. -/
theorem remaining_land_area (L W : ℝ) (h_area : L * W = 800) (h_perimeter : 2 * L + 2 * W = 120)
    (h_diameter : W / 2 * 2 = W) :
    let r := W / 4 in 
    let area_pond := π * r^2 in
    let remaining_area := 800 - area_pond in
    remaining_area = 800 - 25 * π :=
by
  sorry

end remaining_land_area_l676_676113


namespace find_charged_batteries_l676_676106

theorem find_charged_batteries (n : ℕ) (h_n : n ≥ 4) 
  (charged uncharged : finset ℕ) (h_card_charged : charged.card = n) (h_card_uncharged : uncharged.card = n) :
  ∃ (tries : finset (finset ℕ)), 
    (tries.card ≤ n + 2) ∧ 
    ∃ (pair : finset ℕ), (pair ⊆ charged) ∧ (pair.card = 2) ∧ (pair ∈ tries) :=
sorry

end find_charged_batteries_l676_676106


namespace min_value_proof_l676_676355

noncomputable def min_value_condition (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ b + c ≥ a 

theorem min_value_proof (a b c : ℝ) (h : min_value_condition a b c) : 
  ∃ (x : ℝ), (x = ∑ (y : ℝ) in [b/c, c/(a+b)], y) ∧ x = sqrt 2 - 1/2 :=
by sorry

end min_value_proof_l676_676355


namespace find_cos_7theta_l676_676677

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end find_cos_7theta_l676_676677


namespace simplest_form_expression_l676_676069

theorem simplest_form_expression (x y a : ℤ) : 
  (∃ (E : ℚ → Prop), (E (1/3) ∨ E (1/(x-2)) ∨ E ((x^2 * y) / (2*x)) ∨ E (2*a / 8)) → (E (1/(x-2)) ↔ E (1/(x-2)))) :=
by 
  sorry

end simplest_form_expression_l676_676069


namespace even_three_digit_numbers_count_l676_676666

theorem even_three_digit_numbers_count : 
  let valid_combinations : List (ℕ × ℕ) := [(2, 9), (3, 8), (4, 7), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)] in
  let valid_units : List ℕ := [0, 2, 4, 6, 8] in
  valid_combinations.length * valid_units.length = 35 :=
by
  sorry

end even_three_digit_numbers_count_l676_676666


namespace not_divisible_by_n_l676_676346

theorem not_divisible_by_n (n : ℕ) (h1 : n > 1) (h2 : odd n) : ¬ n ∣ 3^n + 1 :=
sorry

end not_divisible_by_n_l676_676346


namespace linear_function_solution_l676_676232

theorem linear_function_solution (f : ℝ → ℝ) 
  (h₁ : ∀ x y, f (x + y) = f x + f y) 
  (h₂ : ∀ (a : ℝ) (x : ℝ), f (a * x) = a * f x) 
  (h₃ : ∀ x, f x = x + 2 * ∫ t in 0..1, f t) : 
  ∀ x, f x = x - 1 := 
by 
  sorry

end linear_function_solution_l676_676232


namespace total_solar_systems_and_planets_l676_676050

theorem total_solar_systems_and_planets (planets : ℕ) (solar_systems_per_planet : ℕ) (h1 : solar_systems_per_planet = 8) (h2 : planets = 20) : (planets + planets * solar_systems_per_planet) = 180 :=
by
  -- translate conditions to definitions
  let solar_systems := planets * solar_systems_per_planet
  -- sum solar systems and planets
  let total := planets + solar_systems
  -- exact proof goal
  exact calc
    (planets + solar_systems) = planets + planets * solar_systems_per_planet : by rfl
                        ... = 20 + 20 * 8                       : by rw [h1, h2]
                        ... = 180                                : by norm_num

end total_solar_systems_and_planets_l676_676050


namespace rashmi_speed_second_day_l676_676780

noncomputable def rashmi_speed (distance speed1 time_late time_early : ℝ) : ℝ :=
  let time1 := distance / speed1
  let on_time := time1 - time_late / 60
  let time2 := on_time - time_early / 60
  distance / time2

theorem rashmi_speed_second_day :
  rashmi_speed 9.999999999999993 5 10 10 = 6 := by
  sorry

end rashmi_speed_second_day_l676_676780


namespace true_propositions_count_l676_676646

def proposition_1 : Prop := ¬(1 = x^0)
def proposition_2 : Prop := ¬(∃ x, 2^x - log 2 x = 0)
def proposition_3 : Prop := ∀ x, x ∈ [2, ∞) → (sqrt (x-1) * (x-2) ≥ 0)
def proposition_4 : Prop := ∀ x, (x < 1 → x < 2) ∧ ¬(x < 1 ↔ x < 2)
def proposition_5 (a : ℝ) (n : ℕ): Prop := ¬(a = 0 → (arithmetic_sequence a n ∨ geometric_sequence a n))
  
theorem true_propositions_count : 1 =
  (if proposition_1 then 1 else 0) +
  (if proposition_2 then 1 else 0) +
  (if proposition_3 then 1 else 0) +
  (if proposition_4 then 1 else 0) +
  (if proposition_5 0 then 1 else 0) := by
  sorry

end true_propositions_count_l676_676646


namespace evaluate_expression_l676_676599

theorem evaluate_expression (a b c : ℝ) (h1 : a = 4) (h2 : b = -4) (h3 : c = 3) : (3 / (a + b + c) = 1) :=
by
  sorry

end evaluate_expression_l676_676599


namespace range_of_function_l676_676033

theorem range_of_function :
  ∀ y : ℝ, (∃ x : ℝ, y = (1 / 2) ^ (x^2 + 2 * x - 1)) ↔ (0 < y ∧ y ≤ 4) :=
by
  sorry

end range_of_function_l676_676033


namespace coprime_with_81_count_l676_676130

theorem coprime_with_81_count : 
  ∃ n, n = Nat.card {m | m ≤ 81 ∧ Nat.coprime m 81} ∧ n = 54 := 
by
  sorry

end coprime_with_81_count_l676_676130


namespace sum_first_9_terms_l676_676300

variable (a₁ d : ℝ)
noncomputable def a (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def S (n : ℕ) : ℝ := n / 2 * (2 * a₁ + (n - 1) * d)

theorem sum_first_9_terms (h : a 3 + a 7 = 4) : S 9 = 18 :=
by
  have h1 : a₁ + 2 * d + (a₁ + 6 * d) = 4 := by rw [a, a, h]
  have h2 : 2 * a₁ + 8 * d = 4 := by linarith [h1]
  have h3 : a₁ + 4 * d = 2 := by linarith [h2]
  rw [S]
  have h4 : 2 * a₁ + 8 * d = 4 := by linarith [h3]
  linarith [h4]

end sum_first_9_terms_l676_676300


namespace sandbox_side_length_l676_676881

theorem sandbox_side_length (side_length : ℝ) (sand_sq_inches_per_pound : ℝ := 80 / 30) (total_sand_pounds : ℝ := 600) :
  (side_length ^ 2 = total_sand_pounds * sand_sq_inches_per_pound) → side_length = 40 := 
by
  sorry

end sandbox_side_length_l676_676881


namespace divisors_log_sum_eq_l676_676040

open BigOperators

/-- Given the sum of the base-10 logarithms of the divisors of \( 10^{2n} = 4752 \), prove that \( n = 12 \). -/
theorem divisors_log_sum_eq (n : ℕ) (h : ∑ a in Finset.range (2*n + 1), ∑ b in Finset.range (2*n + 1), 
  (a * Real.log (2) / Real.log (10) + b * Real.log (5) / Real.log (10)) = 4752) : n = 12 :=
by {
  sorry
}

end divisors_log_sum_eq_l676_676040


namespace soaking_time_l676_676142

theorem soaking_time : 
  let grass_stain_time := 4 
  let marinara_stain_time := 7 
  let num_grass_stains := 3 
  let num_marinara_stains := 1 
  in 
  num_grass_stains * grass_stain_time + num_marinara_stains * marinara_stain_time = 19 := 
by 
  sorry

end soaking_time_l676_676142


namespace sum_abc_eq_five_l676_676822

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_abc_eq_five (A B C : ℕ) (h_pos_A : 0 < A) (h_pos_B : 0 < B) (h_pos_C : 0 < C) 
  (h_coprime : Nat.coprime A B ∧ Nat.coprime B C ∧ Nat.coprime C A)
  (h_eq : A * log_base 180 5 + B * log_base 180 3 + C * log_base 180 2 = 1) :
  A + B + C = 5 :=
sorry

end sum_abc_eq_five_l676_676822


namespace ellipse_equation_and_max_area_l676_676214

theorem ellipse_equation_and_max_area
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ (F : ℝ × ℝ), F = (2 * Real.sqrt 2, 0) ∧ ∀ x y, (x, y) ∈ ellipse a b → dist (x, y) F = 2 * Real.sqrt 2)
  (h4 : dist (3, 0) (2 * Real.sqrt 2, 0) = 3 - 2 * Real.sqrt 2) :
  (ellipse_equation : ∀ x y, x^2 / 9 + y^2 = 1) ∧
  (max_area : ∃ PA PB AB, is_perpendicular PA PB ∧ PA = (3, 0) ∧ AB = ellipse a b ∧ 
           area_triangle PA AB PB = 3 / 8) :=
sorry

end ellipse_equation_and_max_area_l676_676214


namespace sum_of_two_digit_divisors_l676_676148

theorem sum_of_two_digit_divisors (d : ℕ) (h1 : 145 % d = 4) (h2 : 10 ≤ d ∧ d < 100) :
  d = 47 :=
by
  have hd : d ∣ 141 := sorry
  exact sorry

end sum_of_two_digit_divisors_l676_676148


namespace transformation_matrix_correct_l676_676465

def rotation_matrix_30_clockwise : Matrix (Fin 2) (Fin 2) ℝ :=
  !![
    (Math.sqrt 3) / 2, 1 / 2;
    (-1 / 2), (Math.sqrt 3) / 2
  ]

def translation_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    1, 0, 2;
    0, 1, -1;
    0, 0, 1
  ]

def combined_transformation_matrix (R : Matrix (Fin 2) (Fin 2) ℝ) (T : Matrix (Fin 3) (Fin 3) ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  let R_homogeneous : Matrix (Fin 3) (Fin 3) ℝ := !![
    R[0, 0], R[0, 1], 0;
    R[1, 0], R[1, 1], 0;
    0, 0, 1
  ]
  T.mul R_homogeneous

theorem transformation_matrix_correct :
  combined_transformation_matrix rotation_matrix_30_clockwise translation_matrix =
    !![
      (Math.sqrt 3) / 2, 1 / 2, 2;
      (-1 / 2), (Math.sqrt 3) / 2, -1;
      0, 0, 1
    ] :=
  sorry

end transformation_matrix_correct_l676_676465


namespace odd_function_condition_l676_676023

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (a x : ℝ) : ℝ := 
  if ha : (a^2 - x^2 ≥ 0) ∧ (|x + a| - a ≠ 0) then 
    (sqrt (a^2 - x^2)) / (|x + a| - a) 
  else 
    0 -- Out of the domain considering x or a values making divisor zero or negative sqrt.

theorem odd_function_condition (a : ℝ) : is_odd_function (f a) ↔ a > 0 :=
sorry

end odd_function_condition_l676_676023


namespace stocks_higher_price_l676_676769

/-- There are 1980 different stocks, and yesterday's closing prices were all different from today's closing prices.
The overall stock exchange index price increased by 2% today.
The number of stocks that closed at a higher price today is 20 percent greater than the number that closed at a lower price.
Given the above conditions, prove the number of stocks that closed at a higher price today than yesterday is 1080. -/
theorem stocks_higher_price (H L : ℕ) (h1 : H = 1.20 * L) (h2 : H + L = 1980) :
  H = 1080 :=
by
  sorry

end stocks_higher_price_l676_676769


namespace minimal_additional_flights_connectivity_l676_676784

-- Define the cities as a finite set and the initial flight connections as a function
def cities := {A1, A2, A3, A4, A5, A6, A7}
def initial_flights : cities → cities
| A1 := A2
| A2 := A3
| A3 := A4
| A4 := A5
| A5 := A6
| A6 := A7
| A7 := A1

-- Define the additional flights needed to satisfy the given condition
def additional_flights : cities → cities
| A1 := some A3
| A2 := some A4
| A3 := some A5
| A4 := some A6
| A5 := some A7
| _ := none  -- No additional flight from A6 or A7 in this simplified model

-- Define a function that represents all possible flights after adding the additional ones
def all_flights (c : cities) : set cities :=
  {c'} | c' ← initial_flights c ∪ some (additional_flights c)

-- Theorem statement to prove the connectivity in at most 2 stops
theorem minimal_additional_flights_connectivity :
  ∃ f : cities → set cities, 
    (∀ x y : cities, y ∈ f x ∨ ∃ z : cities, z ∈ f x ∧ y ∈ f z) ∧
    (f = initial_flights ∪ additional_flights) := 
sorry

end minimal_additional_flights_connectivity_l676_676784


namespace cubes_of_rational_sum_l676_676742

open Int

theorem cubes_of_rational_sum (a b : ℤ) (h : ∃ (t : ℚ), (t : ℝ) = ∛a + ∛b) : ∃ (x y : ℤ), a = x^3 ∧ b = y^3 :=
by
  sorry

end cubes_of_rational_sum_l676_676742


namespace product_of_solutions_l676_676609

theorem product_of_solutions : ∀ y : ℝ, 
    (|y| = 3 * (|y| - 2)) → (y = 3 ∨ y = -3) → (3 * -3 = -9) :=
by
  intro y
  intro h
  intro hsol
  exact (3 * -3 = -9)
  sorry

end product_of_solutions_l676_676609


namespace solution_interval_l676_676173

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 x + x - 2

theorem solution_interval :
  ∃ x, (1 < x ∧ x < 1.5) ∧ f x = 0 :=
sorry

end solution_interval_l676_676173


namespace find_a_tangent_line_at_minus_one_l676_676248

-- Define the function f with variable a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x - 1

-- Define the derivative of f with variable a
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Given conditions
def condition_1 : Prop := f' 1 = 1
def condition_2 : Prop := f' 2 (1 : ℝ) = 1

-- Prove that a = 2 given f'(1) = 1
theorem find_a : f' 2 (1 : ℝ) = 1 → 2 = 2 := by
  sorry

-- Given a = 2, find the tangent line equation at x = -1
def tangent_line_equation (x y : ℝ) : Prop := 9*x - y + 3 = 0

-- Define the coordinates of the point on the curve at x = -1
def point_on_curve : Prop := f 2 (-1) = -6

-- Prove the tangent line equation at x = -1 given a = 2
theorem tangent_line_at_minus_one (h : true) : tangent_line_equation 9 (f' 2 (-1)) := by
  sorry

end find_a_tangent_line_at_minus_one_l676_676248


namespace max_value_of_fraction_l676_676757

theorem max_value_of_fraction (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 4 * x^2 - 3 * x * y + y^2 = z) : 
  ∃ M, M = 1 ∧ ∀ (a b c : ℝ), (a > 0) → (b > 0) → (c > 0) → (4 * a^2 - 3 * a * b + b^2 = c) → (a * b / c ≤ M) :=
by
  sorry

end max_value_of_fraction_l676_676757


namespace ratio_time_upstream_downstream_l676_676107

variables (Vm Vc Vu Vd Tu Td : ℝ)

-- Definitions based on conditions
def speed_man_still : ℝ := 3.9
def speed_current : ℝ := 1.3

def effective_speed_upstream : ℝ := speed_man_still - speed_current
def effective_speed_downstream : ℝ := speed_man_still + speed_current

-- Main statement to prove
theorem ratio_time_upstream_downstream : effective_speed_upstream = 2.6 ∧ effective_speed_downstream = 5.2 → Tu / Td = 2 := by
  intro h
  have h1 : effective_speed_upstream = 2.6 := and.left h
  have h2 : effective_speed_downstream = 5.2 := and.right h
  rw [h1, h2]
  simp
  exact sorry

end ratio_time_upstream_downstream_l676_676107


namespace total_boxes_used_l676_676516

theorem total_boxes_used (total_oranges : ℕ) (oranges_per_box : ℕ) (damaged_percentage : ℚ) 
  (H1 : total_oranges = 2650) (H2 : oranges_per_box = 10) (H3 : damaged_percentage = 0.05) :
  let number_of_boxes := total_oranges / oranges_per_box in
  let damaged_boxes := (damaged_percentage * number_of_boxes).ceil.to_nat in
  number_of_boxes + damaged_boxes = 279 :=
by
  sorry

end total_boxes_used_l676_676516


namespace intersection_points_on_ellipse_l676_676970

theorem intersection_points_on_ellipse (s x y : ℝ)
  (h_line1 : s * x - 3 * y - 4 * s = 0)
  (h_line2 : x - 3 * s * y + 4 = 0) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1 :=
by
  sorry

end intersection_points_on_ellipse_l676_676970


namespace problem1_problem2_l676_676876

-- Define the conditions
variable (a x : ℝ)
variable (h_gt_zero : x > 0) (a_gt_zero : a > 0)

-- Problem 1: Prove that 0 < x ≤ 300
theorem problem1 (h: 12 * (500 - x) * (1 + 0.005 * x) ≥ 12 * 500) : 0 < x ∧ x ≤ 300 := 
sorry

-- Problem 2: Prove that 0 < a ≤ 5.5 given the conditions
theorem problem2 (h1 : 12 * (a - 13 / 1000 * x) * x ≤ 12 * (500 - x) * (1 + 0.005 * x))
                (h2 : x = 250) : 0 < a ∧ a ≤ 5.5 := 
sorry

end problem1_problem2_l676_676876


namespace brenda_num_cookies_per_box_l676_676565

def numCookiesPerBox (trays : ℕ) (cookiesPerTray : ℕ) (costPerBox : ℚ) (totalSpent : ℚ) : ℚ :=
  let totalCookies := trays * cookiesPerTray
  let numBoxes := totalSpent / costPerBox
  totalCookies / numBoxes

theorem brenda_num_cookies_per_box :
  numCookiesPerBox 3 80 3.5 14 = 60 := by
  sorry

end brenda_num_cookies_per_box_l676_676565


namespace geometric_sequence_sum_six_l676_676238

-- Definitions and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a(n+1) / a(n) = a(1) / a(0)

def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a(n) > 0

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

def a_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n

noncomputable def sequence := λ (n : ℕ), (1/2)^(n - 1)

theorem geometric_sequence_sum_six (a : ℕ → ℝ) (q: ℝ) (h_geom : is_geometric_sequence a) (h_pos : positive_terms a)
    (h1 : a 2 * a 4 = 64) (h2 : a 4 + 2 * a 5 = 8) :
    sum_first_n_terms a 6 = 126 := by
  sorry

end geometric_sequence_sum_six_l676_676238


namespace solve_fx_minus_2_l676_676998

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 6

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem solve_fx_minus_2 (h1 : ∀ x : ℝ, x ≥ 0 → f x = x^2 + x - 6)
  (h2 : even_function f) : {x : ℝ | f (x - 2) > 0} = set_of(λ x : ℝ, x < 0 ∨ x > 4) :=
sorry

end solve_fx_minus_2_l676_676998


namespace sum_of_squares_ge_sqrt_three_times_product_l676_676231

theorem sum_of_squares_ge_sqrt_three_times_product
    (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c ≤ a + b + c) :
    a^2 + b^2 + c^2 ≥ real.sqrt 3 * (a * b * c) :=
by
  sorry

end sum_of_squares_ge_sqrt_three_times_product_l676_676231


namespace solution_set_for_inequality_l676_676180

open Set Real

theorem solution_set_for_inequality : 
  { x : ℝ | (2 * x) / (x + 1) ≤ 1 } = Ioc (-1 : ℝ) 1 := 
sorry

end solution_set_for_inequality_l676_676180


namespace sufficient_conditions_l676_676221

structure Plane (α : Type*) :=
  (points : set α) -- Placeholder, use proper definitions as needed

structure Line (α : Type*) :=
  (points : set α) -- Placeholder, use proper definitions as needed

variables {α : Type*} [euclidean_space α]
variables (a b c : Line α) (α β : Plane α)

-- Conditions
variable (h1 : α ∥ β)
variable (h2a : a ⊆ α)
variable (h3a : b ∥ β)
variable (h2c : a ∥ c)
variable (h3c : b ∥ c)
variable (h3i : α ∩ β = c)
variable (h2b : b ⊆ β)
variable (h4a : a ∥ β)
variable (h4b : b ∥ α)
variable (h4d : a ⟂ c)
variable (h4e : b ⟂ c)

-- Confirm the number of sufficient conditions is exactly 2
theorem sufficient_conditions : (sufficiency_1 h1 h2a h3a ∨
                                  sufficiency_2 h2c h3c ∨
                                  sufficiency_3 h3i h2a h2b h4a h4b ∨
                                  sufficiency_4 h4d h4e) -> 
                                  (sufficiency_1 h1 h2a h3a ∧
                                  sufficiency_2 h2c h3c ∧
                                  sufficiency_3 h3i h2a h2b h4a h4b ∧
                                  sufficiency_4 h4d h4e) = 2 :=
by sorry

end sufficient_conditions_l676_676221


namespace count_remaining_after_removing_multiples_of_4_and_5_l676_676744

open Finset

def T := (range 100).map (λ n, n + 1)

def multiples_of (n : ℕ) (s : Finset ℕ) : Finset ℕ :=
  s.filter (λ x, x % n = 0)

theorem count_remaining_after_removing_multiples_of_4_and_5 :
  card (T \ (multiples_of 4 T ∪ multiples_of 5 T)) = 60 := by
  sorry

end count_remaining_after_removing_multiples_of_4_and_5_l676_676744


namespace highest_price_gt_56000_l676_676021

noncomputable def highest_possible_price (prices : List ℕ) : ℕ := 
  list.max' (List.cons 52000 (List.cons 35000 (List.cons 44000 prices))) sorry

theorem highest_price_gt_56000 
  (x y : ℕ) 
  (h1 : 52000 = median (35000 :: 44000 :: x :: y :: []))
  (h2 : 56000 = median [x, y]) : 
  highest_possible_price [35000, 44000, x, y] > 56000 :=
  sorry

end highest_price_gt_56000_l676_676021


namespace ducks_remaining_after_three_nights_l676_676820

def initial_ducks : ℕ := 320
def first_night_ducks (initial_ducks : ℕ) : ℕ := initial_ducks - (initial_ducks / 4)
def second_night_ducks (first_night_ducks : ℕ) : ℕ := first_night_ducks - (first_night_ducks / 6)
def third_night_ducks (second_night_ducks : ℕ) : ℕ := second_night_ducks - (second_night_ducks * 30 / 100)

theorem ducks_remaining_after_three_nights : 
  third_night_ducks (second_night_ducks (first_night_ducks initial_ducks)) = 140 :=
by
  -- Proof goes here
  sorry

end ducks_remaining_after_three_nights_l676_676820


namespace sum_of_integers_l676_676812

theorem sum_of_integers : (Finset.sum (Finset.filter (λ x : ℤ, (-5 < x) ∧ (x < 3)) (Finset.range 8).image (λ i, i - 4))) = -7 :=
by
  sorry

end sum_of_integers_l676_676812


namespace simplify_and_evaluate_l676_676395

-- Problem statement with conditions translated into Lean
theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 5 + 1) :
  (a / (a^2 - 2*a + 1)) / (1 + 1 / (a - 1)) = Real.sqrt 5 / 5 := sorry

end simplify_and_evaluate_l676_676395


namespace book_price_l676_676264

theorem book_price (x : ℕ) (h1 : x - 1 = 1 + (x - 1)) : x = 2 :=
by
  sorry

end book_price_l676_676264


namespace angle_CHX_is_5_degrees_l676_676545

theorem angle_CHX_is_5_degrees
  {A B C X Y H : Type}
  [IsAcuteTriangle A B C]
  (h₁ : Altitude AX ∧ Altitude BY)
  (h₂ : H = Orthocenter A B C) 
  (h₃ : ∠BAC = 50)
  (h₄ : ∠ABC = 85) :
  ∠CHX = 5 :=
sorry

end angle_CHX_is_5_degrees_l676_676545


namespace solve_equation_l676_676430

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) : (2 / x = 1 / (x + 1)) → x = -2 :=
begin
  sorry
end

end solve_equation_l676_676430


namespace divisibility_properties_l676_676088

theorem divisibility_properties (a b : ℤ) (k : ℕ) :
  (¬(a + b ∣ a^(2*k) + b^(2*k)) ∧ ¬(a - b ∣ a^(2*k) + b^(2*k))) ∧ 
  ((a + b ∣ a^(2*k) - b^(2*k)) ∧ (a - b ∣ a^(2*k) - b^(2*k))) ∧ 
  (a + b ∣ a^(2*k + 1) + b^(2*k + 1)) ∧ 
  (a - b ∣ a^(2*k + 1) - b^(2*k + 1)) := 
by sorry

end divisibility_properties_l676_676088


namespace polynomial_value_at_neg4_is_220_l676_676973

noncomputable def polynomial_value_at_neg4 : ℤ :=
  let f := λ x : ℤ, 3 * x ^ 6 + 5 * x ^ 5 + 6 * x ^ 4 + 79 * x ^ 3 - 8 * x ^ 2 + 35 * x + 12
  f (-4)

theorem polynomial_value_at_neg4_is_220 : polynomial_value_at_neg4 = 220 :=
by 
  -- Include the proof here
  sorry

end polynomial_value_at_neg4_is_220_l676_676973


namespace solve_equation_l676_676447

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end solve_equation_l676_676447


namespace total_non_defective_cars_correct_l676_676904

-- Defining the daily production rate of Factory A
def factoryA_production (day : ℕ) : ℕ :=
  match day with
  | 0 => 60
  | _ => 2 * factoryA_production (day - 1)
  end

-- Defining the daily production rate of Factory B
def factoryB_production (day : ℕ) : ℕ :=
  2 * factoryA_production day

-- Defining the defect rate for Factory A on Tuesday (day 1) and no defects on other days
def factoryA_defective (day : ℕ) : ℕ :=
  match day with
  | 1 => (factoryA_production day) * 5 / 100
  | _ => 0
  end

-- Defining the defect rate for Factory B on Thursday (day 3) and no defects on other days
def factoryB_defective (day : ℕ) : ℕ :=
  match day with
  | 3 => (factoryB_production day) * 3 / 100
  | _ => 0
  end

-- Calculating non-defective cars produced by Factory A
def non_defective_factoryA (day : ℕ) : ℕ :=
  factoryA_production day - factoryA_defective day

-- Calculating non-defective cars produced by Factory B
def non_defective_factoryB (day : ℕ) : ℕ :=
  factoryB_production day - factoryB_defective day

-- Total non-defective cars produced by both factories over the week
def total_non_defective_cars : ℕ :=
  (List.range 5).sum (λ day => non_defective_factoryA day + non_defective_factoryB day)

-- Proof statement (no proof provided, just the statement)
theorem total_non_defective_cars_correct : total_non_defective_cars = 5545 :=
  by sorry

end total_non_defective_cars_correct_l676_676904


namespace negate_prop_p_is_false_l676_676072

/--  For all x ∈ [0, +∞), (log_3 2)^x ≤ 1 holds -/
def prop_p (x : ℝ) := 0 ≤ x → x ∈ Set.Ici 0 → (Real.log 3 2) ^ x ≤ 1

/--  Proposition negation p is a false proposition given 0 < log_3 2 < 1 -/
theorem negate_prop_p_is_false : (¬ ∀ x, prop_p x) = false :=
by
  sorry

end negate_prop_p_is_false_l676_676072


namespace probability_at_least_one_hit_l676_676239

theorem probability_at_least_one_hit (pA pB pC : ℝ) (hA : pA = 0.7) (hB : pB = 0.5) (hC : pC = 0.4) : 
  (1 - ((1 - pA) * (1 - pB) * (1 - pC))) = 0.91 :=
by
  sorry

end probability_at_least_one_hit_l676_676239


namespace lottery_probability_meaning_l676_676863

theorem lottery_probability_meaning :
  (∀ (n : ℕ), n > 1000 → (∃ (p : ℚ), p = 1/1000 → 
  (∀ (k : ℕ), k = 1 → (∀ (event : ℕ → Prop), 
  (event k) = (event k) * p)))) → 
  (∃ (meaning : String), meaning = "The probability of winning by buying one lottery ticket is 1/1000") := 
begin
  sorry
end

end lottery_probability_meaning_l676_676863


namespace andrea_height_l676_676500

-- Definitions
variable (heightOfTree shadowOfTree heightOfAndrea shadowOfAndrea : ℝ)
variable (ratio heightOfTree shadowOfTree shadowOfAndrea : ℝ)

-- Assumptions
axiom height_of_tree : heightOfTree = 60 * 12  -- converting feet to inches
axiom shadow_of_tree : shadowOfTree = 15 * 12  -- converting feet to inches
axiom shadow_of_andrea : shadowOfAndrea = 20
axiom ratio : heightOfTree / shadowOfTree = 4

-- Proof statement
theorem andrea_height : heightOfAndrea = 80 :=
by
  sorry

end andrea_height_l676_676500


namespace inequality_solution_l676_676399

theorem inequality_solution {x : ℝ} :
  (12 * x^2 + 24 * x - 75) / ((3 * x - 5) * (x + 5)) < 4 ↔ -5 < x ∧ x < 5 / 3 := by
  sorry

end inequality_solution_l676_676399


namespace even_function_l676_676207

-- Definition of f and F with the given conditions
variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (x : ℝ)

-- Condition that x is in the interval (-a, a)
def in_interval (a x : ℝ) : Prop := x > -a ∧ x < a

-- Definition of F(x)
def F (x : ℝ) : ℝ := f x + f (-x)

-- The proposition that we want to prove
theorem even_function (h : in_interval a x) : F f x = F f (-x) :=
by
  unfold F
  sorry

end even_function_l676_676207


namespace total_soccer_games_l676_676457

theorem total_soccer_games (months : ℕ) (games_per_month : ℕ) (h_months : months = 3) (h_games_per_month : games_per_month = 9) : months * games_per_month = 27 :=
by
  sorry

end total_soccer_games_l676_676457


namespace tax_calculation_l676_676481

variable (winnings : ℝ) (processing_fee : ℝ) (take_home : ℝ)
variable (tax_percentage : ℝ)

def given_conditions : Prop :=
  winnings = 50 ∧ processing_fee = 5 ∧ take_home = 35

def to_prove : Prop :=
  tax_percentage = 20

theorem tax_calculation (h : given_conditions winnings processing_fee take_home) : to_prove tax_percentage :=
by
  sorry

end tax_calculation_l676_676481


namespace parallel_case_perpendicular_case_l676_676258

noncomputable def a : ℝ × ℝ := (-2, 5)
noncomputable def b : ℝ × ℝ := (1, -1)
noncomputable def c (λ : ℝ) : ℝ × ℝ := (6, λ)

theorem parallel_case (λ : ℝ) (h : 2 * a + b = 6 * λ) : λ = -18 := by
  sorry

theorem perpendicular_case (λ : ℝ) (h : (a - 3 * b).1 * (c λ).1 + (a - 3 * b).2 * (c λ).2 = 0) : λ = 15 / 4 := by
  sorry

end parallel_case_perpendicular_case_l676_676258


namespace prob_exactly_two_hits_in_four_shots_l676_676922

-- Defining the given conditions
def prob_at_least_once_hit := 65 / 81

-- Hypothesizing that the probability of hitting the target in a single shot is p
variable {p : ℚ}

-- The equation for the probability of missing all four shots based on given condition
axiom prob_miss_all_four_shots : (1 - p) ^ 4 = 16 / 81

-- Stating the Lean theorem for the proof
theorem prob_exactly_two_hits_in_four_shots (h : (1 - p) ^ 4 = 16 / 81) : 6 * (p ^ 2) * ((1 - p) ^ 2) = 8 / 27 :=
by
  have p_value : p = 1 / 3 := 
    sorry
  calc
    6 * ((1 / 3) ^ 2) * ((2 / 3) ^ 2)
        = 6 * (1 / 9) * (4 / 9) : by sorry
    ... = 6 * (1 / 9) * (4 / 9) : by sorry
    ... = (6 * 4) / 81          : by sorry
    ... = 24 / 81               : by sorry
    ... = 8 / 27                : by sorry

end prob_exactly_two_hits_in_four_shots_l676_676922


namespace profit_percentage_l676_676534

theorem profit_percentage (CP SP : ℕ) (h1 : CP * 155 = SP * 120) : 
  (SP - CP * 120) / (CP * 120) * 100 = 29.17 := 
by sorry

end profit_percentage_l676_676534


namespace sqrt_sub_sqrt_frac_eq_l676_676909

theorem sqrt_sub_sqrt_frac_eq : (Real.sqrt 3) - (Real.sqrt (1 / 3)) = (2 * Real.sqrt 3) / 3 := 
by 
  sorry

end sqrt_sub_sqrt_frac_eq_l676_676909


namespace angle_MKN_pi_div_2_l676_676831

-- Defining the basic geometrical context
variables {A B C D M N K : Point}

-- Axiomatic definitions
axiom intersect_circles (c₁ c₂ : Circle) (h : c₁ ≠ c₂) : ∃ A B, A ≠ B ∧ A ∈ c₁ ∧ A ∈ c₂ ∧ B ∈ c₁ ∧ B ∈ c₂
axiom line_intersects_circles (A : Point) (c₁ c₂ : Circle) (p : Line) (h₁ : A ∈ p) (h₂ : c₁ ≠ c₂) : ∃ C D, A ≠ C ∧ A ≠ D ∧ C ≠ D ∧ C ∈ c₁ ∧ D ∈ c₂ ∧ C ∈ p ∧ D ∈ p
axiom arc_midpoint (C B : Point) (c : Circle) (A : Point) (h₁ : C ≠ B) (h₂ : C ∈ c) (h₃ : B ∈ c) (h₄ : ¬A ∈ arc C B) : ∃ M, M ≠ A ∧ is_midpoint_arc M C B
axiom segment_midpoint (C D : Point) (h : C ≠ D) : ∃ K, is_midpoint K C D

-- Formalizing the problem
theorem angle_MKN_pi_div_2 (c₁ c₂ : Circle) (p : Line) (h : c₁ ≠ c₂) : 
  let ⟨A, B, h₁, h₂, h₃, h₄, h₅⟩ := intersect_circles c₁ c₂ h in 
  let ⟨C, D, h₆, h₇, h₈, h₉, h₁₀, h₁₁, h₁₂⟩ := line_intersects_circles A c₁ c₂ p h₂ h₃ h₄ h₅ in 
  let ⟨M, h₁₃, h₁₄⟩ := arc_midpoint C B c₁ A h₆ h₉ h₄ h₁₀ in 
  let ⟨N, h₁₅, h₁₆⟩ := arc_midpoint B D c₂ A h₁₁ h₇ h₄ h₁₂ in 
  let ⟨K, h₁₇⟩ := segment_midpoint C D h₈ in 
  ∃ (M N K : Point), angle M K N = π / 2 := sorry

end angle_MKN_pi_div_2_l676_676831


namespace volume_in_30_minutes_l676_676405

-- Define the conditions
def rate_of_pumping := 540 -- gallons per hour
def time_in_hours := 30 / 60 -- 30 minutes as a fraction of an hour

-- Define the volume pumped in 30 minutes
def volume_pumped := rate_of_pumping * time_in_hours

-- State the theorem
theorem volume_in_30_minutes : volume_pumped = 270 := by
  sorry

end volume_in_30_minutes_l676_676405


namespace liters_to_pints_l676_676230

-- Define the condition that 0.75 liters is approximately 1.575 pints
def conversion_factor : ℝ := 1.575 / 0.75

-- State that given 0.75 liters is 1.575 pints, 1.5 liters should be 3.15 pints
theorem liters_to_pints (h : 0.75 = 1.575 / conversion_factor) : 1.5 = 2 * 1.575 / conversion_factor :=
by {
  sorry -- Proof part is omitted
}

end liters_to_pints_l676_676230


namespace parabola_line_intersection_l676_676251

theorem parabola_line_intersection (A B F : ℝ × ℝ) (k : ℝ) 
  (hF : F = (1, 0)) 
  (parabola : ∀ (p : ℝ × ℝ), p ∈ A ∨ p ∈ B → p.2 ^ 2 = 4 * p.1) 
  (line : ∀ (p : ℝ × ℝ), p ∈ A ∨ p ∈ B → p.2 = k * (p.1 - 1)) 
  (dist_product : (dist F A) * (dist F B) = 6) : 
  dist A B = 6 := 
by
  sorry

end parabola_line_intersection_l676_676251


namespace v_3003_eq_3_l676_676013

def g (x : ℕ) : ℕ := 
  if x = 1 then 5 
  else if x = 2 then 3 
  else if x = 3 then 1 
  else if x = 4 then 2 
  else if x = 5 then 4 
  else 0 -- assume 0 for out of bounds for the sake of the definition 

def v : ℕ → ℕ
| 0       := 5
| (n + 1) := g (v n)

theorem v_3003_eq_3 : v 3003 = 3 :=
sorry

end v_3003_eq_3_l676_676013


namespace solve_system_l676_676172

-- Define the parameters u and v
variables {u v : ℤ}

-- Define the conditions
def cond1 := 5 * u = -7 - 2 * v
def cond2 := 3 * u = 4 * v - 25

-- State the theorem to prove the ordered pair (u, v) = (-3, 4) satisfies the conditions
theorem solve_system : ∃ (u v : ℤ), u = -3 ∧ v = 4 ∧ cond1 ∧ cond2 := by
  -- Provide some structure for the proof without solving it
  use (-3, 4)
  split; sorry

end solve_system_l676_676172


namespace correct_option_is_D_l676_676067

-- Define the conditions for each option calculation.
def option_A (a b : ℕ) : Prop := (a^2 * b)^2 = a^2 * b^2
def option_B (a : ℕ) : Prop := a^6 / a^2 = a^3
def option_C (x y : ℕ) : Prop := (3 * x * y^2)^2 = 6 * x^2 * y^4
def option_D (m : ℕ) : Prop := (-m)^7 / (-m)^2 = -m^5

-- State the theorem to prove that option D is correct.
theorem correct_option_is_D (m : ℕ) : option_D m :=
by 
  sorry

end correct_option_is_D_l676_676067


namespace dandelion_plots_l676_676015

open Nat

theorem dandelion_plots:
  let grid_size := 8
  let total_plots := grid_size * grid_size
  let initial_distribution : ℕ → ℕ → ℕ := λ i j => if i = 0 ∧ j = 0 then 19 else (if i = 7 ∧ j = 7 then 6 else 0)
  (∀ i j, 0 ≤ i ∧ i < grid_size ∧ 0 ≤ j ∧ j < grid_size → 
    ((∀ i' j', (i', j') ∈ [(i-1, j), (i+1, j), (i, j-1), (i, j+1)] → abs((initial_distribution i j) - (initial_distribution i' j')) = 1)) →
  {count x // x ∈ (range grid_size × range grid_size) ∧ initial_distribution x.1 x.2 = 19}.card ∈ ({1, 2} : set ℕ)
  :=
  sorry

end dandelion_plots_l676_676015


namespace sector_max_area_l676_676992

-- Define the problem conditions
variables (α : ℝ) (R : ℝ)
variables (h_perimeter : 2 * R + R * α = 40)
variables (h_positive_radius : 0 < R)

-- State the theorem
theorem sector_max_area (h_alpha : α = 2) : 
  1/2 * α * (40 - 2 * R) * R = 100 := 
sorry

end sector_max_area_l676_676992


namespace fixed_point_of_parabolas_l676_676740

theorem fixed_point_of_parabolas (m : ℝ) : (m^2 + m + 1) * 1^2 - 2 * (m^2 + 1) * 1 + m^2 - m + 1 = 1 := 
by
  calc
    (m^2 + m + 1) * 1^2 - 2 * (m^2 + 1) * 1 + m^2 - m + 1 = (m^2 + m + 1) - 2 * (m^2 + 1) + (m^2 - m + 1) : by rw [mul_one, one_mul]
    ... = m^2 + m + 1 - 2 * m^2 - 2 + m^2 - m + 1  : by ring
    ... = 1 : by ring

end fixed_point_of_parabolas_l676_676740


namespace dozens_of_golf_balls_l676_676829

theorem dozens_of_golf_balls (total_balls : ℕ) (dozen_size : ℕ) (h1 : total_balls = 156) (h2 : dozen_size = 12) : total_balls / dozen_size = 13 :=
by
  have h_total : total_balls = 156 := h1
  have h_size : dozen_size = 12 := h2
  sorry

end dozens_of_golf_balls_l676_676829


namespace max_value_a4a7_l676_676226

variable {a : ℕ → ℝ} (d : ℝ)

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d

theorem max_value_a4a7 (h_arith : arithmetic_seq a d) (h_a6 : a 6 = 4) : 
  ∃ M : ℝ, M = 18 ∧
    ∀ d : ℝ, (a 4 * a 7 = (4 - 2 * d) * (4 + d) → (4 - 2 * d) * (4 + d) ≤ M) := 
by 
  sorry

end max_value_a4a7_l676_676226


namespace count_true_statements_if_x_sq_gt_y_sq_then_x_gt_y_l676_676655

theorem count_true_statements_if_x_sq_gt_y_sq_then_x_gt_y :
  let original := ∀ (x y : ℝ), x^2 > y^2 → x > y
  let converse := ∀ (x y : ℝ), x > y → x^2 > y^2
  let inverse := ∀ (x y : ℝ), x^2 ≤ y^2 → x ≤ y
  let contrapositive := ∀ (x y : ℝ), x ≤ y → x^2 ≤ y^2
  (Nat.add (Nat.add (if original then 1 else 0) (if converse then 1 else 0)) (Nat.add (if inverse then 1 else 0) (if contrapositive then 1 else 0))) = 2 
:= sorry

end count_true_statements_if_x_sq_gt_y_sq_then_x_gt_y_l676_676655


namespace part_a_inequality_part_b_inequality_l676_676856

theorem part_a_inequality : 
  let A := (1 + 1/2)/2
  let B := (1 + 1/2 + 1/3)/3 
  in A > B := 
by
  sorry

theorem part_b_inequality :
  let A := (∑ i in Finset.range 2017, 1/(i+1)) / 2017
  let B := (∑ i in Finset.range 2018, 1/(i+1)) / 2018
  in A > B := 
by
  sorry

end part_a_inequality_part_b_inequality_l676_676856


namespace distance_between_points_l676_676947

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 3 3 (-2) (-2) = 5 * real.sqrt 2 :=
by
  sorry

end distance_between_points_l676_676947


namespace num_possible_integers_l676_676282

theorem num_possible_integers (x : ℕ) (h : ⌈Real.sqrt x⌉ = 20) : ∃ n : ℕ, n = 39 :=
by
  have h1 : 19 < Real.sqrt x ∧ Real.sqrt x ≤ 20 := sorry
  have h2 : 361 < x ∧ x ≤ 400 := sorry
  have h3 : ∃ (a b : ℕ), 361 < a ∧ a ≤ 400 ∧ b = a - 361 ∧ b + 1 = 39 := sorry
  use 39
  exact h3.right.right
  sorry

end num_possible_integers_l676_676282


namespace distance_between_points_l676_676946

theorem distance_between_points :
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  (Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 * Real.sqrt 2) :=
by
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  sorry

end distance_between_points_l676_676946


namespace min_value_x_plus_y_l676_676201

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 4 / x + 9 / y = 1) : x + y = 25 :=
sorry

end min_value_x_plus_y_l676_676201


namespace maximum_volume_of_tetrahedron_l676_676724

-- Definitions of the problem conditions
variables {A B C P : Type} [RealType₀ A] [RealType₀ B] [RealType₀ C] [RealType₀ P]
variables (AP BP CP AB BC CA S V : ℝ)
variables (angle_APB angle_BPC angle_CPA : ℝ)
variables [fact (angle_APB = 90)] [fact (angle_BPC = 90)] [fact (angle_CPA = 90)]
variables [fact (AP + BP + CP + AB + BC + CA = S)]

-- Lean statement to prove the maximum volume of the tetrahedron
theorem maximum_volume_of_tetrahedron
  (h1 : AP = BP) 
  (h2 : BP = CP)
  (h3 : AB = BC)
  (h4 : BC = CA)
  (angle_condition : angle_APB = 90 ∧ angle_BPC = 90 ∧ angle_CPA = 90)
  (edge_sum_condition : AP + BP + CP + AB + BC + CA = S) :
  V = S^3 / (162 * (1 + real.sqrt 2)^3) :=
sorry

end maximum_volume_of_tetrahedron_l676_676724


namespace certain_value_of_101n_squared_l676_676701

theorem certain_value_of_101n_squared 
  (n : ℤ) 
  (h : ∀ (n : ℤ), 101 * n^2 ≤ 4979 → n ≤ 7) : 
  4979 = 101 * 7^2 :=
by {
  /- proof goes here -/
  sorry
}

end certain_value_of_101n_squared_l676_676701


namespace volume_of_water_needed_l676_676512

/--
A cylindrical container has a base radius of 1 cm and contains four solid iron spheres, 
each with a radius of 1/2 cm. The four spheres are in contact with each other, 
with the two at the bottom also in contact with the base of the container. 
Water is poured into the container until the water surface just covers all the iron spheres. 
Prove that the volume of water needed is (3 + sqrt 2) / 3 * π cm³.
-/
theorem volume_of_water_needed : 
  ∀ (container_radius sphere_radius : ℝ), 
  container_radius = 1 → 
  sphere_radius = 1/2 → 
  let water_volume := (π * (1 + real.sqrt 2 / 2) - 4 * π * (1/2)^3 / 3) 
  in water_volume = (3 + real.sqrt 2) / 3 * π :=
sorry

end volume_of_water_needed_l676_676512


namespace sum_intervals_ge_one_l676_676369

noncomputable def f (a : ℕ → ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i / (x + a i)

theorem sum_intervals_ge_one (a : ℕ → ℝ) (n : ℕ) (h : ∀ i j, i < j → a i > a j) :
  (∑ i in finset.range n, a i) =
    ∑ i in finset.range n,
      (classical.some (exists_interval (λ x, f a x n ≥ 1) i)).length :=
sorry

end sum_intervals_ge_one_l676_676369


namespace total_unique_plants_l676_676193

theorem total_unique_plants 
  (A B C D : Finset α)
  (hA : A.card = 600)
  (hB : B.card = 500)
  (hC : C.card = 400)
  (hD : D.card = 300)
  (hAB : (A ∩ B).card = 100)
  (hAC : (A ∩ C).card = 150)
  (hBC : (B ∩ C).card = 75)
  (hAD : (A ∩ D).card = 50)
  (hABCD : (A ∩ B ∩ C ∩ D).card = 0) :
  (A ∪ B ∪ C ∪ D).card = 1425 :=
by
  sorry

end total_unique_plants_l676_676193


namespace acute_angle_plane_ABC_alpha_l676_676719

noncomputable def angle_between_planes (C A B : Point) (alpha : Plane) : Real := sorry

theorem acute_angle_plane_ABC_alpha
  (A B C : Point)
  (alpha : Plane)
  (h_right_angle : ∠ ABC = 90°)
  (h_hypotenuse_plane : ∀ p ∈ Line (A, B), p ∈ alpha)
  (h_point_outside_plane : C ∉ alpha)
  (h_angle_AC : angle_with_plane (LineSegment.mk A C) alpha = 30°)
  (h_angle_BC : angle_with_plane (LineSegment.mk B C) alpha = 45°) :
  angle_between_planes (LineSegment.mk A C) (LineSegment.mk C B) alpha = 60° :=
  sorry

end acute_angle_plane_ABC_alpha_l676_676719


namespace find_a_l676_676758

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 2

theorem find_a (a : ℝ) (h : (3 * a * (-1 : ℝ)^2) = 3) : a = 1 :=
by
  sorry

end find_a_l676_676758


namespace relay_race_total_distance_l676_676779

def speed_distances := (6, 2, 8, 3.5, 10, 4, 12, 2.5)

def distance_ralph (d1 : Float) := 2 * d1
def speed_ralph (v1 v2 v3 v4 : Float) := (v1 + v2 + v3 + v4) / 4

def total_distance (d1 d2 d3 d4 dR : Float) := d1 + d2 + d3 + d4 + dR

theorem relay_race_total_distance
  (d1 d2 d3 d4 : Float)
  (v1 v2 v3 v4 : Float)
  (h_d1 : d1 = 2)
  (h_d2 : d2 = 3.5)
  (h_d3 : d3 = 4)
  (h_d4 : d4 = 2.5)
  (h_v1 : v1 = 6)
  (h_v2 : v2 = 8)
  (h_v3 : v3 = 10)
  (h_v4 : v4 = 12) :
  total_distance d1 d2 d3 d4 (distance_ralph d1) = 16 :=
by
  have h_dR : distance_ralph d1 = 4 := by simp [distance_ralph, h_d1]
  simp [total_distance, h_d1, h_d2, h_d3, h_d4, h_dR]
  sorry

end relay_race_total_distance_l676_676779


namespace problem1_solution_correct_problem2_solution_correct_l676_676085

def problem1 (x : ℤ) : Prop := (x - 1) ∣ (x + 3)
def problem2 (x : ℤ) : Prop := (x + 2) ∣ (x^2 + 2)
def solution1 (x : ℤ) : Prop := x = -3 ∨ x = -1 ∨ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5
def solution2 (x : ℤ) : Prop := x = -8 ∨ x = -5 ∨ x = -4 ∨ x = -3 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 4

theorem problem1_solution_correct : ∀ x: ℤ, problem1 x ↔ solution1 x := by
  sorry

theorem problem2_solution_correct : ∀ x: ℤ, problem2 x ↔ solution2 x := by
  sorry

end problem1_solution_correct_problem2_solution_correct_l676_676085


namespace count_integers_satisfy_inequality_l676_676669

theorem count_integers_satisfy_inequality : 
  {x : ℤ | (x - 1)^2 ≤ 9}.toFinset.card = 7 := 
by 
  sorry

end count_integers_satisfy_inequality_l676_676669


namespace find_b_l676_676413

theorem find_b (x : ℝ) (b : ℝ) :
  (3 * x + 9 = 0) → (2 * b * x - 15 = -5) → b = -5 / 3 :=
by
  intros h1 h2
  sorry

end find_b_l676_676413


namespace expression_equality_l676_676263

theorem expression_equality (x : ℝ) (h : x > 0) :
  let expr1 := 2^(x + 1),
      expr2 := (2^(x + 1))^2,
      expr3 := 2^x * 2^x,
      expr4 := 4^x,
      given_expr := 2^x + 2^x
  in (expr1 = given_expr ∧ expr2 ≠ given_expr ∧ expr3 ≠ given_expr ∧ expr4 ≠ given_expr) :=
by
  sorry

end expression_equality_l676_676263


namespace max_pairs_plane_l676_676489

theorem max_pairs_plane (n : ℕ) (points : Fin n → ℝ × ℝ)
(hdist : ∀ i j, (Real.dist (points i) (points j) ≤ 1)) :
  ∃ pairs : Fin n → Fin n × Fin n, (∀ i, Real.dist (points (pairs i).1) (points (pairs i).2) = 1) → n = n :=
sorry

end max_pairs_plane_l676_676489


namespace area_bounded_by_curves_l676_676347

theorem area_bounded_by_curves {r : ℝ} (hr : r > 0) :
  let C1 := λ x : ℝ, 2 * x^2 / (x^2 + 1),
      C2 := λ x : ℝ, (sqrt (r^2 - x^2)),
      perpendicular (tangent1 tangent2 : ℝ → ℝ) (x : ℝ) := 
        (tangent1 x) * (tangent2 x) = -1,
      bounded_area := λ : ℝ, 
        2 * (∫ x in 0..1, (C2 x - C1 x))
  in bounded_area = (real.pi / 2) - 1 :=
by sorry

end area_bounded_by_curves_l676_676347


namespace alyosha_max_gain_one_blue_cube_l676_676896

def max_gain_one_blue_cube (m : ℕ) : ℝ :=
  (2^m : ℝ) / m

theorem alyosha_max_gain_one_blue_cube :
  max_gain_one_blue_cube 100 = (2^100 : ℝ) / 100 :=
by sorry

end alyosha_max_gain_one_blue_cube_l676_676896


namespace integer_solution_count_l676_676670

theorem integer_solution_count : 
  {x : ℤ | (x - 1)^2 ≤ 9}.to_finset.card = 7 := 
by 
  sorry

end integer_solution_count_l676_676670


namespace compoundY_amount_l676_676339

theorem compoundY_amount (Vx Vy Vt : ℝ) (r : Vt = Vx + Vy) (Vd : ℝ) (Hd : Vd = 0.90)
  (Rx : Vy / Vt = 0.01 / 0.06) : 
  Vy * (Vd / Vt) = 0.15 :=
by sorry

#eval compoundY_amount 0.05 0.01 0.06 r 0.90 Hd (by sorry)

end compoundY_amount_l676_676339


namespace solve_equation_l676_676449

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end solve_equation_l676_676449


namespace semicircle_area_l676_676533

theorem semicircle_area (a b : ℝ) (h1 : a = 1) (h2 : b = real.sqrt 3) :
  let d := real.sqrt (a^2 + b^2), r := d / 2 in
  ((π * r^2) / 2) = (π / 2) := 
by
  sorry

end semicircle_area_l676_676533


namespace series_sum_l676_676568

theorem series_sum : 
  (∑ k in Finset.range 14 \ (\Finset.singleton 0 ∪ Finset.singleton 1), 2 * (k + 1) * (1 - (1 / (k + 1)))) = 210 := 
by {
  sorry
}

end series_sum_l676_676568


namespace roots_sum_prod_eq_l676_676420

theorem roots_sum_prod_eq (p q : ℤ) (h1 : p / 3 = 9) (h2 : q / 3 = 20) : p + q = 87 :=
by
  sorry

end roots_sum_prod_eq_l676_676420


namespace surjective_functions_l676_676359

noncomputable def g (m n : ℕ) : ℕ :=
  ∑ k in finset.range (n + 1).filter (λ k, k > 0), ((-1) ^ (n - k)) * nat.choose n k * k ^ m

theorem surjective_functions (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  g m n = ∑ k in finset.range (n + 1).filter (λ k, k > 0), ((-1) ^ (n - k)) * nat.choose n k * k ^ m :=
sorry

end surjective_functions_l676_676359


namespace all_such_functions_l676_676368

theorem all_such_functions (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ) (u v : ℝ), 1 < x ∧ 1 < y ∧ 0 < u ∧ 0 < v →
  f (x^u * y^v) ≤ (f x)^(1/u) * (f y)^(1/v)) :
  ∃ k > 0, ∀ x, 1 < x → f x = x^k :=
begin
  sorry,
end

end all_such_functions_l676_676368


namespace billy_tickets_used_l676_676139

-- Definitions for the number of rides and cost per ride
def ferris_wheel_rides : Nat := 7
def bumper_car_rides : Nat := 3
def ticket_per_ride : Nat := 5

-- Total number of rides
def total_rides : Nat := ferris_wheel_rides + bumper_car_rides

-- Total tickets used
def total_tickets : Nat := total_rides * ticket_per_ride

-- Theorem stating the number of tickets Billy used in total
theorem billy_tickets_used : total_tickets = 50 := by
  sorry

end billy_tickets_used_l676_676139


namespace distance_center_circle_to_hyperbola_l676_676730

noncomputable def hyperbola_circle_distance : ℝ := 2 * Real.sqrt 2

theorem distance_center_circle_to_hyperbola 
  (Hx : ∀ (x : ℝ) (y : ℝ), x^2 - y^2 = 16 → 
  (x, y) = (4, 0) ∨ (x, y) = (4 * Real.sqrt 2, 0) ∨ (x, y) = (-4, 0) ∨ (x, y) = (-4 * Real.sqrt 2, 0))
  (V : ∀ (x : ℝ) (y : ℝ), x = 4 ∨ x = -4)
  (F : ∀ (x : ℝ) (y : ℝ), (x, y) = (4 * Real.sqrt 2, 0) ∨ (x, y) = (-4 * Real.sqrt 2, 0))
  (C : ∀ (x : ℝ) (y : ℝ), x^2 - y^2 = 16 ∧ 
  ∃ (hx : ℝ) (hy : ℝ), (hx, hy) = (2 * Real.sqrt 2, 0) ∧ 
  x^2 + y^2 = hx^2 + hy^2) :
  distance_center_circle_to_hyperbola = 2 * Real.sqrt 2 :=
by
  sorry

end distance_center_circle_to_hyperbola_l676_676730


namespace base12_addition_example_l676_676541

theorem base12_addition_example : 
  (5 * 12^2 + 2 * 12^1 + 8 * 12^0) + (2 * 12^2 + 7 * 12^1 + 3 * 12^0) = (7 * 12^2 + 9 * 12^1 + 11 * 12^0) :=
by sorry

end base12_addition_example_l676_676541


namespace lucas_total_assignments_l676_676117

theorem lucas_total_assignments : 
  ∃ (total_assignments : ℕ), 
  (∀ (points : ℕ), 
    (points ≤ 10 → total_assignments = points * 1) ∧
    (10 < points ∧ points ≤ 20 → total_assignments = 10 * 1 + (points - 10) * 2) ∧
    (20 < points ∧ points ≤ 30 → total_assignments = 10 * 1 + 10 * 2 + (points - 20) * 3)
  ) ∧
  total_assignments = 60 :=
by
  sorry

end lucas_total_assignments_l676_676117


namespace find_m_l676_676217

open Real

variables (m : ℝ)
def vector_a := (1, 2)
def vector_b := (4, 2)
def vector_c := (m * 1 + 4, m * 2 + 2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_m :
  let a := vector_a,
      b := vector_b,
      c := vector_c m in
  (dot_product c a / (magnitude c * magnitude a)) =
  (dot_product c b / (magnitude c * magnitude b)) →
  m = 2 := 
sorry

end find_m_l676_676217


namespace solve_equation_l676_676427

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end solve_equation_l676_676427


namespace decreasing_function_and_sum_property_l676_676700

noncomputable def f (x : ℝ) : ℝ := x / (x - 1)

theorem decreasing_function_and_sum_property
  (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_log : Real.log10 (x + y) = Real.log10 x + Real.log10 y) 
  : 
  (x + y ≥ 4) ∧ ∀ x1 x2 > 1, x1 > x2 → f x1 < f x2 :=
by
  sorry

end decreasing_function_and_sum_property_l676_676700


namespace solve_fractional_eq_l676_676440

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end solve_fractional_eq_l676_676440


namespace add_base7_l676_676127

-- Define the two numbers in base 7 to be added.
def number1 : ℕ := 2 * 7 + 5
def number2 : ℕ := 5 * 7 + 4

-- Define the expected result in base 7.
def expected_sum : ℕ := 1 * 7^2 + 1 * 7 + 2

theorem add_base7 :
  let sum : ℕ := number1 + number2
  sum = expected_sum := sorry

end add_base7_l676_676127


namespace dice_probability_theorem_l676_676186

def at_least_three_same_value_probability (num_dice : ℕ) (num_sides : ℕ) : ℚ :=
  if num_dice = 5 ∧ num_sides = 10 then
    -- Calculating the probability
    (81 / 10000) + (9 / 20000) + (1 / 10000)
  else
    0

theorem dice_probability_theorem :
  at_least_three_same_value_probability 5 10 = 173 / 20000 :=
by
  sorry

end dice_probability_theorem_l676_676186


namespace tom_books_problem_l676_676460

theorem tom_books_problem 
  (original_books : ℕ)
  (books_sold : ℕ)
  (books_bought : ℕ)
  (h1 : original_books = 5)
  (h2 : books_sold = 4)
  (h3 : books_bought = 38) : 
  original_books - books_sold + books_bought = 39 :=
by
  sorry

end tom_books_problem_l676_676460


namespace distance_points_l676_676933

-- Definition of distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Points
def point1 : ℝ × ℝ := (3, 3)
def point2 : ℝ × ℝ := (-2, -2)

-- Main theorem
theorem distance_points : distance point1 point2 = 5 * Real.sqrt 2 :=
by
  sorry

end distance_points_l676_676933


namespace exists_prime_dividing_power_minus_one_not_dividing_n_l676_676345

theorem exists_prime_dividing_power_minus_one_not_dividing_n 
  (n : ℕ) (h1 : n > 3) (h2 : n % 2 = 1)
  (p : ℕ → ℕ) (α : ℕ → ℕ) (k : ℕ) 
  (h3 : n = ∏ i in finset.range k, p i ^ α i) 
  (m : ℕ)
  (h4 : m = n * ∏ i in finset.range k, (1 - 1 / (p i))) :
  ∃ P : ℕ, P ∣ 2 ^ m - 1 ∧ ¬ P ∣ n :=
by
  sorry

end exists_prime_dividing_power_minus_one_not_dividing_n_l676_676345


namespace cubic_root_indentity_sum_find_abc_sum_l676_676009

theorem cubic_root_indentity_sum (a b c : ℕ) (h : a = 81 ∧ b = 9 ∧ c = 8) :
  ∃ x : ℝ, (8 * x^3 - 3 * x^2 - 3 * x - 1 = 0) ∧ x = (∑ y in []: list ℝ, (√[3] y)) / c :=
by
  sorry

theorem find_abc_sum : ℕ :=
by
  let a := 81
  let b := 9
  let c := 8
  have habc: a = 81 ∧ b = 9 ∧ c = 8 := by simp
  exact a + b + c

end cubic_root_indentity_sum_find_abc_sum_l676_676009


namespace original_garden_length_l676_676006

theorem original_garden_length (x : ℝ) (area : ℝ) (reduced_length : ℝ) (width : ℝ) (length_condition : x - reduced_length = width) (area_condition : x * width = area) (given_area : area = 120) (given_reduced_length : reduced_length = 2) : x = 12 := 
by
  sorry

end original_garden_length_l676_676006


namespace simplify_sqrt_expression_l676_676397

def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem simplify_sqrt_expression :
  sqrt 18 - sqrt 8 = sqrt 2 := 
sorry

end simplify_sqrt_expression_l676_676397


namespace which_set_forms_triangle_l676_676480

def satisfies_triangle_inequality (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem which_set_forms_triangle : 
  satisfies_triangle_inequality 4 3 6 ∧ 
  ¬ satisfies_triangle_inequality 1 2 3 ∧ 
  ¬ satisfies_triangle_inequality 7 8 16 ∧ 
  ¬ satisfies_triangle_inequality 9 10 20 :=
by
  sorry

end which_set_forms_triangle_l676_676480


namespace negation_of_proposition_l676_676024

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1)) ↔
  (∃ x₀ : ℝ, x₀ ≤ 0 ∧ (x₀ + 1) * Real.exp x₀ ≤ 1) := 
sorry

end negation_of_proposition_l676_676024


namespace distance_between_points_l676_676938

theorem distance_between_points :
  ∀ (P Q : ℝ × ℝ), P = (3, 3) ∧ Q = (-2, -2) → dist P Q = 5 * real.sqrt 2 :=
begin
  sorry
end

end distance_between_points_l676_676938


namespace arithmetic_mean_equality_l676_676597

variable (x y a b : ℝ)

theorem arithmetic_mean_equality (hx : x ≠ 0) (hy : y ≠ 0) :
  (1 / 2 * ((x + a) / y + (y - b) / x)) = (x^2 + a * x + y^2 - b * y) / (2 * x * y) :=
  sorry

end arithmetic_mean_equality_l676_676597


namespace constant_term_in_expansion_l676_676206

open Real

noncomputable def n : ℝ := ∫ x in (-π/2)..(π/2), (6 * cos x - sin x)

def binomial_expansion_constant_term (n : ℕ) : ℕ :=
  9 -- since by computation the 9th term is the constant term

theorem constant_term_in_expansion :
  binomial_expansion_constant_term (∫ x in (-π/2)..(π/2), (6 * cos x - sin x)) = 9 :=
by
  sorry

end constant_term_in_expansion_l676_676206


namespace base_conversion_problem_l676_676749

theorem base_conversion_problem (n d : ℕ) (hn : 0 < n) (hd : d < 10) 
  (h1 : 3 * n^2 + 2 * n + d = 263) (h2 : 3 * n^2 + 2 * n + 4 = 253 + 6 * d) : 
  n + d = 11 :=
by
  sorry

end base_conversion_problem_l676_676749


namespace distance_between_points_eq_l676_676959

theorem distance_between_points_eq :
  let x1 := 2
  let y1 := -5
  let x2 := -8
  let y2 := 7
  let distance := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
  distance = 2 * Real.sqrt 61 :=
by
  sorry

end distance_between_points_eq_l676_676959


namespace problem_1_problem_2_l676_676699

noncomputable def is_closed_function (f : ℝ → ℝ) :=
  (∃ (a b : ℝ), a < b ∧ ∀ x ∈ set.Icc a b, set.Ico a b ⊆ set.Icc (f a) (f b) ∧ 
  ((∀ x y : ℝ, a ≤ x → x ≤ b → 
    a ≤ y → y ≤ b → x < y → f x ≤ f y) ∨ 
  (∀ x y : ℝ, a ≤ x → x ≤ b → 
    a ≤ y → y ≤ b → x < y → f y ≤ f x)))

-- Problem 1 - Function y = -x^3
theorem problem_1 : is_closed_function (λ x, -x^3) :=
sorry

-- Problem 2 - Function f(x) = x^2 / 2 - x + 1
theorem problem_2 : ¬(∃ (a b : ℝ), a < b ∧ 
  (is_closed_function (λ x, x^2 / 2 - x + 1))) :=
sorry

end problem_1_problem_2_l676_676699


namespace find_marked_price_l676_676031

theorem find_marked_price (cp : ℝ) (d : ℝ) (p : ℝ) (x : ℝ) (h1 : cp = 80) (h2 : d = 0.3) (h3 : p = 0.05) :
  (1 - d) * x = cp * (1 + p) → x = 120 :=
by
  sorry

end find_marked_price_l676_676031


namespace f_inv_of_1_solution_set_of_inequality_l676_676641

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

axiom f_decreasing : ∀ x y : ℝ, x < y → f y < f x
axiom f_A : f (-4) = 1
axiom f_B : f (0) = -1

theorem f_inv_of_1 : f_inv 1 = -4 := by
  sorry

theorem solution_set_of_inequality : {x : ℝ | abs (f (x - 2)) < 1} = set.Ioo (-2) 2 := by
  sorry

end f_inv_of_1_solution_set_of_inequality_l676_676641


namespace smallest_integer_is_17_l676_676195

theorem smallest_integer_is_17
  (a b c d : ℕ)
  (h1 : b = 33)
  (h2 : d = b + 3)
  (h3 : (a + b + c + d) = 120)
  (h4 : a ≤ b)
  (h5 : c > b)
  : a = 17 :=
sorry

end smallest_integer_is_17_l676_676195


namespace fibonacci_sum_l676_676967

def fibonacci : ℤ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n
| (n-1) := fibonacci (n+1) - fibonacci n

theorem fibonacci_sum :
    (fibonacci (-5) + fibonacci (-4) + fibonacci (-3) + fibonacci (-2) + fibonacci (-1) + fibonacci 0 + fibonacci 1 + fibonacci 2 + fibonacci 3 + fibonacci 4 + fibonacci 5) = 16 :=
by
  sorry

end fibonacci_sum_l676_676967


namespace num_true_propositions_l676_676636

variable (a b c : Type) [HasPerp a] [HasPerp b] [HasPerp c] [HasParallel a] [HasParallel b] [HasParallel c] (α : Type) [Plane α]

-- Define conditions
def condition1 : Prop := a ⊥ c ∧ b ⊥ c
def condition2 : Prop := a ⊥ α ∧ b ⊥ α
def condition3 : Prop := a ∥ α ∧ b ∥ α
def condition4 : Prop := a ⊆ α ∧ b ∥ α ∧ coplanar a b

-- Define propositions
def prop1 : Prop := a ⊥ c ∧ b ⊥ c → a ∥ b
def prop2 : Prop := a ⊥ α ∧ b ⊥ α → a ∥ b
def prop3 : Prop := a ∥ α ∧ b ∥ α → a ∥ b
def prop4 : Prop := a ⊆ α ∧ b ∥ α ∧ coplanar a b → a ∥ b

-- Statement with the conclusions
theorem num_true_propositions : 
  (¬ prop1 ∧ prop2 ∧ ¬ prop3 ∧ prop4) → 
  (1 = 1) :=
by
  intros h
  sorry

end num_true_propositions_l676_676636


namespace total_students_l676_676490

-- Definition of the problem conditions
def ratio_boys_girls : ℕ := 8
def ratio_girls : ℕ := 5
def number_girls : ℕ := 160

-- The main theorem statement
theorem total_students (b g : ℕ) (h1 : b * ratio_girls = g * ratio_boys_girls) (h2 : g = number_girls) :
  b + g = 416 :=
sorry

end total_students_l676_676490


namespace angle_CHX_l676_676547

-- Define the conditions and question
variables {A B C H X : Type}
variables [inner_product_space ℝ A]

-- A is, an acute triangle such that
def angle_BAC : ℝ := 50 
def angle_ABC : ℝ := 85 
def orthocenter_H_of_ABC : Prop := sorry  -- We assume some definition for the orthocenter H

-- Our target is to prove angle CHX
theorem angle_CHX {A B C H X : Type} [inner_product_space ℝ A]
  (h1 : ∠ BAC = 50)
  (h2 : ∠ ABC = 85)
  (h3 : orthocenter_H_of_ABC) :
  ∠ CHX = 45 :=
sorry

end angle_CHX_l676_676547


namespace prank_combinations_l676_676048

theorem prank_combinations :
  let monday_choices := 2 in
  let tuesday_choices := 3 in
  let wednesday_choices := 6 in
  let thursday_choices := 5 in
  let friday_choices := 2 in
  let saturday_choices := 1 in
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices * saturday_choices = 360 :=
by
  sorry

end prank_combinations_l676_676048


namespace determine_values_l676_676752

noncomputable def mean (s : Finset ℕ) : ℚ :=
  let products := (s.powerset.filter (λ x : Finset ℕ, x.nonempty)).map (λ t : Finset ℕ, (t.prod id).toℚ)
  (products.sum / products.card)

theorem determine_values :
  ∃ (r : ℕ) (a : ℕ → ℕ) (ar1 : ℕ), 
    r > 1 ∧ 
    mean (Finset.range r.image a) = 13 ∧ 
    mean ((Finset.range r.image a).insert ar1) = 49 ∧ 
    ar1 = 7 :=
  sorry

end determine_values_l676_676752


namespace incorrect_regression_intercept_l676_676003

theorem incorrect_regression_intercept (points : List (ℕ × ℝ)) (h_points : points = [(1, 0.5), (2, 0.8), (3, 1.0), (4, 1.2), (5, 1.5)]) :
  ¬ (∃ (a : ℝ), a = 0.26 ∧ ∀ x : ℕ, x ∈ ([1, 2, 3, 4, 5] : List ℕ) → (∃ y : ℝ, y = 0.24 * x + a)) := sorry

end incorrect_regression_intercept_l676_676003


namespace car_price_is_5_l676_676200

variable (numCars : ℕ) (totalEarnings legoCost carCost : ℕ)

-- Conditions
axiom h1 : numCars = 3
axiom h2 : totalEarnings = 45
axiom h3 : legoCost = 30
axiom h4 : totalEarnings - legoCost = 15
axiom h5 : (totalEarnings - legoCost) / numCars = carCost

-- The proof problem statement
theorem car_price_is_5 : carCost = 5 :=
  by
    -- Here the proof steps would be filled in, but are not required for this task.
    sorry

end car_price_is_5_l676_676200


namespace probability_closer_to_5_than_1_l676_676530

open MeasureTheory ProbabilityTheory

noncomputable def prob_closer_to_5_than_1 : ℝ :=
  let interval : Set ℝ := Icc 0 6
  let closer_to_5 := {x | 3 ≤ x ∧ x ≤ 6}
  let p := measure_theory.volume
  (p closer_to_5 / p interval).to_real

theorem probability_closer_to_5_than_1 {prob_closer_to_5_than_1 = 0.5 : ℝ} : 
  by 
    let interval : Set ℝ := Icc 0 6
    let closer_to_5 := {x | 3 ≤ x ∧ x ≤ 6}
    let p := measure_theory.volume
    exact (p closer_to_5 / p interval).to_real = 0.5
  sorry

end probability_closer_to_5_than_1_l676_676530


namespace min_national_subsidy_min_avg_processing_cost_l676_676717

noncomputable def processing_cost (x : ℝ) : ℝ := x^2 - 50 * x + 900

noncomputable def profit (x : ℝ) : ℝ := 20 * x - processing_cost x

theorem min_national_subsidy (x : ℝ) (hx : 10 ≤ x ∧ x ≤ 15) : -75 ≤ profit x := by
  let P := profit x
  have hp : P = 70 * x - x^2 - 900 := by 
    simp [profit, processing_cost]
    ring
  sorry

noncomputable def avg_processing_cost (x : ℝ) : ℝ := (processing_cost x) / x

theorem min_avg_processing_cost : ∃ x, x = 30 ∧ avg_processing_cost x = 10 := by
  let Q := avg_processing_cost 30
  have hq : Q = 30 + 900 / 30 - 50 := by
    simp [avg_processing_cost, processing_cost]
  sorry

end min_national_subsidy_min_avg_processing_cost_l676_676717


namespace problem_solution_correct_l676_676623

noncomputable def f (x : ℝ) := x * exp (-x)

theorem problem_solution_correct :
  (∀ x, deriv f x - f x = (1 - 2 * x) * exp (-x)) ∧ 
  (f 0 = 0) ∧ 
  (
    (1 = 1) ∧ 
    (- (1 / exp 2) < k < 0) ∧ 
    (∀ x1 x2, 2 < x1 → x2 < ∞ → f ( (x1 + x2) / 2) ≤ (f x1 + f x2) / 2) ∧ 
    (exactly_two_dif_real_solutions a b)
  ) :=
by
  sorry

definition exactly_two_dif_real_solutions (a b : ℝ) : Prop := 
  (f a = f b) ∧ (a ≠ b) ∧ floor (exp a) = 2 ∧ floor (exp b) = 2

end problem_solution_correct_l676_676623


namespace maurice_needs_7_letters_l676_676459
noncomputable def prob_no_job (n : ℕ) : ℝ := (4 / 5) ^ n

theorem maurice_needs_7_letters :
  ∃ n : ℕ, (prob_no_job n) ≤ 1 / 4 ∧ n = 7 :=
by
  sorry

end maurice_needs_7_letters_l676_676459


namespace correct_number_incorrectly_read_as_25_l676_676007

theorem correct_number_incorrectly_read_as_25 :
  (∀ (xs : List ℝ), xs.length = 10 ∧ (10 * 16 = List.sum (xs.map (λ x, if x = 25 then 25 else x))) ∧ 10 * 18 = List.sum xs → (∃ (x : ℝ), List.sum (xs.map (λ y, if x = 25 then x else y)) = 180 - (25 - x))) :=
by
  sorry

end correct_number_incorrectly_read_as_25_l676_676007


namespace initial_kids_count_l676_676824

-- Define the initial number of kids as a variable
def init_kids (current_kids join_kids : Nat) : Nat :=
  current_kids - join_kids

-- Define the total current kids and kids joined
def current_kids : Nat := 36
def join_kids : Nat := 22

-- Prove that the initial number of kids was 14
theorem initial_kids_count : init_kids current_kids join_kids = 14 :=
by
  -- Proof skipped
  sorry

end initial_kids_count_l676_676824


namespace sufficient_condition_l676_676057

theorem sufficient_condition (A B C D : Prop) (h : C → D): C → (A > B) := 
by 
  sorry

end sufficient_condition_l676_676057


namespace band_members_count_l676_676502

theorem band_members_count (earn_per_member_per_gig : ℕ) (total_earned : ℕ) (gigs : ℕ) (per_member_earning : ℕ) :
    earn_per_member_per_gig = 20 → total_earned = 400 → gigs = 5 → per_member_earning = total_earned / gigs →
    (per_member_earning / earn_per_member_per_gig) = 4 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end band_members_count_l676_676502


namespace parallel_tangent_lines_l676_676606

noncomputable def is_tangent (line : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ p : ℝ × ℝ, (circle p.1 p.2) ∧ ∀ q : ℝ × ℝ, line q.1 q.2 → q = p

theorem parallel_tangent_lines (b : ℝ) :
  (∀ x y, 2*x - y + 1 = 0 → false) ∧ 
  (is_tangent (λ x y, 2*x - y + b = 0) (λ x y, x^2 + y^2 = 5)) → 
  (b = 5 ∨ b = -5) :=
sorry

end parallel_tangent_lines_l676_676606


namespace probability_at_least_one_consonant_l676_676083

def letters := ["k", "h", "a", "n", "t", "k", "a", "r"]
def consonants := ["k", "h", "n", "t", "r"]
def vowels := ["a", "a"]

def num_letters := 7
def num_consonants := 5
def num_vowels := 2

def probability_no_consonants : ℚ := (num_vowels / num_letters) * ((num_vowels - 1) / (num_letters - 1))

def complement_rule (p: ℚ) : ℚ := 1 - p

theorem probability_at_least_one_consonant :
  complement_rule probability_no_consonants = 20/21 :=
by
  sorry

end probability_at_least_one_consonant_l676_676083


namespace number_of_possible_integers_l676_676277

theorem number_of_possible_integers (x: ℤ) (h: ⌈real.sqrt ↑x⌉ = 20) : 39 :=
  sorry

end number_of_possible_integers_l676_676277


namespace algebraic_expression_evaluation_l676_676301

theorem algebraic_expression_evaluation (x : ℝ) (h : x^2 + 3 * x - 5 = 2) : 2 * x^2 + 6 * x - 3 = 11 :=
sorry

end algebraic_expression_evaluation_l676_676301


namespace completing_square_to_simplify_eq_l676_676835

theorem completing_square_to_simplify_eq : 
  ∃ (c : ℝ), (∀ x : ℝ, x^2 - 6 * x + 4 = 0 ↔ (x - 3)^2 = c) :=
by
  use 5
  intro x
  constructor
  { intro h
    -- proof conversion process (skipped)
    sorry }
  { intro h
    -- reverse proof process (skipped)
    sorry }

end completing_square_to_simplify_eq_l676_676835


namespace evaluate_expression_l676_676598

theorem evaluate_expression : 3^5 * 6^5 * 3^6 * 6^6 = 18^11 :=
by sorry

end evaluate_expression_l676_676598


namespace smallest_n_l676_676293

noncomputable def b (n : ℕ) : ℝ :=
  if n = 0 then sin^2 (π / 60)
  else 4 * (b (n - 1)) * (1 - (b (n - 1)))

lemma eq_b0 (n : ℕ) : b n = sin^2 (2^n * π / 60) :=
by sorry

theorem smallest_n (n : ℕ) (h : ∃ (n : ℕ), b n = b 0) :
  ∃ (n : ℕ), b n = b 0 ∧ n = 26 :=
by sorry

end smallest_n_l676_676293


namespace number_with_16_multiples_between_1_and_400_is_25_l676_676815

theorem number_with_16_multiples_between_1_and_400_is_25 :
  ∃ n : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 400 → k % n = 0 → (0 < k / n ∧ k / n ≤ 16)) ∧ n = 25 := 
begin
  let n := 25,
  use n,
  split,
  { intros k h1 h2 h3,
    split,
    { exact nat.div_pos h2 (nat.gt_or_eq_of_le ((nat.mod_eq_zero_of_dvd (dvd.intro_left _ h3)).symm ▸ by simp)), },
    { have : k / n = 16, by exact nat.div_eq_of_lt (nat.mod_eq_zero_of_dvd (int.coe_nat_dvd.2 (dvd.intro_left _ h3))),
      rw this,
      exact nat.le_of_lt (nat.div_lt_of_lt_mul h2),
    },
  },
  exact rfl,
end

end number_with_16_multiples_between_1_and_400_is_25_l676_676815


namespace david_marks_in_english_l676_676159

theorem david_marks_in_english
  (marks_math : ℕ)
  (marks_phys : ℕ)
  (marks_chem : ℕ)
  (marks_bio : ℕ)
  (num_subjects : ℕ)
  (avg_marks : ℕ)
  (total_marks : ℕ := avg_marks * num_subjects)
  (known_total : total_marks = marks_math + marks_phys + marks_chem + marks_bio + e) :
  e = 61 :=
by
  have h : marks_math + marks_phys + marks_chem + marks_bio = 299 := by sorry
  have total : total_marks = 360 := by sorry
  show e = 61, from
  calc
    e = total_marks - (marks_math + marks_phys + marks_chem + marks_bio) : by sorry
    ... = 360 - 299 : by rw [total, h]
    ... = 61 : by norm_num

end david_marks_in_english_l676_676159


namespace num_distinct_factors_of_36_l676_676663

/-- Definition of the number 36. -/
def n : ℕ := 36

/-- Prime factorization of 36 is 2^2 * 3^2. -/
def prime_factors : List (ℕ × ℕ) := [(2, 2), (3, 2)]

/-- The number of distinct positive factors of 36 is 9. -/
theorem num_distinct_factors_of_36 : ∃ k : ℕ, k = 9 ∧ 
  ∀ d : ℕ, d ∣ n → d > 0 → List.mem d (List.range (n + 1)) :=
by
  sorry

end num_distinct_factors_of_36_l676_676663


namespace part_a_l676_676488

theorem part_a (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x + y = 2) : 
  (1 / x + 1 / y) ≤ (1 / x^2 + 1 / y^2) := 
sorry

end part_a_l676_676488


namespace common_point_for_parabolas_l676_676155

noncomputable def parabolas_intersect_coord_axes (p q : ℝ) (h_q : q ≠ 0) : Prop :=
  ∀ (x1 x2 : ℝ), x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 ≠ x2 ∧ ((x1 * x2 = q) ∧ (∃ r, circle_through_points (0,q) (x1,0) (x2,0) r))

theorem common_point_for_parabolas (p q : ℝ) (h_q : q ≠ 0) :
  parabolas_intersect_coord_axes p q h_q → 
  ∃ r, circle_through_points (0,q) (√q,0) (-√q,0) (0,1) :=
sorry

end common_point_for_parabolas_l676_676155


namespace solve_fractional_eq_l676_676445

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end solve_fractional_eq_l676_676445


namespace product_adjacent_terms_l676_676253

noncomputable def a : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := 2005 * a (n+1) - a n

theorem product_adjacent_terms (n : ℕ) (hn : n > 0) :
  ∃ k : ℤ, a (n+1) * a n - 1 = 2003 * k^2 :=
sorry

end product_adjacent_terms_l676_676253


namespace angle_CHX_is_5_degrees_l676_676546

theorem angle_CHX_is_5_degrees
  {A B C X Y H : Type}
  [IsAcuteTriangle A B C]
  (h₁ : Altitude AX ∧ Altitude BY)
  (h₂ : H = Orthocenter A B C) 
  (h₃ : ∠BAC = 50)
  (h₄ : ∠ABC = 85) :
  ∠CHX = 5 :=
sorry

end angle_CHX_is_5_degrees_l676_676546


namespace product_consecutive_divisible_by_factorial_l676_676756

theorem product_consecutive_divisible_by_factorial (n : ℕ) (h : n > 0) :
  (∀ m : ℕ, n! ∣ ∏ k in finset.range(n) + m + 1, (n - k)) :=
by
  sorry

end product_consecutive_divisible_by_factorial_l676_676756


namespace total_carrot_sticks_l676_676731

-- Define the number of carrot sticks James ate before and after dinner
def carrot_sticks_before_dinner : Nat := 22
def carrot_sticks_after_dinner : Nat := 15

-- Prove that the total number of carrot sticks James ate is 37
theorem total_carrot_sticks : carrot_sticks_before_dinner + carrot_sticks_after_dinner = 37 :=
  by sorry

end total_carrot_sticks_l676_676731


namespace isosceles_triangle_exists_l676_676156

theorem isosceles_triangle_exists (P h : ℝ) (hP : P > 0) (hh : h > 0) :
  ∃ (A B C : ℝ) (a b c : ℝ), 
  a = b ∧ -- sides AB and AC are equal
  a + a + c = P ∧ -- perimeter condition
  (∃ D, D between B and C ∧ AD = h) -- height condition
  :=
sorry

end isosceles_triangle_exists_l676_676156


namespace math_problem_l676_676850

theorem math_problem :
  (4 ∣ 24) ∧ (¬ (19 ∣ 209) ∧ ¬ (¬ (19 ∣ 57))) ∧ (¬ (¬ (15 ∣ 75) ∧ ¬ (15 ∣ 100))) ∧ (¬ (¬ (14 ∣ 28) ∧ ¬ (14 ∣ 56))) ∧ (9 ∣ 180) :=
by {
  -- Prove 4 | 24
  have A : 4 ∣ 24 := sorry,
  -- Prove !(19 | 209 and !(19 | 57)) (which is B is false)
  have B : ¬ (19 ∣ 209 ∧ ¬ (19 ∣ 57)) := sorry,
  -- Prove !(15 is neither a divisor of 75 nor 100) (which is C is false)
  have C : ¬ (¬ (15 ∣ 75) ∧ ¬ (15 ∣ 100)) := sorry,
  -- Prove !(14 is a divisor of 28 but not of 56) (which is D is false)
  have D : ¬ (¬ (14 ∣ 28) ∧ ¬ (14 ∣ 56)) := sorry,
  -- Prove 9 | 180
  have E : 9 ∣ 180 := sorry,
  -- Combine them all
  exact ⟨A, ⟨B, ⟨C, ⟨D, E⟩⟩⟩⟩
}

end math_problem_l676_676850


namespace find_N_l676_676810

theorem find_N (a b c N : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = a + b) : N = 272 :=
sorry

end find_N_l676_676810


namespace number_of_possible_integers_l676_676275

theorem number_of_possible_integers (x: ℤ) (h: ⌈real.sqrt ↑x⌉ = 20) : 39 :=
  sorry

end number_of_possible_integers_l676_676275


namespace sum_of_squares_consecutive_even_integers_l676_676745

theorem sum_of_squares_consecutive_even_integers (n : ℤ) :
  let t := 12 * n^2 + 8 in (t % 4 = 0) ∧ ∃ n, (12 * n^2 + 8) % 5 = 0 :=
by
  let t := 12 * n^2 + 8
  split
  · have : t % 4 = 0 := by sorry
    exact this
  · use 1
    have : (12 * 1^2 + 8) % 5 = 0 := by sorry
    exact this

end sum_of_squares_consecutive_even_integers_l676_676745


namespace largest_palindrome_times_103_not_palindrome_l676_676607

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse
  
def largest_three_digit_palindrome (k : ℕ) : ℕ :=
  let palindromes := [999, 898, 797, 696, 595, 494, 393, 292, 191] in
  palindromes.find? (λ p, p * k ≥ 100000 ∧ p * k < 1000000 ∧ ¬is_palindrome (p * k)).get_or_else 0

theorem largest_palindrome_times_103_not_palindrome :
  largest_three_digit_palindrome 103 = 999 := by
  sorry

end largest_palindrome_times_103_not_palindrome_l676_676607


namespace ducks_remaining_after_three_nights_l676_676819

def initial_ducks : ℕ := 320
def first_night_ducks (initial_ducks : ℕ) : ℕ := initial_ducks - (initial_ducks / 4)
def second_night_ducks (first_night_ducks : ℕ) : ℕ := first_night_ducks - (first_night_ducks / 6)
def third_night_ducks (second_night_ducks : ℕ) : ℕ := second_night_ducks - (second_night_ducks * 30 / 100)

theorem ducks_remaining_after_three_nights : 
  third_night_ducks (second_night_ducks (first_night_ducks initial_ducks)) = 140 :=
by
  -- Proof goes here
  sorry

end ducks_remaining_after_three_nights_l676_676819


namespace solve_equation_l676_676446

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end solve_equation_l676_676446


namespace find_value_of_b_l676_676883

noncomputable def line_through_two_points := λ x₁ y₁ x₂ y₂ : ℝ, ⟨ x₂ - x₁, y₂ - y₁ ⟩

def scale_vector (v : ℝ × ℝ) (k : ℝ) := (k * v.1, k * v.2)

theorem find_value_of_b :
  let p1 := (-3 : ℝ, 1 : ℝ)
  let p2 := (2 : ℝ, 5 : ℝ)
  let direction_vector := line_through_two_points (p1.1) (p1.2) (p2.1) (p2.2)
  let scaled_direction_vector := scale_vector direction_vector (1 / (direction_vector.1))
  in scaled_direction_vector = (1, 4 / 5) := sorry

end find_value_of_b_l676_676883


namespace scientific_notation_of_8_5_million_l676_676125

theorem scientific_notation_of_8_5_million :
  (8.5 * 10^6) = 8500000 :=
by sorry

end scientific_notation_of_8_5_million_l676_676125


namespace cartesian_equiv_min_distance_eq_sqrt5_l676_676318

-- Definitions from the conditions
def polar_eq (ρ θ : ℝ) : Prop := 2 * ρ * sin θ + ρ * cos θ = 10
def parametric_eq (α : ℝ) : ℝ × ℝ := (3 * cos α, 2 * sin α)

-- Cartesian equation from the parametric equations of C1
def cartesian_eq_C1 (x y : ℝ) : Prop := (x ^ 2 / 9) + (y ^ 2 / 4) = 1

-- Cartesian equation from the polar coordinates of C
def cartesian_eq_C (x y : ℝ) : Prop := x + 2 * y - 10 = 0

-- Minimum distance from a point on C1 to the line derived from C
def minimum_distance (α : ℝ) : ℝ :=
  let x := 3 * cos α
  let y := 2 * sin α
  (abs (x + 2 * y - 10)) / (sqrt 5)

-- Proof obligations
theorem cartesian_equiv {x y : ℝ} :
  ∃ α, parametric_eq α = (x, y) → cartesian_eq_C1 x y :=
sorry

theorem min_distance_eq_sqrt5 {α : ℝ} :
  minimum_distance α = sqrt 5 :=
sorry

end cartesian_equiv_min_distance_eq_sqrt5_l676_676318


namespace direct_proportion_correct_l676_676474

-- Define what it means for y to be a direct proportion of x
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

-- Define the functions from the problem
def f_a (x : ℝ) := -0.1 * x
def f_b (x : ℝ) := 2 * x^2
def f_c (x : ℝ) := real.sqrt (4 * x) -- Mathlib requires sqrt to have nonnegative argument
def f_d (x : ℝ) := 2 * x + 1

-- State the theorem that says f_a represents y as a direct proportion of x
theorem direct_proportion_correct :
  is_direct_proportion f_a ∧
  ¬ is_direct_proportion f_b ∧
  ¬ is_direct_proportion f_c ∧
  ¬ is_direct_proportion f_d :=
by
  -- Proof will be given here
  sorry

end direct_proportion_correct_l676_676474


namespace area_of_large_square_l676_676903

structure Square (s : ℝ) where
  is_sides_midpoints : Prop

-- Given conditions
def semicircle_area (r : ℝ) : ℝ := (π * (r ^ 2)) / 2

theorem area_of_large_square :
  ∀ (s : ℝ),
    (let small_square_side := s / 2,
         area_of_large_semi := semicircle_area (s / 2),
         area_of_small_semi := semicircle_area (small_square_side),
         total_crescent_area := 8 * (area_of_large_semi - area_of_small_semi),
         total_given_area := 5),
    total_crescent_area = total_given_area →
    (s ^ 2) = 10 :=
by
  intros s eq_total_area
  let small_square_side := s / 2
  let area_of_large_semi := semicircle_area (s / 2)
  let area_of_small_semi := semicircle_area small_square_side
  let total_crescent_area := 8 * (area_of_large_semi - area_of_small_semi)
  have h : total_crescent_area = total_given_area := eq_total_area
  sorry

end area_of_large_square_l676_676903


namespace common_chord_equation_l676_676412

noncomputable def common_chord (x y : ℝ) : Prop :=
  (2 * x - 4 * y + 1 = 0)

theorem common_chord_equation : 
  (∀ x y : ℝ, (x - 1)^2 + y^2 = 2 → x^2 + (y - 2)^2 = 4 → common_chord x y) :=
begin
  sorry
end

end common_chord_equation_l676_676412


namespace inequality_correct_l676_676357

theorem inequality_correct (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (1 / a) < (1 / b) :=
sorry

end inequality_correct_l676_676357


namespace ellas_coins_worth_l676_676595

theorem ellas_coins_worth :
  ∀ (n d : ℕ), n + d = 18 → n = d + 2 → 5 * n + 10 * d = 130 := by
  intros n d h1 h2
  sorry

end ellas_coins_worth_l676_676595


namespace percent_not_filler_is_approximately_correct_l676_676521

def burger_weight : ℕ := 150
def filler_weight : ℕ := 40
def percent_not_filler (bw fw : ℕ) : ℝ := (bw - fw : ℝ) / bw * 100 -- Note conversion to ℝ

theorem percent_not_filler_is_approximately_correct :
  percent_not_filler burger_weight filler_weight ≈ 73.33 := by
  sorry

end percent_not_filler_is_approximately_correct_l676_676521


namespace find_n_l676_676084

theorem find_n (n : ℕ) (h : 1 < n) :
  (∀ a b : ℕ, Nat.gcd a b = 1 → (a % n = b % n ↔ (a * b) % n = 1)) →
  (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 12 ∨ n = 24) :=
by
  sorry

end find_n_l676_676084


namespace count_integers_satisfy_inequality_l676_676668

theorem count_integers_satisfy_inequality : 
  {x : ℤ | (x - 1)^2 ≤ 9}.toFinset.card = 7 := 
by 
  sorry

end count_integers_satisfy_inequality_l676_676668


namespace cheaper_mall_A_l676_676055

noncomputable def total_price : ℕ := 699 + 910

def mallA_discount (price : ℕ) : ℕ :=
  let times := price / 200
  in 101 * times

def mallB_discount (price : ℕ) : ℕ :=
  let times := price / 101
  in 50 * times

def final_price_mallA (price : ℕ) : ℕ :=
  let discount := mallA_discount price
  in price - discount

def final_price_mallB (price : ℕ) : ℕ :=
  let discount := mallB_discount price
  in price - discount

theorem cheaper_mall_A : final_price_mallA total_price = 801 ∧ final_price_mallB total_price = 859 ∧ final_price_mallA total_price < final_price_mallB total_price :=
by
  sorry

end cheaper_mall_A_l676_676055


namespace union_of_sets_l676_676654

variable (x : ℝ)

def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | -1 < x ∧ x < 2}
def target : Set ℝ := {x | -1 < x ∧ x < 3}

theorem union_of_sets : (A ∪ B) = target :=
by
  sorry

end union_of_sets_l676_676654


namespace compute_fraction_l676_676914

theorem compute_fraction :
  ( (12^4 + 500) * (24^4 + 500) * (36^4 + 500) * (48^4 + 500) * (60^4 + 500) ) /
  ( (6^4 + 500) * (18^4 + 500) * (30^4 + 500) * (42^4 + 500) * (54^4 + 500) ) = -182 :=
by
  sorry

end compute_fraction_l676_676914


namespace distinct_real_roots_c_l676_676386

theorem distinct_real_roots_c (c : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ + c = 0 ∧ x₂^2 - 4*x₂ + c = 0) ↔ c < 4 := by
  sorry

end distinct_real_roots_c_l676_676386


namespace hexagon_cookie_cutters_count_l676_676923

-- Definitions for the conditions
def triangle_side_count := 3
def triangles := 6
def square_side_count := 4
def squares := 4
def total_sides := 46

-- Given conditions translated to Lean 4
def sides_from_triangles := triangles * triangle_side_count
def sides_from_squares := squares * square_side_count
def sides_from_triangles_and_squares := sides_from_triangles + sides_from_squares
def sides_from_hexagons := total_sides - sides_from_triangles_and_squares
def hexagon_side_count := 6

-- Statement to prove that there are 2 hexagon-shaped cookie cutters
theorem hexagon_cookie_cutters_count : sides_from_hexagons / hexagon_side_count = 2 := by
  sorry

end hexagon_cookie_cutters_count_l676_676923


namespace trapezium_area_proof_l676_676316

-- Define the coordinates and values
variables (E B C A D : Point)
variables (R S : ℝ)

-- Geometry conditions
def equilateral_triangle (E B C : Point) :=
(E.dist(B) = E.dist(C)) ∧ (B.dist(C) = E.dist(B))

def on_line (P1 P2 P : Point) :=
∃ k : ℝ, P = P1 + k • (P2 - P1)

def parallel_lines (l1 l2 : Line) :=
l1.direction ∥ l2.direction

def perpendicular_lines (l1 l2 : Line) :=
l1.direction ⟂ l2.direction

def trapezium_area (A B C D : Point) : ℝ :=
1/2 * (A.dist(C) + B.dist(D)) * (perp_dist_line_point (line(A B)) C)

-- Area of trapezium
def area_ABCD := 9408

-- Problem statement
theorem trapezium_area_proof (H1 : equilateral_triangle E B C) (H2 : on_line E B A)
    (H3 : on_line E C D) (H4 : parallel_lines (line(A D)) (line(B C)))
    (H5 : A.dist B = R) (H6 : C.dist D = R) (H7 : perpendicular_lines (line(A C)) (line(B D))) :
    trapezium_area A B C D = area_ABCD :=
sorry

end trapezium_area_proof_l676_676316


namespace locus_M_is_cylindrical_surface_l676_676626

-- Define the line l and point A
variable (l : Line) (A : Point)

-- Conditions: 
variable (arbitrary_line : Line) 
variable (perpendicular : Perpendicular arbitrary_line l)
variable (M : Point) (N : Point)

noncomputable def common_perpendicular_ends : M ∈ arbitrary_line ∧ N ∈ l ∧ Perpendicular (Line_through M N) l :=
sorry

-- Conclusion:
theorem locus_M_is_cylindrical_surface : 
  (∀ M, (M ∈ arbitrary_line ∧ ∃ t, t ∥ l ∧ M ∈ t)) → ∃ c, cylindrical_surface c ∧ M ∈ c :=
sorry

end locus_M_is_cylindrical_surface_l676_676626


namespace distinct_intersection_points_l676_676163

-- Definitions for the equations and sets
def eq1 (x y : ℝ) := x + y = 7
def eq2 (x y : ℝ) := 2x - 3y = 7
def eq3 (x y : ℝ) := x - y = 2
def eq4 (x y : ℝ) := 3x + 2y = 18

-- Union of solutions of each pair of equations
def solutions := {
  p | ∃ x y : ℝ, (eq1 x y ∨ eq2 x y) ∧ (eq3 x y ∨ eq4 x y)
}

-- Definition of distinct points of intersection
def num_distinct_points := (solutions : set (ℝ × ℝ)).to_finset.card

-- The statement of the problem
theorem distinct_intersection_points : 
  num_distinct_points = 3 :=
by
  sorry

end distinct_intersection_points_l676_676163


namespace total_books_l676_676762

def shelves_of_mystery_books : ℕ := 8
def shelves_of_picture_books : ℕ := 2
def books_per_shelf : ℕ := 7

theorem total_books (shelves_of_mystery_books shelves_of_picture_books books_per_shelf : ℕ) : 
  (shelves_of_mystery_books + shelves_of_picture_books) * books_per_shelf = 70 := 
by
  have total_shelves : ℕ := shelves_of_mystery_books + shelves_of_picture_books
  have total_books : ℕ := total_shelves * books_per_shelf
  calc
    total_books = 10 * 7 := by simp [total_shelves]
    ...        = 70 := by norm_num
sorry

end total_books_l676_676762


namespace min_value_of_fraction_l676_676974

theorem min_value_of_fraction (m n : ℝ) (h1 : 2 * n + m = 4) (h2 : m > 0) (h3 : n > 0) : 
  (∀ n m, 2 * n + m = 4 ∧ m > 0 ∧ n > 0 → ∀ y, y = 2 / m + 1 / n → y ≥ 2) :=
by sorry

end min_value_of_fraction_l676_676974


namespace feuerbach_theorem_l676_676390

theorem feuerbach_theorem
  (ABC : Triangle)
  (A1 B1 C1 : Point)
  (a b c : ℕ)
  (P Q : Point)
  (S Sa : Circle)
  (A1_mid : midpoint ABC.BC A1)
  (B1_mid : midpoint ABC.AC B1)
  (C1_mid : midpoint ABC.AB C1)
  (incircle_touch : circle_touch S ABC.BC P)
  (excircle_touch : circle_touch Sa ABC.BC Q) :
  tangent_to_incircle_and_excircles (circumcircle_of_midpoints A1 B1 C1) S Sa := sorry

end feuerbach_theorem_l676_676390


namespace no_such_natural_number_exists_l676_676589

theorem no_such_natural_number_exists :
  ¬ ∃ (n s : ℕ), n = 2014 * s + 2014 ∧ n % s = 2014 ∧ (n / s) = 2014 :=
by
  sorry

end no_such_natural_number_exists_l676_676589


namespace visible_yellow_bus_length_correct_l676_676019

noncomputable def red_bus_length : ℝ := 48
noncomputable def orange_car_length : ℝ := red_bus_length / 4
noncomputable def yellow_bus_length : ℝ := 3.5 * orange_car_length
noncomputable def green_truck_length : ℝ := 2 * orange_car_length
noncomputable def total_vehicle_length : ℝ := yellow_bus_length + green_truck_length
noncomputable def visible_yellow_bus_length : ℝ := 0.75 * yellow_bus_length

theorem visible_yellow_bus_length_correct :
  visible_yellow_bus_length = 31.5 := 
sorry

end visible_yellow_bus_length_correct_l676_676019


namespace no_natural_number_divisible_2014_l676_676591

theorem no_natural_number_divisible_2014 : ¬∃ n s : ℕ, n = 2014 * s + 2014 := 
by
  -- Assume for contradiction that such numbers exist
  intro ⟨n, s, h⟩,
  -- Consider the transformed equation for contradiction
  have h' : n - s = 2013 * s + 2014, from sorry,
  -- Check the divisibility by 3 leading to contradiction
  have div_contr : (2013 * s + 2014) % 3 = 1, from sorry,
  -- Using the contradiction to close the proof
  sorry

end no_natural_number_divisible_2014_l676_676591


namespace parallel_BE_DF_l676_676364

variable {A B C D E F : Type} [Nonempty A → Nonempty B → Nonempty C → Nonempty D → Nonempty E → Nonempty F]
variable {angle_CBA angle_ACB angle_AB angle_CBX angle_FXB angle_CFD : ℝ}
variable (α β γ : ℝ)
variable (isTriangle : Triangle A B C)
variable (D_on_AC : Point D ∈ Segment A C)
variable (E_on_AC : Point E ∈ Segment A C)
variable (D_between_AE : Point D ∈ Betweenness A E)
variable (F_on_AB_bisector : Point F ∈ Bisector (Angle A C B))
variable (angle_condition : 2 * angle_CBA = 3 * angle_ACB)
variable (angle_CBA_split : angle {B D} = angle {D E} = angle {E B} = angle_CBA / 3)
variable (angle_CFD : angle C F D = (1/2) * angle_ACB)
variable (angle_sum_condition : α + β + γ = 180)
variable (alpha_condition : α = 180 - (5/2) * γ)

theorem parallel_BE_DF : Parallel_Line BE DF := 
  sorry

end parallel_BE_DF_l676_676364


namespace ticket_probability_multiple_of_3_l676_676828

theorem ticket_probability_multiple_of_3 :
  let total_tickets := 24
  let multiples_of_3 := {n | n ∈ Finset.range 24 ∧ (n + 1) % 3 = 0}
  (multiples_of_3.card : ℚ) / total_tickets = 1 / 3 :=
by
  let multiples_of_3 := {n | n ∈ Finset.range 24 ∧ (n + 1) % 3 = 0}
  let favorable_count := multiples_of_3.card
  have h_favorable_count : favorable_count = 8 := by sorry
  have h_total_tickets : total_tickets = 24 := by sorry
  rw [h_favorable_count, h_total_tickets]
  norm_num
  exact eq.refl (1 / 3)

end ticket_probability_multiple_of_3_l676_676828


namespace solve_equation_l676_676437

theorem solve_equation : ∃ x : ℝ, (2 / x) = (1 / (x + 1)) ∧ x = -2 :=
by
  use -2
  split
  { -- Proving the equality part
    show (2 / -2) = (1 / (-2 + 1)),
    simp,
    norm_num
  }
  { -- Proving the equation part
    refl
  }

end solve_equation_l676_676437


namespace solve_fractional_eq_l676_676442

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end solve_fractional_eq_l676_676442


namespace number_above_180_l676_676842

open Nat

theorem number_above_180 : 
  (∃ (n m : ℕ), (n^2 < 180) ∧ (180 <= (n+1)^2) ∧ (m = 180 - n^2 - 1)) → 
  (∃ k, k * k = 13) → 
  find_row_start (n - 1) = 145 ∧ 
  number_in_row 145 11 = 155 :=
begin
  sorry
end

end number_above_180_l676_676842


namespace eccentricity_of_hyperbola_l676_676625

noncomputable theory
open Real

variables (a b c e : ℝ) (P Q : ℝ × ℝ)

def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def is_line (slope t : ℝ) (x y : ℝ) : Prop :=
  y = slope * x + t

def is_focus (a b x : ℝ) : Prop :=
  x = real.sqrt (a^2 + b^2)

theorem eccentricity_of_hyperbola 
  (ha : 0 < a) 
  (hb : 0 < b)
  (hslope : ∃ t, is_line (real.sqrt 2 / 2) t P.1 P.2 ∧ is_line (real.sqrt 2 / 2) t Q.1 Q.2)
  (hintersection : is_hyperbola a b P.1 P.2 ∧ is_hyperbola a b Q.1 Q.2)
  (hprojections : is_focus a b P.1 ∧ is_focus a b Q.1) :
  e = real.sqrt 2 :=
sorry

end eccentricity_of_hyperbola_l676_676625


namespace distance_between_points_l676_676949

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 3 3 (-2) (-2) = 5 * real.sqrt 2 :=
by
  sorry

end distance_between_points_l676_676949


namespace hexagon_area_105_l676_676809

-- Definitions of the problem conditions
def triangle_perimeter (a b c : ℝ) : ℝ := a + b + c
def circum_radius := 10
def area_of_hexagon (a b c : ℝ) : ℝ :=
  let perimeter := triangle_perimeter a b c in
  (5 * perimeter) / 2

-- Main theorem statement to be proved
theorem hexagon_area_105 (a b c : ℝ) (h1 : triangle_perimeter a b c = 42) :
  area_of_hexagon a b c = 105 :=
by
  sorry

end hexagon_area_105_l676_676809


namespace alice_preferred_number_l676_676128

def is_multiple (a b : ℕ) : Prop := ∃ k, a = b * k

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem alice_preferred_number : 
  ∃ x : ℕ, 70 ≤ x ∧ x ≤ 140 ∧ is_multiple x 13 ∧ ¬is_multiple x 3 ∧ is_multiple (digit_sum x) 4 :=
by
  use 130
  have h1 : 70 ≤ 130 := by norm_num
  have h2 : 130 ≤ 140 := by norm_num
  have h3 : is_multiple 130 13 := by use 10; norm_num
  have h4 : ¬is_multiple 130 3 := by 
    intro h
    cases h with k hk
    cases k; norm_num
  have h5 : is_multiple (digit_sum 130) 4 := by 
    have : digit_sum 130 = 4 := by norm_num
    exact ⟨1, by norm_num⟩
  exact ⟨h1, h2, h3, h4, h5⟩

end alice_preferred_number_l676_676128


namespace arithmetic_expression_evaluation_l676_676147

theorem arithmetic_expression_evaluation : 5 * (9 / 3) + 7 * 4 - 36 / 4 = 34 := 
by
  sorry

end arithmetic_expression_evaluation_l676_676147


namespace train_speed_is_79_99_in_km_hr_l676_676120

/-- Define the initial conditions -/
def length_of_train : ℝ := 100
def length_of_bridge : ℝ := 142
def crossing_time : ℝ := 10.889128869690424
def total_distance := length_of_train + length_of_bridge
def speed_m_s := total_distance / crossing_time
def speed_km_hr := speed_m_s * 3.6

/-- The theorem to prove -/
theorem train_speed_is_79_99_in_km_hr :
  speed_km_hr ≈ 79.99 :=
sorry

end train_speed_is_79_99_in_km_hr_l676_676120


namespace f_is_odd_f_is_periodic_l676_676984

section
variable (f : ℝ → ℝ)

-- Given conditions
axiom f_property : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * Real.cos y
axiom f_zero : f 0 = 0
axiom f_pi_div2 : f (Real.pi / 2) = 1

-- Proof goals
theorem f_is_odd : ∀ y : ℝ, f(y) + f(-y) = 0 := by
  sorry

theorem f_is_periodic : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

end

end f_is_odd_f_is_periodic_l676_676984


namespace sum_b_geom_sequence_one_sum_b_geom_sequence_non_one_general_term_b_n_three_harmonic_sum_inequality_l676_676993

variable {a : ℝ} (a_pos : a > 0)
variable {a_n : ℕ → ℝ} (a_geom : ∀ n, a_n (n + 1) = a * a_n n) (a1 : a_n 1 = 1)
variable {b_n : ℕ → ℝ} (b_def : ∀ n, b_n n = a_n n * a_n (n + 1))

-- Problem 1
theorem sum_b_geom_sequence_one (h : a = 1) (n : ℕ) :
  (∑ i in finset.range n, b_n (i + 1)) = n := sorry

theorem sum_b_geom_sequence_non_one (h : a ≠ 1) (n : ℕ) :
  (∑ i in finset.range n, b_n (i + 1)) = a * (1 - a^(2 * n)) / (1 - a^2) := sorry

-- Problem 2
theorem general_term_b_n_three (h : ∀ n, b_n n = 3^n) (n : ℕ) :
  a_n n = if n % 2 = 1 then 3^((n - 1) / 2) else a * 3^((n - 2) / 2) := sorry

-- Problem 3
theorem harmonic_sum_inequality (h : ∀ n, b_n n = n + 2) (n : ℕ) :
  (∑ i in finset.range n, 1 / a_n (i + 1)) > 2 * sqrt (n + 2) - 3 := sorry

end sum_b_geom_sequence_one_sum_b_geom_sequence_non_one_general_term_b_n_three_harmonic_sum_inequality_l676_676993


namespace i_power_eq_one_l676_676898

theorem i_power_eq_one (n : ℤ) (h : n ∈ ({2, 3, 4, 5} : set ℤ)) : (complex.I ^ n = 1) ↔ n = 4 :=
by
  sorry

end i_power_eq_one_l676_676898


namespace pentomino_reflectional_count_l676_676672

def is_reflectional (p : Pentomino) : Prop := sorry -- Define reflectional symmetry property
def is_rotational (p : Pentomino) : Prop := sorry -- Define rotational symmetry property

theorem pentomino_reflectional_count :
  ∀ (P : Finset Pentomino),
  P.card = 15 →
  (∃ (R : Finset Pentomino), R.card = 2 ∧ (∀ p ∈ R, is_rotational p ∧ ¬ is_reflectional p)) →
  (∃ (S : Finset Pentomino), S.card = 7 ∧ (∀ p ∈ S, is_reflectional p)) :=
by
  sorry -- Proof not required as per instructions

end pentomino_reflectional_count_l676_676672


namespace soaking_time_l676_676143

theorem soaking_time : 
  let grass_stain_time := 4 
  let marinara_stain_time := 7 
  let num_grass_stains := 3 
  let num_marinara_stains := 1 
  in 
  num_grass_stains * grass_stain_time + num_marinara_stains * marinara_stain_time = 19 := 
by 
  sorry

end soaking_time_l676_676143


namespace domain_ln_sub_correct_l676_676796

noncomputable def domain_ln_sub (x : ℝ) : Prop :=
  ∃ (x : ℝ), (∀ x, 2 < x → ∃ y, f y = ln (x - 2))

theorem domain_ln_sub_correct : 
  ∃ (x : ℝ), (∀ x, 2 < x → ∃ y, f y = ln (x - 2)) := 
sorry

end domain_ln_sub_correct_l676_676796


namespace distance_between_points_l676_676943

theorem distance_between_points :
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  (Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 * Real.sqrt 2) :=
by
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  sorry

end distance_between_points_l676_676943


namespace place_numbers_in_squares_l676_676836

theorem place_numbers_in_squares : 
  let numbers := {x | x ∈ {1, 2, 3, 4, 5, 6}} in
  ∃ arrangement : (Z × Z × Z × Z × Z × Z),
    (∀ x y : Z, (x, y) ∈ arrangement → (x < y) → number_in_higher_square x y) →
    (count_distinct_arrangements numbers arrangement) = 12 :=
sorry

end place_numbers_in_squares_l676_676836


namespace distinct_sequences_exist_l676_676981

open Nat

theorem distinct_sequences_exist
  (t : ℕ)
  (p : Fin t → ℕ)
  (α : Fin t → ℕ)
  (h_prime : ∀ i, Prime (p i))
  (h_pos : ∀ i, 0 < α i)
  (n : ℕ := (∏ i, p i ^ α i))
  (n_list : Fin 10000 → ℕ)
  (h_distinct_n : ∀ i j, i ≠ j → n_list i ≠ n_list j)
  (h_distinct_largest_prime_power : ∀ i j, i ≠ j → 
    (let m_prime_power (n : ℕ) := (max (Fin n) (λ j, p j ^ α j));
      m_prime_power (n_list i) ≠ m_prime_power (n_list j))
  ) :
  ∃ a : Fin 10000 → ℤ,
  ∀ i j, i ≠ j →
  (∀ m n : ℕ, a i + m * (n_list i : ℤ) ≠ a j + n * (n_list j : ℤ)) := 
sorry

end distinct_sequences_exist_l676_676981


namespace area_of_polygon_l676_676723

theorem area_of_polygon (side_length n : ℕ) (h1 : n = 36) (h2 : 36 * side_length = 72) (h3 : ∀ i, i < n → (∃ a, ∃ b, (a + b = 4) ∧ (i = 4 * a + b))) :
  (n / 4) * side_length ^ 2 = 144 :=
by
  sorry

end area_of_polygon_l676_676723


namespace equal_segment_of_bisectors_and_circumcircle_l676_676799

theorem equal_segment_of_bisectors_and_circumcircle
  (A B C L K N : Point)
  (circumcircle : Circle)
  (h_circumcircle : Circumcircle A B C circumcircle)
  (h_BL : AngleBisector B L)
  (h_BL_extends : ExtendsTo BL circumcircle K)
  (h_ext_angle_bisector : ExternalAngleBisector B (line_through C A) N)
  (h_LN_extends : ExtendsTo LN N)
  (h_BK_BN_eq : BK = BN) :
  LN = 2 * circumcircle.radius := 
sorry

end equal_segment_of_bisectors_and_circumcircle_l676_676799


namespace find_radius_of_cone_l676_676604

noncomputable def CSA := 3298.6722862692827
def l := 30
def π := Real.pi
def expected_radius := 34.99202034341547

theorem find_radius_of_cone (r : ℝ) :
  CSA = π * r * l → r = expected_radius := 
sorry

end find_radius_of_cone_l676_676604


namespace simple_interest_rate_l676_676494

theorem simple_interest_rate (P SI T : ℝ) (hP : P = 800) (hSI : SI = 128) (hT : T = 4) : 
  (SI = P * (R : ℝ) * T / 100) → R = 4 := 
by {
  -- Proof goes here.
  sorry
}

end simple_interest_rate_l676_676494


namespace true_proposition_is_C_l676_676632

-- Define propositions
def p : Prop := ∀ x : ℝ, x > 0 → Real.log (x + 1) > 0
def q : Prop := ∀ a b : ℝ, a > b → a^2 > b^2

-- Assume p is true and q is false
lemma prop_p_true : p := 
  by
  intros x hx
  have h_ineq : x + 1 > 1 := by linarith
  have h_log : Real.log (x + 1) > Real.log 1 := Real.log_lt_log h_ineq (by norm_num)
  linarith

lemma prop_q_false : ¬ q := 
  by
  intro h
  have h_ineq := h (-2) (-3) (by linarith)
  linarith

-- Prove the true proposition among the options
theorem true_proposition_is_C : p ∧ ¬ q :=
by
  exact ⟨prop_p_true, prop_q_false⟩

end true_proposition_is_C_l676_676632


namespace OI_perp_AB_l676_676144

-- Definitions for points, right triangle, angle bisectors and circumcenter
variables {A B C A1 B1 I O : Point}
variables {r : ℝ}

-- Right triangle ABC with ∠C being the right angle
axiom right_triangle (h_right : angle A C B = 90)

-- AA1 and BB1 are the angle bisectors of ∠A and ∠B intersecting at I
axiom angle_bisector_A (h_AA1 : is_angle_bisector A A1)
axiom angle_bisector_B (h_BB1 : is_angle_bisector B B1)
axiom incenter_I (h_I : I = intersection_lines (line A A1) (line B B1))

-- O is the circumcenter of triangle CA1B1
axiom circumcenter_O (h_circumcenter : is_circumcenter O C A1 B1)

-- Prove OI is perpendicular to AB
theorem OI_perp_AB : is_perpendicular (line O I) (line A B) :=
sorry

end OI_perp_AB_l676_676144


namespace trader_sold_80_pens_l676_676119

variable (C : ℝ) -- cost of one pen
variable (N : ℝ) -- number of pens sold
variable (gain : ℝ)

-- Assume gain percentage is 25% and gain is equal to the cost of 20 pens
def gain_percentage := 0.25
def gain_cost := 20 * C
def total_cost_sold := N * C

theorem trader_sold_80_pens (h1 : gain_percentage * total_cost_sold = gain_cost) : N = 80 := 
by
  sorry 

end trader_sold_80_pens_l676_676119


namespace visible_factor_numbers_200_to_250_l676_676146

def is_visible_factor_number (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10).filter (λ x => x ≠ 0), n % d = 0

def count_visible_factor_numbers (start : ℕ) (end_ : ℕ) : ℕ :=
  (List.range' start (end_ - start + 1)).countp is_visible_factor_number

theorem visible_factor_numbers_200_to_250 : count_visible_factor_numbers 200 250 = 22 := 
by sorry

end visible_factor_numbers_200_to_250_l676_676146


namespace wall_volume_l676_676041

-- Define the conditions as outlined in step a
def breadth : ℝ := 40  -- in cm
def height : ℝ := 5 * breadth
def length : ℝ := 8 * height

-- Define the volume calculation
def volume_cm : ℝ := breadth * height * length

-- Convert the volume from cubic centimeters to cubic meters
def volume_m : ℝ := volume_cm / 1000000

-- The theorem to prove that the volume in cubic meters is 12.8
theorem wall_volume : volume_m = 12.8 :=
by
  sorry

end wall_volume_l676_676041


namespace coefficient_of_x2_in_expansion_of_frac_l676_676008

theorem coefficient_of_x2_in_expansion_of_frac (x : ℂ) :
  (coefficient_of_x2 ((x - 1)^6 / x) = -20) :=
begin
  sorry
end

end coefficient_of_x2_in_expansion_of_frac_l676_676008


namespace compare_A_B_C_l676_676478

-- Define the expressions A, B, and C
def A : ℚ := (2010 / 2009) + (2010 / 2011)
def B : ℚ := (2010 / 2011) + (2012 / 2011)
def C : ℚ := (2011 / 2010) + (2011 / 2012)

-- The statement asserting A is the greatest
theorem compare_A_B_C : A > B ∧ A > C := by
  sorry

end compare_A_B_C_l676_676478


namespace value_of_r6_plus_s6_l676_676750

theorem value_of_r6_plus_s6 :
  ∀ r s : ℝ, (r^2 - 2 * r + Real.sqrt 2 = 0) ∧ (s^2 - 2 * s + Real.sqrt 2 = 0) →
  (r^6 + s^6 = 904 - 640 * Real.sqrt 2) :=
by
  intros r s h
  -- Proof skipped
  sorry

end value_of_r6_plus_s6_l676_676750


namespace not_divides_l676_676367

theorem not_divides (d a n : ℕ) (h1 : 3 ≤ d) (h2 : d ≤ 2^(n+1)) : ¬ d ∣ a^(2^n) + 1 := 
sorry

end not_divides_l676_676367


namespace problem1_problem2_problem3_l676_676866

-- Problem 1: Number of ways to place five balls into five boxes with exactly one box empty
theorem problem1 : (number_ways_one_box_empty 5 5 = 1200) :=
sorry

-- Problem 2: Number of ways to place five balls into five boxes with no box left empty and no matching numbers
theorem problem2 : (number_ways_no_box_empty_no_matching 5 5 = 119) :=
sorry

-- Problem 3: Number of ways to place five balls into five boxes with exactly one ball in each box and at least two balls matching their box numbers
theorem problem3 : (number_ways_one_ball_each_box_at_least_two_matching 5 5 = 31) :=
sorry

-- Definitions are implied by the problem statement and conditions.

end problem1_problem2_problem3_l676_676866


namespace poplar_more_than_pine_l676_676042

theorem poplar_more_than_pine (pine poplar : ℕ) (h1 : pine = 180) (h2 : poplar = 4 * pine) : poplar - pine = 540 :=
by
  -- Proof will be filled here
  sorry

end poplar_more_than_pine_l676_676042


namespace sequence_bounded_l676_676629

open Real

noncomputable def seq {n : ℕ} := ℝ -- assuming a sequence of real positive numbers indexed by natural numbers

variable (a : ℕ → seq)
variable (B : ℕ → ℝ) -- B sums the given indices to the expression

-- Given condition that the sum of the sequence is only dependent on the sum of the indices
axiom sequence_property :
  ∀ k n m l : ℕ, (k + n = m + l) → 
  ((a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l))

theorem sequence_bounded :
  ∃ m M : ℝ, ∀ n : ℕ, m ≤ a n ∧ a n ≤ M := sorry

end sequence_bounded_l676_676629


namespace brown_dogs_count_l676_676305

theorem brown_dogs_count (T L N LB : ℕ) (hT : T = 45) (hL : L = 29) (hN : N = 8) (hLB : LB = 9) : 
  ∃ B, B = 17 :=
by
  have : L ∨ B ∨ both = T - N := 37 (45 - 8)
  have : B = (L ∨ B ∨ both) - (L - LB) := 17 (37 - (29 - 9))
  exists 17
  sorry

end brown_dogs_count_l676_676305


namespace sum_geometric_sequence_first_10_terms_l676_676963

theorem sum_geometric_sequence_first_10_terms :
  let a₁ : ℚ := 12
  let r : ℚ := 1 / 3
  let S₁₀ : ℚ := 12 * (1 - (1 / 3)^10) / (1 - 1 / 3)
  S₁₀ = 1062864 / 59049 := by
  sorry

end sum_geometric_sequence_first_10_terms_l676_676963


namespace num_possible_x_l676_676270

theorem num_possible_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 20) : {y : ℕ | 361 < y ∧ y ≤ 400}.card = 39 :=
by
  sorry

end num_possible_x_l676_676270


namespace find_cos_7theta_l676_676676

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end find_cos_7theta_l676_676676


namespace four_digit_number_exists_l676_676920

-- Definitions corresponding to the conditions in the problem
def is_four_digit_number (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def follows_scheme (n : ℕ) (d : ℕ) : Prop :=
  -- Placeholder for the scheme condition
  sorry

-- The Lean statement for the proof problem
theorem four_digit_number_exists :
  ∃ n d1 d2 : ℕ, is_four_digit_number n ∧ follows_scheme n d1 ∧ follows_scheme n d2 ∧ 
  (n = 1014 ∨ n = 1035 ∨ n = 1512) :=
by {
  -- Placeholder for proof steps
  sorry
}

end four_digit_number_exists_l676_676920


namespace resistor_possibilities_l676_676710

theorem resistor_possibilities (n : ℕ) (h : n = 7) : finset.card (finset.filter (λ s, s.card < n) (finset.powerset (finset.range n))) = 63 :=
by
  rw h
  sorry

end resistor_possibilities_l676_676710


namespace solve_equation_l676_676429

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) : (2 / x = 1 / (x + 1)) → x = -2 :=
begin
  sorry
end

end solve_equation_l676_676429


namespace problem_statement_l676_676229

theorem problem_statement (x : ℝ) (h : x + x⁻¹ = 3) : x^2 + x⁻² = 7 :=
sorry

end problem_statement_l676_676229


namespace proposition_four_correct_l676_676401

variable (α β : Plane) (m n : Line)

-- Conditions
def planes_non_coincident : Prop := ¬ (α = β)
def lines_non_coincident : Prop := ¬ (m = n)

/-- Proposition ④: If α ⟂ β, α ∩ β = m, n ⊆ α, and n ⟂ m, then n ⟂ β --/
theorem proposition_four_correct
  (h1 : planes_non_coincident α β)
  (h2 : lines_non_coincident m n)
  (h3 : α ⟂ β)
  (h4 : (α ∩ β) = m)
  (h5 : n ⊆ α)
  (h6 : n ⟂ m) :
  n ⟂ β := 
sorry

end proposition_four_correct_l676_676401


namespace circles_intersect_orthogonally_l676_676776

noncomputable def circle1 (a b : ℝ) : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * a * p.1 + b^2 = 0 }

noncomputable def circle2 (c b : ℝ) : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * c * p.2 - b^2 = 0 }

theorem circles_intersect_orthogonally (a b c : ℝ) 
(ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
(hab : a ≠ c) :
  ∃ p : ℝ × ℝ, p ∈ circle1 a b ∧ p ∈ circle2 c b ∧
  let ⟨x, y⟩ := p in
  x^2 + y^2 - 2 * a * x + b^2 = 0 ∧ x^2 + y^2 - 2 * c * y - b^2 = 0 ∧
  x * y - a * c = 0 := 
sorry

end circles_intersect_orthogonally_l676_676776


namespace shortest_paths_ratio_l676_676112

theorem shortest_paths_ratio (k n : ℕ) (h : k > 0):
  let paths_along_AB := Nat.choose (k * n + n - 1) (n - 1)
  let paths_along_AD := Nat.choose (k * n + n - 1) k * n - 1
  paths_along_AD = k * paths_along_AB :=
by sorry

end shortest_paths_ratio_l676_676112


namespace find_seventh_number_l676_676860

-- Let's denote the 10 numbers as A1, A2, A3, A4, A5, A6, A7, A8, A9, A10.
variables {A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ}

-- The average of all 10 numbers is 60.
def avg_10 (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ) := (A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10) / 10 = 60

-- The average of the first 6 numbers is 68.
def avg_first_6 (A1 A2 A3 A4 A5 A6 : ℝ) := (A1 + A2 + A3 + A4 + A5 + A6) / 6 = 68

-- The average of the last 6 numbers is 75.
def avg_last_6 (A5 A6 A7 A8 A9 A10 : ℝ) := (A5 + A6 + A7 + A8 + A9 + A10) / 6 = 75

-- Proving that the 7th number (A7) is 192.
theorem find_seventh_number (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ) 
  (h1 : avg_10 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10) 
  (h2 : avg_first_6 A1 A2 A3 A4 A5 A6) 
  (h3 : avg_last_6 A5 A6 A7 A8 A9 A10) :
  A7 = 192 :=
by
  sorry

end find_seventh_number_l676_676860


namespace erasers_difference_l676_676556

-- Definitions for the conditions in the problem
def andrea_erasers : ℕ := 4
def anya_erasers : ℕ := 4 * andrea_erasers

-- Theorem statement to prove the final answer
theorem erasers_difference : anya_erasers - andrea_erasers = 12 :=
by
  -- Proof placeholder
  sorry

end erasers_difference_l676_676556


namespace inverse_composition_l676_676495

variables {X Y Z W : Type}
variables (u : X → Y) (v : Y → Z) (w : Z → W)
variables [invertible u] [invertible v] [invertible w]
variable g : X → W := u ∘ v ∘ w

theorem inverse_composition :
  function.inverse g = function.inverse w ∘ function.inverse v ∘ function.inverse u :=
sorry

end inverse_composition_l676_676495


namespace convert_polar_to_rectangular_example_l676_676580

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular_example :
  polar_to_rectangular 6 (5 * Real.pi / 2) = (0, 6) := by
  sorry

end convert_polar_to_rectangular_example_l676_676580


namespace increased_work_l676_676858

variable (W p : ℕ)

theorem increased_work (hW : W > 0) (hp : p > 0) : 
  (W / (7 * p / 8)) - (W / p) = W / (7 * p) := 
sorry

end increased_work_l676_676858


namespace find_divisor_l676_676299

-- Define the variables: N (dividend), Q (quotient), R (remainder), and the unknown divisor D
variables (D : ℚ)

-- Define the given conditions
def Dividend : ℚ := 13 / 3
def Quotient : ℚ := -61
def Remainder : ℚ := -19

-- Declare the target theorem to prove Divisor D
theorem find_divisor :
  Dividend = (D * Quotient) + Remainder → D = -70 / 183 :=
begin
  sorry,
end

end find_divisor_l676_676299


namespace binary_predecessor_l676_676267

def M : ℕ := 84
def N : ℕ := 83
def M_bin : ℕ := 0b1010100
def N_bin : ℕ := 0b1010011

theorem binary_predecessor (H : M = M_bin ∧ N = M - 1) : N = N_bin := by
  sorry

end binary_predecessor_l676_676267


namespace find_m_find_tan_α_l676_676997

variable {α : ℝ} -- Angle α
variable {m : ℝ} -- Coordinate m of point P(m, 1)

-- Given conditions
def point_on_terminal_side (α : ℝ) (P : ℝ × ℝ) (m : ℝ) : Prop :=
  P = (m, 1) ∧ m / Real.sqrt(m^2 + 1) = -1/3

-- Required to prove m value
theorem find_m (h : point_on_terminal_side α (m, 1) m) : m = -Real.sqrt(2) / 4 :=
  sorry

-- Required to prove tan α value
theorem find_tan_α (h : point_on_terminal_side α (m, 1) m) (hm : m = -Real.sqrt(2) / 4) : Real.tan α = -2 * Real.sqrt(2) :=
  sorry

end find_m_find_tan_α_l676_676997


namespace matrix_commute_fraction_l676_676352

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, -1], ![5, 2]]
noncomputable def matrix_B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![x, y], ![z, w]]

theorem matrix_commute_fraction (x y z w : ℝ) (h1 : 5 * y ≠ z)
    (h2 : matrix_A ⬝ matrix_B x y z w = matrix_B x y z w ⬝ matrix_A) : 
    (x - w) / (z - 5 * y) = (1 / 10) :=
sorry

end matrix_commute_fraction_l676_676352


namespace max_value_problem1_l676_676864

theorem max_value_problem1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) : 
  ∃ t, t = (1 / 2) * x * (1 - 2 * x) ∧ t ≤ 1 / 16 := sorry

end max_value_problem1_l676_676864


namespace angle_CHX_l676_676548

-- Define the conditions and question
variables {A B C H X : Type}
variables [inner_product_space ℝ A]

-- A is, an acute triangle such that
def angle_BAC : ℝ := 50 
def angle_ABC : ℝ := 85 
def orthocenter_H_of_ABC : Prop := sorry  -- We assume some definition for the orthocenter H

-- Our target is to prove angle CHX
theorem angle_CHX {A B C H X : Type} [inner_product_space ℝ A]
  (h1 : ∠ BAC = 50)
  (h2 : ∠ ABC = 85)
  (h3 : orthocenter_H_of_ABC) :
  ∠ CHX = 45 :=
sorry

end angle_CHX_l676_676548


namespace nth_prime_gt_3n_l676_676389

theorem nth_prime_gt_3n (n : ℕ) (h : n > 12) : prime n > 3 * n :=
sorry

end nth_prime_gt_3n_l676_676389


namespace pizza_fraction_eaten_l676_676852

theorem pizza_fraction_eaten :
  let a := (1 / 4 : ℚ)
  let r := (1 / 2 : ℚ)
  let n := 6
  (a * (1 - r ^ n) / (1 - r)) = 63 / 128 :=
by
  let a := (1 / 4 : ℚ)
  let r := (1 / 2 : ℚ)
  let n := 6
  sorry

end pizza_fraction_eaten_l676_676852


namespace acute_triangle_circumcircle_property_l676_676720

noncomputable def acute_triangle (A B C : Point) : Prop :=
  triangle A B C ∧ 
  ∀ (α β γ : ℝ), α + β + γ = π ∧ α < π / 2 ∧ β < π / 2 ∧ γ < π / 2

noncomputable def altitude (B B1 C : Point) : Line :=
  line_through_points B B1 ∧ ⟂ line_through_points B C

noncomputable def angle_bisector (P Q R : Point) (angle : ℝ) : Line :=
  bisects_angle (angle P Q R) (line_through_points Q P) (line_through_points Q R)

noncomputable def circumcircle (A B C : Point) : Circle :=
  circle_through_points A B C

theorem acute_triangle_circumcircle_property
  (A B C B1 C1 D E X : Point)
  (h_acute : acute_triangle A B C)
  (h_altitudes : altitude B B1 C ∧ altitude C C1 B)
  (h_bisectors : angle_bisector B B1 C (angle B B1 C) (line_through_points B1 X) ∧
                 angle_bisector C C1 B (angle C C1 B) (line_through_points C1 X))
  (h_intersections : X ∈ (line_through_points D E) ∧ D ∈ (line_through_points B C) ∧ E ∈ (line_through_points B C))
  : ∃ P, P ∈ (circumcircle B E X) ∧ P ∈ (circumcircle C D X) ∧ P ∈ (line_through_points A X) :=
sorry

end acute_triangle_circumcircle_property_l676_676720


namespace max_pairwise_dissimilar_numbers_l676_676046

theorem max_pairwise_dissimilar_numbers : 
  let n := 100
  let digits := {i : ℕ | i = 1 ∨ i = 2}
  let valid_sequences := {seq : list ℕ | seq.length = n ∧ ∀ x ∈ seq, x ∈ digits}
  let operation (seq : list ℕ) (i : ℕ) := 
    let first_five := seq.drop i |>.take 5
    let next_five := seq.drop (i + 5) |>.take 5
    seq.take i ++ next_five ++ first_five ++ seq.drop (i + 10)
  let similar (a b : list ℕ) := ∃ ops : list ℕ, ops.length ≤ n ∧ ∀ i ∈ ops, let swapped := operation a i in a = b
  let dissimilar_max := {group : set similar | ∃ a b ∈ group, similar a b → a = b} in
  finite dissimilar_max → dissimilar_max.card ≤ 21^5 :=
sorry

end max_pairwise_dissimilar_numbers_l676_676046


namespace total_students_l676_676855

theorem total_students (rank_right rank_left : ℕ) (h1 : rank_right = 16) (h2 : rank_left = 6) : rank_right + rank_left - 1 = 21 := 
by
  rw [h1, h2]
  rfl

end total_students_l676_676855


namespace G_is_group_Hom_zero_l676_676342

-- Define the set G
def G : Set (Matrix (Fin 2) (Fin 2) (ZMod 7)) :=
  { M | ∃ a b : ZMod 7, a ≠ 0 ∧ M = ![![a, b], ![0, 1]] }

-- Proof Problem (Part a): G is a group
theorem G_is_group : IsGroup G := 
  sorry

-- Define the homomorphism space
def HomGZ7 := {f : G → ZMod 7 // IsGroupHomomorphism f}

-- Proof Problem (Part b): The set of homomorphisms from G to Z_7 is the zero homomorphism
theorem Hom_zero : HomGZ7 = {0} :=
  sorry

end G_is_group_Hom_zero_l676_676342


namespace total_carrot_sticks_l676_676732

-- Define the number of carrot sticks James ate before and after dinner
def carrot_sticks_before_dinner : Nat := 22
def carrot_sticks_after_dinner : Nat := 15

-- Prove that the total number of carrot sticks James ate is 37
theorem total_carrot_sticks : carrot_sticks_before_dinner + carrot_sticks_after_dinner = 37 :=
  by sorry

end total_carrot_sticks_l676_676732


namespace required_hemispherical_containers_l676_676123

noncomputable def initial_volume : ℝ := 10940
noncomputable def initial_temperature : ℝ := 20
noncomputable def final_temperature : ℝ := 25
noncomputable def expansion_coefficient : ℝ := 0.002
noncomputable def container_volume : ℝ := 4
noncomputable def usable_capacity : ℝ := 0.8

noncomputable def volume_expansion : ℝ := initial_volume * (final_temperature - initial_temperature) * expansion_coefficient
noncomputable def final_volume : ℝ := initial_volume + volume_expansion
noncomputable def usable_volume_per_container : ℝ := container_volume * usable_capacity
noncomputable def number_of_containers_needed : ℝ := final_volume / usable_volume_per_container

theorem required_hemispherical_containers : ⌈number_of_containers_needed⌉ = 3453 :=
by 
  sorry

end required_hemispherical_containers_l676_676123


namespace non_extreme_modification_does_not_change_sum_extreme_modification_changes_sum_sum_of_differences_l676_676851

theorem non_extreme_modification_does_not_change_sum (a : Fin n → ℝ) (h : 2 ≤ n) (k : Fin (n-2)) :
  (∑ i in Finset.range (n - 1), a (i+1) - a i) = (∑ i in Finset.range (n - 1), if i = k then a (i+2) - a (i+1) else a (i+1) - a i) :=
by sorry

theorem extreme_modification_changes_sum (a : Fin n → ℝ) (h1 : 2 ≤ n) :
  ∀ (a1' an' : ℝ), (∑ i in Finset.range (n - 1), a (i+1) - a i) ≠ (∑ i in Finset.range (n - 1), if i = 0 then a (i+1) - a1' else (if i = n-2 then an' - a i else a (i+1) - a i)) :=
by sorry

theorem sum_of_differences (a : Fin n → ℝ) (h : 2 ≤ n) :
  (∑ i in Finset.range (n - 1), a (i+1) - a i) = a (Fin.last n) - a 0 :=
by sorry

end non_extreme_modification_does_not_change_sum_extreme_modification_changes_sum_sum_of_differences_l676_676851


namespace range_of_a_l676_676972

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc (0:ℝ) 1, 2 * a * x^2 - a * x > 3 - a) → a > 24 / 7 := 
by 
  sorry

end range_of_a_l676_676972


namespace four_number_selection_count_correct_l676_676783

open Finset

def valid_tuples (s : Finset ℕ) : Finset (ℕ × ℕ × ℕ × ℕ) :=
  s.product s
    |> filter (λ (abcd : ℕ × ℕ × ℕ × ℕ), abcd.1.1 < abcd.1.2 ∧ abcd.1.2 < abcd.2.1 ∧ abcd.2.1 < abcd.2.2)
    |> filter (λ (abcd : ℕ × ℕ × ℕ × ℕ), abcd.1.1 + abcd.2.1 = abcd.1.2 + abcd.2.2)

noncomputable def four_number_selection_count : ℕ :=
  valid_tuples (range 20).erase 0 |> card

theorem four_number_selection_count_correct :
    four_number_selection_count (Finset.range 20).erase 0 = 525 :=
 by sorry

end four_number_selection_count_correct_l676_676783


namespace positive_difference_between_diagonal_sums_l676_676089

def original_matrix : list (list ℕ) :=
  [[1, 2, 3, 4, 5],
   [6, 7, 8, 9, 10],
   [11, 12, 13, 14, 15],
   [16, 17, 18, 19, 20],
   [21, 22, 23, 24, 25]]

def transformed_matrix : list (list ℕ) :=
  [[1, 2, 3, 4, 5],
   [6, 7, 8, 9, 10],
   [15, 14, 13, 12, 11],
   [16, 17, 18, 19, 20],
   [25, 24, 23, 22, 21]]

def main_diagonal (m : list (list ℕ)) : list ℕ :=
  [m.head!.head!, m.get! 1 |>.get! 1, m.get! 2 |>.get! 2, m.get! 3 |>.get! 3, m.get! 4 |>.get! 4]

def secondary_diagonal (m : list (list ℕ)) : list ℕ :=
  [m.head |>.get! 4, m.get! 1 |>.get! 3, m.get! 2 |>.get! 2, m.get! 3 |>.get! 1, m.get! 4 |>.head!]

def diagonal_sum (diag : list ℕ) : ℕ :=
  diag.sum

theorem positive_difference_between_diagonal_sums :
  let original_m := original_matrix
      new_m := transformed_matrix
      original_main_diag := main_diagonal original_m
      original_secondary_diag := secondary_diagonal original_m
      new_main_diag := main_diagonal new_m
      new_secondary_diag := secondary_diagonal new_m
      original_main_sum := diagonal_sum original_main_diag
      original_secondary_sum := diagonal_sum original_secondary_diag
      new_main_sum := diagonal_sum new_main_diag
      new_secondary_sum := diagonal_sum new_secondary_diag
      original_difference := abs (original_main_sum - original_secondary_sum)
      new_difference := abs (new_main_sum - new_secondary_sum)
  in new_difference = 8 :=
by {
  sorry
}

end positive_difference_between_diagonal_sums_l676_676089


namespace area_of_enclosed_region_l676_676312

-- Define the absolute value function
noncomputable def abs (x : ℝ) : ℝ := if x >= 0 then x else -x

-- Define the curve equation
def curve (x y : ℝ) : Prop := 2 * abs x + 3 * abs y = 5

-- Define the main theorem to prove the area of the enclosed region
theorem area_of_enclosed_region : (∃ x y : ℝ, curve x y) → ∃ area : ℝ, area = 25 / 3 :=
begin
  sorry
end

end area_of_enclosed_region_l676_676312


namespace Woojin_harvested_weight_l676_676482

-- Definitions based on conditions
def younger_brother_harvest : Float := 3.8
def older_sister_harvest : Float := younger_brother_harvest + 8.4
def one_tenth_older_sister : Float := older_sister_harvest / 10
def woojin_extra_g : Float := 3720

-- Convert grams to kilograms
def grams_to_kg (g : Float) : Float := g / 1000

-- Theorem to be proven
theorem Woojin_harvested_weight :
  grams_to_kg (one_tenth_older_sister * 1000 + woojin_extra_g) = 4.94 :=
by
  sorry

end Woojin_harvested_weight_l676_676482


namespace team_B_City_A_degree_l676_676712

theorem team_B_City_A_degree (num_cities : ℕ)
  (teams_per_city : ℕ)
  (max_matches : ℕ)
  (distinct_degrees : set ℕ)
  (team_cityA_A_degree : ℕ)
  (team_cityA_B_degree : ℕ):
  (num_cities = 16) →
  (teams_per_city = 2) →
  (max_matches ≤ 495) → -- since there are at most 495 matches in a complete graph of 32 vertices.
  (distinct_degrees = {0, 1, 2, ..., 30}) →
  (∀ (d1 d2 : ℕ), d1 ∈ distinct_degrees ∧ d2 ∈ distinct_degrees → d1 ≠ d2) →
  (team_cityA_A_degree + team_cityA_B_degree = 30) →
  team_cityA_B_degree = 15 :=
begin
  intros,
  sorry
end

end team_B_City_A_degree_l676_676712


namespace compare_A_B_C_l676_676479

-- Define the expressions A, B, and C
def A : ℚ := (2010 / 2009) + (2010 / 2011)
def B : ℚ := (2010 / 2011) + (2012 / 2011)
def C : ℚ := (2011 / 2010) + (2011 / 2012)

-- The statement asserting A is the greatest
theorem compare_A_B_C : A > B ∧ A > C := by
  sorry

end compare_A_B_C_l676_676479


namespace jesse_money_left_after_mall_l676_676335

theorem jesse_money_left_after_mall :
  ∀ (initial_amount novel_cost lunch_cost total_spent remaining_amount : ℕ),
    initial_amount = 50 →
    novel_cost = 7 →
    lunch_cost = 2 * novel_cost →
    total_spent = novel_cost + lunch_cost →
    remaining_amount = initial_amount - total_spent →
    remaining_amount = 29 :=
by
  intros initial_amount novel_cost lunch_cost total_spent remaining_amount
  sorry

end jesse_money_left_after_mall_l676_676335


namespace largest_quadrilateral_angle_l676_676020

theorem largest_quadrilateral_angle (x : ℝ)
  (h1 : 3 * x + 4 * x + 5 * x + 6 * x = 360) :
  6 * x = 120 :=
by
  sorry

end largest_quadrilateral_angle_l676_676020


namespace cos_phi_minus_A_length_BC_l676_676323

section Problem

variables {A φ : ℝ}
constants (ABC is_triangle : Prop)
constants (area_ABC : ℝ) (AB BC length_ : ℝ)
constants (a b : ℝ × ℝ)
hypothesis h1 : a = (sin A, 1)
hypothesis h2 : b = (cos A, sqrt 3)
hypothesis h3 : ∃ k : ℝ, k ≠ 0 ∧ a = k • b
hypothesis h4 : sin φ = 3/5
hypothesis h5 : 0 < φ ∧ φ < π / 2
hypothesis h6 : area_ABC = 2
hypothesis h7 : AB = 2

theorem cos_phi_minus_A : cos (φ - A) = (3 + 4 * sqrt 3) / 10 := 
sorry

theorem length_BC : BC = 4 := 
sorry

end Problem

end cos_phi_minus_A_length_BC_l676_676323


namespace probability_not_expired_l676_676483

theorem probability_not_expired (total_bottles expired_bottles not_expired_bottles : ℕ) 
  (h1 : total_bottles = 5) 
  (h2 : expired_bottles = 1) 
  (h3 : not_expired_bottles = total_bottles - expired_bottles) :
  (not_expired_bottles / total_bottles : ℚ) = 4 / 5 := 
by
  sorry

end probability_not_expired_l676_676483


namespace sin_sum_inequality_l676_676303

theorem sin_sum_inequality {A B C m : ℝ} (h₁ : A + B + C = π) (h₂ : m ≥ 1) :
  sin (A / m) + sin (B / m) + sin (C / m) ≤ 3 * sin (π / (3 * m)) ∧ 
  (sin (A / m) + sin (B / m) + sin (C / m) = 3 * sin (π / (3 * m)) ↔ A = π / 3 ∧ B = π / 3 ∧ C = π / 3) :=
sorry

end sin_sum_inequality_l676_676303


namespace polynomial_degree_14_l676_676585

theorem polynomial_degree_14 (a b c d e f g h : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) (hg : g ≠ 0)
  (hh : h ≠ 0) :
  degree ((X^5 + C a * X^8 + C b * X^2 + C c) * (X^4 + C d * X^3 + C e * X + C f) * (X^2 + C g * X + C h)) = 14 := 
sorry

end polynomial_degree_14_l676_676585


namespace zero_in_interval_l676_676814

noncomputable def f (x : ℝ) : ℝ := log x / log 3 + 2 * x - 8

theorem zero_in_interval : ∃ x ∈ Set.Ioo 3 4, f x = 0 :=
by {
  -- Given that f(x) = log_base_3(x) + 2x - 8
  have h : ∀ (a b : ℝ), f a * f b < 0 → f a = 0 ∨ f b = 0 ∨ ∃ (c : ℝ), f c = 0 := sorry,
  -- We calculate f(3) and f(4) to find the sign changes
  have f3 : f 3 = (log 3 / log 3) + 2 * 3 - 8 := sorry,
  have f4 : f 4 = (log 4 / log 3) + 2 * 4 - 8 := sorry,
  -- Note we showed that f(3) < 0 and f(4) > 0
  have interval_test : f 3 * f 4 < 0 := sorry,
  -- Apply intermediate value theorem
  exact h 3 4 interval_test
}

end zero_in_interval_l676_676814


namespace area_fraction_of_rhombus_in_square_l676_676772

theorem area_fraction_of_rhombus_in_square :
  let n := 7                 -- grid size
  let side_length := n - 1   -- side length of the square
  let square_area := side_length^2 -- area of the square
  let rhombus_side := Real.sqrt 2 -- side length of the rhombus
  let rhombus_area := 2      -- area of the rhombus
  (rhombus_area / square_area) = 1 / 18 := sorry

end area_fraction_of_rhombus_in_square_l676_676772


namespace find_u_l676_676160

open Matrix
open Real

def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, 2],
  ![4, 0]
]

def I : Matrix (Fin 2) (Fin 2) ℝ := 1

def vec_u : Matrix (Fin 2) (Fin 1) ℝ := ![
  ![0],
  ![32 / 4681]
]

theorem find_u : (B ^ 8 + B ^ 6 + B ^ 4 + B ^ 2 + I) * vec_u = ![
  ![0],
  ![32]
] :=
  sorry

end find_u_l676_676160


namespace scientific_notation_of_coronavirus_diameter_l676_676409

theorem scientific_notation_of_coronavirus_diameter:
  0.000000907 = 9.07 * 10^(-7) :=
sorry

end scientific_notation_of_coronavirus_diameter_l676_676409


namespace geometric_seq_conditions_l676_676722

-- Let a geometric sequence {a_n} be defined with a_1 = 1, a_2 = 4, and a_3 = 16.

theorem geometric_seq_conditions (a : ℕ → ℝ) 
    (h1 : a 1 = 1) 
    (h2 : a 2 = 4) 
    (h3 : a 3 = 16) :
    (h2 = (a 3 = 16)) ∧ (¬ (∀ (q : ℝ), (a 1 = 1 → a 2 = q * a 1 → a 3 = q * a 2))) 
:= sorry

end geometric_seq_conditions_l676_676722


namespace triangle_inequality_inequality_equality_condition_l676_676365

variable (a b c : ℝ)

-- indicating triangle inequality conditions
variable (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)

theorem triangle_inequality_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2*b*(a - b) + b^2*c*(b - c) + c^2*a*(c - a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2*b*(a - b) + b^2*c*(b - c) + c^2*a*(c - a) = 0 ↔ a = b ∧ b = c :=
sorry

end triangle_inequality_inequality_equality_condition_l676_676365


namespace problem1_problem2_l676_676649

-- Problem 1
theorem problem1 (f : ℝ → ℝ) (x : ℝ) (h : ∀ x, f x = abs (x - 1)) :
  f x ≥ (1/2) * (x + 1) ↔ (x ≤ 1/3) ∨ (x ≥ 3) :=
sorry

-- Problem 2
theorem problem2 (g : ℝ → ℝ) (A : Set ℝ) (a : ℝ) 
  (h1 : ∀ x, g x = abs (x - a) - abs (x - 2))
  (h2 : A ⊆ Set.Icc (-1 : ℝ) 3) :
  (1 ≤ a ∧ a < 2) ∨ (2 ≤ a ∧ a ≤ 3) :=
sorry

end problem1_problem2_l676_676649


namespace correct_subset_emptyset_is_0_l676_676550

theorem correct_subset_emptyset_is_0 : (∅ ⊆ {0}) :=
by
  -- Definitions and conditions directly used from step a)
  have A : ¬({0} ∈ {1,2,3}) := -- Definition from condition in step a)
    by sorry
  have C : ¬(0 ∈ ∅) := -- Definition from condition in step a)
    by sorry
  have D : ¬(0 ∩ ∅ = ∅) := -- Definition from condition in step a)
    by sorry

  -- Statement to prove the correct answer
  sorry

end correct_subset_emptyset_is_0_l676_676550


namespace travel_ways_l676_676487

def n_AB := 6
def n_BC := 4
def n_AD := 2
def n_DC := 2

theorem travel_ways : 
  let total_ways_B := n_AB * n_BC in
  let total_ways_D := n_AD * n_DC in
  let total_ways := total_ways_B + total_ways_D in
  total_ways = 28 :=
by
  sorry

end travel_ways_l676_676487


namespace cos_seven_theta_l676_676695

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end cos_seven_theta_l676_676695


namespace additional_baking_soda_correct_l676_676321

/-- The initial conditions provided in the problem statement -/
variables (sugar flour bakingSoda newBakingSoda : ℕ)
variables (ratioSF ratioFB ratioFN : ℚ)
variables (additionalBakingSoda : ℕ)

-- Define the given conditions
def condition1 : ratioSF = 5 / 5 := by simp
def condition2 : ratioFB = 10 / 1 := by simp
def condition3 : ratioFN = 8 / 1 := by simp
def condition4 : sugar = 2400 := rfl

-- Deriving the flour amount from sugar using ratio
def flourAmount : flour = sugar := by simp [condition1, condition4]

-- Initial amount of baking soda from the ratio and flour amount
def initialBakingSoda : bakingSoda = flour / 10 := by 
  rw [flourAmount, condition2]
  exact nat.div_eq_of_lt (flourAmount.trans (by simp))

-- New amount of baking soda from the ratio and flour amount
def newBakingSodaAmount : newBakingSoda = flour / 8 := by 
  rw [flourAmount, condition3]
  exact nat.div_eq_of_lt (flourAmount.trans (by simp))

-- The calculation of additional baking soda
def calculateAdditionalBakingSoda : additionalBakingSoda = newBakingSoda - bakingSoda := by
  rw [newBakingSodaAmount, initialBakingSoda]
  exact sub_self _

theorem additional_baking_soda_correct : additionalBakingSoda = 60 :=
begin
  rw [calculateAdditionalBakingSoda],
  simp [initialBakingSoda, newBakingSodaAmount],
  exact rfl
end

end additional_baking_soda_correct_l676_676321


namespace arc_length_MT_l676_676576

noncomputable def isosceles_triangle (P Q R : Type) (s t : ℝ) (h : PR = QR ∧ PQ > PR) : Prop :=
  ∃ (r : ℝ), r = sqrt (s^2 - (t / 2)^2) / 2 ∧
  ∀ (T : Type) (MTN : Type), intersects_circle_and_lines PQ PR QR P T M N r → arc_len MTN = 360 - 4 * arcsin(t / (2 * s))

theorem arc_length_MT : ∀ (P Q R T MTN : Type) (s t : ℝ)
  (h_isosceles : PR = QR) 
  (h_longer : PQ > PR)
  (h_radius : r = sqrt(s^2 - (t / 2)^2) / 2)
  (h_intersects : intersects_circle_and_lines PQ PR QR P T M N r),
  arc_len MTN = 360 - 4 * arcsin(t / (2 * s)) :=
begin
  sorry
end

end arc_length_MT_l676_676576


namespace train_passing_time_l676_676121

def train_distance_km : ℝ := 10
def train_time_min : ℝ := 15
def train_length_m : ℝ := 111.11111111111111

theorem train_passing_time : 
  let time_to_pass_signal_post := train_length_m / ((train_distance_km * 1000) / (train_time_min * 60))
  time_to_pass_signal_post = 10 :=
by
  sorry

end train_passing_time_l676_676121


namespace imaginary_part_of_fraction_l676_676637

theorem imaginary_part_of_fraction : 
  let i := Complex.I in Complex.imag_part ((5 * i) / (2 - i)) = 2 := by
  sorry

end imaginary_part_of_fraction_l676_676637


namespace simple_interest_rate_l676_676118

-- Definition of the problem's conditions
def principal (P : ℝ) := P
def time (T : ℕ) := 6 -- 6 years
def final_amount (A : ℝ) (P : ℝ) := (7 / 6) * P -- becomes 7/6 of itself

-- Simple interest formula and substitution
def interest (I : ℝ) (P : ℝ) (R : ℝ) (T : ℕ) := (P * R * T) / 100
def given_interest (I : ℝ) (P : ℝ) := P / 6

-- Prove the rate per annum
theorem simple_interest_rate (P R : ℝ) : 
  (P * R * 6) / 100 = P / 6 → R ≈ 2.78 := 
by sorry

end simple_interest_rate_l676_676118


namespace direction_to_orig_position_l676_676728

-- Given definitions from the conditions
variable {T : Type} [AffineSpace T] {point : T}

def rectangular_billiard_table (ABCD : AffineSpace.T T) :=
  ∃ (A B C D : point), is_parallelogram ABCD ∧
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A)

def initial_position (S : point) := ∃ (table: ACL_B), S ∈ table.

theorem direction_to_orig_position
  (table: rectangular_billiard_table ABCD) 
  (S : point, table: rectangular_billiard_table ABCD) (initial_position(S)):
  ∃ (direction : vector),
  (direction = diagonal_AC ∨ direction = diagonal_BD) →
  ball_path S direction table → 
  return_to_initial_position S :=
by
  sorry

end direction_to_orig_position_l676_676728


namespace sum_perpendiculars_eq_l676_676553

theorem sum_perpendiculars_eq (a r : ℝ) (h₁ : a > 0)
  (h_triangle : ∀ (ABC : Triangle), ABC.equilateral)
  (h_inscribed : ∃ (P : Point), P.center_of (InscribedCircle a)) :
  ∑ (PD PE PF : ℝ), PD + PE + PF = 3 * r → r = (a * Real.sqrt 3) / 6 → 
  (PD + PE + PF = (a * Real.sqrt 3) / 2) :=
sorry

end sum_perpendiculars_eq_l676_676553


namespace is_divisible_by_N2_l676_676394

def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

noncomputable def eulers_totient (n : ℕ) : ℕ :=
  Nat.totient n

theorem is_divisible_by_N2 (N1 N2 : ℕ) (h_coprime : are_coprime N1 N2) 
  (k := eulers_totient N2) : 
  (N1 ^ k - 1) % N2 = 0 :=
by
  sorry

end is_divisible_by_N2_l676_676394


namespace cost_per_item_l676_676327

theorem cost_per_item (total_profit : ℝ) (total_customers : ℕ) (purchase_percentage : ℝ) (pays_advertising : ℝ)
    (H1: total_profit = 1000)
    (H2: total_customers = 100)
    (H3: purchase_percentage = 0.80)
    (H4: pays_advertising = 1000)
    : (total_profit / (total_customers * purchase_percentage)) = 12.50 :=
by
  sorry

end cost_per_item_l676_676327


namespace cylinder_base_diameter_l676_676184

theorem cylinder_base_diameter (V : ℝ) (h : ℝ) (d : ℝ) :
  V = 2638.9378290154264 →
  h = 60 →
  d = 2 * real.sqrt (V / (real.pi * h)) →
  d ≈ 7.48 :=
by
  intros V_eq h_eq d_eq
  sorry

end cylinder_base_diameter_l676_676184


namespace total_onions_l676_676763

-- Define the number of onions grown by each individual
def nancy_onions : ℕ := 2
def dan_onions : ℕ := 9
def mike_onions : ℕ := 4

-- Proposition: The total number of onions grown is 15
theorem total_onions : (nancy_onions + dan_onions + mike_onions) = 15 := 
by sorry

end total_onions_l676_676763


namespace initial_distance_is_18_l676_676105

-- Step a) Conditions and Definitions
def distance_covered (v t d : ℝ) : Prop := 
  d = v * t

def increased_speed_time (v t d : ℝ) : Prop := 
  d = (v + 1) * (3 / 4 * t)

def decreased_speed_time (v t d : ℝ) : Prop := 
  d = (v - 1) * (t + 3)

-- Step c) Mathematically Equivalent Proof Problem
theorem initial_distance_is_18 (v t d : ℝ) 
  (h1 : distance_covered v t d) 
  (h2 : increased_speed_time v t d) 
  (h3 : decreased_speed_time v t d) : 
  d = 18 :=
sorry

end initial_distance_is_18_l676_676105


namespace parallelogram_eq_l676_676005

variable (a b e p_a p_b : ℝ)

-- Given conditions
def adjacent_sides := a > 0 ∧ b > 0
def longer_diagonal := e > sqrt(a^2 + b^2)
def projection_pa := p_a > 0 ∧ p_a = abs(e * cos(arccos(a / e)))
def projection_pb := p_b > 0 ∧ p_b = abs(e * cos(arccos(b / e)))

-- The theorem to prove
theorem parallelogram_eq (h1 : adjacent_sides a b) 
                         (h2 : longer_diagonal a b e) 
                         (h3 : projection_pa a e p_a)
                         (h4 : projection_pb b e p_b) : 
    a * p_a + b * p_b = e^2 :=
  sorry

end parallelogram_eq_l676_676005


namespace jeans_cost_before_sales_tax_l676_676975

-- Defining conditions
def original_cost : ℝ := 49
def summer_discount : ℝ := 0.50
def wednesday_discount : ℝ := 10

-- The mathematical equivalent proof problem
theorem jeans_cost_before_sales_tax :
  let discount_price := original_cost * (1 - summer_discount)
  let wednesday_price := discount_price - wednesday_discount
  wednesday_price = 14.50 :=
by
  let discount_price := original_cost * (1 - summer_discount)
  let wednesday_price := discount_price - wednesday_discount
  sorry

end jeans_cost_before_sales_tax_l676_676975


namespace find_correct_result_l676_676845

noncomputable def correct_result : Prop :=
  ∃ (x : ℝ), (-1.25 * x - 0.25 = 1.25 * x) ∧ (-1.25 * x = 0.125)

theorem find_correct_result : correct_result :=
  sorry

end find_correct_result_l676_676845


namespace smallest_integer_m_l676_676612

def largest_square_le (x : ℕ) : ℕ :=
  let sqrt_x := (Nat.sqrt x)
  sqrt_x * sqrt_x

def next_number (x : ℕ) : ℕ :=
  x - largest_square_le x

def sequence_length (m : ℕ) : ℕ :=
  let rec seq_len (n : ℕ) (count : ℕ) : ℕ :=
    if n = 0 then count + 1
    else seq_len (next_number n) (count + 1)
  seq_len m 0

theorem smallest_integer_m : ∃ m : ℕ, m = 52 ∧ sequence_length 52 = 9 := by
  use 52
  split
  . rfl
  . sorry

end smallest_integer_m_l676_676612


namespace num_even_numbers_less_than_50k_l676_676549

theorem num_even_numbers_less_than_50k : 
  let digits := [1, 2, 3, 4, 5] in 
  let even_numbers := [2, 4] in 
  -- The unit digit must be one of the even numbers and therefore has 2 choices
  -- The first digit cannot be 5 since the number must be less than 50,000.
  -- So the first digit has 3 choices (excluding the unit digit)
  -- The remaining 3 positions must arrange the rest digits (3! ways)
  -- Thus there are 2 * 3 * 6 = 36 suitable five-digit numbers
  card {n : ℕ | ∃ (d1 d2 d3 d4 d5 : ℕ), 
    (d1 ∈ digits) ∧ (d2 ∈ digits) ∧ (d3 ∈ digits) ∧ (d4 ∈ digits) ∧ (d5 ∈ even_numbers) ∧ 
    (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d3 ≠ d4 ∧ d3 ≠ d5 ∧ d4 ≠ d5) ∧ 
    n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧ n < 50000 } = 36 :=
by 
  sorry

end num_even_numbers_less_than_50k_l676_676549


namespace cos_seven_theta_l676_676693

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end cos_seven_theta_l676_676693


namespace box_height_l676_676328

theorem box_height (x h : ℕ) 
  (h1 : h = x + 5) 
  (h2 : 6 * x^2 + 20 * x ≥ 150) 
  (h3 : 5 * x + 5 ≥ 25) 
  : h = 9 :=
by 
  sorry

end box_height_l676_676328


namespace mixed_number_fraction_division_and_subtraction_l676_676060

theorem mixed_number_fraction_division_and_subtraction :
  ( (11 / 6) / (11 / 4) ) - (1 / 2) = 1 / 6 := 
sorry

end mixed_number_fraction_division_and_subtraction_l676_676060


namespace misha_total_shots_l676_676846

theorem misha_total_shots (x y : ℕ) 
  (h1 : 18 * x + 5 * y = 99) 
  (h2 : 2 * x + y = 15) 
  (h3 : (15 / 0.9375 : ℝ) = 16) : 
  (¬(x = 0) ∧ ¬(y = 24)) ->
  16 = 16 :=
by
  sorry

end misha_total_shots_l676_676846


namespace range_of_f_l676_676578
   
   noncomputable def f (x : ℝ) : ℝ := (⌊ x ⌋₊ : ℝ) - x + sin x
   
   theorem range_of_f :
     (∀ x : ℝ, -1 ≤ f x) ∧ (∀ x : ℝ, f x ≤ 0) :=
   sorry
   
end range_of_f_l676_676578


namespace modulus_of_conjugate_l676_676621

theorem modulus_of_conjugate (z : ℂ) (hz : (1 + sqrt 3 * complex.i) * z = 1 - complex.i) : complex.abs (complex.conj z) = sqrt 2 / 2 :=
by {
  sorry,
}

end modulus_of_conjugate_l676_676621


namespace product_f_is_multiple_l676_676392

def f (m : ℕ) (k : ℕ) (t : ℕ) : ℕ :=
  t^(1 - k)

theorem product_f_is_multiple (n a : ℕ) (h1 : 1 ≤ n) (h2: a ≤ n) (h3: a % 2 = 1) :
  ∀ k t, (m = 2^k * t) ∧ (t % 2 = 1) → ∃ c, ∏ m in (finset.range n).succ, f m k t = c * a :=
by
  sorry

end product_f_is_multiple_l676_676392


namespace find_valid_three_digits_l676_676965

noncomputable def valid_digits (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

theorem find_valid_three_digits (a b c : ℕ) (n : ℕ) :
  valid_digits a → valid_digits b → valid_digits c →
  b * (10 * a + c) = c * (10 * a + b) + 10 →
  a = 1 ∧ b = c + 1 ∧
  n = 100 * a + 10 * b + c →
  n ∈ {112, 123, 134, 145, 156, 167, 178, 189} :=
by {
  intros ha hb hc h_eq h_n,
  sorry
}

end find_valid_three_digits_l676_676965


namespace domain_of_f_l676_676162

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (6 - Real.sqrt (7 - Real.sqrt x))

theorem domain_of_f : {x : ℝ | 0 ≤ x ∧ x ≤ 49 ∧ 6 - Real.sqrt (7 - Real.sqrt x) ≥ 0} = set.Icc 0 49 :=
by
  sorry

end domain_of_f_l676_676162


namespace candice_gets_home_time_l676_676912

-- Define the Lean problem setup
theorem candice_gets_home_time :
  ∃ (n m : ℕ), (∀ t, 1 ≤ t ∧ t ≤ n - m + 1 → 0 < n - t + 1) ∧
  (1 / 60 * (∑ k in finset.range (n - m + 1), (n - k)) = 2 / 3) ∧ 
  5:00 PM + (n - m + 1).minutes = 5:05 PM :=
by {
  -- We leave the proof as an exercise for the reader
  sorry
}

end candice_gets_home_time_l676_676912


namespace plane_divided_into_four_regions_l676_676154

-- Definition of the conditions
def line1 (x y : ℝ) : Prop := y = 3 * x
def line2 (x y : ℝ) : Prop := y = (1 / 3) * x

-- Proof statement
theorem plane_divided_into_four_regions :
  (∃ f g : ℝ → ℝ, ∀ x, line1 x (f x) ∧ line2 x (g x)) →
  ∃ n : ℕ, n = 4 :=
by sorry

end plane_divided_into_four_regions_l676_676154


namespace distance_between_points_l676_676935

theorem distance_between_points :
  ∀ (P Q : ℝ × ℝ), P = (3, 3) ∧ Q = (-2, -2) → dist P Q = 5 * real.sqrt 2 :=
begin
  sorry
end

end distance_between_points_l676_676935


namespace find_mangoes_l676_676907

def cost_of_grapes : ℕ := 8 * 70
def total_amount_paid : ℕ := 1165
def cost_per_kg_of_mangoes : ℕ := 55

theorem find_mangoes (m : ℕ) : cost_of_grapes + m * cost_per_kg_of_mangoes = total_amount_paid → m = 11 :=
by
  sorry

end find_mangoes_l676_676907


namespace largest_fraction_l676_676849

theorem largest_fraction :
  (∀ x ∈ {5/13, 7/16, 23/46, 203/405}, 51/101 > x) :=
by
  sorry

end largest_fraction_l676_676849


namespace triplet_solutions_l676_676183

theorem triplet_solutions : 
  (∃ (x y z : ℝ), x + y + z = 2 ∧ x^2 + y^2 + z^2 = 26 ∧ x^3 + y^3 + z^3 = 38) ↔
  (1, -3, 4) = (1, -3, 4) ∨ (1, 4, -3) ∨ (-3, 1, 4) ∨ (-3, 4, 1) ∨ (4, 1, -3) ∨ (4, -3, 1) := 
sorry

end triplet_solutions_l676_676183


namespace inclination_angle_of_line_l676_676017

def line_equation (x y : ℝ) : Prop := x * (Real.tan (Real.pi / 3)) + y + 2 = 0

theorem inclination_angle_of_line (x y : ℝ) (h : line_equation x y) : 
  ∃ α : ℝ, α = 2 * Real.pi / 3 ∧ 0 ≤ α ∧ α < Real.pi := by
  sorry

end inclination_angle_of_line_l676_676017


namespace product_of_removed_odd_numbers_l676_676976

theorem product_of_removed_odd_numbers 
  (S : Nat := (1 + 100) * 100 / 2)
  (n : Nat := 100)
  (average_remaining : Nat := 51)
  (a : Nat)
  (odd_pair : a % 2 = 1 ∧ (a + 2) % 2 = 1)
  (sum_removed : Nat := 2 * a + 2)
  (n_remaining : Nat := n - 2)
  (S' : Nat := S - sum_removed) :
  (S' / n_remaining = average_remaining) → (a = 25) → (a + 2 = 27) →
  a * (a + 2) = 675 :=
sorry

end product_of_removed_odd_numbers_l676_676976


namespace valid_base6_number_2015_l676_676129

def is_valid_base6_digit (d : Nat) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def is_base6_number (n : Nat) : Prop :=
  ∀ (digit : Nat), digit ∈ (n.digits 10) → is_valid_base6_digit digit

theorem valid_base6_number_2015 : is_base6_number 2015 := by
  sorry

end valid_base6_number_2015_l676_676129


namespace area_of_triangle_BEF_l676_676326

theorem area_of_triangle_BEF 
(triangle_ABC : Triangle)
(R : ℝ)
(a : ℝ)
(h_inscribed_circle: 
  ∃ (D E F : Point),
    inscribed_circle triangle_ABC D E F ∧
    dist A D = R ∧
    dist D C = a ∧
    touches circle_AC side_AC D ∧
    touches circle_AB side_AB E ∧
    touches circle_BC side_BC F) :
  area (triangle_BEZ : Triangle)
  = R^2 * (R + a)^3 / (2 * (a - R) * (a^2 + R^2)) :=
sorry

end area_of_triangle_BEF_l676_676326


namespace find_unit_price_B_l676_676872

variable (x : ℕ)

def unit_price_B := x
def unit_price_A := x + 50

theorem find_unit_price_B (h : (2000 / unit_price_A x = 1500 / unit_price_B x)) : unit_price_B x = 150 :=
by
  sorry

end find_unit_price_B_l676_676872


namespace log_identity_property_problem_statement_l676_676170

theorem log_identity_property (x : ℝ) (b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) (h3 : x > 0) : 
  b^(Real.logb b x) = x := by
  sorry

theorem problem_statement : 5^(Real.logb 5 3) = 3 := by
  exact log_identity_property 3 5 by norm_num by norm_num by norm_num

end log_identity_property_problem_statement_l676_676170


namespace calculate_ratio_l676_676360

theorem calculate_ratio (l m n : ℝ) :
  let D := (l + 1, 1, 1)
  let E := (1, m + 1, 1)
  let F := (1, 1, n + 1)
  let AB_sq := 4 * ((n - m) ^ 2)
  let AC_sq := 4 * ((l - n) ^ 2)
  let BC_sq := 4 * ((m - l) ^ 2)
  (AB_sq + AC_sq + BC_sq + 3) / (l^2 + m^2 + n^2 + 3) = 8 := by
  sorry

end calculate_ratio_l676_676360


namespace cos_theta_seven_l676_676689

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end cos_theta_seven_l676_676689


namespace investment_problem_l676_676528

theorem investment_problem :
  ∃ (S G : ℝ), S + G = 10000 ∧ 0.06 * G = 0.05 * S + 160 ∧ S = 4000 :=
by
  sorry

end investment_problem_l676_676528


namespace largest_A_l676_676476

def A : ℝ := (2010 / 2009) + (2010 / 2011)
def B : ℝ := (2010 / 2011) + (2012 / 2011)
def C : ℝ := (2011 / 2010) + (2011 / 2012)

theorem largest_A : A > B ∧ A > C := by sorry

end largest_A_l676_676476


namespace fifty_one_numbers_multiple_l676_676198

open Finset

def exists_multiple_of_another (s : Finset ℕ) : Prop :=
  ∃ (a b ∈ s), a ≠ b ∧ (a ∣ b ∨ b ∣ a)

theorem fifty_one_numbers_multiple :
  ∀ (s : Finset ℕ), (s.card = 51) → (∀ x ∈ s, x ∈ (range 101 : Finset ℕ)) → exists_multiple_of_another s :=
by
  intros s h_card h_range
  sorry

end fifty_one_numbers_multiple_l676_676198


namespace quadratic_one_real_root_positive_m_l676_676164

theorem quadratic_one_real_root_positive_m (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ((6 * m)^2 - 4 * 1 * (2 * m) = 0)) → m = 2 / 9 :=
by
  sorry

end quadratic_one_real_root_positive_m_l676_676164


namespace relation_between_a_and_c_l676_676618

-- Given definitions and conditions
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

theorem relation_between_a_and_c (a c : ℝ) (h_nonzero : a ≠ 0) :
  (∀ α β : ℝ, β = 3 * α → 
  quadratic_eq a 6 c α = 0 → 
  quadratic_eq a 6 c β = 0) →
  c = 27 / (4 * a) :=
by
  intro h
  sorry

end relation_between_a_and_c_l676_676618


namespace proof_mn_l676_676525

noncomputable def thm_proof : ℝ :=
by
  let n := sqrt (3 / 13)
  let m := 9 * n
  let res := m * n
  exact res

theorem proof_mn (m n : ℝ) (h1 : m = 9 * n) (h2 : n = sqrt (3 / 13)) :
  m * n = 81 / 13 :=
by
  apply thm_proof, sorry

end proof_mn_l676_676525


namespace solve_equation_l676_676451

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end solve_equation_l676_676451


namespace percentage_less_l676_676079

theorem percentage_less (P T J : ℝ) (hT : T = 0.9375 * P) (hJ : J = 0.8 * T) : (P - J) / P * 100 = 25 :=
by
  sorry

end percentage_less_l676_676079


namespace find_value_l676_676978

def equation := ∃ x : ℝ, x^2 - 2 * x - 3 = 0
def expression (x : ℝ) := 2 * x^2 - 4 * x + 12

theorem find_value :
  (∃ x : ℝ, (x^2 - 2 * x - 3 = 0) ∧ (expression x = 18)) :=
by
  sorry

end find_value_l676_676978


namespace coeff_x7_term_in_expansion_l676_676721

theorem coeff_x7_term_in_expansion :
  let C (n k : ℕ) := Nat.choose n k
  ∃ (c : ℝ), 
    (∀ x : ℝ, (∀ n : ℕ, n = 7 → (2*x - real.sqrt x)^10 = c * x^7 + (polynomial.monomial 10)) →
         c = C 10 6 * 2^4) :=
sorry

end coeff_x7_term_in_expansion_l676_676721


namespace probability_of_at_least_one_solving_l676_676832

variable (P1 P2 : ℝ)

theorem probability_of_at_least_one_solving : 
  (1 - (1 - P1) * (1 - P2)) = P1 + P2 - P1 * P2 := 
sorry

end probability_of_at_least_one_solving_l676_676832


namespace tan_sum_identity_l676_676145

theorem tan_sum_identity :
  ∀ (x y z : ℝ), tan 60 = sqrt 3 → 
  tan (x + y) = (tan x + tan y) / (1 - tan x * tan y) →
  x = 10 → y = 50 → z = 120 →
  (tan x + tan y + tan z) / (tan x * tan y) = - sqrt 3 :=
by
  -- Formal proof to be filled in here
  sorry

end tan_sum_identity_l676_676145


namespace Maria_test_scores_l676_676544

noncomputable def Maria_tests : Prop :=
  ∃ (scores : List ℕ), (length scores = 6) ∧
  (average (take 4 scores) = 76) ∧
  (nth scores 0 = 81) ∧
  (nth scores 1 = 65) ∧
  (average scores = 79) ∧
  (∀ s ∈ scores, s < 95) ∧
  (∀ (i j : ℕ), i ≠ j → nth scores i ≠ nth scores j) ∧
  (sort (· > ·) scores = [94, 93, 92, 81, 78, 75, 65])

theorem Maria_test_scores : Maria_tests :=
  sorry

end Maria_test_scores_l676_676544


namespace angle_AYX_in_incircle_circumcircle_ABC_l676_676351

theorem angle_AYX_in_incircle_circumcircle_ABC :
  ∀ (A B C X Y Z : Point) (Γ : Circle),
  is_incircle A B C Γ ∧
  is_circumcircle X Y Z Γ ∧
  is_isosceles_triangle A B C ∧
  angle A B C = 65 ∧
  angle B A C = 65 ∧
  angle B A B = 50 ∧
  on_side X B C ∧
  on_side Y A B ∧
  on_side Z A C ∧
  is_equilateral_triangle X Y Z →
  angle A Y X = 130 :=
by
  sorry

end angle_AYX_in_incircle_circumcircle_ABC_l676_676351


namespace probability_at_least_two_consecutive_heads_l676_676514

theorem probability_at_least_two_consecutive_heads :
  (∃ h, probability (at_least_two_consecutive_heads 4 = h) ∧ h = 3 / 4) :=
sorry

end probability_at_least_two_consecutive_heads_l676_676514


namespace number_of_students_l676_676406

theorem number_of_students (n : ℕ)
  (h1 : ∃ n, (175 * n) / n = 175)
  (h2 : 175 * n - 40 = 173 * n) :
  n = 20 :=
sorry

end number_of_students_l676_676406


namespace scientific_notation_of_274000000_l676_676768

theorem scientific_notation_of_274000000 :
  274000000 = 2.74 * 10^8 := by
  sorry

end scientific_notation_of_274000000_l676_676768


namespace Joao_received_fraction_l676_676739

variable (A : ℝ) -- Let A be the amount of money each friend gave João.
variables (jMoney : ℝ) (jorgMoney : ℝ) (joseMoney : ℝ) (janMoney : ℝ)

-- Conditions
axiom Jorge_condition : jorgMoney = 5 * A
axiom Jose_condition : joseMoney = 4 * A
axiom Janio_condition : janMoney = 3 * A
axiom same_amount : jMoney = 3 * A

-- Theorem statement proving the fraction of the group's total money João received.
theorem Joao_received_fraction : 
  (3 * A) / (jorgMoney + joseMoney + janMoney) = 1 / 4 :=
by
  -- Simplifying to the equivalent fraction.
  rw [Jorge_condition, Jose_condition, Janio_condition]
  -- Simplifying the numerator and the denominator
  sorry

end Joao_received_fraction_l676_676739


namespace find_angle_A_max_area_l676_676631

-- Definitions and conditions
variable (A B C a b c : ℝ)
variable (acute_abc : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
variable (opposite_sides : b = a * cos C + (sqrt 3 / 3) * c * sin A)
variable (a_value : a = 2)

-- Proof statement for question 1
theorem find_angle_A (acute_abc : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π) 
  (opposite_sides : b = a * cos C + (sqrt 3 / 3) * c * sin A) :
  A = π / 3 := sorry

-- Proof statement for question 2
theorem max_area (acute_abc : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (opposite_sides : b = a * cos C + (sqrt 3 / 3) * c * sin A)
  (a_value : a = 2) :
  ∃ area : ℝ, area ≤ sqrt 3 ∧ (∀ other_area : ℝ, other_area ≤ sqrt 3 → other_area = 
  sqrt 3 → area = sqrt 3) := sorry

end find_angle_A_max_area_l676_676631


namespace problem1_problem2_problem3_l676_676391

def sequence_value (x1 x2 x3 : ℝ) : ℝ :=
  min (|x1|) (min (|x1 + x2| / 2) (|x1 + x2 + x3| / 3))

theorem problem1 : sequence_value (-4) (-3) 2 = 5 / 3 := sorry

theorem problem2 : ∃ s : List ℝ, s.permutations.contains [-4,2,-3] ∧ (sequence_value s.head s.tail.head s.tail.tail.head = 1 / 2)  := sorry

theorem problem3 (a : ℝ) (h : 1 < a) :
  (∀ s : List ℝ, s.permutations.contains [2,-9,a] → sequence_value s.head s.tail.head s.tail.tail.head = 1) →
  (a = 11 ∨ a = 4) := sorry

end problem1_problem2_problem3_l676_676391


namespace count_integers_satisfying_sqrt_condition_l676_676291

theorem count_integers_satisfying_sqrt_condition :
  ∀ x : ℕ, (⌈(Real.sqrt x)⌉ = 20) → (∃ n : ℕ, n = 39) :=
by {
  intro x,
  intro sqrt_x_ceil_eq_twenty,
  sorry
}

end count_integers_satisfying_sqrt_condition_l676_676291


namespace abs_diff_eq_implies_le_l676_676865

theorem abs_diff_eq_implies_le {x y : ℝ} (h : |x - y| = y - x) : x ≤ y := 
by
  sorry

end abs_diff_eq_implies_le_l676_676865


namespace simplify_expression_l676_676924

theorem simplify_expression (x y : ℝ) :
  (x^2 + y^2)⁻¹ + (x⁻² + y⁻²) = (x^2 + y^2) * x⁻² * y⁻² :=
by sorry

end simplify_expression_l676_676924


namespace find_M_range_of_a_l676_676583

def Δ (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

def A : Set ℝ := { x | 4 * x^2 + 9 * x + 2 < 0 }

def B : Set ℝ := { x | -1 < x ∧ x < 2 }

def M : Set ℝ := Δ B A

def P (a: ℝ) : Set ℝ := { x | (x - 2 * a) * (x + a - 2) < 0 }

theorem find_M :
  M = { x | -1/4 ≤ x ∧ x < 2 } :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ M → x ∈ P a) →
  a < -1/8 ∨ a > 9/4 :=
sorry

end find_M_range_of_a_l676_676583


namespace cost_of_paper_l676_676807

noncomputable def cost_of_paper_per_kg (edge_length : ℕ) (coverage_per_kg : ℕ) (expenditure : ℕ) : ℕ :=
  let surface_area := 6 * edge_length * edge_length
  let paper_needed := surface_area / coverage_per_kg
  expenditure / paper_needed

theorem cost_of_paper (h1 : edge_length = 10) (h2 : coverage_per_kg = 20) (h3 : expenditure = 1800) : 
  cost_of_paper_per_kg 10 20 1800 = 60 :=
by
  -- Using the hypothesis to directly derive the result.
  unfold cost_of_paper_per_kg
  sorry

end cost_of_paper_l676_676807


namespace find_radius_r_l676_676892

noncomputable def radius_of_intersection 
  (Sphere_center : ℝ × ℝ × ℝ) 
  (xy_center : ℝ × ℝ × ℝ) 
  (xy_radius : ℝ) 
  (yz_center : ℝ × ℝ × ℝ) : ℝ :=
sqrt ((sqrt (xy_radius ^ 2 + (Sphere_center.3 - xy_center.3) ^ 2)) ^ 2 - (Sphere_center.1 - yz_center.1) ^ 2)

theorem find_radius_r 
  (Sphere_center : ℝ × ℝ × ℝ := (2, 4, -7))
  (xy_center : ℝ × ℝ × ℝ := (2, 4, 0))
  (xy_radius : ℝ := 1)
  (yz_center : ℝ × ℝ × ℝ := (0, 4, -7)) :
  radius_of_intersection Sphere_center xy_center xy_radius yz_center = sqrt 46 :=
by
sorry

end find_radius_r_l676_676892


namespace integer_solution_count_l676_676176

-- Definitions based on the problem's conditions
def f (x : ℝ) := Real.logb 7 x
def g (x : ℝ) := 80 + x - (7^80 : ℝ)

-- Statement of the proof problem:
theorem integer_solution_count :
  let S := (1 / 2) * (7^160 : ℝ) - (2 / 3) * (7^80 : ℝ) + 487 / 6 in
  ∃ (x y : ℤ), y ≥ 80 + x - (7^80 : ℤ) ∧ y ≤ Real.logb 7 x ∧ 
  (((7^80 : ℤ) + 1) * (7^80) / 2 - 7 * ((7^81 - 1) / 6 - 80))
  -- Adding the final number of pairs should satisfy S
  sorry

end integer_solution_count_l676_676176


namespace line_eqn_with_given_conditions_l676_676797

theorem line_eqn_with_given_conditions : 
  ∃(m c : ℝ), (∀ x y : ℝ, y = m*x + c → x + y - 3 = 0) ↔ 
  ∀ x y, x + y = 3 :=
sorry

end line_eqn_with_given_conditions_l676_676797


namespace ducks_remaining_l676_676817

theorem ducks_remaining (D_0 : ℕ) (D_0_eq : D_0 = 320) :
  let D_1 := D_0 - D_0 / 4,
      D_2 := D_1 - D_1 / 6,
      D_3 := D_2 - (3 * D_2) / 10 in
  D_3 = 140 :=
by
  sorry

end ducks_remaining_l676_676817


namespace circle_standard_equation_l676_676640

noncomputable def standard_equation_of_circle : String :=
  let radius := 3
  let h := 0
  let k := 1
  let r_squared := radius^2
  s!"(x - {h})^2 + (y - {k})^2 = {r_squared}"

theorem circle_standard_equation :
  let radius := 3
  let center := (1, 0)
  let symmetric_center := (0, 1)
  let expected_equation := "x^2 + (y - 1)^2 = 9"
  bool :=
  -- proving the symmetry condition
  symmetric_center.1 == center.2 &&
  symmetric_center.2 == center.1 &&
  -- checking the correctness of the standard equation
  standard_equation_of_circle = expected_equation
by simp [standard_equation_of_circle]; sorry

end circle_standard_equation_l676_676640


namespace grid_integer_count_l676_676709

theorem grid_integer_count (grid : Fin 100 × Fin 100 → ℤ)
  (h : ∀ (i j : Fin 100) (i' j' : Fin 100), ((i = i' ∧ (j.val - j'.val).abs = 1) ∨ (j = j' ∧ (i.val - i'.val).abs = 1)) → 
    (grid (i, j) - grid (i', j')).abs ≤ 20) : 
  ∃ (v : ℤ), ∃ (i j k : Fin 100 × Fin 100), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ grid i = v ∧ grid j = v ∧ grid k = v :=
by 
  sorry

end grid_integer_count_l676_676709


namespace complex_power_l676_676574

theorem complex_power : (1 + complex.i)^6 = -8 * complex.i :=
by
  sorry

end complex_power_l676_676574


namespace problem_1_problem_2_problem_3_l676_676349

def M := {n : ℕ | 0 < n ∧ n < 1000}

def circ (a b : ℕ) : ℕ :=
  if a * b < 1000 then a * b
  else 
    let k := (a * b) / 1000
    let r := (a * b) % 1000
    if k + r < 1000 then k + r
    else (k + r) % 1000 + 1

theorem problem_1 : circ 559 758 = 146 := 
by
  sorry

theorem problem_2 : ∃ (x : ℕ) (h : x ∈ M), circ 559 x = 1 ∧ x = 361 :=
by
  sorry

theorem problem_3 : ∀ (a b c : ℕ) (h₁ : a ∈ M) (h₂ : b ∈ M) (h₃ : c ∈ M), circ a (circ b c) = circ (circ a b) c :=
by
  sorry

end problem_1_problem_2_problem_3_l676_676349


namespace sequence_a_n_l676_676651

theorem sequence_a_n (a : ℕ → ℚ)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n + 1) * (a n + 1) = a n) :
  a 6 = 1 / 6 :=
  sorry

end sequence_a_n_l676_676651


namespace min_value_fraction_l676_676638

theorem min_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : -3 ≤ y ∧ y ≤ 1) : (x + y) / x = 0.8 :=
by
  sorry

end min_value_fraction_l676_676638


namespace probability_event_A_l676_676698

open Classical
noncomputable theory

-- Define the finite set of university graduates
def graduates := {A, B, C, D, E}

-- Define the event "either A or B is hired" as a subset
def event_A (s : Set graduates) : Prop := A ∈ s ∨ B ∈ s

-- Define the number of ways to choose 3 out of 5
def comb_5_3 : ℕ := Nat.choose 5 3

-- Define the number of ways to choose 3 out of {C, D, E}
def comb_3_3 : ℕ := Nat.choose 3 3

-- Define the probability of the complement event 
def P_compl_A : ℚ := comb_3_3 / comb_5_3

-- Define the probability of event A 
def P_A : ℚ := 1 - P_compl_A

-- The theorem to prove
theorem probability_event_A : P_A = 9 / 10 := by
  sorry

end probability_event_A_l676_676698


namespace root_in_interval_l676_676018

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 3^x

theorem root_in_interval : ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  have h1 : f 1 = -1 := by
    simp [f, pow_succ']
    norm_num

  have h2 : f 2 = 1 := by
    simp [f, pow_succ']
    norm_num

  have intermediate_value := intermediate_value f h1 h2 (by simp [h1, h2])
  exact intermediate_value

end root_in_interval_l676_676018


namespace part_a_part_b_l676_676778

-- Part (a)
theorem part_a (k : ℕ) (hk : 1 ≤ k) (a : Fin k → ℝ) :
  (∑ i, a i)^2 ≤ k * ∑ i, (a i)^2 :=
sorry

-- Part (b)
theorem part_b (n : ℕ) (hn : 1 ≤ n) (a : Fin n → ℝ)
  (h : ∑ i, a i ≥ Real.sqrt ((n - 1) * ∑ i, (a i)^2)) :
  ∀ i, 0 ≤ a i :=
sorry

end part_a_part_b_l676_676778


namespace janet_total_cost_l676_676124

noncomputable def admission_price : ℝ := 30
noncomputable def num_people : ℕ := 10
noncomputable def num_children : ℕ := 4
noncomputable def soda_price : ℝ := 5
noncomputable def discount : ℝ := 0.20

def total_cost (admission_price : ℝ) (num_people : ℕ) (num_children : ℕ) (soda_price : ℝ) (discount : ℝ) : ℝ := 
  let child_price := admission_price / 2
  let num_adults := num_people - num_children
  let total_before_discount := (num_children * child_price) + (num_adults * admission_price)
  let discount_amount := total_before_discount * discount
  (total_before_discount - discount_amount) + soda_price

theorem janet_total_cost : total_cost admission_price num_people num_children soda_price discount = 197 := 
by
  sorry

end janet_total_cost_l676_676124


namespace monotonic_decreasing_interval_l676_676022

noncomputable def f (x : ℝ) := x^3 - 3 * x
noncomputable def f' (x : ℝ) := 3 * x^2 - 3

theorem monotonic_decreasing_interval :
  ∀ x, x ∈ Ioo (-1 : ℝ) (1 : ℝ) → f' x < 0 :=
by
  sorry

end monotonic_decreasing_interval_l676_676022


namespace solve_equation_l676_676424

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end solve_equation_l676_676424


namespace inverse_sqrt_radius_sum_l676_676458

variable {R1 R2 R3 : ℝ}

theorem inverse_sqrt_radius_sum (h1 : 0 < R1)
                               (h2 : 0 < R2)
                               (h3 : 0 < R3)
                               (cond1 : distances O1 O2 = R1 + R2)
                               (cond2 : distances O2 O3 = R2 + R3)
                               (cond3 : distances O3 O1 = R3 + R1) :
  1 / Real.sqrt R2 = 1 / Real.sqrt R1 + 1 / Real.sqrt R3 := 
sorry

end inverse_sqrt_radius_sum_l676_676458


namespace count_integers_satisfying_sqrt_condition_l676_676288

theorem count_integers_satisfying_sqrt_condition :
  ∀ x : ℕ, (⌈(Real.sqrt x)⌉ = 20) → (∃ n : ℕ, n = 39) :=
by {
  intro x,
  intro sqrt_x_ceil_eq_twenty,
  sorry
}

end count_integers_satisfying_sqrt_condition_l676_676288


namespace find_numbers_satisfying_conditions_l676_676452

theorem find_numbers_satisfying_conditions (x y z : ℝ)
(h1 : x + y + z = 11 / 18)
(h2 : 1 / x + 1 / y + 1 / z = 18)
(h3 : 2 / y = 1 / x + 1 / z) :
x = 1 / 9 ∧ y = 1 / 6 ∧ z = 1 / 3 :=
sorry

end find_numbers_satisfying_conditions_l676_676452


namespace second_train_length_l676_676056

theorem second_train_length
  (train1_length : ℝ)
  (train1_speed_kmph : ℝ)
  (train2_speed_kmph : ℝ)
  (time_to_clear : ℝ)
  (h1 : train1_length = 135)
  (h2 : train1_speed_kmph = 80)
  (h3 : train2_speed_kmph = 65)
  (h4 : time_to_clear = 7.447680047665153) :
  ∃ l2 : ℝ, l2 = 165 :=
by
  let train1_speed_mps := train1_speed_kmph * 1000 / 3600
  let train2_speed_mps := train2_speed_kmph * 1000 / 3600
  let total_distance := (train1_speed_mps + train2_speed_mps) * time_to_clear
  have : total_distance = 300 := by sorry
  have l2 := total_distance - train1_length
  use l2
  have : l2 = 165 := by sorry
  assumption

end second_train_length_l676_676056


namespace solve_equation_l676_676423

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end solve_equation_l676_676423


namespace range_of_a_l676_676011

def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then a * (x - 1)^2 + 1 else (a + 3) * x + 4 * a

def is_increasing (f : ℝ → ℝ) : Prop :=
∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem range_of_a (a : ℝ) :
    is_increasing (λ x, f x a) ↔ (-2/5 <= a ∧ a < 0) :=
sorry

end range_of_a_l676_676011


namespace initial_percentage_is_30_l676_676090

def percentage_alcohol (P : ℝ) : Prop :=
  let initial_alcohol := (P / 100) * 50
  let mixed_solution_volume := 50 + 30
  let final_percentage_alcohol := 18.75
  let final_alcohol := (final_percentage_alcohol / 100) * mixed_solution_volume
  initial_alcohol = final_alcohol

theorem initial_percentage_is_30 :
  percentage_alcohol 30 :=
by
  unfold percentage_alcohol
  sorry

end initial_percentage_is_30_l676_676090


namespace continuous_piecewise_l676_676373

def f (x : ℝ) (a b : ℝ) : ℝ :=
if x > 2 then a * x + 3
else if x >= -2 then x - 5
else 2 * x - b

theorem continuous_piecewise (a b : ℝ) :
  (∀ x > 2, continuous_at (λ x, a * x + 3) x) →
  (∀ x ≤ 2, x ≥ -2, continuous_at (λ x, x - 5) x) →
  (∀ x < -2, continuous_at (λ x, 2 * x - b) x) →
  (∀ x, (∀ ε > 0, ∃ δ > 0, ∀ x', abs (x' - x) < δ → abs (f x' a b - f x a b) < ε)) →
  a + b = 0 :=
by
  sorry

end continuous_piecewise_l676_676373


namespace cos_theta_seven_l676_676687

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end cos_theta_seven_l676_676687


namespace solve_equation_l676_676436

theorem solve_equation : ∃ x : ℝ, (2 / x) = (1 / (x + 1)) ∧ x = -2 :=
by
  use -2
  split
  { -- Proving the equality part
    show (2 / -2) = (1 / (-2 + 1)),
    simp,
    norm_num
  }
  { -- Proving the equation part
    refl
  }

end solve_equation_l676_676436


namespace John_l676_676260

/-- Assume Grant scored 10 points higher on his math test than John.
John received a certain ratio of points as Hunter who scored 45 points on his math test.
Grant's test score was 100. -/
theorem John's_points_to_Hunter's_points_ratio 
  (Grant John Hunter : ℕ) 
  (h1 : Grant = John + 10)
  (h2 : Hunter = 45)
  (h_grant_score : Grant = 100) : 
  (John : ℚ) / (Hunter : ℚ) = 2 / 1 :=
sorry

end John_l676_676260


namespace consumption_increase_ratio_l676_676304

noncomputable def demand_function (p : ℝ) : ℝ := 50 - p
noncomputable def marginal_cost : ℝ := 5
noncomputable def demand_function_new (p : ℝ) : ℝ := 2.5 * (50 - p)

theorem consumption_increase_ratio :
  let Q_initial := demand_function marginal_cost,
      Q_new := (125 - 2.5 * marginal_cost) / 0.8
  in Q_initial = 45 ∧ Q_new = 56.25 → Q_new / Q_initial = 1.25 :=
by
  sorry

end consumption_increase_ratio_l676_676304


namespace sum_of_roots_eq_2021_l676_676614

noncomputable def P (x : ℝ) : ℝ :=
  (x - 1)^2023 + 2 * (x - 2)^2022 + 3 * (x - 3)^2021 + ⋯ + 2022 * (x - 2022)^2 + 2023 * (x - 2023)

theorem sum_of_roots_eq_2021 : 
  (let roots := multiset.map (λ r, r) (polynomial.root_set (polynomial.map polynomial.C_deriv (P x)) ℂ) in
    roots.sum) = 2021 :=
sorry

end sum_of_roots_eq_2021_l676_676614


namespace locus_of_Q_is_ellipse_circle_with_diameter_AB_passes_fixed_point_l676_676983

-- Definitions
def circleE (x y: ℝ) : Prop := (x + 1)^2 + y^2 = 8
def fixedPointF : ℝ × ℝ := (1, 0)
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 2 / 3
def is_on_circle {P : ℝ × ℝ} : Prop := ∃ x y, circleE x y ∧ P = (x, y)
def bisector_intersects (F P : ℝ × ℝ) : ℝ × ℝ := sorry
def perpendicular_bisector (F P Q : ℝ × ℝ) : Prop := sorry
def tangent_to_circleO (A B : ℝ × ℝ) : Prop := sorry

-- Questions to prove
theorem locus_of_Q_is_ellipse :
  ∀ (P Q : ℝ × ℝ), is_on_circle P → 
                   Q = bisector_intersects fixedPointF P → 
                   perpendicular_bisector fixedPointF P Q → 
                   ∃ x y, (Q = (x, y) ∧ (x^2 / 2 + y^2 = 1)) :=
  sorry

theorem circle_with_diameter_AB_passes_fixed_point :
  ∀ (A B : ℝ × ℝ), tangent_to_circleO A B → 
                   ∃ O : ℝ × ℝ, (circle_with_diameter_AB A B) O 
                                 ∧ O = fixedPointF :=
  sorry

-- Helper definition
def circle_with_diameter_AB (A B : ℝ × ℝ) : ℝ × ℝ → Prop := sorry

end locus_of_Q_is_ellipse_circle_with_diameter_AB_passes_fixed_point_l676_676983


namespace find_m_l676_676971

theorem find_m (b y : ℕ) :
  (∃ m : ℕ, ∀ k : ℕ, 1 ≤ k ∧ k ≤ 3 →
   binomial m k * b^(m - k) * y^k = [6, 24, 60].nth (k - 1).get_or_else 0) →
  ∃ m : ℕ, m = 11 :=
by
  sorry

end find_m_l676_676971


namespace pigs_joined_l676_676044

theorem pigs_joined (p1 p2 : ℕ) (h1 : p1 = 64) (h2 : p2 = 86) : p2 - p1 = 22 := by
  rw [h1, h2]
  rfl

end pigs_joined_l676_676044


namespace solve_eq1_solve_eq2_solve_eq3_l676_676788

theorem solve_eq1 (x : ℝ) : 5 * x - 2.9 = 12 → x = 1.82 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

theorem solve_eq2 (x : ℝ) : 10.5 * x + 0.6 * x = 44 → x = 3 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

theorem solve_eq3 (x : ℝ) : 8 * x / 2 = 1.5 → x = 0.375 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

end solve_eq1_solve_eq2_solve_eq3_l676_676788


namespace sum_of_cubes_l676_676402

theorem sum_of_cubes {a b c : ℝ} (h1 : a + b + c = 5) (h2 : a * b + a * c + b * c = 7) (h3 : a * b * c = -18) : 
  a^3 + b^3 + c^3 = 29 :=
by
  -- The proof part is intentionally left out.
  sorry

end sum_of_cubes_l676_676402


namespace smallest_five_angles_sum_l676_676153

-- Definition of Q(x) and related conditions
def Q (x : ℂ) : ℂ := (x^20 - 1)^2 / (x - 1)^2 - x^19

-- Definition of angles and their sum
def angle_sum (angles: Fin 5 → ℝ) : ℝ :=
  angles 0 + angles 1 + angles 2 + angles 3 + angles 4

-- The proof goal
theorem smallest_five_angles_sum :
  ∃ (β : Fin 5 → ℝ), 
    (∀ i : Fin 5, 0 < β i ∧ β i < 1) ∧
    angle_sum β = 183/399 :=
by
  sorry

end smallest_five_angles_sum_l676_676153


namespace hyperbola_eccentricity_eq_l676_676098

theorem hyperbola_eccentricity_eq : 
  ∀ (a b c : ℝ), 
  (a > 0) → (b > 0) → (c > 0) →
  let F := (-c, 0) in
  let circle_eq := λ (x y : ℝ), x^2 + y^2 = a^2 in
  let hyperbola := λ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 in
  let parabola := λ (x y : ℝ), y^2 = 4 * c * x in
  let E := point_on_circle F circle_eq in
  let P := λ (E : ℝ × ℝ), tangent_intersects_parabola_at E parabola in
  let O := (0, 0) in
  (vector E O = vector F O + vector P O / 2) →
  eccentricity(hyperbola) = (sqrt 5 + 1) / 2 :=
by
  sorry

end hyperbola_eccentricity_eq_l676_676098


namespace weights_not_necessarily_equal_l676_676453

theorem weights_not_necessarily_equal (weights : Fin 10 → ℕ) 
  (h : ∀ i : Fin 10, ∃ (groups : Fin 3 → Multiset ℕ), 
          Multiset.card (groups 0) = Multiset.card (groups 1) ∧ 
          Multiset.card (groups 1) = Multiset.card (groups 2) ∧ 
          Multiset.sum (groups 0) = Multiset.sum (groups 1) ∧ 
          Multiset.sum (groups 1) = Multiset.sum (groups 2) ∧
          (Multiset.ofFinTable weights).erase (weights i) = 
          (groups 0) + (groups 1) + (groups 2)) :
          ¬ (∀ i j : Fin 10, weights i = weights j) :=
sorry

end weights_not_necessarily_equal_l676_676453


namespace distance_points_l676_676931

-- Definition of distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Points
def point1 : ℝ × ℝ := (3, 3)
def point2 : ℝ × ℝ := (-2, -2)

-- Main theorem
theorem distance_points : distance point1 point2 = 5 * Real.sqrt 2 :=
by
  sorry

end distance_points_l676_676931


namespace evaluate_expression_l676_676596

theorem evaluate_expression : 
    (81:ℝ)^(1/2) * (64:ℝ)^(-1/3) * (81:ℝ)^(1/4) = (27:ℝ) / (4:ℝ) := 
by
  sorry

end evaluate_expression_l676_676596


namespace correct_options_l676_676419

/-- Definitions and conditions given in the problem. -/
def sound_intensity_level (I I₀: ℝ) : ℝ := 10 * real.log10 (I / I₀)

def I₀ : ℝ := 10^(-12)  -- Threshold of hearing
def max_intensity : ℝ := 1  -- Maximum tolerable sound intensity
def max_intensity_level : ℝ := 120  -- Sound intensity level at max tolerable intensity

-- The range of sound intensity level for the singer is [70, 80] dB.
def singer_intensity_level_min : ℝ := 70
def singer_intensity_level_max : ℝ := 80

/-- Verifying correctness of options B and D -/
theorem correct_options : 
  ¬(sound_intensity_level I₀ I₀ = 0.1) ∧  -- Option A is incorrect
  (10^(-5) <= I ∧ I <= 10^(-4)) ∧  -- Option B is correct
  ¬(∀ I, sound_intensity_level (2 * I) I₀ = 2 * sound_intensity_level I I₀) ∧  -- Option C is incorrect
  (∀ I, sound_intensity_level (10 * I) I₀ = sound_intensity_level I I₀ + 10)  -- Option D is correct
  :=
by {
  sorry
}

end correct_options_l676_676419


namespace f_2007_eq_neg_cos_l676_676620

noncomputable def f : ℕ → (ℝ → ℝ)
| 0     := λ x, Real.sin x
| (n+1) := λ x, deriv (f n) x

/-- Lean 4 statement for proving the function f_2007(x) -/
theorem f_2007_eq_neg_cos : ∀ x : ℝ, f 2007 x = - Real.cos x :=
sorry

end f_2007_eq_neg_cos_l676_676620


namespace Kyro_Fylol_Glyk_l676_676711

variables (Fylol Glyk Kyro Mylo : Type)

-- Conditions
variable (h1 : ∀ x : Fylol, ∃ y : Glyk, y = x)
variable (h2 : ∀ x : Kyro, ∃ y : Glyk, y = x)
variable (h3 : ∀ x : Mylo, ∃ y : Fylol, y = x)
variable (h4 : ∀ x : Kyro, ∃ y : Mylo, y = x)

-- The proof problem:
theorem Kyro_Fylol_Glyk (h1 : ∀ x : Fylol, ∃ y : Glyk, y = x)
                        (h2 : ∀ x : Kyro, ∃ y : Glyk, y = x)
                        (h3 : ∀ x : Mylo, ∃ y : Fylol, y = x)
                        (h4 : ∀ x : Kyro, ∃ y : Mylo, y = x) :
  (∀ x : Kyro, ∃ y : Fylol, y = x) ∧ (∀ x : Kyro, ∃ y : Glyk, y = x) :=
sorry

end Kyro_Fylol_Glyk_l676_676711


namespace empty_rooms_le_1000_l676_676310

/--
In a 50x50 grid where each cell can contain at most one tree, 
with the following rules: 
1. A pomegranate tree has at least one apple neighbor
2. A peach tree has at least one apple neighbor and one pomegranate neighbor
3. An empty room has at least one apple neighbor, one pomegranate neighbor, and one peach neighbor
Show that the number of empty rooms is not greater than 1000.
-/
theorem empty_rooms_le_1000 (apple pomegranate peach : ℕ) (empty : ℕ)
  (h1 : apple + pomegranate + peach + empty = 2500)
  (h2 : ∀ p, pomegranate ≥ p → apple ≥ 1)
  (h3 : ∀ p, peach ≥ p → apple ≥ 1 ∧ pomegranate ≥ 1)
  (h4 : ∀ e, empty ≥ e → apple ≥ 1 ∧ pomegranate ≥ 1 ∧ peach ≥ 1) :
  empty ≤ 1000 :=
sorry

end empty_rooms_le_1000_l676_676310


namespace series_sum_zero_l676_676911

theorem series_sum_zero : (Finset.sum (Finset.range 502) (λ k, (4*k + 1) - (4*k + 2) - (4*k + 3) + (4*k + 4))) = 0 := 
by 
  sorry

end series_sum_zero_l676_676911


namespace cos_seven_theta_l676_676685

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end cos_seven_theta_l676_676685


namespace girls_ran_miles_l676_676859

def boys_laps : ℕ := 34
def extra_laps : ℕ := 20
def lap_distance : ℚ := 1 / 6
def girls_laps : ℕ := boys_laps + extra_laps

theorem girls_ran_miles : girls_laps * lap_distance = 9 := 
by 
  sorry

end girls_ran_miles_l676_676859


namespace quadratic_factorization_l676_676798

theorem quadratic_factorization (a b : ℕ) (h1 : x^2 - 18 * x + 72 = (x - a) * (x - b))
  (h2 : a > b) : 2 * b - a = 0 :=
sorry

end quadratic_factorization_l676_676798


namespace total_people_l676_676816

theorem total_people 
  (people_in_front : Nat)
  (people_behind : Nat)
  (lines : Nat)
  (people_in_front = 2)
  (people_behind = 4)
  (lines = 8) : 
  8 * (2 + 1 + 4) = 56 := 
by 
  sorry

end total_people_l676_676816


namespace coins_mass_comparisons_l676_676790

theorem coins_mass_comparisons (n : ℕ) (h_n : n ≥ 1) 
  (condition : ∀ (comps : list (multiset ℕ × multiset ℕ)), 
    (∀ c in comps, (c.fst.card = c.snd.card) ∧ (c.fst.sum (λ i, x i) = c.snd.sum (λ i, x i)) ) 
    → ∃ x : fin n → ℕ, (∀ i j : fin n, i ≠ j → x i ≠ x j) ∧ 
                        (dimensions_of_null_space condition < 2)) : 
  n < length comps + 1 :=
by sorry

end coins_mass_comparisons_l676_676790


namespace distances_inequality_l676_676729

theorem distances_inequality 
  {A B C P : Type}
  (d_a d_b d_c R_a R_b R_c : ℝ)
  (triangle_ABC : triangle A B C)
  (point_P : inside P triangle_ABC)
  (distances_sides : distances_to_sides P triangle_ABC d_a d_b d_c)
  (distances_vertices : distances_to_vertices P triangle_ABC R_a R_b R_c)
  (angles_ABC : angles A B C):
  3 * (d_a^2 + d_b^2 + d_c^2) ≥ (R_a * sin angles_ABC.A)^2 + (R_b * sin angles_ABC.B)^2 + (R_c * sin angles_ABC.C)^2 :=
by
  sorry

end distances_inequality_l676_676729


namespace sum_of_b_l676_676016

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 2 * n + 1

-- Define the sequence b_n as the average of the first n terms of a_n
def b (n : ℕ) : ℕ := (Finset.range n).sum (λ k, a (k + 1)) / n + 1

-- Define the sum S_n as the sum of the first n terms of b_n
def S (n : ℕ) : ℕ := (Finset.range n).sum (λ k, b (k + 1))

-- The main statement to be proved
theorem sum_of_b (n : ℕ) : S n = n * (n + 5) / 2 :=
sorry

end sum_of_b_l676_676016


namespace integer_solution_count_l676_676671

theorem integer_solution_count : 
  {x : ℤ | (x - 1)^2 ≤ 9}.to_finset.card = 7 := 
by 
  sorry

end integer_solution_count_l676_676671


namespace polynomial_integer_roots_k_l676_676926

theorem polynomial_integer_roots_k :
  ∀ (k : ℤ), 
    (∀ (x : ℤ), (x ^ 3 - (k - 3) * x ^ 2 - 11 * x + (4 * k - 8)) = 0 → x ∈ ℤ) →
    k = 5 :=
by sorry

end polynomial_integer_roots_k_l676_676926


namespace assign_ages_and_surnames_l676_676416

theorem assign_ages_and_surnames 
  (Kostya Vasya Kolya: String) 
  (Semyonov Burov Nikolaev: String) 
  (age_Kostya age_Vasya age_Kolya: Nat) 
  (grandfather_relation: (Semyonov, String) = "Petrov") 
  (age_relation_1: age_Kostya = age_Kolya + 1) 
  (age_relation_2: age_Kolya = age_Nikolaev + 1) 
  (sum_ages_condition: 49 < age_Kostya + age_Vasya + age_Kolya ∧ age_Kostya + age_Vasya + age_Kolya < 53) 
  (maternal_relation: Kolya_mother ≠ "Korobov")  :
  (Semyonov = "Kostya" ∧ Kostya = "Semyonov" ∧ age_Kostya = 18) ∧ 
  (Nikolaev = "Vasya" ∧ Vasya = "Nikolaev" ∧ age_Vasya = 16) ∧ 
  (Burov = "Kolya" ∧ Kolya = "Burov" ∧ age_Kolya = 17) := 
begin
  sorry
end

end assign_ages_and_surnames_l676_676416


namespace quadruple_dimensions_increase_volume_l676_676882

theorem quadruple_dimensions_increase_volume 
  (V_original : ℝ) (quad_factor : ℝ)
  (initial_volume : V_original = 5)
  (quad_factor_val : quad_factor = 4) :
  V_original * (quad_factor ^ 3) = 320 := 
by 
  -- Introduce necessary variables and conditions
  let V_modified := V_original * (quad_factor ^ 3)
  
  -- Assert the calculations based on the given conditions
  have initial : V_original = 5 := initial_volume
  have quad : quad_factor = 4 := quad_factor_val
  
  -- Skip the detailed proof with sorry
  sorry


end quadruple_dimensions_increase_volume_l676_676882


namespace cos_seven_theta_l676_676694

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end cos_seven_theta_l676_676694


namespace bike_trip_distance_l676_676869

theorem bike_trip_distance
  (round_trips : ℕ)
  (total_time_minutes : ℝ)
  (speed_km_per_hour : ℝ)
  (h1 : round_trips = 5)
  (h2 : total_time_minutes = 30)
  (h3 : speed_km_per_hour = 1.75) :
  let total_time_hours := total_time_minutes / 60
      total_distance := speed_km_per_hour * total_time_hours
      total_round_trip_distance := total_distance * round_trips
      one_way_distance := total_round_trip_distance / 2 in
  one_way_distance = 2.1875 :=
by
  sorry

end bike_trip_distance_l676_676869


namespace percentage_less_than_l676_676081

theorem percentage_less_than (P T J : ℝ) 
  (h1 : T = 0.9375 * P) 
  (h2 : J = 0.8 * T) 
  : (P - J) / P * 100 = 25 := 
by
  sorry

end percentage_less_than_l676_676081


namespace beef_weight_after_processing_l676_676114

def original_weight : ℝ := 861.54
def weight_loss_percentage : ℝ := 0.35
def retained_percentage : ℝ := 1 - weight_loss_percentage
def weight_after_processing (w : ℝ) := retained_percentage * w

theorem beef_weight_after_processing :
  weight_after_processing original_weight = 560.001 :=
by
  sorry

end beef_weight_after_processing_l676_676114


namespace possible_quadrilateral_areas_l676_676593

-- Define the problem set up
structure Point where
  x : ℝ
  y : ℝ

structure Square where
  side_length : ℝ
  A : Point
  B : Point
  C : Point
  D : Point

-- Defines the division points on each side of the square
def division_points (A B C D : Point) : List Point :=
  [
    -- Points on AB
    { x := 1, y := 4 }, { x := 2, y := 4 }, { x := 3, y := 4 },
    -- Points on BC
    { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 4, y := 1 },
    -- Points on CD
    { x := 3, y := 0 }, { x := 2, y := 0 }, { x := 1, y := 0 },
    -- Points on DA
    { x := 0, y := 3 }, { x := 0, y := 2 }, { x := 0, y := 1 }
  ]

-- Possible areas calculation using the Shoelace Theorem
def quadrilateral_areas : List ℝ :=
  [6, 7, 7.5, 8, 8.5, 9, 10]

-- Math proof problem in Lean, we need to prove that the quadrilateral areas match the given values
theorem possible_quadrilateral_areas (ABCD : Square) (pts : List Point) :
    (division_points ABCD.A ABCD.B ABCD.C ABCD.D) = [
      { x := 1, y := 4 }, { x := 2, y := 4 }, { x := 3, y := 4 },
      { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 4, y := 1 },
      { x := 3, y := 0 }, { x := 2, y := 0 }, { x := 1, y := 0 },
      { x := 0, y := 3 }, { x := 0, y := 2 }, { x := 0, y := 1 }
    ] → 
    (∃ areas, areas ⊆ quadrilateral_areas) := by
  sorry

end possible_quadrilateral_areas_l676_676593


namespace run_lap_times_l676_676764

theorem run_lap_times (N E : ℕ) (t_N t_E : ℕ) 
  (h1 : t_N + 12 = t_E) 
  (h2 : t_N > 30) 
  (h3 : 7 * t_N = (7 - (7 - 1)) * (t_N + 12)) : 
  t_N = 36 ∧ t_E = 48 :=
begin
  sorry
end

end run_lap_times_l676_676764


namespace smallest_multiple_of_35_with_product_multiple_of_35_l676_676468

def is_multiple_of_35 (n : ℕ) : Prop :=
  n % 35 = 0

def product_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).foldr (λ d acc, d * acc) 1

theorem smallest_multiple_of_35_with_product_multiple_of_35 :
  ∃ n : ℕ, is_multiple_of_35 n ∧ is_multiple_of_35 (product_of_digits n) ∧
  ∀ m : ℕ, is_multiple_of_35 m ∧ is_multiple_of_35 (product_of_digits m) → n ≤ m :=
  ∃ n, is_multiple_of_35 n ∧ is_multiple_of_35 (product_of_digits n) ∧ n = 735

end smallest_multiple_of_35_with_product_multiple_of_35_l676_676468


namespace no_real_roots_of_quadratic_l676_676639

def quadratic (a b c : ℝ) : ℝ × ℝ × ℝ := (a^2, b^2 + a^2 - c^2, b^2)

def discriminant (A B C : ℝ) : ℝ := B^2 - 4 * A * C

theorem no_real_roots_of_quadratic (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : c > 0)
  (h3 : |a - b| < c)
  : (discriminant (a^2) (b^2 + a^2 - c^2) (b^2)) < 0 :=
sorry

end no_real_roots_of_quadratic_l676_676639


namespace probability_theta_in_interval_l676_676880

-- Define the problem statement in Lean 4
theorem probability_theta_in_interval :
  let dice_rolls := finset.product (finset.range 1 (6 + 1)) (finset.range 1 (6 + 1)),
      satisfying_pairs := finset.filter (λ (pair : ℕ × ℕ), pair.1 - pair.2 ≥ 0) dice_rolls
  in (finset.card satisfying_pairs : ℚ) / (finset.card dice_rolls : ℚ) = 7 / 12 :=
by sorry

end probability_theta_in_interval_l676_676880


namespace find_teachers_and_students_l676_676093

-- Mathematical statements corresponding to the problem conditions
def teachers_and_students_found (x y : ℕ) : Prop :=
  (y = 30 * x + 7) ∧ (31 * x = y + 1)

-- The theorem we need to prove
theorem find_teachers_and_students : ∃ x y, teachers_and_students_found x y ∧ x = 8 ∧ y = 247 :=
  by
    sorry

end find_teachers_and_students_l676_676093


namespace line_perpendicular_to_plane_l676_676388

-- Define the conditions of the problem
variables (a b l : Line) (π : Plane)
variable (ha : a ⊆ π)
variable (hb : b ⊆ π)
variable (ha_b_intersect : ∃ p, p ∈ a ∧ p ∈ b)
variable (angle_equal : ∃ α, angle_between l a = α ∧ angle_between l b = α)

-- Define the conclusion to prove
theorem line_perpendicular_to_plane (h : angle_between l a = angle_between l b) :
  perpendicular l π := sorry

end line_perpendicular_to_plane_l676_676388


namespace anya_more_erasers_l676_676557

theorem anya_more_erasers (anya_erasers andrea_erasers : ℕ)
  (h1 : anya_erasers = 4 * andrea_erasers)
  (h2 : andrea_erasers = 4) :
  anya_erasers - andrea_erasers = 12 := by
  sorry

end anya_more_erasers_l676_676557


namespace range_of_a_l676_676802

def f (x : ℝ) : ℝ :=
if x ≥ 0 then (1 / 2) * x - 1 else 1 / x

theorem range_of_a (a : ℝ) : f a ≤ a ↔ a ≥ -1 := 
sorry

end range_of_a_l676_676802


namespace tobias_mowed_four_lawns_l676_676049

-- Let’s define the conditions
def shoe_cost : ℕ := 95
def allowance_per_month : ℕ := 5
def savings_months : ℕ := 3
def lawn_mowing_charge : ℕ := 15
def shoveling_charge : ℕ := 7
def change_after_purchase : ℕ := 15
def num_driveways_shoveled : ℕ := 5

-- Total money Tobias had before buying the shoes
def total_money : ℕ := shoe_cost + change_after_purchase

-- Money saved from allowance
def money_from_allowance : ℕ := allowance_per_month * savings_months

-- Money earned from shoveling driveways
def money_from_shoveling : ℕ := shoveling_charge * num_driveways_shoveled

-- Money earned from mowing lawns
def money_from_mowing : ℕ := total_money - money_from_allowance - money_from_shoveling

-- Number of lawns mowed
def num_lawns_mowed : ℕ := money_from_mowing / lawn_mowing_charge

-- The theorem stating the number of lawns mowed is 4
theorem tobias_mowed_four_lawns : num_lawns_mowed = 4 :=
by
  sorry

end tobias_mowed_four_lawns_l676_676049


namespace log_comparisons_l676_676358

noncomputable def a := Real.log 3 / Real.log 2
noncomputable def b := Real.log 3 / (2 * Real.log 2)
noncomputable def c := 1 / 2

theorem log_comparisons : c < b ∧ b < a := 
by
  sorry

end log_comparisons_l676_676358


namespace smallest_and_largest_negative_number_using_three_ones_l676_676611

theorem smallest_and_largest_negative_number_using_three_ones:
  (∀ x ∈ {-111, -1 / 11}, x ≤ -1 / 11 → x = -1 / 11) ∧
  (∀ y ∈ {-111, -1 / 11}, y ≥ -111 → y = -111) :=
by
  sorry

end smallest_and_largest_negative_number_using_three_ones_l676_676611


namespace admin_in_sample_l676_676506

-- Define the total number of staff members
def total_staff : ℕ := 200

-- Define the number of administrative personnel
def admin_personnel : ℕ := 24

-- Define the sample size taken
def sample_size : ℕ := 50

-- Goal: Prove the number of administrative personnel in the sample
theorem admin_in_sample : 
  (admin_personnel : ℚ) / (total_staff : ℚ) * (sample_size : ℚ) = 6 := 
by
  sorry

end admin_in_sample_l676_676506


namespace weight_of_second_pentagon_l676_676538

theorem weight_of_second_pentagon (s1 s2 : ℕ) (m1 : ℕ) (h1 : s1 = 4) (h2 : m1 = 20) (h3 : s2 = 6) :
  let A (s : ℕ) := (5 / 4) * (s ^ 2) * cot (π / 5)
  m2 = m1 * (A s2 / A s1) →
  m2 = 45 := by
  let A (s : ℕ) := (5 / 4) * (s ^ 2) * cot (π / 5)
  sorry

end weight_of_second_pentagon_l676_676538


namespace find_divisor_l676_676771

variable (dividend quotient remainder divisor : ℕ)

theorem find_divisor (h1 : dividend = 52) (h2 : quotient = 16) (h3 : remainder = 4) (h4 : dividend = divisor * quotient + remainder) : 
  divisor = 3 := by
  sorry

end find_divisor_l676_676771


namespace possible_values_of_n_l676_676029

theorem possible_values_of_n :
  let a := 1500
  let max_r2 := 562499
  let total := max_r2
  let perfect_squares := (750 : Nat)
  total - perfect_squares = 561749 := by
    sorry

end possible_values_of_n_l676_676029


namespace f_correct_c_range_l676_676245

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 9 * x

def correct_f (a b : ℝ) : Prop :=
  f 3 = 0 ∧ (deriv f) 3 = 0

def range_of_c (c : ℝ) : Prop :=
  ∀ x1 x2 ∈ [-1, 1], abs ((-x1^3 + 3 * c * x1) - (-x2^3 + 3 * c * x2)) ≤ 1

theorem f_correct (a b : ℝ) : correct_f a b → f = λ x, x^3 - 6 * x^2 + 9 * x := by
  sorry

theorem c_range (c : ℝ) : range_of_c c ↔ (1 / 6 ≤ c ∧ c ≤ (4 ^ (1 / 3)) / 4) := by
  sorry

end f_correct_c_range_l676_676245


namespace find_y_value_l676_676675

variable (x y z k : ℝ)

-- Conditions
def inverse_relation_y (x y : ℝ) (k : ℝ) : Prop := 5 * y = k / (x^2)
def direct_relation_z (x z : ℝ) : Prop := 3 * z = x

-- Constant from conditions
def k_constant := 500

-- Problem statement
theorem find_y_value (h1 : inverse_relation_y 2 25 k_constant) (h2 : direct_relation_z 4 6) :
  y = 6.25 :=
by
  sorry

-- Auxiliary instance to fulfill the proof requirement
noncomputable def y_value : ℝ := 6.25

end find_y_value_l676_676675


namespace pigs_joined_l676_676043

theorem pigs_joined (p1 p2 : ℕ) (h1 : p1 = 64) (h2 : p2 = 86) : p2 - p1 = 22 := by
  rw [h1, h2]
  rfl

end pigs_joined_l676_676043


namespace find_a_and_tangent_line_l676_676249

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2 * x^2 + a * x - 1

theorem find_a_and_tangent_line (a x y : ℝ) (h1 : f a 1 = x^3 - 2 * x^2 + 2 * x - 1)
  (h2 : (deriv (f a) 1 = 1)) :
  a = 2 ∧ (exists y, (y + 6 = 9 * (x + 1)))) :=
by
sry

end find_a_and_tangent_line_l676_676249


namespace eq_solutions_count_l676_676026

theorem eq_solutions_count : 
  ∃! (n : ℕ), n = 126 ∧ (∀ x y : ℕ, 2*x + 3*y = 768 → x > 0 ∧ y > 0 → ∃ t : ℤ, x = 384 + 3*t ∧ y = -2*t ∧ -127 ≤ t ∧ t <= -1) := sorry

end eq_solutions_count_l676_676026


namespace mn_over_bc_eq_sqrt_l676_676630

variables {A B C M N : Type}
variables (R r BC : ℝ)
variables [triangle ABC]

theorem mn_over_bc_eq_sqrt (MN BC R r : ℝ) :
  MN / BC = sqrt (1 - 2 * r / R) :=
by sorry

end mn_over_bc_eq_sqrt_l676_676630


namespace find_minimum_a_l676_676635

noncomputable def f (a : ℝ) := λ x : ℝ, 2 * a * x^2 - 1 / (a * x)

noncomputable def derivative_f_at_one (a : ℝ) := 4 * a + 1 / a

theorem find_minimum_a (a : ℝ) (h : 0 < a) (k : ℝ) (h_k : k = derivative_f_at_one a) :
  a = 1 / 2 → k = 4 :=
sorry

end find_minimum_a_l676_676635


namespace domain_of_function_l676_676586

open Real

theorem domain_of_function : 
  ∀ x, 
    (x + 1 ≠ 0) ∧ 
    (-x^2 - 3 * x + 4 > 0) ↔ 
    (-4 < x ∧ x < -1) ∨ ( -1 < x ∧ x < 1) := 
by 
  sorry

end domain_of_function_l676_676586


namespace ordered_sum_binomial_coefficient_l676_676497

theorem ordered_sum_binomial_coefficient (n p : ℕ) (h : n ≥ p) :
  {s : fin n → ℕ // (finset.univ.sum s = n) ∧ (∀ i, 0 < s i)} →
  (finset.univ.card : ℕ) = (nat.choose (n-1) (p-1)) :=
begin
    -- Key definitions and assumptions go here.
    intros,
    sorry -- proof steps go here
end

end ordered_sum_binomial_coefficient_l676_676497


namespace geometric_sequence_div_sum_l676_676363

noncomputable def a (n : ℕ) : ℝ := sorry

noncomputable def S (n : ℕ) : ℝ := sorry

theorem geometric_sequence_div_sum 
  (h₁ : S 3 = (1 - (2 : ℝ) ^ 3) / (1 - (2 : ℝ) ^ 2) * a 1)
  (h₂ : S 2 = (1 - (2 : ℝ) ^ 2) / (1 - 2) * a 1)
  (h₃ : 8 * a 2 = a 5) : 
  S 3 / S 2 = 7 / 3 := 
by
  sorry

end geometric_sequence_div_sum_l676_676363


namespace find_pairs_l676_676927

theorem find_pairs (m n : ℕ) (a b : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : a > 1) (h4 : b > 1) (h5 : Nat.gcd a b = 1) :
  (a^m + b^m) % (a^n + b^n) = 0 ↔ ∃ (k : ℕ), k > 0 ∧ m = (2 * k - 1) * n :=
begin
  sorry
end

end find_pairs_l676_676927


namespace largest_possible_number_of_red_socks_l676_676513

noncomputable def max_red_socks (t : ℕ) (r : ℕ) : Prop :=
  t ≤ 1991 ∧
  ((r * (r - 1) + (t - r) * (t - r - 1)) / (t * (t - 1)) = 1 / 2) ∧
  ∀ r', r' ≤ 990 → (t ≤ 1991 ∧
    ((r' * (r' - 1) + (t - r') * (t - r' - 1)) / (t * (t - 1)) = 1 / 2) → r ≤ r')

theorem largest_possible_number_of_red_socks :
  ∃ t r, max_red_socks t r ∧ r = 990 :=
by
  sorry

end largest_possible_number_of_red_socks_l676_676513


namespace Jakes_weight_is_198_l676_676077

variable (Jake Kendra : ℕ)

-- Conditions
variable (h1 : Jake - 8 = 2 * Kendra)
variable (h2 : Jake + Kendra = 293)

theorem Jakes_weight_is_198 : Jake = 198 :=
by
  sorry

end Jakes_weight_is_198_l676_676077


namespace fuel_tank_capacity_l676_676870

-- Define the given conditions
def initial_mileage_per_gallon : ℝ := 32
def modified_efficiency_factor : ℝ := 0.8
def additional_miles : ℝ := 96

-- We need to prove the car's fuel tank holds 15 gallons
theorem fuel_tank_capacity (C : ℝ) (h : C = 15) :
  modified_efficiency_factor * initial_mileage_per_gallon * C = initial_mileage_per_gallon * C + additional_miles :=
by
  -- substitution of C = 15 yields
  have h1: 0.8 * 32 * 15 = 32 * 15 + 96 := by linarith
  sorry

end fuel_tank_capacity_l676_676870


namespace c_time_to_complete_work_l676_676485

variable (A B C : ℝ)

def work_rate_a_b : Prop := A + B = 1 / 10
def work_rate_b_c : Prop := B + C = 1 / 5
def work_rate_c_a : Prop := C + A = 1 / 15

theorem c_time_to_complete_work (h1 : work_rate_a_b A B C)
                                (h2 : work_rate_b_c A B C)
                                (h3 : work_rate_c_a A B C) :
  1 / C = 12 :=
by
  sorry

end c_time_to_complete_work_l676_676485


namespace width_of_river_l676_676461

/-- Two men start from opposite banks of the river. They meet 340 meters away from one of the banks
on their forward journey. They meet 170 meters away from the other bank of the river on their backward journey.
Prove that the width of the river is 340 meters. -/
theorem width_of_river : ∃ W : ℕ, 
  (∀ W, ((340 + (W - 170)) + (170 + (W - 340)) = W) → W = 340) :=
begin
  use 340,
  intro W,
  intros h,
  sorry
end

end width_of_river_l676_676461


namespace find_a_tangent_line_at_minus_one_l676_676247

-- Define the function f with variable a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x - 1

-- Define the derivative of f with variable a
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Given conditions
def condition_1 : Prop := f' 1 = 1
def condition_2 : Prop := f' 2 (1 : ℝ) = 1

-- Prove that a = 2 given f'(1) = 1
theorem find_a : f' 2 (1 : ℝ) = 1 → 2 = 2 := by
  sorry

-- Given a = 2, find the tangent line equation at x = -1
def tangent_line_equation (x y : ℝ) : Prop := 9*x - y + 3 = 0

-- Define the coordinates of the point on the curve at x = -1
def point_on_curve : Prop := f 2 (-1) = -6

-- Prove the tangent line equation at x = -1 given a = 2
theorem tangent_line_at_minus_one (h : true) : tangent_line_equation 9 (f' 2 (-1)) := by
  sorry

end find_a_tangent_line_at_minus_one_l676_676247


namespace longest_line_segment_squared_l676_676518

-- Define the problem conditions
def diameter := 18 : ℝ
def radius := diameter / 2
def central_angle := 90 : ℝ

-- Define the theorem to be proved
theorem longest_line_segment_squared (d : ℝ) (r : ℝ) (theta : ℝ) :=
  d = 18 → r = d / 2 → theta = 90 →
  let l_squared := 2 * (r^2) in
  l_squared = 162 :=
by
  intros h1 h2 h3
  sorry

end longest_line_segment_squared_l676_676518


namespace min_value_xyz_l676_676362

theorem min_value_xyz (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_prod : x * y * z = 8) : 
  x + 3 * y + 6 * z ≥ 18 :=
sorry

end min_value_xyz_l676_676362


namespace total_money_l676_676075

-- Define the problem with conditions and question transformed into proof statement
theorem total_money (A B : ℕ) (h1 : 2 * A / 3 = B / 2) (h2 : B = 484) : A + B = 847 :=
by
  sorry -- Proof to be filled in

end total_money_l676_676075


namespace students_taking_neither_l676_676766

-- Defining given conditions as Lean definitions
def total_students : ℕ := 70
def students_math : ℕ := 42
def students_physics : ℕ := 35
def students_chemistry : ℕ := 25
def students_math_physics : ℕ := 18
def students_math_chemistry : ℕ := 10
def students_physics_chemistry : ℕ := 8
def students_all_three : ℕ := 5

-- Define the problem to prove
theorem students_taking_neither : total_students
  - (students_math - students_math_physics - students_math_chemistry + students_all_three
    + students_physics - students_math_physics - students_physics_chemistry + students_all_three
    + students_chemistry - students_math_chemistry - students_physics_chemistry + students_all_three
    + students_math_physics - students_all_three
    + students_math_chemistry - students_all_three
    + students_physics_chemistry - students_all_three
    + students_all_three) = 0 := by
  sorry

end students_taking_neither_l676_676766


namespace each_child_receives_correct_amount_l676_676877

noncomputable def husband_contribution_per_month := 3 * 450
noncomputable def wife_contribution_per_month := 6 * 315
noncomputable def months := 8
noncomputable def total_husband_contribution := husband_contribution_per_month * months
noncomputable def total_wife_contribution := wife_contribution_per_month * months
noncomputable def total_savings := total_husband_contribution + total_wife_contribution
noncomputable def proportion_saved := 0.75
noncomputable def amount_to_divide := proportion_saved * total_savings
noncomputable def number_of_children := 6
noncomputable def amount_per_child := amount_to_divide / number_of_children

theorem each_child_receives_correct_amount :
  amount_per_child = 3240 := 
by
  sorry

end each_child_receives_correct_amount_l676_676877


namespace probability_of_type_I_error_l676_676066

theorem probability_of_type_I_error 
  (K_squared : ℝ)
  (alpha : ℝ)
  (critical_val : ℝ)
  (h1 : K_squared = 4.05)
  (h2 : alpha = 0.05)
  (h3 : critical_val = 3.841)
  (h4 : 4.05 > 3.841) :
  alpha = 0.05 := 
sorry

end probability_of_type_I_error_l676_676066


namespace min_max_product_l676_676372

theorem min_max_product :
  ∀ (x y z : ℝ), (3 * x ^ 2 + 9 * x * y + 6 * y ^ 2 + z ^ 2 = 1) →
  let p := x ^ 2 + 4 * x * y + 3 * y ^ 2 + z ^ 2 in
  (p = (2 - Real.sqrt 2) / 3 ∨ p = (2 + Real.sqrt 2) / 3) →
  (p = (2 - Real.sqrt 2) / 3) ∧ (p = (2 + Real.sqrt 2) / 3) → 
  ((2 - Real.sqrt 2) / 3) * ((2 + Real.sqrt 2) / 3) = 2 / 3 :=
by sorry

end min_max_product_l676_676372


namespace initial_number_of_kids_l676_676827

theorem initial_number_of_kids (joined kids_total initial : ℕ) (h1 : joined = 22) (h2 : kids_total = 36) (h3 : kids_total = initial + joined) : initial = 14 :=
by 
  -- Proof goes here
  sorry

end initial_number_of_kids_l676_676827


namespace Jesse_remaining_money_l676_676333

-- Define the conditions
def initial_money := 50
def novel_cost := 7
def lunch_cost := 2 * novel_cost
def total_spent := novel_cost + lunch_cost

-- Define the remaining money after spending
def remaining_money := initial_money - total_spent

-- Prove that the remaining money is $29
theorem Jesse_remaining_money : remaining_money = 29 := 
by
  sorry

end Jesse_remaining_money_l676_676333


namespace simplify_frac_l676_676787

theorem simplify_frac :
  (1 / (1 / (Real.sqrt 3 + 2) + 2 / (Real.sqrt 5 - 2))) = 
  (Real.sqrt 3 - 2 * Real.sqrt 5 - 2) :=
by
  sorry

end simplify_frac_l676_676787


namespace squid_wing_area_correct_l676_676149

open set real

noncomputable def larger_quarter_circle_area : ℝ := (π * 5^2) / 4

noncomputable def smaller_quarter_circle_area : ℝ := (π * 2^2) / 4

noncomputable def squid_wing_area : ℝ := larger_quarter_circle_area - smaller_quarter_circle_area

theorem squid_wing_area_correct :
  squid_wing_area = 21 * π / 4 :=
by sorry

end squid_wing_area_correct_l676_676149


namespace num_factors_36_l676_676665

theorem num_factors_36 : ∀ (n : ℕ), n = 36 → (∃ (a b : ℕ), 36 = 2^a * 3^b ∧ a = 2 ∧ b = 2 ∧ (a + 1) * (b + 1) = 9) :=
by
  sorry

end num_factors_36_l676_676665


namespace defective_units_shipped_l676_676078

theorem defective_units_shipped (total_units defective_units shipped_defective_units : ℝ) 
  (h1 : defective_units = 0.06 * total_units)
  (h2 : shipped_defective_units = 0.04 * defective_units) :
  (shipped_defective_units / total_units) * 100 = 0.24 := by
sory

end defective_units_shipped_l676_676078


namespace all_numbers_positive_l676_676982

open Real

theorem all_numbers_positive (n : ℕ) (a : Fin (2 * n + 1) → ℝ) 
  (h : ∀ (s : Finset (Fin (2 * n + 1))) (hn : s.card = n), ∑ i in s, a i < ∑ i in sᶜ, a i) :
  ∀ i, 0 < a i :=
sorry

end all_numbers_positive_l676_676982


namespace omega4_radius_correct_l676_676053

noncomputable def radius_of_omega4 : ℝ :=
  let r1 := 10
  let r2 := 13
  let r3 := 2 * Real.sqrt 2
  -- Define the conditions for orthogonality and tangency
  -- Assuming these conditions are encoded in a more formal way
  if ω1 ω2 ω3 ω4 : Circle -- placeholder, actual definitions required
  if external_tangent (ω1 ω2) P ∧ external_tangent (ω2 ω3) P ∧ orthogonal (ω1 ω3) ∧ orthogonal (ω2 ω3) ∧ orthogonal (ω3 ω4) ∧ external_tangent (ω4 ω1) ∧ external_tangent (ω4 ω2)
  then let r4 := 92 / 61 in r4
  else 0 -- In case the conditions are not met, return a default value

theorem omega4_radius_correct : 
  ∃ (r4 : ℝ), r4 = radius_of_omega4 :=
by
  existsi (92 / 61)
  sorry

end omega4_radius_correct_l676_676053


namespace solve_fractional_eq_l676_676444

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end solve_fractional_eq_l676_676444


namespace sum_series_sin_cos_l676_676962

theorem sum_series_sin_cos (n : ℕ) (α : ℝ) :
  (∑ i in Finset.range n, 2^i * (sin (α / 2^i) * (sin (α / 2^(i+1)))^2))
  = 1 - cos (α / 2^(n-1)) :=
sorry

end sum_series_sin_cos_l676_676962


namespace ratio_hours_per_day_l676_676132

theorem ratio_hours_per_day 
  (h₁ : ∀ h : ℕ, h * 30 = 1200 + (h - 40) * 45 → 40 ≤ h ∧ 6 * 3 ≤ 40)
  (h₂ : 6 * 3 + (x - 6 * 3) / 2 = 24)
  (h₃ : x = 1290) :
  (24 / 2) / 6 = 2 := 
by
  sorry

end ratio_hours_per_day_l676_676132


namespace min_value_f_min_achieved_l676_676960

noncomputable def f (x : ℝ) : ℝ := (1 / (x - 3)) + x

theorem min_value_f : ∀ x : ℝ, x > 3 → f x ≥ 5 :=
by
  intro x hx
  sorry

theorem min_achieved : f 4 = 5 :=
by
  sorry

end min_value_f_min_achieved_l676_676960


namespace cauchy_mean_value_theorem_l676_676014

variable {a b : ℝ} (h : a < b)
variable {f g : ℝ → ℝ}
variable (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
variable (hgz : ∀ x ∈ Ioo a b, deriv g x ≠ 0)

theorem cauchy_mean_value_theorem :
  ∃ x₀ ∈ Ioo a b, (f b - f a) / (g b - g a) = (deriv f x₀) / (deriv g x₀) := by
  sorry

end cauchy_mean_value_theorem_l676_676014


namespace shop_second_selling_price_is_270_l676_676875

noncomputable def original_cost (C : ℝ) : Prop :=
  let store_price := 1.20 * C
  let buy_back_price := 0.50 * store_price
  let diff := C - buy_back_price
  diff = 100

noncomputable def second_selling_price (C : ℝ) : ℝ :=
  let buy_back_price := 0.50 * (1.20 * C)
  buy_back_price + (0.80 * buy_back_price)

theorem shop_second_selling_price_is_270 :
  ∃ C, original_cost C ∧ second_selling_price C = 270 :=
by
  use 250
  have h1 : original_cost 250, from sorry
  have h2 : second_selling_price 250 = 270, from sorry
  exact ⟨h1, h2⟩

end shop_second_selling_price_is_270_l676_676875


namespace triangle_problems_l676_676324

-- lean 4 statement
theorem triangle_problems 
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : 4 * a^2 * (cos B)^2 + 4 * b^2 * (sin A)^2 = 3 * b^2 - 3 * c^2)
    (h2 : b = 1) :
  a + 6 * c * cos B = 0 ∧ 
  ∃ (S : ℝ), S = (sqrt (36 * c^2 - a^2)) * a * c / 12 ∧
      ∀ (S' : ℝ), S' <= 3 / 14 :=
begin
  sorry
end

end triangle_problems_l676_676324


namespace reciprocal_of_sum_fractions_l676_676466

theorem reciprocal_of_sum_fractions : 
  (3 / 4 + 1 / 6)⁻¹ = (12 / 11) := 
by 
  sorry

end reciprocal_of_sum_fractions_l676_676466


namespace train_crossing_time_approximate_l676_676104

def train_speed_kmph := 72
def platform_length_m := 250
def train_length_m := 470.06

def convert_kmph_to_mps (speed_kmph : ℕ) : ℕ := (speed_kmph * 1000) / 3600

noncomputable def total_distance_m := platform_length_m + train_length_m
noncomputable def train_speed_mps := convert_kmph_to_mps train_speed_kmph
noncomputable def time_seconds := total_distance_m / train_speed_mps

theorem train_crossing_time_approximate :
  (train_speed_kmph = 72) →
  (platform_length_m = 250) →
  (train_length_m = 470.06) →
  abs (time_seconds - 36) < 1 :=
by
  intro h1 h2 h3
  sorry

end train_crossing_time_approximate_l676_676104


namespace simplify_trig_expression_l676_676396

theorem simplify_trig_expression (α : ℝ) :
  (sin (2 * π - α) * cos (π + α) * cos (π / 2 + α) * cos (11 * π / 2 - α)) /
  (cos (π - α) * sin (3 * π - α) * sin (-π - α) * sin (9 * π / 2 + α) * tan (π + α)) = -1 :=
by
  sorry

end simplify_trig_expression_l676_676396


namespace digital_earth_advanced_productive_forces_l676_676126

noncomputable theory

variables (DigitalEarth: Type) 
          (globalize_humanity: DigitalEarth → Prop)
          (emphasize_technology: DigitalEarth → Prop)
          (economic_strength_independent: DigitalEarth → Prop)
          (absolute_serves_politics: DigitalEarth → Prop)
          (concentrated_expression_of_advanced_productive_forces: DigitalEarth → Prop)

theorem digital_earth_advanced_productive_forces (d : DigitalEarth) :
  ¬ globalize_humanity d →
  emphasize_technology d →
  ¬ economic_strength_independent d →
  ¬ absolute_serves_politics d →
  concentrated_expression_of_advanced_productive_forces d :=
by
  sorry

end digital_earth_advanced_productive_forces_l676_676126


namespace distance_between_points_l676_676940

theorem distance_between_points :
  ∀ (P Q : ℝ × ℝ), P = (3, 3) ∧ Q = (-2, -2) → dist P Q = 5 * real.sqrt 2 :=
begin
  sorry
end

end distance_between_points_l676_676940


namespace average_score_after_11_matches_l676_676110

theorem average_score_after_11_matches 
  (A : ℝ) 
  (h1 : let total_runs_10 := 10 * A)
  (h2 : let total_runs_11 := total_runs_10 + 98)
  (h3 : let new_average := A + 6)
  (h4 : total_runs_11 = 11 * new_average) : 
  A = 32 := by
    sorry

end average_score_after_11_matches_l676_676110


namespace sum_of_consecutive_odd_integers_l676_676470

theorem sum_of_consecutive_odd_integers (n : ℕ) (h : ∑ k in finset.range n, 2 * k + 1 = 169) : n = 13 :=
by
  -- Proof to be filled in
  sorry

end sum_of_consecutive_odd_integers_l676_676470


namespace chips_sales_l676_676510

theorem chips_sales (total_chips : ℕ) (first_week : ℕ) (second_week : ℕ) (third_week : ℕ) (fourth_week : ℕ)
  (h1 : total_chips = 100)
  (h2 : first_week = 15)
  (h3 : second_week = 3 * first_week)
  (h4 : third_week = fourth_week)
  (h5 : total_chips = first_week + second_week + third_week + fourth_week) : third_week = 20 :=
by
  sorry

end chips_sales_l676_676510


namespace total_time_spent_with_dog_l676_676761

def washing_time : ℕ := 20
def blow_drying_time : ℕ := washing_time / 2
def distance : ℕ := 3
def speed : ℕ := 6
def walking_time : ℝ := (distance / speed) * 60

theorem total_time_spent_with_dog : washing_time + blow_drying_time + walking_time = 60 := by
  sorry

end total_time_spent_with_dog_l676_676761


namespace completing_square_to_simplify_eq_l676_676834

theorem completing_square_to_simplify_eq : 
  ∃ (c : ℝ), (∀ x : ℝ, x^2 - 6 * x + 4 = 0 ↔ (x - 3)^2 = c) :=
by
  use 5
  intro x
  constructor
  { intro h
    -- proof conversion process (skipped)
    sorry }
  { intro h
    -- reverse proof process (skipped)
    sorry }

end completing_square_to_simplify_eq_l676_676834


namespace correct_propositions_l676_676551

def proposition_1 : Prop :=
  ∀ (E : Type) (P : E → Prop),
    (∀ x : E, P x) → P x

def proposition_2 : Prop :=
  ∀ (P : E → Prop), ∃ (p : ℝ), p = 1.1 → false

def proposition_3 : Prop :=
  ∀ (A B : Prop), (¬A ∧ ¬B) → (A ∧ B) → false

def proposition_4 : Prop :=
  ∀ (P : E → Prop) (freq : ℝ → ℕ), 
    stable_value P freq → approximation P freq

def proposition_5 : Prop :=
  ∀ (plant_seed : Prop) (conditions : Prop) (germinates : E → Prop), 
    classical_model plant_seed conditions → false

theorem correct_propositions : 
  proposition_1 ∧ proposition_4 ∧ ¬ proposition_2 ∧ ¬ proposition_3 ∧ ¬ proposition_5 :=
by 
  sorry

end correct_propositions_l676_676551


namespace solve_equation_l676_676433

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) : (2 / x = 1 / (x + 1)) → x = -2 :=
begin
  sorry
end

end solve_equation_l676_676433


namespace problem_l676_676868

theorem problem (a : ℝ) : (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → (a > 3 ∨ a < -1) :=
by
  sorry

end problem_l676_676868


namespace exists_n_not_represented_l676_676741

theorem exists_n_not_represented (a b c d : ℤ) (a_gt_14 : a > 14)
  (h1 : 0 ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ a) :
  ∃ (n : ℕ), ¬ ∃ (x y z : ℤ), n = x * (a * x + b) + y * (a * y + c) + z * (a * z + d) :=
sorry

end exists_n_not_represented_l676_676741


namespace jesse_money_left_after_mall_l676_676336

theorem jesse_money_left_after_mall :
  ∀ (initial_amount novel_cost lunch_cost total_spent remaining_amount : ℕ),
    initial_amount = 50 →
    novel_cost = 7 →
    lunch_cost = 2 * novel_cost →
    total_spent = novel_cost + lunch_cost →
    remaining_amount = initial_amount - total_spent →
    remaining_amount = 29 :=
by
  intros initial_amount novel_cost lunch_cost total_spent remaining_amount
  sorry

end jesse_money_left_after_mall_l676_676336


namespace simplify_expression_l676_676786

theorem simplify_expression :
  (2^8 + 4^5) * (2^3 - (-2)^3)^2 = 327680 := by
  sorry

end simplify_expression_l676_676786


namespace positive_odd_square_free_count_l676_676899

def is_square_free (n : ℕ) : Prop :=
∀ m : ℕ, m > 1 → m * m ≤ n → ¬ m * m ∣ n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def positive_odd_integers := {n : ℕ // n > 1 ∧ n < 100 ∧ is_odd n}

def count_square_free : ℕ :=
finset.card (finset.filter 
  (λ n : positive_odd_integers, is_square_free n.1) 
  (finset.univ : finset positive_odd_integers))

theorem positive_odd_square_free_count :
  count_square_free = 40 := sorry

end positive_odd_square_free_count_l676_676899


namespace number_of_possible_integers_l676_676276

theorem number_of_possible_integers (x: ℤ) (h: ⌈real.sqrt ↑x⌉ = 20) : 39 :=
  sorry

end number_of_possible_integers_l676_676276


namespace volume_of_sphere_in_cone_l676_676891

/-- Define the conditions -/
structure ConeSphere :=
  (base_diameter : ℝ)
  (vertex_angle : ℝ)
  (radius : ℝ)

def cone_sphere_data : ConeSphere :=
  { base_diameter := 12 * Real.sqrt 2,
    vertex_angle := 45,
    radius := 6 }

theorem volume_of_sphere_in_cone (cs : ConeSphere) : 
  cs.vertex_angle = 45 → 
  cs.base_diameter = 12 * Real.sqrt 2 → 
  cs.radius = 6 → 
  (4/3 * Real.pi * cs.radius^3) = 288 * Real.pi :=
by
  intros h_vertex_angle h_base_diameter h_radius
  sorry

end volume_of_sphere_in_cone_l676_676891


namespace numberOfSolutions_l676_676166

theorem numberOfSolutions (n : ℕ) :
  let f := λ x : ℝ, x / 50
  let g := λ x : ℝ, Real.sin x
  (∀ x : ℝ, f x = g x → -50 ≤ x ∧ x ≤ 50) →
  n = 32 :=
sorry

end numberOfSolutions_l676_676166


namespace find_ordered_pair_l676_676608

theorem find_ordered_pair (x y : ℚ) 
  (h1 : 3 * x - 18 * y = 2) 
  (h2 : 4 * y - x = 6) :
  x = -58 / 3 ∧ y = -10 / 3 :=
sorry

end find_ordered_pair_l676_676608


namespace perp_AD_BM_cos_dihedral_angle_half_DB_l676_676901

open Real EuclideanGeometry

structure Rect_midpoint (A B C D M : ℝ × ℝ) :=
(AB_eq : dist A B = 2)
(AD_eq : dist A D = 1)
(M_eq : M = (D.1 + C.1) / 2, (D.2 + C.2) / 2)
(rect_def : (dist A B)^2 + (dist A D)^2 = (dist B C)^2)

theorem perp_AD_BM (A B C D M : ℝ × ℝ) 
(h : Rect_midpoint A B C D M) 
(plane_perp : ∀ P Q R S T U, P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧ U ≠ T ∧ T ≠ S ∧ S ≠ U
                    → dist P Q ≠ 0 ∧ dist Q R ≠ 0 ∧ dist R P ≠ 0 
                    → dist T U ≠ 0 ∧ dist U S ≠ 0 ∧ dist S T ≠ 0 ) :
  ⟂ (vector span (A D)) (vector span (B M)) :=
begin
  sorry
end

theorem cos_dihedral_angle_half_DB (A B C D M E : ℝ × ℝ) 
(h : Rect_midpoint A B C D M) 
(E_half : dist D E = dist E B)
(plane_perp : ∀ P Q R S T U, P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧ U ≠ T ∧ T ≠ S ∧ S ≠ U
                    → dist P Q ≠ 0 ∧ dist Q R ≠ 0 ∧ dist R P ≠ 0 
                    → dist T U ≠ 0 ∧ dist U S ≠ 0 ∧ dist S T ≠ 0)
: 
  cos_dihedral_angle (E A M D) = (sqrt 5) / 5 :=
begin
  sorry
end

end perp_AD_BM_cos_dihedral_angle_half_DB_l676_676901


namespace Jason_spent_on_jacket_l676_676330

/-
Given:
- Amount_spent_on_shorts: ℝ := 14.28
- Total_spent_on_clothing: ℝ := 19.02

Prove:
- Amount_spent_on_jacket = 4.74
-/
def Amount_spent_on_shorts : ℝ := 14.28
def Total_spent_on_clothing : ℝ := 19.02

-- We need to prove:
def Amount_spent_on_jacket : ℝ := Total_spent_on_clothing - Amount_spent_on_shorts 

theorem Jason_spent_on_jacket : Amount_spent_on_jacket = 4.74 := by
  sorry

end Jason_spent_on_jacket_l676_676330


namespace result_of_3_at_2_l676_676297

variable (a b : ℝ)

def op (a b : ℝ) : ℝ := (a ^ b) / 2

theorem result_of_3_at_2 : op 3 2 = 4.5 := 
by
  sorry

end result_of_3_at_2_l676_676297


namespace solve_equation_l676_676439

theorem solve_equation : ∃ x : ℝ, (2 / x) = (1 / (x + 1)) ∧ x = -2 :=
by
  use -2
  split
  { -- Proving the equality part
    show (2 / -2) = (1 / (-2 + 1)),
    simp,
    norm_num
  }
  { -- Proving the equation part
    refl
  }

end solve_equation_l676_676439


namespace consecutive_integers_sum_l676_676032

theorem consecutive_integers_sum (x : ℤ) (h : x * (x + 1) = 440) : x + (x + 1) = 43 :=
by sorry

end consecutive_integers_sum_l676_676032


namespace first_tier_tax_rate_l676_676157

theorem first_tier_tax_rate (price : ℕ) (total_tax : ℕ) (tier1_limit : ℕ) (tier2_rate : ℝ) (tier1_tax_rate : ℝ) :
  price = 18000 →
  total_tax = 1950 →
  tier1_limit = 11000 →
  tier2_rate = 0.09 →
  ((price - tier1_limit) * tier2_rate + tier1_tax_rate * tier1_limit = total_tax) →
  tier1_tax_rate = 0.12 :=
by
  intros hprice htotal htier1 hrate htax_eq
  sorry

end first_tier_tax_rate_l676_676157


namespace distance_between_centers_eq_l676_676122

theorem distance_between_centers_eq (r1 r2 : ℝ) : ∃ d : ℝ, (d = r1 * Real.sqrt 2) := by
  sorry

end distance_between_centers_eq_l676_676122


namespace trapezoidal_prism_volume_l676_676616

-- Definitions from the conditions.
def a : ℝ := 20
def b : ℝ := 15
def h : ℝ := 14
def height_prism : ℝ := 10

-- Proof requirement: Volume of the trapezoidal prism
theorem trapezoidal_prism_volume : 
  let A := (1 / 2) * (a + b) * h 
  in A * height_prism = 2450 := by
    let A := (1 / 2) * (a + b) * h
    let V := A * height_prism
    have hA : A = 17.5 * 14 := by sorry
    have hV : V = 2450 := by sorry
    exact hV

end trapezoidal_prism_volume_l676_676616


namespace cone_vertex_angle_l676_676587

noncomputable def vertex_angle_of_cone (r h : ℝ) : ℝ :=
  π - 2 * arccos ((5 + 2 * sqrt 3) / 13) -- Considering one solution of the given form

theorem cone_vertex_angle
  (V_cone V_sphere : ℝ) (r h r_s : ℝ)
  (H_volumes : (1/3) * π * r^2 * h = 3 * (4/3) * π * r_s^3)
  (H_r_s : r_s = (r * h) / sqrt(h^2 + r^2)) :
  vertex_angle_of_cone r h = π - 2 * arccos ((5 + 2 * sqrt 3) / 13) :=
begin
  sorry
end

end cone_vertex_angle_l676_676587


namespace inscribed_angle_theorem_l676_676793

-- Define the Inscribed Angle Theorem in a Lean statement
theorem inscribed_angle_theorem {O A B C : Point} (hO : IsCenter O)
  (hA : OnCircle O A) (hB : OnCircle O B) (hC : OnCircle O C)
  (hArc : SubtendsArc O A B C) : MeasureAngle A C B = 1/2 * MeasureAngle A O B := 
sorry

end inscribed_angle_theorem_l676_676793


namespace combined_weight_of_bags_l676_676383

variable (WeightOfJamesBag WeightOfOliverBag WeightOfElizabethBag CombinedWeight : ℚ)

theorem combined_weight_of_bags :
  WeightOfJamesBag = 18 →
  WeightOfOliverBag = 1/6 * WeightOfJamesBag →
  WeightOfElizabethBag = 3/4 * WeightOfJamesBag →
  CombinedWeight = 2 * WeightOfOliverBag + WeightOfElizabethBag →
  CombinedWeight = 19.5 :=
by
  intros hJames hOliver hElizabeth hCombined
  rw [hJames, hOliver, hElizabeth] at hCombined
  linarith

end combined_weight_of_bags_l676_676383


namespace cyclic_iff_ap_eq_cp_l676_676862

variable {A B C D P : Type} [euclidean_space A B C D P]

-- Conditions

axiom CONVEX_ABCD : convex A B C D
axiom NOT_BISECTOR_ABCD : ¬(bisector (diagonal B D) ∠ A B C ∧ bisector (diagonal B D) ∠ C D A)
axiom POINT_IN_ABCD : point_in A B C D P
axiom ANGLE_CONDITION1 : ∠ P B C = ∠ D B A
axiom ANGLE_CONDITION2 : ∠ P D C = ∠ B D A

-- Statement

theorem cyclic_iff_ap_eq_cp :
  (cyclic A B C D) ↔ (distance A P = distance C P) :=
by
  sorry

end cyclic_iff_ap_eq_cp_l676_676862


namespace minimize_quadratic_l676_676471

theorem minimize_quadratic (x : ℝ) : (x = -9 / 2) → ∀ y : ℝ, y^2 + 9 * y + 7 ≥ (-9 / 2)^2 + 9 * -9 / 2 + 7 :=
by sorry

end minimize_quadratic_l676_676471


namespace cos_seven_theta_l676_676682

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end cos_seven_theta_l676_676682


namespace total_chairs_l676_676792

/-- Susan loves chairs. In her house, there are red chairs, yellow chairs, blue chairs, and green chairs.
    There are 5 red chairs. There are 4 times as many yellow chairs as red chairs.
    There are 2 fewer blue chairs than yellow chairs. The number of green chairs is half the sum of the number of red chairs and blue chairs (rounded down).
    We want to determine the total number of chairs in Susan's house. -/
theorem total_chairs (r y b g : ℕ) 
  (hr : r = 5)
  (hy : y = 4 * r) 
  (hb : b = y - 2) 
  (hg : g = (r + b) / 2) :
  r + y + b + g = 54 := 
sorry

end total_chairs_l676_676792


namespace median_AD_range_l676_676559

variable (A B C D : Type)
variable [InnerProductSpace ℝ A]
variable [AffineSpace A B]
variable (A B C D : A)
variable (AC AB AD : ℝ)

def median_AD_condition (AC AB AD : ℝ) : Prop :=
  AC = (sqrt (5 - AB) + sqrt (2 * AB - 10) + AB + 1) / 2

theorem median_AD_range
  (h : median_AD_condition A B C D)
  (AC_eq : AC = (sqrt (5 - AB) + sqrt (2 * AB - 10) + AB + 1) / 2)
  (AB_eq : AB = 5) :
  1 < AD ∧ AD < 4 := sorry

end median_AD_range_l676_676559


namespace perimeter_of_square_l676_676861

open Real

theorem perimeter_of_square (a : ℝ) : 
  (let l := √2 * a
       AO := l / 2
       OO1 := l / 4
       O1O2 := l / 8
       O2O3 := l / 16 in
       AO + OO1 + O1O2 + O2O3 = 15 * √2) →
  4 * a = 64 :=
by
  intro h
  have h1: AO + OO1 + O1O2 + O2O3 = (l / 2) + (l / 4) + (l / 8) + (l / 16) by sorry
  rw [h1] at h
  have h2: (l / 2) + (l / 4) + (l / 8) + (l / 16) = l * (15 / 16) by sorry
  rw [h2, mul_comm (15 / 16) l] at h
  have h3: l * (15 / 16) = 15 * √2 by sorry
  rw [h3] at h
  have h4: l = 16 * √2 by sorry
  rw [h4] at h
  have h5: √2 * a = 16 * √2 by rw [h4]
  have h6: a = 16 by sorry
  rw [h6] at *
  exact h6

end perimeter_of_square_l676_676861


namespace least_positive_multiple_of_45_with_product_of_digits_multiple_of_9_l676_676062

noncomputable def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else n.digits 10

noncomputable def product_of_digits (n : ℕ) : ℕ :=
  (digits n).foldl (λ x y => x * y) 1

def is_multiple_of (m n : ℕ) : Prop :=
  ∃ k, n = m * k

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of 45 n ∧ is_multiple_of 9 (product_of_digits n)

theorem least_positive_multiple_of_45_with_product_of_digits_multiple_of_9 : 
  ∀ n, satisfies_conditions n → 495 ≤ n :=
by
  sorry

end least_positive_multiple_of_45_with_product_of_digits_multiple_of_9_l676_676062


namespace solve_equation_l676_676431

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) : (2 / x = 1 / (x + 1)) → x = -2 :=
begin
  sorry
end

end solve_equation_l676_676431


namespace find_number_l676_676499

theorem find_number (x : ℤ) (h : 4 * x - 7 = 13) : x = 5 := 
sorry

end find_number_l676_676499


namespace find_all_positive_integers_l676_676603

def has_at_least_four_divisors (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, a < b ∧ b < c ∧ c < d ∧ a * b * c * d = n

def divisors_form_geometric_sequence (n : ℕ) : Prop :=
  ∃ k (d : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i < k → d i * d (i + 1) = d (0) * n) ∧
  (∀ i, 1 ≤ i ∧ i + 1 < k → d (i + 1) - d i = d (i) - d (i - 1))

theorem find_all_positive_integers (n : ℕ) :
  has_at_least_four_divisors n ∧ divisors_form_geometric_sequence n →
  ∃ p α : ℕ, Nat.Prime p ∧ α ≥ 3 ∧ n = p ^ α :=
sorry

end find_all_positive_integers_l676_676603


namespace clock_angle_7_to_8_l676_676615

noncomputable def angle_between_hands : ℝ → ℝ :=
  λ (t : ℝ), 
    let minute_angle := (t * 60 * 6) % 360 in
    let hour_angle := (210 + 30 * t) % 360 in
    let diff := abs (minute_angle - hour_angle) in
    min diff (360 - diff)

theorem clock_angle_7_to_8 : ∃ t : ℝ, 7 ≤ t ∧ t < 8 ∧ angle_between_hands (t - 7) = 120 :=
by
  use 7 + 16 / 60
  split
  · norm_num
  · split
  · norm_num
  apply eq_of_abs_sub_eq_zero
  calc _ = _ : sorry

end clock_angle_7_to_8_l676_676615


namespace ducks_remaining_l676_676818

theorem ducks_remaining (D_0 : ℕ) (D_0_eq : D_0 = 320) :
  let D_1 := D_0 - D_0 / 4,
      D_2 := D_1 - D_1 / 6,
      D_3 := D_2 - (3 * D_2) / 10 in
  D_3 = 140 :=
by
  sorry

end ducks_remaining_l676_676818


namespace M_union_N_neq_empty_necessary_but_not_sufficient_for_M_intersection_N_l676_676743

variables (M N : Set)

theorem M_union_N_neq_empty_necessary_but_not_sufficient_for_M_intersection_N (M N : Set) :
  (M ∪ N ≠ ∅) ↔ (M ∩ N ≠ ∅ → M ∪ N ≠ ∅) ∧ ¬ (M ∪ N ≠ ∅ → M ∩ N ≠ ∅) := sorry

end M_union_N_neq_empty_necessary_but_not_sufficient_for_M_intersection_N_l676_676743


namespace value_of_b6_plus_b_inv6_l676_676403

theorem value_of_b6_plus_b_inv6 (b : ℝ) (h : 3 = b + b⁻¹) : b^6 + b⁻6 = 322 := 
by 
  sorry

end value_of_b6_plus_b_inv6_l676_676403


namespace remainder_of_452867_div_9_l676_676064

theorem remainder_of_452867_div_9 : (452867 % 9) = 5 := by
  sorry

end remainder_of_452867_div_9_l676_676064


namespace greatest_divisor_of_sum_combinatorics_l676_676341

theorem greatest_divisor_of_sum_combinatorics (N : ℕ)
  (h : N = ∑ i in range 513, i * choose 512 i) :
  ∃ a : ℕ, (2 ^ a ∣ N) ∧ a = 520 :=
by
  sorry

end greatest_divisor_of_sum_combinatorics_l676_676341


namespace unique_zero_function_l676_676921

theorem unique_zero_function (f : ℝ → ℝ) 
    (h1 : ∀ x y : ℝ, f(x + y) + f(x - y) = f(x) * f(y))
    (h2 : tendsto f at_top (𝓝 0)) : 
    ∀ x : ℝ, f(x) = 0 :=
by
  sorry

end unique_zero_function_l676_676921


namespace tan_half_angle_product_values_l676_676266

theorem tan_half_angle_product_values
  (a b : ℝ)
  (h : 3 * (cos a + sin b) + 7 * (cos a * cos b + 1) = 0) :
  ∃ (x : ℝ), x = 3 ∨ x = -3 :=
sorry

end tan_half_angle_product_values_l676_676266


namespace find_difference_in_ticket_costs_l676_676765

-- Conditions
def num_adults : ℕ := 9
def num_children : ℕ := 7
def cost_adult_ticket : ℕ := 11
def cost_child_ticket : ℕ := 7

def total_cost_adults : ℕ := num_adults * cost_adult_ticket
def total_cost_children : ℕ := num_children * cost_child_ticket
def total_tickets : ℕ := num_adults + num_children

-- Discount conditions (not needed for this proof since they don't apply)
def apply_discount (total_tickets : ℕ) (total_cost : ℕ) : ℕ :=
  if total_tickets >= 10 ∧ total_tickets <= 12 then
    total_cost * 9 / 10
  else if total_tickets >= 13 ∧ total_tickets <= 15 then
    total_cost * 85 / 100
  else
    total_cost

-- The main statement to prove
theorem find_difference_in_ticket_costs : total_cost_adults - total_cost_children = 50 := by
  sorry

end find_difference_in_ticket_costs_l676_676765


namespace natasha_average_speed_climbing_l676_676382

-- Let the average speed while climbing be a variable to prove
def averageSpeedClimbing (t1 t2 : ℝ) (S : ℝ) (D : ℝ) : ℝ :=
  D / t1

theorem natasha_average_speed_climbing :
  ∀ (t1 t2 S T D : ℝ), t1 = 4 ∧ t2 = 2 ∧ S = 2 ∧ T = t1 + t2 ∧ 2 * D = S * T → averageSpeedClimbing t1 t2 S D = 1.5 :=
by
  intros t1 t2 S T D h
  sorry

end natasha_average_speed_climbing_l676_676382


namespace smallest_n_common_factor_l676_676165

theorem smallest_n_common_factor (n : ℕ) (h₁ : 11 * n - 4 > 0) (h₂ : 8 * n + 6 > 0) :
  ∃ n, gcd (11 * n - 4) (8 * n + 6) > 1 ∧ n = 85 :=
begin
  sorry
end

end smallest_n_common_factor_l676_676165


namespace broadcasting_methods_count_l676_676501

def advertisements : Type := fin 5  -- We have 5 positions for advertisements

def commercial_ads : fin 3 -- We have 3 commercial ads
def olympic_ads : fin 2 -- We have 2 Olympic promotional ads

def isOlympic (n : fin 5) : Prop := n = 3 ∨ n = 4
def isCommercial (n : fin 5) : Prop := n ≠ 3 ∧ n ≠ 4

noncomputable def count_broadcasting_methods : ℕ :=
  let permute := list.permutations [0, 1, 2, 3, 4] -- Permutations of the positions
  let valid_perms := permute.filter (λ l, l.reverse.head = 4 ∧ ¬ (l.get (3-1) = 3 ∨ l.get 3 = 4)) -- Last ad must be Olympic and no consecutive Olympic ads
  valid_perms.length

theorem broadcasting_methods_count :
  count_broadcasting_methods = 18 := by
  sorry

end broadcasting_methods_count_l676_676501


namespace cube_edge_length_l676_676185

def radius := 2
def edge_length (r : ℕ) := 4 + 2 * r

theorem cube_edge_length :
  ∀ r : ℕ, r = radius → edge_length r = 8 :=
by
  intros r h
  rw [h, edge_length]
  rfl

end cube_edge_length_l676_676185


namespace surface_area_of_regular_triangular_prism_is_correct_l676_676991

-- We define the conditions and question
def sphere_volume := (4 * Real.pi) / 3
def radius_of_sphere (V : ℝ) : ℝ := (3 * V / (4 * Real.pi))^(1/3)
def radius : ℝ := radius_of_sphere sphere_volume

def height_of_prism (r : ℝ) : ℝ := 2 * r
def inscribed_circle_radius (r : ℝ) : ℝ := r
def side_length_of_base (r : ℝ) : ℝ := 2 * r * (Real.sqrt 3) / 3

def surface_area_of_prism (r : ℝ) : ℝ :=
  let h := height_of_prism r
  let a := side_length_of_base r
  2 * (Real.sqrt 3 / 4) * a^2 + 3 * a * h

theorem surface_area_of_regular_triangular_prism_is_correct :
  surface_area_of_prism radius = 18 * Real.sqrt 3 :=
  sorry

end surface_area_of_regular_triangular_prism_is_correct_l676_676991


namespace S_4n_l676_676187

variable {a : ℕ → ℕ}
variable (S : ℕ → ℝ)
variable (n : ℕ)
variable (r : ℝ)
variable (a1 : ℝ)

-- Conditions
axiom geometric_sequence : ∀ n, a (n + 1) = a n * r
axiom positive_terms : ∀ n, 0 < a n
axiom sum_n : S n = a1 * (1 - r^n) / (1 - r)
axiom sum_3n : S (3 * n) = 14
axiom sum_n_value : S n = 2

-- Theorem
theorem S_4n : S (4 * n) = 30 :=
sorry

end S_4n_l676_676187


namespace jump_length_third_frog_l676_676047

theorem jump_length_third_frog (A B C : ℝ) 
  (h1 : A = (B + C) / 2) 
  (h2 : B = (A + C) / 2) 
  (h3 : |B - A| + |(B - C) / 2| = 60) : 
  |C - (A + B) / 2| = 30 :=
sorry

end jump_length_third_frog_l676_676047


namespace cos_seven_theta_l676_676683

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end cos_seven_theta_l676_676683


namespace new_lines_two_points_new_lines_one_common_point_one_and_only_one_new_line_l676_676210

-- Defining a parabola and its translations, along with lines parallel to its axis.
def new_lines (a b : ℝ) (c : ℝ) : ℝ → ℝ → Prop :=
  λ x y, y = (x + a)^2 + b ∨ x = c

/-- 
(a) For any two distinct points in the plane, there exists a new line that passes through both points.
-/
theorem new_lines_two_points (P1 P2 : ℝ × ℝ) (hP : P1 ≠ P2) :
  ∃ (a b c : ℝ), ∀ (x y : ℝ), (new_lines a b c x y → (x = P1.1 ∧ y = P1.2)) ∧ (new_lines a b c x y → (x = P2.1 ∧ y = P2.2)) :=
sorry

/-- 
(b) Any two distinct new lines have at most one common point.
-/
theorem new_lines_one_common_point (a1 b1 c1 a2 b2 c2 : ℝ) (h : a1 ≠ a2 ∨ b1 ≠ b2 ∨ c1 ≠ c2) :
  ∀ x y x' y', (new_lines a1 b1 c1 x y ∧ new_lines a2 b2 c2 x' y') → (x = x' ∧ y = y') :=
sorry

/-- 
(c) For a given new line and a point not lying on it, there exists one and only one new line that passes through the point and has no common point with the given new line.
-/
theorem one_and_only_one_new_line (a b c : ℝ) (P : ℝ × ℝ) (hP : ¬(new_lines a b c P.1 P.2)) :
  ∃ (a' b' c' : ℝ), (∀ x y, new_lines a' b' c' x y → (x = P.1 ∧ y = P.2)) ∧ ∀ x y, (new_lines a b c x y ∧ new_lines a' b' c' x y) → false :=
sorry

end new_lines_two_points_new_lines_one_common_point_one_and_only_one_new_line_l676_676210


namespace shaded_area_l676_676837

theorem shaded_area (d : ℝ) (k : ℝ) (π : ℝ) (r : ℝ)
  (h_diameter : d = 6) 
  (h_radius_large : k = 5)
  (h_small_radius: r = d / 2) :
  ((π * (k * r)^2) - (π * r^2)) = 216 * π :=
by
  sorry

end shaded_area_l676_676837


namespace min_sets_bound_l676_676344

theorem min_sets_bound (A : Type) (n k : ℕ) (S : Finset (Finset A))
  (h₁ : S.card = k)
  (h₂ : ∀ x y : A, x ≠ y → ∃ B ∈ S, (x ∈ B ∧ y ∉ B) ∨ (y ∈ B ∧ x ∉ B)) :
  2^k ≥ n :=
sorry

end min_sets_bound_l676_676344


namespace train_crossing_time_l676_676103

noncomputable def time_to_cross_platform (speed_kmph : ℝ) (length_train length_platform : ℝ) : ℝ :=
  let speed_ms := speed_kmph * (1000 / 3600) in
  let total_distance := length_train + length_platform in
  total_distance / speed_ms

theorem train_crossing_time :
  time_to_cross_platform 72 230.0416 290 = 26.00208 :=
by
  sorry

end train_crossing_time_l676_676103


namespace erasers_difference_l676_676555

-- Definitions for the conditions in the problem
def andrea_erasers : ℕ := 4
def anya_erasers : ℕ := 4 * andrea_erasers

-- Theorem statement to prove the final answer
theorem erasers_difference : anya_erasers - andrea_erasers = 12 :=
by
  -- Proof placeholder
  sorry

end erasers_difference_l676_676555


namespace correct_option_c_l676_676473

theorem correct_option_c : ∃ (n : ℤ), n = -2 ∧ real.cbrt -8 = n := 
by {
  use -2,
  split,
  exact rfl,
  sorry
}

end correct_option_c_l676_676473


namespace jane_percentage_bread_to_treats_l676_676735

variable (T J_b W_b W_t : ℕ) (P : ℕ)

-- Conditions as stated
axiom h1 : J_b = (P * T) / 100
axiom h2 : W_t = T / 2
axiom h3 : W_b = 3 * W_t
axiom h4 : W_b = 90
axiom h5 : J_b + W_b + T + W_t = 225

theorem jane_percentage_bread_to_treats : P = 75 :=
by
-- Proof skeleton
sorry

end jane_percentage_bread_to_treats_l676_676735


namespace max_robots_count_ways_to_place_robots_l676_676526

open Finset

-- Definitions and context setting
def palaceGraph (V E : ℕ) (H : V = 32 ∧ E = 40) := ∃ (G : Type*) [graph G] (V : finset G) (E : finset (G × G)),
  V.card = 32 ∧ E.card = 40 ∧ ∀ (v ∈ V), ∃ u, (u, v) ∈ E ∨ (v, u) ∈ E

-- Given conditions: 32 rooms, 40 corridors, robots move without meeting
def robotInRoom (G : Type*) [graph G] (V : finset G) (H : V.card = 32) := 

-- Maximum matching
def maxMatching (N : ℕ) : Prop :=
∀ (G : Type*) [graph G] (V : finset G) (E : finset (G × G))
  (H1 : V.card = 32) (H2 : E.card = 40),
  (∃ (M : finset (G × G)), M.card = N ∧ is_max_matching M)

-- Proof goal: maximum value of N is 16
theorem max_robots (V E : ℕ) (H : palaceGraph V E (by refl)) :
  maxMatching 16 :=
sorry

-- Number of ways to place 16 robots in 32 rooms considering they are indistinguishable
theorem count_ways_to_place_robots : 
  ∃ N, N = 16 ∧ N.choose 16 = binomial 32 16 :=
sorry

end max_robots_count_ways_to_place_robots_l676_676526


namespace no_natural_number_divisible_2014_l676_676592

theorem no_natural_number_divisible_2014 : ¬∃ n s : ℕ, n = 2014 * s + 2014 := 
by
  -- Assume for contradiction that such numbers exist
  intro ⟨n, s, h⟩,
  -- Consider the transformed equation for contradiction
  have h' : n - s = 2013 * s + 2014, from sorry,
  -- Check the divisibility by 3 leading to contradiction
  have div_contr : (2013 * s + 2014) % 3 = 1, from sorry,
  -- Using the contradiction to close the proof
  sorry

end no_natural_number_divisible_2014_l676_676592


namespace number_of_valid_5_digit_numbers_l676_676496

theorem number_of_valid_5_digit_numbers :
  {ABCDE : ℕ // let A := (ABCDE / 10000) % 10,
               let B := (ABCDE / 1000) % 10,
               let C := (ABCDE / 100) % 10,
               let D := (ABCDE / 10) % 10,
               let E := ABCDE % 10 in
               A ≠ 0 ∧ A + B = C ∧ B + C = D ∧ C + D = E} = 6 :=
sorry

end number_of_valid_5_digit_numbers_l676_676496


namespace isosceles_triangle_vertex_angle_cosine_l676_676240

-- Given conditions
variables {B : ℝ}
axiom sin_B : Real.sin B = 4 / 5

-- Property of isosceles triangle with angles summing to π
-- Compute the cosine of the vertex angle
theorem isosceles_triangle_vertex_angle_cosine (h : B = B) :
  Real.cos (π - 2 * B) = 7 / 25 :=
by
  have h1 : Real.sin B ^ 2 = (4 / 5) ^ 2, from by rw sin_B; norm_num,
  have h2 : Real.cos (2 * B) = 1 - 2 * (Real.sin B) ^ 2, from Real.cos_two_mul B,
  rw [h2, h1],
  norm_num,
  sorry

end isosceles_triangle_vertex_angle_cosine_l676_676240


namespace range_of_omega_l676_676223

noncomputable def monotonic_decreasing_omega (omega : ℝ) : Prop :=
  ∀ x y : ℝ, (π / 2 < x) ∧ (x < π) ∧ (π / 2 < y) ∧ (y < π) ∧ (x < y) →
  sin (omega * y + π / 4) < sin (omega * x + π / 4)

theorem range_of_omega :
  {ω : ℝ | monotonic_decreasing_omega ω} = Icc (1 / 2 : ℝ) (5 / 4 : ℝ) :=
sorry

end range_of_omega_l676_676223


namespace original_price_of_sneakers_l676_676737

-- Total earnings
def total_earnings : ℝ := 232

-- Discount and tax rate
def discount_rate : ℝ := 0.08
def tax_rate : ℝ := 0.07

-- Final price after discount and tax
def final_price (P : ℝ) : ℝ :=
  let discounted_price := P * (1 - discount_rate)
  in discounted_price * (1 + tax_rate)

-- The theorem we want to prove
theorem original_price_of_sneakers : ∃ P : ℝ, final_price P = total_earnings :=
  by
    -- Proof is omitted
    sorry

end original_price_of_sneakers_l676_676737


namespace angle_bisector_construction_l676_676579

theorem angle_bisector_construction 
  (A B C D E F : Type*)
  [point : (Type*)]
  [distance_measurer : point → point → ℝ]
  [line : point → point → set point]
  (t : ℝ)
  (σ : ℝ)
  (Hσ : 0 < σ ∧ σ < 90)
  (H1 : distance_measurer C D = t)
  (H2 : distance_measurer C E = t)
  (H3 : line D E = line E D)
  : line C F is the angle bisector of ∠(A C B) :=
begin
  sorry
end

end angle_bisector_construction_l676_676579


namespace cone_to_cylinder_volume_ratio_l676_676100

noncomputable def cylinder_volume (r h : ℝ) : ℝ := 
  π * r^2 * h

noncomputable def cone_volume (r h : ℝ) : ℝ := 
  (1 / 3) * π * r^2 * h

theorem cone_to_cylinder_volume_ratio : 
  let r := 5 in 
  let h := 18 in 
  let h_cone := h / 3 in 
  let V_cyl := cylinder_volume r h in 
  let V_cone := cone_volume r h_cone in 
  V_cone / V_cyl = 1 / 9 :=
by 
  sorry

end cone_to_cylinder_volume_ratio_l676_676100


namespace find_xy_l676_676674

noncomputable def solution (x y : ℝ) : Prop :=
  (x + y) * complex.I = x - 1

theorem find_xy (x y : ℝ) : solution x y → (x = 1 ∧ y = -1) :=
begin
  intros h,
  -- The proof would go here
  sorry
end

end find_xy_l676_676674


namespace distance_between_points_l676_676955

noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem distance_between_points :
  dist (3, 3) (-2, -2) = 5 * Real.sqrt 2 := 
by
  sorry

end distance_between_points_l676_676955


namespace cos_seven_theta_l676_676692

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end cos_seven_theta_l676_676692


namespace sum_of_sequence_2012_l676_676648

noncomputable def f (x b : ℝ) := x^2 + 2 * b * x

theorem sum_of_sequence_2012 (h : f 1 b = 2) :
  let S_n := λ n, ∑ i in Finset.range n, 1 / (f (i+1) b)
  in S_n 2012 = 2012 / 2013 :=
by
  sorry

end sum_of_sequence_2012_l676_676648


namespace sum_of_marked_angles_l676_676462

-- Let's define the angles as being exterior angles of a quadrilateral
variables (p q r s : ℝ)

-- Given condition: Exterior angles of a quadrilateral sum to 360 degrees
axiom sum_of_exterior_angles_of_quadrilateral (p q r s : ℝ) : p + q + r + s = 360

-- Lean 4 statement for the problem:
theorem sum_of_marked_angles (p q r s : ℝ) (h : p + q + r + s = 360) : 2 * (p + q + r + s) = 720 :=
by {
  -- Using the given condition
  rw h,
  -- Simplifying the expression
  exact eq.refl (2 * 360),
}

end sum_of_marked_angles_l676_676462


namespace unique_parallel_line_through_point_l676_676234

-- Assuming necessary geometric definitions and axioms are available in Mathlib

variable {α β γ : Type} [plane α] [plane β] [plane γ] [line a] [point B]
-- Definitions for parallel planes and line containment
variable {plane_parallel : α ∥ β}
variable {line_in_plane : a ⊂ α}
variable {point_in_plane : B ∈ β}

-- The statement to be proved:
theorem unique_parallel_line_through_point
  (plane_parallel : α ∥ β)
  (line_in_plane : a ⊂ α)
  (point_in_plane : B ∈ β) :
  ∃! b : line, (b ∈ β ∧ B ∈ b ∧ b ∥ a) :=
sorry

end unique_parallel_line_through_point_l676_676234


namespace problem1_problem2_l676_676242

noncomputable def f (x : ℝ) : ℝ :=
  (sqrt 3 / 2) * sin (2 * x) - cos (x) ^ 2 + (1 / 2)

def g (x : ℝ) : ℝ :=
  f (x + π / 6)

def range_f_correct : Prop :=
  ∀ x, x ∈ set.Icc 0 (π / 2) → f x ∈ set.Icc (-1 / 2) 1

def increasing_intervals_g_correct : Prop :=
  ∀ k : ℤ, ∀ x, x ∈ set.Icc (k * π - π / 3) (k * π + π / 6) → deriv (λ x, g x) x > 0

theorem problem1 : range_f_correct := sorry

theorem problem2 : increasing_intervals_g_correct := sorry

end problem1_problem2_l676_676242


namespace max_value_a_plus_b_l676_676354

theorem max_value_a_plus_b
  (a b : ℝ)
  (h1 : 4 * a + 3 * b ≤ 10)
  (h2 : 3 * a + 5 * b ≤ 11) :
  a + b ≤ 156 / 55 :=
sorry

end max_value_a_plus_b_l676_676354


namespace contractor_needs_more_people_l676_676853

theorem contractor_needs_more_people 
  (initial_days : ℕ) (total_people : ℕ) (elapsed_days : ℕ)
  (work_done : ℚ) (total_days : ℕ) (remaining_work : ℚ) : 
  ∃ (additional_people : ℕ), additional_people = 35 :=
by
  have remaining_days := total_days - elapsed_days
  have work_per_day_done := work_done / elapsed_days
  have work_per_day_needed := remaining_work / remaining_days
  have additional_people_needed := work_per_day_needed * total_people - work_per_day_done * total_people / work_per_day_done
  have additional_people := additional_people_needed - total_people
  use 35
  sorry

end contractor_needs_more_people_l676_676853


namespace probability_at_least_40_cents_heads_l676_676000

theorem probability_at_least_40_cents_heads :
  let coins := {50, 25, 10, 5, 1}
  3 / 8 =
    (∑ H in {x ∈ (Finset.powerset coins) | (x.sum ≥ 40)}, (1 / 2) ^ x.card) :=
sorry

end probability_at_least_40_cents_heads_l676_676000


namespace compute_g256_l676_676582

open Real

def g : ℕ → ℝ
| x := if ∃ k : ℕ, (16 : ℝ) ^ k = x then log 16 x else 2 + g (x - 1)

theorem compute_g256 : g 256 = 2 := by
  sorry

end compute_g256_l676_676582


namespace repeating_decimal_addition_and_multiply_l676_676569

theorem repeating_decimal_addition_and_multiply
  (h₀₄ : 0.\overline{4} = (4 : ℚ) / 9)
  (h₀₅ : 0.\overline{5} = (5 : ℚ) / 9) :
  ((0.\overline{4} + 0.\overline{5}) * 3 = 3) := by
  sorry

end repeating_decimal_addition_and_multiply_l676_676569


namespace pieces_from_rod_l676_676673

theorem pieces_from_rod (yards_to_feet : 1 * 3 = 3)
                        (foot_to_inches : 1 * 12 = 12) :
  ∀ (rod_length_yards : ℕ) (piece_length_feet : ℕ) (piece_length_inches : ℕ),
    rod_length_yards = 2187 →
    piece_length_feet = 8 →
    piece_length_inches = 3 →
    (⌊(rod_length_yards * 3) / (piece_length_feet + (piece_length_inches / 12))⌋ : ℕ) = 795 :=
by
  intros rod_length_yards piece_length_feet piece_length_inches h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end pieces_from_rod_l676_676673


namespace P_2010_at_minus_half_l676_676152

theorem P_2010_at_minus_half :
  ∃ P_2010 : ℤ → ℤ, 
  (∀ n : ℕ, P_2010 n = ∑ i in Finset.range (n + 1), i ^ 2010) ∧ 
  P_2010 (-1 / 2) = 0 := 
sorry

end P_2010_at_minus_half_l676_676152


namespace reciprocal_of_5_is_1_div_5_l676_676418

-- Define the concept of reciprocal
def is_reciprocal (a b : ℚ) : Prop := a * b = 1

-- The problem statement: Prove that the reciprocal of 5 is 1/5
theorem reciprocal_of_5_is_1_div_5 : is_reciprocal 5 (1 / 5) :=
by
  sorry

end reciprocal_of_5_is_1_div_5_l676_676418


namespace arrangement_plans_count_l676_676306

-- Define the teachers
inductive Teachers
| A | B | C | D | E | F

-- Define the conditions
def lessons : ℕ := 4
def teacher_set : Finset Teachers := {Teachers.A, Teachers.B, Teachers.C, Teachers.D, Teachers.E, Teachers.F}

def first_lesson_teachers : Finset Teachers := {Teachers.A, Teachers.B}
def fourth_lesson_teachers : Finset Teachers := {Teachers.A, Teachers.C}

-- Statement of the problem
theorem arrangement_plans_count :
  let arrangements := {g : Fin (nat.succ (nat.succ (nat.succ (nat.succ 0)))) → Teachers // 
    g 0 ∈ first_lesson_teachers ∧ g 3 ∈ fourth_lesson_teachers ∧ (Finset.image g (finset.range 4)).card = 4} in
  arrangements.card = 36 :=
sorry

end arrangement_plans_count_l676_676306


namespace Amanda_hiking_trip_l676_676897

-- Define the conditions
variable (x : ℝ) -- the total distance of Amanda's hiking trip
variable (forest_path : ℝ) (plain_path : ℝ)
variable (stream_path : ℝ) (mountain_path : ℝ)

-- Given conditions
axiom h1 : stream_path = (1/4) * x
axiom h2 : forest_path = 25
axiom h3 : mountain_path = (1/6) * x
axiom h4 : plain_path = 2 * forest_path
axiom h5 : stream_path + forest_path + mountain_path + plain_path = x

-- Proposition to prove
theorem Amanda_hiking_trip : x = 900 / 7 :=
by
  sorry

end Amanda_hiking_trip_l676_676897


namespace boat_distance_against_stream_l676_676716

theorem boat_distance_against_stream 
  (v_b : ℝ)
  (v_s : ℝ)
  (distance_downstream : ℝ)
  (t : ℝ)
  (speed_downstream : v_s + v_b = 11)
  (speed_still_water : v_b = 8)
  (time : t = 1) :
  (v_b - (11 - v_b)) * t = 5 :=
by
  -- Here we're given the initial conditions and have to show the final distance against the stream is 5 km
  sorry

end boat_distance_against_stream_l676_676716


namespace original_price_of_radio_l676_676531

theorem original_price_of_radio (selling_price : ℝ) (loss_percentage : ℝ) 
  (selling_price_eq : selling_price = 1330) (loss_percentage_eq : loss_percentage = 0.30) : 
  ∃ C : ℝ, 0.70 * C = selling_price ∧ C = 1900 :=
by 
  intros 
  use 1900
  rw [selling_price_eq, loss_percentage_eq]
  sorry

end original_price_of_radio_l676_676531


namespace find_a_from_distance_condition_l676_676241

-- Define circle's equation and find center
def circle_equation (x y : ℝ) :=
  x^2 + y^2 - 2*x - 8*y + 1 = 0

-- Define line's equation in terms of a
def line_equation (a x y : ℝ) :=
  a*x - y + 1 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ :=
  (1, 4)

-- Define the distance function from the center to the line
def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (a * 1 - 4 + 1) / sqrt (a^2 + 1)

-- The proof statement
theorem find_a_from_distance_condition (a : ℝ) :
  distance_from_center_to_line a = 1 → a = 4 / 3 :=
by
  sorry -- Proof goes here

end find_a_from_distance_condition_l676_676241


namespace prob_1_geo_seq_prob_2_max_sum_prob_3_min_k_l676_676628

-- Define the sequence according to given conditions
def seq_an (n : ℕ) (t : ℝ) : ℝ := 
  sorry -- as described: a_{n+1} = 1/2 a_n + t, a_1 = 1/2

-- Problem 1: Prove that (a_n - 2t) is a geometric sequence
theorem prob_1_geo_seq (t : ℝ) (h_t : t ≠ 1/4) : 
  ∃ (r : ℝ), ∀ n : ℕ, (seq_an n t - 2 * t) = (seq_an 0 t - 2 * t) * r^n := 
sorry

-- Problem 2: Find the maximum sum of first several terms for t = -1/8
theorem prob_2_max_sum : 
  ∃ (n : ℕ), n ≤ 2 ∧ (∑ i in Finset.range (n+1), seq_an i (-1/8)) = 
  (∑ i in Finset.range 3, seq_an i (-1/8)) := 
sorry

-- Problem 3: Find minimum k when t = 0 such that the inequality holds
def seq_cn (n : ℕ) : ℝ := 4 * seq_an n 0 + 1

def sum_cn (n : ℕ) : ℝ := 
  ∑ i in Finset.range (n + 1), seq_cn i

theorem prob_3_min_k : 
  ∃ (k : ℝ), ∀ n : ℕ, n > 0 → 12 * k / (4 + n - sum_cn n) ≥ 2 * n - 7 ∧ k ≥ 1/32 := 
sorry

end prob_1_geo_seq_prob_2_max_sum_prob_3_min_k_l676_676628


namespace circle_area_5pi_l676_676257

attribute [instance] classical.prop_decidable

-- Given conditions
def l1 (x y : ℝ) : Prop := x - √3 * y + 2 = 0
def l2 (x y : ℝ) : Prop := x - √3 * y - 6 = 0
def chord_length (C : set (ℝ × ℝ)) (l : ℝ × ℝ → Prop) : ℝ := 2
def distance_between_lines : ℝ := 4

-- Problem statement
theorem circle_area_5pi (C : set (ℝ × ℝ)) (h1 : ∀ x y, l1 x y → ∃ z, (z√2) ∈ C) (h2 : ∀ x y, l2 x y → ∃ z, (z√2) ∈ C) : 
  ∃ r, (r : ℝ) = √5 ∧ (classical.some h1, (radius)^2 * real.pi) = 5 * real.pi := 
sorry

end circle_area_5pi_l676_676257


namespace closed_broken_line_segments_divisible_by_6_l676_676311

theorem closed_broken_line_segments_divisible_by_6 
(l : List (ℝ × ℝ × ℝ)) 
(h1 : ∀ e ∈ l, ∀ f ∈ l, ¬ e = f → ∥ e ∥ = ∥ f ∥) 
(h2 : ∀ (a b c : ℝ × ℝ × ℝ), a ∈ l → b ∈ l → c ∈ l → ¬ a = b → ¬ b = c → ¬ c = a → a ⊥ b ∧ b ⊥ c ∧ c ⊥ a) 
(h3 : ∑ e in l, e = (0, 0, 0)) 
: l.length % 6 = 0 := 
sorry

end closed_broken_line_segments_divisible_by_6_l676_676311


namespace six_digit_palindromes_count_l676_676262

open Nat

theorem six_digit_palindromes_count :
  let digits := {d | 0 ≤ d ∧ d ≤ 9}
  let a_digits := {a | 1 ≤ a ∧ a ≤ 9}
  let b_digits := digits
  let c_digits := digits
  ∃ (total : ℕ), (∀ a ∈ a_digits, ∀ b ∈ b_digits, ∀ c ∈ c_digits, True) → total = 900 :=
by
  sorry

end six_digit_palindromes_count_l676_676262


namespace length_of_goods_train_l676_676076

theorem length_of_goods_train (v_kmph : ℝ) (p : ℝ) (t : ℝ) (v_mps : ℝ) (Distance : ℝ) : 
  (v_kmph = 72) → (p = 250) → (t = 24) → (v_mps = v_kmph * (5 / 18)) → (Distance = v_mps * t) → 
  (Distance - p = 230) :=
by
  intros hv p_length ht hvmps hDistance
  rw [← hv, ← ht, ← p_length, ← hvmps, ← hDistance]
  sorry

end length_of_goods_train_l676_676076


namespace prime_solution_exists_l676_676161

theorem prime_solution_exists :
  ∃ (p q r : ℕ), p.Prime ∧ q.Prime ∧ r.Prime ∧ (p + q^2 = r^4) ∧ (p = 7) ∧ (q = 3) ∧ (r = 2) := 
by
  sorry

end prime_solution_exists_l676_676161


namespace new_minimum_point_of_translated_graph_l676_676803

noncomputable def sqrt_function (x : ℝ) : ℝ := Real.sqrt x - 5

theorem new_minimum_point_of_translated_graph :
  let f := sqrt_function
  let translated_f := λ x, f (x - 4) + 2
  ∃ p : ℝ × ℝ, p = (4, -1) ∧ ∀ x : ℝ, translated_f x ≥ p.snd :=
by
  sorry

end new_minimum_point_of_translated_graph_l676_676803


namespace incorrect_statement_A_l676_676577

def area_triangle (b h : ℝ) : ℝ := 1 / 2 * b * h
def area_square (s : ℝ) : ℝ := s ^ 2
def area_circle (r : ℝ) : ℝ := real.pi * r ^ 2
def volume_cube (s : ℝ) : ℝ := s ^ 3

theorem incorrect_statement_A (b h : ℝ) :
  area_triangle (3 * b) (2 * h) ≠ 4 * area_triangle b h :=
sorry

end incorrect_statement_A_l676_676577


namespace cover_points_with_circles_l676_676622

/-- Given 100 points on the plane, there exists a collection of circles whose diameters 
total less than 100 and the distance between any two of which is more than 1. --/
theorem cover_points_with_circles (points : Fin 100 -> ℝ × ℝ) :
  ∃ (circles : Fin 100 → ℝ×(ℝ × ℝ)), 
    (∀ i, circles i).1 < 50 ∧ -- this ensures each circle has a diameter < 2*50
    (∑ i in Finset.univ, (circles i).1) < 100 ∧ -- total diameter sum < 100
    (∀ i j, i ≠ j → dist (circles i).2 (circles j).2 > 1) -- centers more than 1 unit apart
  :=
sorry

end cover_points_with_circles_l676_676622


namespace largest_constant_D_l676_676174

theorem largest_constant_D : 
  (∀ x y : ℝ, 2 * x^2 + 2 * y^2 + 3 ≥ D * (x + y)) → D ≤ 2 * sqrt 3 :=
sorry

end largest_constant_D_l676_676174


namespace kenny_jumping_jacks_l676_676340

theorem kenny_jumping_jacks
  (last_week : ℕ := 324)
  (sunday : ℕ := 34)
  (monday : ℕ := 20)
  (tuesday : ℕ := 0)
  (wednesday : ℕ := 123)
  (thursday : ℕ := 64)
  (some_day : ℕ := 61)
  (friday : ℕ) :
  let total_so_far := sunday + monday + tuesday + wednesday + thursday,
      new_target := last_week + 1,
      remaining_needed := new_target - total_so_far in
  friday = remaining_needed - some_day :=
by
  sorry

end kenny_jumping_jacks_l676_676340


namespace integer_values_x_l676_676285

theorem integer_values_x {x : ℝ} (h : ⌈real.sqrt x⌉ = 20) : ∃ n : ℕ, n = 39 :=
by
  have h1 : 19 < real.sqrt x ∧ real.sqrt x ≤ 20 := sorry
  have h2 : 361 < x ∧ x ≤ 400 := sorry
  have x_values : set.Icc 362 400 := sorry
  have n_values : ∃ n : ℕ, n = set.finite.to_finset x_values.card := sorry
  exact n_values

end integer_values_x_l676_676285


namespace stickers_distribution_l676_676261

noncomputable def number_of_ways_to_distribute_stickers (stickers sheets : ℕ) (min_stickers_per_sheet : ℕ) : ℕ :=
  if stickers < sheets * min_stickers_per_sheet then 0
  else if (stickers - sheets * min_stickers_per_sheet) = 0 then 1
  else sorry  -- General case would need a function for combinations if > 0

theorem stickers_distribution :
  number_of_ways_to_distribute_stickers 10 5 2 = 1 :=
by {
  dsimp [number_of_ways_to_distribute_stickers],
  norm_num,
}

end stickers_distribution_l676_676261


namespace cassie_nails_claws_total_l676_676570

theorem cassie_nails_claws_total :
  let dogs := 4
  let parrots := 8
  let cats := 2
  let rabbits := 6
  let lizards := 5
  let tortoises := 3

  let dog_nails := dogs * 4 * 4

  let normal_parrots := 6
  let parrot_with_extra_toe := 1
  let parrot_missing_toe := 1
  let parrot_claws := (normal_parrots * 2 * 3) + (parrot_with_extra_toe * 2 * 4) + (parrot_missing_toe * 2 * 2)

  let normal_cats := 1
  let deformed_cat := 1
  let cat_toes := (1 * 4 * 5) + (1 * 4 * 4) + 1 

  let normal_rabbits := 5
  let deformed_rabbit := 1
  let rabbit_nails := (normal_rabbits * 4 * 9) + (3 * 9 + 2)

  let normal_lizards := 4
  let deformed_lizard := 1
  let lizard_toes := (normal_lizards * 4 * 5) + (deformed_lizard * 4 * 4)
  
  let normal_tortoises := 1
  let tortoise_with_extra_claw := 1
  let tortoise_missing_claw := 1
  let tortoise_claws := (normal_tortoises * 4 * 4) + (3 * 4 + 5) + (3 * 4 + 3)

  let total_nails_claws := dog_nails + parrot_claws + cat_toes + rabbit_nails + lizard_toes + tortoise_claws

  total_nails_claws = 524 :=
by
  sorry

end cassie_nails_claws_total_l676_676570


namespace sheila_earns_per_hour_l676_676393

theorem sheila_earns_per_hour
  (h1 : ∀ day : ℕ, (day = 1 ∨ day = 3 ∨ day = 5) → SheilaWorksHoursPerDay day = 8)
  (h2 : ∀ day : ℕ, (day = 2 ∨ day = 4) → SheilaWorksHoursPerDay day = 6)
  (h3 : SheilaWorksHoursPerDay 6 = 0)
  (h4 : SheilaWorksHoursPerDay 7 = 0)
  (weekly_earnings : ℕ)
  (weekly_earnings = 360) :
  (360 / (8 * 3 + 6 * 2)) = 10 := 
by
  sorry

end sheila_earns_per_hour_l676_676393


namespace eight_div_pow_64_l676_676567

theorem eight_div_pow_64 (h : 64 = 8^2) : 8^15 / 64^7 = 8 := by
  sorry

end eight_div_pow_64_l676_676567


namespace anya_more_erasers_l676_676558

theorem anya_more_erasers (anya_erasers andrea_erasers : ℕ)
  (h1 : anya_erasers = 4 * andrea_erasers)
  (h2 : andrea_erasers = 4) :
  anya_erasers - andrea_erasers = 12 := by
  sorry

end anya_more_erasers_l676_676558


namespace value_of_a2014_plus_b2013_l676_676254

-- Definition of the problem conditions and statement
theorem value_of_a2014_plus_b2013 {a b : ℝ} (h : ({a, b / a, 1} : set ℝ) = {a ^ 2, a + b, 0}) : 
  a ^ 2014 + b ^ 2013 = 1 :=
by
  -- Omitted proof
  sorry

end value_of_a2014_plus_b2013_l676_676254


namespace tangents_divide_side5_correctly_l676_676035

-- Definitions for the side lengths of the pentagon
def side_lengths : list ℚ := [5, 6, 7, 8, 9]

-- The conditions specify that the sides are tangent to a circle, yielding specific tangent lengths
def tangent_segments (x : ℚ) : Prop := 
  let a := side_lengths.head! in
  let b := side_lengths.get! 1 in
  let c := side_lengths.get! 2 in
  let d := side_lengths.get! 3 in
  let e := side_lengths.tail!.tail!.tail!.tail!.head! in
  -- Tangents must satisfy these equations derived from the problem
  a = (x + (2 + x)) ∧
  b = (6 - x) + x ∧
  c = (1 + x) + (6 - x) ∧
  d = (7 - x) + (1 + x) ∧
  e = (2 + x) + (7 - x)

-- The segments on the side of length 5 should be as derived
def division_of_side5 (segment1 segment2 : ℚ) : Prop := 
  side_lengths.head! = segment1 + segment2 ∧
  segment1 = 3 / 2 ∧ 
  segment2 = 5 / 2

theorem tangents_divide_side5_correctly :
  ∃ x : ℚ, tangent_segments x ∧ division_of_side5 (3 / 2) (5 / 2) :=
begin
  sorry
end

end tangents_divide_side5_correctly_l676_676035


namespace distance_between_points_l676_676954

noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem distance_between_points :
  dist (3, 3) (-2, -2) = 5 * Real.sqrt 2 := 
by
  sorry

end distance_between_points_l676_676954


namespace ratio_of_times_l676_676092

theorem ratio_of_times (distance : ℝ) (prev_time : ℝ) (new_speed : ℝ) (prev_speed : ℝ):
  distance = 504 →
  prev_time = 6 →
  new_speed = 56 →
  prev_speed = 84 →
  (distance / new_speed) / prev_time = 3 / 2 :=
by
  intros h_distance h_prev_time h_new_speed h_prev_speed
  rw [h_distance, h_prev_time, h_new_speed, h_prev_speed]
  have h_new_time : distance / new_speed = 9 := by
    rw [h_distance, h_new_speed]
    norm_num
  rw [h_new_time, h_prev_time]
  norm_num
  sorry

end ratio_of_times_l676_676092


namespace construct_triangle_l676_676916

variables (a : ℝ) (α : ℝ) (d : ℝ)

-- Helper definitions
def is_triangle_valid (a α d : ℝ) : Prop := sorry

-- The theorem to be proven
theorem construct_triangle (a α d : ℝ) : is_triangle_valid a α d :=
sorry

end construct_triangle_l676_676916


namespace internal_angle_bisector_external_angle_bisector_l676_676777

noncomputable def internal_angle_bisector_length (a b : ℝ) : ℝ :=
  (a * b * real.sqrt 2) / (a + b)

noncomputable def external_angle_bisector_length (a b : ℝ) : ℝ :=
  (a * b * real.sqrt 2) / (a - b)

theorem internal_angle_bisector (a b : ℝ) :
  ∀ (c : ℝ), ∀ (hypotenuse : c = real.sqrt (a^2 + b^2)), 
    ∃ (l_3 : ℝ), l_3 = internal_angle_bisector_length a b :=
by
  intros
  use internal_angle_bisector_length a b
  sorry

theorem external_angle_bisector (a b : ℝ) :
  ∀ (c : ℝ), ∀ (hypotenuse : c = real.sqrt (a^2 + b^2)), 
    ∃ (d : ℝ), d = external_angle_bisector_length a b :=
by
  intros
  use external_angle_bisector_length a b
  sorry

end internal_angle_bisector_external_angle_bisector_l676_676777


namespace find_multiplier_l676_676109

theorem find_multiplier (A N : ℕ) (h : A = 32) (eqn : N * (A + 4) - 4 * (A - 4) = A) : N = 4 :=
sorry

end find_multiplier_l676_676109


namespace magnitude_power_eight_l676_676602

-- Definition of the complex number a
def a : ℂ := (1 / Real.sqrt 2) + (1 / Real.sqrt 2) * Complex.i

-- Statement of the theorem
theorem magnitude_power_eight : Complex.abs (a^8) = 1 := by
  -- Placeholder for the proof
  sorry

end magnitude_power_eight_l676_676602


namespace range_A_range_B_l676_676071

section
variables (x : ℝ)

def function_A : ℝ := 2^(x - 1)
def function_B : ℝ := 1 / (x^2)

-- Statement: Function A has the range (0, +∞)
theorem range_A : ∀ (y : ℝ), y > 0 ↔ ∃ (x : ℝ), function_A x = y :=
sorry

-- Statement: Function B has the range (0, +∞)
theorem range_B : ∀ (y : ℝ), y > 0 ↔ ∃ (x : ℝ), function_B x = y :=
sorry

end

end range_A_range_B_l676_676071


namespace required_more_visits_l676_676601

-- Define the conditions
def n := 395
def m := 2
def v1 := 135
def v2 := 112
def v3 := 97

-- Define the target statement
theorem required_more_visits : (n * m) - (v1 + v2 + v3) = 446 := by
  sorry

end required_more_visits_l676_676601


namespace george_fees_l676_676199

def initial_loan (principal : ℕ) (initial_rate : ℕ) (weeks : ℕ) : ℕ :=
  -- accumulate the fees weekly
  List.sum $ List.map (λ i, principal * (initial_rate * 2^i) / 100) (List.range weeks)

def additional_loan (principal : ℕ) (initial_rate : ℕ) (weeks : ℕ) : ℕ :=
  -- accumulate the fees weekly
  List.sum $ List.map (λ i, principal * (initial_rate * 2^i) / 100) (List.range weeks)

def total_fees : ℕ :=
  let initial_fee := initial_loan 100 5 4
  let additional_fee := additional_loan 50 4 2
  initial_fee + additional_fee

theorem george_fees : total_fees = 81 := sorry

end george_fees_l676_676199


namespace find_a3b3c3_l676_676747

variables {a b c k : ℝ}

-- The conditions stated in the problem
axiom distinct_real_numbers (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
axiom condition (h : (a^3 + 8) / a = (b^3 + 8) / b ∧ (b^3 + 8) / b = (c^3 + 8) / c)

-- The statement to be proved
theorem find_a3b3c3 (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h : (a^3 + 8) / a = (b^3 + 8) / b ∧ (b^3 + 8) / b = (c^3 + 8) / c) : 
  a^3 + b^3 + c^3 = -24 :=
  sorry

end find_a3b3c3_l676_676747


namespace chef_already_cooked_potatoes_l676_676097

theorem chef_already_cooked_potatoes :
  ∀ (t_total t_per t_remaining t_already : ℕ), 
    t_total = 12 ∧ 
    t_per = 6 ∧ 
    t_remaining = 36 → 
    t_already = t_total - t_remaining / t_per ↔ t_already = 6 := by
  intros t_total t_per t_remaining t_already h
  cases h with h1 h2
  cases h2 with h3 h4
  sorry

end chef_already_cooked_potatoes_l676_676097


namespace work_done_example_l676_676503

variable (F : ℝ → ℝ)
variable a b : ℝ

def work_done (F : ℝ → ℝ) (a b : ℝ) := ∫ x in set.Icc a b, F x

theorem work_done_example :
  work_done (λ x : ℝ, 3 * x^2 - 2 * x + 5) 5 10 = 825 :=
by
  sorry

end work_done_example_l676_676503


namespace hexagon_area_ratio_l676_676839

open Real

theorem hexagon_area_ratio (r s : ℝ) (h_eq_diam : s = r * sqrt 3) :
    (let a1 := (3 * sqrt 3 / 2) * ((3 * r / 4) ^ 2)
     let a2 := (3 * sqrt 3 / 2) * r^2
     a1 / a2 = 9 / 16) :=
by
  sorry

end hexagon_area_ratio_l676_676839


namespace polynomial_solution_l676_676791

theorem polynomial_solution (x : ℂ) (h : x^3 + x^2 + x + 1 = 0) : x^4 + 2 * x^3 + 2 * x^2 + 2 * x + 1 = 0 := 
by {
  sorry
}

end polynomial_solution_l676_676791


namespace fixed_second_intersection_point_of_circles_l676_676995

open_locale classical
noncomputable theory

variables {A B C D E : Type*} [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry E]

theorem fixed_second_intersection_point_of_circles
  (A B C : Point)
  (D : Point)
  (hD : D ∈ line_segment A B)
  (E : Point)
  (hE : E ∈ line_segment C D) :
  ∃ M : Circle,
    ∀ D' E' : Point,
    (D' ∈ line_segment A B) →
    (E' ∈ line_segment C D') →
    second_intersection_point (circumcircle A D' E') (circumcircle C B E') ∈ M := 
sorry

end fixed_second_intersection_point_of_circles_l676_676995


namespace reciprocal_of_expr_l676_676467

-- Definition of the expression to simplify
def expr := (1/3 : ℝ) + (1/4 : ℝ)

-- Statement to prove the reciprocal of the expression
theorem reciprocal_of_expr : (expr⁻¹) = (12/7 : ℝ) :=
by
  sorry

end reciprocal_of_expr_l676_676467


namespace distance_between_points_l676_676957

noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem distance_between_points :
  dist (3, 3) (-2, -2) = 5 * Real.sqrt 2 := 
by
  sorry

end distance_between_points_l676_676957


namespace distance_between_points_l676_676944

theorem distance_between_points :
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  (Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 * Real.sqrt 2) :=
by
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  sorry

end distance_between_points_l676_676944


namespace brick_surface_area_l676_676182

-- Defining the dimensions of the brick
def length := 10
def width := 4
def height := 2

-- Defining the formula for the surface area of a rectangular prism
def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

-- Statement: Prove that the surface area of the brick is 136 cm²
theorem brick_surface_area : surface_area length width height = 136 := by
  sorry

end brick_surface_area_l676_676182


namespace distance_between_points_l676_676952

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 3 3 (-2) (-2) = 5 * real.sqrt 2 :=
by
  sorry

end distance_between_points_l676_676952


namespace polynomial_divisibility_l676_676919

theorem polynomial_divisibility : 
  ∃ k : ℤ, (k = 8) ∧ (∀ x : ℂ, (4 * x^3 - 8 * x^2 + k * x - 16) % (x - 2) = 0) ∧ 
           (∀ x : ℂ, (4 * x^3 - 8 * x^2 + k * x - 16) % (x^2 + 1) = 0) :=
sorry

end polynomial_divisibility_l676_676919


namespace find_x_when_t_is_40_l676_676111

-- Define the initial conditions and the constant ratio
variable (x t : ℚ)
variable (k : ℚ) (hx : x = 4) (ht : t = 10) (hk : k = (5 * x - 6) / (t + 20))

-- Define the proposition to prove
theorem find_x_when_t_is_40 (h : t = 40) : x = 34 / 5 :=
by
  -- Given constant k from initial conditions
  have k_eq : k = 7 / 15 := by
    rw [hx, ht]
    calc
      k = (5 * 4 - 6) / (10 + 20) := hk
      _ = 14 / 30 := by norm_num
      _ = 7 / 15 := by norm_num
    
  -- Consider the value of x when t = 40 using the constant k
  have x_at_40 : (5 * x - 6) / (40 + 20) = 7 / 15 := by
    rw h
    exact k_eq

  -- Solve for x
  sorry

end find_x_when_t_is_40_l676_676111


namespace distance_points_l676_676932

-- Definition of distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Points
def point1 : ℝ × ℝ := (3, 3)
def point2 : ℝ × ℝ := (-2, -2)

-- Main theorem
theorem distance_points : distance point1 point2 = 5 * Real.sqrt 2 :=
by
  sorry

end distance_points_l676_676932


namespace num_possible_integers_l676_676278

theorem num_possible_integers (x : ℕ) (h : ⌈Real.sqrt x⌉ = 20) : ∃ n : ℕ, n = 39 :=
by
  have h1 : 19 < Real.sqrt x ∧ Real.sqrt x ≤ 20 := sorry
  have h2 : 361 < x ∧ x ≤ 400 := sorry
  have h3 : ∃ (a b : ℕ), 361 < a ∧ a ≤ 400 ∧ b = a - 361 ∧ b + 1 = 39 := sorry
  use 39
  exact h3.right.right
  sorry

end num_possible_integers_l676_676278


namespace conditional_probability_rain_given_east_wind_l676_676540

variable {Ω : Type*}
variable [ProbabilitySpace Ω]
variable (A B : Event Ω)

theorem conditional_probability_rain_given_east_wind
  (h_A : A.probability = 3/10)
  (h_B : B.probability = 11/30)
  (h_AB : (A ∩ B).probability = 8/30) :
  A.conditionalProbability B = 8/9 := 
sorry

end conditional_probability_rain_given_east_wind_l676_676540


namespace number_of_players_taking_mathematics_l676_676137

-- Define the conditions
def total_players := 15
def players_physics := 10
def players_both := 4

-- Define the conclusion to be proven
theorem number_of_players_taking_mathematics : (total_players - players_physics + players_both) = 9 :=
by
  -- Placeholder for proof
  sorry

end number_of_players_taking_mathematics_l676_676137


namespace point_outside_circle_l676_676237

theorem point_outside_circle (a b : ℝ) (h : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ ax + by = 1) :
  sqrt (a^2 + b^2) > 1 :=
sorry

end point_outside_circle_l676_676237


namespace find_cos_7theta_l676_676679

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end find_cos_7theta_l676_676679


namespace arrangement_count_l676_676908

-- Define the problem parameters
def males := ["M1", "M2", "M3"]
def females := ["F1", "F2", "F3"]

-- Define the condition that M1 should not be adjacent to M2 and M3
def not_adjacent (seating: List String) : Prop :=
  ∀ i, seating[i] = "M1" → (i > 0 ∧ seating[i-1] ≠ "M2" ∧ seating[i-1] ≠ "M3") ∧
        (i < seating.length-1 ∧ seating[i+1] ≠ "M2" ∧ seating[i+1] ≠ "M3")

-- Calculate the total number of valid arrangements
def calculate_arrangements : ℕ :=
  let case1 := 6 * 24 -- from the first case where no male students are adjacent
  let case2 := 2 * 6 * 12 -- from the second case where two male students are adjacent
  case1 + case2

theorem arrangement_count : calculate_arrangements = 288 :=
by
  sorry -- Skipping the proof as instructed

end arrangement_count_l676_676908


namespace sequence_sum_eq_zero_l676_676988

variable {n : ℕ}
variable {a : ℕ → ℝ}
variables (a_neq : ∀ i : ℕ, a (i+1) ≠ a (i-1))
variable (a_relation : ∀ i : ℕ, a (i+1)^2 - a i * a (i+1) + a i^2 = 0)
variable (a1 : a 1 = 1)
variable (anp1 : a (n+1) = 1)
variable (non_constant : ∃ i : ℕ, a (i) ≠ a (i+1))

theorem sequence_sum_eq_zero (h1 : a_relation) (h2 : a_neq) (h3 : a1) (h4 : anp1) (h5 : non_constant) : 
  ∑ i in Finset.range n, a i = 0 :=
by
  sorry

end sequence_sum_eq_zero_l676_676988


namespace range_of_a_l676_676707

theorem range_of_a (h : ∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) : a > -1 :=
sorry

end range_of_a_l676_676707


namespace max_profit_l676_676523

def profit_function (x : ℕ) := -200 * x + 140000

theorem max_profit (x : ℕ) : 
  (100 - x ≤ 3 * x) ∧ (1200 * x + 1400 * (100 - x)) = 135000 ∧ x = 25 :=
begin
  sorry
end

end max_profit_l676_676523


namespace number_of_teachers_students_possible_rental_plans_economical_plan_l676_676095

-- Definitions of conditions

def condition1 (x y : ℕ) : Prop := y - 30 * x = 7
def condition2 (x y : ℕ) : Prop := 31 * x - y = 1
def capacity_condition (m : ℕ) : Prop := 35 * m + 30 * (8 - m) ≥ 255
def rental_fee_condition (m : ℕ) : Prop := 400 * m + 320 * (8 - m) ≤ 3000

-- Problems to prove

theorem number_of_teachers_students (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 8 ∧ y = 247 := 
by sorry

theorem possible_rental_plans (m : ℕ) (h_cap : capacity_condition m) (h_fee : rental_fee_condition m) : m = 3 ∨ m = 4 ∨ m = 5 := 
by sorry

theorem economical_plan (m : ℕ) (h_fee : rental_fee_condition 3) (h_fee_alt1 : rental_fee_condition 4) (h_fee_alt2 : rental_fee_condition 5) : m = 3 := 
by sorry

end number_of_teachers_students_possible_rental_plans_economical_plan_l676_676095


namespace odd_numbers_count_l676_676102

def is_digit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6

def digits_unique (d1 d2 d3 d4 d5 : ℕ) : Prop :=
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧
  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧
  d3 ≠ d4 ∧ d3 ≠ d5 ∧
  d4 ≠ d5

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def adjacent (d1 d2 : ℕ) (list : List ℕ) : Prop :=
  List.indexOf d1 list + 1 = List.indexOf d2 list ∨ List.indexOf d2 list + 1 = List.indexOf d1 list

def valid_number (d1 d2 d3 d4 d5 : ℕ) : Prop :=
  is_digit d1 ∧ is_digit d2 ∧ is_digit d3 ∧ is_digit d4 ∧ is_digit d5 ∧
  digits_unique d1 d2 d3 d4 d5 ∧
  is_odd d1 ∧
  adjacent 5 6 [d5, d4, d3, d2, d1]

theorem odd_numbers_count : 
  ∃ (f : ℕ × ℕ × ℕ × ℕ × ℕ → Prop) (n : ℕ),
  (∀ x, f x → valid_number x.1 x.2 x.3 x.4 x.5) ∧ 
  (finset.filter f (finset.univ : finset (ℕ × ℕ × ℕ × ℕ × ℕ))).card = 14 :=
by
  sorry

end odd_numbers_count_l676_676102


namespace smallest_positive_period_pi_cos_2x0_value_l676_676244

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + 2 * (cos x) ^ 2 - 1

theorem smallest_positive_period_pi :
  ∃ p > 0, ∀ x : ℝ, f (x + p) = f x ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ (π / 2) → f x ≤ 2 ∧ f x ≥ -1) := 
sorry

theorem cos_2x0_value (x0 : ℝ) (h1 : f x0 = 6 / 5) (h2 : π / 4 ≤ x0 ∧ x0 ≤ π / 2) :
  cos (2 * x0) = (3 - 4 * sqrt 3) / 10 :=
sorry

end smallest_positive_period_pi_cos_2x0_value_l676_676244


namespace prove_inequality_equality_case_l676_676366

variables {n : ℕ}
variables {a b z : Fin n → ℝ} {λa λb λz : ℝ}

noncomputable def given_conditions (a b z : Fin n → ℝ) (λa λb λz : ℝ) : Prop :=
  (∀ i, 0 ≤ a i ∧ 0 ≤ b i ∧ 0 ≤ z i) ∧
  λa > 0 ∧ λb > 0 ∧ λz > 0 ∧
  λa + λb + λz = 1

theorem prove_inequality (a b z : Fin n → ℝ) (λa λb λz : ℝ) 
  (h_cond : given_conditions a b z λa λb λz) :
  (∑ i, (a i)^λa * (b i)^λb * (z i)^λz) ≤ 
  (∑ i, a i)^λa * (∑ i, b i)^λb * (∑ i, z i)^λz :=
sorry

theorem equality_case (a b z : Fin n → ℝ) (λa λb λz : ℝ) 
  (h_cond : given_conditions a b z λa λb λz) :
  (∑ i, (a i)^λa * (b i)^λb * (z i)^λz = 
  (∑ i, a i)^λa * (∑ i, b i)^λb * (∑ i, z i)^λz) ↔
  ∀ i, (a i) / (∑ j, a j) = (b i) / (∑ j, b j) ∧ (b i) / (∑ j, b j) = (z i) / (∑ j, z j) :=
sorry

end prove_inequality_equality_case_l676_676366


namespace solve_fractional_eq_l676_676441

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end solve_fractional_eq_l676_676441


namespace shaded_area_BEDC_l676_676887

-- Defining the conditions as per part (a)
variables (ABCD E : Type) [parallelogram ABCD] [point E]

def area_PAR_150 (a : ℝ) := parallelogram.area ABCD = 150
def height_AE (AE : ℝ) := extends_height A E
def proportion_ED (x : ℝ) := ED = (2 / 3) * CD

-- Theorem statement
theorem shaded_area_BEDC :
  ∀ (a x h : ℝ), area_PAR_150 a -> proportion_ED x -> height_AE h -> 
  shaded_area BEDC = 125 :=
sorry

end shaded_area_BEDC_l676_676887


namespace distance_between_points_l676_676950

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 3 3 (-2) (-2) = 5 * real.sqrt 2 :=
by
  sorry

end distance_between_points_l676_676950


namespace problem_part1_problem_part2_l676_676657

def vec_m (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (x / 4), 1)
def vec_n (x : ℝ) : ℝ × ℝ := (cos (x / 4), cos (x / 4) ^ 2)
def f (x : ℝ) : ℝ := vec_m x.1.1 * vec_n x.2.1 + vec_m x.1.2 * vec_n x.2.2

theorem problem_part1 (x : ℝ) (hfx : f x = 1) : cos (x + π / 3) = 1 / 2 := sorry

variable {a b c A B C : ℝ}
variable (hABC_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
variable (hconstr_trig : (2 * a - c) * cos B = b * cos C)

theorem problem_part2 : 
  let f2A := λ A, (sin (A + π / 6) + 1 / 2) 
  in  sqrt 3 / 2 < sin (A + π / 6) ∧ sin (A + π / 6) ≤ 1 →
      (sqrt 3 + 1) / 2 < f2A 2 * A ∧ f2A 2 * A ≤ 3 / 2 := sorry

end problem_part1_problem_part2_l676_676657


namespace volume_pyramid_proof_l676_676727

-- Definitions based on the problem conditions
variables {α : Type*} [inner_product_space ℝ α]
variables (A B C M : α)
variables (CA B30 ACM_folding : Prop)

-- Translating the conditions to definitions
def right_triangle (A B C : α) : Prop :=
  ∠ C = 90 ∧ ∠ B = 30 ∧ dist A C = 2

def midpoint (M : α) (A B : α) : Prop :=
  dist A M = dist M B

def distance_AB_folding (A B : α) : Prop :=
  dist A B = 2 * sqrt 2

-- Defining the volume of the triangular prism
def volume_pyramid (A B C M : α) : ℝ :=
  volume_of_tetrahedron A B C M

-- Final proof statement
theorem volume_pyramid_proof 
  (h1 : right_triangle A B C)
  (h2 : midpoint M A B)
  (h3 : distance_AB_folding A B) :
  volume_pyramid A B C M = sqrt 2 / 3 :=
  sorry

end volume_pyramid_proof_l676_676727


namespace triangle_probability_l676_676917

theorem triangle_probability :
  let region := { p : ℝ × ℝ × ℝ | 
    0 ≤ p.1 ∧ 0 ≤ p.2 ∧ 0 ≤ p.3 ∧ 
    p.1 + p.2 + p.3 = 1
  }
  let event_A := { p : ℝ × ℝ × ℝ | 
    0 ≤ p.1 ∧ 0 ≤ p.2 ∧ 0 ≤ p.3 ∧ 
    p.1 + p.2 + p.3 = 1 ∧ 
    p.1 + p.2 > p.3 ∧ 
    p.1 + p.3 > p.2 ∧ 
    p.2 + p.3 > p.1
  }
  in (measure_theory.volume event_A / measure_theory.volume region) = 1 / 4 := 
sorry

end triangle_probability_l676_676917


namespace combined_age_l676_676552

-- Conditions as definitions
def AmyAge (j : ℕ) : ℕ :=
  j / 3

def ChrisAge (a : ℕ) : ℕ :=
  2 * a

-- Given condition
def JeremyAge : ℕ := 66

-- Question to prove
theorem combined_age : 
  let j := JeremyAge
  let a := AmyAge j
  let c := ChrisAge a
  a + j + c = 132 :=
by
  sorry

end combined_age_l676_676552


namespace goods_train_length_l676_676517

noncomputable def speed_in_m_per_s (speed_km_per_hr : ℝ) : ℝ :=
  (speed_km_per_hr * 1000) / 3600

theorem goods_train_length (speed_km_per_hr : ℝ) (platform_length_m : ℝ) (time_sec : ℝ) :
  speed_km_per_hr = 60 → platform_length_m = 300 → time_sec = 35 →
  let speed_m_per_s := speed_in_m_per_s speed_km_per_hr in
  let train_length_m := (speed_m_per_s * time_sec) - platform_length_m in
  train_length_m = 283.45 :=
by
  intros h1 h2 h3
  unfold speed_in_m_per_s
  simp [h1, h2, h3]
  norm_num
  sorry

end goods_train_length_l676_676517


namespace incircle_inequality_proof_l676_676714

-- Definitions and conditions
variables {A B C C1 B1 A1 : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space C1]
variables [metric_space B1] [metric_space A1]

-- Distances between points A, B, C and the corresponding touchpoints
variables (AB AC BC AB1 AC1 BC1 : ℝ)

-- Semiperimeter of the triangle
noncomputable def s : ℝ := (AB + AC + BC) / 2

-- Given conditions
axiom incircle_touches_sides : ∀ {A B C A1 B1 C1}, 
  (AB1 = s - BC) ∧ (BC1 = s - AC) ∧ (AC1 = s - AB)

-- Function to be proven
noncomputable def incircle_inequality (AB1 AB AC1 BC1: ℝ) : Prop :=
  sqrt (AB1 / AB) + sqrt (BC1 / BC) + sqrt (AC1 / AC) ≤ 3 / sqrt(2)

-- Main theorem
theorem incircle_inequality_proof
  (A B C A1 B1 C1 : Type) [metric_space A] [metric_space B] [metric_space C] 
  [metric_space A1] [metric_space B1] [metric_space C1] 
  (AB AC BC AB1 AC1 BC1 : ℝ) (h : ∀ {A B C A1 B1 C1}, 
  (AB1 = s - BC) ∧ (BC1 = s - AC) ∧ (AC1 = s - AB)) :
  incircle_inequality AB1 AC1 BC1 :=
by sorry

end incircle_inequality_proof_l676_676714


namespace cake_has_more_calories_l676_676091

-- Define the conditions
def cake_slices : Nat := 8
def cake_calories_per_slice : Nat := 347
def brownie_count : Nat := 6
def brownie_calories_per_brownie : Nat := 375

-- Define the total calories for the cake and the brownies
def total_cake_calories : Nat := cake_slices * cake_calories_per_slice
def total_brownie_calories : Nat := brownie_count * brownie_calories_per_brownie

-- Prove the difference in calories
theorem cake_has_more_calories : 
  total_cake_calories - total_brownie_calories = 526 :=
by
  sorry

end cake_has_more_calories_l676_676091


namespace ellipse_equation_l676_676605

theorem ellipse_equation (focus_shared: ℝ) (point_A: ℝ × ℝ) 
                       (h1 : focus_shared = real.sqrt (4 - 1))
                       (h2 : point_A = (2, 1)) : 
  (∃ a b : ℝ, a = 6 ∧ b = 3 ∧ ∀ x y : ℝ, (x^2 / a + y^2 / b = 1)) :=
sorry

end ellipse_equation_l676_676605


namespace jesse_money_left_after_mall_l676_676334

theorem jesse_money_left_after_mall :
  ∀ (initial_amount novel_cost lunch_cost total_spent remaining_amount : ℕ),
    initial_amount = 50 →
    novel_cost = 7 →
    lunch_cost = 2 * novel_cost →
    total_spent = novel_cost + lunch_cost →
    remaining_amount = initial_amount - total_spent →
    remaining_amount = 29 :=
by
  intros initial_amount novel_cost lunch_cost total_spent remaining_amount
  sorry

end jesse_money_left_after_mall_l676_676334


namespace sum_first_n_terms_sequence_identity_l676_676813

theorem sum_first_n_terms_sequence_identity (n : ℕ) (h : 0 < n) :
  let seq := λ k, (finset.range k).sum (λ p, 2^p)
  let S_n := (finset.range n).sum seq
  S_n = 2^(n+1) - n - 2 :=
by sorry

end sum_first_n_terms_sequence_identity_l676_676813


namespace integer_values_x_l676_676287

theorem integer_values_x {x : ℝ} (h : ⌈real.sqrt x⌉ = 20) : ∃ n : ℕ, n = 39 :=
by
  have h1 : 19 < real.sqrt x ∧ real.sqrt x ≤ 20 := sorry
  have h2 : 361 < x ∧ x ≤ 400 := sorry
  have x_values : set.Icc 362 400 := sorry
  have n_values : ∃ n : ℕ, n = set.finite.to_finset x_values.card := sorry
  exact n_values

end integer_values_x_l676_676287


namespace additional_plates_is_50_l676_676725

def initial_plates : Nat :=
  5 * 2 * 3

def new_sets_addition : Nat :=
  5 * 4 * 4

def additional_plates : Nat :=
  new_sets_addition - initial_plates

theorem additional_plates_is_50 : additional_plates = 50 :=
by
  calc
    additional_plates
        = new_sets_addition - initial_plates : by refl
    ... = 80 - 30 : by refl
    ... = 50 : by refl

end additional_plates_is_50_l676_676725


namespace solve_fraction_equations_l676_676171

theorem solve_fraction_equations :
  ∀ x : ℝ,
    (1/((x - 1)*(x - 2)) + 1/((x - 2)*(x - 3)) + 1/((x - 3)*(x - 4)) + 1/((x - 4)*(x - 5)) = 1/12) →
    (x = 12 ∨ x = -4.5) :=
by {
  intro x,
  intro h,
  sorry,
}

end solve_fraction_equations_l676_676171


namespace complex_power_six_l676_676573

theorem complex_power_six :
  (1 + complex.i)^6 = -8 * complex.i :=
by sorry

end complex_power_six_l676_676573


namespace exists_smallest_h_l676_676968

noncomputable def smallest_h (n : ℕ) :=
  Inf { m : ℕ | ∀ (A : finset (fin m) → ℕ), ∃ a b x y, x ≤ y ∧ x + y ≤ m ∧
    (a + b ∈ A ∧ b + x ∈ A ∧ b + y ∈ A) }

theorem exists_smallest_h (n : ℕ) : ∃ h : ℕ, smallest_h n = h :=
begin
  sorry
end

end exists_smallest_h_l676_676968


namespace f_identity_l676_676748

def f (x : ℝ) : ℝ := (2 * x + 1)^5 - 5 * (2 * x + 1)^4 + 10 * (2 * x + 1)^3 - 10 * (2 * x + 1)^2 + 5 * (2 * x + 1) - 1

theorem f_identity (x : ℝ) : f x = 32 * x^5 :=
by
  -- the proof is omitted
  sorry

end f_identity_l676_676748
