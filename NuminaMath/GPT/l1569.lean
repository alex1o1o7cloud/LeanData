import Mathlib

namespace quad_area_FDBG_l1569_156999

open Real

noncomputable def area_quad_FDBG (AB AC area_ABC : ℝ) : ℝ :=
  let AD := AB / 2
  let AE := AC / 2
  let area_ADE := area_ABC / 4
  let x := 2 * area_ABC / (AB * AC)
  let sin_A := x
  let hyp_ratio := sin_A / (area_ABC / AC)
  let factor := hyp_ratio / 2
  let area_AFG := factor * area_ADE
  area_ABC - area_ADE - 2 * area_AFG

theorem quad_area_FDBG (AB AC area_ABC : ℝ) (hAB : AB = 60) (hAC : AC = 15) (harea : area_ABC = 180) :
  area_quad_FDBG AB AC area_ABC = 117 := by
  sorry

end quad_area_FDBG_l1569_156999


namespace no_six_coins_sum_70_cents_l1569_156969

theorem no_six_coins_sum_70_cents :
  ¬ ∃ (p n d q : ℕ), p + n + d + q = 6 ∧ p + 5 * n + 10 * d + 25 * q = 70 :=
by
  sorry

end no_six_coins_sum_70_cents_l1569_156969


namespace pencils_undefined_l1569_156931

-- Definitions for the conditions given in the problem
def initial_crayons : Nat := 41
def added_crayons : Nat := 12
def total_crayons : Nat := 53

-- Theorem stating the problem's required proof
theorem pencils_undefined (initial_crayons : Nat) (added_crayons : Nat) (total_crayons : Nat) : Prop :=
  initial_crayons = 41 ∧ added_crayons = 12 ∧ total_crayons = 53 → 
  ∃ (pencils : Nat), true
-- Since the number of pencils is unknown and no direct information is given, we represent it as an existential statement that pencils exist in some quantity, but we cannot determine their exact number based on given information.

end pencils_undefined_l1569_156931


namespace adjacent_irreducible_rationals_condition_l1569_156976

theorem adjacent_irreducible_rationals_condition 
  (a b c d : ℕ) 
  (hab_cop : Nat.gcd a b = 1) (hcd_cop : Nat.gcd c d = 1) 
  (h_ab_prod : a * b < 1988) (h_cd_prod : c * d < 1988) 
  (adj : ∀ p q r s, (Nat.gcd p q = 1) → (Nat.gcd r s = 1) → 
                  (p * q < 1988) → (r * s < 1988) →
                  (p / q < r / s) → (p * s - q * r = 1)) : 
  b * c - a * d = 1 :=
sorry

end adjacent_irreducible_rationals_condition_l1569_156976


namespace adam_action_figures_per_shelf_l1569_156946

-- Define the number of shelves and the total number of action figures
def shelves : ℕ := 4
def total_action_figures : ℕ := 44

-- Define the number of action figures per shelf
def action_figures_per_shelf : ℕ := total_action_figures / shelves

-- State the theorem to be proven
theorem adam_action_figures_per_shelf : action_figures_per_shelf = 11 :=
by sorry

end adam_action_figures_per_shelf_l1569_156946


namespace xy_sum_is_2_l1569_156918

theorem xy_sum_is_2 (x y : ℝ) 
  (h1 : (x - 1) ^ 3 + 1997 * (x - 1) = -1)
  (h2 : (y - 1) ^ 3 + 1997 * (y - 1) = 1) : 
  x + y = 2 := 
  sorry

end xy_sum_is_2_l1569_156918


namespace smallest_n_l1569_156989

theorem smallest_n (n : ℕ) (h1 : 1826 % 26 = 6) (h2 : 5 * n % 26 = 6) : n = 20 :=
sorry

end smallest_n_l1569_156989


namespace arrange_2015_integers_l1569_156941

theorem arrange_2015_integers :
  ∃ (f : Fin 2015 → Fin 2015),
    (∀ i, (Nat.gcd ((f i).val + (f (i + 1)).val) 4 = 1 ∨ Nat.gcd ((f i).val + (f (i + 1)).val) 7 = 1)) ∧
    Function.Injective f ∧ 
    (∀ i, 1 ≤ (f i).val ∧ (f i).val ≤ 2015) :=
sorry

end arrange_2015_integers_l1569_156941


namespace solve_inequality_l1569_156902

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 1 / x else (1 / 3) ^ x

theorem solve_inequality : { x : ℝ | |f x| ≥ 1 / 3 } = { x : ℝ | -3 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end solve_inequality_l1569_156902


namespace diego_payment_l1569_156958

theorem diego_payment (d : ℤ) (celina : ℤ) (total : ℤ) (h₁ : celina = 1000 + 4 * d) (h₂ : total = celina + d) (h₃ : total = 50000) : d = 9800 :=
sorry

end diego_payment_l1569_156958


namespace abcd_inequality_l1569_156951

theorem abcd_inequality (a b c d : ℝ) :
  (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) :=
sorry

end abcd_inequality_l1569_156951


namespace find_m_n_sum_l1569_156917

theorem find_m_n_sum (n m : ℝ) (d : ℝ) 
(h1 : ∀ x y, 2*x + y + n = 0) 
(h2 : ∀ x y, 4*x + m*y - 4 = 0) 
(hd : d = (3/5) * Real.sqrt 5) 
: m + n = -3 ∨ m + n = 3 :=
sorry

end find_m_n_sum_l1569_156917


namespace grade_assignment_ways_l1569_156961

/-- Define the number of students and the number of grade choices -/
def num_students : ℕ := 15
def num_grades : ℕ := 4

/-- Define the total number of ways to assign grades -/
def total_ways : ℕ := num_grades ^ num_students

/-- Prove that the total number of ways to assign grades is 4^15 -/
theorem grade_assignment_ways : total_ways = 1073741824 := by
  -- proof here
  sorry

end grade_assignment_ways_l1569_156961


namespace sum_of_digits_of_N_l1569_156993

theorem sum_of_digits_of_N :
  (∃ N : ℕ, 3 * N * (N + 1) / 2 = 3825 ∧ (N.digits 10).sum = 5) :=
by
  sorry

end sum_of_digits_of_N_l1569_156993


namespace complex_number_quadrant_l1569_156945

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l1569_156945


namespace hyperbola_eccentricity_l1569_156955

theorem hyperbola_eccentricity 
  (a b : ℝ) (h1 : 2 * (1 : ℝ) + 1 = 0) (h2 : 0 < a) (h3 : 0 < b) 
  (h4 : b = 2 * a) : 
  (∃ e : ℝ, e = (Real.sqrt 5)) 
:= 
  sorry

end hyperbola_eccentricity_l1569_156955


namespace number_of_pairs_l1569_156930

theorem number_of_pairs (H : ∀ x y : ℕ , 0 < x → 0 < y → x < y → 2 * x * y / (x + y) = 4 ^ 15) :
  ∃ n : ℕ, n = 29 :=
by
  sorry

end number_of_pairs_l1569_156930


namespace any_nat_in_frac_l1569_156907

theorem any_nat_in_frac (n : ℕ) : ∃ x y : ℕ, y ≠ 0 ∧ x^2 = y^3 * n := by
  sorry

end any_nat_in_frac_l1569_156907


namespace problem_statement_l1569_156966

variable {R : Type*} [LinearOrderedField R]

def is_even_function (f : R → R) : Prop := ∀ x : R, f x = f (-x)

theorem problem_statement (f : R → R)
  (h1 : is_even_function f)
  (h2 : ∀ x1 x2 : R, x1 ≤ -1 → x2 ≤ -1 → (x2 - x1) * (f x2 - f x1) < 0) :
  f (-1) < f (-3 / 2) ∧ f (-3 / 2) < f 2 :=
sorry

end problem_statement_l1569_156966


namespace multiplication_schemes_correct_l1569_156974

theorem multiplication_schemes_correct :
  ∃ A B C D E F G H I K L M N P : ℕ,
    A = 7 ∧ B = 7 ∧ C = 4 ∧ D = 4 ∧ E = 3 ∧ F = 0 ∧ G = 8 ∧ H = 3 ∧ I = 3 ∧ K = 8 ∧ L = 8 ∧ M = 0 ∧ N = 7 ∧ P = 7 ∧
    (A * 10 + B) * (C * 10 + D) * (A * 10 + B) = E * 100 + F * 10 + G ∧
    (C * 10 + G) * (K * 10 + L) = A * 100 + M * 10 + C ∧
    E * 100 + F * 10 + G / (H * 1000 + I * 100 + G * 10 + G) = (E * 100 + F * 10 + G) / (H * 1000 + I * 100 + G * 10 + G) ∧
    (A * 100 + M * 10 + C) / (N * 1000 + P * 100 + C * 10 + C) = (A * 100 + M * 10 + C) / (N * 1000 + P * 100 + C * 10 + C) :=
sorry

end multiplication_schemes_correct_l1569_156974


namespace angle_x_in_triangle_l1569_156980

theorem angle_x_in_triangle :
  ∀ (x : ℝ), x + 2 * x + 50 = 180 → x = 130 / 3 :=
by
  intro x h
  sorry

end angle_x_in_triangle_l1569_156980


namespace general_term_of_sequence_l1569_156998

def A := {n : ℕ | ∃ k : ℕ, k + 1 = n }
def B := {m : ℕ | ∃ k : ℕ, 3 * k - 1 = m }

theorem general_term_of_sequence (k : ℕ) : 
  ∃ a_k : ℕ, a_k ∈ A ∩ B ∧ a_k = 9 * k^2 - 9 * k + 2 :=
sorry

end general_term_of_sequence_l1569_156998


namespace gcd_1734_816_1343_l1569_156968

theorem gcd_1734_816_1343 : Int.gcd (Int.gcd 1734 816) 1343 = 17 :=
by
  sorry

end gcd_1734_816_1343_l1569_156968


namespace intersection_A_B_l1569_156975

def setA : Set ℤ := { x | x < -3 }
def setB : Set ℤ := {-5, -4, -3, 1}

theorem intersection_A_B : setA ∩ setB = {-5, -4} := by
  sorry

end intersection_A_B_l1569_156975


namespace arithmetic_sequence_properties_l1569_156925

variable {a : ℕ → ℕ}
variable {n : ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

theorem arithmetic_sequence_properties 
  (a_3_eq_7 : a 3 = 7)
  (a_5_plus_a_7_eq_26 : a 5 + a 7 = 26) :
  (∃ a1 d, (a 1 = a1) ∧ (∀ n, a n = a1 + (n - 1) * d) ∧ d = 2) ∧
  (∀ n, a n = 2 * n + 1) ∧
  (∀ S_n, S_n = n^2 + 2 * n) ∧ 
  ∀ T_n n, (∃ b : (ℕ → ℕ) → ℕ → ℕ, b a n = 1 / (a n ^ 2 - 1)) 
  → T_n = n / (4 * (n + 1)) :=
by
  sorry

end arithmetic_sequence_properties_l1569_156925


namespace candy_division_l1569_156971

theorem candy_division (pieces_of_candy : Nat) (students : Nat) 
  (h1 : pieces_of_candy = 344) (h2 : students = 43) : pieces_of_candy / students = 8 := by
  sorry

end candy_division_l1569_156971


namespace inequality_solution_l1569_156985

theorem inequality_solution :
  { x : ℝ | 0 < x ∧ x ≤ 7/3 ∨ 3 ≤ x } = { x : ℝ | (0 < x ∧ x ≤ 7/3) ∨ 3 ≤ x } :=
sorry

end inequality_solution_l1569_156985


namespace symmetric_line_equation_l1569_156967

theorem symmetric_line_equation (x y : ℝ) :
  (2 : ℝ) * (2 - x) + (3 : ℝ) * (-2 - y) - 6 = 0 → 2 * x + 3 * y + 8 = 0 :=
by
  sorry

end symmetric_line_equation_l1569_156967


namespace fish_problem_l1569_156914

theorem fish_problem : 
  ∀ (B T S : ℕ), 
    B = 10 → 
    T = 3 * B → 
    S = 35 → 
    B + T + S + 2 * S = 145 → 
    S - T = 5 :=
by sorry

end fish_problem_l1569_156914


namespace initial_average_weight_l1569_156979

theorem initial_average_weight
  (A : ℚ) -- Define A as a rational number since we are dealing with division 
  (h1 : 6 * A + 133 = 7 * 151) : -- Condition from the problem translated into an equation
  A = 154 := -- Statement we need to prove
by
  sorry -- Placeholder for the proof

end initial_average_weight_l1569_156979


namespace weekly_salary_correct_l1569_156919

-- Define the daily salaries for each type of worker
def salary_A : ℝ := 200
def salary_B : ℝ := 250
def salary_C : ℝ := 300
def salary_D : ℝ := 350

-- Define the number of each type of worker
def num_A : ℕ := 3
def num_B : ℕ := 2
def num_C : ℕ := 3
def num_D : ℕ := 1

-- Define the total hours worked per day and the number of working days in a week
def hours_per_day : ℕ := 6
def working_days : ℕ := 7

-- Calculate the total daily salary for the team
def daily_salary_team : ℝ :=
  (num_A * salary_A) + (num_B * salary_B) + (num_C * salary_C) + (num_D * salary_D)

-- Calculate the total weekly salary for the team
def weekly_salary_team : ℝ := daily_salary_team * working_days

-- Problem: Prove that the total weekly salary for the team is Rs. 16,450
theorem weekly_salary_correct : weekly_salary_team = 16450 := by
  sorry

end weekly_salary_correct_l1569_156919


namespace y_plus_inv_l1569_156901

theorem y_plus_inv (y : ℝ) (h : y^3 + 1/y^3 = 110) : y + 1/y = 5 := 
by 
sorry

end y_plus_inv_l1569_156901


namespace geometric_ratio_l1569_156981

noncomputable def S (n : ℕ) : ℝ := sorry  -- Let's assume S is a function that returns the sum of the first n terms of the geometric sequence.

-- Conditions
axiom S_10_eq_S_5 : S 10 = 2 * S 5

-- Definition to be proved
theorem geometric_ratio :
  (S 5 + S 10 + S 15) / (S 10 - S 5) = -9 / 2 :=
sorry

end geometric_ratio_l1569_156981


namespace total_food_per_day_l1569_156954

theorem total_food_per_day :
  let num_puppies := 4
  let num_dogs := 3
  let dog_meal_weight := 4
  let dog_meals_per_day := 3
  let dog_food_per_day := dog_meal_weight * dog_meals_per_day
  let total_dog_food_per_day := dog_food_per_day * num_dogs
  let puppy_meal_weight := dog_meal_weight / 2
  let puppy_meals_per_day := dog_meals_per_day * 3
  let puppy_food_per_day := puppy_meal_weight * puppy_meals_per_day
  let total_puppy_food_per_day := puppy_food_per_day * num_puppies
  total_dog_food_per_day + total_puppy_food_per_day = 108 :=
by
  sorry

end total_food_per_day_l1569_156954


namespace parabola_line_intersection_l1569_156970

/-- 
Given a parabola \( y^2 = 2x \), a line passing through the focus of 
the parabola intersects the parabola at points \( A \) and \( B \) where 
the sum of the x-coordinates of \( A \) and \( B \) is equal to 2. 
Prove that such a line exists and there are exactly 3 such lines.
--/
theorem parabola_line_intersection :
  ∃ l₁ l₂ l₃ : (ℝ × ℝ) → (ℝ × ℝ), 
    (∀ p, l₁ p = l₂ p ∧ l₁ p = l₃ p → false) ∧
    ∀ (A B : ℝ × ℝ), 
      (A.2 ^ 2 = 2 * A.1) ∧ 
      (B.2 ^ 2 = 2 * B.1) ∧ 
      (A.1 + B.1 = 2) →
      (∃ k : ℝ, 
        ∀ (x : ℝ), 
          ((A.2 = k * (A.1 - 1)) ∧ (B.2 = k * (B.1 - 1))) ∧ 
          (k * (A.1 - 1) = k * (B.1 - 1)) ∧ 
          (k ≠ 0)) :=
sorry

end parabola_line_intersection_l1569_156970


namespace ratio_of_areas_two_adjacent_triangles_to_one_triangle_l1569_156921

-- Definition of a regular hexagon divided into six equal triangles
def is_regular_hexagon_divided_into_six_equal_triangles (s : ℝ) : Prop :=
  s > 0 -- s is the area of one of the six triangles and must be positive

-- Definition of the area of a region formed by two adjacent triangles
def area_of_two_adjacent_triangles (s r : ℝ) : Prop :=
  r = 2 * s

-- The proof problem statement
theorem ratio_of_areas_two_adjacent_triangles_to_one_triangle (s r : ℝ)
  (hs : is_regular_hexagon_divided_into_six_equal_triangles s)
  (hr : area_of_two_adjacent_triangles s r) : 
  r / s = 2 :=
by
  sorry

end ratio_of_areas_two_adjacent_triangles_to_one_triangle_l1569_156921


namespace triangle_area_l1569_156922

theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (h_perimeter : perimeter = 40) (h_inradius : inradius = 2.5) : 
  (inradius * (perimeter / 2)) = 50 :=
by
  -- Lean 4 statement code
  sorry

end triangle_area_l1569_156922


namespace find_f_pi_six_value_l1569_156995

noncomputable def f (x : ℝ) (f'₀ : ℝ) : ℝ := f'₀ * Real.sin x + Real.cos x

theorem find_f_pi_six_value (f'₀ : ℝ) (h : f'₀ = 2 + Real.sqrt 3) : f (π / 6) f'₀ = 1 + Real.sqrt 3 := 
by
  -- condition from the problem
  let f₀ := f (π / 6) f'₀
  -- final goal to prove
  sorry

end find_f_pi_six_value_l1569_156995


namespace ratio_of_ages_l1569_156935

theorem ratio_of_ages
  (Sandy_age : ℕ)
  (Molly_age : ℕ)
  (h1 : Sandy_age = 49)
  (h2 : Molly_age = Sandy_age + 14) : (Sandy_age : ℚ) / Molly_age = 7 / 9 :=
by
  -- To complete the proof.
  sorry

end ratio_of_ages_l1569_156935


namespace gamma_minus_alpha_l1569_156927

theorem gamma_minus_alpha (α β γ : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < γ) (h4 : γ < 2 * Real.pi)
    (h5 : ∀ x : ℝ, Real.cos (x + α) + Real.cos (x + β) + Real.cos (x + γ) = 0) : 
    γ - α = (4 * Real.pi) / 3 :=
sorry

end gamma_minus_alpha_l1569_156927


namespace perfect_square_trinomial_l1569_156939

theorem perfect_square_trinomial (k : ℤ) : (∃ a : ℤ, (x : ℤ) → x^2 - k * x + 9 = (x - a)^2) → (k = 6 ∨ k = -6) :=
sorry

end perfect_square_trinomial_l1569_156939


namespace monotonicity_f_max_value_f_l1569_156926

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x - 1

theorem monotonicity_f :
  (∀ x, 0 < x ∧ x < Real.exp 1 → f x < f (Real.exp 1)) ∧
  (∀ x, x > Real.exp 1 → f x < f (Real.exp 1)) :=
sorry

theorem max_value_f (m : ℝ) (hm : m > 0) :
  (2 * m ≤ Real.exp 1 → ∃ x ∈ Set.Icc m (2 * m), f x = (Real.log (2 * m)) / (2 * m) - 1) ∧
  (m ≥ Real.exp 1 → ∃ x ∈ Set.Icc m (2 * m), f x = (Real.log m) / m - 1) ∧
  (Real.exp 1 / 2 < m ∧ m < Real.exp 1 → ∃ x ∈ Set.Icc m (2 * m), f x = 1 / Real.exp 1 - 1) :=
sorry

end monotonicity_f_max_value_f_l1569_156926


namespace find_y_z_l1569_156908

theorem find_y_z 
  (y z : ℝ) 
  (h_mean : (8 + 15 + 22 + 5 + y + z) / 6 = 12) 
  (h_diff : y - z = 6) : 
  y = 14 ∧ z = 8 := 
by
  sorry

end find_y_z_l1569_156908


namespace eval_expression_l1569_156994

theorem eval_expression :
  72 + (120 / 15) + (18 * 19) - 250 - (360 / 6) = 112 :=
by sorry

end eval_expression_l1569_156994


namespace parabola_focus_l1569_156957

theorem parabola_focus (x y : ℝ) (h : y^2 = 8 * x) : (x, y) = (2, 0) :=
sorry

end parabola_focus_l1569_156957


namespace expression_evaluation_l1569_156950

theorem expression_evaluation (a b c : ℤ) 
  (h1 : c = a + 8) 
  (h2 : b = a + 4) 
  (h3 : a = 5) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 23/15 :=
by
  sorry

end expression_evaluation_l1569_156950


namespace trains_cross_time_l1569_156956

theorem trains_cross_time
  (length_each_train : ℝ)
  (speed_each_train_kmh : ℝ)
  (relative_speed_m_s : ℝ)
  (total_distance : ℝ)
  (conversion_factor : ℝ) :
  length_each_train = 120 →
  speed_each_train_kmh = 27 →
  conversion_factor = 1000 / 3600 →
  relative_speed_m_s = speed_each_train_kmh * conversion_factor →
  total_distance = 2 * length_each_train →
  total_distance / relative_speed_m_s = 16 :=
by
  sorry

end trains_cross_time_l1569_156956


namespace min_value_a_l1569_156912

theorem min_value_a (a b c d : ℚ) (h₀ : a > 0)
  (h₁ : ∀ n : ℕ, (a * n^3 + b * n^2 + c * n + d).den = 1) :
  a = 1/6 := by
  -- Proof goes here
  sorry

end min_value_a_l1569_156912


namespace number_of_roses_now_l1569_156948

-- Given Conditions
def initial_roses : Nat := 7
def initial_orchids : Nat := 12
def current_orchids : Nat := 20
def orchids_more_than_roses : Nat := 9

-- Question to Prove: 
theorem number_of_roses_now :
  ∃ (R : Nat), (current_orchids = R + orchids_more_than_roses) ∧ (R = 11) :=
by {
  sorry
}

end number_of_roses_now_l1569_156948


namespace solve_quadratic_l1569_156933

theorem solve_quadratic (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 :=
sorry

end solve_quadratic_l1569_156933


namespace max_mn_sq_l1569_156924

theorem max_mn_sq {m n : ℤ} (h1: 1 ≤ m ∧ m ≤ 2005) (h2: 1 ≤ n ∧ n ≤ 2005) 
(h3: (n^2 + 2*m*n - 2*m^2)^2 = 1): m^2 + n^2 ≤ 702036 :=
sorry

end max_mn_sq_l1569_156924


namespace factorize_expression_l1569_156947

theorem factorize_expression (x y : ℝ) :
  9 * x^2 - y^2 - 4 * y - 4 = (3 * x + y + 2) * (3 * x - y - 2) :=
by
  sorry

end factorize_expression_l1569_156947


namespace Kyle_papers_delivered_each_week_proof_l1569_156960

-- Definitions based on identified conditions
def k_m := 100        -- Number of papers delivered from Monday to Saturday
def d_m := 6          -- Number of days from Monday to Saturday
def k_s1 := 90        -- Number of regular customers on Sunday
def k_s2 := 30        -- Number of Sunday-only customers

-- Total number of papers delivered in a week
def total_papers_week := (k_m * d_m) + (k_s1 + k_s2)

theorem Kyle_papers_delivered_each_week_proof :
  total_papers_week = 720 :=
by
  sorry

end Kyle_papers_delivered_each_week_proof_l1569_156960


namespace original_cost_of_article_l1569_156962

theorem original_cost_of_article (x: ℝ) (h: 0.76 * x = 320) : x = 421.05 :=
sorry

end original_cost_of_article_l1569_156962


namespace factorize_expression_l1569_156982

theorem factorize_expression (x : ℝ) : 2 * x ^ 2 - 50 = 2 * (x + 5) * (x - 5) := 
  sorry

end factorize_expression_l1569_156982


namespace perimeter_of_T_shaped_figure_l1569_156963

theorem perimeter_of_T_shaped_figure :
  let a := 3    -- width of the horizontal rectangle
  let b := 5    -- height of the horizontal rectangle
  let c := 2    -- width of the vertical rectangle
  let d := 4    -- height of the vertical rectangle
  let overlap := 1 -- overlap length
  2 * a + 2 * b + 2 * c + 2 * d - 2 * overlap = 26 := by
  sorry

end perimeter_of_T_shaped_figure_l1569_156963


namespace sequence_proofs_l1569_156987

theorem sequence_proofs (a b : ℕ → ℝ) :
  a 1 = 1 ∧ b 1 = 0 ∧ 
  (∀ n, 4 * a (n + 1) = 3 * a n - b n + 4) ∧ 
  (∀ n, 4 * b (n + 1) = 3 * b n - a n - 4) → 
  (∀ n, a n + b n = (1 / 2) ^ (n - 1)) ∧ 
  (∀ n, a n - b n = 2 * n - 1) ∧ 
  (∀ n, a n = (1 / 2) ^ n + n - 1 / 2 ∧ b n = (1 / 2) ^ n - n + 1 / 2) :=
sorry

end sequence_proofs_l1569_156987


namespace mixed_oil_rate_l1569_156903

noncomputable def rate_of_mixed_oil
  (volume1 : ℕ) (price1 : ℕ) (volume2 : ℕ) (price2 : ℕ) : ℚ :=
(total_cost : ℚ) / (total_volume : ℚ)
where
  total_cost := volume1 * price1 + volume2 * price2
  total_volume := volume1 + volume2

theorem mixed_oil_rate :
  rate_of_mixed_oil 10 50 5 66 = 55.33 := 
by
  sorry

end mixed_oil_rate_l1569_156903


namespace function_equivalence_l1569_156936

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x) = 6 * x - 1) : ∀ x : ℝ, f x = 3 * x - 1 :=
by
  sorry

end function_equivalence_l1569_156936


namespace parabola_focus_directrix_l1569_156977

-- Definitions and conditions
def parabola (y a x : ℝ) : Prop := y^2 = a * x
def distance_from_focus_to_directrix (d : ℝ) : Prop := d = 2

-- Statement of the problem
theorem parabola_focus_directrix {a : ℝ} (h : parabola y a x) (h2 : distance_from_focus_to_directrix d) : 
  a = 4 ∨ a = -4 :=
sorry

end parabola_focus_directrix_l1569_156977


namespace no_four_nat_satisfy_l1569_156990

theorem no_four_nat_satisfy:
  ∀ (x y z t : ℕ), 3 * x^4 + 5 * y^4 + 7 * z^4 ≠ 11 * t^4 :=
by
  sorry

end no_four_nat_satisfy_l1569_156990


namespace daily_practice_hours_l1569_156940

-- Define the conditions as given in the problem
def total_hours_practiced_this_week : ℕ := 36
def total_days_in_week : ℕ := 7
def days_could_not_practice : ℕ := 1
def actual_days_practiced := total_days_in_week - days_could_not_practice

-- State the theorem including the question and the correct answer, given the conditions
theorem daily_practice_hours :
  total_hours_practiced_this_week / actual_days_practiced = 6 := 
by
  sorry

end daily_practice_hours_l1569_156940


namespace plane_equation_parametric_l1569_156928

theorem plane_equation_parametric 
  (s t : ℝ)
  (v : ℝ × ℝ × ℝ)
  (x y z : ℝ) 
  (A B C D : ℤ)
  (h1 : v = (2 + s + 2 * t, 3 + 2 * s - t, 1 + s + 3 * t))
  (h2 : A = 7)
  (h3 : B = -1)
  (h4 : C = -5)
  (h5 : D = -6)
  (h6 : A > 0)
  (h7 : Int.gcd A (Int.gcd B (Int.gcd C D)) = 1) :
  7 * x - y - 5 * z - 6 = 0 := 
sorry

end plane_equation_parametric_l1569_156928


namespace solve_for_A_l1569_156988

theorem solve_for_A (A B : ℕ) (h1 : 4 * 10 + A + 10 * B + 3 = 68) (h2 : 10 ≤ 4 * 10 + A) (h3 : 4 * 10 + A < 100) (h4 : 10 ≤ 10 * B + 3) (h5 : 10 * B + 3 < 100) (h6 : A < 10) (h7 : B < 10) : A = 5 := 
by
  sorry

end solve_for_A_l1569_156988


namespace set_subset_l1569_156953

-- Define the sets M and N
def M := {x : ℝ | abs x ≤ 1}
def N := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x ≤ 0}

-- The mathematical statement to be proved
theorem set_subset : N ⊆ M := sorry

end set_subset_l1569_156953


namespace tucker_boxes_l1569_156991

def tissues_per_box := 160
def used_tissues := 210
def left_tissues := 270

def total_tissues := used_tissues + left_tissues

theorem tucker_boxes : total_tissues = tissues_per_box * 3 :=
by
  sorry

end tucker_boxes_l1569_156991


namespace opposite_of_2023_l1569_156949

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l1569_156949


namespace diff_of_squares_value_l1569_156942

theorem diff_of_squares_value :
  535^2 - 465^2 = 70000 :=
by sorry

end diff_of_squares_value_l1569_156942


namespace infinitely_many_not_sum_of_three_fourth_powers_l1569_156986

theorem infinitely_many_not_sum_of_three_fourth_powers : ∀ n : ℕ, n > 0 → n ≡ 5 [MOD 16] → ¬(∃ a b c : ℤ, n = a^4 + b^4 + c^4) :=
by sorry

end infinitely_many_not_sum_of_three_fourth_powers_l1569_156986


namespace triangle_ratio_l1569_156916

theorem triangle_ratio (a b c : ℕ) (r s : ℕ) (h1 : a = 9) (h2 : b = 15) (h3 : c = 18) (h4 : r + s = a) (h5 : r < s) : r * 2 = s :=
by
  sorry

end triangle_ratio_l1569_156916


namespace other_factor_of_product_l1569_156944

def product_has_factors (n : ℕ) : Prop :=
  ∃ a b c d e f : ℕ, n = (2^a) * (3^b) * (5^c) * (7^d) * (11^e) * (13^f) ∧ a ≥ 4 ∧ b ≥ 3

def smallest_w (x : ℕ) : ℕ :=
  if h : x = 1452 then 468 else 1

theorem other_factor_of_product (w : ℕ) : 
  (product_has_factors (1452 * w)) → (w = 468) :=
by
  sorry

end other_factor_of_product_l1569_156944


namespace x_percent_more_than_y_l1569_156996

theorem x_percent_more_than_y (z : ℝ) (hz : z ≠ 0) (y : ℝ) (x : ℝ)
  (h1 : y = 0.70 * z) (h2 : x = 0.84 * z) :
  x = y + 0.20 * y :=
by
  -- proof goes here
  sorry

end x_percent_more_than_y_l1569_156996


namespace find_x_l1569_156964

theorem find_x (x : ℝ) (h : ∑' n : ℕ, (n + 1) * x ^ n = 9) : x = 2 / 3 :=
sorry

end find_x_l1569_156964


namespace vasya_made_mistake_l1569_156929

theorem vasya_made_mistake : 
  ∀ (total_digits : ℕ), 
    total_digits = 301 → 
    ¬∃ (n : ℕ), 
      (n ≤ 9 ∧ total_digits = (n * 1)) ∨ 
      (10 ≤ n ∧ n ≤ 99 ∧ total_digits = (9 * 1) + ((n - 9) * 2)) ∨ 
      (100 ≤ n ∧ total_digits = (9 * 1) + (90 * 2) + ((n - 99) * 3)) := 
by 
  sorry

end vasya_made_mistake_l1569_156929


namespace parabola_focus_distance_l1569_156909

open Real

noncomputable def parabola (P : ℝ × ℝ) : Prop := (P.2)^2 = 4 * P.1
def line_eq (P : ℝ × ℝ) : Prop := abs (P.1 + 2) = 6

theorem parabola_focus_distance (P : ℝ × ℝ) 
  (hp : parabola P) 
  (hl : line_eq P) : 
  dist P (1 / 4, 0) = 5 :=
sorry

end parabola_focus_distance_l1569_156909


namespace code_XYZ_to_base_10_l1569_156905

def base_6_to_base_10 (x y z : ℕ) : ℕ :=
  x * 6^2 + y * 6^1 + z * 6^0

theorem code_XYZ_to_base_10 :
  ∀ (X Y Z : ℕ), 
    X = 5 ∧ Y = 0 ∧ Z = 4 →
    base_6_to_base_10 X Y Z = 184 :=
by
  intros X Y Z h
  cases' h with hX hYZ
  cases' hYZ with hY hZ
  rw [hX, hY, hZ]
  exact rfl

end code_XYZ_to_base_10_l1569_156905


namespace simplify_polynomial_l1569_156992

theorem simplify_polynomial (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 5) - (x^6 + 2 * x^5 + x^4 - x^3 + 7) = 
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - 2 :=
by
  sorry

end simplify_polynomial_l1569_156992


namespace fraction_students_say_dislike_but_actually_like_is_25_percent_l1569_156972

variable (total_students : Nat) (students_like_dancing : Nat) (students_dislike_dancing : Nat) 
         (students_like_dancing_but_say_dislike : Nat) (students_dislike_dancing_and_say_dislike : Nat) 
         (total_say_dislike : Nat)

def fraction_of_students_who_say_dislike_but_actually_like (total_students students_like_dancing students_dislike_dancing 
         students_like_dancing_but_say_dislike students_dislike_dancing_and_say_dislike total_say_dislike : Nat) : Nat :=
    (students_like_dancing_but_say_dislike * 100) / total_say_dislike

theorem fraction_students_say_dislike_but_actually_like_is_25_percent
  (h1 : total_students = 100)
  (h2 : students_like_dancing = 60)
  (h3 : students_dislike_dancing = 40)
  (h4 : students_like_dancing_but_say_dislike = 12)
  (h5 : students_dislike_dancing_and_say_dislike = 36)
  (h6 : total_say_dislike = 48) :
  fraction_of_students_who_say_dislike_but_actually_like total_students students_like_dancing students_dislike_dancing 
    students_like_dancing_but_say_dislike students_dislike_dancing_and_say_dislike total_say_dislike = 25 :=
by sorry

end fraction_students_say_dislike_but_actually_like_is_25_percent_l1569_156972


namespace age_difference_36_l1569_156911

noncomputable def jack_age (a b : ℕ) : ℕ := 10 * a + b
noncomputable def bill_age (b a : ℕ) : ℕ := 10 * b + a

theorem age_difference_36 (a b : ℕ) (h : 10 * a + b + 3 = 3 * (10 * b + a + 3)) :
  jack_age a b - bill_age b a = 36 :=
by sorry

end age_difference_36_l1569_156911


namespace neither_necessary_nor_sufficient_l1569_156938

noncomputable def C1 (m n : ℝ) :=
  (m ^ 2 - 4 * n ≥ 0) ∧ (m > 0) ∧ (n > 0)

noncomputable def C2 (m n : ℝ) :=
  (m > 0) ∧ (n > 0) ∧ (m ≠ n)

theorem neither_necessary_nor_sufficient (m n : ℝ) :
  ¬(C1 m n → C2 m n) ∧ ¬(C2 m n → C1 m n) :=
sorry

end neither_necessary_nor_sufficient_l1569_156938


namespace correct_average_marks_l1569_156913

theorem correct_average_marks
  (n : ℕ) (avg_mks wrong_mk correct_mk correct_avg_mks : ℕ)
  (H1 : n = 10)
  (H2 : avg_mks = 100)
  (H3 : wrong_mk = 50)
  (H4 : correct_mk = 10)
  (H5 : correct_avg_mks = 96) :
  (n * avg_mks - wrong_mk + correct_mk) / n = correct_avg_mks :=
by
  sorry

end correct_average_marks_l1569_156913


namespace solution_set_f_leq_g_range_of_a_l1569_156904

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + abs (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := x + 2

theorem solution_set_f_leq_g (x : ℝ) : f x 1 ≤ g x ↔ (0 ≤ x ∧ x ≤ 2 / 3) := by
  sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ g x) : 2 ≤ a := by
  sorry

end solution_set_f_leq_g_range_of_a_l1569_156904


namespace range_a_part1_range_a_part2_l1569_156952

def A (x : ℝ) : Prop := x^2 - 3*x + 2 ≤ 0
def B (x a : ℝ) : Prop := x = x^2 - 4*x + a
def C (x a : ℝ) : Prop := x^2 - a*x - 4 ≤ 0

def p (a : ℝ) : Prop := ∃ x : ℝ, A x ∧ B x a
def q (a : ℝ) : Prop := ∀ x : ℝ, A x → C x a

theorem range_a_part1 : ¬(p a) → a > 6 := sorry

theorem range_a_part2 : p a ∧ q a → 0 ≤ a ∧ a ≤ 6 := sorry

end range_a_part1_range_a_part2_l1569_156952


namespace inequality_proof_l1569_156934

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / (x^2 + y^2)) + (1 / x^2) + (1 / y^2) ≥ 10 / (x + y)^2 :=
sorry

end inequality_proof_l1569_156934


namespace StepaMultiplication_l1569_156937

theorem StepaMultiplication {a : ℕ} (h1 : Grisha's_answer = (3 / 2) ^ 4 * a)
  (h2 : Grisha's_answer = 81) :
  (∃ (m n : ℕ), m * n = (3 / 2) ^ 3 * a ∧ m < 10 ∧ n < 10) :=
by
  sorry

end StepaMultiplication_l1569_156937


namespace find_m_n_l1569_156959

theorem find_m_n (m n : ℕ) (h1 : m ≥ 0) (h2 : n ≥ 0) (h3 : 3^m - 7^n = 2) : m = 2 ∧ n = 1 := 
sorry

end find_m_n_l1569_156959


namespace books_still_to_read_l1569_156997

-- Define the given conditions
def total_books : ℕ := 22
def books_read : ℕ := 12

-- State the theorem to be proven
theorem books_still_to_read : total_books - books_read = 10 := 
by
  -- skipping the proof
  sorry

end books_still_to_read_l1569_156997


namespace arithmetic_mean_geometric_mean_l1569_156943

theorem arithmetic_mean_geometric_mean (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end arithmetic_mean_geometric_mean_l1569_156943


namespace sum_floor_ceil_eq_seven_l1569_156910

theorem sum_floor_ceil_eq_seven (x : ℝ) 
  (h : ⌊x⌋ + ⌈x⌉ = 7) : 3 < x ∧ x < 4 := 
sorry

end sum_floor_ceil_eq_seven_l1569_156910


namespace find_n_l1569_156920

theorem find_n (n : ℕ) : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ n → n = 29 :=
by
  sorry

end find_n_l1569_156920


namespace freddy_talk_time_dad_l1569_156978

-- Conditions
def localRate : ℝ := 0.05
def internationalRate : ℝ := 0.25
def talkTimeBrother : ℕ := 31
def totalCost : ℝ := 10.0

-- Goal: Prove the duration of Freddy's local call to his dad is 45 minutes
theorem freddy_talk_time_dad : 
  ∃ (talkTimeDad : ℕ), 
    talkTimeDad = 45 ∧
    totalCost = (talkTimeBrother : ℝ) * internationalRate + (talkTimeDad : ℝ) * localRate := 
by
  sorry

end freddy_talk_time_dad_l1569_156978


namespace min_value_of_polynomial_l1569_156983

theorem min_value_of_polynomial : ∃ x : ℝ, (x^2 + x + 1) = 3 / 4 :=
by {
  -- Solution steps are omitted
  sorry
}

end min_value_of_polynomial_l1569_156983


namespace max_sum_of_ten_consecutive_in_hundred_l1569_156965

theorem max_sum_of_ten_consecutive_in_hundred :
  ∀ (s : Fin 100 → ℕ), (∀ i : Fin 100, 1 ≤ s i ∧ s i ≤ 100) → 
  (∃ i : Fin 91, (s i + s (i + 1) + s (i + 2) + s (i + 3) +
  s (i + 4) + s (i + 5) + s (i + 6) + s (i + 7) + s (i + 8) + s (i + 9)) ≥ 505) :=
by
  intro s hs
  sorry

end max_sum_of_ten_consecutive_in_hundred_l1569_156965


namespace sum_of_money_l1569_156984

theorem sum_of_money (A B C : ℝ) (hB : B = 0.65 * A) (hC : C = 0.40 * A) (hC_val : C = 56) :
  A + B + C = 287 :=
by {
  sorry
}

end sum_of_money_l1569_156984


namespace sufficient_condition_not_monotonic_l1569_156923

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 4 * a * x - Real.log x

def sufficient_not_monotonic (a : ℝ) : Prop :=
  (a > 1 / 6) ∨ (a < -1 / 2)

theorem sufficient_condition_not_monotonic (a : ℝ) :
  sufficient_not_monotonic a → ¬(∀ x y : ℝ, 1 < x ∧ x < 3 ∧ 1 < y ∧ y < 3 ∧ x ≠ y → ((f a x - f a y) / (x - y) ≥ 0 ∨ (f a y - f a x) / (y - x) ≥ 0)) :=
by
  sorry

end sufficient_condition_not_monotonic_l1569_156923


namespace polynomial_expansion_l1569_156973

theorem polynomial_expansion (x : ℝ) :
  (5 * x^2 + 3 * x - 7) * (4 * x^3) = 20 * x^5 + 12 * x^4 - 28 * x^3 :=
by 
  sorry

end polynomial_expansion_l1569_156973


namespace necessary_but_not_sufficient_l1569_156932

variable {I : Set ℝ} (f : ℝ → ℝ) (M : ℝ)

theorem necessary_but_not_sufficient :
  (∀ x ∈ I, f x ≤ M) ↔
  (∀ x ∈ I, f x ≤ M ∧ (∃ x ∈ I, f x = M) → M = M ∧ ∃ x ∈ I, f x = M) :=
by
  sorry

end necessary_but_not_sufficient_l1569_156932


namespace joanne_first_hour_coins_l1569_156915

theorem joanne_first_hour_coins 
  (X : ℕ)
  (H1 : 70 = 35 + 35)
  (H2 : 120 = X + 70 + 35)
  (H3 : 35 = 50 - 15) : 
  X = 15 :=
sorry

end joanne_first_hour_coins_l1569_156915


namespace joaozinho_card_mariazinha_card_pedrinho_error_l1569_156900

-- Define the card transformation function
def transform_card (number : ℕ) (color_adjustment : ℕ) : ℕ :=
  (number * 2 + 3) * 5 + color_adjustment

-- The proof problems
theorem joaozinho_card : transform_card 3 4 = 49 :=
by
  sorry

theorem mariazinha_card : ∃ number, ∃ color_adjustment, transform_card number color_adjustment = 76 :=
by
  sorry

theorem pedrinho_error : ∀ number color_adjustment, ¬ transform_card number color_adjustment = 61 :=
by
  sorry

end joaozinho_card_mariazinha_card_pedrinho_error_l1569_156900


namespace minutes_before_noon_l1569_156906

theorem minutes_before_noon (x : ℕ) (h1 : x = 40)
  (h2 : ∀ (t : ℕ), t = 180 - (x + 40) ∧ t = 3 * x) : x = 35 :=
by {
  sorry
}

end minutes_before_noon_l1569_156906
