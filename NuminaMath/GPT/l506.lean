import Mathlib

namespace NUMINAMATH_GPT_mike_pens_l506_50615

-- Definitions based on the conditions
def initial_pens : ℕ := 25
def pens_after_mike (M : ℕ) : ℕ := initial_pens + M
def pens_after_cindy (M : ℕ) : ℕ := 2 * pens_after_mike M
def pens_after_sharon (M : ℕ) : ℕ := pens_after_cindy M - 19
def final_pens : ℕ := 75

-- The theorem we need to prove
theorem mike_pens (M : ℕ) (h : pens_after_sharon M = final_pens) : M = 22 := by
  have h1 : pens_after_sharon M = 2 * (25 + M) - 19 := rfl
  rw [h1] at h
  sorry

end NUMINAMATH_GPT_mike_pens_l506_50615


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l506_50600

theorem speed_of_man_in_still_water
  (V_m V_s : ℝ)
  (cond1 : V_m + V_s = 5)
  (cond2 : V_m - V_s = 7) :
  V_m = 6 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l506_50600


namespace NUMINAMATH_GPT_geometric_sequence_sum_six_l506_50692

theorem geometric_sequence_sum_six (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h1 : 0 < q)
  (h2 : a 1 = 1)
  (h3 : a 3 * a 5 = 64)
  (h4 : ∀ n, a n = a 1 * q^(n-1))
  (h5 : ∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) :
  S 6 = 63 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_six_l506_50692


namespace NUMINAMATH_GPT_area_of_triangle_DEF_l506_50659

theorem area_of_triangle_DEF :
  let D := (0, 2)
  let E := (6, 0)
  let F := (3, 8)
  let base1 := 6
  let height1 := 2
  let base2 := 3
  let height2 := 8
  let base3 := 3
  let height3 := 6
  let area_triangle_DE := 1 / 2 * (base1 * height1)
  let area_triangle_EF := 1 / 2 * (base2 * height2)
  let area_triangle_FD := 1 / 2 * (base3 * height3)
  let area_rectangle := 6 * 8
  ∃ area_def_triangle, 
  area_def_triangle = area_rectangle - (area_triangle_DE + area_triangle_EF + area_triangle_FD) 
  ∧ area_def_triangle = 21 :=
by 
  sorry

end NUMINAMATH_GPT_area_of_triangle_DEF_l506_50659


namespace NUMINAMATH_GPT_pokemon_card_cost_l506_50657

theorem pokemon_card_cost 
  (football_cost : ℝ)
  (num_football_packs : ℕ) 
  (baseball_cost : ℝ) 
  (total_spent : ℝ) 
  (h_football : football_cost = 2.73)
  (h_num_football_packs : num_football_packs = 2)
  (h_baseball : baseball_cost = 8.95)
  (h_total : total_spent = 18.42) :
  (total_spent - (num_football_packs * football_cost + baseball_cost) = 4.01) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_pokemon_card_cost_l506_50657


namespace NUMINAMATH_GPT_min_value_fraction_l506_50602

theorem min_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b^2 - 4 * a * c ≤ 0) :
  (a + b + c) / (2 * a) ≥ 2 :=
  sorry

end NUMINAMATH_GPT_min_value_fraction_l506_50602


namespace NUMINAMATH_GPT_true_discount_different_time_l506_50603

theorem true_discount_different_time (FV TD_initial TD_different : ℝ) (r : ℝ) (initial_time different_time : ℝ) 
  (h1 : r = initial_time / different_time)
  (h2 : FV = 110)
  (h3 : TD_initial = 10)
  (h4 : initial_time / different_time = 1 / 2) :
  TD_different = 2 * TD_initial :=
by
  sorry

end NUMINAMATH_GPT_true_discount_different_time_l506_50603


namespace NUMINAMATH_GPT_equation_of_line_passing_through_ellipse_midpoint_l506_50636

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

theorem equation_of_line_passing_through_ellipse_midpoint
  (x1 y1 x2 y2 : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (1, 1))
  (hA : ellipse x1 y1)
  (hB : ellipse x2 y2)
  (midAB : (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = 1) :
  ∃ (a b c : ℝ), a = 4 ∧ b = 3 ∧ c = -7 ∧ a * P.2 + b * P.1 + c = 0 :=
sorry

end NUMINAMATH_GPT_equation_of_line_passing_through_ellipse_midpoint_l506_50636


namespace NUMINAMATH_GPT_alien_collected_95_units_l506_50698

def convert_base_six_to_ten (n : ℕ) : ℕ :=
  match n with
  | 235 => 2 * 6^2 + 3 * 6^1 + 5 * 6^0
  | _ => 0

theorem alien_collected_95_units : convert_base_six_to_ten 235 = 95 := by
  sorry

end NUMINAMATH_GPT_alien_collected_95_units_l506_50698


namespace NUMINAMATH_GPT_rounding_problem_l506_50628

def given_number : ℝ := 3967149.487234

theorem rounding_problem : (3967149.487234).round = 3967149 := sorry

end NUMINAMATH_GPT_rounding_problem_l506_50628


namespace NUMINAMATH_GPT_mail_distribution_l506_50669

def pieces_per_block (total_pieces blocks : ℕ) : ℕ := total_pieces / blocks

theorem mail_distribution : pieces_per_block 192 4 = 48 := 
by { 
    -- Proof skipped
    sorry 
}

end NUMINAMATH_GPT_mail_distribution_l506_50669


namespace NUMINAMATH_GPT_min_species_needed_l506_50641

theorem min_species_needed (num_birds : ℕ) (h1 : num_birds = 2021)
  (h2 : ∀ (s : ℤ) (x y : ℕ), x ≠ y → (between_same_species : ℕ) → (h3 : between_same_species = y - x - 1) → between_same_species % 2 = 0) :
  ∃ (species : ℕ), num_birds ≤ 2 * species ∧ species = 1011 :=
by
  sorry

end NUMINAMATH_GPT_min_species_needed_l506_50641


namespace NUMINAMATH_GPT_stddev_newData_l506_50684

-- Definitions and conditions
def variance (data : List ℝ) : ℝ := sorry  -- Placeholder for variance definition
def stddev (data : List ℝ) : ℝ := sorry    -- Placeholder for standard deviation definition

-- Given data
def data : List ℝ := sorry                -- Placeholder for the data x_1, x_2, ..., x_8
def newData : List ℝ := data.map (λ x => 2 * x + 1)

-- Given condition
axiom variance_data : variance data = 16

-- Proof of the statement
theorem stddev_newData : stddev newData = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_stddev_newData_l506_50684


namespace NUMINAMATH_GPT_sum_of_squares_500_l506_50670

theorem sum_of_squares_500 : (Finset.range 500).sum (λ x => (x + 1) ^ 2) = 41841791750 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_500_l506_50670


namespace NUMINAMATH_GPT_correct_parentheses_l506_50656

theorem correct_parentheses : (1 * 2 * 3 + 4) * 5 = 50 := by
  sorry

end NUMINAMATH_GPT_correct_parentheses_l506_50656


namespace NUMINAMATH_GPT_vector_subtraction_l506_50643

theorem vector_subtraction (p q: ℝ × ℝ × ℝ) (hp: p = (5, -3, 2)) (hq: q = (-1, 4, -2)) :
  p - 2 • q = (7, -11, 6) :=
by
  sorry

end NUMINAMATH_GPT_vector_subtraction_l506_50643


namespace NUMINAMATH_GPT_sum_cubes_eq_neg_27_l506_50653

theorem sum_cubes_eq_neg_27 (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
 (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
 a^3 + b^3 + c^3 = -27 :=
by
  sorry

end NUMINAMATH_GPT_sum_cubes_eq_neg_27_l506_50653


namespace NUMINAMATH_GPT_find_number_l506_50666

theorem find_number (n : ℕ) (h : (1 / 2 : ℝ) * n + 5 = 13) : n = 16 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l506_50666


namespace NUMINAMATH_GPT_equal_real_roots_l506_50625

theorem equal_real_roots (m : ℝ) : (∃ x : ℝ, x * x - 4 * x - m = 0) → (16 + 4 * m = 0) → m = -4 :=
by
  sorry

end NUMINAMATH_GPT_equal_real_roots_l506_50625


namespace NUMINAMATH_GPT_problem1_problem2_l506_50614

theorem problem1 : (3 + Real.sqrt 5) * (Real.sqrt 5 - 2) = Real.sqrt 5 - 1 :=
  sorry

theorem problem2 : (Real.sqrt 12 + Real.sqrt 27) / Real.sqrt 3 = 5 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l506_50614


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l506_50608

theorem algebraic_expression_evaluation (m : ℝ) (h : m^2 - m - 3 = 0) : m^2 - m - 2 = 1 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l506_50608


namespace NUMINAMATH_GPT_dima_walking_speed_l506_50679

def Dima_station_time := 18 * 60 -- in minutes
def Dima_actual_arrival := 17 * 60 + 5 -- in minutes
def car_speed := 60 -- in km/h
def early_arrival := 10 -- in minutes

def walking_speed (arrival_time actual_arrival car_speed early_arrival : ℕ) : ℕ :=
(car_speed * early_arrival / 60) * (60 / (arrival_time - actual_arrival - early_arrival))

theorem dima_walking_speed :
  walking_speed Dima_station_time Dima_actual_arrival car_speed early_arrival = 6 :=
sorry

end NUMINAMATH_GPT_dima_walking_speed_l506_50679


namespace NUMINAMATH_GPT_calculate_x_times_a_l506_50620

-- Define variables and assumptions
variables (a b x y : ℕ)
variable (hb : b = 4)
variable (hy : y = 2)
variable (h1 : a = 2 * b)
variable (h2 : x = 3 * y)
variable (h3 : a + b = x * y)

-- The statement to be proved
theorem calculate_x_times_a : x * a = 48 :=
by sorry

end NUMINAMATH_GPT_calculate_x_times_a_l506_50620


namespace NUMINAMATH_GPT_largest_interior_angle_of_triangle_l506_50678

theorem largest_interior_angle_of_triangle (a b c ext : ℝ)
    (h1 : a + b + c = 180)
    (h2 : a / 4 = b / 5)
    (h3 : a / 4 = c / 6)
    (h4 : c + 120 = a + 180) : c = 72 :=
by
  sorry

end NUMINAMATH_GPT_largest_interior_angle_of_triangle_l506_50678


namespace NUMINAMATH_GPT_region_transformation_area_l506_50686

-- Define the region T with area 15
def region_T : ℝ := 15

-- Define the transformation matrix
def matrix_M : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![ 3, 4 ],
  ![ 5, -2 ]
]

-- The determinant of the matrix
def det_matrix_M : ℝ := 3 * (-2) - 4 * 5

-- The proven target statement to show that after the transformation, the area of T' is 390
theorem region_transformation_area :
  ∃ (area_T' : ℝ), area_T' = |det_matrix_M| * region_T ∧ area_T' = 390 :=
by
  sorry

end NUMINAMATH_GPT_region_transformation_area_l506_50686


namespace NUMINAMATH_GPT_b_minus_a_less_zero_l506_50675

-- Given conditions
variables {a b : ℝ}

-- Define the condition
def a_greater_b (a b : ℝ) : Prop := a > b

-- Lean 4 proof problem statement
theorem b_minus_a_less_zero (a b : ℝ) (h : a_greater_b a b) : b - a < 0 := 
sorry

end NUMINAMATH_GPT_b_minus_a_less_zero_l506_50675


namespace NUMINAMATH_GPT_rhombus_area_correct_l506_50650

/-- Define the rhombus area calculation in miles given the lengths of its diagonals -/
def scale := 250
def d1 := 6 * scale -- first diagonal in miles
def d2 := 12 * scale -- second diagonal in miles
def areaOfRhombus (d1 d2 : ℕ) : ℕ := (d1 * d2) / 2

theorem rhombus_area_correct :
  areaOfRhombus d1 d2 = 2250000 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_correct_l506_50650


namespace NUMINAMATH_GPT_steve_initial_amount_l506_50634

theorem steve_initial_amount
  (P : ℝ) 
  (h : (1.1^2) * P = 121) : 
  P = 100 := 
by 
  sorry

end NUMINAMATH_GPT_steve_initial_amount_l506_50634


namespace NUMINAMATH_GPT_f_one_zero_inequality_solution_l506_50689

noncomputable def f : ℝ → ℝ := sorry

axiom increasing_f : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom functional_eq : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y
axiom f_six : f 6 = 1

-- Part 1: Prove that f(1) = 0
theorem f_one_zero : f 1 = 0 := sorry

-- Part 2: Prove that ∀ x ∈ (0, (-3 + sqrt 153) / 2), f(x + 3) - f(1 / x) < 2
theorem inequality_solution : ∀ x, 0 < x → x < (-3 + Real.sqrt 153) / 2 → f (x + 3) - f (1 / x) < 2 := sorry

end NUMINAMATH_GPT_f_one_zero_inequality_solution_l506_50689


namespace NUMINAMATH_GPT_cube_split_l506_50645

theorem cube_split (m : ℕ) (h1 : m > 1)
  (h2 : ∃ (p : ℕ), (p = (m - 1) * (m^2 + m + 1) ∨ p = (m - 1)^2 ∨ p = (m - 1)^2 + 2) ∧ p = 2017) :
  m = 46 :=
by {
    sorry
}

end NUMINAMATH_GPT_cube_split_l506_50645


namespace NUMINAMATH_GPT_union_complement_eq_l506_50695

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end NUMINAMATH_GPT_union_complement_eq_l506_50695


namespace NUMINAMATH_GPT_complex_value_l506_50629

open Complex

theorem complex_value (z : ℂ)
  (h : 15 * normSq z = 3 * normSq (z + 3) + normSq (z^2 + 4) + 25) :
  z + (8 / z) = -4 :=
sorry

end NUMINAMATH_GPT_complex_value_l506_50629


namespace NUMINAMATH_GPT_sum_of_roots_eq_neg2_l506_50660

-- Define the quadratic equation.
def quadratic_equation (x : ℝ) : ℝ :=
  x^2 + 2 * x - 1

-- Define a predicate to express that x is a root of the quadratic equation.
def is_root (x : ℝ) : Prop :=
  quadratic_equation x = 0

-- Define the statement that the sum of the two roots of the quadratic equation equals -2.
theorem sum_of_roots_eq_neg2 (x1 x2 : ℝ) (h1 : is_root x1) (h2 : is_root x2) (h3 : x1 ≠ x2) :
  x1 + x2 = -2 :=
  sorry

end NUMINAMATH_GPT_sum_of_roots_eq_neg2_l506_50660


namespace NUMINAMATH_GPT_minor_premise_wrong_l506_50681

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := x^2 + x

theorem minor_premise_wrong : ¬ is_even_function f ∧ ¬ is_odd_function f := 
by
  sorry

end NUMINAMATH_GPT_minor_premise_wrong_l506_50681


namespace NUMINAMATH_GPT_monotonic_intervals_logarithmic_inequality_l506_50631

noncomputable def f (x : ℝ) : ℝ := x^2 - x - Real.log x

theorem monotonic_intervals :
  (∀ x ∈ Set.Ioo 0 1, f x > f (x + 1E-9) ∧ f x < f (x - 1E-9)) ∧ 
  (∀ y ∈ Set.Ioi 1, f y < f (y + 1E-9) ∧ f y > f (y - 1E-9)) := sorry

theorem logarithmic_inequality (a : ℝ) (ha : a > 0) (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (hneq : x1 ≠ x2)
  (h_eq1 : a * x1 + f x1 = x1^2 - x1) (h_eq2 : a * x2 + f x2 = x2^2 - x2) :
  Real.log x1 + Real.log x2 + 2 * Real.log a < 0 := sorry

end NUMINAMATH_GPT_monotonic_intervals_logarithmic_inequality_l506_50631


namespace NUMINAMATH_GPT_find_y_l506_50647

theorem find_y (x y : ℝ) (h1 : x ^ (3 * y) = 8) (h2 : x = 2) : y = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_y_l506_50647


namespace NUMINAMATH_GPT_triangle_ABC_properties_l506_50682

theorem triangle_ABC_properties
  (a b c : ℝ)
  (A B C : ℝ)
  (area_ABC : Real.sqrt 15 * 3 = 1/2 * b * c * Real.sin A)
  (cos_A : Real.cos A = -1/4)
  (b_minus_c : b - c = 2) :
  (a = 8 ∧ Real.sin C = Real.sqrt 15 / 8) ∧
  (Real.cos (2 * A + Real.pi / 6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16) := by
  sorry

end NUMINAMATH_GPT_triangle_ABC_properties_l506_50682


namespace NUMINAMATH_GPT_find_LCM_l506_50665

-- Given conditions
def A := ℕ
def B := ℕ
def h := 22
def productAB := 45276

-- The theorem we want to prove
theorem find_LCM (a b lcm : ℕ) (hcf : ℕ) 
  (H_product : a * b = productAB) (H_hcf : hcf = h) : 
  (lcm = productAB / hcf) → 
  (a * b = hcf * lcm) :=
by
  intros H_lcm
  sorry

end NUMINAMATH_GPT_find_LCM_l506_50665


namespace NUMINAMATH_GPT_polynomial_sum_coeff_l506_50640

-- Definitions for the polynomials given
def poly1 (d : ℤ) : ℤ := 15 * d^3 + 19 * d^2 + 17 * d + 18
def poly2 (d : ℤ) : ℤ := 3 * d^3 + 4 * d + 2

-- The main statement to prove
theorem polynomial_sum_coeff :
  let p := 18
  let q := 19
  let r := 21
  let s := 20
  p + q + r + s = 78 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_sum_coeff_l506_50640


namespace NUMINAMATH_GPT_solve_system_l506_50677

theorem solve_system (x y z w : ℝ) :
  x - y + z - w = 2 ∧
  x^2 - y^2 + z^2 - w^2 = 6 ∧
  x^3 - y^3 + z^3 - w^3 = 20 ∧
  x^4 - y^4 + z^4 - w^4 = 60 ↔
  (x = 1 ∧ y = 2 ∧ z = 3 ∧ w = 0) ∨
  (x = 1 ∧ y = 0 ∧ z = 3 ∧ w = 2) ∨
  (x = 3 ∧ y = 2 ∧ z = 1 ∧ w = 0) ∨
  (x = 3 ∧ y = 0 ∧ z = 1 ∧ w = 2) :=
sorry

end NUMINAMATH_GPT_solve_system_l506_50677


namespace NUMINAMATH_GPT_combined_tax_rate_l506_50654

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (Mork_tax_rate Mindy_tax_rate : ℝ)
  (h1 : Mork_tax_rate = 0.4) (h2 : Mindy_tax_rate = 0.3) (h3 : Mindy_income = 4 * Mork_income) :
  ((Mork_tax_rate * Mork_income + Mindy_tax_rate * Mindy_income) / (Mork_income + Mindy_income)) * 100 = 32 :=
by
  sorry

end NUMINAMATH_GPT_combined_tax_rate_l506_50654


namespace NUMINAMATH_GPT_initial_nickels_eq_l506_50626

variable (quarters : ℕ) (initial_nickels : ℕ) (nickels_borrowed : ℕ) (nickels_left : ℕ)

-- Assumptions based on the problem
axiom quarters_had : quarters = 33
axiom nickels_left_axiom : nickels_left = 12
axiom nickels_borrowed_axiom : nickels_borrowed = 75

-- Theorem to prove: initial number of nickels
theorem initial_nickels_eq :
  initial_nickels = nickels_left + nickels_borrowed :=
by
  sorry

end NUMINAMATH_GPT_initial_nickels_eq_l506_50626


namespace NUMINAMATH_GPT_magnitude_2a_minus_b_l506_50648

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (θ : ℝ) (h_angle : θ = 5 * Real.pi / 6)
variables (h_mag_a : ‖a‖ = 4) (h_mag_b : ‖b‖ = Real.sqrt 3)

theorem magnitude_2a_minus_b :
  ‖2 • a - b‖ = Real.sqrt 91 := by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_magnitude_2a_minus_b_l506_50648


namespace NUMINAMATH_GPT_entrance_sum_2_to_3_pm_exit_sum_2_to_3_pm_no_crowd_control_at_4_pm_l506_50697

noncomputable def f : ℕ → ℕ
| n => if 1 ≤ n ∧ n ≤ 8 then 200 * n + 2000
       else if 9 ≤ n ∧ n ≤ 32 then 360 * (3 ^ ((n - 8) / 12)) + 3000
       else if 33 ≤ n ∧ n ≤ 45 then 32400 - 720 * n
       else 0

noncomputable def g : ℕ → ℕ
| n => if 1 ≤ n ∧ n ≤ 18 then 0
       else if 19 ≤ n ∧ n ≤ 32 then 500 * n - 9000
       else if 33 ≤ n ∧ n ≤ 45 then 8800
       else 0

theorem entrance_sum_2_to_3_pm : f 21 + f 22 + f 23 + f 24 = 17460 := by
  sorry

theorem exit_sum_2_to_3_pm : g 21 + g 22 + g 23 + g 24 = 9000 := by
  sorry

theorem no_crowd_control_at_4_pm : f 28 - g 28 < 80000 := by
  sorry

end NUMINAMATH_GPT_entrance_sum_2_to_3_pm_exit_sum_2_to_3_pm_no_crowd_control_at_4_pm_l506_50697


namespace NUMINAMATH_GPT_sum_in_correct_range_l506_50651

-- Define the mixed numbers
def mixed1 := 1 + 1/4
def mixed2 := 4 + 1/3
def mixed3 := 6 + 1/12

-- Their sum
def sumMixed := mixed1 + mixed2 + mixed3

-- Correct sum in mixed number form
def correctSum := 11 + 2/3

-- Range we need to check
def lowerBound := 11 + 1/2
def upperBound := 12

theorem sum_in_correct_range : sumMixed = correctSum ∧ lowerBound < correctSum ∧ correctSum < upperBound := by
  sorry

end NUMINAMATH_GPT_sum_in_correct_range_l506_50651


namespace NUMINAMATH_GPT_smallest_three_digit_pqr_l506_50668

theorem smallest_three_digit_pqr (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  100 ≤ p * q^2 * r ∧ p * q^2 * r < 1000 → p * q^2 * r = 126 := 
sorry

end NUMINAMATH_GPT_smallest_three_digit_pqr_l506_50668


namespace NUMINAMATH_GPT_area_of_rectangle_ABCD_l506_50624

theorem area_of_rectangle_ABCD :
  ∀ (short_side long_side width length : ℝ),
    (short_side = 6) →
    (long_side = 6 * (3 / 2)) →
    (width = 2 * short_side) →
    (length = long_side) →
    (width * length = 108) :=
by
  intros short_side long_side width length h_short h_long h_width h_length
  rw [h_short, h_long] at *
  sorry

end NUMINAMATH_GPT_area_of_rectangle_ABCD_l506_50624


namespace NUMINAMATH_GPT_min_value_of_sum_l506_50642

theorem min_value_of_sum (x y : ℝ) (h1 : x + 4 * y = 2 * x * y) (h2 : 0 < x) (h3 : 0 < y) : 
  x + y ≥ 9 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_l506_50642


namespace NUMINAMATH_GPT_roots_of_quadratic_are_integers_l506_50627

theorem roots_of_quadratic_are_integers
  (b c : ℤ)
  (Δ : ℤ)
  (h_discriminant: Δ = b^2 - 4 * c)
  (h_perfect_square: ∃ k : ℤ, k^2 = Δ)
  : (∃ x1 x2 : ℤ, x1 * x2 = c ∧ x1 + x2 = -b) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_are_integers_l506_50627


namespace NUMINAMATH_GPT_function_value_bounds_l506_50607

theorem function_value_bounds (x : ℝ) : 
  (x^2 + x + 1) / (x^2 + 1) ≤ 3 / 2 ∧ (x^2 + x + 1) / (x^2 + 1) ≥ 1 / 2 := 
sorry

end NUMINAMATH_GPT_function_value_bounds_l506_50607


namespace NUMINAMATH_GPT_count_integer_values_of_x_l506_50638

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : ∃ (n : ℕ), n = 23 :=
by
  sorry

end NUMINAMATH_GPT_count_integer_values_of_x_l506_50638


namespace NUMINAMATH_GPT_arvin_fifth_day_running_distance_l506_50673

theorem arvin_fifth_day_running_distance (total_km : ℕ) (first_day_km : ℕ) (increment : ℕ) (days : ℕ) 
  (h1 : total_km = 20) (h2 : first_day_km = 2) (h3 : increment = 1) (h4 : days = 5) : 
  first_day_km + (increment * (days - 1)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_arvin_fifth_day_running_distance_l506_50673


namespace NUMINAMATH_GPT_polynomial_product_linear_term_zero_const_six_l506_50667

theorem polynomial_product_linear_term_zero_const_six (a b : ℝ)
  (h1 : (a + 2 * b = 0)) 
  (h2 : b = 6) : (a + b = -6) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_product_linear_term_zero_const_six_l506_50667


namespace NUMINAMATH_GPT_second_number_is_22_l506_50652

noncomputable section

variables (x y : ℕ)

-- Definitions based on the conditions
-- Condition 1: The sum of two numbers is 33
def sum_condition : Prop := x + y = 33

-- Condition 2: The second number is twice the first number
def twice_condition : Prop := y = 2 * x

-- Theorem: Given the conditions, the second number y is 22.
theorem second_number_is_22 (h1 : sum_condition x y) (h2 : twice_condition x y) : y = 22 :=
by
  sorry

end NUMINAMATH_GPT_second_number_is_22_l506_50652


namespace NUMINAMATH_GPT_find_x_l506_50662

theorem find_x (a b x : ℝ) (h_a : a > 0) (h_b : b > 0) (h_x : x > 0)
  (s : ℝ) (h_s1 : s = (a ^ 2) ^ (4 * b)) (h_s2 : s = a ^ (2 * b) * x ^ (3 * b)) :
  x = a ^ 2 :=
sorry

end NUMINAMATH_GPT_find_x_l506_50662


namespace NUMINAMATH_GPT_intersection_eq_l506_50611

-- Definitions for M and N
def M : Set ℤ := Set.univ
def N : Set ℤ := {x : ℤ | x^2 - x - 2 < 0}

-- The theorem to be proved
theorem intersection_eq : M ∩ N = {0, 1} := 
  sorry

end NUMINAMATH_GPT_intersection_eq_l506_50611


namespace NUMINAMATH_GPT_mean_score_of_sophomores_l506_50613

open Nat

variable (s j : ℕ)
variable (m m_s m_j : ℝ)

theorem mean_score_of_sophomores :
  (s + j = 150) →
  (m = 85) →
  (j = 80 / 100 * s) →
  (m_s = 125 / 100 * m_j) →
  (s * m_s + j * m_j = 12750) →
  m_s = 94 := by intros; sorry

end NUMINAMATH_GPT_mean_score_of_sophomores_l506_50613


namespace NUMINAMATH_GPT_canoe_kayak_ratio_l506_50661

-- Define the number of canoes and kayaks
variables (c k : ℕ)

-- Define the conditions
def rental_cost_eq : Prop := 15 * c + 18 * k = 405
def canoe_more_kayak_eq : Prop := c = k + 5

-- Statement to prove
theorem canoe_kayak_ratio (h1 : rental_cost_eq c k) (h2 : canoe_more_kayak_eq c k) : c / k = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_canoe_kayak_ratio_l506_50661


namespace NUMINAMATH_GPT_probability_first_three_cards_spades_l506_50676

theorem probability_first_three_cards_spades :
  let num_spades : ℕ := 13
  let total_cards : ℕ := 52
  let prob_first_spade : ℚ := num_spades / total_cards
  let prob_second_spade_given_first : ℚ := (num_spades - 1) / (total_cards - 1)
  let prob_third_spade_given_first_two : ℚ := (num_spades - 2) / (total_cards - 2)
  let prob_all_three_spades : ℚ := prob_first_spade * prob_second_spade_given_first * prob_third_spade_given_first_two
  prob_all_three_spades = 33 / 2550 :=
by
  sorry

end NUMINAMATH_GPT_probability_first_three_cards_spades_l506_50676


namespace NUMINAMATH_GPT_ratio_ad_bc_l506_50617

theorem ratio_ad_bc (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 5 * c) (h3 : c = 3 * d) : 
  (a * d) / (b * c) = 4 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_ad_bc_l506_50617


namespace NUMINAMATH_GPT_monotonic_increase_interval_range_of_a_l506_50685

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x + 2 * Real.exp x - a * x^2
def h (x : ℝ) : ℝ := x

theorem monotonic_increase_interval :
  ∃ I : Set ℝ, I = Set.Ioi 1 ∧ ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f x ≤ f y := 
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, (g x1 a - h x1) * (g x2 a - h x2) > 0) ↔ a ∈ Set.Iic 1 :=
  sorry

end NUMINAMATH_GPT_monotonic_increase_interval_range_of_a_l506_50685


namespace NUMINAMATH_GPT_solution_set_f_gt_0_l506_50646

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2*x - 3 else - (x^2 - 2*x - 3)

theorem solution_set_f_gt_0 :
  {x : ℝ | f x > 0} = {x : ℝ | x > 3 ∨ (-3 < x ∧ x < 0)} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_f_gt_0_l506_50646


namespace NUMINAMATH_GPT_sewer_runoff_capacity_l506_50680

theorem sewer_runoff_capacity (gallons_per_hour : ℕ) (hours_per_day : ℕ) (days_till_overflow : ℕ)
  (h1 : gallons_per_hour = 1000)
  (h2 : hours_per_day = 24)
  (h3 : days_till_overflow = 10) :
  gallons_per_hour * hours_per_day * days_till_overflow = 240000 := 
by
  -- We'll use sorry here as the placeholder for the actual proof steps
  sorry

end NUMINAMATH_GPT_sewer_runoff_capacity_l506_50680


namespace NUMINAMATH_GPT_roller_coaster_cars_l506_50616

theorem roller_coaster_cars (n : ℕ) (h : ((n - 1) : ℝ) / n = 0.5) : n = 2 :=
sorry

end NUMINAMATH_GPT_roller_coaster_cars_l506_50616


namespace NUMINAMATH_GPT_max_touched_points_by_line_l506_50663

noncomputable section

open Function

-- Definitions of the conditions
def coplanar_circles (circles : Set (Set ℝ)) : Prop :=
  ∀ c₁ c₂ : Set ℝ, c₁ ∈ circles → c₂ ∈ circles → c₁ ≠ c₂ → ∃ p : ℝ, p ∈ c₁ ∧ p ∈ c₂

def max_touched_points (line_circle : ℝ → ℝ) : ℕ :=
  2

-- The theorem statement that needs to be proven
theorem max_touched_points_by_line {circles : Set (Set ℝ)} (h_coplanar : coplanar_circles circles) :
  ∀ line : ℝ → ℝ, (∃ (c₁ c₂ c₃ : Set ℝ), c₁ ∈ circles ∧ c₂ ∈ circles ∧ c₃ ∈ circles ∧ c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃) →
  ∃ (p : ℕ), p = 6 := 
sorry

end NUMINAMATH_GPT_max_touched_points_by_line_l506_50663


namespace NUMINAMATH_GPT_greatest_common_divisor_b_81_l506_50630

theorem greatest_common_divisor_b_81 (a b : ℤ) 
  (h : (1 + Real.sqrt 2) ^ 2012 = a + b * Real.sqrt 2) : Int.gcd b 81 = 3 :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_divisor_b_81_l506_50630


namespace NUMINAMATH_GPT_major_airlines_free_snacks_l506_50609

variable (S : ℝ)

theorem major_airlines_free_snacks (h1 : 0.5 ≤ 1) (h2 : 0.5 = 1) :
  0.5 ≤ S :=
sorry

end NUMINAMATH_GPT_major_airlines_free_snacks_l506_50609


namespace NUMINAMATH_GPT_integer_triplet_solution_l506_50690

def circ (a b : ℤ) : ℤ := a + b - a * b

theorem integer_triplet_solution (x y z : ℤ) :
  circ (circ x y) z + circ (circ y z) x + circ (circ z x) y = 0 ↔
  (x = 0 ∧ y = 0 ∧ z = 2) ∨ (x = 0 ∧ y = 2 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end NUMINAMATH_GPT_integer_triplet_solution_l506_50690


namespace NUMINAMATH_GPT_length_of_goods_train_l506_50621

theorem length_of_goods_train (speed_kmph : ℝ) (platform_length : ℝ) (crossing_time : ℝ) (length_of_train : ℝ) :
  speed_kmph = 96 → platform_length = 360 → crossing_time = 32 → length_of_train = (26.67 * 32 - 360) :=
by
  sorry

end NUMINAMATH_GPT_length_of_goods_train_l506_50621


namespace NUMINAMATH_GPT_penultimate_digit_odd_of_square_last_digit_six_l506_50691

theorem penultimate_digit_odd_of_square_last_digit_six 
  (n : ℕ) 
  (h : (n * n) % 10 = 6) : 
  ((n * n) / 10) % 2 = 1 :=
sorry

end NUMINAMATH_GPT_penultimate_digit_odd_of_square_last_digit_six_l506_50691


namespace NUMINAMATH_GPT_find_abc_l506_50635

theorem find_abc (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : c < 4)
  (h4 : a + b + c = a * b * c) : (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
                                 (a = -3 ∧ b = -2 ∧ c = -1) ∨ 
                                 (a = -1 ∧ b = 0 ∧ c = 1) ∨ 
                                 (a = -2 ∧ b = 0 ∧ c = 2) ∨ 
                                 (a = -3 ∧ b = 0 ∧ c = 3) :=
sorry

end NUMINAMATH_GPT_find_abc_l506_50635


namespace NUMINAMATH_GPT_renovate_total_time_eq_79_5_l506_50622

-- Definitions based on the given conditions
def time_per_bedroom : ℝ := 4
def num_bedrooms : ℕ := 3
def time_per_kitchen : ℝ := time_per_bedroom * 1.5
def time_per_garden : ℝ := 3
def time_per_terrace : ℝ := time_per_garden - 2
def time_per_basement : ℝ := time_per_kitchen * 0.75

-- Total time excluding the living room
def total_time_excl_living_room : ℝ :=
  (num_bedrooms * time_per_bedroom) +
  time_per_kitchen +
  time_per_garden +
  time_per_terrace +
  time_per_basement

-- Time for the living room
def time_per_living_room : ℝ := 2 * total_time_excl_living_room

-- Total time for everything
def total_time : ℝ := total_time_excl_living_room + time_per_living_room

-- The theorem we need to prove
theorem renovate_total_time_eq_79_5 : total_time = 79.5 := by
  sorry

end NUMINAMATH_GPT_renovate_total_time_eq_79_5_l506_50622


namespace NUMINAMATH_GPT_negation_proposition_l506_50664

theorem negation_proposition:
  ¬(∃ x : ℝ, x^2 - x + 1 > 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry -- Proof not required as per instructions

end NUMINAMATH_GPT_negation_proposition_l506_50664


namespace NUMINAMATH_GPT_right_angled_triangle_ratio_3_4_5_l506_50623

theorem right_angled_triangle_ratio_3_4_5 : 
  ∀ (a b c : ℕ), 
  (a = 3 * d) → (b = 4 * d) → (c = 5 * d) → (a^2 + b^2 = c^2) :=
by
  intros a b c h1 h2 h3
  sorry

end NUMINAMATH_GPT_right_angled_triangle_ratio_3_4_5_l506_50623


namespace NUMINAMATH_GPT_sum_of_primes_lt_20_eq_77_l506_50672

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end NUMINAMATH_GPT_sum_of_primes_lt_20_eq_77_l506_50672


namespace NUMINAMATH_GPT_profit_percentage_l506_50693

theorem profit_percentage (C S : ℝ) (h : 30 * C = 24 * S) :
  (S - C) / C * 100 = 25 :=
by sorry

end NUMINAMATH_GPT_profit_percentage_l506_50693


namespace NUMINAMATH_GPT_plums_added_l506_50606

-- Definitions of initial and final plum counts
def initial_plums : ℕ := 17
def final_plums : ℕ := 21

-- The mathematical statement to be proved
theorem plums_added (initial_plums final_plums : ℕ) : final_plums - initial_plums = 4 := by
  -- The proof will be inserted here
  sorry

end NUMINAMATH_GPT_plums_added_l506_50606


namespace NUMINAMATH_GPT_max_sum_squares_of_sides_l506_50632

theorem max_sum_squares_of_sides
  (a : ℝ) (α : ℝ) 
  (hα1 : 0 < α) (hα2 : α < Real.pi / 2) : 
  ∃ b c : ℝ, b^2 + c^2 = a^2 / (1 - Real.cos α) := 
sorry

end NUMINAMATH_GPT_max_sum_squares_of_sides_l506_50632


namespace NUMINAMATH_GPT_carbonate_ions_in_Al2_CO3_3_l506_50612

theorem carbonate_ions_in_Al2_CO3_3 (total_weight : ℕ) (formula : String) 
  (molecular_weight : ℕ) (ions_in_formula : String) : 
  formula = "Al2(CO3)3" → molecular_weight = 234 → ions_in_formula = "CO3" → total_weight = 3 := 
by
  intros formula_eq weight_eq ions_eq
  sorry

end NUMINAMATH_GPT_carbonate_ions_in_Al2_CO3_3_l506_50612


namespace NUMINAMATH_GPT_ultramindmaster_secret_codes_count_l506_50694

/-- 
In the game UltraMindmaster, we need to find the total number of possible secret codes 
formed by placing pegs of any of eight different colors into five slots.
Colors may be repeated, and each slot must be filled.
-/
theorem ultramindmaster_secret_codes_count :
  let colors := 8
  let slots := 5
  colors ^ slots = 32768 := by
    sorry

end NUMINAMATH_GPT_ultramindmaster_secret_codes_count_l506_50694


namespace NUMINAMATH_GPT_minimum_rows_l506_50601

theorem minimum_rows (n : ℕ) (C : ℕ → ℕ) (hC_bounds : ∀ i, 1 ≤ C i ∧ C i ≤ 39) 
  (hC_sum : (Finset.range n).sum C = 1990) :
  ∃ k, k = 12 ∧ ∀ (R : ℕ) (hR : R = 199), 
    ∀ (seating : ℕ → ℕ) (h_seating : ∀ i, seating i ≤ R) 
    (h_seating_capacity : (Finset.range k).sum seating = 1990),
    True := sorry

end NUMINAMATH_GPT_minimum_rows_l506_50601


namespace NUMINAMATH_GPT_circle_equation_and_range_of_a_l506_50649

theorem circle_equation_and_range_of_a :
  (∃ m : ℤ, (x - m)^2 + y^2 = 25 ∧ (abs (4 * m - 29)) = 25) ∧
  (∀ a : ℝ, (a > 0 → (4 * (5 * a - 1)^2 - 4 * (a^2 + 1) > 0 → a > 5 / 12 ∨ a < 0))) :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_and_range_of_a_l506_50649


namespace NUMINAMATH_GPT_find_common_chord_l506_50696

variable (x y : ℝ)

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 3*y = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*x + 2*y + 1 = 0
def common_chord (x y : ℝ) := 6*x + y - 1 = 0

theorem find_common_chord (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : common_chord x y :=
by
  sorry

end NUMINAMATH_GPT_find_common_chord_l506_50696


namespace NUMINAMATH_GPT_value_of_a_plus_b_l506_50674

variable (a b : ℝ)
variable (h1 : |a| = 5)
variable (h2 : |b| = 2)
variable (h3 : a < 0)
variable (h4 : b > 0)

theorem value_of_a_plus_b : a + b = -3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l506_50674


namespace NUMINAMATH_GPT_find_number_l506_50604

-- Define the condition that one-third of a certain number is 300% of 134
def one_third_eq_300percent_number (n : ℕ) : Prop :=
  n / 3 = 3 * 134

-- State the theorem that the number is 1206 given the above condition
theorem find_number (n : ℕ) (h : one_third_eq_300percent_number n) : n = 1206 :=
  by sorry

end NUMINAMATH_GPT_find_number_l506_50604


namespace NUMINAMATH_GPT_sequence_a_2024_l506_50655

theorem sequence_a_2024 (a : ℕ → ℝ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = 1 - 1 / a n) : a 2024 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_2024_l506_50655


namespace NUMINAMATH_GPT_increasing_function_geq_25_l506_50671

theorem increasing_function_geq_25 {m : ℝ} 
  (h : ∀ x y : ℝ, x ≥ -2 ∧ x ≤ y → (4 * x^2 - m * x + 5) ≤ (4 * y^2 - m * y + 5)) :
  (4 * 1^2 - m * 1 + 5) ≥ 25 :=
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_increasing_function_geq_25_l506_50671


namespace NUMINAMATH_GPT_original_profit_percentage_l506_50605

theorem original_profit_percentage
  (P SP : ℝ)
  (h1 : SP = 549.9999999999995)
  (h2 : SP = P * (1 + x / 100))
  (h3 : 0.9 * P * 1.3 = SP + 35) :
  x = 10 := 
sorry

end NUMINAMATH_GPT_original_profit_percentage_l506_50605


namespace NUMINAMATH_GPT_quadrilateral_angle_contradiction_l506_50699

theorem quadrilateral_angle_contradiction (a b c d : ℝ)
  (h : 0 < a ∧ a < 180 ∧ 0 < b ∧ b < 180 ∧ 0 < c ∧ c < 180 ∧ 0 < d ∧ d < 180)
  (sum_eq_360 : a + b + c + d = 360) :
  (¬ (a ≤ 90 ∨ b ≤ 90 ∨ c ≤ 90 ∨ d ≤ 90)) → (90 < a ∧ 90 < b ∧ 90 < c ∧ 90 < d) :=
sorry

end NUMINAMATH_GPT_quadrilateral_angle_contradiction_l506_50699


namespace NUMINAMATH_GPT_tourists_speeds_l506_50687

theorem tourists_speeds (x y : ℝ) :
  (20 / x + 2.5 = 20 / y) →
  (20 / (x - 2) = 20 / (1.5 * y)) →
  x = 8 ∧ y = 4 :=
by
  intros h1 h2
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_tourists_speeds_l506_50687


namespace NUMINAMATH_GPT_horner_operations_count_l506_50688

def polynomial (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

def horner_polynomial (x : ℝ) := (((((5*x + 4)*x + 3)*x + 2)*x + 1)*x + 1)

theorem horner_operations_count (x : ℝ) : 
    (polynomial x = horner_polynomial x) → 
    (x = 2) → 
    (mul_ops : ℕ) = 5 → 
    (add_ops : ℕ) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_horner_operations_count_l506_50688


namespace NUMINAMATH_GPT_total_games_l506_50683

def joan_games_this_year : ℕ := 4
def joan_games_last_year : ℕ := 9

theorem total_games (this_year_games last_year_games : ℕ) 
    (h1 : this_year_games = joan_games_this_year) 
    (h2 : last_year_games = joan_games_last_year) : 
    this_year_games + last_year_games = 13 := 
by
  rw [h1, h2]
  exact rfl

end NUMINAMATH_GPT_total_games_l506_50683


namespace NUMINAMATH_GPT_set_intersection_M_N_l506_50618

theorem set_intersection_M_N (x : ℝ) :
  let M := {x | -4 < x ∧ x < -2}
  let N := {x | x^2 + 5*x + 6 < 0}
  M ∩ N = {x | -3 < x ∧ x < -2} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_M_N_l506_50618


namespace NUMINAMATH_GPT_cube_divided_by_five_tetrahedrons_l506_50619

-- Define the minimum number of tetrahedrons needed to divide a cube
def min_tetrahedrons_to_divide_cube : ℕ := 5

-- State the theorem
theorem cube_divided_by_five_tetrahedrons : min_tetrahedrons_to_divide_cube = 5 :=
by
  -- The proof is skipped, as instructed
  sorry

end NUMINAMATH_GPT_cube_divided_by_five_tetrahedrons_l506_50619


namespace NUMINAMATH_GPT_total_price_for_pizza_l506_50658

-- Definitions based on conditions
def num_friends : ℕ := 5
def amount_per_person : ℕ := 8

-- The claim to be proven
theorem total_price_for_pizza : num_friends * amount_per_person = 40 := by
  -- Since the proof detail is not required, we use 'sorry' to skip the proof.
  sorry

end NUMINAMATH_GPT_total_price_for_pizza_l506_50658


namespace NUMINAMATH_GPT_remainder_3_pow_17_mod_5_l506_50639

theorem remainder_3_pow_17_mod_5 :
  (3^17) % 5 = 3 :=
by
  have h : 3^4 % 5 = 1 := by norm_num
  sorry

end NUMINAMATH_GPT_remainder_3_pow_17_mod_5_l506_50639


namespace NUMINAMATH_GPT_total_legs_l506_50637

theorem total_legs 
  (johnny_legs : ℕ := 2) 
  (son_legs : ℕ := 2) 
  (dog_legs_per_dog : ℕ := 4) 
  (number_of_dogs : ℕ := 2) :
  johnny_legs + son_legs + dog_legs_per_dog * number_of_dogs = 12 := 
sorry

end NUMINAMATH_GPT_total_legs_l506_50637


namespace NUMINAMATH_GPT_graph_intersection_points_l506_50610

open Function

theorem graph_intersection_points (g : ℝ → ℝ) (h_inv : Involutive (invFun g)) : 
  ∃! (x : ℝ), x = 0 ∨ x = 1 ∨ x = -1 → g (x^2) = g (x^6) :=
by sorry

end NUMINAMATH_GPT_graph_intersection_points_l506_50610


namespace NUMINAMATH_GPT_commission_rate_correct_l506_50644

-- Define the given conditions
def base_pay := 190
def goal_earnings := 500
def required_sales := 7750

-- Define the commission rate function
def commission_rate (sales commission : ℕ) : ℚ := (commission : ℚ) / (sales : ℚ) * 100

-- The main statement to prove
theorem commission_rate_correct :
  commission_rate required_sales (goal_earnings - base_pay) = 4 :=
by
  sorry

end NUMINAMATH_GPT_commission_rate_correct_l506_50644


namespace NUMINAMATH_GPT_range_of_m_l506_50633

theorem range_of_m (m : ℝ) 
  (hp : ∀ x : ℝ, 2 * x > m * (x ^ 2 + 1)) 
  (hq : ∃ x0 : ℝ, x0 ^ 2 + 2 * x0 - m - 1 = 0) : 
  -2 ≤ m ∧ m < -1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l506_50633
