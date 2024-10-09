import Mathlib

namespace number_of_dimes_l1904_190484

theorem number_of_dimes (d q : ℕ) (h₁ : 10 * d + 25 * q = 580) (h₂ : d = q + 10) : d = 23 := 
by 
  sorry

end number_of_dimes_l1904_190484


namespace six_letter_word_combinations_l1904_190481

theorem six_letter_word_combinations : ∃ n : ℕ, n = 26 * 26 * 26 := 
sorry

end six_letter_word_combinations_l1904_190481


namespace shares_sum_4000_l1904_190485

variables (w x y z : ℝ)

def relation_z_w : Prop := z = 1.20 * w
def relation_y_z : Prop := y = 1.25 * z
def relation_x_y : Prop := x = 1.35 * y
def w_after_3_years : ℝ := 8 * w
def z_after_3_years : ℝ := 8 * z
def y_after_3_years : ℝ := 8 * y
def x_after_3_years : ℝ := 8 * x

theorem shares_sum_4000 (w : ℝ) :
  relation_z_w w z →
  relation_y_z z y →
  relation_x_y y x →
  x_after_3_years x + y_after_3_years y + z_after_3_years z + w_after_3_years w = 4000 :=
by
  intros h_z_w h_y_z h_x_y
  rw [relation_z_w, relation_y_z, relation_x_y] at *
  sorry

end shares_sum_4000_l1904_190485


namespace Y_pdf_from_X_pdf_l1904_190487

/-- Given random variable X with PDF p(x), prove PDF of Y = X^3 -/
noncomputable def X_pdf (σ : ℝ) (x : ℝ) : ℝ := 
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x ^ 2) / (2 * σ ^ 2))

noncomputable def Y_pdf (σ : ℝ) (y : ℝ) : ℝ :=
  (1 / (3 * σ * y^(2/3) * Real.sqrt (2 * Real.pi))) * Real.exp (-(y^(2/3)) / (2 * σ ^ 2))

theorem Y_pdf_from_X_pdf (σ : ℝ) (y : ℝ) :
  ∀ x : ℝ, X_pdf σ x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x ^ 2) / (2 * σ ^ 2)) →
  Y_pdf σ y = (1 / (3 * σ * y^(2/3) * Real.sqrt (2 * Real.pi))) * Real.exp (-(y^(2/3)) / (2 * σ ^ 2)) :=
sorry

end Y_pdf_from_X_pdf_l1904_190487


namespace negation_of_both_even_l1904_190496

-- Definitions
def even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Main statement
theorem negation_of_both_even (a b : ℕ) : ¬ (even a ∧ even b) ↔ (¬even a ∨ ¬even b) :=
by sorry

end negation_of_both_even_l1904_190496


namespace solve_for_y_l1904_190440

theorem solve_for_y : (12^3 * 6^2) / 432 = 144 := 
by 
  sorry

end solve_for_y_l1904_190440


namespace green_apples_more_than_red_apples_l1904_190416

noncomputable def num_original_green_apples : ℕ := 32
noncomputable def num_more_red_apples_than_green : ℕ := 200
noncomputable def num_delivered_green_apples : ℕ := 340
noncomputable def num_original_red_apples : ℕ :=
  num_original_green_apples + num_more_red_apples_than_green
noncomputable def num_new_green_apples : ℕ :=
  num_original_green_apples + num_delivered_green_apples

theorem green_apples_more_than_red_apples :
  num_new_green_apples - num_original_red_apples = 140 :=
by {
  sorry
}

end green_apples_more_than_red_apples_l1904_190416


namespace expand_and_simplify_product_l1904_190434

variable (x : ℝ)

theorem expand_and_simplify_product :
  (x^2 + 3*x - 4) * (x^2 - 5*x + 6) = x^4 - 2*x^3 - 13*x^2 + 38*x - 24 :=
by
  sorry

end expand_and_simplify_product_l1904_190434


namespace inequality_solution_l1904_190459

theorem inequality_solution (x : ℝ) (h : x ≠ 0) : 
  (1 / (x^2 + 1) > 2 * x^2 / x + 13 / 10) ↔ (x ∈ Set.Ioo (-1.6) 0 ∨ x ∈ Set.Ioi 0.8) :=
by sorry

end inequality_solution_l1904_190459


namespace minimize_distance_sum_l1904_190402

open Real

noncomputable def distance_squared (x y : ℝ × ℝ) : ℝ :=
  (x.1 - y.1)^2 + (x.2 - y.2)^2

theorem minimize_distance_sum : 
  ∀ P : ℝ × ℝ, (P.1 = P.2) → 
    let A : ℝ × ℝ := (1, -1)
    let B : ℝ × ℝ := (2, 2)
    (distance_squared P A + distance_squared P B) ≥ 
    (distance_squared (1, 1) A + distance_squared (1, 1) B) := by
  intro P hP
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (2, 2)
  sorry

end minimize_distance_sum_l1904_190402


namespace voldemort_calorie_intake_limit_l1904_190491

theorem voldemort_calorie_intake_limit :
  let breakfast := 560
  let lunch := 780
  let cake := 110
  let chips := 310
  let coke := 215
  let dinner := cake + chips + coke
  let remaining := 525
  breakfast + lunch + dinner + remaining = 2500 :=
by
  -- to clarify, the statement alone is provided, so we add 'sorry' to omit the actual proof steps
  sorry

end voldemort_calorie_intake_limit_l1904_190491


namespace unique_zero_of_f_l1904_190447

theorem unique_zero_of_f (f : ℝ → ℝ) (h1 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 16) 
  (h2 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 8) (h3 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 4) 
  (h4 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 2) : ¬ ∃ x, f x = 0 ∧ 2 ≤ x ∧ x < 16 := 
by
  sorry

end unique_zero_of_f_l1904_190447


namespace combine_expr_l1904_190441

variable (a b : ℝ)

theorem combine_expr : 3 * (2 * a - 3 * b) - 6 * (a - b) = -3 * b := by
  sorry

end combine_expr_l1904_190441


namespace inscribed_sphere_radius_l1904_190464

noncomputable def radius_inscribed_sphere (S1 S2 S3 S4 V : ℝ) : ℝ :=
  3 * V / (S1 + S2 + S3 + S4)

theorem inscribed_sphere_radius (S1 S2 S3 S4 V R : ℝ) :
  R = radius_inscribed_sphere S1 S2 S3 S4 V :=
by
  sorry

end inscribed_sphere_radius_l1904_190464


namespace find_value_of_a_squared_b_plus_ab_squared_l1904_190405

theorem find_value_of_a_squared_b_plus_ab_squared 
  (a b : ℝ) 
  (h1 : a + b = -3) 
  (h2 : ab = 2) : 
  a^2 * b + a * b^2 = -6 :=
by 
  sorry

end find_value_of_a_squared_b_plus_ab_squared_l1904_190405


namespace matrix_pow_three_l1904_190425

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_pow_three :
  A^3 = !![-4, 2; -2, 1] := by
  sorry

end matrix_pow_three_l1904_190425


namespace delta_value_l1904_190452

theorem delta_value (Δ : ℝ) (h : 4 * 3 = Δ - 6) : Δ = 18 :=
sorry

end delta_value_l1904_190452


namespace problem_l1904_190411

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x + 2
noncomputable def f' (a x : ℝ) : ℝ := a * (Real.log x + 1) + 1
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x - x^2 - (a + 2) * x + a

theorem problem (a x : ℝ) (h : 1 ≤ x) (ha : 0 < a) : f' a x < x^2 + (a + 2) * x + 1 :=
by
  sorry

end problem_l1904_190411


namespace find_c_l1904_190472

-- Given that the function f(x) = 2^x + c passes through the point (2,5),
-- Prove that c = 1.
theorem find_c (c : ℝ) : (∃ (f : ℝ → ℝ), (∀ x, f x = 2^x + c) ∧ (f 2 = 5)) → c = 1 := by
  sorry

end find_c_l1904_190472


namespace y_intercept_of_line_l1904_190427

theorem y_intercept_of_line : ∀ (x y : ℝ), (5 * x - 2 * y - 10 = 0) → (x = 0) → (y = -5) :=
by
  intros x y h1 h2
  sorry

end y_intercept_of_line_l1904_190427


namespace find_f_l1904_190444

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, f (Real.sqrt x + 4) = x + 8 * Real.sqrt x) :
  ∀ (x : ℝ), x ≥ 4 → f x = x^2 - 16 :=
by
  sorry

end find_f_l1904_190444


namespace expr_value_l1904_190446

theorem expr_value : 2 ^ (1 + 2 + 3) - (2 ^ 1 + 2 ^ 2 + 2 ^ 3) = 50 :=
by
  sorry

end expr_value_l1904_190446


namespace simplify_complex_expression_l1904_190497

variables (x y : ℝ) (i : ℂ)

theorem simplify_complex_expression (h : i^2 = -1) :
  (x + 3 * i * y) * (x - 3 * i * y) = x^2 - 9 * y^2 :=
by sorry

end simplify_complex_expression_l1904_190497


namespace Roe_total_savings_l1904_190477

-- Define savings amounts per period
def savings_Jan_to_Jul : Int := 7 * 10
def savings_Aug_to_Nov : Int := 4 * 15
def savings_Dec : Int := 20

-- Define total savings for the year
def total_savings : Int := savings_Jan_to_Jul + savings_Aug_to_Nov + savings_Dec

-- Prove that Roe's total savings for the year is $150
theorem Roe_total_savings : total_savings = 150 := by
  -- Proof goes here
  sorry

end Roe_total_savings_l1904_190477


namespace feeding_ways_correct_l1904_190476

def total_feeding_ways : Nat :=
  (5 * 6 * (5 * 4 * 3 * 2 * 1)^2)

theorem feeding_ways_correct :
  total_feeding_ways = 432000 :=
by
  -- Proof is omitted here
  sorry

end feeding_ways_correct_l1904_190476


namespace number_of_children_coming_to_show_l1904_190456

theorem number_of_children_coming_to_show :
  ∀ (cost_adult cost_child : ℕ) (number_adults total_cost : ℕ),
  cost_adult = 12 →
  cost_child = 10 →
  number_adults = 3 →
  total_cost = 66 →
  ∃ (c : ℕ), 3 = c := by
    sorry

end number_of_children_coming_to_show_l1904_190456


namespace roots_quadratic_l1904_190409

theorem roots_quadratic (m x₁ x₂ : ℝ) (h : m < 0) (h₁ : x₁ < x₂) (hx : ∀ x, (x^2 - x - 6 = m) ↔ (x = x₁ ∨ x = x₂)) : 
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 :=
by {
  sorry
}

end roots_quadratic_l1904_190409


namespace speed_of_B_is_three_l1904_190432

noncomputable def speed_of_B (rounds_per_hour : ℕ) : Prop :=
  let A_speed : ℕ := 2
  let crossings : ℕ := 5
  let time_hours : ℕ := 1
  rounds_per_hour = (crossings - A_speed)

theorem speed_of_B_is_three : speed_of_B 3 :=
  sorry

end speed_of_B_is_three_l1904_190432


namespace problem_statement_l1904_190408

-- Define what it means to be a quadratic equation
def is_quadratic (eqn : String) : Prop :=
  -- In the context of this solution, we'll define a quadratic equation as one
  -- that fits the form ax^2 + bx + c = 0 where a, b, c are constants and a ≠ 0.
  eqn = "x^2 - 2 = 0"

-- We need to formulate a theorem that checks the validity of which equation is quadratic.
theorem problem_statement :
  is_quadratic "x^2 - 2 = 0" :=
sorry

end problem_statement_l1904_190408


namespace evaluate_expression_l1904_190465

variable (y : ℕ)

theorem evaluate_expression (h : y = 3) : 
    (y^(1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19) / y^(2 + 4 + 6 + 8 + 10 + 12)) = 3^58 :=
by
  -- Proof will be done here
  sorry

end evaluate_expression_l1904_190465


namespace difference_even_odd_sums_l1904_190461

def sum_first_n_even_numbers (n : ℕ) : ℕ := n * (n + 1)
def sum_first_n_odd_numbers (n : ℕ) : ℕ := n * n

theorem difference_even_odd_sums : sum_first_n_even_numbers 1001 - sum_first_n_odd_numbers 1001 = 1001 := by
  sorry

end difference_even_odd_sums_l1904_190461


namespace lisa_punch_l1904_190486

theorem lisa_punch (x : ℝ) (H : x = 0.125) :
  (0.3 + x) / (2 + x) = 0.20 :=
by
  sorry

end lisa_punch_l1904_190486


namespace base_seven_sum_of_digits_of_product_l1904_190400

theorem base_seven_sum_of_digits_of_product :
  let a := 24
  let b := 30
  let product := a * b
  let base_seven_product := 105 -- The product in base seven notation
  let sum_of_digits (n : ℕ) : ℕ := n.digits 7 |> List.sum
  sum_of_digits base_seven_product = 6 :=
by
  sorry

end base_seven_sum_of_digits_of_product_l1904_190400


namespace solve_y_determinant_l1904_190443

theorem solve_y_determinant (b y : ℝ) (hb : b ≠ 0) :
  Matrix.det ![
    ![y + b, y, y], 
    ![y, y + b, y], 
    ![y, y, y + b]
  ] = 0 ↔ y = -b / 3 :=
by
  sorry

end solve_y_determinant_l1904_190443


namespace power_sum_l1904_190431

theorem power_sum
: (-2)^(2005) + (-2)^(2006) = 2^(2005) := by
  sorry

end power_sum_l1904_190431


namespace distance_is_20_sqrt_6_l1904_190439

-- Definitions for problem setup
def distance_between_parallel_lines (r d : ℝ) : Prop :=
  ∃ O C D E F P Q : ℝ, 
  40^2 * 40 + (d / 2)^2 * 40 = 40 * r^2 ∧ 
  15^2 * 30 + (d / 2)^2 * 30 = 30 * r^2

-- The main statement to be proved
theorem distance_is_20_sqrt_6 :
  ∀ r d : ℝ,
  distance_between_parallel_lines r d →
  d = 20 * Real.sqrt 6 :=
sorry

end distance_is_20_sqrt_6_l1904_190439


namespace february_first_day_of_week_l1904_190457

theorem february_first_day_of_week 
  (feb13_is_wednesday : ∃ day, day = 13 ∧ day_of_week = "Wednesday") :
  ∃ day, day = 1 ∧ day_of_week = "Friday" :=
sorry

end february_first_day_of_week_l1904_190457


namespace cos_double_angle_l1904_190475

theorem cos_double_angle (k : ℝ) (h : Real.sin (10 * Real.pi / 180) = k) : 
  Real.cos (20 * Real.pi / 180) = 1 - 2 * k^2 := by
  sorry

end cos_double_angle_l1904_190475


namespace fraction_sum_is_five_l1904_190406

noncomputable def solve_fraction_sum (x y z : ℝ) : Prop :=
  (x + 1/y = 5) ∧ (y + 1/z = 2) ∧ (z + 1/x = 3) ∧ 0 < x ∧ 0 < y ∧ 0 < z → 
  (x / y + y / z + z / x = 5)
    
theorem fraction_sum_is_five (x y z : ℝ) : solve_fraction_sum x y z :=
  sorry

end fraction_sum_is_five_l1904_190406


namespace parallelogram_is_central_not_axis_symmetric_l1904_190426

-- Definitions for the shapes discussed in the problem
def is_central_symmetric (shape : Type) : Prop := sorry
def is_axis_symmetric (shape : Type) : Prop := sorry

-- Specific shapes being used in the problem
def rhombus : Type := sorry
def parallelogram : Type := sorry
def equilateral_triangle : Type := sorry
def rectangle : Type := sorry

-- Example additional assumptions about shapes can be added here if needed

-- The problem assertion
theorem parallelogram_is_central_not_axis_symmetric :
  is_central_symmetric parallelogram ∧ ¬ is_axis_symmetric parallelogram :=
sorry

end parallelogram_is_central_not_axis_symmetric_l1904_190426


namespace rotated_intersection_point_l1904_190473

theorem rotated_intersection_point (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : 
  ∃ P : ℝ × ℝ, P = (-Real.sin θ, Real.cos θ) ∧ 
    ∃ φ : ℝ, φ = θ + π / 2 ∧ 
      P = (Real.cos φ, Real.sin φ) := 
by
  sorry

end rotated_intersection_point_l1904_190473


namespace green_caps_percentage_l1904_190478

variable (total_caps : ℕ) (red_caps : ℕ)

def green_caps (total_caps red_caps: ℕ) : ℕ :=
  total_caps - red_caps

def percentage_of_green_caps (total_caps green_caps: ℕ) : ℕ :=
  (green_caps * 100) / total_caps

theorem green_caps_percentage :
  (total_caps = 125) →
  (red_caps = 50) →
  percentage_of_green_caps total_caps (green_caps total_caps red_caps) = 60 :=
by
  intros h1 h2
  rw [h1, h2]
  exact sorry  -- The proof is omitted 

end green_caps_percentage_l1904_190478


namespace total_area_of_combined_shape_l1904_190467

theorem total_area_of_combined_shape
  (length_rectangle : ℝ) (width_rectangle : ℝ) (side_square : ℝ)
  (h_length : length_rectangle = 0.45)
  (h_width : width_rectangle = 0.25)
  (h_side : side_square = 0.15) :
  (length_rectangle * width_rectangle + side_square * side_square) = 0.135 := 
by 
  sorry

end total_area_of_combined_shape_l1904_190467


namespace sum_of_decimals_as_fraction_l1904_190498

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 :=
by
  sorry

end sum_of_decimals_as_fraction_l1904_190498


namespace exists_infinitely_many_n_with_increasing_ω_l1904_190423

open Nat

/--
  Let ω(n) represent the number of distinct prime factors of a natural number n (where n > 1).
  Prove that there exist infinitely many n such that ω(n) < ω(n + 1) < ω(n + 2).
-/
theorem exists_infinitely_many_n_with_increasing_ω (ω : ℕ → ℕ) (hω : ∀ (n : ℕ), n > 1 → ∃ k, ω k < ω (k + 1) ∧ ω (k + 1) < ω (k + 2)) :
  ∃ (infinitely_many : ℕ → Prop), ∀ N : ℕ, ∃ n : ℕ, N < n ∧ infinitely_many n :=
by
  sorry

end exists_infinitely_many_n_with_increasing_ω_l1904_190423


namespace bike_helmet_cost_increase_l1904_190428

open Real

theorem bike_helmet_cost_increase :
  let old_bike_cost := 150
  let old_helmet_cost := 50
  let new_bike_cost := old_bike_cost + 0.10 * old_bike_cost
  let new_helmet_cost := old_helmet_cost + 0.20 * old_helmet_cost
  let old_total_cost := old_bike_cost + old_helmet_cost
  let new_total_cost := new_bike_cost + new_helmet_cost
  let total_increase := new_total_cost - old_total_cost
  let percent_increase := (total_increase / old_total_cost) * 100
  percent_increase = 12.5 :=
by
  sorry

end bike_helmet_cost_increase_l1904_190428


namespace bicycle_owners_no_car_l1904_190437

-- Definitions based on the conditions in (a)
def total_adults : ℕ := 500
def bicycle_owners : ℕ := 450
def car_owners : ℕ := 120
def both_owners : ℕ := bicycle_owners + car_owners - total_adults

-- Proof problem statement
theorem bicycle_owners_no_car : (bicycle_owners - both_owners = 380) :=
by
  -- Placeholder proof
  sorry

end bicycle_owners_no_car_l1904_190437


namespace locus_of_midpoint_l1904_190403

theorem locus_of_midpoint (x y : ℝ) (h : y ≠ 0) :
  (∃ P : ℝ × ℝ, P = (2*x, 2*y) ∧ ((P.1^2 + (P.2-3)^2 = 9))) →
  (x^2 + (y - 3/2)^2 = 9/4) :=
by
  sorry

end locus_of_midpoint_l1904_190403


namespace bowling_ball_weight_l1904_190419

variable {b c : ℝ}

theorem bowling_ball_weight :
  (10 * b = 4 * c) ∧ (3 * c = 108) → b = 14.4 :=
by
  sorry

end bowling_ball_weight_l1904_190419


namespace min_discount_70_percent_l1904_190482

theorem min_discount_70_percent
  (P S : ℝ) (M : ℝ)
  (hP : P = 800)
  (hS : S = 1200)
  (hM : M = 0.05) :
  ∃ D : ℝ, D = 0.7 ∧ S * D - P ≥ P * M :=
by sorry

end min_discount_70_percent_l1904_190482


namespace compare_fractions_l1904_190414

theorem compare_fractions : (-2 / 7) > (-3 / 10) :=
sorry

end compare_fractions_l1904_190414


namespace smallest_number_of_three_l1904_190401

theorem smallest_number_of_three (a b c : ℕ) (h1 : a + b + c = 78) (h2 : b = 27) (h3 : c = b + 5) :
  a = 19 :=
by
  sorry

end smallest_number_of_three_l1904_190401


namespace fraction_spent_on_museum_ticket_l1904_190412

theorem fraction_spent_on_museum_ticket (initial_money : ℝ) (sandwich_fraction : ℝ) (book_fraction : ℝ) (remaining_money : ℝ) (h1 : initial_money = 90) (h2 : sandwich_fraction = 1/5) (h3 : book_fraction = 1/2) (h4 : remaining_money = 12) : (initial_money - remaining_money) / initial_money - (sandwich_fraction * initial_money + book_fraction * initial_money) / initial_money = 1/6 :=
by
  sorry

end fraction_spent_on_museum_ticket_l1904_190412


namespace find_equation_with_new_roots_l1904_190407

variable {p q r s : ℝ}

theorem find_equation_with_new_roots 
  (h_eq : ∀ x, x^2 - p * x + q = 0 ↔ (x = r ∧ x = s))
  (h_r_nonzero : r ≠ 0)
  (h_s_nonzero : s ≠ 0)
  : 
  ∀ x, (x^2 - ((q^2 + 1) * (p^2 - 2 * q) / q^2) * x + (q + 1/q)^2) = 0 ↔ 
       (x = r^2 + 1/(s^2) ∧ x = s^2 + 1/(r^2)) := 
sorry

end find_equation_with_new_roots_l1904_190407


namespace sum_of_series_l1904_190450

theorem sum_of_series : 
  (6 + 16 + 26 + 36 + 46) + (14 + 24 + 34 + 44 + 54) = 300 :=
by
  sorry

end sum_of_series_l1904_190450


namespace find_angle_C_l1904_190490

variable {A B C : ℝ} -- Angles of triangle ABC
variable {a b c : ℝ} -- Sides opposite the respective angles

theorem find_angle_C
  (h1 : 2 * c * Real.cos B = 2 * a + b) : 
  C = 120 :=
  sorry

end find_angle_C_l1904_190490


namespace power_sum_zero_l1904_190493

theorem power_sum_zero (n : ℕ) (h : 0 < n) : (-1:ℤ)^(2*n) + (-1:ℤ)^(2*n+1) = 0 := 
by 
  sorry

end power_sum_zero_l1904_190493


namespace quadratic_eq_has_nonzero_root_l1904_190436

theorem quadratic_eq_has_nonzero_root (b c : ℝ) (h : c ≠ 0) (h_eq : c^2 + b * c + c = 0) : b + c = -1 :=
sorry

end quadratic_eq_has_nonzero_root_l1904_190436


namespace joan_total_spending_l1904_190489

def basketball_game_price : ℝ := 5.20
def basketball_game_discount : ℝ := 0.15 * basketball_game_price
def basketball_game_discounted : ℝ := basketball_game_price - basketball_game_discount

def racing_game_price : ℝ := 4.23
def racing_game_discount : ℝ := 0.10 * racing_game_price
def racing_game_discounted : ℝ := racing_game_price - racing_game_discount

def puzzle_game_price : ℝ := 3.50

def total_before_tax : ℝ := basketball_game_discounted + racing_game_discounted + puzzle_game_price
def sales_tax : ℝ := 0.08 * total_before_tax
def total_with_tax : ℝ := total_before_tax + sales_tax

theorem joan_total_spending : (total_with_tax : ℝ) = 12.67 := by
  sorry

end joan_total_spending_l1904_190489


namespace hyperbola_asymptotes_l1904_190462

theorem hyperbola_asymptotes :
  ∀ {x y : ℝ},
    (x^2 / 9 - y^2 / 16 = 1) →
    (y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intro x y h
  sorry

end hyperbola_asymptotes_l1904_190462


namespace runway_show_duration_l1904_190433

theorem runway_show_duration
  (evening_wear_time : ℝ) (bathing_suits_time : ℝ) (formal_wear_time : ℝ) (casual_wear_time : ℝ)
  (evening_wear_sets : ℕ) (bathing_suits_sets : ℕ) (formal_wear_sets : ℕ) (casual_wear_sets : ℕ)
  (num_models : ℕ) :
  evening_wear_time = 4 → bathing_suits_time = 2 → formal_wear_time = 3 → casual_wear_time = 2.5 →
  evening_wear_sets = 4 → bathing_suits_sets = 2 → formal_wear_sets = 3 → casual_wear_sets = 5 →
  num_models = 10 →
  (evening_wear_time * evening_wear_sets + bathing_suits_time * bathing_suits_sets
   + formal_wear_time * formal_wear_sets + casual_wear_time * casual_wear_sets) * num_models = 415 :=
by
  intros
  sorry

end runway_show_duration_l1904_190433


namespace largest_fully_communicating_sets_eq_l1904_190460

noncomputable def largest_fully_communicating_sets :=
  let total_sets := Nat.choose 99 4
  let non_communicating_sets_per_pod := Nat.choose 48 3
  let total_non_communicating_sets := 99 * non_communicating_sets_per_pod
  total_sets - total_non_communicating_sets

theorem largest_fully_communicating_sets_eq : largest_fully_communicating_sets = 2051652 := by
  sorry

end largest_fully_communicating_sets_eq_l1904_190460


namespace smallest_positive_angle_l1904_190404

theorem smallest_positive_angle (deg : ℤ) (k : ℤ) (h : deg = -2012) : ∃ m : ℤ, m = 148 ∧ 0 ≤ m ∧ m < 360 ∧ (∃ n : ℤ, deg + 360 * n = m) :=
by
  sorry

end smallest_positive_angle_l1904_190404


namespace Jan_older_than_Cindy_l1904_190429

noncomputable def Cindy_age : ℕ := 5
noncomputable def Greg_age : ℕ := 16

variables (Marcia_age Jan_age : ℕ)

axiom Greg_and_Marcia : Greg_age = Marcia_age + 2
axiom Marcia_and_Jan : Marcia_age = 2 * Jan_age

theorem Jan_older_than_Cindy : (Jan_age - Cindy_age) = 2 :=
by
  -- Insert proof here
  sorry

end Jan_older_than_Cindy_l1904_190429


namespace reciprocals_sum_of_roots_l1904_190495

theorem reciprocals_sum_of_roots (r s γ δ : ℚ) (h1 : 7 * r^2 + 5 * r + 3 = 0) (h2 : 7 * s^2 + 5 * s + 3 = 0) (h3 : γ = 1/r) (h4 : δ = 1/s) :
  γ + δ = -5/3 := 
  by 
    sorry

end reciprocals_sum_of_roots_l1904_190495


namespace cyclic_quadrilateral_angles_l1904_190415

theorem cyclic_quadrilateral_angles (ABCD_cyclic : True) (P_interior : True)
  (x y z t : ℝ) (h1 : x + y + z + t = 360)
  (h2 : x + t = 180) :
  x = 180 - y - z :=
by
  sorry

end cyclic_quadrilateral_angles_l1904_190415


namespace num_distinct_prime_factors_330_l1904_190454

theorem num_distinct_prime_factors_330 : 
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x ∈ s, Nat.Prime x ∧ 330 % x = 0 := 
sorry

end num_distinct_prime_factors_330_l1904_190454


namespace apple_equation_l1904_190458

-- Conditions directly from a)
def condition1 (x : ℕ) : Prop := (x - 1) % 3 = 0
def condition2 (x : ℕ) : Prop := (x + 2) % 4 = 0

theorem apple_equation (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : 
  (x - 1) / 3 = (x + 2) / 4 := 
sorry

end apple_equation_l1904_190458


namespace problem1_problem2_l1904_190479

open Set Real

-- Definition of sets A, B, and C
def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }

-- Problem 1: Prove A ∪ B = { x | 1 ≤ x < 10 }
theorem problem1 : A ∪ B = { x : ℝ | 1 ≤ x ∧ x < 10 } :=
sorry

-- Problem 2: Prove the range of a given the conditions
theorem problem2 (a : ℝ) (h1 : (A ∩ C a) ≠ ∅) (h2 : (B ∩ C a) = ∅) : 1 < a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l1904_190479


namespace sum_of_three_largest_ge_50_l1904_190420

theorem sum_of_three_largest_ge_50 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ) :
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧
  a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧
  a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧
  a₆ ≠ a₇ ∧
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0 ∧ a₅ > 0 ∧ a₆ > 0 ∧ a₇ > 0 ∧
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 100 →
  ∃ (x y z : ℕ), (x ≠ y ∧ x ≠ z ∧ y ≠ z) ∧ (x > 0 ∧ y > 0 ∧ z > 0) ∧ (x + y + z ≥ 50) :=
by sorry

end sum_of_three_largest_ge_50_l1904_190420


namespace change_calculation_l1904_190453

-- Definition of amounts and costs
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8
def cost_chicken_wings : ℕ := 6
def cost_chicken_salad : ℕ := 4
def cost_soda : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Main theorem statement
theorem change_calculation
  (total_cost := cost_chicken_wings + cost_chicken_salad + num_sodas * cost_soda + tax)
  (total_amount := lee_amount + friend_amount)
  : total_amount - total_cost = 3 :=
by
  -- Proof steps placeholder
  sorry

end change_calculation_l1904_190453


namespace samantha_birth_year_l1904_190471

theorem samantha_birth_year 
  (first_amc8 : ℕ)
  (amc8_annual : ∀ n : ℕ, n ≥ first_amc8)
  (seventh_amc8 : ℕ)
  (samantha_age : ℕ)
  (samantha_birth_year : ℕ)
  (move_year : ℕ)
  (h1 : first_amc8 = 1983)
  (h2 : seventh_amc8 = first_amc8 + 6)
  (h3 : seventh_amc8 = 1989)
  (h4 : samantha_age = 14)
  (h5 : samantha_birth_year = seventh_amc8 - samantha_age)
  (h6 : move_year = seventh_amc8 - 3) :
  samantha_birth_year = 1975 :=
sorry

end samantha_birth_year_l1904_190471


namespace apples_per_box_l1904_190421

-- Defining the given conditions
variable (apples_per_crate : ℤ)
variable (number_of_crates : ℤ)
variable (rotten_apples : ℤ)
variable (number_of_boxes : ℤ)

-- Stating the facts based on given conditions
def total_apples := apples_per_crate * number_of_crates
def remaining_apples := total_apples - rotten_apples

-- The statement to prove
theorem apples_per_box 
    (hc1 : apples_per_crate = 180)
    (hc2 : number_of_crates = 12)
    (hc3 : rotten_apples = 160)
    (hc4 : number_of_boxes = 100) :
    (remaining_apples apples_per_crate number_of_crates rotten_apples) / number_of_boxes = 20 := 
sorry

end apples_per_box_l1904_190421


namespace isosceles_triangle_side_length_l1904_190413

theorem isosceles_triangle_side_length (a b : ℝ) (h : a < b) : 
  ∃ l : ℝ, l = (b - a) / 2 := 
sorry

end isosceles_triangle_side_length_l1904_190413


namespace exam_correct_answers_count_l1904_190455

theorem exam_correct_answers_count (x y : ℕ) (h1 : x + y = 80) (h2 : 4 * x - y = 130) : x = 42 :=
by {
  -- (proof to be completed later)
  sorry
}

end exam_correct_answers_count_l1904_190455


namespace ellipse_eccentricity_m_l1904_190466

theorem ellipse_eccentricity_m (m : ℝ) (e : ℝ) (h1 : ∀ x y : ℝ, x^2 / m + y^2 = 1) (h2 : e = Real.sqrt 3 / 2) :
  m = 4 ∨ m = 1 / 4 :=
by sorry

end ellipse_eccentricity_m_l1904_190466


namespace find_number_l1904_190499

theorem find_number (x : ℕ) (h : x * 12 = 540) : x = 45 :=
by sorry

end find_number_l1904_190499


namespace coordinates_of_A_l1904_190488

-- Define initial coordinates of point A
def A : ℝ × ℝ := (-2, 4)

-- Define the transformation of moving 2 units upwards
def move_up (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 + units)

-- Define the transformation of moving 3 units to the left
def move_left (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 - units, point.2)

-- Combine the transformations to get point A'
def A' : ℝ × ℝ :=
  move_left (move_up A 2) 3

-- The theorem stating that A' is (-5, 6)
theorem coordinates_of_A' : A' = (-5, 6) :=
by
  sorry

end coordinates_of_A_l1904_190488


namespace sum_of_integers_l1904_190474

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 180) : x + y = 28 :=
by
  sorry

end sum_of_integers_l1904_190474


namespace sum_of_g1_l1904_190468

noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition : ∀ x y : ℝ, g (g (x - y)) = g x + g y - g x * g y - x * y := sorry

theorem sum_of_g1 : g 1 = 1 := 
by
  -- Provide the necessary proof steps to show g(1) = 1
  sorry

end sum_of_g1_l1904_190468


namespace cost_equation_l1904_190492

def cost (W : ℕ) : ℕ :=
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10

theorem cost_equation (W : ℕ) : cost W = 
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10 :=
by
  -- Proof goes here
  sorry

end cost_equation_l1904_190492


namespace speed_of_jakes_dad_second_half_l1904_190418

theorem speed_of_jakes_dad_second_half :
  let distance_to_park := 22
  let total_time := 0.5
  let time_half_journey := total_time / 2
  let speed_first_half := 28
  let distance_first_half := speed_first_half * time_half_journey
  let remaining_distance := distance_to_park - distance_first_half
  let time_second_half := time_half_journey
  let speed_second_half := remaining_distance / time_second_half
  speed_second_half = 60 :=
by
  sorry

end speed_of_jakes_dad_second_half_l1904_190418


namespace copy_pages_15_dollars_l1904_190448

theorem copy_pages_15_dollars (cpp : ℕ) (budget : ℕ) (pages : ℕ) (h1 : cpp = 3) (h2 : budget = 1500) (h3 : pages = budget / cpp) : pages = 500 :=
by
  sorry

end copy_pages_15_dollars_l1904_190448


namespace total_preparation_and_cooking_time_l1904_190445

def time_to_chop_pepper : Nat := 3
def time_to_chop_onion : Nat := 4
def time_to_grate_cheese_per_omelet : Nat := 1
def time_to_cook_omelet : Nat := 5
def num_peppers : Nat := 4
def num_onions : Nat := 2
def num_omelets : Nat := 5

theorem total_preparation_and_cooking_time :
  num_peppers * time_to_chop_pepper +
  num_onions * time_to_chop_onion +
  num_omelets * (time_to_grate_cheese_per_omelet + time_to_cook_omelet) = 50 := 
by
  sorry

end total_preparation_and_cooking_time_l1904_190445


namespace simplify_expression_l1904_190483

theorem simplify_expression (x y : ℝ) (h_x_ne_0 : x ≠ 0) (h_y_ne_0 : y ≠ 0) :
  (25*x^3*y) * (8*x*y) * (1 / (5*x*y^2)^2) = 8*x^2 / y^2 :=
by
  sorry

end simplify_expression_l1904_190483


namespace ratio_r_to_pq_l1904_190469

theorem ratio_r_to_pq (p q r : ℕ) (h₁ : p + q + r = 5000) (h₂ : r = 2000) :
  r / (p + q) = 2 / 3 := 
by
  sorry

end ratio_r_to_pq_l1904_190469


namespace tan_sin_difference_l1904_190463

theorem tan_sin_difference :
  let tan_60 := Real.tan (60 * Real.pi / 180)
  let sin_60 := Real.sin (60 * Real.pi / 180)
  tan_60 - sin_60 = (Real.sqrt 3 / 2) := by
sorry

end tan_sin_difference_l1904_190463


namespace smallest_perimeter_of_consecutive_even_triangle_l1904_190494

theorem smallest_perimeter_of_consecutive_even_triangle (n : ℕ) :
  (2 * n + 2 * n + 2 > 2 * n + 4) ∧
  (2 * n + 2 * n + 4 > 2 * n + 2) ∧
  (2 * n + 2 + 2 * n + 4 > 2 * n) →
  2 * n + (2 * n + 2) + (2 * n + 4) = 18 :=
by 
  sorry

end smallest_perimeter_of_consecutive_even_triangle_l1904_190494


namespace num_administrative_personnel_l1904_190435

noncomputable def total_employees : ℕ := 280
noncomputable def sample_size : ℕ := 56
noncomputable def ordinary_staff_sample : ℕ := 49

theorem num_administrative_personnel (n : ℕ) (h1 : total_employees = 280) 
(h2 : sample_size = 56) (h3 : ordinary_staff_sample = 49) : 
n = 35 := 
by
  have h_proportion : (sample_size - ordinary_staff_sample) / sample_size = n / total_employees := by sorry
  have h_sol : n = (sample_size - ordinary_staff_sample) * (total_employees / sample_size) := by sorry
  have h_n : n = 35 := by sorry
  exact h_n

end num_administrative_personnel_l1904_190435


namespace trapezoid_area_equal_l1904_190424

namespace Geometry

-- Define the areas of the outer and inner equilateral triangles.
def outer_triangle_area : ℝ := 25
def inner_triangle_area : ℝ := 4

-- The number of congruent trapezoids formed between the triangles.
def number_of_trapezoids : ℕ := 4

-- Prove that the area of one trapezoid is 5.25 square units.
theorem trapezoid_area_equal :
  (outer_triangle_area - inner_triangle_area) / number_of_trapezoids = 5.25 := by
  sorry

end Geometry

end trapezoid_area_equal_l1904_190424


namespace common_ratio_geom_series_l1904_190470

theorem common_ratio_geom_series :
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := -16/21
  let a₃ : ℚ := -64/63
  ∃ r : ℚ, r = a₂ / a₁ ∧ r = a₃ / a₂ ∧ r = -4/3 := 
by
  sorry

end common_ratio_geom_series_l1904_190470


namespace cost_of_one_pie_l1904_190442

theorem cost_of_one_pie (x c2 c5 : ℕ) 
  (h1: 4 * x = c2 + 60)
  (h2: 5 * x = c5 + 60) 
  (h3: 6 * x = c2 + c5 + 60) : 
  x = 20 :=
by
  sorry

end cost_of_one_pie_l1904_190442


namespace find_seating_capacity_l1904_190417

theorem find_seating_capacity (x : ℕ) :
  (4 * x + 30 = 5 * x - 10) → (x = 40) :=
by
  intros h
  sorry

end find_seating_capacity_l1904_190417


namespace basketball_students_l1904_190430

variable (C B_inter_C B_union_C B : ℕ)

theorem basketball_students (hC : C = 5) (hB_inter_C : B_inter_C = 3) (hB_union_C : B_union_C = 9) (hInclusionExclusion : B_union_C = B + C - B_inter_C) : B = 7 := by
  sorry

end basketball_students_l1904_190430


namespace quadratic_equation_solutions_l1904_190449

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x^2 + 7 * x = 0 ↔ (x = 0 ∨ x = -7) := 
by 
  intro x
  sorry

end quadratic_equation_solutions_l1904_190449


namespace jars_left_when_boxes_full_l1904_190422

-- Conditions
def jars_in_first_set_of_boxes : Nat := 12 * 10
def jars_in_second_set_of_boxes : Nat := 10 * 30
def total_jars : Nat := 500

-- Question (equivalent proof problem)
theorem jars_left_when_boxes_full : total_jars - (jars_in_first_set_of_boxes + jars_in_second_set_of_boxes) = 80 := 
by
  sorry

end jars_left_when_boxes_full_l1904_190422


namespace circle_radius_l1904_190438

theorem circle_radius (r x y : ℝ) (hx : x = π * r^2) (hy : y = 2 * π * r) (h : x + y = 90 * π) : r = 9 := by
  sorry

end circle_radius_l1904_190438


namespace money_last_weeks_l1904_190480

theorem money_last_weeks (mowing_earning : ℕ) (weeding_earning : ℕ) (spending_per_week : ℕ) 
  (total_amount : ℕ) (weeks : ℕ) :
  mowing_earning = 9 →
  weeding_earning = 18 →
  spending_per_week = 3 →
  total_amount = mowing_earning + weeding_earning →
  weeks = total_amount / spending_per_week →
  weeks = 9 :=
by
  intros
  sorry

end money_last_weeks_l1904_190480


namespace original_price_doubled_l1904_190410

variable (P : ℝ)

-- Given condition: Original price plus 20% equals 351
def price_increased (P : ℝ) : Prop :=
  P + 0.20 * P = 351

-- The goal is to prove that 2 times the original price is 585
theorem original_price_doubled (P : ℝ) (h : price_increased P) : 2 * P = 585 :=
sorry

end original_price_doubled_l1904_190410


namespace upstream_speed_l1904_190451

variable (V_m : ℝ) (V_downstream : ℝ) (V_upstream : ℝ)

def speed_of_man_in_still_water := V_m = 35
def speed_of_man_downstream := V_downstream = 45
def speed_of_man_upstream := V_upstream = 25

theorem upstream_speed
  (h1: speed_of_man_in_still_water V_m)
  (h2: speed_of_man_downstream V_downstream)
  : speed_of_man_upstream V_upstream :=
by
  -- Placeholder for the proof
  sorry

end upstream_speed_l1904_190451
