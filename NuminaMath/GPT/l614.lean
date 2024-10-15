import Mathlib

namespace NUMINAMATH_GPT_num_integers_between_cubed_values_l614_61450

theorem num_integers_between_cubed_values : 
  let a : ℝ := 10.5
  let b : ℝ := 10.7
  let c1 := a^3
  let c2 := b^3
  let first_integer := Int.ceil c1
  let last_integer := Int.floor c2
  first_integer ≤ last_integer → 
  last_integer - first_integer + 1 = 67 :=
by
  sorry

end NUMINAMATH_GPT_num_integers_between_cubed_values_l614_61450


namespace NUMINAMATH_GPT_expand_polynomial_l614_61437

theorem expand_polynomial (N : ℕ) :
  (∃ a b c d : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ (a + b + c + d + 1)^N = 715) ↔ N = 13 := by
  sorry -- Replace with the actual proof when ready

end NUMINAMATH_GPT_expand_polynomial_l614_61437


namespace NUMINAMATH_GPT_right_triangle_BD_length_l614_61417

theorem right_triangle_BD_length (BC AC AD BD : ℝ ) (h_bc: BC = 1) (h_ac: AC = b) (h_ad: AD = 2) :
  BD = Real.sqrt (b^2 - 3) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_BD_length_l614_61417


namespace NUMINAMATH_GPT_jimin_notebooks_proof_l614_61420

variable (m f o n : ℕ)

theorem jimin_notebooks_proof (hm : m = 7) (hf : f = 14) (ho : o = 33) (hn : n = o + m + f) :
  n - o = 21 := by
  sorry

end NUMINAMATH_GPT_jimin_notebooks_proof_l614_61420


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l614_61478

theorem arithmetic_geometric_sequence
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (S : ℕ → ℝ)
  (f : ℕ → ℝ)
  (h₁ : a 1 = 3)
  (h₂ : b 1 = 1)
  (h₃ : b 2 * S 2 = 64)
  (h₄ : b 3 * S 3 = 960)
  : (∀ n, a n = 2 * n + 1) ∧ (∀ n, b n = 8^(n - 1)) ∧ 
    (∀ n, f n = (a n - 1) / (S n + 100)) ∧ 
    (∃ n, f n = 1 / 11 ∧ n = 10) := 
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l614_61478


namespace NUMINAMATH_GPT_maximum_rectangle_area_l614_61433

theorem maximum_rectangle_area (P : ℝ) (hP : P = 36) :
  ∃ (A : ℝ), A = (P / 4) * (P / 4) :=
by
  use 81
  sorry

end NUMINAMATH_GPT_maximum_rectangle_area_l614_61433


namespace NUMINAMATH_GPT_members_of_groups_l614_61447

variable {x y : ℕ}

theorem members_of_groups (h1 : x = y + 10) (h2 : x - 1 = 2 * (y + 1)) :
  x = 17 ∧ y = 7 :=
by
  sorry

end NUMINAMATH_GPT_members_of_groups_l614_61447


namespace NUMINAMATH_GPT_most_likely_number_of_cars_l614_61400

theorem most_likely_number_of_cars 
  (total_time_seconds : ℕ)
  (rate_cars_per_second : ℚ)
  (h1 : total_time_seconds = 180)
  (h2 : rate_cars_per_second = 8 / 15) : 
  ∃ (n : ℕ), n = 100 :=
by
  sorry

end NUMINAMATH_GPT_most_likely_number_of_cars_l614_61400


namespace NUMINAMATH_GPT_buddy_met_boy_students_l614_61442

theorem buddy_met_boy_students (total_students : ℕ) (girl_students : ℕ) (boy_students : ℕ) (h1 : total_students = 123) (h2 : girl_students = 57) : boy_students = 66 :=
by
  sorry

end NUMINAMATH_GPT_buddy_met_boy_students_l614_61442


namespace NUMINAMATH_GPT_combined_age_of_siblings_l614_61475

-- We are given Aaron's age
def aaronAge : ℕ := 15

-- Henry's sister's age is three times Aaron's age
def henrysSisterAge : ℕ := 3 * aaronAge

-- Henry's age is four times his sister's age
def henryAge : ℕ := 4 * henrysSisterAge

-- The combined age of the siblings
def combinedAge : ℕ := aaronAge + henrysSisterAge + henryAge

theorem combined_age_of_siblings : combinedAge = 240 := by
  sorry

end NUMINAMATH_GPT_combined_age_of_siblings_l614_61475


namespace NUMINAMATH_GPT_parabola_line_no_intersection_l614_61464

theorem parabola_line_no_intersection (x y : ℝ) (h : y^2 < 4 * x) :
  ¬ ∃ (x' y' : ℝ), y' = y ∧ y'^2 = 4 * x' ∧ 2 * x' = x + x :=
by sorry

end NUMINAMATH_GPT_parabola_line_no_intersection_l614_61464


namespace NUMINAMATH_GPT_volume_ratio_l614_61455

theorem volume_ratio (V1 V2 M1 M2 : ℝ)
  (h1 : M1 / (V1 - M1) = 1 / 2)
  (h2 : M2 / (V2 - M2) = 3 / 2)
  (h3 : (M1 + M2) / (V1 - M1 + V2 - M2) = 1) :
  V1 / V2 = 9 / 5 :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_l614_61455


namespace NUMINAMATH_GPT_alpha_beta_sum_equal_two_l614_61440

theorem alpha_beta_sum_equal_two (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 17 = 0) 
  (hβ : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := 
sorry

end NUMINAMATH_GPT_alpha_beta_sum_equal_two_l614_61440


namespace NUMINAMATH_GPT_find_number_l614_61496

theorem find_number (N : ℕ) (h : N / 7 = 12 ∧ N % 7 = 5) : N = 89 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l614_61496


namespace NUMINAMATH_GPT_fruit_seller_apples_l614_61432

theorem fruit_seller_apples : 
  ∃ (x : ℝ), (x * 0.6 = 420) → x = 700 :=
sorry

end NUMINAMATH_GPT_fruit_seller_apples_l614_61432


namespace NUMINAMATH_GPT_number_of_rectangles_with_one_gray_cell_l614_61443

theorem number_of_rectangles_with_one_gray_cell 
    (num_gray_cells : Nat) 
    (num_blue_cells : Nat) 
    (num_red_cells : Nat) 
    (blue_rectangles_per_cell : Nat) 
    (red_rectangles_per_cell : Nat)
    (total_gray_cells_calc : num_gray_cells = 2 * 20)
    (num_gray_cells_definition : num_gray_cells = num_blue_cells + num_red_cells)
    (blue_rect_cond : blue_rectangles_per_cell = 4)
    (red_rect_cond : red_rectangles_per_cell = 8)
    (num_blue_cells_calc : num_blue_cells = 36)
    (num_red_cells_calc : num_red_cells = 4)
  : num_blue_cells * blue_rectangles_per_cell + num_red_cells * red_rectangles_per_cell = 176 := 
  by
  sorry

end NUMINAMATH_GPT_number_of_rectangles_with_one_gray_cell_l614_61443


namespace NUMINAMATH_GPT_men_build_wall_l614_61435

theorem men_build_wall (k : ℕ) (h1 : 20 * 6 = k) : ∃ d : ℝ, (30 * d = k) ∧ d = 4.0 := by
  sorry

end NUMINAMATH_GPT_men_build_wall_l614_61435


namespace NUMINAMATH_GPT_cost_of_greenhouses_possible_renovation_plans_l614_61488

noncomputable def cost_renovation (x y : ℕ) : Prop :=
  (2 * x = y + 6) ∧ (x + 2 * y = 48)

theorem cost_of_greenhouses : ∃ x y, cost_renovation x y ∧ x = 12 ∧ y = 18 :=
by {
  sorry
}

noncomputable def renovation_plan (m : ℕ) : Prop :=
  (5 * m + 3 * (8 - m) ≤ 35) ∧ (12 * m + 18 * (8 - m) ≤ 128)

theorem possible_renovation_plans : ∃ m, renovation_plan m ∧ (m = 3 ∨ m = 4 ∨ m = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_greenhouses_possible_renovation_plans_l614_61488


namespace NUMINAMATH_GPT_second_machine_equation_l614_61485

-- Let p1_rate and p2_rate be the rates of printing for machine 1 and 2 respectively.
-- Let x be the unknown time for the second machine to print 500 envelopes.

theorem second_machine_equation (x : ℝ) :
    (500 / 8) + (500 / x) = (500 / 2) :=
  sorry

end NUMINAMATH_GPT_second_machine_equation_l614_61485


namespace NUMINAMATH_GPT_difference_between_numbers_l614_61453

theorem difference_between_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 190) : x - y = 19 :=
by sorry

end NUMINAMATH_GPT_difference_between_numbers_l614_61453


namespace NUMINAMATH_GPT_ms_warren_walking_time_l614_61463

/-- 
Ms. Warren ran at 6 mph for 20 minutes. After the run, 
she walked at 2 mph for a certain amount of time. 
She ran and walked a total of 3 miles.
-/
def time_spent_walking (running_speed walking_speed : ℕ) (running_time_minutes : ℕ) (total_distance : ℕ) : ℕ := 
  let running_time_hours := running_time_minutes / 60;
  let distance_ran := running_speed * running_time_hours;
  let distance_walked := total_distance - distance_ran;
  let time_walked_hours := distance_walked / walking_speed;
  time_walked_hours * 60

theorem ms_warren_walking_time :
  time_spent_walking 6 2 20 3 = 30 :=
by
  sorry

end NUMINAMATH_GPT_ms_warren_walking_time_l614_61463


namespace NUMINAMATH_GPT_distance_between_trees_l614_61448

theorem distance_between_trees (trees : ℕ) (total_length : ℝ) (n : trees = 26) (l : total_length = 500) :
  ∃ d : ℝ, d = total_length / (trees - 1) ∧ d = 20 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l614_61448


namespace NUMINAMATH_GPT_quadrilateral_is_parallelogram_l614_61460

theorem quadrilateral_is_parallelogram
  (A B C D : Type)
  (angle_DAB angle_ABC angle_BAD angle_DCB : ℝ)
  (h1 : angle_DAB = 135)
  (h2 : angle_ABC = 45)
  (h3 : angle_BAD = 45)
  (h4 : angle_DCB = 45) :
  (A B C D : Type) → Prop :=
by
  -- Definitions and conditions are given.
  sorry

end NUMINAMATH_GPT_quadrilateral_is_parallelogram_l614_61460


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l614_61462

open Real

noncomputable def eccentricity_min (m : ℝ) (h₁ : m > 0) (h₂ : m ≥ 2) : ℝ :=
  if h : m = 2 then (sqrt 6)/3 else 0

theorem eccentricity_of_ellipse (m : ℝ) (h₁ : m > 0) (h₂ : m ≥ 2) :
    eccentricity_min m h₁ h₂ = (sqrt 6)/3 := by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l614_61462


namespace NUMINAMATH_GPT_inequality_solution_l614_61473

-- Define the variable x as a real number
variable (x : ℝ)

-- Define the given condition that x is positive
def is_positive (x : ℝ) := x > 0

-- Define the condition that x satisfies the inequality sqrt(9x) < 3x^2
def satisfies_inequality (x : ℝ) := Real.sqrt (9 * x) < 3 * x^2

-- The statement we need to prove
theorem inequality_solution (x : ℝ) (h : is_positive x) : satisfies_inequality x ↔ x > 1 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l614_61473


namespace NUMINAMATH_GPT_work_completion_days_l614_61428

theorem work_completion_days (A B C : ℝ) (h1 : A + B + C = 1/4) (h2 : B = 1/18) (h3 : C = 1/6) : A = 1/36 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_days_l614_61428


namespace NUMINAMATH_GPT_equilateral_triangle_path_l614_61418

noncomputable def equilateral_triangle_path_length (side_length_triangle side_length_square : ℝ) : ℝ :=
  let radius := side_length_triangle
  let rotational_path_length := 4 * 3 * 2 * Real.pi
  let diagonal_length := (Real.sqrt (side_length_square^2 + side_length_square^2))
  let linear_path_length := 2 * diagonal_length
  rotational_path_length + linear_path_length

theorem equilateral_triangle_path (side_length_triangle side_length_square : ℝ) 
  (h_triangle : side_length_triangle = 3) (h_square : side_length_square = 6) :
  equilateral_triangle_path_length side_length_triangle side_length_square = 24 * Real.pi + 12 * Real.sqrt 2 :=
by
  rw [h_triangle, h_square]
  unfold equilateral_triangle_path_length
  sorry

end NUMINAMATH_GPT_equilateral_triangle_path_l614_61418


namespace NUMINAMATH_GPT_tangents_and_fraction_l614_61480

theorem tangents_and_fraction
  (α β : ℝ)
  (tan_diff : Real.tan (α - β) = 2)
  (tan_beta : Real.tan β = 4) :
  (7 * Real.sin α - Real.cos α) / (7 * Real.sin α + Real.cos α) = 7 / 5 :=
sorry

end NUMINAMATH_GPT_tangents_and_fraction_l614_61480


namespace NUMINAMATH_GPT_simplify_trig_l614_61426

open Real

theorem simplify_trig : 
  (sin (30 * pi / 180) + sin (60 * pi / 180)) / (cos (30 * pi / 180) + cos (60 * pi / 180)) = tan (45 * pi / 180) :=
by
  sorry

end NUMINAMATH_GPT_simplify_trig_l614_61426


namespace NUMINAMATH_GPT_min_sum_abc_l614_61421

theorem min_sum_abc (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 1716) :
  a + b + c = 31 :=
sorry

end NUMINAMATH_GPT_min_sum_abc_l614_61421


namespace NUMINAMATH_GPT_evaluate_expression_l614_61491

theorem evaluate_expression : (3 : ℚ) / (1 - (2 : ℚ) / 5) = 5 := sorry

end NUMINAMATH_GPT_evaluate_expression_l614_61491


namespace NUMINAMATH_GPT_r_and_s_earns_per_day_l614_61474

variable (P Q R S : Real)

-- Conditions as given in the problem
axiom cond1 : P + Q + R + S = 2380 / 9
axiom cond2 : P + R = 600 / 5
axiom cond3 : Q + S = 800 / 6
axiom cond4 : Q + R = 910 / 7
axiom cond5 : P = 150 / 3

theorem r_and_s_earns_per_day : R + S = 143.33 := by
  sorry

end NUMINAMATH_GPT_r_and_s_earns_per_day_l614_61474


namespace NUMINAMATH_GPT_complex_power_identity_l614_61434

theorem complex_power_identity (z : ℂ) (i : ℂ) 
  (h1 : z = (1 + i) / Real.sqrt 2) 
  (h2 : z^2 = i) : 
  z^100 = -1 := 
  sorry

end NUMINAMATH_GPT_complex_power_identity_l614_61434


namespace NUMINAMATH_GPT_product_xy_eq_3_l614_61406

variable {x y : ℝ}
variables (h₀ : x ≠ y) (h₁ : x ≠ 0) (h₂ : y ≠ 0)
variable (h₃ : x + (3 / x) = y + (3 / y))

theorem product_xy_eq_3 : x * y = 3 := by
  sorry

end NUMINAMATH_GPT_product_xy_eq_3_l614_61406


namespace NUMINAMATH_GPT_remaining_bottles_after_2_days_l614_61441

-- Definitions based on the conditions:
def initial_bottles : ℕ := 24
def fraction_first_day : ℚ := 1 / 3
def fraction_second_day : ℚ := 1 / 2

-- Theorem statement proving the remaining number of bottles after 2 days
theorem remaining_bottles_after_2_days : 
    (initial_bottles - initial_bottles * fraction_first_day) - 
    ((initial_bottles - initial_bottles * fraction_first_day) * fraction_second_day) = 8 := 
by 
    -- Skipping the proof
    sorry

end NUMINAMATH_GPT_remaining_bottles_after_2_days_l614_61441


namespace NUMINAMATH_GPT_cos_triple_sum_div_l614_61431

theorem cos_triple_sum_div {A B C : ℝ} (h : Real.cos A + Real.cos B + Real.cos C = 0) : 
  (Real.cos (3 * A) + Real.cos (3 * B) + Real.cos (3 * C)) / (Real.cos A * Real.cos B * Real.cos C) = 12 :=
by
  sorry

end NUMINAMATH_GPT_cos_triple_sum_div_l614_61431


namespace NUMINAMATH_GPT_sum_of_possible_values_of_G_F_l614_61454

theorem sum_of_possible_values_of_G_F (G F : ℕ) (hG : 0 ≤ G ∧ G ≤ 9) (hF : 0 ≤ F ∧ F ≤ 9)
  (hdiv : (G + 2 + 4 + 3 + F + 1 + 6) % 9 = 0) : G + F = 2 ∨ G + F = 11 → 2 + 11 = 13 :=
by { sorry }

end NUMINAMATH_GPT_sum_of_possible_values_of_G_F_l614_61454


namespace NUMINAMATH_GPT_simplify_sqrt_sum_l614_61487

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_sum_l614_61487


namespace NUMINAMATH_GPT_sector_area_l614_61493

theorem sector_area (r l : ℝ) (h_r : r = 2) (h_l : l = 3) : (1/2) * l * r = 3 :=
by
  rw [h_r, h_l]
  norm_num

end NUMINAMATH_GPT_sector_area_l614_61493


namespace NUMINAMATH_GPT_part1_part2_l614_61451

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 
  Real.log x + 0.5 * m * x^2 - 2 

def perpendicular_slope_condition (m : ℝ) : Prop := 
  let k := (1 / 1 + m)
  k = -1 / 2

def inequality_condition (m : ℝ) : Prop := 
  ∀ x > 0, 
  Real.log x - 0.5 * m * x^2 + (1 - m) * x + 1 ≤ 0

theorem part1 : perpendicular_slope_condition (-3/2) :=
  sorry

theorem part2 : ∃ m : ℤ, m ≥ 2 ∧ inequality_condition m :=
  sorry

end NUMINAMATH_GPT_part1_part2_l614_61451


namespace NUMINAMATH_GPT_trig_identity_l614_61459

theorem trig_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l614_61459


namespace NUMINAMATH_GPT_parabola_whose_directrix_is_tangent_to_circle_l614_61495

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 9

noncomputable def is_tangent (line_eq : ℝ → ℝ → Prop) (circle_eq : ℝ → ℝ → Prop) : Prop := 
  ∃ p : ℝ × ℝ, (line_eq p.1 p.2) ∧ (circle_eq p.1 p.2) ∧ 
  (∀ q : ℝ × ℝ, (circle_eq q.1 q.2) → (line_eq q.1 q.2) → q = p)

-- Definitions of parabolas
noncomputable def parabola_A_directrix (x y : ℝ) : Prop := y = 2

noncomputable def parabola_B_directrix (x y : ℝ) : Prop := x = 2

noncomputable def parabola_C_directrix (x y : ℝ) : Prop := x = -4

noncomputable def parabola_D_directrix (x y : ℝ) : Prop := y = -1

-- The final statement to prove
theorem parabola_whose_directrix_is_tangent_to_circle :
  is_tangent parabola_D_directrix circle_eq ∧ ¬ is_tangent parabola_A_directrix circle_eq ∧ 
  ¬ is_tangent parabola_B_directrix circle_eq ∧ ¬ is_tangent parabola_C_directrix circle_eq :=
sorry

end NUMINAMATH_GPT_parabola_whose_directrix_is_tangent_to_circle_l614_61495


namespace NUMINAMATH_GPT_pablo_mother_pays_each_page_l614_61415

-- Definitions based on the conditions in the problem
def pages_per_book := 150
def number_books_read := 12
def candy_cost := 15
def money_leftover := 3
def total_money := candy_cost + money_leftover
def total_pages := number_books_read * pages_per_book
def amount_paid_per_page := total_money / total_pages

-- The theorem to be proven
theorem pablo_mother_pays_each_page
    (pages_per_book : ℝ)
    (number_books_read : ℝ)
    (candy_cost : ℝ)
    (money_leftover : ℝ)
    (total_money := candy_cost + money_leftover)
    (total_pages := number_books_read * pages_per_book)
    (amount_paid_per_page := total_money / total_pages) :
    amount_paid_per_page = 0.01 :=
by
  sorry

end NUMINAMATH_GPT_pablo_mother_pays_each_page_l614_61415


namespace NUMINAMATH_GPT_arbitrary_large_sum_of_digits_l614_61481

noncomputable def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem arbitrary_large_sum_of_digits (a : Nat) (h1 : 2 ≤ a) (h2 : ¬ (2 ∣ a)) (h3 : ¬ (5 ∣ a)) :
  ∃ m : Nat, sum_of_digits (a^m) > m :=
by
  sorry

end NUMINAMATH_GPT_arbitrary_large_sum_of_digits_l614_61481


namespace NUMINAMATH_GPT_conference_games_l614_61461

/-- 
Two divisions of 8 teams each, where each team plays 21 games within its division 
and 8 games against the teams of the other division. 
Prove total number of scheduled conference games is 232.
-/
theorem conference_games (div_teams : ℕ) (intra_div_games : ℕ) (inter_div_games : ℕ) (total_teams : ℕ) :
  div_teams = 8 →
  intra_div_games = 21 →
  inter_div_games = 8 →
  total_teams = 2 * div_teams →
  (total_teams * (intra_div_games + inter_div_games)) / 2 = 232 :=
by
  intros
  sorry


end NUMINAMATH_GPT_conference_games_l614_61461


namespace NUMINAMATH_GPT_greater_expected_area_l614_61465

/-- Let X be a random variable representing a single roll of a die, which can take integer values from 1 through 6. -/
def X : Type := { x : ℕ // 1 ≤ x ∧ x ≤ 6 }

/-- Define independent random variables A and B representing the outcomes of Asya’s die rolls, which can take integer values from 1 through 6 with equal probability. -/
noncomputable def A : Type := { a : ℕ // 1 ≤ a ∧ a ≤ 6 }
noncomputable def B : Type := { b : ℕ // 1 ≤ b ∧ b ≤ 6 }

/-- The expected value of a random variable taking integer values from 1 through 6. 
    E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6 = 15.1667 -/
noncomputable def expected_X_squared : ℝ := 91 / 6

/-- The expected value of the product of two independent random variables each taking integer values from 1 through 6. 
    E[A * B] = E[A] * E[B] = 3.5 * 3.5 = 12.25 -/
noncomputable def expected_A_times_B : ℝ := 12.25

/-- Prove that the expected area of Vasya's square is greater than Asya's rectangle.
    i.e., E[X^2] > E[A * B] -/
theorem greater_expected_area : expected_X_squared > expected_A_times_B :=
sorry

end NUMINAMATH_GPT_greater_expected_area_l614_61465


namespace NUMINAMATH_GPT_find_divisor_l614_61411

open Nat

theorem find_divisor 
  (d n : ℕ)
  (h1 : n % d = 3)
  (h2 : 2 * n % d = 2) : 
  d = 4 := 
sorry

end NUMINAMATH_GPT_find_divisor_l614_61411


namespace NUMINAMATH_GPT_joseph_vs_kyle_emily_vs_joseph_emily_vs_kyle_l614_61457

noncomputable def distance_joseph : ℝ := 48 * 2.5 + 60 * 1.5
noncomputable def distance_kyle : ℝ := 70 * 2 + 63 * 2.5
noncomputable def distance_emily : ℝ := 65 * 3

theorem joseph_vs_kyle : distance_joseph - distance_kyle = -87.5 := by
  unfold distance_joseph
  unfold distance_kyle
  sorry

theorem emily_vs_joseph : distance_emily - distance_joseph = -15 := by
  unfold distance_emily
  unfold distance_joseph
  sorry

theorem emily_vs_kyle : distance_emily - distance_kyle = -102.5 := by
  unfold distance_emily
  unfold distance_kyle
  sorry

end NUMINAMATH_GPT_joseph_vs_kyle_emily_vs_joseph_emily_vs_kyle_l614_61457


namespace NUMINAMATH_GPT_league_games_count_l614_61477

theorem league_games_count :
  let num_divisions := 2
  let teams_per_division := 9
  let intra_division_games (teams_per_div : ℕ) := (teams_per_div * (teams_per_div - 1) / 2) * 3
  let inter_division_games (teams_per_div : ℕ) (num_div : ℕ) := teams_per_div * teams_per_div * 2
  intra_division_games teams_per_division * num_divisions + inter_division_games teams_per_division num_divisions = 378 :=
by
  sorry

end NUMINAMATH_GPT_league_games_count_l614_61477


namespace NUMINAMATH_GPT_find_n_arithmetic_sequence_l614_61499

-- Given conditions
def a₁ : ℕ := 20
def aₙ : ℕ := 54
def Sₙ : ℕ := 999

-- Arithmetic sequence sum formula and proof statement of n = 27
theorem find_n_arithmetic_sequence
  (a₁ : ℕ)
  (aₙ : ℕ)
  (Sₙ : ℕ)
  (h₁ : a₁ = 20)
  (h₂ : aₙ = 54)
  (h₃ : Sₙ = 999) : ∃ n : ℕ, n = 27 := 
by
  sorry

end NUMINAMATH_GPT_find_n_arithmetic_sequence_l614_61499


namespace NUMINAMATH_GPT_necessary_condition_of_and_is_or_l614_61498

variable (p q : Prop)

theorem necessary_condition_of_and_is_or (hpq : p ∧ q) : p ∨ q :=
by {
    sorry
}

end NUMINAMATH_GPT_necessary_condition_of_and_is_or_l614_61498


namespace NUMINAMATH_GPT_one_way_ticket_cost_l614_61429

theorem one_way_ticket_cost (x : ℝ) (h : 50 / 26 < x) : x >= 2 :=
by sorry

end NUMINAMATH_GPT_one_way_ticket_cost_l614_61429


namespace NUMINAMATH_GPT_probability_same_number_l614_61497

theorem probability_same_number (n k : ℕ) (h₁ : n = 8) (h₂ : k = 6) : 
  (∃ m : ℝ, 0 ≤ m ∧ m ≤ 1 ∧ m = 1) := by
  sorry

end NUMINAMATH_GPT_probability_same_number_l614_61497


namespace NUMINAMATH_GPT_cost_of_letter_is_0_37_l614_61472

-- Definitions based on the conditions
def total_cost : ℝ := 4.49
def package_cost : ℝ := 0.88
def num_letters : ℕ := 5
def num_packages : ℕ := 3
def letter_cost (L : ℝ) : ℝ := 5 * L
def package_total_cost : ℝ := num_packages * package_cost

-- Theorem that encapsulates the mathematical proof problem
theorem cost_of_letter_is_0_37 (L : ℝ) (h : letter_cost L + package_total_cost = total_cost) : L = 0.37 :=
by sorry

end NUMINAMATH_GPT_cost_of_letter_is_0_37_l614_61472


namespace NUMINAMATH_GPT_find_x_l614_61419

theorem find_x (x : ℝ) : (3 / 4 * 1 / 2 * 2 / 5) * x = 765.0000000000001 → x = 5100.000000000001 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l614_61419


namespace NUMINAMATH_GPT_find_theta_l614_61446

def rectangle : Type := sorry
def angle (α : ℝ) : Prop := 0 ≤ α ∧ α < 180

-- Given conditions in the problem
variables {α β γ δ θ : ℝ}

axiom angle_10 : angle 10
axiom angle_14 : angle 14
axiom angle_33 : angle 33
axiom angle_26 : angle 26

axiom zig_zag_angles (a b c d e f : ℝ) :
  a = 26 ∧ f = 10 ∧
  26 + b = 33 ∧ b = 7 ∧
  e + 10 = 14 ∧ e = 4 ∧
  c = b ∧ d = e ∧
  θ = c + d

theorem find_theta : θ = 11 :=
sorry

end NUMINAMATH_GPT_find_theta_l614_61446


namespace NUMINAMATH_GPT_range_of_a_if_p_is_false_l614_61412

theorem range_of_a_if_p_is_false (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_if_p_is_false_l614_61412


namespace NUMINAMATH_GPT_carla_receives_correct_amount_l614_61402

theorem carla_receives_correct_amount (L B C X : ℝ) : 
  (L + B + C + X) / 3 - (C + X) = (L + B - 2 * C - 2 * X) / 3 :=
by
  sorry

end NUMINAMATH_GPT_carla_receives_correct_amount_l614_61402


namespace NUMINAMATH_GPT_compute_expr_l614_61427

theorem compute_expr : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end NUMINAMATH_GPT_compute_expr_l614_61427


namespace NUMINAMATH_GPT_number_of_whole_numbers_between_sqrts_l614_61484

noncomputable def count_whole_numbers_between_sqrts : ℕ :=
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let min_int := Int.ceil lower_bound
  let max_int := Int.floor upper_bound
  Int.natAbs (max_int - min_int + 1)

theorem number_of_whole_numbers_between_sqrts :
  count_whole_numbers_between_sqrts = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_whole_numbers_between_sqrts_l614_61484


namespace NUMINAMATH_GPT_axis_of_symmetry_l614_61468

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = f (4 - x)) : ∀ y, f 2 = y ↔ f 2 = y := 
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_l614_61468


namespace NUMINAMATH_GPT_initial_total_toys_l614_61409

-- Definitions based on the conditions
def initial_red_toys (R : ℕ) : Prop := R - 2 = 88
def twice_as_many_red_toys (R W : ℕ) : Prop := R - 2 = 2 * W

-- The proof statement: show that initially there were 134 toys in the box
theorem initial_total_toys (R W : ℕ) (hR : initial_red_toys R) (hW : twice_as_many_red_toys R W) : R + W = 134 := 
by sorry

end NUMINAMATH_GPT_initial_total_toys_l614_61409


namespace NUMINAMATH_GPT_lowest_score_dropped_l614_61422

-- Conditions definitions
def total_sum_of_scores (A B C D : ℕ) := A + B + C + D = 240
def total_sum_after_dropping_lowest (A B C : ℕ) := A + B + C = 195

-- Theorem statement
theorem lowest_score_dropped (A B C D : ℕ) (h1 : total_sum_of_scores A B C D) (h2 : total_sum_after_dropping_lowest A B C) : D = 45 := 
sorry

end NUMINAMATH_GPT_lowest_score_dropped_l614_61422


namespace NUMINAMATH_GPT_find_constants_l614_61410

theorem find_constants (A B C : ℚ) :
  (∀ x : ℚ, (8 * x + 1) / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2) → 
  A = 33 / 4 ∧ B = -19 / 4 ∧ C = -17 / 2 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_find_constants_l614_61410


namespace NUMINAMATH_GPT_find_number_l614_61492

theorem find_number (x : ℝ) (h : 7 * x + 21.28 = 50.68) : x = 4.2 :=
sorry

end NUMINAMATH_GPT_find_number_l614_61492


namespace NUMINAMATH_GPT_total_tiles_l614_61449

theorem total_tiles (s : ℕ) (h1 : true) (h2 : true) (h3 : true) (h4 : true) (h5 : 4 * s - 4 = 100): s * s = 676 :=
by
  sorry

end NUMINAMATH_GPT_total_tiles_l614_61449


namespace NUMINAMATH_GPT_hyperbola_transverse_axis_l614_61407

noncomputable def hyperbola_transverse_axis_length (a b : ℝ) : ℝ :=
  2 * a

theorem hyperbola_transverse_axis {a b : ℝ} (h : a > 0) (h_b : b > 0) 
  (eccentricity_cond : Real.sqrt 2 = Real.sqrt (1 + b^2 / a^2))
  (area_cond : ∃ x y : ℝ, x^2 = -4 * Real.sqrt 3 * y ∧ y * y / a^2 - x^2 / b^2 = 1 ∧ 
                 Real.sqrt 3 = 1 / 2 * (2 * Real.sqrt (3 - a^2)) * Real.sqrt 3) :
  hyperbola_transverse_axis_length a b = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_transverse_axis_l614_61407


namespace NUMINAMATH_GPT_find_x_angle_l614_61486

-- Define the conditions
def angles_around_point (a b c d : ℝ) : Prop :=
  a + b + c + d = 360

-- The given problem implies:
-- 120 + x + x + 2x = 360
-- We need to find x such that the above equation holds.
theorem find_x_angle :
  angles_around_point 120 x x (2 * x) → x = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_x_angle_l614_61486


namespace NUMINAMATH_GPT_find_a_l614_61452

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.logb 2 (x^2 + a)

theorem find_a (a : ℝ) : f 3 a = 1 → a = -7 :=
by
  intro h
  unfold f at h
  sorry

end NUMINAMATH_GPT_find_a_l614_61452


namespace NUMINAMATH_GPT_number_of_children_l614_61445

namespace CurtisFamily

variables {m x : ℕ} {xy : ℕ}

/-- Given conditions for Curtis family average ages. -/
def family_average_age (m x xy : ℕ) : Prop := (m + 50 + xy) / (2 + x) = 25

def mother_children_average_age (m x xy : ℕ) : Prop := (m + xy) / (1 + x) = 20

/-- The number of children in Curtis family is 4, given the average age conditions. -/
theorem number_of_children (m xy : ℕ) (h1 : family_average_age m 4 xy) (h2 : mother_children_average_age m 4 xy) : x = 4 :=
by
  sorry

end CurtisFamily

end NUMINAMATH_GPT_number_of_children_l614_61445


namespace NUMINAMATH_GPT_fifteen_times_number_eq_150_l614_61425

theorem fifteen_times_number_eq_150 (n : ℕ) (h : 15 * n = 150) : n = 10 :=
sorry

end NUMINAMATH_GPT_fifteen_times_number_eq_150_l614_61425


namespace NUMINAMATH_GPT_car_capacities_rental_plans_l614_61456

-- Define the capacities for part 1
def capacity_A : ℕ := 3
def capacity_B : ℕ := 4

theorem car_capacities (x y : ℕ) (h₁ : 2 * x + y = 10) (h₂ : x + 2 * y = 11) : 
  x = capacity_A ∧ y = capacity_B := by
  sorry

-- Define the valid rental plans for part 2
def valid_rental_plan (a b : ℕ) : Prop :=
  3 * a + 4 * b = 31

theorem rental_plans (a b : ℕ) (h : valid_rental_plan a b) : 
  (a = 1 ∧ b = 7) ∨ (a = 5 ∧ b = 4) ∨ (a = 9 ∧ b = 1) := by
  sorry

end NUMINAMATH_GPT_car_capacities_rental_plans_l614_61456


namespace NUMINAMATH_GPT_engineer_formula_updated_l614_61414

theorem engineer_formula_updated (T H : ℕ) (hT : T = 5) (hH : H = 10) :
  (30 * T^5) / (H^3 : ℚ) = 375 / 4 := by
  sorry

end NUMINAMATH_GPT_engineer_formula_updated_l614_61414


namespace NUMINAMATH_GPT_extreme_points_of_f_range_of_a_for_f_le_g_l614_61405

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x + (1 / 2) * x^2 + a * x

noncomputable def g (x : ℝ) : ℝ :=
  Real.exp x + (3 / 2) * x^2

theorem extreme_points_of_f (a : ℝ) :
  (∃ (x1 x2 : ℝ), x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0)
    ↔ a < -2 :=
sorry

theorem range_of_a_for_f_le_g :
  (∀ x : ℝ, x > 0 → f x a ≤ g x) ↔ a ≤ Real.exp 1 + 1 :=
sorry

end NUMINAMATH_GPT_extreme_points_of_f_range_of_a_for_f_le_g_l614_61405


namespace NUMINAMATH_GPT_calculation_correct_l614_61483

-- Defining the initial values
def a : ℕ := 20 ^ 10
def b : ℕ := 20 ^ 9
def c : ℕ := 10 ^ 6
def d : ℕ := 2 ^ 12

-- The expression we need to prove
theorem calculation_correct : ((a / b) ^ 3 * c) / d = 1953125 :=
by
  sorry

end NUMINAMATH_GPT_calculation_correct_l614_61483


namespace NUMINAMATH_GPT_expression_eval_l614_61490

theorem expression_eval : (3^2 - 3) - (5^2 - 5) * 2 + (6^2 - 6) = -4 :=
by sorry

end NUMINAMATH_GPT_expression_eval_l614_61490


namespace NUMINAMATH_GPT_distance_walked_by_man_l614_61467

theorem distance_walked_by_man (x t : ℝ) (h1 : d = (x + 0.5) * (4 / 5) * t) (h2 : d = (x - 0.5) * (t + 2.5)) : d = 15 :=
by
  sorry

end NUMINAMATH_GPT_distance_walked_by_man_l614_61467


namespace NUMINAMATH_GPT_pi_bounds_l614_61430

theorem pi_bounds : 
  3.14 < Real.pi ∧ Real.pi < 3.142 ∧
  9.86 < Real.pi ^ 2 ∧ Real.pi ^ 2 < 9.87 := sorry

end NUMINAMATH_GPT_pi_bounds_l614_61430


namespace NUMINAMATH_GPT_hyperbola_asymptote_slopes_l614_61479

theorem hyperbola_asymptote_slopes :
  ∀ (x y : ℝ), 2 * (y^2 / 16) - 2 * (x^2 / 9) = 1 → (∃ m : ℝ, y = m * x ∨ y = -m * x) ∧ m = (Real.sqrt 80) / 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_slopes_l614_61479


namespace NUMINAMATH_GPT_pot_holds_three_liters_l614_61416

theorem pot_holds_three_liters (drips_per_minute : ℕ) (ml_per_drop : ℕ) (minutes : ℕ) (full_pot_volume : ℕ) :
  drips_per_minute = 3 → ml_per_drop = 20 → minutes = 50 → full_pot_volume = (drips_per_minute * ml_per_drop * minutes) / 1000 →
  full_pot_volume = 3 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_pot_holds_three_liters_l614_61416


namespace NUMINAMATH_GPT_units_digit_diff_l614_61482

theorem units_digit_diff (p : ℕ) (hp : p > 0) (even_p : p % 2 = 0) (units_p1_7 : (p + 1) % 10 = 7) : (p^3 % 10) = (p^2 % 10) :=
by
  sorry

end NUMINAMATH_GPT_units_digit_diff_l614_61482


namespace NUMINAMATH_GPT_grace_clyde_ratio_l614_61444

theorem grace_clyde_ratio (C G : ℕ) (h1 : G = C + 35) (h2 : G = 40) : G / C = 8 :=
by sorry

end NUMINAMATH_GPT_grace_clyde_ratio_l614_61444


namespace NUMINAMATH_GPT_football_goals_l614_61404

theorem football_goals :
  (exists A B C : ℕ,
    (A = 3 ∧ B ≠ 1 ∧ (C = 5 ∧ V = 6 ∧ A ≠ 2 ∧ V = 5)) ∨
    (A ≠ 3 ∧ B = 1 ∧ (C ≠ 5 ∧ V = 6 ∧ A = 2 ∧ V ≠ 5))) →
  A + B + C ≠ 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_football_goals_l614_61404


namespace NUMINAMATH_GPT_factorial_division_l614_61401

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division : (factorial 15) / ((factorial 6) * (factorial 9)) = 5005 :=
by
  sorry

end NUMINAMATH_GPT_factorial_division_l614_61401


namespace NUMINAMATH_GPT_regular_price_of_shrimp_l614_61470

theorem regular_price_of_shrimp 
  (discounted_price : ℝ) 
  (discount_rate : ℝ) 
  (quarter_pound_price : ℝ) 
  (full_pound_price : ℝ) 
  (price_relation : quarter_pound_price = discounted_price * (1 - discount_rate) / 4) 
  (discounted_value : quarter_pound_price = 2) 
  (given_discount_rate : discount_rate = 0.6) 
  (given_discounted_price : discounted_price = full_pound_price) 
  : full_pound_price = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_regular_price_of_shrimp_l614_61470


namespace NUMINAMATH_GPT_map_distance_l614_61408

noncomputable def map_scale_distance (actual_distance_km : ℕ) (scale : ℕ) : ℕ :=
  let actual_distance_cm := actual_distance_km * 100000;  -- conversion from kilometers to centimeters
  actual_distance_cm / scale

theorem map_distance (d_km : ℕ) (scale : ℕ) (h1 : d_km = 500) (h2 : scale = 8000000) :
  map_scale_distance d_km scale = 625 :=
by
  rw [h1, h2]
  dsimp [map_scale_distance]
  norm_num
  sorry

end NUMINAMATH_GPT_map_distance_l614_61408


namespace NUMINAMATH_GPT_geometric_sequence_value_of_a_l614_61423

noncomputable def a : ℝ :=
sorry

theorem geometric_sequence_value_of_a
  (is_geometric_seq : ∀ (x y z : ℝ), z / y = y / x)
  (first_term : ℝ)
  (second_term : ℝ)
  (third_term : ℝ)
  (h1 : first_term = 140)
  (h2 : second_term = a)
  (h3 : third_term = 45 / 28)
  (pos_a : a > 0):
  a = 15 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_value_of_a_l614_61423


namespace NUMINAMATH_GPT_age_of_15th_person_l614_61469

variable (avg_age_20 : ℕ) (avg_age_5 : ℕ) (avg_age_9 : ℕ) (A : ℕ)
variable (num_20 : ℕ) (num_5 : ℕ) (num_9 : ℕ)

theorem age_of_15th_person (h1 : avg_age_20 = 15) (h2 : avg_age_5 = 14) (h3 : avg_age_9 = 16)
  (h4 : num_20 = 20) (h5 : num_5 = 5) (h6 : num_9 = 9) :
  (num_20 * avg_age_20) = (num_5 * avg_age_5) + (num_9 * avg_age_9) + A → A = 86 :=
by
  sorry

end NUMINAMATH_GPT_age_of_15th_person_l614_61469


namespace NUMINAMATH_GPT_glucose_solution_volume_l614_61476

theorem glucose_solution_volume
  (h1 : 6.75 / 45 = 15 / x) :
  x = 100 :=
by
  sorry

end NUMINAMATH_GPT_glucose_solution_volume_l614_61476


namespace NUMINAMATH_GPT_largest_s_for_angle_ratio_l614_61438

theorem largest_s_for_angle_ratio (r s : ℕ) (hr : r ≥ 3) (hs : s ≥ 3) (h_angle_ratio : (130 * (r - 2)) * s = (131 * (s - 2)) * r) :
  s ≤ 260 :=
by 
  sorry

end NUMINAMATH_GPT_largest_s_for_angle_ratio_l614_61438


namespace NUMINAMATH_GPT_common_factor_of_polynomial_l614_61494

variables (x y m n : ℝ)

theorem common_factor_of_polynomial :
  ∃ (k : ℝ), (2 * (m - n)) = k ∧ (4 * x * (m - n) + 2 * y * (m - n)^2) = k * (2 * x * (m - n)) :=
sorry

end NUMINAMATH_GPT_common_factor_of_polynomial_l614_61494


namespace NUMINAMATH_GPT_sphere_surface_area_l614_61466

noncomputable def surface_area_of_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem sphere_surface_area (r_circle r_distance : ℝ) :
  (Real.pi * r_circle^2 = 16 * Real.pi) →
  (r_distance = 3) →
  (surface_area_of_sphere (Real.sqrt (r_distance^2 + r_circle^2)) = 100 * Real.pi) := by
sorry

end NUMINAMATH_GPT_sphere_surface_area_l614_61466


namespace NUMINAMATH_GPT_repetend_of_4_div_17_l614_61403

theorem repetend_of_4_div_17 :
  ∃ (r : String), (∀ (n : ℕ), (∃ (k : ℕ), (0 < k) ∧ (∃ (q : ℤ), (4 : ℤ) * 10 ^ (n + 12 * k) / 17 % 10 ^ 12 = q)) ∧ r = "235294117647") :=
sorry

end NUMINAMATH_GPT_repetend_of_4_div_17_l614_61403


namespace NUMINAMATH_GPT_find_a_l614_61436

noncomputable def f (a x : ℝ) := (x - 1)^2 + a * x + Real.cos x

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → 
  a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l614_61436


namespace NUMINAMATH_GPT_lollipop_ratio_l614_61424

/-- Sarah bought 12 lollipops for a total of 3 dollars. Julie gave Sarah 75 cents to pay for the shared lollipops.
Prove that the ratio of the number of lollipops shared to the total number of lollipops bought is 1:4. -/
theorem lollipop_ratio
  (h1 : 12 = lollipops_bought)
  (h2 : 3 = total_cost_dollars)
  (h3 : 75 = amount_paid_cents)
  : (75 / 25) / lollipops_bought = 1/4 :=
sorry

end NUMINAMATH_GPT_lollipop_ratio_l614_61424


namespace NUMINAMATH_GPT_labor_arrangement_count_l614_61458

theorem labor_arrangement_count (volunteers : ℕ) (choose_one_day : ℕ) (days : ℕ) 
    (h_volunteers : volunteers = 7) 
    (h_choose_one_day : choose_one_day = 3) 
    (h_days : days = 2) : 
    (Nat.choose volunteers choose_one_day) * (Nat.choose (volunteers - choose_one_day) choose_one_day) = 140 := 
by
  sorry

end NUMINAMATH_GPT_labor_arrangement_count_l614_61458


namespace NUMINAMATH_GPT_find_x_of_parallel_vectors_l614_61471

theorem find_x_of_parallel_vectors
  (x : ℝ)
  (p : ℝ × ℝ := (2, -3))
  (q : ℝ × ℝ := (x, 6))
  (h : ∃ k : ℝ, q = k • p) :
  x = -4 :=
sorry

end NUMINAMATH_GPT_find_x_of_parallel_vectors_l614_61471


namespace NUMINAMATH_GPT_fraction_of_tomato_plants_in_second_garden_l614_61439

theorem fraction_of_tomato_plants_in_second_garden 
    (total_plants_first_garden : ℕ := 20)
    (percent_tomato_first_garden : ℚ := 10 / 100)
    (total_plants_second_garden : ℕ := 15)
    (percent_total_tomato_plants : ℚ := 20 / 100) :
    (15 : ℚ) * (1 / 3) = 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_tomato_plants_in_second_garden_l614_61439


namespace NUMINAMATH_GPT_waiting_for_stocker_proof_l614_61413

-- Definitions for the conditions
def waiting_for_cart := 3
def waiting_for_employee := 13
def waiting_in_line := 18
def total_shopping_trip_time := 90
def time_shopping := 42

-- Calculate the total waiting time
def total_waiting_time := total_shopping_trip_time - time_shopping

-- Calculate the total known waiting time
def total_known_waiting_time := waiting_for_cart + waiting_for_employee + waiting_in_line

-- Calculate the waiting time for the stocker
def waiting_for_stocker := total_waiting_time - total_known_waiting_time

-- Prove that the waiting time for the stocker is 14 minutes
theorem waiting_for_stocker_proof : waiting_for_stocker = 14 := by
  -- Here the proof steps would normally be included
  sorry

end NUMINAMATH_GPT_waiting_for_stocker_proof_l614_61413


namespace NUMINAMATH_GPT_angle_sum_l614_61489

theorem angle_sum {A B D F G : Type} 
  (angle_A : ℝ) 
  (angle_AFG : ℝ) 
  (angle_AGF : ℝ) 
  (angle_BFD : ℝ)
  (H1 : angle_A = 30)
  (H2 : angle_AFG = angle_AGF)
  (H3 : angle_BFD = 105)
  (H4 : angle_AFG + angle_BFD = 180) 
  : angle_B + angle_D = 75 := 
by 
  sorry

end NUMINAMATH_GPT_angle_sum_l614_61489
