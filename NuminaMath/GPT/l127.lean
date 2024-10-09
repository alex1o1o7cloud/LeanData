import Mathlib

namespace frustum_has_only_two_parallel_surfaces_l127_12765

-- Definitions for the geometric bodies in terms of their properties
structure Pyramid where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 0

structure Prism where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 6

structure Frustum where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 2

structure Cuboid where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 6

-- The main theorem stating that the Frustum is the one with exactly two parallel surfaces.
theorem frustum_has_only_two_parallel_surfaces (pyramid : Pyramid) (prism : Prism) (frustum : Frustum) (cuboid : Cuboid) :
  frustum.parallel_surfaces = 2 ∧
  pyramid.parallel_surfaces ≠ 2 ∧
  prism.parallel_surfaces ≠ 2 ∧
  cuboid.parallel_surfaces ≠ 2 :=
by
  sorry

end frustum_has_only_two_parallel_surfaces_l127_12765


namespace rectangle_dimensions_l127_12725

theorem rectangle_dimensions (x y : ℝ) (h1 : x = 2 * y) (h2 : 2 * (x + y) = 2 * x * y) : 
  (x = 3 ∧ y = 1.5) :=
by
  sorry

end rectangle_dimensions_l127_12725


namespace new_concentration_is_37_percent_l127_12731

-- Conditions
def capacity_vessel_1 : ℝ := 2 -- litres
def alcohol_concentration_vessel_1 : ℝ := 0.35

def capacity_vessel_2 : ℝ := 6 -- litres
def alcohol_concentration_vessel_2 : ℝ := 0.50

def total_poured_liquid : ℝ := 8 -- litres
def final_vessel_capacity : ℝ := 10 -- litres

-- Question: Prove the new concentration of the mixture
theorem new_concentration_is_37_percent :
  (alcohol_concentration_vessel_1 * capacity_vessel_1 + alcohol_concentration_vessel_2 * capacity_vessel_2) / final_vessel_capacity = 0.37 := by
  sorry

end new_concentration_is_37_percent_l127_12731


namespace class_student_count_l127_12711

-- Statement: Prove that under the given conditions, the number of students in the class is 19.
theorem class_student_count (n : ℕ) (avg_students_age : ℕ) (teacher_age : ℕ) (avg_with_teacher : ℕ):
  avg_students_age = 20 → 
  teacher_age = 40 → 
  avg_with_teacher = 21 → 
  21 * (n + 1) = 20 * n + 40 → 
  n = 19 := 
by 
  intros h1 h2 h3 h4 
  sorry

end class_student_count_l127_12711


namespace perpendicular_lines_l127_12770

theorem perpendicular_lines (a : ℝ) :
  (∃ x y : ℝ, x * a + 3 * y - 1 = 0) ∧ (∃ x y : ℝ, 2 * x + (a - 1) * y + 1 = 0) ∧
  (∀ m1 m2 : ℝ, m1 = - a / 3 → m2 = - 2 / (a - 1) → m1 * m2 = -1) →
  a = 3 / 5 :=
sorry

end perpendicular_lines_l127_12770


namespace min_value_expression_l127_12741

theorem min_value_expression (a : ℝ) (h : a > 2) : a + 4 / (a - 2) ≥ 6 :=
by
  sorry

end min_value_expression_l127_12741


namespace spring_length_relationship_maximum_mass_l127_12756

theorem spring_length_relationship (x y : ℝ) : 
  (y = 0.5 * x + 12) ↔ y = 12 + 0.5 * x := 
by sorry

theorem maximum_mass (x y : ℝ) : 
  (y = 0.5 * x + 12) → (y ≤ 20) → (x ≤ 16) :=
by sorry

end spring_length_relationship_maximum_mass_l127_12756


namespace playground_length_l127_12707

theorem playground_length
  (P : ℕ)
  (B : ℕ)
  (h1 : P = 1200)
  (h2 : B = 500)
  (h3 : P = 2 * (100 + B)) :
  100 = 100 :=
 by sorry

end playground_length_l127_12707


namespace pencils_multiple_of_40_l127_12795

theorem pencils_multiple_of_40 :
  ∃ n : ℕ, 640 % n = 0 ∧ n ≤ 40 → ∃ m : ℕ, 40 * m = 40 * n :=
by
  sorry

end pencils_multiple_of_40_l127_12795


namespace tangent_addition_l127_12701

theorem tangent_addition (y : ℝ) (h : Real.tan y = -1) : Real.tan (y + Real.pi / 3) = -1 :=
sorry

end tangent_addition_l127_12701


namespace quadratic_no_real_roots_implies_inequality_l127_12797

theorem quadratic_no_real_roots_implies_inequality (a b c : ℝ) :
  let A := b + c
  let B := a + c
  let C := a + b
  (B^2 - 4 * A * C < 0) → 4 * a * c - b^2 ≤ 3 * a * (a + b + c) :=
by
  intro h
  sorry

end quadratic_no_real_roots_implies_inequality_l127_12797


namespace number_of_female_officers_is_382_l127_12759

noncomputable def F : ℝ := 
  let total_on_duty := 210
  let ratio_male_female := 3 / 2
  let percent_female_on_duty := 22 / 100
  let female_on_duty := total_on_duty * (2 / (3 + 2))
  let total_females := female_on_duty / percent_female_on_duty
  total_females

theorem number_of_female_officers_is_382 : F = 382 := 
by
  sorry

end number_of_female_officers_is_382_l127_12759


namespace unique_solution_l127_12723

noncomputable def pair_satisfying_equation (m n : ℕ) : Prop :=
  2^m - 1 = 3^n

theorem unique_solution : ∀ (m n : ℕ), m > 0 → n > 0 → pair_satisfying_equation m n → (m, n) = (2, 1) :=
by
  intros m n m_pos n_pos h
  sorry

end unique_solution_l127_12723


namespace solve_for_x_l127_12738

theorem solve_for_x (x : ℝ) (h1 : x^2 - 9 ≠ 0) (h2 : x + 3 ≠ 0) :
  (20 / (x^2 - 9) - 3 / (x + 3) = 2) ↔ (x = (-3 + Real.sqrt 385) / 4 ∨ x = (-3 - Real.sqrt 385) / 4) :=
by
  sorry

end solve_for_x_l127_12738


namespace valid_license_plates_count_l127_12777

def num_valid_license_plates := (26 ^ 3) * (10 ^ 4)

theorem valid_license_plates_count : num_valid_license_plates = 175760000 :=
by
  sorry

end valid_license_plates_count_l127_12777


namespace find_x_of_product_eq_72_l127_12726

theorem find_x_of_product_eq_72 (x : ℝ) (h : 0 < x) (hx : x * ⌊x⌋₊ = 72) : x = 9 :=
sorry

end find_x_of_product_eq_72_l127_12726


namespace coprime_powers_l127_12760

theorem coprime_powers (n : ℕ) : Nat.gcd (n^5 + 4 * n^3 + 3 * n) (n^4 + 3 * n^2 + 1) = 1 :=
sorry

end coprime_powers_l127_12760


namespace number_of_unique_outfits_l127_12769

-- Define the given conditions
def num_shirts : ℕ := 8
def num_ties : ℕ := 6
def special_shirt_ties : ℕ := 3
def remaining_shirts := num_shirts - 1
def remaining_ties := num_ties

-- Define the proof problem
theorem number_of_unique_outfits : num_shirts * num_ties - remaining_shirts * remaining_ties + special_shirt_ties = 45 :=
by
  sorry

end number_of_unique_outfits_l127_12769


namespace opening_price_calculation_l127_12763

variable (Closing_Price : ℝ)
variable (Percent_Increase : ℝ)
variable (Opening_Price : ℝ)

theorem opening_price_calculation
    (H1 : Closing_Price = 28)
    (H2 : Percent_Increase = 0.1200000000000001) :
    Opening_Price = Closing_Price / (1 + Percent_Increase) := by
  sorry

end opening_price_calculation_l127_12763


namespace joe_paint_usage_l127_12721

theorem joe_paint_usage :
  let initial_paint := 360
  let first_week_usage := (1 / 3: ℝ) * initial_paint
  let remaining_after_first_week := initial_paint - first_week_usage
  let second_week_usage := (1 / 5: ℝ) * remaining_after_first_week
  let total_usage := first_week_usage + second_week_usage
  total_usage = 168 :=
by
  sorry

end joe_paint_usage_l127_12721


namespace max_roads_no_intersections_l127_12789

theorem max_roads_no_intersections (V : ℕ) (hV : V = 100) : 
  ∃ E : ℕ, E ≤ 3 * V - 6 ∧ E = 294 := 
by 
  sorry

end max_roads_no_intersections_l127_12789


namespace two_pow_n_minus_one_div_by_seven_iff_l127_12728

theorem two_pow_n_minus_one_div_by_seven_iff (n : ℕ) : (7 ∣ (2^n - 1)) ↔ (∃ k : ℕ, n = 3 * k) := by
  sorry

end two_pow_n_minus_one_div_by_seven_iff_l127_12728


namespace find_initial_marbles_l127_12737

def initial_marbles (W Y H : ℕ) : Prop :=
  (W + 2 = 20) ∧ (Y - 5 = 20) ∧ (H + 3 = 20)

theorem find_initial_marbles (W Y H : ℕ) (h : initial_marbles W Y H) : W = 18 :=
  by
    sorry

end find_initial_marbles_l127_12737


namespace probability_value_l127_12703

noncomputable def P (k : ℕ) (c : ℚ) : ℚ := c / (k * (k + 1))

theorem probability_value (c : ℚ) (h : P 1 c + P 2 c + P 3 c + P 4 c = 1) : P 1 c + P 2 c = 5 / 6 := 
by
  sorry

end probability_value_l127_12703


namespace unit_prices_max_colored_tiles_l127_12767

-- Define the given conditions
def condition1 (x y : ℝ) := 40 * x + 60 * y = 5600
def condition2 (x y : ℝ) := 50 * x + 50 * y = 6000

-- Prove the solution for part 1
theorem unit_prices (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 80 ∧ y = 40 := 
sorry

-- Define the condition for the kitchen tiles
def condition3 (a : ℝ) := 80 * a + 40 * (60 - a) ≤ 3400

-- Prove the maximum number of colored tiles for the kitchen
theorem max_colored_tiles (a : ℝ) (h3 : condition3 a) :
  a ≤ 25 := 
sorry

end unit_prices_max_colored_tiles_l127_12767


namespace value_of_x_l127_12744

def condition (x : ℝ) : Prop :=
  3 * x = (20 - x) + 20

theorem value_of_x : ∃ x : ℝ, condition x ∧ x = 10 := 
by
  sorry

end value_of_x_l127_12744


namespace triangle_ineq_sqrt_triangle_l127_12766

open Real

theorem triangle_ineq_sqrt_triangle (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a):
  (∃ u v w : ℝ, u > 0 ∧ v > 0 ∧ w > 0 ∧ a = v + w ∧ b = u + w ∧ c = u + v) ∧ 
  (sqrt (a * b) + sqrt (b * c) + sqrt (c * a) ≤ a + b + c ∧ a + b + c ≤ 2 * sqrt (a * b) + 2 * sqrt (b * c) + 2 * sqrt (c * a)) :=
  sorry

end triangle_ineq_sqrt_triangle_l127_12766


namespace average_four_numbers_l127_12790

variable {x : ℝ}

theorem average_four_numbers (h : (15 + 25 + x + 30) / 4 = 23) : x = 22 :=
by
  sorry

end average_four_numbers_l127_12790


namespace find_d_l127_12796

theorem find_d (c a m d : ℝ) (h : m = (c * a * d) / (a - d)) : d = (m * a) / (m + c * a) :=
by sorry

end find_d_l127_12796


namespace dice_game_probability_l127_12751

def is_valid_roll (d1 d2 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6

def score (d1 d2 : ℕ) : ℕ :=
  max d1 d2

def favorable_outcomes : List (ℕ × ℕ) :=
  [ (1, 1), (1, 2), (2, 1), (2, 2), 
    (1, 3), (2, 3), (3, 1), (3, 2), (3, 3) ]

def total_outcomes : ℕ := 36

def favorable_count : ℕ := favorable_outcomes.length

theorem dice_game_probability : 
  (favorable_count : ℚ) / (total_outcomes : ℚ) = 1 / 4 :=
by
  sorry

end dice_game_probability_l127_12751


namespace wall_area_l127_12788

-- Definition of the width and length of the wall
def width : ℝ := 5.4
def length : ℝ := 2.5

-- Statement of the theorem
theorem wall_area : (width * length) = 13.5 :=
by
  sorry

end wall_area_l127_12788


namespace proof_problem_l127_12702

-- Definitions based on the conditions
def x := 70 + 0.11 * 70
def y := x + 0.15 * x
def z := y - 0.2 * y

-- The statement to prove
theorem proof_problem : 3 * z - 2 * x + y = 148.407 :=
by
  sorry

end proof_problem_l127_12702


namespace forty_percent_of_number_l127_12706

theorem forty_percent_of_number (N : ℝ) 
  (h : (1/4) * (1/3) * (2/5) * N = 35) : 0.4 * N = 420 :=
by
  sorry

end forty_percent_of_number_l127_12706


namespace find_num_students_B_l127_12740

-- Given conditions as definitions
def num_students_A : ℕ := 24
def avg_weight_A : ℚ := 40
def avg_weight_B : ℚ := 35
def avg_weight_class : ℚ := 38

-- The total weight for sections A and B
def total_weight_A : ℚ := num_students_A * avg_weight_A
def total_weight_B (x: ℕ) : ℚ := x * avg_weight_B

-- The number of students in section B
noncomputable def num_students_B : ℕ := 16

-- The proof problem: Prove that number of students in section B is 16
theorem find_num_students_B (x: ℕ) (h: (total_weight_A + total_weight_B x) / (num_students_A + x) = avg_weight_class) : 
  x = 16 :=
by
  sorry

end find_num_students_B_l127_12740


namespace rubert_james_ratio_l127_12755

-- Definitions and conditions from a)
def adam_candies : ℕ := 6
def james_candies : ℕ := 3 * adam_candies
def rubert_candies (total_candies : ℕ) : ℕ := total_candies - (adam_candies + james_candies)
def total_candies : ℕ := 96

-- Statement to prove the ratio
theorem rubert_james_ratio : 
  (rubert_candies total_candies) / james_candies = 4 :=
by
  -- Proof is not required, so we leave it as sorry.
  sorry

end rubert_james_ratio_l127_12755


namespace must_hold_inequality_l127_12720

variable (f : ℝ → ℝ)

noncomputable def condition : Prop := ∀ x > 0, x * (deriv^[2] f) x < 1

theorem must_hold_inequality (h : condition f) : f (Real.exp 1) < f 1 + 1 := 
sorry

end must_hold_inequality_l127_12720


namespace find_missing_surface_area_l127_12722

noncomputable def total_surface_area (areas : List ℕ) : ℕ :=
  areas.sum

def known_areas : List ℕ := [148, 46, 72, 28, 88, 126, 58]

def missing_surface_area : ℕ := 22

theorem find_missing_surface_area (areas : List ℕ) (total : ℕ) (missing : ℕ) :
  total_surface_area areas + missing = total →
  missing = 22 :=
by
  sorry

end find_missing_surface_area_l127_12722


namespace Y_subset_X_l127_12780

def X : Set ℕ := {n | ∃ m : ℕ, n = 4 * m + 2}

def Y : Set ℕ := {t | ∃ k : ℕ, t = (2 * k - 1)^2 + 1}

theorem Y_subset_X : Y ⊆ X := by
  sorry

end Y_subset_X_l127_12780


namespace cookie_recipe_total_cups_l127_12709

theorem cookie_recipe_total_cups (r_butter : ℕ) (r_flour : ℕ) (r_sugar : ℕ) (sugar_cups : ℕ) 
  (h_ratio : r_butter = 1 ∧ r_flour = 2 ∧ r_sugar = 3) (h_sugar : sugar_cups = 9) : 
  r_butter * (sugar_cups / r_sugar) + r_flour * (sugar_cups / r_sugar) + sugar_cups = 18 := 
by 
  sorry

end cookie_recipe_total_cups_l127_12709


namespace initial_printing_presses_l127_12743

theorem initial_printing_presses (P : ℕ) 
  (h1 : 500000 / (9 * P) = 500000 / (12 * 30)) : 
  P = 40 :=
by
  sorry

end initial_printing_presses_l127_12743


namespace part_I_part_II_l127_12732

-- Define the conditions given in the problem
def set_A : Set ℝ := { x | -1 < x ∧ x < 3 }
def set_B (a b : ℝ) : Set ℝ := { x | x^2 - a * x + b < 0 }

-- Part I: Prove that if A = B, then a = 2 and b = -3
theorem part_I (a b : ℝ) (h : set_A = set_B a b) : a = 2 ∧ b = -3 :=
sorry

-- Part II: Prove that if b = 3 and A ∩ B ⊇ B, then the range of a is [-2√3, 4]
theorem part_II (a : ℝ) (b : ℝ := 3) (h : set_A ∩ set_B a b ⊇ set_B a b) : -2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 :=
sorry

end part_I_part_II_l127_12732


namespace ratio_of_areas_of_concentric_circles_l127_12758

theorem ratio_of_areas_of_concentric_circles :
  (∀ (r1 r2 : ℝ), 
    r1 > 0 ∧ r2 > 0 ∧ 
    ((60 / 360) * 2 * Real.pi * r1 = (48 / 360) * 2 * Real.pi * r2)) →
    ((Real.pi * r1 ^ 2) / (Real.pi * r2 ^ 2) = (16 / 25)) :=
by
  intro h
  sorry

end ratio_of_areas_of_concentric_circles_l127_12758


namespace proof_problem_l127_12739

variables {x y z w : ℝ}

-- Condition given in the problem
def condition (x y z w : ℝ) : Prop :=
  (x - y) * (z - w) / ((y - z) * (w - x)) = 1 / 3

-- The statement to be proven
theorem proof_problem (h : condition x y z w) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = 1 :=
by
  sorry

end proof_problem_l127_12739


namespace integer_solutions_for_xyz_eq_4_l127_12708

theorem integer_solutions_for_xyz_eq_4 :
  {n : ℕ // n = 48} :=
sorry

end integer_solutions_for_xyz_eq_4_l127_12708


namespace minimize_total_time_l127_12747

def exercise_time (s : ℕ → ℕ) : Prop :=
  ∀ i, s i < 45

def total_exercises (a : ℕ → ℕ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 25

def minimize_time (a : ℕ → ℕ) (s : ℕ → ℕ) : Prop :=
  ∃ (j : ℕ), (1 ≤ j ∧ j ≤ 7 ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ 7 → if i = j then a i = 25 else a i = 0) ∧
  ∀ i, 1 ≤ i ∧ i ≤ 7 → s i ≥ s j)

theorem minimize_total_time
  (a : ℕ → ℕ) (s : ℕ → ℕ) 
  (h_exercise_time : exercise_time s)
  (h_total_exercises : total_exercises a) :
  minimize_time a s := by
  sorry

end minimize_total_time_l127_12747


namespace avg_price_of_pen_l127_12772

theorem avg_price_of_pen 
  (total_pens : ℕ) (total_pencils : ℕ) (total_cost : ℕ) 
  (avg_price_pencil : ℕ) (total_pens_cost : ℕ) (total_pencils_cost : ℕ)
  (total_cost_eq : total_cost = total_pens_cost + total_pencils_cost)
  (total_pencils_cost_eq : total_pencils_cost = total_pencils * avg_price_pencil)
  (pencils_count : total_pencils = 75) (pens_count : total_pens = 30) 
  (avg_price_pencil_eq : avg_price_pencil = 2)
  (total_cost_eq' : total_cost = 450) :
  total_pens_cost / total_pens = 10 :=
by
  sorry

end avg_price_of_pen_l127_12772


namespace selected_numbers_count_l127_12700

noncomputable def check_num_of_selected_numbers : ℕ := 
  let n := 2015
  let max_num := n * n
  let common_difference := 15
  let starting_number := 14
  let count := (max_num - starting_number) / common_difference + 1
  count

theorem selected_numbers_count : check_num_of_selected_numbers = 270681 := by
  -- Skipping the actual proof
  sorry

end selected_numbers_count_l127_12700


namespace remaining_sugar_l127_12705

-- Conditions as definitions
def total_sugar : ℝ := 9.8
def spilled_sugar : ℝ := 5.2

-- Theorem to prove the remaining sugar
theorem remaining_sugar : total_sugar - spilled_sugar = 4.6 := by
  sorry

end remaining_sugar_l127_12705


namespace almond_walnut_ratio_is_5_to_2_l127_12713

-- Definitions based on conditions
variables (A W : ℕ)
def almond_ratio_to_walnut_ratio := A / (2 * W)
def weight_of_almonds := 250
def total_weight := 350
def weight_of_walnuts := total_weight - weight_of_almonds

-- Theorem to prove
theorem almond_walnut_ratio_is_5_to_2
  (h_ratio : almond_ratio_to_walnut_ratio A W = 250 / 100)
  (h_weights : weight_of_walnuts = 100) :
  A = 5 ∧ 2 * W = 2 := by
  sorry

end almond_walnut_ratio_is_5_to_2_l127_12713


namespace complex_expression_l127_12778

theorem complex_expression (i : ℂ) (h₁ : i^2 = -1) (h₂ : i^4 = 1) :
  (i + i^3)^100 + (i + i^2 + i^3 + i^4 + i^5)^120 = 1 := by
  sorry

end complex_expression_l127_12778


namespace Joey_age_l127_12716

theorem Joey_age (J B : ℕ) (h1 : J + 5 = B) (h2 : J - 4 = B - J) : J = 9 :=
by 
  sorry

end Joey_age_l127_12716


namespace reciprocal_is_1_or_neg1_self_square_is_0_or_1_l127_12712

theorem reciprocal_is_1_or_neg1 (x : ℝ) (hx : x = 1 / x) :
  x = 1 ∨ x = -1 :=
sorry

theorem self_square_is_0_or_1 (x : ℝ) (hx : x = x^2) :
  x = 0 ∨ x = 1 :=
sorry

end reciprocal_is_1_or_neg1_self_square_is_0_or_1_l127_12712


namespace factory_produces_correct_number_of_doors_l127_12776

variable (initial_planned_production : ℕ) (metal_shortage_decrease : ℕ) (pandemic_decrease_factor : ℕ)
variable (doors_per_car : ℕ)

theorem factory_produces_correct_number_of_doors
  (h1 : initial_planned_production = 200)
  (h2 : metal_shortage_decrease = 50)
  (h3 : pandemic_decrease_factor = 50)
  (h4 : doors_per_car = 5) :
  (initial_planned_production - metal_shortage_decrease) * (100 - pandemic_decrease_factor) * doors_per_car / 100 = 375 :=
by
  sorry

end factory_produces_correct_number_of_doors_l127_12776


namespace total_money_l127_12753

theorem total_money (m c : ℝ) (hm : m = 5 / 8) (hc : c = 7 / 20) : m + c = 0.975 := sorry

end total_money_l127_12753


namespace division_quotient_l127_12735

theorem division_quotient (x : ℤ) (y : ℤ) (r : ℝ) (h1 : x > 0) (h2 : y = 96) (h3 : r = 11.52) :
  ∃ q : ℝ, q = (x - r) / y := 
sorry

end division_quotient_l127_12735


namespace sum_of_areas_of_six_rectangles_eq_572_l127_12787

theorem sum_of_areas_of_six_rectangles_eq_572 :
  let lengths := [1, 3, 5, 7, 9, 11]
  let areas := lengths.map (λ x => 2 * x^2)
  areas.sum = 572 :=
by 
  sorry

end sum_of_areas_of_six_rectangles_eq_572_l127_12787


namespace amy_7_mile_run_time_l127_12754

-- Define the conditions
variable (rachel_time_per_9_miles : ℕ) (amy_time_per_4_miles : ℕ) (amy_time_per_mile : ℕ) (amy_time_per_7_miles: ℕ)

-- State the conditions
def conditions : Prop :=
  rachel_time_per_9_miles = 36 ∧
  amy_time_per_4_miles = 1 / 3 * rachel_time_per_9_miles ∧
  amy_time_per_mile = amy_time_per_4_miles / 4 ∧
  amy_time_per_7_miles = amy_time_per_mile * 7

-- The main statement to prove
theorem amy_7_mile_run_time (rachel_time_per_9_miles : ℕ) (amy_time_per_4_miles : ℕ) (amy_time_per_mile : ℕ) (amy_time_per_7_miles: ℕ) :
  conditions rachel_time_per_9_miles amy_time_per_4_miles amy_time_per_mile amy_time_per_7_miles → 
  amy_time_per_7_miles = 21 := 
by
  intros h
  sorry

end amy_7_mile_run_time_l127_12754


namespace vec_parallel_l127_12791

variable {R : Type*} [LinearOrderedField R]

def is_parallel (a b : R × R) : Prop :=
  ∃ k : R, a = (k * b.1, k * b.2)

theorem vec_parallel {x : R} : 
  is_parallel (1, x) (-3, 4) ↔ x = -4/3 := by
  sorry

end vec_parallel_l127_12791


namespace boiling_point_water_standard_l127_12771

def boiling_point_water_celsius : ℝ := 100

theorem boiling_point_water_standard (bp_f : ℝ := 212) (ice_melting_c : ℝ := 0) (ice_melting_f : ℝ := 32) (pot_temp_c : ℝ := 55) (pot_temp_f : ℝ := 131) : boiling_point_water_celsius = 100 :=
by 
  -- Assuming standard atmospheric conditions, the boiling point of water in Celsius is 100.
  sorry

end boiling_point_water_standard_l127_12771


namespace part1_part2_l127_12784

open Set

variable {m x : ℝ}

def A (m : ℝ) : Set ℝ := { x | x^2 - (m+1)*x + m = 0 }
def B (m : ℝ) : Set ℝ := { x | x * m - 1 = 0 }

theorem part1 (h : A m ⊆ B m) : m = 1 :=
by
  sorry

theorem part2 (h : B m ⊂ A m) : m = 0 ∨ m = -1 :=
by
  sorry

end part1_part2_l127_12784


namespace length_of_goods_train_l127_12793

theorem length_of_goods_train
  (speed_man_train : ℕ) (speed_goods_train : ℕ) (passing_time : ℕ)
  (h1 : speed_man_train = 40)
  (h2 : speed_goods_train = 72)
  (h3 : passing_time = 9) :
  (112 * 1000 / 3600) * passing_time = 280 := 
by
  sorry

end length_of_goods_train_l127_12793


namespace mimi_spent_on_clothes_l127_12764

noncomputable def total_cost : ℤ := 8000
noncomputable def cost_adidas : ℤ := 600
noncomputable def cost_nike : ℤ := 3 * cost_adidas
noncomputable def cost_skechers : ℤ := 5 * cost_adidas
noncomputable def cost_clothes : ℤ := total_cost - (cost_adidas + cost_nike + cost_skechers)

theorem mimi_spent_on_clothes :
  cost_clothes = 2600 :=
by
  sorry

end mimi_spent_on_clothes_l127_12764


namespace all_are_knights_l127_12749

-- Definitions for inhabitants as either knights or knaves
inductive Inhabitant
| Knight : Inhabitant
| Knave : Inhabitant

open Inhabitant

-- Functions that determine if an inhabitant is a knight or a knave
def is_knight (x : Inhabitant) : Prop :=
  x = Knight

def is_knave (x : Inhabitant) : Prop :=
  x = Knave

-- Given conditions
axiom A : Inhabitant
axiom B : Inhabitant
axiom C : Inhabitant

axiom statement_A : is_knight A → is_knight B
axiom statement_B : is_knight B → (is_knight A → is_knight C)

-- The proof goal
theorem all_are_knights : is_knight A ∧ is_knight B ∧ is_knight C := by
  sorry

end all_are_knights_l127_12749


namespace calc_value_l127_12782

noncomputable def f : ℝ → ℝ := sorry 

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom non_const_zero : ∃ x : ℝ, f x ≠ 0
axiom functional_eq : ∀ x : ℝ, x * f (x + 1) = (x + 1) * f x

theorem calc_value : f (f (5 / 2)) = 0 :=
sorry

end calc_value_l127_12782


namespace combined_variance_is_178_l127_12773

noncomputable def average_weight_A := 60
noncomputable def variance_A := 100
noncomputable def average_weight_B := 64
noncomputable def variance_B := 200
noncomputable def ratio_A_B := (1, 3)

theorem combined_variance_is_178 :
  let nA := ratio_A_B.1
  let nB := ratio_A_B.2
  let avg_comb := (nA * average_weight_A + nB * average_weight_B) / (nA + nB)
  let var_comb := (nA * (variance_A + (average_weight_A - avg_comb)^2) + 
                   nB * (variance_B + (average_weight_B - avg_comb)^2)) / 
                   (nA + nB)
  var_comb = 178 := 
by
  sorry

end combined_variance_is_178_l127_12773


namespace solve_equation_l127_12775

theorem solve_equation :
  ∀ x : ℝ, (4 * x - 2 * x + 1 - 3 = 0) ↔ (x = 1 ∨ x = -1) :=
by
  intro x
  sorry

end solve_equation_l127_12775


namespace add_fractions_l127_12724

theorem add_fractions : (1 / 6 : ℚ) + (5 / 12) = 7 / 12 := 
by
  sorry

end add_fractions_l127_12724


namespace trigonometric_identity_l127_12786

open Real

theorem trigonometric_identity (α : ℝ) (h1 : tan α = 4/3) (h2 : 0 < α ∧ α < π / 2) :
  sin (π + α) + cos (π - α) = -7/5 :=
by
  sorry

end trigonometric_identity_l127_12786


namespace jeremy_watermelons_l127_12752

theorem jeremy_watermelons :
  ∀ (total_watermelons : ℕ) (weeks : ℕ) (consumption_per_week : ℕ) (eaten_per_week : ℕ),
  total_watermelons = 30 →
  weeks = 6 →
  eaten_per_week = 3 →
  consumption_per_week = total_watermelons / weeks →
  (consumption_per_week - eaten_per_week) = 2 :=
by
  intros total_watermelons weeks consumption_per_week eaten_per_week h1 h2 h3 h4
  sorry

end jeremy_watermelons_l127_12752


namespace garden_to_land_area_ratio_l127_12792

variables (l_ter w_ter l_gard w_gard : ℝ)

-- Condition 1: Width of the land rectangle is 3/5 of its length
def land_conditions : Prop := w_ter = (3 / 5) * l_ter

-- Condition 2: Width of the garden rectangle is 3/5 of its length
def garden_conditions : Prop := w_gard = (3 / 5) * l_gard

-- Problem: Ratio of the area of the garden to the area of the land is 36%.
theorem garden_to_land_area_ratio
  (h_land : land_conditions l_ter w_ter)
  (h_garden : garden_conditions l_gard w_gard) :
  (l_gard * w_gard) / (l_ter * w_ter) = 0.36 := sorry

end garden_to_land_area_ratio_l127_12792


namespace find_n_values_l127_12785

theorem find_n_values : {n : ℕ | n ≥ 1 ∧ n ≤ 6 ∧ ∃ a b c : ℤ, a^n + b^n = c^n + n} = {1, 2, 3} :=
by sorry

end find_n_values_l127_12785


namespace larger_angle_measure_l127_12779

theorem larger_angle_measure (x : ℝ) (hx : 7 * x = 90) : 4 * x = 360 / 7 := by
sorry

end larger_angle_measure_l127_12779


namespace factor_of_quadratic_implies_m_value_l127_12762

theorem factor_of_quadratic_implies_m_value (m : ℤ) : (∀ x : ℤ, (x + 6) ∣ (x^2 - m * x - 42)) → m = 1 := by
  sorry

end factor_of_quadratic_implies_m_value_l127_12762


namespace div_difference_l127_12757

theorem div_difference {a b n : ℕ} (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) (h : n ∣ a^n - b^n) :
  n ∣ ((a^n - b^n) / (a - b)) :=
by
  sorry

end div_difference_l127_12757


namespace nth_equation_l127_12774

theorem nth_equation (n : ℕ) (h : 0 < n) : (10 * n + 5) ^ 2 = n * (n + 1) * 100 + 5 ^ 2 := 
sorry

end nth_equation_l127_12774


namespace Emma_age_ratio_l127_12781

theorem Emma_age_ratio (E M : ℕ) (h1 : E = E) (h2 : E = E) 
(h3 : E - M = 3 * (E - 4 * M)) : E / M = 11 / 2 :=
sorry

end Emma_age_ratio_l127_12781


namespace basketball_substitution_mod_1000_l127_12794

def basketball_substitution_count_mod (n_playing n_substitutes max_subs : ℕ) : ℕ :=
  let no_subs := 1
  let one_sub := n_playing * n_substitutes
  let two_subs := n_playing * (n_playing - 1) * (n_substitutes * (n_substitutes - 1)) / 2
  let three_subs := n_playing * (n_playing - 1) * (n_playing - 2) *
                    (n_substitutes * (n_substitutes - 1) * (n_substitutes - 2)) / 6
  no_subs + one_sub + two_subs + three_subs 

theorem basketball_substitution_mod_1000 :
  basketball_substitution_count_mod 9 9 3 % 1000 = 10 :=
  by 
    -- Here the proof would be implemented
    sorry

end basketball_substitution_mod_1000_l127_12794


namespace min_value_expression_l127_12718

theorem min_value_expression (x y z : ℝ) (h : x - 2 * y + 2 * z = 5) : (x + 5) ^ 2 + (y - 1) ^ 2 + (z + 3) ^ 2 ≥ 36 :=
by
  sorry

end min_value_expression_l127_12718


namespace sequence_propositions_l127_12717

theorem sequence_propositions (a : ℕ → ℝ) (h_seq : a 1 > a 2 ∧ a 2 > a 3 ∧ a 3 > a 4 ∧ a 4 ≥ 0) 
  (h_sub : ∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ 4 → ∃ k, a i - a j = a k) :
  (∀ k, ∃ d, a k = a 1 - d * (k - 1)) ∧
  (∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 ∧ i * a i = j * a j) ∧
  (∃ i, a i = 0) :=
by
  sorry

end sequence_propositions_l127_12717


namespace polyhedron_volume_l127_12798

/-- Each 12 cm × 12 cm square is cut into two right-angled isosceles triangles by joining the midpoints of two adjacent sides. 
    These six triangles are attached to a regular hexagon to form a polyhedron.
    Prove that the volume of the resulting polyhedron is 864 cubic cm. -/
theorem polyhedron_volume :
  let s : ℝ := 12
  let volume_of_cube := s^3
  let volume_of_polyhedron := volume_of_cube / 2
  volume_of_polyhedron = 864 := 
by
  sorry

end polyhedron_volume_l127_12798


namespace find_r_s_l127_12719

def is_orthogonal (v₁ v₂ : ℝ × ℝ × ℝ) : Prop :=
  v₁.1 * v₂.1 + v₁.2.1 * v₂.2.1 + v₁.2.2 * v₂.2.2 = 0

def have_equal_magnitudes (v₁ v₂ : ℝ × ℝ × ℝ) : Prop :=
  v₁.1^2 + v₁.2.1^2 + v₁.2.2^2 = v₂.1^2 + v₂.2.1^2 + v₂.2.2^2

theorem find_r_s (r s : ℝ) :
  is_orthogonal (4, r, -2) (-1, 2, s) ∧
  have_equal_magnitudes (4, r, -2) (-1, 2, s) →
  r = -11 / 4 ∧ s = -19 / 4 :=
by
  intro h
  sorry

end find_r_s_l127_12719


namespace value_of_expression_l127_12748

theorem value_of_expression : 1 + 3^2 = 10 :=
by
  sorry

end value_of_expression_l127_12748


namespace initial_money_in_wallet_l127_12715

theorem initial_money_in_wallet (x : ℝ) 
  (h1 : x = 78 + 16) : 
  x = 94 :=
by
  sorry

end initial_money_in_wallet_l127_12715


namespace squares_on_grid_l127_12745

-- Defining the problem conditions
def grid_size : ℕ := 5
def total_points : ℕ := grid_size * grid_size
def used_points : ℕ := 20

-- Stating the theorem to prove the total number of squares formed
theorem squares_on_grid : 
  (total_points = 25) ∧ (used_points = 20) →
  (∃ all_squares : ℕ, all_squares = 21) :=
by
  intros
  sorry

end squares_on_grid_l127_12745


namespace jebb_take_home_pay_is_4620_l127_12734

noncomputable def gross_salary : ℤ := 6500
noncomputable def federal_tax (income : ℤ) : ℤ :=
  let tax1 := min income 2000 * 10 / 100
  let tax2 := min (max (income - 2000) 0) 2000 * 15 / 100
  let tax3 := max (income - 4000) 0 * 25 / 100
  tax1 + tax2 + tax3

noncomputable def health_insurance : ℤ := 300
noncomputable def retirement_contribution (income : ℤ) : ℤ := income * 7 / 100

noncomputable def total_deductions (income : ℤ) : ℤ :=
  federal_tax income + health_insurance + retirement_contribution income

noncomputable def take_home_pay (income : ℤ) : ℤ :=
  income - total_deductions income

theorem jebb_take_home_pay_is_4620 : take_home_pay gross_salary = 4620 := by
  sorry

end jebb_take_home_pay_is_4620_l127_12734


namespace charge_per_action_figure_l127_12742

-- Definitions according to given conditions
def cost_of_sneakers : ℕ := 90
def saved_amount : ℕ := 15
def num_action_figures : ℕ := 10
def left_after_purchase : ℕ := 25

-- Theorem to prove the charge per action figure
theorem charge_per_action_figure : 
  (cost_of_sneakers - saved_amount + left_after_purchase) / num_action_figures = 10 :=
by 
  sorry

end charge_per_action_figure_l127_12742


namespace T_bisects_broken_line_l127_12750

def midpoint_arc {α : Type*} [LinearOrderedField α] (A B C : α) : α := (A + B + C) / 2
def projection_perpendicular {α : Type*} [LinearOrderedField α] (F A B C : α) : α := sorry -- Define perpendicular projection T

theorem T_bisects_broken_line {α : Type*} [LinearOrderedField α]
  (A B C : α) (F := midpoint_arc A B C) (T := projection_perpendicular F A B C) :
  T = (A + B + C) / 2 :=
sorry

end T_bisects_broken_line_l127_12750


namespace real_numbers_int_approximation_l127_12729

theorem real_numbers_int_approximation:
  ∀ (x y : ℝ), ∃ (m n : ℤ),
  (x - m) ^ 2 + (y - n) * (x - m) + (y - n) ^ 2 ≤ (1 / 3) :=
by
  intros x y
  sorry

end real_numbers_int_approximation_l127_12729


namespace molecular_weight_correct_l127_12761

-- Define the atomic weights
def atomic_weight_Cu : ℝ := 63.546
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

-- Define the number of atoms in the compound
def num_atoms_Cu : ℕ := 1
def num_atoms_C : ℕ := 1
def num_atoms_O : ℕ := 3

-- Define the molecular weight calculation
def molecular_weight : ℝ :=
  num_atoms_Cu * atomic_weight_Cu + 
  num_atoms_C * atomic_weight_C + 
  num_atoms_O * atomic_weight_O

-- Prove the molecular weight of the compound
theorem molecular_weight_correct : molecular_weight = 123.554 :=
by
  sorry

end molecular_weight_correct_l127_12761


namespace no_linear_term_implies_equal_l127_12746

theorem no_linear_term_implies_equal (m n : ℝ) (h : (x : ℝ) → (x + m) * (x - n) - x^2 - (- mn) = 0) : m = n :=
by
  sorry

end no_linear_term_implies_equal_l127_12746


namespace rope_length_total_l127_12736

theorem rope_length_total :
  let length1 := 24
  let length2 := 20
  let length3 := 14
  let length4 := 12
  length1 + length2 + length3 + length4 = 70 :=
by
  sorry

end rope_length_total_l127_12736


namespace fescue_in_Y_l127_12733

-- Define the weight proportions of the mixtures
def weight_X : ℝ := 0.6667
def weight_Y : ℝ := 0.3333

-- Define the proportion of ryegrass in each mixture
def ryegrass_X : ℝ := 0.40
def ryegrass_Y : ℝ := 0.25

-- Define the proportion of ryegrass in the final mixture
def ryegrass_final : ℝ := 0.35

-- Define the proportion of ryegrass contributed by X and Y to the final mixture
def contrib_X : ℝ := weight_X * ryegrass_X
def contrib_Y : ℝ := weight_Y * ryegrass_Y

-- Define the total proportion of ryegrass in the final mixture
def total_ryegrass : ℝ := contrib_X + contrib_Y

-- The lean theorem stating that the percentage of fescue in Y equals 75%
theorem fescue_in_Y :
  total_ryegrass = ryegrass_final →
  (100 - (ryegrass_Y * 100)) = 75 := 
by
  intros h
  sorry

end fescue_in_Y_l127_12733


namespace gcd_8994_13326_37566_l127_12783

-- Define the integers involved
def a := 8994
def b := 13326
def c := 37566

-- Assert the GCD relation
theorem gcd_8994_13326_37566 : Int.gcd a (Int.gcd b c) = 2 := by
  sorry

end gcd_8994_13326_37566_l127_12783


namespace triangle_area_l127_12799

theorem triangle_area (c b : ℝ) (c_eq : c = 15) (b_eq : b = 9) :
  ∃ a : ℝ, a^2 = c^2 - b^2 ∧ (b * a) / 2 = 54 := by
  sorry

end triangle_area_l127_12799


namespace cos_double_angle_l127_12710

-- Define the hypothesis
def cos_alpha (α : ℝ) : Prop := Real.cos α = 1 / 2

-- State the theorem
theorem cos_double_angle (α : ℝ) (h : cos_alpha α) : Real.cos (2 * α) = -1 / 2 := by
  sorry

end cos_double_angle_l127_12710


namespace linear_function_decreasing_y_l127_12714

theorem linear_function_decreasing_y (x1 y1 y2 : ℝ) :
  y1 = -2 * x1 - 7 → y2 = -2 * (x1 - 1) - 7 → y1 < y2 := by
  intros h1 h2
  sorry

end linear_function_decreasing_y_l127_12714


namespace exists_q_r_polynomials_l127_12730

theorem exists_q_r_polynomials (n : ℕ) (p : Polynomial ℝ) 
  (h_deg : p.degree = n) 
  (h_monic : p.leadingCoeff = 1) :
  ∃ q r : Polynomial ℝ, 
    q.degree = n ∧ r.degree = n ∧ 
    (∀ x : ℝ, q.eval x = 0 → r.eval x = 0) ∧
    (∀ y : ℝ, r.eval y = 0 → q.eval y = 0) ∧
    q.leadingCoeff = 1 ∧ r.leadingCoeff = 1 ∧ 
    p = (q + r) / 2 := 
sorry

end exists_q_r_polynomials_l127_12730


namespace ladder_length_l127_12704

/-- The length of the ladder leaning against a wall when it forms
    a 60 degree angle with the ground and the foot of the ladder 
    is 9.493063650744542 m from the wall is 18.986127301489084 m. -/
theorem ladder_length (L : ℝ) (adjacent : ℝ) (θ : ℝ) (cosθ : ℝ) :
  θ = Real.pi / 3 ∧ adjacent = 9.493063650744542 ∧ cosθ = Real.cos θ →
  L = 18.986127301489084 :=
by
  intro h
  sorry

end ladder_length_l127_12704


namespace ratio_of_speeds_l127_12727

theorem ratio_of_speeds (a b v1 v2 S : ℝ)
  (h1 : S = a * (v1 + v2))
  (h2 : S = b * (v1 - v2)) :
  v2 / v1 = (a + b) / (b - a) :=
by
  sorry

end ratio_of_speeds_l127_12727


namespace slope_angle_tangent_line_at_zero_l127_12768

noncomputable def curve (x : ℝ) : ℝ := 2 * x - Real.exp x

noncomputable def slope_at (x : ℝ) : ℝ := 
  (deriv curve) x

theorem slope_angle_tangent_line_at_zero : 
  Real.arctan (slope_at 0) = Real.pi / 4 :=
by
  sorry

end slope_angle_tangent_line_at_zero_l127_12768
