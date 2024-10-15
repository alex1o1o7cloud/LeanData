import Mathlib

namespace NUMINAMATH_GPT_mistaken_divisor_l1659_165908

theorem mistaken_divisor (x : ℕ) (h : 49 * x = 28 * 21) : x = 12 :=
sorry

end NUMINAMATH_GPT_mistaken_divisor_l1659_165908


namespace NUMINAMATH_GPT_simplify_fraction_l1659_165918

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2 - 4) = 272 / 59 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1659_165918


namespace NUMINAMATH_GPT_initial_volume_of_mixture_l1659_165917

-- Define the conditions of the problem as hypotheses
variable (milk_ratio water_ratio : ℕ) (W : ℕ) (initial_mixture : ℕ)
variable (h1 : milk_ratio = 2) (h2 : water_ratio = 1)
variable (h3 : W = 60)
variable (h4 : water_ratio + milk_ratio = 3) -- The sum of the ratios used in the equation

theorem initial_volume_of_mixture : initial_mixture = 60 :=
by
  sorry

end NUMINAMATH_GPT_initial_volume_of_mixture_l1659_165917


namespace NUMINAMATH_GPT_inequality_xyz_l1659_165958

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 1/x + 1/y + 1/z = 3) : 
  (x - 1) * (y - 1) * (z - 1) ≤ (1/4) * (x * y * z - 1) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_xyz_l1659_165958


namespace NUMINAMATH_GPT_math_problem_l1659_165924

theorem math_problem (a b : ℝ) (h : a * b < 0) : a^2 * |b| - b^2 * |a| + a * b * (|a| - |b|) = 0 :=
sorry

end NUMINAMATH_GPT_math_problem_l1659_165924


namespace NUMINAMATH_GPT_area_computation_l1659_165976

noncomputable def areaOfBoundedFigure : ℝ :=
  let x (t : ℝ) := 2 * Real.sqrt 2 * Real.cos t
  let y (t : ℝ) := 5 * Real.sqrt 2 * Real.sin t
  let rectArea := 20
  let integral := ∫ t in (3 * Real.pi / 4)..(Real.pi / 4), 
    (5 * Real.sqrt 2 * Real.sin t) * (-2 * Real.sqrt 2 * Real.sin t)
  (integral / 2) - rectArea

theorem area_computation :
  let x (t : ℝ) := 2 * Real.sqrt 2 * Real.cos t
  let y (t : ℝ) := 5 * Real.sqrt 2 * Real.sin t
  let rectArea := 20
  let integral := ∫ t in (3 * Real.pi / 4)..(Real.pi / 4),
    (5 * Real.sqrt 2 * Real.sin t) * (-2 * Real.sqrt 2 * Real.sin t)
  ((integral / 2) - rectArea) = (5 * Real.pi - 10) :=
by
  sorry

end NUMINAMATH_GPT_area_computation_l1659_165976


namespace NUMINAMATH_GPT_total_cakes_served_today_l1659_165903

def cakes_served_lunch : ℕ := 6
def cakes_served_dinner : ℕ := 9
def total_cakes_served (lunch cakes_served_dinner : ℕ) : ℕ :=
  lunch + cakes_served_dinner

theorem total_cakes_served_today : total_cakes_served cakes_served_lunch cakes_served_dinner = 15 := 
by
  sorry

end NUMINAMATH_GPT_total_cakes_served_today_l1659_165903


namespace NUMINAMATH_GPT_abs_nested_expression_l1659_165992

theorem abs_nested_expression : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry

end NUMINAMATH_GPT_abs_nested_expression_l1659_165992


namespace NUMINAMATH_GPT_new_area_eq_1_12_original_area_l1659_165901

variable (L W : ℝ)
def increased_length (L : ℝ) : ℝ := 1.40 * L
def decreased_width (W : ℝ) : ℝ := 0.80 * W
def original_area (L W : ℝ) : ℝ := L * W
def new_area (L W : ℝ) : ℝ := (increased_length L) * (decreased_width W)

theorem new_area_eq_1_12_original_area (L W : ℝ) :
  new_area L W = 1.12 * (original_area L W) :=
by
  sorry

end NUMINAMATH_GPT_new_area_eq_1_12_original_area_l1659_165901


namespace NUMINAMATH_GPT_count_special_digits_base7_l1659_165944

theorem count_special_digits_base7 : 
  let n := 2401
  let total_valid_numbers := n - 4^4
  total_valid_numbers = 2145 :=
by
  sorry

end NUMINAMATH_GPT_count_special_digits_base7_l1659_165944


namespace NUMINAMATH_GPT_unique_cell_distance_50_l1659_165957

noncomputable def king_dist (A B: ℤ × ℤ) : ℤ :=
  max (abs (A.1 - B.1)) (abs (A.2 - B.2))

theorem unique_cell_distance_50
  (A B C: ℤ × ℤ)
  (hAB: king_dist A B = 100)
  (hBC: king_dist B C = 100)
  (hCA: king_dist C A = 100) :
  ∃! (X: ℤ × ℤ), king_dist X A = 50 ∧ king_dist X B = 50 ∧ king_dist X C = 50 :=
sorry

end NUMINAMATH_GPT_unique_cell_distance_50_l1659_165957


namespace NUMINAMATH_GPT_jane_age_l1659_165922

theorem jane_age (j : ℕ) 
  (h₁ : ∃ (k : ℕ), j - 2 = k^2)
  (h₂ : ∃ (m : ℕ), j + 2 = m^3) :
  j = 6 :=
sorry

end NUMINAMATH_GPT_jane_age_l1659_165922


namespace NUMINAMATH_GPT_speed_of_stream_l1659_165973

theorem speed_of_stream (v_d v_u : ℝ) (h_d : v_d = 13) (h_u : v_u = 8) :
  (v_d - v_u) / 2 = 2.5 :=
by
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_speed_of_stream_l1659_165973


namespace NUMINAMATH_GPT_malcolm_initial_white_lights_l1659_165927

theorem malcolm_initial_white_lights :
  let red_lights := 12
  let blue_lights := 3 * red_lights
  let green_lights := 6
  let bought_lights := red_lights + blue_lights + green_lights
  let remaining_lights := 5
  let total_needed_lights := bought_lights + remaining_lights
  W = total_needed_lights :=
by
  sorry

end NUMINAMATH_GPT_malcolm_initial_white_lights_l1659_165927


namespace NUMINAMATH_GPT_ratio_green_to_yellow_l1659_165980

theorem ratio_green_to_yellow (yellow fish blue fish green fish total fish : ℕ) 
  (h_yellow : yellow = 12)
  (h_blue : blue = yellow / 2)
  (h_total : total = yellow + blue + green)
  (h_aquarium_total : total = 42) : 
  green / yellow = 2 := 
sorry

end NUMINAMATH_GPT_ratio_green_to_yellow_l1659_165980


namespace NUMINAMATH_GPT_minor_axis_of_ellipse_l1659_165911

noncomputable def length_minor_axis 
    (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) (p3 : ℝ × ℝ) (p4 : ℝ × ℝ) (p5 : ℝ × ℝ) : ℝ :=
if h : (p1, p2, p3, p4, p5) = ((1, 0), (1, 3), (4, 0), (4, 3), (6, 1.5)) then 3 else 0

theorem minor_axis_of_ellipse (p1 p2 p3 p4 p5 : ℝ × ℝ) :
  p1 = (1, 0) → p2 = (1, 3) → p3 = (4, 0) → p4 = (4, 3) → p5 = (6, 1.5) →
  length_minor_axis p1 p2 p3 p4 p5 = 3 :=
by sorry

end NUMINAMATH_GPT_minor_axis_of_ellipse_l1659_165911


namespace NUMINAMATH_GPT_hania_age_in_five_years_l1659_165940

-- Defining the conditions
variables (H S : ℕ)

-- First condition: Samir's age will be 20 in five years
def condition1 : Prop := S + 5 = 20

-- Second condition: Samir is currently half the age Hania was 10 years ago
def condition2 : Prop := S = (H - 10) / 2

-- The statement to prove: Hania's age in five years will be 45
theorem hania_age_in_five_years (H S : ℕ) (h1 : condition1 S) (h2 : condition2 H S) : H + 5 = 45 :=
sorry

end NUMINAMATH_GPT_hania_age_in_five_years_l1659_165940


namespace NUMINAMATH_GPT_parabola_symmetry_product_l1659_165915

theorem parabola_symmetry_product (a p m : ℝ) 
  (hpr1 : a ≠ 0) 
  (hpr2 : p > 0) 
  (hpr3 : ∀ (x₀ y₀ : ℝ), y₀^2 = 2*p*x₀ → (a*(y₀ - m)^2 - 3*(y₀ - m) + 3 = x₀ + m)) :
  a * p * m = -3 := 
sorry

end NUMINAMATH_GPT_parabola_symmetry_product_l1659_165915


namespace NUMINAMATH_GPT_quadratic_root_zero_l1659_165946

theorem quadratic_root_zero (a : ℝ) : 
  ((a-1) * 0^2 + 0 + a^2 - 1 = 0) 
  → a ≠ 1 
  → a = -1 := 
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_quadratic_root_zero_l1659_165946


namespace NUMINAMATH_GPT_find_triplets_l1659_165998

theorem find_triplets (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  (a^m + 1 ∣ (a + 1)^n) ↔ ((a = 1) ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2)) :=
by
  sorry

end NUMINAMATH_GPT_find_triplets_l1659_165998


namespace NUMINAMATH_GPT_smaller_angle_at_10_15_p_m_l1659_165949

-- Definitions of conditions
def clock_hours : ℕ := 12
def degrees_per_hour : ℚ := 360 / clock_hours
def minute_hand_position : ℚ := (15 / 60) * 360
def hour_hand_position : ℚ := 10 * degrees_per_hour + (15 / 60) * degrees_per_hour
def absolute_difference : ℚ := |hour_hand_position - minute_hand_position|
def smaller_angle : ℚ := 360 - absolute_difference

-- Prove that the smaller angle is 142.5°
theorem smaller_angle_at_10_15_p_m : smaller_angle = 142.5 := by
  sorry

end NUMINAMATH_GPT_smaller_angle_at_10_15_p_m_l1659_165949


namespace NUMINAMATH_GPT_proof_sum_of_drawn_kinds_l1659_165959

def kindsGrains : Nat := 40
def kindsVegetableOils : Nat := 10
def kindsAnimalFoods : Nat := 30
def kindsFruitsAndVegetables : Nat := 20
def totalKindsFood : Nat := kindsGrains + kindsVegetableOils + kindsAnimalFoods + kindsFruitsAndVegetables
def sampleSize : Nat := 20
def samplingRatio : Nat := sampleSize / totalKindsFood

def numKindsVegetableOilsDrawn : Nat := kindsVegetableOils / 5
def numKindsFruitsAndVegetablesDrawn : Nat := kindsFruitsAndVegetables / 5
def sumVegetableOilsAndFruitsAndVegetablesDrawn : Nat := numKindsVegetableOilsDrawn + numKindsFruitsAndVegetablesDrawn

theorem proof_sum_of_drawn_kinds : sumVegetableOilsAndFruitsAndVegetablesDrawn = 6 := by
  have h1 : totalKindsFood = 100 := by rfl
  have h2 : samplingRatio = 1 / 5 := by
    calc
      sampleSize / totalKindsFood
      _ = 20 / 100 := rfl
      _ = 1 / 5 := by norm_num
  have h3 : numKindsVegetableOilsDrawn = 2 := by
    calc
      kindsVegetableOils / 5
      _ = 10 / 5 := rfl
      _ = 2 := by norm_num
  have h4 : numKindsFruitsAndVegetablesDrawn = 4 := by
    calc
      kindsFruitsAndVegetables / 5
      _ = 20 / 5 := rfl
      _ = 4 := by norm_num
  calc
    sumVegetableOilsAndFruitsAndVegetablesDrawn
    _ = numKindsVegetableOilsDrawn + numKindsFruitsAndVegetablesDrawn := rfl
    _ = 2 + 4 := by rw [h3, h4]
    _ = 6 := by norm_num

end NUMINAMATH_GPT_proof_sum_of_drawn_kinds_l1659_165959


namespace NUMINAMATH_GPT_no_solution_intervals_l1659_165981

theorem no_solution_intervals (a : ℝ) :
  (a < -17 ∨ a > 0) → ¬∃ x : ℝ, 7 * |x - 4 * a| + |x - a^2| + 6 * x - 3 * a = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_intervals_l1659_165981


namespace NUMINAMATH_GPT_joseph_savings_ratio_l1659_165938

theorem joseph_savings_ratio
    (thomas_monthly_savings : ℕ)
    (thomas_years_saving : ℕ)
    (total_savings : ℕ)
    (joseph_total_savings_is_total_minus_thomas : total_savings = thomas_monthly_savings * 12 * thomas_years_saving + (total_savings - thomas_monthly_savings * 12 * thomas_years_saving))
    (thomas_saves_each_month : thomas_monthly_savings = 40)
    (years_saving : thomas_years_saving = 6)
    (total_amount : total_savings = 4608) :
    (total_savings - thomas_monthly_savings * 12 * thomas_years_saving) / (12 * thomas_years_saving) / thomas_monthly_savings = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_joseph_savings_ratio_l1659_165938


namespace NUMINAMATH_GPT_locus_of_M_is_ellipse_l1659_165964

theorem locus_of_M_is_ellipse :
  ∀ (a b : ℝ) (M : ℝ × ℝ),
  a > b → b > 0 → (∃ x y : ℝ, 
  (M = (x, y)) ∧ 
  ∃ (P : ℝ × ℝ),
  (∃ x0 y0 : ℝ, P = (x0, y0) ∧ (x0^2 / a^2 + y0^2 / b^2 = 1)) ∧ 
  P ≠ (a, 0) ∧ P ≠ (-a, 0) ∧
  (∃ t : ℝ, t = (x^2 + y^2 - a^2) / (2 * y)) ∧ 
  (∃ x0 y0 : ℝ, 
    x0 = -x ∧ 
    y0 = 2 * t - y ∧
    x0^2 / a^2 + y0^2 / b^2 = 1)) →
  ∃ (x y : ℝ),
  M = (x, y) ∧ 
  (x^2 / a^2 + y^2 / (a^4 / b^2) = 1) := 
sorry

end NUMINAMATH_GPT_locus_of_M_is_ellipse_l1659_165964


namespace NUMINAMATH_GPT_fraction_subtraction_l1659_165943

theorem fraction_subtraction :
  (12 / 30) - (1 / 7) = 9 / 35 :=
by sorry

end NUMINAMATH_GPT_fraction_subtraction_l1659_165943


namespace NUMINAMATH_GPT_profit_percentage_l1659_165954

variable {C S : ℝ}

theorem profit_percentage (h : 19 * C = 16 * S) :
  ((S - C) / C) * 100 = 18.75 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_l1659_165954


namespace NUMINAMATH_GPT_rational_solutions_quad_eq_iff_k_eq_4_l1659_165952

theorem rational_solutions_quad_eq_iff_k_eq_4 (k : ℕ) (hk : 0 < k) : 
  (∃ x : ℚ, x^2 + 24/k * x + 9 = 0) ↔ k = 4 :=
sorry

end NUMINAMATH_GPT_rational_solutions_quad_eq_iff_k_eq_4_l1659_165952


namespace NUMINAMATH_GPT_ganesh_average_speed_l1659_165945

variable (D : ℝ) (hD : D > 0)

/-- Ganesh's average speed over the entire journey is 45 km/hr.
    Given:
    - Speed from X to Y is 60 km/hr
    - Speed from Y to X is 36 km/hr
--/
theorem ganesh_average_speed :
  let T1 := D / 60
  let T2 := D / 36
  let total_distance := 2 * D
  let total_time := T1 + T2
  (total_distance / total_time) = 45 :=
by
  sorry

end NUMINAMATH_GPT_ganesh_average_speed_l1659_165945


namespace NUMINAMATH_GPT_value_at_2007_l1659_165966

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom symmetric_property (x : ℝ) : f (2 + x) = f (2 - x)
axiom specific_value : f (-3) = -2

theorem value_at_2007 : f 2007 = -2 :=
sorry

end NUMINAMATH_GPT_value_at_2007_l1659_165966


namespace NUMINAMATH_GPT_maximum_value_l1659_165910

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem maximum_value (a b c : ℝ) (h_a : 1 ≤ a ∧ a ≤ 2)
  (h_f1 : f a b c 1 ≤ 1) (h_f2 : f a b c 2 ≤ 1) :
  7 * b + 5 * c ≤ -6 :=
sorry

end NUMINAMATH_GPT_maximum_value_l1659_165910


namespace NUMINAMATH_GPT_smallest_n_for_divisibility_l1659_165971

theorem smallest_n_for_divisibility (n : ℕ) (h1 : 24 ∣ n^2) (h2 : 1080 ∣ n^3) : n = 120 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_divisibility_l1659_165971


namespace NUMINAMATH_GPT_tile_border_ratio_l1659_165947

theorem tile_border_ratio (n : ℕ) (t w : ℝ) (H1 : n = 30)
  (H2 : 900 * t^2 / (30 * t + 30 * w)^2 = 0.81) :
  w / t = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_tile_border_ratio_l1659_165947


namespace NUMINAMATH_GPT_soup_adult_feeding_l1659_165979

theorem soup_adult_feeding (cans_of_soup : ℕ) (cans_for_children : ℕ) (feeding_ratio : ℕ) 
  (children : ℕ) (adults : ℕ) :
  feeding_ratio = 4 → cans_of_soup = 10 → children = 20 →
  cans_for_children = (children / feeding_ratio) → 
  adults = feeding_ratio * (cans_of_soup - cans_for_children) →
  adults = 20 :=
by
  intros h1 h2 h3 h4 h5
  -- proof goes here
  sorry

end NUMINAMATH_GPT_soup_adult_feeding_l1659_165979


namespace NUMINAMATH_GPT_real_roots_of_cubic_equation_l1659_165970

theorem real_roots_of_cubic_equation : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, (x^3 - 2 * x + 1)^2 = 9) ∧ S.card = 2 := 
by
  sorry

end NUMINAMATH_GPT_real_roots_of_cubic_equation_l1659_165970


namespace NUMINAMATH_GPT_cubic_polynomial_coefficients_l1659_165948

theorem cubic_polynomial_coefficients (f g : Polynomial ℂ) (b c d : ℂ) :
  f = Polynomial.C 4 + Polynomial.X * (Polynomial.C 3 + Polynomial.X * (Polynomial.C 2 + Polynomial.X)) →
  (∀ x, Polynomial.eval x f = 0 → Polynomial.eval (x^2) g = 0) →
  g = Polynomial.C d + Polynomial.X * (Polynomial.C c + Polynomial.X * (Polynomial.C b + Polynomial.X)) →
  (b, c, d) = (4, -15, -32) :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_cubic_polynomial_coefficients_l1659_165948


namespace NUMINAMATH_GPT_rectangle_dimensions_l1659_165987

theorem rectangle_dimensions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_area : x * y = 36) (h_perimeter : 2 * x + 2 * y = 30) : 
  (x = 12 ∧ y = 3) ∨ (x = 3 ∧ y = 12) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l1659_165987


namespace NUMINAMATH_GPT_solve_problem_l1659_165929

noncomputable def solution_set : Set ℤ := {x | abs (7 * x - 5) ≤ 9}

theorem solve_problem : solution_set = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_solve_problem_l1659_165929


namespace NUMINAMATH_GPT_find_side_length_l1659_165900

theorem find_side_length
  (a b c : ℝ) 
  (cosine_diff_angle : ℝ) 
  (h_b : b = 5)
  (h_c : c = 4)
  (h_cosine_diff_angle : cosine_diff_angle = 31 / 32) :
  a = 6 := 
sorry

end NUMINAMATH_GPT_find_side_length_l1659_165900


namespace NUMINAMATH_GPT_find_first_number_l1659_165963

theorem find_first_number (x : ℝ) :
  (20 + 40 + 60) / 3 = (x + 70 + 13) / 3 + 9 → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_l1659_165963


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1659_165996

variable (a b c : ℝ) (h_iso : a = b ∨ a = c ∨ b = c) (h_a : a = 6) (h_b : b = 6) (h_c : c = 3)
variable (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)

theorem isosceles_triangle_perimeter : a + b + c = 15 :=
by 
  -- Given definitions and triangle inequality
  have h_valid : a = 6 ∧ b = 6 ∧ c = 3 := ⟨h_a, h_b, h_c⟩
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1659_165996


namespace NUMINAMATH_GPT_contrapositive_l1659_165988

variable (Line Circle : Type) (distance : Line → Circle → ℝ) (radius : Circle → ℝ)
variable (is_tangent : Line → Circle → Prop)

-- Original proposition in Lean notation:
def original_proposition (l : Line) (c : Circle) : Prop :=
  distance l c ≠ radius c → ¬ is_tangent l c

-- Contrapositive of the original proposition:
theorem contrapositive (l : Line) (c : Circle) : Prop :=
  is_tangent l c → distance l c = radius c

end NUMINAMATH_GPT_contrapositive_l1659_165988


namespace NUMINAMATH_GPT_suff_not_necessary_no_real_solutions_l1659_165974

theorem suff_not_necessary_no_real_solutions :
  ∀ m : ℝ, |m| < 1 → (m : ℝ)^2 < 4 ∧ ∃ x, x^2 - m * x + 1 = 0 →
  ∀ a b : ℝ, (a = 1) ∧ (b = -m) ∧ (c = 1) → (b^2 - 4 * a * c) < 0 ∧ (m > -2) ∧ (m < 2) :=
by
  sorry

end NUMINAMATH_GPT_suff_not_necessary_no_real_solutions_l1659_165974


namespace NUMINAMATH_GPT_monogramming_cost_per_stocking_l1659_165930

noncomputable def total_stockings : ℕ := (5 * 5) + 4
noncomputable def price_per_stocking : ℝ := 20 - (0.10 * 20)
noncomputable def total_cost_of_stockings : ℝ := total_stockings * price_per_stocking
noncomputable def total_cost : ℝ := 1035
noncomputable def total_monogramming_cost : ℝ := total_cost - total_cost_of_stockings

theorem monogramming_cost_per_stocking :
  (total_monogramming_cost / total_stockings) = 17.69 :=
by
  sorry

end NUMINAMATH_GPT_monogramming_cost_per_stocking_l1659_165930


namespace NUMINAMATH_GPT_tan_alpha_eq_one_third_cos2alpha_over_expr_l1659_165951

theorem tan_alpha_eq_one_third_cos2alpha_over_expr (α : ℝ) (h : Real.tan α = 1/3) :
  (Real.cos (2 * α)) / (2 * Real.sin α * Real.cos α + (Real.cos α)^2) = 8 / 15 :=
by
  -- This is the point where the proof steps will go, but we leave it as a placeholder.
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_one_third_cos2alpha_over_expr_l1659_165951


namespace NUMINAMATH_GPT_tory_earned_more_than_bert_l1659_165934

open Real

noncomputable def bert_day1_earnings : ℝ :=
  let initial_sales := 12 * 18
  let discounted_sales := 3 * (18 - 0.15 * 18)
  let total_sales := initial_sales - 3 * 18 + discounted_sales
  total_sales * 0.95

noncomputable def tory_day1_earnings : ℝ :=
  let initial_sales := 15 * 20
  let discounted_sales := 5 * (20 - 0.10 * 20)
  let total_sales := initial_sales - 5 * 20 + discounted_sales
  total_sales * 0.95

noncomputable def bert_day2_earnings : ℝ :=
  let sales := 10 * 15
  (sales * 0.95) * 1.4

noncomputable def tory_day2_earnings : ℝ :=
  let sales := 8 * 18
  (sales * 0.95) * 1.4

noncomputable def bert_total_earnings : ℝ := bert_day1_earnings + bert_day2_earnings

noncomputable def tory_total_earnings : ℝ := tory_day1_earnings + tory_day2_earnings

noncomputable def earnings_difference : ℝ := tory_total_earnings - bert_total_earnings

theorem tory_earned_more_than_bert :
  earnings_difference = 71.82 := by
  sorry

end NUMINAMATH_GPT_tory_earned_more_than_bert_l1659_165934


namespace NUMINAMATH_GPT_g_f_of_3_l1659_165965

def f (x : ℝ) : ℝ := x^3 - 4
def g (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 2

theorem g_f_of_3 : g (f 3) = 1704 := by
  sorry

end NUMINAMATH_GPT_g_f_of_3_l1659_165965


namespace NUMINAMATH_GPT_pentagon_area_calc_l1659_165961

noncomputable def pentagon_area : ℝ :=
  let triangle1 := (1 / 2) * 18 * 22
  let triangle2 := (1 / 2) * 30 * 26
  let trapezoid := (1 / 2) * (22 + 30) * 10
  triangle1 + triangle2 + trapezoid

theorem pentagon_area_calc :
  pentagon_area = 848 := by
  sorry

end NUMINAMATH_GPT_pentagon_area_calc_l1659_165961


namespace NUMINAMATH_GPT_extremum_values_of_function_l1659_165969

noncomputable def maxValue := Real.sqrt 2 + 1 / Real.sqrt 2
noncomputable def minValue := -Real.sqrt 2 + 1 / Real.sqrt 2

theorem extremum_values_of_function :
  ∀ x : ℝ, - (Real.sqrt 2) + (1 / Real.sqrt 2) ≤ (Real.sin x + Real.cos x + 1 / Real.sqrt (1 + |Real.sin (2 * x)|)) ∧ 
            (Real.sin x + Real.cos x + 1 / Real.sqrt (1 + |Real.sin (2 * x)|)) ≤ (Real.sqrt 2 + 1 / Real.sqrt 2) := 
by
  sorry

end NUMINAMATH_GPT_extremum_values_of_function_l1659_165969


namespace NUMINAMATH_GPT_molecular_weight_is_correct_l1659_165932

-- Define the masses of the individual isotopes
def H1 : ℕ := 1
def H2 : ℕ := 2
def O : ℕ := 16
def C : ℕ := 13
def N : ℕ := 15
def S : ℕ := 33

-- Define the molecular weight calculation
def molecular_weight : ℕ := (2 * H1) + H2 + O + C + N + S

-- The goal is to prove that the calculated molecular weight is 81
theorem molecular_weight_is_correct : molecular_weight = 81 :=
by 
  sorry

end NUMINAMATH_GPT_molecular_weight_is_correct_l1659_165932


namespace NUMINAMATH_GPT_tangent_slope_is_four_l1659_165942

-- Define the given curve and point
def curve (x : ℝ) : ℝ := 2 * x^2
def point : ℝ × ℝ := (1, 2)

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Define the tangent slope at the given point
def tangent_slope_at_point : ℝ := curve_derivative 1

-- Prove that the tangent slope at point (1, 2) is 4
theorem tangent_slope_is_four : tangent_slope_at_point = 4 :=
by
  -- We state that the slope at x = 1 is 4
  sorry

end NUMINAMATH_GPT_tangent_slope_is_four_l1659_165942


namespace NUMINAMATH_GPT_smallest_angle_half_largest_l1659_165972

open Real

-- Statement of the problem
theorem smallest_angle_half_largest (a b c : ℝ) (α β γ : ℝ)
  (h_sides : a = 4 ∧ b = 5 ∧ c = 6)
  (h_angles : α < β ∧ β < γ)
  (h_cos_alpha : cos α = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_cos_gamma : cos γ = (a^2 + b^2 - c^2) / (2 * a * b)) :
  2 * α = γ := 
sorry

end NUMINAMATH_GPT_smallest_angle_half_largest_l1659_165972


namespace NUMINAMATH_GPT_value_of_expression_l1659_165956

def expr : ℕ :=
  8 + 2 * (3^2)

theorem value_of_expression : expr = 26 :=
  by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1659_165956


namespace NUMINAMATH_GPT_random_phenomenon_l1659_165955

def is_certain_event (P : Prop) : Prop := ∀ h : P, true

def is_random_event (P : Prop) : Prop := ¬is_certain_event P

def scenario1 : Prop := ∀ pressure temperature : ℝ, (pressure = 101325) → (temperature = 100) → true
-- Under standard atmospheric pressure, water heated to 100°C will boil

def scenario2 : Prop := ∃ time : ℝ, true
-- Encountering a red light at a crossroads (which happens at random times)

def scenario3 (a b : ℝ) : Prop := true
-- For a rectangle with length and width a and b respectively, its area is a * b

def scenario4 : Prop := ∀ a b : ℝ, ∃ x : ℝ, a * x + b = 0
-- A linear equation with real coefficients always has one real root

theorem random_phenomenon : is_random_event scenario2 :=
by
  sorry

end NUMINAMATH_GPT_random_phenomenon_l1659_165955


namespace NUMINAMATH_GPT_Alex_dimes_l1659_165936

theorem Alex_dimes : 
    ∃ (d q : ℕ), 10 * d + 25 * q = 635 ∧ d = q + 5 ∧ d = 22 :=
by sorry

end NUMINAMATH_GPT_Alex_dimes_l1659_165936


namespace NUMINAMATH_GPT_digit_expression_equals_2021_l1659_165941

theorem digit_expression_equals_2021 :
  ∃ (f : ℕ → ℕ), 
  (f 0 = 0 ∧
   f 1 = 1 ∧
   f 2 = 2 ∧
   f 3 = 3 ∧
   f 4 = 4 ∧
   f 5 = 5 ∧
   f 6 = 6 ∧
   f 7 = 7 ∧
   f 8 = 8 ∧
   f 9 = 9 ∧
   43 * (8 * 5 + 7) + 0 * 1 * 2 * 6 * 9 = 2021) :=
sorry

end NUMINAMATH_GPT_digit_expression_equals_2021_l1659_165941


namespace NUMINAMATH_GPT_complex_purely_imaginary_condition_l1659_165962

theorem complex_purely_imaginary_condition (a : ℝ) :
  (a = 1 → (a - 1) * (a + 2) + (a + 3) * Complex.I.im = (0 : ℝ)) ∧
  ¬(a = 1 ∧ ¬a = -2 → (a - 1) * (a + 2) + (a + 3) * Complex.I.im = (0 : ℝ)) :=
  sorry

end NUMINAMATH_GPT_complex_purely_imaginary_condition_l1659_165962


namespace NUMINAMATH_GPT_solve_for_x_l1659_165982

theorem solve_for_x : ∀ (x : ℝ), (x ≠ 3) → ((x - 3) / (x + 2) + (3 * x - 6) / (x - 3) = 2) → x = 1 / 2 := 
by
  intros x hx h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1659_165982


namespace NUMINAMATH_GPT_determine_b_l1659_165933

noncomputable def f (x b : ℝ) : ℝ := x^3 - b * x^2 + 1/2

theorem determine_b (b : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 b = 0 ∧ f x2 b = 0) → b = 3/2 :=
by
  sorry

end NUMINAMATH_GPT_determine_b_l1659_165933


namespace NUMINAMATH_GPT_blue_socks_count_l1659_165993

-- Defining the total number of socks
def total_socks : ℕ := 180

-- Defining the number of white socks as two thirds of the total socks
def white_socks : ℕ := (2 * total_socks) / 3

-- Defining the number of blue socks as the difference between total socks and white socks
def blue_socks : ℕ := total_socks - white_socks

-- The theorem to prove
theorem blue_socks_count : blue_socks = 60 := by
  sorry

end NUMINAMATH_GPT_blue_socks_count_l1659_165993


namespace NUMINAMATH_GPT_jose_completion_time_l1659_165985

noncomputable def rate_jose : ℚ := 1 / 30
noncomputable def rate_jane : ℚ := 1 / 6

theorem jose_completion_time :
  ∀ (J A : ℚ), 
    (J + A = 1 / 5) ∧ (J = rate_jose) ∧ (A = rate_jane) → 
    (1 / J = 30) :=
by
  intros J A h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end NUMINAMATH_GPT_jose_completion_time_l1659_165985


namespace NUMINAMATH_GPT_central_angle_unchanged_l1659_165912

theorem central_angle_unchanged (r s : ℝ) (h1 : r ≠ 0) (h2 : s ≠ 0) :
  (s / r) = (2 * s / (2 * r)) :=
by
  sorry

end NUMINAMATH_GPT_central_angle_unchanged_l1659_165912


namespace NUMINAMATH_GPT_age_difference_l1659_165935

theorem age_difference (B_age : ℕ) (A_age : ℕ) (X : ℕ) : 
  B_age = 42 → 
  A_age = B_age + 12 → 
  A_age + 10 = 2 * (B_age - X) → 
  X = 10 :=
by
  intros hB_age hA_age hEquation 
  -- define variables based on conditions
  have hB : B_age = 42 := hB_age
  have hA : A_age = B_age + 12 := hA_age
  have hEq : A_age + 10 = 2 * (B_age - X) := hEquation
  -- expected result
  sorry

end NUMINAMATH_GPT_age_difference_l1659_165935


namespace NUMINAMATH_GPT_carl_garden_area_l1659_165923

theorem carl_garden_area (total_posts : ℕ) (post_interval : ℕ) (x_posts_on_shorter : ℕ) (y_posts_on_longer : ℕ)
  (h1 : total_posts = 26)
  (h2 : post_interval = 5)
  (h3 : y_posts_on_longer = 2 * x_posts_on_shorter)
  (h4 : 2 * x_posts_on_shorter + 2 * y_posts_on_longer - 4 = total_posts) :
  (x_posts_on_shorter - 1) * post_interval * (y_posts_on_longer - 1) * post_interval = 900 := 
by
  sorry

end NUMINAMATH_GPT_carl_garden_area_l1659_165923


namespace NUMINAMATH_GPT_value_of_x_l1659_165905

theorem value_of_x 
    (r : ℝ) (a : ℝ) (x : ℝ) (shaded_area : ℝ)
    (h1 : r = 2)
    (h2 : a = 2)
    (h3 : shaded_area = 2) :
  x = (Real.pi / 3) + (Real.sqrt 3 / 2) - 1 :=
sorry

end NUMINAMATH_GPT_value_of_x_l1659_165905


namespace NUMINAMATH_GPT_remainder_when_690_div_170_l1659_165907

theorem remainder_when_690_div_170 :
  ∃ r : ℕ, ∃ k l : ℕ, 
    gcd (690 - r) (875 - 25) = 170 ∧
    r = 690 % 170 ∧
    l = 875 / 170 ∧
    r = 10 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_when_690_div_170_l1659_165907


namespace NUMINAMATH_GPT_geometric_sequence_a4_l1659_165909

-- Define the geometric sequence and known conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

variables (a : ℕ → ℝ) (q : ℝ)

-- Given conditions:
def a2_eq_4 : Prop := a 2 = 4
def a6_eq_16 : Prop := a 6 = 16

-- The goal is to show a 4 = 8 given the conditions
theorem geometric_sequence_a4 (h_seq : geometric_sequence a q)
  (h_a2 : a2_eq_4 a)
  (h_a6 : a6_eq_16 a) : a 4 = 8 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a4_l1659_165909


namespace NUMINAMATH_GPT_probability_cd_l1659_165983

theorem probability_cd (P_A P_B : ℚ) (h1 : P_A = 1/4) (h2 : P_B = 1/3) :
  (1 - P_A - P_B = 5/12) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_probability_cd_l1659_165983


namespace NUMINAMATH_GPT_kayak_manufacture_total_l1659_165928

theorem kayak_manufacture_total :
  let feb : ℕ := 5
  let mar : ℕ := 3 * feb
  let apr : ℕ := 3 * mar
  let may : ℕ := 3 * apr
  feb + mar + apr + may = 200 := by
  sorry

end NUMINAMATH_GPT_kayak_manufacture_total_l1659_165928


namespace NUMINAMATH_GPT_total_apples_count_l1659_165999

-- Definitions based on conditions
def red_apples := 16
def green_apples := red_apples + 12
def total_apples := green_apples + red_apples

-- Statement to prove
theorem total_apples_count : total_apples = 44 := 
by
  sorry

end NUMINAMATH_GPT_total_apples_count_l1659_165999


namespace NUMINAMATH_GPT_right_triangle_sets_l1659_165939

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

end NUMINAMATH_GPT_right_triangle_sets_l1659_165939


namespace NUMINAMATH_GPT_correct_statement_is_B_l1659_165925

-- Define integers and zero
def is_integer (n : ℤ) : Prop := True
def is_zero (n : ℤ) : Prop := n = 0

-- Define rational numbers
def is_rational (q : ℚ) : Prop := True

-- Positive and negative zero cannot co-exist
def is_positive (n : ℤ) : Prop := n > 0
def is_negative (n : ℤ) : Prop := n < 0

-- Statement A: Integers and negative integers are collectively referred to as integers.
def statement_A : Prop :=
  ∀ n : ℤ, (is_positive n ∨ is_negative n) ↔ is_integer n

-- Statement B: Integers and fractions are collectively referred to as rational numbers.
def statement_B : Prop :=
  ∀ q : ℚ, is_rational q

-- Statement C: Zero can be either a positive integer or a negative integer.
def statement_C : Prop :=
  ∀ n : ℤ, is_zero n → (is_positive n ∨ is_negative n)

-- Statement D: A rational number is either a positive number or a negative number.
def statement_D : Prop :=
  ∀ q : ℚ, (q ≠ 0 → (is_positive q.num ∨ is_negative q.num))

-- The problem is to prove that statement B is the only correct statement.
theorem correct_statement_is_B : statement_B ∧ ¬statement_A ∧ ¬statement_C ∧ ¬statement_D :=
by sorry

end NUMINAMATH_GPT_correct_statement_is_B_l1659_165925


namespace NUMINAMATH_GPT_sum_rational_irrational_not_rational_l1659_165914

theorem sum_rational_irrational_not_rational (r i : ℚ) (hi : ¬ ∃ q : ℚ, i = q) : ¬ ∃ s : ℚ, r + i = s :=
by
  sorry

end NUMINAMATH_GPT_sum_rational_irrational_not_rational_l1659_165914


namespace NUMINAMATH_GPT_find_abc_sum_l1659_165904

theorem find_abc_sum (a b c : ℕ) (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 294) : a + b + c = 8 :=
sorry

end NUMINAMATH_GPT_find_abc_sum_l1659_165904


namespace NUMINAMATH_GPT_no_positive_integer_n_for_perfect_squares_l1659_165921

theorem no_positive_integer_n_for_perfect_squares :
  ∀ (n : ℕ), 0 < n → ¬ (∃ a b : ℤ, (n + 1) * 2^n = a^2 ∧ (n + 3) * 2^(n + 2) = b^2) :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_n_for_perfect_squares_l1659_165921


namespace NUMINAMATH_GPT_triangle_perimeter_l1659_165967

theorem triangle_perimeter (x : ℕ) (h_odd : x % 2 = 1) (h_range : 1 < x ∧ x < 5) : 2 + 3 + x = 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1659_165967


namespace NUMINAMATH_GPT_count_4_digit_numbers_divisible_by_13_l1659_165953

theorem count_4_digit_numbers_divisible_by_13 : 
  ∃ n : ℕ, n = 693 ∧ (∀ k : ℕ, k >= 1000 ∧ k < 10000 ∧ k % 13 = 0 → ∃ m : ℕ, m = (k - 1000) / 13 + 1 ∧ m = n) :=
by {
  -- Solution proof will be placed here.
  sorry
}

end NUMINAMATH_GPT_count_4_digit_numbers_divisible_by_13_l1659_165953


namespace NUMINAMATH_GPT_part_a_solution_part_b_solution_l1659_165989

-- Part (a)
theorem part_a_solution (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y + 13 = 0 ↔ (x = 2 ∧ y = -3) :=
sorry

-- Part (b)
theorem part_b_solution (x y : ℝ) :
  xy - 1 = x - y ↔ ((x = 1 ∨ y = 1) ∨ (x ≠ 1 ∧ y ≠ 1)) :=
sorry

end NUMINAMATH_GPT_part_a_solution_part_b_solution_l1659_165989


namespace NUMINAMATH_GPT_number_of_groups_is_correct_l1659_165994

-- Define the number of students
def number_of_students : ℕ := 16

-- Define the group size
def group_size : ℕ := 4

-- Define the expected number of groups
def expected_number_of_groups : ℕ := 4

-- Prove the expected number of groups when grouping students into groups of four
theorem number_of_groups_is_correct :
  number_of_students / group_size = expected_number_of_groups := by
  sorry

end NUMINAMATH_GPT_number_of_groups_is_correct_l1659_165994


namespace NUMINAMATH_GPT_proof_problem_l1659_165995

variable (a b c d x : ℤ)

-- Conditions
def are_opposite (a b : ℤ) : Prop := a + b = 0
def are_reciprocals (c d : ℤ) : Prop := c * d = 1
def largest_negative_integer (x : ℤ) : Prop := x = -1

theorem proof_problem 
  (h1 : are_opposite a b) 
  (h2 : are_reciprocals c d) 
  (h3 : largest_negative_integer x) :
  x^2 - (a + b - c * d)^(2012 : ℕ) + (-c * d)^(2011 : ℕ) = -1 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1659_165995


namespace NUMINAMATH_GPT_braxton_total_earnings_l1659_165906

-- Definitions of the given problem conditions
def students_ashwood : ℕ := 9
def days_ashwood : ℕ := 4
def students_braxton : ℕ := 6
def days_braxton : ℕ := 7
def students_cedar : ℕ := 8
def days_cedar : ℕ := 6

def total_payment : ℕ := 1080
def daily_wage_per_student : ℚ := total_payment / ((students_ashwood * days_ashwood) + 
                                                   (students_braxton * days_braxton) + 
                                                   (students_cedar * days_cedar))

-- The statement to be proven
theorem braxton_total_earnings :
  (students_braxton * days_braxton * daily_wage_per_student) = 360 := 
by
  sorry -- proof goes here

end NUMINAMATH_GPT_braxton_total_earnings_l1659_165906


namespace NUMINAMATH_GPT_difference_of_roots_l1659_165931

theorem difference_of_roots : 
  let a := 6 + 3 * Real.sqrt 5
  let b := 3 + Real.sqrt 5
  let c := 1
  ∃ x1 x2 : ℝ, (a * x1^2 - b * x1 + c = 0) ∧ (a * x2^2 - b * x2 + c = 0) ∧ x1 ≠ x2 
  ∧ x1 > x2 ∧ (x1 - x2) = (Real.sqrt 6 - Real.sqrt 5) / 3 := 
sorry

end NUMINAMATH_GPT_difference_of_roots_l1659_165931


namespace NUMINAMATH_GPT_translate_line_upwards_l1659_165997

theorem translate_line_upwards {x y : ℝ} (h : y = -2 * x + 1) :
  y = -2 * x + 3 := by
  sorry

end NUMINAMATH_GPT_translate_line_upwards_l1659_165997


namespace NUMINAMATH_GPT_mutually_exclusive_event_is_D_l1659_165913

namespace Problem

def event_A (n : ℕ) (defective : ℕ) : Prop := defective ≥ 2
def mutually_exclusive_event (n : ℕ) : Prop := (∀ (defective : ℕ), defective ≤ 1) ↔ (∀ (defective : ℕ), defective ≥ 2 → false)

theorem mutually_exclusive_event_is_D (n : ℕ) : mutually_exclusive_event n := 
by 
  sorry

end Problem

end NUMINAMATH_GPT_mutually_exclusive_event_is_D_l1659_165913


namespace NUMINAMATH_GPT_vertex_of_parabola_l1659_165968

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := (x + 2)^2 - 1

-- Define the vertex point
def vertex : ℝ × ℝ := (-2, -1)

-- The theorem we need to prove
theorem vertex_of_parabola : ∀ x : ℝ, parabola x = (x + 2)^2 - 1 → vertex = (-2, -1) := 
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1659_165968


namespace NUMINAMATH_GPT_range_of_a_l1659_165960

noncomputable def A : Set ℝ := Set.Ico 1 5 -- A = [1, 5)
noncomputable def B (a : ℝ) : Set ℝ := Set.Iio a -- B = (-∞, a)

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 5 ≤ a :=
sorry

end NUMINAMATH_GPT_range_of_a_l1659_165960


namespace NUMINAMATH_GPT_corrected_mean_l1659_165984

theorem corrected_mean (mean : ℝ) (num_observations : ℕ) 
  (incorrect_observation correct_observation : ℝ)
  (h_mean : mean = 36) (h_num_observations : num_observations = 50)
  (h_incorrect_observation : incorrect_observation = 23) 
  (h_correct_observation : correct_observation = 44)
  : (mean * num_observations + (correct_observation - incorrect_observation)) / num_observations = 36.42 := 
by
  sorry

end NUMINAMATH_GPT_corrected_mean_l1659_165984


namespace NUMINAMATH_GPT_three_x_squared_y_squared_eq_588_l1659_165926

theorem three_x_squared_y_squared_eq_588 (x y : ℤ) 
  (h : y^2 + 3 * x^2 * y^2 = 30 * x^2 + 517) : 
  3 * x^2 * y^2 = 588 :=
sorry

end NUMINAMATH_GPT_three_x_squared_y_squared_eq_588_l1659_165926


namespace NUMINAMATH_GPT_calculation_correct_l1659_165990

theorem calculation_correct : 469111 * 9999 = 4690428889 := 
by sorry

end NUMINAMATH_GPT_calculation_correct_l1659_165990


namespace NUMINAMATH_GPT_probability_two_slate_rocks_l1659_165978

theorem probability_two_slate_rocks 
    (n_slate : ℕ) (n_pumice : ℕ) (n_granite : ℕ)
    (h_slate : n_slate = 12)
    (h_pumice : n_pumice = 16)
    (h_granite : n_granite = 8) :
    (n_slate / (n_slate + n_pumice + n_granite)) * ((n_slate - 1) / (n_slate + n_pumice + n_granite - 1)) = 11 / 105 :=
by
    sorry

end NUMINAMATH_GPT_probability_two_slate_rocks_l1659_165978


namespace NUMINAMATH_GPT_number_of_hens_l1659_165991

theorem number_of_hens
    (H C : ℕ) -- Hens and Cows
    (h1 : H + C = 44) -- Condition 1: The number of heads
    (h2 : 2 * H + 4 * C = 128) -- Condition 2: The number of feet
    : H = 24 :=
by
  sorry

end NUMINAMATH_GPT_number_of_hens_l1659_165991


namespace NUMINAMATH_GPT_value_of_a_minus_b_l1659_165916

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a + b > 0) :
  (a - b = -1) ∨ (a - b = -7) :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l1659_165916


namespace NUMINAMATH_GPT_bottles_produced_by_10_machines_in_4_minutes_l1659_165919

variable (rate_per_machine : ℕ)
variable (total_bottles_per_minute_six_machines : ℕ := 240)
variable (number_of_machines : ℕ := 6)
variable (new_number_of_machines : ℕ := 10)
variable (time_in_minutes : ℕ := 4)

theorem bottles_produced_by_10_machines_in_4_minutes :
  rate_per_machine = total_bottles_per_minute_six_machines / number_of_machines →
  (new_number_of_machines * rate_per_machine * time_in_minutes) = 1600 := 
sorry

end NUMINAMATH_GPT_bottles_produced_by_10_machines_in_4_minutes_l1659_165919


namespace NUMINAMATH_GPT_derivative_of_cos_over_x_l1659_165977

open Real

noncomputable def f (x : ℝ) : ℝ := (cos x) / x

theorem derivative_of_cos_over_x (x : ℝ) (h : x ≠ 0) : 
  deriv f x = - (x * sin x + cos x) / (x^2) :=
sorry

end NUMINAMATH_GPT_derivative_of_cos_over_x_l1659_165977


namespace NUMINAMATH_GPT_female_students_count_l1659_165986

variable (F : ℕ)

theorem female_students_count
    (avg_all_students : ℕ)
    (avg_male_students : ℕ)
    (avg_female_students : ℕ)
    (num_male_students : ℕ)
    (condition1 : avg_all_students = 90)
    (condition2 : avg_male_students = 82)
    (condition3 : avg_female_students = 92)
    (condition4 : num_male_students = 8)
    (condition5 : 8 * 82 + F * 92 = (8 + F) * 90) : 
    F = 32 := 
by 
  sorry

end NUMINAMATH_GPT_female_students_count_l1659_165986


namespace NUMINAMATH_GPT_find_a_minus_b_l1659_165902

theorem find_a_minus_b (a b : ℚ) (h_eq : ∀ x : ℚ, (a * (-5 * x + 3) + b) = x - 9) : 
  a - b = 41 / 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_a_minus_b_l1659_165902


namespace NUMINAMATH_GPT_day50_yearM_minus1_is_Friday_l1659_165937

-- Define weekdays
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Weekday

-- Define days of the week for specific days in given years
def day_of (d : Nat) (reference_day : Weekday) (reference_day_mod : Nat) : Weekday :=
  match (reference_day_mod + d - 1) % 7 with
  | 0 => Sunday
  | 1 => Monday
  | 2 => Tuesday
  | 3 => Wednesday
  | 4 => Thursday
  | 5 => Friday
  | 6 => Saturday
  | _ => Thursday -- This case should never occur due to mod 7

def day250_yearM : Weekday := Thursday
def day150_yearM1 : Weekday := Thursday

-- Theorem to prove
theorem day50_yearM_minus1_is_Friday :
    day_of 50 day250_yearM 6 = Friday :=
sorry

end NUMINAMATH_GPT_day50_yearM_minus1_is_Friday_l1659_165937


namespace NUMINAMATH_GPT_combined_weight_l1659_165950

variable (J S : ℝ)

-- Given conditions
def jake_current_weight := (J = 152)
def lose_weight_equation := (J - 32 = 2 * S)

-- Question: combined weight of Jake and his sister
theorem combined_weight (h1 : jake_current_weight J) (h2 : lose_weight_equation J S) : J + S = 212 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_l1659_165950


namespace NUMINAMATH_GPT_card_combinations_l1659_165920

noncomputable def valid_card_combinations : List (ℕ × ℕ × ℕ × ℕ) :=
  [(1, 2, 7, 8), (1, 3, 6, 8), (1, 4, 5, 8), (2, 3, 6, 7), (2, 4, 5, 7), (3, 4, 5, 6)]

theorem card_combinations (a b c d : ℕ) (h : a ≤ b ∧ b ≤ c ∧ c ≤ d) :
  (1, 2, 7, 8) ∈ valid_card_combinations ∨ 
  (1, 3, 6, 8) ∈ valid_card_combinations ∨ 
  (1, 4, 5, 8) ∈ valid_card_combinations ∨ 
  (2, 3, 6, 7) ∈ valid_card_combinations ∨ 
  (2, 4, 5, 7) ∈ valid_card_combinations ∨ 
  (3, 4, 5, 6) ∈ valid_card_combinations :=
sorry

end NUMINAMATH_GPT_card_combinations_l1659_165920


namespace NUMINAMATH_GPT_trays_from_first_table_l1659_165975

-- Definitions based on conditions
def trays_per_trip : ℕ := 4
def trips : ℕ := 3
def trays_from_second_table : ℕ := 2

-- Theorem statement to prove the number of trays picked up from the first table
theorem trays_from_first_table : trays_per_trip * trips - trays_from_second_table = 10 := by
  sorry

end NUMINAMATH_GPT_trays_from_first_table_l1659_165975
