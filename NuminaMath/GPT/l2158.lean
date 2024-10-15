import Mathlib

namespace NUMINAMATH_GPT_fraction_zero_when_x_eq_3_l2158_215894

theorem fraction_zero_when_x_eq_3 : ∀ x : ℝ, x = 3 → (x^6 - 54 * x^3 + 729) / (x^3 - 27) = 0 :=
by
  intro x hx
  rw [hx]
  sorry

end NUMINAMATH_GPT_fraction_zero_when_x_eq_3_l2158_215894


namespace NUMINAMATH_GPT_find_pos_integers_A_B_l2158_215811

noncomputable def concat (A B : ℕ) : ℕ :=
  let b := Nat.log 10 B + 1
  A * 10 ^ b + B

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def satisfiesConditions (A B : ℕ) : Prop :=
  isPerfectSquare (concat A B) ∧ concat A B = 2 * A * B

theorem find_pos_integers_A_B :
  ∃ (A B : ℕ), A = (5 ^ b + 1) / 2 ∧ B = 2 ^ b * A * 100 ^ m ∧ b % 2 = 1 ∧ ∀ m : ℕ, satisfiesConditions A B :=
sorry

end NUMINAMATH_GPT_find_pos_integers_A_B_l2158_215811


namespace NUMINAMATH_GPT_initial_stops_eq_l2158_215837

-- Define the total number of stops S
def total_stops : ℕ := 7

-- Define the number of stops made after the initial deliveries
def additional_stops : ℕ := 4

-- Define the number of initial stops as a proof problem
theorem initial_stops_eq : total_stops - additional_stops = 3 :=
by
sorry

end NUMINAMATH_GPT_initial_stops_eq_l2158_215837


namespace NUMINAMATH_GPT_quadratic_parabola_equation_l2158_215831

theorem quadratic_parabola_equation :
  ∃ (a b c : ℝ), 
    (∀ x y, y = 3 * x^2 - 6 * x + 5 → (x - 1)*(x - 1) = (x - 1)^2) ∧ -- Original vertex condition and standard form
    (∀ x y, y = -x - 2 → a = 2) ∧ -- Given intersection point condition
    (∀ x y, y = -3 * (x - 1)^2 + 2 → y = -3 * (x - 1)^2 + b ∧ y = -4) → -- Vertex unchanged and direction reversed
    (a, b, c) = (-3, 6, -4) := -- Resulting equation coefficients
sorry

end NUMINAMATH_GPT_quadratic_parabola_equation_l2158_215831


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2158_215887

-- Define what it means for α to be of the form (π/6 + 2kπ) where k ∈ ℤ
def is_pi_six_plus_two_k_pi (α : ℝ) : Prop :=
  ∃ k : ℤ, α = Real.pi / 6 + 2 * k * Real.pi

-- Define the condition sin α = 1 / 2
def sin_is_half (α : ℝ) : Prop :=
  Real.sin α = 1 / 2

-- The theorem stating that the given condition is a sufficient but not necessary condition
theorem sufficient_but_not_necessary (α : ℝ) :
  is_pi_six_plus_two_k_pi α → sin_is_half α ∧ ¬ (sin_is_half α → is_pi_six_plus_two_k_pi α) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2158_215887


namespace NUMINAMATH_GPT_length_of_field_l2158_215827

-- Define the conditions and given facts.
def double_length (w l : ℝ) : Prop := l = 2 * w
def pond_area (l w : ℝ) : Prop := 49 = 1/8 * (l * w)

-- Define the main statement that incorporates the given conditions and expected result.
theorem length_of_field (w l : ℝ) (h1 : double_length w l) (h2 : pond_area l w) : l = 28 := by
  sorry

end NUMINAMATH_GPT_length_of_field_l2158_215827


namespace NUMINAMATH_GPT_kayla_less_than_vika_l2158_215802

variable (S K V : ℕ)
variable (h1 : S = 216)
variable (h2 : S = 4 * K)
variable (h3 : V = 84)

theorem kayla_less_than_vika (S K V : ℕ) (h1 : S = 216) (h2 : S = 4 * K) (h3 : V = 84) : V - K = 30 :=
by
  sorry

end NUMINAMATH_GPT_kayla_less_than_vika_l2158_215802


namespace NUMINAMATH_GPT_opposite_of_lime_is_black_l2158_215874

-- Given colors of the six faces
inductive Color
| Purple | Cyan | Magenta | Silver | Lime | Black

-- Hinged squares forming a cube
structure Cube :=
(top : Color) (bottom : Color) (front : Color) (back : Color) (left : Color) (right : Color)

-- Condition: Magenta is on the top
def magenta_top (c : Cube) : Prop := c.top = Color.Magenta

-- Problem statement: Prove the color opposite to Lime is Black
theorem opposite_of_lime_is_black (c : Cube) (HM : magenta_top c) (HL : c.front = Color.Lime)
    (HBackFace : c.back = Color.Black) : c.back = Color.Black := 
sorry

end NUMINAMATH_GPT_opposite_of_lime_is_black_l2158_215874


namespace NUMINAMATH_GPT_non_congruent_triangles_with_perimeter_11_l2158_215846

theorem non_congruent_triangles_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 → a < b + c → b < a + c → c < a + b → 
  ∃! (a b c : ℕ), (a, b, c) = (2, 4, 5) ∨ (a, b, c) = (3, 4, 4) := 
by sorry

end NUMINAMATH_GPT_non_congruent_triangles_with_perimeter_11_l2158_215846


namespace NUMINAMATH_GPT_probability_not_finishing_on_time_l2158_215862

-- Definitions based on the conditions
def P_finishing_on_time : ℚ := 5 / 8

-- Theorem to prove the required probability
theorem probability_not_finishing_on_time :
  (1 - P_finishing_on_time) = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_probability_not_finishing_on_time_l2158_215862


namespace NUMINAMATH_GPT_find_r_in_geometric_sum_l2158_215834

theorem find_r_in_geometric_sum (S_n : ℕ → ℕ) (r : ℤ)
  (hSn : ∀ n : ℕ, S_n n = 2 * 3^n + r)
  (hgeo : ∀ n : ℕ, n ≥ 2 → S_n n - S_n (n - 1) = 4 * 3^(n - 1))
  (hn1 : S_n 1 = 6 + r) :
  r = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_r_in_geometric_sum_l2158_215834


namespace NUMINAMATH_GPT_measure_of_angle_x_l2158_215858

-- Defining the conditions
def angle_ABC : ℝ := 108
def angle_ABD : ℝ := 180 - angle_ABC
def angle_in_triangle_ABD_1 : ℝ := 26
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- The theorem to prove
theorem measure_of_angle_x (h1 : angle_ABD = 72)
                           (h2 : angle_in_triangle_ABD_1 = 26)
                           (h3 : sum_of_angles_in_triangle angle_ABD angle_in_triangle_ABD_1 x) :
  x = 82 :=
by {
  -- Since this is a formal statement, we leave the proof as an exercise 
  sorry
}

end NUMINAMATH_GPT_measure_of_angle_x_l2158_215858


namespace NUMINAMATH_GPT_adults_in_each_group_l2158_215882

theorem adults_in_each_group (A : ℕ) :
  (∃ n : ℕ, n >= 17 ∧ n * 15 = 255) →
  (∃ m : ℕ, m * A = 255 ∧ m >= 17) →
  A = 15 :=
by
  intros h_child_groups h_adult_groups
  -- Use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_adults_in_each_group_l2158_215882


namespace NUMINAMATH_GPT_function_bounds_l2158_215848

theorem function_bounds {a : ℝ} :
  (∀ x : ℝ, x > 0 → 4 - x^2 + a * Real.log x ≤ 3) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_function_bounds_l2158_215848


namespace NUMINAMATH_GPT_max_value_ratio_l2158_215845

/-- Define the conditions on function f and variables x and y. -/
def conditions (f : ℝ → ℝ) (x y : ℝ) :=
  (∀ x, f (-x) + f x = 0) ∧
  (∀ x1 x2, x1 < x2 → f x1 < f x2) ∧
  f (x^2 - 6 * x) + f (y^2 - 4 * y + 12) ≤ 0

/-- The maximum value of (y - 2) / x under the given conditions. -/
theorem max_value_ratio (f : ℝ → ℝ) (x y : ℝ) (cond : conditions f x y) :
  (y - 2) / x ≤ (Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_GPT_max_value_ratio_l2158_215845


namespace NUMINAMATH_GPT_b_is_nth_power_l2158_215812

theorem b_is_nth_power (b n : ℕ) (h1 : b > 1) (h2 : n > 1) 
    (h3 : ∀ k > 1, ∃ a_k : ℕ, k ∣ (b - a_k^n)) : 
    ∃ A : ℕ, b = A^n :=
sorry

end NUMINAMATH_GPT_b_is_nth_power_l2158_215812


namespace NUMINAMATH_GPT_like_terms_exponents_l2158_215838

theorem like_terms_exponents (m n : ℤ) 
  (h1 : m - 1 = 1) 
  (h2 : m + n = 3) : 
  m = 2 ∧ n = 1 :=
by 
  sorry

end NUMINAMATH_GPT_like_terms_exponents_l2158_215838


namespace NUMINAMATH_GPT_no_real_solutions_l2158_215809

theorem no_real_solutions (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≠ 0) :
    (a = 0) ∨ (a ≠ 0 ∧ 4 * a * b - 3 * a ^ 2 > 0) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_l2158_215809


namespace NUMINAMATH_GPT_Cinderella_solves_l2158_215835

/--
There are three bags labeled as "Poppy", "Millet", and "Mixture". Each label is incorrect.
By inspecting one grain from the bag labeled as "Mixture", Cinderella can determine the exact contents of all three bags.
-/
theorem Cinderella_solves (bag_contents : String → String) (examined_grain : String) :
  (bag_contents "Mixture" = "Poppy" ∨ bag_contents "Mixture" = "Millet") →
  (∀ l, bag_contents l ≠ l) →
  (examined_grain = "Poppy" ∨ examined_grain = "Millet") →
  examined_grain = bag_contents "Mixture" →
  ∃ poppy_bag millet_bag mixture_bag : String,
    poppy_bag ≠ "Poppy" ∧ millet_bag ≠ "Millet" ∧ mixture_bag ≠ "Mixture" ∧
    bag_contents poppy_bag = "Poppy" ∧
    bag_contents millet_bag = "Millet" ∧
    bag_contents mixture_bag = "Mixture" :=
sorry

end NUMINAMATH_GPT_Cinderella_solves_l2158_215835


namespace NUMINAMATH_GPT_sequence_geometric_l2158_215821

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  3 * a n - 2

theorem sequence_geometric (a : ℕ → ℝ) (h : ∀ n, S n a = 3 * a n - 2) :
  ∀ n, a n = (3/2)^(n-1) :=
by
  intro n
  sorry

end NUMINAMATH_GPT_sequence_geometric_l2158_215821


namespace NUMINAMATH_GPT_value_of_x_l2158_215817

theorem value_of_x
  (x : ℝ)
  (h1 : x = 0)
  (h2 : x^2 - 1 ≠ 0) :
  (x = 0) ↔ (x ^ 2 - 1 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l2158_215817


namespace NUMINAMATH_GPT_complex_number_in_fourth_quadrant_l2158_215886

variable {a b : ℝ}

theorem complex_number_in_fourth_quadrant (a b : ℝ): 
  (a^2 + 1 > 0) ∧ (-b^2 - 1 < 0) → 
  ((a^2 + 1, -b^2 - 1).fst > 0 ∧ (a^2 + 1, -b^2 - 1).snd < 0) :=
by
  intro h
  exact h

#check complex_number_in_fourth_quadrant

end NUMINAMATH_GPT_complex_number_in_fourth_quadrant_l2158_215886


namespace NUMINAMATH_GPT_income_difference_l2158_215879

theorem income_difference
  (D W : ℝ)
  (hD : 0.08 * D = 800)
  (hW : 0.08 * W = 840) :
  (W + 840) - (D + 800) = 540 := 
  sorry

end NUMINAMATH_GPT_income_difference_l2158_215879


namespace NUMINAMATH_GPT_money_given_by_school_correct_l2158_215803

-- Definitions from the problem conditions
def cost_per_book : ℕ := 12
def number_of_students : ℕ := 30
def out_of_pocket : ℕ := 40

-- Derived definition from these conditions
def total_cost : ℕ := cost_per_book * number_of_students
def money_given_by_school : ℕ := total_cost - out_of_pocket

-- The theorem stating that the amount given by the school is $320
theorem money_given_by_school_correct : money_given_by_school = 320 :=
by
  sorry -- Proof placeholder

end NUMINAMATH_GPT_money_given_by_school_correct_l2158_215803


namespace NUMINAMATH_GPT_price_of_each_cupcake_l2158_215892

variable (x : ℝ)

theorem price_of_each_cupcake (h : 50 * x + 40 * 0.5 = 2 * 40 + 20 * 2) : x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_price_of_each_cupcake_l2158_215892


namespace NUMINAMATH_GPT_right_pyramid_sum_edges_l2158_215849

theorem right_pyramid_sum_edges (a h : ℝ) (base_side slant_height : ℝ) :
  base_side = 12 ∧ slant_height = 15 ∧ ∀ x : ℝ, a = 117 :=
by
  sorry

end NUMINAMATH_GPT_right_pyramid_sum_edges_l2158_215849


namespace NUMINAMATH_GPT_polynomial_constant_l2158_215851

theorem polynomial_constant
  (P : Polynomial ℤ)
  (h : ∀ Q F G : Polynomial ℤ, P.comp Q = F * G → F.degree = 0 ∨ G.degree = 0) :
  P.degree = 0 :=
by sorry

end NUMINAMATH_GPT_polynomial_constant_l2158_215851


namespace NUMINAMATH_GPT_factorization_result_l2158_215806

theorem factorization_result :
  ∃ (c d : ℕ), (c > d) ∧ ((x^2 - 20 * x + 91) = (x - c) * (x - d)) ∧ (2 * d - c = 1) :=
by
  -- Using the conditions and proving the given equation
  sorry

end NUMINAMATH_GPT_factorization_result_l2158_215806


namespace NUMINAMATH_GPT_find_real_medal_min_weighings_l2158_215857

axiom has_9_medals : Prop
axiom one_real_medal : Prop
axiom real_medal_heavier : Prop
axiom has_balance_scale : Prop

theorem find_real_medal_min_weighings
  (h1 : has_9_medals)
  (h2 : one_real_medal)
  (h3 : real_medal_heavier)
  (h4 : has_balance_scale) :
  ∃ (minimum_weighings : ℕ), minimum_weighings = 2 := 
  sorry

end NUMINAMATH_GPT_find_real_medal_min_weighings_l2158_215857


namespace NUMINAMATH_GPT_find_Natisfy_condition_l2158_215807

-- Define the original number
def N : Nat := 2173913043478260869565

-- Define the function to move the first digit of a number to the end
def move_first_digit_to_end (n : Nat) : Nat := sorry

-- The proof statement
theorem find_Natisfy_condition : 
  let new_num1 := N * 4
  let new_num2 := new_num1 / 5
  move_first_digit_to_end N = new_num2 
:=
  sorry

end NUMINAMATH_GPT_find_Natisfy_condition_l2158_215807


namespace NUMINAMATH_GPT_yunas_math_score_l2158_215808

theorem yunas_math_score (K E M : ℕ) 
  (h1 : (K + E) / 2 = 92) 
  (h2 : (K + E + M) / 3 = 94) : 
  M = 98 :=
sorry

end NUMINAMATH_GPT_yunas_math_score_l2158_215808


namespace NUMINAMATH_GPT_division_result_l2158_215897

theorem division_result (k q : ℕ) (h₁ : k % 81 = 11) (h₂ : 81 > 0) : k / 81 = q + 11 / 81 :=
  sorry

end NUMINAMATH_GPT_division_result_l2158_215897


namespace NUMINAMATH_GPT_molecular_weight_6_moles_C4H8O2_is_528_624_l2158_215865

-- Define the atomic weights of Carbon, Hydrogen, and Oxygen.
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula of C4H8O2.
def num_C_atoms : ℕ := 4
def num_H_atoms : ℕ := 8
def num_O_atoms : ℕ := 2

-- Define the number of moles of C4H8O2.
def num_moles_C4H8O2 : ℝ := 6

-- Define the molecular weight of one mole of C4H8O2.
def molecular_weight_C4H8O2 : ℝ :=
  (num_C_atoms * atomic_weight_C) +
  (num_H_atoms * atomic_weight_H) +
  (num_O_atoms * atomic_weight_O)

-- The total weight of 6 moles of C4H8O2.
def total_weight_6_moles_C4H8O2 : ℝ :=
  num_moles_C4H8O2 * molecular_weight_C4H8O2

-- Theorem stating that the molecular weight of 6 moles of C4H8O2 is 528.624 grams.
theorem molecular_weight_6_moles_C4H8O2_is_528_624 :
  total_weight_6_moles_C4H8O2 = 528.624 :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_molecular_weight_6_moles_C4H8O2_is_528_624_l2158_215865


namespace NUMINAMATH_GPT_original_price_lamp_l2158_215876

theorem original_price_lamp
  (P : ℝ)
  (discount_rate : ℝ)
  (discounted_price : ℝ)
  (discount_is_20_perc : discount_rate = 0.20)
  (new_price_is_96 : discounted_price = 96)
  (price_after_discount : discounted_price = P * (1 - discount_rate)) :
  P = 120 :=
by
  sorry

end NUMINAMATH_GPT_original_price_lamp_l2158_215876


namespace NUMINAMATH_GPT_painted_surface_area_is_33_l2158_215881

/-- 
Problem conditions:
    1. We have 14 unit cubes each with side length 1 meter.
    2. The cubes are arranged in a rectangular formation with dimensions 3x3x1.
The question:
    Prove that the total painted surface area is 33 square meters.
-/
def total_painted_surface_area (cubes : ℕ) (dim_x dim_y dim_z : ℕ) : ℕ :=
  let top_area := dim_x * dim_y
  let side_area := 2 * (dim_x * dim_z + dim_y * dim_z + (dim_z - 1) * dim_x)
  top_area + side_area

theorem painted_surface_area_is_33 :
  total_painted_surface_area 14 3 3 1 = 33 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_painted_surface_area_is_33_l2158_215881


namespace NUMINAMATH_GPT_remaining_leaves_l2158_215805

def initial_leaves := 1000
def first_week_shed := (2 / 5 : ℚ) * initial_leaves
def leaves_after_first_week := initial_leaves - first_week_shed
def second_week_shed := (40 / 100 : ℚ) * leaves_after_first_week
def leaves_after_second_week := leaves_after_first_week - second_week_shed
def third_week_shed := (3 / 4 : ℚ) * second_week_shed
def leaves_after_third_week := leaves_after_second_week - third_week_shed

theorem remaining_leaves (initial_leaves first_week_shed leaves_after_first_week second_week_shed leaves_after_second_week third_week_shed leaves_after_third_week: ℚ) : 
  leaves_after_third_week = 180 := by
  sorry

end NUMINAMATH_GPT_remaining_leaves_l2158_215805


namespace NUMINAMATH_GPT_Dima_impossible_cut_l2158_215869

theorem Dima_impossible_cut (n : ℕ) 
  (h1 : n % 5 = 0) 
  (h2 : n % 7 = 0) 
  (h3 : n ≤ 200) : ¬(n % 6 = 0) :=
sorry

end NUMINAMATH_GPT_Dima_impossible_cut_l2158_215869


namespace NUMINAMATH_GPT_line_passes_point_a_ne_zero_l2158_215861

theorem line_passes_point_a_ne_zero (a : ℝ) (h1 : ∀ (x y : ℝ), (y = 5 * x + a) → (x = a ∧ y = a^2)) (h2 : a ≠ 0) : a = 6 :=
sorry

end NUMINAMATH_GPT_line_passes_point_a_ne_zero_l2158_215861


namespace NUMINAMATH_GPT_verify_graphical_method_l2158_215841

variable {R : Type} [LinearOrderedField R]

/-- Statement of the mentioned conditions -/
def poly (a b c d x : R) : R := a * x^3 + b * x^2 + c * x + d

/-- The main theorem stating the graphical method validity -/
theorem verify_graphical_method (a b c d x0 EJ : R) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) (h3 : 0 < d) (h4 : 0 < x0) (h5 : x0 < 1)
: EJ = poly a b c d x0 := by sorry

end NUMINAMATH_GPT_verify_graphical_method_l2158_215841


namespace NUMINAMATH_GPT_domain_of_fx_l2158_215864

theorem domain_of_fx :
  {x : ℝ | x ≥ 1 ∧ x^2 < 2} = {x : ℝ | 1 ≤ x ∧ x < Real.sqrt 2} := by
sorry

end NUMINAMATH_GPT_domain_of_fx_l2158_215864


namespace NUMINAMATH_GPT_find_f_10_l2158_215896

def f (x : Int) : Int := sorry

axiom condition_1 : f 1 + 1 > 0
axiom condition_2 : ∀ x y : Int, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y
axiom condition_3 : ∀ x : Int, 2 * f x = f (x + 1) - x + 1

theorem find_f_10 : f 10 = 1014 := by
  sorry

end NUMINAMATH_GPT_find_f_10_l2158_215896


namespace NUMINAMATH_GPT_jakes_weight_l2158_215860

theorem jakes_weight
  (J K : ℝ)
  (h1 : J - 8 = 2 * K)
  (h2 : J + K = 290) :
  J = 196 :=
by
  sorry

end NUMINAMATH_GPT_jakes_weight_l2158_215860


namespace NUMINAMATH_GPT_minimum_distance_midpoint_l2158_215899

theorem minimum_distance_midpoint 
    (θ : ℝ)
    (P : ℝ × ℝ := (-4, 4))
    (C1_standard : ∀ (x y : ℝ), (x + 4)^2 + (y - 3)^2 = 1)
    (C2_standard : ∀ (x y : ℝ), x^2 / 64 + y^2 / 9 = 1)
    (Q : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ))
    (M : ℝ × ℝ := (-2 + 4 * Real.cos θ, 2 + 3 / 2 * Real.sin θ))
    (C3_standard : ∀ (x y : ℝ), x - 2*y - 7 = 0) :
    ∃ (θ : ℝ), θ = Real.arcsin (-3/5) ∧ (θ = Real.arccos 4/5) ∧
    (∀ (d : ℝ), d = abs (5 * Real.sin (Real.arctan (4 / 3) - θ) - 13) / Real.sqrt 5 ∧ 
    d = 8 * Real.sqrt 5 / 5) :=
sorry

end NUMINAMATH_GPT_minimum_distance_midpoint_l2158_215899


namespace NUMINAMATH_GPT_total_loaves_served_l2158_215833

-- Definitions based on the conditions provided
def wheat_bread_loaf : ℝ := 0.2
def white_bread_loaf : ℝ := 0.4

-- Statement that needs to be proven
theorem total_loaves_served : wheat_bread_loaf + white_bread_loaf = 0.6 := 
by
  sorry

end NUMINAMATH_GPT_total_loaves_served_l2158_215833


namespace NUMINAMATH_GPT_joe_money_left_l2158_215891

theorem joe_money_left
  (initial_money : ℕ) (notebook_cost : ℕ) (notebooks : ℕ)
  (book_cost : ℕ) (books : ℕ) (pen_cost : ℕ) (pens : ℕ)
  (sticker_pack_cost : ℕ) (sticker_packs : ℕ) (charity : ℕ)
  (remaining_money : ℕ) :
  initial_money = 150 →
  notebook_cost = 4 →
  notebooks = 7 →
  book_cost = 12 →
  books = 2 →
  pen_cost = 2 →
  pens = 5 →
  sticker_pack_cost = 6 →
  sticker_packs = 3 →
  charity = 10 →
  remaining_money = 60 →
  remaining_money = 
    initial_money - 
    ((notebooks * notebook_cost) + 
     (books * book_cost) + 
     (pens * pen_cost) + 
     (sticker_packs * sticker_pack_cost) + 
     charity) := 
by
  intros; sorry

end NUMINAMATH_GPT_joe_money_left_l2158_215891


namespace NUMINAMATH_GPT_food_left_after_bbqs_l2158_215822

noncomputable def mushrooms_bought : ℕ := 15
noncomputable def chicken_bought : ℕ := 20
noncomputable def beef_bought : ℕ := 10

noncomputable def mushrooms_consumed : ℕ := 5 * 3
noncomputable def chicken_consumed : ℕ := 4 * 2
noncomputable def beef_consumed : ℕ := 2 * 1

noncomputable def mushrooms_left : ℕ := mushrooms_bought - mushrooms_consumed
noncomputable def chicken_left : ℕ := chicken_bought - chicken_consumed
noncomputable def beef_left : ℕ := beef_bought - beef_consumed

noncomputable def total_food_left : ℕ := mushrooms_left + chicken_left + beef_left

theorem food_left_after_bbqs : total_food_left = 20 :=
  by
    unfold total_food_left mushrooms_left chicken_left beef_left
    unfold mushrooms_consumed chicken_consumed beef_consumed
    unfold mushrooms_bought chicken_bought beef_bought
    sorry

end NUMINAMATH_GPT_food_left_after_bbqs_l2158_215822


namespace NUMINAMATH_GPT_part_I_part_II_l2158_215815

-- Definition of the sequence a_n with given conditions
def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1 else (n^2 + n) / 2

-- Define the sum of the first n terms S_n
def S_n (n : ℕ) : ℕ :=
  (n + 2) / 3 * a_n n

-- Define the sequence b_n in terms of a_n
def b_n (n : ℕ) : ℚ := 1 / a_n n

-- Define the sum of the first n terms of b_n
def T_n (n : ℕ) : ℚ :=
  2 * (1 - 1 / (n + 1))

-- Theorem statement for part (I)
theorem part_I (n : ℕ) : 
  a_n 2 = 3 ∧ a_n 3 = 6 ∧ (∀ (n : ℕ), n ≥ 2 → a_n n = (n^2 + n) / 2) := sorry

-- Theorem statement for part (II)
theorem part_II (n : ℕ) : 
  T_n n = 2 * (1 - 1 / (n + 1)) := sorry

end NUMINAMATH_GPT_part_I_part_II_l2158_215815


namespace NUMINAMATH_GPT_distinct_roots_condition_l2158_215810

noncomputable def f (x c : ℝ) : ℝ := x^2 + 6*x + c

theorem distinct_roots_condition (c : ℝ) :
  (∀x : ℝ, f (f x c) = 0 → ∃ a b : ℝ, (a ≠ b) ∧ f x c = a * (x - b) * (x - c) ) →
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end NUMINAMATH_GPT_distinct_roots_condition_l2158_215810


namespace NUMINAMATH_GPT_solution_l2158_215877

theorem solution (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 12) : (12 * y - 4)^2 = 128 :=
sorry

end NUMINAMATH_GPT_solution_l2158_215877


namespace NUMINAMATH_GPT_speed_of_boat_in_still_water_l2158_215825

theorem speed_of_boat_in_still_water
    (speed_stream : ℝ)
    (distance_downstream : ℝ)
    (distance_upstream : ℝ)
    (t : ℝ)
    (x : ℝ)
    (h1 : speed_stream = 10)
    (h2 : distance_downstream = 80)
    (h3 : distance_upstream = 40)
    (h4 : t = distance_downstream / (x + speed_stream))
    (h5 : t = distance_upstream / (x - speed_stream)) :
  x = 30 :=
by sorry

end NUMINAMATH_GPT_speed_of_boat_in_still_water_l2158_215825


namespace NUMINAMATH_GPT_abs_neg_six_l2158_215856

theorem abs_neg_six : abs (-6) = 6 := by
  sorry

end NUMINAMATH_GPT_abs_neg_six_l2158_215856


namespace NUMINAMATH_GPT_coin_flip_difference_l2158_215873

/-- The positive difference between the probability of a fair coin landing heads up
exactly 4 times out of 5 flips and the probability of a fair coin landing heads up
5 times out of 5 flips is 1/8. -/
theorem coin_flip_difference :
  (5 * (1 / 2) ^ 5) - ((1 / 2) ^ 5) = (1 / 8) :=
by
  sorry

end NUMINAMATH_GPT_coin_flip_difference_l2158_215873


namespace NUMINAMATH_GPT_cuboid_edge_lengths_l2158_215898

theorem cuboid_edge_lengths (
  a b c : ℕ
) (h_volume : a * b * c + a * b + b * c + c * a + a + b + c = 2000) :
  (a = 28 ∧ b = 22 ∧ c = 2) ∨ 
  (a = 28 ∧ b = 2 ∧ c = 22) ∨
  (a = 22 ∧ b = 28 ∧ c = 2) ∨
  (a = 22 ∧ b = 2 ∧ c = 28) ∨
  (a = 2 ∧ b = 28 ∧ c = 22) ∨
  (a = 2 ∧ b = 22 ∧ c = 28) :=
sorry

end NUMINAMATH_GPT_cuboid_edge_lengths_l2158_215898


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2158_215859

-- Definition of the given conditions and the theorem to prove
theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : a < 0)
  (h2 : ∀ x, ax^2 + bx + c < 0 ↔ x < -2 ∨ x > -1/2) :
  ∀ x, ax^2 - bx + c > 0 ↔ 1/2 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2158_215859


namespace NUMINAMATH_GPT_line_of_intersection_l2158_215844

theorem line_of_intersection :
  ∀ (x y z : ℝ),
    (3 * x + 4 * y - 2 * z + 1 = 0) ∧ (2 * x - 4 * y + 3 * z + 4 = 0) →
    (∃ t : ℝ, x = -1 + 4 * t ∧ y = 1 / 2 - 13 * t ∧ z = -20 * t) :=
by
  intro x y z
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_line_of_intersection_l2158_215844


namespace NUMINAMATH_GPT_parabola_equation_l2158_215889

open Real

theorem parabola_equation (vertex focus : ℝ × ℝ) (h_vertex : vertex = (0, 0)) (h_focus : focus = (0, 3)) :
  ∃ a : ℝ, x^2 = 12 * y := by
  sorry

end NUMINAMATH_GPT_parabola_equation_l2158_215889


namespace NUMINAMATH_GPT_general_admission_tickets_l2158_215854

variable (x y : ℕ)

theorem general_admission_tickets (h1 : x + y = 525) (h2 : 4 * x + 6 * y = 2876) : y = 388 := by
  sorry

end NUMINAMATH_GPT_general_admission_tickets_l2158_215854


namespace NUMINAMATH_GPT_tangent_condition_l2158_215820

theorem tangent_condition (a b : ℝ) : 
    a = b → 
    (∀ x y : ℝ, (y = x + 2 → (x - a)^2 + (y - b)^2 = 2 → y = x + 2)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_condition_l2158_215820


namespace NUMINAMATH_GPT_imaginary_unit_power_l2158_215836

-- Definition of the imaginary unit i
def imaginary_unit_i : ℂ := Complex.I

theorem imaginary_unit_power :
  (imaginary_unit_i ^ 2015) = -imaginary_unit_i := by
  sorry

end NUMINAMATH_GPT_imaginary_unit_power_l2158_215836


namespace NUMINAMATH_GPT_quadratic_roots_x_no_real_solution_y_l2158_215880

theorem quadratic_roots_x (x : ℝ) : 
  x^2 - 4*x + 3 = 0 ↔ (x = 3 ∨ x = 1) := sorry

theorem no_real_solution_y (y : ℝ) : 
  ¬∃ y : ℝ, 4*y^2 - 3*y + 2 = 0 := sorry

end NUMINAMATH_GPT_quadratic_roots_x_no_real_solution_y_l2158_215880


namespace NUMINAMATH_GPT_pies_count_l2158_215867

-- Definitions based on the conditions given in the problem
def strawberries_per_pie := 3
def christine_strawberries := 10
def rachel_strawberries := 2 * christine_strawberries

-- The theorem to prove
theorem pies_count : (christine_strawberries + rachel_strawberries) / strawberries_per_pie = 10 := by
  sorry

end NUMINAMATH_GPT_pies_count_l2158_215867


namespace NUMINAMATH_GPT_friends_count_l2158_215885

-- Define the given conditions
def initial_chicken_wings := 2
def additional_chicken_wings := 25
def chicken_wings_per_person := 3

-- Define the total number of chicken wings
def total_chicken_wings := initial_chicken_wings + additional_chicken_wings

-- Define the target number of friends in the group
def number_of_friends := total_chicken_wings / chicken_wings_per_person

-- The theorem stating that the number of friends is 9
theorem friends_count : number_of_friends = 9 := by
  sorry

end NUMINAMATH_GPT_friends_count_l2158_215885


namespace NUMINAMATH_GPT_same_days_to_dig_scenario_l2158_215870

def volume (depth length breadth : ℝ) : ℝ :=
  depth * length * breadth

def days_to_dig (depth length breadth days : ℝ) : Prop :=
  ∃ (labors : ℝ), 
    (volume depth length breadth) * days = (volume 100 25 30) * 12

theorem same_days_to_dig_scenario :
  days_to_dig 75 20 50 12 :=
sorry

end NUMINAMATH_GPT_same_days_to_dig_scenario_l2158_215870


namespace NUMINAMATH_GPT_geometric_sum_S_40_l2158_215884

variable (S : ℕ → ℝ)

-- Conditions
axiom sum_S_10 : S 10 = 18
axiom sum_S_20 : S 20 = 24

-- Proof statement
theorem geometric_sum_S_40 : S 40 = 80 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sum_S_40_l2158_215884


namespace NUMINAMATH_GPT_otimes_property_l2158_215855

def otimes (a b : ℚ) : ℚ := (a^3) / b

theorem otimes_property : otimes (otimes 2 3) 4 - otimes 2 (otimes 3 4) = 80 / 27 := by
  sorry

end NUMINAMATH_GPT_otimes_property_l2158_215855


namespace NUMINAMATH_GPT_geo_seq_bn_plus_2_general_formula_an_l2158_215801

variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}

-- Conditions
axiom h1 : a 1 = 2
axiom h2 : a 2 = 4
axiom h3 : ∀ n, b n = a (n + 1) - a n
axiom h4 : ∀ n, b (n + 1) = 2 * b n + 2

-- Proof goals
theorem geo_seq_bn_plus_2 : (∀ n, ∃ r : ℕ, b n + 2 = 4 * 2^n) :=
  sorry

theorem general_formula_an : (∀ n, a n = 2^(n + 1) - 2 * n) :=
  sorry

end NUMINAMATH_GPT_geo_seq_bn_plus_2_general_formula_an_l2158_215801


namespace NUMINAMATH_GPT_least_number_subtracted_l2158_215871

theorem least_number_subtracted {
  x : ℕ
} : 
  (∀ (m : ℕ), m ∈ [5, 9, 11] → (997 - x) % m = 3) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_least_number_subtracted_l2158_215871


namespace NUMINAMATH_GPT_first_term_geometric_sequence_b_n_bounded_l2158_215818

-- Definition: S_n = 3a_n - 5n for any n in ℕ*
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 3 * a n - 5 * n

-- The sequence a_n is given such that
-- Proving the first term a_1
theorem first_term (a : ℕ → ℝ) (h : ∀ n, S (n + 1) a = S n a + a n + 1 - 5) : 
  a 1 = 5 / 2 :=
sorry

-- Prove that {a_n + 5} is a geometric sequence with common ratio 3/2
theorem geometric_sequence (a : ℕ → ℝ) (h : ∀ n, S n a = 3 * a n - 5 * n) : 
  ∃ r, (∀ n, a (n + 1) + 5 = r * (a n + 5)) ∧ r = 3 / 2 :=
sorry

-- Prove that there exists m such that b_n < m always holds for b_n = (9n + 4) / (a_n + 5)
theorem b_n_bounded (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h1 : ∀ n, b n = (9 * ↑n + 4) / (a n + 5)) 
  (h2 : ∀ n, a n = (15 / 2) * (3 / 2)^(n-1) - 5) :
  ∃ m, ∀ n, b n < m ∧ m = 88 / 45 :=
sorry

end NUMINAMATH_GPT_first_term_geometric_sequence_b_n_bounded_l2158_215818


namespace NUMINAMATH_GPT_remainder_of_6_pow_50_mod_215_l2158_215828

theorem remainder_of_6_pow_50_mod_215 :
  (6 ^ 50) % 215 = 36 := 
sorry

end NUMINAMATH_GPT_remainder_of_6_pow_50_mod_215_l2158_215828


namespace NUMINAMATH_GPT_zero_points_C_exist_l2158_215847

theorem zero_points_C_exist (A B C : ℝ × ℝ) (hAB_dist : dist A B = 12) (h_perimeter : dist A B + dist A C + dist B C = 52)
    (h_area : abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = 100) : 
    false :=
by
  sorry

end NUMINAMATH_GPT_zero_points_C_exist_l2158_215847


namespace NUMINAMATH_GPT_xyz_value_l2158_215824

noncomputable def find_xyz (x y z : ℝ) 
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h₃ : (x + y + z)^2 = 25) : ℝ :=
  if (x * y * z = 31 / 3) then 31 / 3 else 0  -- This should hold with the given conditions

theorem xyz_value (x y z : ℝ)
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h₃ : (x + y + z)^2 = 25) :
  find_xyz x y z h₁ h₂ h₃ = 31 / 3 :=
by 
  sorry  -- The proof should demonstrate that find_xyz equals 31 / 3 given the conditions

end NUMINAMATH_GPT_xyz_value_l2158_215824


namespace NUMINAMATH_GPT_cindy_arrival_speed_l2158_215890

def cindy_speed (d t1 t2 t3: ℕ) : Prop :=
  (d = 20 * t1) ∧ 
  (d = 10 * (t2 + 3 / 4)) ∧
  (t3 = t1 + 1 / 2) ∧
  (20 * t1 = 10 * (t2 + 3 / 4)) -> 
  (d / (t3) = 12)

theorem cindy_arrival_speed (t1 t2: ℕ) (h₁: t2 = t1 + 3 / 4) (d: ℕ) (h2: d = 20 * t1) (h3: t3 = t1 + 1 / 2) :
  cindy_speed d t1 t2 t3 := by
  sorry

end NUMINAMATH_GPT_cindy_arrival_speed_l2158_215890


namespace NUMINAMATH_GPT_quad_completion_l2158_215830

theorem quad_completion (a b c : ℤ) 
    (h : ∀ x : ℤ, 8 * x^2 - 48 * x - 128 = a * (x + b)^2 + c) : 
    a + b + c = -195 := 
by
  sorry

end NUMINAMATH_GPT_quad_completion_l2158_215830


namespace NUMINAMATH_GPT_division_and_multiplication_result_l2158_215819

theorem division_and_multiplication_result :
  let num : ℝ := 6.5
  let divisor : ℝ := 6
  let multiplier : ℝ := 12
  num / divisor * multiplier = 13 :=
by
  sorry

end NUMINAMATH_GPT_division_and_multiplication_result_l2158_215819


namespace NUMINAMATH_GPT_pyramid_volume_l2158_215878

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * (EF * FG) * QE

theorem pyramid_volume
  (EF FG QE : ℝ)
  (h1 : EF = 10)
  (h2 : FG = 5)
  (h3 : QE = 9) :
  volume_of_pyramid EF FG QE = 150 :=
by
  simp [volume_of_pyramid, h1, h2, h3]
  sorry

end NUMINAMATH_GPT_pyramid_volume_l2158_215878


namespace NUMINAMATH_GPT_term_15_of_sequence_l2158_215863

theorem term_15_of_sequence : 
  ∃ (a : ℕ → ℝ), a 1 = 3 ∧ a 2 = 7 ∧ (∀ n, a (n + 1) = 21 / a n) ∧ a 15 = 3 :=
sorry

end NUMINAMATH_GPT_term_15_of_sequence_l2158_215863


namespace NUMINAMATH_GPT_triangle_is_equilateral_l2158_215893

theorem triangle_is_equilateral (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + ac + bc) : a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_equilateral_l2158_215893


namespace NUMINAMATH_GPT_smallest_positive_integer_satisfying_condition_l2158_215842

-- Define the condition
def isConditionSatisfied (n : ℕ) : Prop :=
  (Real.sqrt n - Real.sqrt (n - 1) < 0.01) ∧ n > 0

-- State the theorem
theorem smallest_positive_integer_satisfying_condition :
  ∃ n : ℕ, isConditionSatisfied n ∧ (∀ m : ℕ, isConditionSatisfied m → n ≤ m) ∧ n = 2501 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_satisfying_condition_l2158_215842


namespace NUMINAMATH_GPT_line_intersects_y_axis_at_5_l2158_215800

theorem line_intersects_y_axis_at_5 :
  ∃ (b : ℝ), ∀ (x y : ℝ), (x - 2 = 0 ∧ y - 9 = 0) ∨ (x - 4 = 0 ∧ y - 13 = 0) →
  (y = 2 * x + b) ∧ (b = 5) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_y_axis_at_5_l2158_215800


namespace NUMINAMATH_GPT_storks_equal_other_birds_l2158_215829

-- Definitions of initial numbers of birds
def initial_sparrows := 2
def initial_crows := 1
def initial_storks := 3
def initial_egrets := 0

-- Birds arriving initially
def sparrows_arrived := 1
def crows_arrived := 3
def storks_arrived := 6
def egrets_arrived := 4

-- Birds leaving after 15 minutes
def sparrows_left := 2
def crows_left := 0
def storks_left := 0
def egrets_left := 1

-- Additional birds arriving after 30 minutes
def additional_sparrows := 0
def additional_crows := 4
def additional_storks := 3
def additional_egrets := 0

-- Final counts
def final_sparrows := initial_sparrows + sparrows_arrived - sparrows_left + additional_sparrows
def final_crows := initial_crows + crows_arrived - crows_left + additional_crows
def final_storks := initial_storks + storks_arrived - storks_left + additional_storks
def final_egrets := initial_egrets + egrets_arrived - egrets_left + additional_egrets

def total_other_birds := final_sparrows + final_crows + final_egrets

-- Theorem statement
theorem storks_equal_other_birds : final_storks - total_other_birds = 0 := by
  sorry

end NUMINAMATH_GPT_storks_equal_other_birds_l2158_215829


namespace NUMINAMATH_GPT_amanda_car_round_trip_time_l2158_215814

theorem amanda_car_round_trip_time :
  (bus_time = 40) ∧ (car_time = bus_time - 5) → (round_trip_time = car_time * 2) → round_trip_time = 70 :=
by
  sorry

end NUMINAMATH_GPT_amanda_car_round_trip_time_l2158_215814


namespace NUMINAMATH_GPT_largest_number_systematic_sampling_l2158_215813

theorem largest_number_systematic_sampling (n k a1 a2: ℕ) (h1: n = 60) (h2: a1 = 3) (h3: a2 = 9) (h4: k = a2 - a1):
  ∃ largest, largest = a1 + k * (n / k - 1) := by
  sorry

end NUMINAMATH_GPT_largest_number_systematic_sampling_l2158_215813


namespace NUMINAMATH_GPT_a_plus_b_l2158_215866

theorem a_plus_b (a b : ℝ) (h : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 :=
sorry

end NUMINAMATH_GPT_a_plus_b_l2158_215866


namespace NUMINAMATH_GPT_Asya_Petya_l2158_215868

theorem Asya_Petya (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) 
  (h : 1000 * a + b = 7 * a * b) : a = 143 ∧ b = 143 :=
by
  sorry

end NUMINAMATH_GPT_Asya_Petya_l2158_215868


namespace NUMINAMATH_GPT_mirror_area_l2158_215832

theorem mirror_area (frame_length frame_width frame_border_length : ℕ) (mirror_area : ℕ)
  (h_frame_length : frame_length = 100)
  (h_frame_width : frame_width = 130)
  (h_frame_border_length : frame_border_length = 15)
  (h_mirror_area : mirror_area = (frame_length - 2 * frame_border_length) * (frame_width - 2 * frame_border_length)) :
  mirror_area = 7000 := by 
    sorry

end NUMINAMATH_GPT_mirror_area_l2158_215832


namespace NUMINAMATH_GPT_verify_first_rope_length_l2158_215850

def length_first_rope : ℝ :=
  let rope1_len := 20
  let rope2_len := 2
  let rope3_len := 2
  let rope4_len := 2
  let rope5_len := 7
  let knots := 4
  let knot_loss := 1.2
  let total_len := 35
  rope1_len

theorem verify_first_rope_length : length_first_rope = 20 := by
  sorry

end NUMINAMATH_GPT_verify_first_rope_length_l2158_215850


namespace NUMINAMATH_GPT_quadratic_has_real_root_l2158_215853

theorem quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := sorry

end NUMINAMATH_GPT_quadratic_has_real_root_l2158_215853


namespace NUMINAMATH_GPT_probability_three_digit_multiple_5_remainder_3_div_7_l2158_215804

theorem probability_three_digit_multiple_5_remainder_3_div_7 :
  (∃ (P : ℝ), P = (26 / 900)) := 
by sorry

end NUMINAMATH_GPT_probability_three_digit_multiple_5_remainder_3_div_7_l2158_215804


namespace NUMINAMATH_GPT_perfect_squares_less_than_500_ending_in_4_l2158_215875

theorem perfect_squares_less_than_500_ending_in_4 : 
  (∃ (squares : Finset ℕ), (∀ n ∈ squares, n < 500 ∧ (n % 10 = 4)) ∧ squares.card = 5) :=
by
  sorry

end NUMINAMATH_GPT_perfect_squares_less_than_500_ending_in_4_l2158_215875


namespace NUMINAMATH_GPT_bacteria_colony_growth_l2158_215843

theorem bacteria_colony_growth (n : ℕ) : 
  (∀ m: ℕ, 4 * 3^m ≤ 500 → m < n) → n = 5 :=
by
  sorry

end NUMINAMATH_GPT_bacteria_colony_growth_l2158_215843


namespace NUMINAMATH_GPT_sum_of_d_and_e_l2158_215872

-- Define the original numbers and their sum
def original_first := 3742586
def original_second := 4829430
def correct_sum := 8572016

-- The given incorrect addition result
def given_sum := 72120116

-- Define the digits d and e
def d := 2
def e := 8

-- Define the correct adjusted sum if we replace d with e
def adjusted_first := 3782586
def adjusted_second := 4889430
def adjusted_sum := 8672016

-- State the final theorem
theorem sum_of_d_and_e : 
  (given_sum != correct_sum) → 
  (original_first + original_second = correct_sum) → 
  (adjusted_first + adjusted_second = adjusted_sum) → 
  (d + e = 10) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_d_and_e_l2158_215872


namespace NUMINAMATH_GPT_quadratic_real_roots_l2158_215895

theorem quadratic_real_roots (k : ℝ) (h : ∀ x : ℝ, k * x^2 - 4 * x + 1 = 0) : k ≤ 4 ∧ k ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l2158_215895


namespace NUMINAMATH_GPT_double_root_equation_correct_statements_l2158_215816

theorem double_root_equation_correct_statements
  (a b c : ℝ) (r₁ r₂ : ℝ)
  (h1 : a ≠ 0)
  (h2 : r₁ = 2 * r₂)
  (h3 : r₁ ≠ r₂)
  (h4 : a * r₁ ^ 2 + b * r₁ + c = 0)
  (h5 : a * r₂ ^ 2 + b * r₂ + c = 0) :
  (∀ (m n : ℝ), (∀ (r : ℝ), r = 2 → (x - r) * (m * x + n) = 0 → 4 * m ^ 2 + 5 * m * n + n ^ 2 = 0)) ∧
  (∀ (p q : ℝ), p * q = 2 → ∃ x, p * x ^ 2 + 3 * x + q = 0 ∧
    (∃ x₁ x₂ : ℝ, x₁ = -1 / p ∧ x₂ = -q ∧ x₁ = 2 * x₂)) ∧
  (2 * b ^ 2 = 9 * a * c) :=
by
  sorry

end NUMINAMATH_GPT_double_root_equation_correct_statements_l2158_215816


namespace NUMINAMATH_GPT_measurable_length_l2158_215823

-- Definitions of lines, rays, and line segments

-- A line is infinitely long with no endpoints.
def isLine (l : Type) : Prop := ∀ x y : l, (x ≠ y)

-- A line segment has two endpoints and a finite length.
def isLineSegment (ls : Type) : Prop := ∃ a b : ls, a ≠ b ∧ ∃ d : ℝ, d > 0

-- A ray has one endpoint and is infinitely long.
def isRay (r : Type) : Prop := ∃ e : r, ∀ x : r, x ≠ e

-- Problem statement
theorem measurable_length (x : Type) : isLineSegment x → (∃ d : ℝ, d > 0) :=
by
  -- Proof is not required
  sorry

end NUMINAMATH_GPT_measurable_length_l2158_215823


namespace NUMINAMATH_GPT_directrix_of_parabola_l2158_215852

theorem directrix_of_parabola (x y : ℝ) : 
  (x^2 = - (1/8) * y) → (y = 1/32) :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l2158_215852


namespace NUMINAMATH_GPT_abs_eq_solution_l2158_215839

theorem abs_eq_solution (x : ℝ) (h : abs (x - 3) = abs (x + 2)) : x = 1 / 2 :=
sorry

end NUMINAMATH_GPT_abs_eq_solution_l2158_215839


namespace NUMINAMATH_GPT_barbara_total_candies_l2158_215826

-- Condition: Barbara originally has 9 candies.
def C1 := 9

-- Condition: Barbara buys 18 more candies.
def C2 := 18

-- Question (proof problem): Prove that the total number of candies Barbara has is 27.
theorem barbara_total_candies : C1 + C2 = 27 := by
  -- Proof steps are not required, hence using sorry.
  sorry

end NUMINAMATH_GPT_barbara_total_candies_l2158_215826


namespace NUMINAMATH_GPT_difference_between_waiter_and_twenty_less_l2158_215888

-- Definitions for the given conditions
def total_slices : ℕ := 78
def ratio_buzz : ℕ := 5
def ratio_waiter : ℕ := 8
def total_ratio : ℕ := ratio_buzz + ratio_waiter
def slices_per_part : ℕ := total_slices / total_ratio
def buzz_share : ℕ := ratio_buzz * slices_per_part
def waiter_share : ℕ := ratio_waiter * slices_per_part
def twenty_less_waiter : ℕ := waiter_share - 20

-- The proof statement
theorem difference_between_waiter_and_twenty_less : 
  waiter_share - twenty_less_waiter = 20 :=
by sorry

end NUMINAMATH_GPT_difference_between_waiter_and_twenty_less_l2158_215888


namespace NUMINAMATH_GPT_seq_arithmetic_l2158_215840

theorem seq_arithmetic (a : ℕ → ℕ) (h : ∀ p q : ℕ, a p + a q = a (p + q)) (h1 : a 1 = 2) :
  ∀ n : ℕ, a n = 2 * n :=
by
  sorry

end NUMINAMATH_GPT_seq_arithmetic_l2158_215840


namespace NUMINAMATH_GPT_seating_arrangement_l2158_215883

def num_ways_seated (total_passengers : ℕ) (window_seats : ℕ) : ℕ :=
  window_seats * (total_passengers - 1) * (total_passengers - 2) * (total_passengers - 3)

theorem seating_arrangement (passengers_seats taxi_window_seats : ℕ)
  (h1 : passengers_seats = 4) (h2 : taxi_window_seats = 2) :
  num_ways_seated passengers_seats taxi_window_seats = 12 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_seating_arrangement_l2158_215883
