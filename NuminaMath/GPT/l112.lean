import Mathlib

namespace shift_parabola_5_units_right_l112_11249

def original_parabola (x : ℝ) : ℝ := x^2 + 3
def shifted_parabola (x : ℝ) : ℝ := (x-5)^2 + 3

theorem shift_parabola_5_units_right : ∀ x : ℝ, shifted_parabola x = original_parabola (x - 5) :=
by {
  -- This is the mathematical equivalence that we're proving
  sorry
}

end shift_parabola_5_units_right_l112_11249


namespace katherine_bottle_caps_l112_11218

-- Define the initial number of bottle caps Katherine has
def initial_bottle_caps : ℕ := 34

-- Define the number of bottle caps eaten by the hippopotamus
def eaten_bottle_caps : ℕ := 8

-- Define the remaining number of bottle caps Katherine should have
def remaining_bottle_caps : ℕ := initial_bottle_caps - eaten_bottle_caps

-- Theorem stating that Katherine will have 26 bottle caps after the hippopotamus eats 8 of them
theorem katherine_bottle_caps : remaining_bottle_caps = 26 := by
  sorry

end katherine_bottle_caps_l112_11218


namespace thousands_digit_is_0_or_5_l112_11231

theorem thousands_digit_is_0_or_5 (n t : ℕ) (h₁ : n > 1000000) (h₂ : n % 40 = t) (h₃ : n % 625 = t) : 
  ((n / 1000) % 10 = 0) ∨ ((n / 1000) % 10 = 5) :=
sorry

end thousands_digit_is_0_or_5_l112_11231


namespace barbara_initial_candies_l112_11248

noncomputable def initialCandies (used left: ℝ) := used + left

theorem barbara_initial_candies (used left: ℝ) (h_used: used = 9.0) (h_left: left = 9) : initialCandies used left = 18 := 
by
  rw [h_used, h_left]
  norm_num
  sorry

end barbara_initial_candies_l112_11248


namespace minimum_area_integer_triangle_l112_11223

theorem minimum_area_integer_triangle :
  ∃ (p q : ℤ), p ≠ 0 ∧ q ≠ 0 ∧ (∃ (p q : ℤ), 2 ∣ (16 * p - 30 * q)) 
  → (∃ (area : ℝ), area = (1/2 : ℝ) * |16 * p - 30 * q| ∧ area = 1) :=
by
  sorry

end minimum_area_integer_triangle_l112_11223


namespace initial_action_figures_l112_11228

theorem initial_action_figures (x : ℕ) (h : x + 4 - 1 = 6) : x = 3 :=
by {
  sorry
}

end initial_action_figures_l112_11228


namespace solve_for_x_l112_11214

theorem solve_for_x (x : ℝ) (h : 3 * x - 5 * x + 6 * x = 150) : x = 37.5 :=
by
  sorry

end solve_for_x_l112_11214


namespace min_value_f_l112_11271

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + 4 * x + 20) + Real.sqrt (x^2 + 2 * x + 10)

theorem min_value_f : ∃ x : ℝ, f x = 5 * Real.sqrt 2 :=
by
  sorry

end min_value_f_l112_11271


namespace total_grains_in_grey_parts_l112_11251

theorem total_grains_in_grey_parts 
  (total_grains_each_circle : ℕ)
  (white_grains_first_circle : ℕ)
  (white_grains_second_circle : ℕ)
  (common_white_grains : ℕ) 
  (h1 : white_grains_first_circle = 87)
  (h2 : white_grains_second_circle = 110)
  (h3 : common_white_grains = 68) :
  (white_grains_first_circle - common_white_grains) +
  (white_grains_second_circle - common_white_grains) = 61 :=
by
  sorry

end total_grains_in_grey_parts_l112_11251


namespace problem_l112_11256

def f (x : ℤ) : ℤ := 3 * x - 1
def g (x : ℤ) : ℤ := 2 * x + 5

theorem problem (h : ℤ) :
  (g (f (g (3))) : ℚ) / f (g (f (3))) = 69 / 206 :=
by
  sorry

end problem_l112_11256


namespace original_population_multiple_of_3_l112_11284

theorem original_population_multiple_of_3 (x y z : ℕ) (h1 : x^2 + 121 = y^2) (h2 : y^2 + 121 = z^2) :
  3 ∣ x^2 :=
sorry

end original_population_multiple_of_3_l112_11284


namespace intersection_is_correct_l112_11230

def M : Set ℤ := {x | x^2 + 3 * x + 2 > 0}
def N : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_is_correct : M ∩ N = {0, 1, 2} := by
  sorry

end intersection_is_correct_l112_11230


namespace avg_of_multiples_of_4_is_even_l112_11244

theorem avg_of_multiples_of_4_is_even (m n : ℤ) (hm : m % 4 = 0) (hn : n % 4 = 0) :
  (m + n) / 2 % 2 = 0 := sorry

end avg_of_multiples_of_4_is_even_l112_11244


namespace extremum_of_f_unique_solution_of_equation_l112_11277

noncomputable def f (x m : ℝ) : ℝ := (1/2) * x^2 - m * Real.log x

theorem extremum_of_f (m : ℝ) (h_pos : 0 < m) :
  ∃ x_min : ℝ, x_min = Real.sqrt m ∧
  ∀ x : ℝ, 0 < x → f x m ≥ f (Real.sqrt m) m :=
sorry

theorem unique_solution_of_equation (m : ℝ) (h_ge_one : 1 ≤ m) :
  ∃! x : ℝ, 0 < x ∧ f x m = x^2 - (m + 1) * x :=
sorry

#check extremum_of_f -- Ensure it can be checked
#check unique_solution_of_equation -- Ensure it can be checked

end extremum_of_f_unique_solution_of_equation_l112_11277


namespace comparison_of_logs_l112_11219

noncomputable def a : ℝ := Real.logb 4 6
noncomputable def b : ℝ := Real.logb 4 0.2
noncomputable def c : ℝ := Real.logb 2 3

theorem comparison_of_logs : c > a ∧ a > b := by
  sorry

end comparison_of_logs_l112_11219


namespace hibiscus_flower_ratio_l112_11205

theorem hibiscus_flower_ratio (x : ℕ) 
  (h1 : 2 + x + 4 * x = 22) : x / 2 = 2 := 
sorry

end hibiscus_flower_ratio_l112_11205


namespace distinct_license_plates_count_l112_11227

def num_digit_choices : Nat := 10
def num_letter_choices : Nat := 26
def num_digits : Nat := 5
def num_letters : Nat := 3

theorem distinct_license_plates_count :
  (num_digit_choices ^ num_digits) * (num_letter_choices ^ num_letters) = 1757600000 := 
sorry

end distinct_license_plates_count_l112_11227


namespace f_1_eq_0_range_x_l112_11298

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_Rstar : ∀ x : ℝ, ¬ (x = 0) → f x = sorry
axiom f_4_eq_1 : f 4 = 1
axiom f_mult : ∀ (x₁ x₂ : ℝ), ¬ (x₁ = 0) → ¬ (x₂ = 0) → f (x₁ * x₂) = f x₁ + f x₂
axiom f_increasing : ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f x₁ < f x₂

theorem f_1_eq_0 : f 1 = 0 := sorry

theorem range_x (x : ℝ) : f (3 * x + 1) + f (2 * x - 6) ≤ 3 → 3 < x ∧ x ≤ 5 := sorry

end f_1_eq_0_range_x_l112_11298


namespace islands_not_connected_by_bridges_for_infinitely_many_primes_l112_11273

open Nat

theorem islands_not_connected_by_bridges_for_infinitely_many_primes :
  ∃ᶠ p in at_top, ∃ n m : ℕ, n ≠ m ∧ ¬(p ∣ (n^2 - m + 1) * (m^2 - n + 1)) :=
sorry

end islands_not_connected_by_bridges_for_infinitely_many_primes_l112_11273


namespace minimum_value_2x_4y_l112_11292

theorem minimum_value_2x_4y (x y : ℝ) (h : x + 2 * y = 3) : 
  ∃ (min_val : ℝ), min_val = 2 ^ (5/2) ∧ (2 ^ x + 4 ^ y = min_val) :=
by
  sorry

end minimum_value_2x_4y_l112_11292


namespace gumballs_per_box_l112_11293

-- Given conditions
def total_gumballs : ℕ := 20
def total_boxes : ℕ := 4

-- Mathematically equivalent proof problem
theorem gumballs_per_box:
  total_gumballs / total_boxes = 5 := by
  sorry

end gumballs_per_box_l112_11293


namespace height_of_table_l112_11222

/-- 
Given:
1. Combined initial measurement (l + h - w + t) = 40
2. Combined changed measurement (w + h - l + t) = 34
3. Width of each wood block (w) = 6 inches
4. Visible edge-on thickness of the table (t) = 4 inches
Prove:
The height of the table (h) is 33 inches.
-/
theorem height_of_table (l h t w : ℕ) (h_combined_initial : l + h - w + t = 40)
    (h_combined_changed : w + h - l + t = 34) (h_width : w = 6) (h_thickness : t = 4) : 
    h = 33 :=
by
  sorry

end height_of_table_l112_11222


namespace parallelogram_area_l112_11258

theorem parallelogram_area (base height : ℝ) (h_base : base = 22) (h_height : height = 14) :
  base * height = 308 := by
  sorry

end parallelogram_area_l112_11258


namespace polygon_sides_sum_l112_11289

theorem polygon_sides_sum (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
by
  sorry

end polygon_sides_sum_l112_11289


namespace custom_op_example_l112_11272

-- Definition of the custom operation
def custom_op (a b : ℕ) : ℕ := 4 * a + 5 * b - a * b

-- The proof statement
theorem custom_op_example : custom_op 7 3 = 22 := by
  sorry

end custom_op_example_l112_11272


namespace no_int_coords_equilateral_l112_11274

--- Define a structure for points with integer coordinates
structure Point :=
(x : ℤ)
(y : ℤ)

--- Definition of the distance squared between two points
def dist_squared (P Q : Point) : ℤ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

--- Statement that given three points with integer coordinates, they cannot form an equilateral triangle
theorem no_int_coords_equilateral (A B C : Point) :
  ¬ (dist_squared A B = dist_squared B C ∧ dist_squared B C = dist_squared C A ∧ dist_squared C A = dist_squared A B) :=
sorry

end no_int_coords_equilateral_l112_11274


namespace min_value_f_l112_11267

def f (x y : ℝ) : ℝ := x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y

theorem min_value_f : ∃ x y : ℝ, f x y = -9 / 5 :=
sorry

end min_value_f_l112_11267


namespace smallest_k_for_mutual_criticism_l112_11275

-- Define a predicate that checks if a given configuration of criticisms lead to mutual criticism
def mutual_criticism_exists (deputies : ℕ) (k : ℕ) : Prop :=
  k ≥ 8 -- This is derived from the problem where k = 8 is the smallest k ensuring a mutual criticism

theorem smallest_k_for_mutual_criticism:
  mutual_criticism_exists 15 8 :=
by
  -- This is the theorem statement with the conditions and correct answer. The proof is omitted.
  sorry

end smallest_k_for_mutual_criticism_l112_11275


namespace integer_roots_l112_11246

-- Define the polynomial
def poly (x : ℤ) : ℤ := x^3 - 4 * x^2 - 11 * x + 24

-- State the theorem
theorem integer_roots : {x : ℤ | poly x = 0} = {-1, 2, 3} := 
  sorry

end integer_roots_l112_11246


namespace min_C_over_D_l112_11241

theorem min_C_over_D (x C D : ℝ) (h1 : x^2 + 1/x^2 = C) (h2 : x + 1/x = D) (hC_pos : 0 < C) (hD_pos : 0 < D) : 
  (∃ m : ℝ, m = 2 * Real.sqrt 2 ∧ ∀ y : ℝ, y = C / D → y ≥ m) :=
  sorry

end min_C_over_D_l112_11241


namespace problem_statement_l112_11235

variable (a b c p q r α β γ : ℝ)

-- Given conditions
def plane_condition : Prop := (a / α) + (b / β) + (c / γ) = 1
def sphere_conditions : Prop := p^3 = α ∧ q^3 = β ∧ r^3 = γ

-- The statement to prove
theorem problem_statement (h_plane : plane_condition a b c α β γ) (h_sphere : sphere_conditions p q r α β γ) :
  (a / p^3) + (b / q^3) + (c / r^3) = 1 := sorry

end problem_statement_l112_11235


namespace time_to_print_800_flyers_l112_11225

theorem time_to_print_800_flyers (x : ℝ) (h1 : 0 < x) :
  (1 / 6) + (1 / x) = 1 / 1.5 ↔ ∀ y : ℝ, 800 / 6 + 800 / x = 800 / 1.5 :=
by sorry

end time_to_print_800_flyers_l112_11225


namespace value_at_minus_two_l112_11201

def f (x : ℝ) (a b c : ℝ) := a * x^5 + b * x^3 + c * x + 1

theorem value_at_minus_two (a b c : ℝ) (h : f 2 a b c = -1) : f (-2) a b c = 3 := by
  sorry

end value_at_minus_two_l112_11201


namespace determine_percentage_of_second_mixture_l112_11266

-- Define the given conditions and question
def mixture_problem (P : ℝ) : Prop :=
  ∃ (V1 V2 : ℝ) (A1 A2 A_final : ℝ),
  V1 = 2.5 ∧ A1 = 0.30 ∧
  V2 = 7.5 ∧ A2 = P / 100 ∧
  A_final = 0.45 ∧
  (V1 * A1 + V2 * A2) / (V1 + V2) = A_final

-- State the theorem
theorem determine_percentage_of_second_mixture : mixture_problem 50 := sorry

end determine_percentage_of_second_mixture_l112_11266


namespace hyperbola_eccentricity_l112_11242

-- Define the conditions of the hyperbola and points
variables {a b c m d : ℝ} (ha : a > 0) (hb : b > 0) 
noncomputable def F1 : ℝ := sorry -- Placeholder for focus F1
noncomputable def F2 : ℝ := sorry -- Placeholder for focus F2
noncomputable def P : ℝ := sorry  -- Placeholder for point P

-- Define the sides of the triangle in terms of an arithmetic progression
def PF2 (m d : ℝ) : ℝ := m - d
def PF1 (m : ℝ) : ℝ := m
def F1F2 (m d : ℝ) : ℝ := m + d

-- Prove that the eccentricity is 5 given the conditions
theorem hyperbola_eccentricity 
  (m d : ℝ) (hc : c = (5 / 2) * d )  
  (h1 : PF1 m = 2 * a)
  (h2 : F1F2 m d = 2 * c)
  (h3 : (PF2 m d)^2 + (PF1 m)^2 = (F1F2 m d)^2 ) :
  (c / a) = 5 := 
sorry

end hyperbola_eccentricity_l112_11242


namespace total_money_9pennies_4nickels_3dimes_l112_11262

def value_of_pennies (num_pennies : ℕ) : ℝ := num_pennies * 0.01
def value_of_nickels (num_nickels : ℕ) : ℝ := num_nickels * 0.05
def value_of_dimes (num_dimes : ℕ) : ℝ := num_dimes * 0.10

def total_value (pennies nickels dimes : ℕ) : ℝ :=
  value_of_pennies pennies + value_of_nickels nickels + value_of_dimes dimes

theorem total_money_9pennies_4nickels_3dimes :
  total_value 9 4 3 = 0.59 :=
by 
  sorry

end total_money_9pennies_4nickels_3dimes_l112_11262


namespace james_pays_37_50_l112_11221

/-- 
James gets 20 singing lessons.
First lesson is free.
After the first 10 paid lessons, he only needs to pay for every other lesson.
Each lesson costs $5.
His uncle pays for half.
Prove that James pays $37.50.
--/

theorem james_pays_37_50 :
  let first_lessons := 1
  let total_lessons := 20
  let paid_lessons := 10
  let remaining_lessons := total_lessons - first_lessons - paid_lessons
  let paid_remaining_lessons := (remaining_lessons + 1) / 2
  let total_paid_lessons := paid_lessons + paid_remaining_lessons
  let cost_per_lesson := 5
  let total_payment := total_paid_lessons * cost_per_lesson
  let payment_by_james := total_payment / 2
  payment_by_james = 37.5 := 
by
  sorry

end james_pays_37_50_l112_11221


namespace parabola_equation_l112_11206

theorem parabola_equation (M : ℝ × ℝ) (hM : M = (5, 3))
    (h_dist : ∀ a : ℝ, |5 + 1/(4*a)| = 6) :
    (y = (1/12)*x^2) ∨ (y = -(1/36)*x^2) :=
sorry

end parabola_equation_l112_11206


namespace intersection_complement_l112_11295

open Set

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Theorem
theorem intersection_complement :
  A ∩ (U \ B) = {1} :=
sorry

end intersection_complement_l112_11295


namespace sums_equal_l112_11229

theorem sums_equal (A B C : Type) (a b c : ℕ) :
  (a + b + c) = (a + (b + c)) ∧
  (a + b + c) = (b + (c + a)) ∧
  (a + b + c) = (c + (a + b)) :=
by 
  sorry

end sums_equal_l112_11229


namespace hyperbola_slope_condition_l112_11263

-- Define the setup
variables (a b : ℝ) (P F1 F2 : ℝ × ℝ)
variables (h : a > 0) (k : b > 0)
variables (hyperbola : (∀ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1)))

-- Define the condition
variables (cond : ∃ (P : ℝ × ℝ), 3 * abs (dist P F1 + dist P F2) ≤ 2 * dist F1 F2)

-- The proof goal
theorem hyperbola_slope_condition : (b / a) ≥ (Real.sqrt 5 / 2) :=
sorry

end hyperbola_slope_condition_l112_11263


namespace y_coord_of_equidistant_point_on_y_axis_l112_11250

/-!
  # Goal
  Prove that the $y$-coordinate of the point P on the $y$-axis that is equidistant from points $A(5, 0)$ and $B(3, 6)$ is \( \frac{5}{3} \).
  Conditions:
  - Point A has coordinates (5, 0).
  - Point B has coordinates (3, 6).
-/

theorem y_coord_of_equidistant_point_on_y_axis :
  ∃ y : ℝ, y = 5 / 3 ∧ (dist (⟨0, y⟩ : ℝ × ℝ) (⟨5, 0⟩ : ℝ × ℝ) = dist (⟨0, y⟩ : ℝ × ℝ) (⟨3, 6⟩ : ℝ × ℝ)) :=
by
  sorry -- Proof omitted

end y_coord_of_equidistant_point_on_y_axis_l112_11250


namespace calculate_r_l112_11260

def a := 0.24 * 450
def b := 0.62 * 250
def c := 0.37 * 720
def d := 0.38 * 100
def sum_bc := b + c
def diff := sum_bc - a
def r := diff / d

theorem calculate_r : r = 8.25 := by
  sorry

end calculate_r_l112_11260


namespace find_integers_l112_11264

theorem find_integers (A B C : ℤ) (hA : A = 500) (hB : B = -1) (hC : C = -500) : 
  (A : ℚ) / 999 + (B : ℚ) / 1000 + (C : ℚ) / 1001 = 1 / (999 * 1000 * 1001) :=
by 
  rw [hA, hB, hC]
  sorry

end find_integers_l112_11264


namespace saree_blue_stripes_l112_11253

theorem saree_blue_stripes :
  ∀ (brown_stripes gold_stripes blue_stripes : ℕ),
    gold_stripes = 3 * brown_stripes →
    blue_stripes = 5 * gold_stripes →
    brown_stripes = 4 →
    blue_stripes = 60 :=
by
  intros brown_stripes gold_stripes blue_stripes h_gold h_blue h_brown
  sorry

end saree_blue_stripes_l112_11253


namespace roots_squared_sum_l112_11208

theorem roots_squared_sum :
  (∀ x, x^2 + 2 * x - 8 = 0 → (x = x1 ∨ x = x2)) →
  x1 + x2 = -2 ∧ x1 * x2 = -8 →
  x1^2 + x2^2 = 20 :=
by
  intros roots_eq_sum_prod_eq
  sorry

end roots_squared_sum_l112_11208


namespace cds_probability_l112_11232

def probability (total favorable : ℕ) : ℚ := favorable / total

theorem cds_probability :
  probability 120 24 = 1 / 5 :=
by
  sorry

end cds_probability_l112_11232


namespace cookies_milk_conversion_l112_11290

theorem cookies_milk_conversion :
  (18 : ℕ) / (3 * 2 : ℕ) / (18 : ℕ) * (9 : ℕ) = (3 : ℕ) :=
by
  sorry

end cookies_milk_conversion_l112_11290


namespace find_second_number_l112_11285

-- Definitions for the conditions
def ratio_condition (x : ℕ) : Prop := 5 * x = 40

-- The theorem we need to prove, i.e., the second number is 8 given the conditions
theorem find_second_number (x : ℕ) (h : ratio_condition x) : x = 8 :=
by sorry

end find_second_number_l112_11285


namespace lee_can_make_36_cookies_l112_11280

-- Conditions
def initial_cups_of_flour : ℕ := 2
def initial_cookies_made : ℕ := 18
def initial_total_flour : ℕ := 5
def spilled_flour : ℕ := 1

-- Define the remaining cups of flour after spilling
def remaining_flour := initial_total_flour - spilled_flour

-- Define the proportion to solve for the number of cookies made with remaining_flour
def cookies_with_remaining_flour (c : ℕ) : Prop :=
  (initial_cookies_made / initial_cups_of_flour) = (c / remaining_flour)

-- The statement to prove
theorem lee_can_make_36_cookies : cookies_with_remaining_flour 36 :=
  sorry

end lee_can_make_36_cookies_l112_11280


namespace james_vegetable_consumption_l112_11238

theorem james_vegetable_consumption : 
  (1/4 + 1/4) * 2 * 7 + 3 = 10 := 
by
  sorry

end james_vegetable_consumption_l112_11238


namespace minimum_sum_of_nine_consecutive_integers_l112_11224

-- We will define the consecutive sequence and the conditions as described.
structure ConsecutiveIntegers (a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ) : Prop :=
(seq : a1 + 1 = a2 ∧ a2 + 1 = a3 ∧ a3 + 1 = a4 ∧ a4 + 1 = a5 ∧ a5 + 1 = a6 ∧ a6 + 1 = a7 ∧ a7 + 1 = a8 ∧ a8 + 1 = a9)
(sq_cond : ∃ k : ℕ, (a1 + a3 + a5 + a7 + a9) = k * k)
(cube_cond : ∃ l : ℕ, (a2 + a4 + a6 + a8) = l * l * l)

theorem minimum_sum_of_nine_consecutive_integers :
  ∃ a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ,
  ConsecutiveIntegers a1 a2 a3 a4 a5 a6 a7 a8 a9 ∧ (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 = 18000) :=
  sorry

end minimum_sum_of_nine_consecutive_integers_l112_11224


namespace arithmetic_mean_l112_11240

variables (x y z : ℝ)

def condition1 : Prop := 1 / (x * y) = y / (z - x + 1)
def condition2 : Prop := 1 / (x * y) = 2 / (z + 1)

theorem arithmetic_mean (h1 : condition1 x y z) (h2 : condition2 x y z) : x = (z + y) / 2 :=
by
  sorry

end arithmetic_mean_l112_11240


namespace fraction_of_satisfactory_grades_is_3_4_l112_11252

def num_grades (grades : String → ℕ) : ℕ := 
  grades "A" + grades "B" + grades "C" + grades "D" + grades "F"

def satisfactory_grades (grades : String → ℕ) : ℕ := 
  grades "A" + grades "B" + grades "C" + grades "D"

def fraction_satisfactory (grades : String → ℕ) : ℚ := 
  satisfactory_grades grades / num_grades grades

theorem fraction_of_satisfactory_grades_is_3_4 
  (grades : String → ℕ)
  (hA : grades "A" = 5)
  (hB : grades "B" = 4)
  (hC : grades "C" = 3)
  (hD : grades "D" = 3)
  (hF : grades "F" = 5) : 
  fraction_satisfactory grades = (3 : ℚ) / 4 := by
{
  sorry
}

end fraction_of_satisfactory_grades_is_3_4_l112_11252


namespace cars_equilibrium_l112_11269

variable (days : ℕ) -- number of days after which we need the condition to hold
variable (carsA_init carsB_init carsA_to_B carsB_to_A : ℕ) -- initial conditions and parameters

theorem cars_equilibrium :
  let cars_total := 192 + 48
  let carsA := carsA_init + (carsB_to_A - carsA_to_B) * days
  let carsB := carsB_init + (carsA_to_B - carsB_to_A) * days
  carsA_init = 192 -> carsB_init = 48 ->
  carsA_to_B = 21 -> carsB_to_A = 24 ->
  cars_total = 192 + 48 ->
  days = 6 ->
  cars_total = carsA + carsB -> carsA = 7 * carsB :=
by
  intros
  sorry

end cars_equilibrium_l112_11269


namespace sum_of_two_integers_l112_11259

noncomputable def sum_of_integers (a b : ℕ) : ℕ :=
a + b

theorem sum_of_two_integers (a b : ℕ) (h1 : a - b = 14) (h2 : a * b = 120) : sum_of_integers a b = 26 := 
by
  sorry

end sum_of_two_integers_l112_11259


namespace find_m_l112_11210

-- Given the condition
def condition (m : ℕ) := (1 / 5 : ℝ)^m * (1 / 4 : ℝ)^2 = 1 / (10 : ℝ)^4

-- Theorem to prove that m is 4 given the condition
theorem find_m (m : ℕ) (h : condition m) : m = 4 :=
sorry

end find_m_l112_11210


namespace incorrect_conclusions_l112_11200

theorem incorrect_conclusions :
  let p := (∀ x y : ℝ, x * y ≠ 6 → x ≠ 2 ∨ y ≠ 3)
  let q := (2, 1) ∈ { p : ℝ × ℝ | p.2 = 2 * p.1 - 3 }
  (p ∨ ¬q) = false ∧ (¬p ∨ q) = false ∧ (p ∧ ¬q) = false :=
by
  sorry

end incorrect_conclusions_l112_11200


namespace interior_angle_of_regular_polygon_l112_11270

theorem interior_angle_of_regular_polygon (n : ℕ) (h_diagonals : n * (n - 3) / 2 = n) :
    n = 5 ∧ (5 - 2) * 180 / 5 = 108 := by
  sorry

end interior_angle_of_regular_polygon_l112_11270


namespace find_additional_discount_percentage_l112_11297

noncomputable def additional_discount_percentage(msrp : ℝ) (max_regular_discount : ℝ) (lowest_price : ℝ) : ℝ :=
  let regular_discount_price := msrp * (1 - max_regular_discount)
  let additional_discount := (regular_discount_price - lowest_price) / regular_discount_price
  additional_discount * 100

theorem find_additional_discount_percentage :
  additional_discount_percentage 40 0.3 22.4 = 20 :=
by
  unfold additional_discount_percentage
  simp
  sorry

end find_additional_discount_percentage_l112_11297


namespace police_station_distance_l112_11211

theorem police_station_distance (thief_speed police_speed: ℝ) (delay chase_time: ℝ) 
  (h_thief_speed: thief_speed = 20) 
  (h_police_speed: police_speed = 40) 
  (h_delay: delay = 1)
  (h_chase_time: chase_time = 4) : 
  ∃ D: ℝ, D = 60 :=
by
  sorry

end police_station_distance_l112_11211


namespace totalSolutions_l112_11220

noncomputable def systemOfEquations (a b c d a1 b1 c1 d1 x y : ℝ) : Prop :=
  a * x^2 + b * x * y + c * y^2 = d ∧ a1 * x^2 + b1 * x * y + c1 * y^2 = d1

theorem totalSolutions 
  (a b c d a1 b1 c1 d1 : ℝ) 
  (h₀ : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)
  (h₁ : a1 ≠ 0 ∨ b1 ≠ 0 ∨ c1 ≠ 0) :
  ∃ x y : ℝ, systemOfEquations a b c d a1 b1 c1 d1 x y :=
sorry

end totalSolutions_l112_11220


namespace jane_purchased_pudding_l112_11236

theorem jane_purchased_pudding (p : ℕ) 
  (ice_cream_cost_per_cone : ℕ := 5)
  (num_ice_cream_cones : ℕ := 15)
  (pudding_cost_per_cup : ℕ := 2)
  (cost_difference : ℕ := 65)
  (total_ice_cream_cost : ℕ := num_ice_cream_cones * ice_cream_cost_per_cone) 
  (total_pudding_cost : ℕ := p * pudding_cost_per_cup) :
  total_ice_cream_cost = total_pudding_cost + cost_difference → p = 5 :=
by
  sorry

end jane_purchased_pudding_l112_11236


namespace production_days_l112_11204

noncomputable def daily_production (n : ℕ) : Prop :=
50 * n + 90 = 58 * (n + 1)

theorem production_days (n : ℕ) (h : daily_production n) : n = 4 :=
by sorry

end production_days_l112_11204


namespace remainder_sum_mod_14_l112_11209

theorem remainder_sum_mod_14 
  (a b c : ℕ) 
  (ha : a % 14 = 5) 
  (hb : b % 14 = 5) 
  (hc : c % 14 = 5) :
  (a + b + c) % 14 = 1 := 
by
  sorry

end remainder_sum_mod_14_l112_11209


namespace solve_fractional_eq_l112_11226

noncomputable def fractional_eq (x : ℝ) : Prop := 
  (3 / (x^2 - 3 * x) + (x - 1) / (x - 3) = 1)

noncomputable def not_zero_denom (x : ℝ) : Prop := 
  (x^2 - 3 * x ≠ 0) ∧ (x - 3 ≠ 0)

theorem solve_fractional_eq : fractional_eq (-3/2) ∧ not_zero_denom (-3/2) :=
by
  sorry

end solve_fractional_eq_l112_11226


namespace remainder_11_pow_1000_mod_500_l112_11243

theorem remainder_11_pow_1000_mod_500 : (11 ^ 1000) % 500 = 1 :=
by
  have h1 : 11 % 5 = 1 := by norm_num
  have h2 : (11 ^ 10) % 100 = 1 := by
    -- Some steps omitted to satisfy conditions; normally would be generalized
    sorry
  have h3 : 500 = 5 * 100 := by norm_num
  -- Further omitted steps aligning with the Chinese Remainder Theorem application.
  sorry

end remainder_11_pow_1000_mod_500_l112_11243


namespace evaluate_expression_l112_11261

theorem evaluate_expression : 
  ((-4 : ℤ) ^ 6) / (4 ^ 4) + (2 ^ 5) * (5 : ℤ) - (7 ^ 2) = 127 :=
by sorry

end evaluate_expression_l112_11261


namespace ratio_of_voters_l112_11281

theorem ratio_of_voters (V_X V_Y : ℝ) 
  (h1 : 0.62 * V_X + 0.38 * V_Y = 0.54 * (V_X + V_Y)) : V_X / V_Y = 2 :=
by
  sorry

end ratio_of_voters_l112_11281


namespace solution_set_I_range_of_m_II_l112_11278

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem solution_set_I : {x : ℝ | 0 ≤ x ∧ x ≤ 3} = {x : ℝ | f x ≤ 3} :=
sorry

theorem range_of_m_II (x : ℝ) (hx : x > 0) : ∃ m : ℝ, ∀ (x : ℝ), f x ≤ m - x - 4 / x → m ≥ 5 :=
sorry

end solution_set_I_range_of_m_II_l112_11278


namespace abs_sum_bound_l112_11202

theorem abs_sum_bound (k : ℝ) : (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 :=
by {
  sorry
}

end abs_sum_bound_l112_11202


namespace sum_fractions_bounds_l112_11212

theorem sum_fractions_bounds {a b c : ℝ} (h : a * b * c = 1) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 < (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) ∧ 
  (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) < 2 :=
  sorry

end sum_fractions_bounds_l112_11212


namespace find_largest_integer_l112_11286

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l112_11286


namespace express_y_in_terms_of_x_l112_11203

theorem express_y_in_terms_of_x (x y : ℝ) (h : 5 * x + y = 1) : y = 1 - 5 * x :=
by
  sorry

end express_y_in_terms_of_x_l112_11203


namespace moles_of_silver_nitrate_needed_l112_11233

structure Reaction :=
  (reagent1 : String)
  (reagent2 : String)
  (product1 : String)
  (product2 : String)
  (ratio_reagent1_to_product2 : ℕ) -- Moles of reagent1 to product2 in the balanced reaction

def silver_nitrate_hydrochloric_acid_reaction : Reaction :=
  { reagent1 := "AgNO3",
    reagent2 := "HCl",
    product1 := "AgCl",
    product2 := "HNO3",
    ratio_reagent1_to_product2 := 1 }

theorem moles_of_silver_nitrate_needed
  (reaction : Reaction)
  (hCl_initial_moles : ℕ)
  (hno3_target_moles : ℕ) :
  hno3_target_moles = 2 →
  (reaction.ratio_reagent1_to_product2 = 1 ∧ hCl_initial_moles = 2) →
  (hno3_target_moles = reaction.ratio_reagent1_to_product2 * 2 ∧ hno3_target_moles = 2) :=
by
  sorry

end moles_of_silver_nitrate_needed_l112_11233


namespace number_of_persons_in_room_l112_11207

theorem number_of_persons_in_room (n : ℕ) (h : n * (n - 1) / 2 = 78) : n = 13 :=
by
  /- We have:
     n * (n - 1) / 2 = 78,
     We need to prove n = 13 -/
  sorry

end number_of_persons_in_room_l112_11207


namespace no_such_polynomial_exists_l112_11255

theorem no_such_polynomial_exists :
  ∀ (P : ℤ → ℤ), (∃ a b c d : ℤ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
                  P a = 3 ∧ P b = 3 ∧ P c = 3 ∧ P d = 4) → false :=
by
  sorry

end no_such_polynomial_exists_l112_11255


namespace borrowed_years_l112_11215

noncomputable def principal : ℝ := 5396.103896103896
noncomputable def interest_rate : ℝ := 0.06
noncomputable def total_returned : ℝ := 8310

theorem borrowed_years :
  ∃ t : ℝ, (total_returned - principal) = principal * interest_rate * t ∧ t = 9 :=
by
  sorry

end borrowed_years_l112_11215


namespace probability_points_one_unit_apart_l112_11213

theorem probability_points_one_unit_apart :
  let points := 10
  let rect_length := 3
  let rect_width := 2
  let total_pairs := (points * (points - 1)) / 2
  let favorable_pairs := 10  -- derived from solution steps
  (favorable_pairs / total_pairs : ℚ) = (2 / 9 : ℚ) :=
by
  sorry

end probability_points_one_unit_apart_l112_11213


namespace compound_interest_rate_l112_11294

theorem compound_interest_rate (SI CI : ℝ) (P1 P2 : ℝ) (T1 T2 : ℝ) (R1 : ℝ) (R : ℝ) 
    (H1 : SI = (P1 * R1 * T1) / 100)
    (H2 : CI = 2 * SI)
    (H3 : CI = P2 * ((1 + R/100)^2 - 1))
    (H4 : P1 = 1272)
    (H5 : P2 = 5000)
    (H6 : T1 = 5)
    (H7 : T2 = 2)
    (H8 : R1 = 10) :
  R = 12 :=
by
  sorry

end compound_interest_rate_l112_11294


namespace fuel_relationship_l112_11279

theorem fuel_relationship (y : ℕ → ℕ) (h₀ : y 0 = 80) (h₁ : y 1 = 70) (h₂ : y 2 = 60) (h₃ : y 3 = 50) :
  ∀ x : ℕ, y x = 80 - 10 * x :=
by
  sorry

end fuel_relationship_l112_11279


namespace no_negatives_l112_11276

theorem no_negatives (x y : ℝ) (h : |x^2 + y^2 - 4*x - 4*y + 5| = |2*x + 2*y - 4|) : 
  ¬ (x < 0) ∧ ¬ (y < 0) :=
by
  sorry

end no_negatives_l112_11276


namespace randy_initial_blocks_l112_11234

theorem randy_initial_blocks (x : ℕ) (used_blocks : ℕ) (left_blocks : ℕ) 
  (h1 : used_blocks = 36) (h2 : left_blocks = 23) (h3 : x = used_blocks + left_blocks) :
  x = 59 := by 
  sorry

end randy_initial_blocks_l112_11234


namespace correct_value_l112_11265

theorem correct_value : ∀ (x : ℕ),  (x / 6 = 12) → (x * 7 = 504) :=
  sorry

end correct_value_l112_11265


namespace tory_needs_to_raise_more_l112_11283

variable (goal : ℕ) (pricePerChocolateChip pricePerOatmealRaisin pricePerSugarCookie : ℕ)
variable (soldChocolateChip soldOatmealRaisin soldSugarCookie : ℕ)

def remainingAmount (goal : ℕ) 
                    (pricePerChocolateChip pricePerOatmealRaisin pricePerSugarCookie : ℕ)
                    (soldChocolateChip soldOatmealRaisin soldSugarCookie : ℕ) : ℕ :=
  let profitFromChocolateChip := soldChocolateChip * pricePerChocolateChip
  let profitFromOatmealRaisin := soldOatmealRaisin * pricePerOatmealRaisin
  let profitFromSugarCookie := soldSugarCookie * pricePerSugarCookie
  let totalProfit := profitFromChocolateChip + profitFromOatmealRaisin + profitFromSugarCookie
  goal - totalProfit

theorem tory_needs_to_raise_more : 
  remainingAmount 250 6 5 4 5 10 15 = 110 :=
by
  -- Proof omitted 
  sorry

end tory_needs_to_raise_more_l112_11283


namespace Allan_more_balloons_l112_11268

-- Define the number of balloons that Allan and Jake brought
def Allan_balloons := 5
def Jake_balloons := 3

-- Prove that the number of more balloons that Allan had than Jake is 2
theorem Allan_more_balloons : (Allan_balloons - Jake_balloons) = 2 := by sorry

end Allan_more_balloons_l112_11268


namespace log_base_4_of_8_l112_11288

noncomputable def log_base_change (b a c : ℝ) : ℝ :=
  Real.log a / Real.log b

theorem log_base_4_of_8 : log_base_change 4 8 10 = 3 / 2 :=
by
  have h1 : Real.log 8 = 3 * Real.log 2 := by
    sorry  -- Use properties of logarithms: 8 = 2^3
  have h2 : Real.log 4 = 2 * Real.log 2 := by
    sorry  -- Use properties of logarithms: 4 = 2^2
  have h3 : log_base_change 4 8 10 = (3 * Real.log 2) / (2 * Real.log 2) := by
    rw [log_base_change, h1, h2]
  have h4 : (3 * Real.log 2) / (2 * Real.log 2) = 3 / 2 := by
    sorry  -- Simplify the fraction
  rw [h3, h4]

end log_base_4_of_8_l112_11288


namespace sum_three_smallest_m_l112_11247

theorem sum_three_smallest_m :
  (∃ a m, 
    (a - 2 + a + a + 2) / 3 = 7 
    ∧ m % 4 = 3 
    ∧ m ≠ 5 ∧ m ≠ 7 ∧ m ≠ 9 
    ∧ (5 + 7 + 9 + m) % 4 = 0 
    ∧ m > 0) 
  → 3 + 11 + 15 = 29 :=
sorry

end sum_three_smallest_m_l112_11247


namespace find_AB_l112_11257

-- Definitions based on conditions
variables (AB CD : ℝ)

-- Given conditions
def area_ratio_condition : Prop :=
  AB / CD = 5 / 3

def sum_condition : Prop :=
  AB + CD = 160

-- The main statement to be proven
theorem find_AB (h_ratio : area_ratio_condition AB CD) (h_sum : sum_condition AB CD) :
  AB = 100 :=
by
  sorry

end find_AB_l112_11257


namespace shaded_rectangle_area_l112_11291

theorem shaded_rectangle_area (side_length : ℝ) (x y : ℝ) 
  (h1 : side_length = 42) 
  (h2 : 4 * x + 2 * y = 168 - 4 * x) 
  (h3 : 2 * (side_length - y) + 2 * x = 168 - 4 * x)
  (h4 : 2 * (2 * x + y) = 168 - 4 * x) 
  (h5 : x = 18) :
  (2 * x) * (4 * x - (side_length - y)) = 540 := 
by
  sorry

end shaded_rectangle_area_l112_11291


namespace calculate_expression_l112_11217

theorem calculate_expression :
  (π - 1)^0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end calculate_expression_l112_11217


namespace find_sum_l112_11282

theorem find_sum (x y : ℝ) (h₁ : 3 * |x| + 2 * x + y = 20) (h₂ : 2 * x + 3 * |y| - y = 30) : x + y = 15 :=
sorry

end find_sum_l112_11282


namespace correct_division_result_l112_11287

theorem correct_division_result {x : ℕ} (h : 3 * x = 90) : x / 3 = 10 :=
by
  -- placeholder for the actual proof
  sorry

end correct_division_result_l112_11287


namespace dart_game_solution_l112_11296

theorem dart_game_solution (x y z : ℕ) (h_x : 8 * x + 9 * y + 10 * z = 100) (h_y : x + y + z > 11) :
  (x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0) :=
by
  sorry

end dart_game_solution_l112_11296


namespace number_of_cuboids_painted_l112_11239

/--
Suppose each cuboid has 6 outer faces and Amelia painted a total of 36 faces.
Prove that the number of cuboids Amelia painted is 6.
-/
theorem number_of_cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) 
  (h1 : total_faces = 36) (h2 : faces_per_cuboid = 6) :
  total_faces / faces_per_cuboid = 6 := 
by {
  sorry
}

end number_of_cuboids_painted_l112_11239


namespace root_implies_m_values_l112_11237

theorem root_implies_m_values (m : ℝ) :
  (∃ x : ℝ, x = 1 ∧ (m + 2) * x^2 - 2 * x + m^2 - 2 * m - 6 = 0) →
  (m = 3 ∨ m = -2) :=
by
  sorry

end root_implies_m_values_l112_11237


namespace probability_not_all_dice_show_different_l112_11245

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l112_11245


namespace symmetric_point_proof_l112_11216

def Point3D := (ℝ × ℝ × ℝ)

def symmetric_point_yOz (p : Point3D) : Point3D :=
  let (x, y, z) := p
  (-x, y, z)

theorem symmetric_point_proof :
  symmetric_point_yOz (1, -2, 3) = (-1, -2, 3) :=
by
  sorry

end symmetric_point_proof_l112_11216


namespace fluorescent_bulbs_switched_on_percentage_l112_11299

theorem fluorescent_bulbs_switched_on_percentage (I F : ℕ) (x : ℝ) (Inc_on F_on total_on Inc_on_ratio : ℝ) 
  (h1 : Inc_on = 0.3 * I) 
  (h2 : total_on = 0.7 * (I + F)) 
  (h3 : Inc_on_ratio = 0.08571428571428571) 
  (h4 : Inc_on_ratio = Inc_on / total_on) 
  (h5 : total_on = Inc_on + F_on) 
  (h6 : F_on = x * F) :
  x = 0.9 :=
sorry

end fluorescent_bulbs_switched_on_percentage_l112_11299


namespace polynomial_exists_int_coeff_l112_11254

theorem polynomial_exists_int_coeff (n : ℕ) (hn : n > 1) : 
  ∃ P : Polynomial ℤ × Polynomial ℤ × Polynomial ℤ → Polynomial ℤ, 
  ∀ x : Polynomial ℤ, P ⟨x^n, x^(n+1), x + x^(n+2)⟩ = x :=
by sorry

end polynomial_exists_int_coeff_l112_11254
