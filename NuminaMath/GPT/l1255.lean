import Mathlib

namespace NUMINAMATH_GPT_square_of_real_not_always_positive_l1255_125567

theorem square_of_real_not_always_positive (a : ℝ) : ¬(a^2 > 0) := 
sorry

end NUMINAMATH_GPT_square_of_real_not_always_positive_l1255_125567


namespace NUMINAMATH_GPT_point_P_x_coordinate_l1255_125531

variable {P : Type} [LinearOrderedField P]

-- Definitions from the conditions
def line_equation (x : P) : P := 0.8 * x
def y_coordinate_P : P := 6
def x_coordinate_P : P := 7.5

-- Theorems to prove that the x-coordinate of P is 7.5.
theorem point_P_x_coordinate (x : P) :
  line_equation x = y_coordinate_P → x = x_coordinate_P :=
by
  intro h
  sorry

end NUMINAMATH_GPT_point_P_x_coordinate_l1255_125531


namespace NUMINAMATH_GPT_exponential_comparisons_l1255_125541

open Real

noncomputable def a : ℝ := 5 ^ (log 3.4 / log 2)
noncomputable def b : ℝ := 5 ^ (log 3.6 / (log 4))
noncomputable def c : ℝ := 5 ^ (log (10 / 3))

theorem exponential_comparisons :
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_GPT_exponential_comparisons_l1255_125541


namespace NUMINAMATH_GPT_ratio_of_height_to_radius_max_volume_l1255_125592

theorem ratio_of_height_to_radius_max_volume (r h : ℝ) (h_surface_area : 2 * Real.pi * r^2 + 2 * Real.pi * r * h = 6 * Real.pi) :
  (exists (max_r : ℝ) (max_h : ℝ), 2 * r * max_r + 2 * r * max_h = 6 * Real.pi ∧ 
                                  max_r = 1 ∧ 
                                  max_h = 2 ∧ 
                                  (max_h / max_r) = 2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_height_to_radius_max_volume_l1255_125592


namespace NUMINAMATH_GPT_smallest_prime_dividing_sum_l1255_125588

theorem smallest_prime_dividing_sum (h1 : Odd 7) (h2 : Odd 9) 
    (h3 : ∀ {a b : ℤ}, Odd a → Odd b → Even (a + b)) :
  ∃ p : ℕ, Prime p ∧ p ∣ (7 ^ 15 + 9 ^ 7) ∧ p = 2 := 
by
  sorry

end NUMINAMATH_GPT_smallest_prime_dividing_sum_l1255_125588


namespace NUMINAMATH_GPT_fraction_tips_l1255_125568

theorem fraction_tips {S : ℝ} (H1 : S > 0) (H2 : tips = (7 / 3 : ℝ) * S) (H3 : bonuses = (2 / 5 : ℝ) * S) :
  (tips / (S + tips + bonuses)) = (5 / 8 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_tips_l1255_125568


namespace NUMINAMATH_GPT_find_g3_value_l1255_125523

def g (n : ℕ) : ℕ :=
  if n < 5 then 2 * n ^ 2 + 3 else 4 * n + 1

theorem find_g3_value : g (g (g 3)) = 341 := by
  sorry

end NUMINAMATH_GPT_find_g3_value_l1255_125523


namespace NUMINAMATH_GPT_data_point_frequency_l1255_125522

theorem data_point_frequency 
  (data : Type) 
  (categories : data → Prop) 
  (group_counts : data → ℕ) :
  ∀ d, categories d → group_counts d = frequency := sorry

end NUMINAMATH_GPT_data_point_frequency_l1255_125522


namespace NUMINAMATH_GPT_smallest_number_l1255_125516

theorem smallest_number (x : ℕ) (h1 : (x + 7) % 8 = 0) (h2 : (x + 7) % 11 = 0) (h3 : (x + 7) % 24 = 0) : x = 257 :=
sorry

end NUMINAMATH_GPT_smallest_number_l1255_125516


namespace NUMINAMATH_GPT_marble_count_l1255_125583

-- Define the variables for the number of marbles
variables (o p y : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := o = 1.3 * p
def condition2 : Prop := y = 1.5 * o

-- Define the total number of marbles based on the conditions
def total_marbles : ℝ := o + p + y

-- The theorem statement that needs to be proved
theorem marble_count (h1 : condition1 o p) (h2 : condition2 o y) : total_marbles o p y = 3.269 * o :=
by sorry

end NUMINAMATH_GPT_marble_count_l1255_125583


namespace NUMINAMATH_GPT_profit_is_55_l1255_125574

-- Define the given conditions:
def cost_of_chocolates (bars: ℕ) (price_per_bar: ℕ) : ℕ :=
  bars * price_per_bar

def cost_of_packaging (bars: ℕ) (cost_per_bar: ℕ) : ℕ :=
  bars * cost_per_bar

def total_sales : ℕ :=
  90

def total_cost (cost_of_chocolates cost_of_packaging: ℕ) : ℕ :=
  cost_of_chocolates + cost_of_packaging

def profit (total_sales total_cost: ℕ) : ℕ :=
  total_sales - total_cost

-- Given values:
def bars: ℕ := 5
def price_per_bar: ℕ := 5
def cost_per_packaging_bar: ℕ := 2

-- Define the profit calculation theorem:
theorem profit_is_55 : 
  profit total_sales (total_cost (cost_of_chocolates bars price_per_bar) (cost_of_packaging bars cost_per_packaging_bar)) = 55 :=
by {
  -- The proof will be inserted here
  sorry
}

end NUMINAMATH_GPT_profit_is_55_l1255_125574


namespace NUMINAMATH_GPT_f_equality_2019_l1255_125536

theorem f_equality_2019 (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), f (m + n) ≥ f m + f (f n) - 1) : 
  f 2019 = 2019 :=
sorry

end NUMINAMATH_GPT_f_equality_2019_l1255_125536


namespace NUMINAMATH_GPT_pipes_fill_cistern_time_l1255_125503

noncomputable def pipe_fill_time : ℝ :=
  let rateA := 1 / 80
  let rateC := 1 / 60
  let combined_rateAB := 1 / 20
  let rateB := combined_rateAB - rateA
  let combined_rateABC := rateA + rateB - rateC
  1 / combined_rateABC

theorem pipes_fill_cistern_time :
  pipe_fill_time = 30 := by
  sorry

end NUMINAMATH_GPT_pipes_fill_cistern_time_l1255_125503


namespace NUMINAMATH_GPT_zoe_correct_percentage_l1255_125549

variable (t : ℝ) -- total number of problems

-- Conditions
variable (chloe_solved_fraction : ℝ := 0.60)
variable (zoe_solved_fraction : ℝ := 0.40)
variable (chloe_correct_percentage_alone : ℝ := 0.75)
variable (chloe_correct_percentage_total : ℝ := 0.85)
variable (zoe_correct_percentage_alone : ℝ := 0.95)

theorem zoe_correct_percentage (h1 : chloe_solved_fraction = 0.60)
                               (h2 : zoe_solved_fraction = 0.40)
                               (h3 : chloe_correct_percentage_alone = 0.75)
                               (h4 : chloe_correct_percentage_total = 0.85)
                               (h5 : zoe_correct_percentage_alone = 0.95) :
  (zoe_correct_percentage_alone * zoe_solved_fraction * 100 + (chloe_correct_percentage_total - chloe_correct_percentage_alone * chloe_solved_fraction) * 100 = 78) :=
sorry

end NUMINAMATH_GPT_zoe_correct_percentage_l1255_125549


namespace NUMINAMATH_GPT_sequence_formula_l1255_125548

theorem sequence_formula (a : ℕ → ℤ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a n - a (n + 1) + 2 = 0) :
  ∀ n : ℕ, a n = 2 * n - 1 := 
sorry

end NUMINAMATH_GPT_sequence_formula_l1255_125548


namespace NUMINAMATH_GPT_sample_second_grade_l1255_125511

theorem sample_second_grade (r1 r2 r3 sample_size : ℕ) (h1 : r1 = 3) (h2 : r2 = 3) (h3 : r3 = 4) (h_sample_size : sample_size = 50) : (r2 * sample_size) / (r1 + r2 + r3) = 15 := by
  sorry

end NUMINAMATH_GPT_sample_second_grade_l1255_125511


namespace NUMINAMATH_GPT_num_ways_for_volunteers_l1255_125575

theorem num_ways_for_volunteers:
  let pavilions := 4
  let volunteers := 5
  let ways_to_choose_A := 4
  let ways_to_choose_B_after_A := 3
  let total_distributions := 
    let case_1 := 2
    let case_2 := (2^3) - 2
    case_1 + case_2
  ways_to_choose_A * ways_to_choose_B_after_A * total_distributions = 72 := 
by
  sorry

end NUMINAMATH_GPT_num_ways_for_volunteers_l1255_125575


namespace NUMINAMATH_GPT_problem_statement_l1255_125538

theorem problem_statement (n : ℕ) (hn : n > 0) : (122 ^ n - 102 ^ n - 21 ^ n) % 2020 = 2019 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1255_125538


namespace NUMINAMATH_GPT_mean_of_two_numbers_l1255_125515

theorem mean_of_two_numbers (a b : ℝ) (mean_twelve : ℝ) (mean_fourteen : ℝ) 
  (h1 : mean_twelve = 60) 
  (h2 : mean_fourteen = 75) 
  (sum_twelve : 12 * mean_twelve = 720) 
  (sum_fourteen : 14 * mean_fourteen = 1050) 
  : (a + b) / 2 = 165 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_two_numbers_l1255_125515


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1255_125565

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 6) 
  (h2 : a 3 + a 5 + a 7 = 78) :
  a 5 = 18 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1255_125565


namespace NUMINAMATH_GPT_int_div_condition_l1255_125540

theorem int_div_condition (n : ℕ) (hn₁ : ∃ m : ℤ, 2^n - 2 = m * n) :
  ∃ k : ℤ, 2^(2^n - 1) - 2 = k * (2^n - 1) :=
by sorry

end NUMINAMATH_GPT_int_div_condition_l1255_125540


namespace NUMINAMATH_GPT_candy_from_sister_l1255_125530

variable (f : ℕ) (e : ℕ) (t : ℕ)

theorem candy_from_sister (h₁ : f = 47) (h₂ : e = 25) (h₃ : t = 62) :
  ∃ x : ℕ, x = t - (f - e) ∧ x = 40 :=
by sorry

end NUMINAMATH_GPT_candy_from_sister_l1255_125530


namespace NUMINAMATH_GPT_basketball_court_perimeter_l1255_125582

variables {Width Length : ℕ}

def width := 17
def length := 31

def perimeter (width length : ℕ) := 2 * (length + width)

theorem basketball_court_perimeter : 
  perimeter width length = 96 :=
sorry

end NUMINAMATH_GPT_basketball_court_perimeter_l1255_125582


namespace NUMINAMATH_GPT_proposition_3_proposition_4_l1255_125547

variables {Plane : Type} {Line : Type} 
variables {α β : Plane} {a b : Line}

-- Assuming necessary properties of parallel planes and lines being subsets of planes
axiom plane_parallel (α β : Plane) : Prop
axiom line_in_plane (l : Line) (p : Plane) : Prop
axiom line_parallel (l m : Line) : Prop
axiom lines_skew (l m : Line) : Prop
axiom lines_coplanar (l m : Line) : Prop
axiom lines_do_not_intersect (l m : Line) : Prop

-- Assume the given conditions
variables (h1 : plane_parallel α β) 
variables (h2 : line_in_plane a α)
variables (h3 : line_in_plane b β)

-- State the equivalent proof problem as propositions to be proved in Lean
theorem proposition_3 (h1 : plane_parallel α β) 
                     (h2 : line_in_plane a α) 
                     (h3 : line_in_plane b β) : 
                     lines_do_not_intersect a b :=
sorry

theorem proposition_4 (h1 : plane_parallel α β) 
                     (h2 : line_in_plane a α) 
                     (h3 : line_in_plane b β) : 
                     lines_coplanar a b ∨ lines_skew a b :=
sorry

end NUMINAMATH_GPT_proposition_3_proposition_4_l1255_125547


namespace NUMINAMATH_GPT_system_non_zero_solution_condition_l1255_125502

theorem system_non_zero_solution_condition (a b c : ℝ) :
  (∃ (x y z : ℝ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∧
    x = b * y + c * z ∧
    y = c * z + a * x ∧
    z = a * x + b * y) ↔
  (2 * a * b * c + a * b + b * c + c * a - 1 = 0) :=
sorry

end NUMINAMATH_GPT_system_non_zero_solution_condition_l1255_125502


namespace NUMINAMATH_GPT_xyz_value_l1255_125550

variable {x y z : ℝ}

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 18) 
                  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 6) : 
                  x * y * z = 4 := 
by
  sorry

end NUMINAMATH_GPT_xyz_value_l1255_125550


namespace NUMINAMATH_GPT_pizza_dough_milk_needed_l1255_125507

variable (milk_per_300 : ℕ) (flour_per_batch : ℕ) (total_flour : ℕ)

-- Definitions based on problem conditions
def milk_per_batch := milk_per_300
def batch_size := flour_per_batch
def used_flour := total_flour

-- The target proof statement
theorem pizza_dough_milk_needed (h1 : milk_per_batch = 60) (h2 : batch_size = 300) (h3 : used_flour = 1500) : 
  (used_flour / batch_size) * milk_per_batch = 300 :=
by
  rw [h1, h2, h3]
  sorry -- proof steps

end NUMINAMATH_GPT_pizza_dough_milk_needed_l1255_125507


namespace NUMINAMATH_GPT_simplify_expression_l1255_125599

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1255_125599


namespace NUMINAMATH_GPT_intersection_complement_l1255_125535

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def B : Set ℝ := {x | 2 < x ∧ x < 4}
def complement_U (A : Set ℝ) : Set ℝ := {x | ¬ (A x)}

theorem intersection_complement (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) :
  B ∩ (complement_U A) = {x | 2 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1255_125535


namespace NUMINAMATH_GPT_Jill_llamas_count_l1255_125525

theorem Jill_llamas_count :
  let initial_pregnant_with_one_calf := 9
  let initial_pregnant_with_twins := 5
  let total_calves_born := (initial_pregnant_with_one_calf * 1) + (initial_pregnant_with_twins * 2)
  let calves_after_trade := total_calves_born - 8
  let initial_pregnant_lamas := initial_pregnant_with_one_calf + initial_pregnant_with_twins
  let total_lamas_after_birth := initial_pregnant_lamas + total_calves_born
  let lamas_after_trade := total_lamas_after_birth - 8 + 2
  let lamas_sold := lamas_after_trade / 3
  let final_lamas := lamas_after_trade - lamas_sold
  final_lamas = 18 :=
by
  sorry

end NUMINAMATH_GPT_Jill_llamas_count_l1255_125525


namespace NUMINAMATH_GPT_total_sequins_correct_l1255_125544

def blue_rows : ℕ := 6
def blue_columns : ℕ := 8
def purple_rows : ℕ := 5
def purple_columns : ℕ := 12
def green_rows : ℕ := 9
def green_columns : ℕ := 6

def total_sequins : ℕ :=
  (blue_rows * blue_columns) + (purple_rows * purple_columns) + (green_rows * green_columns)

theorem total_sequins_correct : total_sequins = 162 := by
  sorry

end NUMINAMATH_GPT_total_sequins_correct_l1255_125544


namespace NUMINAMATH_GPT_total_commute_time_l1255_125528

theorem total_commute_time 
  (first_bus : ℕ) (delay1 : ℕ) (wait1 : ℕ) 
  (second_bus : ℕ) (delay2 : ℕ) (wait2 : ℕ) 
  (third_bus : ℕ) (delay3 : ℕ) 
  (arrival_time : ℕ) :
  first_bus = 40 →
  delay1 = 10 →
  wait1 = 10 →
  second_bus = 50 →
  delay2 = 5 →
  wait2 = 15 →
  third_bus = 95 →
  delay3 = 15 →
  arrival_time = 540 →
  first_bus + delay1 + wait1 + second_bus + delay2 + wait2 + third_bus + delay3 = 240 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_commute_time_l1255_125528


namespace NUMINAMATH_GPT_combination_property_l1255_125586

theorem combination_property (x : ℕ) (hx : 2 * x - 1 ≤ 11 ∧ x ≤ 11) :
  (Nat.choose 11 (2 * x - 1) = Nat.choose 11 x) → (x = 1 ∨ x = 4) :=
by
  sorry

end NUMINAMATH_GPT_combination_property_l1255_125586


namespace NUMINAMATH_GPT_magician_method_N_2k_magician_method_values_l1255_125594

-- (a) Prove that if there is a method for N = k, then there is a method for N = 2k.
theorem magician_method_N_2k (k : ℕ) (method_k : Prop) : 
  (∃ method_N_k : Prop, method_k → method_N_k) → 
  (∃ method_N_2k : Prop, method_k → method_N_2k) :=
sorry

-- (b) Find all values of N for which the magician and the assistant have a method.
theorem magician_method_values (N : ℕ) : 
  (∃ method : Prop, method) ↔ (∃ m : ℕ, N = 2^m) :=
sorry

end NUMINAMATH_GPT_magician_method_N_2k_magician_method_values_l1255_125594


namespace NUMINAMATH_GPT_factorization_of_x_squared_minus_one_l1255_125587

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end NUMINAMATH_GPT_factorization_of_x_squared_minus_one_l1255_125587


namespace NUMINAMATH_GPT_find_fourth_month_sale_l1255_125543

theorem find_fourth_month_sale (s1 s2 s3 s4 s5 : ℕ) (avg_sale nL5 : ℕ)
  (h1 : s1 = 5420)
  (h2 : s2 = 5660)
  (h3 : s3 = 6200)
  (h5 : s5 = 6500)
  (havg : avg_sale = 6300)
  (hnL5 : nL5 = 5)
  (h_average : avg_sale * nL5 = s1 + s2 + s3 + s4 + s5) :
  s4 = 7720 := sorry

end NUMINAMATH_GPT_find_fourth_month_sale_l1255_125543


namespace NUMINAMATH_GPT_min_value_x_y_l1255_125546

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 8 / y = 1) : x + y ≥ 18 := 
sorry

end NUMINAMATH_GPT_min_value_x_y_l1255_125546


namespace NUMINAMATH_GPT_min_fuse_length_l1255_125551

theorem min_fuse_length 
  (safe_distance : ℝ := 70) 
  (personnel_speed : ℝ := 7) 
  (fuse_burning_speed : ℝ := 10.3) : 
  ∃ (x : ℝ), x ≥ 103 := 
by
  sorry

end NUMINAMATH_GPT_min_fuse_length_l1255_125551


namespace NUMINAMATH_GPT_base6_sum_l1255_125590

-- Define each of the numbers in base 6
def base6_555 : ℕ := 5 * 6^2 + 5 * 6^1 + 5 * 6^0
def base6_55 : ℕ := 5 * 6^1 + 5 * 6^0
def base6_5 : ℕ := 5 * 6^0
def base6_1103 : ℕ := 1 * 6^3 + 1 * 6^2 + 0 * 6^1 + 3 * 6^0 

-- The problem statement is to prove the sum equals the expected result in base 6
theorem base6_sum : base6_555 + base6_55 + base6_5 = base6_1103 :=
by
  sorry

end NUMINAMATH_GPT_base6_sum_l1255_125590


namespace NUMINAMATH_GPT_acres_for_corn_l1255_125579

theorem acres_for_corn (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ)
  (total_ratio : beans_ratio + wheat_ratio + corn_ratio = 11)
  (land_parts : total_land / 11 = 94)
  : (corn_ratio = 4) → (total_land = 1034) → 4 * 94 = 376 :=
by
  intros
  sorry

end NUMINAMATH_GPT_acres_for_corn_l1255_125579


namespace NUMINAMATH_GPT_check_not_coverable_boards_l1255_125517

def is_coverable_by_dominoes (m n : ℕ) : Prop :=
  (m * n) % 2 = 0

theorem check_not_coverable_boards:
  (¬is_coverable_by_dominoes 5 5) ∧ (¬is_coverable_by_dominoes 3 7) :=
by
  -- Proof steps are omitted.
  sorry

end NUMINAMATH_GPT_check_not_coverable_boards_l1255_125517


namespace NUMINAMATH_GPT_correct_transformation_l1255_125585

-- Definitions of the points and their mapped coordinates
def C : ℝ × ℝ := (3, -2)
def D : ℝ × ℝ := (4, -3)
def C' : ℝ × ℝ := (1, 2)
def D' : ℝ × ℝ := (-2, 3)

-- Transformation function (as given in the problem)
def skew_reflection_and_vertical_shrink (p : ℝ × ℝ) : ℝ × ℝ :=
  match p with
  | (x, y) => (-y, x)

-- Theorem statement to be proved
theorem correct_transformation :
  skew_reflection_and_vertical_shrink C = C' ∧ skew_reflection_and_vertical_shrink D = D' :=
sorry

end NUMINAMATH_GPT_correct_transformation_l1255_125585


namespace NUMINAMATH_GPT_find_sum_of_abc_l1255_125584

theorem find_sum_of_abc
  (a b c x y : ℕ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a^2 + b^2 + c^2 = 2011)
  (h3 : Nat.gcd a (Nat.gcd b c) = x)
  (h4 : Nat.lcm a (Nat.lcm b c) = y)
  (h5 : x + y = 388)
  :
  a + b + c = 61 :=
sorry

end NUMINAMATH_GPT_find_sum_of_abc_l1255_125584


namespace NUMINAMATH_GPT_correct_option_is_C_l1255_125563

-- Definitions based on the problem conditions
def option_A : Prop := (-3 + (-3)) = 0
def option_B : Prop := (-3 - abs (-3)) = 0
def option_C (a b : ℝ) : Prop := (3 * a^2 * b - 4 * b * a^2) = - a^2 * b
def option_D (x : ℝ) : Prop := (-(5 * x - 2)) = -5 * x - 2

-- The theorem to be proved that option C is the correct calculation
theorem correct_option_is_C (a b : ℝ) : option_C a b :=
sorry

end NUMINAMATH_GPT_correct_option_is_C_l1255_125563


namespace NUMINAMATH_GPT_prove_f_of_pi_div_4_eq_0_l1255_125509

noncomputable
def tan_function (ω : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x)

theorem prove_f_of_pi_div_4_eq_0 
  (ω : ℝ) (hω : ω > 0)
  (h_period : ∀ x : ℝ, tan_function ω (x + π / (4 * ω)) = tan_function ω x) :
  tan_function ω (π / 4) = 0 :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_prove_f_of_pi_div_4_eq_0_l1255_125509


namespace NUMINAMATH_GPT_pipe_cistern_l1255_125560

theorem pipe_cistern (rate: ℚ) (duration: ℚ) (portion: ℚ) : 
  rate = (2/3) / 10 → duration = 8 → portion = 8/15 →
  portion = duration * rate := 
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_pipe_cistern_l1255_125560


namespace NUMINAMATH_GPT_essentially_different_proportions_l1255_125596

theorem essentially_different_proportions (x y z t : α) [DecidableEq α] 
  (h1 : x ≠ y) (h2 : x ≠ z) (h3 : x ≠ t) (h4 : y ≠ z) (h5 : y ≠ t) (h6 : z ≠ t) : 
  ∃ n : ℕ, n = 3 := by
  sorry

end NUMINAMATH_GPT_essentially_different_proportions_l1255_125596


namespace NUMINAMATH_GPT_travel_time_second_bus_l1255_125521

def distance_AB : ℝ := 100 -- kilometers
def passengers_first : ℕ := 20
def speed_first : ℝ := 60 -- kilometers per hour
def breakdown_time : ℝ := 0.5 -- hours
def passengers_second_initial : ℕ := 22
def speed_second_initial : ℝ := 50 -- kilometers per hour
def additional_passengers_speed_decrease : ℝ := 1 -- speed decrease for every additional 2 passengers
def passenger_factor : ℝ := 2
def additional_passengers : ℕ := 20
def total_time_second_bus : ℝ := 2.35 -- hours

theorem travel_time_second_bus :
  let distance_first_half := (breakdown_time * speed_first)
  let remaining_distance := distance_AB - distance_first_half
  let time_to_reach_breakdown := distance_first_half / speed_second_initial
  let new_speed_second_bus := speed_second_initial - (additional_passengers / passenger_factor) * additional_passengers_speed_decrease
  let time_from_breakdown_to_B := remaining_distance / new_speed_second_bus
  total_time_second_bus = time_to_reach_breakdown + time_from_breakdown_to_B := 
sorry

end NUMINAMATH_GPT_travel_time_second_bus_l1255_125521


namespace NUMINAMATH_GPT_simple_interest_sum_l1255_125578

theorem simple_interest_sum (P_SI : ℕ) :
  let P_CI := 5000
  let r_CI := 12
  let t_CI := 2
  let r_SI := 10
  let t_SI := 5
  let CI := (P_CI * (1 + r_CI / 100)^t_CI - P_CI)
  let SI := CI / 2
  (P_SI * r_SI * t_SI / 100 = SI) -> 
  P_SI = 1272 := by {
  sorry
}

end NUMINAMATH_GPT_simple_interest_sum_l1255_125578


namespace NUMINAMATH_GPT_find_x_minus_y_l1255_125537

variables (x y z : ℝ)

theorem find_x_minus_y (h1 : x - (y + z) = 19) (h2 : x - y - z = 7): x - y = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_minus_y_l1255_125537


namespace NUMINAMATH_GPT_sqrt_diff_nat_l1255_125534

open Nat

theorem sqrt_diff_nat (a b : ℕ) (h : 2015 * a^2 + a = 2016 * b^2 + b) : ∃ k : ℕ, a - b = k^2 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_diff_nat_l1255_125534


namespace NUMINAMATH_GPT_find_number_l1255_125561

-- Definitions and conditions for the problem
def N_div_7 (N R_1 : ℕ) : ℕ := (N / 7) * 7 + R_1
def N_div_11 (N R_2 : ℕ) : ℕ := (N / 11) * 11 + R_2
def N_div_13 (N R_3 : ℕ) : ℕ := (N / 13) * 13 + R_3

theorem find_number 
  (N a b c R_1 R_2 R_3 : ℕ) 
  (hN7 : N = 7 * a + R_1)
  (hN11 : N = 11 * b + R_2)
  (hN13 : N = 13 * c + R_3)
  (hQ : a + b + c = 21)
  (hR : R_1 + R_2 + R_3 = 21)
  (hR1_lt : R_1 < 7)
  (hR2_lt : R_2 < 11)
  (hR3_lt : R_3 < 13) : 
  N = 74 :=
sorry

end NUMINAMATH_GPT_find_number_l1255_125561


namespace NUMINAMATH_GPT_product_of_distinct_nonzero_real_satisfying_eq_l1255_125571

theorem product_of_distinct_nonzero_real_satisfying_eq (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
    (h : x + 3/x = y + 3/y) : x * y = 3 :=
by sorry

end NUMINAMATH_GPT_product_of_distinct_nonzero_real_satisfying_eq_l1255_125571


namespace NUMINAMATH_GPT_cost_of_six_burritos_and_seven_sandwiches_l1255_125556

variable (b s : ℝ)
variable (h1 : 4 * b + 2 * s = 5.00)
variable (h2 : 3 * b + 5 * s = 6.50)

theorem cost_of_six_burritos_and_seven_sandwiches : 6 * b + 7 * s = 11.50 :=
  sorry

end NUMINAMATH_GPT_cost_of_six_burritos_and_seven_sandwiches_l1255_125556


namespace NUMINAMATH_GPT_asymptotes_N_are_correct_l1255_125526

-- Given the conditions of the hyperbola M
def hyperbola_M (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / m - y^2 / 6 = 1

-- Eccentricity condition
def eccentricity (m : ℝ) (e : ℝ) : Prop :=
  e = 2 ∧ (m > 0)

-- Given hyperbola N
def hyperbola_N (x y : ℝ) (m : ℝ) : Prop :=
  x^2 - y^2 / m = 1

-- The theorem to be proved
theorem asymptotes_N_are_correct (m : ℝ) (x y : ℝ) :
  hyperbola_M x y 2 → eccentricity 2 2 → hyperbola_N x y m →
  (y = x * Real.sqrt 2 ∨ y = -x * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_asymptotes_N_are_correct_l1255_125526


namespace NUMINAMATH_GPT_fifth_term_arithmetic_sequence_l1255_125508

noncomputable def fifth_term (x y : ℚ) (a1 : ℚ := x + 2 * y) (a2 : ℚ := x - 2 * y) (a3 : ℚ := x + 2 * y^2) (a4 : ℚ := x / (2 * y)) (d : ℚ := -4 * y) : ℚ :=
    a4 + d

theorem fifth_term_arithmetic_sequence (x y : ℚ) (h1 : y ≠ 0) :
  (fifth_term x y - (-((x : ℚ) / 6) - 12)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_fifth_term_arithmetic_sequence_l1255_125508


namespace NUMINAMATH_GPT_sum_of_remainders_mod_8_l1255_125559

theorem sum_of_remainders_mod_8 
  (x y z w : ℕ)
  (hx : x % 8 = 3)
  (hy : y % 8 = 5)
  (hz : z % 8 = 7)
  (hw : w % 8 = 1) :
  (x + y + z + w) % 8 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_mod_8_l1255_125559


namespace NUMINAMATH_GPT_fraction_subtraction_simplest_form_l1255_125576

theorem fraction_subtraction_simplest_form :
  (8 / 24 - 5 / 40 = 5 / 24) :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_simplest_form_l1255_125576


namespace NUMINAMATH_GPT_part_a_part_b_l1255_125539

def triangle := Type
def point := Type

structure TriangleInCircle (ABC : triangle) where
  A : point
  B : point
  C : point
  A1 : point
  B1 : point
  C1 : point
  M : point
  r : Real
  R : Real

theorem part_a (ABC : triangle) (t : TriangleInCircle ABC) :
  ∃ MA MC MB_1, (MA * MC) / MB_1 = 2 * t.r := sorry
  
theorem part_b (ABC : triangle) (t : TriangleInCircle ABC) :
  ∃ MA_1 MC_1 MB, ( (MA_1 * MC_1) / MB) = t.R := sorry

end NUMINAMATH_GPT_part_a_part_b_l1255_125539


namespace NUMINAMATH_GPT_sandrine_washed_160_dishes_l1255_125591

-- Define the number of pears picked by Charles
def charlesPears : ℕ := 50

-- Define the number of bananas cooked by Charles as 3 times the number of pears he picked
def charlesBananas : ℕ := 3 * charlesPears

-- Define the number of dishes washed by Sandrine as 10 more than the number of bananas Charles cooked
def sandrineDishes : ℕ := charlesBananas + 10

-- Prove that Sandrine washed 160 dishes
theorem sandrine_washed_160_dishes : sandrineDishes = 160 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_sandrine_washed_160_dishes_l1255_125591


namespace NUMINAMATH_GPT_intersection_range_l1255_125532

noncomputable def function_f (x: ℝ) : ℝ := abs (x^2 - 4 * x + 3)

theorem intersection_range (b : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ function_f x1 = b ∧ function_f x2 = b ∧ function_f x3 = b) ↔ (0 < b ∧ b ≤ 1) := 
sorry

end NUMINAMATH_GPT_intersection_range_l1255_125532


namespace NUMINAMATH_GPT_original_polygon_sides_l1255_125562

theorem original_polygon_sides {n : ℕ} 
  (h : (n - 2) * 180 = 1620) : n = 10 ∨ n = 11 ∨ n = 12 :=
sorry

end NUMINAMATH_GPT_original_polygon_sides_l1255_125562


namespace NUMINAMATH_GPT_reporters_cover_local_politics_l1255_125566

structure Reporters :=
(total : ℕ)
(politics : ℕ)
(local_politics : ℕ)

def percentages (reporters : Reporters) : Prop :=
  reporters.politics = (40 * reporters.total) / 100 ∧
  reporters.local_politics = (75 * reporters.politics) / 100

theorem reporters_cover_local_politics (reporters : Reporters) (h : percentages reporters) :
  (reporters.local_politics * 100) / reporters.total = 30 :=
by
  -- Proof steps would be added here
  sorry

end NUMINAMATH_GPT_reporters_cover_local_politics_l1255_125566


namespace NUMINAMATH_GPT_min_moves_queens_switch_places_l1255_125589

-- Assume a type representing the board positions
inductive Position where
| first_rank | last_rank 

-- Assume a type representing the queens
inductive Queen where
| black | white

-- Function to count minimum moves for switching places
def min_moves_to_switch_places : ℕ :=
  sorry

theorem min_moves_queens_switch_places :
  min_moves_to_switch_places = 23 :=
  sorry

end NUMINAMATH_GPT_min_moves_queens_switch_places_l1255_125589


namespace NUMINAMATH_GPT_f_is_odd_l1255_125577

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def f (x : ℝ) : ℝ := x * |x|

theorem f_is_odd : is_odd_function f :=
by sorry

end NUMINAMATH_GPT_f_is_odd_l1255_125577


namespace NUMINAMATH_GPT_no_fixed_points_implies_no_double_fixed_points_l1255_125505

theorem no_fixed_points_implies_no_double_fixed_points (f : ℝ → ℝ) (hf : ∀ x, f x ≠ x) :
  ∀ x, f (f x) ≠ x :=
sorry

end NUMINAMATH_GPT_no_fixed_points_implies_no_double_fixed_points_l1255_125505


namespace NUMINAMATH_GPT_diane_harvest_increase_l1255_125533

-- Define the conditions
def last_year_harvest : ℕ := 2479
def this_year_harvest : ℕ := 8564

-- Definition of the increase in honey harvest
def increase_in_harvest : ℕ := this_year_harvest - last_year_harvest

-- The theorem statement we need to prove
theorem diane_harvest_increase : increase_in_harvest = 6085 := 
by
  -- skip the proof for now
  sorry

end NUMINAMATH_GPT_diane_harvest_increase_l1255_125533


namespace NUMINAMATH_GPT_circle_tangent_line_k_range_l1255_125506

theorem circle_tangent_line_k_range
  (k : ℝ)
  (P Q : ℝ × ℝ)
  (c : ℝ × ℝ := (0, 1)) -- Circle center
  (r : ℝ := 1) -- Circle radius
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 2 * y = 0)
  (line_eq : ∀ (x y : ℝ), k * x + y + 3 = 0)
  (dist_pq : Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) = Real.sqrt 3) :
  k ∈ Set.Iic (-Real.sqrt 3) ∪ Set.Ici (Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_line_k_range_l1255_125506


namespace NUMINAMATH_GPT_angle_between_hands_at_3_15_l1255_125512

-- Definitions based on conditions
def minuteHandAngleAt_3_15 : ℝ := 90 -- The position of the minute hand at 3:15 is 90 degrees.

def hourHandSpeed : ℝ := 0.5 -- The hour hand moves at 0.5 degrees per minute.

def hourHandAngleAt_3_15 : ℝ := 3 * 30 + 15 * hourHandSpeed
-- The hour hand starts at 3 o'clock (90 degrees) and moves 0.5 degrees per minute.

-- Statement to prove
theorem angle_between_hands_at_3_15 : abs (minuteHandAngleAt_3_15 - hourHandAngleAt_3_15) = 82.5 :=
by
  sorry

end NUMINAMATH_GPT_angle_between_hands_at_3_15_l1255_125512


namespace NUMINAMATH_GPT_domain_of_sqrt_log_l1255_125593

theorem domain_of_sqrt_log {x : ℝ} : (2 < x ∧ x ≤ 5 / 2) ↔ 
  (5 - 2 * x > 0 ∧ 0 ≤ Real.logb (1 / 2) (5 - 2 * x)) :=
sorry

end NUMINAMATH_GPT_domain_of_sqrt_log_l1255_125593


namespace NUMINAMATH_GPT_volume_ratio_sphere_cylinder_inscribed_l1255_125527

noncomputable def ratio_of_volumes (d : ℝ) : ℝ :=
  let Vs := (4 / 3) * Real.pi * (d / 2)^3
  let Vc := Real.pi * (d / 2)^2 * d
  Vs / Vc

theorem volume_ratio_sphere_cylinder_inscribed (d : ℝ) (h : d > 0) : 
  ratio_of_volumes d = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_volume_ratio_sphere_cylinder_inscribed_l1255_125527


namespace NUMINAMATH_GPT_cylinder_area_ratio_l1255_125572

noncomputable def ratio_of_areas (r h : ℝ) (h_cond : 2 * r / h = h / (2 * Real.pi * r)) : ℝ :=
  let lateral_area := 2 * Real.pi * r * h
  let total_area := lateral_area + 2 * Real.pi * r * r
  lateral_area / total_area

theorem cylinder_area_ratio {r h : ℝ} (h_cond : 2 * r / h = h / (2 * Real.pi * r)) :
  ratio_of_areas r h h_cond = 2 * Real.sqrt Real.pi / (2 * Real.sqrt Real.pi + 1) := 
sorry

end NUMINAMATH_GPT_cylinder_area_ratio_l1255_125572


namespace NUMINAMATH_GPT_largest_value_l1255_125580

noncomputable def a : ℕ := 2 ^ 6
noncomputable def b : ℕ := 3 ^ 5
noncomputable def c : ℕ := 4 ^ 4
noncomputable def d : ℕ := 5 ^ 3
noncomputable def e : ℕ := 6 ^ 2

theorem largest_value : c > a ∧ c > b ∧ c > d ∧ c > e := by
  sorry

end NUMINAMATH_GPT_largest_value_l1255_125580


namespace NUMINAMATH_GPT_locus_equation_of_points_at_distance_2_from_line_l1255_125545

theorem locus_equation_of_points_at_distance_2_from_line :
  {P : ℝ × ℝ | abs ((3 / 5) * P.1 - (4 / 5) * P.2 - (1 / 5)) = 2} =
    {P : ℝ × ℝ | 3 * P.1 - 4 * P.2 - 11 = 0} ∪ {P : ℝ × ℝ | 3 * P.1 - 4 * P.2 + 9 = 0} :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_locus_equation_of_points_at_distance_2_from_line_l1255_125545


namespace NUMINAMATH_GPT_find_abc_l1255_125514

-- Given conditions: a, b, c are positive real numbers and satisfy the given equations.
variables (a b c : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_pos_c : 0 < c)
variable (h1 : a * (b + c) = 152)
variable (h2 : b * (c + a) = 162)
variable (h3 : c * (a + b) = 170)

theorem find_abc : a * b * c = 720 := 
  sorry

end NUMINAMATH_GPT_find_abc_l1255_125514


namespace NUMINAMATH_GPT_man_l1255_125552

noncomputable def man's_rate_in_still_water (downstream upstream : ℝ) : ℝ :=
  (downstream + upstream) / 2

theorem man's_rate_correct :
  let downstream := 6
  let upstream := 3
  man's_rate_in_still_water downstream upstream = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_man_l1255_125552


namespace NUMINAMATH_GPT_price_per_foot_l1255_125554

theorem price_per_foot (area : ℝ) (cost : ℝ) (side_length : ℝ) (perimeter : ℝ) 
  (h1 : area = 289) (h2 : cost = 3740) 
  (h3 : side_length^2 = area) (h4 : perimeter = 4 * side_length) : 
  (cost / perimeter = 55) :=
by
  sorry

end NUMINAMATH_GPT_price_per_foot_l1255_125554


namespace NUMINAMATH_GPT_system_of_equations_solution_l1255_125557

theorem system_of_equations_solution 
  (x y z : ℤ) 
  (h1 : x^2 - y - z = 8) 
  (h2 : 4 * x + y^2 + 3 * z = -11) 
  (h3 : 2 * x - 3 * y + z^2 = -11) : 
  x = -3 ∧ y = 2 ∧ z = -1 :=
sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1255_125557


namespace NUMINAMATH_GPT_Misha_earnings_needed_l1255_125597

-- Define the conditions and the goal in Lean 4
def Misha_current_dollars : ℕ := 34
def Misha_target_dollars : ℕ := 47

theorem Misha_earnings_needed : Misha_target_dollars - Misha_current_dollars = 13 := by
  sorry

end NUMINAMATH_GPT_Misha_earnings_needed_l1255_125597


namespace NUMINAMATH_GPT_points_total_l1255_125518

/--
In a game, Samanta has 8 more points than Mark,
and Mark has 50% more points than Eric. Eric has 6 points.
How many points do Samanta, Mark, and Eric have in total?
-/
theorem points_total (Samanta Mark Eric : ℕ)
  (h1 : Samanta = Mark + 8)
  (h2 : Mark = Eric + Eric / 2)
  (h3 : Eric = 6) :
  Samanta + Mark + Eric = 32 := by
  sorry

end NUMINAMATH_GPT_points_total_l1255_125518


namespace NUMINAMATH_GPT_smallest_value_of_f4_l1255_125558

def f (x : ℝ) : ℝ := (x + 3) ^ 2 - 2

theorem smallest_value_of_f4 : ∀ x : ℝ, f (f (f (f x))) ≥ 23 :=
by 
  sorry -- Proof goes here.

end NUMINAMATH_GPT_smallest_value_of_f4_l1255_125558


namespace NUMINAMATH_GPT_solve_equation_l1255_125553

theorem solve_equation (x : ℝ) (h : (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2) : 
  x = -2/3 :=
sorry

end NUMINAMATH_GPT_solve_equation_l1255_125553


namespace NUMINAMATH_GPT_base_rate_of_second_company_l1255_125573

-- Define the conditions
def United_base_rate : ℝ := 8.00
def United_rate_per_minute : ℝ := 0.25
def Other_rate_per_minute : ℝ := 0.20
def minutes : ℕ := 80

-- Define the total bill equations
def United_total_bill (minutes : ℕ) : ℝ := United_base_rate + United_rate_per_minute * minutes
def Other_total_bill (minutes : ℕ) (B : ℝ) : ℝ := B + Other_rate_per_minute * minutes

-- Define the claim to prove
theorem base_rate_of_second_company : ∃ B : ℝ, Other_total_bill minutes B = United_total_bill minutes ∧ B = 12.00 := by
  sorry

end NUMINAMATH_GPT_base_rate_of_second_company_l1255_125573


namespace NUMINAMATH_GPT_number_of_correct_conclusions_l1255_125510

-- Given conditions
variables {a b c : ℝ} (h₀ : a ≠ 0) (h₁ : c > 3)
           (h₂ : a * 25 + b * 5 + c = 0)
           (h₃ : -b / (2 * a) = 2)
           (h₄ : a < 0)

-- Proof should show:
theorem number_of_correct_conclusions 
  (h₀ : a ≠ 0)
  (h₁ : c > 3)
  (h₂ : 25 * a + 5 * b + c = 0)
  (h₃ : - b / (2 * a) = 2)
  (h₄ : a < 0) :
  (a * b * c < 0) ∧ 
  (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂) ∧ (a * x₁^2 + b * x₁ + c = 2) ∧ (a * x₂^2 + b * x₂ + c = 2)) ∧ 
  (a < -3 / 5) := 
by
  sorry

end NUMINAMATH_GPT_number_of_correct_conclusions_l1255_125510


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1255_125564

noncomputable def repeating_decimal_0_3 : ℚ := 1 / 3
noncomputable def repeating_decimal_0_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_0_2 : ℚ := 2 / 9

theorem repeating_decimal_sum :
  repeating_decimal_0_3 + repeating_decimal_0_6 - repeating_decimal_0_2 = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l1255_125564


namespace NUMINAMATH_GPT_successive_increases_eq_single_l1255_125524

variable (P : ℝ)

def increase_by (initial : ℝ) (pct : ℝ) : ℝ := initial * (1 + pct)
def discount_by (initial : ℝ) (pct : ℝ) : ℝ := initial * (1 - pct)

theorem successive_increases_eq_single (P : ℝ) :
  increase_by (increase_by (discount_by (increase_by P 0.30) 0.10) 0.15) 0.20 = increase_by P 0.6146 :=
  sorry

end NUMINAMATH_GPT_successive_increases_eq_single_l1255_125524


namespace NUMINAMATH_GPT_cubs_more_home_runs_l1255_125513

noncomputable def cubs_home_runs := 2 + 1 + 2
noncomputable def cardinals_home_runs := 1 + 1

theorem cubs_more_home_runs :
  cubs_home_runs - cardinals_home_runs = 3 :=
by
  -- Proof would go here, but we are using sorry to skip it
  sorry

end NUMINAMATH_GPT_cubs_more_home_runs_l1255_125513


namespace NUMINAMATH_GPT_green_flower_percentage_l1255_125569

theorem green_flower_percentage (yellow purple green total : ℕ)
  (hy : yellow = 10)
  (hp : purple = 18)
  (ht : total = 35)
  (hgreen : green = total - (yellow + purple)) :
  ((green * 100) / (yellow + purple)) = 25 := 
by {
  sorry
}

end NUMINAMATH_GPT_green_flower_percentage_l1255_125569


namespace NUMINAMATH_GPT_eggs_left_in_box_l1255_125595

theorem eggs_left_in_box (initial_eggs : ℕ) (taken_eggs : ℕ) (remaining_eggs : ℕ) : 
  initial_eggs = 47 → taken_eggs = 5 → remaining_eggs = initial_eggs - taken_eggs → remaining_eggs = 42 :=
by
  sorry

end NUMINAMATH_GPT_eggs_left_in_box_l1255_125595


namespace NUMINAMATH_GPT_total_legs_in_farm_l1255_125504

theorem total_legs_in_farm (total_animals : ℕ) (total_cows : ℕ) (cow_legs : ℕ) (duck_legs : ℕ) 
  (h_total_animals : total_animals = 15) (h_total_cows : total_cows = 6) 
  (h_cow_legs : cow_legs = 4) (h_duck_legs : duck_legs = 2) :
  total_cows * cow_legs + (total_animals - total_cows) * duck_legs = 42 :=
by
  sorry

end NUMINAMATH_GPT_total_legs_in_farm_l1255_125504


namespace NUMINAMATH_GPT_even_function_expression_l1255_125501

theorem even_function_expression (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = x * (2 * x - 1)) :
  ∀ x, x > 0 → f x = x * (2 * x + 1) :=
by 
  sorry

end NUMINAMATH_GPT_even_function_expression_l1255_125501


namespace NUMINAMATH_GPT_school_fitness_event_participants_l1255_125598

theorem school_fitness_event_participants :
  let p0 := 500 -- initial number of participants in 2000
  let r1 := 0.3 -- increase rate in 2001
  let r2 := 0.4 -- increase rate in 2002
  let r3 := 0.5 -- increase rate in 2003
  let p1 := p0 * (1 + r1) -- participants in 2001
  let p2 := p1 * (1 + r2) -- participants in 2002
  let p3 := p2 * (1 + r3) -- participants in 2003
  p3 = 1365 -- prove that number of participants in 2003 is 1365
:= sorry

end NUMINAMATH_GPT_school_fitness_event_participants_l1255_125598


namespace NUMINAMATH_GPT_quadratic_complete_square_l1255_125519

theorem quadratic_complete_square :
  ∃ a b c : ℝ, (∀ x : ℝ, 4 * x^2 - 40 * x + 100 = a * (x + b)^2 + c) ∧ a + b + c = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_complete_square_l1255_125519


namespace NUMINAMATH_GPT_ellipse_semimajor_axis_value_l1255_125555

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_GPT_ellipse_semimajor_axis_value_l1255_125555


namespace NUMINAMATH_GPT_isosceles_triangle_base_function_l1255_125581

theorem isosceles_triangle_base_function (x : ℝ) (hx : 5 < x ∧ x < 10) :
  ∃ y : ℝ, y = 20 - 2 * x := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_function_l1255_125581


namespace NUMINAMATH_GPT_inequality_example_l1255_125570

theorem inequality_example (a b c : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (sum_eq_one : a + b + c = 1) :
  (a + 1 / a) * (b + 1 / b) * (c + 1 / c) ≥ 1000 / 27 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_example_l1255_125570


namespace NUMINAMATH_GPT_volume_of_prism_l1255_125542

   theorem volume_of_prism (a b c : ℝ)
     (h1 : a * b = 18) (h2 : b * c = 12) (h3 : a * c = 8) :
     a * b * c = 24 * Real.sqrt 3 :=
   sorry
   
end NUMINAMATH_GPT_volume_of_prism_l1255_125542


namespace NUMINAMATH_GPT_meetings_percent_40_l1255_125529

def percent_of_workday_in_meetings (workday_hours : ℕ) (first_meeting_min : ℕ) (second_meeting_min : ℕ) (third_meeting_min : ℕ) : ℕ :=
  (first_meeting_min + second_meeting_min + third_meeting_min) * 100 / (workday_hours * 60)

theorem meetings_percent_40 (workday_hours : ℕ) (first_meeting_min : ℕ) (second_meeting_min : ℕ) (third_meeting_min : ℕ)
  (h_workday : workday_hours = 10) 
  (h_first_meeting : first_meeting_min = 40) 
  (h_second_meeting : second_meeting_min = 2 * first_meeting_min) 
  (h_third_meeting : third_meeting_min = first_meeting_min + second_meeting_min) : 
  percent_of_workday_in_meetings workday_hours first_meeting_min second_meeting_min third_meeting_min = 40 :=
by
  sorry

end NUMINAMATH_GPT_meetings_percent_40_l1255_125529


namespace NUMINAMATH_GPT_english_teachers_count_l1255_125520

theorem english_teachers_count (E : ℕ) 
    (h_prob : 6 / ((E + 6) * (E + 5) / 2) = 1 / 12) : 
    E = 3 :=
by
  sorry

end NUMINAMATH_GPT_english_teachers_count_l1255_125520


namespace NUMINAMATH_GPT_relationship_x_x2_negx_l1255_125500

theorem relationship_x_x2_negx (x : ℝ) (h : x^2 + x < 0) : x < x^2 ∧ x^2 < -x :=
by
  sorry

end NUMINAMATH_GPT_relationship_x_x2_negx_l1255_125500
