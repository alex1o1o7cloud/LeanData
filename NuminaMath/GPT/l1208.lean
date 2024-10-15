import Mathlib

namespace NUMINAMATH_GPT_solution_set_of_inequality_l1208_120881

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x - 14 < 0} = {x : ℝ | -2 < x ∧ x < 7} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1208_120881


namespace NUMINAMATH_GPT_kelly_baking_powder_l1208_120897

variable (current_supply : ℝ) (additional_supply : ℝ)

theorem kelly_baking_powder (h1 : current_supply = 0.3)
                            (h2 : additional_supply = 0.1) :
                            current_supply + additional_supply = 0.4 := 
by
  sorry

end NUMINAMATH_GPT_kelly_baking_powder_l1208_120897


namespace NUMINAMATH_GPT_sum_of_coordinates_is_17_over_3_l1208_120872

theorem sum_of_coordinates_is_17_over_3
  (f : ℝ → ℝ)
  (h1 : 5 = 3 * f 2) :
  (5 / 3 + 4) = 17 / 3 :=
by
  have h2 : f 2 = 5 / 3 := by
    linarith
  have h3 : f⁻¹ (5 / 3) = 2 := by
    sorry -- we do not know more properties of f to conclude this proof step
  have h4 : 2 * f⁻¹ (5 / 3) = 4 := by
    sorry -- similarly, assume for now the desired property
  exact sorry -- finally putting everything together

end NUMINAMATH_GPT_sum_of_coordinates_is_17_over_3_l1208_120872


namespace NUMINAMATH_GPT_paco_initial_sweet_cookies_l1208_120861

theorem paco_initial_sweet_cookies (S : ℕ) (h1 : S - 15 = 7) : S = 22 :=
by
  sorry

end NUMINAMATH_GPT_paco_initial_sweet_cookies_l1208_120861


namespace NUMINAMATH_GPT_ratio_of_perimeters_is_one_l1208_120876

-- Definitions based on the given conditions
def original_rectangle : ℝ × ℝ := (6, 8)
def folded_rectangle : ℝ × ℝ := (3, 8)
def small_rectangle : ℝ × ℝ := (3, 4)
def large_rectangle : ℝ × ℝ := (3, 4)

-- The perimeter function for a rectangle given its dimensions (length, width)
def perimeter (r : ℝ × ℝ) : ℝ := 2 * (r.1 + r.2)

-- The main theorem to prove
theorem ratio_of_perimeters_is_one : 
  perimeter small_rectangle / perimeter large_rectangle = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_perimeters_is_one_l1208_120876


namespace NUMINAMATH_GPT_ducks_drinking_l1208_120870

theorem ducks_drinking (total_d : ℕ) (drank_before : ℕ) (drank_after : ℕ) :
  total_d = 20 → drank_before = 11 → drank_after = total_d - (drank_before + 1) → drank_after = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_ducks_drinking_l1208_120870


namespace NUMINAMATH_GPT_greatest_three_digit_divisible_by_3_6_5_l1208_120851

/-- Define a three-digit number and conditions for divisibility by 3, 6, and 5 -/
def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def is_divisible_by (n : ℕ) (d : ℕ) : Prop := d ∣ n

/-- Greatest three-digit number divisible by 3, 6, and 5 is 990 -/
theorem greatest_three_digit_divisible_by_3_6_5 : ∃ n : ℕ, is_three_digit n ∧ is_divisible_by n 3 ∧ is_divisible_by n 6 ∧ is_divisible_by n 5 ∧ n = 990 :=
sorry

end NUMINAMATH_GPT_greatest_three_digit_divisible_by_3_6_5_l1208_120851


namespace NUMINAMATH_GPT_solve_equation_l1208_120871

theorem solve_equation (x : ℝ) : 
  (x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 ∨ x = -3 + Real.sqrt 6 ∨ x = -3 - Real.sqrt 6) ↔ 
  (x^4 / (2 * x + 1) + x^2 = 6 * (2 * x + 1)) := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1208_120871


namespace NUMINAMATH_GPT_total_food_amount_l1208_120829

-- Define constants for the given problem
def chicken : ℕ := 16
def hamburgers : ℕ := chicken / 2
def hot_dogs : ℕ := hamburgers + 2
def sides : ℕ := hot_dogs / 2

-- Prove the total amount of food Peter will buy is 39 pounds
theorem total_food_amount : chicken + hamburgers + hot_dogs + sides = 39 := by
  sorry

end NUMINAMATH_GPT_total_food_amount_l1208_120829


namespace NUMINAMATH_GPT_coat_price_reduction_l1208_120898

theorem coat_price_reduction (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500) (h2 : reduction_amount = 400) :
  (reduction_amount / original_price) * 100 = 80 :=
by {
  sorry -- This is where the proof would go
}

end NUMINAMATH_GPT_coat_price_reduction_l1208_120898


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1208_120802

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 5) (h2 : |b| = 2) (h3 : a < b) : a + b = -3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1208_120802


namespace NUMINAMATH_GPT_cos_probability_ge_one_half_in_range_l1208_120874

theorem cos_probability_ge_one_half_in_range :
  let interval_length := (Real.pi / 2) - (- (Real.pi / 2))
  let favorable_length := (Real.pi / 3) - (- (Real.pi / 3))
  (favorable_length / interval_length) = (2 / 3) := by
  sorry

end NUMINAMATH_GPT_cos_probability_ge_one_half_in_range_l1208_120874


namespace NUMINAMATH_GPT_max_min_diff_w_l1208_120803

theorem max_min_diff_w (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 4) :
  let w := a^2 + a*b + b^2
  let w1 := max (0^2 + 0*b + b^2) (4^2 + 4*b + b^2)
  let w2 := (2-2)^2 + 12
  w1 - w2 = 4 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_max_min_diff_w_l1208_120803


namespace NUMINAMATH_GPT_intersection_M_N_l1208_120890

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_M_N_l1208_120890


namespace NUMINAMATH_GPT_least_integer_value_l1208_120855

theorem least_integer_value (x : ℤ) : 3 * abs x + 4 < 19 → x = -4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_least_integer_value_l1208_120855


namespace NUMINAMATH_GPT_part_I_part_II_l1208_120819

noncomputable def seq_a : ℕ → ℝ 
| 0       => 1   -- Normally, we start with n = 1, so we set a_0 to some default value.
| (n+1)   => (1 + 1 / (n^2 + n)) * seq_a n + 1 / (2^n)

theorem part_I (n : ℕ) (h: n ≥ 2) : seq_a n ≥ 2 :=
sorry

theorem part_II (n : ℕ) : seq_a n < Real.exp 2 :=
sorry

-- Assumption: ln(1 + x) < x for all x > 0
axiom ln_ineq (x : ℝ) (hx : 0 < x) : Real.log (1 + x) < x

end NUMINAMATH_GPT_part_I_part_II_l1208_120819


namespace NUMINAMATH_GPT_cylinder_radius_inscribed_box_l1208_120820

theorem cylinder_radius_inscribed_box :
  ∀ (x y z r : ℝ),
    4 * (x + y + z) = 160 →
    2 * (x * y + y * z + x * z) = 600 →
    z = 40 - x - y →
    r = (1/2) * Real.sqrt (x^2 + y^2) →
    r = (15 * Real.sqrt 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_radius_inscribed_box_l1208_120820


namespace NUMINAMATH_GPT_exists_xy_l1208_120817

-- Given conditions from the problem
variables (m x0 y0 : ℕ)
-- Integers x0 and y0 are relatively prime
variables (rel_prim : Nat.gcd x0 y0 = 1)
-- y0 divides x0^2 + m
variables (div_y0 : y0 ∣ x0^2 + m)
-- x0 divides y0^2 + m
variables (div_x0 : x0 ∣ y0^2 + m)

-- Main theorem statement
theorem exists_xy 
  (hm : m > 0) 
  (hx0 : x0 > 0) 
  (hy0 : y0 > 0) 
  (rel_prim : Nat.gcd x0 y0 = 1) 
  (div_y0 : y0 ∣ x0^2 + m) 
  (div_x0 : x0 ∣ y0^2 + m) : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ Nat.gcd x y = 1 ∧ y ∣ x^2 + m ∧ x ∣ y^2 + m ∧ x + y ≤ m + 1 := 
sorry

end NUMINAMATH_GPT_exists_xy_l1208_120817


namespace NUMINAMATH_GPT_contractor_absent_days_l1208_120816

variable (x y : ℕ)  -- Number of days worked and absent, both are natural numbers

-- Conditions from the problem
def total_days (x y : ℕ) : Prop := x + y = 30
def total_payment (x y : ℕ) : Prop := 25 * x - 75 * y / 10 = 360

-- Main statement
theorem contractor_absent_days (h1 : total_days x y) (h2 : total_payment x y) : y = 12 :=
by
  sorry

end NUMINAMATH_GPT_contractor_absent_days_l1208_120816


namespace NUMINAMATH_GPT_red_other_side_probability_is_one_l1208_120887

/-- Definitions from the problem conditions --/
def total_cards : ℕ := 10
def green_both_sides : ℕ := 5
def green_red_sides : ℕ := 2
def red_both_sides : ℕ := 3
def red_faces : ℕ := 6 -- 3 cards × 2 sides each

/-- The theorem proves the probability is 1 that the other side is red given that one side seen is red --/
theorem red_other_side_probability_is_one
  (h_total_cards : total_cards = 10)
  (h_green_both : green_both_sides = 5)
  (h_green_red : green_red_sides = 2)
  (h_red_both : red_both_sides = 3)
  (h_red_faces : red_faces = 6) :
  1 = (red_faces / red_faces) :=
by
  -- Write the proof steps here
  sorry

end NUMINAMATH_GPT_red_other_side_probability_is_one_l1208_120887


namespace NUMINAMATH_GPT_ratio_of_vanilla_chips_l1208_120889

-- Definitions from the conditions
variable (V_c S_c V_v S_v : ℕ)
variable (H1 : V_c = S_c + 5)
variable (H2 : S_c = 25)
variable (H3 : V_v = 20)
variable (H4 : V_c + S_c + V_v + S_v = 90)

-- The statement we want to prove
theorem ratio_of_vanilla_chips : S_v / V_v = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_ratio_of_vanilla_chips_l1208_120889


namespace NUMINAMATH_GPT_largest_of_four_numbers_l1208_120885

variables {x y z w : ℕ}

theorem largest_of_four_numbers
  (h1 : x + y + z = 180)
  (h2 : x + y + w = 197)
  (h3 : x + z + w = 208)
  (h4 : y + z + w = 222) :
  max x (max y (max z w)) = 89 :=
sorry

end NUMINAMATH_GPT_largest_of_four_numbers_l1208_120885


namespace NUMINAMATH_GPT_triangular_faces_area_of_pyramid_l1208_120867

noncomputable def total_area_of_triangular_faces (base : ℝ) (lateral : ℝ) : ℝ :=
  let h := Real.sqrt (lateral ^ 2 - (base / 2) ^ 2)
  let area_one_triangle := (1 / 2) * base * h
  4 * area_one_triangle

theorem triangular_faces_area_of_pyramid :
  total_area_of_triangular_faces 8 10 = 32 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_GPT_triangular_faces_area_of_pyramid_l1208_120867


namespace NUMINAMATH_GPT_domain_of_sqrt_sum_l1208_120892

theorem domain_of_sqrt_sum (x : ℝ) : (1 ≤ x ∧ x ≤ 3) ↔ (x - 1 ≥ 0 ∧ 3 - x ≥ 0) := by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_sum_l1208_120892


namespace NUMINAMATH_GPT_find_k_l1208_120853

theorem find_k (k : ℚ) :
  (∃ (x y : ℚ), y = 4 * x + 5 ∧ y = -3 * x + 10 ∧ y = 2 * x + k) →
  k = 45 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1208_120853


namespace NUMINAMATH_GPT_least_n_prime_condition_l1208_120852

theorem least_n_prime_condition : ∃ n : ℕ, (∀ p : ℕ, Prime p → ¬ Prime (p^2 + n)) ∧ (∀ m : ℕ, 
 (m > 0 ∧ ∀ p : ℕ, Prime p → ¬ Prime (p^2 + m)) → m ≥ 5) ∧ n = 5 := by
  sorry

end NUMINAMATH_GPT_least_n_prime_condition_l1208_120852


namespace NUMINAMATH_GPT_initial_walking_rate_proof_l1208_120818

noncomputable def initial_walking_rate (d : ℝ) (v_miss : ℝ) (t_miss : ℝ) (v_early : ℝ) (t_early : ℝ) : ℝ :=
  d / ((d / v_early) + t_early - t_miss)

theorem initial_walking_rate_proof :
  initial_walking_rate 6 5 (7/60) 6 (5/60) = 5 := by
  sorry

end NUMINAMATH_GPT_initial_walking_rate_proof_l1208_120818


namespace NUMINAMATH_GPT_max_min_of_f_on_interval_l1208_120808

-- Conditions
def f (x : ℝ) : ℝ := x^3 - 3 * x + 1
def interval : Set ℝ := Set.Icc (-3) 0

-- Problem statement
theorem max_min_of_f_on_interval : 
  ∃ (max min : ℝ), max = 1 ∧ min = -17 ∧ 
  (∀ x ∈ interval, f x ≤ max) ∧ 
  (∀ x ∈ interval, f x ≥ min) := 
sorry

end NUMINAMATH_GPT_max_min_of_f_on_interval_l1208_120808


namespace NUMINAMATH_GPT_union_of_P_and_Q_l1208_120830

def P : Set ℝ := { x | |x| ≥ 3 }
def Q : Set ℝ := { y | ∃ x, y = 2^x - 1 }

theorem union_of_P_and_Q : P ∪ Q = { y | y ≤ -3 ∨ y > -1 } := by
  sorry

end NUMINAMATH_GPT_union_of_P_and_Q_l1208_120830


namespace NUMINAMATH_GPT_positive_expression_l1208_120837

theorem positive_expression (x y : ℝ) : (x^2 - 4 * x + y^2 + 13) > 0 := by
  sorry

end NUMINAMATH_GPT_positive_expression_l1208_120837


namespace NUMINAMATH_GPT_polygon_sides_with_diagonals_44_l1208_120841

theorem polygon_sides_with_diagonals_44 (n : ℕ) (hD : 44 = n * (n - 3) / 2) : n = 11 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_with_diagonals_44_l1208_120841


namespace NUMINAMATH_GPT_least_values_3198_l1208_120847

theorem least_values_3198 (x y : ℕ) (hX : ∃ n : ℕ, 3198 + n * 9 = 27)
                         (hY : ∃ m : ℕ, 3198 + m * 11 = 11) :
  x = 6 ∧ y = 8 :=
by
  sorry

end NUMINAMATH_GPT_least_values_3198_l1208_120847


namespace NUMINAMATH_GPT_find_number_l1208_120864

theorem find_number (x : ℤ) :
  45 - (x - (37 - (15 - 18))) = 57 → x = 28 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1208_120864


namespace NUMINAMATH_GPT_Q_proper_subset_P_l1208_120869

open Set

def P : Set ℝ := { x | x ≥ 1 }
def Q : Set ℝ := { 2, 3 }

theorem Q_proper_subset_P : Q ⊂ P :=
by
  sorry

end NUMINAMATH_GPT_Q_proper_subset_P_l1208_120869


namespace NUMINAMATH_GPT_speed_conversion_l1208_120821

theorem speed_conversion (v : ℚ) (h : v = 9/36) : v * 3.6 = 0.9 := by
  sorry

end NUMINAMATH_GPT_speed_conversion_l1208_120821


namespace NUMINAMATH_GPT_part1_solve_inequality_part2_range_of_a_l1208_120863

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + a)

theorem part1_solve_inequality (x : ℝ) (h : -2 < x ∧ x < -2/3) :
    f x 1 > 1 :=
by
  sorry

theorem part2_range_of_a (h : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x (a : ℝ) > 0) :
    -5/2 < a ∧ a < -2 :=
by
  sorry

end NUMINAMATH_GPT_part1_solve_inequality_part2_range_of_a_l1208_120863


namespace NUMINAMATH_GPT_base7_to_base10_conversion_l1208_120849

def convert_base_7_to_10 := 243

namespace Base7toBase10

theorem base7_to_base10_conversion :
  2 * 7^2 + 4 * 7^1 + 3 * 7^0 = 129 := by
  -- The original number 243 in base 7 is expanded and evaluated to base 10.
  sorry

end Base7toBase10

end NUMINAMATH_GPT_base7_to_base10_conversion_l1208_120849


namespace NUMINAMATH_GPT_max_ants_collisions_l1208_120811

theorem max_ants_collisions (n : ℕ) (hpos : 0 < n) :
  ∃ (ants : Fin n → ℝ) (speeds: Fin n → ℝ) (finite_collisions : Prop)
    (collisions_bound : ℕ),
  (∀ i : Fin n, speeds i ≠ 0) →
  finite_collisions →
  collisions_bound = (n * (n - 1)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_max_ants_collisions_l1208_120811


namespace NUMINAMATH_GPT_numValidRoutesJackToJill_l1208_120865

noncomputable def numPaths (n m : ℕ) : ℕ :=
  Nat.choose (n + m) n

theorem numValidRoutesJackToJill : 
  let totalRoutes := numPaths 5 3
  let pathsViaDanger := numPaths 2 2 * numPaths 3 1
  totalRoutes - pathsViaDanger = 32 :=
by
  let totalRoutes := numPaths 5 3
  let pathsViaDanger := numPaths 2 2 * numPaths 3 1
  show totalRoutes - pathsViaDanger = 32
  sorry

end NUMINAMATH_GPT_numValidRoutesJackToJill_l1208_120865


namespace NUMINAMATH_GPT_correct_calculation_l1208_120893

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l1208_120893


namespace NUMINAMATH_GPT_pet_store_animals_left_l1208_120833

def initial_birds : Nat := 12
def initial_puppies : Nat := 9
def initial_cats : Nat := 5
def initial_spiders : Nat := 15

def birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_loose : Nat := 7

def birds_left : Nat := initial_birds - birds_sold
def puppies_left : Nat := initial_puppies - puppies_adopted
def cats_left : Nat := initial_cats
def spiders_left : Nat := initial_spiders - spiders_loose

def total_animals_left : Nat := birds_left + puppies_left + cats_left + spiders_left

theorem pet_store_animals_left : total_animals_left = 25 :=
by
  sorry

end NUMINAMATH_GPT_pet_store_animals_left_l1208_120833


namespace NUMINAMATH_GPT_coloring_ways_of_circle_l1208_120823

noncomputable def num_ways_to_color_circle (n : ℕ) (k : ℕ) : ℕ :=
  if h : n % 2 = 1 then -- There are 13 parts; n must be odd (since adjacent matching impossible in even n)
    (k * (k - 1)^(n - 1) : ℕ)
  else
    0

theorem coloring_ways_of_circle :
  num_ways_to_color_circle 13 3 = 6 :=
by
  sorry

end NUMINAMATH_GPT_coloring_ways_of_circle_l1208_120823


namespace NUMINAMATH_GPT_bond_selling_price_l1208_120859

theorem bond_selling_price
    (face_value : ℝ)
    (interest_rate_face : ℝ)
    (interest_rate_selling : ℝ)
    (interest : ℝ)
    (selling_price : ℝ)
    (h1 : face_value = 5000)
    (h2 : interest_rate_face = 0.07)
    (h3 : interest_rate_selling = 0.065)
    (h4 : interest = face_value * interest_rate_face)
    (h5 : interest = selling_price * interest_rate_selling) :
  selling_price = 5384.62 :=
sorry

end NUMINAMATH_GPT_bond_selling_price_l1208_120859


namespace NUMINAMATH_GPT_total_pencils_l1208_120843

theorem total_pencils (reeta_pencils anika_pencils kamal_pencils : ℕ) :
  reeta_pencils = 30 →
  anika_pencils = 2 * reeta_pencils + 4 →
  kamal_pencils = 3 * reeta_pencils - 2 →
  reeta_pencils + anika_pencils + kamal_pencils = 182 :=
by
  intros h_reeta h_anika h_kamal
  sorry

end NUMINAMATH_GPT_total_pencils_l1208_120843


namespace NUMINAMATH_GPT_find_sum_of_a_and_d_l1208_120846

theorem find_sum_of_a_and_d 
  {a b c d : ℝ} 
  (h1 : ab + ac + bd + cd = 42) 
  (h2 : b + c = 6) : 
  a + d = 7 :=
sorry

end NUMINAMATH_GPT_find_sum_of_a_and_d_l1208_120846


namespace NUMINAMATH_GPT_crayons_left_l1208_120844

theorem crayons_left (initial_crayons : ℕ) (crayons_taken : ℕ) : initial_crayons = 7 → crayons_taken = 3 → initial_crayons - crayons_taken = 4 :=
by
  sorry

end NUMINAMATH_GPT_crayons_left_l1208_120844


namespace NUMINAMATH_GPT_max_value_func1_l1208_120883

theorem max_value_func1 (x : ℝ) (h : 0 < x ∧ x < 2) : 
  ∃ y, y = x * (4 - 2 * x) ∧ (∀ z, z = x * (4 - 2 * x) → z ≤ 2) :=
sorry

end NUMINAMATH_GPT_max_value_func1_l1208_120883


namespace NUMINAMATH_GPT_evaluate_expression_l1208_120812

theorem evaluate_expression : 
  (3^2015 + 3^2013 + 3^2012) / (3^2015 - 3^2013 + 3^2012) = 31 / 25 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1208_120812


namespace NUMINAMATH_GPT_train_length_l1208_120845

theorem train_length (v_train : ℝ) (v_man : ℝ) (t : ℝ) (length_train : ℝ)
  (h1 : v_train = 55) (h2 : v_man = 7) (h3 : t = 10.45077684107852) :
  length_train = 180 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1208_120845


namespace NUMINAMATH_GPT_arithmetic_series_product_l1208_120858

theorem arithmetic_series_product (a b c : ℝ) (h1 : a = b - d) (h2 : c = b + d) (h3 : a * b * c = 125) (h4 : 0 < a) (h5 : 0 < b) (h6 : 0 < c) : b ≥ 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_series_product_l1208_120858


namespace NUMINAMATH_GPT_B_pow_97_l1208_120886

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem B_pow_97 : B ^ 97 = B := by
  sorry

end NUMINAMATH_GPT_B_pow_97_l1208_120886


namespace NUMINAMATH_GPT_z_is_imaginary_z_is_purely_imaginary_iff_z_on_angle_bisector_iff_l1208_120877

open Complex

-- Problem definitions
def z (m : ℝ) : ℂ := (2 + I) * m^2 - 2 * (1 - I)

-- Prove that for all m in ℝ, z is imaginary
theorem z_is_imaginary (m : ℝ) : ∃ a : ℝ, z m = a * I :=
  sorry

-- Prove that z is purely imaginary iff m = ±1
theorem z_is_purely_imaginary_iff (m : ℝ) : (∃ b : ℝ, z m = b * I ∧ b ≠ 0) ↔ (m = 1 ∨ m = -1) :=
  sorry

-- Prove that z is on the angle bisector iff m = 0
theorem z_on_angle_bisector_iff (m : ℝ) : (z m).re = -((z m).im) ↔ (m = 0) :=
  sorry

end NUMINAMATH_GPT_z_is_imaginary_z_is_purely_imaginary_iff_z_on_angle_bisector_iff_l1208_120877


namespace NUMINAMATH_GPT_ewan_sequence_has_113_l1208_120899

def sequence_term (n : ℕ) : ℤ := 11 * n - 8

theorem ewan_sequence_has_113 : ∃ n : ℕ, sequence_term n = 113 := by
  sorry

end NUMINAMATH_GPT_ewan_sequence_has_113_l1208_120899


namespace NUMINAMATH_GPT_optimal_play_probability_Reimu_l1208_120891

noncomputable def probability_Reimu_wins : ℚ :=
  5 / 16

theorem optimal_play_probability_Reimu :
  probability_Reimu_wins = 5 / 16 := 
by
  sorry

end NUMINAMATH_GPT_optimal_play_probability_Reimu_l1208_120891


namespace NUMINAMATH_GPT_total_digits_2500_is_9449_l1208_120856

def nth_even (n : ℕ) : ℕ := 2 * n

def count_digits_in_range (start : ℕ) (stop : ℕ) : ℕ :=
  (stop - start) / 2 + 1

def total_digits (n : ℕ) : ℕ :=
  let one_digit := 4
  let two_digit := count_digits_in_range 10 98
  let three_digit := count_digits_in_range 100 998
  let four_digit := count_digits_in_range 1000 4998
  let five_digit := 1
  one_digit * 1 +
  two_digit * 2 +
  (three_digit * 3) +
  (four_digit * 4) +
  (five_digit * 5)

theorem total_digits_2500_is_9449 : total_digits 2500 = 9449 := by
  sorry

end NUMINAMATH_GPT_total_digits_2500_is_9449_l1208_120856


namespace NUMINAMATH_GPT_shortest_distance_between_circles_l1208_120868

-- Conditions
def first_circle (x y : ℝ) : Prop := x^2 - 10 * x + y^2 - 4 * y - 7 = 0
def second_circle (x y : ℝ) : Prop := x^2 + 14 * x + y^2 + 6 * y + 49 = 0

-- Goal: Prove the shortest distance between the two circles is 4
theorem shortest_distance_between_circles : 
  -- Given conditions about the equations of the circles
  (∀ x y : ℝ, first_circle x y ↔ (x - 5)^2 + (y - 2)^2 = 36) ∧ 
  (∀ x y : ℝ, second_circle x y ↔ (x + 7)^2 + (y + 3)^2 = 9) →
  -- Assert the shortest distance between the two circles is 4
  13 - (6 + 3) = 4 :=
by
  sorry

end NUMINAMATH_GPT_shortest_distance_between_circles_l1208_120868


namespace NUMINAMATH_GPT_find_x_l1208_120873

theorem find_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : x = 12 := by
  sorry

end NUMINAMATH_GPT_find_x_l1208_120873


namespace NUMINAMATH_GPT_percent_increase_calculation_l1208_120836

variable (x y : ℝ) -- Declare x and y as real numbers representing the original salary and increment

-- The statement that the percent increase z follows from the given conditions
theorem percent_increase_calculation (h : y + x = x + y) : (y / x) * 100 = ((y / x) * 100) := by
  sorry

end NUMINAMATH_GPT_percent_increase_calculation_l1208_120836


namespace NUMINAMATH_GPT_painting_faces_not_sum_to_nine_l1208_120848

def eight_sided_die_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def pairs_that_sum_to_nine : List (ℕ × ℕ) := [(1, 8), (2, 7), (3, 6), (4, 5)]

theorem painting_faces_not_sum_to_nine :
  let total_pairs := (eight_sided_die_numbers.length * (eight_sided_die_numbers.length - 1)) / 2
  let invalid_pairs := pairs_that_sum_to_nine.length
  total_pairs - invalid_pairs = 24 :=
by
  sorry

end NUMINAMATH_GPT_painting_faces_not_sum_to_nine_l1208_120848


namespace NUMINAMATH_GPT_remainder_7_pow_150_mod_4_l1208_120832

theorem remainder_7_pow_150_mod_4 : (7 ^ 150) % 4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_7_pow_150_mod_4_l1208_120832


namespace NUMINAMATH_GPT_evaluate_expression_l1208_120825

theorem evaluate_expression :
  (- (3 / 4 : ℚ)) / 3 * (- (2 / 5 : ℚ)) = 1 / 10 := 
by
  -- Here is where the proof would go
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1208_120825


namespace NUMINAMATH_GPT_function_properties_l1208_120850

noncomputable def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

theorem function_properties 
  (b c : ℝ) :
  ((c = 0 → (∀ x : ℝ, f (-x) b 0 = -f x b 0)) ∧
   (b = 0 → (∀ x₁ x₂ : ℝ, (x₁ ≤ x₂ → f x₁ 0 c ≤ f x₂ 0 c))) ∧
   (∃ (c : ℝ), ∀ (x : ℝ), f (x + c) b c = f (x - c) b c) ∧
   (¬ ∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ b c = 0 ∧ f x₂ b c = 0 ∧ f x₃ b c = 0))) := 
by
  sorry

end NUMINAMATH_GPT_function_properties_l1208_120850


namespace NUMINAMATH_GPT_minimum_n_of_colored_balls_l1208_120842

theorem minimum_n_of_colored_balls (n : ℕ) (h1 : n ≥ 3)
  (h2 : (n * (n + 1)) / 2 % 10 = 0) : n = 24 :=
sorry

end NUMINAMATH_GPT_minimum_n_of_colored_balls_l1208_120842


namespace NUMINAMATH_GPT_minimum_value_frac_l1208_120879

theorem minimum_value_frac (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (2 / a) + (3 / b) ≥ 5 + 2 * Real.sqrt 6 := 
sorry

end NUMINAMATH_GPT_minimum_value_frac_l1208_120879


namespace NUMINAMATH_GPT_extreme_point_at_1_l1208_120810

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2) * x^2 + (2 * a^3 - a^2) * Real.log x - (a^2 + 2 * a - 1) * x

theorem extreme_point_at_1 (a : ℝ) :
  (∃ x : ℝ, x = 1 ∧ ∀ x > 0, deriv (f a) x = 0 →
  a = -1) := sorry

end NUMINAMATH_GPT_extreme_point_at_1_l1208_120810


namespace NUMINAMATH_GPT_divisibility_of_polynomial_l1208_120805

theorem divisibility_of_polynomial (n : ℕ) (h : n ≥ 1) : 
  ∃ primes : Finset ℕ, primes.card = n ∧ ∀ p ∈ primes, p.Prime ∧ p ∣ (2^(2^n) + 2^(2^(n-1)) + 1) :=
sorry

end NUMINAMATH_GPT_divisibility_of_polynomial_l1208_120805


namespace NUMINAMATH_GPT_toucan_count_l1208_120862

theorem toucan_count :
  (2 + 1 = 3) :=
by simp [add_comm]

end NUMINAMATH_GPT_toucan_count_l1208_120862


namespace NUMINAMATH_GPT_range_of_a_union_B_eq_A_range_of_a_inter_B_eq_empty_l1208_120809

open Set

noncomputable def A (a : ℝ) : Set ℝ := { x : ℝ | a - 1 < x ∧ x < 2 * a + 1 }
def B : Set ℝ := { x : ℝ | 0 < x ∧ x < 1 }

theorem range_of_a_union_B_eq_A (a : ℝ) :
  (A a ∪ B) = A a ↔ (0 ≤ a ∧ a ≤ 1) := by
  sorry

theorem range_of_a_inter_B_eq_empty (a : ℝ) :
  (A a ∩ B) = ∅ ↔ (a ≤ - 1 / 2 ∨ 2 ≤ a) := by
  sorry

end NUMINAMATH_GPT_range_of_a_union_B_eq_A_range_of_a_inter_B_eq_empty_l1208_120809


namespace NUMINAMATH_GPT_problem_f_2019_l1208_120827

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f1 : f 1 = 1/4
axiom f2 : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem problem_f_2019 : f 2019 = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_problem_f_2019_l1208_120827


namespace NUMINAMATH_GPT_M_is_correct_l1208_120804

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x > 2}

def M := {x | x ∈ A ∧ x ∉ B}

theorem M_is_correct : M = {1, 2} := by
  -- Proof needed here
  sorry

end NUMINAMATH_GPT_M_is_correct_l1208_120804


namespace NUMINAMATH_GPT_distribution_of_earnings_l1208_120839

theorem distribution_of_earnings :
  let payments := [10, 15, 20, 25, 30, 50]
  let total_earnings := payments.sum 
  let equal_share := total_earnings / 6
  50 - equal_share = 25 := by
  sorry

end NUMINAMATH_GPT_distribution_of_earnings_l1208_120839


namespace NUMINAMATH_GPT_inverse_proportion_k_value_l1208_120875

theorem inverse_proportion_k_value (k : ℝ) (h₁ : k ≠ 0) (h₂ : (2, -1) ∈ {p : ℝ × ℝ | ∃ (k' : ℝ), k' = k ∧ p.snd = k' / p.fst}) :
  k = -2 := 
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_k_value_l1208_120875


namespace NUMINAMATH_GPT_sufficient_condition_for_quadratic_l1208_120801

theorem sufficient_condition_for_quadratic (a : ℝ) : 
  (∃ (x : ℝ), (x > a) ∧ (x^2 - 5*x + 6 ≥ 0)) ∧ 
  (¬(∀ (x : ℝ), (x^2 - 5*x + 6 ≥ 0) → (x > a))) ↔ 
  a ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_quadratic_l1208_120801


namespace NUMINAMATH_GPT_children_gift_distribution_l1208_120813

theorem children_gift_distribution (N : ℕ) (hN : N > 1) :
  (∀ n : ℕ, n < N → (∃ k : ℕ, k < N ∧ k ≠ n)) →
  (∃ m : ℕ, (N - 1) = 2 * m) :=
by
  sorry

end NUMINAMATH_GPT_children_gift_distribution_l1208_120813


namespace NUMINAMATH_GPT_correct_mean_l1208_120831

-- Definitions of conditions
def n : ℕ := 30
def mean_incorrect : ℚ := 140
def value_correct : ℕ := 145
def value_incorrect : ℕ := 135

-- The statement to be proved
theorem correct_mean : 
  let S_incorrect := mean_incorrect * n
  let Difference := value_correct - value_incorrect
  let S_correct := S_incorrect + Difference
  let mean_correct := S_correct / n
  mean_correct = 140.33 := 
by
  sorry

end NUMINAMATH_GPT_correct_mean_l1208_120831


namespace NUMINAMATH_GPT_production_line_improvement_better_than_financial_investment_l1208_120838

noncomputable def improved_mean_rating (initial_mean : ℝ) := initial_mean + 0.05

noncomputable def combined_mean_rating (mean_unimproved : ℝ) (mean_improved : ℝ) : ℝ :=
  (mean_unimproved * 200 + mean_improved * 200) / 400

noncomputable def combined_variance (variance : ℝ) (combined_mean : ℝ) : ℝ :=
  (2 * variance) - combined_mean ^ 2

noncomputable def increased_returns (grade_a_price : ℝ) (grade_b_price : ℝ) 
  (proportion_upgraded : ℝ) (units_per_day : ℕ) (days_per_year : ℕ) : ℝ :=
  (grade_a_price - grade_b_price) * proportion_upgraded * units_per_day * days_per_year - 200000000

noncomputable def financial_returns (initial_investment : ℝ) (annual_return_rate : ℝ) : ℝ :=
  initial_investment * (1 + annual_return_rate) - initial_investment

theorem production_line_improvement_better_than_financial_investment 
  (initial_mean : ℝ := 9.98) 
  (initial_variance : ℝ := 0.045) 
  (grade_a_price : ℝ := 2000) 
  (grade_b_price : ℝ := 1200) 
  (proportion_upgraded : ℝ := 3 / 8) 
  (units_per_day : ℕ := 200) 
  (days_per_year : ℕ := 365) 
  (initial_investment : ℝ := 200000000) 
  (annual_return_rate : ℝ := 0.082) : 
  combined_mean_rating initial_mean (improved_mean_rating initial_mean) = 10.005 ∧ 
  combined_variance initial_variance (combined_mean_rating initial_mean (improved_mean_rating initial_mean)) = 0.045625 ∧ 
  increased_returns grade_a_price grade_b_price proportion_upgraded units_per_day days_per_year > financial_returns initial_investment annual_return_rate := 
by {
  sorry
}

end NUMINAMATH_GPT_production_line_improvement_better_than_financial_investment_l1208_120838


namespace NUMINAMATH_GPT_smallest_b_to_the_a_l1208_120894

theorem smallest_b_to_the_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = 2^2023) : b^a = 1 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_smallest_b_to_the_a_l1208_120894


namespace NUMINAMATH_GPT_remainder_of_power_mod_l1208_120814

theorem remainder_of_power_mod (a n p : ℕ) (h_prime : Nat.Prime p) (h_a : a < p) :
  (3 : ℕ)^2024 % 17 = 13 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_power_mod_l1208_120814


namespace NUMINAMATH_GPT_conditional_probability_chinese_fail_l1208_120834

theorem conditional_probability_chinese_fail :
  let P_math := 0.16
  let P_chinese := 0.07
  let P_both := 0.04
  P_both / P_chinese = (4 / 7) := by
  let P_math := 0.16
  let P_chinese := 0.07
  let P_both := 0.04
  sorry

end NUMINAMATH_GPT_conditional_probability_chinese_fail_l1208_120834


namespace NUMINAMATH_GPT_cos_A_minus_B_l1208_120888

theorem cos_A_minus_B (A B : Real) 
  (h1 : Real.sin A + Real.sin B = -1) 
  (h2 : Real.cos A + Real.cos B = 1/2) :
  Real.cos (A - B) = -3/8 :=
by
  sorry

end NUMINAMATH_GPT_cos_A_minus_B_l1208_120888


namespace NUMINAMATH_GPT_sum_even_odd_diff_l1208_120822

theorem sum_even_odd_diff (n : ℕ) (h : n = 1500) : 
  let S_odd := n / 2 * (1 + (1 + (n - 1) * 2))
  let S_even := n / 2 * (2 + (2 + (n - 1) * 2))
  (S_even - S_odd) = n :=
by
  sorry

end NUMINAMATH_GPT_sum_even_odd_diff_l1208_120822


namespace NUMINAMATH_GPT_joan_apples_l1208_120895

def initial_apples : ℕ := 43
def additional_apples : ℕ := 27
def total_apples (initial additional: ℕ) := initial + additional

theorem joan_apples : total_apples initial_apples additional_apples = 70 := by
  sorry

end NUMINAMATH_GPT_joan_apples_l1208_120895


namespace NUMINAMATH_GPT_proof_p_and_q_true_l1208_120878

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x
def q : Prop := ∀ x : ℝ, exp x > x

theorem proof_p_and_q_true : p ∧ q :=
by
  -- Assume you have already proven that p and q are true separately
  sorry

end NUMINAMATH_GPT_proof_p_and_q_true_l1208_120878


namespace NUMINAMATH_GPT_discounted_price_correct_l1208_120896

noncomputable def discounted_price (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price - (discount / 100 * original_price)

theorem discounted_price_correct :
  discounted_price 800 30 = 560 :=
by
  -- Correctness of the discounted price calculation
  sorry

end NUMINAMATH_GPT_discounted_price_correct_l1208_120896


namespace NUMINAMATH_GPT_union_of_A_and_B_l1208_120835

open Set

variable {α : Type}

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2, 4} := 
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1208_120835


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1208_120807

theorem necessary_and_sufficient_condition (x : ℝ) : (x > 0) ↔ (1 / x > 0) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1208_120807


namespace NUMINAMATH_GPT_remaining_pie_after_carlos_and_maria_l1208_120800

theorem remaining_pie_after_carlos_and_maria (C M R : ℝ) (hC : C = 0.60) (hM : M = 0.25 * (1 - C)) : R = 1 - C - M → R = 0.30 :=
by
  intro hR
  simp only [hC, hM] at hR
  sorry

end NUMINAMATH_GPT_remaining_pie_after_carlos_and_maria_l1208_120800


namespace NUMINAMATH_GPT_power_function_below_identity_l1208_120882

theorem power_function_below_identity {α : ℝ} :
  (∀ x : ℝ, 1 < x → x^α < x) → α < 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_power_function_below_identity_l1208_120882


namespace NUMINAMATH_GPT_find_t_and_m_l1208_120884

theorem find_t_and_m 
  (t m : ℝ) 
  (ineq : ∀ x : ℝ, x^2 - 3 * x + t < 0 ↔ 1 < x ∧ x < m) : 
  t = 2 ∧ m = 2 :=
sorry

end NUMINAMATH_GPT_find_t_and_m_l1208_120884


namespace NUMINAMATH_GPT_tangent_line_through_point_l1208_120866

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x

theorem tangent_line_through_point (x y : ℝ) (h₁ : y = 2 * Real.log x - x) (h₂ : (1 : ℝ)  ≠ 0) 
  (h₃ : (-1 : ℝ) ≠ 0):
  (x - y - 2 = 0) :=
sorry

end NUMINAMATH_GPT_tangent_line_through_point_l1208_120866


namespace NUMINAMATH_GPT_polynomial_evaluation_l1208_120860

def p (x : ℝ) (a b c d : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + d

theorem polynomial_evaluation
  (a b c d : ℝ)
  (h1 : p 1 a b c d = 1993)
  (h2 : p 2 a b c d = 3986)
  (h3 : p 3 a b c d = 5979) :
  (1 / 4 : ℝ) * (p 11 a b c d + p (-7) a b c d) = 5233 := by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l1208_120860


namespace NUMINAMATH_GPT_final_movie_ticket_price_l1208_120824

variable (initial_price : ℝ) (price_year1 price_year2 price_year3 price_year4 price_year5 : ℝ)

def price_after_years (initial_price : ℝ) : ℝ :=
  let price_year1 := initial_price * 1.12
  let price_year2 := price_year1 * 0.95
  let price_year3 := price_year2 * 1.08
  let price_year4 := price_year3 * 0.96
  let price_year5 := price_year4 * 1.06
  price_year5

theorem final_movie_ticket_price :
  price_after_years 100 = 116.9344512 :=
by
  sorry

end NUMINAMATH_GPT_final_movie_ticket_price_l1208_120824


namespace NUMINAMATH_GPT_vector_dot_product_example_l1208_120815

noncomputable def vector_dot_product (e1 e2 : ℝ) : ℝ :=
  let c := e1 * (-3 * e1)
  let d := (e1 * (2 * e2))
  let e := (e2 * (2 * e2))
  c + d + e

theorem vector_dot_product_example (e1 e2 : ℝ) (unit_vectors : e1^2 = 1 ∧ e2^2 = 1) :
  (e1 - e2) * (e1 - e2) = 1 ∧ (e1 * e2 = 1 / 2) → 
  vector_dot_product e1 e2 = -5 / 2 := by {
  sorry
}

end NUMINAMATH_GPT_vector_dot_product_example_l1208_120815


namespace NUMINAMATH_GPT_sin_theta_plus_pi_over_six_l1208_120826

open Real

theorem sin_theta_plus_pi_over_six (theta : ℝ) (h : sin θ + sin (θ + π / 3) = sqrt 3) :
  sin (θ + π / 6) = 1 := 
sorry

end NUMINAMATH_GPT_sin_theta_plus_pi_over_six_l1208_120826


namespace NUMINAMATH_GPT_sum_bn_over_3_pow_n_plus_1_eq_2_over_5_l1208_120828

noncomputable def b : ℕ → ℚ
| 0     => 2
| 1     => 3
| (n+2) => 2 * b (n+1) + 3 * b n

theorem sum_bn_over_3_pow_n_plus_1_eq_2_over_5 :
  (∑' n : ℕ, (b n) / (3 ^ (n + 1))) = (2 / 5) :=
by
  sorry

end NUMINAMATH_GPT_sum_bn_over_3_pow_n_plus_1_eq_2_over_5_l1208_120828


namespace NUMINAMATH_GPT_quadratic_function_graph_opens_downwards_l1208_120857

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

-- The problem statement to prove
theorem quadratic_function_graph_opens_downwards :
  (∀ x : ℝ, (quadratic_function (x + 1) - quadratic_function x) < (quadratic_function x - quadratic_function (x - 1))) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_quadratic_function_graph_opens_downwards_l1208_120857


namespace NUMINAMATH_GPT_find_number_l1208_120840

theorem find_number
  (a b c : ℕ)
  (h_a1 : a ≤ 3)
  (h_b1 : b ≤ 3)
  (h_c1 : c ≤ 3)
  (h_a2 : a ≠ 3)
  (h_b_condition1 : b ≠ 1 → 2 * a * b < 10)
  (h_b_condition2 : b ≠ 2 → 2 * a * b < 10)
  (h_c3 : c = 3)
  : a = 2 ∧ b = 3 ∧ c = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1208_120840


namespace NUMINAMATH_GPT_tangent_line_at_point_l1208_120854

noncomputable def curve (x : ℝ) : ℝ := Real.exp x + x

theorem tangent_line_at_point :
  (∃ k b : ℝ, (∀ x : ℝ, curve x = k * x + b) ∧ k = 2 ∧ b = 1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_l1208_120854


namespace NUMINAMATH_GPT_larry_jogs_first_week_days_l1208_120880

-- Defining the constants and conditions
def daily_jogging_time := 30 -- Larry jogs for 30 minutes each day
def total_jogging_time_in_hours := 4 -- Total jogging time in two weeks in hours
def total_jogging_time_in_minutes := total_jogging_time_in_hours * 60 -- Convert hours to minutes
def jogging_days_in_second_week := 5 -- Larry jogs 5 days in the second week
def daily_jogging_time_in_week2 := jogging_days_in_second_week * daily_jogging_time -- Total jogging time in minutes in the second week

-- Theorem statement
theorem larry_jogs_first_week_days : 
  (total_jogging_time_in_minutes - daily_jogging_time_in_week2) / daily_jogging_time = 3 :=
by
  -- Definitions and conditions used above should directly appear from the problem statement
  sorry

end NUMINAMATH_GPT_larry_jogs_first_week_days_l1208_120880


namespace NUMINAMATH_GPT_find_m_n_sum_l1208_120806

noncomputable def q : ℚ := 2 / 11

theorem find_m_n_sum {m n : ℕ} (hq : q = m / n) (coprime_mn : Nat.gcd m n = 1) : m + n = 13 := by
  sorry

end NUMINAMATH_GPT_find_m_n_sum_l1208_120806
