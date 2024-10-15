import Mathlib

namespace NUMINAMATH_GPT_power_mod_l103_10366

theorem power_mod (n m : ℕ) (hn : n = 13) (hm : m = 1000) : n ^ 21 % m = 413 :=
by
  rw [hn, hm]
  -- other steps of the proof would go here...
  sorry

end NUMINAMATH_GPT_power_mod_l103_10366


namespace NUMINAMATH_GPT_range_of_a_for_increasing_l103_10315

noncomputable def f (a : ℝ) : (ℝ → ℝ) := λ x => x^3 + a * x^2 + 3 * x

theorem range_of_a_for_increasing (a : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 2 * a * x + 3) ≥ 0) ↔ (-3 ≤ a ∧ a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_increasing_l103_10315


namespace NUMINAMATH_GPT_probability_both_blue_l103_10339

-- Conditions defined as assumptions
def jarC_red := 6
def jarC_blue := 10
def total_buttons_in_C := jarC_red + jarC_blue

def after_transfer_buttons_in_C := (3 / 4) * total_buttons_in_C

-- Carla removes the same number of red and blue buttons
-- and after transfer, 12 buttons remain in Jar C
def removed_buttons := total_buttons_in_C - after_transfer_buttons_in_C
def removed_red_buttons := removed_buttons / 2
def removed_blue_buttons := removed_buttons / 2

def remaining_red_in_C := jarC_red - removed_red_buttons
def remaining_blue_in_C := jarC_blue - removed_blue_buttons
def remaining_buttons_in_C := remaining_red_in_C + remaining_blue_in_C

def total_buttons_in_D := removed_buttons
def transferred_blue_buttons := removed_blue_buttons

-- Probability calculations
def probability_blue_in_C := remaining_blue_in_C / remaining_buttons_in_C
def probability_blue_in_D := transferred_blue_buttons / total_buttons_in_D

-- Proof
theorem probability_both_blue :
  (probability_blue_in_C * probability_blue_in_D) = (1 / 3) := 
by
  -- sorry is used here to skip the actual proof
  sorry

end NUMINAMATH_GPT_probability_both_blue_l103_10339


namespace NUMINAMATH_GPT_g_five_eq_one_l103_10344

noncomputable def g : ℝ → ℝ := sorry

theorem g_five_eq_one 
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0)
  (h3 : ∀ x : ℝ, g x = g (-x)) : 
  g 5 = 1 :=
sorry

end NUMINAMATH_GPT_g_five_eq_one_l103_10344


namespace NUMINAMATH_GPT_exists_eleven_consecutive_numbers_sum_cube_l103_10363

theorem exists_eleven_consecutive_numbers_sum_cube :
  ∃ (n k : ℕ), (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) + (n+8) + (n+9) + (n+10)) = k^3 :=
by
  sorry

end NUMINAMATH_GPT_exists_eleven_consecutive_numbers_sum_cube_l103_10363


namespace NUMINAMATH_GPT_solve_equation_solve_proportion_l103_10334

theorem solve_equation (x : ℚ) :
  (3 + x) * (30 / 100) = 4.8 → x = 13 :=
by sorry

theorem solve_proportion (x : ℚ) :
  (5 / x) = (9 / 2) / (8 / 5) → x = (16 / 9) :=
by sorry

end NUMINAMATH_GPT_solve_equation_solve_proportion_l103_10334


namespace NUMINAMATH_GPT_Jim_remaining_miles_l103_10301

-- Define the total journey miles and miles already driven
def total_miles : ℕ := 1200
def miles_driven : ℕ := 215

-- Define the remaining miles Jim needs to drive
def remaining_miles (total driven : ℕ) : ℕ := total - driven

-- Statement to prove
theorem Jim_remaining_miles : remaining_miles total_miles miles_driven = 985 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_Jim_remaining_miles_l103_10301


namespace NUMINAMATH_GPT_age_ratio_l103_10313

def Kul : ℕ := 22
def Saras : ℕ := 33

theorem age_ratio : (Saras / Kul : ℚ) = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_age_ratio_l103_10313


namespace NUMINAMATH_GPT_decreasing_sequence_b_l103_10311

def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, 2 * a n * a (n + 1) = (a n)^2 + 1

def b_n (a b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = (a n - 1) / (a n + 1)

theorem decreasing_sequence_b {a b : ℕ → ℝ} (h1 : seq_a a) (h2 : b_n a b) :
  ∀ n : ℕ, b (n + 1) < b n :=
by
  sorry

end NUMINAMATH_GPT_decreasing_sequence_b_l103_10311


namespace NUMINAMATH_GPT_mean_score_40_l103_10379

theorem mean_score_40 (mean : ℝ) (std_dev : ℝ) (h_std_dev : std_dev = 10)
  (h_within_2_std_dev : ∀ (score : ℝ), score ≥ mean - 2 * std_dev)
  (h_lowest_score : ∀ (score : ℝ), score = 20 → score = mean - 20) :
  mean = 40 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_mean_score_40_l103_10379


namespace NUMINAMATH_GPT_petya_board_problem_l103_10367

variable (A B Z : ℕ)

theorem petya_board_problem (h1 : A + B + Z = 10) (h2 : A * B = 15) : Z = 2 := sorry

end NUMINAMATH_GPT_petya_board_problem_l103_10367


namespace NUMINAMATH_GPT_express_in_scientific_notation_l103_10354

theorem express_in_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 388800 = a * 10 ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.888 ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_express_in_scientific_notation_l103_10354


namespace NUMINAMATH_GPT_contribution_proof_l103_10393

theorem contribution_proof (total : ℕ) (a_months b_months : ℕ) (a_total b_total a_received b_received : ℕ) :
  total = 3400 →
  a_months = 12 →
  b_months = 16 →
  a_received = 2070 →
  b_received = 1920 →
  (∃ (a_contributed b_contributed : ℕ), a_contributed = 1800 ∧ b_contributed = 1600) :=
by
  sorry

end NUMINAMATH_GPT_contribution_proof_l103_10393


namespace NUMINAMATH_GPT_cost_of_two_pencils_and_one_pen_l103_10349

variable (a b : ℝ)

-- Given conditions
def condition1 : Prop := (5 * a + b = 2.50)
def condition2 : Prop := (a + 2 * b = 1.85)

-- Statement to prove
theorem cost_of_two_pencils_and_one_pen
  (h1 : condition1 a b) 
  (h2 : condition2 a b) : 
  2 * a + b = 1.45 :=
sorry

end NUMINAMATH_GPT_cost_of_two_pencils_and_one_pen_l103_10349


namespace NUMINAMATH_GPT_box_length_is_10_l103_10385

theorem box_length_is_10
  (width height vol_cube num_cubes : ℕ)
  (h₀ : width = 13)
  (h₁ : height = 5)
  (h₂ : vol_cube = 5)
  (h₃ : num_cubes = 130) :
  (num_cubes * vol_cube) / (width * height) = 10 :=
by
  -- Proof steps will be filled here.
  sorry

end NUMINAMATH_GPT_box_length_is_10_l103_10385


namespace NUMINAMATH_GPT_range_of_a_opposite_sides_l103_10305

theorem range_of_a_opposite_sides {a : ℝ} (h : (0 + 0 - a) * (1 + 1 - a) < 0) : 0 < a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_opposite_sides_l103_10305


namespace NUMINAMATH_GPT_smallest_positive_integer_n_mean_squares_l103_10350

theorem smallest_positive_integer_n_mean_squares :
  ∃ n : ℕ, n > 1 ∧ (∃ m : ℕ, (n * m ^ 2 = (n + 1) * (2 * n + 1) / 6) ∧ Nat.gcd (n + 1) (2 * n + 1) = 1 ∧ n = 337) :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_n_mean_squares_l103_10350


namespace NUMINAMATH_GPT_no_prime_divisible_by_77_l103_10389

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → is_divisible p 77 → false :=
by
  sorry

end NUMINAMATH_GPT_no_prime_divisible_by_77_l103_10389


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_function_l103_10342

variable (p q : ℝ) (hp : 0 < p) (hq : 0 < q)

theorem minimum_value_of_quadratic_function : 
  (∃ x : ℝ, x = p) ∧ (∀ x : ℝ, (x^2 - 2 * p * x + 4 * q) ≥ (p^2 - 2 * p * p + 4 * q)) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_function_l103_10342


namespace NUMINAMATH_GPT_cauchy_solution_l103_10397

theorem cauchy_solution (f : ℚ → ℚ) (h : ∀ x y : ℚ, f ((x + y) / 2) = (f x) / 2 + (f y) / 2) : 
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x := 
sorry

end NUMINAMATH_GPT_cauchy_solution_l103_10397


namespace NUMINAMATH_GPT_Brandy_can_safely_drink_20_mg_more_l103_10319

variable (maximum_caffeine_per_day : ℕ := 500)
variable (caffeine_per_drink : ℕ := 120)
variable (number_of_drinks : ℕ := 4)
variable (caffeine_consumed : ℕ := caffeine_per_drink * number_of_drinks)

theorem Brandy_can_safely_drink_20_mg_more :
    caffeine_consumed = caffeine_per_drink * number_of_drinks →
    (maximum_caffeine_per_day - caffeine_consumed) = 20 :=
by
  intros h1
  rw [h1]
  sorry

end NUMINAMATH_GPT_Brandy_can_safely_drink_20_mg_more_l103_10319


namespace NUMINAMATH_GPT_students_average_comparison_l103_10322

theorem students_average_comparison (t1 t2 t3 : ℝ) (h : t1 < t2) (h' : t2 < t3) :
  (∃ t1 t2 t3 : ℝ, t1 < t2 ∧ t2 < t3 ∧ (t1 + t2 + t3) / 3 = (t1 + t3 + 2 * t2) / 4) ∨
  (∀ t1 t2 t3 : ℝ, t1 < t2 ∧ t2 < t3 → 
     (t1 + t3 + 2 * t2) / 4 > (t1 + t2 + t3) / 3) :=
sorry

end NUMINAMATH_GPT_students_average_comparison_l103_10322


namespace NUMINAMATH_GPT_total_spending_is_450_l103_10302

-- Define the costs of items bought by Leonard
def leonard_wallet_cost : ℕ := 50
def pair_of_sneakers_cost : ℕ := 100
def pairs_of_sneakers : ℕ := 2

-- Define the costs of items bought by Michael
def michael_backpack_cost : ℕ := 100
def pair_of_jeans_cost : ℕ := 50
def pairs_of_jeans : ℕ := 2

-- Define the total spending of Leonard and Michael 
def total_spent : ℕ :=
  leonard_wallet_cost + (pair_of_sneakers_cost * pairs_of_sneakers) + 
  michael_backpack_cost + (pair_of_jeans_cost * pairs_of_jeans)

-- The proof statement
theorem total_spending_is_450 : total_spent = 450 := 
by
  sorry

end NUMINAMATH_GPT_total_spending_is_450_l103_10302


namespace NUMINAMATH_GPT_total_cans_collected_l103_10312

-- Definitions based on conditions
def cans_LaDonna : ℕ := 25
def cans_Prikya : ℕ := 2 * cans_LaDonna
def cans_Yoki : ℕ := 10

-- Theorem statement
theorem total_cans_collected : 
  cans_LaDonna + cans_Prikya + cans_Yoki = 85 :=
by
  -- The proof is not required, inserting sorry to complete the statement
  sorry

end NUMINAMATH_GPT_total_cans_collected_l103_10312


namespace NUMINAMATH_GPT_find_y_l103_10335

theorem find_y (x : ℝ) (h : x^2 + (1 / x)^2 = 7) : x + 1 / x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l103_10335


namespace NUMINAMATH_GPT_n_value_l103_10326

theorem n_value (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 → n = 9 :=
by
  sorry

end NUMINAMATH_GPT_n_value_l103_10326


namespace NUMINAMATH_GPT_recycling_weight_l103_10377

theorem recycling_weight :
  let marcus_milk_bottles := 25
  let john_milk_bottles := 20
  let sophia_milk_bottles := 15
  let marcus_cans := 30
  let john_cans := 25
  let sophia_cans := 35
  let milk_bottle_weight := 0.5
  let can_weight := 0.025

  let total_milk_bottles_weight := (marcus_milk_bottles + john_milk_bottles + sophia_milk_bottles) * milk_bottle_weight
  let total_cans_weight := (marcus_cans + john_cans + sophia_cans) * can_weight
  let combined_weight := total_milk_bottles_weight + total_cans_weight

  combined_weight = 32.25 :=
by
  sorry

end NUMINAMATH_GPT_recycling_weight_l103_10377


namespace NUMINAMATH_GPT_quadratic_residue_one_mod_p_l103_10370

theorem quadratic_residue_one_mod_p (p : ℕ) [hp : Fact (Nat.Prime p)] (a : ℕ) :
  (a^2 % p = 1 % p) ↔ (a % p = 1 % p ∨ a % p = (p-1) % p) :=
sorry

end NUMINAMATH_GPT_quadratic_residue_one_mod_p_l103_10370


namespace NUMINAMATH_GPT_time_to_walk_2_miles_l103_10325

/-- I walked 2 miles in a certain amount of time. -/
def walked_distance : ℝ := 2

/-- If I maintained this pace for 8 hours, I would walk 16 miles. -/
def pace_condition (pace : ℝ) : Prop :=
  pace * 8 = 16

/-- Prove that it took me 1 hour to walk 2 miles. -/
theorem time_to_walk_2_miles (t : ℝ) (pace : ℝ) (h1 : walked_distance = pace * t) (h2 : pace_condition pace) :
  t = 1 :=
sorry

end NUMINAMATH_GPT_time_to_walk_2_miles_l103_10325


namespace NUMINAMATH_GPT_count_congruent_to_3_mod_8_l103_10337

theorem count_congruent_to_3_mod_8 : 
  ∃ (count : ℕ), count = 31 ∧ ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 250 → x % 8 = 3 → x = 8 * ((x - 3) / 8) + 3 := sorry

end NUMINAMATH_GPT_count_congruent_to_3_mod_8_l103_10337


namespace NUMINAMATH_GPT_solution_set_inequality_range_a_inequality_l103_10361

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - 2

theorem solution_set_inequality (x : ℝ) (a : ℝ) (h : a = 1) :
  f x a + abs (2*x - 3) > 0 ↔ (x < 2 / 3 ∨ 2 < x) := sorry

theorem range_a_inequality (a : ℝ) :
  (∀ x, f x a < abs (x - 3)) ↔ (1 < a ∧ a < 5) := sorry

end NUMINAMATH_GPT_solution_set_inequality_range_a_inequality_l103_10361


namespace NUMINAMATH_GPT_part_a_part_b_l103_10368

-- the conditions
variables (r R x : ℝ) (h_rltR : r < R)
variables (h_x : x = (R - r) / 2)
variables (h1 : 0 < x)
variables (h12_circles : ∀ i : ℕ, i ∈ Finset.range 12 → ∃ c_i : ℝ × ℝ, True)  -- Informal way to note 12 circles of radius x are placed

-- prove each part
theorem part_a (r R : ℝ) (h_rltR : r < R) : x = (R - r) / 2 :=
sorry

theorem part_b (r R : ℝ) (h_rltR : r < R) (h_x : x = (R - r) / 2) :
  (R / r) = (4 + Real.sqrt 6 - Real.sqrt 2) / (4 - Real.sqrt 6 + Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l103_10368


namespace NUMINAMATH_GPT_infinite_subsets_exists_divisor_l103_10387

-- Definition of the set M
def M : Set ℕ := { n | ∃ a b : ℕ, n = 2^a * 3^b }

-- Infinite family of subsets of M
variable (A : ℕ → Set ℕ)
variables (inf_family : ∀ i, A i ⊆ M)

-- Theorem statement
theorem infinite_subsets_exists_divisor :
  ∃ i j : ℕ, i ≠ j ∧ ∀ x ∈ A i, ∃ y ∈ A j, y ∣ x := by
  sorry

end NUMINAMATH_GPT_infinite_subsets_exists_divisor_l103_10387


namespace NUMINAMATH_GPT_c_d_not_true_l103_10351

variables (Beatles_haircut : Type → Prop) (hooligan : Type → Prop) (rude : Type → Prop)

-- Conditions
axiom a : ∃ x, Beatles_haircut x ∧ hooligan x
axiom b : ∀ y, hooligan y → rude y

-- Prove there is a rude hooligan with a Beatles haircut
theorem c : ∃ z, rude z ∧ Beatles_haircut z ∧ hooligan z :=
sorry

-- Disprove every rude hooligan having a Beatles haircut
theorem d_not_true : ¬(∀ w, rude w ∧ hooligan w → Beatles_haircut w) :=
sorry

end NUMINAMATH_GPT_c_d_not_true_l103_10351


namespace NUMINAMATH_GPT_quadratic_bound_l103_10398

theorem quadratic_bound (a b c : ℝ) :
  (∀ (u : ℝ), |u| ≤ 10 / 11 → ∃ (v : ℝ), |u - v| ≤ 1 / 11 ∧ |a * v^2 + b * v + c| ≤ 1) →
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 2 := by
  sorry

end NUMINAMATH_GPT_quadratic_bound_l103_10398


namespace NUMINAMATH_GPT_james_and_lisa_pizzas_l103_10331

theorem james_and_lisa_pizzas (slices_per_pizza : ℕ) (total_slices : ℕ) :
  slices_per_pizza = 6 →
  2 * total_slices = 3 * 8 →
  total_slices / slices_per_pizza = 2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_james_and_lisa_pizzas_l103_10331


namespace NUMINAMATH_GPT_remainder_of_3_pow_19_mod_10_l103_10380

theorem remainder_of_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_19_mod_10_l103_10380


namespace NUMINAMATH_GPT_range_of_3a_minus_b_l103_10395

theorem range_of_3a_minus_b (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 4) (h2 : -1 ≤ a - b ∧ a - b ≤ 2) :
  -1 ≤ (3 * a - b) ∧ (3 * a - b) ≤ 8 :=
sorry

end NUMINAMATH_GPT_range_of_3a_minus_b_l103_10395


namespace NUMINAMATH_GPT_find_e_l103_10347

-- Conditions
def f (x : ℝ) (b : ℝ) := 5 * x + b
def g (x : ℝ) (b : ℝ) := b * x + 4
def f_comp_g (x : ℝ) (b : ℝ) (e : ℝ) := 15 * x + e

-- Statement to prove
theorem find_e (b e : ℝ) (x : ℝ): 
  (f (g x b) b = f_comp_g x b e) → 
  (5 * b = 15) → 
  (20 + b = e) → 
  e = 23 :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_e_l103_10347


namespace NUMINAMATH_GPT_minimum_bottles_needed_l103_10336

theorem minimum_bottles_needed :
  (∃ n : ℕ, n * 45 ≥ 720 - 20 ∧ (n - 1) * 45 < 720 - 20) ∧ 720 - 20 = 700 :=
by
  sorry

end NUMINAMATH_GPT_minimum_bottles_needed_l103_10336


namespace NUMINAMATH_GPT_jill_tax_on_other_items_l103_10355

-- Define the conditions based on the problem statement.
variables (C : ℝ) (x : ℝ)
def tax_on_clothing := 0.04 * 0.60 * C
def tax_on_food := 0
def tax_on_other_items := 0.01 * x * 0.30 * C
def total_tax_paid := 0.048 * C

-- Prove the required percentage tax on other items.
theorem jill_tax_on_other_items :
  tax_on_clothing C + tax_on_food + tax_on_other_items C x = total_tax_paid C →
  x = 8 :=
by
  sorry

end NUMINAMATH_GPT_jill_tax_on_other_items_l103_10355


namespace NUMINAMATH_GPT_pipe_B_leak_time_l103_10348

theorem pipe_B_leak_time (t_B : ℝ) : (1 / 12 - 1 / t_B = 1 / 36) → t_B = 18 :=
by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_pipe_B_leak_time_l103_10348


namespace NUMINAMATH_GPT_percentage_decrease_after_raise_l103_10358

theorem percentage_decrease_after_raise
  (original_salary : ℝ) (final_salary : ℝ) (initial_raise_percent : ℝ)
  (initial_salary_raised : original_salary * (1 + initial_raise_percent / 100) = 5500): 
  original_salary = 5000 -> final_salary = 5225 -> initial_raise_percent = 10 ->
  ∃ (percentage_decrease : ℝ),
    final_salary = original_salary * (1 + initial_raise_percent / 100) * (1 - percentage_decrease / 100)
    ∧ percentage_decrease = 5 := by
  intros h1 h2 h3
  use 5
  rw [h1, h2, h3]
  simp
  sorry

end NUMINAMATH_GPT_percentage_decrease_after_raise_l103_10358


namespace NUMINAMATH_GPT_product_b1_b13_l103_10369

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Conditions for the arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) := ∀ n m k : ℕ, m > 0 → k > 0 → a (n + m) - a n = a (n + k) - a (n + k - m)

-- Conditions for the geometric sequence
def is_geometric_seq (b : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

-- Given conditions
def conditions (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  (a 3 - (a 7 ^ 2) / 2 + a 11 = 0) ∧ (b 7 = a 7)

theorem product_b1_b13 
  (ha : is_arithmetic_seq a)
  (hb : is_geometric_seq b)
  (h : conditions a b) :
  b 1 * b 13 = 16 :=
sorry

end NUMINAMATH_GPT_product_b1_b13_l103_10369


namespace NUMINAMATH_GPT_mn_equals_neg16_l103_10394

theorem mn_equals_neg16 (m n : ℤ) (h1 : m = -2) (h2 : |n| = 8) (h3 : m + n > 0) : m * n = -16 := by
  sorry

end NUMINAMATH_GPT_mn_equals_neg16_l103_10394


namespace NUMINAMATH_GPT_probability_penny_nickel_dime_heads_l103_10359

noncomputable def probability_heads (n : ℕ) : ℚ := (1 : ℚ) / (2 ^ n)

theorem probability_penny_nickel_dime_heads :
  probability_heads 3 = 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_probability_penny_nickel_dime_heads_l103_10359


namespace NUMINAMATH_GPT_exists_good_number_in_interval_l103_10378

def is_good_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≤ 5

theorem exists_good_number_in_interval (x : ℕ) (hx : x ≠ 0) :
  ∃ g : ℕ, is_good_number g ∧ x ≤ g ∧ g < ((9 * x) / 5) + 1 := 
sorry

end NUMINAMATH_GPT_exists_good_number_in_interval_l103_10378


namespace NUMINAMATH_GPT_basic_computer_price_l103_10314

theorem basic_computer_price :
  ∃ C P : ℝ,
    C + P = 2500 ∧
    (C + 800) + (1 / 5) * (C + 800 + P) = 2500 ∧
    (C + 1100) + (1 / 8) * (C + 1100 + P) = 2500 ∧
    (C + 1500) + (1 / 10) * (C + 1500 + P) = 2500 ∧
    C = 1040 :=
by
  sorry

end NUMINAMATH_GPT_basic_computer_price_l103_10314


namespace NUMINAMATH_GPT_min_value_c_and_d_l103_10338

theorem min_value_c_and_d (c d : ℝ) (h1 : c > 0) (h2 : d > 0)
  (h3 : c^2 - 12 * d ≥ 0)
  (h4 : 9 * d^2 - 4 * c ≥ 0) :
  c + d ≥ 5.74 :=
sorry

end NUMINAMATH_GPT_min_value_c_and_d_l103_10338


namespace NUMINAMATH_GPT_convex_polygon_sides_l103_10306

theorem convex_polygon_sides (n : ℕ) (h : ∀ angle, angle = 45 → angle * n = 360) : n = 8 :=
  sorry

end NUMINAMATH_GPT_convex_polygon_sides_l103_10306


namespace NUMINAMATH_GPT_angle_same_terminal_side_l103_10317

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, 95 = -265 + k * 360 :=
by
  use 1
  norm_num

end NUMINAMATH_GPT_angle_same_terminal_side_l103_10317


namespace NUMINAMATH_GPT_dilation_rotation_l103_10364

noncomputable def center : ℂ := 2 + 3 * Complex.I
noncomputable def scale_factor : ℂ := 3
noncomputable def initial_point : ℂ := -1 + Complex.I
noncomputable def final_image : ℂ := -4 + 12 * Complex.I

theorem dilation_rotation (z : ℂ) :
  z = (-1 + Complex.I) →
  let z' := center + scale_factor * (initial_point - center)
  let rotated_z := center + Complex.I * (z' - center)
  rotated_z = final_image := sorry

end NUMINAMATH_GPT_dilation_rotation_l103_10364


namespace NUMINAMATH_GPT_total_gum_correct_l103_10384

def num_cousins : ℕ := 4  -- Number of cousins
def gum_per_cousin : ℕ := 5  -- Pieces of gum per cousin

def total_gum : ℕ := num_cousins * gum_per_cousin  -- Total pieces of gum Kim needs

theorem total_gum_correct : total_gum = 20 :=
by sorry

end NUMINAMATH_GPT_total_gum_correct_l103_10384


namespace NUMINAMATH_GPT_least_pounds_of_sugar_l103_10388

theorem least_pounds_of_sugar :
  ∃ s : ℝ, (∀ f : ℝ, (f ≥ 6 + s / 2 ∧ f ≤ 2 * s) → s = 4) :=
by {
    use 4,
    sorry
}

end NUMINAMATH_GPT_least_pounds_of_sugar_l103_10388


namespace NUMINAMATH_GPT_division_problem_l103_10343

theorem division_problem : 96 / (8 / 4) = 48 := 
by {
  sorry
}

end NUMINAMATH_GPT_division_problem_l103_10343


namespace NUMINAMATH_GPT_multiple_proof_l103_10353

noncomputable def K := 185  -- Given KJ's stamps
noncomputable def AJ := 370  -- Given AJ's stamps
noncomputable def total_stamps := 930  -- Given total amount

-- Using the conditions to find C
noncomputable def stamps_of_three := AJ + K  -- Total stamps of KJ and AJ
noncomputable def C := total_stamps - stamps_of_three

-- Stating the equivalence we need to prove
theorem multiple_proof : ∃ M: ℕ, M * K + 5 = C := by
  -- The solution proof here if required
  existsi 2
  sorry  -- proof to be completed

end NUMINAMATH_GPT_multiple_proof_l103_10353


namespace NUMINAMATH_GPT_lowest_test_score_dropped_l103_10383

theorem lowest_test_score_dropped (A B C D : ℕ) 
  (h_avg_four : A + B + C + D = 140) 
  (h_avg_three : A + B + C = 120) : 
  D = 20 := 
by
  sorry

end NUMINAMATH_GPT_lowest_test_score_dropped_l103_10383


namespace NUMINAMATH_GPT_gravel_cost_l103_10316

def cost_per_cubic_foot := 8
def cubic_yards := 3
def cubic_feet_per_cubic_yard := 27

theorem gravel_cost :
  (cubic_yards * cubic_feet_per_cubic_yard) * cost_per_cubic_foot = 648 :=
by sorry

end NUMINAMATH_GPT_gravel_cost_l103_10316


namespace NUMINAMATH_GPT_degree_of_divisor_polynomial_l103_10386

theorem degree_of_divisor_polynomial (f d q r : Polynomial ℝ) 
  (hf : f.degree = 15)
  (hq : q.degree = 9)
  (hr : r.degree = 4)
  (hfdqr : f = d * q + r) :
  d.degree = 6 :=
by sorry

end NUMINAMATH_GPT_degree_of_divisor_polynomial_l103_10386


namespace NUMINAMATH_GPT_symmetry_center_range_in_interval_l103_10356

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6) + 1

theorem symmetry_center (k : ℤ) :
  ∃ n : ℤ, ∃ x : ℝ, x = Real.pi / 12 + n * Real.pi / 2 ∧ f x = 1 := 
sorry

theorem range_in_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → ∃ y : ℝ, f y ∈ Set.Icc 0 3 := 
sorry

end NUMINAMATH_GPT_symmetry_center_range_in_interval_l103_10356


namespace NUMINAMATH_GPT_no_lighter_sentence_for_liar_l103_10365

theorem no_lighter_sentence_for_liar
  (total_eggs : ℕ)
  (stolen_eggs1 stolen_eggs2 stolen_eggs3 : ℕ)
  (different_stolen_eggs : stolen_eggs1 ≠ stolen_eggs2 ∧ stolen_eggs2 ≠ stolen_eggs3 ∧ stolen_eggs1 ≠ stolen_eggs3)
  (stolen_eggs1_max : stolen_eggs1 > stolen_eggs2 ∧ stolen_eggs1 > stolen_eggs3)
  (stole_7 : stolen_eggs1 = 7)
  (total_eq_20 : stolen_eggs1 + stolen_eggs2 + stolen_eggs3 = 20) :
  false :=
by
  sorry

end NUMINAMATH_GPT_no_lighter_sentence_for_liar_l103_10365


namespace NUMINAMATH_GPT_obtuse_triangle_side_range_l103_10371

theorem obtuse_triangle_side_range {a : ℝ} (h1 : a > 3) (h2 : (a - 3)^2 < 36) : 3 < a ∧ a < 9 := 
by
  sorry

end NUMINAMATH_GPT_obtuse_triangle_side_range_l103_10371


namespace NUMINAMATH_GPT_remaining_fruit_count_l103_10362

theorem remaining_fruit_count (trees : ℕ) (fruits_per_tree : ℕ) (picked_fraction : ℚ) 
  (trees_eq : trees = 8) (fruits_per_tree_eq : fruits_per_tree = 200) (picked_fraction_eq : picked_fraction = 2/5) :
  let total_fruits := trees * fruits_per_tree
  let picked_fruits := picked_fraction * fruits_per_tree * trees
  let remaining_fruits := total_fruits - picked_fruits
  remaining_fruits = 960 := 
by 
  sorry

end NUMINAMATH_GPT_remaining_fruit_count_l103_10362


namespace NUMINAMATH_GPT_fraction_numerator_greater_than_denominator_l103_10327

theorem fraction_numerator_greater_than_denominator (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ x ≠ 5 / 3 → (8 / 11 < x ∧ x < 5 / 3) ∨ (5 / 3 < x ∧ x ≤ 3) ↔ (8 * x - 3 > 5 - 3 * x) := by
  sorry

end NUMINAMATH_GPT_fraction_numerator_greater_than_denominator_l103_10327


namespace NUMINAMATH_GPT_y_value_l103_10391

theorem y_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h_eq1 : (1 / x) + (1 / y) = 3 / 2) (h_eq2 : x * y = 9) : y = 6 :=
sorry

end NUMINAMATH_GPT_y_value_l103_10391


namespace NUMINAMATH_GPT_parabola_chord_length_eight_l103_10332

noncomputable def parabola_intersection_length (x1 x2: ℝ) (y1 y2: ℝ) : ℝ :=
  if x1 + x2 = 6 ∧ y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 then
    let A := (x1, y1)
    let B := (x2, y2)
    dist A B
  else
    0

theorem parabola_chord_length_eight :
  ∀ (x1 x2 y1 y2 : ℝ), (x1 + x2 = 6) → (y1^2 = 4 * x1) → (y2^2 = 4 * x2) →
  parabola_intersection_length x1 x2 y1 y2 = 8 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_parabola_chord_length_eight_l103_10332


namespace NUMINAMATH_GPT_smallest_k_no_real_roots_l103_10340

theorem smallest_k_no_real_roots :
  ∀ (k : ℤ), (∀ x : ℝ, 3 * x * (k * x - 5) - x^2 + 7 ≠ 0) → k ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_no_real_roots_l103_10340


namespace NUMINAMATH_GPT_original_number_of_men_l103_10328

theorem original_number_of_men (x : ℕ) (h1 : 40 * x = 60 * (x - 5)) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_men_l103_10328


namespace NUMINAMATH_GPT_first_sphere_weight_l103_10374

noncomputable def surface_area (r : ℝ) : ℝ :=
  4 * Real.pi * (r ^ 2)

noncomputable def weight (r1 r2 : ℝ) (W2 : ℝ) : ℝ :=
  let A1 := surface_area r1
  let A2 := surface_area r2
  (W2 * A1) / A2

theorem first_sphere_weight :
  let r1 := 0.15
  let r2 := 0.3
  let W2 := 32
  weight r1 r2 W2 = 8 := 
by
  sorry

end NUMINAMATH_GPT_first_sphere_weight_l103_10374


namespace NUMINAMATH_GPT_roger_final_money_l103_10381

variable (initial_money : ℕ)
variable (spent_money : ℕ)
variable (received_money : ℕ)

theorem roger_final_money (h1 : initial_money = 45) (h2 : spent_money = 20) (h3 : received_money = 46) :
  (initial_money - spent_money + received_money) = 71 :=
by
  sorry

end NUMINAMATH_GPT_roger_final_money_l103_10381


namespace NUMINAMATH_GPT_d_value_l103_10372

theorem d_value (d : ℝ) : (∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15) ↔ (d = 5) := by 
sorry

end NUMINAMATH_GPT_d_value_l103_10372


namespace NUMINAMATH_GPT_probability_x_gt_3y_l103_10323

theorem probability_x_gt_3y :
  let width := 3000
  let height := 3001
  let triangle_area := (1 / 2 : ℚ) * width * (width / 3)
  let rectangle_area := (width : ℚ) * height
  triangle_area / rectangle_area = 1500 / 9003 :=
by 
  sorry

end NUMINAMATH_GPT_probability_x_gt_3y_l103_10323


namespace NUMINAMATH_GPT_find_m_l103_10345

theorem find_m (m : ℝ) (h : 2^2 + 2 * m + 2 = 0) : m = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_l103_10345


namespace NUMINAMATH_GPT_max_cosine_value_l103_10375

theorem max_cosine_value (a b : ℝ) (h : Real.cos (a - b) = Real.cos a + Real.cos b) : 1 ≥ Real.cos a :=
sorry

end NUMINAMATH_GPT_max_cosine_value_l103_10375


namespace NUMINAMATH_GPT_num_zeros_g_l103_10329

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x > 2 then m * (x - 2) / x
  else if 0 < x ∧ x ≤ 2 then 3 * x - x^2
  else 0

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x m - 2

-- Statement to prove
theorem num_zeros_g (m : ℝ) : ∃ n : ℕ, (n = 4 ∨ n = 6) :=
sorry

end NUMINAMATH_GPT_num_zeros_g_l103_10329


namespace NUMINAMATH_GPT_a5_is_3_l103_10320

section
variable {a : ℕ → ℝ} 
variable (h_pos : ∀ n, 0 < a n)
variable (h_a1 : a 1 = 1)
variable (h_a2 : a 2 = Real.sqrt 3)
variable (h_recursive : ∀ n ≥ 2, 2 * (a n)^2 = (a (n + 1))^2 + (a (n - 1))^2)

theorem a5_is_3 : a 5 = 3 :=
  by
  sorry
end

end NUMINAMATH_GPT_a5_is_3_l103_10320


namespace NUMINAMATH_GPT_calendar_matrix_sum_l103_10310

def initial_matrix : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![5, 6, 7], 
    ![8, 9, 10], 
    ![11, 12, 13]]

def modified_matrix (m : Matrix (Fin 3) (Fin 3) ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![m 0 2, m 0 1, m 0 0], 
    ![m 1 0, m 1 1, m 1 2], 
    ![m 2 2, m 2 1, m 2 0]]

def diagonal_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 0 + m 1 1 + m 2 2

def edge_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 1 + m 0 2 + m 2 0 + m 2 1

def total_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  diagonal_sum m + edge_sum m

theorem calendar_matrix_sum :
  total_sum (modified_matrix initial_matrix) = 63 :=
by
  sorry

end NUMINAMATH_GPT_calendar_matrix_sum_l103_10310


namespace NUMINAMATH_GPT_garden_length_l103_10321

noncomputable def length_of_garden : ℝ := 300

theorem garden_length (P : ℝ) (b : ℝ) (A : ℝ) 
  (h₁ : P = 800) (h₂ : b = 100) (h₃ : A = 10000) : length_of_garden = 300 := 
by 
  sorry

end NUMINAMATH_GPT_garden_length_l103_10321


namespace NUMINAMATH_GPT_money_raised_by_full_price_tickets_l103_10352

theorem money_raised_by_full_price_tickets (f h : ℕ) (p revenue total_tickets : ℕ) 
  (full_price : p = 20) (total_cost : f * p + h * (p / 2) = revenue) 
  (ticket_count : f + h = total_tickets) (total_revenue : revenue = 2750)
  (ticket_number : total_tickets = 180) : f * p = 1900 := 
by
  sorry

end NUMINAMATH_GPT_money_raised_by_full_price_tickets_l103_10352


namespace NUMINAMATH_GPT_yellow_green_block_weight_difference_l103_10318

theorem yellow_green_block_weight_difference :
  let yellow_weight := 0.6
  let green_weight := 0.4
  yellow_weight - green_weight = 0.2 := by
  sorry

end NUMINAMATH_GPT_yellow_green_block_weight_difference_l103_10318


namespace NUMINAMATH_GPT_power_div_eq_l103_10309

theorem power_div_eq (a : ℕ) (h : 36 = 6^2) : (6^12 / 36^5) = 36 := by
  sorry

end NUMINAMATH_GPT_power_div_eq_l103_10309


namespace NUMINAMATH_GPT_prob_queen_then_diamond_is_correct_l103_10333

/-- Define the probability of drawing a Queen first and a diamond second -/
def prob_queen_then_diamond : ℚ := (3 / 52) * (13 / 51) + (1 / 52) * (12 / 51)

/-- The probability that the first card is a Queen and the second card is a diamond is 18/221 -/
theorem prob_queen_then_diamond_is_correct : prob_queen_then_diamond = 18 / 221 :=
by
  sorry

end NUMINAMATH_GPT_prob_queen_then_diamond_is_correct_l103_10333


namespace NUMINAMATH_GPT_sum_divisible_by_11_l103_10360

theorem sum_divisible_by_11 (n : ℕ) : (6^(2*n) + 3^n + 3^(n+2)) % 11 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_divisible_by_11_l103_10360


namespace NUMINAMATH_GPT_tanya_body_lotions_l103_10304

variable {F L : ℕ}  -- Number of face moisturizers (F) and body lotions (L) Tanya bought

theorem tanya_body_lotions
  (price_face_moisturizer : ℕ := 50)
  (price_body_lotion : ℕ := 60)
  (num_face_moisturizers : ℕ := 2)
  (total_spent : ℕ := 1020)
  (christy_spending_factor : ℕ := 2)
  (h_together_spent : total_spent = 3 * (num_face_moisturizers * price_face_moisturizer + L * price_body_lotion)) :
  L = 4 :=
by
  sorry

end NUMINAMATH_GPT_tanya_body_lotions_l103_10304


namespace NUMINAMATH_GPT_weight_replacement_proof_l103_10373

noncomputable def weight_of_replaced_person (increase_in_average_weight new_person_weight : ℝ) : ℝ :=
  new_person_weight - (5 * increase_in_average_weight)

theorem weight_replacement_proof (h1 : ∀ w : ℝ, increase_in_average_weight = 5.5) (h2 : new_person_weight = 95.5) :
  weight_of_replaced_person 5.5 95.5 = 68 := by
  sorry

end NUMINAMATH_GPT_weight_replacement_proof_l103_10373


namespace NUMINAMATH_GPT_vertex_of_given_function_l103_10307

noncomputable def vertex_coordinates (f : ℝ → ℝ) : ℝ × ℝ := 
  (-2, 1)  -- Prescribed coordinates for this specific function form.

def function_vertex (x : ℝ) : ℝ :=
  -3 * (x + 2) ^ 2 + 1

theorem vertex_of_given_function : 
  vertex_coordinates function_vertex = (-2, 1) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_given_function_l103_10307


namespace NUMINAMATH_GPT_value_of_2x_l103_10330

theorem value_of_2x (x y z : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_eq : 2 * x = 6 * z) (h_sum : x + y + z = 26) : 2 * x = 6 := 
by
  sorry

end NUMINAMATH_GPT_value_of_2x_l103_10330


namespace NUMINAMATH_GPT_cars_transfer_equation_l103_10346

theorem cars_transfer_equation (x : ℕ) : 100 - x = 68 + x :=
sorry

end NUMINAMATH_GPT_cars_transfer_equation_l103_10346


namespace NUMINAMATH_GPT_diameter_large_circle_correct_l103_10399

noncomputable def diameter_of_large_circle : ℝ :=
  2 * (Real.sqrt 17 + 4)

theorem diameter_large_circle_correct :
  ∃ (d : ℝ), (∀ (r : ℝ), r = Real.sqrt 17 + 4 → d = 2 * r) ∧ d = diameter_of_large_circle := by
    sorry

end NUMINAMATH_GPT_diameter_large_circle_correct_l103_10399


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l103_10392

theorem solve_system_of_inequalities {x : ℝ} :
  (|x^2 + 5 * x| < 6) ∧ (|x + 1| ≤ 1) ↔ (0 ≤ x ∧ x < 2) ∨ (4 < x ∧ x ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l103_10392


namespace NUMINAMATH_GPT_douglas_weight_proof_l103_10300

theorem douglas_weight_proof : 
  ∀ (anne_weight douglas_weight : ℕ), 
  anne_weight = 67 →
  anne_weight = douglas_weight + 15 →
  douglas_weight = 52 :=
by 
  intros anne_weight douglas_weight h1 h2 
  sorry

end NUMINAMATH_GPT_douglas_weight_proof_l103_10300


namespace NUMINAMATH_GPT_students_with_uncool_parents_correct_l103_10324

def total_students : ℕ := 30
def cool_dads : ℕ := 12
def cool_moms : ℕ := 15
def cool_both : ℕ := 9

def students_with_uncool_parents : ℕ :=
  total_students - (cool_dads + cool_moms - cool_both)

theorem students_with_uncool_parents_correct :
  students_with_uncool_parents = 12 := by
  sorry

end NUMINAMATH_GPT_students_with_uncool_parents_correct_l103_10324


namespace NUMINAMATH_GPT_unique_integer_sequence_l103_10382

theorem unique_integer_sequence :
  ∃ a : ℕ → ℤ, a 1 = 1 ∧ a 2 > 1 ∧ ∀ n ≥ 1, (a (n + 1))^3 + 1 = a n * a (n + 2) :=
sorry

end NUMINAMATH_GPT_unique_integer_sequence_l103_10382


namespace NUMINAMATH_GPT_three_digit_numbers_with_2_without_4_l103_10376

theorem three_digit_numbers_with_2_without_4 : 
  ∃ n : Nat, n = 200 ∧
  (∀ x : Nat, 100 ≤ x ∧ x ≤ 999 → 
      (∃ d1 d2 d3,
        d1 ≠ 0 ∧ 
        x = d1 * 100 + d2 * 10 + d3 ∧ 
        (d1 ≠ 4 ∧ d2 ≠ 4 ∧ d3 ≠ 4) ∧
        (d1 = 2 ∨ d2 = 2 ∨ d3 = 2))) :=
sorry

end NUMINAMATH_GPT_three_digit_numbers_with_2_without_4_l103_10376


namespace NUMINAMATH_GPT_triangle_PQR_QR_length_l103_10341

-- Define the given conditions as a Lean statement
theorem triangle_PQR_QR_length 
  (P Q R : ℝ) -- Angles in the triangle PQR in radians
  (PQ QR PR : ℝ) -- Lengths of the sides of the triangle PQR
  (h1 : Real.cos (2 * P - Q) + Real.sin (P + 2 * Q) = 1) 
  (h2 : PQ = 5)
  (h3 : PQ + QR + PR = 12)
  : QR = 3.5 := 
  sorry -- proof omitted

end NUMINAMATH_GPT_triangle_PQR_QR_length_l103_10341


namespace NUMINAMATH_GPT_balls_per_bag_l103_10303

theorem balls_per_bag (total_balls bags_used: Nat) (h1: total_balls = 36) (h2: bags_used = 9) : total_balls / bags_used = 4 := by
  sorry

end NUMINAMATH_GPT_balls_per_bag_l103_10303


namespace NUMINAMATH_GPT_transformed_sum_l103_10396

theorem transformed_sum (n : ℕ) (y : Fin n → ℝ) (s : ℝ) (h : s = (Finset.univ.sum (fun i => y i))) :
  Finset.univ.sum (fun i => 3 * (y i) + 30) = 3 * s + 30 * n :=
by 
  sorry

end NUMINAMATH_GPT_transformed_sum_l103_10396


namespace NUMINAMATH_GPT_profit_per_meal_A_and_B_l103_10390

theorem profit_per_meal_A_and_B (x y : ℝ) 
  (h1 : x + 2 * y = 35) 
  (h2 : 2 * x + 3 * y = 60) : 
  x = 15 ∧ y = 10 :=
sorry

end NUMINAMATH_GPT_profit_per_meal_A_and_B_l103_10390


namespace NUMINAMATH_GPT_second_derivative_at_pi_over_3_l103_10308

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) * (Real.cos x)

theorem second_derivative_at_pi_over_3 : 
  (deriv (deriv f)) (Real.pi / 3) = -1 :=
  sorry

end NUMINAMATH_GPT_second_derivative_at_pi_over_3_l103_10308


namespace NUMINAMATH_GPT_andy_paint_total_l103_10357

-- Define the given ratio condition and green paint usage
def paint_ratio (blue green white : ℕ) : Prop :=
  blue / green = 1 / 2 ∧ white / green = 5 / 2

def green_paint_used (green : ℕ) : Prop :=
  green = 6

-- Define the proof goal: total paint used
def total_paint_used (blue green white : ℕ) : ℕ :=
  blue + green + white

-- The statement to be proved
theorem andy_paint_total (blue green white : ℕ)
  (h_ratio : paint_ratio blue green white)
  (h_green : green_paint_used green) :
  total_paint_used blue green white = 24 :=
  sorry

end NUMINAMATH_GPT_andy_paint_total_l103_10357
