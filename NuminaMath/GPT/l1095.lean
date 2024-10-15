import Mathlib

namespace NUMINAMATH_GPT_ab_cd_value_l1095_109536

theorem ab_cd_value (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = -3) 
  (h3 : a + c + d = 10) 
  (h4 : b + c + d = -1) : 
  ab + cd = -346 / 9 :=
by 
  sorry

end NUMINAMATH_GPT_ab_cd_value_l1095_109536


namespace NUMINAMATH_GPT_nancy_pics_uploaded_l1095_109589

theorem nancy_pics_uploaded (a b n : ℕ) (h₁ : a = 11) (h₂ : b = 8) (h₃ : n = 5) : a + b * n = 51 := 
by 
  sorry

end NUMINAMATH_GPT_nancy_pics_uploaded_l1095_109589


namespace NUMINAMATH_GPT_trapezium_other_parallel_side_l1095_109574

theorem trapezium_other_parallel_side (a b h : ℝ) (area : ℝ) (h_area : area = (1 / 2) * (a + b) * h) (h_a : a = 18) (h_h : h = 20) (h_area_val : area = 380) :
  b = 20 :=
by 
  sorry

end NUMINAMATH_GPT_trapezium_other_parallel_side_l1095_109574


namespace NUMINAMATH_GPT_discountIs50Percent_l1095_109567

noncomputable def promotionalPrice (originalPrice : ℝ) : ℝ :=
  (2/3) * originalPrice

noncomputable def finalPrice (originalPrice : ℝ) : ℝ :=
  0.75 * promotionalPrice originalPrice

theorem discountIs50Percent (originalPrice : ℝ) (h₁ : originalPrice > 0) :
  finalPrice originalPrice = 0.5 * originalPrice := by
  sorry

end NUMINAMATH_GPT_discountIs50Percent_l1095_109567


namespace NUMINAMATH_GPT_base_nine_to_base_ten_conversion_l1095_109529

theorem base_nine_to_base_ten_conversion : 
  (2 * 9^3 + 8 * 9^2 + 4 * 9^1 + 7 * 9^0 = 2149) := 
by 
  sorry

end NUMINAMATH_GPT_base_nine_to_base_ten_conversion_l1095_109529


namespace NUMINAMATH_GPT_inequality_proof_l1095_109527

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) : x^12 - y^12 + 2 * x^6 * y^6 ≤ (Real.pi / 2) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1095_109527


namespace NUMINAMATH_GPT_hexagon_area_eq_l1095_109544

theorem hexagon_area_eq (s t : ℝ) (hs : s^2 = 16) (heq : 4 * s = 6 * t) :
  6 * (t^2 * (Real.sqrt 3) / 4) = 32 * (Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_GPT_hexagon_area_eq_l1095_109544


namespace NUMINAMATH_GPT_distance_from_plate_to_bottom_edge_l1095_109507

theorem distance_from_plate_to_bottom_edge (d : ℝ) : 
  (10 + d + 63 = 20 + d + 53) :=
by
  -- The proof can be completed here.
  sorry

end NUMINAMATH_GPT_distance_from_plate_to_bottom_edge_l1095_109507


namespace NUMINAMATH_GPT_town_population_growth_is_62_percent_l1095_109547

noncomputable def population_growth_proof : ℕ := 
  let p := 22
  let p_square := p * p
  let pop_1991 := p_square
  let pop_2001 := pop_1991 + 150
  let pop_2011 := pop_2001 + 150
  let k := 28  -- Given that 784 = 28^2
  let pop_2011_is_perfect_square := k * k = pop_2011
  let percentage_increase := ((pop_2011 - pop_1991) * 100) / pop_1991
  if pop_2011_is_perfect_square then percentage_increase 
  else 0

theorem town_population_growth_is_62_percent :
  population_growth_proof = 62 :=
by
  sorry

end NUMINAMATH_GPT_town_population_growth_is_62_percent_l1095_109547


namespace NUMINAMATH_GPT_marbles_in_larger_container_l1095_109521

-- Defining the conditions
def volume1 := 24 -- in cm³
def marbles1 := 30 -- number of marbles in the first container
def volume2 := 72 -- in cm³

-- Statement of the theorem
theorem marbles_in_larger_container : (marbles1 / volume1 : ℚ) * volume2 = 90 := by
  sorry

end NUMINAMATH_GPT_marbles_in_larger_container_l1095_109521


namespace NUMINAMATH_GPT_correct_calculation_l1095_109595

theorem correct_calculation (m n : ℝ) : -m^2 * n - 2 * m^2 * n = -3 * m^2 * n :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1095_109595


namespace NUMINAMATH_GPT_solve_for_a_l1095_109524

def f (x : ℝ) : ℝ := x^2 + 10
def g (x : ℝ) : ℝ := x^2 - 6

theorem solve_for_a (a : ℝ) (h : a > 0) (h1 : f (g a) = 18) : a = Real.sqrt (2 * Real.sqrt 2 + 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1095_109524


namespace NUMINAMATH_GPT_max_goods_purchased_l1095_109532

theorem max_goods_purchased (initial_spend : ℕ) (reward_rate : ℕ → ℕ → ℕ) (continuous_reward : Prop) :
  initial_spend = 7020 →
  (∀ x y, reward_rate x y = (x / y) * 20) →
  continuous_reward →
  initial_spend + reward_rate initial_spend 100 + reward_rate (reward_rate initial_spend 100) 100 + 
  reward_rate (reward_rate (reward_rate initial_spend 100) 100) 100 = 8760 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_max_goods_purchased_l1095_109532


namespace NUMINAMATH_GPT_melina_age_l1095_109502

theorem melina_age (A M : ℕ) (alma_score : ℕ := 40) 
    (h1 : A + M = 2 * alma_score) 
    (h2 : M = 3 * A) : 
    M = 60 :=
by 
  sorry

end NUMINAMATH_GPT_melina_age_l1095_109502


namespace NUMINAMATH_GPT_find_natural_number_l1095_109539

variable {A : ℕ}

theorem find_natural_number (h1 : A = 8 * 2 + 7) : A = 23 :=
sorry

end NUMINAMATH_GPT_find_natural_number_l1095_109539


namespace NUMINAMATH_GPT_avg_chem_math_l1095_109509

-- Given conditions
variables (P C M : ℕ)
axiom total_marks : P + C + M = P + 130

-- The proof problem
theorem avg_chem_math : (C + M) / 2 = 65 :=
by sorry

end NUMINAMATH_GPT_avg_chem_math_l1095_109509


namespace NUMINAMATH_GPT_polygon_sides_l1095_109541

theorem polygon_sides (n : ℕ) (a1 d : ℝ) (h1 : a1 = 100) (h2 : d = 10)
  (h3 : ∀ k, 1 ≤ k ∧ k ≤ n → a1 + (k - 1) * d < 180) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1095_109541


namespace NUMINAMATH_GPT_solve_quadratic_eq1_solve_quadratic_eq2_complete_square_l1095_109581

theorem solve_quadratic_eq1 : ∀ x : ℝ, 2 * x^2 + 5 * x + 3 = 0 → (x = -3/2 ∨ x = -1) :=
by
  intro x
  intro h
  sorry

theorem solve_quadratic_eq2_complete_square : ∀ x : ℝ, x^2 - 2 * x - 1 = 0 → (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq1_solve_quadratic_eq2_complete_square_l1095_109581


namespace NUMINAMATH_GPT_common_ratio_is_2_l1095_109587

noncomputable def common_ratio_of_increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n+1) = a n * q) ∧ (∀ m n, m < n → a m < a n)

theorem common_ratio_is_2
  (a : ℕ → ℝ) (q : ℝ)
  (hgeo : common_ratio_of_increasing_geometric_sequence a q)
  (h1 : a 1 + a 5 = 17)
  (h2 : a 2 * a 4 = 16) :
  q = 2 :=
sorry

end NUMINAMATH_GPT_common_ratio_is_2_l1095_109587


namespace NUMINAMATH_GPT_a_beats_b_by_32_meters_l1095_109526

-- Define the known conditions.
def distance_a_in_t : ℕ := 224 -- Distance A runs in 28 seconds
def time_a : ℕ := 28 -- Time A takes to run 224 meters
def distance_b_in_t : ℕ := 224 -- Distance B runs in 32 seconds
def time_b : ℕ := 32 -- Time B takes to run 224 meters

-- Define the speeds.
def speed_a : ℕ := distance_a_in_t / time_a
def speed_b : ℕ := distance_b_in_t / time_b

-- Define the distances each runs in 32 seconds.
def distance_a_in_32_sec : ℕ := speed_a * 32
def distance_b_in_32_sec : ℕ := speed_b * 32

-- The proof statement
theorem a_beats_b_by_32_meters :
  distance_a_in_32_sec - distance_b_in_32_sec = 32 := 
sorry

end NUMINAMATH_GPT_a_beats_b_by_32_meters_l1095_109526


namespace NUMINAMATH_GPT_find_a2_b2_c2_l1095_109549

-- Define the roots, sum of the roots, sum of the product of the roots taken two at a time, and product of the roots
variables {a b c : ℝ}
variable (h_roots : a = b ∧ b = c)
variable (h_sum : a + b + c = 12)
variable (h_sum_products : a * b + b * c + a * c = 47)
variable (h_product : a * b * c = 30)

-- State the theorem
theorem find_a2_b2_c2 : (a^2 + b^2 + c^2) = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a2_b2_c2_l1095_109549


namespace NUMINAMATH_GPT_magic_8_ball_probability_l1095_109548

theorem magic_8_ball_probability :
  let num_questions := 7
  let num_positive := 3
  let positive_probability := 3 / 7
  let negative_probability := 4 / 7
  let binomial_coefficient := Nat.choose num_questions num_positive
  let total_probability := binomial_coefficient * (positive_probability ^ num_positive) * (negative_probability ^ (num_questions - num_positive))
  total_probability = 242112 / 823543 :=
by
  sorry

end NUMINAMATH_GPT_magic_8_ball_probability_l1095_109548


namespace NUMINAMATH_GPT_minimum_value_l1095_109593

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_eq : 2 * m + n = 4) : 
  ∃ (x : ℝ), (x = 2) ∧ (∀ (p q : ℝ), q > 0 → p > 0 → 2 * p + q = 4 → x ≤ (1 / p + 2 / q)) := 
sorry

end NUMINAMATH_GPT_minimum_value_l1095_109593


namespace NUMINAMATH_GPT_cone_volume_l1095_109503

theorem cone_volume (V_cyl : ℝ) (d : ℝ) (π : ℝ) (V_cyl_eq : V_cyl = 81 * π) (h_eq : 2 * (d / 2) = 2 * d) :
  ∃ (V_cone : ℝ), V_cone = 27 * π * (6 ^ (1/3)) :=
by 
  sorry

end NUMINAMATH_GPT_cone_volume_l1095_109503


namespace NUMINAMATH_GPT_problem_statement_l1095_109568

/-- 
  Theorem: If the solution set of the inequality (ax-1)(x+2) > 0 is -3 < x < -2, 
  then a equals -1/3 
--/
theorem problem_statement (a : ℝ) :
  (forall x, (ax-1)*(x+2) > 0 -> -3 < x ∧ x < -2) → a = -1/3 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1095_109568


namespace NUMINAMATH_GPT_remainder_of_x_500_div_x2_plus_1_x2_minus_1_l1095_109528

theorem remainder_of_x_500_div_x2_plus_1_x2_minus_1 :
  (x^500) % ((x^2 + 1) * (x^2 - 1)) = 1 :=
sorry

end NUMINAMATH_GPT_remainder_of_x_500_div_x2_plus_1_x2_minus_1_l1095_109528


namespace NUMINAMATH_GPT_students_per_group_l1095_109566

theorem students_per_group (total_students not_picked_groups groups : ℕ) (h₁ : total_students = 65) (h₂ : not_picked_groups = 17) (h₃ : groups = 8) :
  (total_students - not_picked_groups) / groups = 6 := by
  sorry

end NUMINAMATH_GPT_students_per_group_l1095_109566


namespace NUMINAMATH_GPT_alpha_beta_value_l1095_109576

noncomputable def alpha_beta_sum : ℝ := 75

theorem alpha_beta_value (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h : |Real.sin α - (1 / 2)| + Real.sqrt (Real.tan β - 1) = 0) :
  α + β = α_beta_sum := 
  sorry

end NUMINAMATH_GPT_alpha_beta_value_l1095_109576


namespace NUMINAMATH_GPT_range_of_m_plus_n_l1095_109571

theorem range_of_m_plus_n (f : ℝ → ℝ) (n m : ℝ)
  (h_f_def : ∀ x, f x = x^2 + n * x + m)
  (h_non_empty : ∃ x, f x = 0 ∧ f (f x) = 0)
  (h_condition : ∀ x, f x = 0 ↔ f (f x) = 0) :
  0 < m + n ∧ m + n < 4 :=
by {
  -- Proof needed here; currently skipped
  sorry
}

end NUMINAMATH_GPT_range_of_m_plus_n_l1095_109571


namespace NUMINAMATH_GPT_ocean_depth_350_l1095_109594

noncomputable def depth_of_ocean (total_height : ℝ) (volume_ratio_above_water : ℝ) : ℝ :=
  let volume_ratio_below_water := 1 - volume_ratio_above_water
  let height_below_water := (volume_ratio_below_water^(1 / 3)) * total_height
  total_height - height_below_water

theorem ocean_depth_350 :
  depth_of_ocean 10000 (1 / 10) = 350 :=
by
  sorry

end NUMINAMATH_GPT_ocean_depth_350_l1095_109594


namespace NUMINAMATH_GPT_greatest_integer_for_prime_abs_expression_l1095_109570

open Int

-- Define the quadratic expression and the prime condition
def quadratic_expression (x : ℤ) : ℤ := 6 * x^2 - 47 * x + 15

-- Statement that |quadratic_expression x| is prime
def is_prime_quadratic_expression (x : ℤ) : Prop :=
  Prime (abs (quadratic_expression x))

-- Prove that the greatest integer x such that |quadratic_expression x| is prime is 8
theorem greatest_integer_for_prime_abs_expression :
  ∃ (x : ℤ), is_prime_quadratic_expression x ∧ (∀ (y : ℤ), is_prime_quadratic_expression y → y ≤ x) → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_for_prime_abs_expression_l1095_109570


namespace NUMINAMATH_GPT_final_acid_concentration_l1095_109506

def volume1 : ℝ := 2
def concentration1 : ℝ := 0.40
def volume2 : ℝ := 3
def concentration2 : ℝ := 0.60

theorem final_acid_concentration :
  ((concentration1 * volume1 + concentration2 * volume2) / (volume1 + volume2)) = 0.52 :=
by
  sorry

end NUMINAMATH_GPT_final_acid_concentration_l1095_109506


namespace NUMINAMATH_GPT_sequence_contains_infinitely_many_powers_of_two_l1095_109501

theorem sequence_contains_infinitely_many_powers_of_two (a : ℕ → ℕ) (b : ℕ → ℕ) : 
  (∃ a1, a1 % 5 ≠ 0 ∧ a 0 = a1) →
  (∀ n : ℕ, a (n + 1) = a n + b n) →
  (∀ n : ℕ, b n = a n % 10) →
  (∃ n : ℕ, ∃ k : ℕ, 2^k = a n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_contains_infinitely_many_powers_of_two_l1095_109501


namespace NUMINAMATH_GPT_swimming_problem_l1095_109522

/-- The swimming problem where a man swims downstream 30 km and upstream a certain distance 
    taking 6 hours each time. Given his speed in still water is 4 km/h, we aim to prove the 
    distance swam upstream is 18 km. -/
theorem swimming_problem 
  (V_m : ℝ) (Distance_downstream : ℝ) (Time_downstream : ℝ) (Time_upstream : ℝ) 
  (Distance_upstream : ℝ) (V_s : ℝ)
  (h1 : V_m = 4)
  (h2 : Distance_downstream = 30)
  (h3 : Time_downstream = 6)
  (h4 : Time_upstream = 6)
  (h5 : V_m + V_s = Distance_downstream / Time_downstream)
  (h6 : V_m - V_s = Distance_upstream / Time_upstream) :
  Distance_upstream = 18 := 
sorry

end NUMINAMATH_GPT_swimming_problem_l1095_109522


namespace NUMINAMATH_GPT_final_selling_price_correct_l1095_109511

noncomputable def purchase_price_inr : ℝ := 8000
noncomputable def depreciation_rate_annual : ℝ := 0.10
noncomputable def profit_rate : ℝ := 0.10
noncomputable def discount_rate : ℝ := 0.05
noncomputable def sales_tax_rate : ℝ := 0.12
noncomputable def exchange_rate_at_purchase : ℝ := 80
noncomputable def exchange_rate_at_selling : ℝ := 75

noncomputable def depreciated_value_after_2_years (initial_value : ℝ) : ℝ :=
  initial_value * (1 - depreciation_rate_annual) * (1 - depreciation_rate_annual)

noncomputable def marked_price (initial_value : ℝ) : ℝ :=
  initial_value * (1 + profit_rate)

noncomputable def selling_price_before_tax (marked_price : ℝ) : ℝ :=
  marked_price * (1 - discount_rate)

noncomputable def final_selling_price_inr (selling_price_before_tax : ℝ) : ℝ :=
  selling_price_before_tax * (1 + sales_tax_rate)

noncomputable def final_selling_price_usd (final_selling_price_inr : ℝ) : ℝ :=
  final_selling_price_inr / exchange_rate_at_selling

theorem final_selling_price_correct :
  final_selling_price_usd (final_selling_price_inr (selling_price_before_tax (marked_price purchase_price_inr))) = 124.84 := 
sorry

end NUMINAMATH_GPT_final_selling_price_correct_l1095_109511


namespace NUMINAMATH_GPT_range_of_a_l1095_109543

-- Define proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

-- Define the main theorem
theorem range_of_a (a : ℝ) : (p a ∧ ¬q a) → -1 ≤ a ∧ a < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1095_109543


namespace NUMINAMATH_GPT_mandy_more_cinnamon_l1095_109599

def cinnamon : ℝ := 0.67
def nutmeg : ℝ := 0.5

theorem mandy_more_cinnamon : cinnamon - nutmeg = 0.17 :=
by
  sorry

end NUMINAMATH_GPT_mandy_more_cinnamon_l1095_109599


namespace NUMINAMATH_GPT_t_shirts_per_package_l1095_109518

theorem t_shirts_per_package (total_t_shirts : ℕ) (total_packages : ℕ) (h1 : total_t_shirts = 39) (h2 : total_packages = 3) : total_t_shirts / total_packages = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_t_shirts_per_package_l1095_109518


namespace NUMINAMATH_GPT_total_ice_cream_amount_l1095_109596

theorem total_ice_cream_amount (ice_cream_friday ice_cream_saturday : ℝ) 
  (h1 : ice_cream_friday = 3.25)
  (h2 : ice_cream_saturday = 0.25) : 
  ice_cream_friday + ice_cream_saturday = 3.50 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_total_ice_cream_amount_l1095_109596


namespace NUMINAMATH_GPT_parabola_expression_l1095_109577

theorem parabola_expression :
  ∃ a b : ℝ, (∀ x : ℝ, a * x^2 + b * x - 5 = 0 → (x = -1 ∨ x = 5)) ∧ (a * (-1)^2 + b * (-1) - 5 = 0) ∧ (a * 5^2 + b * 5 - 5 = 0) ∧ (a * 1 - 4 = 1) :=
sorry

end NUMINAMATH_GPT_parabola_expression_l1095_109577


namespace NUMINAMATH_GPT_value_of_a_l1095_109598

theorem value_of_a (a : ℝ) (h : a > 0 ∧ a ≠ 1 ∧ (∃ (y : ℝ), y = 2 ∧ 9 = a ^ y)) : a = 3 := 
  by sorry

end NUMINAMATH_GPT_value_of_a_l1095_109598


namespace NUMINAMATH_GPT_inequality_holds_for_a_in_interval_l1095_109510

theorem inequality_holds_for_a_in_interval:
  (∀ x y : ℝ, 
     2 ≤ x ∧ x ≤ 3 ∧ 3 ≤ y ∧ y ≤ 4 → (3*x - 2*y - a) * (3*x - 2*y - a^2) ≤ 0) ↔ a ∈ Set.Iic (-4) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_a_in_interval_l1095_109510


namespace NUMINAMATH_GPT_axis_angle_set_l1095_109559

def is_x_axis_angle (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi
def is_y_axis_angle (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi + Real.pi / 2

def is_axis_angle (α : ℝ) : Prop := ∃ n : ℤ, α = (n * Real.pi) / 2

theorem axis_angle_set : 
  (∀ α : ℝ, is_x_axis_angle α ∨ is_y_axis_angle α ↔ is_axis_angle α) :=
by 
  sorry

end NUMINAMATH_GPT_axis_angle_set_l1095_109559


namespace NUMINAMATH_GPT_neg_real_root_condition_l1095_109531

theorem neg_real_root_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ (0 < a ∧ a ≤ 1) ∨ (a < 0) :=
by
  sorry

end NUMINAMATH_GPT_neg_real_root_condition_l1095_109531


namespace NUMINAMATH_GPT_positive_number_is_25_over_9_l1095_109586

variable (a : ℚ) (x : ℚ)

theorem positive_number_is_25_over_9 
  (h1 : 2 * a - 1 = -a + 3)
  (h2 : ∃ r : ℚ, r^2 = x ∧ (r = 2 * a - 1 ∨ r = -a + 3)) : 
  x = 25 / 9 := 
by
  sorry

end NUMINAMATH_GPT_positive_number_is_25_over_9_l1095_109586


namespace NUMINAMATH_GPT_incorrect_statement_l1095_109538

-- Define the operation (x * y)
def op (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

-- State the theorem to show the incorrectness of the given statement
theorem incorrect_statement (x y z : ℝ) : op x (y + z) ≠ op x y + op x z :=
  sorry

end NUMINAMATH_GPT_incorrect_statement_l1095_109538


namespace NUMINAMATH_GPT_horizontal_length_circumference_l1095_109592

noncomputable def ratio := 16 / 9
noncomputable def diagonal := 32
noncomputable def computed_length := 32 * 16 / (Real.sqrt 337)
noncomputable def computed_perimeter := 2 * (32 * 16 / (Real.sqrt 337) + 32 * 9 / (Real.sqrt 337))

theorem horizontal_length 
  (ratio : ℝ := 16 / 9) (diagonal : ℝ := 32) : 
  32 * 16 / (Real.sqrt 337) = 512 / (Real.sqrt 337) :=
by sorry

theorem circumference 
  (ratio : ℝ := 16 / 9) (diagonal : ℝ := 32) : 
  2 * (32 * 16 / (Real.sqrt 337) + 32 * 9 / (Real.sqrt 337)) = 1600 / (Real.sqrt 337) :=
by sorry

end NUMINAMATH_GPT_horizontal_length_circumference_l1095_109592


namespace NUMINAMATH_GPT_problem1_problem2_l1095_109508

namespace MathProof

def f (x : ℝ) (m : ℝ) : ℝ := x^2 - (m-1)*x + 2*m

theorem problem1 (m : ℝ) :
  (∀ x, 0 < x → f x m > 0) → -2 * Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2 * Real.sqrt 6 + 5 :=
sorry

theorem problem2 (m : ℝ) :
  (∃ x, 0 < x ∧ x < 1 ∧ f x m = 0) → -2 < m ∧ m < 0 :=
sorry

end MathProof

end NUMINAMATH_GPT_problem1_problem2_l1095_109508


namespace NUMINAMATH_GPT_find_d_l1095_109584

-- Conditions
variables (c d : ℝ)
axiom ratio_cond : c / d = 4
axiom eq_cond : c = 20 - 6 * d

theorem find_d : d = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1095_109584


namespace NUMINAMATH_GPT_max_area_proof_l1095_109535

-- Define the original curve
def original_curve (x : ℝ) : ℝ := x^2 + x - 2

-- Reflective symmetry curve about point (p, 2p)
def transformed_curve (p x : ℝ) : ℝ := -x^2 + (4 * p + 1) * x - 4 * p^2 + 2 * p + 2

-- Intersection conditions
def intersecting_curves (p x : ℝ) : Prop :=
original_curve x = transformed_curve p x

-- Range for valid p values
def valid_p (p : ℝ) : Prop := -1 ≤ p ∧ p ≤ 2

-- Prove the problem statement which involves ensuring the curves intersect in the range
theorem max_area_proof :
  ∀ (p : ℝ), valid_p p → ∀ (x : ℝ), intersecting_curves p x →
  ∃ (A : ℝ), A = abs (original_curve x - transformed_curve p x) :=
by
  intros p hp x hx
  sorry

end NUMINAMATH_GPT_max_area_proof_l1095_109535


namespace NUMINAMATH_GPT_doctors_assignment_l1095_109552

theorem doctors_assignment :
  ∃ (assignments : Finset (Fin 3 → Finset (Fin 5))),
    (∀ h ∈ assignments, (∀ i, ∃ j ∈ h i, True) ∧
      ¬(∃ i j, (A ∈ h i ∧ B ∈ h j ∨ A ∈ h j ∧ B ∈ h i)) ∧
      ¬(∃ i j, (C ∈ h i ∧ D ∈ h j ∨ C ∈ h j ∧ D ∈ h i))) ∧
    assignments.card = 84 :=
sorry

end NUMINAMATH_GPT_doctors_assignment_l1095_109552


namespace NUMINAMATH_GPT_ratio_of_Carla_to_Cosima_l1095_109545

variables (C M : ℝ)

-- Natasha has 3 times as much money as Carla
axiom h1 : 3 * C = 60

-- Carla has the same amount of money as Cosima
axiom h2 : C = M

-- Prove: the ratio of Carla's money to Cosima's money is 1:1
theorem ratio_of_Carla_to_Cosima : C / M = 1 :=
by sorry

end NUMINAMATH_GPT_ratio_of_Carla_to_Cosima_l1095_109545


namespace NUMINAMATH_GPT_four_planes_divide_space_into_fifteen_parts_l1095_109585

-- Define the function that calculates the number of parts given the number of planes.
def parts_divided_by_planes (x : ℕ) : ℕ :=
  (x^3 + 5 * x + 6) / 6

-- Prove that four planes divide the space into 15 parts.
theorem four_planes_divide_space_into_fifteen_parts : parts_divided_by_planes 4 = 15 :=
by sorry

end NUMINAMATH_GPT_four_planes_divide_space_into_fifteen_parts_l1095_109585


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1095_109588

theorem quadratic_no_real_roots :
  ¬ (∃ x : ℝ, x^2 - 2 * x + 3 = 0) ∧
  (∃ x1 x2 : ℝ, x1^2 - 3 * x1 - 1 = 0) ∧ (x2^2 - 3 * x2 = 0) ∧
  ∃ y : ℝ, y^2 - 2 * y + 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1095_109588


namespace NUMINAMATH_GPT_club_membership_l1095_109563

def total_people_in_club (T B TB N : ℕ) : ℕ :=
  T + B - TB + N

theorem club_membership : total_people_in_club 138 255 94 11 = 310 := by
  sorry

end NUMINAMATH_GPT_club_membership_l1095_109563


namespace NUMINAMATH_GPT_total_roses_l1095_109597

theorem total_roses (a : ℕ) (x y k : ℕ) (h1 : 300 ≤ a) (h2 : a ≤ 400)
  (h3 : a = 21 * x + 13) (h4 : a = 15 * y - 8) (h5 : a + 8 = 105 * k) :
  a = 307 :=
sorry

end NUMINAMATH_GPT_total_roses_l1095_109597


namespace NUMINAMATH_GPT_john_total_trip_cost_l1095_109540

noncomputable def total_trip_cost
  (hotel_nights : ℕ) 
  (hotel_rate_per_night : ℝ) 
  (discount : ℝ) 
  (loyal_customer_discount_rate : ℝ) 
  (service_tax_rate : ℝ) 
  (room_service_cost_per_day : ℝ) 
  (cab_cost_per_ride : ℝ) : ℝ :=
  let hotel_cost := hotel_nights * hotel_rate_per_night
  let cost_after_discount := hotel_cost - discount
  let loyal_customer_discount := loyal_customer_discount_rate * cost_after_discount
  let cost_after_loyalty_discount := cost_after_discount - loyal_customer_discount
  let service_tax := service_tax_rate * cost_after_loyalty_discount
  let final_hotel_cost := cost_after_loyalty_discount + service_tax
  let room_service_cost := hotel_nights * room_service_cost_per_day
  let cab_cost := cab_cost_per_ride * 2 * hotel_nights
  final_hotel_cost + room_service_cost + cab_cost

theorem john_total_trip_cost : total_trip_cost 3 250 100 0.10 0.12 50 30 = 985.20 :=
by 
  -- We are skipping the proof but our focus is the statement
  sorry

end NUMINAMATH_GPT_john_total_trip_cost_l1095_109540


namespace NUMINAMATH_GPT_part_a_part_b_l1095_109561

-- Definition for bishops not attacking each other
def bishops_safe (positions : List (ℕ × ℕ)) : Prop :=
  ∀ (b1 b2 : ℕ × ℕ), b1 ∈ positions → b2 ∈ positions → b1 ≠ b2 → 
    (b1.1 + b1.2 ≠ b2.1 + b2.2) ∧ (b1.1 - b1.2 ≠ b2.1 - b2.2)

-- Part (a): 14 bishops on an 8x8 chessboard such that no two attack each other
theorem part_a : ∃ (positions : List (ℕ × ℕ)), positions.length = 14 ∧ bishops_safe positions := 
by
  sorry

-- Part (b): It is impossible to place 15 bishops on an 8x8 chessboard without them attacking each other
theorem part_b : ¬ ∃ (positions : List (ℕ × ℕ)), positions.length = 15 ∧ bishops_safe positions :=
by 
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1095_109561


namespace NUMINAMATH_GPT_quadratic_two_real_roots_quadratic_no_real_roots_l1095_109591

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 - (4 * k - 1) * x + (2 * k^2 - 1) = 0) → 
  k ≤ 9 / 8 :=
by
  sorry

theorem quadratic_no_real_roots (k : ℝ) :
  ¬ (∃ x : ℝ, 2 * x^2 - (4 * k - 1) * x + (2 * k^2 - 1) = 0) → 
  k > 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_two_real_roots_quadratic_no_real_roots_l1095_109591


namespace NUMINAMATH_GPT_triangle_angle_C_and_equilateral_l1095_109534

variables (a b c A B C : ℝ)
variables (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variables (h_perpendicular : (a + c) * (a - c) + (b - a) * b = 0)
variables (h_sine : 2 * (Real.sin (A / 2)) ^ 2 + 2 * (Real.sin (B / 2)) ^ 2 = 1)

theorem triangle_angle_C_and_equilateral (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
                                         (h_perpendicular : (a + c) * (a - c) + (b - a) * b = 0)
                                         (h_sine : 2 * (Real.sin (A / 2)) ^ 2 + 2 * (Real.sin (B / 2)) ^ 2 = 1) :
  C = π / 3 ∧ A = π / 3 ∧ B = π / 3 :=
sorry

end NUMINAMATH_GPT_triangle_angle_C_and_equilateral_l1095_109534


namespace NUMINAMATH_GPT_complement_A_union_B_l1095_109513

def is_positive_integer_less_than_9 (n : ℕ) : Prop :=
  n > 0 ∧ n < 9

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

noncomputable def U := {n : ℕ | is_positive_integer_less_than_9 n}
noncomputable def A := {n ∈ U | is_odd n}
noncomputable def B := {n ∈ U | is_multiple_of_3 n}

theorem complement_A_union_B :
  (U \ (A ∪ B)) = {2, 4, 8} :=
sorry

end NUMINAMATH_GPT_complement_A_union_B_l1095_109513


namespace NUMINAMATH_GPT_correct_algorithm_option_l1095_109512

def OptionA := ("Sequential structure", "Flow structure", "Loop structure")
def OptionB := ("Sequential structure", "Conditional structure", "Nested structure")
def OptionC := ("Sequential structure", "Conditional structure", "Loop structure")
def OptionD := ("Flow structure", "Conditional structure", "Loop structure")

-- The correct structures of an algorithm are sequential, conditional, and loop.
def algorithm_structures := ("Sequential structure", "Conditional structure", "Loop structure")

theorem correct_algorithm_option : algorithm_structures = OptionC := 
by 
  -- This would be proven by logic and checking the options; omitted here with 'sorry'
  sorry

end NUMINAMATH_GPT_correct_algorithm_option_l1095_109512


namespace NUMINAMATH_GPT_James_age_after_x_years_l1095_109557

variable (x : ℕ)
variable (Justin Jessica James : ℕ)

-- Define the conditions
theorem James_age_after_x_years 
  (H1 : Justin = 26) 
  (H2 : Jessica = Justin + 6) 
  (H3 : James = Jessica + 7)
  (H4 : James + 5 = 44) : 
  James + x = 39 + x := 
by 
  -- proof steps go here 
  sorry

end NUMINAMATH_GPT_James_age_after_x_years_l1095_109557


namespace NUMINAMATH_GPT_divisible_by_six_l1095_109542

theorem divisible_by_six (n : ℕ) (hn : n > 0) (h : 72 ∣ n^2) : 6 ∣ n :=
sorry

end NUMINAMATH_GPT_divisible_by_six_l1095_109542


namespace NUMINAMATH_GPT_students_liked_strawberries_l1095_109560

theorem students_liked_strawberries : 
  let total_students := 450 
  let students_oranges := 70 
  let students_pears := 120 
  let students_apples := 147 
  let students_strawberries := total_students - (students_oranges + students_pears + students_apples)
  students_strawberries = 113 :=
by
  sorry

end NUMINAMATH_GPT_students_liked_strawberries_l1095_109560


namespace NUMINAMATH_GPT_moles_of_HCl_used_l1095_109579

theorem moles_of_HCl_used (moles_amyl_alcohol : ℕ) (moles_product : ℕ) : 
  moles_amyl_alcohol = 2 ∧ moles_product = 2 → moles_amyl_alcohol = 2 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_HCl_used_l1095_109579


namespace NUMINAMATH_GPT_geometric_arithmetic_sum_l1095_109550

open Real

noncomputable def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def arithmetic_mean (x y a b c : ℝ) : Prop :=
  2 * x = a + b ∧ 2 * y = b + c

theorem geometric_arithmetic_sum
  (a b c x y : ℝ)
  (habc : geometric_sequence a b c)
  (hxy : arithmetic_mean x y a b c)
  (hx_ne_zero : x ≠ 0)
  (hy_ne_zero : y ≠ 0) :
  (a / x) + (c / y) = 2 := 
by {
  sorry -- Proof omitted as per the prompt
}

end NUMINAMATH_GPT_geometric_arithmetic_sum_l1095_109550


namespace NUMINAMATH_GPT_max_tiles_l1095_109572

/--
Given a rectangular floor of size 180 cm by 120 cm
and rectangular tiles of size 25 cm by 16 cm, prove that the maximum number of tiles
that can be accommodated on the floor without overlapping, where the tiles' edges
are parallel and abutting the edges of the floor and with no tile overshooting the edges,
is 49 tiles.
-/
theorem max_tiles (floor_len floor_wid tile_len tile_wid : ℕ) (h1 : floor_len = 180)
  (h2 : floor_wid = 120) (h3 : tile_len = 25) (h4 : tile_wid = 16) :
  ∃ max_tiles : ℕ, max_tiles = 49 :=
by
  sorry

end NUMINAMATH_GPT_max_tiles_l1095_109572


namespace NUMINAMATH_GPT_distance_between_cities_l1095_109569

theorem distance_between_cities (d : ℝ)
  (meeting_point1 : d - 437 + 437 = d)
  (meeting_point2 : 3 * (d - 437) = 2 * d - 237) :
  d = 1074 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_cities_l1095_109569


namespace NUMINAMATH_GPT_find_y1_l1095_109533

theorem find_y1 
  (y1 y2 y3 : ℝ) 
  (h₀ : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h₁ : (1 - y1)^2 + 2 * (y1 - y2)^2 + 2 * (y2 - y3)^2 + y3^2 = 1 / 2) :
  y1 = (2 * Real.sqrt 2 - 1) / (2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_find_y1_l1095_109533


namespace NUMINAMATH_GPT_men_per_table_correct_l1095_109520

def tables := 6
def women_per_table := 3
def total_customers := 48
def total_women := women_per_table * tables
def total_men := total_customers - total_women
def men_per_table := total_men / tables

theorem men_per_table_correct : men_per_table = 5 := by
  sorry

end NUMINAMATH_GPT_men_per_table_correct_l1095_109520


namespace NUMINAMATH_GPT_algebraic_expression_value_l1095_109525

theorem algebraic_expression_value (a b : ℕ) (h : a - 3 * b = 0) :
  (a - (2 * a * b - b * b) / a) / ((a * a - b * b) / a) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1095_109525


namespace NUMINAMATH_GPT_no_perfect_squares_l1095_109578

theorem no_perfect_squares (x y z t : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : t > 0)
  (h5 : x * y - z * t = x + y) (h6 : x + y = z + t) : ¬(∃ a b : ℕ, a^2 = x * y ∧ b^2 = z * t) := 
by
  sorry

end NUMINAMATH_GPT_no_perfect_squares_l1095_109578


namespace NUMINAMATH_GPT_parabola_directrix_symmetry_l1095_109583

theorem parabola_directrix_symmetry:
  (∃ (d : ℝ), (∀ x : ℝ, x = d ↔ 
  (∃ y : ℝ, y^2 = (1 / 2) * x) ∧
  (∀ y : ℝ, x = (1 / 8)) → x = - (1 / 8))) :=
sorry

end NUMINAMATH_GPT_parabola_directrix_symmetry_l1095_109583


namespace NUMINAMATH_GPT_highest_degree_has_asymptote_l1095_109553

noncomputable def highest_degree_of_px (denom : ℕ → ℕ) (n : ℕ) : ℕ :=
  let deg := denom n
  deg

theorem highest_degree_has_asymptote (p : ℕ → ℕ) (denom : ℕ → ℕ) (n : ℕ)
  (h_denom : denom n = 6) :
  highest_degree_of_px denom n = 6 := by
  sorry

end NUMINAMATH_GPT_highest_degree_has_asymptote_l1095_109553


namespace NUMINAMATH_GPT_john_weekly_calories_l1095_109515

-- Define the calorie calculation for each meal type
def breakfast_calories : ℝ := 500
def morning_snack_calories : ℝ := 150
def lunch_calories : ℝ := breakfast_calories + 0.25 * breakfast_calories
def afternoon_snack_calories : ℝ := lunch_calories - 0.30 * lunch_calories
def dinner_calories : ℝ := 2 * lunch_calories

-- Total calories for Friday
def friday_calories : ℝ := breakfast_calories + morning_snack_calories + lunch_calories + afternoon_snack_calories + dinner_calories

-- Additional treats on Saturday and Sunday
def dessert_calories : ℝ := 350
def energy_drink_calories : ℝ := 220

-- Total calories for each day
def saturday_calories : ℝ := friday_calories + dessert_calories
def sunday_calories : ℝ := friday_calories + 2 * energy_drink_calories
def weekday_calories : ℝ := friday_calories

-- Proof statement
theorem john_weekly_calories : 
  friday_calories = 2962.5 ∧ 
  saturday_calories = 3312.5 ∧ 
  sunday_calories = 3402.5 ∧ 
  weekday_calories = 2962.5 :=
by 
  -- proof expressions would go here
  sorry

end NUMINAMATH_GPT_john_weekly_calories_l1095_109515


namespace NUMINAMATH_GPT_adam_total_cost_l1095_109564

theorem adam_total_cost 
    (sandwiches_count : ℕ)
    (sandwiches_price : ℝ)
    (chips_count : ℕ)
    (chips_price : ℝ)
    (water_count : ℕ)
    (water_price : ℝ)
    (sandwich_discount : sandwiches_count = 4 ∧ sandwiches_price = 4 ∧ sandwiches_count = 3 + 1)
    (tax_rate : ℝ)
    (initial_tax_rate : tax_rate = 0.10)
    (chips_cost : chips_count = 3 ∧ chips_price = 3.50)
    (water_cost : water_count = 2 ∧ water_price = 2) : 
  (3 * sandwiches_price + chips_count * chips_price + water_count * water_price) * (1 + tax_rate) = 29.15 := 
by
  sorry

end NUMINAMATH_GPT_adam_total_cost_l1095_109564


namespace NUMINAMATH_GPT_not_divides_two_pow_n_sub_one_l1095_109558

theorem not_divides_two_pow_n_sub_one (n : ℕ) (h1 : n > 1) : ¬ n ∣ (2^n - 1) :=
sorry

end NUMINAMATH_GPT_not_divides_two_pow_n_sub_one_l1095_109558


namespace NUMINAMATH_GPT_remainder_div_150_by_4_eq_2_l1095_109562

theorem remainder_div_150_by_4_eq_2 :
  (∃ k : ℕ, k > 0 ∧ 120 % k^2 = 24) → 150 % 4 = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_remainder_div_150_by_4_eq_2_l1095_109562


namespace NUMINAMATH_GPT_inequality_sqrt_ab_l1095_109573

theorem inequality_sqrt_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  2 / (1 / a + 1 / b) ≤ Real.sqrt (a * b) :=
sorry

end NUMINAMATH_GPT_inequality_sqrt_ab_l1095_109573


namespace NUMINAMATH_GPT_MikaWaterLeft_l1095_109530

def MikaWaterRemaining (startWater : ℚ) (usedWater : ℚ) : ℚ :=
  startWater - usedWater

theorem MikaWaterLeft :
  MikaWaterRemaining 3 (11 / 8) = 13 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_MikaWaterLeft_l1095_109530


namespace NUMINAMATH_GPT_problem_1_problem_2_l1095_109517

-- Problem 1: Prove that sqrt(6) * sqrt(1/3) - sqrt(16) * sqrt(18) = -11 * sqrt(2)
theorem problem_1 : Real.sqrt 6 * Real.sqrt (1 / 3) - Real.sqrt 16 * Real.sqrt 18 = -11 * Real.sqrt 2 := 
by
  sorry

-- Problem 2: Prove that (2 - sqrt(5)) * (2 + sqrt(5)) + (2 - sqrt(2))^2 = 5 - 4 * sqrt(2)
theorem problem_2 : (2 - Real.sqrt 5) * (2 + Real.sqrt 5) + (2 - Real.sqrt 2) ^ 2 = 5 - 4 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1095_109517


namespace NUMINAMATH_GPT_y1_gt_y2_l1095_109590

theorem y1_gt_y2 (k : ℝ) (y1 y2 : ℝ) 
  (h1 : y1 = (-1)^2 - 4*(-1) + k) 
  (h2 : y2 = 3^2 - 4*3 + k) : 
  y1 > y2 := 
by
  sorry

end NUMINAMATH_GPT_y1_gt_y2_l1095_109590


namespace NUMINAMATH_GPT_total_suitcases_l1095_109555

-- Definitions based on the conditions in a)
def siblings : ℕ := 4
def suitcases_per_sibling : ℕ := 2
def parents : ℕ := 2
def suitcases_per_parent : ℕ := 3
def suitcases_per_Lily : ℕ := 0

-- The statement to be proved
theorem total_suitcases : (siblings * suitcases_per_sibling) + (parents * suitcases_per_parent) + suitcases_per_Lily = 14 :=
by
  sorry

end NUMINAMATH_GPT_total_suitcases_l1095_109555


namespace NUMINAMATH_GPT_total_cost_calc_l1095_109554

variable (a b : ℝ)

def total_cost (a b : ℝ) := 2 * a + 3 * b

theorem total_cost_calc (a b : ℝ) : total_cost a b = 2 * a + 3 * b := by
  sorry

end NUMINAMATH_GPT_total_cost_calc_l1095_109554


namespace NUMINAMATH_GPT_alex_integer_list_count_l1095_109516

theorem alex_integer_list_count : 
  let n := 12 
  let least_multiple := 2^6 * 3^3
  let count := least_multiple / n
  count = 144 :=
by
  sorry

end NUMINAMATH_GPT_alex_integer_list_count_l1095_109516


namespace NUMINAMATH_GPT_james_passenger_count_l1095_109519

theorem james_passenger_count :
  ∀ (total_vehicles trucks buses taxis motorbikes cars trucks_population buses_population taxis_population motorbikes_population cars_population : ℕ),
  total_vehicles = 52 →
  trucks = 12 →
  buses = 2 →
  taxis = 2 * buses →
  motorbikes = total_vehicles - (trucks + buses + taxis + cars) →
  cars = 30 →
  trucks_population = 2 →
  buses_population = 15 →
  taxis_population = 2 →
  motorbikes_population = 1 →
  cars_population = 3 →
  (trucks * trucks_population + buses * buses_population + taxis * taxis_population +
   motorbikes * motorbikes_population + cars * cars_population) = 156 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_james_passenger_count_l1095_109519


namespace NUMINAMATH_GPT_percentage_profit_l1095_109504

theorem percentage_profit (cp sp : ℝ) (h1 : cp = 1200) (h2 : sp = 1680) : ((sp - cp) / cp) * 100 = 40 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_profit_l1095_109504


namespace NUMINAMATH_GPT_absolute_error_2175000_absolute_error_1730000_l1095_109537

noncomputable def absolute_error (a : ℕ) : ℕ :=
  if a = 2175000 then 1
  else if a = 1730000 then 10000
  else 0

theorem absolute_error_2175000 : absolute_error 2175000 = 1 :=
by sorry

theorem absolute_error_1730000 : absolute_error 1730000 = 10000 :=
by sorry

end NUMINAMATH_GPT_absolute_error_2175000_absolute_error_1730000_l1095_109537


namespace NUMINAMATH_GPT_tennis_tournament_matches_l1095_109582

noncomputable def total_matches (players: ℕ) : ℕ :=
  players - 1

theorem tennis_tournament_matches :
  total_matches 104 = 103 :=
by
  sorry

end NUMINAMATH_GPT_tennis_tournament_matches_l1095_109582


namespace NUMINAMATH_GPT_edric_days_per_week_l1095_109565

variable (monthly_salary : ℝ) (hours_per_day : ℝ) (hourly_rate : ℝ) (weeks_per_month : ℝ)
variable (days_per_week : ℝ)

-- Defining the conditions
def monthly_salary_condition : Prop := monthly_salary = 576
def hours_per_day_condition : Prop := hours_per_day = 8
def hourly_rate_condition : Prop := hourly_rate = 3
def weeks_per_month_condition : Prop := weeks_per_month = 4

-- Correct answer
def correct_answer : Prop := days_per_week = 6

-- Proof problem statement
theorem edric_days_per_week :
  monthly_salary_condition monthly_salary ∧
  hours_per_day_condition hours_per_day ∧
  hourly_rate_condition hourly_rate ∧
  weeks_per_month_condition weeks_per_month →
  correct_answer days_per_week :=
by
  sorry

end NUMINAMATH_GPT_edric_days_per_week_l1095_109565


namespace NUMINAMATH_GPT_capacity_of_other_bottle_l1095_109575

theorem capacity_of_other_bottle (x : ℝ) :
  (16 / 3) * (x / 8) + (16 / 3) = 8 → x = 4 := by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_capacity_of_other_bottle_l1095_109575


namespace NUMINAMATH_GPT_range_of_x_l1095_109523

theorem range_of_x
  (x : ℝ)
  (h1 : ∀ m, -1 ≤ m ∧ m ≤ 4 → m * (x^2 - 1) - 1 - 8 * x < 0) :
  0 < x ∧ x < 5 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_x_l1095_109523


namespace NUMINAMATH_GPT_jessica_final_balance_l1095_109546

variable (B : ℝ) (withdrawal : ℝ) (deposit : ℝ)

-- Conditions
def condition1 : Prop := withdrawal = (2 / 5) * B
def condition2 : Prop := deposit = (1 / 5) * (B - withdrawal)

-- Proof goal statement
theorem jessica_final_balance (h1 : condition1 B withdrawal)
                             (h2 : condition2 B withdrawal deposit) :
    (B - withdrawal + deposit) = 360 :=
by
  sorry

end NUMINAMATH_GPT_jessica_final_balance_l1095_109546


namespace NUMINAMATH_GPT_base_number_unique_l1095_109556

theorem base_number_unique (y : ℕ) : (3 : ℝ) ^ 16 = (9 : ℝ) ^ y → y = 8 → (9 : ℝ) = 3 ^ (16 / y) :=
by
  sorry

end NUMINAMATH_GPT_base_number_unique_l1095_109556


namespace NUMINAMATH_GPT_susan_can_drive_with_50_l1095_109500

theorem susan_can_drive_with_50 (car_efficiency : ℕ) (gas_price : ℕ) (money_available : ℕ) 
  (h1 : car_efficiency = 40) (h2 : gas_price = 5) (h3 : money_available = 50) : 
  car_efficiency * (money_available / gas_price) = 400 :=
by
  sorry

end NUMINAMATH_GPT_susan_can_drive_with_50_l1095_109500


namespace NUMINAMATH_GPT_total_cups_l1095_109551

theorem total_cups (b f s : ℕ) (ratio_bt_f_s : b / s = 1 / 5) (ratio_fl_b_s : f / s = 8 / 5) (sugar_cups : s = 10) :
  b + f + s = 28 :=
sorry

end NUMINAMATH_GPT_total_cups_l1095_109551


namespace NUMINAMATH_GPT_isabella_hair_ratio_l1095_109505

-- Conditions in the problem
variable (hair_before : ℕ) (hair_after : ℕ)
variable (hb : hair_before = 18)
variable (ha : hair_after = 36)

-- Definitions based on conditions
def hair_ratio (after : ℕ) (before : ℕ) : ℚ := (after : ℚ) / (before : ℚ)

theorem isabella_hair_ratio : 
  hair_ratio hair_after hair_before = 2 :=
by
  -- plug in the known values
  rw [hb, ha]
  -- show the equation
  norm_num
  sorry

end NUMINAMATH_GPT_isabella_hair_ratio_l1095_109505


namespace NUMINAMATH_GPT_time_for_A_and_D_together_l1095_109580

theorem time_for_A_and_D_together (A_rate D_rate combined_rate : ℝ)
  (hA : A_rate = 1 / 10) (hD : D_rate = 1 / 10) 
  (h_combined : combined_rate = A_rate + D_rate) :
  1 / combined_rate = 5 :=
by
  sorry

end NUMINAMATH_GPT_time_for_A_and_D_together_l1095_109580


namespace NUMINAMATH_GPT_integers_between_sqrt7_and_sqrt77_l1095_109514

theorem integers_between_sqrt7_and_sqrt77 : 
  2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3 ∧ 8 < Real.sqrt 77 ∧ Real.sqrt 77 < 9 →
  ∃ (n : ℕ), n = 6 ∧ ∀ (k : ℕ), (3 ≤ k ∧ k ≤ 8) ↔ (2 < Real.sqrt 7 ∧ Real.sqrt 77 < 9) :=
by sorry

end NUMINAMATH_GPT_integers_between_sqrt7_and_sqrt77_l1095_109514
