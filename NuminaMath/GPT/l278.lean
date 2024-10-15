import Mathlib

namespace NUMINAMATH_GPT_candy_problem_l278_27892

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end NUMINAMATH_GPT_candy_problem_l278_27892


namespace NUMINAMATH_GPT_evaluate_expression_l278_27894

theorem evaluate_expression (x y : ℕ) (h1 : x = 4) (h2 : y = 3) :
  5 * x + 2 * y * 3 = 38 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l278_27894


namespace NUMINAMATH_GPT_s_6_of_30_eq_146_over_175_l278_27804

def s (θ : ℚ) : ℚ := 1 / (2 - θ)

theorem s_6_of_30_eq_146_over_175 : s (s (s (s (s (s 30))))) = 146 / 175 := sorry

end NUMINAMATH_GPT_s_6_of_30_eq_146_over_175_l278_27804


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l278_27806

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x > 1 ∧ y > 1) → (x + y > 2) ∧ ¬((x + y > 2) → (x > 1 ∧ y > 1)) := 
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l278_27806


namespace NUMINAMATH_GPT_rajan_income_l278_27833

theorem rajan_income (x y : ℝ) 
  (h₁ : 7 * x - 6 * y = 1000) 
  (h₂ : 6 * x - 5 * y = 1000) : 
  7 * x = 7000 :=
by 
  sorry

end NUMINAMATH_GPT_rajan_income_l278_27833


namespace NUMINAMATH_GPT_find_a_l278_27838

theorem find_a (a : ℝ) (hne : a ≠ 1) (eq_sets : ∀ x : ℝ, (a-1) * x < a + 5 ↔ 2 * x < 4) : a = 7 :=
sorry

end NUMINAMATH_GPT_find_a_l278_27838


namespace NUMINAMATH_GPT_exists_common_ratio_of_geometric_progression_l278_27800

theorem exists_common_ratio_of_geometric_progression (a r : ℝ) (h_pos : 0 < r) 
(h_eq: a = a * r + a * r^2 + a * r^3) : ∃ r : ℝ, r^3 + r^2 + r - 1 = 0 :=
by sorry

end NUMINAMATH_GPT_exists_common_ratio_of_geometric_progression_l278_27800


namespace NUMINAMATH_GPT_initial_team_sizes_l278_27882

/-- 
On the first day of the sports competition, 1/6 of the boys' team and 1/7 of the girls' team 
did not meet the qualifying standards and were eliminated. During the rest of the competition, 
the same number of athletes from both teams were eliminated for not meeting the standards. 
By the end of the competition, a total of 48 boys and 50 girls did not meet the qualifying standards. 
Moreover, the number of girls who met the qualifying standards was twice the number of boys who did.
We are to prove the initial number of boys and girls in their respective teams.
-/

theorem initial_team_sizes (initial_boys initial_girls : ℕ) :
  (∃ (x : ℕ), 
    initial_boys = x + 48 ∧ 
    initial_girls = 2 * x + 50 ∧ 
    48 - (1 / 6 : ℚ) * (x + 48 : ℚ) = 50 - (1 / 7 : ℚ) * (2 * x + 50 : ℚ) ∧
    initial_girls - 2 * initial_boys = 98 - 2 * 72
  ) ↔ 
  initial_boys = 72 ∧ initial_girls = 98 := 
sorry

end NUMINAMATH_GPT_initial_team_sizes_l278_27882


namespace NUMINAMATH_GPT_total_area_of_removed_triangles_l278_27810

theorem total_area_of_removed_triangles : 
  ∀ (side_length_of_square : ℝ) (hypotenuse_length_of_triangle : ℝ),
  side_length_of_square = 20 →
  hypotenuse_length_of_triangle = 10 →
  4 * (1/2 * (hypotenuse_length_of_triangle^2 / 2)) = 100 :=
by
  intros side_length_of_square hypotenuse_length_of_triangle h_side_length h_hypotenuse_length
  -- Proof would go here, but we add "sorry" to complete the statement
  sorry

end NUMINAMATH_GPT_total_area_of_removed_triangles_l278_27810


namespace NUMINAMATH_GPT_simplify_expression_l278_27888

theorem simplify_expression (m : ℤ) : 
  ((7 * m + 3) - 3 * m * 2) * 4 + (5 - 2 / 4) * (8 * m - 12) = 40 * m - 42 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l278_27888


namespace NUMINAMATH_GPT_fraction_equality_l278_27898

variables (x y : ℝ)

theorem fraction_equality (h : y / 2 = (2 * y - x) / 3) : y / x = 2 :=
sorry

end NUMINAMATH_GPT_fraction_equality_l278_27898


namespace NUMINAMATH_GPT_planar_graph_edge_vertex_inequality_l278_27840

def planar_graph (G : Type _) : Prop := -- Placeholder for planar graph property
  sorry

variables {V E : ℕ}

theorem planar_graph_edge_vertex_inequality (G : Type _) (h : planar_graph G) :
  E ≤ 3 * V - 6 :=
sorry

end NUMINAMATH_GPT_planar_graph_edge_vertex_inequality_l278_27840


namespace NUMINAMATH_GPT_solve_quadratic_eq_l278_27865

theorem solve_quadratic_eq : ∀ x : ℝ, (12 - 3 * x)^2 = x^2 ↔ x = 6 ∨ x = 3 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l278_27865


namespace NUMINAMATH_GPT_masha_wins_l278_27872

def num_matches : Nat := 111

-- Define a function for Masha's optimal play strategy
-- In this problem, we'll denote both players' move range and the condition for winning.
theorem masha_wins (n : Nat := num_matches) (conditions : n > 0 ∧ n % 11 = 0 ∧ (∀ k : Nat, 1 ≤ k ∧ k ≤ 10 → ∃ new_n : Nat, n = k + new_n)) : True :=
  sorry

end NUMINAMATH_GPT_masha_wins_l278_27872


namespace NUMINAMATH_GPT_joe_new_average_l278_27852

def joe_tests_average (a b c d : ℝ) : Prop :=
  ((a + b + c + d) / 4 = 35) ∧ (min a (min b (min c d)) = 20)

theorem joe_new_average (a b c d : ℝ) (h : joe_tests_average a b c d) :
  ((a + b + c + d - min a (min b (min c d))) / 3 = 40) :=
sorry

end NUMINAMATH_GPT_joe_new_average_l278_27852


namespace NUMINAMATH_GPT_living_room_floor_area_l278_27887

-- Define the problem conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area : ℝ := carpet_length * carpet_width    -- Area of the carpet

def percentage_covered_by_carpet : ℝ := 0.75

-- Theorem to prove: the area of the living room floor is 48 square feet
theorem living_room_floor_area (carpet_area : ℝ) (percentage_covered_by_carpet : ℝ) : 
  (A_floor : ℝ) = carpet_area / percentage_covered_by_carpet :=
by
  let carpet_area := 36
  let percentage_covered_by_carpet := 0.75
  let A_floor := 48
  sorry

end NUMINAMATH_GPT_living_room_floor_area_l278_27887


namespace NUMINAMATH_GPT_Stan_pays_magician_l278_27853

theorem Stan_pays_magician :
  let hours_per_day := 3
  let days_per_week := 7
  let weeks := 2
  let hourly_rate := 60
  let total_hours := hours_per_day * days_per_week * weeks
  let total_payment := hourly_rate * total_hours
  total_payment = 2520 := 
by 
  sorry

end NUMINAMATH_GPT_Stan_pays_magician_l278_27853


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l278_27868

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 3) (h2 : (x - y) ^ 2 = 9) : 
  x ^ 2 + y ^ 2 = 15 := sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l278_27868


namespace NUMINAMATH_GPT_prob_all_four_even_dice_l278_27862

noncomputable def probability_even (n : ℕ) : ℚ := (3 / 6)^n

theorem prob_all_four_even_dice : probability_even 4 = 1 / 16 := 
by
  sorry

end NUMINAMATH_GPT_prob_all_four_even_dice_l278_27862


namespace NUMINAMATH_GPT_max_value_of_expression_l278_27819

open Real

theorem max_value_of_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) : 
  2 * x * y + y * z + 2 * z * x ≤ 4 / 7 := 
sorry

end NUMINAMATH_GPT_max_value_of_expression_l278_27819


namespace NUMINAMATH_GPT_income_growth_l278_27861

theorem income_growth (x : ℝ) : 12000 * (1 + x)^2 = 14520 :=
sorry

end NUMINAMATH_GPT_income_growth_l278_27861


namespace NUMINAMATH_GPT_calculate_land_tax_l278_27817

def plot_size : ℕ := 15
def cadastral_value_per_sotka : ℕ := 100000
def tax_rate : ℝ := 0.003

theorem calculate_land_tax :
  plot_size * cadastral_value_per_sotka * tax_rate = 4500 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_land_tax_l278_27817


namespace NUMINAMATH_GPT_slope_of_line_l278_27832

theorem slope_of_line (x₁ y₁ x₂ y₂ : ℝ) (h₁ : 2/x₁ + 3/y₁ = 0) (h₂ : 2/x₂ + 3/y₂ = 0) (h_diff : x₁ ≠ x₂) : 
  (y₂ - y₁) / (x₂ - x₁) = -3/2 :=
sorry

end NUMINAMATH_GPT_slope_of_line_l278_27832


namespace NUMINAMATH_GPT_problem_l278_27864

def isRightTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

def CannotFormRightTriangle (lst : List ℝ) : Prop :=
  ¬isRightTriangle lst.head! lst.tail.head! lst.tail.tail.head!

theorem problem :
  (¬isRightTriangle 3 4 5 ∧ ¬isRightTriangle 5 12 13 ∧ ¬isRightTriangle 2 3 (Real.sqrt 13)) ∧ CannotFormRightTriangle [4, 6, 8] :=
by
  sorry

end NUMINAMATH_GPT_problem_l278_27864


namespace NUMINAMATH_GPT_max_group_size_l278_27873

theorem max_group_size 
  (students_class1 : ℕ) (students_class2 : ℕ) 
  (leftover_class1 : ℕ) (leftover_class2 : ℕ) 
  (h_class1 : students_class1 = 69) 
  (h_class2 : students_class2 = 86) 
  (h_leftover1 : leftover_class1 = 5) 
  (h_leftover2 : leftover_class2 = 6) : 
  Nat.gcd (students_class1 - leftover_class1) (students_class2 - leftover_class2) = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_group_size_l278_27873


namespace NUMINAMATH_GPT_math_problem_l278_27883

noncomputable def log_base (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem math_problem (a b c : ℝ) (h1 : ∃ k : ℤ, log_base c b = k)
  (h2 : log_base a (1 / b) > log_base a (Real.sqrt b) ∧ log_base a (Real.sqrt b) > log_base b (a^2)) :
  (∃ n : ℕ, n = 1 ∧ 
    ((1 / b > Real.sqrt b ∧ Real.sqrt b > a^2) ∨ 
    (Real.log b + log_base a a = 0) ∨ 
    (0 < a ∧ a < b ∧ b < 1) ∨ 
    (a * b = 1))) :=
by sorry

end NUMINAMATH_GPT_math_problem_l278_27883


namespace NUMINAMATH_GPT_total_number_of_workers_l278_27886

variables (W N : ℕ)
variables (average_salary_workers average_salary_techs average_salary_non_techs : ℤ)
variables (num_techs total_salary total_salary_techs total_salary_non_techs : ℤ)

theorem total_number_of_workers (h1 : average_salary_workers = 8000)
                               (h2 : average_salary_techs = 14000)
                               (h3 : num_techs = 7)
                               (h4 : average_salary_non_techs = 6000)
                               (h5 : total_salary = W * 8000)
                               (h6 : total_salary_techs = 7 * 14000)
                               (h7 : total_salary_non_techs = N * 6000)
                               (h8 : total_salary = total_salary_techs + total_salary_non_techs)
                               (h9 : W = 7 + N) : 
                               W = 28 :=
sorry

end NUMINAMATH_GPT_total_number_of_workers_l278_27886


namespace NUMINAMATH_GPT_shop_owner_pricing_l278_27897

theorem shop_owner_pricing (L C M S : ℝ)
  (h1 : C = 0.75 * L)
  (h2 : S = 1.3 * C)
  (h3 : S = 0.75 * M) : 
  M = 1.3 * L := 
sorry

end NUMINAMATH_GPT_shop_owner_pricing_l278_27897


namespace NUMINAMATH_GPT_glass_pieces_same_color_l278_27839

theorem glass_pieces_same_color (r y b : ℕ) (h : r + y + b = 2002) :
  (∃ k : ℕ, ∀ n, n ≥ k → (r + y + b) = n ∧ (r = 0 ∨ y = 0 ∨ b = 0)) ∧
  (∀ (r1 y1 b1 r2 y2 b2 : ℕ),
    r1 + y1 + b1 = 2002 →
    r2 + y2 + b2 = 2002 →
    (∃ k : ℕ, ∀ n, n ≥ k → (r1 = 0 ∨ y1 = 0 ∨ b1 = 0)) →
    (∃ l : ℕ, ∀ m, m ≥ l → (r2 = 0 ∨ y2 = 0 ∨ b2 = 0)) →
    r1 = r2 ∧ y1 = y2 ∧ b1 = b2):=
by
  sorry

end NUMINAMATH_GPT_glass_pieces_same_color_l278_27839


namespace NUMINAMATH_GPT_rate_grapes_l278_27896

/-- Given that Bruce purchased 8 kg of grapes at a rate G per kg, 8 kg of mangoes at the rate of 55 per kg, 
and paid a total of 1000 to the shopkeeper, prove that the rate per kg for the grapes (G) is 70. -/
theorem rate_grapes (G : ℝ) (h1 : 8 * G + 8 * 55 = 1000) : G = 70 :=
by 
  sorry

end NUMINAMATH_GPT_rate_grapes_l278_27896


namespace NUMINAMATH_GPT_canister_ratio_l278_27809

variable (C D : ℝ) -- Define capacities of canister C and canister D
variable (hC_half : 1/2 * C) -- Canister C is 1/2 full of water
variable (hD_third : 1/3 * D) -- Canister D is 1/3 full of water
variable (hD_after : 1/12 * D) -- Canister D contains 1/12 after pouring

theorem canister_ratio (h : 1/2 * C = 1/4 * D) : D / C = 2 :=
by
  sorry

end NUMINAMATH_GPT_canister_ratio_l278_27809


namespace NUMINAMATH_GPT_marie_packs_construction_paper_l278_27801

theorem marie_packs_construction_paper (marie_glue_sticks : ℕ) (allison_glue_sticks : ℕ) (total_allison_items : ℕ)
    (glue_sticks_difference : allison_glue_sticks = marie_glue_sticks + 8)
    (marie_glue_sticks_count : marie_glue_sticks = 15)
    (total_items_allison : total_allison_items = 28)
    (marie_construction_paper_multiplier : ℕ)
    (construction_paper_ratio : marie_construction_paper_multiplier = 6) : 
    ∃ (marie_construction_paper_packs : ℕ), marie_construction_paper_packs = 30 := 
by
  sorry

end NUMINAMATH_GPT_marie_packs_construction_paper_l278_27801


namespace NUMINAMATH_GPT_pool_capacity_l278_27881

variable (C : ℕ)

-- Conditions
def rate_first_valve := C / 120
def rate_second_valve := C / 120 + 50
def combined_rate := C / 48

-- Proof statement
theorem pool_capacity (C_pos : 0 < C) (h1 : rate_first_valve C + rate_second_valve C = combined_rate C) : C = 12000 := by
  sorry

end NUMINAMATH_GPT_pool_capacity_l278_27881


namespace NUMINAMATH_GPT_probability_of_type_I_error_l278_27870

theorem probability_of_type_I_error 
  (K_squared : ℝ)
  (alpha : ℝ)
  (critical_val : ℝ)
  (h1 : K_squared = 4.05)
  (h2 : alpha = 0.05)
  (h3 : critical_val = 3.841)
  (h4 : 4.05 > 3.841) :
  alpha = 0.05 := 
sorry

end NUMINAMATH_GPT_probability_of_type_I_error_l278_27870


namespace NUMINAMATH_GPT_smallest_n_value_l278_27836

-- Define the conditions as given in the problem
def num_birthdays := 365

-- Formulating the main statement
theorem smallest_n_value : ∃ (n : ℕ), (∀ (group_size : ℕ), group_size = 2 * n - 10 → group_size ≥ 3286) ∧ n = 1648 :=
by
  use 1648
  sorry

end NUMINAMATH_GPT_smallest_n_value_l278_27836


namespace NUMINAMATH_GPT_cone_height_l278_27826

theorem cone_height (r h : ℝ) (π : ℝ) (Hπ : Real.pi = π) (slant_height : ℝ) (lateral_area : ℝ) (base_area : ℝ) 
  (H1 : slant_height = 2) 
  (H2 : lateral_area = 2 * π * r) 
  (H3 : base_area = π * r^2) 
  (H4 : lateral_area = 4 * base_area) 
  (H5 : r^2 + h^2 = slant_height^2) 
  : h = π / 2 := by 
sorry

end NUMINAMATH_GPT_cone_height_l278_27826


namespace NUMINAMATH_GPT_determine_counterfeit_coin_l278_27816

-- Definitions and conditions
def coin_weight (coin : ℕ) : ℕ :=
  match coin with
  | 1 => 1 -- 1-kopek coin weighs 1 gram
  | 2 => 2 -- 2-kopeks coin weighs 2 grams
  | 3 => 3 -- 3-kopeks coin weighs 3 grams
  | 5 => 5 -- 5-kopeks coin weighs 5 grams
  | _ => 0 -- Invalid coin denomination, should not happen

def is_counterfeit (coin : ℕ) (actual_weight : ℕ) : Prop :=
  coin_weight coin ≠ actual_weight

-- Statement of the problem to be proved
theorem determine_counterfeit_coin (coins : List (ℕ × ℕ)) :
   (∀ (coin: ℕ) (weight: ℕ) (h : (coin, weight) ∈ coins),
      coin_weight coin = weight ∨ is_counterfeit coin weight) →
   (∃ (counterfeit_coin: ℕ) (weight: ℕ),
      (counterfeit_coin, weight) ∈ coins ∧ is_counterfeit counterfeit_coin weight) :=
sorry

end NUMINAMATH_GPT_determine_counterfeit_coin_l278_27816


namespace NUMINAMATH_GPT_possible_values_of_reciprocal_sum_l278_27874

theorem possible_values_of_reciprocal_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  ∃ y, y = (1/a + 1/b) ∧ (2 ≤ y ∧ ∀ t, t < y ↔ ¬t < 2) :=
by sorry

end NUMINAMATH_GPT_possible_values_of_reciprocal_sum_l278_27874


namespace NUMINAMATH_GPT_primes_divisibility_l278_27813

theorem primes_divisibility
  (p1 p2 p3 p4 q1 q2 q3 q4 : ℕ)
  (hp1_lt_p2 : p1 < p2) (hp2_lt_p3 : p2 < p3) (hp3_lt_p4 : p3 < p4)
  (hq1_lt_q2 : q1 < q2) (hq2_lt_q3 : q2 < q3) (hq3_lt_q4 : q3 < q4)
  (hp4_minus_p1 : p4 - p1 = 8) (hq4_minus_q1 : q4 - q1 = 8)
  (hp1_gt_5 : 5 < p1) (hq1_gt_5 : 5 < q1) :
  30 ∣ (p1 - q1) :=
sorry

end NUMINAMATH_GPT_primes_divisibility_l278_27813


namespace NUMINAMATH_GPT_right_rect_prism_volume_l278_27846

theorem right_rect_prism_volume (a b c : ℝ) 
  (h1 : a * b = 56) 
  (h2 : b * c = 63) 
  (h3 : a * c = 36) : 
  a * b * c = 504 := by
  sorry

end NUMINAMATH_GPT_right_rect_prism_volume_l278_27846


namespace NUMINAMATH_GPT_max_ab_l278_27823

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 40) : 
  ab ≤ 400 :=
sorry

end NUMINAMATH_GPT_max_ab_l278_27823


namespace NUMINAMATH_GPT_is_isosceles_right_triangle_l278_27837

theorem is_isosceles_right_triangle 
  {a b c : ℝ}
  (h : |c^2 - a^2 - b^2| + (a - b)^2 = 0) : 
  a = b ∧ c^2 = a^2 + b^2 :=
sorry

end NUMINAMATH_GPT_is_isosceles_right_triangle_l278_27837


namespace NUMINAMATH_GPT_largest_prime_divisor_of_36_squared_plus_49_squared_l278_27830

theorem largest_prime_divisor_of_36_squared_plus_49_squared :
  Nat.gcd (36^2 + 49^2) 3697 = 3697 :=
by
  -- Since 3697 is prime, and the calculation shows 36^2 + 49^2 is 3697
  sorry

end NUMINAMATH_GPT_largest_prime_divisor_of_36_squared_plus_49_squared_l278_27830


namespace NUMINAMATH_GPT_tangent_line_equation_l278_27857

theorem tangent_line_equation 
    (h_perpendicular : ∃ m1 m2 : ℝ, m1 * m2 = -1 ∧ (∀ y, x + m1 * y = 4) ∧ (x + 4 * y = 4)) 
    (h_tangent : ∀ x : ℝ, y = 2 * x ^ 2 ∧ (∀ y', y' = 4 * x)) :
    ∃ a b c : ℝ, (4 * a - b - c = 0) ∧ (∀ (t : ℝ), a * t + b * (2 * t ^ 2) = 1) :=
sorry

end NUMINAMATH_GPT_tangent_line_equation_l278_27857


namespace NUMINAMATH_GPT_person_left_time_l278_27802

theorem person_left_time :
  ∃ (x y : ℚ), 
    0 ≤ x ∧ x < 1 ∧ 
    0 ≤ y ∧ y < 1 ∧ 
    (120 + 30 * x = 360 * y) ∧
    (360 * x = 150 + 30 * y) ∧
    (4 + x = 4 + 64 / 143) := 
by
  sorry

end NUMINAMATH_GPT_person_left_time_l278_27802


namespace NUMINAMATH_GPT_cube_volume_l278_27875

theorem cube_volume (A V : ℝ) (h : A = 16) : V = 64 :=
by
  -- Here, we would provide the proof, but for now, we end with sorry
  sorry

end NUMINAMATH_GPT_cube_volume_l278_27875


namespace NUMINAMATH_GPT_base_b_representation_1987_l278_27827

theorem base_b_representation_1987 (x y z b : ℕ) (h1 : x + y + z = 25) (h2 : x ≥ 1)
  (h3 : 1987 = x * b^2 + y * b + z) (h4 : 12 < b) (h5 : b < 45) :
  x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 :=
sorry

end NUMINAMATH_GPT_base_b_representation_1987_l278_27827


namespace NUMINAMATH_GPT_trapezoid_area_l278_27814

theorem trapezoid_area (outer_triangle_area inner_triangle_area : ℝ) (congruent_trapezoids : ℕ) 
  (h1 : outer_triangle_area = 36) (h2 : inner_triangle_area = 4) (h3 : congruent_trapezoids = 3) :
  (outer_triangle_area - inner_triangle_area) / congruent_trapezoids = 32 / 3 :=
by sorry

end NUMINAMATH_GPT_trapezoid_area_l278_27814


namespace NUMINAMATH_GPT_evaluate_fraction_l278_27863

theorem evaluate_fraction : (1 / (2 + (1 / (3 + (1 / 4))))) = 13 / 30 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l278_27863


namespace NUMINAMATH_GPT_least_x_value_l278_27842

theorem least_x_value : ∀ x : ℝ, (4 * x^2 + 7 * x + 3 = 5) → x = -2 ∨ x >= -2 := by 
    intro x
    intro h
    sorry

end NUMINAMATH_GPT_least_x_value_l278_27842


namespace NUMINAMATH_GPT_find_five_digit_number_l278_27811

theorem find_five_digit_number : 
  ∃ (A B C D E : ℕ), 
    (0 < A ∧ A ≤ 9) ∧ 
    (0 < B ∧ B ≤ 9) ∧ 
    (0 < C ∧ C ≤ 9) ∧ 
    (0 < D ∧ D ≤ 9) ∧ 
    (0 < E ∧ E ≤ 9) ∧ 
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E) ∧ 
    (B ≠ C ∧ B ≠ D ∧ B ≠ E) ∧ 
    (C ≠ D ∧ C ≠ E) ∧ 
    (D ≠ E) ∧ 
    (2016 = (10 * D + E) * A * B) ∧ 
    (¬ (10 * D + E) % 3 = 0) ∧ 
    (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E = 85132) :=
sorry

end NUMINAMATH_GPT_find_five_digit_number_l278_27811


namespace NUMINAMATH_GPT_area_of_region_ABCDEFGHIJ_l278_27869

/-- 
  Given:
  1. Region ABCDEFGHIJ consists of 13 equal squares.
  2. Region ABCDEFGHIJ is inscribed in rectangle PQRS.
  3. Point A is on line PQ, B is on line QR, E is on line RS, and H is on line SP.
  4. PQ has length 28 and QR has length 26.

  Prove that the area of region ABCDEFGHIJ is 338 square units.
-/
theorem area_of_region_ABCDEFGHIJ 
  (squares : ℕ)             -- Number of squares in region ABCDEFGHIJ
  (len_PQ len_QR : ℕ)       -- Lengths of sides PQ and QR
  (area : ℕ)                 -- Area of region ABCDEFGHIJ
  (h1 : squares = 13)
  (h2 : len_PQ = 28)
  (h3 : len_QR = 26)
  : area = 338 :=
sorry

end NUMINAMATH_GPT_area_of_region_ABCDEFGHIJ_l278_27869


namespace NUMINAMATH_GPT_conic_section_hyperbola_l278_27858

theorem conic_section_hyperbola (x y : ℝ) :
  (x - 3) ^ 2 = 9 * (y + 2) ^ 2 - 81 → conic_section := by
  sorry

end NUMINAMATH_GPT_conic_section_hyperbola_l278_27858


namespace NUMINAMATH_GPT_nancy_pictures_left_l278_27822

-- Given conditions stated in the problem
def picturesZoo : Nat := 49
def picturesMuseum : Nat := 8
def picturesDeleted : Nat := 38

-- The statement of the problem, proving Nancy still has 19 pictures after deletions
theorem nancy_pictures_left : (picturesZoo + picturesMuseum) - picturesDeleted = 19 := by
  sorry

end NUMINAMATH_GPT_nancy_pictures_left_l278_27822


namespace NUMINAMATH_GPT_circle_equation_focus_parabola_origin_l278_27856

noncomputable def parabola_focus (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 4 * p * x

def passes_through_origin (x y : ℝ) : Prop :=
  (0 - x)^2 + (0 - y)^2 = x^2 + y^2

theorem circle_equation_focus_parabola_origin :
  (∃ x y : ℝ, parabola_focus 1 x y ∧ passes_through_origin x y)
    → ∃ k : ℝ, (x^2 - 2 * x + y^2 = k) :=
sorry

end NUMINAMATH_GPT_circle_equation_focus_parabola_origin_l278_27856


namespace NUMINAMATH_GPT_problem_l278_27829

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x ^ 2

noncomputable def f (gx : ℝ) (x : ℝ) : ℝ := (2 - 3 * x ^ 2) / x ^ 2

theorem problem (x : ℝ) (hx : x ≠ 0) : f (g x) x = 3 / 2 :=
  sorry

end NUMINAMATH_GPT_problem_l278_27829


namespace NUMINAMATH_GPT_ledi_age_10_in_years_l278_27889

-- Definitions of ages of Duoduo and Ledi
def duoduo_current_age : ℝ := 10
def years_ago : ℝ := 12.3
def sum_ages_years_ago : ℝ := 12

-- Function to calculate Ledi's current age
def ledi_current_age :=
  (sum_ages_years_ago + years_ago + years_ago) + (duoduo_current_age - years_ago)

-- Function to calculate years from now for Ledi to be 10 years old
def years_until_ledi_age_10 (ledi_age_now : ℝ) : ℝ :=
  10 - ledi_age_now

-- Main statement we need to prove
theorem ledi_age_10_in_years : years_until_ledi_age_10 ledi_current_age = 6.3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ledi_age_10_in_years_l278_27889


namespace NUMINAMATH_GPT_transistors_in_2005_l278_27803

theorem transistors_in_2005
  (initial_count : ℕ)
  (doubles_every : ℕ)
  (triples_every : ℕ)
  (years : ℕ) :
  initial_count = 500000 ∧ doubles_every = 2 ∧ triples_every = 6 ∧ years = 15 →
  (initial_count * 2^(years/doubles_every) + initial_count * 3^(years/triples_every)) = 68500000 :=
by
  sorry

end NUMINAMATH_GPT_transistors_in_2005_l278_27803


namespace NUMINAMATH_GPT_people_who_speak_French_l278_27878

theorem people_who_speak_French (T L N B : ℕ) (hT : T = 25) (hL : L = 13) (hN : N = 6) (hB : B = 9) : 
  ∃ F : ℕ, F = 15 := 
by 
  sorry

end NUMINAMATH_GPT_people_who_speak_French_l278_27878


namespace NUMINAMATH_GPT_solve_for_m_l278_27851

theorem solve_for_m (x y m : ℤ) (h1 : x - 2 * y = -3) (h2 : 2 * x + 3 * y = m - 1) (h3 : x = -y) : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_m_l278_27851


namespace NUMINAMATH_GPT_second_candidate_votes_l278_27818

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℝ) (first_candidate_votes: ℕ)
    (h1 : total_votes = 2400)
    (h2 : first_candidate_percentage = 0.80)
    (h3 : first_candidate_votes = total_votes * first_candidate_percentage) :
    total_votes - first_candidate_votes = 480 := by
    sorry

end NUMINAMATH_GPT_second_candidate_votes_l278_27818


namespace NUMINAMATH_GPT_average_runs_per_game_l278_27850

-- Define the number of games
def games : ℕ := 6

-- Define the list of runs scored in each game
def runs : List ℕ := [1, 4, 4, 5, 5, 5]

-- The sum of the runs
def total_runs : ℕ := List.sum runs

-- The average runs per game
def avg_runs : ℚ := total_runs / games

-- The theorem to prove
theorem average_runs_per_game : avg_runs = 4 := by sorry

end NUMINAMATH_GPT_average_runs_per_game_l278_27850


namespace NUMINAMATH_GPT_digit_B_divisible_by_9_l278_27859

-- Defining the condition for B making 762B divisible by 9
theorem digit_B_divisible_by_9 (B : ℕ) : (15 + B) % 9 = 0 ↔ B = 3 := 
by
  sorry

end NUMINAMATH_GPT_digit_B_divisible_by_9_l278_27859


namespace NUMINAMATH_GPT_similar_area_ratios_l278_27879

theorem similar_area_ratios (a₁ a₂ s₁ s₂ : ℝ) (h₁ : a₁ = s₁^2) (h₂ : a₂ = s₂^2) (h₃ : a₁ / a₂ = 1 / 9) (h₄ : s₁ = 4) : s₂ = 12 :=
by
  sorry

end NUMINAMATH_GPT_similar_area_ratios_l278_27879


namespace NUMINAMATH_GPT_change_received_l278_27855

variable (a : ℝ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a :=
by
  sorry

end NUMINAMATH_GPT_change_received_l278_27855


namespace NUMINAMATH_GPT_triangle_probability_l278_27847

open Classical

theorem triangle_probability :
  let a := 5
  let b := 6
  let lengths := [1, 2, 6, 11]
  let valid_third_side x := 1 < x ∧ x < 11
  let valid_lengths := lengths.filter valid_third_side
  let probability := valid_lengths.length / lengths.length
  probability = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_probability_l278_27847


namespace NUMINAMATH_GPT_mary_initial_pokemon_cards_l278_27849

theorem mary_initial_pokemon_cards (x : ℕ) (torn_cards : ℕ) (new_cards : ℕ) (current_cards : ℕ) 
  (h1 : torn_cards = 6) 
  (h2 : new_cards = 23) 
  (h3 : current_cards = 56) 
  (h4 : current_cards = x - torn_cards + new_cards) : 
  x = 39 := 
by
  sorry

end NUMINAMATH_GPT_mary_initial_pokemon_cards_l278_27849


namespace NUMINAMATH_GPT_heather_final_blocks_l278_27860

def heather_initial_blocks : ℝ := 86.0
def jose_shared_blocks : ℝ := 41.0

theorem heather_final_blocks : heather_initial_blocks + jose_shared_blocks = 127.0 :=
by
  sorry

end NUMINAMATH_GPT_heather_final_blocks_l278_27860


namespace NUMINAMATH_GPT_zeros_not_adjacent_probability_l278_27828

def total_arrangements : ℕ := Nat.factorial 5

def adjacent_arrangements : ℕ := 2 * Nat.factorial 4

def probability_not_adjacent : ℚ := 
  1 - (adjacent_arrangements / total_arrangements)

theorem zeros_not_adjacent_probability :
  probability_not_adjacent = 0.6 := 
by 
  sorry

end NUMINAMATH_GPT_zeros_not_adjacent_probability_l278_27828


namespace NUMINAMATH_GPT_smallest_value_n_l278_27834

def factorial_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625 + n / 3125

theorem smallest_value_n
  (a b c m n : ℕ)
  (h1 : a + b + c = 2003)
  (h2 : a = 2 * b)
  (h3 : a.factorial * b.factorial * c.factorial = m * 10 ^ n)
  (h4 : ¬ (10 ∣ m)) :
  n = 400 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_n_l278_27834


namespace NUMINAMATH_GPT_black_balls_in_box_l278_27877

theorem black_balls_in_box (B : ℕ) (probability : ℚ) 
  (h1 : probability = 0.38095238095238093) 
  (h2 : B / (14 + B) = probability) : 
  B = 9 := by
  sorry

end NUMINAMATH_GPT_black_balls_in_box_l278_27877


namespace NUMINAMATH_GPT_prime_of_form_a2_minus_1_l278_27885

theorem prime_of_form_a2_minus_1 (a : ℕ) (p : ℕ) (ha : a ≥ 2) (hp : p = a^2 - 1) (prime_p : Nat.Prime p) : p = 3 := 
by 
  sorry

end NUMINAMATH_GPT_prime_of_form_a2_minus_1_l278_27885


namespace NUMINAMATH_GPT_inequality_proof_l278_27805

theorem inequality_proof {x y : ℝ} (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
by sorry

end NUMINAMATH_GPT_inequality_proof_l278_27805


namespace NUMINAMATH_GPT_find_erased_number_l278_27895

theorem find_erased_number (x : ℕ) (h : 8 * x = 96) : x = 12 := by
  sorry

end NUMINAMATH_GPT_find_erased_number_l278_27895


namespace NUMINAMATH_GPT_same_graph_iff_same_function_D_l278_27893

theorem same_graph_iff_same_function_D :
  ∀ x : ℝ, (|x| = if x ≥ 0 then x else -x) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_same_graph_iff_same_function_D_l278_27893


namespace NUMINAMATH_GPT_inequality_abc_l278_27891

theorem inequality_abc 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a ≥ b ∧ b ≥ c) 
  (h_sum : a + b + c ≤ 1) : 
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 := 
by sorry

end NUMINAMATH_GPT_inequality_abc_l278_27891


namespace NUMINAMATH_GPT_shortest_routes_l278_27866

theorem shortest_routes
  (side_length : ℝ)
  (refuel_distance : ℝ)
  (total_distance : ℝ)
  (shortest_paths : ℕ) :
  side_length = 10 ∧
  refuel_distance = 30 ∧
  total_distance = 180 →
  shortest_paths = 18 :=
sorry

end NUMINAMATH_GPT_shortest_routes_l278_27866


namespace NUMINAMATH_GPT_at_least_one_gt_one_l278_27835

theorem at_least_one_gt_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : x > 1 ∨ y > 1 :=
sorry

end NUMINAMATH_GPT_at_least_one_gt_one_l278_27835


namespace NUMINAMATH_GPT_nonagon_diagonals_count_eq_27_l278_27821

theorem nonagon_diagonals_count_eq_27 :
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  distinct_diagonals = 27 :=
by
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  have : distinct_diagonals = 27 := sorry
  exact this

end NUMINAMATH_GPT_nonagon_diagonals_count_eq_27_l278_27821


namespace NUMINAMATH_GPT_sqrt_sum_simplify_l278_27867

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_simplify_l278_27867


namespace NUMINAMATH_GPT_mn_sum_l278_27854

theorem mn_sum (M N : ℚ) (h1 : (4 : ℚ) / 7 = M / 63) (h2 : (4 : ℚ) / 7 = 84 / N) : M + N = 183 := sorry

end NUMINAMATH_GPT_mn_sum_l278_27854


namespace NUMINAMATH_GPT_hotdogs_total_l278_27812

theorem hotdogs_total:
  let e := 2.5
  let l := 2 * (e * 2)
  let m := 7
  let h := 1.5 * (e * 2)
  let z := 0.5
  (e * 2 + l + m + h + z) = 30 := 
by
  sorry

end NUMINAMATH_GPT_hotdogs_total_l278_27812


namespace NUMINAMATH_GPT_first_digit_base_9_of_y_l278_27899

def base_3_to_base_10 (n : Nat) : Nat := sorry
def base_10_to_base_9_first_digit (n : Nat) : Nat := sorry

theorem first_digit_base_9_of_y :
  let y := 11220022110022112221
  let base_10_y := base_3_to_base_10 y
  base_10_to_base_9_first_digit base_10_y = 4 :=
by
  let y := 11220022110022112221
  let base_10_y := base_3_to_base_10 y
  show base_10_to_base_9_first_digit base_10_y = 4
  sorry

end NUMINAMATH_GPT_first_digit_base_9_of_y_l278_27899


namespace NUMINAMATH_GPT_log_sum_geometric_sequence_l278_27831

open Real

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), a n ≠ 0 ∧ a (n + 1) / a n = a 1 / a 0

theorem log_sum_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_geo : geometric_sequence a) 
  (h_eq : a 10 * a 11 + a 9 * a 12 = 2 * exp 5) : 
  log (a 1) + log (a 2) + log (a 3) + log (a 4) + log (a 5) + 
  log (a 6) + log (a 7) + log (a 8) + log (a 9) + log (a 10) + 
  log (a 11) + log (a 12) + log (a 13) + log (a 14) + log (a 15) + 
  log (a 16) + log (a 17) + log (a 18) + log (a 19) + log (a 20) = 50 :=
sorry

end NUMINAMATH_GPT_log_sum_geometric_sequence_l278_27831


namespace NUMINAMATH_GPT_inequality_example_l278_27807

theorem inequality_example (a b m : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_m_pos : 0 < m) (h_ba : b = 2) (h_aa : a = 1) :
  (b + m) / (a + m) < b / a :=
sorry

end NUMINAMATH_GPT_inequality_example_l278_27807


namespace NUMINAMATH_GPT_lana_eats_fewer_candies_l278_27843

-- Definitions based on conditions
def canEatNellie : ℕ := 12
def canEatJacob : ℕ := canEatNellie / 2
def candiesBeforeLanaCries : ℕ := 6 -- This is the derived answer for Lana
def initialCandies : ℕ := 30
def remainingCandies : ℕ := 3 * 3 -- After division, each gets 3 candies and they are 3 people

-- Statement to prove how many fewer candies Lana can eat compared to Jacob
theorem lana_eats_fewer_candies :
  canEatJacob = 6 → 
  (initialCandies - remainingCandies = 12 + canEatJacob + candiesBeforeLanaCries) →
  canEatJacob - candiesBeforeLanaCries = 3 :=
by
  intros hJacobEats hCandiesAte
  sorry

end NUMINAMATH_GPT_lana_eats_fewer_candies_l278_27843


namespace NUMINAMATH_GPT_compute_x2_y2_l278_27880

theorem compute_x2_y2 (x y : ℝ) (h1 : 1 < x) (h2 : 1 < y)
  (h3 : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 27 = 9 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x^2 + y^2 = 189 := 
by sorry

end NUMINAMATH_GPT_compute_x2_y2_l278_27880


namespace NUMINAMATH_GPT_circle_properties_l278_27825

theorem circle_properties (C : ℝ) (hC : C = 36) :
  let r := 18 / π
  let d := 36 / π
  let A := 324 / π
  2 * π * r = 36 ∧ d = 2 * r ∧ A = π * r^2 :=
by
  sorry

end NUMINAMATH_GPT_circle_properties_l278_27825


namespace NUMINAMATH_GPT_coins_division_remainder_l278_27820

theorem coins_division_remainder :
  ∃ n : ℕ, (n % 8 = 6 ∧ n % 7 = 5 ∧ n % 9 = 0) :=
sorry

end NUMINAMATH_GPT_coins_division_remainder_l278_27820


namespace NUMINAMATH_GPT_repeating_decimal_value_l278_27876

def repeating_decimal : ℝ := 0.0000253253325333 -- Using repeating decimal as given in the conditions

theorem repeating_decimal_value :
  (10^7 - 10^5) * repeating_decimal = 253 / 990 :=
sorry

end NUMINAMATH_GPT_repeating_decimal_value_l278_27876


namespace NUMINAMATH_GPT_triangle_angle_B_max_sin_A_plus_sin_C_l278_27845

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) (h1 : (a - c) * Real.sin A + c * Real.sin C - b * Real.sin B = 0) 
  (h2 : a / Real.sin A = b / Real.sin B) (h3 : b / Real.sin B = c / Real.sin C) : 
  B = Real.arccos (1/2) := 
sorry

theorem max_sin_A_plus_sin_C (a b c : ℝ) (A B C : ℝ) (h1 : (a - c) * Real.sin A + c * Real.sin C - b * Real.sin B = 0) 
  (h2 : a / Real.sin A = b / Real.sin B) (h3 : b / Real.sin B = c / Real.sin C) 
  (hB : B = Real.arccos (1/2)) : 
  Real.sin A + Real.sin C = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_triangle_angle_B_max_sin_A_plus_sin_C_l278_27845


namespace NUMINAMATH_GPT_solution_set_l278_27871

-- Define the intervals for the solution set
def interval1 : Set ℝ := Set.Ico (5/3) 2
def interval2 : Set ℝ := Set.Ico 2 3

-- Define the function that we need to prove
def equation_holds (x : ℝ) : Prop := Int.floor (Int.floor (3 * x) - 1 / 3) = Int.floor (x + 3)

theorem solution_set :
  { x : ℝ | equation_holds x } = interval1 ∪ interval2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_solution_set_l278_27871


namespace NUMINAMATH_GPT_solve_phi_l278_27848

noncomputable def find_phi (phi : ℝ) : Prop :=
  2 * Real.cos phi - Real.sin phi = Real.sqrt 3 * Real.sin (20 / 180 * Real.pi)

theorem solve_phi (phi : ℝ) :
  find_phi phi ↔ (phi = 140 / 180 * Real.pi ∨ phi = 40 / 180 * Real.pi) :=
sorry

end NUMINAMATH_GPT_solve_phi_l278_27848


namespace NUMINAMATH_GPT_fraction_of_upgraded_sensors_l278_27824

theorem fraction_of_upgraded_sensors (N U : ℕ) (h1 : N = U / 6) :
  (U / (24 * N + U)) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_upgraded_sensors_l278_27824


namespace NUMINAMATH_GPT_sequence_arithmetic_l278_27844

theorem sequence_arithmetic (a : ℕ → Real)
    (h₁ : a 3 = 2)
    (h₂ : a 7 = 1)
    (h₃ : ∃ d, ∀ n, 1 / (1 + a (n + 1)) = 1 / (1 + a n) + d):
    a 11 = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sequence_arithmetic_l278_27844


namespace NUMINAMATH_GPT_fixed_point_of_function_l278_27815

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : (a^(2-2) - 3) = -2 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_function_l278_27815


namespace NUMINAMATH_GPT_unique_solution_condition_l278_27884

variable (c d x : ℝ)

-- Define the equation
def equation : Prop := 4 * x - 7 + c = d * x + 3

-- Lean theorem for the proof problem
theorem unique_solution_condition :
  (∃! x, equation c d x) ↔ d ≠ 4 :=
sorry

end NUMINAMATH_GPT_unique_solution_condition_l278_27884


namespace NUMINAMATH_GPT_class_gpa_l278_27841

theorem class_gpa (n : ℕ) (hn : n > 0) (gpa1 : ℝ := 30) (gpa2 : ℝ := 33) : 
    (gpa1 * (n:ℝ) + gpa2 * (2 * n : ℝ)) / (3 * n : ℝ) = 32 :=
by
  sorry

end NUMINAMATH_GPT_class_gpa_l278_27841


namespace NUMINAMATH_GPT_student_ticket_cost_l278_27808

theorem student_ticket_cost (cost_per_student_ticket : ℝ) :
  (12 * cost_per_student_ticket + 4 * 3 = 24) → cost_per_student_ticket = 1 :=
by
  intros h
  -- We should provide a complete proof here, but for illustration, we use sorry.
  sorry

end NUMINAMATH_GPT_student_ticket_cost_l278_27808


namespace NUMINAMATH_GPT_factorize_polynomial_1_factorize_polynomial_2_triangle_shape_l278_27890

-- Proof for (1)
theorem factorize_polynomial_1 (a : ℝ) : 2*a^2 - 8*a + 8 = 2*(a - 2)^2 :=
by
  sorry

-- Proof for (2)
theorem factorize_polynomial_2 (x y : ℝ) : x^2 - y^2 + 3*x - 3*y = (x - y)*(x + y + 3) :=
by
  sorry

-- Proof for (3)
theorem triangle_shape (a b c : ℝ) (h : a^2 - ab - ac + bc = 0) : 
  (a = b ∨ a = c) :=
by
  sorry

end NUMINAMATH_GPT_factorize_polynomial_1_factorize_polynomial_2_triangle_shape_l278_27890
