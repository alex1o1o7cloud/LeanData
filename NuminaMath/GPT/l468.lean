import Mathlib

namespace NUMINAMATH_GPT_odd_n_cube_minus_n_div_by_24_l468_46818

theorem odd_n_cube_minus_n_div_by_24 (n : ℤ) (h_odd : n % 2 = 1) : 24 ∣ (n^3 - n) :=
sorry

end NUMINAMATH_GPT_odd_n_cube_minus_n_div_by_24_l468_46818


namespace NUMINAMATH_GPT_amy_minimum_disks_l468_46864

theorem amy_minimum_disks :
  ∃ (d : ℕ), (d = 19) ∧ ( ∀ (f : ℕ), 
  (f = 40) ∧ ( ∀ (n m k : ℕ), 
  (n + m + k = f) ∧ ( ∀ (a b c : ℕ),
  (a = 8) ∧ (b = 15) ∧ (c = (f - a - b))
  ∧ ( ∀ (size_a size_b size_c : ℚ),
  (size_a = 0.6) ∧ (size_b = 0.55) ∧ (size_c = 0.45)
  ∧ ( ∀ (disk_space : ℚ),
  (disk_space = 1.44)
  ∧ ( ∀ (x y z : ℕ),
  (x = n * ⌈size_a / disk_space⌉) 
  ∧ (y = m * ⌈size_b / disk_space⌉) 
  ∧ (z = k * ⌈size_c / disk_space⌉)
  ∧ (x + y + z = d)) ∧ (size_a * a + size_b * b + size_c * c ≤ disk_space * d)))))) := sorry

end NUMINAMATH_GPT_amy_minimum_disks_l468_46864


namespace NUMINAMATH_GPT_boys_sitting_10_boys_sitting_11_l468_46862

def exists_two_boys_with_4_between (n : ℕ) : Prop :=
  ∃ (b : Finset ℕ), b.card = n ∧ ∀ (i j : ℕ) (h₁ : i ≠ j) (h₂ : i < 25) (h₃ : j < 25),
    (i + 5) % 25 = j

theorem boys_sitting_10 :
  ¬exists_two_boys_with_4_between 10 :=
sorry

theorem boys_sitting_11 :
  exists_two_boys_with_4_between 11 :=
sorry

end NUMINAMATH_GPT_boys_sitting_10_boys_sitting_11_l468_46862


namespace NUMINAMATH_GPT_perimeter_of_garden_l468_46856

-- Define the area of the square garden
def area_square_garden : ℕ := 49

-- Define the relationship between q and p
def q_equals_p_plus_21 (q p : ℕ) : Prop := q = p + 21

-- Define the length of the side of the square garden
def side_length (area : ℕ) : ℕ := Nat.sqrt area

-- Define the perimeter of the square garden
def perimeter (side_length : ℕ) : ℕ := 4 * side_length

-- Define the perimeter of the square garden as a specific perimeter
def specific_perimeter (side_length : ℕ) : ℕ := perimeter side_length

-- Statement of the theorem
theorem perimeter_of_garden (q p : ℕ) (h1 : q = 49) (h2 : q_equals_p_plus_21 q p) : 
  specific_perimeter (side_length 49) = 28 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_garden_l468_46856


namespace NUMINAMATH_GPT_arccos_cos_solution_l468_46836

theorem arccos_cos_solution (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ (Real.pi / 2)) (h₂ : Real.arccos (Real.cos x) = 2 * x) : 
    x = 0 :=
by 
  sorry

end NUMINAMATH_GPT_arccos_cos_solution_l468_46836


namespace NUMINAMATH_GPT_pencil_count_l468_46881

theorem pencil_count (a : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
    (h3 : a % 10 = 7) (h4 : a % 12 = 9) : a = 237 ∨ a = 297 :=
by {
  sorry
}

end NUMINAMATH_GPT_pencil_count_l468_46881


namespace NUMINAMATH_GPT_total_spent_l468_46865

theorem total_spent (puppy_cost dog_food_cost treats_cost_per_bag toys_cost crate_cost bed_cost collar_leash_cost bags_of_treats discount_rate : ℝ) :
  puppy_cost = 20 →
  dog_food_cost = 20 →
  treats_cost_per_bag = 2.5 →
  toys_cost = 15 →
  crate_cost = 20 →
  bed_cost = 20 →
  collar_leash_cost = 15 →
  bags_of_treats = 2 →
  discount_rate = 0.2 →
  (dog_food_cost + treats_cost_per_bag * bags_of_treats + toys_cost + crate_cost + bed_cost + collar_leash_cost) * (1 - discount_rate) + puppy_cost = 96 :=
by sorry

end NUMINAMATH_GPT_total_spent_l468_46865


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l468_46824

theorem negation_of_universal_proposition (x : ℝ) :
  ¬ (∀ m : ℝ, 0 ≤ m ∧ m ≤ 1 → x + 1 / x ≥ 2^m) ↔ ∃ m : ℝ, (0 ≤ m ∧ m ≤ 1) ∧ (x + 1 / x < 2^m) := by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l468_46824


namespace NUMINAMATH_GPT_pears_sales_l468_46815

variable (x : ℝ)
variable (morning_sales : ℝ := x)
variable (afternoon_sales : ℝ := 2 * x)
variable (evening_sales : ℝ := 3 * afternoon_sales)
variable (total_sales : ℝ := morning_sales + afternoon_sales + evening_sales)

theorem pears_sales :
  (total_sales = 510) →
  (afternoon_sales = 113.34) :=
by
  sorry

end NUMINAMATH_GPT_pears_sales_l468_46815


namespace NUMINAMATH_GPT_total_fish_caught_l468_46845

theorem total_fish_caught (leo_fish : ℕ) (agrey_fish : ℕ) (h1 : leo_fish = 40) (h2 : agrey_fish = leo_fish + 20) :
  leo_fish + agrey_fish = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_fish_caught_l468_46845


namespace NUMINAMATH_GPT_sum_of_first_n_terms_sequence_l468_46826

open Nat

def sequence_term (i : ℕ) : ℚ :=
  if i = 0 then 0 else 1 / (i * (i + 1) / 2 : ℕ)

def sum_of_sequence (n : ℕ) : ℚ :=
  (Finset.range (n+1)).sum fun i => sequence_term i

theorem sum_of_first_n_terms_sequence (n : ℕ) : sum_of_sequence n = 2 * n / (n + 1) := by
  sorry

end NUMINAMATH_GPT_sum_of_first_n_terms_sequence_l468_46826


namespace NUMINAMATH_GPT_solution_set_of_f_prime_gt_zero_l468_46811

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4 * Real.log x

theorem solution_set_of_f_prime_gt_zero :
  {x : ℝ | 0 < x ∧ 2*x - 2 - (4 / x) > 0} = {x : ℝ | 2 < x} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_f_prime_gt_zero_l468_46811


namespace NUMINAMATH_GPT_two_triangles_not_separable_by_plane_l468_46872

/-- Definition of a point in three-dimensional space -/
structure Point (α : Type) :=
(x : α)
(y : α)
(z : α)

/-- Definition of a segment joining two points -/
structure Segment (α : Type) :=
(p1 : Point α)
(p2 : Point α)

/-- Definition of a triangle formed by three points -/
structure Triangle (α : Type) :=
(a : Point α)
(b : Point α)
(c : Point α)

/-- Definition of a plane given by a normal vector and a point on the plane -/
structure Plane (α : Type) :=
(n : Point α)
(p : Point α)

/-- Definition of separation of two triangles by a plane -/
def separates (plane : Plane ℝ) (t1 t2 : Triangle ℝ) : Prop :=
  -- Placeholder for the actual separation condition
  sorry

/-- The theorem to be proved -/
theorem two_triangles_not_separable_by_plane (points : Fin 6 → Point ℝ) :
  ∃ t1 t2 : Triangle ℝ, ¬∃ plane : Plane ℝ, separates plane t1 t2 :=
sorry

end NUMINAMATH_GPT_two_triangles_not_separable_by_plane_l468_46872


namespace NUMINAMATH_GPT_experiment_success_probability_l468_46833

/-- 
There are three boxes, each containing 10 balls. 
- The first box contains 7 balls marked 'A' and 3 balls marked 'B'.
- The second box contains 5 red balls and 5 white balls.
- The third box contains 8 red balls and 2 white balls.

The experiment consists of:
1. Drawing a ball from the first box.
2. If a ball marked 'A' is drawn, drawing from the second box.
3. If a ball marked 'B' is drawn, drawing from the third box.
The experiment is successful if the second ball drawn is red.

Prove that the probability of the experiment being successful is 0.59.
-/
theorem experiment_success_probability (P : ℝ) : 
  P = 0.59 :=
sorry

end NUMINAMATH_GPT_experiment_success_probability_l468_46833


namespace NUMINAMATH_GPT_find_a_l468_46841

theorem find_a (a : ℚ) : (∃ b : ℚ, 4 * (x : ℚ)^2 + 14 * x + a = (2 * x + b)^2) → a = 49 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l468_46841


namespace NUMINAMATH_GPT_vacation_days_l468_46800

-- A plane ticket costs $24 for each person
def plane_ticket_cost : ℕ := 24

-- A hotel stay costs $12 for each person per day
def hotel_stay_cost_per_day : ℕ := 12

-- Total vacation cost is $120
def total_vacation_cost : ℕ := 120

-- The number of days they are planning to stay is 3
def number_of_days : ℕ := 3

-- Prove that given the conditions, the number of days (d) they plan to stay satisfies the total vacation cost
theorem vacation_days (d : ℕ) (plane_ticket_cost hotel_stay_cost_per_day total_vacation_cost : ℕ) 
  (h1 : plane_ticket_cost = 24)
  (h2 : hotel_stay_cost_per_day = 12) 
  (h3 : total_vacation_cost = 120) 
  (h4 : 2 * plane_ticket_cost + (2 * hotel_stay_cost_per_day) * d = total_vacation_cost)
  : d = 3 := sorry

end NUMINAMATH_GPT_vacation_days_l468_46800


namespace NUMINAMATH_GPT_cone_surface_area_ratio_l468_46848

noncomputable def sector_angle := 135
noncomputable def sector_area (B : ℝ) := B
noncomputable def cone (A : ℝ) (B : ℝ) := A

theorem cone_surface_area_ratio (A B : ℝ) (h_sector_angle: sector_angle = 135) (h_sector_area: sector_area B = B) (h_cone_formed: cone A B = A) :
  A / B = 11 / 8 :=
by
  sorry

end NUMINAMATH_GPT_cone_surface_area_ratio_l468_46848


namespace NUMINAMATH_GPT_m_divides_n_l468_46896

theorem m_divides_n 
  (m n : ℕ) 
  (hm_pos : 0 < m) 
  (hn_pos : 0 < n) 
  (h : 5 * m + n ∣ 5 * n + m) 
  : m ∣ n :=
sorry

end NUMINAMATH_GPT_m_divides_n_l468_46896


namespace NUMINAMATH_GPT_solution_to_problem_l468_46899

def problem_statement : Prop :=
  (2.017 * 2016 - 10.16 * 201.7 = 2017)

theorem solution_to_problem : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_solution_to_problem_l468_46899


namespace NUMINAMATH_GPT_enclosed_region_area_l468_46861

theorem enclosed_region_area :
  (∃ x y : ℝ, x ^ 2 + y ^ 2 - 6 * x + 8 * y = -9) →
  ∃ (r : ℝ), r ^ 2 = 16 ∧ ∀ (area : ℝ), area = π * 4 ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_enclosed_region_area_l468_46861


namespace NUMINAMATH_GPT_new_circle_radius_shaded_region_l468_46858

theorem new_circle_radius_shaded_region {r1 r2 : ℝ} 
    (h1 : r1 = 35) 
    (h2 : r2 = 24) : 
    ∃ r : ℝ, π * r^2 = π * (r1^2 - r2^2) ∧ r = Real.sqrt 649 := 
by
  sorry

end NUMINAMATH_GPT_new_circle_radius_shaded_region_l468_46858


namespace NUMINAMATH_GPT_num_regular_soda_l468_46882

theorem num_regular_soda (t d r : ℕ) (h₁ : t = 17) (h₂ : d = 8) (h₃ : r = t - d) : r = 9 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end NUMINAMATH_GPT_num_regular_soda_l468_46882


namespace NUMINAMATH_GPT_factor_x4_plus_81_l468_46816

theorem factor_x4_plus_81 (x : ℝ) : (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9) = x^4 + 81 := 
by 
   sorry

end NUMINAMATH_GPT_factor_x4_plus_81_l468_46816


namespace NUMINAMATH_GPT_find_integers_k_l468_46805

theorem find_integers_k (k : ℤ) : 
  (k = 15 ∨ k = 30) ↔ 
  (k ≥ 3 ∧ ∃ m n : ℤ, 1 < m ∧ m < k ∧ 1 < n ∧ n < k ∧ 
                       Int.gcd m k = 1 ∧ Int.gcd n k = 1 ∧ 
                       m + n > k ∧ k ∣ (m - 1) * (n - 1)) :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_find_integers_k_l468_46805


namespace NUMINAMATH_GPT_more_stable_performance_l468_46852

theorem more_stable_performance (S_A2 S_B2 : ℝ) (hA : S_A2 = 0.2) (hB : S_B2 = 0.09) (h : S_A2 > S_B2) : 
  "B" = "B" :=
by
  sorry

end NUMINAMATH_GPT_more_stable_performance_l468_46852


namespace NUMINAMATH_GPT_trader_bags_correct_l468_46859

-- Definitions according to given conditions
def initial_bags := 55
def sold_bags := 23
def restocked_bags := 132

-- Theorem that encapsulates the problem's question and the proven answer
theorem trader_bags_correct :
  (initial_bags - sold_bags + restocked_bags) = 164 :=
by
  sorry

end NUMINAMATH_GPT_trader_bags_correct_l468_46859


namespace NUMINAMATH_GPT_alok_age_l468_46806

theorem alok_age (B A C : ℕ) (h1 : B = 6 * A) (h2 : B + 10 = 2 * (C + 10)) (h3 : C = 10) : A = 5 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_alok_age_l468_46806


namespace NUMINAMATH_GPT_ticket_number_l468_46869

-- Define the conditions and the problem
theorem ticket_number (x y z N : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy: 0 ≤ y ∧ y ≤ 9) (hz: 0 ≤ z ∧ z ≤ 9) 
(hN1: N = 100 * x + 10 * y + z) (hN2: N = 11 * (x + y + z)) : 
N = 198 :=
sorry

end NUMINAMATH_GPT_ticket_number_l468_46869


namespace NUMINAMATH_GPT_tan_11pi_over_6_l468_46847

theorem tan_11pi_over_6 :
  Real.tan (11 * Real.pi / 6) = -1 / Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_11pi_over_6_l468_46847


namespace NUMINAMATH_GPT_desert_area_2020_correct_desert_area_less_8_10_5_by_2023_l468_46885

-- Define initial desert area
def initial_desert_area : ℝ := 9 * 10^5

-- Define increase in desert area each year as observed
def yearly_increase (n : ℕ) : ℝ :=
  match n with
  | 1998 => 2000
  | 1999 => 4000
  | 2000 => 6001
  | 2001 => 7999
  | 2002 => 10001
  | _    => 0

-- Define arithmetic progression of increases
def common_difference : ℝ := 2000

-- Define desert area in 2020
def desert_area_2020 : ℝ :=
  initial_desert_area + 10001 + 18 * common_difference

-- Statement: Desert area by the end of 2020 is approximately 9.46 * 10^5 hm^2
theorem desert_area_2020_correct :
  desert_area_2020 = 9.46 * 10^5 :=
sorry

-- Define yearly transformation and desert increment with afforestation from 2003
def desert_area_with_afforestation (n : ℕ) : ℝ :=
  if n < 2003 then
    initial_desert_area + yearly_increase n
  else
    initial_desert_area + 10001 + (n - 2002) * (common_difference - 8000)

-- Statement: Desert area will be less than 8 * 10^5 hm^2 by the end of 2023
theorem desert_area_less_8_10_5_by_2023 :
  desert_area_with_afforestation 2023 < 8 * 10^5 :=
sorry

end NUMINAMATH_GPT_desert_area_2020_correct_desert_area_less_8_10_5_by_2023_l468_46885


namespace NUMINAMATH_GPT_vector_opposite_direction_and_magnitude_l468_46825

theorem vector_opposite_direction_and_magnitude
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h1 : a = (-1, 2)) 
  (h2 : ∃ k : ℝ, k < 0 ∧ b = k • a) 
  (hb : ‖b‖ = Real.sqrt 5) :
  b = (1, -2) :=
sorry

end NUMINAMATH_GPT_vector_opposite_direction_and_magnitude_l468_46825


namespace NUMINAMATH_GPT_area_of_PQ_square_l468_46898

theorem area_of_PQ_square (a b c : ℕ)
  (h1 : a^2 = 144)
  (h2 : b^2 = 169)
  (h3 : a^2 + c^2 = b^2) :
  c^2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_area_of_PQ_square_l468_46898


namespace NUMINAMATH_GPT_cone_height_of_semicircular_sheet_l468_46894

theorem cone_height_of_semicircular_sheet (R h : ℝ) (h_cond: h = R) : h = R :=
by
  exact h_cond

end NUMINAMATH_GPT_cone_height_of_semicircular_sheet_l468_46894


namespace NUMINAMATH_GPT_contradiction_assumption_l468_46812

-- Define the numbers x, y, z
variables (x y z : ℝ)

-- Define the assumption that all three numbers are non-positive
def all_non_positive (x y z : ℝ) : Prop := x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0

-- State the proposition to prove using the method of contradiction
theorem contradiction_assumption (h : all_non_positive x y z) : ¬ (x > 0 ∨ y > 0 ∨ z > 0) :=
by
  sorry

end NUMINAMATH_GPT_contradiction_assumption_l468_46812


namespace NUMINAMATH_GPT_quadratic_factored_b_l468_46876

theorem quadratic_factored_b (b : ℤ) : 
  (∃ (m n p q : ℤ), 15 * x^2 + b * x + 30 = (m * x + n) * (p * x + q) ∧ m * p = 15 ∧ n * q = 30 ∧ m * q + n * p = b) ↔ b = 43 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_factored_b_l468_46876


namespace NUMINAMATH_GPT_overall_support_percentage_l468_46857

def men_support_percentage : ℝ := 0.75
def women_support_percentage : ℝ := 0.70
def number_of_men : ℕ := 200
def number_of_women : ℕ := 800

theorem overall_support_percentage :
  ((men_support_percentage * ↑number_of_men + women_support_percentage * ↑number_of_women) / (↑number_of_men + ↑number_of_women) * 100) = 71 := 
by 
sorry

end NUMINAMATH_GPT_overall_support_percentage_l468_46857


namespace NUMINAMATH_GPT_tan_alpha_problem_l468_46884

theorem tan_alpha_problem (α : ℝ) (h : Real.tan α = 3) : (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_problem_l468_46884


namespace NUMINAMATH_GPT_lcm_smallest_value_l468_46829

/-- The smallest possible value of lcm(k, l) for positive 5-digit integers k and l such that gcd(k, l) = 5 is 20010000. -/
theorem lcm_smallest_value (k l : ℕ) (h1 : 10000 ≤ k ∧ k < 100000) (h2 : 10000 ≤ l ∧ l < 100000) (h3 : Nat.gcd k l = 5) : Nat.lcm k l = 20010000 :=
sorry

end NUMINAMATH_GPT_lcm_smallest_value_l468_46829


namespace NUMINAMATH_GPT_find_sum_of_angles_l468_46883

open Real

namespace math_problem

theorem find_sum_of_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : cos (α - β / 2) = sqrt 3 / 2)
  (h2 : sin (α / 2 - β) = -1 / 2) : α + β = 2 * π / 3 :=
sorry

end math_problem

end NUMINAMATH_GPT_find_sum_of_angles_l468_46883


namespace NUMINAMATH_GPT_hyperbola_foci_l468_46802

/-- Define a hyperbola -/
def hyperbola_eq (x y : ℝ) : Prop := 4 * y^2 - 25 * x^2 = 100

/-- Definition of the foci of the hyperbola -/
def foci_coords (c : ℝ) : Prop := c = Real.sqrt 29

/-- Proof that the foci of the hyperbola 4y^2 - 25x^2 = 100 are (0, -sqrt(29)) and (0, sqrt(29)) -/
theorem hyperbola_foci (x y : ℝ) (c : ℝ) (hx : hyperbola_eq x y) (hc : foci_coords c) :
  (x = 0 ∧ (y = -c ∨ y = c)) :=
sorry

end NUMINAMATH_GPT_hyperbola_foci_l468_46802


namespace NUMINAMATH_GPT_speed_of_first_car_l468_46813

theorem speed_of_first_car (v : ℝ) 
  (h1 : ∀ v, v > 0 → (first_speed = 1.25 * v))
  (h2 : 720 = (v + 1.25 * v) * 4) : 
  first_speed = 100 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_first_car_l468_46813


namespace NUMINAMATH_GPT_copper_price_l468_46807

theorem copper_price (c : ℕ) (hzinc : ℕ) (zinc_weight : ℕ) (brass_weight : ℕ) (price_brass : ℕ) (used_copper : ℕ) :
  hzinc = 30 →
  zinc_weight = brass_weight - used_copper →
  brass_weight = 70 →
  price_brass = 45 →
  used_copper = 30 →
  (used_copper * c + zinc_weight * hzinc) = brass_weight * price_brass →
  c = 65 :=
by
  sorry

end NUMINAMATH_GPT_copper_price_l468_46807


namespace NUMINAMATH_GPT_factor_quadratic_l468_46880

theorem factor_quadratic (x : ℝ) : 
  (x^2 + 6 * x + 9 - 16 * x^4) = (-4 * x^2 + 2 * x + 3) * (4 * x^2 + 2 * x + 3) := 
by 
  sorry

end NUMINAMATH_GPT_factor_quadratic_l468_46880


namespace NUMINAMATH_GPT_original_profit_percentage_is_10_l468_46895

-- Define the conditions and the theorem
theorem original_profit_percentage_is_10
  (original_selling_price : ℝ)
  (price_reduction: ℝ)
  (additional_profit: ℝ)
  (profit_percentage: ℝ)
  (new_profit_percentage: ℝ)
  (new_selling_price: ℝ) :
  original_selling_price = 659.9999999999994 →
  price_reduction = 0.10 →
  additional_profit = 42 →
  profit_percentage = 30 →
  new_profit_percentage = 1.30 →
  new_selling_price = original_selling_price + additional_profit →
  ((original_selling_price / (original_selling_price / (new_profit_percentage * (1 - price_reduction)))) - 1) * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_original_profit_percentage_is_10_l468_46895


namespace NUMINAMATH_GPT_fraction_to_decimal_l468_46832

theorem fraction_to_decimal : (58 / 125 : ℚ) = 0.464 := 
by {
  -- proof omitted
  sorry
}

end NUMINAMATH_GPT_fraction_to_decimal_l468_46832


namespace NUMINAMATH_GPT_absolute_value_positive_l468_46867

theorem absolute_value_positive (a : ℝ) (h : a ≠ 0) : |a| > 0 := by
  sorry

end NUMINAMATH_GPT_absolute_value_positive_l468_46867


namespace NUMINAMATH_GPT_Julie_work_hours_per_week_l468_46886

theorem Julie_work_hours_per_week 
  (hours_summer_per_week : ℕ)
  (weeks_summer : ℕ)
  (total_earnings_summer : ℕ)
  (planned_weeks_school_year : ℕ)
  (needed_income_school_year : ℕ)
  (hourly_wage : ℝ := total_earnings_summer / (hours_summer_per_week * weeks_summer))
  (total_hours_needed_school_year : ℝ := needed_income_school_year / hourly_wage)
  (hours_per_week_needed : ℝ := total_hours_needed_school_year / planned_weeks_school_year) :
  hours_summer_per_week = 60 →
  weeks_summer = 8 →
  total_earnings_summer = 6000 →
  planned_weeks_school_year = 40 →
  needed_income_school_year = 10000 →
  hours_per_week_needed = 20 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_Julie_work_hours_per_week_l468_46886


namespace NUMINAMATH_GPT_small_paintings_completed_l468_46878

variable (S : ℕ)

def uses_paint : Prop :=
  3 * 3 + 2 * S = 17

theorem small_paintings_completed : uses_paint S → S = 4 := by
  intro h
  sorry

end NUMINAMATH_GPT_small_paintings_completed_l468_46878


namespace NUMINAMATH_GPT_no_roots_in_disk_l468_46851

noncomputable def homogeneous_polynomial_deg2 (a b c : ℝ) (x y : ℝ) := a * x^2 + b * x * y + c * y^2
noncomputable def homogeneous_polynomial_deg3 (q : ℝ → ℝ → ℝ) (x y : ℝ) := q x y

theorem no_roots_in_disk 
  (a b c : ℝ) (h_poly_deg2 : ∀ x y, homogeneous_polynomial_deg2 a b c x y = a * x^2 + b * x * y + c * y^2)
  (q : ℝ → ℝ → ℝ) (h_poly_deg3 : ∀ x y, homogeneous_polynomial_deg3 q x y = q x y)
  (h_cond : b^2 < 4 * a * c) :
  ∃ k > 0, ∀ x y, x^2 + y^2 < k → homogeneous_polynomial_deg2 a b c x y ≠ homogeneous_polynomial_deg3 q x y ∨ (x = 0 ∧ y = 0) :=
sorry

end NUMINAMATH_GPT_no_roots_in_disk_l468_46851


namespace NUMINAMATH_GPT_find_original_price_each_stocking_l468_46827

open Real

noncomputable def original_stocking_price (total_stockings total_cost_per_stocking discounted_cost monogramming_cost total_cost : ℝ) : ℝ :=
  let stocking_cost_before_monogramming := total_cost - (total_stockings * monogramming_cost)
  let original_price := stocking_cost_before_monogramming / (total_stockings * discounted_cost)
  original_price

theorem find_original_price_each_stocking :
  original_stocking_price 9 122.22 0.9 5 1035 = 122.22 := by
  sorry

end NUMINAMATH_GPT_find_original_price_each_stocking_l468_46827


namespace NUMINAMATH_GPT_find_wrong_observation_value_l468_46855

theorem find_wrong_observation_value :
  ∃ (wrong_value : ℝ),
    let n := 50
    let mean_initial := 36
    let mean_corrected := 36.54
    let observation_incorrect := 48
    let sum_initial := n * mean_initial
    let sum_corrected := n * mean_corrected
    let difference := sum_corrected - sum_initial
    wrong_value = observation_incorrect - difference := sorry

end NUMINAMATH_GPT_find_wrong_observation_value_l468_46855


namespace NUMINAMATH_GPT_simplified_expression_value_l468_46887

noncomputable def expression (a b : ℝ) : ℝ :=
  3 * a ^ 2 - b ^ 2 - (a ^ 2 - 6 * a) - 2 * (-b ^ 2 + 3 * a)

theorem simplified_expression_value :
  expression (-1/2) 3 = 19 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplified_expression_value_l468_46887


namespace NUMINAMATH_GPT_complete_square_h_l468_46819

theorem complete_square_h (x h : ℝ) :
  (∃ a k : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) → h = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_complete_square_h_l468_46819


namespace NUMINAMATH_GPT_inequality_am_gm_l468_46866

variable (a b c d : ℝ)

theorem inequality_am_gm (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (a * c * d) / (1 - b)^2 + (a * b * d) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9  :=
by    
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l468_46866


namespace NUMINAMATH_GPT_sum_of_solutions_l468_46893

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l468_46893


namespace NUMINAMATH_GPT_david_moore_total_time_l468_46892

-- Given conditions
def david_work_rate := 1 / 12
def days_david_worked_alone := 6
def remaining_work_days_together := 3
def total_work := 1

-- Definition of total time taken for both to complete the job
def combined_total_time := 6

-- Proof problem statement in Lean
theorem david_moore_total_time :
  let d_work_done_alone := days_david_worked_alone * david_work_rate
  let remaining_work := total_work - d_work_done_alone
  let combined_work_rate := remaining_work / remaining_work_days_together
  let moore_work_rate := combined_work_rate - david_work_rate
  let new_combined_work_rate := david_work_rate + moore_work_rate
  total_work / new_combined_work_rate = combined_total_time := by
    sorry

end NUMINAMATH_GPT_david_moore_total_time_l468_46892


namespace NUMINAMATH_GPT_positive_integer_count_l468_46803

theorem positive_integer_count (n : ℕ) :
  ∃ (count : ℕ), (count = 122) ∧ 
  (∀ (k : ℕ), 27 < k ∧ k < 150 → ((150 * k)^40 > k^80 ∧ k^80 > 3^240)) :=
sorry

end NUMINAMATH_GPT_positive_integer_count_l468_46803


namespace NUMINAMATH_GPT_sales_volume_increase_30_units_every_5_yuan_initial_sales_volume_750_units_daily_sales_volume_at_540_yuan_l468_46842

def price_reduction_table : List (ℕ × ℕ) := 
  [(5, 780), (10, 810), (15, 840), (20, 870), (25, 900), (30, 930), (35, 960)]

theorem sales_volume_increase_30_units_every_5_yuan :
  ∀ reduction volume1 volume2, (reduction + 5, volume1) ∈ price_reduction_table →
  (reduction + 10, volume2) ∈ price_reduction_table → volume2 - volume1 = 30 := sorry

theorem initial_sales_volume_750_units :
  (5, 780) ∈ price_reduction_table → (10, 810) ∈ price_reduction_table →
  (0, 750) ∉ price_reduction_table → 780 - 30 = 750 := sorry

theorem daily_sales_volume_at_540_yuan :
  ∀ P₀ P₁ volume, P₀ = 600 → P₁ = 540 → 
  (5, 780) ∈ price_reduction_table → (10, 810) ∈ price_reduction_table →
  (15, 840) ∈ price_reduction_table → (20, 870) ∈ price_reduction_table →
  (25, 900) ∈ price_reduction_table → (30, 930) ∈ price_reduction_table →
  (35, 960) ∈ price_reduction_table →
  volume = 750 + (P₀ - P₁) / 5 * 30 → volume = 1110 := sorry

end NUMINAMATH_GPT_sales_volume_increase_30_units_every_5_yuan_initial_sales_volume_750_units_daily_sales_volume_at_540_yuan_l468_46842


namespace NUMINAMATH_GPT_suff_but_not_necc_condition_l468_46897

def x_sq_minus_1_pos (x : ℝ) : Prop := x^2 - 1 > 0
def x_minus_1_pos (x : ℝ) : Prop := x - 1 > 0

theorem suff_but_not_necc_condition : 
  (∀ x : ℝ, x_minus_1_pos x → x_sq_minus_1_pos x) ∧
  (∃ x : ℝ, x_sq_minus_1_pos x ∧ ¬ x_minus_1_pos x) :=
by 
  sorry

end NUMINAMATH_GPT_suff_but_not_necc_condition_l468_46897


namespace NUMINAMATH_GPT_ratio_b_to_c_l468_46838

variable (a b c k : ℕ)

-- Conditions
def condition1 : Prop := a = b + 2
def condition2 : Prop := b = k * c
def condition3 : Prop := a + b + c = 32
def condition4 : Prop := b = 12

-- Question: Prove that ratio of b to c is 2:1
theorem ratio_b_to_c
  (h1 : condition1 a b)
  (h2 : condition2 b k c)
  (h3 : condition3 a b c)
  (h4 : condition4 b) :
  b = 2 * c := 
sorry

end NUMINAMATH_GPT_ratio_b_to_c_l468_46838


namespace NUMINAMATH_GPT_probability_solution_l468_46801

noncomputable def binom_10_7 := Nat.choose 10 7
noncomputable def binom_10_6 := Nat.choose 10 6

theorem probability_solution (p q : ℝ) (h₁ : q = 1 - p) (h₂ : binom_10_7 = 120) (h₃ : binom_10_6 = 210)
  (h₄ : 120 * p ^ 7 * q ^ 3 = 210 * p ^ 6 * q ^ 4) : p = 7 / 11 := 
sorry

end NUMINAMATH_GPT_probability_solution_l468_46801


namespace NUMINAMATH_GPT_total_cats_and_kittens_received_l468_46889

theorem total_cats_and_kittens_received (total_adult_cats : ℕ) (percentage_female : ℕ) (fraction_with_kittens : ℚ) (kittens_per_litter : ℕ) 
  (h1 : total_adult_cats = 100) (h2 : percentage_female = 40) (h3 : fraction_with_kittens = 2 / 3) (h4 : kittens_per_litter = 3) :
  total_adult_cats + ((percentage_female * total_adult_cats / 100) * (fraction_with_kittens * total_adult_cats * kittens_per_litter) / 100) = 181 := by
  sorry

end NUMINAMATH_GPT_total_cats_and_kittens_received_l468_46889


namespace NUMINAMATH_GPT_non_neg_reals_inequality_l468_46873

theorem non_neg_reals_inequality (a b c : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h₃ : a + b + c ≤ 3) :
  (a / (1 + a^2) + b / (1 + b^2) + c / (1 + c^2) ≤ 3/2) ∧
  (3/2 ≤ (1 / (1 + a) + 1 / (1 + b) + 1 / (1 + c))) :=
by
  sorry

end NUMINAMATH_GPT_non_neg_reals_inequality_l468_46873


namespace NUMINAMATH_GPT_minimum_value_frac_inverse_l468_46814

theorem minimum_value_frac_inverse (a b c : ℝ) (h : a + b + c = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a + b)) + (1 / c) ≥ 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_frac_inverse_l468_46814


namespace NUMINAMATH_GPT_greatest_product_l468_46828

theorem greatest_product (x : ℤ) (h : x + (2020 - x) = 2020) : x * (2020 - x) ≤ 1020100 :=
sorry

end NUMINAMATH_GPT_greatest_product_l468_46828


namespace NUMINAMATH_GPT_product_divisible_by_5_l468_46810

theorem product_divisible_by_5 (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h : ∃ k, a * b = 5 * k) : a % 5 = 0 ∨ b % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_divisible_by_5_l468_46810


namespace NUMINAMATH_GPT_cat_ratio_l468_46860

theorem cat_ratio (jacob_cats annie_cats melanie_cats : ℕ)
  (H1 : jacob_cats = 90)
  (H2 : annie_cats = jacob_cats / 3)
  (H3 : melanie_cats = 60) :
  melanie_cats / annie_cats = 2 := 
  by
  sorry

end NUMINAMATH_GPT_cat_ratio_l468_46860


namespace NUMINAMATH_GPT_value_of_a_plus_c_l468_46870

-- Define the polynomials
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- Define the condition for the vertex of polynomial f being a root of g
def vertex_of_f_is_root_of_g (a b c d : ℝ) : Prop :=
  g c d (-a / 2) = 0

-- Define the condition for the vertex of polynomial g being a root of f
def vertex_of_g_is_root_of_f (a b c d : ℝ) : Prop :=
  f a b (-c / 2) = 0

-- Define the condition that both polynomials have the same minimum value
def same_minimum_value (a b c d : ℝ) : Prop :=
  f a b (-a / 2) = g c d (-c / 2)

-- Define the condition that the polynomials intersect at (100, -100)
def polynomials_intersect (a b c d : ℝ) : Prop :=
  f a b 100 = -100 ∧ g c d 100 = -100

-- Lean theorem statement for the problem
theorem value_of_a_plus_c (a b c d : ℝ) 
  (h1 : vertex_of_f_is_root_of_g a b c d)
  (h2 : vertex_of_g_is_root_of_f a b c d)
  (h3 : same_minimum_value a b c d)
  (h4 : polynomials_intersect a b c d) :
  a + c = -400 := 
sorry

end NUMINAMATH_GPT_value_of_a_plus_c_l468_46870


namespace NUMINAMATH_GPT_total_travel_cost_is_47100_l468_46877

-- Define the dimensions of the lawn
def lawn_length : ℝ := 200
def lawn_breadth : ℝ := 150

-- Define the roads' widths and their respective travel costs per sq m
def road1_width : ℝ := 12
def road1_travel_cost : ℝ := 4
def road2_width : ℝ := 15
def road2_travel_cost : ℝ := 5
def road3_width : ℝ := 10
def road3_travel_cost : ℝ := 3
def road4_width : ℝ := 20
def road4_travel_cost : ℝ := 6

-- Define the areas of the roads
def road1_area : ℝ := lawn_length * road1_width
def road2_area : ℝ := lawn_length * road2_width
def road3_area : ℝ := lawn_breadth * road3_width
def road4_area : ℝ := lawn_breadth * road4_width

-- Define the costs for the roads
def road1_cost : ℝ := road1_area * road1_travel_cost
def road2_cost : ℝ := road2_area * road2_travel_cost
def road3_cost : ℝ := road3_area * road3_travel_cost
def road4_cost : ℝ := road4_area * road4_travel_cost

-- Define the total cost
def total_cost : ℝ := road1_cost + road2_cost + road3_cost + road4_cost

-- The theorem statement
theorem total_travel_cost_is_47100 : total_cost = 47100 := by
  sorry

end NUMINAMATH_GPT_total_travel_cost_is_47100_l468_46877


namespace NUMINAMATH_GPT_sum_of_extrema_l468_46849

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

-- Main statement to prove
theorem sum_of_extrema :
  let a := -1
  let b := 1
  let f_min := f a
  let f_max := f b
  f_min + f_max = Real.exp 1 + Real.exp (-1) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_extrema_l468_46849


namespace NUMINAMATH_GPT_max_revenue_l468_46817

variable (x y : ℝ)

-- Conditions
def ads_time_constraint := x + y ≤ 300
def ads_cost_constraint := 500 * x + 200 * y ≤ 90000
def revenue := 0.3 * x + 0.2 * y

-- Question: Prove that the maximum revenue is 70 million yuan
theorem max_revenue (h_time : ads_time_constraint (x := 100) (y := 200))
                    (h_cost : ads_cost_constraint (x := 100) (y := 200)) :
  revenue (x := 100) (y := 200) = 70 := 
sorry

end NUMINAMATH_GPT_max_revenue_l468_46817


namespace NUMINAMATH_GPT_find_x0_l468_46821

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem find_x0 (x_0 : ℝ) (h : f' x_0 = 2) : x_0 = Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x0_l468_46821


namespace NUMINAMATH_GPT_Kim_drink_amount_l468_46863

namespace MathProof

-- Define the conditions
variable (milk_initial t_drinks k_drinks : ℚ)
variable (H1 : milk_initial = 3/4)
variable (H2 : t_drinks = 1/3 * milk_initial)
variable (H3 : k_drinks = 1/2 * (milk_initial - t_drinks))

-- Theorem statement
theorem Kim_drink_amount : k_drinks = 1/4 :=
by
  sorry -- Proof steps would go here, but we're just setting up the statement

end MathProof

end NUMINAMATH_GPT_Kim_drink_amount_l468_46863


namespace NUMINAMATH_GPT_number_added_to_x_is_2_l468_46871

/-- Prove that in a set of integers {x, x + y, x + 4, x + 7, x + 22}, 
    where the mean is 3 greater than the median, the number added to x 
    to get the second integer is 2. --/

theorem number_added_to_x_is_2 (x y : ℤ) (h_pos : 0 < x ∧ 0 < y) 
  (h_median : (x + 4) = ((x + y) + (x + (x + y) + (x + 4) + (x + 7) + (x + 22)) / 5 - 3)) : 
  y = 2 := by
  sorry

end NUMINAMATH_GPT_number_added_to_x_is_2_l468_46871


namespace NUMINAMATH_GPT_quadratic_decreasing_on_nonneg_real_l468_46874

theorem quadratic_decreasing_on_nonneg_real (a b c : ℝ) (h_a : a < 0) (h_b : b < 0) : 
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → (a * x^2 + b * x + c) ≥ (a * y^2 + b * y + c) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_decreasing_on_nonneg_real_l468_46874


namespace NUMINAMATH_GPT_march_first_is_sunday_l468_46843

theorem march_first_is_sunday (days_in_march : ℕ) (num_wednesdays : ℕ) (num_saturdays : ℕ) 
  (h1 : days_in_march = 31) (h2 : num_wednesdays = 4) (h3 : num_saturdays = 4) : 
  ∃ d : ℕ, d = 0 := 
by 
  sorry

end NUMINAMATH_GPT_march_first_is_sunday_l468_46843


namespace NUMINAMATH_GPT_remainder_when_7n_divided_by_4_l468_46840

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end NUMINAMATH_GPT_remainder_when_7n_divided_by_4_l468_46840


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l468_46808

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 : ℝ) = 2 →
  a^2 = 2 * b^2 →
  (c : ℝ) = Real.sqrt (a^2 + b^2) →
  Real.sqrt (a^2 + b^2) = Real.sqrt (3 / 2 * a^2) →
  (e : ℝ) = c / a →
  e = Real.sqrt (6) / 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l468_46808


namespace NUMINAMATH_GPT_pete_books_ratio_l468_46809

theorem pete_books_ratio 
  (M_last : ℝ) (P_last P_this_year M_this_year : ℝ)
  (h1 : P_last = 2 * M_last)
  (h2 : M_this_year = 1.5 * M_last)
  (h3 : P_last + P_this_year = 300)
  (h4 : M_this_year = 75) :
  P_this_year / P_last = 2 :=
by
  sorry

end NUMINAMATH_GPT_pete_books_ratio_l468_46809


namespace NUMINAMATH_GPT_next_in_sequence_is_65_by_19_l468_46868

section
  open Int

  -- Definitions for numerators
  def numerator_sequence : ℕ → ℤ
  | 0 => -3
  | 1 => 5
  | 2 => -9
  | 3 => 17
  | 4 => -33
  | (n + 5) => numerator_sequence n * (-2) + 1

  -- Definitions for denominators
  def denominator_sequence : ℕ → ℕ
  | 0 => 4
  | 1 => 7
  | 2 => 10
  | 3 => 13
  | 4 => 16
  | (n + 5) => denominator_sequence n + 3

  -- Next term in the sequence
  def next_term (n : ℕ) : ℚ :=
    (numerator_sequence (n + 5) : ℚ) / (denominator_sequence (n + 5) : ℚ)

  -- Theorem stating the next number in the sequence
  theorem next_in_sequence_is_65_by_19 :
    next_term 0 = 65 / 19 :=
  by
    unfold next_term
    simp [numerator_sequence, denominator_sequence]
    sorry
end

end NUMINAMATH_GPT_next_in_sequence_is_65_by_19_l468_46868


namespace NUMINAMATH_GPT_sum_of_first_four_terms_of_geometric_sequence_l468_46853

noncomputable def geometric_sum_first_four (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_of_first_four_terms_of_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 : q > 0) 
  (h3 : a 2 = 1) 
  (h4 : ∀ n, a (n + 2) + a (n + 1) = 6 * a n) :
  geometric_sum_first_four a q = 15 / 2 :=
sorry

end NUMINAMATH_GPT_sum_of_first_four_terms_of_geometric_sequence_l468_46853


namespace NUMINAMATH_GPT_total_legs_walking_on_ground_l468_46823

def horses : ℕ := 16
def men : ℕ := 16

def men_walking := men / 2
def men_riding := men / 2

def legs_per_man := 2
def legs_per_horse := 4

def legs_for_men_walking := men_walking * legs_per_man
def legs_for_horses := horses * legs_per_horse

theorem total_legs_walking_on_ground : legs_for_men_walking + legs_for_horses = 80 := 
by
  sorry

end NUMINAMATH_GPT_total_legs_walking_on_ground_l468_46823


namespace NUMINAMATH_GPT_average_temperature_l468_46844

theorem average_temperature (T : Fin 5 → ℝ) (h : T = ![52, 67, 55, 59, 48]) :
    (1 / 5) * (T 0 + T 1 + T 2 + T 3 + T 4) = 56.2 := by
  sorry

end NUMINAMATH_GPT_average_temperature_l468_46844


namespace NUMINAMATH_GPT_find_a_l468_46837

def A : Set ℝ := {0, 2}
def B (a : ℝ) : Set ℝ := {1, a ^ 2}

theorem find_a (a : ℝ) (h : A ∪ B a = {0, 1, 2, 4}) : a = 2 ∨ a = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l468_46837


namespace NUMINAMATH_GPT_find_square_sum_l468_46875

theorem find_square_sum (x y z : ℝ)
  (h1 : x^2 - 6 * y = 10)
  (h2 : y^2 - 8 * z = -18)
  (h3 : z^2 - 10 * x = -40) :
  x^2 + y^2 + z^2 = 50 :=
sorry

end NUMINAMATH_GPT_find_square_sum_l468_46875


namespace NUMINAMATH_GPT_trapezoid_height_l468_46891

theorem trapezoid_height (AD BC : ℝ) (AB CD : ℝ) (h₁ : AD = 25) (h₂ : BC = 4) (h₃ : AB = 20) (h₄ : CD = 13) : ∃ h : ℝ, h = 12 :=
by
  -- Definitions
  let AD := 25
  let BC := 4
  let AB := 20
  let CD := 13
  
  sorry

end NUMINAMATH_GPT_trapezoid_height_l468_46891


namespace NUMINAMATH_GPT_restore_original_price_l468_46830

-- Defining the original price of the jacket
def original_price (P : ℝ) := P

-- Defining the price after each step of reduction
def price_after_first_reduction (P : ℝ) := P * (1 - 0.25)
def price_after_second_reduction (P : ℝ) := price_after_first_reduction P * (1 - 0.20)
def price_after_third_reduction (P : ℝ) := price_after_second_reduction P * (1 - 0.10)

-- Express the condition to restore the original price
theorem restore_original_price (P : ℝ) (x : ℝ) : 
  original_price P = price_after_third_reduction P * (1 + x) → 
  x = 0.85185185 := 
by
  sorry

end NUMINAMATH_GPT_restore_original_price_l468_46830


namespace NUMINAMATH_GPT_height_of_Brixton_l468_46831

theorem height_of_Brixton
  (I Z B Zr : ℕ)
  (h1 : I = Z + 4)
  (h2 : Z = B - 8)
  (h3 : Zr = B)
  (h4 : (I + Z + B + Zr) / 4 = 61) :
  B = 64 := by
  sorry

end NUMINAMATH_GPT_height_of_Brixton_l468_46831


namespace NUMINAMATH_GPT_ratio_of_new_circumference_to_increase_in_area_l468_46804

theorem ratio_of_new_circumference_to_increase_in_area
  (r k : ℝ) (h_k : 0 < k) :
  (2 * π * (r + k)) / (π * (2 * r * k + k ^ 2)) = 2 * (r + k) / (2 * r * k + k ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_new_circumference_to_increase_in_area_l468_46804


namespace NUMINAMATH_GPT_intersection_A_B_l468_46850

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := { x | ∃ k : ℤ, x = 3 * k - 1 }

theorem intersection_A_B :
  A ∩ B = {-1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l468_46850


namespace NUMINAMATH_GPT_trig_problems_l468_46846

variable {A B C : ℝ}
variable {a b c : ℝ}

-- The main theorem statement to prove the magnitude of angle B and find b under given conditions.
theorem trig_problems
  (h₁ : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h₂ : a = Real.sqrt 3)
  (h₃ : c = Real.sqrt 3) :
  Real.cos B = 1 / 2 ∧ b = Real.sqrt 3 := by
sorry

end NUMINAMATH_GPT_trig_problems_l468_46846


namespace NUMINAMATH_GPT_odd_function_and_monotonic_decreasing_l468_46888

variable (f : ℝ → ℝ)

-- Given conditions:
axiom condition_1 : ∀ x y : ℝ, f (x + y) = f x + f y
axiom condition_2 : ∀ x : ℝ, x > 0 → f x < 0

-- Statement to prove:
theorem odd_function_and_monotonic_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2) := by
  sorry

end NUMINAMATH_GPT_odd_function_and_monotonic_decreasing_l468_46888


namespace NUMINAMATH_GPT_time_in_vancouver_l468_46879

theorem time_in_vancouver (toronto_time vancouver_time : ℕ) (h : toronto_time = 18 + 30 / 60) (h_diff : vancouver_time = toronto_time - 3) :
  vancouver_time = 15 + 30 / 60 :=
by
  sorry

end NUMINAMATH_GPT_time_in_vancouver_l468_46879


namespace NUMINAMATH_GPT_polynomial_factorization_l468_46820

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 + (a - b)^2 * (b - c)^2 * (c - a)^2
  = (a - b) * (b - c) * (c - a) * (a + b + c + a * b * c) :=
sorry

end NUMINAMATH_GPT_polynomial_factorization_l468_46820


namespace NUMINAMATH_GPT_reciprocal_of_sum_of_fraction_l468_46839

theorem reciprocal_of_sum_of_fraction (y : ℚ) (h : y = 6 + 1/6) : 1 / y = 6 / 37 := by
  sorry

end NUMINAMATH_GPT_reciprocal_of_sum_of_fraction_l468_46839


namespace NUMINAMATH_GPT_number_of_masters_students_l468_46890

theorem number_of_masters_students (total_sample : ℕ) (ratio_assoc : ℕ) (ratio_undergrad : ℕ) (ratio_masters : ℕ) (ratio_doctoral : ℕ) 
(h1 : ratio_assoc = 5) (h2 : ratio_undergrad = 15) (h3 : ratio_masters = 9) (h4 : ratio_doctoral = 1) (h_total_sample : total_sample = 120) :
  (ratio_masters * total_sample) / (ratio_assoc + ratio_undergrad + ratio_masters + ratio_doctoral) = 36 :=
by
  sorry

end NUMINAMATH_GPT_number_of_masters_students_l468_46890


namespace NUMINAMATH_GPT_min_ring_cuts_l468_46834

/-- Prove that the minimum number of cuts needed to pay the owner daily with an increasing 
    number of rings for 11 days, given a chain of 11 rings, is 2. -/
theorem min_ring_cuts {days : ℕ} {rings : ℕ} : days = 11 → rings = 11 → (∃ cuts : ℕ, cuts = 2) :=
by intros; sorry

end NUMINAMATH_GPT_min_ring_cuts_l468_46834


namespace NUMINAMATH_GPT_monotonic_intervals_range_of_m_l468_46822

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 3 - 2 * x)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 2 * x + m - 3

theorem monotonic_intervals :
  ∀ k : ℤ,
    (
      (∀ x, -Real.pi / 12 + k * Real.pi ≤ x ∧ x ≤ 5 * Real.pi / 12 + k * Real.pi → ∃ (d : ℝ), f x = d)
      ∧
      (∀ x, 5 * Real.pi / 12 + k * Real.pi ≤ x ∧ x ≤ 11 * Real.pi / 12 + k * Real.pi → ∃ (i : ℝ), f x = i)
    ) := sorry

theorem range_of_m (m : ℝ) :
  (∀ x1 : ℝ, Real.pi / 12 ≤ x1 ∧ x1 ≤ Real.pi / 2 → ∃ x2 : ℝ, -2 ≤ x2 ∧ x2 ≤ m ∧ f x1 = g x2 m) ↔ -1 ≤ m ∧ m ≤ 3 := sorry

end NUMINAMATH_GPT_monotonic_intervals_range_of_m_l468_46822


namespace NUMINAMATH_GPT_find_f1_and_f1_l468_46835

theorem find_f1_and_f1' (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_deriv : ∀ x, deriv f x = f' x)
  (h_eq : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f 1 + f' 1 = -3 :=
by sorry

end NUMINAMATH_GPT_find_f1_and_f1_l468_46835


namespace NUMINAMATH_GPT_circles_intersect_if_and_only_if_l468_46854

noncomputable def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 10 * y + 1 = 0

noncomputable def circle2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 2 * y - m = 0

theorem circles_intersect_if_and_only_if (m : ℝ) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) ↔ -1 < m ∧ m < 79 :=
by {
  sorry
}

end NUMINAMATH_GPT_circles_intersect_if_and_only_if_l468_46854
