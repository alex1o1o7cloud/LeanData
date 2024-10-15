import Mathlib

namespace NUMINAMATH_GPT_caleb_puffs_to_mom_l2230_223099

variable (initial_puffs : ℕ) (puffs_to_sister : ℕ) (puffs_to_grandmother : ℕ) (puffs_to_dog : ℕ)
variable (puffs_per_friend : ℕ) (friends : ℕ)

theorem caleb_puffs_to_mom
  (h1 : initial_puffs = 40) 
  (h2 : puffs_to_sister = 3)
  (h3 : puffs_to_grandmother = 5) 
  (h4 : puffs_to_dog = 2) 
  (h5 : puffs_per_friend = 9)
  (h6 : friends = 3)
  : initial_puffs - ( friends * puffs_per_friend + puffs_to_sister + puffs_to_grandmother + puffs_to_dog ) = 3 :=
by
  sorry

end NUMINAMATH_GPT_caleb_puffs_to_mom_l2230_223099


namespace NUMINAMATH_GPT_tangent_line_equation_l2230_223049

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

theorem tangent_line_equation :
  let x1 : ℝ := 1
  let y1 : ℝ := f 1
  ∀ x y : ℝ, 
    (y - y1 = (1 / (x1 + 1)) * (x - x1)) ↔ 
    (x - 2 * y + 2 * Real.log 2 - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l2230_223049


namespace NUMINAMATH_GPT_find_k_values_l2230_223026

theorem find_k_values (a b k : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b % a = 0) 
  (h₄ : ∀ (m : ℤ), (a : ℤ) = k * (a : ℤ) + m ∧ (8 * (b : ℤ)) = k * (b : ℤ) + m) :
  k = 9 ∨ k = 15 :=
by
  { sorry }

end NUMINAMATH_GPT_find_k_values_l2230_223026


namespace NUMINAMATH_GPT_composite_of_squares_l2230_223082

theorem composite_of_squares (n : ℕ) (h1 : 8 * n + 1 = x^2) (h2 : 24 * n + 1 = y^2) (h3 : n > 1) : ∃ a b : ℕ, a ∣ (8 * n + 3) ∧ b ∣ (8 * n + 3) ∧ a ≠ 1 ∧ b ≠ 1 ∧ a ≠ (8 * n + 3) ∧ b ≠ (8 * n + 3) := by
  sorry

end NUMINAMATH_GPT_composite_of_squares_l2230_223082


namespace NUMINAMATH_GPT_slip_4_goes_in_B_l2230_223001

-- Definitions for the slips, cups, and conditions
def slips : List ℝ := [1, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5]
def cupSum (c : Char) : ℝ := 
  match c with
  | 'A' => 6
  | 'B' => 7
  | 'C' => 8
  | 'D' => 9
  | 'E' => 10
  | 'F' => 11
  | _   => 0

def cupAssignments : Char → List ℝ
  | 'F' => [2]
  | 'B' => [3]
  | _   => []

theorem slip_4_goes_in_B :
  (∃ cupA cupB cupC cupD cupE cupF : List ℝ, 
    cupA.sum = cupSum 'A' ∧
    cupB.sum = cupSum 'B' ∧
    cupC.sum = cupSum 'C' ∧
    cupD.sum = cupSum 'D' ∧
    cupE.sum = cupSum 'E' ∧
    cupF.sum = cupSum 'F' ∧
    slips = cupA ++ cupB ++ cupC ++ cupD ++ cupE ++ cupF ∧
    cupF.contains 2 ∧
    cupB.contains 3 ∧
    cupB.contains 4) :=
sorry

end NUMINAMATH_GPT_slip_4_goes_in_B_l2230_223001


namespace NUMINAMATH_GPT_initial_range_without_telescope_l2230_223039

variable (V : ℝ)

def telescope_increases_range (V : ℝ) : Prop :=
  V + 0.875 * V = 150

theorem initial_range_without_telescope (V : ℝ) (h : telescope_increases_range V) : V = 80 :=
by
  sorry

end NUMINAMATH_GPT_initial_range_without_telescope_l2230_223039


namespace NUMINAMATH_GPT_smallest_multiple_of_18_and_40_l2230_223064

-- Define the conditions
def multiple_of_18 (n : ℕ) : Prop := n % 18 = 0
def multiple_of_40 (n : ℕ) : Prop := n % 40 = 0

-- Prove that the smallest number that meets the conditions is 360
theorem smallest_multiple_of_18_and_40 : ∃ n : ℕ, multiple_of_18 n ∧ multiple_of_40 n ∧ ∀ m : ℕ, (multiple_of_18 m ∧ multiple_of_40 m) → n ≤ m :=
  by
    let n := 360
    -- We have to prove that 360 is the smallest number that is a multiple of both 18 and 40
    sorry

end NUMINAMATH_GPT_smallest_multiple_of_18_and_40_l2230_223064


namespace NUMINAMATH_GPT_satisfies_equation_l2230_223028

noncomputable def y (x : ℝ) : ℝ := -Real.sqrt (x^4 - x^2)
noncomputable def dy (x : ℝ) : ℝ := x * (1 - 2 * x^2) / Real.sqrt (x^4 - x^2)

theorem satisfies_equation (x : ℝ) (hx : x ≠ 0) : x * y x * dy x - (y x)^2 = x^4 := 
sorry

end NUMINAMATH_GPT_satisfies_equation_l2230_223028


namespace NUMINAMATH_GPT_value_of_expression_l2230_223073

theorem value_of_expression (m : ℝ) (α : ℝ) (h : m < 0) (h_M : M = (3 * m, -m)) :
  let sin_alpha := -m / (Real.sqrt 10 * -m)
  let cos_alpha := 3 * m / (Real.sqrt 10 * -m)
  (1 / (2 * sin_alpha * cos_alpha + cos_alpha^2) = 10 / 3) :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2230_223073


namespace NUMINAMATH_GPT_students_brought_two_plants_l2230_223093

theorem students_brought_two_plants 
  (a1 a2 a3 a4 a5 : ℕ) (p1 p2 p3 p4 p5 : ℕ)
  (h1 : a1 + a2 + a3 + a4 + a5 = 20)
  (h2 : a1 * p1 + a2 * p2 + a3 * p3 + a4 * p4 + a5 * p5 = 30)
  (h3 : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
        p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5)
  : ∃ a : ℕ, a = 1 ∧ (∃ i : ℕ, p1 = 2 ∨ p2 = 2 ∨ p3 = 2 ∨ p4 = 2 ∨ p5 = 2) :=
sorry

end NUMINAMATH_GPT_students_brought_two_plants_l2230_223093


namespace NUMINAMATH_GPT_milo_running_distance_l2230_223032

theorem milo_running_distance
  (run_speed skateboard_speed cory_speed : ℕ)
  (h1 : skateboard_speed = 2 * run_speed)
  (h2 : cory_speed = 2 * skateboard_speed)
  (h3 : cory_speed = 12) :
  run_speed * 2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_milo_running_distance_l2230_223032


namespace NUMINAMATH_GPT_find_n_l2230_223057

noncomputable def first_term_1 : ℝ := 12
noncomputable def second_term_1 : ℝ := 4
noncomputable def sum_first_series : ℝ := 18

noncomputable def first_term_2 : ℝ := 12
noncomputable def second_term_2 (n : ℝ) : ℝ := 4 + 2 * n
noncomputable def sum_second_series : ℝ := 90

theorem find_n (n : ℝ) : 
  (first_term_1 = 12) → 
  (second_term_1 = 4) → 
  (sum_first_series = 18) →
  (first_term_2 = 12) →
  (second_term_2 n = 4 + 2 * n) →
  (sum_second_series = 90) →
  (sum_second_series = 5 * sum_first_series) →
  n = 6 :=
by
  intros _ _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_find_n_l2230_223057


namespace NUMINAMATH_GPT_ways_to_write_2020_as_sum_of_twos_and_threes_l2230_223042

def write_as_sum_of_twos_and_threes (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n / 2 + 1) else 0

theorem ways_to_write_2020_as_sum_of_twos_and_threes :
  write_as_sum_of_twos_and_threes 2020 = 337 :=
sorry

end NUMINAMATH_GPT_ways_to_write_2020_as_sum_of_twos_and_threes_l2230_223042


namespace NUMINAMATH_GPT_inequality_sum_l2230_223021

variable {a b c d : ℝ}

theorem inequality_sum (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_sum_l2230_223021


namespace NUMINAMATH_GPT_find_sum_of_roots_l2230_223009

open Real

theorem find_sum_of_roots (p q r s : ℝ): 
  r + s = 12 * p →
  r * s = 13 * q →
  p + q = 12 * r →
  p * q = 13 * s →
  p ≠ r →
  p + q + r + s = 2028 := by
  intros
  sorry

end NUMINAMATH_GPT_find_sum_of_roots_l2230_223009


namespace NUMINAMATH_GPT_park_area_l2230_223047

-- Define the width (w) and length (l) of the park
def width : Float := 11.25
def length : Float := 33.75

-- Define the perimeter and area functions
def perimeter (w l : Float) : Float := 2 * (w + l)
def area (w l : Float) : Float := w * l

-- Provide the conditions
axiom width_is_one_third_length : width = length / 3
axiom perimeter_is_90 : perimeter width length = 90

-- Theorem to prove the area given the conditions
theorem park_area : area width length = 379.6875 := by
  sorry

end NUMINAMATH_GPT_park_area_l2230_223047


namespace NUMINAMATH_GPT_positive_integer_sum_l2230_223024

theorem positive_integer_sum (n : ℤ) :
  (n > 0) ∧
  (∀ stamps_cannot_form : ℤ, (∀ a b c : ℤ, 7 * a + n * b + (n + 2) * c ≠ 120) ↔
  ¬ (0 ≤ 7*a ∧ 7*a ≤ 120 ∧ 0 ≤ n*b ∧ n*b ≤ 120 ∧ 0 ≤ (n + 2)*c ∧ (n + 2)*c ≤ 120)) ∧
  (∀ postage_formed : ℤ, (120 < postage_formed ∧ postage_formed ≤ 125 →
  ∃ a b c : ℤ, 7 * a + n * b + (n + 2) * c = postage_formed)) →
  n = 21 :=
by {
  -- proof omittted 
  sorry
}

end NUMINAMATH_GPT_positive_integer_sum_l2230_223024


namespace NUMINAMATH_GPT_contradiction_proof_l2230_223022

theorem contradiction_proof (a b c : ℝ) (h : (a⁻¹ * b⁻¹ * c⁻¹) > 0) : (a ≤ 1) ∧ (b ≤ 1) ∧ (c ≤ 1) → False :=
sorry

end NUMINAMATH_GPT_contradiction_proof_l2230_223022


namespace NUMINAMATH_GPT_solution_proof_l2230_223027

noncomputable def problem_statement : Prop :=
  let a : ℝ := 0.10
  let b : ℝ := 0.50
  let c : ℝ := 500
  a * (b * c) = 25

theorem solution_proof : problem_statement := by
  sorry

end NUMINAMATH_GPT_solution_proof_l2230_223027


namespace NUMINAMATH_GPT_solve_fraction_l2230_223083

theorem solve_fraction :
  (144^2 - 100^2) / 22 = 488 := 
by 
  sorry

end NUMINAMATH_GPT_solve_fraction_l2230_223083


namespace NUMINAMATH_GPT_parabola_directrix_l2230_223011

theorem parabola_directrix (y x : ℝ) (h : y = x^2) : 4 * y + 1 = 0 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l2230_223011


namespace NUMINAMATH_GPT_quadratic_result_l2230_223067

noncomputable def quadratic_has_two_positive_integer_roots (k p : ℕ) : Prop :=
  ∃ x1 x2 : ℕ, x1 > 0 ∧ x2 > 0 ∧ (k - 1) * x1 * x1 - p * x1 + k = 0 ∧ (k - 1) * x2 * x2 - p * x2 + k = 0

theorem quadratic_result (k p : ℕ) (h1 : k = 2) (h2 : quadratic_has_two_positive_integer_roots k p) :
  k^(k*p) * (p^p + k^k) = 1984 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_result_l2230_223067


namespace NUMINAMATH_GPT_two_n_plus_m_is_36_l2230_223071

theorem two_n_plus_m_is_36 (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 
  2 * n + m = 36 :=
sorry

end NUMINAMATH_GPT_two_n_plus_m_is_36_l2230_223071


namespace NUMINAMATH_GPT_calories_in_250_grams_of_lemonade_l2230_223090

theorem calories_in_250_grams_of_lemonade:
  ∀ (lemon_juice_grams sugar_grams water_grams total_grams: ℕ)
    (lemon_juice_cal_per_100 sugar_cal_per_100 total_cal: ℕ),
  lemon_juice_grams = 150 →
  sugar_grams = 150 →
  water_grams = 300 →
  total_grams = lemon_juice_grams + sugar_grams + water_grams →
  lemon_juice_cal_per_100 = 30 →
  sugar_cal_per_100 = 386 →
  total_cal = (lemon_juice_grams * lemon_juice_cal_per_100 / 100) + (sugar_grams * sugar_cal_per_100 / 100) →
  (250:ℕ) * total_cal / total_grams = 260 :=
by
  intros lemon_juice_grams sugar_grams water_grams total_grams lemon_juice_cal_per_100 sugar_cal_per_100 total_cal
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_calories_in_250_grams_of_lemonade_l2230_223090


namespace NUMINAMATH_GPT_farm_horses_cows_difference_l2230_223038

-- Definitions based on provided conditions
def initial_ratio_horses_to_cows (horses cows : ℕ) : Prop := 5 * cows = horses
def transaction (horses cows sold bought : ℕ) : Prop :=
  horses - sold = 5 * cows - 15 ∧ cows + bought = cows + 15

-- Definitions to represent the ratios
def pre_transaction_ratio (horses cows : ℕ) : Prop := initial_ratio_horses_to_cows horses cows
def post_transaction_ratio (horses cows : ℕ) (sold bought : ℕ) : Prop :=
  transaction horses cows sold bought ∧ 7 * (horses - sold) = 17 * (cows + bought)

-- Statement of the theorem
theorem farm_horses_cows_difference :
  ∀ (horses cows : ℕ), 
    pre_transaction_ratio horses cows → 
    post_transaction_ratio horses cows 15 15 →
    (horses - 15) - (cows + 15) = 50 :=
by
  intros horses cows pre_ratio post_ratio
  sorry

end NUMINAMATH_GPT_farm_horses_cows_difference_l2230_223038


namespace NUMINAMATH_GPT_area_of_field_l2230_223030

-- Define the variables and conditions
variables {L W : ℝ}

-- Given conditions
def length_side (L : ℝ) : Prop := L = 30
def fencing_equation (L W : ℝ) : Prop := L + 2 * W = 70

-- Prove the area of the field is 600 square feet
theorem area_of_field : length_side L → fencing_equation L W → (L * W = 600) :=
by
  intros hL hF
  rw [length_side, fencing_equation] at *
  sorry

end NUMINAMATH_GPT_area_of_field_l2230_223030


namespace NUMINAMATH_GPT_cos_pi_over_8_cos_5pi_over_8_l2230_223044

theorem cos_pi_over_8_cos_5pi_over_8 :
  (Real.cos (Real.pi / 8)) * (Real.cos (5 * Real.pi / 8)) = - (Real.sqrt 2 / 4) :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_over_8_cos_5pi_over_8_l2230_223044


namespace NUMINAMATH_GPT_quadratic_min_value_l2230_223075

theorem quadratic_min_value (k : ℝ) :
  (∀ x : ℝ, 3 ≤ x ∧ x ≤ 5 → y = (1/2) * (x - 1) ^ 2 + k) ∧
  (∀ y : ℝ, 3 ≤ y ∧ y ≤ 5 → y ≥ 3) → k = 1 :=
sorry

end NUMINAMATH_GPT_quadratic_min_value_l2230_223075


namespace NUMINAMATH_GPT_minimize_sum_of_distances_l2230_223088

theorem minimize_sum_of_distances (P : ℝ × ℝ) (A : ℝ × ℝ) (F : ℝ × ℝ) 
  (hP_on_parabola : P.2 ^ 2 = 2 * P.1)
  (hA : A = (3, 2)) 
  (hF : F = (1/2, 0)) : 
  |P - A| + |P - F| ≥ |(2, 2) - A| + |(2, 2) - F| :=
by sorry

end NUMINAMATH_GPT_minimize_sum_of_distances_l2230_223088


namespace NUMINAMATH_GPT_infinite_divisibility_1986_l2230_223008

theorem infinite_divisibility_1986 :
  ∃ (a : ℕ → ℕ), a 1 = 39 ∧ a 2 = 45 ∧ (∀ n, a (n+2) = a (n+1) ^ 2 - a n) ∧
  ∀ N, ∃ n > N, 1986 ∣ a n :=
sorry

end NUMINAMATH_GPT_infinite_divisibility_1986_l2230_223008


namespace NUMINAMATH_GPT_sarah_boxes_l2230_223023

theorem sarah_boxes (b : ℕ) 
  (h1 : ∀ x : ℕ, x = 7) 
  (h2 : 49 = 7 * b) :
  b = 7 :=
sorry

end NUMINAMATH_GPT_sarah_boxes_l2230_223023


namespace NUMINAMATH_GPT_movie_theater_ticket_cost_l2230_223035

theorem movie_theater_ticket_cost
  (adult_ticket_cost : ℝ)
  (child_ticket_cost : ℝ)
  (total_moviegoers : ℝ)
  (total_amount_paid : ℝ)
  (number_of_adults : ℝ)
  (H_child_ticket_cost : child_ticket_cost = 6.50)
  (H_total_moviegoers : total_moviegoers = 7)
  (H_total_amount_paid : total_amount_paid = 54.50)
  (H_number_of_adults : number_of_adults = 3)
  (H_number_of_children : total_moviegoers - number_of_adults = 4) :
  adult_ticket_cost = 9.50 :=
sorry

end NUMINAMATH_GPT_movie_theater_ticket_cost_l2230_223035


namespace NUMINAMATH_GPT_pandemic_cut_percentage_l2230_223036

-- Define the conditions
def initial_planned_production : ℕ := 200
def decrease_due_to_metal_shortage : ℕ := 50
def doors_per_car : ℕ := 5
def total_doors_produced : ℕ := 375

-- Define the quantities after metal shortage and before the pandemic
def production_after_metal_shortage : ℕ := initial_planned_production - decrease_due_to_metal_shortage
def doors_after_metal_shortage : ℕ := production_after_metal_shortage * doors_per_car
def cars_after_pandemic : ℕ := total_doors_produced / doors_per_car
def reduction_in_production : ℕ := production_after_metal_shortage - cars_after_pandemic

-- Define the expected percentage cut
def expected_percentage_cut : ℕ := 50

-- Prove that the percentage of production cut due to the pandemic is as required
theorem pandemic_cut_percentage : (reduction_in_production * 100 / production_after_metal_shortage) = expected_percentage_cut := by
  sorry

end NUMINAMATH_GPT_pandemic_cut_percentage_l2230_223036


namespace NUMINAMATH_GPT_smallest_integer_k_l2230_223002

theorem smallest_integer_k :
  ∃ k : ℕ, 
    k > 1 ∧ 
    k % 19 = 1 ∧ 
    k % 14 = 1 ∧ 
    k % 9 = 1 ∧ 
    k = 2395 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_integer_k_l2230_223002


namespace NUMINAMATH_GPT_find_a_for_even_function_l2230_223085

theorem find_a_for_even_function (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = (x + 1)*(x - a) ∧ f (-x) = f x) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_for_even_function_l2230_223085


namespace NUMINAMATH_GPT_math_preference_related_to_gender_l2230_223070

-- Definitions for conditions
def total_students : ℕ := 100
def male_students : ℕ := 55
def female_students : ℕ := total_students - male_students -- 45
def likes_math : ℕ := 40
def female_likes_math : ℕ := 20
def female_not_like_math : ℕ := female_students - female_likes_math -- 25
def male_likes_math : ℕ := likes_math - female_likes_math -- 20
def male_not_like_math : ℕ := male_students - male_likes_math -- 35

-- Calculate Chi-square
def chi_square (a b c d : ℕ) : Float :=
  let numerator := (total_students * (a * d - b * c)^2).toFloat
  let denominator := ((a + b) * (c + d) * (a + c) * (b + d)).toFloat
  numerator / denominator

def k_square : Float := chi_square 20 35 20 25 -- Calculate with given values

-- Prove the result
theorem math_preference_related_to_gender :
  k_square > 7.879 :=
by
  sorry

end NUMINAMATH_GPT_math_preference_related_to_gender_l2230_223070


namespace NUMINAMATH_GPT_functional_equation_solution_l2230_223097

def odd_integers (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem functional_equation_solution (f : ℤ → ℤ)
  (h_odd : ∀ x : ℤ, odd_integers (f x))
  (h_eq : ∀ x y : ℤ, 
    f (x + f x + y) + f (x - f x - y) = f (x + y) + f (x - y)) :
  ∃ (d k : ℤ) (ell : ℕ → ℤ), 
    (∀ i : ℕ, i < d → odd_integers (ell i)) ∧
    ∀ (m : ℤ) (i : ℕ), i < d → 
      f (m * d + i) = 2 * k * m * d + ell i :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l2230_223097


namespace NUMINAMATH_GPT_magnets_per_earring_l2230_223006

theorem magnets_per_earring (M : ℕ) (h : 4 * (3 * M / 2) = 24) : M = 4 :=
by
  sorry

end NUMINAMATH_GPT_magnets_per_earring_l2230_223006


namespace NUMINAMATH_GPT_part_a_part_b_l2230_223052

theorem part_a (a b c d : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  ∃ (a b c d : ℕ), 1 / (a : ℝ) + 1 / (b : ℝ) = 1 / (c : ℝ) + 1 / (d : ℝ) := sorry

theorem part_b (a b c d e : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) : 
  ∃ (a b c d e : ℕ), 1 / (a : ℝ) + 1 / (b : ℝ) = 1 / (c : ℝ) + 1 / (d : ℝ) + 1 / (e : ℝ) := sorry

end NUMINAMATH_GPT_part_a_part_b_l2230_223052


namespace NUMINAMATH_GPT_find_tangent_value_l2230_223080

noncomputable def tangent_value (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧
  (1 / (x₀ + a) = 1)

theorem find_tangent_value : tangent_value 2 :=
  sorry

end NUMINAMATH_GPT_find_tangent_value_l2230_223080


namespace NUMINAMATH_GPT_december_revenue_times_average_l2230_223068

variable (D : ℝ) -- December's revenue
variable (N : ℝ) -- November's revenue
variable (J : ℝ) -- January's revenue

-- Conditions
def revenue_in_november : N = (2/5) * D := by sorry
def revenue_in_january : J = (1/2) * N := by sorry

-- Statement to be proved
theorem december_revenue_times_average :
  D = (10/3) * ((N + J) / 2) :=
by sorry

end NUMINAMATH_GPT_december_revenue_times_average_l2230_223068


namespace NUMINAMATH_GPT_a_in_M_l2230_223089

def M : Set ℝ := { x | x ≤ 5 }
def a : ℝ := 2

theorem a_in_M : a ∈ M :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_a_in_M_l2230_223089


namespace NUMINAMATH_GPT_no_integer_m_l2230_223063

theorem no_integer_m (n r m : ℕ) (hn : 1 ≤ n) (hr : 2 ≤ r) : 
  ¬ (∃ m : ℕ, n * (n + 1) * (n + 2) = m ^ r) :=
sorry

end NUMINAMATH_GPT_no_integer_m_l2230_223063


namespace NUMINAMATH_GPT_james_total_payment_l2230_223091

noncomputable def total_amount_paid : ℕ :=
  let dirt_bike_count := 3
  let off_road_vehicle_count := 4
  let atv_count := 2
  let moped_count := 5
  let scooter_count := 3
  let dirt_bike_cost := dirt_bike_count * 150
  let off_road_vehicle_cost := off_road_vehicle_count * 300
  let atv_cost := atv_count * 450
  let moped_cost := moped_count * 200
  let scooter_cost := scooter_count * 100
  let registration_dirt_bike := dirt_bike_count * 25
  let registration_off_road_vehicle := off_road_vehicle_count * 25
  let registration_atv := atv_count * 30
  let registration_moped := moped_count * 15
  let registration_scooter := scooter_count * 20
  let maintenance_dirt_bike := dirt_bike_count * 50
  let maintenance_off_road_vehicle := off_road_vehicle_count * 75
  let maintenance_atv := atv_count * 100
  let maintenance_moped := moped_count * 60
  let total_cost_of_vehicles := dirt_bike_cost + off_road_vehicle_cost + atv_cost + moped_cost + scooter_cost
  let total_registration_costs := registration_dirt_bike + registration_off_road_vehicle + registration_atv + registration_moped + registration_scooter
  let total_maintenance_costs := maintenance_dirt_bike + maintenance_off_road_vehicle + maintenance_atv + maintenance_moped
  total_cost_of_vehicles + total_registration_costs + total_maintenance_costs

theorem james_total_payment : total_amount_paid = 5170 := by
  -- The proof would be written here
  sorry

end NUMINAMATH_GPT_james_total_payment_l2230_223091


namespace NUMINAMATH_GPT_simplify_polynomials_l2230_223078

theorem simplify_polynomials (x : ℝ) :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 3 * x - 15) = x^2 + 5 * x + 10 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_polynomials_l2230_223078


namespace NUMINAMATH_GPT_perfect_squares_with_property_l2230_223003

open Nat

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k : ℕ, p.Prime ∧ k > 0 ∧ n = p^k

def satisfies_property (n : ℕ) : Prop :=
  ∀ a : ℕ, a ∣ n → a ≥ 15 → is_prime_power (a + 15)

theorem perfect_squares_with_property :
  {n | satisfies_property n ∧ ∃ k : ℕ, n = k^2} = {1, 4, 9, 16, 49, 64, 196} :=
by
  sorry

end NUMINAMATH_GPT_perfect_squares_with_property_l2230_223003


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2230_223025

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 ≥ 0 } = { x : ℝ | x ≤ -2 ∨ 1 ≤ x } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2230_223025


namespace NUMINAMATH_GPT_circle_equation_l2230_223086

theorem circle_equation (a b r : ℝ) 
    (h₁ : b = -4 * a)
    (h₂ : abs (a + b - 1) / Real.sqrt 2 = r)
    (h₃ : (b + 2) / (a - 3) * (-1) = -1)
    (h₄ : a = 1)
    (h₅ : b = -4)
    (h₆ : r = 2 * Real.sqrt 2) :
    ∀ x y: ℝ, (x - 1) ^ 2 + (y + 4) ^ 2 = 8 := 
by
  intros
  sorry

end NUMINAMATH_GPT_circle_equation_l2230_223086


namespace NUMINAMATH_GPT_total_distance_of_trip_l2230_223020

theorem total_distance_of_trip (x : ℚ)
  (highway : x / 4 ≤ x)
  (city : 30 ≤ x)
  (country : x / 6 ≤ x)
  (middle_part_fraction : 1 - 1 / 4 - 1 / 6 = 7 / 12) :
  (7 / 12) * x = 30 → x = 360 / 7 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_of_trip_l2230_223020


namespace NUMINAMATH_GPT_equation_of_curve_E_equation_of_line_l_through_origin_intersecting_E_l2230_223066

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 3, 0)

def is_ellipse (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F₁.1) ^ 2 + P.2 ^ 2) + Real.sqrt ((P.1 - F₂.1) ^ 2 + P.2 ^ 2) = 4

theorem equation_of_curve_E :
  ∀ P : ℝ × ℝ, is_ellipse P ↔ (P.1 ^ 2 / 4 + P.2 ^ 2 = 1) :=
sorry

def intersects_at_origin (C D : ℝ × ℝ) : Prop :=
  C.1 * D.1 + C.2 * D.2 = 0

theorem equation_of_line_l_through_origin_intersecting_E :
  ∀ (l : ℝ → ℝ) (C D : ℝ × ℝ),
    (l 0 = -2) →
    (∀ P : ℝ × ℝ, is_ellipse P ↔ (P.1, P.2) = (C.1, l C.1) ∨ (P.1, P.2) = (D.1, l D.1)) →
    intersects_at_origin C D →
    (∀ x, l x = 2 * x - 2) ∨ (∀ x, l x = -2 * x - 2) :=
sorry

end NUMINAMATH_GPT_equation_of_curve_E_equation_of_line_l_through_origin_intersecting_E_l2230_223066


namespace NUMINAMATH_GPT_remainder_of_x_plus_2_power_2008_l2230_223018

-- Given: x^3 ≡ 1 (mod x^2 + x + 1)
def given_condition : Prop := ∀ x : ℤ, (x^3 - 1) % (x^2 + x + 1) = 0

-- To prove: The remainder when (x + 2)^2008 is divided by x^2 + x + 1 is 1
theorem remainder_of_x_plus_2_power_2008 (x : ℤ) (h : given_condition) :
  ((x + 2) ^ 2008) % (x^2 + x + 1) = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_of_x_plus_2_power_2008_l2230_223018


namespace NUMINAMATH_GPT_tank_filling_time_l2230_223031

-- Define the rates at which pipes fill or drain the tank
def capacity : ℕ := 1200
def rate_A : ℕ := 50
def rate_B : ℕ := 35
def rate_C : ℕ := 20
def rate_D : ℕ := 40

-- Define the times each pipe is open
def time_A : ℕ := 2
def time_B : ℕ := 4
def time_C : ℕ := 3
def time_D : ℕ := 5

-- Calculate the total time for one cycle
def cycle_time : ℕ := time_A + time_B + time_C + time_D

-- Calculate the net amount of water added in one cycle
def net_amount_per_cycle : ℕ := (rate_A * time_A) + (rate_B * time_B) + (rate_C * time_C) - (rate_D * time_D)

-- Calculate the number of cycles needed to fill the tank
def num_cycles : ℕ := capacity / net_amount_per_cycle

-- Calculate the total time to fill the tank
def total_time : ℕ := num_cycles * cycle_time

-- Prove that the total time to fill the tank is 168 minutes
theorem tank_filling_time : total_time = 168 := by
  sorry

end NUMINAMATH_GPT_tank_filling_time_l2230_223031


namespace NUMINAMATH_GPT_net_income_difference_l2230_223058

theorem net_income_difference
    (terry_daily_income : ℝ := 24) (terry_daily_hours : ℝ := 6) (terry_days : ℕ := 7)
    (jordan_daily_income : ℝ := 30) (jordan_daily_hours : ℝ := 8) (jordan_days : ℕ := 6)
    (standard_week_hours : ℝ := 40) (overtime_rate_multiplier : ℝ := 1.5)
    (terry_tax_rate : ℝ := 0.12) (jordan_tax_rate : ℝ := 0.15) :
    jordan_daily_income * jordan_days - jordan_daily_income * jordan_days * jordan_tax_rate 
      + jordan_daily_income * jordan_days * jordan_daily_hours * (overtime_rate_multiplier - 1) * jordan_tax_rate
    - (terry_daily_income * terry_days - terry_daily_income * terry_days * terry_tax_rate 
      + terry_daily_income * terry_days * terry_daily_hours * (overtime_rate_multiplier - 1) * terry_tax_rate) 
      = 32.85 := 
sorry

end NUMINAMATH_GPT_net_income_difference_l2230_223058


namespace NUMINAMATH_GPT_octahedron_vertices_sum_l2230_223033

noncomputable def octahedron_faces_sum (a b c d e f : ℕ) : ℕ :=
  a + b + c + d + e + f

theorem octahedron_vertices_sum (a b c d e f : ℕ) 
  (h : 8 * (octahedron_faces_sum a b c d e f) = 440) : 
  octahedron_faces_sum a b c d e f = 147 :=
by
  sorry

end NUMINAMATH_GPT_octahedron_vertices_sum_l2230_223033


namespace NUMINAMATH_GPT_license_plate_combinations_l2230_223014

def consonants_count := 21
def vowels_count := 5
def digits_count := 10

theorem license_plate_combinations : 
  consonants_count * vowels_count * consonants_count * digits_count * vowels_count = 110250 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_combinations_l2230_223014


namespace NUMINAMATH_GPT_minimum_box_value_l2230_223037

def is_valid_pair (a b : ℤ) : Prop :=
  a * b = 15 ∧ (a^2 + b^2 ≥ 34)

theorem minimum_box_value :
  ∃ (a b : ℤ), is_valid_pair a b ∧ (∀ (a' b' : ℤ), is_valid_pair a' b' → a^2 + b^2 ≤ a'^2 + b'^2) ∧ a^2 + b^2 = 34 :=
by
  sorry

end NUMINAMATH_GPT_minimum_box_value_l2230_223037


namespace NUMINAMATH_GPT_correct_formula_l2230_223045

def table : List (ℕ × ℕ) :=
    [(1, 3), (2, 8), (3, 15), (4, 24), (5, 35)]

theorem correct_formula : ∀ x y, (x, y) ∈ table → y = x^2 + 4 * x + 3 :=
by
  intros x y H
  sorry

end NUMINAMATH_GPT_correct_formula_l2230_223045


namespace NUMINAMATH_GPT_initial_ratio_l2230_223017

variables {p q : ℝ}

theorem initial_ratio (h₁ : p + q = 20) (h₂ : p / (q + 1) = 4 / 3) : p / q = 3 / 2 :=
sorry

end NUMINAMATH_GPT_initial_ratio_l2230_223017


namespace NUMINAMATH_GPT_tomatoes_picked_l2230_223096

theorem tomatoes_picked (initial_tomatoes picked_tomatoes : ℕ)
  (h₀ : initial_tomatoes = 17)
  (h₁ : initial_tomatoes - picked_tomatoes = 8) :
  picked_tomatoes = 9 :=
by
  sorry

end NUMINAMATH_GPT_tomatoes_picked_l2230_223096


namespace NUMINAMATH_GPT_common_divisors_9240_13860_l2230_223060

def num_divisors (n : ℕ) : ℕ :=
  -- function to calculate the number of divisors (implementation is not provided here)
  sorry

theorem common_divisors_9240_13860 :
  let d := Nat.gcd 9240 13860
  d = 924 → num_divisors d = 24 := by
  intros d gcd_eq
  rw [gcd_eq]
  sorry

end NUMINAMATH_GPT_common_divisors_9240_13860_l2230_223060


namespace NUMINAMATH_GPT_trekking_adults_l2230_223019

theorem trekking_adults
  (A : ℕ)
  (C : ℕ)
  (meal_for_adults : ℕ)
  (meal_for_children : ℕ)
  (remaining_food_children : ℕ) :
  C = 70 →
  meal_for_adults = 70 →
  meal_for_children = 90 →
  remaining_food_children = 72 →
  A - 14 = (meal_for_adults - 14) →
  A = 56 :=
sorry

end NUMINAMATH_GPT_trekking_adults_l2230_223019


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2230_223095

-- Definitions of the quadrants as provided in the conditions
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Given point
def point : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem point_in_fourth_quadrant : fourth_quadrant point.fst point.snd :=
sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2230_223095


namespace NUMINAMATH_GPT_count_valid_arrangements_l2230_223051

-- Definitions based on conditions
def total_chairs : Nat := 48

def valid_factor_pairs (n : Nat) : List (Nat × Nat) :=
  [ (2, 24), (3, 16), (4, 12), (6, 8), (8, 6), (12, 4), (16, 3), (24, 2) ]

def count_valid_arrays : Nat := valid_factor_pairs total_chairs |>.length

-- The theorem we want to prove
theorem count_valid_arrangements : count_valid_arrays = 8 := 
  by
    -- proof should be provided here
    sorry

end NUMINAMATH_GPT_count_valid_arrangements_l2230_223051


namespace NUMINAMATH_GPT_students_no_A_l2230_223053

theorem students_no_A
  (total_students : ℕ)
  (A_in_English : ℕ)
  (A_in_math : ℕ)
  (A_in_both : ℕ)
  (total_students_eq : total_students = 40)
  (A_in_English_eq : A_in_English = 10)
  (A_in_math_eq : A_in_math = 18)
  (A_in_both_eq : A_in_both = 6) :
  total_students - ((A_in_English + A_in_math) - A_in_both) = 18 :=
by
  sorry

end NUMINAMATH_GPT_students_no_A_l2230_223053


namespace NUMINAMATH_GPT_midpoint_between_points_l2230_223034

theorem midpoint_between_points : 
  let (x1, y1, z1) := (2, -3, 5)
  let (x2, y2, z2) := (8, 1, 3)
  (1 / 2 * (x1 + x2), 1 / 2 * (y1 + y2), 1 / 2 * (z1 + z2)) = (5, -1, 4) :=
by
  sorry

end NUMINAMATH_GPT_midpoint_between_points_l2230_223034


namespace NUMINAMATH_GPT_curve_is_line_l2230_223048

noncomputable def curve_representation (x y : ℝ) : Prop :=
  (2 * x + 3 * y - 1) * (-1) = 0

theorem curve_is_line (x y : ℝ) (h : curve_representation x y) : 2 * x + 3 * y - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_curve_is_line_l2230_223048


namespace NUMINAMATH_GPT_even_function_a_eq_one_l2230_223098

noncomputable def f (x a : ℝ) : ℝ := x * Real.log (x + Real.sqrt (a + x ^ 2))

theorem even_function_a_eq_one (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_even_function_a_eq_one_l2230_223098


namespace NUMINAMATH_GPT_trig_identity_evaluation_l2230_223029

theorem trig_identity_evaluation :
  4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_evaluation_l2230_223029


namespace NUMINAMATH_GPT_parabola_tangent_to_hyperbola_l2230_223010

theorem parabola_tangent_to_hyperbola (m : ℝ) :
  (∀ x y : ℝ, y = x^2 + 4 → y^2 - m * x^2 = 4) ↔ m = 8 := 
sorry

end NUMINAMATH_GPT_parabola_tangent_to_hyperbola_l2230_223010


namespace NUMINAMATH_GPT_find_A_l2230_223007

theorem find_A (A : ℕ) (h1 : A < 5) (h2 : (9 * 100 + A * 10 + 7) / 10 * 10 = 930) : A = 3 :=
sorry

end NUMINAMATH_GPT_find_A_l2230_223007


namespace NUMINAMATH_GPT_combined_flock_size_l2230_223094

def original_ducks := 100
def killed_per_year := 20
def born_per_year := 30
def years_passed := 5
def another_flock := 150

theorem combined_flock_size :
  original_ducks + years_passed * (born_per_year - killed_per_year) + another_flock = 300 :=
by
  sorry

end NUMINAMATH_GPT_combined_flock_size_l2230_223094


namespace NUMINAMATH_GPT_uber_profit_l2230_223000

-- Define conditions
def income : ℕ := 30000
def initial_cost : ℕ := 18000
def trade_in : ℕ := 6000

-- Define depreciation cost
def depreciation_cost : ℕ := initial_cost - trade_in

-- Define the profit
def profit : ℕ := income - depreciation_cost

-- The theorem to be proved
theorem uber_profit : profit = 18000 := by 
  sorry

end NUMINAMATH_GPT_uber_profit_l2230_223000


namespace NUMINAMATH_GPT_sum_of_squares_of_b_l2230_223061

-- Define the constants
def b1 := 35 / 64
def b2 := 0
def b3 := 21 / 64
def b4 := 0
def b5 := 7 / 64
def b6 := 0
def b7 := 1 / 64

-- The goal is to prove the sum of squares of these constants
theorem sum_of_squares_of_b : 
  (b1 ^ 2 + b2 ^ 2 + b3 ^ 2 + b4 ^ 2 + b5 ^ 2 + b6 ^ 2 + b7 ^ 2) = 429 / 1024 :=
  by
    -- defer the proof
    sorry

end NUMINAMATH_GPT_sum_of_squares_of_b_l2230_223061


namespace NUMINAMATH_GPT_area_of_triangle_with_medians_l2230_223079

theorem area_of_triangle_with_medians
  (s_a s_b s_c : ℝ) :
  (∃ t : ℝ, t = (1 / 3 : ℝ) * ((s_a + s_b + s_c) * (s_b + s_c - s_a) * (s_a + s_c - s_b) * (s_a + s_b - s_c)).sqrt) :=
sorry

end NUMINAMATH_GPT_area_of_triangle_with_medians_l2230_223079


namespace NUMINAMATH_GPT_minimum_omega_l2230_223054

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end NUMINAMATH_GPT_minimum_omega_l2230_223054


namespace NUMINAMATH_GPT_toothpick_removal_l2230_223005

/-- Given 40 toothpicks used to create 10 squares and 15 triangles, with each square formed by 
4 toothpicks and each triangle formed by 3 toothpicks, prove that removing 10 toothpicks is 
sufficient to ensure no squares or triangles remain. -/
theorem toothpick_removal (n : ℕ) (squares triangles : ℕ) (sq_toothpicks tri_toothpicks : ℕ) 
    (total_toothpicks : ℕ) (remove_toothpicks : ℕ) 
    (h1 : n = 40) 
    (h2 : squares = 10) 
    (h3 : triangles = 15) 
    (h4 : sq_toothpicks = 4) 
    (h5 : tri_toothpicks = 3) 
    (h6 : total_toothpicks = n) 
    (h7 : remove_toothpicks = 10) 
    (h8 : (squares * sq_toothpicks + triangles * tri_toothpicks) = total_toothpicks) :
  remove_toothpicks = 10 :=
by
  sorry

end NUMINAMATH_GPT_toothpick_removal_l2230_223005


namespace NUMINAMATH_GPT_cistern_filling_time_l2230_223041

/-
Given the following conditions:
- Pipe A fills the cistern in 10 hours.
- Pipe B fills the cistern in 12 hours.
- Exhaust pipe C drains the cistern in 15 hours.
- Exhaust pipe D drains the cistern in 20 hours.

Prove that if all four pipes are opened simultaneously, the cistern will be filled in 15 hours.
-/

theorem cistern_filling_time :
  let rate_A := 1 / 10
  let rate_B := 1 / 12
  let rate_C := -(1 / 15)
  let rate_D := -(1 / 20)
  let combined_rate := rate_A + rate_B + rate_C + rate_D
  let time_to_fill := 1 / combined_rate
  time_to_fill = 15 :=
by 
  sorry

end NUMINAMATH_GPT_cistern_filling_time_l2230_223041


namespace NUMINAMATH_GPT_part1_part2_l2230_223062

-- Define the conditions
def cost_price := 30
def initial_selling_price := 40
def initial_sales_volume := 600
def sales_decrease_per_yuan := 10

-- Define the profit calculation function
def profit (selling_price : ℕ) : ℕ :=
  let profit_per_unit := selling_price - cost_price
  let new_sales_volume := initial_sales_volume - sales_decrease_per_yuan * (selling_price - initial_selling_price)
  profit_per_unit * new_sales_volume

-- Statements to prove
theorem part1 :
  profit 50 = 10000 :=
by
  sorry

theorem part2 :
  let max_profit_price := 60
  let max_profit := 12000
  max_profit = (fun price => max (profit price) 0) 60 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2230_223062


namespace NUMINAMATH_GPT_elderly_people_pears_l2230_223016

theorem elderly_people_pears (x y : ℕ) :
  (y = x + 1) ∧ (2 * x = y + 2) ↔
  (x = y - 1) ∧ (2 * x = y + 2) := by
  sorry

end NUMINAMATH_GPT_elderly_people_pears_l2230_223016


namespace NUMINAMATH_GPT_number_of_children_l2230_223013

theorem number_of_children (crayons_per_child total_crayons : ℕ) (h1 : crayons_per_child = 12) (h2 : total_crayons = 216) : total_crayons / crayons_per_child = 18 :=
by
  have h3 : total_crayons / crayons_per_child = 216 / 12 := by rw [h1, h2]
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_number_of_children_l2230_223013


namespace NUMINAMATH_GPT_distinct_balls_boxes_l2230_223077

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end NUMINAMATH_GPT_distinct_balls_boxes_l2230_223077


namespace NUMINAMATH_GPT_find_number_l2230_223050

theorem find_number (n : ℕ) (h : n + 19 = 47) : n = 28 :=
by {
    sorry
}

end NUMINAMATH_GPT_find_number_l2230_223050


namespace NUMINAMATH_GPT_ben_mms_count_l2230_223076

theorem ben_mms_count (S M : ℕ) (hS : S = 50) (h_diff : S = M + 30) : M = 20 := by
  sorry

end NUMINAMATH_GPT_ben_mms_count_l2230_223076


namespace NUMINAMATH_GPT_find_b_l2230_223059

noncomputable def h (x : ℝ) : ℝ := x^2 + 9
noncomputable def j (x : ℝ) : ℝ := x^2 + 1

theorem find_b (b : ℝ) (hjb : h (j b) = 15) (b_pos : b > 0) : b = Real.sqrt (Real.sqrt 6 - 1) := by
  sorry

end NUMINAMATH_GPT_find_b_l2230_223059


namespace NUMINAMATH_GPT_email_sequence_correct_l2230_223092

theorem email_sequence_correct :
    ∀ (a b c d e f : Prop),
    (a → (e → (b → (c → (d → f))))) :=
by 
  sorry

end NUMINAMATH_GPT_email_sequence_correct_l2230_223092


namespace NUMINAMATH_GPT_method_one_cost_eq_300_method_two_cost_eq_300_method_one_more_cost_effective_l2230_223081

noncomputable def method_one_cost (x : ℕ) : ℕ := 120 + 10 * x

noncomputable def method_two_cost (x : ℕ) : ℕ := 15 * x

theorem method_one_cost_eq_300 (x : ℕ) : method_one_cost x = 300 ↔ x = 18 :=
by sorry

theorem method_two_cost_eq_300 (x : ℕ) : method_two_cost x = 300 ↔ x = 20 :=
by sorry

theorem method_one_more_cost_effective (x : ℕ) :
  x ≥ 40 → method_one_cost x < method_two_cost x :=
by sorry

end NUMINAMATH_GPT_method_one_cost_eq_300_method_two_cost_eq_300_method_one_more_cost_effective_l2230_223081


namespace NUMINAMATH_GPT_describe_graph_l2230_223084

theorem describe_graph : 
  ∀ (x y : ℝ), x^2 * (x + y + 1) = y^3 * (x + y + 1) ↔ (x^2 = y^3 ∨ y = -x - 1)
:= sorry

end NUMINAMATH_GPT_describe_graph_l2230_223084


namespace NUMINAMATH_GPT_expr_for_pos_x_min_value_l2230_223087

section
variable {f : ℝ → ℝ}
variable {a : ℝ}

def even_func (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def func_def (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, x ≤ 0 → f x = 4^(-x) - a * 2^(-x)

-- Assuming f is even and specified as in the problem for x ≤ 0
axiom ev_func : even_func f
axiom f_condition : 0 < a

theorem expr_for_pos_x (f : ℝ → ℝ) (a : ℝ) (h1 : even_func f) (h2 : func_def f a) : 
  ∀ x, 0 < x → f x = 4^x - a * 2^x :=
sorry -- this aims to prove the function's form for positive x.

theorem min_value (f : ℝ → ℝ) (a : ℝ) (h1 : even_func f) (h2 : func_def f a) :
  (0 < a ∧ a ≤ 2 → ∃ x, 0 < x ∧ f x = 1 - a) ∧
  (2 < a → ∃ x, 0 < x ∧ f x = -a^2 / 4) :=
sorry -- this aims to prove the minimum value on the interval (0, +∞).
end

end NUMINAMATH_GPT_expr_for_pos_x_min_value_l2230_223087


namespace NUMINAMATH_GPT_average_of_new_set_l2230_223074

theorem average_of_new_set (s : List ℝ) (h₁ : s.length = 10) (h₂ : (s.sum / 10) = 7) : 
  ((s.map (λ x => x * 12)).sum / 10) = 84 :=
by
  sorry

end NUMINAMATH_GPT_average_of_new_set_l2230_223074


namespace NUMINAMATH_GPT_solve_quadratic_equation_l2230_223040

theorem solve_quadratic_equation (x : ℝ) :
  (6 * x^2 - 3 * x - 1 = 2 * x - 2) ↔ (x = 1 / 3 ∨ x = 1 / 2) :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l2230_223040


namespace NUMINAMATH_GPT_point_outside_circle_l2230_223043

theorem point_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) : a^2 + b^2 > 1 :=
sorry

end NUMINAMATH_GPT_point_outside_circle_l2230_223043


namespace NUMINAMATH_GPT_base7_sum_correct_l2230_223046

theorem base7_sum_correct : 
  ∃ (A B C : ℕ), 
  A < 7 ∧ B < 7 ∧ C < 7 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (A = 2 ∨ A = 3 ∨ A = 5) ∧
  (A * 49 + B * 7 + C) + (B * 7 + C) = A * 49 + C * 7 + A ∧
  A + B + C = 16 :=
by
  sorry

end NUMINAMATH_GPT_base7_sum_correct_l2230_223046


namespace NUMINAMATH_GPT_man_older_than_son_by_46_l2230_223055

-- Given conditions about the ages
def sonAge : ℕ := 44

def manAge_in_two_years (M : ℕ) : Prop := M + 2 = 2 * (sonAge + 2)

-- The problem to verify
theorem man_older_than_son_by_46 (M : ℕ) (h : manAge_in_two_years M) : M - sonAge = 46 :=
by
  sorry

end NUMINAMATH_GPT_man_older_than_son_by_46_l2230_223055


namespace NUMINAMATH_GPT_corresponding_side_of_larger_triangle_l2230_223056

-- Conditions
variables (A1 A2 : ℕ) (s1 s2 : ℕ)
-- A1 is the area of the larger triangle
-- A2 is the area of the smaller triangle
-- s1 is a side of the smaller triangle = 4 feet
-- s2 is the corresponding side of the larger triangle

-- Given conditions as hypotheses
axiom diff_in_areas : A1 - A2 = 32
axiom ratio_of_areas : A1 = 9 * A2
axiom side_of_smaller_triangle : s1 = 4

-- Theorem to prove the corresponding side of the larger triangle
theorem corresponding_side_of_larger_triangle 
  (h1 : A1 - A2 = 32)
  (h2 : A1 = 9 * A2)
  (h3 : s1 = 4) : 
  s2 = 12 :=
sorry

end NUMINAMATH_GPT_corresponding_side_of_larger_triangle_l2230_223056


namespace NUMINAMATH_GPT_number_of_cubes_l2230_223065

theorem number_of_cubes (L W H V_cube : ℝ) (L_eq : L = 9) (W_eq : W = 12) (H_eq : H = 3) (V_cube_eq : V_cube = 3) :
  L * W * H / V_cube = 108 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cubes_l2230_223065


namespace NUMINAMATH_GPT_vertical_asymptote_l2230_223012

noncomputable def y (x : ℝ) : ℝ := (3 * x + 1) / (7 * x - 10)

theorem vertical_asymptote (x : ℝ) : (7 * x - 10 = 0) → (x = 10 / 7) :=
by
  intro h
  linarith [h]

#check vertical_asymptote

end NUMINAMATH_GPT_vertical_asymptote_l2230_223012


namespace NUMINAMATH_GPT_brick_length_l2230_223072

theorem brick_length (w h SA : ℝ) (h_w : w = 6) (h_h : h = 2) (h_SA : SA = 152) :
  ∃ l : ℝ, 2 * l * w + 2 * l * h + 2 * w * h = SA ∧ l = 8 := 
sorry

end NUMINAMATH_GPT_brick_length_l2230_223072


namespace NUMINAMATH_GPT_range_of_x_when_a_eq_1_p_and_q_range_of_a_when_not_p_sufficient_for_not_q_l2230_223069

-- Define the propositions
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := -x^2 + 5 * x - 6 ≥ 0

-- Question 1: Prove that for a = 1 and p ∧ q is true, the range of x is [2, 3)
theorem range_of_x_when_a_eq_1_p_and_q : 
  ∀ x : ℝ, p 1 x ∧ q x → 2 ≤ x ∧ x < 3 := 
by sorry

-- Question 2: Prove that if ¬p is a sufficient but not necessary condition for ¬q, 
-- then the range of a is (1, 2)
theorem range_of_a_when_not_p_sufficient_for_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬p a x → ¬q x) ∧ (∃ x : ℝ, ¬(¬p a x → ¬q x)) → 1 < a ∧ a < 2 := 
by sorry

end NUMINAMATH_GPT_range_of_x_when_a_eq_1_p_and_q_range_of_a_when_not_p_sufficient_for_not_q_l2230_223069


namespace NUMINAMATH_GPT_simplify_expression_l2230_223015

theorem simplify_expression (s : ℤ) : 120 * s - 32 * s = 88 * s := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2230_223015


namespace NUMINAMATH_GPT_length_of_shorter_side_l2230_223004

/-- 
A rectangular plot measuring L meters by 50 meters is to be enclosed by wire fencing. 
If the poles of the fence are kept 5 meters apart, 26 poles will be needed.
What is the length of the shorter side of the rectangular plot?
-/
theorem length_of_shorter_side
(L: ℝ) 
(h1: ∃ L: ℝ, L > 0) -- There's some positive length for the side L
(h2: ∀ distance: ℝ, distance = 5) -- Poles are kept 5 meters apart
(h3: ∀ poles: ℝ, poles = 26) -- 26 poles will be needed
(h4: 125 = 2 * (L + 50)) -- Use the perimeter calculated
: L = 12.5
:= sorry

end NUMINAMATH_GPT_length_of_shorter_side_l2230_223004
